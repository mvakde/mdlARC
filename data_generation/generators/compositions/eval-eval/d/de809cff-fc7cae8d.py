"""Occlusion-gated complementary stitches, oriented in each host's local frame.

Only NumPy and the standard library are used. Solving does not access generation
parameters, training pairs, labels, RNGs, files, or a learned color convention.
"""
from itertools import permutations

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
MATRICES = {
    "identity": ((1, 0), (0, 1)),
    "r90": ((0, -1), (1, 0)),
    "r180": ((-1, 0), (0, -1)),
    "r270": ((0, 1), (-1, 0)),
    "flip_lr": ((1, 0), (0, -1)),
    "flip_ud": ((-1, 0), (0, 1)),
    "transpose": ((0, 1), (1, 0)),
    "anti_transpose": ((0, -1), (-1, 0)),
}


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(a, color):
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    H, W = a.shape
    for rr, cc in np.argwhere(a == color):
        p = (int(rr), int(cc))
        if seen[p]:
            continue
        stack = [p]
        seen[p] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= q[0] < H and 0 <= q[1] < W and
                        not seen[q] and a[q] == color):
                    seen[q] = True
                    stack.append(q)
        result.append(cells)
    return result


def _line(cells):
    rows = sorted({p[0] for p in cells})
    cols = sorted({p[1] for p in cells})
    if len(rows) == 1 and cols == list(range(cols[0], cols[-1]+1)):
        return "H", rows[0], cols[0], cols[-1]+1
    if len(cols) == 1 and rows == list(range(rows[0], rows[-1]+1)):
        return "V", cols[0], rows[0], rows[-1]+1
    return None


def _bounds_from_guides(horizontal, vertical):
    _, hr, c0, c1 = horizontal
    _, vc, r0, r1 = vertical
    if hr == r0-1:
        hs = "T"
    elif hr == r1:
        hs = "B"
    else:
        return None
    if vc == c0-1:
        vs = "L"
    elif vc == c1:
        vs = "R"
    else:
        return None
    return (r0, c0, r1, c1), (hs, vs)


def _read_controls(a):
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1]))
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) == 7, "Exactly seven semantic color roles are visible"
    singletons = [int(c) for c, n in zip(colors, counts) if n == 1]
    assert len(singletons) == 1, "The stencil anchor is the unique singleton color"
    highlight = singletons[0]
    anchor_global = tuple(map(int, np.argwhere(a == highlight)[0]))
    parts = {int(c): _components(a, int(c)) for c in colors if c != bg}
    long_lines = []
    for color, components in parts.items():
        for comp in components:
            desc = _line(comp)
            if desc and 3 <= len(comp) <= 5:
                long_lines.append((color, desc, comp))
    matches = []
    for hc, hd, hcells in long_lines:
        if hd[0] != "H":
            continue
        for vc, vd, vcells in long_lines:
            if vd[0] != "V" or hc == vc:
                continue
            candidate = _bounds_from_guides(hd, vd)
            if candidate is None:
                continue
            box, sides = candidate
            r0, c0, r1, c1 = box
            if not (1 < r0 < r1 < a.shape[0]-1 and 1 < c0 < c1 < a.shape[1]-1):
                continue
            ar, ac = anchor_global
            if not (r0 <= ar < r1 and c0 <= ac < c1):
                continue
            payload = a[r0:r1, c0:c1]
            extra = set(map(int, np.unique(payload))) - {bg, highlight}
            if len(extra) != 1 or bg not in payload or highlight not in payload:
                continue
            ink = next(iter(extra))
            if ink in (hc, vc) or np.count_nonzero(a == ink) != np.count_nonzero(payload == ink):
                continue
            if np.count_nonzero(payload == ink) < 2:
                continue
            tabs = []
            valid = True
            for color, long in ((hc, hcells), (vc, vcells)):
                for comp in parts[color]:
                    if set(comp) == set(long):
                        continue
                    desc = _line(comp)
                    if len(comp) != 2 or desc is None:
                        valid = False
                        break
                    tabs.append((color, desc, tuple(comp)))
            if not valid or len(tabs) not in (4, 6, 8):
                continue
            fills = tuple(sorted(set(map(int, colors)) - {bg, highlight, ink, hc, vc}))
            if len(fills) != 2 or min(np.count_nonzero(a == f) for f in fills) < 20:
                continue
            # A background moat isolates the complete prototype and its guides.
            rr0 = r0 - ("T" in sides)
            rr1 = r1 + ("B" in sides)
            cc0 = c0 - ("L" in sides)
            cc1 = c1 + ("R" in sides)
            moat = np.concatenate((a[rr0-1, cc0-1:cc1+1], a[rr1, cc0-1:cc1+1],
                                   a[rr0:rr1, cc0-1], a[rr0:rr1, cc1]))
            if not np.all(moat == bg):
                continue
            scene_input = a.copy()
            scene_input[r0:r1, c0:c1] = bg
            for cells in (hcells, vcells):
                for p in cells:
                    scene_input[p] = bg
            for _, _, cells in tabs:
                for p in cells:
                    scene_input[p] = bg
            if set(map(int, np.unique(scene_input))) != {bg, *fills}:
                continue
            matches.append(dict(background=bg, highlight=highlight, ink=ink, fills=fills,
                                prototype_box=box, prototype=payload.copy(),
                                anchor=(ar-r0, ac-c0), guides=dict(zip(sides, (hc, vc))),
                                tabs=tabs, scene_input=scene_input))
    assert len(matches) == 1, "The input must specify exactly one anchored stencil"
    return matches[0]


def _tab_side(box, desc):
    r0, c0, r1, c1, _ = box
    axis, fixed, start, end = desc
    if axis == "H" and c0 <= start < end <= c1:
        if fixed == r0-1:
            return "T"
        if fixed == r1:
            return "B"
    if axis == "V" and r0 <= start < end <= r1:
        if fixed == c0-1:
            return "L"
        if fixed == c1:
            return "R"
    return None


def _associate_tabs(boxes, tabs):
    assignments = [[] for _ in boxes]
    for color, desc, cells in tabs:
        eligible = [(i, _tab_side(b, desc)) for i, b in enumerate(boxes)]
        eligible = [(i, s) for i, s in eligible if s is not None]
        if len(eligible) != 1:
            return None
        i, side = eligible[0]
        assignments[i].append((side, color, cells))
    frames = []
    for group in assignments:
        if (len(group) != 2 or len({c for _, c, _ in group}) != 2 or
                len({s in ("T", "B") for s, _, _ in group}) != 2):
            return None
        frames.append({s: c for s, c, _ in group})
    return frames


def _overlap(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def _geometry(boxes, shape):
    if not 2 <= len(boxes) <= 4:
        return False
    H, W = shape
    links = [set() for _ in boxes]
    for i, b in enumerate(boxes):
        r0, c0, r1, c1, color = b
        if not (1 <= r0 < r1 <= H-1 and 1 <= c0 < c1 <= W-1 and
                min(r1-r0, c1-c0) >= 6 and (r1-r0)*(c1-c0) >= 42):
            return False
        for j in range(i):
            overlap = _overlap(b, boxes[j])
            if overlap:
                if color == boxes[j][4] or overlap < 4:
                    return False
                links[i].add(j)
                links[j].add(i)
    reached = {0}
    for _ in boxes:
        reached |= set().union(*(links[i] for i in reached))
    return len(reached) == len(boxes)


def _paint(shape, bg, boxes, order):
    clean = np.full(shape, bg, dtype=np.int8)
    owner = np.full(shape, -1, dtype=np.int8)
    for i in order:
        r0, c0, r1, c1, color = boxes[i]
        clean[r0:r1, c0:c1] = color
        owner[r0:r1, c0:c1] = i
    return clean, owner


def _supported(clean, owner, boxes):
    for i, b in enumerate(boxes):
        r0, c0, r1, c1, color = b
        visible = owner[r0:r1, c0:c1] == i
        if visible.sum() < max(24, .40*visible.size):
            return False
        if min(visible[0].sum(), visible[-1].sum(), visible[:, 0].sum(), visible[:, -1].sum()) < 3:
            return False
        comps = [p for p in _components((owner == i).astype(np.int8), 1)]
        if len(comps) != 1:
            return False
    return True


def _smooth(a, bg, axis, radius):
    colors = np.unique(a)
    votes = []
    for color in colors:
        count = np.zeros(a.shape, dtype=np.int16)
        padded = np.pad(a, ((radius, radius), (radius, radius)), constant_values=bg)
        H, W = a.shape
        for delta in range(-radius, radius+1):
            r = radius + (delta if axis == 0 else 0)
            c = radius + (delta if axis == 1 else 0)
            count += padded[r:r+H, c:c+W] == color
        votes.append(count)
    votes = np.asarray(votes)
    result = colors[votes.argmax(axis=0)]
    tied_original = np.sum(votes * (a[None] == colors[:, None, None]), axis=0) == votes.max(axis=0)
    ambiguous = np.sum(votes == votes.max(axis=0), axis=0) > 1
    keep = tied_original | ambiguous
    result[keep] = a[keep]
    return result.astype(np.int8)


def _fit_scene(info):
    observed, bg, fills, tabs = (info[k] for k in ("scene_input", "background", "fills", "tabs"))
    proposals = set()
    smoothed = [observed]
    for radius in (1, 2):
        for axis in (0, 1):
            smoothed.append(_smooth(_smooth(observed, bg, axis, radius), bg, 1-axis, radius))
    for sample in smoothed:
        boxes = []
        for color in fills:
            for comp in _components(sample, color):
                if len(comp) < 18:
                    continue
                rows, cols = zip(*comp)
                boxes.append((min(rows), min(cols), max(rows)+1, max(cols)+1, color))
        boxes = tuple(sorted(boxes))
        if _geometry(boxes, observed.shape) and _associate_tabs(boxes, tabs) is not None:
            proposals.add(boxes)
    assert proposals, "No control-compatible rectangle reconstruction"
    candidates = {}
    for boxes in sorted(proposals):
        for order in permutations(range(len(boxes))):
            clean, owner = _paint(observed.shape, bg, boxes, order)
            loss = int(np.count_nonzero(clean != observed))
            if loss > max(8, int(.13*observed.size)):
                continue
            if not _supported(clean, owner, boxes):
                continue
            if any(np.mean(observed[owner == i] == boxes[i][4]) < .80 for i in range(len(boxes))):
                continue
            if any(clean[p] != bg for _, _, cells in tabs for p in cells):
                continue
            key = clean.tobytes()
            if key not in candidates or loss < candidates[key][0]:
                candidates[key] = (loss, clean, owner, boxes, order)
    assert candidates, "Too little intact evidence"
    ranked = sorted(candidates.values(), key=lambda x: (x[0], x[1].tobytes()))
    loss, clean, owner, boxes, order = ranked[0]
    assert all(other[0] > loss+1 for other in ranked[1:]), "Competing visible scenes"
    # Also challenge each unmarked straight boundary by one cell.
    for i in range(len(boxes)):
        for side in range(4):
            for delta in (-1, 1):
                trial = list(boxes)
                b = list(trial[i])
                b[side] += delta
                trial[i] = tuple(b)
                if not _geometry(trial, observed.shape) or _associate_tabs(trial, tabs) is None:
                    continue
                alternative, own = _paint(observed.shape, bg, trial, order)
                if np.array_equal(alternative, clean) or not _supported(alternative, own, trial):
                    continue
                assert np.count_nonzero(alternative != observed) > loss+1, "Ambiguous damaged boundary"
    info.update(clean=clean, owner=owner, boxes=boxes, frames=_associate_tabs(boxes, tabs))
    return info


def _matrix(guides, frame):
    result = np.zeros((2, 2), dtype=int)
    for side, color in guides.items():
        target = next(s for s, c in frame.items() if c == color)
        result += np.outer(NORMAL[target], NORMAL[side])
    assert np.array_equal(result.T @ result, np.eye(2, dtype=int))
    return result


def _name(matrix):
    return next(k for k, m in MATRICES.items() if np.array_equal(matrix, m))


def _analyze(a):
    info = _fit_scene(_read_controls(_grid(a)))
    info["holes"] = [tuple(map(int, p)) for p in np.argwhere(
        (info["scene_input"] == info["background"]) & (info["owner"] >= 0))]
    assert info["holes"], "A damaged laminate must have at least one true erasure"
    relative = np.argwhere(info["prototype"] == info["ink"]) - np.asarray(info["anchor"])
    footprints = {tuple(sorted(map(tuple, relative @ np.asarray(m).T))) for m in MATRICES.values()}
    assert len(footprints) == 8, "The anchored teaching stencil must distinguish all eight symmetries"
    info["offsets"] = [relative @ _matrix(info["guides"], f).T for f in info["frames"]]
    info["transforms"] = [_name(_matrix(info["guides"], f)) for f in info["frames"]]
    return info


def _render(info, *, ignore_ownership=False, offsets=None, holes=None):
    output = info["clean"].copy()
    H, W = output.shape
    fills = info["fills"]
    holes = info["holes"] if holes is None else holes
    offsets = info["offsets"] if offsets is None else offsets
    for r, c in holes:
        host = int(info["owner"][r, c])
        other = next(f for f in fills if f != info["clean"][r, c])
        for dr, dc in offsets[host]:
            q = (r+int(dr), c+int(dc))
            if (0 <= q[0] < H and 0 <= q[1] < W and
                    (ignore_ownership or info["owner"][q] == host)):
                output[q] = other
    for p in holes:
        output[p] = info["highlight"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read the stencil, reconstruct the visible hosts, and stitch their holes."""
    return _render(_analyze(input_grid))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _scene(rng, shape, count):
    H, W = shape
    boxes = []
    for _ in range(160):
        max_side = {2: 20, 3: 14, 4: 11}[count]
        h = int(rng.integers(7, max(8, min(H-2, max_side)+1)))
        w = int(rng.integers(7, max(8, min(W-2, max_side)+1)))
        r = int(rng.integers(1, H-h))
        c = int(rng.integers(1, W-w))
        box = (r, c, r+h, c+w, 1+len(boxes)%2)
        if boxes:
            overlap = _overlap(box, boxes[-1])
            if not 4 <= overlap <= .42*min(h*w, (boxes[-1][2]-boxes[-1][0])*(boxes[-1][3]-boxes[-1][1])):
                continue
            if any(_overlap(box, b) for b in boxes[:-1]):
                continue
        boxes.append(box)
        if len(boxes) == count:
            boxes = tuple(sorted(boxes))
            order = tuple(map(int, rng.permutation(count)))
            clean, owner = _paint(shape, 0, boxes, order)
            if _supported(clean, owner, boxes):
                return clean, owner, boxes
            return None
    return None


def _side_cells(box, side, start, length=2):
    r0, c0, r1, c1 = box[:4]
    if side == "T":
        return [(r0-1, c) for c in range(start, start+length)]
    if side == "B":
        return [(r1, c) for c in range(start, start+length)]
    if side == "L":
        return [(r, c0-1) for r in range(start, start+length)]
    return [(r, c1) for r in range(start, start+length)]


def _make_tabs(rng, clean, boxes):
    tabs = []
    frames = []
    occupied = set()
    for i, box in enumerate(boxes):
        choices = {}
        for side in SIDES:
            start, end = (box[1], box[3]) if side in ("T", "B") else (box[0], box[2])
            pools = []
            for p in range(start, end-1):
                cells = _side_cells(box, side, p)
                if any(not (1 <= r < clean.shape[0]-1 and 1 <= c < clean.shape[1]-1) for r, c in cells):
                    continue
                if any(clean[q] != 0 or q in occupied for q in cells):
                    continue
                desc = _line(cells)
                if any(_tab_side(b, desc) is not None for j, b in enumerate(boxes) if j != i):
                    continue
                if any(abs(q[0]-p0[0])+abs(q[1]-p0[1]) <= 1 for q in cells for p0 in occupied):
                    continue
                pools.append(cells)
            choices[side] = pools
        pairs = [(h, v) for h in ("T", "B") for v in ("L", "R") if choices[h] and choices[v]]
        if not pairs:
            return None
        hside, vside = _pick(rng, pairs)
        colors = [3, 4] if rng.integers(2) else [4, 3]
        frame = {}
        for side, color in zip((hside, vside), colors):
            cells = _pick(rng, choices[side])
            occupied.update(cells)
            tabs.append((color, _line(cells), tuple(cells)))
            frame[side] = color
        frames.append(frame)
    return tabs, frames


def _stencil(rng):
    h, w = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    anchor = (int(rng.integers(h)), int(rng.integers(w)))
    available = [(r, c) for r in range(h) for c in range(w) if (r, c) != anchor]
    for _ in range(80):
        mask = np.zeros((h, w), dtype=np.int8)
        count = int(rng.integers(2, min(9, len(available)-1)+1))
        positions = [available[int(i)] for i in rng.choice(len(available), count, replace=False)]
        # Occasionally grow a connected stroke from a random point.
        if rng.random() < .45:
            positions = [positions[0]]
            for _ in range(count*5):
                r, c = _pick(rng, positions)
                dr, dc = _pick(rng, ((-1, 0), (1, 0), (0, -1), (0, 1)))
                p = (r+dr, c+dc)
                if p in available and p not in positions:
                    positions.append(p)
                if len(positions) == count:
                    break
        for p in positions:
            mask[p] = 5
        mask[anchor] = 6
        offsets = np.argwhere(mask == 5)-np.asarray(anchor)
        patterns = {tuple(sorted(map(tuple, offsets @ np.asarray(m).T))) for m in MATRICES.values()}
        if len(patterns) == 8 and np.count_nonzero(mask == 5) >= 2:
            return mask, anchor
    # An asymmetric two-point footprint works in every sampled extent/anchor.
    for points in permutations(available, 2):
        offsets = np.asarray(points)-np.asarray(anchor)
        patterns = {tuple(sorted(map(tuple, offsets @ np.asarray(m).T))) for m in MATRICES.values()}
        if len(patterns) == 8:
            mask = np.zeros((h, w), dtype=np.int8)
            for p in points:
                mask[p] = 5
            mask[anchor] = 6
            return mask, anchor
    raise AssertionError("No asymmetric stencil")


def _place_prototype(rng, clean, tabs, mask, corner):
    h, w = mask.shape
    guides_sides = (SIDES[corner], SIDES[(corner+1)%4])
    occupied = {p for _, _, cells in tabs for p in cells}
    positions = []
    for r0 in range(2, clean.shape[0]-h-1):
        for c0 in range(2, clean.shape[1]-w-1):
            box = (r0, c0, r0+h, c0+w)
            rr0, rr1 = r0-("T" in guides_sides), r0+h+("B" in guides_sides)
            cc0, cc1 = c0-("L" in guides_sides), c0+w+("R" in guides_sides)
            if np.any(clean[rr0-1:rr1+1, cc0-1:cc1+1] != 0):
                continue
            if any(rr0-1 <= p[0] <= rr1 and cc0-1 <= p[1] <= cc1 for p in occupied):
                continue
            positions.append(box)
    if not positions:
        return None
    return _pick(rng, positions)


def _new_pair(rng, palette, corner, teach=False, shape=None, fallback=False, requested_count=None):
    count = requested_count or int(rng.choice([2, 3, 4], p=[.40, .40, .20]))
    low = 26 if count == 4 else 21
    shape = shape or (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
    if fallback:
        H, W = shape
        boxes = tuple(sorted(((9, 2, H-4, W//2+3, 1), (H//2, W//2-2, H-2, W-2, 2))))
        clean, owner = _paint(shape, 0, boxes, (0, 1))
    else:
        sampled = _scene(rng, shape, count)
        if sampled is None:
            return None
        clean, owner, boxes = sampled
    controls = _make_tabs(rng, clean, boxes)
    if controls is None:
        return None
    tabs, frames = controls
    mask, anchor = _stencil(rng)
    guide_sides = (SIDES[corner], SIDES[(corner+1)%4])
    if teach:
        # Choose a quarter-turn and arrange this host's identities accordingly.
        possible = []
        for t in ("r90", "r270"):
            m = np.asarray(MATRICES[t])
            moved = [next(s for s in SIDES if tuple(m @ NORMAL[g]) == NORMAL[s]) for g in guide_sides]
            if set(moved) == set(frames[0]):
                possible.append((t, moved))
        if not possible:
            return None
        _, moved = _pick(rng, possible)
        guides = dict(zip(guide_sides, (3, 4)))
        frames[0] = dict(zip(moved, (3, 4)))
        tabs = [(frames[0][_tab_side(boxes[0], desc)] if _tab_side(boxes[0], desc) else color, desc, cells)
                for color, desc, cells in tabs]
    else:
        guides = dict(zip(guide_sides, (3, 4) if rng.integers(2) else (4, 3)))
    proto_box = _place_prototype(rng, clean, tabs, mask, corner)
    if proto_box is None:
        return None
    offsets = [(np.argwhere(mask == 5)-np.asarray(anchor)) @ _matrix(guides, f).T for f in frames]
    H, W = shape
    cells = [tuple(map(int, p)) for p in np.argwhere(owner >= 0)]
    holes = []
    full = [[] for _ in boxes]
    seam_clipped = []
    for p in cells:
        host = int(owner[p])
        destinations = [(p[0]+int(dr), p[1]+int(dc)) for dr, dc in offsets[host]]
        inside = [q for q in destinations if 0 <= q[0] < H and 0 <= q[1] < W and owner[q] == host]
        if len(inside) == len(destinations):
            full[host].append(p)
        if inside and any(0 <= q[0] < H and 0 <= q[1] < W and owner[q] >= 0 and owner[q] != host for q in destinations):
            seam_clipped.append(p)
    if teach:
        # First host plus at least one hole in the other fill, both fully visible.
        if not full[0]:
            return None
        first = _pick(rng, full[0])
        holes.append(first)
        others = [p for i, pool in enumerate(full) if boxes[i][4] != boxes[0][4] for p in pool]
        if not others or not seam_clipped:
            return None
        holes.append(_pick(rng, others))
        adjacent = [p for p in cells if owner[p] == owner[first] and p != first and
                    max(abs(p[0]-first[0]), abs(p[1]-first[1])) == 1]
        if not adjacent:
            return None
        holes.append(_pick(rng, adjacent))
        # Make center precedence observable even for a long/off-center brush.
        touched = [(first[0]+int(dr), first[1]+int(dc)) for dr, dc in offsets[0]]
        if not any(p in holes for p in touched):
            holes.append(_pick(rng, touched))
        # A crossing by itself may be invisible: opposite ink already equals
        # the other host's base fill. Force a conflict with that host's stitch.
        rng.shuffle(seam_clipped)
        collision = None
        for p in seam_clipped:
            host = int(owner[p])
            for dr, dc in offsets[host]:
                dest = (p[0]+int(dr), p[1]+int(dc))
                if not (0 <= dest[0] < H and 0 <= dest[1] < W) or dest in holes:
                    continue
                other_host = int(owner[dest])
                if other_host < 0 or other_host == host:
                    continue
                if boxes[other_host][4] == boxes[host][4]:
                    continue
                for er, ec in offsets[other_host]:
                    q = (dest[0]-int(er), dest[1]-int(ec))
                    if (0 <= q[0] < H and 0 <= q[1] < W and owner[q] == other_host
                            and q != dest and p != dest and dest not in (p, q)):
                        collision = (p, q)
                        break
                if collision:
                    break
            if collision:
                break
        if collision is None:
            return None
        for p in collision:
            if p not in holes:
                holes.append(p)
        # Also teach material-edge clipping with a visible background stop.
        edge_candidates = []
        for p in cells:
            host = int(owner[p])
            points = [(p[0]+int(dr), p[1]+int(dc)) for dr, dc in offsets[host]]
            if any(0 <= r < H and 0 <= c < W and owner[r, c] == -1 for r, c in points):
                if any(0 <= r < H and 0 <= c < W and owner[r, c] == host for r, c in points):
                    edge_candidates.append(p)
        if not edge_candidates:
            return None
        p = _pick(rng, edge_candidates)
        if p not in holes:
            holes.append(p)
    target = int(rng.integers(max(2, len(holes)), max(len(holes)+1, 5, min(15, len(cells)//25)+1)))
    rng.shuffle(cells)
    for p in cells:
        if len(holes) >= target:
            break
        if p in holes:
            continue
        # Sparse clusters cannot erase a complete visible boundary or thin arm.
        if sum(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for q in holes) >= 3:
            continue
        holes.append(p)
    observed = clean.copy()
    for p in holes:
        observed[p] = 0
    seam = [p for p in cells if p not in holes and any(
        0 <= q[0] < H and 0 <= q[1] < W and clean[q] not in (0, clean[p])
        for q in ((p[0]-1, p[1]), (p[0]+1, p[1]), (p[0], p[1]-1), (p[0], p[1]+1)))]
    rng.shuffle(seam)
    ordinary = [p for p in cells if p not in holes and p not in seam]
    rng.shuffle(ordinary)
    wrong_count = int(rng.integers(1 if teach else 0, 5))
    errors = []
    pool = seam+ordinary if teach or rng.random() < .65 else ordinary+seam
    for p in pool:
        if len(errors) == wrong_count:
            break
        if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1 for q in errors):
            continue
        observed[p] = 3-int(clean[p])
        errors.append(p)
    reserved = {p for _, _, group in tabs for p in group}
    r0, c0, r1, c1 = proto_box
    reserved |= {(r, c) for r in range(r0-2, r1+2) for c in range(c0-2, c1+2)}
    exterior = []
    for rr, cc in np.argwhere(clean == 0):
        p = (int(rr), int(cc))
        if not (1 <= p[0] < H-1 and 1 <= p[1] < W-1) or p in reserved:
            continue
        region = clean[max(0, p[0]-1):p[0]+2, max(0, p[1]-1):p[1]+2]
        if np.all(region == 0):
            exterior.append(p)
    if not exterior:
        return None
    rng.shuffle(exterior)
    for p in exterior[:int(rng.integers(1, 5))]:
        observed[p] = int(rng.integers(1, 3))
    observed[r0:r1, c0:c1] = mask
    for side, color in guides.items():
        start = c0 if side in ("T", "B") else r0
        length = c1-c0 if side in ("T", "B") else r1-r0
        for p in _side_cells(proto_box, side, start, length):
            observed[p] = color
    for color, _, group in tabs:
        for p in group:
            observed[p] = color
    actual = np.asarray(palette, dtype=np.int8)[observed]
    try:
        info = _analyze(actual)
        if not np.array_equal(info["clean"], np.asarray(palette, dtype=np.int8)[clean]):
            return None
        if set(info["holes"]) != set(holes):
            return None
        if teach and not _teaching_evidence(info):
            return None
        output = _render(info)
        if teach:
            alternate = _swap_host(actual, info, 0)
            alt = _analyze(alternate)
            if np.array_equal(output, _render(alt)):
                return None
        return {"input": actual, "output": output}
    except (AssertionError, StopIteration):
        return None


def _swap_host(a, info, host):
    result = a.copy()
    colors = list(info["guides"].values())
    for color, desc, cells in info["tabs"]:
        if _tab_side(info["boxes"][host], desc) is not None:
            for p in cells:
                result[p] = next(c for c in colors if c != color)
    return result


def _teaching_evidence(info):
    own, clean, bg = info["owner"], info["clean"], info["background"]
    holes = info["holes"]
    H, W = own.shape
    full_colors = set()
    clipped = False
    background_stop = False
    center_hit = False
    demands = {}
    for p in holes:
        host = int(own[p])
        points = [(p[0]+int(dr), p[1]+int(dc)) for dr, dc in info["offsets"][host]]
        if all(0 <= r < H and 0 <= c < W and own[r, c] == host for r, c in points):
            full_colors.add(int(clean[p]))
        clipped |= any(0 <= r < H and 0 <= c < W and own[r, c] >= 0 and own[r, c] != host for r, c in points)
        background_stop |= any(0 <= r < H and 0 <= c < W and own[r, c] == -1 for r, c in points)
        center_hit |= any(q in holes and q != p for q in points)
        for q in points:
            if 0 <= q[0] < H and 0 <= q[1] < W and q not in holes and own[q] >= 0:
                demands.setdefault(q, set()).add(int(clean[p]))
    conflict = any(len(colors) == 2 for colors in demands.values())
    adjacent = any(own[p] == own[q] and max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1
                   for i, p in enumerate(holes) for q in holes[:i])
    errors = info["scene_input"] != clean
    seam_error = any(any(0 <= q[0] < H and 0 <= q[1] < W and clean[q] not in (bg, clean[r, c])
                         for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)))
                     for r, c in np.argwhere(errors & (clean != bg) & (info["scene_input"] != bg)))
    speckle = np.any(errors & (clean == bg))
    return (full_colors == set(info["fills"]) and adjacent and clipped and background_stop
            and center_hit and conflict and seam_error and speckle)


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .25)
    palette = tuple(map(int, rng.permutation(10)[:7]))
    held_corner = int(rng.integers(4))
    train_corners = [i for i in range(4) if i != held_corner]
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(nt+ne):
        if index == 1:
            first = puzzle["train"][0]
            a = _swap_host(first["input"], _analyze(first["input"]), 0)
            if not stable:
                # A visible role permutation accompanies the normalized contrast.
                order = list(map(int, rng.permutation(10)))
                shift = int(rng.integers(1, 10))
                mapping = np.empty(10, dtype=np.int8)
                for j, c in enumerate(order):
                    mapping[c] = order[(j+shift)%10]
                a = mapping[a]
            pair = {"input": a, "output": solve(a)}
        else:
            corner = _pick(rng, train_corners) if index < nt else held_corner
            pair_palette = palette if stable else tuple(map(int, rng.permutation(10)[:7]))
            pair = None
            requested_count = int(rng.choice([2, 3, 4], p=[.40, .40, .20])) if index != 0 else None
            for attempt in range(240):
                candidate = _new_pair(rng, pair_palette, corner, teach=index == 0,
                                      fallback=attempt >= 200, requested_count=requested_count)
                if candidate is None or _key(candidate["input"]) in seen:
                    continue
                if index >= nt and candidate["input"].shape == puzzle["train"][0]["input"].shape:
                    continue
                pair = candidate
                break
            if pair is None:
                raise AssertionError("Bounded scene construction exhausted")
        seen.add(_key(pair["input"]))
        puzzle["train" if index < nt else "test"].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    analyses = {"train": [], "test": []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(out, _render(info)), "Input-derived stitches disagree with label"
            assert np.all(out[info["owner"] == -1] == info["background"]), "Ink escaped its visible host"
            assert set(map(tuple, np.argwhere(out == info["highlight"]))) == set(info["holes"])
            assert not set(map(int, np.unique(out))) & {info["ink"], *info["guides"].values()}
            # Every true erasure changes the answer relative to ordinary repair.
            assert not np.array_equal(out, info["clean"])
            analyses[split].append(info)
    a, b = analyses["train"][:2]
    assert _teaching_evidence(a), "Missing a material/overlap/noise teaching witness"
    assert a["boxes"] and tuple(x[:4] for x in a["boxes"]) == tuple(x[:4] for x in b["boxes"])
    assert a["holes"] == b["holes"] and a["anchor"] == b["anchor"]
    assert np.array_equal(a["prototype"] == a["ink"], b["prototype"] == b["ink"])
    assert set(a["guides"]) == set(b["guides"])
    assert all(set(x) == set(y) for x, y in zip(a["frames"], b["frames"]))
    assert a["transforms"][0] in ("r90", "r270")
    assert np.linalg.det(_matrix(b["guides"], b["frames"][0])) == -1
    assert a["transforms"][1:] == b["transforms"][1:]
    role_map = {a["background"]: b["background"], a["ink"]: b["ink"],
                a["highlight"]: b["highlight"]}
    role_map.update({a["guides"][s]: b["guides"][s] for s in a["guides"]})
    role_map.update({x[4]: y[4] for x, y in zip(a["boxes"], b["boxes"])})
    color_map = np.arange(10, dtype=np.int8)
    for color, replacement in role_map.items():
        color_map[color] = replacement
    assert np.array_equal(color_map[a["scene_input"]], b["scene_input"])
    assert np.array_equal(color_map[a["prototype"]], b["prototype"])
    assert not np.array_equal(color_map[_render(a)], _render(b)), "The correspondence contrast must change ink"
    # A direction-blind mask must visibly fail on the demonstrations.
    relative = np.argwhere(a["prototype"] == a["ink"])-np.asarray(a["anchor"])
    assert not np.array_equal(_render(a), _render(a, offsets=[relative]*len(a["boxes"])))
    train_corners = {frozenset(info["guides"]) for info in analyses["train"]}
    assert all(frozenset(info["guides"]) not in train_corners for info in analyses["test"])
    assert len({tuple(x[:4] for x in info["boxes"]) for values in analyses.values() for info in values}) >= 2


def self_test():
    """Bounded author-owned semantics and diversity checks on 42 seeds."""
    measurements = []
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p["input"]) for s in puzzle.values() for p in s))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                measurements.append((split, info))
                expected = info["clean"].copy()
                # Destination-based union is order independent and checks the
                # simultaneous renderer without using its painting loop.
                for host in range(len(info["boxes"])):
                    dest = set()
                    for r, c in info["holes"]:
                        if info["owner"][r, c] == host:
                            dest.update((r+int(dr), c+int(dc)) for dr, dc in info["offsets"][host])
                    color = next(f for f in info["fills"] if f != info["boxes"][host][4])
                    for r, c in dest:
                        if 0 <= r < expected.shape[0] and 0 <= c < expected.shape[1] and info["owner"][r, c] == host:
                            expected[r, c] = color
                for p in info["holes"]:
                    expected[p] = info["highlight"]
                assert np.array_equal(expected, pair["output"])
                assert np.array_equal(_render(info, holes=list(reversed(info["holes"]))), pair["output"])
                assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
        if seed < 2:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], repeat[s]) for k in p)
    assert len(signatures) == 42
    infos = [i for _, i in measurements]
    assert {len(i["boxes"]) for i in infos} == {2, 3, 4}
    assert {t for i in infos for t in i["transforms"]} == set(MATRICES)
    assert len({i["clean"].shape for i in infos}) >= 45
    assert len({tuple(x[:4] for x in i["boxes"]) for i in infos}) >= 130
    assert len({(i["prototype"].shape, i["anchor"], (i["prototype"] == i["ink"]).tobytes()) for i in infos}) >= 120
    assert {len(i["holes"]) for i in infos} >= {2, 3, 4, 5, 6, 7}
    assert len({i["background"] for i in infos}) == 10
    assert len({i["highlight"] for i in infos}) == 10
    for split in ("train", "test"):
        subset = [i for s, i in measurements if s == split]
        assert {frozenset(i["guides"]) for i in subset} == {frozenset((SIDES[k], SIDES[(k+1)%4])) for k in range(4)}
        assert {t for i in subset for t in i["transforms"]} == set(MATRICES)
    # The supplied verifier exercises every legal combination. Keep this
    # additional author-owned test bounded to count extremes.
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(800+17*a+b, a, b))
    for a, b in ((1, 1), (11, 1), (2, 0), (10, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    original = generate_puzzle(0)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect label accepted")
