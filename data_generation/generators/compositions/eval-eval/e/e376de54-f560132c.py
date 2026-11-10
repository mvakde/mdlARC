"""Fractured calibration plates. Standard library + NumPy, no evidence reads.

The sampler paints an inscription on a constructed square and breaks it apart.
The solver instead segments the visible input, exhaustively packs the footprints,
and recovers the inscription. No sampled parameters are solver inputs.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=False)


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in _N4:
                q = r + dr, c + dc
                if q in cells:
                    cells.remove(q)
                    stack.append(q)
        result.append(np.array(sorted(component), dtype=int))
    return result


def _crop_mask(mask):
    points = np.argwhere(mask)
    low, high = points.min(axis=0), points.max(axis=0) + 1
    return mask[low[0]:high[0], low[1]:high[1]].copy(), tuple(map(int, low))


def _bits(mask, side, offset=(0, 0)):
    return sum(1 << ((int(r) + offset[0]) * side + int(c) + offset[1])
               for r, c in np.argwhere(mask))


def _unbits(bits, side):
    return np.array([bool((bits >> i) & 1) for i in range(side * side)]).reshape(side, side)


def _pack(shapes, inks, side, limit=2, allowed_rotations=(0,1,2,3)):
    """All corner placements; deduplicate footprint AND inscription states.

    A corner contact fixes a rotated bounding box's translation. Each candidate
    must occupy exactly its assigned corner. Thus this search omits no allowed
    translation, and it never assumes the sampled partition is the answer.
    """
    if len(shapes) != 4 or sum(int(s.sum()) for s in shapes) != side * side:
        return []
    corners = (0, side - 1, side * (side - 1), side * side - 1)
    full = (1 << (side * side)) - 1

    def placed(shape, ink, corner, rotation):
        m, b = np.rot90(shape, rotation), np.rot90(ink, rotation)
        h, w = m.shape
        if h > side or w > side:
            return None
        offset = (0 if corner < 2 else side - h, 0 if corner % 2 == 0 else side - w)
        mask, marks = _bits(m, side, offset), _bits(b, side, offset)
        hits = tuple(bool(mask & (1 << q)) for q in corners)
        if sum(hits) != 1 or not hits[corner]:
            return None
        return mask, marks, rotation, offset

    anchor = placed(shapes[0], inks[0], 0, 0)
    if anchor is None:
        return []
    table = {}
    for piece in range(1, 4):
        for corner in range(1, 4):
            choices = {}
            for rotation in allowed_rotations:
                item = placed(shapes[piece], inks[piece], corner, rotation)
                if item is not None:
                    choices.setdefault(item[:2], item)
            table[piece, corner] = list(choices.values())
    states, solutions = set(), []
    masks, pieces, rotations, offsets = [anchor[0]], [0], [0], [anchor[3]]

    def search(corner, remaining, occupied, marks):
        if len(solutions) >= limit:
            return
        if corner == 4:
            if occupied == full:
                state = tuple(masks), marks
                if state not in states:
                    states.add(state)
                    owner = np.full((side, side), -1, dtype=np.int8)
                    for i, mask in enumerate(masks):
                        owner[_unbits(mask, side)] = i
                    solutions.append({"owner": owner, "ink": _unbits(marks, side),
                                      "pieces": tuple(pieces), "rotations": tuple(rotations),
                                      "offsets": tuple(offsets)})
            return
        for piece in remaining:
            for mask, ink, rotation, offset in table[piece, corner]:
                if mask & occupied:
                    continue
                masks.append(mask); pieces.append(piece)
                rotations.append(rotation); offsets.append(offset)
                search(corner + 1, tuple(p for p in remaining if p != piece), occupied | mask, marks | ink)
                masks.pop(); pieces.pop(); rotations.pop(); offsets.pop()
                if len(solutions) >= limit:
                    return

    search(1, (1, 2, 3), anchor[0], anchor[1])
    return solutions


def _strokes(ink):
    """Recover axis, ordinal order, and the single common anchor endpoint."""
    candidates = []
    for transpose in (False, True):
        view = ink.T if transpose else ink
        positions = np.flatnonzero(view.any(axis=1)).tolist()
        if len(positions) not in (3, 5, 7) or min(np.diff(positions), default=0) < 2:
            continue
        runs = [np.flatnonzero(view[p]) for p in positions]
        if any(len(run) == 0 or np.any(np.diff(run) != 1) for run in runs):
            continue
        starts, ends = [int(x[0]) for x in runs], [int(x[-1]) for x in runs]
        left, right = len(set(starts)) == 1, len(set(ends)) == 1
        if left == right:
            continue
        lengths = tuple(map(len, runs))
        reference = lengths[len(runs) // 2]
        if reference < 2:
            continue
        start, direction = (starts[0], 1) if left else (ends[0], -1)
        terminal = start + direction * (reference - 1)
        if not 0 <= terminal < view.shape[1]:
            continue
        target = np.zeros_like(view)
        for p in positions:
            for t in range(reference):
                target[p, start + direction * t] = True
        candidates.append({"transpose": transpose, "positions": tuple(positions),
                           "lengths": lengths, "reference": reference,
                           "start": start, "direction": direction,
                           "inset": start if left else view.shape[1] - 1 - start,
                           "target": target.T if transpose else target})
    _require(len(candidates) == 1, "inscription must have one parallel-stroke interpretation")
    return candidates[0]


def _parse(input_grid):
    grid = _grid(input_grid)
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == bg) > len(border) / 2, "background must dominate the border")
    components = _components(grid != bg)
    _require(len(components) == 4, "four separated connected pieces required")
    for a, b in itertools.combinations(components, 2):
        _require(int(np.abs(a[:, None] - b[None, :]).max(axis=2).min()) >= 2, "pieces need an empty-cell halo")
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    singles = colors[counts == 1]
    repeated = colors[counts > 1]
    _require(len(singles) == 4 and len(repeated) == 2, "four key colors, substrate and ink required")
    repeated_counts = [int(counts[np.flatnonzero(colors == c)[0]]) for c in repeated]
    _require(repeated_counts[0] != repeated_counts[1], "substrate must exceed ink in area")
    substrate = int(repeated[int(np.argmax(repeated_counts))])
    ink_color = int(repeated[int(np.argmin(repeated_counts))])
    points = np.argwhere(np.isin(grid, singles))
    r, c = map(int, points.min(axis=0))
    _require(set(map(tuple, points.tolist())) == {(r, c), (r, c+1), (r+1, c), (r+1, c+1)}, "key must be a 2 by 2 block")
    anchor_ids = [i for i, comp in enumerate(components) if (r, c) in set(map(tuple, comp.tolist()))]
    _require(len(anchor_ids) == 1, "key must be on one piece")
    index = anchor_ids[0]
    components = [components[index]] + [comp for i, comp in enumerate(components) if i != index]
    anchor_points = set(map(tuple, components[0].tolist()))
    _require(all(tuple(p) in anchor_points for p in points), "key cannot bridge pieces")
    key = grid[r:r+2, c:c+2].copy()
    area = sum(len(comp) for comp in components)
    side = math.isqrt(area)
    _require(side * side == area and 8 <= side <= 14, "piece areas must make a supported square")
    shapes, inks, boxes = [], [], []
    for comp in components:
        low, high = comp.min(axis=0), comp.max(axis=0) + 1
        m = np.zeros(tuple(high-low), dtype=bool)
        local = comp - low
        m[local[:, 0], local[:, 1]] = True
        b = grid[low[0]:high[0], low[1]:high[1]] == ink_color
        shapes.append(m); inks.append(b); boxes.append(tuple(map(int, low)))
    return {"grid": grid, "background": bg, "substrate": substrate, "ink_color": ink_color,
            "key": key, "key_position": (r, c), "side": side,
            "components": components, "shapes": shapes, "inks": inks, "boxes": boxes}


def _analyze(input_grid):
    a = _parse(input_grid)
    solutions = _pack(a["shapes"], a["inks"], a["side"])
    _require(len(solutions) == 1, "unique corner territories and reconstructed ink required")
    a["assembly"] = solutions[0]
    a["strokes"] = _strokes(solutions[0]["ink"])
    return a


def _answer(a, target=None):
    if target is None:
        target = a["strokes"]["target"]
    out = np.full((a["side"], a["side"]), a["background"], dtype=np.int8)
    out[target] = a["key"].ravel()[a["assembly"]["owner"][target]]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the calibrated, territory-colored inscription from input only."""
    return _answer(_analyze(input_grid))


def _partition(rng, side, style):
    """Connected corner growth or wandering seams with local seam transfers."""
    n = side
    if style == 0:
        owner = np.empty((n, n), dtype=np.int8)
        y = int(rng.integers(2, n-2))
        top, bottom = int(rng.integers(2, n-1)), int(rng.integers(1, n-1))
        for r in range(n):
            if r < y:
                top = int(np.clip(top + rng.integers(-1, 2), 2 if r < 2 else 1, n-1))
                owner[r, :top], owner[r, top:] = 0, 1
            else:
                bottom = int(np.clip(bottom + rng.integers(-1, 2), 1, n-1))
                owner[r, :bottom], owner[r, bottom:] = 2, 3
        protected = {(0,0), (0,n-1), (n-1,0), (n-1,n-1)}
        for _ in range(n * 3):
            r, c = map(int, rng.integers(0, n, size=2))
            if (r, c) in protected:
                continue
            dr, dc = _N4[int(rng.integers(4))]
            rr, cc = r+dr, c+dc
            if not (0 <= rr < n and 0 <= cc < n):
                continue
            old, new = int(owner[r, c]), int(owner[rr, cc])
            if old == new:
                continue
            owner[r, c] = new
            if len(_components(owner == old)) != 1:
                owner[r, c] = old
        return owner
    owner = np.full((n, n), -1, dtype=np.int8)
    blocks = ((slice(0,2), slice(0,2)), (slice(0,2), slice(n-2,n)),
              (slice(n-2,n), slice(0,2)), (slice(n-2,n), slice(n-2,n)))
    for i, block in enumerate(blocks):
        owner[block] = i
    weights = rng.uniform(0.7, 1.7, size=4)
    if style == 2:
        weights[int(rng.integers(4))] *= float(rng.uniform(3, 7))
    frontier = [set() for _ in range(4)]

    def add_neighbors(r, c, label):
        for dr, dc in _N4:
            rr, cc = r+dr, c+dc
            if 0 <= rr < n and 0 <= cc < n and owner[rr, cc] == -1:
                frontier[label].add((rr, cc))

    for r, c in np.argwhere(owner >= 0):
        add_neighbors(int(r), int(c), int(owner[r,c]))
    for _ in range(n*n-16):
        available = [i for i in range(4) if frontier[i]]
        p = weights[available] / weights[available].sum()
        label = int(rng.choice(available, p=p))
        cells = sorted(frontier[label])
        r, c = cells[int(rng.integers(len(cells)))]
        owner[r, c] = label
        for f in frontier:
            f.discard((r,c))
        add_neighbors(r, c, label)
    return owner


def _make_ink(rng, side, count, rotation, witness, reference=None, inset=0, pattern="mixed"):
    n = side
    if witness:
        gaps = [2, 2, 2, 5]
        if rng.integers(2):
            gaps.reverse()
    else:
        gaps = [2] * (count - 1)
        extra = int(rng.integers(n - 1 - sum(gaps) + 1))
        for _ in range(extra):
            gaps[int(rng.integers(len(gaps)))] += 1
    span = sum(gaps)
    before = int(rng.integers(n - span))
    positions = [before] + [before + int(x) for x in np.cumsum(gaps)]
    capacity = n - inset
    if reference is None:
        reference = int(rng.integers(2, min(9, capacity-1) + 1))
    lengths = [reference] * count
    others = [int(i) for i in rng.permutation(count) if i != count//2]

    def short():
        return int(rng.integers(1, reference))

    def long():
        return int(rng.integers(reference+1, capacity+1))

    if witness:
        majority_short = bool(rng.integers(2))
        for i in others:
            lengths[i] = short() if majority_short else long()
        lengths[others[0]] = long() if majority_short else short()
    elif pattern in ("one_extend", "one_trim"):
        lengths[others[0]] = short() if pattern == "one_extend" else long()
    else:
        for i in others:
            if rng.random() < .2:
                continue
            lengths[i] = short() if pattern == "extend" or (pattern == "mixed" and rng.random() < .5) else long()
        if pattern == "mixed":
            lengths[others[0]], lengths[others[1]] = short(), long()
        elif all(x == reference for x in lengths):
            lengths[others[0]] = short() if pattern == "extend" else long()
    ink, target = np.zeros((n,n), dtype=bool), np.zeros((n,n), dtype=bool)
    for row, length in zip(positions, lengths):
        ink[row, inset:inset+length] = True
        target[row, inset:inset+reference] = True
    return np.rot90(ink, rotation).copy(), np.rot90(target, rotation).copy()


def _interaction(owner, ink, strokes):
    view = owner.T if strokes["transpose"] else owner
    fragments, multicolor, cross_extensions = 0, 0, 0
    for p, length in zip(strokes["positions"], strokes["lengths"]):
        start, direction, ref = strokes["start"], strokes["direction"], strokes["reference"]
        labels = [int(view[p, start+direction*t]) for t in range(length)]
        fragments += len(set(labels)) > 1
        target_labels = [int(view[p, start+direction*t]) for t in range(ref)]
        multicolor += len(set(target_labels)) > 1
        cross_extensions += sum(label != labels[0] for label in target_labels[length:])
    return {"fragmented_strokes": fragments, "multicolor_strokes": multicolor,
            "cross_extensions": cross_extensions,
            "target_territories": tuple(map(int, np.unique(owner[strokes["target"]])))}


def _scatter(rng, arrays, bg):
    """Random ordered shelves (all groupings, both axes), jitter and padding."""
    layouts = []
    for transpose in (False, True):
        dims = [a.shape[::-1] if transpose else a.shape for a in arrays]
        for order in itertools.permutations(range(4)):
            for cuts in range(8):
                groups, group = [], []
                for j, i in enumerate(order):
                    group.append(i)
                    if j == 3 or (cuts >> j) & 1:
                        groups.append(group); group = []
                gap = int(rng.integers(1, 4))
                widths = [sum(dims[i][1] for i in g) + gap*(len(g)-1) for g in groups]
                heights = [max(dims[i][0] for i in g) for g in groups]
                h, w = sum(heights) + gap*(len(groups)-1), max(widths)
                if h <= 28 and w <= 28:
                    layouts.append((transpose, groups, gap, widths, heights, h, w))
    if not layouts:
        return None
    transpose, groups, gap, widths, heights, h, w = layouts[int(rng.integers(len(layouts)))]
    dims = [a.shape[::-1] if transpose else a.shape for a in arrays]
    offsets, row = [None]*4, 0
    for group, width, height in zip(groups, widths, heights):
        col = int(rng.integers(w-width+1))
        for i in group:
            rr = row + int(rng.integers(height-dims[i][0]+1))
            offsets[i] = (col, rr) if transpose else (rr, col)
            col += dims[i][1] + gap
        row += height + gap
    shape = (w,h) if transpose else (h,w)
    # Insets identify the background directly; total margins remain independent
    # of assembly area. All source arrangements can be moved without changing it.
    pads = []
    for size in shape:
        total = int(rng.integers(2, min(8, 30-size)+1))
        before = int(rng.integers(1, total))
        pads.append((before, total-before))
    grid = np.full(tuple(shape[i] + sum(pads[i]) for i in (0,1)), bg, dtype=np.int8)
    for i, a in enumerate(arrays):
        r, c = offsets[i][0]+pads[0][0], offsets[i][1]+pads[1][0]
        occupied = a != bg
        region = grid[r:r+a.shape[0], c:c+a.shape[1]]
        _require(np.all(region[occupied] == bg), "scatter overlap")
        region[occupied] = a[occupied]
    return grid


def _build_pair(rng, plan):
    n, count, rotation, ref, inset, witness, pattern, piece_rotations, style = plan
    for attempt in range(320):
        ink, target = _make_ink(rng, n, count, rotation, witness, ref, inset, pattern)
        # A sampled cut style is an emphasis, not a family constraint. Dense
        # inscriptions can leave the key's clear patch beyond a straight seam;
        # corner growth then allows the anchor to wrap around that seam.
        selected_style = style if attempt < 80 else int(rng.integers(3))
        owner = _partition(rng, n, selected_style)
        masks = [_crop_mask(owner == i)[0] for i in range(4)]
        if any(len(_components(m)) != 1 for m in masks):
            continue
        if witness and any(len({_signature(np.rot90(m,k)) for k in range(4)}) < 4 for m in masks):
            continue
        if witness:
            canonical = [min(_signature(np.rot90(m,k)) for k in range(4)) for m in masks]
            if len(set(canonical)) != 4:
                continue
        base = _pack(masks, [np.zeros_like(m) for m in masks], n)
        if len(base) != 1 or not np.array_equal(base[0]["owner"], owner):
            continue
        local_inks = []
        for i, m in enumerate(masks):
            _, low = _crop_mask(owner == i)
            local_inks.append(ink[low[0]:low[0]+m.shape[0],low[1]:low[1]+m.shape[1]] & m)
        marked = _pack(masks, local_inks, n)
        if len(marked) != 1 or not np.array_equal(marked[0]["ink"],ink):
            continue
        key_locations = [(r,c) for r in range(n-1) for c in range(n-1)
                         if np.all(owner[r:r+2,c:c+2] == 0) and not ink[r:r+2,c:c+2].any()]
        if not key_locations or n*n-int(ink.sum())-4 <= int(ink.sum()):
            continue
        s = _strokes(ink)
        interaction = _interaction(owner, ink, s)
        if witness and (interaction["fragmented_strokes"] < 2 or interaction["cross_extensions"] < 1
                        or len(interaction["target_territories"]) != 4):
            continue
        if interaction["fragmented_strokes"] == 0 or interaction["multicolor_strokes"] == 0:
            continue
        palette = list(map(int, rng.choice(10, 7, replace=False)))
        bg, substrate, ink_color = palette[:3]
        key = np.array(palette[3:], dtype=np.int8).reshape(2,2)
        r, c = key_locations[int(rng.integers(len(key_locations)))]
        material = np.full((n,n), substrate, dtype=np.int8)
        material[ink] = ink_color
        material[r:r+2,c:c+2] = key
        pieces = []
        for i in range(4):
            m, low = _crop_mask(owner == i)
            piece = np.full(m.shape, bg, dtype=np.int8)
            region = material[low[0]:low[0]+m.shape[0],low[1]:low[1]+m.shape[1]]
            piece[m] = region[m]
            pieces.append(np.rot90(piece, piece_rotations[i]).copy())
        grid = _scatter(rng, pieces, bg)
        if grid is None:
            continue
        out = np.full((n,n), bg, dtype=np.int8)
        out[target] = key.ravel()[owner[target]]
        return {"input": grid, "output": out}
    raise RuntimeError(f"bounded search could not construct calibration plan {plan}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base = int(rng.integers(4))
    second = (base + int(rng.choice((1,2,3)))) % 4
    shift = int(rng.integers(3))  # unseen count, longer reference, deeper inset
    first_ref = int(rng.integers(6,10))
    puzzle, seen, train_keys = {"train": [], "test": []}, set(), set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            witness = split == "train" and index == 0
            if witness:
                n, count, rotation, ref, inset = int(rng.integers(12,15)), 5, base, first_ref, 0
                pattern, turns = "mixed", [0, *map(int, rng.permutation((1,2,3)))]
                style = 0
            elif split == "train":
                n = int(rng.integers(8,13)) if index == 1 else int(rng.integers(9,15))
                count = 3 if index == 1 else int(rng.choice((3,5)))
                rotation = second if index == 1 else int(rng.choice((base,second)))
                inset = int(rng.integers(2))
                # The visible substrate, rather than the ink, must be the
                # more numerous of the two repeated foreground colors.
                ceiling = min(9,n-inset-1,(n*n-5)//(2*count)-1)
                choices = [r for r in range(2,ceiling+1) if index != 1 or r != first_ref]
                ref = int(rng.choice(choices))
                pattern = "mixed" if index == 1 else str(rng.choice(("mixed","extend","trim","one_extend","one_trim")))
                turns = [0] + [int(rng.integers(4)) for _ in range(3)]
                style = int(rng.integers(3))
            else:
                rotation = int(rng.choice((base,second)))
                if shift == 0:
                    n, count, inset = int(rng.integers(13,15)), 7, int(rng.integers(2))
                    ref = int(rng.integers(2,min(10,n-inset-1)+1))
                elif shift == 1:
                    n, count, inset = int(rng.integers(12,15)), int(rng.choice((3,5))), 0
                    ref = int(rng.integers(10,n))
                else:
                    n, count, inset = int(rng.integers(10,15)), int(rng.choice((3,5))), int(rng.integers(2,4))
                    ref = int(rng.integers(2,n-inset))
                pattern = str(rng.choice(("mixed","extend","trim","one_extend","one_trim")))
                turns = [0] + [int(rng.integers(4)) for _ in range(3)]
                style = int(rng.integers(3))
            plan = n, count, rotation, ref, inset, witness, pattern, turns, style
            for _ in range(8):
                pair = _build_pair(rng, plan)
                sig = _signature(pair["input"])
                key = tuple(map(int,_parse(pair["input"])["key"].ravel()))
                if split == "train" and index == 1 and key in train_keys:
                    continue
                if sig not in seen:
                    seen.add(sig)
                    if split == "train":
                        train_keys.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unable to make distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    a = _analyze(pair["input"])
    s, assembly = a["strokes"], a["assembly"]
    lengths, ref = s["lengths"], s["reference"]
    distances = [abs(2*p-s["positions"][0]-s["positions"][-1]) for p in s["positions"]]
    closest = [i for i,d in enumerate(distances) if d == min(distances)]
    areas = [int((assembly["owner"] == i).sum()) for i in range(4)]
    fills = [float(m.sum()/m.size) for m in a["shapes"]]
    extensions, trims = sum(l < ref for l in lengths), sum(l > ref for l in lengths)
    return {"side": a["side"], "canvas": a["grid"].shape, "count": len(lengths),
            "reference": ref, "lengths": lengths, "gaps": tuple(map(int,np.diff(s["positions"]))),
            "axis": (int(s["transpose"]), s["direction"]), "inset": s["inset"],
            "nonmedian": sorted(lengths)[len(lengths)//2] != ref,
            "geometric_witness": len(closest) == 1 and closest[0] != len(lengths)//2 and lengths[closest[0]] != ref,
            "pattern": "mixed" if extensions and trims else "extend" if extensions else "trim",
            "changed_strokes": extensions+trims, "extension": max(ref-l for l in lengths),
            "truncation": max(l-ref for l in lengths), "areas": tuple(areas),
            "area_ratio": max(areas)/min(areas), "min_fill": min(fills),
            "rotations": assembly["rotations"], "background": a["background"],
            "substrate": a["substrate"], "ink_color": a["ink_color"],
            "key": tuple(map(int,a["key"].ravel())),
            **_interaction(assembly["owner"], assembly["ink"], s)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train","test"}, "exact puzzle keys required")
    _require(all(isinstance(puzzle[s],list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]),len(puzzle["test"]))
    seen, rows = set(), {"train": [], "test": []}
    for split in ("train","test"):
        for pair in puzzle[split]:
            _require(isinstance(pair,dict) and set(pair) == {"input","output"}, "exact pair keys required")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            a = _analyze(inp)
            target, ink = a["strokes"]["target"], a["assembly"]["ink"]
            _require(np.array_equal(out,_answer(a)), "incorrect calibrated inscription")
            _require(not np.array_equal(target,ink), "calibration must change the ink")
            _require(int(target.sum()) == len(a["strokes"]["lengths"])*a["strokes"]["reference"], "calibrated span invariant")
            _require(np.all(out[~target] == a["background"]), "substrate and key must disappear")
            _require(np.all(out[target] == a["key"].ravel()[a["assembly"]["owner"][target]]), "destination territory determines color")
            for p, length in zip(a["strokes"]["positions"],a["strokes"]["lengths"]):
                view = ink.T if a["strokes"]["transpose"] else ink
                new = target.T if a["strokes"]["transpose"] else target
                start, direction, ref = a["strokes"]["start"],a["strokes"]["direction"],a["strokes"]["reference"]
                prefix = [start+direction*t for t in range(min(length,ref))]
                _require(np.all(view[p,prefix] & new[p,prefix]), "anchor prefix must survive")
            row = measure_pair(pair)
            _require(row["fragmented_strokes"] >= 1 and row["multicolor_strokes"] >= 1,
                     "each example must connect ink fragments and cross a territory")
            rows[split].append(row)
    train = rows["train"]
    _require(len({r["reference"] for r in train}) >= 2 and len({r["count"] for r in train}) >= 2, "training must vary reference and count")
    _require(any(r["nonmedian"] and r["geometric_witness"] and r["pattern"] == "mixed" for r in train), "ordinal reference witness required")
    _require(any(len(r["target_territories"]) == 4 and r["fragmented_strokes"] >= 2 and r["cross_extensions"] > 0 for r in train), "integrated packing/calibration/key witness required")
    _require(any(set(r["rotations"][1:]) == {1,2,3} for r in train), "demonstrate every quarter turn")
    _require(len({r["key"] for r in train}) >= 2, "legend colors must vary")
    for row in rows["test"]:
        _require(row["count"] not in {r["count"] for r in train}
                 or row["reference"] > max(r["reference"] for r in train)
                 or row["inset"] > max(r["inset"] for r in train), "test must transfer a calibrated structural factor")


def _source_pixel(a, target):
    """Find one assembled coordinate back in its original scattered piece."""
    corner = int(a["assembly"]["owner"][target])
    piece = a["assembly"]["pieces"][corner]
    rotation = a["assembly"]["rotations"][corner]
    offset = a["assembly"]["offsets"][corner]
    shape = a["shapes"][piece]
    for r0, c0 in np.argwhere(shape):
        r, c, h, w = int(r0), int(c0), *shape.shape
        for _ in range(rotation):
            r, c, h, w = w-1-c, r, w, h
        if (r+offset[0],c+offset[1]) == target:
            box = a["boxes"][piece]
            return int(r0)+box[0], int(c0)+box[1]
    raise AssertionError("coordinate not in its packed piece")


def self_test():
    """Bounded 42-seed campaign and causal/metamorphic checks; no blind solver."""
    rng = np.random.default_rng(813577)
    records, puzzles = {"train": [], "test": []}, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(tuple(_signature(p["input"]) for s in ("train","test") for p in puzzle[s]))
        if seed < 3:
            other = generate_puzzle(seed)
            _require(all(np.array_equal(p[k],q[k]) for s in puzzle for p,q in zip(puzzle[s],other[s]) for k in p), "determinism")
        for split in records:
            records[split].extend(measure_pair(p) for p in puzzle[split])
        pair = puzzle["train"][0]
        a = _analyze(pair["input"])
        _require(not np.array_equal(_answer(a,a["assembly"]["ink"]),pair["output"]), "ignoring A must change the answer")
        _require(not _pack(a["shapes"],a["inks"],a["side"],allowed_rotations=(0,)),
                 "ignoring B's rotations cannot reconstruct the demonstration")
        # Omitting B's territory coloring: give each stroke its anchor's color.
        fixed = np.full((a["side"],a["side"]),a["background"],dtype=np.int8)
        view = fixed.T if a["strokes"]["transpose"] else fixed
        owner = a["assembly"]["owner"].T if a["strokes"]["transpose"] else a["assembly"]["owner"]
        for p in a["strokes"]["positions"]:
            start, direction = a["strokes"]["start"],a["strokes"]["direction"]
            color = a["key"].ravel()[owner[p,start]]
            for t in range(a["strokes"]["reference"]):
                view[p,start+direction*t] = color
        _require(not np.array_equal(fixed,pair["output"]), "ignoring B's destination territories must change the answer")
        # Shorten ONLY the reference in the returned source input.
        s = a["strokes"]
        p, t = s["positions"][len(s["positions"])//2], s["start"]+s["direction"]*(s["reference"]-1)
        point = (t,p) if s["transpose"] else (p,t)
        original = _source_pixel(a,point)
        altered = pair["input"].copy()
        _require(int(altered[original]) == a["ink_color"], "reference endpoint is visible ink")
        altered[original] = a["substrate"]
        _require(not np.array_equal(solve(altered),pair["output"]), "visible reference must causally control the answer")
        # Literal palette identities are irrelevant.
        perm = rng.permutation(10).astype(np.int8)
        test = puzzle["test"][0]
        _require(np.array_equal(solve(perm[test["input"]]),perm[test["output"]]), "palette permutation must commute")
        # Swap two key entries without moving a footprint or changing any ink.
        key_edit = pair["input"].copy()
        kr, kc = a["key_position"]
        key_edit[kr,kc], key_edit[kr,kc+1] = key_edit[kr,kc+1], key_edit[kr,kc]
        expected = pair["output"].copy()
        x,y = map(int,a["key"][0])
        expected[pair["output"] == x], expected[pair["output"] == y] = y,x
        _require(np.array_equal(solve(key_edit),expected), "visible key must control territory colors")
        if seed < 8:
            arrays = []
            for component, box, shape in zip(a["components"],a["boxes"],a["shapes"]):
                piece = np.full(shape.shape,a["background"],dtype=np.int8)
                local = component-np.array(box)
                piece[local[:,0],local[:,1]] = a["grid"][component[:,0],component[:,1]]
                arrays.append(piece)
            moved = _scatter(rng,arrays,a["background"])
            _require(moved is not None and np.array_equal(solve(moved),pair["output"]),
                     "source location and canvas dimensions must not affect assembly")
    _require(len(puzzles) == 42, "distinct seed puzzles")
    for split, rows in records.items():
        _require(len({r["canvas"] for r in rows}) >= 20, "canvas diversity")
        _require(len({r["reference"] for r in rows}) >= 7, "reference diversity")
        _require(len({r["axis"] for r in rows}) == 4, "all directed cardinal axes")
        _require(len({r["areas"] for r in rows}) >= 20, "fracture geometry diversity")
        _require(max(r["area_ratio"] for r in rows) >= 4, "unequal areas")
        _require(min(r["min_fill"] for r in rows) < .7, "noncompact footprints")
        _require({r["background"] for r in rows} == set(range(10)), "background roles")
        _require(any(r["cross_extensions"] > 0 for r in rows), "cross-fracture extensions")
    _require({r["count"] for r in records["test"]} == {3,5,7}, "test count variation")
    _require({r["pattern"] for r in records["test"]} == {"mixed","extend","trim"}, "correction variation")
    _require(any(r["inset"] >= 2 for r in records["test"]), "inset transfer")
    for train in range(2,11):
        for test in range(1,min(4,train-1)+1):
            p = generate_puzzle(70193,train,test)
            _require(len(p["train"]) == train and len(p["test"]) == test, "explicit counts")
    for train,test in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(3.5,1),(None,1),(4,None)):
        try:
            generate_puzzle(0,train,test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s:[{k:np.array(v).copy() for k,v in p.items()} for p in puzzle[s]] for s in puzzle}
    out = broken["test"][0]["output"]
    out[0,0] = (int(out[0,0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted corrupted output")
    return {"seeds": 42, "puzzles": len(puzzles), "pairs": sum(map(len,records.values())),
            "features": {s:{"counts": sorted({r["count"] for r in rows}),
                            "references": sorted({r["reference"] for r in rows}),
                            "canvas_shapes": len({r["canvas"] for r in rows}),
                            "area_profiles": len({r["areas"] for r in rows}),
                            "max_area_ratio": max(r["area_ratio"] for r in rows),
                            "min_fill": min(r["min_fill"] for r in rows),
                            "patterns": dict(Counter(r["pattern"] for r in rows)),
                            "cross_extension_pairs": sum(r["cross_extensions"] > 0 for r in rows)}
                         for s,rows in records.items()}, "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(),separators=(",",":")))
