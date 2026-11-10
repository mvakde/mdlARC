"""Continuous negative wallpaper, assigned through a deformed room plan.

Only standard-library modules and NumPy are used. Solving never calls generation
and never reads a seed, example collection, file, or sampled parameter.
"""
from __future__ import annotations

from functools import lru_cache
import json
import numpy as np


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid of colors 0..9")
    return a.astype(np.int8, copy=False)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        parts.append(np.asarray(cells, dtype=int))
    return parts


def _bbox(cells):
    r, c = map(int, cells.min(axis=0))
    R, C = map(int, cells.max(axis=0) + 1)
    return r, c, R, C


def _rooms(walls):
    padded = np.pad(~walls, 1, constant_values=True)
    return [p-1 for p in _components(padded)
            if np.all(p > 0) and np.all(p < np.array(padded.shape)-1)]


def _normal_form(walls):
    """Remove strip lengths, retaining every change in the wall arrangement."""
    new_rows = np.r_[True, np.any(walls[1:] != walls[:-1], axis=1)]
    rows = np.flatnonzero(new_rows)
    reduced = walls[rows]
    new_cols = np.r_[True, np.any(reduced[:, 1:] != reduced[:, :-1], axis=0)]
    cols = np.flatnonzero(new_cols)
    return reduced[:, cols], np.cumsum(new_rows)-1, np.cumsum(new_cols)-1


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    result = []
    for a in boxes:
        row = []
        for b in boxes:
            y, x = (-1 if a[k+2] <= b[k] else 1 if b[k+2] <= a[k] else 0
                    for k in (0, 1))
            row.append((y+1)*3+x+1)
        result.append(row)
    return result


def _correspondence(donor_rooms, target_rooms):
    """Unique isomorphism of the directed above/below/left/right spatial graph.

    Overlapping projections are recorded explicitly. Discovery order, areas,
    generation labels, pixel distances, and colors are never matching keys.
    """
    if len(donor_rooms) != len(target_rooms):
        return None
    a, b = _relations(donor_rooms), _relations(target_rooms)
    n = len(a)
    options = [[j for j in range(n) if sorted(a[i]) == sorted(b[j])] for i in range(n)]
    solutions = []

    def search(mapping, used):
        if len(solutions) >= 2:
            return
        if len(mapping) == n:
            solutions.append(dict(mapping))
            return
        remaining = {i: [j for j in options[i] if j not in used and all(
            a[i][k] == b[j][v] and a[k][i] == b[v][j] for k, v in mapping.items())]
            for i in range(n) if i not in mapping}
        i = min(remaining, key=lambda k: len(remaining[k]))
        for j in remaining[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    return solutions[0] if len(solutions) == 1 else None


def _separate(a, b):
    # Half-open boxes need at least one background row or column between them.
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _decode(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    assert all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])), \
        "the input needs its uniform background border"
    candidates = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        frames = []
        for cells in _components(g == color):
            r, c, R, C = _bbox(cells)
            if min(R-r, C-c) < 5:
                continue
            walls = np.zeros((R-r, C-c), dtype=bool)
            walls[cells[:, 0]-r, cells[:, 1]-c] = True
            rooms = _rooms(walls)
            if not 3 <= len(rooms) <= 8:
                continue
            crop = g[r:R, c:C].copy()
            room_mask = np.zeros(crop.shape, dtype=bool)
            colors = []
            for p in rooms:
                room_mask[tuple(p.T)] = True
                values = np.unique(crop[tuple(p.T)])
                if len(values) != 1:
                    break
                colors.append(int(values[0]))
            else:
                if np.any(crop[~(room_mask | walls)] != bg):
                    continue
                if all(x == bg for x in colors):
                    role = "target"
                elif all(x not in (bg, color) for x in colors) and len(set(colors)) >= 2:
                    role = "donor"
                else:
                    continue
                frames.append(dict(role=role, box=(r, c, R, C), crop=crop,
                                   walls=walls, rooms=rooms, colors=colors,
                                   normal=_normal_form(walls)[0]))
        donors = [f for f in frames if f["role"] == "donor"]
        targets = [f for f in frames if f["role"] == "target"]
        for d in donors:
            for t in targets:
                if _separate(d["box"], t["box"]):
                    mapping = _correspondence(d["rooms"], t["rooms"])
                    if mapping is not None:
                        candidates.append((color, d, t, mapping))
    assert len(candidates) == 1, "expected one corresponding donor/target pair"
    wall, donor, target, mapping = candidates[0]
    occupied = np.zeros(g.shape, dtype=bool)
    for f in (donor, target):
        r, c, R, C = f["box"]
        occupied[r:R, c:C] = True
    assert not np.any((g == wall) & ~occupied), "wall color is reserved for frameworks"
    patterns, boxes = {}, []
    for color in map(int, np.unique(g[~occupied])):
        if color == bg:
            continue
        cells = np.argwhere((g == color) & ~occupied)
        box = _bbox(cells)
        r, c, R, C = box
        assert 2 <= R-r <= 3 and 2 <= C-c <= 3, "unregistered stencil"
        assert not occupied[r:R, c:C].any()
        register = g[r:R, c:C]
        assert np.all((register == bg) | (register == color))
        mask = register == color
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        assert 0 < mask.sum() < mask.size
        assert all(_separate(box, other) for other in boxes)
        assert all(_separate(box, f["box"]) for f in (donor, target))
        patterns[color] = mask
        boxes.append(box)
    colors = [None]*len(target["rooms"])
    for i, j in mapping.items():
        colors[j] = donor["colors"][i]
    assert set(colors) <= set(patterns), "every room color needs one external stencil"
    return dict(background=bg, wall=wall, donor=donor, target=target,
                colors=colors, patterns=patterns, stencil_boxes=boxes, mapping=mapping)


def _transforms(mask):
    return [np.rot90(mask, k) for k in range(4)] + [
        np.rot90(np.fliplr(mask), k) for k in range(4)]


def _texture_evidence(info):
    repeated = clipped = False
    starts = {}
    for cells, color in zip(info["target"]["rooms"], info["colors"]):
        mh, mw = info["patterns"][color].shape
        r, c, _, _ = _bbox(cells)
        starts.setdefault(color, set()).add(((r-1) % mh, (c-1) % mw))
        for fixed, moving, period in ((0, 1, mw), (1, 0, mh)):
            for line in np.unique(cells[:, fixed]):
                positions = sorted(cells[cells[:, fixed] == line, moving])
                length = 0
                previous = -2
                for position in positions:
                    length = length+1 if position == previous+1 else 1
                    repeated |= length >= 2*period
                    previous = position
                # Inspect whole maximal runs, not arbitrary prefixes.
                breaks = np.r_[0, np.flatnonzero(np.diff(positions) > 1)+1, len(positions)]
                for a, b in zip(breaks[:-1], breaks[1:]):
                    clipped |= b-a > period and (b-a) % period != 0
    coupled = any(len(phase_set) > 1 for phase_set in starts.values())
    return bool(repeated), bool(clipped), coupled


def _answer(info, *, reset=False, uniform=False, first_color=False, transform=0,
            scan=False, shift=(0, 0)):
    """The optional alternatives are used only for counterexample checks."""
    out = info["target"]["crop"].copy()
    bg = info["background"]
    for index, (cells, color) in enumerate(zip(info["target"]["rooms"], info["colors"])):
        if scan:
            color = info["donor"]["colors"][index]
        if first_color:
            color = info["colors"][0]
        out[tuple(cells.T)] = color
        if uniform:
            continue
        pattern = info["patterns"][color]
        if transform:
            pattern = _transforms(pattern)[transform]
        r0, c0 = cells.min(axis=0) if reset else (1+shift[0], 1+shift[1])
        holes = pattern[(cells[:, 0]-r0) % pattern.shape[0],
                        (cells[:, 1]-c0) % pattern.shape[1]]
        out[tuple(cells[holes].T)] = bg
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Match rooms from wall geometry, then expose their negative wallpaper."""
    return _answer(_decode(input_grid))


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _plan(rng, count):
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3)]
    elbow = rng.random() < .65
    rows, cols = _pick(rng, [s for s in shapes if count+(2 if elbow else 0) <= s[0]*s[1] <= count+5])
    active = np.ones((rows, cols), dtype=bool)
    corners = [(0, 0), (0, cols-1), (rows-1, 0), (rows-1, cols-1)]
    for index in rng.permutation(4):
        p = corners[int(index)]
        if active.sum() > count+(2 if elbow else 0) and rng.random() < .55:
            trial = active.copy()
            trial[p] = False
            if len(_components(trial)) == 1 and trial.any(0).all() and trial.any(1).all():
                active = trial
    plan = np.full(active.shape, -1, dtype=int)
    plan[active] = np.arange(active.sum())
    groups = int(active.sum())
    if elbow:
        elbows = []
        for r in range(rows-1):
            for c in range(cols-1):
                square = [(r, c), (r+1, c), (r, c+1), (r+1, c+1)]
                for omit in range(4):
                    points = [p for j, p in enumerate(square) if j != omit]
                    if all(active[p] for p in points):
                        elbows.append(points)
        if elbows:
            points = _pick(rng, elbows)
            label = int(plan[points[0]])
            for point in points:
                plan[point] = label
            groups -= 2
    while groups > count:
        edges = sorted({(int(plan[r, c]), int(plan[r+dr, c+dc]))
                        for r, c in np.argwhere(active) for dr, dc in ((1, 0), (0, 1))
                        if r+dr < rows and c+dc < cols and plan[r+dr, c+dc] >= 0
                        and plan[r, c] != plan[r+dr, c+dc]})
        valid = []
        for a, b in edges:
            merged = (plan == a) | (plan == b)
            if not _rooms(merged):  # no room may surround another room
                valid.append((a, b))
        assert valid
        a, b = _pick(rng, valid)
        plan[plan == b] = a
        groups -= 1
    # Renumber through a copy, avoiding collisions between old and new labels.
    result = np.full(plan.shape, -1, dtype=int)
    for i, old in enumerate(sorted(set(plan.flat)-{-1})):
        result[plan == old] = i
    return result


def _geometry(plan, heights, widths):
    ys = np.r_[0, np.cumsum(np.asarray(heights)+1)]
    xs = np.r_[0, np.cumsum(np.asarray(widths)+1)]
    labels = np.full((int(ys[-1])+1, int(xs[-1])+1), -1, dtype=int)
    for r, c in np.argwhere(plan >= 0):
        label = plan[r, c]
        labels[ys[r]+1:ys[r+1], xs[c]+1:xs[c+1]] = label
        if r and plan[r-1, c] == label:
            labels[ys[r], xs[c]+1:xs[c+1]] = label
        if c and plan[r, c-1] == label:
            labels[ys[r]+1:ys[r+1], xs[c]] = label
        if r and c and np.all(plan[r-1:r+1, c-1:c+1] == label):
            labels[ys[r], xs[c]] = label
    inside = labels >= 0
    padded = np.pad(inside, 1)
    near = np.zeros(labels.shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr+labels.shape[0], dc:dc+labels.shape[1]]
    labels[near & ~inside] = -2
    assert len(_components(labels == -2)) == 1, "disconnected framework"
    rooms = _rooms(labels == -2)
    assert len(rooms) == len(set(plan.flat)-{-1})
    for p in rooms:
        assert len(np.unique(labels[tuple(p.T)])) == 1 and labels[tuple(p[0])] >= 0
    return labels


def _deform(rng, labels, target):
    """Independently move room edges, allowing scan order to change.

    A local edge recedes into a room, turning its old edge into wall. Requiring
    the visible spatial graph to match later rejects destructive deformations.
    """
    labels = labels.copy()
    for label in sorted(set(labels.flat)-{-1, -2}):
        if rng.random() < .25:
            continue
        for side in rng.permutation(4)[:int(rng.integers(1, 3))]:
            cells = np.argwhere(labels == label)
            axis = int(side)//2
            edge = (cells[:, axis].min() if int(side) % 2 == 0 else cells[:, axis].max())
            trimmed = cells[cells[:, axis] == edge]
            trial = labels == label
            trial[tuple(trimmed.T)] = False
            if not trial.any() or len(_components(trial)) != 1:
                continue
            if target and not np.any(trial[:-1, :-1] & trial[1:, :-1]
                                     & trial[:-1, 1:] & trial[1:, 1:]):
                continue
            labels[tuple(trimmed.T)] = -2
    return labels


def _sizes(rng, n, target, compact):
    if compact:
        return np.full(n, 3 if target else 2, dtype=int)
    low, high = (2, 5) if target else (1, 5)
    sizes = rng.integers(low, high, size=n)
    while sizes.sum()+n+1 > 17:
        choices = np.flatnonzero(sizes > low)
        sizes[int(_pick(rng, choices))] -= 1
    return sizes


@lru_cache(maxsize=4)
def _catalog(height, width):
    result = []
    for bits in range(1, (1 << (height*width))-1):
        mask = np.array([(bits >> k) & 1 for k in range(height*width)],
                        dtype=bool).reshape(height, width)
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        asymmetric = height == width and len({m.tobytes() for m in _transforms(mask)}) == 8
        symmetric = np.array_equal(mask, mask[::-1]) or np.array_equal(mask, mask[:, ::-1])
        primitive = asymmetric and len({np.roll(mask, (r, c), axis=(0, 1)).tobytes()
                                       for r in range(height) for c in range(width)}) == mask.size
        result.append((mask, asymmetric, symmetric, primitive))
    return tuple(result)


def _pattern(rng, labels, color, color_by_label, witness=False, small=False, rectangle=False):
    rooms = [np.argwhere(labels == label) for label, c in enumerate(color_by_label) if c == color]
    shape = ((3, 3) if witness else (2, 2) if small else
             _pick(rng, [(2, 3), (3, 2)]) if rectangle else
             _pick(rng, [(2, 2), (2, 3), (3, 2), (3, 3)]))
    choices = []
    for mask, asymmetric, symmetric, primitive in _catalog(*shape):
        if witness and (not asymmetric or not primitive or 2*mask.sum() == mask.size):
            continue
        if small and mask.sum() != 3:
            continue
        if all(0 < np.count_nonzero(mask[(p[:, 0]-1) % shape[0],
                                         (p[:, 1]-1) % shape[1]]) < len(p) for p in rooms):
            choices.append((mask, asymmetric, symmetric, primitive))
    assert choices, "no stencil exposes both ink and holes"
    style = int(rng.integers(4))
    styled = [entry for entry in choices if (
        entry[0].sum() <= max(2, entry[0].size//3) if style == 0 else
        entry[0].sum() >= entry[0].size-2 if style == 1 else
        entry[2] if style == 2 else True)]
    return _pick(rng, styled or choices)[0].copy()


def _paint(labels, bg, wall, colors=None):
    grid = np.full(labels.shape, bg, dtype=np.int8)
    grid[labels == -2] = wall
    if colors is not None:
        for label, color in enumerate(colors):
            grid[labels == label] = color
    return grid


def _layout(rng, donor, target, patterns, bg):
    """Enumerate fitting shelf arrangements; optionally scatter the registers."""
    dh, dw = donor.shape
    th, tw = target.shape
    n = len(patterns)
    options = []
    gap = int(rng.integers(1, 4))
    bank_gap = int(rng.integers(1, 4))
    for horizontal in (False, True):
        gh, gw = ((max(dh, th), dw+gap+tw) if horizontal else (dh+gap+th, max(dw, tw)))
        for nc in range(1, n+1):
            nr = (n+nc-1)//nc
            bh, bw = nr*4-1, nc*4-1
            for side in (False, True):
                h, w = ((max(gh, bh), gw+bank_gap+bw) if side else (gh+bank_gap+bh, max(gw, bw)))
                if h <= 28 and w <= 28:
                    options.append((horizontal, gh, gw, nc, bh, bw, side, h, w))
    if not options:
        raise AssertionError("frameworks and bank do not fit")
    horizontal, gh, gw, nc, bh, bw, side, h, w = _pick(rng, options)
    if horizontal:
        dp = (int(rng.integers(gh-dh+1)), 0)
        tp = (int(rng.integers(gh-th+1)), dw+gap)
    else:
        dp = (0, int(rng.integers(gw-dw+1)))
        tp = (dh+gap, int(rng.integers(gw-tw+1)))
    # Reverse the order of the two objects without reflecting their contents.
    if rng.random() < .5:
        if horizontal:
            dp, tp = (dp[0], gw-dw-dp[1]), (tp[0], gw-tw-tp[1])
        else:
            dp, tp = (gh-dh-dp[0], dp[1]), (gh-th-tp[0], tp[1])
    if side:
        gp, bp = (int(rng.integers(h-gh+1)), 0), (int(rng.integers(h-bh+1)), gw+bank_gap)
    else:
        gp, bp = (0, int(rng.integers(w-gw+1))), (gh+bank_gap, int(rng.integers(w-bw+1)))
    if rng.random() < .5:
        if side:
            gp, bp = (gp[0], w-gw-gp[1]), (bp[0], w-bw-bp[1])
        else:
            gp, bp = (h-gh-gp[0], gp[1]), (h-bh-bp[0], bp[1])
    extra_r = int(rng.integers(30-h-1))
    extra_c = int(rng.integers(30-w-1))
    top, left = 1+int(rng.integers(extra_r+1)), 1+int(rng.integers(extra_c+1))
    height, width = h+2+extra_r, w+2+extra_c
    dp = (dp[0]+gp[0]+top, dp[1]+gp[1]+left)
    tp = (tp[0]+gp[0]+top, tp[1]+gp[1]+left)
    colors = list(patterns)
    slots = {}
    for slot, index in enumerate(rng.permutation(n)):
        color = colors[int(index)]
        mh, mw = patterns[color].shape
        slots[color] = (bp[0]+top+(slot//nc)*4+int(rng.integers(4-mh)),
                        bp[1]+left+(slot % nc)*4+int(rng.integers(4-mw)))
    if rng.random() < .55:
        forbidden = np.zeros((height, width), dtype=bool)
        for p, shape in ((dp, donor.shape), (tp, target.shape)):
            y, x = p
            hh, ww = shape
            forbidden[y-1:y+hh+1, x-1:x+ww+1] = True
        scattered = {}
        for index in rng.permutation(n):
            color = colors[int(index)]
            mh, mw = patterns[color].shape
            positions = [(y, x) for y in range(1, height-mh) for x in range(1, width-mw)]
            for j in rng.permutation(len(positions)):
                y, x = positions[int(j)]
                if not forbidden[y:y+mh, x:x+mw].any():
                    scattered[color] = (y, x)
                    forbidden[y-1:y+mh+1, x-1:x+mw+1] = True
                    break
            else:
                break
        if len(scattered) == n:
            slots = scattered
    canvas = np.full((height, width), bg, dtype=np.int8)
    for image, (y, x) in ((donor, dp), (target, tp)):
        canvas[y:y+image.shape[0], x:x+image.shape[1]] = image
    for color, (y, x) in slots.items():
        mask = patterns[color]
        canvas[y:y+mask.shape[0], x:x+mask.shape[1]][mask] = color
    return canvas


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _fallback_geometry(count, witness):
    if witness:
        plan = np.array([[0, 1, 1], [2, 2, 2]] if count == 3 else
                        [[0, 1, 2], [3, 3, 3]] if count == 4 else
                        [[0, 1, 2], [3, 3, 4]], dtype=int)
        donor = _geometry(plan, [3, 3], [2, 2, 2])
        target = _geometry(plan, [4, 4], [3, 3, 3])
        # Recede the left upper room, so its right neighbor is discovered first.
        top = np.argwhere(target == 0)[:, 0].min()
        target[(target == 0) & (np.indices(target.shape)[0] == top)] = -2
    else:
        cols = (count+1)//2
        plan = np.arange(2*cols).reshape(2, cols)
        plan[plan >= count] = -1
        donor = _geometry(plan, [2, 2], [2]*cols)
        target = _geometry(plan, [3, 3], [3]*cols)
    return donor, target


def _make_pair(rng, count, witness, color_count=None, force_fallback=False):
    for attempt in range(64):
        try:
            fallback = force_fallback or attempt >= 48
            if fallback:
                d, t = _fallback_geometry(count, witness)
            else:
                plan = _plan(rng, count)
                compact = attempt >= 32
                d = _geometry(plan, _sizes(rng, plan.shape[0], False, compact),
                              _sizes(rng, plan.shape[1], False, compact))
                t = _geometry(plan, _sizes(rng, plan.shape[0], True, compact),
                              _sizes(rng, plan.shape[1], True, compact))
                d, t = _deform(rng, d, False), _deform(rng, t, True)
            palette = list(map(int, rng.permutation(10)))
            bg, wall = palette[:2]
            richness = color_count or int(rng.integers(2, min(6, count)+1))
            room_palette = palette[2:2+richness]
            colors = room_palette + [int(_pick(rng, room_palette)) for _ in range(count-richness)]
            rng.shuffle(colors)
            if fallback and witness:
                a, b = room_palette[:2]
                colors = [a, b, a] if count == 3 else [a, b, a, a] if count == 4 else [
                    a, b, room_palette[2], a, b]
            patterns = {c: _pattern(rng, t, c, colors, witness and i == 0,
                                    witness and i == 1, witness and i == 2)
                        for i, c in enumerate(room_palette)}
            distractors = int(rng.integers(1 if witness else 0, min(2, 8-richness)+1))
            for color in palette[2+richness:2+richness+distractors]:
                # Distractors have the same visible format and mask vocabulary.
                shape = _pick(rng, [(2, 2), (2, 3), (3, 2), (3, 3)])
                patterns[color] = _pick(rng, _catalog(*shape))[0].copy()
            donor, target = _paint(d, bg, wall, colors), _paint(t, bg, wall)
            inp = _layout(rng, donor, target, patterns, bg)
            # Construct labels directly from the abstract plan, independently
            # of the input decoder's spatial-graph room correspondence.
            out = target.copy()
            for label, color in enumerate(colors):
                p = np.argwhere(t == label)
                mask = patterns[color]
                holes = mask[(p[:, 0]-1) % mask.shape[0], (p[:, 1]-1) % mask.shape[1]]
                out[tuple(p.T)] = color
                out[tuple(p[holes].T)] = bg
            info = _decode(inp)
            assert np.array_equal(_answer(info), out), "visible matching disagrees with construction"
            if witness:
                assert not np.array_equal(_answer(info, reset=True), out)
                assert all(not np.array_equal(_answer(info, transform=k), out) for k in range(1, 8))
                assert all(_texture_evidence(info)), "need visible repetition, clipping, and coupled phases"
                assert not np.array_equal(_answer(info, scan=True), out), \
                    "need a counterexample to scan-order matching"
                assert all(not np.array_equal(_answer(info, shift=(r, c)), out)
                           for r in range(6) for c in range(6) if (r, c) != (0, 0)), \
                    "need to expose the wallpaper origin modulo all period sizes"
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("bounded floorplan construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", nt), ("test", ne)):
        for i in range(n):
            witness = split == "train" and i < 2
            count = (int(rng.integers(3, 5)) if i == 0 else 5) if witness else (
                int(rng.integers(3, 6)) if split == "train" else int(rng.integers(6, 9)))
            richness = (2 if i == 0 else 3) if witness else None
            for _ in range(16):
                pair = _make_pair(rng, count, witness, richness)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    return puzzle


def _measure(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    info = _decode(inp)
    expected = _answer(info)
    assert np.array_equal(expected, out), "incorrect wallpaper, color assignment, or crop"
    target, donor = info["target"], info["donor"]
    walls = target["walls"]
    room_mask = np.zeros(out.shape, dtype=bool)
    for cells, color in zip(target["rooms"], info["colors"]):
        room_mask[tuple(cells.T)] = True
        values = out[tuple(cells.T)]
        assert set(map(int, values)) == {info["background"], color}, "each sampled room exposes both signs"
    assert np.array_equal(out[walls], target["crop"][walls]), "changed framework"
    assert np.array_equal(out[~room_mask], target["crop"][~room_mask]), "changed exterior"
    assert not np.array_equal(_answer(info, uniform=True), out), "B has no effect"
    assert not np.array_equal(_answer(info, first_color=True), out), "A has no effect"
    area_signs = {int(np.sign(len(target["rooms"][j])-len(donor["rooms"][i])))
                  for i, j in info["mapping"].items()}
    info["area_signs"] = area_signs
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _count(len(puzzle["train"]), "num_train", 2, 10)
    _count(len(puzzle["test"]), "num_test", 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                _grid(grid)
            key = _key(pair["input"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _measure(pair)
            if split == "train":
                training.append(info)
    assert any(not np.array_equal(_answer(i, reset=True), _answer(i)) for i in training), \
        "training fails to establish continuous phase"
    for k in range(1, 8):
        assert any(not np.array_equal(_answer(i, transform=k), _answer(i)) for i in training), \
            "training leaves a competing stencil orientation"
    assert any(len(set(i["colors"])) < len(i["colors"]) for i in training), \
        "training needs repeated colors in distinct rooms"
    assert any(set(i["patterns"])-set(i["colors"]) for i in training), \
        "training needs an ignored stencil"
    assert any(mask.shape[0] != mask.shape[1] for i in training
               for color, mask in i["patterns"].items() if color in i["colors"]), \
        "training needs a matched rectangular period"
    assert all(any(_texture_evidence(i)[j] for i in training) for j in range(3)), \
        "training needs repetition, clipping, and same-color rooms with different phases"
    assert any(not np.array_equal(_answer(i, scan=True), _answer(i)) for i in training), \
        "training needs a counterexample to scan-order room matching"
    for r in range(6):
        for c in range(6):
            if (r, c) != (0, 0):
                assert any(not np.array_equal(_answer(i, shift=(r, c)), _answer(i)) for i in training), \
                    "training leaves another wallpaper origin"


def _fingerprint(puzzle):
    return tuple(_key(p[role]) for s in ("train", "test") for p in puzzle[s]
                 for role in ("input", "output"))


def self_test():
    """Bounded structural and semantic audit over 42 seeds, plus mutations."""
    signatures, normal_forms, shapes, output_shapes = set(), set(), set(), set()
    backgrounds, walls, periods, room_counts = set(), set(), set(), set()
    nonrectangular = phase_witnesses = mixed_area = sparse = dense = scan_witnesses = 0
    independent_geometry = 0
    height_signs, width_signs = set(), set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        train_counts, test_counts = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _measure(pair)
                pairs += 1
                t, d = info["target"], info["donor"]
                backgrounds.add(info["background"])
                walls.add(info["wall"])
                normal_forms.add(_key(t["normal"]))
                room_counts.add(len(t["rooms"]))
                (train_counts if split == "train" else test_counts).append(len(t["rooms"]))
                shapes.add(np.asarray(pair["input"]).shape)
                output_shapes.add(np.asarray(pair["output"]).shape)
                height_signs.add(int(np.sign(t["crop"].shape[0]-d["crop"].shape[0])))
                width_signs.add(int(np.sign(t["crop"].shape[1]-d["crop"].shape[1])))
                phase_witnesses += not np.array_equal(_answer(info, reset=True), pair["output"])
                scan_witnesses += not np.array_equal(_answer(info, scan=True), pair["output"])
                independent_geometry += not np.array_equal(t["normal"], d["normal"])
                mixed_area += {-1, 1} <= info["area_signs"]
                for p in t["rooms"]:
                    r, c, R, C = _bbox(p)
                    nonrectangular += len(p) != (R-r)*(C-c)
                for mask in info["patterns"].values():
                    periods.add(mask.shape)
                    sparse += mask.sum() <= max(2, mask.size//3)
                    dense += mask.sum() >= mask.size-2
                before = np.asarray(pair["input"]).copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"]), "solver mutated input"
        assert max(train_counts) < min(test_counts), "lost room-count extrapolation"
    assert len(signatures) == 42 and pairs == 210
    assert len(normal_forms) >= 60 and len(shapes) >= 35 and len(output_shapes) >= 25
    assert backgrounds == walls == set(range(10))
    assert periods == {(2, 2), (2, 3), (3, 2), (3, 3)}
    assert room_counts == set(range(3, 9))
    assert min(nonrectangular, phase_witnesses, sparse, dense) >= 40
    assert scan_witnesses >= 84 and independent_geometry >= 40
    assert mixed_area >= 5 and height_signs == width_signs == {-1, 0, 1}
    original = generate_puzzle(0)
    assert _fingerprint(original) == _fingerprint(generate_puzzle(0))
    # Exercise the fitting, visibly distinguished bounded construction directly.
    for count in range(3, 9):
        _measure(_make_pair(np.random.default_rng(730+count), count, False, force_fallback=True))
    for count in (3, 4, 5):
        fallback = _make_pair(np.random.default_rng(770+count), count, True,
                              3 if count == 5 else 2, force_fallback=True)
        info = _measure(fallback)
        assert all(_texture_evidence(info))
        assert not np.array_equal(_answer(info, scan=True), fallback["output"])
    # Literal palette roles are equivariant, including nonzero backgrounds.
    pair = original["train"][0]
    recoloring = np.array([6, 9, 1, 4, 7, 0, 8, 2, 5, 3], dtype=np.int8)
    assert np.array_equal(solve(recoloring[pair["input"]]), recoloring[pair["output"]])
    # Moving the scene must not move the target-relative wallpaper origin.
    for seed in range(42):
        small = generate_puzzle(seed)["train"][0]
        if max(small["input"].shape) <= 28:
            bg = int(small["input"][0, 0])
            assert np.array_equal(solve(np.pad(small["input"], 1, constant_values=bg)), small["output"])
            break
    else:
        raise AssertionError("no compact example for padding check")
    # A distractor has the same syntax but no effect on the answer.
    info = _decode(pair["input"])
    unused = next(iter(set(info["patterns"])-set(info["colors"])))
    changed = pair["input"].copy()
    external = changed == unused  # this color occurs only in its unused register
    changed[external] = info["background"]
    assert np.array_equal(solve(changed), pair["output"])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(901, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
            validate_generated_puzzle(p)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in original[s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("accepted a damaged framework or exterior")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "floorplans": len(normal_forms), "input_shapes": len(shapes),
            "output_shapes": len(output_shapes), "nonrectangular_rooms": nonrectangular,
            "phase_witnesses": phase_witnesses, "mixed_growth_shrinkage": mixed_area,
            "scan_order_witnesses": scan_witnesses, "nonseparable_geometry": independent_geometry,
            "period_shapes": sorted(periods)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
