"""Compass docking, followed by global equality-run contraction.

No parent module, file, seed, or generator state is used by solve().  The
constructor works with colored cell maps, not whole parent puzzle generators.
Only the final answer is contracted: rigid docking uses the original metric.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


SIDES = ("top", "right", "bottom", "left")
STEP = ((-1, 0), (0, 1), (1, 0), (0, -1))
ADJ8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _transform(cells, t):
    points = []
    for (r, c), value in cells.items():
        r, c = int(r), int(c)
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        points.append((r, c, int(value)))
    r0, c0 = min(r for r, c, v in points), min(c for r, c, v in points)
    return {(r - r0, c - c0): v for r, c, v in points}


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _canonical(cells):
    return min(_signature(_transform(cells, t)) for t in range(8))


def _shift(cells, origin):
    return {(r + origin[0], c + origin[1]): v for (r, c), v in cells.items()}


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components(mask, diagonal=True):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in (ADJ8 if diagonal else STEP):
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    component.add(q)
        result.append(component)
    return result


def _groups(indices):
    groups = []
    for i in sorted(indices):
        if not groups or i != groups[-1][-1] + 1:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def _structure(a, background):
    """Peel complete bands, allowing either band to own a crossing rectangle."""
    rows, cols = set(), set()
    h, w = a.shape
    for _ in range(h + w):
        old = len(rows) + len(cols)
        active_c = [c for c in range(w) if c not in cols]
        for r in range(h):
            v = a[r, active_c]
            if len(v) and v[0] != background and np.all(v == v[0]):
                rows.add(r)
        active_r = [r for r in range(h) if r not in rows]
        for c in range(w):
            v = a[active_r, c]
            if len(v) and v[0] != background and np.all(v == v[0]):
                cols.add(c)
        if len(rows) + len(cols) == old:
            break
    rg, cg = _groups(rows), _groups(cols)
    if (not rg or not cg or rg[0][0] != 0 or rg[-1][-1] != h - 1
            or cg[0][0] != 0 or cg[-1][-1] != w - 1
            or len(rows) == h or len(cols) == w):
        raise ValueError("canvas must have a readable colored frame")
    if any(not 1 <= len(g) <= 4 for g in rg + cg):
        raise ValueError("band thickness must be 1..4")
    active_r, active_c = sorted(set(range(h)) - rows), sorted(set(range(w)) - cols)
    row_colors = [int(a[g[0], active_c[0]]) for g in rg]
    col_colors = [int(a[active_r[0], g[0]]) for g in cg]
    if len(set(row_colors + col_colors)) != len(rg) + len(cg):
        raise ValueError("each structural band needs its own identity")
    for g, color in zip(rg, row_colors):
        if not np.all(a[np.ix_(g, active_c)] == color):
            raise ValueError("inconsistent horizontal band")
    for g, color in zip(cg, col_colors):
        if not np.all(a[np.ix_(active_r, g)] == color):
            raise ValueError("inconsistent vertical band")
    for rr, hc in zip(rg, row_colors):
        for cc, vc in zip(cg, col_colors):
            block = a[np.ix_(rr, cc)]
            if not (np.all(block == hc) or np.all(block == vc)):
                raise ValueError("crossing rectangle must show one participating color")
    mask = np.zeros(a.shape, dtype=bool)
    mask[list(rows), :] = True
    mask[:, list(cols)] = True
    return mask, rg, cg


def _regions(a, structural):
    rectangles, sides = [], []
    indices = np.full(a.shape, -1, dtype=np.int16)
    for i, region in enumerate(_components(~structural, diagonal=False)):
        r0, r1 = min(r for r, c in region), 1 + max(r for r, c in region)
        c0, c1 = min(c for r, c in region), 1 + max(c for r, c in region)
        if len(region) != (r1 - r0) * (c1 - c0):
            raise ValueError("destination regions must be rectangular")
        rectangles.append((r0, r1, c0, c1))
        indices[r0:r1, c0:c1] = i
        sides.append(dict(zip(SIDES, (int(a[r0 - 1, c0]), int(a[r0, c1]),
                                     int(a[r1, c0]), int(a[r0, c0 - 1])))))
    return rectangles, indices, sides


def _directions(cells, band_colors):
    core = [p for p, v in cells.items() if v not in band_colors]
    if not core:
        return None
    r0, r1 = min(r for r, c in core), max(r for r, c in core)
    c0, c1 = min(c for r, c in core), max(c for r, c in core)
    labels = {}
    for (r, c), value in cells.items():
        if value in band_colors:
            facing = [s for s, ok in zip(SIDES, (r < r0, c > c1, r > r1, c < c0)) if ok]
            if len(facing) != 1:
                return None
            labels[(r, c)] = facing[0]
    return labels or None


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    background = int(counts.argmax())
    if np.sum(counts == counts[background]) != 1:
        raise ValueError("background must be uniquely most frequent")
    structural, rg, cg = _structure(a, background)
    bands = set(int(v) for v in a[structural])
    sources, targets, target_colors = [], [], set()
    for comp in _components((a != background) & ~structural):
        cells = {p: int(a[p]) for p in comp}
        if len(set(cells.values())) == 1:
            targets.append(comp)
            target_colors.update(cells.values())
        else:
            sources.append(cells)
    if not sources or len(sources) != len(targets) or len(target_colors) != 1:
        raise ValueError("need equally many composite pieces and detached stencils")
    registration = next(iter(target_colors))
    bodies = set()
    for source in sources:
        if registration not in source.values() or not _directions(source, bands):
            raise ValueError("piece needs registration cells and clear outward labels")
        bodies.update(set(source.values()) - bands - {registration})
    if len(bodies) != 1:
        raise ValueError("need one common body role")
    body = next(iter(bodies))
    if len({background, body, registration} | bands) != len(bands) + 3:
        raise ValueError("color roles must be distinct")
    rects, region_index, sides = _regions(a, structural)
    return dict(background=background, body=body, registration=registration,
                bands=bands, structural=structural, sources=sources, targets=targets,
                rectangles=rects, region_index=region_index, sides=sides,
                row_bands=rg, col_bands=cg)


def _choices(source, info, use_labels):
    choices, seen = [], set()
    h, w = info["structural"].shape
    for t in range(8):
        cells = _transform(source, t)
        marks = {p for p, v in cells.items() if v == info["registration"]}
        labels = _directions(cells, info["bands"])
        if labels is None:
            continue
        for j, target in enumerate(info["targets"]):
            if len(target) != len(marks):
                continue
            p, q = min(marks), min(target)
            delta = (q[0] - p[0], q[1] - p[1])
            if {(r + delta[0], c + delta[1]) for r, c in marks} != target:
                continue
            placed = _shift(cells, delta)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in placed):
                continue
            region_ids = {int(info["region_index"][p]) for p in placed}
            if len(region_ids) != 1 or -1 in region_ids:
                continue
            sides = info["sides"][next(iter(region_ids))]
            if use_labels and any(cells[p] != sides[d] for p, d in labels.items()):
                continue
            signature = tuple(sorted(placed.items()))
            if signature not in seen:
                seen.add(signature)
                choices.append((j, placed))
    return choices


def _maps(info, use_labels=True, limit=2):
    choices = [_choices(s, info, use_labels) for s in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    found = {}

    def search(depth, used, occupied):
        if len(found) >= limit:
            return
        if depth == len(order):
            found[tuple(sorted(occupied.items()))] = occupied
            return
        for target, placed in choices[order[depth]]:
            if target not in used and not occupied.keys() & placed.keys():
                search(depth + 1, used | {target}, {**occupied, **placed})

    search(0, set(), {})
    return list(found.values())


def _canvas(a, info, cells):
    result = np.full(a.shape, info["background"], dtype=np.int8)
    result[info["structural"]] = a[info["structural"]]
    for p, v in cells.items():
        result[p] = v
    return result


def _dock(a):
    info = _parse(a)
    maps = _maps(info)
    if len(maps) != 1:
        raise ValueError("input must determine exactly one docked pixel arrangement")
    return _canvas(a, info, maps[0]), info


def _keep(a, axis):
    vectors = a if axis == 0 else a.T
    return np.r_[True, np.any(vectors[1:] != vectors[:-1], axis=1)]


def _collapse(a):
    # Both masks can be computed on the full docked grid. Removing identical
    # vectors on one axis cannot erase a distinction on the other axis.
    return a[np.ix_(_keep(a, 0), _keep(a, 1))].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    docked, _ = _dock(_grid(input_grid))
    return _collapse(docked)


def _path(start, end, rng):
    r, c = start
    result = {(r, c)}
    while (r, c) != end:
        if r != end[0] and (c == end[1] or rng.random() < .5):
            r += 1 if end[0] > r else -1
        else:
            c += 1 if end[1] > c else -1
        result.add((r, c))
    return result


def _pattern(rng, h, w, forced=None):
    if forced == "orthogonal":
        points = {(0, 0), (0, 1)}
    elif forced == "diagonal":
        points = {(0, 0), (1, 1)}
    else:
        points = {(0, 0)}
        size = int(rng.integers(1, min(6, h * w - 1)))
        for _ in range(50):
            if len(points) >= size:
                break
            r, c = sorted(points)[int(rng.integers(len(points)))]
            dr, dc = ADJ8[int(rng.integers(len(ADJ8)))]
            trial = points | {(r + dr, c + dc)}
            if (max(r for r, c in trial) - min(r for r, c in trial) < h
                    and max(c for r, c in trial) - min(c for r, c in trial) < w):
                points = trial
    poses = [_transform({p: 1 for p in points}, t) for t in range(8)]
    poses = [p for p in poses if max(r for r, c in p) < h and max(c for r, c in p) < w]
    points = poses[int(rng.integers(len(poses)))]
    ph, pw = 1 + max(r for r, c in points), 1 + max(c for r, c in points)
    offset = (int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1)))
    return set(_shift(points, offset))


def _piece(rng, sides, palette, anchor=False, forced=None, small=False):
    body, reg = palette[1:3]
    h, w = (int(rng.integers(4, 7)), int(rng.integers(4, 7))) if anchor else (
        int(rng.integers(2, 4 if small else 6)), int(rng.integers(2, 5 if small else 7)))
    if anchor:
        marks = {(h // 2, w // 2)}
        d = int(rng.integers(4))
        selected = [SIDES[d], SIDES[(d + 1) % 4]]
    else:
        marks = _pattern(rng, h, w, forced)
        n = int(rng.choice([1, 2, 3, 4], p=[.08, .57, .23, .12]))
        selected = list(rng.choice(SIDES, n, replace=False))
    labels, endpoints = {}, sorted(marks)
    for d in selected:
        length = w if d in ("top", "bottom") else h
        if anchor:
            positions = [w // 2 if d in ("top", "bottom") else h // 2]
        else:
            n = int(rng.integers(1, min(length, 3) + 1))
            positions = sorted(int(x) for x in rng.choice(length, n, replace=False))
        for i in positions:
            p, q = {"top": ((-1, i), (0, i)), "right": ((i, w), (i, w - 1)),
                    "bottom": ((h, i), (h - 1, i)), "left": ((i, -1), (i, 0))}[d]
            labels[p] = sides[d]
            endpoints.append(q)
    if anchor:
        core = {(r, c) for r in range(h) for c in range(w)}
    else:
        endpoints += [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                      (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
        rng.shuffle(endpoints)
        core = {min(marks)}
        for end in endpoints:
            distance = min(abs(p[0] - end[0]) + abs(p[1] - end[1]) for p in core)
            closest = sorted(p for p in core if abs(p[0] - end[0]) + abs(p[1] - end[1]) == distance)
            core.update(_path(closest[int(rng.integers(len(closest)))], end, rng))
        density = float(rng.uniform(.35, .98))
        for _ in range(h * w):
            if len(core) >= h * w * density:
                break
            frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in STEP
                               if 0 <= r + dr < h and 0 <= c + dc < w} - core)
            if not frontier:
                break
            core.add(frontier[int(rng.integers(len(frontier)))])
    return _transform({**{p: body for p in core}, **{p: reg for p in marks}, **labels}, 0)


def _scene(rng, palette, count, anchor=False, fallback=False):
    low = 18 if count == 1 else 23 if count == 2 else 26
    h, w = (30, 30) if fallback else (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
    layout = "frame" if anchor or fallback else str(rng.choice(["frame", "vertical", "horizontal"]))
    rh = [2, 2] if anchor or fallback else [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
    cw = [2, 2] if anchor or fallback else [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
    rows = [list(range(rh[0])), list(range(h - rh[1], h))]
    cols = [list(range(cw[0])), list(range(w - cw[1], w))]
    if layout != "frame":
        groups, extent, edge = (cols, w, cw) if layout == "vertical" else (rows, h, rh)
        thickness = int(rng.integers(1, 4))
        lo, hi = edge[0] + 7, extent - edge[1] - thickness - 7
        if lo <= hi:
            start = int(rng.integers(lo, hi + 1))
            groups.insert(1, list(range(start, start + thickness)))
    base = np.full((h, w), palette[0], dtype=np.int8)
    rc, cc = palette[3:3 + len(rows)], palette[3 + len(rows):3 + len(rows) + len(cols)]
    for g, color in zip(cols, cc):
        base[:, g] = color
    for g, color in zip(rows, rc):
        base[g, :] = color
    mode = 2 if anchor else int(rng.integers(4))
    if mode == 0:
        bits = np.zeros((len(rows), len(cols)), dtype=int)
    elif mode == 1:
        bits = np.ones((len(rows), len(cols)), dtype=int)
    elif mode == 2:
        values = rng.integers(0, 2, len(cols))
        values[0], values[-1] = 0, 1
        bits = np.tile(values, (len(rows), 1))
    else:
        values = rng.integers(0, 2, len(rows))
        bits = np.tile(values[:, None], (1, len(cols)))
    for i, rr in enumerate(rows):
        for j, cs in enumerate(cols):
            base[np.ix_(rr, cs)] = cc[j] if bits[i, j] else rc[i]
    mask, _, _ = _structure(base, palette[0])
    rectangles, _, sides = _regions(base, mask)
    return base, rectangles, sides


def _origins(cells, rectangles, margin=0):
    h, w = 1 + max(r for r, c in cells), 1 + max(c for r, c in cells)
    return [(r, c) for r0, r1, c0, c1 in rectangles
            for r in range(r0 + margin, r1 - h - margin + 1)
            for c in range(c0 + margin, c1 - w - margin + 1)]


def _construct(rng, palette, count, teaching=None):
    anchor = teaching == "rotate"
    base, rectangles, sides = _scene(rng, palette, count, anchor)
    finals, blocked = [], set()
    for i in range(count):
        for _ in range(24):
            region = int(rng.integers(len(rectangles)))
            forced = ("orthogonal" if i == 0 else "diagonal") if teaching == "reflect" else None
            piece = _piece(rng, sides[region], palette, anchor=anchor, forced=forced, small=count >= 3)
            origins = _origins(piece, [rectangles[region]], margin=int(anchor))
            rng.shuffle(origins)
            for origin in origins:
                placed = _shift(piece, origin)
                if not placed.keys() & blocked:
                    finals.append(placed)
                    blocked.update(_halo(placed))
                    break
            else:
                continue
            break
        else:
            return None
    targets = {p: v for cells in finals for p, v in cells.items() if v == palette[2]}
    blocked = _halo(targets)
    sources = []
    final_union = set().union(*(set(cells) for cells in finals))
    prefer_overlap = bool(rng.integers(2))
    for i, final in enumerate(finals):
        transforms = list(int(t) for t in rng.permutation(8))
        if i == 0 and teaching:
            transforms = [t for t in transforms if (0 < t < 4 if anchor else t >= 4)]
        for t in transforms:
            cells = _transform(final, t)
            if teaching == "reflect" and i == 0 and any(
                    _signature(_transform(cells, k)) == _signature(final) for k in range(4)):
                continue
            origins = _origins(cells, rectangles)
            rng.shuffle(origins)
            viable = []
            for origin in origins:
                placed = _shift(cells, origin)
                if placed.keys() & blocked:
                    continue
                viable.append(placed)
                if bool(placed.keys() & final_union) == prefer_overlap:
                    break
            if viable:
                placed = viable[-1]
                sources.append(placed)
                blocked.update(_halo(placed))
                break
        else:
            return None
    inp, docked = base.copy(), base.copy()
    for p, v in targets.items():
        inp[p] = v
    for source in sources:
        for p, v in source.items():
            inp[p] = v
    for final in finals:
        for p, v in final.items():
            docked[p] = v
    try:
        actual, info = _dock(inp)
        if not np.array_equal(actual, docked):
            return None
        pair = {"input": inp, "output": _collapse(docked)}
        metrics = _inspect(pair)
        if anchor and not (metrics["compass_needed"] and metrics["body_row_run"]
                           and metrics["body_col_run"] and metrics["separated_rows"]
                           and metrics["separated_cols"] and metrics["new_equalities"]):
            return None
        if teaching == "reflect" and not metrics["local_repeat_blocked"]:
            return None
        return pair
    except (ValueError, AssertionError):
        return None


def _fallback(rng, palette, count, teaching):
    """Finite, spacious fallback under the same rule, never a solver exception."""
    base, _, sides = _scene(rng, palette, count, teaching == "rotate", fallback=True)
    finals, sources = [], []
    patterns = [{(1, 1)}, {(1, 1), (1, 2)}, {(1, 1), (2, 1), (2, 2)},
                {(1, 1), (1, 2), (2, 1), (2, 2)}]
    if teaching == "reflect":
        patterns[:2] = [{(1, 1), (1, 2)}, {(1, 1), (2, 2)}]
    slots = [(4, 4), (5 if teaching == "reflect" else 4, 17), (12, 4), (12, 17)]
    for i in range(count):
        if teaching == "rotate":
            cells = _piece(rng, sides[0], palette, anchor=True)
        else:
            ch, cw = (4, 5) if teaching == "reflect" else (3, 4)
            cells = {(r, c): palette[1] for r in range(ch) for c in range(cw)}
            cells.update({p: palette[2] for p in patterns[i]})
            cells.update({(-1, 1): sides[0]["top"], (1, -1): sides[0]["left"]})
            cells = _transform(cells, 0)
        finals.append(_shift(cells, slots[i]))
        t = int(rng.integers(1, 4) if teaching == "rotate" else rng.integers(4, 8))
        sources.append(_shift(_transform(cells, t), (20, 2 + 7 * i)))
    inp, docked = base.copy(), base.copy()
    for cells in finals:
        for p, v in cells.items():
            docked[p] = v
            if v == palette[2]:
                inp[p] = v
    for cells in sources:
        for p, v in cells.items():
            inp[p] = v
    pair = {"input": inp, "output": _collapse(docked)}
    _inspect(pair)
    return pair


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    num_train, num_test = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    puzzle, used = {"train": [], "test": []}, set()
    shared_roles = bool(rng.integers(2))
    roles = [int(v) for v in rng.choice(10, 2, replace=False)]
    previous_bg = None
    for split, n in (("train", num_train), ("test", num_test)):
        for index in range(n):
            colors = [int(v) for v in rng.permutation(10)]
            if shared_roles:
                colors = [c for c in colors if c not in roles]
                bg = next(c for c in colors if c != previous_bg)
                palette = [bg] + roles + [c for c in colors if c != bg]
            else:
                if colors[0] == previous_bg:
                    colors[0], colors[3] = colors[3], colors[0]
                palette = colors
            previous_bg = palette[0]
            teaching = ("rotate" if index == 0 else "reflect" if index == 1 else None) if split == "train" else None
            count = (1 if index == 0 else 2 if index == 1 else int(rng.integers(1, 3))) if split == "train" else int(rng.integers(3, 5))
            for _ in range(72):
                pair = _construct(rng, palette, count, teaching)
                if pair is not None and _key(pair["input"]) not in used:
                    break
            else:
                pair = _fallback(rng, palette, count, teaching)
                if _key(pair["input"]) in used:
                    # Background changes make successive pairs distinct; a new
                    # local palette also distinguishes nonconsecutive fallbacks.
                    for _ in range(100):
                        new_palette = [int(v) for v in rng.permutation(10)]
                        pair = _fallback(rng, new_palette, count, teaching)
                        if _key(pair["input"]) not in used:
                            break
                    else:
                        raise RuntimeError("could not obtain a distinct fallback")
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _separated(a, axis):
    vectors = a if axis == 0 else a.T
    return len({v.tobytes() for v in vectors}) < len(vectors)


def _inspect(pair):
    a, out = _grid(pair["input"]), _grid(pair["output"])
    docked, info = _dock(a)
    keep_r, keep_c = _keep(docked, 0), _keep(docked, 1)
    assert np.array_equal(docked[np.ix_(keep_r, keep_c)], out), "wrong dock-then-collapse output"
    assert out.shape != docked.shape, "run contraction must affect the answer"
    assert not np.array_equal(_collapse(a), out), "docking must affect the answer"
    assert np.all(_keep(out, 0)) and np.all(_keep(out, 1))
    ri, ci = np.flatnonzero(keep_r), np.flatnonzero(keep_c)
    rr, cr = np.diff(np.r_[ri, a.shape[0]]), np.diff(np.r_[ci, a.shape[1]])
    assert np.array_equal(np.repeat(np.repeat(out, rr, axis=0), cr, axis=1), docked)
    assert np.array_equal(_collapse(docked.T).T, out), "axis order must commute"
    final_maps = _maps(info)[0]
    expected_count = sum(len(s) for s in info["sources"])
    assert len(final_maps) == expected_count, "complete rigid pieces must be preserved before contraction"
    for target in info["targets"]:
        assert all(final_maps.get(p) == info["registration"] for p in target)
    assert np.array_equal(a[info["structural"]], docked[info["structural"]])
    body = info["body"]
    br = any(length > 1 and np.any(docked[r] == body) for r, length in zip(ri, rr))
    bc = any(length > 1 and np.any(docked[:, c] == body) for c, length in zip(ci, cr))
    no_compass = _maps(info, use_labels=False, limit=16)
    compass_needed = any(not np.array_equal(_collapse(_canvas(a, info, cells)), out) for cells in no_compass)
    new_equalities = sum(int(np.sum(_keep(a, axis) & ~_keep(docked, axis))) for axis in (0, 1))
    source_cells = set().union(*(set(s) for s in info["sources"]))
    geometries = []
    moves = []
    final_objects = _components((docked != info["background"]) & ~info["structural"])
    final_objects = [{p: int(docked[p]) for p in obj} for obj in final_objects]
    local_repeat_blocked = False
    for final in final_objects:
        r0, r1 = min(r for r, c in final), 1 + max(r for r, c in final)
        c0, c1 = min(c for r, c in final), 1 + max(c for r, c in final)
        local = np.full((r1 - r0, c1 - c0), info["background"], dtype=np.int8)
        for (r, c), v in final.items():
            local[r - r0, c - c0] = v
        local_repeat_blocked |= bool(np.any(~_keep(local, 0) & keep_r[r0:r1]))
        local_repeat_blocked |= bool(np.any(~_keep(local, 1) & keep_c[c0:c1]))
    for source in info["sources"]:
        matches = [f for f in final_objects if _canonical(f) == _canonical(source)]
        assert matches, "rigid colored pattern lost"
        f = matches[0]
        transforms = [t for t in range(8) if _signature(_transform(source, t)) == _signature(f)]
        moves.append("translation" if 0 in transforms else "rotation" if any(t < 4 for t in transforms) else "reflection")
        marks = {p: 1 for p, v in source.items() if v == info["registration"]}
        geometries.append(_canonical(marks))
        mask = np.zeros((1 + max(r for r, c in _transform(source, 0)),
                         1 + max(c for r, c in _transform(source, 0))), dtype=bool)
        for p in _transform(source, 0):
            mask[p] = True
        assert len(_components(mask, diagonal=False)) == 1
    return dict(input_shape=a.shape, output_shape=out.shape, count=len(info["sources"]),
                background=info["background"], layout=(len(info["row_bands"]), len(info["col_bands"])),
                thickness=tuple(len(g) for g in info["row_bands"] + info["col_bands"]),
                compass_needed=compass_needed, body_row_run=br, body_col_run=bc,
                separated_rows=_separated(out, 0), separated_cols=_separated(out, 1),
                local_repeat_blocked=local_repeat_blocked,
                new_equalities=new_equalities, overlap=len(source_cells & final_maps.keys()),
                geometries=geometries, moves=moves, both_shrink=out.shape[0] < a.shape[0] and out.shape[1] < a.shape[1])


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, train, test = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            key = _key(_grid(pair["input"]))
            assert key not in seen
            seen.add(key)
            metrics = _inspect(pair)
            assert 1 <= metrics["count"] <= 4
            (train if split == "train" else test).append(metrics)
    assert any(m["compass_needed"] and m["body_row_run"] and m["body_col_run"]
               and m["separated_rows"] and m["separated_cols"] and m["both_shrink"]
               and m["new_equalities"] for m in train)
    assert any("rotation" in m["moves"] for m in train)
    assert any("reflection" in m["moves"] for m in train)
    assert any(m["local_repeat_blocked"] for m in train)
    assert any(any(len(x) == len(y) and x != y for i, x in enumerate(m["geometries"])
                   for y in m["geometries"][i + 1:]) for m in train)
    assert min(m["count"] for m in test) > max(m["count"] for m in train)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 puzzles streamed in memory; no files, logs, or saved execution traces."""
    shapes, outputs, layouts, patterns, backgrounds, counts, digests = (set() for _ in range(7))
    totals = dict(pairs=0, compass_needed=0, overlap_pairs=0, body_run_pairs=0,
                  separated_repeat_pairs=0, new_equality_pairs=0, local_repeat_blocked_pairs=0)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _inspect(pair)
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                totals["pairs"] += 1
                totals["compass_needed"] += int(m["compass_needed"])
                totals["overlap_pairs"] += int(m["overlap"] > 0)
                totals["body_run_pairs"] += int(m["body_row_run"] or m["body_col_run"])
                totals["separated_repeat_pairs"] += int(m["separated_rows"] or m["separated_cols"])
                totals["new_equality_pairs"] += int(m["new_equalities"] > 0)
                totals["local_repeat_blocked_pairs"] += int(m["local_repeat_blocked"])
                shapes.add(m["input_shape"])
                outputs.add(m["output_shape"])
                layouts.add(m["layout"])
                backgrounds.add(m["background"])
                counts.add(m["count"])
                patterns.update(m["geometries"])
    assert len(digests) == 42 and len(shapes) >= 30 and len(outputs) >= 25
    assert backgrounds == set(range(10)) and counts == {1, 2, 3, 4}
    assert layouts >= {(2, 2), (2, 3), (3, 2)} and len(patterns) >= 15
    assert totals["compass_needed"] >= 42 and totals["body_run_pairs"] >= 42
    assert totals["overlap_pairs"] > 5 and totals["separated_repeat_pairs"] >= 42
    assert totals["local_repeat_blocked_pairs"] >= 42
    first = generate_puzzle(342278199, 6, 3)
    assert _digest(first) == _digest(generate_puzzle(342278199, 6, 3))
    for a, b in ((2, 1), (10, 4), (10, 1)):
        p = generate_puzzle(788324236, a, b)
        assert len(p["train"]) == a and len(p["test"]) == b
    fallback_rng = np.random.default_rng(123)
    fallback_palette = list(range(10))
    for count, teaching in ((1, "rotate"), (2, "reflect"), (3, None), (4, None)):
        p = _fallback(fallback_rng, fallback_palette, count, teaching)
        m = _inspect(p)
        assert m["count"] == count
        if teaching == "reflect":
            assert m["local_repeat_blocked"] and "reflection" in m["moves"]
        if teaching == "rotate":
            assert m["compass_needed"] and m["body_row_run"] and m["body_col_run"]
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal counts")
    # Literal equality, including values at crossings, separated recurrences,
    # all-identical axes, and a single-cell grid, has no palette exception.
    for a in (np.array([[1, 1, 2, 1], [1, 1, 2, 1], [3, 3, 4, 3], [1, 1, 2, 1]]),
              np.full((4, 5), 7), np.array([[9]])):
        c = _collapse(a)
        assert np.array_equal(c, _collapse(c))
        assert np.array_equal(c, _collapse(a.T).T)
    assert np.array_equal(_collapse(np.array([[1, 1, 2, 1], [1, 1, 2, 1], [3, 3, 4, 3], [1, 1, 2, 1]])),
                          np.array([[1, 2, 1], [3, 4, 3], [1, 2, 1]]))
    return dict(seeds=42, **totals, distinct_input_shapes=len(shapes), distinct_output_shapes=len(outputs),
                layouts=sorted(layouts), registration_geometries=len(patterns),
                backgrounds=sorted(backgrounds), piece_counts=sorted(counts),
                independent_solver=False)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
