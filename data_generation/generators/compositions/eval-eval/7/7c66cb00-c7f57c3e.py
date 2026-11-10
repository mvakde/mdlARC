"""Counterstate casting: retain molds and route the opposite ornaments.

No files, global RNG, parent imports, or stored examples are used by this module.
Construction labels use known logical masks; solve rediscovers them from pixels.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                 ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size")
    _need(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _need(bool(np.all((a >= 0) & (a <= 9))), "color range")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _bounds(points):
    return (min(r for r, c in points), max(r for r, c in points),
            min(c for r, c in points), max(c for r, c in points))


def _components(mapping, diagonal=False):
    """Monochromatic components of a coordinate -> color mapping."""
    left, result = set(mapping), []
    steps = EIGHT if diagonal else FOUR
    while left:
        start = min(left)
        left.remove(start)
        color, todo, cells = mapping[start], [start], set()
        while todo:
            p = todo.pop()
            cells.add(p)
            for dr, dc in steps:
                q = p[0] + dr, p[1] + dc
                if q in left and mapping[q] == color:
                    left.remove(q)
                    todo.append(q)
        result.append((int(color), cells))
    return result


def _connected(points):
    return len(_components({p: 1 for p in points}, True)) == 1


def _signature(mapping):
    return tuple(sorted((r, c, int(color)) for (r, c), color in mapping.items()))


def _primitive(crop):
    divisor = 0
    for array in (crop, crop.T):
        for row in array:
            cuts = [0] + (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist() + [len(row)]
            for lo, hi in zip(cuts, cuts[1:]):
                divisor = math.gcd(divisor, hi - lo)
    small = crop[::divisor, ::divisor]
    _need(np.array_equal(np.repeat(np.repeat(small, divisor, 0), divisor, 1), crop),
          "not uniform pixel scaling")
    return divisor, small


def _expand(mapping, anchor, scale):
    return {(anchor[0] + r * scale + dy, anchor[1] + c * scale + dx): color
            for (r, c), color in mapping.items()
            for dy in range(scale) for dx in range(scale)}


def _paint(grid, mapping, anchor, scale):
    for (r, c), color in mapping.items():
        y, x = anchor[0] + r * scale, anchor[1] + c * scale
        grid[y:y + scale, x:x + scale] = color


def _separated(a, b):
    ay, ax, ah, aw = a
    by, bx, bh, bw = b
    return ay + ah < by or by + bh < ay or ax + aw < bx or bx + bw < ax


def _read_molds(grid, stop, bg):
    source = {tuple(map(int, p)): 1 for p in np.argwhere(grid[:stop] != bg)}
    groups = _components(source, True)
    _need(2 <= len(groups) <= 8, "need separate mold occurrences")
    objects = []
    for _, cells in groups:
        r0, r1, c0, c1 = _bounds(cells)
        crop = grid[r0:r1 + 1, c0:c1 + 1]
        _need(np.count_nonzero(crop != bg) == len(cells), "interleaved motifs")
        scale, small = _primitive(crop)
        raw = {(int(r), int(c)): int(small[r, c])
               for r, c in np.argwhere(small != bg)}
        objects.append(dict(box=(r0, c0), scale=scale, raw=_signature(raw), cells=cells))
    signatures = sorted({o["raw"] for o in objects})
    _need(len(signatures) == 2, "exactly two local states required")
    _need({o["raw"] for o in objects if o["scale"] == 1} == set(signatures),
          "both states need unit references")
    maps = [{(r, c): color for r, c, color in sig} for sig in signatures]
    shifts = set()
    for color in set(maps[0].values()) & set(maps[1].values()):
        a = {p for p, v in maps[0].items() if v == color}
        b = {p for p, v in maps[1].items() if v == color}
        ar, _, ac, _ = _bounds(a)
        br, _, bc, _ = _bounds(b)
        delta = ar - br, ac - bc
        if a == {(r + delta[0], c + delta[1]) for r, c in b}:
            shifts.add(delta)
    candidates = []
    for dy, dx in sorted(shifts):
        shifted = {(r + dy, c + dx): v for (r, c), v in maps[1].items()}
        shared = {p: v for p, v in maps[0].items() if shifted.get(p) == v}
        if len(shared) >= 4:
            candidates.append((len(shared), (dy, dx), shifted, shared))
    _need(bool(candidates), "no common registration scaffold")
    best = max(item[0] for item in candidates)
    winners = [item for item in candidates if item[0] == best]
    _need(len(winners) == 1, "tied registration")
    _, delta, shifted, common = winners[0]
    oy, _, ox, _ = _bounds(common)
    templates = [{(r - oy, c - ox): v for (r, c), v in m.items()}
                 for m in (maps[0], shifted)]
    common = {(r - oy, c - ox): v for (r, c), v in common.items()}
    variants = [{p: v for p, v in m.items() if p not in common} for m in templates]
    _need(all(variants), "each state needs changing material")
    stable = set(common.values())
    _need(not (stable & set(variants[0].values()) | stable & set(variants[1].values())),
          "scaffold colors must be exclusive")
    _need(not set(variants[0].values()) & set(variants[1].values()),
          "the two ornament palettes must be disjoint")
    for variant in variants:
        _need(len(_components(variant)) == len(_components(variant, True)),
              "diagonal-only ornament contacts are outside this subfamily")
    union = set(templates[0]) | set(templates[1])
    y0, y1, x0, x1 = _bounds(union)
    occurrences, rectangles = [], []
    for obj in objects:
        state = signatures.index(obj["raw"])
        dy, dx = delta if state else (0, 0)
        s = obj["scale"]
        anchor = obj["box"][0] + (oy - dy) * s, obj["box"][1] + (ox - dx) * s
        rect = anchor[0] + y0 * s, anchor[1] + x0 * s, (y1-y0+1)*s, (x1-x0+1)*s
        y, x, h, w = rect
        _need(0 <= y and y + h <= stop and 0 <= x and x + w <= grid.shape[1],
              "counterpart union outside staging")
        _need(all(_separated(rect, other) for other in rectangles), "mold regions overlap")
        rectangles.append(rect)
        occurrences.append(dict(anchor=anchor, scale=s, state=state, box=obj["box"]))
    return dict(templates=templates, common=common, variants=variants,
                occurrences=occurrences, rectangles=rectangles)


def _casts(info, exchange=True, bbox=False):
    casts = []
    for index, obj in enumerate(info["occurrences"]):
        state = 1 - obj["state"] if exchange else obj["state"]
        anchor, s = obj["anchor"], obj["scale"]
        if bbox:
            y0, _, x0, _ = _bounds(info["templates"][state])
            anchor = obj["box"][0] - y0*s, obj["box"][1] - x0*s
        mapping = _expand(info["variants"][state], anchor, s)
        for color, cells in _components(mapping):
            casts.append(dict(color=color, cells=cells, occurrence=index))
    return casts


def _deposit(info, casts, grouped=False, rail=True):
    grid, bg, stop = info["canonical"], info["background"], info["stop"]
    out = grid.copy()
    out[:stop] = bg
    for obj in info["occurrences"]:
        _paint(out, info["common"], obj["anchor"], obj["scale"])
    if grouped:
        colors = sorted({cast["color"] for cast in casts})
        casts = [dict(color=color, cells=set().union(*(c["cells"] for c in casts
                       if c["color"] == color)), occurrence=-1) for color in colors]
    lookup = {b["route"]: b for b in info["bands"]}
    occupied, records = set(), []
    for cast in casts:
        band = lookup[cast["color"]]
        bottom = max(r for r, c in cast["cells"])
        target = {(band["bottom"] - bottom + r, c) for r, c in cast["cells"]}
        _need(all(band["top"] <= r <= band["bottom"] and band["left"] < c < band["right"]
                  for r, c in target), "cast does not fit its trough")
        _need(not target & occupied, "cast cells collide")
        occupied.update(target)
        for p in target:
            out[p] = band["rail"] if rail else cast["color"]
        records.append(dict(color=cast["color"], cells=cast["cells"], target=target,
                            band=info["bands"].index(band), occurrence=cast["occurrence"]))
    return out, records


def _canonical_candidates(grid, bg):
    h, w = grid.shape
    rows = []
    for row in grid:
        nz = np.flatnonzero(row != bg)
        if not len(nz):
            rows.append(None)
            continue
        left, right = int(nz[0]), int(nz[-1])
        if (right - left + 1 >= max(7, w - 4) and row[left] == row[right] and
                row[left] != row[left + 1] and row[left + 1] != bg and
                np.all(row[left + 1:right] == row[left + 1])):
            rows.append((left, right, int(row[left + 1]), int(row[left])))
        else:
            rows.append(False)
    for first in range(3, h):
        if not rows[first] or rows[first - 1] is not None or any(v is False for v in rows[first:]):
            continue
        bands, r = [], first
        while r < h:
            if rows[r] is None:
                r += 1
                continue
            top = r
            left, right, route, rail = rows[r]
            while r + 1 < h and rows[r + 1] == rows[top]:
                r += 1
            bands.append(dict(top=top, bottom=r, left=left, right=right, route=route, rail=rail))
            r += 1
        if not 2 <= len(bands) <= 4 or len({b["route"] for b in bands}) != len(bands):
            continue
        if any(b["top"] < a["bottom"] + 2 for a, b in zip(bands, bands[1:])):
            continue
        try:
            info = _read_molds(grid, first - 1, bg)
            _need({v for m in info["variants"] for v in m.values()} ==
                  {b["route"] for b in bands}, "troughs must match precisely the ornament colors")
            info.update(canonical=grid, background=bg, bands=bands, stop=first - 1)
            expected, casts = _deposit(info, _casts(info))
            info.update(expected=expected, casts=casts)
            yield info
        except (AssertionError, KeyError):
            continue


def _analyze(input_grid):
    grid = _grid(input_grid)
    # At least one near-side corner is clear; try corner colors, without using
    # a majority-background assumption (the troughs may occupy most of a grid).
    backgrounds = sorted({int(grid[0, 0]), int(grid[0, -1]),
                          int(grid[-1, 0]), int(grid[-1, -1])})
    candidates = []
    for turn in range(4):
        canonical = np.rot90(grid, turn)
        for bg in backgrounds:
            for info in _canonical_candidates(canonical, bg):
                info["turn"] = turn
                info["output"] = np.rot90(info["expected"], -turn).copy()
                candidates.append(info)
    _need(len(candidates) == 1, "input needs one unambiguous staging/trough interpretation")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the two states, scales, molds, material and trough map from input."""
    return _analyze(input_grid)["output"].copy()


def _choose(rng, items):
    items = sorted(items)
    return items[int(rng.integers(len(items)))]


def _patch(rng, core, side, max_depth=2):
    y0, y1, x0, x1 = _bounds(core)
    if side == 0:
        root = _choose(rng, {p for p in core if p[0] == y0})
        first = (y0 - 1, root[1])
        limit = (x0, x1)
    elif side == 1:
        root = _choose(rng, {p for p in core if p[0] == y1})
        first = (y1 + 1, root[1])
        limit = (x0, x1)
    elif side == 2:
        root = _choose(rng, {p for p in core if p[1] == x0})
        first = (root[0], x0 - 1)
        limit = (y0, y1)
    else:
        root = _choose(rng, {p for p in core if p[1] == x1})
        first = (root[0], x1 + 1)
        limit = (y0, y1)
    result = {first}
    for _ in range(int(rng.integers(1, 7))):
        candidates = {(r + dr, c + dc) for r, c in result for dr, dc in FOUR} - result
        def allowed(p):
            depth = (y0-p[0], p[0]-y1, x0-p[1], p[1]-x1)[side]
            return 1 <= depth <= max_depth and limit[0] <= p[1 if side < 2 else 0] <= limit[1]
        candidates = {p for p in candidates if allowed(p)}
        if candidates:
            result.add(_choose(rng, candidates))
    return result


def _design(rng, mode, reserve=False, max_width=5, max_height=7):
    h = 3 if reserve else int(rng.integers(3, min(5, max_height)+1))
    w = 3 if reserve or mode == "horizontal" else int(rng.integers(3, min(5, max_width)+1))
    if mode != "horizontal" and h+2 > max_height and w+2 > max_width:
        h = max_height-2
    cy, cx = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
    kind = 0 if reserve else int(rng.integers(4))
    if kind == 0:
        core = {(cy, c) for c in range(w)} | {(r, cx) for r in range(h)}
    elif kind == 1:
        core = {(r, c) for r in range(h) for c in range(w)
                if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 2:
        core = {(r, 0) for r in range(h)} | {(r, w - 1) for r in range(h)}
        core |= {(h - 1, c) for c in range(w)}
    else:
        core = {(cy, c) for c in range(w)} | {(r, cx) for r in range(h)}
        for _ in range(int(rng.integers(1, 4))):
            p = _choose(rng, core)
            trial = core - {p}
            if len(trial) >= 5 and _connected(trial) and _bounds(trial) == (0, h - 1, 0, w - 1):
                core = trial
    if not reserve:
        for _ in range(int(rng.integers(4))):
            frontier = {(r + dr, c + dc) for r, c in core for dr, dc in FOUR
                        if 0 <= r + dr < h and 0 <= c + dc < w} - core
            if frontier:
                core.add(_choose(rng, frontier))
    inside = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)} - core
    if mode == "combined" and not inside:
        removable = [p for p in sorted(core) if 0 < p[0] < h-1 and 0 < p[1] < w-1
                     and _connected(core - {p})]
        if removable:
            p = _choose(rng, removable)
            core.remove(p)
            inside.add(p)
    if mode == "horizontal":
        a, b = _patch(rng, core, 2, 1), _patch(rng, core, 3, 1)
    else:
        sides = ([0, 1] if h+2 <= max_height else []) + ([2, 3] if w+2 <= max_width else [])
        side = _choose(rng, sides)
        depth = min(2, (max_height-h)//2 if side < 2 else (max_width-w)//2)
        a = _patch(rng, core, side, depth)
        b = a.copy() if mode == "recolor" else _patch(rng, core, side ^ 1, depth)
        if mode in ("combined", "recolor") and inside:
            inner = {_choose(rng, inside)}
            for _ in range(int(rng.integers(3))):
                frontier = {(r+dr, c+dc) for r, c in inner for dr, dc in FOUR} & (inside-inner)
                if frontier:
                    inner.add(_choose(rng, frontier))
            a |= inner
            b |= inner
    if reserve:
        # Keep the finite reserve small without introducing a different rule.
        if mode == "horizontal":
            a, b = {(1, -1)}, {(0, 3)}
        else:
            core = {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)}
            a, b = {(-1, 0), (1, 1)}, {(1, 1), (3, 2)}
    common = {p: 1 for p in core}
    if not reserve and rng.random() < .5:
        points = list(sorted(core))
        rng.shuffle(points)
        for p in points[:int(rng.integers(1, min(3, len(core) - 3) + 1))]:
            common[p] = 2
    templates = []
    for index, support in enumerate((a, b)):
        colors = 1 if reserve or mode == "horizontal" else int(rng.choice([1, 1, 2]))
        colors = min(colors, len(support))
        points = sorted(support)
        rng.shuffle(points)
        template = dict(common)
        template.update({p: 3 + 2*index + j % colors for j, p in enumerate(points)})
        templates.append(template)
    union = set(templates[0]) | set(templates[1])
    y0, y1, x0, x1 = _bounds(union)
    if x1 - x0 + 1 > 5 or y1 - y0 + 1 > 7:
        return None
    for template in templates:
        variant = {p: v for p, v in template.items() if p not in common}
        if (not _connected(template) or len(_components(variant)) != len(_components(variant, True))):
            return None
        occupied = set()
        for color, cells in _components(variant):
            bottom = max(r for r, c in cells)
            target = {(color, r-bottom, c) for r, c in cells}
            if target & occupied:
                return None
            occupied.update(target)
        r0, r1, c0, c1 = _bounds(template)
        crop = np.zeros((r1-r0+1, c1-c0+1), dtype=np.int8)
        _paint(crop, template, (-r0, -c0), 1)
        if _primitive(crop)[0] != 1:
            return None
    return templates, common


def _logical_label(grid, templates, common, placements, bands, bg, stage):
    """Construction path: segment known logical designs before enlarging."""
    out = grid.copy()
    out[:stage] = bg
    lookup, occupied = {b["route"]: b for b in bands}, set()
    pieces = [_components({p: v for p, v in template.items() if p not in common})
              for template in templates]
    for anchor, scale, state in placements:
        _paint(out, common, anchor, scale)
        for color, mask in pieces[1-state]:
            band = lookup[color]
            logical_bottom = max(r for r, c in mask)
            for r, c in mask:
                top = band["bottom"] - (logical_bottom-r+1)*scale + 1
                left = anchor[1] + c*scale
                for dy in range(scale):
                    for dx in range(scale):
                        p = top + dy, left + dx
                        _need(band["top"] <= p[0] <= band["bottom"] and
                              band["left"] < p[1] < band["right"] and p not in occupied,
                              "construction cast collision/bounds")
                        occupied.add(p)
                        out[p] = band["rail"]
    return out


def _make_pair(rng, mode, scales, turn, reserve=False):
    max_width = min(5, (29-len(scales))//sum(scales))
    design = _design(rng, mode, reserve, max_width=max_width,
                     max_height=5 if max(scales) >= 3 else 7)
    if design is None:
        return None
    templates, common = design
    palette = list(map(int, rng.permutation(10)))
    bg = palette[0]
    templates = [{p: palette[v] for p, v in template.items()} for template in templates]
    common = {p: palette[v] for p, v in common.items()}
    union = set(templates[0]) | set(templates[1])
    y0, y1, x0, x1 = _bounds(union)
    uh, uw = y1-y0+1, x1-x0+1
    n = len(scales)
    base_width = uw*sum(scales) + n-1 + 2
    if base_width > 30:
        return None
    width = int(rng.integers(max(12, base_width), 31))
    states = [0, 1] + list(map(int, rng.integers(0, 2, n-2)))
    variants = [{p: v for p, v in template.items() if p not in common} for template in templates]
    depth = {v: 1 for m in variants for v in m.values()}
    for scale, state in zip(scales, states):
        # The first lesson makes routing current ornaments a feasible, wrong
        # answer too, rather than relying on a missing/too-shallow destination.
        for state_to_budget in ((0, 1) if mode == "horizontal" else (1-state,)):
            for color, cells in _components(variants[state_to_budget]):
                r0, r1, _, _ = _bounds(cells)
                depth[color] = max(depth[color], scale*(r1-r0+1))
    routes = list(sorted(depth))
    rng.shuffle(routes)
    gaps = [1 if reserve else int(rng.integers(1, 3)) for _ in routes]
    heights = [depth[color] + (0 if reserve else int(rng.integers(2))) for color in routes]
    required_stage = uh*max(scales)
    room = 30 - sum(gaps) - sum(heights)
    if required_stage > room:
        return None
    stage = int(rng.integers(required_stage, min(room, required_stage+4)+1))
    margin = int(rng.integers(0, min(2, room-stage)+1))
    grid = np.full((stage+sum(gaps)+sum(heights)+margin, width), bg, dtype=np.int8)
    bands, cursor = [], stage
    shared_rail = int(rng.choice([v for v in range(10) if v != bg]))
    for color, height, gap in zip(routes, heights, gaps):
        cursor += gap
        choices = [v for v in range(10) if v not in (bg, color)]
        rail = shared_rail if shared_rail in choices and rng.random() < .35 else int(rng.choice(choices))
        left, right = int(rng.integers(2)), width-1-int(rng.integers(2))
        band = dict(top=cursor, bottom=cursor+height-1, route=color, rail=rail, left=left, right=right)
        bands.append(band)
        grid[cursor:cursor+height, left:right+1] = color
        grid[cursor:cursor+height, left] = rail
        grid[cursor:cursor+height, right] = rail
        cursor += height
    # Begin with a guaranteed horizontal arrangement, then move its rectangles
    # freely. Real target cells, rather than target bounding boxes, must be free.
    left_limit = max(b["left"] for b in bands)+1
    right_limit = min(b["right"] for b in bands)-1
    total_width = uw*sum(scales) + n-1
    if total_width > right_limit-left_limit+1:
        return None
    spare = right_limit-left_limit+1-total_width
    order = list(map(int, rng.permutation(n)))
    rects, cursor = [None]*n, left_limit+int(rng.integers(spare+1))
    for index in order:
        s = scales[index]
        top = int(rng.integers(stage-uh*s+1))
        rects[index] = (top, cursor, uh*s, uw*s)
        cursor += uw*s+1

    def placements_for(rectangles):
        return [((r-y0*s, c-x0*s), s, state)
                for (r, c, h, w), s, state in zip(rectangles, scales, states)]

    blank = grid.copy()
    for _ in range(36 if not reserve else 8):
        index = int(rng.integers(n))
        _, _, rh, rw = rects[index]
        candidate = (int(rng.integers(stage-rh+1)),
                     int(rng.integers(left_limit, right_limit-rw+2)), rh, rw)
        if not all(_separated(candidate, other) for i, other in enumerate(rects) if i != index):
            continue
        previous, rects[index] = rects[index], candidate
        try:
            _logical_label(blank, templates, common, placements_for(rects), bands, bg, stage)
        except AssertionError:
            rects[index] = previous
    placements = placements_for(rects)
    for anchor, scale, state in placements:
        _paint(grid, templates[state], anchor, scale)
    out = _logical_label(grid, templates, common, placements, bands, bg, stage)
    pair = dict(input=np.rot90(grid, turn).copy(), output=np.rot90(out, turn).copy())
    try:
        info = _analyze(pair["input"])
        _need(np.array_equal(pair["output"], info["output"]), "construction/inference mismatch")
        if mode == "horizontal":
            wrong, _ = _deposit(info, _casts(info, exchange=False))
            _need(not np.array_equal(wrong, info["expected"]), "exchange lesson is ambiguous")
            wrong, _ = _deposit(info, _casts(info, bbox=True))
            _need(not np.array_equal(wrong, info["expected"]), "registration lesson is ambiguous")
        return pair
    except AssertionError:
        return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(2))
    directions = list(map(int, rng.permutation(4)))
    fixed_direction = bool(rng.random() < .4)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            if split == "train" and index == 0:
                mode, scales = "horizontal", [1, 1, 2]
            elif split == "train" and index == 1:
                mode, scales = "combined", [1, 1, 1, 1]
            elif split == "train":
                mode = _choose(rng, ["combined", "recolor", "relocate"])
                scales = [1, 1] + list(map(int, rng.integers(1, 3, int(rng.integers(1, 3)))))
            elif regime == 0:
                mode, scales = "combined", [1, 1, 3]
            else:
                mode, scales = _choose(rng, ["combined", "relocate"]), [1]*int(rng.integers(5, 7))
            # Vary directions already in the two minimum demonstrations. Tests
            # reuse one demonstrated direction, avoiding an unshown rotation rule.
            turn = directions[0 if fixed_direction else index % 2]
            pair = None
            for attempt in range(80):
                candidate = _make_pair(rng, mode, scales, turn, reserve=attempt >= 64)
                if candidate is not None and _key(candidate["input"]) not in seen:
                    pair = candidate
                    break
            _need(pair is not None, "bounded constructive reserve exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative_differs(info, kind):
    try:
        out, _ = _deposit(info, _casts(info, exchange=kind != "current", bbox=kind == "bbox"),
                          grouped=kind == "grouped", rail=kind != "no_rail")
    except AssertionError:
        return True  # Invalid placements also refute the alternative.
    return not np.array_equal(out, info["expected"])


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "dimensions must be preserved")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _analyze(inp)
            _need(np.array_equal(out, info["output"]), "wrong counterstate casts")
            _need(not np.array_equal(inp, out), "identity pair")
            # Scaffolds remain at precisely their original cells, even when
            # palettes reuse their color on a rail elsewhere in the grid.
            for obj in info["occurrences"]:
                for p, color in _expand(info["common"], obj["anchor"], obj["scale"]).items():
                    _need(info["canonical"][p] == color and info["expected"][p] == color,
                          "scaffold changed")
            _need({c["color"] for c in info["casts"]} == {b["route"] for b in info["bands"]},
                  "every trough receives material")
            _need(all(max(r for r, c in cast["target"]) == info["bands"][cast["band"]]["bottom"]
                      for cast in info["casts"]), "cast not independently flush")
            infos[split].append(info)
    for alternative in ("current", "grouped", "bbox", "no_rail"):
        _need(any(_alternative_differs(i, alternative) for i in infos["train"]),
              "training does not refute " + alternative)
    _need(any(max(o["scale"] for o in i["occurrences"]) >= 2 for i in infos["train"]),
          "training must demonstrate scale transfer")
    train_directions = {i["turn"] for i in infos["train"]}
    _need(all(i["turn"] in train_directions for i in infos["test"]), "unshown direction")
    _need(all(len(i["occurrences"]) > max(len(t["occurrences"]) for t in infos["train"]) or
              max(o["scale"] for o in i["occurrences"]) >
              max(o["scale"] for t in infos["train"] for o in t["occurrences"])
              for i in infos["test"]), "missing count or scale train/test shift")


def _features(pair):
    info = _analyze(pair["input"])
    shapes = []
    for cast in info["casts"]:
        r0, _, c0, _ = _bounds(cast["cells"])
        shapes.append(tuple(sorted((r-r0, c-c0) for r, c in cast["cells"])))
    a, b = info["templates"]
    return dict(shape=tuple(np.asarray(pair["input"]).shape), direction=info["turn"],
                background=info["background"], molds=len(info["occurrences"]),
                scales=tuple(sorted(o["scale"] for o in info["occurrences"])),
                bands=len(info["bands"]), depths=tuple(b["bottom"]-b["top"]+1 for b in info["bands"]),
                scaffold=tuple(sorted(info["common"])), pieces=tuple(shapes),
                recolor=any(a[p] != b[p] for p in set(a) & set(b)),
                relocate=set(a) != set(b), bbox_shift=_bounds(a) != _bounds(b),
                repeated_rail=len({b["rail"] for b in info["bands"]}) < len(info["bands"]),
                reuse=any(b["rail"] in {x["route"] for x in info["bands"]} for b in info["bands"]),
                inset=any(b["left"] or b["right"] != info["canonical"].shape[1]-1 for b in info["bands"]))


def self_test():
    """Bounded, in-memory suite: 42 seeds, all count combinations, mutations."""
    signatures, shapes, scaffolds, pieces = set(), set(), set(), set()
    directions, backgrounds, counts, band_counts, scales, depths = set(), set(), set(), set(), set(), set()
    flags = {k: False for k in ("recolor", "relocate", "bbox_shift", "repeated_rail", "reuse", "inset")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        again = generate_puzzle(seed)
        fingerprint = tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s]
                            for k in ("input", "output"))
        _need(fingerprint == tuple(_key(p[k]) for s in ("train", "test") for p in again[s]
                                  for k in ("input", "output")), "not deterministic")
        signatures.add(hashlib.sha256(repr(fingerprint).encode()).hexdigest())
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _need(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                f = _features(pair)
                shapes.add(f["shape"])
                scaffolds.add(f["scaffold"])
                pieces.update(f["pieces"])
                directions.add(f["direction"])
                backgrounds.add(f["background"])
                counts.add(f["molds"])
                band_counts.add(f["bands"])
                scales.update(f["scales"])
                depths.add(f["depths"])
                for name in flags:
                    flags[name] |= f[name]
    _need(len(signatures) == 42, "duplicate puzzles across seeds")
    _need(len(shapes) >= 30 and len(scaffolds) >= 25 and len(pieces) >= 20, "narrow geometry coverage")
    _need(directions == set(range(4)) and backgrounds == set(range(10)), "missing directions/backgrounds")
    _need(counts >= {3, 4, 5, 6} and scales == {1, 2, 3}, "narrow count/scale coverage")
    _need(band_counts >= {2, 3, 4} and len(depths) >= 25 and all(flags.values()), "narrow trough/material coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(9000+a*10+b, a, b)
            _need(len(p["train"]) == a and len(p["test"]) == b, "count ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (3.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Exercise reserve construction explicitly, independently of random failures.
    rng = np.random.default_rng(731)
    for mode, schedule in (("horizontal", [1, 1, 2]), ("combined", [1, 1, 3]),
                           ("combined", [1]*6)):
        found = False
        for _ in range(16):
            pair = _make_pair(rng, mode, schedule, 0, reserve=True)
            if pair is not None:
                _need(np.array_equal(solve(pair["input"]), pair["output"]), "reserve mismatch")
                found = True
                break
        _need(found, "reserve failed")
    # Hand-calculated regression, not labeled by _logical_label or _deposit.
    # The two same-colored points are separated vertically, then both reach
    # the far edge. The last occurrence magnifies that counterpart at scale 2.
    inp = np.zeros((19, 21), dtype=np.int8)
    inp[12:14] = 3
    inp[12:14, [0, 20]] = 7
    inp[15:17] = 5
    inp[15:17, [0, 20]] = 8
    common = {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)}
    anchors = ((2, 2, 1, 0), (2, 8, 1, 1), (3, 13, 2, 0))
    for y, x, s, state in anchors:
        mapping = {p: 1 for p in common}
        mapping.update({p: 3 if state == 0 else 5 for p in
                        ({(-1, 0), (1, 1)} if state == 0 else {(1, 1), (3, 2)})})
        for (r, c), color in mapping.items():
            inp[y+r*s:y+(r+1)*s, x+c*s:x+(c+1)*s] = color
    expected = inp.copy()
    expected[:11] = 0
    for y, x, s, state in anchors:
        for r, c in common:
            expected[y+r*s:y+(r+1)*s, x+c*s:x+(c+1)*s] = 1
    expected[13, 8:10] = 7
    expected[16, 3:5] = 8
    expected[15:17, 15:19] = 8
    _need(np.array_equal(solve(inp), expected), "hand-calculated cast regression")
    missing, collision = inp.copy(), inp.copy()
    missing[3, 9] = missing[5, 10] = 0
    collision[5, 10], collision[5, 9] = 0, 5
    for malformed in (missing, collision):
        try:
            solve(malformed)
        except AssertionError:
            continue
        raise AssertionError("missing state or colliding counterstate cast accepted")
    puzzle = generate_puzzle(999)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    output = bad["test"][0]["output"]
    output[0, 0] = (int(output[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                scaffold_masks=len(scaffolds), cast_masks=len(pieces), directions=sorted(directions),
                backgrounds=sorted(backgrounds), mold_counts=sorted(counts),
                band_counts=sorted(band_counts), scales=sorted(scales),
                depth_sequences=len(depths), observed=flags)


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
