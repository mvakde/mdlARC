"""Marked payloads activated by finite diagonal gates and cast into color molds.

Only NumPy and the standard library are used. No parent generator is imported.
The parser, movement rule, and all output-affecting conventions are input-only.
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np

PRIMARY = 5
ANCHOR = 9
FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _need(np.all((a >= 0) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(points, colors, diagonal=False):
    unseen = set(points)
    result = []
    neighbors = EIGHT if diagonal else FOUR
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        color = colors[p]
        stack, cells = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in unseen and colors[q] == color:
                    unseen.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append((int(color), cells))
    return result


def _bounds(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _rectangle(box, halo=0):
    t, b, l, r = box
    return {(y, x) for y in range(t-halo, b+halo+1)
            for x in range(l-halo, r+halo+1)}


def _border(box):
    t, b, l, r = box
    return {(y, x) for y, x in _rectangle(box) if y in (t, b) or x in (l, r)}


def _segment(a, b):
    _need(a[0] < b[0] and abs(b[0]-a[0]) == abs(b[1]-a[1]), "not a diagonal")
    slope = 1 if b[1] > a[1] else -1
    return tuple((a[0]+k, a[1]+slope*k) for k in range(b[0]-a[0]+1)), slope


def _placement(cells, slope, box):
    """All integer translations are an interval; take its far end away from zero."""
    mr, xr, mc, xc = _bounds(cells)
    t, b, l, r = box
    lo, hi = t+1-mr, b-1-xr
    d = -slope
    if d == 1:
        lo, hi = max(lo, l+1-mc), min(hi, r-1-xc)
    else:
        lo, hi = max(lo, xc-(r-1)), min(hi, mc-(l+1))
    _need(lo <= hi and (lo > 0 or hi < 0), "payload has no one-sided chamber interval")
    shift = hi if lo > 0 else lo
    target = {(y+shift, x+d*shift) for y, x in cells}
    return shift, target, (lo, hi)


def _parse(value):
    g = _grid(value)
    h, w = g.shape
    bg = int(g[0, 0])
    _need(all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
          "all corners must show background")
    _need(bg not in (PRIMARY, ANCHOR), "reserved marker as background")
    colored = {tuple(map(int, p)): int(g[tuple(p)]) for p in np.argwhere(g != bg)
               if int(g[tuple(p)]) not in (PRIMARY, ANCHOR)}
    chambers = []
    for color, cells in _components(colored, colored):
        box = _bounds(cells)
        t, b, l, r = box
        if b-t < 3 or r-l < 3 or cells != _border(box):
            continue
        interior = g[t+1:b, l+1:r]
        route = int(interior[0, 0])
        if route not in (bg, PRIMARY, ANCHOR, color) and np.all(interior == route):
            chambers.append(dict(box=box, route=route, rail=color))
    _need(chambers, "no closed color chambers")
    chambers.sort(key=lambda b: b["box"])
    routes = {b["route"] for b in chambers}
    rails = {b["rail"] for b in chambers}
    _need(len(routes) == len(chambers) and not (routes & rails), "chamber color roles")
    claimed, halo = set(), set()
    for b in chambers:
        cells = _rectangle(b["box"])
        _need(not (cells & halo), "chambers need a background gap")
        claimed.update(cells)
        halo.update(_rectangle(b["box"], 1))
        t, bot, l, r = b["box"]
        _need(0 < t < bot < h-1 and 0 < l < r < w-1, "chamber outer margin")
    _need(all(g[p] == bg for p in halo-claimed if 0 <= p[0] < h and 0 <= p[1] < w),
          "chambers must be separate from markers and payloads")
    points = {tuple(map(int, p)) for p in np.argwhere(g != bg)} - claimed
    primary = sorted(p for p in points if g[p] == PRIMARY)
    anchors = {p for p in points if g[p] == ANCHOR}
    source = {p for p in points if g[p] != PRIMARY}
    _need(source and anchors, "no marked payloads")
    source_colors = {p: int(g[p]) for p in source}
    _need(set(source_colors.values()) <= routes | {ANCHOR}, "unmapped source color")
    for p in sorted(anchors):
        neighbors = {int(g[q]) for dr, dc in FOUR
                     if (q := (p[0]+dr, p[1]+dc)) in source and g[q] != ANCHOR}
        _need(len(neighbors) == 1 and neighbors <= routes, "anchor has no unique payload color")
        _need(not any((p[0]+dr, p[1]+dc) in anchors for dr, dc in EIGHT), "touching anchors")
        source_colors[p] = neighbors.pop()
    payloads = []
    four = _components(source, source_colors)
    eight = _components(source, source_colors, True)
    _need({frozenset(c) for _, c in four} == {frozenset(c) for _, c in eight},
          "diagonal-only same-color connection")
    for color, cells in four:
        marks = cells & anchors
        _need(len(marks) == 1 and len(cells) >= 2, "each payload needs one anchor and a body")
        payloads.append(dict(color=color, cells=cells, anchor=next(iter(marks))))
    _need(len(primary) >= 2, "gate endpoints missing")
    degrees = {p: 0 for p in primary}
    gates = []
    for a, b in combinations(primary, 2):
        if abs(a[0]-b[0]) == abs(a[1]-b[1]) and a[0] != b[0]:
            segment, slope = _segment(a, b)
            _need(len(segment) >= 4, "gate too short")
            gates.append(dict(segment=segment, slope=slope))
            degrees[a] += 1
            degrees[b] += 1
    _need(gates and max(degrees.values()) <= 1, "endpoint partners must be unique")
    lookup = {b["route"]: b for b in chambers}
    occupied = set()
    for obj in payloads:
        a = obj["anchor"]
        containing = [gate for gate in gates if a in gate["segment"][1:-1]]
        _need(len(containing) <= 1, "anchor belongs to two gates")
        obj["gate"] = containing[0] if containing else None
        obj["chamber"] = lookup[obj["color"]]
        if containing:
            shift, target, interval = _placement(obj["cells"], containing[0]["slope"],
                                                  obj["chamber"]["box"])
            _need(not (target & occupied), "translated payload collision")
            occupied.update(target)
            obj.update(shift=shift, target=target, interval=interval)
    _need(occupied, "at least one payload must be activated")
    return dict(grid=g, background=bg, chambers=chambers, payloads=payloads,
                gates=gates, primary=primary, unpaired={p for p in primary if not degrees[p]},
                source=source)


def _render(a, objects=None):
    out = a["grid"].copy()
    objects = [x for x in a["payloads"] if x["gate"]] if objects is None else objects
    for obj in objects:
        for p in obj["cells"]:
            out[p] = a["background"]
    for obj in objects:
        for p in obj["target"]:
            out[p] = obj["chamber"]["rail"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input's endpoints, anchors, masks, and visible color chambers."""
    return _render(_parse(input_grid))


def _signature(cells):
    t, b, l, r = _bounds(cells)
    return tuple(sorted((y-t, x-l) for y, x in cells))


def _mask(rng, large=False, core=False):
    if core:
        cells = {(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)}
        # Different outlines while keeping the anchor surrounded by its own color.
        for _ in range(int(rng.integers(5))):
            pool = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in FOUR
                           if -2 <= r+dr <= 2 and -2 <= c+dc <= 2} - cells)
            cells.add(pool[int(rng.integers(len(pool)))])
        return cells
    height = int(rng.integers(1, 6 if large else 5))
    width = int(rng.integers(2, 7 if large else 6))
    kind = int(rng.integers(5))
    if kind == 0:
        cells = {(r, c) for r in range(height) for c in range(width)}
    elif kind == 1 and height >= 3 and width >= 3:
        cells = {(r, c) for r in range(height) for c in range(width)
                 if r in (0, height-1) or c in (0, width-1)}
    elif kind == 2:
        cells = {(r, 0) for r in range(height)} | {(height-1, c) for c in range(width)}
    else:
        cells = {(int(rng.integers(height)), int(rng.integers(width)))}
        goal = int(rng.integers(2, height*width+1))
        while len(cells) < goal:
            pool = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in FOUR
                           if 0 <= r+dr < height and 0 <= c+dc < width} - cells)
            cells.add(pool[int(rng.integers(len(pool)))])
    anchor = sorted(cells)[int(rng.integers(len(cells)))]
    return {(r-anchor[0], c-anchor[1]) for r, c in cells}


def _source_ok(cells, anchor, role, sources, endpoints, shape):
    h, w = shape
    if not all(1 <= r < h-1 and 1 <= c < w-1 for r, c in cells):
        return False
    if cells & set(endpoints):
        return False
    for obj in sources:
        if cells & obj["cells"]:
            return False
        if obj["role"] == role and any((r+dr, c+dc) in obj["cells"]
                                      for r, c in cells for dr, dc in EIGHT):
            return False
        if any((anchor[0]+dr, anchor[1]+dc) in obj["cells"] for dr, dc in FOUR):
            return False
        if any((obj["anchor"][0]+dr, obj["anchor"][1]+dc) in cells for dr, dc in FOUR):
            return False
    return True


def _sample_gate(rng, shape, slope, endpoints, minimum=7):
    h, w = shape
    steps = int(rng.integers(minimum, min(13, h-5, w-5)+1))
    # Reserve a margin for bodies and beyond-endpoint witnesses.
    r = int(rng.integers(2, h-steps-2))
    c = int(rng.integers(2, w-steps-2)) + (steps if slope == -1 else 0)
    a, b = (r, c), (r+steps, c+slope*steps)
    if any(abs(p[0]-q[0]) == abs(p[1]-q[1]) for p in (a, b) for q in endpoints):
        return None
    segment, _ = _segment(a, b)
    return dict(segment=segment, slope=slope)


def _make_chamber(rng, group, sources, endpoints, chambers, shape, far=False):
    """Sample one rectangle around a translated group, then independently settle masks."""
    h, w = shape
    blocked = set(endpoints) | set().union(*(x["cells"] for x in sources))
    occupied = set().union(*(_rectangle(x["box"], 1) for x in chambers)) if chambers else set()
    for _ in range(180):
        sign = int(rng.choice((-1, 1)))
        distance = int(rng.integers(12 if far else 5, 22 if far else 19))
        translated = set().union(*({(r+sign*distance, c-x["slope"]*sign*distance)
                                    for r, c in x["cells"]} for x in group))
        t, b, l, right = _bounds(translated)
        pads = list(map(int, rng.integers(0, 3, size=4)))
        # At least 2x2 interior. Frames have uniform one-cell thickness.
        b = max(b, t+1)
        right = max(right, l+1)
        box = (t-pads[0]-1, b+pads[1]+1, l-pads[2]-1, right+pads[3]+1)
        t, b, l, right = box
        if not (1 <= t < b < h-1 and 1 <= l < right < w-1):
            continue
        if _rectangle(box, 1) & blocked or _rectangle(box) & occupied:
            continue
        target_union, shifts = set(), []
        try:
            for obj in group:
                shift, target, interval = _placement(obj["cells"], obj["slope"], box)
                if obj["active"]:
                    if target_union & target:
                        raise ValueError("collision")
                    target_union.update(target)
                    shifts.append(shift)
        except ValueError:
            continue
        return dict(box=box, role=group[0]["role"]), shifts
    return None


def _near_wrong_chamber(rng, obj, matching, sources, endpoints, chambers, shape, role):
    """An unused closer chamber accepts the geometry but has the wrong route color."""
    far_shift, _, _ = _placement(obj["cells"], obj["slope"], matching["box"])
    blocked = set(endpoints) | set().union(*(x["cells"] for x in sources))
    occupied = set().union(*(_rectangle(x["box"], 1) for x in chambers))
    h, w = shape
    for distance in rng.permutation(np.arange(3, max(4, abs(far_shift)-3))):
        shift = int(distance) * (1 if far_shift > 0 else -1)
        cells = {(r+shift, c-obj["slope"]*shift) for r, c in obj["cells"]}
        t, b, l, right = _bounds(cells)
        box = (t-1, max(b, t+1)+1, l-1, max(right, l+1)+1)
        if not (1 <= box[0] < box[1] < h-1 and 1 <= box[2] < box[3] < w-1):
            continue
        if _rectangle(box, 1) & blocked or _rectangle(box) & occupied:
            continue
        candidate_shift, _, _ = _placement(obj["cells"], obj["slope"], box)
        if abs(candidate_shift) < abs(far_shift):
            return dict(box=box, role=role)
    return None


def _attempt(rng, lesson, slope, is_test, allow_body_only=True):
    if lesson == "independent":
        shape = (int(rng.integers(27, 31)), int(rng.integers(27, 31)))
    elif lesson == "contact":
        shape = (int(rng.integers(24, 31)), int(rng.integers(24, 31)))
    else:
        shape = (int(rng.integers(19, 31)), int(rng.integers(19, 31)))
    gates, endpoints, sources = [], [], []
    count = 2 if is_test else (1 if lesson else int(rng.choice((1, 1, 2))))
    for gi in range(count):
        s = slope if gi == 0 else int(rng.choice((-1, 1)))
        gate = _sample_gate(rng, shape, s, endpoints)
        if gate is None:
            return None
        gates.append(gate)
        endpoints.extend((gate["segment"][0], gate["segment"][-1]))
        n = len(gate["segment"])-1
        offsets = [n-5, n-2] if lesson == "contact" else (
            [2, min(n-2, int(rng.integers(5, n)))] if lesson == "independent" else
            sorted(map(int, rng.choice(np.arange(2, n-1), size=int(rng.choice((1, 1, 2))), replace=False))))
        for oi, k in enumerate(offsets):
            role = oi if lesson == "contact" else gi
            anchor = gate["segment"][k]
            mask = _mask(rng, is_test, core=lesson == "contact")
            # A variable body with a protected 3x3 core. Trim the facing
            # extension before adding an explicit one-cell contact bridge.
            if lesson == "contact":
                mask = {(r, c) for r, c in mask if r*s*c < 2 or r <= 0}
                if oi == 0:
                    mask.add((1, 2*s))
                else:
                    mask = {(r, c) for r, c in mask if not (r < 0 and s*c < -1)}
            cells = {(anchor[0]+r, anchor[1]+c) for r, c in mask}
            if not _source_ok(cells, anchor, role, sources, endpoints, shape):
                return None
            sources.append(dict(cells=cells, anchor=anchor, role=role, slope=s, active=True))
        if lesson == "contact":
            # Same infinite diagonal, beyond the lower endpoint; no activation.
            anchor = (gate["segment"][-1][0]+2, gate["segment"][-1][1]+2*s)
            mask = _mask(rng)
            cells = {(anchor[0]+r, anchor[1]+c) for r, c in mask}
            if not _source_ok(cells, anchor, 1, sources, endpoints, shape):
                return None
            sources.append(dict(cells=cells, anchor=anchor, role=1, slope=s, active=False))
    # A later endpoint may not land in an earlier payload.
    if set(endpoints) & set().union(*(x["cells"] for x in sources)):
        return None
    for obj in sources:
        memberships = [g for g in gates if obj["anchor"] in g["segment"][1:-1]]
        if len(memberships) != int(obj["active"]):
            return None
    chambers = []
    for role in sorted({x["role"] for x in sources}):
        group = [x for x in sources if x["role"] == role]
        result = _make_chamber(rng, group, sources, endpoints, chambers, shape,
                               far=lesson == "independent")
        if result is None:
            return None
        chamber, shifts = result
        if lesson == "independent" and len(set(shifts)) < 2:
            return None
        chambers.append(chamber)
    if lesson == "independent":
        wrong = _near_wrong_chamber(rng, sources[0], chambers[0], sources,
                                     endpoints, chambers, shape, role=1)
        if wrong is None:
            return None
        chambers.append(wrong)
    # Optional untriggered bodies away from every finite span. No hidden role.
    if not lesson:
        for _ in range(int(rng.integers(3))):
            for _ in range(30):
                anchor = (int(rng.integers(2, shape[0]-2)), int(rng.integers(2, shape[1]-2)))
                if any(anchor in g["segment"] for g in gates):
                    continue
                role = int(rng.integers(len(chambers)))
                mask = _mask(rng, is_test)
                cells = {(anchor[0]+r, anchor[1]+c) for r, c in mask}
                if not _source_ok(cells, anchor, role, sources, endpoints, shape):
                    continue
                if any(cells & _rectangle(b["box"], 1) for b in chambers):
                    continue
                sources.append(dict(cells=cells, anchor=anchor, role=role, slope=1, active=False))
                break
    # Extra chambers are genuinely unused; existing routes remain unique.
    if not lesson and len(chambers) < 3 and rng.random() < .3:
        blocked = set(endpoints) | set().union(*(x["cells"] for x in sources))
        for _ in range(30):
            depth, width = int(rng.integers(4, 8)), int(rng.integers(4, 8))
            t, l = int(rng.integers(1, shape[0]-depth)), int(rng.integers(1, shape[1]-width))
            box = (t, t+depth-1, l, l+width-1)
            if _rectangle(box, 1) & blocked or any(_rectangle(box) & _rectangle(b["box"], 1) for b in chambers):
                continue
            chambers.append(dict(box=box, role=len(chambers)))
            break
    # Unpaired endpoints have no accidental diagonal partner; counts may be zero.
    for _ in range(int(rng.choice((0, 0, 1, 2)))):
        for _ in range(30):
            p = (int(rng.integers(1, shape[0]-1)), int(rng.integers(1, shape[1]-1)))
            if any(abs(p[0]-q[0]) == abs(p[1]-q[1]) for q in endpoints):
                continue
            if any(p in x["cells"] for x in sources) or any(p in _rectangle(b["box"], 1) for b in chambers):
                continue
            endpoints.append(p)
            break
    palette = [int(c) for c in rng.permutation(10) if c not in (PRIMARY, ANCHOR)]
    bg = palette[0]
    routes = palette[1:1+len(chambers)]
    rail_choices = palette[1+len(chambers):]
    rails = [int(rng.choice(rail_choices)) for _ in chambers]
    grid = np.full(shape, bg, dtype=np.int8)
    for chamber in chambers:
        role, box = chamber["role"], chamber["box"]
        for p in _rectangle(box):
            grid[p] = routes[role]
        for p in _border(box):
            grid[p] = rails[role]
    for obj in sources:
        for p in obj["cells"]:
            grid[p] = routes[obj["role"]]
        grid[obj["anchor"]] = ANCHOR
    for p in endpoints:
        grid[p] = PRIMARY
    try:
        a = _parse(grid)
    except ValueError:
        return None
    features = _features(a)
    if features["body_only"] and not allow_body_only:
        return None
    if lesson == "contact" and not (features["contact"] and features["beyond_reachable"]
                                     and set(features["signs"]) == {-1, 1}):
        return None
    if lesson == "independent" and not (features["independent"] and features["near_wrong"]):
        return None
    # Construct labels from sampled sources and boxes, independently of parser
    # segmentation and active-marker discovery. Movement is the same defined rule.
    output = grid.copy()
    lookup = {b["role"]: b for b in chambers}
    for obj in sources:
        if obj["active"]:
            for p in obj["cells"]:
                output[p] = bg
    for obj in sources:
        if obj["active"]:
            _, target, _ = _placement(obj["cells"], obj["slope"], lookup[obj["role"]]["box"])
            for p in target:
                output[p] = rails[obj["role"]]
    if not np.array_equal(output, _render(a)):
        return None
    return {"input": grid, "output": output}


def _features(a):
    active = [x for x in a["payloads"] if x["gate"]]
    inactive = [x for x in a["payloads"] if not x["gate"]]
    contact = any(x["color"] != y["color"] and
                  any((r+dr, c+dc) in y["cells"] for r, c in x["cells"] for dr, dc in FOUR)
                  for x, y in combinations(a["payloads"], 2))
    independent = any(x["color"] == y["color"] and x["shift"] != y["shift"]
                      for x, y in combinations(active, 2))
    beyond_reachable = False
    for obj in inactive:
        anchor = obj["anchor"]
        for gate in a["gates"]:
            first = gate["segment"][0]
            slope = gate["slope"]
            if anchor[1]-first[1] == slope*(anchor[0]-first[0]):
                try:
                    _placement(obj["cells"], slope, obj["chamber"]["box"])
                    beyond_reachable = True
                except ValueError:
                    pass
    near_wrong = False
    for obj in active:
        for b in a["chambers"]:
            if b is obj["chamber"]:
                continue
            try:
                shift, _, _ = _placement(obj["cells"], obj["gate"]["slope"], b["box"])
                near_wrong |= shift*obj["shift"] > 0 and abs(shift) < abs(obj["shift"])
            except ValueError:
                pass
    return dict(shape=tuple(a["grid"].shape), chambers=len(a["chambers"]),
                depths=tuple((b["box"][1]-b["box"][0]-1, b["box"][3]-b["box"][2]-1)
                             for b in a["chambers"]), gates=len(a["gates"]),
                lengths=tuple(len(g["segment"]) for g in a["gates"]),
                slopes=tuple(g["slope"] for g in a["gates"]),
                active=len(active), inactive=len(inactive), contact=contact,
                independent=independent, beyond_reachable=beyond_reachable,
                near_wrong=near_wrong, background=a["background"],
                body_only=any(x["cells"] & set(g["segment"][1:-1])
                              for x in inactive for g in a["gates"]),
                shapes=tuple(_signature(x["cells"]) for x in a["payloads"]),
                signs=tuple(1 if x["shift"] > 0 else -1 for x in active),
                unpaired=len(a["unpaired"]),
                unused=len(a["chambers"])-len({x["color"] for x in active}))


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    _need(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(g):
    return tuple(g.shape), g.astype(np.int8).tobytes()


def _reserve(rng, lesson, slope, is_test, seen):
    """Finite constructive reserve, with over fourteen distinct visible inputs.

    These coordinates construct masks and frames, not stored input/output pairs.
    Normal generation samples free geometry; the reserve guarantees termination.
    """
    if lesson == "contact":
        shape = (28, 29)
        endpoints = [(2, 5), (11, 14)]
        sources = [
            (0, (6, 9), {(r, c) for r in range(5, 8) for c in range(8, 11)} | {(7, 11)}),
            (1, (9, 12), {(r, c) for r in range(8, 11) for c in range(11, 14)}),
            (1, (13, 16), {(13, 16), (13, 17)}),
        ]
        boxes = [(10, 15, 1, 6), (1, 9, 16, 24)]
    elif lesson == "independent":
        shape = (28, 27)
        endpoints = [(2, 14), (10, 22)]
        sources = [(0, (4, 16), {(4, 16), (4, 17)}),
                   (0, (8, 20), {(8, 20), (8, 21)})]
        boxes = [(15, 25, 3, 15), (8, 11, 10, 13)]
    else:
        shape = (30, 30)
        endpoints = [(2, 15), (8, 21), (14, 22), (19, 17), (1, 1)]
        sources = [(0, (4, 17), {(4, 17), (4, 18)}),
                   (1, (16, 20), {(16, 20), (17, 20)}),
                   (0, (10, 10), {(10, 10), (10, 11)})]
        boxes = [(13, 18, 2, 8), (21, 27, 23, 28)]
    palette = [int(c) for c in rng.permutation(10) if c not in (PRIMARY, ANCHOR)]
    for serial in range(32):
        colors = palette[serial % 8:] + palette[:serial % 8]
        bg, routes, rails = colors[0], colors[1:3], colors[3:5]
        grid = np.full(shape, bg, dtype=np.int8)
        for role, box in enumerate(boxes):
            for p in _rectangle(box):
                grid[p] = routes[role]
            for p in _border(box):
                grid[p] = rails[role]
        for role, anchor, cells in sources:
            # Two footprint variants add another structural reserve axis.
            if serial // 16 and lesson == "independent":
                cells = {anchor, (anchor[0]+1, anchor[1])}
            elif serial // 16 and lesson == "contact" and anchor == (13, 16):
                cells = {anchor, (anchor[0]+1, anchor[1])}
            elif serial // 16 and lesson is None and anchor == (10, 10):
                cells = {anchor, (anchor[0]+1, anchor[1])}
            for p in cells:
                grid[p] = routes[role]
            grid[anchor] = ANCHOR
        for p in endpoints:
            grid[p] = PRIMARY
        if slope == -1:
            grid = np.fliplr(grid).copy()
        if (serial // 8) % 2:
            grid = np.rot90(grid, 2).copy()
        if _key(grid) in seen:
            continue
        a = _parse(grid)
        f = _features(a)
        if lesson == "contact":
            assert f["contact"] and f["beyond_reachable"] and set(f["signs"]) == {-1, 1}
        if lesson == "independent":
            assert f["independent"] and f["near_wrong"]
        if is_test:
            assert f["gates"] == 2
        return {"input": grid, "output": _render(a)}
    raise AssertionError("reserve has more candidates than the maximum pair count")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    train, test = _counts(num_train, num_test)
    _need(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
          "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    slopes = list(map(int, rng.permutation((-1, 1))))
    lessons = ["contact", "independent"] + [None]*(train-2)
    rng.shuffle(lessons)
    teaching_slopes = {"contact": slopes[0], "independent": slopes[1]}
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", train), ("test", test)):
        allow_body_only = split == "train" or any(
            _features(_parse(p["input"]))["body_only"] for p in puzzle["train"])
        for i in range(count):
            lesson = lessons[i] if split == "train" else None
            slope = teaching_slopes[lesson] if lesson else int(rng.choice((-1, 1)))
            for _ in range(450):
                pair = _attempt(rng, lesson, slope, split == "test", allow_body_only)
                if pair is not None and _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                pair = _reserve(rng, lesson, slope, split == "test", seen)
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, test_features = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "dimensions must be preserved")
            _need(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _parse(inp)
            _need(np.array_equal(out, _render(a)), "wrong output")
            _need(not np.array_equal(inp, out), "identity output")
            changed = set()
            for obj in a["payloads"]:
                if not obj["gate"]:
                    _need(all(out[p] == inp[p] for p in obj["cells"]), "inactive payload altered")
                    continue
                changed.update(obj["cells"] | obj["target"])
                _need(all(out[p] == a["background"] for p in obj["cells"]), "source not cleared")
                _need(all(out[p] == obj["chamber"]["rail"] for p in obj["target"]), "stamp color")
                _need(len(obj["cells"]) == len(obj["target"]), "shape area")
                _need(_signature(obj["cells"]) == _signature(obj["target"]), "rigid translation")
                # Taking one more step away from the source must cross a far wall.
                extra = obj["shift"] + (1 if obj["shift"] > 0 else -1)
                t, b, l, r = obj["chamber"]["box"]
                _need(any(not (t < y+extra < b and l < x-obj["gate"]["slope"]*extra < r)
                          for y, x in obj["cells"]), "not far aligned")
            _need(all(out[p] == inp[p] for p in np.ndindex(inp.shape) if p not in changed),
                  "unrelated cells changed")
            if split == "train":
                training.append((a, _features(a)))
            else:
                test_features.append(_features(a))
    _need(any(f["contact"] for _, f in training), "missing multicolor splitting lesson")
    _need(any(f["independent"] for _, f in training), "missing independent settling lesson")
    _need(any(f["beyond_reachable"] for _, f in training), "missing finite-span lesson")
    _need(any(f["near_wrong"] for _, f in training), "missing color-versus-distance lesson")
    _need({obj["gate"]["slope"] for a, _ in training for obj in a["payloads"] if obj["gate"]}
          == {-1, 1}, "training must activate both gate slopes")
    _need({sign for _, f in training for sign in f["signs"]} == {-1, 1},
          "training must show both directions along a perpendicular")
    _need(not any(f["body_only"] for f in test_features) or
          any(f["body_only"] for _, f in training),
          "an inactive body crossing a gate must first be demonstrated")


def self_test():
    """Bounded structural diversity, semantic regressions, and API checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [_features(_parse(pair["input"])) for p in puzzles
               for split in ("train", "test") for pair in p[split]]
    assert len({tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s])
                for p in puzzles}) == 42
    assert len({r["shape"] for r in records}) >= 35
    assert len({shape for r in records for shape in r["shapes"]}) >= 60
    assert len({r["depths"] for r in records}) >= 60
    assert {r["background"] for r in records} == set(range(10)) - {PRIMARY, ANCHOR}
    assert {s for r in records for s in r["signs"]} == {-1, 1}
    assert {r["gates"] for r in records} == {1, 2}
    assert any(r["unpaired"] for r in records) and any(not r["unpaired"] for r in records)
    assert any(r["inactive"] >= 2 for r in records)
    assert any(r["chambers"] >= 3 for r in records)
    # Counterfactuals use actual returned inputs, not requested generation flags.
    for p in puzzles:
        teaching = [_parse(pair["input"]) for pair in p["train"]]
        infinite_diff, wrong_color_diff = False, False
        for a in teaching:
            truth = _render(a)
            active = [x for x in a["payloads"] if x["gate"]]
            for obj in a["payloads"]:
                if obj["gate"]:
                    # Moving just the switch discards the inherited shape mask.
                    altered = truth.copy()
                    body_pixel = next(q for q in obj["target"]
                                      if q != (obj["anchor"][0]+obj["shift"],
                                               obj["anchor"][1]-obj["gate"]["slope"]*obj["shift"]))
                    altered[body_pixel] = obj["color"]
                    assert not np.array_equal(altered, truth)
                    for b in a["chambers"]:
                        if b is obj["chamber"]:
                            continue
                        try:
                            shift, target, interval = _placement(obj["cells"], obj["gate"]["slope"], b["box"])
                        except ValueError:
                            continue
                        if shift*obj["shift"] > 0 and abs(shift) < abs(obj["shift"]):
                            replacement = dict(obj, target=target, chamber=b)
                            false_output = _render(a, [x for x in active if x is not obj] + [replacement])
                            wrong_color_diff |= not np.array_equal(truth, false_output)
                else:
                    for gate in a["gates"]:
                        first, anchor = gate["segment"][0], obj["anchor"]
                        if anchor[1]-first[1] != gate["slope"]*(anchor[0]-first[0]):
                            continue
                        try:
                            _, target, _ = _placement(obj["cells"], gate["slope"], obj["chamber"]["box"])
                        except ValueError:
                            continue
                        replacement = dict(obj, target=target)
                        infinite_diff |= not np.array_equal(truth, _render(a, active + [replacement]))
        assert infinite_diff and wrong_color_diff
    _coordinate_regressions()
    # Explicitly exercise the reserve in every teaching slope and count limit.
    for lesson in ("contact", "independent", None):
        for slope in (-1, 1):
            seen = set()
            reserve_rng = np.random.default_rng(200)
            for _ in range(14):
                pair = _reserve(reserve_rng, lesson, slope, lesson is None, seen)
                seen.add(_key(pair["input"]))
                assert np.array_equal(solve(pair["input"]), pair["output"])
            assert len(seen) == 14
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(10000+10*train+test, train, test)
            assert len(p["train"]) == train and len(p["test"]) == test
    for args in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(12345, 2, 1)
    q = generate_puzzle(12345, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x)
    before = p["test"][0]["input"].copy()
    solve(p["test"][0]["input"])
    assert np.array_equal(before, p["test"][0]["input"])
    p["test"][0]["output"][0, 0] = PRIMARY
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")


def _coordinate_regressions():
    """Frozen small calculations check whole-mask alignment and finite activation."""
    g = np.zeros((28, 26), dtype=np.int8)
    for p in _rectangle((13, 18, 3, 10)):
        g[p] = 1
    for p in _border((13, 18, 3, 10)):
        g[p] = 2
    for p in _rectangle((20, 25, 7, 15)):
        g[p] = 3
    for p in _border((20, 25, 7, 15)):
        g[p] = 4
    g[6, 13] = g[10, 17] = PRIMARY
    g[8, 15] = g[12, 19] = ANCHOR
    g[9, 15] = g[9, 16] = 1
    g[12, 20] = 3
    expected = g.copy()
    expected[8, 15] = expected[9, 15] = expected[9, 16] = 0
    expected[16, 7] = expected[17, 7] = expected[17, 8] = 2
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(g, turn)), np.rot90(expected, turn))
    # The off-span anchor could reach its chamber at t=11, but does not move.
    assert _placement({(12, 19), (12, 20)}, 1, (20, 25, 7, 15))[0] == 11
    # Column-boundary stopping differs from anchor-only and row-only settling.
    assert _placement({(8, 15), (9, 15), (9, 16)}, 1, (13, 20, 5, 12))[0] == 9
    # Same-color pieces settle independently: their displacements are 12 and 16.
    g = np.zeros((28, 26), dtype=np.int8)
    box = (15, 25, 3, 15)
    for p in _rectangle(box):
        g[p] = 1
    for p in _border(box):
        g[p] = 2
    g[2, 14] = g[10, 22] = PRIMARY
    g[4, 16] = g[8, 20] = ANCHOR
    g[4, 17] = g[8, 21] = 1
    expected = g.copy()
    expected[4, 16:18] = expected[8, 20:22] = 0
    expected[16, 4:6] = expected[24, 4:6] = 2
    assert np.array_equal(solve(g), expected)
    assert {_placement({(4, 16), (4, 17)}, 1, box)[0],
            _placement({(8, 20), (8, 21)}, 1, box)[0]} == {12, 16}
    # A third endpoint on the same diagonal makes partner assignment ambiguous.
    g[12, 24] = PRIMARY
    try:
        solve(g)
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous endpoint pairing accepted")


if __name__ == "__main__":
    self_test()
    print(json.dumps({"self_test": "passed", "seeds": 42}, separators=(",", ":")))
