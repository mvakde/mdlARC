"""Invisible color-coincidence stencils, followed by compass-constrained docking.

Only standard-library modules and NumPy are used. Generation constructs an
answer geometrically; solve independently extracts roles and placements from
the input pixels. No generation state or parent module is consulted by solve.
"""

from collections import Counter
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _transform(cells, t):
    result = {}
    for (r, c), value in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = -c, r
        result[(r, c)] = int(value)
    r0 = min(r for r, _ in result)
    c0 = min(c for _, c in result)
    return {(r-r0, c-c0): v for (r, c), v in result.items()}


def _shift(cells, origin):
    return {(r+origin[0], c+origin[1]): v for (r, c), v in cells.items()}


def _key(a):
    a = np.asarray(a)
    return a.shape, a.astype(np.int8).tobytes()


def _signature(cells):
    return tuple(sorted(_transform(cells, 0).items()))


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    neighbors = NEIGHBORS if diagonal else DIRS
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, component = [p], {p}
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = (r+dr, c+dc)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _halo(points):
    return {(r+dr, c+dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _check_grid(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _structure(grid, bg):
    """Recover straight full bands even where a perpendicular cap covers them."""
    h, w = grid.shape
    rows, cols = set(), set()
    for _ in range(3):
        active_cols = [c for c in range(w) if c not in cols]
        for r in range(h):
            vals = grid[r, active_cols]
            if len(vals) and vals[0] != bg and np.all(vals == vals[0]):
                rows.add(r)
        active_rows = [r for r in range(h) if r not in rows]
        for c in range(w):
            vals = grid[active_rows, c]
            if len(vals) and vals[0] != bg and np.all(vals == vals[0]):
                cols.add(c)
    assert rows or cols
    mask = np.zeros(grid.shape, dtype=bool)
    mask[list(rows), :] = True
    mask[:, list(cols)] = True
    regions = _components(~mask)
    assert 1 <= len(regions) <= 2
    index = np.full(grid.shape, -1, dtype=np.int8)
    sides, rectangles = [], []
    for i, region in enumerate(regions):
        r0, r1 = min(r for r, _ in region), max(r for r, _ in region)+1
        c0, c1 = min(c for _, c in region), max(c for _, c in region)+1
        assert len(region) == (r1-r0)*(c1-c0), "regions must be rectangular"
        rectangles.append((r0, r1, c0, c1))
        colors = {d: set() for d in DIRS}
        for r, c in region:
            index[r, c] = i
            for dr, dc in DIRS:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                    colors[(dr, dc)].add(int(grid[nr, nc]))
        assert all(len(v) <= 1 for v in colors.values())
        sides.append(colors)
    return mask, regions, index, sides, rectangles


def _extract_ruler(grid, component, bg):
    rs, cs = zip(*component)
    r0, c0, r1, c1 = min(rs), min(cs), max(rs)+1, max(cs)+1
    bh, bw = r1-r0, c1-c0
    assert min(bh, bw) in (2, 3) and 4 <= max(bh, bw) <= 8
    sizes = [(2, bw)] if bh < bw else [(bh, 2)]
    candidates = []
    for ah, aw in sizes:
        for r in range(r0, r1-ah+1):
            for c in range(c0, c1-aw+1):
                core = {(r+i, c+j) for i in range(ah) for j in range(aw)}
                if not core <= component:
                    continue
                array = grid[r:r+ah, c:c+aw].copy()
                structural = set(map(int, np.unique(array)))
                if len(structural) != 2 or bg in structural:
                    continue
                ports = component-core
                port_sides = set()
                valid = True
                for pr, pc in ports:
                    if int(grid[pr, pc]) in structural:
                        valid = False
                        break
                    if ah == 2 and r <= pr < r+ah:
                        valid = False
                        break
                    if aw == 2 and c <= pc < c+aw:
                        valid = False
                        break
                    inside = [p for p in core if abs(pr-p[0])+abs(pc-p[1]) == 1]
                    if len(inside) != 1:
                        valid = False
                        break
                    port_sides.add((pr-inside[0][0], pc-inside[0][1]))
                if valid and len(port_sides) <= 1:
                    candidates.append({"arr": array, "origin": (r, c),
                                       "core": core, "ports": ports,
                                       "component": component})
    assert len(candidates) == 1, "ruler core is ambiguous"
    return candidates[0]


def _edge(ruler, side, slot, length):
    labels = ruler["labels"]
    p = tuple(map(int, np.argwhere(labels == side*length+slot)[0]))
    q = tuple(map(int, np.argwhere(labels == (1-side)*length+slot)[0]))
    d = (p[0]-q[0], p[1]-q[1])
    o = ruler["origin"]
    return (o[0]+p[0], o[1]+p[1]), d


def _ray(ruler, side, slot, length, region_index, region):
    p, d = _edge(ruler, side, slot, length)
    r, c = p[0]+d[0], p[1]+d[1]
    h, w = region_index.shape
    points = set()
    while 0 <= r < h and 0 <= c < w and region_index[r, c] == region:
        points.add((r, c))
        r, c = r+d[0], c+d[1]
    return points


def _targets(info, codes=None, equality=True):
    """Infer virtual stencils by intersection of region-bounded ray sets."""
    targets = []
    for region, rulers in enumerate(info["rulers_by_region"]):
        assert len(rulers) == 2
        assert {min(e["arr"].shape) for e in rulers} == {2}
        assert {e["arr"].shape[0] == 2 for e in rulers} == {False, True}
        local_codes = [info["code"], info["code"]] if codes is None else codes[region]
        rays = [{s: _ray(e, info["side"], s, info["length"], info["region_index"], region)
                 for s in code} for e, code in zip(rulers, local_codes)]
        marks = set()
        for s, first in rays[0].items():
            for t, second in rays[1].items():
                if not equality or local_codes[0][s] == local_codes[1][t]:
                    marks.update(first & second)
        targets.append(marks)
    return targets


def _parse(grid):
    counts = np.bincount(grid.ravel(), minlength=10)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(np.argmax(counts))
    structural, regions, index, sides, rectangles = _structure(grid, bg)
    band_colors = set(map(int, np.unique(grid[structural])))
    components = _components((grid != bg) & ~structural, diagonal=True)
    sources, rulers = [], []
    for component in components:
        cells = {p: int(grid[p]) for p in component}
        if set(cells.values()) & band_colors:
            sources.append(cells)
        else:
            rulers.append(_extract_ruler(grid, component, bg))
    assert len(sources) == len(regions) and len(rulers) == 2*len(regions)
    canonical = rulers[0]["arr"]
    if canonical.shape[0] != 2:
        canonical = np.rot90(canonical)
    length = canonical.shape[1]
    forms = [_orient(canonical, t) for t in range(8)]
    assert len({_key(a) for a in forms}) == 8, "colored cores must be asymmetric"
    emitter_colors = set(map(int, np.unique(canonical)))
    labels = np.arange(2*length).reshape(2, length)
    code, active_sides = {}, set()
    by_region = [[] for _ in regions]
    observations = []
    for i, e in enumerate(rulers):
        matches = [t for t, a in enumerate(forms) if np.array_equal(e["arr"], a)]
        assert len(matches) == 1
        e["labels"] = _orient(labels, matches[0])
        e["transform"] = matches[0]
        e["observed"] = {}
        e["region"] = int(index[min(e["core"])])
        assert all(index[p] == e["region"] for p in e["component"])
        by_region[e["region"]].append(e)
        r0, c0 = e["origin"]
        for pr, pc in sorted(e["ports"]):
            adjacent = [p for p in e["core"] if abs(pr-p[0])+abs(pc-p[1]) == 1]
            assert len(adjacent) == 1
            ar, ac = adjacent[0]
            local_id = int(e["labels"][ar-r0, ac-c0])
            side, slot = divmod(local_id, length)
            edge, direction = _edge(e, side, slot, length)
            assert (pr, pc) == (edge[0]+direction[0], edge[1]+direction[1])
            color = int(grid[pr, pc])
            assert slot not in code or code[slot] == color, "conflicting ports"
            code[slot] = color
            active_sides.add(side)
            e["observed"][slot] = color
            observations.append((i, e["region"], slot, color))
    assert len(active_sides) == 1 and 2 <= len(code) <= 5
    assert len(set(code.values())) >= 2
    signal_colors = set(code.values())
    assert signal_colors.isdisjoint(emitter_colors | band_colors | {bg})
    body_roles, registration_roles = set(), set()
    for source in sources:
        values = set(source.values())
        body_roles.update(values & emitter_colors)
        registration_roles.update(values - emitter_colors - band_colors)
        assert len(values & emitter_colors) == 1 and len(values & band_colors) >= 2
    assert len(body_roles) == len(registration_roles) == 1
    body, registration = next(iter(body_roles)), next(iter(registration_roles))
    assert registration not in signal_colors | band_colors | emitter_colors | {bg}
    info = {"background": bg, "structural": structural, "band_colors": band_colors,
            "sources": sources, "rulers": rulers, "rulers_by_region": by_region,
            "regions": regions, "region_index": index, "sides": sides,
            "rectangles": rectangles, "body": body, "registration": registration,
            "emitter_colors": emitter_colors, "signal_colors": signal_colors,
            "length": length, "side": active_sides.pop(), "code": code,
            "observations": observations, "grid": grid}
    info["fixed"] = structural.copy()
    for e in rulers:
        for p in e["component"]:
            info["fixed"][p] = True
    info["targets"] = _targets(info)
    assert all(len(t) >= 2 for t in info["targets"])
    assert all(not info["fixed"][p] for t in info["targets"] for p in t)
    return info


def _label_directions(cells, bands):
    core = [p for p, v in cells.items() if v not in bands]
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    labels = {}
    for (r, c), value in cells.items():
        if value in bands:
            directions = [d for d, yes in (((-1, 0), r < r0), ((1, 0), r > r1),
                                           ((0, -1), c < c0), ((0, 1), c > c1)) if yes]
            if len(directions) != 1:
                return None
            labels[(r, c)] = directions[0]
    return labels


def _candidates(source, info, use_compass=True):
    candidates, seen = [], set()
    h, w = info["grid"].shape
    for t in range(8):
        cells = _transform(source, t)
        marks = {p for p, v in cells.items() if v == info["registration"]}
        labels = _label_directions(cells, info["band_colors"])
        assert labels
        for region, target in enumerate(info["targets"]):
            if len(marks) != len(target):
                continue
            first, anchor = min(marks), min(target)
            delta = (anchor[0]-first[0], anchor[1]-first[1])
            if {(r+delta[0], c+delta[1]) for r, c in marks} != target:
                continue
            placed = _shift(cells, delta)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in placed):
                continue
            if any(info["region_index"][p] != region or info["fixed"][p] for p in placed):
                continue
            if use_compass and any(cells[p] not in info["sides"][region][d]
                                   for p, d in labels.items()):
                continue
            signature = tuple(sorted(placed.items()))
            if signature not in seen:
                seen.add(signature)
                candidates.append((region, placed, t))
    return candidates


def _placements(info, use_compass=True, limit=2):
    choices = [_candidates(source, info, use_compass) for source in info["sources"]]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    answers = {}

    def search(depth, used, occupied, selected):
        if len(answers) >= limit:
            return
        if depth == len(order):
            signature = tuple(sorted(occupied.items()))
            answers[signature] = (dict(occupied), list(selected))
            return
        source_i = order[depth]
        for region, placed, t in choices[source_i]:
            if region in used or placed.keys() & occupied.keys():
                continue
            search(depth+1, used | {region}, {**occupied, **placed},
                   selected+[(source_i, region, placed, t)])
    search(0, set(), {}, [])
    return list(answers.values())


def _reconstruct(grid):
    info = _parse(grid)
    placements = _placements(info)
    assert len(placements) == 1, "input must determine one complete output"
    final, selected = placements[0]
    result = grid.copy()
    for source in info["sources"]:
        for p in source:
            result[p] = info["background"]
    for p, color in final.items():
        result[p] = color
    info["selected"] = selected
    return result, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the code, construct virtual stencils and dock from input alone."""
    grid = _check_grid(input_grid)
    try:
        return _reconstruct(grid)[0]
    except (AssertionError, IndexError, KeyError) as exc:
        raise ValueError("input violates the optical-stencil family") from exc


def _template(rng, length, colors):
    for _ in range(64):
        a = rng.choice(colors, size=(2, length)).astype(np.int8)
        if rng.random() < .5:
            a[0] = colors[0]
            a[1] = colors[1]
            a[1, :int(rng.integers(1, length))] = colors[0]
        if len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    a = np.full((2, length), colors[0], dtype=np.int8)
    a[1, 1:] = colors[1]
    return a


def _base(rng, count, length, palette, fallback=False):
    # Body deliberately shares a core color. Composite compass/registration
    # content, rather than a permanent literal palette, distinguishes sources.
    bg = palette[0]
    thick = 1 if count == 2 or fallback else int(rng.choice([1, 2]))
    full = count == 1 and not fallback and rng.random() < .4
    if fallback:
        h, w = (30, 30) if count == 2 else (27, 27)
    elif count == 2:
        h = int(rng.integers(24, 31))
        w = int(rng.integers(max(27, 2*(length+6)+3), 31))
    else:
        h = int(rng.integers(max(20, length+14), 31))
        w = int(rng.integers(max(18, length+10), 31))
    base = np.full((h, w), bg, dtype=np.int8)
    bands = palette[4:8]
    base[:thick, :] = bands[0]
    base[thick:, :thick] = bands[1]
    base[thick:, -thick:] = bands[2]
    if count == 2:
        low, high = length+6+thick, w-thick-(length+6)-1
        if low > high:
            return None
        divider = (w-1)//2 if fallback else int(rng.integers(low, high+1))
        base[thick:, divider:divider+1] = bands[3]
    elif full:
        base[-thick:, :] = bands[3]
    structural, regions, index, sides, rectangles = _structure(base, bg)
    if len(regions) != count:
        return None
    return base, structural, index, sides, rectangles


def _station(rng, template, side, rectangle, fallback=False):
    r0, r1, c0, c1 = rectangle
    length = template.shape[1]
    ids = np.arange(2*length).reshape(2, length)
    horizontal = int(rng.choice([0, 2, 4, 6]))
    vertical = int(rng.choice([1, 3, 5, 7]))
    if fallback:
        horizontal = 2 if side == 0 else 0
        vertical = 1 if side == 0 else 3
    temporary = []
    for t in (horizontal, vertical):
        e = {"arr": _orient(template, t), "labels": _orient(ids, t),
             "origin": (0, 0), "transform": t}
        temporary.append(e)
    _, hd = _edge(temporary[0], side, 0, length)
    _, vd = _edge(temporary[1], side, 0, length)
    rlo, rhi = (r0+5, r1-length-1) if hd == (1, 0) else (r0+1, r1-length-5)
    clo, chi = (c0+5, c1-length-1) if vd == (0, 1) else (c0+1, c1-length-5)
    if rlo > rhi or clo > chi:
        return None
    wr = rlo if fallback else int(rng.integers(rlo, rhi+1))
    wc = clo if fallback else int(rng.integers(clo, chi+1))
    hr = wr-5 if hd == (1, 0) else wr+length+3
    vc = wc-5 if vd == (0, 1) else wc+length+3
    temporary[0]["origin"] = (hr, wc)
    temporary[1]["origin"] = (wr, vc)
    for e in temporary:
        er, ec = e["origin"]
        e["cells"] = {(er+r, ec+c): int(e["arr"][r, c])
                      for r in range(e["arr"].shape[0]) for c in range(e["arr"].shape[1])}
    return temporary, (wr, wc, length)


def _path(rng, start, end):
    r, c = start
    path = {(r, c)}
    while (r, c) != end:
        if r != end[0] and (c == end[1] or rng.random() < .5):
            r += 1 if end[0] > r else -1
        else:
            c += 1 if end[1] > c else -1
        path.add((r, c))
    return path


def _piece(rng, marks, window, sides, labels, body, reg):
    wr, wc, length = window
    core = {min(marks)}
    endpoints = sorted(marks)
    # Body growth is independent of ray color classes, and preserves gaps.
    endpoints += [(wr+int(rng.integers(length)), wc+int(rng.integers(length)))
                  for _ in range(int(rng.integers(2, 6)))]
    rng.shuffle(endpoints)
    for p in endpoints:
        nearest = min(abs(r-p[0])+abs(c-p[1]) for r, c in core)
        anchors = sorted(q for q in core if abs(q[0]-p[0])+abs(q[1]-p[1]) == nearest)
        core.update(_path(rng, anchors[int(rng.integers(len(anchors)))], p))
    desired = max(len(marks)+4, int(length*length*float(rng.uniform(.30, .80))))
    while len(core) < min(desired, length*length):
        frontier = sorted({(r+dr, c+dc) for r, c in core for dr, dc in DIRS
                           if wr <= r+dr < wr+length and wc <= c+dc < wc+length} - core)
        if not frontier:
            break
        core.add(frontier[int(rng.integers(len(frontier)))])
    cells = {p: reg if p in marks else body for p in core}
    r0, r1 = min(r for r, _ in core), max(r for r, _ in core)
    c0, c1 = min(c for _, c in core), max(c for _, c in core)
    for d in labels:
        supports = sorted(p for p in core if
                          (d == (-1, 0) and p[0] == r0) or
                          (d == (1, 0) and p[0] == r1) or
                          (d == (0, -1) and p[1] == c0) or
                          (d == (0, 1) and p[1] == c1))
        rng.shuffle(supports)
        for r, c in supports[:int(rng.integers(1, min(2, len(supports))+1))]:
            cells[(r+d[0], c+d[1])] = next(iter(sides[d]))
    return cells


def _construct(rng, count, slot_count, demonstration=None, fallback=False):
    length = 5 if fallback else int(rng.integers(max(4, slot_count), 7 if count == 2 else 9))
    palette = list(map(int, rng.permutation(10)))
    bg, body, other, reg = palette[:4]
    made = _base(rng, count, length, palette, fallback)
    if made is None:
        return None
    base, structural, index, sides, rectangles = made
    template = _template(rng, length, [body, other])
    side = int(rng.integers(2))
    if fallback:
        side = 0
    slots = tuple(sorted(map(int, rng.choice(length, slot_count, replace=False))))
    if demonstration == "repeat":
        values = [palette[8], palette[9], palette[8]]
    else:
        values = [palette[8], palette[9]] + list(map(int, rng.choice(palette[8:10], slot_count-2)))
        rng.shuffle(values)
    code = dict(zip(slots, values))
    rulers, windows, marks_list = [], [], []
    for region, rectangle in enumerate(rectangles):
        station = _station(rng, template, side, rectangle, fallback)
        if station is None:
            return None
        es, window = station
        for e in es:
            e["region"] = region
            rulers.append(e)
        windows.append(window)
        # Construction computes crossing coordinates directly, rather than
        # invoking the input decoder's bounded ray-set intersection routine.
        marks = set()
        for s in code:
            hp, hd = _edge(es[0], side, s, length)
            for t in code:
                vp, vd = _edge(es[1], side, t, length)
                if code[s] == code[t]:
                    p = (vp[0], hp[1])
                    if (p[0]-hp[0])*hd[0] > 0 and (p[1]-vp[1])*vd[1] > 0:
                        marks.add(p)
        if len(marks) != sum(list(code.values()).count(v)**2 for v in set(code.values())):
            return None
        marks_list.append(marks)
    inp = base.copy()
    fixed_cells = set()
    for e in rulers:
        for p, value in e["cells"].items():
            inp[p] = value
            fixed_cells.add(p)
    # First demonstration requires information to travel between regions.
    if demonstration == "pool":
        allocation = [int(rng.integers(2)), 2+int(rng.integers(2))]
    elif demonstration == "repeat":
        allocation = [0, 1, 0]
    else:
        allocation = list(map(int, rng.integers(len(rulers), size=slot_count)))
    for s, e_i in zip(slots, allocation):
        e = rulers[e_i]
        edge, d = _edge(e, side, s, length)
        p = (edge[0]+d[0], edge[1]+d[1])
        inp[p] = code[s]
        fixed_cells.add(p)
    if demonstration is None and rng.random() < .35:
        s = int(rng.choice(slots))
        options = [i for i in range(len(rulers)) if i != allocation[slots.index(s)]]
        e = rulers[int(rng.choice(options))]
        edge, d = _edge(e, side, s, length)
        p = (edge[0]+d[0], edge[1]+d[1])
        inp[p] = code[s]
        fixed_cells.add(p)
    finals = []
    for region, (marks, window, region_sides) in enumerate(zip(marks_list, windows, sides)):
        # Top cap + a distinct outer side names the destination in two lanes.
        required = [(-1, 0), (0, -1) if region == 0 else (0, 1)]
        if count == 1:
            choices = [d for d in DIRS if region_sides[d]]
            adjacent = [(a, b) for a in choices for b in choices if a[0]*b[0]+a[1]*b[1] == 0]
            required = list(adjacent[int(rng.integers(len(adjacent)))])
        extra = [d for d in DIRS if region_sides[d] and d not in required]
        if extra and rng.random() < .35:
            required.append(extra[int(rng.integers(len(extra)))])
        final = _piece(rng, marks, window, region_sides, required, body, reg)
        if any(index[p] != region for p in final):
            return None
        if final.keys() & _halo(fixed_cells):
            return None
        finals.append(final)
    blocked = _halo(fixed_cells)
    sources = []
    for i, final in enumerate(finals):
        if demonstration == "pool":
            t = 1 if i == 0 else 4
        elif demonstration == "repeat":
            t = 4
        else:
            t = int(rng.integers(8))
        cells = _transform(final, t)
        ph, pw = 1+max(r for r, _ in cells), 1+max(c for _, c in cells)
        origins = [(r, c) for r0, r1, c0, c1 in rectangles
                   for r in range(r0, r1-ph+1) for c in range(c0, c1-pw+1)]
        rng.shuffle(origins)
        if fallback:
            # All fallback sources fit below the optical stations.
            origins.sort(key=lambda p: (p[0] < 18, abs(p[1]-rectangles[i][2]), p))
        for origin in origins:
            source = _shift(cells, origin)
            if source.keys() & blocked or source == final:
                continue
            sources.append(source)
            blocked.update(_halo(source))
            break
        else:
            return None
    out = inp.copy()
    for source in sources:
        for p, value in source.items():
            if inp[p] != bg:
                return None
            inp[p] = value
    for final in finals:
        for p, value in final.items():
            out[p] = value
    # Output starts from the stationary scene, so original footprints can be
    # overwritten safely by any final piece.
    t = int(rng.integers(8))
    inp, out = _orient(inp, t), _orient(out, t)
    try:
        predicted, info = _reconstruct(inp)
        if not np.array_equal(predicted, out):
            return None
        if demonstration == "pool" and len(_placements(info, use_compass=False)) < 2:
            return None
    except (AssertionError, IndexError, KeyError):
        return None
    return {"input": inp, "output": out}


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", num_train), ("test", num_test)):
        for i in range(number):
            demonstration = ("pool" if i == 0 else "repeat" if i == 1 else None) if split == "train" else None
            count = 2 if demonstration == "pool" else 1 if demonstration == "repeat" else int(rng.integers(1, 3))
            if demonstration == "pool":
                slots = 2
            elif demonstration == "repeat":
                slots = 3
            elif split == "train":
                slots = int(rng.integers(2, 4))
            else:
                slots = int(rng.choice([4, 5])) if count == 1 else 4
            for attempt in range(64):
                pair = _construct(rng, count, slots, demonstration)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                for attempt in range(16):
                    pair = _construct(rng, count, slots, demonstration, fallback=True)
                    if pair is not None and _key(pair["input"]) not in seen:
                        break
                else:
                    raise RuntimeError("bounded construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    code = info["code"]
    slots = tuple(sorted(code))
    local = [[e["observed"] for e in es] for es in info["rulers_by_region"]]
    partial_targets = _targets(info, codes=local)
    old = set().union(*(set(s) for s in info["sources"]))
    new = set().union(*(set(p[2]) for p in info["selected"]))
    moves, shapes, densities, patches, cross_region = [], [], [], [], []
    for source_i, region, final, _ in info["selected"]:
        source = info["sources"][source_i]
        transforms = [t for t in range(8) if _signature(_transform(source, t)) == _signature(final)]
        moves.append("translation" if 0 in transforms else "rotation" if any(t < 4 for t in transforms) else "reflection")
        shape = (1+max(r for r, _ in _transform(final, 0)), 1+max(c for _, c in _transform(final, 0)))
        shapes.append(shape)
        densities.append(len(final)/(shape[0]*shape[1]))
        labels = _label_directions(final, info["band_colors"])
        patches.extend(Counter(labels.values()).values())
        cross_region.append(int(info["region_index"][min(source)]) != region)
    normalized_targets = []
    for target in info["targets"]:
        r0, c0 = min(r for r, _ in target), min(c for _, c in target)
        normalized_targets.append(tuple(sorted((r-r0, c-c0) for r, c in target)))
    return {"shape": info["grid"].shape, "length": info["length"], "slots": slots,
            "slot_count": len(code), "piece_count": len(info["sources"]),
            "target_count": len(info["targets"][0]), "targets": tuple(normalized_targets),
            "background": info["background"], "bands": tuple(sorted(info["band_colors"])),
            "pooled": all(len(e["observed"]) < len(code) for e in info["rulers"]),
            "cross_region_pooling": len(info["regions"]) == 2 and
                all(len({s for e in es for s in e["observed"]}) < len(code)
                    for es in info["rulers_by_region"]),
            "omitting_pool_changes_target": partial_targets != info["targets"],
            "equality_needed": _targets(info, equality=False) != info["targets"],
            "compass_needed": len(_placements(info, use_compass=False)) >= 2,
            "repeated_color": len(set(code.values())) < len(code),
            "allocations": tuple(len(e["observed"]) for e in info["rulers"]),
            "adjacent": any(b-a == 1 for a, b in zip(slots, slots[1:])),
            "end_slot": 0 in slots or info["length"]-1 in slots,
            "moves": tuple(moves), "piece_shapes": tuple(shapes),
            "densities": tuple(densities), "patch_sizes": tuple(patches),
            "cross_region_moves": tuple(cross_region), "overlap": len(old & new),
            "ruler_transforms": tuple(e["transform"] for e in info["rulers"])}


def validate_generated_puzzle(puzzle):
    """Check schema, unique docking, preserved pixels, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _check_grid(pair["input"]), _check_grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            expected, info = _reconstruct(inp)
            assert np.array_equal(expected, out), "incorrect optical docking"
            assert np.array_equal(inp[info["fixed"]], out[info["fixed"]])
            assert not np.array_equal(inp, out)
            for source_i, region, final, _ in info["selected"]:
                assert {p for p, v in final.items() if v == info["registration"]} == info["targets"][region]
                assert any(_signature(_transform(info["sources"][source_i], t)) == _signature(final)
                           for t in range(8))
                core = {p: v for p, v in final.items() if v not in info["band_colors"]}
                r0, r1 = min(r for r, _ in final), max(r for r, _ in final)
                c0, c1 = min(c for _, c in final), max(c for _, c in final)
                mask = np.zeros((r1-r0+1, c1-c0+1), dtype=bool)
                for r, c in final:
                    mask[r-r0, c-c0] = True
                assert len(_components(mask)) == 1, "pieces must stay four-connected"
                assert any(v == info["body"] for v in core.values())
            f = _features(info)
            assert f["equality_needed"]
            features[split].append(f)
    train = features["train"]
    assert any(f["cross_region_pooling"] and f["omitting_pool_changes_target"] for f in train)
    assert any(f["compass_needed"] for f in train)
    assert any(f["repeated_color"] for f in train)
    assert any("rotation" in f["moves"] for f in train)
    assert any("reflection" in f["moves"] for f in train)
    assert min(f["slot_count"] for f in features["test"]) > max(f["slot_count"] for f in train)


def self_test():
    """Bounded 42-seed coverage plus inference and corruption regressions."""
    # Explicit optical fixture: X/Y/X makes four X coincidences and one Y
    # coincidence, rather than nine unfiltered crossings or two color centers.
    ids = np.arange(10).reshape(2, 5)
    h = {"arr": ids, "labels": ids, "origin": (2, 6)}
    v = {"arr": _orient(ids, 1), "labels": _orient(ids, 1), "origin": (6, 2)}
    optical = {"rulers_by_region": [[h, v]], "code": {0: 8, 2: 9, 4: 8},
               "side": 1, "length": 5, "region_index": np.zeros((20, 20), dtype=int)}
    assert _targets(optical) == [{(10, 6), (10, 10), (8, 8), (6, 6), (6, 10)}]
    assert len(_targets(optical, equality=False)[0]) == 9
    optical["code"] = {0: 9, 2: 8, 4: 8}
    assert _targets(optical) == [{(10, 6), (8, 8), (8, 10), (6, 8), (6, 10)}]
    records, fingerprints, normalized = [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        for split in ("train", "test"):
            for pair in p[split]:
                expected, info = _reconstruct(pair["input"])
                assert np.array_equal(expected, pair["output"])
                f = _features(info)
                f["split"] = split
                records.append(f)
                normalized.add((pair["input"].shape, (pair["input"] != info["background"]).tobytes()))
    assert len(fingerprints) == 42 and len(normalized) >= 150
    assert len({f["shape"] for f in records}) >= 40
    assert {f["piece_count"] for f in records} == {1, 2}
    assert {f["slot_count"] for f in records} == {2, 3, 4, 5}
    assert {f["length"] for f in records} >= {4, 5, 6, 7, 8}
    assert {f["background"] for f in records} == set(range(10))
    assert len({target for f in records for target in f["targets"]}) >= 25
    assert len({shape for f in records for shape in f["piece_shapes"]}) >= 20
    assert {m for f in records for m in f["moves"]} == {"translation", "rotation", "reflection"}
    assert any(f["overlap"] for f in records) and any(not f["overlap"] for f in records)
    assert any(any(f["cross_region_moves"]) for f in records)
    assert any(2 in f["patch_sizes"] for f in records)
    assert any(min(f["densities"]) < .5 for f in records)
    assert all(f["equality_needed"] for f in records)
    assert sum(f["compass_needed"] for f in records) >= 42
    # Full-grid isometries and arbitrary color permutations remain inferable.
    p = generate_puzzle(9001, 2, 1)
    for t in range(8):
        recolor = np.random.default_rng(200+t).permutation(10)
        for split in ("train", "test"):
            for pair in p[split]:
                inp = _orient(recolor[pair["input"]], t)
                out = _orient(recolor[pair["output"]], t)
                assert np.array_equal(solve(inp), out)
    # Moving a port to another exact copy keeps the shared stencil and final
    # pieces unchanged; only the stationary port pixels change in the output.
    pair = p["train"][0]
    info = _parse(pair["input"])
    emitter_i, _, slot, color = info["observations"][0]
    old_e = info["rulers"][emitter_i]
    new_e = next(e for e in info["rulers"] if not e["observed"])
    old_edge, old_d = _edge(old_e, info["side"], slot, info["length"])
    new_edge, new_d = _edge(new_e, info["side"], slot, info["length"])
    old_port = (old_edge[0]+old_d[0], old_edge[1]+old_d[1])
    new_port = (new_edge[0]+new_d[0], new_edge[1]+new_d[1])
    shifted_input, shifted_output = pair["input"].copy(), pair["output"].copy()
    for a in (shifted_input, shifted_output):
        a[old_port] = info["background"]
        a[new_port] = color
    assert np.array_equal(solve(shifted_input), shifted_output)
    assert tuple(_key(x["input"]) for x in p["train"]+p["test"]) == tuple(
        _key(x["input"]) for x in (lambda q: q["train"]+q["test"])(generate_puzzle(9001, 2, 1)))
    # Validator must reject a changed stationary pixel and a removed body cell.
    for kind in ("fixed", "body"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        info = _parse(bad["test"][0]["input"])
        if kind == "fixed":
            position = tuple(np.argwhere(info["fixed"])[0])
            bad["test"][0]["output"][position] = info["background"]
        else:
            _, solved_info = _reconstruct(bad["test"][0]["input"])
            position = next(p for _, _, final, _ in solved_info["selected"]
                            for p, value in final.items() if value == info["body"])
            bad["test"][0]["output"][position] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("output corruption was accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({f["shape"] for f in records}),
            "virtual_stencils": len({t for f in records for t in f["targets"]}),
            "piece_shapes": len({s for f in records for s in f["piece_shapes"]}),
            "compass_dependent_pairs": sum(f["compass_needed"] for f in records),
            "old_new_overlap_pairs": sum(bool(f["overlap"]) for f in records)}
