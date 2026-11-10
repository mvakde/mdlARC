"""Corner couriers: compass docking changes the endpoints of ordered rectangles.

Only standard Python and NumPy are used. There is no file I/O, seed lookup,
training-state inference, or saved execution trace in the generator or solver.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N E S W
_AROUND = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or a.dtype.kind not in "iu" or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _transform(cells, t):
    result = []
    for (r, c), value in sorted(cells.items()):
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result.append((r, c, value))
    mr = min(r for r, c, v in result)
    mc = min(c for r, c, v in result)
    return {(r-mr, c-mc): int(v) for r, c, v in result}


def _shift(cells, dr, dc):
    return {(r+dr, c+dc): v for (r, c), v in cells.items()}


def _canon(points):
    points = tuple(points)
    return min(tuple(sorted(_transform({p: 1 for p in points}, t))) for t in range(8))


def _components(points):
    todo, result = set(points), []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, component = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in _AROUND:
                p = r+dr, c+dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
                    component.add(p)
        result.append(component)
    return result


def _halo(points):
    return {(r+dr, c+dc) for r, c in points for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _corner(points):
    points = set(points)
    if len(points) != 3:
        raise ValueError("a corner must be a three-cell L")
    for r, c in sorted(points):
        for dr in (-1, 1):
            for dc in (-1, 1):
                if points == {(r, c), (r+dr, c), (r, c+dc)}:
                    return (r, c), dr, dc
    raise ValueError("malformed L")


def _rectangle(moving, fixed):
    (r, c), dr, dc = _corner(moving)
    (rr, cc), drr, dcc = _corner(fixed)
    if (drr != -dr or dcc != -dc or (rr-r)*dr < 3 or (cc-c)*dc < 3):
        raise ValueError("the two Ls must face inward from opposite corners of a >=4x4 box")
    return min(r, rr), min(c, cc), max(r, rr), max(c, cc)


def _inside(p, b):
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def _labels(cells, bands):
    core = [p for p, v in cells.items() if v not in bands]
    r0, r1 = min(r for r, c in core), max(r for r, c in core)
    c0, c1 = min(c for r, c in core), max(c for r, c in core)
    result = {}
    for (r, c), value in cells.items():
        if value not in bands:
            continue
        ds = [d for d, condition in enumerate((r < r0, c > c1, r > r1, c < c0)) if condition]
        if len(ds) != 1:
            raise ValueError("a compass patch must project beyond exactly one core side")
        result[(r, c)] = ds[0]
    sides = set(result.values())
    if not any(d in sides and (d+1) % 4 in sides for d in range(4)):
        raise ValueError("at least two adjacent compass directions are required")
    return result


def _frame(a, bg):
    # Only the work frame has monochrome horizontal runs of length >=12.
    runs = []
    h, w = a.shape
    for r in range(h):
        c = 0
        while c < w:
            end = c+1
            while end < w and a[r, end] == a[r, c]:
                end += 1
            if end-c >= 12 and a[r, c] != bg:
                runs.append((r, c, end-1, int(a[r, c])))
            c = end
    candidates = []
    for r0, c0, c1, north in runs:
        for r1, cc0, cc1, south in runs:
            if r1-r0 < 11 or (c0, c1) != (cc0, cc1):
                continue
            west, east = int(a[r0+1, c0]), int(a[r0+1, c1])
            if len({bg, north, east, south, west}) != 5:
                continue
            if np.all(a[r0+1:r1, c0] == west) and np.all(a[r0+1:r1, c1] == east):
                candidates.append(((r0, c0, r1, c1), (north, east, south, west)))
    if len(candidates) != 1:
        raise ValueError("work frame is not unique")
    return candidates[0]


def _parse(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    if sum(counts == counts[bg]) != 1:
        raise ValueError("background must be the unique most frequent input color")
    frame, bands = _frame(a, bg)
    r0, c0, r1, c1 = frame
    inner = r0+1, c0+1, r1-1, c1-1
    outside = a != bg
    outside[r0:r1+1, c0:c1+1] = False
    coords = np.argwhere(outside)
    if not len(coords):
        raise ValueError("missing external palette")
    tr, tc = map(int, coords.min(axis=0))
    br, bc = map(int, coords.max(axis=0))
    if bc > tc+1 or not (2 <= br-tr+1 <= 4) or tc+1 >= a.shape[1]:
        raise ValueError("external palette must have two columns and 2..4 rows")
    if any(_inside((r, c), frame) for r in range(tr, br+1) for c in (tc, tc+1)):
        raise ValueError("palette intersects work frame")
    cues = list(map(int, a[tr:br+1, tc]))
    fills = list(map(int, a[tr:br+1, tc+1]))
    if len(set(cues)) != len(cues) or (set(cues) & ({bg} | set(bands))):
        raise ValueError("cue identities must be distinct from background and compass")
    allowed = np.zeros(a.shape, dtype=bool)
    allowed[tr:br+1, tc:tc+2] = True
    if np.any(outside & ~allowed):
        raise ValueError("unexpected external material")
    points = {(r, c) for r in range(r0+1, r1) for c in range(c0+1, c1) if a[r, c] != bg}
    reg_colors = {int(a[p]) for p in points} - set(cues) - set(bands)
    if len(reg_colors) != 1:
        raise ValueError("registration role is not unique")
    reg = next(iter(reg_colors))
    sources, targets, fixed = {}, [], {}
    for component in _components(points):
        cells = {p: int(a[p]) for p in component}
        colors = set(cells.values())
        if colors == {reg}:
            targets.append(component)
        elif len(colors) == 1 and next(iter(colors)) in cues:
            cue = next(iter(colors))
            _corner(component)
            if cue in fixed:
                raise ValueError("multiple stationary corners for one cue")
            fixed[cue] = cells
        else:
            identities = colors & set(cues)
            if len(identities) != 1 or reg not in colors:
                raise ValueError("a courier needs one cue and a registration patch")
            cue = next(iter(identities))
            _corner(p for p, v in cells.items() if v == cue)
            _labels(cells, bands)
            if cue in sources:
                raise ValueError("duplicate courier identity")
            sources[cue] = cells
    if set(sources) != set(cues) or set(fixed) != set(cues) or len(targets) != len(cues):
        raise ValueError("one courier, one stationary L, and one stencil are required per row")
    shapes = [_canon(p for p, v in sources[cue].items() if v == reg) for cue in cues]
    if len(set(shapes)) != len(shapes) or sorted(shapes) != sorted(_canon(t) for t in targets):
        raise ValueError("registration shapes must have a unique congruence match")
    return dict(grid=a, bg=bg, frame=frame, inner=inner, bands=bands, reg=reg,
                cues=cues, fills=fills, sources=sources, targets=targets, fixed=fixed,
                table=(tr, tc, br, tc+1))


def _placements(p, cue, use_compass=True):
    source, fixed = p["sources"][cue], p["fixed"][cue]
    result = {}
    for t in range(8):
        cells = _transform(source, t)
        marks = {q for q, v in cells.items() if v == p["reg"]}
        labels = _labels(cells, p["bands"])
        if use_compass and any(cells[q] != p["bands"][d] for q, d in labels.items()):
            continue
        for ti, target in enumerate(p["targets"]):
            dr, dc = min(target)[0]-min(marks)[0], min(target)[1]-min(marks)[1]
            if {(r+dr, c+dc) for r, c in marks} != target:
                continue
            landed = _shift(cells, dr, dc)
            if not all(_inside(q, p["inner"]) for q in landed):
                continue
            try:
                bounds = _rectangle((q for q, v in landed.items() if v == cue), fixed)
            except ValueError:
                continue
            if not all(_inside(q, bounds) for q, v in landed.items() if v not in p["bands"]):
                continue
            signature = tuple(sorted(landed.items()))
            result[signature] = dict(cue=cue, key=landed, fixed=fixed, bounds=bounds, target=ti)
    return list(result.values())


def _recover(p):
    layers, used = [], set()
    for cue, fill in zip(p["cues"], p["fills"]):
        choices = _placements(p, cue)
        if len(choices) != 1 or choices[0]["target"] in used:
            raise ValueError("compass/stencil docking is not unique")
        layer = dict(choices[0], fill=fill)
        used.add(layer["target"])
        layers.append(layer)
    return layers


def _clean(p):
    result = p["grid"].copy()
    r0, c0, r1, c1 = p["inner"]
    result[r0:r1+1, c0:c1+1] = p["bg"]
    return result


def _compose(p, layers):
    """Constructor rasterizer: paint complete layers from back to front."""
    result = _clean(p)
    for layer in reversed(layers):
        r0, c0, r1, c1 = layer["bounds"]
        result[r0:r1+1, c0:c1+1] = layer["fill"]
        for q, v in {**layer["key"], **layer["fixed"]}.items():
            result[q] = v
    return result


def _predict(p, layers):
    """Input-only rasterizer: the first covering row wins at each location."""
    result = _clean(p)
    claimed = set()
    for layer in layers:
        r0, c0, r1, c1 = layer["bounds"]
        own = {**layer["key"], **layer["fixed"]}
        support = set(own) | {(r, c) for r in range(r0, r1+1) for c in range(c0, c1+1)}
        for q in support - claimed:
            result[q] = own.get(q, layer["fill"])
        claimed.update(support)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(input_grid)
    return _predict(p, _recover(p))


def _core(rng, cue, reg, special=False):
    cue_cells = {(0, 0), (0, 1), (1, 0)}
    start = (1, 2) if special else ((1, 1), (1, 2), (2, 1))[int(rng.integers(3))]
    marks = {start}
    if special and rng.random() < .5:
        marks = {(1, 2), (1, 3), (2, 2), (2, 3)}
    size = len(marks) if special else int(rng.integers(1, 7))
    while len(marks) < size:
        frontier = sorted({(r+dr, c+dc) for r, c in marks for dr, dc in _DIRS
                           if 0 <= r+dr < 5 and 0 <= c+dc < 5} - marks - cue_cells)
        marks.add(frontier[int(rng.integers(len(frontier)))])
    return {**{q: cue for q in cue_cells}, **{q: reg for q in marks}}


def _make_key(rng, core, bands):
    core = _transform(core, int(rng.integers(8)))
    r1, c1 = max(r for r, c in core), max(c for r, c in core)
    first = int(rng.integers(4))
    sides = {first, (first+1) % 4}
    if rng.random() < .3:
        sides.add((first+2) % 4)
    result = dict(core)
    for d in sorted(sides):
        supports = sorted(q for q in core if (q[0] == 0, q[1] == c1, q[0] == r1, q[1] == 0)[d])
        number = int(rng.integers(1, min(2, len(supports))+1))
        for i in rng.choice(len(supports), number, replace=False):
            r, c = supports[int(i)]
            dr, dc = _DIRS[d]
            result[r+dr, c+dc] = bands[d]
    return _transform(result, 0)


def _scene(rng, n):
    palette = list(map(int, rng.permutation(10)))
    bg, reg, bands, cues = palette[0], palette[1], tuple(palette[2:6]), palette[6:6+n]
    side = int(rng.integers(4))  # external table above/right/below/left
    strip = n+2 if side in (0, 2) else int(rng.integers(4, 6))
    fh = int(rng.integers(20 if n >= 3 else 18, 31-strip if side in (0, 2) else 31))
    fw = int(rng.integers(22 if n == 4 else 18, 31-strip if side in (1, 3) else 31))
    h, w = fh+(strip if side in (0, 2) else 0), fw+(strip if side in (1, 3) else 0)
    r0, c0 = (strip if side == 0 else 0), (strip if side == 3 else 0)
    r1, c1 = r0+fh-1, c0+fw-1
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[r0:r1+1, c0] = bands[3]
    grid[r0:r1+1, c1] = bands[1]
    grid[r0, c0:c1+1], grid[r1, c0:c1+1] = bands[0], bands[2]
    if side in (0, 2):
        tr = int(rng.integers(1, strip-n)) + (fh if side == 2 else 0)
        tc = int(rng.integers(0, w-1))
    else:
        tr = int(rng.integers(0, h-n+1))
        tc = int(rng.integers(0, strip-1)) + (fw if side == 1 else 0)
    fills = list(map(int, rng.integers(0, 10, size=n)))
    return dict(grid=grid, bg=bg, reg=reg, bands=bands, cues=cues, fills=fills,
                frame=(r0, c0, r1, c1), inner=(r0+1, c0+1, r1-1, c1-1), table=(tr, tc, tr+n-1, tc+1))


def _final_geometry(rng, p, cores, mode, reach):
    r0, c0, r1, c1 = p["inner"]
    layers, blocked = [], set()
    for cue, core in zip(p["cues"], cores):
        key = _make_key(rng, core, p["bands"])
        vertex, dr, dc = _corner(q for q, v in key.items() if v == cue)
        kh = max(r for r, c in key)+1
        kw = max(c for r, c in key)+1
        placed = False
        for _ in range(100):
            hh = int(rng.integers(max(4, kh-1), min(r1-r0-1, reach)+1))
            ww = int(rng.integers(max(4, kw-1), min(c1-c0-1, reach)+1))
            br = int(rng.integers(r0+1, r1-hh+1))
            bc = int(rng.integers(c0+1, c1-ww+1))
            bounds = br, bc, br+hh-1, bc+ww-1
            vr = bounds[0] if dr == 1 else bounds[2]
            vc = bounds[1] if dc == 1 else bounds[3]
            landed = _shift(key, vr-vertex[0], vc-vertex[1])
            if not all(_inside(q, p["inner"]) for q in landed):
                continue
            if not all(_inside(q, bounds) for q, v in landed.items() if v not in p["bands"]):
                continue
            fr, fc = vr+dr*(hh-1), vc+dc*(ww-1)
            fixed = {q: cue for q in ((fr, fc), (fr-dr, fc), (fr, fc-dc))}
            marks = {q for q, v in landed.items() if v == p["reg"]}
            if marks & _halo(fixed) or (marks | set(fixed)) & blocked:
                continue
            previous = [o["bounds"] for o in layers]
            if previous:
                hits = [_overlap(bounds, b) for b in previous]
                if mode == "disjoint" and any(hits):
                    continue
                if mode == "overlap" and not any(hits):
                    continue
                if mode == "chain" and (not hits[-1] or any(hits[:-1])):
                    continue
                if mode == "nested" and not any(_contains(bounds, b) or _contains(b, bounds) for b in previous):
                    continue
            layers.append(dict(cue=cue, key=landed, fixed=fixed, bounds=bounds))
            blocked.update(_halo(marks | set(fixed)))
            placed = True
            break
        if not placed:
            return None
    return layers


def _witnesses(p, layers, output):
    result = set()
    all_markers = set().union(*(set(o["key"]) | set(o["fixed"]) for o in layers))
    reverse = _predict(p, list(reversed(layers)))
    if any(tuple(q) not in all_markers and sum(_inside(q, o["bounds"]) for o in layers) >= 2
           for q in np.argwhere(output != reverse)):
        result.add("priority")
    for i, layer in enumerate(layers):
        own = {**layer["key"], **layer["fixed"]}
        if any(output[q] != v for q, v in own.items()):
            result.add("occlusion")
        if any(output[q] == layer["cue"] for q in layer["fixed"]):
            result.add("retention")
        r0, c0, r1, c1 = layer["bounds"]
        for r in range(r0, r1+1):
            for c in range(c0, c1+1):
                q = r, c
                if q in own or any(_inside(q, up["bounds"]) or q in up["key"] for up in layers[:i]):
                    continue
                if layer["fill"] != p["bg"] and output[q] == layer["fill"] and p["grid"][q] != output[q]:
                    result.add("filling")
                if layer["fill"] == p["bg"] and output[q] == p["bg"]:
                    for lower in layers[i+1:]:
                        if _inside(q, lower["bounds"]) and lower["fill"] != p["bg"]:
                            result.add("clearing")
    # Omitting docking uses the original cue positions as the endpoints.
    undocked = []
    for layer in layers:
        old = p["sources"][layer["cue"]]
        points = {q for q, v in old.items() if v == layer["cue"]} | set(layer["fixed"])
        b = min(r for r, c in points), min(c for r, c in points), max(r for r, c in points), max(c for r, c in points)
        undocked.append(dict(layer, key=old, bounds=b))
    if not np.array_equal(output, _predict(p, undocked)):
        result.add("docking")
    for i, layer in enumerate(layers):
        for alt in _placements(p, layer["cue"], use_compass=False):
            if alt["bounds"] != layer["bounds"]:
                changed = list(layers)
                changed[i] = dict(alt, fill=layer["fill"])
                prediction = _predict(p, changed)
                excluded = all_markers | set(alt["key"])
                if any(tuple(q) not in excluded and
                       (_inside(q, alt["bounds"]) != _inside(q, layer["bounds"]))
                       for q in np.argwhere(output != prediction)):
                    result.add("compass_geometry")
                    break
        if "compass_geometry" in result:
            break
    return result


def _candidate(rng, n, teaching=None, mode="free", reach=13):
    p = _scene(rng, n)
    cores, shapes = [], set()
    for i, cue in enumerate(p["cues"]):
        for _ in range(60):
            core = _core(rng, cue, p["reg"], special=(i == 0 and teaching is not None))
            shape = _canon(q for q, v in core.items() if v == p["reg"])
            if shape not in shapes:
                break
        else:
            return None
        shapes.add(shape)
        cores.append(core)
    layers = _final_geometry(rng, p, cores, mode, reach)
    if layers is None:
        return None
    # Rank belongs to cue identity, independently of geometry and stencil size.
    layers = [layers[int(i)] for i in rng.permutation(n)]
    p["cues"] = [o["cue"] for o in layers]
    if teaching == "clear":
        p["fills"][0] = p["bg"]
        p["fills"][1] = int(rng.choice([c for c in range(10) if c != p["bg"]]))
    elif teaching == "paint":
        p["fills"][0], p["fills"][1] = map(int, rng.choice([c for c in range(10) if c != p["bg"]], 2, replace=False))
    for layer, fill in zip(layers, p["fills"]):
        layer["fill"] = fill
    tr, tc, _, _ = p["table"]
    for i, (cue, fill) in enumerate(zip(p["cues"], p["fills"])):
        p["grid"][tr+i, tc:tc+2] = (cue, fill)
    blocked = set()
    for layer in layers:
        marks = {q for q, v in layer["key"].items() if v == p["reg"]}
        for q in marks:
            p["grid"][q] = p["reg"]
        for q, v in layer["fixed"].items():
            p["grid"][q] = v
        blocked.update(_halo(marks | set(layer["fixed"])))
    # Pack scrambled keys throughout remaining space, with no source/destination slots.
    for index, layer in enumerate(layers):
        poses = list(map(int, rng.permutation(8)))
        if index == 0 and teaching:
            poses = [t for t in poses if (1 <= t <= 3 if teaching == "paint" else t >= 4)]
        placed = False
        for t in poses:
            cells = _transform(layer["key"], t)
            kh, kw = max(r for r, c in cells)+1, max(c for r, c in cells)+1
            ir0, ic0, ir1, ic1 = p["inner"]
            origins = [(r, c) for r in range(ir0, ir1-kh+2) for c in range(ic0, ic1-kw+2)]
            rng.shuffle(origins)
            for r, c in origins:
                source = _shift(cells, r, c)
                if source.keys() & blocked:
                    continue
                for q, v in source.items():
                    p["grid"][q] = v
                blocked.update(_halo(source))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    output = _compose(p, layers)
    try:
        parsed = _parse(p["grid"])
        recovered = _recover(parsed)
        if not np.array_equal(output, _predict(parsed, recovered)):
            return None
        if teaching:
            needed = {"docking", "compass_geometry", "priority", "occlusion", "retention", "filling"}
            if teaching == "clear":
                needed.add("clearing")
            if not needed <= _witnesses(parsed, recovered, output):
                return None
    except ValueError:
        return None
    return {"input": p["grid"], "output": output}


def _fallback(rng, n, teaching, colors=None):
    """Finite, spacious construction of the identical rule after spatial rejection.

    The first off-center singleton ensures a geometry-changing compass witness.
    This is a packing fallback, with no new semantics or seed-dependent answer.
    """
    colors = list(map(int, rng.permutation(10))) if colors is None else list(colors)
    bg, reg, bands, cues = colors[0], colors[1], tuple(colors[2:6]), colors[6:6+n]
    grid = np.full((30, 30), bg, dtype=np.int8)
    grid[:, 0], grid[:, 24] = bands[3], bands[1]
    grid[0, :25], grid[29, :25] = bands[0], bands[2]
    p = dict(grid=grid, bg=bg, reg=reg, bands=bands, cues=cues,
             frame=(0, 0, 29, 24), inner=(1, 1, 28, 23), table=(1, 27, n, 28))
    shapes = ({(1, 2)}, {(1, 1), (1, 2)}, {(1, 1), (1, 2), (2, 1)},
              {(1, 1), (1, 2), (2, 1), (2, 2)})
    boxes = ((4, 4, 12, 14), (2, 3, 19, 19), (5, 12, 24, 21), (13, 3, 25, 11))
    origins = ((20, 1), (1, 9), (1, 18), (25, 2))
    fills = list(map(int, rng.integers(10, size=n)))
    if teaching:
        fills[:2] = [bg if teaching == "clear" else bands[0], bands[1]]
    layers = []
    for i, cue in enumerate(cues):
        cells = {**{q: cue for q in ((0, 0), (0, 1), (1, 0))},
                 **{q: reg for q in shapes[i]}, (-1, 0): bands[0], (1, -1): bands[3]}
        r0, c0, r1, c1 = boxes[i]
        landed = _shift(cells, r0, c0)
        fixed = {q: cue for q in ((r1, c1), (r1-1, c1), (r1, c1-1))}
        layer = dict(cue=cue, fill=fills[i], key=landed, fixed=fixed, bounds=boxes[i])
        layers.append(layer)
        for q, v in landed.items():
            if v == reg:
                grid[q] = v
        for q, v in fixed.items():
            grid[q] = v
        t = (4 if teaching == "clear" else 1) if i == 0 else (4, 4, 3, 5)[i]
        source = _shift(_transform(cells, t), *origins[i])
        for q, v in source.items():
            grid[q] = v
        grid[1+i, 27:29] = (cue, fills[i])
    output = _compose(p, layers)
    parsed = _parse(grid)
    recovered = _recover(parsed)
    assert np.array_equal(output, _predict(parsed, recovered))
    if teaching:
        needed = {"docking", "compass_geometry", "priority", "occlusion", "retention", "filling"}
        if teaching == "clear":
            needed.add("clearing")
        assert needed <= _witnesses(parsed, recovered, output)
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer from 2 through 10")
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("num_test must be 1..4 and smaller than num_train")
    rng = np.random.default_rng(seed)
    counts = [2, 2] + [int(rng.integers(2, 4)) for _ in range(num_train-2)]
    test_n = max(counts)+1
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            n = counts[i] if split == "train" else test_n
            teaching = ("paint" if i == 0 else "clear" if i == 1 else None) if split == "train" else None
            mode = "overlap" if teaching else str(rng.choice(["free", "free", "overlap", "disjoint", "nested", "chain"]))
            for attempt in range(160):
                # Hard topologies may use free placements after 40 attempts;
                # this changes the sampling preference, never the puzzle rule.
                selected_mode = mode if attempt < 40 else ("overlap" if teaching else "free")
                pair = _candidate(rng, n, teaching, selected_mode, 17 if split == "test" else 13)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                # Twenty distinct frame/registration palettes; at most thirteen
                # earlier inputs exist, so a fresh fallback is guaranteed.
                base_colors = list(map(int, rng.permutation(10)))
                for offset in range(20):
                    colors = [(c+offset % 10) % 10 for c in base_colors]
                    if offset >= 10:
                        colors[0], colors[1] = colors[1], colors[0]
                    pair = _fallback(rng, n, teaching, colors)
                    if _key(pair["input"]) not in seen:
                        break
                else:
                    raise RuntimeError("unexpected duplicate fallback exhaustion")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, witnesses, train_counts, test_counts = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and not np.array_equal(a, b)
            signature = _key(a)
            assert signature not in seen, "repeated input"
            seen.add(signature)
            p = _parse(a)
            layers = _recover(p)
            assert np.array_equal(_predict(p, layers), b), "wrong docking/completion/compositing"
            assert np.array_equal(_compose(p, layers), b), "rasterizers disagree"
            r0, c0, r1, c1 = p["inner"]
            outside = np.ones(a.shape, dtype=bool)
            outside[r0:r1+1, c0:c1+1] = False
            assert np.array_equal(a[outside], b[outside]), "frame/palette/background changed"
            if split == "train":
                witnesses.update(_witnesses(p, layers, b))
                train_counts.append(len(layers))
            else:
                test_counts.append(len(layers))
    assert {"docking", "compass_geometry", "priority", "occlusion", "clearing", "retention", "filling"} <= witnesses
    assert min(test_counts) > max(train_counts), "missing courier-count generalization"


def measure_pair(pair):
    """Bounded descriptive measurements recovered entirely from the public grids."""
    p = _parse(pair["input"])
    layers = _recover(p)
    patterns = [_canon(q for q, v in p["sources"][cue].items() if v == p["reg"]) for cue in p["cues"]]
    bounds = [o["bounds"] for o in layers]
    edges = [(i, j) for i, a in enumerate(bounds) for j, b in enumerate(bounds) if i < j and _overlap(a, b)]
    return dict(shape=list(p["grid"].shape), background=p["bg"], count=len(layers),
                frame=list(p["frame"]), table=list(p["table"]), patterns=patterns,
                bounds=bounds, key_shapes=[(max(r for r, c in o["key"])-min(r for r, c in o["key"])+1,
                                           max(c for r, c in o["key"])-min(c for r, c in o["key"])+1) for o in layers],
                overlaps=len(edges), nested=any(_contains(bounds[i], bounds[j]) or _contains(bounds[j], bounds[i]) for i, j in edges),
                same_count_different_shape=len(set(map(len, patterns))) < len(patterns),
                source_repainted=any(pair["output"][q] != p["bg"] for source in p["sources"].values() for q in source),
                witnesses=sorted(_witnesses(p, layers, pair["output"])))


def self_test():
    """42 seeds, streaming measurements, no files or retained puzzle collection."""
    shapes, patterns, key_shapes, backgrounds, counts, inputs = set(), set(), set(), set(), set(), set()
    feature_counts = dict(overlap=0, disjoint=0, nested=0, equal_count=0, repaint=0)
    for n in (2, 3, 4):
        for teaching in (None, "paint", "clear"):
            pair = _fallback(np.random.default_rng(n), n, teaching)
            assert np.array_equal(solve(pair["input"]), pair["output"])
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"].copy()
                predicted = solve(a)
                assert np.array_equal(a, pair["input"]), "solver mutated input"
                assert np.array_equal(predicted, pair["output"])
                m = measure_pair(pair)
                shapes.add(tuple(m["shape"]))
                patterns.update(m["patterns"])
                key_shapes.update(m["key_shapes"])
                backgrounds.add(m["background"])
                counts.add(m["count"])
                inputs.add(hashlib.sha256(pair["input"].tobytes()).digest())
                for name, active in (("overlap", m["overlaps"] > 0), ("disjoint", m["overlaps"] == 0),
                                     ("nested", m["nested"]), ("equal_count", m["same_count_different_shape"]),
                                     ("repaint", m["source_repainted"])):
                    feature_counts[name] += int(active)
    assert len(inputs) == 210
    assert len(shapes) >= 30 and len(patterns) >= 20 and len(key_shapes) >= 12
    assert backgrounds == set(range(10)) and counts == {2, 3, 4}
    assert all(v >= 3 for v in feature_counts.values()), feature_counts
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(731, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(923, 5, 3), generate_puzzle(923, 5, 3)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], second[s]) for k in ("input", "output"))


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, geometric compass and opaque-layer witnesses")
