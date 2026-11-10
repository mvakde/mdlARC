"""Contour nets: infer a hollow enclosure, collect its cargo, and dock its handle.

No parent module, file, seed, or generation record is used by solve().
"""
from __future__ import annotations

from collections import Counter
import numpy as np

ORTHOGONAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    _require(n > 0 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)), "unaligned vertices")
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return {(a[0] + sr * k, a[1] + sc * k) for k in range(n + 1)}


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k]) or
           max(c[k], d[k]) < min(a[k], b[k]) for k in (0, 1)):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _contour(points):
    """Unique noncrossing full cycle through visible landmarks; reject chords."""
    points = sorted(points)
    n = len(points)
    _require(4 <= n <= 12, "a net needs 4..12 landmarks")
    landmarks = set(points)
    edges, lines = [], []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = points[i], points[j]
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            if not (dr == 0 or dc == 0 or dr == dc):
                continue
            line = _segment(a, b)
            if (line - {a, b}) & landmarks:
                continue
            edges.append((i, j))
            lines.append(line)
    adjacency = [[] for _ in points]
    for e, (i, j) in enumerate(edges):
        adjacency[i].append((j, e))
        adjacency[j].append((i, e))
    conflicts = [set() for _ in edges]
    for e, (a, b) in enumerate(edges):
        for f, (c, d) in enumerate(edges[:e]):
            if not {a, b} & {c, d} and _intersects(points[a], points[b], points[c], points[d]):
                conflicts[e].add(f)
                conflicts[f].add(e)
    solutions, visits = {}, 0

    def visit(path, selected):
        nonlocal visits
        visits += 1
        _require(visits <= 50000, "ambiguous topology exceeds search bound")
        if len(solutions) > 1:
            return
        if len(path) == n:
            for neighbor, edge in adjacency[path[-1]]:
                if neighbor == 0 and not conflicts[edge] & set(selected):
                    key = tuple(sorted(selected + [edge]))
                    solutions[key] = path
            return
        for neighbor, edge in adjacency[path[-1]]:
            if neighbor not in path and not conflicts[edge] & set(selected):
                visit(path + [neighbor], selected + [edge])

    visit([0], [])
    _require(len(solutions) == 1, "net has no unique noncrossing closed contour")
    chosen, order = next(iter(solutions.items()))
    polygon = [points[i] for i in order]
    stroke = set().union(*(lines[e] for e in chosen))
    turns = {_cross(polygon[i - 1], polygon[i], polygon[(i + 1) % n]) for i in range(n)}
    area2 = abs(sum(a[1] * b[0] - b[1] * a[0] for a, b in zip(polygon, polygon[1:] + polygon[:1])))
    _require(area2 > 0 and stroke - landmarks, "empty or already complete net")
    return {"points": points, "polygon": polygon, "stroke": stroke,
            "all_links": set().union(*lines), "area2": area2,
            "concave": any(x < 0 for x in turns) and any(x > 0 for x in turns),
            "chords": len(edges) - len(chosen)}


def _inside(p, polygon):
    """Strict continuous polygon interior; stroke cells are checked separately."""
    r, c = p
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[0] > r) != (b[0] > r):
            crossing = a[1] + (r - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
            if c < crossing:
                inside = not inside
    return inside


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ORTHOGONAL:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _halo(points, diagonal=False):
    dirs = NEIGHBORS if diagonal else ORTHOGONAL
    return set(points) | {(r + dr, c + dc) for r, c in points for dr, dc in dirs}


def _decode(value):
    grid = _grid(value)
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > grid.size / 2, "background must occupy a strict majority")
    foreground = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    anchors = {}
    for component in _components(foreground):
        if len(component) == 1:
            p = next(iter(component))
            if p[0] in (0, h - 1) or p[1] in (0, w - 1):
                anchors.setdefault(int(grid[p]), set()).add(p)
    _require(1 <= len(anchors) <= 3, "need 1..3 docking identities")
    guides = {}
    for color, ps in anchors.items():
        horizontal = len({r for r, c in ps}) == 1 and all(c in (0, w - 1) for r, c in ps)
        vertical = len({c for r, c in ps}) == 1 and all(r in (0, h - 1) for r, c in ps)
        _require(horizontal != vertical and len(ps) <= 2, "inconsistent or corner anchors")
        axis = 0 if horizontal else 1
        guide = next(iter(ps))[axis]
        _require(0 < guide < grid.shape[axis] - 1, "corner anchor")
        guides[color] = axis, guide
    anchor_points = set().union(*anchors.values())
    handles = sorted(p for p in foreground - anchor_points if int(grid[p]) in anchors)
    _require(1 <= len(handles) <= 3, "need 1..3 net handles")
    nets, source_colors = [], set()
    for handle in handles:
        _require(0 < handle[0] < h - 1 and 0 < handle[1] < w - 1, "handle must start inside canvas")
        neighbors = [p for p in foreground - anchor_points if p != handle and
                     max(abs(p[0] - handle[0]), abs(p[1] - handle[1])) == 1]
        _require(len(neighbors) == 1, "handle must adjoin exactly one landmark")
        color = int(grid[neighbors[0]])
        _require(color not in anchors and color not in source_colors, "ambiguous net identity")
        source_colors.add(color)
        points = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
        _require(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in points), "source landmarks must be interior")
        net = _contour(points)
        marker = int(grid[handle])
        axis, guide = guides[marker]
        delta = [0, 0]
        delta[axis] = guide - handle[axis]
        net.update(color=color, marker=marker, handle=handle, axis=axis,
                   guide=guide, delta=tuple(delta), cargo=set())
        _require(handle not in net["stroke"], "handle is on an intended stroke")
        nets.append(net)
    _require(set(int(grid[p]) for p in handles) == set(anchors), "unused anchor")
    # Enclosures are laminar: either disjoint or strictly nested, never crossing.
    for i, a in enumerate(nets):
        for b in nets[:i]:
            _require(not a["stroke"] & b["stroke"], "net outlines touch")
            for pa, qa in zip(a["polygon"], a["polygon"][1:] + a["polygon"][:1]):
                for pb, qb in zip(b["polygon"], b["polygon"][1:] + b["polygon"][:1]):
                    _require(not _intersects(pa, qa, pb, qb), "net outlines cross")
    landmarks = set().union(*(set(n["points"]) for n in nets))
    strokes = set().union(*(n["stroke"] for n in nets))
    _require(not (set(handles) & strokes), "handle intersects a net")
    payload = foreground - anchor_points - set(handles) - landmarks
    _require(all(0 < r < h - 1 and 0 < c < w - 1 for r, c in payload), "cargo must start inside canvas")
    _require(not payload & strokes, "cargo lies on a virtual outline")
    cargo, stationary = [], set()
    for component in _components(payload):
        memberships = []
        for p in component:
            memberships.append(tuple(i for i, n in enumerate(nets) if _inside(p, n["polygon"])))
        _require(len(set(memberships)) == 1, "cargo straddles an enclosure")
        enclosing = memberships[0]
        if enclosing:
            owner = min(enclosing, key=lambda i: nets[i]["area2"])
            _require(sum(nets[i]["area2"] == nets[owner]["area2"] for i in enclosing) == 1,
                     "enclosure ownership tie")
            nets[owner]["cargo"].update(component)
        else:
            owner = None
            stationary.update(component)
        cargo.append({"points": component, "owner": owner})
    return grid, {"background": background, "anchors": anchors, "nets": nets,
                  "cargo": cargo, "stationary": stationary}


def _render(grid, scene, mode="rule"):
    """Alternative modes are diagnostic ablations, never selected by solve."""
    bg, nets = scene["background"], scene["nets"]
    h, w = grid.shape
    output = np.full_like(grid, bg)
    occupied = {}

    def put(p, color):
        if 0 <= p[0] < h and 0 <= p[1] < w:
            _require(p not in occupied, "final footprints collide")
            occupied[p] = color
            output[p] = color

    for color, ps in scene["anchors"].items():
        for p in ps:
            put(p, color)
    deltas = []
    for net in nets:
        d = net["delta"] if mode != "no_docking" else (0, 0)
        deltas.append(d)
        pixels = set(net["points"]) if mode == "no_completion" else (
            net["all_links"] if mode == "all_chords" else net["stroke"])
        for r, c in pixels:
            put((r + d[0], c + d[1]), net["color"])
        r, c = net["handle"]
        put((r + d[0], c + d[1]), net["marker"])
    for cargo in scene["cargo"]:
        owner = cargo["owner"]
        if mode == "no_capture":
            owner = None
        elif mode == "boxes":
            choices = [i for i, net in enumerate(nets) if all(
                min(p[k] for p in net["points"]) < q[k] < max(p[k] for p in net["points"])
                for q in cargo["points"] for k in (0, 1))]
            owner = min(choices, key=lambda i: nets[i]["area2"]) if choices else None
        elif mode == "outermost" and owner is not None:
            choices = [i for i, net in enumerate(nets) if all(_inside(p, net["polygon"]) for p in cargo["points"])]
            owner = max(choices, key=lambda i: nets[i]["area2"])
        d = (0, 0) if owner is None else deltas[owner]
        for r, c in cargo["points"]:
            put((r + d[0], c + d[1]), int(grid[r, c]))
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, scene = _decode(input_grid)
    return _render(grid, scene)


def _transform(points, code):
    ps = []
    for r, c in points:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        ps.append((r, c))
    lo = min(r for r, c in ps), min(c for r, c in ps)
    return [(r - lo[0], c - lo[1]) for r, c in ps]


def _geometry(rng, kind="any", scale=8):
    for _ in range(120):
        choice = kind if kind != "any" else str(rng.choice(["rectangle", "diamond", "bevel", "notch"]))
        h, w = int(rng.integers(4, scale + 1)), int(rng.integers(4, scale + 1))
        if choice == "notch":
            h, w = max(7, h), max(8, w)
            x = int(rng.integers(1, w - 5 + 1))
            y = int(rng.integers(x + 4, w))
            d = int(rng.integers(4, h - 1))
            ps = [(0, 0), (0, x), (d, x), (d, y), (0, y), (0, w), (h, w), (h, 0)]
        elif choice == "diamond":
            radius = int(rng.integers(2, max(3, scale // 2 + 1)))
            ps = [(0, radius), (radius, 2 * radius), (2 * radius, radius), (radius, 0)]
        elif choice == "bevel":
            cuts = [int(rng.integers(0, max(2, min(h, w) // 2))) for _ in range(4)]
            a, b, c, d = cuts
            ps = [(0, a), (0, w - b), (b, w), (h - c, w),
                  (h, w - c), (h, d), (h - d, 0), (a, 0)]
            ps = [p for i, p in enumerate(ps) if p != ps[i - 1]]
        else:
            if rng.random() < .3:
                w = h
            ps = [(0, 0), (0, w), (h, w), (h, 0)]
        ps = _transform(ps, int(rng.integers(8)))
        try:
            net = _contour(ps)
            if choice == "notch" and not net["concave"]:
                continue
            return net
        except ValueError:
            continue
    raise ValueError("geometry sampling exhausted")


def _shift_net(net, origin):
    result = dict(net)
    for name in ("points", "polygon", "stroke", "all_links"):
        ps = [(r + origin[0], c + origin[1]) for r, c in net[name]]
        result[name] = set(ps) if name in ("stroke", "all_links") else ps
    return result


def _place_nets(rng, layout, count, scale):
    if layout == "nested":
        outer_span = int(rng.integers(13, 17)) if count == 2 else int(rng.integers(19, 24))
        hspan, wspan = outer_span, outer_span + int(rng.integers(-1, 2))
        h, w = int(rng.integers(hspan + 4, 31)), int(rng.integers(wspan + 4, 31))
        r, c = int(rng.integers(2, h - hspan - 1)), int(rng.integers(2, w - wspan - 1))
        outer = _contour([(r, c), (r, c + wspan), (r + hspan, c + wspan), (r + hspan, c)])
        nets = [outer]
        if count == 3:
            mid = int(rng.integers(10, 14))
            mr, mc = r + int(rng.integers(2, hspan - mid - 1)), c + int(rng.integers(2, wspan - mid - 1))
            nets.append(_contour([(mr, mc), (mr, mc + mid), (mr + mid, mc + mid), (mr + mid, mc)]))
        parent = nets[-1]
        low = np.min(parent["points"], axis=0)
        high = np.max(parent["points"], axis=0)
        child = _geometry(rng, str(rng.choice(["rectangle", "diamond", "bevel"])), 5 if count == 3 else 7)
        span = np.max(child["points"], axis=0)
        origin = tuple(int(rng.integers(int(low[k]) + 2, int(high[k] - span[k]) - 1)) for k in (0, 1))
        nets.append(_shift_net(child, origin))
        return h, w, nets
    geometries = [_geometry(rng, "notch" if layout == "notch" else "any", scale) for _ in range(count)]
    spans = [np.max(n["points"], axis=0) for n in geometries]
    if count == 1:
        h, w = (int(rng.integers(int(spans[0][k]) + 4, 31)) for k in (0, 1))
    else:
        h, w = int(rng.integers(20, 31)), int(rng.integers(20, 31))
    nets, filled = [], set()
    for net, span in zip(geometries, spans):
        options = [(r, c) for r in range(2, h - int(span[0]) - 1)
                   for c in range(2, w - int(span[1]) - 1)]
        rng.shuffle(options)
        for origin in options:
            moved = _shift_net(net, origin)
            region = moved["stroke"] | {(r, c) for r in range(h) for c in range(w)
                                         if _inside((r, c), moved["polygon"])}
            if not region & _halo(filled, True):
                nets.append(moved)
                filled.update(region)
                break
        else:
            raise ValueError("net placement exhausted")
    return h, w, nets


def _cargo_shape(rng):
    kind = int(rng.integers(5))
    if kind == 0:
        return [(0, 0)]
    if kind == 1:
        length = int(rng.integers(2, 6))
        return _transform([(0, c) for c in range(length)], int(rng.integers(8)))
    if kind == 2:
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        return [(r, c) for r in range(h) for c in range(w)]
    cells = {(0, 0)}
    for _ in range(int(rng.integers(2, 7))):
        frontier = sorted(_halo(cells) - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return sorted(cells)


def _build(rng, plan, palette):
    h, w, nets = _place_nets(rng, plan["layout"], plan["count"], plan["scale"])
    bg = int(palette[0])
    source_colors = list(map(int, palette[1:1 + len(nets)]))
    markers = list(map(int, palette[1 + len(nets):1 + 2 * len(nets)]))
    payload_colors = list(map(int, palette[1 + 2 * len(nets):]))
    inp = np.full((h, w), bg, np.int8)
    all_points = set().union(*(set(n["points"]) for n in nets))
    all_strokes = set().union(*(n["stroke"] for n in nets))
    handles = []
    for net, color, marker in zip(nets, source_colors, markers):
        candidates = sorted({(r + dr, c + dc) for r, c in net["points"] for dr, dc in NEIGHBORS
                             if 0 < r + dr < h - 1 and 0 < c + dc < w - 1
                             and (r + dr, c + dc) not in all_strokes
                             and sum(max(abs(r + dr - a), abs(c + dc - b)) == 1 for a, b in all_points) == 1
                             and (r + dr, c + dc) not in _halo(handles, True)})
        _require(candidates, "no visible unambiguous handle")
        handle = candidates[int(rng.integers(len(candidates)))]
        handles.append(handle)
        net.update(color=color, marker=marker, handle=handle, cargo=set())
        for p in net["points"]:
            inp[p] = color
        inp[handle] = marker
    ownership = {}
    safe = set()
    forbidden = all_strokes | _halo(all_points | set(handles), True)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            p = r, c
            if p in forbidden:
                continue
            safe.add(p)
            enclosing = [i for i, n in enumerate(nets) if _inside(p, n["polygon"])]
            ownership[p] = min(enclosing, key=lambda i: nets[i]["area2"]) if enclosing else None
    cargo, used = [], set()

    def add_cargo(owner, candidates=None, singleton=False, color=None):
        allowed = safe - _halo(used, True)
        positions = sorted(p for p in allowed if ownership[p] == owner and (candidates is None or p in candidates))
        rng.shuffle(positions)
        shapes = [[(0, 0)]] if singleton else [_cargo_shape(rng) for _ in range(5)] + [[(0, 0)]]
        for shape in shapes:
            for r, c in positions:
                cells = {(r + dr, c + dc) for dr, dc in shape}
                if all(p in allowed and ownership[p] == owner for p in cells):
                    base = payload_colors[0] if color is None else color
                    for p in sorted(cells):
                        inp[p] = base
                    if len(cells) >= 4 and rng.random() < .4:
                        p = sorted(cells)[int(rng.integers(len(cells)))]
                        inp[p] = int(rng.choice(payload_colors[1:]))
                    cargo.append({"points": cells, "owner": owner})
                    used.update(cells)
                    if owner is not None:
                        nets[owner]["cargo"].update(cells)
                    return True
        return False

    if plan["layout"] == "notch":
        net = nets[0]
        low, high = np.min(net["points"], axis=0), np.max(net["points"], axis=0)
        recess = {p for p in safe if ownership[p] is None and all(low[k] < p[k] < high[k] for k in (0, 1))}
        _require(add_cargo(None, recess, True), "no notch-recess witness")
    # All nets have captured cargo, and every scene has uncaptured cargo in the same palette.
    for i in range(len(nets)):
        _require(add_cargo(i), "net lacks captured cargo")
        for _ in range(int(rng.integers(0, 3 if plan["test"] else 2))):
            add_cargo(i, color=int(rng.choice(payload_colors)))
    _require(add_cargo(None, singleton=True), "no stationary outside cargo")
    for _ in range(int(rng.integers(0, 4))):
        add_cargo(None, color=int(rng.choice(payload_colors)))
    _require(np.count_nonzero(inp == bg) > inp.size / 2, "crowded source")
    fixed = {p: int(inp[p]) for item in cargo if item["owner"] is None for p in item["points"]}
    final = set(fixed)
    anchors = {}
    source_foreground = {tuple(map(int, p)) for p in np.argwhere(inp != bg)}
    # Choose actual docks from the complete inferred net+cargo footprint, not from landmarks alone.
    order = sorted(range(len(nets)), key=lambda i: -nets[i]["area2"])
    for i in order:
        net = nets[i]
        axes = [plan["axes"][i]] if plan.get("axes") is not None else list(map(int, rng.permutation(2)))
        options = []
        footprint = net["stroke"] | {net["handle"]} | net["cargo"]
        for axis in axes:
            for guide in range(1, (h, w)[axis] - 1):
                d = [0, 0]
                d[axis] = guide - net["handle"][axis]
                if d == [0, 0] and not plan.get("aligned", False):
                    continue
                target = {(r + d[0], c + d[1]) for r, c in footprint}
                inside = {p for p in target if 0 <= p[0] < h and 0 <= p[1] < w}
                clipped = len(inside) < len(target)
                if i == 0 and clipped != plan["clip"]:
                    continue
                if i != 0 and clipped and not plan["clip"]:
                    continue
                if inside & final:
                    continue
                moving_cargo = {(r + d[0], c + d[1]) for r, c in net["cargo"]}
                if not any(0 <= r < h and 0 <= c < w for r, c in moving_cargo):
                    continue
                for ends in ((0,), (1,), (0, 1)):
                    ps = {(guide, 0 if e == 0 else w - 1) if axis == 0 else
                          (0 if e == 0 else h - 1, guide) for e in ends}
                    if ps & (inside | final | _halo(source_foreground)):
                        continue
                    # Existing anchors must remain isolated when a new dock is added.
                    if ps & _halo(set().union(*anchors.values()) if anchors else set()):
                        continue
                    options.append((axis, guide, tuple(d), inside, ps))
        _require(options, "no collision-free dock")
        axis, guide, delta, target, ps = options[int(rng.integers(len(options)))]
        net.update(axis=axis, guide=guide, delta=delta)
        anchors[net["marker"]] = ps
        final.update(target | ps)
        source_foreground.update(ps)
        for p in ps:
            inp[p] = net["marker"]
    scene = {"background": bg, "anchors": anchors, "nets": nets, "cargo": cargo,
             "stationary": set(fixed)}
    output = _render(inp, scene)
    # Grid-only reconstruction must agree with the construction's intended topology and ownership.
    decoded_grid, decoded = _decode(inp)
    _require(np.array_equal(_render(decoded_grid, decoded), output), "construction disagrees with input inference")
    pair = {"input": inp, "output": output}
    stats = _features(pair)
    _require(stats["moving_cargo"] and stats["new_stroke"] and stats["erased"], "uninformative scene")
    if plan["layout"] == "notch":
        _require(stats["boxes_differ"], "recess is not observable after docking")
    if plan["layout"] == "nested":
        _require(stats["outermost_differs"], "innermost ownership not observable")
    return pair


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _alternative_differs(grid, scene, output, mode):
    try:
        return not np.array_equal(_render(grid, scene, mode), output)
    except ValueError:
        # A shortcut creating a collision is also incompatible with the example.
        return True


def _features(pair):
    grid, scene = _decode(pair["input"])
    output = _grid(pair["output"])
    _require(np.array_equal(_render(grid, scene), output), "wrong net transport output")
    nets, bg = scene["nets"], scene["background"]
    # Explicit conservation checks supplement full reconstruction. Distinct role
    # palettes and collision-free destinations make these counts unambiguous.
    color_counts = Counter()
    for color, ps in scene["anchors"].items():
        _require(all(output[p] == color for p in ps), "fixed anchor moved")
        color_counts[color] += len(ps)
    for p in scene["stationary"]:
        _require(output[p] == grid[p], "uncaptured cargo changed")
        color_counts[int(grid[p])] += 1
    for net in nets:
        dr, dc = net["delta"]
        for p in net["stroke"] | net["cargo"] | {net["handle"]}:
            q = p[0] + dr, p[1] + dc
            color = net["color"] if p in net["stroke"] else int(grid[p])
            if 0 <= q[0] < grid.shape[0] and 0 <= q[1] < grid.shape[1]:
                _require(output[q] == color, "net geometry or cargo color changed")
                color_counts[color] += 1
    _require(all(np.count_nonzero(output == c) == count for c, count in color_counts.items()),
             "unexpected material duplication or deletion")
    moving_cargo = new_stroke = clipped = 0
    depths = []
    for net in nets:
        dr, dc = net["delta"]
        moving_cargo += sum((dr or dc) and 0 <= r + dr < grid.shape[0] and 0 <= c + dc < grid.shape[1]
                            for r, c in net["cargo"])
        new_stroke += sum(0 <= r + dr < grid.shape[0] and 0 <= c + dc < grid.shape[1] and
                          grid[r + dr, c + dc] == bg for r, c in net["stroke"] - set(net["points"]))
        footprint = net["stroke"] | net["cargo"] | {net["handle"]}
        clipped += any(not (0 <= r + dr < grid.shape[0] and 0 <= c + dc < grid.shape[1]) for r, c in footprint)
        depths.append(1 + sum(all(_inside(p, other["polygon"]) for p in net["points"])
                              for other in nets if other is not net))
    return {"shape": grid.shape, "background": bg, "nets": len(nets), "depth": max(depths),
            "span": max(max(max(p[k] for p in n["points"]) - min(p[k] for p in n["points"])
                            for k in (0, 1)) for n in nets),
            "concave": sum(n["concave"] for n in nets), "chords": sum(n["chords"] for n in nets),
            "axes": {n["axis"] for n in nets}, "clipped": clipped,
            "moving_cargo": moving_cargo, "new_stroke": new_stroke,
            "erased": int(np.count_nonzero((grid != bg) & (output == bg))),
            "cargo_sizes": [len(c["points"]) for c in scene["cargo"]],
            "stationary": len(scene["stationary"]),
            "boxes_differ": _alternative_differs(grid, scene, output, "boxes"),
            "outermost_differs": _alternative_differs(grid, scene, output, "outermost"),
            "all_chords_differ": _alternative_differs(grid, scene, output, "all_chords"),
            "no_A_differs": _alternative_differs(grid, scene, output, "no_completion"),
            "no_B_differs": _alternative_differs(grid, scene, output, "no_docking"),
            "no_capture_differs": _alternative_differs(grid, scene, output, "no_capture")}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, features = set(), {"train": [], "test": []}
    for split in features:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas must be preserved")
            signature = _signature(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            f = _features(pair)
            _require(f["moving_cargo"] and f["new_stroke"] and f["erased"] and f["stationary"],
                     "missing completion, cargo movement, erasure, or outside preservation")
            _require(f["no_A_differs"] and f["no_B_differs"] and f["no_capture_differs"], "parent ablation has no effect")
            features[split].append(f)
    train = features["train"]
    _require(any(f["concave"] and f["boxes_differ"] for f in train), "no demonstrated concave capture")
    _require(any(f["depth"] >= 2 and f["outermost_differs"] for f in train), "no demonstrated innermost ownership")
    _require(any(f["chords"] and f["all_chords_differ"] for f in train), "no demonstrated chord rejection")
    _require(set().union(*(f["axes"] for f in train)) == {0, 1}, "both docking axes need evidence")
    if any(f["clipped"] for f in features["test"]):
        _require(any(f["clipped"] for f in train), "clipping needs training evidence")
    _require(all(f["nets"] > max(t["nets"] for t in train) or
                 f["depth"] > max(t["depth"] for t in train) or
                 f["span"] > max(t["span"] for t in train) for f in features["test"]),
             "query lacks the intended structural shift")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or high is not None and value > high:
        raise ValueError(f"illegal {name}")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    a = _integer(num_train, "num_train", 2, 10)
    b = _integer(num_test, "num_test", 1, 4)
    _require(a > b, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .3)
    base_palette = rng.permutation(10)
    shift = int(rng.integers(3))  # larger net, three independent nets, deeper nesting
    clipping = bool(rng.random() < .6)
    first_axis = int(rng.integers(2))
    plans = [{"layout": "notch", "count": 1, "scale": 9, "clip": clipping,
              "axes": [first_axis], "test": False},
             {"layout": "nested", "count": 2, "scale": 7, "clip": False,
              "axes": [1 - first_axis, first_axis], "test": False}]
    for _ in range(a - 2):
        count = int(rng.integers(1, 3))
        plans.append({"layout": str(rng.choice(["scattered", "nested"])) if count == 2 else "scattered",
                      "count": count, "scale": int(rng.integers(5, 10)), "clip": clipping and rng.random() < .3,
                      "axes": None, "test": False, "aligned": rng.random() < .2})
    # Randomize demonstrations, preserving their jointly required evidence.
    rng.shuffle(plans)
    for _ in range(b):
        plans.append({"layout": "notch" if shift == 0 else "scattered" if shift == 1 else "nested",
                      "count": 1 if shift == 0 else 3, "scale": 22 if shift == 0 else 7,
                      "clip": clipping and rng.random() < .65, "axes": None, "test": True,
                      "large": shift == 0})
    puzzle, seen = {"train": [], "test": []}, set()
    for index, plan in enumerate(plans):
        palette = base_palette if stable else rng.permutation(10)
        for _ in range(250):
            try:
                pair = _build(rng, plan, palette)
                f = _features(pair)
                if plan.get("large") and f["span"] <= 16:
                    continue
                if index < a and f["span"] > 16:
                    continue
                signature = _signature(pair["input"])
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle["train" if index < a else "test"].append(pair)
                break
            except ValueError:
                continue
        else:
            raise RuntimeError(f"bounded scene construction exhausted: {plan}")
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded 42-seed diversity, ablation, metamorphic, and negative checks."""
    dimensions, structures, backgrounds, shifts, features = set(), set(), set(), set(), []
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        train = [_features(p) for p in puzzle["train"]]
        test = [_features(p) for p in puzzle["test"]]
        if test[0]["depth"] > max(f["depth"] for f in train):
            shifts.add("nesting")
        elif test[0]["nets"] > max(f["nets"] for f in train):
            shifts.add("count")
        else:
            shifts.add("scale")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                f = _features(pair)
                features.append(f)
                dimensions.add(f["shape"])
                backgrounds.add(f["background"])
                roles = {}
                norm = tuple(tuple(roles.setdefault(int(c), len(roles)) for c in row) for row in pair["input"])
                structures.add(norm)
                inp = pair["input"]
                # Exact whole-grid rotations, reflections, and arbitrary palette permutations.
                perm = np.random.default_rng(seed + 100).permutation(10)
                transformed = perm[np.rot90(np.fliplr(inp))].astype(np.int8)
                expected = perm[np.rot90(np.fliplr(pair["output"]))].astype(np.int8)
                _require(np.array_equal(solve(transformed), expected), "geometric/palette metamorphism failed")
    _require(len(signatures) == 42 and len(structures) >= 150 and len(dimensions) >= 35, "insufficient structural variation")
    _require(backgrounds == set(range(10)) and shifts == {"nesting", "count", "scale"}, "palette or shift coverage")
    _require({1, 2, 3} <= {f["nets"] for f in features}, "net count coverage")
    _require(any(f["clipped"] for f in features) and any(not f["clipped"] for f in features), "clipping coverage")
    _require(max(n for f in features for n in f["cargo_sizes"]) >= 5, "cargo geometry variation")
    # Determinism at a nondefault count and every legal explicit count.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1234, a, b))
    p = generate_puzzle(99, 3, 2)
    q = generate_puzzle(99, 3, 2)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ("input", "output")), "seed is not reproducible")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # Corrupt a fixed anchor, a transported cargo cell, and a vacated source cell.
    original = p["train"][0]
    grid, scene = _decode(original["input"])
    anchor = next(iter(next(iter(scene["anchors"].values()))))
    cargo_target = next((r + n["delta"][0], c + n["delta"][1])
                        for n in scene["nets"] for r, c in n["cargo"]
                        if 0 <= r + n["delta"][0] < grid.shape[0] and
                        0 <= c + n["delta"][1] < grid.shape[1])
    vacated = tuple(np.argwhere((grid != scene["background"]) &
                               (original["output"] == scene["background"]))[0])
    for location in (anchor, cargo_target, vacated):
        corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        value = int(corrupted["train"][0]["output"][location])
        corrupted["train"][0]["output"][location] = (value + 1) % 10
        try:
            validate_generated_puzzle(corrupted)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(dimensions),
            "color_normalized_inputs": len(structures), "shifts": sorted(shifts)}
