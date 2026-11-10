"""Anchored glyph routes with a pixel-sized, cancelling companion shadow.

Only standard-library modules and NumPy are used.  The parser and solver operate
on visible input cells; generation metadata is never used to choose an answer.
"""

from collections import deque
import json

import numpy as np


OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sign(x):
    return (x > 0) - (x < 0)


def _segment(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require((dr or dc) and (dr == 0 or dc == 0 or abs(dr) == abs(dc)),
             "unaligned segment")
    return tuple((a[0] + k * _sign(dr), a[1] + k * _sign(dc))
                 for k in range(max(abs(dr), abs(dc)) + 1))


def _edges(points):
    occupied = set(points)
    result = []
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            b = points[j]
            dr, dc = b[0] - a[0], b[1] - a[1]
            if dr == 0 or dc == 0 or abs(dr) == abs(dc):
                if not (set(_segment(a, b)[1:-1]) & occupied):
                    result.append((i, j))
    return result


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersects(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return (_cross(a, b, c) * _cross(a, b, d) <= 0
            and _cross(c, d, a) * _cross(c, d, b) <= 0)


def _route(points):
    """Unique simple spanning cycle first, otherwise unique spanning open path.

    Edge-set equivalence removes arbitrary starting point and reversal. Crossing
    is geometric, including crossings between two diagonal pixels at half cells.
    """
    points = tuple(points)
    n = len(points)
    _require(2 <= n <= 10 and len(set(points)) == n, "need 2..10 distinct vertices")
    edges = _edges(points)
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    _require(all(adjacency), "disconnected landmarks")
    conflicts = [set() for _ in edges]
    for k, (i, j) in enumerate(edges):
        for l, (s, t) in enumerate(edges[:k]):
            if not {i, j} & {s, t} and _intersects(points[i], points[j], points[s], points[t]):
                conflicts[k].add(l)
                conflicts[l].add(k)

    def search(closed):
        found = set()

        def visit(order, visited, selected):
            if len(found) > 1:
                return
            if len(order) == n:
                chosen = selected
                if closed:
                    last = [e for v, e in adjacency[order[-1]] if v == order[0]]
                    if not last or conflicts[last[0]] & selected:
                        return
                    chosen = selected | {last[0]}
                found.add(tuple(sorted(chosen)))
                return
            for v, e in adjacency[order[-1]]:
                if v not in visited and not conflicts[e] & selected:
                    visit(order + (v,), visited | {v}, selected | {e})

        for start in ((0,) if closed else range(n)):
            visit((start,), {start}, set())
            if len(found) > 1:
                break
        return found

    cycles = search(True) if n >= 3 else set()
    _require(len(cycles) <= 1, "competing closed routes")
    found = cycles or search(False)
    _require(len(found) == 1, "no unique simple route")
    selected = tuple(edges[k] for k in next(iter(found)))
    sites = set().union(*(set(_segment(points[i], points[j])) for i, j in selected))
    all_sites = set().union(*(set(_segment(points[i], points[j])) for i, j in edges))
    _require(len(sites) > n, "route must contain a missing lattice site")
    directions = {("H" if points[i][0] == points[j][0] else
                   "V" if points[i][1] == points[j][1] else "D") for i, j in selected}
    return {"points": points, "edges": selected, "sites": sites,
            "closed": bool(cycles), "chord_sites": all_sites - sites,
            "directions": directions}


def _components(grid, color):
    remaining = {tuple(map(int, p)) for p in np.argwhere(grid == color)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, queue = [start], deque([start])
        while queue:
            r, c = queue.popleft()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    component.append(q)
                    queue.append(q)
        result.append(component)
    return result


def _bbox(points):
    return (min(r for r, c in points), min(c for r, c in points),
            max(r for r, c in points), max(c for r, c in points))


def _concave(route):
    if not route["closed"]:
        return False
    points = route["points"]
    neighbors = {i: [] for i in range(len(points))}
    for i, j in route["edges"]:
        neighbors[i].append(j)
        neighbors[j].append(i)
    order = [0, min(neighbors[0])]
    while len(order) < len(points):
        order.append(next(i for i in neighbors[order[-1]] if i != order[-2]))
    signs = {_sign(_cross(points[order[i - 1]], points[order[i]], points[order[(i + 1) % len(order)]]))
             for i in range(len(order))}
    return -1 in signs and 1 in signs


def _gap(a, b):
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _decode(input_grid):
    inp = np.asarray(input_grid)
    _require(inp.ndim == 2 and all(1 <= x <= 30 for x in inp.shape), "invalid grid dimensions")
    _require(np.issubdtype(inp.dtype, np.integer) and np.all((0 <= inp) & (inp <= 9)),
             "invalid cell colors")
    values, counts = np.unique(inp, return_counts=True)
    _require(len(values) == 3 and counts.max() > inp.size // 2, "need three roles and dominant background")
    bg = int(values[np.argmax(counts)])
    fg = [int(c) for c in values if c != bg]
    components = {c: _components(inp, c) for c in fg}
    candidates = [c for c in fg if len(components[c]) == 2
                  and sorted(map(len, components[c]))[0] == 1
                  and sorted(map(len, components[c]))[1] >= 3]
    _require(len(candidates) == 1, "template must be uniquely recognizable")
    ink = candidates[0]
    line = next(c for c in fg if c != ink)
    glyph = next(part for part in components[ink] if len(part) > 1)
    marker = next(part[0] for part in components[ink] if len(part) == 1)
    _require(all(len(part) == 1 for part in components[line]), "landmarks must be isolated")
    vertices = sorted(part[0] for part in components[line])
    tr, tc, br, bc = _bbox(glyph)
    h, w = br - tr + 1, bc - tc + 1
    _require(2 <= h <= 5 and 2 <= w <= 5, "template tight dimensions must be 2..5")
    mask = inp[tr:br + 1, tc:bc + 1] == ink
    pr, pc, er, ec = _bbox(vertices)
    _require(all((r - pr) % 2 == 0 and (c - pc) % 2 == 0 for r, c in vertices),
             "landmarks must lie on pitch-two lattice")
    neighbors = [p for p in vertices if max(abs(p[0] - marker[0]), abs(p[1] - marker[1])) == 1]
    _require(len(neighbors) == 1, "marker needs exactly one neighboring vertex")
    marked = neighbors[0]
    offset = marker[0] - marked[0], marker[1] - marked[1]
    anchor = (marked[0] - pr) // 2, (marked[1] - pc) // 2
    _require(_gap((tr, tc, br, bc), _bbox(vertices + [marker])), "template and diagram need a blank band")
    points = tuple(((r - pr) // 2, (c - pc) // 2) for r, c in vertices)
    route = _route(points)
    return {"shape": inp.shape, "bg": bg, "ink": ink, "line": line,
            "mask": mask, "template_top": (tr, tc), "anchor": anchor,
            "offset": offset, "route": route, "diagram_points": vertices + [marker]}


def _layers(info, sites=None, offset=None, mask=None, anchor=None):
    """Rasterize the two layers before any collision rule or coloring is applied."""
    mask = info["mask"] if mask is None else mask
    h, w = mask.shape
    sites = info["route"]["sites"] if sites is None else sites
    anchor = info["anchor"] if anchor is None else anchor
    offset = info["offset"] if offset is None else offset
    tr, tc = info["template_top"]
    pixels = tuple(tuple(map(int, p)) for p in np.argwhere(mask))
    source, shadow = {}, {}
    for r, c in sorted(sites):
        top, left = tr + (r - anchor[0]) * h, tc + (c - anchor[1]) * w
        special = (r, c) == anchor
        for mr, mc in pixels:
            p = top + mr, left + mc
            q = p[0] + offset[0], p[1] + offset[1]
            _require(all(0 <= x < n for x, n in zip(p, info["shape"]))
                     and all(0 <= x < n for x, n in zip(q, info["shape"])), "clipped layer")
            _require(p not in source and q not in shadow, "same-layer tile overlap")
            source[p] = info["line"] if special else info["ink"]
            shadow[q] = info["ink"] if special else info["line"]
    return source, shadow


def _render(info, sites=None, offset=None, mask=None, anchor=None, mode="cancel"):
    source, shadow = _layers(info, sites, offset, mask, anchor)
    if mode == "no_anchor_inversion":
        source = dict.fromkeys(source, info["ink"])
        shadow = dict.fromkeys(shadow, info["line"])
    out = np.full(info["shape"], info["bg"], dtype=np.int8)
    common = set(source) & set(shadow)
    for p, color in source.items():
        if mode == "source_only" or p not in common or mode == "source_priority":
            out[p] = color
    if mode != "source_only":
        for p, color in shadow.items():
            if p not in common:
                out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the complete anchored, cancelling glyph mosaic from input only."""
    return _render(_decode(input_grid))


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _transform(points, code):
    transformed = []
    for r, c in points:
        if code & 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        transformed.append((r, c))
    r0, c0, _, _ = _bbox(transformed)
    return [(r - r0, c - c0) for r, c in transformed]


def _asymmetric(mask):
    return all(not np.array_equal(mask, other) for other in
               (np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2),
                mask.T, np.flipud(mask).T, np.fliplr(mask).T, np.rot90(mask, 2).T))


def _mask(rng, h, w, informative):
    # A randomized spanning monotone path guarantees connectedness and a tight
    # box. Frontier growth varies density rather than selecting a motif bank.
    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True
    steps = [0] * (h - 1) + [1] * (w - 1)
    rng.shuffle(steps)
    for axis in steps:
        r += axis == 0
        c += axis == 1
        mask[r, c] = True
    goal = int(rng.integers(int(mask.sum()), h * w + 1))
    while int(mask.sum()) < goal:
        frontier = sorted({(a, b) for r, c in np.argwhere(mask)
                           for a, b in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                           if 0 <= a < h and 0 <= b < w and not mask[a, b]})
        mask[_pick(rng, frontier)] = True
    if informative and (mask.all() or not _asymmetric(mask)):
        mask[:] = True
        mask[_pick(rng, ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))] = False
    if rng.integers(2):
        mask = np.flipud(mask)
    if rng.integers(2):
        mask = np.fliplr(mask)
    return mask.copy()


def _polygon(rng, concave=False):
    if concave:
        height, width = int(rng.integers(3, 7)), int(rng.integers(4, 7))
        if rng.random() < .55 or height < 4:
            depth = int(rng.integers(1, max(2, (width - 1) // 2)))
            start = int(rng.integers(1, width - 2 * depth))
            return [(0, 0), (0, start), (depth, start + depth), (0, start + 2 * depth),
                    (0, width), (height, width), (height, 0)]
        start = int(rng.integers(1, width - 1))
        end = int(rng.integers(start + 1, width))
        depth = int(rng.integers(1, height))
        return [(0, 0), (0, start), (depth, start), (depth, end),
                (0, end), (0, width), (height, width), (height, 0)]
    height, width = int(rng.integers(2, 7)), int(rng.integers(2, 7))
    style = int(rng.integers(3))
    if style == 0:
        width = height
    cuts = [0] * 4 if style == 0 else [int(rng.integers(0, min(height, width) // 2 + 1)) for _ in range(4)]
    a, b, c, d = cuts
    points = [(0, a), (0, width - b), (b, width), (height - c, width),
              (height, width - c), (height, d), (height - d, 0), (a, 0)]
    return [p for i, p in enumerate(points) if p != points[i - 1]]


def _geometry(rng, dims, kind, chord=False, mixed=False, concave=False):
    # Full source+shadow bounds use <=29 cells; placement handles the remaining
    # one-cell displacement. Limit vertices by actual template dimensions.
    h, w = dims
    limit_r, limit_c = 29 // h - 1, 29 // w - 1
    for _ in range(180):
        if kind == "cycle":
            points = _polygon(rng, concave)
            closed = True
        else:
            closed = False
            if mixed:
                a, b = int(rng.integers(2, 5)), int(rng.integers(1, 4))
                points = [(0, 0), (0, a), (b, a + b)]
            else:
                n = int(rng.integers(2, 9))
                points = [(0, 0)]
                for _ in range(n - 1):
                    dr, dc = _pick(rng, OFFSETS)
                    length = int(rng.integers(1, 5))
                    points.append((points[-1][0] + length * dr, points[-1][1] + length * dc))
        points = _transform(points, int(rng.integers(8)))
        if len(set(points)) != len(points) or not 2 <= len(points) <= 10:
            continue
        if max(r for r, c in points) > limit_r or max(c for r, c in points) > limit_c:
            continue
        try:
            route = _route(points)
        except ValueError:
            continue
        intended = {tuple(sorted((i, i + 1))) for i in range(len(points) - 1)}
        if closed:
            intended.add((0, len(points) - 1))
        if set(route["edges"]) != intended or route["closed"] != closed:
            continue
        if chord and not route["chord_sites"]:
            continue
        if concave and not _concave(route):
            continue
        if mixed and ("D" not in route["directions"] or len(route["directions"]) < 2):
            continue
        return route
    # Guaranteed small fallbacks within all supported template-size bounds.
    points = ([(0, 0), (0, 1), (1, 2), (0, 3), (0, 4), (3, 4), (3, 0)] if concave else
              [(0, 0), (0, 2), (2, 2), (2, 0)] if kind == "cycle" else
              [(0, 0), (0, 2), (1, 3)] if mixed else [(0, 0), (0, 3)])
    return _route(points)


def _marker_options(route, offsets, previous_anchor):
    options = []
    vertices = [(2 * r, 2 * c) for r, c in route["points"]]
    for anchor in route["points"]:
        if previous_anchor is not None and (anchor[0] == previous_anchor[0] or anchor[1] == previous_anchor[1]):
            continue
        for dr, dc in offsets:
            marker = 2 * anchor[0] + dr, 2 * anchor[1] + dc
            if sum(max(abs(r - marker[0]), abs(c - marker[1])) == 1 for r, c in vertices) == 1:
                options.append((anchor, (dr, dc), marker))
    return options


def _place(rng, mask, route, anchor, offset, marker, colors):
    h, w = mask.shape
    pixels = tuple(tuple(map(int, p)) for p in np.argwhere(mask))
    source = {(h * (r - anchor[0]) + mr, w * (c - anchor[1]) + mc)
              for r, c in route["sites"] for mr, mc in pixels}
    shadow = {(r + offset[0], c + offset[1]) for r, c in source}
    # Include full template box to keep the input template and placement bounds
    # explicit, even when a corner of its mask is blank.
    out_box = _bbox(source | shadow | {(0, 0), (h - 1, w - 1)})
    raw_diagram = [(2 * r, 2 * c) for r, c in route["points"]] + [marker]
    dr0, dc0, dr1, dc1 = _bbox(raw_diagram)
    dh, dw = dr1 - dr0 + 1, dc1 - dc0 + 1
    diagram = [(r - dr0, c - dc0) for r, c in raw_diagram]
    placements = []
    for r in range(out_box[2] - 29, out_box[0] + 31 - dh):
        for c in range(out_box[3] - 29, out_box[1] + 31 - dw):
            box = (r, c, r + dh - 1, c + dw - 1)
            if not _gap((0, 0, h - 1, w - 1), box):
                continue
            union = (min(r, out_box[0]), min(c, out_box[1]),
                     max(box[2], out_box[2]), max(box[3], out_box[3]))
            if union[2] - union[0] < 30 and union[3] - union[1] < 30:
                placements.append((r, c, union))
    _require(bool(placements), "cannot place input diagram")
    if rng.random() < .45:
        areas = [(b[2] - b[0] + 1) * (b[3] - b[1] + 1) for _, _, b in placements]
        threshold = sorted(areas)[len(areas) // 5]
        placements = [p for p, area in zip(placements, areas) if area <= threshold]
    pr, pc, union = _pick(rng, placements)
    bh, bw = union[2] - union[0] + 1, union[3] - union[1] + 1
    padding_mode = int(rng.integers(4))
    gh = bh if padding_mode == 0 else 30 if padding_mode == 1 else int(rng.integers(bh, 31))
    gw = bw if padding_mode == 0 else 30 if padding_mode == 1 else int(rng.integers(bw, 31))
    foreground = int(mask.sum()) + len(diagram)
    # Keep the background strictly above half, rather than relying on which
    # foreground role happens to have the largest count.
    while gh * gw <= 2 * foreground:
        if gh <= gw and gh < 30:
            gh += 1
        else:
            gw += 1
    top = int(rng.integers(gh - bh + 1)) - union[0]
    left = int(rng.integers(gw - bw + 1)) - union[1]
    bg, ink, line = colors
    inp = np.full((gh, gw), bg, dtype=np.int8)
    for mr, mc in pixels:
        inp[top + mr, left + mc] = ink
    for i, (r, c) in enumerate(diagram):
        inp[top + pr + r, left + pc + c] = ink if i == len(diagram) - 1 else line
    # Labels are constructed from the sampled geometry, then checked against
    # the visible-input decoder by validation. No hidden order is supplied there.
    info = {"shape": inp.shape, "bg": bg, "ink": ink, "line": line,
            "mask": mask, "template_top": (top, left), "route": route,
            "anchor": anchor, "offset": offset}
    return {"input": inp, "output": _render(info)}


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    held = _pick(rng, OFFSETS)
    train_offsets = [d for d in OFFSETS if d != held]
    fixed_palette = bool(rng.random() < .20)
    base_colors = tuple(map(int, rng.choice(10, 3, replace=False)))
    concave_enabled = bool(rng.random() < .45)
    first_dims = first_anchor = None
    seen, puzzle = set(), {"train": [], "test": []}
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            teaching = split == "train" and index < 2
            chord = teaching and index == 0
            mixed = teaching and index == 1
            kind = "cycle" if chord else "path" if mixed else _pick(rng, ("cycle", "path"))
            concave = kind == "cycle" and concave_enabled and (chord or rng.random() < .35)
            dims = [(h, w) for h in range(2, 6) for w in range(2, 6)
                    if (not chord or h != w)
                    and (not mixed or (h != first_dims[0] and w != first_dims[1]))]
            if split == "test":
                offsets = [held]
            elif chord:
                offsets = [d for d in train_offsets if (d[0] == 0) != (d[1] == 0)]
            elif mixed:
                offsets = [d for d in train_offsets if d[0] and d[1]]
            else:
                offsets = train_offsets
            for attempt in range(100):
                h, w = _pick(rng, dims)
                mask = _mask(rng, h, w, teaching)
                route = _geometry(rng, (h, w), kind, chord, mixed, concave)
                options = _marker_options(route, offsets, first_anchor if mixed else None)
                if not options:
                    continue
                anchor, offset, marker = _pick(rng, options)
                colors = base_colors if fixed_palette else tuple(map(int, rng.choice(10, 3, replace=False)))
                try:
                    pair = _place(rng, mask, route, anchor, offset, marker, colors)
                except ValueError:
                    # A large diagram with an interior anchor can need more room
                    # than the expanded output alone. Resample the construction.
                    continue
                if _key(pair["input"]) in seen:
                    continue
                info = _decode(pair["input"])
                source, shadow = _layers(info)
                if teaching and not (set(source) & set(shadow)):
                    continue
                if teaching and set(map(int, np.unique(pair["output"]))) != set(colors):
                    continue
                if teaching and np.array_equal(_render(info, mode="no_anchor_inversion"), pair["output"]):
                    continue
                if teaching and np.array_equal(_render(info, mask=np.ones_like(mask)), pair["output"]):
                    continue
                break
            else:
                raise RuntimeError("bounded distinct-example construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if chord:
                first_dims, first_anchor = (h, w), anchor
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, out):
    info = _decode(inp)
    source, shadow = _layers(info)
    sites, points = info["route"]["sites"], info["route"]["points"]
    common = set(source) & set(shadow)
    occupancy = np.full(inp.shape, False)
    for p in set(source) ^ set(shadow):
        occupancy[p] = True
    assert np.array_equal(out != info["bg"], occupancy), "wrong collision occupancy"
    assert np.count_nonzero(out != info["bg"]) == 2 * (len(source) - len(common))
    assert len(source) == len(sites) * int(info["mask"].sum())
    assert np.array_equal(out, _render(info)), "wrong color, anchor, mask or shadow"
    assert np.any(inp != out), "identity pair"
    assert set(map(int, np.unique(out))) <= {info["bg"], info["ink"], info["line"]}
    h, w = info["mask"].shape
    coarse_offset = info["offset"][0] * h, info["offset"][1] * w
    # A coarse-shift alternative may fall outside the canvas. That remains a
    # distinguishing failure; it must never silently clip to become plausible.
    try:
        scaled_wrong = not np.array_equal(out, _render(info, offset=coarse_offset))
    except ValueError:
        scaled_wrong = True
    return {"info": info, "shape": inp.shape, "dims": (h, w),
            "offset": info["offset"], "anchor": info["anchor"],
            "background": info["bg"], "closed": info["route"]["closed"],
            "concave": _concave(info["route"]),
            "vertices": len(points), "sites": len(sites), "overlap": len(common),
            "missing": len(sites) - len(points), "directions": info["route"]["directions"],
            "chords": len(info["route"]["chord_sites"]),
            "chord_needed": not np.array_equal(out, _render(info, sites=sites | info["route"]["chord_sites"])),
            "asymmetric": _asymmetric(info["mask"]), "scaled_wrong": scaled_wrong,
            "completion_needed": not np.array_equal(out, _render(info, sites=set(points))),
            "mask_needed": not np.array_equal(out, _render(info, mask=np.ones_like(info["mask"]))),
            "anchor_inversion_needed": not np.array_equal(out, _render(info, mode="no_anchor_inversion")),
            "priority_wrong": not np.array_equal(out, _render(info, mode="source_priority")),
            "geometry": (info["route"]["closed"], tuple(sorted(points))),
            "mask_signature": (h, w, tuple(map(int, info["mask"].flat))),
            "boundary": any(r in (0, inp.shape[0] - 1) or c in (0, inp.shape[1] - 1)
                            for r, c in set(source) | set(shadow))}


def validate_generated_puzzle(puzzle):
    """Check pixel semantics, family constraints, and demonstration obligations."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            features[split].append(_features(inp, out))
    train = features["train"]
    assert any(f["closed"] and f["chords"] for f in train), "missing contour/chord demonstration"
    assert any(f["chord_needed"] for f in train), "all-alignment shortcut is not distinguished"
    assert any(not f["closed"] and "D" in f["directions"] and len(f["directions"]) >= 2 for f in train)
    assert len({f["dims"][0] for f in train}) >= 2 and len({f["dims"][1] for f in train}) >= 2
    assert len({f["anchor"][0] for f in train}) >= 2 and len({f["anchor"][1] for f in train}) >= 2
    assert any(f["asymmetric"] and f["dims"][0] != f["dims"][1] for f in train)
    assert any((f["offset"][0] == 0) != (f["offset"][1] == 0) for f in train)
    assert any(f["offset"][0] and f["offset"][1] for f in train)
    assert all(f["completion_needed"] for f in train), "A completion has no observable effect"
    assert sum(f["mask_needed"] for f in train) >= 2, "B exact-mask behavior not demonstrated"
    assert sum(f["anchor_inversion_needed"] for f in train) >= 2, "anchor color inversion not demonstrated"
    assert sum(f["priority_wrong"] and f["scaled_wrong"] for f in train) >= 2
    known = {f["offset"] for f in train}
    assert all(f["offset"] not in known for f in features["test"]), "missing held-out vector shift"
    if any(f["concave"] for f in features["test"]):
        assert any(f["concave"] for f in train), "query concavity not demonstrated"


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed variation audit plus semantic and API regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [_features(np.asarray(pair["input"]), np.asarray(pair["output"]))
            for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    assert len({_key(p["train"][0]["input"]) for p in puzzles}) == 42
    assert len({f["shape"] for f in rows}) >= 35
    assert len({f["geometry"] for f in rows}) >= 60
    assert len({f["mask_signature"] for f in rows}) >= 60
    assert {f["background"] for f in rows} == set(range(10))
    assert {f["offset"] for f in rows} == set(OFFSETS)
    assert {f["dims"][0] for f in rows} == {2, 3, 4, 5}
    assert {f["dims"][1] for f in rows} == {2, 3, 4, 5}
    assert len({f["sites"] for f in rows}) >= 12
    assert any(f["vertices"] >= 7 for f in rows)
    assert any(f["concave"] for f in rows)
    assert any(f["vertices"] == 2 for f in rows)
    assert any(f["boundary"] for f in rows)
    assert any(f["info"]["mask"].all() for f in rows)
    assert any(not f["info"]["mask"].all() for f in rows)
    assert _equal(puzzles[0], generate_puzzle(0))
    for nt in range(2, 11):
        for nq in range(1, min(nt - 1, 4) + 1):
            p = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicit topology counterfactual: a square's diagonals must be rejected.
    route = _route(((0, 0), (0, 3), (3, 3), (3, 0)))
    assert route["closed"] and len(route["chord_sites"]) == 4
    vertices = [(0, 0), (0, 1), (0, 3), (0, 5), (1, 5), (3, 5),
                (3, 4), (3, 2), (3, 0), (2, 0)]
    for n in (9, 10):
        route = _route(vertices[:n])
        assert route["closed"] and len(route["sites"]) == 16
        mask = np.array([[1, 0], [1, 1]], dtype=bool)
        anchor, offset, marker = _marker_options(route, [(0, -1)], None)[0]
        pair = _place(np.random.default_rng(n), mask, route, anchor, offset, marker, (0, 2, 6))
        assert np.array_equal(solve(pair["input"]), pair["output"])
    for puzzle in puzzles[:3]:
        for pair in puzzle["train"] + puzzle["test"]:
            for turns in range(4):
                # Whole-grid isometries rotate the mask and encoded vector too.
                a, b = np.rot90(pair["input"], turns), np.rot90(pair["output"], turns)
                assert np.array_equal(solve(a), b)
                assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
    wrong = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    wrong["test"][0]["output"][0, 0] = (int(wrong["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(wrong)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": len(rows), "canvas_shapes": len({f["shape"] for f in rows}),
            "route_geometries": len({f["geometry"] for f in rows}),
            "template_masks": len({f["mask_signature"] for f in rows}),
            "stamp_counts": sorted({f["sites"] for f in rows}),
            "offsets": sorted({f["offset"] for f in rows}),
            "completion_counterexamples": sum(f["completion_needed"] for f in rows),
            "mask_counterexamples": sum(f["mask_needed"] for f in rows)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
