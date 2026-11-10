"""Maximal substrate squares scope sparse drawing completion and translation.

0 is neutral substrate; 5 is an opaque obstacle. All other colors are ink
over neutral substrate. Only strict interiors of locally maximal substrate
squares are writable. See task_spec.md for the explicit layer convention.
No files, parent generators, seeds, or training examples are used by solve.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import numpy as np


INK = (1, 2, 3, 4, 6, 7, 8, 9)
OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=True)


def _squares(mask):
    """Solid squares, not extendable by any complete row or column.

    Window sums find squares directly; full-side tests implement maximal
    RECTANGLE eligibility, rather than merely inability to enlarge a square.
    """
    h, w = mask.shape
    integral = np.pad(mask.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    found = []
    for s in range(2, min(h, w) + 1):
        sums = integral[s:, s:] - integral[:-s, s:] - integral[s:, :-s] + integral[:-s, :-s]
        for r, c in np.argwhere(sums == s * s):
            r, c = int(r), int(c)
            if ((r and mask[r - 1, c:c + s].all())
                    or (r + s < h and mask[r + s, c:c + s].all())
                    or (c and mask[r:r + s, c - 1].all())
                    or (c + s < w and mask[r:r + s, c + s].all())):
                continue
            found.append((r, c, s))
    # This family deliberately excludes conflicting square scopes. Equal-size
    # DISJOINT ties are all selected; neither area nor reading order ranks them.
    for (r, c, s), (t, l, z) in combinations(found, 2):
        if max(r, t) < min(r + s, t + z) and max(c, l) < min(c + s, l + z):
            raise ValueError("overlapping maximal square scopes are outside this family")
    return found


def _aligned(a, b):
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr or dc) and (dr == 0 or dc == 0 or dr == dc)


def _line(a, b):
    if not _aligned(a, b):
        raise ValueError("unaligned endpoints")
    dr, dc = int(b[0] - a[0]), int(b[1] - a[1])
    n = max(abs(dr), abs(dc))
    sr, sc = (dr > 0) - (dr < 0), (dc > 0) - (dc < 0)
    return {(a[0] + i * sr, a[1] + i * sc) for i in range(n + 1)}


def _edges(points):
    occupied = set(points)
    return [(i, j) for i, j in combinations(range(len(points)), 2)
            if _aligned(points[i], points[j])
            and not ((_line(points[i], points[j]) - {points[i], points[j]}) & occupied)]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersect(a, b, c, d):
    if any(max(a[k], b[k]) < min(c[k], d[k]) or max(c[k], d[k]) < min(a[k], b[k])
           for k in (0, 1)):
        return False
    return _cross(a, b, c) * _cross(a, b, d) <= 0 and _cross(c, d, a) * _cross(c, d, b) <= 0


def _topology(points):
    """Unique noncrossing Hamiltonian cycle, else unique Hamiltonian path.

    Reversal and cycle start do not count as different completions. Crossings
    are tested continuously, including crossings halfway between grid cells.
    """
    if len(points) > 12:
        raise ValueError("at most 12 landmarks of a color per scope")
    edges = _edges(points)
    if not edges:
        return None
    adjacency = [[] for _ in points]
    for k, (i, j) in enumerate(edges):
        adjacency[i].append((j, k))
        adjacency[j].append((i, k))
    if not all(adjacency):
        raise ValueError("partly linked color group")
    conflicts = [set() for _ in edges]
    for k, (i, j) in enumerate(edges):
        for l, (u, v) in enumerate(edges[:k]):
            if not {i, j} & {u, v} and _intersect(points[i], points[j], points[u], points[v]):
                conflicts[k].add(l)
                conflicts[l].add(k)

    def search(closed):
        solutions, visits = set(), 0

        def visit(order, used, chosen):
            nonlocal visits
            visits += 1
            if visits > 20000:
                raise ValueError("topology search budget exceeded")
            if len(solutions) > 1:
                return
            if len(order) == len(points):
                selected = chosen
                if closed:
                    closing = [e for q, e in adjacency[order[-1]] if q == order[0]]
                    if not closing or conflicts[closing[0]] & chosen:
                        return
                    selected = chosen | {closing[0]}
                solutions.add(tuple(sorted(selected)))
                return
            for q, e in adjacency[order[-1]]:
                if q not in used and not conflicts[e] & chosen:
                    visit(order + [q], used | {q}, chosen | {e})

        for start in ([0] if closed else range(len(points))):
            visit([start], {start}, set())
            if len(solutions) > 1:
                break
        return solutions

    cycles = search(True) if len(points) >= 3 else set()
    possibilities = cycles if cycles else search(False)
    if len(possibilities) != 1:
        raise ValueError("ambiguous path or contour")
    chosen = [edges[k] for k in next(iter(possibilities))]
    pixels, long_pixels = set(), set()
    lengths = []
    for i, j in chosen:
        cells = _line(points[i], points[j])
        pixels |= cells
        length = len(cells) - 1
        lengths.append(length)
        if length > 1:
            long_pixels |= cells
    all_pixels = set().union(*(_line(points[i], points[j]) for i, j in edges))
    return {"points": points, "edges": chosen, "pixels": pixels,
            "long_pixels": long_pixels, "lengths": lengths,
            "kind": "cycle" if cycles else "path", "chord_pixels": all_pixels - pixels}


def _interior(square):
    r, c, s = square
    return {(i, j) for i in range(r + 1, r + s - 1) for j in range(c + 1, c + s - 1)}


def _scope(inp, square):
    inside = _interior(square)
    groups = {}
    for p in sorted(inside):
        if int(inp[p]) in INK:
            groups.setdefault(int(inp[p]), []).append(p)
    sources, others = [], []
    for color, points in groups.items():
        topology = _topology(points)
        if topology is None:
            others.append((color, points))
        else:
            topology["color"] = color
            sources.append(topology)
    source_pixels = set()
    for source in sources:
        if source_pixels & source["pixels"]:
            raise ValueError("different sources intersect")
        source_pixels |= source["pixels"]
    markers, inert, marked = [], [], set()
    for color, points in others:
        near = [(i, v, p) for p in points for i, source in enumerate(sources)
                for v in source["points"] if max(abs(v[0] - p[0]), abs(v[1] - p[1])) == 1]
        if not near:
            inert.extend(points)
            continue
        if len(points) != 1 or len(near) != 1:
            raise ValueError("ambiguous companion association")
        i, anchor, point = near[0]
        if i in marked or anchor not in sources[i]["long_pixels"]:
            raise ValueError("multiple companions or unit-only anchor")
        marked.add(i)
        dr, dc = point[0] - anchor[0], point[1] - anchor[1]
        full = {(r + dr, c + dc) for r, c in sources[i]["long_pixels"]}
        copy = full & inside
        unit_extra = {(r + dr, c + dc) for r, c in sources[i]["pixels"] - sources[i]["long_pixels"]}
        markers.append({"color": color, "point": point, "source": i,
                        "offset": (dr, dc), "copy": copy, "clipped": full - inside,
                        "unit_extra": (unit_extra & inside) - source_pixels - copy})
    return {"square": square, "inside": inside, "sources": sources,
            "source_pixels": source_pixels, "markers": markers, "inert": inert}


def _analyze(inp):
    squares = _squares(inp != 5)
    scopes = [_scope(inp, s) for s in squares]
    out = inp.copy()
    for info in scopes:
        claims = {}
        for marker in info["markers"]:
            for p in marker["copy"]:
                if inp[p] == 0 and p not in info["source_pixels"]:
                    if p in claims and claims[p] != marker["color"]:
                        raise ValueError("copies compete for a neutral cell")
                    claims[p] = marker["color"]
                    out[p] = marker["color"]
        for source in info["sources"]:
            for p in source["pixels"]:
                if inp[p] == 0:
                    out[p] = source["color"]
    return out, scopes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive scopes, topology, marker offsets and additions from this input."""
    inp = _grid(input_grid)
    return _analyze(inp)[0]


def _normalize(points):
    r0, c0 = min(r for r, c in points), min(c for r, c in points)
    return [(r - r0, c - c0) for r, c in points]


def _dihedral(points, code):
    result = []
    for r, c in points:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        result.append((r, c))
    return _normalize(result)


def _template(rng, size, kind):
    """Procedural vertices; ordered list is used only to make independent labels."""
    if kind == "segment":
        n = int(rng.integers(2, size - 3))
        points, closed = [(0, 0), (0, n)], False
        if rng.random() < .35:
            points[-1] = (n, n)
    elif kind in ("elbow", "unit"):
        w = int(rng.integers(4, size - 2))
        h = int(rng.choice([h for h in range(2, size - 2) if h != w]))
        points, closed = [(0, 0), (0, w), (h, w)], False
        if kind == "unit":
            points.insert(1, (0, 1))
    elif kind == "diamond":
        k = int(rng.integers(2, max(3, (size - 2) // 2)))
        points, closed = [(0, k), (k, 2 * k), (2 * k, k), (k, 0)], True
    elif kind == "walk":
        points, closed = [(0, 0)], False
        for _ in range(int(rng.integers(2, 7))):
            dr, dc = OFFSETS[int(rng.integers(8))]
            length = int(rng.integers(1, 5))
            points.append((points[-1][0] + dr * length, points[-1][1] + dc * length))
    else:
        h, w = (int(rng.integers(3, size - 2)) for _ in range(2))
        if kind == "square":
            w = h
        points, closed = [(0, 0), (0, w), (h, w), (h, 0)], True
        if kind == "bevel":
            points = [(0, 0), (0, w), (h - 1, w), (h, w - 1), (h, 0)]
        if kind == "notch" and h >= 4 and w >= 4:
            k, m = int(rng.integers(1, w)), int(rng.integers(1, h))
            points = [(0, 0), (0, w), (h, w), (h, k), (m, k), (m, 0)]
    return _dihedral(points, int(rng.integers(8))), closed


def _ordered_pixels(points, closed):
    links = list(zip(points, points[1:]))
    if closed:
        links.append((points[-1], points[0]))
    pixels, long_pixels = set(), set()
    for a, b in links:
        cells = _line(a, b)
        pixels |= cells
        if len(cells) > 2:
            long_pixels |= cells
    return pixels, long_pixels


def _drawing(rng, size, mode, offsets, marked):
    colors = [int(c) for c in rng.permutation(INK)]
    kinds = ("segment", "elbow", "square", "rectangle", "diamond", "bevel", "notch", "walk", "unit")
    for attempt in range(160):
        if mode == "witness":
            # A corner-near square creates rejected chords, source/copy overlap
            # and a clipped companion, all visibly distinguishable.
            k = int(rng.integers(3, size - 2))
            vertices = [(1, 1), (1, k + 1), (k + 1, k + 1), (k + 1, 1)]
            code = int(rng.integers(8))
            vertices = _dihedral(vertices, code)
            vertices = [(r + 1, c + 1) for r, c in vertices]
            closed = True
        else:
            kind = "unit" if mode == "unit" else kinds[int(rng.integers(len(kinds)))]
            vertices, closed = _template(rng, size, kind)
            if len(vertices) != len(set(vertices)):
                continue
            h, w = max(r for r, c in vertices) + 1, max(c for r, c in vertices) + 1
            if h > size - 2 or w > size - 2:
                continue
            top, left = int(rng.integers(1, size - h)), int(rng.integers(1, size - w))
            vertices = [(r + top, c + left) for r, c in vertices]
        try:
            pixels, long_pixels = _ordered_pixels(vertices, closed)
            inferred = _topology(sorted(vertices))
            if inferred is None or inferred["pixels"] != pixels or not long_pixels:
                continue
            options = []
            if marked:
                for dr, dc in offsets:
                    if mode == "witness" and not (dr and dc):
                        continue
                    full = {(r + dr, c + dc) for r, c in long_pixels}
                    inside = _interior((0, 0, size))
                    copy = full & inside
                    for anchor in vertices:
                        marker = (anchor[0] + dr, anchor[1] + dc)
                        if (anchor not in long_pixels or marker not in inside or marker in pixels
                                or sum(max(abs(marker[0] - r), abs(marker[1] - c)) == 1 for r, c in vertices) != 1):
                            continue
                        visible = copy - pixels - {marker}
                        overlap = (copy & pixels) - set(vertices)
                        extra = {(r + dr, c + dc) for r, c in pixels - long_pixels}
                        extra = (extra & inside) - pixels - copy - {marker}
                        if len(visible) < 2:
                            continue
                        if mode == "witness" and (not overlap or not full - inside or not inferred["chord_pixels"]):
                            continue
                        if mode == "unit" and not extra:
                            continue
                        options.append((marker, (dr, dc), copy))
                if not options:
                    continue
                marker, offset, copy = options[int(rng.integers(len(options)))]
            else:
                marker, offset, copy = None, None, set()
            inp = np.zeros((size, size), dtype=np.int8)
            for p in vertices:
                inp[p] = colors[0]
            if marker is not None:
                inp[marker] = colors[1]
            # Inert landmarks are unaligned with each other and not adjacent
            # to any vertex. They can never impersonate a companion marker.
            # Keep them off counterexample cells too: otherwise an inert dot
            # could accidentally hide the only unit-stub/chord distinction.
            evidence_cells = inferred["chord_pixels"].copy()
            if marker is not None:
                dr, dc = offset
                evidence_cells |= {(r + dr, c + dc) for r, c in pixels - long_pixels}
            available = [p for p in sorted(_interior((0, 0, size)))
                         if p not in pixels | copy | evidence_cells | ({marker} if marker else set())
                         and all(max(abs(p[0] - r), abs(p[1] - c)) > 1 for r, c in vertices)]
            rng.shuffle(available)
            inert = []
            desired = 2 if mode in ("witness", "unit") else int(rng.integers(4))
            for p in available:
                if len(inert) == desired:
                    break
                if all(not _aligned(p, q) for q in inert):
                    inert.append(p)
                    inp[p] = colors[2]
            out = inp.copy()
            for p in copy:
                if inp[p] == 0:
                    out[p] = colors[1]
            for p in pixels:
                if inp[p] == 0:
                    out[p] = colors[0]
            predicted, info = _analyze(inp)
            if not np.array_equal(predicted, out):
                continue
            if mode in ("witness", "unit") and len(inert) < 2:
                continue
            return inp, out, colors
        except ValueError:
            continue
    # A bounded deterministic fallback still uses the requested visible offset.
    # A perpendicular segment always has a visible translated companion.
    if mode in ("witness", "unit"):
        raise RuntimeError("teaching drawing construction exhausted")
    dr, dc = offsets[0]
    vertices = [(2, 2), (2, size - 3)] if dr else [(2, 2), (size - 3, 2)]
    inp = np.zeros((size, size), dtype=np.int8)
    for p in vertices:
        inp[p] = colors[0]
    if marked:
        inp[vertices[0][0] + dr, vertices[0][1] + dc] = colors[1]
    pixels = _line(*vertices)
    copy = ({(r + dr, c + dc) for r, c in pixels} & _interior((0, 0, size))) if marked else set()
    out = inp.copy()
    for p in copy:
        if inp[p] == 0:
            out[p] = colors[1]
    for p in pixels:
        if inp[p] == 0:
            out[p] = colors[0]
    return inp, out, colors


def _layout(rng, dimensions):
    """Fit randomly assigned rectangular regions in jittered shelves, <=30."""
    candidates = []
    for columns in range(1, min(3, len(dimensions)) + 1):
        for _ in range(12):
            order = list(map(int, rng.permutation(len(dimensions))))
            rows = (len(order) + columns - 1) // columns
            widths = [max(dimensions[order[k]][1] for k in range(j, len(order), columns)) for j in range(columns)]
            heights = [max(dimensions[order[k]][0] for k in range(i * columns, min((i + 1) * columns, len(order)))) for i in range(rows)]
            gap = int(rng.integers(1, 4))
            h, w = sum(heights) + gap * (rows - 1), sum(widths) + gap * (columns - 1)
            if h <= 30 and w <= 30:
                candidates.append((order, columns, widths, heights, gap, h, w))
    if not candidates:
        raise RuntimeError("region layout does not fit")
    order, columns, widths, heights, gap, h, w = candidates[int(rng.integers(len(candidates)))]
    eh, ew = int(rng.integers(min(3, 30 - h) + 1)), int(rng.integers(min(3, 30 - w) + 1))
    top, left = int(rng.integers(eh + 1)), int(rng.integers(ew + 1))
    placements = [None] * len(dimensions)
    for k, index in enumerate(order):
        i, j = divmod(k, columns)
        a, b = dimensions[index]
        r = top + sum(heights[:i]) + gap * i + int(rng.integers(heights[i] - a + 1))
        c = left + sum(widths[:j]) + gap * j + int(rng.integers(widths[j] - b + 1))
        placements[index] = (r, c, a, b)
    return (h + eh, w + ew), placements


def _substrate(rng, shape, regions, connect):
    base = np.zeros(shape, dtype=bool)
    for r, c, h, w in regions:
        base[r:r + h, c:c + w] = True
    intended = {(r, c, h) for r, c, h, w in regions if h == w}
    for attempt in range(25):
        mask = base.copy()
        # Partial attachments preserve local square eligibility, unlike a full
        # extra row. Corridors may unite target and decoy into one component.
        for r, c, h, w in regions:
            for _ in range(int(rng.integers(1, 4))):
                side = int(rng.integers(4))
                if side < 2:
                    start = (r if side == 0 else r + h - 1, c + int(rng.integers(w)))
                    dr, dc = (-1, 0) if side == 0 else (1, 0)
                else:
                    start = (r + int(rng.integers(h)), c if side == 2 else c + w - 1)
                    dr, dc = (0, -1) if side == 2 else (0, 1)
                for step in range(1, int(rng.integers(2, 6))):
                    p = start[0] + step * dr, start[1] + step * dc
                    if 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]:
                        mask[p] = True
        if connect:
            for first, second in zip(regions, regions[1:]):
                a = first[0] + first[2] // 2, first[1] + first[3] // 2
                b = second[0] + second[2] // 2, second[1] + second[3] // 2
                bend = (a[0], b[1]) if rng.random() < .5 else (b[0], a[1])
                for u, v in ((a, bend), (bend, b)):
                    for p in (_line(u, v) if u != v else {u}):
                        mask[p] = True
        try:
            selected = _squares(mask)
            # A corridor can accidentally add a complete side to the decoy
            # (e.g. turn a 9x10 rectangle into a 10x10 square). Keep this
            # counterexample genuinely outside every selected scope.
            r, c, h, w = regions[-1]
            decoy_selected = any(max(r, a) < min(r + h, a + s)
                                 and max(c, b) < min(c + w, b + s)
                                 for a, b, s in selected)
            if intended <= set(selected) and not decoy_selected:
                return mask
        except ValueError:
            pass
    # Disjoint rectangles always retain the same scoped transformation.
    return base


def _pair(rng, mode, offsets, active_count):
    sides = [int(rng.integers(8, 13 if active_count <= 2 else 12)) for _ in range(active_count)]
    if mode == "unit" and sides[0] == sides[1]:
        sides[1] = 8 + (sides[0] - 8 + 1) % 5
    drawings = []
    for i, side in enumerate(sides):
        local_mode = mode if i == 0 else "ordinary"
        marked = i == 0 or bool(rng.random() < .5)
        if mode == "unit" and i > 0:
            marked = False
        drawings.append(_drawing(rng, side, local_mode, offsets, marked))
    # A wide solid region contains a completeable pair and a companion too;
    # every square subwindow in it is extendable. It must stay unchanged.
    dh, dw = int(rng.integers(6, 10)), int(rng.integers(10, 15))
    dimensions = [(s, s) for s in sides] + [(dh, dw)]
    shape, regions = _layout(rng, dimensions)
    mask = _substrate(rng, shape, regions, connect=mode == "unit" or rng.random() < .6)
    inp = np.where(mask, 0, 5).astype(np.int8)
    out = inp.copy()
    for (r, c, h, w), (before, after, colors) in zip(regions, drawings):
        inp[r:r + h, c:c + w] = before
        out[r:r + h, c:c + w] = after
    r, c, h, w = regions[-1]
    source, companion = drawings[0][2][:2]
    for p in ((r + 1, c + 1), (r + 1, c + w - 2)):
        inp[p] = out[p] = source
    inp[r + 2, c + 1] = out[r + 2, c + 1] = companion
    # Offset holding-out is specified in final coordinates, so use no rotation
    # after drawing; source symmetries and shelves supply geometric variation.
    predicted, _ = _analyze(inp)
    if not np.array_equal(predicted, out):
        raise RuntimeError("constructed labels disagree with visible scopes")
    return {"input": inp, "output": out}


def _fingerprint(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _metrics(inp, out, scopes=None):
    if scopes is None:
        _, scopes = _analyze(inp)
    inside = set().union(*(s["inside"] for s in scopes)) if scopes else set()
    outside = {}
    for p in map(tuple, np.argwhere((inp != 0) & (inp != 5))):
        if p not in inside:
            outside.setdefault(int(inp[p]), []).append(p)
    outside_gaps = set()
    for points in outside.values():
        for i, j in _edges(points):
            cells = _line(points[i], points[j])
            if all(inp[p] != 5 for p in cells):
                outside_gaps.update(p for p in cells if inp[p] == 0 and p not in inside)
    offsets, kinds, geometries = set(), set(), set()
    priority = clipped = unit_omit = visible = chord_blank = inert = 0
    for info in scopes:
        inert += len(info["inert"])
        for source in info["sources"]:
            kinds.add(source["kind"])
            geometries.add(tuple(sorted(_normalize(source["points"]))))
            chord_blank += sum(out[p] == 0 for p in source["chord_pixels"])
        for marker in info["markers"]:
            offsets.add(marker["offset"])
            source = info["sources"][marker["source"]]
            priority += sum(inp[p] == 0 and out[p] == source["color"] for p in marker["copy"] & source["pixels"])
            clipped += len(marker["clipped"])
            unit_omit += sum(inp[p] == 0 and out[p] == 0 for p in marker["unit_extra"])
            visible += sum(inp[p] == 0 and out[p] == marker["color"] for p in marker["copy"])
    return {"squares": len(scopes), "active": sum(bool(s["sources"]) for s in scopes),
            "sides": {s["square"][2] for s in scopes}, "outside_gaps": len(outside_gaps),
            "offsets": offsets, "kinds": kinds, "geometry": geometries,
            "priority": priority, "clipped": clipped, "unit_omit": unit_omit,
            "visible": visible, "chord_blank": chord_blank, "inert": inert}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training, queries = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, scopes = _analyze(inp)
            assert scopes and np.array_equal(expected, out), "incorrect scoped completion"
            assert np.any(inp != out), "no completion evidence"
            assert np.array_equal(inp[inp != 0], out[inp != 0]), "original cells changed"
            allowed = np.zeros_like(inp, dtype=bool)
            for info in scopes:
                for p in info["inside"]:
                    allowed[p] = True
                r, c, s = info["square"]
                for p in [(r, j) for j in range(c, c + s)] + [(r + s - 1, j) for j in range(c, c + s)] + [(i, c) for i in range(r, r + s)] + [(i, c + s - 1) for i in range(r, r + s)]:
                    assert inp[p] == out[p], "scope border changed"
            assert not np.any((inp != out) & ~allowed), "drawing escaped square interior"
            metric = _metrics(inp, out, scopes)
            assert metric["outside_gaps"] > 0, "square-gating ablation is inert"
            assert metric["visible"] > 0, "companion translation is inert"
            (training if split == "train" else queries).append(metric)
    for feature in ("priority", "clipped", "unit_omit", "chord_blank", "inert"):
        assert sum(m[feature] for m in training) > 0, "missing demonstration: " + feature
    assert {"path", "cycle"} <= set().union(*(m["kinds"] for m in training))
    assert len({m["active"] for m in training}) >= 2
    shown = set().union(*(m["offsets"] for m in training))
    asked = set().union(*(m["offsets"] for m in queries))
    assert asked - shown, "expected visibly encoded offset generalization"


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(name + " has an illegal count")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    held = CARDINAL[int(rng.integers(4))]
    training_offsets = tuple(d for d in OFFSETS if d != held)
    modes = ["witness", "unit"] + ["ordinary"] * (nt - 2)
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            mode = modes[index] if split == "train" else "ordinary"
            active = 1 if mode == "witness" else 2 if mode == "unit" else int(rng.integers(1 if split == "train" else 2, 4))
            offsets = training_offsets if split == "train" else (held,)
            for retry in range(16):
                pair = _pair(rng, mode, offsets, active)
                key = _fingerprint(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


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
    """42 streamed seeds, semantic counterexamples and mutations; no files."""
    dimensions, geometries, signatures, offsets, sides, active = set(), set(), set(), set(), set(), set()
    totals = {k: 0 for k in ("priority", "clipped", "unit_omit", "chord_blank", "inert", "outside_gaps")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                m = _metrics(inp, out)
                dimensions.add(inp.shape)
                geometries |= m["geometry"]
                offsets |= m["offsets"]
                sides |= m["sides"]
                active.add(m["active"])
                for name in totals:
                    totals[name] += m[name]
                # Completion is equivariant under all eight scene symmetries.
                if seed < 2:
                    for k in range(4):
                        a, b = np.rot90(inp, k), np.rot90(out, k)
                        assert np.array_equal(solve(a), b)
                        assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
        if seed < 2:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
    assert len(signatures) == 42 and len(dimensions) >= 30 and len(geometries) >= 40
    assert offsets == set(OFFSETS) and {8, 9, 10, 11, 12} <= sides and {1, 2, 3} <= active
    assert all(totals.values())
    # Pure A counterexample: a 5x6 region contains many solid square windows,
    # but not a locally maximal square. Adding a full side changes eligibility.
    a = np.full((8, 10), 5, dtype=np.int8)
    a[1:6, 1:6] = 0
    assert _squares(a != 5) == [(1, 1, 5)]
    a[1:6, 6] = 0
    assert _squares(a != 5) == []
    # A one-cell partial side extension does not suppress the 5x5.
    a[2:, 6] = 5
    assert _squares(a != 5) == [(1, 1, 5)]
    # Pure B counterexample: square landmarks have two aligned diagonals,
    # yet the unique noncrossing full cycle leaves their central cell blank.
    b = np.zeros((9, 9), dtype=np.int8)
    for p in ((2, 2), (2, 6), (6, 6), (6, 2)):
        b[p] = 3
    result = solve(b)
    assert result[4, 4] == 0 and result[2, 4] == 3
    # Scope has exact causal force: extend one complete side and the identical
    # landmarks no longer grow, despite their unchanged topology.
    c = np.full((11, 12), 5, dtype=np.int8)
    c[1:10, 1:10] = b
    assert np.any(solve(c) != c)
    c[1:10, 10] = 0
    assert np.array_equal(solve(c), c)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(12345, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validator catches wrong completion, changed old ink, and border leakage.
    original = generate_puzzle(0)
    pair = original["test"][0]
    locations = [tuple(np.argwhere(pair["input"] != pair["output"])[0]),
                 tuple(np.argwhere(pair["input"] == 5)[0]),
                 _squares(pair["input"] != 5)[0][:2]]
    for p in locations:
        bad = {s: [{k: v.copy() for k, v in x.items()} for x in original[s]] for s in original}
        grid = bad["test"][0]["output"]
        grid[p] = (int(grid[p]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, scope/topology/translation checks")
