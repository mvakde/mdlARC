"""Echo chambers: translated sparse walls gate inward perpendicular ribs.

Only visible input determines color roles, contour order, offsets, and ray stops.
The public rule and generation envelope are documented in task_spec.md.
"""
from fractions import Fraction
import json
import numpy as np

OFFSETS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _sub(a, b):
    return a[0] - b[0], a[1] - b[1]


def _aligned(a, b):
    r, c = _sub(b, a)
    return (r != 0 or c != 0) and (r == 0 or c == 0 or abs(r) == abs(c))


def _segment(a, b):
    if not _aligned(a, b):
        raise ValueError("Walls must be horizontal, vertical or 45-degree")
    r, c = _sub(b, a)
    n = max(abs(r), abs(c))
    dr, dc = r // n, c // n
    return tuple((a[0] + k * dr, a[1] + k * dc) for k in range(n + 1))


def _on_edge(p, a, b):
    return (_cross(_sub(p, a), _sub(b, a)) == 0
            and min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
            and min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _intersect(a, b, c, d):
    if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
            or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
        return False
    return (_cross(_sub(b, a), _sub(c, a)) * _cross(_sub(b, a), _sub(d, a)) <= 0
            and _cross(_sub(d, c), _sub(a, c)) * _cross(_sub(d, c), _sub(b, c)) <= 0)


def _edges(points):
    occupied = set(points)
    return [(i, j) for i, a in enumerate(points) for j in range(i + 1, len(points))
            if _aligned(a, points[j]) and not occupied.intersection(_segment(a, points[j])[1:-1])]


def _cycle(points):
    """Find a unique simple cycle through every landmark; reject aligned chords.

    Reversal is the same cycle. Continuous crossing checks also catch half-cell
    crossings. Open paths and ambiguous cycles are outside this new family.
    """
    n = len(points)
    if not 4 <= n <= 12:
        return None
    edges = _edges(points)
    neighbors = [[] for _ in points]
    for e, (i, j) in enumerate(edges):
        neighbors[i].append((j, e))
        neighbors[j].append((i, e))
    if any(len(ns) < 2 for ns in neighbors):
        return None
    conflicts = [set() for _ in edges]
    for e, (a, b) in enumerate(edges):
        for f, (c, d) in enumerate(edges[:e]):
            if not {a, b}.intersection((c, d)) and _intersect(points[a], points[b], points[c], points[d]):
                conflicts[e].add(f)
                conflicts[f].add(e)
    solutions = {}
    visits = 0

    def visit(order, used, chosen):
        nonlocal visits
        visits += 1
        if visits > 60000:
            raise ValueError("Contour search bound exceeded")
        if len(solutions) > 1:
            return
        if len(order) == n:
            closing = [e for v, e in neighbors[order[-1]] if v == 0]
            if closing and not conflicts[closing[0]].intersection(chosen):
                signature = tuple(sorted(chosen + [closing[0]]))
                solutions[signature] = order
            return
        for v, e in neighbors[order[-1]]:
            if v not in used and not conflicts[e].intersection(chosen):
                visit(order + [v], used | {v}, chosen + [e])

    visit([0], {0}, [])
    if len(solutions) > 1:
        raise ValueError("Competing simple closed contours")
    if not solutions:
        return None
    signature, order = next(iter(solutions.items()))
    polygon = [points[i] for i in order]
    sides = list(zip(polygon, polygon[1:] + polygon[:1]))
    pixels = set().union(*(set(_segment(a, b)) for a, b in sides))
    all_pixels = set().union(*(set(_segment(points[i], points[j])) for i, j in edges))
    turns = {_cross(_sub(b, a), _sub(c, b)) for a, b, c in
             zip(polygon, polygon[1:] + polygon[:1], polygon[2:] + polygon[:2])}
    return {"vertices": points, "polygon": polygon, "sides": sides, "pixels": pixels,
            "rejected": all_pixels - pixels, "chords": len(edges) - n,
            "concave": any(t < 0 for t in turns) and any(t > 0 for t in turns)}


def _inside(p, polygon):
    """Strict continuous polygon interior, with boundary excluded."""
    inside = False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _on_edge(p, a, b):
            return False
        if (a[0] > p[0]) != (b[0] > p[0]):
            col = a[1] + (p[0] - a[0]) * (b[1] - a[1]) / (b[0] - a[0])
            if p[1] < col:
                inside = not inside
    return inside


def _ray(point, side, polygon):
    """Return inward integer cells BEFORE the first continuous wall crossing."""
    a, b = side
    dr, dc = _sub(b, a)
    if not dr or not dc or abs(dr) != abs(dc):
        raise ValueError("Only diagonal sides launch ribs")
    d = (1, -1 if dr * dc > 0 else 1)
    choices = [v for v in (d, (-d[0], -d[1]))
               if _inside((point[0] + v[0] / 4, point[1] + v[1] / 4), polygon)]
    if len(choices) != 1:
        raise ValueError("Probe must be in the open interior of one side")
    direction = choices[0]
    hits = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        e, aq = _sub(b, a), _sub(a, point)
        den = _cross(direction, e)
        if den:
            t, u = Fraction(_cross(aq, e), den), Fraction(_cross(aq, direction), den)
            if t > 0 and 0 <= u <= 1:
                hits.append(t)
        elif _cross(aq, direction) == 0:
            # A ray reaching a collinear wall stops at its first endpoint.
            for v in (a, b):
                t = Fraction(v[0] - point[0], direction[0])
                if t > 0:
                    hits.append(t)
    if not hits:
        raise ValueError("A closed chamber must stop every inward ray")
    stop = min(hits)
    cells = {(point[0] + k * direction[0], point[1] + k * direction[1])
             for k in range(1, 31) if k < stop}
    reentry = {(point[0] + k * direction[0], point[1] + k * direction[1])
               for k in range(1, 31) if k > stop
               and _inside((point[0] + k * direction[0], point[1] + k * direction[1]), polygon)}
    return {"point": point, "direction": direction, "stop": stop,
            "cells": cells, "reentry": reentry, "slope": 1 if dr * dc > 0 else -1}


def _translate(points, delta):
    return [(r + delta[0], c + delta[1]) for r, c in points]


def _near(p, vertices):
    return sum(max(abs(p[0] - v[0]), abs(p[1] - v[1])) == 1 for v in vertices)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer) or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _analyze(input_grid):
    g = _grid(input_grid)
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if int(max(counts)) <= g.size / 2:
        raise ValueError("Background must occupy more than half the input")
    groups = {int(v): [tuple(map(int, p)) for p in np.argwhere(g == v)] for v in values if v != bg}
    sources = []
    for color, points in groups.items():
        contour = _cycle(points)
        if contour is not None:
            contour["source_color"] = color
            sources.append(contour)
    if not 1 <= len(sources) <= 3:
        raise ValueError("Expected one to three recoverable source contours")
    source_colors = {s["source_color"] for s in sources}
    all_vertices = [p for s in sources for p in s["vertices"]]
    paired = {}
    for color, points in groups.items():
        if color in source_colors:
            continue
        anchors = [(p, i, v) for p in points for i, s in enumerate(sources) for v in s["vertices"]
                   if max(abs(p[0] - v[0]), abs(p[1] - v[1])) == 1]
        if len(anchors) != 1:
            raise ValueError("Each companion color needs exactly one adjacent anchor")
        anchor, i, vertex = anchors[0]
        if i in paired:
            raise ValueError("A source cannot have two companion anchors")
        paired[i] = (color, points, anchor, _sub(anchor, vertex))
    if len(paired) != len(sources):
        raise ValueError("Every chamber needs its visible translation anchor")
    domains = []
    for i, s in enumerate(sources):
        color, points, anchor, delta = paired[i]
        polygon = _translate(s["polygon"], delta)
        sides = list(zip(polygon, polygon[1:] + polygon[:1]))
        echo = set().union(*(set(_segment(a, b)) for a, b in sides))
        if any(not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]) for r, c in echo):
            raise ValueError("Translated walls must fit without clipping")
        if anchor in s["pixels"]:
            raise ValueError("Offset anchor cannot lie on the source wall")
        probes = set(points) - {anchor}
        if any(_near(p, all_vertices) for p in probes):
            raise ValueError("Probe cannot be confused with an offset anchor")
        rays = []
        for p in sorted(probes):
            eligible = [(a, b) for a, b in sides if a[0] != b[0] and a[1] != b[1]
                        and p not in (a, b) and _on_edge(p, a, b)]
            if eligible:
                if len(eligible) != 1:
                    raise ValueError("Ambiguous probe side")
                rays.append(_ray(p, eligible[0], polygon))
        s.update({"companion_color": color, "anchor": anchor, "offset": delta,
                  "echo_polygon": polygon, "echo_sides": sides, "echo": echo,
                  "probes": probes, "rays": rays})
        # Objects may be scattered; their full rectangular domains do not overlap.
        extent = s["pixels"] | echo
        box = (min(r for r, c in extent), max(r for r, c in extent),
               min(c for r, c in extent), max(c for r, c in extent))
        if any(not (box[1] + 1 < b[0] or b[1] + 1 < box[0]
                    or box[3] + 1 < b[2] or b[3] + 1 < box[2]) for b in domains):
            raise ValueError("Chamber domains must have a separating gap")
        domains.append(box)
        if not any(a[0] != b[0] and a[1] != b[1] for a, b in sides):
            raise ValueError("Chamber must have a diagonal wall")
    return g, bg, sources


def _paint(g, bg, sources, *, ribs=True, translated=True, all_links=False, reenter=False,
           source_gate=False, full_lines=False):
    """Render once; switches are explicit counterfactuals used for validation."""
    out = g.copy()
    source_union = set().union(*(s["pixels"] for s in sources))
    for s in sources:
        for p in s["pixels"]:
            if g[p] == bg:
                out[p] = s["source_color"]
    for s in sources:
        if translated:
            extra = set(s["echo"])
            if all_links:
                extra.update(_translate(s["rejected"], s["offset"]))
            if ribs:
                if source_gate:
                    for p in s["probes"]:
                        for a, b in s["sides"]:
                            if a[0] != b[0] and a[1] != b[1] and p not in (a, b) and _on_edge(p, a, b):
                                extra.update(_ray(p, (a, b), s["polygon"])["cells"])
                else:
                    extra.update(p for ray in s["rays"] for p in ray["cells"])
                    if full_lines:
                        for ray in s["rays"]:
                            q, d = ray["point"], ray["direction"]
                            extra.update((q[0] + k * d[0], q[1] + k * d[1])
                                         for k in range(-30, 31)
                                         if 0 <= q[0] + k * d[0] < g.shape[0]
                                         and 0 <= q[1] + k * d[1] < g.shape[1])
                if reenter:
                    extra.update(p for ray in s["rays"] for p in ray["reentry"])
        else:
            extra = set(s["pixels"])
            if ribs:
                for p in s["probes"]:
                    for a, b in s["sides"]:
                        if a[0] != b[0] and a[1] != b[1] and p not in (a, b) and _on_edge(p, a, b):
                            extra.update(_ray(p, (a, b), s["polygon"])["cells"])
        for p in extra:
            if 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1] and g[p] == bg and p not in source_union:
                out[p] = s["companion_color"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one grid without training examples, generation state, seed or labels."""
    g, bg, sources = _analyze(input_grid)
    return _paint(g, bg, sources)


def _symmetry(points, code):
    out = []
    for r, c in points:
        if code >= 4:
            c = -c
        for _ in range(code % 4):
            r, c = c, -r
        out.append((r, c))
    return out


def _polygon(rng, scale, concave=False):
    if not concave and rng.random() < .3:
        a, b = (int(rng.integers(3, scale + 1)) for _ in range(2))
        points = [(0, b), (a, a + b), (a + b, a), (b, 0)]
    else:
        h, w = (int(rng.integers(6, scale + 5)) for _ in range(2))
        cap = min(h, w) // 3
        cuts = [int(rng.integers(cap + 1)) for _ in range(4)]
        if not any(cuts):
            cuts[int(rng.integers(4))] = min(3, cap)
        a, b, c, d = cuts
        points = [(0, a), (0, w - b), (b, w), (h - c, w),
                  (h, w - c), (h, d), (h - d, 0), (a, 0)]
        points = [p for i, p in enumerate(points) if p != points[i - 1]]
        if concave:
            eligible = [(i, a, b) for i, (a, b) in enumerate(zip(points, points[1:] + points[:1]))
                        if (a[0] == b[0] or a[1] == b[1]) and len(_segment(a, b)) >= 7]
            if not eligible:
                return None
            i, a, b = eligible[int(rng.integers(len(eligible)))]
            length = len(_segment(a, b)) - 1
            sr, sc = (b[0] - a[0]) // length, (b[1] - a[1]) // length
            depth = int(rng.integers(2, min(h, w) - 2))
            notch_width = int(rng.integers(2, length - 2))
            start = int(rng.integers(1, length - notch_width))
            p = (a[0] + start * sr, a[1] + start * sc)
            q = (p[0] + notch_width * sr, p[1] + notch_width * sc)
            inner_p = (p[0] + depth * sc, p[1] - depth * sr)
            inner_q = (q[0] + depth * sc, q[1] - depth * sr)
            points[i + 1:i + 1] = [p, inner_p, inner_q, q]
    points = _symmetry(points, int(rng.integers(8)))
    r0, c0 = min(r for r, c in points), min(c for r, c in points)
    return [(r - r0, c - c0) for r, c in points]


def _local_chamber(rng, offsets, scale, mode):
    """Procedural geometry; no parent generator or bank of solved layouts."""
    for _ in range(240):
        points = _polygon(rng, scale, concave=mode == "concave" or (mode in ("random", "query") and rng.random() < .3))
        if points is None or len(points) > 12:
            continue
        try:
            s = _cycle(sorted(points))
        except ValueError:
            continue
        if s is None:
            continue
        delta = offsets[int(rng.integers(len(offsets)))]
        polygon = _translate(s["polygon"], delta)
        sides = list(zip(polygon, polygon[1:] + polygon[:1]))
        echo = set().union(*(set(_segment(a, b)) for a, b in sides))
        anchors = [p for p in _translate(points, delta) if p not in s["pixels"] and _near(p, points) == 1]
        if not anchors:
            continue
        anchor = anchors[int(rng.integers(len(anchors)))]
        candidates = []
        for a, b in sides:
            if a[0] == b[0] or a[1] == b[1]:
                continue
            for p in _segment(a, b)[1:-1]:
                if p not in points and p != anchor and not _near(p, points):
                    ray = _ray(p, (a, b), polygon)
                    if ray["cells"] - s["pixels"] - echo:
                        candidates.append(ray)
        if mode in ("teaching", "concave") and len(candidates) < 2:
            continue
        if mode == "concave" and (not s["concave"] or not any(x["reentry"] for x in candidates)):
            continue
        chosen = []
        if mode != "silent" and candidates:
            if mode == "concave":
                pool = [x for x in candidates if x["reentry"]]
                chosen.append(pool[int(rng.integers(len(pool)))])
            if mode in ("teaching", "concave"):
                for slope in (-1, 1):
                    pool = [x for x in candidates if x["slope"] == slope and x not in chosen]
                    if pool:
                        chosen.append(pool[int(rng.integers(len(pool)))])
            maximum = min(5, len(candidates))
            target = int(rng.integers(2 if mode in ("teaching", "concave") else 0, maximum + 1))
            for i in rng.permutation(len(candidates)):
                if len(chosen) >= target:
                    break
                x = candidates[int(i)]
                if x not in chosen:
                    chosen.append(x)
        probes = {x["point"] for x in chosen}
        if mode in ("teaching", "concave") and not any(p not in s["pixels"] for p in probes):
            continue
        # Distractors may occupy the source wall, a rejected chord, an orthogonal
        # echo side, the chamber interior, or its surrounding local canvas.
        echo_rejected = set(_translate(s["rejected"], delta)) - echo
        source_diagonal = set().union(*(set(_segment(a, b)[1:-1]) for a, b in s["sides"]
                                       if a[0] != b[0] and a[1] != b[1]))
        source_only = source_diagonal - echo - set(points)
        orthogonal = set().union(*(set(_segment(a, b)[1:-1]) for a, b in sides
                                  if a[0] == b[0] or a[1] == b[1]))
        occupied = s["pixels"] | echo
        r0, r1 = min(r for r, c in occupied) - 1, max(r for r, c in occupied) + 1
        c0, c1 = min(c for r, c in occupied) - 1, max(c for r, c in occupied) + 1
        free = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} - echo
        pools = [source_only, echo_rejected, orthogonal] if mode in ("teaching", "concave") else [
            pool for pool in (source_only, echo_rejected, orthogonal) if rng.random() < .4]
        pools += [free] * int(rng.integers(3))
        for pool in pools:
            available = sorted(p for p in pool if p not in points and p != anchor
                               and p not in probes and not _near(p, points))
            if available:
                probes.add(available[int(rng.integers(len(available)))])
        if len(probes) > 10:
            continue
        companion = sorted(probes | {anchor})
        try:
            # Geometric color roles must be recoverable from this input alone.
            if _cycle(companion) is not None:
                continue
        except ValueError:
            continue
        s.update({"offset": delta, "anchor": anchor, "probes": probes,
                  "echo_polygon": polygon, "echo_sides": sides, "echo": echo})
        s["rays"] = [_ray(p, (a, b), polygon) for p in sorted(probes) for a, b in sides
                     if a[0] != b[0] and a[1] != b[1] and p not in (a, b) and _on_edge(p, a, b)]
        if mode == "teaching" and not s["rejected"]:
            continue
        return s
    return None


def _move_chamber(s, shift):
    out = dict(s)
    for key in ("vertices", "polygon", "echo_polygon"):
        out[key] = _translate(s[key], shift)
    for key in ("pixels", "rejected", "echo", "probes"):
        out[key] = set(_translate(s[key], shift))
    out["anchor"] = _translate([s["anchor"]], shift)[0]
    out["sides"] = list(zip(out["polygon"], out["polygon"][1:] + out["polygon"][:1]))
    out["echo_sides"] = list(zip(out["echo_polygon"], out["echo_polygon"][1:] + out["echo_polygon"][:1]))
    out["rays"] = [dict(x, point=_translate([x["point"]], shift)[0],
                        cells=set(_translate(x["cells"], shift)), reentry=set(_translate(x["reentry"], shift)))
                   for x in s["rays"]]
    return out


def _pack(rng, chambers):
    boxes, placed = [], []
    for s in chambers:
        footprint = s["pixels"] | s["echo"] | s["probes"] | {s["anchor"]}
        r0, r1 = min(r for r, c in footprint), max(r for r, c in footprint)
        c0, c1 = min(c for r, c in footprint), max(c for r, c in footprint)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if max(h, w) > 30:
            return None
        for flat in rng.permutation((31 - h) * (31 - w)):
            r, c = int(flat) // (31 - w), int(flat) % (31 - w)
            box = (r, r + h - 1, c, c + w - 1)
            if all(box[1] + 2 < b[0] or b[1] + 2 < box[0] or box[3] + 2 < b[2] or b[3] + 2 < box[2] for b in boxes):
                boxes.append(box)
                placed.append(_move_chamber(s, (r - r0, c - c0)))
                break
        else:
            return None
    footprint = set().union(*(s["pixels"] | s["echo"] | s["probes"] | {s["anchor"]} for s in placed))
    r0, r1 = min(r for r, c in footprint), max(r for r, c in footprint)
    c0, c1 = min(c for r, c in footprint), max(c for r, c in footprint)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    er, ec = int(rng.integers(31 - h)), int(rng.integers(31 - w))
    before = int(rng.integers(er + 1)), int(rng.integers(ec + 1))
    placed = [_move_chamber(s, (before[0] - r0, before[1] - c0)) for s in placed]
    return (h + er, w + ec), placed


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(g, bg, sources):
    expected = _paint(g, bg, sources)
    rays = [x for s in sources for x in s["rays"]]
    source_union = set().union(*(s["pixels"] for s in sources))
    unpainted_interior = any(expected[r, c] == bg and _inside((r, c), s["echo_polygon"])
                             for s in sources
                             for r in range(min(p[0] for p in s["echo_polygon"]), max(p[0] for p in s["echo_polygon"]) + 1)
                             for c in range(min(p[1] for p in s["echo_polygon"]), max(p[1] for p in s["echo_polygon"]) + 1))
    return {"shape": g.shape, "background": bg, "chambers": len(sources),
            "vertices": [len(s["vertices"]) for s in sources],
            "active": [len(s["rays"]) for s in sources],
            "probes": [len(s["probes"]) for s in sources],
            "offsets": [s["offset"] for s in sources],
            "concave": sum(s["concave"] for s in sources),
            "slopes": sorted({x["slope"] for x in rays}),
            "ray_lengths": [len(x["cells"]) for x in rays],
            "half_stops": sum(x["stop"].denominator == 2 for x in rays),
            "inert": sum(len(s["probes"]) - len(s["rays"]) for s in sources),
            "overlap": sum(bool(g[p] == bg) for s in sources for p in s["echo"] & source_union),
            "omit_a_diff": int(np.count_nonzero(expected != _paint(g, bg, sources, translated=False))),
            "omit_b_diff": int(np.count_nonzero(expected != _paint(g, bg, sources, ribs=False))),
            "chord_diff": int(np.count_nonzero(expected != _paint(g, bg, sources, all_links=True))),
            "reentry_diff": int(np.count_nonzero(expected != _paint(g, bg, sources, reenter=True))),
            "source_gate_diff": int(np.count_nonzero(expected != _paint(g, bg, sources, source_gate=True))),
            "full_line_diff": int(np.count_nonzero(expected != _paint(g, bg, sources, full_lines=True))),
            "unpainted_interior": unpainted_interior,
            "echo_only_active": sum(x["point"] not in s["pixels"] for s in sources for x in s["rays"]),
            "source_only_inert": sum(p in s["pixels"] and p not in s["echo"] for s in sources for p in s["probes"]),
            "original_diagonal_inert": sum(p not in s["echo"] and any(
                a[0] != b[0] and a[1] != b[1] and p not in (a, b) and _on_edge(p, a, b)
                for a, b in s["sides"]) for s in sources for p in s["probes"]),
            "orthogonal_inert": sum(any((a[0] == b[0] or a[1] == b[1])
                and p not in (a, b) and _on_edge(p, a, b)
                for a, b in s["echo_sides"]) for s in sources for p in s["probes"]),
            "geometry": repr(sorted((tuple(sorted(_translate(s["vertices"],
                                           (-min(r for r, c in s["vertices"]), -min(c for r, c in s["vertices"]))))),
                                     s["offset"], len(s["rays"])) for s in sources))}


def _make_pair(rng, bg, offsets, count, mode, used):
    # Some queries exercise the concave stop already demonstrated in training;
    # others retain simpler geometry. This choice never changes the rule.
    query_focus = "concave" if mode == "query" and rng.random() < .35 else "query"
    for attempt in range(90):
        chambers = []
        for i in range(count):
            role = (query_focus if mode == "query" else mode) if i == 0 else "silent" if mode == "teaching" and i == 1 else "random"
            scale = int(rng.integers(5, 9 if count == 1 else 7))
            local_offsets = OFFSETS if mode == "query" and i > 0 else offsets
            s = _local_chamber(rng, local_offsets, scale, role)
            if s is None:
                break
            chambers.append(s)
        if len(chambers) != count:
            continue
        packing = _pack(rng, chambers)
        if packing is None:
            continue
        shape, chambers = packing
        g = np.full(shape, bg, dtype=np.int8)
        colors = [int(v) for v in rng.permutation(10) if v != bg]
        for i, s in enumerate(chambers):
            s["source_color"], s["companion_color"] = colors[2 * i:2 * i + 2]
            for p in s["vertices"]:
                g[p] = s["source_color"]
            for p in s["probes"] | {s["anchor"]}:
                g[p] = s["companion_color"]
        if _key(g) in used:
            continue
        try:
            visible, bg2, inferred = _analyze(g)
        except ValueError:
            continue
        expected = _paint(g, bg, chambers)
        if not np.array_equal(expected, _paint(visible, bg2, inferred)):
            raise AssertionError("Proposed and visible contour reconstructions differ: " + repr([
                (s["source_color"], s["offset"], sorted(s["probes"]), sorted(x["point"] for x in s["rays"]))
                for s in chambers + inferred]))
        f = _features(visible, bg2, inferred)
        if mode == "teaching" and not (f["chord_diff"] and f["original_diagonal_inert"]
                                        and f["echo_only_active"] and f["omit_b_diff"]
                                        and f["source_gate_diff"]
                                        and f["overlap"] and set(f["slopes"]) == {-1, 1}
                                        and len(set(f["offsets"])) >= 2):
            continue
        if mode == "concave" and not (f["reentry_diff"] and f["omit_b_diff"] and f["echo_only_active"]
                                      and f["orthogonal_inert"]):
            continue
        if mode == "query" and not f["omit_b_diff"]:
            continue
        if mode == "query" and query_focus == "concave" and not f["reentry_diff"]:
            continue
        return {"input": g, "output": expected}
    raise RuntimeError(f"Bounded procedural construction exhausted for {mode}/{count}")


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    held = OFFSETS[int(rng.integers(8))]
    train_offsets = tuple(d for d in OFFSETS if d != held)
    bg = int(rng.integers(10))
    background_policy = int(rng.integers(2))
    puzzle, used = {"train": [], "test": []}, set()
    # Random positions for two rich but complementary demonstrations.
    slots = [int(v) for v in rng.choice(nt, 2, replace=False)]
    for i in range(nt):
        mode = "teaching" if i == slots[0] else "concave" if i == slots[1] else "random"
        count = int(rng.integers(2, 4)) if mode == "teaching" else int(rng.integers(1, 3)) if mode == "concave" else int(rng.integers(1, 4))
        pair = _make_pair(rng, bg if background_policy == 0 else int(rng.integers(10)),
                          train_offsets, count, mode, used)
        puzzle["train"].append(pair)
        used.add(_key(pair["input"]))
    for _ in range(nq):
        count = int(rng.integers(1, 4))
        pair = _make_pair(rng, bg if background_policy == 0 else int(rng.integers(10)),
                          (held,), count, "query", used)
        puzzle["test"].append(pair)
        used.add(_key(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            g, bg, sources = _analyze(g)
            assert np.array_equal(_paint(g, bg, sources), out)
            assert np.array_equal(g[g != bg], out[g != bg]), "Original pixels changed"
            assert np.any(g != out)
            for s in sources:
                assert len(s["vertices"]) == len(s["sides"])
                assert s["offset"] in OFFSETS
                for x in s["rays"]:
                    assert all(_inside(p, s["echo_polygon"]) for p in x["cells"])
                    assert all(g[p] == s["companion_color"] for p in (x["point"],))
            records[split].append(_features(g, bg, sources))
    tr = records["train"]
    assert any(x["chord_diff"] for x in tr), "No demonstration rejects aligned chords"
    assert any(x["reentry_diff"] for x in tr), "First-wall stopping not demonstrated"
    assert any(x["original_diagonal_inert"] and x["echo_only_active"] for x in tr)
    assert any(x["source_gate_diff"] for x in tr)
    assert any(x["orthogonal_inert"] for x in tr)
    assert any(x["full_line_diff"] for x in tr)
    assert any(x["unpainted_interior"] for x in tr)
    assert any(x["overlap"] for x in tr)
    assert any(0 in x["active"] for x in tr), "No inactive chamber demonstrated"
    assert set(v for x in tr for v in x["slopes"]) == {-1, 1}
    assert any(max(x["active"]) >= 2 for x in tr)
    assert all(any(x[k] for x in tr) for k in ("omit_a_diff", "omit_b_diff"))
    learned = {d for x in tr for d in x["offsets"]}
    asked = {d for x in records["test"] for d in x["offsets"]}
    assert asked - learned, "The generated direction shift disappeared"
    assert all(set(x["offsets"]) - learned for x in records["test"])
    assert all(x["omit_a_diff"] and x["omit_b_diff"] for x in records["test"])
    return None


def self_test():
    """Bounded 42-seed feature checks, count coverage and semantic regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = []
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ("train", "test"):
            for pair in p[split]:
                g, bg, ss = _analyze(pair["input"])
                records.append(dict(_features(g, bg, ss), split=split))
    assert len({repr([_key(p["input"]) for split in ("train", "test") for p in task[split]]) for task in puzzles}) == 42
    assert len({x["shape"] for x in records}) >= 50
    assert len({x["geometry"] for x in records}) >= 100
    assert {x["background"] for x in records} == set(range(10))
    assert {d for x in records for d in x["offsets"]} == set(OFFSETS)
    assert {x["chambers"] for x in records} == {1, 2, 3}
    assert len({v for x in records for v in x["vertices"]}) >= 5
    assert len({v for x in records for v in x["ray_lengths"]}) >= 5
    assert any(x["half_stops"] for x in records)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(2026, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    a, b = generate_puzzle(314159), generate_puzzle(314159)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # A known between-cell crossing: a digital-boundary-only walk would leak.
    polygon = [(0, 0), (0, 4), (3, 7), (7, 7), (7, 2), (5, 0)]
    ray = _ray((1, 5), ((0, 4), (3, 7)), polygon)
    assert ray["direction"] == (1, -1) and ray["stop"] == Fraction(9, 2)
    assert ray["cells"] == {(2, 4), (3, 3), (4, 2), (5, 1)}
    # All eight symmetries preserve perpendicularity, inferred offsets and walls.
    for split in ("train", "test"):
        for pair in puzzles[0][split]:
            for rotation in range(4):
                for reflected in (False, True):
                    g, out = np.rot90(pair["input"], rotation), np.rot90(pair["output"], rotation)
                    if reflected:
                        g, out = np.fliplr(g), np.fliplr(out)
                    assert np.array_equal(solve(g), out)
    bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in a[s]] for s in a}
    p = bad["test"][0]
    point = tuple(np.argwhere(p["input"] != p["output"])[0])
    p["output"][point] = p["input"][point]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("A missing completion cell was accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({x["shape"] for x in records}),
            "geometries": len({x["geometry"] for x in records})}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
