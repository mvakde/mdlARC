"""Wireframe docking: complete a contour, match its nose, and sweep its outline.

Every decision in solve is reconstructed from the input. Generation paints
labels from its ordered construction, then checks them with the input parser.
Only Python's standard library and NumPy are used.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "bad grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid color")
    return a.astype(np.int8, copy=True)


def _line(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    n = max(abs(dr), abs(dc))
    _require(n > 0 and (dr == 0 or dc == 0 or abs(dr) == abs(dc)), "unaligned edge")
    sr, sc = int(np.sign(dr)), int(np.sign(dc))
    return {(a[0] + i * sr, a[1] + i * sc) for i in range(n + 1)}


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _intersection(a, b, c, d):
    # Continuous intersection also catches crossings between two pixel centers.
    if any(max(a[z], b[z]) < min(c[z], d[z]) or
           max(c[z], d[z]) < min(a[z], b[z]) for z in (0, 1)):
        return False
    return (_cross(a, b, c) * _cross(a, b, d) <= 0 and
            _cross(c, d, a) * _cross(c, d, b) <= 0)


def _outline(points):
    """Find the unique simple full cycle; neither order nor edges are supplied."""
    points = tuple(sorted(points))
    n = len(points)
    _require(4 <= n <= 12, "contours need 4..12 landmarks")
    landmarks = set(points)
    edges, rasters = [], []
    for i, j in combinations(range(n), 2):
        a, b = points[i], points[j]
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        if dr and dc and dr != dc:
            continue
        pixels = _line(a, b)
        if (pixels - {a, b}) & landmarks:
            continue
        edges.append((i, j))
        rasters.append(pixels)
    neighbors = [[] for _ in points]
    for e, (i, j) in enumerate(edges):
        neighbors[i].append((j, e))
        neighbors[j].append((i, e))
    _require(all(len(ns) >= 2 for ns in neighbors), "no closed contour")
    conflicts = [set() for _ in edges]
    for e, f in combinations(range(len(edges)), 2):
        i, j = edges[e]
        k, l = edges[f]
        if not {i, j} & {k, l} and _intersection(points[i], points[j], points[k], points[l]):
            conflicts[e].add(f)
            conflicts[f].add(e)
    solutions = {}
    def walk(order, chosen):
        if len(solutions) > 1:
            return
        if len(order) == n:
            for v, edge in neighbors[order[-1]]:
                if v == 0 and not conflicts[edge].intersection(chosen):
                    signature = tuple(sorted(chosen + [edge]))
                    solutions[signature] = tuple(order)
            return
        for v, edge in neighbors[order[-1]]:
            if v not in order and not conflicts[edge].intersection(chosen):
                walk(order + [v], chosen + [edge])

    walk([0], [])
    _require(len(solutions) == 1, "ambiguous or absent closed contour")
    selected, order = next(iter(solutions.items()))
    pixels = set().union(*(rasters[e] for e in selected))
    _require(bool(pixels - landmarks), "contour has no missing cells")
    turns = [_cross(points[order[i - 1]], points[order[i]],
                    points[order[(i + 1) % n]]) for i in range(n)]
    return {"vertices": landmarks, "pixels": pixels,
            "all_pixels": set().union(*rasters),
            "chords": len(edges) - n,
            "concave": any(t < 0 for t in turns) and any(t > 0 for t in turns),
            "unit_links": sum(len(rasters[e]) == 2 for e in selected),
            "edge_signature": {frozenset((points[edges[e][0]], points[edges[e][1]]))
                               for e in selected}}


def _fork(cells):
    """Recognize a solid base and three separated rectangular positive spokes."""
    parses = []
    for dr, dc in DIRECTIONS:
        out, across = (-dr, -dc), (-dc, dr)
        kbase = min(r * out[0] + c * out[1] for r, c in cells)
        qbase = min(r * across[0] + c * across[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - kbase,
                  r * across[0] + c * across[1] - qbase) for r, c in cells}
        width = max(q for k, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        depth = max(k for k, q in local)
        lens = [sum((k, q) in local for k in range(1, depth + 1)) for q in range(width)]
        if local != ({(0, q) for q in range(width)} |
                     {(k, q) for q, length in enumerate(lens) for k in range(1, length + 1)}):
            continue
        runs, q = [], 0
        while q < width:
            if lens[q] == 0:
                q += 1
                continue
            qa, length = q, lens[q]
            while q < width and lens[q] == length:
                q += 1
            runs.append((qa, q, length))
        if (len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1))
                or runs[1][0] + runs[1][1] != width):
            continue
        parses.append({"direction": (dr, dc), "out": out, "across": across,
                       "kbase": kbase, "qbase": qbase, "runs": runs,
                       "width": width, "lengths": tuple(run[2] for run in runs),
                       "face": set(range(runs[1][0] + qbase, runs[1][1] + qbase)),
                       "tip": kbase + runs[1][2], "cells": cells})
    return parses[0] if len(parses) == 1 else None


def _shift(cells, d, steps=1):
    return {(r + steps * d[0], c + steps * d[1]) for r, c in cells}


def _in_bounds(cells, shape):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells)


def _sweep(cells, direction, distance):
    return set().union(*(_shift(cells, direction, i) for i in range(distance + 1)))


def _parse(a):
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) * 2 > a.size, "background must occupy over half the input")
    groups = {int(c): {tuple(map(int, p)) for p in np.argwhere(a == c)}
              for c in values if int(c) != bg}
    forks, cargo, singletons = [], [], []
    for color, cells in groups.items():
        if len(cells) == 1:
            singletons.append((color, next(iter(cells))))
            continue
        f = _fork(cells)
        if f is not None:
            forks.append(dict(f, color=color))
        else:
            cargo.append(dict(_outline(cells), color=color))
    _require(bool(forks) and bool(cargo), "need docks and cargo")
    for color, point in singletons:
        near = [(i, v) for i, x in enumerate(cargo) for v in x["vertices"]
                if max(abs(point[0] - v[0]), abs(point[1] - v[1])) == 1]
        _require(len(near) == 1, "marker must touch exactly one cargo vertex")
        i, vertex = near[0]
        d = point[0] - vertex[0], point[1] - vertex[1]
        _require(d in DIRECTIONS, "marker must give an orthogonal displacement")
        _require("marker" not in cargo[i], "two markers on one cargo")
        _require(point not in cargo[i]["pixels"], "marker lies on source contour")
        cargo[i].update(marker=point, trail=color, direction=d)
    _require(all("marker" in x for x in cargo), "every cargo needs a marker")
    occupied = set()
    for obj in forks + cargo:
        cells = obj["cells"] if "cells" in obj else obj["pixels"] | {obj["marker"]}
        _require(not occupied.intersection(cells), "input constructions intersect")
        occupied.update(cells)
    return {"background": bg, "forks": forks, "cargo": cargo}


def _match(x, f, shape, obstacles, *, ignore_direction=False, center_only=False, width_only=False):
    if not ignore_direction and x["direction"] != f["direction"]:
        return None
    out, across = f["out"], f["across"]
    k0 = min(r * out[0] + c * out[1] for r, c in x["pixels"])
    face = {r * across[0] + c * across[1] for r, c in x["pixels"]
            if r * out[0] + c * out[1] == k0}
    if len(face) < 2 or face != set(range(min(face), max(face) + 1)):
        return None
    if center_only:
        if min(face) + max(face) != min(f["face"]) + max(f["face"]):
            return None
    elif width_only:
        if len(face) != len(f["face"]):
            return None
    elif face != f["face"]:
        return None
    # The whole cargo envelope must fit between the inner sides of the guards.
    qs = [r * across[0] + c * across[1] for r, c in x["pixels"]]
    left = f["qbase"] + f["runs"][0][1]
    right = f["qbase"] + f["runs"][2][0]
    if min(qs) < left or max(qs) >= right:
        return None
    distance = k0 - f["tip"] - 1
    if distance <= 0:
        return None
    swept = _sweep(x["pixels"], f["direction"], distance)
    if not _in_bounds(swept, shape) or swept.intersection(obstacles):
        return None
    final = _shift(x["pixels"], f["direction"], distance)
    return {"distance": distance, "swept": swept, "final": final,
            "direction": f["direction"], "face_span": len(face)}


def _analyze(a, *, ignore_direction=False, center_only=False, width_only=False):
    info = _parse(a)
    forks, cargo = info["forks"], info["cargo"]
    fixed = set().union(*(f["cells"] for f in forks))
    matches, used_forks, used_routes = [], set(), set()
    for ci, x in enumerate(cargo):
        obstacles = fixed.copy()
        for oi, other in enumerate(cargo):
            if oi != ci:
                obstacles.update(other["pixels"] | {other["marker"]})
        choices = []
        for fi, f in enumerate(forks):
            m = _match(x, f, a.shape, obstacles,
                       ignore_direction=ignore_direction, center_only=center_only, width_only=width_only)
            if m is not None:
                choices.append(dict(m, ci=ci, fi=fi))
        _require(len(choices) <= 1, "cargo matches multiple docks")
        if choices:
            m = choices[0]
            _require(m["fi"] not in used_forks, "two cargo match one dock")
            _require(not used_routes.intersection(m["swept"]), "swept contours interact")
            used_forks.add(m["fi"])
            used_routes.update(m["swept"])
            matches.append(m)
    info["matches"] = matches
    info["unmatched"] = len(cargo) - len(matches)
    return info


def _paint(a, cells, color):
    for p in cells:
        a[p] = color


def _render(a, info, *, vertices_only=False, one_step=False, filled_wake=False, chords=False):
    out = a.copy()
    for x in info["cargo"]:
        _paint(out, x["vertices"] | {x["marker"]}, info["background"])
    for m in info["matches"]:
        x = info["cargo"][m["ci"]]
        pixels = x["vertices"] if vertices_only else x["all_pixels"] if chords else x["pixels"]
        distance = 1 if one_step else m["distance"]
        final = _shift(pixels, m["direction"], distance)
        if filled_wake:
            r0, r1 = min(r for r, c in pixels), max(r for r, c in pixels)
            c0, c1 = min(c for r, c in pixels), max(c for r, c in pixels)
            pixels = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        sweep = _sweep(pixels, m["direction"], distance)
        _paint(out, sweep - final, x["trail"])
        _paint(out, final, x["color"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solver; all palettes, arrows, contours and stops are visible."""
    a = _grid(input_grid)
    return _render(a, _analyze(a))


def _ordered_pixels(points):
    return set().union(*(_line(a, b) for a, b in zip(points, points[1:] + points[:1])))


def _make_contour(rng, kind, compact):
    for attempt in range(160):
        s = int(rng.integers(3, 5 if compact else 8))
        flare = 0 if kind in ("square", "rectangle") else int(rng.integers(1, 2 if compact else 3))
        w = s - 1 + 2 * flare
        if kind == "square":
            s = int(rng.integers(5, 8))
            w = h = s - 1
            points = [(0, 0), (0, w), (h, w), (h, 0)]
        elif kind == "rectangle":
            h = int(rng.integers(3, 5 if compact else 9))
            points = [(0, 0), (0, w), (h, w), (h, 0)]
        elif kind == "trapezoid":
            h = int(rng.integers(flare + 2, flare + (5 if compact else 8)))
            points = [(0, flare), (0, w - flare), (flare, w),
                      (h, w), (h, 0), (flare, 0)]
        elif kind == "beveled":
            h = int(rng.integers(flare + 3, flare + (5 if compact else 8)))
            back = int(rng.integers(1, min(h - flare, w // 2) + 1))
            points = [(0, flare), (0, w - flare), (flare, w),
                      (h - back, w), (h, w - back), (h, back),
                      (h - back, 0), (flare, 0)]
            points = list(dict.fromkeys(points))
        else:
            h = int(rng.integers(flare + 4, flare + (6 if compact else 9)))
            lo = int(rng.integers(flare + 1, h - 2))
            hi = int(rng.integers(lo + 1, h))
            depth = int(rng.integers(1, min(3, w - 1)))
            points = [(0, flare), (0, w - flare), (flare, w), (h, w),
                      (h, 0), (hi, 0), (hi, depth), (lo, depth),
                      (lo, 0), (flare, 0)]
        # Center the nose at q=0..s-1. Flip the side containing any notch.
        points = [(k, q - flare) for k, q in points]
        if rng.integers(2):
            points = [(k, s - 1 - q) for k, q in points]
        try:
            inferred = _outline(points)
            intended = {frozenset((a, b)) for a, b in zip(points, points[1:] + points[:1])}
            if inferred["edge_signature"] != intended:
                continue
            if kind == "concave" and not inferred["concave"]:
                continue
            return points, _ordered_pixels(points), s, flare, h
        except ValueError:
            continue
    # A procedural concave fallback has a uniquely recoverable notched cycle.
    if kind == "concave":
        points = [(0, 0), (0, 2), (1, 3), (5, 3), (5, -1),
                  (3, -1), (3, 1), (2, 1), (2, -1), (1, -1)]
        _outline(points)
        return points, _ordered_pixels(points), 3, 1, 5
    points = [(0, 0), (0, 3), (4, 3), (4, 0)]
    return points, _ordered_pixels(points), 4, 0, 4


def _turn(point, direction):
    # Canonical inward direction is up; this rotates the complete construction.
    r, c = point
    turns = DIRECTIONS.index(direction)
    for _ in range(turns):
        r, c = c, -r
    return r, c


def _make_module(rng, direction, kind, status="active", compact=False, witness=False):
    points, outline, s, flare, height = _make_contour(rng, kind, compact)
    side = 1 if compact else int(rng.integers(1, 3))
    gap = flare + (1 if compact else int(rng.integers(1, 4)))
    overhang = 0 if compact else int(rng.integers(2))
    center = 3 if witness else int(rng.integers(1, 3 if compact else 6))
    if witness:
        left, right = 1, 2
    else:
        left, right = map(int, rng.integers(1, 4 if compact else 7, size=2))
    if kind == "square":
        distance = int(rng.integers(2, height - 1))
    else:
        distance = int(rng.integers(1, 3 if compact else 7))
    start = center + 1 + distance
    points = [(k + start, q) for k, q in points]
    outline = {(k + start, q) for k, q in outline}
    if status == "off_axis":
        points = [(k, q + 1) for k, q in points]
        outline = {(k, q + 1) for k, q in outline}
    # A near miss has the same axis but a two-cell wider central stop.
    q0, dock_span = (-1, s + 2) if status == "wrong_span" else (0, s)
    qa = q0 - gap - side
    qb = q0 + dock_span + gap + side
    structure = {(0, q) for q in range(qa - overhang, qb + overhang)}
    for a, b, length in ((qa, qa + side, left), (q0, q0 + dock_span, center),
                          (qb - side, qb, right)):
        structure.update((k, q) for k in range(1, length + 1) for q in range(a, b))
    arrow = (1, 0) if status == "wrong_direction" else (-1, 0)
    anchors = [p for p in points if p[0] == (start + height if arrow[0] == 1 else start)]
    anchors = [p for p in anchors if (p[0] + arrow[0], p[1]) not in outline
               and sum(max(abs(p[0] + arrow[0] - v[0]), abs(p[1] - v[1])) == 1
                       for v in points) == 1]
    _require(bool(anchors), "no unique marker anchor")
    anchor = anchors[int(rng.integers(len(anchors)))]
    marker = (anchor[0] + arrow[0], anchor[1])
    final = _shift(outline, (-1, 0), distance)
    swept = _sweep(outline, (-1, 0), distance)
    # Reserve even the hypothetical route of the two near-miss cargo objects.
    reserve = structure | swept | {marker}
    module = {"fork": structure, "points": set(points), "outline": outline,
              "marker": {marker}, "final": final, "swept": swept,
              "reserve": reserve, "active": status == "active"}
    for key in ("fork", "points", "outline", "marker", "final", "swept", "reserve"):
        module[key] = {_turn(p, direction) for p in module[key]}
    r0, c0 = min(r for r, c in module["reserve"]), min(c for r, c in module["reserve"])
    for key in ("fork", "points", "outline", "marker", "final", "swept", "reserve"):
        module[key] = _shift(module[key], (-r0, -c0))
    module["size"] = (max(r for r, c in module["reserve"]) + 1,
                       max(c for r, c in module["reserve"]) + 1)
    return module


def _pack(rng, modules):
    order = sorted(range(len(modules)), key=lambda i: -np.prod(modules[i]["size"]))
    for _ in range(45):
        used = np.zeros((30, 30), dtype=bool)
        placements = {}
        for i in order:
            h, w = modules[i]["size"]
            positions = [(r, c) for r in range(31 - h) for c in range(31 - w)
                         if not used[r:r + h, c:c + w].any()]
            if not positions:
                break
            r, c = positions[int(rng.integers(len(positions)))]
            placements[i] = (r, c)
            used[max(0, r - 1):min(30, r + h + 1),
                 max(0, c - 1):min(30, c + w + 1)] = True
        if len(placements) == len(modules):
            result = []
            for i, x in enumerate(modules):
                y = dict(x)
                for key in ("fork", "points", "outline", "marker", "final", "swept", "reserve"):
                    y[key] = _shift(x[key], placements[i])
                result.append(y)
            return result
    # Compact multi-cargo scenes always fit in separate corners. This fallback
    # changes placement only; it introduces no new drawing or solution rule.
    if len(modules) <= 3 and all(max(x["size"]) <= 14 for x in modules):
        corners = [(0, 0), (1, 1), (0, 1), (1, 0)]
        rng.shuffle(corners)
        result = []
        for x, (bottom, right) in zip(modules, corners):
            h, w = x["size"]
            offset = (30 - h if bottom else 0, 30 - w if right else 0)
            y = dict(x)
            for key in ("fork", "points", "outline", "marker", "final", "swept", "reserve"):
                y[key] = _shift(x[key], offset)
            result.append(y)
        return result
    return None


def _draw_pair(rng, modules, background, avoid_trail=None):
    reserve = set().union(*(x["reserve"] for x in modules))
    lo = min(r for r, c in reserve), min(c for r, c in reserve)
    height = max(r for r, c in reserve) - lo[0] + 1
    width = max(c for r, c in reserve) - lo[1] + 1
    eh, ew = int(rng.integers(31 - height)), int(rng.integers(31 - width))
    pad_r, pad_c = int(rng.integers(eh + 1)), int(rng.integers(ew + 1))
    a = np.full((height + eh, width + ew), background, dtype=np.int8)
    b = a.copy()
    colors = list(map(int, rng.permutation([c for c in range(10) if c != background])))
    for x in modules:
        shifted = {key: _shift(x[key], (pad_r - lo[0], pad_c - lo[1]))
                   for key in ("fork", "points", "outline", "marker", "final", "swept")}
        fc, cc, tc = colors.pop(), colors.pop(), colors.pop()
        if x["active"] and tc == avoid_trail:
            # Palette presentation must demonstrate that wake color is local.
            cc, tc = tc, cc
        _paint(a, shifted["fork"], fc)
        _paint(a, shifted["points"], cc)
        _paint(a, shifted["marker"], tc)
        _paint(b, shifted["fork"], fc)
        if x["active"]:
            _paint(b, shifted["swept"] - shifted["final"], tc)
            _paint(b, shifted["final"], cc)
    # The majority convention is enforced visibly, independent of literal color.
    while np.count_nonzero(a == background) * 2 <= a.size:
        axis = 0 if a.shape[0] <= a.shape[1] and a.shape[0] < 30 else 1
        _require(a.shape[axis] < 30, "cannot provide majority background")
        pads = [(0, 0), (0, 0)]
        pads[axis] = (0, 1)
        a, b = np.pad(a, pads, constant_values=background), np.pad(b, pads, constant_values=background)
    return {"input": a, "output": b}


def _pair(rng, background, specs, avoid_trail=None):
    for attempt in range(36):
        modules = [_make_module(rng, **spec) for spec in specs]
        placed = _pack(rng, modules)
        if placed is None:
            continue
        pair = _draw_pair(rng, placed, background, avoid_trail)
        try:
            info = _analyze(pair["input"])
            if (len(info["cargo"]) == len(modules) and len(info["forks"]) == len(modules)
                    and np.array_equal(_render(pair["input"], info), pair["output"])):
                return pair
        except ValueError:
            pass
    raise RuntimeError("bounded unambiguous packing exhausted")


def _valid_int(value, low, high):
    return (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            and low <= int(value) <= high)


def _fingerprint(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _valid_int(num_train, 2, 10) or not _valid_int(num_test, 1, 4) or num_train <= num_test:
        raise ValueError("counts must be 2..10 train, 1..4 test, with train > test")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    held = DIRECTIONS[int(rng.integers(4))]
    familiar = [d for d in DIRECTIONS if d != held]
    rng.shuffle(familiar)
    if familiar[1] == tuple(-v for v in familiar[0]):
        familiar[1], familiar[2] = familiar[2], familiar[1]
    backgrounds = list(map(int, rng.permutation(10)))
    puzzle, seen, observed_trails = {"train": [], "test": []}, set(), set()
    # Random positions for the two complementary demonstrations.
    demo = list(map(int, rng.choice(num_train, size=2, replace=False)))
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            ordinal = i if split == "train" else num_train + i
            bg = backgrounds[ordinal % 10]
            if split == "train" and i == demo[0]:
                specs = [dict(direction=familiar[0], kind="square"),
                         dict(direction=familiar[0], kind="rectangle", status="off_axis", compact=True)]
            elif split == "train" and i == demo[1]:
                specs = [dict(direction=familiar[1], kind="concave", compact=True, witness=True),
                         dict(direction=familiar[0], kind="rectangle", status="wrong_span", compact=True),
                         # Its *marker* still belongs to familiar directions.
                         dict(direction=tuple(-v for v in familiar[2]), kind="rectangle",
                              status="wrong_direction", compact=True)]
            else:
                n = int(rng.integers(1, 4))
                specs = []
                for j in range(n):
                    d = held if split == "test" and j == 0 else familiar[int(rng.integers(3))]
                    kind = str(rng.choice(["rectangle", "trapezoid", "beveled", "concave"]))
                    status = "active" if j == 0 else str(rng.choice(
                        ["active", "wrong_span", "off_axis"], p=[.7, .15, .15]))
                    specs.append(dict(direction=d, kind=kind, status=status, compact=n >= 2))
            for _ in range(12):
                avoid = next(iter(observed_trails)) if split == "train" and i == num_train - 1 and len(observed_trails) == 1 else None
                pair = _pair(rng, bg, specs, avoid)
                fp = _fingerprint(pair["input"])
                if fp not in seen:
                    seen.add(fp)
                    puzzle[split].append(pair)
                    if split == "train":
                        parsed = _analyze(pair["input"])
                        observed_trails.update(parsed["cargo"][m["ci"]]["trail"] for m in parsed["matches"])
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(_valid_int(nt, 2, 10) and _valid_int(nq, 1, 4) and nt > nq, "bad pair counts")
    seen, train_infos = set(), []
    evidence = {name: False for name in ("vertices_only", "one_step", "filled_wake", "chords",
                                        "direction", "size", "axis", "nose", "concavity", "stop", "erasure")}
    train_directions, test_directions, train_trails = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "canvas must be preserved")
            fp = _fingerprint(a)
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            info = _analyze(a)
            _require(bool(info["matches"]), "every example must show docking")
            _require(np.array_equal(_render(a, info), b), "wrong docking or outline wake")
            _require(not np.array_equal(a, b), "identity example")
            for f in info["forks"]:
                _require(min(f["lengths"]) > 0, "empty spoke")
                _require(all(int(b[p]) == f["color"] for p in f["cells"]), "dock changed")
            for m in info["matches"]:
                x, f = info["cargo"][m["ci"]], info["forks"][m["fi"]]
                _require(all(int(b[p]) == x["color"] for p in m["final"]), "final outline broken")
                _require(all(int(b[p]) == x["trail"] for p in m["swept"] - m["final"]), "wake broken")
                closest = min(r * f["out"][0] + c * f["out"][1] for r, c in m["final"])
                _require(closest == f["tip"] + 1, "wrong stop")
                (train_directions if split == "train" else test_directions).add(x["direction"])
            matched = {m["ci"] for m in info["matches"]}
            for ci, x in enumerate(info["cargo"]):
                if ci not in matched:
                    _require(all(int(b[p]) == info["background"] for p in x["vertices"] | {x["marker"]}),
                             "unmatched cargo not erased")
            if split != "train":
                continue
            train_infos.append(info)
            train_trails.update(info["cargo"][m["ci"]]["trail"] for m in info["matches"])
            for name in ("vertices_only", "one_step", "filled_wake", "chords"):
                evidence[name] |= not np.array_equal(_render(a, info, **{name: True}), b)
            for name, flag in (("direction", "ignore_direction"), ("size", "center_only"), ("axis", "width_only")):
                relaxed = _analyze(a, **{flag: True})
                evidence[name] |= not np.array_equal(_render(a, relaxed), b)
            evidence["erasure"] |= info["unmatched"] > 0
            for m in info["matches"]:
                x, f = info["cargo"][m["ci"]], info["forks"][m["fi"]]
                across = f["across"]
                qs = [r * across[0] + c * across[1] for r, c in x["pixels"]]
                evidence["nose"] |= max(qs) - min(qs) + 1 > m["face_span"]
                evidence["concavity"] |= x["concave"]
                evidence["stop"] |= f["lengths"][1] > max(f["lengths"][0], f["lengths"][2])
    _require(all(evidence.values()), "missing training evidence: " + str([k for k, v in evidence.items() if not v]))
    _require(len(train_directions) >= 2, "need two demonstrated travel orientations")
    _require(len({bool(d[0]) for d in train_directions}) == 2, "need both travel axes in training")
    _require(bool(test_directions - train_directions), "missing held-out direction shift")
    _require(len({x["background"] for x in train_infos}) >= 2, "need varied training palettes")
    _require(len(train_trails) >= 2, "local wake colors must be demonstrated")


def self_test():
    """Bounded semantic and diversity checks over 42 seeds; no files or network."""
    shapes, outlines, dock_shapes, backgrounds, directions, counts, signatures = set(), set(), set(), set(), set(), set(), set()
    concave = tapered = multi_active = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_fingerprint(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                info = _analyze(a)
                shapes.add(a.shape)
                backgrounds.add(info["background"])
                counts.add(len(info["cargo"]))
                multi_active += len(info["matches"]) >= 2
                for f in info["forks"]:
                    dock_shapes.add((f["width"], f["lengths"], tuple((r[1] - r[0]) for r in f["runs"])))
                for x in info["cargo"]:
                    lo = min(r for r, c in x["vertices"]), min(c for r, c in x["vertices"])
                    outlines.add(tuple(sorted(_shift(x["vertices"], (-lo[0], -lo[1])))))
                    concave += x["concave"]
                for m in info["matches"]:
                    x, f = info["cargo"][m["ci"]], info["forks"][m["fi"]]
                    directions.add(x["direction"])
                    qs = [r * f["across"][0] + c * f["across"][1] for r, c in x["pixels"]]
                    tapered += max(qs) - min(qs) + 1 > m["face_span"]
                # Whole-scene equivariance checks use no generation metadata.
                if seed < 3:
                    for turns in range(4):
                        rotated = np.rot90(a, turns)
                        assert np.array_equal(solve(rotated), np.rot90(pair["output"], turns))
                        assert np.array_equal(solve(np.fliplr(rotated)), np.fliplr(np.rot90(pair["output"], turns)))
    assert len(signatures) == 42 and len(shapes) >= 30
    assert len(outlines) >= 80 and len(dock_shapes) >= 60
    assert backgrounds == set(range(10)) and directions == set(DIRECTIONS)
    assert counts == {1, 2, 3} and min(concave, tapered, multi_active) > 10
    first = generate_puzzle(803, 2, 1)
    again = generate_puzzle(803, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s]) for k in p)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(19 + nt * 4 + nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    broken = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    p = broken["test"][0]
    m = _analyze(p["input"])["matches"][0]
    p["output"][min(m["final"])] = _analyze(p["input"])["background"]
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted final outline accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "normalized_contours": len(outlines), "dock_geometries": len(dock_shapes),
            "cargo_counts": sorted(counts), "active_directions": len(directions),
            "concave_cargo": int(concave), "tapered_dockings": int(tapered),
            "multiple_active_examples": int(multi_active)}
