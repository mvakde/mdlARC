"""Virtual diagonal docks: color routing with rigid placement in implicit regions.

Only NumPy and the standard library are used. No artifacts are written by this
module. All inference, including direction and the color map, is input-only.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations

import numpy as np


FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _components(grid, points, neighbors=FOUR):
    unseen = set(points)
    result = []
    while unseen:
        start = min(unseen)
        color = int(grid[start])
        unseen.remove(start)
        stack, cells = [start], set()
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in unseen and int(grid[p]) == color:
                    unseen.remove(p)
                    stack.append(p)
        result.append({"color": color, "cells": cells})
    return result


def _coordinates(p, slope, sign):
    r, c = p
    return c - slope * r, sign * (r + slope * c)


def _envelope(strokes, q, t, mode="full"):
    """Continuous intervals, including scanlines between marker centers."""
    if mode == "consecutive":
        pairs = zip(strokes, strokes[1:])
    else:
        pairs = combinations(strokes, 2)
    for a, b in pairs:
        if a[0] < q < b[0] and max(a[1], b[1]) <= t <= min(a[2], b[2]):
            if mode != "centers" or ((t - a[1]) % 2 == (t - b[1]) % 2 == 0):
                return True
    return False


def _parse(input_grid):
    grid = _grid(input_grid)
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) * 2 > grid.size, "background must occupy a strict majority")
    color_points = {int(c): {tuple(map(int, p)) for p in np.argwhere(grid == c)}
                    for c in values if int(c) != bg}
    candidates = []
    for slope in (-1, 1):
        for sign in (-1, 1):
            docks = []
            for paint, points in color_points.items():
                lines = {}
                for r, c in points:
                    lines.setdefault(c - slope * r, []).append(r)
                runs = []
                for q, rows in sorted(lines.items()):
                    current = []
                    for r in sorted(rows):
                        if current and r != current[-1] + 1:
                            runs.append((q, current))
                            current = []
                        current.append(r)
                    runs.append((q, current))
                if len(runs) < 2 or any(len(rs) < 2 for _, rs in runs):
                    continue
                caps, routes, strokes = set(), set(), []
                for q, rs in runs:
                    cap_r = rs[0] - 1 if sign == 1 else rs[-1] + 1
                    cap = (cap_r, q + slope * cap_r)
                    if not (0 <= cap[0] < grid.shape[0] and 0 <= cap[1] < grid.shape[1]):
                        break
                    route = int(grid[cap])
                    if route in (bg, paint):
                        break
                    far_r = rs[-1] if sign == 1 else rs[0]
                    _, lo = _coordinates(cap, slope, sign)
                    _, hi = _coordinates((far_r, q + slope * far_r), slope, sign)
                    strokes.append((q, lo, hi))
                    caps.add(cap)
                    routes.add(route)
                else:
                    if len(routes) == 1 and len({q for q, _, _ in strokes}) == len(strokes):
                        docks.append({"paint": paint, "route": routes.pop(), "strokes": sorted(strokes),
                                      "caps": caps, "rails": points | caps})
            if not docks or len(docks) > 3:
                continue
            paints = {d["paint"] for d in docks}
            routes = {d["route"] for d in docks}
            if len(routes) != len(docks) or paints & routes or paints | routes != set(color_points):
                continue
            rail_points = set().union(*(d["rails"] for d in docks))
            source = set().union(*color_points.values()) - rail_points
            if not source or any(int(grid[p]) not in routes for p in source):
                continue
            earliest = min(lo for d in docks for _, lo, _ in d["strokes"])
            if any(_coordinates(p, slope, sign)[1] >= earliest - 1 for p in source):
                continue
            # Docks are independent by disjoint longitudinal envelopes, never by distance.
            ordered = sorted(docks, key=lambda d: min(x[1] for x in d["strokes"]))
            if any(max(s[2] for s in a["strokes"]) + 1 >= min(s[1] for s in b["strokes"])
                   for a, b in zip(ordered, ordered[1:])):
                continue
            comps = _components(grid, source)
            if {c["color"] for c in comps} != routes:
                continue
            candidates.append({"grid": grid, "background": bg, "slope": slope, "sign": sign,
                               "docks": ordered, "components": comps, "source": source,
                               "rails": rail_points})
    _require(len(candidates) == 1, "input needs one visible dock interpretation")
    return candidates[0]


def _supports(info, mode="full"):
    grid = info["grid"]
    result = {}
    for dock in info["docks"]:
        cells = set()
        for r, c in np.argwhere(grid == info["background"]):
            p = (int(r), int(c))
            q, t = _coordinates(p, info["slope"], info["sign"])
            if _envelope(dock["strokes"], q, t, mode):
                cells.add(p)
        result[dock["route"]] = cells
    return result


def _land(cells, support, direction):
    dr, dc = direction
    # Within a <=30 canvas a positive diagonal translation is at most 29.
    for k in range(29, 0, -1):
        target = {(r + k * dr, c + k * dc) for r, c in cells}
        if target <= support:
            return k, target
    raise ValueError("component has no legal downstream placement")


def _placements(info, mode="full", together=False, nearest=False):
    supports = _supports(info, mode)
    direction = (info["sign"], info["sign"] * info["slope"])
    components = info["components"]
    if together:
        components = [{"color": route, "cells": set().union(*(c["cells"] for c in components
                                                              if c["color"] == route))}
                      for route in sorted(supports)]
    by_route = {d["route"]: d for d in info["docks"]}
    placed, occupied = [], set()
    for comp in components:
        route = comp["color"]
        if nearest:
            choices = []
            for dock in info["docks"]:
                try:
                    k, target = _land(comp["cells"], supports[dock["route"]], direction)
                    choices.append((k, dock["route"], target))
                except ValueError:
                    pass
            _require(choices, "no dock fits")
            k, route, target = min(choices, key=lambda x: x[0])
        else:
            k, target = _land(comp["cells"], supports[route], direction)
        _require(not target & occupied, "stamps collide")
        occupied.update(target)
        placed.append({"source": comp["cells"], "route": route, "paint": by_route[route]["paint"],
                       "shift": k, "target": target})
    return placed


def _render_solution(info, placed):
    out = info["grid"].copy()
    for p in info["source"]:
        out[p] = info["background"]
    for piece in placed:
        for p in piece["target"]:
            out[p] = piece["paint"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer capped strokes, color routing, continuous regions and rigid fits."""
    info = _parse(input_grid)
    return _render_solution(info, _placements(info))


def _shape(rng, limit, large=False):
    if large:
        # An asymmetric 4x4 mask, larger than every <=3x3 training mask.
        return {(r, c) for r in range(4) for c in range(4)} - {(0, 3), (3, 0)}
    h, w = map(int, rng.integers(1, limit + 1, size=2))
    kind = int(rng.integers(5))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1 and h >= 3 and w >= 3:
        cells = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 2:
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    else:
        cells = {(0, 0)}
        goal = int(rng.integers(1, h * w + 1))
        for _ in range(goal - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in FOUR
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _stroke_cells(stroke):
    q, lo, hi = stroke
    return [((t - q) // 2, (t + q) // 2) for t in range(lo, hi + 1, 2)]


def _make_docks(rng, count, lesson):
    width = int(rng.integers(12, 19))
    docks, boundary = [], {}
    cursor = 6
    for index in range(count):
        n = int(rng.integers(3 if lesson == "geometry" and index == 0 else 2, 6))
        choices = [qs for qs in combinations(range(3, width - 2), n - 2)
                   if all(b - a >= 3 for a, b in zip((0,) + qs, qs + (width,)))]
        inner = choices[int(rng.integers(len(choices)))]
        qs = [0, *inner, width]
        span = int(rng.integers(7, 11)) if count == 3 else int(rng.integers(9, 16))
        strokes = []
        for q in qs:
            lo = cursor + int(rng.integers(3))
            lo += (q - lo) % 2
            hi = cursor + span + int(rng.integers(-2, 3))
            hi -= (hi - q) % 2
            hi = max(hi, lo + 4)
            strokes.append((q, lo, hi))
        if lesson == "geometry" and index == 0:
            j = int(rng.integers(1, n - 1))
            q, lo, _ = strokes[j]
            strokes[j] = (q, lo, lo + 4)
            for j in (0, n - 1):
                q, lo, hi = strokes[j]
                hi = max(hi, cursor + 13)
                hi += (q - hi) % 2
                strokes[j] = (q, lo, hi)
            # Teach that the furthest endpoint alone is not a straight landing
            # edge: only one of these two boundaries continues this far.
            j = int(rng.choice([0, n - 1]))
            q, lo, hi = strokes[j]
            strokes[j] = (q, lo, hi + 4)
        rails = set()
        for stroke in strokes:
            cells = _stroke_cells(stroke)
            rails.update(cells)
            for j, p in enumerate(cells):
                boundary[p] = (index, j == 0)
        docks.append({"strokes": strokes, "rails": rails})
        cursor = max(hi for _, _, hi in strokes) + int(rng.integers(2, 5))
    # Pairwise geometry used by construction, independent of the raster parser.
    all_rails = set(boundary)
    for dock in docks:
        support = set()
        for q in range(1, width):
            for t in range(min(s[1] for s in dock["strokes"]), max(s[2] for s in dock["strokes"]) + 1):
                if (t - q) % 2 == 0 and _envelope(dock["strokes"], q, t):
                    support.add(((t - q) // 2, (t + q) // 2))
        dock["support"] = support - all_rails
    return docks, boundary, width


def _span(points):
    return (max(r for r, _ in points) - min(r for r, _ in points) + 1,
            max(c for _, c in points) - min(c for _, c in points) + 1)


def _put_piece(rng, shape, route, docks, width, source, occupied, boundary,
               force_q=None, avoid_shift=None):
    qmin = min(c - r for r, c in shape)
    qmax = max(c - r for r, c in shape)
    max_t = max(r + c for r, c in shape)
    qs = [int(q) for q in rng.permutation(np.arange(1 - qmin, width - qmax))]
    if force_q is not None:
        qs = [force_q]
    for q0 in qs:
        starts = [int(t) for t in rng.permutation(np.arange(-12, 5 - max_t)) if (t - q0) % 2 == 0]
        for t0 in starts:
            r0, c0 = (t0 - q0) // 2, (t0 + q0) // 2
            cells = {(r + r0, c + c0) for r, c in shape}
            if cells & set(source):
                continue
            # Source pieces are unambiguous under either 4- or 8-connectivity.
            if any((r + dr, c + dc) in source for r, c in cells for dr, dc in EIGHT):
                continue
            if max(_span(set(boundary) | set(source) | cells)) > 30:
                continue
            try:
                shift, target = _land(cells, docks[route]["support"], (1, 1))
            except ValueError:
                continue
            if target & occupied or shift == avoid_shift:
                continue
            source.update({p: route for p in cells})
            occupied.update(target)
            return {"route": route, "shape": shape, "q0": q0, "shift": shift,
                    "source": cells, "target": target}
    return None


def _attempt(rng, count, lesson, limit, rotation, large=False):
    docks, boundary, width = _make_docks(rng, count, lesson)
    source, occupied, pieces = {}, set(), []
    # The routing lesson sends congruent pieces on the same transverse line to
    # different docks; a third piece of a repeated color moves independently.
    if lesson == "routing":
        shape = _shape(rng, min(3, limit))
        if len({r for r, c in shape}) < 2 or len({c for r, c in shape}) < 2:
            return None
        first = _put_piece(rng, shape, 0, docks, width, source, occupied, boundary)
        if first is None:
            return None
        pieces.append(first)
        second = _put_piece(rng, shape, 1, docks, width, source, occupied, boundary, force_q=first["q0"])
        if second is None:
            return None
        pieces.append(second)
        third = _put_piece(rng, _shape(rng, limit), 0, docks, width, source, occupied, boundary,
                           avoid_shift=first["shift"])
        if third is None:
            return None
        pieces.append(third)
    for route in map(int, rng.permutation(count)):
        if any(p["route"] == route for p in pieces):
            continue
        shape = _shape(rng, limit, large=large and not pieces)
        piece = _put_piece(rng, shape, route, docks, width, source, occupied, boundary)
        if piece is None:
            return None
        pieces.append(piece)
    extras = int(rng.integers(0, 4))
    for _ in range(extras):
        route = int(rng.integers(count))
        piece = _put_piece(rng, _shape(rng, limit), route, docks, width, source, occupied, boundary)
        if piece is not None:
            pieces.append(piece)
    points = set(boundary) | set(source)
    h, w = _span(points)
    if max(h, w) > 30:
        return None
    pad_r = int(rng.integers(0, min(4, 30 - h) + 1))
    pad_c = int(rng.integers(0, min(4, 30 - w) + 1))
    top, left = int(rng.integers(pad_r + 1)), int(rng.integers(pad_c + 1))
    r0, c0 = min(r for r, c in points) - top, min(c for r, c in points) - left
    palette = list(map(int, rng.permutation(10)))
    bg, routes, paints = palette[0], palette[1:1 + count], palette[1 + count:1 + 2 * count]
    inp = np.full((h + pad_r, w + pad_c), bg, dtype=np.int8)
    for (r, c), (route, cap) in boundary.items():
        inp[r - r0, c - c0] = routes[route] if cap else paints[route]
    for (r, c), route in source.items():
        inp[r - r0, c - c0] = routes[route]
    out = inp.copy()
    for r, c in source:
        out[r - r0, c - c0] = bg
    for piece in pieces:
        for r, c in piece["target"]:
            out[r - r0, c - c0] = paints[piece["route"]]
    pair = {"input": np.rot90(inp, rotation).copy(), "output": np.rot90(out, rotation).copy()}
    try:
        info = _parse(pair["input"])
        if not np.array_equal(pair["output"], _render_solution(info, _placements(info))):
            return None
        if lesson == "geometry":
            features = _features(info)
            if not all(features[k] for k in ("inactive_middle", "continuous", "rigid", "straight_edge")):
                return None
    except ValueError:
        return None
    return pair


def _count(value, lo, hi, name):
    _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
             name + " must be an integer")
    _require(lo <= int(value) <= hi, "illegal " + name)
    return int(value)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # new dock count, larger footprint, or direction transfer
    turns = list(map(int, rng.permutation(4)))
    fixed = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, n in (("train", num_train), ("test", num_test)):
        for i in range(n):
            lesson = ("geometry" if i == 0 else "routing" if i == 1 else None) if split == "train" else None
            if split == "test" and regime == 0 and i == 0:
                count = 3
            elif lesson == "routing":
                count = 2
            else:
                count = int(rng.integers(1, 3 if regime == 0 and split == "train" else 4))
            if lesson == "geometry":
                count = min(count, 2)
            limit = 3 if regime == 1 and split == "train" else int(rng.integers(2, 5))
            if regime == 2:
                rotation = turns[i % 2] if split == "train" else turns[2 + i % 2]
            else:
                rotation = turns[0] if fixed else turns[i % 2]
            large = split == "test" and regime == 1 and i == 0
            for attempt in range(96):
                pair = _attempt(rng, count, lesson, limit, rotation, large)
                if pair is not None and _key(pair["input"]) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("bounded dock construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    placements = _placements(info)
    supports = _supports(info)
    consecutive, centers = _supports(info, "consecutive"), _supports(info, "centers")
    direction = (info["sign"], info["sign"] * info["slope"])
    by_route = {}
    dock_by_route = {d["route"]: d for d in info["docks"]}
    for p in placements:
        by_route.setdefault(p["route"], []).append(p)
    signatures = []
    for comp in info["components"]:
        r0 = min(r for r, c in comp["cells"])
        c0 = min(c for r, c in comp["cells"])
        signatures.append(tuple(sorted((r - r0, c - c0) for r, c in comp["cells"])))
    return {"docks": len(info["docks"]), "strokes": tuple(len(d["strokes"]) for d in info["docks"]),
            "direction": (info["sign"], info["sign"] * info["slope"]),
            "components": len(placements), "masks": signatures,
            "max_area": max(len(c["cells"]) for c in info["components"]),
            "inactive_middle": any(not p["target"] <= consecutive[p["route"]] for p in placements),
            "continuous": any(not p["target"] <= centers[p["route"]] for p in placements),
            "independent": any(len({p["shift"] for p in group}) > 1 for group in by_route.values()),
            "straight_edge": any(p["shift"] != (max(s[2] for s in dock_by_route[p["route"]]["strokes"])
                                                 - max(_coordinates(cell, info["slope"], info["sign"])[1]
                                                       for cell in p["source"])) // 2 for p in placements),
            "rigid": any(_land({cell}, supports[p["route"]], direction)[0] != p["shift"]
                         for p in placements for cell in p["source"]),
            "routing": any(p["route"] != info["docks"][0]["route"] and
                           _can_land(p["source"], supports[info["docks"][0]["route"]], direction)
                           for p in placements),
            "background": info["background"]}


def _can_land(cells, support, direction):
    try:
        _land(cells, support, direction)
        return True
    except ValueError:
        return False


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _parse(inp)
            placed = _placements(info)
            _require(np.array_equal(out, _render_solution(info, placed)), "incorrect output")
            _require(all(out[p] == inp[p] for p in info["rails"]), "rail or cap changed")
            _require(all(out[p] == info["background"] for p in info["source"]), "source not cleared")
            _require(len(_components(inp, info["source"], EIGHT)) == len(info["components"]),
                     "ambiguous source connectivity")
            # Separate active-extrema check of every implicit region.
            supports = _supports(info)
            for dock in info["docks"]:
                for r in range(inp.shape[0]):
                    for c in range(inp.shape[1]):
                        q, t = _coordinates((r, c), info["slope"], info["sign"])
                        active = [s[0] for s in dock["strokes"] if s[1] <= t <= s[2]]
                        expected = (inp[r, c] == info["background"] and len(active) >= 2
                                    and min(active) < q < max(active))
                        _require(((r, c) in supports[dock["route"]]) == bool(expected), "envelope mismatch")
            features[split].append(_features(info))
    for name in ("inactive_middle", "continuous", "independent", "routing", "rigid", "straight_edge"):
        _require(any(f[name] for f in features["train"]), "missing training witness: " + name)
    # Test-only orientation is supported by observing at least two train directions.
    dirs = {f["direction"] for f in features["train"]}
    if any(f["direction"] not in dirs for f in features["test"]):
        _require(len(dirs) >= 2, "unsupported direction transfer")
    return None


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded tests; streams 42 puzzles, retains only small feature sets."""
    hashes, shapes, masks, layouts, backgrounds, directions, counts = (set() for _ in range(7))
    flags = {k: 0 for k in ("inactive_middle", "continuous", "independent", "routing", "rigid", "straight_edge")}
    shifts = {"more_docks": 0, "larger_piece": 0, "new_direction": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        hashes.add(_fingerprint(puzzle))
        local = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _parse(pair["input"])
                f = _features(info)
                local[split].append(f)
                shapes.add(pair["input"].shape)
                masks.update(f["masks"])
                backgrounds.add(info["background"])
                directions.add(f["direction"])
                counts.add(f["docks"])
                layouts.add(tuple(tuple((q - d["strokes"][0][0], lo - d["strokes"][0][1],
                                         hi - d["strokes"][0][1]) for q, lo, hi in d["strokes"])
                                  for d in info["docks"]))
                for k in flags:
                    flags[k] += int(f[k])
        evidence = {
            "more_docks": any(t["docks"] > max(f["docks"] for f in local["train"]) for t in local["test"]),
            "larger_piece": any(t["max_area"] > max(f["max_area"] for f in local["train"]) for t in local["test"]),
            "new_direction": any(t["direction"] not in {f["direction"] for f in local["train"]}
                                 for t in local["test"])}
        _require(any(evidence.values()), "missing observable train/test shift")
        for k in shifts:
            shifts[k] += int(evidence[k])
    _require(len(hashes) == 42 and len(shapes) >= 30 and len(layouts) >= 100, "insufficient structural variation")
    _require(len(masks) >= 25 and len(backgrounds) >= 8 and len(directions) == 4 and counts == {1, 2, 3},
             "insufficient geometry/palette/direction variation")
    _require(all(shifts.values()), "missing shift regime")
    # Hand-calculated geometry: two long rails, a short middle rail, an L and
    # a separated singleton. The L shifts 10; the singleton shifts 11.
    fixture = np.zeros((22, 20), dtype=np.int8)
    for stroke in ((0, 8, 20), (6, 8, 12), (12, 8, 20)):
        for j, (r, c) in enumerate(_stroke_cells(stroke)):
            fixture[r + 6, c + 1] = 1 if j == 0 else 2
    fixture[(3, 3, 4, 1), (3, 4, 3, 4)] = 1
    expected = fixture.copy()
    expected[(3, 3, 4, 1), (3, 4, 3, 4)] = 0
    expected[(13, 13, 14, 12), (13, 14, 13, 15)] = 2
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(fixture, turn)), np.rot90(expected, turn)),
                 "hand-calculated interval/rigid placement fixture failed")
    recolored = fixture.copy()
    recolored[fixture == 1] = 3
    recolored[fixture == 2] = 4
    recolored_out = expected.copy()
    recolored_out[expected == 1] = 3
    recolored_out[expected == 2] = 4
    insufficient = {"train": [{"input": fixture, "output": expected},
                              {"input": recolored, "output": recolored_out}],
                    "test": [{"input": np.rot90(fixture), "output": np.rot90(expected)}]}
    try:
        validate_generated_puzzle(insufficient)
    except ValueError as exc:
        _require("missing training witness" in str(exc), "wrong reason for rejecting insufficient teaching")
    else:
        raise AssertionError("correct labels with insufficient teaching were accepted")
    # Inclusive endpoints and half-step scanlines, checked directly.
    intervals = [(0, 8, 20), (6, 8, 12), (12, 8, 20)]
    _require(_envelope(intervals, 6, 20) and not _envelope(intervals, 6, 21), "endpoint semantics")
    _require(_envelope(intervals, 5, 19) and not _envelope(intervals, 5, 19, "centers"),
             "continuous interval semantics")
    _require(not _envelope(intervals, 5, 19, "consecutive"), "inactive middle witness")
    p = generate_puzzle(11, 2, 1)
    _require(_fingerprint(p) == _fingerprint(generate_puzzle(11, 2, 1)), "nondeterministic")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(100 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(hashes), "shapes": len(shapes),
            "dock_layouts": len(layouts), "component_masks": len(masks), "backgrounds": len(backgrounds),
            "directions": len(directions), "dock_counts": sorted(counts), "witness_pairs": flags,
            "observable_shift_puzzles": shifts}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
