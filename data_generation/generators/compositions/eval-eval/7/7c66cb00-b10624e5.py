"""Prismatic sorting docks: routed, scaled masks settle along crossed axes.

Only the input supplies colors, reference position, scales, and chamber geometry.
No parent generator is imported. Construction expands a sampled unit lattice;
the solver instead reconstructs exact rational binary rasters from input pixels.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np

FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 368743354, 367808565, 439783278, 44286557, 783529779,
              827113622, 952506526, 103748889, 750270055, 662425498,
              356488808, 70159526, 442992604, 106547907, 443068436,
              482283113, 87712647, 806612950, 140572809, 480761879,
              478525638, 280522610, 134629707, 763344609, 827164005,
              178879494, 830228135, 809542689, 520330098, 273388500,
              995351434, 813970121, 816540491, 88681398, 5672291,
              154894576, 983912910, 60761485, 174740449, 579658722, 20174878)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require 2..10 training, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(np.all((0 <= g) & (g <= 9)), "color range")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _panels(h, w, r, c):
    return ((0, r, 0, c), (0, r, c + 1, w),
            (r + 1, h, 0, c), (r + 1, h, c + 1, w))


def _directions(source, destination):
    return (int(np.sign(destination // 2 - source // 2)),
            int(np.sign(destination % 2 - source % 2)))


def _components(grid, points, neighbors=FOUR):
    unseen = set(points)
    result = []
    while unseen:
        start = min(unseen)
        color = int(grid[start])
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in unseen and int(grid[p]) == color:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append({"color": color, "cells": cells})
    return result


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _asymmetric(cells):
    cells = set(_normalize(cells))
    h, w = 1 + max(r for r, c in cells), 1 + max(c for r, c in cells)
    return (cells != {(h - 1 - r, c) for r, c in cells} and
            cells != {(r, w - 1 - c) for r, c in cells} and
            cells != {(h - 1 - r, w - 1 - c) for r, c in cells})


def _square(points):
    _require(bool(points), "missing square")
    r0, c0 = min(r for r, c in points), min(c for r, c in points)
    h = max(r for r, c in points) - r0 + 1
    w = max(c for r, c in points) - c0 + 1
    _require(h == w and len(points) == h * w, "marker must be one solid square")
    return r0, c0, h


def _decode(g):
    bg = int(g[0, 0])
    _require(all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))),
             "four background corners required")
    rows = [r for r in range(1, g.shape[0] - 1)
            if g[r, 0] != bg and np.all(g[r] == g[r, 0])]
    cols = [c for c in range(1, g.shape[1] - 1)
            if g[0, c] != bg and np.all(g[:, c] == g[0, c])]
    crosses = [(r, c) for r in rows for c in cols if g[r, c] == g[r, 0] == g[0, c]]
    _require(len(crosses) == 1, "unique full-length cross required")
    r, c = crosses[0]
    divider = int(g[r, c])
    bounds = _panels(*g.shape, r, c)
    panels = [g[y0:y1, x0:x1] for y0, y1, x0, x1 in bounds]
    references = [p for p, panel in enumerate(panels) if np.any(panel == divider)]
    _require(len(references) == 1, "one divider-colored calibration square required")
    reference = references[0]
    src = panels[reference]
    anchor_points = {tuple(map(int, p)) for p in np.argwhere(src == divider)}
    anchor = _square(anchor_points)
    points = {tuple(map(int, p)) for p in np.argwhere((src != bg) & (src != divider))}
    _require(bool(points), "empty source")
    _require(not any((r + dr, c + dc) in points
                     for r, c in anchor_points for dr, dc in EIGHT),
             "calibration needs a blank moat")
    comps = _components(src, points)
    _require(len(_components(src, points, EIGHT)) == len(comps),
             "diagonal-only same-color contact is outside this family")
    bays = {}
    for p, panel in enumerate(panels):
        if p == reference or np.all(panel == bg):
            continue
        occupied = np.argwhere(panel != bg)
        top, left = map(int, occupied.min(0))
        bottom, right = map(int, occupied.max(0))
        _require(bottom - top >= 4 and right - left >= 4, "chamber too small")
        rail = int(panel[top, left])
        _require(rail not in (bg, divider), "invalid chamber outline color")
        _require(np.all(panel[top, left:right + 1] == rail) and
                 np.all(panel[bottom, left:right + 1] == rail) and
                 np.all(panel[top:bottom + 1, left] == rail) and
                 np.all(panel[top:bottom + 1, right] == rail), "complete thin rectangular outline required")
        interior = panel[top + 1:bottom, left + 1:right]
        values = set(map(int, np.unique(interior))) - {bg}
        _require(len(values) == 1, "one address square inside each chamber")
        route = next(iter(values))
        _require(route not in (rail, divider), "address must differ from its own outline and divider")
        key_cells = {tuple(map(int, q)) for q in np.argwhere(panel == route)
                     if top < q[0] < bottom and left < q[1] < right}
        key = _square(key_cells)
        _require(route not in bays, "addresses must be unique")
        bays[route] = {"panel": p, "rail": rail, "key": key,
                       "key_cells": key_cells, "inside": (top + 1, bottom - 1, left + 1, right - 1),
                       "frame": (top, bottom, left, right)}
    _require(1 <= len(bays) <= 3, "require one to three chambers")
    _require(all(comp["color"] in bays for comp in comps), "unaddressed source component")
    return {"grid": g, "background": bg, "divider": divider, "bounds": bounds,
            "panels": panels, "reference": reference, "anchor": anchor,
            "components": comps, "bays": bays}


def _scaled_pixels(cells, anchor, side, vertical, horizontal):
    """Exact rational scaling, including sub-anchor detail and reductions.

    Source and destination share a common integer refinement. Every destination
    pixel must be entirely foreground or entirely blank; there is no rounding.
    Returned coordinates are relative to the destination key's upper-left.
    """
    ar, ac, source_side = anchor
    relative = {(r - ar, c - ac) for r, c in cells}
    top = min(r for r, c in relative) // source_side * source_side
    left = min(c for r, c in relative) // source_side * source_side
    bottom = (max(r for r, c in relative) // source_side + 1) * source_side
    right = (max(c for r, c in relative) // source_side + 1) * source_side
    raw = np.zeros((bottom - top, right - left), dtype=np.uint8)
    for r, c in relative:
        raw[r - top, c - left] = 1
    expanded = np.repeat(np.repeat(raw, side, axis=0), side, axis=1)
    h, w = expanded.shape[0] // source_side, expanded.shape[1] // source_side
    blocks = expanded.reshape(h, source_side, w, source_side)
    low, high = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    _require(np.array_equal(low, high), "scale straddles a foreground boundary")
    top, left = top * side // source_side, left * side // source_side
    if vertical:
        low = low[::-1]
        top = side - top - h
    if horizontal:
        low = low[:, ::-1]
        left = side - left - w
    return {(int(r) + top, int(c) + left) for r, c in np.argwhere(low)}


def _land(relative, bay, direction, docking=True):
    kr, kc, _ = bay["key"]
    cells = {(r + kr, c + kc) for r, c in relative}
    top, bottom, left, right = bay["inside"]
    v, h = direction
    dr = dc = 0
    if docking and v:
        dr = (bottom - max(r for r, c in cells) if v > 0
              else top - min(r for r, c in cells))
    if docking and h:
        dc = (right - max(c for r, c in cells) if h > 0
              else left - min(c for r, c in cells))
    full = {(r + dr, c + dc) for r, c in cells}
    visible = {(r, c) for r, c in full if top <= r <= bottom and left <= c <= right}
    return full, visible


def _render(data, *, group=False, scale=True, flips=(True, True), docking=True,
            recolor=True, strict=True):
    out = data["grid"].copy()
    source_bounds = data["bounds"][data["reference"]]
    sy, _, sx, _ = source_bounds
    comps = data["components"]
    for comp in comps:
        for r, c in comp["cells"]:
            out[sy + r, sx + c] = data["background"]
    if group:
        groups = {}
        for comp in comps:
            groups.setdefault(comp["color"], set()).update(comp["cells"])
        comps = [{"color": color, "cells": cells} for color, cells in sorted(groups.items())]
    records, used = [], set()
    for comp in comps:
        bay = data["bays"][comp["color"]]
        direction = _directions(data["reference"], bay["panel"])
        side = bay["key"][2] if scale else data["anchor"][2]
        relative = _scaled_pixels(comp["cells"], data["anchor"], side,
                                  bool(direction[0]) and flips[0], bool(direction[1]) and flips[1])
        full, visible = _land(relative, bay, direction, docking)
        if strict:
            _require(bool(visible), "a component disappeared entirely")
            _require(not (visible & bay["key_cells"]), "stamp overwrites address")
        visible -= bay["key_cells"]
        by, _, bx, _ = data["bounds"][bay["panel"]]
        absolute = {(r + by, c + bx) for r, c in visible}
        if strict:
            _require(not (used & absolute), "independent stamps collide")
        used.update(absolute)
        paint = bay["rail"] if recolor else comp["color"]
        for p in absolute:
            out[p] = paint
        records.append({"route": comp["color"], "panel": bay["panel"],
                        "direction": direction, "full": full, "visible": visible,
                        "lost": len(full) - len(visible), "side": side})
    return out, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve every valid family input without generation state or a learned map."""
    return _render(_decode(_grid(input_grid)))[0]


def _mask(rng, maximum, asymmetric=False):
    for _ in range(40):
        h, w = int(rng.integers(1, maximum + 1)), int(rng.integers(1, maximum + 1))
        kind = int(rng.integers(6))
        if kind == 0:
            cells = {(r, c) for r in range(h) for c in range(w)}
        elif kind == 1:
            cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
        elif kind == 2 and min(h, w) >= 3:
            cells = {(r, c) for r in range(h) for c in range(w)
                     if r in (0, h - 1) or c in (0, w - 1)}
            if w < maximum and rng.random() < .7:
                cells.add((int(rng.integers(h)), w))
        elif kind == 3:
            spine = int(rng.integers(w))
            cells = {(r, spine) for r in range(h)}
            for r in range(0, h, 2):
                end = int(rng.integers(w))
                cells.update((r, c) for c in range(min(spine, end), max(spine, end) + 1))
        else:
            cells = {(int(rng.integers(h)), int(rng.integers(w)))}
            goal = int(rng.integers(1, h * w + 1))
            while len(cells) < goal:
                frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in FOUR
                                   if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
                cells.add(frontier[int(rng.integers(len(frontier)))])
        cells = set(_normalize(cells))
        if not asymmetric or _asymmetric(cells):
            return cells
    return {(0, 0), (1, 0), (1, 1)}


def _source(rng, count, repeat, lesson, anchor_side, compact=False):
    if compact:
        anchor = (0, 0, 1)
        shapes = [{(0, 3), (1, 3), (1, 4)},
                  {(2, 3), (2, 4), (3, 4)},
                  {(4, 0), (4, 1), (5, 1)}]
        if repeat is not None:
            # Separated tangent lanes and different normal extrema in both axes.
            shapes = [{(0, 2), (1, 2), (1, 3)},
                      {(2, 0), (2, 1), (3, 0)}, shapes[2]]
        comps = [{"route": i, "cells": shapes[i]} for i in range(count)]
        if repeat is not None:
            comps.append({"route": repeat, "cells": {(4, 4), (4, 5), (5, 4)}})
        return anchor, comps
    h, w = int(rng.integers(6, 10)), int(rng.integers(6, 10))
    if lesson == "independent":
        h, w = min(h, 7), min(w, 7)
    ar, ac = int(rng.integers(h - anchor_side + 1)), int(rng.integers(w - anchor_side + 1))
    anchor = (ar, ac, anchor_side)
    anchor_cells = {(ar + r, ac + c) for r in range(anchor_side) for c in range(anchor_side)}
    array = np.full((h, w), -1, dtype=np.int8)
    for p in anchor_cells:
        array[p] = -2
    comps = []
    order = list(map(int, rng.permutation(count)))
    if repeat is not None:
        order += [repeat] * (1 + int(lesson != "independent" and rng.random() < .4))
    matched_mask = None
    for object_index, route in enumerate(order):
        if lesson == "split" and object_index == 1:
            mask = set(matched_mask)
            for _ in range(int(rng.integers(4))):
                mask = set(_normalize({(c, -r) for r, c in mask}))
        else:
            mask = _mask(rng, 4 if lesson else 5, asymmetric=lesson == "split")
        if object_index == 0:
            matched_mask = mask
        mh, mw = 1 + max(r for r, c in mask), 1 + max(c for r, c in mask)
        candidates = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
        rng.shuffle(candidates)
        placed = False
        for top, left in candidates:
            cells = {(r + top, c + left) for r, c in mask}
            if any(array[p] != -1 for p in cells):
                continue
            if any((r + dr, c + dc) in anchor_cells for r, c in cells for dr, dc in EIGHT):
                continue
            if any(0 <= r + dr < h and 0 <= c + dc < w and array[r + dr, c + dc] == route
                   for r, c in cells for dr, dc in EIGHT):
                continue
            for p in cells:
                array[p] = route
            comps.append({"route": route, "cells": cells})
            placed = True
            break
        if not placed:
            return None
    if lesson == "split" and not any(array[r + dr, c + dc] >= 0 and array[r + dr, c + dc] != route
                                    for comp in comps for route in [comp["route"]]
                                    for r, c in comp["cells"] for dr, dc in FOUR
                                    if 0 <= r + dr < h and 0 <= c + dc < w):
        return None
    return anchor, comps


def _unit_pixels(cells, anchor, unit, direction):
    ar, ac, a = anchor
    v, h = direction
    result = set()
    for r, c in cells:
        y, x = r - ar, c - ac
        if v:
            y = a - 1 - y
        if h:
            x = a - 1 - x
        result.update((y * unit + dr, x * unit + dc)
                      for dr in range(unit) for dc in range(unit))
    return result


def _bay_geometry(rng, relative_parts, side, direction, clipped):
    v, h = direction
    all_cells = set().union(*relative_parts)
    # Along an uncrossed axis the anchor-relative offsets must survive docking.
    minimums, maximums = [], []
    for axis in (0, 1):
        values = [p[axis] for p in all_cells]
        minimums.append(min([0] + values))
        maximums.append(max([side - 1] + values))
    extents = [max(max(p[axis] for p in cells) - min(p[axis] for p in cells) + 1
                   for cells in relative_parts) for axis in (0, 1)]
    dims, key = [], []
    for axis, crosses in enumerate((v, h)):
        if crosses:
            length = max(3, side + extents[axis] + int(rng.integers(1, 4)))
            pos = int(rng.integers(0, 2)) if crosses > 0 else length - side - int(rng.integers(0, 2))
        else:
            pad = int(rng.integers(0, 3))
            length = max(3, maximums[axis] - minimums[axis] + 1 + pad + int(rng.integers(0, 3)))
            pos = -minimums[axis] + pad
        dims.append(length)
        key.append(pos)
    if clipped:
        axis = 0 if not v else 1 if not h else None
        if axis is None:
            return None
        options = []
        for length in range(max(3, side), dims[axis] + 1):
            for pos in range(length - side + 1):
                counts = [sum(0 <= p[axis] + pos < length for p in cells) for cells in relative_parts]
                if all(counts) and sum(counts) < sum(map(len, relative_parts)):
                    options.append((length, pos))
        if not options:
            return None
        dims[axis], key[axis] = options[int(rng.integers(len(options)))]
    return dims, key


def _axis_lengths(rng, minimums):
    low = sum(minimums) + 1
    if low > 30:
        return None
    total = 30 if rng.random() < .15 else int(rng.integers(low, 31))
    extra = total - low
    first = int(rng.integers(extra + 1))
    return (minimums[0] + first, minimums[1] + extra - first)


def _build(rng, reference, count, lesson, clipping, compact=False):
    other_panels = [p for p in range(4) if p != reference]
    if lesson == "split":
        destinations = list(map(int, rng.permutation(other_panels)))
    elif lesson == "independent":
        destinations = list(map(int, rng.permutation([reference ^ 1, reference ^ 2])))
    else:
        destinations = list(map(int, rng.choice(other_panels, count, replace=False)))
    count = len(destinations)
    axial = [i for i, p in enumerate(destinations) if p != reference ^ 3]
    repeat = int(rng.choice(axial)) if axial and (lesson == "independent" or
                                                lesson is None and rng.random() < .5) else None
    # The reserve uses route zero for its repeated pair, guaranteeing lane separation.
    if compact and repeat is not None:
        i = axial[0]
        destinations[0], destinations[i] = destinations[i], destinations[0]
        repeat = 0
    a = 1 if compact else int(rng.choice([1, 1, 2]))
    source_unit = 2 if lesson == "independent" else 1 if compact or lesson == "split" else int(rng.choice([1, 1, 1, 2, 2, 3]))
    sampled = _source(rng, count, repeat, lesson, a, compact)
    if sampled is None:
        return None
    anchor, comps = sampled
    a = anchor[2]
    units = [int(rng.choice([1, 1, 2, 2, 3])) for _ in destinations]
    if lesson == "split":
        units = [1] * count
        units[int(rng.integers(count))] = 2 if compact else int(rng.choice([2, 2, 3]))
    elif lesson == "independent":
        units = [1] * count
    elif compact:
        units = [1] * count
    clip_at = int(rng.choice(axial)) if clipping and axial else None
    if clipping and clip_at is None:
        return None
    geometries = []
    for i, p in enumerate(destinations):
        parts = [_unit_pixels(comp["cells"], anchor, units[i], _directions(reference, p))
                 for comp in comps if comp["route"] == i]
        geo = _bay_geometry(rng, parts, a * units[i], _directions(reference, p), i == clip_at)
        if geo is None:
            return None
        geometries.append(geo)
    ar, ac, _ = anchor
    source_points = {(ar + r, ac + c) for r in range(a) for c in range(a)}
    source_points |= set().union(*(comp["cells"] for comp in comps))
    sr, sc = min(r for r, c in source_points), min(c for r, c in source_points)
    source_h = (max(r for r, c in source_points) - sr + 1) * source_unit
    source_w = (max(c for r, c in source_points) - sc + 1) * source_unit
    minimum_rows, minimum_cols = [3, 3], [3, 3]
    minimum_rows[reference // 2] = source_h + 2
    minimum_cols[reference % 2] = source_w + 2
    for p, (dims, _) in zip(destinations, geometries):
        minimum_rows[p // 2] = max(minimum_rows[p // 2], dims[0] + 4)
        minimum_cols[p % 2] = max(minimum_cols[p % 2], dims[1] + 4)
    heights, widths = _axis_lengths(rng, minimum_rows), _axis_lengths(rng, minimum_cols)
    if heights is None or widths is None:
        return None
    bounds = _panels(sum(heights) + 1, sum(widths) + 1, heights[0], widths[0])
    palette = list(map(int, rng.permutation(10)))
    bg, divider = palette[:2]
    routes, extras = palette[2:2 + count], palette[2 + count:]
    rails = [int(rng.choice([q for q in routes + extras if q != route]))
             if rng.random() < .55 else int(rng.choice(extras)) for route in routes]
    inp = np.full((sum(heights) + 1, sum(widths) + 1), bg, dtype=np.int8)
    inp[heights[0], :] = divider
    inp[:, widths[0]] = divider
    sy0, sy1, sx0, sx1 = bounds[reference]
    source_top = sy0 + int(rng.integers(1, sy1 - sy0 - source_h))
    source_left = sx0 + int(rng.integers(1, sx1 - sx0 - source_w))
    for r, c in source_points:
        color = divider if ar <= r < ar + a and ac <= c < ac + a else None
        if color is None:
            color = routes[next(comp["route"] for comp in comps if (r, c) in comp["cells"])]
        y, x = source_top + (r - sr) * source_unit, source_left + (c - sc) * source_unit
        inp[y:y + source_unit, x:x + source_unit] = color
    built_bays = []
    for i, (p, (dims, key)) in enumerate(zip(destinations, geometries)):
        y0, y1, x0, x1 = bounds[p]
        ih, iw = dims
        top = y0 + int(rng.integers(1, y1 - y0 - ih - 2))
        left = x0 + int(rng.integers(1, x1 - x0 - iw - 2))
        bottom, right = top + ih + 1, left + iw + 1
        inp[top, left:right + 1] = inp[bottom, left:right + 1] = rails[i]
        inp[top:bottom + 1, left] = inp[top:bottom + 1, right] = rails[i]
        kr, kc, side = top + 1 + key[0], left + 1 + key[1], a * units[i]
        inp[kr:kr + side, kc:kc + side] = routes[i]
        built_bays.append({"key": (kr, kc, side), "inside": (top + 1, bottom - 1, left + 1, right - 1),
                           "key_cells": {(kr + r, kc + c) for r in range(side) for c in range(side)}})
    # Labels use sampled lattice cells and known geometry, not solve or its decoder.
    out = inp.copy()
    out[sy0:sy1, sx0:sx1][np.isin(inp[sy0:sy1, sx0:sx1], routes)] = bg
    occupied = set()
    for comp in comps:
        i = comp["route"]
        direction = _directions(reference, destinations[i])
        relative = _unit_pixels(comp["cells"], anchor, units[i], direction)
        _, visible = _land(relative, built_bays[i], direction)
        if not visible or visible & (occupied | built_bays[i]["key_cells"]):
            return None
        occupied.update(visible)
        for p in visible:
            out[p] = rails[i]
    pair = {"input": inp, "output": out}
    try:
        data = _decode(inp)
        predicted, records = _render(data)
        if not np.array_equal(predicted, out):
            return None
        if lesson == "independent" and np.array_equal(out, _render(data, group=True, strict=False)[0]):
            return None
        if clipping and not any(r["lost"] for r in records):
            return None
    except AssertionError:
        return None
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    references = list(map(int, rng.permutation(4)))
    clipping_enabled = bool(rng.random() < .6)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = ("split" if index == 0 else "independent" if index == 1 else None) if split == "train" else None
            reference = references[index % 2] if split == "train" else references[2 + index % 2]
            chambers = 3 if lesson == "split" else 2 if lesson == "independent" else int(rng.integers(1, 4))
            clipping = clipping_enabled and (lesson == "independent" or lesson is None and rng.random() < .6)
            pair = None
            for attempt in range(256):
                candidate = _build(rng, reference, chambers, lesson, clipping, compact=attempt >= 128)
                if candidate is None or _key(candidate["input"]) in seen:
                    continue
                if split == "test":
                    data = _decode(candidate["input"])
                    if np.array_equal(candidate["output"],
                                      _render(data, scale=False, flips=(False, False), strict=False)[0]):
                        continue
                if candidate is not None:
                    pair = candidate
                    break
            _require(pair is not None, "bounded construction reserve exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
        if split == "train":
            rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(data, records):
    comps = data["components"]
    src = data["panels"][data["reference"]]
    points = set().union(*(c["cells"] for c in comps))
    contact = any((r + dr, c + dc) in points and src[r, c] != src[r + dr, c + dc]
                  for r, c in points for dr, dc in FOUR)
    ar, ac, side = data["anchor"]
    fine = any(offset % side for comp in comps for r, c in comp["cells"]
               for dr, dc, offset in ((-1, 0, r - ar), (1, 0, r + 1 - ar),
                                      (0, -1, c - ac), (0, 1, c + 1 - ac))
               if (r + dr, c + dc) not in comp["cells"])
    return {"reference": data["reference"], "shape": data["grid"].shape,
            "chambers": len(data["bays"]), "components": len(comps),
            "reference_side": data["anchor"][2], "contact": contact,
            "clipped": sum(r["lost"] > 0 for r in records),
            "fine": fine,
            "unused": len(data["bays"]) - len({c["color"] for c in comps}),
            "area_tie": any(x["color"] != y["color"] and len(x["cells"]) == len(y["cells"])
                            for i, x in enumerate(comps) for y in comps[i + 1:]),
            "ratios": {(r["side"], data["anchor"][2]) for r in records},
            "masks": tuple(_normalize(c["cells"]) for c in comps),
            "modes": {tuple(bool(x) for x in r["direction"]) for r in records},
            "roles": (data["background"], data["divider"],
                      tuple(sorted((color, b["rail"]) for color, b in data["bays"].items())))}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape preservation")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            data = _decode(inp)
            expected, records = _render(data)
            _require(np.array_equal(out, expected), "incorrect sorting-dock output")
            _require(not np.array_equal(inp, out), "identity pair")
            # Explicitly inspect preservation and per-component mass after clipping.
            sy0, sy1, sx0, sx1 = data["bounds"][data["reference"]]
            source_out = out[sy0:sy1, sx0:sx1]
            _require(not np.any(np.isin(source_out, list(data["bays"]))), "source not cleared")
            for color, bay in data["bays"].items():
                by0, by1, bx0, bx1 = data["bounds"][bay["panel"]]
                panel_out = out[by0:by1, bx0:bx1]
                _require(all(panel_out[p] == color for p in bay["key_cells"]), "address changed")
                top, bottom, left, right = bay["frame"]
                _require(np.all(panel_out[top, left:right + 1] == bay["rail"]) and
                         np.all(panel_out[bottom, left:right + 1] == bay["rail"]) and
                         np.all(panel_out[top:bottom + 1, left] == bay["rail"]) and
                         np.all(panel_out[top:bottom + 1, right] == bay["rail"]), "outline changed")
            target = training if split == "train" else testing
            target.append((data, out, records, _features(data, records)))
    _require(any(f["contact"] for _, _, _, f in training), "need a multicolor splitting lesson")
    _require(any(f["area_tie"] for _, _, _, f in training), "equal-area pieces must show color-based routing")
    _require(any(not np.array_equal(out, _render(d, group=True, strict=False)[0])
                 for d, out, _, _ in training), "need independent same-color docking evidence")
    for option in ({"scale": False}, {"flips": (False, True)}, {"flips": (True, False)},
                   {"docking": False}, {"recolor": False}):
        _require(any(not np.array_equal(out, _render(d, strict=False, **option)[0])
                     for d, out, _, _ in training), f"training fails to distinguish {option}")
    full_modes = set()
    for d, _, records, _ in training:
        for comp, rec in zip(d["components"], records):
            if not rec["lost"] and _asymmetric(comp["cells"]):
                full_modes.add(tuple(bool(x) for x in rec["direction"]))
    _require(full_modes == {(False, True), (True, False), (True, True)},
             "all crossed-axis modes need fully visible asymmetric demonstrations")
    _require(len({d["reference"] for d, _, _, _ in training}) >= 2, "show reference-relative behavior")
    if any(f["clipped"] for _, _, _, f in testing):
        _require(any(f["clipped"] for _, _, _, f in training), "clipping needs training evidence")
    if any(f["unused"] for _, _, _, f in testing):
        _require(any(f["unused"] for _, _, _, f in training), "unused chambers need training evidence")
    for d, out, _, _ in testing:
        _require(not np.array_equal(out, _render(d, scale=False, flips=(False, False), strict=False)[0]),
                 "test must exercise scaling or reflection")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p["input"])), _key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed checks, meaningful ablations, and geometric regressions."""
    shapes, masks, roles, references, counts, scales, signatures = set(), set(), set(), set(), set(), set(), set()
    clipping = repeated = fine = reused = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        if seed in HOST_SEEDS[:3]:
            _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
        train_refs = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _decode(pair["input"])
                _, records = _render(d)
                f = _features(d, records)
                shapes.add(f["shape"])
                roles.add(f["roles"])
                masks.update(f["masks"])
                references.add(f["reference"])
                counts.add(f["chambers"])
                scales.update((n // int(np.gcd(n, k)), k // int(np.gcd(n, k))) for n, k in f["ratios"])
                clipping += bool(f["clipped"])
                repeated += max(Counter(c["color"] for c in d["components"]).values()) > 1
                reused += any(b["rail"] in d["bays"] for b in d["bays"].values())
                fine += bool(f["fine"])
                if split == "train":
                    train_refs.add(d["reference"])
                else:
                    _require(d["reference"] not in train_refs, "test should move to a held-out source quadrant")
    _require(len(signatures) == 42, "42-seed puzzle diversity")
    _require(len(shapes) >= 35 and len(masks) >= 70 and len(roles) >= 100, "insufficient structural variation")
    _require(references == set(range(4)) and counts == {1, 2, 3}, "layout coverage")
    _require({(1, 1), (1, 2), (2, 1)} <= scales and clipping and repeated and fine and reused,
             "missing combined mechanism coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(123 + a * 7 + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exact fine enlargement, exact coarse reduction, and ambiguous reduction.
    cells = {(0, 2), (1, 2), (1, 3)}
    _require(_scaled_pixels(cells, (0, 0, 2), 4, False, True) ==
             {(r, c) for r, c in ((0, -2), (0, -1), (1, -2), (1, -1),
                                  (2, -2), (2, -1), (3, -2), (3, -1),
                                  (2, -4), (2, -3), (3, -4), (3, -3))}, "fine reflection")
    coarse = {(r, c) for r in (0, 1) for c in (4, 5)}
    _require(_scaled_pixels(coarse, (0, 0, 2), 1, False, False) == {(0, 2)}, "exact reduction")
    try:
        _scaled_pixels(cells, (0, 0, 2), 1, False, False)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous fractional scale accepted")
    # Far-wall/corner docking is computed before clipping and excludes the key.
    bay = {"key": (1, 1, 1), "inside": (0, 5, 0, 7)}
    full, visible = _land({(-2, 0), (-1, 0), (-1, 1)}, bay, (1, 1))
    _require(full == visible == {(4, 6), (5, 6), (5, 7)}, "corner docking")
    # Hand-specified full input/output: two independent right-docking Ls, a
    # doubled upward-reflected L, a clipped pixel, and an unused third address.
    fixture = np.zeros((19, 19), dtype=np.int8)
    fixture[8, :] = fixture[:, 9] = 7
    fixture[1, 1] = 7
    first = {(2, 4), (3, 4), (3, 5), (5, 3), (5, 4), (6, 3)}
    second = {(5, 0), (6, 0), (6, 1)}
    for p in first:
        fixture[p] = 1
    for p in second:
        fixture[p] = 2
    for top, bottom, left, right, rail in ((1, 7, 11, 17, 3),
                                            (10, 17, 1, 8, 4), (10, 17, 11, 17, 5)):
        fixture[top, left:right + 1] = fixture[bottom, left:right + 1] = rail
        fixture[top:bottom + 1, left] = fixture[top:bottom + 1, right] = rail
    fixture[2, 12] = 1
    fixture[11:13, 5:7] = 2
    fixture[11, 12] = 6
    expected = fixture.copy()
    for p in first | second:
        expected[p] = 0
    for p in ((3, 16), (4, 16), (4, 15), (6, 16), (6, 15)):
        expected[p] = 3
    expected[13:15, 3:7] = 4
    expected[15:17, 3:5] = 4
    _require(np.array_equal(solve(fixture), expected), "full pixel fixture")
    # Rational 3/2 scaling is legal on a 2-pixel lattice; it is not rounded.
    _require(_scaled_pixels(coarse, (0, 0, 2), 3, False, False) ==
             {(r, c) for r in range(3) for c in range(6, 9)}, "three-halves scale")
    for ref in range(4):
        for lesson in ("split", "independent"):
            p = None
            rng = np.random.default_rng(777 + ref)
            for _ in range(20):
                p = _build(rng, ref, 3 if lesson == "split" else 2, lesson,
                           lesson == "independent", compact=True)
                if p is not None:
                    break
            _require(p is not None and np.array_equal(solve(p["input"]), p["output"]), "reserve regression")
    broken = generate_puzzle(0, 2, 1)
    broken["test"][0]["output"] = broken["test"][0]["output"].copy()
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt label accepted")
    return {"seeds": 42, "grid_shapes": len(shapes), "component_masks": len(masks),
            "role_assignments": len(roles), "scale_ratios": sorted(scales),
            "clipped_pairs": clipping, "repeated_color_pairs": repeated,
            "fine_geometry_pairs": fine, "cross_role_reuse_pairs": reused}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
