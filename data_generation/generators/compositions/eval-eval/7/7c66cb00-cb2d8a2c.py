"""Shadow-steered cargo: component masks execute color-selective baffle commands.

Only NumPy and the standard library are used. Geometry and palette conventions
are decoded from each input. No parent module, seed, or training labels are used
by solve. Coordinates in helpers are canonical: sources above, docks below.
"""
from __future__ import annotations

import json
import numpy as np


_FOUR = ((-1, 0), (1, 0), (0, -1), (0, 1))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((a >= 0) & (a <= 9)), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(grid, points, diagonal=False):
    remaining = set(points)
    result = []
    steps = _EIGHT if diagonal else _FOUR
    while remaining:
        start = min(remaining)
        color = int(grid[start])
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining and int(grid[p]) == color:
                    remaining.remove(p)
                    stack.append(p)
        result.append({"color": color, "cells": tuple(sorted(cells))})
    return result


def _signature(cells):
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _dock_row(row, bg):
    occupied = np.flatnonzero(row != bg)
    if len(occupied) < 7:
        return None
    lo, hi = int(occupied[0]), int(occupied[-1])
    if (len(occupied) == hi-lo+1 and row[lo] == row[hi] and
            row[lo] != row[lo+1] and np.all(row[lo+1:hi] == row[lo+1])):
        return lo, hi, int(row[lo+1]), int(row[lo])
    return None


def _canonical_decode(a, bg):
    h, w = a.shape
    _require(a[0, 0] == a[0, -1] == bg, "staging corners reveal background")
    # Peel the complete dock suffix; neither an asymmetric baffle nor a source
    # can have the two-rail pattern with a non-routing rail color.
    cursor, reversed_docks = h-1, []
    while cursor >= 0:
        while cursor >= 0 and np.all(a[cursor] == bg):
            cursor -= 1
        if cursor < 0:
            break
        info = _dock_row(a[cursor], bg)
        if info is None:
            break
        bottom = cursor
        while cursor >= 0 and _dock_row(a[cursor], bg) == info:
            cursor -= 1
        lo, hi, route, rail = info
        reversed_docks.append(dict(top=cursor+1, bottom=bottom, left=lo,
                                   right=hi, route=route, rail=rail))
    docks = list(reversed(reversed_docks))
    _require(1 <= len(docks) <= 4, "one to four docks required")
    _require(all(1 <= d["bottom"]-d["top"]+1 <= 5 for d in docks), "dock depth")
    _require(all(y["top"] >= x["bottom"]+2 for x, y in zip(docks, docks[1:])),
             "docks need blank separators")
    routes = {d["route"] for d in docks}
    rails = {d["rail"] for d in docks}
    _require(len(routes) == len(docks) and not routes & rails and bg not in routes | rails,
             "distinct route roles; rail roles outside the routing palette")
    lookup = {d["route"]: d for d in docks}
    first_dock = docks[0]["top"]
    _require(first_dock >= 4 and np.all(a[first_dock-1] == bg), "dock separator")
    gates, source_points = [], []
    for r in range(first_dock):
        row = a[r]
        if np.all(row == bg):
            continue
        if not any(np.any(row == rail) for rail in rails):
            _require(set(map(int, row)) <= routes | {bg}, "source colors")
            source_points.extend((r, int(c)) for c in np.flatnonzero(row != bg))
            continue
        occupied = np.flatnonzero(row != bg)
        lo, hi = int(occupied[0]), int(occupied[-1])
        _require(len(occupied) == hi-lo+1, "a baffle is contiguous")
        colors = set(map(int, row[lo:hi+1]))
        candidates = [route for route, dock in lookup.items()
                      if colors == {route, dock["rail"]}]
        _require(len(candidates) == 1, "baffle body and marker roles must match a dock")
        route = candidates[0]
        marker = lookup[route]["rail"]
        marks = tuple(map(int, np.flatnonzero(row == marker)))
        n = len(marks)
        _require(1 <= n <= 4 and hi-lo+1 >= 2*n, "asymmetric unary baffle")
        low = marks == tuple(lo+2*j for j in range(n))
        high = marks == tuple(hi-2*j for j in reversed(range(n)))
        _require(low != high, "one marker-bearing endpoint required")
        gates.append(dict(row=r, left=lo, right=hi, color=route, marker=marker,
                          n=n, side=1 if low else -1, marks=marks))
    _require(1 <= len(gates) <= 5 and source_points, "cargo and one to five baffles required")
    _require(max(r for r, c in source_points)+2 <= gates[0]["row"], "staging precedes baffles")
    _require(all(y["row"] >= x["row"]+2 for x, y in zip(gates, gates[1:])),
             "baffle rows need blank separators")
    comps = _components(a, source_points)
    _require({c["color"] for c in comps} == routes, "every dock receives cargo")
    return dict(canonical=a, background=bg, docks=docks, gates=gates, components=comps)


def _transport(cells, color, gates, *, mode="normal"):
    lo, hi = min(c for r, c in cells), max(c for r, c in cells)
    initial_lo, initial_hi = lo, hi
    offset, events, previous_side = 0, [], None
    for gate in (reversed(gates) if mode == "reverse" else gates):
        match = gate["color"] == color or mode == "all_colors"
        overlap = lo <= gate["right"] and hi >= gate["left"]
        if mode == "source_projection":
            overlap = initial_lo <= gate["right"] and initial_hi >= gate["left"]
        hit = match and (overlap or mode == "unconditional") and mode != "no_baffles"
        if hit:
            d = 2 if mode == "fixed_gap" else gate["n"] if mode == "count_only" else gate["n"]+1
            side = -gate["side"] if mode == "marked_end" else gate["side"]
            if mode == "alternate" and previous_side is not None:
                side = -previous_side
            if mode == "center_clearance":
                center = (lo+hi)//2
                delta = gate["right"]+d-center if side == 1 else gate["left"]-d-center
            else:
                left_anchor = lo if mode == "point_clearance" else hi
                delta = gate["right"]+d-lo if side == 1 else gate["left"]-d-left_anchor
            lo, hi = lo+delta, hi+delta
            offset += delta
            previous_side = side
        events.append(dict(gate=gate, match=match, overlap=overlap, hit=hit,
                           interval=(lo, hi), offset=offset))
    return offset, events


def _targets(analysis, mode="normal", grouping="components", check=True):
    comps = analysis["components"]
    if grouping == "pixels":
        comps = [{"color": c["color"], "cells": (p,)} for c in comps for p in c["cells"]]
    elif grouping == "colors":
        comps = [{"color": color, "cells": tuple(p for c in comps if c["color"] == color
                                                  for p in c["cells"])}
                 for color in sorted({c["color"] for c in comps})]
    docks = {d["route"]: d for d in analysis["docks"]}
    occupied, placements = set(), []
    for comp in comps:
        dock = docks[comp["color"]]
        dx, events = _transport(comp["cells"], comp["color"], analysis["gates"], mode=mode)
        bottom = max(r for r, c in comp["cells"])
        cells = {(dock["bottom"]+r-bottom, c+dx) for r, c in comp["cells"]}
        if check:
            _require(all(0 <= lo <= hi < analysis["canonical"].shape[1]
                         for lo, hi in (e["interval"] for e in events)), "intermediate cargo outside canvas")
            _require(all(dock["top"] <= r <= dock["bottom"] and
                         dock["left"] < c < dock["right"] for r, c in cells), "cargo does not fit its dock")
            _require(not occupied & cells, "actual landing cell collision")
        occupied.update(cells)
        placements.append(dict(component=comp, dock=dock, cells=cells, events=events, dx=dx))
    return placements


def _landing_key(analysis, mode="normal", grouping="components"):
    # Counterfactuals may be out of bounds; coordinates still give a useful
    # semantic witness, without inventing a clipping convention.
    return frozenset((r, c, p["dock"]["rail"])
                     for p in _targets(analysis, mode, grouping, check=False) for r, c in p["cells"])


def _render(analysis):
    out = analysis["canonical"].copy()
    for comp in analysis["components"]:
        for p in comp["cells"]:
            out[p] = analysis["background"]
    for gate in analysis["gates"]:
        out[gate["row"], list(gate["marks"])] = gate["color"]
    for placement in _targets(analysis):
        for p in placement["cells"]:
            out[p] = placement["dock"]["rail"]
    return out


def _decode(input_grid):
    a = _grid(input_grid)
    candidates = []
    for rotation in range(4):
        canonical = np.rot90(a, rotation).copy()
        bg = int(canonical[0, 0])
        try:
            parsed = _canonical_decode(canonical, bg)
            _targets(parsed)
        except AssertionError:
            continue
        parsed["rotation"] = rotation
        candidates.append(parsed)
    _require(len(candidates) == 1, "one staging/baffle/dock orientation required")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the input-derived final state; never consult generation state."""
    analysis = _decode(input_grid)
    return np.rot90(_render(analysis), -analysis["rotation"]).copy()


def _mask(rng, maxh, maxw):
    h, w = int(rng.integers(1, maxh+1)), int(rng.integers(1, maxw+1))
    kind = int(rng.integers(5))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        cells = {(r, 0) for r in range(h)} | {(h-1, c) for c in range(w)}
    elif kind == 2:
        spine = int(rng.integers(w))
        cells = {(r, spine) for r in range(h)}
        for r in range(h):
            end = int(rng.integers(w))
            cells.update((r, c) for c in range(min(spine, end), max(spine, end)+1))
    elif kind == 3:
        r = c = 0
        cells = {(r, c)}
        while r < h-1 or c < w-1:
            if r < h-1 and (c == w-1 or rng.random() < .5):
                r += 1
            else:
                c += 1
            cells.add((r, c))
    else:
        cells = {(int(rng.integers(h)), int(rng.integers(w)))}
        goal = int(rng.integers(1, h*w+1))
        while len(cells) < goal:
            options = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in _FOUR
                              if 0 <= r+dr < h and 0 <= c+dc < w and (r+dr, c+dc) not in cells})
            cells.add(options[int(rng.integers(len(options)))])
    return _signature(cells)


def _place_group(stage, group, bg, rng):
    height = 1+max(r for color, cells in group for r, c in cells)
    width = 1+max(c for color, cells in group for r, c in cells)
    options = [(r, c) for r in range(stage.shape[0]-height+1)
               for c in range(1, stage.shape[1]-width)]
    for index in rng.permutation(len(options)):
        top, left = options[int(index)]
        assignment = {(r+top, c+left): color for color, cells in group for r, c in cells}
        if len(assignment) != sum(len(cells) for color, cells in group):
            continue
        if any(stage[p] != bg for p in assignment):
            continue
        valid = True
        for (r, c), color in assignment.items():
            if any(0 <= r+dr < stage.shape[0] and 0 <= c+dc < stage.shape[1] and
                   (r+dr, c+dc) not in assignment and stage[r+dr, c+dc] == color for dr, dc in _EIGHT):
                valid = False
                break
        if valid:
            for p, color in assignment.items():
                stage[p] = color
            return True
    return False


def _frame(rng, outer, inner, wide=False):
    h = int(rng.integers(3, 6))
    w = int(rng.integers(6, 9)) if wide else int(rng.integers(3, 6))
    wall = {(r, c) for r in range(h) for c in range(w) if r in (0, h-1) or c in (0, w-1)}
    # Sometimes extend one side at a middle row; geometry, not area, is routed.
    if not wide and w < 5 and rng.random() < .35:
        wall.add((int(rng.integers(1, h-1)), w))
    fill = {(r, c) for r in range(1, h-1) for c in range(1, w-1)}
    if rng.random() < .45:
        fill = {(1+r, 1+c) for r, c in _mask(rng, h-2, w-2)}
    return [(outer, tuple(sorted(wall))), (inner, tuple(sorted(fill)))]


def _gate_options(width, comps, color, offsets, anchor, n, side=None, miss_other=False):
    intervals = [(min(c for r, c in comp["cells"])+dx,
                  max(c for r, c in comp["cells"])+dx)
                 for comp, dx in zip(comps, offsets)]
    result = []
    for sign in ((-1, 1) if side is None else (side,)):
        for lo in range(width):
            for hi in range(lo+2*n-1, width):
                if not (intervals[anchor][0] <= hi and intervals[anchor][1] >= lo):
                    continue
                hit = [i for i, (x, y) in enumerate(intervals)
                       if comps[i]["color"] == color and x <= hi and y >= lo]
                if miss_other and len(hit) == sum(c["color"] == color for c in comps):
                    continue
                updates = {}
                for i in hit:
                    x, y = intervals[i]
                    delta = hi+n+1-x if sign == 1 else lo-n-1-y
                    if not (1 <= x+delta <= y+delta <= width-2):
                        break
                    updates[i] = offsets[i]+delta
                else:
                    result.append((lo, hi, sign, updates))
    return result


def _has_contact(a):
    source = {p: comp["color"] for comp in a["components"] for p in comp["cells"]}
    return any(source.get((r+dr, c+dc), color) != color
               for (r, c), color in source.items() for dr, dc in _FOUR)


def _features(a):
    placements = _targets(a)
    programs = [[e["gate"] for e in p["events"] if e["hit"]] for p in placements]
    points = {p for c in a["components"] for p in c["cells"]}
    normal = _landing_key(a)
    return dict(
        direction=a.get("rotation", 0), shape=tuple(a["canonical"].shape),
        routes=len(a["docks"]), gates=len(a["gates"]),
        components=len(a["components"]), contact=_has_contact(a),
        max_width=max(1+max(c for r, c in _signature(x["cells"])) for x in a["components"]),
        masks=tuple(_signature(c["cells"]) for c in a["components"]),
        active_counts={g["n"] for program in programs for g in program},
        active_sides={g["side"] for program in programs for g in program},
        sequential=any(len(program) >= 2 for program in programs),
        repeat=any(x["side"] == y["side"] for program in programs for x, y in zip(program, program[1:])),
        miss=any(e["match"] and not e["overlap"] for p in placements for e in p["events"]),
        wrong_color_overlap=any(not e["match"] and e["overlap"] for p in placements for e in p["events"]),
        independent=normal != _landing_key(a, grouping="colors"),
        extent=normal != _landing_key(a, grouping="pixels"),
        baffle_effect=normal != _landing_key(a, "no_baffles"),
        conditional_effect=normal != _landing_key(a, "unconditional"),
        selective_effect=normal != _landing_key(a, "all_colors"),
        count_effect=normal != _landing_key(a, "fixed_gap"),
        order_effect=normal != _landing_key(a, "reverse"),
        projection_effect=normal != _landing_key(a, "source_projection"),
        edge_effect=(normal != _landing_key(a, "point_clearance") and
                     normal != _landing_key(a, "center_clearance")),
        alternation_effect=normal != _landing_key(a, "alternate"),
        same_connectivity=len(_components(a["canonical"], points, True)) == len(a["components"]),
        dx=tuple(p["dx"] for p in placements),
        shared_rails=len({d["rail"] for d in a["docks"]}) < len(a["docks"]),
    )


def _attempt_pair(rng, maximum, lesson, gate_count, rotation, wide=False, repeat=False,
                  teaching_counts=None):
    count = int(rng.integers(2 if lesson or wide else 1, 5))
    stage_height = int(rng.integers(5, 9))
    width = int(rng.integers(18 if lesson else 12, 31))
    palette = list(map(int, rng.permutation(10)))
    bg, routes = palette[0], palette[1:count+1]
    rails = palette[count+1:2*count+1]
    if count > 1 and rng.random() < .3:
        rails[int(rng.integers(count))] = rails[0]
    stage = np.full((stage_height, width), bg, dtype=np.int8)
    primary = routes[0]
    if lesson == "split" or wide or (count >= 2 and rng.random() < .45):
        if not _place_group(stage, _frame(rng, primary, routes[1], wide), bg, rng):
            return None
    for color in routes:
        if not np.any(stage == color):
            if not _place_group(stage, [(color, _mask(rng, 5, 5))], bg, rng):
                return None
    if lesson == "independent":
        # Keep drawing until a second component exists at a different depth.
        old_bottom = int(np.argwhere(stage == primary)[:, 0].max())
        before = stage.copy()
        for _ in range(12):
            stage = before.copy()
            if _place_group(stage, [(primary, _mask(rng, 5, 5))], bg, rng):
                pts = [tuple(map(int, p)) for p in np.argwhere(stage == primary)]
                pieces = _components(stage, pts)
                if len(pieces) >= 2 and any(max(r for r, c in p["cells"]) != old_bottom for p in pieces):
                    break
        else:
            return None
    # Additional pieces are sampled through the same grammar in both splits.
    for _ in range(int(rng.integers(3)) if not lesson else int(rng.integers(2))):
        color = int(rng.choice(routes))
        _place_group(stage, [(color, _mask(rng, 5, 5))], bg, rng)
    source_points = [tuple(map(int, p)) for p in np.argwhere(stage != bg)]
    comps = _components(stage, source_points)
    if len(_components(stage, source_points, True)) != len(comps):
        return None
    indices = {color: [i for i, c in enumerate(comps) if c["color"] == color] for color in routes}
    anchor_a = max(indices[primary], key=lambda i: len(comps[i]["cells"]))
    offsets, gates = [0]*len(comps), []
    repeat_side = int(rng.choice((-1, 1))) if repeat else None
    for j in range(gate_count):
        if lesson == "split" and j < 2:
            color, anchor, n = primary, anchor_a, teaching_counts[j]
            side = repeat_side
        elif lesson == "split" and j == 2:
            color = routes[1]
            anchor, n, side = indices[color][0], int(rng.integers(1, maximum+1)), None
        elif lesson == "independent" and j == 0:
            color, anchor, n, side = primary, anchor_a, teaching_counts[0], None
        elif lesson == "independent" and j == 1:
            color = routes[1]
            anchor, n, side = indices[color][0], teaching_counts[1], None
        else:
            color = int(rng.choice(routes))
            anchor, n, side = int(rng.choice(indices[color])), int(rng.integers(1, maximum+1)), None
        options = _gate_options(width, comps, color, offsets, anchor, n, side,
                                miss_other=lesson == "independent" and j == 0)
        if not options:
            return None
        lo, hi, sign, updates = options[int(rng.integers(len(options)))]
        for i, dx in updates.items():
            offsets[i] = dx
        gates.append(dict(left=lo, right=hi, color=color, marker=rails[routes.index(color)],
                          n=n, side=sign))
    # Source, gate and dock spacing are independently budgeted. Marker counts
    # do not constrain row gaps: this family encodes lateral clearance only.
    gate_gaps = list(map(int, rng.integers(1, 3, size=gate_count)))
    dock_gaps = list(map(int, rng.integers(1, 3, size=count)))
    minimum_depths = [max(1+max(r for r, c in _signature(comps[i]["cells"])) for i in indices[color])
                      for color in routes]
    minimum = stage_height+sum(gate_gaps)+gate_count+sum(dock_gaps)+sum(minimum_depths)
    if minimum > 30:
        return None
    slack = int(rng.integers(min(30-minimum, 7)+1))
    depths = minimum_depths[:]
    far_margin = 0
    for _ in range(slack):
        k = int(rng.integers(count+1))
        if k == count or depths[k] == 5:
            far_margin += 1
        else:
            depths[k] += 1
    a = np.full((minimum+slack, width), bg, dtype=np.int8)
    a[:stage_height] = stage
    cursor = stage_height
    for gap, gate in zip(gate_gaps, gates):
        cursor += gap
        gate["row"] = cursor
        gate["marks"] = tuple(gate["left"]+2*j if gate["side"] == 1 else gate["right"]-2*j
                              for j in range(gate["n"]))
        a[cursor, gate["left"]:gate["right"]+1] = gate["color"]
        a[cursor, list(gate["marks"])] = gate["marker"]
        cursor += 1
    docks = []
    # Shuffle destination order independently of masks, areas and gate order.
    for index in rng.permutation(count):
        i = int(index)
        cursor += dock_gaps[i]
        cols = [c+offsets[k] for k in indices[routes[i]] for r, c in comps[k]["cells"]]
        left = int(rng.integers(min(3, min(cols))))
        right = width-1-int(rng.integers(min(3, width-1-max(cols))))
        dock = dict(top=cursor, bottom=cursor+depths[i]-1, left=left,
                    right=right, route=routes[i], rail=rails[i])
        docks.append(dock)
        a[cursor:cursor+depths[i], left:right+1] = routes[i]
        a[cursor:cursor+depths[i], [left, right]] = rails[i]
        cursor += depths[i]
    analysis = dict(canonical=a, background=bg, docks=docks, gates=gates, components=comps)
    try:
        out = _render(analysis)
        features = _features(analysis)
        if not (features["extent"] and features["baffle_effect"]):
            return None
        if lesson == "split" and not (features["contact"] and features["sequential"] and
                                       features["selective_effect"] and features["count_effect"] and
                                       features["order_effect"] and features["projection_effect"] and
                                       (not repeat or (features["repeat"] and features["alternation_effect"]))):
            return None
        if lesson == "independent" and not (features["independent"] and features["conditional_effect"] and
                                             features["edge_effect"]):
            return None
        pair = {"input": np.rot90(a, rotation).copy(), "output": np.rot90(out, rotation).copy()}
        parsed = _decode(pair["input"])
        _require(np.array_equal(pair["output"], np.rot90(_render(parsed), -parsed["rotation"])),
                 "construction/parser disagree")
        return pair
    except AssertionError:
        return None


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    maximum = int(rng.integers(2, 5))
    shift = int(rng.integers(3))  # held-out direction, longer program, wider cargo
    turns = list(map(int, rng.permutation(4)))
    mixed = bool(rng.integers(2))
    repeat = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    training_features = []
    for split, size in (("train", num_train), ("test", num_test)):
        for index in range(size):
            lesson = ("split" if index == 0 else "independent" if index == 1 else None) if split == "train" else None
            gate_count = 3 if lesson == "split" else 2 if lesson == "independent" else int(rng.integers(1, 4))
            if split == "test" and shift == 1:
                gate_count = int(rng.integers(4, 6))
            if shift == 0:
                rotation = turns[index % 2] if split == "train" else turns[2+index % 2]
            elif split == "train":
                rotation = turns[index % 2] if mixed else turns[0]
            else:
                rotation = turns[int(rng.integers(2))] if mixed else turns[0]
            wide = split == "test" and shift == 2
            teaching_counts = None
            if lesson == "split":
                teaching_counts = list(map(int, rng.choice(np.arange(1, maximum+1), 2, replace=False)))
            elif lesson == "independent":
                missing = sorted(set(range(1, maximum+1))-training_features[0]["active_counts"])
                teaching_counts = missing+[int(rng.integers(1, maximum+1)) for _ in range(2-len(missing))]
                teaching_counts = [teaching_counts[int(j)] for j in rng.permutation(2)]
            for _ in range(400):
                pair = _attempt_pair(rng, maximum, lesson, gate_count, rotation, wide,
                                     repeat=repeat and lesson == "split", teaching_counts=teaching_counts)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                f = _features(_decode(pair["input"]))
                if split == "train" and index == 1:
                    if training_features[0]["active_sides"] | f["active_sides"] != {-1, 1}:
                        continue
                    if training_features[0]["active_counts"] | f["active_counts"] != set(range(1, maximum+1)):
                        continue
                if split == "test" and f["repeat"] and not any(x["repeat"] and x["alternation_effect"]
                                                               for x in training_features):
                    continue
                seen.add(_key(pair["input"]))
                puzzle[split].append(pair)
                if split == "train":
                    training_features.append(f)
                break
            else:
                raise RuntimeError("bounded cargo/baffle construction exhausted")
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "list splits")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4 and
             len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen, analyses, features = set(), {}, {}
    for split in ("train", "test"):
        analyses[split], features[split] = [], []
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "dimensions preserved")
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            parsed = _decode(a)
            expected = np.rot90(_render(parsed), -parsed["rotation"])
            _require(np.array_equal(expected, b), "incorrect transport/docking/normalization")
            f = _features(parsed)
            _require(f["same_connectivity"], "generated grouping must be unambiguous under diagonal contact")
            _require(f["extent"] and f["baffle_effect"], "both inherited mechanisms must change landing cells")
            # Check each rigid mask directly, including its width and holes.
            landings = _targets(parsed)
            for landing in landings:
                _require(_signature(landing["cells"]) == _signature(landing["component"]["cells"]),
                         "mask shape changed")
                _require(max(r for r, c in landing["cells"]) == landing["dock"]["bottom"], "far-edge alignment")
            _require(sum(len(x["cells"]) for x in landings) ==
                     len({p for x in landings for p in x["cells"]}), "landing cell collision")
            analyses[split].append(parsed)
            features[split].append(f)
    train, test = features["train"], features["test"]
    for name in ("contact", "independent", "conditional_effect", "selective_effect", "count_effect",
                 "sequential", "order_effect", "projection_effect", "edge_effect"):
        _require(any(f[name] for f in train), "missing training witness: "+name)
    counts = set().union(*(f["active_counts"] for f in train))
    _require(len(counts) >= 2 and set().union(*(f["active_sides"] for f in train)) == {-1, 1},
             "count and opposite-end demonstrations")
    _require(set().union(*(f["active_counts"] for f in test)) <= counts, "unseen test clearance")
    if any(f["repeat"] for f in test):
        _require(any(f["repeat"] and f["alternation_effect"] for f in train),
                 "repeated detours need an answer-relevant training witness")
    directions = {f["direction"] for f in train}
    if any(f["direction"] not in directions for f in test):
        _require(len(directions) >= 2, "rotation holdout needs two training directions")
    _require(all(f["direction"] not in directions for f in test) or
             min(f["gates"] for f in test) > max(f["gates"] for f in train) or
             min(f["max_width"] for f in test) > max(f["max_width"] for f in train),
             "no intended structural train/test shift")


def self_test():
    """Bounded 42-seed audit, count API checks, and literal semantic regression."""
    puzzles, records, normalized = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                a = _decode(pair["input"])
                f = _features(a)
                f["split"] = split
                records.append(f)
                roles = {a["background"]: 0}
                structure = tuple(roles.setdefault(int(c), len(roles)) for c in a["canonical"].flat)
                normalized.add((a["canonical"].shape, structure))
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "input-only solve")
    def puzzle_key(p):
        return tuple((_key(x["input"]), _key(x["output"])) for s in ("train", "test") for x in p[s])
    _require(len({puzzle_key(p) for p in puzzles}) == 42, "cross-seed duplication")
    _require(puzzle_key(puzzles[0]) == puzzle_key(generate_puzzle(0)), "seed reproducibility")
    _require(len({f["shape"] for f in records}) >= 30, "shape variation")
    _require(len(normalized) >= 160, "structural variation beyond palette")
    _require(len({s for f in records for s in f["masks"]}) >= 35, "mask variation")
    for split in ("train", "test"):
        fs = [f for f in records if f["split"] == split]
        _require({f["direction"] for f in fs} == set(range(4)), "all four directions")
        _require({f["routes"] for f in fs} >= {1, 2, 3, 4} if split == "train" else
                 {f["routes"] for f in fs} >= {1, 2, 3}, "dock-count variation")
        _require(set().union(*(f["active_counts"] for f in fs)) == {1, 2, 3, 4}, "unary-count coverage")
        for name in ("repeat", "miss", "wrong_color_overlap", "shared_rails", "independent", "order_effect"):
            _require(any(f[name] for f in fs), "missing observed feature: "+name)
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(123, train, test)
            _require(len(p["train"]) == train and len(p["test"]) == test, "explicit counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    _literal_regression()
    corrupted = {s: [{k: np.array(v).copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({f["shape"] for f in records}),
            "normalized_structures": len(normalized), "masks": len({s for f in records for s in f["masks"]}),
            "dock_counts": sorted({f["routes"] for f in records}),
            "baffle_counts": sorted({f["gates"] for f in records}),
            "feature_pairs": {name: sum(f[name] for f in records) for name in
                              ("repeat", "miss", "wrong_color_overlap", "independent", "order_effect", "shared_rails")}}


def _literal_regression():
    # A horizontal three-cell cargo and a two-row L share a route but not a
    # displacement; a touching second color obeys only its own marked baffle.
    a = np.zeros((19, 20), dtype=np.int8)
    a[0, 9:12] = 1
    a[0, 12] = a[1, 12] = 2
    a[3, 1] = a[4, 1:3] = 1
    a[6, 8:14] = 1
    a[6, [13, 11]] = 3  # left bypass: right edge becomes 8-3=5
    a[8, 9:14] = 2
    a[8, 9] = 4         # right bypass: left edge becomes 13+2=15
    a[11:14, 1:19] = 1
    a[11:14, [0, 19]] = 3
    a[15:18, 1:19] = 2
    a[15:18, [0, 19]] = 4
    b = a.copy()
    b[:5] = 0
    b[6, 8:14] = 1
    b[8, 9:14] = 2
    b[13, 3:6] = 3
    b[12, 1] = b[13, 1:3] = 3
    b[16:18, 15] = 4
    for turn in range(4):
        for palette in (np.arange(10), np.array([8, 6, 0, 9, 5, 1, 2, 3, 4, 7])):
            actual = solve(palette[np.rot90(a, turn)])
            _require(np.array_equal(actual, palette[np.rot90(b, turn)]), "literal width/conditional/color regression")
    collision = a.copy()
    collision[3:5, 1:3] = 0
    collision[3, 4] = collision[4, 4:6] = 1
    try:
        solve(collision)
    except AssertionError:
        pass
    else:
        raise AssertionError("landing collision accepted")
    # Explicit two-command program: the L goes right to columns 11..13,
    # then left to columns 5..7. Reversing encounter order changes the result.
    a = np.zeros((19, 22), dtype=np.int8)
    a[0, 7] = a[1, 7:10] = 1
    a[6, 6:10] = 1
    a[6, 6] = 2
    a[9, 10:16] = 1
    a[9, [15, 13]] = 2
    a[14:17, 1:21] = 1
    a[14:17, [0, 21]] = 2
    b = a.copy()
    b[:2] = 0
    b[6, 6:10] = b[9, 10:16] = 1
    b[15, 5] = b[16, 5:8] = 2
    _require(np.array_equal(solve(a), b), "literal ordered-command regression")
    analysis = _decode(a)
    _require(_landing_key(analysis) != _landing_key(analysis, "reverse"), "order witness disappeared")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
