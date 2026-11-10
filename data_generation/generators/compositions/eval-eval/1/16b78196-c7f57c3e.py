"""Counterfactual tab assembly: crossing the band exchanges a key's state.

Only the standard library and NumPy are used. The input reader and global search
receive no construction state. Search is exhaustive for the specified full-width,
column-convex layer family, with distinct colored outputs as the uniqueness unit.
"""
from __future__ import annotations

import json
import math
from functools import reduce

import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((g >= 0) & (g <= 9))), "invalid colors")
    return g.astype(int, copy=True)


def _components(mask, diagonal=True):
    todo = set(map(tuple, np.argwhere(mask)))
    groups = []
    steps = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
             if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while todo:
        first = min(todo)
        todo.remove(first)
        stack, group = [first], []
        while stack:
            r, c = stack.pop()
            group.append((r, c))
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        groups.append(np.array(sorted(group), dtype=int))
    return groups


def _dilate(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    return np.logical_or.reduce([padded[r:r+h, c:c+w]
                                 for r in range(3) for c in range(3)])


def _sig(mapping):
    return tuple(sorted((int(r), int(c), int(v)) for (r, c), v in mapping.items()))


def _primitive(crop):
    runs = []
    for array in (crop, crop.T):
        for row in array:
            breaks = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            runs.extend(map(int, np.diff(breaks)))
    scale = reduce(math.gcd, runs)
    small = crop[::scale, ::scale]
    _require(np.array_equal(np.repeat(np.repeat(small, scale, 0), scale, 1), crop),
             "inconsistent pixel blocks")
    return scale, small


def _expand(template, scale):
    return {(r*scale+dr, c*scale+dc): color
            for (r, c), color in template.items()
            for dr in range(scale) for dc in range(scale)}


def _registered_states(objects, background):
    active = [o for o in objects if o["active"]]
    _require(len(active) >= 2, "two visible key states are required")
    signatures = sorted({o["raw"] for o in active})
    _require(len(signatures) == 2, "expected exactly two key states")
    minimum = min(o["scale"] for o in active)
    _require({o["raw"] for o in active if o["scale"] == minimum} == set(signatures),
             "both states must be visible at a common reference scale")
    maps = [{(r, c): color for r, c, color in s} for s in signatures]
    shifts = set()
    for color in set(maps[0].values()) & set(maps[1].values()):
        a = {p for p, v in maps[0].items() if v == color}
        b = {p for p, v in maps[1].items() if v == color}
        amin = np.min(list(a), axis=0)
        bmin = np.min(list(b), axis=0)
        dr, dc = map(int, amin-bmin)
        if a == {(r+dr, c+dc) for r, c in b}:
            shifts.add((dr, dc))
    candidates = []
    for dr, dc in sorted(shifts):
        aligned = {(r+dr, c+dc): v for (r, c), v in maps[1].items()}
        common = {p: v for p, v in maps[0].items() if aligned.get(p) == v}
        if len(common) >= 2:
            candidates.append((len(common), aligned, common))
    _require(bool(candidates), "missing invariant scaffold")
    best = max(x[0] for x in candidates)
    candidates = [x for x in candidates if x[0] == best]
    _require(len(candidates) == 1, "ambiguous scaffold registration")
    _, aligned, common = candidates[0]
    origin = np.min(list(common), axis=0)
    templates = [{(int(r-origin[0]), int(c-origin[1])): v
                  for (r, c), v in m.items()} for m in (maps[0], aligned)]
    common = {(int(r-origin[0]), int(c-origin[1])): v for (r, c), v in common.items()}
    _require(set(templates[0]) != set(templates[1]), "states must have different silhouettes")
    _require(any(templates[0][p] != templates[1][p]
                 for p in set(templates[0]) & set(templates[1])), "states must recolor an accent")
    for obj in active:
        obj["state"] = signatures.index(obj["raw"])
        t = templates[obj["state"]]
        minimum_coord = np.min(list(t), axis=0)
        obj["anchor"] = obj["box"] - minimum_coord*obj["scale"]
        expanded = _expand(t, obj["scale"])
        actual = {(int(r), int(c)): int(v)
                  for (r, c), v in zip(obj["cells"], obj["colors"])}
        predicted = {(r+int(obj["anchor"][0]), c+int(obj["anchor"][1])): v
                     for (r, c), v in expanded.items()}
        _require(actual == predicted, "template normalization failed")
    return templates, common


def _read_input(grid):
    g = _grid(grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(int(counts[bg])*2 > g.size, "background must be a strict majority")
    candidates = []
    for color in np.unique(g):
        if color == bg:
            continue
        for cells in _components(g == color, diagonal=False):
            for transpose in (False, True):
                cross = cells[:, 0 if transpose else 1]
                if len(np.unique(cross)) == g.shape[0 if transpose else 1]:
                    candidates.append((int(color), transpose, cells))
    _require(len(candidates) == 1, "expected one connected spanning band")
    band, transpose, band_cells = candidates[0]
    _require(len(band_cells) == np.count_nonzero(g == band), "band color used elsewhere")
    g = g.T.copy() if transpose else g
    mask = g == band
    edges = []
    for c in range(g.shape[1]):
        rows = np.flatnonzero(mask[:, c])
        _require(np.array_equal(rows, np.arange(rows[0], rows[-1]+1)), "nonconvex band")
        edges.append((int(rows[0]), int(rows[-1]+1)))
    edges = np.array(edges, dtype=int)
    _require(int(edges[:, 0].max()) < int(edges[:, 1].min()), "band needs a solid common core")
    objects = []
    occupied = mask.copy()
    for cells in _components((g != bg) & ~mask):
        _require(not np.any(_dilate(occupied)[cells[:, 0], cells[:, 1]]), "input objects must be separated")
        occupied[cells[:, 0], cells[:, 1]] = True
        colors = g[cells[:, 0], cells[:, 1]].copy()
        above = bool(np.all(cells[:, 0] < edges[cells[:, 1], 0]))
        below = bool(np.all(cells[:, 0] >= edges[cells[:, 1], 1]))
        _require(above != below, "each piece must lie wholly on one side")
        active = len(np.unique(colors)) > 1
        obj = {"cells": cells, "colors": colors, "side": -1 if above else 1,
               "active": active, "scale": 1}
        if active:
            lo, hi = cells.min(axis=0), cells.max(axis=0)+1
            crop = g[lo[0]:hi[0], lo[1]:hi[1]]
            _require(np.count_nonzero(crop != bg) == len(cells), "interleaved objects")
            scale, small = _primitive(crop)
            obj.update(scale=scale, box=lo,
                       raw=_sig({(int(r), int(c)): int(small[r, c])
                                 for r, c in np.argwhere(small != bg)}))
        objects.append(obj)
    _require(any(not o["active"] for o in objects), "ordinary fillers are required")
    templates, common = _registered_states(objects, bg)
    return {"grid": g, "background": bg, "band": band, "transpose": transpose,
            "edges": edges, "objects": objects, "templates": templates, "common": common}


def _description(mapping, side):
    coords = np.array(sorted(mapping), dtype=int)
    colors = np.array([mapping[tuple(p)] for p in coords], dtype=int)
    x = coords[:, 1] - coords[:, 1].min()
    y = side*coords[:, 0]
    width = int(x.max()) + 1
    lo, hi = [], []
    for c in range(width):
        values = np.sort(y[x == c])
        _require(len(values) > 0 and np.array_equal(values, np.arange(values[0], values[-1]+1)),
                 "pieces must be full-width, column-convex layers")
        lo.append(int(values[0]))
        hi.append(int(values[-1])+1)
    origin = min(lo)
    return {"lo": tuple(v-origin for v in lo), "hi": tuple(v-origin for v in hi),
            "local": np.column_stack((y-origin, x)), "colors": colors,
            "origin": origin, "cross_origin": int(coords[:, 1].min())}


def _assemblies(input_grid, toggle_mode="crossing"):
    info = _read_input(input_grid)
    g, bg, band = info["grid"], info["background"], info["band"]
    objects, edges = info["objects"], info["edges"]
    h, w = g.shape
    base = np.full_like(g, bg)
    base[g == band] = band
    candidates, root_matches = [], set()
    for side in (-1, 1):
        descriptions, target_states = [], []
        for obj in objects:
            if obj["active"]:
                flip = side != obj["side"] if toggle_mode == "crossing" else toggle_mode == "always"
                state = obj["state"] ^ int(flip)
                mapping = _expand(info["templates"][state], obj["scale"])
                target_states.append(state)
            else:
                mapping = {tuple(p): int(v) for p, v in zip(obj["cells"], obj["colors"])}
                target_states.append(None)
            descriptions.append(_description(mapping, side))
        lookup = {}
        for i, desc in enumerate(descriptions):
            lookup.setdefault(desc["lo"], []).append(i)
        boundary = 1-edges[:, 0] if side == -1 else edges[:, 1]
        limit = 1 if side == -1 else h
        for width in sorted({len(d["lo"]) for d in descriptions}):
            for start in range(w-width+1):
                initial = tuple(map(int, boundary[start:start+width]))

                def extend(profile, used, placements):
                    offset = min(profile)
                    if used and len(set(profile)) == 1:
                        footprint = frozenset((side*u, start+c)
                                              for u in range(min(initial), profile[0])
                                              for c in range(width))
                        candidates.append({"bits": used, "placements": placements,
                                           "side": side, "start": start, "width": width,
                                           "initial": initial, "end": profile[0],
                                           "footprint": footprint})
                    key = tuple(v-offset for v in profile)
                    for i in lookup.get(key, ()):
                        if used & (1 << i):
                            continue
                        d = descriptions[i]
                        next_profile = tuple(offset+v for v in d["hi"])
                        if max(next_profile) > limit:
                            continue
                        cells = np.column_stack((side*(d["local"][:, 0]+offset),
                                                 d["local"][:, 1]+start))
                        if np.any(cells[:, 0] < 0) or np.any(cells[:, 0] >= h):
                            continue
                        if np.any(base[cells[:, 0], cells[:, 1]] != bg):
                            continue
                        obj = objects[i]
                        anchor = None
                        if obj["active"]:
                            anchor = np.array((side*(offset-d["origin"]), start-d["cross_origin"]))
                            if np.array_equal(anchor, obj["anchor"]):
                                continue
                        elif set(map(tuple, cells)) == set(map(tuple, obj["cells"])):
                            continue
                        if not used:
                            root_matches.add((i, side, start))
                        placement = {"id": i, "cells": cells, "colors": d["colors"],
                                     "state": target_states[i], "anchor": anchor}
                        extend(next_profile, used | (1 << i), placements+[placement])

                extend(initial, 0, [])
    by_piece = [[] for _ in objects]
    for candidate in candidates:
        for i in range(len(objects)):
            if candidate["bits"] & (1 << i):
                by_piece[i].append(candidate)
    full = (1 << len(objects))-1
    solutions = {}

    def cover(used, footprint, chosen):
        if len(solutions) >= 2:
            return
        if used == full:
            answer = base.copy()
            for tab in chosen:
                for p in tab["placements"]:
                    answer[p["cells"][:, 0], p["cells"][:, 1]] = p["colors"]
            solutions.setdefault(answer.tobytes(), (answer, chosen))
            return
        options_by_id = [(i, [t for t in by_piece[i]
                              if not t["bits"] & used and not t["footprint"] & footprint])
                         for i in range(len(objects)) if not used & (1 << i)]
        _, options = min(options_by_id, key=lambda x: len(x[1]))
        for tab in options:
            cover(used | tab["bits"], footprint | tab["footprint"], chosen+[tab])

    cover(0, frozenset(), [])
    info["root_matches"] = len(root_matches)
    info["candidates"] = len(candidates)
    return list(solutions.values()), info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique assembly exclusively from the supplied input pixels."""
    solutions, info = _assemblies(input_grid)
    _require(len(solutions) == 1, "input does not determine a unique colored assembly")
    answer = solutions[0][0]
    return (answer.T if info["transpose"] else answer).copy()


def _profile(rng, width, amplitude):
    # Include steps, irregular teeth and slopes; guarantee a nonflat contour.
    mode = int(rng.integers(3))
    if mode == 0:
        p = rng.integers(0, amplitude+1, width)
    elif mode == 1:
        p = np.floor(np.linspace(0, amplitude, width)).astype(int)
        if rng.integers(2):
            p = p[::-1].copy()
    else:
        p = np.zeros(width, dtype=int)
        p[int(rng.integers(1, width)):] = amplitude
        if rng.integers(2):
            p = p[::-1].copy()
    p -= p.min()
    if p.max() == 0:
        p[int(rng.integers(width))] = amplitude
    return p


def _design(rng, width, compact=False, palette=None, shifted=False):
    palette = list(map(int, rng.permutation(10))) if palette is None else list(palette)
    bg, band, carrier = palette[:3]
    core = 1 if compact else int(rng.integers(1, 3))
    rich_colors = bool(rng.random() < .35)
    stable = core == 2 and rng.random() < .5
    cursor = 3
    stable_color = palette[cursor] if stable else None
    cursor += int(stable)
    accents = []
    for _ in range(2):
        amount = 2 if rich_colors else 1
        accents.append(palette[cursor:cursor+amount])
        cursor += amount
    fillers = palette[cursor:]
    key_column = int(rng.integers(1, width-1))
    templates, contours = [], []
    extra_state = int(rng.integers(2)) if shifted else None
    extra_edge = int(rng.integers(2)) if shifted else None
    for state in range(2):
        top_amp = (2 if state == extra_state and extra_edge == 0 else 1) if compact else int(rng.integers(1, 3))
        bottom_amp = (2 if state == extra_state and extra_edge == 1 else 1) if compact else int(rng.integers(1, 3))
        top = _profile(rng, width, top_amp)
        bottom = _profile(rng, width, bottom_amp)
        _require(not np.array_equal(top.max()-top, bottom),
                 "resample a self-mating key that would obscure layer order")
        mapping = {(r, c): carrier for r in range(core) for c in range(width)}
        mapping[0, key_column] = accents[state][0]
        if stable:
            mapping[1, key_column] = stable_color
        for c in range(width):
            for r in range(-int(top[c]), 0):
                mapping[r, c] = accents[state][0]
            for r in range(core, core+int(bottom[c])):
                mapping[r, c] = accents[state][-1]
        templates.append(mapping)
        contours.append((top, bottom))
    _require(set(templates[0]) != set(templates[1]), "resample identical state supports")
    _require(not np.array_equal(contours[0][0], contours[1][0])
             and not np.array_equal(contours[0][1], contours[1][1]),
             "resample keys with indistinguishable docking contours")
    if shifted:
        _require(contours[0][0].max() != contours[1][0].max()
                 or contours[0][1].max() != contours[1][1].max(),
                 "teaching states must shift their occupied bounding box")
    return {"bg": bg, "band": band, "templates": templates, "fillers": fillers,
            "width": width, "palette": palette}


def _layer(lo, hi, side, color):
    return {(side*u, c): int(color) for c in range(len(lo))
            for u in range(int(lo[c]), int(hi[c]))}


def _scatter(rng, base, pieces, sides, bg, edges):
    h, w = base.shape
    normalized = []
    for mapping in pieces:
        cells = np.array(sorted(mapping), dtype=int)
        colors = np.array([mapping[tuple(p)] for p in cells], dtype=int)
        cells -= cells.min(axis=0)
        normalized.append((cells, colors))
    for _ in range(16):
        result = base.copy()
        forbidden = _dilate(base != bg)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i])+float(rng.random())*5)
        successful = True
        for i in order:
            cells, colors = normalized[i]
            ph, pw = cells.max(axis=0)+1
            valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in cells:
                valid &= ~forbidden[r:r+valid.shape[0], c:c+valid.shape[1]]
            rows, cols = np.indices(valid.shape)
            for r, c in cells:
                valid &= ((rows+r < edges[cols+c, 0]) if sides[i] == -1
                          else (rows+r >= edges[cols+c, 1]))
            choices = np.argwhere(valid)
            if not len(choices):
                successful = False
                break
            origin = choices[int(rng.integers(len(choices)))]
            target = cells+origin
            result[target[:, 0], target[:, 1]] = colors
            forbidden = _dilate(result != bg)
        if successful:
            return result
    raise ValueError("scatter capacity exhausted")


def _make_pair(rng, design, states, scales, crossing, groups, rotation,
               reflection, preferred_side=None):
    bg, band = design["bg"], design["band"]
    sides = [int(rng.choice([-1, 1])) for _ in groups]
    if len(groups) > 1 and rng.random() < .65:
        sides[1] = -sides[0]
    if preferred_side is not None:
        sides[0] = preferred_side
        if len(groups) > 1:
            sides[1] = -sides[0]
    widths = [design["width"]*scales[group[0]] for group in groups]
    _require(all(len({scales[i] for i in group}) == 1 for group in groups), "mixed-width group")
    min_cross = max(14, max(sum(w for w, s in zip(widths, sides) if s == side)
                            + 2*sum(s == side for s in sides) for side in (-1, 1)))
    _require(min_cross <= 30, "tab width capacity")
    cross = int(rng.integers(min_cross, 31))
    reserved = {-1: set(), 1: set()}
    starts = []
    for width, side in zip(widths, sides):
        choices = [x for x in range(cross-width+1)
                   if not set(range(x-1, x+width+1)) & reserved[side]]
        _require(bool(choices), "tab spacing capacity")
        start = choices[int(rng.integers(len(choices)))]
        starts.append(start)
        reserved[side].update(range(start, start+width))
    # These are fixed band protrusions, including details that are never filled.
    edges_out = {-1: rng.integers(0, 3, cross), 1: rng.integers(0, 3, cross)}
    tab_data, depth = [], {-1: 0, 1: 0}
    filler_palette = design["fillers"][:int(rng.integers(1, min(3, len(design["fillers"]))+1))]
    filler_index = 0
    for side, start, width, group in zip(sides, starts, widths, groups):
        layers = []
        current = None
        initial = None
        for active_id in group:
            output_state = states[active_id] ^ int(crossing[active_id])
            expanded = _expand(design["templates"][output_state], scales[active_id])
            d = _description(expanded, side)
            lo, hi = np.array(d["lo"]), np.array(d["hi"])
            if current is None:
                offset = int(rng.integers(0, 2))
                initial = lo+offset
                edges_out[side][start:start+width] = initial
            else:
                offset = max(int(np.max(current+1-lo)),
                             int(np.max(np.maximum(current[:-1], current[1:])+1-
                                        np.minimum(lo[:-1], lo[1:]))))
                offset += int(rng.integers(0, 2))
                color = filler_palette[filler_index % len(filler_palette)]
                filler_index += 1
                layers.append((None, _layer(current, lo+offset, side, color), None))
            target_anchor = (side*(offset-d["origin"]), -d["cross_origin"])
            mapping = {(r+target_anchor[0], c+target_anchor[1]): v
                       for (r, c), v in expanded.items()}
            input_map = _expand(design["templates"][states[active_id]], scales[active_id])
            layers.append((active_id, mapping, input_map))
            current = hi+offset
        terminal = int(current.max())+int(rng.integers(1, 3))
        if rng.random() < .35:
            shape = _profile(rng, width, int(rng.integers(1, 3)))
            offset = max(int(np.max(current+1-shape)),
                         int(np.max(np.maximum(current[:-1], current[1:])+1-
                                    np.minimum(shape[:-1], shape[1:]))))
            middle = shape+offset
            color = filler_palette[filler_index % len(filler_palette)]
            filler_index += 1
            layers.append((None, _layer(current, middle, side, color), None))
            current = middle
            terminal = int(current.max())+int(rng.integers(1, 3))
        color = filler_palette[filler_index % len(filler_palette)]
        filler_index += 1
        layers.append((None, _layer(current, np.full(width, terminal), side, color), None))
        depth[side] = max(depth[side], terminal)
        tab_data.append((side, start, layers))
    core = int(rng.integers(2, 5))
    depth = {s: max(depth[s], int(edges_out[s].max())) for s in (-1, 1)}
    # Both source sides need room; scatter capacity depends on actual silhouettes.
    min_height = max(22, core+depth[-1]+depth[1]+2)
    _require(min_height <= 30, "tab depth capacity")
    height = int(rng.integers(min_height, 31))
    low = max(depth[-1], 8)
    high = min(height-core-depth[1], height-core-8)
    _require(low <= high, "source-side clearance")
    center = int(rng.integers(low, high+1))
    base = np.full((height, cross), bg, dtype=int)
    for c in range(cross):
        base[center-edges_out[-1][c]:center+core+edges_out[1][c], c] = band
    output = base.copy()
    input_pieces, input_sides = [], []
    for side, start, layers in tab_data:
        origin = center-1 if side == -1 else center+core
        for active_id, mapping, input_map in layers:
            destination = {(r+origin, c+start): v for (r, c), v in mapping.items()}
            for p, color in destination.items():
                _require(output[p] == bg, "construction overlap")
                output[p] = color
            input_pieces.append(input_map if active_id is not None else mapping)
            input_sides.append(side*(-1 if crossing[active_id] else 1) if active_id is not None
                               else int(rng.choice([-1, 1])))
    band_edges = np.array([(int(np.flatnonzero(base[:, c] == band)[0]),
                            int(np.flatnonzero(base[:, c] == band)[-1])+1)
                           for c in range(cross)])
    inp = _scatter(rng, base, input_pieces, input_sides, bg, band_edges)
    _require(np.count_nonzero(inp == bg)*2 > inp.size, "majority background capacity")
    solutions, info = _assemblies(inp)
    _require(len(solutions) == 1, "ambiguous or impossible assembly")
    _require(np.array_equal(solutions[0][0], output), "construction/search disagreement")
    # Rotate/reflect the entire pair, never individual pieces during solving.
    inp, output = np.rot90(inp, rotation), np.rot90(output, rotation)
    if reflection:
        inp, output = inp[:, ::-1], output[:, ::-1]
    return {"input": inp.copy(), "output": output.copy()}


def _checked_counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _checked_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    stable_palette = bool(rng.random() < .25)
    teaching_width = int(rng.integers(3, 6))
    for _ in range(100):
        try:
            teaching = _design(rng, teaching_width, compact=True, palette=palette, shifted=True)
            break
        except ValueError:
            continue
    else:
        raise RuntimeError("could not construct two different key states")
    first_rotation = int(rng.integers(4))
    first_reflection = bool(rng.integers(2))
    teaching_side = int(rng.choice([-1, 1]))
    phase = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for attempt in range(240):
                try:
                    if split == "train" and index < 2:
                        design = teaching
                        scales = [1, 1] if index == 0 else [1, 1, 2]
                        rotation = first_rotation
                    elif split == "test":
                        scale = 3 if rng.random() < .6 else 2
                        widths = [w for w in range(3, 6) if w != teaching_width and (scale != 3 or w <= 4)]
                        width = int(rng.choice(widths))
                        design = _design(rng, width, compact=scale == 3,
                                         palette=palette if stable_palette else None)
                        scales = [1, 1, scale]
                        rotation = (first_rotation+1) % 4 if index == 0 else int(rng.integers(4))
                    else:
                        width = int(rng.integers(3, 7))
                        design = _design(rng, width, compact=False,
                                         palette=palette if stable_palette else None)
                        scales = [1, 1] + ([int(rng.choice([1, 2]))] if rng.random() < .7 else [])
                        if rng.random() < .2:
                            scales.append(1)
                        rotation = int(rng.integers(4))
                    n = len(scales)
                    states = [0, 1]+list(map(int, rng.integers(0, 2, n-2)))
                    crossing = [False, False]+[bool(scales[i] > 1 and rng.random() < .65)
                                               for i in range(2, n)]
                    crossing[phase ^ (index % 2)] = True
                    if split == "train" and index == 1:
                        crossing[2] = True
                    if split == "train" and index < 2:
                        groups = [[0, 1]] if n == 2 else [[0, 1], [2]]
                    elif n == 2:
                        groups = [[0, 1]] if rng.random() < .4 else [[0], [1]]
                    elif n == 3:
                        groups = [[0, 1], [2]] if split == "test" or rng.random() < .6 else [[0], [1], [2]]
                    else:
                        groups = [[0, 1], [2], [3]]
                    teaching_pair = split == "train" and index < 2
                    reflection = first_reflection if teaching_pair else bool(rng.integers(2))
                    pair = _make_pair(rng, design, states, scales, crossing, groups, rotation,
                                      reflection, teaching_side if teaching_pair else None)
                    _analyze_pair(pair)
                    key = _key(pair["input"])
                    _require(key not in seen, "duplicate input")
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError(f"bounded generation exhausted at {split} {index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape and not np.array_equal(inp, out), "invalid output shape or identity")
    solutions, info = _assemblies(inp)
    _require(len(solutions) == 1, "nonunique answer")
    expected, tabs = solutions[0]
    actual = out.T if info["transpose"] else out
    _require(np.array_equal(expected, actual), "incorrect conditional assembly")
    _require(np.array_equal(info["grid"] == info["band"], actual == info["band"]), "band altered")
    used, crossed, retained, scales, directions = set(), 0, 0, set(), set()
    unit_directions, unit_sides = set(), set()
    changed_support = 0
    for tab in tabs:
        # Fixed band plus placed payload must fill the whole rectangle exactly.
        _require(all(actual[p] != info["background"] for p in tab["footprint"]), "tab contains a hole")
        for placement in tab["placements"]:
            i = placement["id"]
            _require(i not in used, "piece reused")
            used.add(i)
            obj = info["objects"][i]
            cells = placement["cells"]
            _require(np.all(actual[cells[:, 0], cells[:, 1]] == placement["colors"]), "piece color mismatch")
            if obj["active"]:
                flip = tab["side"] != obj["side"]
                _require(placement["state"] == (obj["state"] ^ int(flip)), "crossing parity incorrect")
                source_scaffold = _expand(info["common"], obj["scale"])
                delta = placement["anchor"]-obj["anchor"]
                _require(bool(np.any(delta)), "key scaffold did not move")
                for (r, c), color in source_scaffold.items():
                    dest = (r+int(placement["anchor"][0]), c+int(placement["anchor"][1]))
                    _require(int(actual[dest]) == color, "scaffold not preserved by translation")
                crossed += int(flip)
                retained += int(not flip)
                scales.add(obj["scale"])
                directions.add(obj["state"] if flip else -1)
                if obj["scale"] == 1:
                    unit_sides.add(tab["side"])
                    if flip:
                        unit_directions.add(obj["state"])
                changed_support += int(flip and set(info["templates"][0]) != set(info["templates"][1]))
            else:
                delta = cells.min(axis=0)-obj["cells"].min(axis=0)
                _require(set(map(tuple, obj["cells"]+delta)) == set(map(tuple, cells)), "filler changed shape")
                _require(np.array_equal(np.sort(obj["colors"]), np.sort(placement["colors"])), "filler recolored")
    _require(used == set(range(len(info["objects"]))), "not every piece used")
    _require(crossed > 0 and retained > 0 and changed_support > 0, "missing causal witnesses")
    # Admission filters for clear demonstrations. Conservation of the ordinary
    # input palette histogram would make 'never exchange' hard to distinguish;
    # an all-exchange inventory could similarly disguise occurrence identities.
    out_hist = np.bincount(actual.ravel(), minlength=10)
    _require(not np.array_equal(np.bincount(inp.ravel(), minlength=10), out_hist),
             "never-exchange inventory shortcut is not distinguished")
    always_hist = np.zeros(10, dtype=int)
    always_hist[info["band"]] = np.count_nonzero(info["grid"] == info["band"])
    for obj in info["objects"]:
        values = (list(_expand(info["templates"][1-obj["state"]], obj["scale"]).values())
                  if obj["active"] else obj["colors"])
        always_hist += np.bincount(values, minlength=10)
    always_hist[info["background"]] += actual.size-int(always_hist.sum())
    _require(not np.array_equal(always_hist, out_hist), "always-exchange inventory shortcut is not distinguished")
    info.update(tabs=tabs, crossed=crossed, retained=retained, scales=tuple(sorted(scales)),
                directions=directions, unit_directions=unit_directions,
                unit_sides=unit_sides, output=actual)
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _checked_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_pair(pair)
            key = _key(np.asarray(pair["input"]))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    _require(any(max(i["scales"]) >= 2 for i in infos["train"]), "training lacks scaled docking")
    a, b = infos["train"][:2]
    _require(tuple(_sig(t) for t in a["templates"]) == tuple(_sig(t) for t in b["templates"]),
             "teaching examples must share a local vocabulary")
    _require(a["unit_directions"] | b["unit_directions"] == {0, 1}, "training lacks both exchange directions")
    _require(len(a["unit_sides"]) == 1 and a["unit_sides"] == b["unit_sides"],
             "teaching contrast does not isolate source-side dependence")
    scaffold_colors = list(set(a["common"].values()))
    _require(not np.array_equal(np.isin(a["grid"], scaffold_colors), np.isin(a["output"], scaffold_colors)),
             "training does not distinguish exchange in place from assembly")
    return None


def _fingerprint(puzzle):
    return tuple(_key(np.asarray(p[k])) for s in ("train", "test")
                 for p in puzzle[s] for k in ("input", "output"))


def _structural_vocabulary(info):
    """Remove color names, common translation, dihedral pose and state order."""
    variants = []
    for rotation in range(4):
        for reflection in (False, True):
            maps = []
            for template in info["templates"]:
                mapping = {}
                for (r, c), color in template.items():
                    for _ in range(rotation):
                        r, c = c, -r
                    mapping[r, -c if reflection else c] = color
                maps.append(mapping)
            origin = np.min(list(set(maps[0]) | set(maps[1])), axis=0)
            for order in ((0, 1), (1, 0)):
                colors, encoded = {}, []
                for state in order:
                    encoded.append(tuple((int(r-origin[0]), int(c-origin[1]),
                                          colors.setdefault(color, len(colors)))
                                         for r, c, color in _sig(maps[state])))
                variants.append(tuple(encoded))
    return min(variants)


def _negative_checks(puzzle):
    info = _analyze_pair(puzzle["train"][0])
    g, bg, band = info["grid"], info["background"], info["band"]
    # Replace both references by the shallower state at their scaffold anchors.
    def box_area(template):
        points = np.array(list(template))
        return int(np.prod(points.max(axis=0)-points.min(axis=0)+1))
    shallow = min(info["templates"], key=box_area)
    missing = g.copy()
    for obj in info["objects"]:
        if obj["active"]:
            missing[obj["cells"][:, 0], obj["cells"][:, 1]] = bg
    for obj in info["objects"]:
        if obj["active"]:
            for (r, c), color in _expand(shallow, obj["scale"]).items():
                point = r+int(obj["anchor"][0]), c+int(obj["anchor"][1])
                _require(0 <= point[0] < g.shape[0] and 0 <= point[1] < g.shape[1], "missing-state fixture clipped")
                missing[point] = color
    try:
        solve(missing)
    except ValueError:
        pass
    else:
        raise AssertionError("absent second state accepted")
    # Duplicate the actual root at another boundary alignment. The same chain
    # then has two destinations; solve must reject rather than choose the first.
    tab = info["tabs"][0]
    width, start = tab["width"], tab["start"]
    for other in range(g.shape[1]-width+1):
        if not (other+width < start or start+width < other):
            continue
        clone = g.copy()
        valid = True
        for c in range(width):
            new_band = g[:, start+c] == band
            column = clone[:, other+c]
            if np.any(new_band & (column != bg) & (column != band)):
                valid = False
                break
            column[column == band] = bg
            column[new_band] = band
        if not valid:
            continue
        try:
            alternatives, _ = _assemblies(clone)
        except ValueError:
            continue
        if len(alternatives) == 2:
            try:
                solve(clone)
            except ValueError:
                return
            raise AssertionError("two different socket answers accepted")
    raise AssertionError("could not build bounded duplicate-root fixture")


def self_test():
    """Bounded checks on 42 seeds, explicit counts, ablations and mutations."""
    fingerprints, shapes, vocabularies, tab_counts = set(), set(), set(), set()
    structural_vocabularies = set()
    object_counts, scale_schedules, colors, widths, depths = set(), set(), set(), set(), set()
    orientations, repeated_fillers, edge_contacts, multi_keys = set(), 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed < 3:
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "nondeterminism")
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                _require(np.array_equal(solve(np.asarray(pair["input"]).copy()), pair["output"]), "solve mismatch")
                shapes.add(pair["input"].shape)
                orientations.add(info["transpose"])
                tab_counts.add(len(info["tabs"]))
                object_counts.add(len(info["objects"]))
                scale_schedules.add(tuple(sorted(o["scale"] for o in info["objects"] if o["active"])))
                vocabularies.add(tuple(_sig(t) for t in info["templates"]))
                structural_vocabularies.add(_structural_vocabulary(info))
                colors.add((info["background"], info["band"]))
                widths.update(t["width"] for t in info["tabs"])
                depths.update(t["end"]-min(t["initial"]) for t in info["tabs"])
                fillers = [int(o["colors"][0]) for o in info["objects"] if not o["active"]]
                repeated_fillers += len(set(fillers)) < len(fillers)
                payloads = [(g != info["background"]) & (g != info["band"])
                            for g in (info["grid"], info["output"])]
                edge_contacts += any(np.any(m[0]) or np.any(m[-1]) or np.any(m[:, 0]) or np.any(m[:, -1])
                                     for m in payloads)
                multi_keys += any(sum(info["objects"][p["id"]]["active"] for p in t["placements"]) > 1
                                  for t in info["tabs"])
                # At the correct scaffold destinations, neither 'never exchange'
                # nor 'always exchange' can reproduce the required key cells.
                for mode in ("never", "always"):
                    alternate, _ = _assemblies(pair["input"], toggle_mode=mode)
                    _require(all(not np.array_equal(a, info["output"]) for a, _ in alternate),
                             "a shortcut reproduces the answer")
        if seed == 0:
            _negative_checks(puzzle)
            damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
            damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
            try:
                validate_generated_puzzle(damaged)
            except ValueError:
                pass
            else:
                raise AssertionError("corrupted label accepted")
    _require(len(fingerprints) == 42 and len(shapes) >= 25, "insufficient puzzle/shape variation")
    _require(len(vocabularies) >= 70 and len(colors) >= 25, "insufficient local templates/palettes")
    _require(len(structural_vocabularies) >= 60, "insufficient color/pose-normalized key geometry")
    _require(tab_counts >= {1, 2, 3} and len(object_counts) >= 4, "insufficient grouping variation")
    _require(len(scale_schedules) >= 5 and any(3 in s for s in scale_schedules), "insufficient scale variation")
    _require(len(widths) >= 8 and len(depths) >= 8, "insufficient interlock geometry")
    _require(orientations == {False, True} and repeated_fillers >= 20 and edge_contacts >= 20 and multi_keys >= 30,
             "insufficient structural variation")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(9000+nt*10+ns, nt, ns)
            _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == ns, "wrong explicit counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "puzzles": len(fingerprints), "shapes": len(shapes),
            "local_vocabularies": len(vocabularies), "role_assignments": len(colors),
            "structural_key_vocabularies": len(structural_vocabularies),
            "tab_counts": sorted(tab_counts), "piece_counts": sorted(object_counts),
            "scale_schedules": sorted(scale_schedules), "tab_widths": sorted(widths),
            "tab_depths": sorted(depths), "reused_filler_pairs": repeated_fillers,
            "edge_contact_pairs": edge_contacts, "multi_key_tabs_pairs": multi_keys,
            "semantic_check": "input-derived authored search, not independent certification"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
