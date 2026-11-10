"""Complementary reflection consensus with scaffold-registered state exchange.

No parent modules, examples, seeds, or labels are used by the input reader.
The construction renderer and input-derived reconstruction are checked together.
"""

from math import gcd
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= grid) & (grid <= 9)), "colors must be 0..9")
    return grid.astype(int, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _bounds(points):
    return (min(r for r, c in points), max(r for r, c in points),
            min(c for r, c in points), max(c for r, c in points))


def _signature(mapping):
    return tuple(sorted((r, c, int(color)) for (r, c), color in mapping.items()))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = r + dr, c + dc
                    if p in remaining:
                        remaining.remove(p)
                        group.add(p)
                        stack.append(p)
        groups.append(group)
    return groups


def _connected(points):
    r0, r1, c0, c1 = _bounds(points)
    mask = np.zeros((r1-r0+1, c1-c0+1), dtype=bool)
    for r, c in points:
        mask[r-r0, c-c0] = True
    return len(_components(mask)) == 1


def _primitive(crop):
    scale = 0
    for matrix in (crop, crop.T):
        for row in matrix:
            edges = [0, *(np.flatnonzero(row[1:] != row[:-1]) + 1).tolist(), len(row)]
            for a, b in zip(edges, edges[1:]):
                scale = gcd(scale, b-a)
    small = crop[::scale, ::scale].copy()
    _require(np.array_equal(crop, np.repeat(np.repeat(small, scale, 0), scale, 1)),
             "glyph is not an exact square-block enlargement")
    return scale, small


def _register(raw_maps):
    """Align the two cropped states using their unchanged colored structure."""
    a, b = raw_maps
    shifts = set()
    for color in sorted(set(a.values()) & set(b.values())):
        pa = {p for p, v in a.items() if v == color}
        pb = {p for p, v in b.items() if v == color}
        ar, _, ac, _ = _bounds(pa)
        br, _, bc, _ = _bounds(pb)
        shift = ar-br, ac-bc
        if pa == {(r+shift[0], c+shift[1]) for r, c in pb}:
            shifts.add(shift)
    candidates = []
    for dr, dc in sorted(shifts):
        shifted = {(r+dr, c+dc): v for (r, c), v in b.items()}
        common = {p: v for p, v in a.items() if shifted.get(p) == v}
        if len(common) >= 3:
            candidates.append((len(common), (dr, dc), shifted, common))
    _require(bool(candidates), "no invariant colored scaffold")
    score = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == score]
    _require(len(best) == 1, "ambiguous scaffold registration")
    _, shift, aligned_b, common = best[0]
    r0, r1, c0, c1 = _bounds(common)
    templates = [{(r-r0, c-c0): v for (r, c), v in m.items()}
                 for m in (a, aligned_b)]
    scaffold = {(r-r0, c-c0): v for (r, c), v in common.items()}
    return templates, scaffold, (r1-r0+1, c1-c0+1), (r0, c0), shift


def _separated(a, b):
    r, c, h, w = a
    y, x, hh, ww = b
    return r+h+1 <= y or y+hh+1 <= r or c+w+1 <= x or x+ww+1 <= c


def _rectangle(anchor, scale, union_bounds):
    r0, r1, c0, c1 = union_bounds
    return (anchor[0]+r0*scale, anchor[1]+c0*scale,
            (r1-r0+1)*scale, (c1-c0+1)*scale)


def _draw(grid, template, anchor, scale):
    for (r, c), color in template.items():
        y, x = anchor[0]+r*scale, anchor[1]+c*scale
        grid[y:y+scale, x:x+scale] = color


def _axis_votes(occurrences, shape, *, states=True, scales=True):
    votes = {}
    for i, a in enumerate(occurrences):
        for j in range(i+1, len(occurrences)):
            b = occurrences[j]
            if states and a["state"] == b["state"]:
                continue
            if scales and a["scale"] != b["scale"]:
                continue
            for orientation in (0, 1):  # 0 vertical, 1 horizontal
                coordinate = 1-orientation
                perpendicular = orientation
                if a["center2"][perpendicular] != b["center2"][perpendicular]:
                    continue
                aa, bb = a["center2"][coordinate], b["center2"][coordinate]
                if aa == bb or (aa+bb) % 2:
                    continue
                axis2 = (aa+bb)//2
                if 0 <= axis2 <= 2*(shape[coordinate]-1):
                    votes.setdefault((orientation, axis2), set()).update((i, j))
    return votes


def _read_input(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg]*2 > grid.size, "majority background required")
    objects = []
    for cells in _components(grid != bg):
        r0, r1, c0, c1 = _bounds(cells)
        crop = grid[r0:r1+1, c0:c1+1]
        _require(np.count_nonzero(crop != bg) == len(cells), "interleaved glyphs")
        scale, small = _primitive(crop)
        mapping = {(int(r), int(c)): int(small[r, c])
                   for r, c in np.argwhere(small != bg)}
        objects.append({"box": (r0, c0), "scale": scale,
                        "raw": _signature(mapping), "cells": cells})
    signatures = sorted({o["raw"] for o in objects})
    _require(len(signatures) == 2, "exactly two normalized glyph states required")
    _require({o["raw"] for o in objects if o["scale"] == 1} == set(signatures),
             "both states must be visible at unit scale")
    raw_maps = [{(r, c): v for r, c, v in sig} for sig in signatures]
    templates, scaffold, common_shape, origin, shift = _register(raw_maps)
    union_bounds = _bounds(set(templates[0]) | set(templates[1]))
    occurrences, rectangles = [], []
    for obj in objects:
        state = signatures.index(obj["raw"])
        s = obj["scale"]
        dr, dc = shift if state else (0, 0)
        anchor = (obj["box"][0]+(origin[0]-dr)*s,
                  obj["box"][1]+(origin[1]-dc)*s)
        rect = _rectangle(anchor, s, union_bounds)
        r, c, h, w = rect
        _require(0 <= r and r+h <= grid.shape[0] and 0 <= c and c+w <= grid.shape[1],
                 "a complete replacement state would be clipped")
        _require(all(_separated(rect, other) for other in rectangles),
                 "replacement rectangles must be separated")
        rectangles.append(rect)
        occurrences.append({"anchor": anchor, "scale": s, "state": state,
                            "center2": (2*anchor[0]+common_shape[0]*s-1,
                                        2*anchor[1]+common_shape[1]*s-1),
                            "rectangle": rect, "cells": obj["cells"]})
    _require(len({o["center2"] for o in occurrences}) == len(occurrences),
             "duplicate scaffold centers")
    votes = _axis_votes(occurrences, grid.shape)
    _require(bool(votes), "no complementary reflection support")
    maximum = max(map(len, votes.values()))
    winners = [axis for axis, sites in votes.items() if len(sites) == maximum]
    _require(maximum >= 4 and len(winners) == 1, "tied or insufficient axis consensus")
    axis = winners[0]
    keep = votes[axis]
    expected = np.full(grid.shape, bg, dtype=int)
    for i in sorted(keep):
        obj = occurrences[i]
        _draw(expected, templates[1-obj["state"]], obj["anchor"], obj["scale"])
    return {"grid": grid, "background": bg, "templates": templates,
            "scaffold": scaffold, "common_shape": common_shape,
            "occurrences": occurrences, "axis": axis, "votes": votes,
            "keep": keep, "expected": expected, "union_bounds": union_bounds}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the input's glyphs, scales and centers."""
    return _read_input(input_grid)["expected"].copy()


def _patch(rng, core):
    r0, r1, c0, c1 = _bounds(core)
    side = int(rng.integers(4))
    if side < 2:
        roots = sorted(p for p in core if p[0] == (r0 if side == 0 else r1))
        direction, lateral = ((-1, 0) if side == 0 else (1, 0)), (0, 1)
    else:
        roots = sorted(p for p in core if p[1] == (c0 if side == 2 else c1))
        direction, lateral = ((0, -1) if side == 2 else (0, 1)), (1, 0)
    root = roots[int(rng.integers(len(roots)))]
    p = root[0]+direction[0], root[1]+direction[1]
    patch = {p}
    if rng.random() < .55:
        sign = int(rng.choice([-1, 1]))
        patch.add((p[0]+lateral[0]*sign, p[1]+lateral[1]*sign))
    return patch


def _transform_map(mapping, rotation, reflected):
    result = {}
    for (r, c), color in mapping.items():
        for _ in range(rotation):
            r, c = c, -r
        result[r, -c if reflected else c] = color
    return result


def _sample_design(rng, change, *, compact=False, fallback=False):
    """Procedural small footprints and patches; no stored template vocabulary."""
    for attempt in range(80):
        if compact or fallback:
            core = {(0, 0), (0, 1), (1, 0), (1, 1)}
            moving = {(1, 1)} if change != "relocate" else set()
        else:
            h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
            points = [(r, c) for r in range(h) for c in range(w)]
            n = int(rng.integers(4, h*w+1))
            chosen = rng.choice(len(points), n, replace=False)
            core = {points[int(i)] for i in chosen}
            if not _connected(core):
                continue
            moving = set()
            if change != "relocate":
                candidates = sorted(core)
                rng.shuffle(candidates)
                moving = set(candidates[:int(rng.integers(1, min(2, len(core)-3)+1))])
        stable = core-moving
        if len(stable) < 3 or _bounds(stable) != _bounds(core):
            continue
        if fallback:
            a = {(-1, 0)}
            b = {(0, -1)} if change != "recolor" else a
        else:
            a = _patch(rng, core)
            b = a if change == "recolor" else _patch(rng, core)
        if change != "recolor" and a == b:
            continue
        bounds = _bounds(core | a | b)
        limit = 3 if compact or fallback else 4
        if bounds[1]-bounds[0]+1 > limit or bounds[3]-bounds[2]+1 > limit:
            continue
        shared = {p: 1 for p in stable}
        if len(stable) >= 5 and rng.random() < .45:
            shared[sorted(stable)[int(rng.integers(len(stable)))]] = 2
        maps = []
        for state, patch in enumerate((a, b)):
            mapping = dict(shared)
            for j, p in enumerate(sorted(patch)):
                mapping[p] = (3 if change == "relocate" else 3+state)
                if len(patch) > 1 and j == 1 and rng.random() < .45:
                    mapping[p] = 7 if change == "relocate" else 7+state
            for p in moving:
                mapping[p] = 5+state
            maps.append(mapping)
        if not all(_connected(m) for m in maps):
            continue
        rotation, reflected = int(rng.integers(4)), bool(rng.integers(2))
        maps = [_transform_map(m, rotation, reflected) for m in maps]
        common = {p: v for p, v in maps[0].items() if maps[1].get(p) == v}
        r0, _, c0, _ = _bounds(common)
        maps = [{(r-r0, c-c0): v for (r, c), v in m.items()} for m in maps]
        valid = True
        for m in maps:
            rr0, rr1, cc0, cc1 = _bounds(m)
            crop = np.zeros((rr1-rr0+1, cc1-cc0+1), dtype=int)
            for (r, c), color in m.items():
                crop[r-rr0, c-cc0] = color
            valid &= _primitive(crop)[0] == 1
        if not valid:
            continue
        # Check the intended common registration even before rendering copies.
        try:
            aligned, common, _, _, _ = _register(maps)
        except ValueError:
            continue
        if tuple(map(_signature, aligned)) != tuple(map(_signature, maps)):
            continue
        palette = [int(c) for c in rng.permutation(10)]
        colored = [{p: palette[v] for p, v in m.items()} for m in maps]
        return palette[0], colored
    raise ValueError("bounded design sampling exhausted")


def _recipe(kind, shift, rng):
    def pair(scales, states, role):
        return {"scales": tuple(scales), "states": tuple(states), "role": role}
    side = int(rng.integers(2))
    if kind == "teach_same":
        return [pair((1, 1), (side, 1-side), "good"),
                pair((2, 2), (1-side, side), "good"),
                pair((1, 1), (side, side), "same"),
                pair((1,), (1-side,), "single")]
    if kind == "teach_votes":
        return [pair((1, 1), (side, 1-side), "good"),
                pair((1, 1), (side, 1-side), "good"),
                pair((2, 2), (side, 1-side), "false"),
                pair((1, 2), (1-side, side), "mismatch"),
                pair((1,), (side,), "fixed")]
    good_scales = [1, 3] if kind == "test" and shift == "scale" else (
        [1, 1, 2] if kind == "test" else [1, int(rng.choice([1, 2]))])
    groups = [pair((s, s), (int(rng.integers(2)), 0), "good") for s in good_scales]
    for group in groups:
        group["states"] = (group["states"][0], 1-group["states"][0])
    role = str(rng.choice(["same", "mismatch"]))
    groups.append(pair((1, 1) if role == "same" else (1, 2),
                       (side, side) if role == "same" else (side, 1-side), role))
    if role == "same" or kind != "test":
        groups.append(pair((1,), (side,), "single"))
    return groups


def _columns(group, width, axis2, common_width, bounds):
    s0 = group["scales"][0]
    lo, hi = bounds[2], bounds[3]
    candidates = []
    if len(group["scales"]) == 1:
        if group["role"] == "fixed":
            numerator = axis2-common_width*s0+1
            anchors = [numerator//2] if numerator % 2 == 0 else []
        else:
            anchors = range(-lo*s0, width-(hi+1)*s0+1)
        for c in anchors:
            if 0 <= c+lo*s0 and c+(hi+1)*s0 <= width:
                if group["role"] == "single" and 2*c+common_width*s0-1 == axis2:
                    continue
                candidates.append((c,))
        return candidates
    s1 = group["scales"][1]
    if common_width*(s0+s1) % 2:
        return []
    for c in range(-lo*s0, width-(hi+1)*s0+1):
        cc = axis2-c-common_width*(s0+s1)//2+1
        left0, right0 = c+lo*s0, c+(hi+1)*s0
        left1, right1 = cc+lo*s1, cc+(hi+1)*s1
        if (0 <= left1 and right1 <= width and right0+1 <= left1
                and 2*c+common_width*s0-1 < axis2):
            candidates.append((c, cc))
    return candidates


def _row_offsets(group, common_height, bounds):
    s0 = group["scales"][0]
    if any(common_height*(s0-s) % 2 for s in group["scales"]):
        raise ValueError("unequal scales need compatible center parity")
    offsets = [common_height*(s0-s)//2 for s in group["scales"]]
    low = min(dr+bounds[0]*s for dr, s in zip(offsets, group["scales"]))
    high = max(dr+(bounds[1]+1)*s for dr, s in zip(offsets, group["scales"]))
    return [dr-low for dr in offsets], high-low


def _arrange(rng, templates, groups, force_fixed):
    common = {p for p, v in templates[0].items() if templates[1].get(p) == v}
    cr0, cr1, cc0, cc1 = _bounds(common)
    assert cr0 == cc0 == 0
    ch, cw = cr1+1, cc1+1
    bounds = _bounds(set(templates[0]) | set(templates[1]))
    offsets_heights = [_row_offsets(g, ch, bounds) for g in groups]
    min_height = sum(h for offsets, h in offsets_heights)+len(groups)-1
    _require(min_height <= 30, "row packing too tall")
    widths = list(range(18, 31))
    rng.shuffle(widths)
    for width in widths:
        axes = list(range(max(1, width-8), min(2*width-2, width+7)+1))
        rng.shuffle(axes)
        for axis2 in axes:
            if force_fixed and axis2 % 2 != 1:
                continue
            column_options = []
            chosen_axes = []
            for group in groups:
                options = []
                if group["role"] == "false":
                    for other_axis in range(max(0, axis2-9), min(2*width-2, axis2+9)+1):
                        if abs(other_axis-axis2) >= 2:
                            options += [(other_axis, columns) for columns in
                                        _columns(group, width, other_axis, cw, bounds)]
                else:
                    options = [(axis2, columns) for columns in
                               _columns(group, width, axis2, cw, bounds)]
                if not options:
                    break
                other_axis, columns = options[int(rng.integers(len(options)))]
                column_options.append(columns)
                chosen_axes.append(other_axis)
            if len(column_options) != len(groups):
                continue
            height = int(rng.integers(min_height, min(30, min_height+6)+1))
            order = list(map(int, rng.permutation(len(groups))))
            spare = height-min_height
            extras = [0]*(len(groups)+1)
            for _ in range(spare):
                extras[int(rng.integers(len(extras)))] += 1
            tops = [0]*len(groups)
            y = extras[0]
            for pos, index in enumerate(order):
                tops[index] = y
                y += offsets_heights[index][1]+1+extras[pos+1]

            def placements(index, top, columns):
                group = groups[index]
                offsets = offsets_heights[index][0]
                return [{"anchor": (top+dr, c), "scale": s, "state": state,
                         "keep": group["role"] == "good"}
                        for dr, c, s, state in zip(offsets, columns, group["scales"], group["states"])]

            local = [placements(i, tops[i], column_options[i]) for i in range(len(groups))]
            # Reposition whole groups. Small groups can occupy the same rows at
            # different reflection distances, without a rigid row-slot layout.
            for _ in range(70):
                i = int(rng.integers(len(groups)))
                available = _columns(groups[i], width, chosen_axes[i], cw, bounds)
                columns = available[int(rng.integers(len(available)))]
                top = int(rng.integers(0, height-offsets_heights[i][1]+1))
                candidate = placements(i, top, columns)
                other_rects = [_rectangle(p["anchor"], p["scale"], bounds)
                               for j, group in enumerate(local) if j != i for p in group]
                rects = [_rectangle(p["anchor"], p["scale"], bounds) for p in candidate]
                if all(_separated(a, b) for a in rects for b in other_rects):
                    local[i] = candidate
            return (height, width), [p for group in local for p in group]
    raise ValueError("no complete symmetric packing")


def _make_pair(rng, kind, shift, orientation):
    compact = kind == "teach_votes"
    change = "combined" if kind in ("teach_same", "teach_votes", "test") else str(
        rng.choice(["recolor", "relocate", "combined"]))
    groups = _recipe(kind, shift, rng)
    for attempt in range(36):
        fallback = attempt >= 28
        try:
            bg, templates = _sample_design(rng, change, compact=compact, fallback=fallback)
            shape, placements = _arrange(rng, templates, groups, compact)
            inp = np.full(shape, bg, dtype=int)
            out = inp.copy()
            for p in placements:
                _draw(inp, templates[p["state"]], p["anchor"], p["scale"])
                if p["keep"]:
                    _draw(out, templates[1-p["state"]], p["anchor"], p["scale"])
            if rng.random() < .5:
                inp, out = np.fliplr(inp), np.fliplr(out)
            if rng.random() < .5:
                inp, out = np.flipud(inp), np.flipud(out)
            if orientation:
                inp, out = inp.T, out.T
            pair = {"input": inp.copy(), "output": out.copy()}
            info = _read_input(pair["input"])
            if not np.array_equal(info["expected"], pair["output"]):
                continue
            _check_witness(info, kind)
            return pair
        except ValueError:
            continue
    raise RuntimeError("bounded complementary-pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, "num_train")
    ns = _count(num_test, 1, 4, "num_test")
    if nt <= ns:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["scale", "count"]))
    orientation = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            kind = ("teach_same" if i == 0 else "teach_votes" if i == 1 else "extra") if split == "train" else "test"
            for _ in range(20):
                pair = _make_pair(rng, kind, shift, orientation)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _render_alternative(info, sites, *, exchange):
    grid = np.full(info["grid"].shape, info["background"], dtype=int)
    for i in sorted(sites):
        obj = info["occurrences"][i]
        state = 1-obj["state"] if exchange else obj["state"]
        _draw(grid, info["templates"][state], obj["anchor"], obj["scale"])
    return grid


def _check_witness(info, kind):
    objects, axis, keep = info["occurrences"], info["axis"], info["keep"]
    _require(len(keep) < len(objects), "an erased-occurrence witness is required")
    if kind == "teach_same":
        geometry = _axis_votes(objects, info["grid"].shape, states=False).get(axis, set())
        _require(bool(geometry-keep), "missing same-state decoy demonstration")
        _require({objects[i]["scale"] for i in keep} == {1, 2}, "missing scale-two retention")
        _require(_bounds(info["templates"][0]) != _bounds(info["templates"][1]),
                 "first demonstration must change the decorated bounding box")
        _require(axis[1] != info["grid"].shape[1-axis[0]]-1,
                 "first demonstration must distinguish the canvas-center shortcut")
    if kind == "teach_votes":
        _require(len(keep) == 4 and all(objects[i]["scale"] == 1 for i in keep),
                 "second demonstration must have two valid unit pairs")
        unscaled = _axis_votes(objects, info["grid"].shape, scales=False).get(axis, set())
        _require(bool(unscaled-keep), "missing unequal-scale decoy demonstration")
        coordinate = 1-axis[0]
        _require(any(i not in keep and o["center2"][coordinate] == axis[1]
                     for i, o in enumerate(objects)), "missing axis-centered singleton")
        areas = {a: sum(len(info["templates"][objects[i]["state"]])*objects[i]["scale"]**2
                        for i in sites) for a, sites in info["votes"].items()}
        _require(any(a != axis and area > areas[axis] for a, area in areas.items()),
                 "missing count-versus-area competing axis")


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists required")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, 2, 10, "training length")
    _count(ns, 1, 4, "test length")
    _require(nt > ns, "training length must exceed test length")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            info = _read_input(inp)
            _require(np.array_equal(out, info["expected"]), "incorrect complementary exchange")
            bg, objects, keep = info["background"], info["occurrences"], info["keep"]
            for i, obj in enumerate(objects):
                if i not in keep:
                    r, c, h, w = obj["rectangle"]
                    _require(np.all(out[r:r+h, c:c+w] == bg), "outlier was not wholly erased")
                else:
                    for (r, c), color in info["scaffold"].items():
                        y, x = obj["anchor"][0]+r*obj["scale"], obj["anchor"][1]+c*obj["scale"]
                        _require(np.all(out[y:y+obj["scale"], x:x+obj["scale"]] == color),
                                 "supported scaffold moved or changed")
            _check_witness(info, "teach_same" if split == "train" and index == 0 else
                           "teach_votes" if split == "train" and index == 1 else "other")
            b_only = _render_alternative(info, range(len(objects)), exchange=True)
            a_only = _render_alternative(info, keep, exchange=False)
            _require(not np.array_equal(out, b_only), "symmetry deletion makes no difference")
            _require(not np.array_equal(out, a_only), "state exchange makes no difference")
            infos[split].append(info)
    train_max = max(o["scale"] for i in infos["train"] for n, o in enumerate(i["occurrences"]) if n in i["keep"])
    for info in infos["test"]:
        kept_scales = [o["scale"] for n, o in enumerate(info["occurrences"]) if n in info["keep"]]
        _require(max(kept_scales) > train_max or len(info["keep"]) > max(len(i["keep"]) for i in infos["train"]),
                 "test lacks the intended scale/count shift")
    _require(len({i["axis"][0] for split in infos.values() for i in split}) == 1,
             "generated puzzle changes reflection orientation")


def _geometry(points):
    variants = []
    for rotation in range(4):
        for reflected in (False, True):
            transformed = _transform_map({p: 1 for p in points}, rotation, reflected)
            r0, _, c0, _ = _bounds(transformed)
            variants.append(tuple(sorted((r-r0, c-c0) for r, c in transformed)))
    return min(variants)


def _vocabulary(info):
    keys = []
    for order in ((0, 1), (1, 0)):
        roles, encoded = {}, []
        for state in order:
            encoded.append(tuple((r, c, roles.setdefault(v, len(roles)))
                                 for r, c, v in _signature(info["templates"][state])))
        keys.append(tuple(encoded))
    return min(keys)


def self_test():
    """Bounded 42-seed structural coverage, algebraic checks and mutations."""
    shapes, backgrounds, geometries, vocabularies = set(), set(), set(), set()
    axes, scales, supported_counts, fingerprints = set(), set(), set(), set()
    schedules, frequencies, erased_counts, edge_count = set(), set(), set(), 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p["input"]) for split in puzzle.values() for p in split))
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[role], q[role]) for split in puzzle
                       for p, q in zip(puzzle[split], again[split]) for role in ("input", "output"))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                # Two exchanges restore exactly the supported source collection.
                cleaned = _render_alternative(info, info["keep"], exchange=False)
                assert np.array_equal(solve(pair["output"]), cleaned)
                shapes.add(pair["input"].shape)
                backgrounds.add(info["background"])
                geometries.add(_geometry(info["scaffold"]))
                vocabularies.add(_vocabulary(info))
                axes.add((info["axis"][0], info["axis"][1] % 2))
                objects = info["occurrences"]
                schedule = tuple(sorted(o["scale"] for o in objects))
                schedules.add(schedule)
                frequencies.add(tuple(sorted(sum(o["state"] == s for o in objects) for s in (0, 1))))
                scales.update(o["scale"] for n, o in enumerate(objects) if n in info["keep"])
                supported_counts.add(len(info["keep"]))
                erased_counts.add(len(objects)-len(info["keep"]))
                bg = info["background"]
                edge_count += any(np.any(g[0] != bg) or np.any(g[-1] != bg)
                                  or np.any(g[:, 0] != bg) or np.any(g[:, -1] != bg)
                                  for g in pair.values())
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(geometries) >= 8 and len(vocabularies) >= 45
    assert backgrounds == set(range(10))
    assert axes == {(orientation, parity) for orientation in (0, 1) for parity in (0, 1)}
    assert scales == {1, 2, 3} and supported_counts == {4, 6}
    assert len(schedules) >= 5 and len(frequencies) >= 3 and len(erased_counts) >= 2
    assert edge_count >= 5
    # An equal checkerboard of states has both horizontal and vertical support.
    # This is a real axis tie, not a missing-template or packing failure.
    reference = _read_input(generate_puzzle(300, 2, 1)["train"][1]["input"])
    bg, templates = reference["background"], reference["templates"]
    tied = np.full((24, 24), bg, dtype=int)
    for anchor, state in (((4, 4), 0), ((4, 15), 1), ((15, 4), 1), ((15, 15), 0)):
        _draw(tied, templates[state], anchor, 1)
    try:
        solve(tied)
    except ValueError as error:
        assert "tied" in str(error)
    else:
        raise AssertionError("tied reflection axes accepted")
    missing = np.full((24, 24), bg, dtype=int)
    for anchor in ((4, 4), (4, 15), (15, 4), (15, 15)):
        _draw(missing, templates[0], anchor, 1)
    try:
        solve(missing)
    except ValueError as error:
        assert "two normalized" in str(error)
    else:
        raise AssertionError("absent second state accepted")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(10000+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(190, 2, 1)
    for mutate in ("kept_scaffold", "discarded", "background"):
        bad = {split: [{role: np.array(grid, copy=True) for role, grid in pair.items()}
                       for pair in pairs] for split, pairs in sample.items()}
        pair = bad["test"][0]
        info = _read_input(pair["input"])
        if mutate == "kept_scaffold":
            obj = info["occurrences"][min(info["keep"])]
            r, c = min(info["scaffold"])
            point = obj["anchor"][0]+r*obj["scale"], obj["anchor"][1]+c*obj["scale"]
        elif mutate == "discarded":
            obj = next(o for n, o in enumerate(info["occurrences"]) if n not in info["keep"])
            point = min(obj["cells"])
        else:
            point = tuple(np.argwhere(pair["input"] == info["background"])[0])
        pair["output"][point] = (int(pair["output"][point])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {mutate} mutation")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "grid_shapes": len(shapes),
            "scaffold_geometries": len(geometries), "state_vocabularies": len(vocabularies),
            "scale_schedules": len(schedules), "state_frequencies": len(frequencies),
            "edge_pairs": edge_count, "supported_counts": sorted(supported_counts),
            "kept_scales": sorted(scales), "erased_counts": sorted(erased_counts)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
