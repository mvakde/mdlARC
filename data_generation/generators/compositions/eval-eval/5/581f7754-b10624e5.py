"""Calibrated stencil docking: four boundary bands constrain one similarity map.

Construction paints a base lattice. The input-only decoder instead measures
registration squares and reconstructs an exact rational raster from actual pixels.
No parent files, seed state, or construction records are used by solve().
"""
from __future__ import annotations

from collections import Counter
import numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _check(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "illegal grid dimensions")
    _check(np.issubdtype(g.dtype, np.integer), "noninteger grid")
    _check(bool(np.all((g >= 0) & (g <= 9))), "illegal colors")
    return g.astype(np.int8, copy=False)


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _decode(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    corners = [int(g[r, c]) for r, c in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))]
    _check(len(set(corners)) == 1, "corners must identify background")
    bg = corners[0]
    components = _components(map(tuple, np.argwhere(g != bg).tolist()))
    objects = [p for p in components if len({int(g[q]) for q in p}) > 1]
    _check(len(objects) == 1, "expected one connected multicolored stencil")
    source = objects[0]
    guides, fixed = {}, {}
    for p in components:
        if p == source:
            continue
        color = int(g[next(iter(p))])
        rows, cols = {r for r, _ in p}, {c for _, c in p}
        vertical = len(rows) == 1 and next(iter(rows)) in (0, h-1)
        horizontal = len(cols) == 1 and next(iter(cols)) in (0, w-1)
        _check(vertical != horizontal, "guide band must occupy exactly one outer edge")
        axis = 1 if vertical else 0
        coordinates = sorted({q[axis] for q in p})
        _check(coordinates == list(range(coordinates[0], coordinates[-1]+1)), "broken band")
        _check(0 < coordinates[0] and coordinates[-1] < g.shape[axis]-1, "corner band")
        band = (axis, coordinates[0], len(coordinates))
        if color in guides:
            _check(guides[color]["band"] == band, "inconsistent opposite-edge bands")
        else:
            guides[color] = {"band": band, "ends": set()}
        end = next(iter(rows if vertical else cols))
        guides[color]["ends"].add(end)
        fixed.update({q: color for q in p})
    _check(len(guides) == 4, "expected four guide identities")
    axes = [[c for c in sorted(guides) if guides[c]["band"][0] == axis] for axis in (0, 1)]
    _check(all(len(cs) == 2 for cs in axes), "expected two bands on each axis")
    widths = {v["band"][2] for v in guides.values()}
    _check(len(widths) == 1, "all destination bands must have equal width")
    target_side = widths.pop()
    marks = {}
    for color in guides:
        cells = [p for p in source if int(g[p]) == color]
        _check(cells, "missing registration square")
        rs, cs = zip(*cells)
        side = max(rs)-min(rs)+1
        _check(side == max(cs)-min(cs)+1 and len(cells) == side*side,
               "registration color must form one solid square")
        marks[color] = (min(rs), min(cs), side)
    source_sides = {m[2] for m in marks.values()}
    _check(len(source_sides) == 1, "source registration squares must be equal")
    source_side = source_sides.pop()
    _check(set(int(g[p]) for p in source) - set(guides), "stencil needs payload")
    signs = []
    for axis, colors in enumerate(axes):
        a, b = colors
        ds = marks[b][axis]-marks[a][axis]
        dt = guides[b]["band"][1]-guides[a]["band"][1]
        _check(ds != 0 and dt != 0 and abs(dt)*source_side == abs(ds)*target_side,
               "guide separation is not compatible with uniform scaling")
        signs.append(1 if ds*dt > 0 else -1)
    return {"grid": g, "bg": bg, "source": source, "guides": guides,
            "fixed": fixed, "marks": marks, "axes": axes,
            "s": source_side, "t": target_side, "signs": tuple(signs)}


def _transformed(data, scale=True, reflect=True):
    """Exact area rasterization; reject cells that straddle a color boundary."""
    g, bg = data["grid"], data["bg"]
    s, t = data["s"], data["t"] if scale else data["s"]
    origin = [data["marks"][data["axes"][axis][0]][axis] for axis in (0, 1)]
    points = np.array(sorted(data["source"]), dtype=int)
    lo = ((points.min(0)-origin)//s)*s
    hi = ((points.max(0)+1-origin+s-1)//s)*s
    raw = np.full(tuple(hi-lo), bg, dtype=np.int8)
    for r, c in data["source"]:
        raw[r-origin[0]-lo[0], c-origin[1]-lo[1]] = g[r, c]
    refined = np.repeat(np.repeat(raw, t, 0), t, 1)
    nh, nw = refined.shape[0]//s, refined.shape[1]//s
    blocks = refined.reshape(nh, s, nw, s)
    low, high = blocks.min((1, 3)), blocks.max((1, 3))
    _check(np.array_equal(low, high), "ambiguous fractional rasterization")
    offset = (-lo*t//s).tolist()
    signs = data["signs"] if reflect else (1, 1)
    for axis in (0, 1):
        if signs[axis] < 0:
            low = np.flip(low, axis)
            offset[axis] = low.shape[axis]-offset[axis]-t
    return low, offset, origin


def _paste(grid, raster, top, left, bg):
    h, w = grid.shape
    r0, c0 = max(0, top), max(0, left)
    r1, c1 = min(h, top+raster.shape[0]), min(w, left+raster.shape[1])
    if r0 < r1 and c0 < c1:
        part = raster[r0-top:r1-top, c0-left:c1-left]
        window = grid[r0:r1, c0:c1]
        window[part != bg] = part[part != bg]


def _prediction(data, guides=True, scale=True, reflect=True):
    raster, offset, origin = _transformed(data, scale, reflect)
    target = ([data["guides"][data["axes"][axis][0]]["band"][1] for axis in (0, 1)]
              if guides else origin)
    top, left = target[0]-offset[0], target[1]-offset[1]
    out = np.full_like(data["grid"], data["bg"])
    _paste(out, raster, top, left, data["bg"])
    for p, color in data["fixed"].items():
        _check(int(out[p]) == data["bg"], "target stencil overlaps a fixed band")
        out[p] = color
    total = int(np.count_nonzero(raster != data["bg"]))
    visible = int(np.count_nonzero(out != data["bg"]))-len(data["fixed"])
    return out, total-visible


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the complete coordinate map from this input only."""
    return _prediction(_decode(input_grid))[0]


def _skeleton(rng, style, limit):
    """Coarse footprints: growth, ring, comb, stair, notched slab, two lobes."""
    if style == 0:
        cells = {(0, 0)}
        for _ in range(int(rng.integers(9, 20))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIRS}-cells)
            choices = [p for p in frontier if max([p[0]]+[r for r, _ in cells])-min([p[0]]+[r for r, _ in cells]) < limit
                       and max([p[1]]+[c for _, c in cells])-min([p[1]]+[c for _, c in cells]) < limit]
            if not choices:
                break
            cells.add(choices[int(rng.integers(len(choices)))])
    elif style == 1:
        h, w = int(rng.integers(3, limit+1)), int(rng.integers(3, limit+1))
        cells = {(r, c) for r in range(h) for c in range(w) if r in (0, h-1) or c in (0, w-1)}
    elif style == 2:
        n = int(rng.integers(3, limit+1))
        cells = {(r, 0) for r in range(n)}
        for r in range(0, n, 2):
            sign = int(rng.choice([-1, 1]))
            cells.update((r, sign*c) for c in range(1, int(rng.integers(2, 4))))
    elif style == 3:
        n = int(rng.integers(3, limit+1))
        cells = {(r, r) for r in range(n)} | {(r, r+1) for r in range(n-1)}
        for r in range(n-1):
            if rng.random() < .5:
                cells.add((r+1, r))
    elif style == 4:
        h, w = int(rng.integers(3, limit+1)), int(rng.integers(3, limit+1))
        cells = {(r, c) for r in range(h) for c in range(w)}
        depth = int(rng.integers(1, h-1))
        cells -= {(r, w//2) for r in range(depth)}
    else:
        cells = {(r, c) for r in range(2) for c in range(2)}
        cells |= {(r+2, c+2) for r in range(2) for c in range(2)}
        cells |= {(1, 2), (2, 2)}
        if rng.random() < .5:
            cells |= {(0, 2), (3, 1)}
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    return sorted(cells)


def _base_motif(rng, palette, a, style, extra_branches):
    bg, colors, payload = palette[0], palette[1:5], palette[5:]
    cells = _skeleton(rng, style, 5 if a == 1 else 4)
    assignments = None
    for _ in range(80):
        indices = rng.choice(len(cells), 4, replace=False)
        selected = [cells[int(i)] for i in indices]
        # First two identities are vertical bands; last two horizontal bands.
        if abs(selected[0][1]-selected[1][1]) >= 2 and abs(selected[2][0]-selected[3][0]) >= 2:
            assignments = selected
            break
    if assignments is None:
        return None
    detail_count = int(rng.integers(1, min(4, len(payload))+1))
    detail_colors = payload[:detail_count]
    motif = {}
    for r, c in cells:
        color = int(rng.choice(detail_colors))
        for dr in range(a):
            for dc in range(a):
                motif[r*a+dr, c*a+dc] = color
    marked = set()
    marks = []
    for (r, c), color in zip(assignments, colors):
        marks.append((r*a, c*a))
        for dr in range(a):
            for dc in range(a):
                p = r*a+dr, c*a+dc
                motif[p] = color
                marked.add(p)
    # Thin colored appendages link coarse squares to sub-anchor geometry and
    # create boundary interactions without removing registration information.
    for _ in range(extra_branches):
        direction = DIRS[int(rng.integers(4))]
        candidates = sorted(p for p in motif if (p[0]+direction[0], p[1]+direction[1]) not in motif)
        p = candidates[int(rng.integers(len(candidates)))]
        color = int(rng.choice(detail_colors))
        for distance in range(1, int(rng.integers(3, 6))):
            q = p[0]+distance*direction[0], p[1]+distance*direction[1]
            if q not in marked:
                motif[q] = color
    rs, cs = zip(*motif)
    r0, c0 = min(rs), min(cs)
    raw = np.full((max(rs)-r0+1, max(cs)-c0+1), bg, np.int8)
    for (r, c), color in motif.items():
        raw[r-r0, c-c0] = color
    return raw, [(r-r0, c-c0) for r, c in marks]


def _visible_count(mask, top, left, h, w):
    r0, c0 = max(0, -top), max(0, -left)
    r1, c1 = min(mask.shape[0], h-top), min(mask.shape[1], w-left)
    return int(mask[r0:r1, c0:c1].sum()) if r0 < r1 and c0 < c1 else 0


def _build_pair(rng, plan):
    palette = list(map(int, rng.permutation(10)))
    bg, colors = palette[0], palette[1:5]
    a = plan.get("anchor_lattice") or int(rng.choice([1, 1, 2]))
    u, v = plan["units"] if plan["units"] else tuple(map(int, rng.integers(1, 4, size=2)))
    style = int(rng.integers(6))
    built = _base_motif(rng, palette, a, style, int(rng.integers(1, 4)))
    if built is None:
        return None
    raw, marks = built
    src = np.repeat(np.repeat(raw, u, 0), u, 1)
    dst = np.repeat(np.repeat(raw, v, 0), v, 1)
    signs = plan["signs"] if plan["signs"] else tuple(int(rng.choice([-1, 1])) for _ in range(2))
    transformed_marks = [list(p) for p in marks]
    for axis in (0, 1):
        if signs[axis] < 0:
            dst = np.flip(dst, axis)
            for p in transformed_marks:
                p[axis] = raw.shape[axis]-p[axis]-a
    transformed_marks = np.array(transformed_marks, dtype=int)*v
    side = a*v
    minimum = np.maximum(np.array(src.shape)+2, transformed_marks.max(0)-transformed_marks.min(0)+side+2)
    if not plan["clip"]:
        minimum = np.maximum(minimum, np.array(dst.shape)+2)
    if np.any(minimum > 30):
        return None
    h, w = [int(rng.integers(int(n), min(30, int(n)+7)+1)) for n in minimum]
    mask = dst != bg
    total = int(mask.sum())
    lo = 1-transformed_marks.min(0)
    hi = np.array([h, w])-1-transformed_marks.max(0)-side
    targets = []
    for top in range(int(lo[0]), int(hi[0])+1):
        for left in range(int(lo[1]), int(hi[1])+1):
            visible = _visible_count(mask, top, left, h, w)
            lost = total-visible
            if bool(lost) == plan["clip"] and visible > 4*side*side and lost <= .45*total:
                targets.append((top, left))
    if not targets:
        return None
    top, left = targets[int(rng.integers(len(targets)))]
    final_marks = transformed_marks + [top, left]
    inp, out = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    _paste(out, dst, top, left, bg)
    for i, color in enumerate(colors):
        axis = 1 if i < 2 else 0
        start = int(final_marks[i, axis])
        # Mandatory demonstrations include single and bilateral bands on both axes.
        ends = [0, 1] if i == plan["bilateral"] or rng.random() < .22 else [int(rng.integers(2))]
        for end in ends:
            positions = [(0 if end == 0 else h-1, c) for c in range(start, start+side)] if axis == 1 else [
                (r, 0 if end == 0 else w-1) for r in range(start, start+side)]
            for p in positions:
                if int(out[p]) != bg or int(inp[p]) != bg:
                    return None
                inp[p] = out[p] = color
    boundary_mask = inp != bg
    blocked = boundary_mask.copy()
    for dr, dc in DIRS:
        shifted = np.zeros_like(blocked)
        if dr == -1:
            shifted[:-1] = boundary_mask[1:]
        elif dr == 1:
            shifted[1:] = boundary_mask[:-1]
        elif dc == -1:
            shifted[:, :-1] = boundary_mask[:, 1:]
        else:
            shifted[:, 1:] = boundary_mask[:, :-1]
        blocked |= shifted
    blocked[0, 0] = blocked[0, -1] = blocked[-1, 0] = blocked[-1, -1] = True
    rr, cc = np.meshgrid(np.arange(h-src.shape[0]+1), np.arange(w-src.shape[1]+1), indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    valid = np.ones(len(rr), bool)
    for dr, dc in np.argwhere(src != bg):
        valid &= ~blocked[rr+dr, cc+dc]
    choices = np.flatnonzero(valid)
    if not len(choices):
        return None
    k = int(rng.choice(choices))
    _paste(inp, src, int(rr[k]), int(cc[k]), bg)
    if np.array_equal(inp, out):
        return None
    try:
        data = _decode(inp)
        predicted, lost = _prediction(data)
        if not np.array_equal(predicted, out) or bool(lost) != plan["clip"]:
            return None
        # The four guide identities all remain complete and visible after docking.
        _check(_different_alternative(data, out, guides=False), "guides do not affect example")
        if signs != (1, 1):
            _check(_different_alternative(data, out, reflect=False), "unobservable reflection")
        if plan.get("anchor_lattice") == 2 and not _features({"input": inp, "output": out})["fine"]:
            return None
    except AssertionError:
        return None
    return {"input": inp, "output": out}


def _signature(g):
    return g.shape, g.astype(np.int8).tobytes()


def _fallback_pair(rng, plan, seen):
    """Finite spacious construction: fourteen origins defeat at most 13 inputs.

    This uses the same four constraints, with a thin arm outside their rectangle.
    Its arm clips away from horizontal guide rows, so fixed bands stay untouched.
    """
    palette = list(map(int, rng.permutation(10)))
    bg, colors, payload = palette[0], palette[1:5], palette[5]
    u, v = plan["units"] or tuple(map(int, rng.integers(1, 3, size=2)))
    signs = plan["signs"] or tuple(int(rng.choice([-1, 1])) for _ in range(2))
    a = 2
    marks = [(1, 1), (1, 5), (3, 3), (6, 3)]
    raw = np.full((8, 10), bg, np.int8)
    raw[2, 1:7] = payload
    raw[2:8, 3] = payload
    raw[5, 3:10] = payload
    for (r, c), color in zip(marks, colors):
        raw[r:r+a, c:c+a] = color
    src = np.repeat(np.repeat(raw, u, 0), u, 1)
    dst = np.repeat(np.repeat(raw, v, 0), v, 1)
    transformed_marks = np.array(marks, dtype=int)
    for axis in (0, 1):
        if signs[axis] < 0:
            dst = np.flip(dst, axis)
            transformed_marks[:, axis] = raw.shape[axis]-transformed_marks[:, axis]-a
    left = (30-8*v if signs[1] > 0 else -2*v) if plan["clip"] else 3
    top = 3
    positions = transformed_marks*v + [top, left]
    template = np.full((30, 30), bg, np.int8)
    out = template.copy()
    _paste(out, dst, top, left, bg)
    for i, color in enumerate(colors):
        axis = 1 if i < 2 else 0
        start = int(positions[i, axis])
        ends = [0, 29] if i == plan["bilateral"] else [0 if i % 2 == 0 else 29]
        for end in ends:
            ps = [(end, c) for c in range(start, start+a*v)] if axis == 1 else [(r, end) for r in range(start, start+a*v)]
            for p in ps:
                _check(int(out[p]) == bg, "fallback collision")
                template[p] = out[p] = color
    for variant in range(14):
        inp = template.copy()
        _paste(inp, src, 4+variant//6, 3+variant % 6, bg)
        if _signature(inp) not in seen:
            pair = {"input": inp, "output": out}
            d = _decode(inp)
            prediction, lost = _prediction(d)
            _check(np.array_equal(prediction, out) and bool(lost) == plan["clip"], "invalid fallback")
            return pair
    raise AssertionError("fourteen fallback origins exhausted by fewer than fourteen inputs")


def _different_alternative(data, out, **kwargs):
    try:
        alternate = _prediction(data, **kwargs)[0]
    except AssertionError:
        # An alternative that puts payload on a fixed band is already contradicted.
        return True
    return not np.array_equal(alternate, out)


def _witness(data, out):
    return (_different_alternative(data, out, guides=False)
            and _different_alternative(data, out, reflect=False))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    result, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train" and i == 0:
                plan = {"units": (1, 2), "signs": (1, -1), "clip": False, "bilateral": 0,
                        "anchor_lattice": 2}
            elif split == "train" and i == 1:
                plan = {"units": (2, 1), "signs": (-1, 1), "clip": True, "bilateral": 2}
            else:
                signs = (((1, 1), (1, -1), (-1, 1))[int(rng.integers(3))] if split == "train" else
                         (-1, -1) if i == 0 else None)
                plan = {"units": None, "signs": signs,
                        "clip": bool(rng.random() < (.65 if split == "test" else .35)),
                        "bilateral": int(rng.integers(4))}
            for _ in range(600):
                pair = _build_pair(rng, plan)
                if pair is not None and _signature(pair["input"]) not in seen:
                    break
            else:
                pair = _fallback_pair(rng, plan, seen)
            seen.add(_signature(pair["input"]))
            result[split].append(pair)
    rng.shuffle(result["train"])
    validate_generated_puzzle(result)
    return result


def _features(pair):
    d = _decode(pair["input"])
    predicted, lost = _prediction(d)
    out = _grid(pair["output"])
    _check(np.array_equal(predicted, out), "wrong docking output")
    _check(out.shape == d["grid"].shape, "canvas changed")
    _check(not np.array_equal(d["grid"], out), "identity example")
    # Direct registration checks supplement whole-grid reconstruction.
    for color, guide in d["guides"].items():
        points = [tuple(p) for p in np.argwhere(out == color).tolist() if tuple(p) not in d["fixed"]]
        _check(len(points) == d["t"]**2, "registration square clipped or duplicated")
        rs, cs = zip(*points)
        top = [min(rs), min(cs)]
        _check(max(rs)-min(rs)+1 == d["t"] and max(cs)-min(cs)+1 == d["t"], "registration shape altered")
        _check(top[guide["band"][0]] == guide["band"][1], "registration misses guide band")
    points = np.array(sorted(d["source"]))
    origin = points.min(0)
    normalized = {}
    structure = []
    for r, c in sorted(d["source"]):
        color = int(d["grid"][r, c])
        normalized.setdefault(color, len(normalized))
        structure.append((int(r-origin[0]), int(c-origin[1]), normalized[color]))
    frame = [d["marks"][d["axes"][axis][0]][axis] for axis in (0, 1)]
    fine = False
    for r, c in d["source"]:
        for dr, dc in DIRS:
            neighbor = r+dr, c+dc
            color = int(d["grid"][neighbor]) if neighbor in d["source"] else d["bg"]
            if color != int(d["grid"][r, c]):
                axis = 0 if dr else 1
                boundary = (r if dr else c) + int(dr > 0 or dc > 0)
                fine |= bool((boundary-frame[axis]) % d["s"])
    return {"shape": tuple(out.shape), "source_side": d["s"], "target_side": d["t"],
            "signs": d["signs"], "clipped": bool(lost), "lost": lost,
            "cells": len(d["source"]), "structure": tuple(structure), "fine": bool(fine),
            "bilateral": sum(len(g["ends"]) == 2 for g in d["guides"].values()),
            "bg": d["bg"], "data": d}


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, train = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(inp)
            _check(key not in seen, "duplicate input")
            seen.add(key)
            features = _features({"input": inp, "output": out})
            if split == "train":
                train.append((features, out))
    _check(any(f["target_side"] > f["source_side"] and not f["clipped"] for f, _ in train),
           "training lacks a complete enlargement")
    _check(any(f["target_side"] < f["source_side"] for f, _ in train), "training lacks reduction")
    _check(any(f["clipped"] for f, _ in train), "clipping needs training evidence")
    _check(any(f["fine"] for f, _ in train), "sub-anchor geometry needs training evidence")
    for axis in (0, 1):
        _check({f["signs"][axis] for f, _ in train} == {-1, 1}, "reflection order not demonstrated")
    _check(any(_witness(f["data"], out) for f, out in train), "parent ablations not distinguished")
    _check(any(_different_alternative(f["data"], out, scale=False) for f, out in train),
           "scaling shortcut not distinguished")


def self_test():
    """Bounded 42-seed coverage, count checks, metamorphisms and corruption checks."""
    signatures, shapes, structures, signs, scales, backgrounds = set(), set(), set(), set(), set(), set()
    clips, bilateral, fine, cells = Counter(), set(), 0, []
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprint = tuple(_signature(q[k]) for split in ("train", "test") for q in p[split] for k in ("input", "output"))
        signatures.add(fingerprint)
        if seed < 3:
            q = generate_puzzle(seed)
            _check(fingerprint == tuple(_signature(x[k]) for split in ("train", "test") for x in q[split] for k in ("input", "output")), "nondeterminism")
        for split in ("train", "test"):
            for pair in p[split]:
                f = _features(pair)
                shapes.add(f["shape"])
                structures.add(f["structure"])
                signs.add(f["signs"])
                scales.add((f["source_side"], f["target_side"]))
                backgrounds.add(f["bg"])
                clips[(split, f["clipped"])] += 1
                bilateral.add(f["bilateral"])
                fine += f["fine"]
                cells.append(f["cells"])
    _check(len(signatures) == 42 and len(shapes) >= 40 and len(structures) >= 100, "structural diversity")
    _check(signs == {(-1, -1), (-1, 1), (1, -1), (1, 1)}, "reflection combinations missing")
    _check(len(scales) >= 12 and backgrounds == set(range(10)), "scale/palette diversity")
    _check(len(bilateral) >= 3 and fine >= 20 and max(cells) > 100, "geometry diversity")
    _check(all(clips[s, clip] for s in ("train", "test") for clip in (False, True)), "clipping diversity")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(1000+a*10+b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(2026, 2, 1)
    rng = np.random.default_rng(777)
    for turns in (1, 2, 3):
        q = {s: [] for s in p}
        for split in p:
            for pair in p[split]:
                palette = rng.permutation(10)
                q[split].append({k: palette[np.rot90(v, turns)].astype(np.int8) for k, v in pair.items()})
        validate_generated_puzzle(q)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    # The fallback is geometric, not a latent alternative rule. Exercise every
    # orientation and clipping choice with reduction, equality and enlargement.
    for u, v in ((1, 2), (2, 1), (1, 1)):
        for signs_pair in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            for clip in (False, True):
                plan = {"units": (u, v), "signs": signs_pair, "clip": clip, "bilateral": 2}
                seen = set()
                for _ in range(14):
                    pair = _fallback_pair(np.random.default_rng(23), plan, seen)
                    _features(pair)
                    seen.add(_signature(pair["input"]))
                _check(len(seen) == 14, "fallback uniqueness")
    # A mixed-color destination cell must not be rounded to a guessed color.
    enlargement = next(pair for pair in p["train"] if _features(pair)["target_side"] > _features(pair)["source_side"])
    d = _decode(enlargement["input"])
    ambiguous = dict(d, t=3)
    try:
        _transformed(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous fine-detail fractional scaling accepted")
    # A damaged registration is visibly invalid, even before an output is known.
    broken_input = enlargement["input"].copy()
    color = next(iter(d["marks"]))
    r, c, _ = d["marks"][color]
    broken_input[r, c] = d["bg"]
    try:
        solve(broken_input)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-square registration accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "canvas_shapes": len(shapes),
            "colored_source_structures": len(structures), "side_pairs": sorted(scales),
            "source_cells_range": [min(cells), max(cells)], "fine_examples": fine,
            "clipping": {f"{s}_{clip}": n for (s, clip), n in clips.items()}}
