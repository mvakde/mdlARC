"""Window stencils: gray stencils are stamped at matching pins and reveal the patchwork beneath.

Family rule (fixed conventions: gray 5 = stencil/window cells, black 0 = output ground):
  * The input is a patchwork "field" of solid colored regions (one or several colors;
    never gray, never black).  On it lie gray stencils, each holding exactly one colored
    pin cell, plus isolated single-cell pins elsewhere.
  * Every isolated pin whose color matches a stencil's pin receives a translated copy of
    that stencil, aligned pin-to-pin (parent A: anchor-matched template stamping).
  * Gray cells carry location only: each copied gray cell becomes the field color that
    lies beneath it at the destination, while every field cell not covered by a copy
    becomes black (parent B: gray mask rendered in the companion color, companion
    becomes black).  Because the field is a patchwork, one copy may show several colors.
  * Copy pins keep their pin color.  Source stencils and unmatched pins vanish.
"""
from __future__ import annotations

import numpy as np

GRAY = 5
BLACK = 0
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
LAYOUTS = ("uniform", "vstripes", "hstripes", "quilt", "guillotine", "inset")
PATCH_LAYOUTS = LAYOUTS[1:]
SHIFTS = ("mild", "layout", "corner", "scale", "roles", "dense")
SHORTCUTS = ("uniform_copy", "source_color", "global_color", "gray_copy", "keep_decoys",
             "no_stamping", "no_reveal")
NEIGH8 = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
NEIGH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


# --------------------------------------------------------------------------- counts
def _check_counts(num_train, num_test):
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


# --------------------------------------------------------------------------- mask helpers
def _shift(mask, dy, dx):
    """result[r, c] = mask[r + dy, c + dx]; cells outside the canvas read as False."""
    height, width = mask.shape
    out = np.zeros_like(mask)
    y0, y1 = max(0, -dy), min(height, height - dy)
    x0, x1 = max(0, -dx), min(width, width - dx)
    if y0 < y1 and x0 < x1:
        out[y0:y1, x0:x1] = mask[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def _dilate(mask):
    out = mask.copy()
    for dy, dx in NEIGH8:
        out |= _shift(mask, dy, dx)
    return out


def _placements(free, offsets):
    """Positions p such that p + o is inside the canvas and free for every offset o."""
    ok = np.ones_like(free)
    for dy, dx in offsets:
        ok &= _shift(free, dy, dx)
    return ok


def _components8(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask)
    comps = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, cells = [(int(y), int(x))], []
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for dy, dx in NEIGH8:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(sorted(cells))
    return comps


# --------------------------------------------------------------------------- analysis / solve
def _field_colors(g):
    """Field colors are the non-gray colors that occur as orthogonally adjacent equal pairs."""
    colors = set()
    horizontal = g[:, 1:] == g[:, :-1]
    vertical = g[1:, :] == g[:-1, :]
    colors.update(int(c) for c in np.unique(g[:, 1:][horizontal]))
    colors.update(int(c) for c in np.unique(g[1:, :][vertical]))
    colors.discard(GRAY)
    return colors


def _analyze(grid, strict=False):
    """Decompose an input into field, stencils, pins and copies using only the grid."""
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    field = _field_colors(g)
    fg = (g == GRAY) | ~np.isin(g, sorted(field))
    stencils, markers, problems = {}, [], []
    for comp in _components8(fg):
        gray = [c for c in comp if g[c] == GRAY]
        if gray:
            pins = [c for c in comp if g[c] != GRAY]
            if len(pins) != 1:
                problems.append("stencil without exactly one pin")
                continue
            pin = pins[0]
            color = int(g[pin])
            if color in stencils:
                problems.append("two stencils share a pin color")
                continue
            stencils[color] = {
                "pin": pin,
                "cells": comp,
                "carriers": [(y - pin[0], x - pin[1]) for y, x in gray],
            }
        elif len(comp) == 1:
            markers.append((comp[0], int(g[comp[0]])))
        else:
            problems.append("non-gray foreground cluster")
    copies, unmatched = [], []
    for (y, x), color in markers:
        if color not in stencils:
            unmatched.append(((y, x), color))
            continue
        cells = [(y + dy, x + dx) for dy, dx in stencils[color]["carriers"]]
        inside = [(cy, cx) for cy, cx in cells if 0 <= cy < height and 0 <= cx < width]
        if len(inside) != len(cells):
            problems.append("copy leaves the canvas")
        if any(fg[c] for c in inside):
            problems.append("copy covers foreground")
        copies.append({"pin": (y, x), "color": color, "carriers": inside,
                       "colors": [int(g[c]) for c in inside]})
    if strict and problems:
        raise AssertionError("; ".join(sorted(set(problems))))
    return {"grid": g, "field": field, "fg": fg, "stencils": stencils, "markers": markers,
            "copies": copies, "unmatched": unmatched, "problems": problems}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    g = info["grid"]
    out = np.zeros_like(g)
    for copy in info["copies"]:
        for cell in copy["carriers"]:
            out[cell] = g[cell]
        out[copy["pin"]] = copy["color"]
    return out


# --------------------------------------------------------------------------- field layouts
def _lengths(rng, total, parts, minimum):
    extra = total - parts * minimum
    cuts = np.sort(rng.integers(0, extra + 1, size=parts - 1))
    bounds = np.concatenate(([0], cuts, [extra]))
    return [int(v) + minimum for v in np.diff(bounds)]


def _region_map(rng, layout, height, width):
    labels = np.zeros((height, width), dtype=int)
    if layout == "uniform":
        return labels
    if layout in ("vstripes", "hstripes"):
        span = width if layout == "vstripes" else height
        count = int(rng.integers(2, min(5, span // 3) + 1))
        start = 0
        for index, size in enumerate(_lengths(rng, span, count, 3)):
            if layout == "vstripes":
                labels[:, start:start + size] = index
            else:
                labels[start:start + size, :] = index
            start += size
        return labels
    if layout == "quilt":
        rows = int(rng.integers(2, min(3, height // 4) + 1))
        cols = int(rng.integers(2, min(3, width // 4) + 1))
        ys = np.cumsum([0] + _lengths(rng, height, rows, 4))
        xs = np.cumsum([0] + _lengths(rng, width, cols, 4))
        for i in range(rows):
            for j in range(cols):
                labels[ys[i]:ys[i + 1], xs[j]:xs[j + 1]] = i * cols + j
        return labels
    if layout == "guillotine":
        leaves = [(0, 0, height, width)]
        target = int(rng.integers(3, 6))
        while len(leaves) < target:
            splittable = [i for i, (_, _, h, w) in enumerate(leaves) if h >= 8 or w >= 8]
            if not splittable:
                break
            y, x, h, w = leaves.pop(splittable[int(rng.integers(len(splittable)))])
            vertical = w >= 8 and (h < 8 or rng.random() < w / (h + w))
            if vertical:
                cut = int(rng.integers(4, w - 3))
                leaves += [(y, x, h, cut), (y, x + cut, h, w - cut)]
            else:
                cut = int(rng.integers(4, h - 3))
                leaves += [(y, x, cut, w), (y + cut, x, h - cut, w)]
        for index, (y, x, h, w) in enumerate(leaves):
            labels[y:y + h, x:x + w] = index
        return labels
    # inset: rectangles of other colors lying on a base color, never touching each other
    count = int(rng.integers(1, 4))
    taken = np.zeros((height, width), dtype=bool)
    placed = 0
    for _ in range(60):
        if placed == count:
            break
        h = int(rng.integers(3, max(4, min(9, height - 2))))
        w = int(rng.integers(3, max(4, min(9, width - 2))))
        if h > height - 2 or w > width - 2:
            continue
        y = int(rng.integers(0, height - h + 1))
        x = int(rng.integers(0, width - w + 1))
        if taken[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1].any():
            continue
        placed += 1
        taken[y:y + h, x:x + w] = True
        labels[y:y + h, x:x + w] = placed
    if placed == 0:
        y, x = (height - 3) // 2, (width - 3) // 2
        labels[y:y + 3, x:x + 3] = 1
    return labels


def _adjacent_labels(labels):
    pairs = set()
    for a, b in ((labels[:, 1:], labels[:, :-1]), (labels[1:, :], labels[:-1, :])):
        diff = a != b
        for p, q in zip(a[diff].tolist(), b[diff].tolist()):
            pairs.add((min(p, q), max(p, q)))
    return pairs


def _color_regions(rng, labels, palette):
    count = int(labels.max()) + 1
    neighbors = {i: set() for i in range(count)}
    for p, q in _adjacent_labels(labels):
        neighbors[p].add(q)
        neighbors[q].add(p)
    for _ in range(200):
        order = [int(i) for i in rng.permutation(count)]
        assigned = {}
        ok = True
        for region in order:
            banned = {assigned[n] for n in neighbors[region] if n in assigned}
            unused = [c for c in palette if c not in assigned.values() and c not in banned]
            allowed = unused or [c for c in palette if c not in banned]
            if not allowed:
                ok = False
                break
            assigned[region] = int(allowed[int(rng.integers(len(allowed)))])
        if ok and set(assigned.values()) == set(palette):
            lookup = np.array([assigned[i] for i in range(count)])
            return lookup[labels]
    return None


# --------------------------------------------------------------------------- stencils
def _normalize(points):
    y0 = min(y for y, _ in points)
    x0 = min(x for _, x in points)
    return sorted({(y - y0, x - x0) for y, x in points})


def _grow(rng, h, w, target, moves):
    points = {(int(rng.integers(h)), int(rng.integers(w)))}
    while len(points) < target:
        options = sorted({(y + dy, x + dx) for y, x in points for dy, dx in moves
                          if 0 <= y + dy < h and 0 <= x + dx < w} - points)
        if not options:
            break
        points.add(options[int(rng.integers(len(options)))])
    return points


STENCIL_WEIGHTS = np.array([0.16, 0.15, 0.16, 0.16, 0.19, 0.13, 0.05])


def _sample_stencil(rng, small=False, weights=None):
    weights = STENCIL_WEIGHTS if weights is None else np.asarray(weights, dtype=float)
    for _ in range(100):
        mode = int(rng.choice(len(weights), p=weights / weights.sum()))
        if mode == 0:  # filled block
            shapes = [(2, 2), (2, 3), (3, 2), (3, 3)] + ([] if small else [(2, 4), (4, 2), (3, 4), (4, 3)])
            h, w = shapes[int(rng.integers(len(shapes)))]
            points = {(y, x) for y in range(h) for x in range(w)}
        elif mode == 1:  # ring / outline with a window hole
            h, w = int(rng.integers(3, 5)), int(rng.integers(3, 4 if small else 6))
            points = {(y, x) for y in range(h) for x in range(w) if y in (0, h - 1) or x in (0, w - 1)}
            if rng.random() < 0.3:
                points.discard((int(rng.choice([0, h - 1])), int(rng.integers(1, w - 1))))
        elif mode == 2:  # diagonal-only growth
            h, w = int(rng.integers(3, 5 if small else 6)), int(rng.integers(3, 5 if small else 6))
            points = _grow(rng, h, w, int(rng.integers(3, 8)), ((1, 1), (1, -1), (-1, 1), (-1, -1)))
        elif mode == 3:  # T / cross / L junction
            h, w = int(rng.integers(3, 5 if small else 6)), int(rng.integers(3, 5 if small else 6))
            bar = int(rng.choice([0, h // 2, h - 1]))
            stem = int(rng.choice([0, w // 2, w - 1]))
            points = {(bar, x) for x in range(w)} | {(y, stem) for y in range(h)}
        elif mode == 4:  # irregular eight-neighbor growth
            h, w = int(rng.integers(2, 5)), int(rng.integers(3, 5 if small else 6))
            points = _grow(rng, h, w, int(rng.integers(3, min(11, h * w) + 1)), NEIGH8)
        elif mode == 6:  # tiny: the pin plus one or two gray cells (sparsest parent-B mask)
            points = {(0, 0), (int(rng.integers(0, 2)), 1)}
            if rng.random() < 0.5:
                points.add((int(rng.integers(0, 2)), 2) if rng.random() < 0.5 else (1, 0))
        else:  # narrow bar with bumps
            length = int(rng.integers(3, 6 if small else 7))
            points = {(1, x) for x in range(length)}
            for x in range(length):
                if rng.random() < 0.45:
                    points.add((int(rng.choice([0, 2])), x))
            if rng.random() < 0.5:
                points = {(x, y) for y, x in points}
        points = _normalize(points)
        flat = len({y for y, _ in points}) < 2 or len({x for _, x in points}) < 2
        if mode != 6 and (len(points) < 3 or flat):
            continue
        if len(_components8(_points_mask(points))) != 1:
            continue
        tips = [p for p in points if all((p[0] + dy, p[1] + dx) not in points for dy, dx in NEIGH4)]
        roll = rng.random()
        if tips and roll < 0.35:
            pin = tips[int(rng.integers(len(tips)))]
        elif mode in (0, 1) and roll < 0.6:
            corners = [p for p in points if p[0] in (0, max(y for y, _ in points))
                       and p[1] in (0, max(x for _, x in points))] or points
            pin = corners[int(rng.integers(len(corners)))]
        else:
            pin = points[int(rng.integers(len(points)))]
        return {"points": points, "pin": pin,
                "offsets": [(y - pin[0], x - pin[1]) for y, x in points],
                "carriers": [(y - pin[0], x - pin[1]) for y, x in points if (y, x) != pin]}
    raise RuntimeError("stencil sampling failed")


def _points_mask(points):
    h = max(y for y, _ in points) + 1
    w = max(x for _, x in points) + 1
    mask = np.zeros((h, w), dtype=bool)
    for p in points:
        mask[p] = True
    return mask


# --------------------------------------------------------------------------- pair builder
def _footprint_stats(field, offsets):
    """Number of distinct field colors under each candidate copy's gray cells (indexed by pin)."""
    height, width = field.shape
    padded = np.full((len(offsets), height, width), -1, dtype=int)
    for i, (dy, dx) in enumerate(offsets):
        y0, y1 = max(0, -dy), min(height, height - dy)
        x0, x1 = max(0, -dx), min(width, width - dx)
        if y0 < y1 and x0 < x1:
            padded[i, y0:y1, x0:x1] = field[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    ordered = np.sort(padded, axis=0)
    return 1 + (np.diff(ordered, axis=0) != 0).sum(axis=0)


def _choose(rng, mask):
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return None
    y, x = cells[int(rng.integers(len(cells)))]
    return int(y), int(x)


def _build_pair(rng, spec):
    height, width = spec["size"]
    layout = spec["layout"]
    labels = _region_map(rng, layout, height, width)
    regions = int(labels.max()) + 1
    n_pins = len(spec["mults"])
    pool = [int(c) for c in rng.permutation(COLORS)]
    pins = []
    for color in spec.get("prefer_pins", []):
        if color in pool and len(pins) < n_pins:
            pins.append(color)
            pool.remove(color)
    while len(pins) < n_pins:
        pins.append(pool.pop(0))
    n_decoy_colors = 0 if spec["decoys"] == 0 else min(spec["decoys"], int(rng.integers(1, 3)))
    decoy_colors = [pool.pop(0) for _ in range(n_decoy_colors)]
    room = len(pool)
    if regions == 1:
        n_field = 1
    else:
        n_field = min(regions, room, int(rng.choice([2, 3, 4], p=[0.4, 0.4, 0.2])))
        n_field = max(2, n_field, min(regions, room, spec["straddle"]))
    want = [c for c in spec.get("prefer_field", []) if c in pool]
    palette = want[:1] + [c for c in pool if c not in want[:1]]
    palette = palette[:n_field]
    if len(palette) < n_field:
        return None
    field = _color_regions(rng, labels, palette)
    if field is None:
        return None
    stencils = [_sample_stencil(rng, spec.get("small", False), spec.get("weights")) for _ in range(n_pins)]
    grid = field.copy()
    solid = np.zeros((height, width), dtype=bool)    # source stencil cells
    points = np.zeros((height, width), dtype=bool)   # isolated pins and decoys
    reserved = np.zeros((height, width), dtype=bool)  # copy footprints (must stay field)
    for stencil, color in zip(stencils, pins):
        # Sources keep two background cells from each other for a readable decomposition.
        pos = _choose(rng, _placements(~_dilate(_dilate(solid)), stencil["offsets"]))
        if pos is None:
            return None
        for dy, dx in stencil["offsets"]:
            cell = (pos[0] + dy, pos[1] + dx)
            grid[cell] = GRAY
            solid[cell] = True
        grid[pos] = color
    point_ok = lambda: ~_dilate(_dilate(solid)) & ~_dilate(points) & ~reserved
    order = [i for i, m in enumerate(spec["mults"]) for _ in range(m)]
    first = order[0] if order else None
    order = order[1:]
    rng.shuffle(order)
    order = ([first] if first is not None else []) + order
    copies = []
    border = np.pad(np.zeros((height - 2, width - 2), dtype=bool), 1, constant_values=True)
    for step, index in enumerate(order):
        stencil = stencils[index]
        ok = point_ok() & _placements(~solid & ~points & ~reserved, stencil["carriers"])
        if not ok.any():
            return None
        need = spec["straddle"] if step == 0 else 0
        roll = rng.random()
        if need or roll < 0.3:
            spread = _footprint_stats(field, stencil["carriers"])
            better = ok & (spread >= max(2, need))
            if need and not better.any():
                return None
            if better.any():
                ok = better
        elif roll < 0.5:
            edge = np.zeros_like(ok)
            for dy, dx in stencil["offsets"]:
                edge |= _shift(border, dy, dx)
            if (ok & edge).any():
                ok = ok & edge
        elif roll < 0.65 and reserved.any():
            touch = ok & ~_placements(~_dilate(reserved), stencil["offsets"])
            if touch.any():
                ok = touch
        pos = _choose(rng, ok)
        grid[pos] = pins[index]
        points[pos] = True
        for dy, dx in stencil["offsets"]:
            reserved[pos[0] + dy, pos[1] + dx] = True
        copies.append((index, pos))
    for k in range(spec["decoys"]):
        pos = _choose(rng, point_ok())
        if pos is None:
            return None
        grid[pos] = decoy_colors[0] if k < 2 or len(decoy_colors) == 1 else decoy_colors[k % len(decoy_colors)]
        points[pos] = True
    out = np.zeros_like(grid)
    for index, pos in copies:
        for dy, dx in stencils[index]["carriers"]:
            cell = (pos[0] + dy, pos[1] + dx)
            out[cell] = field[cell]
        out[pos] = pins[index]
    info = _analyze(grid)
    if info["problems"] or info["field"] != set(palette) or not np.array_equal(solve(grid), out):
        return None
    return {"input": grid.astype(int), "output": out.astype(int),
            "meta": {"layout": layout, "pins": pins, "field": palette, "decoys": decoy_colors,
                     "mults": list(spec["mults"])}}


def _make_pair(rng, spec):
    spec = dict(spec)
    for attempt in range(60):
        pair = _build_pair(rng, spec)
        if pair is not None:
            return pair
        if attempt % 12 == 11:
            h, w = spec["size"]
            spec["size"] = (min(30, h + 2), min(30, w + 2))
    return None


# --------------------------------------------------------------------------- shortcuts
def _mode(values):
    values = list(values)
    return max(sorted(set(values)), key=values.count)


def _shortcut_outputs(grid):
    """Outputs of tempting partial rules, used to prove that the demonstrations exclude them."""
    info = _analyze(grid)
    g = info["grid"]
    field_cells = [int(v) for v in g[~info["fg"]]]
    global_color = _mode(field_cells)
    source_color = {}
    for color, stencil in info["stencils"].items():
        ring = np.zeros_like(info["fg"])
        for cell in stencil["cells"]:
            ring[cell] = True
        ring = _dilate(ring) & ~info["fg"]
        source_color[color] = _mode(int(v) for v in g[ring]) if ring.any() else global_color
    results = {}
    for name in ("uniform_copy", "source_color", "global_color", "gray_copy", "keep_decoys"):
        out = np.zeros_like(g)
        for copy in info["copies"]:
            if name == "uniform_copy":
                paint = [_mode(copy["colors"])] * len(copy["colors"])
            elif name == "source_color":
                paint = [source_color[copy["color"]]] * len(copy["colors"])
            elif name == "global_color":
                paint = [global_color] * len(copy["colors"])
            elif name == "gray_copy":
                paint = [GRAY] * len(copy["colors"])
            else:
                paint = copy["colors"]
            for cell, value in zip(copy["carriers"], paint):
                out[cell] = value
            out[copy["pin"]] = copy["color"]
        if name == "keep_decoys":
            for cell, color in info["unmatched"]:
                out[cell] = color
        results[name] = out
    # Parent-B-only reading: recolor gray in place, nothing moves.
    in_place = np.zeros_like(g)
    for color, stencil in info["stencils"].items():
        for cell in stencil["cells"]:
            in_place[cell] = source_color[color] if g[cell] == GRAY else color
    results["no_stamping"] = in_place
    # Parent-A-only reading: stamp gray copies, keep the field as background.
    keep = g.copy()
    keep[info["fg"]] = global_color
    for copy in info["copies"]:
        for cell in copy["carriers"]:
            keep[cell] = GRAY
        keep[copy["pin"]] = copy["color"]
    results["no_reveal"] = keep
    return results


# --------------------------------------------------------------------------- puzzle
def _pair_features(pair):
    info = _analyze(pair["input"])
    spreads = [len(set(c["colors"])) for c in info["copies"]]
    counts = {}
    for c in info["copies"]:
        counts[c["color"]] = counts.get(c["color"], 0) + 1
    return {"info": info, "spread": max(spreads) if spreads else 0, "stencils": len(info["stencils"]),
            "copies": len(info["copies"]), "max_mult": max(counts.values()) if counts else 0,
            "decoys": len(info["unmatched"]), "fields": len(info["field"])}


def _sizes(rng, low, high):
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _mults(rng, n_stencils, cap, total_cap):
    mults = [int(rng.choice([1, 2, 3, 4], p=[0.40, 0.35, 0.18, 0.07])) for _ in range(n_stencils)]
    mults = [min(m, cap) for m in mults]
    while sum(mults) > total_cap:
        i = int(np.argmax(mults))
        mults[i] -= 1
    return mults


def _build_puzzle(rng, num_train, num_test):
    compact = bool(rng.random() < 0.25)
    shift = str(rng.choice(SHIFTS, p=[0.12, 0.2, 0.2, 0.16, 0.16, 0.16]))
    held = str(rng.choice(["quilt", "guillotine", "inset", "vstripes", "hstripes"]))
    if compact:
        base_layouts = ["uniform", "vstripes", "hstripes", str(rng.choice(["quilt", "inset", "guillotine"]))]
    else:
        base_layouts = list(LAYOUTS)
    train_layouts = [l for l in base_layouts if not (shift == "layout" and l == held)]
    patch_train = [l for l in train_layouts if l != "uniform"]
    low, high = (12, 17) if compact else (10, 22)
    if shift == "scale":
        low, high = (10, 16) if not compact else (11, 15)
    max_stencils = 2 if (compact or shift == "dense") else 3
    # Puzzle-level stencil vocabulary preference (filled, ring, diagonal, junction, irregular, bar, tiny).
    weights = 0.5 * STENCIL_WEIGHTS + 0.5 * rng.dirichlet(np.full(len(STENCIL_WEIGHTS), 1.2))
    if compact:
        weights[6] = 0.0
    train = []
    for index in range(num_train):
        if index == 0:
            layout = str(rng.choice(patch_train))
            corner = shift != "corner" and layout in ("quilt", "guillotine") and rng.random() < 0.3
            spec = {"layout": layout, "mults": [2, 1] if rng.random() < 0.6 else [2, 2],
                    "decoys": int(rng.integers(1, 4)), "straddle": 3 if corner else 2}
            rng.shuffle(spec["mults"])
        else:
            # At most one literal parent-B pair (single companion field) per training set.
            uniform_ok = index == 1 and "uniform" in train_layouts and rng.random() < 0.45
            layout = "uniform" if uniform_ok else str(rng.choice(patch_train))
            n = int(rng.integers(1, max_stencils + 1))
            spec = {"layout": layout, "mults": _mults(rng, n, 3 if compact else 4, 5 if shift == "dense" else 7),
                    "decoys": int(rng.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15])),
                    "straddle": 2 if (layout != "uniform" and rng.random() < 0.35) else 0}
        spec["size"] = _sizes(rng, low, high)
        spec["small"] = compact
        spec["weights"] = weights
        pair = _make_pair(rng, spec)
        if pair is None:
            return None
        train.append(pair)
    feats = [_pair_features(p) for p in train]
    if shift == "corner" and any(f["spread"] >= 3 for f in feats):
        return None  # the three-color reveal is reserved for the test in corner-shift puzzles
    train_max_mult = max(f["max_mult"] for f in feats)
    train_max_copies = max(f["copies"] for f in feats)
    train_field = sorted({c for p in train for c in p["meta"]["field"]})
    train_pins = sorted({c for p in train for c in p["meta"]["pins"]})
    train_area = max(p["input"].size for p in train)
    test = []
    for index in range(num_test):
        if shift == "layout":
            layout = held if index == 0 or rng.random() < 0.5 else str(rng.choice(patch_train))
        elif shift == "corner":
            layout = str(rng.choice(["quilt", "guillotine"])) if index == 0 else str(rng.choice(PATCH_LAYOUTS))
        elif index == 0:
            layout = str(rng.choice(patch_train))
        else:
            layout = str(rng.choice(train_layouts if rng.random() < 0.85 else ["uniform"]))
        if shift == "dense":
            n = 3
            mults = [min(train_max_mult, 2)] * 3
            while sum(mults) <= train_max_copies and min(mults) < train_max_mult:
                mults[int(np.argmin(mults))] += 1
        else:
            n = int(rng.integers(1, max_stencils + 1))
            mults = _mults(rng, n, train_max_mult, 7)
        spec = {"layout": layout, "mults": mults,
                "decoys": int(rng.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.25, 0.15])),
                "straddle": (3 if shift == "corner" and index == 0 else 2) if layout != "uniform" and (
                    index == 0 or rng.random() < 0.4) else 0,
                "small": compact, "weights": weights}
        if shift == "scale":
            spec["size"] = (int(rng.integers(21, 31)), int(rng.integers(21, 31)))
        elif shift == "dense":
            spec["size"] = _sizes(rng, max(low, 16), min(30, high + 4))
        else:
            spec["size"] = _sizes(rng, low, high)
        if shift == "roles":
            spec["prefer_pins"] = [int(c) for c in rng.permutation(train_field)]
            spec["prefer_field"] = [int(c) for c in rng.permutation(train_pins)]
        pair = _make_pair(rng, spec)
        if pair is None:
            return None
        test.append(pair)
    if shift == "scale" and min(p["input"].size for p in test) <= train_area:
        return None
    if shift == "roles" and not any(set(p["meta"]["pins"]) & set(train_field)
                                    and set(p["meta"]["field"]) & set(train_pins) for p in test):
        return None
    if shift == "corner" and not any(_pair_features(p)["spread"] >= 3 for p in test):
        return None
    # The role-establishing pair is not always first; list position carries no meaning.
    train = [train[int(i)] for i in rng.permutation(num_train)]
    test = [test[int(i)] for i in rng.permutation(num_test)]
    puzzle = {"train": [{"input": p["input"], "output": p["output"]} for p in train],
              "test": [{"input": p["input"], "output": p["output"]} for p in test]}
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return None
    plan = {"compact": compact, "shift": shift, "held": held if shift == "layout" else None,
            "layouts": [p["meta"]["layout"] for p in train + test]}
    return puzzle, plan


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        built = _build_puzzle(rng, num_train, num_test)
        if built is not None:
            return built[0]
    raise RuntimeError(f"could not build a puzzle for seed {seed}")


# --------------------------------------------------------------------------- validation
def _has_gap(cells):
    ys = [y for y, _ in cells]
    xs = [x for _, x in cells]
    return (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1) > len(cells)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    feats = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            assert inp.shape == out.shape
            key = (inp.shape, inp.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert not (inp == BLACK).any(), "inputs never use black"
            assert not (out == GRAY).any(), "outputs never keep gray"
            info = _analyze(inp, strict=True)
            assert info["stencils"], "each input needs a stencil"
            assert info["field"], "each input needs a field"
            assert set(info["stencils"]).isdisjoint(info["field"])
            matched = {c["color"] for c in info["copies"]}
            assert matched == set(info["stencils"]), "every stencil needs a matching pin"
            footprint = np.zeros(inp.shape, dtype=int)
            for c in info["copies"]:
                for cell in c["carriers"] + [c["pin"]]:
                    footprint[cell] += 1
                assert all(v in info["field"] for v in c["colors"]), "copies reveal field cells"
            assert footprint.max() <= 1, "copies may touch but never overlap"
            assert np.array_equal(solve(inp), out), "output must follow the rule"
            feats[split].append(_pair_features({"input": inp}))
    # Demonstrations must expose both parents and rule out the tempting partial rules.
    tr = feats["train"]
    assert any(f["spread"] >= 2 for f in tr), "training must show a copy straddling field colors"
    assert any(f["decoys"] for f in tr), "training must show unmatched pins vanishing"
    assert any(f["max_mult"] >= 2 for f in tr), "training must show one-to-many stamping"
    assert any(f["stencils"] >= 2 for f in tr), "training must show several stencils"
    assert sum(f["fields"] >= 2 for f in tr) >= min(2, len(tr) - 1), "patchwork must be shown repeatedly"
    assert any(_has_gap(st["cells"]) for f in tr for st in f["info"]["stencils"].values()), \
        "training must show non-gray cells inside a stencil's bounding box staying black"
    refuted = set()
    for pair in train:
        out = np.asarray(pair["output"])
        refuted.update(name for name, guess in _shortcut_outputs(np.asarray(pair["input"])).items()
                       if not np.array_equal(guess, out))
    for name in SHORTCUTS:
        assert name in refuted, f"training does not exclude shortcut {name}"
    assert max(f["max_mult"] for f in feats["test"]) <= max(f["max_mult"] for f in tr)
    assert any(f["spread"] >= 2 for f in feats["test"]), "a test copy must straddle field colors"


# --------------------------------------------------------------------------- self test
def _generate_with_plan(seed, num_train=4, num_test=1):
    """Same draws as generate_puzzle, also returning the puzzle-level plan (diagnostics only)."""
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        built = _build_puzzle(rng, num_train, num_test)
        if built is not None:
            return built
    raise RuntimeError(f"could not build a puzzle for seed {seed}")


def _same(p, q):
    return all(len(p[s]) == len(q[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(p[s], q[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    # Count handling: boundary legal counts and illegal requests.
    for a, b in ((2, 1), (10, 4), (5, 2), (3, 2)):
        p = generate_puzzle(7, a, b)
        validate_generated_puzzle(p)
        assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (3.0, 1), (True, 1)):
        try:
            generate_puzzle(3, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {a}/{b}")
    shapes, signatures, layouts, shifts = set(), set(), set(), set()
    stencil_counts, mults, spreads, decoys, stencil_sizes = set(), set(), set(), set(), set()
    role_switch = uniform_pairs = edge_copies = touching = compact = 0
    for seed in range(42):
        p, plan = _generate_with_plan(seed)
        assert _same(p, generate_puzzle(seed)), "plan replay must match generate_puzzle"
        assert _same(p, generate_puzzle(seed)), "nondeterministic"
        validate_generated_puzzle(p)
        shifts.add(plan["shift"])
        layouts.update(plan["layouts"])
        compact += int(plan["compact"])
        signatures.add(b"".join(np.asarray(x["input"]).tobytes() for x in p["train"]))
        feats = {s: [_pair_features(x) for x in p[s]] for s in ("train", "test")}
        tr, te = feats["train"], feats["test"][0]
        # Shift-specific promises, measured from grids (at least one test pair carries the shift).
        tests = feats["test"]
        n_train = len(p["train"])
        if plan["shift"] == "layout":
            assert any(l not in plan["layouts"][:n_train] for l in plan["layouts"][n_train:])
        elif plan["shift"] == "corner":
            assert any(f["spread"] >= 3 for f in tests) and max(f["spread"] for f in tr) == 2
        elif plan["shift"] == "scale":
            assert min(f["info"]["grid"].size for f in tests) > max(f["info"]["grid"].size for f in tr)
        elif plan["shift"] == "dense":
            assert min(f["stencils"] for f in tests) > max(f["stencils"] for f in tr)
        elif plan["shift"] == "roles":
            pins = {c for f in tr for c in f["info"]["stencils"]}
            fields = {c for f in tr for c in f["info"]["field"]}
            assert any(set(f["info"]["stencils"]) & fields and set(f["info"]["field"]) & pins for f in tests)
        # Omitting either parent concept changes at least one demonstrated answer and the test.
        test_out = np.asarray(p["test"][0]["output"])
        shortcuts = _shortcut_outputs(np.asarray(p["test"][0]["input"]))
        for name in ("no_stamping", "no_reveal", "uniform_copy", "gray_copy"):
            assert not np.array_equal(shortcuts[name], test_out), name
        pins_by_pair, fields_by_pair = [], []
        for f in tr + feats["test"]:
            info = f["info"]
            h, w = info["grid"].shape
            shapes.add((h, w))
            stencil_counts.add(f["stencils"])
            mults.add(f["max_mult"])
            spreads.add(f["spread"])
            decoys.add(f["decoys"])
            uniform_pairs += int(f["fields"] == 1)
            stencil_sizes.update(len(v["carriers"]) + 1 for v in info["stencils"].values())
            pins_by_pair.append(set(info["stencils"]))
            fields_by_pair.append(set(info["field"]))
            cells = [c for cp in info["copies"] for c in cp["carriers"] + [cp["pin"]]]
            edge_copies += int(any(y in (0, h - 1) or x in (0, w - 1) for y, x in cells))
            mask = np.zeros((h, w), dtype=bool)
            hit = False
            for cp in info["copies"]:
                m = np.zeros((h, w), dtype=bool)
                for c in cp["carriers"] + [cp["pin"]]:
                    m[c] = True
                hit |= bool((_dilate(m) & mask).any())
                mask |= m
            touching += int(hit)
        role_switch += int(any(a & b for a in pins_by_pair for b in fields_by_pair))
    assert len(signatures) == 42
    assert len(shapes) >= 60, len(shapes)
    assert shifts == set(SHIFTS) and layouts == set(LAYOUTS)
    assert 4 <= compact <= 20
    assert {1, 2, 3} <= stencil_counts and {1, 2, 3, 4} <= mults and {1, 2, 3} <= spreads
    assert 0 in decoys and max(decoys) >= 3
    assert min(stencil_sizes) <= 3 and max(stencil_sizes) >= 12
    assert role_switch >= 15 and uniform_pairs >= 8 and edge_copies >= 60 and touching >= 40
