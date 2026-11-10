"""Negative-space printing strips. Standard library and NumPy only.

Generation uses cavity coordinates to make labels. Solving rediscovers the key,
background regions, ownership, crop and count from the supplied input alone.
No files, parent generators, caches or generation metadata are used by solve.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= v <= 30 for v in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "color range")
    return a


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    group.append(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _bounds(cells):
    rr, cc = zip(*cells)
    return min(rr), max(rr) + 1, min(cc), max(cc) + 1


def _crop(mask):
    r0, r1, c0, c1 = _bounds(np.argwhere(mask))
    return mask[r0:r1, c0:c1].copy()


def _key(grid, bg):
    """The unique dense multicolor component on exactly one canvas edge."""
    h, w = grid.shape
    candidates = []
    for cells in _components(grid != bg, diagonal=True):
        r0, r1, c0, c1 = _bounds(cells)
        sh, sw = r1 - r0, c1 - c0
        if len(cells) != sh * sw or len({int(grid[q]) for q in cells}) < 2:
            continue
        edges = [r0 == 0, c0 == 0, r1 == h, c1 == w]
        if sum(edges) != 1:
            continue
        side = edges.index(True)  # top, left, bottom, right
        if side in (0, 2) and sh == 2 and sw >= 2:
            outer = grid[r0 if side == 0 else r1 - 1, c0:c1]
            inner = grid[r0 + 1 if side == 0 else r1 - 2, c0:c1]
        elif side in (1, 3) and sw == 2 and sh >= 2:
            outer = grid[r0:r1, c0 if side == 1 else c1 - 1]
            inner = grid[r0:r1, c0 + 1 if side == 1 else c1 - 2]
        else:
            continue
        sources = tuple(map(int, outer))
        if len(set(sources)) == len(sources):
            candidates.append({"side": side, "sources": sources,
                               "fills": tuple(map(int, inner)),
                               "mapping": dict(zip(sources, map(int, inner))),
                               "cells": frozenset(cells)})
    _require(len(candidates) == 1, "one unambiguous local edge key required")
    return candidates[0]


def _analyze(input_grid):
    g = _grid(input_grid)
    colors, frequencies = np.unique(g, return_counts=True)
    _require(np.count_nonzero(frequencies == frequencies.max()) == 1,
             "background must be the unique modal color")
    bg = int(colors[np.argmax(frequencies)])
    key = _key(g, bg)
    objects, ids = [], np.full(g.shape, -1, dtype=int)
    for cells in _components(g != bg, diagonal=True):
        if frozenset(cells) == key["cells"]:
            continue
        ink = {int(g[q]) for q in cells}
        _require(len(ink) == 1, "contours must be monochrome and separated")
        index = len(objects)
        for q in cells:
            ids[q] = index
        objects.append({"color": next(iter(ink)), "cells": cells, "holes": []})
    regions, active = [], []
    h, w = g.shape
    for cells in _components(g == bg):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells):
            continue
        boundary = set()
        for r, c in cells:
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if g[q] != bg:
                    boundary.add(q)
        owners = {int(ids[q]) for q in boundary}
        _require(len(owners) == 1 and -1 not in owners, "ambiguous cavity ownership")
        owner = next(iter(owners))
        color = objects[owner]["color"]
        region = {"cells": cells, "owner": owner, "source": color,
                  "fill": key["mapping"].get(color)}
        regions.append(region)
        objects[owner]["holes"].append(region)
        if region["fill"] is not None:
            active.append(region)
    _require(bool(active), "at least one listed enclosed cavity required")
    occupied = [q for region in active for q in region["cells"]]
    r0, r1, c0, c1 = _bounds(occupied)
    tile = np.full((r1 - r0, c1 - c0), bg, dtype=np.int8)
    for region in active:
        for r, c in region["cells"]:
            tile[r - r0, c - c0] = region["fill"]
    axis = 0 if key["side"] in (0, 2) else 1
    n = len(active)
    _require(n * (tile.shape[axis] + 1) - 1 <= 30, "print strip exceeds 30")
    return {"background": bg, "key": key, "objects": objects,
            "regions": regions, "active": active, "tile": tile,
            "count": n, "axis": axis, "bounds": (r0, r1, c0, c1)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Extract only listed cavities; repeat their joint crop along the key normal."""
    d = _analyze(input_grid)
    t, n, axis = d["tile"], d["count"], d["axis"]
    shape = list(t.shape)
    shape[axis] = n * (shape[axis] + 1) - 1
    out = np.full(shape, d["background"], dtype=np.int8)
    period = t.shape[axis] + 1
    # Modular sampling deliberately differs from label construction's concat.
    for r in range(shape[0]):
        for c in range(shape[1]):
            q = r if axis == 0 else c
            if q % period < period - 1:
                out[r, c] = t[r % period, c] if axis == 0 else t[r, c % period]
    return out


def _hole(rng, h, w):
    """Connected, row-convex negative space, including simple rectangles."""
    a = np.ones((h, w), dtype=bool)
    if min(h, w) > 1 and rng.random() < .7:
        left, right = 0, w - 1
        for r in range(1, h):
            anchor = int(rng.integers(left, right + 1))
            left = int(rng.integers(anchor + 1))
            right = int(rng.integers(anchor, w))
            a[r] = False
            a[r, left:right + 1] = True
        if a.all():
            a[-1, -1] = False
    if rng.random() < .5:
        a = a[::-1].copy()
    if rng.random() < .5:
        a = a[:, ::-1].copy()
    return _crop(a)


def _dilate(mask):
    a = mask.copy()
    a[1:] |= mask[:-1]
    a[:-1] |= mask[1:]
    a[:, 1:] |= mask[:, :-1]
    a[:, :-1] |= mask[:, 1:]
    return a


def _object(rng, holes, color, axial_limit, thickness=1, diamond=False):
    """Carve distinct cavities into one wall; optionally use diagonal corners."""
    m = len(holes)
    options = []
    for columns in range(1, m + 1):
        rows = [holes[i:i + columns] for i in range(0, m, columns)]
        height = sum(max(a.shape[0] for a in row) for row in rows) + len(rows) - 1
        width = max(sum(a.shape[1] for a in row) + len(row) - 1 for row in rows)
        # A horizontal shelf always fits and bounds total width across objects.
        horizontal_width = sum(a.shape[1] for a in holes) + m - 1
        if height <= axial_limit and width <= horizontal_width:
            options.append((rows, height, width))
    rows, height, width = options[int(rng.integers(len(options)))]
    spare = int(rng.integers(min(2, axial_limit - height) + 1))
    height += spare
    void = np.zeros((height + 2 * thickness, width + 2 * thickness), dtype=bool)
    placed, row_start = [], thickness
    # Put spare axial space above/below or between silhouettes, not in the rule.
    row_start += int(rng.integers(spare + 1)) if len(rows) == 1 else 0
    for row in rows:
        rh = max(a.shape[0] for a in row)
        row_width = sum(a.shape[1] for a in row) + len(row) - 1
        col_start = thickness + int(rng.integers(width - row_width + 1))
        for a in row:
            rr = row_start + int(rng.integers(rh - a.shape[0] + 1))
            cells = [(rr + int(r), col_start + int(c)) for r, c in np.argwhere(a)]
            for q in cells:
                void[q] = True
            placed.append(cells)
            col_start += a.shape[1] + 1
        row_start += rh + 1
    if m == 1 and thickness == 1 and (diamond or rng.random() < .45):
        wall = _dilate(void) & ~void
    else:
        wall = ~void
        # Clip corners only where no cavity has a four-neighbor dependency.
        needed = _dilate(void)
        for r, c in ((0, 0), (0, wall.shape[1] - 1),
                     (wall.shape[0] - 1, 0), (wall.shape[0] - 1, wall.shape[1] - 1)):
            if not needed[r, c] and rng.random() < .65:
                wall[r, c] = False
    # Trim optional empty margins while keeping the cavity coordinates exact.
    r0, r1, c0, c1 = _bounds(np.argwhere(wall))
    placed = [[(r - r0, c - c0) for r, c in cells] for cells in placed]
    wall = wall[r0:r1, c0:c1].copy()
    _require(len(_components(wall, True)) == 1, "fragmented contour construction")
    enclosed = [set(cells) for cells in _components(~wall)
                if not any(r in (0, wall.shape[0] - 1) or c in (0, wall.shape[1] - 1)
                           for r, c in cells)]
    _require({frozenset(c) for c in enclosed} == {frozenset(c) for c in placed},
             "constructed cavity mismatch")
    return {"wall": wall, "holes": placed, "color": color,
            "hole_bounds": _bounds([q for cells in placed for q in cells])}


def _open_object(rng, color):
    h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
    wall = np.ones((h, w), dtype=bool)
    wall[1:-1, 1:-1] = False
    wall[0, int(rng.integers(1, w - 1))] = False
    wall = np.rot90(wall, int(rng.integers(4))).copy()
    return {"wall": wall, "holes": [], "color": color}


def _group_indices(rng, n):
    groups, current = [], []
    for i in range(n):
        current.append(i)
        if i == n - 1 or rng.random() < .52:
            groups.append(current)
            current = []
    return groups


def _make_pair(rng, n, turns, lesson=None, palette=None):
    axial_limit = 31 // n - 1
    key_length = int(rng.integers(3 if lesson is not None else 2, 6))
    sources = list(range(1, key_length + 1))
    fills = list(map(int, rng.choice(np.arange(1, 9), key_length)))
    if lesson is not None:
        fills[:3] = [2, 6, 3]  # one lookup, chain, and self-map are visible.
    mapping = dict(zip(sources, fills))
    if lesson == 0:
        a = np.array([[1, 0], [1, 1]], dtype=bool)
        holes = [np.rot90(a, int(rng.integers(4))).copy() for _ in range(2)]
        groups, inks = [[0], [1]], [1, 3]
    elif lesson == 1:
        a = np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)
        a = np.rot90(a, int(rng.integers(4))).copy()
        holes = [a, np.ones((1, 1), dtype=bool), np.ones((1, 1), dtype=bool)]
        groups, inks = [[0, 1], [2]], [1, 2]
    else:
        max_width = max(1, min(7, (26 - 3 * n) // n))
        holes = [_hole(rng, int(rng.integers(1, min(8, axial_limit) + 1)),
                       int(rng.integers(1, max_width + 1))) for _ in range(n)]
        groups = _group_indices(rng, n)
        inks = list(map(int, rng.choice(sources, len(groups))))
    # Allocate wall thickness from the *whole* cross-axis packing budget.
    horizontal_base = sum(a.shape[1] for a in holes) + n + 2 * len(groups) - 1
    extra_budget = max(0, (28 - horizontal_base) // 2)
    active_objects = []
    for i, (group, ink) in enumerate(zip(groups, inks)):
        thick = 1
        if lesson is None and extra_budget and rng.random() < .28:
            thick = 2
            extra_budget -= 1
        active_objects.append(_object(rng, [holes[j] for j in group], ink, axial_limit,
                                      thick, diamond=lesson == 1 and i == 1))
    spans = [o["hole_bounds"][1] - o["hole_bounds"][0] for o in active_objects]
    top_wall = max(o["hole_bounds"][0] for o in active_objects)
    bottom_wall = max(o["wall"].shape[0] - o["hole_bounds"][1] for o in active_objects)
    band_height = min(axial_limit, max(spans) + int(rng.integers(4)),
                      20 - top_wall - bottom_wall)
    _require(band_height >= max(spans), "joint wall/cavity canvas budget")
    band_start = min(max(int(rng.integers(5, 8)), top_wall + 3),
                     23 - band_height - bottom_wall)
    sum_width = sum(o["wall"].shape[1] for o in active_objects)
    min_width = max(12, sum_width + len(active_objects) - 1)
    _require(min_width <= 30, "cross-axis packing budget")
    w = int(rng.integers(min_width, 31))
    active_bottom = band_start + band_height + bottom_wall
    min_height = max(17, active_bottom + 7)
    _require(min_height <= 30, "axial canvas packing budget")
    h = int(rng.integers(min_height, 31))
    g = np.zeros((h, w), dtype=np.int8)
    order = list(map(int, rng.permutation(key_length)))
    key_start = int(rng.integers(1, w - key_length))
    g[0, key_start:key_start + key_length] = [sources[i] for i in order]
    g[1, key_start:key_start + key_length] = [fills[i] for i in order]
    blocked = np.zeros(g.shape, dtype=bool)
    blocked[:3] = True
    placements = []

    def reserve(obj, r, c):
        sh, sw = obj["wall"].shape
        _require(0 <= r <= h - sh and 0 <= c <= w - sw, "object outside canvas")
        _require(not blocked[r:r + sh, c:c + sw].any(), "object/key separation failure")
        g[r:r + sh, c:c + sw][obj["wall"]] = obj["color"]
        blocked[max(0, r - 1):min(h, r + sh + 1),
                max(0, c - 1):min(w, c + sw + 1)] = True

    # Random 2D placement inside the printable axial band. Retry budget is fixed;
    # a variable-width shelf fallback fits by the cross-axis budget above.
    for attempt in range(12):
        placements = []
        trial = blocked.copy()
        for obj in active_objects:
            sh, sw = obj["wall"].shape
            r0, r1, _, _ = obj["hole_bounds"]
            possible = [(r, c)
                        for r in range(band_start - r0,
                                       band_start + band_height - r1 + 1)
                        for c in range(w - sw + 1)
                        if r >= 3 and r + sh <= h and not trial[r:r + sh, c:c + sw].any()]
            if not possible:
                break
            r, c = possible[int(rng.integers(len(possible)))]
            placements.append((r, c))
            trial[max(0, r - 1):min(h, r + sh + 1),
                  max(0, c - 1):min(w, c + sw + 1)] = True
        if len(placements) == len(active_objects):
            break
    else:
        slack = w - sum_width - len(active_objects) + 1
        c = int(rng.integers(slack + 1))
        placements = []
        for obj in active_objects:
            r0, r1, _, _ = obj["hole_bounds"]
            r = band_start - r0 + int(rng.integers(band_height - (r1 - r0) + 1))
            placements.append((r, c))
            c += obj["wall"].shape[1] + 1
    known = []
    for obj, (r, c) in zip(active_objects, placements):
        reserve(obj, r, c)
        known += [(rr + r, cc + c, mapping[obj["color"]])
                  for cells in obj["holes"] for rr, cc in cells]

    # Listed open shapes and unlisted closed shapes are meaningful controls.
    # Their placement is independent of the printable stencil's crop.
    extras = []
    if lesson is not None or rng.random() < .7:
        inert = 6 if lesson == 0 else 9 if lesson == 1 else int(rng.choice([c for c in range(1, 10) if c not in sources]))
        extra_hole = _hole(rng, int(rng.integers(1, 4)), int(rng.integers(1, 4)))
        extras.append(_object(rng, [extra_hole], inert, 5, diamond=rng.random() < .4))
    if lesson is not None or rng.random() < .6:
        extras.append(_open_object(rng, int(rng.choice(sources))))
    # A seven-row clear zone below all active walls guarantees these two controls
    # fit. Flexible placement elsewhere is preferred when it is available.
    for obj in extras:
        sh, sw = obj["wall"].shape
        possible = [(r, c) for r in range(3, h - sh + 1) for c in range(w - sw + 1)
                    if not blocked[r:r + sh, c:c + sw].any()]
        _require(bool(possible), "control contour packing failed")
        # Keep the first control out of the guaranteed bottom shelf unless that
        # shelf can still fit the other one horizontally.
        if len(extras) == 2 and obj is extras[0]:
            possible = [(r, c) for r, c in possible if r + sh < active_bottom or c == 0] or possible
        r, c = possible[int(rng.integers(len(possible)))]
        reserve(obj, r, c)

    rr, cc, _ = zip(*known)
    r0, r1, c0, c1 = min(rr), max(rr) + 1, min(cc), max(cc) + 1
    tile = np.zeros((r1 - r0, c1 - c0), dtype=np.int8)
    for r, c, color in known:
        tile[r - r0, c - c0] = color
    gap = np.zeros((1, tile.shape[1]), dtype=np.int8)
    pieces = [a for i in range(n) for a in ((tile,) if i == n - 1 else (tile, gap))]
    output = np.concatenate(pieces, axis=0)
    # Thick, sparse multi-hole walls can otherwise outnumber background in a
    # compact canvas. Add only blank space, preserving key edge and all geometry.
    frequencies = np.bincount(g.ravel(), minlength=10)
    deficit = max(0, int(frequencies[1:].max() - frequencies[0] + 1))
    if deficit:
        add_cols = min(30 - g.shape[1], (deficit + g.shape[0] - 1) // g.shape[0])
        g = np.pad(g, ((0, 0), (0, add_cols)), constant_values=0)
        deficit = max(0, deficit - add_cols * g.shape[0])
        add_rows = min(30 - g.shape[0], (deficit + g.shape[1] - 1) // g.shape[1])
        g = np.pad(g, ((0, add_rows), (0, 0)), constant_values=0)
    _require(np.count_nonzero(g == 0) > max(np.count_nonzero(g == c) for c in range(1, 10)),
             "constructed background is not uniquely modal")
    if palette is None:
        palette = rng.permutation(10)
    return {"input": np.rot90(palette[g], turns).astype(np.int8, copy=True),
            "output": np.rot90(palette[output], turns).astype(np.int8, copy=True)}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(f"{name} outside {lo}..{hi}")
    return int(value)


def _signature(g):
    a = np.asarray(g)
    return (a.shape, a.astype(np.int8, copy=False).tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    first_turn = int(rng.integers(4))
    second_turn = (first_turn + int(rng.choice([1, 3]))) % 4
    familiar = [first_turn, second_turn]
    palette0 = rng.permutation(10)
    palette1 = rng.permutation(10)
    if palette1[0] == palette0[0]:
        palette1[[0, 1]] = palette1[[1, 0]]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, amount in (("train", nt), ("test", nq)):
        for index in range(amount):
            lesson = index if split == "train" and index < 2 else None
            n = 2 + index if lesson is not None else int(rng.integers(1, 4) if split == "train" else rng.integers(4, 7))
            turns = familiar[index] if lesson is not None else int(rng.integers(4))
            palette = palette0 if lesson == 0 else palette1 if lesson == 1 else None
            for _ in range(12):
                pair = _make_pair(rng, n, turns, lesson, palette)
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded uniqueness attempts exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "list splits required")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, train, test = set(), [], []
    for split, analyses in (("train", train), ("test", test)):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_signature(inp) not in seen, "duplicate input")
            seen.add(_signature(inp))
            d = _analyze(inp)
            _require(np.array_equal(solve(inp), out), "incorrect print strip")
            n, axis, tile, bg = d["count"], d["axis"], d["tile"], d["background"]
            for k in range(n):
                start = k * (tile.shape[axis] + 1)
                sl = [slice(None), slice(None)]
                sl[axis] = slice(start, start + tile.shape[axis])
                _require(np.array_equal(out[tuple(sl)], tile), "tile changed between copies")
                if k < n - 1:
                    sl[axis] = start + tile.shape[axis]
                    _require(np.all(out[tuple(sl)] == bg), "separator must be one background line")
            analyses.append(d)
    # These are evidence requirements for this generator's curriculum, rather
    # than extra rules needed by solve on a single well-formed family member.
    areas = [sum(len(r["cells"]) for r in d["active"]) for d in train]
    _require(any(a["count"] != b["count"] and areas[i] == areas[j]
                 for i, a in enumerate(train) for j, b in enumerate(train)),
             "training must distinguish cavity count from area")
    _require({d["axis"] for d in train} == {0, 1}, "both key-normal axes need demonstration")
    _require(any(len(o["holes"]) > 1 and o["color"] in d["key"]["mapping"]
                 for d in train for o in d["objects"]), "multi-cavity counting witness missing")
    _require(any(not o["holes"] and o["color"] in d["key"]["mapping"]
                 for d in train for o in d["objects"]), "listed open-contour witness missing")
    _require(any(o["holes"] and o["color"] not in d["key"]["mapping"]
                 for d in train for o in d["objects"]), "unlisted closed-contour witness missing")
    _require(any(len(r["cells"]) != (_bounds(r["cells"])[1] - _bounds(r["cells"])[0]) *
                 (_bounds(r["cells"])[3] - _bounds(r["cells"])[2])
                 for d in train for r in d["active"]), "nonrectangular negative space missing")
    _require(any(r["fill"] in d["key"]["mapping"] and
                 d["key"]["mapping"][r["fill"]] != r["fill"]
                 for d in train for r in d["active"]), "one-step lookup witness missing")
    _require(any(r["source"] == r["fill"] for d in train for r in d["active"]),
             "self-map witness missing")
    _require(min(d["count"] for d in test) > max(d["count"] for d in train),
             "generated tests must extrapolate cavity count")


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for p in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(p[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded, streaming 42-seed audit; returns compact descriptive metrics."""
    shapes, output_shapes, structures, hole_shapes = set(), set(), set(), set()
    sides, counts, backgrounds, multiplicities, signatures = set(), set(), set(), set(), set()
    pairs = diagonal_witnesses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        if seed in (0, 17, 41):
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterminism")
        diagonal_in_train = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pairs += 1
                g, out = pair["input"], pair["output"]
                d = _analyze(g)
                shapes.add(g.shape)
                output_shapes.add(out.shape)
                sides.add(d["key"]["side"])
                counts.add(d["count"])
                backgrounds.add(d["background"])
                roles, normalized = {}, []
                for c in g.flat:
                    normalized.append(roles.setdefault(int(c), len(roles)))
                structures.add((g.shape, bytes(normalized)))
                for region in d["active"]:
                    r0, r1, c0, c1 = _bounds(region["cells"])
                    mask = np.zeros((r1 - r0, c1 - c0), dtype=np.int8)
                    for r, c in region["cells"]:
                        mask[r - r0, c - c0] = 1
                    hole_shapes.add(_signature(mask))
                multiplicities.update(len(o["holes"]) for o in d["objects"])
                if split == "train":
                    enclosed8 = {q for cells in _components(g == d["background"], True)
                                 if not any(r in (0, g.shape[0] - 1) or c in (0, g.shape[1] - 1)
                                            for r, c in cells) for q in cells}
                    diagonal_in_train |= any(q not in enclosed8 for r in d["active"] for q in r["cells"])
        _require(diagonal_in_train, "four-neighbor enclosure not demonstrated")
        diagonal_witnesses += diagonal_in_train
        # Omitting A's repetition, or B's selection, is observably wrong.
        d = _analyze(puzzle["train"][0]["input"])
        if d["count"] == 1:
            d = next(_analyze(p["input"]) for p in puzzle["train"]
                     if _analyze(p["input"])["count"] > 1)
        _require(d["count"] > 1, "omitting repetition would be invisible")
        d = next(_analyze(p["input"]) for p in puzzle["train"]
                 if any(o["holes"] and o["color"] not in _analyze(p["input"])["key"]["mapping"]
                        for o in _analyze(p["input"])["objects"]))
        _require(len(d["regions"]) > d["count"], "omitting keyed selection would be invisible")
    _require(len(signatures) == 42, "duplicate puzzles")
    _require(len(shapes) >= 40 and len(output_shapes) >= 50, "geometry diversity collapsed")
    _require(len(structures) >= 200 and len(hole_shapes) >= 35, "structure diversity collapsed")
    _require(sides == {0, 1, 2, 3} and counts == set(range(1, 7)), "orientation/count coverage")
    _require(backgrounds == set(range(10)) and max(multiplicities) >= 3, "palette/topology coverage")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(12345 + a * 10 + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "ignored explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0)
    # Input-only invariance: removing an inert contour cannot change the print.
    for pair in p["train"]:
        g = pair["input"].copy()
        d = _analyze(g)
        inert = next((o for o in d["objects"] if o["holes"] and
                      o["color"] not in d["key"]["mapping"]), None)
        if inert is not None:
            for q in inert["cells"]:
                g[q] = d["background"]
            _require(np.array_equal(solve(g), pair["output"]), "inert-object dependence")
            break
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"].flat[0] = (int(broken["test"][0]["output"].flat[0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(signatures),
            "input_shapes": len(shapes), "output_shapes": len(output_shapes),
            "color_normalized_inputs": len(structures), "cavity_shapes": len(hole_shapes),
            "key_sides": sorted(sides), "cavity_counts": sorted(counts),
            "backgrounds": sorted(backgrounds), "holes_per_object": sorted(multiplicities),
            "puzzles_with_diagonal_training_witness": diagonal_witnesses}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
