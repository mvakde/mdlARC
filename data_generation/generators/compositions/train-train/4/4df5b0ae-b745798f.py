"""Corner-routed size packing.

Each input holds monochrome L-triomino "compass" markers (one per used color,
distinct orientations) and ordinary monochrome pieces.  A marker's vertex
corner (parent B: orientation -> canvas corner) names the canvas corner that
receives every piece of the marker's color.  Those pieces are sorted by pixel
count and rigidly packed as adjacent bounding boxes along the horizontal edge
of that corner, smallest at the corner (parent A: size-sorted bottom-left
packing, generalised to all four corners).  Markers and an optional enclosing
frame disappear; everything else becomes background.
"""
from __future__ import annotations

import json

import numpy as np

CORNERS = ("TL", "TR", "BL", "BR")
# missing cell (r, c) of the 2x2 box -> vertex corner (opposite cell)
_MISSING_TO_CORNER = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}
_CORNER_TO_MISSING = {v: k for k, v in _MISSING_TO_CORNER.items()}

_SOURCE_INPUTS = {
    # Parent original inputs (A: 4df5b0ae, B: b745798f) must never be reproduced.
    "2222222222/2777777882/2770777772/2677777772/2677777772/2677557772/2777557772/2777777772/2777777772/2222222222",
    "88228888855/88288888885/88888888888/88188888885/88118888855/88888888888/88888888888/88888888888/88888888888/88888888888/88888888888",
    "88808/88800/22888/82744/87748",
}


# --------------------------------------------------------------------------
# shared analysis (used by solve and validation)
# --------------------------------------------------------------------------

def _most_common(values):
    vals, counts = np.unique(np.asarray(values).ravel(), return_counts=True)
    best = counts.max()
    return int(vals[counts == best].min())


def _frame_and_background(grid):
    """Return (frame_color or None, background, interior slice bounds)."""
    h, w = grid.shape
    if h >= 5 and w >= 5:
        per = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
        if np.all(per == per[0]):
            bg = _most_common(grid[1:-1, 1:-1])
            if bg != int(per[0]):
                return int(per[0]), bg, (1, h - 1, 1, w - 1)
    return None, _most_common(grid), (0, h, 0, w)


def _components(grid, bg, bounds):
    r0, r1, c0, c1 = bounds
    seen = np.zeros(grid.shape, dtype=bool)
    comps = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if seen[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if r0 <= ny < r1 and c0 <= nx < c1 and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            top, left = min(ys), min(xs)
            hh, ww = max(ys) - top + 1, max(xs) - left + 1
            mask = np.zeros((hh, ww), dtype=bool)
            for y, x in cells:
                mask[y - top, x - left] = True
            comps.append({"color": color, "cells": sorted(cells), "top": top, "left": left,
                          "mask": mask, "size": len(cells)})
    comps.sort(key=lambda d: (d["top"], d["left"]))
    return comps


def _marker_corner(mask):
    if mask.shape != (2, 2) or int(mask.sum()) != 3:
        return None
    missing = tuple(int(v) for v in np.argwhere(~mask)[0])
    return _MISSING_TO_CORNER[missing]


def _marker_mask(corner):
    m = np.ones((2, 2), dtype=bool)
    m[_CORNER_TO_MISSING[corner]] = False
    return m


def _pack_positions(masks, corner, height, width):
    """Top-left positions for a corner's row: masks[0] is the marker, the rest are
    pieces already in packing order.  Boxes are edge-aligned and separated by one
    background column, running away from the corner along the horizontal edge."""
    pos = []
    x = 0 if corner in ("TL", "BL") else width
    for m in masks:
        hh, ww = m.shape
        top = 0 if corner in ("TL", "TR") else height - hh
        if corner in ("TL", "BL"):
            pos.append((top, x))
            x += ww + 1
        else:
            x -= ww
            pos.append((top, x))
            x -= 1
    return pos


def _corner_row(info, color, corner, key):
    group = sorted((p for p in info["pieces"] if p["color"] == color), key=key)
    return [_marker_mask(corner)] + [p["mask"] for p in group]


def analyse(grid):
    grid = np.asarray(grid, dtype=int)
    frame, bg, bounds = _frame_and_background(grid)
    comps = _components(grid, bg, bounds)
    markers, pieces = {}, []
    for comp in comps:
        corner = _marker_corner(comp["mask"])
        if corner is not None:
            if comp["color"] in markers:
                raise ValueError("two markers share a color")
            markers[comp["color"]] = {"corner": corner, **comp}
        else:
            pieces.append(comp)
    return {"frame": frame, "bg": bg, "bounds": bounds, "markers": markers, "pieces": pieces}


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    info = analyse(grid)
    h, w = grid.shape
    out = np.full((h, w), info["bg"], dtype=int)
    for color, marker in info["markers"].items():
        masks = _corner_row(info, color, marker["corner"], lambda p: (p["size"], p["top"], p["left"]))
        for m, (top, left) in zip(masks, _pack_positions(masks, marker["corner"], h, w)):
            if top < 0 or left < 0 or top + m.shape[0] > h or left + m.shape[1] > w:
                continue  # never happens for generated inputs; keeps solve total
            region = out[top:top + m.shape[0], left:left + m.shape[1]]
            region[m] = color
    return out


# --------------------------------------------------------------------------
# shape sampling
# --------------------------------------------------------------------------

def _trim(mask):
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def _is_l3(mask):
    return _marker_corner(mask) is not None


def _grow(rng, size, max_dim, stringy):
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = []
        for y, x in cells:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                p = (y + dy, x + dx)
                if p in cells:
                    continue
                ys = [q[0] for q in cells] + [p[0]]
                xs = [q[1] for q in cells] + [p[1]]
                if max(ys) - min(ys) + 1 > max_dim or max(xs) - min(xs) + 1 > max_dim:
                    continue
                nb = sum((p[0] + a, p[1] + b) in cells for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))
                frontier.append((p, nb))
        if not frontier:
            return None
        weights = np.array([(1.0 / nb ** 2) if stringy else float(nb ** 2) for _, nb in frontier])
        p = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))][0]
        cells.add(p)
    ys = [q[0] for q in cells]
    xs = [q[1] for q in cells]
    mask = np.zeros((max(ys) - min(ys) + 1, max(xs) - min(xs) + 1), dtype=bool)
    for y, x in cells:
        mask[y - min(ys), x - min(xs)] = True
    return mask


def _shape(rng, kind, size_hint=None):
    for _ in range(50):
        if kind == "rect":
            hh, ww = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            m = np.ones((hh, ww), dtype=bool)
        elif kind == "bar":
            n = int(rng.integers(1, 7))
            m = np.ones((1, n) if rng.random() < 0.5 else (n, 1), dtype=bool)
        elif kind == "ell":
            a, b = int(rng.integers(3, 6)), int(rng.integers(2, 5))
            m = np.zeros((a, b), dtype=bool)
            m[:, 0] = True
            m[-1, :] = True
            m = np.rot90(m, int(rng.integers(4)))
        elif kind == "sparse":
            size = size_hint if size_hint else int(rng.integers(4, 9))
            m = _grow(rng, size, 6, True)
        else:
            size = size_hint if size_hint else int(rng.integers(4, 12))
            m = _grow(rng, size, 5, False)
        if m is None:
            continue
        m = _trim(m)
        if not _is_l3(m):
            return m
    return np.ones((1, 2), dtype=bool)


def _random_shape(rng):
    kind = rng.choice(["rect", "bar", "ell", "sparse", "blob", "blob"])
    return _shape(rng, str(kind))


def _contrast_pair(rng):
    """A sparse piece with the larger bounding box but fewer pixels than a solid one."""
    for _ in range(200):
        sparse = _shape(rng, "sparse", int(rng.integers(4, 8)))
        hh, ww = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        solid = np.ones((hh, ww), dtype=bool)
        if solid.sum() > sparse.sum() and solid.size < sparse.size:
            return [sparse, solid]
    return [np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool), np.ones((2, 3), dtype=bool)]


# --------------------------------------------------------------------------
# pair construction
# --------------------------------------------------------------------------

def _dilate(mask):
    h, w = mask.shape
    out = np.zeros((h + 2, w + 2), dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out[dy:dy + h, dx:dx + w] |= mask
    return out


def _place_inputs(rng, items, h, w, bounds, gap):
    """items: list of (mask, color). Returns grid of colors (-1 background) or None."""
    r0, r1, c0, c1 = bounds
    canvas = np.full((h, w), -1, dtype=int)
    order = sorted(range(len(items)), key=lambda i: -items[i][0].sum())
    for i in order:
        mask, color = items[i]
        mh, mw = mask.shape
        if mh > r1 - r0 or mw > c1 - c0:
            return None
        placed = False
        for _ in range(300):
            top = int(rng.integers(r0, r1 - mh + 1))
            left = int(rng.integers(c0, c1 - mw + 1))
            region = canvas[top:top + mh, left:left + mw]
            if np.any(region[mask] >= 0):
                continue
            # neighbourhood check (8-neighbour ring)
            padded = np.pad(canvas, 1, constant_values=-1)
            near = padded[top:top + mh + 2, left:left + mw + 2][_dilate(mask)]
            if np.any(near == color):
                continue
            if gap and np.any(near >= 0):
                continue
            region[mask] = color
            placed = True
            break
        if not placed:
            return None
    return canvas


def _make_pair(rng, spec):
    """spec: corners (list), multi (set of corners needing >=2 pieces), contrast corner or None,
    bg, frame flag, size range, gap flag, max pieces per group."""
    corners = spec["corners"]
    palette = [c for c in range(10) if c != spec["bg"]]
    colors = [int(c) for c in rng.choice(palette, size=len(corners), replace=False)]
    frame_color = None
    if spec["frame"]:
        rest = [c for c in palette if c not in colors]
        frame_color = int(rng.choice(rest))
    groups = []
    for corner, color in zip(corners, colors):
        lo = 2 if corner in spec["multi"] else 1
        n = int(rng.integers(lo, spec["max_pieces"] + 1))
        masks = _contrast_pair(rng) if corner == spec.get("contrast") else []
        tries = 0
        while len(masks) < n and tries < 200:
            tries += 1
            m = _random_shape(rng)
            if all(int(m.sum()) != int(o.sum()) for o in masks):
                masks.append(m)
        masks.sort(key=lambda m: int(m.sum()))
        groups.append({"corner": corner, "color": color, "masks": masks})
    # output footprint requirements
    def row_w(g):
        return 2 + sum(m.shape[1] + 1 for m in g["masks"])
    top_w = sum(row_w(g) for g in groups if g["corner"][0] == "T")
    bot_w = sum(row_w(g) for g in groups if g["corner"][0] == "B")
    max_h = max(m.shape[0] for g in groups for m in g["masks"])
    total = sum(int(m.sum()) for g in groups for m in g["masks"]) + 3 * len(groups)
    lo_h, hi_h = spec["size"]
    need_w = max(top_w, bot_w, lo_h)
    if need_w > 30:
        return None
    w = int(rng.integers(need_w, min(30, max(need_w, hi_h) + 3) + 1))
    min_h = max(lo_h, 2 * max_h + 2, int(np.ceil(total * 2.6 / w)))
    if min_h > 30:
        return None
    h = int(rng.integers(min_h, min(30, max(min_h, hi_h)) + 1))
    border = 1 if frame_color is not None else 0
    bounds = (border, h - border, border, w - border)
    items = []
    for g in groups:
        items.append((_marker_mask(g["corner"]), g["color"]))
        items.extend((m, g["color"]) for m in g["masks"])
    canvas = _place_inputs(rng, items, h, w, bounds, spec["gap"])
    if canvas is None:
        return None
    grid = np.where(canvas >= 0, canvas, spec["bg"])
    if frame_color is not None:
        grid[0, :] = grid[-1, :] = frame_color
        grid[:, 0] = grid[:, -1] = frame_color
    out = solve(grid)
    if not _pair_ok(grid, out, groups, frame_color, spec["bg"]):
        return None
    return {"input": grid.astype(int), "output": out.astype(int)}


def _bbox_overlap(a, b):
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def _pair_ok(grid, out, groups, frame_color, bg):
    info = analyse(grid)
    if info["bg"] != bg or info["frame"] != frame_color:
        return False
    if {c: m["corner"] for c, m in info["markers"].items()} != {g["color"]: g["corner"] for g in groups}:
        return False
    if len(info["pieces"]) != sum(len(g["masks"]) for g in groups):
        return False
    r0, r1, c0, c1 = info["bounds"]
    interior = grid[r0:r1, c0:c1]
    if np.sum(interior == bg) * 2 <= interior.size:
        return False
    return _output_boxes_disjoint(info, grid.shape)


def _output_boxes_disjoint(info, shape):
    h, w = shape
    boxes = []
    for color, marker in info["markers"].items():
        masks = _corner_row(info, color, marker["corner"], lambda p: p["size"])
        for m, (t, l) in zip(masks, _pack_positions(masks, marker["corner"], h, w)):
            if t < 0 or l < 0 or t + m.shape[0] > h or l + m.shape[1] > w:
                return False
            boxes.append((color, (t, l, m.shape[0], m.shape[1])))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            t, l, hh, ww = boxes[i][1]
            grown = (t - 1, l - 1, hh + 2, ww + 2)  # different corners keep a clear gap
            if boxes[i][0] != boxes[j][0] and _bbox_overlap(grown, boxes[j][1]):
                return False
    return True


# --------------------------------------------------------------------------
# puzzle construction
# --------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, v in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("need 2 <= num_train <= 10, 1 <= num_test <= 4, num_train > num_test")


def _train_plans(rng, num_train):
    """Distribute all four corners over training pairs, each as a multi-piece group."""
    order = list(rng.permutation(CORNERS))
    plans = [set() for _ in range(num_train)]
    multi = [set() for _ in range(num_train)]
    # at least two pairs carry the mandatory coverage when possible
    carriers = min(num_train, int(rng.integers(2, 4)))
    for i, corner in enumerate(order):
        k = i % carriers
        plans[k].add(str(corner))
        multi[k].add(str(corner))
    for k in range(num_train):
        if not plans[k] or rng.random() < 0.35:
            extra = int(rng.integers(1, 3))
            for corner in rng.permutation(CORNERS)[:extra]:
                plans[k].add(str(corner))
    perm = rng.permutation(num_train)
    plans = [plans[i] for i in perm]
    multi = [multi[i] for i in perm]
    return plans, multi


def _sort_corners(corners):
    return [c for c in CORNERS if c in corners]


def _grid_key(grid):
    return "/".join("".join(str(int(v)) for v in row) for row in np.asarray(grid))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _attempt in range(60):
        puzzle = _attempt_puzzle(rng, num_train, num_test)
        if puzzle is None:
            continue
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError(f"could not generate puzzle for seed {seed}")


def _attempt_puzzle(rng, num_train, num_test):
    bg_mode = "fixed" if rng.random() < 0.55 else "vary"
    base_bg = int(rng.integers(10))
    frame_mode = rng.choice(["none", "none", "train", "both"])
    gap_rate = float(rng.choice([0.0, 0.4, 0.8]))
    shift = str(rng.choice(["groups", "scale", "pieces", "mixed"]))
    plans, multis = _train_plans(rng, num_train)
    contrast_pair = int(rng.integers(num_train))
    frame_train = set()
    if frame_mode != "none":
        frame_train = {int(i) for i in rng.choice(num_train, size=int(rng.integers(1, max(2, num_train // 2) + 1)),
                                                  replace=False)}
    pairs, seen = [], set(_SOURCE_INPUTS)
    for i in range(num_train + num_test):
        is_test = i >= num_train
        bg = base_bg if bg_mode == "fixed" else int(rng.integers(10))
        if is_test:
            if shift in ("groups", "mixed"):
                k = int(rng.choice([3, 4, 4]))
            else:
                k = int(rng.integers(1, 5))
            corners = set(str(c) for c in rng.permutation(CORNERS)[:k])
            if corners == {"BL"}:
                corners.add(str(rng.choice(["TL", "TR", "BR"])))
            multi = {str(rng.choice(sorted(corners)))}
            size = (12, 22) if shift in ("scale", "mixed") else (8, 16)
            max_pieces = 4 if shift in ("pieces", "mixed") else 3
            frame = frame_mode == "both" and rng.random() < 0.6
        else:
            corners, multi = plans[i], multis[i]
            size = (8, 15)
            max_pieces = 3
            frame = i in frame_train
        contrast = None
        if not is_test and i == contrast_pair:
            contrast = str(rng.choice(sorted(multi))) if multi else str(rng.choice(sorted(corners)))
            multi = set(multi) | {contrast}
        spec = {"corners": _sort_corners(corners), "multi": set(multi), "contrast": contrast, "bg": bg,
                "frame": bool(frame), "size": size, "gap": bool(rng.random() < gap_rate),
                "max_pieces": max_pieces}
        pair = None
        for _ in range(40):
            pair = _make_pair(rng, spec)
            if pair is not None and _grid_key(pair["input"]) not in seen:
                break
            pair = None
        if pair is None:
            return None
        seen.add(_grid_key(pair["input"]))
        pairs.append(pair)
    return {"train": pairs[:num_train], "test": pairs[num_train:]}


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def _order_inputs(group):
    return [p["size"] for p in sorted(group, key=lambda p: (p["left"], p["top"]))]


def _alt_bl_only(grid):
    """Shortcut ignoring parent B: pack every piece bottom-left by size, drop markers."""
    info = analyse(grid)
    h, w = grid.shape
    out = np.full((h, w), info["bg"], dtype=int)
    pieces = sorted(info["pieces"], key=lambda p: p["size"])
    masks = [p["mask"] for p in pieces]
    for p, m, (t, l) in zip(pieces, masks, _pack_positions(masks, "BL", h, w)):
        if l + m.shape[1] > w:
            break
        out[t:t + m.shape[0], l:l + m.shape[1]][m] = p["color"]
    return out


def _alt_bbox_order(grid):
    """Shortcut ignoring parent A's pixel-count ranking: sort by bounding-box area."""
    info = analyse(grid)
    h, w = grid.shape
    out = np.full((h, w), info["bg"], dtype=int)
    for color, marker in info["markers"].items():
        masks = _corner_row(info, color, marker["corner"], lambda p: (p["mask"].size, p["top"], p["left"]))
        for m, (t, l) in zip(masks, _pack_positions(masks, marker["corner"], h, w)):
            if 0 <= l and l + m.shape[1] <= w:
                out[t:t + m.shape[0], l:l + m.shape[1]][m] = color
    return out


def _alt_nearest_corner(grid):
    """Shortcut ignoring marker orientation: send each color to the marker's nearest corner."""
    info = analyse(grid)
    h, w = grid.shape
    out = np.full((h, w), info["bg"], dtype=int)
    for color, marker in info["markers"].items():
        corner = ("T" if marker["top"] + 1 < h / 2 else "B") + ("L" if marker["left"] + 1 < w / 2 else "R")
        masks = _corner_row(info, color, corner, lambda p: p["size"])
        for m, (t, l) in zip(masks, _pack_positions(masks, corner, h, w)):
            if t < 0 or l < 0 or l + m.shape[1] > w:
                continue
            out[t:t + m.shape[0], l:l + m.shape[1]][m] = color
    return out


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    keys = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a = np.asarray(pair["input"])
            b = np.asarray(pair["output"])
            for g in (a, b):
                assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
                assert np.issubdtype(g.dtype, np.integer) and g.min() >= 0 and g.max() <= 9
            assert a.shape == b.shape
            key = _grid_key(a)
            assert key not in keys and key not in _SOURCE_INPUTS, "duplicate or source input"
            keys.add(key)
            info = analyse(a)
            assert info["markers"], "no marker"
            corners = [m["corner"] for m in info["markers"].values()]
            assert len(set(corners)) == len(corners), "marker orientations repeat within a pair"
            for p in info["pieces"]:
                assert p["color"] in info["markers"], "piece without a marker"
                assert _marker_corner(p["mask"]) is None
            for color in info["markers"]:
                sizes = [p["size"] for p in info["pieces"] if p["color"] == color]
                assert sizes, "marker without pieces"
                assert len(set(sizes)) == len(sizes), "tied sizes inside a group"
            if info["frame"] is not None:
                assert info["frame"] not in info["markers"]
            assert _output_boxes_disjoint(info, a.shape), "groups collide in output"
            assert np.array_equal(solve(a), b), "output disagrees with rule"
            assert not np.array_equal(a, b)
            infos[split].append((a, b, info))
    # demonstration requirements
    multi_corners = set()
    contrast = False
    multi_group_pair = False
    location_break = False
    for a, b, info in infos["train"]:
        if len(info["markers"]) >= 2:
            multi_group_pair = True
        h, w = a.shape
        for color, m in info["markers"].items():
            group = [p for p in info["pieces"] if p["color"] == color]
            if len(group) >= 2:
                multi_corners.add(m["corner"])
                by_px = [id(p) for p in sorted(group, key=lambda p: p["size"])]
                by_box = [id(p) for p in sorted(group, key=lambda p: (p["mask"].size, p["size"]))]
                boxes = [p["mask"].size for p in group]
                if by_px != by_box and len(set(boxes)) == len(boxes):
                    contrast = True
            quad = ("T" if m["top"] + 1 < h / 2 else "B") + ("L" if m["left"] + 1 < w / 2 else "R")
            if quad != m["corner"]:
                location_break = True
    assert multi_corners == set(CORNERS), "training must show every corner with >=2 pieces"
    assert contrast, "training must separate pixel count from bounding-box area"
    assert multi_group_pair, "training must route at least two colors in one pair"
    assert location_break, "training must show a marker away from its corner"
    if any(info["frame"] is not None for _, _, info in infos["test"]):
        assert any(info["frame"] is not None for _, _, info in infos["train"]), "test frame undemonstrated"
    # shortcuts must fail somewhere in the puzzle
    for alt in (_alt_bl_only, _alt_nearest_corner):
        assert any(not np.array_equal(alt(a), b) for split in infos for a, b, _ in infos[split])
    assert any(not np.array_equal(_alt_bl_only(a), b) for a, b, _ in infos["test"]), "test solvable by A alone"


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------

SELF_TEST_SEEDS = list(range(42))


def self_test() -> None:
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    signatures = set()
    shapes = set()
    corner_sets = set()
    frames = bgs = 0
    test_bl_fail = test_nearest_fail = test_bbox_fail = 0
    for seed in SELF_TEST_SEEDS:
        p = generate_puzzle(seed)
        q = generate_puzzle(seed)
        sig = json.dumps({k: [{r: np.asarray(v).tolist() for r, v in pr.items()} for pr in p[k]] for k in p})
        assert sig == json.dumps({k: [{r: np.asarray(v).tolist() for r, v in pr.items()} for pr in q[k]] for k in q})
        signatures.add(sig)
        validate_generated_puzzle(p)
        for pair in p["train"] + p["test"]:
            a = np.asarray(pair["input"])
            info = analyse(a)
            shapes.add(a.shape)
            corner_sets.add(tuple(sorted(m["corner"] for m in info["markers"].values())))
            frames += info["frame"] is not None
        bgs += len({analyse(np.asarray(pr["input"]))["bg"] for pr in p["train"] + p["test"]}) > 1
        for pair in p["test"]:
            a, b = np.asarray(pair["input"]), np.asarray(pair["output"])
            test_bl_fail += not np.array_equal(_alt_bl_only(a), b)
            test_nearest_fail += not np.array_equal(_alt_nearest_corner(a), b)
            test_bbox_fail += not np.array_equal(_alt_bbox_order(a), b)
    assert len(signatures) == len(SELF_TEST_SEEDS)
    assert len(shapes) >= 30 and len(corner_sets) >= 10
    assert frames >= 5 and bgs >= 5
    assert test_bl_fail == len(SELF_TEST_SEEDS) and test_nearest_fail >= 20
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7, nt, ns)
            assert len(p["train"]) == nt and len(p["test"]) == ns


if __name__ == "__main__":
    self_test()
    print("self_test passed")
