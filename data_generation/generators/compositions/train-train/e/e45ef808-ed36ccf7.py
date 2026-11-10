"""Turn, drop, then survey.

A rigid block floating in the air above a bottom-supported terrain makes one
counterclockwise quarter-turn inside its own square bounding box (parent B),
falls straight down until it rests on the landscape, and becomes part of the
skyline.  The final skyline is then surveyed as in parent A: the air above the
tallest surface column (rightmost on a tie) is filled with the peak guide (4)
and the air above the unique shortest surface column with the valley guide (9).
"""
from __future__ import annotations

from collections import deque

import numpy as np

PEAK_GUIDE = 4
VALLEY_GUIDE = 9
ROLE_POOL = (0, 1, 2, 3, 5, 6, 7, 8)
STYLES = ("smooth", "terraces", "ridge", "notches", "flat", "jagged")
SHIFTS = ("larger_piece", "piece_tie", "valley_fill", "open")
TRANSFORMS = ("ccw", "cw", "half", "identity", "transpose", "antitranspose",
              "mirror", "flip")
SHORTCUTS = TRANSFORMS[1:] + ("terrain_only", "leftmost_tie", "no_drop",
                              "static", "drop_then_rotate")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# Rule (shared by solve, validation and generation)
# ---------------------------------------------------------------------------

def _transform(box, name):
    if name == "ccw":
        return np.rot90(box, 1)
    if name == "cw":
        return np.rot90(box, -1)
    if name == "half":
        return np.rot90(box, 2)
    if name == "identity":
        return box.copy()
    if name == "transpose":
        return box.T.copy()
    if name == "antitranspose":
        return np.rot90(box, 2).T.copy()
    if name == "mirror":
        return box[:, ::-1].copy()
    if name == "flip":
        return box[::-1, :].copy()
    raise ValueError(name)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue, cells = deque([(r, c)]), []
            while queue:
                y, x = queue.popleft()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            result.append(sorted(cells))
    return result


def _parse(grid):
    """Read roles, terrain stacks and floating pieces from an input grid."""
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    g = g.astype(int)
    h, w = g.shape
    if h < 6 or w < 3:
        raise ValueError("grid too small for this family")
    border, air, terrain = int(g[0, 0]), int(g[1, 0]), int(g[h - 1, 0])
    if not (np.all(g[0] == border) and np.all(g[1] == air) and np.all(g[h - 1] == terrain)):
        raise ValueError("top border, first air row or bottom terrain row is not uniform")
    if len({border, air, terrain}) != 3:
        raise ValueError("border, air and terrain must be distinct")
    if np.any(g[1:] == border):
        raise ValueError("border colour appears below the border")
    surf = np.zeros(w, dtype=int)
    for c in range(w):
        rows = np.flatnonzero(g[:, c] == terrain)
        top = int(rows[0])
        if not np.all(g[top:, c] == terrain):
            raise ValueError("terrain stack is not bottom supported")
        surf[c] = top
    mask = np.zeros(g.shape, dtype=bool)
    mask[1:] = (g[1:] != air) & (g[1:] != terrain)
    pieces = []
    for cells in _components(mask):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        k = r1 - r0 + 1
        if c1 - c0 + 1 != k:
            raise ValueError("a floating piece must have a square bounding box")
        if r1 >= int(surf[c0:c1 + 1].min()):
            raise ValueError("a piece must float above the terrain")
        box = np.full((k, k), -1, dtype=int)
        for r, c in cells:
            box[r - r0, c - c0] = g[r, c]
        pieces.append({"r0": r0, "c0": c0, "k": k, "box": box, "cells": cells})
    pieces.sort(key=lambda p: p["c0"])
    for left, right in zip(pieces, pieces[1:]):
        if right["c0"] <= left["c0"] + left["k"] - 1:
            raise ValueError("piece column spans overlap")
    return {"grid": g, "H": h, "W": w, "border": border, "air": air,
            "terrain": terrain, "surf": surf, "pieces": pieces}


def _land(info, transform="ccw", drop=True):
    """Remove every piece, transform it inside its box and let it fall."""
    h, air = info["H"], info["air"]
    out = info["grid"].copy()
    for piece in info["pieces"]:
        for r, c in piece["cells"]:
            out[r, c] = air
    offsets = {}
    for index in sorted(range(len(info["pieces"])),
                        key=lambda i: -(info["pieces"][i]["r0"] + info["pieces"][i]["k"])):
        piece = info["pieces"][index]
        box = _transform(piece["box"], transform)
        ii, jj = np.nonzero(box >= 0)
        rows, cols, vals = ii + piece["r0"], jj + piece["c0"], box[ii, jj]
        if np.any(out[rows, cols] != air):
            return None, None
        d = 0
        if drop:
            while np.all(rows + d + 1 < h) and np.all(out[rows + d + 1, cols] == air):
                d += 1
        out[rows + d, cols] = vals
        offsets[index] = d
    return out, offsets


def _surface(grid, air):
    h, w = grid.shape
    surf = np.array([1 + int(np.flatnonzero(grid[1:, c] != air)[0]) for c in range(w)])
    return surf, h - surf


def _draw(grid, surf, peak, valley):
    out = grid.copy()
    out[1:surf[peak], peak] = PEAK_GUIDE
    out[1:surf[valley], valley] = VALLEY_GUIDE
    return out


def _survey(grid, air):
    surf, heights = _surface(grid, air)
    peaks = np.flatnonzero(heights == heights.max())
    valleys = np.flatnonzero(heights == heights.min())
    peak, valley = int(peaks[-1]), int(valleys[0])
    return _draw(grid, surf, peak, valley), {
        "surf": surf, "heights": heights, "peaks": peaks, "valleys": valleys,
        "peak": peak, "valley": valley}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    landed, _ = _land(info, "ccw")
    if landed is None:
        raise ValueError("piece cannot turn in place")
    output, _ = _survey(landed, info["air"])
    return output


# ---------------------------------------------------------------------------
# Analysis of one pair: family constraints, measurements and shortcut answers
# ---------------------------------------------------------------------------

def _symmetries(box):
    images = [_transform(box, name) for name in TRANSFORMS]
    keys = {img.tobytes() for img in images}
    return len(keys), bool(np.array_equal(np.rot90(box), box))


def _guide_mask(grid):
    return (grid == PEAK_GUIDE) | (grid == VALLEY_GUIDE)


def _analyze(inp, out=None):
    inp = np.asarray(inp)
    info = _parse(inp)
    g, h, w, air = info["grid"], info["H"], info["W"], info["air"]
    _require(not np.any(np.isin(g, (PEAK_GUIDE, VALLEY_GUIDE))), "guide colour in input")
    pieces = info["pieces"]
    _require(1 <= len(pieces) <= 2, "one or two floating pieces expected")
    terrain_mask = g == info["terrain"]
    win_cols = set()
    colors = []
    for piece in pieces:
        k, r0, c0 = piece["k"], piece["r0"], piece["c0"]
        _require(2 <= k <= 4, "piece size outside 2..4")
        _require(r0 >= 2, "piece touches the first air row")
        n_images, quarter_invariant = _symmetries(piece["box"])
        _require(not quarter_invariant, "piece is invariant under a quarter-turn")
        piece["images"] = n_images
        for r, c in piece["cells"]:
            near = terrain_mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
            _require(not near.any(), "piece touches terrain in the input")
        win_cols.update(range(c0, c0 + k))
        colors.append(tuple(sorted({int(v) for v in piece["box"].flat if v >= 0})))
        _require(not set(colors[-1]) & {info["border"], info["air"], info["terrain"]},
                 "piece colour reuses a role colour")
    for left, right in zip(pieces, pieces[1:]):
        _require(right["c0"] >= left["c0"] + left["k"] + 1, "pieces need a gap column")
    landed, offsets = _land(info, "ccw")
    _require(landed is not None, "piece cannot turn in place")
    _require(all(d >= 1 for d in offsets.values()), "every piece must fall")
    expected, s = _survey(landed, air)
    if out is not None:
        out = np.asarray(out)
        _require(out.shape == expected.shape and np.array_equal(out, expected),
                 "output disagrees with the turn-drop-survey rule")
    heights = s["heights"]
    _require(len(s["valleys"]) == 1, "final skyline must have a unique lowest column")
    _require(heights.max() > heights.min(), "flat final skyline")
    _require(s["surf"][s["peak"]] >= 2, "peak guide would be empty")
    peak, valley = s["peak"], s["valley"]
    terrain_heights = h - info["surf"]
    t_peaks = np.flatnonzero(terrain_heights == terrain_heights.max())
    t_valleys = np.flatnonzero(terrain_heights == terrain_heights.min())
    if len(s["peaks"]) > 1:
        inside = [int(p) in win_cols for p in s["peaks"]]
        tie_kind = "piece" if all(inside) else ("terrain" if not any(inside) else "mixed")
    else:
        tie_kind = None

    alternatives = {}
    for name in TRANSFORMS[1:]:
        alt, _ = _land(info, name)
        alternatives[name] = None if alt is None else _survey(alt, air)[0]
    alternatives["terrain_only"] = _draw(landed, s["surf"], int(t_peaks[-1]), int(t_valleys[0]))
    alternatives["leftmost_tie"] = _draw(landed, s["surf"], int(s["peaks"][0]), valley)
    floating, _ = _land(info, "ccw", drop=False)
    alternatives["no_drop"] = None if floating is None else _survey(floating, air)[0]
    alternatives["static"] = _survey(g, air)[0]
    plain, plain_offsets = _land(info, "identity")
    late = plain.copy()
    ok = True
    for index, piece in enumerate(pieces):
        d = plain_offsets[index]
        box = piece["box"]
        ii, jj = np.nonzero(box >= 0)
        late[ii + piece["r0"] + d, jj + piece["c0"]] = air
    for index, piece in enumerate(pieces):
        d = plain_offsets[index]
        box = np.rot90(piece["box"])
        ii, jj = np.nonzero(box >= 0)
        rows, cols = ii + piece["r0"] + d, jj + piece["c0"]
        if np.any(late[rows, cols] != air):
            ok = False
            break
        late[rows, cols] = box[ii, jj]
    alternatives["drop_then_rotate"] = _survey(late, air)[0] if ok else None
    differs = {name: alt is None or not np.array_equal(alt, expected)
               for name, alt in alternatives.items()}
    identity = alternatives["identity"]
    rotation_moves_guides = not np.array_equal(_guide_mask(identity), _guide_mask(expected)) or \
        not np.array_equal(identity[_guide_mask(identity)], expected[_guide_mask(expected)])
    return {
        "shape": (h, w), "roles": (info["border"], info["air"], info["terrain"]),
        "n_pieces": len(pieces), "ks": tuple(p["k"] for p in pieces),
        "colors": tuple(colors), "multicolor": any(len(c) > 1 for c in colors),
        "asymmetric": any(p["images"] == 8 for p in pieces),
        "cells": tuple(len(p["cells"]) for p in pieces),
        "heights": tuple(int(x) for x in heights),
        "terrain_heights": tuple(int(x) for x in terrain_heights),
        "peak": peak, "valley": valley, "n_peaks": len(s["peaks"]), "tie_kind": tie_kind,
        "piece_peak": peak in win_cols, "valley_in_piece": valley in win_cols,
        "valley_filled": len(t_valleys) == 1 and int(t_valleys[0]) in win_cols,
        "terrain_valley_unique": len(t_valleys) == 1,
        "peak_guide": int(s["surf"][peak] - 1), "valley_guide": int(s["surf"][valley] - 1),
        "order": "valley_left" if valley < peak else "peak_left",
        "drops": tuple(int(offsets[i]) for i in range(len(pieces))),
        "underside_matters": any(offsets[i] != plain_offsets[i] for i in range(len(pieces))),
        "differs": differs, "rotation_moves_guides": rotation_moves_guides,
        "piece_shapes": tuple(p["box"].tobytes() for p in pieces),
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if not 2 <= int(num_train) <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= int(num_test) <= 4:
        raise ValueError("num_test must be in 1..4")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")


def _context(rng):
    shift = SHIFTS[int(rng.integers(len(SHIFTS)))]
    size_mode = ("fixed", "height", "width", "both")[int(rng.integers(4))]
    base_h = int(rng.integers(12 if shift == "larger_piece" else 11, 19))
    base_w = int(rng.integers(10, 21))
    roles = [int(x) for x in rng.choice(ROLE_POOL, size=3, replace=False)]
    return {
        "shift": shift,
        "size_mode": size_mode,
        "base_shape": (base_h, base_w),
        "palette_mode": "stable" if rng.random() < 0.45 else "vary",
        "roles": tuple(roles),
        "count_mode": "single" if rng.random() < 0.55 else "mixed",
        "color_mode": ("mono", "multi", "mixed")[int(rng.integers(3))],
    }


def _base_plan(rng, ctx, training):
    two = ctx["count_mode"] == "mixed" and rng.random() < 0.4
    if ctx["color_mode"] == "mono":
        colors = 1
    elif ctx["color_mode"] == "multi":
        colors = int(rng.integers(2, 4))
    else:
        colors = int(rng.integers(1, 4))
    return {"n_pieces": 2 if two else 1, "ks": (2, 3, 4), "colors": colors,
            "piece_peak": None, "tie": None, "tie_kind": None, "valley_fill": None,
            "asym": False, "rot_guides": False, "order": None, "require_differs": (),
            "min_peak_guide": 1, "min_valley_guide": 1,
            "min_drop": 2 if rng.random() < 0.6 else 1,
            "underside": bool(rng.random() < 0.4), "training": training}


def _plans(rng, ctx, num_train, num_test):
    shift = ctx["shift"]
    first_order = ("valley_left", "peak_left")[int(rng.integers(2))]
    other_order = "peak_left" if first_order == "valley_left" else "valley_left"
    train_ks = (2, 3) if shift == "larger_piece" else (2, 3, 4)
    plans = []
    for index in range(num_train):
        plan = _base_plan(rng, ctx, True)
        plan["ks"] = train_ks
        if index == 0:
            plan.update(n_pieces=1, ks=(3,) if shift == "larger_piece" else (3, 4),
                        piece_peak=True, tie=False, asym=True, rot_guides=True,
                        require_differs=("terrain_only", "drop_then_rotate"),
                        order=first_order, min_peak_guide=3, min_valley_guide=5, min_drop=2)
            if ctx["color_mode"] == "mixed":
                plan["colors"] = int(rng.integers(2, 4))
        elif index == 1:
            plan.update(piece_peak=False, tie=True, tie_kind="terrain",
                        valley_fill=shift != "valley_fill", order=other_order)
            if ctx["count_mode"] == "mixed":
                plan["n_pieces"] = 2
        else:
            roll = rng.random()
            plan["piece_peak"] = bool(rng.random() < 0.45)
            plan["tie"] = True if roll < 0.3 else (False if roll < 0.7 else None)
            if shift == "valley_fill":
                plan["valley_fill"] = False
            elif rng.random() < 0.3:
                plan["valley_fill"] = True
                plan["piece_peak"] = False
            if plan["tie"] and plan["piece_peak"] and shift == "piece_tie":
                plan["tie_kind"] = "piece"
        if shift == "piece_tie" and plan["tie_kind"] is None and plan["tie"] is not False:
            plan["tie_kind"] = "not_mixed"
        plans.append(plan)
    for index in range(num_test):
        plan = _base_plan(rng, ctx, False)
        strict = index == 0 or rng.random() < 0.5
        if shift == "larger_piece":
            plan["ks"] = (4,) if strict else (2, 3, 4)
            plan["piece_peak"] = bool(rng.random() < 0.6)
        elif shift == "piece_tie" and strict:
            plan.update(piece_peak=True, tie=True, tie_kind="mixed")
        elif shift == "valley_fill" and strict:
            plan.update(piece_peak=False, valley_fill=True)
        else:
            plan["piece_peak"] = bool(rng.random() < 0.5)
            plan["tie"] = bool(rng.random() < 0.35)
            if shift == "piece_tie" and plan["tie"]:
                plan["tie_kind"] = "mixed" if plan["piece_peak"] else "terrain"
            if shift == "valley_fill" and not plan["piece_peak"] and rng.random() < 0.5:
                plan["valley_fill"] = True
        if ctx["count_mode"] == "single":
            plan["n_pieces"] = 1
        if ctx["color_mode"] == "mixed" and plan["colors"] == 1 and rng.random() < 0.5:
            plan["colors"] = 2
        plans.append(plan)
    return plans


def _shape(rng, ctx):
    h, w = ctx["base_shape"]
    mode = ctx["size_mode"]
    if mode in ("height", "both"):
        h = int(rng.integers(11, 21))
    if mode in ("width", "both"):
        w = int(rng.integers(9, 22))
    return h, w


def _roles(rng, ctx, index):
    roles = list(ctx["roles"])
    if ctx["palette_mode"] == "vary" and index > 0:
        changed = [i for i in range(3) if rng.random() < 0.5] or [int(rng.integers(3))]
        for i in changed:
            options = [c for c in ROLE_POOL if c not in roles]
            roles[i] = int(rng.choice(options))
    return tuple(roles)


def _terrain(rng, w, lo, hi, style):
    x = np.arange(w, dtype=float)
    if style in ("smooth", "terraces"):
        count = int(rng.integers(2, max(3, w // 3) + 1))
        knots = np.unique(np.concatenate(([0, w - 1], rng.choice(w, size=min(count, w), replace=False))))
        values = np.interp(x, knots, rng.uniform(lo, hi, size=len(knots)))
        if style == "terraces":
            step = int(rng.integers(2, 4))
            values = lo + np.round((values - lo) / step) * step
    elif style == "ridge":
        values = np.full(w, float(lo))
        for _ in range(int(rng.integers(1, 4))):
            center, radius = rng.uniform(0, w - 1), rng.uniform(1.5, max(2.5, w / 2.5))
            values += rng.uniform(1.0, hi - lo) * np.maximum(0, 1 - np.abs(x - center) / radius)
    elif style == "notches":
        values = np.full(w, float(hi))
        for _ in range(int(rng.integers(1, 4))):
            center, radius = rng.uniform(0, w - 1), rng.uniform(1.0, max(2.0, w / 3.5))
            values -= rng.uniform(1.0, hi - lo) * np.maximum(0, 1 - np.abs(x - center) / radius)
    elif style == "flat":
        values = np.full(w, float(rng.integers(lo, hi + 1)))
        bumps = rng.random(w) < 0.25
        values[bumps] += rng.integers(-1, 2, size=int(bumps.sum()))
    else:
        values = rng.integers(lo, hi + 1, size=w).astype(float)
    return np.clip(np.rint(values), lo, hi).astype(int)


def _polyomino(rng, k):
    if k == 2:
        mask = np.ones((2, 2), dtype=bool)
        mask[int(rng.integers(2)), int(rng.integers(2))] = False
        return mask
    target = int(rng.integers(5, 8)) if k == 3 else int(rng.integers(7, 12))
    while True:
        start = (int(rng.integers(k)), int(rng.integers(k)))
        cells = {start}
        while True:
            rows = {r for r, _ in cells}
            cols = {c for _, c in cells}
            full = len(rows) == k and len(cols) == k
            if full and len(cells) >= target:
                break
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                               if 0 <= r + dr < k and 0 <= c + dc < k} - cells)
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        mask = np.zeros((k, k), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if mask.any(axis=0).all() and mask.any(axis=1).all() and mask.sum() < k * k:
            return mask


def _piece_box(rng, k, n_colors, free, need_asym, top_single):
    for _ in range(200):
        mask = _polyomino(rng, k)
        cells = np.argwhere(mask)
        count = min(n_colors, len(cells))
        palette = [int(c) for c in rng.choice(free, size=count, replace=False)]
        box = np.full((k, k), -1, dtype=int)
        box[mask] = palette[0]
        if count > 1:
            order = rng.permutation(len(cells))
            # every extra colour gets at least one cell; a few more cells may share it
            for i, color in enumerate(palette[1:]):
                r, c = cells[order[i]]
                box[r, c] = color
            extra = order[count - 1:]
            for idx in extra:
                if rng.random() < 0.3:
                    r, c = cells[idx]
                    box[r, c] = palette[int(rng.integers(1, count))]
        images, quarter = _symmetries(box)
        if quarter:
            continue
        if need_asym and images < 8:
            continue
        if top_single and np.count_nonzero(np.rot90(box)[0] >= 0) != 1:
            continue
        return box
    return None


def _windows(rng, w, ks, target=None):
    total = sum(ks) + len(ks) - 1
    free = w - total
    if free < 2:
        return None
    for _ in range(30):
        cuts = np.sort(rng.integers(0, free + 1, size=len(ks)))
        starts = [int(cuts[i]) + sum(ks[:i]) + i for i in range(len(ks))]
        if target is None or any(s <= target < s + k for s, k in zip(starts, ks)):
            return starts
    return None


def _build(h, w, roles, heights, pieces):
    border, air, terrain = roles
    grid = np.full((h, w), air, dtype=int)
    grid[0] = border
    for c, height in enumerate(heights):
        grid[h - int(height):, c] = terrain
    for r0, c0, box in pieces:
        k = box.shape[0]
        region = grid[r0:r0 + k, c0:c0 + k]
        region[box >= 0] = box[box >= 0]
    return grid


def _pick_close(rng, columns, heights, goal):
    """Usually raise a column that is already near the goal height (natural relief)."""
    columns = list(columns)
    if rng.random() < 0.7:
        gaps = sorted(set(goal - int(heights[c]) for c in columns))[:2]
        columns = [c for c in columns if goal - int(heights[c]) in gaps]
    return int(rng.choice(columns))


def _attempt(rng, ctx, plan, index):
    h, w = _shape(rng, ctx)
    roles = _roles(rng, ctx, index)
    free = [c for c in ROLE_POOL if c not in roles]
    n = plan["n_pieces"]
    ks = [int(rng.choice(plan["ks"])) for _ in range(n)]
    if n == 2 and sum(ks) + 4 > w:
        ks = [2, 2] if w >= 8 else [2]
        n = len(ks)
    tier = int(rng.choice(3, p=(0.25, 0.4, 0.35)))
    sky = int(rng.integers(*((1, 3), (3, 6), (6, max(7, h // 2 + 1)))[tier]))
    hi = h - 1 - sky
    lo = int(rng.integers(1, max(2, hi - 2)))
    if hi - lo < 2:
        return None
    style = STYLES[int(rng.integers(len(STYLES)))]
    heights = _terrain(rng, w, lo, hi, style)
    target = None
    if plan["piece_peak"] and rng.random() < 0.75:
        target = int(rng.choice(np.flatnonzero(heights >= np.percentile(heights, 60))))
    elif plan["valley_fill"]:
        target = int(np.argmin(heights)) if rng.random() < 0.6 else int(rng.integers(w))
    starts = _windows(rng, w, ks, target)
    if starts is None:
        return None
    order = rng.permutation(n)
    win_cols, near_cols = set(), set()
    for s, k in zip(starts, ks):
        cap = h - k - 3
        if cap < 1:
            return None
        lo_c, hi_c = max(0, s - 1), min(w, s + k + 1)
        heights[lo_c:hi_c] = np.minimum(heights[lo_c:hi_c], cap)
        win_cols.update(range(s, s + k))
        near_cols.update(range(lo_c, hi_c))
    if plan["valley_fill"]:
        primary = int(order[0])
        s, k = starts[primary], ks[primary]
        column = int(rng.integers(s, s + k))
        others = [c for c in range(w) if c != column]
        m_other = int(heights[others].min())
        if m_other >= 2:
            heights[column] = int(rng.integers(max(1, m_other - 3), m_other))
        else:
            heights[column] = 1
            for c in others:
                if heights[c] == 1:
                    heights[c] = 2
    pieces = []
    for i, (s, k) in enumerate(zip(starts, ks)):
        primary = i == int(order[0])
        box = _piece_box(rng, k, plan["colors"], free,
                         need_asym=plan["asym"] and primary,
                         top_single=primary and plan["piece_peak"] is True and plan["tie"] is False)
        if box is None:
            return None
        lo_c, hi_c = max(0, s - 1), min(w, s + k + 1)
        min_surf = int((h - heights[lo_c:hi_c]).min())
        top_limit = min_surf - k - 1
        if top_limit < 2:
            return None
        r0 = int(rng.integers(2, top_limit + 1))
        pieces.append((r0, s, box))

    lower_cols = [c for c in range(w) if c not in win_cols]
    raise_cols = [c for c in range(w) if c not in near_cols]

    def post_heights():
        grid = _build(h, w, roles, heights, pieces)
        landed, _ = _land(_parse(grid), "ccw")
        return _surface(landed, roles[1])[1]

    post = post_heights()
    piece_top = int(max(post[c] for c in win_cols))
    if plan["piece_peak"] is True:
        for c in lower_cols:
            heights[c] = max(1, min(int(heights[c]), piece_top - 1))
    elif plan["piece_peak"] is False:
        if max(int(heights[c]) for c in lower_cols) <= piece_top:
            if piece_top + 1 > h - 2 or not raise_cols:
                return None
            goal = min(h - 2, piece_top + int(rng.integers(1, 4)))
            c = _pick_close(rng, raise_cols, heights, goal)
            heights[c] = goal
    post = post_heights()
    top = int(post.max())
    maxima = [c for c in range(w) if post[c] == top]
    if plan["tie"] is True and len(maxima) == 1:
        valley_now = int(np.argmin(post))
        candidates = [c for c in raise_cols if post[c] < top and c != valley_now]
        if plan["tie_kind"] == "mixed" and maxima[0] not in win_cols:
            return None
        if not candidates:
            return None
        c = _pick_close(rng, candidates, heights, top)
        heights[c] = top
    elif plan["tie"] is False and len(maxima) > 1:
        movable = [c for c in maxima if c not in win_cols]
        if len(movable) < len(maxima) - 1:
            return None
        keep = maxima[-1] if maxima[-1] in win_cols else int(rng.choice(maxima))
        for c in maxima:
            if c != keep:
                if c in win_cols:
                    return None
                heights[c] -= 1
    post = post_heights()
    low = int(post.min())
    minima = [c for c in range(w) if post[c] == low]
    if len(minima) > 1:
        movable = [c for c in minima if c in lower_cols]
        if low >= 2 and movable:
            heights[int(rng.choice(movable))] = low - 1
        else:
            keep = minima[int(rng.integers(len(minima)))]
            for c in minima:
                if c == keep:
                    continue
                if c not in raise_cols:
                    return None
                heights[c] += 1
    grid = _build(h, w, roles, heights, pieces)
    output = solve(grid)
    info = _analyze(grid, output)
    if plan["piece_peak"] is not None and info["piece_peak"] != plan["piece_peak"]:
        return None
    if plan["tie"] is not None and (info["n_peaks"] > 1) != plan["tie"]:
        return None
    if plan["tie_kind"] == "not_mixed" and info["tie_kind"] == "mixed":
        return None
    if plan["tie_kind"] in ("terrain", "mixed", "piece") and info["tie_kind"] != plan["tie_kind"]:
        return None
    if plan["valley_fill"] is not None and info["valley_filled"] != plan["valley_fill"]:
        return None
    if plan["asym"] and not info["asymmetric"]:
        return None
    if plan["rot_guides"] and not info["rotation_moves_guides"]:
        return None
    if plan["order"] is not None and info["order"] != plan["order"]:
        return None
    if not all(info["differs"][name] for name in plan["require_differs"]):
        return None
    if min(info["drops"]) < plan["min_drop"]:
        return None
    if plan["underside"] and not info["underside_matters"]:
        return None
    if info["peak_guide"] < plan["min_peak_guide"] or info["valley_guide"] < plan["min_valley_guide"]:
        return None
    if plan["n_pieces"] != info["n_pieces"] and plan["training"] and index == 1:
        return None
    return {"input": grid, "output": output}


def _make_pair(rng, ctx, plan, index, seen):
    for _ in range(3000):
        try:
            pair = _attempt(rng, ctx, plan, index)
        except (ValueError, AssertionError):
            continue
        if pair is None:
            continue
        key = pair["input"].tobytes() + bytes(pair["input"].shape)
        if key in seen:
            continue
        seen.add(key)
        return pair
    raise RuntimeError("could not construct a pair for plan")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    num_train, num_test = int(num_train), int(num_test)
    rng = np.random.default_rng(seed)
    ctx = _context(rng)
    plans = _plans(rng, ctx, num_train, num_test)
    last_error = None
    for _ in range(8):
        seen = set()
        pairs = []
        for index, plan in enumerate(plans):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pairs.append(_make_pair(pair_rng, ctx, plan, index, seen))
        train = [pairs[i] for i in rng.permutation(num_train)]
        puzzle = {"train": train, "test": pairs[num_train:]}
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError as error:
            last_error = error
            continue
        return puzzle
    raise RuntimeError(f"could not build a valid puzzle: {last_error}")


# ---------------------------------------------------------------------------
# Validation and self-test
# ---------------------------------------------------------------------------

def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    infos = []
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair must contain exactly input and output")
        for role in ("input", "output"):
            grid = np.asarray(pair[role])
            _require(grid.ndim == 2 and 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                     "grid dimensions must be 1..30")
            _require(np.issubdtype(grid.dtype, np.integer) and grid.min() >= 0 and grid.max() <= 9,
                     "grid colours must be integers 0..9")
        inp = np.asarray(pair["input"])
        key = inp.tobytes() + bytes(inp.shape)
        _require(key not in seen, "duplicate input")
        seen.add(key)
        infos.append(_analyze(inp, pair["output"]))
    tr, te = infos[:len(train)], infos[len(train):]
    _require(any(i["n_peaks"] == 1 for i in tr), "training lacks a unique tallest column")
    _require(any(i["n_peaks"] > 1 for i in tr), "training lacks a tied tallest column")
    _require({i["order"] for i in tr} == {"valley_left", "peak_left"},
             "training must show both peak/valley orders")
    _require(any(i["piece_peak"] for i in tr), "training never lets the landed piece form the peak")
    _require(any(i["rotation_moves_guides"] for i in tr),
             "training never shows the turn changing a guide")
    _require(any(i["peak_guide"] >= 3 and i["valley_guide"] >= 5 for i in tr),
             "training needs one substantial pair of guides")
    for name in SHORTCUTS:
        _require(any(i["differs"][name] for i in tr), f"training does not refute shortcut {name}")
    for info in te:
        _require(info["differs"]["identity"], "test turn must be visible (ignoring B must fail)")
    if any(i["n_pieces"] > 1 for i in te):
        _require(any(i["n_pieces"] > 1 for i in tr), "two-piece tests need a two-piece demonstration")
    if any(i["multicolor"] for i in te):
        _require(any(i["multicolor"] for i in tr), "multicolour tests need a multicolour demonstration")
    if any(i["n_peaks"] > 1 for i in te):
        _require(any(i["n_peaks"] > 1 for i in tr), "tied tests need a tied demonstration")
    return True


def self_test():
    counts = [(4, 1), (2, 1), (3, 2), (5, 1), (6, 3), (10, 4), (7, 2)]
    shifts, sizes, ks, boxes = set(), set(), set(), set()
    stats = {"tie_mixed": 0, "valley_filled": 0, "piece_peak": 0, "two_piece": 0,
             "multicolor": 0, "fixed_size": 0, "varied_size": 0, "stable_roles": 0,
             "varied_roles": 0, "test_valley_filled": 0, "test_k4": 0}
    signatures = set()
    for seed in range(42):
        nt, nq = counts[seed % len(counts)]
        puzzle = generate_puzzle(seed, nt, nq)
        _require(len(puzzle["train"]) == nt and len(puzzle["test"]) == nq, "counts ignored")
        validate_generated_puzzle(puzzle)
        signatures.add(b"".join(np.asarray(p["input"]).tobytes()
                                for p in puzzle["train"] + puzzle["test"]))
        rng = np.random.default_rng(seed)
        shifts.add(_context(rng)["shift"])
        infos = [_analyze(p["input"], p["output"]) for p in puzzle["train"] + puzzle["test"]]
        for pair in puzzle["train"] + puzzle["test"]:
            as_list = np.asarray(pair["input"]).tolist()
            _require(np.array_equal(solve(as_list), pair["output"]), "solve must read only its input")
        shapes = {i["shape"] for i in infos}
        roles = {i["roles"] for i in infos}
        stats["fixed_size"] += len(shapes) == 1
        stats["varied_size"] += len(shapes) > 1
        stats["stable_roles"] += len(roles) == 1
        stats["varied_roles"] += len(roles) > 1
        for i in infos:
            sizes.add(i["shape"])
            ks.update(i["ks"])
            boxes.update(i["piece_shapes"])
            stats["tie_mixed"] += i["tie_kind"] == "mixed"
            stats["valley_filled"] += i["valley_filled"]
            stats["piece_peak"] += i["piece_peak"]
            stats["two_piece"] += i["n_pieces"] == 2
            stats["multicolor"] += i["multicolor"]
        for i in infos[nt:]:
            stats["test_valley_filled"] += i["valley_filled"]
            stats["test_k4"] += 4 in i["ks"]
    _require(len(signatures) == 42, "seeds repeat puzzles")
    _require(shifts == set(SHIFTS), "train/test shift modes missing")
    _require(ks == {2, 3, 4} and len(sizes) >= 40 and len(boxes) >= 120,
             "block sizes, block patterns or grid sizes lack variety")
    _require(all(v >= 3 for v in stats.values()), f"family regimes missing: {stats}")
    a = generate_puzzle(123, 5, 2)
    b = generate_puzzle(123, 5, 2)
    _require(all(np.array_equal(x[r], y[r]) for s in ("train", "test")
                 for x, y in zip(a[s], b[s]) for r in ("input", "output")),
             "generation is not deterministic")
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.5, 1), (True, 1)):
        try:
            generate_puzzle(5, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {bad}")
    return True
