"""Swinging bead arms: each motif is the hub of compressed arms of colored beads.

Every arm swings one quarter-turn counterclockwise about its hub (parent A) and,
while swinging, unfolds into full copies of the hub's own shape, one copy per bead,
in the bead colors and with the bead spacings preserved between copies (parent B).
Beads are erased; hubs stay; copies leaving the canvas are clipped.
"""
from __future__ import annotations

import numpy as np


_VEC = {"R": (0, 1), "U": (-1, 0), "L": (0, -1), "D": (1, 0)}
_CCW = {"R": "U", "U": "L", "L": "D", "D": "R"}
_CW = {v: k for k, v in _CCW.items()}
_CYCLE = ("R", "U", "L", "D")
_SIDE = {"U": "top", "D": "bottom", "L": "left", "R": "right"}
_ANTI = {"R": "U", "U": "R", "L": "D", "D": "L"}      # anti-diagonal mirror
_MAIN = {"R": "D", "D": "R", "L": "U", "U": "L"}      # main-diagonal mirror
_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))


# --------------------------------------------------------------------------
# Input parsing shared by solve() and validation.  Nothing here uses the seed.
# --------------------------------------------------------------------------

def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(fg):
    height, width = fg.shape
    seen = np.zeros(fg.shape, dtype=bool)
    comps = []
    for r, c in zip(*np.nonzero(fg)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, cells = [(r, c)], []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and fg[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        comps.append(cells)
    return comps


def _ray(box, direction, height, width):
    """Cells on the hub's centre line from just outside its box to the border."""
    r0, c0, h, w = box
    cr, cc = r0 + h // 2, c0 + w // 2
    if direction == "R":
        return [(cr, c) for c in range(c0 + w, width)]
    if direction == "L":
        return [(cr, c) for c in range(c0 - 1, -1, -1)]
    if direction == "D":
        return [(r, cc) for r in range(r0 + h, height)]
    return [(r, cc) for r in range(r0 - 1, -1, -1)]


def _gaps(box, direction, cells):
    """Blank runs: hub box -> first bead, then bead -> bead."""
    r0, c0, h, w = box
    if direction == "R":
        edge, coords = c0 + w - 1, [c for _, c in cells]
    elif direction == "L":
        edge, coords = c0, [c for _, c in cells]
    elif direction == "D":
        edge, coords = r0 + h - 1, [r for r, _ in cells]
    else:
        edge, coords = r0, [r for r, _ in cells]
    gaps, previous = [], edge
    for value in coords:
        gaps.append(abs(value - previous) - 1)
        previous = value
    return gaps


def _bead_cells(box, direction, gaps):
    r0, c0, h, w = box
    cr, cc = r0 + h // 2, c0 + w // 2
    cells = []
    if direction == "R":
        x = c0 + w - 1
        for g in gaps:
            x += g + 1
            cells.append((cr, x))
    elif direction == "L":
        x = c0
        for g in gaps:
            x -= g + 1
            cells.append((cr, x))
    elif direction == "D":
        y = r0 + h - 1
        for g in gaps:
            y += g + 1
            cells.append((y, cc))
    else:
        y = r0
        for g in gaps:
            y -= g + 1
            cells.append((y, cc))
    return cells


def _copy_origins(box, lane, gaps):
    """Top-left corners of the unfolded copies laid out along `lane`."""
    r0, c0, h, w = box
    origins = []
    if lane == "U":
        edge = r0
        for g in gaps:
            top = edge - 1 - g - h + 1
            origins.append((top, c0))
            edge = top
    elif lane == "D":
        edge = r0 + h - 1
        for g in gaps:
            top = edge + 1 + g
            origins.append((top, c0))
            edge = top + h - 1
    elif lane == "L":
        edge = c0
        for g in gaps:
            left = edge - 1 - g - w + 1
            origins.append((r0, left))
            edge = left
    else:
        edge = c0 + w - 1
        for g in gaps:
            left = edge + 1 + g
            origins.append((r0, left))
            edge = left + w - 1
    return origins


def _stamp(grid, mask, top, left, color):
    height, width = grid.shape
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                r, c = top + y, left + x
                if 0 <= r < height and 0 <= c < width:
                    grid[r, c] = color


def _visible(top, left, h, w, height, width):
    rows = max(0, min(height, top + h) - max(0, top))
    cols = max(0, min(width, left + w) - max(0, left))
    return rows * cols


def _in_box(cell, box):
    r0, c0, h, w = box
    return r0 <= cell[0] < r0 + h and c0 <= cell[1] < c0 + w


def _sees(cell, box, boxes):
    """True if `cell` shares a row or column band with `box` with no hub between."""
    r, c = cell
    r0, c0, h, w = box
    if r0 <= r < r0 + h:
        span = [(r, x) for x in (range(c + 1, c0) if c < c0 else range(c0 + w, c))]
    elif c0 <= c < c0 + w:
        span = [(y, c) for y in (range(r + 1, r0) if r < r0 else range(r0 + h, r))]
    else:
        return False
    return not any(_in_box(p, b) for p in span for b in boxes)


def _parse(grid):
    grid = np.asarray(grid, dtype=int)
    height, width = grid.shape
    bg = _background(grid)
    fg = grid != bg
    tokens, hubs = {}, []
    blocked = np.zeros(grid.shape, dtype=bool)
    for cells in _components(fg):
        if len(cells) == 1:
            tokens[cells[0]] = int(grid[cells[0]])
            continue
        rows = [p[0] for p in cells]
        cols = [p[1] for p in cells]
        r0, c0 = min(rows), min(cols)
        h, w = max(rows) - r0 + 1, max(cols) - c0 + 1
        mask = np.zeros((h, w), dtype=bool)
        for y, x in cells:
            mask[y - r0, x - c0] = True
            blocked[y, x] = True
        colors = sorted({int(grid[p]) for p in cells})
        hubs.append({"box": (r0, c0, h, w), "mask": mask, "colors": colors,
                     "color": colors[0], "cells": cells, "arms": {}})
    claims = {}
    for index, hub in enumerate(hubs):
        r0, c0, h, w = hub["box"]
        for direction in _CYCLE:
            if (direction in "LR" and h % 2 == 0) or (direction in "UD" and w % 2 == 0):
                continue
            found = []
            for cell in _ray(hub["box"], direction, height, width):
                if blocked[cell]:
                    break
                if cell in tokens:
                    found.append(cell)
            if found:
                hub["arms"][direction] = found
                for cell in found:
                    claims.setdefault(cell, []).append((index, direction))
    return {"bg": bg, "hubs": hubs, "tokens": tokens, "claims": claims,
            "shape": (height, width)}


def _render(grid, parsed, *, lane_of=None, mask_of=None, color_of=None,
            gaps_of=None, erase=True, drop_partial=False, singletons=False,
            radius=False):
    """Draw the parsed scene under the true rule or under a shortcut variant."""
    out = np.array(grid, dtype=int, copy=True)
    height, width = out.shape
    bg = parsed["bg"]
    if erase:
        for cell in parsed["claims"]:
            out[cell] = bg
    for hub in parsed["hubs"]:
        r0, c0, h, w = hub["box"]
        cr, cc = r0 + h // 2, c0 + w // 2
        for direction, cells in hub["arms"].items():
            lane = lane_of(direction) if lane_of else _CCW[direction]
            colors = [parsed["tokens"][cell] for cell in cells]
            if color_of:
                colors = [color_of(hub, color) for color in colors]
            if singletons:
                vr, vc = _VEC[lane]
                for (y, x), color in zip(cells, colors):
                    distance = abs(y - cr) + abs(x - cc)
                    r, c = cr + vr * distance, cc + vc * distance
                    if 0 <= r < height and 0 <= c < width:
                        out[r, c] = color
                continue
            gaps = _gaps(hub["box"], direction, cells)
            if gaps_of:
                gaps = gaps_of(gaps)
            mask = mask_of(hub["mask"]) if mask_of else hub["mask"]
            mh, mw = mask.shape
            if radius:
                origins = []
                vr, vc = _VEC[lane]
                for y, x in cells:
                    distance = abs(y - cr) + abs(x - cc)
                    if lane == "U":
                        origins.append((cr - distance - mh + 1, c0))
                    elif lane == "D":
                        origins.append((cr + distance, c0))
                    elif lane == "L":
                        origins.append((r0, cc - distance - mw + 1))
                    else:
                        origins.append((r0, cc + distance))
            else:
                origins = _copy_origins((r0, c0, mh, mw), lane, gaps)
            for (top, left), color in zip(origins, colors):
                seen = _visible(top, left, mh, mw, height, width)
                if drop_partial and seen < mh * mw:
                    continue
                _stamp(out, mask, top, left, color)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase every bead, then unfold each arm a quarter-turn counterclockwise."""
    grid = np.asarray(input_grid, dtype=int)
    return _render(grid, _parse(grid))


# --------------------------------------------------------------------------
# Shortcut variants used to prove that the demonstrations exclude them.
# --------------------------------------------------------------------------

def _variant(grid, name):
    grid = np.asarray(grid, dtype=int)
    parsed = _parse(grid)
    if name == "no_swing":            # parent B only: unfold along the arm itself
        return _render(grid, parsed, lane_of=lambda d: d)
    if name == "no_unfold":           # parent A only: rotate each bead as a point
        return _render(grid, parsed, singletons=True)
    if name == "clockwise":
        return _render(grid, parsed, lane_of=lambda d: _CW[d])
    if name == "half_turn":
        return _render(grid, parsed, lane_of=lambda d: _CCW[_CCW[d]])
    if name == "anti_mirror":
        return _render(grid, parsed, lane_of=lambda d: _ANTI[d])
    if name == "main_mirror":
        return _render(grid, parsed, lane_of=lambda d: _MAIN[d])
    if name == "radius":              # keep each bead's distance instead of the gaps
        return _render(grid, parsed, radius=True)
    if name == "uniform_gap":
        return _render(grid, parsed, gaps_of=lambda g: [g[0]] * len(g))
    if name == "rotate_copies":       # copies turn with the arm
        return _render(grid, parsed, mask_of=lambda m: np.rot90(m))
    if name == "hub_color":
        return _render(grid, parsed, color_of=lambda hub, color: hub["color"])
    if name == "keep_beads":
        return _render(grid, parsed, erase=False)
    if name == "drop_partial":
        return _render(grid, parsed, drop_partial=True)
    raise KeyError(name)


_VARIANTS = ("no_swing", "no_unfold", "clockwise", "half_turn", "anti_mirror",
             "main_mirror", "radius", "uniform_gap", "rotate_copies",
             "hub_color", "keep_beads", "drop_partial")


# --------------------------------------------------------------------------
# Scene analysis (from the input/output pair only).
# --------------------------------------------------------------------------

def _mask_ok(mask):
    h, w = mask.shape
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return False
    total = int(mask.sum())
    if total < 4 or total >= h * w:
        return False
    if not (mask.any(axis=0).all() and mask.any(axis=1).all()):
        return False
    return len(_components(mask)) == 1


def _analyze(inp, out):
    inp = np.asarray(inp, dtype=int)
    out = np.asarray(out, dtype=int)
    assert inp.shape == out.shape, "shape changed"
    height, width = inp.shape
    values, counts = np.unique(inp, return_counts=True)
    order = np.argsort(counts)[::-1]
    assert len(counts) == 1 or counts[order[0]] > counts[order[1]], "background is not unique"
    parsed = _parse(inp)
    bg, hubs, tokens, claims = parsed["bg"], parsed["hubs"], parsed["tokens"], parsed["claims"]
    assert hubs, "no hub"
    assert set(claims) == set(tokens), "a bead belongs to no hub"
    assert all(len(v) == 1 for v in claims.values()), "a bead belongs to two hubs"
    boxes = [hub["box"] for hub in hubs]
    for cell, ((owner, _),) in claims.items():
        assert not any(_sees(cell, box, boxes) for i, box in enumerate(boxes) if i != owner), \
            "a bead lines up with a foreign hub"
    feats = {"dirs": set(), "arms_per_hub": [], "token_counts": [], "gaps": [],
             "unequal": False, "multi": False, "partial": False, "vanish": False,
             "clip_sides": set(), "takeover": False, "nonsquare": False,
             "shapes": set(), "asym": False, "color_diff": False,
             "hub_color_bead": False, "visible_copies": 0, "fit_multi": False,
             "hub_count": len(hubs), "dims": (height, width), "bg": bg,
             "lanes": set(), "max_gap": 0, "multi_visible": False}
    occupied = {}                 # cell -> (hub index, arm)
    for index, hub in enumerate(hubs):
        r0, c0, h, w = hub["box"]
        assert len(hub["colors"]) == 1, "hub is not monochrome"
        assert _mask_ok(hub["mask"]), "hub mask is not a legal motif"
        assert hub["arms"], "hub without arm"
        feats["arms_per_hub"].append(len(hub["arms"]))
        feats["shapes"].add((h, w))
        feats["nonsquare"] |= h != w
        feats["asym"] |= h != w or not np.array_equal(np.rot90(hub["mask"]), hub["mask"])
        own_beads = {cell for cells in hub["arms"].values() for cell in cells}
        for direction, cells in hub["arms"].items():
            lane = _CCW[direction]
            gaps = _gaps(hub["box"], direction, cells)
            assert min(gaps) >= 1, "beads must be separated by blanks"
            feats["dirs"].add(direction)
            feats["lanes"].add(lane)
            feats["token_counts"].append(len(cells))
            feats["gaps"].extend(gaps)
            feats["max_gap"] = max(feats["max_gap"], max(gaps))
            feats["multi"] |= len(cells) >= 2
            feats["unequal"] |= len(cells) >= 2 and len(set(gaps)) > 1
            all_full = True
            shown = 0
            origins = _copy_origins(hub["box"], lane, gaps)
            assert _visible(*origins[0], h, w, height, width) > 0, "first copy of an arm is off-canvas"
            for (top, left), cell in zip(origins, cells):
                color = tokens[cell]
                feats["color_diff"] |= color != hub["color"]
                feats["hub_color_bead"] |= color == hub["color"]
                seen = _visible(top, left, h, w, height, width)
                if seen == 0:
                    feats["vanish"] = True
                    feats["clip_sides"].add(_SIDE[lane])
                    all_full = False
                    continue
                if seen < h * w:
                    feats["partial"] = True
                    feats["clip_sides"].add(_SIDE[lane])
                    all_full = False
                feats["visible_copies"] += 1
                shown += 1
                for y in range(h):
                    for x in range(w):
                        r, c = top + y, left + x
                        if not (0 <= r < height and 0 <= c < width):
                            continue
                        if (r, c) in own_beads and (r, c) not in cells:
                            feats["takeover"] = True
                        key = occupied.get((r, c))
                        assert key is None, "two copies overlap"
                        occupied[(r, c)] = (index, direction)
            if all_full and len(cells) >= 2 and len(set(gaps)) > 1:
                feats["fit_multi"] = True
            feats["multi_visible"] |= shown >= 2
    # Copies keep clear of other hubs, other hubs' beads and other arms' copies.
    for (r, c), (index, direction) in occupied.items():
        for other, hub in enumerate(hubs):
            if other == index:
                continue
            r0, c0, h, w = hub["box"]
            assert not (r0 - 1 <= r <= r0 + h and c0 - 1 <= c <= c0 + w), "copy touches another hub"
            assert not any((r, c) in cells for cells in hub["arms"].values()), "copy covers a foreign bead"
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                key = occupied.get((r + dy, c + dx))
                assert key is None or key == (index, direction), "copies of two arms touch"
    assert feats["visible_copies"] >= 1, "nothing visible is unfolded"
    expected = solve(inp)
    assert np.array_equal(expected, out), "output disagrees with the rule"
    assert not np.array_equal(inp, out), "identity pair"
    return feats


# --------------------------------------------------------------------------
# Generation.
# --------------------------------------------------------------------------

def _space(box, direction, height, width):
    r0, c0, h, w = box
    return {"R": width - (c0 + w), "L": c0, "U": r0, "D": height - (r0 + h)}[direction]


def _make_mask(rng, h, w):
    style = ("random", "random", "ring", "sparse", "source")[int(rng.integers(0, 5))]
    for _ in range(300):
        if style == "ring":
            mask = np.ones((h, w), dtype=bool)
            mask[1:-1, 1:-1] = False
            if rng.random() < 0.7:
                border = [(y, x) for y in range(h) for x in range(w)
                          if y in (0, h - 1) or x in (0, w - 1)]
                y, x = border[int(rng.integers(0, len(border)))]
                mask[y, x] = False
            if rng.random() < 0.3:
                mask[h // 2, w // 2] = True
        elif style == "source" and (h, w) == (3, 3):
            mask = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=bool)
            mask = np.rot90(mask, int(rng.integers(0, 4)))
            if rng.random() < 0.5:
                mask = mask[:, ::-1]
        else:
            low, high = (0.38, 0.62) if style == "sparse" else (0.5, 0.85)
            mask = rng.random((h, w)) < float(rng.uniform(low, high))
        mask = np.ascontiguousarray(mask)
        if _mask_ok(mask):
            return mask
    mask = np.ones((h, w), dtype=bool)
    mask[1:-1, 1:-1] = False
    return mask


def _asymmetric(mask):
    return mask.shape[0] != mask.shape[1] or not np.array_equal(np.rot90(mask), mask)


_GAP_CDF = {hi: np.cumsum(np.array([5, 4, 3, 2, 1, 1][:hi], dtype=float)) /
            sum([5, 4, 3, 2, 1, 1][:hi]) for hi in range(1, 7)}


def _sample_gaps(rng, k, gap_hi, uniform):
    """Blank-run lengths 1..gap_hi, short runs more common."""
    if uniform:
        return [int(rng.integers(1, gap_hi + 1))] * k
    draws = np.searchsorted(_GAP_CDF[gap_hi], rng.random(k), side="right")
    return [int(min(d, gap_hi - 1)) + 1 for d in draws]


def _sample_arm(rng, box, direction, height, width, k_range, gap_hi, spec):
    """Pick blank runs for one arm that fit the input and meet a clipping wish."""
    space_in = _space(box, direction, height, width)
    lane = _CCW[direction]
    size = box[2] if lane in "UD" else box[3]
    space_out = _space(box, lane, height, width)
    low, high = k_range
    if spec.get("multi") or spec.get("unequal"):
        low = max(low, 2)
    high = min(high, space_in // 2)
    if high < low:
        return None
    if low == 1 and high >= 2 and rng.random() < 0.45:
        low = 2                               # favour arms whose spacing is visible
    pref = spec.get("pref")
    for _ in range(30):
        k = int(rng.integers(low, high + 1))
        uniform = k >= 2 and not spec.get("unequal") and rng.random() < 0.25
        gaps = _sample_gaps(rng, k, gap_hi, uniform)
        if sum(gaps) + k > space_in:
            continue
        if spec.get("unequal") and len(set(gaps)) < 2:
            continue
        starts, offset = [], 0
        for g in gaps:
            offset += g
            starts.append(offset)
            offset += size
        full = [s + size <= space_out for s in starts]
        part = [s < space_out < s + size for s in starts]
        gone = [s >= space_out for s in starts]
        if gone[0] and pref != "vanish":
            continue                          # keep at least one copy in view
        if pref == "fit" and not all(full):
            continue
        if pref == "partial" and not any(part):
            continue
        if pref == "clip" and all(full):
            continue
        if pref == "vanish" and not (any(gone) and not gone[0]):
            continue
        return gaps
    return None


class _Scene:
    def __init__(self, height, width):
        self.height, self.width = height, width
        self.halo = np.zeros((height, width), dtype=bool)        # input objects, dilated
        self.hub_halo = np.zeros((height, width), dtype=bool)    # hub boxes, dilated
        self.hub_box = np.zeros((height, width), dtype=bool)
        self.beads = np.zeros((height, width), dtype=bool)
        self.copies = np.zeros((height, width), dtype=bool)
        self.copy_halo = np.zeros((height, width), dtype=bool)
        self.corridor = np.zeros((height, width), dtype=bool)    # hub edge .. last bead
        self.line_rows, self.line_cols = set(), set()
        self.hubs = []

    def _dilate(self, cells):
        out = set()
        for r, c in cells:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = r + dy, c + dx
                    if 0 <= y < self.height and 0 <= x < self.width:
                        out.add((y, x))
        return out

    def describe(self, box, arms):
        r0, c0, h, w = box
        box_cells = [(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)]
        beads, corridor, copies = [], [], []
        for direction, gaps in arms.items():
            cells = _bead_cells(box, direction, gaps)
            beads.extend(cells)
            ray = _ray(box, direction, self.height, self.width)
            corridor.extend(ray[:ray.index(cells[-1]) + 1])
            for top, left in _copy_origins(box, _CCW[direction], gaps):
                for y in range(top, top + h):
                    for x in range(left, left + w):
                        if 0 <= y < self.height and 0 <= x < self.width:
                            copies.append((y, x))
        return box_cells, beads, corridor, copies

    def can_add(self, box, arms):
        r0, c0, h, w = box
        box_cells, beads, corridor, copies = self.describe(box, arms)
        cr, cc = r0 + h // 2, c0 + w // 2
        for cell in box_cells:
            if self.halo[cell] or self.copy_halo[cell] or self.corridor[cell]:
                return False
        for cell in beads:
            if self.halo[cell] or self.copies[cell]:
                return False
            if cell[0] in self.line_rows or cell[1] in self.line_cols:
                return False
        if np.any(self.beads[cr, :]) or np.any(self.beads[:, cc]):
            return False
        for cell in corridor:
            if self.hub_box[cell]:
                return False
        for cell in copies:
            if self.hub_halo[cell] or self.beads[cell] or self.copy_halo[cell]:
                return False
        boxes = [hub["box"] for hub in self.hubs] + [box]
        for hub in self.hubs:
            if any(_sees(cell, hub["box"], boxes) for cell in beads):
                return False
            for direction, gaps in hub["arms"].items():
                if any(_sees(cell, box, boxes) for cell in _bead_cells(hub["box"], direction, gaps)):
                    return False
        return True

    def add(self, box, arms, mask, hub_color, bead_colors):
        r0, c0, h, w = box
        box_cells, beads, corridor, copies = self.describe(box, arms)
        for cell in self._dilate(box_cells):
            self.halo[cell] = True
            self.hub_halo[cell] = True
        for cell in box_cells:
            self.hub_box[cell] = True
        for cell in self._dilate(beads):
            self.halo[cell] = True
        for cell in beads:
            self.beads[cell] = True
        for cell in corridor:
            self.corridor[cell] = True
        for cell in copies:
            self.copies[cell] = True
        for cell in self._dilate(copies):
            self.copy_halo[cell] = True
        self.line_rows.add(r0 + h // 2)
        self.line_cols.add(c0 + w // 2)
        self.hubs.append({"box": box, "arms": arms, "mask": mask,
                          "color": hub_color, "bead_colors": bead_colors})


def _dims(rng, plan):
    size = plan.get("size")
    if size is None:
        load = sum(len(spec["dirs"]) for spec in plan["hubs"]) + 2 * len(plan["hubs"])
        big = any(max(spec["shape"]) == 5 for spec in plan["hubs"])
        if load >= 10 or big:
            size = ("large", "large", "elongated", "medium")[int(rng.integers(0, 4))]
        elif load >= 6:
            size = ("medium", "large", "elongated", "medium")[int(rng.integers(0, 4))]
        else:
            size = ("compact", "medium", "elongated", "large")[int(rng.integers(0, 4))]
    if size == "compact":
        return int(rng.integers(9, 15)), int(rng.integers(9, 15))
    if size == "medium":
        return int(rng.integers(12, 21)), int(rng.integers(12, 21))
    if size == "large":
        return int(rng.integers(18, 31)), int(rng.integers(18, 31))
    short, long = int(rng.integers(9, 16)), int(rng.integers(18, 31))
    return (short, long) if rng.random() < 0.5 else (long, short)


def _bead_palette(rng, bg, hub_color, k, shared):
    pool = [c for c in (shared if shared else range(10)) if c != bg]
    others = [c for c in pool if c != hub_color] or pool
    mode = int(rng.integers(0, 4))
    if mode == 0:                                   # all distinct where possible
        chosen = list(rng.permutation(others))[:k]
        while len(chosen) < k:
            chosen.append(int(rng.choice(others)))
        colors = [int(c) for c in chosen]
    elif mode == 1:                                 # heavy repeats
        few = [int(c) for c in rng.choice(others, size=min(2, len(others)), replace=False)]
        colors = [int(rng.choice(few)) for _ in range(k)]
    else:
        colors = [int(rng.choice(others)) for _ in range(k)]
    if rng.random() < 0.12:                         # a bead may share the hub colour
        colors[int(rng.integers(0, k))] = hub_color
    return colors


def _build_pair(rng, plan, colors):
    """Rejection-sample one scene that realises `plan`."""
    for _ in range(90):
        height, width = _dims(rng, plan)
        scene = _Scene(height, width)
        bg = colors["bg"]
        hub_pool = [c for c in range(10) if c != bg]
        used_hub_colors = []
        success = True
        for spec in plan["hubs"]:
            h, w = spec["shape"]
            if h + 2 > height or w + 2 > width:
                success = False
                break
            mask = spec.get("mask")
            if mask is None:
                mask = _make_mask(rng, h, w)
                if spec.get("asym"):
                    for _ in range(20):
                        if _asymmetric(mask):
                            break
                        mask = _make_mask(rng, h, w)
                    if not _asymmetric(mask):
                        success = False
                        break
            placed = False
            for _ in range(45):
                r0 = int(rng.integers(0, height - h + 1))
                c0 = int(rng.integers(0, width - w + 1))
                box = (r0, c0, h, w)
                arms = {}
                for direction in spec["dirs"]:
                    gaps = _sample_arm(rng, box, direction, height, width, spec["k"],
                                       plan["gap_hi"], spec.get("arm", {}).get(direction, {}))
                    if gaps is None:
                        break
                    arms[direction] = gaps
                if len(arms) != len(spec["dirs"]):
                    continue
                if scene.can_add(box, arms):
                    placed = True
                    break
            if not placed:
                success = False
                break
            choices = [c for c in hub_pool if c not in used_hub_colors] or hub_pool
            hub_color = int(rng.choice(choices))
            used_hub_colors.append(hub_color)
            bead_colors = {d: _bead_palette(rng, bg, hub_color, len(g), colors.get("beads"))
                           for d, g in arms.items()}
            scene.add(box, arms, mask, hub_color, bead_colors)
        if not success:
            continue
        inp = np.full((height, width), bg, dtype=int)
        expected = inp.copy()
        for hub in scene.hubs:
            r0, c0, h, w = hub["box"]
            _stamp(inp, hub["mask"], r0, c0, hub["color"])
            _stamp(expected, hub["mask"], r0, c0, hub["color"])
            for direction, gaps in hub["arms"].items():
                for cell, color in zip(_bead_cells(hub["box"], direction, gaps),
                                       hub["bead_colors"][direction]):
                    inp[cell] = color
        for hub in scene.hubs:
            for direction, gaps in hub["arms"].items():
                origins = _copy_origins(hub["box"], _CCW[direction], gaps)
                for (top, left), color in zip(origins, hub["bead_colors"][direction]):
                    _stamp(expected, hub["mask"], top, left, color)
        try:
            feats = _analyze(inp, expected)
        except AssertionError:
            continue
        parsed = _parse(inp)
        if len(parsed["hubs"]) != len(scene.hubs):
            continue
        if not all(check(feats) for check in plan.get("require", ())):
            continue
        return {"input": inp, "output": expected}, feats
    return None


def _random_dirs(rng, count):
    return tuple(str(d) for d in rng.permutation(_CYCLE)[:count])


def _shape(rng, params):
    pool = params["shapes"]
    return pool[int(rng.integers(0, len(pool)))]


def _random_hub(rng, params, max_arms=3, k=None, crowded=False):
    weights = np.array([0.4, 0.4, 0.2][:max_arms])
    arms = int(rng.choice(np.arange(1, max_arms + 1), p=weights / weights.sum()))
    dirs = _random_dirs(rng, arms)
    arm = {}
    fit = 0.15 if crowded else 0.35
    for d in dirs:
        roll = rng.random()
        arm[d] = {"pref": "fit" if roll < fit else ("clip" if roll < fit + 0.15 else None)}
    return {"dirs": dirs, "shape": _shape(rng, params), "k": k or params["train_k"], "arm": arm}


def _puzzle_params(rng):
    regime = ("square", "square", "mixed", "mixed", "wide")[int(rng.integers(0, 5))]
    if regime == "square":
        shapes = [(3, 3)]
    elif regime == "mixed":
        shapes = [(3, 3), (3, 3), (3, 5), (5, 3)]
    else:
        shapes = [(3, 3), (3, 5), (5, 3), (5, 5)]
    bg_mode = ("zero", "zero", "fixed", "vary", "vary")[int(rng.integers(0, 5))]
    fixed_bg = 0 if bg_mode == "zero" else int(rng.integers(0, 10))
    shared = None
    if rng.random() < 0.25:
        shared = [int(c) for c in rng.choice([c for c in range(10) if c != fixed_bg],
                                             size=int(rng.integers(3, 6)), replace=False)]
    modes = ["pinwheel", "crowd", "long", "clipside", "shape", "big", "sparse", "plain"]
    weights = np.array([18, 14, 16, 14, 8, 12, 9, 9], dtype=float)
    count = 1 if rng.random() < 0.5 else 2
    chosen = [str(m) for m in rng.choice(modes, size=count, replace=False,
                                         p=weights / weights.sum())]
    return {"shapes": shapes, "regime": regime, "bg_mode": bg_mode, "fixed_bg": fixed_bg,
            "shared": shared, "max_hubs": int(rng.choice([1, 2, 2, 3])),
            "train_k": (1, int(rng.choice([2, 3, 3]))),
            "gap_hi": int(rng.choice([3, 4, 4, 5])), "modes": chosen}


def _pair_colors(rng, params):
    if params["bg_mode"] == "vary":
        bg = 0 if rng.random() < 0.35 else int(rng.integers(0, 10))
    else:
        bg = params["fixed_bg"]
    shared = params["shared"]
    if shared and bg in shared:
        shared = [c for c in shared if c != bg]
    return {"bg": bg, "beads": shared}


def _train_plans(rng, params, num_train):
    first = str(rng.choice(_CYCLE))
    second, third = _CCW[first], _CCW[_CCW[first]]
    fourth = _CCW[third]
    nonsquare = [s for s in params["shapes"] if s[0] != s[1]]
    demo_shape = (3, 3)
    if nonsquare and rng.random() < 0.5:
        demo_shape = nonsquare[int(rng.integers(0, len(nonsquare)))]
    demo = {"dirs": (first, second, third), "shape": demo_shape, "k": params["train_k"],
            "asym": True,
            "arm": {first: {"unequal": True, "pref": "fit"}, second: {"pref": None},
                    third: {"pref": None}}}
    if params["train_k"][1] < 2:
        demo["arm"][first] = {"pref": "fit"}
    plans = [{"hubs": [demo], "gap_hi": params["gap_hi"],
              "require": (lambda f: f["takeover"] and f["fit_multi"],)}]
    clip_hub = {"dirs": (fourth,) + (( _CCW[fourth],) if rng.random() < 0.4 else ()),
                "shape": _shape(rng, params), "k": params["train_k"],
                "arm": {fourth: {"pref": "partial"}}}
    if nonsquare and demo_shape[0] == demo_shape[1]:
        clip_hub["shape"] = nonsquare[int(rng.integers(0, len(nonsquare)))]
    second_hubs = [clip_hub]
    if params["max_hubs"] >= 2:
        second_hubs.append(_random_hub(rng, params, max_arms=2))
    plans.append({"hubs": second_hubs, "gap_hi": params["gap_hi"],
                  "require": (lambda f: f["partial"],)})
    for index in range(2, num_train):
        count = int(rng.integers(1, params["max_hubs"] + 1))
        plans.append({"hubs": [_random_hub(rng, params, max_arms=3 if count < 3 else 2,
                                           crowded=count >= 2) for _ in range(count)],
                      "gap_hi": params["gap_hi"]})
    # occasional extra hub on the demo pair
    if params["max_hubs"] >= 2 and rng.random() < 0.5:
        plans[0]["hubs"].append(_random_hub(rng, params, max_arms=2))
    return plans


def _test_plan(rng, params, mode, train_feats):
    train_max_hubs = max(f["hub_count"] for f in train_feats)
    train_max_k = max(max(f["token_counts"]) for f in train_feats)
    train_shapes = set().union(*(f["shapes"] for f in train_feats))
    train_sides = set().union(*(f["clip_sides"] for f in train_feats))
    gap_hi = params["gap_hi"]
    if mode == "pinwheel":
        hub = {"dirs": _CYCLE, "shape": (3, 3) if rng.random() < 0.6 else _shape(rng, params),
               "k": (1, max(2, train_max_k)), "arm": {}}
        hubs = [hub]
        if train_max_hubs >= 2 and rng.random() < 0.5:
            hubs.append(_random_hub(rng, params, max_arms=2, crowded=True))
        return {"hubs": hubs, "gap_hi": gap_hi, "size": "large",
                "require": (lambda f: max(f["arms_per_hub"]) == 4,)}
    if mode == "crowd" and train_max_hubs < 2:
        mode = "pinwheel" if rng.random() < 0.5 else "long"
    if mode == "crowd":
        count = min(4, train_max_hubs + 1)
        hubs = [_random_hub(rng, params, max_arms=2, k=(1, 2), crowded=True)
                for _ in range(count)]
        for hub in hubs:
            hub["shape"] = (3, 3) if rng.random() < 0.7 else hub["shape"]
        return {"hubs": hubs, "gap_hi": min(gap_hi, 3), "size": "large",
                "require": (lambda f, n=count: f["hub_count"] == n,)}
    if mode == "long":
        target = train_max_k + 1 + int(rng.integers(0, 2))
        d = str(rng.choice(_CYCLE))
        hub = {"dirs": (d,) + ((_CCW[_CCW[d]],) if rng.random() < 0.4 else ()),
               "shape": (3, 3), "k": (target, target), "arm": {}}
        hubs = [hub]
        if train_max_hubs >= 2 and rng.random() < 0.6:
            hubs.append(_random_hub(rng, params, max_arms=2, k=(1, 2), crowded=True))
        return {"hubs": hubs, "gap_hi": min(gap_hi, 3), "size": "large",
                "require": (lambda f, t=target: max(f["token_counts"]) >= t,)}
    if mode == "clipside":
        fresh = [d for d in _CYCLE if _SIDE[_CCW[d]] not in train_sides]
        if fresh:
            d = str(rng.choice(fresh))
            hub = {"dirs": (d,) + ((_CCW[d],) if rng.random() < 0.5 else ()),
                   "shape": _shape(rng, params), "k": (2, max(2, train_max_k)),
                   "arm": {d: {"pref": "vanish" if rng.random() < 0.4 else "partial"}}}
            hubs = [hub]
            if train_max_hubs >= 2 and rng.random() < 0.6:
                hubs.append(_random_hub(rng, params, max_arms=2, crowded=True))
            side = _SIDE[_CCW[d]]
            return {"hubs": hubs, "gap_hi": gap_hi,
                    "require": (lambda f, s=side: s in f["clip_sides"],)}
        mode = "long"
    if mode == "shape":
        unseen = [s for s in _SHAPES if s not in train_shapes]
        if params["regime"] != "square" and unseen and any(s[0] != s[1] for s in train_shapes):
            shape = unseen[int(rng.integers(0, len(unseen)))]
            hub = _random_hub(rng, params, max_arms=2)
            hub["shape"] = shape
            hubs = [hub]
            if train_max_hubs >= 2 and rng.random() < 0.5:
                hubs.append(_random_hub(rng, params, max_arms=2, crowded=True))
            return {"hubs": hubs, "gap_hi": gap_hi, "size": "large",
                    "require": (lambda f, s=shape: s in f["shapes"],)}
        mode = "big"
    if mode == "sparse":
        hub = _random_hub(rng, params, max_arms=2, k=(2, max(2, train_max_k)))
        hub["dirs"] = hub["dirs"][:1] if rng.random() < 0.6 else hub["dirs"]
        return {"hubs": [hub], "gap_hi": gap_hi}
    if mode == "big":
        count = max(1, min(3, train_max_hubs))
        hubs = [_random_hub(rng, params, max_arms=3 if count < 3 else 2,
                            k=(1, max(2, train_max_k)), crowded=count >= 2)
                for _ in range(count)]
        area = max(f["dims"][0] * f["dims"][1] for f in train_feats)
        arms = max(sum(f["arms_per_hub"]) for f in train_feats)
        return {"hubs": hubs, "gap_hi": gap_hi, "size": "large",
                "require": (lambda f, a=area, n=min(arms, 3):
                            f["dims"][0] * f["dims"][1] > a and sum(f["arms_per_hub"]) >= n,)}
    count = max(int(rng.integers(1, train_max_hubs + 1)), int(rng.integers(1, train_max_hubs + 1)))
    return {"hubs": [_random_hub(rng, params, max_arms=3 if count < 3 else 2,
                                 crowded=count >= 2) for _ in range(count)], "gap_hi": gap_hi}


def _training_ok(train_feats, test_feats, train_pairs):
    union = lambda key: set().union(*(f[key] for f in train_feats))
    if union("dirs") != set(_CYCLE):
        return False
    if not any(f["unequal"] for f in train_feats) or not any(f["fit_multi"] for f in train_feats):
        return False
    if not any(f["partial"] for f in train_feats) or not any(f["takeover"] for f in train_feats):
        return False
    if not any(f["asym"] for f in train_feats) or not any(f["color_diff"] for f in train_feats):
        return False
    if any(f["nonsquare"] for f in test_feats) and not any(f["nonsquare"] for f in train_feats):
        return False
    if max(f["hub_count"] for f in test_feats) >= 2 and max(f["hub_count"] for f in train_feats) < 2:
        return False
    for name in _VARIANTS:
        if all(np.array_equal(_variant(p["input"], name), p["output"]) for p in train_pairs):
            return False
    return True


def _check_counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("pair counts must be integers")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _attempt(rng, num_train, num_test):
    params = _puzzle_params(rng)
    plans = _train_plans(rng, params, num_train)
    train, train_feats, seen = [], [], set()
    for plan in plans:
        result = _build_pair(rng, plan, _pair_colors(rng, params))
        if result is None:
            return None
        pair, feats = result
        key = (pair["input"].shape, pair["input"].tobytes())
        if key in seen:
            return None
        seen.add(key)
        train.append(pair)
        train_feats.append(feats)
    test, test_feats = [], []
    for index in range(num_test):
        mode = params["modes"][index % len(params["modes"])]
        clean = mode in ("crowd", "shape", "big", "sparse", "plain") and rng.random() < 0.4
        result = None
        for attempt_mode in (mode, mode, "long", "pinwheel", "plain"):
            plan = _test_plan(rng, params, attempt_mode, train_feats)
            plan["require"] = tuple(plan.get("require", ())) + (lambda f: f["multi_visible"],)
            if clean and attempt_mode == mode:
                for hub in plan["hubs"]:
                    hub["arm"] = {d: dict(hub.get("arm", {}).get(d, {}), pref="fit")
                                  for d in hub["dirs"]}
                plan["require"] += (lambda f: not f["clip_sides"],)
            result = _build_pair(rng, plan, _pair_colors(rng, params))
            if result is not None:
                break
        if result is None:
            return None
        pair, feats = result
        key = (pair["input"].shape, pair["input"].tobytes())
        if key in seen:
            return None
        seen.add(key)
        test.append(pair)
        test_feats.append(feats)
    if not _training_ok(train_feats, test_feats, train):
        return None
    order = rng.permutation(len(train))
    return {"train": [train[int(i)] for i in order], "test": test}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is not None:
            validate_generated_puzzle(puzzle)
            return puzzle
    raise RuntimeError("could not build a puzzle for this seed")


# --------------------------------------------------------------------------
# Validation and self test.
# --------------------------------------------------------------------------

def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "train/test keys"
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "lists"
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "illegal pair counts"
    seen = set()
    feats = {"train": [], "test": []}
    pairs = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys"
            grids = {}
            for role in ("input", "output"):
                grid = np.asarray(pair[role])
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), "grid size"
                assert np.issubdtype(grid.dtype, np.integer), "integer cells"
                assert np.all((grid >= 0) & (grid <= 9)), "colours 0..9"
                grids[role] = grid.astype(int)
            key = (grids["input"].shape, grids["input"].tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            feats[split].append(_analyze(grids["input"], grids["output"]))
            pairs[split].append(grids)
    train = feats["train"]
    assert set().union(*(f["dirs"] for f in train)) == set(_CYCLE), "training must swing all four arm directions"
    assert any(f["unequal"] for f in train), "training must show unequal bead spacing"
    assert any(f["fit_multi"] for f in train), "training must show a fully visible unequal arm"
    assert any(f["partial"] for f in train), "training must show a clipped copy"
    assert any(f["takeover"] for f in train), "training must show a copy over an erased bead"
    assert any(f["asym"] for f in train), "training must show an asymmetric hub"
    if any(f["nonsquare"] for f in feats["test"]):
        assert any(f["nonsquare"] for f in train), "non-square test hub needs a training example"
    if max(f["hub_count"] for f in feats["test"]) >= 2:
        assert max(f["hub_count"] for f in train) >= 2, "multi-hub test needs a training example"
    for name in _VARIANTS:
        assert any(not np.array_equal(_variant(p["input"], name), p["output"])
                   for p in pairs["train"]), f"training does not exclude shortcut {name}"
    for f in feats["test"]:
        assert f["multi_visible"], "every test arm set must show bead spacing on two copies"
    for p in pairs["test"]:
        for name in ("no_swing", "no_unfold"):
            assert not np.array_equal(_variant(p["input"], name), p["output"]), \
                f"test pair does not need both parents ({name})"


def _puzzle_key(puzzle):
    return tuple((np.asarray(p["input"]).tobytes(), np.asarray(p["output"]).tobytes(),
                  np.asarray(p["input"]).shape) for s in ("train", "test") for p in puzzle[s])


def self_test() -> None:
    """Bounded check of 42 seeds: validity, both-parent necessity and spread."""
    shapes, dims, hub_counts, arm_counts, bead_counts = set(), set(), set(), set(), set()
    clip_sides, backgrounds, keys = set(), set(), set()
    n = {name: 0 for name in (
        "pinwheel", "vanish", "nonsquare", "bead_hub_color", "uniform_arms", "big_gap",
        "varied_bg", "longer_test", "crowd_test", "more_arms_test", "sparse_test",
        "new_side_test", "new_shape_test", "clean_test", "takeover_test")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(_puzzle_key(puzzle))
        train = [_analyze(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_analyze(p["input"], p["output"]) for p in puzzle["test"]]
        puzzle_bgs = set()
        for f in train + test:
            shapes |= f["shapes"]
            dims.add(f["dims"])
            hub_counts.add(f["hub_count"])
            arm_counts.update(f["arms_per_hub"])
            bead_counts.update(f["token_counts"])
            clip_sides |= f["clip_sides"]
            backgrounds.add(f["bg"])
            puzzle_bgs.add(f["bg"])
            n["pinwheel"] += max(f["arms_per_hub"]) == 4
            n["vanish"] += f["vanish"]
            n["nonsquare"] += f["nonsquare"]
            n["bead_hub_color"] += f["hub_color_bead"]
            n["big_gap"] += f["max_gap"] >= 4
        for split in ("train", "test"):
            for p in puzzle[split]:
                for hub in _parse(p["input"])["hubs"]:
                    for d, cells in hub["arms"].items():
                        gaps = _gaps(hub["box"], d, cells)
                        n["uniform_arms"] += len(cells) >= 2 and len(set(gaps)) == 1
                # Omitting either parent's mechanism must change every answer.
                for name in ("no_swing", "no_unfold"):
                    assert not np.array_equal(_variant(p["input"], name), p["output"]), name
        n["varied_bg"] += len(puzzle_bgs) > 1
        most_beads = max(max(f["token_counts"]) for f in train)
        most_hubs = max(f["hub_count"] for f in train)
        most_arms = max(max(f["arms_per_hub"]) for f in train)
        sides = set().union(*(f["clip_sides"] for f in train))
        seen_shapes = set().union(*(f["shapes"] for f in train))
        n["longer_test"] += any(max(f["token_counts"]) > most_beads for f in test)
        n["crowd_test"] += any(f["hub_count"] > most_hubs for f in test)
        n["more_arms_test"] += any(max(f["arms_per_hub"]) > most_arms for f in test)
        n["sparse_test"] += any(f["hub_count"] == 1 and sum(f["arms_per_hub"]) <= 2 for f in test)
        n["new_side_test"] += any(f["clip_sides"] - sides for f in test)
        n["new_shape_test"] += any(f["shapes"] - seen_shapes for f in test)
        n["clean_test"] += any(not f["clip_sides"] for f in test)
        n["takeover_test"] += any(f["takeover"] for f in test)
    assert len(keys) == 42, "seeds collide"
    assert len(shapes) == 4 and len(dims) >= 60, "shape diversity"
    assert hub_counts >= {1, 2, 3} and arm_counts == {1, 2, 3, 4}, "count diversity"
    assert len(bead_counts) >= 3 and clip_sides == {"top", "bottom", "left", "right"}
    assert len(backgrounds) >= 6 and n["varied_bg"] >= 8, "palette diversity"
    floors = {"pinwheel": 5, "vanish": 20, "nonsquare": 20, "bead_hub_color": 10,
              "uniform_arms": 20, "big_gap": 15, "longer_test": 2, "crowd_test": 2,
              "more_arms_test": 4, "sparse_test": 4, "new_side_test": 4, "new_shape_test": 2,
              "clean_test": 4, "takeover_test": 8}
    for name, floor in floors.items():
        assert n[name] >= floor, f"{name} too rare: {n[name]} < {floor}"
    first = generate_puzzle(7, 5, 2)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(7, 5, 2)), "nondeterministic"
    for counts in ((2, 1), (10, 4), (3, 2)):
        p = generate_puzzle(1000 + counts[0], *counts)
        assert (len(p["train"]), len(p["test"])) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(5, *counts)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {counts}")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
