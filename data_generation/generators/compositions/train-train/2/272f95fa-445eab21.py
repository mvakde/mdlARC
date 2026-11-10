"""Largest-frame cross partition.

Several one-cell-thick hollow rectangular frames lie on a uniform background.
The frame with the largest outer bounding-box area (parent B) is the anchor.
Its two border rows and two border columns are extended across the whole
grid in the anchor colour, producing the 2-horizontal x 2-vertical divider
partition of parent A.  Each of the five plus-shaped regions of that 3x3
partition (top arm, left arm, centre = the anchor's hollow, right arm,
bottom arm) is flooded with the colour of the frame lying inside it; a
plus-shaped region without a frame stays background.  The four corner
regions are left exactly as they were, including any frames inside them.
"""
from __future__ import annotations

import numpy as np

CROSS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))
ARMS = ((0, 1), (1, 0), (1, 2), (2, 1))


# ----------------------------------------------------------------------------
# Input analysis and the input-only solver
# ----------------------------------------------------------------------------

def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    order = np.lexsort((values, -counts))
    return int(values[order[0]])


def _components(grid, bg):
    """4-connected single-colour components of non-background cells."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            comps.append({"color": color, "cells": cells, "r0": min(ys), "r1": max(ys),
                          "c0": min(xs), "c1": max(xs)})
    for comp in comps:
        comp["h"] = comp["r1"] - comp["r0"] + 1
        comp["w"] = comp["c1"] - comp["c0"] + 1
        comp["area"] = comp["h"] * comp["w"]
    return comps


def _band(lo, hi, a, b):
    """Band index of span [lo, hi] relative to divider lines a < b (None if it straddles)."""
    if hi < a:
        return 0
    if lo > b:
        return 2
    if lo > a and hi < b:
        return 1
    return None


def _band_span(index, a, b, size):
    return ((0, a - 1), (a + 1, b - 1), (b + 1, size - 1))[index]


def _analyze(grid):
    grid = np.asarray(grid, dtype=int)
    bg = _background(grid)
    frames = _components(grid, bg)
    if not frames:
        raise ValueError("no frames")
    anchor = max(frames, key=lambda f: (f["area"], -f["r0"], -f["c0"]))
    for f in frames:
        if f is anchor:
            f["zone"] = None
            continue
        rb = _band(f["r0"], f["r1"], anchor["r0"], anchor["r1"])
        cb = _band(f["c0"], f["c1"], anchor["c0"], anchor["c1"])
        f["zone"] = None if rb is None or cb is None else (rb, cb)
    return bg, frames, anchor


def solve(input_grid):
    grid = np.asarray(input_grid, dtype=int)
    bg, frames, anchor = _analyze(grid)
    out = grid.copy()
    H, W = grid.shape
    a0, a1, b0, b1 = anchor["r0"], anchor["r1"], anchor["c0"], anchor["c1"]
    for f in frames:
        if f is anchor or f["zone"] not in CROSS:
            continue
        rs = _band_span(f["zone"][0], a0, a1, H)
        cs = _band_span(f["zone"][1], b0, b1, W)
        out[rs[0]:rs[1] + 1, cs[0]:cs[1] + 1] = f["color"]
    col = anchor["color"]
    out[a0, :] = col
    out[a1, :] = col
    out[:, b0] = col
    out[:, b1] = col
    return out


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _interior(h, w):
    return (h - 2) * (w - 2)


def _foil_ok(kind, fh, fw, h, w):
    if kind == "perim":
        return fh + fw > h + w
    if kind == "equal":
        return fh + fw == h + w
    if kind == "short":
        return min(fh, fw) > min(h, w)
    return True


def _pick_dims(rng, zh, zw, h, w, kind, small_bias):
    area, inner = h * w, _interior(h, w)
    cands = []
    for fh in range(3, zh + 1):
        for fw in range(3, zw + 1):
            if fh * fw >= area or _interior(fh, fw) >= inner:
                continue
            if not _foil_ok(kind, fh, fw, h, w):
                continue
            cands.append((fh, fw))
    if not cands:
        return None
    if kind is None and small_bias:
        small = [c for c in cands if max(c) <= 6]
        if small:
            cands = small
    return cands[int(rng.integers(len(cands)))]


def _ring_mask(shape, r0, c0, h, w):
    m = np.zeros(shape, dtype=bool)
    m[r0, c0:c0 + w] = True
    m[r0 + h - 1, c0:c0 + w] = True
    m[r0:r0 + h, c0] = True
    m[r0:r0 + h, c0 + w - 1] = True
    return m


def _dilate_rect(blocked, r0, c0, h, w):
    H, W = blocked.shape
    blocked[max(0, r0 - 1):min(H, r0 + h + 1), max(0, c0 - 1):min(W, c0 + w + 1)] = True


def _alt_clear(frames, shape, index):
    """True when frame `index`'s extended border lines hit no other frame."""
    H, W = shape
    other = np.zeros(shape, dtype=bool)
    for j, f in enumerate(frames):
        if j != index:
            other |= _ring_mask(shape, f["r0"], f["c0"], f["h"], f["w"])
    f = frames[index]
    lines = np.zeros(shape, dtype=bool)
    lines[f["r0"], :] = True
    lines[f["r0"] + f["h"] - 1, :] = True
    lines[:, f["c0"]] = True
    lines[:, f["c0"] + f["w"] - 1] = True
    return not np.any(lines & other)


def _try_pair(rng, spec):
    lo, hi = spec["size"]
    H = int(rng.integers(lo, hi + 1))
    W = int(rng.integers(lo, hi + 1))
    if spec.get("elong", True) and rng.random() < 0.25:  # occasional elongated canvas
        long_side = max(H, W)
        if long_side >= 18:
            short = int(rng.integers(10, long_side * 2 // 3 + 1))
            if rng.random() < 0.5:
                H, W = long_side, short
            else:
                H, W = short, long_side
    need_center = spec["center"] == "need"
    min_side = 7 if need_center else 5
    max_h = min(15, H - 2, max(min_side, int(H * 0.6)))
    max_w = min(15, W - 2, max(min_side, int(W * 0.6)))
    if max_h < min_side or max_w < min_side:
        return None
    h = int(rng.integers(min_side, max_h + 1))
    w = int(rng.integers(min_side, max_w + 1))
    if spec["foil"] in ("perim", "equal") and rng.random() < 0.7:
        # compact anchors leave room for long thin foils
        h = min(h, int(rng.integers(min_side, min_side + 4)))
        w = min(w, int(rng.integers(min_side, min_side + 4)))
    elif spec["foil"] == "short" and rng.random() < 0.8:
        # a thin, long anchor can lose the short-side comparison yet win on area
        if rng.random() < 0.5:
            h = int(rng.integers(min_side, min_side + 2))
        else:
            w = int(rng.integers(min_side, min_side + 2))
    if spec["edge"]:
        # anchor hugs one or two edges: some arms are one or two cells thick
        top = int(rng.integers(1, 3)) if rng.random() < 0.5 else int(rng.integers(1, H - h))
        left = int(rng.integers(1, 3)) if rng.random() < 0.5 else int(rng.integers(1, W - w))
        if rng.random() < 0.5:
            top = H - h - top if H - h - top >= 1 else top
        if rng.random() < 0.5:
            left = W - w - left if W - w - left >= 1 else left
        if min(top, H - h - top, left, W - w - left) > 2:
            return None
    else:
        top = int(rng.integers(3, H - h - 2)) if H - h - 2 > 3 else int(rng.integers(1, H - h))
        left = int(rng.integers(3, W - w - 2)) if W - w - 2 > 3 else int(rng.integers(1, W - w))
    if not (1 <= top <= H - h - 1 and 1 <= left <= W - w - 1):
        return None
    a0, a1, b0, b1 = top, top + h - 1, left, left + w - 1

    blocked = _ring_mask((H, W), a0, b0, h, w)
    grown = blocked.copy()
    grown[1:, :] |= blocked[:-1, :]
    grown[:-1, :] |= blocked[1:, :]
    g2 = grown.copy()
    g2[:, 1:] |= grown[:, :-1]
    g2[:, :-1] |= grown[:, 1:]
    blocked = g2
    # nothing else may sit on the future divider lines
    line_rows = (a0, a1)
    line_cols = (b0, b1)

    frames = [{"r0": a0, "c0": b0, "h": h, "w": w, "zone": None}]

    # zone plan
    arms = list(ARMS)
    rng.shuffle(arms)
    n_arms = int(rng.integers(spec["arms"][0], spec["arms"][1] + 1))
    plan = arms[:n_arms]
    if need_center:
        plan = [(1, 1)] + plan
    elif spec["center"] == "maybe" and h >= 7 and w >= 7 and rng.random() < spec.get("center_p", 0.35):
        plan = [(1, 1)] + plan
    corners = list(CORNERS)
    rng.shuffle(corners)
    n_corner = int(rng.integers(spec["corners"][0], spec["corners"][1] + 1))
    plan += corners[:n_corner]

    foil_zone = None
    if spec["foil"]:
        cand = [z for z in plan if z != (1, 1)]
        if not cand:
            return None
        foil_zone = cand[int(rng.integers(len(cand)))]

    placed_zones = []
    for zone in plan:
        rs = _band_span(zone[0], a0, a1, H)
        cs = _band_span(zone[1], b0, b1, W)
        zh, zw = rs[1] - rs[0] + 1, cs[1] - cs[0] + 1
        if zh < 3 or zw < 3:
            continue
        kind = spec["foil"] if zone == foil_zone else None
        ok = False
        for _ in range(12):
            dims = _pick_dims(rng, zh, zw, h, w, kind, rng.random() < 0.55)
            if dims is None:
                break
            fh, fw = dims
            r = int(rng.integers(rs[0], rs[1] - fh + 2))
            c = int(rng.integers(cs[0], cs[1] - fw + 2))
            if np.any(blocked[r:r + fh, c:c + fw]):
                continue
            if any(r <= x <= r + fh - 1 for x in line_rows) or any(c <= x <= c + fw - 1 for x in line_cols):
                continue
            frames.append({"r0": r, "c0": c, "h": fh, "w": fw, "zone": zone})
            _dilate_rect(blocked, r, c, fh, fw)
            placed_zones.append(zone)
            ok = True
            break
        if not ok and zone in (foil_zone,) + ((1, 1),) * need_center:
            return None
    cross_count = sum(z in CROSS for z in placed_zones)
    corner_count = sum(z in CORNERS for z in placed_zones)
    if cross_count < 1 or cross_count < spec["min_cross"]:
        return None
    if corner_count < spec["corners"][0]:
        return None
    for f in frames:
        f["r1"] = f["r0"] + f["h"] - 1
        f["c1"] = f["c0"] + f["w"] - 1
    if spec["corner_clear"]:
        if not any(frames[i]["zone"] in CORNERS and _alt_clear(frames, (H, W), i)
                   for i in range(1, len(frames))):
            return None

    # colours
    bg = spec["bg"]
    pool = [c for c in range(10) if c != bg]
    rng.shuffle(pool)
    anchor_color = pool[0]
    rest = pool[1:]
    colors = [anchor_color]
    for i in range(1, len(frames)):
        colors.append(rest[(i - 1) % len(rest)])
    others = list(range(1, len(frames)))
    if spec["share"] and len(others) >= 2:
        rng.shuffle(others)
        i, j = others[0], others[1]
        colors[j] = colors[i]
    for f, col in zip(frames, colors):
        f["color"] = int(col)

    grid = np.full((H, W), bg, dtype=int)
    for f in frames:
        grid[_ring_mask((H, W), f["r0"], f["c0"], f["h"], f["w"])] = f["color"]
    if not _input_ok(grid):
        return None
    return grid


def _input_ok(grid):
    try:
        _check_pair_input(grid)
    except AssertionError:
        return False
    return True


def _check_pair_input(grid):
    """Assert every family invariant on one input grid."""
    grid = np.asarray(grid, dtype=int)
    H, W = grid.shape
    assert 1 <= H <= 30 and 1 <= W <= 30
    values, counts = np.unique(grid, return_counts=True)
    top = sorted(counts)[::-1]
    assert len(top) >= 2 and top[0] > top[1], "background must be strictly most frequent"
    bg, frames, anchor = _analyze(grid)
    assert len(frames) >= 2, "need an anchor and at least one other frame"
    for f in frames:
        assert f["h"] >= 3 and f["w"] >= 3, "frames are at least 3x3"
        ring = _ring_mask(grid.shape, f["r0"], f["c0"], f["h"], f["w"])
        assert len(f["cells"]) == int(ring.sum()), "component must be a complete one-cell ring"
        assert all(ring[y, x] for y, x in f["cells"]), "component must be a rectangle outline"
    areas = sorted((f["area"] for f in frames), reverse=True)
    assert areas[0] > areas[1], "unique largest outer area"
    inner = max(_interior(f["h"], f["w"]) for f in frames if f is not anchor)
    assert _interior(anchor["h"], anchor["w"]) > inner, "outer and hollow rankings agree"
    assert anchor["r0"] >= 1 and anchor["c0"] >= 1 and anchor["r1"] <= H - 2 and anchor["c1"] <= W - 2
    # separation: no two frames touch, even diagonally
    owner = -np.ones(grid.shape, dtype=int)
    for i, f in enumerate(frames):
        for y, x in f["cells"]:
            owner[y, x] = i
    for i, f in enumerate(frames):
        for y, x in f["cells"]:
            win = owner[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            assert np.all((win == -1) | (win == i)), "frames must be separated by background"
    used = {}
    for f in frames:
        if f is anchor:
            continue
        assert f["zone"] is not None, "every frame lies inside one region of the partition"
        if f["zone"] in CROSS:
            assert f["zone"] not in used, "at most one frame per plus-shaped region"
            used[f["zone"]] = True
        assert f["color"] != anchor["color"], "anchor colour is unique"
    assert used, "at least one frame is flooded"
    return bg, frames, anchor


def _pair_features(grid):
    bg, frames, anchor = _analyze(grid)
    others = [f for f in frames if f is not anchor]
    ab = anchor["h"] + anchor["w"]
    feats = {
        "perim": any(f["h"] + f["w"] > ab for f in others),
        "equal": any(f["h"] + f["w"] == ab for f in others),
        "short": any(min(f["h"], f["w"]) > min(anchor["h"], anchor["w"]) for f in others),
        "corner": any(f["zone"] in CORNERS for f in others),
        "center": any(f["zone"] == (1, 1) for f in others),
        "empty_arm": any(z not in {f["zone"] for f in others} for z in ARMS),
        "share": len({f["color"] for f in others}) < len(others),
        "count": len(frames),
        "bg": bg,
    }
    shape = grid.shape
    fl = [dict(f) for f in frames]
    idx = frames.index(anchor)
    feats["alt_clear"] = any(_alt_clear(fl, shape, i) for i in range(len(fl)) if i != idx)
    return feats


def _check_counts(num_train, num_test):
    if isinstance(num_train, bool) or isinstance(num_test, bool):
        raise ValueError("counts must be integers")
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("need 2..10 train, 1..4 test, and num_train > num_test")


PROFILES = ("grow", "shrink", "crowd", "foil", "edge", "steady")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    num_train, num_test = int(num_train), int(num_test)
    rng = np.random.default_rng(seed)
    profile = PROFILES[int(rng.integers(len(PROFILES)))]
    vary_bg = rng.random() < 0.5
    base_bg = int(rng.integers(10)) if rng.random() < 0.6 else 0
    use_center = rng.random() < 0.7
    use_share = rng.random() < 0.4
    total = num_train + num_test

    idx = list(range(num_train))
    perim_at = int(rng.choice(idx))
    clear_at = int(rng.choice(idx))
    center_at = int(rng.choice([j for j in idx if j != perim_at])) if use_center else -1
    share_at = int(rng.choice(idx)) if use_share else -1
    free = [j for j in idx if j not in (perim_at, center_at, clear_at)]
    short_at = int(rng.choice(free)) if free and rng.random() < 0.6 else -1

    bgs = []
    for i in range(total):
        if vary_bg and i > 0:
            bgs.append(int(rng.integers(10)))
        else:
            bgs.append(base_bg)
    if vary_bg and all(b == bgs[0] for b in bgs[:num_train]):
        bgs[1] = (bgs[0] + 1 + int(rng.integers(9))) % 10

    pairs = []
    seen = set()
    for i in range(total):
        test = i >= num_train
        spec = {"size": (12, 24), "arms": (2, 4), "corners": (0, 3), "min_cross": 2,
                "foil": None, "edge": False, "center": "maybe" if use_center else "no",
                "corner_clear": False, "share": False, "bg": bgs[i]}
        if not test:
            if i == perim_at:
                spec["foil"] = "perim"
            elif i == short_at:
                spec["foil"] = "short"
            if i == clear_at:
                spec["corner_clear"] = True
                spec["corners"] = (1, 3)
            if i == center_at:
                spec["center"] = "need"
            if i == share_at:
                spec["share"] = True
            elif use_share and rng.random() < 0.3:
                spec["share"] = True
        else:
            spec["share"] = use_share and rng.random() < 0.6
            spec["center_p"] = 0.6
            if rng.random() < 0.6:
                spec["corners"] = (1, 3)
            if rng.random() < 0.5:
                spec["foil"] = str(rng.choice(["perim", "short"]))
        if profile == "grow":
            spec["size"] = (22, 30) if test else (12, 18)
            spec["elong"] = not test
            if test:
                spec["arms"] = (2, 4)
        elif profile == "shrink":
            spec["size"] = (12, 16) if test else (20, 30)
            spec["elong"] = test
            if test:
                spec["corners"] = (0, 1)
                spec["foil"] = None
        elif profile == "crowd":
            if test:
                spec["size"] = (22, 30)
                spec["arms"] = (3, 4)
                spec["corners"] = (2, 3)
                spec["min_cross"] = 3
            else:
                spec["arms"] = (1, 2)
                spec["min_cross"] = 1
                spec["corners"] = (spec["corners"][0], 1)
        elif profile == "foil":
            if test:
                spec["foil"] = str(rng.choice(["equal", "short"]))
                spec["size"] = (18, 30)
        elif profile == "edge":
            if test:
                spec["edge"] = True
                spec["size"] = (16, 28)
            else:
                spec["size"] = (14, 26)
        if spec["foil"] and spec["size"][1] < 22 and profile != "shrink":
            spec["size"] = (spec["size"][0], 22)
        if spec["foil"] and (spec["corner_clear"] or spec["center"] == "need"):
            spec["min_cross"] = 1
        grid = None
        for attempt in range(600):
            if attempt == 300:
                # relax an optional density target; required evidence is never dropped
                spec["min_cross"] = 1
            if attempt == 450 and not (not test and i == perim_at):
                # optional foils (short-side training extra, test foils) may be dropped
                spec["foil"] = None
            grid = _try_pair(rng, spec)
            if grid is None:
                continue
            key = grid.tobytes() + bytes(grid.shape)
            if key in seen:
                grid = None
                continue
            feats = _pair_features(grid)
            if spec["foil"] and not feats[spec["foil"]]:
                grid = None
                continue
            if spec["center"] == "need" and not feats["center"]:
                grid = None
                continue
            if spec["share"] and not feats["share"]:
                grid = None
                continue
            seen.add(key)
            break
        if grid is None:
            raise RuntimeError(f"generation failed for seed {seed} pair {i} profile {profile} spec {spec}")
        pairs.append({"input": grid.tolist(), "output": solve(grid).tolist()})
    puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
    _fix_center_evidence(puzzle)
    return puzzle


def _fix_center_evidence(puzzle):
    # generation never puts a nested centre frame into a test unless training has one
    test_center = any(_pair_features(np.array(p["input"]))["center"] for p in puzzle["test"])
    train_center = any(_pair_features(np.array(p["input"]))["center"] for p in puzzle["train"])
    if test_center and not train_center:
        raise RuntimeError("centre fill used in test without training evidence")


# ----------------------------------------------------------------------------
# Validation and self test
# ----------------------------------------------------------------------------

def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    feats = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            assert inp.ndim == 2 and inp.shape == out.shape
            assert inp.min() >= 0 and inp.max() <= 9 and out.min() >= 0 and out.max() <= 9
            key = inp.tobytes() + bytes(inp.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            _check_pair_input(inp)
            assert np.array_equal(solve(inp), out), "output must follow the rule"
            assert not np.array_equal(inp, out)
            feats[split].append(_pair_features(inp))
    tr = feats["train"]
    assert any(f["perim"] for f in tr), "training needs a border-count foil"
    assert any(f["corner"] for f in tr), "training needs an untouched corner frame"
    assert any(f["alt_clear"] for f in tr), "training must refute 'anchor = frame with clear lines'"
    assert any(f["empty_arm"] for f in tr), "training needs an empty arm"
    if any(f["center"] for f in feats["test"]):
        assert any(f["center"] for f in tr), "centre fill must be demonstrated in training"
    if any(f["share"] for f in feats["test"]):
        assert any(f["share"] for f in tr), "shared colours must be demonstrated in training"
    return True


def _shortcut_outputs(grid):
    """Outputs of plausible wrong rules, used by self_test to check that demos refute them."""
    grid = np.asarray(grid, dtype=int)
    bg, frames, anchor = _analyze(grid)
    results = {}
    # B omitted: anchor chosen by most border cells (ties -> first)
    by_border = max(frames, key=lambda f: (len(f["cells"]), -f["r0"], -f["c0"]))
    results["border_anchor"] = _render(grid, frames, by_border)
    by_side = max(frames, key=lambda f: (max(f["h"], f["w"]), -f["r0"], -f["c0"]))
    results["longest_side_anchor"] = _render(grid, frames, by_side)
    # A omitted: every frame floods its region, corners included
    results["flood_corners"] = _render(grid, frames, anchor, corners=True)
    # A omitted: arms take the anchor colour instead of their frame's colour
    results["anchor_colour_arms"] = _render(grid, frames, anchor, arm_color=anchor["color"])
    return results


def _render(grid, frames, anchor, corners=False, arm_color=None):
    H, W = grid.shape
    out = grid.copy()
    a0, a1, b0, b1 = anchor["r0"], anchor["r1"], anchor["c0"], anchor["c1"]
    for f in frames:
        if f is anchor:
            continue
        rb = _band(f["r0"], f["r1"], a0, a1)
        cb = _band(f["c0"], f["c1"], b0, b1)
        if rb is None or cb is None:
            continue
        if (rb, cb) in CROSS or corners:
            rs = _band_span(rb, a0, a1, H)
            cs = _band_span(cb, b0, b1, W)
            out[rs[0]:rs[1] + 1, cs[0]:cs[1] + 1] = f["color"] if arm_color is None else arm_color
    out[a0, :] = anchor["color"]
    out[a1, :] = anchor["color"]
    out[:, b0] = anchor["color"]
    out[:, b1] = anchor["color"]
    return out


def self_test():
    shapes, counts, profiles_seen, bgs_varied = set(), set(), set(), 0
    refuted = {"border_anchor": 0, "longest_side_anchor": 0, "flood_corners": 0, "anchor_colour_arms": 0}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        assert all(np.array_equal(np.asarray(a["input"]), np.asarray(b["input"]))
                   for a, b in zip(puzzle["train"] + puzzle["test"], again["train"] + again["test"]))
        signatures.add(str(puzzle))
        bgset = set()
        for pair in puzzle["train"] + puzzle["test"]:
            inp = np.asarray(pair["input"])
            shapes.add(inp.shape)
            f = _pair_features(inp)
            counts.add(f["count"])
            bgset.add(f["bg"])
        bgs_varied += len(bgset) > 1
        for name in refuted:
            if any(not np.array_equal(_shortcut_outputs(np.asarray(p["input"]))[name], np.asarray(p["output"]))
                   for p in puzzle["train"]):
                refuted[name] += 1
    assert len(signatures) == 42
    assert len(shapes) >= 40, shapes
    assert len(counts) >= 4, counts
    assert bgs_varied >= 5
    # Each shortcut must be contradicted by training in every puzzle.
    for name, n in refuted.items():
        assert n == 42, (name, n)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (3, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return True


if __name__ == "__main__":
    self_test()
    print("self_test ok")
