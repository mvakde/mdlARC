"""Keyed landing pads: falling packet strips are caught only by solid 2x2 pads of their leading color.

A packet (k vertical strips of height h) hangs from the top boundary.  The packet is
removed and every strip falls straight down its own column, independently.  A strip
is caught only by a *separate solid 2x2 square* whose color equals the strip's
*bottom (leading) cell*.  It passes through every other object: same-colored
fragments that are not complete squares, complete squares of other colors, and
squares whose color appears only higher up in the strip.  A caught strip rests
directly on top of its pad; an uncaught strip comes to rest on the bottom boundary.
Strips keep their top-to-bottom color order.  Every other cell is unchanged.

Standard library + NumPy only.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))

FRAGMENTS = {
    "dot": ((0, 0),),
    "dom_h": ((0, 0), (0, 1)),
    "dom_v": ((0, 0), (1, 0)),
    "line3_h": ((0, 0), (0, 1), (0, 2)),
    "line3_v": ((0, 0), (1, 0), (2, 0)),
    "ltri_a": ((0, 0), (1, 0), (1, 1)),
    "ltri_b": ((0, 0), (0, 1), (1, 0)),
    "ltri_c": ((0, 0), (0, 1), (1, 1)),
    "ltri_d": ((0, 1), (1, 0), (1, 1)),
    "bar4_h": ((0, 0), (0, 1), (0, 2), (0, 3)),
    "bar4_v": ((0, 0), (1, 0), (2, 0), (3, 0)),
    "t_down": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "t_up": ((0, 1), (1, 0), (1, 1), (1, 2)),
    "t_right": ((0, 0), (1, 0), (2, 0), (1, 1)),
    "t_left": ((0, 1), (1, 0), (1, 1), (2, 1)),
    "s_h": ((0, 1), (0, 2), (1, 0), (1, 1)),
    "z_h": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "s_v": ((0, 0), (1, 0), (1, 1), (2, 1)),
    "z_v": ((0, 1), (1, 0), (1, 1), (2, 0)),
    "l4_a": ((0, 0), (1, 0), (2, 0), (2, 1)),
    "l4_b": ((0, 1), (1, 1), (2, 0), (2, 1)),
    "l4_c": ((0, 0), (0, 1), (0, 2), (1, 0)),
    "l4_d": ((0, 0), (0, 1), (0, 2), (1, 2)),
}
FRAGMENT_NAMES = tuple(FRAGMENTS)
# Near-miss fragments used for the "same color but not a complete square" decoy.
NEAR_MISS = ("ltri_a", "ltri_b", "ltri_c", "ltri_d", "dom_h", "dom_v", "line3_h", "s_h", "z_h", "t_down", "bar4_h")

ROLES = ("shape", "color", "order", "bottom", "double", "deep", "free")
SHORTCUTS = ("any", "color", "square", "anystrip", "top", "never", "lowest", "rigid")


# ---------------------------------------------------------------------------
# Input-only analysis and solver
# ---------------------------------------------------------------------------

def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _packet(grid, bg):
    """Return (c0, k, h) of the packet hanging from row 0, or None."""
    cols = np.flatnonzero(grid[0] != bg)
    if cols.size == 0:
        return None
    c0, c1 = int(cols.min()), int(cols.max())
    h = 0
    while h < grid.shape[0] and np.all(grid[h, c0:c1 + 1] != bg):
        h += 1
    return c0, c1 - c0 + 1, h


def _components(grid, bg, skip):
    """8-connected same-color components of non-background cells outside `skip`."""
    H, W = grid.shape
    label = -np.ones((H, W), dtype=int)
    comps = []
    for r in range(H):
        for c in range(W):
            if grid[r, c] == bg or skip[r, c] or label[r, c] >= 0:
                continue
            color = int(grid[r, c])
            idx = len(comps)
            stack, cells = [(r, c)], []
            label[r, c] = idx
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and label[yy, xx] < 0 and not skip[yy, xx] \
                                and grid[yy, xx] == color:
                            label[yy, xx] = idx
                            stack.append((yy, xx))
            rows = [p[0] for p in cells]
            colsx = [p[1] for p in cells]
            square = len(cells) == 4 and max(rows) - min(rows) == 1 and max(colsx) - min(colsx) == 1
            comps.append({"color": color, "cells": cells, "square": square, "top": min(rows)})
    return label, comps


def _analyze(grid):
    grid = np.asarray(grid, dtype=int)
    bg = _background(grid)
    pk = _packet(grid, bg)
    if pk is None:
        return None
    c0, k, h = pk
    skip = np.zeros(grid.shape, dtype=bool)
    skip[:h, c0:c0 + k] = True
    label, comps = _components(grid, bg, skip)
    strips = [tuple(int(v) for v in grid[:h, c0 + i]) for i in range(k)]
    return {"grid": grid, "bg": bg, "c0": c0, "k": k, "h": h, "label": label, "comps": comps, "strips": strips}


def _column_objects(info, col):
    """Objects met by a strip falling down column `col`, in order: (row, comp_index)."""
    grid, label, h = info["grid"], info["label"], info["h"]
    seen, out = set(), []
    for r in range(h, grid.shape[0]):
        idx = int(label[r, col])
        if idx >= 0 and idx not in seen:
            seen.add(idx)
            out.append((r, idx))
    return out


def _landing_rows(info, mode="truth"):
    """Top row of each strip's resting place under a given (true or shortcut) catch rule."""
    H = info["grid"].shape[0]
    h = info["h"]
    tops = []
    for i, seq in enumerate(info["strips"]):
        col = info["c0"] + i
        key = seq[-1]
        hits = []
        for r, idx in _column_objects(info, col):
            comp = info["comps"][idx]
            if mode in ("truth", "lowest", "rigid"):
                ok = comp["square"] and comp["color"] == key
            elif mode == "any":
                ok = True
            elif mode == "color":
                ok = comp["color"] == key
            elif mode == "square":
                ok = comp["square"]
            elif mode == "anystrip":
                ok = comp["square"] and comp["color"] in seq
            elif mode == "top":
                ok = comp["square"] and comp["color"] == seq[0]
            elif mode == "never":
                ok = False
            else:
                raise ValueError(mode)
            if ok:
                hits.append(r)
        if hits:
            tops.append((hits[-1] if mode == "lowest" else hits[0]) - h)
        else:
            tops.append(H - h)
    if mode == "rigid":
        drop = min(tops)
        tops = [drop] * len(tops)
    return tops


def _render(info, tops):
    out = info["grid"].copy()
    c0, k, h = info["c0"], info["k"], info["h"]
    out[:h, c0:c0 + k] = info["bg"]
    for i, seq in enumerate(info["strips"]):
        for j, color in enumerate(seq):
            out[tops[i] + j, c0 + i] = color
    return out


def _predict(grid, mode="truth"):
    info = _analyze(grid)
    if info is None:
        return np.asarray(grid, dtype=int).copy()
    return _render(info, _landing_rows(info, mode))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Drop each packet strip until a solid 2x2 square of its bottom color catches it."""
    return _predict(np.asarray(input_grid, dtype=int), "truth")


# ---------------------------------------------------------------------------
# Strip features (used by generation checks and validation)
# ---------------------------------------------------------------------------

def _strip_features(info):
    """Per strip: what it passed before resting and how it came to rest."""
    tops = _landing_rows(info, "truth")
    h = info["h"]
    feats = []
    for i, seq in enumerate(info["strips"]):
        col = info["c0"] + i
        key = seq[-1]
        caught = tops[i] != info["grid"].shape[0] - h
        catch_row = tops[i] + h
        passed, below_match = [], False
        for r, idx in _column_objects(info, col):
            comp = info["comps"][idx]
            if caught and r >= catch_row:
                if r > catch_row and comp["square"] and comp["color"] == key:
                    below_match = True
                continue
            passed.append(comp)
        f = {
            "caught": caught,
            "passed": len(passed),
            "shape": any(c["color"] == key and not c["square"] for c in passed),
            "color": any(c["square"] and c["color"] not in seq for c in passed),
            "order": any(c["square"] and c["color"] == seq[0] != key for c in passed),
            "double": caught and below_match,
        }
        roles = {"free"}
        if caught and f["shape"]:
            roles.add("shape")
        if caught and f["color"]:
            roles.add("color")
        if caught and f["order"]:
            roles.add("order")
        if not caught and len(passed) >= 1:
            roles.add("bottom")
        if f["double"]:
            roles.add("double")
        if caught and len(passed) >= 3:
            roles.add("deep")
        f["roles"] = roles
        feats.append(f)
    return feats


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

class _Canvas:
    def __init__(self, H, W, bg, h):
        self.H, self.W, self.bg, self.h = H, W, bg, h
        self.grid = np.full((H, W), bg, dtype=int)
        self.obj = np.zeros((H, W), dtype=bool)
        self.reserved = np.zeros((H, W), dtype=bool)

    def free(self, cells, top_min):
        for r, c in cells:
            if not (top_min <= r < self.H and 0 <= c < self.W) or self.reserved[r, c]:
                return False
            if self.obj[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].any():
                return False
        return True

    def put(self, cells, color):
        for r, c in cells:
            self.grid[r, c] = color
            self.obj[r, c] = True


def _place_in_column(rng, cv, col, shape, color, cursor, catch_h=None, max_rows=None):
    """Place `shape` so it covers column `col` with top row >= cursor. Returns bottom row or None."""
    H, W = cv.H, cv.W
    offsets = sorted({dc for _, dc in shape})
    tops = list(range(cursor, H))
    if max_rows is not None:
        tops = tops[:max_rows]
    for r in tops:
        dcs = list(offsets)
        rng.shuffle(dcs)
        for dc in dcs:
            cells = [(r + a, col - dc + b) for a, b in shape]
            if not cv.free(cells, cv.h + 1):
                continue
            if catch_h is not None:
                # footprint above the pad plus a one-cell halo must stay empty (reserved)
                if r - catch_h < 0:
                    continue
                halo = [(rr, cc) for rr in range(max(r - catch_h - 1, 0), r)
                        for cc in (col - 1, col, col + 1) if 0 <= cc < W]
                if any(cv.obj[rr, cc] for rr, cc in halo):
                    continue
                for rr, cc in halo:
                    cv.reserved[rr, cc] = True
            cv.put(cells, color)
            return max(a for a, _ in shape) + r
    return None


def _column_plan(rng, role, seq, palette):
    """List of (kind, color) items from top to bottom, ending in a catch pad or nothing."""
    key = seq[-1]
    others = [c for c in palette if c not in seq]
    order_ok = seq[0] != key

    def decoy():
        options = ["shape", "plain"]
        if others:
            options.append("color")
        if order_ok:
            options.append("order")
        kind = options[int(rng.integers(len(options)))]
        if kind == "shape":
            return ("frag", key)
        if kind == "color":
            return ("square", others[int(rng.integers(len(others)))])
        if kind == "order":
            return ("square", seq[0])
        return ("frag", palette[int(rng.integers(len(palette)))])

    if role == "shape":
        items = [("frag", key)] + [decoy() for _ in range(int(rng.integers(0, 2)))]
        rng.shuffle(items)
        return items + [("catch", key)]
    if role == "double":
        items = [("frag", key)] if rng.random() < 0.6 else [decoy()]
        return items + [("catch", key), ("square", key)]
    if role == "color":
        if not others:
            return None
        items = [("square", others[int(rng.integers(len(others)))])]
        if rng.random() < 0.4:
            items.append(decoy())
        rng.shuffle(items)
        return items + [("catch", key)]
    if role == "order":
        if not order_ok:
            return None
        items = [("square", seq[0])]
        if rng.random() < 0.3:
            items.append(decoy())
        rng.shuffle(items)
        return items + [("catch", key)]
    if role == "bottom":
        n = int(rng.integers(1, 3))
        return [decoy() for _ in range(n)]
    if role == "deep":
        n = int(rng.integers(3, 5))
        items = [decoy() for _ in range(n)]
        if not any(it == ("frag", key) for it in items):
            items[int(rng.integers(n))] = ("frag", key)
        return items + [("catch", key)]
    # free
    n = int(rng.integers(0, 3))
    items = [decoy() for _ in range(n)]
    if rng.random() < 0.6:
        items.append(("catch", key))
    return items


def _make_strips(rng, roles, h, keys, palette):
    strips = []
    for role in roles:
        key = keys[int(rng.integers(len(keys)))]
        seq = [palette[int(rng.integers(len(palette)))] for _ in range(h - 1)] + [key]
        if role == "order" and seq[0] == key:
            choices = [c for c in palette if c != key]
            seq[0] = choices[int(rng.integers(len(choices)))]
        if rng.random() < 0.15 and role not in ("order",):
            seq = [key] * h
        strips.append(tuple(int(v) for v in seq))
    return strips


def _build_pair(rng, spec):
    """One attempt to construct an input grid for a pair spec; returns grid or None."""
    H, W, h, k, c0 = spec["H"], spec["W"], spec["h"], spec["k"], spec["c0"]
    bg, palette, keys, roles = spec["bg"], spec["palette"], spec["keys"], spec["roles"]
    strips = _make_strips(rng, roles, h, keys, palette)
    cv = _Canvas(H, W, bg, h)
    plans = []
    for i, role in enumerate(roles):
        plan = _column_plan(rng, role, strips[i], palette)
        if plan is None:
            return None
        plans.append(plan)
    # bottom footprints of strips that should reach the floor
    for i, plan in enumerate(plans):
        if not any(kind == "catch" for kind, _ in plan):
            for r in range(H - h - 1, H):
                for cc in (c0 + i - 1, c0 + i, c0 + i + 1):
                    if 0 <= cc < W:
                        cv.reserved[r, cc] = True
    order = list(range(k))
    rng.shuffle(order)
    for i in order:
        col = c0 + i
        cursor = h + 1 + int(rng.integers(0, 3))
        for kind, color in plans[i]:
            if kind == "frag":
                name = NEAR_MISS[int(rng.integers(len(NEAR_MISS)))] if color == strips[i][-1] and rng.random() < 0.7 \
                    else FRAGMENT_NAMES[int(rng.integers(len(FRAGMENT_NAMES)))]
                bottom = _place_in_column(rng, cv, col, FRAGMENTS[name], color, cursor, max_rows=7)
            elif kind == "square":
                bottom = _place_in_column(rng, cv, col, SQUARE, color, cursor, max_rows=7)
            else:  # catch pad, leave room for the strip above it
                bottom = _place_in_column(rng, cv, col, SQUARE, color, cursor + int(rng.integers(0, 2)),
                                          catch_h=h, max_rows=12)
            if bottom is None:
                if kind == "catch":
                    return None
                continue  # an optional decoy that does not fit; role checks decide later
            cursor = bottom + 2 + int(rng.integers(0, 2))
    # filler objects anywhere below the gap row
    n_fill = int(rng.integers(spec["fill"][0], spec["fill"][1] + 1))
    fill_colors = list(palette) + list(spec.get("extra", []))
    for _ in range(n_fill):
        color = fill_colors[int(rng.integers(len(fill_colors)))]
        shape = SQUARE if rng.random() < 0.4 else FRAGMENTS[FRAGMENT_NAMES[int(rng.integers(len(FRAGMENT_NAMES)))]]
        for _try in range(12):
            r = int(rng.integers(h + 1, H))
            c = int(rng.integers(0, W))
            cells = [(r + a, c + b) for a, b in shape]
            if cv.free(cells, h + 1):
                cv.put(cells, color)
                break
    for i, seq in enumerate(strips):
        for j, color in enumerate(seq):
            cv.grid[j, c0 + i] = color
    grid = cv.grid
    if not _pair_structure_ok(grid):
        return None
    info = _analyze(grid)
    feats = _strip_features(info)
    for i, role in enumerate(roles):
        if role != "free" and role not in feats[i]["roles"]:
            return None
    if spec.get("max_passed") is not None and any(f["passed"] > spec["max_passed"] for f in feats):
        return None
    return grid


def _pair_structure_ok(grid):
    try:
        _check_pair_input(grid)
    except AssertionError:
        return False
    return True


def _check_pair_input(grid):
    grid = np.asarray(grid, dtype=int)
    H, W = grid.shape
    assert 1 <= H <= 30 and 1 <= W <= 30
    bg = _background(grid)
    assert np.count_nonzero(grid == bg) > grid.size // 2, "background must dominate"
    pk = _packet(grid, bg)
    assert pk is not None, "no packet on the top boundary"
    c0, k, h = pk
    assert 2 <= k <= 5 and 2 <= h <= 4, "packet size"
    assert np.all(grid[:h, c0:c0 + k] != bg), "packet must be a solid rectangle"
    assert h < H and np.all(grid[h] == bg), "gap row under packet must be empty"
    rest = grid[:h + 1].copy()
    rest[:h, c0:c0 + k] = bg
    assert np.all(rest == bg), "nothing else in the packet rows"
    info = _analyze(grid)
    # objects must be single-colored and mutually separated (8-neighborhood)
    nonbg = grid != bg
    nonbg[:h, c0:c0 + k] = False
    seen = np.zeros_like(nonbg)
    count = 0
    for r in range(H):
        for c in range(W):
            if nonbg[r, c] and not seen[r, c]:
                stack, n = [(r, c)], 0
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    n += 1
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            yy, xx = y + dy, x + dx
                            if 0 <= yy < H and 0 <= xx < W and nonbg[yy, xx] and not seen[yy, xx]:
                                seen[yy, xx] = True
                                stack.append((yy, xx))
                count += 1
    assert count == len(info["comps"]), "objects must be single-colored and separated"
    for comp in info["comps"]:
        cells = set(comp["cells"])
        assert len(cells) <= 4, "objects have at most four cells"
        if not comp["square"]:
            for (y, x) in cells:
                assert not {(y, x + 1), (y + 1, x), (y + 1, x + 1)} <= cells, "fragment contains a 2x2 window"
    # landing footprints must be empty background in the input
    tops = _landing_rows(info, "truth")
    foot = np.zeros(grid.shape, dtype=bool)
    for i, top in enumerate(tops):
        foot[top:top + h, c0 + i] = True
    for i, top in enumerate(tops):
        col = c0 + i
        for r in range(top - 1, top + h):
            for c in (col - 1, col, col + 1):
                if not (0 <= r < H and 0 <= c < W) or foot[r, c] or (r < h and c0 <= c < c0 + k):
                    continue
                assert grid[r, c] == bg, "landed strip must not touch other objects (except its pad below)"
        for r in range(top, top + h):
            if r >= h:
                assert grid[r, col] == bg, "landing footprint must be empty"
    return info


def _sample_counts(num_train, num_test):
    if not (isinstance(num_train, (int, np.integer)) and isinstance(num_test, (int, np.integer))):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("need 2..10 train, 1..4 test, and num_train > num_test")


def _palette(rng, n_fg, avoid=()):
    colors = [c for c in range(10)]
    rng.shuffle(colors)
    bg = colors[0]
    fg = [c for c in colors[1:] if c not in avoid][:n_fg]
    return int(bg), [int(c) for c in fg]


def _puzzle_params(rng):
    mode = ("wider", "taller", "deeper", "newkeys")[int(rng.integers(4))]
    palette_mode = "fixed" if rng.random() < 0.5 else "varied"
    base_h = int(rng.choice([2, 3, 4, 4]))
    if mode == "taller":
        base_h = int(rng.choice([2, 3]))
    h_vary = rng.random() < 0.35
    return {"mode": mode, "palette_mode": palette_mode, "base_h": base_h, "h_vary": h_vary}


def _pair_spec(rng, params, split, roles_required, fixed_palette, train_max_h, test_keys_avoid):
    mode = params["mode"]
    # strip height
    if mode == "taller":
        h = int(rng.integers(2, params["base_h"] + 1)) if split == "train" else \
            int(rng.integers(train_max_h + 1, 5))
    else:
        h = params["base_h"]
        if params["h_vary"]:
            h = int(rng.integers(2, 5))
    # packet width
    if mode == "wider":
        k = int(rng.integers(2, 4)) if split == "train" else int(rng.integers(4, 6))
    else:
        k = int(rng.integers(2, 6))
    k = max(k, len(roles_required))
    roles = list(roles_required) + ["free"] * (k - len(roles_required))
    rng.shuffle(roles)
    # canvas
    deep_test = mode == "deeper" and split == "test"
    if deep_test:
        free = [i for i, r in enumerate(roles) if r == "free"]
        roles[free[0] if free else int(rng.integers(k))] = "deep"
    Hmin = 22 if deep_test else max(10, 3 * h + 5)
    Wmin = max(k + 4, 7)
    aspect = ("tall", "wide", "wide", "square", "free")[int(rng.integers(5))]
    if aspect == "wide" and Hmin <= 24:
        H = int(rng.integers(Hmin, min(27, Hmin + 8) + 1))
        W = int(rng.integers(min(H + 3, 30), 31))
    elif aspect == "square":
        H = int(rng.integers(max(Hmin, Wmin), 31))
        W = H
    elif aspect == "tall":
        W = int(rng.integers(Wmin, Wmin + 8))
        H = int(rng.integers(max(Hmin, W + 3), 31))
    else:
        H = int(rng.integers(Hmin, 31))
        W = int(rng.integers(Wmin, 31))
    c0 = int(rng.integers(0, W - k + 1))
    # palette
    if fixed_palette is not None:
        bg, fg = fixed_palette
        fg = list(fg)
    else:
        bg, fg = _palette(rng, int(rng.integers(3, 6)))
    if mode == "newkeys":
        if split == "train":
            key_pool = [c for c in fg if c not in test_keys_avoid] or fg
        else:
            key_pool = [c for c in fg if c in test_keys_avoid] or fg
    else:
        key_pool = fg
    nk = int(rng.integers(1, min(3, k, len(key_pool)) + 1))
    keys = [int(c) for c in rng.permutation(key_pool)[:nk]]
    extra = [c for c in range(10) if c != bg and c not in fg]
    extra = [int(c) for c in rng.permutation(extra)[:int(rng.integers(0, 2))]] if extra else []
    area = H * W
    fill = (0, max(1, area // 60)) if rng.random() < 0.5 else (area // 90, max(1, area // 45))
    max_passed = 2 if (mode == "deeper" and split == "train") else None
    return {"H": H, "W": W, "h": h, "k": k, "c0": c0, "bg": bg, "palette": fg, "keys": keys,
            "roles": roles, "fill": fill, "extra": extra, "max_passed": max_passed}


def _try_puzzle(rng, num_train, num_test):
    params = _puzzle_params(rng)
    fixed = None
    if params["palette_mode"] == "fixed":
        n_fg = 5 if params["mode"] == "newkeys" else int(rng.integers(3, 6))
        fixed = _palette(rng, n_fg)
    # disjoint key pools for the newkeys shift
    if params["mode"] == "newkeys":
        if fixed is not None:
            test_keys = list(fixed[1][:2])
        else:
            test_keys = [int(c) for c in rng.permutation(10)[:3]]
    else:
        test_keys = []
    # role requirements: two demonstrations jointly carry every required contrast
    train_roles = [["shape", "bottom"], ["color", "order"]]
    if rng.random() < 0.5:
        train_roles = [["double", "bottom"], ["color", "order", "shape"]]
    for _ in range(num_train - 2):
        pick = list(rng.choice(["shape", "color", "order", "bottom", "double", "free"],
                               size=int(rng.integers(0, 3)), replace=False))
        train_roles.append(pick)
    if "double" not in sum(train_roles, []):
        train_roles[0].append("double") if len(train_roles[0]) < 3 else train_roles[-1].append("double")
    test_roles = []
    for _ in range(num_test):
        pick = list(rng.choice(["shape", "color", "order", "double", "bottom"],
                               size=int(rng.integers(1, 3)), replace=False))
        test_roles.append(pick)

    pairs, seen = [], set()
    train_max_h = 0
    specs_out = []
    for split, role_lists in (("train", train_roles), ("test", test_roles)):
        for roles in role_lists:
            grid = None
            for _ in range(60):
                spec = _pair_spec(rng, params, split, roles, fixed, max(train_max_h, 2), test_keys)
                if params["mode"] == "newkeys" and params["palette_mode"] == "varied":
                    # keys must come from the reserved test pool (test) or avoid it (train)
                    pool = [c for c in spec["palette"] if (c in test_keys) == (split == "test")]
                    if len(pool) == 0:
                        continue
                    spec["keys"] = [int(c) for c in rng.permutation(pool)[:min(len(pool), 2)]]
                g = _build_pair(rng, spec)
                if g is None:
                    continue
                key = g.tobytes() + bytes(g.shape)
                if key in seen:
                    continue
                grid = g
                seen.add(key)
                break
            if grid is None:
                return None
            if split == "train":
                train_max_h = max(train_max_h, _analyze(grid)["h"])
            pairs.append((split, grid))
            specs_out.append(spec)
    train = [{"input": g.tolist(), "output": solve(g).tolist()} for s, g in pairs if s == "train"]
    test = [{"input": g.tolist(), "output": solve(g).tolist()} for s, g in pairs if s == "test"]
    order = rng.permutation(len(train))
    train = [train[i] for i in order]
    puzzle = {"train": train, "test": test}
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return None
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    if num_train is None:  # automatic count requests (e.g. sampling tools) are seed-determined
        num_train = int(rng.integers(3, 7)) if num_test is None else max(int(rng.integers(3, 7)), int(num_test) + 1)
    if num_test is None:
        num_test = int(rng.integers(1, min(3, int(num_train) - 1) + 1)) if isinstance(num_train, (int, np.integer)) else 1
    _sample_counts(num_train, num_test)
    for _ in range(400):
        puzzle = _try_puzzle(rng, int(num_train), int(num_test))
        if puzzle is not None:
            return puzzle
    raise RuntimeError(f"could not generate puzzle for seed {seed}")


# ---------------------------------------------------------------------------
# Validation and self test
# ---------------------------------------------------------------------------

def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    train_roles = set()
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert set(pair) == {"input", "output"}
            gi = np.asarray(pair["input"], dtype=int)
            go = np.asarray(pair["output"], dtype=int)
            assert gi.ndim == 2 and gi.shape == go.shape
            assert gi.min() >= 0 and gi.max() <= 9 and go.min() >= 0 and go.max() <= 9
            key = gi.tobytes() + bytes(gi.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _check_pair_input(gi)
            assert np.array_equal(solve(gi), go), "output must follow the rule"
            assert not np.array_equal(gi, go)
            # strips keep their color sequences and the rest of the grid is untouched
            changed = gi != go
            assert changed.sum() <= 2 * info["k"] * info["h"]
            if split == "train":
                for f in _strip_features(info):
                    train_roles |= f["roles"]
    for role in ("shape", "color", "order", "bottom", "double"):
        assert role in train_roles, f"training lacks the {role} contrast"
    caught_any = False
    for pair in train:
        info = _analyze(np.asarray(pair["input"], dtype=int))
        if any(f["caught"] for f in _strip_features(info)):
            caught_any = True
    assert caught_any
    # every simplified rule is contradicted by the training demonstrations
    for mode in SHORTCUTS:
        assert any(not np.array_equal(_predict(np.asarray(p["input"]), mode), np.asarray(p["output"]))
                   for p in train), f"training does not refute shortcut {mode}"
    # the test set must also refute the main single-parent shortcuts
    for mode in ("any", "color", "square", "never", "rigid"):
        assert any(not np.array_equal(_predict(np.asarray(p["input"]), mode), np.asarray(p["output"]))
                   for p in test), f"test does not refute shortcut {mode}"
    return True


def self_test():
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted {bad}")
    # hand-made example: strip passes a same-colored L and a red square, lands on its own pad
    g = np.zeros((12, 6), dtype=int)
    g[0:2, 1] = [3, 4]          # strip: top 3, bottom 4
    g[0:2, 2] = [4, 2]          # strip: top 4, bottom 2
    g[3, 1] = 4; g[4, 1] = 4; g[4, 0] = 4      # L fragment of 4 (not square)
    g[6:8, 1:3] = 3             # complete square of color 3 (order decoy for strip 1)
    g[10:12, 0:2] = 4           # pad of color 4
    out = solve(g)
    assert list(out[8:10, 1]) == [3, 4], out
    assert list(out[10:12, 2]) == [4, 2], out
    assert np.all(out[0:2, 1:3] == 0)
    shapes, modes, ks, hs, sigs = set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed, 3 + seed % 3, 1 + seed % 2)
        validate_generated_puzzle(p)
        q = generate_puzzle(seed, 3 + seed % 3, 1 + seed % 2)
        assert all(np.array_equal(np.asarray(a["input"]), np.asarray(b["input"]))
                   for a, b in zip(p["train"] + p["test"], q["train"] + q["test"]))
        for pair in p["train"] + p["test"]:
            gi = np.asarray(pair["input"])
            info = _analyze(gi)
            shapes.add(gi.shape)
            ks.add(info["k"])
            hs.add(info["h"])
        sigs.add(str(p["test"][0]["input"]))
    assert len(sigs) == 42
    assert len(shapes) >= 30 and ks >= {2, 3, 4, 5} and hs >= {2, 3, 4}
    return True


if __name__ == "__main__":
    self_test()
    print("ok")
