"""Narrowest enclosed cross-section.

Several closed one-cell outlines (rectangles, diamonds, tips, pinched vases,
slanted bands) sit on a background, each with optional interior accent cells.
Every interior row of every closed outline is a horizontal wall-to-wall span
(parent A).  Only interiors that are truly enclosed count (parent B).  The
globally narrowest interior span(s) are found across all enclosures and only
those interior rows are flooded with the colour of the outline that encloses
them, preserving accents (parent B's fill); every wider interior row, every
open (broken) outline and all exterior background stay unchanged.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# ----------------------------------------------------------------------------
# Input analysis shared by solve(), the validator and the self-test.
# ----------------------------------------------------------------------------


def _dilate8(mask):
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src = mask[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
            out[max(0, dr):h - max(0, -dr), max(0, dc):w - max(0, -dc)] |= src
    return out


def _background(g):
    vals, counts = np.unique(g, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def analyze(grid):
    """Return background, exterior mask, outline colours and enclosures.

    Each enclosure is a dict with ``cells`` (list of (r, c)), ``color`` (the
    enclosing outline colour, or None if ambiguous) and ``rows``
    {r: (cmin, cmax, count)}.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background(g)
    ext = np.zeros((h, w), bool)
    dq = deque()
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and g[r, c] == bg and not ext[r, c]:
                ext[r, c] = True
                dq.append((r, c))
    while dq:
        r, c = dq.popleft()
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= nr < h and 0 <= nc < w and not ext[nr, nc] and g[nr, nc] == bg:
                ext[nr, nc] = True
                dq.append((nr, nc))
    touch = _dilate8(ext) & (g != bg)
    outline_colors = sorted({int(v) for v in g[touch]})
    interior = (~ext) & ~np.isin(g, outline_colors)
    seen = np.zeros((h, w), bool)
    enclosures = []
    for r0 in range(h):
        for c0 in range(w):
            if not interior[r0, c0] or seen[r0, c0]:
                continue
            cells = []
            seen[r0, c0] = True
            dq.append((r0, c0))
            while dq:
                r, c = dq.popleft()
                cells.append((r, c))
                for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                    if 0 <= nr < h and 0 <= nc < w and interior[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            mask = np.zeros((h, w), bool)
            for r, c in cells:
                mask[r, c] = True
            rim = _dilate8(mask) & ~mask
            rim_colors = {int(v) for v in g[rim]}
            rows = {}
            for r, c in cells:
                lo, hi, n = rows.get(r, (c, c, 0))
                rows[r] = (min(lo, c), max(hi, c), n + 1)
            enclosures.append({
                "cells": cells,
                "mask": mask,
                "color": rim_colors.pop() if len(rim_colors) == 1 else None,
                "rows": rows,
            })
    return {"bg": bg, "ext": ext, "outline_colors": outline_colors,
            "interior": interior, "enclosures": enclosures}


def _selected_rows(info, measure="span"):
    """(enclosure index, row) pairs attaining the global minimum measure."""
    g_rows = []
    for k, enc in enumerate(info["enclosures"]):
        for r, (lo, hi, _n) in enc["rows"].items():
            g_rows.append((k, r, lo, hi))
    if not g_rows:
        return set(), None
    if measure == "span":
        vals = {(k, r): hi - lo + 1 for k, r, lo, hi in g_rows}
    else:  # background-cell count in the interior row
        grid = info["_grid"]
        vals = {(k, r): int(np.sum(grid[r, lo:hi + 1] == info["bg"])) for k, r, lo, hi in g_rows}
    m = min(vals.values())
    return {key for key, v in vals.items() if v == m}, m


def solve(input_grid):
    g = np.asarray(input_grid, dtype=int)
    info = analyze(g)
    out = g.copy()
    sel, _m = _selected_rows(info)
    for k, r in sel:
        enc = info["enclosures"][k]
        lo, hi, _n = enc["rows"][r]
        if enc["color"] is None:
            continue
        for c in range(lo, hi + 1):
            if g[r, c] == info["bg"]:
                out[r, c] = enc["color"]
    return out


# ----------------------------------------------------------------------------
# Alternative (shortcut) rules used to prove that both parents matter.
# ----------------------------------------------------------------------------


def shortcut_rowpairs(input_grid):
    """Parent A only: fill the globally shortest background run bounded by two
    equal non-background colours in a row (ignores enclosure)."""
    g = np.asarray(input_grid, dtype=int)
    bg = _background(g)
    runs = []
    for r in range(g.shape[0]):
        c = 0
        w = g.shape[1]
        while c < w:
            if g[r, c] != bg:
                c += 1
                continue
            s = c
            while c < w and g[r, c] == bg:
                c += 1
            if s > 0 and c < w and g[r, s - 1] == g[r, c]:
                runs.append((c - s, r, s, c, int(g[r, c])))
    out = g.copy()
    if runs:
        m = min(x[0] for x in runs)
        for n, r, s, e, col in runs:
            if n == m:
                out[r, s:e] = col
    return out


def shortcut_fill_all(input_grid):
    """Parent B only: fill every enclosed interior."""
    g = np.asarray(input_grid, dtype=int)
    info = analyze(g)
    out = g.copy()
    for enc in info["enclosures"]:
        if enc["color"] is None:
            continue
        for r, c in enc["cells"]:
            if g[r, c] == info["bg"]:
                out[r, c] = enc["color"]
    return out


def shortcut_per_enclosure(input_grid):
    """Local minimum: each enclosure fills its own narrowest rows."""
    g = np.asarray(input_grid, dtype=int)
    info = analyze(g)
    out = g.copy()
    for enc in info["enclosures"]:
        m = min(hi - lo + 1 for lo, hi, _ in enc["rows"].values())
        for r, (lo, hi, _n) in enc["rows"].items():
            if hi - lo + 1 == m:
                for c in range(lo, hi + 1):
                    if g[r, c] == info["bg"]:
                        out[r, c] = enc["color"]
    return out


def shortcut_smallest_area(input_grid):
    """Fill the whole enclosure(s) with the fewest interior cells."""
    g = np.asarray(input_grid, dtype=int)
    info = analyze(g)
    out = g.copy()
    if not info["enclosures"]:
        return out
    m = min(len(e["cells"]) for e in info["enclosures"])
    for enc in info["enclosures"]:
        if len(enc["cells"]) == m:
            for r, c in enc["cells"]:
                if g[r, c] == info["bg"]:
                    out[r, c] = enc["color"]
    return out


# ----------------------------------------------------------------------------
# Shape construction.
# ----------------------------------------------------------------------------

MAX_IW = 9   # maximum interior width
MAX_IH = 7   # maximum interior height


def _step(rng, L, R, lb, first, style):
    """Monotone outward step: the interior never narrows away from its waist."""
    opts = {"sym": [(1, 1), (1, 1), (0, 0)], "left": [(1, 0), (1, 0), (0, 0)],
            "right": [(0, 1), (0, 1), (0, 0)], "rand": [(1, 0), (0, 1), (1, 1), (0, 0)]}[style]
    for _ in range(20):
        dl, dr = opts[int(rng.integers(len(opts)))]
        if first and dl + dr == 0:
            continue
        nL, nR = L - dl, R + dr
        if nR - nL + 1 > MAX_IW:
            break
        if nR - nL + 1 >= lb:
            return nL, nR
    if R - L + 1 >= lb:
        return L, R
    return L - 1, R + 1


def _profile(rng, w0, lb, above, below, band, slant):
    """Rows (L, R) of an interior: a waist band of width w0, outward rows >= lb."""
    style = str(rng.choice(["sym", "sym", "rand", "left", "right"]))
    band_rows = [(0, w0 - 1)]
    for _ in range(1, band):
        L, R = band_rows[-1]
        band_rows.append((L + slant, R + slant))
    strict = lb > w0
    up = []
    L, R = band_rows[0]
    for i in range(above):
        L, R = _step(rng, L, R, lb, strict and i == 0, style)
        up.append((L, R))
    down = []
    L, R = band_rows[-1]
    for i in range(below):
        L, R = _step(rng, L, R, lb, strict and i == 0, style)
        down.append((L, R))
    rows = up[::-1] + band_rows + down
    shift = min(L for L, _ in rows)
    return [(L - shift, R - shift) for L, R in rows]


def _make_shape(rng, kind, w0, lb):
    """Return interior rows for a shape kind with waist width w0."""
    for _ in range(60):
        if kind == "rect":
            h = int(rng.integers(1, 6))
            rows = _profile(rng, w0, lb, 0, 0, h, 0)
        elif kind == "tip":
            band = int(rng.choice([1, 1, 2]))
            n = int(rng.integers(1, 5))
            if rng.random() < 0.5:
                rows = _profile(rng, w0, lb, 0, n, band, 0)
            else:
                rows = _profile(rng, w0, lb, n, 0, band, 0)
        elif kind == "waist":
            band = int(rng.choice([1, 1, 2]))
            a = int(rng.integers(1, 4))
            b = int(rng.integers(1, 4))
            rows = _profile(rng, w0, lb, a, b, band, 0)
        elif kind == "slant":
            band = int(rng.integers(2, 5))
            slant = int(rng.choice([-1, 1])) if w0 >= 2 else 0
            a = int(rng.integers(0, 2))
            b = int(rng.integers(0, 2))
            rows = _profile(rng, w0, lb, a, b, band, slant)
        else:
            raise ValueError(kind)
        width = max(R for _, R in rows) + 1
        if len(rows) <= MAX_IH and width <= MAX_IW + 2:
            return rows
    return _profile(rng, w0, lb, 0, 0, 2, 0)


def _ring(rows):
    """Local masks (interior, ring) with a one-cell ring border."""
    ih = len(rows)
    iw = max(R for _, R in rows) + 1
    inner = np.zeros((ih + 2, iw + 2), bool)
    for r, (L, R) in enumerate(rows):
        inner[r + 1, L + 1:R + 2] = True
    ring = _dilate8(inner) & ~inner
    return inner, ring


def _open_ring(rng, rows, ring):
    """Remove one ring cell so that the interior leaks to the exterior."""
    cands = []
    n = len(rows)
    for r, (L, R) in enumerate(rows):
        up = rows[r - 1] if r > 0 else None
        dn = rows[r + 1] if r + 1 < n else None
        if (up is None or up[0] >= L) and (dn is None or dn[0] >= L):
            cands.append((r + 1, L))
        if (up is None or up[1] <= R) and (dn is None or dn[1] <= R):
            cands.append((r + 1, R + 2))
    L0, R0 = rows[0]
    for c in range(L0, R0 + 1):
        cands.append((0, c + 1))
    Ln, Rn = rows[-1]
    for c in range(Ln, Rn + 1):
        cands.append((n + 1, c + 1))
    cands = [p for p in cands if ring[p]]
    r, c = cands[int(rng.integers(len(cands)))]
    ring = ring.copy()
    ring[r, c] = False
    return ring


# ----------------------------------------------------------------------------
# Pair construction.
# ----------------------------------------------------------------------------

SHAPE_KINDS = ("rect", "tip", "waist", "slant")


def _palette(rng, n_outline, n_accent):
    cols = [int(x) for x in rng.permutation(10)]
    bg = cols[0]
    outline = cols[1:1 + n_outline]
    accent = cols[1 + n_outline:1 + n_outline + n_accent]
    return bg, outline, accent


def _place(rng, sizes, H, W, tight_pair):
    """Random non-touching placement of ring bounding boxes (1-cell gap)."""
    occ = np.zeros((H, W), bool)
    pos = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][0] * sizes[i][1])
    if tight_pair is not None:
        i, j = tight_pair
        hi_, wi = sizes[i]
        hj, wj = sizes[j]
        if wi + wj + 1 > W or max(hi_, hj) > H:
            return None
        for _ in range(60):
            c0 = int(rng.integers(0, W - (wi + wj + 1) + 1))
            ri = int(rng.integers(0, H - hi_ + 1))
            lo = max(0, ri - hj + 2)
            hi2 = min(H - hj, ri + hi_ - 2)
            if lo > hi2:
                continue
            rj = int(rng.integers(lo, hi2 + 1))
            pos[i] = (ri, c0)
            pos[j] = (rj, c0 + wi + 1)
            occ[max(0, ri - 1):ri + hi_ + 1, max(0, c0 - 1):c0 + wi + 1] = True
            occ[max(0, rj - 1):rj + hj + 1, max(0, c0 + wi):c0 + wi + 1 + wj + 1] = True
            break
        else:
            return None
        order = [k for k in order if k not in (i, j)]
    for k in order:
        h, w = sizes[k]
        if h > H or w > W:
            return None
        for _ in range(250):
            r = int(rng.integers(0, H - h + 1))
            c = int(rng.integers(0, W - w + 1))
            if not occ[r:r + h, c:c + w].any():
                pos[k] = (r, c)
                occ[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
                break
        else:
            return None
    return pos


def _build_pair(rng, cfg, spec):
    """Try to construct one pair; returns (input, output, meta) or None."""
    m = spec["m"]
    mode = spec["mode"]
    shapes = []  # dicts: rows, holder(bool), closed(bool)
    if mode == "partial":
        kind = str(rng.choice(["tip", "waist", "waist", "slant"]))
        if kind == "slant" and m == 1:
            kind = "waist"
        shapes.append({"rows": _make_shape(rng, kind, m, m + 1), "kind": kind, "closed": True})
    elif mode == "rect":
        shapes.append({"rows": _make_shape(rng, "rect", m, m + 1), "kind": "rect", "closed": True})
    else:  # tie
        n_hold = 2 if rng.random() < 0.75 else 3
        for _ in range(n_hold):
            kind = str(rng.choice(SHAPE_KINDS))
            if kind == "slant" and m == 1:
                kind = "tip"
            shapes.append({"rows": _make_shape(rng, kind, m, m + 1), "kind": kind, "closed": True})
    for _ in range(spec["n_other"]):
        kind = str(rng.choice(SHAPE_KINDS))
        w0 = int(rng.integers(m + 1, min(MAX_IW, m + 4) + 1))
        shapes.append({"rows": _make_shape(rng, kind, w0, w0), "kind": kind, "closed": True})
    if spec["decoy"]:
        w0 = int(rng.integers(1, m + 1))
        kind = str(rng.choice(["tip", "waist", "rect"]))
        rows = _make_shape(rng, kind, w0, w0)
        if len(rows) < 2:
            rows = rows + [rows[-1]]
        shapes.append({"rows": rows, "kind": "decoy", "closed": False})

    # rings
    for s in shapes:
        inner, ring = _ring(s["rows"])
        if not s["closed"]:
            ring = _open_ring(rng, s["rows"], ring)
        s["inner"], s["ring"] = inner, ring

    # palette
    pal = cfg["palette_mode"]
    if pal == "vary":
        bg, outline, accents = _palette(rng, cfg["n_outline"], cfg["n_accent"])
    else:
        bg, outline, accents = cfg["bg"], cfg["outline"], cfg["accent"]
    closed_idx = [i for i, s in enumerate(shapes) if s["closed"]]
    if len(outline) == 1:
        for s in shapes:
            s["color"] = outline[0]
    else:
        order = [outline[int(rng.integers(len(outline)))] for _ in shapes]
        # make sure at least two outline colours appear among closed shapes
        if len({order[i] for i in closed_idx}) < 2:
            order[closed_idx[-1]] = [c for c in outline if c != order[closed_idx[0]]][0]
        for s, col in zip(shapes, order):
            s["color"] = col
    if spec["tight"]:
        # the tight pair shares a colour so a row-pair shortcut sees an endpoint pair
        a, b = spec["tight_pair_ids"](len(closed_idx))
        shapes[closed_idx[b]]["color"] = shapes[closed_idx[a]]["color"]
        tight_pair = (closed_idx[a], closed_idx[b])
    else:
        tight_pair = None
    if spec["decoy"]:
        shapes[-1]["color"] = shapes[int(rng.integers(len(closed_idx)))]["color"]

    # accents
    n_acc_cols = len(accents)
    holders = [i for i in closed_idx if min(R - L + 1 for L, R in shapes[i]["rows"]) == m]
    for i in closed_idx:
        s = shapes[i]
        s["acc"] = {}
    acc_total = int(rng.integers(1, 2 * len(closed_idx) + 1))
    cells = []
    for i in closed_idx:
        for r, (L, R) in enumerate(shapes[i]["rows"]):
            for c in range(L, R + 1):
                cells.append((i, r, c))
    if spec["accent_in_fill"] and m >= 2:
        hi = holders[int(rng.integers(len(holders)))]
        min_rows = [r for r, (L, R) in enumerate(shapes[hi]["rows"]) if R - L + 1 == m]
        r = min_rows[int(rng.integers(len(min_rows)))]
        L, R = shapes[hi]["rows"][r]
        c = int(rng.integers(L, R + 1))
        shapes[hi]["acc"][(r, c)] = accents[0]
        acc_total -= 1
    for _ in range(max(0, acc_total)):
        i, r, c = cells[int(rng.integers(len(cells)))]
        if (r, c) in shapes[i]["acc"]:
            continue
        shapes[i]["acc"][(r, c)] = accents[int(rng.integers(n_acc_cols))]
    if spec["two_accents"] and n_acc_cols >= 2:
        used = {v for i in closed_idx for v in shapes[i]["acc"].values()}
        if len(used) < 2:
            for i in closed_idx:
                if shapes[i]["acc"]:
                    k = next(iter(shapes[i]["acc"]))
                    shapes[i]["acc"][k] = [a for a in accents if a not in used][0]
                    break
            used = {v for i in closed_idx for v in shapes[i]["acc"].values()}
            if len(used) < 2:
                return None

    # canvas
    sizes = [s["ring"].shape for s in shapes]
    area = sum((h + 1) * (w + 1) for h, w in sizes)
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    for _ in range(12):
        side = int(np.ceil(np.sqrt(area * float(rng.uniform(1.6, 2.6)))))
        H = int(np.clip(rng.integers(side - 3, side + 4), max_h, 30))
        W = int(np.clip(int(np.ceil(area * float(rng.uniform(1.6, 2.6)) / H)), max_w, 30))
        if tight_pair is not None:
            W = max(W, min(30, sizes[tight_pair[0]][1] + sizes[tight_pair[1]][1] + 1))
        pos = _place(rng, sizes, H, W, tight_pair)
        if pos is not None:
            break
    else:
        return None

    grid = np.full((H, W), bg, dtype=int)
    expected = grid.copy()
    for s, (r0, c0) in zip(shapes, pos):
        h, w = s["ring"].shape
        sub = grid[r0:r0 + h, c0:c0 + w]
        sub[s["ring"]] = s["color"]
        if s["closed"]:
            for (r, c), col in s["acc"].items():
                sub[r + 1, c + 1] = col
        s["origin"] = (r0, c0)
    expected = grid.copy()
    # expected output from the construction model
    for i in closed_idx:
        s = shapes[i]
        r0, c0 = s["origin"]
        for r, (L, R) in enumerate(s["rows"]):
            if R - L + 1 == m:
                for c in range(L, R + 1):
                    if (r, c) not in s["acc"]:
                        expected[r0 + r + 1, c0 + c + 1] = s["color"]
    meta = {"shapes": shapes, "m": m, "mode": mode, "bg": bg}
    return grid, expected, meta


def _pair_ok(grid, expected, meta, spec):
    info = analyze(grid)
    info["_grid"] = grid
    shapes = meta["shapes"]
    n_closed = sum(s["closed"] for s in shapes)
    if len(info["enclosures"]) != n_closed:
        return False
    if any(e["color"] is None for e in info["enclosures"]):
        return False
    vals, counts = np.unique(grid, return_counts=True)
    order = np.sort(counts)
    if len(order) > 1 and order[-1] <= order[-2]:
        return False
    if int(vals[int(np.argmax(counts))]) != meta["bg"]:
        return False
    out = solve(grid)
    if not np.array_equal(out, expected):
        return False
    sel_span, m = _selected_rows(info, "span")
    sel_bg, _ = _selected_rows(info, "bg")
    if sel_span != sel_bg or m != meta["m"]:
        return False
    # every selected row shows a visible change
    for k, r in sel_span:
        lo, hi, _ = info["enclosures"][k]["rows"][r]
        if not np.any(grid[r, lo:hi + 1] == info["bg"]):
            return False
    touched = {k for k, _ in sel_span}
    if len(touched) == len(info["enclosures"]):
        return False
    if spec["mode"] == "tie" and len(touched) < 2:
        return False
    if spec["mode"] == "partial":
        k = next(iter(touched))
        if len(touched) != 1 or len([1 for kk, _ in sel_span if kk == k]) == len(info["enclosures"][k]["rows"]):
            return False
    if spec["mode"] == "rect":
        k = next(iter(touched))
        if len(touched) != 1 or len([1 for kk, _ in sel_span if kk == k]) != len(info["enclosures"][k]["rows"]):
            return False
    if spec["accent_in_fill"] and meta["m"] >= 2 and not _accent_in_fill(grid, info, sel_span):
        return False
    if spec["must_beat_rowpair"] and np.array_equal(shortcut_rowpairs(grid), out):
        return False
    return True


def _accent_in_fill(grid, info, sel):
    for k, r in sel:
        lo, hi, _ = info["enclosures"][k]["rows"][r]
        if np.any(grid[r, lo:hi + 1] != info["bg"]):
            return True
    return False


def _sample_config(rng):
    cfg = {}
    cfg["palette_mode"] = str(rng.choice(["mono", "multi", "vary"], p=[0.3, 0.4, 0.3]))
    cfg["n_accent"] = 2 if rng.random() < 0.25 else 1
    cfg["n_outline"] = 1 if cfg["palette_mode"] == "mono" else int(rng.integers(2, 5))
    if cfg["palette_mode"] == "vary" and rng.random() < 0.35:
        cfg["n_outline"] = 1
    bg, outline, accent = _palette(rng, cfg["n_outline"], cfg["n_accent"])
    if rng.random() < 0.45:  # keep a source-like black background often
        if bg != 0:
            swap = 0
            outline = [bg if x == swap else x for x in outline]
            accent = [bg if x == swap else x for x in accent]
            bg = 0
    cfg.update(bg=bg, outline=outline, accent=accent)
    cfg["trap"] = str(rng.choice(["decoy", "tight", "both"], p=[0.4, 0.35, 0.25]))
    return cfg


def _pair_specs(rng, cfg, num_train, num_test):
    specs = []
    train_modes = ["partial", "tie" if rng.random() < 0.6 else "rect"]
    if num_train >= 3:
        train_modes.append("rect" if train_modes[1] == "tie" else "tie")
    while len(train_modes) < num_train:
        train_modes.append(str(rng.choice(["partial", "rect", "tie"], p=[0.45, 0.2, 0.35])))
    trap_idx = int(rng.integers(num_train))
    accent_idx = int(rng.integers(num_train))
    decoy_allowed = cfg["trap"] in ("decoy", "both")
    tight_allowed = cfg["trap"] in ("tight", "both")
    for i, mode in enumerate(train_modes):
        trap_here = i == trap_idx
        decoy = decoy_allowed and (trap_here and (cfg["trap"] != "both" or rng.random() < 0.5)
                                   or rng.random() < 0.25)
        tight = tight_allowed and ((trap_here and not decoy) or rng.random() < 0.25)
        if trap_here and not decoy and not tight:
            decoy = True
        specs.append({
            "split": "train", "mode": mode,
            "m": int(rng.choice([1, 2, 2, 3, 3, 4, 5])),
            "n_other": int(rng.integers(1, 3)) if i < 2 else int(rng.integers(1, 4)),
            "decoy": decoy, "tight": tight,
            "accent_in_fill": i == accent_idx or rng.random() < 0.4,
            "two_accents": cfg["n_accent"] == 2 and i == 0,
            "must_beat_rowpair": trap_here,
        })
        if i == accent_idx and specs[-1]["m"] < 2:
            specs[-1]["m"] = int(rng.integers(2, 5))
    for i in range(num_test):
        mode = str(rng.choice(["partial", "rect", "tie"], p=[0.45, 0.2, 0.35]))
        specs.append({
            "split": "test", "mode": mode,
            "m": int(rng.choice([1, 2, 3, 3, 4, 4, 5, 6])),
            "n_other": int(rng.integers(2, 4)),
            "decoy": decoy_allowed and rng.random() < 0.6,
            "tight": tight_allowed and rng.random() < 0.5,
            "accent_in_fill": rng.random() < 0.5,
            "two_accents": False,
            "must_beat_rowpair": False,
        })
    for s in specs:
        s["tight_pair_ids"] = _tight_ids
    return specs


def _tight_ids(n_closed):
    # holder (index 0) next to the first non-holder (index n_closed-1)
    return 0, n_closed - 1


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not isinstance(num_train, (int, np.integer)) or not isinstance(num_test, (int, np.integer)):
        raise ValueError("counts must be integers")
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("illegal train/test counts")
    rng = np.random.default_rng(seed)
    for _attempt in range(30):
        cfg = _sample_config(rng)
        specs = _pair_specs(rng, cfg, num_train, num_test)
        pairs, seen = [], set()
        ok = True
        for spec in specs:
            for _ in range(80):
                res = _build_pair(rng, cfg, spec)
                if res is None:
                    continue
                grid, expected, meta = res
                key = grid.tobytes() + bytes(grid.shape)
                if key in seen or not _pair_ok(grid, expected, meta, spec):
                    continue
                seen.add(key)
                pairs.append({"input": grid.tolist(), "output": expected.tolist()})
                break
            else:
                ok = False
                break
        if not ok:
            continue
        puzzle = {"train": pairs[:num_train], "test": pairs[num_train:]}
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError(f"could not generate puzzle for seed {seed}")


# ----------------------------------------------------------------------------
# Validation and self-test.
# ----------------------------------------------------------------------------


def _decoys(grid, info):
    """Outline-coloured 8-components that enclose nothing (open outlines)."""
    g = np.asarray(grid)
    oc = np.isin(g, info["outline_colors"])
    near = np.zeros_like(oc)
    for enc in info["enclosures"]:
        near |= _dilate8(enc["mask"])
    seen = np.zeros_like(oc)
    count = 0
    h, w = g.shape
    for r0 in range(h):
        for c0 in range(w):
            if not oc[r0, c0] or seen[r0, c0]:
                continue
            comp = []
            dq = deque([(r0, c0)])
            seen[r0, c0] = True
            while dq:
                r, c = dq.popleft()
                comp.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and oc[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            dq.append((nr, nc))
            if not any(near[p] for p in comp):
                count += 1
    return count


def _pair_features(inp, out):
    g = np.asarray(inp, dtype=int)
    o = np.asarray(out, dtype=int)
    info = analyze(g)
    info["_grid"] = g
    sel, m = _selected_rows(info)
    touched = {k for k, _ in sel}
    full = any(len([1 for kk, _ in sel if kk == k]) == len(info["enclosures"][k]["rows"]) for k in touched)
    partial = any(len([1 for kk, _ in sel if kk == k]) < len(info["enclosures"][k]["rows"]) for k in touched)
    accents = set(int(v) for v in np.unique(g[info["interior"]]) if v != info["bg"])
    return {
        "info": info, "sel": sel, "m": m, "touched": touched,
        "n_enc": len(info["enclosures"]), "full": full, "partial": partial,
        "cross": len(touched) >= 2, "multi_row": len(sel) >= 2,
        "accent_fill": _accent_in_fill(g, info, sel),
        "accents": accents,
        "beats_rowpair": not np.array_equal(shortcut_rowpairs(g), o),
        "beats_fill_all": not np.array_equal(shortcut_fill_all(g), o),
        "beats_local": not np.array_equal(shortcut_per_enclosure(g), o),
        "beats_area": not np.array_equal(shortcut_smallest_area(g), o),
        "decoys": _decoys(g, info),
        "kinds_rows": sorted(len(e["rows"]) for e in info["enclosures"]),
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    seen = set()
    feats = {"train": [], "test": []}
    for split in ("train", "test"):
        assert len(puzzle[split]) >= 1
        for pair in puzzle[split]:
            assert set(pair) == {"input", "output"}
            g = np.asarray(pair["input"], dtype=int)
            o = np.asarray(pair["output"], dtype=int)
            assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
            assert g.shape == o.shape
            assert g.min() >= 0 and g.max() <= 9 and o.min() >= 0 and o.max() <= 9
            key = g.tobytes() + bytes(g.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            vals, counts = np.unique(g, return_counts=True)
            cs = np.sort(counts)
            assert len(cs) >= 3 and cs[-1] > cs[-2], "background must be the strict majority"
            assert np.array_equal(solve(g), o), "output disagrees with the rule"
            f = _pair_features(g, o)
            info = f["info"]
            assert f["n_enc"] >= 2, "need at least two closed outlines"
            assert all(e["color"] is not None for e in info["enclosures"])
            assert len(f["touched"]) < f["n_enc"], "some enclosure must remain untouched"
            assert not np.array_equal(g, o), "output must change"
            assert not (f["accents"] & set(info["outline_colors"])), "accent colour reused as outline"
            for e in info["enclosures"]:
                for r, (lo, hi, n) in e["rows"].items():
                    assert hi - lo + 1 == n, "interior row must be contiguous"
            sel_bg, _ = _selected_rows(info, "bg")
            assert sel_bg == f["sel"], "span and background-count measures must agree"
            for k, r in f["sel"]:
                lo, hi, _ = info["enclosures"][k]["rows"][r]
                assert np.any(g[r, lo:hi + 1] == info["bg"]), "selected row must visibly change"
            diff = g != o
            assert np.all(info["interior"][diff]), "changes only inside enclosures"
            feats[split].append(f)
    tr = feats["train"]
    assert any(f["partial"] for f in tr), "training must show a partial fill"
    assert any(f["cross"] or (f["full"] and f["multi_row"]) for f in tr), "training must show ties/full fills"
    assert any(f["beats_rowpair"] for f in tr), "training must defeat the row-pair shortcut"
    assert any(f["accent_fill"] for f in tr), "training must preserve an accent in a filled row"
    if any(f["decoys"] for f in feats["test"]):
        assert any(f["decoys"] for f in tr), "open outlines in test must be shown in training"
    if any(len(f["accents"]) >= 2 for f in feats["test"]):
        assert any(len(f["accents"]) >= 2 for f in tr)
    return True


SELF_TEST_SEEDS = list(range(42))


def self_test():
    stats = {"m": set(), "shapes": set(), "nenc": set(), "decoy": 0, "cross": 0, "full": 0,
             "partial": 0, "rowpair": 0, "local": 0, "area": 0, "bg": set(), "palettes": set(),
             "test_m_new": 0, "sizes": set()}
    sigs = set()
    for seed in SELF_TEST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        if seed < 8:
            assert p == generate_puzzle(seed), "nondeterministic"
        sigs.add(str(p))
        train_m = set()
        for split in ("train", "test"):
            for pair in p[split]:
                f = _pair_features(pair["input"], pair["output"])
                g = np.asarray(pair["input"])
                stats["m"].add(f["m"])
                stats["nenc"].add(f["n_enc"])
                stats["sizes"].add(g.shape)
                stats["bg"].add(f["info"]["bg"])
                stats["palettes"].add(tuple(sorted(int(v) for v in np.unique(g))))
                stats["decoy"] += f["decoys"] > 0
                stats["cross"] += f["cross"]
                stats["full"] += f["full"]
                stats["partial"] += f["partial"]
                stats["rowpair"] += f["beats_rowpair"]
                stats["local"] += f["beats_local"]
                stats["area"] += f["beats_area"]
                for e in f["info"]["enclosures"]:
                    widths = tuple(hi - lo for lo, hi, _ in (e["rows"][r] for r in sorted(e["rows"])))
                    stats["shapes"].add(widths)
                if split == "train":
                    train_m.add(f["m"])
                elif f["m"] not in train_m:
                    stats["test_m_new"] += 1
    assert len(sigs) == len(SELF_TEST_SEEDS)
    assert len(stats["m"]) >= 5 and len(stats["nenc"]) >= 3 and len(stats["shapes"]) >= 150
    assert stats["decoy"] >= 20 and stats["cross"] >= 40 and stats["full"] >= 30 and stats["partial"] >= 60
    assert stats["rowpair"] >= 60 and stats["local"] >= 100 and stats["area"] >= 100
    assert len(stats["bg"]) >= 5 and len(stats["sizes"]) >= 60
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (3, 3), (3, 5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {k: (len(v) if isinstance(v, set) else v) for k, v in stats.items()}


if __name__ == "__main__":
    print(self_test())
