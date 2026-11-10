"""Mirror-wall windows: a full half-line is reflected through a mirror wall and
travels outward until the first obstacle.

Parent A (6d0aefbc) contributes horizontal reflection across a seam
(``abc`` becomes ``abc|cba``: the image is reversed, not duplicated).
Parent B (95755ff2) contributes a closed hollow window whose interior lines are
painted by line-of-sight projection that stops before the first pre-existing
blocker, with centrally opposite instruction halves.

Every family member: background canvas (all border cells background); one or
two closed convex windows, each drawn in one outline colour with a straight
mirror wall of the same colour crossing it.  On each line that crosses the
wall, one side may be completely filled (the source).  The other side (the
target) has background cells and possibly blockers.  Starting next to the
wall and moving outward, target cell at distance d receives source cell at
distance d, until the first non-background target cell (blocker or outline)
or until the source is exhausted.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# ----------------------------------------------------------------------------
# Rule analysis (input only)
# ----------------------------------------------------------------------------

VARIANTS = ("rule", "translate", "skip", "nearest", "farthest", "inward",
            "only_a", "only_b")


def _exterior(g, bg):
    h, w = g.shape
    ext = np.zeros((h, w), bool)
    q = deque()
    for y in range(h):
        for x in range(w):
            if (y in (0, h - 1) or x in (0, w - 1)) and g[y, x] == bg:
                ext[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not ext[ny, nx] and g[ny, nx] == bg:
                ext[ny, nx] = True
                q.append((ny, nx))
    return ext


def _components(mask):
    h, w = mask.shape
    seen = np.zeros_like(mask)
    comps = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not seen[y, x]:
                seen[y, x] = True
                comp, q = [], deque([(y, x)])
                while q:
                    cy, cx = q.popleft()
                    comp.append((cy, cx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True
                                q.append((ny, nx))
                comps.append(sorted(comp))
    return comps


def _analyze(grid):
    """Recover background, windows, mirror walls and crossing lines from a grid.

    Each line is a pair of runs (side a = left/top, side b = right/bottom),
    each listed from the wall outward.  Raises ValueError for non-members.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    bg = int(g[0, 0])
    h, w = g.shape
    ext = _exterior(g, bg)
    windows = []
    for comp in _components(~ext):
        cells = set(comp)
        outline = int(g[comp[0]])  # topmost-leftmost cell is on the outline
        inside = {c for c in cells if g[c] != outline}
        hcells = [(y, x) for (y, x) in comp if g[y, x] == outline
                  and (y, x - 1) in inside and (y, x + 1) in inside]
        vcells = [(y, x) for (y, x) in comp if g[y, x] == outline
                  and (y - 1, x) in inside and (y + 1, x) in inside]
        if hcells and not vcells and len({x for _, x in hcells}) == 1:
            orient, mirror = "h", hcells
        elif vcells and not hcells and len({y for y, _ in vcells}) == 1:
            orient, mirror = "v", vcells
        else:
            raise ValueError("window without a single straight mirror wall")
        lines = []
        for (y, x) in mirror:
            if orient == "h":
                steps = ((0, -1), (0, 1))
            else:
                steps = ((-1, 0), (1, 0))
            runs = []
            for dy, dx in steps:
                run, cy, cx = [], y + dy, x + dx
                while 0 <= cy < h and 0 <= cx < w and g[cy, cx] != outline:
                    if (cy, cx) not in inside:
                        raise ValueError("open window")
                    run.append((cy, cx))
                    cy, cx = cy + dy, cx + dx
                runs.append(run)
            lines.append({"wall": (y, x), "a": runs[0], "b": runs[1]})
        windows.append({"outline": outline, "orient": orient,
                        "index": mirror[0][1] if orient == "h" else mirror[0][0],
                        "cells": cells, "inside": inside, "lines": lines})
    return {"bg": bg, "exterior": ext, "windows": windows}


def _line_roles(g, bg, line):
    a = [int(g[c]) for c in line["a"]]
    b = [int(g[c]) for c in line["b"]]
    full_a = bool(a) and all(v != bg for v in a)
    full_b = bool(b) and all(v != bg for v in b)
    if full_a and not full_b:
        return "a", a, line["b"]
    if full_b and not full_a:
        return "b", b, line["a"]
    return None, None, None


def _apply(grid, variant="rule", info=None):
    g = np.asarray(grid, dtype=int)
    if info is None:
        info = _analyze(g)
    bg = info["bg"]
    out = g.copy()
    for win in info["windows"]:
        for line in win["lines"]:
            side, src, tgt = _line_roles(g, bg, line)
            if side is None:
                continue
            if variant == "only_a" and side != "a":
                continue
            if variant == "only_b" and side != "b":
                continue
            s, t = len(src), len(tgt)
            if variant == "inward":
                for d in range(t, 0, -1):
                    cell = tgt[d - 1]
                    if g[cell] != bg:
                        break
                    if d <= s:
                        out[cell] = src[d - 1]
                continue
            for d in range(1, t + 1):
                if d > s:
                    break
                cell = tgt[d - 1]
                if g[cell] != bg:
                    if variant == "skip":
                        continue
                    break
                if variant == "translate":
                    value = src[s - d]
                elif variant == "nearest":
                    value = src[0]
                elif variant == "farthest":
                    value = src[-1]
                else:
                    value = src[d - 1]
                out[cell] = value
    return out


def solve(input_grid):
    """Reflect each full half-line through its mirror wall, outward, until the first obstacle."""
    return _apply(np.asarray(input_grid, dtype=int), "rule")


# ----------------------------------------------------------------------------
# Window construction (local frame: mirror wall is a vertical column)
# ----------------------------------------------------------------------------

SHAPES = ("diamond", "hexagon", "octagon", "rect", "trapezoid", "kite")
TEXTURES = ("random", "bands", "diag", "blocks", "words", "rows_mixed", "mixed")


def _profile(kind, H, k):
    ys = np.arange(H)
    if kind == "diag":
        e = np.minimum(np.minimum(ys, H - 1 - ys), k)
    elif kind == "grow":
        e = np.minimum(ys, k)
    elif kind == "shrink":
        e = np.minimum(H - 1 - ys, k)
    else:
        e = np.zeros(H, int)
    return e.astype(int)


def _shape_geometry(rng, shape, H):
    """Return (l, r, a, b, W): per-row outline columns and flat top range."""
    half = (H - 1) // 2
    if shape == "diamond":
        f, tl, kl, tr, kr = 1, "diag", H, "diag", H
    elif shape == "hexagon":
        f = int(rng.integers(2, 6))
        tl = tr = "diag"
        kl = kr = H
    elif shape == "octagon":
        f = int(rng.integers(2, 6))
        tl = tr = "diag"
        kl = kr = int(rng.integers(1, max(2, half)))
    elif shape == "rect":
        f = int(rng.integers(5, 12))
        tl = tr = "flat"
        kl = kr = 0
    elif shape == "trapezoid":
        f = int(rng.integers(2, 7))
        t = str(rng.choice(["grow", "shrink"]))
        tl = t
        tr = t if rng.random() < 0.6 else "flat"
        kl = int(rng.integers(1, max(2, H - 1)))
        kr = int(rng.integers(1, max(2, H - 1))) if tr != "flat" else 0
    else:  # kite: independent convex profiles per side
        f = int(rng.integers(1, 5))
        tl = str(rng.choice(["diag", "grow", "shrink", "flat"]))
        tr = str(rng.choice(["diag", "grow", "shrink"]))
        kl = int(rng.integers(1, H))
        kr = int(rng.integers(1, H))
    el = _profile(tl, H, kl)
    er = _profile(tr, H, kr)
    a = int(el.max())
    b = a + f - 1
    l = a - el
    r = b + er
    W = int(r.max()) + 1
    return l, r, a, b, W


def _useful_rows(l, r, m, H):
    rows = []
    for y in range(1, H - 1):
        ls, rs = m - 1 - int(l[y]), int(r[y]) - 1 - m
        if ls >= 1 and rs >= 1:
            rows.append((y, ls, rs))
    return rows


def _make_geometry(rng, cfg, max_rows, max_cols):
    """Sample a window outline + mirror in the local frame.  Returns dict or None."""
    for _ in range(60):
        shape = cfg["shape"]
        if cfg.get("offset") and shape == "diamond":
            shape = str(rng.choice(["hexagon", "octagon", "rect", "trapezoid", "kite"]))
        hmin, hmax = cfg["hrange"]
        hmax = min(hmax, max_rows)
        if hmax < hmin:
            return None
        H = int(rng.integers(hmin, hmax + 1))
        if shape == "diamond" and H % 2 == 0:
            H -= 1
            if H < 5:
                continue
        l, r, a, b, W = _shape_geometry(rng, shape, H)
        if W > max_cols or W < 5:
            continue
        cands = []
        for m in range(a, b + 1):
            rows = _useful_rows(l, r, m, H)
            if len(rows) < cfg.get("min_lines", 3):
                continue
            imbalance = sum(abs(ls - rs) for _, ls, rs in rows)
            cands.append((imbalance, m, rows))
        if not cands:
            continue
        if cfg.get("offset"):
            offs = [c for c in cands if c[0] >= 2]
            if not offs:
                continue
            imb, m, rows = offs[int(rng.integers(len(offs)))]
        else:
            best = min(c[0] for c in cands)
            if shape in ("hexagon", "octagon", "rect", "diamond") and best > len(cands[0][2]):
                continue
            pick = [c for c in cands if c[0] == best]
            imb, m, rows = pick[int(rng.integers(len(pick)))]
        return {"shape": shape, "H": H, "W": W, "l": l, "r": r, "m": m, "rows": rows}
    return None


def _assign_sides(rng, cfg, rows):
    n = len(rows)
    arr = cfg["arrangement"]
    first = cfg.get("pin_side") or str(rng.choice(["a", "b"]))
    other = "b" if first == "a" else "a"
    if arr == "pinwheel":
        split = int(np.clip(n // 2 + int(rng.integers(-1, 2)), 1, n - 1))
        sides = [first] * split + [other] * (n - split)
    elif arr == "one_side":
        sides = [first] * n
    elif arr == "bands":
        sides, cur = [], first
        while len(sides) < n:
            sides += [cur] * int(rng.integers(2, 5))
            cur = "b" if cur == "a" else "a"
        sides = sides[:n]
        if len(set(sides)) < 2:
            sides[-1] = other
    else:
        sides = [str(rng.choice(["a", "b"])) for _ in range(n)]
        if len(set(sides)) < 2:
            sides[int(rng.integers(n))] = other if sides[0] == first else first
    blank = cfg.get("blank", 0.0)
    if n >= 4 and blank > 0:
        for i in range(n):
            if rng.random() < blank and sum(s != "n" for s in sides) > 3:
                sides[i] = "n"
    return sides


def _row_values(rng, texture, y, s, pal, state):
    k = len(pal)
    if texture == "mixed":
        texture = state.setdefault(("mix", y), str(rng.choice(["random", "words", "rows", "bands"])))
    if texture == "rows_mixed":
        texture = "rows" if rng.random() < 0.35 else "random"
    if texture == "random":
        return [int(pal[int(rng.integers(k))]) for _ in range(s)]
    if texture == "bands":
        bw, ph = state.setdefault("bands", (int(rng.integers(1, 3)), int(rng.integers(k))))
        return [int(pal[((d - 1) // bw + ph) % k]) for d in range(1, s + 1)]
    if texture == "diag":
        bw, sg = state.setdefault("diag", (int(rng.integers(1, 3)), int(rng.choice([-1, 1]))))
        return [int(pal[(((d - 1) + sg * y) // bw) % k]) for d in range(1, s + 1)]
    if texture == "blocks":
        bw, bh = state.setdefault("blocks", (int(rng.integers(1, 4)), int(rng.integers(1, 4))))
        return [int(pal[((d - 1) // bw + y // bh) % k]) for d in range(1, s + 1)]
    if texture == "words":
        word = [int(pal[int(rng.integers(k))]) for _ in range(int(rng.integers(2, 4)))]
        if len(set(word)) == 1:
            word[-1] = int(pal[(list(pal).index(word[0]) + 1) % k])
        return [word[(d - 1) % len(word)] for d in range(1, s + 1)]
    if texture == "rows":
        c = int(pal[int(rng.integers(k))])
        return [c] * s
    if texture == "palin":
        half = [int(pal[int(rng.integers(k))]) for _ in range((s + 1) // 2)]
        return half + half[: s // 2][::-1]
    raise ValueError(texture)


def _make_sources(rng, cfg, geo, pal):
    sides = _assign_sides(rng, cfg, geo["rows"])
    state = {}
    srcs = []
    for (y, ls, rs), side in zip(geo["rows"], sides):
        if side == "n":
            srcs.append(None)
            continue
        s = ls if side == "a" else rs
        srcs.append(_row_values(rng, cfg["texture"], y, s, pal, state))
    return sides, srcs


def _make_blockers(rng, cfg, geo, sides):
    density = cfg["density"]
    p_row, max_b = {"none": (0.0, 0), "sparse": (0.3, 1), "medium": (0.5, 2),
                    "dense": (0.8, 3)}[density]
    blk = []
    for (y, ls, rs), side in zip(geo["rows"], sides):
        cells = {}
        if side == "n":
            for sd, length in (("a", ls), ("b", rs)):
                if length >= 2 and rng.random() < p_row:
                    cells[sd] = sorted(int(v) for v in rng.choice(np.arange(1, length + 1), size=1, replace=False))
        else:
            t = rs if side == "a" else ls
            tside = "b" if side == "a" else "a"
            if t >= 2 and rng.random() < p_row:
                nb = int(rng.integers(1, min(max_b, t - 1) + 1))
                pos = set()
                if density == "dense" and rng.random() < 0.3:
                    pos.add(1)
                while len(pos) < nb:
                    pos.add(int(rng.integers(1, t + 1)))
                cells[tside] = sorted(pos)
        blk.append(cells)
    return blk


def _render_window(geo, sides, srcs, blk, pal_role):
    """Return local input/output arrays (horizontal frame) built from geometry."""
    H, W, l, r, m = geo["H"], geo["W"], geo["l"], geo["r"], geo["m"]
    bg, outline, blockers = pal_role["bg"], pal_role["outline"], pal_role["blockers"]
    inp = np.full((H, W), bg, int)
    for y in range(H):
        if y in (0, H - 1):
            inp[y, l[y]:r[y] + 1] = outline
        else:
            inp[y, l[y]] = outline
            inp[y, r[y]] = outline
    inp[1:H - 1, m] = outline

    def cell(y, side, d):
        return (y, m - d) if side == "a" else (y, m + d)

    bi = 0
    for (y, ls, rs), side, src, cells in zip(geo["rows"], sides, srcs, blk):
        for sd, ds in cells.items():
            for d in ds:
                inp[cell(y, sd, d)] = blockers[bi % len(blockers)]
                bi += 1
        if src is not None:
            for d, v in enumerate(src, 1):
                inp[cell(y, side, d)] = v
    out = inp.copy()
    painted = 0
    for (y, ls, rs), side, src in zip(geo["rows"], sides, srcs):
        if src is None:
            continue
        tside = "b" if side == "a" else "a"
        t = rs if side == "a" else ls
        for d in range(1, min(t, len(src)) + 1):
            c = cell(y, tside, d)
            if inp[c] != bg:
                break
            out[c] = src[d - 1]
            painted += 1
    return inp, out, painted


def _orient_window(rng, inp, out, orient, flips):
    if flips:
        if rng.random() < 0.5:
            inp, out = inp[:, ::-1], out[:, ::-1]
        if rng.random() < 0.5:
            inp, out = inp[::-1, :], out[::-1, :]
    if orient == "v":
        inp, out = inp.T, out.T
    return np.ascontiguousarray(inp), np.ascontiguousarray(out)


# ----------------------------------------------------------------------------
# Pair and puzzle generation
# ----------------------------------------------------------------------------

_HRANGES = {"small": (5, 9), "medium": (7, 13), "large": (11, 19)}


def _pair_palette(rng, base, cfg):
    bg, outline = base["bg"], base["outline"]
    if cfg.get("recolor"):
        bg = 0 if rng.random() < 0.5 else int(rng.integers(10))
        outline = int(rng.choice([c for c in range(10) if c != bg]))
    free = [c for c in range(10) if c not in (bg, outline)]
    if cfg.get("fresh_palette") or not set(base["src"]) <= set(free):
        k = int(rng.integers(2, 5))
        src = [int(c) for c in rng.choice(free, size=k, replace=False)]
    else:
        src = list(base["src"])
    if base["blk"] in free and not cfg.get("new_blocker"):
        blk = [base["blk"]]
    else:
        blk = [int(rng.choice(free))]
    if cfg.get("blocker_overlap"):
        blk = [int(rng.choice(src))]
    elif blk[0] in src and len(free) > len(src):
        blk = [int(rng.choice([c for c in free if c not in src]))]
    if cfg.get("two_blockers"):
        extra = [c for c in free if c not in src and c != blk[0]]
        if extra:
            blk.append(int(rng.choice(extra)))
    return {"bg": bg, "outline": outline, "src": src, "blockers": blk}


def _build_pair(rng, cfg, base, twin=None):
    """Build one pair.  ``twin`` reuses window geometry/sources, resampling blockers."""
    pal = twin["pal"] if twin else _pair_palette(rng, base, cfg)
    nwin = cfg["n_win"]
    stack = cfg.get("stack") if nwin == 2 else None
    if twin:
        stack = twin["stack"]
    elif stack is None and nwin == 2:
        stack = bool(rng.random() < 0.5)
    windows = []
    for wi in range(nwin):
        orient = cfg["orients"][wi]
        if nwin == 1:
            lim_r, lim_c = 26, 26
        elif stack:
            lim_r, lim_c = 12, 26
        else:
            lim_r, lim_c = 26, 13
        if twin:
            geo, sides, srcs, flipkey = twin["windows"][wi]
        else:
            mr, mc = (lim_r, lim_c) if orient == "h" else (lim_c, lim_r)
            geo = _make_geometry(rng, cfg, mr, mc)
            if geo is None:
                return None
            sides, srcs = _make_sources(rng, cfg, geo, pal["src"])
            flipkey = int(rng.integers(4)) if cfg.get("flips", True) else 0
        blk = _make_blockers(rng, cfg, geo, sides)
        inp, out, painted = _render_window(geo, sides, srcs, blk, {
            "bg": pal["bg"], "outline": pal["outline"], "blockers": pal["blockers"]})
        if flipkey & 1:
            inp, out = inp[:, ::-1], out[:, ::-1]
        if flipkey & 2:
            inp, out = inp[::-1, :], out[::-1, :]
        if orient == "v":
            inp, out = inp.T, out.T
        windows.append({"inp": np.ascontiguousarray(inp), "out": np.ascontiguousarray(out),
                        "painted": painted, "store": (geo, sides, srcs, flipkey),
                        "orient": orient, "shape": geo["shape"]})
    if any(w["painted"] == 0 for w in windows):
        return None
    # canvas assembly
    bg = pal["bg"]
    marg = [int(rng.integers(1, 4)) for _ in range(4)] if not twin else twin["marg"]
    if nwin == 1:
        w0 = windows[0]
        h, w = w0["inp"].shape
        Hc, Wc = h + marg[0] + marg[1], w + marg[2] + marg[3]
        if Hc > 30 or Wc > 30:
            return None
        gi = np.full((Hc, Wc), bg, int)
        go = gi.copy()
        gi[marg[0]:marg[0] + h, marg[2]:marg[2] + w] = w0["inp"]
        go[marg[0]:marg[0] + h, marg[2]:marg[2] + w] = w0["out"]
        places = None
    else:
        gap = int(rng.integers(1, 4)) if not twin else twin["gap"]
        a, b = windows
        if stack:
            Hc = a["inp"].shape[0] + gap + b["inp"].shape[0] + marg[0] + marg[1]
            Wc = max(a["inp"].shape[1], b["inp"].shape[1]) + marg[2] + marg[3]
        else:
            Hc = max(a["inp"].shape[0], b["inp"].shape[0]) + marg[0] + marg[1]
            Wc = a["inp"].shape[1] + gap + b["inp"].shape[1] + marg[2] + marg[3]
        if Hc > 30 or Wc > 30:
            return None
        gi = np.full((Hc, Wc), bg, int)
        go = gi.copy()
        if twin:
            places = twin["places"]
        else:
            places = []
            if stack:
                y = marg[0]
                for wdw in windows:
                    hh, ww = wdw["inp"].shape
                    x = marg[2] + int(rng.integers(0, Wc - marg[2] - marg[3] - ww + 1))
                    places.append((y, x))
                    y += hh + gap
            else:
                x = marg[2]
                for wdw in windows:
                    hh, ww = wdw["inp"].shape
                    y = marg[0] + int(rng.integers(0, Hc - marg[0] - marg[1] - hh + 1))
                    places.append((y, x))
                    x += ww + gap
        for wdw, (y, x) in zip(windows, places):
            hh, ww = wdw["inp"].shape
            gi[y:y + hh, x:x + ww] = wdw["inp"]
            go[y:y + hh, x:x + ww] = wdw["out"]
    gap_v = twin["gap"] if twin else (gap if nwin == 2 else 0)
    record = {"pal": pal, "windows": [w["store"] for w in windows], "marg": marg,
              "gap": gap_v, "places": places, "stack": stack}
    meta = {"orients": [w["orient"] for w in windows], "shapes": [w["shape"] for w in windows],
            "n_win": nwin, "twin": twin is not None, "texture": cfg["texture"],
            "arrangement": cfg["arrangement"], "density": cfg["density"]}
    return gi, go, record, meta


def _base_cfg(rng, o1):
    other = "v" if o1 == "h" else "h"
    n_win = 2 if rng.random() < 0.18 else 1
    orients = [o1 if rng.random() < 0.65 else other for _ in range(n_win)]
    arr = str(rng.choice(["pinwheel", "one_side", "bands", "random"], p=[0.5, 0.17, 0.21, 0.12]))
    return {
        "n_win": n_win, "orients": orients,
        "shape": str(rng.choice(SHAPES)),
        "offset": bool(rng.random() < 0.35),
        "arrangement": arr,
        "texture": str(rng.choice(TEXTURES)),
        "density": str(rng.choice(["none", "sparse", "medium", "dense"], p=[0.1, 0.3, 0.4, 0.2])),
        "hrange": _HRANGES[str(rng.choice(["small", "medium", "large"], p=[0.35, 0.45, 0.2]))],
        "blank": 0.08 if rng.random() < 0.3 else 0.0,
        "flips": True,
        "fresh_palette": bool(rng.random() < 0.3),
        "new_blocker": bool(rng.random() < 0.2),
        "blocker_overlap": bool(rng.random() < 0.12),
        "two_blockers": bool(rng.random() < 0.12),
        "recolor": False,
    }


MODES = ("orientation", "arrangement", "occlusion", "clipping", "palindrome",
         "multiwindow", "scale")


def _plan(rng, mode, nt, ns):
    o1 = str(rng.choice(["h", "v"]))
    o2 = "v" if o1 == "h" else "h"
    recolor = bool(rng.random() < 0.25)
    train = [_base_cfg(rng, o1) for _ in range(nt)]
    test = [_base_cfg(rng, o1) for _ in range(ns)]
    for c in train + test:
        c["recolor"] = recolor and rng.random() < 0.6
    if mode == "orientation":
        for c in train:
            c["n_win"], c["orients"] = 1, [o1]
        for j in rng.choice(nt, size=max(1, nt // 3), replace=False):
            train[int(j)]["orients"] = [o2]
        for c in test:
            c["orients"] = [o2] * c["n_win"]
    elif mode == "arrangement":
        side = str(rng.choice(["a", "b"]))
        for c in train:
            c["arrangement"], c["pin_side"], c["flips"] = "pinwheel", side, False
            c["n_win"], c["orients"] = 1, [o1]
        for c in test:
            c["flips"] = False
            c["n_win"], c["orients"] = 1, [o1]
            c["arrangement"] = str(rng.choice(["pinwheel", "bands", "random"]))
            if c["arrangement"] == "pinwheel":
                c["pin_side"] = "b" if side == "a" else "a"
    elif mode == "occlusion":
        for c in train:
            c["density"] = str(rng.choice(["sparse", "medium"]))
        for c in test:
            c["density"] = "dense"
            c["hrange"] = _HRANGES["medium"]
    elif mode == "clipping":
        for c in train:
            c["offset"] = bool(rng.random() < 0.4)
        train[int(rng.integers(nt))]["offset"] = True
        for c in test:
            c["offset"] = True
            c["shape"] = str(rng.choice(["rect", "trapezoid", "kite", "octagon"]))
    elif mode == "palindrome":
        for c in train:
            if c["texture"] == "rows_mixed":
                c["texture"] = "random"
        for c in test:
            c["texture"] = "palin"
            c["density"] = str(rng.choice(["sparse", "medium", "dense"]))
    elif mode == "multiwindow":
        for i, c in enumerate(train):
            c["n_win"] = 1
            c["orients"] = [o1 if i % 2 == 0 else o2]
            c["hrange"] = _HRANGES[str(rng.choice(["small", "medium"]))]
        for c in test:
            c["n_win"] = 2
            c["orients"] = [o1, o2] if rng.random() < 0.6 else [str(rng.choice([o1, o2]))] * 2
            c["hrange"] = _HRANGES["small"]
    elif mode == "scale":
        for c in train:
            c["hrange"] = _HRANGES["small"]
            if c["n_win"] == 2:
                c["n_win"], c["orients"] = 1, c["orients"][:1]
        for c in test:
            c["n_win"], c["orients"] = 1, [str(rng.choice([o1, o2]))]
            c["hrange"] = _HRANGES["large"]
            c["density"] = str(rng.choice(["medium", "dense"]))
    for c in train + test:
        if c["n_win"] == 2:
            c["hrange"] = (5, min(c["hrange"][1], 11))
    return train, test


def _check_counts(num_train, num_test):
    for v in (num_train, num_test):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
            raise ValueError("counts must be integers")
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("need 2..10 train, 1..4 test and num_train > num_test")
    return int(num_train), int(num_test)


def _generate(seed, num_train, num_test):
    if num_train is None:
        num_train = 4
    if num_test is None:
        num_test = 1
    nt, ns = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = str(rng.choice(MODES, p=[0.17, 0.15, 0.16, 0.16, 0.1, 0.13, 0.13]))
    bg0 = 0 if rng.random() < 0.7 else int(rng.integers(1, 10))
    outline0 = int(rng.choice([c for c in range(10) if c != bg0]))
    free = [c for c in range(10) if c not in (bg0, outline0)]
    src0 = [int(c) for c in rng.choice(free, size=int(rng.integers(2, 5)), replace=False)]
    blk0 = int(rng.choice([c for c in free if c not in src0]))
    base = {"bg": bg0, "outline": outline0, "src": src0, "blk": blk0}
    for _attempt in range(400):
        train_cfg, test_cfg = _plan(rng, mode, nt, ns)
        pairs, metas, seen = [], [], set()
        twin_from = None
        if nt >= 3 and rng.random() < 0.5:
            twin_from = (0, int(rng.integers(1, nt)))
        ok = True
        records = []
        for i, cfg in enumerate(train_cfg + test_cfg):
            built = None
            for _ in range(40):
                twin = None
                if twin_from and i == twin_from[1]:
                    twin = records[twin_from[0]]
                    cfg = dict(train_cfg[twin_from[0]])
                    cfg["density"] = str(rng.choice(["none", "sparse", "medium", "dense"]))
                built = _build_pair(rng, cfg, base, twin)
                if built is None:
                    continue
                key = built[0].tobytes() + bytes(built[0].shape)
                if key in seen or np.array_equal(built[0], built[1]):
                    built = None
                    continue
                seen.add(key)
                break
            if built is None:
                ok = False
                break
            gi, go, rec, meta = built
            records.append(rec)
            pairs.append({"input": gi, "output": go})
            metas.append(meta)
        if not ok:
            continue
        puzzle = {"train": pairs[:nt], "test": pairs[nt:]}
        try:
            _check_evidence(puzzle, require_translate_test=(mode != "palindrome"))
        except ValueError:
            continue
        puzzle = {k: [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in v]
                  for k, v in puzzle.items()}
        return puzzle, {"mode": mode, "pairs": metas}
    raise RuntimeError("could not generate a puzzle satisfying evidence constraints")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    return _generate(seed, num_train, num_test)[0]


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def _pair_facts(grid):
    g = np.asarray(grid, dtype=int)
    info = _analyze(g)
    bg = info["bg"]
    h, w = g.shape
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    if np.any(border != bg):
        raise ValueError("border must be background")
    if not info["windows"]:
        raise ValueError("no window")
    facts = {"orients": set(), "unequal": False, "palin_only": True, "sides": set()}
    for win in info["windows"]:
        active = 0
        facts["orients"].add(win["orient"])
        for line in win["lines"]:
            side, src, tgt = _line_roles(g, bg, line)
            if side is None:
                continue
            active += 1
            facts["sides"].add(side)
            if src != src[::-1]:
                facts["palin_only"] = False
            s, t = len(src), len(tgt)
            reach = 0
            for d in range(1, min(s, t) + 1):
                if g[tgt[d - 1]] != bg:
                    break
                reach = d
            if s != t and reach == min(s, t):
                facts["unequal"] = True
        if active < 2:
            raise ValueError("window needs at least two active lines")
    return info, facts


def _check_evidence(puzzle, require_translate_test=True):
    infos = {}
    for split in ("train", "test"):
        for i, p in enumerate(puzzle[split]):
            info, facts = _pair_facts(p["input"])
            infos[(split, i)] = (info, facts)
            truth = _apply(p["input"], "rule", info)
            if not np.array_equal(truth, np.asarray(p["output"])):
                raise ValueError("output disagrees with rule")
    train = puzzle["train"]
    for variant in VARIANTS[1:]:
        if all(np.array_equal(_apply(p["input"], variant, infos[("train", i)][0]),
                              np.asarray(p["output"])) for i, p in enumerate(train)):
            raise ValueError(f"training does not refute shortcut {variant}")
    tr_or = set().union(*(infos[("train", i)][1]["orients"] for i in range(len(train))))
    tr_uneq = any(infos[("train", i)][1]["unequal"] for i in range(len(train)))
    for i, p in enumerate(puzzle["test"]):
        info, facts = infos[("test", i)]
        if not facts["orients"] <= tr_or:
            raise ValueError("test orientation unseen in training")
        if facts["unequal"] and not tr_uneq:
            raise ValueError("test clipping unseen in training")
        out = np.asarray(p["output"])
        if np.array_equal(_apply(p["input"], "skip", info), out):
            raise ValueError("test does not exercise first-blocker stopping")
        if np.array_equal(_apply(p["input"], "translate", info), out):
            if require_translate_test or not facts["palin_only"]:
                raise ValueError("test does not exercise reflection")


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train/test")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check_counts(nt, ns)
    seen = set()
    for split in ("train", "test"):
        for p in puzzle[split]:
            if not isinstance(p, dict) or set(p) != {"input", "output"}:
                raise ValueError("pair must have exactly input/output")
            gi, go = np.asarray(p["input"]), np.asarray(p["output"])
            for g in (gi, go):
                if g.ndim != 2 or not (1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30):
                    raise ValueError("bad grid shape")
                if not np.issubdtype(g.dtype, np.integer) or g.min() < 0 or g.max() > 9:
                    raise ValueError("bad colours")
            if gi.shape != go.shape:
                raise ValueError("output must keep input shape")
            key = gi.tobytes() + bytes(gi.shape)
            if key in seen:
                raise ValueError("duplicate input")
            seen.add(key)
            if np.array_equal(gi, go):
                raise ValueError("identity pair")
            if not np.array_equal(solve(gi), go):
                raise ValueError("solve disagrees")
            # only interior background cells may change
            info = _analyze(gi)
            changed = gi != go
            if np.any(changed & (gi != info["bg"])):
                raise ValueError("a non-background cell changed")
            inside = np.zeros_like(changed)
            for win in info["windows"]:
                for c in win["inside"]:
                    inside[c] = True
            if np.any(changed & ~inside):
                raise ValueError("a cell outside a window changed")
    _check_evidence(puzzle, require_translate_test=False)
    return True


def self_test():
    seeds = list(range(42))
    modes, shapes, orients, nwin, sizes, textures = set(), set(), set(), set(), set(), set()
    twins = 0
    for s in seeds:
        puzzle, meta = _generate(s, 4, 1)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(s, 4, 1)
        assert again == puzzle, "nondeterministic"
        modes.add(meta["mode"])
        for pm in meta["pairs"]:
            shapes.update(pm["shapes"])
            orients.update(pm["orients"])
            nwin.add(pm["n_win"])
            textures.add(pm["texture"])
            twins += pm["twin"]
        for split in ("train", "test"):
            for p in puzzle[split]:
                sizes.add(np.asarray(p["input"]).shape)
    assert len(modes) >= 5, modes
    assert len(shapes) >= 5, shapes
    assert orients == {"h", "v"}, orients
    assert nwin == {1, 2}, nwin
    assert len(textures) >= 5, textures
    assert len(sizes) >= 30, len(sizes)
    assert twins >= 3, twins
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for a in (2, 6, 10):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(7, a, b))
    return True


if __name__ == "__main__":
    self_test()
    print("self_test passed")
