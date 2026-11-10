"""Marker-line zone legend: a composition of ARC 3de23699 and 77fdfe62.

Rule (see task_spec.md):
  * Four identical marker cells sit at the corners of an axis-aligned window
    (parent A).  The output is the strict interior of that window.
  * The marker rows and columns, extended across the canvas, act as invisible
    separator lines (parent B's frame).  They cut the exterior into four diagonal
    corner zones; each zone holds at most one monochrome key object.
  * The interior is split into four equal quadrants (parent B).  Payload cells in
    a quadrant take the colour of the key in the matching diagonal zone; when that
    zone holds no key (or does not exist because the window is flush with the
    canvas edge) they take the marker colour (parent A's recolouring).
"""
from __future__ import annotations

import numpy as np

ZONES = ("UL", "UR", "LL", "LR")
SHIFT_MODES = ("scale", "flush", "palette", "aspect")
MASK_STYLES = ("noise", "smooth", "blobs", "strokes", "sparse")
PLACE_STYLES = ("corner", "adjacent", "random")
KEY_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


# --------------------------------------------------------------------------
# counts
# --------------------------------------------------------------------------
def _count(value, name, lower, upper):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(num_train, num_test):
    test = 1 if num_test is None else _count(num_test, "num_test", 1, 4)
    train = max(4, test + 1) if num_train is None else _count(num_train, "num_train", 2, 10)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


# --------------------------------------------------------------------------
# input-only analysis and solver
# --------------------------------------------------------------------------
def _quadrants(h, w):
    hh, ww = h // 2, w // 2
    return {
        "UL": (slice(0, hh), slice(0, ww)),
        "UR": (slice(0, hh), slice(ww, w)),
        "LL": (slice(hh, h), slice(0, ww)),
        "LR": (slice(hh, h), slice(ww, w)),
    }


def _candidates(g):
    """All readings of the grid as marker window + payload + zone keys."""
    H, W = g.shape
    found = []
    for colour in np.unique(g).tolist():
        pos = np.argwhere(g == colour)
        if len(pos) != 4:
            continue
        rows = sorted(set(pos[:, 0].tolist()))
        cols = sorted(set(pos[:, 1].tolist()))
        if len(rows) != 2 or len(cols) != 2:
            continue
        (r1, r2), (c1, c2) = rows, cols
        h, w = r2 - r1 - 1, c2 - c1 - 1
        if h < 2 or w < 2 or h % 2 or w % 2:
            continue
        ring = np.concatenate([g[r1, c1 + 1:c2], g[r2, c1 + 1:c2], g[r1 + 1:r2, c1], g[r1 + 1:r2, c2]])
        if np.any(ring != ring[0]) or int(ring[0]) == colour:
            continue
        bg = int(ring[0])
        interior = g[r1 + 1:r2, c1 + 1:c2]
        inks = set(np.unique(interior).tolist()) - {bg}
        if len(inks) != 1:
            continue
        payload = inks.pop()
        outside = np.ones_like(g, dtype=bool)
        outside[r1:r2 + 1, c1:c2 + 1] = False
        keys, cells, ok = {}, {z: [] for z in ZONES}, True
        for r, c in np.argwhere(outside & (g != bg)).tolist():
            value = int(g[r, c])
            vert = "U" if r < r1 else ("L" if r > r2 else None)
            horz = "L" if c < c1 else ("R" if c > c2 else None)
            if vert is None or horz is None or value in (colour, payload):
                ok = False
                break
            zone = vert + horz
            if keys.setdefault(zone, value) != value:
                ok = False
                break
            cells[zone].append((r, c))
        if not ok:
            continue
        found.append({
            "r1": r1, "r2": r2, "c1": c1, "c2": c2, "bg": bg, "marker": int(colour),
            "payload": int(payload), "keys": keys, "key_cells": cells,
            "interior": interior.copy(),
            "margins": {"top": r1, "bottom": H - 1 - r2, "left": c1, "right": W - 1 - c2},
        })
    return found


def _analyse(grid):
    g = np.asarray(grid)
    if g.ndim != 2 or not np.issubdtype(g.dtype, np.integer):
        raise ValueError("input must be a 2D integer grid")
    found = _candidates(g.astype(int))
    if len(found) != 1:
        raise ValueError(f"expected exactly one marker window, found {len(found)}")
    info = found[0]
    info["zone_exists"] = {
        "UL": info["margins"]["top"] > 0 and info["margins"]["left"] > 0,
        "UR": info["margins"]["top"] > 0 and info["margins"]["right"] > 0,
        "LL": info["margins"]["bottom"] > 0 and info["margins"]["left"] > 0,
        "LR": info["margins"]["bottom"] > 0 and info["margins"]["right"] > 0,
    }
    return info


def _render(info, keys=None, default=None):
    interior = info["interior"]
    keys = info["keys"] if keys is None else keys
    default = info["marker"] if default is None else default
    out = interior.copy()
    for zone, (rs, cs) in _quadrants(*interior.shape).items():
        block = out[rs, cs]
        block[interior[rs, cs] == info["payload"]] = keys.get(zone, default)
    return out


def solve(input_grid):
    """Crop the marker window's strict interior and ink each quadrant's payload
    with the key of its diagonal exterior zone, or the marker colour if absent."""
    return _render(_analyse(np.asarray(input_grid, dtype=int))).astype(int)


# --------------------------------------------------------------------------
# alternative (shortcut) readings used to prove that both parents matter
# --------------------------------------------------------------------------
def _alt_no_legend(info):
    """Parent A alone: every payload cell takes the marker colour."""
    return _render(info, keys={})


def _alt_missing_keeps_payload(info):
    return _render(info, default=info["payload"])


def _alt_missing_erased(info):
    return _render(info, default=info["bg"])


def _alt_tight_crop(info):
    """Ignore the marker window: crop to the payload bounding box, split there."""
    interior = info["interior"]
    pos = np.argwhere(interior == info["payload"])
    (a, b), (c, d) = pos.min(0), pos.max(0)
    sub = interior[a:b + 1, c:d + 1]
    out = sub.copy()
    for zone, (rs, cs) in _quadrants(*sub.shape).items():
        block = out[rs, cs]
        block[sub[rs, cs] == info["payload"]] = info["keys"].get(zone, info["marker"])
    return out


def _alt_b_literal(grid):
    """Parent B alone: centre inside a one-cell frame, keys at the grid corners."""
    g = np.asarray(grid)
    if g.shape[0] < 6 or g.shape[1] < 6:
        return None
    centre = g[2:-2, 2:-2]
    if centre.shape[0] % 2 or centre.shape[1] % 2:
        return None
    values, counts = np.unique(g, return_counts=True)
    bg = values[np.argmax(counts)]
    corners = {"UL": g[0, 0], "UR": g[0, -1], "LL": g[-1, 0], "LR": g[-1, -1]}
    out = centre.copy()
    for zone, (rs, cs) in _quadrants(*centre.shape).items():
        block = out[rs, cs]
        block[centre[rs, cs] != bg] = corners[zone]
    return out


def _nearest_conflict(info):
    """True if some payload cell in a keyed quadrant has a unique Manhattan-nearest
    key cell of a different colour (so 'nearest key' differs from 'quadrant')."""
    keyed = [(r, c, info["keys"][z]) for z in ZONES for r, c in info["key_cells"][z]]
    if len({k for _, _, k in keyed}) < 2:
        return False
    pts = np.array([(r, c) for r, c, _ in keyed])
    colours = np.array([k for _, _, k in keyed])
    interior = info["interior"]
    for zone, (rs, cs) in _quadrants(*interior.shape).items():
        if zone not in info["keys"]:
            continue
        for i, j in np.argwhere(interior[rs, cs] == info["payload"]).tolist():
            r = info["r1"] + 1 + (rs.start or 0) + i
            c = info["c1"] + 1 + (cs.start or 0) + j
            d = np.abs(pts[:, 0] - r) + np.abs(pts[:, 1] - c)
            near = set(colours[d == d.min()].tolist())
            if len(near) == 1 and near.pop() != info["keys"][zone]:
                return True
    return False


def _payload_in(info, zone):
    rs, cs = _quadrants(*info["interior"].shape)[zone]
    return bool(np.any(info["interior"][rs, cs] == info["payload"]))


def _bbox_is_interior(info):
    pos = np.argwhere(info["interior"] == info["payload"])
    h, w = info["interior"].shape
    return bool(pos[:, 0].min() == 0 and pos[:, 1].min() == 0 and pos[:, 0].max() == h - 1 and pos[:, 1].max() == w - 1)


def _background_dominates(grid, bg):
    counts = np.bincount(np.asarray(grid).ravel(), minlength=10)
    return bool(counts[bg] > np.delete(counts, bg).max())


def _differs(a, b):
    return b is None or np.shape(a) != np.shape(b) or bool(np.any(np.asarray(a) != np.asarray(b)))


# --------------------------------------------------------------------------
# generation helpers
# --------------------------------------------------------------------------
def _even(rng, lo, hi):
    lo += lo % 2
    hi -= hi % 2
    return int(rng.choice(np.arange(lo, hi + 1, 2)))


def _sample_interior(rng, cfg, split, role):
    mode = cfg["shift"]
    if mode == "scale":
        if split == "train":
            h, w = _even(rng, 2, 8), _even(rng, 2, 8)
        else:
            h, w = _even(rng, 10, 22), _even(rng, 8, 22)
            if rng.random() < 0.5:
                h, w = w, h
    elif mode == "aspect":
        if split == "train":
            h = _even(rng, 2, 12)
            w = _even(rng, max(2, (h + 1) // 2), min(24, 2 * h))
            if rng.random() < 0.5:
                h, w = w, h
        else:
            short = 2 if rng.random() < 0.75 else 4
            long = _even(rng, max(10, 4 * short), 24)
            h, w = (short, long) if rng.random() < 0.5 else (long, short)
    else:
        band = rng.choice(["small", "medium", "large", "thin", "mixed"], p=[0.22, 0.3, 0.18, 0.12, 0.18])
        if band == "small":
            h, w = _even(rng, 2, 6), _even(rng, 2, 6)
        elif band == "medium":
            h, w = _even(rng, 4, 12), _even(rng, 4, 12)
        elif band == "large":
            h, w = _even(rng, 10, 24), _even(rng, 10, 26)
        elif band == "thin":
            h, w = 2, _even(rng, 8, 20)
        else:
            h, w = _even(rng, 2, 18), _even(rng, 2, 18)
        if rng.random() < 0.5:
            h, w = w, h
    if role.get("margin"):
        h, w = max(h, 4), max(w, 4)
    return h, w


def _zone_sides(zone):
    return ("top" if zone[0] == "U" else "bottom"), ("left" if zone[1] == "L" else "right")


def _sample_margins(rng, cfg, h, w, present, zero_sides, all_positive, wide):
    need = {s: 0 for s in ("top", "bottom", "left", "right")}
    for zone in present:
        for side in _zone_sides(zone):
            need[side] = 1
    if all_positive:
        need = {s: 1 for s in need}
    for side in zero_sides:
        if need[side]:
            return None
    margins = {}
    for a, b, size in (("top", "bottom", h), ("left", "right", w)):
        avail = 30 - (size + 2)
        lo = [need[a], need[b]]
        for idx, side in enumerate((a, b)):
            if not need[side] and side not in zero_sides and rng.random() < cfg["zero_rate"]:
                zero_sides = tuple(zero_sides) + (side,)
        if sum(lo) > avail:
            return None
        cap = int(rng.choice([1, 2, 3, 5, 8, 12], p=[0.12, 0.2, 0.22, 0.2, 0.16, 0.1]))
        if wide:
            cap = max(cap, 5)
        extra = int(rng.integers(0, min(avail - sum(lo), cap) + 1))
        if a in zero_sides and b in zero_sides:
            extra = 0
        if a in zero_sides:
            ea = 0
        elif b in zero_sides:
            ea = extra
        else:
            ea = int(rng.integers(0, extra + 1))
        margins[a], margins[b] = lo[0] + ea, lo[1] + (extra - ea)
        if a in zero_sides:
            margins[a] = 0
        if b in zero_sides:
            margins[b] = 0
    return margins


def _mask_noise(rng, h, w):
    base = rng.uniform(0.15, 0.6)
    m = np.zeros((h, w), dtype=bool)
    for rs, cs in _quadrants(h, w).values():
        d = float(np.clip(base + rng.normal(0, 0.15), 0.05, 0.9))
        m[rs, cs] = rng.random(m[rs, cs].shape) < d
    return m


def _mask_smooth(rng, h, w):
    f = rng.random((h, w))
    for _ in range(int(rng.integers(1, 3))):
        p = np.pad(f, 1, mode="edge")
        f = sum(p[i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9.0
    return f > np.quantile(f, rng.uniform(0.45, 0.8))


def _mask_blobs(rng, h, w):
    m = np.zeros((h, w), dtype=bool)
    for _ in range(int(rng.integers(1, 4))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        m[r, c] = True
        frontier = [(r, c)]
        target = int(rng.integers(2, max(3, h * w // 5) + 1))
        size = 1
        while size < target and frontier:
            r, c = frontier[int(rng.integers(len(frontier)))]
            nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                    if 0 <= r + dr < h and 0 <= c + dc < w and not m[r + dr, c + dc]]
            if not nbrs:
                frontier.remove((r, c))
                continue
            r, c = nbrs[int(rng.integers(len(nbrs)))]
            m[r, c] = True
            frontier.append((r, c))
            size += 1
    return m


def _mask_strokes(rng, h, w):
    m = np.zeros((h, w), dtype=bool)
    dirs = ((0, 1), (1, 0), (1, 1), (1, -1))
    for _ in range(int(rng.integers(1, 5))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        dr, dc = dirs[int(rng.integers(4))]
        for _ in range(int(rng.integers(2, max(h, w) + 1))):
            if not (0 <= r < h and 0 <= c < w):
                break
            m[r, c] = True
            r, c = r + dr, c + dc
    return m


def _mask_sparse(rng, h, w):
    m = np.zeros((h, w), dtype=bool)
    k = int(rng.integers(1, max(2, h * w // 6) + 1))
    for idx in rng.choice(h * w, size=min(k, h * w), replace=False):
        m[idx // w, idx % w] = True
    return m


_MASKERS = {"noise": _mask_noise, "smooth": _mask_smooth, "blobs": _mask_blobs,
            "strokes": _mask_strokes, "sparse": _mask_sparse}


def _sample_mask(rng, cfg, h, w, need_q, need_margin):
    style = MASK_STYLES[int(rng.choice(len(MASK_STYLES), p=cfg["mask_w"]))]
    m = _MASKERS[style](rng, h, w)
    quads = _quadrants(h, w)
    if h >= 4 and w >= 4:
        if rng.random() < 0.18:
            free = [z for z in ZONES if z not in need_q]
            if free:
                rs, cs = quads[free[int(rng.integers(len(free)))]]
                m[rs, cs] = False
        if rng.random() < 0.10:
            rs, cs = quads[ZONES[int(rng.integers(4))]]
            m[rs, cs] = True
    allowed = np.ones((h, w), dtype=bool)
    if need_margin:
        sides = [s for s in ("top", "bottom") if h >= 4] + [s for s in ("left", "right") if w >= 4]
        if not sides:
            return None
        chosen = rng.choice(sides, size=int(rng.integers(1, min(3, len(sides)) + 1)), replace=False)
        for side in chosen:
            dim = h if side in ("top", "bottom") else w
            depth = 1 if dim < 8 else int(rng.integers(1, 3))
            if side == "top":
                allowed[:depth, :] = False
            elif side == "bottom":
                allowed[h - depth:, :] = False
            elif side == "left":
                allowed[:, :depth] = False
            else:
                allowed[:, w - depth:] = False
        m &= allowed
    for zone in need_q:
        rs, cs = quads[zone]
        if not m[rs, cs].any():
            cells = np.argwhere(allowed[rs, cs])
            if not len(cells):
                return None
            for k in rng.choice(len(cells), size=min(len(cells), int(rng.integers(1, 4))), replace=False):
                i, j = cells[k]
                m[(rs.start or 0) + i, (cs.start or 0) + j] = True
    if not m.any():
        cells = np.argwhere(allowed)
        i, j = cells[int(rng.integers(len(cells)))]
        m[i, j] = True
    return m


def _zone_box(zone, H, W, r1, r2, c1, c2):
    rows = (0, r1 - 1) if zone[0] == "U" else (r2 + 1, H - 1)
    cols = (0, c1 - 1) if zone[1] == "L" else (c2 + 1, W - 1)
    return rows, cols


def _place_key(rng, cfg, grid, zone, box, colour, style):
    (ra, rb), (ca, cb) = box
    shape = ((0, 0),)
    if rng.random() < cfg["object_rate"]:
        shape = KEY_SHAPES[int(rng.integers(len(KEY_SHAPES)))]
    sh = max(i for i, _ in shape) + 1
    sw = max(j for _, j in shape) + 1
    if sh > rb - ra + 1 or sw > cb - ca + 1:
        shape, sh, sw = ((0, 0),), 1, 1
    tops = (ra, rb - sh + 1)
    lefts = (ca, cb - sw + 1)
    far_r = tops[0] if zone[0] == "U" else tops[1]
    near_r = tops[1] if zone[0] == "U" else tops[0]
    far_c = lefts[0] if zone[1] == "L" else lefts[1]
    near_c = lefts[1] if zone[1] == "L" else lefts[0]
    if style == "corner":
        tr, tc = far_r, far_c
    elif style == "adjacent":
        tr, tc = near_r, near_c
    else:
        tr = int(rng.integers(tops[0], tops[1] + 1))
        tc = int(rng.integers(lefts[0], lefts[1] + 1))
    for i, j in shape:
        grid[tr + i, tc + j] = colour


def _key_colours(rng, pool, present, distinct):
    pool = list(pool)
    if distinct or len(present) < 2:
        if len(pool) < len(present):
            return None
        chosen = rng.choice(pool, size=len(present), replace=False).tolist()
    else:
        chosen = [int(pool[int(rng.integers(len(pool)))]) for _ in present]
        if len(chosen) == 4 and len(set(chosen)) == 1:
            return None
    return {z: int(c) for z, c in zip(present, chosen)}


# --------------------------------------------------------------------------
# pair and puzzle construction
# --------------------------------------------------------------------------
def _pair_palette(rng, cfg, split):
    if split == "test" and cfg.get("test_palette") is not None:
        return cfg["test_palette"](rng)
    if cfg["roles"] == "fixed":
        return cfg["fixed"]
    if cfg["roles"] == "pooled":
        return cfg["bg"], int(rng.choice(cfg["pool_m"])), int(rng.choice(cfg["pool_p"]))
    bg = 0 if rng.random() < 0.5 else int(rng.integers(10))
    others = [c for c in range(10) if c != bg]
    m, p = rng.choice(others, size=2, replace=False).tolist()
    return bg, int(m), int(p)


def _choose_present(rng, role, split, existing):
    kind = role["kind"]
    if kind == "anchor":
        return list(ZONES)
    if kind == "default":
        k = int(rng.integers(1, 4))
        return sorted(rng.choice(ZONES, size=k, replace=False).tolist())
    if split == "test":
        pool = [z for z in ZONES if existing[z]]
        if not pool:
            return None
        k = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.3, 0.3, 0.15]))
        k = min(k, len(pool))
        return sorted(rng.choice(pool, size=k, replace=False).tolist())
    k = int(rng.choice([0, 1, 2, 3, 4], p=[0.1, 0.17, 0.25, 0.24, 0.24]))
    return sorted(rng.choice(ZONES, size=k, replace=False).tolist())


def _try_pair(rng, cfg, split, role):
    h, w = _sample_interior(rng, cfg, split, role)
    zero_sides = ()
    if role.get("flush"):
        side = ("top", "bottom", "left", "right")[int(rng.integers(4))]
        zero_sides = (side,)
        if rng.random() < 0.4:
            other = ("left", "right") if side in ("top", "bottom") else ("top", "bottom")
            zero_sides = (side, other[int(rng.integers(2))])
    existing = {z: not any(s in zero_sides for s in _zone_sides(z)) for z in ZONES}
    present = _choose_present(rng, role, split, existing)
    if present is None:
        return None
    all_positive = cfg["shift"] == "flush" and split == "train"
    margins = _sample_margins(rng, cfg, h, w, present, zero_sides, all_positive,
                              wide=role["kind"] == "anchor")
    if margins is None:
        return None
    exists = {z: all(margins[s] > 0 for s in _zone_sides(z)) for z in ZONES}
    missing = [z for z in ZONES if z not in present]
    need_q = set()
    need_margin = bool(role.get("margin"))
    if role["kind"] == "anchor":
        need_q = set(ZONES)
    elif role["kind"] == "default":
        need_q = {present[int(rng.integers(len(present)))], missing[int(rng.integers(len(missing)))]}
    elif split == "test":
        need_q = {present[int(rng.integers(len(present)))]}
        if role.get("flush"):
            gone = [z for z in ZONES if not exists[z]]
            need_q.add(gone[int(rng.integers(len(gone)))])
        elif missing and rng.random() < 0.7:
            need_q.add(missing[int(rng.integers(len(missing)))])
        else:
            need_margin = True
    mask = _sample_mask(rng, cfg, h, w, need_q, need_margin)
    if mask is None:
        return None
    bg, marker, payload = _pair_palette(rng, cfg, split)
    pool = [c for c in range(10) if c not in (bg, marker, payload)]
    distinct = role["kind"] == "anchor" or rng.random() < 0.72
    keys = _key_colours(rng, pool, present, distinct)
    if keys is None:
        return None
    if split == "test" and cfg.get("fresh") and present and rng.random() < 0.8:
        # palette shift: try to show one key colour never seen in training
        fresh = [c for c in cfg["fresh"] if c not in (bg, marker, payload)]
        if fresh and not any(k in fresh for k in keys.values()):
            keys[present[int(rng.integers(len(present)))]] = int(fresh[int(rng.integers(len(fresh)))])
            if len(keys) == 4 and len(set(keys.values())) == 1:
                return None
            if distinct and len(set(keys.values())) < len(keys):
                return None
    H = margins["top"] + h + 2 + margins["bottom"]
    W = margins["left"] + w + 2 + margins["right"]
    r1, c1 = margins["top"], margins["left"]
    r2, c2 = r1 + h + 1, c1 + w + 1
    grid = np.full((H, W), bg, dtype=int)
    for r in (r1, r2):
        for c in (c1, c2):
            grid[r, c] = marker
    grid[r1 + 1:r2, c1 + 1:c2][mask] = payload
    weights = cfg["place_w"]
    for zone in present:
        style = PLACE_STYLES[int(rng.choice(3, p=weights))]
        _place_key(rng, cfg, grid, zone, _zone_box(zone, H, W, r1, r2, c1, c2), keys[zone], style)
    if not _background_dominates(grid, bg):
        return None
    try:
        info = _analyse(grid)
    except ValueError:
        return None
    if not _role_ok(info, grid, role, split):
        return None
    return {"input": grid, "output": _render(info)}


def _role_ok(info, grid, role, split):
    keys = info["keys"]
    if role["kind"] == "anchor":
        if len(keys) != 4 or len(set(keys.values())) != 4:
            return False
        if not all(_payload_in(info, z) for z in ZONES):
            return False
        if not _nearest_conflict(info):
            return False
    if role["kind"] == "default":
        miss = [z for z in ZONES if z not in keys]
        if not miss or not keys:
            return False
        if not any(_payload_in(info, z) for z in miss) or not any(_payload_in(info, z) for z in keys):
            return False
    if role.get("margin") and _bbox_is_interior(info):
        return False
    if split == "test":
        return _test_ok(info, grid) and (not role.get("flush") or _flush_ok(info))
    return True


def _test_ok(info, grid):
    out = _render(info)
    if not any(_payload_in(info, z) for z in info["keys"]):
        return False
    a_matters = (_differs(out, _alt_missing_keeps_payload(info)) or not _bbox_is_interior(info))
    return bool(a_matters and _differs(out, _alt_no_legend(info)) and _differs(out, _alt_b_literal(grid)))


def _flush_ok(info):
    m = info["margins"]
    if min(m.values()) != 0:
        return False
    return any(not info["zone_exists"][z] and _payload_in(info, z) for z in ZONES)


def _make_pair(rng, cfg, split, role, tries=200):
    for _ in range(tries):
        pair = _try_pair(rng, cfg, split, role)
        if pair is not None:
            return pair
    return None


def _config(rng):
    shift = SHIFT_MODES[int(rng.choice(4, p=[0.28, 0.26, 0.24, 0.22]))]
    cfg = {
        "shift": shift,
        "zero_rate": float(rng.choice([0.0, 0.1, 0.25])) if shift != "flush" else 0.0,
        "object_rate": float(rng.choice([0.0, 0.0, 0.25, 0.5])),
        "place_w": rng.dirichlet([1.2, 1.2, 1.6]),
        "mask_w": rng.dirichlet([1.0] * len(MASK_STYLES)),
    }
    bg = 0 if rng.random() < 0.55 else int(rng.integers(10))
    others = [c for c in range(10) if c != bg]
    if shift == "palette":
        picks = rng.choice(others, size=4, replace=False).tolist()
        n_m = 1 if rng.random() < 0.6 else 2
        n_p = 1 if rng.random() < 0.6 else 2
        cfg.update(roles="pooled", bg=bg, pool_m=picks[:n_m], pool_p=picks[2:2 + n_p])
    elif rng.random() < 0.45:
        m, p = rng.choice(others, size=2, replace=False).tolist()
        cfg.update(roles="fixed", fixed=(bg, int(m), int(p)))
    else:
        cfg.update(roles="varied")
    return cfg


def _palette_shift(cfg, train_infos):
    """Test palette for the 'palette' shift: training payload colours become
    markers, training marker colours become payloads (as in source 3de23699)."""
    train_m = {i["marker"] for i in train_infos}
    train_p = {i["payload"] for i in train_infos}
    train_k = {k for i in train_infos for k in i["keys"].values()}
    seen = train_m | train_p | train_k | {i["bg"] for i in train_infos}
    bg = cfg["bg"]

    def draw(rng):
        m_pool = sorted((train_p - train_m - {bg})) or sorted(train_k - train_m - {bg})
        m = int(rng.choice(m_pool))
        p_pool = sorted(train_m - {bg, m}) or sorted(train_k - {bg, m})
        p = int(rng.choice(p_pool))
        return bg, m, p

    cfg["test_palette"] = draw
    cfg["fresh"] = sorted(set(range(10)) - seen)


def _attempt(rng, num_train, num_test):
    cfg = _config(rng)
    roles = [{"kind": "anchor"}, {"kind": "default"}] + [{"kind": "free"} for _ in range(num_train - 2)]
    roles[int(rng.integers(len(roles)))]["margin"] = True
    order = rng.permutation(num_train)
    roles = [roles[i] for i in order]
    train, seen = [], set()
    for role in roles:
        for _ in range(8):
            pair = _make_pair(rng, cfg, "train", role)
            if pair is None:
                return None
            key = pair["input"].tobytes() + bytes(pair["input"].shape)
            if key not in seen:
                break
        else:
            return None
        seen.add(key)
        train.append(pair)
    infos = [_analyse(p["input"]) for p in train]
    if cfg["shift"] == "palette":
        _palette_shift(cfg, infos)
    test = []
    for _ in range(num_test):
        role = {"kind": "test", "flush": cfg["shift"] == "flush"}
        for _ in range(8):
            pair = _make_pair(rng, cfg, "test", role)
            if pair is None:
                return None
            key = pair["input"].tobytes() + bytes(pair["input"].shape)
            if key not in seen:
                break
        else:
            return None
        seen.add(key)
        test.append(pair)
    puzzle = {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in train],
              "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in test]}
    try:
        validate_generated_puzzle(puzzle)
    except (AssertionError, ValueError):
        return None
    return puzzle


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _resolve_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(400):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError(f"could not generate a puzzle for seed {seed}")


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def _shift_report(train_infos, test_infos):
    area = lambda i: i["interior"].size
    aspect = lambda i: max(i["interior"].shape) / min(i["interior"].shape)
    train_m = {i["marker"] for i in train_infos}
    train_other = {i["payload"] for i in train_infos} | {k for i in train_infos for k in i["keys"].values()}
    return {
        "scale": min(map(area, test_infos)) > max(map(area, train_infos)),
        "flush": all(min(i["margins"].values()) > 0 for i in train_infos)
        and all(_flush_ok(i) for i in test_infos),
        "palette": all(i["marker"] not in train_m and i["marker"] in train_other for i in test_infos),
        "aspect": max(map(aspect, train_infos)) <= 2 and min(map(aspect, test_infos)) >= 4,
    }


def _connected(cells):
    cells = set(cells)
    if not cells:
        return True
    stack, seen = [next(iter(cells))], set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        stack.extend(n for n in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)) if n in cells)
    return seen == cells


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys"
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test, "pair counts"
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys"
            grids = {}
            for role in ("input", "output"):
                g = np.asarray(pair[role])
                assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "grid size"
                assert np.issubdtype(g.dtype, np.integer) and g.min() >= 0 and g.max() <= 9, "grid colours"
                grids[role] = g.astype(int)
            key = (grids["input"].shape, grids["input"].tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyse(grids["input"])
            assert np.array_equal(_render(info), grids["output"]), "output does not follow the rule"
            assert np.array_equal(solve(grids["input"]), grids["output"]), "solve mismatch"
            h, w = info["interior"].shape
            assert h % 2 == 0 and w % 2 == 0 and h >= 2 and w >= 2, "interior must be even"
            assert np.any(info["interior"] == info["payload"]), "empty payload"
            assert len({info["bg"], info["marker"], info["payload"]}) == 3, "roles must differ"
            for zone in ZONES:
                assert _connected(info["key_cells"][zone]), "key must be one object"
            ks = list(info["keys"].values())
            assert not (len(ks) == 4 and len(set(ks)) == 1), "four identical keys mimic markers"
            g = grids["input"]
            assert _background_dominates(g, info["bg"]), "background must be the strictly most frequent colour"
            infos[split].append((info, g))
    train = [i for i, _ in infos["train"]]
    shown = {z for i in train for z in i["keys"] if _payload_in(i, z)}
    assert shown == set(ZONES), "training must demonstrate all four zone-to-quadrant links"
    assert any(len({i["keys"][z] for z in i["keys"] if _payload_in(i, z)}) >= 2 for i in train), \
        "training needs a pair whose quadrants take different key colours"
    assert any(any(z not in i["keys"] and _payload_in(i, z) for z in ZONES) for i in train), \
        "training must show the marker-colour default for a keyless quadrant"
    assert any(not _bbox_is_interior(i) for i in train), "training must separate window crop from tight crop"
    assert any(_nearest_conflict(i) for i in train), "training must separate quadrant rule from nearest key"
    assert any(_differs(_render(i), _alt_b_literal(g)) for i, g in infos["train"]), "B-literal reading not refuted"
    for info, g in infos["test"]:
        assert _test_ok(info, g), "test pair must depend on both parent concepts"
    shifts = _shift_report(train, [i for i, _ in infos["test"]])
    assert any(shifts.values()), "no train/test shift"
    return shifts


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------
def self_test():
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {bad}")
    stats = {"shift": set(), "key_counts": set(), "objects": 0, "zero_margin": 0, "shapes": set(),
             "bg": set(), "marker": set(), "payload": set(), "test_keycounts": set(), "styles": set()}
    for seed in range(42):
        n_train = 2 + seed % 9
        n_test = 1 + seed % min(4, n_train - 1)
        puzzle = generate_puzzle(seed, n_train, n_test)
        assert len(puzzle["train"]) == n_train and len(puzzle["test"]) == n_test
        shifts = validate_generated_puzzle(puzzle)
        stats["shift"].update(k for k, v in shifts.items() if v)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyse(np.asarray(pair["input"]))
                stats["key_counts"].add(len(info["keys"]))
                if split == "test":
                    stats["test_keycounts"].add(len(info["keys"]))
                stats["objects"] += any(len(c) > 1 for c in info["key_cells"].values())
                stats["zero_margin"] += min(info["margins"].values()) == 0
                stats["shapes"].add(info["interior"].shape)
                stats["bg"].add(info["bg"])
                stats["marker"].add(info["marker"])
                stats["payload"].add(info["payload"])
    assert generate_puzzle(7) == generate_puzzle(7), "nondeterministic"
    assert stats["shift"] == set(SHIFT_MODES), f"shift modes {stats['shift']}"
    assert stats["key_counts"] == {0, 1, 2, 3, 4}, stats["key_counts"]
    assert stats["objects"] > 0 and stats["zero_margin"] > 0
    assert len(stats["shapes"]) >= 40, len(stats["shapes"])
    for role in ("bg", "marker", "payload"):
        assert len(stats[role]) >= 6, (role, stats[role])
    # corrupted inputs are rejected rather than guessed
    broken_inputs = []
    for pair in generate_puzzle(3, 6, 1)["train"]:
        base = np.asarray(pair["input"])
        info = _analyse(base)
        spare = [c for c in range(10) if c not in (info["bg"], info["marker"], info["payload"])
                 and c not in info["keys"].values()][0]
        extra = base.copy()
        extra[info["r1"], info["c1"] + 1] = info["marker"]          # fifth marker cell
        broken_inputs.append(extra)
        if info["margins"]["top"] > 0:
            strip = base.copy()
            strip[info["r1"] - 1, info["c1"] + 1] = spare             # key in a non-diagonal strip
            broken_inputs.append(strip)
        for zone in info["keys"]:
            r, c = info["key_cells"][zone][0]
            mixed = base.copy()
            box = _zone_box(zone, *base.shape, info["r1"], info["r2"], info["c1"], info["c2"])
            for rr in range(box[0][0], box[0][1] + 1):
                for cc in range(box[1][0], box[1][1] + 1):
                    if (rr, cc) not in info["key_cells"][zone] and mixed[rr, cc] == info["bg"]:
                        mixed[rr, cc] = spare                       # second colour in a zone
                        broken_inputs.append(mixed)
                        break
                else:
                    continue
                break
    assert len(broken_inputs) >= 6
    for broken in broken_inputs:
        try:
            solve(broken)
        except ValueError:
            continue
        raise AssertionError("ambiguous or malformed input accepted")
    return True
