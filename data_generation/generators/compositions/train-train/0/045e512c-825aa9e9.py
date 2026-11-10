"""Stencil rainfall: template rays are stamped in the air, then every stamped copy
settles under gravity onto a fixed scaffold while keeping a one-cell moat.

Parent A (045e512c) contributes the complete template, the partial first-copy
fragments that pick a ray direction and colour, the stride of template size
plus one, ray repetition to the grid edge and clipping by any edge the ray
leaves through.  Parent B (825aa9e9) contributes the fixed scaffold (steps,
posts, notches, teeth, bays, wells, separated or attached shelves), straight
rigid falls, bottom-first settling and the eight-neighbour one-cell clearance.
The connecting concept is that A's one-cell gap between copies and B's one-cell
scaffold clearance become a single "moat" rule: every settled copy keeps one
background cell (eight-neighbourhood) between itself and the scaffold, the
suspended template and every other settled copy.

Canonical frame: gravity points toward the grid edge that is completely
covered by the scaffold colour.  Within that frame rays may point left, right,
up-left, up-right, straight up (away from the ground), down-left or down-right.
Straight down is impossible: its copies could only leave the grid through the
ground edge.  Down-diagonal rays always leave through a side edge before they
reach the ground's moat.
"""
from __future__ import annotations

import numpy as np

# (row step, column step) in units of the template stride (h + 1, w + 1).
DIRECTIONS = {"L": (0, -1), "R": (0, 1), "UL": (-1, -1), "UR": (-1, 1), "U": (-1, 0),
              "DL": (1, -1), "DR": (1, 1)}
NAMES = {value: key for key, value in DIRECTIONS.items()}
NEIGHBORS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
TERRAIN_STYLES = ("steps", "posts", "notched", "teeth", "bays", "rugged",
                  "wells", "ramp", "comb", "cantilever")
SIMPLE_STYLES = ("steps", "posts", "notched", "teeth", "bays")
MASK_STYLES = ("irregular", "irregular", "pendant", "pendant", "arch", "t_down", "cross", "ring",
               "cup", "ell", "stairs", "comb", "H", "zigzag", "X", "diag", "hook", "frame_stem")
RIGID_STYLES = ("stairs", "X", "diag", "irregular", "diag")
GAP_STYLES = ("arch", "comb", "H", "X", "irregular", "comb")  # bottom rows that can split apart
# Features a test may rely on only if some training pair already shows them.
GATED = ("ledge", "ledge_under", "clip_fall", "top_clip_fall", "rigid", "stack3",
         "moving_stack", "u_fall", "rigid8") + tuple("dir:" + name for name in DIRECTIONS)
# Every training set must refute these shortcuts / show these mechanisms.
CORE_REQUIRED = ("stack", "moving_stack", "order", "moat8", "shape", "diag", "template_fixed")
STRUCTURAL = ("stack3", "ledge", "ledge_under", "ledge_catch_h", "top_clip_fall", "rigid", "rigid8",
              "clip_fall", "u_fall")


# ----------------------------------------------------------------------------
# Small grid helpers
# ----------------------------------------------------------------------------

def _shift_or(out, mask, dr, dc):
    h, w = mask.shape
    rs = slice(max(0, -dr), h - max(0, dr))
    rd = slice(max(0, dr), h - max(0, -dr))
    cs = slice(max(0, -dc), w - max(0, dc))
    cd = slice(max(0, dc), w - max(0, -dc))
    out[rd, cd] |= mask[rs, cs]


def _dilate(mask, four=False):
    out = mask.copy()
    for dr, dc in (NEIGHBORS4 if four else NEIGHBORS8):
        _shift_or(out, mask, dr, dc)
    return out


def _components(cells, four=False):
    """Connected components of a list of cells (8- or 4-connectivity)."""
    remaining = set(cells)
    steps = NEIGHBORS4 if four else NEIGHBORS8
    result = []
    for start in sorted(remaining):
        if start not in remaining:
            continue
        remaining.discard(start)
        stack, comp = [start], [start]
        while stack:
            y, x = stack.pop()
            for dr, dc in steps:
                q = (y + dr, x + dc)
                if q in remaining:
                    remaining.discard(q)
                    stack.append(q)
                    comp.append(q)
        result.append(sorted(comp))
    return result


def _components8(mask):
    return _components([(int(r), int(c)) for r, c in np.argwhere(mask)])


def _cells_mask(shape, cells):
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


# ----------------------------------------------------------------------------
# Input-only solver
# ----------------------------------------------------------------------------

def _parse(grid):
    """Recover every role from one input grid.  Raises ValueError if illegal."""
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a nonempty 2D array")
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(values) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("background colour is not a strict plurality")
    background = int(values[order[0]])
    found = []
    for rot in range(4):
        bottom = np.rot90(grid, rot)[-1]
        if np.all(bottom == bottom[0]) and int(bottom[0]) != background:
            found.append(rot)
    if len(found) != 1:
        raise ValueError("expected exactly one edge fully covered by the scaffold")
    rot = found[0]
    canon = np.ascontiguousarray(np.rot90(grid, rot))
    scaffold_color = int(canon[-1, 0])
    scaffold = canon == scaffold_color
    rest = (canon != background) & ~scaffold
    pieces = []
    for color in sorted({int(v) for v in canon[rest]}):
        for comp in _components8(canon == color):
            pieces.append((len(comp), color, comp))
    pieces.sort(key=lambda item: -item[0])
    if not pieces or (len(pieces) > 1 and pieces[0][0] == pieces[1][0]):
        raise ValueError("no unique largest template object")
    _, template_color, cells = pieces[0]
    if int((canon == template_color).sum()) != len(cells):
        raise ValueError("template colour must belong to the template only")
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    h, w = max(rows) - r0 + 1, max(cols) - c0 + 1
    stencil = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        stencil[r - r0, c - c0] = True
    rays, fragments = {}, {}
    for r, c in np.argwhere(rest & (canon != template_color)):
        r, c = int(r), int(c)
        i, j = (r - r0) // (h + 1), (c - c0) // (w + 1)
        if (i, j) not in NAMES:
            raise ValueError("fragment outside the seven legal first-copy boxes")
        lr, lc = r - r0 - i * (h + 1), c - c0 - j * (w + 1)
        if not (0 <= lr < h and 0 <= lc < w and stencil[lr, lc]):
            raise ValueError("fragment cell is not on the template stencil")
        color = int(canon[r, c])
        if rays.setdefault((i, j), color) != color:
            raise ValueError("a ray fragment must be monochrome")
        fragments.setdefault((i, j), []).append((r, c))
    if not rays:
        raise ValueError("no ray fragments")
    return {"grid": canon, "rot": rot, "background": background,
            "scaffold_color": scaffold_color, "scaffold": scaffold,
            "template_color": template_color, "template": sorted(cells),
            "origin": (r0, c0), "size": (h, w), "stencil": stencil,
            "rays": rays, "fragments": fragments}


def _emit(scene, rays=None):
    """Draw every ray exactly as parent A would: stride size+1, clip at edges."""
    height, width = scene["grid"].shape
    r0, c0 = scene["origin"]
    h, w = scene["size"]
    offsets = [(int(a), int(b)) for a, b in np.argwhere(scene["stencil"])]
    rays = scene["rays"] if rays is None else rays
    copies = []
    for (i, j), color in sorted(rays.items()):
        k = 1
        while k <= 64:
            br, bc = r0 + k * i * (h + 1), c0 + k * j * (w + 1)
            cells = [(br + a, bc + b) for a, b in offsets
                     if 0 <= br + a < height and 0 <= bc + b < width]
            if not cells:
                break
            copies.append({"dir": (i, j), "index": k, "color": color, "cells": cells,
                           "full": len(cells) == len(offsets), "band": j * k,
                           "top_clipped": br < 0,
                           "top": min(r for r, _ in cells), "bottom": max(r for r, _ in cells)})
            k += 1
    return copies


def _settle(shape, obstacles, pieces, mode="moat", collide=None):
    """Drop pieces straight down, lowest first.  Returns fall distances.

    mode "moat": one-cell eight-neighbour clearance from obstacles and from
    previously settled pieces (the family rule).  Alternatives used only to
    prove that demonstrations distinguish shortcuts: "touch" (clearance from
    obstacles only, pieces may touch), "four" (four-neighbour clearance),
    "none" (no clearance at all).
    """
    height = shape[0]
    four = mode == "four"
    blocked = obstacles.copy() if mode == "none" else _dilate(obstacles, four=four)
    collide = pieces if collide is None else collide
    order = sorted(range(len(pieces)),
                   key=lambda n: (-max(r for r, _ in pieces[n]), min(c for _, c in pieces[n]), n))
    falls = [0] * len(pieces)
    for n in order:
        body = collide[n]
        fall = 0
        while all(r + fall + 1 < height and not blocked[r + fall + 1, c] for r, c in body):
            fall += 1
        falls[n] = fall
        landed = _cells_mask(shape, [(r + fall, c) for r, c in pieces[n]])
        if mode in ("touch", "none"):
            blocked |= landed
        else:
            blocked |= _dilate(landed, four=four)
    return falls


def _settle_top_first(shape, obstacles, pieces):
    """Shortcut: highest piece first, while unlanded pieces still block."""
    height = shape[0]
    pos = [0] * len(pieces)
    order = sorted(range(len(pieces)),
                   key=lambda n: (max(r for r, _ in pieces[n]), min(c for _, c in pieces[n]), n))
    for n in order:
        others = obstacles.copy()
        for m, piece in enumerate(pieces):
            if m != n:
                for r, c in piece:
                    others[r + pos[m], c] = True
        blocked = _dilate(others)
        fall = 0
        while all(r + fall + 1 < height and not blocked[r + fall + 1, c] for r, c in pieces[n]):
            fall += 1
        pos[n] = fall
    return pos


def _render(scene, copies, falls, template_fall=0):
    canon = scene["grid"]
    out = np.full(canon.shape, scene["background"], dtype=int)
    out[scene["scaffold"]] = scene["scaffold_color"]
    for r, c in scene["template"]:
        out[r + template_fall, c] = scene["template_color"]
    for copy, fall in zip(copies, falls):
        for r, c in copy["cells"]:
            out[r + fall, c] = copy["color"]
    return out


def _obstacles(scene, scaffold=None):
    obstacles = (scene["scaffold"] if scaffold is None else scaffold).copy()
    for r, c in scene["template"]:
        obstacles[r, c] = True
    return obstacles


def _solve_scene(scene):
    copies = _emit(scene)
    falls = _settle(scene["grid"].shape, _obstacles(scene), [c["cells"] for c in copies])
    return _render(scene, copies, falls), copies, falls


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the output from the input alone."""
    scene = _parse(input_grid)
    out, _, _ = _solve_scene(scene)
    return np.rot90(out, -scene["rot"]).astype(int)


# ----------------------------------------------------------------------------
# Shortcut solvers and feature analysis (used by generation and validation)
# ----------------------------------------------------------------------------

def _alternatives(scene):
    """Canonical outputs of plausible shortcut rules."""
    shape = scene["grid"].shape
    copies = _emit(scene)
    bodies = [c["cells"] for c in copies]
    obstacles = _obstacles(scene)
    alt = {"no_fall": _render(scene, copies, [0] * len(copies))}
    for mode in ("touch", "four", "none"):
        alt[mode] = _render(scene, copies, _settle(shape, obstacles, bodies, mode=mode))
    boxes = []
    for body in bodies:
        rs = [r for r, _ in body]
        cs = [c for _, c in body]
        boxes.append([(r, c) for r in range(min(rs), max(rs) + 1) for c in range(min(cs), max(cs) + 1)])
    alt["bbox"] = _render(scene, copies, _settle(shape, obstacles, bodies, collide=boxes))
    flat = {}
    for (i, j), color in sorted(scene["rays"].items(), key=lambda item: (item[0][0] != 0, item[0])):
        flat.setdefault((0, j) if j else (i, j), color)
    flat_copies = _emit(scene, flat)
    alt["diag_flat"] = _render(scene, flat_copies,
                               _settle(shape, obstacles, [c["cells"] for c in flat_copies]))
    # Parent-B-only reading: the raw fragments themselves fall.
    frags = [scene["fragments"][d] for d in sorted(scene["fragments"])]
    frag_falls = _settle(shape, obstacles, frags)
    out = np.full(shape, scene["background"], dtype=int)
    out[scene["scaffold"]] = scene["scaffold_color"]
    for r, c in scene["template"]:
        out[r, c] = scene["template_color"]
    for d, cells, fall in zip(sorted(scene["fragments"]), frags, frag_falls):
        for r, c in cells:
            out[r + fall, c] = scene["rays"][d]
    alt["fragments_fall"] = out
    # Template treated as one more falling payload.
    falls = _settle(shape, scene["scaffold"], [scene["template"]] + bodies)
    alt["template_falls"] = _render(scene, copies, falls[1:], template_fall=falls[0])
    # Parent-B-literal segmentation: each 4-connected piece of a copy falls alone;
    # "split8" is the same with 8-connected pieces (differs only for clipped
    # copies whose visible part is disconnected).
    for name, four in (("split4", True), ("split8", False)):
        pieces, colors = [], []
        for copy in copies:
            for comp in _components(copy["cells"], four=four):
                pieces.append(comp)
                colors.append(copy["color"])
        split = [{"cells": p, "color": col} for p, col in zip(pieces, colors)]
        alt[name] = _render(scene, split, _settle(shape, obstacles, pieces))
    # Highest-first processing in which unlanded copies still block.
    alt["top_first"] = _render(scene, copies, _settle_top_first(shape, obstacles, bodies))
    return alt


def _overhang(scaffold):
    """Scaffold cells with background somewhere below them (ledges and arms)."""
    height, width = scaffold.shape
    support = np.zeros_like(scaffold)
    for c in range(width):
        r = height - 1
        while r >= 0 and scaffold[r, c]:
            support[r, c] = True
            r -= 1
    return scaffold & ~support


def _features(scene):
    """Visible-grid features of one canonical scene, all derived from the input."""
    shape = scene["grid"].shape
    out, copies, falls = _solve_scene(scene)
    alt = _alternatives(scene)
    obstacles = _obstacles(scene)
    feats = {"dir:" + NAMES[d] for d in scene["rays"]}
    for name, key in (("no_fall", "a_only"), ("fragments_fall", "b_only"), ("touch", "stack"),
                      ("four", "moat8"), ("bbox", "shape"), ("diag_flat", "diag"),
                      ("template_falls", "template_fixed"), ("none", "moat"),
                      ("split4", "rigid"), ("split8", "rigid8"), ("top_first", "order")):
        if not np.array_equal(alt[name], out):
            feats.add(key)
    if any(d[0] > 0 for d in scene["rays"]):
        feats.add("down")
    solo = [_settle(shape, obstacles, [copy["cells"]])[0] for copy in copies]
    resting = [n for n in range(len(copies)) if solo[n] != falls[n]]
    if resting:
        feats.add("rests_on_copy")
    if any(copies[n]["dir"] != (-1, 0) and falls[n] > 0 for n in resting):
        feats.add("moving_stack")
    bands = {}
    for n, copy in enumerate(copies):
        if copy["band"]:
            bands.setdefault(copy["band"], []).append(n)
    if any(len(ns) >= 3 and sum(n in resting for n in ns) >= 2 for ns in bands.values()):
        feats.add("stack3")
    over = _overhang(scene["scaffold"])
    if over.any():
        bare = _settle(shape, _obstacles(scene, scene["scaffold"] & ~over), [c["cells"] for c in copies])
        caught = [n for n in range(len(copies)) if bare[n] != falls[n]]
        if caught:
            feats.add("ledge")
            if any(copies[n]["dir"][0] == 0 for n in caught):
                feats.add("ledge_catch_h")
            over_cells = [(int(r), int(c)) for r, c in np.argwhere(over)]
            for n in caught:
                for m in bands.get(copies[n]["band"], []):
                    if m == n:
                        continue
                    top = copies[m]["top"] + falls[m]
                    cmin = min(c for _, c in copies[m]["cells"]) - 1
                    cmax = max(c for _, c in copies[m]["cells"]) + 1
                    if any(r < top and cmin <= c <= cmax for r, c in over_cells):
                        feats.add("ledge_under")
    if any(copy["dir"] == (-1, 0) and fall > 0 for copy, fall in zip(copies, falls)):
        feats.add("u_fall")  # a straight-up copy whose stencil leaves room to sink
    if any((not copy["full"]) and fall > 0 for copy, fall in zip(copies, falls)):
        feats.add("clip_fall")
    if any(copy["top_clipped"] and fall > 0 for copy, fall in zip(copies, falls)):
        feats.add("top_clip_fall")
    if any(not copy["full"] for copy in copies):
        feats.add("clip")
    falling = [f for copy, f in zip(copies, falls) if copy["dir"] != (-1, 0)]
    rows = {copy["top"] + f for copy, f in zip(copies, falls) if copy["dir"] != (-1, 0)}
    info = {"falls": falls, "falling": falling, "copies": len(copies),
            "distinct_falls": len(set(falling)), "landing_rows": len(rows),
            "moved": sum(f > 0 for f in falls)}
    return feats, info


# ----------------------------------------------------------------------------
# Generation: templates, fragments, directions
# ----------------------------------------------------------------------------

def _connected8(mask):
    return bool(mask.any()) and len(_components8(mask)) == 1


def _random_mask(rng, h, w):
    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[0, 0] = True
    while r < h - 1 or c < w - 1:
        steps = []
        if r < h - 1:
            steps.append((1, 0))
        if c < w - 1:
            steps.append((0, 1))
        if r < h - 1 and c < w - 1:
            steps.append((1, 1))
        dr, dc = steps[int(rng.integers(len(steps)))]
        r, c = r + dr, c + dc
        mask[r, c] = True
    goal = int(rng.integers(int(mask.sum()), h * w))
    while int(mask.sum()) < goal:
        frontier = sorted({(int(y) + dr, int(x) + dc) for y, x in np.argwhere(mask)
                           for dr, dc in NEIGHBORS8
                           if 0 <= int(y) + dr < h and 0 <= int(x) + dc < w
                           and not mask[int(y) + dr, int(x) + dc]})
        y, x = frontier[int(rng.integers(len(frontier)))]
        mask[y, x] = True
    if rng.random() < 0.5:
        mask = mask[:, ::-1]
    if rng.random() < 0.5:
        mask = mask[::-1, :]
    return mask


def _diag_line(h, w):
    m = np.zeros((h, w), dtype=bool)
    if h >= w:
        for r in range(h):
            m[r, (r * (w - 1)) // max(1, h - 1)] = True
    else:
        for c in range(w):
            m[(c * (h - 1)) // max(1, w - 1), c] = True
    return m


def _style_mask(rng, style, h, w):
    m = np.zeros((h, w), dtype=bool)
    if style == "irregular":
        return _random_mask(rng, h, w)
    if style == "pendant":  # broad body with a one-cell-wide pendant stem
        body = int(rng.integers(1, h))
        m[:body, :] = True
        if body >= 3 and rng.random() < 0.5:
            m[1:body - 1, 1:w - 1] = False
        col = int(rng.choice([0, w - 1, int(rng.integers(w))]))
        m[body:, col] = True
    elif style == "arch":  # legs hang down at both sides
        m[0, :] = True
        m[:, 0] = True
        m[:, w - 1] = True
    elif style == "t_down":
        m[0, :] = True
        m[:, int(rng.choice([(w - 1) // 2, w // 2]))] = True
    elif style == "cross":
        m[int(rng.choice([(h - 1) // 2, h // 2])), :] = True
        m[:, int(rng.choice([(w - 1) // 2, w // 2]))] = True
    elif style == "ring":
        m[0, :] = m[-1, :] = True
        m[:, 0] = m[:, -1] = True
    elif style == "cup":
        m[-1, :] = True
        m[:, 0] = m[:, -1] = True
    elif style == "ell":
        m[:, 0] = True
        m[int(rng.choice([0, h - 1])), :] = True
        if rng.random() < 0.5:
            m = m[:, ::-1]
    elif style == "stairs":
        for r in range(h):
            lo = int(round(r * (w - 1) / max(1, h - 1)))
            hi = min(w - 1, lo + int(rng.integers(0, 2)))
            m[r, lo:hi + 1] = True
        for r in range(1, h):
            prev = np.flatnonzero(m[r - 1])
            cur = np.flatnonzero(m[r])
            if cur.min() > prev.max() + 1:
                m[r, prev.max() + 1:cur.min()] = True
        if rng.random() < 0.5:
            m = m[:, ::-1]
    elif style == "comb":  # a bar with two or more hanging teeth of varied length
        m[0, :] = True
        cols = [c for c in range(w) if rng.random() < 0.6]
        if len(cols) < 2:
            cols = [0, w - 1]
        for c in cols:
            m[1:int(rng.integers(2, h + 1)), c] = True
        m[1:, cols[int(rng.integers(len(cols)))]] = True
        if rng.random() < 0.3:
            m = m[::-1, :]
    elif style == "H":
        m[:, 0] = m[:, -1] = True
        m[int(rng.integers(1, max(2, h - 1))), :] = True
    elif style == "zigzag":
        k = int(rng.integers(0, w))
        m[0, :k + 1] = True
        m[:, k] = True
        m[-1, k:] = True
        if rng.random() < 0.5:
            m = m[:, ::-1]
    elif style == "X":
        m = _diag_line(h, w) | _diag_line(h, w)[:, ::-1]
    elif style == "diag":
        m = _diag_line(h, w)
        if rng.random() < 0.5:
            m = m[:, ::-1]
        if rng.random() < 0.4:  # a short foot or cap keeps it from being a bare line
            r = int(rng.choice([0, h - 1]))
            c = int(np.flatnonzero(m[r])[0])
            m[r, max(0, c - 1):min(w, c + 2)] = True
    elif style == "hook":
        m[0, :] = True
        m[:, 0] = True
        m[:int(rng.integers(2, max(3, h))), w - 1] = True
        if rng.random() < 0.5:
            m = m[:, ::-1]
    else:  # frame_stem: a small ring with a stem below it
        ring = int(rng.integers(3, h)) if h >= 4 else 2
        m[0, :] = m[ring - 1, :] = True
        m[:ring, 0] = m[:ring, -1] = True
        m[ring:, int(rng.integers(w))] = True
    return m


def _mask_ok(m):
    h, w = m.shape
    return (3 <= int(m.sum()) < h * w and m[0].any() and m[-1].any()
            and m[:, 0].any() and m[:, -1].any() and _connected8(m))


def _sample_mask(rng, h, w, rigid=False, gap=False):
    styles = GAP_STYLES if gap else (RIGID_STYLES if rigid else MASK_STYLES)
    for attempt in range(300):
        style = styles[int(rng.integers(len(styles)))]
        m = _style_mask(rng, style, h, w)
        if not _mask_ok(m):
            continue
        if rigid and attempt < 250:
            cells = [(int(r), int(c)) for r, c in np.argwhere(m)]
            if len(_components(cells, four=True)) < 2:
                continue
        if gap and attempt < 250:
            bottom = [(0, int(c)) for c in np.flatnonzero(m[-1])]
            if len(_components(bottom)) < 2:
                continue
        return m
    raise RuntimeError("template sampling failed")


def _sample_size(rng, ctx, rigid=False):
    if ctx["regime"] == "minimal":
        return int(rng.choice([2, 3, 4])), int(rng.choice([2, 3, 4]))
    h = int(rng.choice([2, 3, 4, 5, 6], p=[0.24, 0.38, 0.21, 0.11, 0.06]))
    w = int(rng.choice([2, 3, 4, 5, 6], p=[0.16, 0.36, 0.27, 0.13, 0.08]))
    if rigid:
        h, w = max(h, 3), max(w, 3)
    return h, w


def _fragment(rng, stencil):
    points = [(int(r), int(c)) for r, c in np.argwhere(stencil)]
    count = len(points)
    h, w = stencil.shape
    mode = int(rng.integers(6))
    if mode == 0:
        chosen = [points[int(rng.integers(count))]]
    elif mode == 1:  # nearly complete
        keep = count - int(rng.integers(1, min(3, count - 1) + 1))
        idx = rng.choice(count, size=max(1, keep), replace=False)
        chosen = [points[int(i)] for i in idx]
    elif mode == 2:  # a contiguous run along one side of the box
        side = int(rng.integers(4))
        edge = [p for p in points if (p[0] == 0, p[0] == h - 1, p[1] == 0, p[1] == w - 1)[side]]
        edge.sort(key=lambda p: (p[1], p[0]) if side < 2 else (p[0], p[1]))
        length = int(rng.integers(1, max(1, min(len(edge), count - 1)) + 1))
        start = int(rng.integers(0, len(edge) - length + 1))
        chosen = edge[start:start + length]
    elif mode == 3:  # a corner region
        corner = ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1))[int(rng.integers(4))]
        ordered = sorted(points, key=lambda p: (abs(p[0] - corner[0]) + abs(p[1] - corner[1]), p))
        chosen = ordered[:int(rng.integers(1, min(4, count - 1) + 1))]
    elif mode == 4:  # connected piece
        size = int(rng.integers(1, count))
        chosen = {points[int(rng.integers(count))]}
        while len(chosen) < size:
            frontier = sorted(p for p in points if p not in chosen and
                              any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1 for q in chosen))
            chosen.add(frontier[int(rng.integers(len(frontier)))])
        chosen = sorted(chosen)
    else:
        size = int(rng.integers(1, count))
        idx = rng.choice(count, size=size, replace=False)
        chosen = [points[int(i)] for i in idx]
    if len(chosen) >= count:
        chosen = chosen[:count - 1]
    return sorted(set(chosen))


def _sample_dirs(rng, ctx, focus, fixed=None):
    if fixed is not None:
        return set(fixed)
    dirs = set()
    down_ok = ctx["down"]
    side = "L" if rng.random() < 0.5 else "R"
    if "stack3" in focus and down_ok:
        dirs.update({side, "U" + side, "D" + side})
    elif focus & {"stack", "moving_stack", "order", "rests_on_copy", "ledge_under"}:
        if down_ok and rng.random() < 0.3:
            dirs.update({side, "D" + side})
        else:
            dirs.update({side, "U" + side})
    if "down" in focus and down_ok and not (dirs & {"DL", "DR"}):
        dirs.add("D" + ("L" if rng.random() < 0.5 else "R"))
    if focus & {"top_clip_fall", "rigid8"} and not (dirs & {"UL", "UR"}):
        dirs.add("U" + side)
    for name in sorted(focus):
        if name.startswith("dir:"):
            dirs.add(name[4:])
    if ctx["regime"] == "minimal":
        target = int(rng.choice([1, 2], p=[0.4, 0.6]))
    elif ctx["regime"] == "maximal":
        target = int(rng.choice([3, 4, 5, 6], p=[0.25, 0.35, 0.25, 0.15]))
    else:
        target = int(rng.choice([1, 2, 3, 4, 5], p=[0.18, 0.36, 0.28, 0.13, 0.05]))
    pool = ["L", "R", "UL", "UR", "U"] + (["DL", "DR"] if down_ok else [])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 0.35] + ([0.55, 0.55] if down_ok else []))
    while len(dirs) < target:
        dirs.add(pool[int(rng.choice(len(pool), p=weights / weights.sum()))])
    if all(DIRECTIONS[d][1] == 0 for d in dirs):
        dirs.add(("L", "R", "UL", "UR")[int(rng.integers(4))])
    return dirs


# ----------------------------------------------------------------------------
# Generation: layout, terrain, shelves
# ----------------------------------------------------------------------------

def _layout(rng, h, w, dirs, focus, large, ctx):
    """Template origin, grid size and the up-level count.

    Horizontal rays reach their edge.  A side holding only an up-diagonal ray
    is usually trimmed to its reach, sometimes extended beyond it so empty
    outer bands show where the ray left through the top.  Down-diagonal rays
    must leave through their side before reaching the ground's moat, which
    bounds the side width by the grid depth.  Sides without rays may be wide.
    """
    span = w + 1
    need_two = "ledge_under" in focus
    up = any(DIRECTIONS[d][0] < 0 for d in dirs)
    clip_top = False
    if up:
        lo_m = 2 if need_two and (dirs & {"UL", "UR"}) else 1
        m = int(rng.integers(lo_m, (3 if large else 2) + 1))
        if ctx["top_clip"] and (focus & {"top_clip_fall", "rigid8"} or rng.random() < 0.4):
            delta = int(rng.integers(2, h + 1))
            if "rigid8" in focus and rng.random() < 0.6:
                delta = 2  # only the bottom row of the clipped copy stays visible
            clip_top = True
        else:
            delta = int(rng.integers(0, 2))
        r0 = m * (h + 1) + delta
    else:
        m = 0
        r0 = int(rng.integers(1, 5))
    reach = m + (1 if clip_top else 0)
    hi = 5 if large else 4
    wide = {"minimal": 0.15, "maximal": 0.45}.get(ctx["regime"], 0.35)
    # Deepest band a down-diagonal copy may occupy and still keep a moat row
    # plus one ground row below it inside a 30-row grid.
    kmax = (30 - 3 - (r0 + h - 1)) // (h + 1)

    def side(name):
        hor, upd, dnd = name in dirs, ("U" + name) in dirs, ("D" + name) in dirs
        if not (hor or upd or dnd):
            if rng.random() < wide:
                return int(rng.integers(span + 1, int(2.5 * span) + 1))
            return int(rng.integers(1, span + 1))
        low = 2 if need_two and (upd or dnd) else 1
        if hor:
            lo_f = max(low, m + 1) if (clip_top and upd) else low
            full = int(rng.integers(lo_f, max(lo_f, hi) + 1))
        elif upd:
            if not dnd and rng.random() < 0.3:
                full = int(rng.integers(reach + 1, reach + 3))
            else:
                lo_f = max(low, m + 1) if clip_top else low
                full = int(rng.integers(lo_f, max(lo_f, reach) + 1))
        else:
            full = int(rng.integers(low, 4))
        extra = int(rng.integers(0, w + 1))
        if dnd:
            if kmax < low:
                return None
            if full >= kmax:
                full, extra = kmax, min(extra, 1)
        return full * span + extra

    left = side("L")
    right = side("R")
    if left is None or right is None:
        return None
    c0 = left
    width = c0 + w + right
    depth = int(rng.integers(4 if large else 3, (13 if large else 10) + 1))
    height = r0 + h + depth
    if dirs & {"DL", "DR"}:
        deepest = 0
        for d in ("DL", "DR"):
            if d not in dirs:
                continue
            j = DIRECTIONS[d][1]
            k = 1
            while True:
                bc = c0 + k * j * (w + 1)
                if bc + w - 1 < 0 or bc >= width:
                    break
                deepest = max(deepest, r0 + k * (h + 1) + h - 1)
                k += 1
        height = max(height, deepest + 3 + int(rng.integers(0, 4)))
    return r0, c0, height, width, m


def _terrain(rng, width, cap, style):
    g = np.ones(width, dtype=int)
    if cap <= 1:
        return g
    if style == "steps":
        c, level = 0, int(rng.integers(1, cap + 1))
        while c < width:
            run = int(rng.integers(2, 6))
            g[c:c + run] = level
            level = int(np.clip(level + int(rng.choice([-3, -2, -1, 1, 2, 3])), 1, cap))
            c += run
    elif style == "posts":
        base = int(rng.integers(1, min(3, cap) + 1))
        g[:] = base
        for _ in range(int(rng.integers(2, max(3, width // 3) + 1))):
            c = int(rng.integers(width))
            g[c:c + int(rng.integers(1, 3))] = int(rng.integers(base, cap + 1))
    elif style == "notched":
        base = int(rng.integers(max(1, cap - 2), cap + 1))
        g[:] = base
        for _ in range(int(rng.integers(2, max(3, width // 4) + 1))):
            c = int(rng.integers(width))
            g[c:c + int(rng.integers(1, 5))] = max(1, base - int(rng.integers(1, cap + 1)))
    elif style == "teeth":
        c = 0
        while c < width:
            run = int(rng.integers(5, 12))
            base = int(rng.integers(1, cap + 1))
            period = int(rng.integers(2, 5))
            offset = int(rng.integers(period))
            tooth = int(rng.integers(1, 3))
            for x in range(c, min(width, c + run)):
                g[x] = base + (tooth if (x - c) % period == offset else 0)
            c += run
    elif style == "bays":
        c = 0
        while c < width:
            wall = int(rng.integers(1, 3))
            g[c:c + wall] = cap
            c += wall
            bay = int(rng.integers(3, 8))
            g[c:c + bay] = int(rng.integers(1, max(2, cap - 1) + 1))
            c += bay
    elif style == "wells":  # high plateau pierced by narrow pits
        base = int(rng.integers(max(1, cap - 2), cap + 1))
        g[:] = base
        c = int(rng.integers(0, 3))
        while c < width:
            pit = int(rng.integers(1, 3))
            g[c:c + pit] = int(rng.integers(1, max(1, base - 2) + 1))
            c += pit + int(rng.integers(2, 6))
    elif style == "ramp":  # monotone slopes that turn at the limits
        level = int(rng.integers(1, cap + 1))
        step = 1 if rng.random() < 0.5 else -1
        c = 0
        while c < width:
            run = int(rng.integers(1, 3))
            g[c:c + run] = level
            if not 1 <= level + step <= cap:
                step = -step
            level = int(np.clip(level + step, 1, cap))
            c += run
    elif style == "comb":  # one-wide teeth and one-wide slots
        c = 0
        while c < width:
            run = int(rng.integers(6, 14))
            base = int(rng.integers(1, max(1, cap - 2) + 1))
            tooth = int(rng.integers(2, 5))
            offset = int(rng.integers(0, 2))
            for x in range(c, min(width, c + run)):
                g[x] = base + (tooth if (x - c) % 2 == offset else 0)
            c += run
    elif style == "cantilever":  # tall walls over low floors; arms are attached later
        g[:] = int(rng.integers(1, 3))
        c = int(rng.integers(0, 3))
        while c < width:
            wall = int(rng.integers(1, 3))
            g[c:c + wall] = int(rng.integers(max(1, cap - 3), cap + 1))
            c += wall + int(rng.integers(3, 8))
    else:  # rugged
        c = 0
        while c < width:
            run = int(rng.integers(1, 5))
            g[c:c + run] = int(rng.integers(1, cap + 1))
            c += run
    return np.clip(g, 1, cap)


def _place_shelves(rng, scaffold, zone, copies, geometry, focus, style):
    """Add separated ledges and wall-attached arms in the scaffold colour.

    Every piece keeps a moat from all stamped copies and the template (it lies
    outside `zone`).  Floating ledges keep a moat from all other scaffold;
    attached arms touch only the wall they grow from and overhang background.
    """
    height, width = scaffold.shape
    r0, c0, h, w, ground_top = geometry
    added = []
    goal = 1 + int(rng.random() < 0.3) + int(rng.random() < 0.1)
    want_under = "ledge_under" in focus
    p_attach = 0.65 if style == "cantilever" else 0.2
    bands = {}
    for copy in copies:
        if copy["band"]:
            bands.setdefault(copy["band"], []).append(copy)
    pairs = []
    for band in sorted(bands):
        group = sorted(bands[band], key=lambda cp: cp["top"])
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                upper, lower = group[a], group[b]
                if upper["full"] and lower["top"] - upper["bottom"] >= 4:
                    pairs.append((upper, lower))
    band_ids = sorted(bands)
    for _ in range(90):
        if len(added) >= goal:
            break
        thick = 1 if rng.random() < 0.8 else 2
        roll = rng.random()
        if want_under and not added and pairs:
            mode = "between"
        elif roll < p_attach:
            mode = "attached"
        elif pairs and roll < p_attach + 0.3:
            mode = "between"
        elif len(band_ids) >= 2 and roll < p_attach + 0.5:
            mode = "span"
        else:
            mode = "random"
        attach_col = None
        pillar = None
        if mode == "between":
            upper, lower = pairs[int(rng.integers(len(pairs)))]
            lo, hi = upper["bottom"] + 2, lower["top"] - 2 - (thick - 1)
            if lo > hi:
                continue
            y = int(rng.integers(lo, hi + 1))
            cmin = min(c for _, c in upper["cells"])
            cmax = max(c for _, c in upper["cells"])
            x0 = int(rng.integers(max(0, cmin - 2), cmax + 1))
            lo1, hi1 = max(x0 + 1, cmin), min(width - 1, cmax + 2)
            if lo1 > hi1:
                continue
            x1 = int(rng.integers(lo1, hi1 + 1))
        elif mode == "span":
            k = int(rng.integers(len(band_ids) - 1))
            if band_ids[k + 1] != band_ids[k] + 1:
                continue
            gap = c0 + band_ids[k] * (w + 1) + w
            if not 1 <= gap < width - 1:
                continue
            x0 = int(rng.integers(max(0, gap - w), gap))
            x1 = int(rng.integers(gap + 1, min(width - 1, gap + w) + 1))
            y = int(rng.integers(1, max(2, height - 3)))
        elif mode == "attached" and rng.random() < 0.7:
            # A gallows: a pillar rising from the ground carries an arm that
            # reaches under one copy (B's slab kept attached to its support).
            targets = [cp for cp in copies if cp["band"]]
            if not targets:
                continue
            target = targets[int(rng.integers(len(targets)))]
            cmin = min(c for _, c in target["cells"])
            cmax = max(c for _, c in target["cells"])
            y = int(rng.integers(target["bottom"] + 2, target["bottom"] + 6))
            x0 = int(rng.integers(max(0, cmin - 1), cmax + 1))
            x1 = int(rng.integers(min(width - 1, max(x0 + 1, cmin)), min(width - 1, cmax + 1) + 1))
            col = x0 - 1 if rng.random() < 0.5 else x1 + 1
            if x1 <= x0 or not 0 <= col < width or y + thick - 1 >= height - 3:
                continue
            if any(ground_top[x] <= y + thick for x in range(x0, x1 + 1)):
                continue
            if ground_top[col] > y:
                pillar = np.zeros_like(scaffold)
                pillar[y:int(ground_top[col]), col] = True
            attach_col = col
        elif mode == "attached":
            col = int(rng.integers(width))
            top = int(ground_top[col])
            if top > height - 4:
                continue
            step = 1 if rng.random() < 0.5 else -1
            length = int(rng.integers(2, 6))
            y = int(rng.integers(top, min(height - 4, top + 3) + 1))
            span_cols = [col + step * (i + 1) for i in range(length)]
            if not all(0 <= x < width for x in span_cols):
                continue
            if any(ground_top[x] <= y + thick for x in span_cols):
                continue
            x0, x1 = min(span_cols), max(span_cols)
            attach_col = col
        else:
            if height < 6 or width < 3:
                break
            y = int(rng.integers(1, height - 3))
            length = int(rng.integers(2, 10))
            x0 = int(rng.integers(0, width - 1))
            x1 = min(width - 1, x0 + length - 1)
        if y < 1 or y + thick - 1 >= height - 2:
            continue
        piece = np.zeros_like(scaffold)
        piece[y:y + thick, x0:x1 + 1] = True
        touch = _dilate(piece) & scaffold
        if pillar is not None:
            piece |= pillar
        if (piece & zone).any():
            continue
        if attach_col is None:
            if touch.any():
                continue
        else:
            touch[:, attach_col] = False
            support = scaffold | piece
            if touch.any() or not support[y:y + thick, attach_col].all():
                continue
        scaffold = scaffold | piece
        added.append(piece)
    return added


def _colors(rng, ctx, n_rays):
    background = ctx["background"] if ctx["background"] is not None else int(rng.integers(10))
    used = {background}
    if ctx["scaffold"] is not None and ctx["scaffold"] != background:
        scaffold = ctx["scaffold"]
    else:
        scaffold = int(rng.choice([c for c in range(10) if c not in used]))
    used.add(scaffold)
    template = int(rng.choice([c for c in range(10) if c not in used]))
    used.add(template)
    available = [c for c in range(10) if c not in used]
    if n_rays == 1:
        k = 1
    else:
        mode = rng.random()
        k = 1 if mode < 0.2 else (n_rays if mode < 0.7 else int(rng.integers(1, n_rays + 1)))
    rays = [int(c) for c in rng.choice(available, size=min(k, len(available)), replace=False)]
    return background, scaffold, template, rays


def _canonical_pair(rng, ctx, focus, large=False, dirs_fixed=None):
    """One canonical (gravity-down) input grid, or None on failure."""
    rigid, gap = "rigid" in focus, "rigid8" in focus
    h, w = _sample_size(rng, ctx, rigid or gap)
    stencil = _sample_mask(rng, h, w, rigid=rigid and not gap, gap=gap)
    dirs = _sample_dirs(rng, ctx, focus, dirs_fixed)
    if not ctx["down"] and dirs & {"DL", "DR"}:
        return None
    lay = _layout(rng, h, w, dirs, focus, large, ctx)
    if lay is None:
        return None
    r0, c0, height, width, _ = lay
    if not (6 <= height <= 30 and 6 <= width <= 30):
        return None
    offsets = [(int(a), int(b)) for a, b in np.argwhere(stencil)]
    template = [(r0 + a, c0 + b) for a, b in offsets]
    scene0 = {"grid": np.zeros((height, width), dtype=int), "origin": (r0, c0), "size": (h, w),
              "stencil": stencil, "rays": {DIRECTIONS[d]: 1 for d in sorted(dirs)}}
    copies = _emit(scene0)
    occupied = _cells_mask((height, width), template + [cell for c in copies for cell in c["cells"]])
    zone = _dilate(occupied)
    if zone[-1].any():
        return None
    lowest = np.full(width, -1)
    for c in range(width):
        rows = np.flatnonzero(zone[:, c])
        if len(rows):
            lowest[c] = rows.max()
    cap = height - (r0 + h) - 1
    maxg = np.minimum(cap, height - 1 - lowest)
    styles = ctx["styles"]
    style = styles[int(rng.integers(len(styles)))]
    g = np.minimum(_terrain(rng, width, cap, style), maxg)
    if (g < 1).any():
        return None
    scaffold = np.zeros((height, width), dtype=bool)
    for c in range(width):
        scaffold[height - g[c]:, c] = True
    if ctx["ledges"] and ("ledge" in focus or "ledge_under" in focus or style == "cantilever"
                          or rng.random() < 0.6):
        geometry = (r0, c0, h, w, height - g)
        pieces = _place_shelves(rng, scaffold, zone, copies, geometry, focus, style)
        template_mask = _cells_mask((height, width), template)
        bodies = [c["cells"] for c in copies]

        def falls_with(parts):
            obstacles = scaffold | template_mask
            for part in parts:
                obstacles = obstacles | part
            return _settle((height, width), obstacles, bodies)

        # Keep only shelves that change at least one landing.
        n = 0
        while n < len(pieces):
            others = pieces[:n] + pieces[n + 1:]
            if falls_with(others) == falls_with(pieces):
                pieces = others
            else:
                n += 1
        for piece in pieces:
            scaffold |= piece
    names = sorted(dirs)
    background, scaffold_color, template_color, ray_colors = _colors(rng, ctx, len(names))
    order = [names[int(i)] for i in rng.permutation(len(names))]
    palette = {}
    for n, d in enumerate(order):
        palette[d] = ray_colors[n] if n < len(ray_colors) else ray_colors[int(rng.integers(len(ray_colors)))]
    if rng.random() < 0.15:  # same-side rays share one colour; only the moat separates them
        for side in ("L", "R"):
            group = [d for d in names if d.endswith(side)]
            if len(group) >= 2:
                for d in group:
                    palette[d] = palette[group[0]]
                break
    grid = np.full((height, width), background, dtype=int)
    grid[scaffold] = scaffold_color
    for r, c in template:
        grid[r, c] = template_color
    for d in names:
        i, j = DIRECTIONS[d]
        for a, b in _fragment(rng, stencil):
            grid[r0 + i * (h + 1) + a, c0 + j * (w + 1) + b] = palette[d]
    return grid, (background, scaffold_color, template_color)


# ----------------------------------------------------------------------------
# Generation: pairs and puzzles
# ----------------------------------------------------------------------------

def _pair_ok(feats, info, rich=False):
    if "a_only" not in feats or "b_only" not in feats:
        return False
    falling = info["falling"]
    if len(falling) < (3 if rich else 2) or info["landing_rows"] < (3 if rich else 2):
        return False
    if info["distinct_falls"] < 2:
        return False
    return sum(f > 0 for f in falling) >= 0.6 * len(falling)


def _test_ok(feats):
    return "moving_stack" in feats or "ledge" in feats


def _make_pair(rng, ctx, focus, rot, large=False, avoid_dirsets=(), attempts=60, rich=False,
               dirs_fixed=None):
    """Generate a pair (in orientation `rot`) whose features include focus."""
    best = None
    for _ in range(attempts):
        made = _canonical_pair(rng, ctx, focus, large=large, dirs_fixed=dirs_fixed)
        if made is None:
            continue
        grid, roles = made
        final = np.rot90(grid, -rot)
        try:
            scene = _parse(final)
        except ValueError:
            continue
        if (scene["rot"], scene["background"], scene["scaffold_color"],
                scene["template_color"]) != (rot,) + roles:
            continue
        feats, info = _features(scene)
        if not _pair_ok(feats, info, rich):
            continue
        dirset = frozenset(NAMES[d] for d in scene["rays"])
        out = np.rot90(_solve_scene(scene)[0], -rot)
        record = {"input": np.ascontiguousarray(final).astype(int),
                  "output": np.ascontiguousarray(out).astype(int),
                  "features": feats, "info": info, "dirs": dirset, "rot": rot,
                  "area": int(final.size), "tarea": int(scene["size"][0] * scene["size"][1])}
        score = len(focus & feats) - (0.5 if dirset in avoid_dirsets else 0)
        if focus <= feats and dirset not in avoid_dirsets:
            return record
        if best is None or score > best[0]:
            best = (score, record)
    return best[1] if best else None


def _resolve_counts(num_train, num_test):
    if num_test is None:
        num_test = 1
    if num_train is None:
        num_train = max(4, num_test + 1) if isinstance(num_test, (int, np.integer)) else 4
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _puzzle_context(rng, simplify=False):
    u = rng.random()
    regime = "minimal" if (u < 1 / 14 or simplify) else ("maximal" if u < 1 / 14 + 1 / 7 else "normal")
    styles = list(SIMPLE_STYLES if regime == "minimal" else TERRAIN_STYLES)
    rng.shuffle(styles)
    if regime == "minimal":
        keep = int(rng.integers(1, 3))
    elif regime == "maximal":
        keep = int(rng.integers(4, len(styles) + 1))
    else:
        keep = int(rng.integers(2, len(styles) + 1))
    styles = styles[:keep]
    ctx = {"regime": regime, "rot": int(rng.integers(4))}
    if regime == "minimal":
        ctx.update(mixed=False, ledges=False, ledge_under=False, down=False, top_clip=False,
                   rigid=False, rigid8=False, stack3=False)
        ctx["background"] = int(rng.integers(10))
        ctx["scaffold"] = int(rng.choice([c for c in range(10) if c != ctx["background"]]))
    else:
        maximal = regime == "maximal"
        ctx["mixed"] = bool(rng.random() < (0.4 if maximal else 0.18))
        ctx["ledges"] = bool(maximal or rng.random() < 0.5)
        ctx["ledge_under"] = bool(rng.random() < (0.8 if maximal else 0.6))
        ctx["down"] = bool(maximal or rng.random() < 0.4)
        ctx["top_clip"] = bool(maximal or rng.random() < 0.45)
        ctx["rigid"] = bool(maximal or rng.random() < 0.55)
        ctx["rigid8"] = bool(ctx["top_clip"] and rng.random() < (0.8 if maximal else 0.45))
        ctx["stack3"] = bool(ctx["down"] and rng.random() < (0.8 if maximal else 0.5))
        ctx["background"] = int(rng.integers(10)) if rng.random() < 0.6 else None
        ctx["scaffold"] = None
        if ctx["background"] is not None and rng.random() < 0.5:
            ctx["scaffold"] = int(rng.choice([c for c in range(10) if c != ctx["background"]]))
    if not ctx["ledges"]:
        styles = [s for s in styles if s != "cantilever"] or ["steps"]
    ctx["styles"] = tuple(sorted(styles))
    return ctx


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _resolve_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for attempt in range(40):
        ctx = _puzzle_context(rng, simplify=attempt >= 30)
        puzzle = _assemble(rng, ctx, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError("could not generate a puzzle")


def _required(rng, ctx, num_train):
    required = set(CORE_REQUIRED)
    optional = []
    if ctx["ledges"]:
        optional.append("ledge")
        if ctx["ledge_under"]:
            optional.append("ledge_under")
    if ctx["down"]:
        optional.append("down")
        if ctx["stack3"]:
            optional.append("stack3")
    if ctx["top_clip"]:
        optional.append("top_clip_fall")
    if ctx["rigid"]:
        optional.append("rigid")
    if ctx["rigid8"]:
        optional.append("rigid8")
    limit = {2: 2, 3: 3}.get(num_train, len(optional))
    if len(optional) > limit:
        idx = sorted(int(i) for i in rng.choice(len(optional), size=limit, replace=False))
        optional = [optional[i] for i in idx]
        if "ledge_under" in optional and "ledge" not in optional:
            optional.append("ledge")
    return required | set(optional)


def _new_structure(cand, train):
    same = [r for r in train if r["dirs"] == cand["dirs"]]
    if not same:
        return False
    struct = {f for r in same for f in r["features"] if f in STRUCTURAL}
    return (bool({f for f in cand["features"] if f in STRUCTURAL} - struct)
            or cand["tarea"] > max(r["tarea"] for r in same)
            or cand["area"] > max(r["area"] for r in same))


def _assemble(rng, ctx, num_train, num_test):
    required = _required(rng, ctx, num_train)
    if ctx["mixed"]:
        rots = [int(rng.integers(4)) for _ in range(num_train)]
        if len(set(rots)) == 1:
            k = int(rng.integers(num_train))
            rots[k] = (rots[k] + int(rng.integers(1, 4))) % 4
    else:
        rots = [ctx["rot"]] * num_train
    train, covered, seen = [], set(), set()
    for slot in range(num_train):
        remaining = num_train - slot
        missing = sorted(required - covered)
        missing = [missing[int(i)] for i in rng.permutation(len(missing))]
        need = -(-len(missing) // remaining) if missing else 0
        focus = set(missing[:need])
        record = _make_pair(rng, ctx, focus, rots[slot], large=rng.random() < 0.3)
        if record is None:
            return None
        key = record["input"].tobytes() + bytes(record["input"].shape)
        if key in seen:
            return None
        seen.add(key)
        train.append(record)
        covered |= record["features"]
    if not required <= covered:
        return None
    train_dirsets = {r["dirs"] for r in train}
    train_rots = {r["rot"] for r in train}
    gated_ok = {f for f in GATED if f in covered}
    copy_counts = sorted(r["info"]["copies"] for r in train)
    typical = copy_counts[len(copy_counts) // 2]
    max_area = max(r["area"] for r in train)
    max_copies = max(r["info"]["copies"] for r in train)
    interesting = {"moving_stack", "ledge_under", "clip_fall", "shape", "moat8", "stack3",
                   "top_clip_fall", "rigid", "rigid8", "ledge_catch_h", "u_fall"}
    tests = []
    for _ in range(num_test):
        u = rng.random()
        regime = "novel" if u < 0.55 else ("familiar" if u < 0.85 else "same_scale")
        best = None
        for attempt in range(14):
            if attempt == 9 and best is None:
                regime = "novel"
            focus = set()
            roll = rng.random()
            if roll < 0.25 and "ledge_under" in gated_ok:
                focus.add("ledge_under")
            elif roll < 0.35 and "ledge" in gated_ok:
                focus.add("ledge")
            elif roll < 0.5 and "stack3" in gated_ok:
                focus.add("stack3")
            elif roll < 0.6 and "top_clip_fall" in gated_ok:
                focus.add("top_clip_fall")
            elif roll < 0.8:
                focus.add("moving_stack")
            if ctx["mixed"]:
                rot = sorted(train_rots)[int(rng.integers(len(train_rots)))] \
                    if rng.random() < 0.7 else int(rng.integers(4))
            else:
                rot = ctx["rot"]
            dirs_fixed = None
            if regime == "familiar":
                options = sorted(sorted(d) for d in train_dirsets)
                dirs_fixed = options[int(rng.integers(len(options)))]
                focus = {f for f in focus if not f.startswith("dir:")}
            cand = _make_pair(rng, ctx, focus, rot,
                              large=regime != "same_scale" and rng.random() < 0.6,
                              avoid_dirsets=train_dirsets if regime == "novel" else (),
                              attempts=20, rich=regime == "novel", dirs_fixed=dirs_fixed)
            if cand is None:
                continue
            gated = {f for f in cand["features"] if f in GATED}
            if not gated <= gated_ok or not _test_ok(cand["features"]):
                continue
            if cand["rot"] not in train_rots and len(train_rots) < 2:
                continue
            key = cand["input"].tobytes() + bytes(cand["input"].shape)
            if key in seen:
                continue
            bonus = 0.5 * len(cand["features"] & interesting)
            if regime == "novel":
                score = (3 * (cand["dirs"] not in train_dirsets)
                         + (cand["info"]["copies"] >= typical) + bonus)
            elif regime == "familiar":
                score = 3 * (cand["dirs"] in train_dirsets and _new_structure(cand, train)) + bonus
            else:
                score = (3 * (cand["area"] <= max_area and cand["info"]["copies"] <= max_copies)
                         + 1.5 * (cand["dirs"] not in train_dirsets) + bonus)
            if best is None or score > best[0]:
                best = (score, attempt, cand)
            if score >= 5:
                break
        if best is None:
            return None
        record = best[2]
        seen.add(record["input"].tobytes() + bytes(record["input"].shape))
        tests.append(record)
    order = rng.permutation(num_train)
    train = [train[int(i)] for i in order]
    strip = lambda r: {"input": r["input"].tolist(), "output": r["output"].tolist()}
    return {"train": [strip(r) for r in train], "test": [strip(r) for r in tests]}


# ----------------------------------------------------------------------------
# Validation and self test
# ----------------------------------------------------------------------------

def _analyse_pair(pair):
    grid = np.asarray(pair["input"], dtype=int)
    out = np.asarray(pair["output"], dtype=int)
    scene = _parse(grid)
    feats, info = _features(scene)
    return scene, feats, info, out


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "train/test keys"
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list), "train/test lists"
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "counts"
    seen = set()
    covered, train_rots = set(), set()
    analysed = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys"
            grid = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for g in (grid, out):
                assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30, "size"
                assert np.issubdtype(g.dtype, np.integer) and g.min() >= 0 and g.max() <= 9, "colours"
            assert grid.shape == out.shape, "output keeps the input size"
            key = (grid.shape, grid.astype(int).tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            scene, feats, info, out = _analyse_pair(pair)
            assert np.array_equal(solve(grid), out), "solve disagrees with stored output"
            # Roles: background, scaffold, template and ray colours are distinct.
            ray_colors = set(scene["rays"].values())
            roles = {scene["background"], scene["scaffold_color"], scene["template_color"]}
            assert len(roles) == 3 and not (ray_colors & roles), "role colours must be distinct"
            assert all(d in NAMES and d != (1, 0) for d in scene["rays"]), "legal directions"
            n_stencil = int(scene["stencil"].sum())
            for d, cells in scene["fragments"].items():
                assert 1 <= len(cells) < n_stencil, "fragments must be partial"
            copies = _emit(scene)
            blocked = _dilate(_obstacles(scene))
            for copy in copies:
                assert not any(blocked[r, c] for r, c in copy["cells"]), "copy starts inside the moat"
            (r0, c0), (h, w) = scene["origin"], scene["size"]
            height, width = scene["grid"].shape
            assert r0 >= 1 and c0 >= 1 and r0 + h < height and c0 + w < width, "template floats"
            assert "a_only" in feats and "b_only" in feats, "both parent concepts must matter"
            assert info["landing_rows"] >= 2 and len(info["falling"]) >= 2, "terrain must shape landings"
            assert info["distinct_falls"] >= 2, "copies must fall by different amounts"
            assert sum(f > 0 for f in info["falling"]) >= 0.6 * len(info["falling"]), "most copies fall"
            if split == "train":
                covered |= feats
                train_rots.add(scene["rot"])
            else:
                assert _test_ok(feats), "a test must show a stack or a shelf interaction"
            analysed[split].append((scene, feats, info))
    for name in CORE_REQUIRED:
        assert name in covered, f"training does not demonstrate {name}"
    for scene, feats, info in analysed["test"]:
        gated = {f for f in feats if f in GATED}
        assert gated <= covered, f"test uses undemonstrated features {sorted(gated - covered)}"
        assert scene["rot"] in train_rots or len(train_rots) >= 2, "test gravity orientation is untaught"


def _pits(g):
    """Count one- or two-column pits at least two cells deeper than both sides."""
    count, c = 0, 1
    while c < len(g) - 1:
        if g[c] + 2 <= min(g[c - 1], g[c + 1]):
            count, c = count + 1, c + 2
        elif c + 2 < len(g) and g[c] == g[c + 1] and g[c] + 2 <= min(g[c - 1], g[c + 2]):
            count, c = count + 1, c + 3
        else:
            c += 1
    return count


def _ramp_levels(g):
    """Longest monotone staircase of unit steps whose treads are one or two wide."""
    runs = []
    for v in g:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    best = cur = 1
    prev = 0
    for i in range(1, len(runs)):
        d = runs[i][0] - runs[i - 1][0]
        ok = abs(d) == 1 and runs[i][1] <= 2 and runs[i - 1][1] <= 2
        cur = cur + 1 if ok and (cur == 1 or d == prev) else (2 if ok else 1)
        prev = d
        best = max(best, cur)
    return best


def _measure(puzzle):
    """Returned-grid measurements used by self_test."""
    m = {"rots_train": set(), "dirsets": set(), "feats_train": set(), "feats_test": set(),
         "wide_side": 0, "beyond": 0, "attached": 0, "wells": 0, "comb": 0, "ramp": 0,
         "masks": set(), "sizes": set(), "shapes": set(), "bg": set(), "rays": set(),
         "new_test_dirset": 0, "familiar_test": 0}
    train_sets = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            scene, feats, info, _ = _analyse_pair(pair)
            dirset = frozenset(NAMES[d] for d in scene["rays"])
            m["dirsets"].add(dirset)
            m["rays"].add(len(dirset))
            m["masks"].add(scene["stencil"].tobytes() + bytes(scene["stencil"].shape))
            m["sizes"].add(scene["size"])
            m["shapes"].add(np.asarray(pair["input"]).shape)
            m["bg"].add(scene["background"])
            if split == "train":
                m["rots_train"].add(scene["rot"])
                m["feats_train"] |= feats
                train_sets.append(dirset)
            else:
                m["feats_test"] |= feats
                if dirset in train_sets:
                    m["familiar_test"] += 1
                else:
                    m["new_test_dirset"] += 1
            (r0, c0), (h, w) = scene["origin"], scene["size"]
            width = scene["grid"].shape[1]
            for name, extent in (("L", c0), ("R", width - c0 - w)):
                used = dirset & {name, "U" + name, "D" + name}
                if not used and extent > w + 1:
                    m["wide_side"] += 1
                if used == {"U" + name}:
                    copies = [c for c in _emit(scene) if c["dir"] == DIRECTIONS["U" + name]]
                    if extent // (w + 1) > max(c["index"] for c in copies):
                        m["beyond"] += 1
            scaffold = scene["scaffold"]
            over = _overhang(scaffold)
            if (over & _dilate(scaffold & ~over)).any():
                m["attached"] += 1
            g = [int(v) for v in (scaffold & ~over).sum(axis=0)]
            m["wells"] += _pits(g) >= 2
            m["comb"] += sum(1 for c in range(1, len(g) - 1) if g[c] >= max(g[c - 1], g[c + 1]) + 2) >= 3
            m["ramp"] += _ramp_levels(g) >= 4
    return m


def self_test() -> None:
    signatures = set()
    totals = {"rot": set(), "dirsets": set(), "masks": set(), "sizes": set(), "shapes": set(),
              "bg": set(), "rays": set()}
    counts = {k: 0 for k in ("mixed", "down", "top_clip_fall", "stack3", "rigid_test", "ledge",
                             "ledge_under", "ledge_catch_h", "wide_side", "beyond", "attached",
                             "wells", "comb", "ramp", "U", "clip_fall", "new_test_dirset",
                             "familiar_test", "rigid_train")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert puzzle == generate_puzzle(seed), "nondeterministic generation"
        signatures.add(repr(puzzle))
        m = _measure(puzzle)
        totals["rot"] |= m["rots_train"]
        for key in ("dirsets", "masks", "sizes", "shapes", "bg", "rays"):
            totals[key] |= m[key]
        flags = m["feats_train"] | m["feats_test"]
        counts["mixed"] += len(m["rots_train"]) >= 2
        counts["down"] += "down" in flags
        counts["top_clip_fall"] += "top_clip_fall" in flags
        counts["stack3"] += "stack3" in flags
        counts["rigid_test"] += "rigid" in m["feats_test"]
        counts["rigid_train"] += "rigid" in m["feats_train"]
        counts["ledge"] += "ledge" in flags
        counts["ledge_under"] += "ledge_under" in flags
        counts["ledge_catch_h"] += "ledge_catch_h" in flags
        counts["U"] += "dir:U" in flags
        counts["clip_fall"] += "clip_fall" in flags
        for key in ("wide_side", "beyond", "attached", "wells", "comb", "ramp",
                    "new_test_dirset", "familiar_test"):
            counts[key] += m[key] > 0
        assert "moving_stack" in m["feats_train"], "training shows a falling copy resting on a copy"
        assert not ("rigid" in m["feats_test"] and "rigid" not in m["feats_train"]), "rigid gate"
    assert len(signatures) == 42, "seeds must give distinct puzzles"
    assert len(totals["rot"]) == 4, "all gravity directions occur"
    assert len(totals["dirsets"]) >= 20, "direction sets vary"
    assert len(totals["masks"]) >= 80, "template stencils vary"
    assert len(totals["sizes"]) >= 14, "template sizes vary"
    assert len(totals["shapes"]) >= 80, "grid shapes vary"
    assert len(totals["bg"]) >= 8 and totals["rays"] >= {1, 2, 3, 4, 5}, "palette and ray counts vary"
    assert counts["mixed"] >= 4, "mixed-gravity puzzles occur"
    assert counts["down"] >= 10, "down-diagonal rays occur"
    assert counts["top_clip_fall"] >= 6, "top-clipped copies fall"
    assert counts["stack3"] >= 4, "three-high stacks occur"
    assert counts["rigid_train"] >= 10, "diagonally joined copies fall rigidly in training"
    assert counts["ledge"] >= 12 and counts["ledge_under"] >= 5, "shelves capture copies"
    assert counts["attached"] >= 3, "wall-attached arms occur"
    assert counts["wide_side"] >= 8 and counts["beyond"] >= 2, "wide empty sides occur"
    assert counts["wells"] >= 2 and counts["comb"] >= 2 and counts["ramp"] >= 2, "terrain styles vary"
    assert counts["clip_fall"] >= 10 and counts["U"] >= 6, "clipping and vertical rays occur"
    assert counts["new_test_dirset"] >= 15 and counts["familiar_test"] >= 5, "test regimes vary"
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {a}/{b}")
