"""Noisy-panel halo: a composition of ARC 3194b014 (A) and 3a301edc (B).

A uniform canvas holds one to three rectangular panels filled with speckled
multi-color noise.  Every panel contains several compact monochrome patches.
Inside each panel the patch with the most cells (A: noise/object separation
and largest-area selection) is the panel's hidden "core".  Its color paints a
new rectangular halo around the whole panel (B: the panel plays the role of
the enclosing frame).  The halo's top/bottom thickness equals the core's row
extent and its left/right thickness equals the core's column extent (B: the
core is the ruler); on each axis both sides shrink together to the nearer
canvas margin (B: symmetric boundary clipping).  Panels, patches, noise and
the background outside the halos are preserved.
"""
from __future__ import annotations

import numpy as np

_SIDE = 30
_MIN_OBJECT_AREA = 4
_MIN_DENSITY = 0.4
_MAX_OBJECT_DIM = 8
_MIN_NOISE_SHARE = 0.4
_SPREAD = 0.6


# --------------------------------------------------------------------------
# Generic grid helpers
# --------------------------------------------------------------------------

def _components(mask) -> list[list[tuple[int, int]]]:
    """4-connected components of a boolean mask as lists of (row, col)."""
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    rows = mask.tolist()
    seen = [[False] * width for _ in range(height)]
    result = []
    for r in range(height):
        row = rows[r]
        for c in range(width):
            if not row[c] or seen[r][c]:
                continue
            seen[r][c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < height and 0 <= nx < width and rows[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            result.append(cells)
    return result


def _color_stats(sub: np.ndarray, color: int) -> dict:
    mask = sub == color
    rr, cc = np.nonzero(mask)
    r0, r1, c0, c1 = int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())
    comps = _components(mask)
    return {"area": int(mask.sum()), "comps": len(comps),
            "largest_comp": max(len(x) for x in comps),
            "r0": r0, "r1": r1, "c0": c0, "c1": c1,
            "h": r1 - r0 + 1, "w": c1 - c0 + 1,
            "box": (r1 - r0 + 1) * (c1 - c0 + 1)}


def _background(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0], grid[-1], grid[:, 0], grid[:, -1]])
    values, counts = np.unique(border, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _analyze(grid, strict: bool = False) -> dict:
    """Read panels, their noise/object colors, winning patch and halo."""
    g = np.asarray(grid).astype(int)
    height, width = g.shape
    bg = _background(g)
    if strict:
        border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
        assert np.all(border == bg), "canvas border must be uniform background"
    panels = []
    for cells in _components(g != bg):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if strict:
            assert len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1), "panel must be a full rectangle"
        sub = g[r0:r1 + 1, c0:c1 + 1]
        stats = {int(c): _color_stats(sub, int(c)) for c in np.unique(sub) if int(c) != bg}
        objects = {c: s for c, s in stats.items() if s["comps"] == 1}
        noise = {c: s for c, s in stats.items() if s["comps"] >= 2}
        if not objects:
            continue
        best = max(s["area"] for s in objects.values())
        leaders = sorted(c for c, s in objects.items() if s["area"] == best)
        winner = leaders[0]
        win = objects[winner]
        top, bottom = r0, height - 1 - r1
        left, right = c0, width - 1 - c1
        vt = min(win["h"], top, bottom)
        ht = min(win["w"], left, right)
        panels.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1,
                       "ph": r1 - r0 + 1, "pw": c1 - c0 + 1,
                       "stats": stats, "objects": objects, "noise": noise,
                       "winner": winner, "tied": len(leaders) > 1,
                       "h": win["h"], "w": win["w"],
                       "top": top, "bottom": bottom, "left": left, "right": right,
                       "vt": vt, "ht": ht})
    return {"bg": bg, "height": height, "width": width, "panels": panels}


def _paint(grid: np.ndarray, panels: list[dict], color_key="winner",
           thickness=None) -> np.ndarray:
    """Paint each panel's halo, then restore the panels themselves."""
    out = grid.copy()
    height, width = grid.shape
    for p in panels:
        vt, ht = (p["vt"], p["ht"]) if thickness is None else thickness(p, height, width)
        if isinstance(vt, tuple):
            (up, down), (lf, rt) = vt, ht
        else:
            up = down = vt
            lf = rt = ht
        out[max(0, p["r0"] - up):p["r1"] + down + 1,
            max(0, p["c0"] - lf):p["c1"] + rt + 1] = p[color_key]
    for p in panels:
        out[p["r0"]:p["r1"] + 1, p["c0"]:p["c1"] + 1] = grid[p["r0"]:p["r1"] + 1, p["c0"]:p["c1"] + 1]
    return out


def solve(input_grid) -> np.ndarray:
    """Derive the output from the input alone."""
    grid = np.array(input_grid, dtype=int)
    info = _analyze(grid)
    return _paint(grid, info["panels"])


# --------------------------------------------------------------------------
# Shapes (compact monochrome patches)
# --------------------------------------------------------------------------

def _shape_ok(shape: np.ndarray) -> bool:
    if shape.sum() < _MIN_OBJECT_AREA:
        return False
    if not (shape.any(axis=1).all() and shape.any(axis=0).all()):
        return False
    if len(_components(shape)) != 1:
        return False
    padded = np.pad(~shape, 1, constant_values=True)
    if len(_components(padded)) != 1:  # no enclosed holes
        return False
    return shape.sum() / shape.size >= _MIN_DENSITY + 0.05 - 1e-9


def _carve(rng, shape: np.ndarray, target: int) -> np.ndarray | None:
    """Remove cells (keeping extents, connectivity, no holes) down to target."""
    shape = shape.copy()
    if target > int(shape.sum()):
        return None
    while int(shape.sum()) > target:
        cells = np.argwhere(shape)
        for i in rng.permutation(len(cells)):
            r, c = (int(x) for x in cells[i])
            shape[r, c] = False
            if (shape.any(axis=1).all() and shape.any(axis=0).all()
                    and len(_components(shape)) == 1
                    and len(_components(np.pad(~shape, 1, constant_values=True))) == 1):
                break
            shape[r, c] = True
        else:
            return None
    return shape


_STYLES = ("solid", "nibble", "stepped", "tapered", "lobed", "tail", "notch")


def _styled(rng, h: int, w: int, style: str | None = None) -> np.ndarray:
    if style is None:
        style = _STYLES[int(rng.choice(len(_STYLES), p=(.26, .16, .12, .12, .12, .11, .11)))]
    s = np.ones((h, w), dtype=bool)
    if style == "nibble" and h * w >= 4:
        k = int(rng.integers(1, max(2, h * w // 4) + 1))
        carved = _carve(rng, s, h * w - k)
        if carved is not None:
            s = carved
    elif style == "stepped" and h >= 2 and w >= 3:
        cut = int(rng.integers(1, max(2, w // 2)))
        pivot = int(rng.integers(1, h))
        s[:pivot, w - cut:] = False
        s[pivot:, :cut] = False
    elif style == "tapered" and h >= 3 and w >= 3:
        pivot = int(rng.integers(0, h))
        step = int(rng.integers(1, 3))
        for r in range(h):
            span = max(2, w - abs(r - pivot) * step)
            left = (w - span) // 2
            s[r, :left] = False
            s[r, left + span:] = False
    elif style == "lobed" and h >= 3 and w >= 3:
        a = int(rng.integers(1, max(2, h // 2)))
        b = int(rng.integers(1, max(2, w // 2)))
        s[:a, :b] = False
        a = int(rng.integers(1, max(2, h // 2)))
        b = int(rng.integers(1, max(2, w // 2)))
        s[h - a:, w - b:] = False
    elif style == "tail" and h >= 3 and w >= 2:
        tail = int(rng.integers(1, max(2, h // 2) + 1))
        span = int(rng.integers(1, min(2, w - 1) + 1))
        left = int(rng.integers(0, w - span + 1))
        s[h - tail:, :left] = False
        s[h - tail:, left + span:] = False
    elif style == "notch" and h >= 3 and w >= 3:
        depth = int(rng.integers(1, max(2, h // 2)))
        span = int(rng.integers(1, w - 1))
        left = int(rng.integers(1, w - span))
        s[:depth, left:left + span] = False
    if rng.random() < 0.5:
        s = s[::-1, :]
    if rng.random() < 0.5:
        s = s[:, ::-1]
    s = np.ascontiguousarray(s)
    if not _shape_ok(s):
        s = np.ones((h, w), dtype=bool)
    return s


def _shape_with_area(rng, h: int, w: int, target: int) -> np.ndarray | None:
    if target < _MIN_OBJECT_AREA or target > h * w or target < (_MIN_DENSITY + 0.05) * h * w:
        return None
    for _ in range(4):
        base = _styled(rng, h, w)
        if int(base.sum()) < target:
            base = np.ones((h, w), dtype=bool)
        shape = _carve(rng, base, target)
        if shape is not None and _shape_ok(shape):
            return shape
    return None


def _object_set(rng, k: int, wh: int, ww: int, max_dim: int, mode: str):
    """Winner first, then losers with strictly fewer cells."""
    dense = mode in ("inverse", "equal_box")
    winner = _styled(rng, wh, ww, "solid" if dense and rng.random() < 0.6 else
                     ("nibble" if dense else None))
    wa = int(winner.sum())
    if wa <= _MIN_OBJECT_AREA:
        return None
    shapes = [winner]
    if mode in ("inverse", "equal_box", "close"):
        loser = None
        for _ in range(40):
            if mode == "inverse":
                lh = wh + int(rng.integers(0, 3))
                lw = ww + int(rng.integers(0, 3))
                if lh * lw <= wh * ww:
                    continue
                target = wa - int(rng.integers(1, 7))
            elif mode == "equal_box":
                lh, lw = wh, ww
                target = wa - int(rng.integers(1, 5))
            else:
                lh = int(rng.integers(1, max_dim + 1))
                lw = int(rng.integers(1, max_dim + 1))
                target = wa - int(rng.integers(1, 4))
            if lh > max_dim or lw > max_dim:
                continue
            loser = _shape_with_area(rng, lh, lw, target)
            if loser is not None:
                break
        if loser is None:
            return None
        shapes.append(loser)
    margin = {"easy": 4, "natural": 2}.get(mode, 1)
    while len(shapes) < k:
        for _ in range(40):
            top = max_dim if mode != "natural" else max(2, min(max_dim, max(wh, ww)))
            lh = int(rng.integers(1, top + 1))
            lw = int(rng.integers(1, top + 1))
            if lh == 1 and lw == 1:
                continue
            shape = _styled(rng, lh, lw)
            if _MIN_OBJECT_AREA <= int(shape.sum()) <= wa - margin:
                shapes.append(shape)
                break
        else:
            return None
    return shapes


def _place(rng, shapes, ph: int, pw: int):
    placed = []
    for shape in shapes:
        sh, sw = shape.shape
        if sh > ph or sw > pw:
            return None
        for _ in range(150):
            r = int(rng.integers(0, ph - sh + 1))
            c = int(rng.integers(0, pw - sw + 1))
            if all(r + sh + 1 <= pr or pr + qh + 1 <= r or c + sw + 1 <= pc or pc + qw + 1 <= c
                   for pr, pc, qh, qw in placed):
                placed.append((r, c, sh, sw))
                break
        else:
            return None
    return [(r, c) for r, c, _, _ in placed]


def _panel_ok(panel: np.ndarray, noise, objects) -> bool:
    ph, pw = panel.shape
    noise_cells = int(np.isin(panel, list(noise)).sum())
    if noise_cells < _MIN_NOISE_SHARE * panel.size:
        return False
    for color in noise:
        if not (panel == color).any():
            return False
        s = _color_stats(panel, color)
        if s["comps"] < 3 or s["h"] < _SPREAD * ph or s["w"] < _SPREAD * pw:
            return False
    for color in objects:
        if _color_stats(panel, color)["comps"] != 1:
            return False
    return True


def _make_panel(rng, ph, pw, shapes, colors, noise):
    positions = _place(rng, shapes, ph, pw)
    if positions is None:
        return None
    weights = rng.uniform(0.6, 1.6, size=len(noise))
    if rng.random() < 0.3:
        weights[int(rng.integers(0, len(noise)))] *= 2.0
    weights = weights / weights.sum()
    for _ in range(25):
        panel = rng.choice(np.array(noise), size=(ph, pw), p=weights).astype(int)
        for shape, color, (r, c) in zip(shapes, colors, positions):
            panel[r:r + shape.shape[0], c:c + shape.shape[1]][shape] = color
        if _panel_ok(panel, noise, colors):
            return panel
    return None


# --------------------------------------------------------------------------
# Canvas layout
# --------------------------------------------------------------------------

def _axis(rng, p: int, ruler: int, mode: str, witness: bool, min_clip: int = 1):
    """Margins (before, after) for one axis of a single-panel canvas."""
    if mode == "full":
        need = p + 2 * ruler + (2 if witness else 0)
        if need > _SIDE:
            return None
        budget = _SIDE - need
        if not witness and rng.random() < 0.2:
            extra = (0, int(rng.integers(0, min(budget, 4) + 1)))
        else:
            cap = budget if rng.random() < 0.35 else min(budget, 4)
            extra = (int(rng.integers(0, cap + 1)), int(rng.integers(0, cap + 1)))
            while sum(extra) > budget:
                extra = (extra[0] // 2, extra[1])
        base = ruler + (1 if witness else 0)
        pair = (base + extra[0], base + extra[1])
    elif mode == "clip":
        if ruler <= min_clip:
            return None
        m = int(rng.integers(min_clip, ruler))
        low = m + 1 if witness else m
        high = min(_SIDE - p - m, low + 8)
        if low > high:
            return None
        pair = (m, int(rng.integers(low, high + 1)))
    else:  # free: any margins, may or may not clip
        if p + 2 > _SIDE:
            return None
        cap = min(_SIDE - p - 1, ruler + 4)
        a = int(rng.integers(1, cap + 1))
        b = int(rng.integers(1, min(cap, _SIDE - p - a) + 1))
        pair = (a, b)
    if rng.random() < 0.5:
        pair = (pair[1], pair[0])
    return pair


def _footprint(r, c, ph, pw, h, w, height, width):
    vt = min(h, r, height - r - ph)
    ht = min(w, c, width - c - pw)
    return (r - vt, r + ph - 1 + vt, c - ht, c + pw - 1 + ht)


def _separated(a, b) -> bool:
    return a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]


def _multi_layout(rng, items):
    """items: list of (ph, pw, h, w). Returns (height, width, positions)."""
    for _ in range(120):
        height = int(rng.integers(14, _SIDE + 1))
        width = int(rng.integers(14, _SIDE + 1))
        if rng.random() < 0.5:
            height = max(height, int(rng.integers(22, _SIDE + 1)))
            width = max(width, int(rng.integers(22, _SIDE + 1)))
        positions, prints = [], []
        for ph, pw, h, w in items:
            if ph + 2 > height or pw + 2 > width:
                break
            for _ in range(60):
                r = int(rng.integers(1, height - ph))
                c = int(rng.integers(1, width - pw))
                fp = _footprint(r, c, ph, pw, h, w, height, width)
                if all(_separated(fp, other) for other in prints):
                    positions.append((r, c))
                    prints.append(fp)
                    break
            else:
                break
        if len(positions) == len(items):
            return height, width, positions
    return None


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------

def _palette(rng, spec, counts):
    bg = spec["bg"]
    noise = spec.get("noise")
    if noise is None or bg in noise or len(noise) != spec["n_noise"]:
        pool = [c for c in range(10) if c != bg]
        noise = tuple(int(x) for x in rng.choice(pool, size=spec["n_noise"], replace=False))
    rest = [c for c in range(10) if c != bg and c not in noise]
    need = sum(counts)
    if need > len(rest):
        return None
    wanted = spec.get("winner_color")
    chosen = []
    if wanted is not None and wanted in rest:
        chosen.append(int(wanted))
        rest = [c for c in rest if c != wanted]
    chosen += [int(x) for x in rng.choice(rest, size=need - len(chosen), replace=False)]
    groups, i = [], 0
    for k in counts:
        groups.append(chosen[i:i + k])
        i += k
    return bg, tuple(noise), groups


def _object_count(rng, lo, hi, ph, pw):
    hi = min(hi, 2 + (ph * pw) // 30)
    return int(rng.integers(lo, max(lo, hi) + 1))


def _build_single(rng, spec):
    lo, hi = spec["wdim"]
    vmode, hmode = spec["vmode"], spec["hmode"]
    vwit, hwit = spec["vwit"], spec["hwit"]
    for _ in range(80):
        h = int(rng.integers(lo, hi + 1))
        w = int(rng.integers(lo, hi + 1))
        if spec.get("big_ruler") and max(h, w) < 6:
            continue
        if spec["nonsquare"] and h == w:
            continue
        if (vmode == "clip" and h < 2) or (hmode == "clip" and w < 2):
            continue
        if (spec.get("thick_axis") == "v" and h < 3) or (spec.get("thick_axis") == "h" and w < 3):
            continue
        sizes = []
        for p_mode, wit, ruler in ((vmode, vwit, h), (hmode, hwit, w)):
            plo, phi = spec["panel"]
            if p_mode == "full":
                phi = min(phi, _SIDE - 2 * ruler - (2 if wit else 0))
            elif p_mode == "clip":
                phi = min(phi, _SIDE - 2 - (1 if wit else 0))
            else:
                phi = min(phi, _SIDE - 2)
            plo = max(plo, ruler + 1, 5)
            if plo > phi:
                break
            sizes.append(int(rng.integers(plo, phi + 1)))
        if len(sizes) != 2:
            continue
        ph, pw = sizes
        k = _object_count(rng, spec["k"][0], spec["k"][1], ph, pw)
        max_dim = min(_MAX_OBJECT_DIM - 1, ph, pw, max(h, w) + 2)
        shapes = _object_set(rng, k, h, w, max_dim, spec["mode"])
        if shapes is None or sum(int(s.sum()) for s in shapes) > (1 - _MIN_NOISE_SHARE) * ph * pw:
            continue
        palette = _palette(rng, spec, [len(shapes)])
        if palette is None:
            continue
        bg, noise, groups = palette
        panel = _make_panel(rng, ph, pw, shapes, groups[0], noise)
        if panel is None:
            continue
        thick_v = spec.get("thick_axis") == "v"
        vm = _axis(rng, ph, h, vmode, vwit, 2 if thick_v else 1)
        hm = _axis(rng, pw, w, hmode, hwit, 2 if spec.get("thick_axis") == "h" else 1)
        if vm is None or hm is None:
            continue
        grid = np.full((ph + vm[0] + vm[1], pw + hm[0] + hm[1]), bg, dtype=int)
        grid[vm[0]:vm[0] + ph, hm[0]:hm[0] + pw] = panel
        return grid, [groups[0][0]]
    return None


def _build_multi(rng, spec):
    n = spec["panels"]
    lo, hi = spec["wdim"]
    for _ in range(60):
        items, shape_sets = [], []
        for _ in range(n):
            h = int(rng.integers(lo, min(hi, 4) + 1))
            w = int(rng.integers(lo, min(hi, 4) + 1))
            ph = int(rng.integers(max(5, h + 1), 10 + 1))
            pw = int(rng.integers(max(5, w + 1), 10 + 1))
            k = _object_count(rng, 2, 3, ph, pw)
            mode = spec["mode"] if not items else _random_mode(rng)
            shapes = _object_set(rng, k, h, w, min(ph, pw, max(h, w) + 2, 6), mode)
            if shapes is None or sum(int(s.sum()) for s in shapes) > (1 - _MIN_NOISE_SHARE) * ph * pw:
                break
            items.append((ph, pw, h, w))
            shape_sets.append(shapes)
        if len(items) != n:
            continue
        palette = _palette(rng, spec, [len(s) for s in shape_sets])
        if palette is None:
            continue
        bg, noise, groups = palette
        panels = [_make_panel(rng, ph, pw, shapes, colors, noise)
                  for (ph, pw, _, _), shapes, colors in zip(items, shape_sets, groups)]
        if any(p is None for p in panels):
            continue
        layout = _multi_layout(rng, items)
        if layout is None:
            continue
        height, width, positions = layout
        grid = np.full((height, width), bg, dtype=int)
        for panel, (r, c) in zip(panels, positions):
            grid[r:r + panel.shape[0], c:c + panel.shape[1]] = panel
        return grid, [g[0] for g in groups]
    return None


def _build_pair(rng, spec):
    built = _build_multi(rng, spec) if spec["panels"] > 1 else _build_single(rng, spec)
    if built is None:
        return None
    grid, winners = built
    try:
        info = _check_pair(grid)
    except AssertionError:
        return None
    if sorted(p["winner"] for p in info["panels"]) != sorted(winners):
        return None
    if len(info["panels"]) != spec["panels"]:
        return None
    return grid, solve(grid)


# --------------------------------------------------------------------------
# Puzzle plan
# --------------------------------------------------------------------------

def _random_mode(rng):
    return str(rng.choice(("natural", "easy", "close", "inverse", "equal_box"),
                          p=(.3, .25, .15, .18, .12)))


def _spec(rng, role, ctx, trap=False):
    spec = {"role": role, "panels": 1, "vmode": "full", "hmode": "full",
            "vwit": False, "hwit": False, "nonsquare": False,
            "wdim": (2, ctx["cap"]), "panel": (5, 14), "k": (2, 4),
            "n_noise": int(rng.choice((2, 3), p=(.6, .4))),
            "mode": "inverse" if trap else _random_mode(rng)}
    axes_modes = ("full", "full", "free", "clip")
    if role == "axis":
        spec.update(nonsquare=True, vwit=True, hwit=True)
    elif role == "clip":
        clip_v = bool(rng.random() < 0.5)
        other = "full" if not ctx["allow_double"] else str(rng.choice(("full", "free")))
        spec.update(vmode="clip" if clip_v else other, hmode=other if clip_v else "clip",
                    vwit=clip_v, hwit=not clip_v)
    elif role in ("trap", "mixed", "matching"):
        v = str(rng.choice(axes_modes))
        hm = str(rng.choice(axes_modes))
        if not ctx["allow_double"] and v != "full" and hm != "full":
            hm = "full"
        spec.update(vmode=v, hmode=hm)
        if role != "trap" and ctx["multi"] and rng.random() < 0.4:
            spec["panels"] = ctx["train_panels"]
    elif role == "multi":
        spec.update(panels=ctx["train_panels"], n_noise=2)
    elif role == "double_clip":
        spec.update(vmode="clip", hmode="clip", vwit=bool(rng.random() < 0.5),
                    thick_axis="v" if rng.random() < 0.5 else "h")
        if rng.random() < 0.6:
            spec["mode"] = "inverse"
        spec["hwit"] = not spec["vwit"] or bool(rng.random() < 0.5)
    elif role == "big_ruler":
        spec.update(wdim=(2, 7), big_ruler=True, panel=(7, 16))
        big_v = bool(rng.random() < 0.5)
        spec.update(vmode="full" if big_v else str(rng.choice(axes_modes)),
                    hmode=str(rng.choice(axes_modes)) if big_v else "full")
    elif role == "big_field":
        spec.update(panel=(15, 24), wdim=(3, 6), k=(3, 5), vmode="free", hmode="free",
                    n_noise=int(rng.choice((2, 3, 4), p=(.35, .4, .25))))
    elif role == "role_swap":
        spec.update(mode="inverse", vmode=str(rng.choice(axes_modes)),
                    hmode=str(rng.choice(axes_modes)))
        if ctx["multi"] and rng.random() < 0.4:
            spec["panels"] = ctx["train_panels"]
    elif role == "more_panels":
        spec.update(panels=3, n_noise=2)
    if spec["panels"] > 1:
        spec.update(wdim=(2, min(4, ctx["cap"])), n_noise=2)
    if role == "big_field":
        spec["panel"] = (15, 24)
    elif role == "big_ruler":
        spec["panel"] = (7, 16)
    else:
        spec["panel"] = (5, ctx["panel_cap"])
    return spec


def _plan(rng, num_train, num_test):
    multi = num_train >= 3 and rng.random() < 0.45
    shifts = ["double_clip", "big_ruler", "big_field", "role_swap", "matching"]
    weights = [.22, .18, .2, .2, .2]
    if multi:
        shifts.append("more_panels")
        weights.append(.3)
    weights = np.array(weights) / sum(weights)
    shift = shifts[int(rng.choice(len(shifts), p=weights))]
    ctx = {"multi": multi, "shift": shift,
           "cap": 4 if shift == "big_ruler" else 7,
           "allow_double": shift != "double_clip",
           "panel_cap": 13 if shift == "big_field" else 16,
           "train_panels": 2 if shift == "more_panels" else int(rng.choice((2, 3)))}
    roles = ["axis", "clip"] + (["multi"] if multi else [])
    traps = [False] * len(roles)
    if len(roles) < num_train:
        roles.append("trap")
        traps.append(True)
    else:
        traps[int(rng.integers(0, 2))] = True
    while len(roles) < num_train:
        roles.append("mixed")
        traps.append(False)
    order = rng.permutation(num_train)
    train = [_spec(rng, roles[i], ctx, traps[i]) for i in order]
    for spec in train:
        spec["no_double"] = not ctx["allow_double"]
    extra = ["matching", "double_clip", "big_field", "role_swap"] + (["more_panels"] if multi else [])
    tests = [shift] + [extra[int(rng.integers(0, len(extra)))] for _ in range(num_test - 1)]
    test = [_spec(rng, role, ctx) for role in tests]
    return ctx, train, test


def _check_counts(num_train, num_test):
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


def _signature(info):
    return tuple(sorted((p["ph"], p["pw"], p["h"], p["w"], p["vt"], p["ht"]) for p in info["panels"]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    ctx, train_specs, test_specs = _plan(rng, num_train, num_test)
    fixed_bg = rng.random() < 0.4
    bg0 = 0 if rng.random() < 0.5 else int(rng.integers(0, 10))
    fixed_noise = None
    if rng.random() < 0.35:
        pool = [c for c in range(10) if c != bg0]
        fixed_noise = tuple(int(x) for x in rng.choice(pool, size=2, replace=False))
    puzzle = {"train": [], "test": []}
    seen, train_sigs, train_noise, train_winners = set(), set(), set(), set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for base in specs:
            for attempt in range(160):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if split == "test" and attempt >= 40 and attempt % 2:
                    # Rare dead ends: keep one rule, relax only the test regime.
                    spec = _spec(child, "matching", ctx)
                else:
                    spec = dict(base)
                spec["bg"] = bg0 if fixed_bg else int(child.integers(0, 10))
                swap = split == "test" and spec["role"] == "role_swap"
                if fixed_noise is not None and spec["n_noise"] == 2 and attempt < 30 and not swap:
                    spec["noise"] = fixed_noise
                if swap:
                    options = sorted(c for c in train_noise if c != spec["bg"])
                    if options:
                        spec["winner_color"] = int(options[int(child.integers(0, len(options)))])
                built = _build_pair(child, spec)
                if built is None:
                    continue
                inp, out = built
                key = (inp.shape, inp.tobytes())
                info = _analyze(inp)
                sig = _signature(info)
                if key in seen or (split == "test" and sig in train_sigs):
                    continue
                winners = {q["winner"] for q in info["panels"]}
                if split == "train" and len(train_winners) == 1 and winners <= train_winners:
                    continue
                if not _spec_satisfied(spec, info, inp, out, split):
                    continue
                break
            else:
                raise RuntimeError(f"could not build {split} pair {base['role']} for seed {seed}")
            seen.add(key)
            if split == "train":
                train_sigs.add(sig)
                for p in info["panels"]:
                    train_noise.update(p["noise"])
                    train_winners.add(p["winner"])
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _spec_satisfied(spec, info, inp, out, split) -> bool:
    panels = info["panels"]
    role = spec["role"]
    broken = _broken_shortcuts(inp, out)
    if split == "test" and not _test_informative(broken):
        return False
    ok = True
    if spec.get("no_double") and any(p["vt"] < p["h"] and p["ht"] < p["w"] for p in panels):
        return False
    if role == "axis":
        ok = (_axis_witness(panels) and _not_edge_witness(panels)
              and {"uniform_min", "uniform_max", "transposed", "nearest_edge"} <= broken)
    if role == "clip":
        ok = ok and _clip_witness(panels) and "independent_clip" in broken
    if role == "double_clip":
        p = panels[0]
        ok = ok and p["vt"] < p["h"] and p["ht"] < p["w"] and _clip_witness(panels)
    if spec["mode"] == "inverse":
        ok = ok and _trap_witness(panels) and "largest_box" in broken
    if role == "role_swap" and spec.get("winner_color") is not None:
        ok = ok and any(p["winner"] == spec["winner_color"] for p in panels)
    return ok


# --------------------------------------------------------------------------
# Witness predicates and validation
# --------------------------------------------------------------------------

def _axis_witness(panels):
    return any(p["h"] != p["w"] and p["vt"] == p["h"] and p["ht"] == p["w"] for p in panels)


def _not_edge_witness(panels):
    return any((p["vt"] == p["h"] and min(p["top"], p["bottom"]) > p["h"]) or
               (p["ht"] == p["w"] and min(p["left"], p["right"]) > p["w"]) for p in panels)


def _clip_witness(panels):
    return any((p["vt"] < p["h"] and p["top"] != p["bottom"]) or
               (p["ht"] < p["w"] and p["left"] != p["right"]) for p in panels)


def _trap_witness(panels):
    return any(any(s["box"] > p["objects"][p["winner"]]["box"]
                   for c, s in p["objects"].items() if c != p["winner"]) for p in panels)


def _check_pair(grid) -> dict:
    g = np.asarray(grid).astype(int)
    info = _analyze(g, strict=True)
    panels = info["panels"]
    assert 1 <= len(panels) <= 3, "one to three panels"
    assert len(_components(g != info["bg"])) == len(panels), "every component is a panel"
    object_colors, noise_colors = [], set()
    for p in panels:
        assert not p["tied"], "unique largest patch"
        assert len(p["noise"]) >= 2 and len(p["objects"]) >= 2
        for c, s in p["noise"].items():
            assert s["comps"] >= 3 and s["h"] >= _SPREAD * p["ph"] and s["w"] >= _SPREAD * p["pw"], \
                "noise colors are dispersed across the panel"
        for c, s in p["objects"].items():
            assert s["area"] >= _MIN_OBJECT_AREA and s["area"] >= _MIN_DENSITY * s["box"]
            assert max(s["h"], s["w"]) <= _MAX_OBJECT_DIM
        noise_cells = sum(s["area"] for s in p["noise"].values())
        assert noise_cells >= _MIN_NOISE_SHARE * p["ph"] * p["pw"]
        win = p["objects"][p["winner"]]
        assert win["h"] >= 2 and win["w"] >= 2 and win["area"] >= 4
        assert p["vt"] >= 1 and p["ht"] >= 1
        object_colors.extend(p["objects"])
        noise_colors.update(p["noise"])
    assert len(object_colors) == len(set(object_colors)), "object colors are unique in a grid"
    assert not noise_colors & set(object_colors), "color roles are consistent in a grid"
    prints = [(p["r0"] - p["vt"], p["r1"] + p["vt"], p["c0"] - p["ht"], p["c1"] + p["ht"]) for p in panels]
    for i in range(len(prints)):
        for j in range(i + 1, len(prints)):
            assert _separated(prints[i], prints[j]), "halo footprints never touch"
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for grid in (inp, out):
                assert grid.ndim == 2 and 1 <= grid.shape[0] <= _SIDE and 1 <= grid.shape[1] <= _SIDE
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = (inp.shape, inp.astype(np.int64).tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _check_pair(inp)
            assert np.array_equal(out, solve(inp)), "output follows the rule"
            assert not np.array_equal(out, inp)
            infos[split].append(info)
    train_panels = [p for info in infos["train"] for p in info["panels"]]
    assert _axis_witness(train_panels), "train shows unclipped axis-wise ruler"
    assert _not_edge_witness(train_panels), "train shows a halo stopping short of the edge"
    assert _clip_witness(train_panels), "train shows symmetric clipping with unequal margins"
    assert _trap_witness(train_panels), "train shows a larger box losing on cell count"
    assert len({p["winner"] for p in train_panels}) >= 2
    survived = set(_REQUIRED_SHORTCUTS)
    for pair in train:
        survived -= _broken_shortcuts(np.asarray(pair["input"]), pair["output"])
    assert not survived, f"training does not refute shortcuts {sorted(survived)}"
    for pair in test:
        assert _test_informative(_broken_shortcuts(np.asarray(pair["input"]), pair["output"])), \
            "test pair must exercise ruler, clipping or selection"
    if any(len(info["panels"]) > 1 for info in infos["test"]):
        assert any(len(info["panels"]) > 1 for info in infos["train"]), "multi-panel taught first"


# --------------------------------------------------------------------------
# Shortcut predictions (used by self_test and documentation)
# --------------------------------------------------------------------------

def _shortcut_outputs(grid) -> dict:
    g = np.asarray(grid).astype(int)
    info = _analyze(g)
    panels = info["panels"]
    height, width = g.shape
    result = {}

    def sym(vh):
        def f(p, H, W):
            v, h = vh(p)
            return min(v, p["top"], p["bottom"]), min(h, p["left"], p["right"])
        return f

    result["uniform_min"] = _paint(g, panels, thickness=sym(lambda p: (min(p["h"], p["w"]),) * 2))
    result["uniform_max"] = _paint(g, panels, thickness=sym(lambda p: (max(p["h"], p["w"]),) * 2))
    result["transposed"] = _paint(g, panels, thickness=sym(lambda p: (p["w"], p["h"])))
    result["outline_one"] = _paint(g, panels, thickness=lambda p, H, W: (1, 1))
    result["nearest_edge"] = _paint(g, panels, thickness=lambda p, H, W: (
        min(p["top"], p["bottom"]), min(p["left"], p["right"])))
    result["independent_clip"] = _paint(g, panels, thickness=lambda p, H, W: (
        (min(p["h"], p["top"]), min(p["h"], p["bottom"])),
        (min(p["w"], p["left"]), min(p["w"], p["right"]))))
    # Selection shortcuts: change which patch is the core.
    alt_bbox, alt_freq, alt_comp = [], [], []
    for p in panels:
        objs = p["objects"]
        best = max(objs, key=lambda c: (objs[c]["box"], objs[c]["area"], -c))
        q = dict(p, winner=best, h=objs[best]["h"], w=objs[best]["w"])
        q["vt"], q["ht"] = min(q["h"], p["top"], p["bottom"]), min(q["w"], p["left"], p["right"])
        alt_bbox.append(q)
        freq = max(p["stats"], key=lambda c: (p["stats"][c]["area"], -c))
        alt_freq.append(dict(p, winner=freq))
        comp = max(p["stats"], key=lambda c: (p["stats"][c]["largest_comp"], -c))
        alt_comp.append(dict(p, winner=comp))
    result["largest_box"] = _paint(g, alt_bbox)
    result["most_frequent_color"] = _paint(g, alt_freq)
    result["largest_component_any_color"] = _paint(g, alt_comp)
    return result


_REQUIRED_SHORTCUTS = ("uniform_min", "uniform_max", "transposed", "outline_one", "nearest_edge",
                       "independent_clip", "largest_box", "most_frequent_color")
_GEOMETRY_SHORTCUTS = ("uniform_min", "uniform_max", "transposed", "nearest_edge",
                       "independent_clip", "largest_box")


def _broken_shortcuts(inp, out) -> set:
    out = np.asarray(out)
    return {name for name, pred in _shortcut_outputs(inp).items() if not np.array_equal(pred, out)}


def _test_informative(broken: set) -> bool:
    """A test must not be answerable by a 1-cell outline and must defeat at
    least two geometric/selection shortcuts besides the color-frequency one."""
    return ("outline_one" in broken and "most_frequent_color" in broken
            and len(broken & set(_GEOMETRY_SHORTCUTS)) >= 2)


def _puzzles_equal(a, b) -> bool:
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _hand_example():
    panel = np.array([[int(ch) for ch in row] for row in (
        "5855858",
        "8222535",
        "5222838",
        "8585333",
        "5858585",
        "8558858")])
    grid = np.zeros((14, 17), dtype=int)
    grid[5:11, 3:10] = panel
    expected = grid.copy()
    expected[3:13, 0:13] = 2        # vertical 2 (core rows), horizontal 3 (core cols, left limit 3)
    expected[5:11, 3:10] = panel
    clipped = np.full((12, 15), 4, dtype=int)
    clipped[1:7, 2:9] = panel       # top margin 1 -> both vertical sides shrink to 1
    clipped_expected = clipped.copy()
    clipped_expected[0:8, 0:11] = 2  # horizontal: min(3, 2, 6) = 2
    clipped_expected[1:7, 2:9] = panel
    return [(grid, expected), (clipped, clipped_expected)]


def self_test() -> None:
    # Hand-built semantics, independent of the sampler: color 3 has the
    # larger bounding box (3x3) but fewer cells (5 < 6), so color 2 rules.
    for grid, expected in _hand_example():
        assert np.array_equal(solve(grid), expected)
    # Count handling.
    for bad in ((1, 1), (11, 1), (4, 0), (4, 5), (6, 5), (2, 2), (3, 4), (4, 4), (True, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")
    for nt, ns in ((2, 1), (10, 4), (5, 3), (3, 2)):
        p = generate_puzzle(7 + nt * 10 + ns, nt, ns)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    assert _puzzles_equal(generate_puzzle(123, 5, 2), generate_puzzle(123, 5, 2))
    assert not _puzzles_equal(generate_puzzle(124, 5, 2), generate_puzzle(125, 5, 2))
    # 42-seed variation and shortcut study.
    shapes, palettes, winners_dims, panel_counts, bgs = set(), set(), set(), set(), set()
    metrics = dict(double_clip_test=0, exact_fit=0, multi_train=0, multi_test=0,
                   test_shortcut_breaks=0, tests=0, comp_shortcut_fails=0, zero_object=0,
                   zero_noise=0, zero_bg=0, inverse_test=0, styles=set())
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        broken = {name: False for name in _REQUIRED_SHORTCUTS}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _analyze(inp)
                shapes.add(inp.shape)
                palettes.add(tuple(sorted(np.unique(inp).tolist())))
                panel_counts.add(len(info["panels"]))
                bgs.add(info["bg"])
                metrics["zero_bg"] += info["bg"] == 0
                for p in info["panels"]:
                    winners_dims.add((p["h"], p["w"]))
                    metrics["zero_object"] += 0 in p["objects"]
                    metrics["zero_noise"] += 0 in p["noise"]
                    if p["r0"] - p["vt"] == 0 and p["r1"] + p["vt"] == info["height"] - 1:
                        metrics["exact_fit"] += 1
                if len(info["panels"]) > 1:
                    metrics["multi_" + split] += 1
                shortcuts = _shortcut_outputs(inp)
                wrong = [n for n in _REQUIRED_SHORTCUTS if not np.array_equal(shortcuts[n], out)]
                if split == "train":
                    for n in wrong:
                        broken[n] = True
                else:
                    metrics["tests"] += 1
                    metrics["test_shortcut_breaks"] += len(wrong) >= 3
                    metrics["comp_shortcut_fails"] += not np.array_equal(
                        shortcuts["largest_component_any_color"], out)
                    metrics["double_clip_test"] += any(p["vt"] < p["h"] and p["ht"] < p["w"]
                                                       for p in info["panels"])
                    metrics["inverse_test"] += _trap_witness(info["panels"])
        assert all(broken.values()), f"seed {seed}: a shortcut survives training {broken}"
    assert len(shapes) >= 60 and len(palettes) >= 80
    assert len(winners_dims) >= 25 and panel_counts == {1, 2, 3} and len(bgs) >= 8
    assert metrics["exact_fit"] >= 3 and metrics["multi_train"] >= 8 and metrics["multi_test"] >= 4
    assert metrics["double_clip_test"] >= 6 and metrics["inverse_test"] >= 6
    assert metrics["test_shortcut_breaks"] >= 0.8 * metrics["tests"]
    assert metrics["zero_object"] and metrics["zero_noise"] and metrics["zero_bg"]
