"""Symmetrizing mat: widen a picture's frame until the framed interior is mirror-symmetric.

Every input is a picture inside a solid one-cell (sometimes two-cell) frame.
The interior holds a left-right symmetric composition near the middle and
some asymmetric debris closer to the edges.  The output paints whole
concentric rings with the frame colour, from the outside in, and stops at
the first ring whose remaining interior equals its own left-right mirror.

Parent A (6f8cd79b) supplies the frame: a complete one-cell perimeter ring,
corners included, recoloured with the frame colour while the interior is
preserved.  Parent B (72ca375d) supplies the selection test: exact
left-right mirror symmetry of a bounded window, with an axis through a
column (odd width) or between two columns (even width).  The two interact:
the candidate windows are exactly the nested frame interiors, and the frame
only stops growing when B's symmetry test succeeds.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Core geometry and the solver
# ---------------------------------------------------------------------------


def _depth_map(height: int, width: int) -> np.ndarray:
    """Ring index of each cell: 0 on the outer perimeter, 1 just inside, ..."""
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    return np.minimum(np.minimum(rows, height - 1 - rows),
                      np.minimum(cols, width - 1 - cols))


def _mismatch(grid: np.ndarray) -> np.ndarray:
    return grid != grid[:, ::-1]


def _paint_rings(grid: np.ndarray, thickness: int) -> np.ndarray:
    out = grid.copy()
    out[_depth_map(*grid.shape) < thickness] = grid[0, 0]
    return out


def _target_thickness(grid: np.ndarray) -> int:
    """Smallest frame thickness whose interior is left-right symmetric."""
    mism = _mismatch(grid)
    if not mism.any():
        return 0
    return int(_depth_map(*grid.shape)[mism].max()) + 1


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.array(input_grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("input must be a non-empty 2D grid")
    thickness = _target_thickness(grid)
    if thickness == 0:
        return grid.copy()
    # The frame colour is the colour of the input's outer ring (its corner).
    return _paint_rings(grid, thickness)


# ---------------------------------------------------------------------------
# Mask helpers (B-style figures)
# ---------------------------------------------------------------------------


def _tight(mask: np.ndarray) -> bool:
    return bool(mask.any() and mask[0].any() and mask[-1].any()
                and mask[:, 0].any() and mask[:, -1].any())


def _connected(mask: np.ndarray) -> bool:
    cells = [tuple(int(v) for v in p) for p in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    height, width = mask.shape
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and 0 <= nc < width and mask[nr, nc]
                        and (nr, nc) not in seen):
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return len(seen) == len(cells)


def _lr_symmetric(mask: np.ndarray) -> bool:
    return bool(np.array_equal(mask, mask[:, ::-1]))


def _sym_mask(rng: np.random.Generator, height: int, width: int,
              allow_solid: bool = True) -> np.ndarray:
    half = (width + 1) // 2
    for _ in range(300):
        density = float(rng.uniform(0.45, 0.85))
        left = rng.random((height, half)) < density
        right = left[:, :-1][:, ::-1] if width % 2 else left[:, ::-1]
        mask = np.concatenate((left, right), axis=1)
        if not (_tight(mask) and _connected(mask)):
            continue
        if not allow_solid and mask.all() and height * width > 2:
            continue
        return mask
    return np.ones((height, width), dtype=bool)


def _asym_mask(rng: np.random.Generator, height: int, width: int) -> np.ndarray | None:
    if height < 2 or width < 2:
        return None
    for _ in range(300):
        count = int(rng.integers(2, height * width))
        mask = np.zeros((height, width), dtype=bool)
        mask.flat[rng.choice(height * width, size=count, replace=False)] = True
        if _tight(mask) and _connected(mask) and not _lr_symmetric(mask):
            return mask
    return None


def _dilate(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    padded = np.zeros((height + 2, width + 2), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            padded[dr:dr + height, dc:dc + width] |= mask
    return padded


# ---------------------------------------------------------------------------
# Canvas construction
# ---------------------------------------------------------------------------


class _Canvas:
    def __init__(self, height: int, width: int, background: int):
        self.h, self.w = height, width
        self.bg = background
        self.grid = np.full((height, width), background, dtype=int)
        self.blocked = np.zeros((height, width), dtype=bool)
        self.depth = _depth_map(height, width)

    def allowed(self, lo: int, hi: int, forbid: tuple[int, ...] = ()) -> np.ndarray:
        ok = (self.depth >= lo) & (self.depth <= hi) & ~self.blocked
        for ring in forbid:
            ok &= self.depth != ring
        return ok

    @staticmethod
    def candidates(allowed: np.ndarray, mask: np.ndarray,
                   need: np.ndarray | None = None) -> np.ndarray:
        height, width = allowed.shape
        mh, mw = mask.shape
        if mh > height or mw > width:
            return np.zeros((0, 2), dtype=int)
        rows, cols = height - mh + 1, width - mw + 1
        ok = np.ones((rows, cols), dtype=bool)
        touch = np.zeros((rows, cols), dtype=bool)
        for dr, dc in np.argwhere(mask):
            ok &= allowed[dr:dr + rows, dc:dc + cols]
            if need is not None:
                touch |= need[dr:dr + rows, dc:dc + cols]
        if need is not None:
            ok &= touch
        return np.argwhere(ok)

    def fits(self, allowed: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> bool:
        mh, mw = mask.shape
        if r0 < 0 or c0 < 0 or r0 + mh > self.h or c0 + mw > self.w:
            return False
        return bool(np.all(allowed[r0:r0 + mh, c0:c0 + mw][mask]))

    def put(self, mask: np.ndarray, r0: int, c0: int, colors) -> None:
        mh, mw = mask.shape
        view = self.grid[r0:r0 + mh, c0:c0 + mw]
        if np.ndim(colors) == 0:
            view[mask] = int(colors)
        else:
            view[mask] = np.asarray(colors)[mask]
        halo = _dilate(mask)
        top, left = r0 - 1, c0 - 1
        rs, cs = max(0, top), max(0, left)
        re, ce = min(self.h, top + mh + 2), min(self.w, left + mw + 2)
        self.blocked[rs:re, cs:ce] |= halo[rs - top:re - top, cs - left:ce - left]

    # -- placements ---------------------------------------------------------
    def place_axis(self, rng, mask, colors, lo, hi, forbid=()) -> bool:
        """Place a figure centred on the vertical mirror axis."""
        mh, mw = mask.shape
        if (self.w - mw) % 2:
            return False
        c0 = (self.w - mw) // 2
        allowed = self.allowed(lo, hi, forbid)
        rows = [r for r in range(self.h - mh + 1) if self.fits(allowed, mask, r, c0)]
        if not rows:
            return False
        r0 = rows[int(rng.integers(len(rows)))]
        self.put(mask, r0, c0, colors)
        return True

    def place_pair(self, rng, mask, left_colors, right_colors, lo, hi,
                   forbid=(), need=None, right_mask=None) -> bool:
        """Place a figure on the left half and a mirrored copy on the right."""
        mh, mw = mask.shape
        right_mask = mask[:, ::-1] if right_mask is None else right_mask
        allowed = self.allowed(lo, hi, forbid)
        cands = self.candidates(allowed, mask, need)
        cands = [tuple(int(v) for v in p) for p in cands if 2 * (p[1] + mw) <= self.w - 1]
        if not cands:
            return False
        order = rng.permutation(len(cands))
        for index in order[:60]:
            r0, c0 = cands[int(index)]
            c1 = self.w - c0 - mw
            if not self.fits(allowed, right_mask, r0, c1):
                continue
            rc = right_colors
            if np.ndim(rc) != 0:
                rc = np.asarray(rc)
            self.put(mask, r0, c0, left_colors)
            self.put(right_mask, r0, c1, rc)
            return True
        return False

    def place_free(self, rng, mask, colors, lo, hi, forbid=(), need=None,
                   off_axis=False) -> bool:
        mh, mw = mask.shape
        allowed = self.allowed(lo, hi, forbid)
        cands = self.candidates(allowed, mask, need)
        if off_axis:
            cands = [p for p in cands if 2 * int(p[1]) + mw - 1 != self.w - 1]
        if len(cands) == 0:
            return False
        r0, c0 = (int(v) for v in cands[int(rng.integers(len(cands)))])
        self.put(mask, r0, c0, colors)
        return True


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

_OFFENDER_KINDS = ("dot", "bar", "offaxis", "asym", "colorpair", "nearpair")
# Offenders that are individually left-right symmetric (or pairs of such
# figures): B's per-figure test would not flag them, yet they break the
# symmetry of the framed window because of where they sit.
_SELF_SYMMETRIC_KINDS = ("dot", "bar", "offaxis", "colorpair")
# "defect": one wrong cell inside a long symmetric band (B's near-mirror idea
# applied to a large object that the frame partially covers).
_DEEP_KINDS = _OFFENDER_KINDS + ("defect",)


def _offender_mask(rng, kind: str) -> np.ndarray | None:
    if kind == "dot":
        return np.ones((1, 1), dtype=bool)
    if kind == "bar":
        length = int(rng.integers(2, 5))
        return np.ones((1, length) if rng.random() < 0.5 else (length, 1), dtype=bool)
    if kind == "offaxis":
        height = int(rng.integers(1, 4))
        width = int(rng.integers(2, 4))
        return _sym_mask(rng, height, width)
    if kind == "asym":
        return _asym_mask(rng, int(rng.integers(2, 4)), int(rng.integers(2, 4)))
    if kind == "colorpair":
        choice = rng.random()
        if choice < 0.4:
            return np.ones((1, 1), dtype=bool)
        if choice < 0.7:
            return _sym_mask(rng, int(rng.integers(1, 3)), int(rng.integers(2, 4)))
        return _asym_mask(rng, 2, int(rng.integers(2, 4)))
    if kind == "nearpair":
        return _asym_mask(rng, int(rng.integers(2, 4)), int(rng.integers(2, 4)))
    raise ValueError(kind)


def _place_offender(canvas: _Canvas, rng, kind: str, palette, lo, hi,
                    forbid, need) -> bool:
    mask = _offender_mask(rng, kind)
    if mask is None:
        return False
    color = int(rng.choice(palette))
    if kind in ("dot", "bar", "offaxis", "asym"):
        return canvas.place_free(rng, mask, color, lo, hi, forbid, need, off_axis=True)
    if kind == "colorpair":
        other = int(rng.choice([c for c in palette if c != color]))
        if rng.random() < 0.5:
            color, other = other, color
        return canvas.place_pair(rng, mask, color, other, lo, hi, forbid, need)
    # nearpair: the mirrored copy loses one cell or gains one bounding-box cell
    right = mask[:, ::-1].copy()
    cells = np.argwhere(right)
    holes = np.argwhere(~right)
    if len(holes) and (len(cells) < 3 or rng.random() < 0.5):
        r, c = holes[int(rng.integers(len(holes)))]
        right[r, c] = True
    else:
        r, c = cells[int(rng.integers(len(cells)))]
        right[r, c] = False
        if not right.any():
            return False
    if rng.random() < 0.5:
        return canvas.place_pair(rng, mask, color, color, lo, hi, forbid, need,
                                 right_mask=right)
    # Put the defective copy on the left instead.
    return canvas.place_pair(rng, right[:, ::-1], color, color, lo, hi, forbid,
                             need, right_mask=mask[:, ::-1])


def _emblem_colors(rng, mask: np.ndarray, palette, two_tone: bool):
    first = int(rng.choice(palette))
    if not two_tone:
        return first, (first,)
    second = int(rng.choice([c for c in palette if c != first]))
    height, width = mask.shape
    half = (width + 1) // 2
    style = rng.random()
    if style < 0.35:
        left = np.zeros((height, half), dtype=bool)
        left[rng.random(height) < 0.5, :] = True
    elif style < 0.6 and half > 1:
        left = np.zeros((height, half), dtype=bool)
        left[:, half - int(rng.integers(1, half)):] = True
    else:
        left = rng.random((height, half)) < 0.35
    right = left[:, :-1][:, ::-1] if width % 2 else left[:, ::-1]
    choose = np.concatenate((left, right), axis=1)
    colors = np.where(choose, second, first)
    used = tuple(sorted({int(v) for v in colors[mask]}))
    return colors, used


def _choose_dims(rng, d: int, parity: int, need_h: int, need_w: int,
                 lo_h: int, hi_h: int, lo_w: int, hi_w: int):
    min_h = max(8, lo_h, 2 * (d + 1) + need_h)
    min_w = max(8, lo_w, 2 * (d + 1) + need_w)
    if min_h > hi_h or min_w > hi_w:
        return None
    height = int(rng.integers(min_h, hi_h + 1))
    widths = [w for w in range(min_w, hi_w + 1) if w % 2 == parity]
    if not widths:
        return None
    width = int(widths[int(rng.integers(len(widths)))])
    return height, width


def _build_pair(rng, spec: dict):
    """Return (input, output, meta) for one pair, or None on failure."""
    t, added = spec["t"], spec["added"]
    d = t + added - 1                     # innermost ring that must be painted
    margin = spec["margin"]
    satellite = spec["satellite"]
    need_h = 3 + 2 * margin + (2 if satellite else 0)
    need_w = 4 + 2 * margin + (2 if satellite else 0)
    dims = _choose_dims(rng, d, spec["parity"], need_h, need_w, *spec["dim_range"])
    if dims is None:
        return None
    height, width = dims
    bg, fc = spec["bg"], spec["fc"]
    palette = [c for c in range(10) if c not in (bg, fc)]
    canvas = _Canvas(height, width, bg)
    depth = canvas.depth
    gap = ()
    if spec["gap"] and added >= 2:
        gap = (int(rng.integers(t, d)),)
    big = 99
    meta = {"t": t, "added": added, "d": d, "shape": (height, width),
            "gap": bool(gap), "margin": margin, "satellite": False,
            "stripe": None, "zone_sym": 0, "deep_kind": None,
            "band": spec["band"], "extras": [], "two_tone": spec["two_tone"]}

    # 1. Optional symmetric band that runs from the frame through the core:
    #    a full-width row ("row") or a mirrored pair of full-height columns
    #    ("pillars").  Whole rings are painted, so its ends get covered.
    stripe = spec["stripe"]
    if spec["deep_kind"] == "defect" and not stripe:
        stripe = "pillars" if spec["band"] == "tb" else "row"
    crossing = np.zeros((height, width), dtype=bool)
    if stripe:
        first = int(rng.choice(palette))
        second = int(rng.choice([c for c in palette if c != first]))
        period = int(rng.integers(2, 4))
        patterned = bool(rng.random() < 0.5)
        if stripe == "row":
            rows = list(range(d + 1, height - 1 - d))
            if not rows:
                return None
            r = int(rows[int(rng.integers(len(rows)))])
            span = np.ones((1, width - 2 * t), dtype=bool)
            cols = np.arange(span.shape[1])
            dist = np.minimum(cols, span.shape[1] - 1 - cols)
            colors = np.where(patterned & (dist % period == 0), second, first).reshape(span.shape)
            canvas.put(span, r, t, colors)
            crossing[r, t:width - t] = True
        else:
            cols = list(range(d + 1, (width - 3) // 2 + 1))
            if not cols:
                return None
            c = int(cols[int(rng.integers(len(cols)))])
            span = np.ones((height - 2 * t, 1), dtype=bool)
            rows = np.arange(span.shape[0])
            dist = np.minimum(rows, span.shape[0] - 1 - rows)
            colors = np.where(patterned & (dist % period == 0), second, first).reshape(span.shape)
            canvas.put(span, t, c, colors)
            canvas.put(span, t, width - 1 - c, colors)
            crossing[t:height - t, c] = True
            crossing[t:height - t, width - 1 - c] = True
        meta["stripe"] = stripe

    # 2. The central emblem: a B-style left-right symmetric figure on the axis.
    emblem_lo = d + 2 if margin else d + 1
    avail_h = height - 2 * emblem_lo
    avail_w = width - 2 * emblem_lo
    share = 0.6 if satellite else 0.8
    max_eh = min(6, avail_h, max(2, int(avail_h * share)))
    ew_choices = [w for w in range(2, min(11, avail_w) + 1) if w % 2 == width % 2
                  and w <= max(3, int(avail_w * share))]
    if max_eh < 2 or not ew_choices:
        return None
    eh = int(rng.integers(2, max_eh + 1))
    ew = int(ew_choices[int(rng.integers(len(ew_choices)))])
    emblem = _sym_mask(rng, eh, ew, allow_solid=rng.random() < 0.15)
    colors, emblem_used = _emblem_colors(rng, emblem, palette, spec["two_tone"])
    if not canvas.place_axis(rng, emblem, colors, emblem_lo, big):
        return None

    # 3. Survivors: a mirrored pair of asymmetric figures inside the core.
    if satellite:
        sh, sw = int(rng.integers(1, 3)), int(rng.integers(2, 4))
        mask = _asym_mask(rng, max(2, sh), sw)
        color = int(rng.choice(palette))
        if mask is None or not canvas.place_pair(rng, mask, color, color, d + 1, big):
            return None
        meta["satellite"] = True
    if spec["axis_deco"]:
        mw = 1 if width % 2 else 2
        mw += 2 * int(rng.integers(0, 2))
        mask = _sym_mask(rng, int(rng.integers(1, 3)), mw)
        canvas.place_axis(rng, mask, int(rng.choice(palette)), d + 1, big)

    # 4. Symmetric material inside the zone that will be covered anyway.
    for _ in range(spec["zone_sym"]):
        if rng.random() < 0.6:
            mask = _offender_mask(rng, rng.choice(["dot", "bar", "asym"]))
            if mask is None:
                continue
            color = int(rng.choice(palette))
            if canvas.place_pair(rng, mask, color, color, t, d):
                meta["zone_sym"] += 1
        else:
            mw = 1 if width % 2 else 2
            mw += 2 * int(rng.integers(0, 2))
            mask = _sym_mask(rng, 1, mw)
            if canvas.place_axis(rng, mask, int(rng.choice(palette)), t, d):
                meta["zone_sym"] += 1

    # 5. The deepest offender: it decides how far the frame must grow.
    need = (depth == d)
    if spec["band"] == "tb":
        cols = np.arange(width)[None, :]
        need = need & (cols > d) & (cols < width - 1 - d)
    elif spec["band"] == "side":
        rows = np.arange(height)[:, None]
        need = need & (rows > d) & (rows < height - 1 - d)
    kinds = [spec["deep_kind"]] + [k for k in ("dot", "bar") if k != spec["deep_kind"]]
    placed = None
    for kind in kinds:
        if kind == "defect":
            # One wrong cell inside the symmetric band, exactly on ring d.
            spots = np.argwhere(crossing & need & (2 * np.arange(width)[None, :] != width - 1))
            if len(spots):
                r, c = (int(v) for v in spots[int(rng.integers(len(spots)))])
                old = int(canvas.grid[r, c])
                choices = [bg] + [col for col in palette if col != old]
                canvas.grid[r, c] = int(choices[int(rng.integers(len(choices)))])
                placed = kind
                break
            continue
        for _ in range(4):
            if _place_offender(canvas, rng, kind, palette, t, d, gap, need):
                placed = kind
                break
        if placed:
            break
    if placed is None:
        return None
    meta["deep_kind"] = placed

    # 6. Additional, shallower (or equally deep) debris.
    for _ in range(spec["n_extra"]):
        kind = str(rng.choice(_OFFENDER_KINDS))
        if _place_offender(canvas, rng, kind, palette, t, d, gap, None):
            meta["extras"].append(kind)

    # 7. The existing frame.
    grid = canvas.grid
    grid[depth < t] = fc
    if _target_thickness(grid) != d + 1:
        return None
    mism = _mismatch(grid)
    if gap and np.any(mism & (depth == gap[0])):
        return None
    output = solve(grid)
    core = depth > d
    if not np.any(output[core] != bg):
        return None
    meta["emblem_colors"] = emblem_used
    return grid, output, meta


# ---------------------------------------------------------------------------
# Puzzle construction
# ---------------------------------------------------------------------------


def _check_counts(num_train, num_test) -> tuple[int, int]:
    values = []
    for name, value, default, lo, hi in (("num_train", num_train, 4, 2, 10),
                                         ("num_test", num_test, 1, 1, 4)):
        if value is None:
            value = default
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in [{lo}, {hi}]")
        values.append(value)
    if values[0] <= values[1]:
        raise ValueError("num_train must exceed num_test")
    return values[0], values[1]


def _pair_spec(rng, base: dict, **overrides) -> dict:
    added = overrides.pop("added", None)
    if added is None:
        added = int(rng.choice([1, 2, 2, 3, 3, 4]))
    spec = {
        "t": base["t_choice"](),
        "added": added,
        "parity": int(rng.integers(2)),
        "margin": bool(rng.random() < 0.5),
        "satellite": bool(rng.random() < 0.45),
        "gap": bool(rng.random() < 0.4),
        "band": str(rng.choice(["any", "any", "tb", "side"])),
        "deep_kind": str(rng.choice(_DEEP_KINDS)),
        "n_extra": int(rng.choice([0, 1, 1, 2, 2, 3])),
        "zone_sym": int(rng.choice([0, 0, 1, 2])),
        "stripe": [None, "row", "pillars"][int(rng.choice(3, p=[0.8, 0.1, 0.1]))],
        "axis_deco": bool(rng.random() < 0.3),
        "two_tone": bool(rng.random() < 0.35),
        "dim_range": (8, 23, 8, 23),
    }
    spec.update(overrides)
    spec["bg"] = base["bg_choice"]()
    spec["fc"] = base["fc_choice"](spec["bg"])
    return spec


def _attempt(rng, num_train: int, num_test: int):
    thick = bool(rng.random() < 0.3)
    bg_mode = str(rng.choice(["zero", "fixed", "varied"], p=[0.45, 0.25, 0.30]))
    fixed_bg = 0 if bg_mode == "zero" else int(rng.integers(10))
    fc_fixed = bool(rng.random() < 0.5)
    fixed_fc = int(rng.choice([c for c in range(10) if c != fixed_bg]))

    def bg_choice():
        return int(rng.integers(10)) if bg_mode == "varied" else fixed_bg

    def fc_choice(bg):
        if fc_fixed and fixed_fc != bg:
            return fixed_fc
        return int(rng.choice([c for c in range(10) if c != bg]))

    thick_index = int(rng.integers(num_train)) if thick else -1

    def t_choice():
        return 2 if thick and rng.random() < 0.35 else 1

    base = {"bg_choice": bg_choice, "fc_choice": fc_choice, "t_choice": t_choice}

    # Training: two "duty" pairs guarantee the distinguishing evidence;
    # the remaining pairs are free draws.  Order is shuffled afterwards.
    first_added = int(rng.integers(2, 5))
    second_added = int(rng.choice([a for a in (1, 2, 3, 4) if a != first_added]))
    parity0 = int(rng.integers(2))
    specs = [
        _pair_spec(rng, base, added=first_added, parity=parity0, margin=True,
                   gap=True, deep_kind=str(rng.choice(_SELF_SYMMETRIC_KINDS))),
        _pair_spec(rng, base, added=second_added, parity=1 - parity0,
                   satellite=True, band="tb"),
    ]
    for _ in range(num_train - 2):
        specs.append(_pair_spec(rng, base))
    if thick:
        specs[thick_index]["t"] = 2

    train, metas, seen = [], [], set()
    for spec in specs:
        result = None
        for _ in range(40):
            result = _build_pair(rng, spec)
            if result is not None and result[0].tobytes() + bytes(result[0].shape) not in seen:
                break
            result = None
        if result is None:
            return None
        grid, output, meta = result
        seen.add(grid.tobytes() + bytes(grid.shape))
        train.append({"input": grid, "output": output})
        metas.append(meta)
    order = rng.permutation(num_train)
    train = [train[int(i)] for i in order]
    metas = [metas[int(i)] for i in order]

    max_added = max(m["added"] for m in metas)
    max_h = max(m["shape"][0] for m in metas)
    max_w = max(m["shape"][1] for m in metas)
    added_values = sorted({m["added"] for m in metas})
    test, test_metas = [], []
    for index in range(num_test):
        regime = str(rng.choice(["deeper", "larger", "familiar"], p=[0.4, 0.35, 0.25]))
        if regime == "deeper" and max_added >= 6:
            regime = "larger"
        if regime == "larger" and max_h >= 30 and max_w >= 30:
            regime = "familiar"
        overrides = {}
        if regime == "deeper":
            overrides["added"] = int(min(6, max_added + int(rng.integers(1, 3))))
            overrides["dim_range"] = (8, 30, 8, 30)
        elif regime == "larger":
            overrides["added"] = int(rng.choice(added_values))
            overrides["dim_range"] = (max_h, 30, max_w, 30)
        else:
            overrides["added"] = int(rng.choice(added_values))
            overrides["dim_range"] = (8, max_h, 8, max_w)
        spec = _pair_spec(rng, base, **overrides)
        if spec["deep_kind"] == "defect" and not any(m["stripe"] for m in metas):
            # A flaw hidden inside a long band is only fair once training has
            # shown a band being cut by the growing frame.
            spec["deep_kind"] = str(rng.choice(_OFFENDER_KINDS))
        result = None
        for _ in range(40):
            result = _build_pair(rng, spec)
            if result is None:
                continue
            grid = result[0]
            key = grid.tobytes() + bytes(grid.shape)
            if key in seen:
                result = None
                continue
            if regime == "larger" and grid.shape[0] * grid.shape[1] <= max_h * max_w:
                result = None
                continue
            break
        if result is None:
            return None
        grid, output, meta = result
        seen.add(grid.tobytes() + bytes(grid.shape))
        meta["regime"] = regime
        test.append({"input": grid, "output": output})
        test_metas.append(meta)
    return {"train": train, "test": test}, {"train": metas, "test": test_metas}


def _generate(seed: int, num_train: int = 4, num_test: int = 1):
    num_train, num_test = _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    for _ in range(60):
        result = _attempt(rng, num_train, num_test)
        if result is None:
            continue
        puzzle, meta = result
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle, meta
    raise RuntimeError("could not generate a valid puzzle")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    return _generate(seed, num_train, num_test)[0]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _frame_thickness(grid: np.ndarray) -> int:
    depth = _depth_map(*grid.shape)
    fc = grid[0, 0]
    t = 0
    while t <= int(depth.max()) and np.all(grid[depth == t] == fc):
        t += 1
    return t


def _background(grid: np.ndarray, t: int) -> int:
    inner = grid[_depth_map(*grid.shape) >= t]
    values, counts = np.unique(inner, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(grid: np.ndarray, t: int, bg: int):
    """8-connected single-colour components of interior non-background cells."""
    depth = _depth_map(*grid.shape)
    free = (depth >= t) & (grid != bg)
    seen = np.zeros(grid.shape, dtype=bool)
    comps = []
    height, width = grid.shape
    for r0, c0 in np.argwhere(free):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        color = grid[r0, c0]
        stack, cells = [(r0, c0)], []
        seen[r0, c0] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width and not seen[nr, nc]
                            and free[nr, nc] and grid[nr, nc] == color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        cells = np.array(cells)
        top, left = cells.min(axis=0)
        mask = np.zeros(tuple(cells.max(axis=0) - (top, left) + 1), dtype=bool)
        mask[cells[:, 0] - top, cells[:, 1] - left] = True
        comps.append({"cells": cells, "mask": mask,
                      "depths": depth[cells[:, 0], cells[:, 1]]})
    return comps


def _shortcut_predictions(grid: np.ndarray) -> dict:
    """Plausible simpler rules; each must be refuted by some training pair."""
    depth = _depth_map(*grid.shape)
    t = _frame_thickness(grid)
    bg = _background(grid, t)
    fc = grid[0, 0]
    preds = {}
    # Parent A alone: draw one more frame ring.
    preds["one_more_ring"] = _paint_rings(grid, t + 1)
    # Symmetry repair without rings: paint just the mismatched cells.
    cells = grid.copy()
    cells[_mismatch(grid)] = fc
    preds["mismatch_cells_only"] = cells
    # Rings, but only the asymmetric ones (not cumulative from the outside).
    rings = grid.copy()
    mism = _mismatch(grid)
    for ring in set(int(v) for v in depth[mism]):
        rings[depth == ring] = fc
    preds["asymmetric_rings_only"] = rings
    comps = _components(grid, t, bg)
    # Parent B alone: grow until every individually asymmetric figure is covered.
    asym = [c for c in comps if not _lr_symmetric(c["mask"])]
    k = max([t] + [int(c["depths"].max()) + 1 for c in asym])
    preds["cover_asymmetric_figures"] = _paint_rings(grid, k)
    if comps:
        largest = max(comps, key=lambda c: len(c["cells"]))
        others = [c for c in comps if c is not largest]
        k = max([t] + [int(c["depths"].max()) + 1 for c in others])
        preds["cover_all_but_largest"] = _paint_rings(grid, k)
        preds["grow_to_largest"] = _paint_rings(grid, max(t, int(largest["depths"].min())))
    # Other symmetry notions.
    tb = grid != grid[::-1, :]
    k = int(depth[tb].max()) + 1 if tb.any() else t
    preds["top_bottom_symmetry"] = _paint_rings(grid, max(t, k))
    both = tb | mism
    k = int(depth[both].max()) + 1 if both.any() else t
    preds["both_axes_symmetry"] = _paint_rings(grid, max(t, k))
    mask_only = (grid != bg) != (grid[:, ::-1] != bg)
    k = int(depth[mask_only].max()) + 1 if mask_only.any() else t
    preds["shape_only_symmetry"] = _paint_rings(grid, max(t, k))
    return preds


_REQUIRED_REFUTATIONS = (
    "one_more_ring", "mismatch_cells_only", "asymmetric_rings_only",
    "cover_asymmetric_figures", "cover_all_but_largest", "grow_to_largest",
    "top_bottom_symmetry", "both_axes_symmetry",
)


def _pair_facts(pair: dict) -> dict:
    grid = np.asarray(pair["input"])
    output = np.asarray(pair["output"])
    assert grid.ndim == 2 and output.shape == grid.shape
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer) and np.issubdtype(output.dtype, np.integer)
    assert grid.min() >= 0 and grid.max() <= 9 and output.min() >= 0 and output.max() <= 9
    height, width = grid.shape
    assert height >= 8 and width >= 8
    depth = _depth_map(height, width)
    fc = int(grid[0, 0])
    t = _frame_thickness(grid)
    assert 1 <= t <= 2, "input frame must be one or two solid rings"
    assert not np.any(grid[depth >= t] == fc), "frame colour reserved for the frame"
    bg = _background(grid, t)
    assert bg != fc
    inner = grid[depth >= t]
    assert np.count_nonzero(inner == bg) * 2 > inner.size, "background must dominate"
    thickness = _target_thickness(grid)
    assert thickness > t, "every pair must widen the frame"
    expected = _paint_rings(grid, thickness)
    assert np.array_equal(output, expected), "output is not the symmetrizing frame"
    core = depth >= thickness
    assert height - 2 * thickness >= 3 and width - 2 * thickness >= 3
    assert np.any(grid[core] != bg), "the symmetric core must show content"
    assert np.array_equal(output[core], grid[core])
    assert np.all(output[~core] == fc), "every added ring is solid frame colour"
    window = output[thickness:height - thickness, thickness:width - thickness]
    assert np.array_equal(window, window[:, ::-1])
    return {"t": t, "thickness": thickness, "added": thickness - t, "bg": bg,
            "fc": fc, "shape": (height, width)}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid = np.asarray(pair["input"])
            key = (grid.shape, grid.astype(np.int64).tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            facts[split].append(_pair_facts(pair))
            assert np.array_equal(solve(grid), np.asarray(pair["output"]))
    tr = facts["train"]
    assert len({f["added"] for f in tr}) >= 2, "training must vary the growth"
    assert max(f["added"] for f in tr) >= 2, "training must show multi-ring growth"
    assert len({f["shape"][1] % 2 for f in tr}) == 2, "both mirror-axis parities"
    if any(f["t"] == 2 for f in facts["test"]):
        assert any(f["t"] == 2 for f in tr), "thick starting frames must be demonstrated"
    refuted = set()
    for pair in train:
        grid = np.asarray(pair["input"])
        output = np.asarray(pair["output"])
        for name, pred in _shortcut_predictions(grid).items():
            if not np.array_equal(pred, output):
                refuted.add(name)
    missing = [name for name in _REQUIRED_REFUTATIONS if name not in refuted]
    assert not missing, f"training does not refute shortcuts: {missing}"
    # No ambiguity: any listed alternative that survives all demonstrations
    # must also agree with the intended answer on every test input.
    for pair in test:
        grid = np.asarray(pair["input"])
        output = np.asarray(pair["output"])
        for name, pred in _shortcut_predictions(grid).items():
            if name not in refuted:
                assert np.array_equal(pred, output), f"test separates unrefuted {name}"


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def _fingerprint(puzzle: dict) -> tuple:
    return tuple((split, np.asarray(p["input"]).shape, np.asarray(p["input"]).tobytes(),
                  np.asarray(p["output"]).tobytes())
                 for split in ("train", "test") for p in puzzle[split])


def self_test() -> None:
    results = [_generate(seed) for seed in range(42)]
    puzzles = [p for p, _ in results]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    shapes, added, thick, parities = set(), set(), set(), set()
    frame_colors, backgrounds, kinds, regimes = set(), set(), set(), set()
    stats = {"satellite": 0, "stripe": 0, "gap": 0, "margin": 0, "zone_sym": 0,
             "two_tone": 0, "tb_band": 0, "deeper_than_train": 0,
             "larger_than_train": 0, "within_puzzle_fc_change": 0,
             "within_puzzle_bg_change": 0, "self_symmetric_deepest": 0}
    for puzzle, meta in results:
        validate_generated_puzzle(puzzle)
        train_added = [m["added"] for m in meta["train"]]
        train_area = max(m["shape"][0] * m["shape"][1] for m in meta["train"])
        fcs = {int(np.asarray(p["input"])[0, 0]) for s in ("train", "test") for p in puzzle[s]}
        bgs = {_background(np.asarray(p["input"]), m["t"])
               for s in ("train", "test") for p, m in zip(puzzle[s], meta[s])}
        stats["within_puzzle_fc_change"] += len(fcs) > 1
        stats["within_puzzle_bg_change"] += len(bgs) > 1
        frame_colors |= fcs
        backgrounds |= bgs
        for split in ("train", "test"):
            for pair, m in zip(puzzle[split], meta[split]):
                grid = np.asarray(pair["input"])
                shapes.add(grid.shape)
                added.add(m["added"])
                thick.add(m["t"])
                parities.add(grid.shape[1] % 2)
                kinds.add(m["deep_kind"])
                stats["satellite"] += m["satellite"]
                stats["stripe"] += m["stripe"] is not None
                stats["gap"] += m["gap"]
                stats["margin"] += m["margin"]
                stats["zone_sym"] += m["zone_sym"] > 0
                stats["two_tone"] += m["two_tone"]
                stats["tb_band"] += m["band"] == "tb"
                stats["self_symmetric_deepest"] += m["deep_kind"] in _SELF_SYMMETRIC_KINDS
                # Independent restatement of the rule: grow ring by ring.
                k = m["t"]
                while True:
                    window = grid[k:grid.shape[0] - k, k:grid.shape[1] - k]
                    if np.array_equal(window, window[:, ::-1]):
                        break
                    k += 1
                assert k == m["d"] + 1
                expected = grid.copy()
                expected[:k, :] = grid[0, 0]
                expected[-k:, :] = grid[0, 0]
                expected[:, :k] = grid[0, 0]
                expected[:, -k:] = grid[0, 0]
                assert np.array_equal(expected, np.asarray(pair["output"]))
                if split == "test":
                    regimes.add(m["regime"])
                    stats["deeper_than_train"] += m["added"] > max(train_added)
                    stats["larger_than_train"] += grid.size > train_area
    assert len(shapes) >= 80, len(shapes)
    assert added >= {1, 2, 3, 4, 5}, added
    assert thick == {1, 2} and parities == {0, 1}
    assert len(frame_colors) >= 9 and len(backgrounds) >= 7
    assert set(_DEEP_KINDS) <= kinds, kinds
    assert regimes == {"deeper", "larger", "familiar"}, regimes
    assert stats["satellite"] >= 42 and stats["gap"] >= 42 and stats["margin"] >= 60
    assert stats["stripe"] >= 5 and stats["zone_sym"] >= 20 and stats["two_tone"] >= 20
    assert stats["tb_band"] >= 42 and stats["self_symmetric_deepest"] >= 42
    assert stats["deeper_than_train"] >= 8 and stats["larger_than_train"] >= 8
    assert stats["within_puzzle_fc_change"] >= 10 and stats["within_puzzle_bg_change"] >= 5
    # Determinism and explicit count boundaries.
    for seed, counts in ((7, (2, 1)), (8, (10, 4)), (9, (3, 2)), (10, (5, 4))):
        first = generate_puzzle(seed, *counts)
        assert (len(first["train"]), len(first["test"])) == counts
        assert _fingerprint(first) == _fingerprint(generate_puzzle(seed, *counts))
        validate_generated_puzzle(first)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {counts}")
