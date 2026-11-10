"""Docking conveyors: a composition of ARC 5daaa586 (A) and 79cce52d (B).

Rule.  Two full-span horizontal guides and two full-span vertical guides, in
four distinct colors, bound an inclusive window (A).  Exactly one guide color
also appears off the guide lines; that guide is the active "dock".  Inside the
window every track perpendicular to the dock (rows for a left/right dock,
columns for a top/bottom dock) holds at most one marker of the dock color.  A
marked track is a cyclic conveyor belt (B): it is rotated toward the dock until
its marker sits directly beside the dock, and the payload cells that were
between the dock and the marker wrap around to the far end of the track.
Unmarked tracks, the guides, and the crossing cells are kept unchanged.  The
output is the inclusive window; everything outside it (including exterior
dock-colored distractors) is discarded.

Only the standard library and NumPy are used.
"""
from __future__ import annotations

import numpy as np

SIDES = ("top", "bottom", "left", "right")
VERTICAL_DOCKS = ("left", "right")      # vertical guide -> row tracks
HORIZONTAL_DOCKS = ("top", "bottom")    # horizontal guide -> column tracks
STYLES = ("patches", "regions", "bars", "sparse", "diagonal", "mosaic", "rings")
SCHEDULES = ("unseen", "seen", "mixed")
PALETTE_REGIMES = ("fixed", "guides_fixed", "local")
SIZE_REGIMES = ("compact", "medium", "large", "mixed", "growing")
DEPTH_MODES = ("spread", "shallow", "deep", "stair")
# canonical (tracks, length) ranges; canonical frame = dock on the left
_BANDS = {
    "compact": ((3, 6), (4, 8)),
    "medium": ((5, 11), (6, 12)),
    "large": ((9, 17), (10, 18)),
    "xl": ((14, 23), (15, 24)),
}
_ALTERNATIVES = ("paint", "insert", "away", "nearest", "opposite", "crop")


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _side_map(k: int) -> dict:
    """Canonical side -> final side after np.rot90(grid, k)."""
    probe = np.zeros((5, 5), dtype=int)
    probe[0, 2], probe[4, 2], probe[2, 0], probe[2, 4] = 1, 2, 3, 4
    rotated = np.rot90(probe, k)
    where = {int(rotated[0, 2]): "top", int(rotated[4, 2]): "bottom",
             int(rotated[2, 0]): "left", int(rotated[2, 4]): "right"}
    return {"top": where[1], "bottom": where[2], "left": where[3], "right": where[4]}


_SIDE_MAPS = {k: _side_map(k) for k in range(4)}
_K_FOR_DOCK = {_SIDE_MAPS[k]["left"]: k for k in range(4)}


def _to_view(inner: np.ndarray, side: str) -> np.ndarray:
    """View of the window interior in which tracks are rows docking toward column 0."""
    view = inner.T if side in HORIZONTAL_DOCKS else inner
    if side in ("right", "bottom"):
        view = view[:, ::-1]
    return np.array(view)


def _from_view(view: np.ndarray, side: str) -> np.ndarray:
    if side in ("right", "bottom"):
        view = view[:, ::-1]
    if side in HORIZONTAL_DOCKS:
        view = view.T
    return np.array(view)


# --------------------------------------------------------------------------
# input-only parsing and solving
# --------------------------------------------------------------------------
def _dominant(line: np.ndarray) -> tuple[int, int]:
    values, counts = np.unique(line, return_counts=True)
    i = int(np.argmax(counts))
    return int(values[i]), int(counts[i])


def _parse(grid) -> dict:
    """Recover the window, guide colors, dock side and markers from an input grid."""
    g = np.asarray(grid)
    if g.ndim != 2 or g.shape[0] < 5 or g.shape[1] < 5:
        raise ValueError("input must be a 2D grid of at least 5x5")
    g = g.astype(int)
    height, width = g.shape
    border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    values, counts = np.unique(border, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(order) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("background is ambiguous")
    bg = int(values[order[0]])

    def guide_color(line):
        color, count = _dominant(line)
        if color != bg and len(line) - count <= 2:
            return color
        return None

    rows = [r for r in range(height) if guide_color(g[r]) is not None]
    cols = [c for c in range(width) if guide_color(g[:, c]) is not None]
    if len(rows) != 2 or len(cols) != 2:
        raise ValueError("expected exactly two horizontal and two vertical guides")
    top, bottom = rows
    left, right = cols
    if bottom - top < 3 or right - left < 3:
        raise ValueError("window interior must be at least 2x2")
    colors = {"top": guide_color(g[top]), "bottom": guide_color(g[bottom]),
              "left": guide_color(g[:, left]), "right": guide_color(g[:, right])}
    if len(set(colors.values())) != 4:
        raise ValueError("guide colors must be distinct")
    # guide lines must be uniform apart from the four crossings
    for side, index in (("top", top), ("bottom", bottom)):
        line = g[index].copy()
        for side_v, c in (("left", left), ("right", right)):
            if line[c] not in (colors[side], colors[side_v]):
                raise ValueError("crossing cell must show one of its guides")
            line[c] = colors[side]
        if np.any(line != colors[side]):
            raise ValueError("horizontal guide is broken")
    for side, index in (("left", left), ("right", right)):
        line = g[:, index].copy()
        for side_h, r in (("top", top), ("bottom", bottom)):
            if line[r] not in (colors[side], colors[side_h]):
                raise ValueError("crossing cell must show one of its guides")
            line[r] = colors[side]
        if np.any(line != colors[side]):
            raise ValueError("vertical guide is broken")
    off = np.ones_like(g, dtype=bool)
    off[[top, bottom], :] = False
    off[:, [left, right]] = False
    active = [s for s in SIDES if np.any(g[off] == colors[s])]
    if len(active) != 1:
        raise ValueError("exactly one guide color must appear off the guides")
    side = active[0]
    marker = colors[side]
    inner = g[top + 1:bottom, left + 1:right]
    view = _to_view(inner, side)
    depths = {}
    for t in range(view.shape[0]):
        hits = np.flatnonzero(view[t] == marker)
        if len(hits) > 1:
            raise ValueError("a track holds more than one marker")
        if len(hits) == 1:
            depths[t] = int(hits[0])
    if not depths:
        raise ValueError("no marker inside the window")
    exterior = off.copy()
    exterior[top:bottom + 1, left:right + 1] = False
    return {"bg": bg, "top": top, "bottom": bottom, "left": left, "right": right,
            "colors": colors, "side": side, "marker": marker, "depths": depths,
            "view": view, "distractors": int(np.sum(g[exterior] == marker)),
            "grid": g}


def _render(info: dict, view: np.ndarray) -> np.ndarray:
    g = info["grid"]
    crop = g[info["top"]:info["bottom"] + 1, info["left"]:info["right"] + 1].copy()
    crop[1:-1, 1:-1] = _from_view(view, info["side"])
    return crop


def _docked_view(info: dict, mode: str = "dock") -> np.ndarray:
    """Apply the true rule (mode='dock') or a shortcut alternative to the view."""
    view = info["view"].copy()
    length = view.shape[1]
    marker = info["marker"]
    for t, d in info["depths"].items():
        row = view[t].copy()
        if mode == "dock":
            view[t] = np.roll(row, -d)
        elif mode == "paint":            # parent A alone: fill dock-to-marker segment
            view[t, :d + 1] = marker
        elif mode == "insert":           # slide only the marker, no wraparound
            rest = np.concatenate([row[:d], row[d + 1:]])
            view[t] = np.concatenate([[marker], rest])
        elif mode == "away":             # parent B's direction: away from the reference
            view[t] = np.roll(row, d)
        elif mode == "nearest":          # dock against whichever end is closer
            view[t] = np.roll(row, -d) if d <= length - 1 - d else np.roll(row, length - 1 - d)
        elif mode == "opposite":         # dock against the inactive opposite guide
            view[t] = np.roll(row, length - 1 - d)
        elif mode == "crop":             # parent B ignored: crop only
            pass
        else:
            raise ValueError(mode)
    return view


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    return _render(info, _docked_view(info, "dock"))


def _alternative(input_grid: np.ndarray, mode: str) -> np.ndarray:
    info = _parse(input_grid)
    return _render(info, _docked_view(info, mode))


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _check_counts(num_train, num_test) -> tuple[int, int]:
    """Legal counts: 2..10 train, 1..4 test, train > test.  None means the default."""
    def coerce(value, name):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        return int(value)
    train = coerce(num_train, "num_train")
    test = coerce(num_test, "num_test")
    if test is None:
        test = 1
    if train is None:
        train = max(4, test + 1)
    if not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_palette(rng, plan: dict) -> tuple[int, dict, list]:
    if plan["palette"] in ("fixed", "guides_fixed"):
        bg, side_colors = plan["bg"], plan["side_colors"]
    else:
        perm = [int(x) for x in rng.permutation(10)]
        bg, side_colors = perm[0], dict(zip(SIDES, perm[1:5]))
    if plan["palette"] == "fixed":
        return bg, side_colors, list(plan["payload_colors"])
    return bg, side_colors, _sample_payload_colors(rng, bg, side_colors)


def _sample_payload_colors(rng, bg: int, side_colors: dict) -> list:
    remaining = [c for c in range(10) if c != bg and c not in side_colors.values()]
    include_bg = rng.random() < 0.4
    extra_count = int(rng.integers(1 if include_bg else 2, 5))
    extra = [int(c) for c in rng.choice(remaining, size=extra_count, replace=False)]
    return ([bg] if include_bg else []) + extra


def _sample_size(rng, band: str) -> tuple[int, int]:
    (t_lo, t_hi), (l_lo, l_hi) = _BANDS[band]
    return int(rng.integers(t_lo, t_hi + 1)), int(rng.integers(l_lo, l_hi + 1))


def _sample_margins(rng, inner: int) -> tuple[int, int]:
    available = 30 - (inner + 2)
    cap = min(6, available - 1)
    before = int(rng.integers(1, cap + 1))
    after = int(rng.integers(1, min(6, available - before) + 1))
    return before, after


def _make_payload(rng, n: int, length: int, style: str, colors: list) -> np.ndarray:
    k = len(colors)
    pal = np.array(colors, dtype=int)
    if style == "patches":
        payload = np.full((n, length), pal[0])
        for _ in range(int(rng.integers(2, 4 + (n * length) // 30 + 1))):
            h = int(rng.integers(1, max(1, (2 * n) // 3) + 1))
            w = int(rng.integers(1, max(1, (2 * length) // 3) + 1))
            r0 = int(rng.integers(0, n - h + 1))
            c0 = int(rng.integers(0, length - w + 1))
            payload[r0:r0 + h, c0:c0 + w] = pal[int(rng.integers(1, k))]
    elif style == "regions":
        m = int(rng.integers(3, max(3, min(9, (n * length) // 4)) + 1))
        pts = np.stack([rng.integers(0, n, m), rng.integers(0, length, m)], axis=1)
        owner = [pal[i % k] for i in range(m)]
        owner = [owner[i] for i in rng.permutation(m)]
        rr, cc = np.mgrid[0:n, 0:length]
        dist = np.abs(rr[..., None] - pts[:, 0]) + np.abs(cc[..., None] - pts[:, 1])
        payload = np.array(owner)[np.argmin(dist, axis=2)]
    elif style == "bars":
        pattern, c = [], 0
        previous = -1
        while c < length:
            w = int(rng.integers(1, 4))
            choices = [x for x in range(k) if x != previous]
            previous = int(rng.choice(choices))
            pattern += [pal[previous]] * w
            c += w
        payload = np.tile(np.array(pattern[:length]), (n, 1))
        for _ in range(int(rng.integers(0, 2 + n // 3))):
            r0 = int(rng.integers(0, n))
            c0 = int(rng.integers(0, length))
            payload[r0, c0:c0 + int(rng.integers(1, 3))] = pal[int(rng.integers(0, k))]
    elif style == "sparse":
        payload = np.full((n, length), pal[0])
        shapes = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
        target = rng.uniform(0.18, 0.4) * n * length
        filled, tries = 0, 0
        while filled < target and tries < 200:
            tries += 1
            h, w = shapes[int(rng.integers(len(shapes)))]
            if h > n or w > length:
                continue
            r0 = int(rng.integers(0, n - h + 1))
            c0 = int(rng.integers(0, length - w + 1))
            if np.any(payload[r0:r0 + h, c0:c0 + w] != pal[0]):
                continue
            payload[r0:r0 + h, c0:c0 + w] = pal[int(rng.integers(1, k))]
            filled += h * w
    elif style == "diagonal":
        slope = int(rng.choice((-1, 1)))
        w = int(rng.integers(1, 3))
        rr, cc = np.mgrid[0:n, 0:length]
        payload = pal[((cc + slope * rr + 40) // w) % k]
    elif style == "mosaic":
        bh, bw = int(rng.integers(1, 3)), int(rng.integers(1, 4))
        blocks = rng.integers(0, k, size=(n // bh + 1, length // bw + 1))
        payload = pal[np.repeat(np.repeat(blocks, bh, axis=0), bw, axis=1)[:n, :length]]
    elif style == "rings":
        w = int(rng.integers(1, 3))
        cr = int(rng.integers(-1, 2))
        rr, cc = np.mgrid[0:n, 0:length]
        depth = np.minimum.reduce([rr, cc, n - 1 - rr + cr, length - 1 - cc])
        payload = pal[(np.maximum(depth, 0) // w) % k]
    else:
        raise ValueError(style)
    return np.asarray(payload, dtype=int)


def _sample_marks(rng, n: int, length: int, mode: str, *, need_far: bool,
                  need_unmarked: bool, dense: bool) -> dict:
    max_marked = n - 1 if need_unmarked else n
    for _ in range(60):
        if dense:
            frac = rng.uniform(0.7, 1.0)
        else:
            frac = rng.uniform(0.3, 0.95)
        count = int(np.clip(round(frac * n), 2, max(2, max_marked)))
        tracks = sorted(int(t) for t in rng.choice(n, size=count, replace=False))
        if mode == "shallow":
            depths = rng.integers(0, max(2, length // 2 + 1), size=count)
        elif mode == "deep":
            depths = rng.integers(length // 2, length, size=count)
        elif mode == "stair":
            step = int(rng.choice((-2, -1, 1, 1, 2)))
            start = int(rng.integers(0, length))
            depths = np.array([(start + step * i) % length for i in range(count)])
        else:
            depths = rng.integers(0, length, size=count)
        marks = {t: int(d) for t, d in zip(tracks, depths)}
        nonzero = {d for d in marks.values() if d > 0}
        if len(nonzero) < 2:
            continue
        if not any(2 <= d <= length - 2 for d in marks.values()):
            continue
        if need_far and not any(2 * d > length - 1 and d <= length - 2 for d in marks.values()):
            continue
        if need_unmarked and len(marks) >= n:
            continue
        return marks
    return {}


def _place_distractors(rng, grid, geometry, count: int, marker: int, on_track: bool) -> int:
    height, width = grid.shape
    top, bottom, left, right = geometry
    cells = []
    track_cells = []
    corners = {(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)}
    for r in range(height):
        if r in (top, bottom):
            continue
        for c in range(width):
            if c in (left, right) or (top <= r <= bottom and left <= c <= right):
                continue
            if (r, c) in corners:
                continue
            cells.append((r, c))
            if top < r < bottom:
                track_cells.append((r, c))
    count = min(count, len(cells) // 3)
    chosen = []
    if on_track and track_cells and count > 0:
        chosen.append(track_cells[int(rng.integers(len(track_cells)))])
    pool = [cell for cell in cells if cell not in chosen]
    extra = max(0, count - len(chosen))
    if extra:
        idx = rng.choice(len(pool), size=min(extra, len(pool)), replace=False)
        chosen += [pool[int(i)] for i in idx]
    for r, c in chosen:
        grid[r, c] = marker
    return len(chosen)


def _make_pair(rng, plan: dict, spec: dict):
    side = spec["side"]
    k = _K_FOR_DOCK[side]
    smap = _SIDE_MAPS[k]
    for _ in range(80):
        bg, side_colors, payload_colors = _sample_palette(rng, plan)
        canon = {c: side_colors[smap[c]] for c in SIDES}      # canonical role -> color
        marker = canon["left"]
        n, length = _sample_size(rng, spec["band"])
        mt, mb = _sample_margins(rng, n)
        ml, mr = _sample_margins(rng, length)
        height, width = mt + n + 2 + mb, ml + length + 2 + mr
        top, bottom, left, right = mt, mt + n + 1, ml, ml + length + 1
        grid = np.full((height, width), bg, dtype=int)
        grid[top, :] = canon["top"]
        grid[bottom, :] = canon["bottom"]
        grid[:, left] = canon["left"]
        grid[:, right] = canon["right"]
        for r, hside in ((top, "top"), (bottom, "bottom")):
            for c, vside in ((left, "left"), (right, "right")):
                grid[r, c] = canon[hside] if rng.random() < 0.5 else canon[vside]
        style = spec["style"]
        payload = _make_payload(rng, n, length, style, payload_colors)
        if len(np.unique(payload)) < 2:
            continue
        marks = _sample_marks(rng, n, length, spec["depth_mode"], need_far=spec["far"],
                              need_unmarked=spec["unmarked"], dense=spec["dense"])
        if not marks:
            continue
        for t, d in marks.items():
            payload[t, d] = marker
        # the wraparound must be visible on at least one marked track
        visible = 0
        for t, d in marks.items():
            row = payload[t]
            rolled = np.roll(row, -d)
            inserted = np.concatenate([[marker], row[:d], row[d + 1:]])
            if not np.array_equal(rolled, inserted):
                visible += 1
        if visible < max(1, len(marks) // 3):
            continue
        grid[top + 1:bottom, left + 1:right] = payload
        n_distract = {"none": 0, "sparse": int(rng.integers(1, 4)), "medium": int(rng.integers(4, 9)),
                      "heavy": int(rng.integers(9, 17))}[spec["clutter"]]
        placed = _place_distractors(rng, grid, (top, bottom, left, right), n_distract, marker,
                                    on_track=spec["clutter"] != "none")
        if spec["clutter"] != "none" and placed == 0:
            continue
        output = grid[top:bottom + 1, left:right + 1].copy()
        for t, d in marks.items():
            output[1 + t, 1:-1] = np.roll(output[1 + t, 1:-1], -d)
        inp = np.rot90(grid, k).copy()
        out = np.rot90(output, k).copy()
        try:
            info = _parse(inp)
        except ValueError:
            continue
        if info["bg"] != bg or info["side"] != side or len(info["depths"]) != len(marks):
            continue
        if not np.array_equal(solve(inp), out):
            continue
        meta = {"side": side, "style": style, "band": spec["band"], "marks": len(marks),
                "depths": sorted(marks.values()), "length": length, "tracks": n,
                "distractors": placed, "clutter": spec["clutter"], "payload_colors": len(payload_colors),
                "bg": bg, "depth_mode": spec["depth_mode"]}
        return inp, out, meta
    return None


def _plan(rng, num_train: int, num_test: int) -> dict:
    plan = {}
    plan["schedule"] = str(rng.choice(SCHEDULES, p=(0.45, 0.3, 0.25)))
    plan["palette"] = str(rng.choice(PALETTE_REGIMES, p=(0.25, 0.25, 0.5)))
    perm = [int(x) for x in rng.permutation(10)]
    plan["bg"], plan["side_colors"] = perm[0], dict(zip(SIDES, perm[1:5]))
    plan["payload_colors"] = _sample_payload_colors(rng, plan["bg"], plan["side_colors"])
    plan["size"] = str(rng.choice(SIZE_REGIMES, p=(0.15, 0.25, 0.2, 0.25, 0.15)))
    shared_style = rng.random() < 0.4
    style0 = str(rng.choice(STYLES))

    # ---- dock sides: training always shows both a row dock and a column dock
    if plan["schedule"] == "unseen":
        held = str(rng.choice(SIDES))
        pool = [s for s in SIDES if s != held]
        if num_train >= 3:
            train_sides = pool + [str(rng.choice(pool)) for _ in range(num_train - 3)]
        else:
            vertical = [s for s in pool if s in VERTICAL_DOCKS]
            horizontal = [s for s in pool if s in HORIZONTAL_DOCKS]
            train_sides = [str(rng.choice(vertical)), str(rng.choice(horizontal))]
        test_sides = [held] + [held if rng.random() < 0.5 else str(rng.choice(SIDES))
                               for _ in range(num_test - 1)]
    else:
        while True:
            train_sides = [str(rng.choice(SIDES)) for _ in range(num_train)]
            if any(s in VERTICAL_DOCKS for s in train_sides) and any(s in HORIZONTAL_DOCKS for s in train_sides):
                break
        source = sorted(set(train_sides)) if plan["schedule"] == "seen" else list(SIDES)
        test_sides = [str(rng.choice(source)) for _ in range(num_test)]
    order = rng.permutation(num_train)
    train_sides = [train_sides[int(i)] for i in order]

    # ---- size bands
    def band(is_test):
        size = plan["size"]
        if size == "mixed":
            return str(rng.choice(("compact", "medium", "large", "xl"), p=(0.25, 0.35, 0.28, 0.12)))
        if size == "growing":
            return str(rng.choice(("large", "xl"))) if is_test else str(rng.choice(("compact", "medium")))
        if size == "large":
            return str(rng.choice(("large", "xl"), p=(0.7, 0.3)))
        return size

    # ---- training duties spread across random pairs
    duties = {name: int(rng.integers(num_train)) for name in ("far", "unmarked", "distract")}
    specs = []
    for i in range(num_train + num_test):
        is_test = i >= num_train
        side = test_sides[i - num_train] if is_test else train_sides[i]
        clutter = str(rng.choice(("none", "sparse", "medium", "heavy"), p=(0.25, 0.35, 0.25, 0.15)))
        challenge = str(rng.choice(("plain", "dense", "deep", "clutter"))) if is_test else "plain"
        if not is_test and duties["distract"] == i and clutter == "none":
            clutter = "sparse"
        if challenge == "clutter":
            clutter = "heavy"
        depth_mode = "deep" if challenge == "deep" else str(rng.choice(DEPTH_MODES, p=(0.45, 0.2, 0.15, 0.2)))
        specs.append({
            "side": side, "band": band(is_test), "style": style0 if shared_style else str(rng.choice(STYLES)),
            "clutter": clutter, "depth_mode": depth_mode,
            "far": (not is_test and duties["far"] == i) or rng.random() < 0.3,
            "unmarked": (not is_test and duties["unmarked"] == i) or (rng.random() < 0.6 and challenge != "dense"),
            "dense": challenge == "dense" or rng.random() < 0.2,
            "test": is_test, "challenge": challenge,
        })
    plan["specs"] = specs
    plan["train_sides"], plan["test_sides"] = train_sides, test_sides
    plan["shared_style"] = bool(shared_style)
    return plan


def _demonstrations_ok(train: list, test: list) -> bool:
    infos = [_parse(p["input"]) for p in train]
    sides = {info["side"] for info in infos}
    if not (sides & set(VERTICAL_DOCKS)) or not (sides & set(HORIZONTAL_DOCKS)):
        return False
    for mode in _ALTERNATIVES:
        if all(np.array_equal(_alternative(p["input"], mode), p["output"]) for p in train):
            return False
    if not any(info["distractors"] > 0 for info in infos):
        return False
    if not any(len(info["depths"]) < info["view"].shape[0] for info in infos):
        return False
    for p in test:
        for mode in ("paint", "insert", "crop", "opposite"):
            if np.array_equal(_alternative(p["input"], mode), p["output"]):
                return False
    return True


def _generate(seed: int, num_train: int = 4, num_test: int = 1):
    num_train, num_test = _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    for _ in range(60):
        plan = _plan(rng, num_train, num_test)
        pairs, metas = [], []
        for spec in plan["specs"]:
            made = _make_pair(rng, plan, spec)
            if made is None:
                break
            pairs.append({"input": made[0], "output": made[1]})
            metas.append(made[2])
        if len(pairs) != num_train + num_test:
            continue
        keys = {p["input"].tobytes() + bytes(str(p["input"].shape), "ascii") for p in pairs}
        if len(keys) != len(pairs):
            continue
        train, test = pairs[:num_train], pairs[num_train:]
        if not _demonstrations_ok(train, test):
            continue
        puzzle = {"train": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in train],
                  "test": [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p in test]}
        meta = {"schedule": plan["schedule"], "palette": plan["palette"], "size": plan["size"],
                "shared_style": plan["shared_style"], "pairs": metas}
        return puzzle, meta
    raise RuntimeError("could not build a puzzle for this seed")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    return _generate(seed, num_train, num_test)[0]


# --------------------------------------------------------------------------
# validation and self test
# --------------------------------------------------------------------------
def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_generated_puzzle(puzzle: dict) -> None:
    """Re-derive every family invariant from the grids alone; raise AssertionError/ValueError."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    seen = set()
    pairs = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for grid in (inp, out):
                _require(grid.ndim == 2 and 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "integer cells")
                _require(np.all((grid >= 0) & (grid <= 9)), "colors 0..9")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _parse(inp)
            colors = info["colors"]
            # every guide has a nonempty exterior margin on both sides
            h, w = inp.shape
            _require(info["top"] >= 1 and info["left"] >= 1 and info["bottom"] <= h - 2
                     and info["right"] <= w - 2, "margins")
            _require(info["bg"] not in colors.values(), "background differs from guides")
            inner = inp[info["top"] + 1:info["bottom"], info["left"] + 1:info["right"]]
            inactive = [colors[s] for s in SIDES if s != info["side"]]
            _require(not np.any(np.isin(inner, inactive)), "inactive guide colors stay on the guides")
            depths = info["depths"]
            _require(len(depths) >= 2, "at least two marked tracks")
            _require(len({d for d in depths.values() if d > 0}) >= 2, "two distinct nonzero dock distances")
            expected = solve(inp)
            _require(np.array_equal(expected, out), "output follows the docking rule")
            _require(out.shape == (info["bottom"] - info["top"] + 1, info["right"] - info["left"] + 1), "inclusive crop")
            # docking is a per-track cyclic permutation: multisets preserved, markers touch the dock
            view_out = _to_view(out[1:-1, 1:-1], info["side"])
            for t in range(info["view"].shape[0]):
                _require(sorted(view_out[t]) == sorted(info["view"][t]), "track content preserved")
            for t in depths:
                _require(view_out[t, 0] == info["marker"], "marker docks beside its guide")
            _require(not np.array_equal(_alternative(inp, "insert"), out), "wraparound visible")
            pairs.append((split, info, inp, out))
    train_infos = [info for split, info, _, _ in pairs if split == "train"]
    sides = {info["side"] for info in train_infos}
    _require(sides & set(VERTICAL_DOCKS) and sides & set(HORIZONTAL_DOCKS), "both dock orientations trained")
    for mode in _ALTERNATIVES:
        _require(any(not np.array_equal(_alternative(inp, mode), out)
                     for split, _, inp, out in pairs if split == "train"), f"training refutes {mode}")
    _require(any(info["distractors"] for info in train_infos), "training shows exterior distractors")
    _require(any(len(info["depths"]) < info["view"].shape[0] for info in train_infos), "training shows an unmarked track")
    for split, info, inp, out in pairs:
        if split == "test":
            for mode in ("paint", "insert", "crop", "opposite"):
                _require(not np.array_equal(_alternative(inp, mode), out), f"test separates {mode}")


def self_test() -> None:
    """Bounded checks: counts, determinism, invariants, symmetry of solve, variation over 42 seeds."""
    import collections
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4, 5)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted {bad}")
    for bad in (1.5, "3", True):
        try:
            generate_puzzle(0, bad, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("non-integer count accepted")
    for counts in ((2, 1), (3, 2), (10, 4), (5, 4)):
        p = generate_puzzle(7, *counts)
        validate_generated_puzzle(p)
        _require((len(p["train"]), len(p["test"])) == counts, "explicit counts honored")
    stats = collections.defaultdict(collections.Counter)
    shapes = set()
    signatures = set()
    for seed in range(42):
        puzzle, meta = _generate(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(str(puzzle))
        stats["schedule"][meta["schedule"]] += 1
        stats["palette"][meta["palette"]] += 1
        stats["size"][meta["size"]] += 1
        for i, pm in enumerate(meta["pairs"]):
            split = "train" if i < len(puzzle["train"]) else "test"
            stats[split + "_side"][pm["side"]] += 1
            stats["style"][pm["style"]] += 1
            stats["band"][pm["band"]] += 1
            stats["clutter"][pm["clutter"]] += 1
            stats["marks"][pm["marks"]] += 1
            stats["max_depth"][max(pm["depths"])] += 1
        for pair in puzzle["train"] + puzzle["test"]:
            shapes.add(np.asarray(pair["input"]).shape)
        if seed < 6:
            # the rule is orientation-free: solve commutes with every rotation/reflection
            inp = np.asarray(puzzle["test"][0]["input"])
            out = np.asarray(puzzle["test"][0]["output"])
            for f in (np.rot90, np.transpose, np.fliplr, np.flipud, lambda a: np.rot90(a, 3)):
                _require(np.array_equal(solve(f(inp)), f(out)), "solve is dihedrally equivariant")
        if seed < 3:
            _require(str(generate_puzzle(seed)) == str(puzzle), "deterministic")
    _require(len(signatures) == 42, "distinct puzzles")
    _require(len(stats["test_side"]) == 4 and len(stats["train_side"]) == 4, "all dock sides")
    _require(len(stats["style"]) >= 6, "payload styles")
    _require(len(stats["schedule"]) == 3 and len(stats["palette"]) == 3, "puzzle regimes")
    _require(len(stats["band"]) == 4 and len(stats["clutter"]) == 4, "geometry and clutter bands")
    _require(len(shapes) >= 60, "input shape diversity")
    _require(min(stats["marks"]) == 2 and max(stats["marks"]) >= 12, "marker count range")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
