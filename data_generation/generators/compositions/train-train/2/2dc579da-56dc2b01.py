"""Nested docking: extract the marked compartment and rebuild a partition in it.

Input: a fill-coloured canvas cut by complete divider bands into a 1-4 by 1-4
array of compartments (never a single compartment).  Each band is a run of
complete rows (or columns) without any fill cell; each band has its own
thickness (1-4) and its own colour, and where a horizontal and a vertical band
cross, one orientation is consistently drawn on top.  Exactly one compartment
is nonuniform: it holds a content block (one or more parts, one to four
colours, never a band colour).  That compartment sits at an end of every
divided axis, so it touches at most one band per axis (its "walls").

Output: the marked compartment, cropped without dividers (parent A).  Inside
it the whole content block slides rigidly, as one bounding box, toward each
wall until flush (parent B), and every wall is copied - its own colour, its
own thickness, full crop length - immediately beyond the far edge of the
docked block (parent B's bracketing line, drawn as a translated copy of that
wall).  Where the two copies cross, the crossing repeats the input's junction
of the two walls.  The crop therefore becomes a smaller partitioned grid whose
marked sub-compartment is exactly the content's bounding box.

solve() reads every convention from the input alone: fill is the strict
majority colour, bands are the fill-free complete rows and columns, and the
walls with their colours, thicknesses and junction are the bands adjacent to
the marked compartment.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

MAX_SIDE = 30
# Docking kind -> (vertical sign, horizontal sign); "+" means toward larger
# indices (down/right).  The kind names describe the movement direction.
DOCK = {
    "DR": ("+", "+"), "DL": ("+", "-"), "UR": ("-", "+"), "UL": ("-", "-"),
    "D": ("+", None), "U": ("-", None), "R": (None, "+"), "L": (None, "-"),
}
CORNERS = ("DR", "DL", "UR", "UL")
EDGES = ("D", "U", "R", "L")
OPPOSITE = {"DR": "UL", "UL": "DR", "DL": "UR", "UR": "DL"}
# Strips that complete a corner's direction coverage (the two opposite signs).
COMPLEMENT = {"DR": ("U", "L"), "DL": ("U", "R"), "UR": ("D", "L"), "UL": ("D", "R")}
REGIMES = ("unseen_corner", "scale_up", "scale_down", "layout_shift", "shared",
           "thickness_up", "compose_axes", "strip_test")
REGIME_P = (0.2, 0.12, 0.08, 0.14, 0.16, 0.1, 0.08, 0.12)
TIERS = ("minimal", "maximal", "normal")
TIER_P = (1 / 14, 1 / 7, 1 - 1 / 14 - 1 / 7)
SIMPLE_STYLES = ("single", "bar", "blob", "hollow")
RICH_STYLES = ("single", "bar", "blob", "hollow", "multi", "pieces", "scatter",
               "diagonal", "glyph", "framed", "pattern", "nested")
IRREGULAR = ("blob", "hollow", "multi", "pieces", "scatter", "diagonal", "glyph", "nested")
PIECE_STYLES = ("pieces", "scatter", "nested")
MULTI_COLOUR_STYLES = ("multi", "framed")
# Smallest box side each style needs (so compartments are sized for it).
STYLE_MIN = {"single": 1, "bar": 2, "blob": 2, "multi": 2, "hollow": 3, "pieces": 3,
             "scatter": 3, "diagonal": 2, "glyph": 3, "framed": 3, "pattern": 2, "nested": 5}
SMALL_STYLES = ("bar", "blob", "multi", "diagonal", "pattern")
# Styles that can span a strip's whole unwalled extent without forming a
# fill-free line (which would read as a divider band).
FULL_SPAN_STYLES = ("blob", "multi", "scatter", "diagonal", "pieces", "pattern")
PLACEMENTS = ("flush", "near_wall", "middle", "near_outer", "outer_edge")
PLACEMENT_P = (0.18, 0.2, 0.14, 0.24, 0.24)
WALLED_BINS = ((3, 6), (5, 9), (7, 12), (10, 16))
WALLED_P = (0.25, 0.35, 0.28, 0.12)
UNWALLED_BINS = ((3, 8), (6, 14), (10, 20), (16, 30))
UNWALLED_P = (0.25, 0.35, 0.25, 0.15)


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
def _resolve_counts(num_train, num_test):
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
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


# ---------------------------------------------------------------------------
# Input analysis and the rule (shared by solve and validation)
# ---------------------------------------------------------------------------
def _runs(indices):
    runs = []
    for i in indices:
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [(a, b) for a, b in runs]


def _cells(n, bands):
    out, start = [], 0
    for a, b in bands:
        if a > start:
            out.append((start, a - 1))
        start = b + 1
    if start < n:
        out.append((start, n - 1))
    return out


def _wall(lo, hi, bands):
    """Sign and band index of the single divider band touching [lo, hi]."""
    before = [k for k, b in enumerate(bands) if b[1] == lo - 1]
    after = [k for k, b in enumerate(bands) if b[0] == hi + 1]
    if before and after:
        raise ValueError("marked compartment is walled on both sides of one axis")
    if after:
        return "+", after[0]
    if before:
        return "-", before[0]
    return None, None


def _uniform(segment, what):
    values = np.unique(segment)
    if len(values) != 1:
        raise ValueError(f"{what} is not uniform")
    return int(values[0])


def _analyze(grid):
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("grid must be a nonempty 2D array")
    g = g.astype(int)
    height, width = g.shape
    values, counts = np.unique(g, return_counts=True)
    if len(values) < 3:
        raise ValueError("need fill, divider and content colours")
    order = np.argsort(-counts, kind="stable")
    if counts[order[0]] == counts[order[1]]:
        raise ValueError("fill colour is not a strict majority")
    fill = int(values[order[0]])
    free = g != fill
    band_rows = [int(r) for r in np.flatnonzero(free.all(axis=1))]
    band_cols = [int(c) for c in np.flatnonzero(free.all(axis=0))]
    if not band_rows and not band_cols:
        raise ValueError("no divider band")
    if len(band_rows) == height or len(band_cols) == width:
        raise ValueError("no compartments")
    row_bands, col_bands = _runs(band_rows), _runs(band_cols)
    in_row = np.zeros(height, dtype=bool)
    in_row[band_rows] = True
    in_col = np.zeros(width, dtype=bool)
    in_col[band_cols] = True
    h_colours = [_uniform(g[a:b + 1][:, ~in_col], "horizontal band") for a, b in row_bands]
    v_colours = [_uniform(g[:, a:b + 1][~in_row], "vertical band") for a, b in col_bands]
    priority = None
    for i, (a, b) in enumerate(row_bands):
        for j, (c, d) in enumerate(col_bands):
            cross = _uniform(g[a:b + 1, c:d + 1], "band crossing")
            if h_colours[i] == v_colours[j]:
                if cross != h_colours[i]:
                    raise ValueError("crossing colour differs from both bands")
                continue
            if cross == h_colours[i]:
                found = "h"
            elif cross == v_colours[j]:
                found = "v"
            else:
                raise ValueError("crossing colour differs from both bands")
            if priority not in (None, found):
                raise ValueError("inconsistent crossing priority")
            priority = found
    row_cells, col_cells = _cells(height, row_bands), _cells(width, col_bands)
    marked = [(i, j) for i, (r0, r1) in enumerate(row_cells)
              for j, (c0, c1) in enumerate(col_cells)
              if np.any(free[r0:r1 + 1, c0:c1 + 1])]
    if len(marked) != 1:
        raise ValueError("expected exactly one nonuniform compartment")
    i, j = marked[0]
    (r0, r1), (c0, c1) = row_cells[i], col_cells[j]
    vsign, vband = _wall(r0, r1, row_bands)
    hsign, hband = _wall(c0, c1, col_bands)
    if vsign is None and hsign is None:
        raise ValueError("marked compartment touches no divider")
    crop = g[r0:r1 + 1, c0:c1 + 1]
    band_colours = set(h_colours) | set(v_colours)
    content_colours = {int(v) for v in np.unique(crop)} - {fill}
    if content_colours & band_colours:
        raise ValueError("content uses a divider colour")
    cells = np.argwhere(crop != fill)
    (a0, b0), (a1, b1) = cells.min(axis=0), cells.max(axis=0)
    t_r = row_bands[vband][1] - row_bands[vband][0] + 1 if vsign else 0
    t_c = col_bands[hband][1] - col_bands[hband][0] + 1 if hsign else 0
    colour_r = h_colours[vband] if vsign else None
    colour_c = v_colours[hband] if hsign else None
    junction = int(g[row_bands[vband][0], col_bands[hband][0]]) if vsign and hsign else None
    return {
        "grid": g, "fill": fill, "row_bands": row_bands, "col_bands": col_bands,
        "h_colours": h_colours, "v_colours": v_colours, "priority": priority,
        "row_cells": row_cells, "col_cells": col_cells,
        "marked": (i, j), "box": (r0, r1, c0, c1), "crop": crop,
        "origin": (int(a0), int(b0)),
        "block": crop[a0:a1 + 1, b0:b1 + 1].copy(),
        "vsign": vsign, "hsign": hsign, "vband": vband, "hband": hband,
        "t_r": t_r, "t_c": t_c, "colour_r": colour_r, "colour_c": colour_c,
        "junction": junction,
    }


def _dock(extent, size, start, sign, thickness):
    """Docked start of the block and the copied wall range along one axis."""
    if sign is None:
        return start, None
    if size + thickness > extent:
        raise ValueError("no room for the copied wall beyond the docked block")
    if sign == "+":
        docked = extent - size
        return docked, (docked - thickness, docked)
    return 0, (size, size + thickness)


def _render(info):
    height, width = info["crop"].shape
    block = info["block"]
    top, rows = _dock(height, block.shape[0], info["origin"][0], info["vsign"], info["t_r"])
    left, cols = _dock(width, block.shape[1], info["origin"][1], info["hsign"], info["t_c"])
    out = np.full((height, width), info["fill"], dtype=int)
    has_rows = rows is not None and rows[1] > rows[0]
    has_cols = cols is not None and cols[1] > cols[0]
    if has_rows:
        out[rows[0]:rows[1], :] = info["colour_r"]
    if has_cols:
        out[:, cols[0]:cols[1]] = info["colour_c"]
    if has_rows and has_cols:
        out[rows[0]:rows[1], cols[0]:cols[1]] = info["junction"]
    mask = block != info["fill"]
    region = out[top:top + block.shape[0], left:left + block.shape[1]]
    region[mask] = block[mask]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Crop the unique marked compartment, dock its content, copy its walls."""
    return _render(_analyze(input_grid))


# ---------------------------------------------------------------------------
# Content construction (blocks use -1 for holes inside the bounding box)
# ---------------------------------------------------------------------------
def _pick(rng, low, high):
    return int(rng.integers(low, max(low, high) + 1))


def _choice(rng, items, p=None):
    return items[int(rng.choice(len(items), p=p))]


def _frontier(mask):
    near = np.zeros_like(mask)
    near[1:, :] |= mask[:-1, :]
    near[:-1, :] |= mask[1:, :]
    near[:, 1:] |= mask[:, :-1]
    near[:, :-1] |= mask[:, 1:]
    return np.argwhere(near & ~mask)


def _spanning_mask(rng, h, w, density):
    """A 4-connected mask touching all four sides of its h x w box."""
    if h == 1 or w == 1:
        return np.ones((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=bool)
    hub = (int(rng.integers(h)), int(rng.integers(w)))
    mask[hub] = True
    targets = [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
               (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
    for tr, tc in targets:
        r, c = hub
        while (r, c) != (tr, tc):
            moves = []
            if r != tr:
                moves.append((r + (1 if tr > r else -1), c))
            if c != tc:
                moves.append((r, c + (1 if tc > c else -1)))
            r, c = moves[int(rng.integers(len(moves)))]
            mask[r, c] = True
    target = max(int(mask.sum()), int(round(density * h * w)))
    while int(mask.sum()) < target:
        options = _frontier(mask)
        r, c = options[int(rng.integers(len(options)))]
        mask[r, c] = True
    return mask


def _voronoi(rng, mask, colours):
    cells = np.argwhere(mask)
    k = min(len(colours), len(cells), int(rng.integers(2, 5)))
    seeds = cells[rng.choice(len(cells), size=k, replace=False)]
    block = np.full(mask.shape, -1, dtype=int)
    for r, c in cells:
        distance = np.abs(seeds[:, 0] - r) + np.abs(seeds[:, 1] - c)
        block[r, c] = colours[int(np.argmin(distance))]
    return block


def _size(rng, low, high, bias, exact=False):
    """A dimension in [low, high]; bias 0 small, 1 medium, 2 large."""
    if high < low:
        return None
    if exact:
        return high
    if bias == 0:
        return _pick(rng, low, min(high, low + 2))
    if bias == 1:
        return _pick(rng, low, max(low, min(high, low + max(2, (high - low) // 2 + 1))))
    return _pick(rng, max(low, high - max(1, (high - low) // 3)), high)


def _crop_block(block):
    cells = np.argwhere(block >= 0)
    (a0, b0), (a1, b1) = cells.min(axis=0), cells.max(axis=0)
    return block[a0:a1 + 1, b0:b1 + 1]


def _orient(rng, block, hmax, wmax):
    """Random rotation/reflection that still fits the h x w limits."""
    options = []
    for k in range(4):
        for flip in (False, True):
            cand = np.rot90(block, k)
            if flip:
                cand = cand[:, ::-1]
            if cand.shape[0] <= hmax and cand.shape[1] <= wmax:
                options.append(cand)
    if not options:
        return None
    return np.ascontiguousarray(options[int(rng.integers(len(options)))])


def _st_single(rng, hmax, wmax, colours, bias, exact):
    return np.array([[colours[0]]])


def _st_bar(rng, hmax, wmax, colours, bias, exact):
    options = [o for o, m in (("v", hmax), ("h", wmax)) if m >= 2]
    if not options:
        return None
    vertical = options[int(rng.integers(len(options)))] == "v"
    limit = hmax if vertical else wmax
    length = _size(rng, 2, limit, bias, exact[0] if vertical else exact[1])
    line = np.full(length, colours[0])
    if len(colours) >= 2 and rng.random() < 0.5:
        if rng.random() < 0.5:
            line[1::2] = colours[1]
        else:
            line[_pick(rng, 1, length - 1):] = colours[1]
    across = wmax if vertical else hmax
    width = 2 if bias == 2 and across >= 2 and rng.random() < 0.3 else 1
    if vertical:
        return np.repeat(line.reshape(-1, 1), width, axis=1)
    return np.repeat(line.reshape(1, -1), width, axis=0)


def _st_blob(rng, hmax, wmax, colours, bias, exact):
    if hmax < 2 or wmax < 2:
        return None
    h = _size(rng, 2, hmax, bias, exact[0])
    w = _size(rng, 2, wmax, bias, exact[1])
    mask = _spanning_mask(rng, h, w, float(rng.uniform(0.3, 0.85)))
    if len(colours) >= 2 and rng.random() < 0.5:
        return _voronoi(rng, mask, colours)
    return np.where(mask, colours[0], -1)


def _st_multi(rng, hmax, wmax, colours, bias, exact):
    if hmax < 2 or wmax < 2 or len(colours) < 2:
        return None
    h = _size(rng, 2, hmax, bias, exact[0])
    w = _size(rng, 2, wmax, bias, exact[1])
    mask = _spanning_mask(rng, h, w, float(rng.uniform(0.35, 0.9)))
    return _voronoi(rng, mask, colours)


def _st_hollow(rng, hmax, wmax, colours, bias, exact):
    if hmax < 3 or wmax < 3:
        return None
    h = _size(rng, 3, hmax, bias, exact[0])
    w = _size(rng, 3, wmax, bias, exact[1])
    k = 2 if min(h, w) >= 6 and rng.random() < 0.25 else 1
    block = np.full((h, w), -1, dtype=int)
    block[:k, :] = block[-k:, :] = block[:, :k] = block[:, -k:] = colours[0]
    if len(colours) >= 2 and rng.random() < 0.6:
        if rng.random() < 0.5:
            block[:k, :] = block[-k:, :] = colours[1]
        else:
            block[:, :k] = block[:, -k:] = colours[1]
    if rng.random() < 0.35:
        side = int(rng.integers(4))
        if side == 0:
            block[:k, _pick(rng, k, w - k - 1)] = -1
        elif side == 1:
            block[-k:, _pick(rng, k, w - k - 1)] = -1
        elif side == 2:
            block[_pick(rng, k, h - k - 1), :k] = -1
        else:
            block[_pick(rng, k, h - k - 1), -k:] = -1
    return block


def _st_pieces(rng, hmax, wmax, colours, bias, exact):
    axes = [a for a, m in ((0, hmax), (1, wmax)) if m >= 3]
    if not axes:
        return None
    axis = axes[int(rng.integers(len(axes)))]
    along_max = hmax if axis == 0 else wmax
    across_max = wmax if axis == 0 else hmax
    k = 3 if along_max >= 5 and rng.random() < 0.4 else 2
    if along_max >= 7 and rng.random() < 0.2:
        k = 4
    along = _size(rng, 2 * k - 1, along_max, bias, exact[axis])
    across = _size(rng, 1, across_max, bias, exact[1 - axis])
    spare = along - (2 * k - 1)
    sizes, gaps = [1] * k, [1] * (k - 1)
    for _ in range(spare):
        if rng.random() < 0.6:
            sizes[int(rng.integers(k))] += 1
        else:
            gaps[int(rng.integers(k - 1))] += 1
    block = np.full((along, across), -1, dtype=int)
    position = 0
    for index in range(k):
        extent = _pick(rng, 1, across)
        offset = _pick(rng, 0, across - extent)
        mask = _spanning_mask(rng, sizes[index], extent, float(rng.uniform(0.4, 0.95)))
        colour = colours[int(rng.integers(len(colours)))] if rng.random() < 0.6 else colours[0]
        block[position:position + sizes[index], offset:offset + extent][mask] = colour
        position += sizes[index] + (gaps[index] if index < k - 1 else 0)
    if axis == 1:
        block = block.T
    return _crop_block(block)


def _st_scatter(rng, hmax, wmax, colours, bias, exact):
    if hmax < 3 or wmax < 3:
        return None
    h = _size(rng, 3, hmax, max(bias, 1), exact[0])
    w = _size(rng, 3, wmax, max(bias, 1), exact[1])
    block = np.full((h, w), -1, dtype=int)
    wanted = _pick(rng, 2, 2 + min(3, (h * w) // 12))
    placed = 0
    for _ in range(wanted):
        for _attempt in range(25):
            ph = _pick(rng, 1, min(2, h))
            pw = _pick(rng, 1, min(3, w))
            r = int(rng.integers(0, h - ph + 1))
            c = int(rng.integers(0, w - pw + 1))
            if np.any(block[max(0, r - 1):r + ph + 1, max(0, c - 1):c + pw + 1] >= 0):
                continue
            mask = _spanning_mask(rng, ph, pw, float(rng.uniform(0.5, 1.0)))
            colour = colours[int(rng.integers(len(colours)))]
            block[r:r + ph, c:c + pw][mask] = colour
            placed += 1
            break
    if placed < 2:
        return None
    return _crop_block(block)


def _st_diagonal(rng, hmax, wmax, colours, bias, exact):
    small = min(hmax, wmax)
    if small < 2:
        return None
    kinds = ["line"]
    if small >= 3:
        kinds += ["x", "vee"]
    if max(hmax, wmax) >= 4:
        kinds.append("stair")
    kind = kinds[int(rng.integers(len(kinds)))]
    if kind == "line":
        k = _size(rng, 2, small, bias)
        block = np.full((k, k), -1, dtype=int)
        for i in range(k):
            block[i, i] = colours[0]
    elif kind == "x":
        k = _size(rng, 3, small, bias)
        block = np.full((k, k), -1, dtype=int)
        for i in range(k):
            block[i, i] = block[i, k - 1 - i] = colours[0]
    elif kind == "vee":
        k = _size(rng, 2, max(2, min(small, (max(hmax, wmax) + 1) // 2)), bias)
        block = np.full((k, 2 * k - 1), -1, dtype=int)
        for i in range(k):
            block[i, i] = block[i, 2 * k - 2 - i] = colours[0]
    else:
        big = max(hmax, wmax)
        step = _pick(rng, 2, min(3, big - 1))
        rows = max(2, min(small, big // step))
        rows = _pick(rng, 2, rows)
        block = np.full((rows, rows * step), -1, dtype=int)
        for i in range(rows):
            block[i, i * step:(i + 1) * step] = colours[0]
    if len(colours) >= 2 and rng.random() < 0.5:
        cells = np.argwhere(block >= 0)
        for n, (r, c) in enumerate(cells):
            if n % 2:
                block[r, c] = colours[1]
    return _orient(rng, block, hmax, wmax)


def _st_glyph(rng, hmax, wmax, colours, bias, exact):
    if hmax < 3 or wmax < 3:
        return None
    h = _size(rng, 3, hmax, bias, exact[0])
    w = _size(rng, 3, wmax, bias, exact[1])
    s = 2 if min(h, w) >= 7 and rng.random() < 0.3 else 1
    kind = _choice(rng, ("plus", "T", "L", "U", "H", "C", "F"))
    mask = np.zeros((h, w), dtype=bool)
    mr = _pick(rng, 1, max(1, h - s - 1)) if h - s >= 2 else 0
    mc = _pick(rng, 1, max(1, w - s - 1)) if w - s >= 2 else 0
    strokes_v = np.zeros_like(mask)
    if kind == "plus":
        mask[mr:mr + s, :] = True
        strokes_v[:, mc:mc + s] = True
    elif kind == "T":
        mask[:s, :] = True
        strokes_v[:, mc:mc + s] = True
    elif kind == "L":
        strokes_v[:, :s] = True
        mask[-s:, :] = True
    elif kind == "U":
        strokes_v[:, :s] = True
        strokes_v[:, -s:] = True
        mask[-s:, :] = True
    elif kind == "H":
        strokes_v[:, :s] = True
        strokes_v[:, -s:] = True
        mask[mr:mr + s, :] = True
    elif kind == "C":
        strokes_v[:, :s] = True
        mask[:s, :] = True
        mask[-s:, :] = True
    else:  # F
        strokes_v[:, :s] = True
        mask[:s, :] = True
        mask[mr:mr + s, :max(s + 1, w - 1)] = True
    second = colours[1] if len(colours) >= 2 and rng.random() < 0.6 else colours[0]
    block = np.where(mask, colours[0], -1)
    block[strokes_v & ~mask] = second
    block[strokes_v & mask] = second if rng.random() < 0.5 else colours[0]
    return _orient(rng, _crop_block(block), hmax, wmax)


def _st_framed(rng, hmax, wmax, colours, bias, exact):
    if hmax < 3 or wmax < 3 or len(colours) < 2:
        return None
    h = _size(rng, 3, hmax, bias, exact[0])
    w = _size(rng, 3, wmax, bias, exact[1])
    block = np.full((h, w), colours[0], dtype=int)
    block[1:-1, 1:-1] = colours[1]
    if len(colours) >= 3 and h >= 5 and w >= 5 and rng.random() < 0.5:
        block[2:-2, 2:-2] = colours[2]
    return block


def _st_pattern(rng, hmax, wmax, colours, bias, exact):
    if hmax < 2 or wmax < 2:
        return None
    h = _size(rng, 2, hmax, bias, exact[0])
    w = _size(rng, 2, wmax, bias, exact[1])
    other = colours[1] if len(colours) >= 2 and rng.random() < 0.6 else -1
    kind = _choice(rng, ("checker", "rows", "cols"))
    period = 2 if kind == "checker" or rng.random() < 0.6 else 3
    rr, cc = np.indices((h, w))
    if kind == "checker":
        on = (rr + cc) % 2 == 0
    elif kind == "rows":
        on = rr % period == 0
    else:
        on = cc % period == 0
    block = np.where(on, colours[0], other)
    if not np.any(block >= 0):
        return None
    return _crop_block(block)


def _st_nested(rng, hmax, wmax, colours, bias, exact):
    if hmax < 5 or wmax < 5:
        return None
    h = _size(rng, 5, hmax, bias, exact[0])
    w = _size(rng, 5, wmax, bias, exact[1])
    block = np.full((h, w), -1, dtype=int)
    block[0, :] = block[-1, :] = block[:, 0] = block[:, -1] = colours[0]
    ih, iw = h - 4, w - 4
    ph, pw = _pick(rng, 1, ih), _pick(rng, 1, iw)
    r, c = 2 + _pick(rng, 0, ih - ph), 2 + _pick(rng, 0, iw - pw)
    inner = colours[1] if len(colours) >= 2 and rng.random() < 0.7 else colours[0]
    block[r:r + ph, c:c + pw] = inner
    if rng.random() < 0.3:
        block[0, _pick(rng, 1, w - 2)] = -1
    return block


STYLE_BUILDERS = {
    "single": _st_single, "bar": _st_bar, "blob": _st_blob, "multi": _st_multi,
    "hollow": _st_hollow, "pieces": _st_pieces, "scatter": _st_scatter,
    "diagonal": _st_diagonal, "glyph": _st_glyph, "framed": _st_framed,
    "pattern": _st_pattern, "nested": _st_nested,
}


def _content(rng, style, hmax, wmax, colours, bias, exact=(False, False)):
    """Return (style, block) with -1 marking holes inside the bounding box."""
    for name in (style, "blob", "bar", "single"):
        block = STYLE_BUILDERS[name](rng, hmax, wmax, colours, bias, exact)
        if block is not None and block.shape[0] <= hmax and block.shape[1] <= wmax \
                and np.any(block >= 0):
            return name, _crop_block(np.asarray(block, dtype=int))
    raise ValueError("no content fits")


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
def _axis_extents(rng, sign, count, thicknesses, marked, narrow_p, equal_p, other_hi=14):
    """Compartment extents along one axis and the index of the marked one."""
    if sign is None:
        return [marked], 0
    index = 0 if sign == "+" else count - 1
    n_other = count - 1
    budget = MAX_SIDE - marked - sum(thicknesses)
    if budget < n_other:
        raise ValueError("axis does not fit")
    if equal_p and rng.random() < equal_p and marked * n_other <= budget:
        others = [marked] * n_other
    else:
        cap = budget // n_other
        if cap < 2 and narrow_p == 0:
            raise ValueError("axis would need narrow compartments")
        others = []
        for _ in range(n_other):
            if cap < 2 or rng.random() < narrow_p:
                others.append(1)
            else:
                others.append(_pick(rng, 2, min(other_hi, cap + int(rng.integers(0, 3)))))
        while sum(others) > budget:
            k = int(np.argmax(others))
            others[k] -= 1
        if min(others) < 1 or (narrow_p == 0 and min(others) < 2):
            raise ValueError("axis does not fit")
    others.insert(index, marked)
    return others, index


def _thicknesses(rng, n_bands, wall_index, spec, axis):
    if n_bands == 0:
        return []
    hi = spec["t_max"]

    def draw():
        if rng.random() >= spec["thick_p"]:
            return 1
        value = _pick(rng, 2, min(3, hi))
        if hi >= 4 and rng.random() < 0.08:
            value = 4
        return value

    mode = spec["thick_mode"]
    if mode == "thin":
        values = [1] * n_bands
    elif mode == "uniform":
        values = [draw()] * n_bands
    else:
        values = [draw() for _ in range(n_bands)]
    forced = spec["wall_t"].get(axis)
    if forced is not None:
        values[wall_index] = forced
        if mode == "uniform":
            values = [forced] * n_bands
    if spec["band_t_diff"] == axis and n_bands >= 2:
        wall = values[wall_index]
        choices = [t for t in range(1, max(2, min(3, hi)) + 1) if t != wall]
        for k in range(n_bands):
            if k != wall_index and values[k] == wall:
                values[k] = choices[int(rng.integers(len(choices)))]
    if spec["uniform_axis"]:
        values = [values[wall_index]] * n_bands
    return [int(v) for v in values]


def _band_colours(rng, n_r, n_c, wall_r, wall_c, spec, palette):
    """Colours of every horizontal and vertical band for one pair."""
    b0, b1, b2 = palette["bands"]
    mode = spec["band_mode"]
    if mode == "single":
        return [b0] * n_r, [b0] * n_c
    if mode == "axis":
        h, v = [b0] * n_r, [b1] * n_c
    else:
        pool = [b0, b1, b2]
        h = [pool[int(rng.integers(3))] for _ in range(n_r)]
        v = [pool[int(rng.integers(3))] for _ in range(n_c)]
    if spec["band_c_diff"] == "r" and n_r >= 2:
        h = [b1] * n_r
        h[wall_r] = b0
    if spec["band_c_diff"] == "c" and n_c >= 2:
        v = [b1] * n_c
        v[wall_c] = b0
    if spec["corner_c_diff"] and n_r and n_c and h[wall_r] == v[wall_c]:
        v[wall_c] = b2 if h[wall_r] != b2 else b1
    if spec["same_walls"] and n_r and n_c:
        v[wall_c] = h[wall_r]
    if spec["uniform_axis_colour"]:
        if n_r:
            h = [h[wall_r]] * n_r
        if n_c:
            v = [v[wall_c]] * n_c
    return h, v


def _gap_options(category, free):
    if category == "flush":
        return [0]
    if category == "near_wall":
        return [g for g in range(1, free + 1) if g < free - g]
    if category == "middle":
        return [g for g in range(1, free) if abs(2 * g - free) <= 1]
    if category == "near_outer":
        return [g for g in range(1, free) if g > free - g]
    return [free]


def _placement(rng, spec, walled, frees):
    """Wall gaps per walled axis; at least one axis moves unless docked."""
    gaps = {}
    for axis in walled:
        free = frees[axis]
        category = spec["place"].get(axis)
        options = _gap_options(category, free) if category else []
        if not options:
            pool = [c for c in PLACEMENTS if not (c == "flush" and spec["no_flush"])]
            weights = np.array([PLACEMENT_P[PLACEMENTS.index(c)] for c in pool])
            for _ in range(8):
                category = _choice(rng, pool, weights / weights.sum())
                options = _gap_options(category, free)
                if options:
                    break
            if not options:
                options = list(range(1, free + 1))
        gaps[axis] = options[int(rng.integers(len(options)))]
    if spec["docked"]:
        return {axis: 0 for axis in walled}
    if all(g == 0 for g in gaps.values()):
        movable = [a for a in walled if spec["place"].get(a) != "flush"] or list(walled)
        axis = movable[int(rng.integers(len(movable)))]
        gaps[axis] = _pick(rng, 1, frees[axis])
    return gaps


def _paint(rows, cols, t_r, t_c, h_colours, v_colours, fill, priority):
    height = sum(rows) + sum(t_r)
    width = sum(cols) + sum(t_c)
    grid = np.full((height, width), fill, dtype=int)

    def layout(extents, thick):
        bands, starts, pos = [], [], 0
        for k, extent in enumerate(extents):
            starts.append(pos)
            pos += extent
            if k < len(thick):
                bands.append((pos, pos + thick[k] - 1))
                pos += thick[k]
        return bands, starts

    row_bands, row_starts = layout(rows, t_r)
    col_bands, col_starts = layout(cols, t_c)

    def horizontal():
        for (a, b), colour in zip(row_bands, h_colours):
            grid[a:b + 1, :] = colour

    def vertical():
        for (a, b), colour in zip(col_bands, v_colours):
            grid[:, a:b + 1] = colour

    if priority == "h":
        vertical()
        horizontal()
    else:
        horizontal()
        vertical()
    return grid, row_bands, col_bands, row_starts, col_starts


def _build_pair(rng, spec, palette):
    kind = spec["kind"]
    vsign, hsign = DOCK[kind]
    n_r = spec["n_rows"] if vsign else 1
    n_c = spec["n_cols"] if hsign else 1
    wall_r = (0 if vsign == "+" else n_r - 2) if vsign else None
    wall_c = (0 if hsign == "+" else n_c - 2) if hsign else None
    fill = palette["fill"]
    if spec["classic"]:
        lo, hi = spec["walled"]
        q = _pick(rng, max(3, lo), max(3, lo, min(hi, 14)))
        rows, cols, t_r, t_c = [q, q], [q, q], [1], [1]
        mi = 0 if vsign == "+" else 1
        mj = 0 if hsign == "+" else 1
        h_cols, v_cols = [palette["bands"][0]], [palette["bands"][0]]
    else:
        t_r = _thicknesses(rng, n_r - 1, wall_r, spec, "r")
        t_c = _thicknesses(rng, n_c - 1, wall_c, spec, "c")
        h_cols, v_cols = _band_colours(rng, n_r - 1, n_c - 1, wall_r, wall_c, spec, palette)

        need = STYLE_MIN[spec["style"]]

        def marked_extent(sign, thick, walled_range, unwalled_range):
            if sign is None:
                lo, hi = unwalled_range
                lo = max(lo, need)
                return _pick(rng, lo, max(lo, min(MAX_SIDE, hi)))
            lo, hi = walled_range
            wall_t = thick[0 if sign == "+" else len(thick) - 1]
            low = max(lo, wall_t + 2, need + wall_t + (0 if spec["tight"] else 1))
            return _pick(rng, low, max(low, min(hi, MAX_SIDE - 2)))

        hc0 = marked_extent(vsign, t_r, spec["walled"], spec["unwalled"])
        wc0 = marked_extent(hsign, t_c, spec["walled"], spec["unwalled"])
        rows, mi = _axis_extents(rng, vsign, n_r, t_r, hc0, spec["narrow_p"], spec["equal_p"], spec["other_hi"])
        cols, mj = _axis_extents(rng, hsign, n_c, t_c, wc0, spec["narrow_p"], spec["equal_p"], spec["other_hi"])
    hc, wc = rows[mi], cols[mj]
    tw_r = t_r[wall_r] if vsign else 0
    tw_c = t_c[wall_c] if hsign else 0
    hmax = hc - tw_r if vsign else hc
    wmax = wc - tw_c if hsign else wc
    if vsign and not spec["tight"] and hmax >= 3:
        hmax -= 1
    if hsign and not spec["tight"] and wmax >= 3:
        wmax -= 1
    span = spec["full_span"] and spec["style"] in FULL_SPAN_STYLES
    exact = [span and vsign is None, span and hsign is None]
    if spec["tight"]:
        # A tight fit: the box fills the room left by the copied wall.
        tight_axes = [k for k, sign in ((0, vsign), (1, hsign)) if sign]
        exact[tight_axes[int(rng.integers(len(tight_axes)))]] = True
    exact = tuple(exact)
    style, block = _content(rng, spec["style"], hmax, wmax, palette["content"], spec["bias"], exact)
    hs, ws = block.shape
    walled = [a for a, s in (("r", vsign), ("c", hsign)) if s]
    frees = {"r": hc - hs, "c": wc - ws}
    gaps = _placement(rng, spec, walled, frees)

    def offset(axis, sign, extent, size):
        if sign is None:
            free = extent - size
            if free and rng.random() < 0.25:
                return 0 if rng.random() < 0.5 else free
            return _pick(rng, 0, free)
        gap = gaps[axis]
        return extent - size - gap if sign == "+" else gap

    off_r = offset("r", vsign, hc, hs)
    off_c = offset("c", hsign, wc, ws)
    grid, row_bands, col_bands, row_starts, col_starts = _paint(
        rows, cols, t_r, t_c, h_cols, v_cols, fill, spec["priority"])
    if grid.shape[0] > MAX_SIDE or grid.shape[1] > MAX_SIDE:
        raise ValueError("grid too large")
    mask = block >= 0
    r0, c0 = row_starts[mi] + off_r, col_starts[mj] + off_c
    grid[r0:r0 + hs, c0:c0 + ws][mask] = block[mask]

    # Output built directly from the construction (independent of solve).
    output = np.full((hc, wc), fill, dtype=int)
    colour_r = h_cols[wall_r] if vsign else None
    colour_c = v_cols[wall_c] if hsign else None
    if vsign == "+":
        top = hc - hs
        rows_copy = (top - tw_r, top)
    elif vsign == "-":
        top, rows_copy = 0, (hs, hs + tw_r)
    else:
        top, rows_copy = off_r, None
    if hsign == "+":
        left = wc - ws
        cols_copy = (left - tw_c, left)
    elif hsign == "-":
        left, cols_copy = 0, (ws, ws + tw_c)
    else:
        left, cols_copy = off_c, None
    if rows_copy:
        output[rows_copy[0]:rows_copy[1], :] = colour_r
    if cols_copy:
        output[:, cols_copy[0]:cols_copy[1]] = colour_c
    if rows_copy and cols_copy:
        junction = colour_r if (colour_r == colour_c or spec["priority"] == "h") else colour_c
        output[rows_copy[0]:rows_copy[1], cols_copy[0]:cols_copy[1]] = junction
    output[top:top + hs, left:left + ws][mask] = block[mask]

    info = _analyze(grid)
    expected_box = (row_starts[mi], row_starts[mi] + hc - 1, col_starts[mj], col_starts[mj] + wc - 1)
    if (info["fill"] != fill or info["row_bands"] != row_bands or info["col_bands"] != col_bands
            or info["box"] != expected_box):
        raise ValueError("input does not read back as constructed")
    if not np.array_equal(_render(info), output):
        raise ValueError("construction disagrees with the rule")
    _check_input(info)
    return {"input": grid, "output": output, "style": style, "info": info}


# ---------------------------------------------------------------------------
# Puzzle planning
# ---------------------------------------------------------------------------
def _knobs(rng, tier, regime, n_train):
    strips_ok = n_train >= 3
    if tier == "minimal":
        return {"stable": True, "multi_band": False, "thick_p": 0.0, "per_band": 0.0,
                "dense_p": 0.0, "narrow_p": 0.0, "strip_p": 0.0, "styles": SIMPLE_STYLES,
                "multi_p": 0.0, "classic": bool(rng.random() < 0.6), "docked_p": 0.0,
                "equal_p": 0.4, "full_span_p": 0.0}
    if tier == "maximal":
        knobs = {"stable": False, "multi_band": True, "thick_p": 0.6, "per_band": 0.7,
                 "dense_p": 0.5, "narrow_p": 0.15, "strip_p": 0.3 if strips_ok else 0.0,
                 "styles": RICH_STYLES, "multi_p": 0.7, "classic": False,
                 "docked_p": 0.35 if n_train >= 3 else 0.0, "equal_p": 0.15, "full_span_p": 0.3}
    else:
        rich = bool(rng.random() < 0.78)
        knobs = {
            "stable": bool(rng.random() < 0.2),
            "multi_band": bool(rng.random() < 0.3),
            "thick_p": float(_choice(rng, (0.0, 0.3, 0.5, 0.8), (0.15, 0.3, 0.35, 0.2))),
            "per_band": float(_choice(rng, (0.0, 0.5, 0.8), (0.3, 0.4, 0.3))),
            "dense_p": float(_choice(rng, (0.0, 0.25, 0.5), (0.2, 0.5, 0.3))),
            "narrow_p": float(_choice(rng, (0.0, 0.1), (0.6, 0.4))),
            "strip_p": float(_choice(rng, (0.0, 0.2, 0.35), (0.3, 0.4, 0.3))) if strips_ok else 0.0,
            "styles": RICH_STYLES if rich else SIMPLE_STYLES,
            "multi_p": float(_choice(rng, (0.35, 0.55, 0.8))) if rich else 0.2,
            "classic": bool(rng.random() < 0.25),
            "docked_p": 0.1 if n_train >= 3 else 0.0,
            "equal_p": 0.25, "full_span_p": 0.15,
        }
    if regime == "compose_axes":
        knobs["multi_band"] = False
        knobs["classic"] = False
    if regime == "thickness_up":
        knobs["thick_p"] = max(knobs["thick_p"], 0.4)
    if regime == "layout_shift":
        knobs["dense_p"] = min(knobs["dense_p"], 0.15)
    if regime == "strip_test":
        knobs["strip_p"] = max(knobs["strip_p"], 0.2)
    return knobs


def _train_kinds(rng, n_train, regime, held, knobs):
    if regime == "compose_axes":
        kinds = list(EDGES)
        while len(kinds) < n_train:
            kinds.append(EDGES[int(rng.integers(4))])
        return [kinds[i] for i in rng.permutation(n_train)]
    corners = [c for c in CORNERS if c != held]
    patterns, weights = ["opposite"], [0.4]
    if n_train >= 3:
        patterns.append("three")
        weights.append(0.3)
        if knobs["strip_p"] > 0:
            patterns.append("corner_strips")
            weights.append(0.3)
    weights = np.array(weights) / sum(weights)
    pattern = _choice(rng, patterns, weights)
    if regime == "strip_test":
        pattern = "corner_strips"  # strips must be demonstrated before strip tests
    if pattern == "opposite":
        pairs = [c for c in corners if OPPOSITE[c] in corners]
        first = pairs[int(rng.integers(len(pairs)))]
        kinds = [first, OPPOSITE[first]]
    elif pattern == "three":
        kinds = [corners[i] for i in rng.permutation(len(corners))[:3]]
    else:
        first = corners[int(rng.integers(len(corners)))]
        kinds = [first, *COMPLEMENT[first]]
    while len(kinds) < n_train:
        if knobs["strip_p"] > 0 and rng.random() < knobs["strip_p"]:
            options = list(EDGES)
        else:
            options = corners
        unused = [k for k in options if k not in kinds]
        if unused and rng.random() < 0.7:
            options = unused
        kinds.append(options[int(rng.integers(len(options)))])
    return [kinds[i] for i in rng.permutation(n_train)]


def _sizes(regime, split, primary, tier):
    if tier == "minimal":
        if primary and regime == "scale_up":
            return (((3, 6),), (1.0,), ((4, 10),), (1.0,)) if split == "train" else \
                (((9, 13),), (1.0,), ((10, 18),), (1.0,))
        return ((3, 9),), (1.0,), ((4, 14),), (1.0,)
    if primary and regime == "scale_up":
        return (((3, 7),), (1.0,), ((3, 12),), (1.0,)) if split == "train" else \
            (((11, 17),), (1.0,), ((12, 28),), (1.0,))
    if primary and regime == "scale_down":
        return (((8, 15),), (1.0,), ((8, 24),), (1.0,)) if split == "train" else \
            (((4, 7),), (1.0,), ((4, 10),), (1.0,))
    return WALLED_BINS, WALLED_P, UNWALLED_BINS, UNWALLED_P


def _base_spec(rng, plan, kind, split, primary):
    knobs, regime, tier = plan["knobs"], plan["regime"], plan["tier"]
    wb, wp, ub, up = _sizes(regime, split, primary, tier)
    styles = knobs["styles"]
    weights = np.array([0.5 if s == "single" else 1.0 for s in styles])
    style = _choice(rng, styles, weights / weights.sum())
    vsign, hsign = DOCK[kind]
    thick_mode = "thin"
    if knobs["thick_p"] > 0:
        thick_mode = "per_band" if rng.random() < knobs["per_band"] else "uniform"

    def count():
        if knobs["dense_p"] and rng.random() < knobs["dense_p"]:
            return 4 if rng.random() < 0.3 else 3
        return 2

    multi = rng.random() < knobs["multi_p"] or style in MULTI_COLOUR_STYLES
    n_colours = 1 if not multi else (4 if rng.random() < 0.15 else _pick(rng, 2, 3))
    band_mode = "single"
    if knobs["multi_band"]:
        band_mode = _choice(rng, ("single", "axis", "band"), (0.3, 0.45, 0.25))
    return {
        "kind": kind, "split": split, "primary": primary,
        "n_rows": count(), "n_cols": count(),
        "thick_mode": thick_mode, "thick_p": knobs["thick_p"], "t_max": 3 + (1 if tier == "maximal" else 0),
        "wall_t": {}, "band_t_diff": None, "uniform_axis": False,
        "band_mode": band_mode, "band_c_diff": None, "corner_c_diff": False,
        "same_walls": False, "uniform_axis_colour": False,
        "walled": wb[int(rng.choice(len(wb), p=wp))], "unwalled": ub[int(rng.choice(len(ub), p=up))],
        "narrow_p": knobs["narrow_p"], "equal_p": knobs["equal_p"],
        "other_hi": int(_choice(rng, (4, 8, 14), (0.25, 0.4, 0.35))),
        "style": style, "bias": int(rng.choice(3, p=(0.3, 0.4, 0.3))), "n_colours": n_colours,
        "place": {}, "no_flush": False, "docked": False,
        "tight": bool(rng.random() < 0.12), "classic": False,
        "full_span": bool((vsign is None or hsign is None) and style in FULL_SPAN_STYLES
                          and rng.random() < knobs["full_span_p"] * 2),
        "priority": plan["priority"], "single_component": False,
    }


def _set_style(rng, spec, style):
    spec["style"] = style
    if style in MULTI_COLOUR_STYLES and spec["n_colours"] < 2:
        spec["n_colours"] = _pick(rng, 2, 3)


def _plan(rng, n_train, n_test, force_tier=None):
    tier = force_tier or _choice(rng, TIERS, TIER_P)
    if tier == "minimal":
        regime = _choice(rng, ("unseen_corner", "shared", "scale_up"), (0.4, 0.4, 0.2))
    else:
        while True:
            regime = _choice(rng, REGIMES, REGIME_P)
            if regime == "compose_axes" and n_train < 4:
                continue
            if regime == "strip_test" and n_train < 3:
                continue
            break
    knobs = _knobs(rng, tier, regime, n_train)
    plan = {"tier": tier, "regime": regime, "knobs": knobs,
            "priority": "h" if rng.random() < 0.5 else "v",
            "base": [int(v) for v in rng.permutation(10)]}
    held = CORNERS[int(rng.integers(4))] if regime == "unseen_corner" else None
    plan["held"] = held
    kinds = _train_kinds(rng, n_train, regime, held, knobs)
    specs = [_base_spec(rng, plan, kind, "train", True) for kind in kinds]

    if regime == "thickness_up":
        for spec in specs:
            spec["t_max"] = 2
    if regime == "layout_shift":
        for spec in specs:
            spec["n_rows"] = spec["n_rows"] if spec["n_rows"] <= 3 else 3
            spec["n_cols"] = spec["n_cols"] if spec["n_cols"] <= 3 else 3

    # Classic anchor: a centred 2x2 thin cross with simple content.
    anchor = None
    if knobs["classic"]:
        corner_idx = [i for i, k in enumerate(kinds) if k in CORNERS]
        if corner_idx:
            anchor = corner_idx[0] if rng.random() < 0.5 else corner_idx[int(rng.integers(len(corner_idx)))]
            specs[anchor].update(classic=True, style=_choice(rng, ("single", "bar", "blob")),
                                 bias=int(rng.integers(2)), n_colours=1, band_mode="single",
                                 n_rows=2, n_cols=2, full_span=False)
    free = [i for i in range(n_train) if i != anchor]
    corner_free = [i for i in free if kinds[i] in CORNERS]

    def pick(indices):
        return indices[int(rng.integers(len(indices)))]

    def walled_axis(i):
        axes = [a for a, s in (("r", DOCK[kinds[i]][0]), ("c", DOCK[kinds[i]][1])) if s]
        return axes[int(rng.integers(len(axes)))]

    # Rare already-docked pair (copies drawn, nothing moves).
    docked = None
    if knobs["docked_p"] and len(free) >= 3 and rng.random() < knobs["docked_p"]:
        docked = pick(free)
        specs[docked]["docked"] = True
    movers = [i for i in free if i != docked]
    corner_movers = [i for i in corner_free if i != docked]

    # Demonstration duties (all checked again on the returned grids).
    irregular = [s for s in IRREGULAR if s in knobs["styles"]]
    _set_style(rng, specs[pick(movers)], _choice(rng, irregular))
    flush_axis = None
    if corner_movers:
        i = pick(corner_movers)
        flush_axis = (i, _choice(rng, ("r", "c")))
        specs[i]["place"][flush_axis[1]] = "flush"
    else:
        i = pick(movers)
        axis = walled_axis(i)
        specs[i]["place"][axis] = "near_wall"
        specs[i]["walled"] = (max(6, specs[i]["walled"][0]), max(9, specs[i]["walled"][1]))
    options = [(i, a) for i in movers for a, s in (("r", DOCK[kinds[i]][0]), ("c", DOCK[kinds[i]][1]))
               if s and (i, a) != flush_axis and a not in specs[i]["place"]]
    if options:
        i, axis = options[int(rng.integers(len(options)))]
        specs[i]["place"][axis] = _choice(rng, ("near_outer", "outer_edge"))

    if knobs["thick_p"] > 0 or regime == "thickness_up":
        i = pick(free)
        axis = walled_axis(i)
        specs[i]["wall_t"][axis] = 2 if regime == "thickness_up" else _pick(rng, 2, 3)
        if regime == "thickness_up":
            others = [j for j in free if j != i]
            if others:
                j = pick(others)
                axis_j = walled_axis(j)
                specs[j]["wall_t"][axis_j] = 1
        if knobs["per_band"] > 0 and knobs["dense_p"] > 0:
            i = pick(free)
            axis = walled_axis(i)
            specs[i]["n_rows" if axis == "r" else "n_cols"] = max(3, specs[i]["n_rows" if axis == "r" else "n_cols"])
            specs[i]["thick_mode"] = "per_band"
            specs[i]["band_t_diff"] = axis
    if knobs["multi_band"]:
        if corner_free:
            i = pick(corner_free)
            specs[i]["band_mode"] = "axis" if rng.random() < 0.7 else "band"
            specs[i]["corner_c_diff"] = True
        if knobs["dense_p"] > 0:
            i = pick(free)
            axis = walled_axis(i)
            specs[i]["n_rows" if axis == "r" else "n_cols"] = max(3, specs[i]["n_rows" if axis == "r" else "n_cols"])
            specs[i]["band_mode"] = "band"
            specs[i]["band_c_diff"] = axis
    if any(s in knobs["styles"] for s in PIECE_STYLES):
        _set_style(rng, specs[pick(movers)], _choice(rng, PIECE_STYLES))
    if regime == "layout_shift" or knobs["dense_p"] > 0:
        i = pick(free)
        axis = walled_axis(i)
        specs[i]["n_rows" if axis == "r" else "n_cols"] = 3
    if docked is not None:
        specs[docked]["place"] = {}
    plan.update(train_specs=specs, train_kinds=kinds, anchor=anchor, docked=docked,
                n_train=n_train, n_test=n_test)
    return plan


def _palette(rng, plan, spec):
    if plan["knobs"]["stable"]:
        base = plan["base"]
        tail = [int(c) for c in rng.permutation(base[4:])]
        order = base[:4] + tail
    else:
        order = [int(v) for v in rng.permutation(10)]
    forced = spec.get("force_band_colour")
    if forced is not None and forced != order[1]:
        rest = [c for c in order if c != forced]
        order = [rest[0], forced] + rest[1:]
    fill, bands, rest = order[0], order[1:4], order[4:]
    content = [int(c) for c in rest][:spec["n_colours"]]
    return {"fill": int(fill), "bands": [int(b) for b in bands], "content": content}


def _demonstrated(infos):
    feats = [_features(info) for info in infos]
    return {
        "strips": any(not f["two_d"] for f in feats),
        "corners": any(f["two_d"] for f in feats),
        "strip_kinds": sorted({f["kind"] for f in feats if not f["two_d"]}),
        "kinds": sorted({f["kind"] for f in feats}),
        "wall_t": sorted({t for f in feats for t in f["wall_t"]}),
        "wall_vs_band_t": any(f["wall_vs_band_t"] for f in feats),
        "wall_colours": sorted({c for f in feats for c in f["wall_colours"]}),
        "two_colour_corner": any(f["two_colour_corner"] for f in feats),
        "wall_vs_band_colour": any(f["wall_vs_band_colour"] for f in feats),
        "dense": max(f["max_count"] for f in feats),
        "piecewise": any(f["piecewise"] for f in feats),
        "flush": any(f["flush"] for f in feats),
    }


def _test_specs(rng, plan, demo):
    regime, n_test = plan["regime"], plan["n_test"]
    knobs = plan["knobs"]
    train_kinds = plan["train_kinds"]
    allowed = list(CORNERS) + (list(EDGES) if demo["strips"] else [])
    specs = []
    for k in range(n_test):
        primary = k == 0 or rng.random() < 0.5
        if regime == "unseen_corner" and k == 0:
            kind = plan["held"]
        elif regime == "compose_axes" and primary:
            kind = CORNERS[int(rng.integers(4))]
        elif regime == "strip_test" and primary and demo["strips"]:
            unseen = [e for e in EDGES if e not in demo["strip_kinds"]]
            pool = unseen if (k == 0 and unseen) else list(EDGES)
            kind = pool[int(rng.integers(len(pool)))]
        elif regime == "shared" or not primary:
            kind = train_kinds[int(rng.integers(len(train_kinds)))]
        else:
            kind = allowed[int(rng.integers(len(allowed)))]
        spec = _base_spec(rng, plan, kind, "test", primary)
        if spec["style"] in ("single", "bar") and rng.random() < 0.7:
            rich = [s for s in IRREGULAR if s in knobs["styles"]]
            _set_style(rng, spec, _choice(rng, rich))
        if regime == "scale_down" and primary and STYLE_MIN[spec["style"]] > 2:
            _set_style(rng, spec, _choice(rng, [s for s in SMALL_STYLES if s in knobs["styles"]]))
        # Layout: never denser than shown, except layout_shift's single step.
        limit = 2 if demo["dense"] < 3 else 4
        if regime == "layout_shift" and primary:
            spec["n_rows"], spec["n_cols"] = _choice(
                rng, ((3, 2), (2, 3), (3, 3), (4, 2), (2, 4), (4, 3), (3, 4), (4, 4)))
            if kind in EDGES:  # a strip's only divided axis carries the shift
                key = "n_rows" if DOCK[kind][0] else "n_cols"
                spec[key] = 3 if rng.random() < 0.6 else 4
        spec["n_rows"] = min(spec["n_rows"], limit)
        spec["n_cols"] = min(spec["n_cols"], limit)
        # Thickness: constant if training shows one wall thickness.
        if len(demo["wall_t"]) == 1:
            spec["thick_mode"] = "uniform"
            spec["wall_t"] = {"r": demo["wall_t"][0], "c": demo["wall_t"][0]}
            spec["uniform_axis"] = True
        else:
            spec["t_max"] = max(demo["wall_t"]) if regime != "thickness_up" else spec["t_max"]
            if not demo["wall_vs_band_t"]:
                spec["uniform_axis"] = True
        if regime == "thickness_up" and primary:
            top = max(demo["wall_t"]) + 1
            value = top if rng.random() < 0.7 else min(4, top + 1)
            spec["wall_t"] = {"r": value, "c": value}
            spec["t_max"] = max(3, value)
            spec["uniform_axis"] = True
        # Colours: only what the training pairs demonstrate.
        if len(demo["wall_colours"]) == 1:
            spec["band_mode"] = "single"
            spec["force_band_colour"] = demo["wall_colours"][0]
        if spec["band_mode"] != "single":
            if kind in CORNERS and not demo["two_colour_corner"]:
                spec["same_walls"] = True
            if not demo["wall_vs_band_colour"]:
                spec["uniform_axis_colour"] = True
        if not demo["piecewise"]:
            spec["single_component"] = True
            if spec["style"] in PIECE_STYLES or spec["style"] == "pattern":
                _set_style(rng, spec, _choice(rng, ("blob", "hollow", "glyph", "diagonal")
                                              if knobs["styles"] == RICH_STYLES else ("blob", "hollow")))
        if not demo["flush"]:
            spec["no_flush"] = True
        specs.append(spec)
    return specs


def _build_with_retries(rng, plan, spec, attempts=30):
    for attempt in range(attempts):
        if attempt == attempts // 2 and spec["full_span"]:
            spec = dict(spec, full_span=False)  # best effort only
        palette = _palette(rng, plan, spec)
        try:
            built = _build_pair(rng, spec, palette)
        except (ValueError, AssertionError):
            continue
        if spec["single_component"] and _components(built["info"]["block"] != built["info"]["fill"])[1] > 1:
            continue
        return built
    raise ValueError("pair construction failed")


def _attempt(rng, n_train, n_test, force_tier=None):
    plan = _plan(rng, n_train, n_test, force_tier)
    built = [_build_with_retries(rng, plan, spec) for spec in plan["train_specs"]]
    demo = _demonstrated([b["info"] for b in built])
    tests = [_build_with_retries(rng, plan, spec) for spec in _test_specs(rng, plan, demo)]
    pairs = [{"input": b["input"].tolist(), "output": b["output"].tolist()} for b in built + tests]
    puzzle = {"train": pairs[:n_train], "test": pairs[n_train:]}
    validate_generated_puzzle(puzzle)
    plan["styles"] = [b["style"] for b in built + tests]
    return puzzle, plan


def _generate(seed, num_train, num_test):
    n_train, n_test = _resolve_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for index in range(100):
        try:
            return _attempt(rng, n_train, n_test, "minimal" if index >= 70 else None)
        except (ValueError, AssertionError):
            continue
    raise RuntimeError(f"could not generate a valid puzzle for seed {seed}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    puzzle, _ = _generate(seed, num_train, num_test)
    return puzzle


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _check_input(info):
    g = info["grid"]
    height, width = g.shape
    assert 1 <= height <= MAX_SIDE and 1 <= width <= MAX_SIDE, "grid size out of range"
    for bands, n in ((info["row_bands"], height), (info["col_bands"], width)):
        assert len(bands) <= 3, "at most four compartments per axis"
        assert all(0 < a and b < n - 1 for a, b in bands), "divider band on the canvas edge"
        assert all(1 <= b - a + 1 <= 4 for a, b in bands), "band thickness must be 1-4"
    block = info["block"]
    content = set(int(v) for v in np.unique(block)) - {info["fill"]}
    assert 1 <= len(content) <= 4, "content uses one to four colours"
    return True


def _gaps(info):
    """(axis, wall gap, outer gap) for every walled axis."""
    height_c, width_c = info["crop"].shape
    block = info["block"]
    out = []
    for axis, sign, extent, size, start in (("r", info["vsign"], height_c, block.shape[0], info["origin"][0]),
                                            ("c", info["hsign"], width_c, block.shape[1], info["origin"][1])):
        if sign is None:
            continue
        wall = extent - start - size if sign == "+" else start
        outer = start if sign == "+" else extent - start - size
        out.append((axis, wall, outer))
    return out


def _content_canvas(info):
    """Content alone (no copied walls) at its rigidly docked position."""
    block = info["block"]
    height, width = info["crop"].shape
    top, _ = _dock(height, block.shape[0], info["origin"][0], info["vsign"], info["t_r"])
    left, _ = _dock(width, block.shape[1], info["origin"][1], info["hsign"], info["t_c"])
    out = np.full((height, width), info["fill"], dtype=int)
    mask = block != info["fill"]
    out[top:top + block.shape[0], left:left + block.shape[1]][mask] = block[mask]
    return out


def _gravity(info, vertical_first):
    canvas = info["crop"].copy()
    fill = info["fill"]
    steps = [(0, info["vsign"]), (1, info["hsign"])]
    if not vertical_first:
        steps.reverse()
    for axis, sign in steps:
        if sign is None:
            continue
        lines = canvas if axis == 1 else canvas.T
        result = np.full_like(lines, fill)
        for k, line in enumerate(lines):
            cells = line[line != fill]
            if sign == "+":
                result[k, len(line) - len(cells):] = cells
            else:
                result[k, :len(cells)] = cells
        canvas = result if axis == 1 else result.T
    return canvas


def _components(mask):
    labels = np.zeros(mask.shape, dtype=int)
    count = 0
    for r, c in np.argwhere(mask):
        if labels[r, c]:
            continue
        count += 1
        stack = [(r, c)]
        labels[r, c] = count
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                            and mask[ny, nx] and not labels[ny, nx]):
                        labels[ny, nx] = count
                        stack.append((ny, nx))
    return labels, count


def _band_list(info, axis):
    bands = info["row_bands"] if axis == "r" else info["col_bands"]
    colours = info["h_colours"] if axis == "r" else info["v_colours"]
    return [(b - a + 1, c) for (a, b), c in zip(bands, colours)]


def _features(info):
    gaps = _gaps(info)
    rigid = _content_canvas(info)
    gravity = all(not np.array_equal(_gravity(info, order), rigid) for order in (True, False))
    _, pieces = _components(info["block"] != info["fill"])
    walls, wall_t, wall_colours = [], [], []
    vs_t = vs_c = False
    for axis, sign, index in (("r", info["vsign"], info["vband"]), ("c", info["hsign"], info["hband"])):
        if sign is None:
            continue
        bands = _band_list(info, axis)
        t, colour = bands[index]
        wall_t.append(t)
        wall_colours.append(colour)
        others = [b for k, b in enumerate(bands) if k != index]
        vs_t |= any(o[0] != t for o in others)
        vs_c |= any(o[1] != colour for o in others)
        walls.append(axis)
    two_d = info["vsign"] is not None and info["hsign"] is not None
    piecewise = False
    if pieces >= 2:
        alternative = _shortcut_or_none(info, "piecewise", None)
        piecewise = alternative is None or not np.array_equal(alternative, _render(info))
    return {
        "kind": _kind_name(info), "two_d": two_d,
        "outer": any(o < w for _, w, o in gaps), "inner": any(w < o for _, w, o in gaps),
        "flush": any(w == 0 for _, w, _o in gaps), "docked": all(w == 0 for _, w, _o in gaps),
        "gravity": gravity, "pieces": pieces, "piecewise": piecewise,
        "wall_t": wall_t, "wall_colours": wall_colours,
        "wall_vs_band_t": vs_t, "wall_vs_band_colour": vs_c,
        "two_colour_corner": two_d and info["colour_r"] != info["colour_c"],
        "max_count": max(len(info["row_cells"]), len(info["col_cells"])),
        "priority": info["priority"],
    }


def _kind_name(info):
    return (("D" if info["vsign"] == "+" else "U" if info["vsign"] == "-" else "")
            + ("R" if info["hsign"] == "+" else "L" if info["hsign"] == "-" else ""))


# Shortcut hypotheses: simpler or different rules that a solver might infer.
SHORTCUTS = ("crop_only", "no_dock", "gravity_v", "gravity_h", "nearest_side", "farther_side",
             "fixed_direction", "moved_only", "piecewise", "constant_thickness", "thin_copy",
             "axis_max_thickness", "axis_min_thickness", "far_band_thickness",
             "constant_colour", "far_band_colour", "dominant_band_colour", "swap_colours",
             "flip_junction")
ALWAYS_REFUTED = ("crop_only", "no_dock", "gravity_v", "gravity_h", "nearest_side",
                  "farther_side", "fixed_direction")


def _reference(train_infos):
    ref = {"v": None, "h": None, "t": None, "colour": None}
    for info in train_infos:
        if ref["v"] is None and info["vsign"]:
            ref["v"] = info["vsign"]
        if ref["h"] is None and info["hsign"]:
            ref["h"] = info["hsign"]
    first = train_infos[0]
    ref["t"] = first["t_r"] if first["vsign"] else first["t_c"]
    ref["colour"] = first["colour_r"] if first["vsign"] else first["colour_c"]
    return ref


def _with_copies(info, canvas):
    """Copies of the walls beyond whatever extent a shortcut leaves."""
    cells = np.argwhere(canvas != info["fill"])
    if not len(cells):
        return None
    (a0, b0), (a1, b1) = cells.min(axis=0), cells.max(axis=0)
    out = np.full(canvas.shape, info["fill"], dtype=int)
    rows = cols = None
    if info["vsign"] == "+":
        rows = (a0 - info["t_r"], a0)
    elif info["vsign"] == "-":
        rows = (a1 + 1, a1 + 1 + info["t_r"])
    if info["hsign"] == "+":
        cols = (b0 - info["t_c"], b0)
    elif info["hsign"] == "-":
        cols = (b1 + 1, b1 + 1 + info["t_c"])
    if rows and (rows[0] < 0 or rows[1] > out.shape[0]):
        return None
    if cols and (cols[0] < 0 or cols[1] > out.shape[1]):
        return None
    if rows:
        out[rows[0]:rows[1], :] = info["colour_r"]
    if cols:
        out[:, cols[0]:cols[1]] = info["colour_c"]
    if rows and cols:
        out[rows[0]:rows[1], cols[0]:cols[1]] = info["junction"]
    out[canvas != info["fill"]] = canvas[canvas != info["fill"]]
    return out


def _shortcut_or_none(info, name, ref):
    try:
        return _shortcut(info, name, ref)
    except ValueError:
        return None


def _axis_bands(info, axis):
    return _band_list(info, axis)


def _shortcut(info, name, ref):
    """Output of a plausible alternative rule (None when it cannot apply)."""
    walls = [(axis, sign, index) for axis, sign, index in
             (("r", info["vsign"], info["vband"]), ("c", info["hsign"], info["hband"])) if sign]
    if name == "crop_only":
        return info["crop"]
    if name == "no_dock":
        return _with_copies(info, info["crop"].copy())
    if name in ("gravity_v", "gravity_h"):
        return _with_copies(info, _gravity(info, name == "gravity_v"))
    if name in ("nearest_side", "farther_side"):
        changed = dict(info)
        for axis, wall, outer in _gaps(info):
            key = "vsign" if axis == "r" else "hsign"
            flip = outer < wall if name == "nearest_side" else wall < outer
            if flip:
                changed[key] = "-" if info[key] == "+" else "+"
        return _render(changed)
    if name == "fixed_direction":
        changed = dict(info)
        if info["vsign"] and ref["v"]:
            changed["vsign"] = ref["v"]
        if info["hsign"] and ref["h"]:
            changed["hsign"] = ref["h"]
        return _render(changed)
    if name == "moved_only":
        changed = dict(info)
        for axis, wall, _outer in _gaps(info):
            if wall == 0:
                changed["t_r" if axis == "r" else "t_c"] = 0
        return _render(changed)
    if name == "piecewise":
        labels, count = _components(info["block"] != info["fill"])
        if count < 2:
            return _render(info)
        canvas = np.full(info["crop"].shape, info["fill"], dtype=int)
        for label in range(1, count + 1):
            piece = np.where(labels == label, info["block"], info["fill"])
            cells = np.argwhere(labels == label)
            (a0, b0), (a1, b1) = cells.min(axis=0), cells.max(axis=0)
            part = dict(info, block=piece[a0:a1 + 1, b0:b1 + 1],
                        origin=(info["origin"][0] + a0, info["origin"][1] + b0))
            placed = _content_canvas(part)
            canvas[placed != info["fill"]] = placed[placed != info["fill"]]
        return _with_copies(info, canvas)
    if name == "constant_thickness":
        return _render(dict(info, t_r=ref["t"] if info["vsign"] else 0,
                            t_c=ref["t"] if info["hsign"] else 0))
    if name == "thin_copy":
        return _render(dict(info, t_r=min(info["t_r"], 1), t_c=min(info["t_c"], 1)))
    if name in ("axis_max_thickness", "axis_min_thickness", "far_band_thickness"):
        changed = dict(info)
        for axis, sign, index in walls:
            bands = _axis_bands(info, axis)
            if name == "axis_max_thickness":
                value = max(t for t, _ in bands)
            elif name == "axis_min_thickness":
                value = min(t for t, _ in bands)
            else:
                value = bands[-1][0] if sign == "+" else bands[0][0]
            changed["t_r" if axis == "r" else "t_c"] = value
        return _render(changed)
    if name == "constant_colour":
        return _render(dict(info, colour_r=ref["colour"], colour_c=ref["colour"], junction=ref["colour"]))
    if name in ("far_band_colour", "dominant_band_colour"):
        changed = dict(info)
        if name == "dominant_band_colour":
            g = info["grid"]
            mask = np.zeros(g.shape, dtype=bool)
            for a, b in info["row_bands"]:
                mask[a:b + 1, :] = True
            for a, b in info["col_bands"]:
                mask[:, a:b + 1] = True
            values, counts = np.unique(g[mask], return_counts=True)
            dominant = int(values[int(np.argmax(counts))])
        for axis, sign, index in walls:
            bands = _axis_bands(info, axis)
            colour = dominant if name == "dominant_band_colour" else (bands[-1][1] if sign == "+" else bands[0][1])
            changed["colour_r" if axis == "r" else "colour_c"] = colour
        if changed["vsign"] and changed["hsign"]:
            changed["junction"] = changed["colour_r"] if info["junction"] == info["colour_r"] else changed["colour_c"]
        return _render(changed)
    if name == "swap_colours":
        if not (info["vsign"] and info["hsign"]):
            return _render(info)
        return _render(dict(info, colour_r=info["colour_c"], colour_c=info["colour_r"]))
    if name == "flip_junction":
        if not (info["vsign"] and info["hsign"]):
            return _render(info)
        other = info["colour_c"] if info["junction"] == info["colour_r"] else info["colour_r"]
        return _render(dict(info, junction=other))
    raise KeyError(name)


def _fits(prediction, output):
    return prediction is not None and prediction.shape == output.shape and np.array_equal(prediction, output)


def _shortcut_report(train, test):
    """For each shortcut: (fits every training pair, mispredicts some test pair)."""
    ref = _reference([info for info, _ in train])
    report = {}
    for name in SHORTCUTS:
        fits_train = all(_fits(_shortcut_or_none(info, name, ref), out) for info, out in train)
        misses_test = any(not _fits(_shortcut_or_none(info, name, ref), out) for info, out in test)
        report[name] = (fits_train, misses_test)
    return report


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "train/test keys"
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list), "splits must be lists"
    _resolve_counts(len(train), len(test))
    seen, pairs = set(), []
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys"
        grid_in = np.asarray(pair["input"])
        grid_out = np.asarray(pair["output"])
        for g in (grid_in, grid_out):
            assert g.ndim == 2 and 1 <= g.shape[0] <= MAX_SIDE and 1 <= g.shape[1] <= MAX_SIDE, "shape"
            assert np.issubdtype(g.dtype, np.integer) and g.min() >= 0 and g.max() <= 9, "colours"
        key = (grid_in.shape, grid_in.astype(int).tobytes())
        assert key not in seen, "duplicate input"
        seen.add(key)
        info = _analyze(grid_in)
        _check_input(info)
        assert np.array_equal(_render(info), grid_out), "output does not follow the rule"
        assert grid_out.shape == info["crop"].shape, "output must be the marked compartment"
        assert not np.array_equal(grid_out, info["crop"]), "output equals the unchanged crop"
        pairs.append((info, grid_out.astype(int)))
    feats = [_features(info) for info, _ in pairs]
    tr, te = feats[:len(train)], feats[len(train):]
    kinds = {f["kind"] for f in tr}
    assert {k[0] for k in kinds if k[0] in "DU"} == {"D", "U"}, "training shows both vertical directions"
    assert {k[-1] for k in kinds if k[-1] in "LR"} == {"L", "R"}, "training shows both horizontal directions"
    if any(f["two_d"] for f in te):
        assert any(f["two_d"] for f in tr) or kinds >= set(EDGES), \
            "corner tests need a training corner or strips in all four directions"
    if any(not f["two_d"] for f in te):
        assert any(not f["two_d"] for f in tr), "one-wall tests need a one-wall training pair"
    assert any(f["gravity"] for f in tr), "training separates rigid docking from cell gravity"
    assert any(f["outer"] for f in tr), "training refutes nearest-side docking"
    assert any(f["inner"] for f in tr), "training refutes farther-side docking"
    assert not any(f["docked"] for f in te), "test content must move"
    if any(f["flush"] for f in te):
        assert any(f["flush"] for f in tr), "already-flush test axes need a flush training axis"
    assert sum(f["docked"] for f in tr) <= 1, "at most one already-docked training pair"
    if any(f["max_count"] >= 3 for f in te):
        assert any(f["max_count"] >= 3 for f in tr), "denser partitions need a training example"
    priorities = {f["priority"] for f in feats if f["priority"]}
    assert len(priorities) <= 1, "crossing priority is consistent within a puzzle"
    # No simpler hypothesis may fit every training pair yet change a test answer.
    report = _shortcut_report(pairs[:len(train)], pairs[len(train):])
    for name, (fits_train, misses_test) in report.items():
        assert not (fits_train and misses_test), f"ambiguous: shortcut {name} fits training"
        if name in ALWAYS_REFUTED:
            assert not fits_train, f"training must refute {name}"


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------
def self_test() -> None:
    # Hand-made case: top-left marked quadrant, thickness-1 cross.
    grid = np.array([
        [0, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 3, 3, 0, 0, 5, 0, 0, 0],
        [0, 0, 3, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 0, 0],
        [5, 5, 5, 5, 5, 5, 5, 5, 5],
        [0, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 0, 0],
    ])
    expected = np.array([
        [0, 0, 5, 0, 0],
        [0, 0, 5, 0, 0],
        [5, 5, 5, 5, 5],
        [0, 0, 5, 3, 3],
        [0, 0, 5, 0, 3],
    ])
    assert np.array_equal(solve(grid), expected), "hand-made corner case"
    # One thick horizontal wall: content keeps its columns.
    grid = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 8],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 4, 1, 1],
        [1, 1, 1, 4, 4, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])
    expected = np.array([
        [1, 1, 1, 1, 4, 1, 1],
        [1, 1, 1, 4, 4, 1, 1],
        [8, 8, 8, 8, 8, 8, 8],
        [8, 8, 8, 8, 8, 8, 8],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ])
    assert np.array_equal(solve(grid), expected), "hand-made single-wall case"
    # Two-colour partition, vertical bands on top; per-band thickness on rows:
    # wall (rows 6-7, colour 2, 2 thick) differs from the far band (row 10, colour 6).
    grid = np.full((13, 9), 0)
    grid[6:8, :] = 2
    grid[10, :] = 6
    grid[:, 5] = 3
    grid[1, 1] = grid[1, 2] = grid[2, 2] = 7
    expected = np.array([
        [0, 0, 3, 0, 0],
        [0, 0, 3, 0, 0],
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
        [0, 0, 3, 7, 7],
        [0, 0, 3, 0, 7],
    ])
    expected[2:4, :] = 2
    expected[2:4, 2] = 3
    assert np.array_equal(solve(grid), expected), "hand-made two-colour corner case"
    info = _analyze(grid)
    assert info["priority"] == "v" and info["t_r"] == 2 and info["colour_r"] == 2 and info["colour_c"] == 3
    # A middle compartment (walls on both sides of one axis) is out of scope.
    grid = np.full((7, 11), 0)
    grid[:, 3] = grid[:, 7] = 5
    grid[3, :] = 5
    grid[1, 5] = 4
    try:
        solve(grid)
    except ValueError:
        pass
    else:
        raise AssertionError("middle compartments must be rejected")

    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")
    for n_train, n_test in ((2, 1), (3, 2), (10, 4), (5, 4)):
        puzzle = generate_puzzle(11, n_train, n_test)
        assert len(puzzle["train"]) == n_train and len(puzzle["test"]) == n_test
        validate_generated_puzzle(puzzle)
    assert generate_puzzle(5) == generate_puzzle(5), "nondeterministic"
    # The bounded fallback tier produces valid puzzles on its own.
    puzzle, plan = _attempt(np.random.default_rng(3), 3, 1, "minimal")
    assert plan["tier"] == "minimal"
    validate_generated_puzzle(puzzle)
    # The validator rejects a wrong answer (copied wall in the wrong colour).
    broken = generate_puzzle(7)
    out = np.array(broken["test"][0]["output"])
    info = _analyze(np.array(broken["test"][0]["input"]))
    colour = info["colour_r"] if info["vsign"] else info["colour_c"]
    out[out == colour] = (colour + 1) % 10 if (colour + 1) % 10 != info["fill"] else (colour + 2) % 10
    broken["test"][0]["output"] = out.tolist()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a wrong output")

    tallies = {key: Counter() for key in ("tier", "regime", "kind", "layout", "wall_t", "style",
                                          "shift", "feature", "misses")}
    shapes, signatures = set(), set()
    for seed in range(42):
        puzzle, plan = _generate(seed, 4, 1)
        validate_generated_puzzle(puzzle)
        signatures.add(repr(puzzle))
        tallies["tier"][plan["tier"]] += 1
        tallies["regime"][plan["regime"]] += 1
        tallies["style"].update(plan["styles"])
        pairs = [(_analyze(np.asarray(p["input"])), np.asarray(p["output"]))
                 for p in puzzle["train"] + puzzle["test"]]
        train_kinds = set()
        for index, (info, out) in enumerate(pairs):
            assert np.array_equal(solve(info["grid"]), out)
            # Dropping either parent never reproduces an answer.
            assert out.shape != info["grid"].shape, "B without the crop would match"
            assert not np.array_equal(out, info["crop"]), "A's unchanged crop would match"
            f = _features(info)
            tallies["kind"][f["kind"]] += 1
            tallies["layout"][(len(info["row_cells"]), len(info["col_cells"]))] += 1
            tallies["wall_t"].update(f["wall_t"])
            for key in ("two_colour_corner", "wall_vs_band_t", "wall_vs_band_colour", "flush",
                        "docked", "piecewise"):
                tallies["feature"][key] += bool(f[key])
            tallies["feature"]["colours>=2"] += len(set(np.unique(info["block"])) - {info["fill"]}) >= 2
            shapes.add(info["grid"].shape)
            if index < 4:
                train_kinds.add(f["kind"])
            else:
                tallies["shift"]["unseen_kind" if f["kind"] not in train_kinds else "seen_kind"] += 1
        report = _shortcut_report(pairs[:4], pairs[4:])
        for name, (fits_train, misses_test) in report.items():
            assert not (fits_train and misses_test), (seed, name)
            if misses_test:
                tallies["misses"][name] += 1
    assert len(signatures) == 42, "seeds must give different puzzles"
    assert set(tallies["regime"]) == set(REGIMES), tallies["regime"]
    assert len(tallies["tier"]) == 3, tallies["tier"]
    assert set(tallies["kind"]) == set(DOCK), tallies["kind"]
    assert len(tallies["layout"]) >= 12, tallies["layout"]
    assert any(max(k) == 4 for k in tallies["layout"]), tallies["layout"]
    assert {1, 2, 3} <= set(tallies["wall_t"]), tallies["wall_t"]
    assert len(tallies["style"]) >= 11, tallies["style"]
    for key in ("two_colour_corner", "wall_vs_band_t", "wall_vs_band_colour", "flush", "piecewise"):
        assert tallies["feature"][key] >= 3, (key, tallies["feature"])
    assert tallies["feature"]["docked"] >= 1, tallies["feature"]
    assert tallies["feature"]["colours>=2"] >= 40, tallies["feature"]
    assert tallies["shift"]["unseen_kind"] >= 8 and tallies["shift"]["seen_kind"] >= 8, tallies["shift"]
    assert len(shapes) >= 80, len(shapes)
    # Shortcuts are refuted in training and also miss real tests.
    for name in ("gravity_v", "nearest_side", "farther_side", "fixed_direction", "moved_only"):
        assert tallies["misses"][name] >= 3, (name, tallies["misses"])


if __name__ == "__main__":
    self_test()
    print("self_test passed")
