"""Legend-oriented glyph map.

Each input holds one solid rectangular panel tiled by equal glyph tiles (parent A's
fixed-width segmentation of oriented binary glyphs) and, outside it, a legend of
three swatches.  A swatch is one carrier-coloured tile whose glyph is drawn in a
label colour.  The swatches stand at three corners of an axis-aligned rectangle,
exactly like parent B's three external corner keys, and the legend is drawn in the
*target* frame.

Rule: rotate or reflect the whole panel (tile layout and every glyph inside the
tiles) with the unique dihedral operation that makes the panel's three corner
tiles show exactly the three legend glyphs in the legend's corner arrangement and
puts the panel's blank corner tile at the legend's empty corner.  Then replace
every tile of the re-oriented panel by a solid block of the label colour of the
legend glyph it now matches exactly (parent A's whole-tile decoding); blank tiles
stay carrier.  The output is that re-oriented, decoded panel alone.

Glyph identity is read only in the corrected frame: an upper bar seen in a
vertically flipped panel is a lower bar.  The legend therefore serves both as the
codebook and as the orientation key.
"""

from __future__ import annotations

from collections import deque

import numpy as np


OPS = ("identity", "rot90", "rot180", "rot270",
       "transpose", "antitranspose", "flip_v", "flip_h")
SWAPS = frozenset({"rot90", "rot270", "transpose", "antitranspose"})
PLAIN = ("rot180", "flip_v", "flip_h")
INVERSE = {"identity": "identity", "rot90": "rot270", "rot180": "rot180",
           "rot270": "rot90", "transpose": "transpose",
           "antitranspose": "antitranspose", "flip_v": "flip_v", "flip_h": "flip_h"}
CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
TILE_SIZES = ((3, 3), (2, 3), (3, 2), (3, 4), (4, 3), (4, 4), (2, 2), (2, 4), (4, 2))
TILE_WEIGHTS = np.array((3.0, 1.5, 1.5, 1.5, 1.5, 1.0, 0.7, 0.5, 0.5))
SHIFT_MODES = ("novel_op", "larger_panel", "new_tile", "trap_shift")
MIN_TILES = 6
MAX_SIDE = 30


# --------------------------------------------------------------------------
# Geometry helpers shared by the solver and the generator
# --------------------------------------------------------------------------

def _apply(op, x):
    x = np.asarray(x)
    if op == "identity":
        y = x
    elif op == "rot90":
        y = np.rot90(x, 1)
    elif op == "rot180":
        y = np.rot90(x, 2)
    elif op == "rot270":
        y = np.rot90(x, 3)
    elif op == "transpose":
        y = x.T
    elif op == "antitranspose":
        y = np.rot90(x, 2).T
    elif op == "flip_v":
        y = x[::-1, :]
    elif op == "flip_h":
        y = x[:, ::-1]
    else:
        raise ValueError(f"unknown operation {op!r}")
    return np.array(y, copy=True)


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    found = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue, cells = deque([(r, c)]), []
            while queue:
                y, x = queue.popleft()
                cells.append((y, x))
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if (0 <= ny < height and 0 <= nx < width
                            and mask[ny, nx] and not seen[ny, nx]):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            found.append(cells)
    return found


def _parse(grid):
    """Read background, panel, carrier and the three-swatch legend from an input."""
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        raise ValueError("input must be a non-empty 2D grid")
    g = g.astype(int)
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    values, counts = np.unique(border, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    rects = []
    for cells in _components(g != bg):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        top, left = min(rows), min(cols)
        height, width = max(rows) - top + 1, max(cols) - left + 1
        if height * width != len(cells):
            raise ValueError("every object must be a solid rectangle")
        rects.append((top, left, height, width))
    if len(rects) != 4:
        raise ValueError("expected one panel and three legend swatches")
    rects.sort(key=lambda t: (-(t[2] * t[3]), t[0], t[1]))
    ptop, pleft, ph, pw = rects[0]
    swatches = rects[1:]
    a, b = swatches[0][2], swatches[0][3]
    if any((s[2], s[3]) != (a, b) for s in swatches) or ph * pw <= a * b:
        raise ValueError("legend swatches must be equal and smaller than the panel")
    colour_sets = [set(int(v) for v in np.unique(g[r:r + a, c:c + b]))
                   for r, c, _, _ in swatches]
    if any(len(s) != 2 for s in colour_sets):
        raise ValueError("a swatch holds exactly carrier and one label colour")
    common = set.intersection(*colour_sets)
    if len(common) != 1:
        raise ValueError("swatches must share one carrier colour")
    carrier = common.pop()
    rows = sorted({s[0] for s in swatches})
    cols = sorted({s[1] for s in swatches})
    if len(rows) != 2 or len(cols) != 2:
        raise ValueError("swatches must stand at three corners of a rectangle")
    legend = []
    for r, c, _, _ in swatches:
        box = g[r:r + a, c:c + b]
        label = next(iter(set(int(v) for v in np.unique(box)) - {carrier}))
        legend.append((box != carrier, label, (rows.index(r), cols.index(c))))
    corners_used = {entry[2] for entry in legend}
    if len(corners_used) != 3 or len({entry[1] for entry in legend}) != 3:
        raise ValueError("legend corners and labels must be distinct")
    for i in range(3):
        for j in range(i + 1, 3):
            if np.array_equal(legend[i][0], legend[j][0]):
                raise ValueError("legend glyphs must be distinct")
    missing = next(c for c in CORNERS if c not in corners_used)
    panel = g[ptop:ptop + ph, pleft:pleft + pw]
    if carrier not in set(int(v) for v in np.unique(panel)):
        raise ValueError("panel must use the carrier colour")
    return {"bg": bg, "carrier": carrier, "a": a, "b": b, "legend": legend,
            "missing": missing, "panel": panel, "panel_box": rects[0],
            "swatches": swatches, "spans": (rows[1] - rows[0], cols[1] - cols[0])}


def _tile_masks(q, a, b, carrier):
    if q.shape[0] % a or q.shape[1] % b:
        return None
    rows, cols = q.shape[0] // a, q.shape[1] // b
    if rows < 2 or cols < 2:
        return None
    return [[q[i * a:(i + 1) * a, j * b:(j + 1) * b] != carrier for j in range(cols)]
            for i in range(rows)]


def _corner_tile(tiles, bits):
    return tiles[bits[0] * (len(tiles) - 1)][bits[1] * (len(tiles[0]) - 1)]


def _corner_match(tiles, legend, missing):
    return (all(np.array_equal(_corner_tile(tiles, bits), mask)
                for mask, _, bits in legend)
            and not _corner_tile(tiles, missing).any())


def _decode(tiles, a, b, carrier, legend):
    rows, cols = len(tiles), len(tiles[0])
    out = np.full((rows * a, cols * b), carrier, dtype=int)
    for i in range(rows):
        for j in range(cols):
            mask = tiles[i][j]
            if not mask.any():
                continue
            label = next((lab for m, lab, _ in legend if np.array_equal(m, mask)), None)
            if label is None:
                return None
            out[i * a:(i + 1) * a, j * b:(j + 1) * b] = label
    return out


def _orientations(info):
    found = []
    for op in OPS:
        tiles = _tile_masks(_apply(op, info["panel"]), info["a"], info["b"], info["carrier"])
        if tiles is not None and _corner_match(tiles, info["legend"], info["missing"]):
            found.append(op)
    return found


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Orient the panel by its legend, then paint each tile with its glyph's label."""
    info = _parse(input_grid)
    ops = _orientations(info)
    if len(ops) != 1:
        raise ValueError("the legend does not select a unique orientation")
    tiles = _tile_masks(_apply(ops[0], info["panel"]), info["a"], info["b"], info["carrier"])
    out = _decode(tiles, info["a"], info["b"], info["carrier"], info["legend"])
    if out is None:
        raise ValueError("a panel tile does not match any legend glyph")
    return out


# --------------------------------------------------------------------------
# Diagnostics (used by validation and self_test; solve does not need them)
# --------------------------------------------------------------------------

def _d4_equal(m1, m2):
    return any(_apply(op, m1).shape == m2.shape and np.array_equal(_apply(op, m1), m2)
               for op in OPS)


def _analyze(grid):
    info = _parse(grid)
    ops = _orientations(info)
    assert len(ops) == 1, "orientation is not unique"
    op = ops[0]
    a, b, carrier, legend = info["a"], info["b"], info["carrier"], info["legend"]
    panel = info["panel"]
    tiles = _tile_masks(_apply(op, panel), a, b, carrier)
    out = _decode(tiles, a, b, carrier, legend)
    assert out is not None, "some tile is not a legend glyph"
    rows, cols = len(tiles), len(tiles[0])
    kinds = [[next((k for k, (m, _, _) in enumerate(legend) if np.array_equal(m, t)), -1)
              for t in row] for row in tiles]
    # Shortcut 1: choose any orientation whose tiles all decode (ignore corner positions).
    content_alternatives = []
    for other in OPS:
        if other == op:
            continue
        alt_tiles = _tile_masks(_apply(other, panel), a, b, carrier)
        if alt_tiles is None:
            continue
        alt = _decode(alt_tiles, a, b, carrier, legend)
        if alt is not None and (alt.shape != out.shape or not np.array_equal(alt, out)):
            content_alternatives.append(other)
    # Shortcut 2: read glyphs exactly in the input frame, then move tiles with op.
    h, w = (b, a) if op in SWAPS else (a, b)
    raw = np.full(panel.shape, -1, dtype=int)
    for i in range(panel.shape[0] // h):
        for j in range(panel.shape[1] // w):
            mask = panel[i * h:(i + 1) * h, j * w:(j + 1) * w] != carrier
            if not mask.any():
                raw[i * h:(i + 1) * h, j * w:(j + 1) * w] = carrier
                continue
            for m, lab, _ in legend:
                if m.shape == mask.shape and np.array_equal(m, mask):
                    raw[i * h:(i + 1) * h, j * w:(j + 1) * w] = lab
    moved = _apply(op, raw)
    trap = bool(np.any((moved >= 0) & (moved != carrier) & (moved != out)))
    # Shortcut 3: never re-orient; decode in place up to rotation/reflection.
    unoriented = None
    for th, tw in ((a, b), (b, a)):
        if panel.shape[0] % th or panel.shape[1] % tw:
            continue
        guess = np.full(panel.shape, carrier, dtype=int)
        for i in range(panel.shape[0] // th):
            for j in range(panel.shape[1] // tw):
                mask = panel[i * th:(i + 1) * th, j * tw:(j + 1) * tw] != carrier
                if mask.any():
                    hits = [lab for m, lab, _ in legend if _d4_equal(mask, m)]
                    guess[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = hits[0] if len(hits) == 1 else -1
        unoriented = guess
        break
    unoriented_differs = (unoriented is None or unoriented.shape != out.shape
                          or not np.array_equal(unoriented, out))
    related_legend = any(_d4_equal(legend[i][0], legend[j][0])
                         for i in range(3) for j in range(i + 1, 3))
    inks = sorted(set(int(v) for v in np.unique(panel)) - {carrier})
    glyph_tiles = sum(k >= 0 for row in kinds for k in row)
    return {"op": op, "a": a, "b": b, "rows": rows, "cols": cols, "tiles": rows * cols,
            "bg": info["bg"], "carrier": carrier, "inks": inks,
            "labels": sorted(lab for _, lab, _ in legend),
            "legend_masks": tuple(sorted(m.tobytes() + bytes(m.shape) for m, _, _ in legend)),
            "missing": info["missing"], "spans": info["spans"], "output": out,
            "content_alternatives": content_alternatives, "trap": trap,
            "unoriented_differs": unoriented_differs, "related_legend": related_legend,
            "blank_tiles": rows * cols - glyph_tiles, "kinds": kinds,
            "side": _legend_side(info), "panel_shape": panel.shape}


def _legend_side(info):
    ptop, pleft, ph, pw = info["panel_box"]
    a, b = info["a"], info["b"]
    tops = [s[0] for s in info["swatches"]]
    lefts = [s[1] for s in info["swatches"]]
    if max(lefts) + b <= pleft:
        return "left"
    if min(lefts) >= pleft + pw:
        return "right"
    if max(tops) + a <= ptop:
        return "above"
    return "below"


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _mask(a, b, cells):
    m = np.zeros((a, b), dtype=bool)
    for r, c in cells:
        m[r, c] = True
    return m


_BANKS = {}


def _mask_bank(a, b):
    if (a, b) in _BANKS:
        return _BANKS[(a, b)]
    found = {}

    def add(cells):
        m = _mask(a, b, cells)
        if 1 <= int(m.sum()) < a * b:
            found.setdefault(m.tobytes(), m)

    mr, mc = a // 2, b // 2
    row = lambda r: [(r, c) for c in range(b)]
    col = lambda c: [(r, c) for r in range(a)]
    for r in range(a):
        add(row(r))
    for c in range(b):
        add(col(c))
    for point in ((0, 0), (0, b - 1), (a - 1, 0), (a - 1, b - 1), (mr, mc),
                  (0, mc), (a - 1, mc), (mr, 0), (mr, b - 1)):
        add([point])
    for r0 in (0, a - 1):
        for c0 in (0, b - 1):
            add(row(r0) + col(c0))
            dr, dc = (1 if r0 == 0 else -1), (1 if c0 == 0 else -1)
            add([(r0, c0), (r0 + dr, c0), (r0, c0 + dc)])
    add(row(0) + col(mc))
    add(row(a - 1) + col(mc))
    add(col(0) + row(mr))
    add(col(b - 1) + row(mr))
    add(row(mr) + col(mc))
    add(row(0) + row(a - 1))
    add(col(0) + col(b - 1))
    add(row(0) + row(a - 1) + col(0) + col(b - 1))
    n = max(a, b)
    diag = [(int(t * (a - 1) / (n - 1) + 0.5), int(t * (b - 1) / (n - 1) + 0.5))
            for t in range(n)]
    anti = [(r, b - 1 - c) for r, c in diag]
    add(diag)
    add(anti)
    add(diag + anti)
    add([(0, 0), (0, b - 1), (a - 1, 0), (a - 1, b - 1)])
    add([(0, mc), (a - 1, mc), (mr, 0), (mr, b - 1)])
    add([(r, c) for r in range(a) for c in range(b) if (r + c) % 2 == 0])
    add(row(0)[: (b + 1) // 2] + col(0)[: (a + 1) // 2])
    add([(r, c) for r in range(a) for c in range(b) if r <= c * (a - 1) / max(1, b - 1)])
    _BANKS[(a, b)] = list(found.values())
    return _BANKS[(a, b)]


def _source_vocabulary():
    """Parent A's five source glyphs plus the D4 partners that create frame traps."""
    upper = _mask(3, 3, [(0, 0), (0, 1), (0, 2)])
    lower = _mask(3, 3, [(2, 0), (2, 1), (2, 2)])
    border = _mask(3, 3, [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)])
    centre = _mask(3, 3, [(1, 1)])
    anti = _mask(3, 3, [(0, 2), (1, 1), (2, 0)])
    main = _mask(3, 3, [(0, 0), (1, 1), (2, 2)])
    left = _mask(3, 3, [(0, 0), (1, 0), (2, 0)])
    right = _mask(3, 3, [(0, 2), (1, 2), (2, 2)])
    return [upper, lower, border, centre, anti, main, left, right]


def _random_shape(rng, a, b):
    size = int(rng.integers(2, max(3, (a * b) // 2 + 1)))
    cells = {(int(rng.integers(a)), int(rng.integers(b)))}
    while len(cells) < size:
        frontier = sorted({(r + dr, c + dc) for r, c in cells
                           for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if 0 <= r + dr < a and 0 <= c + dc < b} - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return _mask(a, b, cells)


def _contains(pool, mask):
    return any(np.array_equal(mask, m) for m in pool)


def _legend_masks(rng, a, b, role, op, source_style=False):
    if source_style and (a, b) == (3, 3):
        pool = _source_vocabulary()
    else:
        pool = list(_mask_bank(a, b))
        for _ in range(3):
            shape = _random_shape(rng, a, b)
            if not _contains(pool, shape):
                pool.append(shape)
    if role == "trap":
        inverse = INVERSE[op]
        if op == "identity" or (a != b and op in SWAPS):
            return None
        candidates = [m for m in pool if not np.array_equal(_apply(inverse, m), m)]
        closed = [m for m in candidates if _contains(pool, _apply(inverse, m))]
        if closed and (source_style or rng.random() < 0.5):
            candidates = closed
        first = candidates[int(rng.integers(len(candidates)))]
        second = _apply(inverse, first)
        rest = [m for m in pool if not np.array_equal(m, first) and not np.array_equal(m, second)]
        masks = [first, second, rest[int(rng.integers(len(rest)))]]
    elif role == "symmetric":
        options = [s for s in OPS[1:] if a == b or s not in SWAPS]
        masks = None
        for index in rng.permutation(len(options)):
            symmetry = options[int(index)]
            invariant = [m for m in pool if np.array_equal(_apply(symmetry, m), m)]
            if len(invariant) >= 3:
                picks = rng.choice(len(invariant), 3, replace=False)
                masks = [invariant[int(i)] for i in picks]
                break
        if masks is None:
            return None
    else:
        picks = rng.choice(len(pool), 3, replace=False)
        masks = [pool[int(i)] for i in picks]
    return [masks[int(i)] for i in rng.permutation(3)]


def _place(rng, panel, a, b, legend, bg, carrier, max_extra=6):
    ph, pw = panel.shape
    for _ in range(60):
        span_r = a + 1 + int(rng.integers(0, max_extra + 1))
        span_c = b + 1 + int(rng.integers(0, max_extra + 1))
        lh, lw = span_r + a, span_c + b
        side = int(rng.integers(4))
        gap = int(rng.integers(2, 5))
        if side < 2:
            offset = int(rng.integers(-(lh - 1), ph))
            top = min(0, offset)
            height, width = max(ph, offset + lh) - top, pw + gap + lw
            pr, lr = -top, offset - top
            pc, lc = (lw + gap, 0) if side == 0 else (0, pw + gap)
        else:
            offset = int(rng.integers(-(lw - 1), pw))
            left = min(0, offset)
            height, width = ph + gap + lh, max(pw, offset + lw) - left
            pc, lc = -left, offset - left
            pr, lr = (lh + gap, 0) if side == 2 else (0, ph + gap)
        if height + 2 > MAX_SIDE or width + 2 > MAX_SIDE:
            continue
        mt = int(rng.integers(1, min(3, MAX_SIDE - height - 1) + 1))
        mb = int(rng.integers(1, min(3, MAX_SIDE - height - mt) + 1))
        ml = int(rng.integers(1, min(3, MAX_SIDE - width - 1) + 1))
        mrt = int(rng.integers(1, min(3, MAX_SIDE - width - ml) + 1))
        canvas = np.full((height + mt + mb, width + ml + mrt), bg, dtype=int)
        canvas[mt + pr:mt + pr + ph, ml + pc:ml + pc + pw] = panel
        for mask, label, (rb, cb) in legend:
            r0, c0 = mt + lr + rb * span_r, ml + lc + cb * span_c
            swatch = np.full((a, b), carrier, dtype=int)
            swatch[mask] = label
            canvas[r0:r0 + a, c0:c0 + b] = swatch
        return canvas
    return None


def _make_pair(rng, op, a, b, rows, cols, role, colours, blank_p, mixed_ink,
               source_style=False):
    bg, carrier, inks, labels = colours
    masks = _legend_masks(rng, a, b, role, op, source_style)
    if masks is None:
        return None
    missing = CORNERS[int(rng.integers(4))]
    used = [c for c in CORNERS if c != missing]
    bits = [used[int(i)] for i in rng.permutation(3)]
    weights = rng.dirichlet(np.ones(3) * 1.5) * (1.0 - blank_p)
    probabilities = np.concatenate(([blank_p], weights))
    kinds = rng.choice(4, size=(rows, cols), p=probabilities) - 1
    for k, (rb, cb) in enumerate(bits):
        kinds[rb * (rows - 1), cb * (cols - 1)] = k
    kinds[missing[0] * (rows - 1), missing[1] * (cols - 1)] = -1
    canonical = np.full((rows * a, cols * b), carrier, dtype=int)
    output = np.full((rows * a, cols * b), carrier, dtype=int)
    used_inks = set()
    for i in range(rows):
        for j in range(cols):
            k = int(kinds[i, j])
            if k < 0:
                continue
            ink = inks[int(rng.integers(len(inks)))] if mixed_ink else inks[0]
            used_inks.add(ink)
            block = canonical[i * a:(i + 1) * a, j * b:(j + 1) * b]
            block[masks[k]] = ink
            output[i * a:(i + 1) * a, j * b:(j + 1) * b] = labels[k]
    if mixed_ink and len(used_inks) < 2:
        return None
    panel = _apply(INVERSE[op], canonical)
    legend = [(masks[k], labels[k], bits[k]) for k in range(3)]
    grid = _place(rng, panel, a, b, legend, bg, carrier, 3 if source_style else 6)
    if grid is None:
        return None
    return {"input": grid, "output": output}


def _check_count(name, value, low, high, default):
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _weighted_size(rng, exclude=()):
    options = [i for i, s in enumerate(TILE_SIZES) if s not in exclude]
    weights = TILE_WEIGHTS[options] / TILE_WEIGHTS[options].sum()
    return TILE_SIZES[options[int(rng.choice(len(options), p=weights))]]


def _sample_rc(rng, a, b, low, high):
    for _ in range(200):
        rows, cols = int(rng.integers(2, 8)), int(rng.integers(2, 8))
        if low <= rows * cols <= high and rows * a <= 22 and cols * b <= 22:
            return rows, cols
    return None


def _colours(rng, convention):
    if convention["stable"]:
        bg, carrier, inks = convention["bg"], convention["carrier"], convention["inks"]
    else:
        bg = 0 if rng.random() < 0.4 else int(rng.integers(10))
        rest = [c for c in rng.permutation(10).tolist() if c != bg]
        carrier = rest[0]
        inks = rest[1:1 + convention["ink_count"]]
    free = [c for c in range(10) if c not in {bg, carrier, *inks}]
    labels = [free[int(i)] for i in rng.choice(len(free), 3, replace=False)]
    return bg, carrier, list(inks), labels


def _training_ops(rng, num_train, pool):
    """Cycle shuffled operations; force one axis-swapping and one plain turn/flip."""
    ops = []
    while len(ops) < num_train:
        ops += [pool[int(i)] for i in rng.permutation(len(pool))]
    ops = ops[:num_train]
    plains = [p for p in PLAIN if p in pool]
    swaps = [s for s in OPS if s in SWAPS and s in pool]
    if not any(op in PLAIN for op in ops):
        index = ops.index("identity") if "identity" in ops else int(rng.integers(num_train))
        ops[index] = plains[int(rng.integers(len(plains)))]
    if not any(op in SWAPS for op in ops):
        index = next(i for i, op in enumerate(ops)
                     if op not in PLAIN or sum(o in PLAIN for o in ops) > 1)
        ops[index] = swaps[int(rng.integers(len(swaps)))]
    return ops


def _trap_ok(op, tile):
    return op != "identity" and (op in PLAIN or tile[0] == tile[1])


def _build(rng, num_train, num_test):
    # Puzzle-level conventions.  The "source" regime stays close to both parents:
    # parent A's five 3x3 source glyphs, parent B's three source reflections in
    # training, one stable palette and small panels.
    source_style = bool(rng.random() < 0.25)
    modes = ("novel_op", "larger_panel", "trap_shift") if source_style else SHIFT_MODES
    shift = modes[int(rng.integers(len(modes)))]
    tile_mode = "fixed" if source_style or rng.random() < 0.65 else "mixed"
    base_tile = (3, 3) if source_style else _weighted_size(rng)
    mixed_ink = (not source_style) and bool(rng.random() < 0.4)
    ink_count = int(rng.integers(2, 4)) if mixed_ink else 1
    blank_p = 0.12 if source_style else float(rng.uniform(0.0, 0.35))
    stable = source_style or bool(rng.random() < 0.35)
    if source_style and rng.random() < 0.5:
        bg0, carrier0, inks0 = 0, 1, [5]
    else:
        bg0 = 0 if rng.random() < 0.5 else int(rng.integers(10))
        others = [c for c in rng.permutation(10).tolist() if c != bg0]
        carrier0, inks0 = others[0], others[1:1 + ink_count]
    convention = {"stable": stable, "bg": bg0, "carrier": carrier0,
                  "inks": inks0, "ink_count": ink_count}

    # Operations.
    non_identity = list(OPS[1:])
    if source_style:
        train_pool = ["transpose", "antitranspose", "flip_v"]
        novel_pool = ["rot90", "rot180", "rot270", "flip_h"]
    else:
        train_pool = list(OPS) if rng.random() < 0.3 else list(non_identity)
        novel_pool = non_identity
    novel = novel_pool[int(rng.integers(len(novel_pool)))] if shift == "novel_op" else None
    train_ops = _training_ops(rng, num_train, [op for op in train_pool if op != novel])

    # Tile sizes per pair (output frame).
    test_tile = None
    if shift == "new_tile":
        test_tile = _weighted_size(rng, exclude=(base_tile,) if tile_mode == "fixed" else ())
    if tile_mode == "fixed":
        train_tiles = [base_tile] * num_train
    else:
        train_tiles = [_weighted_size(rng, exclude=(test_tile,) if test_tile else ())
                       for _ in range(num_train)]
    test_tiles = []
    for t in range(num_test):
        if t == 0 and test_tile is not None:
            test_tiles.append(test_tile)
        elif tile_mode == "fixed" and test_tile is None:
            test_tiles.append(base_tile)
        else:
            test_tiles.append(_weighted_size(rng))

    # Roles: training always has one frame-trap pair and one corner-necessary pair.
    trap_candidates = [i for i in range(num_train) if _trap_ok(train_ops[i], train_tiles[i])]
    trap_index = trap_candidates[int(rng.integers(len(trap_candidates)))]
    sym_options = [i for i in range(num_train) if i != trap_index]
    sym_index = sym_options[int(rng.integers(len(sym_options)))]
    train_roles = []
    for i in range(num_train):
        if i == trap_index:
            train_roles.append("trap")
        elif i == sym_index:
            train_roles.append("symmetric")
        else:
            draw = rng.random()
            if draw < 0.15 and _trap_ok(train_ops[i], train_tiles[i]):
                train_roles.append("trap")
            elif draw < 0.3:
                train_roles.append("symmetric")
            else:
                train_roles.append("random")

    trap_ops = {train_ops[i] for i, r in enumerate(train_roles) if r == "trap"}
    familiar = ["transpose", "antitranspose", "flip_v"] if source_style else non_identity
    test_ops, test_roles = [], []
    for t in range(num_test):
        tile = test_tiles[t]
        if t == 0 and shift == "novel_op":
            op = novel
            role = "trap" if rng.random() < 0.35 and _trap_ok(op, tile) else "random"
        elif t == 0 and shift == "trap_shift":
            options = [o for o in non_identity if _trap_ok(o, tile) and o not in trap_ops]
            if not options:
                options = [o for o in non_identity if _trap_ok(o, tile)]
            op, role = options[int(rng.integers(len(options)))], "trap"
        else:
            op = familiar[int(rng.integers(len(familiar)))]
            draw = rng.random()
            role = ("trap" if draw < 0.25 and _trap_ok(op, tile) else
                    "symmetric" if draw < 0.45 else "random")
        test_ops.append(op)
        test_roles.append(role)

    # Panel sizes (tile rows/cols in the output frame).
    if shift == "larger_panel":
        train_range, test0_range = (MIN_TILES, 9), (12, 49)
    elif source_style:
        train_range, test0_range = (MIN_TILES, 12), (MIN_TILES, 16)
    else:
        train_range, test0_range = (MIN_TILES, 24), (MIN_TILES, 30)
    later_range = (MIN_TILES, 16) if source_style else (MIN_TILES, 30)

    pairs = {"train": [], "test": []}
    seen = set()
    plan = ([("train", train_ops[i], train_tiles[i], train_roles[i], train_range)
             for i in range(num_train)] +
            [("test", test_ops[t], test_tiles[t], test_roles[t],
              test0_range if t == 0 else later_range) for t in range(num_test)])
    for split, op, (a, b), role, (low, high) in plan:
        for _ in range(80):
            rc = _sample_rc(rng, a, b, low, high)
            if rc is None:
                raise AssertionError("no panel size fits")
            colours = _colours(rng, convention)
            pair = _make_pair(rng, op, a, b, rc[0], rc[1], role, colours, blank_p,
                              mixed_ink, source_style)
            if pair is None:
                continue
            key = pair["input"].tobytes() + bytes(pair["input"].shape)
            if key in seen:
                continue
            try:
                info = _analyze(pair["input"])
            except (AssertionError, ValueError):
                continue
            if info["op"] != op or not np.array_equal(info["output"], pair["output"]):
                continue
            if role == "trap" and not info["trap"]:
                continue
            if role == "symmetric" and not info["content_alternatives"]:
                continue
            if split == "test" and not info["unoriented_differs"]:
                continue
            seen.add(key)
            pairs[split].append(pair)
            break
        else:
            raise AssertionError("could not build a pair")
    return pairs


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _check_count("num_train", num_train, 2, 10, 4)
    num_test = _check_count("num_test", num_test, 1, 4, 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    for _ in range(40):
        try:
            puzzle = _build(rng, num_train, num_test)
            validate_generated_puzzle(puzzle)
            return puzzle
        except AssertionError:
            continue
    raise RuntimeError("could not generate a valid puzzle")


# --------------------------------------------------------------------------
# Validation and self test
# --------------------------------------------------------------------------

def _as_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))
    return grid.astype(int)


def _puzzle_infos(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _as_grid(pair["input"]), _as_grid(pair["output"])
            key = inp.tobytes() + bytes(inp.shape)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(inp)
            assert np.array_equal(info["output"], out), "output disagrees with rule"
            assert np.array_equal(solve(inp), out)
            assert info["tiles"] >= MIN_TILES
            roles = {info["bg"], info["carrier"]}
            assert len(roles) == 2 and not roles & set(info["labels"])
            assert not set(info["inks"]) & (roles | set(info["labels"]))
            assert set(np.unique(out).tolist()) <= set(info["labels"]) | {info["carrier"]}
            assert set(info["labels"]) <= set(np.unique(out).tolist())
            infos[split].append(info)
    return infos


def validate_generated_puzzle(puzzle: dict) -> None:
    infos = _puzzle_infos(puzzle)
    train, test = infos["train"], infos["test"]
    train_ops = {x["op"] for x in train}
    assert len(train_ops) >= 2, "training needs several orientations"
    assert any(x["op"] in SWAPS for x in train), "training needs an axis-swapping case"
    assert any(x["op"] in PLAIN for x in train), "training needs a non-swapping turn/flip"
    assert any(x["trap"] for x in train), "training must show frame-relative glyph identity"
    assert any(x["content_alternatives"] for x in train), "training must need the corners"
    for x in test:
        assert x["op"] != "identity" and x["unoriented_differs"]
    first = test[0]
    novelty = (first["op"] not in train_ops
               or first["tiles"] > max(x["tiles"] for x in train)
               or (first["a"], first["b"]) not in {(x["a"], x["b"]) for x in train}
               or (first["trap"] and first["op"] not in {x["op"] for x in train if x["trap"]}))
    assert novelty, "first test must shift something relative to training"


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    stats = {"ops": set(), "tiles": set(), "sides": set(), "missing": set(), "shapes": set(),
             "bgs": set(), "mixed": 0, "stable": 0, "related": 0, "test_traps": 0,
             "test_ambiguous": 0, "novel_op": 0, "larger": 0, "new_tile": 0,
             "blank_interior": 0, "tile_counts": set(), "spans": set()}
    for puzzle in puzzles:
        infos = _puzzle_infos(puzzle)
        validate_generated_puzzle(puzzle)
        train, test = infos["train"], infos["test"]
        everything = train + test
        for x in everything:
            stats["ops"].add(x["op"])
            stats["tiles"].add((x["a"], x["b"]))
            stats["sides"].add(x["side"])
            stats["missing"].add(x["missing"])
            stats["shapes"].add(x["panel_shape"])
            stats["bgs"].add(x["bg"])
            stats["tile_counts"].add(x["tiles"])
            stats["spans"].add(x["spans"])
            stats["blank_interior"] += x["blank_tiles"] > 1
        stats["mixed"] += any(len(x["inks"]) > 1 for x in everything)
        stats["stable"] += len({(x["bg"], x["carrier"], tuple(x["inks"])) for x in everything}) == 1
        stats["related"] += any(x["related_legend"] for x in everything)
        stats["test_traps"] += any(x["trap"] for x in test)
        stats["test_ambiguous"] += any(x["content_alternatives"] for x in test)
        stats["novel_op"] += test[0]["op"] not in {x["op"] for x in train}
        stats["larger"] += test[0]["tiles"] > max(x["tiles"] for x in train)
        stats["new_tile"] += (test[0]["a"], test[0]["b"]) not in {(x["a"], x["b"]) for x in train}
    assert stats["ops"] == set(OPS)
    assert len(stats["tiles"]) >= 6 and len(stats["sides"]) == 4 and len(stats["missing"]) == 4
    assert len(stats["shapes"]) >= 40 and len(stats["bgs"]) >= 6 and len(stats["spans"]) >= 20
    assert len(stats["tile_counts"]) >= 12 and stats["blank_interior"] >= 30
    assert stats["mixed"] >= 5 and stats["stable"] >= 5 and stats["related"] >= 30
    assert stats["test_traps"] >= 5 and stats["test_ambiguous"] >= 3
    assert stats["novel_op"] >= 5 and stats["larger"] >= 5 and stats["new_tile"] >= 3
    # Determinism, distinctness and explicit counts.
    again = generate_puzzle(7)
    assert all(np.array_equal(p[k], q[k]) for s in ("train", "test")
               for p, q in zip(puzzles[7][s], again[s]) for k in ("input", "output"))
    for nt, ns in ((2, 1), (3, 2), (10, 4), (5, 4), (10, 1)):
        p = generate_puzzle(500 + nt * 10 + ns, nt, ns)
        assert len(p["train"]) == nt and len(p["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # A corrupted output must be rejected.
    broken = generate_puzzle(3)
    broken["test"][0]["output"] = np.array(broken["test"][0]["output"], copy=True)
    broken["test"][0]["output"][0, 0] = (broken["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
