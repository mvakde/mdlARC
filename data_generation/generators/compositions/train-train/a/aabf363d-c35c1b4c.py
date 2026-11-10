"""Pinned mirror repair (v2).

Every input contains one or more damaged mirror-symmetric figures.  Each figure that
must change is accompanied by a *pin*: a single cell whose color occurs nowhere else in
the grid, isolated by background, lying on the figure's mirror line and separated from
the figure by background along that line.  The mirror line runs through the pin's cell
centre in the direction in which the pin sees its figure:

* a pin above or below its figure marks a vertical line (the pin's column);
* a pin left or right of its figure marks a horizontal line (the pin's row);
* a pin diagonally off a figure's corner/tip (with no figure in its row or column)
  marks the diagonal line through the pin.

Output: every cell of the pinned figure is reflected across the pin's line and the
union (original cells plus reflections) is painted entirely in the pin color.  Repaired
positions overwrite background or clutter.  The pin is erased to background.  Clutter
(colors whose 4-connected pieces have at most three cells), unpinned objects (decoys)
and every other cell are unchanged.

Parent A (aabf363d) contributes the isolated single-cell swatch that recolors the whole
object and is then erased.  Parent B (c35c1b4c) contributes reflective completion of a
damaged figure whose missing mirror positions may hold background or distractors that
must be overwritten while other distractors are preserved.  The connecting concept is
that the swatch is also a mirror pin: its position chooses the figure, the axis
orientation and the exact axis line.
"""
from __future__ import annotations

import numpy as np

_DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
         "ul": (-1, -1), "ur": (-1, 1), "dl": (1, -1), "dr": (1, 1)}
_ORTHO = ("up", "down", "left", "right")
_DIAG = ("ul", "ur", "dl", "dr")
_TYPE_OF = {"up": "v", "down": "v", "left": "h", "right": "h",
            "ul": "m", "dr": "m", "ur": "a", "dl": "a"}
_TYPE_DIRS = {"v": ("down", "up"), "h": ("right", "left"), "m": ("dr", "ul"), "a": ("dl", "ur")}
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left",
        "ul": "dr", "dr": "ul", "ur": "dl", "dl": "ur"}
_TYPES = ("v", "h", "m", "a")
_FIGURE_MIN_COMPONENT = 4
_CLUTTER_MAX_COMPONENT = 3
_CLUTTER_MIN_COUNT = 3
_ORTH_FAMILIES = ("stepped", "blob", "frame", "branch", "carved", "comb", "arch", "ladder",
                  "nested", "dumbbell", "hourglass", "chevron", "glyph")
_ORTH_WEIGHTS = (0.11, 0.11, 0.1, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05, 0.06, 0.07, 0.07, 0.07)
_DIAG_FAMILIES = ("corner", "triangle", "band", "frame", "blob", "carved", "kite", "steps")
_DIAG_WEIGHTS = (0.17, 0.15, 0.12, 0.12, 0.16, 0.1, 0.09, 0.09)
_SHIFTS = ("more_figures", "role_swap", "new_pin_side", "half_test", "bigger", "mixed_test",
           "far_pins")
_ORIENT_SETS = {"v": ("v",), "h": ("h",), "vh": ("v", "h"), "d": ("m", "a"),
                "vhd": ("v", "h", "m", "a")}


class _Retry(Exception):
    """Internal signal: resample the current pair or puzzle."""


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _choose(rng, options, weights=None):
    if weights is None:
        return options[int(rng.integers(len(options)))]
    w = np.asarray(weights, dtype=float)
    return options[int(rng.choice(len(options), p=w / w.sum()))]


# ----------------------------------------------------------------------------
# Grid utilities
# ----------------------------------------------------------------------------

def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        steps = steps + ((1, 1), (1, -1), (-1, 1), (-1, -1))
    seen = np.zeros_like(mask)
    comps = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, comp = [(int(r), int(c))], []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy, dx in steps:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(comp)
    return comps


def _largest(mask):
    return max((len(c) for c in _components(mask)), default=0)


def _crop(mask):
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1].copy()


def _component_size_at(grid, r, c, limit):
    """Size of the 4-connected same-color component at (r, c), capped at limit + 1."""
    color = grid[r, c]
    height, width = grid.shape
    seen = {(r, c)}
    stack = [(r, c)]
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in seen and grid[ny, nx] == color:
                seen.add((ny, nx))
                if len(seen) > limit:
                    return len(seen)
                stack.append((ny, nx))
    return len(seen)


def _cells(mask):
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


# ----------------------------------------------------------------------------
# Mirror geometry.  t2 is twice the axis value of the line in the coordinate u.
#   v: u = c      h: u = r      m (down-right/up-left sight): u = r - c
#   a (down-left/up-right sight): u = r + c
# ----------------------------------------------------------------------------

def _u(kind, cell):
    r, c = cell
    return c if kind == "v" else r if kind == "h" else r - c if kind == "m" else r + c


def _line_t2(kind, pin):
    return 2 * _u(kind, pin)


def _mirror(kind, cell, t2):
    r, c = cell
    if kind == "v":
        return (r, t2 - c)
    if kind == "h":
        return (t2 - r, c)
    k = t2 // 2
    if kind == "m":
        return (c + k, r - k)
    return (k - c, k - r)


def _reflect(cells, pin, direction):
    kind = _TYPE_OF[direction]
    t2 = _line_t2(kind, pin)
    return [_mirror(kind, cell, t2) for cell in cells]


# ----------------------------------------------------------------------------
# Input-only analysis and solver
# ----------------------------------------------------------------------------

def _first_hit(g, bg, r, c, name):
    height, width = g.shape
    dy, dx = _DIRS[name]
    y, x, dist = r + dy, c + dx, 1
    while 0 <= y < height and 0 <= x < width and g[y, x] == bg:
        y, x, dist = y + dy, x + dx, dist + 1
    if 0 <= y < height and 0 <= x < width:
        return dist, int(g[y, x])
    return None


def _analyze(grid):
    g = np.asarray(grid, dtype=int)
    colors, counts = np.unique(g, return_counts=True)
    count = {int(c): int(n) for c, n in zip(colors, counts)}
    bg = int(colors[int(np.argmax(counts))])
    pins = []
    for color, n in sorted(count.items()):
        if color != bg and n == 1:
            r, c = np.argwhere(g == color)[0]
            pins.append((int(r), int(c), color))
    pins.sort()
    figure_colors = {c for c, n in count.items()
                     if c != bg and n >= 2 and _largest(g == c) >= _FIGURE_MIN_COMPONENT}
    links = []
    for r, c, pin_color in pins:
        found = {"ortho": [], "diag": []}
        for group, names in (("ortho", _ORTHO), ("diag", _DIAG)):
            for order, name in enumerate(names):
                hit = _first_hit(g, bg, r, c, name)
                if hit is not None and hit[1] in figure_colors:
                    found[group].append((hit[0], order, name, hit[1]))
            found[group].sort()
        choice = None
        if found["ortho"]:
            choice = found["ortho"][0]
        elif found["diag"]:
            choice = found["diag"][0]
        links.append({"pin": (r, c), "pin_color": pin_color,
                      "ortho": [(d, n, col) for d, _, n, col in found["ortho"]],
                      "diag": [(d, n, col) for d, _, n, col in found["diag"]],
                      "choice": None if choice is None else (choice[0], choice[2], choice[3])})
    return {"bg": bg, "count": count, "pins": pins, "figure_colors": figure_colors, "links": links}


def solve(input_grid):
    """Derive the output from the input alone."""
    g = np.array(input_grid, dtype=int)
    height, width = g.shape
    info = _analyze(g)
    out = g.copy()
    for link in info["links"]:
        if link["choice"] is None:
            continue
        _, direction, fig_color = link["choice"]
        cells = _cells(g == fig_color)
        for y, x in cells + _reflect(cells, link["pin"], direction):
            if 0 <= y < height and 0 <= x < width:
                out[y, x] = link["pin_color"]
        out[link["pin"]] = info["bg"]
    return out


# ----------------------------------------------------------------------------
# Canonical shapes.
#   orth: vertical mirror axis at the centre column (width 2h+1).
#   diag: symmetric under transposition (mirror line r == c), square bounding box.
# ----------------------------------------------------------------------------

def _mirror_half(half):
    rows, k = half.shape
    h = k - 1
    full = np.zeros((rows, 2 * h + 1), dtype=bool)
    full[:, :k] |= half
    full[:, h:] |= half[:, ::-1]
    return full


def _grow_half(rng, rows, h):
    half = np.zeros((rows, h + 1), dtype=bool)
    area = rows * (h + 1)
    target = int(rng.integers(max(3, area * 35 // 100), max(4, area * 80 // 100) + 1))
    half[int(rng.integers(rows)), h] = True
    style = int(rng.integers(3))
    while half.sum() < target:
        frontier = sorted({(y + dy, x + dx) for y, x in zip(*np.nonzero(half))
                           for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if 0 <= y + dy < rows and 0 <= x + dx <= h and not half[y + dy, x + dx]})
        if not frontier:
            break
        weights = np.array([sum(1 for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                                if 0 <= fy + dy < rows and 0 <= fx + dx <= h and half[fy + dy, fx + dx])
                            for fy, fx in frontier], dtype=float)
        if style == 0:
            weights = (weights + 0.5) ** 2
        elif style == 1:
            weights = 1.0 / (weights + 0.5) ** 2
        else:
            weights = np.ones_like(weights)
        pick = frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))]
        half[pick] = True
    return half


def _stepped_half(rng, rows, h):
    half = np.zeros((rows, h + 1), dtype=bool)
    w = int(rng.integers(0, h + 1))
    widths = []
    for _ in range(rows):
        widths.append(w)
        w = int(np.clip(w + int(rng.choice([-2, -1, 0, 0, 1, 2])), 0, h))
    widths[int(rng.integers(rows))] = h
    for r, w in enumerate(widths):
        half[r, h - w:] = True
    return half


def _carve_pairs(rng, full, n, mirror, off_axis):
    """Remove n mirror pairs of cells while keeping one 4-connected piece."""
    for _ in range(12 * n):
        if n <= 0:
            break
        cand = [p for p in _cells(full) if off_axis(p)]
        if len(cand) <= 2:
            break
        p = cand[int(rng.integers(len(cand)))]
        trial = full.copy()
        trial[p] = trial[mirror(p)] = False
        if trial.sum() >= 6 and len(_components(trial)) == 1 and _crop(trial).shape == full.shape:
            full = trial
            n -= 1
    return full


def _orth_raw(rng, fam, rows, h):
    """Half-masks (rows x h+1, axis column h) or full masks for each orthogonal family."""
    if fam == "stepped":
        return _mirror_half(_stepped_half(rng, rows, h))
    if fam == "blob":
        return _mirror_half(_grow_half(rng, rows, h))
    if fam == "frame":
        full = np.zeros((rows, 2 * h + 1), dtype=bool)
        full[0, :] = full[-1, :] = full[:, 0] = full[:, -1] = True
        variant = int(rng.integers(6))
        if variant == 1:
            k = int(rng.integers(1, h + 1))
            full[0, h - k + 1:h + k] = False
        elif variant == 2:
            full[:, h] = True
        elif variant == 3 and rows >= 5:
            full[rows // 2, :] = True
        elif variant == 4:
            k = int(rng.integers(1, h + 1))
            full[-1, h - k + 1:h + k] = False
        elif variant == 5 and rows >= 6 and h >= 3:
            full[1, :] = full[-2, :] = full[:, 1] = full[:, -2] = True
        return full
    if fam == "branch":
        half = np.zeros((rows, h + 1), dtype=bool)
        half[:, h] = True
        n_arms = int(rng.integers(1, min(3, rows) + 1))
        arm_rows = rng.choice(rows, n_arms, replace=False)
        for i, r in enumerate(arm_rows):
            a = h if i == 0 else int(rng.integers(1, h + 1))
            half[r, h - a:h] = True
            if rng.random() < 0.5:
                rr = int(r) + int(rng.choice([-1, 1]))
                if 0 <= rr < rows:
                    half[rr, h - a] = True
        return _mirror_half(half)
    if fam == "carved":
        base = _stepped_half(rng, rows, h) if rng.random() < 0.6 else _grow_half(rng, rows, h)
        full = _crop(_mirror_half(base))
        hh = full.shape[1] // 2
        return _carve_pairs(rng, full, int(rng.integers(1, 4)),
                            lambda p: (p[0], 2 * hh - p[1]), lambda p: p[1] < hh)
    half = np.zeros((rows, h + 1), dtype=bool)
    if fam == "comb":
        half[-1, :] = True
        start = int(rng.integers(0, 2))
        teeth = list(range(start, h + 1, 2)) or [h]
        for col in teeth:
            height = int(rng.integers(1, rows))
            half[rows - 1 - height:rows - 1, col] = True
        return _mirror_half(half)
    if fam == "arch":
        th = 1 if rows < 5 else int(rng.integers(1, 3))
        lw = 1 if h < 3 else int(rng.integers(1, 3))
        half[:th, :] = True
        half[:, :lw] = True
        if h >= 4 and rng.random() < 0.5:
            col = int(rng.integers(lw + 1, h))
            half[th:th + int(rng.integers(1, rows - th + 1)), col] = True
        return _mirror_half(half)
    if fam == "ladder":
        half[:, 0] = True
        rungs = sorted(set(int(r) for r in rng.choice(rows, min(rows, int(rng.integers(2, 5))), replace=False)))
        for r in rungs:
            half[r, :] = True
        return _mirror_half(half)
    if fam == "nested":
        if rows < 7 or h < 3:
            return _orth_raw(rng, "frame", rows, h)
        full = np.zeros((rows, 2 * h + 1), dtype=bool)
        full[0, :] = full[-1, :] = full[:, 0] = full[:, -1] = True
        full[2, 2:-2] = full[-3, 2:-2] = True
        full[2:-2, 2] = full[2:-2, -3] = True
        stem = 1 if rng.random() < 0.5 else rows - 2
        full[stem, h] = True
        if stem == rows - 2 and rng.random() < 0.5:
            full[0, h] = False
        return full
    if fam == "dumbbell":
        if h >= 3 and rng.random() < 0.5:
            k = int(rng.integers(2, min(rows, 4) + 1))
            kb = int(rng.integers(1, max(2, h - 1)))
            half = np.zeros((k, h + 1), dtype=bool)
            half[:, :kb] = True
            half[k // 2, kb:] = True
            return _mirror_half(half)
        b = int(rng.integers(1, h + 1))
        kb = int(rng.integers(1, 3))
        rows = max(rows, 2 * kb + 2)
        half = np.zeros((rows, h + 1), dtype=bool)
        half[:kb, h - b:] = True
        half[-kb:, h - int(rng.integers(1, h + 1)):] = True
        half[:, h] = True
        return _mirror_half(half)
    if fam == "hourglass":
        mid = rows // 2
        base = int(rng.integers(0, 2))
        if rng.random() < 0.5:
            widths = [min(h, base + abs(r - mid)) for r in range(rows)]
        else:
            widths = [max(base, h - abs(r - mid)) for r in range(rows)]
        for r, w in enumerate(widths):
            half[r, h - w:] = True
        return _mirror_half(half)
    if fam == "chevron":
        t = 2 if h < 4 else int(rng.integers(2, 4))
        depth = min(rows, h + 1)
        half = np.zeros((depth, h + 1), dtype=bool)
        for r in range(depth):
            d = depth - 1 - r
            half[r, max(0, h - d - t + 1):h - d + 1] = True
        if rng.random() < 0.5:
            shaft = int(rng.integers(1, 4))
            half = np.vstack([half, np.zeros((shaft, h + 1), dtype=bool)])
            half[depth:, h] = True
        return _mirror_half(half)
    if fam == "glyph":
        letter = _choose(rng, ("H", "T", "U", "A", "Y", "M", "I"))
        if letter == "H":
            half[:, 0] = True
            half[rows // 2, :] = True
        elif letter == "T":
            half[0, :] = True
            half[:, h] = True
            if rows >= 5 and rng.random() < 0.5:
                half[1, :] = True
        elif letter == "U":
            half[:, 0] = True
            half[-1, :] = True
        elif letter == "A":
            half[0, 1:] = True
            half[1:, 0] = True
            half[1, 1] = True
            half[min(rows - 1, rows // 2 + 1), :] = True
        elif letter == "Y":
            k = max(1, min(rows - 1, h))
            for r in range(k):
                c0 = int(round(r * h / max(1, k)))
                half[r, c0:min(h, c0 + 1) + 1] = True
            half[k:, h] = True
            half[k - 1, h] = True
        elif letter == "M":
            half[:, 0] = True
            for r in range(min(rows, h + 1)):
                half[r, min(h, r):min(h, r + 1) + 1] = True
            half[0, :2] = True
        else:  # I-beam
            half[0, :] = True
            half[-1, :] = True
            half[:, h] = True
        return _mirror_half(half)
    raise ValueError(fam)


def _orth_ok(full):
    if full.shape[1] % 2 == 0 or not np.array_equal(full, full[:, ::-1]):
        return False
    hh = full.shape[1] // 2
    return (full.sum() >= 6 and int(full[:, :hh].sum()) >= 1 and len(_components(full)) == 1
            and full[:, hh].any())


def _orth_shape(rng, fam, rows, h):
    for attempt in range(30):
        f = fam if attempt < 20 else _choose(rng, ("stepped", "blob"))
        try:
            full = _orth_raw(rng, f, rows, h)
        except (ValueError, IndexError):
            continue
        if not full.any():
            continue
        full = _crop(full)
        if _orth_ok(full):
            return full, f
    raise _Retry("orth shape")


def _diag_raw(rng, fam, size):
    s = size
    rr, cc = np.indices((s, s))
    m = np.zeros((s, s), dtype=bool)
    if fam == "corner":
        t = 1 if s < 4 else int(rng.integers(1, 3))
        m[:t, :] = True
        m[:, :t] = True
        if s >= 5 and rng.random() < 0.4:
            b = int(rng.integers(1, 3))
            m[t:t + b, s - 1] = True
            m[s - 1, t:t + b] = True
    elif fam == "triangle":
        k = s - 1
        variant = int(rng.integers(3))
        if variant == 0:
            m = rr + cc <= k
        elif variant == 1:
            m = (rr == 0) | (cc == 0) | ((rr + cc >= k - 1) & (rr + cc <= k))
        else:
            m = rr + cc >= k
    elif fam == "band":
        m = np.abs(rr - cc) <= 1
        hl = int(rng.integers(2, max(3, s // 2) + 1))
        if rng.random() < 0.5:
            m[0, :hl] = True
            m[:hl, 0] = True
        else:
            m[s - hl:, s - hl:] = True
    elif fam == "frame":
        m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
        variant = int(rng.integers(3))
        if variant == 1 and s >= 4:
            m |= np.abs(rr - cc) <= 1
        elif variant == 2 and s >= 4:
            o = int(rng.integers(1, s // 2))
            m[0, :o] = False
            m[:o, 0] = False
    elif fam == "blob":
        half = np.zeros((s, s), dtype=bool)
        seed = int(rng.integers(s))
        half[seed, seed] = True
        lower = rr >= cc
        target = int(rng.integers(max(3, lower.sum() * 3 // 10), max(4, lower.sum() * 7 // 10) + 1))
        while half.sum() < target:
            frontier = sorted({(y + dy, x + dx) for y, x in zip(*np.nonzero(half))
                               for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                               if 0 <= y + dy < s and 0 <= x + dx < s and lower[y + dy, x + dx]
                               and not half[y + dy, x + dx]})
            if not frontier:
                break
            half[frontier[int(rng.integers(len(frontier)))]] = True
        m = half | half.T
    elif fam == "carved":
        base = (rr + cc <= s - 1 + int(rng.integers(0, s))) if rng.random() < 0.6 else np.ones((s, s), bool)
        m = _carve_pairs(rng, _crop(base), int(rng.integers(1, 4)),
                         lambda p: (p[1], p[0]), lambda p: p[0] < p[1])
    elif fam == "kite":
        u = np.abs(rr - cc)
        sm = rr + cc
        width = int(rng.integers(2, max(3, s // 2 + 1) + 1))
        bound = width * np.minimum(sm, 2 * (s - 1) - sm) / max(1, s - 1)
        m = (u <= 1) | (u <= bound)
    else:  # steps: diagonal spine with symmetric arms
        m = np.abs(rr - cc) <= 1
        picks = rng.choice(s, min(s, int(rng.integers(2, 4))), replace=False)
        for i in picks:
            a = int(rng.integers(1, s - int(i) + 1)) if s - int(i) > 1 else 1
            m[i, i:min(s, i + a + 1)] = True
            m[i:min(s, i + a + 1), i] = True
    return m


def _diag_ok(full):
    if full.shape[0] != full.shape[1] or not np.array_equal(full, full.T):
        return False
    upper = int(np.triu(full, 1).sum())
    return full.sum() >= 6 and upper >= 1 and len(_components(full)) == 1 and np.diag(full).any()


def _diag_shape(rng, fam, size):
    for attempt in range(30):
        f = fam if attempt < 20 else _choose(rng, ("corner", "blob", "triangle"))
        full = _diag_raw(rng, f, size)
        if not full.any():
            continue
        full = _crop(full)
        if _diag_ok(full):
            return full, f
    raise _Retry("diag shape")


def _canon_mirror(kind, shape):
    """Mirror map and signed offset from the axis in a canonical frame."""
    if kind == "orth":
        w = shape[1]
        a = (w - 1) // 2
        return (lambda p: (p[0], w - 1 - p[1])), (lambda p: p[1] - a)
    return (lambda p: (p[1], p[0])), (lambda p: p[0] - p[1])


_PIECES = (
    ((0, 0),), ((0, 0),), ((0, 0),), ((0, 0),),
    ((0, 0), (0, 1)), ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (1, 0)),
)


def _add_satellites(rng, full, kind, n_pairs):
    """Add mirrored satellite parts (1-3 cells) that are 8-separated from the body."""
    pad = 2
    canvas = np.pad(full, pad)
    body = _cells(canvas)
    mirror, offset = _canon_mirror(kind, canvas.shape)
    height, width = canvas.shape
    added = 0
    for _ in range(40):
        if added >= n_pairs:
            break
        piece = _PIECES[int(rng.integers(4, len(_PIECES)))] if rng.random() < 0.6 else ((0, 0),)
        r0, c0 = int(rng.integers(height)), int(rng.integers(width))
        cells = [(r0 + y, c0 + x) for y, x in piece]
        if not all(0 <= y < height and 0 <= x < width and offset((y, x)) < 0 for y, x in cells):
            continue
        twin = [mirror(p) for p in cells]
        both = cells + twin
        occupied = canvas.copy()
        ok = True
        for y, x in both:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and occupied[ny, nx]:
                        ok = False
        if ok:
            ok = min(abs(y - a) + abs(x - b) for y, x in cells for a, b in body) <= 3
        if not ok:
            continue
        if any(abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1 for a in cells for b in twin):
            continue
        for p in both:
            canvas[p] = True
        added += 1
    if not added:
        raise _Retry("satellite")
    return _crop(canvas)


def _trap_shape(rng, kind, fam, size_rows, size_h, need_both):
    """A remnant V that is self-symmetric about a line other than the true axis, and the
    full figure V | mirror(V).  Returns full, visible, removed in the canonical frame."""
    for _ in range(25):
        if kind == "orth":
            if rng.random() < 0.5:
                v, _f = _orth_shape(rng, fam, size_rows, max(1, size_h))
            else:
                k = max(1, size_h)
                rows = size_rows
                half = _stepped_half(rng, rows, k) if rng.random() < 0.6 else _grow_half(rng, rows, k)
                v = _crop(np.hstack([half, half[:, ::-1]]))
                if len(_components(v)) != 1:
                    continue
            w = v.shape[1]
            centre2 = w - 1
            cols = [c for c in range(w) if v[:, c].any() and 2 * c != centre2]
            if need_both:
                cols = [c for c in cols if 0 < c < w - 1 and v[:, :c].any() and v[:, c + 1:].any()]
            if not cols:
                continue
            a = cols[int(rng.integers(len(cols)))]
            hh = max(a, w - 1 - a)
            full = np.zeros((v.shape[0], 2 * hh + 1), dtype=bool)
            vis = np.zeros_like(full)
            vis[:, hh - a:hh - a + w] = v
            full = vis | vis[:, ::-1]
        else:
            size = max(3, size_rows)
            v, _f = _diag_shape(rng, fam, size)
            t = int(rng.choice([-2, -1, 1, 2]))
            n = v.shape[0] + abs(t)
            vis = np.zeros((n, n), dtype=bool)
            if t > 0:
                vis[t:t + v.shape[0], :v.shape[1]] = v
            else:
                vis[:v.shape[0], -t:-t + v.shape[1]] = v
            if not np.diag(vis).any():
                continue
            full = vis | vis.T
            rows = np.nonzero(full.any(axis=1))[0]
            lo, hi = int(rows[0]), int(rows[-1]) + 1
            full, vis = full[lo:hi, lo:hi].copy(), vis[lo:hi, lo:hi].copy()
            if need_both and not (np.tril(vis, -1).any() and np.triu(vis, 1).any()):
                continue
        ok = _orth_ok(full) if kind == "orth" else _diag_ok(full)
        removed = full & ~vis
        if not ok or not removed.any() or _largest(vis) < _FIGURE_MIN_COMPONENT:
            continue
        return full, vis, removed
    raise _Retry("trap")


# ----------------------------------------------------------------------------
# Damage (canonical frame)
# ----------------------------------------------------------------------------

def _pick_cells(rng, cells, k):
    cells = list(cells)
    if k >= len(cells):
        return cells
    if rng.random() < 0.5:
        idx = sorted(int(i) for i in rng.choice(len(cells), k, replace=False))
        return [cells[i] for i in idx]
    cell_set = set(cells)
    chosen = [cells[int(rng.integers(len(cells)))]]
    chosen_set = set(chosen)
    while len(chosen) < k:
        frontier = sorted({(y + dy, x + dx) for y, x in chosen
                           for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                           if (y + dy, x + dx) in cell_set and (y + dy, x + dx) not in chosen_set})
        if not frontier:
            frontier = sorted(cell_set - chosen_set)
        pick = frontier[int(rng.integers(len(frontier)))]
        chosen.append(pick)
        chosen_set.add(pick)
    return chosen


def _damage(rng, full, kind, style):
    """Remove off-axis cells; every mirror pair keeps at least one cell."""
    if style == "intact":
        return full.copy(), np.zeros_like(full)
    mirror, offset = _canon_mirror(kind, full.shape)
    low = [p for p in _cells(full) if offset(p) < 0]
    n = len(low)
    side = int(rng.integers(2))
    if style == "half":
        chosen, sides = low, [side] * n
    elif style == "edge":
        far = max(-offset(p) for p in low)
        chosen = [p for p in low if -offset(p) == far]
        others = [p for p in low if -offset(p) != far]
        if others:
            chosen += _pick_cells(rng, others, int(rng.integers(0, len(others) // 2 + 1)))
        sides = [side] * len(chosen)
    elif style in ("light", "heavy"):
        lo, hi = (0.15, 0.4) if style == "light" else (0.5, 0.85)
        k = max(1, int(round(n * rng.uniform(lo, hi))))
        chosen = _pick_cells(rng, low, k)
        sides = [side] * len(chosen)
    elif style == "two":
        if n < 2:
            raise _Retry("two-sided damage needs two pairs")
        k = max(2, int(round(n * rng.uniform(0.3, 0.7))))
        chosen = _pick_cells(rng, low, k)
        sides = [int(s) for s in rng.integers(0, 2, len(chosen))]
        sides[0], sides[1] = 0, 1
        order = rng.permutation(len(chosen))
        sides = [sides[i] for i in order]
    else:
        raise ValueError(style)
    removed = np.zeros_like(full)
    for p, s in zip(chosen, sides):
        removed[p if s == 0 else mirror(p)] = True
    visible = full & ~removed
    if not removed.any() or _largest(visible) < _FIGURE_MIN_COMPONENT:
        raise _Retry("damage destroyed the figure")
    return visible, removed


def _orient(arr, kind, direction, flip):
    if kind == "orth":
        if flip:
            arr = arr[::-1]
        if _TYPE_OF[direction] == "h":
            arr = arr.T
        return arr.copy()
    if flip:
        arr = arr[::-1, ::-1]
    if direction == "ul":
        arr = arr[::-1, ::-1]
    elif direction == "dl":
        arr = arr[:, ::-1]
    elif direction == "ur":
        arr = arr[::-1, :]
    return arr.copy()


_SIZES = {"orth": {"small": ((3, 5), (1, 2)), "medium": ((4, 8), (2, 4)), "large": ((8, 14), (5, 9))},
          "diag": {"small": (3, 4), "medium": (4, 7), "large": (7, 11)}}


def _make_figure(rng, spec):
    direction = spec["dir"]
    kind = "orth" if _TYPE_OF[direction] in ("v", "h") else "diag"
    size = spec["size"]
    for _ in range(30):
        try:
            if kind == "orth":
                (r0, r1), (h0, h1) = _SIZES["orth"][size]
                rows, h = int(rng.integers(r0, r1 + 1)), int(rng.integers(h0, h1 + 1))
                fam = spec.get("family") or _choose(rng, _ORTH_FAMILIES, _ORTH_WEIGHTS)
            else:
                s0, s1 = _SIZES["diag"][size]
                rows = h = int(rng.integers(s0, s1 + 1))
                fam = spec.get("family") or _choose(rng, _DIAG_FAMILIES, _DIAG_WEIGHTS)
            if spec["style"] == "trap":
                full, visible, removed = _trap_shape(rng, kind, fam, rows, h // 2 if kind == "orth" else h,
                                                     spec.get("avoid_half", False))
                multipart = False
            else:
                full, fam = (_orth_shape(rng, fam, rows, h) if kind == "orth" else _diag_shape(rng, fam, rows))
                multipart = bool(spec.get("multipart"))
                if multipart:
                    full = _add_satellites(rng, full, kind, int(rng.integers(1, 3)))
                visible, removed = _damage(rng, full, kind, spec["style"])
        except _Retry:
            continue
        mirror, offset = _canon_mirror(kind, full.shape)
        vis_off = [offset(p) for p in _cells(visible)]
        rem_off = [offset(p) for p in _cells(removed)]
        two_sided = any(v < 0 for v in rem_off) and any(v > 0 for v in rem_off)
        side_missing = not (any(v < 0 for v in vis_off) and any(v > 0 for v in vis_off))
        if spec["style"] == "two" and not two_sided:
            continue
        if spec.get("avoid_half") and side_missing:
            continue
        if spec["style"] == "half" and not side_missing:
            continue
        flip = bool(rng.random() < 0.5)
        full, visible, removed = (_orient(x, kind, direction, flip) for x in (full, visible, removed))
        return {"full": full, "visible": visible, "removed": removed, "family": fam, "kind": kind,
                "two_sided": two_sided, "multipart": multipart}
    raise _Retry("figure")


def _pin_layout(visible, direction, gap):
    """Local pin position and sightline cells (between pin and first visible figure cell)."""
    fh, fw = visible.shape
    pins = {"down": (-gap - 1, fw // 2), "up": (fh + gap, fw // 2),
            "right": (fh // 2, -gap - 1), "left": (fh // 2, fw + gap),
            "dr": (-gap - 1, -gap - 1), "ul": (fh + gap, fw + gap),
            "dl": (-gap - 1, fw + gap), "ur": (fh + gap, -gap - 1)}
    pin = pins[direction]
    dy, dx = _DIRS[direction]
    y, x = pin[0] + dy, pin[1] + dx
    sight = []
    for _ in range(80):
        if 0 <= y < fh and 0 <= x < fw and visible[y, x]:
            return pin, sight
        sight.append((y, x))
        y, x = y + dy, x + dx
    raise _Retry("pin cannot see its figure")


# ----------------------------------------------------------------------------
# Decoys
# ----------------------------------------------------------------------------

def _decoy_shape(rng, kind, figures):
    if kind == "twin" and figures:
        fig = figures[int(rng.integers(len(figures)))]
        full = fig["full"]
        canon = "orth" if fig["kind"] == "orth" else "diag"
        # Re-damage the same completed shape (bring it back to a canonical frame first).
        d = fig["spec"]["dir"]
        base = full
        if canon == "orth":
            if _TYPE_OF[d] == "h":
                base = base.T
        else:
            base = _orient(base, "diag", d, False)
        vis, _ = _damage(rng, base.copy(), canon, _choose(rng, ("light", "heavy", "two", "edge")))
        mask = _crop(_orient(vis, canon, d, False))
    elif kind == "bar":
        mask = np.ones((1, int(rng.integers(4, 9))), dtype=bool)
    elif kind == "block":
        mask = np.ones((int(rng.integers(2, 4)), int(rng.integers(2, 5))), dtype=bool)
    elif kind == "blob":
        half = _grow_half(rng, int(rng.integers(3, 6)), int(rng.integers(2, 4)))
        mask = _crop(half)
    elif kind == "big":
        if rng.random() < 0.5:
            full, _ = _orth_shape(rng, _choose(rng, _ORTH_FAMILIES, _ORTH_WEIGHTS),
                                  int(rng.integers(6, 11)), int(rng.integers(3, 6)))
            mask, _ = _damage(rng, full, "orth", _choose(rng, ("light", "two", "heavy")))
        else:
            full, _ = _diag_shape(rng, _choose(rng, _DIAG_FAMILIES, _DIAG_WEIGHTS), int(rng.integers(5, 8)))
            mask, _ = _damage(rng, full, "diag", _choose(rng, ("light", "two")))
        mask = _crop(mask)
    else:  # sym: small lightly damaged symmetric shape
        if rng.random() < 0.7:
            full, _ = _orth_shape(rng, _choose(rng, _ORTH_FAMILIES, _ORTH_WEIGHTS), int(rng.integers(3, 6)), 2)
            mask, _ = _damage(rng, full, "orth", "light")
        else:
            full, _ = _diag_shape(rng, _choose(rng, _DIAG_FAMILIES, _DIAG_WEIGHTS), int(rng.integers(3, 6)))
            mask, _ = _damage(rng, full, "diag", "light")
        mask = _crop(mask)
    if kind != "twin" and rng.random() < 0.5:
        mask = mask.T.copy()
    if _largest(mask) < _FIGURE_MIN_COMPONENT:
        raise _Retry("decoy")
    return mask


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

def _dilate_cells(cells):
    return {(y + dy, x + dx) for y, x in cells for dy in (-1, 0, 1) for dx in (-1, 0, 1)}


def _figure_part(fig, index):
    cells = _cells(fig["full"])
    core = cells + [fig["pin"]] + fig["sight"]
    halo = _dilate_cells(cells + [fig["pin"]]) | set(fig["sight"])
    return {"core": core, "halo": halo, "objs": [(p, index) for p in _cells(fig["visible"])],
            "pins": [(fig["pin"], fig["spec"]["dir"], index)]}


def _mask_part(mask, index):
    cells = _cells(mask)
    return {"core": cells, "halo": _dilate_cells(cells), "objs": [(p, index) for p in cells], "pins": []}


def _make_group(parts):
    """parts: list of (part, (oy, ox)); returns a placement item."""
    core, halo, objs, pins = [], set(), [], []
    for part, (oy, ox) in parts:
        core += [(y + oy, x + ox) for y, x in part["core"]]
        halo |= {(y + oy, x + ox) for y, x in part["halo"]}
        objs += [((y + oy, x + ox), i) for (y, x), i in part["objs"]]
        pins += [((y + oy, x + ox), d, i) for (y, x), d, i in part["pins"]]
    ys = [p[0] for p in core]
    xs = [p[1] for p in core]
    return {"core": core, "halo": sorted(halo), "objs": objs, "pins": pins,
            "bounds": (min(ys), min(xs), max(ys), max(xs))}


def _try_layout(rng, height, width, items, diag_puzzle):
    """Place items one at a time; every offset must keep all pin rays unambiguous."""
    core = np.zeros((height, width), dtype=bool)
    halo = np.zeros((height, width), dtype=bool)
    cls = np.full((height, width), -1, dtype=int)
    pins = []
    offsets = []
    for item in items:
        r0, c0, r1, c1 = item["bounds"]
        span_r, span_c = r1 - r0 + 1, c1 - c0 + 1
        if span_r > height or span_c > width:
            return None
        for _ in range(80):
            oy = int(rng.integers(0, height - span_r + 1)) - r0
            ox = int(rng.integers(0, width - span_c + 1)) - c0
            item_core = [(y + oy, x + ox) for y, x in item["core"]]
            item_halo = [(y + oy, x + ox) for y, x in item["halo"]
                         if 0 <= y + oy < height and 0 <= x + ox < width]
            if any(core[p] for p in item_halo) or any(halo[p] for p in item_core):
                continue
            objs = [((y + oy, x + ox), i) for (y, x), i in item["objs"]]
            for p, i in objs:
                cls[p] = i
            new_pins = [((y + oy, x + ox), d, i) for (y, x), d, i in item["pins"]]
            if not _ray_ok(cls, pins + new_pins, diag_puzzle):
                for p, _ in objs:
                    cls[p] = -1
                continue
            pins += new_pins
            for p in item_core:
                core[p] = True
            for p in item_halo:
                halo[p] = True
            offsets.append((oy, ox))
            break
        else:
            return None
    return offsets


def _stack_offset(fig_a, b_shape, b_axis, sep):
    """Offset of object B beyond figure A along A's sightline, centred on A's line."""
    d = fig_a["spec"]["dir"]
    fh, fw = fig_a["full"].shape
    bh, bw = b_shape
    if d == "down":
        return (fh + sep, fw // 2 - b_axis)
    if d == "up":
        return (-sep - bh, fw // 2 - b_axis)
    if d == "right":
        return (fh // 2 - b_axis, fw + sep)
    return (fh // 2 - b_axis, -sep - bw)


def _ray_ok(grid_class, pins_info, diag_puzzle):
    """Check pin rays on the painted figure/decoy classes before clutter."""
    height, width = grid_class.shape
    for pin, direction, own in pins_info:
        own_dist = None
        dy, dx = _DIRS[direction]
        y, x, k = pin[0] + dy, pin[1] + dx, 1
        while 0 <= y < height and 0 <= x < width:
            if grid_class[y, x] >= 0:
                if grid_class[y, x] != own:
                    return False
                own_dist = k
                break
            y, x, k = y + dy, x + dx, k + 1
        if own_dist is None:
            return False
        is_diag = direction in _DIAG
        for name in _ORTHO + _DIAG:
            if name == direction:
                continue
            dy, dx = _DIRS[name]
            y, x, k = pin[0] + dy, pin[1] + dx, 1
            strict = (name in _ORTHO) or is_diag
            while 0 <= y < height and 0 <= x < width:
                v = grid_class[y, x]
                if v >= 0:
                    if strict:
                        return False
                    if diag_puzzle and v != own and k <= own_dist:
                        return False
                    break
                y, x, k = y + dy, x + dx, k + 1
    return True


# ----------------------------------------------------------------------------
# Pair construction
# ----------------------------------------------------------------------------

def _build_pair(rng, plan):
    figures = []
    for spec in plan["figures"]:
        fig = _make_figure(rng, spec)
        pin, sight = _pin_layout(fig["visible"], spec["dir"], spec["gap"])
        fig.update(spec=spec, pin=pin, sight=sight)
        figures.append(fig)
    decoys = []
    for dspec in plan["decoys"]:
        mask = _decoy_shape(rng, dspec["kind"], figures)
        decoys.append({"mask": mask, "color": dspec["color"], "kind": dspec["kind"]})

    # Groups: stacked pairs of pinned figures, decoys beyond a figure, singletons.
    groups, used_f, used_d = [], set(), set()
    for i, j in plan.get("stacks", ()):
        a, b = figures[i], figures[j]
        sep = int(rng.integers(1, 4))
        b_axis = (b["full"].shape[1] // 2) if _TYPE_OF[b["spec"]["dir"]] == "v" else (b["full"].shape[0] // 2)
        off = _stack_offset(a, b["full"].shape, b_axis, sep)
        groups.append(({"f": [(i, (0, 0)), (j, off)], "d": []},
                       [(_figure_part(a, i), (0, 0)), (_figure_part(b, j), off)]))
        used_f |= {i, j}
    for k, i in plan.get("decoy_beyond", ()):
        if i in used_f or k >= len(decoys):
            continue
        a, dmask = figures[i], decoys[k]["mask"]
        vertical = _TYPE_OF[a["spec"]["dir"]] == "v"
        lines = [c for c in range(dmask.shape[1]) if dmask[:, c].any()] if vertical else \
                [r for r in range(dmask.shape[0]) if dmask[r].any()]
        b_axis = lines[int(rng.integers(len(lines)))]
        off = _stack_offset(a, dmask.shape, b_axis, int(rng.integers(1, 4)))
        groups.append(({"f": [(i, (0, 0))], "d": [(k, off)]},
                       [(_figure_part(a, i), (0, 0)), (_mask_part(dmask, 100 + k), off)]))
        used_f.add(i)
        used_d.add(k)
    for i, fig in enumerate(figures):
        if i not in used_f:
            groups.append(({"f": [(i, (0, 0))], "d": []}, [(_figure_part(fig, i), (0, 0))]))
    for k, dec in enumerate(decoys):
        if k not in used_d:
            groups.append(({"f": [], "d": [(k, (0, 0))]}, [(_mask_part(dec["mask"], 100 + k), (0, 0))]))
    order = rng.permutation(len(groups))
    groups = [groups[int(i)] for i in order]
    items = [_make_group(parts) for _, parts in groups]
    need_r = max(it["bounds"][2] - it["bounds"][0] + 1 for it in items)
    need_c = max(it["bounds"][3] - it["bounds"][1] + 1 for it in items)
    area = sum((it["bounds"][2] - it["bounds"][0] + 3) * (it["bounds"][3] - it["bounds"][1] + 3) for it in items)
    cap = plan.get("max_dim", 30)
    lo_r = max(need_r + int(rng.integers(0, 4)), plan.get("min_rows", 0), 6)
    lo_c = max(need_c + int(rng.integers(0, 4)), plan.get("min_cols", 0), 6)
    if lo_r > cap or lo_c > cap or need_r > cap or need_c > cap:
        raise _Retry("too large")
    height = min(cap, lo_r + int(rng.integers(0, 6)))
    width = min(cap, lo_c + int(rng.integers(0, 6)))
    scale = 1.3 + 0.5 * rng.random()
    while height * width < scale * area and (height < cap or width < cap):
        if (rng.random() < 0.5 and height < cap) or width >= cap:
            height += 1
        else:
            width += 1
    offsets = None
    for _ in range(14):
        offsets = _try_layout(rng, height, width, items, plan.get("diag_puzzle", False))
        if offsets is not None:
            break
        height, width = min(cap, height + 1), min(cap, width + 1)
    if offsets is None:
        raise _Retry("layout")

    bg = plan["bg"]
    grid = np.full((height, width), bg, dtype=int)
    grid_class = np.full((height, width), -1, dtype=int)  # figure index, or 100 + decoy index
    blocked = np.zeros((height, width), dtype=bool)
    placed = [None] * len(figures)
    for (members, _), (gy, gx) in zip(groups, offsets):
        for i, (oy, ox) in members["f"]:
            fig = figures[i]
            spec = fig["spec"]
            oy, ox = oy + gy, ox + gx
            vis = [(r + oy, c + ox) for r, c in _cells(fig["visible"])]
            rem = [(r + oy, c + ox) for r, c in _cells(fig["removed"])]
            full = [(r + oy, c + ox) for r, c in _cells(fig["full"])]
            pin = (fig["pin"][0] + oy, fig["pin"][1] + ox)
            for p in vis:
                grid[p] = spec["color"]
                grid_class[p] = i
            grid[pin] = spec["pin_color"]
            for p in full:
                blocked[p] = True
            for y, x in fig["sight"]:
                blocked[y + oy, x + ox] = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    y, x = pin[0] + dy, pin[1] + dx
                    if 0 <= y < height and 0 <= x < width:
                        blocked[y, x] = True
            placed[i] = {"spec": spec, "vis": vis, "rem": rem, "full": full, "pin": pin}
        for k, (oy, ox) in members["d"]:
            dec = decoys[k]
            for r, c in _cells(dec["mask"]):
                grid[r + gy + oy, c + gx + ox] = dec["color"]
                grid_class[r + gy + oy, c + gx + ox] = 100 + k
                blocked[r + gy + oy, c + gx + ox] = True

    pins_info = [(f["pin"], f["spec"]["dir"], i) for i, f in enumerate(placed)]
    if not _ray_ok(grid_class, pins_info, plan.get("diag_puzzle", False)):
        raise _Retry("ray conflict")

    clutter = list(plan["clutter"])

    def fits(y, x, color):
        old = grid[y, x]
        grid[y, x] = color
        ok = _component_size_at(grid, y, x, _CLUTTER_MAX_COMPONENT) <= _CLUTTER_MAX_COMPONENT
        grid[y, x] = old
        return ok

    for fig in placed:
        fill = fig["spec"]["fill"]
        if not clutter or fill == "bg":
            continue
        for y, x in fig["rem"]:
            if fill == "mixed" and rng.random() < 0.5:
                continue
            for k in rng.permutation(len(clutter)):
                if fits(y, x, clutter[int(k)]):
                    grid[y, x] = clutter[int(k)]
                    break

    if clutter:
        free = (~blocked) & (grid == bg)
        nonbg = int((grid != bg).sum())
        target = int(plan["density"] * free.sum())
        target = min(target, max(0, (height * width) // 2 - nonbg - 2))
        placed_cells = 0
        attempts = 0
        while placed_cells < target and attempts < 40 * max(1, target):
            attempts += 1
            piece = _PIECES[int(rng.integers(len(_PIECES)))]
            if rng.random() < 0.5:
                piece = tuple((x, y) for y, x in piece)
            color = clutter[int(rng.integers(len(clutter)))]
            oy, ox = int(rng.integers(height)), int(rng.integers(width))
            cells = [(y + oy, x + ox) for y, x in piece]
            if not all(0 <= y < height and 0 <= x < width and not blocked[y, x] and grid[y, x] == bg
                       for y, x in cells):
                continue
            for p in cells:
                grid[p] = color
            if _component_size_at(grid, cells[0][0], cells[0][1], _CLUTTER_MAX_COMPONENT) > _CLUTTER_MAX_COMPONENT:
                for p in cells:
                    grid[p] = bg
                continue
            placed_cells += len(cells)
        if plan.get("hug"):
            ring = set()
            for fig in placed:
                for y, x in fig["full"]:
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width and not blocked[ny, nx] and grid[ny, nx] == bg:
                            ring.add((ny, nx))
            ring = sorted(ring)
            want = int(rng.integers(2, 6))
            for idx in rng.permutation(len(ring))[:4 * want]:
                if want <= 0 or int((grid != bg).sum()) * 2 >= height * width - 2:
                    break
                y, x = ring[int(idx)]
                if grid[y, x] != bg:
                    continue
                color = clutter[int(rng.integers(len(clutter)))]
                if fits(y, x, color):
                    grid[y, x] = color
                    want -= 1
        for color in clutter:
            tries = 0
            while int((grid == color).sum()) < _CLUTTER_MIN_COUNT and tries < 400:
                tries += 1
                y, x = int(rng.integers(height)), int(rng.integers(width))
                if not blocked[y, x] and grid[y, x] == bg and fits(y, x, color):
                    grid[y, x] = color
            if int((grid == color).sum()) < _CLUTTER_MIN_COUNT:
                raise _Retry("clutter count")
    if int((grid == bg).sum()) * 2 <= height * width:
        raise _Retry("background not dominant")

    out = grid.copy()
    for fig in placed:
        for p in fig["full"]:
            out[p] = fig["spec"]["pin_color"]
        out[fig["pin"]] = bg
    if not np.array_equal(solve(grid), out):
        raise _Retry("solver mismatch")
    meta = {"dims": (height, width),
            "clutter_overwrite": any(grid[p] in clutter for f in placed for p in f["rem"])}
    return {"input": grid, "output": out}, meta


# ----------------------------------------------------------------------------
# Puzzle planning
# ----------------------------------------------------------------------------

def _resolve_counts(num_train, num_test):
    if num_train is None:
        num_train = max(4, (num_test or 1) + 1) if num_test is not None else 4
    if num_test is None:
        num_test = 1
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _plan_puzzle(rng, num_train, num_test, force_regime=None):
    roll = rng.random()
    regime = force_regime or ("minimal" if roll < 1 / 14 else "maximal" if roll < 1 / 14 + 1 / 7 else "normal")
    if regime == "minimal":
        orient = _choose(rng, ("v", "h"))
        clutter_mode = "none"
        decoy_mode = False
        features = {"multipart": False, "intact": False, "stack": False}
        pool = ["role_swap", "new_pin_side", "half_test", "bigger", "far_pins"]
        k = 1
        bg_mode = _choose(rng, ("zero", "fixed"))
    elif regime == "maximal":
        orient = _choose(rng, ("vhd", "d", "vh"), (0.6, 0.1, 0.3))
        clutter_mode = _choose(rng, ("dense", "heavy"), (0.4, 0.6))
        decoy_mode = True
        features = {"multipart": True, "intact": True, "stack": bool(rng.random() < 0.7)}
        pool = ["more_figures", "role_swap", "new_pin_side", "half_test", "bigger", "far_pins"]
        k = 2
        bg_mode = _choose(rng, ("zero", "fixed", "vary"), (0.3, 0.3, 0.4))
    else:
        orient = _choose(rng, ("v", "h", "vh", "d", "vhd"), (0.2, 0.15, 0.37, 0.1, 0.18))
        clutter_mode = _choose(rng, ("none", "sparse", "dense", "heavy"), (0.2, 0.3, 0.28, 0.22))
        decoy_mode = bool(rng.random() < 0.35)
        features = {"multipart": bool(rng.random() < 0.35), "intact": bool(rng.random() < 0.3),
                    "stack": bool(rng.random() < 0.3)}
        pool = ["more_figures", "role_swap", "new_pin_side", "half_test", "bigger", "far_pins"]
        k = 1 if rng.random() < 0.55 else 2
        bg_mode = _choose(rng, ("zero", "fixed", "vary"), (0.4, 0.25, 0.35))
    types = _ORIENT_SETS[orient]
    if len(types) >= 2 and regime != "minimal":
        pool.append("mixed_test")
    shifts = [pool[int(i)] for i in rng.choice(len(pool), k, replace=False)]
    if "new_pin_side" in shifts:
        features["stack"] = False
    n_clutter_colors = 0 if clutter_mode == "none" else (
        int(rng.integers(2, 4)) if clutter_mode == "heavy" else int(rng.integers(1, 3)))
    return {"regime": regime, "orient_mode": orient, "types": types, "clutter_mode": clutter_mode,
            "bg_mode": bg_mode, "decoy_mode": decoy_mode, "features": features, "shifts": shifts,
            "num_train": num_train, "num_test": num_test,
            "fixed_bg": 0 if bg_mode == "zero" else int(rng.integers(0, 10)),
            "more_m": int(rng.integers(1, 4)), "n_clutter_colors": n_clutter_colors,
            "train_dir": {t: _choose(rng, _TYPE_DIRS[t]) for t in _TYPES}}


def _figure_counts(rng, puzzle):
    nt, ns = puzzle["num_train"], puzzle["num_test"]
    if puzzle["regime"] == "minimal":
        return [1] * nt, [1] * ns
    if "more_figures" in puzzle["shifts"]:
        m = puzzle["more_m"]
        train = [int(rng.integers(1, m + 1)) for _ in range(nt)]
        train[int(rng.integers(nt))] = m
        test = [m + 1] * ns
    elif puzzle["regime"] == "maximal":
        train = [int(_choose(rng, (1, 2, 3, 4), (0.15, 0.3, 0.35, 0.2))) for _ in range(nt)]
        test = [int(_choose(rng, (2, 3, 4), (0.2, 0.45, 0.35))) for _ in range(ns)]
    else:
        train = [int(_choose(rng, (1, 2, 3, 4), (0.36, 0.36, 0.2, 0.08))) for _ in range(nt)]
        test = [int(_choose(rng, (1, 2, 3, 4), (0.22, 0.36, 0.3, 0.12))) for _ in range(ns)]
    if "mixed_test" in puzzle["shifts"]:
        test = [max(2, n) for n in test]
    return train, test


def _orientations(rng, puzzle, train_counts, test_counts):
    types = list(puzzle["types"])
    if len(types) == 1:
        return [[types[0]] * n for n in train_counts], [[types[0]] * n for n in test_counts]
    nt = len(train_counts)
    if "mixed_test" in puzzle["shifts"]:
        per_pair = [_choose(rng, types) for _ in range(nt)]
        need = list(rng.permutation(types))[:nt]
        slots = list(rng.permutation(nt))[:len(need)]
        for s, t in zip(slots, need):
            per_pair[int(s)] = t
        seen = sorted(set(per_pair))
        train = [[o] * n for o, n in zip(per_pair, train_counts)]
        test = []
        for n in test_counts:
            o = [_choose(rng, seen) for _ in range(n)]
            a, b = rng.choice(len(seen), 2, replace=False)
            o[0], o[1] = seen[int(a)], seen[int(b)]
            test.append([o[int(t)] for t in rng.permutation(n)])
        return train, test
    train = [[_choose(rng, types) for _ in range(n)] for n in train_counts]
    slots = [(i, j) for i, n in enumerate(train_counts) for j in range(n)]
    for t, s in zip(rng.permutation(types), rng.permutation(len(slots))):
        i, j = slots[int(s)]
        train[i][j] = str(t)
    seen = sorted({o for row in train for o in row})
    test = [[_choose(rng, seen) for _ in range(n)] for n in test_counts]
    return train, test


def _pair_colors(rng, bg, n_fig, n_clutter, n_decoy_colors, prefer_fig=(), prefer_pin=()):
    pool = [c for c in range(10) if c != bg]
    rng.shuffle(pool)
    fig_colors, pin_colors = [], []
    for preferred, target in ((list(prefer_fig), fig_colors), (list(prefer_pin), pin_colors)):
        rng.shuffle(preferred)
        for c in preferred:
            if len(target) < n_fig and c in pool:
                target.append(c)
                pool.remove(c)
    while len(fig_colors) < n_fig:
        fig_colors.append(pool.pop())
    while len(pin_colors) < n_fig:
        pin_colors.append(pool.pop())
    n_decoy_colors = min(n_decoy_colors, len(pool))
    n_clutter = min(n_clutter, len(pool) - n_decoy_colors)
    clutter = [pool.pop() for _ in range(n_clutter)]
    decoys = [pool.pop() for _ in range(n_decoy_colors)]
    return fig_colors, pin_colors, clutter, decoys


def _size_class(rng, n, shrink):
    weights = {1: (0.2, 0.45, 0.35), 2: (0.3, 0.5, 0.2), 3: (0.45, 0.45, 0.1), 4: (0.62, 0.36, 0.02)}[n]
    cls = _choose(rng, ("small", "medium", "large"), weights)
    if shrink >= 1 and cls == "large":
        cls = "medium"
    if shrink >= 2:
        cls = "small"
    return cls


def _gap(rng, puzzle, split):
    if "far_pins" in puzzle["shifts"]:
        if split == "train":
            return int(_choose(rng, (1, 2, 3)))
        return int(_choose(rng, (4, 5, 6, 7), (0.35, 0.3, 0.2, 0.15)))
    if puzzle["regime"] == "minimal":
        return int(_choose(rng, (1, 2, 3)))
    return int(_choose(rng, (1, 2, 3, 4, 5, 6, 7), (0.24, 0.24, 0.2, 0.12, 0.09, 0.06, 0.05)))


def _build_puzzle(rng, puzzle):
    num_train, num_test = puzzle["num_train"], puzzle["num_test"]
    shifts = puzzle["shifts"]
    feats = puzzle["features"]
    train_counts, test_counts = _figure_counts(rng, puzzle)
    train_orients, test_orients = _orientations(rng, puzzle, train_counts, test_counts)
    half_test = "half_test" in shifts

    # Training coverage slots: two-sided damage, a best-fit-refuting figure, clutter overwrite.
    slots = [(i, j) for i, n in enumerate(train_counts) for j in range(n)]
    two_slot, fit_slot = [slots[int(k)] for k in rng.choice(len(slots), 2, replace=False)]
    clutter_slot = slots[int(rng.integers(len(slots)))]
    feats = dict(feats)
    multi_fig = [s for s in slots if train_counts[s[0]] >= 2 and s not in (two_slot, fit_slot)]
    if not multi_fig:
        feats["intact"] = False
    intact_slot = multi_fig[int(rng.integers(len(multi_fig)))] if feats["intact"] else None
    others = [s for s in slots if s != fit_slot]
    multi_slot = others[int(rng.integers(len(others)))] if feats["multipart"] else None

    def style_for(split, i, j, n):
        if split == "test" and half_test:
            return "half"
        if split == "train":
            if (i, j) == two_slot:
                return "two"
            if (i, j) == fit_slot:
                return "trap" if half_test or rng.random() < 0.4 else "half"
        pool = ["light", "heavy", "two", "edge", "half", "trap"]
        weights = [0.21, 0.16, 0.23, 0.16, 0.12, 0.06]
        if split == "train" and half_test:
            pool, weights = pool[:4] + ["trap"], weights[:4] + [0.06]
        if feats["intact"] and n >= 2:
            pool.append("intact")
            weights.append(0.08)
        return _choose(rng, tuple(pool), tuple(weights))

    styles = {"train": [], "test": []}
    for split, counts in (("train", train_counts), ("test", test_counts)):
        for i, n in enumerate(counts):
            row = [style_for(split, i, j, n) for j in range(n)]
            if split == "train" and intact_slot is not None and intact_slot[0] == i:
                row[intact_slot[1]] = "intact"
            if all(s == "intact" for s in row):
                row[0] = "light"
            styles[split].append(row)

    decoy_flags = {"train": [0] * num_train, "test": [0] * num_test}
    if puzzle["decoy_mode"]:
        for split, count in (("train", num_train), ("test", num_test)):
            for i in range(count):
                if rng.random() < (0.5 if split == "train" else 0.6):
                    decoy_flags[split][i] = int(_choose(rng, (1, 2, 3), (0.55, 0.3, 0.15)))
        roomy = [i for i in range(num_train) if train_counts[i] <= 3] or list(range(num_train))
        i = roomy[int(rng.integers(len(roomy)))]
        decoy_flags["train"][i] = max(1, decoy_flags["train"][i])

    density_range = {"none": (0, 0), "sparse": (0.03, 0.08), "dense": (0.09, 0.17),
                     "heavy": (0.18, 0.28)}[puzzle["clutter_mode"]]

    train_bgs = []
    roles = {"fig": set(), "pin": set()}
    pairs = {"train": [], "test": []}
    metas = {"train": [], "test": []}
    seen = set()
    diag_puzzle = any(t in ("m", "a") for t in puzzle["types"])

    def bg_for(index, split):
        if puzzle["bg_mode"] != "vary":
            return puzzle["fixed_bg"]
        if split == "train" and index == 1:
            return int(rng.choice([c for c in range(10) if c != train_bgs[0]]))
        if split == "test" and "role_swap" in shifts and rng.random() < 0.5:
            pref = sorted(roles["pin"] | roles["fig"])
            if pref:
                return int(pref[int(rng.integers(len(pref)))])
        return 0 if rng.random() < 0.35 else int(rng.integers(0, 10))

    for split, counts, orients in (("train", train_counts, train_orients), ("test", test_counts, test_orients)):
        for index, n in enumerate(counts):
            for attempt in range(30):
                shrink = 0 if attempt < 8 else 1 if attempt < 18 else 2
                bg = bg_for(index, split)
                prefer_fig = prefer_pin = ()
                if split == "test" and "role_swap" in shifts:
                    prefer_fig = sorted(roles["pin"] - {bg})
                    prefer_pin = sorted(roles["fig"] - {bg})
                n_dec = decoy_flags[split][index]
                avail = 9 - 2 * n
                n_dec_colors = 0
                if n_dec:
                    n_dec_colors = 1 if (rng.random() < 0.4 or avail <= 1) else min(n_dec, max(1, avail - 1))
                    n_dec_colors = min(n_dec_colors, avail)
                    if n_dec_colors == 0:
                        n_dec = 0
                n_cl = min(puzzle["n_clutter_colors"], avail - n_dec_colors)
                if split == "train" and index == clutter_slot[0] and puzzle["n_clutter_colors"]:
                    n_cl = max(1, n_cl)
                    n_dec_colors = min(n_dec_colors, avail - n_cl)
                    if n_dec_colors <= 0:
                        n_dec, n_dec_colors = 0, 0
                fig_colors, pin_colors, clutter, dec_colors = _pair_colors(
                    rng, bg, n, n_cl, n_dec_colors, prefer_fig, prefer_pin)
                specs = []
                for j in range(n):
                    t = orients[index][j]
                    if "new_pin_side" in shifts:
                        d = puzzle["train_dir"][t]
                        if split == "test":
                            d = _OPP[d]
                    else:
                        d = _choose(rng, _TYPE_DIRS[t])
                    style = styles[split][index][j]
                    if clutter:
                        fill = _choose(rng, ("bg", "clutter", "mixed"), (0.35, 0.3, 0.35))
                        if split == "train" and (index, j) == clutter_slot:
                            fill = "clutter"
                    else:
                        fill = "bg"
                    multipart = False
                    if feats["multipart"] and style not in ("trap",):
                        multipart = bool(rng.random() < 0.3) or (split == "train" and (index, j) == multi_slot)
                    specs.append({"dir": d, "style": style, "fill": fill, "color": fig_colors[j],
                                  "pin_color": pin_colors[j], "multipart": multipart,
                                  "size": _size_class(rng, n, shrink), "gap": _gap(rng, puzzle, split),
                                  "avoid_half": split == "train" and half_test})
                decoys = []
                if n_dec and dec_colors:
                    for k in range(n_dec):
                        kind = _choose(rng, ("bar", "block", "blob", "sym", "twin", "big"),
                                       (0.12, 0.12, 0.14, 0.22, 0.25, 0.15))
                        if kind == "big" and shrink:
                            kind = "sym"
                        decoys.append({"kind": kind, "color": dec_colors[k % len(dec_colors)]})
                stacks, beyond = [], []
                if feats["stack"] and n >= 2 and rng.random() < 0.5:
                    for t in ("v", "h"):
                        idx = [j for j in range(n) if orients[index][j] == t]
                        if len(idx) >= 2:
                            a, b = idx[0], idx[1]
                            specs[b]["dir"] = _OPP[specs[a]["dir"]]
                            stacks.append((a, b))
                            break
                if decoys and rng.random() < 0.3:
                    cand = [j for j in range(n) if orients[index][j] in ("v", "h")
                            and all(j not in s for s in stacks)]
                    if cand:
                        beyond.append((0, cand[int(rng.integers(len(cand)))]))
                plan = {"bg": bg, "figures": specs, "decoys": decoys, "clutter": clutter,
                        "density": float(rng.uniform(*density_range)), "stacks": stacks,
                        "decoy_beyond": beyond, "diag_puzzle": diag_puzzle,
                        "hug": bool(clutter) and rng.random() < 0.5}
                if "bigger" in shifts:
                    if split == "train":
                        plan["max_dim"] = 20
                    else:
                        plan["min_rows"] = max(m["dims"][0] for m in metas["train"]) + 2
                        plan["min_cols"] = max(m["dims"][1] for m in metas["train"]) + 2
                try:
                    pair, meta = _build_pair(rng, plan)
                except _Retry:
                    continue
                if split == "train" and index == clutter_slot[0] and clutter and not meta["clutter_overwrite"]:
                    continue
                if "bigger" in shifts and split == "train" and max(meta["dims"]) > 20:
                    continue
                key = pair["input"].tobytes() + bytes(pair["input"].shape)
                if key in seen:
                    continue
                seen.add(key)
                break
            else:
                raise _Retry("pair")
            if split == "train":
                train_bgs.append(int(plan["bg"]))
                roles["fig"].update(fig_colors)
                roles["pin"].update(pin_colors)
            pairs[split].append({"input": pair["input"], "output": pair["output"]})
            metas[split].append(meta)
    return pairs


def _generate_with_plan(seed, num_train=4, num_test=1):
    """Diagnostics helper: the puzzle plus the puzzle-level plan that produced it."""
    num_train, num_test = _resolve_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for attempt in range(80):
        force = None if attempt < 50 else "normal" if attempt < 65 else "minimal"
        plan = _plan_puzzle(rng, num_train, num_test, force)
        try:
            pairs = _build_puzzle(rng, plan)
            validate_generated_puzzle(pairs)
        except (_Retry, AssertionError):
            continue
        puzzle = {split: [{"input": p["input"].astype(int).tolist(), "output": p["output"].astype(int).tolist()}
                          for p in pairs[split]] for split in ("train", "test")}
        return puzzle, plan
    raise RuntimeError(f"could not build a puzzle for seed {seed}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    return _generate_with_plan(seed, num_train, num_test)[0]


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def _ray_cells(shape, pin, name):
    height, width = shape
    dy, dx = _DIRS[name]
    y, x = pin[0] + dy, pin[1] + dx
    out = []
    while 0 <= y < height and 0 <= x < width:
        out.append((y, x))
        y, x = y + dy, x + dx
    return out


def _self_symmetric_elsewhere(kind, cells, axis_t2):
    """True if the visible cells are mirror-symmetric about a parallel line != pin line."""
    us = [_u(kind, p) for p in cells]
    t2 = min(us) + max(us)
    if t2 == axis_t2 or (kind in ("m", "a") and t2 % 2):
        return False
    cell_set = set(cells)
    return all(_mirror(kind, p, t2) in cell_set for p in cells)


def _pair_facts(inp, out):
    _require(inp.shape == out.shape, "input/output shapes differ")
    height, width = inp.shape
    info = _analyze(inp)
    bg = info["bg"]
    _require(info["count"][bg] * 2 > inp.size, "background must be a strict majority")
    _require(info["links"], "every pair needs at least one pin")
    pin_colors = {link["pin_color"] for link in info["links"]}
    linked = []
    for link in info["links"]:
        r, c = link["pin"]
        ok = len(link["ortho"]) == 1 or (not link["ortho"] and len(link["diag"]) == 1)
        _require(ok, "each pin must see exactly one figure")
        window = inp[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        _require(int((window != bg).sum()) == 1, "pin must be isolated by background")
        linked.append(link["choice"][2])
    _require(len(set(linked)) == len(linked), "a figure is claimed by two pins")
    clutter_colors = set(info["count"]) - {bg} - pin_colors - info["figure_colors"]
    for color in clutter_colors:
        _require(info["count"][color] >= _CLUTTER_MIN_COUNT, "clutter color too rare")
        _require(_largest(inp == color) <= _CLUTTER_MAX_COMPONENT, "clutter component too large")
    _require(np.array_equal(solve(inp), out), "solve(input) differs from output")
    figure_class = np.isin(inp, sorted(info["figure_colors"]))
    decoy_colors = info["figure_colors"] - set(linked)
    facts = {"bg": bg, "figures": [], "decoys": len(decoy_colors),
             "pins": [link["pin"] for link in info["links"]],
             "decoy_objects": sum(len(_components(inp == c, True)) for c in decoy_colors),
             "clutter_colors": sorted(clutter_colors), "shape": inp.shape,
             "clutter_fraction": float(np.isin(inp, sorted(clutter_colors)).sum()) / inp.size if clutter_colors else 0.0}
    completions = []
    for link in info["links"]:
        dist, direction, fig_color = link["choice"]
        kind = _TYPE_OF[direction]
        cells = _cells(inp == fig_color)
        mirrored = _reflect(cells, link["pin"], direction)
        _require(all(0 <= y < height and 0 <= x < width for y, x in mirrored), "reflection leaves the grid")
        completed = set(cells) | set(mirrored)
        repairs = completed - set(cells)
        overwritten = {int(inp[p]) for p in repairs}
        _require(overwritten <= ({bg} | clutter_colors), "repairs may only overwrite background or clutter")
        for other in completions:
            _require(not (completed & other), "completed figures overlap")
        completions.append(completed)
        # Full-ray invariants: no other figure/decoy cell on any other ray.
        for name in _ORTHO + _DIAG:
            if name == direction:
                continue
            ray = _ray_cells(inp.shape, link["pin"], name)
            hits = [k for k, p in enumerate(ray, 1) if figure_class[p]]
            if name in _ORTHO or direction in _DIAG:
                _require(not hits, "a pin is aligned with another object")
        axis_t2 = _line_t2(kind, link["pin"])
        us = [_u(kind, p) for p in cells]
        rep_u = [_u(kind, p) for p in repairs]
        ys = [p[0] for p in completed]
        xs = [p[1] for p in completed]
        dy, dx = _DIRS[direction]
        k = 1
        while k < 60 and not (min(ys) <= link["pin"][0] + k * dy <= max(ys)
                              and min(xs) <= link["pin"][1] + k * dx <= max(xs)):
            k += 1
        across = (max(xs) - min(xs) + 1) if kind == "v" else (max(ys) - min(ys) + 1) if kind == "h" else \
            max(_u(kind, p) for p in completed) - min(_u(kind, p) for p in completed) + 1
        along = (max(ys) - min(ys) + 1) if kind == "v" else (max(xs) - min(xs) + 1) if kind == "h" else \
            max(max(ys) - min(ys), max(xs) - min(xs)) + 1
        comp_mask = np.zeros(inp.shape, dtype=bool)
        for p in completed:
            comp_mask[p] = True
        facts["figures"].append({
            "type": kind, "direction": direction, "distance": dist, "gap": k - 1,
            "asym": min(us) + max(us) != axis_t2,
            "two_sided": any(2 * v < axis_t2 for v in rep_u) and any(2 * v > axis_t2 for v in rep_u),
            "half_missing": not (any(2 * v < axis_t2 for v in us) and any(2 * v > axis_t2 for v in us)),
            "trap": _self_symmetric_elsewhere(kind, cells, axis_t2),
            "clutter_overwrite": bool(overwritten & clutter_colors),
            "bg_overwrite": bg in overwritten,
            "repairs": len(repairs), "intact": not repairs, "size": len(completed),
            "multipart": len(_components(comp_mask)) > 1,
            "across": int(across), "along": int(along), "color": fig_color,
            "pin_color": link["pin_color"], "completed": completed})
    _require(any(f["repairs"] for f in facts["figures"]), "every pair needs at least one repair")
    # In puzzles-with-diagonals checks are done at puzzle level; record diagonal hits here.
    facts["diag_foreign"] = []
    for link, f in zip(info["links"], facts["figures"]):
        if f["direction"] in _DIAG:
            continue
        for name in _DIAG:
            for k, p in enumerate(_ray_cells(inp.shape, link["pin"], name), 1):
                if figure_class[p]:
                    if int(inp[p]) != f["color"] and k <= f["distance"]:
                        facts["diag_foreign"].append(link["pin"])
                    break
    # Separation: completed figures (+ pins) and decoys never touch.
    objects = [f["completed"] | {link["pin"]} for f, link in zip(facts["figures"], info["links"])]
    for color in decoy_colors:
        objects.append(set(_cells(inp == color)))
    for a in range(len(objects)):
        halo = _dilate_cells(objects[a])
        for b in range(a + 1, len(objects)):
            _require(not (halo & objects[b]), "objects must be separated by background")
    for link in info["links"]:
        _require(out[link["pin"]] == bg, "pin must be erased")
    if clutter_colors:
        _require(any((out == color).any() for color in clutter_colors), "some clutter must survive")
    return facts


_CORE_SHORTCUTS = ("recolor_only", "keep_color", "repairs_only", "bbox_center", "best_fit",
                   "majority_side", "low_to_high", "high_to_low", "keep_pin")


def _best_fit_t2(kind, cells):
    us = [_u(kind, p) for p in cells]
    lo, hi = 2 * min(us), 2 * max(us)
    centre = min(us) + max(us)
    cell_set = set(cells)
    best = None
    for t2 in range(lo, hi + 1):
        if kind in ("m", "a") and t2 % 2:
            continue
        score = sum(1 for p in cells if _mirror(kind, p, t2) in cell_set)
        key = (-score, abs(t2 - centre), t2)
        if best is None or key < best[0]:
            best = (key, t2)
    return best[1]


def _shortcut(grid, mode):
    """Plausible simpler rules, used to prove the training pairs refute them."""
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    info = _analyze(g)
    bg = info["bg"]
    pin_colors = {link["pin_color"] for link in info["links"]}
    clutter = set(info["count"]) - {bg} - pin_colors - info["figure_colors"]
    out = g.copy()
    for link in info["links"]:
        if link["choice"] is None:
            continue
        _, direction, fig_color = link["choice"]
        pin, pin_color = link["pin"], link["pin_color"]
        kind = _TYPE_OF[direction]
        if mode.startswith("force_"):
            kind = mode[-1]
        cells = _cells(g == fig_color)
        us = [_u(kind, p) for p in cells]
        t2 = _line_t2(kind, pin)
        if mode == "bbox_center":
            t2 = min(us) + max(us)
            if kind in ("m", "a") and t2 % 2:
                t2 -= 1
        elif mode == "best_fit":
            t2 = _best_fit_t2(kind, cells)
        mirrored = [_mirror(kind, p, t2) for p in cells]
        low = sum(2 * v < t2 for v in us)
        high = sum(2 * v > t2 for v in us)
        keep = None
        if mode == "low_to_high" or (mode == "majority_side" and low >= high):
            keep = [2 * v < t2 for v in us]
        elif mode == "high_to_low" or mode == "majority_side":
            keep = [2 * v > t2 for v in us]
        if keep is not None:
            mirrored = [m for m, k in zip(mirrored, keep) if k]
        cell_set = set(cells)
        repairs = [m for m in mirrored if 0 <= m[0] < height and 0 <= m[1] < width and m not in cell_set]
        if mode == "no_clutter":
            repairs = [m for m in repairs if g[m] == bg]
        if mode == "recolor_only":
            repairs = []
        old_color = fig_color if mode in ("keep_color", "repairs_only") else pin_color
        new_color = fig_color if mode == "keep_color" else pin_color
        for p in cells:
            out[p] = old_color
        for p in repairs:
            if g[p] == bg or g[p] in clutter:
                out[p] = new_color
        if mode not in ("keep_color", "keep_pin"):
            out[pin] = bg
    return out


def _refuted(pairs, mode):
    return any(not np.array_equal(_shortcut(np.asarray(p["input"]), mode), np.asarray(p["output"]))
               for p in pairs)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test, "pair counts")
    seen = set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for grid in (inp, out):
                _require(grid.ndim == 2 and 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "grid dims")
                _require(np.issubdtype(grid.dtype, np.integer), "integer cells")
                _require(grid.min() >= 0 and grid.max() <= 9, "colors 0..9")
            key = inp.astype(int).tobytes() + bytes(inp.shape)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            facts[split].append(_pair_facts(inp.astype(int), out.astype(int)))
    train_figs = [f for pf in facts["train"] for f in pf["figures"]]
    test_figs = [f for pf in facts["test"] for f in pf["figures"]]
    all_pairs = facts["train"] + facts["test"]
    if any(f["direction"] in _DIAG for pf in all_pairs for f in pf["figures"]):
        _require(not any(pf["diag_foreign"] for pf in all_pairs),
                 "a foreign object is nearer along a diagonal than a pin's own figure")
    _require(any(f["asym"] for f in train_figs), "training must show an off-centre visible figure")
    _require(any(f["two_sided"] for f in train_figs), "training must show two-sided damage")
    _require(any(f["bg_overwrite"] for f in train_figs), "training must repair background cells")
    has_clutter = any(pf["clutter_colors"] for pf in all_pairs)
    if has_clutter:
        _require(any(f["clutter_overwrite"] for f in train_figs), "training must show clutter overwrite")
    train_types = {f["type"] for f in train_figs}
    test_types = {f["type"] for f in test_figs}
    _require(test_types <= train_types, "test axis type unseen in training")
    if any(pf["decoys"] for pf in facts["test"]):
        _require(any(pf["decoys"] for pf in facts["train"]), "test decoy unseen in training")
    if any(f["intact"] for f in test_figs):
        _require(any(f["intact"] for f in train_figs), "test intact figure unseen in training")
    if any(f["multipart"] for f in test_figs):
        _require(any(f["multipart"] for f in train_figs), "test multi-part figure unseen in training")
    for mode in _CORE_SHORTCUTS:
        _require(_refuted(puzzle["train"], mode), f"training does not refute shortcut {mode}")
    if has_clutter:
        _require(_refuted(puzzle["train"], "no_clutter"), "training does not refute no_clutter")
    for t in _TYPES:
        if test_types - {t}:
            _require(_refuted(puzzle["train"], "force_" + t), f"training does not refute force_{t}")
    return facts


# ----------------------------------------------------------------------------
# Self test
# ----------------------------------------------------------------------------

def _grid(rows):
    return np.array([[int(ch) for ch in row] for row in rows])


def _hand_example():
    """In-family example: a vertical pin (4) with two-sided damage and a clutter cell (5)
    on a repair position, plus a horizontal pin (6) whose visible figure is off-centre."""
    rows = ("00000000050",
            "00040000000",
            "00000000000",
            "02225000000",
            "00202000000",
            "00222200000",
            "00000000000",
            "00000080000",
            "00000088000",
            "00000088806",
            "00000080000",
            "55000000000")
    expected = ("00000000050",
                "00000000000",
                "00000000000",
                "04444400000",
                "00404000000",
                "04444400000",
                "00000000000",
                "00000060000",
                "00000066000",
                "00000066600",
                "00000066000",
                "55000060000")
    return _grid(rows), _grid(expected)


def _hand_diag_example():
    """Diagonal pins: 5 sees the corner of 3 down-right (main diagonal), 6 sees the
    triangle 4 down-left (anti-diagonal).  Both figures lose cells on one side."""
    rows = ("500000000006",
            "000000000000",
            "003300044400",
            "003000004400",
            "003000000000",
            "003000000000",
            "000000000000")
    expected = ("000000000000",
                "000000000000",
                "005555066600",
                "005000006600",
                "005000000600",
                "005000000000",
                "000000000000")
    return _grid(rows), _grid(expected)


def _twin_like(inp, pair_facts):
    """A decoy that fits inside a translate of a completed figure and covers half of it."""
    info = _analyze(inp)
    linked = {f["color"] for f in pair_facts["figures"]}
    for color in info["figure_colors"] - linked:
        cells = _cells(inp == color)
        y0, x0 = min(p[0] for p in cells), min(p[1] for p in cells)
        dec = {(y - y0, x - x0) for y, x in cells}
        for f in pair_facts["figures"]:
            comp = f["completed"]
            if len(comp) < 8 or len(dec) * 2 < len(comp):
                continue
            fy, fx = min(p[0] for p in comp), min(p[1] for p in comp)
            fig = {(y - fy, x - fx) for y, x in comp}
            fh, fw = max(p[0] for p in fig) + 1, max(p[1] for p in fig) + 1
            for oy in range(fh):
                for ox in range(fw):
                    if all((y + oy, x + ox) in fig for y, x in dec):
                        return True
    return False


def self_test():
    grid, want = _hand_example()
    _require(np.array_equal(solve(grid), want), "hand example")
    hand = _pair_facts(grid, want)
    _require([f["type"] for f in hand["figures"]] == ["v", "h"], "hand example orientations")
    _require(hand["figures"][0]["two_sided"] and hand["figures"][0]["clutter_overwrite"], "hand damage")
    _require(hand["figures"][1]["asym"], "hand off-centre figure")
    for mode in _CORE_SHORTCUTS + ("no_clutter", "force_v", "force_h"):
        _require(not np.array_equal(_shortcut(grid, mode), want), f"hand example misses shortcut {mode}")
    dgrid, dwant = _hand_diag_example()
    _require(np.array_equal(solve(dgrid), dwant), "diagonal hand example")
    dfacts = _pair_facts(dgrid, dwant)
    _require(sorted(f["type"] for f in dfacts["figures"]) == ["a", "m"], "diagonal hand types")
    for mode in ("recolor_only", "keep_color", "force_v", "force_h", "force_m", "force_a", "bbox_center"):
        _require(not np.array_equal(_shortcut(dgrid, mode), dwant), f"diagonal hand misses {mode}")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4, 5), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, num_train=a, num_test=b)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {a}/{b}")
    signatures = set()
    stats = {"types": set(), "direction": set(), "nfig": set(), "bg": set(), "shapes": set(),
             "shifts": set(), "regimes": set(), "orient_modes": set(), "decoy_puzzles": 0,
             "multi_decoy_pairs": 0, "clutter_puzzles": 0, "heavy_pairs": 0, "two_sided": 0,
             "off_centre": 0, "clutter_overwrite": 0, "test_reuses_pin_color": 0, "large": 0,
             "far_gap": 0, "intact": 0, "multipart": 0, "trap": 0, "colinear_pairs": 0, "twin_pairs": 0,
             "pairs": 0}
    for seed in range(42):
        puzzle, plan = _generate_with_plan(seed)
        if seed < 3:
            _require(puzzle == generate_puzzle(seed), "nondeterministic")
        facts = validate_generated_puzzle(puzzle)
        signatures.add(repr(puzzle))
        stats["shifts"].update(plan["shifts"])
        stats["regimes"].add(plan["regime"])
        stats["orient_modes"].add(plan["orient_mode"])
        all_pairs = facts["train"] + facts["test"]
        stats["decoy_puzzles"] += any(pf["decoys"] for pf in all_pairs)
        stats["clutter_puzzles"] += any(pf["clutter_colors"] for pf in all_pairs)
        train_pin_colors = {f["pin_color"] for pf in facts["train"] for f in pf["figures"]}
        stats["test_reuses_pin_color"] += any(f["color"] in train_pin_colors
                                              for pf in facts["test"] for f in pf["figures"])
        for pair, pf in zip(puzzle["train"] + puzzle["test"], all_pairs):
            stats["twin_pairs"] += pf["decoys"] > 0 and _twin_like(np.asarray(pair["input"]), pf)
            stats["pairs"] += 1
            stats["nfig"].add(len(pf["figures"]))
            stats["bg"].add(pf["bg"])
            stats["shapes"].add(pf["shape"])
            stats["multi_decoy_pairs"] += pf["decoy_objects"] >= 2
            stats["heavy_pairs"] += pf["clutter_fraction"] >= 0.15
            lines = [(f["type"], _line_t2(f["type"], p)) for f, p in zip(pf["figures"], pf["pins"])]
            stats["colinear_pairs"] += len(set(lines)) < len(lines)
            for f in pf["figures"]:
                stats["types"].add(f["type"])
                stats["direction"].add(f["direction"])
                stats["two_sided"] += f["two_sided"]
                stats["off_centre"] += f["asym"]
                stats["clutter_overwrite"] += f["clutter_overwrite"]
                stats["large"] += f["across"] >= 13 or f["along"] >= 11
                stats["far_gap"] += f["gap"] >= 4
                stats["intact"] += f["intact"]
                stats["multipart"] += f["multipart"]
                stats["trap"] += f["trap"]
    _require(len(signatures) == 42, "seeds must give distinct puzzles")
    _require(stats["types"] == set(_TYPES), "vertical, horizontal and both diagonal axes")
    _require(stats["direction"] == set(_DIRS), "pins on all eight sides")
    _require(stats["nfig"] >= {1, 2, 3, 4}, "figure counts 1..4")
    _require(len(stats["bg"]) >= 4, "background colors vary")
    _require(len(stats["shapes"]) >= 40, "grid shapes vary")
    _require(stats["shifts"] == set(_SHIFTS), "every train/test shift occurs")
    _require(stats["regimes"] == {"minimal", "normal", "maximal"}, "minimal and maximal regimes occur")
    _require(stats["decoy_puzzles"] >= 3 and stats["multi_decoy_pairs"] >= 1 and stats["twin_pairs"] >= 1,
             "decoys vary, including unpinned twins")
    _require(stats["colinear_pairs"] >= 1, "figures sometimes share one mirror line")
    _require(stats["clutter_puzzles"] >= 15 and stats["heavy_pairs"] >= 3, "clutter density varies")
    _require(stats["clutter_overwrite"] >= 15 and stats["two_sided"] >= 30 and stats["off_centre"] >= 30,
             "damage variety")
    _require(stats["large"] >= 5 and stats["far_gap"] >= 10, "large figures and far pins")
    _require(stats["intact"] >= 1 and stats["multipart"] >= 3 and stats["trap"] >= 5,
             "intact, multi-part and apparent-symmetry figures")
    _require(stats["test_reuses_pin_color"] >= 3, "some tests reuse a training pin color as a figure color")
    return {k: (sorted(v) if isinstance(v, set) and len(v) <= 12 else len(v) if isinstance(v, set) else v)
            for k, v in stats.items()}


if __name__ == "__main__":
    print(self_test())
