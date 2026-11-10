"""Composed ARC family: lonely cells pair up or vanish.

Parent A (070dd51e) joins two same-coloured endpoints that share a row or a
column with an inclusive segment; at a crossing the vertical segment stays on
top.  Parent B (42a50994) deletes every signal cell with no signal cell among
its eight neighbours and keeps everything that touches something, orthogonally
or diagonally.

Composed rule (derived from the input alone):

1. The background is the uniquely most frequent input colour.
2. A non-background cell is *lonely* when none of its eight neighbours is
   non-background (cells outside the grid count as background).
3. Two lonely cells of the same colour that share a row or a column, with only
   background strictly between them, are partners.  Fill the inclusive segment
   between partners with their colour.  All horizontal segments are drawn
   first and all vertical segments afterwards, so verticals win at crossings.
4. On the completed picture, every non-background cell that still has no
   non-background cell among its eight neighbours is erased.  Unpaired lonely
   cells therefore vanish, unless a newly drawn segment now touches them.

Clusters (eight-connected groups of two or more cells, diagonal contact
included) are never lonely and are kept unchanged.
"""
from __future__ import annotations

import numpy as np

_N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG_KINDS = ("dimer_d", "diag_chain", "zigzag", "xcross")
_CLUSTER_KINDS = ("dimer_o", "dimer_d", "diag_chain", "zigzag", "xcross", "bar",
                  "stair", "ell", "tee", "blob", "hook", "ring", "square")
_REGIMES = ("matched", "larger", "network", "smaller", "rescue", "palette")
# Shortcuts that every training set must refute.
_TRAIN_REFUTES = ("no_lines", "no_sweep", "sweep_before_lines", "four_neighbor",
                  "horizontal_on_top", "longer_on_top", "keep_line_colored_specks",
                  "exactly_two_colors", "pair_ignores_loneliness")
# Diagonal-only clusters whose same-coloured cells also face each other across a
# background gap: they refute both four-neighbour isolation and cluster pairing.
_GAP_DIAG_KINDS = ("zigzag", "xcross")
# Shortcuts that every test pair must refute (both parents are needed per test).
_TEST_REFUTES = ("no_lines", "no_sweep")


# ---------------------------------------------------------------------------
# Rule implementation (input only)
# ---------------------------------------------------------------------------

def _neighbor_sum(mask, offsets=_N8):
    m = np.asarray(mask, dtype=np.int32)
    h, w = m.shape
    padded = np.pad(m, 1)
    total = np.zeros((h, w), dtype=np.int32)
    for dr, dc in offsets:
        total += padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return total


def _background(grid):
    counts = np.bincount(np.asarray(grid).ravel(), minlength=10)
    return int(np.argmax(counts))


def _lonely(grid, bg, offsets=_N8):
    nonbg = grid != bg
    return nonbg & (_neighbor_sum(nonbg, offsets) == 0)


def _find_pairs(grid, bg, offsets=_N8):
    """Partners: same-coloured lonely cells in one row/column, only background between."""
    lone = _lonely(grid, bg, offsets)
    h, w = grid.shape
    pairs = []
    for r in range(h):
        groups = {}
        for j in np.flatnonzero(lone[r]):
            groups.setdefault(int(grid[r, j]), []).append(int(j))
        for colour, js in sorted(groups.items()):
            for a, b in zip(js, js[1:]):
                if np.all(grid[r, a + 1:b] == bg):
                    pairs.append(("h", colour, r, a, b))
    for j in range(w):
        groups = {}
        for r in np.flatnonzero(lone[:, j]):
            groups.setdefault(int(grid[r, j]), []).append(int(r))
        for colour, rs in sorted(groups.items()):
            for a, b in zip(rs, rs[1:]):
                if np.all(grid[a + 1:b, j] == bg):
                    pairs.append(("v", colour, j, a, b))
    return pairs


def _find_pairs_any(grid, bg):
    """Shortcut pairing: any same-coloured cells facing each other across background."""
    h, w = grid.shape
    pairs = []
    for r in range(h):
        groups = {}
        for j in np.flatnonzero(grid[r] != bg):
            groups.setdefault(int(grid[r, j]), []).append(int(j))
        for colour, js in sorted(groups.items()):
            for a, b in zip(js, js[1:]):
                if b - a >= 2 and np.all(grid[r, a + 1:b] == bg):
                    pairs.append(("h", colour, r, a, b))
    for j in range(w):
        groups = {}
        for r in np.flatnonzero(grid[:, j] != bg):
            groups.setdefault(int(grid[r, j]), []).append(int(r))
        for colour, rs in sorted(groups.items()):
            for a, b in zip(rs, rs[1:]):
                if b - a >= 2 and np.all(grid[a + 1:b, j] == bg):
                    pairs.append(("v", colour, j, a, b))
    return pairs


def _draw(grid, pairs, order="vertical_on_top"):
    out = grid.copy()
    if order == "vertical_on_top":
        seq = [p for p in pairs if p[0] == "h"] + [p for p in pairs if p[0] == "v"]
    elif order == "horizontal_on_top":
        seq = [p for p in pairs if p[0] == "v"] + [p for p in pairs if p[0] == "h"]
    elif order == "longer_on_top":
        seq = sorted(pairs, key=lambda p: (p[4] - p[3], p[0] == "v"))
    else:  # pragma: no cover - internal misuse
        raise ValueError(order)
    for orient, colour, fixed, lo, hi in seq:
        if orient == "h":
            out[fixed, lo:hi + 1] = colour
        elif orient == "v":
            out[lo:hi + 1, fixed] = colour
        else:  # pragma: no cover - internal misuse
            raise ValueError(orient)
    return out


def _sweep(grid, bg, offsets=_N8):
    out = grid.copy()
    out[_lonely(grid, bg, offsets)] = bg
    return out


def solve(input_grid):
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("input must be a non-empty 2D grid")
    bg = _background(grid)
    return _sweep(_draw(grid, _find_pairs(grid, bg)), bg)


# ---------------------------------------------------------------------------
# Shortcut rules used to prove that the demonstrations are informative
# ---------------------------------------------------------------------------

def _endpoint_mask(shape, pairs):
    mask = np.zeros(shape, dtype=bool)
    for orient, _c, fixed, lo, hi in pairs:
        if orient == "h":
            mask[fixed, lo] = mask[fixed, hi] = True
        else:
            mask[lo, fixed] = mask[hi, fixed] = True
    return mask


def _diagonal_partners(grid, bg):
    lone = _lonely(grid, bg)
    cells = [(int(r), int(c)) for r, c in zip(*np.nonzero(lone))]
    found = []
    for i, (r1, c1) in enumerate(cells):
        for r2, c2 in cells[i + 1:]:
            if grid[r1, c1] != grid[r2, c2] or abs(r2 - r1) != abs(c2 - c1) or r1 == r2:
                continue
            dr = 1 if r2 > r1 else -1
            dc = 1 if c2 > c1 else -1
            steps = abs(r2 - r1)
            if all(grid[r1 + k * dr, c1 + k * dc] == bg for k in range(1, steps)):
                found.append(((r1, c1), (r2, c2), (dr, dc), steps, int(grid[r1, c1])))
    return found


def _shortcuts(grid):
    """Outputs of plausible simpler rules, each ignoring part of the composition."""
    grid = np.asarray(grid, dtype=int)
    bg = _background(grid)
    pairs = _find_pairs(grid, bg)
    drawn = _draw(grid, pairs)
    expected = _sweep(drawn, bg)
    lone_in = _lonely(grid, bg)
    endpoints = _endpoint_mask(grid.shape, pairs)
    alts = {"no_lines": _sweep(grid, bg), "no_sweep": drawn}
    early = drawn.copy()
    early[lone_in & ~endpoints] = bg
    alts["sweep_before_lines"] = early
    p4 = _find_pairs(grid, bg, _N4)
    alts["four_neighbor"] = _sweep(_draw(grid, p4), bg, _N4)
    alts["horizontal_on_top"] = _sweep(_draw(grid, pairs, "horizontal_on_top"), bg)
    alts["longer_on_top"] = _sweep(_draw(grid, pairs, "longer_on_top"), bg)
    line_colours = {p[1] for p in pairs}
    keep = expected.copy()
    removed = lone_in & ~endpoints & (expected == bg)
    for r, c in zip(*np.nonzero(removed)):
        if int(grid[r, c]) in line_colours:
            keep[r, c] = grid[r, c]
    alts["keep_line_colored_specks"] = keep
    counts = np.bincount(grid.ravel(), minlength=10)
    two = [p for p in pairs if counts[p[1]] == 2]
    alts["exactly_two_colors"] = _sweep(_draw(grid, two), bg)
    # Clusters also emit segments: pairing ignores B's loneliness test.
    alts["pair_ignores_loneliness"] = _sweep(_draw(grid, _find_pairs_any(grid, bg)), bg)
    # Diagonal partners are joined too (eight-direction reading of the pairing).
    diag = drawn.copy()
    for (r1, c1), _end, (dr, dc), steps, colour in _diagonal_partners(grid, bg):
        for k in range(steps + 1):
            if drawn[r1 + k * dr, c1 + k * dc] == bg:
                diag[r1 + k * dr, c1 + k * dc] = colour
    alts["diagonal_lines"] = _sweep(diag, bg)
    # Specks survive only when touching a line of their own colour.
    same = expected.copy()
    rescued = lone_in & ~endpoints & (expected != bg)
    for r, c in zip(*np.nonzero(rescued)):
        colour = grid[r, c]
        r0, r1, c0, c1 = max(0, r - 1), min(grid.shape[0], r + 2), max(0, c - 1), min(grid.shape[1], c + 2)
        window = drawn[r0:r1, c0:c1]
        own = (window == colour) & (grid[r0:r1, c0:c1] == bg)
        if not np.any(own):
            same[r, c] = bg
    alts["rescue_same_color_only"] = same
    # Specks survive when a line passes within two cells (Chebyshev radius 2).
    near = expected.copy()
    line_cells = (drawn != bg) & (grid == bg) | endpoints
    padded = np.pad(line_cells, 2)
    for r, c in zip(*np.nonzero(lone_in & ~endpoints & (expected == bg))):
        if np.any(padded[r:r + 5, c:c + 5]):
            near[r, c] = grid[r, c]
    alts["rescue_radius_two"] = near
    return alts


# ---------------------------------------------------------------------------
# Analysis and validation
# ---------------------------------------------------------------------------

def _as_grid(value, name):
    array = np.asarray(value)
    if array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape):
        raise ValueError(f"{name} must be a 2D grid with sides 1..30")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    if np.any(array < 0) or np.any(array > 9):
        raise ValueError(f"{name} colours must be 0..9")
    return array.astype(int)


def _analyze(grid):
    """Check family invariants of one input and report its observable features."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if int(np.sum(counts == counts[bg])) != 1 or 2 * counts[bg] < grid.size:
        raise ValueError("background is not a unique majority colour")
    nonbg = grid != bg
    n8 = _neighbor_sum(nonbg)
    n4 = _neighbor_sum(nonbg, _N4)
    lone = nonbg & (n8 == 0)
    pairs = _find_pairs(grid, bg)
    if not pairs:
        raise ValueError("input has no partner pair")
    endpoints = _endpoint_mask(grid.shape, pairs)
    if int(endpoints.sum()) != 2 * len(pairs):
        raise ValueError("a lonely cell belongs to two partner pairs")
    for colour in range(10):
        if colour == bg:
            continue
        lone_c = lone & (grid == colour)
        other_c = nonbg & ~lone & (grid == colour)
        lr, lc = np.nonzero(lone_c)
        orow, ocol = np.nonzero(other_c)
        if set(lr.tolist()) & set(orow.tolist()) or set(lc.tolist()) & set(ocol.tolist()):
            raise ValueError("a lonely cell is aligned with a same-coloured cluster")
        for axis_cells in (lr, lc):
            values, freq = np.unique(axis_cells, return_counts=True)
            if np.any(freq > 2):
                raise ValueError("more than two same-coloured lonely cells share a line")
        # Two same-coloured lonely cells on one line must be partners.
        for r in set(lr.tolist()):
            js = np.flatnonzero(lone_c[r])
            if len(js) == 2 and not np.all(grid[r, js[0] + 1:js[1]] == bg):
                raise ValueError("blocked same-coloured row alignment")
        for c in set(lc.tolist()):
            rs = np.flatnonzero(lone_c[:, c])
            if len(rs) == 2 and not np.all(grid[rs[0] + 1:rs[1], c] == bg):
                raise ValueError("blocked same-coloured column alignment")
    drawn = _draw(grid, pairs)
    expected = _sweep(drawn, bg)
    specks = lone & ~endpoints
    rescued = specks & (expected != bg)
    removed = specks & (expected == bg)
    horizontals = [p for p in pairs if p[0] == "h"]
    verticals = [p for p in pairs if p[0] == "v"]
    crossings = []
    for _o, cv, col, r1, r2 in verticals:
        for _o2, ch, row, a, b in horizontals:
            if r1 < row < r2 and a < col < b:
                crossings.append((r2 - r1 + 1, b - a + 1, cv != ch))
    line_colours = {p[1] for p in pairs}
    cluster_mask = nonbg & ~lone
    # Count eight-connected clusters.
    seen = np.zeros_like(cluster_mask)
    clusters = 0
    for r, c in zip(*np.nonzero(cluster_mask)):
        if seen[r, c]:
            continue
        clusters += 1
        stack = [(int(r), int(c))]
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            for dy, dx in _N8:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and cluster_mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
    edge_lines = sum(1 for o, _c, f, _a, _b in pairs if (o == "h" and f in (0, h - 1)) or (o == "v" and f in (0, w - 1)))
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    corner_endpoints = sum(1 for r, c in zip(*np.nonzero(endpoints)) if (int(r), int(c)) in corners)
    rescued_other = 0
    for r, c in zip(*np.nonzero(rescued)):
        r0, r1, c0, c1 = max(0, r - 1), min(h, r + 2), max(0, c - 1), min(w, c + 2)
        new = (drawn[r0:r1, c0:c1] != bg) & (grid[r0:r1, c0:c1] == bg)
        if np.any(new & (drawn[r0:r1, c0:c1] != grid[r, c])):
            rescued_other += 1
    return {
        "shape": (h, w), "bg": bg, "pairs": pairs, "segments": len(pairs),
        "horizontal": len(horizontals), "vertical": len(verticals),
        "crossings": crossings, "clusters": clusters,
        "diag_only_cells": int(np.sum(cluster_mask & (n4 == 0))),
        "rescued": int(rescued.sum()), "rescued_other_colour": rescued_other,
        "removed": int(removed.sum()),
        "line_colour_decoys": int(sum(1 for r, c in zip(*np.nonzero(removed)) if int(grid[r, c]) in line_colours)),
        "diagonal_decoys": len(_diagonal_partners(grid, bg)),
        "edge_lines": edge_lines, "corner_endpoints": corner_endpoints,
        "colours": len(set(np.unique(grid).tolist()) - {bg}),
        "density": float(nonbg.mean()),
        "expected": expected,
    }


def _check_counts(num_train, num_test):
    values = []
    for value, name, lo, hi, default in ((num_train, "num_train", 2, 10, 4), (num_test, "num_test", 1, 4, 1)):
        if value is None:
            value = default
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be in {lo}..{hi}")
        values.append(value)
    if values[0] <= values[1]:
        raise ValueError("num_train must exceed num_test")
    return values


def validate_generated_puzzle(puzzle):
    """Raise ValueError unless the puzzle is a well-formed member of the family."""
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("puzzle must have exactly train/test keys")
    if not isinstance(puzzle["train"], list) or not isinstance(puzzle["test"], list):
        raise ValueError("train/test must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("pair must have exactly input/output keys")
            inp = _as_grid(pair["input"], "input")
            out = _as_grid(pair["output"], "output")
            if inp.shape != out.shape:
                raise ValueError("output shape differs from input shape")
            key = (inp.shape, inp.tobytes())
            if key in seen:
                raise ValueError("duplicate input")
            seen.add(key)
            features = _analyze(inp)
            if not np.array_equal(features["expected"], out):
                raise ValueError("output does not follow the rule")
            if not np.array_equal(solve(inp), out):
                raise ValueError("solve disagrees with output")
            features["alts"] = {k: not np.array_equal(v, out) for k, v in _shortcuts(inp).items()}
            info[split].append(features)
    train, test = info["train"], info["test"]
    for name in _TRAIN_REFUTES:
        if not any(f["alts"][name] for f in train):
            raise ValueError(f"training does not refute shortcut {name}")
    for f in test:
        for name in _TEST_REFUTES:
            if not f["alts"][name]:
                raise ValueError(f"test pair does not refute shortcut {name}")
    for name in test[0]["alts"]:
        if any(f["alts"][name] for f in test) and not any(f["alts"][name] for f in train):
            raise ValueError(f"test refutes {name} but training never does")
    orders = {(lv < lh) for lv, lh, distinct in (c for f in train for c in f["crossings"]) if distinct and lv != lh}
    if orders != {True, False}:
        raise ValueError("training must show shorter and longer vertical winners")
    if not any(f["diag_only_cells"] for f in train):
        raise ValueError("training lacks a diagonally connected cluster")
    if not any(f["rescued"] for f in train):
        raise ValueError("training lacks a speck anchored by a new segment")
    return info


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


class _Scene:
    def __init__(self, h, w, bg):
        self.h, self.w, self.bg = h, w, bg
        self.grid = np.full((h, w), bg, dtype=np.int64)
        self.path = np.zeros((h, w), dtype=bool)
        self.hpath = np.zeros((h, w), dtype=bool)
        self.vpath = np.zeros((h, w), dtype=bool)
        self.lone = np.zeros((h, w), dtype=bool)
        self.cluster = np.zeros((h, w), dtype=bool)
        self.rows = [set() for _ in range(10)]
        self.cols = [set() for _ in range(10)]
        self.lone_rows = [set() for _ in range(10)]
        self.lone_cols = [set() for _ in range(10)]
        self.diag_d = [set() for _ in range(10)]
        self.diag_a = [set() for _ in range(10)]
        self.segments = []

    def copy(self):
        other = _Scene.__new__(_Scene)
        other.h, other.w, other.bg = self.h, self.w, self.bg
        for name in ("grid", "path", "hpath", "vpath", "lone", "cluster"):
            setattr(other, name, getattr(self, name).copy())
        for name in ("rows", "cols", "lone_rows", "lone_cols", "diag_d", "diag_a"):
            setattr(other, name, [set(s) for s in getattr(self, name)])
        other.segments = list(self.segments)
        return other

    def inside(self, r, c):
        return 0 <= r < self.h and 0 <= c < self.w

    def window(self, array, r, c, radius=1):
        return array[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1]

    def lone_ok(self, r, c, colour, *, near_path=True, ignore_diag=False):
        if not self.inside(r, c) or colour == self.bg:
            return False
        if self.grid[r, c] != self.bg or self.path[r, c]:
            return False
        if np.any(self.window(self.grid, r, c) != self.bg):
            return False
        if r in self.rows[colour] or c in self.cols[colour]:
            return False
        if not ignore_diag and ((r - c) in self.diag_d[colour] or (r + c) in self.diag_a[colour]):
            return False
        if not near_path and np.any(self.window(self.path, r, c)):
            return False
        return True

    def mark_lone(self, r, c, colour):
        self.grid[r, c] = colour
        self.lone[r, c] = True
        self.rows[colour].add(r)
        self.cols[colour].add(c)
        self.lone_rows[colour].add(r)
        self.lone_cols[colour].add(c)
        self.diag_d[colour].add(r - c)
        self.diag_a[colour].add(r + c)

    def add_segment(self, orient, colour, fixed, lo, hi):
        if hi - lo < 2 or lo < 0:
            return False
        if orient == "h":
            if not (0 <= fixed < self.h and hi < self.w):
                return False
            ends = ((fixed, lo), (fixed, hi))
            if np.any(self.hpath[fixed, lo:hi + 1]) or np.any(self.grid[fixed, lo + 1:hi] != self.bg):
                return False
        else:
            if not (0 <= fixed < self.w and hi < self.h):
                return False
            ends = ((lo, fixed), (hi, fixed))
            if np.any(self.vpath[lo:hi + 1, fixed]) or np.any(self.grid[lo + 1:hi, fixed] != self.bg):
                return False
        if not all(self.lone_ok(r, c, colour) for r, c in ends):
            return False
        for r, c in ends:
            self.mark_lone(r, c, colour)
        if orient == "h":
            self.path[fixed, lo:hi + 1] = True
            self.hpath[fixed, lo:hi + 1] = True
        else:
            self.path[lo:hi + 1, fixed] = True
            self.vpath[lo:hi + 1, fixed] = True
        self.segments.append((orient, colour, fixed, lo, hi))
        return True

    def add_cluster(self, cells, colour):
        if colour == self.bg:
            return False
        mask = np.zeros((self.h, self.w), dtype=bool)
        for r, c in cells:
            if not self.inside(r, c) or self.grid[r, c] != self.bg or self.path[r, c]:
                return False
            mask[r, c] = True
        rows = {r for r, _ in cells}
        cols = {c for _, c in cells}
        if rows & self.lone_rows[colour] or cols & self.lone_cols[colour]:
            return False
        halo = _neighbor_sum(mask) > 0
        if np.any(halo & (self.lone | self.cluster)):
            return False
        self.grid[mask] = colour
        self.cluster |= mask
        self.rows[colour] |= rows
        self.cols[colour] |= cols
        return True


def _transform(cells, turns, flip):
    pts = list(cells)
    for _ in range(turns):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(r, -c) for r, c in pts]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return sorted({(r - r0, c - c0) for r, c in pts})


def _shape(rng, kind):
    if kind == "dimer_o":
        cells = [(0, 0), (0, 1)]
    elif kind == "dimer_d":
        cells = [(0, 0), (1, 1)]
    elif kind == "diag_chain":
        cells = [(i, i) for i in range(_ri(rng, 3, 5))]
    elif kind == "zigzag":
        cells = [(i % 2, i) for i in range(_ri(rng, 3, 6))]
    elif kind == "xcross":
        cells = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0)]
    elif kind == "bar":
        cells = [(0, i) for i in range(_ri(rng, 3, 6))]
    elif kind == "stair":
        width = _ri(rng, 2, 3)
        cells = [(s, s * width + k) for s in range(_ri(rng, 2, 3)) for k in range(width)]
    elif kind == "ell":
        a, b = _ri(rng, 2, 4), _ri(rng, 1, 3)
        cells = [(i, 0) for i in range(a)] + [(a - 1, k) for k in range(1, b + 1)]
    elif kind == "tee":
        arm = _ri(rng, 1, 2)
        cells = [(0, k) for k in range(2 * arm + 1)] + [(i, arm) for i in range(1, _ri(rng, 2, 3))]
    elif kind == "square":
        a, b = _ri(rng, 2, 3), _ri(rng, 2, 3)
        cells = [(i, k) for i in range(a) for k in range(b)]
    elif kind == "ring":
        a, b = _ri(rng, 3, 7), _ri(rng, 3, 7)
        cells = [(i, k) for i in range(a) for k in range(b) if i in (0, a - 1) or k in (0, b - 1)]
    elif kind == "hook":
        cells = [(0, 0)]
        n = _ri(rng, 3, 7)
        guard = 0
        while len(cells) < n and guard < 50:
            guard += 1
            dr, dc = _pick(rng, _N8)
            r, c = cells[-1][0] + dr, cells[-1][1] + dc
            if (r, c) in cells:
                continue
            touching = sum(1 for y, x in cells[:-1] if abs(y - r) <= 1 and abs(x - c) <= 1)
            if touching == 0:
                cells.append((r, c))
    else:  # blob
        cells = {(0, 0)}
        n = _ri(rng, 3, 8)
        while len(cells) < n:
            r, c = _pick(rng, sorted(cells))
            dr, dc = _pick(rng, _N8)
            cells.add((r + dr, c + dc))
        cells = sorted(cells)
    return _transform(cells, _ri(rng, 0, 3), bool(rng.integers(2)))


def _place_cluster(rng, scene, colours, kinds, hole_prob=0.0):
    for _ in range(12):
        kind = _pick(rng, kinds)
        shape = _shape(rng, kind)
        sh = max(r for r, _ in shape) + 1
        sw = max(c for _, c in shape) + 1
        if sh > scene.h or sw > scene.w:
            continue
        for _ in range(10):
            top, left = _ri(rng, 0, scene.h - sh), _ri(rng, 0, scene.w - sw)
            colour = _pick(rng, colours)
            cells = [(top + r, left + c) for r, c in shape]
            if scene.add_cluster(cells, colour):
                if kind == "ring" and sh >= 5 and sw >= 5 and rng.random() < hole_prob:
                    for _ in range(6):
                        rr, cc = top + _ri(rng, 2, sh - 3), left + _ri(rng, 2, sw - 3)
                        speck = _pick(rng, colours)
                        if scene.lone_ok(rr, cc, speck, near_path=False):
                            scene.mark_lone(rr, cc, speck)
                            break
                return kind
    return None


def _segment_length(rng, n, lo=3):
    if n < lo:
        return None
    span = n - lo
    return lo + int(round(span * rng.beta(1.2, 1.4)))


def _place_random_segment(rng, scene, colours, edge_bias, min_len=3):
    for _ in range(30):
        orient = "h" if rng.random() < 0.5 else "v"
        n, m = (scene.w, scene.h) if orient == "h" else (scene.h, scene.w)
        length = _segment_length(rng, n, min_len)
        if length is None:
            continue
        lo = _ri(rng, 0, n - length)
        fixed = _pick(rng, (0, m - 1)) if rng.random() < edge_bias else _ri(rng, 0, m - 1)
        if scene.add_segment(orient, _pick(rng, colours), fixed, lo, lo + length - 1):
            return True
    return False


def _place_crossing(rng, scene, colours, order):
    """One vertical crossing one horizontal; order is 'short', 'long' or 'any'."""
    h, w = scene.h, scene.w
    if len(colours) < 2:
        return False
    for _ in range(60):
        if order == "short":
            if w < 4:
                return False
            lv = _ri(rng, 3, min(h, w - 1))
            lh = _ri(rng, lv + 1, w)
        elif order == "long":
            if h < 4:
                return False
            lh = _ri(rng, 3, min(w, h - 1))
            lv = _ri(rng, lh + 1, h)
        else:
            lv, lh = _ri(rng, 3, h), _ri(rng, 3, w)
        r1 = _ri(rng, 0, h - lv)
        a = _ri(rng, 0, w - lh)
        row = _ri(rng, r1 + 1, r1 + lv - 2)
        col = _ri(rng, a + 1, a + lh - 2)
        cv = _pick(rng, colours)
        ch = _pick(rng, [c for c in colours if c != cv])
        trial = scene.copy()
        if trial.add_segment("h", ch, row, a, a + lh - 1) and trial.add_segment("v", cv, col, r1, r1 + lv - 1):
            scene.__dict__.update(trial.__dict__)
            return True
    return False


def _place_network(rng, scene, colours, kind):
    """A fan (one line crossing several) or a grid of crossing lines."""
    h, w = scene.h, scene.w
    for _ in range(25):
        trial = scene.copy()
        if kind == "fan":
            many = _ri(rng, 2, 4)
            nv, nh = (1, many) if rng.random() < 0.5 else (many, 1)
        else:
            nv, nh = _ri(rng, 2, 3), _ri(rng, 2, 3)
        if h < 2 * nh + 2 or w < 2 * nv + 2:
            return False
        rows = sorted(int(x) for x in rng.choice(np.arange(1, h - 1), nh, replace=False))
        cols = sorted(int(x) for x in rng.choice(np.arange(1, w - 1), nv, replace=False))
        # Parallel lines of one network keep a background gap so each stays legible.
        if any(b - a < 2 for a, b in zip(rows, rows[1:])) or any(b - a < 2 for a, b in zip(cols, cols[1:])):
            continue
        ok = True
        for col in cols:
            lo, hi = _ri(rng, 0, rows[0] - 1), _ri(rng, rows[-1] + 1, h - 1)
            ok = ok and trial.add_segment("v", _pick(rng, colours), col, lo, hi)
        for row in rows:
            lo, hi = _ri(rng, 0, cols[0] - 1), _ri(rng, cols[-1] + 1, w - 1)
            ok = ok and trial.add_segment("h", _pick(rng, colours), row, lo, hi)
        if ok:
            scene.__dict__.update(trial.__dict__)
            return True
    return False


def _place_rescue(rng, scene, colours, other_colour=False):
    long_segments = [s for s in scene.segments if s[4] - s[3] >= 4]
    if not long_segments:
        return False
    for _ in range(25):
        orient, line_colour, fixed, lo, hi = _pick(rng, long_segments)
        t = _ri(rng, lo + 2, hi - 2)
        side = _pick(rng, (-1, 1))
        r, c = (fixed + side, t) if orient == "h" else (t, fixed + side)
        options = [x for x in colours if x != line_colour] if other_colour else list(colours)
        if not options:
            return False
        colour = _pick(rng, options)
        if scene.lone_ok(r, c, colour):
            scene.mark_lone(r, c, colour)
            return True
    return False


def _place_near_miss(rng, scene, colours):
    if not scene.segments:
        return False
    for _ in range(25):
        orient, _lc, fixed, lo, hi = _pick(rng, scene.segments)
        t = _ri(rng, lo, hi)
        side = _pick(rng, (-2, 2))
        shift = _ri(rng, -1, 1)
        r, c = (fixed + side, t + shift) if orient == "h" else (t + shift, fixed + side)
        colour = _pick(rng, colours)
        if scene.lone_ok(r, c, colour, near_path=False):
            scene.mark_lone(r, c, colour)
            return True
    return False


def _place_free(rng, scene, colours):
    for _ in range(40):
        r, c = _ri(rng, 0, scene.h - 1), _ri(rng, 0, scene.w - 1)
        colour = _pick(rng, colours)
        if scene.lone_ok(r, c, colour, near_path=False):
            scene.mark_lone(r, c, colour)
            return True
    return False


def _place_decoy(rng, scene):
    """A speck sharing a line colour but aligned with nothing of that colour."""
    line_colours = sorted({s[1] for s in scene.segments})
    if not line_colours:
        return False
    for _ in range(60):
        r, c = _ri(rng, 0, scene.h - 1), _ri(rng, 0, scene.w - 1)
        colour = _pick(rng, line_colours)
        if scene.lone_ok(r, c, colour, near_path=False):
            scene.mark_lone(r, c, colour)
            return True
    return False


def _place_diagonal_decoy(rng, scene, colours):
    """Two same-coloured specks on a clear diagonal: diagonal alignment never pairs."""
    for _ in range(80):
        steps = _ri(rng, 2, 5)
        dr, dc = 1, _pick(rng, (-1, 1))
        r, c = _ri(rng, 0, scene.h - 1), _ri(rng, 0, scene.w - 1)
        r2, c2 = r + steps * dr, c + steps * dc
        colour = _pick(rng, colours)
        if not scene.inside(r2, c2):
            continue
        if not (scene.lone_ok(r, c, colour, near_path=False) and scene.lone_ok(r2, c2, colour, near_path=False)):
            continue
        between = [(r + k * dr, c + k * dc) for k in range(1, steps)]
        if any(scene.grid[y, x] != scene.bg or scene.path[y, x] for y, x in between):
            continue
        scene.mark_lone(r, c, colour)
        scene.mark_lone(r2, c2, colour)
        return True
    return False


def _build_pair(rng, plan):
    """Return an input grid for the plan, or None if placement failed."""
    for _ in range(25):
        scene = _Scene(plan["h"], plan["w"], plan["bg"])
        colours = plan["colours"]
        needs = plan["needs"]
        ok = True
        if "cross_short" in needs:
            ok = ok and _place_crossing(rng, scene, colours, "short")
        if "cross_long" in needs:
            ok = ok and _place_crossing(rng, scene, colours, "long")
        if not ok:
            continue
        style = plan["style"]
        if style in ("fan", "network"):
            if not _place_network(rng, scene, colours, style) and "network" in needs:
                continue
        elif style == "single":
            _place_crossing(rng, scene, colours, _pick(rng, ("short", "long", "any")))
        for _ in range(plan["n_seg"]):
            _place_random_segment(rng, scene, colours, plan["edge_bias"])
        if not scene.segments:
            _place_random_segment(rng, scene, colours, plan["edge_bias"])
        if "rescue" in needs and not any(s[4] - s[3] >= 4 for s in scene.segments):
            _place_random_segment(rng, scene, colours, plan["edge_bias"], min_len=5)
        if not scene.segments:
            continue
        if "diag_cluster" in needs and not _place_cluster(rng, scene, colours, _GAP_DIAG_KINDS):
            continue
        for _ in range(plan["n_clusters"]):
            _place_cluster(rng, scene, colours, plan["kinds"], plan["hole_prob"])
        if "rescue" in needs and not _place_rescue(rng, scene, colours, other_colour=True):
            continue
        for _ in range(plan["n_rescue"]):
            _place_rescue(rng, scene, colours)
        if "decoy" in needs and not _place_decoy(rng, scene):
            continue
        for _ in range(plan["n_decoy"]):
            _place_decoy(rng, scene)
        if "diag_decoy" in needs and not _place_diagonal_decoy(rng, scene, colours):
            continue
        if "near" in needs and not _place_near_miss(rng, scene, colours):
            continue
        for _ in range(plan["n_near"]):
            _place_near_miss(rng, scene, colours)
        for _ in range(plan["n_free"]):
            _place_free(rng, scene, colours)
        if "dust" in needs and not _place_free(rng, scene, colours):
            continue
        grid = scene.grid.copy()
        counts = np.bincount(grid.ravel(), minlength=10)
        if 2 * counts[scene.bg] < grid.size or np.sum(counts == counts[scene.bg]) != 1:
            continue
        # The solver must recover exactly the constructed segments.
        if sorted(_find_pairs(grid, scene.bg)) != sorted(scene.segments):
            continue
        return grid
    return None


def _dims(rng, lo, hi):
    lo, hi = max(lo, 6), min(hi, 30)
    kind = _pick(rng, ("free", "free", "square", "tall", "wide"))
    if kind == "square" or hi - lo < 4:
        side = _ri(rng, lo, hi)
        return (side, side) if kind == "square" else (_ri(rng, lo, hi), _ri(rng, lo, hi))
    if kind == "free":
        return _ri(rng, lo, hi), _ri(rng, lo, hi)
    short = _ri(rng, lo, hi - 4)
    long = _ri(rng, short + 3, hi)
    return (long, short) if kind == "tall" else (short, long)


def _counts(rng, h, w, density):
    area = h * w
    scale = {"sparse": 0.6, "moderate": 1.0, "busy": 1.5}[density]
    n_seg = _ri(rng, 0, max(1, int(area * scale / 90)))
    n_clusters = _ri(rng, 0, max(1, min(8, int(area * scale / 70))))
    n_free = _ri(rng, 1, max(1, min(16, int(area * scale / 45))))
    n_rescue = _ri(rng, 0, 1 + int(scale + area / 300))
    return min(n_seg, 7), n_clusters, n_free, n_rescue


def _plan(rng, puzzle, role, needs, band=None):
    lo, hi = band or puzzle["band"]
    h, w = _dims(rng, lo, hi)
    if puzzle["palette_mode"] == "fixed" and role != "palette_test":
        bg = puzzle["bg"]
    elif role == "palette_test":
        bg = _pick(rng, [c for c in range(10) if c not in puzzle["train_bgs"]] or list(range(10)))
    else:
        bg = _ri(rng, 0, 9)
    others = [c for c in range(10) if c != bg]
    size = _ri(rng, max(3, puzzle["min_palette"]), 9)
    colours = sorted(int(c) for c in rng.choice(others, size, replace=False))
    density = puzzle["density"] if rng.random() < 0.7 else _pick(rng, ("sparse", "moderate", "busy"))
    n_seg, n_clusters, n_free, n_rescue = _counts(rng, h, w, density)
    style = _pick(rng, ("free", "free", "single", "single", "fan", "network", "none"))
    plan = {"h": h, "w": w, "bg": bg, "colours": colours, "needs": set(needs), "style": style,
            "n_seg": n_seg, "n_clusters": n_clusters, "n_free": n_free, "n_rescue": n_rescue,
            "n_decoy": _ri(rng, 0, 2) if rng.random() < 0.5 else 0,
            "n_near": _ri(rng, 0, 2), "edge_bias": _pick(rng, (0.0, 0.15, 0.35)),
            "kinds": puzzle["kinds"], "hole_prob": 0.6}
    if "network" in needs:
        plan["style"] = _pick(rng, ("fan", "network"))
    if role == "anchor":
        plan["style"] = _pick(rng, ("free", "none", "single"))
        plan["n_seg"] = min(plan["n_seg"], 3)
    return plan


def _make_puzzle(rng, num_train, num_test):
    regime = _pick(rng, _REGIMES)
    puzzle = {
        "regime": regime,
        "palette_mode": _pick(rng, ("fixed", "vary", "vary")),
        "bg": _ri(rng, 0, 9),
        "density": _pick(rng, ("sparse", "moderate", "busy")),
        "band": _pick(rng, ((8, 16), (10, 22), (12, 30), (7, 30))),
        "min_palette": _ri(rng, 3, 7),
        "kinds": tuple(sorted(set(_DIAG_KINDS) | set(
            str(k) for k in rng.choice(_CLUSTER_KINDS, _ri(rng, 4, len(_CLUSTER_KINDS)), replace=False)))),
        "train_bgs": set(),
    }
    diag_decoys = rng.random() < 0.35
    if regime == "palette":
        puzzle["palette_mode"] = "fixed"
    train_band = test_band = None
    if regime == "larger":
        train_band, test_band = (8, 16), (18, 30)
    elif regime == "smaller":
        train_band, test_band = (14, 26), (7, 12)
    orders = ["cross_short", "cross_long"]
    if rng.random() < 0.5:
        orders.reverse()
    anchor_needs = [{orders[0], "rescue", "decoy", "dust", "near"},
                    {orders[1], "diag_cluster", "dust"}]
    if rng.random() < 0.5:
        anchor_needs[1].add("rescue")
    host = _ri(rng, 0, num_train - 1)
    train_grids = []
    for index in range(num_train):
        needs = set(anchor_needs[index]) if index < 2 else set()
        if diag_decoys and index == host:
            needs.add("diag_decoy")
        role = "anchor" if index < 2 else "train"
        band = (max(10, (train_band or puzzle["band"])[0]), max(16, (train_band or puzzle["band"])[1])) \
            if index < 2 else train_band
        grid = None
        for _ in range(6):
            grid = _build_pair(rng, _plan(rng, puzzle, role, needs, band))
            if grid is not None and all(not np.array_equal(grid, g) for g in train_grids):
                break
            grid = None
        if grid is None:
            return None
        train_grids.append(grid)
        puzzle["train_bgs"].add(_background(grid))
    test_grids = []
    for index in range(num_test):
        needs = {"dust"}
        role = "test"
        if regime == "network":
            needs.add("network")
        elif regime == "rescue":
            needs.add("rescue")
        elif regime == "palette":
            role = "palette_test"
        if rng.random() < 0.5:
            needs.add(_pick(rng, ("cross_short", "cross_long")))
        if diag_decoys and rng.random() < 0.5:
            needs.add("diag_decoy")
        grid = None
        for _ in range(6):
            plan = _plan(rng, puzzle, role, needs, test_band)
            if regime == "rescue":
                plan["n_rescue"] += _ri(rng, 1, 3)
            if regime == "larger":
                plan["n_seg"] += _ri(rng, 1, 3)
            grid = _build_pair(rng, plan)
            if grid is not None and all(not np.array_equal(grid, g) for g in train_grids + test_grids):
                break
            grid = None
        if grid is None:
            return None
        test_grids.append(grid)
    order = rng.permutation(num_train)
    result = {
        "train": [{"input": train_grids[i].tolist(), "output": solve(train_grids[i]).tolist()} for i in order],
        "test": [{"input": g.tolist(), "output": solve(g).tolist()} for g in test_grids],
    }
    try:
        validate_generated_puzzle(result)
    except ValueError:
        return None
    return result


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(200):
        puzzle = _make_puzzle(rng, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError("could not generate a valid puzzle")


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def self_test():
    seeds = list(range(42))
    stats = {"shapes": set(), "bgs": set(), "segments": set(), "crossings": set(), "clusters": set(),
             "test_rescue": 0, "diag_decoy": 0, "edge": 0, "corner": 0, "multi_bg": 0,
             "test_larger": 0, "test_smaller": 0, "test_cross": 0, "colour_reuse": 0,
             "test_short_vertical": 0, "test_new_bg": 0, "test_rescue_other": 0}
    signatures = set()
    for seed in seeds:
        puzzle = generate_puzzle(seed)
        info = validate_generated_puzzle(puzzle)
        signatures.add(repr(puzzle))
        everything = info["train"] + info["test"]
        for f in everything:
            stats["shapes"].add(f["shape"])
            stats["bgs"].add(f["bg"])
            stats["segments"].add(f["segments"])
            stats["crossings"].add(len(f["crossings"]))
            stats["clusters"].add(f["clusters"])
            stats["edge"] += f["edge_lines"] > 0
            stats["corner"] += f["corner_endpoints"] > 0
            stats["colour_reuse"] += f["line_colour_decoys"] > 0
        stats["multi_bg"] += len({f["bg"] for f in everything}) > 1
        stats["test_rescue"] += any(f["rescued"] for f in info["test"])
        stats["test_cross"] += any(f["crossings"] for f in info["test"])
        stats["diag_decoy"] += any(f["diagonal_decoys"] for f in everything)
        stats["test_short_vertical"] += any(lv < lh and d for f in info["test"] for lv, lh, d in f["crossings"])
        stats["test_new_bg"] += any(f["bg"] not in {x["bg"] for x in info["train"]} for f in info["test"])
        stats["test_rescue_other"] += any(f["rescued_other_colour"] for f in info["test"])
        train_area = max(f["shape"][0] * f["shape"][1] for f in info["train"])
        train_min = min(f["shape"][0] * f["shape"][1] for f in info["train"])
        test_area = [f["shape"][0] * f["shape"][1] for f in info["test"]]
        stats["test_larger"] += any(a > train_area for a in test_area)
        stats["test_smaller"] += any(a < train_min for a in test_area)
        for pair in puzzle["train"] + puzzle["test"]:
            for split_name in ("input", "output"):
                arr = np.asarray(pair[split_name])
                assert arr.ndim == 2 and 1 <= arr.shape[0] <= 30 and 1 <= arr.shape[1] <= 30
            assert np.array_equal(solve(np.asarray(pair["input"])), np.asarray(pair["output"]))
    assert len(signatures) == len(seeds), "seeds must give distinct puzzles"
    assert generate_puzzle(7) == generate_puzzle(7), "generation must be deterministic"
    assert len(stats["shapes"]) >= 60, stats["shapes"]
    assert len(stats["bgs"]) == 10, stats["bgs"]
    assert min(stats["segments"]) == 1 and max(stats["segments"]) >= 7, stats["segments"]
    assert 0 in stats["crossings"] and max(stats["crossings"]) >= 4, stats["crossings"]
    assert 0 in stats["clusters"] and max(stats["clusters"]) >= 6, stats["clusters"]
    assert stats["test_rescue"] >= 10, stats
    assert stats["test_cross"] >= 15, stats
    assert stats["diag_decoy"] >= 6, stats
    assert stats["multi_bg"] >= 15, stats
    assert stats["test_larger"] >= 5 and stats["test_smaller"] >= 5, stats
    assert stats["test_short_vertical"] >= 10 and stats["test_new_bg"] >= 10, stats
    assert stats["test_rescue_other"] >= 8, stats
    assert stats["edge"] >= 30 and stats["corner"] >= 3, stats
    # Every legal count combination works; illegal ones raise ValueError.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(3, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {a}/{b}")
    return True


if __name__ == "__main__":
    self_test()
    print("self_test passed")
