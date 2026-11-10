"""Gravity well: scattered tokens collapse into a 2x2 placeholder.

Composition of ARC 1e0a9b12 (stable column gravity with gap removal) and
d89b689b (four diagonal singletons replace the matching corners of a solid
2x2 placeholder).

Rule.  The input has a majority background and exactly one colour whose cells
form exactly one solid 2x2 block: the placeholder (the "well").  Its centre
point splits the grid into four quadrants: a cell is *upper* when its row is at
or above the well's top row and *lower* otherwise; it is *left* when its column
is at or left of the well's left column and *right* otherwise.  Every other
non-background cell is a token (multi-cell objects, including solid 2x2 decoys
of a colour that also appears elsewhere, are just tokens).  The well is consumed
and the tokens move in two stable gravity phases aimed at its centre lines:

1. Horizontal phase.  In every row, left tokens slide right and pack gaplessly
   ending in the well's left column; right tokens slide left and pack starting
   in the well's right column.  Left-to-right order is preserved.
2. Vertical phase.  In every column, upper tokens fall and pack gaplessly
   ending in the well's top row; lower tokens rise and pack starting in the
   well's bottom row.  Top-to-bottom order is preserved.

Each quadrant becomes a gapless Young-diagram staircase anchored at its own
corner of the well.  A corner whose quadrant has no token is background.  With
exactly one token per quadrant this is d89b689b; restricted to one well column
the vertical phase is 1e0a9b12's stable compaction aimed at the well.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


QUADRANTS = ("ul", "ur", "ll", "lr")
_VERTICAL_PARTNER = {"ul": "ll", "ll": "ul", "ur": "lr", "lr": "ur"}

# d89b689b's first source pair: the one-token-per-quadrant special case.
_PARENT_B_SOURCE = (
    ("0050000000", "0000000090", "0000000000", "0000000000", "0000880000",
     "0000880000", "0000000001", "0000000000", "0200000000", "0000000000"),
    ("0000000000", "0000000000", "0000000000", "0000000000", "0000590000",
     "0000210000", "0000000000", "0000000000", "0000000000", "0000000000"),
)

# SHA-256 of the compact JSON of every parent source / selected-variation input.
_PARENT_INPUT_FINGERPRINTS = frozenset((
    "04323cc61db6e384b603ce8491f9d53c024678fba1826df94fba40be4dd286b0",
    "05247283a94a3ee1f23fd3a3fbb5844578b704f01bb45a7adf3efee528251100",
    "054f554b5bb324f1ec4485a0a0502b07a4bb9e1a91c37aed98384a58073f9fd8",
    "13a1e1fe5e5f35b068b3d5e9496df52cdcad62301aef2cd7245f1713278f6e5c",
    "141bb0b05c74e739d76aeb48d3b5cf2863a4af61593cf2eb3c22b74b9d5d9b93",
    "22e58a0c413b84da84ac5f3afffca760f30c11ad0a68ea74373506ca8adb0e2a",
    "3c06f7873955340466f36a6e4c3c1c95161ace583c40c02d05d149cb8404d518",
    "436c5832b66ea1b50a35aa16abe1ccbcc4346c4c89531838c1c3bc250c8b11d6",
    "70ab31a69bbc649f69d75d1345c8337bd5873561487ca5e6089d165d555db6d7",
    "78b2ff94b98c02bd885dbcd024c19658cf21e4b9b7d291a269a17d7e283d4293",
    "7eaa0c048e3b65391a128159ed47b3d7528d5a9caf5c166663cb50631bdd5abe",
    "83217995c3724888eabf0323d7f332b2754c961953074385747359fd1814762e",
    "9cd9389e487cc9db2a49018cec241a11482b12a430532071cb4f32b73a08d71b",
    "b21b905757b1bc108464759204c1437d59cde6079d40b995d52791a7a5b43183",
    "cc389a89422d206f92c52ff3528b8faee140f8292d6f983a8e1fcc82841fe291",
    "ea5853dd9013b32ec1af7aa1d37a6a7f58c34effd2cdf5604c057d8205bfd431",
    "f98957b1d97ef5d1ce7728ccff0fbc51571c031bb30c2918b9f15ccca545507d",
    "faa80c408d5ef161a4344eb7b36ee2927ee9ad19ceb84c85e59807c3b29f4693",
))


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _rows(lines):
    return np.array([[int(ch) for ch in line] for line in lines], dtype=int)


def _fingerprint(grid):
    data = json.dumps(np.asarray(grid, dtype=int).tolist(), separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


# --------------------------------------------------------------------------
# Reading an input grid
# --------------------------------------------------------------------------

def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    if len(order) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("background colour is ambiguous")
    return int(colors[order[0]])


def _find_well(grid, background):
    """Return (colour, top, left) of the unique colour that is exactly a solid 2x2."""
    found = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        rows, cols = np.nonzero(grid == color)
        if (rows.size == 4 and rows.max() - rows.min() == 1
                and cols.max() - cols.min() == 1):
            found.append((color, int(rows.min()), int(cols.min())))
    if len(found) != 1:
        raise ValueError("expected exactly one solid 2x2 placeholder colour")
    return found[0]


def _read(grid):
    grid = np.asarray(grid, dtype=int)
    background = _background(grid)
    well, top, left = _find_well(grid, background)
    cells = np.argwhere((grid != background) & (grid != well))
    tokens = [(int(r), int(c), int(grid[r, c])) for r, c in cells]
    return background, well, top, left, tokens


def _quadrant(row, col, top, left):
    return ("u" if row <= top else "l") + ("l" if col <= left else "r")


# --------------------------------------------------------------------------
# The transformation
# --------------------------------------------------------------------------

def _horizontal(pos, top, left):
    groups = {}
    for i, (r, c) in enumerate(pos):
        groups.setdefault((r, c <= left), []).append(i)
    for (_, is_left), members in groups.items():
        members.sort(key=lambda i: pos[i][1])
        n = len(members)
        for k, i in enumerate(members):
            pos[i][1] = left - (n - 1 - k) if is_left else left + 1 + k


def _vertical(pos, top, left):
    groups = {}
    for i, (r, c) in enumerate(pos):
        groups.setdefault((c, r <= top), []).append(i)
    for (_, is_upper), members in groups.items():
        members.sort(key=lambda i: pos[i][0])
        n = len(members)
        for k, i in enumerate(members):
            pos[i][0] = top - (n - 1 - k) if is_upper else top + 1 + k


def _pack(tokens, top, left, order="hv"):
    """Final (row, col) of every token; order 'hv' is the family rule."""
    pos = [[r, c] for r, c, _ in tokens]
    for phase in order:
        (_horizontal if phase == "h" else _vertical)(pos, top, left)
    return [(p[0], p[1]) for p in pos]


def _render(shape, background, tokens, places):
    out = np.full(shape, background, dtype=int)
    for (_, _, color), (r, c) in zip(tokens, places):
        out[r, c] = color
    return out


def solve(input_grid):
    grid = np.asarray(input_grid, dtype=int)
    background, _, top, left, tokens = _read(grid)
    return _render(grid.shape, background, tokens, _pack(tokens, top, left, "hv"))


# --------------------------------------------------------------------------
# Independent cell-by-cell simulation and shortcut answers (validation only)
# --------------------------------------------------------------------------

def _simulate(grid):
    """Move tokens one cell at a time until each phase is stable."""
    grid = np.asarray(grid, dtype=int)
    background, well, top, left, _ = _read(grid)
    g = grid.copy()
    g[g == well] = background
    height, width = g.shape

    def settle(steps):
        moved = True
        while moved:
            moved = False
            for r in range(height):
                for c in range(width):
                    if g[r, c] == background:
                        continue
                    dr, dc = steps(r, c)
                    if (dr, dc) == (0, 0):
                        continue
                    nr, nc = r + dr, c + dc
                    if g[nr, nc] == background:
                        g[nr, nc], g[r, c] = g[r, c], background
                        moved = True

    settle(lambda r, c: (0, 1) if c < left else (0, -1) if c > left + 1 else (0, 0))
    settle(lambda r, c: (1, 0) if r < top else (-1, 0) if r > top + 1 else (0, 0))
    return g


def _shortcut_vertical_first(grid):
    background, _, top, left, tokens = _read(grid)
    return _render(np.shape(grid), background, tokens, _pack(tokens, top, left, "vh"))


def _shortcut_floor_gravity(grid):
    """1e0a9b12 alone: everything (well included) falls to the bottom edge."""
    grid = np.asarray(grid, dtype=int)
    background = _background(grid)
    out = np.full_like(grid, background)
    for c in range(grid.shape[1]):
        seq = grid[:, c][grid[:, c] != background]
        if seq.size:
            out[grid.shape[0] - seq.size:, c] = seq
    return out


def _shortcut_corner_only(grid):
    """d89b689b alone: each quadrant's nearest token fills its well corner."""
    grid = np.asarray(grid, dtype=int)
    background, _, top, left, tokens = _read(grid)
    out = np.full(grid.shape, background, dtype=int)
    best = {}
    for r, c, color in tokens:
        q = _quadrant(r, c, top, left)
        cr, cc = top + (q[0] == "l"), left + (q[1] == "r")
        key = (abs(r - cr) + abs(c - cc), r, c)
        if q not in best or key < best[q][0]:
            best[q] = (key, color, cr, cc)
    for _, color, cr, cc in best.values():
        out[cr, cc] = color
    return out


def _shortcut_column_only(grid):
    """Every token keeps its column and only falls toward the well's rows."""
    background, _, top, left, tokens = _read(grid)
    return _render(np.shape(grid), background, tokens, _pack(tokens, top, left, "v"))


def _shortcut_well_columns(grid):
    """Every token lands in the well column on its side (no row packing)."""
    background, _, top, left, tokens = _read(grid)
    groups = {}
    for i, (r, c, _) in enumerate(tokens):
        groups.setdefault(_quadrant(r, c, top, left), []).append(i)
    places = [None] * len(tokens)
    for q, members in groups.items():
        members.sort(key=lambda i: (tokens[i][0], tokens[i][1]))
        col = left + (q[1] == "r")
        n = len(members)
        for k, i in enumerate(members):
            places[i] = (top - (n - 1 - k), col) if q[0] == "u" else (top + 1 + k, col)
    if any(not 0 <= r < np.shape(grid)[0] for r, _ in places):
        return None
    return _render(np.shape(grid), background, tokens, places)


def _rigid_rows(tokens, top, left, sort_by_length):
    """Horizontal phase, then whole packed rows move rigidly toward the well."""
    pos = [[r, c] for r, c, _ in tokens]
    _horizontal(pos, top, left)
    groups = {}
    for i, (r, c) in enumerate(pos):
        groups.setdefault((c <= left, r <= top), []).append(i)
    for (_, upper), members in groups.items():
        rows = sorted({pos[i][0] for i in members})
        if sort_by_length:
            length = {r: sum(1 for i in members if pos[i][0] == r) for r in rows}
            rows.sort(key=lambda r: (length[r], r) if upper else (-length[r], r))
        n = len(rows)
        target = {r: (top - (n - 1 - k) if upper else top + 1 + k) for k, r in enumerate(rows)}
        for i in members:
            pos[i][0] = target[pos[i][0]]
    return [(p[0], p[1]) for p in pos]


def _shortcut_row_block(grid):
    """Pack rows, then slide each packed row rigidly toward the well."""
    background, _, top, left, tokens = _read(grid)
    return _render(np.shape(grid), background, tokens, _rigid_rows(tokens, top, left, False))


def _shortcut_sort_rows(grid):
    """Pack rows, then restack rows rigidly with the longest nearest the well."""
    background, _, top, left, tokens = _read(grid)
    return _render(np.shape(grid), background, tokens, _rigid_rows(tokens, top, left, True))


def _shortcut_column_block(grid):
    """Pack columns first, then slide each packed column rigidly sideways."""
    background, _, top, left, tokens = _read(grid)
    pos = [[r, c] for r, c, _ in tokens]
    _vertical(pos, top, left)
    groups = {}
    for i, (r, c) in enumerate(pos):
        groups.setdefault((c <= left, r <= top), []).append(i)
    for (is_left, _), members in groups.items():
        cols = sorted({pos[i][1] for i in members})
        n = len(cols)
        target = {c: (left - (n - 1 - k) if is_left else left + 1 + k) for k, c in enumerate(cols)}
        for i in members:
            pos[i][1] = target[pos[i][1]]
    return _render(np.shape(grid), background, tokens, [(p[0], p[1]) for p in pos])


# The first four must fail on every test; all must fail on some training pair.
_SHORTCUTS = (_shortcut_floor_gravity, _shortcut_corner_only, _shortcut_column_only,
              _shortcut_vertical_first, _shortcut_well_columns, _shortcut_row_block,
              _shortcut_sort_rows, _shortcut_column_block)
_TEST_SHORTCUTS = _SHORTCUTS[:4]


def _refutes(grid, output, shortcut):
    guess = shortcut(grid)
    return guess is None or not np.array_equal(guess, output)


# --------------------------------------------------------------------------
# Structural features shared by generation and validation
# --------------------------------------------------------------------------

def _canonical(cells):
    best = None
    for code in range(8):
        pts = []
        for r, c in cells:
            if code & 1:
                r, c = c, r
            if code & 2:
                r = -r
            if code & 4:
                c = -c
            pts.append((r, c))
        mr = min(p[0] for p in pts)
        mc = min(p[1] for p in pts)
        key = tuple(sorted((r - mr, c - mc) for r, c in pts))
        if best is None or key < best:
            best = key
    return best


def _components(cells, linked):
    seen, comps = set(), []
    for start in sorted(cells):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for nb in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if nb in cells and nb not in seen and linked(cells[nb], cells[(r, c)]):
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _mono_squares(grid, background):
    a = grid[:-1, :-1]
    same = (a == grid[1:, :-1]) & (a == grid[:-1, 1:]) & (a == grid[1:, 1:])
    return int(np.count_nonzero(same & (a != background)))


def _aspect(height, width):
    if height >= 1.5 * width:
        return "tall"
    if width >= 1.5 * height:
        return "wide"
    return "square"


def _sector(pos, size):
    frac = (pos + 1) / size
    return 0 if frac < 1 / 3 else (1 if frac <= 2 / 3 else 2)


def _features(grid):
    grid = np.asarray(grid, dtype=int)
    background, well, top, left, tokens = _read(grid)
    height, width = grid.shape
    places = _pack(tokens, top, left, "hv")
    output = _render(grid.shape, background, tokens, places)
    vertical_first = _shortcut_vertical_first(grid)
    quads = {q: 0 for q in QUADRANTS}
    row_groups, col_groups = {}, {}
    row_band = col_band = 0
    for (r, c, color), (fr, fc) in zip(tokens, places):
        quads[_quadrant(r, c, top, left)] += 1
        row_groups.setdefault((r, c <= left), []).append((c, color))
        col_groups.setdefault((fc, fr <= top), []).append((fr, color))
        if r in (top, top + 1):
            row_band += 1
        if c in (left, left + 1):
            col_band += 1

    def ordered(groups):
        for members in groups.values():
            seq = [color for _, color in sorted(members)]
            if len(seq) >= 2 and seq != seq[::-1]:
                return True
        return False

    empties = frozenset(q for q, n in quads.items() if n == 0)
    visible_gap = any(_VERTICAL_PARTNER[q] not in empties for q in empties)
    color_at = {(r, c): color for r, c, color in tokens}
    move = {(r, c): (fr - r, fc - c) for (r, c, _), (fr, fc) in zip(tokens, places)}
    broken = multi = False
    for comp in _components(color_at, lambda a, b: a == b):
        if len(comp) >= 2:
            multi = True
            if len({move[p] for p in comp}) > 1:
                broken = True
    shapes, straddle = set(), False
    for comp in _components(color_at, lambda a, b: True):
        if len(comp) >= 3:
            shapes.add(_canonical(comp))
        if len(comp) >= 2 and len({_quadrant(r, c, top, left) for r, c in comp}) > 1:
            straddle = True
    edge = top == 0 or left == 0 or top == height - 2 or left == width - 2
    return {
        "background": background, "well": well, "top": top, "left": left,
        "shape": grid.shape, "area": int(grid.size), "count": len(tokens), "quads": quads,
        "max_quad": max(quads.values()), "empties": empties,
        "visible_gap": visible_gap, "row_band": row_band, "col_band": col_band,
        "row_stack": any(len(m) >= 2 for m in row_groups.values()),
        "row_order": ordered(row_groups), "col_order": ordered(col_groups),
        "dphase": int(np.count_nonzero(output != vertical_first)),
        "phase_matters": not np.array_equal(output, vertical_first),
        "inversion": _refutes(grid, output, _shortcut_sort_rows),
        "broken_object": broken, "multi_object": multi,
        "shapes": frozenset(shapes), "straddle": straddle,
        "colors": len({color for _, _, color in tokens}),
        "token_colors": frozenset(color for _, _, color in tokens),
        "edge": edge, "decoys": _mono_squares(grid, background) - 1,
        "aspect": _aspect(height, width),
        "sector": (_sector(top, height), _sector(left, width)),
    }


def _big_test(f, train_feats):
    """A test that is larger than every training grid and has 150+ cells."""
    return f["area"] >= 150 and f["area"] > max(t["area"] for t in train_feats)


def _big_content_ok(f):
    return (f["count"] >= max(10, int(np.ceil(0.03 * f["area"])))
            and f["max_quad"] >= 3 and f["dphase"] >= 4)


def _shift_reasons(train_feats, f):
    """Measured train/test differences of one test pair."""
    reasons = set()
    if f["max_quad"] > max(t["max_quad"] for t in train_feats):
        reasons.add("denser")
    if f["area"] > max(t["area"] for t in train_feats):
        reasons.add("larger")
    if f["row_band"] + f["col_band"] > max(t["row_band"] + t["col_band"] for t in train_feats):
        reasons.add("band")
    train_empties = set().union(*(t["empties"] for t in train_feats))
    if f["empties"] - train_empties or len(f["empties"]) > max(len(t["empties"]) for t in train_feats):
        reasons.add("empty")
    if (f["well"] not in {t["well"] for t in train_feats}
            or f["background"] not in {t["background"] for t in train_feats}
            or f["token_colors"] - set().union(*(t["token_colors"] for t in train_feats))):
        reasons.add("colour")
    if (f["aspect"] not in {t["aspect"] for t in train_feats}
            or f["sector"] not in {t["sector"] for t in train_feats}):
        reasons.add("geometry")
    if f["shapes"] - set().union(*(t["shapes"] for t in train_feats)):
        reasons.add("object")
    return reasons


# --------------------------------------------------------------------------
# Generation: puzzle-level parameters
# --------------------------------------------------------------------------

_SIZE_RANGES = {
    "tiny": ((4, 5), (4, 6)),
    "compact": ((4, 9), (4, 9)),
    "medium": ((8, 15), (8, 15)),
    "large": ((14, 24), (14, 24)),
    "tall": ((14, 30), (4, 10)),
    "wide": ((4, 10), (14, 30)),
}
_SIZE_MODES = {
    "compact": (("compact", "medium", "tiny"), (0.55, 0.33, 0.12)),
    "medium": (("medium", "compact"), (0.8, 0.2)),
    "mixed": (("compact", "medium", "large", "tall", "wide", "tiny"), (0.2, 0.28, 0.2, 0.13, 0.13, 0.06)),
    "stretched": (("tall", "wide", "medium"), (0.4, 0.4, 0.2)),
    "big": (("medium", "large"), (0.45, 0.55)),
}


def _orientations(base):
    out = set()
    for code in range(8):
        pts = []
        for r, c in base:
            if code & 1:
                r, c = c, r
            if code & 2:
                r = -r
            if code & 4:
                c = -c
            pts.append((r, c))
        mr = min(p[0] for p in pts)
        mc = min(p[1] for p in pts)
        out.add(tuple(sorted((r - mr, c - mc) for r, c in pts)))
    return sorted(out)


_OBJECT_SHAPES = {
    "bar": [tuple((0, k) for k in range(n)) for n in (2, 3, 4, 5)]
           + [tuple((k, 0) for k in range(n)) for n in (2, 3, 4, 5)],
    "l3": _orientations(((0, 0), (1, 0), (1, 1))),
    "l4": _orientations(((0, 0), (1, 0), (2, 0), (2, 1))),
    "t4": _orientations(((0, 0), (0, 1), (0, 2), (1, 1))),
    "s4": _orientations(((0, 0), (0, 1), (1, 1), (1, 2))),
    "plus": [((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))],
    "diag": [((0, 0), (1, 1)), ((0, 1), (1, 0)), ((0, 0), (1, 1), (2, 2)), ((0, 2), (1, 1), (2, 0))],
    "frame": [((0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2))],
    "u5": _orientations(((0, 0), (1, 0), (1, 1), (1, 2), (0, 2))),
}
_SHORT_BARS = [((0, 0), (0, 1)), ((0, 0), (1, 0)), ((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (2, 0))]
_HELD_OUT_CANDIDATES = ("plus", "frame", "u5", "t4", "s4")


def _pick(rng, options, weights=None):
    if weights is None:
        return options[int(rng.integers(len(options)))]
    w = np.asarray(weights, dtype=float)
    return options[int(rng.choice(len(options), p=w / w.sum()))]


def _puzzle_parameters(rng):
    minimal = bool(rng.random() < 0.07)
    colours = [int(c) for c in rng.permutation(10)]
    if rng.random() < 0.5:
        colours.remove(0)
        colours.insert(0, 0)
    if minimal:
        bg_regime, well_regime = "fixed", "fixed"
    else:
        bg_regime = _pick(rng, ("fixed", "per_pair", "reuse"), (0.5, 0.33, 0.17))
        well_regime = _pick(rng, ("fixed", "rotate", "per_pair"), (0.45, 0.3, 0.25))
    n_wells = 1 if well_regime == "fixed" else (int(rng.integers(2, 4)) if well_regime == "rotate"
                                                 else int(rng.integers(3, 5)))
    if bg_regime == "reuse":
        bgs = colours[:int(rng.integers(2, 5))]
        others = [int(c) for c in rng.permutation(10)]
        wells = others[:n_wells]
        if len(set(wells) - {bgs[0]}) == 0:
            wells.append(next(c for c in others if c != bgs[0]))
        palette = list(range(10))
    else:
        n_bgs = 1 if bg_regime == "fixed" else int(rng.integers(2, 4))
        n_wells = min(n_wells, 3)
        bgs = colours[:n_bgs]
        wells = colours[n_bgs:n_bgs + n_wells]
        rest = colours[n_bgs + n_wells:]
        palette = rest[:int(rng.integers(4, len(rest) + 1))]
    if minimal:
        objects, bands, empties, edge, decoy = "none", False, False, False, False
        size_mode = _pick(rng, ("compact", "medium"))
    else:
        objects = _pick(rng, ("none", "bars", "rich"), (0.3, 0.25, 0.45))
        bands = bool(rng.random() < 0.5)
        empties = bool(rng.random() < 0.4)
        edge = empties and bool(rng.random() < 0.4)
        decoy = bool(rng.random() < 0.1)
        size_mode = _pick(rng, tuple(_SIZE_MODES), (0.2, 0.25, 0.3, 0.15, 0.1))
    shifts = []
    if not minimal:
        kinds = ["none", "denser", "larger", "colour", "geometry"]
        weights = [0.1, 0.2, 0.2, 0.12, 0.1]
        if bands:
            kinds.append("band")
            weights.append(0.14)
        if empties:
            kinds.append("empty")
            weights.append(0.12)
        if objects == "rich":
            kinds.append("object")
            weights.append(0.14)
        primary = _pick(rng, kinds, weights)
        if primary != "none":
            shifts.append(primary)
            if rng.random() < 0.25:
                extra = [k for k in ("colour", "geometry", "larger", "object", "empty", "denser")
                         if k in kinds and k != primary and not (k == "larger" and primary == "geometry")
                         and not (k == "geometry" and primary == "larger")]
                if extra:
                    shifts.append(_pick(rng, extra))
    classes = []
    held_out = None
    if objects == "rich":
        if "object" in shifts:
            held_out = _pick(rng, _HELD_OUT_CANDIDATES)
        pool = [c for c in _OBJECT_SHAPES if c not in ("bar", held_out)]
        rng.shuffle(pool)
        classes = ["bar"] + pool[:int(rng.integers(2, 6))]
    elif objects == "bars":
        classes = ["shortbar"]
    geometry = None
    if "geometry" in shifts:
        geometry = {"aspect": _pick(rng, ("tall", "wide"))}
    if "larger" in shifts and size_mode == "big":
        size_mode = "medium"
    # Held-out resources keep the colour and empty shifts feasible: training
    # never uses the held colour as a token and never empties the held quadrant.
    held_colour = None
    if "colour" in shifts:
        spare = [c for c in palette if c not in bgs and c not in wells]
        held_colour = _pick(rng, spare)
        palette = [c for c in palette if c != held_colour]
    held_quadrant = _pick(rng, QUADRANTS) if "empty" in shifts else None
    return {"minimal": minimal, "bg_regime": bg_regime, "well_regime": well_regime,
            "held_colour": held_colour, "held_quadrant": held_quadrant,
            "bgs": bgs, "wells": wells, "palette": palette, "objects": objects,
            "classes": classes, "held_out": held_out,
            "multicolour": objects == "rich" and bool(rng.random() < 0.6),
            "bands": bands, "empties": empties, "edge": edge, "decoy": decoy,
            "size_mode": size_mode, "shifts": shifts, "geometry": geometry}


# --------------------------------------------------------------------------
# Generation: pair construction
# --------------------------------------------------------------------------

class _Canvas:
    def __init__(self, height, width, top, left):
        self.height, self.width, self.top, self.left = height, width, top, left
        self.cells = {}

    def inside(self, p):
        return 0 <= p[0] < self.height and 0 <= p[1] < self.width

    def is_well(self, p):
        return self.top <= p[0] <= self.top + 1 and self.left <= p[1] <= self.left + 1

    def free(self, p):
        return self.inside(p) and not self.is_well(p) and p not in self.cells

    def square_free(self, placed):
        merged = dict(self.cells)
        merged.update(placed)
        for (r, c), color in placed.items():
            for dr in (-1, 0):
                for dc in (-1, 0):
                    window = [(r + dr + a, c + dc + b) for a in (0, 1) for b in (0, 1)]
                    if all(merged.get(p) == color for p in window):
                        return False
        return True

    def place(self, placed, allow_square=False):
        if not all(self.free(p) for p in placed):
            return False
        if not allow_square and not self.square_free(placed):
            return False
        self.cells.update(placed)
        return True


def _region(q, height, width, top, left, open_bands):
    extra = 1 if open_bands else 0
    rows = range(0, top + extra) if q[0] == "u" else range(top + 2 - extra, height)
    cols = range(0, left + extra) if q[1] == "l" else range(left + 2 - extra, width)
    return [(r, c) for r in rows for c in cols
            if not (top <= r <= top + 1 and left <= c <= left + 1)]


def _band_cells(height, width, top, left):
    rows = [(r, c) for r in (top, top + 1) for c in range(width) if c not in (left, left + 1)]
    cols = [(r, c) for c in (left, left + 1) for r in range(height) if r not in (top, top + 1)]
    return rows, cols


def _sample_shape(rng, P, spec, ctx):
    role = spec["role"]
    shifts = spec.get("shifts", ())
    geometry = P["geometry"]
    if role == "test" and "larger" in shifts:
        floor = ctx["max_area"]
        for _ in range(120):
            if rng.random() < 0.3:
                h, w = int(rng.integers(8, 15)), int(rng.integers(24, 31))
                if rng.random() < 0.5:
                    h, w = w, h
            else:
                h, w = int(rng.integers(14, 31)), int(rng.integers(14, 31))
            if h * w > floor and (h, w) not in ctx["shapes"]:
                return h, w
        return 30, 30
    if role == "test" and spec.get("geo_mode") == "aspect":
        for _ in range(60):
            long_side = int(rng.integers(12, 31))
            short = int(rng.integers(4, max(5, int(long_side / 1.5)) + 1))
            h, w = (long_side, short) if geometry["aspect"] == "tall" else (short, long_side)
            if _aspect(h, w) == geometry["aspect"] and h * w >= 30:
                return h, w
    styles, weights = _SIZE_MODES[P["size_mode"]]
    minimum = {"core": 7, "contrast-sparse": 6, "contrast-empty": 6}.get(role, 4)
    if spec.get("edge"):
        minimum = max(minimum, 5)
    if role == "test" and "denser" in shifts:
        minimum = max(minimum, 10)
    tiny_ok = spec.get("tiny_ok", False)
    h = w = 10
    for _ in range(80):
        style = _pick(rng, styles, weights)
        (h0, h1), (w0, w1) = _SIZE_RANGES[style]
        h, w = int(rng.integers(h0, h1 + 1)), int(rng.integers(w0, w1 + 1))
        h, w = max(h, minimum), max(w, minimum)
        if "larger" in P["shifts"] and role != "test":
            h, w = min(h, 18), min(w, 18)
        if role == "test" and "larger" not in shifts and ctx["max_area"] and h * w > ctx["max_area"] \
                and rng.random() < 0.5:
            continue
        if geometry and role != "test" and _aspect(h, w) == geometry["aspect"]:
            continue
        if h * w < 21 and not tiny_ok:
            continue
        return h, w
    return max(h, 6), max(w, 6)


def _sample_well(rng, height, width, P, spec):
    def axis(size):
        low, high = 1, size - 3
        if high <= low:
            return low
        style = rng.random()
        if style < 0.3:
            centre = (size - 2) // 2 + int(rng.integers(-1, 2))
            return int(np.clip(centre, low, high))
        if style < 0.75:
            return int(rng.integers(low, high + 1))
        if style < 0.9:
            return int(rng.choice([low, high]))
        span = max(1, (high - low + 1) // 4)
        return int(rng.integers(low, low + span)) if rng.random() < 0.5 else int(rng.integers(high - span + 1, high + 1))

    geometry = P["geometry"]
    for _ in range(40):
        top, left = axis(height), axis(width)
        if spec.get("edge"):
            # Sides a kind can force empty: 'u'/'l' prefixes are upper/lower rows,
            # 'l'/'r' suffixes are left/right columns.
            kinds = {"top": "u.", "bottom": "l.", "left": ".l", "right": ".r",
                     "top-left": "ul", "top-right": "ur", "bottom-left": "ll", "bottom-right": "lr"}
            held = P.get("held_quadrant") if spec["role"] != "test" else None
            options = [k for k, side in kinds.items()
                       if held is None or not (side[0] == held[0] or side[1] == held[1])]
            kind = _pick(rng, options, [2.0 if "-" not in k else 1.0 for k in options])
            side = kinds[kind]
            if side[0] == "u":
                top = 0
            elif side[0] == "l":
                top = height - 2
            if side[1] == "l":
                left = 0
            elif side[1] == "r":
                left = width - 2
            return top, left
        sector = (_sector(top, height), _sector(left, width))
        if spec["role"] == "test" and spec.get("geo_mode") == "sector":
            if sector[0] != 1 and sector[1] != 1:
                return top, left
            continue
        if geometry and spec["role"] != "test" and sector[0] != 1 and sector[1] != 1:
            continue
        return top, left
    return top, left


def _level_count(rng, level, area):
    if area <= 0:
        return 0
    if level == "sparse":
        n = int(rng.integers(1, 3))
    elif level == "light":
        n = int(rng.integers(3, 6))
    elif level == "medium":
        n = int(np.clip(round(area * rng.uniform(0.10, 0.20)), 2, 8))
    else:
        n = int(np.clip(round(area * rng.uniform(0.18, 0.35)), 4, 16))
    return int(min(n, area))


def _choose_levels(rng, P, spec, ctx, live, big):
    role = spec["role"]
    if role == "core":
        return {q: _pick(rng, ("light", "medium"), (0.4, 0.6)) for q in live}
    if role == "contrast-sparse":
        return {q: "sparse" for q in live}
    allowed = ["sparse", "light", "medium", "dense"]
    weights = [0.25, 0.3, 0.3, 0.15]
    if "denser" in P["shifts"] and role != "test":
        allowed, weights = allowed[:3], weights[:3]
    if big:
        allowed, weights = ["medium", "dense"], [0.5, 0.5]
    profile = _pick(rng, ("uniform", "lopsided", "independent"), (0.35, 0.3, 0.35))
    if profile == "uniform":
        level = _pick(rng, allowed, weights)
        return {q: level for q in live}
    if profile == "lopsided" and live:
        heavy = _pick(rng, live)
        rich = [a for a in allowed if a in ("medium", "dense")] or allowed
        out = {q: _pick(rng, [a for a in allowed if a in ("sparse", "light")] or allowed) for q in live}
        out[heavy] = _pick(rng, rich)
        return out
    return {q: _pick(rng, allowed, weights) for q in live}


def _choose_empties(rng, P, spec, ctx):
    role = spec["role"]
    if role in ("core", "contrast-sparse") or not P["empties"]:
        return set()
    if role == "contrast-empty":
        options = [{q} for q in QUADRANTS]
        options += [set(p) for p in (("ul", "ur"), ("ll", "lr"), ("ul", "lr"), ("ur", "ll"))]
        options += [set(QUADRANTS) - {q} for q in QUADRANTS]
        weights = [3.0] * 4 + [1.0] * 4 + [0.5] * 4
        held = P.get("held_quadrant")
        if held is not None:
            keep = [i for i, o in enumerate(options) if held not in o and len(o) <= 2]
            options = [options[i] for i in keep]
            weights = [weights[i] for i in keep]
        return set(_pick(rng, options, weights))
    if role == "test" and "empty" in spec.get("shifts", ()):
        fresh = [q for q in QUADRANTS if q not in ctx["empty_quads"]]
        if fresh:
            chosen = {_pick(rng, fresh)}
            if rng.random() < 0.3:
                chosen.add(_pick(rng, [q for q in QUADRANTS if q not in chosen]))
            return chosen
        keep = _pick(rng, QUADRANTS)
        return set(QUADRANTS) - {keep}
    if "empty" in P["shifts"] and role != "test":
        return set()
    if role == "test" and not ctx["empty_demo"]:
        return set()
    if rng.random() < 0.25:
        k = _pick(rng, (1, 2, 3), (0.65, 0.25, 0.1))
        order = [QUADRANTS[int(i)] for i in rng.permutation(4)]
        return set(order[:k])
    return set()


def _pair_colours(rng, P, spec, ctx):
    role = spec["role"]
    novel = spec.get("novel", ())
    if role != "test":
        i = spec["index"]
        bg = P["bgs"][i % len(P["bgs"])]
        wells = [w for w in P["wells"] if w != bg]
        well = wells[i % len(wells)] if wells else next(c for c in range(10) if c != bg)
    else:
        held = P.get("held_colour")
        seen_bgs = sorted(ctx["bgs"])
        if "bg" in novel and len(seen_bgs) >= 2:
            bg = _pick(rng, [c for c in range(10) if c not in seen_bgs and c != held])
        else:
            bg = _pick(rng, seen_bgs)
        seen_wells = sorted(ctx["wells"])
        if "well" in novel and len(seen_wells) >= 2:
            well = _pick(rng, [c for c in range(10) if c not in seen_wells and c not in (bg, held)])
        else:
            source = P["wells"] if len(seen_wells) >= 2 else seen_wells
            options = [w for w in source if w != bg] or [w for w in seen_wells if w != bg]
            well = _pick(rng, options) if options else next(c for c in range(10) if c != bg)
    pool = [c for c in P["palette"] if c not in (bg, well)]
    if not pool:
        pool = [c for c in range(10) if c not in (bg, well)]
    rng.shuffle(pool)
    if role == "core":
        k = int(rng.integers(min(3, len(pool)), len(pool) + 1))
    elif rng.random() < (0.85 if role == "test" else 0.8):
        k = int(rng.integers(min(3, len(pool)), len(pool) + 1))
    else:
        k = int(rng.integers(1, min(2, len(pool)) + 1))
    colours = [int(c) for c in pool[:k]]
    if "tokens" in novel:
        fresh = [c for c in range(10) if c not in (bg, well) and c not in ctx["token_colors"]]
        if fresh:
            rng.shuffle(fresh)
            extra = fresh[:int(rng.integers(1, min(2, len(fresh)) + 1))]
            colours = [int(c) for c in extra] + colours[:max(1, len(colours) - len(extra))]
    return int(bg), int(well), colours


def _make_painter(rng, scheme, colours, multicolour):
    cache = {}

    def paint(q, r, c, kind):
        if scheme == "quadrant":
            return cache.setdefault(q, int(rng.choice(colours)))
        if scheme == "row":
            return cache.setdefault((q, "r", r), int(rng.choice(colours)))
        if scheme == "column":
            return cache.setdefault((q, "c", c), int(rng.choice(colours)))
        return int(rng.choice(colours))

    def object_colours(q, cells, force_mono=False):
        if multicolour and not force_mono and len(colours) >= 2 and rng.random() < 0.45:
            if rng.random() < 0.5:
                a, b = (int(x) for x in rng.choice(colours, size=2, replace=False))
                return [a if k % 2 == 0 else b for k in range(len(cells))]
            out = [int(rng.choice(colours)) for _ in cells]
            if len(set(out)) == 1:
                out[0] = int(next(x for x in colours if x != out[0]))
            return out
        r, c = cells[0]
        return [paint(q, r, c, "object")] * len(cells)

    return paint, object_colours


def _object_offsets(rng, classes):
    name = _pick(rng, classes)
    if name == "shortbar":
        return _pick(rng, _SHORT_BARS)
    return _pick(rng, _OBJECT_SHAPES[name])


def _fill_quadrant(rng, cv, q, cells, n, style, paint, object_colours, classes,
                   force_mono=False):
    if n <= 0 or not cells:
        return 0
    cellset = set(cells)
    placed = 0
    if style == "objects" and classes:
        for _ in range(8 * n):
            if placed >= n - 1:
                break
            offsets = _object_offsets(rng, classes)
            r0, c0 = cells[int(rng.integers(len(cells)))]
            pts = [(r0 + dr, c0 + dc) for dr, dc in offsets]
            if placed + len(pts) > max(n, len(pts) if placed == 0 else n):
                continue
            if any(p not in cellset or not cv.free(p) for p in pts):
                continue
            cols = object_colours(q, pts, force_mono)
            if cv.place(dict(zip(pts, cols))):
                placed += len(pts)
    free = [p for p in cells if cv.free(p)]
    need = n - placed
    if need <= 0 or not free:
        return placed
    corner = (cv.top + (q[0] == "l"), cv.left + (q[1] == "r"))
    if style == "rows":
        rows = sorted({r for r, _ in free})
        m = max(1, min(len(rows), int(np.ceil(need / rng.integers(2, 4)))))
        chosen = {int(x) for x in rng.choice(rows, size=m, replace=False)}
        pool = [p for p in free if p[0] in chosen]
    elif style == "columns":
        cols = sorted({c for _, c in free})
        m = max(1, min(len(cols), int(np.ceil(need / rng.integers(2, 4)))))
        chosen = {int(x) for x in rng.choice(cols, size=m, replace=False)}
        pool = [p for p in free if p[1] in chosen]
    elif style in ("single_row", "single_col"):
        axis = 0 if style == "single_row" else 1
        lines = {}
        for p in free:
            lines.setdefault(p[axis], []).append(p)
        best = max(len(v) for v in lines.values())
        keys = sorted(k for k, v in lines.items() if len(v) >= min(need, best))
        pool = lines[_pick(rng, keys)]
    elif style == "diagonal":
        pool = []
        for _ in range(3):
            r, c = free[int(rng.integers(len(free)))]
            dr, dc = _pick(rng, ((1, 1), (1, -1), (-1, 1), (-1, -1)))
            while (r, c) in cellset:
                if cv.free((r, c)) and (r, c) not in pool:
                    pool.append((r, c))
                r, c = r + dr, c + dc
            if len(pool) >= need:
                break
        if len(pool) < need:
            pool += [p for p in free if p not in pool]
        pool = pool[:max(need, 1)]
    elif style in ("far", "near"):
        dist = np.array([abs(r - corner[0]) + abs(c - corner[1]) + rng.uniform(0, 1.5) for r, c in free])
        order = np.argsort(-dist if style == "far" else dist, kind="stable")
        pool = [free[int(i)] for i in order[:max(need * 2, need + 2)]]
    else:
        pool = free
    if len(pool) < need:
        pool = pool + [p for p in free if p not in set(pool)]
    order = rng.permutation(len(pool))
    for i in order:
        if placed >= n:
            break
        r, c = pool[int(i)]
        color = paint(q, r, c, "single")
        if cv.place({(r, c): color}) or cv.place({(r, c): paint(q, r, c, "retry")}):
            placed += 1
    return placed


def _mirror(cv, regions, empties, paint, rng):
    """Reflect the richest quadrant into the other live quadrants."""
    live = [q for q in QUADRANTS if q not in empties]
    counts = {q: sum(1 for p in cv.cells if p in set(regions[q])) for q in live}
    source = max(live, key=lambda q: (counts[q], q))
    cells = [(p, col) for p, col in list(cv.cells.items()) if p in set(regions[source])]
    t, l = cv.top, cv.left
    recolour = rng.random() < 0.5
    for q in live:
        if q == source:
            continue
        flip_r = q[0] != source[0]
        flip_c = q[1] != source[1]
        region = set(regions[q])
        mapping = {}
        for (r, c), col in cells:
            nr = 2 * t + 1 - r if flip_r else r
            nc = 2 * l + 1 - c if flip_c else c
            if (nr, nc) in region and cv.free((nr, nc)):
                if recolour:
                    col = mapping.setdefault(col, paint(q, nr, nc, "mirror"))
                cv.place({(nr, nc): col})


def _place_straddler(rng, cv, P, empties, colours, object_colours):
    """A bar that crosses a centre line (open regime only)."""
    t, l, height, width = cv.top, cv.left, cv.height, cv.width
    for _ in range(30):
        length = int(rng.integers(3, 6))
        if rng.random() < 0.5:
            rows = [r for r in range(height) if r not in (t, t + 1)]
            if not rows:
                continue
            r = _pick(rng, rows)
            start = int(rng.integers(l - length + 2, l + 1))
            pts = [(r, start + k) for k in range(length)]
        else:
            cols = [c for c in range(width) if c not in (l, l + 1)]
            if not cols:
                continue
            c = _pick(rng, cols)
            start = int(rng.integers(t - length + 2, t + 1))
            pts = [(start + k, c) for k in range(length)]
        if any(not cv.free(p) or _quadrant(p[0], p[1], t, l) in empties for p in pts):
            continue
        q = _quadrant(pts[0][0], pts[0][1], t, l)
        if cv.place(dict(zip(pts, object_colours(q, pts)))):
            return True
    return False


def _place_band_line(rng, cv, empties, paint):
    """2-4 tokens along one side of a well row or well column (open regime).
    They are already on their floor line, so they only slide straight in."""
    t, l, height, width = cv.top, cv.left, cv.height, cv.width
    for _ in range(20):
        if rng.random() < 0.5:
            r = _pick(rng, (t, t + 1))
            cols = range(0, l) if rng.random() < 0.5 else range(l + 2, width)
            cells = [(r, c) for c in cols]
        else:
            c = _pick(rng, (l, l + 1))
            rows = range(0, t) if rng.random() < 0.5 else range(t + 2, height)
            cells = [(r, c) for r in rows]
        if not cells or _quadrant(cells[0][0], cells[0][1], t, l) in empties:
            continue
        q = _quadrant(cells[0][0], cells[0][1], t, l)
        free = [p for p in cells if cv.free(p)]
        k = min(len(free), int(rng.integers(2, 5)))
        if k < 2:
            continue
        placed = 0
        for i in rng.choice(len(free), size=k, replace=False):
            p = free[int(i)]
            placed += cv.place({p: paint(q, p[0], p[1], "band")})
        if placed >= 2:
            return True
    return False


def _place_decoy(rng, cv, regions, empties, colours):
    live = [q for q in QUADRANTS if q not in empties]
    for _ in range(40):
        q = _pick(rng, live)
        region = set(regions[q])
        if not region:
            continue
        r, c = _pick(rng, sorted(region))
        block = [(r + a, c + b) for a in (0, 1) for b in (0, 1)]
        if not all(p in region and cv.free(p) for p in block):
            continue
        color = int(rng.choice(colours))
        ring = [(r + a, c + b) for a in (-1, 0, 1, 2) for b in (-1, 0, 1, 2)
                if not (0 <= a <= 1 and 0 <= b <= 1)]
        if any(cv.cells.get(p) == color for p in ring):
            continue
        if not cv.place({p: color for p in block}, allow_square=True):
            continue
        if any(col == color and p not in block for p, col in cv.cells.items()):
            return True
        others = [p for p in sorted(region) if cv.free(p) and p not in ring] + \
                 [p for qq in live for p in regions[qq] if cv.free(p) and p not in ring]
        for p in others[:200]:
            if cv.place({p: color}):
                return True
        for p in block:
            del cv.cells[p]
    return False


def _build_input(rng, P, spec, ctx):
    role = spec["role"]
    shifts = spec.get("shifts", ())
    bg, well, colours = _pair_colours(rng, P, spec, ctx)
    height, width = _sample_shape(rng, P, spec, ctx)
    top, left = _sample_well(rng, height, width, P, spec)
    area = height * width
    budget = (2 * area - 1) // 5 - 4
    band_shift = "band" in P["shifts"]
    open_bands = P["bands"] and (role in ("core", "test") or rng.random() < 0.6)
    if role == "contrast-sparse":
        open_bands = False
    region_open = open_bands and not (band_shift and role != "test")
    empties = _choose_empties(rng, P, spec, ctx)
    regions = {q: _region(q, height, width, top, left, region_open) for q in QUADRANTS}
    empties |= {q for q in QUADRANTS if not regions[q]}
    live = [q for q in QUADRANTS if q not in empties]
    if spec.get("tiny_ok") and role == "free" and 1 <= budget < len(live):
        # A 4x4 / 4x5 canvas holds at most two tokens: keep that many corners.
        live = [live[int(i)] for i in sorted(rng.permutation(len(live))[:budget])]
        empties = set(QUADRANTS) - set(live)
    if not live:
        return None
    if len(live) > budget:
        return None
    big = (role == "test" and ctx["max_area"] and area >= 150 and area > ctx["max_area"])
    levels = _choose_levels(rng, P, spec, ctx, live, big)
    counts = {q: _level_count(rng, levels[q], len(regions[q])) for q in live}
    if role == "core":
        counts = {q: max(2, n) for q, n in counts.items()}
    if role == "test" and "denser" in shifts:
        heavy = max(live, key=lambda q: len(regions[q]))
        target = min(len(regions[heavy]), max(ctx["max_quad"] + 1, counts[heavy], 9))
        counts[heavy] = int(target + rng.integers(0, 3))
    reserve = 6 if (open_bands or spec.get("decoy")) else 0
    total = sum(counts.values())
    if total > budget - reserve:
        scale = max(budget - reserve, len(live)) / max(total, 1)
        counts = {q: max(1, int(n * scale)) for q, n in counts.items()}
    scheme = "confetti"
    if role not in ("core",):
        scheme = _pick(rng, ("confetti", "quadrant", "row", "column"), (0.55, 0.15, 0.15, 0.15))
    paint, object_colours = _make_painter(rng, scheme, colours, P["multicolour"])
    cv = _Canvas(height, width, top, left)
    styles = ["scatter", "rows", "columns", "diagonal", "single_row", "single_col", "far", "near"]
    weights = [0.26, 0.13, 0.13, 0.1, 0.08, 0.08, 0.11, 0.11]
    classes = list(P["classes"])
    if classes:
        styles.append("objects")
        weights.append(0.45)
    forced = {}
    if role == "core" and classes:
        forced[_pick(rng, live)] = "objects"
    held_quadrant = None
    if role == "test" and "object" in shifts and P["held_out"]:
        held_quadrant = max(live, key=lambda q: len(regions[q]))
        forced[held_quadrant] = "objects"
    mirror = role in ("free", "test") and len(live) >= 2 and rng.random() < 0.08
    order = list(live)
    if mirror:
        order = [max(live, key=lambda q: len(regions[q]))]
    for q in order:
        style = forced.get(q) or _pick(rng, styles, weights)
        q_classes = [P["held_out"]] if q == held_quadrant else classes
        n = counts[q]
        if q == held_quadrant:
            n = max(n, len(_OBJECT_SHAPES[P["held_out"]][0]) + 1)
        _fill_quadrant(rng, cv, q, regions[q], n, style, paint, object_colours, q_classes,
                       force_mono=(role == "core"))
    if mirror:
        _mirror(cv, regions, empties, paint, rng)
        for q in live:
            if not any(p in set(regions[q]) for p in cv.cells):
                _fill_quadrant(rng, cv, q, regions[q], 1, "scatter", paint, object_colours, [])
    if open_bands:
        row_cells, col_cells = _band_cells(height, width, top, left)
        boost = 0
        if role == "core":
            boost = 1
        elif role == "test" and band_shift:
            boost = max(int(rng.integers(2, 4)), ctx["max_band"] // 2 + 1)
        elif rng.random() < 0.4:
            boost = 1
        for pool in (row_cells, col_cells):
            pool = [p for p in pool if _quadrant(p[0], p[1], top, left) not in empties and cv.free(p)]
            if pool and boost:
                picks = rng.choice(len(pool), size=min(boost, len(pool)), replace=False)
                for i in picks:
                    p = pool[int(i)]
                    cv.place({p: paint(_quadrant(p[0], p[1], top, left), p[0], p[1], "band")})
        if not band_shift and region_open and rng.random() < 0.25:
            _place_straddler(rng, cv, P, empties, colours, object_colours)
        if not band_shift and region_open and role in ("free", "test") and rng.random() < 0.2:
            _place_band_line(rng, cv, empties, paint)
    if spec.get("decoy"):
        if not _place_decoy(rng, cv, regions, empties, colours):
            return None
    grid = np.full((height, width), bg, dtype=int)
    grid[top:top + 2, left:left + 2] = well
    for (r, c), color in cv.cells.items():
        grid[r, c] = color
    return grid, bg, well


def _template_input(rng, P, spec, ctx, variant):
    """Deterministic constructive fallback.  The layout (relative to the well)
    refutes every shortcut on its own; colours, size, offset and an optional
    mirror image vary with the attempt number."""
    role = spec["role"]
    bg, well, colours = _pair_colours(rng, P, spec, ctx)
    pool = [c for c in range(10) if c not in (bg, well)]
    colours = list(dict.fromkeys(colours + [c for c in pool if c not in colours]))
    a, b, c3 = colours[0], colours[1], colours[2]
    height = 8 + (variant % 5)
    width = 8 + ((variant // 5) % 5)
    top = 3 + (variant % 2) * min(1, height - 8)
    left = 3 + ((variant // 2) % 2) * min(1, width - 8)
    t, l = top, left
    pick = lambda k: colours[(variant + k) % len(colours)]
    cells = {(t - 3, l - 3): a, (t - 3, l - 1): b, (t - 1, l - 2): c3,
             (t - 2, l + 3): pick(1), (t + 2, l + 2): pick(2), (t + 4, l + 3): pick(3)}
    open_bands = P["bands"] and role in ("core",)
    if role == "core" and P["objects"] != "none":
        e = pick(4)
        cells[(t + 3, l - 3)] = e
        cells[(t + 3, l - 2)] = e
        cells[(t + 2, l - 1)] = pick(5)
    else:
        cells[(t + 3, l - 2)] = pick(4)
        cells[(t + 2, l - 1)] = pick(5)
    if open_bands:
        cells[(t - 3, l + 1)] = pick(6)
        cells[(t + 1, l + 3)] = pick(7)
    if role == "contrast-empty":
        emptied = [(t - 2, l + 3), (t - 3, l + 1)] if P.get("held_quadrant") == "lr" \
            else [(t + 2, l + 2), (t + 4, l + 3), (t + 1, l + 3)]
        for p in emptied:
            cells.pop(p, None)
    if role == "contrast-sparse":
        cells.pop((t - 3, l - 3), None)
    grid = np.full((height, width), bg, dtype=int)
    grid[t:t + 2, l:l + 2] = well
    for (r, c), color in cells.items():
        grid[r, c] = color
    if (variant // 3) % 2:
        grid = grid[:, ::-1]
    if (variant // 7) % 2:
        grid = grid[::-1, :]
    return np.ascontiguousarray(grid), bg, well


def _pair_ok(grid, bg, well, P, spec, ctx):
    role = spec["role"]
    if _mono_squares(grid, bg) != 1 + (1 if spec.get("decoy") else 0):
        return None
    if np.count_nonzero(grid != bg) * 5 >= grid.size * 2:
        return None
    key = grid.tobytes() + bytes(grid.shape)
    if key in ctx["inputs"] or _fingerprint(grid) in _PARENT_INPUT_FINGERPRINTS:
        return None
    try:
        f = _features(grid)
    except ValueError:
        return None
    if f["background"] != bg or f["well"] != well or f["count"] < 2:
        return None
    if spec.get("edge") and not f["edge"]:
        return None
    if spec.get("decoy") and f["decoys"] < 1:
        return None
    output = solve(grid)
    if role == "core":
        if f["empties"] or f["edge"] or f["dphase"] < 4 or f["colors"] < 3:
            return None
        if not (f["row_order"] and f["col_order"]):
            return None
        if any(not _refutes(grid, output, s) for s in _SHORTCUTS):
            return None
        if P["bands"] and not (f["row_band"] >= 1 and f["col_band"] >= 1):
            return None
        if P["objects"] != "none" and not f["broken_object"]:
            return None
    if spec.get("phase2") and not f["phase_matters"]:
        return None
    if role == "contrast-empty":
        if not (f["empties"] and f["visible_gap"] and f["max_quad"] >= 2):
            return None
    if role == "contrast-sparse":
        if f["empties"] or f["max_quad"] > 2 or f["count"] < 5:
            return None
    if role == "free" and "empty" in P["shifts"] and f["empties"] and not spec.get("edge"):
        return None
    if role != "test" and P.get("held_quadrant") in f["empties"]:
        return None
    if role == "test":
        if not (f["phase_matters"] and f["max_quad"] >= 2):
            return None
        if any(not _refutes(grid, output, s) for s in _TEST_SHORTCUTS):
            return None
        if not _test_gating_ok(f, ctx):
            return None
        feats = ctx["feats"]
        if _big_test(f, feats) and not _big_content_ok(f):
            return None
        reasons = _shift_reasons(feats, f)
        if not set(spec.get("shifts", ())) <= reasons:
            return None
        if "colour" in spec.get("shifts", ()):
            novel = spec.get("novel", ())
            if "well" in novel and f["well"] in ctx["wells"]:
                return None
            if "bg" in novel and f["background"] in ctx["bgs"]:
                return None
    return f


def _test_gating_ok(f, ctx):
    if f["empties"] and not ctx["empty_demo"]:
        return False
    if f["row_band"] and not ctx["row_band_demo"]:
        return False
    if f["col_band"] and not ctx["col_band_demo"]:
        return False
    if f["broken_object"] and not ctx["broken_demo"]:
        return False
    if f["edge"] and not (ctx["edge_demo"] and ctx["empty_demo"]):
        return False
    if f["decoys"] and not ctx["decoy_demo"]:
        return False
    if f["well"] not in ctx["wells"] and len(ctx["wells"]) < 2:
        return False
    if f["background"] not in ctx["bgs"] and len(ctx["bgs"]) < 2:
        return False
    return True


def _relax(spec, stage):
    spec = dict(spec)
    if stage >= 1:
        for key in ("decoy", "edge"):
            spec.pop(key, None)
        if spec["role"] == "test":
            spec["shifts"] = list(spec.get("shifts", ()))[:1]
            spec.pop("geo_mode", None) if "geometry" not in spec["shifts"] else None
    if stage >= 2 and spec["role"] == "test":
        spec["shifts"] = []
        spec["novel"] = ()
        spec.pop("geo_mode", None)
    return spec


def _make_pair(rng, P, spec, ctx):
    for stage, attempts in ((0, 260), (1, 120), (2, 120)):
        trial = _relax(spec, stage)
        for _ in range(attempts):
            built = _build_input(rng, P, trial, ctx)
            if built is None:
                continue
            grid, bg, well = built
            f = _pair_ok(grid, bg, well, P, trial, ctx)
            if f is not None:
                return {"input": grid, "output": solve(grid)}, f
    base = _relax(spec, 2)
    for variant in range(200):
        grid, bg, well = _template_input(rng, P, base, ctx, variant)
        f = _pair_ok(grid, bg, well, P, base, ctx)
        if f is not None:
            return {"input": grid, "output": solve(grid)}, f
    raise RuntimeError(f"could not build a {spec['role']} pair")


def _new_context():
    return {"inputs": set(), "shapes": set(), "max_area": 0, "max_quad": 0, "max_band": 0,
            "empty_quads": set(), "empty_demo": False, "row_band_demo": False,
            "col_band_demo": False, "broken_demo": False, "edge_demo": False,
            "decoy_demo": False, "wells": set(), "bgs": set(), "token_colors": set(),
            "feats": []}


def _update_context(ctx, grid, f):
    ctx["inputs"].add(grid.tobytes() + bytes(grid.shape))
    ctx["shapes"].add(tuple(grid.shape))
    ctx["max_area"] = max(ctx["max_area"], grid.size)
    ctx["max_quad"] = max(ctx["max_quad"], f["max_quad"])
    ctx["max_band"] = max(ctx["max_band"], f["row_band"] + f["col_band"])
    if f["empties"]:
        ctx["empty_quads"].update(f["empties"])
        ctx["empty_demo"] = ctx["empty_demo"] or f["visible_gap"]
    ctx["row_band_demo"] |= f["row_band"] > 0
    ctx["col_band_demo"] |= f["col_band"] > 0
    ctx["broken_demo"] |= f["broken_object"]
    ctx["edge_demo"] |= f["edge"]
    ctx["decoy_demo"] |= f["decoys"] > 0
    ctx["wells"].add(f["well"])
    ctx["bgs"].add(f["background"])
    ctx["token_colors"].update(f["token_colors"])
    ctx["feats"].append(f)


def _training_specs(P, num_train):
    specs = [{"role": "core"}, {"role": "contrast-empty" if P["empties"] else "contrast-sparse"}]
    specs += [{"role": "free", "tiny_ok": P["empties"] and "empty" not in P["shifts"]}
              for _ in range(num_train - 2)]
    free = list(range(2, num_train))
    if free:
        specs[free[0]]["phase2"] = True
    targets = free[1:] + free[:1]
    if P["edge"]:
        specs[targets[0] if targets else 1]["edge"] = True
    if P["decoy"]:
        if len(targets) >= 2:
            specs[targets[1]]["decoy"] = True
        elif targets:
            specs[targets[0]]["decoy"] = True
        else:
            specs[0]["decoy"] = True
    for i, spec in enumerate(specs):
        spec["index"] = i
    return specs


def _test_spec(rng, P, ctx):
    spec = {"role": "test", "shifts": list(P["shifts"]), "index": 0,
            "tiny_ok": ctx["empty_demo"]}
    if P["edge"] and ctx["edge_demo"] and ctx["empty_demo"] and rng.random() < 0.45:
        spec["edge"] = True
    if P["decoy"] and ctx["decoy_demo"] and rng.random() < 0.5:
        spec["decoy"] = True
    novel = set()
    if "colour" in spec["shifts"]:
        novel = {"tokens"}
        for kind, seen in (("well", ctx["wells"]), ("bg", ctx["bgs"])):
            if len(seen) >= 2 and rng.random() < 0.6:
                novel.add(kind)
    elif spec["shifts"]:
        # Incidental recolouring alongside another shift (only once demonstrated).
        if rng.random() < 0.2:
            novel.add("tokens")
        for kind, seen in (("well", ctx["wells"]), ("bg", ctx["bgs"])):
            if len(seen) >= 2 and rng.random() < 0.25:
                novel.add(kind)
    spec["novel"] = tuple(sorted(novel))
    if "geometry" in spec["shifts"]:
        spec["geo_mode"] = _pick(rng, ("aspect", "sector"))
    return spec


def _resolve_counts(num_train, num_test):
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _resolve_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    P = _puzzle_parameters(rng)
    ctx = _new_context()
    train = []
    for spec in _training_specs(P, num_train):
        pair, f = _make_pair(rng, P, spec, ctx)
        _update_context(ctx, pair["input"], f)
        train.append(pair)
    order = rng.permutation(num_train)
    train = [train[int(i)] for i in order]
    test = []
    for _ in range(num_test):
        pair, _ = _make_pair(rng, P, _test_spec(rng, P, ctx), ctx)
        ctx["inputs"].add(pair["input"].tobytes() + bytes(pair["input"].shape))
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _young_staircases(out, background, top, left):
    """Each quadrant of the output is gapless toward both well centre lines."""
    height, width = out.shape
    filled = out != background
    for r in range(height):
        left_cells = np.flatnonzero(filled[r, :left + 1])
        right_cells = np.flatnonzero(filled[r, left + 1:]) + left + 1
        if left_cells.size and not np.array_equal(
                left_cells, np.arange(left + 1 - left_cells.size, left + 1)):
            return False
        if right_cells.size and not np.array_equal(
                right_cells, np.arange(left + 1, left + 1 + right_cells.size)):
            return False
    for c in range(width):
        up = np.flatnonzero(filled[:top + 1, c])
        down = np.flatnonzero(filled[top + 1:, c]) + top + 1
        if up.size and not np.array_equal(up, np.arange(top + 1 - up.size, top + 1)):
            return False
        if down.size and not np.array_equal(down, np.arange(top + 1, top + 1 + down.size)):
            return False
    return True


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    feats = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            inp = np.asarray(pair["input"])
            out = np.asarray(pair["output"])
            for g in (inp, out):
                _require(g.ndim == 2 and all(1 <= s <= 30 for s in g.shape), "bad grid shape")
                _require(np.issubdtype(g.dtype, np.integer), "grid must be integer")
                _require(g.min() >= 0 and g.max() <= 9, "colour outside 0..9")
            _require(inp.shape == out.shape, "output shape differs from input")
            key = inp.tobytes() + bytes(inp.shape)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(_fingerprint(inp) not in _PARENT_INPUT_FINGERPRINTS, "parent input copied")
            f = _features(inp)
            background, well, top, left = f["background"], f["well"], f["top"], f["left"]
            _require(np.count_nonzero(inp != background) * 5 < inp.size * 2,
                     "background must dominate the input")
            _require(f["decoys"] >= 0, "the placeholder must be a solid 2x2 block")
            _require(np.array_equal(out, solve(inp)), "output disagrees with the rule")
            _require(np.array_equal(out, _simulate(inp)), "output disagrees with simulation")
            _require(not np.any(out == well), "placeholder colour survived")
            tok_in = np.sort(inp[(inp != background) & (inp != well)])
            tok_out = np.sort(out[out != background])
            _require(np.array_equal(tok_in, tok_out), "token multiset changed")
            _require(_young_staircases(out, background, top, left), "output not packed")
            for q, n in f["quads"].items():
                cr, cc = top + (q[0] == "l"), left + (q[1] == "r")
                _require((out[cr, cc] != background) == (n > 0),
                         "well corner fill disagrees with quadrant occupancy")
            feats[split].append(f)
    tr = feats["train"]
    phase_pairs = sum(f["phase_matters"] for f in tr)
    _require(phase_pairs >= (2 if len(tr) >= 3 else 1), "training shows phase order too rarely")
    _require(max(f["dphase"] for f in tr) >= 4, "phase-order evidence too faint")
    _require(any(f["row_order"] for f in tr), "training never shows horizontal order")
    _require(any(f["col_order"] for f in tr), "training never shows vertical order")
    _require(any(not f["empties"] for f in tr), "training lacks a full four-corner fill")
    _require(any(f["max_quad"] >= 2 for f in tr), "training lacks multi-token quadrants")
    _require(any(f["inversion"] for f in tr), "training never shows a row-length inversion")
    ctx = _new_context()
    for pair, f in zip(train, tr):
        _update_context(ctx, np.asarray(pair["input"]), f)
    for f in feats["test"]:
        _require(f["phase_matters"], "test does not distinguish vertical-first packing")
        _require(f["max_quad"] >= 2, "test reduces to the corner-only rule")
        _require(not f["empties"] or ctx["empty_demo"], "empty quadrant in test is not demonstrated")
        _require(not f["row_band"] or ctx["row_band_demo"], "row-band token not demonstrated")
        _require(not f["col_band"] or ctx["col_band_demo"], "column-band token not demonstrated")
        _require(not f["broken_object"] or ctx["broken_demo"], "object breaking not demonstrated")
        _require(not f["edge"] or (ctx["edge_demo"] and ctx["empty_demo"]),
                 "edge-touching well not demonstrated")
        _require(not f["decoys"] or ctx["decoy_demo"], "decoy 2x2 block not demonstrated")
        _require(f["well"] in ctx["wells"] or len(ctx["wells"]) >= 2,
                 "placeholder recolouring not demonstrated")
        _require(f["background"] in ctx["bgs"] or len(ctx["bgs"]) >= 2,
                 "background change not demonstrated")
        if _big_test(f, tr):
            _require(_big_content_ok(f), "larger test is too empty")
    # Every shortcut that drops part of the rule must fail on some training
    # pair; the parent-only shortcuts and vertical-first must fail on every test.
    for shortcut in _SHORTCUTS:
        refuted = [_refutes(np.asarray(p["input"]), np.asarray(p["output"]), shortcut) for p in train]
        _require(any(refuted), f"training never refutes {shortcut.__name__}")
    for pair in test:
        inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
        for shortcut in _SHORTCUTS:
            if _refutes(inp, out, shortcut):
                continue
            _require(shortcut not in _TEST_SHORTCUTS, f"test solved by {shortcut.__name__}")


def _shift_evidence(feats):
    reasons = set()
    for f in feats["test"]:
        reasons |= _shift_reasons(feats["train"], f)
    return reasons


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def self_test():
    source_in, source_out = (_rows(rows) for rows in _PARENT_B_SOURCE)
    _require(np.array_equal(solve(source_in), source_out),
             "rule must reduce to d89b689b with one token per quadrant")
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4.0, 1), (True, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {bad}")
    stats = {"shapes": set(), "reasons": set(), "multi_bg": 0, "role_reuse": 0,
             "new_test_well": 0, "new_test_bg": 0, "new_test_tokens": 0, "edge": 0, "three_empty": 0,
             "object_shapes": set(), "straddle": 0, "decoy": 0, "tiny": 0,
             "in_distribution": 0, "band": 0, "empty": 0, "broken": 0, "backgrounds": set()}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed < 8:
            again = generate_puzzle(seed)
            for split in ("train", "test"):
                for a, b in zip(puzzle[split], again[split]):
                    _require(np.array_equal(a["input"], b["input"])
                             and np.array_equal(a["output"], b["output"]), "nondeterministic")
        feats = {s: [_features(p["input"]) for p in puzzle[s]] for s in ("train", "test")}
        allf = feats["train"] + feats["test"]
        reasons = _shift_evidence(feats)
        stats["reasons"] |= reasons
        stats["in_distribution"] += not (reasons & {"denser", "larger"})
        stats["multi_bg"] += len({f["background"] for f in allf}) > 1
        roles = {}
        for f in allf:
            roles.setdefault(f["background"], set()).add("bg")
            roles.setdefault(f["well"], set()).add("well")
            for c in f["token_colors"]:
                roles.setdefault(c, set()).add("token")
        stats["role_reuse"] += any(len(v) > 1 for v in roles.values())
        train_wells = {f["well"] for f in feats["train"]}
        train_tokens = set().union(*(f["token_colors"] for f in feats["train"]))
        stats["new_test_well"] += any(f["well"] not in train_wells for f in feats["test"])
        stats["new_test_bg"] += any(f["background"] not in {t["background"] for t in feats["train"]}
                                    for f in feats["test"])
        stats["new_test_tokens"] += any(f["token_colors"] - train_tokens for f in feats["test"])
        stats["edge"] += any(f["edge"] for f in allf)
        stats["three_empty"] += any(len(f["empties"]) == 3 for f in allf)
        stats["straddle"] += any(f["straddle"] for f in allf)
        stats["decoy"] += any(f["decoys"] for f in allf)
        stats["tiny"] += any(min(f["shape"]) < 6 for f in allf)
        stats["band"] += any(f["row_band"] or f["col_band"] for f in allf)
        stats["empty"] += any(f["empties"] for f in allf)
        stats["broken"] += any(f["broken_object"] for f in allf)
        for f in allf:
            stats["shapes"].add(tuple(f["shape"]))
            stats["object_shapes"] |= {s for s in f["shapes"] if len(s) >= 4}
            stats["backgrounds"].add(f["background"])
    _require(len(stats["shapes"]) >= 60, "too few grid shapes")
    _require(stats["reasons"] >= {"denser", "larger", "band", "empty", "colour", "geometry", "object"},
             "missing measured shift kinds")
    for key, low in (("multi_bg", 6), ("role_reuse", 6), ("new_test_well", 2), ("new_test_bg", 2),
                     ("new_test_tokens", 6), ("edge", 3), ("three_empty", 1), ("straddle", 3),
                     ("decoy", 1), ("tiny", 3), ("in_distribution", 5), ("band", 8),
                     ("empty", 8), ("broken", 10)):
        _require(stats[key] >= low, f"too few puzzles exercising {key}")
    _require(len(stats["object_shapes"]) >= 8, "object vocabulary too narrow")
    _require(len(stats["backgrounds"]) >= 6, "too few background colours")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(7, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "wrong pair counts")
    return {k: (len(v) if isinstance(v, set) else v) for k, v in stats.items()}


if __name__ == "__main__":
    print(self_test())
