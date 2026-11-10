"""Onion growth along token rays (reviewed and improved generator).

Composition of parent A (2a28add5: anchor-relative, colour-blind, two-sided
compaction of the non-background cells of each line, with erasure of lines that
have no anchor) and parent B (bda2d7a6: concentric one-cell rings whose colours
repeat a role cycle by depth, and the one-phase predecessor relation across a
ring boundary).

Rule.  Every input contains exactly one rectangular "onion" core made of
concentric, uniform, one-cell rings.  Read from the border inward, the ring
colours repeat a cycle c0, c1, ..., c(p-1) of p distinct colours: the cycle is
the ring sequence up to its first repeat.  p = 3 (parent B's three-role cycle)
in every puzzle's training set; some puzzles also use 2- or 4-colour cycles.
Every row and every column crossing the core is processed on each side
separately: its non-background cells (tokens, any colour, any spacing, any
shape they belong to) are counted and replaced by a contiguous run of the same
length touching the core.  The run cell at distance d from the core gets
c[(-d) mod p], the onion's own cycle continued outward (B's predecessor
relation applied once per new layer).  The core is copied unchanged; every
other cell, including the four corner regions that share no line with the core
and the unused arm cells, becomes background.
"""
from __future__ import annotations

import numpy as np

MAX_SIDE = 30
MIN_CORE_SIDE = 5
MIN_BACKGROUND_SHARE = 0.45  # readability: background dominates the area outside the core
PERIODS = (2, 3, 4)
DEPTHS = (3, 4, 5, 6, 7, 8, 9)
SIDES = (0, 1, 2, 3)  # top, bottom, left, right arm

# Shortcut predictors that ignore part of the rule.  Training sets must refute
# each applicable one on at least one pair (see _evidence).
ALTERNATIVES = (
    "in_place",     # B only: recolour tokens where they lie (no compaction)
    "uniform",      # A only: compacted runs in one paint colour (c[p-1])
    "keep_colors",  # A-like gravity: compacted runs keep their token colours
    "forward",      # cycle continued in the wrong direction (c1, c2, ...)
    "reflect",      # mirror the core across its border (c0, c1, ...)
    "inside_out",   # read the core's rings centre-outward and repeat them
    "swap_sides",   # side counts exchanged (count is not side specific)
    "keep_corners", # corner distractors survive
    "mod3",         # hard-coded three-role cycle taken from the first three rings
)

LEVELS = ("empty", "single", "sparse", "medium", "dense", "full")
OBJECT_KINDS = ("wall", "block", "blob", "streak", "diagonal")
COLOURINGS = ("cell", "line", "arm", "object", "single")
TEST_MODES = ("familiar", "longer", "depth", "edge", "structured", "single_arm",
              "very_long", "orientation", "new_bg", "period")


# ---------------------------------------------------------------------------
# Shared geometry and rule
# ---------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _depths(height, width):
    rows = np.minimum(np.arange(height), height - 1 - np.arange(height))[:, None]
    cols = np.minimum(np.arange(width), width - 1 - np.arange(width))[None, :]
    return np.minimum(rows, cols)


def _core_block(height, width, cycle):
    return np.take(np.asarray(cycle, dtype=int), _depths(height, width) % len(cycle))


def _outward(cycle, distance):
    """Colour of the ring `distance` cells outside the core's border."""
    return int(cycle[(-distance) % len(cycle)])


def _background(grid):
    return int(np.bincount(grid.ravel(), minlength=10).argmax())


def _background_share(grid, background, rect):
    """Fraction of the cells outside the core that are background."""
    r0, c0, r1, c1 = rect
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0:r1 + 1, c0:c1 + 1] = False
    return float(np.mean(grid[outside] == background)) if outside.any() else 1.0


def _ring_sequence(block):
    height, width = block.shape
    sequence = []
    for k in range((min(height, width) + 1) // 2):
        ring = np.concatenate([block[k, k:width - k], block[height - 1 - k, k:width - k],
                               block[k:height - k, k], block[k:height - k, width - 1 - k]])
        if not np.all(ring == ring[0]):
            return None
        sequence.append(int(ring[0]))
    return sequence


def _ring_cycle(block, background):
    """The block's ring cycle if it is a valid core, else None."""
    if min(block.shape) < MIN_CORE_SIDE:
        return None
    sequence = _ring_sequence(block)
    if sequence is None or len(sequence) < 3 or background in sequence:
        return None
    period = len(set(sequence))
    if period not in PERIODS:
        return None
    cycle = tuple(sequence[:period])
    if len(set(cycle)) != period:
        return None
    if any(sequence[k] != cycle[k % period] for k in range(len(sequence))):
        return None
    return cycle


def _find_cores(grid, background):
    """All maximal rectangles that are valid concentric cycle cores.

    A candidate's size comes from the run of its depth-1 ring colour, so it never
    depends on what lies outside the core (tokens may share core colours)."""
    height, width = grid.shape
    rows = grid.tolist()
    found = []
    for r in range(height - MIN_CORE_SIDE + 1):
        top, below = rows[r], rows[r + 1]
        for c in range(width - MIN_CORE_SIDE + 1):
            x = top[c]
            if x == background or top[c + 1] != x or below[c] != x:
                continue
            y = below[c + 1]
            if y == x or y == background:
                continue
            k = c + 1
            while k < width and below[k] == y:
                k += 1
            if k >= width or below[k] != x:
                continue
            m = r + 1
            while m < height and rows[m][c + 1] == y:
                m += 1
            if m >= height or rows[m][c + 1] != x:
                continue
            h, w = m - r + 1, k - c + 1
            if min(h, w) < MIN_CORE_SIDE:
                continue
            cycle = _ring_cycle(grid[r:r + h, c:c + w], background)
            if cycle is not None:
                found.append((r, c, r + h - 1, c + w - 1, cycle))

    def inside(a, b):
        return a is not b and b[0] <= a[0] and b[1] <= a[1] and a[2] <= b[2] and a[3] <= b[3]

    return [f for f in found if not any(inside(f, g) for g in found)]


def _arm_lines(shape, rect):
    """(axis, line index, side, cells ordered by distance 1, 2, ... from the core)."""
    height, width = shape
    r0, c0, r1, c1 = rect
    for r in range(r0, r1 + 1):
        yield "h", r, 0, [(r, c0 - d) for d in range(1, c0 + 1)]
        yield "h", r, 1, [(r, c1 + d) for d in range(1, width - c1)]
    for c in range(c0, c1 + 1):
        yield "v", c, 0, [(r0 - d, c) for d in range(1, r0 + 1)]
        yield "v", c, 1, [(r1 + d, c) for d in range(1, height - r1)]


def _parse(grid):
    grid = np.asarray(grid, dtype=int)
    _require(grid.ndim == 2, "grid must be two-dimensional")
    background = _background(grid)
    cores = _find_cores(grid, background)
    _require(len(cores) == 1, f"expected exactly one concentric core, found {len(cores)}")
    r0, c0, r1, c1, cycle = cores[0]
    return grid, background, (r0, c0, r1, c1), cycle


def _render(grid, background, rect, cycle, mode="rule"):
    r0, c0, r1, c1 = rect
    period = len(cycle)
    out = np.full(grid.shape, background, dtype=int)
    out[r0:r1 + 1, c0:c1 + 1] = grid[r0:r1 + 1, c0:c1 + 1]
    rings = _ring_sequence(grid[r0:r1 + 1, c0:c1 + 1])
    first3 = rings[:3]
    lines = list(_arm_lines(grid.shape, rect))
    tokens = {}
    for axis, index, side, cells in lines:
        tokens[axis, index, side] = [(d, int(grid[p])) for d, p in enumerate(cells, 1)
                                     if grid[p] != background]
    for axis, index, side, cells in lines:
        found = tokens[axis, index, side]
        if mode == "in_place":
            for d, _ in found:
                out[cells[d - 1]] = _outward(cycle, d)
            continue
        count = len(found)
        if mode == "swap_sides":
            count = min(len(tokens[axis, index, 1 - side]), len(cells))
        for d in range(1, count + 1):
            if mode == "uniform":
                colour = int(cycle[-1])
            elif mode == "keep_colors":
                colour = found[d - 1][1]
            elif mode == "forward":
                colour = int(cycle[d % period])
            elif mode == "reflect":
                colour = int(cycle[(d - 1) % period])
            elif mode == "inside_out":
                colour = int(rings[len(rings) - 1 - ((d - 1) % len(rings))])
            elif mode == "mod3":
                colour = int(first3[(-d) % 3])
            else:
                colour = _outward(cycle, d)
            out[cells[d - 1]] = colour
    if mode == "keep_corners":
        rows = np.r_[0:r0, r1 + 1:grid.shape[0]]
        cols = np.r_[0:c0, c1 + 1:grid.shape[1]]
        if rows.size and cols.size:
            out[np.ix_(rows, cols)] = grid[np.ix_(rows, cols)]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the output from the input alone."""
    grid, background, rect, cycle = _parse(input_grid)
    return _render(grid, background, rect, cycle)


# ---------------------------------------------------------------------------
# Facts read back from returned grids (used by evidence, validator, self-test)
# ---------------------------------------------------------------------------

def _locate(cell, rect):
    """('arm', side, lateral index), ('corner',) or None for core cells."""
    r, c = cell
    r0, c0, r1, c1 = rect
    in_rows, in_cols = r0 <= r <= r1, c0 <= c <= c1
    if in_rows and in_cols:
        return None
    if in_cols:
        return ("arm", 0 if r < r0 else 1, c - c0)
    if in_rows:
        return ("arm", 2 if c < c0 else 3, r - r0)
    return ("corner",)


def _facts(source, target):
    grid, background, rect, cycle = _parse(source)
    target = np.asarray(target, dtype=int)
    r0, c0, r1, c1 = rect
    height, width = grid.shape
    arms, arm_colours, core_coloured = {}, set(), 0
    axes = set()
    for axis, index, side, cells in _arm_lines(grid.shape, rect):
        spots = [d for d, p in enumerate(cells, 1) if grid[p] != background]
        arms[axis, index, side] = (spots, len(cells))
        if spots:
            axes.add(axis)
        for d in spots:
            colour = int(grid[cells[d - 1]])
            arm_colours.add(colour)
            core_coloured += colour in cycle
    corner = grid.copy()
    corner[r0:r1 + 1, :] = background
    corner[:, c0:c1 + 1] = background
    corner_mask = corner != background
    margins = (r0, height - 1 - r1, c0, width - 1 - c1)
    refuted = frozenset(m for m in ALTERNATIVES
                        if not np.array_equal(_render(grid, background, rect, cycle, m), target))
    return {
        "grid": grid, "bg": background, "rect": rect, "cycle": cycle, "p": len(cycle),
        "depth": (min(r1 - r0, c1 - c0) + 2) // 2, "core_shape": (r1 - r0 + 1, c1 - c0 + 1),
        "arms": arms, "axes": axes, "arm_colours": arm_colours, "core_coloured": core_coloured,
        "arm_tokens": sum(len(s) for s, _ in arms.values()),
        "max_count": max([len(s) for s, _ in arms.values()] + [0]),
        "corner_tokens": int(corner_mask.sum()),
        "corner_colours": set(int(x) for x in np.unique(grid[corner_mask])),
        "margins": margins, "open_sides": sum(m > 0 for m in margins), "refuted": refuted,
    }


def _evidence(train_facts, test_facts=()):
    """Missing demonstration properties (empty set = training teaches the rule)."""
    missing = set()
    periods = {f["p"] for f in train_facts} | {f["p"] for f in test_facts}
    applicable = [m for m in ALTERNATIVES if m != "mod3" or periods - {3}]
    refuted = set().union(*(f["refuted"] for f in train_facts))
    missing |= {f"alt:{m}" for m in applicable if m not in refuted}
    if not any(f["depth"] > f["p"] for f in train_facts):
        missing.add("depth")  # the cycle visibly repeats inside some core
    if not any(f["max_count"] >= f["p"] + 1 for f in train_facts):
        missing.add("long")  # the outward run visibly wraps the cycle
    axes = set().union(*(f["axes"] for f in train_facts))
    if axes != {"h", "v"}:
        missing.add("axes")
    if not any(len(f["arm_colours"]) >= 2 for f in train_facts):
        missing.add("colors")
    if len({f["cycle"] for f in train_facts}) < 2:
        missing.add("roles")
    if not any(f["p"] == 3 and f["arm_tokens"] for f in train_facts):
        missing.add("period3")
    trained = {f["p"] for f in train_facts if f["arm_tokens"]}
    if any(f["p"] not in trained for f in test_facts):
        missing.add("test_period")
    return missing


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _regime(seed):
    if seed % 14 == 0:
        return "minimal"
    if seed % 7 == 1:
        return "rich"
    return "standard"


def _norm(weights):
    weights = np.asarray(weights, dtype=float)
    return weights / weights.sum()


def _puzzle_style(rng, regime):
    minimal, rich = regime == "minimal", regime == "rich"

    def pick(options):
        return options[int(rng.integers(len(options)))]

    order = [int(c) for c in rng.permutation(10)]
    varied = (not minimal) and rng.random() < (0.55 if rich else 0.25)
    if minimal:
        depth_p = _norm([.4, .35, .25, 0, 0, 0, 0])
        level_p = _norm([.1, .2, .4, .3, 0, 0])
    else:
        taper = np.array([1, 1, 1, 1, 1, .35, .25])  # depths 8-9 stay rare
        depth_p = _norm((rng.dirichlet(np.ones(7) * 1.5) * 0.4 + np.array([.22, .24, .2, .15, .11, .05, .03]) * 0.6) * taper)
        level_p = _norm(rng.dirichlet(np.ones(6) * 2.0) * 0.5 + np.array([.1, .08, .26, .28, .18, .1]) * 0.5)
    share = (not minimal) and rng.random() < (0.6 if rich else 0.3)
    mixed_periods = (not minimal) and rng.random() < (0.5 if rich else 0.18)
    if minimal:
        tests = [.5, .25, .25, 0, 0, 0, 0, 0, 0, 0]
    elif rich:
        tests = [.1, .12, .1, .1, .14, .1, .1, .08, .1, .1]
    else:
        tests = [.28, .14, .12, .08, .1, .05, .06, .06, .08, .08]
    return {
        "regime": regime,
        "backgrounds": order[:int(rng.integers(2, 4))] if varied else order[:1],
        "rect_p": pick([0.0, 0.25]) if minimal else pick([0.65, 0.9]) if rich else pick([0.0, 0.35, 0.65, 0.9]),
        "elong_p": 0.0 if minimal else pick([0.4, 0.7]) if rich else pick([0.0, 0.3, 0.6]),
        "depth_p": depth_p,
        "scale_p": _norm([.45, .4, .15, 0]) if minimal else rng.dirichlet(np.ones(4) * 2.0),
        "edge_p": 0.0 if minimal else pick([0.08, 0.15]) if rich else pick([0.0, 0.06, 0.12]),
        "extreme_p": 0.0 if minimal else pick([0.12, 0.2]) if rich else pick([0.0, 0.05, 0.1]),
        "level_p": level_p,
        "sparse_p": pick([0.15, 0.3]) if minimal else pick([0.0, 0.15]) if rich else pick([0.0, 0.15, 0.3]),
        "structure_p": 0.0 if minimal else pick([0.55, 0.8]) if rich else pick([0.0, 0.3, 0.6]),
        "object_p": rng.dirichlet(np.ones(len(OBJECT_KINDS)) * 2.0),
        "frame_p": 0.0 if minimal else pick([0.2, 0.35]) if rich else pick([0.0, 0.08, 0.2]),
        "long_p": 0.0 if minimal else pick([0.15, 0.3]) if rich else pick([0.0, 0.1, 0.2]),
        "corner_p": _norm([.4, .6, 0, 0]) if minimal else rng.dirichlet(np.ones(4) * 2.0),
        "corner_style_p": _norm([1, 0, 0]) if minimal else rng.dirichlet(np.ones(3) * 2.0),
        "corner_colour_p": _norm([1, 0, 0]) if minimal else rng.dirichlet(np.ones(3) * 2.0),
        "colouring_p": _norm([.5, .25, 0, 0, .25]) if minimal else rng.dirichlet(np.ones(len(COLOURINGS)) * 1.5),
        "token_k": int(rng.integers(1, 4)) if minimal else int(rng.integers(1, 7)),
        "share": share,
        "period_p": _norm([.3, .38, .32]) if mixed_periods else _norm([0, 1, 0]),
        "test_p": _norm(tests),
    }


def _roles(rng, background, reference, period):
    pool = [c for c in range(10) if c != background]
    if reference is not None and background not in reference:
        ref = [int(c) for c in reference]
        top = min(period, len(ref))
        mode = int(rng.choice(4, p=(0.22, 0.24, 0.30, 0.24)))
        shared = min(top, (top, 2, 1, 0)[mode])
        outside = [c for c in pool if c not in ref]
        if period - shared <= len(outside):
            kept = [int(x) for x in rng.choice(ref, size=shared, replace=False)] if shared else []
            fresh = [int(x) for x in rng.choice(outside, size=period - shared, replace=False)] \
                if period > shared else []
            colours = kept + fresh
            for _ in range(12):
                order = tuple(colours[i] for i in rng.permutation(period))
                if order != tuple(ref):
                    return order
            return tuple(colours)
    return tuple(int(x) for x in rng.choice(pool, size=period, replace=False))


def _line_count(rng, length, activity):
    if length == 0 or rng.random() >= activity:
        return 0
    kind = int(rng.choice(4, p=(0.22, 0.36, 0.32, 0.10)))
    if kind == 0:
        return 1
    if kind == 1:
        return int(rng.integers(1, min(3, length) + 1))
    if kind == 2:
        return int(rng.integers(1, length + 1))
    return length


def _placements(rng, length, count, style=None):
    """Distances (1-based, from the core) of `count` tokens in a line of `length` cells."""
    if count == 0:
        return []
    if count >= length:
        return list(range(1, length + 1))
    if style is None:
        style = int(rng.choice(5, p=(0.45, 0.18, 0.12, 0.15, 0.10)))
    if style == 0:
        return sorted(int(x) for x in rng.choice(np.arange(1, length + 1), size=count, replace=False))
    if style == 1:
        start = int(rng.integers(1, length - count + 2))
        return list(range(start, start + count))
    if style == 2:
        return list(range(1, count + 1))
    if style == 3:
        return list(range(length - count + 1, length + 1))
    near = int(rng.integers(0, count))
    if length - near - 1 >= count - near:
        rest = rng.choice(np.arange(near + 2, length + 1), size=count - near, replace=False)
    else:
        rest = np.arange(length - count + near + 1, length + 1)
    return sorted(set(list(range(1, near + 1)) + [int(x) for x in rest]))


def _gapped(rng, length, count):
    """Placement whose tokens are not already packed against the core."""
    for _ in range(20):
        spots = _placements(rng, length, count, style=0)
        if spots != list(range(1, count + 1)):
            return spots
    return list(range(length - count + 1, length + 1))


def _arm_info(shape, rect):
    """Per side: (number of crossing lines, arm length)."""
    height, width = shape
    r0, c0, r1, c1 = rect
    return {0: (c1 - c0 + 1, r0), 1: (c1 - c0 + 1, height - 1 - r1),
            2: (r1 - r0 + 1, c0), 3: (r1 - r0 + 1, width - 1 - c1)}


def _arm_cell(rect, side, i, t):
    r0, c0, r1, c1 = rect
    if side == 0:
        return r0 - t, c0 + i
    if side == 1:
        return r1 + t, c0 + i
    if side == 2:
        return r0 + i, c0 - t
    return r0 + i, c1 + t


class _Material:
    """Token cells (cell -> piece id) plus each piece's kind."""

    def __init__(self, shape, rect):
        self.shape, self.rect = shape, rect
        self.cells, self.kinds = {}, []

    def piece(self, kind):
        self.kinds.append(kind)
        return len(self.kinds) - 1

    def put(self, cell, piece):
        r, c = cell
        if 0 <= r < self.shape[0] and 0 <= c < self.shape[1] and _locate(cell, self.rect) is not None:
            self.cells[(int(r), int(c))] = piece

    def set_line(self, side, i, length, spots, kind="scatter"):
        for t in range(1, length + 1):
            self.cells.pop(_arm_cell(self.rect, side, i, t), None)
        for t in spots:
            self.put(_arm_cell(self.rect, side, i, t), self.piece(kind))

    def line_spots(self, side, i, length):
        return [t for t in range(1, length + 1) if _arm_cell(self.rect, side, i, t) in self.cells]


def _object_cells(rng, kind, n, length):
    """Arm-frame cells (lateral i, distance t); i may overhang into corners."""
    if kind == "wall":
        t0 = int(rng.integers(1, length + 1))
        thick = int(rng.integers(1, min(3, length - t0 + 1) + 1))
        if rng.random() < 0.4 or n < 3:
            a, b = 0, n
        else:
            a = int(rng.integers(-2, n - 1))
            b = int(rng.integers(a + 2, n + 3))
        return [(i, t) for i in range(a, b) for t in range(t0, t0 + thick)]
    if kind == "block":
        bw = int(rng.integers(2, min(5, n + 2) + 1))
        bd = int(rng.integers(1, min(4, length) + 1))
        i0 = int(rng.integers(-1, n))
        t0 = int(rng.integers(1, length - bd + 2))
        return [(i, t) for i in range(i0, i0 + bw) for t in range(t0, t0 + bd)]
    if kind == "blob":
        size = int(rng.integers(4, 13))
        cells = [(int(rng.integers(0, n)), int(rng.integers(1, length + 1)))]
        for _ in range(size * 6):
            if len(cells) >= size:
                break
            ci, ct = cells[int(rng.integers(len(cells)))]
            di, dt = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            cell = (ci + di, ct + dt)
            if -2 <= cell[0] < n + 2 and 1 <= cell[1] <= length and cell not in cells:
                cells.append(cell)
        return cells
    if kind == "streak":
        i = int(rng.integers(0, n))
        t0 = int(rng.integers(1, length + 1))
        span = int(rng.integers(2, max(2, length - t0 + 1) + 1))
        return [(i, t) for t in range(t0, min(length, t0 + span - 1) + 1)]
    i, t = int(rng.integers(0, n)), int(rng.integers(1, length + 1))
    di, dt = (1 if rng.random() < 0.5 else -1), (1 if rng.random() < 0.5 else -1)
    cells = []
    for _ in range(int(rng.integers(3, max(3, min(n + 2, length + 2)) + 1))):
        if not (-2 <= i < n + 2 and 1 <= t <= length):
            break
        cells.append((i, t))
        i, t = i + di, t + dt
    return cells


def _scatter_arm(rng, material, side, n, length, level):
    if level == "single":
        i = int(rng.integers(n))
        material.set_line(side, i, length, _placements(rng, length, _line_count(rng, length, 1.0)))
        return
    if level == "full":
        for i in range(n):
            count = length if rng.random() < 0.4 else int(rng.integers(max(1, length // 2), length + 1))
            material.set_line(side, i, length, _placements(rng, length, count))
        return
    low, high = {"sparse": (0.1, 0.3), "medium": (0.3, 0.6), "dense": (0.6, 0.95)}[level]
    activity = float(rng.uniform(low, high))
    for i in range(n):
        spots = _placements(rng, length, _line_count(rng, length, activity))
        for t in spots:
            material.put(_arm_cell(material.rect, side, i, t), material.piece("scatter"))


def _structured_arm(rng, style, material, side, n, length, level):
    amount = {"single": 1, "sparse": 1, "medium": 2, "dense": 3, "full": 4}[level]
    for _ in range(amount):
        kind = OBJECT_KINDS[int(rng.choice(len(OBJECT_KINDS), p=style["object_p"]))]
        piece = material.piece(kind)
        for i, t in _object_cells(rng, kind, n, length):
            material.put(_arm_cell(material.rect, side, i, t), piece)
    if rng.random() < 0.3:
        _scatter_arm(rng, material, side, n, length, "sparse")


def _add_frames(rng, material, margins, clean):
    """Loose rectangular rings around the core; their corner parts lie in corner regions."""
    r0, c0, r1, c1 = material.rect
    reach = max(margins)
    if reach < 1:
        return
    first = 2 if clean and reach >= 2 else 1
    count = 1 if reach - first < 2 or rng.random() < 0.7 else 2
    distances = sorted(int(x) for x in rng.choice(np.arange(first, reach + 1),
                                                  size=min(count, reach - first + 1), replace=False))
    for k in distances:
        a, b, c, d = r0 - k, c0 - k, r1 + k, c1 + k
        ring = [(a, x) for x in range(b, d + 1)] + [(c, x) for x in range(b, d + 1)]
        ring += [(y, b) for y in range(a + 1, c)] + [(y, d) for y in range(a + 1, c)]
        options = (1.0, 0.9) if clean else (1.0, 0.85, 0.65)
        keep = float(options[int(rng.integers(len(options)))])
        kept = [cell for cell in ring if rng.random() < keep]
        if k == 1 and len(kept) == len(ring):
            kept.pop(int(rng.integers(len(kept))))  # a complete distance-1 ring could extend the core
        piece = material.piece("frame_uniform" if clean or rng.random() < 0.6 else "frame")
        for cell in kept:
            material.put(cell, piece)


def _corner_regions(shape, rect):
    height, width = shape
    r0, c0, r1, c1 = rect
    regions = []
    for rows in ((0, r0), (r1 + 1, height)):
        for cols in ((0, c0), (c1 + 1, width)):
            if rows[1] > rows[0] and cols[1] > cols[0]:
                regions.append((rows, cols))
    return regions


def _add_corners(rng, style, material, density):
    regions = _corner_regions(material.shape, material.rect)
    cells = [(r, c) for (ra, rb), (ca, cb) in regions for r in range(ra, rb) for c in range(ca, cb)]
    if not cells or not density:
        return
    if density == 1:
        target = int(rng.integers(1, min(4, len(cells)) + 1))
    elif density == 2:
        target = max(1, int(len(cells) * rng.uniform(0.08, 0.2)))
    else:
        target = max(1, int(len(cells) * rng.uniform(0.25, 0.45)))
    mode = int(rng.choice(3, p=style["corner_style_p"]))  # cells, objects, mixed
    placed = sum(1 for cell in material.cells if _locate(cell, material.rect) == ("corner",))
    for _ in range(4 * target + 8):
        if placed >= target:
            break
        if mode == 0 or (mode == 2 and rng.random() < 0.5) or target <= 2:
            cell = cells[int(rng.integers(len(cells)))]
            if cell not in material.cells:
                material.put(cell, material.piece("corner"))
                placed += 1
            continue
        (ra, rb), (ca, cb) = regions[int(rng.integers(len(regions)))]
        r, c = int(rng.integers(ra, rb)), int(rng.integers(ca, cb))
        shape = int(rng.integers(4))
        if shape == 0:
            offsets = [(0, j) for j in range(int(rng.integers(2, 5)))]
        elif shape == 1:
            offsets = [(j, 0) for j in range(int(rng.integers(2, 5)))]
        elif shape == 2:
            offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
        else:
            step = 1 if rng.random() < 0.5 else -1
            offsets = [(j, step * j) for j in range(int(rng.integers(3, 5)))]
        piece = material.piece("corner_object")
        for dr, dc in offsets:
            cell = (r + dr, c + dc)
            if ra <= cell[0] < rb and ca <= cell[1] < cb and cell not in material.cells:
                material.put(cell, piece)
                placed += 1


def _colour(rng, style, material, palette, corner_palette):
    scheme = COLOURINGS[int(rng.choice(len(COLOURINGS), p=style["colouring_p"]))]
    chosen = {}

    def pick(key, options):
        if key not in chosen:
            chosen[key] = int(options[int(rng.integers(len(options)))])
        return chosen[key]

    colours = {}
    for cell in sorted(material.cells):
        piece = material.cells[cell]
        kind = material.kinds[piece]
        where = _locate(cell, material.rect)
        if kind in ("corner", "corner_object"):
            colours[cell] = pick(("corner", piece), corner_palette)
        elif where == ("corner",) or kind == "frame_uniform":
            colours[cell] = pick(("piece", piece), palette)  # overhangs and frames keep one colour
        elif scheme == "cell":
            colours[cell] = pick(("cell", cell), palette)
        elif scheme == "line":
            colours[cell] = pick(("line", where[1], where[2]), palette)
        elif scheme == "arm":
            colours[cell] = pick(("arm", where[1]), palette)
        elif scheme == "object":
            colours[cell] = pick(("piece", piece), palette)
        else:
            colours[cell] = pick(("single",), palette)
    return colours


def _core_dims(rng, style, depth, orient):
    short = max(MIN_CORE_SIDE, 2 * depth - int(rng.integers(0, 2)))
    cap = MAX_SIDE - 2
    if short >= cap or (orient is None and rng.random() >= style["rect_p"]):
        return short, short
    if orient is not None or rng.random() < style["elong_p"]:
        low = min(4, cap - short)
        diff = int(rng.integers(low, cap - short + 1))
    else:
        diff = int(rng.integers(1, min(8, cap - short) + 1))
    longer = short + diff
    wide = (orient == "wide") if orient is not None else rng.random() < 0.5
    return (short, longer) if wide else (longer, short)


def _sample_pair(rng, style, background, reference, req):
    """One (pair, facts) satisfying the request, or None."""
    needs = set(req.get("needs", ()))
    allowed = tuple(req.get("periods", PERIODS))
    for _ in range(60):
        # ---- cycle length and depth
        if "period" in req:
            period = req["period"]
        elif "p3" in needs:
            period = 3
        elif any(n.startswith("period:") for n in needs):
            period = int(next(n for n in needs if n.startswith("period:")).split(":")[1])
        else:
            weights = [style["period_p"][j] if q in allowed else 0.0 for j, q in enumerate(PERIODS)]
            period = PERIODS[int(rng.choice(3, p=_norm(weights)))] if sum(weights) > 0 else 3
        if "depth" in req and req["depth"] < max(3, period):
            period = 3
        low = max(3, period)
        if "depth" in req:
            depth = req["depth"]
        elif "depth" in needs:
            options = [d for d in DEPTHS if d >= low and d > period and d % period and d <= 8]
            depth = int(rng.choice(options, p=_norm([1.0 / (1 + j) for j in range(len(options))])))
        elif req.get("small_core"):
            depth = low
        else:
            weights = [w if d >= low else 0.0 for d, w in zip(DEPTHS, style["depth_p"])]
            depth = int(rng.choice(DEPTHS, p=_norm(weights)))
        height, width = _core_dims(rng, style, depth, req.get("orient"))

        # ---- edge contact
        closed = set(req.get("edges", ()))
        if req.get("single_arm"):
            closed = set(SIDES) - {int(rng.integers(4))}
        elif not closed:
            if rng.random() < style["extreme_p"]:
                if rng.random() < 0.5:
                    closed = {0, 1} if rng.random() < 0.5 else {2, 3}
                else:
                    closed = set(int(s) for s in rng.permutation(4)[:3])
            for s in SIDES:
                if rng.random() < style["edge_p"]:
                    closed.add(s)
            if len(closed) == 4:
                closed.discard(int(rng.integers(4)))
        if needs & {"axes", "asym", "corner", "long", "gap"} and len(closed) > 1:
            closed = set(sorted(closed)[:1]) if rng.random() < 0.5 else set()
        open_sides = [s for s in SIDES if s not in closed]

        # ---- margins
        scale = int(rng.choice(4, p=style["scale_p"]))
        lo_m, hi_m = ((1, 3), (2, 6), (3, 10), (5, 14))[scale]
        margins = [0 if s in closed else int(rng.integers(lo_m, hi_m + 1)) for s in SIDES]
        min_arm = int(req.get("min_arm", 0))
        if "long" in needs:
            min_arm = max(min_arm, max(4, period + 1) + 1)
        pinned = None
        if min_arm:
            pinned = open_sides[int(rng.integers(len(open_sides)))]
            margins[pinned] = max(margins[pinned], min_arm + int(rng.integers(0, 3)))
        elif rng.random() < style["long_p"]:
            pinned = open_sides[int(rng.integers(len(open_sides)))]
            margins[pinned] = MAX_SIDE
        failed = False
        for (a, b), size in (((0, 1), height), ((2, 3), width)):
            while size + margins[a] + margins[b] > MAX_SIDE:
                floor = {s: (min_arm if s == pinned else (0 if s in closed else 1)) for s in (a, b)}
                options = [s for s in (a, b) if margins[s] > floor[s]]
                if not options:
                    failed = True
                    break
                unpinned = [s for s in options if s != pinned]
                side = max(unpinned or options, key=lambda s: margins[s])
                margins[side] -= 1
        if failed:
            continue
        top, bottom, left, right = margins
        grid_h, grid_w = height + top + bottom, width + left + right
        rect = (top, left, top + height - 1, left + width - 1)

        # ---- colours
        cycle = _roles(rng, background, reference, period)
        grid = np.full((grid_h, grid_w), background, dtype=int)
        grid[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1] = _core_block(height, width, cycle)
        non_bg = [c for c in range(10) if c != background]
        share = style["share"] and not req.get("no_share")
        available = non_bg if share else [c for c in non_bg if c not in cycle]
        k = int(min(len(available), max(1, style["token_k"] + int(rng.integers(-1, 2)))))
        if "colors" in needs:
            k = max(k, 2)
        palette = [int(x) for x in rng.choice(available, size=k, replace=False)]
        if share and not set(palette) & set(cycle) and ("share" in needs or rng.random() < 0.75):
            palette[int(rng.integers(k))] = int(cycle[int(rng.integers(period))])
        corner_mode = int(rng.choice(3, p=style["corner_colour_p"]))  # same, separate, mixed
        others = [c for c in available if c not in palette]
        if corner_mode and others:
            extra = [int(x) for x in rng.choice(others, size=min(len(others), int(rng.integers(1, 4))), replace=False)]
            corner_palette = extra if corner_mode == 1 else palette + extra
        else:
            corner_palette = palette

        # ---- arm material
        material = _Material(grid.shape, rect)
        info = _arm_info(grid.shape, rect)
        arms_open = [s for s in SIDES if info[s][1] > 0]
        structured_test = bool(req.get("structured"))
        sparse_pair = (not needs and not min_arm and not structured_test and rng.random() < style["sparse_p"])
        frame_roll = rng.random()
        clean_frame = not sparse_pair and frame_roll < (0.25 if structured_test else style["frame_p"] * 0.45)
        overlay_frame = not clean_frame and frame_roll < (0.5 if structured_test else style["frame_p"])
        if clean_frame:
            _add_frames(rng, material, margins, clean=True)
            if rng.random() < 0.35:
                for s in arms_open:
                    _scatter_arm(rng, material, s, info[s][0], info[s][1], "single")
        elif sparse_pair:
            lines = [(s, i) for s in arms_open for i in range(info[s][0])]
            for j in rng.choice(len(lines), size=min(len(lines), int(rng.integers(1, 4))), replace=False):
                s, i = lines[int(j)]
                length = info[s][1]
                material.set_line(s, i, length, _placements(rng, length, _line_count(rng, length, 1.0)))
        else:
            for s in arms_open:
                n, length = info[s]
                level = LEVELS[int(rng.choice(len(LEVELS), p=style["level_p"]))]
                if structured_test and level in ("empty", "single"):
                    level = "medium"
                if level == "empty":
                    continue
                if structured_test or rng.random() < style["structure_p"]:
                    _structured_arm(rng, style, material, s, n, length, level)
                else:
                    _scatter_arm(rng, material, s, n, length, level)
            if overlay_frame:
                _add_frames(rng, material, margins, clean=False)

        # ---- witness patches on positions
        def lines_of(sides, min_length):
            return [(s, i) for s in sides if info[s][1] >= min_length for i in range(info[s][0])]

        if min_arm:
            n, length = info[pinned]
            if length < min_arm:
                continue
            count = int(rng.integers(min_arm, length + 1))
            spots = _gapped(rng, length, count) if count < length else list(range(1, length + 1))
            material.set_line(pinned, int(rng.integers(n)), length, spots)
        if "gap" in needs:
            candidates = lines_of(arms_open, 2)
            if not candidates:
                continue
            s, i = candidates[int(rng.integers(len(candidates)))]
            length = info[s][1]
            material.set_line(s, i, length, _gapped(rng, length, int(rng.integers(1, length))))
        if "axes" in needs:
            for sides in ((0, 1), (2, 3)):
                if not any(_locate(cell, rect)[1] in sides for cell in material.cells
                           if _locate(cell, rect)[0] == "arm"):
                    candidates = lines_of([s for s in sides if s in arms_open], 2)
                    if not candidates:
                        break
                    s, i = candidates[int(rng.integers(len(candidates)))]
                    length = info[s][1]
                    material.set_line(s, i, length, _gapped(rng, length, int(rng.integers(1, length))))
        if "asym" in needs:
            pairs = [(a, b) for a, b in ((0, 1), (2, 3)) if info[a][1] and info[b][1]]
            if not pairs:
                continue
            a, b = pairs[int(rng.integers(len(pairs)))]
            i = int(rng.integers(info[a][0]))
            if len(material.line_spots(a, i, info[a][1])) == len(material.line_spots(b, i, info[b][1])):
                x = int(rng.integers(1, info[a][1] + 1))
                options = [y for y in range(0, info[b][1] + 1) if y != x]
                y = int(options[int(rng.integers(len(options)))])
                material.set_line(a, i, info[a][1], _placements(rng, info[a][1], x))
                material.set_line(b, i, info[b][1], _placements(rng, info[b][1], y))
        density = int(rng.choice(4, p=style["corner_p"]))
        if "corner" in needs and density == 0:
            density = 1
        _add_corners(rng, style, material, density)

        # ---- colouring and colour patches
        colours = _colour(rng, style, material, palette, corner_palette)
        for cell, colour in colours.items():
            grid[cell] = colour
        arm_cells = sorted(cell for cell in material.cells if _locate(cell, rect)[0] == "arm")
        if not arm_cells:
            continue
        if "colors" in needs and len({int(grid[cell]) for cell in arm_cells}) < 2 and len(arm_cells) >= 2:
            first, second = arm_cells[0], arm_cells[-1]
            grid[second] = next(c for c in palette if c != grid[first])

        # ---- acceptance
        totals = np.bincount(grid.ravel(), minlength=10)
        if totals[background] <= max(int(totals[c]) for c in range(10) if c != background):
            continue
        if _background_share(grid, background, rect) < MIN_BACKGROUND_SHARE:
            continue
        if _find_cores(grid, background) != [rect + (cycle,)]:
            continue
        output = _render(grid, background, rect, cycle)
        if np.array_equal(output, grid):
            continue
        facts = _facts(grid, output)
        if not _needs_met(facts, needs, req):
            continue
        return {"input": grid, "output": output}, facts
    return None


def _needs_met(facts, needs, req):
    refuted = facts["refuted"]
    checks = {
        "long": facts["max_count"] >= facts["p"] + 1,
        "gap": "in_place" in refuted,
        "axes": facts["axes"] == {"h", "v"},
        "colors": len(facts["arm_colours"]) >= 2,
        "asym": "swap_sides" in refuted,
        "corner": "keep_corners" in refuted,
        "depth": facts["depth"] > facts["p"] and "inside_out" in refuted,
        "kc": "keep_colors" in refuted,
        "p3": facts["p"] == 3 and {"forward", "reflect"} <= refuted,
        "share": facts["core_coloured"] > 0,
    }
    for need in needs:
        if need.startswith("period:"):
            if facts["p"] != int(need.split(":")[1]) or "mod3" not in refuted:
                return False
        elif not checks.get(need, True):
            return False
    if facts["max_count"] < int(req.get("min_arm", 0)):
        return False
    if req.get("single_arm") and facts["open_sides"] != 1:
        return False
    if any(facts["margins"][s] for s in req.get("edges", ())):
        return False
    if "depth" in req and facts["depth"] != req["depth"]:
        return False
    if "period" in req and facts["p"] != req["period"]:
        return False
    orient = req.get("orient")
    h, w = facts["core_shape"]
    if orient and not ((orient == "wide" and w > h) or (orient == "tall" and h > w)):
        return False
    return True


_WITNESS = {
    "alt:in_place": "gap", "alt:uniform": "long", "alt:keep_colors": "kc", "alt:forward": "p3",
    "alt:reflect": "p3", "alt:inside_out": "depth", "alt:swap_sides": "asym",
    "alt:keep_corners": "corner", "depth": "depth", "long": "long", "axes": "axes",
    "colors": "colors", "period3": "p3",
}


def _attempt(rng, style, num_train, num_test):
    train = []
    for _ in range(num_train):
        background = int(style["backgrounds"][int(rng.integers(len(style["backgrounds"])))])
        reference = train[int(rng.integers(len(train)))][1]["cycle"] if train else None
        sampled = _sample_pair(rng, style, background, reference, {})
        if sampled is None:
            return None
        train.append(sampled)

    for step in range(36):
        facts = [f for _, f in train]
        missing = _evidence(facts)
        if not missing:
            break
        ordered = sorted(missing)
        chosen = ordered if step >= 18 else [ordered[int(rng.integers(len(ordered)))]]
        needs = set()
        for item in chosen:
            if item == "alt:mod3":
                non3 = sorted({f["p"] for f in facts} - {3})
                if non3:
                    needs.add(f"period:{non3[int(rng.integers(len(non3)))]}")
            elif item in _WITNESS:
                needs.add(_WITNESS[item])
        if any(n.startswith("period:") for n in needs):
            needs.discard("p3")
        # Replace a pair whose loss costs the least evidence.
        costs = [len(_evidence(facts[:j] + facts[j + 1:])) for j in range(len(train))]
        best = min(costs)
        options = [j for j, cost in enumerate(costs) if cost == best]
        index = options[int(rng.integers(len(options)))]
        background = int(style["backgrounds"][int(rng.integers(len(style["backgrounds"])))])
        others = [f["cycle"] for j, f in enumerate(facts) if j != index]
        reference = others[int(rng.integers(len(others)))] if others else None
        sampled = _sample_pair(rng, style, background, reference, {"needs": needs})
        if sampled is None:
            continue
        candidate = train[:index] + [sampled] + train[index + 1:]
        if len(_evidence([f for _, f in candidate])) < len(missing) or step >= 24:
            train = candidate
    facts = [f for _, f in train]
    if _evidence(facts):
        return None
    # Shared-palette puzzles show at least one ring-coloured arm token in training (review V5).
    for _ in range(8 if style["share"] and not any(f["core_coloured"] for f in facts) else 0):
        index = int(rng.integers(len(train)))
        background = int(style["backgrounds"][int(rng.integers(len(style["backgrounds"])))])
        others = [f["cycle"] for j, f in enumerate(facts) if j != index]
        sampled = _sample_pair(rng, style, background, others[int(rng.integers(len(others)))],
                               {"needs": {"share"}})
        if sampled is None:
            continue
        candidate = train[:index] + [sampled] + train[index + 1:]
        if not _evidence([f for _, f in candidate]):
            train = candidate
            facts = [f for _, f in train]
            break

    train_depths = {f["depth"] for f in facts}
    train_max = max(f["max_count"] for f in facts)
    backgrounds = sorted({f["bg"] for f in facts})
    periods = tuple(sorted({f["p"] for f in facts}))
    orients = {("wide" if f["core_shape"][1] > f["core_shape"][0] else "tall")
               for f in facts if f["core_shape"][0] != f["core_shape"][1]}
    feasible = {"new_bg": len(backgrounds) >= 2, "period": len(periods) > 1 or periods != (3,),
                "orientation": len(orients) == 1, "longer": train_max + 1 <= 25,
                "very_long": max(train_max + 1, 15) <= 25}
    weights = _norm([w if feasible.get(m, True) else 0.0 for m, w in zip(TEST_MODES, style["test_p"])])
    test = []
    for _ in range(num_test):
        mode = TEST_MODES[int(rng.choice(len(TEST_MODES), p=weights))]
        background = int(backgrounds[int(rng.integers(len(backgrounds)))])
        reference = facts[int(rng.integers(len(facts)))]["cycle"]
        req = {"periods": periods}
        if mode == "longer" and train_max + 1 <= 25:
            req["min_arm"] = train_max + 1
        elif mode == "very_long" and max(train_max + 1, 15) <= 25:
            req.update(min_arm=max(train_max + 1, 15), small_core=True)
        elif mode == "depth":
            fresh = [d for d in DEPTHS if d not in train_depths and d <= 8]
            if fresh:
                req["depth"] = int(fresh[int(rng.integers(len(fresh)))])
                req["periods"] = (3,)
        elif mode == "edge":
            req["edges"] = tuple(int(s) for s in rng.permutation(4)[:int(rng.integers(1, 3))])
        elif mode == "single_arm":
            req["single_arm"] = True
        elif mode == "structured":
            req["structured"] = True
        elif mode == "orientation" and len(orients) == 1:
            req["orient"] = "tall" if "wide" in orients else "wide"
        elif mode == "new_bg" and len(backgrounds) >= 2:
            fresh = [c for c in range(10) if c not in backgrounds]
            background = int(fresh[int(rng.integers(len(fresh)))])
            if background in reference:
                reference = None
        elif mode == "period":
            non3 = [p for p in periods if p != 3]
            if non3:
                req["period"] = int(non3[int(rng.integers(len(non3)))])
        sampled = _sample_pair(rng, style, background, reference, req)
        if sampled is None:
            fallback = int(backgrounds[int(rng.integers(len(backgrounds)))])
            sampled = _sample_pair(rng, style, fallback, None, {"periods": periods})
        if sampled is None:
            return None
        test.append(sampled)

    seen = set()
    for pair, _ in train + test:
        key = (pair["input"].shape, pair["input"].tobytes())
        if key in seen:
            return None
        seen.add(key)
    if _evidence(facts, [f for _, f in test]):
        return None
    as_lists = lambda pairs: [{"input": p["input"].tolist(), "output": p["output"].tolist()} for p, _ in pairs]
    return {"train": as_lists(train), "test": as_lists(test)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    regime = _regime(seed)
    for attempt in range(60):
        if attempt % 10 == 0:
            style = _puzzle_style(rng, regime)
        puzzle = _attempt(rng, style, num_train, num_test)
        if puzzle is not None:
            validate_generated_puzzle(puzzle)
            return puzzle
    raise RuntimeError(f"could not generate a puzzle for seed {seed}")


# ---------------------------------------------------------------------------
# Validation and self-test
# ---------------------------------------------------------------------------

def _as_grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2, "grids must be two-dimensional")
    _require(np.issubdtype(grid.dtype, np.integer), "grid cells must be integers")
    _require(all(1 <= d <= MAX_SIDE for d in grid.shape), "grid dimensions must be 1..30")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), "colours must be 0..9")
    return grid.astype(int)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys must be train/test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "illegal pair counts")
    seen = set()
    facts = {"train": [], "test": []}
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pairs need input/output")
            source, target = _as_grid(pair["input"]), _as_grid(pair["output"])
            _require(source.shape == target.shape, "output must keep the input size")
            _require(not np.array_equal(source, target), "identity pair")
            key = (source.shape, source.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            grid, background, rect, cycle = _parse(source)
            totals = np.bincount(source.ravel(), minlength=10)
            _require(totals[background] > max(int(totals[c]) for c in range(10) if c != background),
                     "background must be the strict majority colour")
            _require(background not in cycle, "core colours exclude background")
            _require(_background_share(source, background, rect) >= MIN_BACKGROUND_SHARE,
                     "background must dominate the area outside the core")
            r0, c0, r1, c1 = rect
            _require(np.array_equal(target[r0:r1 + 1, c0:c1 + 1], source[r0:r1 + 1, c0:c1 + 1]),
                     "core must be unchanged")
            _require(set(int(x) for x in np.unique(target)) <= {background, *cycle}, "output palette")
            _require(np.array_equal(solve(source), target), "solve disagrees with stored output")
            facts[split].append(_facts(source, target))
    missing = _evidence(facts["train"], facts["test"])
    _require(not missing, f"training does not demonstrate every part of the rule: {sorted(missing)}")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _arm_segments(facts, side):
    """Per crossing line of one arm: list of (start, end) token runs along the ray."""
    grid, background, rect = facts["grid"], facts["bg"], facts["rect"]
    n, length = _arm_info(grid.shape, rect)[side]
    result = []
    for i in range(n):
        runs, start = [], None
        for t in range(1, length + 2):
            filled = t <= length and grid[_arm_cell(rect, side, i, t)] != background
            if filled and start is None:
                start = t
            if not filled and start is not None:
                runs.append((start, t - 1))
                start = None
        result.append(runs)
    return result


def _signatures(facts):
    """Grid-level structural signatures of one pair (measured, not sampled metadata)."""
    grid, background, rect = facts["grid"], facts["bg"], facts["rect"]
    r0, c0, r1, c1 = rect
    height, width = grid.shape
    wall = False
    for side in SIDES:
        segments = _arm_segments(facts, side)
        for i in range(len(segments)):
            for seg in segments[i]:
                if seg[0] < 2:
                    continue
                run = 1
                while i + run < len(segments) and seg in segments[i + run]:
                    run += 1
                if run >= 5:
                    wall = True
    frame = False
    for k in range(1, max(facts["margins"]) + 1):
        ring = [(r0 - k, x) for x in range(c0 - k, c1 + k + 1)] + [(r1 + k, x) for x in range(c0 - k, c1 + k + 1)]
        ring += [(y, c0 - k) for y in range(r0 - k + 1, r1 + k)] + [(y, c1 + k) for y in range(r0 - k + 1, r1 + k)]
        ring = [cell for cell in ring if 0 <= cell[0] < height and 0 <= cell[1] < width]
        corners = [cell for cell in ring if _locate(cell, rect) == ("corner",)]
        if len(ring) >= 16 and corners:
            filled = np.mean([grid[cell] != background for cell in ring])
            corner_fill = np.mean([grid[cell] != background for cell in corners])
            inner = [_arm_cell(rect, side, i, k - 1) for side in SIDES if k > 1 and facts["margins"][side] >= k
                     for i in range(_arm_info(grid.shape, rect)[side][0])]
            gap = np.mean([grid[cell] == background for cell in inner]) if inner else 1.0
            if filled >= 0.8 and corner_fill >= 0.8 and gap >= 0.7:
                frame = True
    occupied = {side: sum(len(s) for s in _arm_segments(facts, side)) for side in SIDES}
    lopsided = any(facts["margins"][a] and facts["margins"][b] and (occupied[a] == 0) != (occupied[b] == 0)
                   for a, b in ((0, 1), (2, 3)))
    # Uneven multi-line clump: a 4-connected token component inside one arm whose lines hold
    # different numbers of its cells (compaction then deforms it line by line).
    uneven = False
    for side in SIDES:
        n, length = _arm_info(grid.shape, rect)[side]
        cells = {(i, t) for i in range(n) for t in range(1, length + 1)
                 if grid[_arm_cell(rect, side, i, t)] != background}
        seen = set()
        for start in sorted(cells):
            if start in seen:
                continue
            stack, component = [start], []
            seen.add(start)
            while stack:
                i, t = stack.pop()
                component.append((i, t))
                for nxt in ((i + 1, t), (i - 1, t), (i, t + 1), (i, t - 1)):
                    if nxt in cells and nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            per_line = {}
            for i, _ in component:
                per_line[i] = per_line.get(i, 0) + 1
            if len(component) >= 5 and len(per_line) >= 2 and len(set(per_line.values())) >= 2:
                uneven = True
    # Per-line or per-arm colouring: every multi-token line is monochrome although the arms use
    # several colours.
    line_colours = []
    for axis, index, side, cells in _arm_lines(grid.shape, rect):
        colours = {int(grid[c]) for c in cells if grid[c] != background}
        if sum(grid[c] != background for c in cells) >= 2:
            line_colours.append(len(colours))
    line_mono = bool(line_colours) and max(line_colours) == 1 and len(facts["arm_colours"]) >= 2
    return {"wall": wall, "frame": frame, "lopsided": lopsided, "uneven": uneven, "line_mono": line_mono}


def _diversity(seeds):
    """Measured structural statistics over generated puzzles (returned grids only)."""
    stats = {key: 0 for key in (
        "pairs", "rect", "aspect10", "deep8", "edge_any", "opposite", "single_arm", "core_coloured",
        "period2", "period4", "wall", "frame", "lopsided", "corner_new_colour", "long_run15",
        "mixed_period_puzzles", "share_puzzles", "test_longer", "test_fresh_depth", "test_single_arm",
        "test_new_bg", "test_period_non3", "test_orientation", "test_structured", "varied_bg",
        "role_swap", "light_puzzles", "max_arm", "uneven", "line_mono")}
    shapes, depths, cycles, backgrounds = set(), set(), set(), set()
    for seed in seeds:
        puzzle = generate_puzzle(seed)
        facts = {s: [_facts(p["input"], p["output"]) for p in puzzle[s]] for s in ("train", "test")}
        train = facts["train"]
        train_max = max(f["max_count"] for f in train)
        train_depths = {f["depth"] for f in train}
        train_bgs = {f["bg"] for f in train}
        orients = {f["core_shape"][0] > f["core_shape"][1] for f in train if f["core_shape"][0] != f["core_shape"][1]}
        stats["mixed_period_puzzles"] += len({f["p"] for s in facts for f in facts[s]}) > 1
        stats["share_puzzles"] += any(f["core_coloured"] for s in facts for f in facts[s])
        stats["varied_bg"] += len({f["bg"] for s in facts for f in facts[s]}) > 1
        sets = {}
        for f in train:
            sets.setdefault(frozenset(f["cycle"]), set()).add(f["cycle"])
        stats["role_swap"] += any(len(v) > 1 for v in sets.values())
        stats["light_puzzles"] += max(f["arm_tokens"] for f in train) <= 25
        for split in ("train", "test"):
            for f in facts[split]:
                sig = _signatures(f)
                h, w = f["core_shape"]
                margins = f["margins"]
                stats["pairs"] += 1
                stats["rect"] += h != w
                stats["aspect10"] += abs(h - w) >= 10
                stats["deep8"] += f["depth"] >= 8
                stats["edge_any"] += min(margins) == 0
                stats["opposite"] += (margins[0] == margins[1] == 0) or (margins[2] == margins[3] == 0)
                stats["single_arm"] += f["open_sides"] == 1
                stats["core_coloured"] += f["core_coloured"] > 0
                stats["period2"] += f["p"] == 2
                stats["period4"] += f["p"] == 4
                stats["wall"] += sig["wall"]
                stats["frame"] += sig["frame"]
                stats["lopsided"] += sig["lopsided"]
                stats["uneven"] += sig["uneven"]
                stats["line_mono"] += sig["line_mono"]
                stats["corner_new_colour"] += bool(f["corner_colours"] - f["arm_colours"])
                stats["long_run15"] += f["max_count"] >= 15
                stats["max_arm"] = max(stats["max_arm"], f["max_count"])
                shapes.add(f["grid"].shape)
                depths.add(f["depth"])
                cycles.add(f["cycle"])
                backgrounds.add(f["bg"])
                if split == "test":
                    stats["test_longer"] += f["max_count"] > train_max
                    stats["test_fresh_depth"] += f["depth"] not in train_depths
                    stats["test_single_arm"] += f["open_sides"] == 1
                    stats["test_new_bg"] += f["bg"] not in train_bgs
                    stats["test_period_non3"] += f["p"] != 3
                    stats["test_orientation"] += (h != w and len(orients) == 1 and (h > w) not in orients)
                    stats["test_structured"] += sig["wall"] or sig["frame"]
    stats.update(shapes=len(shapes), depths=sorted(depths), cycles=len(cycles), backgrounds=len(backgrounds))
    return stats


def self_test() -> None:
    # Robust core detection: core-coloured tokens touching the core (review D1).
    grid = np.zeros((12, 12), dtype=int)
    grid[3:9, 3:9] = _core_block(6, 6, (1, 2, 3))
    grid[2, 3], grid[1, 3], grid[3, 9], grid[3, 11] = 1, 5, 1, 4
    assert _find_cores(grid, 0) == [(3, 3, 8, 8, (1, 2, 3))]
    expected = np.zeros_like(grid)
    expected[3:9, 3:9] = grid[3:9, 3:9]
    expected[2, 3], expected[1, 3], expected[3, 9], expected[3, 10] = 3, 2, 3, 2
    assert np.array_equal(solve(grid), expected)
    # A complete one-colour ring at distance 1 is itself a ring of a larger valid core (review
    # D2), so generation must never leave one; gapped rings stay tokens.
    grid = np.zeros((14, 14), dtype=int)
    grid[4:10, 4:10] = _core_block(6, 6, (1, 2, 3))
    grid[3, 3:11] = grid[10, 3:11] = grid[3:11, 3] = grid[3:11, 10] = 3
    assert _find_cores(grid, 0) == [(3, 3, 10, 10, (3, 1, 2))]
    grid[3, 3] = 0
    assert _find_cores(grid, 0) == [(4, 4, 9, 9, (1, 2, 3))]
    # Two- and four-colour cycles continue outward with the predecessor relation.
    for cycle in ((4, 7), (1, 2, 3, 6)):
        depth = max(3, len(cycle)) + 1
        side = 2 * depth - 1
        grid = np.zeros((side + 8, side + 2), dtype=int)
        grid[1:side + 1, 1:side + 1] = _core_block(side, side, cycle)
        grid[side + 3, 2], grid[side + 6, 2], grid[side + 7, 2] = 5, 8, 9
        out = solve(grid)
        assert [int(out[side + d, 2]) for d in (1, 2, 3)] == [_outward(cycle, d) for d in (1, 2, 3)]
        assert int(out[side + 4, 2]) == 0

    stats = _diversity(range(42))
    # Thresholds sit below the values measured for seeds 0..41 (see review_coverage.md).
    assert stats["shapes"] >= 100 and stats["cycles"] >= 120 and stats["backgrounds"] >= 9, stats
    assert set(stats["depths"]) >= {3, 4, 5, 6, 7, 8}, stats
    assert stats["rect"] >= 70 and stats["aspect10"] >= 12 and stats["deep8"] >= 4, stats
    assert stats["edge_any"] >= 25 and stats["opposite"] >= 4 and stats["single_arm"] >= 3, stats
    assert stats["core_coloured"] >= 30 and stats["share_puzzles"] >= 8, stats
    assert stats["period2"] >= 1 and stats["period4"] >= 4 and stats["mixed_period_puzzles"] >= 3, stats
    assert stats["wall"] >= 15 and stats["frame"] >= 8 and stats["lopsided"] >= 60, stats
    assert stats["uneven"] >= 90 and stats["line_mono"] >= 25 and stats["corner_new_colour"] >= 50, stats
    assert stats["long_run15"] >= 4 and stats["max_arm"] >= 20, stats
    assert stats["test_longer"] >= 10 and stats["test_fresh_depth"] >= 12 and stats["test_structured"] >= 6, stats
    assert min(stats["test_single_arm"], stats["test_period_non3"], stats["test_orientation"],
               stats["test_new_bg"]) >= 1, stats
    assert stats["varied_bg"] >= 8 and stats["role_swap"] >= 10 and stats["light_puzzles"] >= 3, stats

    for counts in ((2, 1), (3, 2), (10, 1), (10, 4), (5, 4)):
        puzzle = generate_puzzle(7 + sum(counts), *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        assert _same(puzzle, generate_puzzle(7 + sum(counts), *counts))
    assert not _same(generate_puzzle(0), generate_puzzle(1))

    # The validator must reject corrupted labels and every applicable shortcut rule.
    for seed in (0, 9):  # seed 9 mixes 2-, 3- and 4-colour cycles, so `mod3` applies
        sample = generate_puzzle(seed)
        validate_generated_puzzle(sample)
        periods = {len(_parse(p["input"])[3]) for s in sample for p in sample[s]}
        assert seed == 0 or periods - {3}, periods
        broken = {s: [dict(p) for p in sample[s]] for s in sample}
        target = np.array(broken["train"][0]["output"])
        target[0, 0] = (target[0, 0] + 1) % 10
        broken["train"][0]["output"] = target.tolist()
        modes = [None] + [m for m in ALTERNATIVES if m != "mod3" or periods - {3}]
        for mode in modes:
            if mode is not None:
                broken = {s: [dict(p) for p in sample[s]] for s in sample}
                for p in broken["train"]:
                    grid, background, rect, cycle = _parse(p["input"])
                    p["output"] = _render(grid, background, rect, cycle, mode).tolist()
                assert any(p["output"] != q["output"] for p, q in zip(broken["train"], sample["train"])), mode
            try:
                validate_generated_puzzle(broken)
            except ValueError:
                continue
            raise AssertionError(f"validator accepted a corrupted puzzle ({mode})")
    for bad in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(5, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {bad}")
