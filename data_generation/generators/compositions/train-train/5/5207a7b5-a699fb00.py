"""Dotted rails that hang arithmetic staircase curtains.

Composition of ARC 5207a7b5 (parent A: one seed bar defines a staircase of
bars, two cells longer per column to the left and two cells shorter per column
to the right, clipped at the canvas edge) and ARC a699fb00 (parent B: a
horizontal dotted chain marker-gap-marker... has each single bracketed gap
filled with a connector colour; separate chains and lone markers stay apart).

Composed rule (see task_spec.md):
  * A *rail* is a horizontal dotted chain of one marker colour with exactly one
    background cell between consecutive markers.  Exactly one of its gaps is
    already plugged by the end of a vertical *seed* bar (length L >= 2) whose
    colour differs from the marker.  The seed hangs below the rail (DOWN) or
    stands above it (UP); its other end is free.
  * Every gap of the rail is painted with the seed colour (B, with the
    connector colour supplied by the seed).
  * Every rail column j between the first and last marker receives a bar that
    starts in the rail row and extends towards the seed's side.  Its length is
    max(1, L + 2 * (c - j)), where c is the seed column (A's staircase), clipped
    at the canvas edge.  Marker columns take the marker colour and gap columns
    the seed colour, so the completed rail's alternation continues down the
    curtain.  Columns outside the rail receive nothing.
  * Lone markers, cells outside curtains and the background are unchanged.
"""
from __future__ import annotations

import numpy as np

DOWN = 1
UP = -1

# Training-pair roles.
_RICH = "rich"
_SAME_ROW = "same_row"
_CLIP = "clip"
_RIGHT_CUT = "right_cut"
_NO_CLIP = "no_clip"


def _marker_count(rng, low, high):
    """Markers per rail; two-marker rails (no visible gap) are kept rare."""
    if high <= low:
        return low
    if low == 2 and high >= 3 and rng.random() < 0.85:
        return int(rng.integers(3, high + 1))
    return int(rng.integers(low, high + 1))


# ---------------------------------------------------------------------------
# Input-only rule
# ---------------------------------------------------------------------------

def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_rails(grid):
    """Detect (row, a, b, c, L, orient, seed_colour, marker) for every rail.

    A seed is a vertical run (length >= 2) of one non-background colour.  The
    end of the run that sits horizontally between two equal cells of another
    non-background colour lies in the rail row; those neighbours are markers.
    The rail extends while the pattern background, marker continues.
    """
    grid = np.asarray(grid, dtype=int)
    height, width = grid.shape
    bg = _background(grid)
    rails = []
    for col in range(width):
        row = 0
        while row < height:
            colour = int(grid[row, col])
            if colour == bg:
                row += 1
                continue
            end = row
            while end + 1 < height and grid[end + 1, col] == colour:
                end += 1
            if end > row and 0 < col < width - 1:
                for rail_row, orient in ((row, DOWN), (end, UP)):
                    left = int(grid[rail_row, col - 1])
                    right = int(grid[rail_row, col + 1])
                    if left == right and left not in (bg, colour):
                        a = col - 1
                        while (a - 2 >= 0 and grid[rail_row, a - 1] == bg
                               and grid[rail_row, a - 2] == left):
                            a -= 2
                        b = col + 1
                        while (b + 2 < width and grid[rail_row, b + 1] == bg
                               and grid[rail_row, b + 2] == left):
                            b += 2
                        rails.append((rail_row, a, b, col, end - row + 1,
                                      orient, colour, left))
                        break
            row = end + 1
    return rails


def _bar_span(height, row, length, orient):
    """Inclusive row interval of a bar of `length` cells anchored at `row`."""
    if orient == DOWN:
        return row, min(height - 1, row + length - 1)
    return max(0, row - length + 1), row


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    out = grid.copy()
    height = grid.shape[0]
    for row, a, b, c, length, orient, seed, marker in _find_rails(grid):
        for j in range(a, b + 1):
            extent = max(1, length + 2 * (c - j))
            top, bottom = _bar_span(height, row, extent, orient)
            out[top:bottom + 1, j] = marker if (j - a) % 2 == 0 else seed
    return out


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------

def _footprint(height, width, rail):
    row, a, b, c, length, orient = rail[:6]
    mask = np.zeros((height, width), dtype=bool)
    for j in range(a, b + 1):
        top, bottom = _bar_span(height, row, max(1, length + 2 * (c - j)), orient)
        mask[top:bottom + 1, j] = True
    return mask


def _dilate(mask):
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr:dr + h, dc:dc + w]
    return out


def _right_cut(width, rail):
    """True when the rail ends while the staircase would still hang a bar."""
    row, a, b, c, length = rail[:5]
    return b + 1 < width and length - 2 * (b + 1 - c) >= 2


def _floored(rail):
    """True when some gap right of the seed gets only its rail cell (the
    staircase has run out there, but B's gap completion still applies)."""
    row, a, b, c, length = rail[:5]
    return any(length + 2 * (c - j) <= 1 for j in range(c + 1, b) if (j - a) % 2 == 1)


def _clipped(height, rail):
    """True when the longest (leftmost) bar is cut by the canvas edge."""
    row, a, _, c, length, orient = rail[:6]
    longest = length + 2 * (c - a)
    if orient == DOWN:
        return row + longest > height
    return row - longest + 1 < 0


def _render(height, width, bg, marker, rails, lones):
    inp = np.full((height, width), bg, dtype=int)
    for row, a, b, c, length, orient, seed in rails:
        inp[row, a:b + 1:2] = marker
        top, bottom = _bar_span(height, row, length, orient)
        inp[top:bottom + 1, c] = seed
    for row, col in lones:
        inp[row, col] = marker
    out = inp.copy()
    for row, a, b, c, length, orient, seed in rails:
        for j in range(a, b + 1):
            top, bottom = _bar_span(height, row, max(1, length + 2 * (c - j)), orient)
            out[top:bottom + 1, j] = marker if (j - a) % 2 == 0 else seed
    return inp, out


class _Layout:
    """Occupancy bookkeeping so objects and curtains never touch."""

    def __init__(self, height, width):
        self.height, self.width = height, width
        self.occupied = np.zeros((height, width), dtype=bool)
        self.markers = np.zeros((height, width), dtype=bool)
        self.rails = []
        self.lones = []

    def fits(self, rail):
        row, a, b, c, length, orient = rail[:6]
        if a < 0 or b >= self.width or not 0 <= row < self.height:
            return False
        top, bottom = _bar_span(self.height, row, length, orient)
        # The seed must be complete: its free end may not touch the edge.
        if bottom - top + 1 != length or top == 0 and orient == UP:
            return False
        if orient == DOWN and bottom == self.height - 1:
            return False
        mask = _footprint(self.height, self.width, rail)
        if np.any(mask & _dilate(self.occupied)):
            return False
        for col in (a - 2, b + 2):
            if 0 <= col < self.width and self.markers[row, col]:
                return False
        return True

    def add(self, rail):
        row, a, b = rail[0], rail[1], rail[2]
        self.occupied |= _footprint(self.height, self.width, rail)
        self.markers[row, a:b + 1:2] = True
        self.rails.append(rail)

    def lone_fits(self, row, col):
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        if _dilate(self.occupied)[row, col]:
            return False
        for other in (col - 2, col + 2):
            if 0 <= other < self.width and self.markers[row, other]:
                return False
        # A lone marker in a rail row must not sit two cells from its ends.
        for rail in self.rails:
            if rail[0] == row and col in (rail[1] - 2, rail[2] + 2):
                return False
        return True

    def add_lone(self, row, col):
        self.occupied[row, col] = True
        self.markers[row, col] = True
        self.lones.append((row, col))


def _sample_rail(rng, height, width, orient, kind, edge_seed=False):
    """Sample (row, a, b, c, L, orient) or None. kind: rich/clip/right_cut/plain."""
    max_markers = (width + 1) // 2
    right_limit = width
    if kind == _RIGHT_CUT:
        # Seed in the last or second-to-last gap and long enough that the
        # staircase would still continue past the rail's right end.
        max_markers = (width - 1 + 1) // 2
        right_limit = width - 1
        if max_markers < 2:
            return None
        k = _marker_count(rng, 2, min(8, max_markers))
        gap = k - 2 if k < 3 or rng.random() < 0.6 else k - 3
        need = 2 * (2 * (k - 2 - gap) + 2) + 2
        if need > min(13, height - 2):
            return None
        length = int(rng.integers(need, min(13, height - 2) + 1))
    elif kind == _RICH:
        if max_markers < 4:
            return None
        k = int(rng.integers(4, min(9, max_markers) + 1))
        gap = int(rng.integers(1, k - 2))
        length = int(rng.integers(6, 10))
    else:
        if max_markers < 2 or height < 4:
            return None
        k = _marker_count(rng, 2, min(12, max_markers))
        if edge_seed:
            gap = 0 if rng.random() < 0.5 else k - 2
        else:
            gap = int(rng.integers(0, k - 1))
        length = int(rng.integers(2, min(12, height - 2) + 1))
    span = 2 * k - 1
    if right_limit - span < 0:
        return None
    a = int(rng.integers(0, right_limit - span + 1))
    b = a + span - 1
    c = a + 2 * gap + 1
    longest = length + 2 * (c - a)
    if orient == DOWN:
        low, high = 0, height - length - 1
        if kind in (_RICH, _NO_CLIP):
            high = min(high, height - longest - 1)
        elif kind == _CLIP:
            low = max(low, height - longest + 1)
    else:
        low, high = length, height - 1
        if kind in (_RICH, _NO_CLIP):
            low = max(low, longest)
        elif kind == _CLIP:
            high = min(high, longest - 2)
    if low > high:
        return None
    row = int(rng.integers(low, high + 1))
    return (row, a, b, c, length, orient)


def _sample_rail_in_row(rng, height, width, row, orient, col_low, col_high):
    """Rail confined to columns [col_low, col_high] in a fixed row."""
    available = col_high - col_low + 1
    max_markers = (available + 1) // 2
    max_length = height - 1 - row if orient == DOWN else row
    max_length = min(max_length, 12)
    if max_markers < 2 or max_length < 2:
        return None
    k = _marker_count(rng, 2, min(10, max_markers))
    gap = int(rng.integers(0, k - 1))
    length = int(rng.integers(2, max_length + 1))
    span = 2 * k - 1
    a = int(rng.integers(col_low, col_high - span + 2))
    return (row, a, a + span - 1, a + 2 * gap + 1, length, orient)


def _dims(rng, plan):
    low_h, low_w = 6, 5
    if plan.get(_RICH):
        low_h, low_w = max(low_h, 15), max(low_w, 9)
    if plan.get(_SAME_ROW):
        low_w = max(low_w, 12)
    rails = plan.get("rails", 1)
    low_w = max(low_w, min(18, 2 + 3 * rails))
    low_h = max(low_h, min(18, 2 + 3 * rails))
    roll = 0.0 if plan.get("compact") else rng.random()
    if roll < 0.4:
        # Compact canvas near the lower bound.
        height = int(rng.integers(low_h, min(30, low_h + 5) + 1))
        width = int(rng.integers(low_w, min(30, low_w + 5) + 1))
    elif roll < 0.6:
        height = int(rng.integers(max(low_h, 18), 31))
        width = int(rng.integers(max(low_w, 18), 31))
    else:
        height = int(rng.integers(low_h, 31))
        width = int(rng.integers(low_w, 31))
    return height, width


def _pick_colour(rng, pool, avoid=()):
    choices = [c for c in pool if c not in avoid]
    if not choices:
        choices = list(pool)
    return int(rng.choice(choices))


def _build_pair(rng, plan, bg, marker, orient_mode):
    """Build one pair satisfying `plan`; returns (input, output)."""
    pool = [c for c in range(10) if c not in (bg, marker)]
    for _ in range(400):
        height, width = _dims(rng, plan)
        lay = _Layout(height, width)
        colours = []
        plain = _NO_CLIP if plan.get(_NO_CLIP) else "plain"

        def orient_for(index):
            if orient_mode == "down":
                return DOWN
            if orient_mode == "up":
                return UP
            if plan.get("mixed") and index < 2:
                return plan["mixed_order"][index]
            return DOWN if rng.random() < 0.5 else UP

        def place(kind, index, edge=False):
            orient = orient_for(index)
            for _ in range(40):
                rail = _sample_rail(rng, height, width, orient, kind, edge)
                if rail is not None and lay.fits(rail):
                    colour = _pick_colour(rng, pool, colours[-1:] if rng.random() < 0.7 else ())
                    colours.append(colour)
                    lay.add(rail + (colour,))
                    return True
            return False

        ok = True
        index = 0
        if plan.get(_RICH):
            ok = place(_RICH, index)
            index += 1
        if ok and plan.get(_SAME_ROW):
            orient_first = orient_for(index)
            orient_second = orient_for(index + 1)
            ok = False
            for _ in range(60):
                first = _sample_rail(rng, height, width, orient_first,
                                     _CLIP if plan.get(_CLIP) else plain)
                if first is None or not lay.fits(first):
                    continue
                row, a, b = first[0], first[1], first[2]
                separation = int(rng.choice([2, 2, 3, 3, 4, 5]))
                side = rng.random() < 0.5
                options = []
                if b + 1 + separation + 2 < width:
                    options.append((b + 1 + separation, width - 1))
                if a - 1 - separation - 2 >= 0:
                    options.append((0, a - 1 - separation))
                if not options:
                    continue
                low, high = options[0] if side or len(options) == 1 else options[1]
                second = _sample_rail_in_row(rng, height, width, row, orient_second, low, high)
                if second is None:
                    continue
                # Check both together against the current layout.
                if not lay.fits(second):
                    continue
                trial = _Layout(height, width)
                trial.occupied = lay.occupied.copy()
                trial.markers = lay.markers.copy()
                trial.rails = list(lay.rails)
                trial.add(first + (0,))
                if not trial.fits(second):
                    continue
                c1 = _pick_colour(rng, pool)
                c2 = _pick_colour(rng, pool, (c1,))
                lay.add(first + (c1,))
                lay.add(second + (c2,))
                colours.extend([c1, c2])
                ok = True
                break
            index += 2
        if ok and plan.get(_CLIP) and not any(_clipped(height, r) for r in lay.rails):
            ok = place(_CLIP, index)
            index += 1
        if ok and plan.get(_RIGHT_CUT) and not any(_right_cut(width, r) for r in lay.rails):
            ok = place(_RIGHT_CUT, index)
            index += 1
        if ok and plan.get("edge_seed"):
            ok = place(plain, index, edge=True)
            index += 1
        if ok and not lay.rails:
            ok = place(plain, index)
            index += 1
        if not ok:
            continue
        for _ in range(plan.get("extra", 0)):
            place(plain, index)
            index += 1
        if not lay.rails:
            continue
        if plan.get("mixed"):
            if len({r[5] for r in lay.rails}) < 2:
                continue
        if plan.get(_NO_CLIP) and any(_clipped(height, r) for r in lay.rails):
            continue
        if plan.get("floor") and not any(_floored(r) for r in lay.rails):
            continue
        # Break marker: a lone marker two or three background cells past a rail end.
        if plan.get("break"):
            placed = False
            for rail in rng.permutation(len(lay.rails)):
                row, a, b = lay.rails[int(rail)][:3]
                spots = [a - 3, a - 4, b + 3, b + 4]
                rng.shuffle(spots)
                for col in spots:
                    if lay.lone_fits(row, int(col)):
                        lay.add_lone(row, int(col))
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                continue
        for _ in range(plan.get("lones", 0)):
            for _ in range(30):
                row = int(rng.integers(0, height))
                col = int(rng.integers(0, width))
                if lay.lone_fits(row, col):
                    lay.add_lone(row, col)
                    break
        inp, out = _render(height, width, bg, marker, lay.rails, lay.lones)
        # The input must parse back into exactly the intended rails.
        found = sorted(r[:7] for r in _find_rails(inp))
        if found != sorted(lay.rails):
            continue
        values, counts = np.unique(inp, return_counts=True)
        if int(values[int(np.argmax(counts))]) != bg or 2 * counts[values == bg][0] <= inp.size:
            continue
        return inp, out
    raise RuntimeError("could not build a pair")


# ---------------------------------------------------------------------------
# Puzzle generation
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    if num_test is None:
        num_test = 1
    if num_train is None:
        num_train = 4 if isinstance(num_test, bool) or not isinstance(num_test, (int, np.integer)) \
            else max(4, int(num_test) + 1)
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be in 2..10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _train_plans(rng, num_train, orient_mode):
    mixed_order = [DOWN, UP] if rng.random() < 0.5 else [UP, DOWN]
    plans = [
        {_RICH: True, "rails": 1 + int(rng.random() < 0.35), "extra": int(rng.random() < 0.35),
         "lones": int(rng.integers(0, 2))},
        {_SAME_ROW: True, _CLIP: True, "rails": 2, "extra": int(rng.random() < 0.3),
         "lones": int(rng.integers(0, 3)), "break": rng.random() < 0.3, "floor": True},
    ]
    if orient_mode == "mixed":
        plans[1]["mixed"] = True
        plans[1]["mixed_order"] = mixed_order
    # A rail whose curtain is cut at its right end although the staircase
    # would continue: shows the rail, not the staircase, bounds the curtain.
    plans[int(rng.integers(0, 2))][_RIGHT_CUT] = True
    for _ in range(num_train - 2):
        roll = rng.random()
        plan = {"rails": 1, "extra": int(rng.integers(0, 3)), "lones": int(rng.integers(0, 3))}
        if roll < 0.25:
            plan[_SAME_ROW] = True
        if rng.random() < 0.4:
            plan[_CLIP] = True
        if rng.random() < 0.2:
            plan["break"] = True
        if rng.random() < 0.15:
            plan["edge_seed"] = True
        plan["rails"] = 1 + plan["extra"] + int(plan.get(_SAME_ROW, False))
        plans.append(plan)
    return plans


def _test_plan(rng):
    if rng.random() < 0.2:
        # Compact test: one or two rails on a small canvas.
        plan = {"compact": True, "extra": int(rng.random() < 0.4), "lones": int(rng.integers(0, 2))}
        if rng.random() < 0.5:
            plan[_CLIP] = True
        elif rng.random() < 0.6:
            plan[_NO_CLIP] = True
        if rng.random() < 0.4:
            plan["edge_seed"] = True
        plan["rails"] = 1 + plan["extra"] + int(plan.get("edge_seed", False))
        return plan
    extra = int(rng.choice([0, 1, 1, 2, 2, 3]))
    plan = {"extra": extra, "lones": int(rng.integers(0, 4))}
    if rng.random() < 0.5:
        plan[_SAME_ROW] = True
    if rng.random() < 0.45:
        plan[_CLIP] = True
    elif rng.random() < 0.5:
        plan[_NO_CLIP] = True
    if rng.random() < 0.35:
        plan["break"] = True
    if rng.random() < 0.4:
        plan["edge_seed"] = True
    if rng.random() < 0.3:
        plan[_RIGHT_CUT] = True
    plan["rails"] = 1 + extra + int(plan.get(_SAME_ROW, False))
    return plan


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    orient_mode = str(rng.choice(["down", "down", "up", "mixed", "mixed"]))
    palette_mode = "variable" if rng.random() < 0.45 else "stable"
    base = [int(v) for v in rng.permutation(10)]
    base_bg = base[0] if rng.random() < 0.6 else 0
    base_marker = base[1] if base[1] != base_bg else base[2]
    child_seeds = rng.integers(0, 2**63 - 1, size=num_train + num_test)
    palette_seeds = rng.integers(0, 2**63 - 1, size=num_train + num_test)

    def palette(index, train_index=None):
        if palette_mode == "stable" or train_index == 0:
            return base_bg, base_marker
        prng = np.random.default_rng(int(palette_seeds[index]))
        bg = int(prng.integers(0, 10)) if prng.random() < 0.7 else base_bg
        marker = int(prng.choice([c for c in range(10) if c != bg]))
        if train_index == 1 and (bg, marker) == (base_bg, base_marker):
            marker = int(prng.choice([c for c in range(10) if c not in (bg, base_marker)]))
        return bg, marker

    plans = _train_plans(rng, num_train, orient_mode)
    seen = set()
    train = []
    for index, plan in enumerate(plans):
        prng = np.random.default_rng(int(child_seeds[index]))
        bg, marker = palette(index, index)
        for _ in range(20):
            inp, out = _build_pair(prng, plan, bg, marker, orient_mode)
            key = (inp.shape, inp.tobytes())
            if key not in seen:
                seen.add(key)
                train.append({"input": inp, "output": out})
                break
        else:
            raise RuntimeError("duplicate training inputs")
    order = rng.permutation(num_train)
    train = [train[int(i)] for i in order]
    train_orients = set()
    for pair in train:
        train_orients.update(r[5] for r in _find_rails(pair["input"]))
    test = []
    for index in range(num_test):
        prng = np.random.default_rng(int(child_seeds[num_train + index]))
        bg, marker = palette(num_train + index)
        plan = _test_plan(prng)
        for _ in range(20):
            inp, out = _build_pair(prng, plan, bg, marker, orient_mode)
            key = (inp.shape, inp.tobytes())
            if key not in seen and {r[5] for r in _find_rails(inp)} <= train_orients:
                seen.add(key)
                test.append({"input": inp, "output": out})
                break
        else:
            raise RuntimeError("could not build a test pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _decompose(inp):
    """Split an input into background, marker, rails and lone markers."""
    height, width = inp.shape
    values, counts = np.unique(inp, return_counts=True)
    bg = int(values[int(np.argmax(counts))])
    _require(2 * int(counts[values == bg][0]) > inp.size, "background is not a majority")
    rails = _find_rails(inp)
    _require(len(rails) >= 1, "no rail with a seed")
    claimed = np.zeros(inp.shape, dtype=bool)
    markers = {r[7] for r in rails}
    _require(len(markers) == 1, "rails use different marker colours")
    marker = markers.pop()
    for row, a, b, c, length, orient, seed, _ in rails:
        _require(seed not in (bg, marker), "seed colour clashes with a role")
        cells = np.zeros(inp.shape, dtype=bool)
        cells[row, a:b + 1:2] = True
        top, bottom = _bar_span(height, row, length, orient)
        _require(bottom - top + 1 == length and length >= 2, "seed is clipped or short")
        _require(not (orient == DOWN and bottom == height - 1)
                 and not (orient == UP and top == 0), "seed touches the far edge")
        cells[top:bottom + 1, c] = True
        _require(not np.any(cells & claimed), "rails overlap")
        claimed |= cells
        _require(np.all(inp[row, a:b + 1:2] == marker), "rail marker missing")
        gaps = [j for j in range(a + 1, b, 2) if j != c]
        _require(all(inp[row, j] == bg for j in gaps), "rail gap not empty")
        # The free end of the seed must not look like a rail cell.
        free = bottom if orient == DOWN else top
        if 0 < c < width - 1:
            _require(not (inp[free, c - 1] == inp[free, c + 1] != bg), "ambiguous seed end")
    lones = []
    rest = (inp != bg) & ~claimed
    padded = np.pad(inp != bg, 1)
    for row, col in zip(*np.nonzero(rest)):
        _require(inp[row, col] == marker, "unexplained non-background colour")
        window = padded[row:row + 3, col:col + 3]
        _require(int(window.sum()) == 1, "lone marker touches another object")
        for other in (col - 2, col + 2):
            if 0 <= other < width:
                _require(inp[row, other] != marker, "lone markers form an unseeded chain")
        lones.append((int(row), int(col)))
    return bg, marker, [r[:7] for r in rails], lones


def _expected(inp, rails, marker):
    """Cell-by-cell construction of the composed output."""
    height, width = inp.shape
    out = inp.copy()
    for i in range(height):
        for j in range(width):
            for row, a, b, c, length, orient, seed in rails:
                if not a <= j <= b:
                    continue
                extent = max(1, length + 2 * (c - j))
                distance = (i - row) * orient
                if 0 <= distance < extent:
                    out[i, j] = marker if (j - a) % 2 == 0 else seed
    return out


def _variant(inp, rails, marker, *, step=2, bounds="rail", stripes=True):
    """Near-miss rules used to show that the demonstrations rule them out.

    bounds="rail"   : the composed rule (curtain only under the rail);
    bounds="canvas" : staircase across the whole canvas width (ignores B's extent);
    bounds="row"    : rail runs from the first to the last marker in its row
                      (ignores B's exact one-cell gap, joining breaks/rails);
    step            : length change per column (A uses 2);
    stripes=False   : the whole curtain in the seed colour.
    """
    height, width = inp.shape
    out = inp.copy()
    for row, a, b, c, length, orient, seed in rails:
        lo, hi = a, b
        if bounds == "canvas":
            lo, hi = 0, width - 1
        elif bounds == "row":
            cols = np.flatnonzero(inp[row] == marker)
            lo, hi = int(cols.min()), int(cols.max())
        for j in range(lo, hi + 1):
            top, bottom = _bar_span(height, row, max(1, length + step * (c - j)), orient)
            colour = marker if (j - a) % 2 == 0 else seed
            out[top:bottom + 1, j] = colour
            if not stripes and top < bottom:
                body = slice(top, bottom) if orient == UP else slice(top + 1, bottom + 1)
                out[body, j] = seed
    return out


def _shortcut_a_only(inp, rails):
    """Parent A alone: seed-coloured staircase per seed, no rail role."""
    height, width = inp.shape
    out = inp.copy()
    for row, a, b, c, length, orient, seed in rails:
        for j in range(width):
            extent = length + 2 * (c - j)
            if extent > 0:
                top, bottom = _bar_span(height, row, extent, orient)
                out[top:bottom + 1, j] = seed
    return out


def _shortcut_b_only(inp, rails, marker):
    """Parent B alone: only the rail gaps are completed (with the seed colour)."""
    out = inp.copy()
    for row, a, b, c, length, orient, seed in rails:
        out[row, a + 1:b:2] = seed
    return out


def _as_array(grid):
    array = np.asarray(grid)
    _require(array.ndim == 2, "grid is not 2D")
    _require(np.issubdtype(array.dtype, np.integer), "grid is not integer")
    _require(1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30, "grid size out of range")
    _require(bool(np.all((array >= 0) & (array <= 9))), "colour out of range")
    return array.astype(int)


def _rich(height, rail):
    row, a, b, c, length, orient = rail[:6]
    return (not _clipped(height, rail) and c - a >= 3 and b - c >= 3 and length >= 6)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair keys")
            inp, out = _as_array(pair["input"]), _as_array(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            bg, marker, rails, lones = _decompose(inp)
            expected = _expected(inp, rails, marker)
            _require(np.array_equal(out, expected), "output does not follow the rule")
            _require(np.array_equal(solve(inp), out), "solve disagrees")
            _require(not np.array_equal(inp, out), "identity pair")
            # Curtains and lone markers never touch one another.
            height, width = inp.shape
            masks = [_footprint(height, width, r) for r in rails]
            for row, col in lones:
                m = np.zeros(inp.shape, dtype=bool)
                m[row, col] = True
                masks.append(m)
            for i in range(len(masks)):
                for j in range(i + 1, len(masks)):
                    _require(not np.any(masks[i] & _dilate(masks[j])), "objects touch")
            # Both parent contributions matter on every pair.
            _require(not np.array_equal(_shortcut_a_only(inp, rails), out), "A-only shortcut works")
            _require(not np.array_equal(_shortcut_b_only(inp, rails, marker), out), "B-only shortcut works")
            _require(not np.array_equal(_variant(inp, rails, marker, stripes=False), out),
                     "monochrome curtain works")
            info[split].append({
                "shape": inp.shape, "bg": bg, "marker": marker, "rails": rails, "lones": lones,
                "unbounded": not np.array_equal(_variant(inp, rails, marker, bounds="canvas"), out),
                "joined": not np.array_equal(_variant(inp, rails, marker, bounds="row"), out),
                "step1": not np.array_equal(_variant(inp, rails, marker, step=1), out),
            })
    tr = info["train"]
    _require(any(_rich(p["shape"][0], r) for p in tr for r in p["rails"]),
             "training lacks an unclipped two-sided staircase")
    _require(any(_clipped(p["shape"][0], r) for p in tr for r in p["rails"]),
             "training lacks a clipped staircase")
    _require(any(_floored(r) for p in tr for r in p["rails"]),
             "training never fills a gap beyond the staircase")
    _require(any(len({r[0] for r in p["rails"]}) < len(p["rails"]) for p in tr),
             "training lacks two rails in one row")
    _require(any(p["unbounded"] for p in tr), "training never bounds the curtain by the rail")
    _require(any(_right_cut(p["shape"][1], r) for p in tr for r in p["rails"]),
             "training never cuts a continuing staircase at a rail end")
    _require(any(p["joined"] for p in tr), "training never shows a rail break")
    _require(any(p["step1"] for p in tr), "training does not fix the two-cell step")
    _require(len({r[6] for p in tr for r in p["rails"]}) >= 2, "training shows one seed colour")
    train_orients = {r[5] for p in tr for r in p["rails"]}
    _require({r[5] for p in info["test"] for r in p["rails"]} <= train_orients,
             "test orientation not demonstrated")


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def self_test() -> None:
    # Hand-made example: rail 1.1.1.1.1.1 with a 3-long seed hanging from column 6.
    grid = np.zeros((10, 13), dtype=int)
    grid[1, 1:12:2] = 1
    grid[1:4, 6] = 3
    grid[1, 6] = 3
    expected = np.array([[int(ch) for ch in row] for row in (
        "0000000000000",
        "0131313131310",
        "0131313000000",
        "0131313000000",
        "0131310000000",
        "0131310000000",
        "0131300000000",
        "0131300000000",
        "0131000000000",
        "0131000000000")])
    _require(np.array_equal(solve(grid), expected), "hand example failed")
    # Upward seed mirrors the curtain; a lone marker three cells away stays put.
    up = np.zeros((8, 12), dtype=int)
    up[6, 2:9:2] = 5
    up[4:7, 5] = 7
    up[6, 11] = 5
    solved = solve(up)
    _require(solved[6, 11] == 5 and solved[6, 10] == 0, "break was bridged")
    _require(list(solved[:, 2]) == [5, 5, 5, 5, 5, 5, 5, 0], "up curtain wrong")
    _require(list(solved[:, 6]) == [0, 0, 0, 0, 0, 0, 5, 0], "up curtain wrong")
    _require(list(solved[:, 4]) == [0, 0, 5, 5, 5, 5, 5, 0], "up curtain wrong")
    _require(list(solved[:, 8]) == [0, 0, 0, 0, 0, 0, 5, 0], "right end wrong")

    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (4, 5), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")
    first = generate_puzzle(7, 3, 2)
    again = generate_puzzle(7, 3, 2)
    for split in ("train", "test"):
        for p, q in zip(first[split], again[split]):
            _require(np.array_equal(p["input"], q["input"]) and np.array_equal(p["output"], q["output"]),
                     "nondeterministic")
    for counts in ((2, 1), (10, 4), (10, 1), (5, 4)):
        puzzle = generate_puzzle(11, *counts)
        _require((len(puzzle["train"]), len(puzzle["test"])) == counts, "count ignored")

    signatures, shapes = set(), set()
    modes = set()
    up_rails = down_rails = same_row_tests = clipped_tests = break_tests = edge_tests = 0
    multi_rail_tests = palette_varied = seed_colours_in_pair = 0
    unclipped_tests = right_cut_tests = mixed_pairs = floored_pairs = small_pairs = 0
    rail_lengths, seed_lengths = set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        sig = tuple((p["input"].tobytes(), p["input"].shape) for s in ("train", "test")
                    for p in puzzle[s])
        signatures.add(sig)
        orients = set()
        palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                shapes.add(inp.shape)
                bg, marker, rails, lones = _decompose(inp)
                palettes.add((bg, marker))
                seed_colours_in_pair += len({r[6] for r in rails}) >= 2
                mixed_pairs += len({r[5] for r in rails}) == 2
                floored_pairs += any(_floored(r) for r in rails)
                small_pairs += inp.size <= 150
                for row, a, b, c, length, orient, colour in rails:
                    orients.add(orient)
                    up_rails += orient == UP
                    down_rails += orient == DOWN
                    rail_lengths.add((b - a) // 2 + 1)
                    seed_lengths.add(length)
                if split == "test":
                    rows = [r[0] for r in rails]
                    same_row_tests += len(set(rows)) < len(rows)
                    clipped_tests += any(_clipped(inp.shape[0], r) for r in rails)
                    multi_rail_tests += len(rails) >= 3
                    edge_tests += any(r[3] in (r[1] + 1, r[2] - 1) for r in rails)
                    break_tests += any(row == r[0] and col in (r[1] - 3, r[1] - 4, r[2] + 3, r[2] + 4)
                                       for row, col in lones for r in rails)
                    unclipped_tests += not any(_clipped(inp.shape[0], r) for r in rails)
                    right_cut_tests += any(_right_cut(inp.shape[1], r) for r in rails)
        modes.add(tuple(sorted(orients)))
        palette_varied += len(palettes) > 1
    _require(len(signatures) == 42, "seeds repeat puzzles")
    _require(len(shapes) >= 60, "too few grid shapes")
    _require(len(modes) == 3, "orientation modes missing")
    _require(up_rails >= 20 and down_rails >= 40, "orientation mix too thin")
    _require(same_row_tests >= 8 and clipped_tests >= 8 and break_tests >= 3, "test regimes narrow")
    _require(edge_tests >= 8 and multi_rail_tests >= 5, "test structures narrow")
    _require(palette_varied >= 10 and seed_colours_in_pair >= 20, "palette roles narrow")
    _require(len(rail_lengths) >= 8 and len(seed_lengths) >= 9, "rail geometry narrow")
    _require(unclipped_tests >= 6 and right_cut_tests >= 6, "test clipping regimes narrow")
    _require(mixed_pairs >= 15 and floored_pairs >= 60 and small_pairs >= 10, "pair structures narrow")


if __name__ == "__main__":
    self_test()
    print("self_test passed")
