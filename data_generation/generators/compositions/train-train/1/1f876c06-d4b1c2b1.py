"""Composition of 1f876c06 (diagonal colour pairing) and d4b1c2b1 (palette-count zoom).

Rule -- "zoom the dots, keep the threads thin":

* k is the number of distinct colours in the input, background included.
* Every input cell becomes a solid k-by-k block of its colour (nearest-neighbour zoom).
* Every colour that occurs exactly twice lies on a common 45-degree diagonal with at
  least one background cell between the two cells.  In the zoomed output its two
  blocks are joined by a one-cell-wide diagonal thread of that colour that runs from
  the facing corner of one block to the facing corner of the other, i.e. it paints
  the main diagonal (down-right pairs) or the anti-diagonal (down-left pairs) of
  every intermediate background block.
* Colours that occur once are zoomed but never threaded; they still count towards k.

Only the standard library and NumPy are used.
"""
from __future__ import annotations

import math

import numpy as np

MAX_SIDE = 30
MIN_SIDE = 3
MAX_INPUT_SIDE = 15
MIN_K = 2
MAX_K = 7
MIN_BG_FRACTION = 0.4

SHIFT_MODES = ("extrapolate", "practiced", "smaller", "mixed")
SHIFT_WEIGHTS = (0.35, 0.25, 0.15, 0.25)
SHAPE_STYLES = ("fixed", "square", "free")
SHAPE_WEIGHTS = (0.2, 0.25, 0.55)
COLOR_STYLES = ("fixed_bg", "varying_bg", "stable_pool")
COLOR_WEIGHTS = (0.35, 0.4, 0.25)
PLACEMENTS = ("spread", "long", "short", "close", "edge")
PLACEMENT_WEIGHTS = (0.22, 0.24, 0.14, 0.24, 0.16)
PROFILES = ("random", "parallel", "both")


def _side_cap(k):
    """Largest input side whose k-fold zoom still fits in 30."""
    return min(MAX_INPUT_SIDE, MAX_SIDE // k)


# ---------------------------------------------------------------------------
# The rule (input only)
# ---------------------------------------------------------------------------

def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _diagonal_pairs(grid):
    """Return (background, threaded pairs, singletons) read from the input alone."""
    bg = _background(grid)
    pairs, singles = [], []
    for color in np.unique(grid).tolist():
        if color == bg:
            continue
        cells = sorted((int(r), int(c)) for r, c in np.argwhere(grid == color))
        if len(cells) == 1:
            singles.append((int(color), cells[0]))
        elif len(cells) == 2:
            (r1, c1), (r2, c2) = cells
            gap = r2 - r1
            if gap >= 2 and abs(c2 - c1) == gap:
                pairs.append((int(color), (r1, c1), (r2, c2)))
    return bg, pairs, singles


def solve(input_grid):
    """Zoom by the palette size; thread every same-coloured diagonal pair thinly."""
    grid = np.asarray(input_grid, dtype=int)
    k = int(np.unique(grid).size)
    out = np.repeat(np.repeat(grid, k, axis=0), k, axis=1)
    _, pairs, _ = _diagonal_pairs(grid)
    for color, (r1, c1), (r2, c2) in pairs:
        step = 1 if c2 > c1 else -1
        for j in range(1, r2 - r1):
            top, left = (r1 + j) * k, (c1 + step * j) * k
            for i in range(k):
                out[top + i, left + (i if step > 0 else k - 1 - i)] = color
    return out


# ---------------------------------------------------------------------------
# Independent checks used by the validator
# ---------------------------------------------------------------------------

def _zoom(grid, k):
    height, width = grid.shape
    out = np.zeros((height * k, width * k), dtype=int)
    for r in range(height):
        for c in range(width):
            out[r * k:(r + 1) * k, c * k:(c + 1) * k] = grid[r, c]
    return out


def _reference_output(grid, info):
    """Corner-to-corner fine-resolution tracing (a different construction from solve)."""
    k, bg = info["k"], info["bg"]
    out = _zoom(grid, k)
    for color, (r1, c1), (r2, c2) in info["pairs"]:
        step = 1 if c2 > c1 else -1
        row = r1 * k + k - 1
        col = c1 * k + (k - 1 if step > 0 else 0)
        length = (r2 - r1 - 1) * k
        for t in range(1, length + 1):
            if out[row + t, col + step * t] != bg:
                raise AssertionError("thread would overwrite a non-background block")
            out[row + t, col + step * t] = color
        landing = (row + length + 1, col + step * (length + 1))
        corner = (r2 * k, c2 * k if step > 0 else c2 * k + k - 1)
        if landing != corner:
            raise AssertionError("thread does not end at the partner block's corner")
    return out


def _chained_output(grid, info):
    """Shortcut: complete diagonals at input scale, then zoom (thick staircase)."""
    coarse = grid.copy()
    for color, (r1, c1), (r2, c2) in info["pairs"]:
        step = 1 if c2 > c1 else -1
        for j in range(1, r2 - r1):
            coarse[r1 + j, c1 + step * j] = color
    return _zoom(coarse, info["k"])


def _inspect_input(grid):
    """Check family invariants of one input and return its structural description."""
    if grid.ndim != 2:
        raise AssertionError("input must be 2-D")
    height, width = grid.shape
    if not (MIN_SIDE <= height <= MAX_INPUT_SIDE and MIN_SIDE <= width <= MAX_INPUT_SIDE):
        raise AssertionError("input sides must be 3..15")
    colors, counts = np.unique(grid, return_counts=True)
    k = int(colors.size)
    if not MIN_K <= k <= MAX_K:
        raise AssertionError("palette size must be 2..7")
    if height * k > MAX_SIDE or width * k > MAX_SIDE:
        raise AssertionError("zoomed output would exceed 30")
    order = np.argsort(-counts, kind="stable")
    bg, bg_count = int(colors[order[0]]), int(counts[order[0]])
    if bg_count <= int(counts[order[1]]) or bg_count < 3:
        raise AssertionError("background must be the unique most frequent colour")
    if bg_count < MIN_BG_FRACTION * height * width:
        raise AssertionError("background is too sparse")
    pairs, singles, footprint = [], [], set()
    for color, count in zip(colors.tolist(), counts.tolist()):
        if color == bg:
            continue
        cells = sorted((int(r), int(c)) for r, c in np.argwhere(grid == color))
        if count == 1:
            singles.append((color, cells[0]))
            continue
        if count != 2:
            raise AssertionError("each foreground colour must occur once or twice")
        (r1, c1), (r2, c2) = cells
        gap = r2 - r1
        if gap < 2 or abs(c2 - c1) != gap:
            raise AssertionError("a repeated colour must form a 45-degree pair with gap >= 2")
        step = 1 if c2 > c1 else -1
        path = [(r1 + j, c1 + step * j) for j in range(gap + 1)]
        if any(grid[p] != bg for p in path[1:-1]):
            raise AssertionError("cells strictly between a pair must be background")
        if footprint.intersection(path):
            raise AssertionError("pair footprints must be disjoint")
        footprint.update(path)
        pairs.append((color, (r1, c1), (r2, c2)))
    if not pairs:
        raise AssertionError("every example needs at least one threaded pair")
    for _, cell in singles:
        if cell in footprint:
            raise AssertionError("singleton on a pair footprint")
    directions = sorted({1 if p[2][1] > p[1][1] else -1 for p in pairs})
    return {"k": k, "bg": bg, "pairs": pairs, "singles": singles, "shape": (height, width),
            "m": len(pairs), "s": len(singles), "directions": directions,
            "gaps": [p[2][0] - p[1][0] for p in pairs],
            "nonbg_cells": 2 * len(pairs) + len(singles)}


def _as_grid(value):
    array = np.asarray(value)
    if array.ndim != 2 or not all(1 <= n <= MAX_SIDE for n in array.shape):
        raise AssertionError("grid dimensions must be 1..30")
    if not np.issubdtype(array.dtype, np.integer):
        raise AssertionError("grid cells must be integers")
    if np.any(array < 0) or np.any(array > 9):
        raise AssertionError("colours must be 0..9")
    return array.astype(int)


def validate_generated_puzzle(puzzle):
    """Raise AssertionError when a puzzle violates the family's invariants."""
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle must have exactly train/test")
    train, test = puzzle["train"], puzzle["test"]
    if not (isinstance(train, list) and isinstance(test, list)):
        raise AssertionError("splits must be lists")
    if not (2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)):
        raise AssertionError("illegal pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise AssertionError("pairs must have exactly input/output")
            grid, out = _as_grid(pair["input"]), _as_grid(pair["output"])
            key = (grid.shape, tuple(grid.flatten().tolist()))
            if key in seen:
                raise AssertionError("duplicate input")
            seen.add(key)
            info = _inspect_input(grid)
            k = info["k"]
            if out.shape != (grid.shape[0] * k, grid.shape[1] * k):
                raise AssertionError("output is not the k-fold zoom size")
            expected = _reference_output(grid, info)
            if not np.array_equal(out, expected):
                raise AssertionError("output differs from the thin-thread zoom")
            if np.array_equal(out, _zoom(grid, k)):
                raise AssertionError("output equals a plain zoom (pairing irrelevant)")
            if np.array_equal(out, _chained_output(grid, info)):
                raise AssertionError("output equals the thick staircase shortcut")
            if not np.array_equal(np.asarray(solve(grid)), out):
                raise AssertionError("solve disagrees with the reference")
            infos[split].append(info)
    tr = infos["train"]
    if len({i["k"] for i in tr}) < 2:
        raise AssertionError("training must show at least two zoom factors")
    if len({(i["shape"][0] * i["k"], i["shape"][1] * i["k"]) for i in tr}) < 2:
        raise AssertionError("training outputs must not all share one size")
    if not any(i["s"] >= 1 for i in tr):
        raise AssertionError("training must show an unthreaded singleton colour")
    if not any(i["m"] >= 2 for i in tr):
        raise AssertionError("training must show several pairs in one grid")
    if {d for i in tr for d in i["directions"]} != {-1, 1}:
        raise AssertionError("training must show both diagonal directions")
    shortcuts = {
        "k = height": lambda i: i["shape"][0],
        "k = width": lambda i: i["shape"][1],
        "k = pairs + 1": lambda i: i["m"] + 1,
        "k = foreground cells": lambda i: i["nonbg_cells"],
    }
    for name, guess in shortcuts.items():
        if all(guess(i) == i["k"] for i in tr):
            raise AssertionError(f"training does not refute shortcut {name}")
    return True


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not (2 <= num_train <= 10):
        raise ValueError("num_train must be in 2..10")
    if not (1 <= num_test <= 4):
        raise ValueError("num_test must be in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _pick(rng, options, weights=None):
    if weights is None:
        return options[int(rng.integers(len(options)))]
    return options[int(rng.choice(len(options), p=np.asarray(weights) / sum(weights)))]


def _plan_ks(rng, num_train, num_test):
    mode = _pick(rng, SHIFT_MODES, SHIFT_WEIGHTS)
    if mode == "extrapolate":
        top = int(rng.integers(4, MAX_K + 1))
        pool = list(range(MIN_K, top))
    elif mode == "smaller":
        low = int(rng.integers(2, 4))
        pool = list(range(low + 1, 7))
    elif mode == "practiced":
        lo = int(rng.integers(2, 5))
        hi = int(rng.integers(lo + 1, 7))
        pool = list(range(lo, hi + 1))
    else:
        pool = list(range(2, 7))
    contrast = None
    if rng.random() < 0.65:
        contrast = (_pick(rng, ("add", "remove")), int(rng.integers(0, num_train - 1)))
    for _ in range(200):
        ks = [int(rng.choice(pool)) for _ in range(num_train)]
        if contrast is not None:
            kind, pos = contrast
            if kind == "add":
                base = int(rng.choice(pool[:-1]))
                ks[pos], ks[pos + 1] = base, base + 1
            else:
                base = int(rng.choice(pool[1:]))
                ks[pos], ks[pos + 1] = base, base - 1
        if len(set(ks)) >= 2 and max(ks) >= 3:
            break
    else:  # pragma: no cover - pools always have two values
        raise RuntimeError("could not plan training factors")
    if mode == "extrapolate":
        tests = [top] + [int(rng.integers(2, MAX_K + 1)) for _ in range(num_test - 1)]
    elif mode == "smaller":
        tests = [low] + [int(rng.integers(2, MAX_K + 1)) for _ in range(num_test - 1)]
    elif mode == "practiced":
        seen = sorted(set(ks))
        tests = [int(rng.choice(seen)) for _ in range(num_test)]
    else:
        values = [int(v) for v in rng.permutation(np.arange(2, MAX_K + 1))]
        tests = values[:num_test]
    return mode, ks, tests, contrast


def _directions(rng, m, profile):
    if profile == "both" and m >= 2:
        dirs = [1, -1] + [int(rng.choice((1, -1))) for _ in range(m - 2)]
        return [int(d) for d in rng.permutation(dirs)]
    if profile == "parallel":
        return [int(rng.choice((1, -1)))] * m
    return [int(rng.choice((1, -1))) for _ in range(m)]


def _near_mask(occupied):
    height, width = occupied.shape
    near = np.zeros_like(occupied)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            shifted = np.zeros_like(occupied)
            rs = slice(max(dr, 0), height + min(dr, 0))
            rd = slice(max(-dr, 0), height + min(-dr, 0))
            cs = slice(max(dc, 0), width + min(dc, 0))
            cd = slice(max(-dc, 0), width + min(-dc, 0))
            shifted[rd, cd] = occupied[rs, cs]
            near |= shifted
    return near & ~occupied


def _place(rng, height, width, m, s, profile, placement):
    occupied = np.zeros((height, width), dtype=bool)
    pairs = []
    max_gap = min(height, width) - 1
    if max_gap < 2:
        return None
    for step in _directions(rng, m, profile):
        near = _near_mask(occupied)
        by_gap = {}
        for gap in range(2, max_gap + 1):
            found = []
            cols = range(width - gap) if step > 0 else range(gap, width)
            for r1 in range(height - gap):
                for c1 in cols:
                    cells = [(r1 + j, c1 + step * j) for j in range(gap + 1)]
                    if any(occupied[p] for p in cells):
                        continue
                    found.append(cells)
            if found:
                by_gap[gap] = found
        if not by_gap:
            return None
        gaps = sorted(by_gap)
        if placement == "long":
            gap = gaps[-1] if rng.random() < 0.6 else gaps[int(rng.integers(max(0, len(gaps) - 3), len(gaps)))]
        elif placement == "short":
            short = [g for g in gaps if g <= 3] or gaps[:1]
            gap = int(rng.choice(short))
        else:
            weights = np.asarray(gaps, dtype=float) ** 1.2
            gap = int(gaps[int(rng.choice(len(gaps), p=weights / weights.sum()))])
        options = by_gap[gap]
        if placement == "close" and occupied.any():
            preferred = [o for o in options if any(near[p] for p in o)]
        elif placement == "spread" and occupied.any():
            preferred = [o for o in options if not any(near[p] for p in o)]
        elif placement == "edge":
            preferred = [o for o in options
                         if any(p[0] in (0, height - 1) or p[1] in (0, width - 1) for p in (o[0], o[-1]))]
        else:
            preferred = []
        options = preferred or options
        cells = options[int(rng.integers(len(options)))]
        for p in cells:
            occupied[p] = True
        pairs.append((cells[0], cells[-1]))
    free = [tuple(int(v) for v in p) for p in np.argwhere(~occupied)]
    if len(free) < s:
        return None
    if placement == "close":
        near = _near_mask(occupied)
        ranked = [p for p in free if near[p]]
        rest = [p for p in free if not near[p]]
        pool = ranked if len(ranked) >= s and rng.random() < 0.7 else free
        del rest
    elif placement == "edge":
        border = [p for p in free if p[0] in (0, height - 1) or p[1] in (0, width - 1)]
        pool = border if len(border) >= s else free
    else:
        pool = free
    chosen = rng.choice(len(pool), size=s, replace=False) if s else []
    singles = [pool[int(i)] for i in chosen]
    return pairs, singles


def _shape(rng, style, cap, fixed_shape, m, s):
    if style == "fixed":
        return fixed_shape
    low = MIN_SIDE if m == 1 else min(4, cap)

    def side(lower):
        if rng.random() < 0.5:
            lower = max(lower, (cap + 1) // 2)
        return int(rng.integers(lower, cap + 1))

    if style == "square":
        edge = side(low)
        return edge, edge
    if rng.random() < 0.25 and cap >= 6:
        short = int(rng.integers(MIN_SIDE if m == 1 else 4, max(MIN_SIDE + 1, cap // 2) + 1))
        short = min(short, cap)
        return (short, cap) if rng.random() < 0.5 else (cap, short)
    return side(MIN_SIDE), side(MIN_SIDE)


class _Palette:
    def __init__(self, rng, style, max_fg):
        self.rng, self.style = rng, style
        self.bg = 0 if rng.random() < 0.45 else int(rng.integers(10))
        others = [c for c in range(10) if c != self.bg]
        self.pool = [int(c) for c in rng.permutation(others)[:max(max_fg, 1)]]

    def draw(self, n_fg, fresh_fg=False):
        rng = self.rng
        if self.style == "varying_bg":
            bg = int(rng.integers(10))
        else:
            bg = self.bg
        others = [c for c in range(10) if c != bg]
        if self.style == "stable_pool" and not fresh_fg:
            source = [c for c in self.pool if c != bg]
            if len(source) >= n_fg:
                return bg, [int(c) for c in rng.permutation(source)[:n_fg]]
        return bg, [int(c) for c in rng.permutation(others)[:n_fg]]


def _render_input(height, width, bg, fg, pairs, singles):
    grid = np.full((height, width), bg, dtype=int)
    for idx, (a, b) in enumerate(pairs):
        grid[a] = grid[b] = fg[idx]
    for idx, p in enumerate(singles):
        grid[p] = fg[len(pairs) + idx]
    return grid


def _sample_m(rng, k, need_multi, need_single):
    hi = k - 1 - (1 if need_single else 0)
    lo = 2 if need_multi else 1
    if hi < lo:
        return None
    values = list(range(lo, hi + 1))
    weights = [1.0 + (0.6 if 2 <= v <= 3 else 0.0) for v in values]
    return int(values[int(rng.choice(len(values), p=np.asarray(weights) / sum(weights)))])


def _make_example(rng, k, spec, shape_style, fixed_shape, palette, seen, extra_cap_k=None):
    cap = _side_cap(max(k, extra_cap_k or k))
    for _ in range(40):
        m = _sample_m(rng, k, spec.get("multi", False), spec.get("single", False))
        if m is None:
            return None
        s = k - 1 - m
        height, width = _shape(rng, shape_style, cap, fixed_shape, m, s)
        height, width = min(height, cap), min(width, cap)
        cells = height * width
        spare = 1 if extra_cap_k and extra_cap_k > k else 0
        if 2 * m + s + spare > math.floor((1 - MIN_BG_FRACTION) * cells) or cells - (2 * m + s + spare) < 3:
            continue
        profile = spec.get("profile") or _pick(rng, PROFILES, (0.45, 0.25, 0.30))
        if spec.get("multi"):
            profile = "both"
        placement = spec.get("placement") or _pick(rng, PLACEMENTS, PLACEMENT_WEIGHTS)
        layout = _place(rng, height, width, m, s, profile, placement)
        if layout is None:
            continue
        pairs, singles = layout
        bg, fg = palette.draw(m + s, fresh_fg=spec.get("fresh", False))
        grid = _render_input(height, width, bg, fg, pairs, singles)
        key = (grid.shape, tuple(grid.flatten().tolist()))
        if key in seen:
            continue
        try:
            _inspect_input(grid)
        except AssertionError:
            continue
        return grid
    return None


def _contrast(rng, base, kind, palette, seen):
    """Recolour one cell so the palette size changes by one while the layout stays."""
    info = _inspect_input(base)
    grid = base.copy()
    if kind == "remove":
        if not info["singles"]:
            return None
        color, cell = info["singles"][int(rng.integers(len(info["singles"])))]
        grid[cell] = info["bg"]
    else:
        footprint = set()
        for _, (r1, c1), (r2, c2) in info["pairs"]:
            step = 1 if c2 > c1 else -1
            footprint.update((r1 + j, c1 + step * j) for j in range(r2 - r1 + 1))
        free = [tuple(int(v) for v in p) for p in np.argwhere(base == info["bg"]) if tuple(int(v) for v in p) not in footprint]
        if not free:
            return None
        used = set(np.unique(base).tolist())
        preferred = [c for c in palette.pool if c not in used]
        candidates = preferred or [c for c in range(10) if c not in used]
        if not candidates:
            return None
        grid[free[int(rng.integers(len(free)))]] = int(rng.choice(candidates))
    key = (grid.shape, tuple(grid.flatten().tolist()))
    if key in seen:
        return None
    try:
        _inspect_input(grid)
    except AssertionError:
        return None
    return grid


def _attempt(rng, num_train, num_test):
    mode, train_ks, test_ks, contrast = _plan_ks(rng, num_train, num_test)
    shape_style = _pick(rng, SHAPE_STYLES, SHAPE_WEIGHTS)
    color_style = _pick(rng, COLOR_STYLES, COLOR_WEIGHTS)
    all_ks = train_ks + test_ks
    fixed_shape = None
    if shape_style == "fixed":
        cap = _side_cap(max(all_ks))
        fixed_shape = (int(rng.integers(4, cap + 1)), int(rng.integers(4, cap + 1)))
    palette = _Palette(rng, color_style, max(all_ks) - 1)

    specs = [dict() for _ in range(num_train)]
    derived = None
    if contrast is not None:
        kind, pos = contrast
        derived = pos + 1
        if kind == "remove":
            specs[pos]["single"] = True
    eligible = [i for i in range(num_train) if train_ks[i] >= 3 and i != derived]
    if not eligible:
        return None
    teacher = int(rng.choice(eligible))
    specs[teacher]["multi"] = True
    if contrast is None:
        single_options = [i for i in eligible if i != teacher or train_ks[i] >= 4]
        if not single_options:
            return None
        specs[int(rng.choice(single_options))]["single"] = True

    seen, train = set(), [None] * num_train
    for i in range(num_train):
        if i == derived:
            continue
        extra = train_ks[i + 1] if contrast is not None and i == contrast[1] else None
        grid = _make_example(rng, train_ks[i], specs[i], shape_style, fixed_shape, palette, seen, extra)
        if grid is None:
            return None
        seen.add((grid.shape, tuple(grid.flatten().tolist())))
        train[i] = grid
    if derived is not None:
        grid = _contrast(rng, train[derived - 1], contrast[0], palette, seen)
        if grid is None:
            return None
        seen.add((grid.shape, tuple(grid.flatten().tolist())))
        train[derived] = grid

    tests = []
    for k in test_ks:
        spec = {}
        roll = rng.random()
        if roll < 0.3:
            spec["placement"] = "close"
            spec["profile"] = "both"
        elif roll < 0.55:
            spec["placement"] = "long"
        if color_style == "stable_pool" and rng.random() < 0.3:
            spec["fresh"] = True
        grid = _make_example(rng, k, spec, shape_style, fixed_shape, palette, seen)
        if grid is None:
            return None
        seen.add((grid.shape, tuple(grid.flatten().tolist())))
        tests.append(grid)

    puzzle = {"train": [{"input": g, "output": solve(g)} for g in train],
              "test": [{"input": g, "output": solve(g)} for g in tests]}
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return None
    meta = {"mode": mode, "shape_style": shape_style, "color_style": color_style,
            "contrast": None if contrast is None else contrast[0],
            "train_ks": train_ks, "test_ks": test_ks}
    return puzzle, meta


def _generate(seed, num_train=4, num_test=1):
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(400):
        result = _attempt(rng, num_train, num_test)
        if result is not None:
            puzzle, meta = result
            out = {split: [{"input": p["input"].tolist(), "output": np.asarray(p["output"]).tolist()}
                           for p in puzzle[split]] for split in ("train", "test")}
            return out, meta
    raise RuntimeError(f"could not generate a puzzle for seed {seed}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    puzzle, _ = _generate(seed, num_train, num_test)
    return puzzle


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

def _expect_rejection(puzzle, label):
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        return
    raise AssertionError(f"validator accepted a corrupted puzzle: {label}")


def self_test():
    # Hand-built regression: k = 3 (background 0, pair 3, singleton 6).
    grid = np.array([[3, 0, 0, 0, 6],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 3, 0]])
    out = solve(grid)
    assert out.shape == (12, 15)
    assert np.all(out[0:3, 0:3] == 3) and np.all(out[9:12, 9:12] == 3)
    assert np.all(out[0:3, 12:15] == 6)
    thread = [(t, t) for t in range(3, 9)]
    for r, c in thread:
        assert out[r, c] == 3
    assert int(np.sum(out == 3)) == 18 + len(thread)
    # Anti-diagonal regression with even k = 4 (pairs 2 and 5, singleton 7).
    grid = np.array([[0, 0, 0, 2, 0],
                     [0, 7, 0, 0, 0],
                     [0, 2, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [5, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 5, 0, 0]])
    out = solve(grid)
    assert out.shape == (28, 20)
    # pair 2: (0,3) -> (2,1); intermediate block (1,2) gets its anti-diagonal.
    for i in range(4):
        assert out[4 + i, 8 + 3 - i] == 2
    # pair 5: (4,0) -> (6,2); intermediate block (5,1) gets its main diagonal.
    for i in range(4):
        assert out[20 + i, 4 + i] == 5
    assert np.all(out[4:8, 4:8] == 7)

    # Generated puzzles: validity, determinism and variation over 42 seeds.
    modes, shapes, test_ks, bgs, styles, contrasts = set(), set(), set(), set(), set(), set()
    max_gap, threads, both_tests, singleton_tests = 0, 0, 0, 0
    for seed in range(42):
        puzzle, meta = _generate(seed, 4, 1)
        validate_generated_puzzle(puzzle)
        modes.add(meta["mode"])
        styles.add((meta["shape_style"], meta["color_style"]))
        contrasts.add(meta["contrast"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_input(np.asarray(pair["input"]))
                shapes.add(info["shape"])
                bgs.add(info["bg"])
                max_gap = max(max_gap, max(info["gaps"]))
                threads += info["m"]
                if split == "test":
                    test_ks.add(info["k"])
                    both_tests += info["directions"] == [-1, 1]
                    singleton_tests += info["s"] >= 1
    assert len(modes) == len(SHIFT_MODES), modes
    assert len(test_ks) >= 5, test_ks
    assert len(shapes) >= 25, len(shapes)
    assert len(bgs) >= 6, bgs
    assert len(styles) >= 5, styles
    assert contrasts >= {None, "add", "remove"}, contrasts
    assert max_gap >= 8 and both_tests >= 5 and singleton_tests >= 10

    # Seed reproducibility and count handling.
    assert generate_puzzle(7, 5, 2) == generate_puzzle(7, 5, 2)
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {bad}")

    # Corruptions that the validator must reject.
    base, _ = _generate(3, 4, 1)
    pair = base["train"][0]
    grid = np.asarray(pair["input"])
    info = _inspect_input(grid)
    corrupted = {"train": [dict(p) for p in base["train"]], "test": base["test"]}
    corrupted["train"][0] = {"input": pair["input"], "output": _chained_output(grid, info).tolist()}
    _expect_rejection(corrupted, "thick staircase")
    corrupted["train"][0] = {"input": pair["input"], "output": _zoom(grid, info["k"]).tolist()}
    _expect_rejection(corrupted, "plain zoom")
    corrupted["train"][0] = {"input": pair["input"], "output": _zoom(grid, info["k"] - 1).tolist()}
    _expect_rejection(corrupted, "foreground-only factor")
    broken = np.asarray(pair["output"]).copy()
    _, (r1, c1), (r2, c2) = info["pairs"][0]
    step = 1 if c2 > c1 else -1
    k = info["k"]
    broken[r1 * k + k, c1 * k + (k if step > 0 else -1)] = info["bg"]
    corrupted["train"][0] = {"input": pair["input"], "output": broken.tolist()}
    _expect_rejection(corrupted, "missing thread cell")
    corrupted["train"][0] = pair
    corrupted["test"] = [dict(corrupted["train"][1])]
    _expect_rejection(corrupted, "duplicate input")
    return True


if __name__ == "__main__":
    self_test()
    print("self_test passed")
