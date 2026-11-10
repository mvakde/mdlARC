"""Weighted push: weights push same-coloured horizontal bars left by their cell count.

Every input is a uniform majority background holding monochrome objects, segmented
as 8-connected single-colour components:

* bars    - components lying in exactly one row (horizontal runs);
* weights - components spanning two or more rows (any silhouette: blobs, rings,
            glyphs, diagonal chains, lattices, serpentines, vertical lines, ...).

Rule: every weight is erased.  Every bar of colour c moves left by the total
number of weight cells of colour c (zero when c has no weight).  Bar cells pushed
past the left edge are clipped; a bar pushed completely past the edge vanishes.
Grid size and background are unchanged.

Parent A (32e9702f) supplies the horizontal bars, their leftward translation that
keeps row/colour/length, and left-edge clipping.  Parent B (dce56571) supplies the
reading of an irregular, possibly diagonal-linked figure purely as a unary cell
count, and the erasure of that figure.  The connection is a colour key: the count
of a colour's weight mass becomes the displacement of the bars sharing its colour.

Weights never touch any other object.  In some puzzles bars of different colours
touch each other (stacked or end-to-end); segmentation stays per colour, so such
stacks shear apart when their colours carry different pushes.
"""

from __future__ import annotations

import numpy as np

_N8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAG = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_THEMES = ("heavier", "more_groups", "bigger", "new_colors", "compact", "longer")
_EXCLUSIVE = (frozenset(("bigger", "compact")), frozenset(("heavier", "compact")),
              frozenset(("more_groups", "compact")))
_FEATURES = ("static", "orphan", "diagonal", "contact", "split")

# Shortcuts that every puzzle's training pairs must refute.
_ALWAYS_REFUTED = ("unit_shift", "bbox_width", "bbox_height", "bbox_area", "stop_at_wall",
                   "uniform_total", "color_total", "nearest_weight", "extend_left",
                   "bars_stay")

_BLOB_STYLES = ("random", "compact", "branch", "wide", "tall")
# (style, weight, needs_sparse_option)
_FULL_STYLES = (("blob", 0.30, False), ("ring", 0.09, False), ("glyph", 0.10, False),
                ("serpent", 0.08, False), ("flat", 0.07, False), ("vline", 0.05, False),
                ("rect", 0.04, False), ("chain", 0.10, True), ("lattice", 0.07, True),
                ("xcross", 0.06, True), ("pendant", 0.06, True))
_SPARSE_STYLES = ("chain", "lattice", "xcross", "pendant")


# --------------------------------------------------------------------------
# Input-only analysis and solver
# --------------------------------------------------------------------------

def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(grid: np.ndarray, background: int, neighbours=_N8, colorblind=False) -> list:
    """Connected components of non-background cells (per colour unless colorblind)."""
    rows = np.asarray(grid).tolist()
    height, width = len(rows), len(rows[0])
    seen = [[False] * width for _ in range(height)]
    found = []
    for r in range(height):
        for c in range(width):
            if seen[r][c] or rows[r][c] == background:
                continue
            color = rows[r][c]
            seen[r][c] = True
            stack, cells = [(r, c)], []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in neighbours:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and not seen[ny][nx]:
                        value = rows[ny][nx]
                        if (value != background) if colorblind else (value == color):
                            seen[ny][nx] = True
                            stack.append((ny, nx))
            found.append((int(color), sorted(cells)))
    return found


def _parse(grid: np.ndarray, neighbours=_N8):
    """Split a grid into background, one-row bars and multi-row weights."""
    background = _background(grid)
    bars, weights = [], []
    for color, cells in _components(grid, background, neighbours):
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        if len(rows) == 1:
            bars.append({"color": color, "row": rows[0], "start": cols[0], "end": cols[-1] + 1})
        else:
            weights.append({"color": color, "cells": cells,
                            "height": rows[-1] - rows[0] + 1, "width": cols[-1] - cols[0] + 1})
    return background, bars, weights


def _push_amounts(weights) -> dict:
    amounts = {}
    for weight in weights:
        amounts[weight["color"]] = amounts.get(weight["color"], 0) + len(weight["cells"])
    return amounts


def _paint(shape, background, spans) -> np.ndarray:
    """spans: (row, lo, hi, colour) in unclipped column coordinates."""
    result = np.full(shape, background, dtype=int)
    for row, lo, hi, color in spans:
        lo, hi = max(0, lo), min(shape[1], hi)
        if hi > lo:
            result[row, lo:hi] = color
    return result


def solve(input_grid) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    background, bars, weights = _parse(grid)
    amounts = _push_amounts(weights)
    spans = []
    for bar in bars:
        shift = amounts.get(bar["color"], 0)
        spans.append((bar["row"], bar["start"] - shift, bar["end"] - shift, bar["color"]))
    return _paint(grid.shape, background, spans)


# --------------------------------------------------------------------------
# Plausible alternative rules (used to prove the demonstrations discriminate)
# --------------------------------------------------------------------------

def _chebyshev(bar, cells) -> int:
    return min(max(abs(bar["row"] - r), max(bar["start"] - c, c - (bar["end"] - 1), 0))
               for r, c in cells)


def _shortcut_outputs(input_grid) -> dict:
    grid = np.asarray(input_grid, dtype=int)
    shape = grid.shape
    background, bars, weights = _parse(grid)
    amounts = _push_amounts(weights)
    widths, heights, boxes, largest = {}, {}, {}, {}
    for weight in weights:
        color = weight["color"]
        widths[color] = widths.get(color, 0) + weight["width"]
        heights[color] = heights.get(color, 0) + weight["height"]
        boxes[color] = boxes.get(color, 0) + weight["width"] * weight["height"]
        largest[color] = max(largest.get(color, 0), len(weight["cells"]))
    total = sum(amounts.values())

    def shifted(shift_of):
        return _paint(shape, background, [
            (b["row"], b["start"] - shift_of(b), b["end"] - shift_of(b), b["color"]) for b in bars])

    def nearest_area(bar):
        best, area = None, 0
        for weight in weights:
            distance = _chebyshev(bar, weight["cells"])
            if best is None or distance < best:
                best, area = distance, len(weight["cells"])
        return area

    def nearest_piece(bar):
        best, area = None, 0
        for weight in weights:
            if weight["color"] != bar["color"]:
                continue
            distance = _chebyshev(bar, weight["cells"])
            if best is None or distance < best:
                best, area = distance, len(weight["cells"])
        return area

    def stop_at_wall():
        spans = []
        for b in bars:
            shift = min(amounts.get(b["color"], 0), b["start"])
            spans.append((b["row"], b["start"] - shift, b["end"] - shift, b["color"]))
        return _paint(shape, background, spans)

    def color_total(bar):
        others = sum(o["end"] - o["start"] for o in bars if o["color"] == bar["color"] and o is not bar)
        return amounts.get(bar["color"], 0) + others

    four_background, four_bars, four_weights = _parse(grid, _N4)
    four_amounts = _push_amounts(four_weights)
    four_conn = _paint(shape, four_background, [
        (b["row"], b["start"] - four_amounts.get(b["color"], 0),
         b["end"] - four_amounts.get(b["color"], 0), b["color"]) for b in four_bars])

    # Colour-blind clusters: touching bars of different colours read as one object.
    cluster_of, cluster_rows = {}, {}
    for index, (_, cells) in enumerate(_components(grid, background, _N8, colorblind=True)):
        cluster_rows[index] = len({r for r, _ in cells})
        for cell in cells:
            cluster_of[cell] = index
    bar_cluster = [cluster_of[(b["row"], b["start"])] for b in bars]
    cluster_push = {}
    for b, index in zip(bars, bar_cluster):
        cluster_push[index] = max(cluster_push.get(index, 0), amounts.get(b["color"], 0))
    colorblind = _paint(shape, background, [
        (b["row"], b["start"] - amounts.get(b["color"], 0), b["end"] - amounts.get(b["color"], 0),
         b["color"]) for b, index in zip(bars, bar_cluster) if cluster_rows[index] == 1])
    rigid = _paint(shape, background, [
        (b["row"], b["start"] - cluster_push[index], b["end"] - cluster_push[index], b["color"])
        for b, index in zip(bars, bar_cluster)])

    # Weights as obstacles: a bar stops against the first weight cell on its path.
    weight_cols = {}
    for weight in weights:
        for r, c in weight["cells"]:
            weight_cols.setdefault(r, []).append(c)
    block_spans = []
    for b in bars:
        shift = amounts.get(b["color"], 0)
        lo, hi = b["start"] - shift, b["end"] - shift
        if shift:
            blockers = [c for c in weight_cols.get(b["row"], ()) if c < b["start"]]
            if blockers and max(blockers) >= b["start"] - shift:
                lo = max(blockers) + 1
                hi = lo + b["end"] - b["start"]
        block_spans.append((b["row"], lo, hi, b["color"]))

    return {
        # Parent A taken literally: every bar moves one column; counts ignored.
        "unit_shift": shifted(lambda b: 1),
        # Geometry instead of quantity (parent B's count omitted).
        "bbox_width": shifted(lambda b: widths.get(b["color"], 0)),
        "bbox_height": shifted(lambda b: heights.get(b["color"], 0)),
        "bbox_area": shifted(lambda b: boxes.get(b["color"], 0)),
        # Clipping omitted: bars slide but stop against the wall at full length.
        "stop_at_wall": stop_at_wall(),
        # Colour pairing omitted: every bar moves by the pair's total weight.
        "uniform_total": shifted(lambda b: total),
        # Counting every other cell of the colour instead of only the weights.
        "color_total": shifted(color_total),
        # Spatial pairing instead of colour pairing.
        "nearest_weight": shifted(nearest_area),
        # Count used as a length change instead of a translation.
        "extend_left": _paint(shape, background, [
            (b["row"], b["start"] - amounts.get(b["color"], 0), b["end"], b["color"]) for b in bars]),
        # Translation omitted: weights vanish, bars stay.
        "bars_stay": shifted(lambda b: 0),
        # Parent A's default motion for bars that have no weight.
        "static_unit": shifted(lambda b: amounts.get(b["color"], 1)),
        # Four-connected segmentation (diagonal weight cells read as bars).
        "four_conn": four_conn,
        # Split weights: only the largest piece, or the nearest same-colour piece, counts.
        "largest_piece": shifted(lambda b: largest.get(b["color"], 0)),
        "nearest_piece": shifted(nearest_piece),
        # Contact: a multi-row multicolour stack is read as a weight and erased.
        "colorblind": colorblind,
        # Contact: a touching cluster moves rigidly by its largest push.
        "rigid_cluster": rigid,
        # Weights treated as obstacles instead of being erased first.
        "weights_block": _paint(shape, background, block_spans),
    }


# --------------------------------------------------------------------------
# Family analysis used by validation and generation
# --------------------------------------------------------------------------

def _four_pieces(cells) -> int:
    remaining = set(cells)
    pieces = 0
    while remaining:
        pieces += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in _N4:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
    return pieces


def _holes(cells) -> list:
    """Background cells enclosed by a figure (not 4-reachable from outside its box)."""
    cells = set(cells)
    r0 = min(r for r, _ in cells) - 1
    r1 = max(r for r, _ in cells) + 1
    c0 = min(c for _, c in cells) - 1
    c1 = max(c for _, c in cells) + 1
    seen = {(r0, c0)}
    stack = [(r0, c0)]
    while stack:
        r, c = stack.pop()
        for dr, dc in _N4:
            p = (r + dr, c + dc)
            if r0 <= p[0] <= r1 and c0 <= p[1] <= c1 and p not in cells and p not in seen:
                seen.add(p)
                stack.append(p)
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)
            if (r, c) not in cells and (r, c) not in seen]


def _weight_traits(cells) -> dict:
    """Descriptive classification of a returned weight (for coverage measurement)."""
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    h, w, n = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1, len(cells)
    pieces = _four_pieces(cells)
    return {"n": n, "h": h, "w": w, "pieces": pieces, "vline": w == 1,
            "rect": w >= 2 and n == h * w, "sparse": pieces == n,
            "ring": pieces == 1 and bool(_holes(cells))}


def _analyze_pair(pair) -> dict:
    """Check the family's input conventions and return descriptive facts."""
    source = np.asarray(pair["input"], dtype=int)
    target = np.asarray(pair["output"], dtype=int)
    assert source.ndim == 2 and source.shape == target.shape
    height, width = source.shape
    background, bars, weights = _parse(source)
    assert int(np.sum(source == background)) * 2 > source.size, "background must be a strict majority"
    rows = source.tolist()
    for weight in weights:
        color = weight["color"]
        for r, c in weight["cells"]:
            for dr, dc in _N8:
                y, x = r + dr, c + dc
                if 0 <= y < height and 0 <= x < width:
                    assert rows[y][x] in (background, color), "weights never touch other objects"
        for r, c in _holes(weight["cells"]):
            assert rows[r][c] == background, "weight holes stay empty"
    amounts = _push_amounts(weights)
    pieces = {}
    for weight in weights:
        pieces[weight["color"]] = pieces.get(weight["color"], 0) + 1
    bar_colors = {b["color"] for b in bars}
    assert bars, "every pair has bars"
    assert any(b["color"] in amounts for b in bars), "some bar is pushed"
    statuses = []
    for bar in bars:
        shift = amounts.get(bar["color"], 0)
        if shift == 0:
            status = "static"
        elif bar["start"] >= shift:
            status = "intact"
        elif bar["end"] > shift:
            status = "clip"
        else:
            status = "vanish"
        bar["shift"] = shift
        bar["status"] = status
        statuses.append(status)
    assert any(s in ("intact", "clip") for s in statuses), "some pushed bar survives"
    # Bars in one row never overtake or overlap after moving (no collision convention).
    by_row = {}
    for bar in bars:
        by_row.setdefault(bar["row"], []).append(bar)
    for row_bars in by_row.values():
        row_bars.sort(key=lambda b: b["start"])
        for left, right in zip(row_bars, row_bars[1:]):
            assert right["start"] - right["shift"] >= left["end"] - left["shift"], "bars overtake"
    assert np.array_equal(target, solve(source)), "output must follow the rule"
    assert not np.array_equal(source, target)
    groups = {}
    for bar in bars:
        if bar["color"] in amounts:
            groups.setdefault(bar["color"], []).append(bar)
    contact = False
    for bar in bars:
        for c in range(bar["start"], bar["end"]):
            for dr, dc in _N8:
                y, x = bar["row"] + dr, c + dc
                if 0 <= y < height and 0 <= x < width and rows[y][x] not in (background, bar["color"]):
                    contact = True
    weight_cells = {cell for weight in weights for cell in weight["cells"]}
    crossing = any((b["row"], c) in weight_cells for b in bars if b["shift"]
                   for c in range(max(0, b["start"] - b["shift"]), b["start"]))
    return {
        "shape": (height, width),
        "background": background,
        "bars": bars,
        "weights": weights,
        "amounts": amounts,
        "pieces": pieces,
        "groups": groups,
        "statuses": statuses,
        "static": any(s == "static" for s in statuses),
        "orphan": any(c not in bar_colors for c in amounts),
        "diagonal": any(_four_pieces(w["cells"]) > 1 for w in weights),
        "contact": contact,
        "split": any(pieces[c] > 1 for c in groups),
        "crossing": crossing,
        "same_row": any(len(v) > 1 for v in by_row.values()),
        "max_len": max(b["end"] - b["start"] for b in bars),
        "colors": {int(v) for v in np.unique(source)} - {background},
    }


def _refuted(pairs) -> set:
    names = set()
    for pair in pairs:
        target = np.asarray(pair["output"], dtype=int)
        for name, guess in _shortcut_outputs(pair["input"]).items():
            if not np.array_equal(guess, target):
                names.add(name)
    return names


def _train_summary(train_pairs) -> dict:
    problems = []
    facts = [_analyze_pair(p) for p in train_pairs]
    statuses = {s for f in facts for s in f["statuses"]}
    if "clip" not in statuses:
        problems.append("no partially clipped bar in training")
    if "intact" not in statuses:
        problems.append("no intact pushed bar in training")
    if not any(len(set(f["amounts"][c] for c in f["groups"])) >= 2 for f in facts):
        problems.append("no training pair with two different push amounts")
    if not any(len(bars) >= 2 for f in facts for bars in f["groups"].values()):
        problems.append("no pushed colour with several bars in training")
    for f in facts:
        for bars in f["groups"].values():
            if not any(b["status"] in ("intact", "clip") for b in bars):
                problems.append("training push amount not measurable")
    refuted = _refuted(train_pairs)
    for name in _ALWAYS_REFUTED:
        if name not in refuted:
            problems.append(f"training does not refute {name}")
    return {"facts": facts, "refuted": refuted, "problems": problems}


def _test_problems(summary, test_pairs) -> list:
    problems = []
    for name in sorted(_refuted(test_pairs) - summary["refuted"]):
        problems.append(f"test needs {name} refuted in training")
    test_facts = [_analyze_pair(p) for p in test_pairs]
    for key in _FEATURES:
        if any(f[key] for f in test_facts) and not any(f[key] for f in summary["facts"]):
            problems.append(f"test shows {key} without a training demonstration")
    return problems


def _training_problems(train_pairs, test_pairs=()) -> list:
    """Return the list of missing demonstrations (empty when sufficient)."""
    summary = _train_summary(train_pairs)
    problems = list(summary["problems"])
    if test_pairs:
        problems += _test_problems(summary, test_pairs)
    return problems


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = np.asarray(pair[name])
                assert grid.ndim == 2 and 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            key = (np.asarray(pair["input"]).shape, np.asarray(pair["input"], dtype=int).tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
    problems = _training_problems(puzzle["train"], puzzle["test"])
    assert not problems, problems
    test_facts = [_analyze_pair(p) for p in puzzle["test"]]
    assert any(s in ("clip", "vanish") for f in test_facts for s in f["statuses"]), \
        "test must exercise the left edge"


# --------------------------------------------------------------------------
# Weight shapes
# --------------------------------------------------------------------------

def _normalize(cells) -> list:
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return sorted({(r - top, c - left) for r, c in cells})


def _dims(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _ok(shape, n, max_rows, max_cols):
    if shape is None or len(shape) != n:
        return None
    h, w = _dims(shape)
    if h < 2 or h > max_rows or w > max_cols:
        return None
    return shape


def _blob(rng, n, max_rows, max_cols, growth, pendant=False, exact=None):
    """Random 4-connected growth, optionally with diagonal-only pendant cells."""
    extra = (1 if n < 8 else int(rng.integers(1, 3))) if pendant else 0
    core = n - extra
    if core < (3 if pendant else 2):
        return None
    for _ in range(14):
        cells = {(0, 0)}
        failed = False
        while len(cells) < core:
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
            frontier = set()
            for r, c in cells:
                for dr, dc in _N4:
                    p = (r + dr, c + dc)
                    if p in cells:
                        continue
                    if (max(r1, p[0]) - min(r0, p[0]) + 1 > max_rows or
                            max(c1, p[1]) - min(c0, p[1]) + 1 > max_cols):
                        continue
                    frontier.add(p)
            if not frontier:
                failed = True
                break
            frontier = sorted(frontier)
            scores = []
            for p in frontier:
                degree = sum((p[0] + dr, p[1] + dc) in cells for dr, dc in _N4)
                if growth == "compact":
                    scores.append(float(degree * degree))
                elif growth == "branch":
                    scores.append(1.0 / (degree * degree))
                elif growth == "wide":
                    scores.append(4.0 if r0 <= p[0] <= r1 else 0.6)
                elif growth == "tall":
                    scores.append(4.0 if c0 <= p[1] <= c1 else 0.6)
                else:
                    scores.append(1.0)
            probabilities = np.asarray(scores) / float(np.sum(scores))
            cells.add(frontier[int(rng.choice(len(frontier), p=probabilities))])
        if failed:
            continue
        for _ in range(extra):
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            options = set()
            for r, c in cells:
                for dr, dc in _DIAG:
                    p = (r + dr, c + dc)
                    if p in cells or any((p[0] + a, p[1] + b) in cells for a, b in _N4):
                        continue
                    if (max(max(rows), p[0]) - min(min(rows), p[0]) + 1 > max_rows or
                            max(max(cols), p[1]) - min(min(cols), p[1]) + 1 > max_cols):
                        continue
                    options.add(p)
            if not options:
                failed = True
                break
            options = sorted(options)
            cells.add(options[int(rng.integers(len(options)))])
        if failed:
            continue
        shape = _normalize(cells)
        if exact is not None and _dims(shape) != exact:
            continue
        if _dims(shape)[1] == 1:
            continue  # vertical lines are their own (rarer) style
        if _ok(shape, n, max_rows, max_cols):
            return shape
    return None


def _ring(rng, n, max_rows, max_cols):
    options = []
    for h in range(3, max_rows + 1):
        for w in range(3, max_cols + 1):
            perimeter = 2 * (h + w) - 4
            if perimeter <= n <= perimeter + (h - 2) * (w - 2) - 1:
                options.append((h, w))
    if not options:
        return None
    h, w = options[int(rng.integers(len(options)))]
    cells = {(r, c) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)}
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    while len(cells) < n:
        free = [p for p in interior if p not in cells]
        candidates = [p for p in free if any((p[0] + a, p[1] + b) in cells for a, b in _N4)]
        if len(free) <= 1 or not candidates:
            return None
        cells.add(candidates[int(rng.integers(len(candidates)))])
    return _ok(_normalize(cells), n, max_rows, max_cols)


def _glyph(rng, n, max_rows, max_cols):
    """Plus/T/L glyphs with unequal arms, or H/U-like frames of two verticals."""
    if n < 3:
        return None
    if rng.random() < 0.6:
        for _ in range(40):
            weights = rng.random(4) + 0.05
            arms = np.floor(weights / weights.sum() * (n - 1)).astype(int)
            while arms.sum() < n - 1:
                arms[int(rng.integers(4))] += 1
            up, down, left, right = (int(a) for a in arms)
            if up + down < 1 or left + right < 1:
                continue
            if up + down + 1 > max_rows or left + right + 1 > max_cols:
                continue
            cells = {(0, 0)}
            cells |= {(-i, 0) for i in range(1, up + 1)} | {(i, 0) for i in range(1, down + 1)}
            cells |= {(0, -i) for i in range(1, left + 1)} | {(0, i) for i in range(1, right + 1)}
            return _ok(_normalize(cells), n, max_rows, max_cols)
        return None
    for _ in range(40):
        h1 = int(rng.integers(2, max(3, max_rows + 1)))
        h2 = int(rng.integers(2, max(3, max_rows + 1)))
        gap = n - h1 - h2
        if gap < 1 or gap + 2 > max_cols:
            continue
        off = int(rng.integers(-(h2 - 1), h1))
        lo, hi = max(0, off), min(h1, off + h2) - 1
        if hi < lo:
            continue
        cross = int(rng.integers(lo, hi + 1))
        cells = {(r, 0) for r in range(h1)} | {(off + r, gap + 1) for r in range(h2)}
        cells |= {(cross, c) for c in range(1, gap + 1)}
        shape = _ok(_normalize(cells), n, max_rows, max_cols)
        if shape:
            return shape
    return None


def _xcross(rng, n, max_rows, max_cols):
    """Diagonal X / V / diagonal-line figures: every link is diagonal."""
    for _ in range(40):
        weights = rng.random(4) * (rng.random(4) < 0.75) + 1e-3
        arms = np.floor(weights / weights.sum() * (n - 1)).astype(int)
        while arms.sum() < n - 1:
            arms[int(rng.integers(4))] += 1
        ul, ur, dl, dr = (int(a) for a in arms)
        cells = {(0, 0)}
        cells |= {(-i, -i) for i in range(1, ul + 1)} | {(-i, i) for i in range(1, ur + 1)}
        cells |= {(i, -i) for i in range(1, dl + 1)} | {(i, i) for i in range(1, dr + 1)}
        shape = _ok(_normalize(cells), n, max_rows, max_cols)
        if shape:
            return shape
    return None


def _chain(rng, n, max_rows, max_cols):
    """Short runs on consecutive rows linked only diagonally (parent B train 3 style)."""
    for _ in range(30):
        runs, total = [], 0
        while total < n:
            length = min(n - total, int(rng.integers(1, 4)))
            runs.append(length)
            total += length
        if len(runs) < 2:
            runs = [1, n - 1]
        if len(runs) > max_rows:
            continue
        start, end = 0, runs[0]
        cells = [(0, c) for c in range(start, end)]
        for index, length in enumerate(runs[1:], 1):
            if rng.random() < 0.5:
                start, end = end, end + length
            else:
                start, end = start - length, start
            cells += [(index, c) for c in range(start, end)]
        shape = _ok(_normalize(cells), n, max_rows, max_cols)
        if shape:
            return shape
    return None


def _lattice(rng, n, max_rows, max_cols):
    """Checkerboard-like figure grown only through diagonal steps (no 4-adjacency)."""
    for _ in range(20):
        cells = {(0, 0)}
        while len(cells) < n:
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            frontier = []
            for r, c in cells:
                for dr, dc in _DIAG:
                    p = (r + dr, c + dc)
                    if p in cells:
                        continue
                    if (max(max(rows), p[0]) - min(min(rows), p[0]) + 1 > max_rows or
                            max(max(cols), p[1]) - min(min(cols), p[1]) + 1 > max_cols):
                        continue
                    frontier.append(p)
            if not frontier:
                break
            frontier = sorted(set(frontier))
            scores = np.asarray([1.0 + sum((p[0] + a, p[1] + b) in cells for a, b in _DIAG)
                                 for p in frontier])
            cells.add(frontier[int(rng.choice(len(frontier), p=scores / scores.sum()))])
        shape = _ok(_normalize(cells), n, max_rows, max_cols)
        if shape:
            return shape
    return None


def _serpent(rng, n, max_rows, max_cols):
    """Thin 4-connected path with a vertical bias (tall, meandering figures)."""
    if n < 3:
        return None
    for _ in range(30):
        path = [(0, 0)]
        cells = {(0, 0)}
        while len(path) < n:
            r, c = path[-1]
            rows = [y for y, _ in cells]
            cols = [x for _, x in cells]
            moves, scores = [], []
            for (dr, dc), score in (((1, 0), 2.0), ((0, 1), 1.0), ((0, -1), 1.0), ((-1, 0), 0.3)):
                p = (r + dr, c + dc)
                if p in cells:
                    continue
                if any((p[0] + a, p[1] + b) in cells and (p[0] + a, p[1] + b) != (r, c) for a, b in _N4):
                    continue
                if (max(max(rows), p[0]) - min(min(rows), p[0]) + 1 > max_rows or
                        max(max(cols), p[1]) - min(min(cols), p[1]) + 1 > max_cols):
                    continue
                moves.append(p)
                scores.append(score)
            if not moves:
                break
            scores = np.asarray(scores)
            p = moves[int(rng.choice(len(moves), p=scores / scores.sum()))]
            path.append(p)
            cells.add(p)
        shape = _ok(_normalize(cells), n, max_rows, max_cols)
        if shape and _dims(shape)[1] >= 2:
            return shape
    return None


def _flat(rng, n, max_rows, max_cols, sparse):
    """Two-row figure: a long run with a shorter run above or below it."""
    if n < 3:
        return None
    for _ in range(30):
        top = int(rng.integers(1, n))
        bottom = n - top
        if max(top, bottom) > max_cols:
            continue
        if sparse and rng.random() < 0.3:
            offset = top if rng.random() < 0.5 else -bottom
        else:
            offset = int(rng.integers(-(bottom - 1), top))
        if top == bottom and offset == 0:
            continue
        cells = [(0, c) for c in range(top)] + [(1, offset + c) for c in range(bottom)]
        shape = _ok(_normalize(cells), n, max_rows, max_cols)
        if shape:
            return shape
    return None


def _rect(rng, n, max_rows, max_cols):
    options = [(h, n // h) for h in range(2, max_rows + 1)
               if n % h == 0 and 2 <= n // h <= max_cols]
    if not options:
        return None
    h, w = options[int(rng.integers(len(options)))]
    return [(r, c) for r in range(h) for c in range(w)]


def _weight_shape(rng, n, max_rows, max_cols, vocab, sparse, require_sparse=False):
    """Return (shape, style) for a weight of exactly n cells, or (None, None)."""
    if n == 2:
        if (require_sparse or (sparse and rng.random() < 0.5)):
            return [(0, 0), (1, 1)] if rng.random() < 0.5 else [(0, 1), (1, 0)], "domino"
        return ([(0, 0), (1, 0)], "domino") if max_rows >= 2 and not require_sparse else (None, None)
    if require_sparse:
        styles = [s for s in _SPARSE_STYLES]
        styles = [styles[int(i)] for i in rng.permutation(len(styles))]
    elif vocab == "blob":
        styles = ["blob"] + (["pendant"] if sparse and rng.random() < 0.45 else [])
        styles = [styles[int(i)] for i in rng.permutation(len(styles))]
    else:
        pool = [(s, p) for s, p, needs_sparse in _FULL_STYLES if sparse or not needs_sparse]
        names = [s for s, _ in pool]
        probs = np.asarray([p for _, p in pool])
        styles = [names[int(i)] for i in rng.choice(len(names), size=len(names), replace=False,
                                                     p=probs / probs.sum())]
    if not require_sparse and "blob" not in styles:
        styles.append("blob")
    for style in styles:
        if style == "blob":
            shape = _blob(rng, n, max_rows, max_cols, str(rng.choice(_BLOB_STYLES)))
        elif style == "pendant":
            shape = _blob(rng, n, max_rows, max_cols, str(rng.choice(_BLOB_STYLES)), pendant=True)
        elif style == "ring":
            shape = _ring(rng, n, max_rows, max_cols)
        elif style == "glyph":
            shape = _glyph(rng, n, max_rows, max_cols)
        elif style == "serpent":
            shape = _serpent(rng, n, max_rows, max_cols)
        elif style == "flat":
            shape = _flat(rng, n, max_rows, max_cols, sparse)
        elif style == "vline":
            shape = [(r, 0) for r in range(n)] if n <= max_rows else None
        elif style == "rect":
            shape = _rect(rng, n, max_rows, max_cols)
        elif style == "chain":
            shape = _chain(rng, n, max_rows, max_cols) if n <= 3 * max_rows else None
        elif style == "lattice":
            shape = _lattice(rng, n, max_rows, max_cols) if n <= 13 else None
        else:
            shape = _xcross(rng, n, max_rows, max_cols)
        if shape is not None:
            if not sparse and _four_pieces(shape) > 1:
                continue
            if require_sparse and _four_pieces(shape) < 2:
                continue
            return shape, style
    return None, None


# --------------------------------------------------------------------------
# Canvas bookkeeping and placement
# --------------------------------------------------------------------------

def _dilate(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for dr in (0, 1, 2):
        for dc in (0, 1, 2):
            out |= padded[dr:dr + height, dc:dc + width]
    return out


class _Canvas:
    def __init__(self, height, width, contact):
        self.shape = (height, width)
        self.contact = contact
        self.occ = np.zeros((height, width), dtype=bool)
        self.weight = np.zeros((height, width), dtype=bool)
        self.hole = np.zeros((height, width), dtype=bool)
        self.barkey = np.full((height, width), -1, dtype=int)
        self.rows = {}

    def place_weight(self, rng, shape):
        height, width = self.shape
        h, w = _dims(shape)
        if h > height or w > width:
            return None
        blocked = _dilate(self.occ) | self.hole
        valid = np.ones((height - h + 1, width - w + 1), dtype=bool)
        for r, c in shape:
            valid &= ~blocked[r:r + height - h + 1, c:c + width - w + 1]
        holes = _holes(shape)
        for r, c in holes:
            valid &= ~self.occ[r:r + height - h + 1, c:c + width - w + 1]
        tops, lefts = np.nonzero(valid)
        if len(tops) == 0:
            return None
        index = int(rng.integers(len(tops)))
        top, left = int(tops[index]), int(lefts[index])
        cells = [(top + r, left + c) for r, c in shape]
        for r, c in cells:
            self.occ[r, c] = True
            self.weight[r, c] = True
        for r, c in holes:
            self.hole[top + r, left + c] = True
        return cells

    def row_ok(self, row, start, end, shift, gap):
        for other in self.rows.get(row, ()):
            if other["start"] < start:
                left, right = (other["start"], other["end"], other["shift"]), (start, end, shift)
            else:
                left, right = (start, end, shift), (other["start"], other["end"], other["shift"])
            if right[0] - right[2] < left[1] - left[2] + gap:
                return False
        return True

    def place_bar(self, rng, key, shift, kind, min_len=1, len_cap=30, want_contact=False,
                  prefer_rows=False):
        height, width = self.shape
        if kind == "intact":
            low, high = min_len, min(width - shift, len_cap)
        elif kind == "clip":
            low, high = max(2, min_len), min(width, len_cap)
        elif kind == "vanish":
            low, high = min_len, min(shift, len_cap)
        elif kind == "static":
            low, high = min_len, min(width - 2, len_cap)
        else:
            low, high = min_len, min(width, len_cap)
        forbidden = self.occ | self.hole | _dilate(self.weight)
        if self.contact:
            forbidden |= _dilate(self.barkey == key)
        else:
            forbidden |= _dilate(self.occ)
        csum = np.zeros((height, width + 1), dtype=int)
        csum[:, 1:] = np.cumsum(forbidden, axis=1)
        tsum = None
        if want_contact:
            touch = _dilate((self.barkey >= 0) & (self.barkey != key))
            tsum = np.zeros((height, width + 1), dtype=int)
            tsum[:, 1:] = np.cumsum(touch, axis=1)
        gap = 0 if self.contact else 1
        for length in _length_options(rng, low, high, kind, width):
            if kind == "intact":
                lo, hi = shift, width - length
            elif kind == "clip":
                lo, hi = max(0, shift - length + 1), min(shift - 1, width - length)
            elif kind == "vanish":
                lo, hi = 0, shift - length
            else:
                lo, hi = 0, width - length
            if hi < lo:
                continue
            starts = np.arange(lo, hi + 1)
            ok = (csum[:, starts + length] - csum[:, starts]) == 0
            rows, index = np.nonzero(ok)
            if len(rows) == 0:
                continue
            candidates = list(zip(rows.tolist(), starts[index].tolist()))
            if tsum is not None:
                touching = [(r, s) for r, s in candidates if tsum[r, s + length] - tsum[r, s] > 0]
                if touching:
                    candidates = touching
            if prefer_rows and self.rows:
                shared = [(r, s) for r, s in candidates if r in self.rows]
                if shared:
                    candidates = shared
            for i in rng.permutation(len(candidates))[:120]:
                row, start = candidates[int(i)]
                if self.row_ok(row, start, start + length, shift, gap):
                    bar = {"row": row, "start": start, "end": start + length, "shift": shift, "key": key}
                    self.rows.setdefault(row, []).append(bar)
                    self.occ[row, start:start + length] = True
                    self.barkey[row, start:start + length] = key
                    return bar
        return None


def _length_options(rng, low, high, kind, width) -> list:
    if high < low:
        return []
    roll = rng.random()
    if kind == "clip" and high == width and roll < 0.05:
        preferred = width
    elif roll < 0.14:
        preferred = low
    elif roll < 0.62:
        preferred = int(rng.integers(low, min(high, 4) + 1)) if low <= min(high, 4) else low
    elif roll < 0.86:
        preferred = int(rng.integers(low, min(high, 9) + 1)) if low <= min(high, 9) else low
    else:
        preferred = int(rng.integers(low, high + 1))
    rest = [x for x in range(low, high + 1) if x != preferred]
    rest = [int(rest[int(i)]) for i in rng.permutation(len(rest))[:5]]
    return [preferred] + rest


# --------------------------------------------------------------------------
# Puzzle and pair planning
# --------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    values = []
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        values.append(int(value))
    train, test = values
    if not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _puzzle_params(rng) -> dict:
    palette = [int(c) for c in rng.permutation(10)]
    roll = rng.random()
    tier = "minimal" if roll < 1 / 14 else ("rich" if roll < 1 / 14 + 1 / 7 else "normal")
    themes = set()
    if tier != "minimal":
        count = int(rng.choice((1, 2))) if tier == "rich" else int(rng.choice((0, 1, 1, 1, 2)))
        for theme in (str(t) for t in rng.permutation(_THEMES)):
            if len(themes) >= count:
                break
            if any(frozenset((theme, other)) in _EXCLUSIVE for other in themes):
                continue
            themes.add(theme)
    if tier == "minimal":
        stable = True
    elif tier == "rich":
        stable = False
    else:
        stable = bool(rng.random() < 0.4)
    backgrounds = palette[:1] if stable else palette[:int(rng.integers(2, 4))]
    if tier == "minimal":
        options = {"static": False, "orphan": False, "diagonal": False, "contact": False, "split": False}
        vocab = "blob"
    elif tier == "rich":
        options = {"static": True, "orphan": True, "diagonal": True, "contact": True, "split": True}
        vocab = "full"
    else:
        options = {"static": bool(rng.random() < 0.45), "orphan": bool(rng.random() < 0.3),
                   "diagonal": bool(rng.random() < 0.5), "contact": bool(rng.random() < 0.3),
                   "split": bool(rng.random() < 0.22)}
        vocab = "full" if rng.random() < 0.8 else "blob"
    params = {
        "palette": palette,
        "tier": tier,
        "backgrounds": backgrounds,
        "novel": palette[8:] if "new_colors" in themes else [],
        "themes": themes,
        "vocab": vocab,
        "train_cap": int(rng.integers(5, 13)) if "heavier" in themes else 24,
        "len_cap": int(rng.integers(3, 8)) if "longer" in themes else 30,
        "group_cap": int(rng.choice((2, 3))) if "more_groups" in themes else 5,
    }
    params.update(options)
    return params


def _side(rng, bands) -> int:
    weights = np.asarray([b[2] for b in bands], dtype=float)
    lo, hi, _ = bands[int(rng.choice(len(bands), p=weights / weights.sum()))]
    return int(rng.integers(lo, hi + 1))


_H_BANDS = ((4, 7, 0.17), (8, 15, 0.5), (16, 30, 0.33))
_W_BANDS = ((8, 12, 0.2), (13, 20, 0.45), (21, 30, 0.35))


def _sample_dims(rng, params, role, context, needs):
    themes = params["themes"]
    busy = sum(bool(needs.get(k)) for k in _FEATURES) + (role == "witness")
    for _ in range(80):
        if role == "test" and "bigger" in themes:
            height, width = int(rng.integers(12, 31)), int(rng.integers(16, 31))
        elif role == "test" and "compact" in themes:
            height, width = int(rng.integers(4, 11)), int(rng.integers(8, 17))
        else:
            height, width = _side(rng, _H_BANDS), _side(rng, _W_BANDS)
        if params["tier"] == "minimal":
            height, width = min(height, 16), min(width, 22)
        if busy:
            height, width = max(height, 5 + 2 * busy), max(width, 12 + busy)
        if role != "test":
            if "bigger" in themes and height * width > 380:
                continue
            if "compact" in themes and height * width < 150:
                continue
        else:
            if "bigger" in themes and height * width <= context["max_area"]:
                continue
            if "compact" in themes and height * width >= context["min_area"]:
                continue
            if "heavier" in themes:
                width = max(width, min(30, context["max_n"] + int(rng.integers(4, 9))))
            if "more_groups" in themes:
                height, width = max(height, int(rng.integers(10, 16))), max(width, int(rng.integers(15, 23)))
            if "longer" in themes:
                width = max(width, min(30, context["max_len"] + int(rng.integers(3, 8))))
        return min(height, 30), min(width, 30)
    return None


def _sample_push(rng, cap, low) -> int:
    roll = rng.random()
    if roll < 0.04:
        value = 2
    elif roll < 0.6:
        value = int(rng.integers(3, 9))
    elif roll < 0.87:
        value = int(rng.integers(9, 15))
    else:
        value = int(rng.integers(15, 25))
    if not low <= value <= cap:
        value = int(rng.integers(low, cap + 1))
    return value


def _split_sizes(rng, n) -> list:
    parts = 3 if n >= 9 and rng.random() < 0.25 else 2
    smallest = 3 if n >= 6 else 2
    positions = np.arange(smallest, n - smallest + 1)
    cuts = sorted(int(x) for x in rng.choice(positions, size=parts - 1, replace=False)) \
        if len(positions) >= parts - 1 else None
    if cuts is None:
        return [n]
    sizes = [b - a for a, b in zip([0] + cuts, cuts + [n])]
    if min(sizes) < 2:
        return [n // 2, n - n // 2]
    return sizes


def _sample_plan(rng, params, role, context, needs):
    dims = _sample_dims(rng, params, role, context, needs)
    if dims is None:
        return None
    height, width = dims
    area = height * width
    tier, themes = params["tier"], params["themes"]
    cap = min(24, width - 2, max(2, area // 6))
    if tier == "minimal":
        cap = min(cap, 10)
    if role != "test" and "heavier" in themes:
        cap = min(cap, params["train_cap"])
    low = 3 if tier == "minimal" else 2
    if cap < low:
        return None
    if role == "witness":
        group_count = 2 if tier == "minimal" else int(rng.choice((2, 3, 4), p=(0.45, 0.35, 0.2)))
    elif tier == "minimal":
        group_count = int(rng.choice((1, 2), p=(0.55, 0.45)))
    elif tier == "rich":
        group_count = int(rng.choice((2, 3, 4, 5), p=(0.3, 0.35, 0.25, 0.1)))
    else:
        group_count = int(rng.choice((1, 2, 3, 4), p=(0.3, 0.35, 0.23, 0.12)))
    if "more_groups" in themes:
        if role == "test":
            group_count = int(rng.integers(context["max_groups"] + 1, 6))
        else:
            group_count = min(group_count, params["group_cap"])
    pushes = [_sample_push(rng, cap, low) for _ in range(group_count)]
    heavy = None
    if role == "test" and "heavier" in themes:
        heavy = int(rng.integers(group_count))
        lo = context["max_n"] + 1
        if lo > cap:
            return None
        pushes[heavy] = int(rng.integers(lo, min(cap, context["max_n"] + 8) + 1))
    if group_count >= 2 and rng.random() < 0.18:
        i, j = (int(x) for x in rng.choice(group_count, size=2, replace=False))
        if j != heavy:
            pushes[j] = pushes[i]
    if role == "witness" and len(set(pushes)) == 1:
        if pushes[0] < cap:
            pushes[1] = pushes[0] + 1
        elif pushes[0] - 1 >= low:
            pushes[1] = pushes[0] - 1
        else:
            return None
    order = [int(i) for i in rng.permutation(group_count)]
    sparse_group = order[0] if needs.get("diagonal") else None
    split_group = None
    if params["split"] and (needs.get("split") or rng.random() < 0.35):
        candidates = [i for i in order if pushes[i] >= 4]
        if candidates:
            split_group = candidates[0]
        elif needs.get("split"):
            return None
    intact_group, clip_group = order[0], order[-1]
    multi_group = order[int(rng.integers(group_count))]
    kind_pool, kind_probs = ("intact", "clip", "vanish", "free"), (0.34, 0.26, 0.18, 0.22)
    groups = []
    for index, amount in enumerate(pushes):
        if tier == "minimal":
            count = int(rng.choice((1, 2, 3), p=(0.45, 0.35, 0.2)))
        else:
            count = int(rng.choice((1, 2, 3, 4, 5), p=(0.34, 0.3, 0.18, 0.11, 0.07)))
        if role == "witness" and index == multi_group:
            count = max(2, count)
        kinds = [str(k) for k in rng.choice(kind_pool, size=count, p=kind_probs)]
        if role != "test":
            kinds[0] = "intact" if rng.random() < 0.6 else "clip"
        if role == "witness":
            if index == intact_group:
                kinds[0] = "intact"
            if index == clip_group:
                kinds[0] = "clip"
        pieces = _split_sizes(rng, amount) if index == split_group else [amount]
        groups.append({"n": amount, "kinds": kinds, "pieces": pieces,
                       "sparse": index == sparse_group})
    if role == "test" and "longer" in themes:
        groups[int(rng.integers(group_count))]["kinds"].append("long")
    if role == "test" and needs.get("edge"):
        if not any(k in ("clip", "vanish") for g in groups for k in g["kinds"]):
            target = groups[int(rng.integers(group_count))]
            target["kinds"][int(rng.integers(len(target["kinds"])))] = \
                "clip" if rng.random() < 0.6 else "vanish"
    statics = []
    if params["static"] and (needs.get("static") or rng.random() < 0.5):
        kinds_count = int(rng.integers(1, 4 if tier == "rich" else 3))
        statics = [int(rng.choice((1, 2, 3), p=(0.5, 0.35, 0.15))) for _ in range(kinds_count)]
    orphans = []
    if params["orphan"] and (needs.get("orphan") or rng.random() < 0.4):
        orphan_count = 2 if tier == "rich" and rng.random() < 0.3 else 1
        orphans = [_sample_push(rng, min(cap, 12), 2) for _ in range(orphan_count)]
    budget = 9 - (len(params["novel"]) if role != "test" else 0)
    while group_count + len(statics) + len(orphans) > budget:
        if len(orphans) > (1 if needs.get("orphan") else 0):
            orphans.pop()
        elif len(statics) > (1 if needs.get("static") else 0):
            statics.pop()
        else:
            return None
    objects = sum(len(g["pieces"]) + len(g["kinds"]) for g in groups) + sum(statics) + len(orphans)
    if objects > max(4, area // 13):
        return None
    if sum(pushes) + sum(orphans) > 0.3 * area:
        return None
    contact = bool(params["contact"] and (needs.get("contact") or rng.random() < 0.7))
    match = None
    singles = [i for i in order if i not in (split_group, heavy, sparse_group)]
    if params["vocab"] == "full" and len(singles) >= 2 and rng.random() < 0.18:
        match = (singles[0], singles[1])
    return {"height": height, "width": width, "groups": groups, "statics": statics,
            "orphans": orphans, "contact": contact, "contact_rate": 0.55 if needs.get("contact") else 0.3,
            "match": match, "novel": role == "test" and bool(params["novel"]),
            "convoy": tier != "minimal" and bool(rng.random() < 0.25),
            "len_cap": params["len_cap"] if role != "test" else 30,
            "min_long": context.get("max_len", 0) + 1 if role == "test" else 1,
            "cap": cap}


def _build_pair(rng, plan, params, role):
    height, width = plan["height"], plan["width"]
    canvas = _Canvas(height, width, plan["contact"])
    max_rows = max(2, min(11, height - 1))
    max_cols = max(2, min(10, (width + 1) // 2))
    sparse = params["diagonal"]
    shapes = {}
    for index, group in enumerate(plan["groups"]):
        for piece, size in enumerate(group["pieces"]):
            shape, _ = _weight_shape(rng, size, max_rows, max_cols, params["vocab"], sparse,
                                     require_sparse=group["sparse"] and piece == 0)
            if shape is None:
                return None
            shapes[(index, piece)] = shape
    for index, size in enumerate(plan["orphans"]):
        shape, _ = _weight_shape(rng, size, max_rows, max_cols, params["vocab"], sparse)
        if shape is None:
            return None
        shapes[(200 + index, 0)] = shape
    if plan["match"] is not None:
        # Teaching configuration: same bounding box, different cell count.
        i, j = plan["match"]
        h, w = _dims(shapes[(i, 0)])
        current = len(shapes[(i, 0)])
        options = [n for n in range(max(2, h + w - 1), h * w) if n != current and n <= plan["cap"]]
        for n in [options[int(k)] for k in rng.permutation(len(options))[:4]]:
            shape = _blob(rng, n, h, w, str(rng.choice(_BLOB_STYLES)), exact=(h, w))
            if shape is not None and (sparse or _four_pieces(shape) == 1):
                shapes[(j, 0)] = shape
                plan["groups"][j]["n"] = n
                plan["groups"][j]["pieces"] = [n]
                break
    weights = []
    for (key, piece), shape in sorted(shapes.items(), key=lambda kv: -len(kv[1])):
        cells = canvas.place_weight(rng, shape)
        if cells is None:
            return None
        weights.append((key, cells))
    bar_specs = []
    for index, group in enumerate(plan["groups"]):
        for kind in group["kinds"]:
            bar_specs.append((index, kind, group["n"]))
    for index, count in enumerate(plan["statics"]):
        bar_specs.extend((100 + index, "static", 0) for _ in range(count))
    priority = {"long": 0, "clip": 1, "intact": 2, "vanish": 3, "free": 4, "static": 5}
    bar_specs.sort(key=lambda spec: priority[spec[1]])
    bars = []
    for key, kind, amount in bar_specs:
        want = plan["contact"] and bool(bars) and rng.random() < plan["contact_rate"]
        convoy = plan["convoy"] and rng.random() < 0.6
        if kind == "long":
            bar = canvas.place_bar(rng, key, amount, "free", min_len=plan["min_long"],
                                   want_contact=want, prefer_rows=convoy)
        else:
            bar = canvas.place_bar(rng, key, amount, kind, len_cap=plan["len_cap"],
                                   want_contact=want, prefer_rows=convoy)
        if bar is None:
            return None
        bars.append(bar)
    background = int(rng.choice(params["backgrounds"]))
    excluded = {background} | (set() if role == "test" else set(params["novel"]))
    pool = [c for c in range(10) if c not in excluded]
    keys = list(range(len(plan["groups"])))
    keys += [100 + index for index in range(len(plan["statics"]))]
    keys += [200 + index for index in range(len(plan["orphans"]))]
    if len(pool) < len(keys):
        return None
    chosen = [int(c) for c in rng.permutation(pool)[:len(keys)]]
    color_of = dict(zip(keys, chosen))
    if plan["novel"] and not any(color_of[i] in params["novel"] for i in range(len(plan["groups"]))):
        target = int(rng.integers(len(plan["groups"])))
        new = int(rng.choice(params["novel"]))
        for key, value in list(color_of.items()):
            if value == new:
                color_of[key] = color_of[target]
        color_of[target] = new
    source = np.full((height, width), background, dtype=int)
    for key, cells in weights:
        for r, c in cells:
            source[r, c] = color_of[key]
    target = np.full((height, width), background, dtype=int)
    for bar in bars:
        color = color_of[bar["key"]]
        source[bar["row"], bar["start"]:bar["end"]] = color
        lo, hi = max(0, bar["start"] - bar["shift"]), bar["end"] - bar["shift"]
        if hi > lo:
            target[bar["row"], lo:hi] = color
    if int(np.sum(source == background)) * 2 <= source.size:
        return None
    return {"input": source, "output": target}


def _pair_refutes(pair, names) -> bool:
    target = np.asarray(pair["output"])
    guesses = _shortcut_outputs(pair["input"])
    return all(not np.array_equal(guesses[name], target) for name in names)


def _themes_hold(params, facts, context) -> bool:
    themes = params["themes"]
    area = facts["shape"][0] * facts["shape"][1]
    if "heavier" in themes and max(facts["amounts"][c] for c in facts["groups"]) <= context["max_n"]:
        return False
    if "more_groups" in themes and len(facts["groups"]) <= context["max_groups"]:
        return False
    if "bigger" in themes and area <= context["max_area"]:
        return False
    if "compact" in themes and area >= context["min_area"]:
        return False
    if "new_colors" in themes and not set(facts["groups"]) & set(params["novel"]):
        return False
    if "longer" in themes and facts["max_len"] <= context["max_len"]:
        return False
    return True


def _make_pair(rng, params, role, context, needs, seen):
    for _ in range(260):
        plan = _sample_plan(rng, params, role, context, needs)
        if plan is None:
            continue
        pair = _build_pair(rng, plan, params, role)
        if pair is None:
            continue
        key = (pair["input"].shape, pair["input"].tobytes())
        if key in seen:
            continue
        try:
            facts = _analyze_pair(pair)
        except AssertionError:
            continue
        if role != "test" and not all(any(b["status"] in ("intact", "clip") for b in bars)
                                      for bars in facts["groups"].values()):
            continue
        if any(needs.get(feature) and not facts[feature] for feature in _FEATURES):
            continue
        if needs.get("contact") and not _pair_refutes(pair, ("colorblind", "rigid_cluster")):
            continue
        if needs.get("split") and not _pair_refutes(pair, ("largest_piece", "nearest_piece")):
            continue
        if needs.get("diagonal") and not _pair_refutes(pair, ("four_conn",)):
            continue
        if role == "test":
            if needs.get("edge") and not any(s in ("clip", "vanish") for s in facts["statuses"]):
                continue
            if not _themes_hold(params, facts, context):
                continue
        return pair, facts
    raise RuntimeError(f"could not build a {role} pair")


def _context(train_facts) -> dict:
    return {
        "max_n": max(n for f in train_facts for c, n in f["amounts"].items() if c in f["groups"]),
        "max_groups": max(len(f["groups"]) for f in train_facts),
        "max_area": max(f["shape"][0] * f["shape"][1] for f in train_facts),
        "min_area": min(f["shape"][0] * f["shape"][1] for f in train_facts),
        "max_len": max(f["max_len"] for f in train_facts),
    }


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    params = _puzzle_params(rng)
    for _ in range(40):
        carriers = {}
        for feature in _FEATURES:
            if params[feature]:
                carriers[feature] = int(rng.integers(num_train))
        train, train_facts, seen = [], [], set()
        try:
            for index in range(num_train):
                role = "witness" if index == 0 else "train"
                needs = {feature: True for feature, slot in carriers.items() if slot == index}
                pair, facts = _make_pair(rng, params, role, {}, needs, seen)
                seen.add((pair["input"].shape, pair["input"].tobytes()))
                train.append(pair)
                train_facts.append(facts)
        except RuntimeError:
            continue
        summary = _train_summary(train)
        if summary["problems"]:
            continue
        context = _context(train_facts)
        test, test_seen = [], set(seen)
        for index in range(num_test):
            needs = {"edge": index == 0}
            pair = None
            for _ in range(16):
                try:
                    candidate, _ = _make_pair(rng, params, "test", context, needs, test_seen)
                except RuntimeError:
                    break
                if not _test_problems(summary, [candidate]):
                    pair = candidate
                    break
            if pair is None:
                break
            test_seen.add((pair["input"].shape, pair["input"].tobytes()))
            test.append(pair)
        if len(test) != num_test:
            continue
        order = rng.permutation(num_train)
        puzzle = {"train": [train[int(i)] for i in order],
                  "test": [test[int(i)] for i in rng.permutation(num_test)]}
        validate_generated_puzzle(puzzle)
        return puzzle
    raise RuntimeError("could not build a puzzle")


# --------------------------------------------------------------------------
# Self test
# --------------------------------------------------------------------------

def _same(left, right) -> bool:
    return all(len(left[s]) == len(right[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(left[s], right[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def _hand_cases():
    """Hand-built inputs with independently written expected outputs."""
    cases = []
    grid = np.zeros((7, 12), dtype=int)
    grid[0, 0:2] = 2; grid[1, 1] = 2; grid[1, 2] = 2          # weight of 2: 4 cells
    grid[4, 5:8] = 2                                          # bar of 2 -> moves 4
    grid[6, 2:5] = 2                                          # bar of 2 -> clipped
    grid[2, 8:10] = 3; grid[3, 9] = 3                         # weight of 3: 3 cells
    grid[0, 6:8] = 3                                          # bar of 3 -> moves 3
    grid[5, 10:12] = 4                                        # static bar
    expected = np.zeros((7, 12), dtype=int)
    expected[4, 1:4] = 2; expected[6, 0:1] = 2; expected[0, 3:5] = 3; expected[5, 10:12] = 4
    cases.append((grid, expected))
    # Contact stack: 3 over 4, pushes 4 and 3, shears apart.
    grid = np.zeros((8, 14), dtype=int)
    grid[0, 10:12] = 3; grid[1, 10] = 3; grid[1, 12] = 3      # weight 3: 4 cells, diagonal link
    grid[5, 4:9] = 3; grid[6, 4:9] = 4                        # touching stacked bars
    grid[3, 1:3] = 4; grid[2, 1] = 4                          # weight 4: 3 cells
    expected = np.zeros((8, 14), dtype=int)
    expected[5, 0:5] = 3; expected[6, 1:6] = 4
    cases.append((grid, expected))
    # Split weight: pieces of 3 and 4 cells push the bar 7.
    grid = np.zeros((8, 14), dtype=int)
    grid[0, 0] = 2; grid[1, 0:2] = 2
    grid[0, 11] = 2; grid[1, 11:13] = 2; grid[2, 12] = 2
    grid[5, 9:13] = 2
    expected = np.zeros((8, 14), dtype=int)
    expected[5, 2:6] = 2
    cases.append((grid, expected))
    # Vertical line weight (5) and checkerboard weight (5 cells, no 4-adjacency).
    grid = np.full((9, 15), 1, dtype=int)
    grid[1:6, 13] = 5; grid[7, 6:9] = 5
    for r, c in ((0, 2), (0, 4), (1, 3), (2, 2), (2, 4)):
        grid[r, c] = 7
    grid[4, 3:7] = 7; grid[8, 0:2] = 6
    expected = np.full((9, 15), 1, dtype=int)
    expected[7, 1:4] = 5; expected[4, 0:2] = 7; expected[8, 0:2] = 6
    cases.append((grid, expected))
    return cases


def self_test() -> None:
    # Count handling.
    for train, test in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        puzzle = generate_puzzle(31337 + train * 10 + test, train, test)
        assert len(puzzle["train"]) == train and len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(7, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {train}/{test}")
    for grid, expected in _hand_cases():
        assert np.array_equal(solve(grid), expected)
    # Contact shortcuts differ on the stack case; split shortcuts differ on the split case.
    cases = _hand_cases()
    assert _pair_refutes({"input": cases[1][0], "output": cases[1][1]}, ("colorblind", "rigid_cluster"))
    assert _pair_refutes({"input": cases[2][0], "output": cases[2][1]}, ("largest_piece", "nearest_piece"))
    # Determinism and cross-seed variety over 42 seeds.
    first = generate_puzzle(4242, 6, 2)
    assert _same(first, generate_puzzle(4242, 6, 2))
    assert not _same(first, generate_puzzle(4243, 6, 2))
    keys = ("static", "orphan", "diagonal", "contact", "split", "same_row", "test_vanish",
            "test_clip", "novel_test", "bg_varies", "four_groups", "one_group_test",
            "tall_canvas", "wide_canvas", "long_bar", "busy_row", "push2", "push15",
            "ring", "vline", "sparse", "tall_weight", "rect", "minimal", "rich")
    stats = {key: 0 for key in keys}
    shapes, amounts, themes, structures = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_facts = [_analyze_pair(p) for p in puzzle["train"]]
        test_facts = [_analyze_pair(p) for p in puzzle["test"]]
        train_colors = set().union(*(f["colors"] for f in train_facts))
        stats["bg_varies"] += len({f["background"] for f in train_facts + test_facts}) > 1
        for f in train_facts + test_facts:
            height, width = f["shape"]
            shapes.add(f["shape"])
            amounts.update(f["amounts"].values())
            for key in ("static", "orphan", "diagonal", "contact", "split", "same_row"):
                stats[key] += f[key]
            stats["four_groups"] += len(f["groups"]) >= 4
            stats["tall_canvas"] += height > 1.4 * width
            stats["wide_canvas"] += width > 2.5 * height
            stats["long_bar"] += f["max_len"] >= 10
            rows = [b["row"] for b in f["bars"]]
            stats["busy_row"] += max(rows.count(r) for r in rows) >= 3
            stats["push2"] += any(n == 2 for c, n in f["amounts"].items() if c in f["groups"])
            stats["push15"] += any(n >= 15 for c, n in f["amounts"].items() if c in f["groups"])
            for weight in f["weights"]:
                traits = _weight_traits(weight["cells"])
                stats["ring"] += traits["ring"]
                stats["vline"] += traits["vline"]
                stats["sparse"] += traits["sparse"]
                stats["rect"] += traits["rect"]
                stats["tall_weight"] += traits["h"] >= 7
        for f in test_facts:
            stats["test_vanish"] += "vanish" in f["statuses"]
            stats["test_clip"] += "clip" in f["statuses"]
            stats["novel_test"] += bool(f["colors"] - train_colors)
            stats["one_group_test"] += len(f["groups"]) == 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                mapping = {}
                structures.add(tuple(mapping.setdefault(int(v), len(mapping))
                                     for v in np.asarray(pair["input"]).flat))
        params = _puzzle_params(np.random.default_rng(seed))
        themes.add(tuple(sorted(params["themes"])))
        stats["minimal"] += params["tier"] == "minimal"
        stats["rich"] += params["tier"] == "rich"
        # The answer depends on both parents on every test pair.
        for pair in puzzle["test"]:
            guesses = _shortcut_outputs(pair["input"])
            target = np.asarray(pair["output"])
            assert not np.array_equal(guesses["unit_shift"], target)
            assert not np.array_equal(guesses["bars_stay"], target)
    assert len(shapes) >= 60 and len(amounts) >= 18 and len(structures) >= 200
    assert len(themes) >= 8
    assert all(value > 0 for value in stats.values()), stats
