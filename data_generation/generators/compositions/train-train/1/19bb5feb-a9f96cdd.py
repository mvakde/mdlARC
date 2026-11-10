"""Keyed diagonal stamps: carrier panels compress into 2x2 keys that are stamped
diagonally around isolated markers of the same carrier color.

Rule (input-only):
  * The most frequent color is the background.
  * Each carrier panel is a solid rectangle (carrier color frame) containing 2..4
    monochrome payload blocks of one common size placed at two row levels and two
    column levels (parent A). Its 2x2 key is read by relative level order:
    upper-left, upper-right, lower-left, lower-right. An unoccupied slot is empty.
  * Every isolated single cell whose color equals a panel's carrier color is a
    marker bound to that panel (new binding concept: the carrier color, which A
    discards, now names the key).
  * Output = input, except each marker cell becomes background and its four
    diagonal neighbours (parent B) are painted with the bound key:
    NW <- upper-left, NE <- upper-right, SW <- lower-left, SE <- lower-right.
    Empty key slots paint nothing; diagonals outside the canvas are clipped.
  * Panels are left unchanged.
"""
from __future__ import annotations

import numpy as np

# (slot row, slot col, d_row, d_col) for NW, NE, SW, SE.
_DIRECTIONS = ((0, 0, -1, -1), (0, 1, -1, 1), (1, 0, 1, -1), (1, 1, 1, 1))
_DIRECTION_NAMES = ("NW", "NE", "SW", "SE")

_PATTERNS = {
    "full": ((1, 1), (1, 1)),
    "miss_nw": ((0, 1), (1, 1)),
    "miss_ne": ((1, 0), (1, 1)),
    "miss_sw": ((1, 1), (0, 1)),
    "miss_se": ((1, 1), (1, 0)),
    "diag_main": ((1, 0), (0, 1)),
    "diag_anti": ((0, 1), (1, 0)),
}
_PATTERN_WEIGHTS = {
    "full": 0.28, "miss_nw": 0.11, "miss_ne": 0.11, "miss_sw": 0.11,
    "miss_se": 0.11, "diag_main": 0.14, "diag_anti": 0.14,
}
_MISSING_PATTERNS = ("miss_nw", "miss_ne", "miss_sw", "miss_se", "diag_main", "diag_anti")
# Missing patterns where a reading-order fill of the present colors differs from slot order.
_ORDER_SENSITIVE = ("miss_nw", "miss_ne", "miss_sw", "diag_main", "diag_anti")
_BLOCK_SHAPES = ((1, 1), (2, 2), (3, 3), (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2))

_MIN_MARKER_PANEL_DISTANCE = 3
_MIN_MARKER_MARKER_DISTANCE = 3
_MIN_PANEL_GAP = 2
# Soft panel extent caps per scale regime for 1, 2 and 3 panels.
_SCALE_CAPS = {"small": (8, 7, 7), "medium": (11, 9, 8), "large": (15, 11, 9)}


class _Retry(Exception):
    """Internal signal: resample this pair or puzzle."""


# ---------------------------------------------------------------------------
# Input analysis shared by solve() and validation.
# ---------------------------------------------------------------------------

def _components(grid, mask):
    """4-connected same-color components of cells where mask is True."""
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            color = int(grid[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height and 0 <= nx < width and mask[ny, nx]
                            and not seen[ny, nx] and int(grid[ny, nx]) == color):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            result.append((color, cells))
    return result


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _analyze(grid):
    grid = np.asarray(grid, dtype=int)
    height, width = grid.shape
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(counts.argmax())
    components = _components(grid, grid != background)

    panels = []
    for color, cells in components:
        r0, r1, c0, c1 = _bbox(cells)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        if (sub == background).any() or (sub == color).all():
            continue
        frame = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]])
        if not (frame == color).all():
            continue
        panels.append({"carrier": color, "box": (r0, r1, c0, c1)})
    panels.sort(key=lambda p: (p["box"][0], p["box"][2]))

    for panel in panels:
        r0, r1, c0, c1 = panel["box"]
        sub = grid[r0:r1 + 1, c0:c1 + 1]
        blocks = []
        for color, cells in _components(sub, sub != panel["carrier"]):
            br0, br1, bc0, bc1 = _bbox(cells)
            blocks.append({"color": color, "top": br0 + r0, "left": bc0 + c0,
                           "h": br1 - br0 + 1, "w": bc1 - bc0 + 1, "size": len(cells)})
        tops = sorted({b["top"] for b in blocks})
        lefts = sorted({b["left"] for b in blocks})
        key, conflict = None, False
        if len(tops) == 2 and len(lefts) == 2:
            key = [[None, None], [None, None]]
            for block in blocks:
                i, j = tops.index(block["top"]), lefts.index(block["left"])
                if key[i][j] is None:
                    key[i][j] = block["color"]
                else:
                    conflict = True
        panel.update(blocks=blocks, tops=tops, lefts=lefts, key=key, conflict=conflict)

    covered = np.zeros(grid.shape, dtype=bool)
    for panel in panels:
        r0, r1, c0, c1 = panel["box"]
        covered[r0:r1 + 1, c0:c1 + 1] = True
    carrier_index = {}
    for index, panel in enumerate(panels):
        carrier_index.setdefault(panel["carrier"], index)
    markers, strays = [], []
    for color, cells in components:
        if all(covered[r, c] for r, c in cells):
            continue
        if len(cells) == 1 and color in carrier_index:
            markers.append({"pos": cells[0], "color": color, "panel": carrier_index[color]})
        else:
            strays.append({"color": color, "cells": cells})
    markers.sort(key=lambda m: m["pos"])
    return {"grid": grid, "shape": (height, width), "background": background,
            "counts": counts, "panels": panels, "markers": markers, "strays": strays}


def _render(analysis, binding=None, keys=None):
    """Stamp every marker. binding: marker index -> panel index; keys: panel index -> key."""
    grid = analysis["grid"]
    height, width = analysis["shape"]
    out = grid.copy()
    for index, marker in enumerate(analysis["markers"]):
        row, col = marker["pos"]
        out[row, col] = analysis["background"]
        panel_index = marker["panel"] if binding is None else binding[index]
        key = (analysis["panels"][panel_index]["key"] if keys is None
               else keys[panel_index])
        if key is None:
            continue
        for i, j, dr, dc in _DIRECTIONS:
            color = key[i][j]
            rr, cc = row + dr, col + dc
            if color is not None and 0 <= rr < height and 0 <= cc < width:
                out[rr, cc] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid, dtype=int)
    return _render(_analyze(grid))


# ---------------------------------------------------------------------------
# Alternative (shortcut) readings used to prove the demonstrations matter.
# ---------------------------------------------------------------------------

def _reading_order_key(key):
    colors = [key[i][j] for i, j, _, _ in _DIRECTIONS if key[i][j] is not None]
    colors += [None] * (4 - len(colors))
    return [[colors[0], colors[1]], [colors[2], colors[3]]]


def _box_distance(pos, box):
    row, col = pos
    r0, r1, c0, c1 = box
    dy = max(r0 - row, 0, row - r1)
    dx = max(c0 - col, 0, col - c1)
    return max(dy, dx), dy + dx


def _shortcut_reading_order(grid):
    analysis = _analyze(grid)
    keys = [None if p["key"] is None else _reading_order_key(p["key"]) for p in analysis["panels"]]
    return _render(analysis, keys=keys)


def _shortcut_nearest_panel(grid):
    analysis = _analyze(grid)
    binding = []
    for marker in analysis["markers"]:
        options = [(_box_distance(marker["pos"], p["box"]), i) for i, p in enumerate(analysis["panels"])]
        binding.append(min(options)[1])
    return _render(analysis, binding=binding)


def _shortcut_fixed_key(grid, key):
    analysis = _analyze(grid)
    return _render(analysis, keys=[key] * len(analysis["panels"]))


def _shortcut_quadrant(grid):
    """Assign blocks to slots by the panel's geometric midlines instead of level order."""
    analysis = _analyze(grid)
    keys = []
    for panel in analysis["panels"]:
        r0, r1, c0, c1 = panel["box"]
        key = [[None, None], [None, None]]
        for block in panel["blocks"]:
            cy = block["top"] + (block["h"] - 1) / 2.0
            cx = block["left"] + (block["w"] - 1) / 2.0
            i = 0 if cy < (r0 + r1) / 2.0 else 1
            j = 0 if cx < (c0 + c1) / 2.0 else 1
            key[i][j] = block["color"]
        keys.append(key)
    return _render(analysis, keys=keys)


# ---------------------------------------------------------------------------
# Validation.
# ---------------------------------------------------------------------------

def _as_grid(value):
    array = np.asarray(value)
    if array.ndim != 2 or not (1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("grid cells must be integers")
    if array.min() < 0 or array.max() > 9:
        raise ValueError("grid colors must be 0..9")
    return array.astype(int)


def _key_tuple(key):
    return tuple(tuple(-1 if v is None else int(v) for v in row) for row in key)


def _pattern_name(key):
    occupancy = tuple(tuple(int(v is not None) for v in row) for row in key)
    for name, pattern in _PATTERNS.items():
        if pattern == occupancy:
            return name
    return None


def _marker_effects(pos, key, shape):
    row, col = pos
    height, width = shape
    painted, clipped, missing = [], [], []
    for (i, j, dr, dc), name in zip(_DIRECTIONS, _DIRECTION_NAMES):
        inside = 0 <= row + dr < height and 0 <= col + dc < width
        present = key[i][j] is not None
        if present and inside:
            painted.append(name)
        elif present:
            clipped.append(name)
        elif inside:
            missing.append(name)
    return painted, clipped, missing


def _pair_problems(input_grid, output_grid):
    problems = []
    analysis = _analyze(input_grid)
    grid = analysis["grid"]
    counts = sorted(analysis["counts"].tolist(), reverse=True)
    if counts[0] <= counts[1]:
        problems.append("background is not strictly the most frequent color")
    panels, markers = analysis["panels"], analysis["markers"]
    if not panels:
        problems.append("no carrier panel")
    if analysis["strays"]:
        problems.append("non-panel object that is not a marker")
    carriers = [p["carrier"] for p in panels]
    if len(set(carriers)) != len(carriers):
        problems.append("two panels share a carrier color")
    block_colors = set()
    for p_index, panel in enumerate(panels):
        blocks = panel["blocks"]
        if not 2 <= len(blocks) <= 4:
            problems.append("panel must contain 2..4 payload blocks")
        shapes = {(b["h"], b["w"]) for b in blocks}
        if len(shapes) != 1:
            problems.append("payload blocks in a panel must share one size")
        for block in blocks:
            if block["size"] != block["h"] * block["w"]:
                problems.append("payload block is not a solid rectangle")
            if (block["h"], block["w"]) not in _BLOCK_SHAPES:
                problems.append("unsupported payload block size")
            block_colors.add(block["color"])
        if panel["key"] is None or panel["conflict"]:
            problems.append("payload blocks do not reveal exactly two row and two column levels")
            continue
        if len(shapes) == 1:
            bh, bw = next(iter(shapes))
            if panel["tops"][1] - panel["tops"][0] < bh + 1 or panel["lefts"][1] - panel["lefts"][0] < bw + 1:
                problems.append("payload levels must be separated by carrier cells")
        if _pattern_name(panel["key"]) is None:
            problems.append("unknown occupancy pattern")
        for q_index in range(p_index + 1, len(panels)):
            a, b = panel["box"], panels[q_index]["box"]
            row_gap = max(b[0] - a[1] - 1, a[0] - b[1] - 1, 0)
            col_gap = max(b[2] - a[3] - 1, a[2] - b[3] - 1, 0)
            if max(row_gap, col_gap) < _MIN_PANEL_GAP:
                problems.append("panels are too close")
    if block_colors & (set(carriers) | {analysis["background"]}):
        problems.append("payload colors must differ from background and carrier colors")
    if len({_key_tuple(p["key"]) for p in panels if p["key"] is not None}) != len(panels):
        problems.append("panels in one input must carry different keys")
    bound = {m["panel"] for m in markers}
    if bound != set(range(len(panels))):
        problems.append("every panel needs at least one marker")
    for index, marker in enumerate(markers):
        for panel in panels:
            if _box_distance(marker["pos"], panel["box"])[0] < _MIN_MARKER_PANEL_DISTANCE:
                problems.append("marker too close to a panel")
        for other in markers[index + 1:]:
            (r1, c1), (r2, c2) = marker["pos"], other["pos"]
            if max(abs(r1 - r2), abs(c1 - c2)) < _MIN_MARKER_MARKER_DISTANCE:
                problems.append("markers too close together")
    output = np.asarray(output_grid, dtype=int)
    if output.shape != grid.shape:
        problems.append("output shape differs from input shape")
    elif not np.array_equal(output, _render(analysis)):
        problems.append("output does not follow the keyed diagonal stamp rule")
    return problems, analysis


def _pair_facts(analysis):
    facts = {"painted": set(), "clip": False, "missing": False, "compound": False,
             "misbind": False, "four": False, "keys": [], "patterns": [],
             "n_panels": len(analysis["panels"]), "shapes": set(),
             "background": analysis["background"]}
    panels = analysis["panels"]
    for panel in panels:
        facts["keys"].append(_key_tuple(panel["key"]))
        facts["patterns"].append(_pattern_name(panel["key"]))
        facts["shapes"].add((panel["blocks"][0]["h"], panel["blocks"][0]["w"]))
    for marker in analysis["markers"]:
        key = panels[marker["panel"]]["key"]
        painted, clipped, missing = _marker_effects(marker["pos"], key, analysis["shape"])
        facts["painted"].update(painted)
        if painted and clipped:
            facts["clip"] = True
        if painted and missing:
            facts["missing"] = True
        if painted and clipped and missing:
            facts["compound"] = True
        if len(painted) == 4:
            facts["four"] = True
        own = _box_distance(marker["pos"], panels[marker["panel"]]["box"])
        for index, panel in enumerate(panels):
            if index != marker["panel"]:
                other = _box_distance(marker["pos"], panel["box"])
                if other[0] < own[0] and other[1] < own[1]:
                    facts["misbind"] = True
    return facts


def _puzzle_problems(puzzle):
    problems = []
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        return ["puzzle must have exactly train/test keys"]
    train, test = puzzle["train"], puzzle["test"]
    if not isinstance(train, list) or not isinstance(test, list):
        return ["train/test must be lists"]
    if not (2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)):
        problems.append("illegal pair counts")
    seen = set()
    facts = {"train": [], "test": []}
    grids = {"train": [], "test": []}
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                return ["pair must have exactly input/output keys"]
            try:
                inp, out = _as_grid(pair["input"]), _as_grid(pair["output"])
            except ValueError as error:
                return [str(error)]
            key = (inp.shape, inp.tobytes())
            if key in seen:
                problems.append("duplicate input")
            seen.add(key)
            pair_problems, analysis = _pair_problems(inp, out)
            problems.extend(f"{split}: {p}" for p in pair_problems)
            if pair_problems:
                continue
            facts[split].append(_pair_facts(analysis))
            grids[split].append((inp, out))
    if problems:
        return problems

    tf, sf = facts["train"], facts["test"]
    painted = set().union(*(f["painted"] for f in tf))
    if painted != set(_DIRECTION_NAMES):
        problems.append("training must paint all four diagonal directions")
    if not any(f["missing"] for f in tf):
        problems.append("training must show an empty key slot inside the canvas")
    if not any(f["clip"] for f in tf):
        problems.append("training must show boundary clipping of an occupied slot")
    train_keys = [k for f in tf for k in f["keys"]]
    if len(set(train_keys)) < 2:
        problems.append("training keys must vary")
    if any(k in set(train_keys) for f in sf for k in f["keys"]):
        problems.append("test keys must be new")
    if not sf[0]["compound"]:
        problems.append("first test needs a marker with both clipping and an empty slot")
    multi_test = any(f["n_panels"] > 1 for f in sf)
    multi_any = multi_test or any(f["n_panels"] > 1 for f in tf)
    if multi_any and not any(f["misbind"] and f["n_panels"] > 1 for f in tf):
        problems.append("multi-panel puzzles need a training marker nearer a non-matching panel")
    test_shapes = set().union(*(f["shapes"] for f in sf))
    train_shapes = set().union(*(f["shapes"] for f in tf))
    if not test_shapes <= train_shapes:
        problems.append("test block sizes must appear in training")

    # Shortcut checks: each must fail somewhere in training and somewhere in test.
    first_key = _analyze(grids["train"][0][0])["panels"][0]["key"]
    shortcuts = {
        "reading-order key": _shortcut_reading_order,
        "fixed first-train key": lambda g: _shortcut_fixed_key(g, first_key),
    }
    if multi_any:
        shortcuts["nearest-panel binding"] = _shortcut_nearest_panel
    for name, function in shortcuts.items():
        for split in ("train", "test"):
            pool = grids[split][1:] if (split == "train" and name.startswith("fixed")) else grids[split]
            if name.startswith("nearest") and split == "test" and not multi_test:
                continue
            if not any(not np.array_equal(function(i), o) for i, o in pool):
                problems.append(f"{name} shortcut is not refuted in {split}")
    return problems


def validate_generated_puzzle(puzzle: dict) -> None:
    problems = _puzzle_problems(puzzle)
    if problems:
        raise ValueError("; ".join(sorted(set(problems))))


# ---------------------------------------------------------------------------
# Generation.
# ---------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    num_train = 4 if num_train is None else num_train
    num_test = 1 if num_test is None else num_test
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


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def _sample_params(rng):
    panel_mode = _pick(rng, ("single",) * 3 + ("mixed",) * 5 + ("multi",) * 2)
    size_mode = _pick(rng, ("source",) * 1 + ("uniform",) * 1 + ("mixed",) * 2)
    uniform_shape = _pick(rng, _BLOCK_SHAPES)
    background_mode = "stable" if rng.random() < 0.4 else "vary"
    scale = _pick(rng, ("small",) * 3 + ("medium",) * 4 + ("large",) * 3)
    return {
        "panel_mode": panel_mode,
        "size_mode": size_mode,
        "uniform_shape": uniform_shape,
        "mixed_shapes": tuple(_BLOCK_SHAPES[i] for i in rng.permutation(len(_BLOCK_SHAPES))[:int(rng.integers(3, 6))]),
        "background_mode": background_mode,
        "background": _sample_background(rng),
        "edge_rate": float(rng.uniform(0.15, 0.45)),
        "corner_rate": float(rng.uniform(0.1, 0.4)),
        "offcenter_rate": float(rng.uniform(0.15, 0.55)),
        "scale": scale,
        "density": int(rng.integers(1, 3)) + (1 if scale == "large" else 0),
        "show_four": bool(rng.random() < 0.7),
        "repeat_rate": float(rng.uniform(0.1, 0.4)),
    }


def _sample_background(rng):
    return 0 if rng.random() < 0.3 else int(rng.integers(10))


def _block_shape(rng, params):
    if params["size_mode"] == "source":
        return (2, 2)
    if params["size_mode"] == "uniform":
        return params["uniform_shape"]
    return _pick(rng, params["mixed_shapes"])


def _sample_pattern(rng, allowed=None):
    names = list(_PATTERNS) if allowed is None else list(allowed)
    weights = np.array([_PATTERN_WEIGHTS[n] for n in names], dtype=float)
    return names[int(rng.choice(len(names), p=weights / weights.sum()))]


def _sample_axis(rng, block, cap, offcenter):
    """Panel extent along one axis plus the two payload level offsets inside it."""
    lowest = 2 * block + 3
    highest = max(lowest, min(cap, lowest + 7))
    span = highest - lowest
    extent = lowest + min(int(rng.integers(0, span + 1)), int(rng.integers(0, span + 1)))
    if rng.random() < 0.3:
        extent = lowest + int(rng.integers(0, span + 1))
    slack = extent - lowest
    need = block + 2  # smallest slack that lets both levels sit on one side of the midline
    if offcenter and cap + 3 - lowest >= need:
        slack = max(slack, min(need + int(rng.integers(0, 3)), cap + 3 - lowest))
        extent = lowest + slack
        big = int(rng.integers((block + 1 + slack) // 2 + 1, slack + 1))
        rest = slack - big
        gap_extra = int(rng.integers(0, rest + 1))
        other = rest - gap_extra
        before, after = (1 + big, 1 + other) if rng.random() < 0.5 else (1 + other, 1 + big)
        first = before
        second = first + block + 1 + gap_extra
        assert second + block + after == extent
        return extent, (first, second)
    options = [(a, b) for a in range(1, extent) for b in range(a + block + 1, extent)
               if b + block <= extent - 1]
    return extent, _pick(rng, options)


def _sample_panel(rng, params, shape, pattern, cap):
    bh, bw = shape
    offcenter = rng.random() < params["offcenter_rate"]
    height, (t0, t1) = _sample_axis(rng, bh, cap, offcenter and rng.random() < 0.7)
    width, (l0, l1) = _sample_axis(rng, bw, cap, offcenter and rng.random() < 0.7)
    return {"h": height, "w": width, "shape": shape, "pattern": pattern,
            "tops": (t0, t1), "lefts": (l0, l1)}


def _assign_key_colors(rng, panel, pool, repeat_rate):
    occupancy = _PATTERNS[panel["pattern"]]
    slots = [(i, j) for i, j, _, _ in _DIRECTIONS if occupancy[i][j]]
    colors = [int(c) for c in rng.choice(pool, size=len(slots), replace=len(pool) < len(slots))]
    if len(slots) >= 2 and rng.random() < repeat_rate:
        a, b = rng.choice(len(slots), size=2, replace=False)
        colors[int(b)] = colors[int(a)]
    key = [[None, None], [None, None]]
    for (i, j), color in zip(slots, colors):
        key[i][j] = color
    panel["key"] = key


def _box_gap(a, b):
    row_gap = max(b[0] - a[1] - 1, a[0] - b[1] - 1, 0)
    col_gap = max(b[2] - a[3] - 1, a[2] - b[3] - 1, 0)
    return max(row_gap, col_gap)


def _place_panels(rng, height, width, panels):
    boxes = []
    for panel in panels:
        if panel["h"] > height or panel["w"] > width:
            return None
        for _ in range(80):
            r0 = int(rng.integers(0, height - panel["h"] + 1))
            c0 = int(rng.integers(0, width - panel["w"] + 1))
            box = (r0, r0 + panel["h"] - 1, c0, c0 + panel["w"] - 1)
            if all(_box_gap(box, other) >= _MIN_PANEL_GAP for other in boxes):
                boxes.append(box)
                break
        else:
            return None
    return boxes


def _canvas_size(rng, params, panels):
    max_h = max(p["h"] for p in panels)
    max_w = max(p["w"] for p in panels)
    area = sum((p["h"] + 4) * (p["w"] + 4) for p in panels)
    extra = 5 * (len(panels) - 1)
    for _ in range(40):
        if params["scale"] == "large":
            height = int(rng.integers(max_h + 4, 31))
            width = int(rng.integers(max_w + 4, 31))
            factor = 1.6
        elif params["scale"] == "medium":
            height = int(rng.integers(max_h + 3, min(30, max_h + 9 + extra) + 1))
            width = int(rng.integers(max_w + 3, min(30, max_w + 10 + extra) + 1))
            factor = 1.4
        else:
            height = int(rng.integers(max_h + 2, min(30, max_h + 6 + extra) + 1))
            width = int(rng.integers(max_w + 2, min(30, max_w + 7 + extra) + 1))
            factor = 1.15
        if height <= 30 and width <= 30 and height * width >= factor * area:
            return height, width
    return None


def _effects(pos, key, shape):
    return _marker_effects(pos, key, shape)


def _demand_ok(name, pos, panel_index, panels, boxes, shape):
    key = panels[panel_index]["key"]
    painted, clipped, missing = _effects(pos, key, shape)
    if name == "four":
        return len(painted) == 4
    if name == "missing":
        return bool(painted) and bool(missing)
    if name == "clip":
        return bool(painted) and bool(clipped)
    if name == "compound":
        return bool(painted) and bool(clipped) and bool(missing)
    if name == "misbind":
        own = _box_distance(pos, boxes[panel_index])
        return any(_box_distance(pos, box)[0] < own[0] and _box_distance(pos, box)[1] < own[1]
                   for index, box in enumerate(boxes) if index != panel_index)
    return True


def _choose_marker(rng, candidates, taken, predicate, edge_rate, shape, corner_rate=0.0):
    height, width = shape
    free = [p for p in candidates
            if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= _MIN_MARKER_MARKER_DISTANCE for q in taken)
            and predicate(p)]
    if not free:
        return None
    edges = [p for p in free if p[0] in (0, height - 1) or p[1] in (0, width - 1)]
    corners = [p for p in edges if p[0] in (0, height - 1) and p[1] in (0, width - 1)]
    if edges and rng.random() < edge_rate:
        if corners and rng.random() < corner_rate:
            return _pick(rng, corners)
        return _pick(rng, edges)
    return _pick(rng, free)


def _build_pair(rng, params, background, n_panels, demands, forced_patterns, avoid_keys):
    """Return (input, output) satisfying the demands, or raise _Retry."""
    for _ in range(40):
        carriers = [int(c) for c in rng.choice([c for c in range(10) if c != background],
                                                size=n_panels, replace=False)]
        pool = [c for c in range(10) if c != background and c not in carriers]
        cap = _SCALE_CAPS[params["scale"]][min(n_panels, 3) - 1]
        panels = []
        for index in range(n_panels):
            pattern = forced_patterns.get(index) or _sample_pattern(rng)
            panel = _sample_panel(rng, params, _block_shape(rng, params), pattern, cap)
            _assign_key_colors(rng, panel, pool, params["repeat_rate"])
            panels.append(panel)
        keys = [_key_tuple(p["key"]) for p in panels]
        if len(set(keys)) != len(keys) or any(k in avoid_keys for k in keys):
            continue
        size = _canvas_size(rng, params, panels)
        if size is None:
            continue
        height, width = size
        boxes = _place_panels(rng, height, width, panels)
        if boxes is None:
            continue
        candidates = [(r, c) for r in range(height) for c in range(width)
                      if min(_box_distance((r, c), box)[0] for box in boxes) >= _MIN_MARKER_PANEL_DISTANCE]
        markers = []  # (pos, panel index)
        ok = True
        for name, panel_index in demands:
            def predicate(pos, name=name, panel_index=panel_index):
                return _demand_ok(name, pos, panel_index, panels, boxes, (height, width))
            pos = _choose_marker(rng, candidates, [m[0] for m in markers], predicate, 0.5, (height, width),
                                 params["corner_rate"])
            if pos is None:
                ok = False
                break
            markers.append((pos, panel_index))
        if not ok:
            continue
        for panel_index in range(n_panels):
            target = int(rng.integers(1, params["density"] + 1))
            have = sum(1 for _, p in markers if p == panel_index)
            for _ in range(max(0, target - have)):
                pos = _choose_marker(rng, candidates, [m[0] for m in markers], lambda p: True,
                                     params["edge_rate"], (height, width), params["corner_rate"])
                if pos is None:
                    break
                markers.append((pos, panel_index))
        if {p for _, p in markers} != set(range(n_panels)):
            continue

        grid = np.full((height, width), background, dtype=int)
        for panel, box, carrier in zip(panels, boxes, carriers):
            r0, r1, c0, c1 = box
            grid[r0:r1 + 1, c0:c1 + 1] = carrier
            bh, bw = panel["shape"]
            for i, j, _, _ in _DIRECTIONS:
                color = panel["key"][i][j]
                if color is not None:
                    top, left = r0 + panel["tops"][i], c0 + panel["lefts"][j]
                    grid[top:top + bh, left:left + bw] = color
        for (row, col), panel_index in markers:
            grid[row, col] = carriers[panel_index]
        output = grid.copy()
        for (row, col), panel_index in markers:
            output[row, col] = background
            key = panels[panel_index]["key"]
            for i, j, dr, dc in _DIRECTIONS:
                rr, cc = row + dr, col + dc
                if key[i][j] is not None and 0 <= rr < height and 0 <= cc < width:
                    output[rr, cc] = key[i][j]
        counts = sorted(np.bincount(grid.ravel(), minlength=10).tolist(), reverse=True)
        if counts[0] <= counts[1]:
            continue
        problems, _ = _pair_problems(grid, output)
        if problems:
            continue
        return grid, output
    raise _Retry("pair")


def _train_plan(rng, params, num_train):
    """Per training pair: panel count, demand list and forced patterns."""
    demands = []
    if params["show_four"]:
        demands.append("four")
    demands += ["missing", "clip"]
    multi = params["panel_mode"] != "single"
    if multi:
        demands.append("misbind")
    order = [int(i) for i in rng.permutation(num_train)]
    plans = [{"demands": [], "n_panels": 1} for _ in range(num_train)]
    for index, name in enumerate(demands):
        plans[order[index % num_train]]["demands"].append(name)
    for plan in plans:
        if params["panel_mode"] == "single":
            plan["n_panels"] = 1
        elif params["panel_mode"] == "multi":
            plan["n_panels"] = 3 if rng.random() < 0.25 else 2
        else:
            plan["n_panels"] = 2 if rng.random() < 0.4 else 1
        if "misbind" in plan["demands"]:
            plan["n_panels"] = max(plan["n_panels"], 2)
    if params["panel_mode"] == "mixed" and num_train >= 3 and all(p["n_panels"] > 1 for p in plans):
        for plan in plans:
            if "misbind" not in plan["demands"]:
                plan["n_panels"] = 1
                break
    return plans


def _materialize_demands(rng, plan, used_patterns, test=False):
    """Map demand names to panel indices and forced key patterns."""
    forced, assigned = {}, []
    n_panels = plan["n_panels"]
    free_panels = list(range(n_panels))
    for name in plan["demands"]:
        if name == "four":
            index = free_panels.pop(0) if free_panels else 0
            forced[index] = "full"
        elif name in ("missing", "compound"):
            index = free_panels.pop(0) if free_panels else 0
            if forced.get(index) == "full":
                index = (index + 1) % n_panels
            options = [p for p in _ORDER_SENSITIVE if p not in used_patterns] if test else []
            forced[index] = _pick(rng, options) if options else _sample_pattern(rng, _ORDER_SENSITIVE)
        elif name == "clip":
            index = int(rng.integers(n_panels))
        elif name == "misbind":
            index = int(rng.integers(n_panels))
        else:
            continue
        assigned.append((name, index))
    return assigned, forced


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(60):
        params = _sample_params(rng)
        try:
            train, test = [], []
            train_keys, used_patterns, seen = set(), set(), set()
            for plan in _train_plan(rng, params, num_train):
                background = (params["background"] if params["background_mode"] == "stable"
                              else _sample_background(rng))
                demands, forced = _materialize_demands(rng, plan, used_patterns)
                inp, out = _build_pair(rng, params, background, plan["n_panels"], demands, forced, set())
                analysis = _analyze(inp)
                for panel in analysis["panels"]:
                    train_keys.add(_key_tuple(panel["key"]))
                    used_patterns.add(_pattern_name(panel["key"]))
                train.append({"input": inp, "output": out})
            train_counts = {len(_analyze(p["input"])["panels"]) for p in train}
            train_shapes = {(b["h"], b["w"]) for p in train
                            for panel in _analyze(p["input"])["panels"] for b in panel["blocks"][:1]}
            for t_index in range(num_test):
                background = (params["background"] if params["background_mode"] == "stable"
                              else _sample_background(rng))
                if params["panel_mode"] == "single":
                    n_panels = 1
                elif params["panel_mode"] == "multi":
                    n_panels = 3 if (3 in train_counts and rng.random() < 0.4) else 2
                else:
                    n_panels = 2 if rng.random() < 0.5 else 1
                plan = {"n_panels": n_panels, "demands": []}
                if t_index == 0:
                    plan["demands"].append("compound")
                elif rng.random() < 0.5:
                    plan["demands"].append("clip")
                if n_panels > 1 and (t_index == 0 or rng.random() < 0.5):
                    plan["demands"].append("misbind")
                demands, forced = _materialize_demands(rng, plan, used_patterns, test=(t_index == 0))
                local = dict(params)
                local["size_mode"] = "mixed"
                local["mixed_shapes"] = tuple(sorted(train_shapes))
                inp, out = _build_pair(rng, local, background, n_panels, demands, forced, train_keys)
                test.append({"input": inp, "output": out})
            for pair in train + test:
                key = (pair["input"].shape, pair["input"].tobytes())
                if key in seen:
                    raise _Retry("duplicate")
                seen.add(key)
            puzzle = {"train": train, "test": test}
            if _puzzle_problems(puzzle):
                continue
            return puzzle
        except _Retry:
            continue
    raise RuntimeError(f"could not generate a valid puzzle for seed {seed}")


# ---------------------------------------------------------------------------
# Self test.
# ---------------------------------------------------------------------------

def self_test() -> None:
    import collections

    stats = collections.Counter()
    canvas_shapes, block_shapes, patterns, panel_counts = set(), set(), set(), set()
    backgrounds_vary = 0
    refuted = collections.Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        again = generate_puzzle(seed)
        for split in ("train", "test"):
            for a, b in zip(puzzle[split], again[split]):
                assert np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        backgrounds = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(solve(inp), out)
                analysis = _analyze(inp)
                canvas_shapes.add(inp.shape)
                backgrounds.add(analysis["background"])
                panel_counts.add(len(analysis["panels"]))
                for panel in analysis["panels"]:
                    block_shapes.add((panel["blocks"][0]["h"], panel["blocks"][0]["w"]))
                    patterns.add(_pattern_name(panel["key"]))
                facts = _pair_facts(analysis)
                stats[f"{split}_compound"] += facts["compound"]
                stats[f"{split}_misbind"] += facts["misbind"]
                stats[f"{split}_pairs"] += 1
        backgrounds_vary += len(backgrounds) > 1
        first_key = _analyze(puzzle["train"][0]["input"])["panels"][0]["key"]
        for pair in puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            if not np.array_equal(_shortcut_reading_order(inp), out):
                refuted["reading_order"] += 1
            if not np.array_equal(_shortcut_fixed_key(inp, first_key), out):
                refuted["fixed_key"] += 1
            if not np.array_equal(_shortcut_quadrant(inp), out):
                refuted["quadrant"] += 1
            if len(_analyze(inp)["panels"]) > 1 and not np.array_equal(_shortcut_nearest_panel(inp), out):
                refuted["nearest"] += 1
    assert len(canvas_shapes) >= 30, canvas_shapes
    assert len(block_shapes) >= 6, block_shapes
    assert patterns == set(_PATTERNS), patterns
    assert panel_counts >= {1, 2, 3}, panel_counts
    assert backgrounds_vary >= 10, backgrounds_vary
    assert refuted["reading_order"] >= 42 and refuted["fixed_key"] >= 42, refuted
    assert stats["test_compound"] >= 42 and stats["train_misbind"] >= 10, stats

    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")
    for num_train in range(2, 11):
        for num_test in range(1, min(4, num_train - 1) + 1):
            puzzle = generate_puzzle(7, num_train, num_test)
            assert len(puzzle["train"]) == num_train and len(puzzle["test"]) == num_test
            validate_generated_puzzle(puzzle)


if __name__ == "__main__":
    self_test()
    print("self_test passed")
