"""Ported cores: a lane's beam is absorbed by a same-colored marker and pushes a foreign marker to the wall.

Composition of parent A (7d7772cc) and parent B (9def23fe), revised after review.

* Parent A contributes the aligned, lane-wise identity comparison across a structural boundary:
  a movable symbol is compared with the stationary reference symbol of its own lane; equality
  sends it to the cells just outside the boundary, inequality sends it to the outer grid edge.
* Parent B contributes the solid core that projects one-cell-wide rays from its rows and columns to
  the grid boundary, independently per directional lane, where a marker anywhere in the lane can
  cancel the whole ray.
* Connecting concepts: the stationary references sit on the core's outer ring as colored ports.
  Each port is both A's comparison key and B's emitter, so the beam has the port's color.  One
  per-lane comparison decides everything: a marker with the port's color absorbs the beam (no beam
  anywhere in the lane) and docks against its port; a marker of any other color does not stop the
  beam, which fills the lane and pins the marker at the grid wall.  Body-colored ring cells are not
  ports; their lanes are inert, and anything loose in them stays where it is.

Revision (see review.md / review_coverage.md): markers may be straight bars of 1..3 cells along
their lane (moved rigidly), a grid may hold two cores whose lanes never cross, cores may be large,
elongated, flush with one wall or span the grid like A's barrier, inert cells may also sit in
silent lanes, corner-row extensions or as small shapes, and the demonstrations now also rule out
"same side" / "neighbouring port" / "opposite port" comparisons.

Only NumPy and the standard library are used.  ``solve`` reads everything from its input grid.
"""
from __future__ import annotations

import numpy as np

SIDES = ("up", "down", "left", "right")
AXIS = {"up": "vertical", "down": "vertical", "left": "horizontal", "right": "horizontal"}
AXES = ("vertical", "horizontal")
AXIS_SIDES = {"vertical": ("up", "down"), "horizontal": ("left", "right")}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
MIN_SIDE, MAX_SIDE = 5, 30
MAX_BAR = 3
BANDS = {"small": (5, 13), "medium": (12, 21), "large": (19, 30)}


# ----------------------------------------------------------------------------------------------
# Shared geometry (a core is (top, left, bottom, right))
# ----------------------------------------------------------------------------------------------

def _lane_indices(side, core):
    """Rows or columns of the core that own a lane on ``side`` (corner rows/columns never do)."""
    top, left, bottom, right = core
    if side in ("up", "down"):
        return list(range(left + 1, right))
    return list(range(top + 1, bottom))


def _ring_cell(side, index, core):
    top, left, bottom, right = core
    if side == "up":
        return (top, index)
    if side == "down":
        return (bottom, index)
    if side == "left":
        return (index, left)
    return (index, right)


def _lane_cells(side, index, core, shape):
    """Lane cells ordered outward: first the dock cell next to the core, last the wall cell."""
    top, left, bottom, right = core
    height, width = shape
    if side == "up":
        return [(r, index) for r in range(top - 1, -1, -1)]
    if side == "down":
        return [(r, index) for r in range(bottom + 1, height)]
    if side == "left":
        return [(index, c) for c in range(left - 1, -1, -1)]
    return [(index, c) for c in range(right + 1, width)]


def _components(mask):
    """8-connected components of a boolean mask, as lists of cells."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    found = []
    for r0, c0 in zip(*np.nonzero(mask)):
        r0, c0 = int(r0), int(c0)
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        stack, cells = [(r0, c0)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        found.append(cells)
    return found


# ----------------------------------------------------------------------------------------------
# Input-only solver
# ----------------------------------------------------------------------------------------------

def _find_cores(grid, background):
    """Solid blocks (>=3x3) whose corners and interior share one color: (body, top, left, bottom, right)."""
    found = []
    for value in np.unique(grid):
        color = int(value)
        if color == background:
            continue
        for cells in _components(grid == color):
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
            if bottom - top < 2 or right - left < 2:
                continue
            box = grid[top:bottom + 1, left:right + 1]
            if np.any(box == background):
                continue
            if not (box[0, 0] == color and box[0, -1] == color and box[-1, 0] == color and box[-1, -1] == color):
                continue
            if not np.all(box[1:-1, 1:-1] == color):
                continue
            found.append((color, top, left, bottom, right))
    return sorted(found, key=lambda item: (item[1], item[2]))


def _resolve_lane(cells, port, values, background):
    """Return the output colors of one port lane given its input colors."""
    loose = [i for i, v in enumerate(values) if v != background]
    result = [background] * len(cells)
    if not loose:
        return [port] * len(cells)
    color, size = values[loose[0]], len(loose)
    if color == port:  # absorbed: no beam anywhere, the marker docks against its port
        for i in range(size):
            result[i] = color
    else:  # pushed: the beam fills the lane and pins the marker at the wall
        for i in range(len(cells) - size):
            result[i] = port
        for i in range(len(cells) - size, len(cells)):
            result[i] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.array(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("input grid must be two-dimensional")
    background = int(grid[0, 0])
    cores = _find_cores(grid, background)
    if not cores:
        raise ValueError("no ported core found")
    out = grid.copy()
    for body, top, left, bottom, right in cores:
        core = (top, left, bottom, right)
        for side in SIDES:
            for index in _lane_indices(side, core):
                port = int(grid[_ring_cell(side, index, core)])
                if port == body:
                    continue  # body-colored ring cell: no emitter, the lane is inert
                cells = _lane_cells(side, index, core, grid.shape)
                if not cells:
                    continue
                values = [int(grid[cell]) for cell in cells]
                for cell, value in zip(cells, _resolve_lane(cells, port, values, background)):
                    out[cell] = value
    return out


# ----------------------------------------------------------------------------------------------
# Structural analysis used by generation, validation and self-test
# ----------------------------------------------------------------------------------------------

def _analyze(input_grid, output_grid=None):
    """Parse one pair, check the family conventions, and describe its cores, lanes and inert cells.

    The expected output is rebuilt lane by lane here (independently of ``solve``'s loop) and
    compared with ``output_grid`` when one is supplied.
    """
    grid = np.asarray(input_grid).astype(int)
    assert grid.ndim == 2, "grid must be 2-D"
    height, width = grid.shape
    assert MIN_SIDE <= height <= MAX_SIDE and MIN_SIDE <= width <= MAX_SIDE, "grid size outside family range"
    background = int(grid[0, 0])
    corners = ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1))
    for corner in corners:
        assert int(grid[corner]) == background, "grid corners must show the background"
    found = _find_cores(grid, background)
    assert 1 <= len(found) <= 2, "one or two ported cores"
    cores = [(t, l, b, r) for _, t, l, b, r in found]
    bodies = [body for body, *_ in found]
    in_core = np.zeros(grid.shape, dtype=bool)
    for top, left, bottom, right in cores:
        assert not in_core[top:bottom + 1, left:right + 1].any(), "cores never overlap"
        in_core[top:bottom + 1, left:right + 1] = True
        for corner in corners:
            assert not (top <= corner[0] <= bottom and left <= corner[1] <= right), "cores never cover a grid corner"
    if len(cores) == 2:
        (t0, l0, b0, r0), (t1, l1, b1, r1) = cores
        assert b0 + 1 < t1 or b1 + 1 < t0, "two cores use separated row bands"
        assert r0 + 1 < l1 or r1 + 1 < l0, "two cores use separated column bands"
    for body in set(bodies):
        inside = sum(int(np.sum(grid[t:b + 1, l:r + 1] == body)) for (t, l, b, r), bd in zip(cores, bodies) if bd == body)
        assert int(np.sum(grid == body)) == inside, "body colors occur only inside their cores"
    expected = grid.copy()
    port_lane = {}
    lanes, silent_lanes = [], {}
    in_side_silent = 0
    flush_sides = 0
    ports_by_key = {}
    for ci, (core, body) in enumerate(zip(cores, bodies)):
        for side in SIDES:
            indices = _lane_indices(side, core)
            if not _lane_cells(side, indices[0], core, grid.shape):
                flush_sides += 1
                for index in indices:
                    assert int(grid[_ring_cell(side, index, core)]) == body, "a side flush with the wall has no ports"
                continue
            side_ports = [index for index in indices if int(grid[_ring_cell(side, index, core)]) != body]
            if side_ports:
                in_side_silent += len(indices) - len(side_ports)
            for index in indices:
                port = int(grid[_ring_cell(side, index, core)])
                cells = _lane_cells(side, index, core, grid.shape)
                if port == body:
                    for cell in cells:
                        silent_lanes.setdefault(cell, (ci, side, index))
                    continue
                assert port != background and port not in bodies, "ports are never background or body colored"
                ports_by_key[(ci, side, index)] = port
                for cell in cells:
                    assert not in_core[cell], "a port lane never enters a core"
                    assert cell not in port_lane, "port lanes never cross"
                    port_lane[cell] = len(lanes)
                lanes.append({"core": ci, "side": side, "axis": AXIS[side], "index": index, "port": port,
                              "length": len(cells), "cells": cells})
    assert lanes, "at least one port"
    loose_owner = {}
    for li, lane in enumerate(lanes):
        cells = lane["cells"]
        values = [int(grid[cell]) for cell in cells]
        loose = [i for i, v in enumerate(values) if v != background]
        lane.update(marker=None, start=None, size=0, kind="open")
        if loose:
            size = len(loose)
            color = values[loose[0]]
            assert loose == list(range(loose[0], loose[0] + size)), "a lane holds one contiguous marker"
            assert all(values[i] == color for i in loose), "a marker has one color"
            assert color not in bodies, "marker colors differ from body colors"
            assert size <= MAX_BAR, "markers are at most three cells long"
            assert len(cells) >= (2 if size == 1 else size + 2), "a marker leaves room to move"
            lane.update(marker=color, start=loose[0] + 1, size=size,
                        kind="match" if color == lane["port"] else "mismatch")
            for i in loose:
                loose_owner[cells[i]] = li
        # Independent rebuild of the lane (does not reuse the solver's helper).
        length, port = len(cells), lane["port"]
        if lane["kind"] == "open":
            for cell in cells:
                expected[cell] = port
        else:
            k, color = lane["size"], lane["marker"]
            for i in loose:
                expected[cells[i]] = background
            destination = range(k) if lane["kind"] == "match" else range(length - k, length)
            if lane["kind"] == "mismatch":
                for i in range(length - k):
                    expected[cells[i]] = port
            for i in destination:
                expected[cells[i]] = color
    # Same-colored markers of different lanes never touch side by side (each marker is one object).
    for cell, li in loose_owner.items():
        for dr, dc in ((0, 1), (1, 0)):
            other = (cell[0] + dr, cell[1] + dc)
            if other in loose_owner and loose_owner[other] != li:
                assert int(grid[cell]) != int(grid[other]), "same-colored markers of two lanes never touch"
    port_colors = [lane["port"] for lane in lanes]
    for lane in lanes:
        length, start, size = lane["length"], lane["start"], lane["size"]
        others = set(port_colors) - {lane["port"]}
        pushed = lane["kind"] == "mismatch"
        lane["inner"] = pushed and start + size - 1 < length
        lane["far"] = lane["kind"] == "match" and start >= 2
        lane["cross"] = pushed and lane["marker"] in others
        neighbours = {ports_by_key.get((lane["core"], lane["side"], lane["index"] + d)) for d in (-1, 1)}
        lane["adjacent"] = pushed and lane["marker"] in neighbours - {None}
        lane["opposite"] = pushed and ports_by_key.get((lane["core"], OPPOSITE[lane["side"]], lane["index"])) == lane["marker"]
        lane["noop"] = ((lane["kind"] == "match" and start == 1)
                        or (pushed and start + size - 1 == length))
        del lane["cells"]
    # Everything else is inert: it must be background, or an isolated small object off every port lane.
    loose_mask = np.zeros(grid.shape, dtype=bool)
    for r in range(height):
        for c in range(width):
            if not in_core[r, c] and (r, c) not in port_lane and int(grid[r, c]) != background:
                loose_mask[r, c] = True
    extension = set()
    for top, left, bottom, right in cores:
        for r in (top, bottom):
            extension.update((r, c) for c in range(width) if not left <= c <= right)
        for c in (left, right):
            extension.update((r, c) for r in range(height) if not top <= r <= bottom)
    inert = []
    for cells in _components(loose_mask):
        colors = {int(grid[cell]) for cell in cells}
        assert len(colors) == 1 and len(cells) <= 3, "inert objects are small and single-colored"
        color = colors.pop()
        assert color not in bodies, "inert colors differ from body colors"
        for r, c in cells:
            for top, left, bottom, right in cores:
                assert not (top - 1 <= r <= bottom + 1 and left - 1 <= c <= right + 1), "inert cells never touch a core"
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    assert (r + dr, c + dc) not in loose_owner, "inert cells never touch a marker"
        if all(cell in silent_lanes for cell in cells):
            category = "silent"
        elif all(cell in extension for cell in cells):
            category = "extension"
        else:
            category = "quadrant"
        if len(cells) > 1:
            assert category == "quadrant" and not any(cell in silent_lanes or cell in extension for cell in cells), \
                "inert shapes sit in corner regions"
        inert.append({"cells": sorted(cells), "color": color, "category": category})
    if output_grid is not None:
        out = np.asarray(output_grid)
        assert out.shape == grid.shape, "output keeps the input size"
        assert np.array_equal(out, expected), "output does not follow the ported-core rule"
        assert not np.array_equal(out, grid), "pair must not be an identity"
    return {"background": background, "bodies": bodies, "cores": cores, "shape": (height, width),
            "lanes": lanes, "inert": inert, "in_side_silent": in_side_silent, "flush_sides": flush_sides,
            "sides": sorted({lane["side"] for lane in lanes}, key=SIDES.index),
            "colors": sorted({int(v) for v in np.unique(grid)})}


def _has(info, kind, *, axis=None, far=False, inner=False, cross=False, adjacent=False,
         opposite=False, bar=False):
    for lane in info["lanes"]:
        if lane["kind"] != kind or (axis and lane["axis"] != axis):
            continue
        if far and not lane["far"]:
            continue
        if inner and not lane["inner"]:
            continue
        if cross and not lane["cross"]:
            continue
        if adjacent and not lane["adjacent"]:
            continue
        if opposite and not lane["opposite"]:
            continue
        if bar and lane["size"] < 2:
            continue
        return True
    return False


def _features(info):
    return {"bars": any(lane["size"] >= 2 for lane in info["lanes"]),
            "twin": len(info["cores"]) == 2,
            "inert": bool(info["inert"]),
            "silent_inert": any(obj["category"] == "silent" for obj in info["inert"])}


def _item_ok(info, item):
    """Whether a pair actually shows one demonstration item (measured from the grids)."""
    if ":" in item:
        role, axis = item.split(":")
        if role == "open":
            return _has(info, "open", axis=axis)
        if role == "far":
            return _has(info, "match", axis=axis, far=True)
        return _has(info, "mismatch", axis=axis, inner=True)
    if item == "adjacent":
        return _has(info, "mismatch", inner=True, adjacent=True)
    if item == "opposite":
        return _has(info, "mismatch", opposite=True)
    if item == "cross":
        return _has(info, "mismatch", inner=True, cross=True)
    if item == "gap":
        return info["in_side_silent"] > 0
    if item == "bar_match":
        return _has(info, "match", far=True, bar=True)
    if item == "bar_push":
        return _has(info, "mismatch", inner=True, bar=True)
    if item == "open":
        return _has(info, "open")
    if item == "far":
        return _has(info, "match", far=True)
    if item == "noop":
        return any(lane["noop"] for lane in info["lanes"])
    return _features(info).get(item, True)


TRAIN_ITEMS = ("open:vertical", "open:horizontal", "far:vertical", "far:horizontal",
               "push:vertical", "push:horizontal", "cross", "adjacent", "opposite", "gap")


def _training_problems(infos, test_infos=()):
    """Coverage the demonstrations must provide; returns a list of missing items."""
    problems = []
    sides = set()
    for info in infos:
        sides.update(info["sides"])
    if sides != set(SIDES):
        problems.append("training does not show ports on all four sides")
    for item in TRAIN_ITEMS:
        if not any(_item_ok(info, item) for info in infos):
            problems.append(f"training lacks {item}")
    test_features = [_features(info) for info in test_infos]
    if any(f["bars"] for f in test_features):
        for item in ("bar_match", "bar_push"):
            if not any(_item_ok(info, item) for info in infos):
                problems.append(f"tests use bars but training lacks {item}")
    for feature in ("twin", "inert", "silent_inert"):
        if any(f[feature] for f in test_features) and not any(_features(info)[feature] for info in infos):
            problems.append(f"tests use {feature} without a training demonstration")
    return problems


def _test_problems(info):
    return [f"test lacks {item}" for item in ("open", "far", "cross") if not _item_ok(info, item)]


# ----------------------------------------------------------------------------------------------
# Generation helpers
# ----------------------------------------------------------------------------------------------

def _validate_counts(num_train, num_test):
    values = []
    for name, value, default in (("num_train", num_train, 4), ("num_test", num_test, 1)):
        if value is None:
            value = default
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        values.append(int(value))
    num_train, num_test = values
    if not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _draw(rng, values, weights=None):
    values = list(values)
    if weights is None:
        return values[int(rng.integers(len(values)))]
    p = np.asarray(weights, dtype=float)
    return values[int(rng.choice(len(values), p=p / p.sum()))]


def _split(rng, extra, need_a, need_b):
    """Margin on side a; the rest goes to side b.  Mostly spread out, sometimes hugging a side."""
    low, high = need_a, extra - need_b
    if high < low:
        return None
    u = rng.random()
    if u < 0.1:
        return low
    if u < 0.2:
        return high
    return int(rng.integers(low, high + 1))


def _axis_fit(rng, core, need_a, need_b, band):
    low, high = BANDS[band]
    minimum = core + need_a + need_b
    low = max(low, minimum, MIN_SIDE)
    high = min(MAX_SIDE, max(high, low))
    if low > MAX_SIDE:
        return None
    total = int(rng.integers(low, high + 1))
    a = _split(rng, total - core, need_a, need_b)
    return None if a is None else (total, a)


def _active_need(rng):
    """Minimum margin on a side with ports: mostly 3+, sometimes 2, rarely 1 (one-cell beams only)."""
    u = rng.random()
    return 1 if u < 0.07 else (2 if u < 0.5 else 3)


def _geometry_options(sides):
    options = ["classic", "big", "bar"]
    if len(sides) < 4:
        options.append("flush")
    if len({AXIS[s] for s in sides}) == 1:
        options.append("barrier")
    return options


def _single_layout(rng, sides, kind, band):
    need = {s: (_active_need(rng) if s in sides else 1) for s in SIDES}
    if kind == "flush":
        need[_draw(rng, [s for s in SIDES if s not in sides])] = 0
    if kind in ("classic", "flush"):
        cap = 5 if band == "small" else 10
        h, w = int(rng.integers(3, cap + 1)), int(rng.integers(3, cap + 1))
        vertical = _axis_fit(rng, h, need["up"], need["down"], band)
        horizontal = _axis_fit(rng, w, need["left"], need["right"], band)
        if vertical is None or horizontal is None:
            return None
        (height, top), (width, left) = vertical, horizontal
    elif kind == "big":
        low, high = BANDS["large" if band == "large" else "medium"]
        height = int(rng.integers(max(low, 14), high + 1))
        width = int(rng.integers(max(low, 14), high + 1))
        margins = {s: int(rng.integers(need[s], need[s] + (5 if s in sides else 3))) for s in SIDES}
        h = height - margins["up"] - margins["down"]
        w = width - margins["left"] - margins["right"]
        if min(h, w) < 3 or max(h, w) < 11:
            return None
        top, left = margins["up"], margins["left"]
    elif kind == "bar":
        long_side, short_side = int(rng.integers(8 if band == "small" else 10, 25)), int(rng.integers(3, 5))
        if rng.random() < 0.5:
            h, w = short_side, long_side
        else:
            h, w = long_side, short_side
        across = "medium" if band == "small" else band
        along = "medium" if band == "small" else "large"
        vertical = _axis_fit(rng, h, need["up"], need["down"], across if h < 6 else along)
        horizontal = _axis_fit(rng, w, need["left"], need["right"], across if w < 6 else along)
        if vertical is None or horizontal is None:
            return None
        (height, top), (width, left) = vertical, horizontal
    else:  # barrier: the core spans the grid between two opposite walls, like A's straight barrier
        axis = AXIS[sides[0]]
        low, high = BANDS[band]
        span = int(rng.integers(max(low, 7), max(high, 7) + 1))
        thick = int(rng.integers(3, 9))
        if axis == "horizontal":  # lanes run left/right; the core spans every row
            fit = _axis_fit(rng, thick, need["left"], need["right"], band)
            if fit is None:
                return None
            (width, left), height, top, h, w = fit, span, 0, span, thick
        else:
            fit = _axis_fit(rng, thick, need["up"], need["down"], band)
            if fit is None:
                return None
            (height, top), width, left, h, w = fit, span, 0, thick, span
    if need["up"] == 0:
        top = 0
    elif need["down"] == 0:
        top = height - h
    if need["left"] == 0:
        left = 0
    elif need["right"] == 0:
        left = width - w
    core = (top, left, top + h - 1, left + w - 1)
    if core[2] > height - 1 or core[3] > width - 1 or not (MIN_SIDE <= height <= MAX_SIDE and MIN_SIDE <= width <= MAX_SIDE):
        return None
    return {"height": height, "width": width, "cores": [core], "active": [list(sides)], "kind": kind}


def _partition(rng, extra, parts):
    cuts = sorted(int(v) for v in rng.integers(0, extra + 1, size=parts - 1))
    bounds = [0] + cuts + [extra]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _twin_layout(rng, sides0, sides1, band):
    """Two cores in diagonal placement with separated row and column bands."""
    band = "medium" if band == "small" else band
    low, high = BANDS[band]
    h0, w0, h1, w1 = (int(v) for v in rng.integers(3, 8, size=4))
    main = rng.random() < 0.5  # main: core 0 upper-left / core 1 lower-right; else upper-right / lower-left
    need_top = _active_need(rng) if "up" in sides0 else 1
    need_bottom = _active_need(rng) if "down" in sides1 else 1
    rows_min = need_top + h0 + 1 + h1 + need_bottom
    if main:
        need_left = _active_need(rng) if "left" in sides0 else 1
        need_right = _active_need(rng) if "right" in sides1 else 1
        first_w, second_w = w0, w1
    else:
        need_left = _active_need(rng) if "left" in sides1 else 1
        need_right = _active_need(rng) if "right" in sides0 else 1
        first_w, second_w = w1, w0
    cols_min = need_left + first_w + 1 + second_w + need_right
    if rows_min > MAX_SIDE or cols_min > MAX_SIDE:
        return None
    height = int(rng.integers(max(low, rows_min), max(min(MAX_SIDE, high), rows_min) + 1))
    width = int(rng.integers(max(low, cols_min), max(min(MAX_SIDE, high), cols_min) + 1))
    a, g, _ = _partition(rng, height - rows_min, 3)
    top0 = need_top + a
    top1 = top0 + h0 + 1 + g
    a, g, _ = _partition(rng, width - cols_min, 3)
    first_left = need_left + a
    second_left = first_left + first_w + 1 + g
    if main:
        left0, left1 = first_left, second_left
    else:
        left1, left0 = first_left, second_left
    cores = [(top0, left0, top0 + h0 - 1, left0 + w0 - 1), (top1, left1, top1 + h1 - 1, left1 + w1 - 1)]
    return {"height": height, "width": width, "cores": cores, "active": [list(sides0), list(sides1)], "kind": "twin"}


def _side_cells(core, side, shape):
    cells = set()
    for index in _lane_indices(side, core):
        cells.update(_lane_cells(side, index, core, shape))
    return cells


def _resolve_twin_conflicts(rng, layout):
    """Drop sides of the second core whose lanes would cross the first core's lanes."""
    shape = (layout["height"], layout["width"])
    core0, core1 = layout["cores"]
    active0, active1 = layout["active"]
    cells0 = set()
    for side in active0:
        cells0 |= _side_cells(core0, side, shape)
    keep = [s for s in active1 if not (_side_cells(core1, s, shape) & cells0)]
    if not keep:
        spare = [s for s in SIDES if not (_side_cells(core1, s, shape) & cells0)
                 and _lane_cells(s, _lane_indices(s, core1)[0], core1, shape)]
        if not spare:
            return False
        keep = [_draw(rng, spare)]
    layout["active"][1] = sorted(keep, key=SIDES.index)
    return True


def _choose_ports(rng, indices, density, force_all=False):
    n = len(indices)
    style = "all" if force_all else _draw(rng, ("all", "alternate", "random"), density)
    if style == "all" or n == 1:
        return list(indices)
    if style == "alternate":
        chosen = list(indices[int(rng.integers(2))::2])
        return chosen or list(indices[:1])
    k = int(rng.integers(1, n + 1))
    return sorted(int(v) for v in rng.choice(indices, size=k, replace=False))


def _pair_palette(rng, style, kind, twin, forbidden=()):
    if style["palette_mode"] == "fixed":
        background, body, body2 = style["fixed"]
    else:
        background = 0 if rng.random() < 0.4 else int(rng.integers(10))
        body = int(_draw(rng, [c for c in range(10) if c != background]))
        body2 = body if rng.random() < 0.4 else int(_draw(rng, [c for c in range(10) if c not in (background, body)]))
    bodies = [body, body2] if twin else [body]
    rest = [c for c in range(10) if c != background and c not in bodies and c not in forbidden]
    low = 1 if kind == "plain" else 2
    count = int(rng.integers(low, min(5, len(rest) - 1) + 1))
    ports = [int(c) for c in rng.choice(rest, size=count, replace=False)]
    rest = [c for c in rest if c not in ports]
    foreign_count = int(rng.integers(1 if count == 1 else 0, min(2, len(rest)) + 1))
    foreign = [int(c) for c in rng.choice(rest, size=foreign_count, replace=False)] if foreign_count else []
    return {"background": background, "bodies": bodies, "ports": ports, "foreign": foreign}


def _bar_size(rng, length, bars, force=False):
    options = [k for k in (2, 3) if length >= k + 2]
    if force:
        return int(_draw(rng, options)) if options else None
    if not bars or not options or rng.random() < 0.5:
        return 1
    return int(_draw(rng, options, [2, 1][:len(options)]))


def _match_start(rng, length, size, far=None):
    """1-based depth of the marker's near end (1 = docked)."""
    if far is None:
        far = rng.random() < 0.8
    if far and length - size + 1 >= 2:
        return int(rng.integers(2, length - size + 2))
    return 1


def _push_start(rng, length, size, inner=None):
    if inner is None:
        inner = rng.random() < 0.8
    if inner and length - size >= 1:
        return int(rng.integers(1, length - size + 1))
    return length - size + 1


def _place_inert(rng, grid, cores, bodies, port_cells, marker_cells, colors, kinds, need):
    """Place small inert objects off every port lane; returns the categories placed or None."""
    height, width = grid.shape
    background = int(grid[0, 0])
    blocked = np.zeros(grid.shape, dtype=bool)
    for top, left, bottom, right in cores:
        blocked[max(0, top - 1):bottom + 2, max(0, left - 1):right + 2] = True
    for r, c in port_cells:
        blocked[r, c] = True
    for r, c in marker_cells:
        blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    for r, c in ((0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)):
        blocked[r, c] = True
    category = {}
    for (top, left, bottom, right), body in zip(cores, bodies):
        core = (top, left, bottom, right)
        for r in (top, bottom):
            for c in range(width):
                if not left <= c <= right:
                    category.setdefault((r, c), "extension")
        for c in (left, right):
            for r in range(height):
                if not top <= r <= bottom:
                    category.setdefault((r, c), "extension")
        for side in SIDES:
            for index in _lane_indices(side, core):
                if int(grid[_ring_cell(side, index, core)]) != body:
                    continue
                cells = _lane_cells(side, index, core, grid.shape)
                for depth, cell in enumerate(cells, 1):
                    if 2 <= depth < len(cells):
                        category[cell] = "silent"
                    else:
                        category.setdefault(cell, "lane-end")
    aligned = np.zeros(grid.shape, dtype=bool)
    for top, left, bottom, right in cores:
        aligned[top:bottom + 1, :] = True
        aligned[:, left:right + 1] = True

    def free(cell):
        r, c = cell
        return 0 <= r < height and 0 <= c < width and not blocked[r, c] and int(grid[r, c]) == background

    pools = {"silent": [], "extension": [], "quadrant": []}
    for r in range(height):
        for c in range(width):
            if not free((r, c)):
                continue
            kind = category.get((r, c))
            if kind in ("silent", "extension"):
                pools[kind].append((r, c))
            elif kind is None and not aligned[r, c]:
                pools["quadrant"].append((r, c))
    wanted = list(need)
    for _ in range(int(rng.integers(1, 4))):
        wanted.append(_draw(rng, kinds))
    placed = []
    for kind in wanted:
        shape_kind = kind == "shape"
        pool = pools["quadrant" if shape_kind else kind]
        order = rng.permutation(len(pool)) if pool else []
        done = False
        for i in order:
            anchor = pool[int(i)]
            cells = [anchor]
            if shape_kind:
                dr, dc = _draw(rng, ((0, 1), (1, 0)))
                cells.append((anchor[0] + dr, anchor[1] + dc))
                if rng.random() < 0.5:
                    er, ec = (1, 0) if dc else (0, 1)
                    cells.append((cells[-1][0] + er, cells[-1][1] + ec))
                if not all(free(cell) and cell in pools["quadrant"] for cell in cells):
                    continue
            elif not free(anchor):
                continue
            color = int(_draw(rng, colors))
            for r, c in cells:
                grid[r, c] = color
            for r, c in cells:
                blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
            placed.append(kind)
            done = True
            break
        if not done and kind in need:
            return None
    return placed


# ----------------------------------------------------------------------------------------------
# Pair construction
# ----------------------------------------------------------------------------------------------

def _make_pair(rng, style, plan):
    """Build one pair following ``plan``; every requested item is verified on the returned grids."""
    palette = plan["palette"]
    background = palette["background"]
    for _ in range(60):
        band = _draw(rng, ("small", "medium", "large"), style["bands"])
        if plan["twin"]:
            other = sorted(int(i) for i in rng.choice(4, size=int(_draw(rng, (1, 2, 3), (3, 3, 1))), replace=False))
            layout = _twin_layout(rng, plan["sides"], [SIDES[i] for i in other], band)
            if layout is None or not _resolve_twin_conflicts(rng, layout):
                continue
        else:
            options = [k for k in _geometry_options(plan["sides"]) if style["geometry"].get(k, 0) > 0]
            kind = plan.get("geometry")
            if kind not in options:
                kind = _draw(rng, options, [style["geometry"][k] for k in options])
            layout = _single_layout(rng, plan["sides"], kind, band)
            if layout is None:
                continue
        height, width = layout["height"], layout["width"]
        shape = (height, width)
        cores = layout["cores"]
        bodies = palette["bodies"][:len(cores)]
        grid = np.full(shape, background, dtype=int)
        for (top, left, bottom, right), body in zip(cores, bodies):
            grid[top:bottom + 1, left:right + 1] = body
        lanes = []
        port_draw = palette["ports"]

        def new_lane(ci, side, index):
            return {"core": ci, "side": side, "axis": AXIS[side], "index": index,
                    "cells": _lane_cells(side, index, cores[ci], shape), "port": int(_draw(rng, port_draw)),
                    "plan": None, "keep": False, "fixed": False}

        for ci, core in enumerate(cores):
            for side in layout["active"][ci]:
                indices = _lane_indices(side, core)
                if not _lane_cells(side, indices[0], core, shape):
                    continue
                force = plan.get("dense", False) and ci == 0
                for index in _choose_ports(rng, indices, style["density"], force_all=force):
                    lanes.append(new_lane(ci, side, index))
        if not lanes or any(not any(l["core"] == ci for l in lanes) for ci in range(len(cores))):
            continue
        items = plan["items"]
        # Make sure there are at least two port colors when the palette allows it.
        used = sorted({l["port"] for l in lanes})
        if len(palette["ports"]) >= 2 and len(used) < 2 and len(lanes) >= 2:
            lanes[int(rng.integers(len(lanes)))]["port"] = int(_draw(rng, [c for c in palette["ports"] if c != used[0]]))
        lookup = {(l["core"], l["side"], l["index"]): l for l in lanes}
        ok = True
        if "adjacent" in items:  # two neighbouring ports of different colors on one side
            spots = []
            for ci, core in enumerate(cores):
                for side in layout["active"][ci]:
                    indices = _lane_indices(side, core)
                    if len(indices) >= 2 and len(_lane_cells(side, indices[0], core, shape)) >= 2:
                        spots.extend((ci, side, i) for i in indices[:-1])
            if not spots:
                ok = False
            else:
                ci, side, i = spots[int(rng.integers(len(spots)))]
                pair = []
                for index in (i, i + 1):
                    if (ci, side, index) not in lookup:
                        lane = new_lane(ci, side, index)
                        lanes.append(lane)
                        lookup[(ci, side, index)] = lane
                    pair.append(lookup[(ci, side, index)])
                if pair[0]["port"] == pair[1]["port"]:
                    pair[1]["port"] = int(_draw(rng, [c for c in palette["ports"] if c != pair[0]["port"]]))
                for lane in pair:
                    lane["fixed"] = True
        if ok and "opposite" in items:  # one row/column with ports of different colors on both sides
            spots = []
            for ci, core in enumerate(cores):
                for axis in AXES:
                    a, b = AXIS_SIDES[axis]
                    if a in layout["active"][ci] and b in layout["active"][ci]:
                        idx = _lane_indices(a, core)
                        if len(_lane_cells(a, idx[0], core, shape)) >= 2 or len(_lane_cells(b, idx[0], core, shape)) >= 2:
                            spots.extend((ci, a, b, i) for i in idx)
            if not spots:
                ok = False
            else:
                ci, a, b, i = spots[int(rng.integers(len(spots)))]
                pair = []
                for side in (a, b):
                    if (ci, side, i) not in lookup:
                        lane = new_lane(ci, side, i)
                        lanes.append(lane)
                        lookup[(ci, side, i)] = lane
                    pair.append(lookup[(ci, side, i)])
                if pair[0]["port"] == pair[1]["port"] and not pair[1]["fixed"]:
                    pair[1]["port"] = int(_draw(rng, [c for c in palette["ports"] if c != pair[0]["port"]]))
                elif pair[0]["port"] == pair[1]["port"]:
                    pair[0]["port"] = int(_draw(rng, [c for c in palette["ports"] if c != pair[1]["port"]]))
                for lane in pair:
                    lane["fixed"] = True
        if not ok:
            continue
        if "gap" in items:  # a body-colored ring cell on a side that also carries ports
            has_gap = False
            for ci, core in enumerate(cores):
                for side in layout["active"][ci]:
                    indices = _lane_indices(side, core)
                    present = [i for i in indices if (ci, side, i) in lookup]
                    if present and len(present) < len(indices):
                        has_gap = True
            if not has_gap:
                removable = [l for l in lanes if not l["fixed"]
                             and sum(1 for o in lanes if o["core"] == l["core"] and o["side"] == l["side"]) >= 2]
                if not removable:
                    continue
                victim = removable[int(rng.integers(len(removable)))]
                lanes.remove(victim)
                del lookup[(victim["core"], victim["side"], victim["index"])]
        free = list(range(len(lanes)))
        all_ports = [l["port"] for l in lanes]

        def take(condition=None, axis=None, min_length=2):
            pool = [i for i in free if len(lanes[i]["cells"]) >= min_length
                    and (axis is None or lanes[i]["axis"] == axis)
                    and (condition is None or condition(lanes[i]))]
            if not pool:
                return None
            choice = pool[int(rng.integers(len(pool)))]
            free.remove(choice)
            return lanes[choice]

        def size_for(lane, force_bar=False):
            return _bar_size(rng, len(lane["cells"]), plan["bars"], force=force_bar)

        def neighbour_colors(lane):
            return sorted({lookup[k]["port"] for k in ((lane["core"], lane["side"], lane["index"] - 1),
                                                       (lane["core"], lane["side"], lane["index"] + 1))
                           if k in lookup and lookup[k]["port"] != lane["port"]})

        def opposite_color(lane):
            other = lookup.get((lane["core"], OPPOSITE[lane["side"]], lane["index"]))
            return other["port"] if other is not None and other["port"] != lane["port"] else None

        order = (["novel_port"] if plan.get("novel", (None, None))[1] == "port" else []) + \
            [i for i in ("adjacent", "opposite", "bar_match", "bar_push") if i in items] + \
            [i for i in items if i.startswith(("far:", "push:"))] + \
            [i for i in ("far", "cross") if i in items] + \
            [i for i in items if i.startswith("open")] + \
            [i for i in ("noop",) if i in items] + \
            (["novel_foreign"] if plan.get("novel", (None, None))[1] == "foreign" else [])
        for item in order:
            lane = None
            if item == "novel_port":
                lane = take()
                if lane is not None:
                    lane["port"] = plan["novel"][0]
                    size = size_for(lane)
                    lane["plan"] = ("match", _match_start(rng, len(lane["cells"]), size), lane["port"], size)
            elif item == "adjacent":
                lane = take(condition=lambda l: bool(neighbour_colors(l)))
                if lane is not None:
                    size = size_for(lane)
                    lane["plan"] = ("mismatch", _push_start(rng, len(lane["cells"]), size, inner=True),
                                    int(_draw(rng, neighbour_colors(lane))), size)
            elif item == "opposite":
                lane = take(condition=lambda l: opposite_color(l) is not None)
                if lane is not None:
                    size = size_for(lane)
                    lane["plan"] = ("mismatch", _push_start(rng, len(lane["cells"]), size), opposite_color(lane), size)
            elif item in ("bar_match", "bar_push"):
                lane = take(min_length=4)
                if lane is not None:
                    size = size_for(lane, force_bar=True)
                    length = len(lane["cells"])
                    if item == "bar_match":
                        lane["plan"] = ("match", _match_start(rng, length, size, far=True), lane["port"], size)
                    else:
                        pool = sorted(set(all_ports + palette["foreign"]) - {lane["port"]})
                        if pool:
                            lane["plan"] = ("mismatch", _push_start(rng, length, size, inner=True),
                                            int(_draw(rng, pool)), size)
                        else:
                            lane = None
            elif item.startswith("far"):
                lane = take(axis=item.split(":")[1] if ":" in item else None)
                if lane is not None:
                    size = size_for(lane)
                    lane["plan"] = ("match", _match_start(rng, len(lane["cells"]), size, far=True), lane["port"], size)
            elif item.startswith("push") or item == "cross":
                cross = item == "cross"
                lane = take(axis=item.split(":")[1] if ":" in item else None,
                            condition=(lambda l: any(c != l["port"] for c in all_ports)) if cross else None)
                if lane is not None:
                    others = sorted(set(all_ports) - {lane["port"]})
                    pool = others if cross else sorted(set(others + palette["foreign"]))
                    if not pool:
                        lane = None
                    else:
                        size = size_for(lane)
                        lane["plan"] = ("mismatch", _push_start(rng, len(lane["cells"]), size, inner=True),
                                        int(_draw(rng, pool)), size)
            elif item.startswith("open"):
                lane = take(axis=item.split(":")[1] if ":" in item else None, min_length=1)
                if lane is not None:
                    lane["plan"] = ("open", None, None, 0)
            elif item == "noop":
                lane = take()
                if lane is not None:
                    length = len(lane["cells"])
                    size = size_for(lane)
                    pool = sorted(set(all_ports + palette["foreign"]) - {lane["port"]})
                    if pool and rng.random() < 0.5:
                        lane["plan"] = ("mismatch", length - size + 1, int(_draw(rng, pool)), size)
                    else:
                        lane["plan"] = ("match", 1, lane["port"], size)
                    lane["keep"] = True
                else:
                    continue  # optional extra
            elif item == "novel_foreign":
                lane = take(condition=lambda l: l["port"] != plan["novel"][0])
                if lane is not None:
                    size = size_for(lane)
                    lane["plan"] = ("mismatch", _push_start(rng, len(lane["cells"]), size), plan["novel"][0], size)
            if lane is None:
                ok = False
                break
            lane["keep"] = lane["keep"] or item in ("adjacent", "opposite", "cross") or item.startswith(("far", "push", "bar"))
        if not ok:
            continue
        all_ports = [l["port"] for l in lanes]
        weights = style["outcomes"]
        for i in free:
            lane = lanes[i]
            length = len(lane["cells"])
            others = sorted(set(all_ports) - {lane["port"]})
            pool = others + [c for c in palette["foreign"] if c not in others]
            choice = "open" if length < 2 else _draw(rng, ("open", "match", "mismatch"), weights)
            if choice == "mismatch" and not pool:
                choice = "match"
            if choice == "open":
                lane["plan"] = ("open", None, None, 0)
                continue
            size = size_for(lane)
            if choice == "match":
                lane["plan"] = ("match", _match_start(rng, length, size), lane["port"], size)
                continue
            same_side = [c for c in neighbour_colors(lane)] + ([opposite_color(lane)] if opposite_color(lane) else [])
            u = rng.random()
            if same_side and u < 0.3:
                source = same_side
            elif others and (u < 0.65 or not palette["foreign"]):
                source = others
            else:
                source = palette["foreign"] or others
            lane["plan"] = ("mismatch", _push_start(rng, length, size), int(_draw(rng, source)), size)
        # Parent A's layout: sometimes all markers of a side share one track parallel to the core.
        for ci in range(len(cores)):
            for side in layout["active"][ci]:
                if rng.random() >= style["track"]:
                    continue
                members = [l for l in lanes if l["core"] == ci and l["side"] == side and l["plan"][0] != "open"]
                if len(members) < 2:
                    continue
                low, high = 1, 99
                for l in members:
                    kind_name, start, color, size = l["plan"]
                    length = len(l["cells"])
                    if kind_name == "match":
                        lo, hi = (2 if l["keep"] and start >= 2 else 1), length - size + 1
                        if l["keep"] and start == 1:
                            hi = 1
                    else:
                        lo, hi = 1, length - size + 1
                        if l["keep"] and start + size - 1 < length:
                            hi = length - size
                        elif l["keep"]:
                            lo = length - size + 1
                    low, high = max(low, lo), min(high, hi)
                if low > high:
                    continue
                colors = {}
                clash = False
                for l in members:
                    colors[l["index"]] = l["plan"][2]
                for l in members:
                    if colors.get(l["index"] + 1) == l["plan"][2]:
                        clash = True
                if clash:
                    continue
                depth = int(rng.integers(low, high + 1))
                for l in members:
                    l["plan"] = (l["plan"][0], depth, l["plan"][2], l["plan"][3])
        port_cells, marker_cells = set(), set()
        for lane in lanes:
            grid[_ring_cell(lane["side"], lane["index"], cores[lane["core"]])] = lane["port"]
            port_cells.update(lane["cells"])
            kind_name, start, color, size = lane["plan"]
            if kind_name != "open":
                for cell in lane["cells"][start - 1:start - 1 + size]:
                    grid[cell] = color
                    marker_cells.add(cell)
        output = grid.copy()
        for lane in lanes:
            kind_name, start, color, size = lane["plan"]
            cells = lane["cells"]
            if kind_name == "open":
                for cell in cells:
                    output[cell] = lane["port"]
                continue
            for cell in cells[start - 1:start - 1 + size]:
                output[cell] = background
            if kind_name == "match":
                for cell in cells[:size]:
                    output[cell] = color
            else:
                for cell in cells[:len(cells) - size]:
                    output[cell] = lane["port"]
                for cell in cells[len(cells) - size:]:
                    output[cell] = color
        if plan["inert"]:
            before = grid.copy()
            colors = sorted(set(palette["ports"]) | set(palette["foreign"]) | {l["port"] for l in lanes})
            placed = _place_inert(rng, grid, cores, bodies, port_cells, marker_cells, colors,
                                  plan["inert"], plan.get("inert_need", ()))
            if not placed:
                continue
            changed = grid != before
            output[changed] = grid[changed]
        try:
            info = _analyze(grid, output)
        except AssertionError:
            continue
        if len(info["cores"]) != len(cores) or not np.array_equal(solve(grid), output):
            continue
        wanted = [i for i in items if i not in ("noop",)] + plan.get("check", [])
        if any(not _item_ok(info, item) for item in wanted):
            continue
        return grid, output, info
    return None


# ----------------------------------------------------------------------------------------------
# Puzzle construction
# ----------------------------------------------------------------------------------------------

def _style(rng):
    calm = rng.random() < 1 / 14
    background = 0 if rng.random() < 0.5 else int(rng.integers(10))
    body = int(_draw(rng, [c for c in range(10) if c != background]))
    body2 = body if rng.random() < 0.5 else int(_draw(rng, [c for c in range(10) if c not in (background, body)]))
    if calm:
        geometry = {"classic": 1.0}
    else:
        geometry = _draw(rng, (
            {"classic": 4, "big": 1, "bar": 1.5, "flush": 1.5, "barrier": 1.5},
            {"classic": 3, "big": 2, "bar": 2, "flush": 1, "barrier": 2},
            {"classic": 6, "big": 1, "bar": 1, "flush": 1, "barrier": 1},
            {"classic": 2, "big": 1, "bar": 2, "flush": 2, "barrier": 3},
        ))
    return {
        "calm": calm,
        "palette_mode": "fixed" if calm or rng.random() < 0.5 else "varying",
        "fixed": (background, body, body2),
        "bands": [0, 1, 0] if calm else _draw(rng, ([3, 2, 1], [2, 2, 1], [1, 1, 1], [1, 2, 2], [2, 3, 1])),
        "density": _draw(rng, ([1, 1, 2], [2, 1, 1], [1, 2, 1], [1, 1, 1])),
        "outcomes": _draw(rng, ([1, 1, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [1, 1, 3])),
        "track": 0.0 if calm else _draw(rng, (0.0, 0.25, 0.5)),
        "geometry": geometry,
        "inert": "none" if calm else _draw(rng, ("none", "basic", "rich"), (0.4, 0.2, 0.4)),
        "bars": (not calm) and rng.random() < 0.3,
        "twin": (not calm) and rng.random() < 0.2,
        "teach": 2 if calm else int(_draw(rng, (1, 2, 3), (0.25, 0.45, 0.3))),
        "novel": 0.0 if calm else 0.5,
    }


def _random_sides(rng, count):
    return [SIDES[i] for i in sorted(int(v) for v in rng.choice(4, size=count, replace=False))]


def _plain_geometry(rng, style, weights):
    """Pick a geometry first, then sides that suit it (barrier cores only carry one axis)."""
    kinds = sorted(style["geometry"])
    kind = _draw(rng, kinds, [style["geometry"][k] for k in kinds])
    if kind == "barrier":
        axis = _draw(rng, AXES)
        sides = list(AXIS_SIDES[axis]) if rng.random() < 0.7 else [_draw(rng, AXIS_SIDES[axis])]
        return kind, sorted(sides, key=SIDES.index)
    count = int(_draw(rng, (1, 2, 3, 4), weights))
    if kind == "flush":
        count = min(count, 3)
    return kind, _random_sides(rng, count)


def _can_add(plan, side):
    if plan.get("geometry") == "barrier":
        return AXIS[side] == AXIS[plan["sides"][0]]
    if plan.get("geometry") == "flush":
        return len(plan["sides"]) < 3
    return True


def _inert_kinds(style):
    if style["inert"] == "basic":
        return ["quadrant"]
    if style["inert"] == "rich":
        return ["quadrant", "shape", "extension", "silent"]
    return []


def _teaching_groups(rng, style, count):
    extra = ["cross", "adjacent", "opposite", "gap"]
    if style["bars"]:
        extra += ["bar_match", "bar_push"]
    if style["twin"]:
        extra.append("twin")
    if style["inert"] == "rich":
        extra.append("silent_inert")
    elif style["inert"] == "basic":
        extra.append("inert")
    first = _draw(rng, AXES)
    second = AXES[1 - AXES.index(first)]
    if count == 1:
        return [[f"{r}:{a}" for r in ("open", "far", "push") for a in AXES] + extra]
    if count == 2:
        groups = [[f"far:{first}", f"push:{second}", f"open:{first}"],
                  [f"far:{second}", f"push:{first}", f"open:{second}"]]
    else:
        core = [f"{r}:{a}" for r in ("open", "far", "push") for a in AXES]
        order = rng.permutation(len(core))
        groups = [[], [], []]
        for n, i in enumerate(order):
            groups[n % 3].append(core[int(i)])
    for item in extra:
        groups[int(rng.integers(len(groups)))].append(item)
    return groups


def _sides_for(rng, items, weights=(2, 3, 3, 2)):
    axes = sorted({item.split(":")[1] for item in items if ":" in item})
    sides = set()
    for axis in axes:
        sides.add(_draw(rng, AXIS_SIDES[axis]))
    if "opposite" in items:
        axis = _draw(rng, axes) if axes else _draw(rng, AXES)
        sides.update(AXIS_SIDES[axis])
    target = int(_draw(rng, (1, 2, 3, 4), weights))
    while len(sides) < target:
        sides.add(_draw(rng, [s for s in SIDES if s not in sides]))
    return sorted(sides, key=SIDES.index)


def _attempt(rng, num_train, num_test):
    style = _style(rng)
    count = min(style["teach"], num_train)
    groups = _teaching_groups(rng, style, count)
    teach = sorted(int(i) for i in rng.choice(num_train, size=count, replace=False))
    kinds = _inert_kinds(style)
    plans = []
    for i in range(num_train):
        if i in teach:
            items = groups[teach.index(i)]
            plan = {"items": list(items), "kind": "teach", "sides": _sides_for(rng, items),
                    "twin": "twin" in items, "bars": style["bars"],
                    "inert": kinds if ("silent_inert" in items or "inert" in items) else [],
                    "inert_need": ["silent"] if "silent_inert" in items else (["quadrant"] if "inert" in items else [])}
        else:
            twin = style["twin"] and rng.random() < 0.3
            geometry, sides = _plain_geometry(rng, style, (3, 3, 2, 1))
            plan = {"items": [], "kind": "plain", "sides": sides, "geometry": None if twin else geometry,
                    "twin": twin, "bars": style["bars"],
                    "inert": kinds if (kinds and rng.random() < 0.4) else [], "inert_need": []}
        if rng.random() < 0.3:
            plan["items"] = plan["items"] + ["noop"]
        plans.append(plan)
    covered = set()
    for plan in plans:
        covered.update(plan["sides"])
    for side in SIDES:
        if side not in covered:
            options = [plan for plan in plans if _can_add(plan, side)]
            target = options[int(rng.integers(len(options)))] if options else plans[int(rng.integers(num_train))]
            if not _can_add(target, side):
                target["geometry"] = None
            target["sides"] = sorted(set(target["sides"]) | {side}, key=SIDES.index)
    train, infos = [], []
    for plan in plans:
        plan["palette"] = _pair_palette(rng, style, plan["kind"], plan["twin"])
        made = _make_pair(rng, style, plan)
        if made is None:
            return None
        train.append(made)
        infos.append(made[2])
    if _training_problems(infos):
        return None
    shown = {key: any(_features(info)[key] for info in infos) for key in ("bars", "twin", "inert", "silent_inert")}
    shown["bars"] = all(any(_item_ok(info, item) for info in infos) for item in ("bar_match", "bar_push"))
    train_colors = set()
    for info in infos:
        train_colors.update(info["colors"])
    test, test_infos = [], []
    for _ in range(num_test):
        geometry, sides = _plain_geometry(rng, style, (1.2, 2, 3, 3) if not style["calm"] else (0, 0, 1, 1))
        n_sides = len(sides)
        items = ["open", "far", "adjacent" if rng.random() < 0.6 else "cross"]
        if rng.random() < 0.5:
            items.append("noop")
        twin = shown["twin"] and rng.random() < 0.5
        bars = shown["bars"]
        if bars and rng.random() < 0.5:
            items.append(_draw(rng, ("bar_match", "bar_push")))
        inert = []
        need = []
        if shown["inert"] and rng.random() < 0.5:
            inert = [k for k in kinds if k != "silent" or shown["silent_inert"]] or ["quadrant"]
            if shown["silent_inert"] and rng.random() < 0.5:
                need = ["silent"]
        plan = {"items": items, "kind": "test", "sides": sides, "twin": twin, "bars": bars,
                "geometry": None if twin else geometry,
                "inert": inert, "inert_need": need, "dense": n_sides == 1, "check": ["cross"]}
        novel = None
        palette = None
        if rng.random() < style["novel"]:
            probe = _pair_palette(rng, style, "test", twin)
            fresh = [c for c in range(10) if c not in train_colors
                     and c != probe["background"] and c not in probe["bodies"]]
            if fresh:
                colour = int(_draw(rng, fresh))
                novel = (colour, _draw(rng, ("foreign", "port")))
                palette = _pair_palette(rng, {**style, "palette_mode": "fixed",
                                              "fixed": (probe["background"], probe["bodies"][0], probe["bodies"][-1])},
                                        "test", twin, forbidden=(colour,))
        plan["palette"] = palette or _pair_palette(rng, style, "test", twin)
        if novel is not None:
            plan["novel"] = novel
        made = _make_pair(rng, style, plan)
        if made is None:
            return None
        test.append(made)
        test_infos.append(made[2])
    if _training_problems(infos, test_infos) or any(_test_problems(info) for info in test_infos):
        return None
    keys = {made[0].tobytes() + bytes(str(made[0].shape), "ascii") for made in train + test}
    if len(keys) != num_train + num_test:
        return None
    return {"train": [{"input": g.tolist(), "output": o.tolist()} for g, o, _ in train],
            "test": [{"input": g.tolist(), "output": o.tolist()} for g, o, _ in test]}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _validate_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    for _ in range(300):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError("could not build a puzzle for this seed")


# ----------------------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------------------

def _as_grid(value):
    array = np.asarray(value)
    assert array.ndim == 2 and 1 <= array.shape[0] <= 30 and 1 <= array.shape[1] <= 30, "bad grid shape"
    assert np.issubdtype(array.dtype, np.integer), "grid cells must be integers"
    assert np.all((array >= 0) & (array <= 9)), "grid colors must be 0..9"
    return array.astype(int)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle needs train/test only"
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list), "splits must be lists"
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "illegal counts"
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys"
            grid, out = _as_grid(pair["input"]), _as_grid(pair["output"])
            key = (grid.shape, grid.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(grid, out)
            assert np.array_equal(solve(grid), out), "solve disagrees with stored output"
            infos[split].append(info)
    problems = _training_problems(infos["train"], infos["test"])
    assert not problems, "; ".join(problems)
    for info in infos["test"]:
        problems = _test_problems(info)
        assert not problems, "; ".join(problems)


# ----------------------------------------------------------------------------------------------
# Alternative rules (each ignores part of the composition) used to show both parents matter
# ----------------------------------------------------------------------------------------------

def _shortcut(input_grid, mode):
    grid = np.array(input_grid, dtype=int)
    background = int(grid[0, 0])
    cores = _find_cores(grid, background)
    if mode == "largest_core_only":
        cores = sorted(cores, key=lambda c: -((c[3] - c[1] + 1) * (c[4] - c[2] + 1)))[:1]
    out = grid.copy()
    ports = {}
    for ci, (body, top, left, bottom, right) in enumerate(cores):
        core = (top, left, bottom, right)
        for side in SIDES:
            for index in _lane_indices(side, core):
                port = int(grid[_ring_cell(side, index, core)])
                if port != body:
                    ports[(ci, side, index)] = port
    all_ports = set(ports.values())
    for ci, (body, top, left, bottom, right) in enumerate(cores):
        core = (top, left, bottom, right)
        for side in SIDES:
            for index in _lane_indices(side, core):
                port = int(grid[_ring_cell(side, index, core)])
                cells = _lane_cells(side, index, core, grid.shape)
                if not cells:
                    continue
                loose = [cell for cell in cells if grid[cell] != background]
                if port == body:
                    if mode == "inert_moves" and loose:  # silent-lane cells treated as pushed markers
                        color = int(grid[loose[0]])
                        for cell in loose:
                            out[cell] = background
                        for cell in cells[len(cells) - len(loose):]:
                            out[cell] = color
                    continue
                color = int(grid[loose[0]]) if loose else None
                size = len(loose)
                beam = body if mode == "body_beam" else port
                if mode == "b_only":  # parent B alone: any marker cancels its ray, markers stay put
                    if color is None:
                        for cell in cells:
                            out[cell] = beam
                    continue
                if mode == "obstacle":  # the beam stops in front of whatever it meets
                    for cell in cells:
                        if grid[cell] != background:
                            break
                        out[cell] = beam
                    continue
                if color is None:
                    if mode != "a_only":
                        for cell in cells:
                            out[cell] = beam
                    continue
                for cell in loose:
                    out[cell] = background
                if mode == "any_port":
                    same = color in all_ports
                elif mode == "same_side_any":
                    same = color in {p for (c, s, _), p in ports.items() if c == ci and s == side}
                elif mode == "neighbour":
                    same = color == port or color in {ports.get((ci, side, index + d)) for d in (-1, 1)}
                elif mode == "opposite":
                    same = color == port or color == ports.get((ci, OPPOSITE[side], index))
                else:
                    same = color == port
                if mode == "single_cell_bars":
                    size = 1
                if mode == "inverted":  # a matching key docks and lets the beam pass; a foreign key jams it
                    if same:
                        for cell in cells[size:]:
                            out[cell] = beam
                        for cell in cells[:size]:
                            out[cell] = color
                    else:
                        for cell in cells[len(cells) - size:]:
                            out[cell] = color
                    continue
                if mode == "beam_always":
                    for cell in cells:
                        out[cell] = beam
                if same:
                    for cell in cells[:size]:
                        out[cell] = color
                else:
                    if mode != "a_only":
                        for cell in cells[:len(cells) - size]:
                            out[cell] = beam
                    for cell in cells[len(cells) - size:]:
                        out[cell] = color
    return out


SHORTCUTS = ("a_only", "b_only", "obstacle", "any_port", "body_beam", "beam_always", "inverted")
LANE_SHORTCUTS = ("same_side_any", "neighbour", "opposite")
FEATURE_SHORTCUTS = {"inert_moves": "silent_inert", "single_cell_bars": "bars", "largest_core_only": "twin"}


# ----------------------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------------------

def _differs(puzzle, split, mode):
    return [not np.array_equal(_shortcut(p["input"], mode), np.asarray(p["output"])) for p in puzzle[split]]


def _family_stats(seeds):
    """Measure the returned grids of ``generate_puzzle(seed)`` (default counts) for each seed.

    Every puzzle is validated and every shortcut check is asserted along the way.
    """
    shapes, cores, side_sets, palettes, signatures = set(), set(), set(), set(), set()
    stats = {"puzzles": 0, "pairs": 0, "decoy_puzzles": 0, "novel_tests": 0, "noop_lanes": 0, "fixed": 0,
             "varying": 0, "matches": 0, "mismatches": 0, "opens": 0, "one_cell_beams": 0, "lanes": 0,
             "two_cell": 0, "long_lanes": 0, "big_core": 0, "flush": 0, "barrier": 0, "bar_moves": 0,
             "bar_puzzles": 0, "twin_pairs": 0, "twin_puzzles": 0, "twin_tests": 0, "silent_inert": 0,
             "inert_shapes": 0, "extension_inert": 0, "one_side_tests": 0, "track_sides": 0,
             "adjacent_decoys": 0, "opposite_decoys": 0, "small_grids": 0, "test_not_richest": 0}
    for seed in seeds:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        stats["puzzles"] += 1
        signatures.add(repr(puzzle))
        infos = [_analyze(p["input"], p["output"]) for p in puzzle["train"]]
        tests = [_analyze(p["input"], p["output"]) for p in puzzle["test"]]
        train_colors = set().union(*(set(info["colors"]) for info in infos))
        stats["fixed" if len({(i["background"], i["bodies"][0]) for i in infos + tests}) == 1 else "varying"] += 1
        stats["decoy_puzzles"] += any(info["inert"] for info in infos + tests)
        stats["one_side_tests"] += sum(len(info["sides"]) == 1 for info in tests)
        stats["twin_puzzles"] += any(len(info["cores"]) == 2 for info in infos + tests)
        stats["twin_tests"] += sum(len(info["cores"]) == 2 for info in tests)
        stats["bar_puzzles"] += any(_features(info)["bars"] for info in infos + tests)
        stats["test_not_richest"] += any(len(i["lanes"]) > len(tests[0]["lanes"]) for i in infos)
        for info in tests:
            if any(c not in train_colors for c in info["colors"]):
                stats["novel_tests"] += 1
        for info in infos + tests:
            stats["pairs"] += 1
            shapes.add(info["shape"])
            height, width = info["shape"]
            stats["small_grids"] += max(height, width) <= 13
            for top, left, bottom, right in info["cores"]:
                h, w = bottom - top + 1, right - left + 1
                cores.add((h, w))
                stats["big_core"] += max(h, w) > 10
                stats["barrier"] += h == height or w == width
            stats["flush"] += info["flush_sides"] > 0
            stats["twin_pairs"] += len(info["cores"]) == 2
            side_sets.add(tuple(info["sides"]))
            palettes.add(tuple(info["colors"]))
            stats["silent_inert"] += sum(o["category"] == "silent" for o in info["inert"])
            stats["extension_inert"] += sum(o["category"] == "extension" for o in info["inert"])
            stats["inert_shapes"] += sum(len(o["cells"]) > 1 for o in info["inert"])
            depths = {}
            for lane in info["lanes"]:
                stats[{"open": "opens", "match": "matches", "mismatch": "mismatches"}[lane["kind"]]] += 1
                stats["noop_lanes"] += lane["noop"]
                stats["one_cell_beams"] += lane["length"] == 1
                stats["lanes"] += 1
                stats["two_cell"] += lane["length"] == 2
                stats["long_lanes"] += lane["length"] >= 8
                stats["bar_moves"] += lane["size"] >= 2 and not lane["noop"]
                stats["adjacent_decoys"] += lane["adjacent"]
                stats["opposite_decoys"] += lane["opposite"]
                if lane["start"]:
                    depths.setdefault((lane["core"], lane["side"]), []).append(lane["start"])
            stats["track_sides"] += sum(len(v) >= 3 and len(set(v)) == 1 for v in depths.values())
        # Ignoring either parent's contribution (or taking a tempting shortcut) must fail.
        for mode in SHORTCUTS:
            assert all(_differs(puzzle, "test", mode)), f"shortcut {mode} solves seed {seed} test"
            assert any(_differs(puzzle, "train", mode)), f"shortcut {mode} fits seed {seed} training"
        for mode in LANE_SHORTCUTS:
            assert any(_differs(puzzle, "train", mode)), f"lane shortcut {mode} fits seed {seed} training"
        features = {key: any(_features(info)[key] for info in infos + tests) for key in ("bars", "twin", "silent_inert")}
        for mode, feature in FEATURE_SHORTCUTS.items():
            if features[feature]:
                assert any(_differs(puzzle, "train", mode)), f"{mode} fits seed {seed} training"
    stats.update(distinct_puzzles=len(signatures), grid_shapes=len(shapes), core_shapes=len(cores),
                 side_sets=len(side_sets), palettes=len(palettes))
    return stats


def self_test() -> None:
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, 5)):
        try:
            generate_puzzle(0, num_train=bad[0], num_test=bad[1])
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")
    assert generate_puzzle(3, num_train=None, num_test=None) == generate_puzzle(3), "None means default counts"
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(5 + a * 7 + b, num_train=a, num_test=b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
            validate_generated_puzzle(puzzle)
    for seed in range(0, 42, 7):
        first, again = generate_puzzle(seed), generate_puzzle(seed)
        assert first == again, f"seed {seed} is not deterministic"
    stats = _family_stats(range(42))
    assert stats["distinct_puzzles"] == 42, "seeds must give distinct puzzles"
    assert stats["grid_shapes"] >= 80, stats
    assert stats["core_shapes"] >= 40, stats
    assert stats["side_sets"] >= 13 and stats["palettes"] >= 80, stats
    assert stats["fixed"] >= 5 and stats["varying"] >= 5, stats
    assert stats["decoy_puzzles"] >= 8 and stats["novel_tests"] >= 5, stats
    assert stats["noop_lanes"] >= 20 and stats["one_cell_beams"] >= 3, stats
    assert min(stats["matches"], stats["mismatches"], stats["opens"]) >= 100, stats
    assert stats["two_cell"] <= 0.15 * stats["lanes"] and stats["long_lanes"] >= 0.2 * stats["lanes"], stats
    assert stats["big_core"] >= 10 and stats["flush"] >= 5 and stats["barrier"] >= 5, stats
    assert stats["bar_moves"] >= 20 and stats["bar_puzzles"] >= 4, stats
    assert stats["twin_pairs"] >= 5 and stats["twin_tests"] >= 1, stats
    assert stats["silent_inert"] >= 5 and stats["inert_shapes"] >= 2 and stats["extension_inert"] >= 2, stats
    assert stats["one_side_tests"] >= 1 and stats["small_grids"] >= 15 and stats["test_not_richest"] >= 10, stats
    assert stats["adjacent_decoys"] >= 42 and stats["opposite_decoys"] >= 42, stats


if __name__ == "__main__":
    self_test()
    print("self_test passed")
