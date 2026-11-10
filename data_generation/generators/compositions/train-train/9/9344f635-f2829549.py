"""Stencil beams: a composition of ARC 9344f635 (A) and f2829549 (B).

Rule (input only):
1. The input holds two congruent, coordinate-aligned panels separated by one
   full-length divider line (a column for side-by-side panels, a row for
   stacked panels).  The divider colour occurs nowhere else.
2. The background is the most frequent panel colour.  In each panel the most
   frequent other colour is that panel's wall colour; every remaining colour in
   a panel forms straight two-cell domino seeds.
3. A panel-sized output is built from the overlay.  A cell is *open* when
   neither panel has a wall there (B: intersection of empty space).
4. Every horizontal seed broadcasts its colour along its whole row and every
   vertical seed along its whole column (A), regardless of which panel holds
   the seed.  A beam is visible only on open cells; it passes behind walls and
   reappears beyond them.  Where a horizontal and a vertical beam cross on an
   open cell the horizontal colour wins (A).
5. Open cells lit by no beam take the divider colour (B's marker, visibly
   encoded by the divider); every walled cell becomes background.
"""
from __future__ import annotations

import numpy as np

_PANEL_MIN, _PANEL_MAX = 4, 14
_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_ALTERNATIVES = (
    "ignore_beams", "ignore_stencil", "vertical_priority", "walls_and",
    "first_panel_walls_only", "second_panel_walls_only", "beams_stop_at_walls",
    "first_panel_seeds_only", "second_panel_seeds_only",
    "beam_host_walls_only", "beam_other_walls_only",
)


# ----------------------------------------------------------------------------
# Input analysis and the generator-owned solver
# ----------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    checked = []
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        checked.append(value)
    if checked[0] <= checked[1]:
        raise ValueError("num_train must exceed num_test")
    return checked[0], checked[1]


def _components(mask):
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask)
    components = []
    for row, column in zip(*np.nonzero(mask)):
        if seen[row, column]:
            continue
        seen[row, column] = True
        stack, component = [(int(row), int(column))], []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for dy, dx in _NEIGHBOURS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        components.append(sorted(component))
    return components


def _split(grid):
    """Return (orientation, divider colour, first panel, second panel)."""
    height, width = grid.shape
    found = []
    if width >= 3 and width % 2 == 1:
        middle = width // 2
        line = grid[:, middle]
        rest = np.concatenate([grid[:, :middle].ravel(), grid[:, middle + 1:].ravel()])
        if np.all(line == line[0]) and not np.any(rest == line[0]):
            found.append(("vertical", int(line[0]), grid[:, :middle], grid[:, middle + 1:]))
    if height >= 3 and height % 2 == 1:
        middle = height // 2
        line = grid[middle]
        rest = np.concatenate([grid[:middle].ravel(), grid[middle + 1:].ravel()])
        if np.all(line == line[0]) and not np.any(rest == line[0]):
            found.append(("horizontal", int(line[0]), grid[:middle], grid[middle + 1:]))
    if len(found) != 1:
        raise ValueError("input needs exactly one full divider line in a unique colour")
    return found[0]


def _plurality(values):
    colours, counts = np.unique(np.asarray(values).ravel(), return_counts=True)
    if len(colours) == 0:
        return None
    order = np.argsort(-counts, kind="stable")
    if len(order) > 1 and counts[order[0]] == counts[order[1]]:
        raise ValueError("ambiguous plurality colour")
    return int(colours[order[0]])


def _analyze(grid):
    grid = np.asarray(grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    orientation, divider, first, second = _split(grid)
    panels = (first, second)
    background = _plurality(np.concatenate([first.ravel(), second.ravel()]))
    wall_colours, wall_masks, seeds = [], [], []
    for index, panel in enumerate(panels):
        wall = _plurality(panel[panel != background])
        wall_colours.append(wall)
        wall_masks.append(panel == wall if wall is not None else np.zeros(panel.shape, dtype=bool))
        for colour in sorted(set(int(c) for c in np.unique(panel)) - {background, wall}):
            for component in _components(panel == colour):
                if len(component) != 2:
                    raise ValueError("seed colours must form two-cell dominoes")
                (r1, c1), (r2, c2) = component
                axis = "h" if r1 == r2 else "v"
                seeds.append({
                    "panel": index, "axis": axis, "index": r1 if axis == "h" else c1,
                    "colour": colour, "cells": ((r1, c1), (r2, c2)),
                })
    rows, cols = {}, {}
    for seed in seeds:
        lines = rows if seed["axis"] == "h" else cols
        if lines.setdefault(seed["index"], seed["colour"]) != seed["colour"]:
            raise ValueError("one row or column carries two beam colours")
    open_mask = ~wall_masks[0] & ~wall_masks[1]
    return {
        "orientation": orientation, "divider": divider, "background": background,
        "wall_colours": tuple(wall_colours), "walls": tuple(wall_masks), "panels": panels,
        "seeds": seeds, "rows": rows, "cols": cols, "open": open_mask, "shape": first.shape,
    }


def _paint(info, open_mask, *, stencil=True, horizontal_last=True, rows=None, cols=None):
    height, width = info["shape"]
    rows = info["rows"] if rows is None else rows
    cols = info["cols"] if cols is None else cols
    output = np.full((height, width), info["background"], dtype=int)
    output[open_mask] = info["divider"]
    passes = (("v", cols), ("h", rows)) if horizontal_last else (("h", rows), ("v", cols))
    for axis, lines in passes:
        for index, colour in lines.items():
            if axis == "h":
                visible = open_mask[index] if stencil else np.ones(width, dtype=bool)
                output[index, visible] = colour
            else:
                visible = open_mask[:, index] if stencil else np.ones(height, dtype=bool)
                output[visible, index] = colour
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    return _paint(info, info["open"])


# ----------------------------------------------------------------------------
# Shortcut rules used to prove that the demonstrations need both parents
# ----------------------------------------------------------------------------

def _seed_lines(info, panel):
    rows, cols = {}, {}
    for seed in info["seeds"]:
        if seed["panel"] == panel:
            (rows if seed["axis"] == "h" else cols)[seed["index"]] = seed["colour"]
    return rows, cols


def _alternative(info, name):
    open_mask = info["open"]
    first, second = info["walls"]
    if name == "ignore_beams":
        output = np.full(info["shape"], info["background"], dtype=int)
        output[open_mask] = info["divider"]
        for seed in info["seeds"]:
            for cell in seed["cells"]:
                output[cell] = seed["colour"]
        return output
    if name == "ignore_stencil":
        return _paint(info, open_mask, stencil=False)
    if name == "vertical_priority":
        return _paint(info, open_mask, horizontal_last=False)
    if name == "walls_and":
        return _paint(info, ~(first & second))
    if name == "first_panel_walls_only":
        return _paint(info, ~first)
    if name == "second_panel_walls_only":
        return _paint(info, ~second)
    if name in ("first_panel_seeds_only", "second_panel_seeds_only"):
        rows, cols = _seed_lines(info, 0 if name.startswith("first") else 1)
        return _paint(info, open_mask, rows=rows, cols=cols)
    if name in ("beam_host_walls_only", "beam_other_walls_only"):
        # Beams respect only the walls of the panel holding (or not holding)
        # their seed; the marker still uses the two-panel intersection.
        output = np.full(info["shape"], info["background"], dtype=int)
        output[open_mask] = info["divider"]
        for axis in ("v", "h"):
            for seed in info["seeds"]:
                if seed["axis"] != axis:
                    continue
                host = seed["panel"] if name == "beam_host_walls_only" else 1 - seed["panel"]
                visible = ~info["walls"][host]
                for cell in _line_cells(info, axis, seed["index"]):
                    if visible[cell]:
                        output[cell] = seed["colour"]
        return output
    if name == "beams_stop_at_walls":
        height, width = info["shape"]
        output = np.full(info["shape"], info["background"], dtype=int)
        output[open_mask] = info["divider"]
        reached = {"h": [], "v": []}
        for seed in info["seeds"]:
            cells = set(seed["cells"])
            for (y, x) in seed["cells"]:
                dy, dx = (0, 1) if seed["axis"] == "h" else (1, 0)
                for sign in (-1, 1):
                    ny, nx = y + sign * dy, x + sign * dx
                    while 0 <= ny < height and 0 <= nx < width and open_mask[ny, nx]:
                        cells.add((ny, nx))
                        ny, nx = ny + sign * dy, nx + sign * dx
            reached[seed["axis"]].append((cells, seed["colour"]))
        for axis in ("v", "h"):
            for cells, colour in reached[axis]:
                for cell in cells:
                    output[cell] = colour
        return output
    raise KeyError(name)


# ----------------------------------------------------------------------------
# Pair-level facts shared by generation, validation and self_test
# ----------------------------------------------------------------------------

def _line_cells(info, axis, index):
    height, width = info["shape"]
    if axis == "h":
        return [(index, x) for x in range(width)]
    return [(y, index) for y in range(height)]


def _pair_facts(info, output):
    open_mask = info["open"]
    first, second = info["walls"]
    rows, cols = info["rows"], info["cols"]
    seeds_by_line = {}
    for seed in info["seeds"]:
        seeds_by_line.setdefault((seed["axis"], seed["index"]), []).append(seed)
    extensions = []
    blocked_line_cells = 0
    blocked_by = {"own": False, "other": False}
    for (axis, index), seeds in seeds_by_line.items():
        own_cells = {cell for seed in seeds for cell in seed["cells"]}
        hosts = {seed["panel"] for seed in seeds}
        visible = 0
        for cell in _line_cells(info, axis, index):
            if open_mask[cell]:
                if cell in own_cells:
                    continue
                if axis == "v" and cell[0] in rows:
                    continue
                visible += 1
            else:
                blocked_line_cells += 1
                for host in hosts:
                    host_wall = (first, second)[host][cell]
                    other_wall = (second, first)[host][cell]
                    if host_wall and not other_wall:
                        blocked_by["own"] = True
                    if other_wall and not host_wall:
                        blocked_by["other"] = True
        extensions.append(visible)
    crossings = [
        (r, c) for r in rows for c in cols
        if open_mask[r, c] and rows[r] != cols[c]
    ]
    states = set(zip(first.ravel().tolist(), second.ravel().tolist()))
    unlit = int(np.count_nonzero(output == info["divider"]))
    redundant = any(len(seeds) > 1 for seeds in seeds_by_line.values())
    return {
        "extensions": extensions, "blocked_line_cells": blocked_line_cells,
        "blocked_by": blocked_by, "crossings": crossings, "states": states,
        "unlit": unlit, "redundant": redundant,
        "hosts": {seed["panel"] for seed in info["seeds"]},
        "open_fraction": float(open_mask.mean()),
    }


def _check_pair(info, output, split):
    """Family invariants for one pair; raises AssertionError on violation."""
    height, width = info["shape"]
    assert _PANEL_MIN <= height <= _PANEL_MAX and _PANEL_MIN <= width <= _PANEL_MAX
    background, divider = info["background"], info["divider"]
    walls = info["wall_colours"]
    assert all(wall is not None for wall in walls), "each panel needs walls"
    assert background != divider and divider not in walls and background not in walls
    for panel, wall_mask in zip(info["panels"], info["walls"]):
        assert int(wall_mask.sum()) >= 4
        sizes = [len(component) for component in _components(wall_mask)]
        assert max(sizes) >= 3 and 2 not in sizes, "walls never look like domino seeds"
    assert info["seeds"], "each pair needs at least one seed"
    occupied = {}
    for seed in info["seeds"]:
        assert seed["colour"] not in (background, divider) + walls
        for cell in seed["cells"]:
            assert cell not in occupied, "seeds may not overlap in the overlay"
            occupied[cell] = seed["panel"]
            other = info["panels"][1 - seed["panel"]]
            assert other[cell] == background, "seed cells are open in the other panel"
    for seed in info["seeds"]:
        for (y, x) in seed["cells"]:
            for dy, dx in _NEIGHBOURS:
                neighbour = (y + dy, x + dx)
                if neighbour in seed["cells"]:
                    continue
                assert occupied.get(neighbour) != seed["panel"], "seeds in one panel may not touch"
    facts = _pair_facts(info, output)
    assert min(facts["extensions"]) >= 1, "every beam must be visible beyond its seed"
    assert facts["blocked_line_cells"] >= 1, "some beam must pass behind a wall"
    assert 0.15 <= facts["open_fraction"] <= 0.82
    if split == "train":
        assert facts["unlit"] >= 1, "training outputs show the divider-coloured marker"
    assert not np.array_equal(output, _alternative(info, "ignore_beams"))
    assert not np.array_equal(output, _alternative(info, "ignore_stencil"))
    return facts


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

def _sample_mask(rng, height, width):
    kinds = ("texture", "bars", "blocks", "holes", "motif", "walk", "diagonal", "sparse")
    weights = (0.10, 0.20, 0.20, 0.12, 0.08, 0.16, 0.06, 0.08)
    for _ in range(40):
        kind = str(rng.choice(kinds, p=weights))
        mask = np.zeros((height, width), dtype=bool)
        if kind == "texture":
            mask = rng.random((height, width)) < rng.uniform(0.15, 0.42)
        elif kind == "bars":
            for _ in range(int(rng.integers(2, 6))):
                thick = 1 if rng.random() < 0.8 else 2
                if rng.random() < 0.5:
                    y, x = int(rng.integers(height)), int(rng.integers(width))
                    mask[y:y + thick, x:x + int(rng.integers(2, width + 1))] = True
                else:
                    y, x = int(rng.integers(height)), int(rng.integers(width))
                    mask[y:y + int(rng.integers(2, height + 1)), x:x + thick] = True
        elif kind == "blocks":
            for _ in range(int(rng.integers(2, 6))):
                bh = int(rng.integers(1, max(2, height // 2) + 1))
                bw = int(rng.integers(1, max(2, width // 2) + 1))
                y, x = int(rng.integers(0, height - bh + 1)), int(rng.integers(0, width - bw + 1))
                mask[y:y + bh, x:x + bw] = True
        elif kind == "holes":
            y0, x0 = int(rng.integers(0, height // 3 + 1)), int(rng.integers(0, width // 3 + 1))
            y1 = int(rng.integers(max(y0 + 2, height - height // 3), height + 1))
            x1 = int(rng.integers(max(x0 + 2, width - width // 3), width + 1))
            mask[y0:y1, x0:x1] = True
            for _ in range(int(rng.integers(2, 5))):
                bh = int(rng.integers(1, max(2, (y1 - y0) // 2) + 1))
                bw = int(rng.integers(1, max(2, (x1 - x0) // 2) + 1))
                y, x = int(rng.integers(y0, max(y0 + 1, y1 - bh + 1))), int(rng.integers(x0, max(x0 + 1, x1 - bw + 1)))
                mask[y:y + bh, x:x + bw] = False
        elif kind == "motif":
            th, tw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
            tile = rng.random((th, tw)) < rng.uniform(0.25, 0.5)
            if not tile.any() or tile.all():
                continue
            mask = np.tile(tile, (height // th + 1, width // tw + 1))[:height, :width]
            mask = np.roll(mask, (int(rng.integers(th)), int(rng.integers(tw))), axis=(0, 1))
        elif kind == "walk":
            for _ in range(int(rng.integers(1, 4))):
                y, x = int(rng.integers(height)), int(rng.integers(width))
                for _ in range(int(rng.integers(height * width // 10 + 2, height * width // 5 + 4))):
                    mask[y, x] = True
                    dy, dx = _NEIGHBOURS[int(rng.integers(4))]
                    y, x = min(height - 1, max(0, y + dy)), min(width - 1, max(0, x + dx))
        elif kind == "diagonal":
            period = int(rng.integers(3, 6))
            slope = 1 if rng.random() < 0.5 else -1
            yy, xx = np.indices((height, width))
            mask = ((yy + slope * xx + int(rng.integers(period))) % period) == 0
            mask &= rng.random((height, width)) < 0.9
        else:
            mask = rng.random((height, width)) < 0.1
            for _ in range(int(rng.integers(1, 3))):
                bh, bw = int(rng.integers(1, 3)), int(rng.integers(2, 4))
                y, x = int(rng.integers(0, max(1, height - bh + 1))), int(rng.integers(0, max(1, width - bw + 1)))
                mask[y:y + bh, x:x + bw] = True
        if 0.1 <= mask.mean() <= 0.55 and mask.sum() >= 5:
            return mask
    return rng.random((height, width)) < 0.3


def _wall_pair(rng, height, width, mode):
    first = _sample_mask(rng, height, width)
    if mode == "independent":
        second = _sample_mask(rng, height, width)
    elif mode == "shifted":
        dy = dx = 0
        while dy == 0 and dx == 0:
            dy, dx = int(rng.integers(-2, 3)), int(rng.integers(-2, 3))
        second = np.roll(first, (dy, dx), axis=(0, 1))
    elif mode == "mirror":
        second = first[:, ::-1].copy() if rng.random() < 0.5 else first[::-1].copy()
    elif mode == "contained":
        second = first & (rng.random((height, width)) < rng.uniform(0.35, 0.75))
    elif mode == "disjoint":
        second = _sample_mask(rng, height, width) & ~first
    elif mode == "identical":
        second = first.copy()
    elif mode == "near_complement":
        second = ~first & (rng.random((height, width)) < rng.uniform(0.5, 0.75))
    else:
        raise ValueError(mode)
    if rng.random() < 0.5:
        first, second = second, first
    return first.copy(), second.copy()


def _place_seeds(rng, height, width, specs):
    overlay, panel_cells, placed = set(), (set(), set()), []
    for panel, axis, index, _ in specs:
        if axis == "h":
            options = [((index, x), (index, x + 1)) for x in range(width - 1)]
        else:
            options = [((y, index), (y + 1, index)) for y in range(height - 1)]
        for choice in rng.permutation(len(options)):
            cells = options[int(choice)]
            if any(cell in overlay for cell in cells):
                continue
            if any((y + dy, x + dx) in panel_cells[panel] for (y, x) in cells for dy, dx in _NEIGHBOURS):
                continue
            break
        else:
            return None
        overlay.update(cells)
        panel_cells[panel].update(cells)
        placed.append(cells)
    return placed


def _pair_shape(rng, regime):
    if regime == "anchor":
        low, high = 6, 10
    elif regime == "train":
        low, high = 4, 11
    elif rng.random() < 0.35:
        low, high = 5, 9
    else:
        low, high = 8, 14
    height = int(rng.integers(low, high + 1))
    width = height if rng.random() < 0.08 else int(rng.integers(low, high + 1))
    return height, width


def _line_counts(rng, regime, height, width):
    """Numbers of horizontal and vertical beams; single-axis pairs stay occasional."""
    cap_h, cap_v = max(1, height // 2), max(1, width // 2)
    for _ in range(40):
        if regime == "anchor":
            n_h, n_v = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            if rng.random() < 0.4:
                n_h += 1 if rng.random() < 0.5 else 0
                n_v += 0 if n_h > 1 and rng.random() < 0.5 else 1
        else:
            if regime == "train":
                total = int(rng.choice([1, 2, 3, 4, 5], p=[0.12, 0.25, 0.28, 0.20, 0.15]))
                single_axis = 0.15
            elif rng.random() < 0.3:
                total, single_axis = int(rng.integers(1, 4)), 0.15
            else:
                total, single_axis = int(rng.integers(3, 8)), 0.15
            if total == 1 or rng.random() < single_axis:
                n_h = total if rng.random() < 0.5 else 0
                n_v = total - n_h
            else:
                n_h = 1 + int(rng.binomial(total - 2, 0.5))
                n_v = total - n_h
        if n_h <= cap_h and n_v <= cap_v and n_h + n_v >= 1:
            if regime != "anchor" or (n_h >= 1 and n_v >= 1):
                return n_h, n_v
    return (1, 1) if regime == "anchor" else (1, 0)


def _wall_colours(rng, tone, available):
    first = int(rng.choice(available))
    if tone == "shared" or (tone == "free" and rng.random() < 0.4):
        return first, first
    second = int(rng.choice([c for c in available if c != first]))
    return first, second


def _build_pair(rng, regime, orientation, background, divider, tone, redundancy, crossing=False):
    """Return (input, output, info) for one pair or raise RuntimeError.

    ``crossing`` asks for at least one open crossing of differently coloured
    horizontal and vertical beams whenever the pair has both axes.
    """
    anchor = regime == "anchor"
    crossing = crossing or anchor
    modes = ["independent", "independent", "independent", "shifted", "shifted",
             "mirror", "contained", "contained", "disjoint", "disjoint"]
    if regime == "train":
        modes += ["identical"]
    if regime == "test":
        modes += ["near_complement", "near_complement", "identical"]
    for _ in range(400):
        height, width = _pair_shape(rng, regime)
        n_h, n_v = _line_counts(rng, regime, height, width)
        available = [c for c in range(10) if c not in (background, divider)]
        wall_first, wall_second = _wall_colours(rng, tone, available)
        seed_pool = [c for c in available if c not in (wall_first, wall_second)]
        total = n_h + n_v
        palette_size = int(rng.integers(1 if not anchor else 2, min(4, total, len(seed_pool)) + 1))
        palette = [int(c) for c in rng.choice(seed_pool, size=palette_size, replace=False)]
        rows = [int(r) for r in rng.choice(height, size=n_h, replace=False)]
        cols = [int(c) for c in rng.choice(width, size=n_v, replace=False)]
        row_colours = [int(rng.choice(palette)) for _ in rows]
        col_colours = [int(rng.choice(palette)) for _ in cols]
        if crossing and rows and cols and all(rc == cc for rc in row_colours for cc in col_colours):
            spare = [c for c in palette if c != row_colours[0]] or [c for c in seed_pool if c != row_colours[0]]
            col_colours[0] = int(rng.choice(spare))
        lines = [("h", r, c) for r, c in zip(rows, row_colours)] + [("v", c, k) for c, k in zip(cols, col_colours)]
        specs = []
        for axis, index, colour in lines:
            host = int(rng.integers(2))
            specs.append((host, axis, index, colour))
            if redundancy and rng.random() < 0.3:
                specs.append((1 - host if rng.random() < 0.6 else host, axis, index, colour))
        if anchor and len({host for host, _, _, _ in specs}) < 2:
            host, axis, index, colour = specs[-1]
            specs[-1] = (1 - host, axis, index, colour)
        specs = [specs[int(i)] for i in rng.permutation(len(specs))]
        mode = str(rng.choice(modes))
        walls = _wall_pair(rng, height, width, mode)
        placement = _place_seeds(rng, height, width, specs)
        if placement is None:
            continue
        for cells in placement:
            for cell in cells:
                walls[0][cell] = walls[1][cell] = False
        if crossing and rows and cols:
            options = [(r, c) for r, rc in zip(rows, row_colours) for c, cc in zip(cols, col_colours) if rc != cc]
            r, c = options[int(rng.integers(len(options)))]
            walls[0][r, c] = walls[1][r, c] = False
        panels = []
        for wall_mask, wall_colour in zip(walls, (wall_first, wall_second)):
            panels.append(np.where(wall_mask, wall_colour, background).astype(int))
        for (host, _, _, colour), cells in zip(specs, placement):
            for cell in cells:
                panels[host][cell] = colour
        if orientation == "vertical":
            grid = np.concatenate([panels[0], np.full((height, 1), divider), panels[1]], axis=1)
        else:
            grid = np.concatenate([panels[0], np.full((1, width), divider), panels[1]], axis=0)
        grid = grid.astype(int)
        try:
            info = _analyze(grid)
        except ValueError:
            continue
        if info["background"] != background or info["wall_colours"] != (wall_first, wall_second):
            continue
        if len(info["seeds"]) != len(specs):
            continue
        output = _paint(info, info["open"])
        try:
            facts = _check_pair(info, output, "test" if regime == "test" else "train")
        except AssertionError:
            continue
        if anchor and not _anchor_ok(info, output, facts):
            continue
        return grid, output, info
    raise RuntimeError("could not build a pair")


def _anchor_ok(info, output, facts):
    if not (info["rows"] and info["cols"] and facts["crossings"]):
        return False
    if facts["hosts"] != {0, 1} or len(facts["states"]) != 4:
        return False
    if not (facts["blocked_by"]["own"] and facts["blocked_by"]["other"]):
        return False
    return all(not np.array_equal(output, _alternative(info, name)) for name in _ALTERNATIVES)


def _distinct_colour(rng, exclude):
    return int(rng.choice([c for c in range(10) if c not in exclude]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    orientation_policy = str(rng.choice(["vertical", "horizontal", "mixed"], p=[0.45, 0.25, 0.30]))
    divider_policy = str(rng.choice(["fixed", "varying"]))
    background_policy = str(rng.choice(["fixed", "varying"], p=[0.6, 0.4]))
    tone = str(rng.choice(["distinct", "shared", "free"], p=[0.5, 0.2, 0.3]))
    base_background = int(rng.integers(10))
    base_divider = _distinct_colour(rng, {base_background})

    if orientation_policy == "mixed":
        first_two = ["vertical", "horizontal"] if rng.random() < 0.5 else ["horizontal", "vertical"]
        train_orientations = first_two + [str(rng.choice(["vertical", "horizontal"])) for _ in range(num_train - 2)]
    else:
        train_orientations = [orientation_policy] * num_train

    def colours_for(index, previous):
        """Background and divider for one pair; index 1 forces visible variation."""
        divider = base_divider if divider_policy == "fixed" or index == 0 else None
        if background_policy == "fixed" or index == 0:
            background = base_background
        else:
            exclude = {divider} if divider is not None else set()
            if index == 1:
                exclude.add(previous["background"])
            background = _distinct_colour(rng, exclude)
        if divider is None:
            exclude = {background}
            if index == 1:
                exclude.add(previous["divider"])
            divider = _distinct_colour(rng, exclude)
        return background, divider

    puzzle = {"train": [], "test": []}
    seen = set()
    taught_redundancy = False
    previous = None
    for index in range(num_train):
        background, divider = colours_for(index, previous)
        previous = {"background": background, "divider": divider}
        regime = "anchor" if index == 0 else "train"
        for _ in range(20):
            grid, output, info = _build_pair(
                rng, regime, train_orientations[index], background, divider, tone,
                redundancy=index > 0 and rng.random() < 0.5,
                crossing=rng.random() < 0.3,
            )
            key = (grid.shape, grid.tobytes())
            if key not in seen:
                break
        else:
            raise RuntimeError("could not build distinct inputs")
        seen.add(key)
        taught_redundancy |= _pair_facts(info, output)["redundant"]
        puzzle["train"].append({"input": grid, "output": output})
    trained_orientations = sorted(set(train_orientations))
    for index in range(num_test):
        background, divider = colours_for(2, previous)
        orientation = str(rng.choice(trained_orientations))
        for _ in range(20):
            grid, output, info = _build_pair(
                rng, "test", orientation, background, divider, tone,
                redundancy=taught_redundancy and rng.random() < 0.5,
                crossing=rng.random() < 0.75,
            )
            key = (grid.shape, grid.tobytes())
            if key not in seen:
                break
        else:
            raise RuntimeError("could not build distinct inputs")
        seen.add(key)
        puzzle["test"].append({"input": grid, "output": output})
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]
    validate_generated_puzzle(puzzle)
    return puzzle


# ----------------------------------------------------------------------------
# Validation and self-test
# ----------------------------------------------------------------------------

def _as_grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((grid >= 0) & (grid <= 9)))
    return grid.astype(int)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _as_grid(pair["input"]), _as_grid(pair["output"])
            key = (grid.shape, grid.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _analyze(grid)
            assert np.array_equal(output, _paint(info, info["open"])), "output disagrees with the rule"
            assert np.array_equal(output, solve(grid))
            facts = _check_pair(info, output, split)
            records[split].append((info, output, facts))
    train_records, test_records = records["train"], records["test"]
    # Every shortcut that drops part of either parent is contradicted in training.
    for name in _ALTERNATIVES:
        assert any(not np.array_equal(output, _alternative(info, name)) for info, output, _ in train_records), name
    assert any(facts["crossings"] for _, _, facts in train_records), "crossing priority not shown"
    states = set().union(*(facts["states"] for _, _, facts in train_records))
    assert len(states) == 4, "training must show all four wall states"
    assert set().union(*(facts["hosts"] for _, _, facts in train_records)) == {0, 1}
    assert any(facts["blocked_by"]["own"] for _, _, facts in train_records)
    assert any(facts["blocked_by"]["other"] for _, _, facts in train_records)
    train_orientations = {info["orientation"] for info, _, _ in train_records}
    assert all(info["orientation"] in train_orientations for info, _, _ in test_records)
    if any(facts["redundant"] for _, _, facts in test_records):
        assert any(facts["redundant"] for _, _, facts in train_records)
    train_dividers = {info["divider"] for info, _, _ in train_records}
    if any(info["divider"] not in train_dividers for info, _, _ in test_records):
        assert len(train_dividers) >= 2, "a new marker needs a demonstrated divider cue"
    train_backgrounds = {info["background"] for info, _, _ in train_records}
    if any(info["background"] not in train_backgrounds for info, _, _ in test_records):
        assert len(train_backgrounds) >= 2


def _puzzles_equal(first, second):
    return all(
        len(first[s]) == len(second[s]) and all(
            np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[s], second[s]))
        for s in ("train", "test"))


_WORKED_INPUT = ("00000805550", "22010800500", "00110800000", "10000850003", "11000855003")
_WORKED_OUTPUT = ("80003", "22002", "88003", "08883", "00883")


def self_test() -> None:
    # The documented worked example and every shortcut it rules out.
    grid = np.array([[int(c) for c in row] for row in _WORKED_INPUT])
    expected = np.array([[int(c) for c in row] for row in _WORKED_OUTPUT])
    assert np.array_equal(solve(grid), expected)
    worked = _analyze(grid)
    _check_pair(worked, expected, "train")
    assert all(not np.array_equal(expected, _alternative(worked, name)) for name in _ALTERNATIVES)

    count = {key: 0 for key in (
        "mixed", "stacked_only", "side_only", "varying_divider", "varying_background",
        "new_test_divider", "new_test_background", "new_test_seed_colour",
        "test_larger", "test_not_larger", "shared_walls", "distinct_walls", "redundant",
        "adjacent_beams", "axis_only", "identical_walls", "disjoint_walls", "nested_walls",
        "seeds_both_panels", "tests")}
    shapes, line_totals, open_fractions, fingerprints = set(), set(), [], set()
    test_alt_differs = {name: 0 for name in _ALTERNATIVES}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzles_equal(puzzle, generate_puzzle(seed)), "seed must reproduce the puzzle"
        validate_generated_puzzle(puzzle)
        fingerprints.add(b"".join(p["input"].tobytes() + p["output"].tobytes()
                                  for p in puzzle["train"] + puzzle["test"]))
        n_train = len(puzzle["train"])
        records = [(_analyze(p["input"]), p["output"]) for p in puzzle["train"] + puzzle["test"]]
        train = records[:n_train]
        layouts = {info["orientation"] for info, _ in records}
        count["mixed"] += len(layouts) == 2
        count["stacked_only"] += layouts == {"horizontal"}
        count["side_only"] += layouts == {"vertical"}
        count["varying_divider"] += len({info["divider"] for info, _ in records}) > 1
        count["varying_background"] += len({info["background"] for info, _ in records}) > 1
        train_dividers = {info["divider"] for info, _ in train}
        train_backgrounds = {info["background"] for info, _ in train}
        train_seed_colours = {seed_["colour"] for info, _ in train for seed_ in info["seeds"]}
        train_area = max(info["shape"][0] * info["shape"][1] for info, _ in train)
        for position, (info, output) in enumerate(records):
            facts = _pair_facts(info, output)
            first, second = info["walls"]
            shapes.add(info["shape"])
            line_totals.add(len(info["rows"]) + len(info["cols"]))
            open_fractions.append(facts["open_fraction"])
            walls = info["wall_colours"]
            count["shared_walls"] += walls[0] == walls[1]
            count["distinct_walls"] += walls[0] != walls[1]
            count["redundant"] += facts["redundant"]
            rows, cols = sorted(info["rows"]), sorted(info["cols"])
            count["adjacent_beams"] += any(b - a == 1 for line in (rows, cols) for a, b in zip(line, line[1:]))
            count["axis_only"] += not (rows and cols)
            count["identical_walls"] += bool(np.array_equal(first, second))
            count["disjoint_walls"] += not np.any(first & second)
            count["nested_walls"] += bool(np.all(first <= second) or np.all(second <= first))
            count["seeds_both_panels"] += facts["hosts"] == {0, 1}
            if position >= n_train:
                count["tests"] += 1
                count["new_test_divider"] += info["divider"] not in train_dividers
                count["new_test_background"] += info["background"] not in train_backgrounds
                count["new_test_seed_colour"] += bool({s["colour"] for s in info["seeds"]} - train_seed_colours)
                area = info["shape"][0] * info["shape"][1]
                count["test_larger"] += area > train_area
                count["test_not_larger"] += area <= train_area
                for name in _ALTERNATIVES:
                    test_alt_differs[name] += not np.array_equal(output, _alternative(info, name))
    tests = count["tests"]
    assert len(fingerprints) == 42
    assert count["mixed"] >= 5 and count["stacked_only"] >= 4 and count["side_only"] >= 8
    assert count["varying_divider"] >= 10 and count["varying_background"] >= 8
    assert count["new_test_divider"] >= 4 and count["new_test_background"] >= 4
    assert count["new_test_seed_colour"] >= 12
    assert count["test_larger"] >= 10 and count["test_not_larger"] >= 10
    assert len(shapes) >= 45 and len(line_totals) >= 6
    assert count["shared_walls"] >= 15 and count["distinct_walls"] >= 80
    assert count["redundant"] >= 10 and count["adjacent_beams"] >= 20 and count["axis_only"] >= 10
    assert count["identical_walls"] >= 5 and count["disjoint_walls"] >= 10 and count["nested_walls"] >= 10
    assert count["seeds_both_panels"] >= 80
    assert min(open_fractions) >= 0.15 and max(open_fractions) <= 0.82
    # Dropping either parent changes every test; finer shortcuts change most.
    assert test_alt_differs["ignore_beams"] == tests and test_alt_differs["ignore_stencil"] == tests
    for name in _ALTERNATIVES:
        assert test_alt_differs[name] >= tests // 2, (name, test_alt_differs[name])

    for num_train, num_test in ((2, 1), (10, 4), (5, 3), (3, 2)):
        puzzle = generate_puzzle(777 + num_train, num_train, num_test)
        assert len(puzzle["train"]) == num_train and len(puzzle["test"]) == num_test
    for arguments in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(5, *arguments)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {arguments}")
