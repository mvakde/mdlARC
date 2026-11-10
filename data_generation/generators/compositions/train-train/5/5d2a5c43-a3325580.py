"""Largest coalitions: an occupancy overlay changes grouping and shared area.

Only standard-library modules and NumPy are used. The solver has no access to
sampling parameters. Expected outputs are built from construction records and
then checked against reconstructed input geometry.
"""
from __future__ import annotations

from collections import deque
from itertools import combinations
import json

import numpy as np


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= s <= 30 for s in grid.shape),
             "Grid dimensions must be 1..30")
    _require(grid.dtype.kind in "iu" and np.all((0 <= grid) & (grid <= 9)),
             "Grid cells must be integers in 0..9")
    return grid.astype(np.int8, copy=False)


def _key(value):
    grid = np.asarray(value)
    return grid.shape, grid.tobytes()


def _components(mask, diagonal=False):
    """Return actual occupied coordinates, not bounding rectangles."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = NEIGHBORS
    if diagonal:
        steps += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    result = []
    for r, c in zip(*np.where(mask)):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        cells = []
        while queue:
            y, x = queue.popleft()
            cells.append((y, x))
            for dy, dx in steps:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        result.append(tuple(cells))
    return result


def _extract(input_grid):
    grid = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        if grid.shape[axis] % 2 != 1 or grid.shape[axis] < 7:
            continue
        half = grid.shape[axis] // 2
        if axis == 1:
            first, stripe, second = grid[:, :half], grid[:, half], grid[:, half + 1:]
        else:
            first, stripe, second = grid[:half], grid[half], grid[half + 1:]
        sep = int(stripe.flat[0])
        if np.all(stripe == sep) and not np.any(first == sep) and not np.any(second == sep):
            candidates.append((axis, first, second, sep))
    _require(len(candidates) == 1, "Expected one unambiguous central separator")
    axis, first, second, sep = candidates[0]
    _require(min(first.shape) >= 3, "Maps need a background border")
    bg = int(first[0, 0])
    for panel in (first, second):
        _require(np.all(panel[0] == bg) and np.all(panel[-1] == bg)
                 and np.all(panel[:, 0] == bg) and np.all(panel[:, -1] == bg),
                 "Each map must have the same complete background border")
    values, counts = np.unique(np.concatenate((first.ravel(), second.ravel())), return_counts=True)
    repeated = [int(v) for v, n in zip(values, counts) if v != bg and n > 1]
    _require(len(repeated) == 1, "Exactly one repeated neutral ink color is required")
    ink = repeated[0]
    labels = [int(v) for v, n in zip(values, counts) if v != bg and v != ink and n == 1]
    _require(2 <= len(labels) <= 7, "Expected 2..7 unique beacons")
    beacons = []
    for color in labels:
        for layer, panel in enumerate((first, second)):
            coords = np.argwhere(panel == color)
            if len(coords):
                r, c = map(int, coords[0])
                beacons.append((r, c, color, layer))
    _require(len({(b[0], b[1]) for b in beacons}) == len(beacons), "Conflicting beacon coordinates")
    _require(len({b[1] for b in beacons}) == len(beacons), "Beacon columns must be distinct")
    left, right = first != bg, second != bg
    union = left | right
    networks = []
    for cells in _components(union):
        cell_set = set(cells)
        members = [b for b in beacons if b[:2] in cell_set]
        _require(members and len(cells) <= 30, "Networks need beacons and area at most 30")
        networks.append({"cells": cells, "area": len(cells), "beacons": members})
    _require(2 <= len(networks) <= 7, "Expected 2..7 union networks")
    maximum = max(n["area"] for n in networks)
    winners = [n for n in networks if n["area"] == maximum]
    members = sorted((b for n in winners for b in n["beacons"]), key=lambda b: b[1])
    return {"first": first, "second": second, "left": left, "right": right,
            "union": union, "background": bg, "ink": ink, "separator": sep,
            "orientation": "horizontal" if axis == 0 else "vertical",
            "beacons": beacons, "networks": networks, "winners": winners,
            "maximum": maximum, "members": members}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles and winning memberships entirely from the given input."""
    analysis = _extract(input_grid)
    return np.tile(np.asarray([b[2] for b in analysis["members"]], dtype=np.int8),
                   (analysis["maximum"], 1))


def _normalize(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _shape(rng, area, style=None):
    styles = ["line", "block", "elbow", "branch", "grow"]
    rings = [(h, w) for h in range(3, 8) for w in range(3, 8)
             if 2 * h + 2 * w - 4 == area]
    if rings:
        styles += ["ring", "ring"]
    if style is None:
        style = styles[int(rng.integers(len(styles)))]
    if style == "line":
        cells = {(r, 0) for r in range(area)}
    elif style == "block":
        width = int(np.ceil(np.sqrt(area)))
        cells = {(i // width, i % width) for i in range(area)}
    elif style == "elbow":
        stem = (area + 1) // 2
        cells = {(r, 0) for r in range(stem)} | {(stem - 1, c) for c in range(1, area - stem + 1)}
    elif style == "branch":
        cells = {(0, 0)}
        rays = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for i in range(area - 1):
            dr, dc = rays[i % 4]
            length = 1 + i // 4
            cells.add((dr * length, dc * length))
    elif style == "ring":
        h, w = rings[int(rng.integers(len(rings)))]
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    else:
        side = int(np.ceil(np.sqrt(area))) + int(rng.integers(1, 3))
        cells = {(side // 2, side // 2)}
        while len(cells) < area:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in NEIGHBORS
                               if 0 <= r + dr < side and 0 <= c + dc < side} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    # Rotate and reflect independently of the shape's status or output rank.
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.integers(2):
        cells = {(r, -c) for r, c in cells}
    return _normalize(cells)


def _dimensions(cells):
    return 1 + max(r for r, _ in cells), 1 + max(c for _, c in cells)


def _place(rng, shapes, memberships, orientation, tie_lesson=False):
    """Free 2-D placement, including overlapping horizontal spans."""
    short_max, long_max = 14, 27
    # Transpose long thin silhouettes if they would exceed the doubled axis.
    adjusted = []
    for index, (shape, demand) in enumerate(zip(shapes, memberships)):
        h, w = _dimensions(shape)
        max_h, max_w = (long_max - 2, short_max - 2) if orientation == "vertical" else (short_max - 2, long_max - 2)
        needed_width = 3 if tie_lesson and index == 0 else demand
        if h > max_h or w > max_w or w < needed_width:
            shape = _normalize({(c, r) for r, c in shape})
        adjusted.append(shape)
    for expansion in range(4):
        total = sum(len(s) for s in adjusted)
        short = min(14, max(10, int(np.ceil(np.sqrt(total * 1.6)))) + expansion)
        long = min(27, max(14, int(np.ceil(total * 3.6 / (short - 2)))) + 3 * expansion)
        short = min(14, short + int(rng.integers(2)))
        long = min(27, long + int(rng.integers(3)))
        h, w = (long, short) if orientation == "vertical" else (short, long)
        h = max(h, max(_dimensions(s)[0] for s in adjusted) + 2)
        w = max(w, max(_dimensions(s)[1] for s in adjusted) + 2)
        if h > (27 if orientation == "vertical" else 14) or w > (14 if orientation == "vertical" else 27):
            continue
        for _ in range(32):
            occupied = set()
            placed = [None] * len(adjusted)
            order = sorted(range(len(adjusted)), key=lambda i: -len(adjusted[i]))
            success = True
            for depth, index in enumerate(order):
                shape = adjusted[index]
                sh, sw = _dimensions(shape)
                if tie_lesson and depth == 1:
                    # Deliberately teach diagonal non-connection. The first two
                    # placed networks are the tied winners. Candidate offsets
                    # vary freely around their boundaries, without a fixed lane.
                    offsets = sorted({(y + dy - r, x + dx - c)
                                      for y, x in occupied for r, c in shape
                                      for dy, dx in ((-1, -1), (-1, 1), (1, -1), (1, 1))
                                      if 1 <= y + dy - r < h - sh
                                      and 1 <= x + dx - c < w - sw})
                    rng.shuffle(offsets)
                else:
                    offsets = [(int(rng.integers(1, h - sh)), int(rng.integers(1, w - sw)))
                               for _ in range(100)]
                for top, left in offsets:
                    cells = {(top + r, left + c) for r, c in shape}
                    if cells & occupied or any((r + dr, c + dc) in occupied
                                                for r, c in cells for dr, dc in NEIGHBORS):
                        continue
                    placed[index] = tuple(sorted(cells))
                    occupied.update(cells)
                    break
                else:
                    success = False
                    break
            if not success:
                continue
            # Match each requested beacon to a different horizontal coordinate.
            options = [sorted({c for _, c in cells}) for cells in placed]
            search_order = sorted(range(len(placed)), key=lambda i: len(options[i]) - memberships[i])
            chosen = [None] * len(placed)

            def assign(depth, used):
                if depth == len(search_order):
                    if tie_lesson:
                        # Put the singleton winner's beacon between the two
                        # members of the other winner: global spatial ordering
                        # must be demonstrated before arbitrary tests use it.
                        return chosen[0][0] < chosen[1][0] < chosen[0][-1]
                    return True
                i = search_order[depth]
                available = [c for c in options[i] if c not in used]
                choices = list(combinations(available, memberships[i]))
                rng.shuffle(choices)
                for cols in choices:
                    chosen[i] = cols
                    if assign(depth + 1, used | set(cols)):
                        return True
                return False

            if assign(0, set()):
                beacon_cells = []
                for cells, cols in zip(placed, chosen):
                    points = []
                    for col in cols:
                        choices = [(r, c) for r, c in cells if c == col]
                        points.append(choices[int(rng.integers(len(choices)))])
                    beacon_cells.append(points)
                return (h, w), placed, beacon_cells
    raise RuntimeError("Bounded placement did not fit these shapes")


def _layer_masks(rng, shape, beacon_cells, mode):
    n = len(shape)
    tags = np.zeros(n, dtype=np.int8)
    if mode == "identical":
        tags[:] = 3
    elif mode == "one_layer":
        tags[:] = 1 + int(rng.integers(2))
    elif mode == "contained":
        tags[:] = 1
        tags[rng.choice(n, size=int(rng.integers(1, n)), replace=False)] = 3
    elif mode == "disjoint":
        tags[:] = rng.integers(1, 3, size=n)
        if n >= 2:
            tags[0], tags[-1] = 1, 2
    else:
        if mode == "checker":
            tags[:] = [1 + (r + c) % 2 for r, c in shape]
        else:
            tags[:] = rng.choice([1, 2, 3], size=n, p=[.35, .35, .30])
        # A teaching mixed network has non-beacon examples of every OR case
        # except empty/empty, which is already supplied by the map background.
        neutral = [i for i, p in enumerate(shape) if p not in beacon_cells]
        if len(neutral) >= 3:
            rng.shuffle(neutral)
            evidence = (1, 1, 2, 2, 3) if mode == "checker" and len(neutral) >= 5 else (1, 2, 3)
            for i, tag in zip(neutral, evidence):
                tags[i] = tag
    if rng.integers(2):
        tags = np.where(tags == 1, 2, np.where(tags == 2, 1, tags))
    return tags


def _sample_spec(rng, lesson, shift):
    if lesson == "assembly":
        maximum = int(rng.choice([7, 8, 10, 11, 13]))
        areas = [maximum, maximum - 1]
        memberships = [2, 1]
        if rng.integers(2):
            areas.append(int(rng.integers(2, maximum - 1)))
            memberships.append(1)
        styles = ["block", "line"] + [None] * (len(areas) - 2)
        modes = ["checker", "identical"] + ["mixed"] * (len(areas) - 2)
    elif lesson == "tie":
        maximum = int(rng.integers(5, 14))
        areas = [maximum, maximum, int(rng.integers(2, maximum))]
        memberships = [2, 1, 1]
        styles = [None] * 3
        modes = ["mixed", "disjoint", "mixed"]
    else:
        if lesson == "train":
            count, maximum = int(rng.integers(2, 4)), int(rng.integers(5, 14))
        else:
            count = int(rng.integers(4, 6)) if shift != "area" else int(rng.integers(2, 5))
            maximum = int(rng.integers(6, 16)) if shift == "count" else int(rng.integers(16, 24))
        winners = int(rng.integers(1, min(3, count) + 1))
        areas = [maximum] * winners + [int(rng.integers(2, maximum)) for _ in range(count - winners)]
        memberships = [1] * count
        # Additional labels are independent of winner status, and never exceed
        # the seven available identity colors.
        for _ in range(int(rng.integers(1, min(3, 7 - count) + 1))):
            candidates = [i for i in range(count) if memberships[i] < min(3, areas[i])]
            memberships[int(rng.choice(candidates))] += 1
        styles = [None] * count
        modes = [str(rng.choice(["mixed", "checker", "disjoint", "contained", "identical", "one_layer"]))
                 for _ in range(count)]
    return areas, memberships, styles, modes


def _same(a, b):
    return a is not None and b is not None and np.array_equal(a, b)


def _counterfactual(analysis, mode):
    """Ablations preserve every beacon, so missing labels cannot explain failure."""
    left, right = analysis["left"], analysis["right"]
    mask = {"first": left, "second": right, "xor": left ^ right,
            "and": left & right}.get(mode, analysis["union"]).copy()
    for r, c, _, _ in analysis["beacons"]:
        mask[r, c] = True
    groups = _components(mask, diagonal=mode == "eight")
    scored = []
    for cells in groups:
        points = set(cells)
        members = [b for b in analysis["beacons"] if b[:2] in points]
        if mode == "sum":
            score = sum(int(left[r, c]) + int(right[r, c]) for r, c in cells)
        elif mode == "box":
            h, w = _dimensions(_normalize(cells))
            score = h * w
        else:
            score = len(cells)
        scored.append((score, members))
    maximum = max(s for s, _ in scored)
    members = [b for s, labels in scored if s == maximum for b in labels]
    if mode == "all":
        members = analysis["beacons"]
    members = sorted(members, key=lambda b: b[1])
    if not members:
        return None
    return np.tile(np.asarray([b[2] for b in members], dtype=np.int8), (maximum, 1))


def _assembly_witness(analysis):
    if len(analysis["winners"]) != 1 or len(analysis["members"]) != 2:
        return False
    winner = analysis["winners"][0]
    if len(analysis["networks"]) == 1:
        return False
    h, w = _dimensions(_normalize(winner["cells"]))
    if not any(n["area"] < winner["area"] and
               max(_dimensions(_normalize(n["cells"]))) > max(h, w)
               for n in analysis["networks"]):
        return False
    fragmented = False
    for mask in (analysis["left"], analysis["right"]):
        local = np.zeros_like(mask)
        for r, c in winner["cells"]:
            local[r, c] = mask[r, c]
        fragmented |= len(_components(local)) > 1
    target = np.tile(np.array([b[2] for b in analysis["members"]], dtype=np.int8),
                     (analysis["maximum"], 1))
    winner_colors = {b[2] for b in analysis["members"]}
    for mode in ("first", "second"):
        alone = _counterfactual(analysis, mode)
        if alone is None or not winner_colors.isdisjoint(map(int, alone[0])):
            return False
    return fragmented and all(not _same(_counterfactual(analysis, m), target)
                              for m in ("first", "second", "xor", "and", "sum", "box", "all"))


def _make_pair(rng, lesson, shift, orientation, structural):
    bg, ink, sep = structural
    for _ in range(100):
        areas, demands, styles, modes = _sample_spec(rng, lesson, shift)
        shapes = [_shape(rng, area, style) for area, style in zip(areas, styles)]
        try:
            shape, networks, beacon_cells = _place(rng, shapes, demands, orientation,
                                                   tie_lesson=lesson == "tie")
        except RuntimeError:
            continue
        first, second = [np.full(shape, bg, dtype=np.int8) for _ in range(2)]
        colors = [c for c in range(10) if c not in structural]
        rng.shuffle(colors)
        labels_by_network = []
        for cells, labels, mode in zip(networks, beacon_cells, modes):
            tags = _layer_masks(rng, cells, labels, mode)
            for (r, c), tag in zip(cells, tags):
                if tag & 1:
                    first[r, c] = ink
                if tag & 2:
                    second[r, c] = ink
            records = []
            for r, c in labels:
                choices = [panel for panel in (first, second) if panel[r, c] != bg]
                choices[int(rng.integers(len(choices)))][r, c] = color = colors.pop()
                records.append((r, c, color))
            labels_by_network.append(records)
        if orientation == "vertical":
            stripe = np.full((shape[0], 1), sep, dtype=np.int8)
            grid = np.concatenate((first, stripe, second), axis=1)
        else:
            stripe = np.full((1, shape[1]), sep, dtype=np.int8)
            grid = np.concatenate((first, stripe, second), axis=0)
        maximum = max(areas)
        survivors = sorted((b for area, records in zip(areas, labels_by_network)
                            if area == maximum for b in records), key=lambda b: b[1])
        expected = np.tile(np.array([b[2] for b in survivors], dtype=np.int8), (maximum, 1))
        analysis = _extract(grid)
        if lesson == "assembly":
            if not _assembly_witness(analysis):
                continue
            members = analysis["members"]
            if [b[2] for b in sorted(members, key=lambda b: (b[0], b[1]))] == [b[2] for b in members]:
                continue
            if sorted(b[2] for b in members) == [b[2] for b in members]:
                continue
        _require(np.array_equal(expected, solve(grid)), "Construction and geometry disagree")
        return {"input": grid, "output": expected}
    raise RuntimeError("Bounded pair generation exhausted")


def _check_count(value, lo, hi, name):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
             and lo <= int(value) <= hi, f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _check_count(num_train, 2, 10, "num_train")
    num_test = _check_count(num_test, 1, 4, "num_test")
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, size=3, replace=False)))
    varying_palette = bool(rng.integers(2))
    orientation_mode = str(rng.choice(["vertical", "horizontal", "both"]))
    shift = str(rng.choice(["count", "area", "joint"]))
    lessons = ["assembly", "tie"]
    rng.shuffle(lessons)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = lessons[index] if split == "train" and index < 2 else split
            if orientation_mode == "both":
                orientation = ("vertical", "horizontal")[index] if split == "train" and index < 2 else str(rng.choice(["vertical", "horizontal"]))
            else:
                orientation = orientation_mode
            if varying_palette:
                structural = tuple(map(int, rng.choice(10, size=3, replace=False)))
                # Guarantee that training already demonstrates structural changes.
                if split == "train" and index == 1:
                    old = _extract(puzzle["train"][0]["input"])
                    while structural == (old["background"], old["ink"], old["separator"]):
                        structural = tuple(map(int, rng.choice(10, size=3, replace=False)))
            else:
                structural = palette
            for _ in range(24):
                pair = _make_pair(rng, lesson, shift, orientation, structural)
                signature = _key(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("Could not obtain distinct inputs")
            seen.add(signature)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "Splits must be lists")
    nt = _check_count(len(puzzle["train"]), 2, 10, "training count")
    ns = _check_count(len(puzzle["test"]), 1, 4, "test count")
    _require(nt > ns, "Training count must exceed test count")
    analyses = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            signature = _key(grid)
            _require(signature not in seen, "Inputs must be distinct")
            seen.add(signature)
            a = _extract(grid)
            _require(np.array_equal(solve(grid), output), "Incorrect output")
            _require(output.shape == (a["maximum"], len(a["members"])), "Wrong unary dimensions")
            _require(all(np.all(output[:, i] == b[2]) for i, b in enumerate(a["members"])),
                     "Wrong beacon membership or column color")
            analyses[split].append(a)
    train = analyses["train"]
    _require(any(_assembly_witness(a) for a in train), "Missing assembled coalition teaching evidence")
    _require(any(len(a["winners"]) == 2 and sorted(len(n["beacons"]) for n in a["winners"]) == [1, 2]
                 and len(a["networks"]) > 2 for a in train), "Missing unequal-membership tie lesson")
    for mode in ("first", "second", "xor", "and", "sum", "box", "all", "eight"):
        _require(any(not _same(_counterfactual(a, mode), pair["output"])
                     for a, pair in zip(train, puzzle["train"])), f"Training does not reject shortcut {mode}")
    _require(any([b[2] for b in sorted(a["members"], key=lambda b: (b[0], b[1]))]
                 != [b[2] for b in a["members"]] for a in train), "Row and column order never differ")
    _require(any(sorted(b[2] for b in a["members"]) != [b[2] for b in a["members"]]
                 for a in train), "Numeric and spatial order never differ")
    _require(any(len(a["members"]) == 3 and any(
                 {a["members"][0][2], a["members"][-1][2]} == {b[2] for b in n["beacons"]}
                 for n in a["winners"]) for a in train), "Missing interleaved-membership order lesson")
    orientations = {a["orientation"] for a in train}
    palettes = {(a["background"], a["ink"], a["separator"]) for a in train}
    for a in analyses["test"]:
        _require(a["orientation"] in orientations, "Unshown separator orientation")
        roles = (a["background"], a["ink"], a["separator"])
        _require(len(palettes) > 1 or roles in palettes, "Unshown changing-palette convention")
        _require(len(a["networks"]) > max(len(t["networks"]) for t in train)
                 or a["maximum"] > max(t["maximum"] for t in train), "No structural extrapolation")


def _puzzle_key(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def _regression_input(networks, beacons, first_cells=None, second_cells=None, shape=(12, 14)):
    """Small hand-authored occupancy cases, separate from random sampling."""
    full = set().union(*map(set, networks))
    first_cells = full if first_cells is None else set(first_cells)
    second_cells = full if second_cells is None else set(second_cells)
    first, second = [np.zeros(shape, dtype=np.int8) for _ in range(2)]
    for r, c in first_cells:
        first[r, c] = 5
    for r, c in second_cells:
        second[r, c] = 5
    for r, c, color in beacons:
        if first[r, c]:
            first[r, c] = color
        else:
            second[r, c] = color
    return np.concatenate((first, np.full((shape[0], 1), 9, dtype=np.int8), second), axis=1)


def self_test():
    """Bounded structural, semantic and diversity checks on exactly 42 base seeds."""
    signatures, shapes, outputs, palettes, geometries = set(), set(), set(), set(), set()
    areas, counts, ties, members, relations, orientations = set(), set(), set(), set(), set(), set()
    diagonal_cases = rings = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _extract(pair["input"])
                shapes.add(pair["input"].shape)
                outputs.add(pair["output"].shape)
                palettes.add((a["background"], a["ink"], a["separator"]))
                orientations.add(a["orientation"])
                areas.add(a["maximum"])
                counts.add(len(a["networks"]))
                ties.add(len(a["winners"]))
                members.update(len(n["beacons"]) for n in a["networks"])
                if not _same(_counterfactual(a, "eight"), pair["output"]):
                    diagonal_cases += 1
                for n in a["networks"]:
                    cells = n["cells"]
                    geometries.add(_normalize(cells))
                    l = np.array([a["left"][r, c] for r, c in cells])
                    r = np.array([a["right"][r, c] for r, c in cells])
                    if not l.any() or not r.any():
                        relations.add("one_layer")
                    if np.array_equal(l, r):
                        relations.add("identical")
                    if not (l & r).any():
                        relations.add("disjoint")
                    if (l & r).any() and (l & ~r).any() and (~l & r).any():
                        relations.add("mixed")
                    points = set(_normalize(cells))
                    h, w = _dimensions(points)
                    # A contained background cell in the normalized bounding box.
                    empty = np.ones((h + 2, w + 2), dtype=bool)
                    for y, x in points:
                        empty[y + 1, x + 1] = False
                    rings += len(_components(empty)) > 1
        if seed in (0, 17, 41):
            _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "Nondeterministic generation")
    _require(len(signatures) == 42, "Repeated puzzle across seeds")
    _require(len(shapes) >= 25 and len(outputs) >= 15 and len(geometries) >= 70,
             "Insufficient structural diversity")
    _require(len(palettes) >= 25 and len(areas) >= 12 and counts >= {2, 3, 4, 5}, "Insufficient role/area/count diversity")
    _require(ties >= {1, 2, 3} and members >= {1, 2, 3}, "Insufficient tie/membership diversity")
    _require(orientations == {"vertical", "horizontal"} and len(relations) == 4,
             "Missing layer or orientation variation")
    _require(diagonal_cases > 0 and rings > 0, "Missing topology variation")

    # A second-layer bridge unites two labelled fragments and reverses the
    # winner. Overlap is counted once; the intact competitor has eight cells.
    path = {(2, c) for c in range(1, 10)}
    competitor = {(r, 11) for r in range(4, 12)}
    bridge = _regression_input(
        [path, competitor], [(2, 1, 2), (2, 9, 8), (7, 11, 3)],
        first_cells=(path - {(2, 5)}) | competitor,
        second_cells={(2, 2), (2, 5), (2, 8)} | competitor, shape=(14, 14))
    _require(np.array_equal(solve(bridge), np.array([[2, 8]] * 9)), "Bridge/overlap winner reversal")
    for mode in ("first", "second"):
        _require(np.array_equal(_counterfactual(_extract(bridge), mode), np.array([[3]] * 8)),
                 "Omitting a layer should reverse the winner")
    # A ring's hole stays empty, two diagonally touching networks stay separate,
    # and largest-component ties keep every member in GLOBAL column order.
    ring = {(r, c) for r in range(1, 4) for c in range(1, 4) if r in (1, 3) or c in (1, 3)}
    line = {(4, c) for c in range(4, 12)}
    tiny = {(8, 2), (9, 2)}
    example = _regression_input([ring, line, tiny], [(3, 1, 8), (1, 3, 2), (4, 7, 4), (9, 2, 3)])
    _require(np.array_equal(solve(example), np.array([[8, 2, 4]] * 8)), "Ring/tie/diagonal regression")
    _require(not _same(_counterfactual(_extract(example), "eight"), solve(example)), "Diagonal witness failed")
    # Area 30 and an empty second layer are valid beyond the sampler's maximum.
    big = {(r, c) for r in range(1, 6) for c in range(1, 7)}
    small = {(8, 9), (9, 9), (10, 9)}
    boundary = _regression_input([big, small], [(2, 2, 6), (9, 9, 7)], second_cells=[])
    _require(np.array_equal(solve(boundary), np.full((30, 1), 6)), "Area-30/blank-layer boundary")
    # Interleaved memberships, so grouping each coalition's labels first is wrong.
    upper = {(2, c) for c in range(1, 10)}
    lower = {(6, c) for c in range(3, 12)}
    interleave = _regression_input([upper, lower], [(2, 1, 7), (2, 9, 2), (6, 5, 4)])
    _require(np.array_equal(solve(interleave), np.array([[7, 4, 2]] * 9)), "Global membership order")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(10000 + 10 * nt + ns, nt, ns))
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {nt}/{ns}")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "input_shapes": len(shapes),
            "output_shapes": len(outputs), "network_silhouettes": len(geometries),
            "structural_palettes": len(palettes), "maximum_areas": sorted(areas),
            "network_counts": sorted(counts), "tie_counts": sorted(ties),
            "members_per_network": sorted(members), "layer_relations": sorted(relations),
            "diagonal_sensitive_pairs": diagonal_cases, "networks_with_holes": int(rings)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
