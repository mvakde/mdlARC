"""Newly sealed reservoirs: oriented, registered wall stencils.

No files, example tables, or random state are consulted by solve().
Generation and all tests are bounded and retain no execution traces.
"""
from collections import deque
import hashlib
import numpy as np


SIDES = ("T", "R", "B", "L")
NORMALS = ((-1, 0), (0, 1), (1, 0), (0, -1))
STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 169192863, 324995394, 930168104, 204148907, 315352447,
              143779368, 802630497, 904209255, 953236791, 690351487,
              807050296, 198578963, 914747173, 63442562, 255247475,
              696933968, 323112438, 345668975, 922633383, 443093722,
              392209618, 529515945, 895171444, 259423934, 196647470,
              646952554, 853660087, 700934441, 799410177, 922568231,
              658059955, 521126623, 962556783, 973484002, 787169851,
              285675410, 664504559, 654907910, 785730649, 872123325,
              624187961)


def _transform(array, operation):
    base = np.fliplr(array) if operation >= 4 else array
    return np.rot90(base, operation % 4).copy()


def _side_map(operation):
    matrix = np.eye(2, dtype=int)
    rotation = np.array([[0, -1], [1, 0]], dtype=int)
    for _ in range(operation % 4):
        matrix = rotation @ matrix
    if operation >= 4:
        matrix = matrix @ np.diag([1, -1])
    return {side: SIDES[NORMALS.index(tuple(matrix @ normal))]
            for side, normal in zip(SIDES, NORMALS)}


MAPS = tuple(_side_map(k) for k in range(8))


def _matching_operation(guides, controls):
    matches = [k for k, mapping in enumerate(MAPS)
               if all(controls.get(mapping[s]) == color for s, color in guides.items())]
    if len(matches) != 1:
        raise ValueError("Two distinct adjacent identities must fix one symmetry")
    return matches[0]


def _line(grid, side):
    return {"T": grid[0, :], "R": grid[:, -1],
            "B": grid[-1, :], "L": grid[:, 0]}[side]


def _shift(cells, dr, dc=0):
    return {(r + dr, c + dc) for r, c in cells}


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in STEPS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _exterior(walls, bounds):
    top, bottom, left, right = bounds
    start = top, left
    reached, queue = {start}, deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            q = r + dr, c + dc
            if (top <= q[0] <= bottom and left <= q[1] <= right
                    and q not in walls and q not in reached):
                reached.add(q)
                queue.append(q)
    return reached


def _regions(a, b):
    walls = a | b
    top, bottom, left, right = _bbox(walls)
    bounds = top - 1, bottom + 1, left - 1, right + 1
    universe = {(r, c) for r in range(top, bottom + 1)
                for c in range(left, right + 1)}
    holes = universe - walls - _exterior(walls, bounds)
    # Both tests use the same padded box. Every stencil is a subset of walls.
    # A merged empty component therefore lies in one component of each stencil.
    newly = holes & _exterior(a, bounds) & _exterior(b, bounds)
    return walls, newly, holes - newly


def _paint_result(a, b, palette, fill="new", blend="union"):
    bg, fg, ink = palette[:3]
    if blend == "xor":
        a, b = a ^ b, set()
    elif blend == "anchor":
        b = set()
    elif blend == "mover":
        a = set()
    elif blend in ("anchor_opaque", "mover_opaque"):
        front, back = (a, b) if blend == "anchor_opaque" else (b, a)
        t, bot, l, right = _bbox(front)
        a = front | {p for p in back if not (t <= p[0] <= bot and l <= p[1] <= right)}
        b = set()
    walls, newly, private = _regions(a, b)
    ink_cells = newly if fill == "new" else newly | private if fill == "all" else set()
    top, bottom, left, right = _bbox(walls)
    result = np.full((bottom - top + 1, right - left + 1), bg, dtype=np.int8)
    for r, c in walls:
        result[r - top, c - left] = fg
    for r, c in ink_cells:
        result[r - top, c - left] = ink
    return result


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer)
            or any(n < 1 or n > 30 for n in grid.shape)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return grid


def _guide_description(cells, shape, color):
    if not cells or any(r in (0, shape[0] - 1) or c in (0, shape[1] - 1) for r, c in cells):
        return None
    t, b, l, r = _bbox(cells)
    if t == b and len(cells) == r - l + 1 and len(cells) >= 3:
        return {"axis": "H", "line": t, "start": l, "end": r, "color": color, "cells": cells}
    if l == r and len(cells) == b - t + 1 and len(cells) >= 3:
        return {"axis": "V", "line": l, "start": t, "end": b, "color": color, "cells": cells}
    return None


def _panel_candidates(grid, bg, controls):
    descriptions = []
    for color in controls.values():
        for cells in _components({tuple(map(int, p)) for p in np.argwhere(grid == color)}):
            description = _guide_description(cells, grid.shape, color)
            if description is not None:
                descriptions.append(description)
    candidates = []
    for h in [d for d in descriptions if d["axis"] == "H"]:
        for v in [d for d in descriptions if d["axis"] == "V"]:
            if h["color"] == v["color"]:
                continue
            hside = "T" if v["start"] == h["line"] + 1 else "B" if v["end"] == h["line"] - 1 else None
            vside = "L" if v["line"] == h["start"] - 1 else "R" if v["line"] == h["end"] + 1 else None
            if hside is None or vside is None:
                continue
            top, bottom, left, right = v["start"], v["end"], h["start"], h["end"]
            t0, b0 = top - (hside == "T"), bottom + (hside == "B")
            l0, r0 = left - (vside == "L"), right + (vside == "R")
            if t0 < 2 or l0 < 2 or b0 > grid.shape[0] - 3 or r0 > grid.shape[1] - 3:
                continue
            payload = grid[top:bottom + 1, left:right + 1].copy()
            if min(payload.shape) < 3 or set(np.unique(payload)) & set(controls.values()):
                continue
            guide_cells = h["cells"] | v["cells"]
            payload_cells = {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)}
            box = {(r, c) for r in range(t0, b0 + 1) for c in range(l0, r0 + 1)}
            moat = {(r, c) for r in range(t0 - 1, b0 + 2) for c in range(l0 - 1, r0 + 2)} - box
            if any(grid[p] != bg for p in (box - guide_cells - payload_cells) | moat):
                continue
            candidates.append({"payload": payload, "guides": {hside: h["color"], vside: v["color"]},
                               "box": box, "bounds": (top, bottom, left, right)})
    if len(candidates) != 2 or candidates[0]["box"] & candidates[1]["box"]:
        raise ValueError("Expected two uniquely guide-defined, separated stencils")
    accounted = candidates[0]["box"] | candidates[1]["box"]
    if any(grid[r, c] != bg and (r, c) not in accounted
           for r in range(1, grid.shape[0] - 1) for c in range(1, grid.shape[1] - 1)):
        raise ValueError("Unexpected material outside the two stencil rectangles")
    return candidates


def _decode(input_grid):
    grid = _grid(input_grid)
    if min(grid.shape) < 7:
        raise ValueError("Canvas too small for two stencils and the orientation key")
    bg = int(grid[0, 0])
    if any(int(grid[p]) != bg for p in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("All four frame corners are background")
    controls = {}
    for side in SIDES:
        values = _line(grid, side)[1:-1]
        if np.all(values == bg):
            continue
        if len(np.unique(values)) != 1:
            raise ValueError("Controls must be complete clean outer lines")
        controls[side] = int(values[0])
    if (len(controls) != 2 or len(set(controls.values())) != 2
            or {SIDES.index(s) % 2 for s in controls} != {0, 1}):
        raise ValueError("Expected two differently colored adjacent outer controls")
    panels = _panel_candidates(grid, bg, controls)
    other = set(map(int, np.unique(grid))) - {bg, *controls.values()}
    if len(other) != 2:
        raise ValueError("Expected one wall color and one registration/ink color")
    for panel in panels:
        k = _matching_operation(panel["guides"], controls)
        panel["operation"] = k
        panel["oriented"] = _transform(panel["payload"], k)
    assignments = []
    for ink in sorted(other):
        fg = next(c for c in other if c != ink)
        pieces, axis, length = [], None, None
        for panel in panels:
            payload = panel["oriented"]
            bar = {tuple(map(int, p)) for p in np.argwhere(payload == ink)}
            if not 2 <= len(bar) <= 4:
                break
            t, bot, l, right = _bbox(bar)
            local_axis = 0 if t == bot else 1 if l == right else None
            if local_axis is None or len(bar) != max(bot - t, right - l) + 1:
                break
            if axis is not None and (axis != local_axis or length != len(bar)):
                break
            axis, length = local_axis, len(bar)
            origin = min(p[axis] for p in bar), min(p[1 - axis] for p in bar)
            cells = {(int(p[axis]) - origin[0], int(p[1 - axis]) - origin[1])
                     for p in np.argwhere(payload == fg)}
            negative = any((-1, c) in cells for c in range(length))
            positive = any((1, c) in cells for c in range(length))
            if negative == positive or not cells:
                break
            pieces.append({"negative": negative, "cells": cells, "panel": panel})
        if len(pieces) == 2 and pieces[0]["negative"] != pieces[1]["negative"]:
            pieces.sort(key=lambda p: not p["negative"])
            assignments.append({"palette": (bg, fg, ink), "axis": axis, "length": length,
                                "pieces": pieces, "panels": panels, "controls": controls})
    if len(assignments) != 1:
        raise ValueError("Registration color, axis, and opposite backing sides must be unique")
    return assignments[0]


def _predict(info, seam=-1, fill="new", blend="union"):
    a = info["pieces"][0]["cells"]
    b = _shift(info["pieces"][1]["cells"], seam)
    result = _paint_result(a, b, info["palette"], fill, blend)
    return result.T.copy() if info["axis"] else result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Match local identities, register walls, and fill only cooperative holes."""
    return _predict(_decode(input_grid))


def _boundary(region):
    return {(r + dr, c + dc) for r, c in region for dr, dc in STEPS} - region


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _make_geometry(rng, teaching=False):
    """Build small irregular wall networks, then split their seam ownership.

    Coordinates here are construction-only. solve() never calls this function.
    """
    length = int(rng.integers(2, 5))
    widths = [int(rng.integers(1, 4)), int(rng.integers(1, 4))]
    while sum(widths) + length > 9:
        widths[int(np.argmax(widths))] -= 1
    sides = [-1, 1] if teaching or rng.random() < .75 else [_pick(rng, [-1, 1])]
    open_side = _pick(rng, sides) if teaching or (len(sides) == 2 and rng.random() < .35) else None
    walls, regions = set(), []
    for side in sides:
        up, down = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        max_width = widths[0 if side == -1 else 1]
        constant_width = bool(rng.random() < .35)
        width = max_width if constant_width else int(rng.integers(1, max_width + 1))
        region = set()
        for row in range(-up, down + 1):
            if not constant_width:
                width = min(max_width, max(1, width + int(rng.integers(-1, 2))))
            for d in range(width):
                region.add((row, -2 - d if side == -1 else length + 1 + d))
        wall = _boundary(region)
        # A missing side wall makes an open bay, not another fill target.
        if side == open_side:
            edge = min(c for r, c in region if r == 0) - 1 if side == -1 else max(c for r, c in region if r == 0) + 1
            wall.discard((0, edge))
        # Partitions create private upper/lower pockets within the larger wing.
        if rng.random() < .55:
            dividers = ([r for r in (-2, 1, 2) if -up < r < down])
            if dividers:
                row = _pick(rng, dividers)
                wall.update(p for p in region if p[0] == row)
        # A full rib can split a wide wing into two independently sealed rooms;
        # partial ribs instead change a chamber's boundary and free area.
        if constant_width and max_width == 3 and rng.random() < .55:
            column = -3 if side == -1 else length + 2
            wall.update((row, column) for row in range(-up, down + 1))
        elif max_width == 3 and rng.random() < .5:
            row = int(rng.integers(-up, down + 1))
            points = sorted(p for p in region if p[0] == row)
            if len(points) == 3:
                wall.add(points[int(rng.integers(3))])
        walls.update(wall)
        regions.append(region)
    walls.update((r, c) for r in (-1, 0) for c in range(length))
    if teaching:
        # Copy a cooperative empty region as a private hole above the seam.
        # Equal shape and area, opposite treatment: neither hole-size ranking
        # nor a shape lookup can explain the teaching answer.
        upper, lower = {p for p in walls if p[0] < 0}, {p for p in walls if p[0] >= 0}
        new_before = _regions(upper, lower)[1]
        if not new_before:
            return None
        cavity = _pick(rng, _components(new_before))
        ct, cb, cl, cr = _bbox(cavity)
        hole_top = min(r for r, c in walls) - (cb - ct + 1) - 2
        private_cavity = _shift(cavity, hole_top - ct, 1 - cl)
        rim = _boundary(private_cavity)
        walls.update(rim)
        # The column-zero stroke is outside the copied cavity, connecting its
        # left rim to the registration support without altering either cavity.
        lowest_left = max(r for r, c in rim if c == 0)
        walls.update((r, 0) for r in range(lowest_left, 0))
    elif rng.random() < .8:
        top = min(r for r, c in walls) - 3
        left = int(rng.integers(length))
        walls.update((r, c) for r in range(top, top + 3) for c in range(left, left + 3)
                     if r in (top, top + 2) or c in (left, left + 2))
        walls.update((r, left) for r in range(top + 2, 0))
    a = {p for p in walls if p[0] < 0}
    b = {p for p in walls if p[0] >= 0}
    # Shared ordinary pixels deliberately occur as well as collision-free joins.
    # No ordinary pixel ever occupies its own registration bar.
    if teaching or rng.random() < .6:
        for cells, row in ((a, 0), (b, -1)):
            options = sorted(p for p in walls if p[0] == row and not 0 <= p[1] < length)
            if options:
                chosen = rng.choice(len(options), int(rng.integers(1, len(options) + 1)), replace=False)
                cells.update(options[int(i)] for i in chosen)
    merged, newly, private = _regions(a, b)
    if not newly or (teaching and not private):
        return None
    # B's local marker is at row 0, its desired assembled marker is at row -1.
    return a, _shift(b, 1), length


def _payload(rng, cells, length, palette):
    bg, fg, ink = palette[:3]
    bar = {(0, c) for c in range(length)}
    t, b, l, r = _bbox(cells | bar)
    # At most one extra blank strip per axis, on a randomly chosen side.
    if rng.random() < .6:
        if rng.integers(2):
            t -= 1
        else:
            b += 1
    if rng.random() < .6:
        if rng.integers(2):
            l -= 1
        else:
            r += 1
    payload = np.full((b - t + 1, r - l + 1), bg, dtype=np.int8)
    for y, x in cells:
        payload[y - t, x - l] = fg
    for y, x in bar:
        payload[y - t, x - l] = ink
    return payload


def _guide_box(payload, guides, bg):
    h, w = payload.shape
    t, l = int("T" in guides), int("L" in guides)
    result = np.full((h + 1, w + 1), bg, dtype=np.int8)
    result[t:t + h, l:l + w] = payload
    for side, color in guides.items():
        if side == "T":
            result[0, l:l + w] = color
        elif side == "B":
            result[-1, l:l + w] = color
        elif side == "L":
            result[t:t + h, 0] = color
        else:
            result[t:t + h, -1] = color
    return result


def _stage(rng, boxes, controls, bg):
    if rng.integers(2):
        boxes = boxes[::-1]
    (ah, aw), (bh, bw) = [box.shape for box in boxes]
    modes = []
    if aw + bw + 1 <= 26:
        modes.append("horizontal")
    if ah + bh + 1 <= 26:
        modes.append("vertical")
    if ah + bh + 1 <= 26 and aw + bw + 1 <= 26:
        modes.extend(["diagonal", "anti_diagonal"])
    if not modes or max(ah, aw, bh, bw) > 26:
        return None
    mode = _pick(rng, modes)
    if mode == "horizontal":
        gap = int(rng.integers(1, min(4, 26 - aw - bw) + 1))
        positions = [(0, 0), (int(rng.integers(0, min(5, 26 - bh) + 1)), aw + gap)]
    elif mode == "vertical":
        gap = int(rng.integers(1, min(4, 26 - ah - bh) + 1))
        positions = [(0, 0), (ah + gap, int(rng.integers(0, min(5, 26 - bw) + 1)))]
    else:
        gy = int(rng.integers(1, min(4, 26 - ah - bh) + 1))
        gx = int(rng.integers(1, min(4, 26 - aw - bw) + 1))
        positions = [(0, 0), (ah + gy, aw + gx)] if mode == "diagonal" else [(bh + gy, 0), (0, aw + gx)]
    h = max(y + box.shape[0] for (y, x), box in zip(positions, boxes))
    w = max(x + box.shape[1] for (y, x), box in zip(positions, boxes))
    if max(h, w) > 26:
        return None
    H, W = int(rng.integers(h + 4, min(30, h + 8) + 1)), int(rng.integers(w + 4, min(30, w + 8) + 1))
    py, px = int(rng.integers(2, H - h - 1)), int(rng.integers(2, W - w - 1))
    grid = np.full((H, W), bg, dtype=np.int8)
    for box, (y, x) in zip(boxes, positions):
        grid[py + y:py + y + box.shape[0], px + x:px + x + box.shape[1]] = box
    for side, color in controls.items():
        _line(grid, side)[1:-1] = color
    return grid


def _corner(operation):
    return tuple(sorted((MAPS[operation]["T"], MAPS[operation]["L"])))


def _corner_signature(operations):
    return tuple(sorted(_corner(k) for k in operations))


def _asymmetric(grid):
    return len({_key(_transform(grid, k)) for k in range(8)}) == 8


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _make_pair(rng, palette, key_operation, held, is_test=False, teaching=False):
    choices = [(a, b) for a in range(8) for b in range(8)
               if (_corner_signature((a, b)) == held) == is_test
               and (not teaching or (a >= 4) != (b >= 4))]
    for _ in range(120):
        geometry = _make_geometry(rng, teaching)
        if geometry is None:
            continue
        a, b, length = geometry
        sources = _pick(rng, choices)
        boxes = []
        for cells, source in zip((a, b), sources):
            payload = _transform(_payload(rng, cells, length, palette), source)
            guides = {MAPS[source]["T"]: palette[3], MAPS[source]["L"]: palette[4]}
            boxes.append(_guide_box(payload, guides, palette[0]))
        controls = {MAPS[key_operation]["T"]: palette[3], MAPS[key_operation]["L"]: palette[4]}
        grid = _stage(rng, boxes, controls, palette[0])
        if grid is None:
            continue
        expected = _transform(_paint_result(a, _shift(b, -1), palette), key_operation)
        if teaching and not _asymmetric(expected):
            continue
        try:
            info = _decode(grid)
        except ValueError:
            continue
        if not np.array_equal(_predict(info), expected):
            raise AssertionError("Construction and input-only geometry disagree")
        if teaching:
            alternatives = [(0, "new", "union"), (1, "new", "union"), (-1, "new", "xor"),
                            (-1, "new", "anchor_opaque"), (-1, "new", "mover_opaque")]
            if any(np.array_equal(_predict(info, *alt), expected) for alt in alternatives):
                continue
        return {"input": grid, "output": expected}
    raise RuntimeError("Bounded reservoir construction exhausted")


def _count(value, name, lower, upper):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f"{name} must be an integer in {lower}..{upper}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    stable = bool(rng.random() < .45)
    palette = tuple(map(int, rng.choice(10, 5, replace=False)))
    key_operation = int(rng.integers(8))
    corners = sorted({_corner(k) for k in range(8)})
    held = _pick(rng, [(a, b) for i, a in enumerate(corners) for b in corners[i:]])
    first = _make_pair(rng, palette, key_operation, held, teaching=True)
    # Controlled correspondence reversal: the entire interior is identical.
    second_input = first["input"].copy()
    controls = _decode(second_input)["controls"]
    for side, color in controls.items():
        _line(second_input, side)[1:-1] = palette[4] if color == palette[3] else palette[3]
    # This is an input-derived prediction; no seed metadata chooses the answer.
    second = {"input": second_input, "output": solve(second_input)}
    puzzle, used = {"train": [first, second], "test": []}, {_key(first["input"]), _key(second_input)}
    for split, number in (("train", nt - 2), ("test", ne)):
        for _ in range(number):
            for attempt in range(24):
                local_palette = palette if stable else tuple(map(int, rng.choice(10, 5, replace=False)))
                pair = _make_pair(rng, local_palette, int(rng.integers(8)), held, is_test=split == "test")
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded distinct-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    a = info["pieces"][0]["cells"]
    b = _shift(info["pieces"][1]["cells"], -1)
    walls, newly, private = _regions(a, b)
    blank_edges = sum(np.all(_line(p["payload"], s) == info["palette"][0])
                      for p in info["panels"] for s in SIDES)
    return {"new_cells": len(newly), "private_cells": len(private),
            "new_chambers": len(_components(newly)), "private_chambers": len(_components(private)),
            "overlap": len(a & b), "wall_cells": len(walls), "axis": info["axis"],
            "length": info["length"], "blank_edges": int(blank_edges),
            "operations": tuple(p["operation"] for p in info["panels"]),
            "corners": tuple(sorted(tuple(sorted(p["guides"])) for p in info["panels"]))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _key(inp)
            assert key not in seen, "Duplicate input"
            seen.add(key)
            info = _decode(inp)
            assert np.array_equal(_predict(info), out), "Wrong orientation, registration, walls, ink, or crop"
            feature = _features(info)
            bg, fg, ink = info["palette"]
            assert np.count_nonzero(out == fg) == feature["wall_cells"], "Wall union must preserve every wall pixel"
            assert np.count_nonzero(out == ink) == feature["new_cells"], "Only cooperative holes receive ink"
            assert set(np.unique(out)) <= {bg, fg, ink}
            assert feature["new_cells"] > 0, "Each example must exercise cooperative closure"
            assert all(np.any(_line(out, side) == fg) for side in SIDES), "Tight crop is defined by walls"
            infos[split].append(info)
    a, b = puzzle["train"][:2]
    ga, gb = np.asarray(a["input"]), np.asarray(b["input"])
    assert ga.shape == gb.shape and np.array_equal(ga[1:-1, 1:-1], gb[1:-1, 1:-1])
    ca, cb = infos["train"][0]["controls"], infos["train"][1]["controls"]
    assert set(ca) == set(cb) and all(ca[s] != cb[s] for s in ca)
    assert not np.array_equal(a["output"], b["output"]), "Guide identity contrast must change the answer"
    first = infos["train"][0]
    assert _features(first)["private_cells"] > 0
    assert _features(first)["overlap"] > 0
    fa = first["pieces"][0]["cells"]
    fb = _shift(first["pieces"][1]["cells"], -1)
    _, newly, private = _regions(fa, fb)
    def normalized_component(cells):
        t, bot, l, right = _bbox(cells)
        return frozenset(_shift(cells, -t, -l))
    new_shapes = {normalized_component(c) for c in _components(newly)}
    old_shapes = {normalized_component(c) for c in _components(private)}
    assert new_shapes & old_shapes, "Training needs congruent old/new holes with different treatment"
    alternatives = [(0, "new", "union"), (1, "new", "union"), (-1, "none", "union"),
                    (-1, "all", "union"), (-1, "new", "xor"), (-1, "new", "anchor"),
                    (-1, "new", "mover"), (-1, "new", "anchor_opaque"), (-1, "new", "mover_opaque")]
    assert all(not np.array_equal(_predict(first, *alternative), a["output"]) for alternative in alternatives)
    # A held-out combination of the two local guide corners, observable without
    # wall-generation labels. The rule itself does not depend on this split.
    train_corners = {_features(i)["corners"] for i in infos["train"]}
    test_corners = {_features(i)["corners"] for i in infos["test"]}
    assert len(test_corners) == 1 and train_corners.isdisjoint(test_corners)


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                shape, payload = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(payload)
    return digest.hexdigest()


def _topology_regression():
    """A hand-specified three-room map, outside the wing sampler's templates.

    One large private hole is subdivided by the other stencil. Both resulting
    holes must stay unfilled, as must the separate small private hole.
    """
    def ring(top, bottom, left, right):
        return {(r, c) for r in range(top, bottom + 1) for c in range(left, right + 1)
                if r in (top, bottom) or c in (left, right)}

    walls = set().union(*(ring(-2, 2, left, left + 2) for left in (-5, 4, 8)))
    walls |= ring(-7, -3, 5, 10) | ring(-6, -4, 0, 2)
    walls |= {(r, c) for r in (-1, 0) for c in range(3)}
    a, b = {p for p in walls if p[0] < 0}, {p for p in walls if p[0] >= 0}
    divider = {(-5, c) for c in range(6, 10)}
    b |= divider
    walls |= divider
    expected_new = {(r, c) for r in (-1, 0, 1) for c in (-4, 5, 9)}
    expected_private = {(-5, 1)} | {(r, c) for r in (-6, -4) for c in range(6, 10)}
    actual_walls, new, private = _regions(a, b)
    assert actual_walls == walls and new == expected_new and private == expected_private
    palette, rng = (0, 5, 8, 2, 3), np.random.default_rng(9182)
    boxes = []
    for cells, operation in ((a, 2), (_shift(b, 1), 4)):
        payload = _transform(_payload(rng, cells, 3, palette), operation)
        guides = {MAPS[operation]["T"]: palette[3], MAPS[operation]["L"]: palette[4]}
        boxes.append(_guide_box(payload, guides, palette[0]))
    target = 5
    controls = {MAPS[target]["T"]: palette[3], MAPS[target]["L"]: palette[4]}
    scene = _stage(rng, boxes, controls, palette[0])
    assert scene is not None
    top, bottom, left, right = _bbox(walls)
    expected = np.zeros((bottom - top + 1, right - left + 1), dtype=np.int8)
    for r, c in walls:
        expected[r - top, c - left] = palette[1]
    for r, c in expected_new:
        expected[r - top, c - left] = palette[2]
    expected = _transform(expected, target)
    for operation in range(8):
        assert np.array_equal(solve(_transform(scene, operation)), _transform(expected, operation))
    permutation = np.roll(np.arange(10, dtype=np.int8), 3)
    assert np.array_equal(solve(permutation[scene]), permutation[expected])


def self_test():
    """42 full puzzles, small mutation checks, and bounded count regressions."""
    _topology_regression()
    signatures, input_shapes, output_shapes, palettes, geometries = set(), set(), set(), set(), set()
    features, operations, corner_pairs = [], set(), set()
    example = None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        if seed == 0:
            example = puzzle
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                feature = _features(info)
                features.append(feature)
                operations.update(feature["operations"])
                corner_pairs.add(feature["corners"])
                input_shapes.add(pair["input"].shape)
                output_shapes.add(pair["output"].shape)
                palettes.add(info["palette"])
                wall_mask = (pair["output"] == info["palette"][1]).astype(np.int8)
                geometries.add(min(_key(_transform(wall_mask, k)) for k in range(8)))
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
    assert len(signatures) == 42 and len(geometries) >= 100
    assert len(input_shapes) >= 40 and len(output_shapes) >= 35 and len(palettes) >= 35
    assert operations == set(range(8)) and len(corner_pairs) == 10
    assert {f["axis"] for f in features} == {0, 1}
    assert {f["length"] for f in features} == {2, 3, 4}
    assert any(f["overlap"] == 0 for f in features) and any(f["overlap"] > 0 for f in features)
    assert any(f["private_cells"] == 0 for f in features) and any(f["private_cells"] > 0 for f in features)
    assert len({f["new_chambers"] for f in features}) >= 2
    assert any(f["blank_edges"] > 0 for f in features)
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        puzzle = generate_puzzle(2718, nt, ne)
        assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(2718, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    for kind in ("cell", "extent", "guide"):
        mutant = {s: [{k: v.copy() for k, v in p.items()} for p in example[s]] for s in example}
        pair = mutant["test"][0]
        if kind == "cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "extent":
            pair["output"] = pair["output"][1:, :]
        else:
            info = _decode(pair["input"])
            marker_color = next(iter(info["controls"].values()))
            position = next(tuple(map(int, p)) for p in np.argwhere(pair["input"] == marker_color)
                            if 1 < p[0] < pair["input"].shape[0] - 2 and 1 < p[1] < pair["input"].shape[1] - 2)
            pair["input"][position] = info["palette"][0]
        try:
            validate_generated_puzzle(mutant)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("Corrupted sample accepted")
    return {"seeds": 42, "pairs": len(features), "distinct_puzzles": len(signatures),
            "input_shapes": len(input_shapes), "output_shapes": len(output_shapes),
            "wall_patterns_up_to_symmetry": len(geometries), "role_triples": len(palettes),
            "local_symmetries": sorted(operations), "guide_corner_combinations": len(corner_pairs),
            "registration_lengths": sorted({f["length"] for f in features}),
            "new_chamber_counts": sorted({f["new_chambers"] for f in features}),
            "private_chamber_counts": sorted({f["private_chambers"] for f in features}),
            "collision_free_pairs": sum(f["overlap"] == 0 for f in features),
            "overlapping_pairs": sum(f["overlap"] > 0 for f in features),
            "blank_edge_pairs": sum(f["blank_edges"] > 0 for f in features)}
