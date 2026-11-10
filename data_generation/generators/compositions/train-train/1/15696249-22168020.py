"""Taper blueprints: copy a tile over the completed taper hanging from its guide line.

Composition of ARC 15696249 (parent A) and ARC 22168020 (parent B).

Every input is a small fully coloured tile (3..5 by 3..5, colours 1..9) with exactly
one full monochromatic row or column: the guide (parent A).  Every other cell of the
guide colour belongs to tapering arms that hang from the guide on one or both sides,
two endpoints per line, converging to a one- or two-cell tip (parent B).

Rule: on every line parallel to the guide, mark the cells from the leftmost to the
rightmost guide-coloured cell (parent B's span completion, oriented by parent A's
guide).  The marked cells form a solid wedge whose base is the guide.  The output is
an (h*h) x (w*w) canvas of h x w macro-cells (parent A's macro layout); every marked
macro-cell receives a verbatim copy of the whole input, every other macro-cell is 0.
With no arms this is exactly parent A; without the copies it is parent B's fill.
"""
from __future__ import annotations

import json

import numpy as np

BLANK = 0
COLORS = tuple(range(1, 10))
SIZES = (3, 4, 5)
_SHIFTS = ("held_out_index", "double_sided", "larger", "outnumbered", "familiar")
_SHIFT_P = (0.25, 0.2, 0.2, 0.15, 0.2)

# 3x3 inputs of parent A (task and selected variation); never reused as inputs.
_PARENT_A_INPUTS = {
    ((3, 3), (1, 1, 1, 6, 2, 2, 2, 2, 6)), ((3, 3), (2, 4, 3, 2, 3, 4, 2, 3, 4)),
    ((3, 3), (3, 1, 6, 3, 6, 1, 3, 1, 6)), ((3, 3), (4, 4, 6, 3, 3, 3, 6, 6, 4)),
    ((3, 3), (6, 6, 3, 4, 4, 3, 4, 4, 3)), ((3, 3), (4, 7, 7, 1, 1, 1, 7, 4, 7)),
    ((3, 3), (4, 3, 1, 4, 1, 1, 4, 1, 3)), ((3, 3), (1, 1, 1, 3, 4, 7, 0, 5, 2)),
    ((3, 3), (8, 0, 3, 8, 3, 8, 8, 8, 3)), ((3, 3), (9, 2, 7, 3, 5, 7, 2, 5, 7)),
}


# --------------------------------------------------------------------------- rule

def _as_grid(value) -> np.ndarray:
    grid = np.asarray(value)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty 2D array")
    return grid.astype(int)


def _full_lines(grid: np.ndarray) -> list[tuple[str, int]]:
    lines = [("row", r) for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])]
    lines += [("col", c) for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c])]
    return lines


def _find_guide(grid: np.ndarray) -> tuple[str, int, int]:
    lines = _full_lines(grid)
    if len(lines) != 1:
        raise ValueError("expected exactly one full monochromatic row or column")
    kind, index = lines[0]
    color = int(grid[index, 0] if kind == "row" else grid[0, index])
    return kind, index, color


def _line_mask(grid: np.ndarray, axis: str, color: int) -> np.ndarray:
    """Per row ('row') or per column ('col'): span between extreme cells of color."""
    canon = grid if axis == "row" else grid.T
    mask = np.zeros(canon.shape, dtype=bool)
    for r in range(canon.shape[0]):
        hits = np.flatnonzero(canon[r] == color)
        if hits.size:
            mask[r, hits[0]:hits[-1] + 1] = True
    return mask if axis == "row" else mask.T


def _enclosed_mask(grid: np.ndarray, color: int) -> np.ndarray:
    """Alternative reading: colour cells plus regions they 4-enclose (not the rule)."""
    walls = grid == color
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    stack = [(r, c) for r in range(h) for c in range(w)
             if (r in (0, h - 1) or c in (0, w - 1)) and not walls[r, c]]
    for r, c in stack:
        outside[r, c] = True
    while stack:
        r, c = stack.pop()
        for a, b in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= a < h and 0 <= b < w and not walls[a, b] and not outside[a, b]:
                outside[a, b] = True
                stack.append((a, b))
    return ~outside


def _replicate(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h * h, w * w), BLANK, dtype=int)
    for r, c in zip(*np.nonzero(mask)):
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = grid
    return out


def solve(input_grid) -> np.ndarray:
    """Input-only solver: guide -> completed taper parallel to it -> macro copies."""
    grid = _as_grid(input_grid)
    kind, _, color = _find_guide(grid)
    return _replicate(grid, _line_mask(grid, kind, color))


# ----------------------------------------------------------------------- analysis

def _key(grid: np.ndarray) -> tuple:
    return (tuple(int(n) for n in grid.shape), tuple(int(v) for v in grid.flat))


def _analyze(grid) -> dict:
    """Recover guide, taper sides and shortcut-relevant properties from an input."""
    grid = _as_grid(grid)
    kind, index, color = _find_guide(grid)
    canon = grid if kind == "row" else grid.T
    depth_axis, length = canon.shape
    problems = []
    sides = {}
    for side, step in (("before", -1), ("after", 1)):
        spans, ended, line = [], False, index + step
        while 0 <= line < depth_axis:
            hits = np.flatnonzero(canon[line] == color)
            if hits.size == 0:
                ended = True
            else:
                if ended:
                    problems.append(f"{side} arm lines are not consecutive")
                if hits.size > 2:
                    problems.append(f"{side} line has more than two endpoints")
                spans.append((int(hits[0]), int(hits[-1])))
            line += step
        if spans:
            sides[side] = spans
    for side, spans in sides.items():
        previous = (0, length - 1)
        for left, right in spans:
            if left < previous[0] or right > previous[1]:
                problems.append(f"{side} spans are not nested")
            previous = (left, right)
        if spans[-1][1] - spans[-1][0] + 1 > 2:
            problems.append(f"{side} taper does not converge to a tip")
    parallel = _line_mask(grid, kind, color)
    perpendicular = _line_mask(grid, "col" if kind == "row" else "row", color)
    counts = np.bincount(grid.ravel(), minlength=10)
    others = [int(counts[c]) for c in range(10) if c != color]
    raw = grid == color
    return {
        "kind": kind, "index": index, "color": color, "depth_axis": depth_axis,
        "length": length, "shape": tuple(int(n) for n in grid.shape),
        "sides": sides, "anchor": not sides, "taper_ok": not problems,
        "problems": problems,
        "gap": bool(np.any(parallel & ~raw)),
        "perp_diff": not np.array_equal(parallel, perpendicular),
        "enclosed_diff": not np.array_equal(parallel, _enclosed_mask(grid, color)),
        "outnumbered": int(counts[color]) <= max(others),
        "double": len(sides) == 2,
        "colors": tuple(sorted(int(c) for c in np.unique(grid))),
        "mask_cells": int(parallel.sum()),
    }


def _demonstration_problems(train: list[dict], test: list[dict]) -> list[str]:
    """Puzzle-level evidence requirements shared by the generator and validator."""
    problems = []
    tapered = [i for i in train if not i["anchor"]]
    if sum(i["anchor"] for i in train) > 1:
        problems.append("more than one arm-less training anchor")
    if any(i["anchor"] for i in test):
        problems.append("a test pair has no arms")
    if any(not i["gap"] for i in test):
        problems.append("a test pair is solved by copying at raw guide-colour cells")
    if sum(i["gap"] for i in tapered) < min(2, len(train)):
        problems.append("training shows too few completed gaps")
    if not any(i["outnumbered"] for i in train):
        problems.append("guide colour is always the most frequent training colour")
    if len({(i["kind"], i["index"]) for i in train}) < 2:
        problems.append("training shows only one guide position")
    train_kinds = {i["kind"] for i in train}
    train_sides = {s for i in tapered for s in i["sides"]}
    train_shapes = {i["shape"] for i in train}
    for info in test:
        if info["kind"] not in train_kinds:
            problems.append("test guide orientation never trained")
        if not set(info["sides"]) <= train_sides:
            problems.append("test taper side never trained")
        if info["perp_diff"] and not any(i["perp_diff"] for i in tapered):
            problems.append("fill axis matters in test but not in training")
        if info["enclosed_diff"] and not any(i["enclosed_diff"] for i in tapered):
            problems.append("enclosed-region fill differs in test but not in training")
        position = (info["kind"], info["index"])
        if position not in {(i["kind"], i["index"]) for i in train}:
            indices = {i["index"] for i in train if i["kind"] == info["kind"]}
            if len(indices) < 2:
                problems.append("held-out guide index lacks two trained indices")
        if info["shape"] not in train_shapes and len(train_shapes) < 2:
            problems.append("unseen test shape without two training shapes")
        if info["shape"][0] != info["shape"][1] and not any(
                s[0] != s[1] for s in train_shapes):
            problems.append("rectangular test without rectangular training")
    return problems


# --------------------------------------------------------------------- generation

def _check_counts(num_train, num_test) -> tuple[int, int]:
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


def _sample_decor(rng, guide: int) -> list[int]:
    count = int(rng.choice((1, 2, 3, 4), p=(0.2, 0.35, 0.3, 0.15)))
    pool = [c for c in COLORS if c != guide]
    return [int(c) for c in rng.choice(pool, size=count, replace=False)]


def _settings(rng, num_train: int) -> dict:
    s = {"orient_mode": _pick(rng, ("row", "col", "mixed", "mixed")),
         "size_mode": "fixed" if rng.random() < 0.3 else "varied",
         "shift": _SHIFTS[int(rng.choice(len(_SHIFTS), p=_SHIFT_P))]}
    if s["shift"] == "larger":
        s["size_mode"] = "varied"
    low = 4 if s["shift"] in ("double_sided", "held_out_index") else 3
    s["fixed_shape"] = (int(rng.integers(low, 6)), int(rng.choice(SIZES, p=(0.2, 0.35, 0.45))))
    s["palette_mode"] = _pick(rng, ("shared", "guide_shared", "varied", "varied", "varied"))
    s["guide_color"] = int(_pick(rng, COLORS))
    s["decor"] = _sample_decor(rng, s["guide_color"])
    s["field_prob"] = float(_pick(rng, (0.15, 0.5, 0.5, 0.85)))
    s["arm_mode"] = _pick(rng, ("continuous", "gapped", "mixed", "mixed"))
    s["anchor"] = num_train >= 3 and rng.random() < 0.25
    s["test_kind"] = _pick(rng, ("row", "col")) if s["orient_mode"] == "mixed" else s["orient_mode"]
    return s


def _min_continuous_depth(length: int) -> int:
    return max(1, (length - 1) // 2)


def _sample_taper(rng, length: int, depth_max: int, arm: str):
    """Nested spans moving inward from the guide until the width is at most two."""
    if depth_max < 1:
        return None
    if arm == "continuous" and depth_max < _min_continuous_depth(length):
        arm = "gapped"
    max_step = 1 if arm == "continuous" else 2
    for _ in range(40):
        left, right, spans, repeated = 0, length - 1, [], False
        while len(spans) < depth_max:
            first = not spans
            dl, dr = int(rng.integers(max_step + 1)), int(rng.integers(max_step + 1))
            if dl + dr == 0:
                if not repeated and rng.random() < (0.35 if first else 0.2):
                    repeated = True
                elif rng.random() < 0.5:
                    dl = 1
                else:
                    dr = 1
            width = right - left + 1
            while width - dl - dr < 1:
                if dl >= dr:
                    dl -= 1
                else:
                    dr -= 1
            left, right = left + dl, right - dr
            spans.append((left, right))
            if right - left + 1 <= 2:
                break
        if right - left + 1 > 2:
            continue
        spare = depth_max - len(spans)
        if spare > 0 and rng.random() < 0.3:
            spans.extend([spans[-1]] * int(rng.integers(1, min(2, spare) + 1)))
        return spans
    return None


def _render(rng, spec) -> np.ndarray | None:
    depth_axis, length, index = spec["P"], spec["L"], spec["index"]
    guide = spec["guide_color"]
    canon = np.zeros((depth_axis, length), dtype=int)
    figure = np.zeros((depth_axis, length), dtype=bool)
    canon[index, :] = guide
    figure[index, :] = True
    for side in spec["sides"]:
        step = -1 if side == "before" else 1
        depth_max = index if side == "before" else depth_axis - 1 - index
        spans = _sample_taper(rng, length, depth_max, spec["arm"])
        if spans is None:
            return None
        for t, (left, right) in enumerate(spans, 1):
            line = index + step * t
            for col in (left, right):
                canon[line, col] = guide
                figure[line, col] = True
    free = np.argwhere(~figure)
    decor = list(spec["decor"])
    if spec["style"] == "field":
        canon[~figure] = decor[0]
        extras = decor[1:]
        if extras:
            for r, c in free:
                if rng.random() < spec["density"]:
                    canon[r, c] = _pick(rng, extras)
    else:
        weights = rng.dirichlet(np.full(len(decor), 0.9))
        for r, c in free:
            canon[r, c] = decor[int(rng.choice(len(decor), p=weights))]
    for _ in range(30):
        bad = [line for line in _full_lines(canon) if line != ("row", index)]
        if not bad:
            break
        kind, i = bad[int(rng.integers(len(bad)))]
        cells = [(r, c) for r, c in free if (r == i if kind == "row" else c == i)]
        if not cells:
            return None
        r, c = cells[int(rng.integers(len(cells)))]
        current = int(canon[r, c])
        choices = [d for d in decor if d != current]
        if not choices:
            choices = [d for d in COLORS if d not in (guide, current)]
        canon[r, c] = _pick(rng, choices)
    else:
        return None
    return canon if spec["orient"] == "row" else canon.T.copy()


def _layouts(depth_axis: int, req: dict, orient: str) -> list[tuple[int, tuple]]:
    options = []
    for index in range(depth_axis):
        if (orient, index) in req.get("exclude", ()) or index in req.get("avoid", ()):
            continue
        if req.get("anchor"):
            options.append((index, ()))
            continue
        for sides in (("before",), ("after",), ("before", "after")):
            depths = [index if s == "before" else depth_axis - 1 - index for s in sides]
            if min(depths) < 1:
                continue
            if req.get("single") and len(sides) == 2:
                continue
            if req.get("double") and len(sides) != 2:
                continue
            if not set(req.get("must_sides", ())) <= set(sides):
                continue
            if (req.get("gap") or req.get("perp") or req.get("leak")) and max(depths) < 2:
                continue
            options.append((index, sides))
    return options


def _make_spec(rng, s: dict, req: dict):
    orient = req.get("orient")
    if orient is None:
        orient = _pick(rng, ("row", "col")) if s["orient_mode"] == "mixed" else s["orient_mode"]
    for _ in range(40):
        if s["size_mode"] == "fixed":
            depth_axis, length = s["fixed_shape"]
        else:
            depth_axis = int(rng.choice(SIZES, p=(0.25, 0.35, 0.4)))
            length = int(rng.choice(SIZES, p=(0.2, 0.35, 0.45)))
        area = depth_axis * length
        if area > req.get("max_area", 99) or area < req.get("min_area", 0):
            continue
        if req.get("rect") and depth_axis == length:
            continue
        options = _layouts(depth_axis, req, orient)
        if not options:
            continue
        weights = np.array([0.55 if len(sides) == 2 else 1.0 for _, sides in options])
        index, sides = options[int(rng.choice(len(options), p=weights / weights.sum()))]
        break
    else:
        return None
    if req.get("perp") or req.get("leak"):
        arm = "gapped"
    elif s["arm_mode"] == "mixed":
        arm = _pick(rng, ("continuous", "gapped"))
    else:
        arm = s["arm_mode"]
    if req.get("mosaic"):
        style = "mosaic"
    elif req.get("outnumber"):
        style = "field" if rng.random() < 0.75 else "mosaic"
    else:
        style = "field" if rng.random() < s["field_prob"] else "mosaic"
    if s["palette_mode"] == "shared":
        guide, decor = s["guide_color"], list(s["decor"])
    elif s["palette_mode"] == "guide_shared":
        guide = s["guide_color"]
        decor = _sample_decor(rng, guide)
    else:
        guide = int(_pick(rng, COLORS))
        decor = _sample_decor(rng, guide)
    if style == "mosaic" and req.get("outnumber") and len(decor) > 2:
        decor = decor[:2]
    return {"orient": orient, "P": depth_axis, "L": length, "index": index, "sides": sides,
            "arm": arm, "style": style, "guide_color": guide, "decor": decor,
            "density": float(rng.uniform(0.05, 0.35)), "req": req}


def _matches(info: dict, spec: dict) -> bool:
    req = spec["req"]
    if not info["taper_ok"] or info["kind"] != spec["orient"] or info["index"] != spec["index"]:
        return False
    if set(info["sides"]) != set(spec["sides"]):
        return False
    if (req.get("gap") or req.get("test")) and not info["gap"]:
        return False
    if req.get("perp") and not info["perp_diff"]:
        return False
    if req.get("leak") and not info["enclosed_diff"]:
        return False
    return not (req.get("outnumber") and not info["outnumbered"])


def _make_pair(rng, s: dict, req: dict, used: set):
    for _ in range(6):
        spec = _make_spec(rng, s, req)
        if spec is None:
            continue
        for _ in range(25):
            tile = _render(rng, spec)
            if tile is None:
                continue
            key = _key(tile)
            if key in used or key in _PARENT_A_INPUTS:
                continue
            try:
                info = _analyze(tile)
            except ValueError:
                continue
            if _matches(info, spec):
                used.add(key)
                return tile, info
    return None


def _test_requirements(rng, s: dict, j: int) -> dict:
    req = {"test": True, "gap": True}
    if rng.random() < 0.45:
        req["outnumber"] = True
    if s["shift"] == "larger":
        req["min_area"] = 16
    if s["shift"] == "held_out_index":
        req["orient"] = s["test_kind"]
    if j == 0 and s["shift"] == "double_sided":
        req["double"] = True
    if j == 0 and s["shift"] == "outnumbered":
        req.update(outnumber=True, mosaic=True)
    return req


def _training_requirements(rng, s: dict, tests: list[dict], num_train: int) -> list[dict]:
    reqs = [dict() for _ in range(num_train)]
    order = [int(i) for i in rng.permutation(num_train)]
    if s["anchor"]:
        reqs[order.pop()]["anchor"] = True
    active = order
    cursor = [0]

    def take(predicate=lambda r: True) -> dict:
        for k in range(len(active)):
            slot = reqs[active[(cursor[0] + k) % len(active)]]
            if predicate(slot):
                cursor[0] = (cursor[0] + k + 1) % len(active)
                return slot
        return reqs[active[cursor[0] % len(active)]]

    for kind in sorted({t["kind"] for t in tests}):
        take(lambda r: "orient" not in r)["orient"] = kind
    sides = sorted({side for t in tests for side in t["sides"]})
    if s["shift"] == "double_sided":
        for r in reqs:
            r["single"] = True
        for side in ("before", "after"):
            slot = take(lambda r: not r.get("must_sides"))
            slot.setdefault("must_sides", set()).add(side)
    else:
        for side in sides:
            take().setdefault("must_sides", set()).add(side)
    if s["shift"] == "held_out_index":
        first = tests[0]
        exclude = {(first["kind"], first["index"])}
        for r in reqs:
            r["exclude"] = exclude
        same = [r for r in reqs if r.get("orient") == first["kind"] and not r.get("anchor")]
        while len(same) < 2:
            slot = take(lambda r: "orient" not in r)
            if slot.get("orient", first["kind"]) != first["kind"] or any(slot is r for r in same):
                break
            slot["orient"] = first["kind"]
            same.append(slot)
        for r in same:
            r["diverse"] = True
    for k, idx in enumerate(active):
        if k < 2 or rng.random() < 0.75:
            reqs[idx]["gap"] = True
    if any(t["perp_diff"] for t in tests):
        take()["perp"] = True
    if any(t["enclosed_diff"] for t in tests):
        take()["leak"] = True
    take()["outnumber"] = True
    if s["shift"] == "larger":
        limit = min(t["shape"][0] * t["shape"][1] for t in tests) - 1
        for r in reqs:
            r["max_area"] = limit
    if s["size_mode"] == "varied" and any(t["shape"][0] != t["shape"][1] for t in tests):
        take()["rect"] = True
    return reqs


def _attempt(rng, num_train: int, num_test: int):
    s = _settings(rng, num_train)
    used: set = set()
    test_pairs = []
    for j in range(num_test):
        made = _make_pair(rng, s, _test_requirements(rng, s, j), used)
        if made is None:
            return None
        test_pairs.append(made)
    test_infos = [info for _, info in test_pairs]
    train_pairs = []
    indices: dict[str, set] = {"row": set(), "col": set()}
    for req in _training_requirements(rng, s, test_infos, num_train):
        if req.get("diverse"):
            req = dict(req, avoid=set(indices[req["orient"]]))
        made = _make_pair(rng, s, req, used)
        if made is None:
            return None
        indices[made[1]["kind"]].add(made[1]["index"])
        train_pairs.append(made)
    order = rng.permutation(num_train)
    train_pairs = [train_pairs[int(i)] for i in order]
    if _demonstration_problems([i for _, i in train_pairs], test_infos):
        return None
    return {split: [{"input": tile.tolist(), "output": solve(tile).tolist()} for tile, _ in pairs]
            for split, pairs in (("train", train_pairs), ("test", test_pairs))}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    for _ in range(500):
        puzzle = _attempt(rng, num_train, num_test)
        if puzzle is not None:
            return puzzle
    raise RuntimeError("could not construct a puzzle satisfying the demonstration rules")


# --------------------------------------------------------------------- validation

def _strict_grid(value) -> np.ndarray:
    grid = np.asarray(value)
    if grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer):
        raise AssertionError("grids must be 2D integer arrays")
    if not all(1 <= n <= 30 for n in grid.shape) or grid.min() < 0 or grid.max() > 9:
        raise AssertionError("grid dimensions must be 1..30 and colours 0..9")
    return grid.astype(int)


def _expected_blocks(info: dict) -> set[tuple[int, int]]:
    """Blueprint rebuilt from the analysed spans (independent of _line_mask)."""
    cells = set()
    for u in range(info["length"]):
        cells.add((info["index"], u))
    for side, spans in info["sides"].items():
        step = -1 if side == "before" else 1
        for t, (left, right) in enumerate(spans, 1):
            for u in range(left, right + 1):
                cells.add((info["index"] + step * t, u))
    if info["kind"] == "col":
        cells = {(u, t) for t, u in cells}
    return cells


def _check_output(inp: np.ndarray, out: np.ndarray, info: dict) -> None:
    h, w = inp.shape
    if out.shape != (h * h, w * w):
        raise AssertionError("output must be an h*h by w*w macro layout")
    copied = set()
    for r in range(h):
        for c in range(w):
            block = out[r * h:(r + 1) * h, c * w:(c + 1) * w]
            if np.array_equal(block, inp):
                copied.add((r, c))
            elif not np.all(block == BLANK):
                raise AssertionError("macro-cells must be verbatim copies or blank")
    expected = _expected_blocks(info)
    if copied != expected:
        raise AssertionError("copied macro-cells differ from the completed taper")
    raw = {(int(r), int(c)) for r, c in zip(*np.nonzero(inp == info["color"]))}
    if not raw <= copied:
        raise AssertionError("a guide-coloured cell lacks its copy")


def validate_generated_puzzle(puzzle: dict) -> None:
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise AssertionError("puzzle must have exactly train and test")
    try:
        _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise AssertionError("pairs must have exactly input and output")
            inp, out = _strict_grid(pair["input"]), _strict_grid(pair["output"])
            if not all(3 <= n <= 5 for n in inp.shape):
                raise AssertionError("input tiles must be 3..5 by 3..5")
            if np.any(inp == BLANK):
                raise AssertionError("0 is reserved for blank macro-cells")
            key = _key(inp)
            if key in seen or key in _PARENT_A_INPUTS:
                raise AssertionError("duplicate or parent-copied input")
            seen.add(key)
            try:
                info = _analyze(inp)
            except ValueError as error:
                raise AssertionError(str(error)) from error
            if not info["taper_ok"]:
                raise AssertionError("; ".join(info["problems"]))
            _check_output(inp, out, info)
            infos[split].append(info)
    problems = _demonstration_problems(infos["train"], infos["test"])
    if problems:
        raise AssertionError("; ".join(problems))


# ----------------------------------------------------------------------- self test

def _shortcuts(inp: np.ndarray) -> dict:
    """Outputs of plausible rules that drop one parent's contribution."""
    kind, index, color = _find_guide(inp)
    strip = np.zeros(inp.shape, dtype=bool)
    if kind == "row":
        strip[index, :] = True
    else:
        strip[:, index] = True
    filled = inp.copy()
    filled[_line_mask(inp, kind, color)] = color
    rows, cols = np.nonzero(inp == color)
    box = np.zeros(inp.shape, dtype=bool)
    box[rows.min():rows.max() + 1, cols.min():cols.max() + 1] = True
    other = "col" if kind == "row" else "row"
    counts = np.bincount(inp.ravel(), minlength=10)
    dominant = int(np.argmax(counts))
    return {
        "parent_A_only": _replicate(inp, strip),
        "parent_B_only": filled,
        "raw_guide_cells": _replicate(inp, inp == color),
        "bounding_box": _replicate(inp, box),
        "perpendicular_fill": _replicate(inp, _line_mask(inp, other, color)),
        "enclosed_fill": _replicate(inp, _enclosed_mask(inp, color)),
        "dominant_colour": _replicate(inp, _line_mask(inp, kind, dominant)),
        "completed_copies": _replicate(filled, _line_mask(inp, kind, color)),
    }


def self_test() -> None:
    for bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(0, *bad)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {bad}")
    stats = {k: set() for k in ("shapes", "positions", "guide_colors", "shifts", "palettes")}
    tally = dict(double=0, perp=0, anchor=0, outnumbered=0, mosaic_like=0, tests=0,
                 test_rows=0, test_cols=0, held_out=0, unseen_shape=0, mixed_kinds=0)
    beaten = {}
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        text = json.dumps(puzzle, sort_keys=True)
        if text != json.dumps(generate_puzzle(seed), sort_keys=True):
            raise AssertionError(f"seed {seed} is not deterministic")
        signatures.add(text)
        train_infos = [_analyze(p["input"]) for p in puzzle["train"]]
        if len({i["kind"] for i in train_infos}) == 2:
            tally["mixed_kinds"] += 1
        trained = {(i["kind"], i["index"]) for i in train_infos}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = np.array(pair["input"]), np.array(pair["output"])
                if not np.array_equal(solve(inp), out):
                    raise AssertionError("solve disagrees with a generated output")
                info = _analyze(inp)
                stats["shapes"].add(info["shape"])
                stats["guide_colors"].add(info["color"])
                stats["palettes"].add(info["colors"])
                stats["positions"].add((info["kind"], info["index"], tuple(sorted(info["sides"]))))
                tally["double"] += info["double"]
                tally["perp"] += info["perp_diff"]
                tally["anchor"] += info["anchor"]
                tally["outnumbered"] += info["outnumbered"]
                tally["mosaic_like"] += len(info["colors"]) >= 4
                if split == "test":
                    tally["tests"] += 1
                    tally["test_rows"] += info["kind"] == "row"
                    tally["test_cols"] += info["kind"] == "col"
                    tally["held_out"] += (info["kind"], info["index"]) not in trained
                    tally["unseen_shape"] += info["shape"] not in {i["shape"] for i in train_infos}
                    for name, guess in _shortcuts(inp).items():
                        if guess.shape != out.shape or not np.array_equal(guess, out):
                            beaten[name] = beaten.get(name, 0) + 1
    if len(signatures) != 42:
        raise AssertionError("seeds collide")
    for name in ("parent_A_only", "parent_B_only", "raw_guide_cells", "bounding_box",
                 "completed_copies"):
        if beaten.get(name, 0) != tally["tests"]:
            raise AssertionError(f"shortcut {name} solves some generated test")
    checks = {
        "shapes": len(stats["shapes"]) >= 8, "positions": len(stats["positions"]) >= 14,
        "guide colours": len(stats["guide_colors"]) == 9, "palettes": len(stats["palettes"]) >= 60,
        "double-sided": tally["double"] >= 15, "fill axis": tally["perp"] >= 10,
        "anchors": tally["anchor"] >= 2, "outnumbered": tally["outnumbered"] >= 60,
        "mosaic": tally["mosaic_like"] >= 30, "row tests": tally["test_rows"] >= 8,
        "column tests": tally["test_cols"] >= 8, "held-out tests": tally["held_out"] >= 5,
        "unseen test shapes": tally["unseen_shape"] >= 5, "mixed puzzles": tally["mixed_kinds"] >= 8,
        "perpendicular shortcut": beaten.get("perpendicular_fill", 0) >= 3,
        "enclosed shortcut": beaten.get("enclosed_fill", 0) >= 3,
        "dominant shortcut": beaten.get("dominant_colour", 0) >= 18,
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"variation thresholds failed: {failed}; {tally}")
    sample = generate_puzzle(0)
    inp = np.array(sample["test"][0]["input"])
    for name, guess in _shortcuts(inp).items():
        if guess.shape != (inp.shape[0] ** 2, inp.shape[1] ** 2) or np.array_equal(
                guess, np.array(sample["test"][0]["output"])):
            continue
        broken = json.loads(json.dumps(sample))
        broken["test"][0]["output"] = guess.tolist()
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            continue
        raise AssertionError(f"validator accepted the {name} shortcut output")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1000 + 17 * train + test, train, test)
            if (len(puzzle["train"]), len(puzzle["test"])) != (train, test):
                raise AssertionError("wrong pair counts")
            validate_generated_puzzle(puzzle)


if __name__ == "__main__":
    self_test()
    print("self_test passed")
