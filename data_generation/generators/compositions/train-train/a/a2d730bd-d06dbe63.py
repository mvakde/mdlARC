"""Docking staircases: composition of a2d730bd (capped rectangle connectors) and
d06dbe63 (seeded two-arm staircase with next-step clipping).

Every singleton seed emits two staircase arms: the upper arm repeats
up, up, right, right and the lower arm repeats down, down, left, left.  An arm
stops when its next unit step would leave the grid OR enter a solid rectangle.
An arm that runs off the grid is painted in the seed's colour.  An arm stopped
by a rectangle "docks": the whole arm takes the rectangle's colour and gets a
three-cell cap perpendicular to the blocked step, lying in the background layer
against that rectangle face.  Seeds, rectangles and other cells are unchanged.
"""
from __future__ import annotations

import json

import numpy as np

UP_CYCLE = ((-1, 0), (-1, 0), (0, 1), (0, 1))
DOWN_CYCLE = ((1, 0), (1, 0), (0, -1), (0, -1))
CYCLES = (UP_CYCLE, DOWN_CYCLE)


# --------------------------------------------------------------------------
# Input analysis and the input-only solver
# --------------------------------------------------------------------------

def _components(grid, bg):
    """4-connected same-colour components of non-background cells."""
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg:
                continue
            color = int(grid[r, c])
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and grid[ny, nx] == color:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((color, sorted(cells)))
    return comps


def _analyze(grid):
    grid = np.asarray(grid, dtype=int)
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    seeds, rects = [], []
    rect_id = -np.ones(grid.shape, dtype=int)
    for color, cells in _components(grid, bg):
        if len(cells) == 1:
            seeds.append((cells[0], color))
        else:
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            box = (min(rows), min(cols), max(rows), max(cols))
            rect_id[box[0]:box[2] + 1, box[1]:box[3] + 1] = len(rects)
            rects.append({"color": color, "box": box, "solid":
                          len(cells) == (box[2] - box[0] + 1) * (box[3] - box[1] + 1)})
    return bg, sorted(seeds), rects, rect_id


def _trace(shape, rect_id, seed, cycle):
    """Walk one arm.  Returns (cells, stop) where stop is ("edge", step_index)
    or ("rect", rect_index, step_vector, blocked_cell)."""
    h, w = shape
    r, c = seed
    cells, k = [], 0
    while True:
        dr, dc = cycle[k % 4]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return cells, ("edge", k % 4)
        if rect_id[nr, nc] >= 0:
            return cells, ("rect", int(rect_id[nr, nc]), (dr, dc), (nr, nc))
        r, c = nr, nc
        cells.append((r, c))
        k += 1


def _arm_plan(grid):
    """All arms of all seeds with their colour and painted cells."""
    grid = np.asarray(grid, dtype=int)
    bg, seeds, rects, rect_id = _analyze(grid)
    arms = []
    for pos, color in seeds:
        for which, cycle in enumerate(CYCLES):
            cells, stop = _trace(grid.shape, rect_id, pos, cycle)
            arm = {"seed": pos, "seed_color": color, "which": which, "cells": cells,
                   "stop": stop, "cap": []}
            if stop[0] == "rect":
                rect = rects[stop[1]]
                arm["color"] = rect["color"]
                (dr, dc) = stop[2]
                last = cells[-1] if cells else pos
                pr, pc = (dc, dr)  # perpendicular to the blocked step
                arm["cap"] = [(last[0] + pr, last[1] + pc), last, (last[0] - pr, last[1] - pc)]
            else:
                arm["color"] = color
            arms.append(arm)
    return bg, seeds, rects, rect_id, arms


def solve(input_grid):
    grid = np.asarray(input_grid, dtype=int)
    out = grid.copy()
    h, w = grid.shape
    _, _, _, _, arms = _arm_plan(grid)
    for arm in arms:
        for (r, c) in arm["cells"] + arm["cap"]:
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = arm["color"]
    return out


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def _check_counts(num_train, num_test):
    for name, value in (("num_train", num_train), ("num_test", num_test)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    num_train, num_test = int(num_train), int(num_test)
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("need 2..10 train, 1..4 test and num_train > num_test")
    return num_train, num_test


def _pair_issues(grid):
    """Family invariants on an input grid; returns (problems, features)."""
    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    bg, seeds, rects, rect_id, arms = _arm_plan(grid)
    problems = []
    if not seeds:
        problems.append("no seed")
    if not rects:
        problems.append("no rectangle")
    if int(np.sum(grid == bg)) * 2 <= grid.size:
        problems.append("background not a clear majority")
    seed_colors = {c for _, c in seeds}
    for i, rect in enumerate(rects):
        y0, x0, y1, x1 = rect["box"]
        if not rect["solid"] or y1 - y0 < 1 or x1 - x0 < 1:
            problems.append("rectangle not solid 2x2+")
        if rect["color"] in seed_colors:
            problems.append("rectangle shares a seed colour")
        ring = grid[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2]
        inner = (ring != bg) & (ring != rect["color"])
        if inner.any():
            problems.append("rectangle touches another object")
        if np.sum(ring == rect["color"]) != (y1 - y0 + 1) * (x1 - x0 + 1):
            problems.append("rectangle touches same colour")
    for (r, c), _ in seeds:
        patch = grid[max(0, r - 1):r + 2, max(0, c - 1):c + 2]
        if int(np.sum(patch != bg)) != 1:
            problems.append("seed not isolated")
    seed_cells = {p for p, _ in seeds}
    owner = {}
    features = {"escape": 0, "dock_v": 0, "dock_h": 0, "near_miss": 0,
                "double_dock": 0, "edge_run": 0, "dock_colors": set()}
    for idx, arm in enumerate(arms):
        cells = arm["cells"]
        if len(cells) < 2:
            problems.append("arm shorter than two cells")
        if any(p in seed_cells for p in cells):
            problems.append("arm crosses a seed")
        stop = arm["stop"]
        if stop[0] == "rect":
            (dr, dc), (br, bc) = stop[2], stop[3]
            pr, pc = dc, dr
            side = [(br + pr, bc + pc), (br - pr, bc - pc)]
            if not all(0 <= y < h and 0 <= x < w and rect_id[y, x] == stop[1] for y, x in side):
                problems.append("dock not on an interior face cell")
            for (y, x) in arm["cap"]:
                if not (0 <= y < h and 0 <= x < w) or grid[y, x] != bg:
                    problems.append("cap leaves grid or covers an object")
            features["dock_v" if dr else "dock_h"] += 1
            features["dock_colors"].add(rects[stop[1]]["color"])
            if rects[stop[1]]["color"] == arm["seed_color"]:
                problems.append("dock colour equals seed colour")
            check = cells[:-1]
        else:
            features["escape"] += 1
            check = cells
            # The ray ran along an edge before stopping (next-step, not touch).
            if any((y in (0, h - 1) or x in (0, w - 1)) for y, x in cells[:-1]):
                features["edge_run"] += 1
        for (y, x) in check:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and rect_id[ny, nx] >= 0:
                    features["near_miss"] += 1
                    break
            else:
                continue
            break
        for p in set(cells) | set(arm["cap"]):
            if p in owner and owner[p] != idx:
                problems.append("arms overlap")
            owner[p] = idx
    for k in range(0, len(arms), 2):
        if arms[k]["stop"][0] == "rect" and arms[k + 1]["stop"][0] == "rect":
            features["double_dock"] += 1
    features["seeds"] = len(seeds)
    features["rects"] = len(rects)
    return problems, features


def _rect_for_dock(rng, h, w, blocked, step, max_side):
    """Rectangle whose face (facing -step) contains `blocked` as an interior cell."""
    br, bc = blocked
    dr, dc = step
    depth = int(rng.integers(2, max_side + 1))
    span = int(rng.integers(3, max_side + 2))
    offset = int(rng.integers(1, span - 1))  # interior position along the face
    if dr != 0:  # vertical step: face is a row
        r0 = br if dr > 0 else br - depth + 1
        return r0, bc - offset, r0 + depth - 1, bc - offset + span - 1
    c0 = bc if dc > 0 else bc - depth + 1
    return br - offset, c0, br - offset + span - 1, c0 + depth - 1


def _try_pair(rng, cfg):
    h = int(rng.integers(cfg["hmin"], cfg["hmax"] + 1))
    w = int(rng.integers(cfg["wmin"], cfg["wmax"] + 1))
    colors = [int(x) for x in rng.permutation(10)]
    bg = colors[0]
    n_seed_colors = 1 if cfg["seeds"] == 1 or rng.random() < 0.5 else 2
    seed_palette = colors[1:1 + n_seed_colors]
    rect_palette = colors[1 + n_seed_colors:1 + n_seed_colors + int(rng.integers(1, 4))]
    grid = np.full((h, w), bg, dtype=int)
    max_side = max(3, min(7, min(h, w) // 4))

    seeds = []
    for _ in range(cfg["seeds"]):
        for _ in range(40):
            r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            if all(max(abs(r - y), abs(c - x)) >= 4 for (y, x) in seeds):
                seeds.append((r, c))
                break
    if len(seeds) != cfg["seeds"]:
        return None
    for (r, c) in seeds:
        grid[r, c] = seed_palette[int(rng.integers(len(seed_palette)))]

    def can_place(box):
        y0, x0, y1, x1 = box
        if y0 < 0 or x0 < 0 or y1 >= h or x1 >= w:
            return False
        if (grid[max(0, y0 - 1):y1 + 2, max(0, x0 - 1):x1 + 2] != bg).any():
            return False
        return True

    def place(box):
        y0, x0, y1, x1 = box
        grid[y0:y1 + 1, x0:x1 + 1] = rect_palette[int(rng.integers(len(rect_palette)))]

    # Decide which arms should dock; arms are (seed index, which cycle).
    arm_ids = [(s, k) for s in range(len(seeds)) for k in range(2)]
    order = [arm_ids[i] for i in rng.permutation(len(arm_ids))]
    docks = set()
    for s in range(len(seeds)):
        if cfg["double_dock_seed"] == s:
            docks |= {(s, 0), (s, 1)}
        elif rng.random() < cfg["dock_rate"]:
            docks.add((s, int(rng.integers(2))))
    for s, k in order:
        if (s, k) not in docks:
            continue
        _, _, _, rect_id = _analyze(grid)
        cells, stop = _trace(grid.shape, rect_id, seeds[s], CYCLES[k])
        if stop[0] == "rect":
            continue
        if len(cells) < 4:
            return None
        placed = False
        for _ in range(25):
            idx = int(rng.integers(2, len(cells)))  # dock after cell idx (1-based count)
            last = cells[idx - 1]
            step = CYCLES[k][idx % 4]
            if cfg.get("want_axis") is not None and rng.random() < 0.7:
                if (step[0] != 0) != (cfg["want_axis"] == "v"):
                    continue
            blocked = (last[0] + step[0], last[1] + step[1])
            box = _rect_for_dock(rng, h, w, blocked, step, max_side)
            y0, x0, y1, x1 = box
            if any(y0 <= y <= y1 and x0 <= x <= x1 for (y, x) in cells[:idx]):
                continue
            if any(y0 - 1 <= y <= y1 + 1 and x0 - 1 <= x <= x1 + 1 for (y, x) in seeds):
                continue
            if can_place(box):
                place(box)
                placed = True
                break
        if not placed:
            return None
    # Distractor rectangles, sometimes hugging an arm so it is passed closely.
    for _ in range(cfg["extra"]):
        for _ in range(30):
            hh = int(rng.integers(2, max_side + 1))
            ww = int(rng.integers(2, max_side + 1))
            y0 = int(rng.integers(0, h - hh + 1))
            x0 = int(rng.integers(0, w - ww + 1))
            box = (y0, x0, y0 + hh - 1, x0 + ww - 1)
            if can_place(box):
                place(box)
                break
    problems, feats = _pair_issues(grid)
    if problems:
        return None
    if feats["dock_v"] + feats["dock_h"] == 0:
        return None
    if not cfg["allow_double"] and feats["double_dock"]:
        return None
    if cfg["double_dock_seed"] is not None and not feats["double_dock"]:
        return None
    for need in cfg["needs"]:
        if not feats[need]:
            return None
    return grid, feats


def _make_pair(rng, cfg):
    for _ in range(4000):
        result = _try_pair(rng, cfg)
        if result is not None:
            return result
    raise RuntimeError("could not build a pair")


TRAIN_NEEDS = ("escape", "dock_v", "dock_h", "near_miss")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    # Puzzle-level regime: scene density and canvas ranges vary between seeds.
    regime = ["sparse", "mixed", "crowded"][int(rng.integers(3))]
    shape_mode = ["square", "tall", "wide", "free"][int(rng.integers(4))]

    def ranges(split):
        lo, hi = (10, 20) if split == "train" else (15, 30)
        if regime == "crowded":
            lo, hi = lo + 3, min(30, hi + 4)
        if shape_mode == "tall":
            return lo + 3, hi, lo, max(lo, hi - 6)
        if shape_mode == "wide":
            return lo, max(lo, hi - 6), lo + 3, hi
        return lo, hi, lo, hi

    # Assign the required training features (escape, both dock axes, a near miss)
    # to concrete training pairs so the demonstrations always expose them.
    # The two docking axes go to different pairs (a single seed may dock only
    # one arm during training); escape and near-miss go anywhere.
    needs = [[] for _ in range(num_train)]
    slots = [int(x) for x in rng.permutation(num_train)]
    needs[slots[0]].append("dock_v")
    needs[slots[1]].append("dock_h")
    needs[int(rng.integers(num_train))].append("escape")
    needs[int(rng.integers(num_train))].append("near_miss")
    pairs, seen = [], set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            hmin, hmax, wmin, wmax = ranges(split)
            if split == "train":
                n_seeds = 1 if regime == "sparse" else int(rng.integers(1, 3))
                if regime == "crowded":
                    n_seeds = int(rng.integers(1, 4))
                want_axis = None
                if "dock_v" in needs[i] and "dock_h" not in needs[i]:
                    want_axis = "v"
                elif "dock_h" in needs[i] and "dock_v" not in needs[i]:
                    want_axis = "h"
                cfg = {"seeds": n_seeds, "double_dock_seed": None, "allow_double": False,
                       "dock_rate": 0.8, "needs": needs[i], "want_axis": want_axis,
                       "extra": int(rng.integers(0, 3)) + (1 if "near_miss" in needs[i] else 0)}
            else:
                n_seeds = int(rng.integers(1, 3)) if regime == "sparse" else int(rng.integers(2, 4))
                cfg = {"seeds": n_seeds, "double_dock_seed": int(rng.integers(n_seeds)),
                       "allow_double": True, "dock_rate": 0.6, "needs": [], "want_axis": None,
                       "extra": int(rng.integers(0, 3))}
            cfg.update(hmin=hmin, hmax=hmax, wmin=wmin, wmax=wmax)
            for _ in range(50):
                grid, _ = _make_pair(rng, cfg)
                key = grid.tobytes() + bytes(grid.shape)
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise RuntimeError("duplicate inputs")
            pairs.append((split, {"input": grid.tolist(), "output": solve(grid).tolist()}))
    return {"train": [p for s, p in pairs if s == "train"],
            "test": [p for s, p in pairs if s == "test"]}


# --------------------------------------------------------------------------
# Validation and self test
# --------------------------------------------------------------------------

def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    totals = {k: 0 for k in TRAIN_NEEDS}
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert set(pair) == {"input", "output"}
            inp = np.asarray(pair["input"], dtype=int)
            out = np.asarray(pair["output"], dtype=int)
            assert inp.ndim == 2 and inp.shape == out.shape
            assert 1 <= inp.shape[0] <= 30 and 1 <= inp.shape[1] <= 30
            assert inp.min() >= 0 and inp.max() <= 9
            key = json.dumps(inp.tolist())
            assert key not in seen, "duplicate input"
            seen.add(key)
            problems, feats = _pair_issues(inp)
            assert not problems, problems
            assert feats["dock_v"] + feats["dock_h"] >= 1, "pair has no docking arm"
            assert np.array_equal(solve(inp), out), "output disagrees with rule"
            bg = int(np.bincount(inp.ravel()).argmax())
            changed = inp != out
            assert changed.any()
            assert np.all(inp[changed] == bg), "rule only paints background"
            assert set(np.unique(out)) == set(np.unique(inp)), "no new colours"
            if split == "train":
                assert not feats["double_dock"], "double docking reserved for test"
                for k in TRAIN_NEEDS:
                    totals[k] += feats[k]
            else:
                assert feats["double_dock"] >= 1, "test must dock both arms of a seed"
    assert all(totals[k] for k in TRAIN_NEEDS), totals


def self_test():
    a = generate_puzzle(11, 5, 2)
    b = generate_puzzle(11, 5, 2)
    assert json.dumps(a) == json.dumps(b)
    for args in ((1, 1), (11, 1), (4, 0), (3, 3), (2, 5)):
        try:
            generate_puzzle(0, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Hand-built check of the rule on a tiny grid.
    g = np.zeros((9, 9), dtype=int)
    g[6, 3] = 4
    g[1:3, 4:9] = 2  # upper arm: (5,3),(4,3) then right (4,4),(4,5)... blocked moving up
    expected = g.copy()
    for p in ((5, 3), (4, 3), (4, 4), (4, 5), (3, 5)):
        expected[p] = 2
    expected[3, 4] = expected[3, 6] = 2
    for p in ((7, 3), (8, 3), (8, 2), (8, 1)):
        expected[p] = 4
    assert np.array_equal(solve(g), expected), solve(g)
    sigs, shapes, seeds_counts, rect_counts = set(), set(), set(), set()
    for s in range(42):
        p = generate_puzzle(1000 + s, 2 + s % 4, 1 + s % 2)
        validate_generated_puzzle(p)
        sigs.add(json.dumps(p))
        for pair in p["train"] + p["test"]:
            inp = np.asarray(pair["input"])
            shapes.add(inp.shape)
            _, f = _pair_issues(inp)
            seeds_counts.add(f["seeds"])
            rect_counts.add(f["rects"])
    assert len(sigs) == 42
    assert len(shapes) >= 20 and len(seeds_counts) >= 3 and len(rect_counts) >= 3


if __name__ == "__main__":
    self_test()
    print("ok")
