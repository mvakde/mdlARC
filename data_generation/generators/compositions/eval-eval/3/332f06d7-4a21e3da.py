"""Border-cut rigid partitions, synchronized corner motion, and a shared jam.

Only the input supplies semantic roles and geometry. No parent code, seed,
construction metadata, or demonstration is used by solve().
"""
from collections import deque
import json

import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a


def _neighbors(p):
    r, c = p
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(cells):
    todo = set(cells)
    result = []
    while todo:
        root = min(todo)
        todo.remove(root)
        found, queue = {root}, [root]
        for p in queue:
            for q in _neighbors(p):
                if q in todo:
                    todo.remove(q)
                    found.add(q)
                    queue.append(q)
        result.append(found)
    return result


def _rim(shape):
    h, w = shape
    return {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)} | {
        (r, c) for r in range(h) for c in (0, w - 1)}


def _partitions(cells, ar, ac, sides):
    groups, fixed = {}, set()
    for r, c in sorted(cells):
        if r == ar or c == ac:
            fixed.add((r, c))
            continue
        v = ("top" if r < ar else "bottom") if ar is not None else next(iter(sides))
        s = ("left" if c < ac else "right") if ac is not None else next(iter(sides))
        groups.setdefault((v, s), set()).add((r, c))
    return groups, fixed


def _cut_features(cells, ar, ac, sides):
    groups, _ = _partitions(cells, ar, ac, sides)
    if len(groups) != (4 if len(sides) == 2 else 2):
        return None
    gap = short = False
    for axis, coordinate in ((0, ac), (1, ar)):
        if coordinate is None:
            continue
        hits = sorted(p[axis] for p in cells if p[1 - axis] == coordinate)
        if not hits:
            return None
        gap |= any(b > a + 1 for a, b in zip(hits, hits[1:]))
        side = next(s for s in sides if s in (("top", "bottom") if axis == 0 else ("left", "right")))
        far = max if side in ("top", "left") else min
        short |= far(hits) != far(p[axis] for p in cells)
    retained = {k: v for k, v in groups.items() if set(k) & sides}
    return {"gap": gap, "short": short,
            "fragmented": any(len(_components(v)) > 1 for v in retained.values()),
            "groups": groups, "retained": retained}


def _parse(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    _require(h >= 7 and w >= 7, "scene too small for the family")
    colors, counts = np.unique(a, return_counts=True)
    _require(len(colors) == 4, "four distinct visible roles required")
    rim = _rim(a.shape)
    rim_values = [int(a[p]) for p in sorted(rim)]
    bg = max(map(int, colors), key=rim_values.count)
    _require(rim_values.count(bg) > len(rim) // 2, "background must dominate the rim")
    positions = {int(v): set(map(tuple, np.argwhere(a == v))) for v in colors}
    candidates = [v for v in positions if v != bg and 1 <= len(positions[v]) <= 2
                  and positions[v] <= rim]
    _require(len(candidates) == 1, "isolated edge markers are ambiguous")
    mark = candidates[0]
    border_others = {int(a[p]) for p in rim} - {bg, mark}
    _require(len(border_others) == 1, "walls must be visibly rooted at the edge")
    wall = next(iter(border_others))
    body = next(iter(set(positions) - {bg, mark, wall}))
    _require(len(positions[wall]) >= 3, "wall material must have at least three cells")
    _require(any(q in positions[wall] & rim for p in positions[wall] & rim
                 for q in _neighbors(p)), "walls need an adjacent rim pair")
    _require(all(part & rim for part in _components(positions[wall])), "floating wall component")
    cells = positions[body]
    _require(not cells & rim and len(cells) >= 5, "payload must be interior")
    _require(len(_components(cells)) == 1, "source payload must be connected")
    markers = {}
    ar = ac = None
    for r, c in sorted(positions[mark]):
        choices = [s for ok, s in ((r == 0, "top"), (r == h - 1, "bottom"),
                                   (c == 0, "left"), (c == w - 1, "right")) if ok]
        _require(len(choices) == 1 and choices[0] not in markers, "invalid marker position")
        side = choices[0]
        markers[side] = (r, c)
        if side in ("top", "bottom"):
            _require(ac is None, "opposite-side markers are not supported")
            ac = int(c)
        else:
            _require(ar is None, "opposite-side markers are not supported")
            ar = int(r)
    sides = set(markers)
    _require(len(sides) == 1 or (ar is not None and ac is not None), "markers must be adjacent")
    # Entire cut axes are clear of walls, not just the completed ray spans.
    _require(ac is None or not np.any(a[:, ac] == wall), "wall on guide axis")
    _require(ar is None or not np.any(a[ar, :] == wall), "wall on guide axis")
    features = _cut_features(cells, ar, ac, sides)
    _require(features is not None, "every cut region must contain payload")
    groups, fixed = _partitions(cells, ar, ac, sides)
    rays = set()
    for side, (r, c) in markers.items():
        if side in ("top", "bottom"):
            hits = [x for x, y in cells if y == c]
            span = range(0, max(hits) + 1) if side == "top" else range(min(hits), h)
            rays.update((x, c) for x in span)
        else:
            hits = [y for x, y in cells if x == r]
            span = range(0, max(hits) + 1) if side == "left" else range(min(hits), w)
            rays.update((r, y) for y in span)
    return {"grid": a, "bg": bg, "body": body, "mark": mark, "wall": wall,
            "walls": positions[wall], "cells": cells, "markers": markers,
            "sides": sides, "ar": ar, "ac": ac, "fixed": fixed, "rays": rays,
            "groups": groups, "retained": features["retained"], **features}


def _goal_delta(cells, corner, shape):
    rs, cs = zip(*cells)
    dr = -min(rs) if corner[0] == "top" else shape[0] - 1 - max(rs)
    dc = -min(cs) if corner[1] == "left" else shape[1] - 1 - max(cs)
    return int(dr), int(dc)


def _pose(cells, corner, shape, tick):
    dr, dc = _goal_delta(cells, corner, shape)
    vr = int(np.sign(dr)) * min(tick, abs(dr))
    vc = int(np.sign(dc)) * min(tick, abs(dc))
    return {(r + vr, c + vc) for r, c in cells}


def _limit(groups, shape):
    return max(max(map(abs, _goal_delta(cells, k, shape))) for k, cells in groups.items())


def _bbox(cells):
    rs, cs = zip(*cells)
    return {(r, c) for r in range(min(rs), max(rs) + 1)
            for c in range(min(cs), max(cs) + 1)}


def _clock(groups, shape, walls):
    terminal = _limit(groups, shape)
    for tick in range(1, terminal + 1):
        if any(_pose(cells, corner, shape, tick) & walls for corner, cells in groups.items()):
            return tick - 1, False
    return terminal, True


def _render(info, ticks, groups=None):
    # ticks is either one shared clock value or a per-partition counterfactual.
    groups = info["retained"] if groups is None else groups
    out = np.array(info["grid"], dtype=np.int8, copy=True)
    out[out == info["body"]] = info["bg"]
    for p in info["fixed"]:
        out[p] = info["body"]
    for p in info["rays"] - info["fixed"]:
        out[p] = info["mark"]
    for corner, cells in groups.items():
        tick = ticks[corner] if isinstance(ticks, dict) else ticks
        for p in _pose(cells, corner, out.shape, tick):
            out[p] = info["body"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Cut by visible border guides; stop all parts at the first wall contact."""
    info = _parse(input_grid)
    tick, _ = _clock(info["retained"], info["grid"].shape, info["walls"])
    return _render(info, tick)


def _diagnostics(info):
    groups, shape, walls = info["retained"], info["grid"].shape, info["walls"]
    tick, success = _clock(groups, shape, walls)
    independent = {k: _clock({k: cells}, shape, walls)[0] for k, cells in groups.items()}
    filled = {k: _bbox(cells) for k, cells in groups.items()}
    box_tick, _ = _clock(filled, shape, walls)
    expected = _render(info, tick)
    # An incomplete B interpretation cuts into connected components instead
    # of collective regions, but retains the SAME wall rule and shared clock.
    components = [(k, part) for k, cells in groups.items() for part in _components(cells)]
    component_terminal = max(max(map(abs, _goal_delta(v, k, shape))) for k, v in components)
    component_tick = component_terminal
    for t in range(1, component_terminal + 1):
        if any(_pose(v, k, shape, t) & walls for k, v in components):
            component_tick = t - 1
            break
    component_output = np.array(info["grid"], dtype=np.int8, copy=True)
    component_output[component_output == info["body"]] = info["bg"]
    for p in info["fixed"]:
        component_output[p] = info["body"]
    for p in info["rays"] - info["fixed"]:
        component_output[p] = info["mark"]
    for k, part in components:
        for p in _pose(part, k, shape, component_tick):
            component_output[p] = info["body"]
    return {"tick": tick, "success": success, "terminal": _limit(groups, shape),
            "coupled": not np.array_equal(expected, _render(info, independent)),
            "bbox_witness": box_tick != tick,
            "without_A": not np.array_equal(expected, _render(info, _limit(groups, shape))),
            "without_collective_B": not np.array_equal(expected, component_output),
            "partial_clamp": any(min(map(abs, _goal_delta(v, k, shape))) < tick
                                 < max(map(abs, _goal_delta(v, k, shape)))
                                 for k, v in groups.items()),
            "unequal_axes": any(abs(dr) != abs(dc) for dr, dc in
                                (_goal_delta(v, k, shape) for k, v in groups.items())),
            "deleted": sum(len(v) for k, v in info["groups"].items() if k not in groups),
            "partition_sizes": tuple(sorted(len(v) for v in groups.values())),
            "partition_components": tuple(sorted(len(_components(v)) for v in groups.values())),
            "distance_pairs": tuple(sorted(tuple(map(abs, _goal_delta(v, k, shape)))
                                           for k, v in groups.items()))}


def _shape(rng, h, w, double, witness):
    sides = {"top", "right"} if double else {"top"}
    for _ in range(45):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        cells = {(r, c)}
        targets = [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                   (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
        rng.shuffle(targets)
        targets += [tuple(map(int, rng.integers((h, w))))
                    for _ in range(int(rng.integers(0, 4)))]
        for tr, tc in targets:
            vertical_first = bool(rng.integers(2))
            while (r, c) != (tr, tc):
                if r != tr and (c == tc or vertical_first):
                    r += int(np.sign(tr - r))
                else:
                    c += int(np.sign(tc - c))
                cells.add((r, c))
        for _ in range(int(rng.integers(0, max(1, h * w // 8)))):
            frontier = sorted({q for p in cells for q in _neighbors(p)
                               if 0 <= q[0] < h and 0 <= q[1] < w and q not in cells})
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        axes = [(ar, ac) for ac in range(1, w - 1)
                for ar in (range(1, h - 1) if double else [None])]
        rng.shuffle(axes)
        for ar, ac in axes:
            f = _cut_features(cells, ar, ac, sides)
            if f is not None and (not witness or (f["gap"] and f["fragmented"] and f["short"])):
                return cells, ar, ac
    # A finite connected comb fallback; optional left-side growth changes its outline.
    ar, ac = h // 2, w // 2
    cells = {(r, ac - 1) for r in range(h)} | {(ar, c) for c in range(w)}
    cells |= {(0, ac), (0, ac + 1), (h - 2, ac)}
    cells |= {(r, w - 1) for r in range(ar - 1, h)}
    cells |= {(r, c) for r in range(h) for c in range(ac - 1) if rng.random() < .22}
    # Attach any optional cells by a horizontal path to the continuous left rail.
    for r, c in sorted(cells.copy()):
        if c < ac - 1:
            cells.update((r, x) for x in range(c, ac))
    f = _cut_features(cells, ar if double else None, ac, sides)
    _require(f and (not witness or (f["gap"] and f["fragmented"] and f["short"])), "fallback cut")
    return cells, ar if double else None, ac


def _sweep(groups, shape, upto):
    return set().union(*(_pose(v, k, shape, t) for t in range(upto + 1)
                         for k, v in groups.items()))


def _rooted_paths(safe):
    """Shortest safe paths starting at adjacent pairs on an outer edge."""
    h, w = safe.shape
    rim = _rim(safe.shape)
    roots = set()
    for p in sorted(rim):
        if safe[p] and any(q in rim and safe[q] for q in _neighbors(p)):
            roots.add(p)
    parent = {p: None for p in sorted(roots)}
    queue = deque(sorted(roots))
    while queue:
        p = queue.popleft()
        for q in _neighbors(p):
            if 0 <= q[0] < h and 0 <= q[1] < w and safe[q] and q not in parent:
                parent[q] = p
                queue.append(q)
    return parent, roots


def _path_wall(target, parent, roots, safe, rng):
    wall = {target}
    p = target
    while parent[p] is not None:
        p = parent[p]
        wall.add(p)
    rim = _rim(safe.shape)
    adjacent = sorted(q for q in _neighbors(p) if q in roots and q in rim)
    if adjacent:
        wall.add(adjacent[int(rng.integers(len(adjacent)))])
    # A small safe branch gives bars, hooks, and uneven barrier thickness.
    for _ in range(int(rng.integers(1, 6))):
        options = sorted({q for p in wall for q in _neighbors(p)
                          if 0 <= q[0] < safe.shape[0] and 0 <= q[1] < safe.shape[1]
                          and safe[q] and q not in wall})
        if options:
            wall.add(options[int(rng.integers(len(options)))])
    return wall


def _safe_mask(shape, forbidden):
    mask = np.ones(shape, dtype=bool)
    for p in forbidden:
        mask[p] = False
    return mask


def _make_pair(rng, double, jam, scale, roles, turn, reflect, witness, need_box):
    bg, body, mark, wall_color = roles
    for _ in range(72):
        ph, pw = map(int, rng.integers(scale[0], scale[1] + 1, size=2))
        cells, ar, ac = _shape(rng, ph, pw, double, witness)
        h = int(rng.integers(ph + 8, min(30, ph + 19) + 1))
        w = int(rng.integers(pw + 8, min(30, pw + 19) + 1))
        r0 = int(rng.integers(3, h - ph - 2))
        c0 = int(rng.integers(3, w - pw - 2))
        cells = {(r + r0, c + c0) for r, c in cells}
        ar = ar + r0 if ar is not None else None
        ac += c0
        sides = {"top", "right"} if double else {"top"}
        groups, _ = _partitions(cells, ar, ac, sides)
        kept = {k: v for k, v in groups.items() if set(k) & sides}
        shape = (h, w)
        axes = {(r, ac) for r in range(h)}
        if double:
            axes |= {(ar, c) for c in range(w)}
        reserve = axes | _bbox(cells)
        terminal = _limit(kept, shape)
        all_swept = _sweep(kept, shape, terminal)
        safe_full = _safe_mask(shape, reserve | all_swept)
        parent, roots = _rooted_paths(safe_full)
        if not roots:
            continue
        # A barrier in a reachable void rejects treating a sparse shape as a solid box.
        box_swept = _sweep({k: _bbox(v) for k, v in kept.items()}, shape, terminal)
        voids = sorted((box_swept - all_swept - reserve) & set(parent))
        if need_box and not voids:
            continue
        choices = voids if voids else sorted(set(parent) - _rim(shape))
        if not choices:
            continue
        walls = _path_wall(choices[int(rng.integers(len(choices)))], parent, roots, safe_full, rng)
        expected_tick = terminal
        if jam:
            options = []
            for k, v in kept.items():
                peer_terminal = max(_limit({q: x}, shape) for q, x in kept.items() if q != k)
                for contact in range(2, min(_limit({k: v}, shape), peer_terminal - 1, 10) + 1):
                    if witness and not any(min(map(abs, _goal_delta(x, q, shape))) < contact - 1
                                           < max(map(abs, _goal_delta(x, q, shape)))
                                           for q, x in kept.items()):
                        continue
                    options.append((k, contact))
            rng.shuffle(options)
            chosen = None
            for k, contact in options:
                peers = {q: v for q, v in kept.items() if q != k}
                forbidden = reserve | _sweep(kept, shape, contact - 1) | _sweep(peers, shape, terminal)
                safe_pre = _safe_mask(shape, forbidden)
                pmap, proots = _rooted_paths(safe_pre)
                targets = sorted((_pose(kept[k], k, shape, contact) - forbidden) & set(pmap))
                if targets:
                    target = targets[int(rng.integers(len(targets)))]
                    chosen = _path_wall(target, pmap, proots, safe_pre, rng)
                    expected_tick = contact - 1
                    break
            if chosen is None:
                continue
            walls |= chosen
        rim = _rim(shape)
        if len(walls) < 3 or len(walls & rim) >= len(rim) // 2 - 2 or len(walls) > h * w // 3:
            continue
        inp = np.full(shape, bg, dtype=np.int8)
        for p in cells:
            inp[p] = body
        for p in walls:
            inp[p] = wall_color
        inp[0, ac] = mark
        if double:
            inp[ar, -1] = mark
        info = _parse(inp)
        d = _diagnostics(info)
        if d["success"] == jam or d["tick"] != expected_tick or d["tick"] < 1:
            continue
        if jam and not d["coupled"]:
            continue
        if need_box and not d["bbox_witness"]:
            continue
        if need_box and not d["unequal_axes"]:
            continue
        if witness and not jam and not d["without_collective_B"]:
            continue
        if witness and not (info["gap"] and info["fragmented"] and info["short"]):
            continue
        out = _render(info, expected_tick)
        # Labels use the constructed contact time; solve independently recovers it from pixels.
        _require(np.array_equal(solve(inp), out), "constructed clock disagrees with input")
        def orient(a):
            return np.array(np.rot90(np.fliplr(a) if reflect else a, turn), copy=True)
        return {"input": orient(inp), "output": orient(out)}
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in {lo}..{hi}")
    return value


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 2, 10, "num_train")
    m = _count(num_test, 1, 4, "num_test")
    if n <= m:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    base_roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    vary_palette = bool(rng.integers(2))
    arity_mode = int(rng.integers(3))  # single only, double only, mixed
    shift = int(rng.integers(2))  # larger source, or held-out rotation
    base_turn = int(rng.integers(4))
    reflection = bool(rng.integers(2))
    training = [(arity_mode == 1, False, True, True),
                (arity_mode != 0, True, True, False)]
    for _ in range(n - 2):
        double = bool(rng.integers(2)) if arity_mode == 2 else arity_mode == 1
        training.append((double, bool(rng.integers(2)), False, False))
    rng.shuffle(training)
    testing = [(bool(rng.integers(2)) if arity_mode == 2 else arity_mode == 1,
                bool(rng.integers(2)), False, False) for _ in range(m)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, schedule in (("train", training), ("test", testing)):
        for double, jam, witness, need_box in schedule:
            roles = tuple(map(int, rng.choice(10, 4, replace=False))) if vary_palette else base_roles
            if shift == 0:
                scale = (5, 8) if split == "train" else (9, 11)
                turn = int(rng.integers(4))
            else:
                scale = (5, 10)
                turn = base_turn if split == "train" else (base_turn + int(rng.integers(1, 4))) % 4
            for _ in range(16):
                pair = _make_pair(rng, double, jam, scale, roles, turn, reflection, witness, need_box)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(type(puzzle) is dict and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal counts")
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(type(pair) is dict and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _parse(inp)
            d = _diagnostics(info)
            expected = _render(info, d["tick"])
            _require(np.array_equal(out, expected), "incorrect shared-clock transformation")
            _require(not np.array_equal(inp, out), "identity pair")
            _require(d["tick"] >= 1, "generated examples must visibly move")
            _require(np.array_equal(inp == info["wall"], out == info["wall"]), "wall material changed")
            _require(sum(out[p] == info["body"] for p in info["fixed"]) == len(info["fixed"]),
                     "axial payload moved")
            _require(int((out == info["body"]).sum()) == len(info["cells"]) - d["deleted"],
                     "payload lost, merged, or duplicated")
            for k, cells in info["retained"].items():
                target = _pose(cells, k, inp.shape, d["tick"])
                _require(not target & info["walls"], "footprint overlaps a wall")
                _require(all(out[p] == info["body"] for p in target), "partition not translated collectively")
            if not d["success"]:
                _require(any(_pose(v, k, inp.shape, d["tick"] + 1) & info["walls"]
                             for k, v in info["retained"].items()), "no first-contact witness")
            records[split].append((info, d))
    training = records["train"]
    _require(any(d["success"] for _, d in training), "training lacks complete docking")
    _require(any(d["coupled"] and d["without_A"] for _, d in training), "training lacks a shared jam")
    _require(any(d["coupled"] and d["partial_clamp"] for _, d in training),
             "training lacks a visibly clamped jam trajectory")
    _require(any(d["bbox_witness"] and d["success"] for _, d in training), "training lacks void-aware collision")
    _require(any(i["gap"] and i["fragmented"] and i["short"] for i, _ in training),
             "training lacks guide-gap / collective-fragment / axial-endpoint witnesses")
    _require(any(d["without_collective_B"] for _, d in training), "collective cut concept is not exposed")
    arities = {len(i["markers"]) for i, _ in training}
    _require({len(i["markers"]) for i, _ in records["test"]} <= arities, "unseen marker arity")
    if 2 in {len(i["markers"]) for i, _ in records["test"]}:
        _require(any(len(i["markers"]) == 2 and d["deleted"] > 0 for i, d in training),
                 "quadrant deletion not demonstrated")


def self_test():
    """Bounded 42-seed variation audit, counterfactuals, and negative checks."""
    # Coordinate-specified examples check clamping, simultaneous first contact,
    # guide gaps and collective fragment spacing without construction labels.
    source = {(r, c) for r in (2, 4) for c in range(4, 9)} | {(3, 4)}
    for mode in ("complete", "jam", "tie", "immediate"):
        inp = np.zeros((11, 13), dtype=np.int8)
        for p in source:
            inp[p] = 2
        inp[0, 6] = 3
        if mode == "complete":
            walls = {(10, 0), (10, 1), (10, 2)}
            left = {(r, c) for r in (0, 2) for c in (0, 1)} | {(1, 0)}
            right = {(r, c) for r in (0, 2) for c in (11, 12)}
        elif mode in ("jam", "tie"):
            walls = {(0, 0), (1, 0), (2, 0)}
            if mode == "tie":
                walls |= {(0, 12), (1, 12), (2, 12)}
            left = {(r, c) for r in (0, 2) for c in (1, 2)} | {(1, 1)}
            right = {(r, c) for r in (0, 2) for c in (10, 11)}
        else:
            walls = {(1, c) for c in range(4)} | {(2, 0)}
            left = {p for p in source if p[1] < 6}
            right = {p for p in source if p[1] > 6}
        for p in walls:
            inp[p] = 8
        expected = inp.copy()
        expected[expected == 2] = 0
        for p in left | right | {(2, 6), (4, 6)}:
            expected[p] = 2
        for r in (0, 1, 3):
            expected[r, 6] = 3
        for rotation in range(4):
            for reflection in (False, True):
                def orient(a):
                    return np.rot90(np.fliplr(a) if reflection else a, rotation)
                _require(np.array_equal(solve(orient(inp)), orient(expected)), "hand-set " + mode)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    structures, shapes, palettes, cuts, distances = set(), set(), set(), set(), set()
    jam_count = success_count = 0
    source_sizes, wall_sizes = set(), set()
    orientation_by_split = {"train": set(), "test": set()}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                untouched = inp.copy()
                _require(np.array_equal(solve(inp), pair["output"]), "input-only solver mismatch")
                _require(np.array_equal(inp, untouched), "solver mutated input")
                info = _parse(inp)
                d = _diagnostics(info)
                success_count += d["success"]
                jam_count += not d["success"]
                roles = tuple(info[k] for k in ("bg", "body", "mark", "wall"))
                norm = np.zeros_like(inp)
                for code, role in enumerate(roles):
                    norm[inp == role] = code
                structures.add(_key(norm))
                shapes.add(inp.shape)
                palettes.add(roles)
                cuts.add(d["partition_sizes"])
                distances.add(d["distance_pairs"])
                source_sizes.add(len(info["cells"]))
                wall_sizes.add(len(info["walls"]))
                orientation_by_split[split].add(tuple(sorted(info["sides"])))
    _require(len(structures) >= 180, "geometry variation collapsed")
    _require(len(shapes) >= 50 and any(h != w for h, w in shapes), "canvas variation collapsed")
    _require(len(palettes) >= 30, "palette role variation collapsed")
    _require(len(cuts) >= 70 and len(distances) >= 100, "cut or displacement variation collapsed")
    _require(len(source_sizes) >= 35 and len(wall_sizes) >= 15, "material counts collapsed")
    _require(min(jam_count, success_count) >= 60, "outcome variation collapsed")
    _require(all(len(v) == 8 for v in orientation_by_split.values()), "marker configurations missing")
    _require(_fingerprint(generate_puzzle(7, 6, 3)) == _fingerprint(generate_puzzle(7, 6, 3)),
             "seed determinism")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(100 + 10 * n + m, n, m)
            _require(len(p["train"]) == n and len(p["test"]) == m, "explicit counts ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
           for s in ("train", "test")}
    info = _parse(bad["train"][0]["input"])
    p = next(iter(info["walls"]))
    bad["train"][0]["output"][p] = info["bg"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted changed wall")
    return {"seeds": 42, "pairs": 210, "structures": len(structures),
            "shapes": len(shapes), "palettes": len(palettes), "cut_sizes": len(cuts),
            "distance_profiles": len(distances), "jams": int(jam_count),
            "complete_releases": int(success_count), "independent_verification": False}


def _fingerprint(puzzle):
    return tuple(_key(pair[k]) for s in ("train", "test") for pair in puzzle[s]
                 for k in ("input", "output"))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
