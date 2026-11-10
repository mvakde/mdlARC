"""Perforated printing plates: geometry selects motion; true holes print its sweep.

Only standard-library modules and NumPy are used. No file I/O or latent solver
state. Generation paints local expected scenes; solving reconstructs global
objects and trajectories from input cells alone.
"""
from __future__ import annotations

import hashlib
import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRECTIONS:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) + 1, max(cols) + 1


def _rectangle(box):
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _paint(a, points, color):
    for p in points:
        a[p] = color


def _plate_candidate(cells, grid, background):
    box = r0, c0, r1, c1 = _box(cells)
    if min(r1 - r0, c1 - c0) < 3:
        return None
    rim = ([(r0, c) for c in range(c0, c1)] + [(r1 - 1, c) for c in range(c0, c1)]
           + [(r, c0) for r in range(r0, r1)] + [(r, c1 - 1) for r in range(r0, r1)])
    colors = {int(grid[p]) for p in rim}
    if len(colors) != 1 or background in colors:
        return None
    footprint = _rectangle(box)
    if cells != {p for p in footprint if grid[p] != background}:
        return None
    return {"box": box, "cells": cells, "footprint": footprint, "color": colors.pop()}


def _parse_fork(cells):
    """Find a base and three positive extruded runs, without length templates."""
    parses = []
    for outward in _DIRECTIONS:
        dr, dc = outward
        perpendicular = -dc, dr
        kp = [r * dr + c * dc for r, c in cells]
        qp = [r * perpendicular[0] + c * perpendicular[1] for r, c in cells]
        kmin, qmin = min(kp), min(qp)
        local = {(r * dr + c * dc - kmin,
                  r * perpendicular[0] + c * perpendicular[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum(1 for k, x in local if x == q and k > 0) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected.update((k, q) for q, n in enumerate(lengths) for k in range(1, n + 1))
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, n = q, lengths[q]
            while q < width and lengths[q] == n:
                q += 1
            runs.append((start, q, n))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        a, b, tip = runs[1]
        if a + b != width or runs[0][0] != 0 or runs[-1][1] != width:
            continue
        if runs[0][1] != width - runs[-1][0]:
            continue
        parses.append({"out": outward, "perp": perpendicular, "kmin": kmin, "qmin": qmin,
                       "width": width, "q0": a, "span": b - a, "stop": tip + 1,
                       "lengths": tuple(run[2] for run in runs), "cells": cells})
    _require(len(parses) == 1, "fork must have one geometric interpretation")
    return parses[0]


def _cell(fork, k, q):
    k, q = k + fork["kmin"], q + fork["qmin"]
    dr, dc = fork["out"]
    pr, pc = fork["perp"]
    return k * dr + q * pr, k * dc + q * pc


def _candidate(fork, plate, grid, background):
    local = {(r * fork["out"][0] + c * fork["out"][1] - fork["kmin"],
              r * fork["perp"][0] + c * fork["perp"][1] - fork["qmin"])
             for r, c in plate["footprint"]}
    qs = {q for _, q in local}
    if qs != set(range(fork["q0"], fork["q0"] + fork["span"])):
        return None
    k0, k1 = min(k for k, _ in local), max(k for k, _ in local)
    if k0 <= fork["stop"]:
        return None
    route = {_cell(fork, k, q) for k in range(fork["stop"], k1 + 1) for q in qs}
    h, w = grid.shape
    if any(not (0 <= r < h and 0 <= c < w) or
           ((r, c) not in plate["footprint"] and grid[r, c] != background) for r, c in route):
        return None
    final = {_cell(fork, k, q) for k in range(fork["stop"], fork["stop"] + k1 - k0 + 1)
             for q in qs}
    return {"distance": k0 - fork["stop"], "length": k1 - k0 + 1,
            "route": route, "final": final}


def _analyze(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    _require(int(counts.max()) > grid.size / 2, "background must exceed half the canvas")
    background = int(colors[counts.argmax()])
    plates, forks = [], []
    for cells in _components(grid != background):
        plate = _plate_candidate(cells, grid, background)
        if plate is not None:
            plates.append(plate)
        else:
            color_set = {int(grid[p]) for p in cells}
            _require(len(color_set) == 1, "fork must be monochrome")
            fork = _parse_fork(cells)
            fork["color"] = color_set.pop()
            forks.append(fork)
    fills = tuple(sorted({p["color"] for p in plates}))
    _require(len(fills) == 2 and forks, "need two plate materials and at least one fork")
    _require(all(f["color"] not in fills for f in forks), "fork and plate roles must differ")
    for plate in plates:
        color = plate["color"]
        _require(all(int(grid[p]) in (background, *fills) for p in plate["footprint"]),
                 "unexpected plate corruption")
        _require(sum(grid[p] == color for p in plate["footprint"]) >= .70 * len(plate["footprint"]),
                 "too little material evidence")
        plate["holes"] = sorted(p for p in plate["footprint"] if grid[p] == background)
        plate["wrong"] = sorted(p for p in plate["footprint"] if grid[p] not in (background, color))
    matches, used, swept = [], set(), set()
    for fi, fork in enumerate(forks):
        candidates = []
        for pi, plate in enumerate(plates):
            m = _candidate(fork, plate, grid, background)
            if m is not None:
                candidates.append(dict(m, fi=fi, pi=pi))
        _require(len(candidates) <= 1, "ambiguous receiver")
        if candidates:
            m = candidates[0]
            _require(m["pi"] not in used, "plate matches multiple forks")
            _require(swept.isdisjoint(m["route"]), "intersecting swept rectangles")
            swept.update(m["route"])
            used.add(m["pi"])
            matches.append(m)
    return {"grid": grid, "background": background, "fills": fills, "plates": plates,
            "forks": forks, "matches": matches, "unmatched": set(range(len(plates))) - used}


def _ring(shape, p):
    r, c = p
    return [(rr, cc) for rr in range(max(0, r - 1), min(shape[0], r + 2))
            for cc in range(max(0, c - 1), min(shape[1], c + 2)) if (rr, cc) != p]


def _render(a, *, no_print=False, wrong_as_holes=False, stationary=False,
            own_ring=False, centers_first=False, reverse=False):
    """Simultaneous canvas-coordinate impressions. Options are ablation probes."""
    out = a["grid"].copy()
    for p in a["plates"]:
        _paint(out, p["footprint"], a["background"])
    rings, centers = {}, {}
    matches = list(reversed(a["matches"])) if reverse else a["matches"]
    for match in matches:
        plate, fork = a["plates"][match["pi"]], a["forks"][match["fi"]]
        distance = 0 if stationary else match["distance"]
        dr, dc = fork["out"]
        final = {(r - distance * dr, c - distance * dc) for r, c in plate["footprint"]}
        _paint(out, final, plate["color"])
        if no_print:
            continue
        holes = plate["holes"] + (plate["wrong"] if wrong_as_holes else [])
        if reverse:
            holes = list(reversed(holes))
        other = next(color for color in a["fills"] if color != plate["color"])
        ink = plate["color"] if own_ring else other
        for r, c in holes:
            for step in range(distance + 1):
                center = r - step * dr, c - step * dc
                _require(center not in centers or centers[center] == fork["color"], "center conflict")
                centers[center] = fork["color"]
                for q in _ring(out.shape, center):
                    _require(q not in rings or rings[q] == ink, "ring conflict")
                    rings[q] = ink
    layers = (centers, rings) if centers_first else (rings, centers)
    for layer in layers:
        for point, color in layer.items():
            out[point] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive everything, including both inks and stopping points, from input."""
    return _render(_analyze(input_grid))


def _int(rng, low, high):
    return int(rng.integers(low, high + 1))


def _module(rng, bh, bw, palette, index, mode, active, style, rotation):
    """Construct a local scene and answer without using the global parser."""
    background, f0, f1, *fork_colors = palette
    color = (f0, f1)[index % 2]
    fork_color = fork_colors[index]
    depth, cross = (bh, bw) if rotation % 2 == 0 else (bw, bh)
    minimum = 4 if (style in ("teach", "merge") or not active) else 3
    span = _int(rng, minimum, min(7, cross - 4))
    choices = [(side, gap) for side in (1, 2) for gap in (1, 2)
               if span + 2 * (side + gap) <= cross]
    side, clearance = choices[_int(rng, 0, len(choices) - 1)]
    width = span + 2 * (side + clearance)
    q0 = side + clearance
    length = _int(rng, 4 if style in ("teach", "merge") else 3, min(7, depth - 4))
    cap = min(6, depth - length - 2)
    _require(cap >= 2, "local box too small")
    if mode == "standard":
        middle = _int(rng, 1, cap - 1)
        left = right = _int(rng, middle + 1, cap)
    elif mode == "long":
        middle = _int(rng, 2, cap)
        left, right = _int(rng, 1, middle - 1), _int(rng, 1, middle - 1)
    else:
        left, middle, right = (_int(rng, 1, cap) for _ in range(3))
        if mode == "unequal" and left == right:
            right = right % cap + 1
    travel_gap = _int(rng, 1, min(7, depth - max(left, middle, right) - length - 1))
    start = max(left, middle, right) + 1 + travel_gap
    height = start + length
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    inp[0, :] = fork_color
    inp[1:left + 1, :side] = fork_color
    inp[1:middle + 1, q0:q0 + span] = fork_color
    inp[1:right + 1, width - side:] = fork_color
    out[:] = inp
    actual_span = span if active else span - 1
    inp[start:start + length, q0:q0 + actual_span] = color
    interior = [(k, q) for k in range(1, length - 1) for q in range(1, actual_span - 1)]
    holes, wrong = [], []
    if style == "teach":
        k = _int(rng, 1, length - 2)
        q = int(rng.choice([1, actual_span - 2]))
        holes = [(k, q)]
        # A distinct perpendicular coordinate makes the wrong-as-hole shortcut
        # observably add a center track, even when axial tracks overlap.
        wrong = [(_int(rng, 1, length - 2), actual_span - 1 - q)]
    elif style == "merge":
        k = _int(rng, 1, length - 2)
        q = _int(rng, 1, actual_span - 3)
        holes = [(k, q), (k, q + 1)]
    else:
        target = _int(rng, 1, min(6, max(1, len(interior) // 3)))
        for i in rng.permutation(len(interior)):
            p = interior[int(i)]
            trial = inp[start:start + length, q0:q0 + actual_span].copy()
            for hk, hq in holes + [p]:
                trial[hk, hq] = background
            if len(_components(trial != background)) == 1:
                holes.append(p)
            if len(holes) == target:
                break
    if style != "teach":
        pool = [p for p in interior if p not in holes]
        rng.shuffle(pool)
        wrong = pool[:min(len(pool), _int(rng, 0, 2))]
    for k, q in holes:
        inp[start + k, q0 + q] = background
    other = f1 if color == f0 else f0
    for k, q in wrong:
        inp[start + k, q0 + q] = other
    if active:
        out[middle + 1:middle + 1 + length, q0:q0 + span] = color
        distance = start - middle - 1
        center_points = set()
        # Independent local renderer: row intervals enumerate time explicitly.
        for hk, hq in holes:
            for row in range(start + hk, start + hk - distance - 1, -1):
                col = q0 + hq
                for rr in range(row - 1, row + 2):
                    for cc in range(col - 1, col + 2):
                        if (rr, cc) != (row, col):
                            out[rr, cc] = other
                center_points.add((row, col))
        for point in center_points:
            out[point] = fork_color
    return np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()


def _make_pair(rng, count, ordinal, palette, global_rotation, teach_index):
    for attempt in range(24):
        h, w = _int(rng, 20, 30), _int(rng, 24 if count == 2 else 26, 30)
        # Every band fits the minimum geometry, even with a horizontal module.
        available = w - count + 1
        base = available // count
        widths = [base] * count
        for i in rng.permutation(count)[:available - base * count]:
            widths[int(i)] += 1
        if count == 2 and base >= 12:
            delta = _int(rng, -1, 1)
            widths[0] += delta
            widths[1] -= delta
        inp = np.full((h, w), palette[0], dtype=np.int8)
        out = inp.copy()
        cursor = 0
        try:
            for j, band_width in enumerate(widths):
                active = not (teach_index == 1 and j == 1)
                if teach_index == 0:
                    style = "teach"
                    mode = "standard" if j == 0 else "long"
                elif teach_index == 1 and j == 0:
                    style, mode = "merge", "unequal"
                elif count == 3 and j == 0:
                    style, mode = "merge", str(rng.choice(["standard", "long", "unequal"]))
                else:
                    style, mode = "random", str(rng.choice(["standard", "long", "unequal", "free"]))
                rotation = int(rng.choice([0, 2])) if attempt == 23 else _int(rng, 0, 3)
                # Swap fill roles freely; both remain present because the first
                # two indices are consecutive modulo two.
                local_in, local_out = _module(rng, h, band_width, palette, j, mode, active, style, rotation)
                mh, mw = local_in.shape
                r, c = _int(rng, 0, h - mh), cursor + _int(rng, 0, band_width - mw)
                inp[r:r + mh, c:c + mw] = local_in
                out[r:r + mh, c:c + mw] = local_out
                cursor += band_width + 1
            inp, out = np.rot90(inp, global_rotation).copy(), np.rot90(out, global_rotation).copy()
            a = _analyze(inp)
            expected_active = count - (1 if teach_index == 1 else 0)
            if len(a["forks"]) != count or len(a["plates"]) != count or len(a["matches"]) != expected_active:
                continue
            if not np.array_equal(_render(a), out):
                continue
            if teach_index == 0 and np.array_equal(_render(a, wrong_as_holes=True), out):
                continue
            if teach_index == 1 and np.array_equal(_render(a, centers_first=True), out):
                continue
            return {"input": inp, "output": out}
        except ValueError:
            if attempt == 23:
                raise
    raise RuntimeError("bounded construction failed")


def _check_count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _check_count(num_train, 2, 10, "num_train")
    num_test = _check_count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.permutation(10)[:6]))
    policy, frame = _int(rng, 0, 2), _int(rng, 0, 3)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            ordinal = index if split == "train" else num_train + index
            palette = base_palette
            if policy == 1:
                rest = [c for c in range(10) if c not in base_palette[:3]]
                palette = base_palette[:3] + tuple(map(int, rng.choice(rest, 3, replace=False)))
            elif policy == 2:
                palette = tuple(map(int, rng.permutation(10)[:6]))
            if rng.integers(2):
                palette = (palette[0], palette[2], palette[1], *palette[3:])
            count = 2 if split == "train" else 3
            teach_index = index if split == "train" and index < 2 else -1
            for _ in range(12):
                pair = _make_pair(rng, count, ordinal, palette,
                                  (frame + (index if split == "train" else 2 + index)) % 4,
                                  teach_index)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded distinct-input construction failed")
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def _witnesses(a, out):
    no_b = not np.array_equal(_render(a, no_print=True), out)
    no_a = not np.array_equal(_render(a, stationary=True), out)
    wrong = not np.array_equal(_render(a, wrong_as_holes=True), out)
    precedence = not np.array_equal(_render(a, centers_first=True), out)
    blank = any(out[p] == a["background"] for m in a["matches"] for p in m["route"] - m["final"])
    both = {a["plates"][m["pi"]]["color"] for m in a["matches"]} == set(a["fills"])
    inks = len({a["forks"][m["fi"]]["color"] for m in a["matches"]}) >= 2
    standard = any(a["forks"][m["fi"]]["lengths"][1] <
                   min(a["forks"][m["fi"]]["lengths"][::2]) for m in a["matches"])
    long = any(a["forks"][m["fi"]]["lengths"][1] >
               max(a["forks"][m["fi"]]["lengths"][::2]) for m in a["matches"])
    return {"printing": no_b, "motion": no_a, "wrong_fill_exclusion": wrong,
            "center_precedence": precedence, "blank_body": blank,
            "both_complements": both, "receiver_inks": inks, "short_middle": standard,
            "long_middle": long, "selection": bool(a["unmatched"])}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _check_count(nt, 2, 10, "num_train")
    _check_count(ne, 1, 4, "num_test")
    _require(nt > ne, "invalid split counts")
    seen, witnesses = set(), {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "changed canvas shape")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(inp)
            expected = _render(a)
            _require(np.array_equal(expected, out), "input-derived answer disagrees")
            _require(not np.array_equal(inp, out) and bool(a["matches"]), "no active transformation")
            for fork in a["forks"]:
                _require(min(fork["lengths"]) >= 1, "nonpositive prong")
                _require(all(out[p] == fork["color"] for p in fork["cells"]), "fork changed")
            for pi in a["unmatched"]:
                _require(all(out[p] == a["background"] for p in a["plates"][pi]["footprint"]),
                         "unmatched plate not erased")
            for m in a["matches"]:
                plate, fork = a["plates"][m["pi"]], a["forks"][m["fi"]]
                _require(bool(plate["holes"]), "selected plates must demonstrate printing")
                for r, c in plate["holes"]:
                    for t in range(m["distance"] + 1):
                        p = r - t * fork["out"][0], c - t * fork["out"][1]
                        _require(set(_ring(inp.shape, p)) <= m["route"], "print leaves its sweep")
                        _require(out[p] == fork["color"], "center lost to a ring")
            if split == "train":
                for name, value in _witnesses(a, out).items():
                    witnesses[name] = witnesses.get(name, False) or value
    _require(all(witnesses.values()), "training lacks a shortcut-distinguishing witness")


def _normalized_hash(a):
    roles, flat = {}, []
    for c in a.flat:
        c = int(c)
        flat.append(roles.setdefault(c, len(roles)))
    return hashlib.sha256(bytes(a.shape) + bytes(flat)).hexdigest()


def self_test():
    """Stream 42 seeds; keep only aggregate counters and short hashes, no files."""
    shapes, structures, puzzles, geometry, directions = set(), set(), set(), set(), set()
    backgrounds, fills, spans, distances, holes_seen, defects = set(), set(), set(), set(), set(), set()
    merged = mixed_axes = inactive = pairs = 0
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(bytes(inp.shape) + inp.tobytes() + out.tobytes())
                a = _analyze(inp)
                shapes.add(inp.shape)
                structures.add(_normalized_hash(inp))
                backgrounds.add(a["background"])
                fills.update(a["fills"])
                inactive += bool(a["unmatched"])
                axis = set()
                for f in a["forks"]:
                    directions.add(f["out"])
                    geometry.add((f["width"], f["span"], f["lengths"]))
                    spans.add(f["span"])
                    axis.add(f["out"][0] == 0)
                mixed_axes += len(axis) == 2
                for m in a["matches"]:
                    p = a["plates"][m["pi"]]
                    distances.add(m["distance"])
                    holes_seen.add(len(p["holes"]))
                    defects.add(len(p["wrong"]))
                    merged += any(max(abs(r - rr), abs(c - cc)) <= 2
                                  for i, (r, c) in enumerate(p["holes"]) for rr, cc in p["holes"][:i])
                _require(np.array_equal(_render(a, reverse=True), out), "order-dependent stamps")
                pairs += 1
        puzzles.add(digest.hexdigest())
        pair = puzzle["test"][0]
        rotation = seed % 4
        _require(np.array_equal(solve(np.rot90(pair["input"], rotation)),
                                np.rot90(pair["output"], rotation)), "rotation changes rule")
        perm = np.random.default_rng(seed + 1000).permutation(10)
        _require(np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]]), "palette changes rule")
        if seed == 0:
            first = puzzle
    _require(len(puzzles) == 42 and len(structures) == 210, "insufficient structural variation")
    _require(len(shapes) >= 35 and len(geometry) >= 100, "insufficient geometric variation")
    _require(directions == set(_DIRECTIONS) and len(spans) >= 4, "missing orientations or widths")
    _require(len(backgrounds) >= 8 and len(fills) >= 8 and len(distances) >= 7, "narrow roles or motion")
    _require(merged >= 42 and mixed_axes >= 30 and inactive == 42, "missing interactions")
    _require(max(holes_seen) >= 4 and max(defects) >= 2, "narrow corruption")
    repeat = generate_puzzle(0)
    _require(all(np.array_equal(p[k], q[k]) for s in ("train", "test")
                 for p, q in zip(first[s], repeat[s]) for k in ("input", "output")), "nondeterminism")
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1234 + nt, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid label accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(puzzles),
            "normalized_inputs": len(structures), "shapes": len(shapes),
            "fork_geometries": len(geometry), "directions": len(directions),
            "spans": sorted(spans), "travel_distances": sorted(distances),
            "hole_counts": sorted(holes_seen), "wrong_fill_counts": sorted(defects),
            "background_colors": len(backgrounds), "fill_colors": len(fills),
            "merged_plate_cases": merged, "mixed_axis_pairs": mixed_axes, "inactive_pairs": inactive}
