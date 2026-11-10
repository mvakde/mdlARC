"""Unfold a ribbed spine in its intrinsic moving coordinate frame.

Only NumPy and the Python standard library are used. Neither generation nor
solving loads parent files, stored puzzles, or construction metadata.
"""

import copy
import json
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _add(p, v, k=1):
    return p[0] + k * v[0], p[1] + k * v[1]


def _step(a, b):
    return b[0] - a[0], b[1] - a[1]


def _turns(path):
    steps = [_step(a, b) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _neighbors(p):
    return [_add(p, v) for v in DIRS]


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _components(cells):
    cells = set(cells)
    result = []
    while cells:
        p = min(cells)
        cells.remove(p)
        todo, group = [p], {p}
        while todo:
            for q in _neighbors(todo.pop()):
                if q in cells:
                    cells.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return result


def _trace(cells):
    """One monochromatic induced chain, oriented by its higher endpoint."""
    if not 6 <= len(cells) <= 30:
        raise ValueError("Spine length must be 6..30")
    adjacent = {p: [q for q in _neighbors(p) if q in cells] for p in cells}
    ends = [p for p in cells if len(adjacent[p]) == 1]
    if (len(ends) != 2 or ends[0][0] == ends[1][0]
            or any(len(v) not in (1, 2) for v in adjacent.values())):
        raise ValueError("Spine must have two endpoints on different rows")
    previous, current = None, min(ends)
    path = []
    while current is not None:
        path.append(current)
        onward = [q for q in adjacent[current] if q != previous]
        previous, current = current, onward[0] if onward else None
        if len(path) > len(cells):
            raise ValueError("Spine cycle")
    if len(path) != len(cells):
        raise ValueError("Disconnected spine")
    return path


def _straight_station(path, i):
    return (1 <= i < len(path) - 2
            and _step(path[i-1], path[i]) == _step(path[i], path[i+1])
            == _step(path[i+1], path[i+2]))


def _rib_cells(path, i, side, radius):
    t = _step(path[i], path[i+1])
    n = (-t[1] * side, t[0] * side)
    return {_add(path[j], n, k) for j in (i, i+1)
            for k in range(1, radius+1)}


def _parse_with_color(g, background, spine):
    cells = {tuple(map(int, p)) for p in np.argwhere(g == spine)}
    path = _trace(cells)
    index = {p: i for i, p in enumerate(path)}
    payload = {tuple(map(int, p)) for p in np.argwhere((g != background) & (g != spine))}
    groups = _components(payload)
    if not groups:
        raise ValueError("At least one rib is required")
    ribs = []
    for group in groups:
        r0, r1 = min(r for r, c in group), max(r for r, c in group)
        c0, c1 = min(c for r, c in group), max(c for r, c in group)
        if len(group) != (r1-r0+1) * (c1-c0+1):
            raise ValueError("Ribs must be full rectangles")
        touched = {q for p in group for q in _neighbors(p) if q in cells}
        positions = sorted(index[q] for q in touched)
        if len(positions) != 2 or positions[1] != positions[0] + 1:
            raise ValueError("Each rib must hinge on two consecutive spine cells")
        i = positions[0]
        if not _straight_station(path, i):
            raise ValueError("Ribs cannot hinge on endpoints or turns")
        radius = len(group) // 2
        if not 2 <= radius <= 14:
            raise ValueError("Rib reach must be 2..14")
        matches = [s for s in (-1, 1) if _rib_cells(path, i, s, radius) == group]
        if len(matches) != 1:
            raise ValueError("Rib is not perpendicular to its hinge")
        side = matches[0]
        t = _step(path[i], path[i+1])
        normal = (-t[1] * side, t[0] * side)
        # Columns are measured outward from the hinge; side determines where
        # that order is placed in the final canvas.
        block = np.array([[g[_add(path[j], normal, k)] for k in range(1, radius+1)]
                          for j in (i, i+1)], dtype=np.int8)
        ribs.append({"index": i, "side": side, "radius": radius,
                     "block": block, "tangent": t, "cells": group})
    # A rib component touches exactly its two hinge pixels, so all other
    # contacts, branches through ribs, and accidental bridges are rejected.
    ribs.sort(key=lambda rib: (rib["index"], rib["side"]))
    return {"background": background, "spine": spine, "path": path, "ribs": ribs}


def _analyze(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ValueError("A clear background border is required")
    candidates = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        try:
            candidates.append(_parse_with_color(g, bg, color))
        except ValueError:
            pass
    if len(candidates) != 1:
        raise ValueError("Input must identify exactly one ribbed spine")
    return candidates[0]


def _unfold(d, order=None, all_right=False, swap_rows=False):
    radius = max(r["radius"] for r in d["ribs"])
    n = len(d["path"])
    out = np.full((n, 2*radius+1), d["background"], dtype=np.int8)
    out[:, radius] = d["spine"]
    for rib in d["ribs"]:
        side = 1 if all_right else rib["side"]
        block = rib["block"][::-1] if swap_rows else rib["block"]
        for row in range(2):
            # all_right is a diagnostic alternative, never a solver mode.
            for k in range(1, rib["radius"]+1):
                target = radius + side*k
                if all_right and out[rib["index"]+row, target] != d["background"]:
                    continue
                out[rib["index"]+row, target] = block[row, k-1]
    return out if order is None else out[order].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only the input, trace the spine, and unfold its attached ribs."""
    return _unfold(_analyze(input_grid))


def _evidence(d):
    correct = _unfold(d)
    raster = sorted(range(len(d["path"])), key=lambda i: d["path"][i])
    reverse_raster = raster[::-1]
    columns = sorted({p[1] for p in d["path"]})
    sweeps = []
    for cs in (columns, columns[::-1]):
        for reverse_last in (False, True):
            sweeps.append([i for c in cs for i in sorted(
                (i for i, p in enumerate(d["path"]) if p[1] == c),
                key=lambda i: d["path"][i][0], reverse=reverse_last and c == cs[-1])])
    return {
        "topology": all(not np.array_equal(correct, _unfold(d, order=o))
                        for o in (raster, reverse_raster)),
        "sweep": all(not np.array_equal(correct, _unfold(d, order=o)) for o in sweeps),
        "sides": not np.array_equal(correct, _unfold(d, all_right=True)),
        "rotation": not np.array_equal(correct, _unfold(d, swap_rows=True)),
        "padding": len({r["radius"] for r in d["ribs"]}) > 1,
        "duplicate": len({_key(r["block"]) for r in d["ribs"]}) < len(d["ribs"]),
        "late_max": max(r["radius"] for r in d["ribs"]) > d["ribs"][0]["radius"],
        "above": min(r for r, c in d["path"]) < d["path"][0][0],
    }


def _walk(rng, length, persistence):
    path, used, previous = [(0, 0)], {(0, 0)}, None
    for _ in range(length - 1):
        choices, weights = [], []
        for t in DIRS:
            p = _add(path[-1], t)
            if p in used or sum(q in used for q in _neighbors(p)) != 1:
                continue
            choices.append((p, t))
            weights.append(persistence if t == previous else 1.0)
        if not choices:
            return None
        w = np.asarray(weights)
        p, previous = choices[int(rng.choice(len(choices), p=w/w.sum()))]
        path.append(p)
        used.add(p)
    return path


def _transform_path(path, rotation, reflection):
    path = [(r, -c if reflection else c) for r, c in path]
    for _ in range(rotation):
        path = [(c, -r) for r, c in path]
    if path[0][0] == path[-1][0]:
        return None
    return path if path[0][0] < path[-1][0] else path[::-1]


def _fallback_path(rng, axis, need_above=False, winding=False):
    # A variable hairpin is a bounded fallback only. Main sampling is an
    # induced random walk and is not restricted to these three segments.
    a, b, c = int(rng.integers(5, 8)), int(rng.integers(5, 8)), int(rng.integers(5, 8))
    if a == c:
        c = 5 if a != 5 else 6
    path = [(0, 0)]
    segments = [((0, 1), a), ((1, 0), b), ((0, -1), c)]
    if winding:
        lengths = np.array([4, 3, 2, 3, 4, 3, 3, 1])
        lengths += rng.multinomial(int(rng.integers(7)), np.full(8, 1/8))
        directions = ((0, 1), (1, 0), (0, -1), (1, 0),
                      (0, 1), (1, 0), (0, -1), (-1, 0))
        segments = list(zip(directions, map(int, lengths)))
    if need_above and axis == 1:
        segments.insert(0, ((-1, 0), 2))
    for t, count in segments:
        for _ in range(count):
            path.append(_add(path[-1], t))
    rotation = (0 if axis == 1 else 3) if need_above else int(rng.choice((0, 2))) if axis == 1 else int(rng.choice((1, 3)))
    return _transform_path(path, rotation, bool(rng.integers(2)))


def _bounds(points):
    return (min(p[0] for p in points), max(p[0] for p in points),
            min(p[1] for p in points), max(p[1] for p in points))


def _attach(rng, path, maximum, target, required_axis, force_long=False):
    spine = set(path)
    options = [(i, s) for i in range(1, len(path)-2) if _straight_station(path, i)
               for s in (-1, 1)]
    rng.shuffle(options)
    if required_axis is not None:
        # Place the required opposite tangents first, without fixing their
        # indices, their sides, or their payloads.
        required = []
        for sign in (-1, 1):
            found = next((o for o in options
                          if _step(path[o[0]], path[o[0]+1])[required_axis] == sign), None)
            if found is None:
                return None
            required.append(found)
        options = required + [o for o in options if o not in required]
    occupied, ribs = set(), []
    for i, side in options:
        lengths = list(range(2, maximum+1))
        rng.shuffle(lengths)
        if force_long and not ribs:
            lengths = [maximum]
        for radius in lengths:
            group = _rib_cells(path, i, side, radius)
            contacts = {q for p in group for q in _neighbors(p) if q in spine}
            if group & spine or contacts != {path[i], path[i+1]}:
                continue
            if group & occupied or any(q in occupied for p in group for q in _neighbors(p)):
                continue
            bounds = _bounds(spine | occupied | group)
            if bounds[1]-bounds[0]+3 > 30 or bounds[3]-bounds[2]+3 > 30:
                continue
            ribs.append({"index": i, "side": side, "radius": radius})
            occupied |= group
            break
        if len(ribs) >= target:
            break
    if required_axis is not None:
        directions = {_step(path[r["index"]], path[r["index"]+1])[required_axis] for r in ribs}
        if not {-1, 1} <= directions:
            return None
    return ribs


def _payloads(rng, ribs, palette, witness, preserve_long=False):
    for rib in ribs:
        shape = (2, rib["radius"])
        mode = int(rng.integers(5))
        if mode == 0:
            block = np.full(shape, rng.choice(palette), dtype=np.int8)
        elif mode == 1:
            block = np.repeat(rng.choice(palette, size=(2, 1)), shape[1], axis=1)
        elif mode == 2:
            block = np.repeat(rng.choice(palette, size=(1, shape[1])), 2, axis=0)
        else:
            block = rng.choice(palette, size=shape)
        rib["block"] = block.astype(np.int8)
    if witness:
        # Choose the repeated occurrences anywhere; they remain separate ribs.
        if preserve_long:
            maximum = max(r["radius"] for r in ribs)
            pivot = int(rng.choice([j for j, r in enumerate(ribs) if r["radius"] == maximum]))
            a, b = map(int, rng.choice([j for j in range(len(ribs)) if j != pivot], 2, replace=False))
        else:
            a, b, pivot = map(int, rng.choice(len(ribs), 3, replace=False))
        common = min(ribs[a]["radius"], ribs[b]["radius"])
        for j in (a, b):
            ribs[j]["radius"] = common
            ribs[j]["block"] = ribs[j]["block"][:, :common].copy()
        ribs[b]["block"] = ribs[a]["block"].copy()
        shape = (2, ribs[pivot]["radius"])
        for _ in range(20):
            block = rng.choice(palette, size=shape).astype(np.int8)
            if not np.array_equal(block, block[::-1]) and not np.array_equal(block, block[:, ::-1]):
                break
        else:
            block = np.full(shape, palette[0], dtype=np.int8)
            block[0, 0], block[1, -1] = palette[1], palette[2]
        ribs[pivot]["block"] = block


def _paint(rng, path, ribs, background, spine):
    points = set(path)
    for rib in ribs:
        points |= _rib_cells(path, rib["index"], rib["side"], rib["radius"])
    r0, r1, c0, c1 = _bounds(points)
    bh, bw = r1-r0+1, c1-c0+1
    extra_h = int(rng.integers(0, min(5, 28-bh)+1))
    extra_w = int(rng.integers(0, min(5, 28-bw)+1))
    top, left = 1 + int(rng.integers(extra_h+1)), 1 + int(rng.integers(extra_w+1))
    grid = np.full((bh+2+extra_h, bw+2+extra_w), background, dtype=np.int8)
    def put(p, color):
        grid[p[0]-r0+top, p[1]-c0+left] = color
    for p in path:
        put(p, spine)
    radius = max(r["radius"] for r in ribs)
    output = np.full((len(path), 2*radius+1), background, dtype=np.int8)
    output[:, radius] = spine
    for rib in ribs:
        i, side = rib["index"], rib["side"]
        t = _step(path[i], path[i+1])
        normal = (-t[1]*side, t[0]*side)
        for row in range(2):
            for k in range(1, rib["radius"]+1):
                color = rib["block"][row, k-1]
                put(_add(path[i+row], normal, k), color)
                output[i+row, radius+side*k] = color
    return {"input": grid, "output": output}


def _make_pair(rng, background, spine, axis=None, maximum=5, dense=False,
               winding=False, force_long=False, need_above=False,
               max_ribs=5, min_turns=0, max_turns=29):
    witness = axis is not None
    for attempt in range(260):
        if attempt < 180:
            length = int(rng.integers(22 if witness or dense else 6, 31))
            path = _walk(rng, length, float(rng.uniform(2.0, 5.0) if winding else rng.uniform(5.0, 16.0)))
            if path is None:
                continue
            path = _transform_path(path, int(rng.integers(4)), bool(rng.integers(2)))
        else:
            path = _fallback_path(rng, axis if witness else int(rng.integers(2)), need_above, min_turns >= 6)
        if path is None:
            continue
        if not min_turns <= _turns(path) <= max_turns:
            continue
        if need_above and min(r for r, c in path) >= path[0][0]:
            continue
        target = int(rng.integers(5 if dense else 3 if witness else 1, 7 if dense else max_ribs+1))
        ribs = _attach(rng, path, maximum, target, axis, force_long)
        if ribs is None or len(ribs) < (5 if dense else 3 if witness else 1):
            continue
        if force_long and max(r["radius"] for r in ribs) != maximum:
            continue
        pool = [c for c in range(10) if c not in (background, spine)]
        palette = rng.choice(pool, size=int(rng.integers(3 if witness else 1, 9)), replace=False)
        _payloads(rng, ribs, palette, witness, force_long)
        pair = _paint(rng, path, ribs, background, spine)
        try:
            d = _analyze(pair["input"])
        except ValueError:
            continue
        e = _evidence(d)
        if witness and not all(e[k] for k in ("topology", "sides", "rotation", "padding", "duplicate")):
            continue
        if not np.array_equal(_unfold(d), pair["output"]):
            raise AssertionError("Construction and input-only reconstruction disagree")
        return pair
    raise RuntimeError("Bounded ribbed-spine construction exhausted")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(3))  # greater reach, greater density, or more winding
    shared_background = int(rng.integers(10)) if rng.random() < .5 else None
    first_axis = int(rng.integers(2))
    need_above = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(30):
                bg = int(rng.integers(10)) if shared_background is None else shared_background
                spine = int(rng.choice([c for c in range(10) if c != bg]))
                axis = (first_axis if i == 0 else 1-first_axis) if split == "train" and i < 2 else None
                if split == "test" and i == 0:
                    # The first query is a real topological witness, rather
                    # than hoping a short decorated walk defeats every sweep.
                    axis = 1
                if mode == 0:
                    maximum = int(rng.integers(3, 5)) if split == "train" else int(rng.integers(5, 11))
                else:
                    maximum = int(rng.integers(3, 7))
                    if axis is None and rng.random() < .2:
                        maximum = int(rng.integers(8, 15))
                pair = _make_pair(rng, bg, spine, axis, maximum,
                                  dense=mode == 1 and split == "test",
                                  winding=mode == 2 and split == "test",
                                  force_long=mode == 0 and split == "test",
                                  need_above=need_above and split == "train" and i == 1,
                                  max_ribs=4 if mode == 1 and split == "train" else 5,
                                  min_turns=6 if mode == 2 and split == "test" else 0,
                                  max_turns=4 if mode == 2 and split == "train" else 29)
                d = _analyze(pair["input"])
                if split == "test" and i == 0 and not all(_evidence(d)[k] for k in ("topology", "sweep", "sides", "rotation")):
                    continue
                # Above-endpoint queries must have an explicit demonstration.
                if split == "test" and _evidence(d)["above"] and not any(_evidence(_analyze(p["input"]))["above"] for p in puzzle["train"]):
                    continue
                if _key(pair["input"]) not in seen:
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a distinct, taught example")
        if split == "train":
            rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, info = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(g) not in seen, "Duplicate input"
            seen.add(_key(g))
            d = _analyze(g)
            assert np.array_equal(out, _unfold(d)), "Incorrect unfolded ribs"
            radius = max(r["radius"] for r in d["ribs"])
            assert out.shape == (len(d["path"]), 2*radius+1)
            assert np.all(out[:, radius] == d["spine"])
            # Every colored input cell survives exactly once, with no new ink.
            assert Counter(map(int, g[g != d["background"]])) == Counter(map(int, out[out != d["background"]]))
            assert all(np.all(out[row, :radius] == d["background"]) and np.all(out[row, radius+1:] == d["background"])
                       for row in (0, len(out)-1)), "Endpoint rows must be unadorned"
            info[split].append(d)
    ev = [_evidence(d) for d in info["train"]]
    for concept in ("topology", "sweep", "sides", "rotation", "padding", "duplicate"):
        assert any(e[concept] for e in ev), "Missing training evidence: " + concept
    tangents = {r["tangent"] for d in info["train"] for r in d["ribs"]}
    assert tangents == set(DIRS), "Training must demonstrate all four local directions"
    assert any(all(_evidence(d)[k] for k in ("topology", "sweep", "sides", "rotation")) for d in info["test"])
    if any(_evidence(d)["above"] for d in info["test"]):
        assert any(e["above"] for e in ev), "Higher endpoint needs evidence"
    # This checks the actual arrays, without storing the sampled shift mode.
    def maximum_reach(d):
        return max(r["radius"] for r in d["ribs"])
    shifted_reach = min(map(maximum_reach, info["test"])) > max(map(maximum_reach, info["train"]))
    shifted_count = min(len(d["ribs"]) for d in info["test"]) > max(len(d["ribs"]) for d in info["train"])
    shifted_turns = min(_turns(d["path"]) for d in info["test"]) > max(_turns(d["path"]) for d in info["train"])
    assert shifted_reach or shifted_count or shifted_turns, "Missing visible train/test shift"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed semantic, variation, symmetry, and mutation checks."""
    signatures, shapes, layouts, sizes, reaches, counts, backgrounds = set(), set(), set(), set(), set(), set(), set()
    summary = Counter()
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        if seed == 0:
            sample = puzzle
        for split in ("train", "test"):
            for p in puzzle[split]:
                d = _analyze(p["input"])
                path = d["path"]
                shapes.add((p["input"].shape, p["output"].shape))
                layouts.add(tuple((r-path[0][0], c-path[0][1]) for r, c in path))
                sizes.add(len(path))
                reaches.add(max(r["radius"] for r in d["ribs"]))
                counts.add(len(d["ribs"]))
                backgrounds.add(d["background"])
                for k, v in _evidence(d).items():
                    summary[k] += int(v)
                summary["bilateral"] += any(sum(r["index"] == i for r in d["ribs"]) == 2 for i in range(len(path)))
                assert np.array_equal(solve(p["input"].tolist()), p["output"])
    assert len(signatures) == 42 and len(shapes) >= 100 and len(layouts) >= 100
    assert len(sizes) >= 15 and len(reaches) >= 5 and len(counts) >= 5
    assert len(backgrounds) == 10
    for name in ("topology", "sweep", "sides", "rotation", "padding", "duplicate", "late_max", "above", "bilateral"):
        assert summary[name] >= 8, (name, summary[name])
    assert _puzzle_key(sample) == _puzzle_key(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert (len(p["train"]), len(p["test"])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Translation padding and literal recoloring preserve the family rule.
    p = sample["test"][0]
    permutation = np.array([7, 4, 9, 1, 5, 0, 2, 8, 6, 3], dtype=np.int8)
    assert np.array_equal(solve(permutation[p["input"]]), permutation[p["output"]])
    # Horizontal reflection preserves higher-endpoint order and swaps sides.
    assert np.array_equal(solve(np.fliplr(p["input"])), np.fliplr(p["output"]))
    # All input rotations: output stays vertical, with order recomputed from
    # the upper endpoint. Reversing traversal reverses both output axes.
    markers = np.zeros_like(p["input"])
    path = _analyze(p["input"])["path"]
    markers[path[0]], markers[path[-1]] = 1, 2
    for k in range(4):
        rotated = np.rot90(p["input"], k)
        m = np.rot90(markers, k)
        start, end = np.argwhere(m == 1)[0], np.argwhere(m == 2)[0]
        if start[0] == end[0]:
            try:
                solve(rotated)
            except ValueError:
                continue
            raise AssertionError("Tied endpoints accepted")
        expected = p["output"] if start[0] < end[0] else p["output"][::-1, ::-1]
        assert np.array_equal(solve(rotated), expected)
    bad = copy.deepcopy(sample)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    # Explicit scope boundaries need not depend on a lucky sampled seed.
    path = [(r, 0) for r in range(6)]
    ribs = [{"index": 2, "side": 1, "radius": 14,
             "block": np.array([[2 + (k % 3) for k in range(14)],
                                [4 - (k % 3) for k in range(14)]], dtype=np.int8)},
            {"index": 2, "side": -1, "radius": 2,
             "block": np.full((2, 2), 3, dtype=np.int8)}]
    edge = _paint(np.random.default_rng(42), path, ribs, 0, 8)
    assert edge["output"].shape == (6, 29)
    assert np.array_equal(solve(edge["input"]), edge["output"])
    damaged = edge["input"].copy()
    d = _analyze(damaged)
    damaged[min(d["ribs"][0]["cells"])] = d["background"]
    horizontal = _paint(np.random.default_rng(43), [(0, c) for c in range(6)],
                        [{"index": 2, "side": 1, "radius": 2,
                          "block": np.full((2, 2), 3, dtype=np.int8)}], 0, 8)
    broken_spine = edge["input"].copy()
    broken_spine[d["path"][1]] = d["background"]
    for malformed in (damaged, horizontal["input"], broken_spine):
        try:
            solve(malformed)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed rib, tied endpoints, or broken spine accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shape_pairs": len(shapes),
            "spine_geometries": len(layouts), "path_lengths": sorted(sizes),
            "maximum_reaches": sorted(reaches), "rib_counts": sorted(counts),
            "evidence_counts": dict(summary)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
