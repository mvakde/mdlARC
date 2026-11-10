"""Rerouted color tape: topology selects a cyclic word for a replacement ray.

No files, training labels, RNG or generation metadata are consulted by solve.
Generation keeps only one puzzle at a time; self_test writes no artifacts.
"""
from __future__ import annotations

import hashlib
import numpy as np


REPAIR = 8
DIRECTION = 9
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 991253655, 809393317, 983003281, 529425091, 639084649,
              500260002, 612523966, 993957538, 191291642, 534141156,
              711385478, 677537913, 609789891, 849388171, 454008920,
              469838751, 712923122, 629679330, 389496211, 894969432,
              734117953, 633649497, 918379490, 512933712, 772549379,
              375444825, 623008900, 134302235, 905238567, 365474200,
              313251612, 619498956, 383246369, 670142998, 989549001,
              921013071, 185779481, 864005670, 93782101, 453908666,
              786087107)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _ray(p, d, shape):
    result = []
    q = _add(p, d)
    while 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
        result.append(q)
        q = _add(q, d)
    return result


def _line(a, b):
    _require(a[0] == b[0] or a[1] == b[1], "nonorthogonal segment")
    d = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [a] + _ray(a, d, (31, 31))[:max(abs(b[0]-a[0]), abs(b[1]-a[1]))]


def _polyline(vertices):
    path = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        path.extend(_line(a, b)[1:])
    return path


def _steps(path):
    return [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]


def _bends(path):
    ds = _steps(path)
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _coordinate_orders(cells):
    return [sorted(cells, key=lambda p, a=a, s=s, t=t: (s*p[a], t*p[1-a]))
            for a in (0, 1) for s in (-1, 1) for t in (-1, 1)]


def _analyze(input_grid):
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(bg < REPAIR and counts[bg] * 2 > inp.size, "background must have strict majority")
    markers = [tuple(map(int, p)) for p in np.argwhere(inp == DIRECTION)]
    _require(len(markers) == 1, "exactly one directional marker is required")
    q = markers[0]
    cells = {tuple(map(int, p)) for p in np.argwhere((inp != bg) & (inp != DIRECTION))}
    _require(len(cells) >= 3, "missing tape")
    adjacency = {p: _neighbors(p, cells) for p in cells}
    _require(all(len(ns) in (1, 2) for ns in adjacency.values()), "tape must be a simple chain")
    ends = [p for p in cells if len(adjacency[p]) == 1]
    _require(len(ends) == 2, "tape must have two endpoints")
    pivots = _neighbors(q, cells)
    _require(len(pivots) == 1, "marker must touch exactly one pivot")
    pivot = pivots[0]
    d = q[0] - pivot[0], q[1] - pivot[1]
    old_first = _add(pivot, (-d[0], -d[1]))
    _require(old_first in cells and len(adjacency[pivot]) == 2, "pivot must have an opposite tail")
    kept_first = next(p for p in adjacency[pivot] if p != old_first)
    _require((kept_first[0]-pivot[0])*d[0] + (kept_first[1]-pivot[1])*d[1] == 0,
             "pivot must be a corner")

    def branch(first):
        previous, current, walk = pivot, first, []
        for _ in range(len(cells)):
            _require(current not in walk, "cycle in tape")
            walk.append(current)
            ns = [p for p in adjacency[current] if p != previous]
            if not ns:
                return walk
            _require(len(ns) == 1, "branching tape")
            previous, current = current, ns[0]
        raise ValueError("nonterminating tape")

    old, kept = branch(old_first), branch(kept_first)
    _require(len(old) + len(kept) + 1 == len(cells) and not set(old) & set(kept), "disconnected tape")
    end = old[-1]
    _require(end[0] in (0, inp.shape[0]-1) or end[1] in (0, inp.shape[1]-1), "old tail must reach an edge")
    repaired = inp.copy()
    repair_sites = [p for p in cells if inp[p] == REPAIR]
    for p in repair_sites:
        neighbors = adjacency[p]
        colors = {int(inp[n]) for n in neighbors}
        _require(REPAIR not in colors and len(colors) == 1, "repair requires agreeing visible path neighbors")
        repaired[p] = next(iter(colors))
    ray = _ray(pivot, d, inp.shape)
    seq = [int(repaired[p]) for p in old]
    return {"input": inp, "background": bg, "cells": cells, "pivot": pivot,
            "direction": d, "old": old, "kept": kept, "ray": ray,
            "sequence": seq, "repaired": repaired, "repair_sites": repair_sites}


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze(input_grid)
    out = info["repaired"].copy()
    out[out == DIRECTION] = info["background"]
    for p in info["old"]:
        out[p] = info["background"]
    word = info["sequence"]
    for i, p in enumerate(info["ray"]):
        out[p] = word[i % len(word)]
    return out


def _axis(rng, extent):
    n = int(rng.integers(4, min(7, (extent-1)//3 + 1) + 1))
    gaps = 3 + rng.multinomial(extent - 1 - 3*(n-1), np.full(n-1, 1/(n-1)))
    return [0] + list(map(int, np.cumsum(gaps)))


def _coarse_walk(rng, rows, cols, rich):
    nr, nc = len(rows), len(cols)
    side = int(rng.integers(4))
    if side % 2 == 0:
        start = (0 if side == 0 else nr-1, int(rng.integers(1, nc-1)))
    else:
        start = (int(rng.integers(1, nr-1)), nc-1 if side == 1 else 0)
    inward = DIRS[(side+2) % 4]
    path = [start, _add(start, inward)]
    used = set(path)
    target = int(rng.integers(8 if rich else 5, min(nr*nc, 23 if rich else 17) + 1))
    for _ in range(target - 2):
        options = [p for d in DIRS if (p := _add(path[-1], d)) not in used
                   and 0 < p[0] < nr-1 and 0 < p[1] < nc-1]
        if not options:
            break
        p = options[int(rng.integers(len(options)))]
        path.append(p)
        used.add(p)
    return [(rows[r], cols[c]) for r, c in path]


def _geometry_candidates(path, shape, kind, bend_shift):
    cells = set(path)
    if len(cells) != len(path) or any(len(_neighbors(p, cells)) != (1 if i in (0, len(path)-1) else 2)
                                      for i, p in enumerate(path)):
        return []
    options = []
    # Six cells permit a visible equal-color repair neighborhood, a return of
    # that color, and a nonpalindromic word at the same time.
    for at in range(6, len(path)-6):
        p, prev, nxt = path[at], path[at-1], path[at+1]
        old_d = prev[0]-p[0], prev[1]-p[1]
        next_d = nxt[0]-p[0], nxt[1]-p[1]
        if old_d[0]*next_d[0] + old_d[1]*next_d[1] != 0:
            continue
        d = -old_d[0], -old_d[1]
        q = _add(p, d)
        ray = _ray(p, d, shape)
        if len(ray) < 3 or q in cells or _neighbors(q, cells) != [p]:
            continue
        old = list(reversed(path[:at]))
        kept = path[at+1:]
        intersections = set(ray) & set(kept)
        # Clean transverse crossings only, with intact retained evidence both sides.
        if any(len(ns := _neighbors(x, cells)) != 2 or
               any((n[0]-x[0])*d[0] + (n[1]-x[1])*d[1] != 0 for n in ns)
               for x in intersections):
            continue
        bends = _bends([p] + old)
        if kind == "repeat" and not (len(ray) >= len(old)+2 and not intersections and bends == 0):
            continue
        if kind == "cross_crop" and not (len(old) >= len(ray)+3 and intersections and 2 <= bends <= (2 if bend_shift else 5)):
            continue
        if kind == "query" and bend_shift and bends < 3:
            continue
        if kind == "free" and bend_shift and bends > 2:
            continue
        if kind != "repeat" and any(old[:len(ray)] == order[:len(ray)] for order in _coordinate_orders(old)):
            continue
        options.append((at, d))
    return options


def _fallback_geometry(kind):
    if kind == "repeat":
        shape = (24, 25)
        vertices = [(23, 8), (17, 8), (17, 14), (9, 14), (9, 21), (3, 21)]
        pivot = (17, 8)
    elif kind in ("cross_crop", "free"):
        shape = (30, 25)
        vertices = [(0, 19), (24, 19), (24, 9), (18, 9), (18, 15), (7, 15), (7, 4), (0, 4)]
        pivot = (18, 9)
    else:
        shape = (30, 25)
        vertices = [(4, 24), (4, 20), (27, 20), (27, 9), (22, 9), (22, 4), (15, 4), (15, 15), (0, 15)]
        pivot = (22, 9)
    path = _polyline(vertices)
    return path, path.index(pivot), (-1, 0), shape


def _cross_geometry(rng):
    """A variable U-tail witness; its retained continuation grows procedurally.

    The U reverses its vertical direction, so both coordinate sweep axes fail.
    The ray is long enough to expose that difference before boundary truncation.
    """
    h, w = int(rng.integers(23, 31)), int(rng.integers(21, 31))
    r = int(rng.integers(14, h-5))
    c = int(rng.integers(5, w-12))
    drop = int(rng.integers(3, min(6, h-r-1)))
    gap = int(rng.integers(6, min(w-c-2, r-drop-2)+1))
    far = c + gap
    inner = int(rng.integers(c+3, far-2))
    left = int(rng.integers(1, c-2))
    cross = int(rng.integers(2, r-3))
    pivot = (r, c)
    vertices = [(0, far), (r+drop, far), (r+drop, c), pivot]
    if rng.random() < .5:
        vertices += [(r, inner), (cross, inner), (cross, left)]
    else:
        vertices += [(r, left), (cross, left), (cross, inner)]
    path = _polyline(vertices)
    occupied = set(path)
    q = (r-1, c)
    previous = _steps(path)[-1]
    for _ in range(int(rng.integers(0, 32))):
        candidates, weights = [], []
        for d in DIRS:
            x = _add(path[-1], d)
            if not (0 <= x[0] < h and 0 <= x[1] < w) or x in occupied:
                continue
            if abs(x[0]-q[0]) + abs(x[1]-q[1]) <= 1:
                continue
            if _neighbors(x, occupied) != [path[-1]]:
                continue
            candidates.append((x, d))
            weights.append(5 if d == previous else 1)
        if not candidates:
            break
        probability = np.asarray(weights, dtype=float)
        x, previous = candidates[int(rng.choice(len(candidates), p=probability/probability.sum()))]
        occupied.add(x)
        path.append(x)
    return path, path.index(pivot), (-1, 0), (h, w)


def _repeat_query_geometry(rng):
    """A three-bend tail reaching a nearby side edge feeds a longer ray.

    Unlike the straight-tail teaching case, this word reverses vertical
    direction before its endpoint. The moving tape also crosses retained tape.
    """
    h, w = int(rng.integers(27, 31)), int(rng.integers(21, 31))
    r, c = h - int(rng.integers(5, 8)), w - int(rng.integers(7, 10))
    drop, gap, rise = int(rng.integers(2, 4)), int(rng.integers(3, 5)), int(rng.integers(3, 6))
    pivot = (r, c)
    cross = int(rng.integers(3, r-6))
    left = int(rng.integers(1, c-3))
    right = c + int(rng.integers(2, gap+1))
    end = int(rng.integers(0, cross-1))
    vertices = [(r+drop-rise, w-1), (r+drop-rise, c+gap),
                (r+drop, c+gap), (r+drop, c), pivot,
                (r, left), (cross, left), (cross, right), (end, right)]
    path = _polyline(vertices)
    return path, path.index(pivot), (-1, 0), (h, w)


def _geometry(rng, kind, bend_shift, large=False):
    if kind == "cross_crop":
        return _cross_geometry(rng)
    if kind == "query_repeat":
        return _repeat_query_geometry(rng)
    for _ in range(320):
        low = 21 if large else 12
        shape = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        rows, cols = _axis(rng, shape[0]), _axis(rng, shape[1])
        path = _polyline(_coarse_walk(rng, rows, cols, kind == "query"))
        candidates = _geometry_candidates(path, shape, kind, bend_shift)
        if candidates:
            at, d = candidates[int(rng.integers(len(candidates)))]
            return path, at, d, shape
    return _fallback_geometry(kind)


def _sequence(rng, n, palette):
    colors = [int(rng.choice(palette))]
    for _ in range(n-1):
        colors.append(colors[-1] if rng.random() < .58 else int(rng.choice(palette)))
    return colors


def _replay(seq, n):
    return [seq[i % len(seq)] for i in range(n)]


def _sweep_words(info):
    # Alternatives that read the tail by coordinates instead of its connectivity.
    old, grid = info["old"], info["repaired"]
    return [[int(grid[p]) for p in order] for order in _coordinate_orders(old)]


def _features(info):
    seq, ray = info["sequence"], info["ray"]
    replay = _replay(seq, len(ray))
    crossings = set(ray) & set(info["kept"])
    skipped, cursor = [], 0
    for p in ray:
        skipped.append(seq[cursor % len(seq)])
        if p not in crossings:
            cursor += 1
    padding = seq[:len(ray)] + [seq[-1]] * max(0, len(ray)-len(seq))
    return {"shape": tuple(info["input"].shape), "tail_length": len(seq),
            "ray_length": len(ray), "old_bends": _bends([info["pivot"]] + info["old"]),
            "kept_bends": _bends([info["pivot"]] + info["kept"]),
            "direction": info["direction"], "crossings": len(crossings),
            "new_old_overlaps": len(set(ray) & set(info["old"])),
            "repeat": len(ray) > len(seq), "crop": len(ray) < len(seq),
            "repair_count": len(info["repair_sites"]),
            "tail_repairs": len(set(info["repair_sites"]) & set(info["old"])),
            "replayed_repairs": sum(p in info["repair_sites"] for p in info["old"][:len(ray)]),
            "pivot_repair": info["pivot"] in info["repair_sites"],
            "recurrence": len([c for i,c in enumerate(seq) if i == 0 or c != seq[i-1]]) > len(set(seq)),
            "reverse_differs": replay != _replay(seq[::-1], len(ray)),
            "phase_differs": replay != _replay([int(info["repaired"][info["pivot"]])] + seq, len(ray)),
            "padding_differs": replay != padding,
            "sweep_differs": all(replay != _replay(word, len(ray)) for word in _sweep_words(info)),
            "pivot_flood_differs": any(c != int(info["repaired"][info["pivot"]]) for c in replay),
            "crossing_advances": replay != skipped,
            "crossing_overwrite": any(int(info["repaired"][p]) != replay[i] for i,p in enumerate(ray) if p in crossings)}


def _make_pair(rng, params, kind, desired_direction, bend_shift, large=False):
    path, at, d, shape = _geometry(rng, kind, bend_shift, large)
    pivot = path[at]
    old = list(reversed(path[:at]))
    ray = _ray(pivot, d, shape)
    bg = int(rng.choice(params["backgrounds"]))
    palette = [c for c in range(8) if c != bg]
    rng.shuffle(palette)
    palette = palette[:int(rng.integers(2, 6))]
    # Reject shortcuts on actual colored answers, not only on geometric features.
    for _ in range(100):
        colors = _sequence(rng, len(path), palette)
        seq = [colors[at-1-i] for i in range(at)]
        if len(set(_replay(seq, len(ray)))) < 2 or _replay(seq, len(ray)) == _replay(seq[::-1], len(ray)):
            continue
        repaired = np.full(shape, bg, dtype=np.int8)
        for p, c in zip(path, colors):
            repaired[p] = c
        # Place one repair in the transported word, forcing genuine reconstruction.
        candidates = [p for i,p in enumerate(old) if i > 0 and i < min(len(old)-1, len(ray))
                      and repaired[old[i-1]] == repaired[p] == repaired[old[i+1]]]
        if not candidates:
            continue
        inp = repaired.copy()
        inp[_add(pivot, d)] = DIRECTION
        inp[candidates[int(rng.integers(len(candidates)))]] = REPAIR
        # Additional repair sites include bends, pivots, retained cells and endpoints.
        sites = list(path)
        rng.shuffle(sites)
        for p in sites:
            if np.count_nonzero(inp == REPAIR) >= int(params["repair_budget"]):
                break
            ns = _neighbors(p, set(path))
            if inp[p] == REPAIR or any(inp[x] == REPAIR for x in ns):
                continue
            if all(repaired[x] == repaired[p] for x in ns) and rng.random() < .45:
                inp[p] = REPAIR
        info = _analyze(inp)
        f = _features(info)
        if not f["pivot_flood_differs"] or not f["recurrence"] or not f["phase_differs"]:
            continue
        if kind in ("repeat", "query_repeat") and not f["padding_differs"]:
            continue
        if kind != "repeat" and not f["sweep_differs"]:
            continue
        if kind == "cross_crop" and not (f["crossing_overwrite"] and f["crossing_advances"]):
            continue
        # Ground truth is rendered from the construction path, separately from solve.
        out = np.full(shape, bg, dtype=np.int8)
        for p, c in zip(path[at:], colors[at:]):
            out[p] = c
        for i, p in enumerate(ray):
            out[p] = seq[i % len(seq)]
        # Rotate input AND label so every marker direction has the same semantics.
        k = (DIRS.index(d) - DIRS.index(desired_direction)) % 4
        inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
        if rng.random() < .5:
            # Reflect along the ray axis: direction is unchanged.
            axis = 1 if desired_direction[0] else 0
            inp, out = np.flip(inp, axis).copy(), np.flip(out, axis).copy()
        return {"input": inp, "output": out}
    raise RuntimeError("bounded color construction exhausted")


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and low <= value <= high, name + " must be a legal integer count")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _input_key(a):
    return a.shape, a.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    backgrounds = list(map(int, rng.choice(8, int(rng.integers(1, 4)), replace=False)))
    params = {"backgrounds": backgrounds, "repair_budget": int(rng.integers(2, 7))}
    mode = int(rng.integers(3))  # bend extrapolation, held-out orientation, or scale shift
    base = int(rng.integers(4))
    train_dirs = [DIRS[base], DIRS[(base+1) % 4]]
    test_dirs = [DIRS[(base+2) % 4], DIRS[(base+3) % 4]]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", ntrain), ("test", ntest)):
        for i in range(count):
            if split == "test":
                kind = "query_repeat" if rng.random() < .35 else "query"
            elif i == 0:
                kind = "repeat"
            elif i == 1:
                kind = "cross_crop"
            else:
                kind = str(rng.choice(("repeat", "cross_crop", "free")))
            directions = train_dirs if split == "train" or mode == 0 else test_dirs
            desired = directions[i % 2] if i < 2 else directions[int(rng.integers(2))]
            for _ in range(12):
                try:
                    pair = _make_pair(rng, params, kind, desired, mode == 0,
                                      large=(split == "test" and mode == 2))
                except RuntimeError:
                    # A tiny destination can make the joint color witnesses
                    # incompatible. Resample the geometry within the same rule.
                    continue
                key = _input_key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded pair construction exhausted: uniqueness or color witnesses")
            seen.add(key)
            puzzle[split].append(pair)
    # Witness order is not a solving convention.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "dimensions must be preserved")
            key = _input_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(inp)
            _require(np.array_equal(solve(inp), out), "wrong output")
            _require(not np.any(np.isin(out, [REPAIR, DIRECTION])), "annotation survived")
            # Exact local preservation/deletion plus per-position periodicity.
            ray = set(info["ray"])
            old = set(info["old"])
            for p in info["cells"] - old - ray:
                _require(out[p] == info["repaired"][p], "retained tape changed")
            for p in old - ray:
                _require(out[p] == info["background"], "old tail was retained")
            f = _features(info)
            _require(f["replayed_repairs"] >= 1 and f["pivot_flood_differs"] and f["reverse_differs"] and f["phase_differs"]
                     and f["recurrence"], "insufficient mechanism evidence")
            features[split].append(f)
    train = features["train"]
    _require(any(f["repeat"] and f["padding_differs"] for f in train) and any(f["crop"] for f in train), "teach cyclic replay and truncation")
    _require(any(f["crossing_overwrite"] and f["crossing_advances"] for f in train), "teach ray-over-retained crossing")
    _require(any(f["sweep_differs"] for f in train), "teach connectivity rather than coordinate sweep")
    _require(len({f["direction"] for f in train}) >= 2, "teach two orientations")
    return None


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 supplied seeds, meaningful negatives, all count boundaries; no disk I/O."""
    # Exact short-ray/equal-length fixtures exercise semantics beyond sampling
    # quotas. The expected outputs are explicit placements, not solve-derived.
    for width in (6, 10):
        inp = np.zeros((6, width), dtype=np.int8)
        for p,c in (((3,4),4), ((3,5),9), ((3,3),1), ((3,2),2),
                    ((2,2),8), ((1,2),2), ((0,2),3), ((4,4),5), ((5,4),5)):
            inp[p] = c
        expected = np.zeros_like(inp)
        expected[3,4], expected[4,4], expected[5,4] = 4, 5, 5
        expected[3,5:] = [1] if width == 6 else [1,2,2,2,3]
        unchanged = inp.copy()
        assert np.array_equal(solve(inp), expected)
        assert np.array_equal(inp, unchanged)
        # A marked pivot can be reconstructed from agreeing path neighbors.
        inp[3,4], inp[4,4] = REPAIR, 1
        expected[3,4], expected[4,4] = 1, 1
        assert np.array_equal(solve(inp), expected)
    # Malformed topology and ambiguous repairs must fail before any rendering.
    for rows in (("003000", "002000", "008000", "002849", "000050", "000050"),
                 ("000000", "010000", "018900", "010000", "000000", "000000")):
        inp = np.array([list(map(int,row)) for row in rows], dtype=np.int8)
        try:
            solve(inp)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous or branching input accepted")
    signatures, shapes, geometries, tails, bends, crossings, directions, backgrounds = (set() for _ in range(8))
    repeat = crop = pivot_repairs = marked_bends = query_repeat = query_crop = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                f = _features(info)
                shapes.add(f["shape"])
                p = info["pivot"]
                geometries.add(tuple((r-p[0], c-p[1]) for r,c in info["old"]+info["kept"]))
                tails.add(f["tail_length"])
                bends.add(f["old_bends"])
                crossings.add(f["crossings"])
                directions.add(f["direction"])
                backgrounds.add(info["background"])
                repeat += f["repeat"]
                crop += f["crop"]
                query_repeat += split == "test" and f["repeat"]
                query_crop += split == "test" and f["crop"]
                pivot_repairs += f["pivot_repair"]
                marked_bends += sum(_bends([ns[0], x, ns[1]]) == 1 for x in info["repair_sites"]
                                    if len(ns := _neighbors(x, info["cells"])) == 2)
                # No-B alternative: repair but keep the original route and erase the marker.
                no_b = info["repaired"].copy()
                no_b[no_b == DIRECTION] = info["background"]
                assert not np.array_equal(no_b, pair["output"])
                # No-A alternative: B's monochromatic pivot-color ray.
                no_a = pair["output"].copy()
                for x in info["ray"]:
                    no_a[x] = info["repaired"][p]
                assert not np.array_equal(no_a, pair["output"])
                for k in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
        # Full-output mutation must fail, including preserved and newly written pixels.
        bad = {s: [dict(q) for q in puzzle[s]] for s in puzzle}
        bad["test"][0]["output"] = bad["test"][0]["output"].copy()
        a = bad["test"][0]["output"]
        a[0, 0] = (int(a[0, 0])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt output accepted")
    assert len(signatures) == 42 and len(shapes) >= 60 and len(geometries) >= 120
    assert len(tails) >= 20 and len(bends) >= 5 and max(crossings) >= 2
    assert directions == set(DIRS) and backgrounds == set(range(8))
    assert repeat >= 42 and crop >= 42 and pivot_repairs > 0 and marked_bends > 0
    assert query_repeat >= 8 and query_crop >= 8
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(176+nt*4+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    assert _puzzle_digest(generate_puzzle(17)) == _puzzle_digest(generate_puzzle(17))
    for nt, nq in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (3,1.5), (None,1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
