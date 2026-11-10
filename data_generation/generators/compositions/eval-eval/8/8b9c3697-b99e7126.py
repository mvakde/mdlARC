"""Ghost cargo: complete self-similar tile constellations before docking them.

Only NumPy and the standard library are used. No files, traces, or random state
are used by solve(). Generation keeps at most one small candidate in memory.
"""
from itertools import combinations
import numpy as np


_DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _rect(r, c, h, w):
    return {(y, x) for y in range(r, r + h) for x in range(c, c + w)}


def _bounds(cells):
    ys, xs = zip(*cells)
    return min(ys), min(xs), max(ys) + 1, max(xs) + 1


def _paint(grid, cells, color):
    for p in cells:
        grid[p] = color


def _fork(cells):
    """Parse a one-cell base and three centered, positive rectangular prongs."""
    parses = []
    for outward in _DIRS:
        perp = -outward[1], outward[0]
        kmin = min(r * outward[0] + c * outward[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * outward[0] + c * outward[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for k, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 30)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, n in enumerate(lengths) for k in range(1, n + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if not lengths[q]:
                q += 1
                continue
            a, n = q, lengths[q]
            while q < width and lengths[q] == n:
                q += 1
            runs.append((a, q, n))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        a, b, center = runs[1]
        if a + b != width:
            continue
        parses.append({"out": outward, "perp": perp, "kmin": kmin, "qmin": qmin,
                       "span": b - a, "q0": a, "stop": center + 1,
                       "lengths": tuple(t[2] for t in runs), "cells": cells})
    _require(len(parses) == 1, "fork must have one unambiguous positive-prong interpretation")
    return parses[0]


def _registrations(mask, positions):
    """All translations on an unbounded coarse plane, without boundary hints."""
    active = set(map(tuple, np.argwhere(mask)))
    p = min(positions)
    return sorted((p[0] - y, p[1] - x) for y, x in active
                  if all((r - p[0] + y, c - p[1] + x) in active for r, c in positions))


def _cargo(tiles, shape):
    h, w = shape
    tile = tiles[0][2]
    _require(all(np.array_equal(t, tile) for r, c, t in tiles), "one ink must identify identical tiles")
    # Spacing is fixed by the family; a one-row fragment need not encode a
    # second, unobserved pitch. Both pitches follow from the tile dimensions.
    rbase, cbase = min(r for r, c, t in tiles), min(c for r, c, t in tiles)
    _require(all((r - rbase) % (h + 1) == 0 and (c - cbase) % (w + 1) == 0
                 for r, c, t in tiles), "tiles must share one one-cell-gap lattice")
    positions = {((r - rbase) // (h + 1), (c - cbase) // (w + 1)) for r, c, t in tiles}
    return tile, (rbase, cbase), positions


def _infer(grid):
    a = _grid(grid)
    bg = int(a[0, 0])
    _require(bg != 0 and np.all(a[0] == bg) and np.all(a[-1] == bg)
             and np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg), "uniform nonblack background border required")
    _require(0 not in a, "black is reserved for the output trail")
    objects = _components(map(tuple, np.argwhere(a != bg)))
    forks, raw_tiles, structural = [], [], set()
    for cells in objects:
        r, c, b, d = _bounds(cells)
        colors = {int(a[p]) for p in cells}
        if cells == _rect(r, c, b - r, d - c):
            _require(len(colors) == 2 and 2 <= b - r <= 3 and 2 <= d - c <= 3,
                     "cargo tiles must be filled two-color 2..3 by 2..3 rectangles")
            raw_tiles.append((r, c, a[r:b, c:d].copy(), cells))
        else:
            _require(len(colors) == 1, "a fork must be monochrome")
            f = _fork(cells)
            forks.append(f)
            structural.update(colors)
    _require(forks and len(structural) == 1, "all forks share one structural color")
    structure = structural.pop()
    groups, physical = {}, set()
    for r, c, tile, cells in raw_tiles:
        colors = set(map(int, np.unique(tile)))
        _require(structure in colors, "tile face must have the fork color")
        ink = (colors - {structure}).pop()
        groups.setdefault(ink, []).append((r, c, tile))
        physical |= cells
    cargos = []
    for ink, tiles in sorted(groups.items()):
        tile, base, positions = _cargo(tiles, tiles[0][2].shape)
        mask = tile == ink
        h, w = mask.shape
        _require(len(positions) >= 2 and 3 <= int(mask.sum()) < h * w, "a strict multi-tile clue is required")
        active = set(map(tuple, np.argwhere(mask)))
        _require({r for r, c in active} >= {0, h - 1} and {c for r, c in active} >= {0, w - 1},
                 "glyph must span its tile bounding box")
        origins = _registrations(mask, positions)
        _require(len(origins) == 1, "ambiguous glyph translation")
        origin = origins[0]
        complete = {(origin[0] + r, origin[1] + c) for r, c in active}
        _require(positions < complete, "at least one tile must be missing")
        origins_px = {(base[0] + r * (h + 1), base[1] + c * (w + 1)) for r, c in complete}
        pixels = {(r + y, c + x): int(tile[y, x]) for r, c in origins_px
                  for y in range(h) for x in range(w)}
        _require(all(0 < r < a.shape[0] - 1 and 0 < c < a.shape[1] - 1 for r, c in pixels),
                 "completed cargo must fit inside the background border")
        observed = set().union(*(_rect(r, c, h, w) for r, c, t in tiles))
        _require(all(p in observed or int(a[p]) == bg for p in pixels), "a missing tile collides with an object")
        cargos.append({"ink": ink, "tile": tile, "mask": mask, "pixels": pixels,
                       "observed": observed, "missing": set(pixels) - observed,
                       "visible_count": len(positions), "active_count": len(active),
                       "fragment": tuple(sorted((r - origin[0], c - origin[1]) for r, c in positions))})
    _require(cargos, "no cargo")
    for i, cargo in enumerate(cargos):
        _require(all(set(cargo["pixels"]).isdisjoint(other["pixels"]) for other in cargos[:i]),
                 "completed cargos overlap")
    return a, {"background": bg, "structure": structure, "forks": forks,
               "cargos": cargos, "physical": physical}


def _match(fork, cargo, a, bg, check_span=True):
    pixels = cargo["pixels"]
    out, perp = fork["out"], fork["perp"]
    ks = [r * out[0] + c * out[1] - fork["kmin"] for r, c in pixels]
    qs = [r * perp[0] + c * perp[1] - fork["qmin"] for r, c in pixels]
    if check_span and (min(qs), max(qs) + 1) != (fork["q0"], fork["q0"] + fork["span"]):
        return None
    if not check_span and min(qs) + max(qs) + 1 != 2 * fork["q0"] + fork["span"]:
        return None
    distance = min(ks) - fork["stop"]
    if distance < 1:
        return None
    dr, dc = -out[0], -out[1]
    swept = {(r + t * dr, c + t * dc) for r, c in pixels for t in range(distance + 1)}
    if any(not (0 < r < a.shape[0] - 1 and 0 < c < a.shape[1] - 1)
           or (p not in cargo["observed"] and int(a[p]) != bg)
           for p in swept for r, c in [p]):
        return None
    final = {(r + distance * dr, c + distance * dc): color for (r, c), color in pixels.items()}
    return {"distance": distance, "delta": (distance * dr, distance * dc),
            "swept": swept, "final": final, "trail": swept - set(final)}


def _analyze(grid):
    a, info = _infer(grid)
    matches, used, swept = [], set(), set()
    for fi, f in enumerate(info["forks"]):
        candidates = [(ci, m) for ci, cargo in enumerate(info["cargos"])
                      if (m := _match(f, cargo, a, info["background"])) is not None]
        _require(len(candidates) <= 1, "a fork has multiple compatible cargos")
        if candidates:
            ci, m = candidates[0]
            _require(ci not in used and swept.isdisjoint(m["swept"]), "ambiguous or intersecting motions")
            used.add(ci)
            swept |= m["swept"]
            matches.append(dict(m, fi=fi, ci=ci))
    _require(matches, "at least one completed cargo must dock")
    info["matches"], info["used"] = matches, used
    return a, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, info = _analyze(input_grid)
    result = a.copy()
    _paint(result, info["physical"], info["background"])
    for m in info["matches"]:
        _paint(result, m["trail"], 0)
        for p, color in m["final"].items():
            result[p] = color
    return result


def _symmetries(mask):
    variants = [np.rot90(mask, k) for k in range(4)]
    variants += [np.fliplr(x) for x in variants]
    return {i for i, v in enumerate(variants) if v.shape == mask.shape and np.array_equal(v, mask)}


def _design(rng, shape, nose=False, asymmetric=False):
    h, w = shape
    if nose:
        mask = np.array([[1, 0], [1, 1]], dtype=bool)
        if rng.integers(2):
            mask = mask[:, ::-1].copy()
        return mask, {(1, 0), (1, 1)}
    for attempt in range(80):
        mask = np.zeros(shape, dtype=bool)
        n = int(rng.integers(3, min(h * w - 1, 7) + 1))
        mask.flat[rng.choice(h * w, n, replace=False)] = True
        active = sorted(map(tuple, np.argwhere(mask)))
        if not {r for r, c in active} >= {0, h - 1} or not {c for r, c in active} >= {0, w - 1}:
            continue
        if asymmetric and _symmetries(mask) != {0}:
            continue
        fragments = []
        # At most 119 strict fragments for the seven-active-cell cap. Choose a
        # count first so densely supplied patterns do not dominate by binomial mass.
        for size in range(2, n):
            options = [set(s) for s in combinations(active, size)
                       if _registrations(mask, set(s)) == [(0, 0)]]
            if options:
                fragments.append(options)
        if fragments:
            options = fragments[int(rng.integers(len(fragments)))]
            return mask, options[int(rng.integers(len(options)))]
    # Full-bounding-box unequal-leg L has a unique opposite-endpoint fragment.
    mask = np.zeros(shape, dtype=bool)
    mask[0, 0] = mask[0, w - 1] = mask[h - 1, 0] = True
    return mask, {(0, w - 1), (h - 1, 0)}


def _module(rng, shape, colors, rotation, *, nose=False, asymmetric=False, decoy=True, counterfork=False):
    bg, structure, ink, fake_ink = colors
    mask, visible = _design(rng, shape, nose=nose, asymmetric=asymmetric)
    h, w = shape
    mh, mw = h * h + h - 1, w * w + w - 1
    center = 1 if nose else int(rng.integers(1, 4))
    guards = (int(rng.integers(3, 6)),) * 2 if nose else tuple(map(int, rng.integers(1, 7, 2)))
    distance = int(rng.integers(1, 4 if counterfork else 6))
    start, span, q0 = center + 1 + distance, mw, 2
    width = span + 4
    structure_cells = _rect(0, 0, 1, width)
    structure_cells |= _rect(1, 0, guards[0], 1) | _rect(1, width - 1, guards[1], 1)
    structure_cells |= _rect(1, q0, center, span)
    tile = np.where(mask, ink, structure).astype(np.int8)
    original, complete, final, swept = {}, {}, {}, set()
    for y, x in map(tuple, np.argwhere(mask)):
        for r in range(h):
            for c in range(w):
                p = start + y * (h + 1) + r, q0 + x * (w + 1) + c
                complete[p] = int(tile[r, c])
                if (y, x) in visible:
                    original[p] = int(tile[r, c])
                final[p[0] - distance, p[1]] = int(tile[r, c])
                swept |= {(p[0] - t, p[1]) for t in range(distance + 1)}
    if decoy:
        fake_shape = (2, 3 if w == 2 else 2)
        fm, fv = _design(rng, fake_shape)
        fh, fw = fake_shape
        fake_width = fw * fw + fw - 1
        fq = q0 + (span - fake_width) // 2
        fake_start = start + mh + int(rng.integers(2, 4))
        ft = np.where(fm, fake_ink, structure)
        for y, x in fv:
            for r in range(fh):
                for c in range(fw):
                    original[fake_start + y * (fh + 1) + r, fq + x * (fw + 1) + c] = int(ft[r, c])
        # Reserve invisible decoy tiles too; otherwise a cropped clue could
        # imply a constellation crossing the canvas border.
        reserve = _rect(fake_start, fq, fh * fh + fh - 1, fake_width)
        if counterfork:
            # An opposed, long-center fork has a clear route to the decoy but
            # the wrong cross-section. This isolates width matching from
            # alignment, obstruction, and mere nearest-object heuristics.
            base = fake_start + fh * fh + fh - 1 + 6
            structure_cells |= _rect(base, 0, 1, width)
            structure_cells |= _rect(base - 1, 0, 1, 1) | _rect(base - 1, width - 1, 1, 1)
            structure_cells |= _rect(base - 3, q0, 3, span)
    else:
        reserve = set()
    all_cells = structure_cells | set(complete) | set(original) | reserve
    r0, c0, r1, c1 = _bounds(all_cells)
    a = np.full((r1 - r0, c1 - c0), bg, dtype=np.int8)
    b = a.copy()
    for p in structure_cells:
        a[p[0] - r0, p[1] - c0] = b[p[0] - r0, p[1] - c0] = structure
    for p, color in original.items():
        a[p[0] - r0, p[1] - c0] = color
    for p in swept - set(final):
        b[p[0] - r0, p[1] - c0] = 0
    for p, color in final.items():
        b[p[0] - r0, p[1] - c0] = color
    return np.rot90(a, rotation).copy(), np.rot90(b, rotation).copy()


def _place(rng, modules, bg):
    """Random separated placements; a finite two-band fallback fits small modules."""
    if len(modules) == 1:
        a, b = modules[0]
        h = int(rng.integers(max(18, a.shape[0] + 2), 31))
        w = int(rng.integers(max(18, a.shape[1] + 2), 31))
        r, c = int(rng.integers(1, h - a.shape[0])), int(rng.integers(1, w - a.shape[1]))
        grid, output = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
        grid[r:r + a.shape[0], c:c + a.shape[1]] = a
        output[r:r + b.shape[0], c:c + b.shape[1]] = b
        return {"input": grid, "output": output}
    h, w = int(rng.integers(27, 31)), int(rng.integers(27, 31))
    for attempt in range(64):
        placed = []
        for a, b in modules:
            r = int(rng.integers(1, h - a.shape[0]))
            c = int(rng.integers(1, w - a.shape[1]))
            if any(not (r + a.shape[0] < y or y + hh < r or c + a.shape[1] < x or x + ww < c)
                   for y, x, hh, ww in placed):
                break
            placed.append((r, c, *a.shape))
        if len(placed) == len(modules):
            break
    else:
        # Two 2x2-cargo modules are at most 9x15 before rotation.
        h, w = 30, 30
        modules = [(a if a.shape[0] <= 9 else a.T, b if b.shape[0] <= 9 else b.T) for a, b in modules]
        placed = [(1, 1, *modules[0][0].shape), (19, 30 - modules[1][0].shape[1] - 1, *modules[1][0].shape)]
    grid, output = np.full((h, w), bg, np.int8), np.full((h, w), bg, np.int8)
    for (a, b), (r, c, hh, ww) in zip(modules, placed):
        grid[r:r + hh, c:c + ww] = a
        output[r:r + hh, c:c + ww] = b
    return {"input": grid, "output": output}


def _integer(value, lo, hi):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and lo <= int(value) <= hi


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(num_train, 2, 10) or not _integer(num_test, 1, 4) or num_train <= num_test:
        raise ValueError("require integer counts 2..10 train, 1..4 test, train > test")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    puzzle, seen = {"train": [], "test": []}, set()
    rotation0 = int(rng.integers(4))
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            for attempt in range(40):
                palette = list(map(int, rng.choice(np.arange(1, 10), 5, replace=False)))
                bg, structure, ink, fake_ink, ink2 = palette
                rotation = (rotation0 + index) % 4 if split == "train" and index < 2 else int(rng.integers(4))
                multi = split == "train" and index >= 2 and bool(rng.integers(2))
                if multi:
                    modules = [_module(rng, (2, 2), (bg, structure, ci, fake_ink), int(rng.integers(4)), decoy=False)
                               for ci in (ink, ink2)]
                else:
                    shape = (3, 3) if split == "test" else ((2, 2) if index == 0 else (2, 3))
                    modules = [_module(rng, shape, (bg, structure, ink, fake_ink), rotation,
                                       nose=split == "train" and index == 0,
                                       asymmetric=split == "train" and index == 1,
                                       counterfork=split == "train" and index < 2)]
                pair = _place(rng, modules, bg)
                key = pair["input"].shape, pair["input"].tobytes()
                try:
                    expected = solve(pair["input"])
                except ValueError:
                    continue
                if key not in seen and np.array_equal(expected, pair["output"]):
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("bounded cargo placement attempts exhausted")
    return puzzle


def _counterfactuals(a, info):
    """Useful diagnostic alternatives, never used to produce ground truth."""
    no_b, box_trail = a.copy(), a.copy()
    for b in (no_b, box_trail):
        _paint(b, info["physical"], info["background"])
    nose_changes_stop = False
    for m in info["matches"]:
        cargo, f = info["cargos"][m["ci"]], info["forks"][m["fi"]]
        dr, dc = m["delta"]
        # Keep the correct displacement but omit all inferred tiles and their paths.
        visible_swept = {(r - t * f["out"][0], c - t * f["out"][1])
                         for r, c in cargo["observed"] for t in range(m["distance"] + 1)}
        visible_final = {(r + dr, c + dc): int(a[r, c]) for r, c in cargo["observed"]}
        _paint(no_b, visible_swept - set(visible_final), 0)
        for p, color in visible_final.items():
            no_b[p] = color
        r0, c0, r1, c1 = _bounds(cargo["pixels"])
        sweep_box = {(r - t * f["out"][0], c - t * f["out"][1])
                     for r, c in _rect(r0, c0, r1 - r0, c1 - c0) for t in range(m["distance"] + 1)}
        _paint(box_trail, sweep_box - set(m["final"]), 0)
        for p, color in m["final"].items():
            box_trail[p] = color
        projection = lambda p: p[0] * f["out"][0] + p[1] * f["out"][1]
        nose_changes_stop |= min(map(projection, cargo["observed"])) != min(map(projection, cargo["pixels"]))
    # B alone completes every constellation at its original location, with no
    # receiving test, rigid motion, trail, or distractor removal.
    no_a = a.copy()
    for cargo in info["cargos"]:
        for p, color in cargo["pixels"].items():
            no_a[p] = color
    return no_a, no_b, box_trail, nose_changes_stop


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "pairs must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(_integer(nt, 2, 10) and _integer(ne, 1, 4) and nt > ne, "invalid counts")
    seen, train_info = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "output dimensions changed")
            key = a.shape, a.tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _, info = _analyze(a)
            _require(np.array_equal(solve(a), b), "incorrect docking output")
            _require(np.any(b == 0) and not np.array_equal(a, b), "missing swept path")
            for f in info["forks"]:
                _require(all(int(b[p]) == info["structure"] for p in f["cells"]), "fork changed")
            for m in info["matches"]:
                cargo = info["cargos"][m["ci"]]
                _require(cargo["missing"] and m["trail"], "completion and travel must both matter")
                _require(all(int(b[p]) == color for p, color in m["final"].items()), "whole-tile copy failed")
                _require(all(int(b[p]) == 0 for p in m["trail"]), "trail is not the exact swept shape")
                _require(len(m["final"]) == cargo["active_count"] * cargo["tile"].size, "cargo area changed")
            for ci, cargo in enumerate(info["cargos"]):
                if ci not in info["used"]:
                    _require(all(int(b[p]) == info["background"] for p in cargo["observed"]), "decoy survived")
            no_a, no_b, box, nose = _counterfactuals(a, info)
            _require(not np.array_equal(b, no_a) and not np.array_equal(b, no_b), "one parent can be omitted")
            if split == "train":
                train_info.append((info, box, nose, b))
    _require(train_info[0][2], "first demonstration must expose an invisible leading tile")
    _require(any(len(i["used"]) < len(i["cargos"]) for i, box, nose, b in train_info), "missing erasure witness")
    _require(any(_match(f, c, np.asarray(pair["input"]), info["background"], check_span=False) is not None
                 and _match(f, c, np.asarray(pair["input"]), info["background"]) is None
                 for pair, (info, box, nose, b) in zip(puzzle["train"], train_info)
                 for ci, c in enumerate(info["cargos"]) if ci not in info["used"]
                 for f in info["forks"]), "missing isolated width-matching witness")
    _require(any(not np.array_equal(box, b) for i, box, nose, b in train_info), "bounding-box trails fit all demonstrations")
    orientation = set(range(8))
    for info, box, nose, b in train_info:
        for m in info["matches"]:
            orientation &= _symmetries(info["cargos"][m["ci"]]["mask"])
    _require(orientation == {0}, "demonstrations must distinguish same-orientation completion")


def self_test():
    """42 seeds, streamed statistics, every legal count, and semantic mutations."""
    shapes, designs, fragments, forks, directions, palettes, normalized, signatures = (set() for _ in range(8))
    matched_counts, missing_counts, distances = set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signature = tuple((x["input"].shape, x["input"].tobytes()) for s in ("train", "test") for x in p[s])
        signatures.add(signature)
        for s in ("train", "test"):
            for pair in p[s]:
                a, info = _analyze(pair["input"])
                shapes.add(a.shape)
                palettes.add((info["background"], info["structure"], tuple(c["ink"] for c in info["cargos"])))
                roles = {}
                normalized.add((a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                matched_counts.add(len(info["matches"]))
                for f in info["forks"]:
                    forks.add((f["span"], f["lengths"]))
                    directions.add(f["out"])
                for c in info["cargos"]:
                    designs.add((c["mask"].shape, c["mask"].tobytes()))
                    fragments.add((c["mask"].shape, c["mask"].tobytes(), c["fragment"]))
                    missing_counts.add(c["active_count"] - c["visible_count"])
                distances.update(m["distance"] for m in info["matches"])
    _require(len(signatures) == 42 and len(normalized) >= 180, "insufficient structural variation")
    _require(len(shapes) >= 30 and len(designs) >= 50 and len(fragments) >= 100, "insufficient glyph variation")
    _require(len(forks) >= 80 and len(palettes) >= 100 and len(directions) == 4, "insufficient receiving variation")
    _require(matched_counts == {1, 2} and {1, 2, 3} <= missing_counts and distances == {1, 2, 3, 4, 5}, "missing count/travel variation")
    p, q = generate_puzzle(123, 3, 2), generate_puzzle(123, 3, 2)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x), "nondeterminism")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * nt + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (3.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(7, 2, 1)
    pair = p["test"][0]
    _, info = _analyze(pair["input"])
    m = info["matches"][0]
    for point, replacement in ((min(m["trail"]), info["background"]), (min(m["final"]), 0), ((0, 0), 0)):
        old = int(pair["output"][point])
        pair["output"][point] = replacement
        try:
            validate_generated_puzzle(p)
        except ValueError:
            pass
        else:
            raise AssertionError("semantic corruption accepted")
        pair["output"][point] = old
    ambiguous = np.array([[1, 1, 1], [1, 0, 1]], dtype=bool)
    _require(len(_registrations(ambiguous, {(0, 0), (0, 1)})) > 1, "registration ties must be detected")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "glyph_designs": len(designs),
            "glyph_fragments": len(fragments), "fork_geometries": len(forks),
            "color_normalized_inputs": len(normalized), "role_palettes": len(palettes),
            "matched_counts": sorted(matched_counts), "missing_counts": sorted(missing_counts),
            "distances": sorted(map(int, distances)), "independent_solver": False}
