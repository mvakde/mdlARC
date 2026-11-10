"""Perforated cargo on rewired rails. No file I/O or parent imports."""
from __future__ import annotations

from itertools import product
import hashlib

import numpy as np


BACKGROUND, RAIL, WALL, STENCIL = 0, 8, 5, 9
INKS = (1, 2, 3, 4, 6, 7)
DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and bool(np.all((0 <= a) & (a <= 9))),
             "grid must contain integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _normalize(grid):
    # The unique complete boundary line is a visible gravity cue.
    candidates = [(np.rot90(grid, k).copy(), k) for k in range(4)
                  if np.all(np.rot90(grid, k)[-1] == WALL)]
    _require(len(candidates) == 1, "expected one boundary floor")
    return candidates[0]


def _read_codes(header):
    """Read source -> target -> registration rays, including initial holes.

    Codes are at most seven cells long, contain at least two filled cells,
    and contain no run of three holes. Three blanks terminate a ray. A
    boundary or another non-background/non-stencil cell also terminates it.
    Whole-header exact coverage and unique source labels reject ambiguity.
    """
    h, w = header.shape
    candidates = []
    for r, c in np.argwhere(np.isin(header, INKS)):
        r, c = int(r), int(c)
        source = int(header[r, c])
        for dr, dc in DIRECTIONS:
            tr, tc = r + dr, c + dc
            if not (0 <= tr < h and 0 <= tc < w):
                continue
            target = int(header[tr, tc])
            if target not in INKS or target == source:
                continue
            q = (r + 2 * dr, c + 2 * dc)
            offsets, blanks = [], 0
            for j in range(10):
                rr, cc = q[0] + j * dr, q[1] + j * dc
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                color = int(header[rr, cc])
                if color == STENCIL:
                    offsets.append(j)
                    blanks = 0
                elif color == BACKGROUND:
                    blanks += 1
                    if blanks == 3:
                        break
                else:
                    break
            if len(offsets) < 2 or offsets[-1] >= 7:
                continue
            mask = {(q[0] + j * dr, q[1] + j * dc) for j in offsets}
            candidates.append({"source": source, "target": target, "q": q,
                               "direction": (dr, dc), "bits": tuple(offsets),
                               "length": offsets[-1] + 1,
                               "mask": mask, "labels": {(r, c), (tr, tc)}})
    _require(1 <= len(candidates) <= 3, "missing or ambiguous stencil instructions")
    _require(len({x["source"] for x in candidates}) == len(candidates), "repeated source instruction")
    cells = set()
    for code in candidates:
        footprint = code["mask"] | code["labels"]
        _require(not (cells & footprint), "overlapping or ambiguous instructions")
        cells |= footprint
    actual = {tuple(map(int, p)) for p in np.argwhere(header != BACKGROUND)}
    _require(cells == actual, "unassigned header content")
    return sorted(candidates, key=lambda x: x["source"])


def _neighbors(cell, cells):
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRECTIONS if (r + dr, c + dc) in cells]


def _trace(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    paths = []
    while remaining:
        start = min(remaining)
        component, todo = {start}, [start]
        for cell in todo:
            for neighbor in _neighbors(cell, remaining):
                if neighbor not in component:
                    component.add(neighbor)
                    todo.append(neighbor)
        ends = []
        for cell in component:
            degree = len(_neighbors(cell, component))
            _require(degree in (1, 2), "rail branch or isolated rail")
            if degree == 1:
                ends.append(cell)
        _require(len(ends) == 2, "rail cycle")
        path, previous = [min(ends)], None
        while len(path) < len(component):
            options = [p for p in _neighbors(path[-1], component) if p != previous]
            _require(len(options) == 1, "ambiguous rail traversal")
            previous = path[-1]
            path.append(options[0])
        _require(path[-1] == max(ends), "invalid traversal")
        steps = {b[0] - a[0] for a, b in zip(path, path[1:]) if b[0] != a[0]}
        _require(len(steps) <= 1, "rail has an internal gravity reversal")
        paths.append(path)
        remaining -= component
    return sorted(paths, key=lambda p: (p[0], p[-1]))


def _rewire(core):
    h, w = core.shape
    _require(h >= 5 and h % 2 == 1 and w >= 14, "invalid rail core")
    _require(np.all(np.isin(core[::2], (BACKGROUND, RAIL) + INKS)), "invalid lane color")
    _require(np.all(np.isin(core[1::2], (BACKGROUND, WALL))), "invalid separator color")
    _require(np.all(np.isin(core[::2, [0, w - 1]], (RAIL,) + INKS)), "missing outer terminal")
    turned = core.copy()
    turned[np.isin(turned, INKS)] = RAIL
    centers, writes, overlaps = [], {}, 0
    for r in range(1, h, 2):
        xs = set(map(int, np.flatnonzero(core[r] == WALL)))
        for c0 in sorted(c for c in xs if c - 1 not in xs):
            _require({c0, c0 + 1, c0 + 2} <= xs and c0 + 3 not in xs, "divider is not a triple")
            c = c0 + 1
            _require(2 <= c <= w - 3, "divider too close to outer terminal")
            for rr in (r - 1, r + 1):
                _require(np.all(np.isin(core[rr, c - 1:c + 2], (RAIL,) + INKS)),
                         "divider neighborhood contains a break")
                _require(core[rr, c] == RAIL, "token would be overwritten by a divider")
            for rr in range(r - 1, r + 2):
                for cc in range(c - 1, c + 2):
                    value = WALL if cc == c else RAIL
                    if (rr, cc) in writes:
                        _require(writes[rr, cc] == value, "conflicting divider rotations")
                        overlaps += 1
                    writes[rr, cc] = value
            centers.append((r, c))
    _require(bool(centers), "no dividers")
    for p, value in writes.items():
        turned[p] = value
    paths = _trace(turned == RAIL)
    endpoints = {p for path in paths for p in (path[0], path[-1])}
    tokens = {tuple(map(int, p)): int(core[tuple(p)]) for p in np.argwhere(np.isin(core, INKS))}
    _require(set(tokens) <= endpoints, "cargo markers must be exposed terminals")
    for r, c in endpoints:
        _require(r % 2 == 0 and (c in (0, w - 1) or core[r, c - 1] == BACKGROUND
                                 or core[r, c + 1] == BACKGROUND), "unexposed path endpoint")
    return turned, paths, tokens, centers, overlaps


def _fit(bits, occupied, length, start=1):
    """Nearest inward registration whose opaque cells do not collide."""
    for origin in range(start, length - max(bits)):
        footprint = {origin + i for i in bits}
        if not (footprint & occupied):
            return origin
    raise ValueError("cargo cannot fit on this path")


def _pack(structural, paths, tokens, codes, mode="rule"):
    """Intrinsic path coordinates couple stencil geometry to gravity packing.

    Alternative modes are explicit counterfactual diagnostics, never solve
    conventions: adjacent, intervals, solid, original, centered, reverse_queue.
    """
    output = structural.copy()
    records = []
    for path in paths:
        marked = [p for p in (path[0], path[-1]) if p in tokens]
        if not marked:
            continue
        level = path[0][0] == path[-1][0]
        inward = path if path[0][0] > path[-1][0] else path[::-1]
        ordered_markers = [p for p in (inward[0], inward[-1]) if p in tokens]
        if mode == "reverse_queue" and not level:
            ordered_markers.reverse()
        occupied = set()
        previous_length = 0
        for index, marker in enumerate(ordered_markers):
            code = codes[tokens[marker]]
            bits = code["bits"]
            if mode == "solid":
                bits = tuple(range(code["length"]))
            elif mode == "centered":
                # Incorrectly aligning the first opaque cell, erasing a registration hole.
                bits = tuple(i - min(bits) for i in bits)
            route = inward
            if level or mode == "original":
                route = path if marker == path[0] else path[::-1]
                origin = 0
            elif index == 0:
                origin = 0
            elif mode == "adjacent":
                origin = index
            elif mode == "intervals":
                origin = previous_length
            else:
                origin = _fit(bits, occupied, len(route))
            _require(origin + max(bits) < len(route), "stencil extends past path boundary")
            paint_indices = {origin + i for i in bits}
            footprint = {route[i] for i in paint_indices}
            # All paths are disjoint; level-end copies must also be disjoint.
            if mode == "rule":
                _require(all(output[p] == RAIL for p in footprint), "cargo paint collision")
            for p in footprint:
                output[p] = code["target"]
            records.append({"source": tokens[marker], "target": code["target"],
                            "marker": marker, "origin": origin, "level": level,
                            "length": code["length"], "bits": bits,
                            "route": route, "footprint": footprint})
            occupied |= paint_indices
            previous_length = max(previous_length, origin + code["length"])
    return output, records


def _analyze(input_grid):
    original = _grid(input_grid)
    canonical, rotation = _normalize(original)
    rows = np.flatnonzero(np.any(canonical == RAIL, axis=1))
    _require(len(rows) >= 3, "missing lanes")
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    _require(rows.tolist() == list(range(r0, r1, 2)), "lanes are not parallel")
    _require(r0 >= 4 and r1 <= canonical.shape[0] - 2, "missing instruction or floor clearance")
    _require(np.all(canonical[r0 - 1] == BACKGROUND), "no header clearance")
    _require(np.all(canonical[r1:-1] == BACKGROUND), "floor clearance is not blank")
    instructions = _read_codes(canonical[:r0 - 1])
    codes = {x["source"]: x for x in instructions}
    core = canonical[r0:r1]
    structural, paths, tokens, centers, overlaps = _rewire(core)
    _require(bool(tokens) and set(tokens.values()) == set(codes), "unused instruction or unknown cargo")
    packed, records = _pack(structural, paths, tokens, codes)
    answer = canonical.copy()
    answer[:r0] = BACKGROUND
    answer[r0:r1] = packed
    return {"original": original, "canonical": canonical, "rotation": rotation,
            "r0": r0, "r1": r1, "core": core, "structural": structural,
            "paths": paths, "tokens": tokens, "codes": codes, "centers": centers,
            "overlaps": overlaps, "records": records, "answer": answer}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one input; no seed, demonstrations, generation state, or I/O."""
    info = _analyze(input_grid)
    return np.rot90(info["answer"], -info["rotation"]).copy()


def _valid_bits(bits):
    return (2 <= len(bits) <= 7 and bits[-1] == 1 and sum(bits) >= 2
            and not any(bits[i:i + 3] == (0, 0, 0) for i in range(len(bits) - 2)))


def _bit_catalog():
    # Enumerating binary strips is bounded (252 raw candidates), not a shape-template bank.
    return [bits for n in range(2, 8) for bits in product((0, 1), repeat=n) if _valid_bits(bits)]


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _sample_codes(rng, palette, teaching, is_test, excluded):
    catalog = _bit_catalog()
    n = 1 if teaching == 0 else 2 if teaching == 1 else int(rng.integers(2 if is_test else 1, 4))
    colors = list(map(int, rng.permutation(palette)))
    if teaching in (0, 1):
        colors = list(palette)
    mappings = [(colors[i], _choose(rng, [c for c in palette if c != colors[i]])) for i in range(n)]
    if teaching == 0:
        mappings[0] = (palette[0], palette[3])
    elif teaching == 1:
        mappings = [(palette[0], palette[4]), (palette[1], palette[0])]
    result = []
    for i, (source, target) in enumerate(mappings):
        choices = catalog
        if teaching == 0:
            choices = [b for b in catalog if b[0] == 1 and
                       1 < _fit(tuple(j for j, v in enumerate(b) if v),
                                {j for j, v in enumerate(b) if v}, 2 * len(b)) < len(b)]
        elif teaching == 1:
            choices = [b for b in catalog if (b[0] == 0) == (i == 0)]
        if is_test and i == 0:
            choices = [b for b in choices if b not in excluded]
        bits = _choose(rng, choices)
        result.append({"source": source, "target": target,
                       "bits": tuple(j for j, v in enumerate(bits) if v), "length": len(bits)})
    return result


def _header(rng, codes, width, height):
    empty = np.zeros((height, width), dtype=np.int8)
    grid = empty.copy()
    reserved = set()
    for i, code in enumerate(codes):
        for attempt in range(240):
            dr, dc = _choose(rng, DIRECTIONS)
            # First two visible labels precede the registered strip.
            relative = [(j * dr, j * dc) for j in range(-2, code["length"])]
            rlo, rhi = min(r for r, c in relative), max(r for r, c in relative)
            clo, chi = min(c for r, c in relative), max(c for r, c in relative)
            if rhi - rlo >= height or chi - clo >= width:
                continue
            q = (int(rng.integers(-rlo, height - rhi)), int(rng.integers(-clo, width - chi)))
            cells = {(q[0] + r, q[1] + c) for r, c in relative}
            if cells & reserved:
                continue
            trial = grid.copy()
            trial[q[0] - 2 * dr, q[1] - 2 * dc] = code["source"]
            trial[q[0] - dr, q[1] - dc] = code["target"]
            for j in code["bits"]:
                trial[q[0] + j * dr, q[1] + j * dc] = STENCIL
            try:
                parsed = _read_codes(trial)
            except ValueError:
                continue
            expected = {(x["source"], x["target"], x["bits"]) for x in codes[:i + 1]}
            if {(x["source"], x["target"], x["bits"]) for x in parsed} == expected:
                grid, reserved = trial, reserved | cells
                break
        else:
            # Always fits: at most three vertical strips in a 9-by-14 header.
            fallback = np.zeros((max(9, height), width), dtype=np.int8)
            for j, item in enumerate(codes):
                c = 2 + 4 * j
                fallback[0, c], fallback[1, c] = item["source"], item["target"]
                for offset in item["bits"]:
                    fallback[2 + offset, c] = STENCIL
            _read_codes(fallback)
            return fallback
    return grid


def _base(lanes, width, centers, breaks):
    core = np.zeros((2 * lanes - 1, width), dtype=np.int8)
    core[::2] = RAIL
    for r, c in centers:
        core[r, c - 1:c + 2] = WALL
    for p in breaks:
        core[p] = BACKGROUND
    return core


def _layout(rng, max_length, require_level):
    for attempt in range(60):
        lanes, width = int(rng.integers(3, 10)), int(rng.integers(14, 31))
        centers, previous = [], []
        for r in range(1, 2 * lanes - 2, 2):
            row = []
            wanted = int(rng.choice((0, 1, 1, 1, 2, 2, 3)))
            for c in map(int, rng.permutation(np.arange(2, width - 2))):
                if len(row) == wanted:
                    break
                if all(abs(c - x) >= 4 for x in row) and all(abs(c - x) >= 2 for x in previous):
                    row.append(c)
            centers.extend((r, c) for c in sorted(row))
            previous = row
        if not centers:
            continue
        breaks = []
        for r in range(0, 2 * lanes - 1, 2):
            ports = sorted((c, rr) for rr, c in centers if abs(rr - r) == 1)
            for (a, side), (b, next_side) in zip(ports, ports[1:]):
                if side == next_side:
                    breaks.append((r, int(rng.integers(a + 2, b - 1))))
        pool = [(r, c) for r in range(0, 2 * lanes - 1, 2) for c in range(2, width - 2)
                if all(abs(c - cc) >= 2 for rr, cc in centers if abs(rr - r) == 1)]
        rng.shuffle(pool)
        for p in pool[:int(rng.integers(0, 5))]:
            if all(p[0] != r or abs(p[1] - c) >= 3 for r, c in breaks):
                breaks.append(p)
        core = _base(lanes, width, centers, breaks)
        try:
            structural, paths, _, _, _ = _rewire(core)
        except ValueError:
            continue
        moving = [p for p in paths if p[0][0] != p[-1][0] and len(p) >= max_length]
        long = [p for p in moving if len(p) >= 2 * max_length]
        levels = [p for p in paths if p[0][0] == p[-1][0] and len(p) >= max_length]
        if len(moving) >= 2 and long and (levels or not require_level):
            return core, structural, paths
    # Constructive, bounded fallback; all code lengths <=7 fit.
    core = _base(3, 26, [(1, 8), (3, 17)], [(0, 15)])
    structural, paths, _, _, _ = _rewire(core)
    return core, structural, paths


def _put_cargo(rng, core, structural, paths, codes, require_level):
    mapping = {x["source"]: x for x in codes}
    max_length = max(x["length"] for x in codes)
    moving = [p for p in paths if p[0][0] != p[-1][0] and len(p) >= max_length]
    long = [p for p in moving if len(p) >= 2 * max_length]
    double = _choose(rng, long)
    other = _choose(rng, [p for p in moving if p is not double])
    tokens = {}

    def add(path, low_source=None, high_source=None):
        low, high = max(path[0], path[-1]), min(path[0], path[-1])
        if low_source is not None:
            tokens[low] = low_source
        if high_source is not None:
            tokens[high] = high_source

    add(double, codes[0]["source"], codes[min(1, len(codes) - 1)]["source"])
    add(other, high_source=codes[0]["source"])
    if require_level:
        levels = [p for p in paths if p[0][0] == p[-1][0] and len(p) >= max_length]
        path = _choose(rng, levels)
        tokens[_choose(rng, (path[0], path[-1]))] = codes[0]["source"]
    for code in codes:
        if code["source"] in tokens.values():
            continue
        pool = [p for p in paths if len(p) >= code["length"] and
                p[0] not in tokens and p[-1] not in tokens]
        _require(bool(pool), "no path for an instruction")
        path = _choose(rng, pool)
        tokens[_choose(rng, (path[0], path[-1]))] = code["source"]
    for path in paths:
        if path[0] in tokens or path[-1] in tokens or rng.random() < .48:
            continue
        choices = [c for c in codes if c["length"] <= len(path)]
        if not choices:
            continue
        a = _choose(rng, choices)
        if path[0][0] != path[-1][0] and len(path) >= 2 * max_length and rng.random() < .4:
            add(path, a["source"], _choose(rng, codes)["source"])
        else:
            tokens[_choose(rng, (path[0], path[-1]))] = a["source"]
    output, records = _pack(structural, paths, tokens, mapping)
    inp = core.copy()
    for p, source in tokens.items():
        inp[p] = source
    return inp, output, records


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not lo <= int(value) <= hi:
        raise ValueError(name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(INKS)))
    puzzle, seen, familiar = {"train": [], "test": []}, set(), set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            teaching = index if split == "train" and index < 2 else None
            codes = _sample_codes(rng, palette, teaching, split == "test", familiar)
            for attempt in range(30):
                core, structural, paths = _layout(rng, max(x["length"] for x in codes), teaching == 1)
                try:
                    inp_core, out_core, records = _put_cargo(rng, core, structural, paths, codes, teaching == 1)
                except ValueError:
                    continue
                if teaching == 0 and (not np.any(np.isin(out_core[1::2], INKS)) or
                                      np.array_equal(_straight(structural, records), out_core)):
                    continue
                header_h = int(rng.integers(max(5, max(x["length"] for x in codes) + 2), 11))
                header = _header(rng, codes, core.shape[1], header_h)
                gap = np.zeros((1, core.shape[1]), dtype=np.int8)
                floor = np.full((1, core.shape[1]), WALL, dtype=np.int8)
                inp = np.vstack((header, gap, inp_core, gap, floor))
                out = np.vstack((np.zeros_like(header), gap, out_core, gap, floor))
                k = int(rng.integers(4))
                inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
                key = (inp.shape, inp.tobytes())
                if key in seen:
                    continue
                _require(np.array_equal(solve(inp), out), "construction differs from input decoding")
                seen.add(key)
                puzzle[split].append({"input": inp, "output": out})
                break
            else:
                raise RuntimeError("bounded construction exhausted")
            if split == "train":
                familiar.update(tuple(int(j in c["bits"]) for j in range(c["length"])) for c in codes)
    return puzzle


def _counterfactual(info, mode):
    core, _ = _pack(info["structural"], info["paths"], info["tokens"], info["codes"], mode)
    return core


def _straight(structural, records):
    """Wrong alternative: ordinary straight stamps at otherwise correct origins."""
    result = structural.copy()
    h, w = result.shape
    for rec in records:
        route, q = rec["route"], rec["origin"]
        r, c = route[q]
        if q + 1 < len(route):
            dr, dc = route[q + 1][0] - r, route[q + 1][1] - c
        else:
            dr, dc = r - route[q - 1][0], c - route[q - 1][1]
        for j in rec["bits"]:
            rr, cc = r + j * dr, c + j * dc
            if 0 <= rr < h and 0 <= cc < w:
                result[rr, cc] = rec["target"]
    return result


def _features(info):
    records = info["records"]
    groups = {}
    for x in records:
        groups.setdefault(tuple(x["route"]), []).append(x)
    double = [g for g in groups.values() if len(g) == 2 and not g[0]["level"]]
    return {"shape": info["original"].shape, "lanes": (info["core"].shape[0] + 1) // 2,
            "width": info["core"].shape[1], "floor_rotation": info["rotation"],
            "dividers": len(info["centers"]), "overlaps": info["overlaps"],
            "breaks": int(np.count_nonzero(info["core"][::2] == BACKGROUND)),
            "codes": len(info["codes"]), "markers": len(info["tokens"]),
            "level_copies": sum(x["level"] for x in records),
            "moved_single": any(len(g) == 1 and not g[0]["level"] and
                                g[0]["marker"] != g[0]["route"][0] for g in groups.values()),
            "mixed_queue": any(g[0]["source"] != g[1]["source"] for g in double),
            "interlocked": any(1 < g[1]["origin"] < g[0]["length"] for g in double),
            "nonadjacent": any(g[1]["origin"] > 1 for g in double)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _count(len(puzzle["train"]), "training count", 2, 10)
    _count(len(puzzle["test"]), "test count", 1, 4)
    _require(len(puzzle["train"]) > len(puzzle["test"]), "invalid split counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _analyze(inp)
            expected = np.rot90(info["answer"], -info["rotation"])
            _require(np.array_equal(expected, out), "incorrect output")
            analyses[split].append(info)
            # Per-token, not just per-color conservation: every opaque bit is deposited once.
            ink_count = sum(len(x["bits"]) for x in info["records"])
            output_core = info["answer"][info["r0"]:info["r1"]]
            _require(np.count_nonzero(np.isin(output_core, INKS)) == ink_count, "lost or overlapping opaque bits")
            _require(not np.any(out == STENCIL), "instruction was not erased")
            # Divider/floor geometry survives; paint occupies only genuine rewired rail cells.
            _require(np.all(output_core[info["structural"] != RAIL] ==
                            info["structural"][info["structural"] != RAIL]), "nonrail changed")
    training = analyses["train"]
    features = [_features(x) for x in training]
    _require(any(f["interlocked"] and f["nonadjacent"] for f in features), "no interlocking packing demonstration")
    _require(any(f["mixed_queue"] for f in features), "no colored queue order demonstration")
    _require(any(f["moved_single"] for f in features), "no transport demonstration")
    _require(any(f["level_copies"] for f in features), "no level-end demonstration")
    _require(any(min(c["bits"]) > 0 for x in training for c in x["codes"].values()), "no registration hole demonstration")
    for mode in ("adjacent", "intervals", "solid", "original", "centered", "reverse_queue"):
        _require(any(not np.array_equal(_counterfactual(x, mode), x["answer"][x["r0"]:x["r1"]])
                     for x in training), "training fails to distinguish shortcut: " + mode)
    _require(any(not np.array_equal(_straight(x["structural"], x["records"]),
                                    x["answer"][x["r0"]:x["r1"]]) for x in training),
             "training must distinguish path unfolding from straight stamping")
    known = {tuple(int(j in c["bits"]) for j in range(c["length"]))
             for x in training for c in x["codes"].values()}
    for info in analyses["test"]:
        local = {tuple(int(j in c["bits"]) for j in range(c["length"])) for c in info["codes"].values()}
        _require(bool(local - known), "test must read a new visible code")
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = _grid(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Streaming checks: 42 puzzles, no files, no retained grid corpus."""
    # Hand-worked full-grid regression. A six-cell 101101 stencil occurs
    # twice on a bent 22-cell route. Registrations 0 and 4 produce occupied
    # indices 0,2,3,4,5,6,7,9, including a different lane after the bend.
    manual = np.zeros((13, 14), dtype=np.int8)
    manual[0, 0:2] = (1, 2)
    manual[0, [2, 4, 5, 7]] = STENCIL
    manual[6:11:2] = RAIL
    manual[7, 3:6] = WALL
    manual[9, 7:10] = WALL
    manual[6, 13] = manual[10, 0] = 1
    manual[-1] = WALL
    expected = np.zeros_like(manual)
    expected[6:11:2] = RAIL
    expected[6:9, 4] = WALL
    expected[8:11, 8] = WALL
    expected[7, [3, 5]] = RAIL
    expected[9, [7, 9]] = RAIL
    expected[10, [0, 2, 3, 4, 5, 6, 7]] = 2
    expected[8, 7] = 2
    expected[-1] = WALL
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(manual, k)), np.rot90(expected, k)),
                 "hand-worked bend/interlocking regression failed")
    # A leading registration hole stays unpainted at the sink.
    hole = manual.copy()
    hole[:6] = BACKGROUND
    hole[0, 0:2] = (1, 2)
    hole[0, [3, 4]] = STENCIL  # registered strip 011
    hole[6, 13], hole[10, 0], hole[6, 0] = RAIL, RAIL, 1
    hole_expected = expected.copy()
    hole_expected[hole_expected == 2] = RAIL
    hole_expected[8, [1, 2]] = 2
    _require(np.array_equal(solve(hole), hole_expected), "hand-worked registration-hole regression failed")
    digests, shapes, layouts, strips, directions, lanes, breaks, dividers = (set() for _ in range(8))
    totals = {"pairs": 0, "mixed_queues": 0, "interlocked_pairs": 0, "compatible_overlaps": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_digest(puzzle))
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic generation")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                before = np.asarray(pair["input"]).copy()
                _require(np.array_equal(solve(before), pair["output"]), "solve mismatch")
                _require(np.array_equal(before, pair["input"]), "solve mutated its argument")
                info = _analyze(before)
                f = _features(info)
                shapes.add(f["shape"])
                lanes.add(f["lanes"])
                breaks.add(f["breaks"])
                dividers.add(f["dividers"])
                directions.add(f["floor_rotation"])
                layouts.add((info["core"].shape, info["structural"].tobytes()))
                strips.update((c["bits"], c["length"]) for c in info["codes"].values())
                totals["pairs"] += 1
                totals["mixed_queues"] += int(f["mixed_queue"])
                totals["interlocked_pairs"] += int(f["interlocked"])
                totals["compatible_overlaps"] += int(f["overlaps"] > 0)
    _require(len(digests) == 42 and len(shapes) >= 35 and len(layouts) >= 100, "insufficient structural variation")
    _require(len(strips) >= 40 and directions == set(range(4)), "insufficient code or orientation variation")
    _require(len(lanes) >= 6 and len(breaks) >= 5 and len(dividers) >= 8, "insufficient topology variation")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(901, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupt = generate_puzzle(7, 2, 1)
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    return dict(totals, seeds=42, shapes=len(shapes), layouts=len(layouts), strips=len(strips),
                floor_directions=len(directions), lane_counts=sorted(lanes),
                break_counts=sorted(breaks), divider_counts=sorted(dividers))
