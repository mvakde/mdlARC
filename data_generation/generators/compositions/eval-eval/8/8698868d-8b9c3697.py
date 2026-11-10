"""Count-keyed stencils: cavity-selected plates sweep negative space into forks.

Standard library and NumPy only. Prediction uses only the supplied input.
Construction paints projection intervals; solve independently simulates each
unit translation of recovered cavity cells. No parent generator is imported.
"""

from itertools import permutations

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # outward vectors


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask, diagonal=False):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else _STEPS
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                point = r + dr, c + dc
                if point in unseen:
                    unseen.remove(point)
                    cells.add(point)
                    stack.append(point)
        result.append(cells)
    return result


def _bounds(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _rectangle(r0, r1, c0, c1):
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _paint(grid, cells, color):
    for point in cells:
        grid[point] = color


def _checked_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return grid.astype(int, copy=False)


def _parse_fork(cells, color, grid, background):
    parses = []
    for out in _DIRECTIONS:
        perp = (-out[1], out[0])
        kmin = min(r * out[0] + c * out[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if width < 8 or max(k for k, _ in local) < 3:
            continue
        full = set(range(width))
        if any({q for k, q in local if k == d} != full for d in (0, 2)):
            continue
        tally_q = sorted(full - {q for k, q in local if k == 1})
        if (len(tally_q) > 4 or any(q in (0, width - 1) for q in tally_q)
                or any(b - a < 2 for a, b in zip(tally_q, tally_q[1:]))):
            continue
        lengths = [sum((k, q) in local for k in range(3, 31)) for q in range(width)]
        expected = _rectangle(0, 3, 0, width) - {(1, q) for q in tally_q}
        expected |= {(k, q) for q, length in enumerate(lengths) for k in range(3, 3 + length)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if (len(runs) != 3 or runs[0][0] != 0 or runs[2][1] != width
                or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1))
                or runs[0][1] != width - runs[2][0]):
            continue
        q0, q1, length = runs[1]
        if q0 + q1 != width or q1 - q0 < 4:
            continue
        fork = {"out": out, "perp": perp, "kmin": kmin, "qmin": qmin,
                "q0": q0, "q1": q1, "stop": 3 + length, "width": width,
                "runs": tuple(runs), "code": len(tally_q), "color": color,
                "cells": cells}
        tally = {_world(fork, 1, q) for q in tally_q}
        if any(int(grid[p]) != background for p in tally):
            continue
        fork["tally"] = tally
        parses.append(fork)
    if len(parses) != 1:
        raise ValueError("a fork must have one three-deep tally base and three positive prongs")
    return parses[0]


def _world(fork, k, q):
    k, q = k + fork["kmin"], q + fork["qmin"]
    out, perp = fork["out"], fork["perp"]
    return k * out[0] + q * perp[0], k * out[1] + q * perp[1]


def _local(fork, cells):
    out, perp = fork["out"], fork["perp"]
    return {(r * out[0] + c * out[1] - fork["kmin"],
             r * perp[0] + c * perp[1] - fork["qmin"]) for r, c in cells}


def _candidate(fork, plate, structures, shape, *, topology=True, span=True):
    k0, k1, q0, q1 = _bounds(_local(fork, plate["envelope"]))
    if k0 <= fork["stop"]:
        return None
    if span:
        if (q0, q1) != (fork["q0"], fork["q1"]):
            return None
    elif q0 + q1 != fork["q0"] + fork["q1"]:
        return None
    if topology and plate["count"] != fork["code"]:
        return None
    route = {_world(fork, k, q) for k in range(fork["stop"], k1) for q in range(q0, q1)}
    if (not route.isdisjoint(structures)
            or any(not (0 <= r < shape[0] and 0 <= c < shape[1]) for r, c in route)):
        return None
    distance = k0 - fork["stop"]
    dr, dc = fork["out"]
    final = {(r - distance * dr, c - distance * dc) for r, c in plate["envelope"]}
    return {"distance": distance, "route": route, "final": final,
            "start": k0, "length": k1 - k0}


def _analyze(input_grid):
    grid = _checked_grid(input_grid)
    border = _perimeter(grid)
    if not np.all(border == border[0]):
        raise ValueError("the uniform canvas perimeter must identify the background")
    background = int(border[0])
    forks, plates = [], []
    for cells in _components(grid != background):
        r0, r1, c0, c1 = _bounds(cells)
        box = grid[r0:r1, c0:c1]
        colors = set(int(grid[p]) for p in cells)
        if len(colors) != 1:
            raise ValueError("touching differently colored objects")
        color = next(iter(colors))
        if min(box.shape) >= 4 and np.all(_perimeter(box) == color):
            if set(map(int, np.unique(box))) - {color, background}:
                raise ValueError("a plate must be binary")
            holes = {(r + r0, c + c0) for r, c in np.argwhere(box == background)}
            plates.append({"cells": cells, "envelope": _rectangle(r0, r1, c0, c1),
                           "holes": holes, "count": len(_components(box == background)),
                           "count8": len(_components(box == background, True)),
                           "color": color, "box": (r0, r1, c0, c1),
                           "shape": box.shape})
        else:
            forks.append(_parse_fork(cells, color, grid, background))
    if not forks or not plates:
        raise ValueError("scene needs forks and plates")
    if {f["color"] for f in forks} & {p["color"] for p in plates}:
        raise ValueError("plate foreground and fork ink must be distinct roles")
    structures = set().union(*(f["cells"] for f in forks))
    matches, used, swept = [], set(), set()
    for fi, fork in enumerate(forks):
        candidates = []
        for pi, plate in enumerate(plates):
            candidate = _candidate(fork, plate, structures, grid.shape)
            if candidate is not None:
                candidates.append((pi, candidate))
        if len(candidates) > 1:
            raise ValueError("multiple matching plates for one fork")
        if candidates:
            pi, candidate = candidates[0]
            if pi in used or not swept.isdisjoint(candidate["route"]):
                raise ValueError("conflicting assignments or swept envelopes")
            used.add(pi)
            swept |= candidate["route"]
            matches.append(dict(candidate, fi=fi, pi=pi))
    return {"background": background, "forks": forks, "plates": plates,
            "matches": matches, "structures": structures,
            "unmatched": len(plates) - len(used)}


def _trace(fork, plate, match):
    dr, dc = fork["out"]
    stamped = {(r - j * dr, c - j * dc)
               for j in range(match["distance"] + 1) for r, c in plate["holes"]}
    return stamped - match["final"]


def _render(grid, analysis):
    output = grid.copy()
    for plate in analysis["plates"]:
        _paint(output, plate["envelope"], analysis["background"])
    for fork in analysis["forks"]:
        _paint(output, fork["tally"], fork["color"])
    for match in analysis["matches"]:
        fork = analysis["forks"][match["fi"]]
        plate = analysis["plates"][match["pi"]]
        _paint(output, _trace(fork, plate, match), fork["color"])
        _paint(output, match["final"], fork["color"])
        distance, (dr, dc) = match["distance"], fork["out"]
        foreground = {(r - distance * dr, c - distance * dc) for r, c in plate["cells"]}
        _paint(output, foreground, plate["color"])
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all roles and stencil motions from the input alone."""
    grid = _checked_grid(input_grid)
    return _render(grid, _analyze(grid))


def _cavities(rng, height, width, count, mode="random"):
    mask = np.zeros((height, width), dtype=bool)
    if count == 0:
        return mask
    if mode == "diagonal":
        mask[1, 1] = mask[2, 2] = mask[2, 3] = True
        assert count == 2
        return mask
    if count == 1 and (mode == "ring" or rng.random() < 0.35):
        mask[1:-1, 1:-1] = True
        return mask
    if count == 4 and min(height, width) >= 5 and rng.random() < 0.25:
        for r in (1, height - 2):
            for c in (1, width - 2):
                mask[r, c] = True
        return mask
    # Arbitrary isolated seeds, with a finite parity fallback.
    for _ in range(20):
        trial = mask.copy()
        for index in rng.permutation((height - 2) * (width - 2)):
            r, c = divmod(int(index), width - 2)
            r, c = r + 1, c + 1
            trial[r, c] = True
            if len(_components(trial)) != int(trial.sum()) or len(_components(~trial)) != 1:
                trial[r, c] = False
                continue
            if int(trial.sum()) == count:
                mask = trial
                break
        if int(mask.sum()) == count:
            break
    else:
        cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)
                 if (r + c) % 2 == 0]
        for r, c in cells[:count]:
            mask[r, c] = True
        if len(_components(~mask)) != 1:
            raise ValueError("unfeasible cavity capacity")
    target = int(rng.integers(count, (height - 2) * (width - 2) + 1))
    # Grow only through a neighboring cavity, preserving both topologies.
    for _ in range((height - 2) * (width - 2)):
        if int(mask.sum()) >= target:
            break
        candidates = []
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                if mask[r, c] or not any(mask[r + dr, c + dc] for dr, dc in _STEPS):
                    continue
                trial = mask.copy()
                trial[r, c] = True
                if len(_components(trial)) == count and len(_components(~trial)) == 1:
                    candidates.append((r, c))
        if not candidates:
            break
        mask[_choose(rng, candidates)] = True
    assert len(_components(mask)) == count and len(_components(~mask)) == 1
    return mask


def _tally_positions(rng, width, code):
    # Enumerating combinations is tiny (at most four dots in a small base).
    from itertools import combinations
    positions = [p for p in combinations(range(1, width - 1), code)
                 if all(b - a >= 2 for a, b in zip(p, p[1:]))]
    return _choose(rng, positions)


def _module(rng, background, ink, plate_colors, code, direction,
            *, kind="small", active=True):
    """Construct one mechanism; output uses direct interval projection."""
    p = int(rng.integers(5 if code > 2 or kind == "topology" else 4, 9))
    clearance = int(rng.integers(1, 3))
    guard = int(rng.integers(1, 3))
    if kind in ("small", "long_center") and p + 2 * (clearance + guard) > 13:
        clearance = guard = 1
    width = p + 2 * (clearance + guard)
    q0 = clearance + guard
    center = int(rng.integers(1, 4))
    depth = int(rng.integers(4 if code <= 2 else 5, 8))
    gap = int(rng.integers(1, 5))
    if kind == "geometry":
        gap = int(rng.integers(2, 5))
    if kind in ("small", "long_center"):
        center = 3 if kind == "long_center" else int(rng.integers(1, 3))
        depth = int(rng.integers(4 if code <= 2 else 5, 7))
        gap = int(rng.integers(1, max(2, 14 - 3 - center - depth)))
    stop = 3 + center
    side_cap = min(8, stop + depth + gap - 4)
    sides = tuple(map(int, rng.integers(1, side_cap + 1, size=2)))
    if kind == "long_center":
        sides = tuple(map(int, rng.integers(1, center, size=2)))
    plates = []
    if kind == "topology":
        code = 2
        depth = int(rng.integers(5, 8))
        near_depth = int(rng.integers(4, 7))
        near_start = stop + int(rng.integers(1, 3))
        plates.append((near_start, q0, _cavities(rng, near_depth, p, 1, "ring"), plate_colors[1], False))
        start = near_start + near_depth + int(rng.integers(1, 3))
        holes = _cavities(rng, depth, p, 2, "diagonal")
    else:
        start = stop + gap
        holes = _cavities(rng, depth, p, code, "ring" if kind == "geometry" else "random")
    if active:
        plates.append((start, q0, holes, plate_colors[0], True))
    if kind == "geometry":
        wrong_width = p - 2 if p >= 6 else p + 2
        wrong_q0 = q0 + (p - wrong_width) // 2
        wrong_depth = int(rng.integers(4, 6))
        plates.append((start + depth + int(rng.integers(1, 3)), wrong_q0,
                       _cavities(rng, wrong_depth, wrong_width, code, "ring"), plate_colors[1], False))
    height = max(3 + max(sides + (center,)),
                 max((k + mask.shape[0] for k, _, mask, _, _ in plates), default=3))
    low_q = min(0, min((q for _, q, _, _, _ in plates), default=0))
    high_q = max(width, max((q + mask.shape[1] for _, q, mask, _, _ in plates), default=width))
    if height > 28 or high_q - low_q > 28:
        raise ValueError("module too large")
    inp = np.full((height, high_q - low_q), background, dtype=int)
    inp[:3, -low_q:width - low_q] = ink
    for q in _tally_positions(rng, width, code):
        inp[1, q - low_q] = background
    for qa, qb, length in ((0, guard, sides[0]), (q0, q0 + p, center),
                            (width - guard, width, sides[1])):
        inp[3:3 + length, qa - low_q:qb - low_q] = ink
    out = inp.copy()
    out[:3, -low_q:width - low_q] = ink
    # Paint originals only into input. Accepted output is synthesized directly.
    for k, q, mask, color, accepted in plates:
        ph, pw = mask.shape
        inp[k:k + ph, q - low_q:q - low_q + pw] = np.where(mask, background, color)
        if accepted:
            final_end = stop + ph
            for r, c in np.argwhere(mask):
                # One moving hole produces one longitudinal interval.
                a, b = stop + int(r), k + int(r) + 1
                for kk in range(a, b):
                    if not (stop <= kk < final_end and q0 <= q + int(c) < q0 + p):
                        out[kk, q + int(c) - low_q] = ink
            out[stop:final_end, q - low_q:q - low_q + pw] = np.where(mask, ink, color)
    # Mapping canonical downward outward motion to any desired direction.
    rotations = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
    k = rotations[direction]
    return np.rot90(inp, k).copy(), np.rot90(out, k).copy()


def _shelf_pack(shapes):
    """Finite packing fallback; arbitrary margins are added later."""
    n = len(shapes)
    for order in permutations(range(n)):
        for cuts in range(1 << (n - 1)):
            rows, row = [], []
            for i, index in enumerate(order):
                row.append(index)
                if i == n - 1 or cuts & (1 << i):
                    rows.append(row)
                    row = []
            if (any(sum(shapes[j][1] for j in row) + len(row) - 1 > 28 for row in rows)
                    or sum(max(shapes[j][0] for j in row) for row in rows) + len(rows) - 1 > 28):
                continue
            positions, r = {}, 0
            for row in rows:
                c = 0
                for index in row:
                    positions[index] = (r, c)
                    c += shapes[index][1] + 1
                r += max(shapes[j][0] for j in row) + 1
            return [positions[j] for j in range(n)]
    return None


def _pack(rng, shapes):
    fallback = _shelf_pack(shapes)
    if fallback is None:
        raise ValueError("module rectangles cannot fit")
    order = sorted(range(len(shapes)), key=lambda j: -shapes[j][0] * shapes[j][1])
    for _ in range(12):
        blocked = np.zeros((28, 28), dtype=bool)
        positions = {}
        for index in order:
            h, w = shapes[index]
            free = [(r, c) for r in range(29 - h) for c in range(29 - w)
                    if not blocked[r:r + h, c:c + w].any()]
            if not free:
                break
            r, c = _choose(rng, free)
            positions[index] = r, c
            blocked[max(0, r - 1):min(28, r + h + 1), max(0, c - 1):min(28, c + w + 1)] = True
        if len(positions) == len(shapes):
            return [positions[j] for j in range(len(shapes))]
    return fallback


def _place(rng, background, modules):
    positions = _pack(rng, [a.shape for a, _ in modules])
    low_r, low_c = min(r for r, _ in positions), min(c for _, c in positions)
    positions = [(r - low_r, c - low_c) for r, c in positions]
    h = max(r + a.shape[0] for (r, _), (a, _) in zip(positions, modules))
    w = max(c + a.shape[1] for (_, c), (a, _) in zip(positions, modules))
    top = int(rng.integers(1, min(4, 30 - h)))
    bottom = int(rng.integers(1, min(4, 31 - h - top)))
    left = int(rng.integers(1, min(4, 30 - w)))
    right = int(rng.integers(1, min(4, 31 - w - left)))
    inp = np.full((h + top + bottom, w + left + right), background, dtype=int)
    out = inp.copy()
    for (r, c), (a, b) in zip(positions, modules):
        r, c = r + top, c + left
        inp[r:r + a.shape[0], c:c + a.shape[1]] = a
        out[r:r + b.shape[0], c:c + b.shape[1]] = b
    return {"input": inp, "output": out}


def _make_pair(rng, index, split, held_out, reference_out=None, reference_ink=None):
    familiar = [d for d in _DIRECTIONS if d != held_out]
    for _ in range(64):
        background = int(rng.integers(10))
        palette = list(map(int, rng.permutation([c for c in range(10) if c != background])))
        if split == "train" and index == 1 and palette[0] == reference_ink:
            palette[0], palette[1] = palette[1], palette[0]
        inks, plate_colors = palette[:3], palette[3:]
        modules = []
        if split == "train" and index == 0:
            modules.append(_module(rng, background, inks[0], plate_colors, 2,
                                   _choose(rng, familiar), kind="topology"))
        elif split == "train" and index == 1:
            # Two different orientations include a zero-key, trail-free move.
            perpendicular = [d for d in familiar if d[0] * reference_out[0] + d[1] * reference_out[1] == 0]
            first = _choose(rng, perpendicular)
            directions = [first, _choose(rng, [d for d in familiar if d != first])]
            modules.append(_module(rng, background, inks[0], plate_colors, 1,
                                   directions[0], kind="geometry"))
            modules.append(_module(rng, background, inks[1], plate_colors[2:], 0,
                                   directions[1], kind="long_center"))
        else:
            count = int(rng.integers(1, 3)) if split == "train" else int(rng.integers(2, 4))
            for j in range(count):
                code = int(rng.integers(5))
                direction = held_out if split == "test" and j == 0 else _choose(
                    rng, familiar if split == "train" else list(_DIRECTIONS))
                active = j == 0 or rng.random() >= 0.18
                colors = plate_colors[j:] + plate_colors[:j]
                if j and rng.random() < 0.25:
                    first_color = _choose(rng, plate_colors)
                    colors = [first_color] + [c for c in plate_colors if c != first_color]
                ink = _choose(rng, inks[:j]) if j and rng.random() < 0.25 else inks[j]
                modules.append(_module(rng, background, ink, colors, code, direction, active=active))
            if rng.random() < 0.5:
                h, w = int(rng.integers(4, 7)), int(rng.integers(5, 7))
                k = int(rng.integers(3))
                mask = _cavities(rng, h, w, k)
                plate = np.where(mask, background, plate_colors[-1])
                modules.append((plate, np.full_like(plate, background)))
        try:
            pair = _place(rng, background, modules)
            # Detect unintended alignments and cross-module interference.
            if np.array_equal(solve(pair["input"]), pair["output"]):
                return pair
        except ValueError:
            pass
    raise RuntimeError("bounded scene construction exhausted")


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _legal_count(value, lo, hi):
    return (isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
            and lo <= int(value) <= hi)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _legal_count(num_train, 2, 10) or not _legal_count(num_test, 1, 4) or num_train <= num_test:
        raise ValueError("counts require 2..10 train, 1..4 test, and train > test")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    held_out = _choose(rng, list(_DIRECTIONS))
    puzzle, used = {"train": [], "test": []}, set()
    reference_out = reference_ink = None
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            for _ in range(32):
                pair = _make_pair(rng, index, split, held_out, reference_out, reference_ink)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    if split == "train" and index == 0:
                        a = _analyze(pair["input"])
                        fork = a["forks"][a["matches"][0]["fi"]]
                        reference_out, reference_ink = fork["out"], fork["color"]
                    break
            else:
                raise RuntimeError("distinct input construction exhausted")
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _witnesses(analysis):
    topology_witness = geometry_witness = False
    trace_witness = asymmetric = zero_active = central_long = False
    for match in analysis["matches"]:
        fork = analysis["forks"][match["fi"]]
        plate = analysis["plates"][match["pi"]]
        zero_active |= plate["count"] == 0
        trace = _trace(fork, plate, match)
        trace_witness |= bool(trace) and trace != match["route"] - match["final"]
        central_long |= fork["runs"][1][2] > max(fork["runs"][0][2], fork["runs"][2][2])
        r0, r1, c0, c1 = plate["box"]
        mask = np.zeros(plate["shape"], dtype=bool)
        for r, c in plate["holes"]:
            mask[r - r0, c - c0] = True
        asymmetric |= (not np.array_equal(mask, np.flipud(mask))
                       and not np.array_equal(mask, np.fliplr(mask)))
        for other in analysis["plates"]:
            if other is plate:
                continue
            geometric = _candidate(fork, other, analysis["structures"], analysis["shape"], topology=False)
            topology_witness |= bool(geometric is not None
                and geometric["start"] < match["start"] and other["count"] < plate["count"]
                and len(other["holes"]) > len(plate["holes"]) and plate["count8"] < plate["count"])
            count_only = _candidate(fork, other, analysis["structures"], analysis["shape"], span=False)
            geometry_witness |= bool(count_only is not None
                and geometric is None and count_only["start"] > match["start"])
    return {"topology": topology_witness, "geometry": geometry_witness,
            "trace": trace_witness, "asymmetric": asymmetric, "zero": zero_active,
            "central_long": central_long}


def validate_generated_puzzle(puzzle):
    """Validate input-derived semantics and minimum-training counterexamples."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert _legal_count(len(puzzle["train"]), 2, 10)
    assert _legal_count(len(puzzle["test"]), 1, 4)
    assert len(puzzle["train"]) > len(puzzle["test"])
    used, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = (_checked_grid(pair[name]) for name in ("input", "output"))
            assert inp.shape == out.shape and _key(inp) not in used
            used.add(_key(inp))
            a = _analyze(inp)
            a["shape"] = inp.shape
            assert a["matches"] and not np.array_equal(inp, out)
            assert np.array_equal(solve(inp), out), "incorrect stencil, substitution or motion"
            for fork in a["forks"]:
                assert all(int(out[p]) == fork["color"] for p in fork["cells"] | fork["tally"])
                assert all(length > 0 for _, _, length in fork["runs"])
            for plate in a["plates"]:
                assert len(_components(inp[plate["box"][0]:plate["box"][1],
                                           plate["box"][2]:plate["box"][3]] == plate["color"])) == 1
            for match in a["matches"]:
                fork, plate = a["forks"][match["fi"]], a["plates"][match["pi"]]
                assert fork["code"] == plate["count"]
                assert len(match["final"]) == len(plate["envelope"])
                assert _trace(fork, plate, match).isdisjoint(a["structures"])
                assert all(int(out[p]) == fork["color"] for p in _trace(fork, plate, match))
            analyses[split].append(a)
    witnesses = [_witnesses(a) for a in analyses["train"]]
    for name in ("topology", "geometry", "trace", "asymmetric", "zero", "central_long"):
        assert any(w[name] for w in witnesses), "missing training witness: " + name
    nonzero_moves = [(a["forks"][m["fi"]]["out"], a["forks"][m["fi"]]["color"])
                    for a in analyses["train"] for m in a["matches"]
                    if a["plates"][m["pi"]]["count"] > 0]
    assert len({ink for _, ink in nonzero_moves}) >= 2, "fixed literal ink not refuted"
    assert any(d[0] * e[0] + d[1] * e[1] == 0 for d, _ in nonzero_moves for e, _ in nonzero_moves)
    trace_inks = {a["forks"][m["fi"]]["color"] for a in analyses["train"] for m in a["matches"]
                  if _trace(a["forks"][m["fi"]], a["plates"][m["pi"]], m)}
    assert len(trace_inks) >= 2, "two distinct local trail inks must be demonstrated"
    train_dirs = {f["out"] for a in analyses["train"] for f in a["forks"]}
    assert 2 <= len(train_dirs) <= 3
    for a in analyses["test"]:
        assert any(f["out"] not in train_dirs for f in a["forks"])


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def _focused_semantics():
    # A manually drawn member uses guard thickness three and center length five,
    # beyond the sampler's present thickness/center ranges, with an inactive fork
    # and a one-step overlapping motion that leaves no external cavity trace.
    inp = np.zeros((19, 28), dtype=int)
    inp[1:4, 1:13] = 2
    inp[2, 6] = 0
    inp[4:5, 1:4] = 2
    inp[4:9, 5:9] = 2
    inp[4:6, 10:13] = 2
    inp[10:15, 5:9] = 3
    inp[11, 6] = inp[12, 6] = inp[12, 7] = 0
    inp[1:4, 17:25] = 4
    inp[2, 20] = 0
    inp[4:6, 17:18] = 4
    inp[4:5, 19:23] = 4
    inp[4:7, 24:25] = 4
    expected = inp.copy()
    expected[10:15, 5:9] = 0
    expected[2, 6] = 2
    expected[2, 20] = 4
    expected[9:14, 5:9] = 3
    expected[10, 6] = expected[11, 6] = expected[11, 7] = 2
    before = inp.copy()
    assert np.array_equal(solve(inp), expected) and np.array_equal(inp, before)
    a = _analyze(inp)
    assert len(a["matches"]) == 1 and len(a["forks"]) == 2
    m = a["matches"][0]
    assert m["distance"] == 1 and not _trace(a["forks"][m["fi"]], a["plates"][m["pi"]], m)
    for rotation in range(4):
        assert np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(expected, rotation))
    lookup = (np.arange(10) + 3) % 10
    assert np.array_equal(solve(lookup[inp]), lookup[expected])
    malformed = inp.copy()
    malformed[4:9, 5:9] = 0
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("zero-length middle prong accepted")
    ambiguous = np.pad(inp, ((0, 8), (0, 0)))
    ambiguous[18:23, 5:9] = 3
    ambiguous[19, 6] = ambiguous[20, 6] = ambiguous[20, 7] = 0
    try:
        solve(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError("two valid plates for one fork accepted")

    # Explicitly draw counterfactual answers, rather than merely count features.
    puzzle = generate_puzzle(2523, 2, 1)
    ablations = set()
    for pair in puzzle["train"]:
        grid, output = pair["input"], pair["output"]
        a = _analyze(grid)
        a["shape"] = grid.shape
        witness = _witnesses(a)
        for kind in ("topology", "geometry"):
            if not witness[kind]:
                continue
            for index, match in enumerate(a["matches"]):
                fork = a["forks"][match["fi"]]
                for pi, plate in enumerate(a["plates"]):
                    if pi == match["pi"]:
                        continue
                    candidate = _candidate(fork, plate, a["structures"], grid.shape,
                                           topology=kind != "topology", span=kind != "geometry")
                    if candidate is None:
                        continue
                    replacement = dict(candidate, fi=match["fi"], pi=pi)
                    matches = list(a["matches"])
                    matches[index] = replacement
                    assert not np.array_equal(_render(grid, dict(a, matches=matches)), output)
                    ablations.add(kind)
        eight = [m for m in a["matches"]
                 if a["plates"][m["pi"]]["count8"] == a["forks"][m["fi"]]["code"]]
        if len(eight) != len(a["matches"]):
            assert not np.array_equal(_render(grid, dict(a, matches=eight)), output)
            ablations.add("eight_connectivity")
        bounding = output.copy()
        for m in a["matches"]:
            _paint(bounding, m["route"] - m["final"], a["forks"][m["fi"]]["color"])
        if witness["trace"]:
            assert not np.array_equal(bounding, output)
            ablations.add("bounding_trace")
    assert ablations == {"topology", "geometry", "eight_connectivity", "bounding_trace"}
    # A valid solved scene with weaker demonstrations must fail the validator.
    for pair in puzzle["train"]:
        a = _analyze(pair["input"])
        a["shape"] = pair["input"].shape
        if not _witnesses(a)["topology"]:
            continue
        p = a["plates"][a["matches"][0]["pi"]]
        r0, r1, c0, c1 = p["box"]
        box = pair["input"][r0:r1, c0:c1]
        box[:] = p["color"]
        box[1, 1] = box[-2, -2] = a["background"]
        pair["output"] = solve(pair["input"])
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("training with no connectivity/area inversion witness accepted")
    return sorted(ablations)


def self_test():
    """Bounded 42-seed semantic/diversity audit; returns descriptive metrics."""
    shapes, patterns, fork_shapes, distances, codes, directions, backgrounds = (set() for _ in range(7))
    signatures, counts, relations, cavity_areas = set(), set(), set(), set()
    symmetric = asymmetric = inactive = short = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                backgrounds.add(a["background"])
                counts.add(len(a["forks"]))
                inactive += len(a["forks"]) - len(a["matches"])
                for f in a["forks"]:
                    codes.add(f["code"])
                    directions.add(f["out"])
                    fork_shapes.add((f["width"], f["runs"]))
                    l, c, r = [run[2] for run in f["runs"]]
                    relations.add((np.sign(c - l).item(), np.sign(c - r).item()))
                for p in a["plates"]:
                    r0, r1, c0, c1 = p["box"]
                    mask = pair["input"][r0:r1, c0:c1] == a["background"]
                    patterns.add((mask.shape, mask.tobytes()))
                    cavity_areas.update(map(len, _components(mask)))
                    if p["count"]:
                        symmetric += np.array_equal(mask, np.flipud(mask)) and np.array_equal(mask, np.fliplr(mask))
                        asymmetric += not np.array_equal(mask, np.flipud(mask)) and not np.array_equal(mask, np.fliplr(mask))
                for m in a["matches"]:
                    distances.add(m["distance"])
                    short += m["distance"] < m["length"]
    assert len(signatures) == 42 and len(shapes) >= 20
    assert len(patterns) >= 70 and len(fork_shapes) >= 70
    assert codes == set(range(5)) and directions == set(_DIRECTIONS)
    assert counts == {1, 2, 3} and len(backgrounds) >= 8
    assert len(distances) >= 6 and max(cavity_areas) >= 9
    assert symmetric and asymmetric and inactive and short
    assert (1, 1) in relations and (-1, -1) in relations
    assert _puzzle_key(generate_puzzle(123, 2, 1)) == _puzzle_key(generate_puzzle(123, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(900 + nt * 10 + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (2.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = generate_puzzle(11, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("semantic corruption accepted")
    ablations = _focused_semantics()
    return {"seeds": 42, "distinct_puzzles": len(signatures), "grid_shapes": len(shapes),
            "cavity_patterns": len(patterns), "fork_geometries": len(fork_shapes),
            "travel_distances": sorted(distances), "codes": sorted(codes),
            "fork_counts": sorted(counts), "backgrounds": sorted(backgrounds),
            "max_cavity_area": max(cavity_areas), "symmetric_plates": int(symmetric),
            "asymmetric_plates": int(asymmetric), "inactive_forks": inactive,
            "moves_shorter_than_plate": short, "answer_changing_ablations": ablations,
            "manual_extended_member": True, "independent_solver": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
