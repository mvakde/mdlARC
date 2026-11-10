"""Balanced thread winders: reflected material unrolls into finite pinwheel arms.

Only standard-library modules and NumPy are used. No evidence file is needed at
runtime. Clockwise, unit-cell material conservation and radial symmetry are fixed
family conventions; colors, budgets and geometry are observable in each input.
"""
from collections import Counter
from itertools import permutations
import hashlib
import json
import numpy as np


DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 658749427, 358181115, 754571359, 479850066, 528944128,
              916894191, 879379342, 99372277, 111487771, 532363947,
              516858893, 739094652, 770286127, 762872300, 811545664,
              840647680, 210419211, 881710650, 726320358, 796701582,
              434861039, 616769331, 652227563, 39258308, 457945940,
              993772648, 768155988, 344924667, 367659556, 111943234,
              412584275, 23607088, 395424664, 192497825, 618320225,
              233203839, 274122848, 691776688, 985518786, 906174454,
              675630472)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _cw(v):
    return v[1], -v[0]


def _point(center, direction, d, t):
    v = DIRECTIONS[direction]
    w = _cw(v)
    return center[0] + d*v[0] + t*w[0], center[1] + d*v[1] + t*w[1]


def _sector(y, x):
    if -y > abs(x):
        return 0
    if x > abs(y):
        return 1
    if y > abs(x):
        return 2
    if -x > abs(y):
        return 3
    return None


def _decode(input_grid):
    """Recover the cross and four color-grouped radial reservoirs, input only."""
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    if len(colors) != 6 or int(counts.max()) <= a.size // 2:
        raise ValueError("six roles and a strict-majority background are required")
    bg = int(colors[int(counts.argmax())])
    candidates = []
    for mark in map(int, colors):
        if mark == bg:
            continue
        coords = np.argwhere(a == mark)
        if len(coords) % 4:
            continue
        length = len(coords) // 4
        if length < 1:
            continue
        cy = (int(coords[:, 0].min()) + int(coords[:, 0].max())) // 2
        cx = (int(coords[:, 1].min()) + int(coords[:, 1].max())) // 2
        expected = {_point((cy, cx), arm, d, 0)
                    for arm in range(4) for d in range(1, length+1)}
        if set(map(tuple, coords.tolist())) != expected or a[cy, cx] != bg:
            continue
        reservoirs = {}
        valid = True
        for color in map(int, colors):
            if color in (bg, mark):
                continue
            pts = np.argwhere(a == color)
            sectors = {_sector(int(y)-cy, int(x)-cx) for y, x in pts}
            if len(sectors) != 1 or None in sectors:
                valid = False
                break
            arm = sectors.pop()
            if arm in reservoirs:
                valid = False
                break
            v, w = DIRECTIONS[arm], _cw(DIRECTIONS[arm])
            cells = {((int(y)-cy)*v[0] + (int(x)-cx)*v[1],
                      (int(y)-cy)*w[0] + (int(x)-cx)*w[1]) for y, x in pts}
            if any(d < 2 for d, t in cells):
                valid = False
                break
            core = {(d, t) for d, t in cells if (d, -t) in cells}
            if not core or not any(t != 0 for d, t in core):
                valid = False
                break
            reservoirs[arm] = {"color": color, "cells": cells, "core": core}
        if valid and len(reservoirs) == 4:
            candidates.append((bg, mark, length, (cy, cx), reservoirs))
    if len(candidates) != 1:
        raise ValueError("input must identify one hollow cross and four reservoirs")
    return candidates[0]


def _path(length, budget, arm):
    """Ordered new cells, relative to center, excluding the neutral spoke tip.

    Walk on the infinite plane, counting out-of-window steps as real material.
    The four streams are disjoint rotational phases of parent B's recurrence.
    """
    radial = DIRECTIONS[arm]
    radius = max(2, length)
    y, x = length*radial[0], length*radial[1]
    cells = []

    def move(v, steps):
        nonlocal y, x
        for _ in range(min(steps, budget-len(cells))):
            y, x = y+v[0], x+v[1]
            cells.append((y, x))

    move(radial, radius-length)
    forward = _cw(radial)
    k = 0
    while len(cells) < budget:
        move(forward, 4*k+2)
        move(_cw(forward), radius-2)
        move(forward, radius)
        forward = _cw(forward)
        k += 1
    return cells


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    bg, mark, length, center, reservoirs = _decode(a)
    out = np.full(a.shape, bg, dtype=np.int8)
    out[a == mark] = mark
    for arm, info in reservoirs.items():
        for dy, dx in _path(length, len(info["core"]), arm):
            y, x = center[0]+dy, center[1]+dx
            if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
                if out[y, x] != bg:
                    raise ValueError("winding streams must be disjoint from each other and the cross")
                out[y, x] = info["color"]
    return out


def _support_scores(cells):
    """Convolution check independent of reflected-coordinate construction."""
    m = max(abs(t) for d, t in cells)
    scores = np.zeros(4*m+1, dtype=int)
    for d in sorted({d for d, t in cells}):
        row = np.zeros(2*m+1, dtype=int)
        for d2, t in cells:
            if d2 == d:
                row[t+m] = 1
        scores += np.convolve(row, row)
    return scores, 2*m


def _analytic_prefix(shape, center, length, budget, arm, hand=1):
    """Closed-form stage intervals; no path walker or accumulated endpoints.

    Canonical coordinates put the north spoke at x=0. Four-stage cycle starts
    are explicit. Intervals exclude their first endpoint, so every unit of
    material is counted once. hand=-1 is a diagnostic alternative only.
    """
    yy, xx = np.indices(shape)
    yy, xx = yy-center[0], xx-center[1]
    v, w = DIRECTIONS[arm], _cw(DIRECTIONS[arm])
    y = -(yy*v[0] + xx*v[1])
    x = hand*(yy*w[0] + xx*w[1])
    radius = max(2, length)
    result = np.zeros(shape, dtype=bool)
    initial = radius-length
    if initial and budget:
        result |= (x == 0) & (y == -radius)
    for k in range(budget+1):
        used = initial + 2*k*(k+radius-1)
        if used >= budget:
            break
        q, s = divmod(k, 4)
        sy, sx = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                  (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))[s]
        fy, fx = ((0, 1), (1, 0), (0, -1), (-1, 0))[s]
        jy, jx = fx, -fy
        first, jog = 4*k+2, radius-2
        segments = ((sy, sx, fy, fx, first),
                    (sy+first*fy, sx+first*fx, jy, jx, jog),
                    (sy+first*fy+jog*jy, sx+first*fx+jog*jx, fy, fx, radius))
        for y0, x0, dy, dx, steps in segments:
            take = max(0, min(steps, budget-used))
            along = (y-y0)*dy + (x-x0)*dx
            across = (y-y0)*dx - (x-x0)*dy
            result |= (across == 0) & (along >= 1) & (along <= take)
            used += steps
    return result


def _reference_output(a, decoded=None, mode="supported", hand=1):
    bg, mark, length, center, reservoirs = decoded or _decode(a)
    out = np.full(a.shape, bg, dtype=np.int8)
    out[a == mark] = mark
    for arm, info in reservoirs.items():
        scores, zero = _support_scores(info["cells"])
        budget = len(info["cells"]) if mode == "raw" else int(scores[zero])
        if mode == "pairs_only":
            budget = sum(t != 0 for d, t in info["core"])
        if mode == "half":
            budget //= 2
        if mode == "straight":
            mask = np.zeros(a.shape, dtype=bool)
            for d in range(length+1, length+budget+1):
                y, x = _point(center, arm, d, 0)
                if 0 <= y < a.shape[0] and 0 <= x < a.shape[1]:
                    mask[y, x] = True
        else:
            mask = _analytic_prefix(a.shape, center, length, budget, arm, hand)
        assert np.all(out[mask] == bg), "overlapping winding phases"
        out[mask] = info["color"]
    return out


def _make_core(rng, length, depth, width, target, isolated_axis=False):
    pairs = [(d, t) for d in range(2, depth+1)
             for t in range(1, min(width, d-1)+1)]
    if isolated_axis:
        pairs.remove((depth, 1))
        axis_cells = [(depth, 0)]
        target = min(target, 2*len(pairs)+1)
        target = target if target % 2 else target-1
    else:
        options = [n for n in range(min(3, depth-length)+1)
                   if (target-n) % 2 == 0 and 4 <= target-n <= 2*len(pairs)]
        n = int(rng.choice(options))
        axis_cells = [(int(d), 0) for d in rng.choice(
            np.arange(length+1, depth+1), n, replace=False)]
    n_pairs = (target-len(axis_cells)) // 2
    assert 2 <= n_pairs <= len(pairs)
    family = int(rng.integers(5))
    phase = int(rng.integers(2, depth+1))
    # Rank mirrored pairs using profiles, contours, bands, or separated islands.
    # Exact target population is independent of geometry.
    ranked = []
    for d, t in pairs:
        limit = min(width, d-1)
        if family == 0:
            score = d + .7*t
        elif family == 1:
            score = abs(t-limit)*3 + abs(d-phase)*.25
        elif family == 2:
            score = (d % 2)*4 + t*.2
        elif family == 3:
            score = abs(d-phase)*.8 + abs(t-(1+d % width))*.7
        else:
            score = float(rng.uniform(0, 5))
        ranked.append((score + float(rng.uniform(0, 1.4)), d, t))
    core = set(axis_cells)
    for _, d, t in sorted(ranked)[:n_pairs]:
        core.add((d, t))
        core.add((d, -t))
    assert len(core) == target
    return core


def _noise(rng, core, depth, width, requested):
    candidates = [(d, t) for d in range(2, depth+1)
                  for t in range(-min(width+1, d-1), min(width+1, d-1)+1)
                  if t and (d, t) not in core and (d, -t) not in core]
    rng.shuffle(candidates)
    chosen = set()
    # Favor both attached branches and remote marks without connectivity rules.
    candidates.sort(key=lambda p: (min(abs(p[0]-d)+abs(p[1]-t) for d, t in core)
                                    if rng.random() < .55 else float(rng.uniform(0, 5))))
    for p in candidates:
        if len(chosen) == requested:
            break
        if (p[0], -p[1]) in chosen:
            continue
        scores, zero = _support_scores(core | chosen | {p})
        if int(scores[zero]) == len(core) and np.count_nonzero(scores == scores.max()) == 1:
            if int(scores.argmax()) == zero:
                chosen.add(p)
    if len(chosen) != requested:
        raise RuntimeError("insufficient unambiguous noise positions")
    return chosen


def _sample_pair(rng, split, index, regime, palette):
    is_train = split == "train"
    if is_train:
        length = (2, 3)[index] if index < 2 else int(rng.choice(
            [2, 3] if regime == "scale" else [1, 2, 3, 4]))
    elif regime == "scale":
        length = 4
    else:
        length = int(rng.choice([2, 3]))
    if not is_train and regime == "longer":
        low, high, depth, width = 28, 44, int(rng.integers(8, 10)), 4
    else:
        low, high = (8, 20) if regime == "longer" else (10, 26)
        depth, width = int(rng.integers(6, 9)), int(rng.integers(3, 5))
    depth = max(depth, length+2)
    capacity = 2*sum(min(width, d-1) for d in range(2, depth+1)) + min(3, depth-length)
    high = min(high, capacity-2)
    reservoirs = []
    teaching_total = int(rng.choice([14, 16, 18, 20]))
    teaching_noise = int(rng.choice([3, 5]))
    for arm in range(4):
        target = int(rng.integers(low, high+1))
        isolate = is_train and index < 2 and arm == 0
        teaching_arm = is_train and index == 0 and arm in (0, 1)
        requested = int(rng.integers(1, 7))
        if teaching_arm:
            requested = teaching_noise-2*arm
            target = teaching_total-requested
            isolate = True
        # Every loop has a small fixed bound and discards its candidate in memory.
        for attempt in range(24):
            core = _make_core(rng, length, depth, width, target, isolate)
            try:
                noise = _noise(rng, core, depth, width, requested)
                break
            except RuntimeError:
                if attempt == 15 and not teaching_arm:
                    requested = 1
        else:
            raise RuntimeError("bounded reservoir construction exhausted")
        reservoirs.append((core, noise))
    # Inputs always fit fully. All training continuations, plus one extra cell,
    # also fit, making endpoint counts observable without a clipping assumption.
    points = {_point((0, 0), arm, d, t) for arm, (core, noise) in enumerate(reservoirs)
              for d, t in core | noise}
    points |= {_point((0, 0), arm, d, 0) for arm in range(4) for d in range(1, length+1)}
    # Keep a blank input-only margin sufficient for the nominal reservoir depth.
    points |= {(depth, 0), (-depth, 0), (0, depth), (0, -depth)}
    if is_train:
        for arm, (core, noise) in enumerate(reservoirs):
            points.update(_path(length, len(core)+len(noise)+1, arm))
    miny, maxy = min(y for y, x in points), max(y for y, x in points)
    minx, maxx = min(x for y, x in points), max(x for y, x in points)
    minimum = (maxy-miny+1, maxx-minx+1)
    if is_train and regime == "window":
        minimum = tuple(max(25, n) for n in minimum)
    upper = tuple(min(30, n+4) for n in minimum) if not is_train and regime == "window" else (30, 30)
    h, w = (int(rng.integers(lo, hi+1)) for lo, hi in zip(minimum, upper))
    center = (int(rng.integers(-miny, h-maxy)), int(rng.integers(-minx, w-maxx)))
    bg, mark, *threads = palette
    a = np.full((h, w), bg, dtype=np.int8)
    for arm in range(4):
        for d in range(1, length+1):
            a[_point(center, arm, d, 0)] = mark
        core, noise = reservoirs[arm]
        for d, t in core | noise:
            p = _point(center, arm, d, t)
            assert a[p] == bg
            a[p] = threads[arm]
    if np.count_nonzero(a == bg) <= a.size // 2:
        raise RuntimeError("reservoirs exceed majority-background capacity")
    return {"input": a, "output": solve(a)}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= value <= high:
        raise ValueError(name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, "num_train", 2, 10)
    ntest = _count(num_test, "num_test", 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(["longer", "window", "scale"]))
    puzzle, seen = {"train": [], "test": []}, set()
    first_palette = [int(c) for c in rng.choice(10, 6, replace=False)]
    for split, count in (("train", ntrain), ("test", ntest)):
        for index in range(count):
            palette = [int(c) for c in rng.choice(10, 6, replace=False)]
            if split == "train" and index == 0:
                palette = first_palette
            if split == "train" and index == 1:
                palette = [first_palette[2], first_palette[0], first_palette[1],
                           first_palette[4], first_palette[5], first_palette[3]]
            for attempt in range(24):
                try:
                    pair = _sample_pair(rng, split, index, regime, palette)
                    break
                except RuntimeError:
                    continue
            else:
                raise RuntimeError("bounded pair construction exhausted")
            if _key(pair["input"]) in seen:
                # Finite deterministic collision handling: the five nonbackground
                # roles give 120 distinct recolorings, but a puzzle has <=14 inputs.
                for perm in permutations(palette[1:]):
                    mapping = dict(zip(palette[1:], perm))
                    alternate = {role: np.array([[mapping.get(int(c), int(c)) for c in row]
                                                for row in grid], dtype=np.int8)
                                 for role, grid in pair.items()}
                    if _key(alternate["input"]) not in seen:
                        pair = alternate
                        break
                else:
                    raise AssertionError("finite palette space exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _components(cells):
    remaining, sizes = set(cells), []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            d, t = stack.pop()
            size += 1
            for q in ((d-1, t), (d+1, t), (d, t-1), (d, t+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        sizes.append(size)
    return sizes


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            decoded = _decode(a)
            bg, mark, length, center, reservoirs = decoded
            assert set(map(int, np.unique(b))) == set(map(int, np.unique(a)))
            assert b[center] == bg and np.array_equal(a == mark, b == mark)
            expected = _reference_output(a, decoded)
            assert np.array_equal(b, expected), "wrong symmetry budget or winding prefix"
            assert not np.any((b[:-1, :-1] != bg) & (b[1:, :-1] != bg)
                              & (b[:-1, 1:] != bg) & (b[1:, 1:] != bg)), "filled 2x2 in thread"
            for arm, info in reservoirs.items():
                scores, zero = _support_scores(info["cells"])
                assert int(scores.argmax()) == zero and np.count_nonzero(scores == scores.max()) == 1
                budget = int(scores[zero])
                assert budget == len(info["core"]) >= 4
                assert any(t for d, t in info["core"])
                stream = _path(length, budget, arm)
                assert len(set(stream)) == budget
                assert np.count_nonzero(b == info["color"]) <= budget
                if split == "train":
                    assert np.count_nonzero(b == info["color"]) == budget, "training must expose full budgets"
            if split == "train":
                training.append((a, b, decoded))
    # Observable counterexamples, checked even for the minimum two demonstrations.
    assert any(not np.array_equal(b, _reference_output(a, dec, mode="raw")) for a, b, dec in training)
    assert any(not np.array_equal(b, _reference_output(a, dec, hand=-1)) for a, b, dec in training)
    assert any(not np.array_equal(b, _reference_output(a, dec, mode="straight")) for a, b, dec in training)
    assert any(not np.array_equal(b, _reference_output(a, dec, mode="half")) for a, b, dec in training)
    assert any(not np.array_equal(b, _reference_output(a, dec, mode="pairs_only")) for a, b, dec in training)
    assert any(len(info["core"]) > max(_components(info["core"]))
               for a, b, dec in training for info in dec[4].values())
    assert {dec[2] for a, b, dec in training} >= {2, 3}
    assert len({dec[0] for a, b, dec in training}) >= 2
    # The first demonstration deliberately equalizes two raw populations.
    first = training[0][2][4]
    assert len(first[0]["cells"]) == len(first[1]["cells"])
    assert len(first[1]["core"])-len(first[0]["core"]) == 2
    assert len(first[0]["core"]) % 2 == 1


def measure_puzzle(puzzle):
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a, b = np.asarray(pair["input"]), np.asarray(pair["output"])
            bg, mark, length, center, reservoirs = _decode(a)
            arms = []
            for arm, info in sorted(reservoirs.items()):
                budget = len(info["core"])
                path = _path(length, budget, arm)
                visible = [0 <= center[0]+y < a.shape[0] and 0 <= center[1]+x < a.shape[1] for y, x in path]
                left = False
                reentry = False
                for yes in visible:
                    reentry |= left and yes
                    left |= not yes
                arms.append({"direction": arm, "color": info["color"], "budget": budget,
                             "noise": len(info["cells"])-budget,
                             "axis_cells": sum(t == 0 for d, t in info["core"]),
                             "components": len(_components(info["core"])),
                             "depth": max(d for d, t in info["cells"]),
                             "width": max(abs(t) for d, t in info["cells"]),
                             "visible": int(sum(visible)), "clipped": not all(visible),
                             "reentry": bool(reentry)})
            records.append({"split": split, "shape": list(a.shape), "background": bg,
                            "cross_color": mark, "length": length, "center": list(center), "arms": arms})
    return records


def self_test():
    """42 supplied seeds, boundary counts, mutations, and analytic cross-checks.

    Streams one puzzle at a time; performs no disk I/O and keeps no trace payloads.
    Returns small, JSON-compatible aggregate measurements for review.
    """
    shapes, backgrounds, marks, lengths, budgets, silhouettes, signatures = (set() for _ in range(7))
    totals = Counter()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                assert np.array_equal(solve(a.copy()), b)
                signature.update(bytes(a.shape) + a.tobytes() + b.tobytes())
                bg, mark, length, center, reservoirs = _decode(a)
                shapes.add(a.shape)
                backgrounds.add(bg)
                marks.add(mark)
                lengths.add(length)
                for info in reservoirs.values():
                    budgets.add(len(info["core"]))
                    silhouettes.add(tuple(sorted(info["core"])))
        signatures.add(signature.digest())
        records = measure_puzzle(puzzle)
        for r in records:
            totals[r["split"]+"_pairs"] += 1
            totals["rectangles"] += int(r["shape"][0] != r["shape"][1])
            totals["off_center"] += int(any(abs(2*p-n+1) > 1 for p, n in zip(r["center"], r["shape"])))
            for arm in r["arms"]:
                totals["reservoirs"] += 1
                totals["noise_cells"] += arm["noise"]
                totals["axis_cells"] += arm["axis_cells"]
                totals["disconnected_reservoirs"] += int(arm["components"] > 1)
                totals["clipped_arms"] += int(arm["clipped"])
                totals["reentering_arms"] += int(arm["reentry"])
        if seed == 0:
            other = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], other[s])
                       for k in ("input", "output"))
            for s in puzzle:
                for pair in puzzle[s]:
                    a, b = pair["input"], pair["output"]
                    assert np.array_equal(solve(np.rot90(a)), np.rot90(b))
                    remap = np.array([7, 4, 9, 2, 0, 8, 3, 1, 6, 5], dtype=np.int8)
                    assert np.array_equal(solve(remap[a]), remap[b])
    assert len(signatures) == 42 and len(shapes) >= 35
    assert backgrounds == marks == set(range(10))
    assert {1, 2, 3, 4} <= lengths and len(budgets) >= 20
    assert len(silhouettes) >= 150 and totals["clipped_arms"] > 0
    assert totals["disconnected_reservoirs"] > 100
    # Cross-check all radii, stream lengths, orientations, and clipping windows.
    for length in range(1, 7):
        paths = [_path(length, 100, arm) for arm in range(4)]
        assert all(len(set(p)) == 100 for p in paths)
        assert len(set().union(*map(set, paths))) == 400
        for arm, stream in enumerate(paths):
            for budget in (1, 2, 7, 19, 43, 100):
                for shape, center in (((30, 30), (14, 15)), ((13, 25), (4, 9))):
                    walked = np.zeros(shape, dtype=bool)
                    for y, x in stream[:budget]:
                        y, x = y+center[0], x+center[1]
                        if 0 <= y < shape[0] and 0 <= x < shape[1]:
                            walked[y, x] = True
                    assert np.array_equal(walked, _analytic_prefix(shape, center, length, budget, arm))
    # A legal, unsampled larger reservoir protects re-entry and budgets >44.
    # North consumes 125 cells; step 96 exits a 29x29 window and step 123
    # returns at (28,0). Stopping at the first exit loses that visible cell.
    fixture = np.zeros((29, 29), dtype=np.int8)
    center = (14, 14)
    for arm in range(4):
        for d in range(1, 4):
            fixture[_point(center, arm, d, 0)] = 1
        if arm == 0:
            pairs = [(d, t) for d in range(2, 15) for t in range(1, d)][:62]
            material = {(d, sign*t) for d, t in pairs for sign in (-1, 1)} | {(13, 0)}
            material.add((14, 1))  # unmatched noise, not material
        else:
            material = {(3, -1), (3, 1), (5, -1), (5, 1)}
        for d, t in material:
            fixture[_point(center, arm, d, t)] = arm+2
    decoded = _decode(fixture)
    assert len(decoded[4][0]["core"]) == 125
    answer = solve(fixture)
    assert np.array_equal(answer, _reference_output(fixture, decoded))
    assert answer[28, 0] == 2
    path = _path(3, 125, 0)
    first_exit = next(i for i, (y, x) in enumerate(path)
                      if not (0 <= y+14 < 29 and 0 <= x+14 < 29))
    assert any(0 <= y+14 < 29 and 0 <= x+14 < 29 for y, x in path[first_exit+1:])
    # All legal counts, including every maximum requested split, stay bounded.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            generate_puzzle(10000+nt*10+nq, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, False), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for mutation in ("raw", "reverse", "pixel"):
        p = generate_puzzle(0, 2, 1)
        pair = p["test"][0]
        if mutation == "raw":
            pair["output"] = _reference_output(pair["input"], mode="raw")
        elif mutation == "reverse":
            pair["output"] = _reference_output(pair["input"], hand=-1)
        else:
            pair["output"][0, 0] = (int(pair["output"][0, 0])+1) % 10
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted wrong material or path")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "backgrounds": sorted(backgrounds), "cross_colors": sorted(marks),
            "spoke_lengths": sorted(lengths), "thread_budgets": sorted(budgets),
            "supported_silhouettes": len(silhouettes), **dict(totals)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
