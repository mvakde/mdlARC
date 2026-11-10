"""Opposing pulse programs on one folded, induced orthogonal cable.

Only standard-library modules and NumPy are used. No files are read or written.
Generation paints from its explicit construction; solve reconstructs input pixels.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np


GATE = 5
CARRIER = 8
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
WITNESS_PERIODS = ((2, 3), (2, 5), (3, 4), (3, 5), (4, 5), (4, 6))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return a.shape, a.tobytes()


def _runs(sequence):
    result = []
    for color in sequence:
        color = int(color)
        if result and result[-1][0] == color:
            result[-1] = (color, result[-1][1] + 1)
        else:
            result.append((color, 1))
    return result


def _expand(program):
    return [int(c) for c, size in program for _ in range(size)]


def _decode(input_grid):
    a = _grid(input_grid)
    hist = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(hist.argmax())
    if hist[bg] * 2 <= a.size or bg in (GATE, CARRIER):
        raise ValueError("background must be a strict majority, distinct from 5 and 8")
    cells = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    adjacent = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRS
                    if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = sorted(p for p in cells if len(adjacent[p]) == 1)
    if len(ends) != 2 or any(len(q) not in (1, 2) for q in adjacent.values()):
        raise ValueError("foreground must form an unbranched orthogonal chain")
    path, previous, current = [], None, ends[0]
    while current is not None:
        path.append(current)
        choices = [q for q in adjacent[current] if q != previous]
        previous, current = current, choices[0] if choices else None
        if len(path) > len(cells):
            raise ValueError("cyclic cable")
    if len(path) != len(cells):
        raise ValueError("disconnected cable")
    seq = [int(a[p]) for p in path]
    gates = [i for i, c in enumerate(seq) if c == GATE]
    if len(gates) != 2:
        raise ValueError("exactly two input gates are required")
    first, last = gates
    if first == 0 or last == len(path) - 1 or last <= first + 1:
        raise ValueError("each gate needs an outer program and an inner receiving section")
    if (any(c != CARRIER for c in seq[first + 1:last])
            or CARRIER in seq[:first] or CARRIER in seq[last + 1:]):
        raise ValueError("the 8-colored receiving section must be exactly between the gates")
    return {"grid": a, "background": bg, "path": path,
            "gates": (path[first], path[last]), "gate_indices": (first, last),
            "core": path[first + 1:last],
            "programs": (_runs(seq[:first][::-1]), _runs(seq[last + 1:]))}


def _pulse(program, clock, priority="near"):
    firing = [(rank, c, length) for rank, (c, length) in enumerate(program)
              if clock % length == 0]
    if not firing:
        return None
    if priority == "short":
        winner = min(firing, key=lambda x: (x[2], x[0]))
    elif priority == "long":
        winner = min(firing, key=lambda x: (-x[2], x[0]))
    elif priority == "far_program":
        winner = firing[-1]
    else:
        winner = firing[0]
    return winner[1]


def _render(info, alternative="rule"):
    """Cellwise arithmetic; alternatives are used only for teaching audits."""
    output = info["grid"].copy()
    core, programs = info["core"], info["programs"]
    m = len(core)
    if alternative == "merge_colors":
        merged = []
        for program in programs:
            totals = {}
            for color, size in program:
                totals[color] = totals.get(color, 0) + size
            merged.append(list(totals.items()))
        programs = merged
    result = []
    for i, point in enumerate(core):
        distances = [i + 1, m - i]
        straight = [abs(point[0] - gate[0]) + abs(point[1] - gate[1])
                    for gate in info["gates"]]
        clocks = straight if alternative == "manhattan" else distances
        colors = []
        for program, d in zip(programs, clocks):
            clock = d - 1
            if alternative == "endpoint_phase":
                clock += 1 + sum(n for _, n in program)
            if alternative == "tile":
                expanded = _expand(program)
                colors.append(expanded[clock % len(expanded)])
            else:
                colors.append(_pulse(program, clock, alternative))
        metric = straight if alternative in ("manhattan", "straight_priority") else distances
        active = [j for j, c in enumerate(colors) if c is not None]
        color = CARRIER
        if alternative == "half":
            closest = [j for j in range(2) if distances[j] == min(distances)]
            active = [j for j in active if j in closest]
        if len(active) == 1:
            color = colors[active[0]]
        elif len(active) == 2:
            if metric[0] == metric[1]:
                keep = alternative == "no_meeting" or (alternative == "same_color_keeps" and colors[0] == colors[1])
                color = colors[0] if keep else GATE
            else:
                color = colors[int(metric[1] < metric[0])]
        if alternative == "always_midpoint" and distances[0] == distances[1]:
            color = GATE
        result.append(color)
    targets = core
    if alternative == "raster":
        targets = sorted(core)
    elif alternative == "reverse_raster":
        targets = sorted(core, reverse=True)
    for point, color in zip(targets, result):
        output[point] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the result exclusively from this input; either endpoint can be traced first."""
    return _render(_decode(input_grid))


def _transform(path, rotation, reflect):
    result = [(r, -c if reflect else c) for r, c in path]
    for _ in range(rotation):
        result = [(c, -r) for r, c in result]
    top, left = min(r for r, _ in result), min(c for _, c in result)
    return [(r - top, c - left) for r, c in result]


def _random_path(rng, length):
    """Subdivided self-avoiding walks, mixed with native induced walks.

    Subdivision by two leaves a cell between neighboring coarse edges, so a
    self-avoiding coarse walk becomes an induced fine-grid chain. Native walks
    allow odd section lengths and frequent turns as well.
    """
    for _ in range(48):
        spacing = int(rng.choice((1, 2), p=(.30, .70)))
        coarse = [(0, 0)]
        occupied = set(coarse)
        required = math.ceil((length - 1) / spacing)
        extent = 29 // spacing
        minr = maxr = minc = maxc = 0
        last_direction = None
        persistence = float(rng.choice((.8, 1.5, 3.0, 7.0)))
        for _ in range(required):
            options, weights = [], []
            r, c = coarse[-1]
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in occupied:
                    continue
                if max(maxr, q[0]) - min(minr, q[0]) > extent:
                    continue
                if max(maxc, q[1]) - min(minc, q[1]) > extent:
                    continue
                if spacing == 1 and sum((q[0] + a, q[1] + b) in occupied
                                        for a, b in DIRS) != 1:
                    continue
                options.append((q, (dr, dc)))
                weights.append(persistence if (dr, dc) == last_direction else 1.0)
            if not options:
                break
            weights = np.asarray(weights)
            q, last_direction = options[int(rng.choice(len(options), p=weights / weights.sum()))]
            coarse.append(q)
            occupied.add(q)
            minr, maxr = min(minr, q[0]), max(maxr, q[0])
            minc, maxc = min(minc, q[1]), max(maxc, q[1])
        if len(coarse) != required + 1:
            continue
        path = [(0, 0)]
        for a, b in zip(coarse, coarse[1:]):
            for t in range(1, spacing + 1):
                path.append((spacing * a[0] + t * (b[0] - a[0]),
                             spacing * a[1] + t * (b[1] - a[1])))
        return _transform(path[:length], int(rng.integers(4)), bool(rng.integers(2)))
    return None


def _snake(length, width):
    path, direction = [(0, 0)], 1
    while len(path) < length:
        for _ in range(width - 1):
            path.append((path[-1][0], path[-1][1] + direction))
            if len(path) == length:
                return path
        for _ in range(2):
            path.append((path[-1][0] + 1, path[-1][1]))
            if len(path) == length:
                return path
        direction *= -1
    return path


def _construct(rng, path, programs, m, bg):
    """Paint independently from construction data, using far-to-near overwrites."""
    left, right = map(_expand, programs)
    seq = left[::-1] + [GATE] + [CARRIER] * m + [GATE] + right
    assert len(seq) == len(path)
    bh, bw = max(r for r, _ in path) + 1, max(c for _, c in path) + 1
    if bh > 30 or bw > 30:
        return None
    if rng.random() < .75:
        h, w = min(30, bh + int(rng.integers(5))), min(30, bw + int(rng.integers(5)))
    else:
        h, w = int(rng.integers(bh, 31)), int(rng.integers(bw, 31))
    while h * w <= 2 * len(path):
        if (h <= w and h < 30) or w == 30:
            h += 1
        else:
            w += 1
    top, col = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
    path = [(r + top, c + col) for r, c in path]
    inp = np.full((h, w), bg, dtype=np.int8)
    for point, color in zip(path, seq):
        inp[point] = color
    signals = []
    for program in programs:
        signal = np.full(m, -1, dtype=int)
        for color, size in reversed(program):
            signal[::size] = color
        signals.append(signal)
    signal0, signal1 = signals[0], signals[1][::-1]
    out = inp.copy()
    for i, point in enumerate(path[len(left) + 1:len(left) + 1 + m]):
        a, b = int(signal0[i]), int(signal1[i])
        if a < 0 and b < 0:
            continue
        if a < 0:
            color = b
        elif b < 0:
            color = a
        elif 2 * i == m - 1:
            color = GATE
        else:
            color = a if 2 * i < m - 1 else b
        out[point] = color
    return {"input": inp, "output": out}


def _geometry_witness(pair, priority=False):
    info = _decode(pair["input"])
    rivals = ("manhattan", "raster", "reverse_raster")
    if priority:
        rivals += ("straight_priority",)
    return all(not np.array_equal(pair["output"], _render(info, rule))
               for rule in rivals)


def _make_pair(rng, programs, m, bg, require_geometry, require_priority=False):
    length = sum(n for program in programs for _, n in program) + m + 2
    for _ in range(28 if require_geometry else 1):
        path = _random_path(rng, length)
        if path is None:
            break
        pair = _construct(rng, path, programs, m, bg)
        if pair is not None and (not require_geometry or _geometry_witness(pair, require_priority)):
            return pair
    # A finite geometric safety net, generated at the requested length.
    for width in range(8, 25):
        path = _snake(length, width)
        for rotation in range(4):
            pair = _construct(rng, _transform(path, rotation, False), programs, m, bg)
            if pair is not None and (not require_geometry or _geometry_witness(pair, require_priority)):
                return pair
    raise RuntimeError("bounded geometry search exhausted")


def _program(rng, count, maximum, palette, first_size=None):
    colors = []
    for _ in range(count):
        choices = [c for c in palette if not colors or c != colors[-1]]
        colors.append(int(rng.choice(choices)))
    if count >= 3 and rng.random() < .65:
        colors[2] = colors[0]
        if count == 4 and colors[3] == colors[2]:
            colors[3] = int(rng.choice([c for c in palette if c != colors[2]]))
    sizes = [int(rng.integers(1, maximum + 1)) for _ in range(count)]
    if first_size is not None:
        sizes[0] = first_size
    return list(zip(colors, sizes))


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("num_train must be 2..10 and num_test 1..4, both integers")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    a, b = WITNESS_PERIODS[int(rng.integers(len(WITNESS_PERIODS)))]
    common = math.lcm(a, b)
    shift = str(rng.choice(("length", "mixture", "period")))
    shared_bg = int(rng.choice([c for c in range(10) if c not in (GATE, CARRIER)]))
    stable_background = bool(rng.integers(2))
    witness_colors = list(map(int, rng.choice([c for c in range(10)
                                             if c not in (GATE, CARRIER, shared_bg)], 2, replace=False)))
    x, y = witness_colors
    plans = [([(x, a), (y, b), (x, a + 1)], [(y, b), (x, a)], 2 * common + 1),
             ([(y, b), (x, a)], [(x, a), (y, b)], 2 * common + 2)]
    # Witness slots are shuffled; their semantic evidence does not depend on index.
    if rng.random() < .5:
        plans.reverse()
    puzzle, seen = {"train": [], "test": []}, set()
    train_core_lengths = []
    for split, count in (("train", num_train), ("test", num_test)):
        unresolved_ties = set()
        if split == "test":
            for rival in CONDITIONAL_RIVALS:
                if all(np.array_equal(p["output"], _render(_decode(p["input"]), rival)) for p in puzzle["train"]):
                    unresolved_ties.add(rival)
        for index in range(count):
            witness = split == "train" and index < 2
            bg = shared_bg
            if not stable_background and not witness:
                bg = int(rng.choice([c for c in range(10) if c not in (GATE, CARRIER)]))
            palette = [c for c in range(10) if c not in (GATE, CARRIER, bg)]
            if witness:
                p0, p1, m = plans[index]
            else:
                maximum = b if shift == "period" else int(rng.integers(3, 9))
                max_runs = 3 if shift == "mixture" and split == "train" else 4
                counts = [int(rng.integers(1, max_runs + 1)) for _ in range(2)]
                p0 = _program(rng, counts[0], maximum, palette)
                p1 = _program(rng, counts[1], maximum, palette)
                if split == "train":
                    m = int(rng.integers(8, 2 * common + 3))
                elif shift == "length":
                    m = int(rng.integers(max(train_core_lengths) + 1,
                                         min(80, max(train_core_lengths) + 21)))
                elif shift == "mixture":
                    p0 = _program(rng, 4, maximum, palette)
                    m = int(rng.integers(13, max(24, 2 * common + 3)))
                else:
                    novel = int(rng.integers(b + 1, min(10, b + 3) + 1))
                    p0 = _program(rng, counts[0], maximum, palette, first_size=novel)
                    m = int(rng.integers(2 * novel + 3, max(2 * novel + 4, 2 * common + 5)))
            if split == "test" and index == 0:
                # A period-2 signal can be a mere checkerboard because the square
                # lattice preserves distance parity. Expose a longer period with
                # a unique source color, and allow its second pulse on its near half.
                size = p0[0][1] if shift == "period" else int(rng.integers(3, maximum + 1))
                color = int(rng.choice([c for c in palette if c not in {q[0] for q in p0[1:]}]))
                p0[0] = (color, size)
                m = max(m, 2 * size + 3)
                if p0[0][0] == p1[0][0]:
                    excluded = {p0[0][0]} | ({p1[1][0]} if len(p1) > 1 else set())
                    p1[0] = (int(rng.choice([c for c in palette if c not in excluded])), p1[0][1])
            programs = (p0, p1)
            if split == "test" and m % 2:
                candidates = [_pulse(p, (m - 1) // 2) for p in programs]
                both = all(c is not None for c in candidates)
                if (("same_color_keeps" in unresolved_ties and both and candidates[0] == candidates[1])
                        or ("always_midpoint" in unresolved_ties and not both)):
                    # Changing the sampled length avoids testing a distinction
                    # that a minimal curriculum has not yet demonstrated.
                    m += 1
            # Mandatory examples show both intrinsic timing and intrinsic priority.
            geometry = witness or (split == "test" and index == 0)
            for _ in range(8):
                pair = _make_pair(rng, programs, m, bg, geometry,
                                  require_priority=witness and m % 2 == 1)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
            puzzle[split].append(pair)
            seen.add(key)
            if split == "train":
                train_core_lengths.append(m)
    validate_generated_puzzle(puzzle)
    return puzzle


RIVALS = ("short", "long", "far_program", "merge_colors", "half", "tile",
          "manhattan", "straight_priority", "raster", "reverse_raster",
          "endpoint_phase", "no_meeting")
CONDITIONAL_RIVALS = ("same_color_keeps", "always_midpoint")


def _teaching_audit(pairs):
    """Compare whole visible answers, not sampled parameter labels."""
    defeated = set()
    meetings, far_fill, recurrent, short_near, long_near = False, False, False, False, False
    priority_edges = set()
    for pair in pairs:
        info = _decode(pair["input"])
        out, m = np.asarray(pair["output"]), len(info["core"])
        for rival in RIVALS:
            if rival not in defeated and not np.array_equal(out, _render(info, rival)):
                defeated.add(rival)
        meetings |= any(out[p] == GATE for p in info["core"])
        for program in info["programs"]:
            recurrent |= len({c for c, _ in program}) < len(program)
        for i, point in enumerate(info["core"]):
            ds = (i + 1, m - i)
            pulses = [_pulse(program, d - 1) for program, d in zip(info["programs"], ds)]
            if ds[0] != ds[1]:
                near = int(ds[1] < ds[0])
                far = 1 - near
                far_fill |= (pulses[near] is None and pulses[far] is not None
                             and out[point] == pulses[far])
            for side, (program, d) in enumerate(zip(info["programs"], ds)):
                if ds[side] >= ds[1 - side]:
                    continue
                firing = [(rank, c, n) for rank, (c, n) in enumerate(program) if (d - 1) % n == 0]
                if len(firing) < 2 or out[point] != firing[0][1]:
                    continue
                for _, c, n in firing[1:]:
                    if c != firing[0][1]:
                        short_near |= firing[0][2] < n
                        long_near |= firing[0][2] > n
                        priority_edges.add((firing[0][1], c))
    return {"defeated": defeated, "meeting": meetings, "far_fill": far_fill,
            "recurrence": recurrent, "short_near": short_near, "long_near": long_near,
            "color_cycle": any((b, a) in priority_edges for a, b in priority_edges)}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    extents = {s: {"core": 0, "runs": 0, "period": 0} for s in ("train", "test")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            info = _decode(inp)
            extents[split]["core"] = max(extents[split]["core"], len(info["core"]))
            extents[split]["runs"] = max(extents[split]["runs"], max(map(len, info["programs"])))
            extents[split]["period"] = max(extents[split]["period"],
                                           max(n for p in info["programs"] for _, n in p))
            assert np.array_equal(out, _render(info)), "pulse timing, competition, or preservation error"
            mask = np.zeros(inp.shape, dtype=bool)
            for point in info["core"]:
                mask[point] = True
            assert np.array_equal(inp[~mask], out[~mask]), "a source, gate, or background cell changed"
            assert np.array_equal(inp == info["background"], out == info["background"]), "cable topology changed"
            assert not np.array_equal(inp, out)
    audit = _teaching_audit(puzzle["train"])
    assert any(extents["test"][k] > extents["train"][k] for k in extents["train"]), "missing visible train/test shift"
    assert audit["defeated"] == set(RIVALS), "training leaves a documented rival rule unresolved"
    assert all(audit[k] for k in ("meeting", "far_fill", "recurrence", "short_near", "long_near", "color_cycle"))
    assert any(_geometry_witness(p) for p in puzzle["test"]), "test must use actual cable geometry"
    for rival in CONDITIONAL_RIVALS:
        if all(np.array_equal(p["output"], _render(_decode(p["input"]), rival)) for p in puzzle["train"]):
            assert all(np.array_equal(p["output"], _render(_decode(p["input"]), rival)) for p in puzzle["test"]), "untaught midpoint distinction"


def _features(pair):
    info = _decode(pair["input"])
    path, core = info["path"], info["core"]
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    h, w = np.asarray(pair["input"]).shape
    return {"shape": (h, w), "length": len(path), "core_length": len(core),
            "turns": sum(a != b for a, b in zip(steps, steps[1:])),
            "background": info["background"],
            "periods": tuple(n for p in info["programs"] for _, n in p),
            "run_counts": tuple(map(len, info["programs"])),
            "recurrence": any(len({c for c, _ in p}) < len(p) for p in info["programs"]),
            "border": any(r in (0, h - 1) or c in (0, w - 1) for r, c in path),
            "layout": tuple((r - path[0][0], c - path[0][1]) for r, c in path)}


def self_test():
    # A hand-computed example: nearest-first red-2 / green-3 versus green-3 / red-2.
    # M=13 has active equal-distance signals at its center, which becomes a gate.
    program0, program1 = [(2, 2), (3, 3)], [(3, 3), (2, 2)]
    rng = np.random.default_rng(2026)
    fixture = _construct(rng, _transform(_snake(25, 8), 0, False), (program0, program1), 13, 0)
    info = _decode(fixture["input"])
    expected = [2, 8, 2, 3, 2, 8, 5, 8, 2, 3, 2, 8, 3]
    # Trace direction is arbitrary; identify the explicitly constructed first gate.
    actual = [int(fixture["output"][p]) for p in info["core"]]
    assert actual in (expected, expected[::-1])
    assert np.array_equal(solve(fixture["input"]), fixture["output"])
    # Short receiving sections, same-color meetings, one-sided and absent pulses.
    for left, right, m, expected in (
            ([(2, 2)], [(3, 3)], 1, [5]),
            ([(2, 2)], [(2, 3)], 1, [5]),
            ([(2, 2)], [(3, 3)], 3, [2, 8, 3]),
            ([(2, 2)], [(3, 3)], 5, [2, 3, 2, 8, 3]),
            ([(2, 2)], [(3, 3)], 6, [2, 8, 2, 8, 2, 3]),
            ([(2, 1)], [(3, 1)], 2, [2, 3])):
        n = sum(size for p in (left, right) for _, size in p) + m + 2
        fixed = _construct(rng, _transform(_snake(n, 8), 0, False), (left, right), m, 0)
        info = _decode(fixed["input"])
        actual = [int(fixed["output"][p]) for p in info["core"]]
        assert actual in (expected, expected[::-1]), (m, actual, expected)
        assert np.array_equal(solve(fixed["input"]), fixed["output"])
    shapes, layouts, lengths, periods, backgrounds, turns, run_counts, hashes = (set() for _ in range(8))
    recurrence = border = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _features(pair)
                shapes.add(info["shape"])
                layouts.add(info["layout"])
                lengths.add(info["core_length"])
                periods.update(info["periods"])
                backgrounds.add(info["background"])
                turns.add(info["turns"])
                run_counts.update(info["run_counts"])
                recurrence += info["recurrence"]
                border += info["border"]
                digest.update(pair["input"].tobytes())
                assert np.array_equal(solve(pair["input"]), pair["output"])
                # Symmetries preserve gates' roles and intrinsic distances.
                if seed < 3:
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
                        assert np.array_equal(solve(np.fliplr(np.rot90(pair["input"], k))),
                                              np.fliplr(np.rot90(pair["output"], k)))
                    mapping = np.arange(10)
                    symbols = [c for c in range(10) if c not in (GATE, CARRIER)]
                    mapping[symbols] = symbols[1:] + symbols[:1]
                    assert np.array_equal(solve(mapping[pair["input"]]), mapping[pair["output"]])
        hashes.add(digest.digest())
    assert len(hashes) == 42
    assert len(shapes) >= 60 and len(layouts) >= 150
    assert len(lengths) >= 20 and len(turns) >= 15
    assert set(range(1, 9)) <= periods and run_counts == {1, 2, 3, 4}
    assert len(backgrounds) == 8 and recurrence >= 42 and border >= 20
    # Every legal explicit count, deterministic repeats, and invalid scalar types.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(3100 + 10 * a + b, a, b)
    one, two = generate_puzzle(87, 2, 1), generate_puzzle(87, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in one for x, y in zip(one[s], two[s]) for k in x)
    for counts in ((1, 1), (11, 1), (2, 2), (4, 0), (6, 5), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupted output, missing gate, a branch, and disconnected/cyclic foreground.
    wrong = {s: [{k: v.copy() for k, v in p.items()} for p in one[s]] for s in one}
    target = _decode(wrong["test"][0]["input"])["core"][0]
    wrong["test"][0]["output"][target] = CARRIER
    try:
        validate_generated_puzzle(wrong)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    missing = one["train"][0]["input"].copy()
    info = _decode(missing)
    missing[info["gates"][0]] = info["programs"][0][0][0]
    try:
        solve(missing)
    except ValueError:
        pass
    else:
        raise AssertionError("missing gate accepted")
    for coords in (((1, 1), (1, 2), (2, 1), (2, 2)),
                   ((1, 2), (2, 1), (2, 2), (2, 3)),
                   ((1, 1), (2, 1), (4, 4))):
        bad = np.zeros((6, 6), dtype=int)
        for p in coords:
            bad[p] = 2
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid topology accepted")
