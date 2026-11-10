"""Self-sizing ribbon reader. All inference uses visible input pixels.

No files, traces, caches, or logs are written by this module. A puzzle contains
at most 14 pairs of 30x30 int8 inputs and 30x1 int8 outputs (<14 KB raw arrays).
Construction retries are bounded and retain only the current candidate.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 885253434, 592418224, 625285811, 136537668, 209970935,
              94224944, 778614062, 169492319, 305876201, 115161986, 350175613,
              691471315, 24531105, 675211983, 547633436, 117691472, 35993786,
              59283130, 977238607, 163694152, 856655753, 711911210, 551638165,
              507291260, 479319786, 608162149, 31377122, 968544596, 299301186,
              557615526, 256534513, 483271652, 317345688, 673188434, 653169583,
              565465617, 898900052, 281511390, 47199588, 217538492, 939165681)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a


def _runs(sequence):
    result = []
    for i, color in enumerate(sequence):
        if i == 0 or color != sequence[i - 1]:
            result.append([i, i + 1, int(color)])
        else:
            result[-1][1] += 1
    return result


def _parse(value):
    a = _grid(value)
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    _require(len(np.unique(border)) == 1, "the outer border must be wall")
    wall = int(border[0])
    colors, counts = np.unique(a[a != wall], return_counts=True)
    _require(len(colors) >= 3, "floor and at least two ink colors are required")
    _require(int(np.sum(counts == counts.max())) == 1, "ambiguous floor color")
    floor = int(colors[int(counts.argmax())])
    allowed = a != wall
    connected = {tuple(np.argwhere(allowed)[0])}
    frontier = list(connected)
    for r, c in frontier:
        for dr, dc in DIRS:
            q = r + dr, c + dc
            if (0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]
                    and allowed[q] and q not in connected):
                connected.add(q)
                frontier.append(q)
    _require(len(connected) == int(allowed.sum()), "disconnected floor region")
    cells = set(map(tuple, np.argwhere((a != wall) & (a != floor))))
    _require(6 <= len(cells) <= 30, "ribbon must contain 6..30 cells")
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = sorted(p for p in cells if len(neighbors[p]) == 1)
    _require(len(ends) == 2 and all(1 <= len(v) <= 2 for v in neighbors.values()),
             "ink must be an induced simple chain")
    _require(ends[0][0] != ends[1][0], "endpoints must have unequal rows")
    previous, current = None, ends[0]
    path, visited = [], set()
    while current is not None:
        _require(current not in visited, "ribbon cycle")
        visited.add(current)
        path.append(current)
        next_cells = [q for q in neighbors[current] if q != previous]
        previous, current = current, next_cells[0] if next_cells else None
    _require(visited == cells, "disconnected ink")
    sequence = [int(a[p]) for p in path]
    runs = _runs(sequence)
    sizes = np.empty(len(path), dtype=int)
    for lo, hi, _ in runs:
        _require(hi - lo in (1, 3, 5, 7), "ink run lengths must be 1, 3, 5, or 7")
        sizes[lo:hi] = hi - lo
    return {"grid": a, "wall": wall, "floor": floor, "path": path,
            "sequence": sequence, "runs": runs, "sizes": sizes, "allowed": allowed}


def _prefix(allowed, path, sizes, shape="square"):
    """Direct footprint enumeration, also used for counterfactual diagnostics."""
    h, w = allowed.shape
    for i, ((r, c), k) in enumerate(zip(path, sizes)):
        radius = int(k) // 2
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if shape == "diamond" and abs(dr) + abs(dc) > radius:
                    continue
                if shape == "cross" and dr != 0 and dc != 0:
                    continue
                if not (0 <= r + dr < h and 0 <= c + dc < w and allowed[r + dr, c + dc]):
                    return i
    return len(path)


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    allowed = info["allowed"]
    # Configuration masks come from the pixels, not a sampled corridor width.
    placements = {}
    for k in set(map(int, info["sizes"])):
        valid = np.zeros(allowed.shape, dtype=bool)
        radius = k // 2
        if min(allowed.shape) >= k:
            windows = np.lib.stride_tricks.sliding_window_view(allowed, (k, k))
            valid[radius:allowed.shape[0] - radius, radius:allowed.shape[1] - radius] = windows.all(axis=(-1, -2))
        placements[k] = valid
    reached = 0
    for p, k in zip(info["path"], info["sizes"]):
        if not placements[int(k)][p]:
            break
        reached += 1
    _require(reached > 0, "reader must fit at the first ink cell")
    return np.asarray(info["sequence"][:reached], dtype=np.int8).reshape(-1, 1)


def _raster_matches(path, sequence):
    color = dict(zip(path, sequence))
    for transpose in (False, True):
        for first in (-1, 1):
            for second in (-1, 1):
                order = sorted(path, key=lambda p: (first * p[int(transpose)], second * p[1 - int(transpose)]))
                if [color[p] for p in order] == list(sequence):
                    return True
    return False


def _walk(rng, length):
    path, used = [(0, 0)], {(0, 0)}
    previous = None
    persistence = float(rng.uniform(0.4, 5.0))
    for _ in range(length - 1):
        choices, weights = [], []
        for dr, dc in DIRS:
            q = path[-1][0] + dr, path[-1][1] + dc
            if q in used or sum((q[0] + a, q[1] + b) in used for a, b in DIRS) != 1:
                continue
            if max(max(r for r, _ in path), q[0]) - min(min(r for r, _ in path), q[0]) >= 20:
                continue
            if max(max(c for _, c in path), q[1]) - min(min(c for _, c in path), q[1]) >= 20:
                continue
            choices.append((q, (dr, dc)))
            weights.append(persistence if previous == (dr, dc) else 1.0)
        if not choices:
            return None
        probs = np.asarray(weights) / sum(weights)
        q, previous = choices[int(rng.choice(len(choices), p=probs))]
        path.append(q)
        used.add(q)
    return _orient(rng, path)


def _orient(rng, path):
    turns, reflection = int(rng.integers(4)), int(rng.choice((-1, 1)))
    path = [(r, c * reflection) for r, c in path]
    for _ in range(turns):
        path = [(c, -r) for r, c in path]
    if path[0][0] == path[-1][0]:
        return None
    return path if path[0][0] < path[-1][0] else path[::-1]


def _fallback_path(rng, length, above=False):
    path, heading = [(2, 0)] if above else [(0, 0)], -1 if above else 1
    if above:
        for (dr, dc), count in (((-1, 0), 2), ((0, 1), 4), ((1, 0), 4)):
            for _ in range(count):
                path.append((path[-1][0] + dr, path[-1][1] + dc))
    while len(path) < length:
        for direction, count in (((0, heading), 4), ((1, 0), 3)):
            for _ in range(min(count, length - len(path))):
                path.append((path[-1][0] + direction[0], path[-1][1] + direction[1]))
        heading *= -1
    if above:
        # This explicit endpoint witness remains above its higher endpoint.
        sign = int(rng.choice((-1, 1)))
        return [(r, c * sign) for r, c in path]
    result = _orient(rng, path)
    return path if result is None else result


def _sizes(rng, length, maximum):
    # Random odd compositions, with a late large run available for a stop.
    for _ in range(64):
        runs, remaining = [], length
        while remaining:
            options = [k for k in (1, 3, 5, 7) if k <= min(maximum, remaining)]
            k = int(rng.choice(options))
            runs.append(k)
            remaining -= k
        if 1 in runs and 3 in runs and maximum in runs and any(k >= 3 and sum(runs[:i]) >= 4 for i, k in enumerate(runs)):
            return runs
    runs = [1, 3, maximum, 1]
    remaining = length - sum(runs)
    while remaining:
        k = min(3, remaining) if remaining % 2 else 1
        runs.append(k)
        remaining -= k
    return runs


def _colors(rng, sizes, palette):
    labels = []
    for _ in sizes:
        options = [c for c in palette if not labels or c != labels[-1]]
        labels.append(int(rng.choice(options)))
    # The same ink must be seen at different run lengths: size is local to
    # the run, not a dictionary attached to a literal color. Three required
    # distinct sizes ensure two-color alternation is a finite fallback.
    if not any(labels[i] == labels[j] and sizes[i] != sizes[j]
               for i in range(len(labels)) for j in range(i + 2, len(labels))):
        labels = [palette[i % 2] for i in range(len(sizes))]
    return [c for k, c in zip(sizes, labels) for _ in range(k)]


def _render_base(rng, path, sequence, run_sizes, wall, floor):
    min_r, min_c = min(r for r, _ in path), min(c for _, c in path)
    path = [(r - min_r + 5, c - min_c + 5) for r, c in path]
    bh, bw = max(r for r, _ in path) + 6, max(c for _, c in path) + 6
    allowed = np.zeros((bh, bw), dtype=bool)
    expanded = [k for k in run_sizes for _ in range(k)]
    for (r, c), k in zip(path, expanded):
        radius = k // 2 + int(rng.choice((0, 0, 0, 1, 2)))
        radius = min(4, radius)
        allowed[r - radius:r + radius + 1, c - radius:c + radius + 1] = True
    # Crop to the actual chamber union and choose asymmetric surrounding walls.
    points = np.argwhere(allowed)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    allowed = allowed[lo[0]:hi[0], lo[1]:hi[1]]
    padding = []
    for dimension in allowed.shape:
        extra = int(rng.integers(2, min(7, 30 - dimension) + 1))
        before = int(rng.integers(1, extra))
        padding.append((before, extra - before))
    allowed = np.pad(allowed, padding)
    path = [(r - int(lo[0]) + padding[0][0], c - int(lo[1]) + padding[1][0]) for r, c in path]
    grid = np.full(allowed.shape, wall, dtype=np.int8)
    grid[allowed] = floor
    for p, c in zip(path, sequence):
        grid[p] = c
    return grid


def _block(rng, original):
    """Carve a notch joined to existing wall, preserving every ink pixel."""
    info = _parse(original)
    path, sizes, allowed = info["path"], info["sizes"], info["allowed"]
    ink = set(path)
    candidates = [(i, dr, dc, offset) for i in range(4, len(path) - 1)
                  if sizes[i] >= 3 for dr, dc in DIRS for offset in range(1, int(sizes[i]) // 2 + 1)]
    for ix in rng.permutation(len(candidates))[:160]:
        i, dr, dc, offset = candidates[int(ix)]
        p = path[i][0] + dr * offset, path[i][1] + dc * offset
        removed = []
        while 0 <= p[0] < allowed.shape[0] and 0 <= p[1] < allowed.shape[1] and allowed[p]:
            if p in ink:
                removed = []
                break
            removed.append(p)
            p = p[0] + dr, p[1] + dc
        if not removed:
            continue
        mask = allowed.copy()
        for p in removed:
            mask[p] = False
        reached = _prefix(mask, path, sizes)
        if not 5 <= reached < len(path) - 1:
            continue
        # Keep a partially read run, teaching that the complete visible run
        # determines size even when only its beginning is emitted.
        if not (info["sequence"][reached] == info["sequence"][reached - 1]):
            continue
        if _prefix(mask, path, sizes, "diamond") <= reached:
            continue
        if not any(_prefix(mask, [path[j]], [sizes[j]]) == 1 for j in range(reached + 1, len(path))):
            continue  # Continuing after the obstruction must be distinguishable.
        grid = original.copy()
        for p in removed:
            grid[p] = info["wall"]
        try:
            _parse(grid)
        except ValueError:
            continue
        return {"input": grid, "output": np.asarray(info["sequence"][:reached], dtype=np.int8).reshape(-1, 1)}
    return None


def _build(rng, length, maximum, wall, floor, palette, witness=False, need_block=False):
    for attempt in range(128):
        path = _walk(rng, length) if attempt < 96 else _fallback_path(rng, length, above=witness)
        if path is None:
            continue
        if witness and min(r for r, _ in path) >= path[0][0]:
            continue
        run_sizes = _sizes(rng, length, maximum)
        sequence = _colors(rng, run_sizes, palette)
        if witness and (sequence == sequence[::-1] or _raster_matches(path, sequence)):
            continue
        grid = _render_base(rng, path, sequence, run_sizes, wall, floor)
        try:
            info = _parse(grid)
        except ValueError:
            continue
        if witness and _prefix(info["allowed"], info["path"], [maximum] * length) == length:
            continue
        pair = {"input": grid, "output": np.asarray(sequence, dtype=np.int8).reshape(-1, 1)}
        blocked = _block(rng, grid) if need_block else None
        if need_block and blocked is None:
            continue
        if witness:
            constant = [int(info["sizes"][0])] * length
            if (_prefix(info["allowed"], info["path"], constant) == length and
                    _prefix(blocked["input"] != wall, info["path"], constant) == len(blocked["output"])):
                continue
        return pair, blocked
    raise RuntimeError("bounded ribbon construction exhausted")


def _counts(num_train, num_test):
    for n, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(not isinstance(n, (bool, np.bool_)) and isinstance(n, (int, np.integer))
                 and low <= n <= high, "illegal example count")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    wall, floor = palette[:2]
    inks = palette[2:2 + int(rng.integers(2, 7))]
    puzzle = {"train": [], "test": []}
    successful, blocked = _build(rng, int(rng.integers(14, 23)), 5, wall, floor, inks,
                                 witness=True, need_block=True)
    puzzle["train"] = [successful, blocked] if rng.integers(2) else [blocked, successful]
    used = {_key(pair["input"]) for pair in puzzle["train"]}
    bigger_runs = bool(rng.integers(2))
    for split, count in (("train", num_train - 2), ("test", num_test)):
        for _ in range(count):
            for _attempt in range(32):
                length = int(rng.integers(14, 23) if split == "train" else rng.integers(24, 31))
                maximum = 7 if split == "test" and bigger_runs else 5
                want_block = bool(rng.integers(2))
                clear, stop = _build(rng, length, maximum, wall, floor, inks, need_block=want_block)
                pair = stop if want_block else clear
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded uniqueness retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing, roles = set(), [], [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            info, output = _parse(pair["input"]), _grid(pair["output"])
            key = _key(info["grid"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            expected_length = _prefix(info["allowed"], info["path"], info["sizes"])
            expected = np.asarray(info["sequence"][:expected_length], dtype=np.int8).reshape(-1, 1)
            _require(expected_length > 0 and np.array_equal(output, expected), "wrong ribbon prefix")
            _require(np.array_equal(solve(info["grid"]), output), "solver disagrees with footprint enumeration")
            info["reached"] = expected_length
            roles.add((info["wall"], info["floor"]))
            (training if split == "train" else testing).append(info)
    _require(len(roles) == 1, "structural palette must be shared within a puzzle")
    successes = [i for i in training if i["reached"] == len(i["path"])]
    stops = [i for i in training if 0 < i["reached"] < len(i["path"])]
    _require(successes and stops, "training needs both complete reading and stopping")
    _require(any(not _raster_matches(i["path"], i["sequence"]) and
                 i["sequence"] != i["sequence"][::-1] and
                 len(_runs(i["sequence"])) > len(set(i["sequence"])) for i in successes),
             "training needs ordered, recurrent, directional evidence")
    _require(any(min(r for r, _ in i["path"]) < i["path"][0][0] for i in successes),
             "training must teach higher endpoint rather than topmost ink")
    _require(any(any(a[2] == b[2] and a[1] - a[0] != b[1] - b[0]
                     for a in i["runs"] for b in i["runs"]) for i in successes),
             "training must distinguish run size from ink identity")
    _require(any(_prefix(i["allowed"], i["path"], [max(i["sizes"])] * len(i["path"])) < len(i["path"])
                 for i in successes), "a constant maximum-size reader must be distinguishable")
    _require(any(_prefix(i["allowed"], i["path"], [i["sizes"][0]] * len(i["path"])) != i["reached"]
                 for i in training), "a constant initial-size reader must be distinguishable")
    _require(any(i["sequence"][i["reached"]] == i["sequence"][i["reached"] - 1] and
                 _prefix(i["allowed"], i["path"], i["sizes"], "diamond") > i["reached"] for i in stops),
             "training must show partial runs and square-corner collisions")
    _require(any(any(_prefix(i["allowed"], [i["path"][j]], [i["sizes"][j]]) == 1
                         for j in range(i["reached"] + 1, len(i["path"]))) for i in stops),
             "training must distinguish stopping from skipping bad centers")
    _require(min(len(i["path"]) for i in testing) > max(len(i["path"]) for i in training),
             "test ribbons must extrapolate in length")


def _fixture():
    sizes, colors = [1, 3, 1, 5, 1, 3, 1], [2, 3, 4, 2, 4, 3, 2]
    path = [(6, 5 + i) for i in range(sum(sizes))]
    sequence = [c for k, c in zip(sizes, colors) for _ in range(k)]
    expanded = [k for k in sizes for _ in range(k)]
    grid = np.zeros((13, 25), dtype=np.int8)
    for (r, c), k in zip(path, expanded):
        radius = k // 2
        grid[r - radius:r + radius + 1, c - radius:c + radius + 1] = 1
    for p, c in zip(path, sequence):
        grid[p] = c
    blocked = grid.copy()
    blocked[:5, 14] = 0
    return np.rot90(grid, -1).copy(), np.rot90(blocked, -1).copy(), sequence


def self_test():
    """Bounded 42-seed coverage, direct fixtures, symmetries and mutations."""
    clear, notch, sequence = _fixture()
    assert solve(clear).ravel().tolist() == sequence
    assert solve(notch).ravel().tolist() == sequence[:7]
    info = _parse(notch)
    assert _prefix(info["allowed"], info["path"], info["sizes"], "diamond") == 9
    split_run = notch.copy()
    split_run[info["path"][8]], split_run[info["path"][9]] = 5, 6
    assert len(solve(split_run)) == len(sequence)  # 5 becomes 3+1+1; walls unchanged.
    translated = np.pad(notch, 1)
    assert solve(translated).ravel().tolist() == sequence[:7]
    permutation = np.asarray([8, 3, 9, 0, 6, 4, 7, 2, 5, 1], dtype=np.int8)
    assert solve(permutation[notch]).ravel().tolist() == permutation[sequence[:7]].tolist()
    # Reflections preserve order; 180 degrees reverses endpoints and can
    # change which obstruction is encountered, so recompute from pixels.
    for grid in (clear, notch):
        for rotation in range(4):
            for reflection in (False, True):
                a = np.rot90(grid, rotation)
                a = np.fliplr(a) if reflection else a
                try:
                    parsed = _parse(a)
                except ValueError:
                    continue  # Horizontal equal-row endpoints are excluded.
                n = _prefix(parsed["allowed"], parsed["path"], parsed["sizes"])
                if n:
                    assert solve(a).ravel().tolist() == parsed["sequence"][:n]
    shapes, structures, geometries, profiles, palettes, turns, outcomes = (set() for _ in range(7))
    pairs = partial = square_witnesses = larger_tests = above = 0
    fingerprints = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _parse(pair["input"])
                n, length = len(pair["output"]), len(i["path"])
                pairs += 1
                shapes.add(i["grid"].shape)
                palettes.add((i["wall"], i["floor"]))
                profiles.add(tuple(hi - lo for lo, hi, _ in i["runs"]))
                roles = {i["wall"]: 0, i["floor"]: 1}
                normalized = np.array([roles.setdefault(int(c), len(roles)) for c in i["grid"].flat], dtype=np.int8)
                structures.add((i["grid"].shape, hashlib.sha256(normalized.tobytes()).hexdigest()))
                origin = i["path"][0]
                geometries.add(tuple((r - origin[0], c - origin[1]) for r, c in i["path"]))
                steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(i["path"], i["path"][1:])]
                turns.add(sum(a != b for a, b in zip(steps, steps[1:])))
                outcomes.add((split, n == length))
                partial += int(n < length and i["sequence"][n] == i["sequence"][n - 1])
                square_witnesses += int(_prefix(i["allowed"], i["path"], i["sizes"], "diamond") > n)
                larger_tests += int(split == "test" and max(i["sizes"]) == 7)
                above += int(min(r for r, _ in i["path"]) < origin[0])
    assert len(fingerprints) == 42 and pairs == 210
    assert len(shapes) >= 50 and len(structures) >= 150 and len(geometries) >= 120
    assert len(profiles) >= 100 and len(palettes) >= 25 and len(turns) >= 9
    assert partial >= 42 and square_witnesses >= 42 and larger_tests >= 8 and above >= 5
    assert outcomes == {("train", True), ("train", False), ("test", True), ("test", False)}
    first = generate_puzzle(77, 2, 1)
    again = generate_puzzle(77, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], again[s]) for k in p)
    generate_puzzle(88, 10, 4)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "pairs": pairs, "input_shapes": len(shapes), "structures": len(structures),
            "oriented_geometries": len(geometries), "run_size_profiles": len(profiles),
            "structural_palettes": len(palettes), "turn_counts": sorted(turns), "partial_runs": partial,
            "square_corner_witnesses": square_witnesses, "size_7_tests": larger_tests,
            "interior_above_start": above, "independent_solver": False}
