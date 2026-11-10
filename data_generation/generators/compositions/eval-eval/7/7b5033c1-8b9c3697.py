"""A folded ribbon is consumed into width-matched, outward-growing fork queues.

Only NumPy and the standard library are used. Construction paints its own
labels; solve reconstructs topology and receiver geometry from input pixels.
"""

from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer) and
             np.all((grid >= 0) & (grid <= 9)), "grid colors must be integers 0..9")
    return grid


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                point = r + dr, c + dc
                if point in unseen:
                    unseen.remove(point)
                    component.add(point)
                    stack.append(point)
        result.append(component)
    return result


def _trace(cells):
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in DIRS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    ends = sorted(p for p in cells if len(neighbors[p]) == 1)
    if len(ends) != 2 or any(len(v) not in (1, 2) for v in neighbors.values()):
        return None
    if ends[0][0] == ends[1][0]:
        return None
    path, previous, current = [], None, ends[0]
    while current is not None:
        if current in path:
            return None
        path.append(current)
        choices = [p for p in neighbors[current] if p != previous]
        previous, current = current, choices[0] if choices else None
    return path if len(path) == len(cells) else None


def _fork(cells):
    """Infer base direction and positive rectangular prongs without templates."""
    parses = []
    for outward in DIRS:
        perp = (-outward[1], outward[0])
        kmin = min(r * outward[0] + c * outward[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * outward[0] + c * outward[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for _, q in local) + 1
        if {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum(k > 0 and qq == q for k, qq in local) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, length in enumerate(lengths)
                     for k in range(1, length + 1)}
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
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        q0, q1, depth = runs[1]
        if q0 + q1 != width:
            continue
        parses.append({"out": outward, "perp": perp, "kmin": kmin, "qmin": qmin,
                       "span": q1 - q0, "q0": q0, "stop": depth + 1,
                       "width": width, "lengths": tuple(x[2] for x in runs),
                       "cells": cells})
    _require(len(parses) == 1, "receiver must have one base and three positive prongs")
    return parses[0]


def _cell(fork, k, q):
    k, q = k + fork["kmin"], q + fork["qmin"]
    return (k * fork["out"][0] + q * fork["perp"][0],
            k * fork["out"][1] + q * fork["perp"][1])


def _runs(sequence):
    result = []
    for color in sequence:
        color = int(color)
        if result and result[-1][0] == color:
            result[-1] = color, result[-1][1] + 1
        else:
            result.append((color, 1))
    return result


def _signature(sequence, keys):
    return tuple((s, tuple(c for c, n in _runs(sequence) if n == s)) for s in sorted(keys))


def _global_color_signature(sequence, keys):
    grouped = [c for c in dict.fromkeys(sequence) for _ in range(sequence.count(c))]
    return _signature(grouped, keys)


def _spatial_orders(path):
    orders = [sorted(path)]
    cols = sorted({c for _, c in path})
    for columns in (cols, cols[::-1]):
        for reverse_last in (False, True):
            orders.append([p for c in columns for p in sorted(
                (p for p in path if p[1] == c), reverse=reverse_last and c == columns[-1])])
    return orders


def _order_witness(path, sequence, keys):
    colors = dict(zip(path, sequence))
    actual = _signature(sequence, keys)
    return (actual != _signature(sequence[::-1], keys) and
            all(actual != _signature([colors[p] for p in order], keys)
                for order in _spatial_orders(path)))


def _analyze(input_grid):
    grid = _grid(input_grid)
    hist = np.bincount(grid.ravel().astype(int), minlength=10)
    background = int(hist.argmax())
    _require(hist[background] > grid.size / 2, "background must have strict majority")
    cells = {tuple(map(int, p)) for p in np.argwhere(grid != background)}
    paths, forks = [], []
    for component in _components(cells):
        path = _trace(component)
        if path is not None:
            paths.append(path)
        else:
            _require(len({int(grid[p]) for p in component}) == 1,
                     "each receiver must be monochromatic")
            fork = _fork(component)
            fork["color"] = int(grid[min(component)])
            forks.append(fork)
    _require(len(paths) == 1 and forks, "one ribbon and at least one receiver required")
    path = paths[0]
    sequence = [int(grid[p]) for p in path]
    _require(len(set(sequence)) >= 2, "ribbon must be multicolored")
    _require(set(sequence).isdisjoint(f["color"] for f in forks),
             "ribbon and receiver colors must be distinct roles")
    keys = [f["span"] for f in forks]
    _require(len(keys) == len(set(keys)), "receiver widths must be unique")
    runs = _runs(sequence)
    destinations, taken = {}, set()
    for fork in forks:
        queue = [c for c, n in runs if n == fork["span"]]
        bars = []
        for i, color in enumerate(queue):
            bar = {_cell(fork, fork["stop"] + i, q)
                   for q in range(fork["q0"], fork["q0"] + fork["span"])}
            _require(all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]
                         and int(grid[r, c]) == background for r, c in bar),
                     "queue must fit in clear background")
            _require(taken.isdisjoint(bar), "receiver queues must not overlap")
            taken.update(bar)
            bars.append((color, bar))
        destinations[fork["span"]] = bars
    return {"grid": grid, "background": background, "path": path,
            "sequence": sequence, "runs": runs, "forks": forks,
            "keys": set(keys), "destinations": destinations}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from visible pixels; no generation parameters are consulted."""
    info = _analyze(input_grid)
    output = info["grid"].copy()
    for point in info["path"]:
        output[point] = info["background"]
    for bars in info["destinations"].values():
        for color, bar in bars:
            for point in bar:
                output[point] = color
    return output


def _walk(rng, length, persistence):
    path, occupied, previous = [(0, 0)], {(0, 0)}, None
    for _ in range(length - 1):
        choices, weights = [], []
        for dr, dc in DIRS:
            point = path[-1][0] + dr, path[-1][1] + dc
            if point in occupied or sum((point[0] + a, point[1] + b) in occupied
                                        for a, b in DIRS) != 1:
                continue
            if max([p[0] for p in path] + [point[0]]) - min([p[0] for p in path] + [point[0]]) >= 13:
                continue
            if max([p[1] for p in path] + [point[1]]) - min([p[1] for p in path] + [point[1]]) >= 13:
                continue
            choices.append((point, (dr, dc)))
            weights.append(persistence if (dr, dc) == previous else 1.0)
        if not choices:
            return None
        weights = np.asarray(weights, dtype=float)
        point, previous = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        path.append(point)
        occupied.add(point)
    return path


def _normalize(path):
    r0, c0 = min(r for r, _ in path), min(c for _, c in path)
    return [(r - r0, c - c0) for r, c in path]


def _fallbacks(length):
    """Induced serpents with an upper excursion, not the main shape vocabulary."""
    for lane in range(6, 12):
        for depth in range(3, 7):
            path = [(2, 0)]
            segments = [((-1, 0), 2), ((0, 1), lane), ((1, 0), depth)]
            for row in range(depth, 13, 2):
                segments.append(((0, -1 if (row - depth) % 4 == 0 else 1), lane))
                if row + 2 <= 12:
                    segments.append(((1, 0), 2))
            for (dr, dc), count in segments:
                for _ in range(count):
                    path.append((path[-1][0] + dr, path[-1][1] + dc))
            if len(path) < length:
                continue
            for flip in (1, -1):
                candidate = [(r, flip * c) for r, c in path[:length]]
                traced = _trace(set(candidate))
                if traced is not None:
                    yield traced


def _make_path(rng, sequence, keys, witness=False, above=False):
    if witness and _signature(sequence, keys) == _signature(sequence[::-1], keys):
        raise RuntimeError("packet colors do not distinguish traversal direction")
    persistence = float(rng.choice((0.5, 1.0, 3.0, 8.0)))
    for _ in range(260):
        path = _walk(rng, len(sequence), persistence)
        if path is None:
            continue
        path = _trace(set(path))
        if path is None:
            continue
        if above and min(r for r, _ in path) >= path[0][0]:
            continue
        if witness and not _order_witness(path, sequence, keys):
            continue
        return _normalize(path)
    for path in _fallbacks(len(sequence)):
        if above and min(r for r, _ in path) >= path[0][0]:
            continue
        if witness and not _order_witness(path, sequence, keys):
            continue
        return _normalize(path)
    raise RuntimeError("could not construct a distinguishing ribbon")


def _module(rng, span, colors, color, outward, relation):
    options = [(t, g, tail) for t in (1, 2) for g in (1, 2) for tail in (0, 1)
               if span + 2 * (t + g + tail) <= 13]
    thickness, gap, tail = options[int(rng.integers(len(options)))]
    width = span + 2 * (thickness + gap + tail)
    q0 = (width - span) // 2
    if relation == "short":
        center = int(rng.integers(1, 3))
        left, right = map(int, rng.integers(center + 1, 7, size=2))
    elif relation == "long":
        center = int(rng.integers(3, 5))
        left, right = map(int, rng.integers(1, center, size=2))
    else:
        center = int(rng.integers(1, 5))
        left, right = map(int, rng.integers(1, 7, size=2))
    structure = {(0, q) for q in range(width)}
    for a, b, depth in ((tail, tail + thickness, left), (q0, q0 + span, center),
                         (width - tail - thickness, width - tail, right)):
        structure |= {(k, q) for k in range(1, depth + 1) for q in range(a, b)}
    bars = [(c, {(center + 1 + i, q) for q in range(q0, q0 + span)})
            for i, c in enumerate(colors)]
    reserved = structure | set().union(*(p for _, p in bars)) if bars else structure
    perp = (-outward[1], outward[0])

    def raw(points):
        return {(k * outward[0] + q * perp[0], k * outward[1] + q * perp[1]) for k, q in points}

    all_raw = raw(reserved)
    r0, c0 = min(r for r, _ in all_raw), min(c for _, c in all_raw)

    def normalized(points):
        return {(r - r0, c - c0) for r, c in raw(points)}

    reserved = normalized(reserved)
    return {"input": [(color, normalized(structure))],
            "output": [(color, normalized(structure))] + [(c, normalized(p)) for c, p in bars],
            "shape": (max(r for r, _ in reserved) + 1, max(c for _, c in reserved) + 1)}


def _pack(rng, modules, background):
    """Free rectangle packing, with a guaranteed <=27x27 two-shelf fallback."""
    placement = None
    for attempt in range(16):
        h, w = map(int, rng.integers(24, 31, size=2)) if attempt < 12 else (30, 30)
        occupied = np.zeros((h, w), dtype=bool)
        offsets = {}
        order = sorted(range(len(modules)), key=lambda i: -np.prod(modules[i]["shape"]))
        if attempt % 2:
            order = list(map(int, rng.permutation(len(modules))))
        for i in order:
            bh, bw = modules[i]["shape"]
            candidates = [(r, c) for r in range(h - bh + 1) for c in range(w - bw + 1)
                          if not occupied[r:r + bh, c:c + bw].any()]
            if not candidates:
                break
            r, c = candidates[int(rng.integers(len(candidates)))]
            offsets[i] = r, c
            occupied[max(0, r - 1):min(h, r + bh + 1),
                     max(0, c - 1):min(w, c + bw + 1)] = True
        if len(offsets) == len(modules):
            placement = offsets
            break
    if placement is None:
        order = list(map(int, rng.permutation(len(modules))))
        placement, top = {}, 0
        for start in range(0, len(order), 2):
            row = order[start:start + 2]
            left = 0
            for i in row:
                placement[i] = top, left
                left += modules[i]["shape"][1] + 1
            top += max(modules[i]["shape"][0] for i in row) + 1
    min_r, min_c = min(r for r, _ in placement.values()), min(c for _, c in placement.values())
    placement = {i: (r - min_r, c - min_c) for i, (r, c) in placement.items()}
    bh = max(r + modules[i]["shape"][0] for i, (r, c) in placement.items())
    bw = max(c + modules[i]["shape"][1] for i, (r, c) in placement.items())
    h, w = int(rng.integers(bh, 31)), int(rng.integers(bw, 31))
    foreground = sum(len(points) for m in modules for _, points in m["input"])
    while h * w <= 2 * foreground:
        if (h <= w and h < 30) or w == 30:
            h += 1
        else:
            w += 1
    top, left = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
    grids = {role: np.full((h, w), background, dtype=np.int8) for role in ("input", "output")}
    for i, module in enumerate(modules):
        r0, c0 = placement[i]
        for role in grids:
            for color, points in module[role]:
                for r, c in points:
                    grids[role][r0 + top + r, c0 + left + c] = color
    return grids


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def _sample_pair(rng, params, split, index, background):
    a, b, u = params["widths"]
    active_keys = [a, b, u] if split == "test" and params["mode"] == "activation" else [a, b]
    if split == "train" and index >= 2 and rng.random() < 0.35:
        active_keys = [int(rng.choice((a, b)))]
    palette = list(map(int, rng.permutation([c for c in range(10) if c != background])))
    gate_colors = palette[:len(active_keys)]
    cargo_colors = palette[len(active_keys):len(active_keys) + int(rng.integers(3, 5))]
    if split == "train" and index == 0:
        lengths = [a, b, a, u, b, a]
        ci = [0, 1, 2, 0, 2, 1]
        while sum(lengths) < 18:
            lengths.append(u)
            ci.append((ci[-1] + 1) % len(cargo_colors))
        colors = [cargo_colors[i] for i in ci]
    else:
        if split == "train":
            lengths = [a] * int(rng.integers(1, 4)) + [b] * int(rng.integers(1, 4))
            lengths += [u] * int(rng.integers(1, 3))
            if index >= 2 and len(active_keys) == 2 and rng.random() < 0.3:
                # An inactive receiver is geometric context, not an exception.
                empty_key = int(rng.choice((a, b)))
                lengths = [n for n in lengths if n != empty_key]
        elif params["mode"] == "activation":
            lengths = [s for s in active_keys for _ in range(int(rng.integers(1, 4)))]
            rejected = int(rng.choice([s for s in range(1, 6) if s not in active_keys]))
            lengths += [rejected] * int(rng.integers(1, 3))
        else:
            heavy, other = (a, b) if int(rng.integers(2)) else (b, a)
            lengths = [heavy] * int(rng.integers(4, 7)) + [other] * int(rng.integers(1, 4)) + [u]
        rng.shuffle(lengths)
        colors = []
        for _ in lengths:
            colors.append(int(rng.choice([c for c in cargo_colors if not colors or c != colors[-1]])))
    sequence = [c for c, n in zip(colors, lengths) for _ in range(n)]
    witness = index == 0
    path = _make_path(rng, sequence, active_keys, witness, split == "train" and index == 0)
    modules = [{"input": [(c, {p}) for p, c in zip(path, sequence)], "output": [],
                "shape": (max(r for r, _ in path) + 1, max(c for _, c in path) + 1)}]
    directions = params["directions"][:2] if index == 0 else params["directions"][2:]
    if split == "test" or index > 1:
        directions = [DIRS[int(rng.integers(4))] for _ in active_keys]
    for i, (s, color) in enumerate(zip(active_keys, gate_colors)):
        queue = [c for c, n in zip(colors, lengths) if n == s]
        relation = ("short" if i == 0 else "long") if split == "train" and index < 2 else "free"
        modules.append(_module(rng, s, queue, color, directions[i], relation))
    return _pack(rng, modules, background)


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _counts(num_train, num_test)
    params = {"widths": list(map(int, rng.permutation(np.arange(1, 5))[:3])),
              "mode": str(rng.choice(("activation", "extension"))),
              "directions": [DIRS[int(i)] for i in rng.permutation(4)]}
    shared_bg = int(rng.integers(10)) if rng.random() < 0.35 else None
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            background = shared_bg if shared_bg is not None else int(rng.integers(10))
            for _ in range(48):
                try:
                    pair = _sample_pair(rng, params, split, index, background)
                except RuntimeError:
                    continue
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded generation could not produce a distinct example")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "exact train/test keys required")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal counts")
    infos, seen = {"train": [], "test": []}, set()
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "exact pair keys required")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(grid) not in seen, "duplicate input")
            seen.add(_key(grid))
            info = _analyze(grid)
            _require(grid.shape == output.shape and np.array_equal(solve(grid), output), "incorrect output")
            _require(not np.array_equal(grid, output), "identity example")
            _require(all(int(output[p]) == info["background"] for p in info["path"]), "ribbon not consumed")
            for fork in info["forks"]:
                _require(all(int(output[p]) == fork["color"] for p in fork["cells"]), "receiver changed")
            expected_counts = Counter()
            for color, length in info["runs"]:
                if length in info["keys"]:
                    expected_counts[color] += length
            actual_counts = Counter(int(output[p]) for bars in info["destinations"].values() for _, bar in bars for p in bar)
            _require(expected_counts == actual_counts, "accepted packet cells/colors not conserved")
            infos[split].append(info)
    train, test = infos["train"], infos["test"]
    _require(any(_order_witness(i["path"], i["sequence"], i["keys"]) for i in train), "training must defeat spatial/reverse shortcuts")
    _require(_order_witness(test[0]["path"], test[0]["sequence"], test[0]["keys"]), "test must require path order")
    _require(any(min(r for r, _ in i["path"]) < i["path"][0][0] and
                 _signature(i["sequence"], i["keys"]) != _signature(i["sequence"][::-1], i["keys"])
                 for i in train), "higher endpoint must be taught above interior geometry")
    _require(any(n not in i["keys"] for i in train for _, n in i["runs"]), "training must teach rejection")
    _require(any(len({c for c, n in i["runs"] if n == s}) >= 2 and
                 len([c for c, n in i["runs"] if n == s]) >= 3
                 for i in train for s in i["keys"]), "training must teach ordered accumulation")
    _require(any(len([c for c, _ in i["runs"]]) > len({c for c, _ in i["runs"]}) for i in train), "color recurrence missing")
    _require(any(_signature(i["sequence"], i["keys"]) !=
                 _global_color_signature(i["sequence"], i["keys"]) for i in train),
             "training must defeat global color counting")
    forks = [f for i in train for f in i["forks"]]
    _require(any(f["lengths"][1] < min(f["lengths"][0], f["lengths"][2]) for f in forks)
             and any(f["lengths"][1] > max(f["lengths"][0], f["lengths"][2]) for f in forks), "both central stopping relations must be taught")
    _require({f["out"] for f in forks} == set(DIRS), "all orientations must be demonstrated")
    training_keys = set.union(*(i["keys"] for i in train))
    rejected_keys = {n for i in train for _, n in i["runs"] if n not in i["keys"]}
    max_train = max(len(bars) for i in train for bars in i["destinations"].values())
    for i in test:
        activation = bool((i["keys"] - training_keys) & rejected_keys)
        extension = max(len(bars) for bars in i["destinations"].values()) > max_train
        _require(activation or extension, "test needs a visible key or queue shift")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded checks over 42 seeds plus count and malformed-input regressions."""
    shapes, paths, geometries, backgrounds, spans, run_lengths, queue_sizes = set(), set(), set(), set(), set(), set(), set()
    puzzles, modes, receiver_counts = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze(pair["input"])
                shapes.add(pair["input"].shape)
                paths.add(tuple(_normalize(info["path"])))
                backgrounds.add(info["background"])
                run_lengths.update(n for _, n in info["runs"])
                for f in info["forks"]:
                    spans.add(f["span"])
                    geometries.add((f["out"], f["width"], f["span"], f["lengths"]))
                queue_sizes.update(len(bars) for bars in info["destinations"].values())
                receiver_counts.add(len(info["forks"]))
                if split == "test":
                    modes.add(len(info["forks"]))
    assert len(puzzles) == 42 and len(paths) >= 100 and len(shapes) >= 25
    assert len(geometries) >= 100 and len(backgrounds) == 10
    assert spans == {1, 2, 3, 4} and {1, 2, 3, 4} <= run_lengths
    assert {0, 1, 2, 3, 4, 5, 6} <= queue_sizes and modes == {2, 3}
    assert receiver_counts == {1, 2, 3}
    assert _puzzle_key(generate_puzzle(17)) == _puzzle_key(generate_puzzle(17))
    # A rule-level fixture beyond sampler bounds: span five, center depth seven,
    # an unused span-two receiver, singleton rejected runs, and border contact.
    grid = np.zeros((16, 30), dtype=int)
    path = [(0, c) for c in range(5)] + [(r, 4) for r in range(1, 5)] + [(4, c) for c in range(5, 8)]
    sequence = [2] * 5 + [3] + [4] * 5 + [2]
    for point, color in zip(path, sequence):
        grid[point] = color
    grid[15, 5:14] = 1
    grid[14, 5] = 1
    grid[12:15, 13] = 1
    grid[8:15, 7:12] = 1
    grid[15, 19:25] = 6
    grid[13:15, 19] = 6
    grid[12:15, 24] = 6
    grid[13:15, 21:23] = 6
    expected = grid.copy()
    for point in path:
        expected[point] = 0
    expected[7, 7:12] = 2
    expected[6, 7:12] = 4
    assert np.array_equal(solve(grid), expected)
    assert not _analyze(grid)["destinations"][2]
    # All eight image symmetries: recompute the higher endpoint. Rotations can
    # reverse arrival order even though a receiver's queue still grows outward.
    endpoints = np.zeros_like(grid)
    endpoints[path[0]], endpoints[path[-1]] = 1, 2
    first, second = np.zeros_like(grid, dtype=bool), np.zeros_like(grid, dtype=bool)
    first[7, 7:12], second[6, 7:12] = True, True
    for rotation in range(4):
        for reflection in (False, True):
            def transform(array):
                result = np.rot90(array, rotation)
                return np.fliplr(result) if reflection else result
            markers = transform(endpoints)
            reverse = np.argwhere(markers == 1)[0, 0] > np.argwhere(markers == 2)[0, 0]
            target = transform(expected).copy()
            if reverse:
                target[transform(first)], target[transform(second)] = 4, 2
            assert np.array_equal(solve(transform(grid)), target)
    # Alter visible geometry without altering packet colors: removing the
    # unused receiver is harmless, while removing the accepting receiver erases
    # all cargo and leaves only the formerly inactive fork.
    without_key = grid.copy()
    for point in next(f["cells"] for f in _analyze(grid)["forks"] if f["span"] == 5):
        without_key[point] = 0
    answer = solve(without_key)
    assert not np.array_equal(answer, expected) and not np.any(np.isin(answer, (2, 3, 4)))
    key_cells = next(f["cells"] for f in _analyze(grid)["forks"] if f["span"] == 5)
    other_cells = next(f["cells"] for f in _analyze(grid)["forks"] if f["span"] == 2)
    duplicate = grid.copy()
    for point in other_cells:
        duplicate[point] = 0
    for r, c in key_cells:
        duplicate[r, c + 15] = 6
    overflow = grid.copy()
    for point in key_cells:
        overflow[point] = 0
    for r, c in key_cells:
        overflow[r - 8, c + 10] = 1
    for malformed_grid in (duplicate, overflow):
        try:
            solve(malformed_grid)
        except ValueError:
            continue
        raise AssertionError("duplicate width or overflowing queue accepted")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + ne, nt, ne)
            assert (len(p["train"]), len(p["test"])) == (nt, ne)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    bad = generate_puzzle(12, 2, 1)
    info = _analyze(bad["test"][0]["input"])
    point = next(iter(next(bars for bars in info["destinations"].values() if bars)[0][1]))
    bad["test"][0]["output"][point] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted docked bar accepted")
    for cells in ({(1, 1), (1, 2), (1, 3)}, {(1, 2), (2, 1), (2, 2), (2, 3)},
                  {(1, 1), (1, 2), (2, 1), (2, 2)}, {(1, 1), (2, 1), (4, 4)}):
        assert _trace(cells) is None
    malformed = {(0, q) for q in range(7)} | {(k, q) for k in range(1, 4) for q in (0, 6)}
    try:
        _fork(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("missing central prong accepted")
