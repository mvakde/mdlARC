"""Read the first encounters of reflection orbits along an erased lattice trail.

Only standard library and NumPy. No files, examples, seeds, or construction
metadata enter solve(). See task_spec.md for the intended rule and conventions.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape),
             "grids must be two dimensional with dimensions 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid colors must be integers in 0..9")
    return a.astype(np.int8, copy=False)


def _orbit(point, shape, axes):
    r, c = point
    h, w = shape
    a, b = axes
    rr, cc = a - r, b - c
    rows = {r, rr} if 0 <= rr < h else {r}
    cols = {c, cc} if 0 <= cc < w else {c}
    return tuple(sorted((y, x) for y in rows for x in cols))


def _orbits(shape, axes):
    seen = set()
    result = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            if (r, c) not in seen:
                group = _orbit((r, c), shape, axes)
                seen.update(group)
                result.append(group)
    return result


def _trace(cells):
    """Trace actual pixels, never an assumed sweep or construction order."""
    cells = set(cells)
    _require(len(cells) >= 2, "trail is too short")
    neighbors = {p: [(p[0] + dr, p[1] + dc) for dr, dc in _DIRECTIONS
                     if (p[0] + dr, p[1] + dc) in cells] for p in cells}
    endpoints = sorted(p for p in cells if len(neighbors[p]) == 1)
    _require(len(endpoints) == 2 and all(1 <= len(v) <= 2 for v in neighbors.values()),
             "trail must be one induced nonbranching chain")
    _require(endpoints[0][0] != endpoints[1][0], "endpoint rows must differ")
    previous, current = None, endpoints[0]
    path, visited = [], set()
    while current is not None:
        _require(current not in visited, "cyclic trail")
        visited.add(current)
        path.append(current)
        choices = [p for p in neighbors[current] if p != previous]
        previous, current = current, choices[0] if choices else None
    _require(len(path) == len(cells), "disconnected trail")
    return path


def _axis_candidates(grid, marker, dimension):
    """All nontrivial in-bounds fold sums, without an overlap threshold."""
    data = grid if dimension == 0 else grid.T
    n = data.shape[0]
    result = []
    for fold in range(1, 2 * n - 2):
        first = np.arange(max(0, fold - n + 1), (fold + 1) // 2)
        if first.size == 0:
            continue
        second = fold - first
        x, y = data[first], data[second]
        visible = (x != marker) & (y != marker)
        # Requiring a visible comparison excludes entirely unobserved axes.
        if visible.any() and np.all(x[visible] == y[visible]):
            result.append(fold)
    return result


def _first_orbits(path, shape, axes):
    seen, result = set(), []
    for point in path:
        group = _orbit(point, shape, axes)
        if group not in seen:
            seen.add(group)
            result.append(group)
    return result


def _first_diagonal_count(path, shape, axes):
    """Count emitted first visits whose ONLY witness is doubly reflected."""
    cells, seen, count = set(path), set(), 0
    for r, c in path:
        group = _orbit((r, c), shape, axes)
        if group in seen:
            continue
        seen.add(group)
        count += (len(group) == 4 and (axes[0] - r, c) in cells
                  and (r, axes[1] - c) in cells
                  and (axes[0] - r, axes[1] - c) not in cells)
    return int(count)


def _interpret(input_grid):
    """Infer erasure role, path and folds entirely from the supplied input."""
    grid = _grid(input_grid)
    candidates = []
    for value in np.unique(grid):
        marker = int(value)
        cells = [tuple(map(int, p)) for p in np.argwhere(grid == marker)]
        try:
            path = _trace(cells)
        except ValueError:
            continue
        rows = _axis_candidates(grid, marker, 0)
        cols = _axis_candidates(grid, marker, 1)
        if len(rows) != 1 or len(cols) != 1:
            continue
        axes = (rows[0], cols[0])
        values = {}
        valid = True
        for group in _orbits(grid.shape, axes):
            witnesses = {int(grid[p]) for p in group if grid[p] != marker}
            if len(witnesses) != 1:
                valid = False
                break
            values[group] = next(iter(witnesses))
        if not valid:
            continue
        ordered = _first_orbits(path, grid.shape, axes)
        if not 1 <= len(ordered) <= 30:
            continue
        candidates.append({"marker": marker, "path": path, "axes": axes,
                           "values": values, "ordered": ordered,
                           "shape": grid.shape})
    _require(len(candidates) == 1, "input must identify one marker and two unique folds")
    return candidates[0]


def _answer(info, order=None):
    groups = info["ordered"] if order is None else order
    return np.asarray([info["values"][group] for group in groups], dtype=np.int8).reshape(-1, 1)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only solution: first recovered value per mirror orbit along trail."""
    return _answer(_interpret(input_grid))


def _steps(path):
    return [(q[0] - p[0], q[1] - p[1]) for p, q in zip(path, path[1:])]


def _alternative_orders(path, shape, axes):
    """Natural order shortcuts, for evidence checks rather than decoding."""
    orders = [path[::-1], sorted(path), sorted(path, key=lambda p: (p[1], p[0]))]
    columns = sorted({c for _, c in path})
    for cs in (columns, columns[::-1]):
        for reverse_last in (False, True):
            orders.append([p for c in cs for p in sorted(
                (p for p in path if p[1] == c), reverse=reverse_last and c == cs[-1])])
    result = [_first_orbits(order, shape, axes) for order in orders]
    result.append(sorted(_first_orbits(path, shape, axes)))
    return result


def _evidence(info):
    path, axes, shape = info["path"], info["axes"], info["shape"]
    counts = Counter(_orbit(p, shape, axes) for p in path)
    steps = _steps(path)
    answer = _answer(info).ravel().tolist()
    triple = sum(n == 3 for n in counts.values())
    repeated_color = any(answer[i] == answer[j] and any(v != answer[i] for v in answer[i + 1:j])
                         for i in range(len(answer)) for j in range(i + 2, len(answer)))
    alternatives_differ = all(not np.array_equal(_answer(info, order), _answer(info))
                              for order in _alternative_orders(path, shape, axes))
    cells = set(path)
    blocked_lr = blocked_tb = diagonal_only = fringe = 0
    for r, c in path:
        rr, cc = axes[0] - r, axes[1] - c
        lr = 0 <= cc < shape[1] and cc != c and (r, cc) not in cells
        tb = 0 <= rr < shape[0] and rr != r and (rr, c) not in cells
        diag = (0 <= rr < shape[0] and 0 <= cc < shape[1]
                and (rr, cc) not in cells)
        blocked_lr += not lr
        blocked_tb += not tb
        diagonal_only += not lr and not tb and diag
        fringe += not 0 <= rr < shape[0] or not 0 <= cc < shape[1]
    return {"length": len(path), "unique": len(counts), "triple": triple,
            "revisited": sum(n > 1 for n in counts.values()),
            "turns": sum(a != b for a, b in zip(steps, steps[1:])),
            "above": min(r for r, _ in path) < path[0][0],
            "reversal": (0, 1) in steps and (0, -1) in steps,
            "up": (-1, 0) in steps, "recurring_color": repeated_color,
            "alternatives_differ": alternatives_differ,
            "blocked_lr": int(blocked_lr), "blocked_tb": int(blocked_tb),
            "diagonal_only": int(diagonal_only), "fringe": int(fringe),
            "first_diagonal_only": _first_diagonal_count(path, shape, axes),
            "border": any(r in (0, shape[0] - 1) or c in (0, shape[1] - 1) for r, c in path),
            "shape": shape, "axes": axes, "marker": info["marker"],
            "payload_colors": len(set(info["values"].values())),
            "output_colors": len(set(answer))}


def _grow_path(rng, shape, axes, length, need_above):
    h, w = shape
    # Only prospective trail vertices need orbit tuples. Avoid constructing a
    # full field of them for every rejected short walk; this preserves the
    # exact candidate order and RNG draws of the eager construction.
    available = [(r, c) for r in range(h) for c in range(w)
                 if (0 <= axes[0] - r < h and axes[0] - r != r)
                 or (0 <= axes[1] - c < w and axes[1] - c != c)]
    point = available[int(rng.integers(len(available)))]
    path, occupied = [point], {point}
    counts = Counter([_orbit(point, shape, axes)])
    previous = None
    persistence = float(rng.uniform(0.7, 4.5))
    for _ in range(length - 1):
        choices, weights = [], []
        for dr, dc in _DIRECTIONS:
            q = (point[0] + dr, point[1] + dc)
            if not (0 <= q[0] < h and 0 <= q[1] < w) or q in occupied:
                continue
            if sum((q[0] + a, q[1] + b) in occupied for a, b in _DIRECTIONS) != 1:
                continue
            group = _orbit(q, shape, axes)
            if counts[group] + 1 >= len(group):
                continue
            choices.append((q, (dr, dc), group))
            weights.append((persistence if previous == (dr, dc) else 1.0)
                           * (1.4 if counts[group] else 1.0))
        if not choices:
            return None
        probs = np.asarray(weights, dtype=float)
        point, previous, group = choices[int(rng.choice(len(choices), p=probs / probs.sum()))]
        path.append(point)
        occupied.add(point)
        counts[group] += 1
    if path[0][0] == path[-1][0]:
        return None
    if path[0][0] > path[-1][0]:
        path.reverse()
    steps = _steps(path)
    if (not any(n == 3 for n in counts.values()) or not 8 <= len(counts) <= 30
            or not ((0, 1) in steps and (0, -1) in steps and (-1, 0) in steps)):
        return None
    if need_above and min(r for r, _ in path) >= path[0][0]:
        return None
    if _first_diagonal_count(path, shape, axes) == 0:
        return None
    order = _first_orbits(path, shape, axes)
    if any(order == other for other in _alternative_orders(path, shape, axes)):
        return None
    return path


def _field(rng, shape, axes, palette):
    """Orbit-based textures with independent reflected speckles."""
    result = np.empty(shape, dtype=np.int8)
    groups = _orbits(shape, axes)
    style = int(rng.integers(4))
    bh, bw = (int(x) for x in rng.integers(2, 5, size=2))
    shift = int(rng.integers(len(palette)))
    coarse = rng.integers(len(palette), size=shape)
    for group in groups:
        r, c = group[0]
        if style == 0:
            index = int(rng.integers(len(palette)))
        elif style == 1:
            index = int(coarse[r // bh, c // bw])
        elif style == 2:
            index = (r // bh + c // bw + shift) % len(palette)
        else:
            index = (max(r // bh, c // bw) + shift) % len(palette)
        if rng.random() < 0.16:
            index = int(rng.integers(len(palette)))
        for point in group:
            result[point] = palette[index]
    # Make the selected palette real without constraining the output palette.
    for index, color in zip(rng.permutation(len(groups)), palette):
        for point in groups[int(index)]:
            result[point] = color
    return result


def _sample_pair(rng, marker, target, need_above):
    for _ in range(600):
        length = int(rng.integers(27, 37) if target else rng.integers(16, 27))
        shape = tuple(int(x) for x in rng.integers(8, 23, size=2))
        offsets = [0 if rng.random() < 0.35 else int(rng.integers(-3, 4)) for _ in range(2)]
        axes = (shape[0] - 1 + offsets[0], shape[1] - 1 + offsets[1])
        path = _grow_path(rng, shape, axes, length, need_above)
        if path is None:
            continue
        palette = [int(c) for c in rng.permutation(10) if c != marker]
        palette = palette[:int(rng.integers(2, 10))]
        for _ in range(12):
            clean = _field(rng, shape, axes, palette)
            inp = clean.copy()
            for point in path:
                inp[point] = marker
            try:
                info = _interpret(inp)
            except ValueError:
                continue
            if info["marker"] != marker or info["axes"] != axes:
                continue
            evidence = _evidence(info)
            if evidence["recurring_color"] and evidence["alternatives_differ"]:
                # Ground truth uses the construction's clean witness values and
                # route; it is not obtained by assigning solve(inp) as the label.
                ordered = _first_orbits(path, shape, axes)
                out = np.asarray([clean[group[0]] for group in ordered], dtype=np.int8).reshape(-1, 1)
                return {"input": inp, "output": out}
    return None


def _fallback_pair(marker, target, variant):
    """Finite constructive termination guard, separately exercised in self_test.

    These induced chains have a diagonal-only FIRST visit, upper interior
    cells, horizontal reversals, and non-raster first-orbit order. Training
    and test lengths stay in their bands. Texture variants supply distinct
    inputs without affecting the route or the rule.
    """
    if target:
        path = [(1, 8)]
        segments = [((0, 1), 2), ((1, 0), 1), ((0, 1), 1),
                    ((1, 0), 3), ((0, -1), 3), ((-1, 0), 1),
                    ((0, -1), 5), ((-1, 0), 4), ((0, -1), 3),
                    ((1, 0), 4), ((0, 1), 1), ((1, 0), 1)]
    else:
        path = [(4, 8)]
        segments = [((0, 1), 1), ((-1, 0), 1), ((0, 1), 2),
                    ((-1, 0), 3), ((0, -1), 4), ((1, 0), 2),
                    ((0, -1), 2), ((1, 0), 2), ((0, 1), 1),
                    ((1, 0), 1), ((0, 1), 1), ((1, 0), 2), ((0, -1), 1)]
    for (dr, dc), count in segments:
        for _ in range(count):
            path.append((path[-1][0] + dr, path[-1][1] + dc))
    shape, axes = (10, 12), (9, 11)
    palette = [c for c in range(10) if c != marker]
    # Bounded local seeds only affect construction; solve has no access to them.
    for trial in range(128):
        local = np.random.default_rng(10000 * variant + trial)
        local.shuffle(palette)
        clean = _field(local, shape, axes, palette[:2 + variant % 8])
        inp = clean.copy()
        for point in path:
            inp[point] = marker
        try:
            info = _interpret(inp)
        except ValueError:
            continue
        f = _evidence(info)
        if (f["first_diagonal_only"] and f["above"] and f["recurring_color"]
                and f["alternatives_differ"]):
            out = np.asarray([clean[g[0]] for g in _first_orbits(path, shape, axes)], dtype=np.int8).reshape(-1, 1)
            return {"input": inp, "output": out}
    raise RuntimeError("constructive fallback failed")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and low <= int(value) <= high, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             "seed must be an integer")
    rng = np.random.default_rng(int(seed))
    marker = int(rng.integers(10))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for index in range(count):
            pair = None
            for retry in range(4):
                candidate = _sample_pair(rng, marker, split == "test", split == "train" and index == 0)
                if candidate is None:
                    candidate = _fallback_pair(marker, split == "test", 4 * index + retry)
                if _signature(candidate["input"]) not in seen:
                    pair = candidate
                    break
            if pair is None:
                raise RuntimeError("distinct input construction exhausted")
            puzzle[split].append(pair)
            seen.add(_signature(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check pixel semantics and demonstration witnesses, including ablations."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, markers, evidence = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            sig = _signature(inp)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            info = _interpret(inp)
            markers.add(info["marker"])
            _require(np.array_equal(out, _answer(info)), "incorrect first-orbit readout")
            f = _evidence(info)
            _require(f["triple"] > 0 and f["first_diagonal_only"] > 0,
                     "both folds must be necessary at an emitted first visit")
            _require(f["length"] > f["unique"], "orbit revisits must matter")
            _require(f["recurring_color"], "different orbits of recurring colors must remain separate")
            _require(f["alternatives_differ"], "ordered readout must defeat natural sweep shortcuts")
            # Explicit parent omissions: raw A tracing has marker values and
            # wrong length; B-only canonical-orbit reading has wrong order.
            _require(not np.array_equal(out, np.full((f["length"], 1), info["marker"])),
                     "ignoring reflection must change the answer")
            _require(not np.array_equal(out, _answer(info, sorted(info["ordered"]))),
                     "ignoring path order must change the answer")
            evidence[split].append(f)
    _require(len(markers) == 1, "one shared erasure role per puzzle")
    _require(any(f["above"] for f in evidence["train"]), "teach upper endpoint versus uppermost cell")
    # Train/test length bands are sampling shifts, not additional solve rules.


def _digest(puzzle):
    h = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed study; retains only small sets and scalar counters."""
    signatures, shapes, geometries, markers, lengths, outputs = set(), set(), set(), set(), set(), set()
    offsets, parity, palette_sizes, turns = set(), set(), set(), set()
    above = border = fringe = triple = pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_digest(puzzle))
        if seed in (0, 17, 41):
            _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "nondeterministic puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _interpret(pair["input"])
                f = _evidence(info)
                _require(np.array_equal(solve(pair["input"].copy()), pair["output"]), "solver mismatch")
                # Separate direct pixel witness check, without a repaired field.
                expected, seen = [], set()
                for r, c in info["path"]:
                    group = _orbit((r, c), info["shape"], info["axes"])
                    if group in seen:
                        continue
                    seen.add(group)
                    values = [int(pair["input"][p]) for p in group if pair["input"][p] != info["marker"]]
                    _require(values and len(set(values)) == 1, "lost reflection witness")
                    expected.append(values[0])
                _require(expected == pair["output"].ravel().tolist(), "pixel witness check failed")
                shapes.add(info["shape"])
                origin = info["path"][0]
                geometries.add(tuple((r - origin[0], c - origin[1]) for r, c in info["path"]))
                markers.add(info["marker"])
                lengths.add(f["length"])
                outputs.add(f["unique"])
                offsets.add(tuple(int(np.sign(a - d + 1)) for a, d in zip(info["axes"], info["shape"])))
                parity.add(tuple(a % 2 for a in info["axes"]))
                palette_sizes.add(f["payload_colors"])
                turns.add(f["turns"])
                above += f["above"]
                border += f["border"]
                fringe += f["fringe"] > 0
                triple += f["triple"]
                pairs += 1
        _require(max(int(np.count_nonzero(p["input"] == _interpret(p["input"])["marker"])) for p in puzzle["train"])
                 < min(int(np.count_nonzero(p["input"] == _interpret(p["input"])["marker"])) for p in puzzle["test"]),
                 "intended longer-trail test shift missing")
    _require(len(signatures) == 42 and pairs == 210, "42-seed suite incomplete")
    _require(len(shapes) >= 50 and len(geometries) >= 150, "geometry diversity collapsed")
    _require(len(markers) >= 8 and len(palette_sizes) >= 7, "palette roles collapsed")
    _require(len(lengths) >= 18 and len(outputs) >= 12 and len(turns) >= 8, "count diversity collapsed")
    _require(len(offsets) == 9 and len(parity) == 4, "fold diversity collapsed")
    _require(above >= 42 and border >= 15 and fringe >= 10 and triple >= pairs, "missing structural coverage")
    # The host exercises the entire legal count matrix. Keep the built-in
    # study bounded while including the minimum and maximum explicit counts.
    for nt, ns in ((2, 1), (5, 4), (7, 3), (10, 1), (10, 4)):
        p = generate_puzzle(5000 + nt * 10 + ns, nt, ns)
        _require(len(p["train"]) == nt and len(p["test"]) == ns, "wrong explicit counts")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for marker in range(10):
        for target in (False, True):
            p = _fallback_pair(marker, target, marker)
            info = _interpret(p["input"])
            _require(np.array_equal(solve(p["input"]), p["output"]), "fallback mismatch")
            _require(_evidence(info)["above"], "fallback must teach upper endpoint")
    # Valid small cases outside sampling bands: two erased partners emit one
    # value; a simple two-cell path emits two; odd axes retain their singleton.
    for shape, axes, path in (((2, 30), (1, 29), [(0, 1), (1, 1)]),
                              ((6, 8), (5, 7), [(0, 0), (1, 0)]),
                              ((7, 9), (6, 8), [(0, 0), (0, 1), (1, 1)])):
        for trial in range(100):
            clean = _field(np.random.default_rng(trial), shape, axes, list(range(1, 10)))
            inp = clean.copy()
            for point in path:
                inp[point] = 0
            try:
                _interpret(inp)
            except ValueError:
                continue
            expected = np.asarray([clean[group[0]] for group in _first_orbits(path, shape, axes)]).reshape(-1, 1)
            _require(np.array_equal(solve(inp), expected), "small out-of-band fixture mismatch")
            break
        else:
            raise AssertionError("small fixture construction exhausted")
    # A wrong output value must be detected without trusting the label.
    p = generate_puzzle(0, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    for cells in ({(0, 0), (0, 1), (0, 2)}, {(1, 0), (1, 1), (1, 2), (0, 1)},
                  {(0, 0), (0, 1), (1, 0), (1, 1)}, {(0, 0), (1, 0), (3, 3)}):
        try:
            _trace(cells)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed trail accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, count boundaries, orbit and shortcut checks")
