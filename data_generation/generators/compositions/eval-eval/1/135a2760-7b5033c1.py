"""Periodic route selection and repair through a chain of false detours.

Only Python's standard library and NumPy are used. Geometry and sequence are
sampled independently; solve() receives nothing except the visible input.
"""
from __future__ import annotations

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return (grid.shape, grid.tobytes())


def _neighbors(cells):
    return {q: sorted((q[0] + dr, q[1] + dc) for dr, dc in _DIRS
                      if (q[0] + dr, q[1] + dc) in cells) for q in cells}


def _blocks(graph):
    """Tarjan edge blocks; each admissible nontrivial block is one cycle."""
    clock = 0
    discovery, low, stack, blocks = {}, {}, [], []

    def visit(q, parent):
        nonlocal clock
        discovery[q] = low[q] = clock
        clock += 1
        for v in graph[q]:
            if v == parent:
                continue
            if v not in discovery:
                stack.append((q, v))
                visit(v, q)
                low[q] = min(low[q], low[v])
                if low[v] >= discovery[q]:
                    block = []
                    while True:
                        edge = stack.pop()
                        block.append(edge)
                        if edge == (q, v):
                            break
                    blocks.append(block)
            elif discovery[v] < discovery[q]:
                stack.append((q, v))
                low[q] = min(low[q], discovery[v])

    visit(min(graph), None)
    if len(discovery) != len(graph):
        raise ValueError("disconnected foreground")
    return blocks


def _read_routes(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    if int(counts.max()) * 2 <= grid.size:
        raise ValueError("background must occupy more than half the grid")
    cells = {tuple(map(int, q)) for q in np.argwhere(grid != background)}
    if not 12 <= len(cells) <= 70:
        raise ValueError("unsupported foreground size")
    graph = _neighbors(cells)
    ends = sorted(q for q in cells if len(graph[q]) == 1)
    if (len(ends) != 2 or ends[0][0] == ends[1][0]
            or any(len(v) not in (1, 2, 3) for v in graph.values())):
        raise ValueError("need two unequal-height endpoints and no extra branches")
    cycle_count = 0
    for block in _blocks(graph):
        if len(block) == 1:
            continue
        degrees = {}
        for a, b in block:
            degrees[a] = degrees.get(a, 0) + 1
            degrees[b] = degrees.get(b, 0) + 1
        if len(block) != len(degrees) or set(degrees.values()) != {2}:
            raise ValueError("detours must be separate simple cycles")
        cycle_count += 1
    if not 1 <= cycle_count <= 3:
        raise ValueError("expected one to three detours")
    start, finish = ends  # Sorted by row, so start is the higher endpoint.
    routes = []

    def walk(q, path, visited):
        if q == finish:
            if len(path) > 30:
                raise ValueError("every candidate route must have at most 30 cells")
            routes.append(tuple(path))
            return
        # A partial walk can go around a cycle back toward an already visited
        # junction and die there. Its length is not an endpoint-route length.
        for v in graph[q]:
            if v not in visited:
                walk(v, path + [v], visited | {v})

    walk(start, [start], {start})
    if (len(routes) != 2 ** cycle_count or not routes
            or min(map(len, routes)) < 12
            or set().union(*(set(p) for p in routes)) != cells):
        raise ValueError("cycles must form a single chain between the endpoints")
    return {"grid": grid, "background": background, "cells": cells,
            "graph": graph, "routes": routes, "cycles": cycle_count}


def _periodic_explanations(sequence):
    """Enumerate all strict-majority repairs with >=3 observations per phase.

    No construction period, palette dictionary, or noise location is supplied.
    Period multiples producing the identical repair are harmless.
    """
    n = len(sequence)
    allowance = max(1, n // 8)
    for period in range(2, n // 3 + 1):
        word = []
        errors = 0
        for phase in range(period):
            group = sequence[phase::period]
            histogram = np.bincount(group, minlength=10)
            color = int(histogram.argmax())
            if int(histogram[color]) * 2 <= len(group):
                break
            word.append(color)
            errors += len(group) - int(histogram[color])
        else:
            if errors <= allowance and len(set(word)) >= 2:
                repaired = tuple(word[i % period] for i in range(n))
                yield errors, period, repaired


def _resolve(input_grid):
    info = _read_routes(input_grid)
    explanations = {}
    for route in info["routes"]:
        seq = np.asarray([int(info["grid"][q]) for q in route], dtype=np.int8)
        for errors, period, repaired in _periodic_explanations(seq):
            identity = (route, repaired)
            old = explanations.get(identity)
            if old is None or (errors, period) < old:
                explanations[identity] = (errors, period)
    if not explanations:
        raise ValueError("no sufficiently supported periodic route")
    best = min(e[0] for e in explanations.values())
    winners = [(key, value) for key, value in explanations.items() if value[0] == best]
    if len(winners) != 1:
        raise ValueError("periodic route or repair is ambiguous")
    (route, repaired), (errors, period) = winners[0]
    info.update(route=route, repaired=repaired, errors=errors, period=period)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Choose the unique most periodic route, repair it, and straighten it."""
    return np.asarray(_resolve(input_grid)["repaired"], dtype=np.int8).reshape(-1, 1)


def _walk(rng, length, persistence):
    path, occupied = [(0, 0)], {(0, 0)}
    previous = None
    for _ in range(length - 1):
        candidates, weights = [], []
        for dr, dc in _DIRS:
            q = (path[-1][0] + dr, path[-1][1] + dc)
            if q in occupied:
                continue
            if sum((q[0] + a, q[1] + b) in occupied for a, b in _DIRS) != 1:
                continue
            candidates.append((q, (dr, dc)))
            weights.append(persistence if previous == (dr, dc) else 1.0)
        if not candidates:
            return None
        weights = np.asarray(weights)
        q, previous = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
        occupied.add(q)
        path.append(q)
    return path


def _ears(rng, path, count):
    """Add rectangular bypasses only when the pixel graph is exactly intended.

    They replace disjoint straight intervals of an independently grown walk.
    Both routes remain candidates: bypasses are not labeled as true or false.
    """
    occupied, covered, ears = set(path), set(), []
    extra = 0
    for _ in range(count):
        options = []
        for i in range(1, len(path) - 2):
            dr, dc = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
            for span in range(1, 5):
                j = i + span
                if j >= len(path) - 1 or covered.intersection(range(i, j + 1)):
                    continue
                if any(path[t + 1] != (path[t][0] + dr, path[t][1] + dc)
                       for t in range(i, j)):
                    continue
                for depth in range(1, 4):
                    if (span > 1 and depth == 1) or len(path) + extra + 2 * depth > 30:
                        continue
                    for sign in (-1, 1):
                        pr, pc = sign * dc, -sign * dr
                        branch = [path[i]]
                        for a, b, steps in ((pr, pc, depth), (dr, dc, span), (-pr, -pc, depth)):
                            for _ in range(steps):
                                branch.append((branch[-1][0] + a, branch[-1][1] + b))
                        new = set(branch[1:-1])
                        if new & occupied:
                            continue
                        combined = occupied | new
                        intended = {q: {branch[t - 1], branch[t + 1]}
                                    for t, q in enumerate(branch[1:-1], 1)}
                        if any(set(v) != intended[q] for q, v in _neighbors(combined).items() if q in new):
                            continue
                        options.append((i, j, depth, branch))
        if not options:
            return None
        i, j, depth, branch = options[int(rng.integers(len(options)))]
        ears.append((i, j, branch))
        covered.update(range(i, j + 1))
        occupied.update(branch[1:-1])
        extra += 2 * depth
    return sorted(ears)


def _chosen_path(path, ears, mask):
    at = {i: (j, branch, take) for (i, j, branch), take in zip(ears, mask)}
    chosen, i = [path[0]], 0
    while i < len(path) - 1:
        if i in at and at[i][2]:
            j, branch, _ = at[i]
            chosen.extend(branch[1:])
            i = j
        else:
            i += 1
            chosen.append(path[i])
    return chosen


def _transform(rng, all_cells, route):
    rotation, reflection = int(rng.integers(4)), int(rng.choice((-1, 1)))

    def trans(q):
        r, c = q[0], reflection * q[1]
        for _ in range(rotation):
            r, c = c, -r
        return r, c

    cells, route = {trans(q) for q in all_cells}, [trans(q) for q in route]
    if route[0][0] == route[-1][0]:
        cells, route = {(c, -r) for r, c in cells}, [(c, -r) for r, c in route]
    if route[0][0] > route[-1][0]:
        route.reverse()
    top, left = min(r for r, _ in cells), min(c for _, c in cells)
    return ({(r - top, c - left) for r, c in cells},
            [(r - top, c - left) for r, c in route])


def _embed(rng, cells, route, colors, background, symbols):
    bh, bw = max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1
    h = min(30, bh + int(rng.integers(0, 6)))
    w = min(30, bw + int(rng.integers(0, 6)))
    while h * w <= 2 * len(cells):
        if (h <= w and h < 30) or w == 30:
            h += 1
        else:
            w += 1
    top, left = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
    grid = np.full((h, w), background, dtype=np.int8)
    # Rejected branches share the data palette; no special decoy color exists.
    for r, c in sorted(cells):
        grid[r + top, c + left] = int(rng.choice(symbols))
    shifted = [(r + top, c + left) for r, c in route]
    for q, color in zip(shifted, colors):
        grid[q] = color
    return grid, tuple(shifted)


def _sequence(rng, length, period, background):
    distinct = int(rng.integers(2, min(4, period) + 1))
    symbols = [int(c) for c in rng.permutation([c for c in range(10) if c != background])[:distinct]]
    for _ in range(64):
        word = [int(rng.choice(symbols)) for _ in range(period)]
        if set(word) != set(symbols):
            continue
        if any(all(word[i] == word[i % d] for i in range(period))
               for d in range(1, period) if period % d == 0):
            continue
        break
    else:
        word = symbols + [symbols[0]] * (period - distinct)
        # A singleton second symbol makes this fallback word primitive.
    clean = tuple(word[i % period] for i in range(length))
    noisy = list(clean)
    errors = int(rng.integers(1, min(2, max(1, length // 8)) + 1))
    for phase in rng.choice(period, size=errors, replace=False):
        index = int(rng.choice(np.arange(int(phase), length, period)))
        noisy[index] = int(rng.choice([c for c in symbols if c != noisy[index]]))
    return clean, noisy, symbols


def _sweeps(info):
    """Color sequences under common spatial sweeps, using repaired route cells."""
    colors = dict(zip(info["route"], info["repaired"]))
    for axis in (0, 1):
        for descending in (False, True):
            order = sorted(info["route"], key=lambda q: (q[axis], q[1 - axis]), reverse=descending)
            yield tuple(colors[q] for q in order)


def _make_pair(rng, bubbles, period, background, low, high, force_long=False,
               force_short=False, sweep_witness=False, allow_above=True,
               need_above=False):
    # All retries are bounded. The final phase uses straight constructive
    # scaffolds with separated ears, still with sampled lengths and colors.
    for attempt in range(640):
        minimum = max(12, 3 * period - 2 * bubbles)
        maximum = min(high, 30 - 2 * bubbles)
        base_length = int(rng.integers(minimum, maximum + 1))
        if attempt < 480:
            persistence = float(rng.choice((0.6, 1.5, 4.0, 12.0)))
            path = _walk(rng, base_length, persistence)
            if path is None:
                continue
            ears = _ears(rng, path, bubbles)
            if ears is None:
                continue
        else:
            if need_above:
                path = [(0, 0), (-1, 0), (-2, 0), (-2, 1), (-2, 2), (-2, 3)]
                path += [(r, 3) for r in range(-1, base_length - 7)]
                ears = _ears(rng, path, bubbles)
                if ears is None:
                    continue
            else:
                path = [(i, 0) for i in range(base_length)]
                ears = [(2 + 3 * b, 3 + 3 * b,
                         [(2 + 3 * b, 0), (2 + 3 * b, 1),
                          (3 + 3 * b, 1), (3 + 3 * b, 0)]) for b in range(bubbles)]
        mask = [bool(rng.integers(2)) for _ in ears]
        # Alternation has lattice parity: skipping an even number of cells
        # cannot disrupt it. Do not manufacture a uniquely identifiable long
        # period-2 branch; ambiguous proposals are excluded by the same rule.
        if period == 2:
            mask = [False] * bubbles
        if force_long:
            mask[int(rng.integers(bubbles))] = True
        if force_short:
            mask[int(rng.integers(bubbles))] = False
        chosen = _chosen_path(path, ears, mask)
        if not max(low, 3 * period) <= len(chosen) <= high:
            continue
        cells = set(path).union(*(set(branch) for _, _, branch in ears))
        cells, chosen = _transform(rng, cells, chosen)
        above = min(r for r, _ in cells) < chosen[0][0]
        if (above and not allow_above) or (need_above and not above):
            continue
        clean, noisy, symbols = _sequence(rng, len(chosen), period, background)
        grid, shifted = _embed(rng, cells, chosen, noisy, background, symbols)
        try:
            info = _resolve(grid)
        except ValueError:
            continue
        if info["route"] != shifted or info["repaired"] != clean or info["period"] != period:
            continue
        if not info["cells"] - set(shifted):
            continue
        if sweep_witness and (clean == clean[::-1] or any(s == clean for s in _sweeps(info))):
            continue
        lengths = [len(p) for p in info["routes"]]
        if force_long and len(chosen) == min(lengths):
            continue
        if force_short and len(chosen) == max(lengths):
            continue
        return {"input": grid, "output": np.asarray(clean, dtype=np.int8).reshape(-1, 1)}
    raise RuntimeError("bounded periodic-route construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Two train/test plans vary only input sampling, never the solution rule.
    more_detours = bool(rng.integers(2))
    shared_background = bool(rng.integers(2))
    allow_above = bool(rng.integers(2))
    background = int(rng.integers(10))
    first_counts = list(map(int, rng.permutation((1, 2))))
    first_period = int(rng.choice((3, 4, 5)))
    second_period = int(rng.choice([p for p in (2, 3, 4, 5) if p != first_period]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            bubbles = (first_counts[index] if split == "train" and index < 2 else
                       (int(rng.integers(1, 3)) if split == "train" and more_detours else
                        (3 if split == "test" and more_detours else int(rng.integers(1, 4)))))
            period = (first_period if split == "train" and index == 0 else
                      (second_period if split == "train" and index == 1 else int(rng.integers(2, 6))))
            bg = background if shared_background else int(rng.integers(10))
            low, high = ((14, 30) if more_detours else ((14, 23) if split == "train" else (24, 30)))
            for _ in range(16):
                pair = _make_pair(rng, bubbles, period, bg, low, high,
                                  force_long=split == "train" and index == 0,
                                  force_short=split == "train" and index == 1,
                                  sweep_witness=split == "train" and index == 0,
                                  allow_above=allow_above,
                                  need_above=allow_above and split == "train" and index == 0)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded unique-input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
        raise AssertionError("pair must have exactly input/output")
    a, b = _grid(pair["input"]), _grid(pair["output"])
    info = _resolve(a)
    assert np.array_equal(b, np.asarray(info["repaired"]).reshape(-1, 1)), "wrong repaired readout"
    assert b.shape == (len(info["route"]), 1)
    noisy = tuple(int(a[q]) for q in info["route"])
    assert sum(x != y for x, y in zip(noisy, info["repaired"])) == info["errors"]
    assert info["errors"] <= max(1, len(noisy) // 8)
    assert len(noisy) // info["period"] >= 3
    assert len(info["cells"] - set(info["route"])) > 0, "detours must matter"
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    infos, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _validate_pair(pair)
            key = _key(info["grid"])
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(info)
    train = infos["train"]
    # Evidence checks are separate from the input-only rule and its parser.
    assert len({i["cycles"] for i in train}) >= 2
    assert len({i["period"] for i in train}) >= 2
    assert any(len(i["route"]) > min(map(len, i["routes"])) for i in train)
    assert any(len(i["route"]) < max(map(len, i["routes"])) for i in train)
    assert any(i["errors"] > 0 for i in train), "A's repair must change a training answer"
    assert any(i["repaired"] != i["repaired"][::-1]
               and all(s != i["repaired"] for s in _sweeps(i)) for i in train)
    if any(min(r for r, _ in i["cells"]) < i["route"][0][0] for i in infos["test"]):
        assert any(min(r for r, _ in i["cells"]) < i["route"][0][0]
                   and i["repaired"] != i["repaired"][::-1] for i in train)


def self_test():
    """Bounded semantic, counterfactual, interface and 42-seed coverage checks."""
    puzzles, shapes, layouts, periods, lengths, cycles, roles = [], set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _validate_pair(pair)
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add(info["grid"].shape)
                r0, c0 = min(r for r, _ in info["cells"]), min(c for _, c in info["cells"])
                layouts.add(tuple(sorted((r - r0, c - c0) for r, c in info["cells"])))
                periods.add(info["period"])
                lengths.add(len(info["route"]))
                cycles.add(info["cycles"])
                roles.add(info["background"])
                # Omitting A copies the wrong colors; omitting B counts all
                # foreground cells and therefore gives the wrong strip height.
                assert info["errors"] >= 1
                assert len(info["cells"]) != len(pair["output"])
    signatures = {tuple(_key(p[r]) for s in ("train", "test") for p in puzzle[s]
                        for r in ("input", "output")) for puzzle in puzzles}
    assert len(signatures) == 42
    assert len(shapes) >= 50 and len(layouts) >= 120
    assert periods == {2, 3, 4, 5} and cycles == {1, 2, 3}
    assert len(lengths) >= 12 and roles == set(range(10))
    assert tuple(_key(p[r]) for s in ("train", "test") for p in generate_puzzle(17)[s]
                 for r in ("input", "output")) in signatures
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(91, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{r: g.copy() for r, g in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError("incorrect answer accepted")

    # Fixed geometry with two different valid answers: colors, rather than a
    # geometric shortest/longest preference, must choose the route.
    backbone = [(r, 0) for r in range(14)]
    ear = [(5, 0), (5, 1), (5, 2), (6, 2), (7, 2), (7, 1), (7, 0)]
    all_cells = set(backbone) | set(ear)
    fixtures = []
    for take in (False, True):
        route = _chosen_path(backbone, [(5, 7, ear)], [take])
        expected = tuple((1, 2, 3)[i % 3] for i in range(len(route)))
        noisy = list(expected)
        noisy[2] = 1 if noisy[2] != 1 else 2
        for trial in range(64):
            grid = np.zeros((18, 6), dtype=np.int8)
            fixture_rng = np.random.default_rng(trial)
            for q in sorted(all_cells):
                grid[q] = int(fixture_rng.integers(1, 4))
            for q, c in zip(route, noisy):
                grid[q] = c
            try:
                info = _resolve(grid)
            except ValueError:
                continue
            if info["route"] == tuple(route) and info["repaired"] == expected:
                assert np.array_equal(solve(grid), np.asarray(expected).reshape(-1, 1))
                fixtures.append(grid)
                break
        else:
            raise AssertionError("same-geometry counterfactual fixture failed")
    assert np.array_equal(fixtures[0] != 0, fixtures[1] != 0)
    assert solve(fixtures[0]).shape != solve(fixtures[1]).shape

    # A parity-colored loop makes both branches perfectly periodic: reject
    # the tie, rather than silently selecting the shortest or first DFS route.
    ambiguous = np.zeros((18, 6), dtype=np.int8)
    for r, c in all_cells:
        ambiguous[r, c] = 1 + (r + c) % 2
    malformed = fixtures[0].copy()
    malformed[-1, -1] = 1
    for bad_input in (ambiguous, malformed, fixtures[0].T):
        try:
            solve(bad_input)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous, disconnected, or tied-endpoint input accepted")

    # The input-only rule is not limited to the construction's periods 2..5.
    backbone = [(r, 0) for r in range(26)]
    for period in range(6, 11):
        # At period 10 place the loop near the end: an unsuccessful DFS walk
        # reaches 30 cells before dying at a visited junction. It must not
        # invalidate the two complete routes, which have lengths 26 and 30.
        first = 22 if period == 10 else 5
        ear = [(first, 0), (first, 1), (first, 2), (first + 1, 2),
               (first + 2, 2), (first + 2, 1), (first + 2, 0)]
        route = _chosen_path(backbone, [(first, first + 2, ear)], [True])
        word = [1] * (period - 3) + [2, 3, 4]
        expected = tuple(word[i % period] for i in range(30))
        for trial in range(64):
            grid = np.zeros((28, 7), dtype=np.int8)
            for q in set(backbone) | set(ear):
                grid[q] = 1 + trial % 4
            for q, c in zip(route, expected):
                grid[q] = c
            grid[route[1]] = 4
            try:
                info = _resolve(grid)
            except ValueError:
                continue
            if info["route"] == tuple(route) and info["repaired"] == expected:
                assert np.array_equal(solve(grid), np.asarray(expected).reshape(-1, 1))
                break
        else:
            raise AssertionError("longer-period fixture failed")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
