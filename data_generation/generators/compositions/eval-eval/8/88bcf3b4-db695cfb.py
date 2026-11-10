"""Balanced winding: diagonal incidence releases two counted, flexible arms.

No parent code is imported. All role and obstacle assignments are input-only.
Only NumPy and Python's standard library are used.
"""
from itertools import permutations

import numpy as np


DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.tobytes()


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _points(grid, color):
    return {tuple(map(int, p)) for p in np.argwhere(grid == color)}


def _components(cells, directions=DIR8):
    unseen = set(cells)
    result = []
    while unseen:
        p = min(unseen)
        todo, component = [p], {p}
        unseen.remove(p)
        while todo:
            r, c = todo.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _near(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _local(p, root, slope, direction, axis=0):
    return (direction * (p[axis] - root[axis]),
            -slope * direction * (p[1 - axis] - root[1 - axis]))


def _global(p, root, slope, direction, axis=0):
    u, v = p
    q = [0, 0]
    q[axis] = root[axis] + direction * u
    q[1 - axis] = root[1 - axis] - slope * direction * v
    return tuple(q)


def _next_cell(arm, info):
    u = arm["half"] + 1
    v = _profile(arm["obstacle"], u)[u]
    return _global((u, v), arm["root"], info["slope"], arm["direction"], info["axis"])


def _extra_space(info, output):
    return any(_inside(_next_cell(a, info), output.shape)
               and output[_next_cell(a, info)] == info["background"] for a in info["arms"])


def _silhouette_matters(arm):
    obs = arm["obstacle"]
    if not obs:
        return False
    box = {(u, v) for u in range(min(u for u, v in obs), max(u for u, v in obs) + 1)
           for v in range(min(v for u, v in obs), max(v for u, v in obs) + 1)}
    actual = _profile(obs, arm["half"])
    return any(arm["path"][u - 1] in arm["visible"] and actual[u] !=
               max(v + 1 - abs(u - t) for t, v in box)
               for u in range(1, arm["half"] + 1))


def _profile(obstacle, n):
    """One-cell exterior clearance envelope, including the fixed origin.

    Admissible silhouettes have a diagonal tangent through the origin and a
    convex exposed contour. The slope sequence can only turn +1 -> 0 -> -1.
    """
    if not obstacle:
        return list(range(n + 1))
    assert min(u for u, v in obstacle) >= 2
    assert max(v - u for u, v in obstacle) == -1
    far = max(u for u, v in obstacle)
    values = [max(v + 1 - abs(u - t) for t, v in obstacle)
              for u in range(max(n, far + 2) + 1)]
    slopes = [b - a for a, b in zip(values, values[1:])]
    assert values[0] == 0 and all(s in (-1, 0, 1) for s in slopes)
    assert slopes == sorted(slopes, reverse=True), "Exposed contour must be convex"
    assert not set(enumerate(values)) & obstacle
    return values[:n + 1]


def _assignment(grid, background, roles, axis):
    endpoint_color, eyelet_color, yarn_color, obstacle_color = roles
    endpoints = sorted(_points(grid, endpoint_color))
    assert len(endpoints) == 2
    a, b = endpoints
    span = b[0] - a[0]
    assert span >= 4 and abs(b[1] - a[1]) == span
    slope = 1 if b[1] > a[1] else -1
    bridge = {(a[0] + u, a[1] + slope * u) for u in range(span + 1)}
    roots = _points(grid, eyelet_color)
    assert 1 <= len(roots) <= 5
    assert all(not _near(p, q) for p in roots for q in roots if p != q)
    yarn = _points(grid, yarn_color)
    spools = {}
    for component in _components(yarn):
        assert 4 <= len(component) <= 40 and len(component) % 2 == 0
        touching = [p for p in roots if any(_near(p, q) for q in component)]
        assert len(touching) == 1 and touching[0] not in spools
        spools[touching[0]] = component
    assert set(spools) == roots
    active = roots & bridge
    assert 1 <= len(active) <= 3
    obstacles = _components(_points(grid, obstacle_color))
    assert obstacles and len(obstacles) <= 2 * len(active)
    assignments = {}
    for component in obstacles:
        matches = []
        for root in sorted(active):
            for direction in (-1, 1):
                local = {_local(p, root, slope, direction, axis) for p in component}
                try:
                    _profile(local, 1)
                except AssertionError:
                    continue
                matches.append((root, direction, local))
        assert len(matches) == 1, "Obstacle must have one tangent-compatible arm"
        root, direction, local = matches[0]
        assert (root, direction) not in assignments, "At most one obstacle per arm"
        assignments[root, direction] = local
    inactive_yarn = set().union(*(spools[p] for p in roots - active))
    obstacle_cells = _points(grid, obstacle_color)
    assert not bridge & (obstacle_cells | inactive_yarn)
    output = grid.copy()
    for root in active:
        for p in spools[root]:
            output[p] = background
    for p in bridge - roots:
        assert output[p] in (background, endpoint_color)
        output[p] = endpoint_color
    arms = []
    union = set()
    for root in sorted(active):
        half = len(spools[root]) // 2
        for direction in (-1, 1):
            obstacle = assignments.get((root, direction), set())
            profile = _profile(obstacle, half)
            path = [_global((u, profile[u]), root, slope, direction, axis)
                    for u in range(1, half + 1)]
            visible = {p for p in path if _inside(p, grid.shape)}
            assert not visible & (bridge | inactive_yarn | obstacle_cells | roots)
            for p in visible:
                assert output[p] in (background, yarn_color)
                output[p] = yarn_color
            arms.append({"root": root, "direction": direction, "half": half,
                         "obstacle": obstacle, "path": path, "visible": visible,
                         "wrapped": bool(obstacle) and any(profile[u] != u and
                             _inside(path[u - 1], grid.shape) for u in range(1, half + 1))})
            union.update(visible)
    analysis = {"background": background, "roles": roles, "slope": slope, "axis": axis, "shape": grid.shape,
                "span": span, "bridge": bridge, "roots": roots,
                "active": active, "spools": spools, "arms": arms,
                "obstacles": obstacles, "thread_union": union}
    analysis["extra_space"] = _extra_space(analysis, output)
    return output, analysis


def _interpret(value):
    grid = _grid(value)
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert len(values) == 5 and int(counts.max()) * 2 > grid.size
    colors = sorted(set(map(int, values)) - {background})
    candidates = []
    for roles in permutations(colors):
        for axis in (0, 1):
            try:
                candidates.append(_assignment(grid, background, roles, axis))
            except AssertionError:
                pass
    assert len(candidates) == 1, "Input must determine one semantic role assignment"
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the complete answer from the supplied grid alone."""
    return _interpret(input_grid)[0]


def _obstacle(rng):
    kind = int(rng.integers(5))
    if kind == 0:
        raw = {(0, 0)}
    elif kind == 1:
        h = int(rng.integers(2, 5))
        raw = {(u, 0) for u in range(h)}
    elif kind == 2:
        h, w = map(int, rng.integers(1, 4, size=2))
        raw = {(u, v) for u in range(h) for v in range(w)}
    elif kind == 3:
        radius = int(rng.integers(1, 3))
        raw = {(u, v) for u in range(2 * radius + 1)
               for v in range(-radius, radius + 1)
               if abs(u - radius) + abs(v) <= radius}
    else:
        ramp, plateau = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        # Thin convex cap: increasing, plateau, decreasing. Thickness varies.
        heights = list(range(ramp)) + [ramp] * plateau + list(range(ramp - 1, -1, -1))
        thickness = int(rng.integers(1, 3))
        raw = {(u, v - d) for u, v in enumerate(heights) for d in range(thickness)}
    near = int(rng.integers(2, 6))
    offset = near - 1 - max(v - u for u, v in raw)
    result = {(u + near, v + offset) for u, v in raw}
    _profile(result, 1)
    return result


def _grow_spool(rng, root, n, occupied, roots, shape, diagonal=False):
    radius = int(rng.integers(3, 7))
    allowed = {(r, c) for r in range(max(0, root[0] - radius), min(shape[0], root[0] + radius + 1))
               for c in range(max(0, root[1] - radius), min(shape[1], root[1] + radius + 1))
               if (r, c) not in occupied and (r, c) != root
               and not any(_near((r, c), q) for q in roots if q != root)}
    # Also separate the monochrome input spools and the fixed obstacle tokens.
    allowed = {p for p in allowed if not any(_near(p, q) for q in occupied if q not in roots)}
    first = sorted(p for p in allowed if _near(p, root))
    if not first:
        return None
    p = first[int(rng.integers(len(first)))]
    cells = {p}
    if diagonal:
        second = sorted(q for q in allowed if abs(q[0] - p[0]) == 1 and abs(q[1] - p[1]) == 1)
        if not second:
            return None
        cells.add(second[int(rng.integers(len(second)))])
        allowed -= {(p[0] + dr, p[1] + dc) for dr, dc in DIR4}
    style = int(rng.integers(3))
    while len(cells) < n:
        frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in DIR8} & allowed - cells)
        if not frontier:
            return None
        neighbors = np.array([sum(_near(q, p) for p in cells) for q in frontier], dtype=float)
        weights = (neighbors + 1) ** 2 if style == 0 else (neighbors + 1) ** -2 if style == 1 else np.ones(len(frontier))
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _candidate(rng, roles, background, active_count, inactive_count, slope, lesson, regime):
    # A single shared crossbar; eyelets are selected freely along it.
    minimum = 16 if active_count == 1 else 23 if active_count == 2 else 28
    h, w = map(int, rng.integers(minimum, 31, size=2))
    if regime == "short" and active_count == 1 and not lesson:
        h = int(rng.integers(13, 23))
    steps = int(rng.integers(max(6, 6 * active_count), min(h, w)))
    r0 = int(rng.integers(h - steps))
    c0 = int(rng.integers(w - steps)) + (steps if slope == -1 else 0)
    segment = [(r0 + u, c0 + slope * u) for u in range(steps + 1)]
    eligible = list(range(2, steps - 1))
    rng.shuffle(eligible)
    chosen = []
    for u in eligible:
        if all(abs(u - v) >= 6 for v in chosen):
            chosen.append(u)
        if len(chosen) == active_count:
            break
    if len(chosen) != active_count:
        return None
    active = [segment[u] for u in sorted(chosen)]
    roots = active.copy()
    if lesson == "clipped":
        # A preserved spool on the infinite line beyond an endpoint explicitly
        # teaches the finite incidence test, not merely diagonal alignment.
        extensions = [(r0 + u, c0 + slope * u)
                      for u in list(range(-10, -3)) + list(range(steps + 4, steps + 11))]
        extensions = [p for p in extensions if _inside(p, (h, w))]
        if not extensions:
            return None
        roots.append(extensions[int(rng.integers(len(extensions)))])
    for flat in rng.permutation(h * w):
        if len(roots) == active_count + inactive_count:
            break
        p = int(flat) // w, int(flat) % w
        if p not in segment and all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 6
                                    for q in roots + [segment[0], segment[-1]]):
            roots.append(p)
    if len(roots) != active_count + inactive_count:
        return None
    obstacle_cells, obstacle_sets = set(), []
    for index, root in enumerate(active):
        directions = (-1, 1) if lesson == "visible" else (int(rng.choice((-1, 1))),) if lesson == "clipped" and index == 0 else tuple(d for d in (-1, 1) if rng.random() < .65)
        for direction in directions:
            try:
                obs = _obstacle(rng)
            except AssertionError:
                return None
            if lesson == "visible" and index == 0 and direction == -1:
                # A sloping silhouette cannot be replaced by its bounding box.
                near = int(rng.integers(2, 5))
                obs = {(near, near - 1), (near + 1, near), (near + 2, near - 1)}
            cells = {_global(p, root, slope, direction) for p in obs}
            if (not all(_inside(p, (h, w)) for p in cells) or set(segment) & cells
                    or any(_near(p, q) for p in cells for q in obstacle_cells | set(roots))):
                return None
            obstacle_cells.update(cells)
            obstacle_sets.append((root, direction, obs))
    if not obstacle_cells:
        return None
    occupied = {segment[0], segment[-1]} | set(roots) | obstacle_cells
    spools = {}
    for index, root in enumerate(roots):
        if lesson == "visible" and root in active:
            far = max(u for q, d, obs in obstacle_sets if q == root for u, v in obs)
            half = far + int(rng.integers(2, 5))
        elif lesson == "clipped" and root in active:
            half = int(rng.integers(18, 21))
        elif regime == "short":
            half = int(rng.integers(2, 6))
        elif regime == "long":
            half = int(rng.integers(6, 14))
        else:
            half = int(rng.integers(2, 14))
        spool = _grow_spool(rng, root, 2 * half, occupied, set(roots), (h, w),
                            diagonal=lesson == "visible" and index == 0)
        if spool is None:
            return None
        spools[root] = spool
        occupied.update(spool)
    grid = np.full((h, w), background, dtype=np.int8)
    for p in (segment[0], segment[-1]):
        grid[p] = roles[0]
    for p in roots:
        grid[p] = roles[1]
    for component in spools.values():
        for p in component:
            grid[p] = roles[2]
    for p in obstacle_cells:
        grid[p] = roles[3]
    try:
        output, info = _interpret(grid)
    except AssertionError:
        return None
    if info["roles"] != tuple(roles) or len(info["active"]) != active_count:
        return None
    if lesson == "visible":
        if (not all(len(a["visible"]) == a["half"] for a in info["arms"])
                or not all(a["wrapped"] for a in info["arms"])
                or not _extra_space(info, output)
                or len(_components(spools[active[0]], DIR4)) == 1):
            return None
    if lesson == "clipped":
        if not any(len(a["visible"]) < a["half"] for a in info["arms"]):
            return None
    return {"input": grid, "output": output}


def _fallback(rng, roles, background, slope, lesson, used, active_count=1, inactive_count=1, regime="recombine"):
    """Finite emergency construction, with the same input grammar and lessons."""
    visible = lesson != "clipped"
    root = (14, 14) if visible else (10, 10)
    half, span = (8, 6) if visible else (20, 9)
    for index in rng.permutation(16):
        grid = np.full((30, 30), background, dtype=np.int8)
        def reflect(p):
            return (p[0], p[1]) if slope == 1 else (p[0], 29 - p[1])
        if lesson is None:
            for p in ((1, 1), (28, 28)):
                grid[reflect(p)] = roles[0]
            positions = {1: (14,), 2: (6, 22), 3: (6, 14, 22)}[active_count]
            budget_half = (6 if regime == "long" else 2 if regime == "short" else 3) + int(index) // 4
            slack_width = 3 + int(index) % 4
            for t in positions:
                grid[reflect((t, t))] = roles[1]
                yarn = {(t + 1, t + 1)} | {
                    (t + 2 + k // slack_width, t + 2 + k % slack_width)
                    for k in range(2 * budget_half - 1)}
                for p in yarn:
                    grid[reflect(p)] = roles[2]
                grid[reflect((t + 2, t - 1))] = roles[3]
            inert_roots = [(1 + int(index) // 8, 20 + int(index) % 8),
                           (20 + int(index) % 8, 1 + int(index) // 8)]
            for inert in inert_roots[:inactive_count]:
                grid[reflect(inert)] = roles[1]
                for dr, dc in ((0, 1), (0, 2), (1, 1), (1, 2)):
                    grid[reflect((inert[0] + dr, inert[1] + dc))] = roles[2]
            # Without inert ports, vary translation by reflecting both axes.
            if inactive_count == 0 and int(index) % 2:
                grid = grid[::-1, ::-1].copy()
            if _key(grid) not in used:
                output, _ = _interpret(grid)
                return {"input": grid, "output": output}
            continue
        for p in ((root[0] - span, root[1] - span), (root[0] + span, root[1] + span)):
            grid[reflect(p)] = roles[0]
        grid[reflect(root)] = roles[1]
        if visible:
            yarn = {(root[0] + 1, root[1] + 1)} | {
                (root[0] + 2 + i, root[1] + 2 + j) for i in range(3) for j in range(5)}
            obstacles = [(-1, {(3, 2), (4, 3), (5, 2)}), (1, {(3, 2)})]
        else:
            yarn = {(root[0] + i, root[1] + j) for i in range(1, 7) for j in range(1, 8)}
            yarn -= {(root[0] + 6, root[1] + 6), (root[0] + 6, root[1] + 7)}
            obstacles = [(1, {(3, 2)})]
        for p in yarn:
            grid[reflect(p)] = roles[2]
        for d, obs in obstacles:
            for p in obs:
                grid[reflect(_global(p, root, 1, d))] = roles[3]
        inert = (1 + int(index) // 8, 20 + int(index) % 8) if visible else (20 + int(index) % 8,) * 2
        grid[reflect(inert)] = roles[1]
        side = 1 if visible or int(index) < 8 else -1
        for dr, dc in ((0, 1), (0, 2), (1, 1), (1, 2)):
            grid[reflect((inert[0] + dr, inert[1] + side * dc))] = roles[2]
        if _key(grid) not in used:
            output, _ = _interpret(grid)
            return {"input": grid, "output": output}
    raise AssertionError("Emergency construction capacity exhausted")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, size=4, replace=False)))
    backgrounds = [c for c in range(10) if c not in roles]
    constant_bg = int(rng.choice(backgrounds)) if rng.random() < .35 else None
    regime = str(rng.choice(("count", "short", "orientation")))
    base_axis = int(rng.integers(2))
    base_slope = int(rng.choice((-1, 1)))
    slots = list(map(int, rng.choice(num_train, size=2, replace=False)))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            lesson = ("visible" if index == slots[0] else "clipped") if split == "train" and index in slots else None
            slope = base_slope if lesson == "visible" else -base_slope if lesson == "clipped" else int(rng.choice((-1, 1)))
            if lesson:
                active_count = 1
            elif regime == "count":
                active_count = 1 if split == "train" else int(rng.integers(2, 4))
            else:
                active_count = int(rng.choice((1, 1, 1, 2, 2, 3)))
            inactive_count = 1 if lesson == "clipped" else int(rng.choice((0, 0, 1, 2)))
            background = constant_bg if constant_bg is not None else int(rng.choice(backgrounds))
            pair_regime = ("short" if split == "test" else "long") if regime == "short" else "recombine"
            axis = (base_axis if split == "train" else 1 - base_axis) if regime == "orientation" else int(rng.integers(2))
            canonical_used = used if axis == 0 else {
                _key(np.frombuffer(data, dtype=np.int8).reshape(shape).T.copy())
                for shape, data in used}
            for attempt in range(320):
                pair = _candidate(rng, roles, background, active_count, inactive_count, slope, lesson, pair_regime)
                if pair is not None and _key(pair["input"]) not in canonical_used:
                    break
            else:
                pair = _fallback(rng, roles, background, slope, lesson, canonical_used,
                                 active_count, inactive_count, pair_regime)
            if axis:
                pair = {k: v.T.copy() for k, v in pair.items()}
            assert _key(pair["input"]) not in used
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    grid = _grid(pair["input"])
    _, info = _interpret(grid)
    silhouettes = []
    for component in info["obstacles"]:
        r0, c0 = min(r for r, c in component), min(c for r, c in component)
        silhouettes.append(tuple(sorted((r - r0, c - c0) for r, c in component)))
    return {"shape": grid.shape, "roles": info["roles"], "background": info["background"],
            "slope": info["slope"], "axis": info["axis"], "span": info["span"], "active": len(info["active"]),
            "inactive": len(info["roots"] - info["active"]),
            "budgets": tuple(sorted(len(s) for s in info["spools"].values())),
            "obstacle_count": len(silhouettes), "silhouettes": tuple(sorted(silhouettes)),
            "clipped": any(len(a["visible"]) < a["half"] for a in info["arms"]),
            "wrapped": any(a["wrapped"] for a in info["arms"]),
            "straight_arm": any(not a["obstacle"] for a in info["arms"]),
            "diagonal_slack": any(len(_components(s, DIR4)) > 1 for s in info["spools"].values()),
            "overlap": sum(len(a["visible"]) for a in info["arms"]) > len(info["thread_union"])}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    used, role_sets, lessons, queries = set(), set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in used
            used.add(_key(inp))
            expected, info = _interpret(inp)
            assert np.array_equal(expected, out) and not np.array_equal(inp, out)
            role_sets.add(info["roles"])
            endpoints, eyelet, yarn, obstacle = info["roles"]
            fixed = (inp == endpoints) | (inp == eyelet) | (inp == obstacle)
            assert np.array_equal(inp[fixed], out[fixed]), "Fixed material changed"
            for root, spool in info["spools"].items():
                if root not in info["active"]:
                    assert all(out[p] == yarn for p in spool), "Off-crossbar spool activated"
            for arm in info["arms"]:
                assert len(arm["path"]) * 2 == len(info["spools"][arm["root"]])
                assert len(set(arm["path"])) == arm["half"]
                chain = [arm["root"]] + arm["path"]
                assert all(_near(p, q) for p, q in zip(chain, chain[1:]))
                assert all(out[p] == yarn for p in arm["visible"])
            if split == "train":
                lessons.append(info)
            else:
                queries.append(info)
    assert len(role_sets) == 1, "Palette roles must be coherent within a puzzle"
    assert {i["slope"] for i in lessons} == {-1, 1}
    assert any(i["roots"] - i["active"] for i in lessons), "Missing selective activation witness"
    assert any(any(c - min(i["bridge"])[1] == i["slope"] * (r - min(i["bridge"])[0])
                       for r, c in i["roots"] - i["active"]) for i in lessons), "Missing beyond-endpoint witness"
    assert any(any(len(a["visible"]) < a["half"] for a in i["arms"]) for i in lessons)
    assert any(all(len(a["visible"]) == a["half"] and a["wrapped"] for a in i["arms"])
               and any(len(_components(i["spools"][r], DIR4)) > 1 for r in i["active"])
               and i["extra_space"]
               for i in lessons), "Missing visible counting and wrapping witness"
    assert any(any(_silhouette_matters(a) for a in i["arms"]) for i in lessons), "Missing silhouette witness"
    assert any(any(not a["obstacle"] for a in i["arms"]) for i in lessons)
    # Both ablations are observably wrong on training: dropping activation
    # clears preserved inert material; dropping A's contour uses straight rays.
    assert any(any(a["wrapped"] for a in i["arms"]) for i in lessons)
    assert len({len(i["spools"][r]) for i in lessons for r in i["active"]}) >= 2
    count_shift = all(len(i["active"]) == 1 for i in lessons) and all(len(i["active"]) >= 2 for i in queries)
    budget_shift = max(len(i["spools"][r]) for i in queries for r in i["active"]) < min(len(i["spools"][r]) for i in lessons for r in i["active"])
    axis_shift = len({i["axis"] for i in lessons}) == len({i["axis"] for i in queries}) == 1 and lessons[0]["axis"] != queries[0]["axis"]
    assert count_shift or budget_shift or axis_shift, "Missing structural train/test shift"


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _ablation(value, omission):
    """Deliberately simpler rules used only to check teaching counterexamples."""
    grid = _grid(value)
    _, info = _interpret(grid)
    endpoint, eyelet, yarn, obstacle = info["roles"]
    bg = info["background"]
    selected = info["roots"] if omission == "activation" else info["active"]
    output = grid.copy()
    for root in selected:
        for p in info["spools"][root]:
            output[p] = bg
    for p in info["bridge"] - info["roots"]:
        if output[p] in (bg, endpoint):
            output[p] = endpoint
    arms = {(a["root"], a["direction"]): a for a in info["arms"]}
    for root in sorted(selected):
        for d in (-1, 1):
            obs = arms.get((root, d), {}).get("obstacle", set())
            n = 30 if omission == "count" else len(info["spools"][root]) // 2
            profile = list(range(n + 1)) if omission == "wrapping" else _profile(obs, n)
            for u in range(1, n + 1):
                p = _global((u, profile[u]), root, info["slope"], d, info["axis"])
                if _inside(p, grid.shape) and output[p] in (bg, yarn):
                    output[p] = yarn
    return output


def self_test():
    # Explicit geometric regressions, without sampling or stored examples.
    assert _profile({(3, 2)}, 7) == [0, 1, 2, 3, 2, 1, 0, -1]
    assert _profile({(3, 2), (4, 3), (5, 2)}, 8) == [0, 1, 2, 3, 4, 3, 2, 1, 0]
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for task in puzzles:
        for omission in ("activation", "count", "wrapping"):
            assert any(not np.array_equal(_ablation(p["input"], omission), p["output"])
                       for p in task["train"]), omission
    records = [measure_pair(p) for task in puzzles for s in ("train", "test") for p in task[s]]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({r["shape"] for r in records}) >= 35
    assert len({r["roles"] for r in records}) >= 30
    assert {r["slope"] for r in records} == {-1, 1}
    assert {r["axis"] for r in records} == {0, 1}
    assert {r["active"] for r in records} >= {1, 2, 3}
    assert {r["inactive"] for r in records} >= {0, 1, 2}
    assert len({b for r in records for b in r["budgets"]}) >= 10
    assert len({s for r in records for s in r["silhouettes"]}) >= 15
    assert len({r["span"] for r in records}) >= 12
    assert any(r["overlap"] for r in records)
    assert all(any(r[name] for r in records) for name in ("clipped", "wrapped", "straight_arm", "diagonal_slack"))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(771 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    pair = puzzles[0]["test"][0]
    original = pair["input"].copy()
    assert np.array_equal(solve(original), pair["output"])
    assert np.array_equal(original, pair["input"]), "Solver mutated its input"
    for k in range(4):
        assert np.array_equal(solve(np.rot90(original, k)), np.rot90(pair["output"], k))
    assert np.array_equal(solve(original[:, ::-1]), pair["output"][:, ::-1])
    recolor = np.random.default_rng(431).permutation(10)
    assert np.array_equal(solve(recolor[original]), recolor[pair["output"]])
    for lesson in ("visible", "clipped", None):
        used = set()
        for index in range(14):
            p = _fallback(np.random.default_rng(index), (1, 2, 3, 4), 0, -1,
                          lesson, used, 3 if lesson is None else 1, 0 if lesson is None else 1)
            assert np.array_equal(solve(p["input"]), p["output"])
            used.add(_key(p["input"]))
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({r["shape"] for r in records}),
            "palettes": len({r["roles"] for r in records}),
            "silhouettes": len({s for r in records for s in r["silhouettes"]})}


if __name__ == "__main__":
    print(self_test())
