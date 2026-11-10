"""Peeling cargo on a rail: square collision redistributes nested frames.

Only NumPy and the standard library are used. All interpretation comes from
the queried grid; generation metadata never enters solve().
"""
from collections import Counter
import json

import numpy as np


RAIL, START, GOAL = 2, 3, 4
STRUCTURAL_CHOICES = (0, 1, 5, 6, 7, 8, 9)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC color")
    return a


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        _require(isinstance(value, (int, np.integer))
                 and not isinstance(value, (bool, np.bool_))
                 and low <= value <= high, f"illegal {name}")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _neighbors(p, shape):
    r, c = p
    for x, y in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= x < shape[0] and 0 <= y < shape[1]:
            yield x, y


def _ring(center, radius):
    r, c = center
    for x in range(r - radius, r + radius + 1):
        yield x, c - radius
        yield x, c + radius
    for y in range(c - radius + 1, c + radius):
        yield r - radius, y
        yield r + radius, y


def _fits(allowed, p, radius):
    r, c = p
    if r - radius < 0 or c - radius < 0:
        return False
    if r + radius >= allowed.shape[0] or c + radius >= allowed.shape[1]:
        return False
    return bool(allowed[r - radius:r + radius + 1,
                        c - radius:c + radius + 1].all())


def _draw_ring(a, center, radius, color):
    for p in _ring(center, radius):
        a[p] = color


def _draw_cargo(a, center, colors):
    radius = len(colors)
    for depth, color in enumerate(colors):
        _draw_ring(a, center, radius - depth, color)
    a[center] = START


def _parse(input_grid):
    a = _grid(input_grid)
    wall = int(a[0, 0])
    _require(wall not in (RAIL, START, GOAL), "wall uses a marker color")
    _require(np.all(a[0] == wall) and np.all(a[-1] == wall)
             and np.all(a[:, 0] == wall) and np.all(a[:, -1] == wall),
             "a uniform wall border is required")
    anchors = []
    for color in (START, GOAL):
        cells = np.argwhere(a == color)
        _require(len(cells) == 1, "each endpoint marker must occur once")
        anchors.append(tuple(map(int, cells[0])))
    start, goal = anchors
    floor_candidates = {int(a[p]) for p in _neighbors(goal, a.shape)
                        if int(a[p]) not in (wall, RAIL, START, GOAL)}
    _require(len(floor_candidates) == 1,
             "goal chamber must visibly identify the floor color")
    floor = floor_candidates.pop()
    route_mask = np.isin(a, (RAIL, START, GOAL))
    for p in map(tuple, np.argwhere(route_mask)):
        degree = sum(bool(route_mask[q]) for q in _neighbors(p, a.shape))
        _require(degree == (1 if p in (start, goal) else 2),
                 "rail must be an unbranched path with marked endpoints")
    path, previous, p = [], None, start
    visited = set()
    while True:
        _require(p not in visited, "rail contains a loop")
        visited.add(p)
        path.append(p)
        if p == goal:
            break
        choices = [q for q in _neighbors(p, a.shape)
                   if route_mask[q] and q != previous]
        _require(len(choices) == 1, "ambiguous rail")
        previous, p = p, choices[0]
    _require(len(path) == int(route_mask.sum()), "disconnected rail pixels")
    structural = (wall, floor, RAIL, START, GOAL)
    cargo_cells = np.argwhere(~np.isin(a, structural))
    _require(len(cargo_cells) > 0, "cargo frames are absent")
    radius = int(np.max(np.abs(cargo_cells - np.asarray(start))))
    _require(1 <= radius <= 4, "cargo must have one through four frames")
    allowed = a != wall
    _require(_fits(allowed, start, radius), "launch footprint hits a wall")
    _require(_fits(allowed, goal, radius),
             "goal chamber must accommodate the original cargo")
    colors = []
    for r in range(radius, 0, -1):
        values = {int(a[p]) for p in _ring(start, r) if a[p] != RAIL}
        _require(len(values) == 1 and not (values & set(structural)),
                 "each launch frame must encode one nonstructural color")
        colors.append(values.pop())
    _require(all(max(abs(int(x) - start[0]), abs(int(y) - start[1])) <= radius
                 for x, y in cargo_cells), "extra colored objects")
    _require(len(path) > radius + 1, "rail is too short")
    heading = np.subtract(path[1], path[0])
    _require(all(np.array_equal(np.subtract(path[i + 1], path[i]), heading)
                 for i in range(radius + 1)),
             "launch rail must exit the cargo in a straight radial ray")
    inside = [p for p in path if max(abs(p[0] - start[0]), abs(p[1] - start[1])) <= radius]
    _require(inside == path[:radius + 1], "rail reenters the launch cargo")
    # No isolated floor areas are permitted. A rail can still guide motion
    # through a wide chamber with many otherwise valid square placements.
    seen, queue = {start}, [start]
    for p in queue:
        for q in _neighbors(p, a.shape):
            if allowed[q] and q not in seen:
                seen.add(q)
                queue.append(q)
    _require(len(seen) == int(allowed.sum()), "disconnected traversable region")
    directions = [tuple(np.subtract(q, p)) for p, q in zip(path, path[1:])]
    turns = sum(x != y for x, y in zip(directions, directions[1:]))
    return {"wall": wall, "floor": floor, "start": start, "goal": goal,
            "radius": radius, "colors": tuple(colors), "allowed": allowed,
            "path": path, "turns": turns}


def _movement_history(model):
    """Literal motion: test the next whole footprint, peel, and retry."""
    radius = model["radius"]
    events = []
    for p, q in zip(model["path"], model["path"][1:]):
        while not _fits(model["allowed"], q, radius):
            _require(radius > 0, "the rail enters a wall")
            events.append((p, radius, model["colors"][model["radius"] - radius]))
            radius -= 1
    return events, radius


def _rail_capacities(model):
    """Maximum fitting radius at each rail cell, derived from wall pixels."""
    walls = np.argwhere(~model["allowed"])
    h, w = model["allowed"].shape
    capacities = []
    for p in model["path"]:
        wall_distance = int(np.max(np.abs(walls - p), axis=1).min())
        capacities.append(min(wall_distance - 1,
                              p[0], p[1], h - 1 - p[0], w - 1 - p[1]))
    return capacities


def _clearance_history(model):
    """Author cross-check: prefix minima of Chebyshev wall clearances.

    This uses wall coordinates rather than square windows. It is an authored
    cross-check, not an independent reference solver.
    """
    capacities = _rail_capacities(model)
    current, events = model["radius"], []
    for i, capacity in enumerate(capacities[1:], 1):
        for radius in range(current, capacity, -1):
            events.append((model["path"][i - 1], radius,
                           model["colors"][model["radius"] - radius]))
        current = min(current, capacity)
    return events, current


def _render_history(inp, model, events, remaining):
    out = np.array(inp, dtype=np.int64, copy=True)
    r, c = model["start"]
    radius = model["radius"]
    out[r - radius:r + radius + 1, c - radius:c + radius + 1] = model["floor"]
    for p in model["path"]:
        out[p] = RAIL
    for center, shell_radius, color in events:
        _draw_ring(out, center, shell_radius, color)
    offset = radius - remaining
    _draw_cargo(out, model["goal"], model["colors"][offset:])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from visible walls, rail, endpoints and nested frame colors."""
    model = _parse(input_grid)
    events, remaining = _movement_history(model)
    return _render_history(input_grid, model, events, remaining)


def _path_between_nodes(nodes):
    path = [nodes[0]]
    for a, b in zip(nodes, nodes[1:]):
        step = tuple(np.sign(np.subtract(b, a)))
        p = a
        while p != b:
            p = p[0] + int(step[0]), p[1] + int(step[1])
            path.append(p)
    return path


def _sample_nodes(rng, room_radius, min_nodes, max_nodes, min_turns=0):
    side = 2 * room_radius + 1
    limit = min(5, 29 // (side + 1))
    shapes = [(h, w) for h in range(1, limit + 1) for w in range(1, limit + 1)
              if h * w >= min_nodes]
    for _ in range(80):
        h, w = shapes[int(rng.integers(len(shapes)))]
        target = int(rng.integers(min_nodes, min(max_nodes, h * w) + 1))
        node = int(rng.integers(h)), int(rng.integers(w))
        walk, occupied = [node], {node}
        while len(walk) < target:
            choices = [q for q in _neighbors(walk[-1], (h, w)) if q not in occupied]
            if not choices:
                break
            node = choices[int(rng.integers(len(choices)))]
            walk.append(node)
            occupied.add(node)
        directions = [tuple(np.subtract(b, a)) for a, b in zip(walk, walk[1:])]
        turns = sum(a != b for a, b in zip(directions, directions[1:]))
        if len(walk) < min_nodes or turns < min_turns:
            continue

        def coordinates(count):
            gaps = np.full(count - 1, side + 1, dtype=int)
            budget = 28 - side - int(gaps.sum())
            for j in rng.permutation(count - 1):
                extra = int(rng.integers(0, min(4, budget) + 1))
                gaps[j] += extra
                budget -= extra
            return np.r_[room_radius, room_radius + np.cumsum(gaps)]

        rows, cols = coordinates(h), coordinates(w)
        return [(int(rows[x]), int(cols[y])) for x, y in walk]
    raise ValueError("bounded route sampling exhausted")


def _sample_colors(rng, palette, radius, kind):
    a, b = map(int, rng.choice(palette, 2, replace=False))
    if kind == "clear_evidence":
        return [a, b, a]
    if kind == "peel_evidence":
        return [a, a, b]
    style = int(rng.integers(5))
    if style == 0:
        return [a] * radius
    if style == 1:
        return list(map(int, rng.choice(palette, radius, replace=False)))
    if style == 2:
        return [a if i % 2 == 0 else b for i in range(radius)]
    if style == 3:
        cut = int(rng.integers(1, radius + 1))
        return [a] * cut + [b] * (radius - cut)
    return list(map(int, rng.choice(palette, radius)))


def _construct_input(rng, wall, floor, radius, kind, node_bounds,
                     room_radius=None, min_turns=0):
    if room_radius is None:
        room_radius = radius + int(rng.integers(2))
    s = room_radius
    nodes = _sample_nodes(rng, s, *node_bounds, min_turns=min_turns)
    allowed = np.zeros((30, 30), dtype=bool)
    for r, c in nodes:
        allowed[r - s:r + s + 1, c - s:c + s + 1] = True
    edges = len(nodes) - 1
    if kind == "clear_evidence":
        capacities = list(map(int, rng.integers(radius, s + 1, size=edges)))
        # A passable narrowing must be shown even with only two demonstrations.
        capacities[0] = radius
    elif kind == "peel_evidence":
        capacities = [2] + [s] * (edges - 2) + [1]
    elif rng.random() < 0.22:
        capacities = list(map(int, rng.integers(radius, s + 1, size=edges)))
    else:
        capacities = list(map(int, rng.integers(0, s + 1, size=edges)))
    for i, (a, b, cap) in enumerate(zip(nodes, nodes[1:], capacities)):
        negative, positive = cap, int(rng.integers(cap, s + 1))
        if kind == "peel_evidence" and i == 0:
            negative, positive = 2, 4  # Width 7 still clips a centered 7-square.
        if rng.integers(2):
            negative, positive = positive, negative
        if a[0] == b[0]:
            left, right = sorted((a[1], b[1]))
            allowed[a[0] - negative:a[0] + positive + 1,
                    left + s + 1:right - s] = True
        else:
            top, bottom = sorted((a[0], b[0]))
            allowed[top + s + 1:bottom - s,
                    a[1] - negative:a[1] + positive + 1] = True
    path = _path_between_nodes(nodes)
    lo, hi = np.argwhere(allowed).min(axis=0), np.argwhere(allowed).max(axis=0) + 1
    allowed = allowed[lo[0]:hi[0], lo[1]:hi[1]]
    path = [(p[0] - int(lo[0]), p[1] - int(lo[1])) for p in path]
    padding = []
    for dimension in allowed.shape:
        extra = int(rng.integers(2, min(8, 30 - dimension) + 1))
        before = int(rng.integers(1, extra))
        padding.append((before, extra - before))
    allowed = np.pad(allowed, padding)
    path = [(p[0] + padding[0][0], p[1] + padding[1][0]) for p in path]
    inp = np.full(allowed.shape, wall, dtype=np.int64)
    inp[allowed] = floor
    palette = [c for c in STRUCTURAL_CHOICES if c not in (wall, floor)]
    colors = _sample_colors(rng, palette, radius, kind)
    _draw_cargo(inp, path[0], colors)
    for p in path:
        inp[p] = RAIL
    inp[path[0]], inp[path[-1]] = START, GOAL
    inp = np.rot90(inp, int(rng.integers(4)))
    if rng.integers(2):
        inp = np.fliplr(inp)
    return np.array(inp, copy=True)


def _paint_invariants(inp, out, model, events, remaining):
    """Check last-valid shells, no wall changes, conservation and core delivery."""
    paint = np.zeros(inp.shape, dtype=bool)
    material = Counter()
    for center, radius, color in events:
        _require(_fits(model["allowed"], center, radius), "shed shell hits a wall")
        for p in _ring(center, radius):
            _require(not paint[p], "deposited shells overlap")
            paint[p] = True
        material[color] += 8 * radius
    for depth, color in enumerate(model["colors"][model["radius"] - remaining:]):
        radius = remaining - depth
        for p in _ring(model["goal"], radius):
            _require(not paint[p], "final cargo overlaps a deposited shell")
            paint[p] = True
        material[color] += 8 * radius
    _require(not paint[model["goal"]], "a shell covers the delivered core")
    _require(out.shape == inp.shape, "canvas shape changed")
    _require(np.all(out[inp == model["wall"]] == model["wall"]), "wall changed")
    _require(np.count_nonzero(out == START) == 1 and out[model["goal"]] == START,
             "core was not delivered intact")
    _require(not np.any(out == GOAL), "goal marker must be replaced")
    palette = set(model["colors"])
    expected_mass = 4 * model["radius"] * (model["radius"] + 1)
    _require(sum(material.values()) == expected_mass, "frame area is not conserved")
    for color in palette:
        _require(np.count_nonzero(out == color) == material[color],
                 "frame color or multiplicity was lost")
    _require(np.array_equal(np.isin(out, list(palette)), paint), "extra frame paint")
    base = np.array(inp, copy=True)
    r, c = model["start"]
    R = model["radius"]
    base[r - R:r + R + 1, c - R:c + R + 1] = model["floor"]
    for p in model["path"]:
        base[p] = RAIL
    exempt = paint.copy()
    exempt[model["goal"]] = True
    _require(np.array_equal(out[~exempt], base[~exempt]), "unrelated scenery changed")


def _key(a):
    a = np.asarray(a, dtype=np.int64)
    return a.shape, a.tobytes()


def _make_pair(rng, wall, floor, radius, kind, bounds, used,
               room_radius=None, min_turns=0):
    for _ in range(128):
        try:
            inp = _construct_input(rng, wall, floor, radius, kind, bounds,
                                   room_radius, min_turns)
            model = _parse(inp)
            events, remaining = _clearance_history(model)
            out = _render_history(inp, model, events, remaining)
            _paint_invariants(inp, out, model, events, remaining)
            if kind == "peel_evidence":
                _require(remaining == 1 and len({p for p, _, _ in events}) == 2,
                         "evidence must peel at two distinct stations")
            _require(_key(inp) not in used, "duplicate input")
            used.add(_key(inp))
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise ValueError("bounded pair construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    wall, floor = map(int, rng.choice(STRUCTURAL_CHOICES, 2, replace=False))
    regime = int(rng.integers(3))
    used = set()
    train = [
        _make_pair(rng, wall, floor, 3, "clear_evidence", (2, 4), used,
                   room_radius=4),
        _make_pair(rng, wall, floor, 3, "peel_evidence", (3, 4), used,
                   room_radius=4),
    ]
    for _ in range(num_train - 2):
        radius = int(rng.integers(3, 5) if regime == 1 else rng.integers(1, 4))
        train.append(_make_pair(rng, wall, floor, radius, "free", (2, 4), used))
    rng.shuffle(train)
    test = []
    for _ in range(num_test):
        if regime == 0:
            test.append(_make_pair(rng, wall, floor, 4, "free", (2, 4), used))
        elif regime == 1:
            radius = int(rng.integers(1, 3))
            test.append(_make_pair(rng, wall, floor, radius, "free", (2, 12), used))
        else:
            test.append(_make_pair(rng, wall, floor, 3, "free", (5, 9), used,
                                   room_radius=3, min_turns=3))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternatives(inp, model, events, remaining):
    full = _render_history(inp, model, [], model["radius"])
    flat_model = dict(model, colors=(model["colors"][0],) * model["radius"])
    flat_events = [(p, r, flat_model["colors"][0]) for p, r, _ in events]
    flat = _render_history(inp, flat_model, flat_events, remaining)
    reversed_model = dict(model, colors=model["colors"][::-1])
    reversed_events = [(p, r, reversed_model["colors"][model["radius"] - r])
                       for p, r, _ in events]
    reverse = _render_history(inp, reversed_model, reversed_events, remaining)
    at_launch = [(model["start"], r, color) for _, r, color in events]
    global_minimum = _render_history(inp, model, at_launch, remaining)
    compressed = tuple(color for i, color in enumerate(model["colors"])
                       if i == 0 or color != model["colors"][i - 1])
    compressed_model = dict(model, radius=len(compressed), colors=compressed)
    compressed_events, compressed_remaining = _movement_history(compressed_model)
    # Clear the original launch picture, even under this alternative.
    compressed_out = _render_history(inp, model, [], 0)
    for p, r, color in compressed_events:
        _draw_ring(compressed_out, p, r, color)
    _draw_cargo(compressed_out, model["goal"], compressed[len(compressed) - compressed_remaining:])

    # Treat a contiguous same-color band as indivisible when collision occurs.
    radius, band_events = model["radius"], []
    for p, q in zip(model["path"], model["path"][1:]):
        while not _fits(model["allowed"], q, radius):
            color = model["colors"][model["radius"] - radius]
            while radius > 0 and model["colors"][model["radius"] - radius] == color:
                band_events.append((p, radius, color))
                radius -= 1
    whole_band = _render_history(inp, model, band_events, radius)

    # Incorrectly react to every clearance decrease, including passable ones.
    radius, drop_events = model["radius"], []
    capacities = _rail_capacities(model)
    for i, (p, q) in enumerate(zip(model["path"], model["path"][1:]), 1):
        if radius > 0 and capacities[i] < capacities[i - 1]:
            drop_events.append((p, radius, model["colors"][model["radius"] - radius]))
            radius -= 1
        while not _fits(model["allowed"], q, radius):
            drop_events.append((p, radius, model["colors"][model["radius"] - radius]))
            radius -= 1
    every_drop = _render_history(inp, model, drop_events, radius)
    return {"ignore_A": full, "ignore_B": flat, "reverse_layers": reverse,
            "global_minimum_at_launch": global_minimum, "collapse_color_runs": compressed_out,
            "peel_whole_color_band": whole_band, "peel_every_clearance_drop": every_drop}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train/test keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, conventions = set(), set()
    evidence = {name: False for name in ("ignore_A", "ignore_B", "reverse_layers",
                                        "global_minimum_at_launch", "collapse_color_runs",
                                        "peel_whole_color_band", "peel_every_clearance_drop")}
    training_features = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "each pair must have exactly input/output keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "inputs must be distinct")
            seen.add(_key(inp))
            model = _parse(inp)
            conventions.add((model["wall"], model["floor"]))
            events, remaining = _movement_history(model)
            _require((events, remaining) == _clearance_history(model),
                     "footprint and wall-clearance checks disagree")
            expected = solve(inp)
            _require(np.array_equal(out, expected), "incorrect peeling transformation")
            _paint_invariants(inp, out, model, events, remaining)
            if split == "train":
                training_features.append((model, events, remaining))
                for name, alternative in _alternatives(inp, model, events, remaining).items():
                    evidence[name] |= not np.array_equal(out, alternative)
    _require(len(conventions) == 1, "wall/floor roles must stay constant within a puzzle")
    _require(all(evidence.values()), "training leaves a tested shortcut indistinguishable")
    _require(any(not events and min(_rail_capacities(model)) < _rail_capacities(model)[0]
                 for model, events, _ in training_features),
             "training must show intact delivery through a passable narrowing")
    _require(any(len({p for p, _, _ in events}) >= 2 and remaining >= 1
                 for _, events, remaining in training_features),
             "training must show progressive peeling with a surviving frame")


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for s in ("train", "test") for p in puzzle[s])


def _normalized(a):
    roles = {}
    return a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)


def _station_profile(model, events, remaining):
    """Physical layer removals grouped by station, independent of palette."""
    groups, previous = [], None
    for p, radius, _ in events:
        if p != previous:
            groups.append([])
        groups[-1].append(radius)
        previous = p
    return model["radius"], tuple(tuple(g) for g in groups), remaining


def self_test():
    """Bounded fixtures, mutations, all count pairs, and 42-seed diversity."""
    # Explicit straight fixture. Clearances fall 2 -> 1 -> 0; no nominal
    # passage count or generator path metadata determines its expected answer.
    inp = np.full((9, 27), 0, dtype=int)
    inp[1:8, 1:8] = inp[1:8, 11:18] = inp[1:8, 20:26] = 1
    inp[3:6, 8:11] = 1
    inp[4, 18:20] = 1
    _draw_cargo(inp, (4, 4), [5, 6])
    inp[4, 4:24] = RAIL
    inp[4, 4], inp[4, 23] = START, GOAL
    expected = inp.copy()
    expected[2:7, 2:7] = 1
    expected[4, 4:24] = RAIL
    _draw_ring(expected, (4, 5), 2, 5)
    _draw_ring(expected, (4, 16), 1, 6)
    expected[4, 23] = START
    _require(np.array_equal(solve(inp), expected), "explicit two-gate fixture failed")
    for turns in range(4):
        for reflect in (False, True):
            a, b = np.rot90(inp, turns), np.rot90(expected, turns)
            if reflect:
                a, b = np.fliplr(a), np.fliplr(b)
            _require(np.array_equal(solve(a), b), "rotation/reflection changed semantics")

    # A corner of the next footprint hits a wall even though its center and
    # perpendicular cross-section are clear. The peel position is explicit.
    corner = np.full((15, 15), 0, dtype=int)
    corner[1:14, 1:14] = 1
    corner[8, 8] = 0
    _draw_cargo(corner, (4, 4), [5, 6])
    corner[4, 4:11] = RAIL
    corner[4:12, 10] = RAIL
    corner[4, 4], corner[11, 10] = START, GOAL
    corner_expected = corner.copy()
    corner_expected[2:7, 2:7] = 1
    corner_expected[4, 4:11] = RAIL
    _draw_ring(corner_expected, (5, 10), 2, 5)
    _draw_cargo(corner_expected, (11, 10), [6])
    _require(np.array_equal(solve(corner), corner_expected),
             "corner collision fixture failed")

    signatures, structures, shapes, codes, profiles = set(), set(), set(), set(), set()
    station_profiles = set()
    radii, remaining_values, turns_seen, colors_seen = set(), set(), set(), set()
    wall_colors, floor_colors, shifts = set(), set(), set()
    event_counts = Counter()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        signatures.add(_fingerprint(puzzle))
        train_radii, train_turns = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                model = _parse(pair["input"])
                events, remaining = _movement_history(model)
                structures.add(_normalized(pair["input"]))
                shapes.add(pair["input"].shape)
                codes.add(model["colors"])
                profiles.add((model["radius"], tuple((r, c) for _, r, c in events), remaining))
                station_profiles.add(_station_profile(model, events, remaining))
                radii.add(model["radius"])
                remaining_values.add(remaining)
                turns_seen.add(model["turns"])
                colors_seen.update(map(int, np.unique(pair["input"])))
                wall_colors.add(model["wall"])
                floor_colors.add(model["floor"])
                event_counts[len({p for p, _, _ in events})] += 1
                if split == "train":
                    train_radii.append(model["radius"])
                    train_turns.append(model["turns"])
                else:
                    if model["radius"] > max(train_radii):
                        shifts.add("larger")
                    elif model["radius"] < min(train_radii):
                        shifts.add("smaller")
                    elif model["turns"] > max(train_turns):
                        shifts.add("more_turns")
    _require(len(signatures) == 42, "seeds repeated whole puzzles")
    _require(len(structures) >= 150 and len(shapes) >= 50, "geometry variation collapsed")
    _require(len(codes) >= 50 and len(station_profiles) >= 10,
             "layer codes or physical station histories collapsed")
    _require(radii == {1, 2, 3, 4} and remaining_values == {0, 1, 2, 3, 4},
             "cargo scales or outcomes missing")
    _require(0 in turns_seen and max(turns_seen) >= 5, "route turn coverage missing")
    _require(colors_seen == set(range(10)), "literal color coverage missing")
    _require(wall_colors == set(STRUCTURAL_CHOICES)
             and floor_colors == set(STRUCTURAL_CHOICES), "structural palette roles collapsed")
    _require(shifts == {"larger", "smaller", "more_turns"}, "shift regimes missing")
    _require(event_counts[0] >= 42 and event_counts[2] >= 42, "training evidence missing")
    _require(_fingerprint(generate_puzzle(90210, 7, 3))
             == _fingerprint(generate_puzzle(90210, 7, 3)), "nondeterminism")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(1100 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (2.0, 1), (True, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for mutation in ("wall", "core", "frame"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ("train", "test")}
        pair = bad["train"][0]
        model = _parse(pair["input"])
        if mutation == "wall":
            p = (0, 0)
        elif mutation == "core":
            p = model["goal"]
        else:
            p = tuple(np.argwhere(np.isin(pair["output"], model["colors"]))[0])
        pair["output"][p] = model["floor"]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("validator accepted " + mutation + " corruption")
    return {"seeds": 42, "pairs": 210, "distinct_puzzles": len(signatures),
            "normalized_inputs": len(structures), "input_shapes": len(shapes),
            "color_codes": len(codes), "peeling_profiles": len(profiles),
            "station_profiles": len(station_profiles),
            "radii": sorted(radii), "remaining_radii": sorted(remaining_values),
            "turns": sorted(turns_seen), "shift_regimes": sorted(shifts),
            "peeling_stations": dict(sorted(event_counts.items())),
            "independent_reference_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
