"""Keyed conveyor cargo: local D4 instructions change a moving footprint.

No parent code, examples, seed, or generation metadata is used by solve.
Only Python's standard library and NumPy are required.
"""
from __future__ import annotations

import numpy as np


SIDES = ("T", "R", "B", "L")
NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
ADJACENT = [frozenset((SIDES[i], SIDES[(i + 1) % 4])) for i in range(4)]
MATRICES = (
    np.array(((1, 0), (0, 1))),       # identity
    np.array(((0, -1), (1, 0))),      # counterclockwise
    np.array(((-1, 0), (0, -1))),     # half turn
    np.array(((0, 1), (-1, 0))),      # clockwise
    np.array(((1, 0), (0, -1))),      # left/right reflection
    np.array(((-1, 0), (0, 1))),      # top/bottom reflection
    np.array(((0, 1), (1, 0))),       # main diagonal
    np.array(((0, -1), (-1, 0))),     # anti-diagonal
)
SIDE_MAPS = [
    {s: next(t for t in SIDES if tuple(mat @ NORMAL[s]) == NORMAL[t])
     for s in SIDES} for mat in MATRICES
]


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _transform(payload, operation):
    """Apply a signed coordinate map, including the rectangle's full extent."""
    coordinates = MATRICES[operation] @ np.indices(payload.shape).reshape(2, -1)
    coordinates -= coordinates.min(axis=1, keepdims=True)
    result = np.empty(tuple(coordinates.max(axis=1) + 1), dtype=np.int8)
    result[coordinates[0], coordinates[1]] = payload.ravel()
    return result


def _symmetry(guides, controls):
    candidates = [i for i, mapping in enumerate(SIDE_MAPS)
                  if all(controls.get(mapping[s]) == color for s, color in guides.items())]
    _require(len(candidates) == 1, "ambiguous side correspondence")
    return candidates[0]


def _box(center, shape):
    r, c = center
    h, w = shape
    return r - h // 2, c - w // 2, r + h // 2 + 1, c + w // 2 + 1


def _guide_points(center, shape, side):
    r0, c0, r1, c1 = _box(center, shape)
    if side == "T":
        return [(r0 - 1, c) for c in range(c0, c1)]
    if side == "B":
        return [(r1, c) for c in range(c0, c1)]
    if side == "L":
        return [(r, c0 - 1) for r in range(r0, r1)]
    return [(r, c1) for r in range(r0, r1)]


def _ticks(center, radius, side):
    r, c = center
    dr, dc = NORMAL[side]
    return [(r + dr * (radius + 1) + dc * t,
             c + dc * (radius + 1) + dr * t) for t in (-radius, radius)]


def _in_bounds(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, shape):
    for dr, dc in NORMAL.values():
        q = p[0] + dr, p[1] + dc
        if _in_bounds(q, shape):
            yield q


def _decode(input_grid):
    inp = _grid(input_grid)
    starts, goals = np.argwhere(inp == 3), np.argwhere(inp == 5)
    _require(len(starts) == len(goals) == 1, "one cargo and one delivery pivot required")
    start, goal = tuple(map(int, starts[0])), tuple(map(int, goals[0]))
    matches = []
    for h in (3, 5, 7):
        for w in (3, 5, 7):
            bounds = _box(start, (h, w))
            r0, c0, r1, c1 = bounds
            if r0 < 1 or c0 < 1 or r1 >= inp.shape[0] or c1 >= inp.shape[1]:
                continue
            payload = inp[r0:r1, c0:c1].copy()
            colors = set(map(int, np.unique(payload)))
            if 3 not in colors or len(colors - {3}) != 2 or min(colors - {3}) < 6:
                continue
            if np.count_nonzero(payload == 3) != 1:
                continue
            bars = {}
            for side in SIDES:
                values = [int(inp[p]) for p in _guide_points(start, (h, w), side)]
                if len(set(values)) == 1 and values[0] >= 6 and values[0] not in colors:
                    bars[side] = values[0]
            if (frozenset(bars) not in ADJACENT or len(set(bars.values())) != 2
                    or colors | set(bars.values()) != {3, 6, 7, 8, 9}):
                continue
            matches.append((payload, bounds, bars))
    _require(len(matches) == 1, "source guides must define one centered rectangle")
    payload, bounds, guides = matches[0]
    r0, c0, r1, c1 = bounds

    # The departure midpoint uniquely reveals the rail hidden under the tile.
    exits = []
    for side in SIDES:
        dr, dc = NORMAL[side]
        distance = (payload.shape[0] if dr else payload.shape[1]) // 2 + 1
        p = start[0] + dr * distance, start[1] + dc * distance
        if _in_bounds(p, inp.shape) and inp[p] == 2:
            exits.append(side)
    _require(len(exits) == 1 and exits[0] not in guides, "source needs one unkeyed departure")
    dr, dc = NORMAL[exits[0]]
    hidden_length = (payload.shape[0] if dr else payload.shape[1]) // 2
    hidden = [(start[0] + dr * t, start[1] + dc * t) for t in range(hidden_length + 1)]
    terrain = inp.copy()
    terrain[r0:r1, c0:c1] = 1
    for p in hidden:
        terrain[p] = 2
    rail = np.isin(terrain, (2, 4, 5))
    points = set(map(tuple, np.argwhere(rail)))
    for p in points:
        degree = sum(q in points for q in _neighbors(p, inp.shape))
        _require(degree == (1 if p in (start, goal) else 2), "rail is not a simple pivot path")
    route, previous, current = [start], None, start
    while current != goal:
        options = [q for q in _neighbors(current, inp.shape) if q in points and q != previous]
        _require(len(options) == 1, "rail continuation is ambiguous")
        previous, current = current, options[0]
        _require(current not in route, "rail has a loop")
        route.append(current)
    _require(len(route) == len(points), "disconnected rail")
    marker_colors = set(guides.values())
    radius = (max(payload.shape) + 1) // 2
    commands, controls_by_pivot = {}, {}
    expected_markers = set()
    for side in guides:
        expected_markers.update(_guide_points(start, payload.shape, side))
    stations = [p for p in route if terrain[p] == 4]
    _require(stations, "at least one orientation station required")
    for p in stations:
        controls = {}
        for side in SIDES:
            cells = _ticks(p, radius, side)
            _require(all(_in_bounds(q, inp.shape) for q in cells), "key extends past canvas")
            values = [int(inp[q]) for q in cells]
            if values[0] == values[1] and values[0] in marker_colors:
                controls[side] = values[0]
                expected_markers.update(cells)
            else:
                _require(values == [0, 0], "broken or extraneous station key")
        _require(frozenset(controls) in ADJACENT and set(controls.values()) == marker_colors,
                 "station requires two distinct adjacent identities")
        br0, bc0, br1, bc1 = _box(p, (2 * radius + 1, 2 * radius + 1))
        _require(br0 >= 0 and bc0 >= 0 and br1 <= inp.shape[0] and bc1 <= inp.shape[1],
                 "station room extends past canvas")
        _require(bool(np.all(np.isin(terrain[br0:br1, bc0:bc1], (1, 2, 4, 5)))),
                 "station is not a clear reorientation room")
        commands[p] = _symmetry(guides, controls)
        controls_by_pivot[p] = controls
    actual_markers = set(map(tuple, np.argwhere(np.isin(inp, tuple(marker_colors)))))
    _require(actual_markers == expected_markers, "marker ink outside guides and station keys")
    cargo_colors = set(map(int, np.unique(payload))) - {3}
    outside = np.ones(inp.shape, dtype=bool)
    outside[r0:r1, c0:c1] = False
    _require(not np.any(np.isin(inp[outside], tuple(cargo_colors))), "cargo colors outside source")
    return dict(input=inp, terrain=terrain, payload=payload, bounds=bounds, guides=guides,
                start=start, goal=goal, route=route, commands=commands,
                controls=controls_by_pivot, radius=radius, hidden=hidden)


def _fits(terrain, center, shape):
    r0, c0, r1, c1 = _box(center, shape)
    return (r0 >= 0 and c0 >= 0 and r1 <= terrain.shape[0] and c1 <= terrain.shape[1]
            and bool(np.all(np.isin(terrain[r0:r1, c0:c1], (1, 2, 4, 5)))))


def _span(terrain, center, heading):
    """Actual floor cross-section through the rail, perpendicular to travel."""
    pr, pc = heading[1], heading[0]
    lengths = []
    for sign in (-1, 1):
        length = 0
        q = center[0] + sign * pr, center[1] + sign * pc
        while _in_bounds(q, terrain.shape) and terrain[q] in (1, 2, 4, 5):
            length += 1
            q = q[0] + sign * pr, q[1] + sign * pc
        lengths.append(length)
    return tuple(lengths)


def _simulate(info, *, ignore_keys=False, ignore_collisions=False):
    terrain, payload = info["terrain"], info["payload"]
    oriented, operation = payload, 0
    destination = info["start"]
    visited, events, failed = [], [], None
    offset_collision = False
    for step, p in enumerate(info["route"][1:], 1):
        if not ignore_collisions and not _fits(terrain, p, oriented.shape):
            failed = p
            old = info["route"][step - 1]
            heading = p[0] - old[0], p[1] - old[1]
            longitudinal = oriented.shape[0 if heading[0] else 1] // 2
            leading = p[0] + heading[0] * longitudinal, p[1] + heading[1] * longitudinal
            if _in_bounds(leading, terrain.shape) and terrain[leading] in (1, 2, 4, 5):
                left, right = _span(terrain, leading, heading)
                required = oriented.shape[1 if heading[0] else 0]
                offset_collision = left + right + 1 >= required
            break
        destination = p
        if p in info["commands"]:
            visited.append(p)
            if not ignore_keys:
                operation = info["commands"][p]
                oriented = _transform(payload, operation)
                _require(_fits(terrain, p, oriented.shape), "keyed orientation does not fit its room")
                events.append((p, operation))
    out = terrain.copy()
    r0, c0, r1, c1 = _box(destination, oriented.shape)
    _require(r0 >= 0 and c0 >= 0 and r1 <= out.shape[0] and c1 <= out.shape[1],
             "final cargo extends past canvas")
    out[r0:r1, c0:c1] = oriented
    return out, dict(destination=destination, operation=operation, visited=visited, events=events,
                     failed=failed, travel=info["route"].index(destination),
                     success=destination == info["goal"], offset_collision=offset_collision,
                     final_payload=oriented)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode and execute the visible rail and local keys, without latent state."""
    return _simulate(_decode(input_grid))[0]


def _payload(rng, shape, fill, accent):
    center = shape[0] // 2, shape[1] // 2
    positions = [tuple(map(int, p)) for p in np.ndindex(shape) if tuple(p) != center]
    for _ in range(32):
        a = np.full(shape, fill, dtype=np.int8)
        count = int(rng.integers(2 if shape[0] == shape[1] else 1,
                                 max(3, len(positions) // 3 + 1)))
        chosen = [positions[int(i)] for i in rng.permutation(len(positions))[:count]]
        if rng.random() < .55:
            chosen = [chosen[0]]
            for _ in range(count - 1):
                choices = [q for p in chosen for q in _neighbors(p, shape)
                           if q != center and q not in chosen]
                if choices:
                    chosen.append(_pick(rng, choices))
        for p in chosen:
            a[p] = accent
        a[center] = 3
        if len({_key(_transform(a, i)) for i in range(8)}) == 8:
            return a
    # No protected corner code in sampled motifs; this finite fallback only
    # repairs a residual symmetry after bounded motif attempts.
    a = np.full(shape, fill, dtype=np.int8)
    a[0, 0] = a[0, 1] = accent
    a[center] = 3
    _require(len({_key(_transform(a, i)) for i in range(8)}) == 8, "asymmetry repair failed")
    return a


def _sample_walk(rng, m, count, *, forced_guides=None, random_attempts=128):
    maximum_axis = 4 if m == 3 else 3 if m == 5 else 2
    layouts = [(nr, nc) for nr in range(1, maximum_axis + 1)
               for nc in range(1, maximum_axis + 1)
               if nr * nc >= count and (nr > 1 or nc > 1)]
    for attempt in range(random_attempts + 1):
        if attempt == random_attempts:
            # Finite fallback for unlucky long-walk sampling. It changes
            # construction geometry only; decoding and movement are identical.
            n = int(np.ceil(np.sqrt(count)))
            _require(n <= maximum_axis, "too many rooms for cargo scale")
            walk = [(r, c) for r in range(n)
                    for c in (range(n) if r % 2 == 0 else range(n - 1, -1, -1))][:count]
            eligible = [i for i in range(8) if forced_guides is None
                        or SIDE_MAPS[i]["R"] not in ADJACENT[forced_guides]]
            operation = _pick(rng, eligible)
            walk = [tuple(map(int, MATRICES[operation] @ p)) for p in walk]
        else:
            nr, nc = _pick(rng, layouts)
            current = int(rng.integers(nr)), int(rng.integers(nc))
            walk, used = [current], {current}
            while len(walk) < count:
                options = [p for p in _neighbors(walk[-1], (nr, nc)) if p not in used]
                if not options:
                    break
                current = _pick(rng, options)
                walk.append(current)
                used.add(current)
            if len(walk) != count:
                continue
        heading = tuple(np.subtract(walk[1], walk[0]))
        exit_side = next(s for s in SIDES if NORMAL[s] == heading)
        if forced_guides is not None and exit_side in ADJACENT[forced_guides]:
            continue
        # Crop unused lattice rows/columns, then space the occupied walk.
        lo = np.min(walk, axis=0)
        walk = [tuple(np.subtract(p, lo)) for p in walk]
        dimensions = tuple(np.max(walk, axis=0) + 1)
        radius, q = (m + 1) // 2, m + 2
        axes, totals = [], []
        for n in dimensions:
            n = int(n)
            gaps = [2] * (n - 1)
            budget = 30 - (n * q + sum(gaps) + 2)
            for i in rng.permutation(n - 1):
                extra = int(rng.integers(min(4, budget) + 1))
                gaps[int(i)] += extra
                budget -= extra
            extent = n * q + sum(gaps) + 2
            padding = int(rng.integers(min(6, 30 - extent) + 1))
            before = int(rng.integers(padding + 1))
            coords = [radius + 1 + before]
            for gap in gaps:
                coords.append(coords[-1] + q + gap)
            axes.append(coords)
            totals.append(extent + padding)
        centers = [(axes[0][int(r)], axes[1][int(c)]) for r, c in walk]
        return centers, tuple(totals), exit_side
    raise AssertionError("bounded room walk construction exhausted")


def _expanded_route(centers):
    result = [centers[0]]
    for a, b in zip(centers, centers[1:]):
        dr, dc = np.sign(np.subtract(b, a))
        current = a
        while current != b:
            current = current[0] + int(dr), current[1] + int(dc)
            result.append(current)
    return result


def _intervals(m, width):
    half = m // 2
    return [(left, left + width - 1) for left in range(-half, 1)
            if 0 <= left + width - 1 <= half]


def _passage(rng, m, transverse, kind):
    options = [(width, interval) for width in range(1, m + 1)
               for interval in _intervals(m, width)
               if ((interval[0] <= -(transverse // 2) and interval[1] >= transverse // 2)
                   if kind == "pass" else
                   (width >= transverse and not (interval[0] <= -(transverse // 2)
                                                 and interval[1] >= transverse // 2))
                   if kind == "offset" else width < transverse)]
    _require(options, "no compatible passage interval")
    return _pick(rng, options)


def _render(rng, payload, centers, canvas, guides, operations, intervals):
    m, start = max(payload.shape), centers[0]
    radius = (m + 1) // 2
    terrain = np.zeros(canvas, dtype=np.int8)
    for p in centers:
        r0, c0, r1, c1 = _box(p, (2 * radius + 1, 2 * radius + 1))
        terrain[r0:r1, c0:c1] = 1
    for a, b, (_, (low, high)) in zip(centers, centers[1:], intervals):
        if a[0] == b[0]:
            c0, c1 = sorted((a[1], b[1]))
            terrain[a[0] + low:a[0] + high + 1, c0:c1 + 1] = 1
        else:
            r0, r1 = sorted((a[0], b[0]))
            terrain[r0:r1 + 1, a[1] + low:a[1] + high + 1] = 1
    route = _expanded_route(centers)
    for p in route:
        terrain[p] = 2
    terrain[centers[-1]] = 5
    commands = dict(zip(centers[1:-1], operations))
    controls = {}
    for p, operation in commands.items():
        terrain[p] = 4
        controls[p] = {SIDE_MAPS[operation][s]: color for s, color in guides.items()}
        for side, color in controls[p].items():
            for cell in _ticks(p, radius, side):
                _require(terrain[cell] == 0, "station key collides with passage")
                terrain[cell] = color
    for side, color in guides.items():
        for p in _guide_points(start, payload.shape, side):
            _require(terrain[p] == 1, "source bar collides with rail")
            terrain[p] = color
    r0, c0, r1, c1 = _box(start, payload.shape)
    inp = terrain.copy()
    inp[r0:r1, c0:c1] = payload
    construction = dict(terrain=terrain, payload=payload, start=start, goal=centers[-1],
                        route=route, commands=commands)
    output, details = _simulate(construction)
    decoded = _decode(inp)
    _require(decoded["route"] == route and decoded["commands"] == commands,
             "visible input disagrees with construction")
    _require(np.array_equal(solve(inp), output), "input-only prediction disagrees")
    return {"input": inp, "output": output}, details


def _guides(rng, shape, exit_side, palette, *, forbidden=None, forced=None):
    candidates = [i for i, sides in enumerate(ADJACENT)
                  if exit_side not in sides and i != forbidden and (forced is None or i == forced)]
    _require(candidates, "no compatible source guide sides")
    corner = _pick(rng, candidates)
    sides = [s for s in SIDES if s in ADJACENT[corner]]
    identities = palette[:2] if bool(rng.integers(2)) else palette[:2][::-1]
    return dict(zip(sides, map(int, identities)))


def _transverse(shape, a, b):
    return shape[1 if a[0] != b[0] else 0]


def _contrast(rng, palette, forbidden=None):
    shape = _pick(rng, ((3, 5), (5, 3)))
    payload = _payload(rng, shape, int(palette[2]), int(palette[3]))
    centers, canvas, exit_side = _sample_walk(rng, 5, 4)
    guides = _guides(rng, shape, exit_side, palette, forbidden=forbidden)
    pass_ops = [i for i in range(1, 8)
                if _transverse(_transform(payload, i).shape, centers[1], centers[2]) == 3]
    first = _pick(rng, pass_ops)
    controls = {SIDE_MAPS[first][s]: c for s, c in guides.items()}
    swapped = dict(zip(controls, reversed(list(controls.values()))))
    blocked_first = _symmetry(guides, swapped)
    second = _pick(rng, [i for i in (1, 3) if i not in (first, blocked_first)])
    operations = [first, second]
    dimensions = [payload.shape] + [_transform(payload, i).shape for i in operations]
    intervals = []
    for j, (a, b) in enumerate(zip(centers, centers[1:])):
        transverse = _transverse(dimensions[j], a, b)
        if j == 1:
            # The same 3- or 4-cell throat passes 3 and blocks 5.
            width = int(rng.integers(3, 5))
            options = [interval for interval in _intervals(5, width)
                       if interval[0] <= -1 and interval[1] >= 1]
            intervals.append((width, _pick(rng, options)))
        else:
            intervals.append(_passage(rng, 5, transverse, "pass"))
    passing, pd = _render(rng, payload, centers, canvas, guides, operations, intervals)
    blocked, bd = _render(rng, payload, centers, canvas, guides, [blocked_first, second], intervals)
    _require(pd["success"] and not bd["success"] and len(bd["visited"]) == 1,
             "teaching contrast lacks conditional passage")
    _require(bd["operation"] != second and bd["travel"] > 0,
             "later instruction is not visibly suppressed")
    return [passing, blocked]


def _random_pair(rng, palette, shape, count, outcome, *, forbidden=None, forced=None,
                 offset=False, reached=1):
    m = max(shape)
    payload = _payload(rng, shape, int(palette[2]), int(palette[3]))
    centers, canvas, exit_side = _sample_walk(rng, m, count, forced_guides=forced)
    guides = _guides(rng, shape, exit_side, palette, forbidden=forbidden, forced=forced)
    operations = [int(rng.integers(8)) for _ in centers[1:-1]]
    block_edge = len(centers) if outcome else int(rng.integers(reached, len(centers) - 1))
    if offset:
        # An adequate-width throat fails only because the pivot hugs an edge.
        block_edge = reached
        operations[block_edge - 1] = _pick(rng, [i for i in range(8)
                                  if _transverse(_transform(payload, i).shape,
                                                 centers[block_edge], centers[block_edge + 1]) < m])
    dimensions = [payload.shape] + [_transform(payload, i).shape for i in operations]
    intervals = []
    for j, (a, b) in enumerate(zip(centers, centers[1:])):
        transverse = _transverse(dimensions[j], a, b)
        kind = "pass" if j < block_edge else "offset" if j == block_edge and offset else "block"
        if j > block_edge:
            width = int(rng.integers(1, m + 1))
            intervals.append((width, _pick(rng, _intervals(m, width))))
        else:
            intervals.append(_passage(rng, m, transverse, kind))
    pair, details = _render(rng, payload, centers, canvas, guides, operations, intervals)
    _require(details["success"] == outcome and details["travel"] > 0, "wrong movement outcome")
    if offset:
        _require(details["offset_collision"], "missing off-center teaching collision")
    _require(len(details["visited"]) >= reached, "test needs a reached relay")
    return pair


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(np.arange(6, 10))))
    stable = bool(rng.random() < .3)
    regime = int(rng.integers(3))
    held = int(rng.integers(4)) if regime == 2 else None
    train = _contrast(rng, palette, forbidden=held)
    used = {_key(p["input"]) for p in train}

    def fresh(split, index):
        for attempt in range(48):
            local_palette = palette if stable else tuple(map(int, rng.permutation(np.arange(6, 10))))
            if split == "train":
                is_offset = index == 2
                shape = _pick(rng, ((3, 5), (5, 3))) if is_offset else _pick(
                    rng, ((3, 3), (3, 5), (5, 3), (5, 5)))
                count = int(rng.integers(3, 5 if regime != 2 else 7))
                success = False if is_offset else True if index == 3 else bool(rng.integers(2))
                pair = _random_pair(rng, local_palette, shape, count, success,
                                    forbidden=held, offset=is_offset)
            else:
                shape = (_pick(rng, ((3, 7), (7, 3), (5, 7), (7, 5), (7, 7))) if regime == 1
                         else _pick(rng, ((3, 5), (5, 3), (5, 5))))
                count = 4 if regime == 1 else int(rng.integers(5, 10))
                success = bool(rng.integers(2))
                # Alignment generalization is tested only after its witness.
                # First test always reaches at least two stations.
                is_offset = num_train >= 3 and not success and rng.random() < .35
                if is_offset and shape[0] == shape[1]:
                    shape = (3, max(shape))
                pair = _random_pair(rng, local_palette, shape, count, success,
                                    forced=held, offset=is_offset, reached=2)
            if _key(pair["input"]) not in used:
                used.add(_key(pair["input"]))
                return pair
        raise AssertionError("distinct pair construction exhausted")

    for index in range(2, num_train):
        train.append(fresh("train", index))
    train = [train[int(i)] for i in rng.permutation(num_train)]
    puzzle = {"train": train, "test": [fresh("test", i) for i in range(num_test)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, details):
    centers = [info["start"]] + [p for p in info["route"] if info["terrain"][p] in (4, 5)]
    spans, headings = [], []
    for a, b in zip(centers, centers[1:]):
        heading = tuple(map(int, np.sign(np.subtract(b, a))))
        middle = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        left, right = _span(info["terrain"], middle, heading)
        spans.append((left + right + 1, left, right))
        headings.append(heading)
    return dict(shape=info["input"].shape, cargo_shape=info["payload"].shape,
                keys=len(info["commands"]), visited=len(details["visited"]),
                operations=tuple(info["commands"].values()),
                active=tuple(op for _, op in details["events"]),
                widths=tuple(w[0] for w in spans), offsets=tuple((w[1], w[2]) for w in spans),
                turns=sum(a != b for a, b in zip(headings, headings[1:])),
                heading=headings[0], guides=frozenset(info["guides"]),
                success=details["success"], travel=details["travel"],
                offset_collision=details["offset_collision"])


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _counts(n, m)
    seen, rows = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "scene dimensions changed")
            signature = _key(inp)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            info = _decode(inp)
            expected, details = _simulate(info)
            _require(np.array_equal(out, expected), "incorrect motion, orientation, or scenery")
            # Explicit payload conservation and unchanged-scene checks.
            r0, c0, r1, c1 = _box(details["destination"], details["final_payload"].shape)
            _require(np.array_equal(np.sort(out[r0:r1, c0:c1].ravel()),
                                   np.sort(info["payload"].ravel())), "cargo cells not conserved")
            untouched = np.ones(inp.shape, dtype=bool)
            a, b, c, d = info["bounds"]
            untouched[a:c, b:d] = False
            untouched[r0:r1, c0:c1] = False
            _require(np.array_equal(inp[untouched], out[untouched]), "unrelated scenery changed")
            if not details["success"]:
                _require(out[info["goal"]] == 5 and details["failed"] is not None,
                         "blocked cargo consumed delivery")
                _require(not _fits(info["terrain"], details["failed"], details["final_payload"].shape),
                         "cargo stopped before a legal move")
            rows[split].append((info, details, _features(info, details)))
    training = rows["train"]
    _require(any(d["success"] for _, d, _ in training), "training needs delivery")
    _require(any(not d["success"] and d["travel"] > 0 for _, d, _ in training),
             "training needs a moving collision")
    _require(any(not np.array_equal(_simulate(i, ignore_keys=True)[0], _simulate(i)[0])
                 for i, _, _ in training), "omitting B does not change training")
    _require(any(not d["success"] and not np.array_equal(_simulate(i, ignore_collisions=True)[0],
                                                        _simulate(i)[0])
                 for i, d, _ in training), "omitting A does not change training")
    contrasts = []
    for index, (a, ad, _) in enumerate(training):
        for b, bd, _ in training[index + 1:]:
            if a["input"].shape != b["input"].shape or ad["success"] == bd["success"]:
                continue
            if (not np.array_equal(a["payload"], b["payload"]) or a["guides"] != b["guides"]
                    or a["route"] != b["route"]):
                continue
            changed = set(map(tuple, np.argwhere(a["input"] != b["input"])))
            for p in a["commands"]:
                ticks = {q for side in a["controls"][p] for q in _ticks(p, a["radius"], side)}
                if changed == ticks and set(a["controls"][p]) == set(b["controls"][p]):
                    contrasts.append((a, ad, b, bd))
    _require(contrasts, "missing exact station-identity teaching contrast")
    a, ad, b, bd = contrasts[0]
    stopped_i, stopped_d = (a, ad) if not ad["success"] else (b, bd)
    _require(len(stopped_d["visited"]) == 1 and len(stopped_i["commands"]) >= 2,
             "contrast must suppress a later instruction")
    _require(any(op != stopped_d["operation"] for p, op in stopped_i["commands"].items()
                 if p not in stopped_d["visited"]), "later key is not distinguishable")
    _require(any(op in (1, 3) for _, d, _ in training for _, op in d["events"]),
             "training needs an asymmetric reached quarter turn")
    if n >= 3:
        _require(any(d["offset_collision"] for _, d, _ in training), "missing rail-alignment witness")
    train_shapes = [max(i["payload"].shape) for i, _, _ in training]
    train_keys = [len(i["commands"]) for i, _, _ in training]
    train_guides = {frozenset(i["guides"]) for i, _, _ in training}
    for info, details, _ in rows["test"]:
        _require(max(info["payload"].shape) > max(train_shapes)
                 or len(info["commands"]) > max(train_keys)
                 or frozenset(info["guides"]) not in train_guides, "missing documented test shift")
        _require(not details["offset_collision"] or any(d["offset_collision"] for _, d, _ in training),
                 "off-center test lacks teaching evidence")


def self_test() -> None:
    puzzles, records = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                expected, details = _simulate(info)
                _require(np.array_equal(expected, pair["output"]), "42-seed solver disagreement")
                records.append(dict(_features(info, details), split=split))
        if seed < 3:
            repeated = generate_puzzle(seed)
            _require(all(_key(p[k]) == _key(q[k]) for s in puzzle
                         for p, q in zip(puzzle[s], repeated[s]) for k in p), "nondeterminism")
    _require(len({tuple(_key(p["input"]) for s in puzzle for p in puzzle[s])
                  for puzzle in puzzles}) == 42, "whole-puzzle repetition")
    _require(len({r["shape"] for r in records}) >= 35, "too few scene dimensions")
    _require({i for r in records for i in r["operations"]} == set(range(8)), "missing D4 controls")
    _require({i for r in records for i in r["active"]} == set(range(8)), "missing reached D4 controls")
    _require({r["heading"] for r in records} == set(NORMAL.values()), "missing departure direction")
    _require({r["guides"] for r in records} == set(ADJACENT), "missing source guide corner")
    _require(max(r["keys"] for r in records) >= 5 and max(r["turns"] for r in records) >= 4,
             "room walks lack structural variation")
    _require(any(r["turns"] == 0 for r in records), "straight relay missing")
    _require({r["success"] for r in records if r["split"] == "test"} == {False, True},
             "fixed test outcome")
    _require(len({r["widths"] for r in records}) >= 60, "width profiles lack diversity")
    _require(any(max(r["cargo_shape"]) == 7 for r in records), "larger cargo missing")
    _require(any(r["cargo_shape"][0] == r["cargo_shape"][1] for r in records), "square cargo missing")
    _require(any(r["offset_collision"] for r in records if r["split"] == "test"),
             "off-center generalization missing")

    # Direct coordinate fixtures check all eight names against NumPy operations.
    numbered = np.arange(15, dtype=np.int8).reshape(3, 5)
    expected = (numbered, np.rot90(numbered), np.rot90(numbered, 2), np.rot90(numbered, 3),
                np.fliplr(numbered), np.flipud(numbered), numbered.T, np.rot90(numbered.T, 2))
    for i, answer in enumerate(expected):
        _require(np.array_equal(_transform(numbered, i), answer), "signed symmetry coordinate error")
    # Explicit physical coordinates: a 3x5 tile crosses width 3, but a 5x3
    # orientation cannot; width 3 also blocks the small tile on an edge rail.
    terrain = np.zeros((11, 23), dtype=np.int8)
    terrain[2:9, 1:8] = terrain[2:9, 15:22] = 1
    terrain[4:7, 8:15] = 1
    _require(_fits(terrain, (5, 10), (3, 5)), "small transverse fixture failed")
    _require(not _fits(terrain, (5, 10), (5, 3)), "large transverse fixture passed")
    terrain[4:7, 8:15] = 0
    terrain[5:8, 8:15] = 1
    _require(not _fits(terrain, (5, 10), (3, 5)), "pivot alignment fixture passed")

    # Hand-set relay coordinates distinguish absolute instructions from
    # composing operations, and prove an unreachable later key is inert.
    body = np.full((3, 5), 8, dtype=np.int8)
    body[0, 1] = body[2, 4] = 9
    body[1, 2] = 3
    rail = [(5, c) for c in range(3, 26)]
    clear = np.ones((11, 29), dtype=np.int8)
    clear[5, 3:26] = 2
    clear[5, 9] = clear[5, 17] = 4
    clear[5, 25] = 5
    fixture = dict(terrain=clear, payload=body, start=(5, 3), goal=(5, 25),
                   route=rail, commands={(5, 9): 1, (5, 17): 4})
    answer, details = _simulate(fixture)
    expected = clear.copy()
    expected[4:7, 23:28] = np.fliplr(body)
    _require(np.array_equal(answer, expected) and details["visited"] == [(5, 9), (5, 17)],
             "absolute relay fixture failed")
    throat = np.zeros((11, 29), dtype=np.int8)
    throat[2:9, 1:15] = throat[2:9, 19:28] = 1
    throat[4:7, 15:19] = 1
    throat[5, 3:26] = 2
    throat[5, 11] = throat[5, 22] = 4
    throat[5, 25] = 5
    fixture.update(terrain=throat, commands={(5, 11): 1, (5, 22): 4})
    answer, details = _simulate(fixture)
    expected = throat.copy()
    expected[3:8, 12:15] = np.rot90(body)
    _require(np.array_equal(answer, expected) and details["destination"] == (5, 13)
             and details["visited"] == [(5, 11)] and answer[5, 25] == 5,
             "first collision / unreached instruction fixture failed")

    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(611 + 13 * a + b, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "explicit counts ignored")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")

    # Exercise the finite long-walk fallback directly, including every held
    # source-guide corner. The same input-only decoder verifies these scenes.
    fallback_rng = np.random.default_rng(901)
    for shape, count in (((3, 3), 6), ((3, 5), 9), ((5, 7), 4)):
        for corner in range(4):
            centers, canvas, exit_side = _sample_walk(fallback_rng, max(shape), count,
                                                      forced_guides=corner, random_attempts=0)
            guides = _guides(fallback_rng, shape, exit_side, (6, 7, 8, 9), forced=corner)
            body = _payload(fallback_rng, shape, 8, 9)
            operations = [int(fallback_rng.integers(8)) for _ in centers[1:-1]]
            dimensions = [shape] + [_transform(body, i).shape for i in operations]
            intervals = [_passage(fallback_rng, max(shape), _transverse(d, a, b), "pass")
                         for d, a, b in zip(dimensions, centers, centers[1:])]
            pair, details = _render(fallback_rng, body, centers, canvas, guides, operations, intervals)
            _require(details["success"] and np.array_equal(solve(pair["input"]), pair["output"]),
                     "finite fallback violates conveyor rule")

    original = puzzles[0]
    for kind in ("output_cell", "rail", "key", "guide"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        if kind == "output_cell":
            pair["output"][0, 0] = (int(pair["output"][0, 0]) + 1) % 10
        elif kind == "rail":
            cell = next(p for p in info["route"] if pair["input"][p] == 2)
            pair["input"][cell] = 1
        elif kind == "key":
            p = next(iter(info["commands"]))
            side = next(iter(info["controls"][p]))
            pair["input"][_ticks(p, info["radius"], side)[0]] = 0
        else:
            side = next(iter(info["guides"]))
            pair["input"][_guide_points(info["start"], info["payload"].shape, side)[0]] = 1
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("validator accepted " + kind + " mutation")

    # Removing the alignment teaching examples must not legitimize an
    # off-center test, even when the controlled contrast remains intact.
    witness = next(p for p in puzzles
                   if _simulate(_decode(p["test"][0]["input"]))[1]["offset_collision"])
    contrast = None
    for i, a in enumerate(witness["train"]):
        for b in witness["train"][i + 1:]:
            if a["input"].shape == b["input"].shape and np.count_nonzero(a["input"] != b["input"]) == 4:
                contrast = [a, b]
    _require(contrast is not None, "alignment negative fixture lacks a controlled contrast")
    try:
        validate_generated_puzzle({"train": contrast, "test": witness["test"]})
    except AssertionError:
        pass
    else:
        raise AssertionError("off-center test accepted without teaching evidence")


if __name__ == "__main__":
    self_test()
    print("Keyed conveyor self-test passed (42 seeds and all legal counts).")
