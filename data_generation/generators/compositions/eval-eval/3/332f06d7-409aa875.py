"""Resonant Sweep: footprint reachability and endpoint locks form a fixed point.

Only standard-library modules and NumPy are used. No parent data, construction
metadata, example labels, seed, or learned palette is available to solve().
"""
from collections import Counter, deque
from itertools import combinations

import numpy as np


DISTANCE = 5
SINGLE = 6
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid colors")
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape, diagonal=False):
    r, c = p
    directions = DIRECTIONS if diagonal else DIRECTIONS[:4]
    for dr, dc in directions:
        q = r + dr, c + dc
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], {start}
        while todo:
            for q in _neighbors(todo.pop(), mask.shape, diagonal):
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    todo.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _arrow(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return (tip, direction, tuple(sorted((r + a, c + b) for a, b in offsets)))


def _endpoint(arrow):
    (r, c), (dr, dc), _ = arrow
    return r + DISTANCE * dr, c + DISTANCE * dc


def _parse_arrows(mask):
    arrows = []
    for component in _components(mask):
        _require(len(component) == 3, "arrow components must be separate triominoes")
        matches = [_arrow(tip, direction) for tip in component for direction in DIRECTIONS
                   if _arrow(tip, direction)[2] == component]
        _require(len(matches) == 1, "ambiguous arrowhead")
        arrows.append(matches[0])
    _require(bool(arrows), "missing arrows")
    return arrows


def _parse_gates(mask):
    gates = _components(mask)
    _require(bool(gates), "missing shutters")
    for cells in gates:
        rows, cols = zip(*cells)
        height, width = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
        _require(min(height, width) == 1 and 2 <= len(cells) <= 6,
                 "shutter must be a straight bar of length 2..6")
        _require(height * width == len(cells), "shutter has a hole")
    return gates


def _parse_square(mask):
    points = np.argwhere(mask)
    _require(len(points) > 0, "missing brush")
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    size = int(hi[0] - lo[0])
    _require(size >= 2 and hi[1] - lo[1] == size and len(points) == size * size,
             "brush is not one solid square of side at least two")
    return tuple(map(int, lo)), size


def _decode(grid):
    grid = _grid(grid)
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    _require(len(np.unique(border)) == 1, "a uniform wall frame is required")
    wall = int(border[0])
    colors = set(map(int, np.unique(grid)))
    _require(len(colors) == 5 and SINGLE not in colors, "input needs five roles excluding color 6")
    others = colors - {wall}
    counts = Counter(map(int, grid.ravel()))
    floor = max(others, key=lambda c: counts[c])
    _require(all(counts[floor] > counts[c] for c in others - {floor}),
             "floor must be the unique most common nonwall role")
    roles = {}
    for color in others - {floor}:
        for role, parser in (("brush", _parse_square), ("arrows", _parse_arrows),
                             ("gates", _parse_gates)):
            try:
                value = parser(grid == color)
            except AssertionError:
                continue
            _require(role not in roles, "ambiguous marker typing")
            roles[role] = color, value
    _require(set(roles) == {"brush", "arrows", "gates"}, "marker roles are not identifiable")
    _require(len({v[0] for v in roles.values()}) == 3, "marker roles overlap")
    brush, (start, k) = roles["brush"]
    arrow_color, arrows = roles["arrows"]
    gate_color, gates = roles["gates"]
    base = (grid != wall) & (grid != gate_color)
    return dict(grid=grid, wall=wall, floor=floor, brush=brush, start=start, k=k,
                arrow_color=arrow_color, arrows=arrows, gate_color=gate_color,
                gates=gates, base=base)


def _sweep(allowed, start, k):
    """Orthogonal configuration-space reachability, then footprint union."""
    if k > min(allowed.shape):
        return np.zeros(allowed.shape, bool), np.zeros((0, 0), bool)
    valid = np.lib.stride_tricks.sliding_window_view(allowed, (k, k)).all(axis=(-1, -2))
    reach = np.zeros(valid.shape, bool)
    _require(valid[start], "brush starts in an invalid placement")
    reach[start] = True
    queue = deque([start])
    while queue:
        for q in _neighbors(queue.popleft(), valid.shape):
            if valid[q] and not reach[q]:
                reach[q] = True
                queue.append(q)
    swept = np.zeros(allowed.shape, bool)
    for r, c in np.argwhere(reach):
        swept[r:r + k, c:c + k] = True
    return swept, reach


def _run(info, *, point=False, all_active=False, object_count=False, no_shots=False):
    """Monotone least fixed point. Flags are explicit shortcut checks only."""
    k = 1 if point else info["k"]
    opened, active = set(), set()
    opening_waves = []
    initial = None
    # Every nonterminal iteration opens at least one new shutter.
    for _ in range(len(info["gates"]) + 1):
        allowed = info["base"].copy()
        for i in opened:
            for p in info["gates"][i]:
                allowed[p] = True
        swept, reach = _sweep(allowed, info["start"], k)
        if initial is None:
            initial = swept.copy()
        active = set() if no_shots else {
            i for i, arrow in enumerate(info["arrows"]) if all_active or swept[arrow[0]]}
        counts = Counter(_endpoint(info["arrows"][i]) for i in active)
        newly = set()
        for i, gate in enumerate(info["gates"]):
            resonant = (sum(counts[p] for p in gate) >= 2 if object_count
                        else any(counts[p] >= 2 for p in gate))
            if resonant and i not in opened:
                newly.add(i)
        if not newly:
            return dict(opened=opened, active=active, counts=counts, swept=swept,
                        reach=reach, initial=initial, opening_waves=opening_waves)
        opened |= newly
        opening_waves.append(tuple(sorted(newly)))
    raise AssertionError("fixed point did not terminate")


def _paint(info, state):
    out = info["grid"].copy()
    out[(out == info["floor"]) & state["swept"]] = info["brush"]
    for i, gate in enumerate(info["gates"]):
        for p in gate:
            if i in state["opened"]:
                out[p] = info["brush"] if state["swept"][p] else info["floor"]
            elif any(state["counts"][q] for q in gate):
                out[p] = SINGLE
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _paint(info, _run(info))


def _contains(room, p):
    r0, r1, c0, c1 = room
    return r0 <= p[0] < r1 and c0 <= p[1] < c1


def _partition(rng, count):
    """A random guillotine partition; cuts need not form a regular lattice."""
    h, w = map(int, rng.integers(25, 31, size=2))
    if rng.random() < .2:
        w = h
    margins = tuple(map(int, rng.integers(1, 3, size=4)))
    rooms = [(margins[0], h - margins[1], margins[2], w - margins[3])]
    for _ in range(count - 1):
        options = []
        for i, (r0, r1, c0, c1) in enumerate(rooms):
            for axis, lo, hi in ((0, r0, r1), (1, c0, c1)):
                for cut in range(lo + 8, hi - 8):
                    options.append((i, axis, cut))
        _require(bool(options), "partition does not fit")
        i, axis, cut = options[int(rng.integers(len(options)))]
        r0, r1, c0, c1 = rooms.pop(i)
        rooms.extend([(r0, cut, c0, c1), (cut + 1, r1, c0, c1)] if axis == 0 else
                     [(r0, r1, c0, cut), (r0, r1, cut + 1, c1)])
    base = np.zeros((h, w), bool)
    for r0, r1, c0, c1 in rooms:
        base[r0:r1, c0:c1] = True
        # Independent wall islands and peninsulas produce nonrectangular sweeps.
        for _ in range(int(rng.integers(0, 4))):
            a, b = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            r = int(rng.integers(r0, r1 - a + 1))
            c = int(rng.integers(c0, c1 - b + 1))
            base[r:r + a, c:c + b] = False
    edges = {}
    for i, a in enumerate(rooms):
        for j in range(i + 1, len(rooms)):
            b = rooms[j]
            if a[1] + 1 == b[0] or b[1] + 1 == a[0]:
                row = a[1] if a[1] + 1 == b[0] else b[1]
                lo, hi = max(a[2], b[2]), min(a[3], b[3])
                if hi - lo >= 3:
                    edges[i, j] = 0, row, lo, hi
            elif a[3] + 1 == b[2] or b[3] + 1 == a[2]:
                col = a[3] if a[3] + 1 == b[2] else b[3]
                lo, hi = max(a[0], b[0]), min(a[1], b[1])
                if hi - lo >= 3:
                    edges[i, j] = 1, col, lo, hi
    return base, rooms, edges


def _paths(edges, length):
    graph = {}
    for u, v in edges:
        graph.setdefault(u, set()).add(v)
        graph.setdefault(v, set()).add(u)
    result = []
    def visit(path):
        if len(path) == length + 1:
            result.append(path)
        else:
            for v in sorted(graph[path[-1]] - set(path)):
                visit(path + [v])
    for u in sorted(graph):
        visit([u])
    return result


def _gate_options(edge, width):
    axis, fixed, lo, hi = edge
    return [tuple((fixed, p) if axis == 0 else (p, fixed) for p in range(left, left + width))
            for left in range(lo, hi - width + 1)]


def _separate(arrow, occupied):
    return not any((r + dr, c + dc) in occupied for r, c in arrow[2]
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _shot_options(endpoint, room, base, occupied, brush_cells, swept=None, active=None):
    result = []
    for direction in DIRECTIONS:
        tip = tuple(endpoint[a] - DISTANCE * direction[a] for a in (0, 1))
        arrow = _arrow(tip, direction)
        if not all(_contains(room, p) and base[p] and p not in brush_cells for p in arrow[2]):
            continue
        if not _separate(arrow, occupied):
            continue
        if active is not None and bool(swept[tip]) != active:
            continue
        result.append(arrow)
    return result


def _choose_shots(rng, gate, u, v, style, rooms, base, occupied, brush_cells, swept):
    choices = []
    for endpoint in gate:
        first = _shot_options(endpoint, rooms[u], base, occupied, brush_cells, swept, True)
        if style == "bootstrap":
            second = _shot_options(endpoint, rooms[v], base, occupied, brush_cells, swept, False)
            choices.extend((a, b) for a in first for b in second if _separate(b, set(a[2])))
        elif style == "near":
            for other in gate:
                if other == endpoint:
                    continue
                second = _shot_options(other, rooms[u], base, occupied, brush_cells, swept, True)
                choices.extend((a, b) for a in first for b in second if _separate(b, set(a[2])))
        else:
            choices.extend((a, b) for a, b in combinations(first, 2) if _separate(b, set(a[2])))
            # A third coincident shooter is a compatible structural extension.
            if rng.random() < .25:
                choices.extend(group for group in combinations(first, 3)
                               if all(_separate(b, set(a[2])) for a, b in combinations(group, 2)))
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _make_pair(rng, colors, k, kind, stages):
    wall, floor, brush, arrow_color, gate_color = colors
    for attempt in range(160):
        try:
            count = int(rng.integers(max(3, stages + 1), 6))
            base, rooms, edges = _partition(rng, count)
            candidates = _paths(edges, 3 if kind == "near" else stages)
            _require(bool(candidates), "no suitable dependency path")
            path = candidates[int(rng.integers(len(candidates)))]
            if kind == "near":
                # Path a-s-b-t: open s-b, distinct singles s-a, and an unbootstrapped b-t lock.
                a, source, b, t = path
                plans = [(source, b, "resonate", False),
                         (source, a, "near", False), (b, t, "bootstrap", False)]
            else:
                source = path[0]
                plans = [(u, v, "resonate", kind == "narrow" and i == stages - 1)
                         for i, (u, v) in enumerate(zip(path, path[1:]))]
            # A connected, one-cell-neck alcove is an explicit footprint witness.
            r0, r1, c0, c1 = rooms[source]
            a = max(2, k)
            base[r0:r0 + a, c0:c0 + a] = True
            base[r0:r0 + a + 1, c0 + a] = False
            base[r0 + a, c0:c0 + a + 1] = False
            neck = r0 + a, c0 + a // 2
            base[neck] = True
            protected_floor = {(r, c) for r in range(r0, r0 + a)
                               for c in range(c0, c0 + a)} | {neck}
            protected_walls = ({(r0 - 1, c) for c in range(c0, c0 + a)} |
                               {(r, c0 - 1) for r in range(r0, r0 + a)})
            valid = np.lib.stride_tricks.sliding_window_view(base, (k, k)).all(axis=(-1, -2))
            starts = [tuple(map(int, p)) for p in np.argwhere(valid)
                      if _contains(rooms[source], tuple(p)) and
                      (p[0] >= r0 + a + 1 or p[1] >= c0 + a + 1)]
            _require(bool(starts), "no brush start")
            start = starts[int(rng.integers(len(starts)))]
            _require(_sweep(base, start, 1)[0][neck], "alcove neck is disconnected from the source")
            brush_cells = {(r, c) for r in range(start[0], start[0] + k)
                           for c in range(start[1], start[1] + k)}
            gates, arrows, occupied, planned_open = [], [], set(), []
            used_edges = set()
            for u, v, style, narrow in plans:
                key = tuple(sorted((u, v)))
                edge = edges[key]
                width = k - 1 if narrow else int(rng.integers(max(2, k), min(6, k + 2) + 1))
                # Thin shutters still have at least two cells; k=2 uses a static throat instead.
                _require(width >= 2, "narrow aperture needs k >= 3")
                allowed = base.copy()
                for gate in planned_open:
                    for p in gate:
                        allowed[p] = True
                swept, _ = _sweep(allowed, start, k)
                options = [g for g in _gate_options(edge, width) if not set(g) & protected_walls]
                chosen = None
                for index in rng.permutation(len(options)):
                    gate = options[int(index)]
                    shots = _choose_shots(rng, gate, u, v, style, rooms, base,
                                          occupied, brush_cells, swept)
                    if shots is None:
                        continue
                    trial = allowed.copy()
                    for p in gate:
                        trial[p] = True
                    after, _ = _sweep(trial, start, k)
                    gained = sum(after[p] for p in np.ndindex(base.shape) if _contains(rooms[v], p))
                    if (not narrow and style == "resonate" and gained < k * k) or (narrow and gained):
                        continue
                    chosen = gate, shots
                    break
                _require(chosen is not None, "signals or aperture do not fit")
                gate, shots = chosen
                gates.append(gate)
                arrows.extend(shots)
                occupied.update(p for arrow in shots for p in arrow[2])
                used_edges.add(key)
                if style == "resonate":
                    planned_open.append(gate)
            # Other adjacency edges carry visible but initially uncommanded shutters.
            for key, edge in edges.items():
                if key not in used_edges:
                    width = int(rng.integers(2, min(6, edge[3] - edge[2]) + 1))
                    options = [g for g in _gate_options(edge, width) if not set(g) & protected_walls]
                    _require(bool(options), "uncommanded gate would bypass the source alcove neck")
                    gates.append(options[int(rng.integers(len(options)))])
            gate_cells = {p for gate in gates for p in gate}
            # Misses are real signals, useful evidence that only shutter endpoints act.
            for _ in range(int(rng.integers(0, 4))):
                for _ in range(50):
                    tip = int(rng.integers(base.shape[0])), int(rng.integers(base.shape[1]))
                    direction = DIRECTIONS[int(rng.integers(8))]
                    arrow = _arrow(tip, direction)
                    endpoint = _endpoint(arrow)
                    if not all(0 <= r < base.shape[0] and 0 <= c < base.shape[1] and
                               base[r, c] and (r, c) not in brush_cells for r, c in arrow[2]):
                        continue
                    if not _separate(arrow, occupied) or endpoint in gate_cells:
                        continue
                    if not (0 <= endpoint[0] < base.shape[0] and 0 <= endpoint[1] < base.shape[1]):
                        continue
                    arrows.append(arrow)
                    occupied.update(arrow[2])
                    break
            inp = np.full(base.shape, wall, np.int8)
            inp[base] = floor
            inp[start[0]:start[0] + k, start[1]:start[1] + k] = brush
            for gate in gates:
                for p in gate:
                    inp[p] = gate_color
            for arrow in arrows:
                for p in arrow[2]:
                    inp[p] = arrow_color
            info = _decode(inp)
            state = _run(info)
            # A beam deliberately crosses a wall before its gate endpoint.
            # This teaches projection rather than first-obstacle ray tracing.
            if not _features(info, state)["through_wall"]:
                crossing_candidates = []
                for i in state["active"]:
                    arrow = info["arrows"][i]
                    if _endpoint(arrow) not in gate_cells:
                        continue
                    for step in range(1, DISTANCE):
                        p = tuple(arrow[0][a] + step * arrow[1][a] for a in (0, 1))
                        if inp[p] == floor and p not in protected_floor:
                            crossing_candidates.append(p)
                _require(bool(crossing_candidates), "no endpoint-only witness fits")
                inp[crossing_candidates[int(rng.integers(len(crossing_candidates)))]] = wall
                info = _decode(inp)
                state = _run(info)
            _require(_features(info, state)["through_wall"], "wall crossing shooter is not active")
            expected_waves = 1 if kind == "near" else stages
            _require(len(state["opening_waves"]) == expected_waves, "wrong activation depth")
            _require(np.any(state["swept"] & ~state["initial"] & (inp == floor)),
                     "opening must expose new floor")
            _require(not np.array_equal(_paint(info, state), _paint(info, _run(info, point=True))),
                     "missing footprint contrast")
            if kind == "near":
                _require(not np.array_equal(_paint(info, state), _paint(info, _run(info, all_active=True))),
                         "bootstrap shortcut is not distinguished")
                _require(not np.array_equal(_paint(info, state), _paint(info, _run(info, object_count=True))),
                         "object-wide count shortcut is not distinguished")
            out = _paint(info, state)
            turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
            def transform(grid):
                a = np.rot90(grid, turns)
                return np.array(np.fliplr(a) if reflect else a, dtype=np.int8, copy=True)
            return {"input": transform(inp), "output": transform(out)}
        except AssertionError:
            continue
    raise RuntimeError(f"bounded construction exhausted for {kind}, k={k}, stages={stages}")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(f"{name} is outside {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.choice([c for c in range(10) if c != SINGLE], 5, replace=False)))
    variable_terrain = bool(rng.integers(2))
    train_k = int(rng.integers(2, 4))
    near_k = int(rng.integers(2, 4))
    train_specs = [(train_k, "cascade", 2), (near_k, "near", 1)]
    for _ in range(num_train - 2):
        k = int(rng.integers(2, 4))
        kind = ("cascade", "near", "narrow" if k >= 3 else "cascade")[int(rng.integers(3))]
        stages = 1 if kind == "near" else (2 if kind == "narrow" else int(rng.integers(1, 3)))
        train_specs.append((k, kind, stages))
    train_specs = [train_specs[int(i)] for i in rng.permutation(num_train)]
    scale_shift = bool(rng.integers(2))
    test_specs = []
    for _ in range(num_test):
        k = 4 if scale_shift else int(rng.choice((train_k, near_k)))
        kind = "narrow" if k >= 3 and rng.random() < .4 else "cascade"
        test_specs.append((k, kind, 3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for k, kind, stages in specs:
            colors = palette.copy()
            if variable_terrain:
                remaining = [c for c in range(10) if c not in palette[2:] and c != SINGLE]
                colors[:2] = list(map(int, rng.choice(remaining, 2, replace=False)))
            for _ in range(8):
                pair = _make_pair(rng, colors, k, kind, stages)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not produce distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _features(info, state):
    raw = Counter(_endpoint(a) for a in info["arrows"])
    near = any(i not in state["opened"] and sum(state["counts"][p] for p in gate) >= 2
               and all(state["counts"][p] < 2 for p in gate)
               for i, gate in enumerate(info["gates"]))
    bootstrap = any(i not in state["opened"] and any(raw[p] >= 2 for p in gate)
                    for i, gate in enumerate(info["gates"]))
    through_wall = any(
        info["grid"][arrow[0][0] + n * arrow[1][0], arrow[0][1] + n * arrow[1][1]] == info["wall"]
        for i, arrow in enumerate(info["arrows"]) if i in state["active"]
        and any(_endpoint(arrow) in gate for gate in info["gates"])
        for n in range(1, DISTANCE))
    return dict(shape=info["grid"].shape, k=info["k"], arrows=len(info["arrows"]),
                gates=len(info["gates"]), directions=tuple(sorted({a[1] for a in info["arrows"]})),
                widths=tuple(sorted(len(gate) for gate in info["gates"])),
                waves=len(state["opening_waves"]), opened=len(state["opened"]),
                active=len(state["active"]), swept=int(state["swept"].sum()), near=near,
                bootstrap=bootstrap, through_wall=bool(through_wall),
                narrow_open=any(i in state["opened"] and len(gate) < info["k"]
                                for i, gate in enumerate(info["gates"])))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    witnesses_a = witnesses_b = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape changed")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(inp)
            state = _run(info)
            expected = _paint(info, state)
            _require(np.array_equal(out, expected), "incorrect sweep or shutter state")
            _require(not np.array_equal(inp, out), "identity example")
            _require(np.array_equal(out[inp == info["wall"]], inp[inp == info["wall"]]), "walls changed")
            _require(np.array_equal(out[inp == info["arrow_color"]], inp[inp == info["arrow_color"]]),
                     "arrowheads changed")
            _require(all(state["swept"][info["arrows"][i][0]] for i in state["active"]),
                     "an unreachable arrow fired")
            _require(all(any(state["counts"][p] >= 2 for p in info["gates"][i])
                         for i in state["opened"]), "nonresonant shutter opened")
            features = _features(info, state)
            (training if split == "train" else testing).append(features)
            if split == "train":
                witnesses_a |= not np.array_equal(out, _paint(info, _run(info, point=True)))
                witnesses_b |= not np.array_equal(out, _paint(info, _run(info, no_shots=True)))
    _require(witnesses_a and witnesses_b, "training must distinguish both parent ablations")
    _require(any(f["near"] for f in training), "missing separate-single-hit demonstration")
    _require(any(f["bootstrap"] for f in training), "missing unreachable-shooter demonstration")
    _require(any(f["waves"] == 2 for f in training), "missing two-wave demonstration")
    _require(any(f["through_wall"] for f in training), "missing endpoint-only projection demonstration")
    _require(max(f["waves"] for f in training) == 2 and all(f["waves"] == 3 for f in testing),
             "missing taught depth extrapolation")


def _regressions():
    """Small authored fixtures with explicit expected regions, not sampled paths."""
    for size, style in ((3, "resonate"), (4, "resonate"), (3, "near"), (3, "bootstrap")):
        inp = np.zeros((19, 23), np.int8)
        inp[1:9, 1:22] = inp[10:18, 1:22] = 1
        inp[9, 10:13] = 4
        inp[1:1 + size, 18:18 + size] = 2
        shots = ([_arrow((4, 5), (1, 1)), _arrow((4, 15), (1, -1))] if style == "resonate"
                 else [_arrow((4, 10), (1, 0)), _arrow((4, 6), (1, 1))] if style == "near"
                 else [_arrow((4, 5), (1, 1)), _arrow((14, 10), (-1, 0))])
        for arrow in shots:
            for p in arrow[2]:
                inp[p] = 3
        expected = inp.copy()
        expected[1:9, 1:22][inp[1:9, 1:22] == 1] = 2
        if style == "resonate":
            expected[9, 10:13] = 2 if size == 3 else 1
            if size == 3:
                expected[10:18, 1:22] = 2
        else:
            expected[9, 10:13] = SINGLE
        _require(np.array_equal(solve(inp), expected), "explicit size/resonance/causality fixture failed")
    # Same coordinate rule continues through a second newly exposed pair.
    inp = np.zeros((27, 23), np.int8)
    for lo, hi in ((1, 9), (10, 18), (19, 26)):
        inp[lo:hi, 1:22] = 1
    inp[9, 10:13] = inp[18, 10:13] = 4
    inp[1:4, 1:4] = 2
    for tip in ((4, 5), (4, 15), (13, 5), (13, 15)):
        for p in _arrow(tip, (1, 1) if tip[1] == 5 else (1, -1))[2]:
            inp[p] = 3
    # A central wall pixel lies on the first diagonal shot, with space around it.
    inp[5, 6] = 0
    expected = inp.copy()
    expected[(inp == 1) | (inp == 4)] = 2
    _require(np.array_equal(solve(inp), expected), "explicit two-wave/through-wall fixture failed")
    state = _run(_decode(inp))
    _require(len(state["opening_waves"]) == 2, "cascade was not causally staged")


def self_test():
    """Bounded structural diversity, ablation, symmetry, and mutation checks."""
    _regressions()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, structures, fingerprints, palettes = [], set(), set(), set()
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        fingerprint = tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s]
                            for k in ("input", "output"))
        fingerprints.add(fingerprint)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _decode(inp)
                state = _run(info)
                records.append(dict(_features(info, state), split=split))
                roles = (info["wall"], info["floor"], info["brush"], info["arrow_color"], info["gate_color"])
                palettes.add(roles)
                norm = np.zeros(inp.shape, np.int8)
                for code, color in enumerate(roles):
                    norm[inp == color] = code
                structures.add(_key(norm))
                before = inp.copy()
                _require(np.array_equal(solve(inp), pair["output"]), "solver disagreement")
                _require(np.array_equal(inp, before), "solve mutated its input")
                rotated = np.rot90(inp).copy()
                _require(np.array_equal(solve(rotated), np.rot90(pair["output"])), "rotation changed the rule")
                _require(np.array_equal(solve(np.fliplr(inp).copy()), np.fliplr(pair["output"])),
                         "reflection changed the rule")
        if seed < 3:
            other = generate_puzzle(seed)
            _require(all(np.array_equal(a[k], b[k]) for s in puzzle
                         for a, b in zip(puzzle[s], other[s]) for k in a), "seed is nondeterministic")
    _require(len(fingerprints) == 42, "whole-puzzle repetition")
    _require(len(structures) >= 150, "insufficient structural variation")
    _require(len({f["shape"] for f in records}) >= 20, "insufficient canvas variation")
    _require(len(palettes) >= 35, "insufficient palette variation")
    _require({f["k"] for f in records} == {2, 3, 4}, "brush scale coverage")
    _require({d for f in records for d in f["directions"]} == set(DIRECTIONS), "direction coverage")
    _require(len({f["arrows"] for f in records}) >= 7, "arrow count variation")
    _require(len({f["widths"] for f in records}) >= 15, "aperture width variation")
    _require(any(f["narrow_open"] for f in records), "opening and traversability were conflated")
    _require(any(f["through_wall"] for f in records if f["split"] == "train"), "endpoint-only beam evidence")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(31415, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    bad = {s: [{k: a.copy() for k, a in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"][0, 0] = SINGLE
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a changed wall")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, legal counts, invariants, symmetry, ablations")
