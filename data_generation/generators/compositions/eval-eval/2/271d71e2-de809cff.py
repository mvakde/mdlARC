"""Repair-powered rover: input-derived geometry, costs, and accessibility.

Only standard-library modules and NumPy are used. There is no file I/O, parent
import, cached puzzle, or training-dependent convention in the solver.
"""
from itertools import permutations

import numpy as np


_DIRECTIONS = ("U", "D", "L", "R")
_DELTAS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "Invalid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "Noninteger grid")
    _need(np.all((0 <= g) & (g <= 9)), "Invalid color")
    return g.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y, x in np.argwhere(mask):
        p = (int(y), int(x))
        if seen[p]:
            continue
        stack = [p]
        seen[p] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in _DELTAS.values():
                q = (y + dy, x + dx)
                if (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                        and mask[q] and not seen[q]):
                    seen[q] = True
                    stack.append(q)
        result.append(cells)
    return result


def _bounds(cells):
    ys, xs = zip(*cells)
    return min(ys), min(xs), max(ys) + 1, max(xs) + 1


def _counter_coordinates(bounds, direction):
    t, l, b, r = bounds
    if direction == "U":
        return [(y, x) for y in range(t + 1, b - 1) for x in range(l + 1, r - 1)]
    if direction == "D":
        return [(y, x) for y in range(b - 2, t, -1) for x in range(r - 2, l, -1)]
    if direction == "L":
        return [(y, x) for x in range(l + 1, r - 1) for y in range(b - 2, t, -1)]
    return [(y, x) for x in range(r - 2, l, -1) for y in range(t + 1, b - 1)]


def _bar(bounds, direction, distance, shape):
    t, l, b, r = bounds
    if direction == "U":
        points = [(t - 1 - distance, x) for x in range(l, r)]
    elif direction == "D":
        points = [(b + distance, x) for x in range(l, r)]
    elif direction == "L":
        points = [(y, l - 1 - distance) for y in range(t, b)]
    else:
        points = [(y, r + distance) for y in range(t, b)]
    if any(not (0 <= y < shape[0] and 0 <= x < shape[1]) for y, x in points):
        return None
    return points


def _exact_bar(g, points, color):
    if points is None or any(g[p] != color for p in points):
        return False
    first, last = points[0], points[-1]
    dy, dx = (0, 1) if first[0] == last[0] else (1, 0)
    for y, x in ((first[0] - dy, first[1] - dx), (last[0] + dy, last[1] + dx)):
        if 0 <= y < g.shape[0] and 0 <= x < g.shape[1] and g[y, x] == color:
            return False
    return True


def _read_rover(g):
    explanations = []
    colors = set(map(int, np.unique(g)))
    _need(0 in colors and len(colors) == 4, "Expected background and three visible roles")
    for frame in sorted(colors - {0}):
        for cells in _components(g == frame):
            bounds = _bounds(cells)
            t, l, b, r = bounds
            if min(b - t, r - l) < 3 or len(cells) != 2 * (b - t + r - l) - 4:
                continue
            if any(y not in (t, b - 1) and x not in (l, r - 1) for y, x in cells):
                continue
            fills = colors - {0, frame}
            if not set(map(int, np.unique(g[t + 1:b - 1, l + 1:r - 1]))) <= fills:
                continue
            for direction in _DIRECTIONS:
                attached = _bar(bounds, direction, 0, g.shape)
                if attached is None:
                    continue
                spent = int(g[attached[0]])
                if spent not in fills or not _exact_bar(g, attached, spent):
                    continue
                target = None
                for distance in range(1, 31):
                    segment = _bar(bounds, direction, distance, g.shape)
                    if segment is None:
                        break
                    if _exact_bar(g, segment, spent):
                        target = segment
                        break
                    if any(g[p] != 0 for p in segment):
                        break
                # One empty transverse strip separates the two marker bars.
                if target is None or distance < 2:
                    continue
                remaining = next(iter(fills - {spent}))
                coords = _counter_coordinates(bounds, direction)
                values = [int(g[p]) for p in coords]
                advanced = values.count(spent)
                if values != [spent] * advanced + [remaining] * (len(values) - advanced):
                    continue
                footprint = _bounds(list(cells) + target)
                mask = np.zeros(g.shape, dtype=bool)
                y0, x0, y1, x1 = footprint
                mask[y0:y1, x0:x1] = True
                explanations.append(dict(bounds=bounds, frame=frame, spent=spent,
                                         remaining=remaining, direction=direction,
                                         distance=distance, advanced=advanced,
                                         fuel=len(coords) - advanced, coords=coords,
                                         attached=attached, target=target, road=mask))
    _need(len(explanations) == 1, "Rover lacks a unique geometric explanation")
    return explanations[0]


def _recover_scene(g, rover):
    observed = g.copy()
    observed[rover["road"]] = 0
    boxes = []
    for color in (rover["spent"], rover["remaining"]):
        for cells in _components(observed == color):
            t, l, b, r = _bounds(cells)
            if len(cells) < 12:
                continue  # Isolated ordinary wrong-fill pixels or speckles.
            _need(min(b - t, r - l) >= 5, "Terrain component is too thin")
            _need(not np.any(rover["road"][t:b, l:r]), "A sheet crosses the road")
            boxes.append((t, l, b, r, color))
    _need(2 <= len(boxes) <= 4, "Expected two to four recoverable sheets")
    _need({b[4] for b in boxes} == {rover["spent"], rover["remaining"]}, "Missing fill role")
    scenes = {}
    for order in permutations(boxes):
        clean = np.zeros(g.shape, dtype=np.int8)
        for t, l, b, r, color in order:
            clean[t:b, l:r] = color
        key = clean.tobytes()
        if key not in scenes:
            scenes[key] = (int(np.count_nonzero(clean != observed)), clean)
    ranked = sorted(scenes.values(), key=lambda item: item[0])
    loss, clean = ranked[0]
    _need(len(ranked) == 1 or ranked[1][0] > loss + 1, "Ambiguous visible covering order")
    for t, l, b, r, color in boxes:
        visible = clean[t:b, l:r] == color
        correct = observed[t:b, l:r] == color
        _need(np.count_nonzero(visible) >= 12, "Insufficient visible sheet")
        _need(np.count_nonzero(visible & correct) >= .70 * np.count_nonzero(visible),
              "Insufficient intact evidence")
    _need(loss <= max(10, .18 * g.size), "Excessive corruption")
    return clean, tuple(sorted(boxes))


def _ring(shape, point):
    y, x = point
    return {(yy, xx) for yy in range(max(0, y - 1), min(shape[0], y + 2))
            for xx in range(max(0, x - 1), min(shape[1], x + 2)) if (yy, xx) != point}


def _ring_demands(clean, centers, rover):
    center_set = set(centers)
    demands = {}
    for p in centers:
        color = rover["remaining"] if clean[p] == rover["spent"] else rover["spent"]
        for q in _ring(clean.shape, p) - center_set:
            _need(q not in demands or demands[q] == color, "Conflicting repair rings")
            demands[q] = color
    return demands


def _analyze_input(value):
    g = _grid(value)
    rover = _read_rover(g)
    clean, boxes = _recover_scene(g, rover)
    centers = tuple(tuple(map(int, p)) for p in np.argwhere((clean != 0) & (g == 0)))
    _need(bool(centers), "There must be at least one erasure")
    _need(all(clean[p] == clean[q] or max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 2
              for i, p in enumerate(centers) for q in centers[:i]),
          "Opposite-fill erasures must have disjoint neighborhoods")
    _ring_demands(clean, centers, rover)  # Check compatibility before accessibility.
    rings = {p: _ring(g.shape, p) for p in centers}
    contact = []
    free = []
    for k in range(rover["distance"] + 1):
        bar = set(_bar(rover["bounds"], rover["direction"], k, g.shape))
        touched = {p for p in centers if rings[p] & bar}
        contact.append(touched)
        free.append(any(clean[p] == rover["spent"] for p in touched))
    return dict(input=g, rover=rover, clean=clean, boxes=boxes,
                centers=centers, contact=contact, free=free)


def _walk(a, free=None, fuel=None):
    r = a["rover"]
    free = a["free"] if free is None else free
    fuel = r["fuel"] if fuel is None else fuel
    paid = movement = 0
    visited = set(a["contact"][0])
    for k in range(1, r["distance"] + 1):
        cost = 0 if free[k] else 1
        if paid + cost > fuel:
            break
        paid += cost
        movement = k
        visited.update(a["contact"][k])
    return movement, paid, visited


def _render(a, movement, paid, visited):
    r = a["rover"]
    out = a["clean"].copy()
    for p, color in _ring_demands(a["clean"], visited, r).items():
        out[p] = color
    for p in visited:
        out[p] = r["frame"]
    for p in r["target"]:
        out[p] = r["spent"]
    dy, dx = _DELTAS[r["direction"]]
    t, l, b, rr = r["bounds"]
    bounds = (t + dy * movement, l + dx * movement,
              b + dy * movement, rr + dx * movement)
    t, l, b, rr = bounds
    out[t:b, l:rr] = r["frame"]
    for i, p in enumerate(_counter_coordinates(bounds, r["direction"])):
        out[p] = r["spent"] if i < r["advanced"] + paid else r["remaining"]
    for p in _bar(bounds, r["direction"], 0, out.shape):
        out[p] = r["spent"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive roles, terrain, step prices, and reachable repairs from this input."""
    a = _analyze_input(input_grid)
    return _render(a, *_walk(a))


def _signature(g):
    a = np.asarray(g)
    return a.shape, tuple(map(int, a.flat))


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sample_pair(rng, palette, lesson, turns):
    spent, remaining, frame = palette
    distance = int(rng.integers(14, 21)) if lesson == "test" else int(rng.integers(12, 19))
    start = int(rng.integers(3, 6))
    cluster_length = int(rng.integers(2, 4))
    groups = [list(range(start, start + cluster_length))]
    if lesson == "test":
        second = groups[0][-1] + int(rng.integers(4, 6))
        if second + 1 > distance - 3:
            distance = min(20, second + 4)
        groups.append([second, second + 1])
        if distance >= groups[-1][-1] + 8 and rng.random() < .40:
            third = groups[-1][-1] + 4
            groups.append([third, third + 1])
    charging = [k for group in groups for k in group]
    noncharging_k = min(distance - 3, groups[0][-1] + 3)
    # Counter and canvas capacities are chosen after the desired fuel amount.
    free_steps = {k for center in charging for k in (center - 1, center, center + 1)}
    total_cost = sum(k not in free_steps for k in range(1, distance + 1))
    if lesson == "frontier":
        fuel = start - 2  # Exhaustion immediately before the first charging zone.
    elif lesson == "arrival":
        fuel = total_cost + 1
    elif lesson == "test":
        choice = int(rng.integers(3))
        if choice == 0:
            fuel = start - 2
        elif choice == 1:
            fuel = sum(k not in free_steps for k in range(1, groups[1][0]))
        else:
            fuel = total_cost + int(rng.integers(0, 3))
    else:
        fuel = int(rng.choice([0, start - 2, total_cost, total_cost + 2]))
    cross = int(rng.integers(2, 6)) if lesson in ("frontier", "arrival") else int(rng.integers(1, 6))
    min_depth = max(1, (fuel + 1 + cross - 1) // cross)
    max_depth = min(7, 27 - distance)
    if min_depth > max_depth:
        return None
    depth = int(rng.integers(min_depth, max_depth + 1))
    if lesson == "random" and rng.random() < .18:
        fuel = depth * cross  # Fresh counter.
    goal_margin = int(rng.integers(0, 3))
    bottom_margin = int(rng.integers(0, 3))
    attached_row = goal_margin + distance
    height = attached_row + depth + 3 + bottom_margin
    if height > 30:
        return None
    left_space, right_space = int(rng.integers(7, 12)), int(rng.integers(7, 12))
    outer_width = cross + 2
    width = left_space + outer_width + right_space
    if width > 30:
        return None
    l, r = left_space, left_space + outer_width
    g = np.zeros((height, width), dtype=np.int8)
    road = np.zeros_like(g, dtype=bool)
    road[goal_margin:attached_row + depth + 3, l:r] = True
    charge_side = int(rng.integers(2))
    clipped_job = lesson != "frontier" and rng.random() < .35
    if clipped_job:
        # A noncharging sheet can reach the destination row without merging
        # with its differently colored bar. Its corner erasure clips naturally.
        height -= goal_margin
        attached_row -= goal_margin
        goal_margin = 0
        g = np.zeros((height, width), dtype=np.int8)
        road = np.zeros_like(g, dtype=bool)
        road[:attached_row + depth + 3, l:r] = True
    boxes = []
    for side in (0, 1):
        color = spent if side == charge_side else remaining
        relevant = charging if side == charge_side else [noncharging_k]
        low = int(rng.integers(2, min(relevant) + 1))
        high_low = max(max(relevant) + 1, low + 7)
        if high_low > distance - 2:
            low = 2
            high_low = max(max(relevant) + 1, 9)
        if high_low > distance - 2:
            return None
        high = int(rng.integers(high_low, distance - 1))
        t, b = attached_row - high, attached_row - low + 1
        if clipped_job and side != charge_side:
            t = 0
        space = left_space if side == 0 else right_space
        sheet_width = int(rng.integers(5, space + 1))
        x0, x1 = (l - sheet_width, l) if side == 0 else (r, r + sheet_width)
        boxes.append((t, x0, b, x1, color))
    count = int(rng.integers(2, 5))
    extra_sides = list(map(int, rng.permutation(2)))[:count - 2]
    for side in extra_sides:
        t, x0, b, x1, color = boxes[side]
        # A differently colored front sheet partly covers this base, while its
        # road-facing edge and top/bottom bounding evidence stay exposed.
        if b - t < 7:
            continue
        et = int(rng.integers(t + 1, b - 5))
        eb = int(rng.integers(et + 5, b))
        if side == 0:
            er = l - int(rng.integers(2, 3))
            el = int(rng.integers(0, max(1, er - 4)))
        else:
            el = r + int(rng.integers(2, 3))
            er = int(rng.integers(el + 5, width + 1))
        other = remaining if color == spent else spent
        boxes.append((et, el, eb, er, other))
    clean = np.zeros_like(g)
    for t, x0, b, x1, color in boxes:
        clean[t:b, x0:x1] = color
    _need(not np.any(clean[road]), "Construction put terrain on road")
    lane_col = l - 1 if charge_side == 0 else r
    other_col = r if charge_side == 0 else l - 1
    centers = [(attached_row - k, lane_col) for k in charging]
    centers.append((attached_row - noncharging_k, other_col))
    if clipped_job:
        centers.append((0, other_col))
    if lesson in ("arrival", "test") and noncharging_k + 1 <= distance - 3:
        centers.append((attached_row - noncharging_k - 1, other_col))
    # Deeper erasures are ordinary repair jobs, inaccessible to this scanner.
    offroad = [tuple(map(int, p)) for p in np.argwhere(clean != 0)
               if (p[1] <= l - 3 or p[1] >= r + 2)
               and all(max(abs(int(p[0]) - q[0]), abs(int(p[1]) - q[1])) > 2 for q in centers)]
    rng.shuffle(offroad)
    target_extra = int(rng.integers(1, 4 if lesson != "test" else 7))
    pseudo = {"spent": spent, "remaining": remaining}
    for p in offroad:
        if target_extra == 0:
            break
        if any(clean[p] != clean[q] and max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 2
               for q in centers):
            continue
        centers.append(p)
        target_extra -= 1
    if not any(p[1] <= l - 3 or p[1] >= r + 2 for p in centers):
        return None
    try:
        _ring_demands(clean, centers, pseudo)
    except AssertionError:
        return None
    g = clean.copy()
    for p in centers:
        g[p] = 0
    # Sparse ordinary errors never alter a component's recoverable bounds.
    forbidden = set(centers)
    for p in centers:
        forbidden.update(_ring(g.shape, p))
    wrong_candidates = [(y, x) for y in range(1, height - 1) for x in range(1, width - 1)
                        if clean[y, x] != 0 and (y, x) not in forbidden
                        and np.all(clean[y - 1:y + 2, x - 1:x + 2] == clean[y, x])]
    rng.shuffle(wrong_candidates)
    for p in wrong_candidates[:int(rng.integers(0, 4))]:
        if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in forbidden):
            g[p] = remaining if clean[p] == spent else spent
            forbidden.add(p)
    exterior = [(y, x) for y in range(height) for x in range(width)
                if not road[y, x] and clean[y, x] == 0
                and not np.any(clean[max(0, y - 1):y + 2, max(0, x - 1):x + 2])
                and (x < l - 1 or x > r)]
    rng.shuffle(exterior)
    speckles = []
    for p in exterior:
        if len(speckles) >= 3:
            break
        if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in speckles):
            g[p] = int(rng.choice((spent, remaining)))
            speckles.append(p)
    bounds = (attached_row + 1, l, attached_row + depth + 3, r)
    t, x0, b, x1 = bounds
    g[t:b, x0:x1] = frame
    coords = _counter_coordinates(bounds, "U")
    for i, p in enumerate(coords):
        g[p] = spent if i < len(coords) - fuel else remaining
    for k in (0, distance):
        for p in _bar(bounds, "U", k, g.shape):
            g[p] = spent
    # Reject corruption that destroyed required geometric evidence, rather than
    # asking the solver to remember the intended boxes or hidden clean scene.
    try:
        a = _analyze_input(g)
        if not np.array_equal(a["clean"], clean) or len(a["boxes"]) != len(boxes):
            return None
        out = _render(a, *_walk(a))
    except AssertionError:
        return None
    return {"input": np.rot90(g, turns).copy(), "output": np.rot90(out, turns).copy()}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(np.arange(1, 10), 3, replace=False)))
    vary_palette = bool(rng.integers(2))
    first_turn = int(rng.integers(4))
    second_turn = (first_turn + int(rng.choice((1, 3)))) % 4
    lesson_positions = list(map(int, rng.permutation(num_train)))
    lessons = {lesson_positions[0]: ("frontier", first_turn),
               lesson_positions[1]: ("arrival", second_turn)}
    train_turns = [first_turn, second_turn]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            if split == "train":
                lesson, turns = lessons.get(i, ("random", int(rng.integers(4))))
                train_turns.append(turns)
            else:
                lesson, turns = "test", int(rng.choice(train_turns))
            for _ in range(160):
                palette = (tuple(map(int, rng.choice(np.arange(1, 10), 3, replace=False)))
                           if vary_palette else base_palette)
                pair = _sample_pair(rng, palette, lesson, turns)
                if pair is not None and _signature(pair["input"]) not in seen:
                    seen.add(_signature(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded rover construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _free_runs(free):
    return sum(bool(value) and (i == 0 or not free[i - 1]) for i, value in enumerate(free))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, train, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "Canvas dimensions changed")
            signature = _signature(inp)
            _need(signature not in seen, "Duplicate input")
            seen.add(signature)
            a = _analyze_input(inp)
            movement, paid, visited = _walk(a)
            _need(np.array_equal(out, _render(a, movement, paid, visited)), "Incorrect answer")
            r = a["rover"]
            _need(paid <= r["fuel"] and movement <= r["distance"], "Invalid stopping")
            _need(paid == sum(not a["free"][k] for k in range(1, movement + 1)), "Wrong cost")
            if movement < r["distance"]:
                _need(not a["free"][movement + 1] and paid == r["fuel"], "Premature stop")
            _need(all(out[p] == r["frame"] for p in visited), "A center was overwritten")
            potential_contacts = set().union(*a["contact"])
            _need(bool(set(a["centers"]) - potential_contacts), "Missing off-corridor erasure")
            (train if split == "train" else tests).append((a, movement, paid, visited))
    _need(any(m < a["rover"]["distance"] for a, m, _, _ in train), "No fuel frontier lesson")
    _need(any(m == a["rover"]["distance"] and paid < a["rover"]["fuel"]
              for a, m, paid, _ in train), "No destination-limited lesson")
    _need(len({a["rover"]["direction"] for a, _, _, _ in train}) >= 2, "No rotation lesson")
    _need(any(m != min(a["rover"]["fuel"], a["rover"]["distance"])
              for a, m, _, _ in train), "Ignoring B does not change travel")
    _need(any(_walk(a, free=[bool(c) for c in a["contact"]])[:2] != (m, paid)
              for a, m, paid, _ in train), "Complement color has no demonstrated effect")
    _need(any(any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
                       and a["clean"][p] == a["clean"][q]
                       for p in visited for q in visited if p != q)
              for a, _, _, visited in train), "No merged-ring lesson")
    _need(all(_free_runs(a["free"]) == 1 for a, _, _, _ in train), "Training cost topology changed")
    _need(all(_free_runs(a["free"]) >= 2 for a, _, _, _ in tests), "Missing test reachability shift")


def self_test():
    """Bounded 42-seed checks and specific semantic counterexamples."""
    shapes, geometries, palettes, directions, counts, stops = set(), set(), set(), set(), set(), set()
    puzzles = []
    free_zero, clipped, occluded, wrong_noise, inaccessible = 0, 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_input(pair["input"])
                r = a["rover"]
                movement, paid, visited = _walk(a)
                shapes.add(pair["input"].shape)
                geometries.add((pair["input"].shape, r["bounds"], a["boxes"], r["distance"]))
                palettes.add((r["spent"], r["remaining"], r["frame"]))
                directions.add(r["direction"])
                counts.add(len(a["boxes"]))
                stops.add("arrival" if movement == r["distance"] else "fuel")
                # Reaching a free position after the last paid step must work.
                consumed = 0
                for k in range(1, movement + 1):
                    if a["free"][k] and consumed == r["fuel"]:
                        free_zero += 1
                    consumed += int(not a["free"][k])
                clipped += any(len(_ring(a["clean"].shape, p)) < 8 for p in visited)
                occluded += any(max(t, tt) < min(b, bb) and max(l, ll) < min(rr, rrr)
                                for i, (t, l, b, rr, _) in enumerate(a["boxes"])
                                for tt, ll, bb, rrr, _ in a["boxes"][:i])
                ordinary = (a["input"] != a["clean"]) & ~r["road"] & (a["input"] != 0)
                wrong_noise += bool(np.any(ordinary))
                inaccessible += bool(set(a["centers"]) - visited)
                # Equivariance tests counter ordering and movement in all directions.
                if seed < 3:
                    for turns in range(4):
                        _need(np.array_equal(solve(np.rot90(pair["input"], turns)),
                                             np.rot90(pair["output"], turns)), "Rotation regression")
    _need(len(set(puzzles)) == 42, "Seed diversity collapsed")
    _need(len(shapes) >= 30 and len(geometries) >= 150, "Insufficient geometric variation")
    _need(len(palettes) >= 30 and directions == set(_DIRECTIONS), "Role/orientation variation")
    _need(counts == {2, 3, 4} and stops == {"arrival", "fuel"}, "Structural/outcome variation")
    _need(free_zero >= 42 and occluded > 10 and wrong_noise > 10 and inaccessible >= 210 and clipped > 5,
          "Missing inherited or interaction behavior")
    repeat = generate_puzzle(0)
    _need(puzzles[0] == tuple(_signature(p["input"]) for s in ("train", "test") for p in repeat[s]),
          "Nondeterminism")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + 10 * nt + ne, nt, ne)
            _need(len(p["train"]) == nt and len(p["test"]) == ne, "Explicit counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in repeat[s]] for s in repeat}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator used an incorrect label as evidence")
    # Explicit clipped, compatible and incompatible neighborhood probes.
    clean = np.ones((7, 8), dtype=np.int8)
    clean[:, 4:] = 2
    r = {"spent": 1, "remaining": 2}
    points = [(0, 0), (0, 1)]
    _need(_ring_demands(clean, points, r) == _ring_demands(clean, points[::-1], r),
          "Ring ordering changed the answer")
    _need(len(_ring(clean.shape, (0, 0))) == 3, "Incorrect canvas clipping")
    try:
        _ring_demands(clean, [(3, 3), (3, 4)], r)
    except AssertionError:
        pass
    else:
        raise AssertionError("Conflicting complementary rings accepted")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "geometries": len(geometries),
            "palettes": len(palettes), "sheet_counts": sorted(counts),
            "free_steps_with_empty_battery": free_zero, "overlapping_sheet_pairs": occluded,
            "ordinary_noise_pairs": wrong_noise, "pairs_with_unstamped_holes": inaccessible,
            "generated_clipped_stamp_pairs": clipped}
