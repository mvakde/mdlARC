"""Compass cargo: local D4 docking interleaved with rail-constrained collision.

Only NumPy and the standard library are used. Prediction reconstructs all roles,
waypoints, poses, and scenery from the supplied grid, never from generation state.
"""
from collections import defaultdict
import numpy as np


FLOOR, WALL, RAIL, DOCK = 0, 5, 8, 9
ROLE_COLORS = (1, 2, 3, 4, 6, 7)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, component = [start], {start}
        for r, c in queue:
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    queue.append(q)
        result.append(component)
    return result


def _transform(cells, t):
    """Transform about (0,0), the registration pivot; do not normalize the box."""
    result = {}
    for (r, c), color in cells.items():
        if t >= 4:
            c = -c
        for _ in range(t % 4):
            r, c = c, -r
        result[(r, c)] = color
    return result


def _map_key(cells):
    return tuple(sorted(cells.items()))


def _label_directions(cells, labels):
    core = [p for p, v in cells.items() if v not in labels]
    _require(bool(core), "cargo core missing")
    r0, r1 = min(p[0] for p in core), max(p[0] for p in core)
    c0, c1 = min(p[1] for p in core), max(p[1] for p in core)
    result = {}
    for (r, c), v in cells.items():
        if v not in labels:
            continue
        sides = [d for yes, d in ((r < r0, DIRS[0]), (c > c1, DIRS[1]),
                                  (r > r1, DIRS[2]), (c < c0, DIRS[3])) if yes]
        _require(len(sides) == 1, "label does not project beyond one core side")
        _require(v not in result or result[v] == sides[0], "one identity faces several sides")
        result[v] = sides[0]
    _require(set(result) == set(labels), "cargo label missing")
    a, b = result.values()
    _require(a[0] * b[0] + a[1] * b[1] == 0, "labels must name adjacent sides")
    return result


def _segment(a, p, q):
    _require((p[0] == q[0]) != (p[1] == q[1]), "rail link must be straight")
    dr = int(np.sign(q[0] - p[0]))
    dc = int(np.sign(q[1] - p[1]))
    length = abs(q[0] - p[0]) + abs(q[1] - p[1])
    return [(p[0] + i * dr, p[1] + i * dc) for i in range(length + 1)]


def _parse(inp):
    a = _array(inp)
    colored = set(map(tuple, np.argwhere(~np.isin(a, (FLOOR, WALL, RAIL)))))
    components = _components(colored)
    cargos = [p for p in components if any(a[q] == DOCK for q in p)
              and len({int(a[q]) for q in p}) > 1]
    _require(len(cargos) == 1, "input needs exactly one composite cargo")
    source = cargos[0]
    pivots = [p for p in source if a[p] == DOCK]
    _require(len(pivots) == 1, "cargo needs one registration pivot")
    start = pivots[0]
    cells = {(int(r - start[0]), int(c - start[1])): int(a[r, c]) for r, c in source}
    posts = {p: int(a[p]) for p in colored - source if a[p] != DOCK}
    labels = set(posts.values())
    _require(len(labels) == 2 and labels <= set(ROLE_COLORS), "need two exterior compass identities")
    bodies = set(cells.values()) - labels - {DOCK}
    _require(len(bodies) == 1 and bodies <= set(ROLE_COLORS), "need one separate body role")
    body = next(iter(bodies))
    _label_directions(cells, labels)
    stations = sorted(map(tuple, np.argwhere(a == DOCK)))
    _require(len(stations) >= 2, "rail needs at least two docks")

    # A link may be obscured only by the known source cargo. Consecutive docks
    # in a row or column are eligible only when all intervening rail is visible.
    groups = [defaultdict(list), defaultdict(list)]
    for p in stations:
        groups[0][p[0]].append(p)
        groups[1][p[1]].append(p)
    edges = set()
    for grouped in groups:
        for line in grouped.values():
            line.sort()
            for p, q in zip(line, line[1:]):
                pixels = _segment(a, p, q)
                if any(a[x] == RAIL for x in pixels) and all(
                    a[x] in (RAIL, DOCK) or x in source for x in pixels
                ):
                    edges.add((p, q))
    adjacency = {p: [] for p in stations}
    for p, q in sorted(edges):
        adjacency[p].append(q)
        adjacency[q].append(p)
    _require(len(edges) == len(stations) - 1, "dock graph must be a chain")
    _require(all(1 <= len(v) <= 2 for v in adjacency.values()), "rail has branches or isolated docks")
    _require(len(adjacency[start]) == 1, "cargo must start at an endpoint")
    ordered, previous, p = [], None, start
    while True:
        _require(p not in ordered, "rail cycle")
        ordered.append(p)
        following = [q for q in adjacency[p] if q != previous]
        if not following:
            break
        previous, p = p, following[0]
    _require(len(ordered) == len(stations), "disconnected rail")
    route = [start]
    for p, q in zip(ordered, ordered[1:]):
        route.extend(_segment(a, p, q)[1:])
    track = set(route)
    _require(set(map(tuple, np.argwhere(a == RAIL))) <= track, "unaccounted rail pixels")
    _require(len(track) == len(route), "rail intersects itself")

    compasses = {p: {} for p in stations}
    for (r, c), v in posts.items():
        scores = [(int((r - p[0]) ** 2 + (c - p[1]) ** 2), p) for p in stations]
        scores.sort()
        _require(scores[0][0] != scores[1][0], "compass association tie")
        p = scores[0][1]
        dr, dc = int(r - p[0]), int(c - p[1])
        _require(abs(dr) != abs(dc), "diagonal compass direction is ambiguous")
        direction = (int(np.sign(dr)), 0) if abs(dr) > abs(dc) else (0, int(np.sign(dc)))
        _require(v not in compasses[p] or compasses[p][v] == direction, "contradictory wall marks")
        compasses[p][v] = direction
    poses = {}
    for p in stations:
        compass = compasses[p]
        _require(set(compass) == labels, "each dock needs both compass identities")
        candidates = {}
        for t in range(8):
            candidate = _transform(cells, t)
            if _label_directions(candidate, labels) == compass:
                candidates[_map_key(candidate)] = candidate
        _require(len(candidates) == 1, "dock must determine one distinct cargo pose")
        poses[p] = next(iter(candidates.values()))

    static = a.copy()
    for p in source:
        static[p] = FLOOR
    for p in track:
        static[p] = RAIL
    for p in stations:
        static[p] = DOCK
    return {"input": a, "static": static, "source": source, "cells": cells,
            "start": start, "stations": ordered, "route": route, "poses": poses,
            "labels": labels, "body": body, "compasses": compasses, "posts": posts}


def _fits(static, pivot, cells):
    h, w = static.shape
    return all(0 <= pivot[0] + r < h and 0 <= pivot[1] + c < w
               and static[pivot[0] + r, pivot[1] + c] in (FLOOR, RAIL, DOCK)
               for r, c in cells)


def _paste(static, pivot, cells):
    result = static.copy()
    for (r, c), v in cells.items():
        result[pivot[0] + r, pivot[1] + c] = v
    return result


def _walk(info, mode="normal", collisions=True):
    route, poses, static = info["route"], info["poses"], info["static"]
    fixed = poses[route[0]] if mode == "first" else poses[route[-1]] if mode == "last" else None
    cargo = fixed if fixed is not None else poses[route[0]]
    _require(_fits(static, route[0], cargo), "starting docking pose cannot fit")
    position, reached, failure = route[0], 0, None
    for index, q in enumerate(route[1:], 1):
        next_cargo = fixed if fixed is not None else poses.get(q, cargo)
        if collisions and (not _fits(static, q, cargo) or not _fits(static, q, next_cargo)):
            failure = q
            break
        position, cargo, reached = q, next_cargo, index
    # Collision-free ablation is used only where the endpoint footprint fits.
    _require(_fits(static, position, cargo), "final cargo does not fit")
    output = _paste(static, position, cargo)
    return output, {"destination": position, "cargo": cargo, "travel": reached,
                    "success": failure is None, "failure": failure}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the exact transformation from this input alone."""
    return _walk(_parse(input_grid))[0]


def _path_between(rng, start, end):
    r, c = start
    points = {(r, c)}
    while (r, c) != end:
        if r != end[0] and (c == end[1] or rng.random() < 0.5):
            r += int(np.sign(end[0] - r))
        else:
            c += int(np.sign(end[1] - c))
        points.add((r, c))
    return points


def _make_cargo(rng, radius, palette, large=False):
    body, label1, label2 = palette
    h, w = map(int, rng.integers(2, radius + 1, size=2))
    if large:
        if rng.integers(2):
            h = int(rng.integers(4, radius + 1))
        else:
            w = int(rng.integers(4, radius + 1))
    pivot = (h // 2, w // 2)
    top = sorted(map(int, rng.choice(w, size=int(rng.integers(1, min(2, w) + 1)), replace=False)))
    left = sorted(map(int, rng.choice(h, size=int(rng.integers(1, min(2, h) + 1)), replace=False)))
    endpoints = [(0, c) for c in top] + [(r, 0) for r in left]
    endpoints += [(0, int(rng.integers(w))), (h - 1, int(rng.integers(w))),
                  (int(rng.integers(h)), 0), (int(rng.integers(h)), w - 1)]
    rng.shuffle(endpoints)
    core = {pivot}
    for q in endpoints:
        distances = {p: abs(p[0] - q[0]) + abs(p[1] - q[1]) for p in core}
        anchors = sorted(p for p in core if distances[p] == min(distances.values()))
        core.update(_path_between(rng, anchors[int(rng.integers(len(anchors)))], q))
    target = int(rng.integers(len(core), h * w + 1))
    while len(core) < target:
        frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in DIRS
                           if 0 <= r + dr < h and 0 <= c + dc < w} - core)
        core.add(frontier[int(rng.integers(len(frontier)))])
    cells = {p: body for p in core}
    cells[pivot] = DOCK
    cells.update({(-1, c): label1 for c in top})
    cells.update({(r, -1): label2 for r in left})
    result = {(r - pivot[0], c - pivot[1]): v for (r, c), v in cells.items()}
    _require(max(max(abs(r), abs(c)) for r, c in result) < radius, "cargo exceeds chamber")
    return result


def _lattice_walk(rng, nr, nc, count):
    """Tiny bounded-depth randomized DFS, not a bank of route templates."""
    def search(path, used):
        if len(path) == count:
            return path
        r, c = path[-1]
        options = [(r + dr, c + dc) for dr, dc in DIRS
                   if 0 <= r + dr < nr and 0 <= c + dc < nc and (r + dr, c + dc) not in used]
        rng.shuffle(options)
        for p in options:
            found = search(path + [p], used | {p})
            if found is not None:
                return found
        return None
    starts = [(r, c) for r in range(nr) for c in range(nc)]
    rng.shuffle(starts)
    for p in starts:
        found = search([p], {p})
        if found is not None:
            return found
    raise AssertionError("no lattice walk fits")


def _room_coordinates(rng, n, radius):
    base, minimum_gap = 2 * radius + 3, 2 * radius + 3
    spare = 30 - base - (n - 1) * minimum_gap
    gaps = [minimum_gap] * (n - 1)
    for i in rng.permutation(n - 1):
        extra = int(rng.integers(0, min(spare, 5) + 1))
        gaps[int(i)] += extra
        spare -= extra
    before = int(rng.integers(spare + 1))
    after = int(rng.integers(spare - before + 1))
    coordinates = [radius + 1 + before]
    for gap in gaps:
        coordinates.append(coordinates[-1] + gap)
    return coordinates, coordinates[-1] + radius + 2 + after


def _place_compass(static, station, directions, radius, rng=None):
    for color, (dr, dc) in directions.items():
        signs = [1] if rng is None else ([-1, 1] if rng.random() < .3 else [int(rng.choice((-1, 1)))])
        for sign in signs:
            # Normal displacement dominates; tangent radius lies outside all doors.
            p = (station[0] + dr * (radius + 1) + dc * sign * radius,
                 station[1] + dc * (radius + 1) + dr * sign * radius)
            _require(static[p] == WALL, "wall compass overlaps a passage")
            static[p] = color


def _paint_passage(static, a, b, radius, negative, positive):
    if a[0] == b[0]:
        lo, hi = sorted((a[1], b[1]))
        static[a[0] - negative:a[0] + positive + 1, lo + radius:hi - radius + 1] = FLOOR
    else:
        lo, hi = sorted((a[0], b[0]))
        static[lo + radius:hi - radius + 1, a[1] - negative:a[1] + positive + 1] = FLOOR


def _construct(rng, radius, count, blocked, teaching=None, large=False):
    palette = tuple(map(int, rng.choice(ROLE_COLORS, 3, replace=False)))
    cargo = _make_cargo(rng, radius, palette, large)
    labels = set(palette[1:])
    max_nodes = 3 if radius == 3 else 2
    shapes = [(r, c) for r in range(1, max_nodes + 1) for c in range(1, max_nodes + 1)
              if r * c >= count]
    nr, nc = shapes[int(rng.integers(len(shapes)))]
    walk = _lattice_walk(rng, nr, nc, count)
    rows, h = _room_coordinates(rng, nr, radius)
    cols, w = _room_coordinates(rng, nc, radius)
    stations = [(rows[r], cols[c]) for r, c in walk]
    static = np.full((h, w), WALL, dtype=np.int8)
    for r, c in stations:
        static[r - radius + 1:r + radius, c - radius + 1:c + radius] = FLOOR
    transformations = list(map(int, rng.integers(0, 8, size=count)))
    source_t = int(rng.integers(8))
    source = _transform(cargo, source_t)
    if teaching == "success":
        # Endpoint must require reflection relative to the input, not merely rotation.
        reflected = [_transform(source, t) for t in range(4, 8)]
        wanted = reflected[int(rng.integers(4))]
        transformations[-1] = next(t for t in range(8) if _transform(cargo, t) == wanted)
    lower_block = 3 if teaching == "longer" else 1 if teaching in ("blocked", "larger") else 0
    first_block = int(rng.integers(lower_block, count - 1)) if blocked else None
    if teaching == "blocked":
        transformations[first_block] = int(rng.choice([t for t in range(8) if t != transformations[0]]))
        transformations[-1] = int(rng.choice([t for t in range(8) if t != transformations[first_block]]))
    poses = {p: _transform(cargo, t) for p, t in zip(stations, transformations)}
    for i, (a, b) in enumerate(zip(stations, stations[1:])):
        pose = poses[a]
        axis = 0 if a[0] == b[0] else 1
        required_neg = max(0, -min(p[axis] for p in pose))
        required_pos = max(0, max(p[axis] for p in pose))
        if first_block is None or i < first_block:
            negative = int(rng.integers(required_neg, radius))
            positive = int(rng.integers(required_pos, radius))
        elif i == first_block:
            choices = [j for j, need in enumerate((required_neg, required_pos)) if need > 0]
            side = choices[int(rng.integers(len(choices)))]
            negative, positive = map(int, rng.integers(0, radius, size=2))
            if side == 0:
                negative = int(rng.integers(required_neg))
            else:
                positive = int(rng.integers(required_pos))
        else:
            negative, positive = map(int, rng.integers(0, radius, size=2))
        _paint_passage(static, a, b, radius, negative, positive)
    route = [stations[0]]
    for a, b in zip(stations, stations[1:]):
        route.extend(_segment(static, a, b)[1:])
    for p in route:
        static[p] = RAIL
    for p in stations:
        static[p] = DOCK
        _place_compass(static, p, _label_directions(poses[p], labels), radius, rng)
    inp = _paste(static, stations[0], source)
    # Construction has known geometry, but its answer is certified again after
    # pose/palette/global transformations by the input-only reconstruction.
    out, _ = _walk({"route": route, "poses": poses, "static": static})
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    def transformed(a):
        a = np.rot90(a, turns)
        return np.array(np.fliplr(a) if reflect else a, copy=True)
    pair = {"input": transformed(inp), "output": transformed(out)}
    info = _inspect_pair(pair)
    _require(info["success"] == (not blocked), "requested outcome not achieved")
    if teaching:
        _require(info["travel"] > 0 and info["completed_links"] >= 1, "teaching motion is too short")
    if teaching == "success":
        _require(info["reflection"], "success must demonstrate reflection")
    if teaching == "blocked":
        _require(info["destination"] not in info["stations"], "teaching stop must be between docks")
        _require(info["first_ablation"] and info["last_ablation"], "local compass interaction missing")
        _require(info["first_motion_ablation"], "local compass must change collision position/outcome")
    if teaching == "longer":
        _require(info["completed_links"] >= 3, "test must exercise the longer reachable relay")
    return pair


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " is out of range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    longer_route = bool(rng.integers(2))
    training_specs = [(3, int(rng.integers(3, 5)), False, "success", False),
                      (3, int(rng.integers(3, 5)), True, "blocked", False)]
    training_specs += [(3, int(rng.integers(2, 5)), bool(rng.integers(2)), None, False)
                       for _ in range(num_train - 2)]
    rng.shuffle(training_specs)
    tests = [(3, int(rng.integers(5, 8)), bool(rng.integers(2)), "longer", False)
             if longer_route else (int(rng.integers(4, 6)), int(rng.integers(3, 5)),
                                   bool(rng.integers(2)), "larger", True)
             for _ in range(num_test)]
    puzzle, used = {"train": [], "test": []}, set()
    for split, specs in (("train", training_specs), ("test", tests)):
        for spec in specs:
            for _ in range(64):
                try:
                    pair = _construct(rng, *spec)
                    key = _key(pair["input"])
                    if key not in used:
                        used.add(key)
                        puzzle[split].append(pair)
                        break
                except AssertionError:
                    continue
            else:
                raise AssertionError("bounded construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = _array(pair["input"]), _array(pair["output"])
    _require(inp.shape == out.shape, "shape must be preserved")
    info = _parse(inp)
    expected, motion = _walk(info)
    _require(np.array_equal(out, expected), "output violates rail/compass/footprint rule")
    moving = { (motion["destination"][0] + r, motion["destination"][1] + c)
               for r, c in motion["cargo"] }
    mask = np.ones(inp.shape, dtype=bool)
    for p in moving:
        mask[p] = False
    _require(np.array_equal(out[mask], info["static"][mask]), "scenery changed outside cargo")
    for v in {info["body"]} | info["labels"]:
        _require(int(np.sum(inp == v)) == int(np.sum(out == v)), "cargo color pixels not preserved")
    rotations = {_map_key(_transform(info["cells"], t)) for t in range(4)}
    reflected = _map_key(motion["cargo"]) not in rotations
    first_output, first_motion = _walk(info, mode="first")
    last_output, _ = _walk(info, mode="last")
    endpoint_output, _ = _walk(info, collisions=False)
    route_index = {p: i for i, p in enumerate(info["route"])}
    completed = sum(route_index[p] <= motion["travel"] for p in info["stations"][1:])
    headings = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(info["route"], info["route"][1:])]
    # Pixel measurements: cross-sections at passage midpoints, not sampled widths.
    widths = []
    for a, b in zip(info["stations"], info["stations"][1:]):
        r, c = (a[0] + b[0]) // 2, (a[1] + b[1]) // 2
        dr, dc = (1, 0) if a[0] == b[0] else (0, 1)
        width = 1
        for sign in (-1, 1):
            nr, nc = r + sign * dr, c + sign * dc
            while 0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1] and info["static"][nr, nc] in (0, 8, 9):
                width += 1
                nr, nc = nr + sign * dr, nc + sign * dc
        widths.append(width)
    r0, r1 = min(p[0] for p in info["cells"]), max(p[0] for p in info["cells"])
    c0, c1 = min(p[1] for p in info["cells"]), max(p[1] for p in info["cells"])
    core = [p for p, v in info["cells"].items() if v not in info["labels"]]
    core_shape = (max(p[0] for p in core) - min(p[0] for p in core) + 1,
                  max(p[1] for p in core) - min(p[1] for p in core) + 1)
    return {**info, **motion, "reflection": reflected, "completed_links": completed,
            "first_ablation": not np.array_equal(out, first_output),
            "first_motion_ablation": first_motion["destination"] != motion["destination"]
                                      or first_motion["success"] != motion["success"],
            "last_ablation": not np.array_equal(out, last_output),
            "collision_ablation": not np.array_equal(out, endpoint_output),
            "turns": sum(a != b for a, b in zip(headings, headings[1:])),
            "widths": tuple(widths), "cargo_shape": (r1 - r0 + 1, c1 - c0 + 1),
            "core_shape": core_shape, "core_sparse": len(core) < core_shape[0] * core_shape[1],
            "sparse": len(info["cells"]) < (r1 - r0 + 1) * (c1 - c0 + 1)}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal counts")
    seen, training, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            info = _inspect_pair(pair)
            key = _key(info["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == "train":
                training.append(info)
            else:
                tests.append(info)
    _require(any(i["success"] and i["travel"] > 0 and i["reflection"] for i in training),
             "need a moving reflection-dependent success")
    _require(any(not i["success"] and i["travel"] > 0 and i["completed_links"] >= 1
                 and i["destination"] not in i["stations"] and i["collision_ablation"]
                 and i["first_ablation"] and i["last_ablation"] and i["first_motion_ablation"] for i in training),
             "need a moving blockage exposing intermediate compass and collision")
    maximum_docks = max(len(i["stations"]) for i in training)
    maximum_core = max(max(i["core_shape"]) for i in training)
    _require(all((len(i["stations"]) > maximum_docks and i["completed_links"] >= 3)
                 or (max(i["core_shape"]) > maximum_core and i["completed_links"] >= 1)
                 for i in tests), "test must extend reachable relay length or cargo size")


def _handwritten_relay(wide, equal_throats=False):
    """Three explicit docks; independently specified expected stopping coordinates."""
    static = np.full((15, 27), WALL, dtype=np.int8)
    stations = [(7, 4), (7, 13), (7, 22)]
    for r, c in stations:
        static[r - 2:r + 3, c - 2:c + 3] = FLOOR
    static[6:9 if equal_throats else 10, 7:11] = FLOOR
    static[6:9 if wide else 8, 16:20] = FLOOR
    if not wide:
        static[6, 16:20] = WALL
    static[7, 4:23] = RAIL
    for p in stations:
        static[p] = DOCK
    cargo = {(0, 0): DOCK, (0, 1): 3, (1, 0): 3, (1, 1): 3,
             (2, 0): 3, (2, 1): 3, (-1, 0): 1, (1, -1): 2}
    poses = [_transform(cargo, t) for t in ((1, 0, 2) if equal_throats else (0, 1, 2))]
    for p, pose in zip(stations, poses):
        _place_compass(static, p, _label_directions(pose, {1, 2}), 3)
    inp = _paste(static, stations[0], _transform(cargo, 4))
    destination = (7, 14) if equal_throats else stations[-1] if wide else (7, 15)
    expected = _paste(static, destination, poses[1] if equal_throats or not wide else poses[-1])
    return inp, expected


def self_test():
    for wide in (False, True):
        inp, expected = _handwritten_relay(wide)
        _require(np.array_equal(solve(inp), expected), "handwritten relay regression")
    inp, expected = _handwritten_relay(True, equal_throats=True)
    info = _parse(inp)
    _require(np.array_equal(solve(inp), expected), "equal-width orientation/clearance regression")
    _require(_walk(info, mode="first")[1]["success"], "compass omission must change reachability")
    # A wall in the empty top-right box corner must not block this sparse cargo.
    static = np.full((13, 23), WALL, dtype=np.int8)
    static[5:8, 3:20] = FLOOR
    static[6, 4:19] = RAIL
    stations = [(6, 4), (6, 18)]
    cargo = {(0, 0): DOCK, (0, 1): 3, (1, 0): 3, (1, 1): 3, (-1, 0): 1, (0, -1): 2}
    for p in stations:
        static[p] = DOCK
        _place_compass(static, p, {1: DIRS[0], 2: DIRS[3]}, 3)
    static[5, 19] = WALL
    inp = _paste(static, stations[0], cargo)
    expected = _paste(static, stations[-1], cargo)
    _require(np.array_equal(solve(inp), expected), "empty-corner collision regression")
    rectangle = {(r, c): 3 for r in range(-1, 2) for c in range(-1, 2)}
    _require(not _fits(static, stations[-1], rectangle), "rectangle shortcut fixture ineffective")

    records, fingerprints, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                i = _inspect_pair(pair)
                records.append(i)
                # Normalize semantic roles, including arbitrary label identities.
                normalized = i["input"].copy()
                normal_colors = {0: 0, 5: 1, 8: 2, 9: 3, i["body"]: 4}
                normal_colors.update({v: 5 + j for j, v in enumerate(sorted(i["labels"]))})
                for v, code in normal_colors.items():
                    normalized[i["input"] == v] = code
                structures.add(_key(normalized))
    _require(len(fingerprints) == 42, "puzzles repeat across seeds")
    _require(len(structures) >= 150, "too few role-normalized structures")
    _require(len({i["input"].shape for i in records}) >= 25, "canvas geometry collapsed")
    _require({len(i["stations"]) for i in records} >= {2, 3, 4, 5, 6, 7}, "dock counts collapsed")
    _require(max(i["turns"] for i in records) >= 4, "longer turning rails missing")
    _require(len({i["cargo_shape"] for i in records}) >= 8, "cargo extents collapsed")
    _require(len({len(i["cells"]) for i in records}) >= 10, "cargo cell counts collapsed")
    _require(len({i["widths"] for i in records}) >= 60, "passage profiles collapsed")
    _require(sum(i["core_sparse"] for i in records) >= 25, "body indentations/gaps missing")
    _require({i["body"] for i in records} == set(ROLE_COLORS), "body palettes collapsed")
    _require(set().union(*(i["labels"] for i in records)) == set(ROLE_COLORS), "compass palettes collapsed")
    compass_poses = {tuple(i["compasses"][p][v] for v in sorted(i["labels"]))
                    for i in records for p in i["stations"]}
    _require(len(compass_poses) == 8, "local D4 compass poses collapsed")
    for a, b in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        first, second = generate_puzzle(827, a, b), generate_puzzle(827, a, b)
        _require(all(np.array_equal(x[k], y[k]) for s in first for x, y in zip(first[s], second[s])
                     for k in ("input", "output")), "seed determinism failed")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (None, 1), (2.0, 1), (True, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(0)
    for mutation in ("wall", "teleport", "freeze"):
        bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in original[s]]
               for s in ("train", "test")}
        pair = next(p for p in bad["train"] if not _inspect_pair(p)["success"])
        info = _parse(pair["input"])
        if mutation == "wall":
            p = tuple(np.argwhere(pair["output"] == WALL)[0])
            pair["output"][p] = FLOOR
        else:
            pair["output"] = _walk(info, collisions=False)[0] if mutation == "teleport" else _walk(info, mode="first")[0]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted " + mutation)


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
