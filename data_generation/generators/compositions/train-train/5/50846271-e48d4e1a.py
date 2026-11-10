"""Invisible cross barriers: reconstruct solid geometry before diagonal travel.

Only the standard library and NumPy are used.  Parsing and solving do not read
files, training pairs, generation metadata, or random state.
"""
from __future__ import annotations

from itertools import product
import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC color")
    return grid.astype(np.int8, copy=False)


def _body(row, col, radius):
    return frozenset([(row, col)] +
                     [(row + d, col) for d in range(-radius, radius + 1)] +
                     [(row, col + d) for d in range(-radius, radius + 1)])


def _tips(row, col, radius):
    return frozenset(((row-radius, col), (row+radius, col),
                      (row, col-radius), (row, col+radius)))


def _inside(cells, shape):
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _marker(shape, corner, axis, length):
    h, w = shape
    bottom, right = divmod(corner, 2)
    r, c = (h-1 if bottom else 0), (w-1 if right else 0)
    dr, dc = (-1 if bottom else 1), (-1 if right else 1)
    return frozenset((r + (i*dr if axis == 0 else 0),
                      c + (i*dc if axis == 1 else 0)) for i in range(length))


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, todo = {start}, [start]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(frozenset(component))
    return result


def _parse(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    corners = ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))
    active = [i for i, p in enumerate(corners) if grid[p] != 0]
    _require(len(active) == 1, "one nonzero corner must identify the command")
    corner = active[0]
    vehicle_color = int(grid[corners[corner]])
    boundary = {(r, c) for r, c in np.ndindex(grid.shape)
                if r in (0, h-1) or c in (0, w-1)}
    command = frozenset(p for p in boundary if grid[p] != 0)
    length = len(command)
    _require(1 <= length <= 6 and all(grid[p] == vehicle_color for p in command),
             "malformed command run")
    axes = [axis for axis in (0, 1)
            if length < grid.shape[axis] and
            command == _marker(grid.shape, corner, axis, length)]
    _require(bool(axes), "command must be a contiguous corner run")
    bottom, right = divmod(corner, 2)
    direction = (-1 if bottom else 1, -1 if right else 1)

    palette, amounts = np.unique(grid, return_counts=True)
    counts = dict(zip(map(int, palette), map(int, amounts)))
    others = [c for c in counts if c not in (0, vehicle_color)]
    _require(len(others) >= 2, "need filler and barrier clues")
    filler = max(others, key=lambda c: counts[c])
    _require(all(counts[filler] > n for c, n in counts.items()
                 if c not in (0, filler)), "filler must be the most frequent nonzero color")
    barrier_colors = sorted(set(others) - {filler})
    _require(1 <= len(barrier_colors) <= 4, "need one to four barriers")

    vehicle_mask = grid == vehicle_color
    vehicle_mask = vehicle_mask.copy()
    for p in command:
        vehicle_mask[p] = False
    vehicles = []
    for cells in _components(vehicle_mask):
        rows, cols = zip(*cells)
        row, col = (min(rows)+max(rows))//2, (min(cols)+max(cols))//2
        radius = (max(rows)-min(rows))//2
        _require(radius in (1, 2) and cells == _body(row, col, radius),
                 "vehicle must be a complete finite plus")
        _require(all(0 < r < h-1 and 0 < c < w-1 for r, c in cells),
                 "initial vehicles must not touch the boundary")
        vehicles.append((row, col, radius))
    _require(1 <= len(vehicles) <= 3, "need one to three vehicles")

    barriers = []
    occupied = frozenset()
    for color in barrier_colors:
        clues = frozenset(tuple(map(int, p)) for p in np.argwhere(grid == color))
        rows, cols = sorted({p[0] for p in clues}), sorted({p[1] for p in clues})
        candidates = []
        for row, col, radius in product(rows, cols, (2, 3, 4)):
            if not (radius < row < h-1-radius and radius < col < w-1-radius):
                continue
            if len(_tips(row, col, radius) & clues) < 3:
                continue
            cells = _body(row, col, radius)
            if clues <= cells and all(grid[p] in (color, filler) for p in cells):
                candidates.append((color, row, col, radius, cells))
        _require(len(candidates) == 1, "barrier must have a unique three-tip reconstruction")
        barrier = candidates[0]
        _require(len(barrier[4] - clues) >= 2, "barrier needs a concealed portion")
        _require(occupied.isdisjoint(barrier[4]), "barriers must not overlap")
        occupied |= barrier[4]
        barriers.append(barrier)
    for v in vehicles:
        _require(_body(*v).isdisjoint(occupied), "vehicle starts inside a barrier")
    sweeps = [_sweep(v, length, direction) for v in vehicles]
    for i, sweep in enumerate(sweeps):
        _require(all(sweep.isdisjoint(other) for other in sweeps[:i]),
                 "potential vehicle sweeps must be disjoint")
    _require(any(grid[p] == filler and p not in occupied for p in np.ndindex(grid.shape)),
             "unrelated filler is required")
    return {"grid": grid, "corner": corner, "axes": frozenset(axes),
            "length": length, "direction": direction, "command": command,
            "vehicle_color": vehicle_color, "filler": filler,
            "vehicles": tuple(vehicles), "barriers": tuple(barriers),
            "solid": occupied}


def _at(vehicle, distance, direction):
    row, col, radius = vehicle
    dr, dc = direction
    return _body(row + distance*dr, col + distance*dc, radius)


def _sweep(vehicle, length, direction):
    return frozenset().union(*(_at(vehicle, t, direction) for t in range(length+1)))


def _travel(vehicle, length, direction, solid, shape):
    for step in range(1, length+1):
        body = _at(vehicle, step, direction)
        if not _inside(body, shape):
            return step-1, "boundary"
        if not body.isdisjoint(solid):
            return step-1, "barrier"
    return length, "budget"


def _paint(grid, vehicle_color, command, vehicles, barriers, length, direction):
    solid = frozenset().union(*(b[4] for b in barriers))
    output = grid.copy()
    for p in command:
        output[p] = 0
    for vehicle in vehicles:
        for p in _body(*vehicle):
            output[p] = 0
    for color, _, _, _, cells in barriers:
        for p in cells:
            output[p] = color
    for vehicle in vehicles:
        distance, _ = _travel(vehicle, length, direction, solid, grid.shape)
        for p in _at(vehicle, distance, direction):
            output[p] = vehicle_color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _parse(input_grid)
    return _paint(scene["grid"], scene["vehicle_color"], scene["command"],
                  scene["vehicles"], scene["barriers"], scene["length"],
                  scene["direction"])


def _reflect(grid, corner):
    bottom, right = divmod(corner, 2)
    return grid[::(-1 if bottom else 1), ::(-1 if right else 1)].copy()


def _scene(rng, palette, radii, length, axis, vehicle_count, barrier_count,
           mode="block", hit=2, excluded_shapes=frozenset(), controlled=False):
    """Construct in TL coordinates. Labels use explicit construction geometry.

    A bounded rejection search chooses arbitrary vehicle positions and local
    blocker offsets. There is no fixed-layout fallback or whole-parent call.
    """
    filler, vehicle_color, *wall_pool = palette
    for _attempt in range(280):
        minimum = 17 if vehicle_count == 1 else 22 if vehicle_count == 2 else 26
        if mode == "tunnel":
            minimum = max(minimum, 25)
        shape = tuple(map(int, rng.integers(minimum, 31, size=2)))
        if shape in excluded_shapes:
            continue
        h, w = shape
        vehicles, sweeps = [], []
        for index in range(vehicle_count):
            radius = (1 if index == 0 else 2) if controlled else int(rng.integers(1, 3))
            if mode == "tunnel" and index == 0:
                radius = 1
            placed = False
            for _ in range(120):
                if mode == "boundary" and index == 0:
                    margin = int(rng.integers(1, min(length, 3)))
                    if rng.integers(2):
                        row = h-1-radius-margin
                        col = int(rng.integers(radius+1, w-radius-length-1))
                    else:
                        row = int(rng.integers(radius+1, h-radius-length-1))
                        col = w-1-radius-margin
                else:
                    row = int(rng.integers(radius+1, h-radius-length-1))
                    col = int(rng.integers(radius+1, w-radius-length-1))
                v = (row, col, radius)
                sweep = _sweep(v, length, (1, 1))
                if any(not sweep.isdisjoint(s) for s in sweeps):
                    continue
                body = _body(*v)
                neighbors = {(r+dr, c+dc) for r, c in body
                             for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}
                if any(not neighbors.isdisjoint(_body(*old)) for old in vehicles):
                    continue
                vehicles.append(v)
                sweeps.append(sweep)
                placed = True
                break
            if not placed:
                break
        if len(vehicles) != vehicle_count:
            continue

        wall_colors = [wall_pool[int(i)] for i in rng.permutation(len(wall_pool))]
        barriers, hidden_contacts = [], {}
        all_sources = frozenset().union(*(_body(*v) for v in vehicles))
        all_sweeps = frozenset().union(*sweeps)
        if mode in ("block", "tunnel"):
            row, col, a = vehicles[0]
            options = []
            for b in radii:
                extent = a+b
                offsets = list(product(range(-extent, extent+1), repeat=2))
                for di, dj in offsets:
                    wr, wc = row+hit+di, col+hit+dj
                    if not (b < wr < h-1-b and b < wc < w-1-b):
                        continue
                    cells = _body(wr, wc, b)
                    if not cells.isdisjoint(all_sources):
                        continue
                    if any(not cells.isdisjoint(s) for s in sweeps[1:]):
                        continue
                    distance, reason = _travel(vehicles[0], length, (1, 1), cells, shape)
                    if reason != "barrier" or distance != hit-1:
                        continue
                    contact = cells & _at(vehicles[0], hit, (1, 1))
                    if len(_tips(wr, wc, b) - contact) < 3:
                        continue
                    if (row+hit, col+hit) in cells:
                        continue
                    if mode == "tunnel" and not cells.isdisjoint(_at(vehicles[0], length, (1, 1))):
                        continue
                    options.append((b, wr, wc, cells, contact))
            if not options:
                continue
            b, wr, wc, cells, contact = options[int(rng.integers(len(options)))]
            barriers.append((wall_colors[0], wr, wc, b, cells))
            hidden_contacts[wall_colors[0]] = contact

        # Decoys are genuine hidden barriers outside the potential routes.
        while len(barriers) < barrier_count:
            placed = False
            for trial in range(250):
                if controlled and len(barriers) == 1:
                    previous_radius = barriers[0][3]
                    b = next(r for r in radii if r != previous_radius)
                else:
                    b = int(rng.choice(radii))
                wr = int(rng.integers(b+1, h-b-1))
                wc = int(rng.integers(b+1, w-b-1))
                cells = _body(wr, wc, b)
                if (not cells.isdisjoint(all_sweeps) or
                        any(not cells.isdisjoint(old[4]) for old in barriers)):
                    continue
                color = wall_colors[len(barriers)]
                barriers.append((color, wr, wc, b, cells))
                placed = True
                break
            if not placed:
                break
        if len(barriers) != barrier_count:
            continue

        density = float(rng.uniform(.30, .52))
        grid = np.where(rng.random(shape) < density, filler, 0).astype(np.int8)
        grid[0, :] = grid[-1, :] = 0
        grid[:, 0] = grid[:, -1] = 0
        for color, wr, wc, b, cells in barriers:
            forbidden = hidden_contacts.get(color, frozenset())
            available_tips = sorted(_tips(wr, wc, b) - forbidden)
            rng.shuffle(available_tips)
            clues = set(available_tips[:3])
            pool = sorted(cells - clues - forbidden)
            maximum = min(len(cells)-2, len(cells-forbidden))
            count = int(rng.integers(3, maximum+1))
            for index in rng.permutation(len(pool))[:count-3]:
                clues.add(pool[int(index)])
            for p in cells:
                grid[p] = color if p in clues else filler
        for v in vehicles:
            for p in _body(*v):
                grid[p] = vehicle_color
        # Explicitly expose passability of ordinary filler on a free route.
        free_index = 1 if len(vehicles) > 1 else 0
        passable = sorted((_at(vehicles[free_index], 1, (1, 1)) - all_sources) -
                          frozenset().union(*(b[4] for b in barriers)))
        passable = [p for p in passable if 0 < p[0] < h-1 and 0 < p[1] < w-1]
        if passable:
            grid[passable[int(rng.integers(len(passable)))]] = filler
        command = _marker(shape, 0, axis, length)
        for p in command:
            grid[p] = vehicle_color
        output = _paint(grid, vehicle_color, command, vehicles, barriers, length, (1, 1))
        # Check parseability while construction geometry is still available.
        parsed = _parse(grid)
        _require(set(parsed["vehicles"]) == set(vehicles), "constructed vehicle mismatch")
        _require({(b[0], b[1], b[2], b[3]) for b in parsed["barriers"]} ==
                 {(b[0], b[1], b[2], b[3]) for b in barriers}, "constructed barrier mismatch")
        return grid, output, vehicles, barriers
    raise RuntimeError("bounded scene construction exhausted")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(np.arange(1, 10), size=6, replace=False)))
    radii = tuple(sorted(map(int, rng.choice([2, 3, 4], size=2, replace=False))))
    primary = int(rng.integers(4))
    secondary = int(rng.choice([i for i in range(4) if i != primary]))
    axis = int(rng.integers(2))
    inp, out, vehicles, barriers = _scene(
        rng, palette, radii, 3, axis, 2, int(rng.integers(2, 4)),
        mode="block", hit=2, controlled=True)
    short = inp.copy()
    for p in _marker(short.shape, 0, axis, 3):
        short[p] = 0
    short[0, 0] = palette[1]
    short_out = _paint(short, palette[1], {(0, 0)}, vehicles, barriers, 1, (1, 1))
    puzzle = {"train": [
        {"input": _reflect(short, primary), "output": _reflect(short_out, primary)},
        {"input": _reflect(inp, primary), "output": _reflect(out, primary)}], "test": []}
    commands = [(primary, axis, 1), (primary, axis, 3)]
    used = {_key(p["input"]) for p in puzzle["train"]}
    train_shapes = {inp.shape}
    for index in range(2, nt):
        if index == 2:
            corner, edge, length, mode, movers, hit = secondary, axis, 2, "block", 2, 1
        elif index == 3:
            corner, edge, length, mode, movers, hit = secondary, 1-axis, 6, "tunnel", 2, 2
        elif index == 4:
            corner, edge, length, mode, movers, hit = primary, axis, 3, "boundary", 2, 1
        else:
            corner, edge = int(rng.integers(4)), int(rng.integers(2))
            length = int(rng.choice([1, 2, 3, 6]))
            mode = "boundary" if length > 1 and rng.random() < .25 else "block"
            movers = int(rng.integers(1, 4))
            hit = int(rng.integers(1, length+1))
        walls = int(rng.integers(1, 4))
        if movers == 3:
            walls = min(walls, 2)
        for _ in range(10):
            before, after, _, _ = _scene(rng, palette, radii, length, edge, movers,
                                         walls, mode=mode, hit=hit)
            before, after = _reflect(before, corner), _reflect(after, corner)
            if _key(before) not in used:
                break
        else:
            raise RuntimeError("could not generate a distinct training input")
        used.add(_key(before))
        train_shapes.add(before.shape)
        commands.append((corner, edge, length))
        puzzle["train"].append({"input": before, "output": after})
    train_corners = sorted({c for c, _, _ in commands})
    lengths = sorted({length for _, _, length in commands})
    max_train_vehicles = max(len(_parse(p["input"])["vehicles"]) for p in puzzle["train"])
    for _index in range(ne):
        corner = int(rng.choice(train_corners))
        edge = axis if nt < 4 else int(rng.integers(2))
        length = int(rng.choice(lengths))
        movers = int(rng.choice([2, 3]))
        walls = (4 if movers == 2 or max_train_vehicles == 3
                 else int(rng.choice([3, 4])))
        mode = "boundary" if nt >= 5 and length > 1 and rng.random() < .20 else "block"
        hit = int(rng.integers(1, min(3, length)+1))
        for _ in range(10):
            before, after, _, _ = _scene(
                rng, palette, radii, length, edge, movers, walls,
                mode=mode, hit=hit, excluded_shapes=train_shapes)
            before, after = _reflect(before, corner), _reflect(after, corner)
            if _key(before) not in used:
                break
        else:
            raise RuntimeError("could not generate a distinct test input")
        used.add(_key(before))
        puzzle["test"].append({"input": before, "output": after})
    validate_generated_puzzle(puzzle)
    return puzzle


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _effects(scene):
    grid = scene["grid"]
    visible = frozenset(p for b in scene["barriers"] for p in b[4] if grid[p] == b[0])
    records = []
    for vehicle in scene["vehicles"]:
        distance, reason = _travel(vehicle, scene["length"], scene["direction"],
                                   scene["solid"], grid.shape)
        clue_distance, _ = _travel(vehicle, scene["length"], scene["direction"], visible, grid.shape)
        point_distance, _ = _travel((*vehicle[:2], 0), scene["length"], scene["direction"],
                                    scene["solid"], grid.shape)
        endpoint = _at(vehicle, scene["length"], scene["direction"])
        through_noise = any(grid[p] == scene["filler"] for p in
                            (_sweep(vehicle, distance, scene["direction"]) - _body(*vehicle))
                            if _inside([p], grid.shape) and p not in scene["solid"])
        records.append({"distance": distance, "reason": reason,
                        "hidden_contact": clue_distance != distance,
                        "body_matters": point_distance != distance,
                        "tunnel": reason == "barrier" and _inside(endpoint, grid.shape)
                                  and endpoint.isdisjoint(scene["solid"]),
                        "through_noise": through_noise})
    return records


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    scenes, effects, seen = [], [], set()
    for pair in puzzle["train"] + puzzle["test"]:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
        inp, out = _grid(pair["input"]), _grid(pair["output"])
        _require(inp.shape == out.shape, "canvas changed")
        _require(_key(inp) not in seen, "duplicate input")
        seen.add(_key(inp))
        scene = _parse(inp)
        _require(np.array_equal(solve(inp.copy()), out), "incorrect completed-barrier motion output")
        _require(not np.array_equal(inp, out), "identity pair")
        # Verify exact preservation outside the explicitly affected footprints.
        footprint = set(scene["command"]) | set(scene["solid"])
        for v in scene["vehicles"]:
            distance, _ = _travel(v, scene["length"], scene["direction"], scene["solid"], inp.shape)
            footprint.update(_body(*v))
            footprint.update(_at(v, distance, scene["direction"]))
        _require(all(inp[p] == out[p] for p in np.ndindex(inp.shape) if p not in footprint),
                 "unrelated field content changed")
        scenes.append(scene)
        effects.append(_effects(scene))
    train, test = scenes[:nt], scenes[nt:]
    role_pairs = {(s["filler"], s["vehicle_color"]) for s in scenes}
    _require(len(role_pairs) == 1, "filler or vehicle role changed within puzzle")
    clue_colors = {b[0] for s in scenes for b in s["barriers"]}
    _require(len(clue_colors) <= 4, "too many puzzle-wide barrier colors")
    _require(train[0]["length"] == 1 and train[1]["length"] == 3,
             "controlled unary contrast missing")
    _require(train[0]["corner"] == train[1]["corner"] and
             train[0]["vehicles"] == train[1]["vehicles"] and
             train[0]["barriers"] == train[1]["barriers"], "controlled geometry changed")
    first, second = train[0]["grid"].copy(), train[1]["grid"].copy()
    for s, g in ((train[0], first), (train[1], second)):
        for p in s["command"]:
            g[p] = 0
    _require(np.array_equal(first, second), "controlled floor or clues changed")
    _require(all(e["distance"] == 1 for e in effects[0]), "short lesson should move both vehicles")
    _require(any(e["distance"] == 3 and e["through_noise"] for e in effects[1]),
             "free vehicle must demonstrate unary travel through filler")
    _require(any(e["distance"] == 1 and e["hidden_contact"] and e["body_matters"]
                 for e in effects[1]), "concealed rigid-body collision lesson missing")
    if nt >= 3:
        _require(train[2]["length"] == 2 and train[2]["corner"] != train[0]["corner"],
                 "second corner lesson missing")
    if nt >= 4:
        _require(any(e["tunnel"] for e in effects[3]) and train[3]["length"] == 6,
                 "clear-endpoint collision lesson missing")
    if nt >= 5:
        _require(any(e["reason"] == "boundary" for e in effects[4]), "boundary lesson missing")
    train_lengths = {s["length"] for s in train}
    train_corners = {s["corner"] for s in train}
    demonstrated_radii = {b[3] for s in train for b in s["barriers"]}
    has_boundary = any(e["reason"] == "boundary" for group in effects[:nt] for e in group)
    for i, s in enumerate(test):
        _require(s["length"] in train_lengths and s["corner"] in train_corners,
                 "test command was not demonstrated")
        _require(s["grid"].shape not in {t["grid"].shape for t in train}, "test needs a new canvas shape")
        _require(len(s["vehicles"]) > max(len(t["vehicles"]) for t in train) or
                 len(s["barriers"]) > max(len(t["barriers"]) for t in train),
                 "test must exceed a demonstrated object count")
        _require(all(b[3] in demonstrated_radii for b in s["barriers"]), "test barrier scale not demonstrated")
        if any(e["reason"] == "boundary" for e in effects[nt+i]):
            _require(has_boundary, "test boundary stop lacks demonstration")
        if nt < 4:
            _require(bool(s["axes"] & train[1]["axes"]), "new command edge without demonstration")
    return None


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def self_test():
    """Bounded, deterministic family checks with descriptive diversity metrics."""
    shapes, palettes, fingerprints = set(), set(), set()
    corners, axes, radii, mover_radii, object_counts, budgets = set(), set(), set(), set(), set(), set()
    hidden = body = tunnels = boundaries = free_noise = 0
    clue_counts, densities, guard_values = set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                s = _parse(pair["input"])
                shapes.add(s["grid"].shape)
                palettes.add((s["filler"], s["vehicle_color"], tuple(b[0] for b in s["barriers"])))
                corners.add(s["corner"])
                if s["length"] > 1:
                    axes.update(s["axes"])
                budgets.add(s["length"])
                radii.update(b[3] for b in s["barriers"])
                mover_radii.update(v[2] for v in s["vehicles"])
                object_counts.add((len(s["vehicles"]), len(s["barriers"])))
                densities.append(float(np.mean(s["grid"] == s["filler"])))
                for b in s["barriers"]:
                    clue_counts.add(sum(s["grid"][p] == b[0] for p in b[4]))
                    for p in _tips(b[1], b[2], b[3]+1):
                        if _inside([p], s["grid"].shape):
                            guard_values.add(int(s["grid"][p]))
                for e in _effects(s):
                    hidden += e["hidden_contact"]
                    body += e["body_matters"]
                    tunnels += e["tunnel"]
                    boundaries += e["reason"] == "boundary"
                    free_noise += e["through_noise"] and e["reason"] == "budget"
        if seed < 4:
            pair = puzzle["test"][0]
            original = pair["input"].copy()
            _require(np.array_equal(solve(original), pair["output"]) and
                     np.array_equal(original, pair["input"]), "solve mutates input")
            _require(np.array_equal(solve(np.rot90(original)), np.rot90(pair["output"])),
                     "rotation equivariance failed")
            perm = np.r_[0, np.random.default_rng(seed+500).permutation(np.arange(1, 10))]
            _require(np.array_equal(solve(perm[original]), perm[pair["output"]]),
                     "nonzero palette permutation failed")
    _require(len(fingerprints) == 42 and len(shapes) >= 35 and len(palettes) >= 40,
             "insufficient seed, geometry or palette diversity")
    _require(corners == {0, 1, 2, 3} and axes == {0, 1} and radii == {2, 3, 4} and
             mover_radii == {1, 2} and budgets == {1, 2, 3, 6}, "role/geometry coverage missing")
    _require(len(object_counts) >= 5 and len(clue_counts) >= 8 and
             max(densities)-min(densities) > .12 and len(guard_values) >= 8,
             "object, clue, density or beyond-tip variation missing")
    _require(hidden >= 42 and body >= 42 and tunnels >= 42 and free_noise >= 42,
             "a parent interaction is not exercised")
    # All legal explicit count combinations, including the minimum regime.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(10000+100*nt+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    deterministic = generate_puzzle(917372413, 6, 3)
    _require(_fingerprint(deterministic) == _fingerprint(generate_puzzle(917372413, 6, 3)),
             "generation is not deterministic")
    # Extra counts exercise boundary behavior absent from the four-train defaults.
    _require(any(e["reason"] == "boundary" for p in deterministic["train"]
                 for e in _effects(_parse(p["input"]))), "boundary case absent")
    damaged = {k: [{role: grid.copy() for role, grid in p.items()} for p in pairs]
               for k, pairs in deterministic.items()}
    damaged["test"][0]["output"][0, 0] = _parse(damaged["test"][0]["input"])["filler"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "shapes": len(shapes),
            "palettes": len(palettes), "object_count_combinations": len(object_counts),
            "hidden_contact_vehicles": int(hidden), "body_sensitive_vehicles": int(body),
            "clear_endpoint_blocked_vehicles": int(tunnels),
            "free_vehicles_crossing_noise": int(free_noise),
            "observed_density_range": [min(densities), max(densities)],
            "note": "Generator-owned checks; no independent semantic approval."}
