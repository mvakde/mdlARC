"""A fuelled cartridge follows a route through identity-keyed orientation bends.

Only NumPy and the standard library are used.  solve() reads one input; generation
parameters and the demonstration outputs are never needed to decode it.
"""
from __future__ import annotations

import numpy as np

BACKGROUND, SPENT, FUEL, FRAME = 0, 1, 2, 8
SIDES = ("T", "R", "B", "L")
NORMALS = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
SIDE_MAPS = {
    "identity": dict(zip(SIDES, ("T", "R", "B", "L"))),
    "r90": dict(zip(SIDES, ("L", "T", "R", "B"))),
    "r180": dict(zip(SIDES, ("B", "L", "T", "R"))),
    "r270": dict(zip(SIDES, ("R", "B", "L", "T"))),
    "flip_lr": dict(zip(SIDES, ("T", "L", "B", "R"))),
    "flip_ud": dict(zip(SIDES, ("B", "R", "T", "L"))),
    "transpose": dict(zip(SIDES, ("L", "B", "R", "T"))),
    "anti_transpose": dict(zip(SIDES, ("R", "T", "L", "B"))),
}
ROTATIONS = {"r90", "r270"}
REFLECTIONS = {"flip_lr", "flip_ud", "transpose", "anti_transpose"}
HOST_SEEDS = (0, 555474185, 50590113, 273617111, 720368853, 695151001,
              112487735, 316688629, 237684261, 130590765, 661684217,
              33662317, 109552998, 980719588, 337050559, 444236318,
              281833512, 338718850, 217696501, 325873714, 332917074,
              708586531, 318082754, 642495055, 214487854, 719028322,
              833449145, 98924951, 667754179, 92932162, 944514903,
              199125831, 881052697, 94870199, 766203015, 907180993,
              120044589, 132630124, 706604728, 549298445, 187808706,
              845406705)


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "invalid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "cells must be integers")
    _need(np.all((0 <= g) & (g <= 9)), "invalid colors")
    return g.astype(np.int8, copy=False)


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _pick(rng, items):
    return items[int(rng.integers(len(items)))]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _neighbors(p, shape):
    return [_add(p, d) for d in NORMALS.values()
            if 0 <= p[0] + d[0] < shape[0] and 0 <= p[1] + d[1] < shape[1]]


def _side_between(a, b):
    delta = (b[0] - a[0], b[1] - a[1])
    return next(s for s, d in NORMALS.items() if d == delta)


def _transform(a, name):
    if name == "identity":
        b = a
    elif name.startswith("r"):
        b = np.rot90(a, int(name[1:]) // 90)
    elif name == "flip_lr":
        b = np.fliplr(a)
    elif name == "flip_ud":
        b = np.flipud(a)
    elif name == "transpose":
        b = a.T
    elif name == "anti_transpose":
        b = np.rot90(a.T, 2)
    else:
        raise ValueError("unknown symmetry")
    return b.copy()


def _symmetry(current, target):
    choices = [name for name, mapping in SIDE_MAPS.items()
               if all(target.get(mapping[side]) == color for side, color in current.items())]
    _need(len(choices) == 1, "keys do not define a unique symmetry")
    return choices[0]


def _components(mask):
    seen = set()
    result = []
    for raw in np.argwhere(mask):
        start = tuple(map(int, raw))
        if start in seen:
            continue
        component, stack = [], [start]
        seen.add(start)
        while stack:
            p = stack.pop()
            component.append(p)
            for q in _neighbors(p, mask.shape):
                if mask[q] and q not in seen:
                    seen.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _bar_positions(bounds, side):
    t, l, h, w = bounds
    if side == "T":
        return [(t - 1, x) for x in range(l, l + w)]
    if side == "B":
        return [(t + h, x) for x in range(l, l + w)]
    if side == "L":
        return [(y, l - 1) for y in range(t, t + h)]
    return [(y, l + w) for y in range(t, t + h)]


def _scan_positions(shape, direction):
    """Rotate travel upward, then raster; return coordinates in the given block."""
    h, w = shape
    if direction == "T":
        return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    if direction == "B":
        return [(r, c) for r in range(h - 2, 0, -1) for c in range(w - 2, 0, -1)]
    if direction == "L":
        return [(r, c) for c in range(1, w - 1) for r in range(h - 2, 0, -1)]
    return [(r, c) for c in range(w - 2, 0, -1) for r in range(1, h - 1)]


def _cargo_signature(block):
    cargo = block.copy()
    cargo[(cargo == SPENT) | (cargo == FUEL)] = -1
    return cargo


def _asymmetric_cargo(block):
    cargo = _cargo_signature(block)
    return len({_key(_transform(cargo, name)) for name in SIDE_MAPS}) == 8


def _decode(input_grid):
    g = _grid(input_grid)
    frame_components = _components(g == FRAME)
    rectangles = []
    for component in frame_components:
        yy, xx = zip(*component)
        t, b, l, r = min(yy), max(yy), min(xx), max(xx)
        h, w = b - t + 1, r - l + 1
        if h < 4 or w < 4 or len(component) != 2 * (h + w) - 4:
            continue
        if any(y not in (t, b) and x not in (l, r) for y, x in component):
            continue
        if FRAME in g[t + 1:b, l + 1:r]:
            continue
        rectangles.append((t, l, h, w))
    _need(len(rectangles) == 1, "there must be one framed cartridge")
    bounds = rectangles[0]
    t, l, h, w = bounds
    _need(t >= 1 and l >= 1 and t + h < g.shape[0] and l + w < g.shape[1],
          "source requires room for its guides")
    block = g[t:t + h, l:l + w].copy()
    guides = {}
    guide_cells = set()
    for side in SIDES:
        positions = _bar_positions(bounds, side)
        colors = {int(g[p]) for p in positions}
        if len(colors) == 1 and next(iter(colors)) not in (BACKGROUND, SPENT, FUEL, FRAME):
            guides[side] = next(iter(colors))
            guide_cells.update(positions)
    _need(len(guides) == 2 and len(set(guides.values())) == 2, "source needs two identities")
    guide_sides = list(guides)
    _need(np.dot(NORMALS[guide_sides[0]], NORMALS[guide_sides[1]]) == 0,
          "source guides must be adjacent")
    markers = set(guides.values())
    _need(not markers & set(map(int, np.unique(block))), "keys must not occur inside cartridge")
    outside = np.ones(g.shape, dtype=bool)
    outside[t:t + h, l:l + w] = False
    route_colors = set(map(int, np.unique(g[outside]))) - markers - {BACKGROUND, FRAME}
    _need(len(route_colors) == 1, "there must be one locally identifiable route color")
    track = next(iter(route_colors))
    _need(track not in (SPENT, FUEL), "route cannot use a counter color")
    _need(track not in block, "route color is reserved outside cartridge")
    terminal = [p for component in frame_components for p in component
                if not (t <= p[0] < t + h and l <= p[1] < l + w)]
    _need(len(terminal) == 1, "one single-cell terminal is required")
    end = terminal[0]
    cells = {tuple(map(int, p)) for p in np.argwhere(g == track)} | {end}
    source_cells = {(y, x) for y in range(t, t + h) for x in range(l, l + w)}
    starts = [p for p in cells if any(q in source_cells for q in _neighbors(p, g.shape))]
    _need(len(starts) == 1 and starts[0] != end, "route must attach once to the cartridge")
    start = starts[0]
    degree = {p: sum(q in cells for q in _neighbors(p, g.shape)) for p in cells}
    _need(degree[start] == degree[end] == 1 and
          all(degree[p] == 2 for p in cells - {start, end}), "route branches or touches itself")
    route, previous, current = [], None, start
    while True:
        _need(current not in route, "route contains a cycle")
        route.append(current)
        if current == end:
            break
        options = [q for q in _neighbors(current, g.shape) if q in cells and q != previous]
        _need(len(options) == 1, "route order is ambiguous")
        previous, current = current, options[0]
    _need(len(route) == len(cells) and len(route) >= 3, "unconnected route material")
    direction = _side_between(route[0], route[1])
    attached = [q for q in _neighbors(start, g.shape) if q in source_cells]
    _need(len(attached) == 1 and _side_between(attached[0], start) == direction,
          "route must initially leave source straight")
    keys, key_cells, bends = {}, set(), []
    for index in range(1, len(route) - 1):
        p = route[index]
        used = {_side_between(p, route[index - 1]), _side_between(p, route[index + 1])}
        used_sides = list(used)
        bend = np.dot(NORMALS[used_sides[0]], NORMALS[used_sides[1]]) == 0
        if bend:
            bends.append(index)
        local = {s: int(g[_add(p, NORMALS[s])]) for s in SIDES
                 if s not in used and _add(p, NORMALS[s]) in _neighbors(p, g.shape)
                 and int(g[_add(p, NORMALS[s])]) in markers}
        if local:
            _need(bend and len(local) == 2 and set(local.values()) == markers,
                  "a checkpoint must have both identities on unused bend sides")
            keys[index] = local
            key_cells.update(_add(p, NORMALS[s]) for s in local)
    _need(keys, "route must contain a keyed bend")
    expected = np.zeros_like(g)
    expected[t:t + h, l:l + w] = block
    for side, color in guides.items():
        for p in _bar_positions(bounds, side):
            expected[p] = color
    for p in route:
        expected[p] = FRAME if p == end else track
    for index, local in keys.items():
        for side, color in local.items():
            expected[_add(route[index], NORMALS[side])] = color
    _need(np.array_equal(g, expected), "unexplained or overlapping input material")
    for p in key_cells:
        _need(sum(q in cells for q in _neighbors(p, g.shape)) == 1,
              "key must touch only its own checkpoint")
        _need(not any(q in guide_cells for q in _neighbors(p, g.shape)), "key touches source guide")
    ordered = [int(block[p]) for p in _scan_positions(block.shape, direction)
               if int(block[p]) in (SPENT, FUEL)]
    spent = ordered.count(SPENT)
    _need(ordered and ordered == [SPENT] * spent + [FUEL] * (len(ordered) - spent),
          "initial counter is not a travel-relative spent prefix")
    return dict(grid=g, bounds=bounds, block=block, guides=guides, track=track,
                route=route, keys=keys, bends=bends, fuel=ordered.count(FUEL),
                capacity=len(ordered), spent=spent)


def _journey(info, mode="normal"):
    """Modes other than normal are deliberately wrong shortcut witnesses."""
    block, guides = info["block"].copy(), dict(info["guides"])
    distance = len(info["route"]) - 1
    moves = distance if mode == "unlimited" else min(info["fuel"], distance)
    operations = []
    for index in range(1, moves + 1):
        direction = _side_between(info["route"][index - 1], info["route"][index])
        if mode == "fixed_scan":
            direction = _side_between(info["route"][0], info["route"][1])
        if index in info["keys"] and mode == "early_keys":
            target = info["keys"][index]
            name = _symmetry(guides, target)
            block = _transform(block, name)
            guides = dict(target)
            operations.append((index, name))
        pending = [p for p in _scan_positions(block.shape, direction) if block[p] == FUEL]
        if pending:
            block[pending[0]] = SPENT
        if index in info["keys"] and mode not in ("no_keys", "defer_keys", "early_keys"):
            target = info["keys"][index]
            name = _symmetry(guides, target)
            block = _transform(block, name)
            guides = dict(target)
            operations.append((index, name))
    if mode == "defer_keys":
        reached = [i for i in info["keys"] if i <= moves]
        if reached:
            block = _transform(block, _symmetry(info["guides"], info["keys"][max(reached)]))
    return block, dict(movement=moves, operations=operations, guides=guides,
                       fuel_left=int(np.count_nonzero(block == FUEL)))


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _journey(_decode(input_grid))[0]


def _make_cargo(rng, h, w, colors):
    for _ in range(80):
        block = np.full((h + 2, w + 2), FRAME, dtype=np.int8)
        block[1:-1, 1:-1] = SPENT
        n = int(rng.integers(3, min(h * w - 2, 9) + 1))
        positions = rng.choice(h * w, size=n, replace=False)
        for i, flat in enumerate(positions):
            block[1 + int(flat) // w, 1 + int(flat) % w] = colors[i % len(colors)]
        # Holes are payload cells too; they are skipped by the fuel raster.
        if rng.random() < .25:
            free = np.argwhere(block[1:-1, 1:-1] == SPENT)
            rr, cc = map(int, free[int(rng.integers(len(free)))])
            block[rr + 1, cc + 1] = BACKGROUND
        if _asymmetric_cargo(block):
            return block
    return None


def _make_route(rng, start, blocked, segments, short):
    # Random orthogonal polylines; no fixed rows, slots, or lane bank.
    for _ in range(100):
        route, bends = [start], []
        delta = NORMALS["R"]
        valid = True
        for segment in range(segments):
            length = int(rng.integers(3, 5 if short else 9))
            for _ in range(length):
                q = _add(route[-1], delta)
                if not (1 <= q[0] <= 28 and 1 <= q[1] <= 28) or blocked[q]:
                    valid = False
                    break
                if q in route or any(abs(q[0] - p[0]) + abs(q[1] - p[1]) == 1
                                     for p in route[:-1]):
                    valid = False
                    break
                route.append(q)
            if not valid:
                break
            if segment < segments - 1:
                bends.append(len(route) - 1)
                perpendicular = ((delta[1], -delta[0]), (-delta[1], delta[0]))
                delta = _pick(rng, perpendicular)
        if not valid:
            continue
        # Reserve both potential key positions at every bend, including plain
        # bends, so later sampling never needs to repair a crowded checkpoint.
        occupied = set(route)
        free_by_bend = {}
        for i in bends:
            used = {_side_between(route[i], route[i - 1]), _side_between(route[i], route[i + 1])}
            free = [s for s in SIDES if s not in used]
            positions = [_add(route[i], NORMALS[s]) for s in free]
            if any(not (1 <= p[0] <= 28 and 1 <= p[1] <= 28) or blocked[p] or p in occupied
                   or sum(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for q in route) != 1
                   for p in positions):
                valid = False
                break
            free_by_bend[i] = free
        if valid:
            return route, bends, free_by_bend
    return None


def _sample_pair(rng, palette, lesson, is_test=False):
    # Bounded rejection of geometry that cannot support the requested evidence.
    test_regime = (_pick(rng, ("destination", "destination", "fuel", "fuel", "exact", "on_key"))
                   if is_test else None)
    for attempt in range(300):
        teaching = lesson in ("destination", "fuel")
        low = 4 if teaching or is_test else 2
        h, w = int(rng.integers(low, 8)), int(rng.integers(low, 8))
        if h * w < 7:
            continue
        block = _make_cargo(rng, h, w, palette[3:])
        if block is None:
            continue
        outer_h, outer_w = block.shape
        t = int(rng.integers(3, 23 - outer_h))
        l = int(rng.integers(2, 5))
        bounds = t, l, outer_h, outer_w
        start = (t + outer_h // 2, l + outer_w)
        keyed_count = int(rng.integers(3, 5)) if is_test else (2 if teaching else int(rng.integers(1, 3)))
        ordinary_count = int(rng.integers(0, 3)) if not teaching else 0
        segments = keyed_count + ordinary_count + 1
        blocked = np.zeros((30, 30), dtype=bool)
        blocked[t - 2:t + outer_h + 2, l - 2:l + outer_w + 2] = True
        # The first two cells outside the exit are the one permitted opening.
        blocked[start] = False
        blocked[_add(start, NORMALS["R"])] = False
        short = teaching or attempt > 150 or (test_regime == "destination" and rng.random() < .5)
        sampled = _make_route(rng, start, blocked, segments, short=short)
        if sampled is None:
            continue
        route, bends, free = sampled
        if teaching:
            key_indices = bends
        else:
            key_indices = sorted(map(int, rng.choice(bends, keyed_count, replace=False)))
        # At a teaching quarter-turn, choose the source corner compatible with
        # the first bend's geometry. This is observable, not a hidden convention.
        first_target = free[key_indices[0]]
        source_hside = "T" if "T" in first_target else "B"
        if not teaching and rng.random() < .5:
            source_hside = "B" if source_hside == "T" else "T"
        source_guides = {"L": palette[1], source_hside: palette[2]}
        current = dict(source_guides)
        keys = {}
        for j, index in enumerate(key_indices):
            sides = free[index]
            candidates = [dict(zip(sides, (palette[1], palette[2]))),
                          dict(zip(sides, (palette[2], palette[1])))]
            if teaching and j == 0:
                eligible = ROTATIONS if lesson == "destination" else REFLECTIONS
                candidates = [c for c in candidates if _symmetry(current, c) in eligible]
            elif lesson == "destination" and j == 1:
                candidates = [c for c in candidates if _symmetry(current, c) in REFLECTIONS]
            if not candidates:
                break
            keys[index] = dict(_pick(rng, candidates))
            current = keys[index]
        if len(keys) != keyed_count:
            continue
        capacity = int(np.count_nonzero(block == SPENT))
        distance = len(route) - 1
        if lesson == "destination":
            if capacity < distance + 2:
                continue
            fuel = int(rng.integers(distance + 1, capacity))
        elif lesson == "fuel":
            lower, upper = key_indices[0] + 2, min(capacity - 1, key_indices[1] - 1)
            if lower > upper:
                continue
            fuel = int(rng.integers(lower, upper + 1))
        elif is_test:
            # More keys in the map, and at least two actually reached. Both
            # exhaustion and terminal arrival remain possible at every pose.
            required = 2 if keyed_count == 3 else 3
            lower = key_indices[required - 1]
            if capacity < lower:
                continue
            if test_regime == "on_key":
                fuel = lower
            elif test_regime == "destination":
                if capacity < distance + 2:
                    continue
                fuel = int(rng.integers(distance + 1, capacity + 1))
            elif test_regime == "exact":
                if capacity < distance:
                    continue
                fuel = distance
            else:
                upper = min(capacity, distance - 1)
                if lower > upper:
                    continue
                fuel = int(rng.integers(lower, upper + 1))
        else:
            regime = _pick(rng, ("fuel", "destination", "exact", "fresh", "zero"))
            if regime == "zero":
                fuel = 0
            elif regime == "fresh":
                fuel = capacity
            elif regime == "destination" and capacity > distance:
                fuel = int(rng.integers(distance + 1, capacity + 1))
            elif regime == "exact" and capacity >= distance:
                fuel = distance
            else:
                fuel = int(rng.integers(1, min(capacity, distance) + 1))
        grid = np.zeros((30, 30), dtype=np.int8)
        grid[t:t + outer_h, l:l + outer_w] = block
        for side, color in source_guides.items():
            for p in _bar_positions(bounds, side):
                grid[p] = color
        for p in route:
            grid[p] = palette[0]
        grid[route[-1]] = FRAME
        for index, local in keys.items():
            for side, color in local.items():
                grid[_add(route[index], NORMALS[side])] = color
        # Crop asymmetrically with a minimum one-cell exterior margin. Then
        # freely rotate or reflect the scene. Rebuild the initial prefix in the
        # resulting travel-relative raster; reflecting A's bits alone is wrong.
        yy, xx = np.nonzero(grid)
        top = max(0, int(yy.min()) - int(rng.integers(1, 4)))
        left = max(0, int(xx.min()) - int(rng.integers(1, 4)))
        bottom = min(30, int(yy.max()) + int(rng.integers(2, 5)))
        right = min(30, int(xx.max()) + int(rng.integers(2, 5)))
        grid = _transform(grid[top:bottom, left:right], str(_pick(rng, tuple(SIDE_MAPS))))
        try:
            info = _decode(grid)
            r, c, bh, bw = info["bounds"]
            cartridge = grid[r:r + bh, c:c + bw]
            scan = [p for p in _scan_positions(cartridge.shape, _side_between(info["route"][0], info["route"][1]))
                    if cartridge[p] == SPENT]
            for i, p in enumerate(scan):
                cartridge[p] = SPENT if i < capacity - fuel else FUEL
            info = _decode(grid)
            answer, journey = _journey(info)
            if lesson == "destination":
                if any(np.array_equal(answer, _journey(info, mode)[0])
                       for mode in ("defer_keys", "no_keys", "early_keys", "fixed_scan")):
                    continue
            if lesson == "fuel" and np.array_equal(answer, _journey(info, "unlimited")[0]):
                continue
            if test_regime == "destination" and np.array_equal(answer, _journey(info, "defer_keys")[0]):
                continue
            return dict(input=grid.copy(), output=answer)
        except AssertionError:
            continue
    raise RuntimeError("bounded route construction failed")


def _counts(num_train, num_test):
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
           for n in (num_train, num_test)):
        raise ValueError("example counts must be integers")
    if not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return int(num_train), int(num_test)


def _palette(rng):
    # Counter states and the frame/terminal have fixed meanings. The other six
    # colors exchange route, marker and cargo roles, inferable from geometry.
    colors = list(map(int, rng.permutation([3, 4, 5, 6, 7, 9])))
    return colors[:3] + colors[3:3 + int(rng.integers(1, 4))]


def _swapped_contrast(pair):
    g = pair["input"].copy()
    info = _decode(g)
    last = max(i for i in info["keys"] if i <= min(info["fuel"], len(info["route"]) - 1))
    colors = list(info["keys"][last].values())
    for side, color in info["keys"][last].items():
        g[_add(info["route"][last], NORMALS[side])] = colors[1] if color == colors[0] else colors[0]
    return dict(input=g, output=solve(g))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared = bool(rng.random() < .35)
    common_palette = _palette(rng)
    puzzle = dict(train=[], test=[])
    seen = set()
    destination = _sample_pair(rng, common_palette, "destination")
    fuel = _sample_pair(rng, common_palette if shared else _palette(rng), "fuel")
    puzzle["train"].extend((destination, fuel))
    seen.update(_key(p["input"]) for p in puzzle["train"])
    if nt >= 3:
        contrast = _swapped_contrast(destination)
        puzzle["train"].append(contrast)
        seen.add(_key(contrast["input"]))
    for split, count in (("train", nt), ("test", ne)):
        while len(puzzle[split]) < count:
            for _ in range(20):
                pair = _sample_pair(rng, common_palette if shared else _palette(rng),
                                    "extra", is_test=split == "test")
                if _key(pair["input"]) not in seen:
                    puzzle[split].append(pair)
                    seen.add(_key(pair["input"]))
                    break
            else:
                raise RuntimeError("could not construct a distinct input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), dict(train=[], test=[])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g, output = _grid(pair["input"]), _grid(pair["output"])
            _need(_key(g) not in seen, "duplicate input")
            seen.add(_key(g))
            info = _decode(g)
            expected, result = _journey(info)
            _need(np.array_equal(expected, output), "incorrect journey output")
            moves = min(info["fuel"], len(info["route"]) - 1)
            _need(result["movement"] == moves and result["fuel_left"] == info["fuel"] - moves,
                  "one fuel unit per route step")
            _need(np.count_nonzero(output == SPENT) == info["spent"] + moves,
                  "spent count must advance exactly with movement")
            # Unwind the actual orientations: every cargo/hole and the complete
            # frame must survive unchanged. Only fuel -> spent is permitted.
            unwound = output.copy()
            inverse = {"identity": "identity", "r90": "r270", "r180": "r180", "r270": "r90",
                       "flip_lr": "flip_lr", "flip_ud": "flip_ud", "transpose": "transpose",
                       "anti_transpose": "anti_transpose"}
            for _, name in reversed(result["operations"]):
                unwound = _transform(unwound, inverse[name])
            _need(np.array_equal(_cargo_signature(unwound), _cargo_signature(info["block"])),
                  "cargo, holes, frame or exact rectangle changed")
            _need(len(result["operations"]) == sum(i <= moves for i in info["keys"]),
                  "unreached checkpoint applied, or arrival checkpoint omitted")
            _need(_asymmetric_cargo(info["block"]), "cargo must reveal orientation")
            records[split].append((info, result, expected))
    train = records["train"]
    _need(any(i["fuel"] > len(i["route"]) - 1 and len(r["operations"]) >= 2
              for i, r, _ in train), "training must reach the terminal with surplus fuel")
    _need(any(0 < i["fuel"] < len(i["route"]) - 1 and 0 < len(r["operations"]) < len(i["keys"])
              for i, r, _ in train), "training must exhaust fuel between checkpoints")
    active_operations = {name for _, r, _ in train for _, name in r["operations"]}
    _need(bool(active_operations & ROTATIONS) and bool(active_operations & REFLECTIONS),
          "training must demonstrate a quarter turn and a reflection")
    _need(any(not np.array_equal(answer, _journey(i, "unlimited")[0]) for i, _, answer in train),
          "ignoring A must change a demonstration")
    _need(any(not np.array_equal(answer, _journey(i, "no_keys")[0]) for i, _, answer in train),
          "ignoring B must change a demonstration")
    _need(any(not np.array_equal(answer, _journey(i, "defer_keys")[0]) for i, _, answer in train),
          "training must show interleaving, not only a final symmetry")
    _need(any(not np.array_equal(answer, _journey(i, "early_keys")[0]) for i, _, answer in train),
          "training must show arrival spending before the key")
    _need(any(not np.array_equal(answer, _journey(i, "fixed_scan")[0]) for i, _, answer in train),
          "training must distinguish travel-relative from fixed scanning")
    _need(any(not np.array_equal(answer, _swapped_contrast(dict(input=i["grid"], output=answer))["output"])
              for i, r, answer in train if r["operations"]),
          "marker identities must affect an answer")
    max_train_keys = max(len(i["keys"]) for i, _, _ in train)
    _need(all(len(i["keys"]) > max_train_keys and len(r["operations"]) >= 2
              for i, r, _ in records["test"]), "missing checkpoint-sequence test shift")
    if len(train) >= 3:
        _need(any(np.array_equal(_swapped_contrast(p)["input"], q["input"])
                  for p in puzzle["train"]
                  if _decode(p["input"])["fuel"] > len(_decode(p["input"])["route"]) - 1
                  for q in puzzle["train"] if p is not q), "missing controlled identity contrast")


def self_test():
    """Bounded checks on 42 named seeds, every count, and semantic mutations."""
    features = []
    signatures = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p["input"]) for split in puzzle.values() for p in split)
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                output, journey = _journey(info)
                features.append(dict(split=split, shape=pair["input"].shape,
                                     cartridge=info["block"].shape, keys=len(info["keys"]),
                                     bends=len(info["bends"]), distance=len(info["route"]) - 1,
                                     capacity=info["capacity"], fuel=info["fuel"],
                                     source=tuple(sorted(info["guides"])),
                                     track=info["track"], markers=tuple(sorted(info["guides"].values())),
                                     operations=tuple(name for _, name in journey["operations"]),
                                     movement=journey["movement"], remaining=journey["fuel_left"],
                                     hole=bool(np.any(info["block"] == BACKGROUND)),
                                     on_key=journey["movement"] in info["keys"]))
                # solve must accept a fresh array without changing the input.
                copy = pair["input"].copy()
                _need(np.array_equal(solve(copy), output) and np.array_equal(copy, pair["input"]),
                      "solve state leak or input mutation")
    _need(len(signatures) == 42, "seed variation collapsed")
    _need(len({f["shape"] for f in features}) >= 35, "canvas variation collapsed")
    _need(len({f["cartridge"] for f in features}) >= 12, "cartridge geometry collapsed")
    _need(len({f["distance"] for f in features}) >= 15, "route-length variation collapsed")
    _need({f["keys"] for f in features} >= {1, 2, 3, 4}, "key-count variation collapsed")
    _need({f["source"] for f in features} == {tuple(sorted((SIDES[j], SIDES[(j + 1) % 4])))
                                              for j in range(4)}, "source corners collapsed")
    _need({name for f in features for name in f["operations"]} == set(SIDE_MAPS),
          "not all eight local symmetries were observed")
    _need(len({f["track"] for f in features}) == 6 and len({f["markers"] for f in features}) >= 10,
          "local palette roles collapsed")
    _need(any(f["hole"] for f in features) and any(f["on_key"] and f["split"] == "test" for f in features),
          "missing payload-hole or exact-checkpoint boundary case")
    _need(any(f["fuel"] == 0 for f in features) and any(f["fuel"] == f["capacity"] for f in features),
          "missing exhausted or fresh initial counters")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(12345 + 17 * nt + ne, nt, ne)
            _need(len(p["train"]) == nt and len(p["test"]) == ne, "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for seed in (0, 123456):
        p, q = generate_puzzle(seed), generate_puzzle(seed)
        _need(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s])
                  for k in ("input", "output")), "nondeterministic puzzle")
    p = generate_puzzle(0)
    for mutation in ("output", "terminal", "key"):
        bad = {s: [{k: g.copy() for k, g in pair.items()} for pair in p[s]] for s in p}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        if mutation == "output":
            pair["output"][1, 1] = (int(pair["output"][1, 1]) + 1) % 10
        elif mutation == "terminal":
            pair["input"][info["route"][-1]] = BACKGROUND
        else:
            i = min(info["keys"])
            side = next(iter(info["keys"][i]))
            pair["input"][_add(info["route"][i], NORMALS[side])] = BACKGROUND
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("validator accepted corrupt material")
    return {"seeds": 42, "pairs": len(features), "canvas_shapes": len({f["shape"] for f in features}),
            "cartridge_shapes": len({f["cartridge"] for f in features}),
            "route_lengths": len({f["distance"] for f in features}),
            "local_symmetries": sorted({name for f in features for name in f["operations"]})}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
