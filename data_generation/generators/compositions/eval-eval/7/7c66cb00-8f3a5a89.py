"""Exterior docks: seeded topology controls color-routed rigid cargo placement.

Only Python's standard library and NumPy are used. No parent code or examples
are needed by generation, validation, or the input-only solver.
"""
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(np.all((g >= 0) & (g <= 9)), "colors outside 0..9")
    return g


def _flood(allowed, start):
    result = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        return result
    result[start] = True
    stack = [start]
    h, w = allowed.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            x, y = r + dr, c + dc
            if 0 <= x < h and 0 <= y < w and allowed[x, y] and not result[x, y]:
                result[x, y] = True
                stack.append((x, y))
    return result


def _components(mask, neighbors=N4):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _exterior(region):
    allowed = np.pad(~region, 1, constant_values=True)
    return _flood(allowed, (0, 0))[1:-1, 1:-1]


def _socket(cells, normal, route=None, rail=None):
    return dict(cells=tuple(cells), normal=normal, route=route, rail=rail)


def _front(gate):
    dr, dc = gate["normal"]
    return tuple((r + dr, c + dc) for r, c in gate["cells"])


def _target(cells, gate):
    """One rigid translation; preserve the coordinate parallel to the dock."""
    dr, dc = gate["normal"]
    r, c = gate["cells"][0]
    if dr:
        leading = min(x for x, y in cells) if dr > 0 else max(x for x, y in cells)
        shift = r + dr - leading
        return tuple((x + shift, y) for x, y in cells)
    leading = min(y for x, y in cells) if dc > 0 else max(y for x, y in cells)
    shift = c + dc - leading
    return tuple((x, y + shift) for x, y in cells)


def _parallel_fit(cells, gate):
    # Endpoint cells are rails; cargo projects strictly between them.
    axis = 1 if gate["normal"][0] else 0
    lo, hi = gate["cells"][0][axis], gate["cells"][-1][axis]
    return all(lo < p[axis] < hi for p in cells)


def _decode(grid):
    g = _grid(grid)
    h, w = g.shape
    wall = int(g[0, 0])
    _require(np.all(g[0] == wall) and np.all(g[-1] == wall) and
             np.all(g[:, 0] == wall) and np.all(g[:, -1] == wall), "uniform outer wall")
    values, counts = np.unique(g, return_counts=True)
    background = int(values[np.argmax(counts)])
    _require(np.count_nonzero(counts == counts.max()) == 1 and background != wall,
             "background must be uniquely most frequent")
    singletons = values[counts == 1]
    _require(len(singletons) == 1, "one uniquely colored singleton seed")
    marker = int(singletons[0])
    start = tuple(map(int, np.argwhere(g == marker)[0]))
    excluded = {background, wall, marker}
    gates = []
    # Read all straight endpoint/interior/endpoint legends, in either axis.
    for axis in (0, 1):
        cross, extent = (h, w) if axis == 0 else (w, h)
        for fixed in range(1, cross - 1):
            for low in range(1, extent - 3):
                p = (fixed, low) if axis == 0 else (low, fixed)
                rail = int(g[p])
                q = (fixed, low + 1) if axis == 0 else (low + 1, fixed)
                route = int(g[q])
                if rail in excluded or route in excluded or route == rail:
                    continue
                high = low + 1
                while high < extent - 1 and int(g[(fixed, high) if axis == 0 else (high, fixed)]) == route:
                    high += 1
                if high - low < 3 or high >= extent - 1:
                    continue
                end = (fixed, high) if axis == 0 else (high, fixed)
                if int(g[end]) != rail:
                    continue
                cells = tuple((fixed, k) if axis == 0 else (k, fixed) for k in range(low, high + 1))
                for normal in ((1, 0), (-1, 0)) if axis == 0 else ((0, 1), (0, -1)):
                    dr, dc = normal
                    if (all(int(g[r - dr, c - dc]) == wall for r, c in cells)
                            and all(int(g[r + dr, c + dc]) != wall for r, c in cells)):
                        gates.append(_socket(cells, normal, route, rail))
    _require(bool(gates), "no docks")
    blocked = g == wall
    dock_cells = set()
    for gate in gates:
        _require(not dock_cells.intersection(gate["cells"]), "overlapping or ambiguous docks")
        dock_cells.update(gate["cells"])
        for p in gate["cells"]:
            blocked[p] = True
    routes = {gate["route"] for gate in gates}
    rails = {gate["rail"] for gate in gates}
    _require(not routes & rails and not (routes | rails) & excluded, "distinct color roles")
    for gate in gates:
        _require(all(not blocked[p] for p in _front(gate)), "dock must face floor")
    region = _flood(~blocked, start)
    outside = _exterior(region)
    active = [gate for gate in gates if all(region[p] for p in _front(gate))
              and all(outside[p] for p in gate["cells"])]
    lookup = {}
    for gate in active:
        _require(gate["route"] not in lookup, "two active docks for one route")
        lookup[gate["route"]] = gate
    cargo = []
    for route in sorted(routes):
        for cells in _components((g == route) & ~blocked):
            selected = bool(region[cells[0]])
            _require(all(bool(region[p]) == selected for p in cells), "cargo crosses a wall")
            cargo.append(dict(color=route, cells=cells, selected=selected))
    _require(bool(cargo), "no cargo")
    free_values = {int(g[p]) for p in map(tuple, np.argwhere(~blocked))}
    _require(free_values <= routes | {background, marker}, "unknown floor color")
    occupied, placements = set(), []
    for comp in cargo:
        if not comp["selected"]:
            continue
        _require(comp["color"] in lookup, "selected cargo lacks an exterior dock")
        gate = lookup[comp["color"]]
        cells = _target(comp["cells"], gate)
        _require(_parallel_fit(cells, gate), "cargo projection crosses a dock endpoint")
        _require(all(0 <= r < h and 0 <= c < w and region[r, c] and (r, c) != start
                     for r, c in cells), "docked mask must fit selected floor and avoid seed")
        _require(not occupied.intersection(cells), "docked cargo collision")
        occupied.update(cells)
        placements.append((comp, gate, cells))
    _require(bool(placements), "no selected cargo")
    return dict(grid=g, wall=wall, background=background, marker=marker, start=start,
                blocked=blocked, gates=gates, active=active, region=region,
                outside=outside, cargo=cargo, placements=placements)


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _decode(input_grid)
    result = a["grid"].copy()
    # All sources clear before any destination is written; placement is simultaneous.
    for comp in a["cargo"]:
        for p in comp["cells"]:
            result[p] = a["background"]
    for comp, gate, cells in a["placements"]:
        for p in cells:
            result[p] = gate["rail"]
    return result


def _locations(walls, region, outside, interior_length, external):
    h, w = walls.shape
    n = interior_length + 2
    result = []
    for axis in (0, 1):
        cross, extent = (h, w) if axis == 0 else (w, h)
        for fixed in range(1, cross - 1):
            for low in range(1, extent - n):
                cells = tuple((fixed, k) if axis == 0 else (k, fixed) for k in range(low, low + n))
                if not all(walls[p] and bool(outside[p]) == external for p in cells):
                    continue
                for normal in ((1, 0), (-1, 0)) if axis == 0 else ((0, 1), (0, -1)):
                    dr, dc = normal
                    if (all(walls[r - dr, c - dc] for r, c in cells) and
                            all(region[r + dr, c + dc] for r, c in cells)):
                        result.append(_socket(cells, normal))
    return result


def _shape(rng, max_h=5, max_w=5):
    h, w = int(rng.integers(1, max_h + 1)), int(rng.integers(1, max_w + 1))
    kind = int(rng.integers(6))
    if kind == 0:
        cells = {(r, c) for r in range(h) for c in range(w)}
    elif kind == 1:
        cells = {(r, 0) for r in range(h)} | {(h - 1, c) for c in range(w)}
    elif kind == 2 and h >= 3 and w >= 3:
        cells = {(r, c) for r in range(h) for c in range(w)
                 if r in (0, h - 1) or c in (0, w - 1)}
    elif kind == 3:
        cells = {(r, w // 2) for r in range(h)} | {(h // 2, c) for c in range(w)}
    else:
        cells = {(0, 0)}
        goal = int(rng.integers(1, h * w + 1))
        while len(cells) < goal:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in N4
                               if 0 <= r + dr < h and 0 <= c + dc < w} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _place_group(rng, g, region, gates, group, bg, occupied, depth_constraint=None, alternate=None):
    h, w = g.shape
    gh = 1 + max(r for color, mask in group for r, c in mask)
    gw = 1 + max(c for color, mask in group for r, c in mask)
    candidates = [(r, c) for r in range(2, h - gh - 1) for c in range(2, w - gw - 1)]
    rng.shuffle(candidates)
    lookup = {gate["route"]: gate for gate in gates}
    for top, left in candidates:
        source, targets, writes = set(), set(), []
        good = True
        for color, mask in group:
            cells = tuple((top + r, left + c) for r, c in mask)
            gate = lookup[color]
            dest = _target(cells, gate)
            if alternate is not None and color == alternate["route"]:
                other = _target(cells, alternate)
                if (not _parallel_fit(other, alternate) or
                        not all(0 <= r < h and 0 <= c < w and region[r, c] for r, c in other) or
                        sum(abs(other[0][k] - cells[0][k]) for k in (0, 1)) >=
                        sum(abs(dest[0][k] - cells[0][k]) for k in (0, 1))):
                    good = False
                    break
            if depth_constraint is not None:
                axis, previous = depth_constraint
                lead = min(p[axis] for p in cells) if gate["normal"][axis] > 0 else max(p[axis] for p in cells)
                if lead == previous:
                    good = False
                    break
            if (not _parallel_fit(dest, gate) or source.intersection(cells) or
                    targets.intersection(dest) or occupied.intersection(dest) or
                    not all(region[p] and int(g[p]) == bg for p in cells) or
                    not all(0 <= r < h and 0 <= c < w and region[r, c] for r, c in dest)):
                good = False
                break
            # No diagonal-only joins between independently sampled same-color pieces.
            if any(0 <= r + dr < h and 0 <= c + dc < w and int(g[r + dr, c + dc]) == color
                   for r, c in cells for dr, dc in N8):
                good = False
                break
            source.update(cells)
            targets.update(dest)
            writes.append((color, cells))
        if good:
            for color, cells in writes:
                for p in cells:
                    g[p] = color
            occupied.update(targets)
            return writes
    return None


def _scene(rng, witness, target_large, count, complex_scene, rotation, reflect, extra_room=False):
    divided = witness or extra_room or rng.random() < .8
    h = int(rng.integers(22 if witness or extra_room else 18 if divided else 14, 31))
    w = int(rng.integers(24 if witness else 22 if divided else 16, 31))
    walls = np.zeros((h, w), bool)
    walls[:2] = walls[-2:] = True
    walls[:, :2] = walls[:, -2:] = True
    if divided:
        room_width = int(rng.integers(8, w - 13))
        cut = room_width + 2
        walls[:, cut:cut + 2] = True
        left = (h // 2, 3)
        right = (h // 2, w - 4)
        sizes = [int(_flood(~walls, p).sum()) for p in (left, right)]
        if witness and abs(sizes[0] - sizes[1]) < 2 * (h - 4):
            return None
        side = int(np.argmax(sizes) if target_large else np.argmin(sizes)) if witness else int(rng.integers(2))
        anchor = (left, right)[side]
        # Tests add a third room. It can split the selected side, making the
        # branch itself part of the docking boundary, or the excluded side.
        if extra_room:
            row = int(rng.integers(h // 2 - 2, h // 2 + 3))
            branch_side = int(rng.integers(2))
            if branch_side == 1:
                walls[row:row + 2, cut + 2:] = True
            else:
                walls[row:row + 2, :cut] = True
            if branch_side == side:
                anchor = (3, anchor[1])
    else:
        anchor = (h // 2, w // 2)
    # Short attached masses create concave boundaries and uneven wall profiles.
    for _ in range(int(rng.integers(1 if complex_scene else 0, 5 if complex_scene else 2))):
        row = int(rng.integers(3, h - 4))
        depth = int(rng.integers(1, 4))
        if rng.random() < .5:
            walls[row:row + int(rng.integers(1, 4)), 2:2 + depth] = True
        else:
            walls[row:row + int(rng.integers(1, 4)), w - 2 - depth:w - 2] = True
    if walls[anchor]:
        return None
    region = _flood(~walls, anchor)
    # A spaced solid island supplies an inert dock with a duplicate route label.
    add_island = witness or rng.random() < .7
    if add_island:
        ih, iw = int(rng.integers(2, 4)), int(rng.integers(6, 9))
        positions = [(r, c) for r in range(4, h - ih - 3) for c in range(4, w - iw - 3)
                     if np.all(region[r - 1:r + ih + 1, c - 1:c + iw + 1])]
        rng.shuffle(positions)
        if not positions:
            return None
        r, c = positions[0]
        walls[r:r + ih, c:c + iw] = True
        if walls[anchor]:
            available = np.argwhere(region & ~walls)
            anchor = tuple(map(int, available[0]))
        region = _flood(~walls, anchor)
    # Optional plain islands coexist with the island dock.
    for _ in range(int(rng.integers(0, 3)) if complex_scene else 0):
        positions = [(r, c) for r in range(4, h - 5) for c in range(4, w - 5)
                     if np.all(region[r - 1:r + 3, c - 1:c + 3])]
        if positions:
            r, c = positions[int(rng.integers(len(positions)))]
            walls[r:r + 2, c:c + 2] = True
            if walls[anchor]:
                anchor = tuple(map(int, np.argwhere(region & ~walls)[0]))
            region = _flood(~walls, anchor)
    exterior = _exterior(region)
    palette = list(map(int, rng.permutation(10)))
    bg, wall, marker = palette[:3]
    routes = palette[3:3 + count]
    # Shared or different endpoint colors are both visibly encoded.
    rail_pool = palette[3 + count:3 + 2 * count]
    rails = rail_pool if rng.random() < .65 else [rail_pool[0]] * count
    g = np.full((h, w), bg, np.int8)
    g[walls] = wall
    gates, used, reserved = [], set(), set()
    for i in range(count):
        length = int(rng.integers(5, 10))
        possibilities = _locations(walls, region, exterior, length, True)
        if i == 0 and witness:
            possibilities = [x for x in possibilities if x["normal"][0]]
        if i == 1 and witness:
            possibilities = [x for x in possibilities if x["normal"][1]]
        rng.shuffle(possibilities)
        gate = None
        for candidate in possibilities:
            dr, dc = candidate["normal"]
            backing = {(r - dr, c - dc) for r, c in candidate["cells"]}
            if (set(candidate["cells"]) & reserved or backing & used or
                    any((r + dr, c + dc) in used for r, c in candidate["cells"] for dr, dc in N8)):
                continue
            gate = candidate
            break
        if gate is None:
            return None
        gate["route"], gate["rail"] = routes[i], rails[i]
        gates.append(gate)
        used.update(gate["cells"])
        reserved.update(gate["cells"])
        reserved.update((r - gate["normal"][0], c - gate["normal"][1]) for r, c in gate["cells"])
    all_gates = gates.copy()
    if add_island:
        possibilities = []
        for length in range(3 if witness else 2, 7):
            possibilities.extend(_locations(walls, region, exterior, length, False))
        if witness:
            possibilities = [x for x in possibilities if x["normal"][0]]
        if not possibilities:
            return None
        decoy = possibilities[int(rng.integers(len(possibilities)))]
        decoy["route"], decoy["rail"] = routes[0], rails[0]
        all_gates.append(decoy)
    # Other rooms can display the same legend; only the seed-facing docks act.
    if divided and rng.random() < .55:
        others = _components(~walls & ~region)
        cells = others[int(rng.integers(len(others)))]
        other_region = np.zeros_like(region)
        for p in cells:
            other_region[p] = True
        possibilities = _locations(walls, other_region, exterior, int(rng.integers(2, 7)), True)
        rng.shuffle(possibilities)
        for candidate in possibilities:
            dr, dc = candidate["normal"]
            backing = {(r - dr, c - dc) for r, c in candidate["cells"]}
            if (set(candidate["cells"]) & reserved or backing & used or
                    any((r + dr, c + dc) in used for r, c in candidate["cells"] for dr, dc in N8)):
                continue
            candidate["route"], candidate["rail"] = routes[0], rails[0]
            all_gates.append(candidate)
            break
    for gate in all_gates:
        for p in gate["cells"][1:-1]:
            g[p] = gate["route"]
        g[gate["cells"][0]] = g[gate["cells"][-1]] = gate["rail"]
    occupied = set()
    if witness == 1:
        fh, fw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        ring = tuple((r, c) for r in range(fh) for c in range(fw)
                     if r in (0, fh - 1) or c in (0, fw - 1))
        inner = tuple((r, c) for r in range(1, fh - 1) for c in range(1, fw - 1))
        if not _place_group(rng, g, region, gates, [(routes[0], ring), (routes[1], inner)], bg,
                            occupied, alternate=decoy):
            return None
    elif witness == 2:
        twin_mask = _shape(rng, 3, 2)
        first = _place_group(rng, g, region, gates, [(routes[0], twin_mask)], bg,
                             occupied, alternate=decoy)
        if not first:
            return None
        axis = 0 if gates[0]["normal"][0] else 1
        cells = first[0][1]
        lead = min(p[axis] for p in cells) if gates[0]["normal"][axis] > 0 else max(p[axis] for p in cells)
        if not _place_group(rng, g, region, gates, [(routes[0], _shape(rng, 3, 2))], bg,
                            occupied, (axis, lead), alternate=decoy):
            return None
        if not _place_group(rng, g, region, gates, [(routes[1], twin_mask)], bg, occupied):
            return None
    # Every exterior dock may receive zero or several pieces, but ensure used routes.
    for route in routes:
        if np.any((g == route) & ~walls):
            continue
        if not _place_group(rng, g, region, gates, [(route, _shape(rng, 4, 4))], bg, occupied):
            if not _place_group(rng, g, region, gates, [(route, ((0, 0),))], bg, occupied):
                return None
    for _ in range(int(rng.integers(1, 7 if complex_scene else 4))):
        route = int(rng.choice(routes))
        _place_group(rng, g, region, gates, [(route, _shape(rng, 6 if complex_scene else 4,
                                                         6 if complex_scene else 4))], bg, occupied)
    # A colored piece in another room demonstrates selection-based erasure.
    unselected = ~walls & ~region
    if divided:
        locations = [tuple(map(int, p)) for p in np.argwhere(unselected)]
        rng.shuffle(locations)
        for p in locations[:int(rng.integers(2, 7))]:
            g[p] = int(rng.choice(routes))
    marker_choices = [tuple(map(int, p)) for p in np.argwhere(region & (g == bg))
                      if tuple(map(int, p)) not in occupied]
    if not marker_choices:
        return None
    p = marker_choices[int(rng.integers(len(marker_choices)))]
    g[p] = marker
    g = np.rot90(g, rotation).copy()
    if reflect:
        g = np.fliplr(g).copy()
    try:
        a = _decode(g)
        # Reject any accidental parsing change or source component merger.
        if len(a["gates"]) != len(all_gates) or len(a["active"]) != count:
            return None
        for route in routes:
            mask = (g == route) & ~a["blocked"]
            if len(_components(mask)) != len(_components(mask, N8)):
                return None
        pair = {"input": g, "output": solve(g)}
        if witness:
            f = _features(pair)
            if not (f["small" if witness == 1 else "large"] and f["inert_fit"] and f["nearer_inert"] and
                    f["contact" if witness == 1 else "independent"] and f["erased"]):
                return None
        return pair
    except AssertionError:
        return None


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value


def _key(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    base_turn = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = index + 1 if split == "train" and index < 2 else 0
            # The two witnesses teach topology/routing. Test recombination shifts
            # either dock count, direction, or boundary complexity.
            dock_count = 2 if witness else (3 if split == "test" and regime == 0
                                           else int(rng.integers(1, 4)))
            complex_scene = split == "test" or (not witness and rng.random() < .6)
            turn = (base_turn + index % 2) % 4 if split == "train" else (
                (base_turn + 2 + index % 2) % 4 if regime == 1 else int(rng.integers(4)))
            pair = None
            for attempt in range(160):
                candidate = _scene(rng, witness, witness == 2, dock_count,
                                   complex_scene, turn, bool(rng.integers(2)), split == "test")
                if candidate is not None and _key(candidate["input"]) not in seen:
                    pair = candidate
                    break
            if pair is None:
                # A bounded simpler resampling retains the requested witnesses.
                for attempt in range(160):
                    candidate = _scene(rng, witness, witness == 2, 2 if witness else 1,
                                       False, turn, bool(rng.integers(2)), split == "test")
                    if candidate is not None and _key(candidate["input"]) not in seen:
                        pair = candidate
                        break
            _require(pair is not None, "bounded scene construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a = _decode(pair["input"])
    selected = [c for c in a["cargo"] if c["selected"]]
    contact = any(pair["input"][r, c] != pair["input"][r + dr, c + dc]
                  and any((r + dr, c + dc) in x["cells"] for x in selected)
                  for comp in selected for r, c in comp["cells"] for dr, dc in N4
                  if 0 <= r + dr < pair["input"].shape[0] and 0 <= c + dc < pair["input"].shape[1])
    independent = False
    for i, (comp, gate, dest) in enumerate(a["placements"]):
        axis = 0 if gate["normal"][0] else 1
        extreme = min if gate["normal"][axis] > 0 else max
        for other, og, od in a["placements"][i + 1:]:
            if comp["color"] == other["color"] and extreme(p[axis] for p in comp["cells"]) != extreme(p[axis] for p in other["cells"]):
                independent = True
    inert = [g for g in a["gates"] if all(a["region"][p] for p in _front(g))
             and not any(a["outside"][p] for p in g["cells"])]
    inert_fit = sum(any(comp["color"] == gate["route"] and
                        _parallel_fit(_target(comp["cells"], gate), gate) and
                        set(_target(comp["cells"], gate)) != set(dest) and
                        all(0 <= r < a["grid"].shape[0] and 0 <= c < a["grid"].shape[1]
                            and a["region"][r, c] and (r, c) != a["start"]
                            for r, c in _target(comp["cells"], gate))
                        for comp, active_gate, dest in a["placements"]) for gate in inert)
    nearer_inert = any(comp["color"] == gate["route"] and
                      _parallel_fit(_target(comp["cells"], gate), gate) and
                      all(0 <= r < a["grid"].shape[0] and 0 <= c < a["grid"].shape[1]
                          and a["region"][r, c] and (r, c) != a["start"]
                          for r, c in _target(comp["cells"], gate)) and
                      sum(abs(_target(comp["cells"], gate)[0][k] - comp["cells"][0][k]) for k in (0, 1)) <
                      sum(abs(dest[0][k] - comp["cells"][0][k]) for k in (0, 1))
                      for gate in inert for comp, active_gate, dest in a["placements"])
    regions = _components(~a["blocked"])
    areas = [len(c) for c in regions]
    selected_area = int(a["region"].sum())
    masks = []
    for comp in selected:
        r0 = min(r for r, c in comp["cells"])
        c0 = min(c for r, c in comp["cells"])
        masks.append(tuple((r - r0, c - c0) for r, c in comp["cells"]))
    return dict(shape=tuple(pair["input"].shape), docks=len(a["active"]),
                directions=tuple(sorted({g["normal"] for g in a["active"]})),
                lengths=tuple(len(g["cells"]) - 2 for g in a["active"]),
                compartments=len(regions), small=selected_area < max(areas),
                large=len(areas) > 1 and selected_area == max(areas),
                islands=len(_components(a["blocked"] & ~a["outside"])),
                inert=len(inert), inert_fit=inert_fit, nearer_inert=nearer_inert,
                erased=sum(not c["selected"] for c in a["cargo"]),
                cargo=len(selected), contact=bool(contact), independent=independent,
                masks=tuple(masks), roles=(a["background"], a["wall"], a["marker"]),
                rails=tuple(sorted({g["rail"] for g in a["active"]})))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split type")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape preservation")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _decode(inp)
            _require(np.array_equal(out, solve(inp)), "wrong output")
            _require(np.array_equal(out[a["blocked"]], inp[a["blocked"]]), "structure changed")
            _require(out[a["start"]] == a["marker"], "seed changed")
            # Count exact translated occupancy per destination; all cargo clears
            # before these masks are laid down, so source/target overlap is legal.
            targets = {p for c, g, cells in a["placements"] for p in cells}
            for comp in a["cargo"]:
                for p in comp["cells"]:
                    if p not in targets:
                        _require(out[p] == a["background"], "uncleared cargo")
            for route in {g["route"] for g in a["gates"]}:
                mask = (inp == route) & ~a["blocked"]
                _require(len(_components(mask)) == len(_components(mask, N8)), "ambiguous diagonal grouping")
            if split == "train":
                training.append(_features(pair))
    for feature in ("contact", "independent", "inert_fit", "nearer_inert", "erased", "small", "large"):
        _require(any(f[feature] for f in training), "missing training witness: " + feature)
    _require(any(len(f["directions"]) >= 2 for f in training), "direction must be taught geometrically")


def _fixture():
    """Hand-constructed regression, with an explicitly written target mask."""
    g = np.zeros((20, 24), np.int8)
    g[:2] = g[-2:] = 5
    g[:, :2] = g[:, -2:] = 5
    g[:, 10:12] = 5
    g[1, 2:10] = [3, 1, 1, 1, 1, 1, 1, 3]
    g[3:11, 1] = [4, 2, 2, 2, 2, 2, 2, 4]
    g[11:13, 3:9] = 5
    g[11, 3:9] = [3, 1, 1, 1, 1, 3]
    g[1, 13:21] = [3, 1, 1, 1, 1, 1, 1, 3]
    sources = ((5, 4), (5, 5), (6, 4), (8, 7), (8, 8), (7, 4), (7, 5), (8, 5), (6, 14), (6, 15))
    for p in sources[:5] + sources[8:]:
        g[p] = 1
    for p in sources[5:8]:
        g[p] = 2
    g[14, 7] = 6
    out = g.copy()
    for p in sources:
        out[p] = 0
    for p in ((2, 4), (2, 5), (3, 4), (2, 7), (2, 8)):
        out[p] = 3
    for p in ((7, 2), (7, 3), (8, 3)):
        out[p] = 4
    return {"input": g, "output": out}


def _semantic_regressions():
    pair = _fixture()
    _require(np.array_equal(solve(pair["input"]), pair["output"]), "hand-labeled fixture")
    a = _decode(pair["input"])
    # Omitting A's independent components: translate all cells of each color
    # with a shared extremum. This leaves one red piece away from its dock.
    grouped = pair["input"].copy()
    for comp in a["cargo"]:
        for p in comp["cells"]:
            grouped[p] = a["background"]
    for gate in a["active"]:
        cells = tuple(p for comp in a["cargo"] if comp["selected"] and comp["color"] == gate["route"]
                      for p in comp["cells"])
        for p in _target(cells, gate):
            grouped[p] = gate["rail"]
    _require(not np.array_equal(grouped, pair["output"]), "A ablation did not change answer")
    # Omitting B's island exclusion gives a geometrically feasible wrong dock.
    wrong_island = pair["output"].copy()
    for p in ((2, 4), (2, 5), (3, 4)):
        wrong_island[p] = 0
    for p in ((9, 4), (9, 5), (10, 4)):
        _require(a["region"][p], "island counterexample must fit")
        wrong_island[p] = 3
    _require(not np.array_equal(wrong_island, pair["output"]), "B island ablation did not change answer")
    # Across-room marker movement selects the opposite exterior dock of color 1.
    across = pair["input"].copy()
    across[14, 7], across[14, 16] = 0, 6
    expected = pair["input"].copy()
    for comp in a["cargo"]:
        for p in comp["cells"]:
            expected[p] = 0
    expected[14, 7], expected[14, 16] = 0, 6
    expected[2, 14:16] = 3
    _require(np.array_equal(solve(across), expected), "marker must select the room and its docks")
    # Recoloring the dock endpoints changes the cargo, with geometry unchanged.
    recolored = pair["input"].copy()
    recolored[1, 2] = recolored[1, 9] = 7
    changed = pair["output"].copy()
    changed[1, 2] = changed[1, 9] = 7
    for p in ((2, 4), (2, 5), (3, 4), (2, 7), (2, 8)):
        changed[p] = 7
    _require(np.array_equal(solve(recolored), changed), "endpoint color must control recoloring")


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in a)


def self_test():
    _semantic_regressions()
    records, signatures = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(x["input"]) for s in p for x in p[s]))
        records.extend(_features(x) for s in p for x in p[s])
        _require(all(_features(x)["compartments"] <= 2 for x in p["train"]) and
                 all(_features(x)["compartments"] == 3 for x in p["test"]), "topology shift")
        for pair in p["test"]:
            before = pair["input"].copy()
            _require(np.array_equal(solve(before), pair["output"]), "input-only solve")
            _require(np.array_equal(before, pair["input"]), "input mutation")
    _require(len(signatures) == 42, "cross-seed repeats")
    _require(len({f["shape"] for f in records}) >= 35, "dimension diversity")
    _require({f["docks"] for f in records} >= {1, 2, 3}, "dock count diversity")
    _require({d for f in records for d in f["directions"]} == set(N4), "direction diversity")
    _require(len({m for f in records for m in f["masks"]}) >= 35, "mask diversity")
    _require({f["compartments"] for f in records} >= {1, 2, 3}, "room topology diversity")
    _require(any(f["islands"] == 0 for f in records) and any(f["islands"] >= 2 for f in records), "island diversity")
    _require(len({f["roles"] for f in records}) >= 35, "palette role diversity")
    p = generate_puzzle(0, 2, 1)
    _require(_same(p, generate_puzzle(0, 2, 1)), "nondeterminism")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            q = generate_puzzle(900 + nt * 5 + nq, nt, nq)
            _require(len(q["train"]) == nt and len(q["test"]) == nq, "count not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    # Whole-scene rotations/reflections preserve the rule in all eight poses.
    for pair in p["train"] + p["test"]:
        for k in range(4):
            for flip in (False, True):
                inp, out = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
                if flip:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                _require(np.array_equal(solve(inp), out), "dihedral equivariance")
    # Changing the seed within the selected room cannot change the cargo stamps.
    pair = p["train"][0]
    a = _decode(pair["input"])
    targets = {p for c, g, cells in a["placements"] for p in cells}
    candidates = [tuple(map(int, x)) for x in np.argwhere(a["region"] & (pair["input"] == a["background"]))
                  if tuple(map(int, x)) not in targets]
    moved = pair["input"].copy()
    moved[a["start"]] = a["background"]
    moved[candidates[0]] = a["marker"]
    expected = pair["output"].copy()
    expected[a["start"]] = a["background"]
    expected[candidates[0]] = a["marker"]
    _require(np.array_equal(solve(moved), expected), "seed movement within room")
    broken = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all legal counts, and semantic regressions")
