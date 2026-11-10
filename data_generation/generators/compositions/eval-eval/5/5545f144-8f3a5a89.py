"""Floodgate courier: rigid motion changes the region that exposes its route.

Only NumPy and the standard library are used. Generation and self-tests perform
no disk I/O. The solver reads the query grid alone and keeps no global state.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with axes 1..30")
    return a.astype(np.int8, copy=True)


def _flood(allowed, start):
    reached = np.zeros_like(allowed, dtype=bool)
    if not allowed[start]:
        raise ValueError("flood seed is blocked")
    reached[start] = True
    stack = [start]
    h, w = allowed.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not reached[nr, nc]:
                reached[nr, nc] = True
                stack.append((nr, nc))
    return reached


def _outer(region):
    return _flood(np.pad(~region, 1, constant_values=True), (0, 0))


def _dilate(mask, diagonal=False):
    h, w = mask.shape
    p = np.pad(mask, 1)
    out = np.zeros_like(mask, dtype=bool)
    for dr, dc in N8 if diagonal else N4:
        out |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _rim(region, diagonal=True):
    ext = _outer(region)
    return region & _dilate(ext, diagonal)[1:-1, 1:-1]


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _pose(mask, tip, facing):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate(p, facing - 1) for p in mask)}


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    if (dr == 0) == (dc == 0):
        return None
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _component(mask, start):
    h, w = mask.shape
    found, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in N8:
            q = r + dr, c + dc
            if 0 <= q[0] < h and 0 <= q[1] < w and mask[q] and q not in found:
                found.add(q)
                stack.append(q)
    return found


def _read_input(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 4 or 0 not in colors:
        raise ValueError("floor and exactly three nonzero roles are required")
    frequency = {int(c): int(n) for c, n in zip(colors, counts) if c}
    noses = [c for c, n in frequency.items() if n == 1]
    if len(noses) != 1:
        raise ValueError("nose must be the unique singleton color")
    nose = noses[0]
    st = max(frequency, key=frequency.get)
    if list(frequency.values()).count(frequency[st]) != 1 or st == nose:
        raise ValueError("structure must be the most frequent nonzero color")
    ink = next(c for c in frequency if c not in (st, nose))
    tip = tuple(int(x) for x in np.argwhere(grid == nose)[0])
    mover = _component((grid == ink) | (grid == nose), tip)
    matches = []
    for facing in range(4):
        offsets = {_rotate((r - tip[0], c - tip[1]), 1 - facing) for r, c in mover}
        if (len(offsets) >= 3 and all(c <= 0 for r, c in offsets)
                and {p for p in offsets if p[1] == 0} == {(0, 0)}
                and max(abs(r) for r, c in offsets) > 0
                and {(-r, c) for r, c in offsets} == offsets):
            matches.append((tuple(sorted(offsets)), facing))
    if len(matches) != 1:
        raise ValueError("key needs a unique symmetric pointing axis")
    mask, facing = matches[0]
    dots = {tuple(int(x) for x in q) for q in np.argwhere(grid == ink)} - mover
    h, w = grid.shape
    wall = grid == st
    gates, floor_dots = set(), set()
    for q in sorted(dots):
        r, c = q
        if any((r + dr, c + dc) in dots or (r + dr, c + dc) in mover for dr, dc in N8):
            raise ValueError("checkpoints must be separated singletons")
        neighbors = [0 <= r + dr < h and 0 <= c + dc < w and wall[r + dr, c + dc]
                     for dr, dc in N4]
        pairs = int(neighbors[0] and neighbors[2]) + int(neighbors[1] and neighbors[3])
        if pairs == 1 and sum(neighbors) == 2:
            gates.add(q)
        elif not any(neighbors):
            floor_dots.add(q)
        else:
            raise ValueError("checkpoint is neither a thin door nor clear floor")
    for q in gates:
        wall[q] = True
    if not gates:
        raise ValueError("at least one door is required")
    if any(wall[q] for q in mover):
        raise ValueError("initial key overlaps a wall")
    return dict(st=st, ink=ink, nose=nose, tip=tip, mask=mask, facing=facing,
                gates=gates, floor_dots=floor_dots, wall=wall, mover=mover)


def _finish(wall, selected, mask, tip, facing, ink, nose, st):
    out = np.where(wall, st, 0).astype(np.int8)
    out[_rim(selected)] = nose
    cells = _pose(mask, tip, facing)
    for q in cells:
        out[q] = ink
    out[tip] = nose
    return out


def _analyze(input_grid):
    grid = _grid(input_grid)
    a = _read_input(grid)
    wall, tip, facing = a["wall"].copy(), a["tip"], a["facing"]
    remaining_gates, remaining_floor = set(a["gates"]), set(a["floor_dots"])
    route, directions, opened, growth = [tip], [], set(), []
    initial_region = _flood(~wall, tip)
    first_active = None
    # A checkpoint is consumed each iteration, so this bound is exact.
    for _ in range(len(remaining_gates) + len(remaining_floor) + 1):
        selected = _flood(~wall, tip)
        exterior = _outer(selected)[1:-1, 1:-1]
        adjacent = _dilate(selected)
        active = {q for q in remaining_floor if selected[q]}
        active |= {q for q in remaining_gates if adjacent[q] and exterior[q]}
        if first_active is None:
            first_active = set(active)
        if not active:
            break
        rays = {}
        for q in sorted(active):
            d = _direction(tip, q)
            if d is not None:
                rays.setdefault(d, []).append(q)
        if len(rays) != 1:
            raise ValueError("active checkpoints must occupy exactly one ray")
        d, choices = next(iter(rays.items()))
        if not directions and d != facing:
            raise ValueError("first move must follow the visible nose")
        destination = min(choices, key=lambda q: _distance(tip, q))
        before_area = int(selected.sum())
        if destination in remaining_gates:
            remaining_gates.remove(destination)
            opened.add(destination)
            wall[destination] = False
        else:
            remaining_floor.remove(destination)
        pose = _pose(a["mask"], destination, d)
        if any(not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]) or wall[r, c]
               for r, c in pose):
            raise ValueError("arrival key must fit the open map")
        # Future checkpoints must stay visibly distinguishable from the key.
        if any(max(abs(q[0] - r), abs(q[1] - c)) <= 1
               for q in remaining_gates | remaining_floor for r, c in pose):
            raise ValueError("arrival key touches a remaining checkpoint")
        route.append(destination)
        directions.append(d)
        tip, facing = destination, d
        after_area = int(_flood(~wall, tip).sum())
        if destination in opened:
            growth.append(after_area - before_area)
    else:
        raise ValueError("nonterminating route")
    out = _finish(wall, selected, a["mask"], tip, facing, a["ink"], a["nose"], a["st"])
    closer_island = any(_direction(a["tip"], q) == directions[0]
                        and _distance(a["tip"], q) < _distance(a["tip"], route[1])
                        and q not in first_active
                        for q in a["gates"]) if directions else False
    a.update(route=route, directions=directions, opened=opened, growth=growth,
             selected=selected, initial_region=initial_region, final_wall=wall,
             remaining_gates=remaining_gates, remaining_floor=remaining_floor,
             closer_island=closer_island)
    return out, a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive door activation, travel, final contour, and key from one input."""
    return _analyze(input_grid)[0]


def _mask(extension, solid, tails):
    cells = {(0, -i) for i in range(extension)}
    cells |= {(-1, -extension), (1, -extension)}
    if solid:
        cells.add((0, -extension))
    if tails:
        cells |= {(-1, -extension - 1), (1, -extension - 1)}
    return tuple(sorted(cells))


MASKS = tuple(_mask(e, solid, tails)
              for e in (1, 2) for solid in (False, True) for tails in (False, True))


def _bounds(lengths):
    result, start = [], 0
    for n in lengths:
        result.append((start, start + n))
        start += n + 1
    return result


def _itinerary(rng, rows, cols, doors):
    paths = []
    # At most six rooms: exhaustive simple walks are small and bounded.
    def extend(path):
        if len(path) == doors + 1:
            paths.append(tuple(path))
            return
        r, c = path[-1]
        for dr, dc in N4:
            q = r + dr, c + dc
            if 0 <= q[0] < rows and 0 <= q[1] < cols and q not in path:
                extend(path + [q])
    for r in range(rows):
        for c in range(cols):
            extend([(r, c)])
    return paths[int(rng.integers(len(paths)))]


def _door(rng, first, second, rb, cb):
    r, c = first
    nr, nc = second
    if r == nr:
        y = int(rng.integers(rb[r][0] + 2, rb[r][1] - 2))
        return y, cb[min(c, nc)][1]
    x = int(rng.integers(cb[c][0] + 2, cb[c][1] - 2))
    return rb[min(r, nr)][1], x


def _room_points(rng, entry, exit_, room):
    """Route inside one rectangle, with zero, one, or two interior bends."""
    r0, r1, c0, c1 = room
    ev = entry[1] in (c0 - 1, c1)
    xv = exit_[1] in (c0 - 1, c1)
    if ev != xv:
        elbow = (entry[0], exit_[1]) if ev else (exit_[0], entry[1])
        return [elbow, exit_]
    if (ev and entry[0] == exit_[0]) or (not ev and entry[1] == exit_[1]):
        return [exit_]
    if (ev and abs(entry[0] - exit_[0]) < 3) or (not ev and abs(entry[1] - exit_[1]) < 3):
        raise ValueError("opposite doors have cramped offset")
    if ev:
        middle = int(rng.integers(c0 + 2, c1 - 2))
        return [(entry[0], middle), (exit_[0], middle), exit_]
    middle = int(rng.integers(r0 + 2, r1 - 2))
    return [(middle, entry[1]), (middle, exit_[1]), exit_]


def _protected(shape, route, mask, ds):
    occupied = np.zeros(shape, dtype=bool)
    for i, tip in enumerate(route):
        for r, c in _pose(mask, tip, ds[max(0, i - 1)]):
            if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                raise ValueError("pose exceeds canvas")
            occupied[r, c] = True
    # Keep added geometry away from checkpoints and the initially visible key.
    dots = np.zeros(shape, dtype=bool)
    for q in route:
        dots[q] = True
    return occupied | _dilate(dots, True) | dots


def _candidate(rng, topology, doors, mask, palette, require_decoy, require_turn):
    rows, cols = topology
    def lengths(count):
        lo, hi = ((12, 23) if count == 1 else (9, 14) if count == 2 else (8, 10))
        return [int(x) for x in rng.integers(lo, hi, count)]
    rb, cb = _bounds(lengths(rows)), _bounds(lengths(cols))
    h, w = rb[-1][1], cb[-1][1]
    role = np.zeros((h, w), dtype=np.int8)
    for _, end in rb[:-1]:
        role[end, :] = 1
    for _, end in cb[:-1]:
        role[:, end] = 1
    rooms = _itinerary(rng, rows, cols, doors)
    gates = [_door(rng, a, b, rb, cb) for a, b in zip(rooms, rooms[1:])]
    def rectangle(room):
        r, c = room
        return (*rb[r], *cb[c])
    r0, r1, c0, c1 = rectangle(rooms[0])
    first = gates[0]
    if first[1] == c1:
        start = (first[0], c0 + 3)
    elif first[1] == c0 - 1:
        start = (first[0], c1 - 4)
    elif first[0] == r1:
        start = (r0 + 3, first[1])
    else:
        start = (r1 - 4, first[1])
    route = [start, first]
    for i in range(1, len(gates)):
        route.extend(_room_points(rng, gates[i - 1], gates[i], rectangle(rooms[i])))
    # Finish inside the last chamber, with a turn whenever room geometry permits.
    r0, r1, c0, c1 = rectangle(rooms[-1])
    entry = gates[-1]
    if entry[1] in (c0 - 1, c1):
        col = int(rng.integers(c0 + 2, c1 - 2))
        route.append((entry[0], col))
        choices = [r for r in range(r0 + 2, r1 - 2) if abs(r - entry[0]) >= 3]
        if choices and (require_turn or rng.random() < .75):
            route.append((int(rng.choice(choices)), col))
    else:
        row = int(rng.integers(r0 + 2, r1 - 2))
        route.append((row, entry[1]))
        choices = [c for c in range(c0 + 2, c1 - 2) if abs(c - entry[1]) >= 3]
        if choices and (require_turn or rng.random() < .75):
            route.append((row, int(rng.choice(choices))))
    # Optional nearest-first subdivisions, leaving the first witness ray intact.
    expanded = [route[0]]
    for i, (a, b) in enumerate(zip(route, route[1:])):
        d = _direction(a, b)
        distance = _distance(a, b)
        if d is None or distance < 3:
            raise ValueError("cramped route")
        if i and distance >= 8 and rng.random() < .40:
            length = int(rng.integers(3, distance - 2))
            dr, dc = N4[d]
            expanded.append((a[0] + dr * length, a[1] + dc * length))
        expanded.append(b)
    route = expanded
    ds = [_direction(a, b) for a, b in zip(route, route[1:])]
    if require_turn and ds[0] == ds[-1]:
        raise ValueError("first witness must demonstrate rotation")
    protected = _protected(role.shape, route, mask, ds)
    for q in gates:
        role[q] = 2
    # An island door can lie closer than the active first door on the same ray.
    decoys = []
    dr, dc = N4[ds[0]]
    for distance in range(3, _distance(start, first) - 2):
        q = start[0] + dr * distance, start[1] + dc * distance
        cells = {q, (q[0] + dc, q[1] - dr), (q[0] - dc, q[1] + dr)}
        if all(1 <= r < h - 1 and 1 <= c < w - 1 and not role[r, c]
               and not protected[r, c] for r, c in cells):
            decoys.append((q, cells))
    if decoys and (require_decoy or rng.random() < .70):
        q, cells = decoys[int(rng.integers(len(decoys)))]
        for cell in cells:
            role[cell] = 1
        role[q] = 2
    elif require_decoy:
        raise ValueError("no room for the island witness")
    # Small, solid islands and wall lobes vary the contour independently.
    for _ in range(int(rng.integers(0, 7))):
        sh, sw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        r, c = int(rng.integers(h - sh + 1)), int(rng.integers(w - sw + 1))
        cells = np.zeros(role.shape, bool)
        cells[r:r + sh, c:c + sw] = True
        if np.any(protected & cells) or np.any((role == 2) & (_dilate(cells, True) | cells)):
            continue
        # Do not create ambiguous thin-door classifications near ordinary dots.
        role[cells] = 1
    for q in set(route[1:]) - set(gates):
        if role[q] != 0:
            raise ValueError("floor checkpoint overlaps structure")
        role[q] = 2
    for q in _pose(mask, start, ds[0]):
        if role[q] != 0:
            raise ValueError("initial key overlaps another role")
        role[q] = 2
    role[start] = 3
    # Construct labels using the sampled route, rather than asking solve for them.
    final_wall = role == 1
    for q in set(zip(*np.where(role == 2))) - set(route[1:]) - _pose(mask, start, ds[0]):
        final_wall[q] = True  # island door, still sealed
    selected = _flood(~final_wall, route[-1])
    expected = _finish(final_wall, selected, mask, route[-1], ds[-1], 2, 3, 1)
    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
    role, expected = np.rot90(role, rotation), np.rot90(expected, rotation)
    if reflection:
        role, expected = np.fliplr(role), np.fliplr(expected)
    mapping = np.array((0, *palette), dtype=np.int8)
    pair = {"input": mapping[role], "output": mapping[expected]}
    predicted, info = _analyze(pair["input"])
    if not np.array_equal(predicted, pair["output"]) or len(info["opened"]) != doors:
        raise ValueError("constructed itinerary not recovered from input")
    if require_decoy and not info["closer_island"]:
        raise ValueError("missing inactive closer door")
    if any(g <= 1 for g in info["growth"]):
        raise ValueError("each route door must expose a new chamber")
    return pair


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shared = bool(rng.random() < .25)
    palette = tuple(int(x) for x in rng.choice(np.arange(1, 10), 3, replace=False))
    first_mask = int(rng.integers(len(MASKS)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = split == "train" and index == 0
            if split == "test":
                topology = (2, 3) if rng.random() < .5 else (3, 2)
                doors = int(rng.integers(4, 6))
            elif witness:
                topology = (1, 2) if rng.random() < .5 else (2, 1)
                doors = 1
            elif index == 1:
                topology, doors = (2, 2), int(rng.integers(2, 4))
            else:
                topologies = ((1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
                topology = topologies[int(rng.integers(len(topologies)))]
                doors = int(rng.integers(1, min(3, topology[0] * topology[1] - 1) + 1))
            mask_index = first_mask if witness else int(rng.integers(len(MASKS)))
            if split == "train" and index == 1 and mask_index == first_mask:
                mask_index = (mask_index + int(rng.integers(1, len(MASKS)))) % len(MASKS)
            if not shared:
                palette = tuple(int(x) for x in rng.choice(np.arange(1, 10), 3, replace=False))
            for duplicate_retry in range(12):
                for attempt in range(96):
                    try:
                        pair = _candidate(rng, topology, doors, MASKS[mask_index], palette, witness, witness)
                        break
                    except ValueError:
                        continue
                else:
                    raise RuntimeError("bounded map construction exhausted")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            prediction, a = _analyze(inp)
            assert np.array_equal(out, prediction), "incorrect map or rigid terminal key"
            assert a["directions"] and not a["remaining_floor"]
            assert a["growth"] and all(g > 1 for g in a["growth"])
            assert int(a["selected"].sum()) > int(a["initial_region"].sum())
            assert len(a["opened"]) == len(a["growth"])
            assert int(np.count_nonzero(out == a["ink"])) == len(a["mask"]) - 1
            assert all(out[q] == a["st"] for q in a["remaining_gates"])
            assert not np.array_equal(inp, out)
            if split == "train":
                training.append(a)
    assert len({a["mask"] for a in training}) >= 2, "demonstrate actual-shape preservation"
    assert len({len(a["opened"]) for a in training}) >= 2, "demonstrate repeated growth"
    assert any(a["closer_island"] and a["facing"] != a["directions"][-1] for a in training)
    assert any(np.any(_rim(a["selected"]) & ~_rim(a["selected"], False)) for a in training)


def measure_pair(pair):
    _, a = _analyze(pair["input"])
    return {"shape": list(np.asarray(pair["input"]).shape), "doors": len(a["opened"]),
            "waypoints": len(a["directions"]), "area": len(a["mask"]),
            "mask": a["mask"], "initial": a["facing"], "final": a["directions"][-1],
            "turns": sum(x != y for x, y in zip(a["directions"], a["directions"][1:])),
            "growth": a["growth"], "closer_island": a["closer_island"],
            "sealed_doors": len(a["remaining_gates"]),
            "palette": (a["st"], a["ink"], a["nose"])}


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed semantic and structural tests; no files or saved traces."""
    shapes, masks, palettes, signatures = set(), set(), set(), set()
    directions, opened_counts, route_counts, areas, turns = set(), set(), set(), set(), set()
    decoys, rotations = 0, 0
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(_puzzle_digest(p))
        for split in ("train", "test"):
            for pair in p[split]:
                _, a = _analyze(pair["input"])
                m = measure_pair(pair)
                shapes.add(tuple(m["shape"]))
                masks.add(a["mask"])
                palettes.add(m["palette"])
                directions.add((m["initial"], m["final"]))
                opened_counts.add(m["doors"])
                route_counts.add(m["waypoints"])
                areas.add(m["area"])
                turns.add(m["turns"])
                decoys += int(m["closer_island"])
                rotations += int(m["initial"] != m["final"])
        assert len({_key(pair["input"]) for pair in p["train"] + p["test"]}) == 5
    assert len(signatures) == 42 and len(shapes) >= 50 and len(masks) == len(MASKS)
    assert len(palettes) >= 40 and opened_counts == {1, 2, 3, 4, 5}
    assert all({p[i] for p in palettes} == set(range(1, 10)) for i in range(3))
    assert {a for a, b in directions} == {0, 1, 2, 3}
    assert {b for a, b in directions} == {0, 1, 2, 3}
    assert len(route_counts) >= 8 and len(areas) >= 4 and len(turns) >= 5
    assert decoys >= 42 and rotations >= 42
    # Whole-puzzle determinism, all explicit counts, and type/range rejections.
    assert _puzzle_digest(generate_puzzle(27)) == _puzzle_digest(generate_puzzle(27))
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + nt * 5 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(0, 2, 1)
    attachment_checked = replacement_checked = island_checked = False
    for pair in p["train"]:
        inp, out = pair["input"], pair["output"]
        before = inp.copy()
        assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
        for k in range(4):
            for reflect in (False, True):
                x, y = np.rot90(inp, k), np.rot90(out, k)
                if reflect:
                    x, y = np.fliplr(x), np.fliplr(y)
                assert np.array_equal(solve(x), y), "rotation/reflection equivariance"
        mapping = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], np.int8)
        assert np.array_equal(solve(mapping[inp]), mapping[out]), "visible color roles"
        _, a = _analyze(inp)
        # Inserting an isolated solid island must not add any rim around it.
        forbidden = np.zeros(inp.shape, bool)
        for i, tip in enumerate(a["route"]):
            for q in _pose(a["mask"], tip, a["directions"][max(0, i - 1)]):
                forbidden[q] = True
        clear = (a["initial_region"] & (inp == 0)
                 & ~(_dilate(forbidden, True) | forbidden)
                 & ~_rim(a["selected"]))
        for q in map(tuple, np.argwhere(clear)):
            r, c = q
            if not (1 <= r < inp.shape[0] - 1 and 1 <= c < inp.shape[1] - 1):
                continue
            if not np.all(a["initial_region"][r - 1:r + 2, c - 1:c + 2]):
                continue
            changed, expected = inp.copy(), out.copy()
            changed[q] = expected[q] = a["st"]
            try:
                predicted, b = _analyze(changed)
            except ValueError:
                continue
            assert b["route"] == a["route"] and np.array_equal(predicted, expected)
            island_checked = True
            break
        # A visible, previously unsampled mask is still moved by its actual cells.
        novel_mask = tuple(sorted({(0, 0), (-1, -1), (0, -1), (1, -1), (-2, -2), (2, -2)}))
        for mask in (novel_mask,) + MASKS:
            if mask == a["mask"]:
                continue
            changed = inp.copy()
            for q in a["mover"]:
                changed[q] = 0
            cells = _pose(mask, a["tip"], a["facing"])
            if any(not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1]) or changed[r, c] != 0
                   for r, c in cells):
                continue
            for q in cells:
                changed[q] = a["ink"]
            changed[a["tip"]] = a["nose"]
            try:
                predicted, b = _analyze(changed)
            except ValueError:
                continue
            expected = _finish(a["final_wall"], a["selected"], mask, a["route"][-1],
                               a["directions"][-1], a["ink"], a["nose"], a["st"])
            assert b["mask"] == mask and b["route"] == a["route"]
            assert np.array_equal(predicted, expected) and not np.array_equal(expected, out)
            replacement_checked = True
            break
        if a["closer_island"]:
            # Without B, a closer island checkpoint would be visited/opened.
            q = min((q for q in a["gates"] if _direction(a["tip"], q) == a["directions"][0]),
                    key=lambda q: _distance(a["tip"], q))
            assert q not in a["opened"] and out[q] == a["st"]
            assert _distance(a["tip"], q) < _distance(a["tip"], a["route"][1])
            # Connect the island to the exterior: the SAME marked door now opens.
            d = a["directions"][0]
            for sign in (-1, 1):
                if d in (1, 3):
                    rr = range(0, q[0]) if sign < 0 else range(q[0] + 1, inp.shape[0])
                    connector = [(r, q[1]) for r in rr]
                else:
                    cc = range(0, q[1]) if sign < 0 else range(q[1] + 1, inp.shape[1])
                    connector = [(q[0], c) for c in cc]
                if any(inp[cell] not in (0, a["st"]) for cell in connector):
                    continue
                changed = inp.copy()
                for cell in connector:
                    changed[cell] = a["st"]
                try:
                    predicted, b = _analyze(changed)
                except ValueError:
                    continue
                assert b["route"][1] == q and q in b["opened"]
                assert predicted[q] != a["st"] and not np.array_equal(predicted, out)
                attachment_checked = True
                break
            # Without A's rotation, the final ink mask is different.
            if a["facing"] != a["directions"][-1]:
                assert _pose(a["mask"], a["route"][-1], a["facing"]) != _pose(
                    a["mask"], a["route"][-1], a["directions"][-1])
    assert attachment_checked and replacement_checked and island_checked
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    corrupted["test"][0]["output"][0, 0] = (int(corrupted["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "masks": len(masks), "palettes": len(palettes), "island_witnesses": decoys,
            "rotated_finishes": rotations, "door_counts": sorted(opened_counts),
            "waypoint_counts": sorted(route_counts), "turn_counts": sorted(turns),
            "island_insertion": island_checked, "exterior_attachment": attachment_checked,
            "key_replacement": replacement_checked}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
