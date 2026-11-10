"""Counter-linked lanes with opposite cyclic transfers of bar attributes.

Only standard-library modules and NumPy are used. No files, seeds, or training
examples are consulted by solve. The sampled labels are drawn from construction
attributes; validation re-extracts those attributes from the visible input.
"""
from collections import Counter
import heapq
import json

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRECTIONS = ("up", "down", "left", "right")


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "Invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "Cells must be integers")
    _require(np.all((a >= 0) & (a <= 9)), "Invalid colors")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _neighbors(cell, shape):
    r, c = cell
    h, w = shape
    for dr, dc in STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask):
    remaining = set(zip(*np.where(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        for cell in queue:
            for other in _neighbors(cell, mask.shape):
                if other in remaining:
                    remaining.remove(other)
                    queue.append(other)
        result.append(queue)
    return result


def _dilate(mask):
    expanded = mask.copy()
    expanded[1:] |= mask[:-1]
    expanded[:-1] |= mask[1:]
    expanded[:, 1:] |= mask[:, :-1]
    expanded[:, :-1] |= mask[:, 1:]
    return expanded


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    for cell in cells:
        result[cell] = True
    return result


def _counter(grid):
    candidates = []
    for color, count in zip(*np.unique(grid, return_counts=True)):
        if count != 8:
            continue
        rr, cc = np.where(grid == color)
        r, c = int(rr.min()), int(cc.min())
        if int(rr.max()) != r + 2 or int(cc.max()) != c + 2:
            continue
        patch = grid[r:r+3, c:c+3]
        marker = int(patch[1, 1])
        ring = np.ones((3, 3), dtype=bool)
        ring[1, 1] = False
        if np.all(patch[ring] == color) and marker != color and np.count_nonzero(grid == marker) == 1:
            candidates.append((r, c, int(color), marker))
    _require(len(candidates) == 1, "Need one eight-cell ring with a unique center marker")
    return candidates[0]


def _bars(grid, cells):
    """Recover ordered fixed roots and independent attributes in one lane."""
    parts = _components(_mask(grid.shape, cells))
    _require(3 <= len(parts) <= 6, "Each payload array needs 3..6 bars")
    bounds = []
    for part in parts:
        rr, cc = zip(*part)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        _require(r0 == r1 or c0 == c1, "Bars must be straight and one cell wide")
        _require(len(part) == max(r1-r0, c1-c0) + 1, "Bars must be contiguous")
        colors = {int(grid[p]) for p in part}
        _require(len(colors) == 1, "A bar must be monochromatic")
        bounds.append((r0, r1, c0, c1, next(iter(colors)), len(part)))
    vertical = any(b[1] > b[0] for b in bounds)
    horizontal = any(b[3] > b[2] for b in bounds)
    _require(not (vertical and horizontal), "Bars in an array must be parallel")
    if not vertical and not horizontal:
        vertical = len({b[0] for b in bounds}) == 1
    if vertical:
        _require(all(b[2] == b[3] for b in bounds), "Invalid vertical array")
        bounds.sort(key=lambda b: b[2])
        if len({b[1] for b in bounds}) == 1:
            direction, base = "up", bounds[0][1]
        else:
            _require(len({b[0] for b in bounds}) == 1, "Missing common baseline")
            direction, base = "down", bounds[0][0]
        slots = [int(b[2]) for b in bounds]
        roots = [(int(base), s) for s in slots]
    else:
        _require(all(b[0] == b[1] for b in bounds), "Invalid horizontal array")
        bounds.sort(key=lambda b: b[0])
        if len({b[3] for b in bounds}) == 1:
            direction, base = "left", bounds[0][3]
        else:
            _require(len({b[2] for b in bounds}) == 1, "Missing common baseline")
            direction, base = "right", bounds[0][2]
        slots = [int(b[0]) for b in bounds]
        roots = [(s, int(base)) for s in slots]
    spacings = np.diff(slots)
    _require(len(set(int(s) for s in spacings)) == 1 and spacings[0] >= 2,
             "Slots must be regularly spaced and separated")
    return dict(direction=direction, roots=roots, gap=int(spacings[0])-1,
                colors=[int(b[4]) for b in bounds], lengths=[int(b[5]) for b in bounds],
                cells=[p for part in parts for p in part])


def _scene(grid):
    a = _grid(grid)
    wall = int(a[0, 0])
    _require(all(np.all(edge == wall) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
             "A uniform wall frame must enclose the device")
    r, c, slot, marker = _counter(a)
    _require(wall not in (slot, marker), "Structural colors must differ")
    candidates = [(int(np.count_nonzero(a == x)), int(x)) for x in np.unique(a)
                  if x not in (wall, slot, marker)]
    _require(candidates, "Missing floor")
    candidates.sort(reverse=True)
    _require(len(candidates) == 1 or candidates[0][0] > candidates[1][0],
             "Floor must be the unique most frequent non-wall, non-counter color")
    bg = candidates[0][1]
    payload = ~np.isin(a, (wall, bg, slot, marker))
    collector = _mask(a.shape, [(r+i, c+j) for i in range(3) for j in range(3)])
    distance = np.full(a.shape, -1, dtype=int)
    queue = list(zip(*np.where(collector)))
    distance[collector] = 0
    for cell in queue:
        for other in _neighbors(cell, a.shape):
            if a[other] != wall and distance[other] < 0:
                distance[other] = distance[cell] + 1
                queue.append(other)
    active = payload & (distance >= 0)
    bars = _bars(a, list(zip(*np.where(active))))
    _require(sum(bars["lengths"]) <= 9, "Selected payload exceeds counter capacity")
    decoys = []
    for region in _components((a != wall) & (distance < 0)):
        cargo = [p for p in region if payload[p]]
        _require(cargo, "Every disconnected floor region must contain an array")
        decoys.append(_bars(a, cargo))
    return dict(wall=wall, background=bg, slot=slot, marker=marker, counter=(r, c),
                collector=collector, region=distance >= 0, distance=distance,
                payload=payload, bars=bars, decoys=decoys)


def _bar_cells(root, length, direction):
    dr, dc = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
    return [(root[0] + j*dr, root[1] + j*dc) for j in range(int(length))]


def _transform(a, scene, color_shift=1, length_shift=-1, fill=True):
    result = a.copy()
    b = scene["bars"]
    for cell in b["cells"]:
        result[cell] = scene["background"]
    colors = np.roll(b["colors"], color_shift)
    lengths = np.roll(b["lengths"], length_shift)
    stream = []
    for root, color, length in zip(b["roots"], colors, lengths):
        for cell in _bar_cells(root, length, b["direction"]):
            _require(0 <= cell[0] < a.shape[0] and 0 <= cell[1] < a.shape[1]
                     and a[cell] != scene["wall"], "Rebuilt bar must fit the lane")
            result[cell] = int(color)
        stream.extend([int(color)] * int(length))
    if fill:
        r, c = scene["counter"]
        for i, color in enumerate(stream):
            result[r+i//3, c+i%3] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Select by wall reachability; cycle attributes; serialize new bar cells."""
    a = _grid(input_grid)
    return _transform(a, _scene(a))


def _observable(b):
    c, l = np.array(b["colors"]), np.array(b["lengths"])
    return (not np.array_equal(np.roll(c, 1), np.roll(c, -1))
            and not np.array_equal(np.roll(l, -1), np.roll(l, 1))
            and not np.array_equal(c, np.roll(c, 1))
            and not np.array_equal(l, np.roll(l, -1)))


def _lengths(rng, n, total=None, informative=False):
    for _ in range(64):
        if total is None and rng.random() < .18:
            values = [int(rng.integers(1, 9//n + 1))] * n
        else:
            amount = int(rng.integers(n, 10)) if total is None else total
            values = np.ones(n, dtype=int)
            for _ in range(amount-n):
                values[int(rng.integers(n))] += 1
            values = values.tolist()
        if not informative or not np.array_equal(np.roll(values, 1), np.roll(values, -1)):
            return values
    raise RuntimeError("Cannot sample informative lengths")


def _room(rng, n, direction, palette, total=None, anchor=False, decoy=False):
    lengths = ([int(x) for x in rng.integers(1, 5, n)] if decoy
               else _lengths(rng, n, total, anchor))
    colors = [int(x) for x in rng.choice(palette, size=n, replace=False)]
    if not anchor and rng.random() < .4:
        pool = rng.choice(palette, size=int(rng.integers(1, min(n, len(palette))+1)), replace=False)
        colors = [int(x) for x in rng.choice(pool, size=n)]
    gap = 1 if decoy else int(rng.integers(1, 3))
    before, after = (int(rng.integers(1, 4)) for _ in range(2))
    span = max(lengths) + int(rng.integers(2, 5))
    cross = n + (n-1)*gap + before + after
    h, w = (span, cross) if direction in ("up", "down") else (cross, span)
    return dict(h=h, w=w, colors=colors, lengths=lengths, direction=direction,
                gap=gap, before=before)


def _positions(shape, room, occupied):
    h, w = shape
    rh, rw = room["h"], room["w"]
    return [(r, c) for r in range(1, h-rh) for c in range(1, w-rw)
            if not occupied[r-1:r+rh+1, c-1:c+rw+1].any()]


def _place(floor, room, position):
    room = dict(room, r=position[0], c=position[1])
    floor[room["r"]:room["r"]+room["h"], room["c"]:room["c"]+room["w"]] = True
    return room


def _room_roots(room):
    r, c, h, w = (room[x] for x in ("r", "c", "h", "w"))
    slots = [room["before"] + i*(room["gap"]+1) for i in range(len(room["colors"]))]
    direction = room["direction"]
    if direction == "up": return [(r+h-1, c+s) for s in slots]
    if direction == "down": return [(r, c+s) for s in slots]
    if direction == "left": return [(r+s, c+w-1) for s in slots]
    return [(r+s, c) for s in slots]


def _paint_room(grid, room, shifted=False):
    # Construction labels use the sampled room attributes, not solve's parser.
    n = len(room["colors"])
    for i, root in enumerate(_room_roots(room)):
        color = room["colors"][(i-1) % n] if shifted else room["colors"][i]
        length = room["lengths"][(i+1) % n] if shifted else room["lengths"][i]
        for cell in _bar_cells(root, length, room["direction"]):
            grid[cell] = color


def _rectangle_distance(position, room, counter):
    r, c = position
    cr, cc = counter
    return max(cr-r-room["h"]+1, r-cr-2, 0) + max(cc-c-room["w"]+1, c-cc-2, 0)


def _route(rng, floor, counter, room):
    h, w = floor.shape
    cr, cc = counter
    collector = _mask(floor.shape, [(cr+i, cc+j) for i in range(3) for j in range(3)])
    chamber = np.zeros_like(floor)
    r, c, rh, rw = (room[x] for x in ("r", "c", "h", "w"))
    chamber[r:r+rh, c:c+rw] = True
    others = floor & ~collector & ~chamber
    forbidden = _dilate(others)
    inside = np.zeros_like(floor)
    inside[1:-1, 1:-1] = True
    starts = list(zip(*np.where(_dilate(collector) & ~floor & ~forbidden & ~_dilate(chamber) & inside)))
    goals = list(zip(*np.where(_dilate(chamber) & ~floor & ~forbidden & ~_dilate(collector) & inside)))
    # Keep the wall immediately behind all bar roots intact.
    direction = room["direction"]
    goals = [(rr, co) for rr, co in goals if not (
        (direction == "up" and rr == r+rh) or (direction == "down" and rr == r-1)
        or (direction == "left" and co == c+rw) or (direction == "right" and co == c-1))]
    if not starts or not goals:
        return None
    allowed_base = ~_dilate(floor) & inside
    for _ in range(8):
        start = starts[int(rng.integers(len(starts)))]
        end = goals[int(rng.integers(len(goals)))]
        allowed = allowed_base.copy()
        allowed[start] = allowed[end] = True
        costs = rng.integers(1, 12, size=floor.shape)
        distance, previous = {start: 0}, {}
        heap = [(0, start)]
        while heap:
            cost, cell = heapq.heappop(heap)
            if cost != distance[cell]:
                continue
            if cell == end:
                path = [cell]
                while path[-1] != start:
                    path.append(previous[path[-1]])
                return path
            for other in _neighbors(cell, floor.shape):
                new_cost = cost + int(costs[other])
                if allowed[other] and new_cost < distance.get(other, 10**9):
                    distance[other] = new_cost
                    previous[other] = cell
                    heapq.heappush(heap, (new_cost, other))
    return None


def _nearer_decoy(scene):
    r, c = scene["counter"]
    def distance(cells):
        return min(max(r-rr, rr-r-2, 0) + max(c-cc, cc-c-2, 0) for rr, cc in cells)
    return any(distance(d["cells"]) < distance(scene["bars"]["cells"]) for d in scene["decoys"])


def _make_pair(rng, roles, direction, request):
    bg, wall, slot, marker = roles
    palette = [c for c in range(10) if c not in roles]
    for _ in range(96):
        room = _room(rng, request["n"], direction, palette, request.get("total"), request["anchor"])
        h = int(rng.integers(max(17, room["h"]+4), 31))
        w = int(rng.integers(max(17, room["w"]+4), 31))
        counter = ((h-3)//2 + int(rng.integers(-1, 2)),
                   (w-3)//2 + int(rng.integers(-1, 2)))
        floor = np.zeros((h, w), dtype=bool)
        r, c = counter
        floor[r:r+3, c:c+3] = True
        positions = _positions(floor.shape, room, floor)
        if not positions:
            continue
        if request["nearer"]:
            positions.sort(key=lambda p: _rectangle_distance(p, room, counter), reverse=True)
            positions = positions[:max(1, len(positions)//4)]
        room = _place(floor, room, positions[int(rng.integers(len(positions)))])
        decoys = []
        for j in range(request["decoys"]):
            decoy = _room(rng, int(rng.integers(3, 5)), direction, palette, decoy=True)
            positions = _positions(floor.shape, decoy, floor)
            if not positions:
                continue
            if request["nearer"] and j == 0:
                positions.sort(key=lambda p: _rectangle_distance(p, decoy, counter))
                positions = positions[:max(1, len(positions)//5)]
            decoys.append(_place(floor, decoy, positions[int(rng.integers(len(positions)))]))
        if request["decoys"] and not decoys:
            continue
        path = _route(rng, floor, counter, room)
        if path is None:
            continue
        for cell in path:
            floor[cell] = True
        # Add bounded dead ends without opening any disconnected chamber.
        rooms_mask = floor.copy()
        for cell in path:
            rooms_mask[cell] = False
        forbidden = _dilate(rooms_mask)
        for _ in range(int(rng.integers(4))):
            tip = path[int(rng.integers(len(path)))]
            for _ in range(int(rng.integers(1, 7))):
                options = [p for p in _neighbors(tip, floor.shape)
                           if 0 < p[0] < h-1 and 0 < p[1] < w-1 and not floor[p]
                           and not forbidden[p] and sum(floor[q] for q in _neighbors(p, floor.shape)) == 1]
                if not options:
                    break
                tip = options[int(rng.integers(len(options)))]
                floor[tip] = True
        a = np.full((h, w), wall, dtype=np.int8)
        a[floor] = bg
        a[r:r+3, c:c+3] = slot
        a[r+1, c+1] = marker
        for array in [room] + decoys:
            _paint_room(a, array)
        scene = _scene(a)
        if request["nearer"] and not _nearer_decoy(scene):
            continue
        b = a.copy()
        for root, length in zip(_room_roots(room), room["lengths"]):
            for cell in _bar_cells(root, length, direction):
                b[cell] = bg
        _paint_room(b, room, shifted=True)
        stream = []
        n = len(room["colors"])
        for i in range(n):
            stream.extend([room["colors"][(i-1) % n]] * room["lengths"][(i+1) % n])
        for i, color in enumerate(stream):
            b[r+i//3, c+i%3] = color
        return {"input": a, "output": b}
    raise RuntimeError("Bounded lane construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(int(seed))
    roles = tuple(int(c) for c in rng.choice(10, size=4, replace=False))
    direction = str(rng.choice(DIRECTIONS))
    rich_n = int(rng.integers(3, 6))
    requests = [dict(n=3, total=4, anchor=True, decoys=int(rng.integers(1, 3)), nearer=False),
                dict(n=rich_n, total=int(rng.integers(max(6, rich_n+1), 10)),
                     anchor=True, decoys=int(rng.integers(1, 4)), nearer=True)]
    for _ in range(nt-2):
        requests.append(dict(n=int(rng.integers(3, 6)), anchor=False,
                             decoys=int(rng.integers(4)), nearer=False))
    requests = [requests[int(i)] for i in rng.permutation(nt)]
    # A new sequence length, rather than a new hidden convention, is the shift.
    test_n = max(q["n"] for q in requests) + 1
    tests = [dict(n=test_n, anchor=False, decoys=int(rng.integers(4)), nearer=False)
             for _ in range(ns)]
    result, seen = {"train": [], "test": []}, set()
    for split, specifications in (("train", requests), ("test", tests)):
        for specification in specifications:
            for _ in range(16):
                pair = _make_pair(rng, roles, direction, specification)
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    result[split].append(pair)
                    break
            else:
                raise RuntimeError("Bounded distinct-input construction exhausted")
    validate_generated_puzzle(result)
    return result


def pair_features(input_grid):
    a = _grid(input_grid)
    s = _scene(a)
    b = s["bars"]
    return dict(shape=list(a.shape), count=len(b["colors"]), total=sum(b["lengths"]),
                direction=b["direction"], gap=b["gap"], decoys=len(s["decoys"]),
                repeated_colors=len(set(b["colors"])) < len(b["colors"]),
                repeated_lengths=len(set(b["lengths"])) < len(b["lengths"]),
                all_equal_lengths=len(set(b["lengths"])) == 1,
                nearer_decoy=_nearer_decoy(s),
                path_distance=max(int(s["distance"][p]) for p in b["cells"]),
                palette=[s[k] for k in ("background", "wall", "slot", "marker")])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    scenes, seen, roles = {"train": [], "test": []}, set(), set()
    for split in scenes:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema")
            a, out = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == out.shape, "Shape must be preserved")
            _require(_key(a) not in seen, "Inputs must be distinct")
            seen.add(_key(a))
            s = _scene(a)
            b = s["bars"]
            scenes[split].append(s)
            roles.add(tuple(s[k] for k in ("background", "wall", "slot", "marker")))
            _require(np.array_equal(out, solve(a)), "Input-only solve disagrees")
            _require(np.array_equal(a[~s["region"]], out[~s["region"]]), "Disconnected lanes changed")
            _require(np.array_equal(a[a == s["wall"]], out[a == s["wall"]]), "Walls changed")
            # Inspect the returned lane separately from the counter readout.
            new_payload = s["region"] & ~s["collector"] & ~np.isin(
                out, (s["background"], s["wall"], s["slot"], s["marker"]))
            new = _bars(out, list(zip(*np.where(new_payload))))
            _require(new["roots"] == b["roots"], "Slot roots moved")
            _require(new["colors"] == np.roll(b["colors"], 1).tolist(), "Wrong color cycle")
            _require(new["lengths"] == np.roll(b["lengths"], -1).tolist(), "Wrong length cycle")
            _require(Counter(new["colors"]) == Counter(b["colors"])
                     and Counter(new["lengths"]) == Counter(b["lengths"]), "Attribute multisets changed")
            stream = [color for color, length in zip(new["colors"], new["lengths"]) for _ in range(length)]
            r, c = s["counter"]
            before, after = a[r:r+3, c:c+3].ravel(), out[r:r+3, c:c+3].ravel()
            _require(after[:len(stream)].tolist() == stream, "Counter must serialize new bars")
            _require(np.array_equal(before[len(stream):], after[len(stream):]), "Unfilled counter slots changed")
            _require(not np.array_equal(out, _transform(a, s, fill=False)), "Counter contribution invisible")
    _require(len(roles) == 1, "Structural palette conventions must be shared within a puzzle")
    train = scenes["train"]
    _require(any(sum(s["bars"]["lengths"]) == 4 for s in train), "Need a marker-preserving four-slot demonstration")
    _require(any(sum(s["bars"]["lengths"]) >= 6 and _observable(s["bars"]) for s in train),
             "Need a marker-overwriting informative demonstration")
    _require(any(s["decoys"] for s in train) and any(_nearer_decoy(s) for s in train),
             "Training must refute all-lanes and nearest-lane shortcuts")
    for cs in (-1, 0, 1):
        for ls in (-1, 0, 1):
            if (cs, ls) != (1, -1):
                _require(any(not np.array_equal(np.asarray(pair["output"]), _transform(np.asarray(pair["input"]), s, cs, ls))
                             for pair, s in zip(puzzle["train"], train)), "Training fails to identify opposite cycles")
    maximum = max(len(s["bars"]["colors"]) for s in train)
    _require(all(len(s["bars"]["colors"]) == maximum+1 for s in scenes["test"]), "Missing sequence-length test shift")


def self_test():
    """Bounded 42-seed diversity, evidence, API, and mutation checks; no files."""
    # Hand-calculated [2,4,6] / [1,2,3] regression, with a disconnected array.
    example = np.ones((21, 23), dtype=np.int8)
    example[2:7, 2:11] = 0
    example[3, 11:17] = 0
    example[3:11, 16] = 0
    example[10, 13:17] = 0
    example[14:19, 8:17] = 0
    example[9:12, 10:13] = 8
    example[10, 11] = 9
    example[6, 3] = 2
    example[5:7, 5] = 4
    example[4:7, 7] = 6
    example[17:19, 9] = 3
    example[16:19, 11] = 5
    example[18, 13] = 7
    expected = example.copy()
    expected[6, 3] = expected[5:7, 5] = expected[4:7, 7] = 0
    expected[5:7, 3] = 6
    expected[4:7, 5] = 2
    expected[6, 7] = 4
    record = np.array([[6, 6, 2], [2, 2, 4], [8, 8, 8]], dtype=np.int8)
    # Transform the geometry, then independently write the displayed counter.
    for flip in (False, True):
        for transpose in (False, True):
            a = example[::-1].copy() if flip else example.copy()
            b = expected[::-1].copy() if flip else expected.copy()
            if transpose:
                a, b = a.T.copy(), b.T.copy()
            r, c, _, _ = _counter(a)
            b[r:r+3, c:c+3] = record
            _require(np.array_equal(solve(a), b), "Hand-calculated cycle/readout regression")
    overflow = example.copy()
    overflow[2:6, 3] = 2
    try:
        solve(overflow)
    except AssertionError:
        pass
    else:
        raise AssertionError("Counter overflow accepted")
    fingerprints, features = set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(pair["input"]) for split in ("train", "test") for pair in puzzle[split]))
        features.extend(dict(pair_features(pair["input"]), split=split)
                        for split in ("train", "test") for pair in puzzle[split])
    _require(len(fingerprints) == 42, "Seed diversity collapsed")
    _require(len({tuple(f["shape"]) for f in features}) >= 40, "Shape diversity collapsed")
    _require({f["count"] for f in features} == {3, 4, 5, 6}, "Missing bar counts")
    _require({f["direction"] for f in features} == set(DIRECTIONS), "Missing directions")
    _require({f["total"] for f in features} == set(range(3, 10)), "Missing counter loads")
    _require({f["gap"] for f in features} == {1, 2}, "Missing slot spacings")
    for flag in ("repeated_colors", "repeated_lengths", "all_equal_lengths", "nearer_decoy"):
        _require(sum(f[flag] for f in features) >= 8, "Missing structural regime: " + flag)
    _require(any(f["decoys"] == 0 for f in features) and any(f["decoys"] >= 2 for f in features), "Missing lane-count variation")
    for nt, ns in ((2, 1), (10, 1), (10, 4)):
        a, b = generate_puzzle(8127, nt, ns), generate_puzzle(8127, nt, ns)
        _require(all(np.array_equal(x[k], y[k]) for split in ("train", "test")
                     for x, y in zip(a[split], b[split]) for k in ("input", "output")), "Nondeterministic puzzle")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (3.0, 1), (True, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    p = generate_puzzle(903, 2, 1)
    original = p["test"][0]["input"].copy()
    solve(original)
    _require(np.array_equal(original, p["test"][0]["input"]), "solve mutated its input")
    p["test"][0]["output"] = original
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt label accepted")
    return dict(seeds=42, pairs=len(features), distinct_puzzles=len(fingerprints),
                shapes=len({tuple(f["shape"]) for f in features}),
                counts=sorted({f["count"] for f in features}), totals=sorted({f["total"] for f in features}),
                directions=sorted({f["direction"] for f in features}),
                decoy_counts=sorted({f["decoys"] for f in features}),
                path_distance_range=[min(f["path_distance"] for f in features), max(f["path_distance"] for f in features)],
                flags={flag: sum(f[flag] for f in features) for flag in
                       ("repeated_colors", "repeated_lengths", "all_equal_lengths", "nearer_decoy")})


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
