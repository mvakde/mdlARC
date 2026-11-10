"""Defect-built airlocks: reconstruct material, invert holes, dock a square.

Only NumPy and the standard library are used. The input-only solver has no
access to a seed, generation parameters, demonstrations, or supplied labels.
"""
from __future__ import annotations

from collections import deque
import hashlib

import numpy as np


BLANK, HIGHLIGHT, VEHICLE, SOCKET = 0, 7, 8, 9
MATERIAL_COLORS = tuple(range(1, 7))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid cells must be integers")
    _require(np.all((a >= 0) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape):
    r, c = p
    for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    for p in np.argwhere(mask):
        p = tuple(map(int, p))
        if seen[p]:
            continue
        queue = [p]
        seen[p] = True
        for q in queue:
            for t in _neighbors(q, mask.shape):
                if mask[t] and not seen[t]:
                    seen[t] = True
                    queue.append(t)
        groups.append(queue)
    return groups


def _bounds(points):
    a = np.asarray(points, dtype=int)
    lo, hi = a.min(axis=0), a.max(axis=0)+1
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _slice(box):
    r0, c0, r1, c1 = box
    return slice(r0, r1), slice(c0, c1)


def _contains(box, point):
    r0, c0, r1, c1 = box
    return r0 <= point[0] < r1 and c0 <= point[1] < c1


def _contact(a, b):
    return (((a[2] == b[0] or b[2] == a[0]) and max(a[1], b[1]) < min(a[3], b[3]))
            or ((a[3] == b[1] or b[3] == a[1]) and max(a[0], b[0]) < min(a[2], b[2])))


def _square(points):
    box = _bounds(points)
    k = box[2]-box[0]
    _require(k == box[3]-box[1] and len(points) == k*k, "vehicle must be a solid square")
    return (box[0], box[1]), k


def _ring(shape, p):
    r, c = p
    return [(x, y) for x in range(max(0, r-1), min(shape[0], r+2))
            for y in range(max(0, c-1), min(shape[1], c+2)) if (x, y) != p]


def _mark_holes(clean, centers, floor, wall):
    """Combine ring demands; centers are neutral and take precedence."""
    center_set = set(centers)
    demands = {}
    for p in centers:
        _require(clean[p] in (floor, wall), "hole outside recovered material")
        color = wall if clean[p] == floor else floor
        for q in _ring(clean.shape, p):
            if q in center_set:
                continue
            _require(q not in demands or demands[q] == color, "incompatible complementary rings")
            demands[q] = color
    result = clean.copy()
    for p, color in demands.items():
        result[p] = color
    for p in centers:
        result[p] = HIGHLIGHT
    return result


def _reachable(mask, start):
    result = np.zeros(mask.shape, dtype=bool)
    if not (0 <= start[0] < mask.shape[0] and 0 <= start[1] < mask.shape[1] and mask[start]):
        return result
    queue = deque([start])
    result[start] = True
    while queue:
        for q in _neighbors(queue.popleft(), mask.shape):
            if mask[q] and not result[q]:
                result[q] = True
                queue.append(q)
    return result


def _placements(allowed, k):
    return np.lib.stride_tricks.sliding_window_view(allowed, (k, k)).all(axis=(-1, -2))


def _decode(inp):
    """Recover visible roles, rectangles, holes and the ordered room chain."""
    colors = set(map(int, np.unique(inp)))
    materials = colors & set(MATERIAL_COLORS)
    _require(len(materials) == 2 and colors <= materials | {BLANK, VEHICLE, SOCKET},
             "input needs two materials, blank, vehicle and sockets")
    _require(VEHICLE in colors and SOCKET in colors, "missing docking objects")
    start, k = _square(list(map(tuple, np.argwhere(inp == VEHICLE))))
    _require(3 <= k <= 5, "vehicle side must be 3..5")
    socket_positions, floor_colors = [], set()
    for points in _components(inp == SOCKET):
        box = _bounds(points)
        r, c, r1, c1 = box
        _require((r1-r, c1-c) == (k, k), "socket size differs from vehicle")
        expected = np.ones((k, k), dtype=bool)
        expected[1:-1, 1:-1] = False
        _require(np.array_equal(inp[_slice(box)] == SOCKET, expected), "socket is not an intact hollow square")
        interior = set(map(int, np.unique(inp[r+1:r1-1, c+1:c1-1])))
        _require(len(interior) == 1 and interior <= materials, "socket interior must identify floor")
        floor_colors.update(interior)
        socket_positions.append((r, c))
    _require(len(floor_colors) == 1, "sockets disagree about floor")
    floor = next(iter(floor_colors))
    wall = next(iter(materials - {floor}))

    # Intact outer sides determine bounds. Small wrong-fill islands and
    # exterior speckles cannot pass the component support threshold.
    room_mask = np.isin(inp, (floor, VEHICLE, SOCKET))
    rooms = []
    for points in _components(room_mask):
        box = _bounds(points)
        if len(points) >= 12 and min(box[2]-box[0], box[3]-box[1]) >= 5:
            rooms.append(box)
    panels = []
    for points in _components(inp == wall):
        box = _bounds(points)
        if len(points) >= 8 and min(box[2]-box[0], box[3]-box[1]) >= 3:
            panels.append(box)
    _require(2 <= len(rooms) <= 8 and len(panels) == len(rooms)-1, "unsupported room/panel count")
    clean = np.zeros(inp.shape, dtype=np.int8)
    for boxes, color, observed, density in ((rooms, floor, room_mask, .70),
                                             (panels, wall, inp == wall, .45)):
        for box in boxes:
            sl = _slice(box)
            _require(np.all(clean[sl] == BLANK), "recovered rectangles overlap")
            mask = observed[sl]
            _require(mask.mean() >= density, "insufficient intact rectangle evidence")
            _require(min(mask[0].sum(), mask[-1].sum(), mask[:, 0].sum(), mask[:, -1].sum()) >= 2,
                     "rectangle boundaries are not supported")
            clean[sl] = color
    _require(np.all(clean[inp == VEHICLE] == floor) and np.all(clean[inp == SOCKET] == floor),
             "docking object outside floor room")
    all_boxes = rooms+panels
    links = [[j for j, b in enumerate(all_boxes) if j != i and _contact(a, b)]
             for i, a in enumerate(all_boxes)]
    n = len(rooms)
    _require(all(all(j >= n for j in links[i]) for i in range(n)), "rooms must be separated by panels")
    _require(all(len(links[i]) == 2 and all(j < n for j in links[i]) for i in range(n, len(links))),
             "each panel must join exactly two rooms")
    _require(sorted(len(links[i]) for i in range(n)) == [1, 1]+[2]*(n-2), "room graph must be a chain")
    start_rooms = [i for i, box in enumerate(rooms) if _contains(box, start)]
    _require(len(start_rooms) == 1 and len(links[start_rooms[0]]) == 1, "vehicle must start in a terminal room")
    chain = [start_rooms[0]]
    while len(chain) < len(all_boxes):
        options = [j for j in links[chain[-1]] if j not in chain]
        _require(len(options) == 1, "chain disconnected or ambiguous")
        chain.append(options[0])
    ordered_rooms = [rooms[i] for i in chain[::2]]
    ordered_panels = [panels[i-n] for i in chain[1::2]]
    pads = [start]
    for index, box in enumerate(ordered_rooms):
        sockets = [p for p in socket_positions if _contains(box, p)]
        _require(len(sockets) == (0 if index == 0 else 1), "one socket is required in each later room")
        if sockets:
            p = sockets[0]
            _require(_contains(box, (p[0]+k-1, p[1]+k-1)), "socket crosses room edge")
            pads.append(p)
    _require(_contains(ordered_rooms[0], (start[0]+k-1, start[1]+k-1)), "vehicle crosses room edge")
    centers = list(map(tuple, np.argwhere((clean != BLANK) & (inp == BLANK))))
    _require(centers, "no meaningful erasures")
    wrong = (inp != clean) & (inp != BLANK) & ~np.isin(inp, (VEHICLE, SOCKET))
    _require(np.count_nonzero(wrong) <= max(12, inp.size//12), "too much ordinary noise")
    return {"clean": clean, "floor": floor, "wall": wall, "start": start, "k": k,
            "rooms": ordered_rooms, "panels": ordered_panels, "pads": pads, "centers": centers,
            "wrong": wrong}


def _analyze(input_grid):
    inp = _grid(input_grid)
    data = _decode(inp)
    clean, floor, wall, k, pads, centers = (data[key] for key in ("clean", "floor", "wall", "k", "pads", "centers"))
    terrain = _mark_holes(clean, centers, floor, wall)
    allowed = np.isin(terrain, (floor, HIGHLIGHT))
    # The sockets remain legal parking sites, even on the unreachable side.
    for r, c in pads:
        _require(allowed[r:r+k, c:c+k].all(), "a ring interferes with a parking footprint")
        _require(np.all(clean[r:r+k, c:c+k] == floor), "parking square leaves its room")
    valid = _placements(allowed, k)
    reach = _reachable(valid, pads[0])
    levels = [bool(reach[p]) for p in pads]
    _require(levels[0] and levels == sorted(levels, reverse=True), "reachable docks must form a chain prefix")
    level = sum(levels)-1
    destination = pads[level]
    output = terrain.copy()
    output[inp == SOCKET] = SOCKET
    r, c = destination
    output[r:r+k, c:c+k] = VEHICLE
    data.update({"terrain": terrain, "allowed": allowed, "valid": valid, "reachable": reach,
                 "levels": levels, "level": level, "destination": destination, "output": output})
    return data


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the transformed grid using only its visible input conventions."""
    return _analyze(input_grid)["output"]


def _axis(rng, count, low, high, limit):
    for _ in range(12):
        gaps = [int(rng.integers(3, 6)) for _ in range(count-1)]
        if count*low+sum(gaps) <= limit:
            break
    else:
        gaps = [3]*(count-1)
    _require(count*low+sum(gaps) <= limit, "lattice does not fit")
    sizes = [low]*count
    budget = limit-sum(sizes)-sum(gaps)
    for i in rng.permutation(count):
        extra = int(rng.integers(min(high-low, budget)+1))
        sizes[int(i)] += extra
        budget -= extra
    coords = [0]
    for i in range(count-1):
        coords.append(coords[-1]+sizes[i]+gaps[i])
    return coords, sizes


def _walk(rng, shape, count):
    for _ in range(40):
        start = (int(rng.integers(shape[0])), int(rng.integers(shape[1])))
        path = [start]
        while len(path) < count:
            options = [q for q in _neighbors(path[-1], shape) if q not in path]
            if not options:
                break
            path.append(options[int(rng.integers(len(options)))])
        if len(path) == count:
            return path
    # A serpentine lattice route is a finite geometric fallback, not a new rule.
    path = [(r, c) for r in range(shape[0])
            for c in (range(shape[1]) if r % 2 == 0 else range(shape[1]-1, -1, -1))]
    return path[:count]


def _pad_at(rng, room, k):
    r0, c0, r1, c1 = room
    offsets = []
    for length in (r1-r0, c1-c0):
        if rng.random() < .60:
            offsets.append(int(rng.choice([0, length-k])))
        else:
            offsets.append(int(rng.integers(length-k+1)))
    return r0+offsets[0], c0+offsets[1]


def _draw_socket(inp, p, k):
    r, c = p
    inp[r, c:c+k] = inp[r+k-1, c:c+k] = SOCKET
    inp[r:r+k, c] = inp[r:r+k, c+k-1] = SOCKET


def _layout(rng, k, room_count, blocked, floor, wall):
    low, high = k+2, min(9, k+4)
    shapes = [(a, b) for a in range(1, 4) for b in range(1, 4)
              if a*b >= room_count and max(a, b)*low+3*(max(a, b)-1) <= 30
              and a*b <= room_count+2]
    nr, nc = shapes[int(rng.integers(len(shapes)))]
    rows, heights = _axis(rng, nr, low, high, 30)
    cols, widths = _axis(rng, nc, low, high, 30)
    path = _walk(rng, (nr, nc), room_count)
    rooms = [(rows[r], cols[c], rows[r]+heights[r], cols[c]+widths[c]) for r, c in path]
    panels, horizontal = [], []
    for a, b in zip(rooms, rooms[1:]):
        if a[0] == b[0]:
            left, right = sorted((a, b), key=lambda x: x[1])
            breadth = int(rng.integers(k+1, a[2]-a[0]+1))
            offset = int(rng.integers(a[2]-a[0]-breadth+1))
            panels.append((a[0]+offset, left[3], a[0]+offset+breadth, right[1]))
            horizontal.append(True)
        else:
            top, bottom = sorted((a, b), key=lambda x: x[0])
            breadth = int(rng.integers(k+1, a[3]-a[1]+1))
            offset = int(rng.integers(a[3]-a[1]-breadth+1))
            panels.append((top[2], a[1]+offset, bottom[0], a[1]+offset+breadth))
            horizontal.append(False)
    shape = (max(b[2] for b in rooms), max(b[3] for b in rooms))
    clean = np.zeros(shape, dtype=np.int8)
    for boxes, color in ((rooms, floor), (panels, wall)):
        for box in boxes:
            clean[_slice(box)] = color
    pads = [_pad_at(rng, room, k) for room in rooms]
    pad_mask = np.zeros(shape, dtype=bool)
    for r, c in pads:
        pad_mask[r:r+k, c:c+k] = True
    inp = clean.copy()
    r, c = pads[0]
    inp[r:r+k, c:c+k] = VEHICLE
    for p in pads[1:]:
        _draw_socket(inp, p, k)

    first_block = int(rng.integers(1, len(panels))) if blocked else len(panels)
    centers = []
    # A cluster fills the panel's thickness, so its neighborhood constructs
    # an aperture. Its transverse extent, not the number of holes, sets size.
    for index, (box, is_horizontal) in enumerate(zip(panels, horizontal)):
        r0, c0, r1, c1 = box
        breadth = r1-r0 if is_horizontal else c1-c0
        must_pass = index < first_block
        if index == first_block:
            aperture = int(rng.integers(3, k))
        elif must_pass or not blocked:
            aperture = int(rng.integers(k, breadth+1))
        else:
            aperture = int(rng.integers(3, breadth+1))
        windows = [(int(rng.integers(breadth-aperture+1)), aperture)]
        # Two separate small doors defeat a total-open-area shortcut.
        if breadth >= 7 and (index == first_block or (not must_pass and rng.random() < .35)):
            if k > 3 and rng.random() < .65:
                offset = int(rng.integers(breadth-7+1))
                windows = [(offset, 3), (offset+4, 3)]
        for offset, aperture in windows:
            if is_horizontal:
                new = [(r, c) for r in range(r0+offset+1, r0+offset+aperture-1)
                       for c in range(c0+1, c1-1)]
            else:
                new = [(r, c) for r in range(r0+1, r1-1)
                       for c in range(c0+offset+1, c0+offset+aperture-1)]
            centers.extend(new)
    for p in centers:
        inp[p] = BLANK

    # Floor holes show the reverse polarity and can affect approaches.
    candidates = []
    for r0, c0, r1, c1 in rooms:
        corners = [(r0, c0), (r0, c1-1), (r1-1, c0), (r1-1, c1-1)]
        candidates.extend(corners)
        if rng.random() < .35:
            for r, c in corners:
                candidates.append((r+(1 if r == r0 else -1), c))
                candidates.append((r, c+(1 if c == c0 else -1)))
    rng.shuffle(candidates)
    floor_centers = []
    target = int(rng.integers(1, min(6, room_count+1)))
    for p in candidates:
        if len(floor_centers) >= target:
            break
        if p in floor_centers or clean[p] != floor or pad_mask[p]:
            continue
        if any(pad_mask[q] for q in _ring(shape, p)):
            continue
        if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2 for q in centers):
            continue
        if sum(_contains(room, p) and _contains(room, q) for room in rooms for q in floor_centers) >= 3:
            continue
        floor_centers.append(p)
        inp[p] = BLANK
    _require(floor_centers, "layout leaves no evident floor defect")
    centers.extend(floor_centers)

    # Sparse wrong-fill noise is ordinary damage, not a marker instruction.
    int_noise_target = int(rng.integers(1, 5))
    candidates = []
    for box in rooms+panels:
        r0, c0, r1, c1 = box
        candidates.extend((r, c) for r in range(r0+1, r1-1) for c in range(c0+1, c1-1)
                          if not pad_mask[r, c] and inp[r, c] == clean[r, c])
    rng.shuffle(candidates)
    wrong = []
    for p in candidates:
        if len(wrong) >= int_noise_target:
            break
        if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2 for q in wrong):
            continue
        # Do not let a wrong-fill island attach to a real other-material edge.
        opposite = wall if clean[p] == floor else floor
        if any(inp[q] == opposite for q in _ring(shape, p)):
            continue
        inp[p] = opposite
        wrong.append(p)

    # Crop the unused lattice margins, then pad freely within the ARC bound.
    occupied = np.argwhere(clean != BLANK)
    r0, c0, r1, c1 = _bounds(occupied)
    clean, inp = clean[r0:r1, c0:c1], inp[r0:r1, c0:c1]
    padding = []
    for length in inp.shape:
        extra = int(rng.integers(min(5, 30-length)+1))
        before = int(rng.integers(extra+1))
        padding.append((before, extra-before))
    clean, inp = np.pad(clean, padding), np.pad(inp, padding)
    exterior = []
    for p in np.argwhere(clean == BLANK):
        p = tuple(map(int, p))
        if all(clean[q] == BLANK for q in _ring(clean.shape, p)):
            exterior.append(p)
    rng.shuffle(exterior)
    speckles = []
    speckle_target = int(rng.integers(1, 5))
    for p in exterior:
        if len(speckles) >= speckle_target:
            break
        if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2 for q in speckles):
            continue
        inp[p] = int(rng.choice([floor, wall]))
        speckles.append(p)
    return inp, clean, first_block


def _make_pair(rng, k, room_count, blocked, palette, seen):
    floor, wall = palette
    for _ in range(120):
        try:
            inp, intended, first_block = _layout(rng, k, room_count, blocked, floor, wall)
            data = _analyze(inp)
            _require(np.array_equal(data["clean"], intended), "input reconstruction differs from intended rectangles")
            _require(data["level"] == (first_block if blocked else room_count-1), "approach obstacle changes requested docking contrast")
            _require(data["level"] > 0, "examples must demonstrate travel")
            turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
            def transform(a):
                b = np.rot90(a, turns)
                return np.array(np.fliplr(b) if reflect else b, dtype=np.int8, copy=True)
            pair = {"input": transform(inp), "output": transform(data["output"])}
            signature = (pair["input"].shape, pair["input"].tobytes())
            _require(signature not in seen, "duplicate input")
            _require(np.array_equal(solve(pair["input"]), pair["output"]), "orientation changes interpretation")
            seen.add(signature)
            return pair
        except AssertionError:
            continue
    raise RuntimeError("bounded airlock construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.choice(MATERIAL_COLORS, 2, replace=False)))
    vary_palette = bool(rng.integers(2))
    scale_shift = bool(rng.integers(2))
    # Even two demonstrations contain a same-size success/stop comparison.
    specs = [(4, int(rng.integers(3, 5)), False), (4, int(rng.integers(3, 5)), True)]
    for _ in range(int(num_train)-2):
        k = int(rng.integers(3, 5))
        blocked = bool(k > 3 and rng.integers(2))
        specs.append((k, int(rng.integers(3 if blocked else 2, 5)), blocked))
    train_specs = [specs[int(i)] for i in rng.permutation(len(specs))]
    test_specs = []
    for _ in range(int(num_test)):
        k = 5 if scale_shift else int(rng.integers(3, 6))
        test_specs.append((k, int(rng.integers(5, 7)), bool(k > 3 and rng.integers(2))))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, group in (("train", train_specs), ("test", test_specs)):
        for k, rooms, blocked in group:
            palette = tuple(map(int, rng.choice(MATERIAL_COLORS, 2, replace=False))) if vary_palette else base_palette
            puzzle[split].append(_make_pair(rng, k, rooms, blocked, palette, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, data=None):
    data = _analyze(inp) if data is None else data
    clean, centers, floor, wall = (data[key] for key in ("clean", "centers", "floor", "wall"))
    wall_holes = [p for p in centers if clean[p] == wall]
    floor_holes = [p for p in centers if clean[p] == floor]
    hole_mask = np.zeros(clean.shape, dtype=bool)
    for p in wall_holes:
        hole_mask[p] = True
    window_counts = [len(_components(hole_mask[_slice(box)])) for box in data["panels"]]
    # Slot room sizes can differ along the same axis. Determine heading from
    # panel contact, instead of mistaking that difference for a diagonal turn.
    headings = []
    for a, b in zip(data["rooms"], data["rooms"][1:]):
        headings.append((0, int(np.sign(b[1]-a[1]))) if a[0] == b[0]
                        else (int(np.sign(b[0]-a[0])), 0))
    normalized = np.zeros(clean.shape, dtype=np.int8)
    for role, color in enumerate((floor, wall, HIGHLIGHT, VEHICLE, SOCKET), 1):
        normalized[np.asarray(inp) == color] = role
    return {"shape": clean.shape, "k": data["k"], "rooms": len(data["rooms"]),
            "turns": sum(a != b for a, b in zip(headings, headings[1:])),
            "level": data["level"], "success": data["level"] == len(data["pads"])-1,
            "floor": floor, "wall": wall, "floor_holes": len(floor_holes), "wall_holes": len(wall_holes),
            "ordinary_noise": int(data["wrong"].sum()), "multiple_windows": any(n > 1 for n in window_counts),
            "clipped": any(p[0] in (0, clean.shape[0]-1) or p[1] in (0, clean.shape[1]-1) for p in centers),
            "floor_cluster": any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) == 1
                                 for i, p in enumerate(floor_holes) for q in floor_holes[:i]),
            "rectangle_sizes": tuple((b[2]-b[0], b[3]-b[1]) for b in data["rooms"]),
            "structure": hashlib.sha256(normalized.tobytes()+str(clean.shape).encode()).hexdigest()}


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "illegal pair counts")
    seen, training, test = set(), [], []
    for split, rows in (("train", training), ("test", test)):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape must be preserved")
            signature = (inp.shape, inp.tobytes())
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            data = _analyze(inp)
            _require(np.array_equal(out, data["output"]), "supplied answer disagrees with input-only rule")
            f = _features(inp, data)
            _require(f["floor_holes"] > 0 and f["wall_holes"] > 0, "both inversion polarities must be visible")
            _require(f["level"] > 0, "a generated example must move")
            _require(np.count_nonzero(out == VEHICLE) == data["k"]**2, "square area changes")
            _require(np.all(out[tuple(np.array(data["centers"]).T)] == HIGHLIGHT), "a center was overwritten")
            # All original barrier panels are closed. Suppressing B leaves
            # every later dock unreachable, unlike the generated answer.
            base_reach = _reachable(_placements(data["clean"] == data["floor"], data["k"]), data["start"])
            _require(not any(base_reach[p] for p in data["pads"][1:]), "B ablation does not change docking")
            # Point connectivity is deliberately insufficient: stopped square
            # examples still have a point route through every airlock.
            point_reach = _reachable(data["allowed"], data["start"])
            _require(point_reach[data["pads"][-1]], "point path should traverse all gates")
            rows.append(f)
    _require(any(f["success"] for f in training) and any(not f["success"] for f in training),
             "training must expose success and footprint blockage")
    _require(any(f["ordinary_noise"] > 0 for f in training), "training must distinguish noise from erasures")
    _require(min(f["rooms"] for f in test) > max(f["rooms"] for f in training), "missing room-chain train/test shift")


def _fingerprint(puzzle):
    return tuple((np.asarray(p["input"]).shape, np.asarray(p["input"]).tobytes(), np.asarray(p["output"]).tobytes())
                 for split in ("train", "test") for p in puzzle[split])


def _fixture(k, second_aperture):
    """An analytic three-room case, not a sampled or stored parent puzzle."""
    floor, wall = 2, 5
    inp = np.full((9, 27), floor, dtype=np.int8)
    inp[:, 7:10] = inp[:, 17:20] = wall
    inp[4:4+k, 0:k] = VEHICLE
    _draw_socket(inp, (4, 10), k)
    _draw_socket(inp, (4, 20), k)
    inp[3:6, 8] = BLANK  # first door has height five
    # Aperture starts at row 2; its erasure strip is inset by one.
    inp[3:3+second_aperture-2, 18] = BLANK
    inp[0, 0] = BLANK    # floor defect creates a clipped obstacle
    inp[1, 12] = wall    # ordinary wrong-fill error, not an instruction
    return inp


def self_test():
    """Bounded checks and grid-derived diversity evidence across 42 seeds."""
    for k, aperture, destination in ((4, 3, (4, 10)), (3, 3, (4, 20)), (4, 4, (4, 20))):
        inp = _fixture(k, aperture)
        data = _analyze(inp)
        _require(data["destination"] == destination, "analytic size/aperture contrast failed")
        _require(data["output"][1, 12] == 2, "wrong-fill noise became a defect")
        _require(data["output"][0, 0] == HIGHLIGHT and data["output"][0, 1] == 5,
                 "floor inversion or clipping failed")
        _require(np.array_equal(_mark_holes(data["clean"], data["centers"], 2, 5),
                                _mark_holes(data["clean"], list(reversed(data["centers"])), 2, 5)),
                 "ring order affects output")
    # Two three-row windows have six open rows in total, but a four-square
    # still cannot cross the wall row between them.
    split = _fixture(4, 3)
    split[7, 18] = BLANK
    _require(_analyze(split)["destination"] == (4, 10), "total-open-width shortcut accepted")
    # The semantic rule includes immediate blockage, although teaching
    # samples are conditioned to demonstrate positive movement.
    closed_first = _fixture(4, 3)
    closed_first[3:6, 8] = 5
    closed_first[4, 8] = BLANK
    _require(_analyze(closed_first)["destination"] == (4, 0), "zero-travel family member failed")
    clean = np.full((5, 8), 2, dtype=np.int8)
    clean[:, 4:] = 5
    try:
        _mark_holes(clean, [(2, 3), (2, 4)], 2, 5)
    except AssertionError:
        pass
    else:
        raise AssertionError("conflicting ring demands were accepted")
    puzzles, features = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _analyze(pair["input"])
                f = _features(pair["input"], data)
                f["split"] = split
                features.append(f)
                # A separate array-shift ring check verifies centers and both
                # polarity assignments without calling the ring composer.
                clean, floor, wall = (data[key] for key in ("clean", "floor", "wall"))
                expected = clean.copy()
                holes = (pair["input"] == BLANK) & (clean != BLANK)
                for material, opposite in ((floor, wall), (wall, floor)):
                    mask = holes & (clean == material)
                    paint = np.zeros_like(mask)
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == dc == 0:
                                continue
                            rs, cs = slice(max(0, dr), min(mask.shape[0], mask.shape[0]+dr)), slice(max(0, dc), min(mask.shape[1], mask.shape[1]+dc))
                            sr, sc = slice(max(0, -dr), min(mask.shape[0], mask.shape[0]-dr)), slice(max(0, -dc), min(mask.shape[1], mask.shape[1]-dc))
                            paint[rs, cs] |= mask[sr, sc]
                    expected[paint & ~holes] = opposite
                expected[holes] = HIGHLIGHT
                _require(np.array_equal(expected, data["terrain"]), "array-shift ring check failed")
    _require(len(set(puzzles)) == 42, "seeds repeat whole puzzles")
    _require(len({f["structure"] for f in features}) >= 150, "insufficient structural variation")
    _require(len({f["shape"] for f in features}) >= 35, "insufficient shape variation")
    _require({f["k"] for f in features} == {3, 4, 5}, "missing vehicle scales")
    _require(len({f["rooms"] for f in features}) >= 4, "missing room counts")
    _require(len({f["turns"] for f in features}) >= 4, "missing route geometry")
    _require({f["floor"] for f in features} == set(MATERIAL_COLORS)
             and {f["wall"] for f in features} == set(MATERIAL_COLORS), "material palettes collapsed")
    _require(any(f["multiple_windows"] for f in features), "parallel narrow apertures missing")
    _require(any(f["floor_cluster"] for f in features), "floor defect clusters missing")
    _require(any(f["clipped"] for f in features) and any(not f["clipped"] for f in features), "boundary cases collapsed")
    _require(len({f["rectangle_sizes"] for f in features}) >= 75, "room dimensions collapsed")
    _require({f["success"] for f in features if f["split"] == "test"} == {False, True}, "test outcome is fixed")
    _require(_fingerprint(generate_puzzle(7)) == puzzles[7], "seed reproducibility failed")
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            puzzle = generate_puzzle(103+n*7+m, n, m)
            _require(len(puzzle["train"]) == n and len(puzzle["test"]) == m, "explicit count ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid explicit counts accepted")
    bad = generate_puzzle(0)
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a damaged answer")
    return {"seeds": 42, "pairs": len(features), "structures": len({f["structure"] for f in features}),
            "shapes": len({f["shape"] for f in features}), "vehicle_sizes": sorted({f["k"] for f in features}),
            "room_counts": sorted({f["rooms"] for f in features}), "turn_counts": sorted({f["turns"] for f in features}),
            "multiple_window_pairs": sum(f["multiple_windows"] for f in features),
            "clipped_pairs": sum(f["clipped"] for f in features),
            "test_successes": sum(f["success"] for f in features if f["split"] == "test"),
            "test_stops": sum(not f["success"] for f in features if f["split"] == "test"),
            "semantic_verification": "author-owned analytic and invariant checks"}
