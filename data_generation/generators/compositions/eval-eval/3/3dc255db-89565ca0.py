"""Room-distance evacuation. Standard library + NumPy; no files or latent state.

Generation paints labels from rectangle intervals. solve instead reads pixels,
repairs marked gaps, flood-fills rooms and constructs their dual graph.
All diagnostics are bounded and self_test retains only hashes and feature sets.
"""
from collections import deque
from itertools import combinations
import hashlib

import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return bytes(a.shape) + a.tobytes()


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                h.update(_key(pair[role]))
    return h.hexdigest()


def _arrow(base, direction):
    y, x = base
    dy, dx = _DIRECTIONS[direction]
    py, px = -dx, dy
    tip = (y + 2 * dy, x + 2 * dx)
    shoulders = ((y + dy + py, x + dx + px),
                 (y + dy - py, x + dx - px))
    return tip, (tip,) + shoulders


def _ray(shape, port, quantity=None):
    dy, dx = _DIRECTIONS[port["direction"]]
    y, x = port["tip"]
    result = []
    if quantity == 0:
        return result
    while True:
        y, x = y + dy, x + dx
        if not (0 <= y < shape[0] and 0 <= x < shape[1]):
            return result
        result.append((y, x))
        if quantity is not None and len(result) == quantity:
            return result


def _distances(graph):
    n = len(graph)
    distances = np.full((n, n), n + 1, dtype=int)
    for start in range(n):
        distances[start, start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for w in sorted(graph[v]):
                if distances[start, w] == n + 1:
                    distances[start, w] = distances[start, v] + 1
                    queue.append(w)
    if np.any(distances > n):
        raise ValueError("room graph must be connected")
    return distances


def _owners(distances, ports):
    d = distances[:, [p["room"] for p in ports]]
    minima = d.min(axis=1)
    result = d.argmin(axis=1)
    result[(d == minima[:, None]).sum(axis=1) != 1] = -1
    return result


def _flood_rooms(walls, body):
    """Pixel-derived rooms, not generation rectangles."""
    t, l, b, r = body
    labels = np.full(walls.shape, -1, dtype=int)
    rooms = []
    for y in range(t + 1, b):
        for x in range(l + 1, r):
            if walls[y, x] or labels[y, x] >= 0:
                continue
            index = len(rooms)
            stack, cells = [(y, x)], []
            labels[y, x] = index
            while stack:
                yy, xx = stack.pop()
                cells.append((yy, xx))
                for dy, dx in _DIRECTIONS:
                    a, c = yy + dy, xx + dx
                    if (t < a < b and l < c < r and not walls[a, c]
                            and labels[a, c] < 0):
                        labels[a, c] = index
                        stack.append((a, c))
            ys, xs = zip(*cells)
            a, c, d, e = min(ys), min(xs), max(ys), max(xs)
            if len(cells) != (d - a + 1) * (e - c + 1) or min(d-a+1, e-c+1) < 3:
                raise ValueError("elementary rooms must be rectangles with interiors >=3")
            rooms.append((a - 1, c - 1, d + 1, e + 1))
    if not rooms:
        raise ValueError("no rooms")
    return labels, rooms


def _pixel_graph(walls, labels):
    n = int(labels.max()) + 1
    graph = [set() for _ in range(n)]
    h, w = walls.shape
    for y, x in np.argwhere(walls):
        y, x = int(y), int(x)
        for dy, dx in ((1, 0), (0, 1)):
            if 0 <= y-dy < h and y+dy < h and 0 <= x-dx < w and x+dx < w:
                a, b = int(labels[y-dy, x-dx]), int(labels[y+dy, x+dx])
                if a >= 0 and b >= 0 and a != b:
                    graph[a].add(b)
                    graph[b].add(a)
    return graph


def _partial(walls, body):
    degree = np.zeros(walls.shape, dtype=np.int8)
    degree[1:, :] += walls[:-1, :]
    degree[:-1, :] += walls[1:, :]
    degree[:, 1:] += walls[:, :-1]
    degree[:, :-1] += walls[:, 1:]
    t, l, b, r = body
    return bool(np.any(walls[t+1:b, l+1:r] & (degree[t+1:b, l+1:r] == 3)))


def _runs(line):
    edges = np.flatnonzero(np.diff(np.r_[False, line, False]))
    return [(int(b-a), int(a), int(b-1)) for a, b in zip(edges[::2], edges[1::2])]


def _read(input_grid):
    inp = _grid(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    order = np.argsort(counts)
    if len(colors) < 3 or counts[order[-1]] <= inp.size // 2:
        raise ValueError("background must be a unique majority")
    bg, wall = int(colors[order[-1]]), int(colors[order[-2]])
    if counts[order[-2]] == counts[order[-3]]:
        raise ValueError("wall frequency must exceed all other nonbackground colors")
    visible = inp == wall
    horizontal = np.zeros_like(visible)
    vertical = np.zeros_like(visible)
    horizontal[:, 1:-1] = visible[:, :-2] & visible[:, 2:]
    vertical[1:-1, :] = visible[:-2, :] & visible[2:, :]
    repaired = (inp != bg) & ~visible & (horizontal | vertical)
    walls = visible | repaired
    horizontal_runs = [(length, y, a, b) for y, line in enumerate(walls)
                       for length, a, b in _runs(line)]
    width = max(v[0] for v in horizontal_runs)
    full = [(y, a, b) for length, y, a, b in horizontal_runs if length == width]
    t, b = min(v[0] for v in full), max(v[0] for v in full)
    l, r = full[0][1:]
    if (width < 9 or b-t+1 < 9 or any((a, c) != (l, r) for _, a, c in full)
            or not walls[t, l:r+1].all() or not walls[b, l:r+1].all()
            or not walls[t:b+1, l].all() or not walls[t:b+1, r].all()):
        raise ValueError("cannot identify rectangular body")
    body = (t, l, b, r)
    labels, rooms = _flood_rooms(walls, body)
    # Require exactly the union of rectangular room boundaries inside the body.
    expected = np.zeros_like(walls)
    for a, c, d, e in rooms:
        expected[a, c:e+1] = expected[d, c:e+1] = True
        expected[a:d+1, c] = expected[a:d+1, e] = True
    if not np.array_equal(expected[t:b+1, l:r+1], walls[t:b+1, l:r+1]):
        raise ValueError("unexplained wall inside body")
    ports, extra = [], set()
    for direction in range(4):
        bases = ([(t, x) for x in range(l+2, r-1)] if direction == 0 else
                 [(y, r) for y in range(t+2, b-1)] if direction == 1 else
                 [(b, x) for x in range(l+2, r-1)] if direction == 2 else
                 [(y, l) for y in range(t+2, b-1)])
        dy, dx = _DIRECTIONS[direction]
        for base in bases:
            tip, arrow = _arrow(base, direction)
            if all(0 <= y < inp.shape[0] and 0 <= x < inp.shape[1] and walls[y, x]
                   for y, x in arrow):
                room = int(labels[base[0]-dy, base[1]-dx])
                if room < 0:
                    raise ValueError("outlet base must border a room")
                ports.append(dict(base=base, tip=tip, direction=direction, room=room))
                extra.update(arrow)
    outside = walls.copy()
    outside[t:b+1, l:r+1] = False
    if len(ports) < 2 or len({p["room"] for p in ports}) != len(ports):
        raise ValueError("need distinct outlet rooms")
    if set(map(tuple, np.argwhere(outside))) != extra:
        raise ValueError("unexplained exterior wall pixels")
    values = set(map(int, inp[labels >= 0])) - {bg}
    if len(values) != 1:
        raise ValueError("need exactly one interior payload color")
    payload = values.pop()
    cells = [tuple(map(int, p)) for p in np.argwhere((labels >= 0) & (inp == payload))]
    graph = _pixel_graph(walls, labels)
    distances = _distances(graph)
    owners = _owners(distances, ports)
    return dict(inp=inp, background=bg, wall=wall, payload=payload, walls=walls,
                repaired=repaired, body=body, rooms=rooms, labels=labels, graph=graph,
                distances=distances, ports=ports, owners=owners, cells=cells)


def _render(data, mode="graph"):
    """Counterfactuals share only drawing, not the ownership/counting decision."""
    out = data["inp"].copy()
    quantities = [0] * len(data["ports"])
    occupied = [set() for _ in data["ports"]]
    for cell in data["cells"]:
        room = int(data["labels"][cell])
        owner = int(data["owners"][room])
        if mode == "physical":
            ds = [abs(cell[0]-p["tip"][0]) + abs(cell[1]-p["tip"][1])
                  for p in data["ports"]]
            owner = ds.index(min(ds)) if ds.count(min(ds)) == 1 else -1
        if owner >= 0:
            out[cell] = data["background"]
            quantities[owner] += 1
            occupied[owner].add(room)
    if mode == "rooms":
        quantities = [len(v) for v in occupied]
    for port, q in zip(data["ports"], quantities):
        for cell in _ray(out.shape, port, q):
            out[cell] = data["payload"]
    return out, quantities


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer exclusively from the supplied input pixels."""
    return _render(_read(input_grid))[0]


def _capacity(room):
    t, l, b, r = room
    return ((b-t)//4) * ((r-l)//4)


def _partition(rng, h, w, number):
    rooms = [(0, 0, h-1, w-1)]
    for _ in range(number-1):
        options = []
        for index, (t, l, b, r) in enumerate(rooms):
            for axis, start, end in ((0, t, b), (1, l, r)):
                for cut in range(start+4, end-3):
                    halves = (((t, l, cut, r), (cut, l, b, r)) if axis == 0
                              else ((t, l, b, cut), (t, cut, b, r)))
                    capacity = sum(_capacity(v) for j, v in enumerate(rooms) if j != index)
                    if capacity + sum(_capacity(v) for v in halves) >= number:
                        options.append((index, halves))
        if not options:
            raise AssertionError("capacity-preserving partition became stuck")
        index, halves = options[int(rng.integers(len(options)))]
        rooms[index:index+1] = halves
    return rooms


def _interval_graph(rooms):
    """Construction-side graph: interval intersections, independent of flood fill."""
    graph = [set() for _ in rooms]
    for i, a in enumerate(rooms):
        for j, b in enumerate(rooms[:i]):
            horizontal = (a[0] == b[2] or a[2] == b[0]) and min(a[3], b[3])-max(a[1], b[1]) >= 2
            vertical = (a[1] == b[3] or a[3] == b[1]) and min(a[2], b[2])-max(a[0], b[0]) >= 2
            if horizontal or vertical:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _port_candidates(rng, rooms, h, w):
    candidates = []
    for index, (t, l, b, r) in enumerate(rooms):
        for direction, available, start, end in (
                (0, t == 0, l+2, r-2), (1, r == w-1, t+2, b-2),
                (2, b == h-1, l+2, r-2), (3, l == 0, t+2, b-2)):
            if not available or start > end:
                continue
            center = int(rng.integers(start, end+1))
            base = ((0, center) if direction == 0 else (center, w-1) if direction == 1
                    else (h-1, center) if direction == 2 else (center, 0))
            tip, _ = _arrow(base, direction)
            candidates.append(dict(room=index, direction=direction, base=base, tip=tip))
    return candidates


def _physical_counterexample(rooms, owners, ports):
    for room, owner in enumerate(owners):
        if owner < 0 or room == ports[owner]["room"]:
            continue
        t, l, b, r = rooms[room]
        # Distances are piecewise linear; corners and center are useful witnesses.
        ys = sorted({t+1, (t+b)//2, b-1})
        xs = sorted({l+1, (l+r)//2, r-1})
        for cell in ((y, x) for y in ys for x in xs):
            ds = [abs(cell[0]-p["tip"][0]) + abs(cell[1]-p["tip"][1]) for p in ports]
            if ds.count(min(ds)) == 1 and ds.index(min(ds)) != owner:
                return cell, room
    return None


def _select_ports(rng, rooms, h, w, number, witness):
    candidates = _port_candidates(rng, rooms, h, w)
    distances = _distances(_interval_graph(rooms))
    # Enumerate a finite candidate list; do not retain any grids for retries.
    choices = list(combinations(range(len(candidates)), number))
    if not choices:
        return None
    for raw in rng.permutation(len(choices))[:160]:
        ports = [candidates[i] for i in choices[int(raw)]]
        if len({p["room"] for p in ports}) != number:
            continue
        if any(a["direction"] == b["direction"] and
               abs(a["base"][0]-b["base"][0]) + abs(a["base"][1]-b["base"][1]) < 4
               for a, b in combinations(ports, 2)):
            continue
        owners = _owners(distances, ports)
        counterexample = _physical_counterexample(rooms, owners, ports)
        nonlocal_rooms = [i for i, o in enumerate(owners) if o >= 0 and i != ports[o]["room"]]
        if witness and (not np.any(owners < 0) or counterexample is None or not nonlocal_rooms):
            continue
        return ports, owners, distances, counterexample
    return None


def _interior(room):
    t, l, b, r = room
    return [(y, x) for y in range(t+1, b) for x in range(l+1, r)]


def _draw_pair(rng, rooms, h, w, selection, roles, witness, low):
    ports, owners, distances, counterexample = selection
    bg, wall, payload, noise = roles
    number = len(ports)
    # The first two quantities vary independently of room size and geometry.
    targets = [low, low+1] + [int(rng.integers(1, 7)) for _ in range(number-2)]
    if witness not in (1, 2):
        targets = [int(rng.integers(1, 8)) for _ in ports]
        targets[0] = int(rng.integers(1, 4))
    margins = [int(rng.integers(3, 7)) for _ in range(4)]  # U,R,D,L
    guaranteed = range(2) if witness in (1, 2) else range(1)
    for i in guaranteed:
        direction = ports[i]["direction"]
        margins[direction] = max(margins[direction], targets[i]+3)
    if h+margins[0]+margins[2] > 30 or w+margins[1]+margins[3] > 30:
        return None
    top, left = margins[0], margins[3]
    shape = (h+margins[0]+margins[2], w+margins[1]+margins[3])
    inp = np.full(shape, bg, dtype=np.int8)
    labels = np.full(shape, -1, dtype=int)
    for index, (t, l, b, r) in enumerate(rooms):
        t, l, b, r = t+top, l+left, b+top, r+left
        inp[t, l:r+1] = inp[b, l:r+1] = wall
        inp[t:b+1, l] = inp[t:b+1, r] = wall
        labels[t+1:b, l+1:r] = index
    world_ports = []
    for port in ports:
        base = (port["base"][0]+top, port["base"][1]+left)
        tip, arrow = _arrow(base, port["direction"])
        for cell in arrow:
            inp[cell] = wall
        world_ports.append(dict(port, base=base, tip=tip))
    chosen = set()
    # Route a deliberately nonlocal, physically misleading payload in witness 1.
    if witness in (1, 3) and counterexample is not None:
        cell, room = counterexample
        chosen.add(cell)
    for owner, target in enumerate(targets):
        current = sum(owners[next(i for i, box in enumerate(rooms)
                                  if box[0] < c[0] < box[2] and box[1] < c[1] < box[3])] == owner
                      for c in chosen)
        source_rooms = ([ports[owner]["room"]] if witness in (1, 2) else
                        [i for i, destination in enumerate(owners) if destination == owner])
        candidates = [c for room in source_rooms for c in _interior(rooms[room])]
        rng.shuffle(candidates)
        # Deliberately place a connected pair in one room for the high count.
        if owner == 1 and target-current >= 2:
            t, l, b, r = rooms[ports[owner]["room"]]
            y, x = int(rng.integers(t+1, b)), int(rng.integers(l+1, r-1))
            chosen.update(((y, x), (y, x+1)))
            current += 2
        for cell in candidates:
            if current >= target:
                break
            if cell not in chosen:
                chosen.add(cell)
                current += 1
    tied = [i for i, owner in enumerate(owners) if owner < 0]
    if tied:
        for _ in range(int(rng.integers(1, min(3, len(tied))+1))):
            room = tied[int(rng.integers(len(tied))) ]
            candidates = _interior(rooms[room])
            chosen.add(candidates[int(rng.integers(len(candidates)))])
    world_cells = sorted((y+top, x+left) for y, x in chosen)
    for cell in world_cells:
        inp[cell] = payload
    # Single isolated wall occlusions. Internal damage actually opens a room
    # boundary if mistaken for background; adjacent supports are retained.
    wallmask = inp == wall
    eligible = []
    for y in range(top+1, top+h-1):
        for x in range(left+1, left+w-1):
            if not wallmask[y, x]:
                continue
            straight_v = (wallmask[y-1, x] and wallmask[y+1, x]
                          and labels[y, x-1] >= 0 and labels[y, x+1] >= 0)
            straight_h = (wallmask[y, x-1] and wallmask[y, x+1]
                          and labels[y-1, x] >= 0 and labels[y+1, x] >= 0)
            if straight_h or straight_v:
                eligible.append((y, x))
    rng.shuffle(eligible)
    damaged = []
    amount = int(rng.integers(1, 5)) if witness or rng.random() < .8 else 0
    for cell in eligible:
        if len(damaged) == amount:
            break
        if all(max(abs(cell[0]-p[0]), abs(cell[1]-p[1])) > 1 for p in damaged):
            inp[cell] = noise
            damaged.append(cell)
    # Optional exterior specks, kept off complete ray axes and wall supports.
    axes = {c for port in world_ports for c in _ray(shape, port)}
    sites = [(y, x) for y in range(shape[0]) for x in range(shape[1])
             if not (top-2 <= y < top+h+2 and left-2 <= x < left+w+2)
             and (y, x) not in axes]
    if sites:
        for i in rng.choice(len(sites), size=min(int(rng.integers(0, 4)), len(sites)), replace=False):
            inp[sites[int(i)]] = noise
    # Construction truth never calls the input reader or solve.
    out = inp.copy()
    quantities = [0] * number
    for cell in world_cells:
        owner = int(owners[labels[cell]])
        if owner >= 0:
            out[cell] = bg
            quantities[owner] += 1
    for port, quantity in zip(world_ports, quantities):
        for cell in _ray(shape, port, quantity):
            out[cell] = payload
    pair = {"input": inp, "output": out}
    data = _read(inp)
    if set(data["rooms"]) != {(t+top, l+left, b+top, r+left) for t,l,b,r in rooms}:
        raise AssertionError("rectangle construction and pixel recovery disagree")
    if not np.array_equal(out, _render(data)[0]):
        raise AssertionError("construction and pixel-derived graph disagree")
    if witness in (1, 2):
        _, qs = _render(data)
        full = {q for p, q in zip(data["ports"], qs) if len(_ray(shape, p)) > q > 0}
        if len(full) < 2:
            return None
        if witness == 1:
            if not any(data["owners"][data["labels"][c]] < 0 for c in world_cells):
                return None
            if not _partial(data["walls"], data["body"]):
                return None
            if np.array_equal(_render(data, "physical")[0], out):
                return None
        if np.array_equal(_render(data, "rooms")[0], out):
            return None
    if witness == 3 and (np.array_equal(_render(data, "physical")[0], out)
                         or np.array_equal(_render(data, "rooms")[0], out)):
        return None
    return pair


def _make_pair(rng, roles, number_rooms, number_ports, witness, low):
    dimensions = [(h, w) for h in range(9, 19) for w in range(9, 19)
                  if _capacity((0, 0, h-1, w-1)) >= number_rooms]
    for _ in range(160):
        h, w = dimensions[int(rng.integers(len(dimensions)))]
        rooms = _partition(rng, h, w, number_rooms)
        selection = _select_ports(rng, rooms, h, w, number_ports, witness in (1, 3))
        if selection is None:
            continue
        pair = _draw_pair(rng, rooms, h, w, selection, roles, witness, low)
        if pair is not None:
            return pair
    raise AssertionError("bounded geometry search could not expose required witnesses")


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= value <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    local_roles = bool(rng.integers(2))
    low = int(rng.integers(1, 3))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            # Every minimal training set exposes two and three competing sources.
            nports = 2 + index % 2 if split == "train" else 4 if index == 0 else int(rng.integers(3, 5))
            nrooms = int(rng.integers(4, 7)) if split == "train" else int(rng.integers(7, 11))
            palette = tuple(map(int, rng.choice(10, 4, replace=False))) if local_roles else roles
            for _ in range(8):
                witness = index+1 if split == "train" and index < 2 else 3 if split == "test" else 0
                pair = _make_pair(rng, palette, nrooms, nports, witness, low)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("bounded uniqueness retry exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(data, output):
    predicted, quantities = _render(data)
    assert np.array_equal(predicted, output), "wrong room-distance export"
    tied = [c for c in data["cells"] if data["owners"][data["labels"][c]] < 0]
    departing = [c for c in data["cells"] if data["owners"][data["labels"][c]] >= 0]
    assert departing and all(output[c] == data["background"] for c in departing)
    assert all(output[c] == data["payload"] for c in tied)
    assert np.all(output[data["inp"] == data["wall"]] == data["wall"])
    assert np.all(output[data["repaired"]] == data["inp"][data["repaired"]])
    full = {q for p, q in zip(data["ports"], quantities)
            if 0 < q < len(_ray(output.shape, p))}
    complete_owners = {i for i, (p, q) in enumerate(zip(data["ports"], quantities))
                       if 0 < q < len(_ray(output.shape, p))}
    departing_set = set(departing)
    component_counterexample = any(
        int(data["owners"][data["labels"][cell]]) in complete_owners and
        any((cell[0]+dy, cell[1]+dx) in departing_set and
            data["labels"][cell[0]+dy, cell[1]+dx] == data["labels"][cell]
            for dy, dx in _DIRECTIONS)
        for cell in departing)
    punctured_partition = False
    for y, x in np.argwhere(data["repaired"]):
        for dy, dx in ((1, 0), (0, 1)):
            if 0 <= y-dy and y+dy < output.shape[0] and 0 <= x-dx and x+dx < output.shape[1]:
                a, b = data["labels"][y-dy, x-dx], data["labels"][y+dy, x+dx]
                punctured_partition |= a >= 0 and b >= 0 and a != b
    nonlocal_payload = any(data["owners"][data["labels"][c]] >= 0 and
                          data["labels"][c] != data["ports"][data["owners"][data["labels"][c]]]["room"]
                          for c in data["cells"])
    partial = _partial(data["walls"], data["body"])
    return dict(ties=bool(tied), full=full, nonlocal_payload=nonlocal_payload,
                partial=partial, damage=bool(punctured_partition),
                component_counterexample=component_counterexample,
                physical_fails=not np.array_equal(_render(data, "physical")[0], output),
                room_count_fails=not np.array_equal(_render(data, "rooms")[0], output))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, room_counts, port_counts = set(), [], {}, {}
    for split in ("train", "test"):
        room_counts[split], port_counts[split] = [], []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            data = _read(inp)
            evidence = _evidence(data, out)
            room_counts[split].append(len(data["rooms"]))
            port_counts[split].append(len(data["ports"]))
            if split == "train":
                training.append(evidence)
            else:
                assert evidence["nonlocal_payload"] and evidence["physical_fails"] and evidence["room_count_fails"]
    for field in ("ties", "nonlocal_payload", "damage", "physical_fails", "room_count_fails",
                  "component_counterexample", "partial"):
        assert any(e[field] for e in training), f"training lacks {field} witness"
    assert len(set().union(*(e["full"] for e in training))) >= 2, "constant-length shortcut survives"
    assert min(room_counts["test"]) > max(room_counts["train"]), "missing room-count shift"
    assert max(port_counts["train"]) == 3 and port_counts["test"][0] == 4


def self_test():
    """42 streaming seeds; no disk writes, traces, caches, or retained puzzles."""
    fingerprints, shapes, topologies, room_counts, palettes, directions = (set() for _ in range(6))
    clipped = complete = partial = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_digest(puzzle))
        if seed < 2:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _read(pair["input"])
                shapes.add(pair["input"].shape)
                room_counts.add(len(data["rooms"]))
                palettes.add((data["background"], data["wall"], data["payload"]))
                topologies.add(tuple(sorted(tuple(sorted(int(v) for v in data["distances"][i]))
                                            for i in range(len(data["rooms"])))))
                directions.update(p["direction"] for p in data["ports"])
                _, qs = _render(data)
                clipped += sum(q > len(_ray(pair["input"].shape, p)) for p, q in zip(data["ports"], qs))
                complete += sum(0 < q < len(_ray(pair["input"].shape, p)) for p, q in zip(data["ports"], qs))
                partial += _evidence(data, pair["output"])["partial"]
                if seed < 3:
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(pair["input"], k)), np.rot90(pair["output"], k))
                    assert np.array_equal(solve(pair["input"].T), pair["output"].T)
                    permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
                    assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert len(fingerprints) == 42 and len(shapes) >= 25 and len(topologies) >= 15
    assert set(range(4, 11)) <= room_counts and len(palettes) >= 25
    assert directions == set(range(4)) and clipped > 0 and complete > 100 and partial > 50
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(9100+nt*10+ne, nt, ne)
        assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
