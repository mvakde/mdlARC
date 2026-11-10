"""Perimeter carousel: seeded topology supplies a cyclic length-sorting carrier.

Only NumPy and the standard library are used. No parent code/data is imported.
The solver reconstructs the wall underneath socket anchors before selecting a
room; colored bars are annotations, not obstacles to room connectivity.
"""
from __future__ import annotations

from collections import deque
import numpy as np

N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _components(mask):
    remaining = mask.copy()
    result = []
    while np.any(remaining):
        first = int(np.flatnonzero(remaining)[0])
        component = _flood(remaining, divmod(first, mask.shape[1]))
        result.append([tuple(cell) for cell in np.argwhere(component).tolist()])
        remaining[component] = False
    return result


def _flood(mask, start):
    if not mask[start]:
        return np.zeros(mask.shape, bool)
    # A row bitset performs exactly the same orthogonal flood as a cell queue,
    # but each horizontal run propagates in a few integer operations. This also
    # handles the padded outside mask without changing corner connectivity.
    h, w = mask.shape
    packed = np.packbits(mask, axis=1, bitorder="little")
    allowed = [int.from_bytes(row.tobytes(), "little") for row in packed]
    reached = [0] * h
    reached[start[0]] = 1 << start[1]
    queue = deque([start[0]])
    queued = [False] * h
    queued[start[0]] = True
    while queue:
        r = queue.popleft()
        queued[r] = False
        row = reached[r]
        while True:
            expanded = (row | (row << 1) | (row >> 1)) & allowed[r]
            if expanded == row:
                break
            row = expanded
        reached[r] = row
        for neighbor in (r - 1, r + 1):
            if 0 <= neighbor < h:
                addition = row & allowed[neighbor] & ~reached[neighbor]
                if addition:
                    reached[neighbor] |= addition
                    if not queued[neighbor]:
                        queue.append(neighbor)
                        queued[neighbor] = True
    data = b"".join(row.to_bytes(packed.shape[1], "little") for row in reached)
    byte_rows = np.frombuffer(data, dtype=np.uint8).reshape(h, packed.shape[1])
    return np.unpackbits(byte_rows, axis=1, count=w, bitorder="little").astype(bool)


def _halo(mask):
    padded = np.pad(mask, 1)
    h, w = mask.shape
    answer = np.zeros_like(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            answer |= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return answer


def _external(region):
    return _flood(np.pad(~region, 1, constant_values=True), (0, 0))


def _edge(cell, direction):
    """Clockwise directed edge: the room is on its right (screen coordinates)."""
    r, c = cell
    return (((r, c), (r, c + 1)), ((r, c + 1), (r + 1, c + 1)),
            ((r + 1, c + 1), (r + 1, c)), ((r + 1, c), (r, c)))[direction]


def _boundary(region, outside):
    edges = {}
    for r, c in np.argwhere(region):
        r, c = int(r), int(c)
        for d, (dr, dc) in enumerate(N4):
            if outside[r + 1 + dr, c + 1 + dc]:
                a, b = _edge((r, c), d)
                if a in edges:
                    raise ValueError("external boundary has a diagonal junction")
                edges[a] = (b, (r, c), d)
    if not edges:
        raise ValueError("empty boundary")
    start = min(edges)
    walk, vertex = [], start
    for _ in range(len(edges)):
        if vertex not in edges:
            raise ValueError("broken boundary")
        nxt, cell, direction = edges[vertex]
        walk.append((vertex, nxt, cell, direction))
        vertex = nxt
        if vertex == start:
            break
    if vertex != start or len(walk) != len(edges):
        raise ValueError("external boundary must be one simple loop")
    return walk


def _port(anchor, inward, width, depth, color=1):
    # Width always runs along a positive coordinate, independent of travel order.
    dr, dc = inward
    tangent = (0, 1) if dr else (1, 0)
    strip = tuple((anchor[0] + j * tangent[0], anchor[1] + j * tangent[1])
                  for j in range(width))
    return {"strip": strip, "inward": inward, "width": width,
            "depth": depth, "color": color}


def _cells(port, depth=None):
    depth = port["depth"] if depth is None else depth
    dr, dc = port["inward"]
    return [(r + k * dr, c + k * dc) for r, c in port["strip"]
            for k in range(depth + 1)]


def _parse(grid):
    a = np.asarray(grid)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("grid must be a 2D integer array")
    if not all(1 <= n <= 30 for n in a.shape) or np.any((a < 0) | (a > 9)):
        raise ValueError("grid outside ARC bounds")
    st = int(a[0, 0])
    if not (np.all(a[0] == st) and np.all(a[-1] == st)
            and np.all(a[:, 0] == st) and np.all(a[:, -1] == st)):
        raise ValueError("outermost frame must be uniform wall")
    values, counts = np.unique(a, return_counts=True)
    nonwall = [(int(n), int(v)) for v, n in zip(values, counts) if v != st]
    if len(nonwall) < 3:
        raise ValueError("missing background, marker or bars")
    bg = max(nonwall)[1]
    if sum(n == max(nonwall)[0] for n, _ in nonwall) != 1:
        raise ValueError("background is not uniquely dominant among nonwalls")
    singletons = [v for n, v in nonwall if n == 1 and v != bg]
    if len(singletons) != 1:
        raise ValueError("marker must be the only singleton color")
    marker = singletons[0]
    marker_cell = tuple(int(v) for v in np.argwhere(a == marker)[0])
    skeleton = a.copy()
    skeleton[marker_cell] = bg
    ports = []
    h, w = a.shape
    for color in values:
        color = int(color)
        if color in (st, bg, marker):
            continue
        for component in _components(a == color):
            rs, cs = zip(*component)
            r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
            if len(component) != (r1 - r0 + 1) * (c1 - c0 + 1):
                raise ValueError("bars must be solid rectangles")
            choices = []
            for d, (dr, dc) in enumerate(N4):
                if d == 0:
                    strip = [(r0, c) for c in range(c0, c1 + 1)]
                    depth, width = r1 - r0, c1 - c0 + 1
                elif d == 1:
                    strip = [(r, c1) for r in range(r0, r1 + 1)]
                    depth, width = c1 - c0, r1 - r0 + 1
                elif d == 2:
                    strip = [(r1, c) for c in range(c0, c1 + 1)]
                    depth, width = r1 - r0, c1 - c0 + 1
                else:
                    strip = [(r, c0) for r in range(r0, r1 + 1)]
                    depth, width = c1 - c0, r1 - r0 + 1
                if depth >= 2 and all(0 <= r + dr < h and 0 <= c + dc < w
                                     and a[r + dr, c + dc] == st for r, c in strip):
                    choices.append({"strip": tuple(strip), "inward": (-dr, -dc),
                                    "width": width, "depth": depth, "color": color})
            if len(choices) != 1:
                raise ValueError("socket must have exactly one fully backed end")
            p = choices[0]
            for cell in component:
                skeleton[cell] = bg
            for cell in p["strip"]:
                skeleton[cell] = st
            ports.append(p)
    region = _flood(skeleton == bg, marker_cell)
    exterior = _external(region)
    walk = _boundary(region, exterior)
    origins = [i for i, e in enumerate(walk) if e[2] == marker_cell]
    if len(origins) != 1:
        raise ValueError("marker must be on a flat external boundary segment")
    origin = origins[0]
    edge_index = {(e[2], e[3]): i for i, e in enumerate(walk)}
    active, islands, excluded = [], [], []
    for p in ports:
        dr, dc = p["inward"]
        first = [(r + dr, c + dc) for r, c in p["strip"]]
        if all(region[x] for x in first):
            if all(exterior[r + 1, c + 1] for r, c in p["strip"]):
                d = N4.index((-dr, -dc))
                indices = [edge_index[(x, d)] for x in first]
                p["rank"] = min((i - origin) % len(walk) for i in indices)
                active.append(p)
            elif all(not exterior[r + 1, c + 1] for r, c in p["strip"]):
                islands.append(p)
            else:
                raise ValueError("socket straddles a topological boundary")
        elif all(not region[x] for x in first):
            excluded.append(p)
        else:
            raise ValueError("socket straddles compartments")
    active.sort(key=lambda p: p["rank"])
    if len(active) < 2:
        raise ValueError("need at least two external sockets")
    max_depth = max(p["depth"] for p in active)
    occupied = np.zeros(a.shape, bool)
    for p in active:
        dr, dc = p["inward"]
        for r, c in p["strip"]:
            for k in range(1, max_depth + 1):
                x, y = r + k * dr, c + k * dc
                if not (0 <= x < h and 0 <= y < w and region[x, y]):
                    raise ValueError("a transferable depth does not fit")
        mask = np.zeros(a.shape, bool)
        for cell in _cells(p, max_depth):
            mask[cell] = True
        if np.any(mask & occupied) or mask[marker_cell]:
            raise ValueError("transferred bars can collide")
        occupied |= mask
    for p in islands:
        if any(occupied[x] for x in _cells(p)):
            raise ValueError("external bars can hit an island bar")
    rim = _halo(exterior)[1:-1, 1:-1] & region
    return {"grid": a, "wall": st, "background": bg, "marker": marker,
            "marker_cell": marker_cell, "skeleton": skeleton, "region": region,
            "exterior": exterior, "walk": walk, "rim": rim,
            "active": active, "islands": islands, "excluded": excluded}


def _render(info, depths):
    out = info["skeleton"].copy()
    out[info["rim"]] = info["marker"]
    for p in info["islands"]:
        for cell in _cells(p):
            out[cell] = p["color"]
    for p, depth in zip(info["active"], depths):
        for cell in _cells(p, depth):
            out[cell] = p["color"]
    out[info["marker_cell"]] = info["marker"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all roles, topology, cyclic origin and depths from input alone."""
    info = _parse(input_grid)
    return _render(info, sorted(p["depth"] for p in info["active"]))


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _candidate_ports(skeleton, region, cap):
    """Enumerate sockets that fit all depths up to cap, on straight wall faces."""
    h, w = skeleton.shape
    # Check every possible first interior cell in parallel. The four-dimensional
    # mask retains the old row/column/direction/width enumeration order, so this
    # optimization does not change candidate ordering or seeded RNG choices.
    padding = cap + 3
    room_pad = np.pad(region, padding)
    wall_pad = np.pad(skeleton, padding)
    valid = np.zeros((h, w, 4, 3), bool)
    for direction, (dr, dc) in enumerate(N4):
        tr, tc = (0, 1) if dr else (1, 0)
        fits = np.ones((h, w), bool)
        for j in range(3):
            for k in range(cap):
                r = padding + k * dr + j * tr
                c = padding + k * dc + j * tc
                fits &= room_pad[r:r + h, c:c + w]
            for k in (1, 2):
                r = padding - k * dr + j * tr
                c = padding - k * dc + j * tc
                fits &= wall_pad[r:r + h, c:c + w]
            valid[:, :, direction, j] = fits
    result = []
    for r, c, direction, width_index in np.argwhere(valid).tolist():
        dr, dc = N4[direction]
        ar, ac = r - dr, c - dc
        width = width_index + 1
        p = _port((ar, ac), (dr, dc), width, cap)
        mask = np.zeros(skeleton.shape, bool)
        if dr:
            mask[min(ar, ar + cap * dr):max(ar, ar + cap * dr) + 1, ac:ac + width] = True
        else:
            mask[ar:ar + width, min(ac, ac + cap * dc):max(ac, ac + cap * dc) + 1] = True
        p["mask"] = mask
        result.append(p)
    return result


def _select_ports(rng, candidates, count, reserved):
    selected = []
    busy = reserved.copy()
    blocked = _halo(busy)
    order = rng.permutation(len(candidates))
    for i in order:
        p = candidates[int(i)]
        # Chebyshev contact is symmetric. Dilate busy only after an acceptance,
        # instead of dilating every rejected candidate footprint.
        if not np.any(p["mask"] & blocked):
            selected.append(p)
            busy |= p["mask"]
            if len(selected) == count:
                return selected, busy
            blocked = _halo(busy)
    return None, busy


def _floorplan(rng, rooms, witness):
    # Carve chambers out of a solid wall field. Two-cell frames make every
    # colored anchor backed and keep the outermost border untouched.
    h = int(rng.integers(23 if rooms > 1 else 17, 31))
    w = int(rng.integers(25 if rooms > 1 else 19, 31))
    wall = np.ones((h, w), bool)
    if rooms == 1:
        rectangles = [(2, h - 2, 2, w - 2)]
    else:
        cut = int(rng.integers(11, w - 11 + 1))
        rectangles = [(2, h - 2, 2, cut), (2, h - 2, cut + 2, w - 2)]
        if rooms == 3:
            side = int(rng.integers(2))
            r0, r1, c0, c1 = rectangles.pop(side)
            split = int(rng.integers(10, h - 10 + 1))
            rectangles.extend([(r0, split, c0, c1), (split + 2, r1, c0, c1)])
    for r0, r1, c0, c1 in rectangles:
        wall[r0:r1, c0:c1] = False
    # Intrusions introduce extra turns; length and location are independent.
    for r0, r1, c0, c1 in rectangles:
        for _ in range(int(rng.integers(0, 4))):
            side = int(rng.integers(4))
            depth = int(rng.integers(1, 3))
            if side in (0, 2):
                x = int(rng.integers(c0 + 2, max(c0 + 3, c1 - 2)))
                end = min(c1 - 1, x + int(rng.integers(1, 4)))
                lo = r0 if side == 0 else r1 - depth
                wall[lo:lo + depth, x:end] = True
            else:
                x = int(rng.integers(r0 + 2, max(r0 + 3, r1 - 2)))
                end = min(r1 - 1, x + int(rng.integers(1, 5)))
                lo = c0 if side == 3 else c1 - depth
                wall[x:end, lo:lo + depth] = True
    regions = _components(~wall)
    if len(regions) != rooms:
        return None
    areas = [len(c) for c in regions]
    if witness == 1:
        chosen = int(np.argmin(areas))
    else:
        chosen = int(rng.integers(len(regions)))
    region = np.zeros(wall.shape, bool)
    for cell in regions[chosen]:
        region[cell] = True
    # An island is optional except in the first witness. It is placed with a
    # broad free halo, making its hole status unambiguous.
    island_count = 1 if witness == 1 else int(rng.integers(0, 3))
    for _ in range(island_count):
        ih, iw = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        windows = np.lib.stride_tricks.sliding_window_view(region, (ih + 4, iw + 4))
        fits = windows.all(axis=(-2, -1))
        # Same bounds and row-major order as the original explicit scan.
        places = (np.argwhere(fits[2:-2, 2:-2]) + 4).tolist()
        if places:
            r, c = places[int(rng.integers(len(places)))]
            wall[r:r + ih, c:c + iw] = True
            region[r:r + ih, c:c + iw] = False
    if bool(rng.integers(2)):
        wall, region = wall.T.copy(), region.T.copy()
    if bool(rng.integers(2)):
        wall, region = np.flipud(wall).copy(), np.flipud(region).copy()
    if bool(rng.integers(2)):
        wall, region = np.fliplr(wall).copy(), np.fliplr(region).copy()
    return wall, region


def _sample(rng, palette, count, cap, witness, rooms, require_sort=False, allowed_normals=None):
    floor = _floorplan(rng, rooms, witness)
    if floor is None:
        return None
    wall, region = floor
    exterior = _external(region)
    try:
        walk = _boundary(region, exterior)
    except ValueError:
        return None
    candidates = _candidate_ports(wall, region, cap)
    external_candidates = [p for p in candidates if all(exterior[r + 1, c + 1] for r, c in p["strip"])]
    if allowed_normals is not None:
        external_candidates = [p for p in external_candidates if p["inward"] in allowed_normals]
    island_candidates = [p for p in _candidate_ports(wall, region, 2)
                         if all(not exterior[r + 1, c + 1] for r, c in p["strip"])]
    reserved = np.zeros(wall.shape, bool)
    island_ports = []
    if island_candidates and (witness == 1 or rng.random() < .7):
        p = island_candidates[int(rng.integers(len(island_candidates)))]
        island_ports.append(p)
        reserved |= p["mask"]
    elif witness == 1:
        return None
    active, busy = _select_ports(rng, external_candidates, count, reserved)
    if active is None or len({p["inward"] for p in active}) < 2:
        return None
    # Pick the origin after reserving all possible transferred bar cells.
    edge_counts = {}
    for e in walk:
        edge_counts[e[2]] = edge_counts.get(e[2], 0) + 1
    busy_halo = _halo(busy)
    starts = [e[2] for e in walk if edge_counts[e[2]] == 1 and not busy_halo[e[2]]]
    if not starts:
        return None
    marker_cell = starts[int(rng.integers(len(starts)))]
    other_ports = []
    other_region = ~wall & ~region
    for cells in _components(other_region):
        mask = np.zeros(wall.shape, bool)
        for x in cells:
            mask[x] = True
        choices = _candidate_ports(wall, mask, 2)
        picked, busy = _select_ports(rng, choices, int(rng.integers(1, 3)), busy)
        if picked:
            other_ports.extend(picked)
    if witness == 1 and not other_ports:
        return None
    bg, st, marker = palette[:3]
    accents = palette[3:]
    grid = np.where(wall, st, bg).astype(np.int8)
    fixed_ids = {id(p) for p in island_ports + other_ports}
    for p in island_ports + active + other_ports:
        p["color"] = int(rng.choice(accents))
        p["depth"] = 2 if id(p) in fixed_ids else int(rng.integers(2, cap + 1))
        for cell in _cells(p):
            grid[cell] = p["color"]
    grid[marker_cell] = marker
    try:
        info = _parse(grid)
    except (ValueError, KeyError):
        return None
    # Enforce explanatory evidence using visible arrays, not sampled labels.
    depths = [p["depth"] for p in info["active"]]
    if require_sort and (len(set(depths)) < 3 or depths == sorted(depths)):
        return None
    if witness:
        if len(set(depths)) < 3 or depths == sorted(depths) or depths[::-1] == sorted(depths):
            return None
        if witness == 1:
            if [p["depth"] for p in sorted(info["active"], key=lambda p: (p["depth"] + 1) * p["width"])] == sorted(depths):
                return None
    output = _render(info, sorted(depths))
    pair = {"input": grid, "output": output}
    if witness == 1:
        features = _features(pair)
        if not all(features[k] for k in ("origin_wrong", "row_wrong", "column_wrong", "small",
                                        "area_strict", "diagonal", "hole_evidence")):
            return None
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.permutation(10)]
    stable_palette = bool(rng.integers(2))
    cap = int(rng.integers(4, 7))
    puzzle, seen, train_normals = {"train": [], "test": []}, set(), set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            witness = index + 1 if split == "train" and index < 2 else 0
            count = 3 if witness == 1 else 4 if witness == 2 else int(rng.integers(3, 5) if split == "train" else rng.integers(5, 7))
            for attempt in range(500):
                colors = palette if stable_palette else [int(x) for x in rng.permutation(10)]
                rooms = 2 if witness == 1 else int(rng.integers(1, 4))
                pair = _sample(rng, colors, count, cap, witness, rooms,
                               require_sort=split == "test",
                               allowed_normals=train_normals if split == "test" else None)
                if pair is None:
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == "train":
                        train_normals.update(p["inward"] for p in _parse(pair["input"])["active"])
                    break
            else:
                raise RuntimeError("bounded socket placement exhausted")
    rng.shuffle(puzzle["train"])
    return puzzle


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(pair):
    info = _parse(pair["input"])
    output = np.asarray(pair["output"])
    active = info["active"]
    depths = [p["depth"] for p in active]
    areas = [p["depth"] for p in sorted(active, key=lambda p: (p["depth"] + 1) * p["width"])]
    # Reverse the socket order while retaining the marker cut.
    backwards = _render(info, list(reversed(sorted(depths))))
    rowwise = sorted(active, key=lambda p: min(p["strip"]))
    row_depth = {id(p): d for p, d in zip(rowwise, sorted(depths))}
    row_output = _render(info, [row_depth[id(p)] for p in active])
    columnwise = sorted(active, key=lambda p: min((c, r) for r, c in p["strip"]))
    column_depth = {id(p): d for p, d in zip(columnwise, sorted(depths))}
    column_output = _render(info, [column_depth[id(p)] for p in active])
    absolute = sorted(active, key=lambda p: (p["rank"] + next(i for i, e in enumerate(info["walk"]) if e[2] == info["marker_cell"])) % len(info["walk"]))
    abs_depth = {id(p): d for p, d in zip(absolute, sorted(depths))}
    sizes = [len(c) for c in _components(info["skeleton"] == info["background"])]
    orth = np.zeros(info["grid"].shape, bool)
    h, w = orth.shape
    for dr, dc in N4:
        orth |= info["exterior"][1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    diagonal = info["rim"] & ~orth & (info["grid"] == info["background"])
    island_wall = (info["skeleton"] == info["wall"]) & ~info["exterior"][1:-1, 1:-1]
    hole_neighbors = _halo(island_wall) & info["region"] & ~info["rim"]
    return {"shape": info["grid"].shape, "count": len(active),
            "depths": tuple(depths), "widths": tuple(p["width"] for p in active),
            "normals": len({p["inward"] for p in active}), "rooms": len(sizes),
            "small": int(info["region"].sum()) < max(sizes),
            "islands": len(info["islands"]), "excluded": len(info["excluded"]),
            "roles": (info["background"], info["wall"], info["marker"]),
            "area_wrong": areas != sorted(depths),
            "area_strict": any(p["depth"] < q["depth"] and (p["depth"] + 1) * p["width"] > (q["depth"] + 1) * q["width"] for p in active for q in active),
            "diagonal": int(np.count_nonzero(diagonal & (output == info["marker"]))),
            "hole_evidence": bool(np.any(hole_neighbors & (info["grid"] == info["background"]) & (output == info["background"]))),
            "reverse_wrong": depths[::-1] != sorted(depths),
            "ccw_wrong": not np.array_equal(backwards, pair["output"]),
            "row_wrong": not np.array_equal(row_output, pair["output"]),
            "column_wrong": not np.array_equal(column_output, pair["output"]),
            "origin_wrong": not np.array_equal(_render(info, [abs_depth[id(p)] for p in active]), pair["output"]),
            "changed_depths": depths != sorted(depths)}


def validate_generated_puzzle(puzzle):
    """Check topology, conservation, fixed sockets, ignored islands and evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, train_features = set(), []
    normal_sets = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = np.asarray(pair["input"]), np.asarray(pair["output"])
            for a in (inp, out):
                assert a.ndim == 2 and np.issubdtype(a.dtype, np.integer)
                assert all(1 <= n <= 30 for n in a.shape) and np.all((0 <= a) & (a <= 9))
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            normal_sets[split].update(p["inward"] for p in info["active"])
            assert np.array_equal(out, solve(inp.copy()))
            # Read destination depths directly, without sorting/painter helpers.
            assigned = []
            for p in info["active"]:
                depths = []
                dr, dc = p["inward"]
                for r, c in p["strip"]:
                    assert out[r, c] == p["color"]
                    k = 1
                    while (0 <= r + k * dr < out.shape[0] and 0 <= c + k * dc < out.shape[1]
                           and out[r + k * dr, c + k * dc] == p["color"]):
                        k += 1
                    depths.append(k - 1)
                assert len(set(depths)) == 1
                assigned.append(depths[0])
            assert all(a <= b for a, b in zip(assigned, assigned[1:]))
            remaining = [p["depth"] for p in info["active"]]
            for d in assigned:
                assert d in remaining
                remaining.remove(d)
            assert not remaining
            for p in info["islands"]:
                assert all(out[x] == inp[x] for x in _cells(p))
            for p in info["excluded"]:
                assert all(out[x] == info["wall"] for x in p["strip"])
                assert all(out[x] == info["background"] for x in _cells(p) if x not in p["strip"])
            assert out[info["marker_cell"]] == info["marker"]
            if split == "train":
                train_features.append(_features(pair))
    for feature in ("area_wrong", "area_strict", "reverse_wrong", "ccw_wrong", "row_wrong",
                    "column_wrong", "origin_wrong", "changed_depths", "small", "diagonal", "hole_evidence"):
        assert any(f[feature] for f in train_features), f"missing training witness: {feature}"
    assert any(f["islands"] and f["excluded"] for f in train_features)
    assert len({f["count"] for f in train_features}) >= 2
    assert normal_sets["test"] <= normal_sets["train"], "test socket normal lacks demonstration"


def _semantic_regressions():
    """A hand-built scene with a hand-painted expected answer, without sorting."""
    wall = np.ones((20, 28), bool)
    wall[2:18, 2:16] = False
    wall[2:18, 18:26] = False
    wall[5:7, 14:16] = True  # External concavity, including diagonal-only rim.
    wall[7:10, 6:9] = True   # An enclosed island, absent from the outer walk.
    grid = np.where(wall, 5, 0).astype(np.int8)
    ports = [_port((1, 7), (1, 0), 2, 4, 1),
             _port((9, 16), (0, -1), 1, 2, 3),
             _port((18, 10), (-1, 0), 3, 3, 4),
             _port((8, 8), (0, 1), 1, 2, 6),
             _port((1, 21), (1, 0), 1, 2, 7),
             _port((14, 26), (0, -1), 1, 2, 8)]
    for p in ports:
        for cell in _cells(p):
            grid[cell] = p["color"]
    grid[2, 3] = 2
    expected = np.where(wall, 5, 0).astype(np.int8)
    # Rectangular room rim plus the concavity's corner-inclusive halo.
    expected[2, 2:16] = expected[17, 2:16] = 2
    expected[2:18, 2] = expected[2:18, 15] = 2
    for r in range(4, 8):
        for c in range(13, 16):
            if not wall[r, c]:
                expected[r, c] = 2
    expected[wall] = 5
    # The clockwise sequence is top, east, south: 4,2,3 becomes 2,3,4.
    expected[1:4, 7:9] = 1
    expected[9, 13:17] = 3
    expected[14:19, 10:13] = 4
    expected[8, 8:11] = 6
    predicted = solve(grid)
    assert np.array_equal(predicted, expected)
    assert predicted[4, 13] == 2   # Touches the concavity only diagonally.
    assert predicted[6, 7] == 0    # Adjacent to island, no external rim.
    info = _parse(grid)
    assert len(info["active"]) == 3 and len(info["islands"]) == 1 and len(info["excluded"]) == 2
    assert not np.array_equal(_render(info, [p["depth"] for p in info["active"]]), expected)
    # Same room, different cut: the inventory is constant, assignments rotate.
    moved = grid.copy()
    moved[2, 3], moved[3, 15] = 0, 2
    changed = solve(moved)
    assert changed[5, 7] == 1 and predicted[5, 7] == 0
    assert not np.array_equal(changed, predicted)
    # Across a wall: previously erased sockets now survive; old ones disappear.
    across = grid.copy()
    across[2, 3], across[2, 19] = 0, 2
    across_out = solve(across)
    assert across_out[2, 21] == 7 and across_out[14, 25] == 8
    assert across_out[2, 7] == 0 and across_out[8, 9] == 0
    # Attach the island to the exterior. Its socket now joins the carousel.
    attached = grid.copy()
    attached[8, 2:6] = 5
    attached_info = _parse(attached)
    assert len(attached_info["active"]) == 4 and not attached_info["islands"]
    attached_out = solve(attached)
    assert attached_out[8, 11] == 6 and predicted[8, 11] == 0


def self_test():
    _semantic_regressions()
    puzzles, features = [], []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                features.append(_features(pair))
                features[-1]["split"] = split
                copied = pair["input"].copy()
                solve(copied)
                assert np.array_equal(copied, pair["input"])
    assert len(set(puzzles)) == 42
    assert len({f["shape"] for f in features}) >= 30
    assert {f["count"] for f in features} >= {3, 4, 5, 6}
    assert {f["rooms"] for f in features} >= {1, 2, 3}
    assert {w for f in features for w in f["widths"]} == {1, 2, 3}
    assert max(f["normals"] for f in features) == 4
    assert any(len(set(f["depths"])) < f["count"] for f in features)
    assert all({f["roles"][i] for f in features} == set(range(10)) for i in range(3))
    assert any(f["islands"] == 0 for f in features)
    assert any(f["excluded"] == 0 for f in features)
    a, b = generate_puzzle(90210, 2, 1), generate_puzzle(90210, 2, 1)
    assert all(_key(x[role]) == _key(y[role]) for s in a for x, y in zip(a[s], b[s]) for role in x)
    validate_generated_puzzle(generate_puzzle(90211, 10, 4))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    # A meaningful mutation must fail even though it obeys ARC schema.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    info = _parse(bad["test"][0]["input"])
    p = info["active"][0]
    r, c = p["strip"][0]
    dr, dc = p["inward"]
    bad["test"][0]["output"][r + dr, c + dc] = info["background"]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt socket accepted")
    return {"seeds_checked": 42, "distinct_puzzles": len(set(puzzles)),
            "pairs_checked": len(features), "distinct_shapes": len({f["shape"] for f in features}),
            "active_counts": sorted({f["count"] for f in features}),
            "room_counts": sorted({f["rooms"] for f in features}),
            "widths": sorted({w for f in features for w in f["widths"]}),
            "normal_counts": sorted({f["normals"] for f in features}),
            "depths": sorted({d for f in features for d in f["depths"]}),
            "island_bar_counts": sorted({f["islands"] for f in features}),
            "erased_bar_counts": sorted({f["excluded"] for f in features}),
            "test_room_counts": sorted({f["rooms"] for f in features if f["split"] == "test"}),
            "test_island_bar_counts": sorted({f["islands"] for f in features if f["split"] == "test"}),
            "test_normal_counts": sorted({f["normals"] for f in features if f["split"] == "test"}),
            "diagonal_witness_pairs": sum(bool(f["diagonal"]) for f in features),
            "semantic_regressions": "passed", "independent_verification": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
