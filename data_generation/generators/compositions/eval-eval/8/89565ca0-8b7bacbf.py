"""Successor relays: compartment counts gate endpoint paint propagation.

Only standard-library modules and NumPy are used. The decoder never receives
generation metadata. Clean partitions and intended edges construct labels;
pixel-derived reconstruction checks them before an example is returned.
"""

from collections import deque
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _neighbors(cells, shape, steps=D8):
    h, w = shape
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - set(cells)


def _components(cells, steps=D8):
    left = set(cells)
    result = []
    while left:
        root = min(left)
        left.remove(root)
        group, queue = {root}, [root]
        for r, c in queue:
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    group.add(p)
                    queue.append(p)
        result.append(group)
    return result


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _rooms(mask):
    """Bounded four-connected complement regions; pad to model the exterior."""
    h, w = mask.shape
    free = {(r, c) for r in range(-1, h + 1) for c in range(-1, w + 1)
            if not (0 <= r < h and 0 <= c < w and mask[r, c])}
    return [g for g in _components(free, D4)
            if all(0 <= r < h and 0 <= c < w for r, c in g)]


def _walls(shape, boxes):
    a = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        a[t, l:r + 1] = a[b, l:r + 1] = True
        a[t:b + 1, l] = a[t:b + 1, r] = True
    return a


def _euler_rooms(mask):
    vertices = int(mask.sum())
    edges = int((mask[:, :-1] & mask[:, 1:]).sum())
    edges += int((mask[:-1] & mask[1:]).sum())
    solid_faces = int((mask[:-1, :-1] & mask[1:, :-1]
                       & mask[:-1, 1:] & mask[1:, 1:]).sum())
    return edges - vertices + 1 - solid_faces


def _recover_station(grid, color, background):
    visible = grid == color
    sites = np.argwhere(visible)
    if len(sites) < 8:
        return None
    t, l = map(int, sites.min(axis=0))
    b, r = map(int, sites.max(axis=0))
    if min(b - t, r - l) < 2:
        return None
    mask = visible.copy()
    # One-cell non-background interruptions have opposite visible supports.
    for y in range(t, b + 1):
        for x in range(l, r + 1):
            if grid[y, x] == background or visible[y, x]:
                continue
            horizontal = x > l and x < r and visible[y, x - 1] and visible[y, x + 1]
            vertical = y > t and y < b and visible[y - 1, x] and visible[y + 1, x]
            if horizontal or vertical:
                mask[y, x] = True
    local = mask[t:b + 1, l:r + 1]
    if not (local[0].all() and local[-1].all()
            and local[:, 0].all() and local[:, -1].all()):
        return None
    rooms = [{(y + t, x + l) for y, x in cells} for cells in _rooms(local)]
    if not rooms:
        return None
    boxes = []
    for cells in rooms:
        yy, xx = zip(*cells)
        a, c, d, e = min(yy), min(xx), max(yy), max(xx)
        if len(cells) != (d - a + 1) * (e - c + 1):
            return None
        boxes.append((a - 1, c - 1, d + 1, e + 1))
    if not np.array_equal(mask, _walls(grid.shape, boxes)):
        return None
    return dict(color=int(color), count=len(rooms), mask=mask, rooms=rooms,
                box=(t, l, b, r))


def _cup_offsets():
    shapes = []
    cells = {(0, -1), (0, 1), (1, -1), (1, 0), (1, 1)}
    for _ in range(4):
        shapes.append(cells)
        cells = {(-c, r) for r, c in cells}
    return shapes


def _propagate(counts, edges, sources):
    """Sources are (station index, paint); directed edges require exactly +1."""
    adjacency = [set() for _ in counts]
    for i, j in edges:
        adjacency[i].add(j)
        adjacency[j].add(i)
    paints = {}
    for root, paint in sources:
        seen, queue = {root}, [root]
        for i in queue:
            assert i not in paints or paints[i] == paint, "Conflicting supplies"
            paints[i] = paint
            for j in sorted(adjacency[i]):
                if j not in seen and counts[j] == counts[i] + 1:
                    seen.add(j)
                    queue.append(j)
    return paints


def _analyze(input_grid):
    a = np.asarray(input_grid)
    if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Expected an integer two-dimensional grid")
    if not all(1 <= n <= 30 for n in a.shape) or np.any((a < 0) | (a > 9)):
        raise ValueError("Grid outside ARC bounds")
    values, freq = np.unique(a, return_counts=True)
    order = np.argsort(freq)
    assert len(values) >= 5 and freq[order[-1]] > freq[order[-2]]
    bg = int(values[order[-1]])
    sources = []
    for r, c in np.argwhere(a != bg):
        r, c = int(r), int(c)
        neighbors = {p for p in _neighbors({(r, c)}, a.shape) if a[p] != bg}
        offsets = {(y - r, x - c) for y, x in neighbors}
        if offsets not in _cup_offsets():
            continue
        colors = {int(a[p]) for p in neighbors}
        if len(colors) == 1 and int(a[r, c]) not in colors:
            sources.append(((r, c), int(a[r, c]), colors.pop()))
    assert sources, "No visible source cup"
    wire_colors = {w for _, _, w in sources}
    assert len(wire_colors) == 1, "Family uses one common wire role"
    wire = wire_colors.pop()
    paint_colors = {p for _, p, _ in sources}
    assert bg not in paint_colors and wire not in paint_colors
    stations, leftovers = [], []
    for color in values:
        color = int(color)
        if color in {bg, wire} | paint_colors:
            continue
        s = _recover_station(a, color, bg)
        if s is None:
            leftovers.append(color)
        else:
            stations.append(s)
    assert stations and len(leftovers) == 1, "Walls/noise roles not unique"
    noise = leftovers[0]
    for s in stations:
        assert np.all(a[s["mask"] & (a != s["color"])] == noise)
        assert _euler_rooms(s["mask"]) == s["count"]
        # Frames contain only their walls/noise and background rooms.
        t, l, b, r = s["box"]
        contents = a[t:b + 1, l:r + 1]
        assert set(map(int, np.unique(contents))) <= {bg, noise, s["color"]}
        assert all(all(a[p] == bg for p in room) for room in s["rooms"])
    wall_cells = [set(map(tuple, np.argwhere(s["mask"]))) for s in stations]
    ports = [_neighbors(cells, a.shape) for cells in wall_cells]
    components = _components(set(map(tuple, np.argwhere(a == wire))))
    edges, seed_stations, incidences = set(), [], []
    for cells in components:
        attached = [i for i, near in enumerate(ports) if cells & near]
        local_sources = [(point, paint) for point, paint, _ in sources
                         if cells & _neighbors({point}, a.shape)]
        assert len(attached) <= 2 and len(local_sources) <= 1
        if local_sources:
            assert len(attached) == 1, "A source lead must reach one first station"
            seed_stations.append((attached[0], local_sources[0][1]))
        elif len(attached) == 2:
            edges.add(tuple(attached))
        else:
            assert len(attached) == 1, "Unmarked wire is a link or a pendant tail"
        incidences.append((cells, attached, local_sources))
    assert len(seed_stations) == len(sources)
    counts = [s["count"] for s in stations]
    paints = _propagate(counts, edges, seed_stations)
    return dict(bg=bg, wire=wire, noise=noise, stations=stations, edges=edges,
                sources=seed_stations, cups=sources, paints=paints, wires=incidences)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compute the complete answer using only the visible input."""
    a = np.asarray(input_grid)
    info = _analyze(a)
    out = a.copy()
    for i, color in info["paints"].items():
        for room in info["stations"][i]["rooms"]:
            for p in room:
                if a[p] == info["bg"]:
                    out[p] = color
    return out


def _partition(rng, height, width, count, style):
    outer = (0, 0, height - 1, width - 1)
    boxes = [outer]
    first_axis = None
    for step in range(count - 1):
        choices = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, end in ((0, t, b), (1, l, r)):
                if style == "strip" and first_axis is not None and axis != first_axis:
                    continue
                if style == "partial" and step == 1 and axis == first_axis:
                    continue
                for cut in range(start + 2, end - 1):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    future = boxes[:i] + list(halves) + boxes[i + 1:]
                    capacity = sum(((d - a) // 2) * ((e - c) // 2) for a, c, d, e in future)
                    if capacity >= count:
                        choices.append((i, axis, halves))
        if not choices:
            return None
        i, axis, halves = choices[int(rng.integers(len(choices)))]
        if step == 0:
            first_axis = axis
        boxes[i:i + 1] = halves
    return _walls((height, width), boxes)


def _is_partial(mask):
    ys = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
    xs = np.flatnonzero(np.any(mask[:-1] & mask[1:], axis=0))
    return any(not mask[y].all() for y in ys) or any(not mask[:, x].all() for x in xs)


def _topology_key(station):
    """Remove palette, position, and numerical spacing from room boundaries."""
    boxes = []
    for room in station["rooms"]:
        yy, xx = zip(*room)
        boxes.append((min(yy) - 1, min(xx) - 1, max(yy) + 1, max(xx) + 1))
    ys = sorted({y for t, _, b, _ in boxes for y in (t, b)})
    xs = sorted({x for _, l, _, r in boxes for x in (l, r)})
    ymap, xmap = {v: i for i, v in enumerate(ys)}, {v: i for i, v in enumerate(xs)}
    return tuple(sorted((ymap[t], xmap[l], ymap[b], xmap[r]) for t, l, b, r in boxes))


def _place_stations(rng, shape, counts, wall_colors, force_partial, fallback=False):
    h, w = shape
    stations, occupied = [], set()
    for i, (count, color) in enumerate(zip(counts, wall_colors)):
        placed = False
        for _ in range(110):
            if fallback:
                bh, bw = 7, 7
                slot = i
                top, left = 2 + (slot // 3) * 14, 2 + (slot % 3) * 9
            else:
                bh, bw = int(rng.integers(4, 11)), int(rng.integers(4, 11))
                if rng.random() < .2:
                    bh, bw = (int(rng.integers(3, 6)), int(rng.integers(8, 15)))
                if rng.random() < .5:
                    bh, bw = bw, bh
                if bh >= h - 1 or bw >= w - 1:
                    continue
                top, left = int(rng.integers(1, h - bh)), int(rng.integers(1, w - bw))
            if ((bh - 1) // 2) * ((bw - 1) // 2) < count:
                continue
            style = "partial" if force_partial and count >= 3 else str(rng.choice(["strip", "mixed", "partial"]))
            mask = _partition(rng, bh, bw, count, style)
            if mask is None or (force_partial and count >= 3 and not _is_partial(mask)):
                continue
            box_cells = {(y, x) for y in range(top, top + bh) for x in range(left, left + bw)}
            if box_cells & (occupied | _neighbors(occupied, shape)):
                continue
            cells = {(y + top, x + left) for y, x in np.argwhere(mask)}
            rooms = [{(y + top, x + left) for y, x in room} for room in _rooms(mask)]
            stations.append(dict(color=color, count=count, cells=cells, rooms=rooms,
                                 filled_box=box_cells, local_mask=mask))
            occupied |= box_cells
            placed = True
            break
        if not placed:
            return None
    return stations


def _place_cups(rng, shape, stations, root_paints, fallback=False):
    occupied = set().union(*(s["filled_box"] for s in stations))
    cups = []
    for k, (root, paint) in enumerate(root_paints):
        for _ in range(200):
            source = (0, 1)
            wires = {(0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}
            for _ in range(int(rng.integers(4))):
                source = (-source[1], source[0])
                wires = {(-c, r) for r, c in wires}
            allcells = wires | {source}
            minr, minc = min(r for r, c in allcells), min(c for r, c in allcells)
            allcells = {(r - minr, c - minc) for r, c in allcells}
            source = source[0] - minr, source[1] - minc
            wires = {(r - minr, c - minc) for r, c in wires}
            ch, cw = max(r for r, c in allcells) + 1, max(c for r, c in allcells) + 1
            rr, cc = int(rng.integers(shape[0] - ch + 1)), int(rng.integers(shape[1] - cw + 1))
            source = source[0] + rr, source[1] + cc
            wires = {(r + rr, c + cc) for r, c in wires}
            cells = wires | {source}
            if cells & (occupied | _neighbors(occupied, shape)):
                continue
            cups.append(dict(root=root, paint=paint, point=source, cells=cells, wires=wires))
            occupied |= cells
            break
        else:
            return None
    return cups


def _path(rng, starts, goals, allowed, shape):
    starts = sorted(starts)
    rng.shuffle(starts)
    queue = deque(starts)
    previous = {p: None for p in starts}
    directions = [D8[i] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = previous[end]
    return result


def _route(rng, shape, stations, cups, edges):
    canvas = {(r, c) for r in range(shape[0]) for c in range(shape[1])}
    all_boxes = set().union(*(s["filled_box"] for s in stations))
    all_cups = set().union(*(s["cells"] for s in cups))
    used, links = set(), []
    jobs = [("source", k, cup["root"]) for k, cup in enumerate(cups)]
    shuffled_edges = list(edges)
    rng.shuffle(shuffled_edges)
    jobs += [("edge", i, j) for i, j in shuffled_edges]
    for kind, i, j in jobs:
        targets = {j} if kind == "source" else {i, j}
        other_boxes = set().union(*(s["filled_box"] for n, s in enumerate(stations) if n not in targets)) if len(targets) < len(stations) else set()
        forbidden = all_boxes | _neighbors(other_boxes, shape)
        forbidden |= used | _neighbors(used, shape)
        forbidden |= all_cups | _neighbors(all_cups, shape)
        allowed = canvas - forbidden
        goals = (_neighbors(stations[j]["cells"], shape) - all_boxes) & allowed
        if kind == "source":
            cup = cups[i]
            # Permit leaving the cup, while keeping its five-neighbor signature.
            extension = _neighbors(cup["wires"], shape) - forbidden
            extension |= _neighbors(cup["wires"], shape) - (all_boxes | other_boxes | used | _neighbors(used, shape) | (all_cups - cup["cells"]) | _neighbors(all_cups - cup["cells"], shape))
            extension -= {cup["point"]} | _neighbors({cup["point"]}, shape)
            allowed |= extension | cup["wires"]
            starts = cup["wires"]
        else:
            starts = (_neighbors(stations[i]["cells"], shape) - all_boxes) & allowed
        if not starts or not goals:
            return None
        path = _path(rng, starts, goals, allowed, shape)
        if path is None:
            return None
        if kind == "source":
            path |= cups[i]["wires"]
        used |= path
        links.append(path)
    return used, links


def _damage(rng, grid, stations, bg, noise, required):
    candidates = []
    for s in stations:
        cells = s["cells"]
        for y, x in sorted(cells):
            near = _neighbors({(y, x)}, grid.shape) & cells
            if len(near) == 2 and ((y - 1, x) in near and (y + 1, x) in near
                                  or (y, x - 1) in near and (y, x + 1) in near):
                candidates.append((y, x, s["color"]))
    rng.shuffle(candidates)
    gaps = set()
    limit = int(rng.integers(1 if required else 0, max(2, len(stations) + 2)))
    for y, x, color in candidates:
        if len(gaps) >= limit:
            break
        if (y, x) in _neighbors(gaps, grid.shape):
            continue
        grid[y, x] = noise
        gaps.add((y, x))
    if required and not gaps:
        return False
    # Exterior-only isolated speckles never act as sources or extra walls.
    boxes = set().union(*(s["filled_box"] for s in stations))
    occupied = set(map(tuple, np.argwhere(grid != bg))) | boxes
    exterior = {(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])} - occupied - _neighbors(occupied, grid.shape)
    sites = sorted(exterior)
    rng.shuffle(sites)
    speckles = set()
    for p in sites:
        if p in _neighbors(speckles, grid.shape):
            continue
        grid[p] = noise
        speckles.add(p)
        if len(speckles) >= int(rng.integers(2, 9)):
            break
    return bool(gaps or speckles)


def _crop_rotate(rng, a, b, bg):
    sites = np.argwhere(a != bg)
    lo, hi = sites.min(axis=0), sites.max(axis=0)
    t, l = [max(0, int(x) - int(rng.integers(4))) for x in lo]
    bottom, right = [min(n, int(x) + 1 + int(rng.integers(4))) for x, n in zip(hi, a.shape)]
    a, b = a[t:bottom, l:right], b[t:bottom, l:right]
    rotation = int(rng.integers(4))
    a, b = np.rot90(a, rotation), np.rot90(b, rotation)
    if rng.random() < .5:
        a, b = np.fliplr(a), np.fliplr(b)
    return a.copy(), b.copy()


def _scene(rng, counts, edges, roots, roles, force_partial=False, fallback=False):
    bg, wire, noise, paint_pool = roles
    root_paints = [(root, paint_pool[k]) for k, root in enumerate(roots)]
    reserved = {bg, wire, noise} | {p for _, p in root_paints}
    wall_colors = list(map(int, rng.choice([c for c in range(10) if c not in reserved], size=len(counts), replace=False)))
    n = len(counts)
    shape = (30, 30) if fallback else (int(rng.integers(22 if n <= 4 else 26, 31)), int(rng.integers(22 if n <= 4 else 26, 31)))
    stations = _place_stations(rng, shape, counts, wall_colors, force_partial, fallback)
    if stations is None:
        return None
    cups = _place_cups(rng, shape, stations, root_paints, fallback)
    if cups is None:
        return None
    routing = _route(rng, shape, stations, cups, edges)
    if routing is None:
        return None
    wires, _ = routing
    a = np.full(shape, bg, dtype=np.int8)
    for s in stations:
        for p in s["cells"]:
            a[p] = s["color"]
    for p in wires:
        a[p] = wire
    for cup in cups:
        a[cup["point"]] = cup["paint"]
    if not _damage(rng, a, stations, bg, noise, force_partial):
        return None
    intended_paints = _propagate(counts, edges, root_paints)
    b = a.copy()
    for i, paint in intended_paints.items():
        for room in stations[i]["rooms"]:
            for p in room:
                b[p] = paint
    # Check the entire graph, count and mask, not merely the resulting label.
    try:
        info = _analyze(a)
    except (AssertionError, ValueError):
        return None
    if (info["bg"], info["wire"], info["noise"]) != (bg, wire, noise):
        return None
    index = {s["color"]: i for i, s in enumerate(info["stations"])}
    if set(index) != set(wall_colors):
        return None
    for s in stations:
        recovered = info["stations"][index[s["color"]]]
        if recovered["count"] != s["count"] or set(map(tuple, np.argwhere(recovered["mask"]))) != s["cells"]:
            return None
    expected_edges = {tuple(sorted((index[wall_colors[i]], index[wall_colors[j]]))) for i, j in edges}
    expected_sources = sorted((index[wall_colors[i]], paint) for i, paint in root_paints)
    if info["edges"] != expected_edges or sorted(info["sources"]) != expected_sources:
        return None
    if not np.array_equal(solve(a), b):
        return None
    a, b = _crop_rotate(rng, a, b, bg)
    return {"input": a, "output": b}


def _generic_spec(rng, maximum, dual=None):
    dual = bool(rng.random() < .4) if dual is None else dual
    n = int(rng.integers(4, 6 if dual else 7))
    counts = list(map(int, rng.integers(1, maximum + 1, size=n)))
    split = int(rng.integers(2, n - 1)) if dual else n
    groups = [list(range(split)), list(range(split, n))] if dual else [list(range(n))]
    edges, roots = [], []
    for group in groups:
        roots.append(group[0])
        degree = {i: 0 for i in group}
        for k, j in enumerate(group[1:], 1):
            choices = [i for i in group[:k] if degree[i] < 3]
            i = int(rng.choice(choices))
            edges.append((i, j))
            degree[i] += 1
            degree[j] += 1
        base = int(rng.integers(1, min(4, maximum)))
        counts[group[0]] = base
        counts[group[1]] = base + 1
    # Some scenes have an unsupplied leaf even if its count fits a nearby chain.
    if not dual and n >= 5 and rng.random() < .25:
        leaf = n - 1
        edges = [(i, j) for i, j in edges if i != leaf and j != leaf]
    return counts, edges, roots


def _legal_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} outside {low}..{high}")
    return value


def _make_pair(rng, spec, roles, witness=False):
    counts, edges, roots = spec
    for attempt in range(120):
        pair = _scene(rng, counts, edges, roots, roles, witness, fallback=attempt >= 95)
        if pair is not None and (not witness or _features(pair)["diagonal"]):
            return pair
    raise RuntimeError("Could not place a readable relay graph after bounded attempts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _legal_count(num_train, "num_train", 2, 10)
    ne = _legal_count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    roles = (palette[0], palette[1], palette[2], palette[3:5])
    local_roles = bool(rng.random() < .5)
    base0, base1 = int(rng.integers(1, 3)), int(rng.integers(1, 4))
    equal = int(rng.integers(1, 5))
    lesson0 = ([base0, base0 + 1, base0 + 2, base0 + 3], [(0, 1), (0, 2), (2, 3)], [0])
    lesson1 = ([base1, base1 + 1, base1 + 2, equal, equal], [(0, 1), (1, 2), (3, 4)], [0, 3])
    train_maximum = max(max(lesson0[0]), max(lesson1[0]))
    extrapolate = bool(rng.random() < .5)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, number in (("train", nt), ("test", ne)):
        for i in range(number):
            witness = split == "train" and i < 2
            if witness:
                spec = lesson0 if i == 0 else lesson1
            else:
                maximum = train_maximum + (1 if split == "test" and extrapolate else 0)
                spec = _generic_spec(rng, maximum)
                if split == "test" and extrapolate:
                    # Larger count at a non-source station, not an unseen rule.
                    choices = [j for j in range(len(spec[0])) if j not in spec[2]]
                    spec[0][int(rng.choice(choices))] = maximum
            pair_roles = roles
            if local_roles:
                p = list(map(int, rng.permutation(10)))
                pair_roles = (p[0], p[1], p[2], p[3:5])
            for _ in range(6):
                pair = _make_pair(rng, spec, pair_roles, witness)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Duplicate inputs after bounded retries")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a = np.asarray(pair["input"])
    info = _analyze(a)
    ns = info["stations"]
    counts = [s["count"] for s in ns]
    degree = [0] * len(ns)
    for i, j in info["edges"]:
        degree[i] += 1
        degree[j] += 1
    reached = set(info["paints"])
    jump = equal = blocked_successor = chain = False
    for i, j in info["edges"]:
        equal |= counts[i] == counts[j] and ((i in reached) != (j in reached))
        for start, end in ((i, j), (j, i)):
            jump |= start in reached and end not in reached and counts[end] >= counts[start] + 2
            blocked_successor |= start not in reached and end not in reached and counts[end] == counts[start] + 1
            chain |= start in reached and end in reached and counts[end] == counts[start] + 1 and start not in {k for k, _ in info["sources"]}
    partial = any(_is_partial(s["mask"][s["box"][0]:s["box"][2] + 1, s["box"][1]:s["box"][3] + 1]) for s in ns)
    damaged = any(np.any(s["mask"] & (a == info["noise"])) for s in ns)
    diagonal = any(any((r + 1, c + dc) in cells and (r + 1, c) not in cells and (r, c + dc) not in cells
                       for r, c in cells for dc in (-1, 1)) for cells, _, _ in info["wires"])
    return dict(stations=len(ns), counts=tuple(sorted(counts)), sources=len(info["sources"]),
                reached=len(reached), inactive=len(ns) - len(reached), jump=jump,
                equal=equal, blocked_successor=blocked_successor, chain=chain,
                partial=partial, damaged=damaged, branch=max(degree) >= 3,
                diagonal=diagonal, bg=info["bg"], shape=a.shape)


def _omission_witnesses(info):
    """Check causal contributions against two explicit count/route ablations."""
    counts = [s["count"] for s in info["stations"]]
    real = set(info["paints"])
    physical = _propagate([1] * len(counts), [], info["sources"])
    adjacency = [set() for _ in counts]
    for i, j in info["edges"]:
        adjacency[i].add(j)
        adjacency[j].add(i)
    # Removing A means every physical link transmits, regardless of its count.
    todo = list(physical)
    for i in todo:
        for j in adjacency[i]:
            if j not in physical:
                physical[j] = physical[i]
                todo.append(j)
    without_a = set(physical) != real
    # Removing B means consecutive counts transmit without a visible string.
    complete = [(i, j) for i in range(len(counts)) for j in range(i)
                if abs(counts[i] - counts[j]) == 1]
    global_reached = set()
    for source in info["sources"]:
        global_reached.update(_propagate(counts, complete, [source]))
    without_b = global_reached != real
    return without_a, without_b


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, training, omissions = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = (np.asarray(pair[k]) for k in ("input", "output"))
            for grid in (a, b):
                assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            assert a.shape == b.shape and not np.array_equal(a, b)
            assert _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            assert np.array_equal(solve(a), b), "Incorrect relay activation or cavity paint"
            assert np.all(a[a != b] == info["bg"]), "Only room background may change"
            for i, station in enumerate(info["stations"]):
                paint = info["paints"].get(i, info["bg"])
                for room in station["rooms"]:
                    assert all(b[p] == paint for p in room), "Partial room fill"
            if split == "train":
                training.append(_features(pair))
                omissions.append(_omission_witnesses(info))
    for feature in ("jump", "equal", "blocked_successor", "chain", "partial", "damaged", "diagonal"):
        assert any(f[feature] for f in training), f"Missing training evidence: {feature}"
    assert any(f["sources"] == 2 for f in training), "Teach simultaneous independent paint sources"
    assert all(any(w[k] for w in omissions) for k in (0, 1)), "Both parent omissions must change a training answer"


def self_test():
    """Bounded variation audit, graph regressions, equivariance and count API."""
    puzzles, records, normalized, topologies = [], [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for pair in p["train"] + p["test"]:
            records.append(_features(pair))
            a = pair["input"]
            colors = {}
            normalized.add((a.shape, tuple(colors.setdefault(int(v), len(colors)) for v in a.flat)))
            info = _analyze(a)
            topologies.update(_topology_key(s) for s in info["stations"])
    assert len(normalized) >= 190
    assert len({f["shape"] for f in records}) >= 25
    assert len({f["counts"] for f in records}) >= 45
    assert len(topologies) >= 40
    assert {1, 2} <= {f["sources"] for f in records}
    assert {4, 5, 6} <= {f["stations"] for f in records}
    assert len({f["bg"] for f in records}) >= 8
    assert all(any(f[k] for f in records) for k in ("branch", "diagonal", "partial", "damaged"))
    # Activation stops at a +2 jump; the following +1 edge does not restart it.
    assert _propagate([2, 3, 4, 5], [(0, 1), (0, 2), (2, 3)], [(0, 7)]) == {0: 7, 1: 7}
    # An equal-count link blocks a reachable duplicate and a downstream successor.
    assert _propagate([1, 2, 2, 3], [(0, 1), (1, 2), (2, 3)], [(0, 6)]) == {0: 6, 1: 6}
    # A second, longer increasing route may bypass the blocked shortcut.
    assert _propagate([1, 2, 3], [(0, 1), (1, 2), (0, 2)], [(0, 4)]) == {0: 4, 1: 4, 2: 4}
    again = generate_puzzle(0)
    assert tuple(_key(q[k]) for s in puzzles[0] for q in puzzles[0][s] for k in q) == tuple(_key(q[k]) for s in again for q in again[s] for k in q)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(200 + nt * 10 + ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
    for pair in puzzles[0]["train"] + puzzles[0]["test"]:
        a, b = pair["input"], pair["output"]
        for operation in (lambda g: np.rot90(g), np.fliplr, lambda g: g.T):
            assert np.array_equal(solve(operation(a)), operation(b))
        assert np.array_equal(solve(permutation[a]), permutation[b])
    bad = {s: [{k: a.copy() for k, a in q.items()} for q in puzzles[0][s]] for s in puzzles[0]}
    q = bad["test"][0]
    r, c = np.argwhere(q["input"] != q["output"])[0]
    q["output"][r, c] = q["input"][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect cavity fill was accepted")


if __name__ == "__main__":
    self_test()
    print("Successor-relay self-test passed (42 seeds).")
