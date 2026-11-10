"""Reachable corner stamps: infer a die, then complete only accessible sockets.

Only standard-library modules and NumPy are used. Nothing is read from disk,
and generation, solving and validation do not save execution traces.
"""

from collections import deque
import hashlib
import numpy as np


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _need(bool(np.all((0 <= a) & (a <= 9))), "color bounds")
    return a.astype(np.int8, copy=True)


def _counts(num_train, num_test):
    for name, value, lo, hi in (("num_train", num_train, 2, 10),
                                ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(name + " must be an integer")
        if not lo <= int(value) <= hi:
            raise ValueError(name + " outside its legal range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _rotate(point, q):
    u, v = point
    return ((u, v), (v, -u), (-u, -v), (-v, u))[q % 4]


def _corners(rect):
    t, l, b, r = rect
    return ((t, l, 0), (t, r, 1), (b, r, 2), (b, l, 3))


def _positions(corner, pattern):
    y, x, q = corner
    return {(y + a, x + b) for a, b in (_rotate(p, q) for p in pattern)}


def _connected(points):
    if not points:
        return False
    seen = {min(points)}
    queue = list(seen)
    for u, v in queue:
        for p in ((u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)):
            if p in points and p not in seen:
                seen.add(p)
                queue.append(p)
    return len(seen) == len(points)


def _rooms(allowed):
    """Maximal open rectangles with both dimensions >=5 are room interiors.

    Passages are at most four cells across, so cannot themselves be rooms.
    This reads geometry through accent marks, which are traversable floor.
    """
    h, w = allowed.shape
    result = []
    for top in range(h - 4):
        common = np.ones(w, dtype=bool)
        for bottom in range(top, h):
            common &= allowed[bottom]
            if bottom - top < 4:
                continue
            changes = np.diff(np.concatenate(([False], common, [False])).astype(np.int8))
            for left, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
                left, end = int(left), int(end)
                if end - left < 5:
                    continue
                if top and allowed[top - 1, left:end].all():
                    continue
                if bottom + 1 < h and allowed[bottom + 1, left:end].all():
                    continue
                result.append((top, left, bottom, end - 1))
    _need(bool(result), "no rooms")
    occupied = np.zeros(allowed.shape, dtype=bool)
    for t, l, b, r in result:
        _need(not occupied[t:b + 1, l:r + 1].any(), "ambiguous overlapping rooms")
        occupied[t:b + 1, l:r + 1] = True
    return sorted(result)


def _read_scene(inp):
    wall = int(inp[0, 0])
    _need(all(np.all(edge == wall) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1])),
          "uniform wall border required")
    allowed = inp != wall
    _need(_connected({tuple(map(int, p)) for p in np.argwhere(allowed)}),
          "floor network must be connected for a point-sized tool")
    rects = _rooms(allowed)
    centers = {int(inp[(t + b) // 2, (l + r) // 2]) for t, l, b, r in rects}
    _need(len(centers) == 1 and wall not in centers, "room centers must expose floor color")
    floor = centers.pop()
    corners = [c for rect in rects for c in _corners(rect)]
    assigned = {c: [] for c in corners}
    for y, x in np.argwhere((inp != wall) & (inp != floor)):
        y, x = int(y), int(x)
        owners = [rect for rect in rects if rect[0] <= y <= rect[2] and rect[1] <= x <= rect[3]]
        _need(len(owners) == 1, "accent outside a room")
        candidates = _corners(owners[0])
        distances = [(y - a) ** 2 + (x - b) ** 2 for a, b, _ in candidates]
        _need(distances.count(min(distances)) == 1, "corner association tie")
        corner = candidates[distances.index(min(distances))]
        assigned[corner].append((y, x))
    occurrences = []
    for corner in corners:
        coords = assigned[corner]
        if not coords:
            continue
        colors = {int(inp[p]) for p in coords}
        _need(len(colors) == 1, "one accent per socket")
        y, x, q = corner
        clue = frozenset(_rotate((a - y, b - x), -q) for a, b in coords)
        _need(all(0 <= u <= 3 and 0 <= v <= 3 for u, v in clue), "inset stencil bounds")
        occurrences.append((corner, colors.pop(), clue))
    _need(len(occurrences) >= 2, "missing reference or request")
    maximal = {p for _, _, p in occurrences if all(other <= p for _, _, other in occurrences)}
    _need(len(maximal) == 1, "no complete common reference")
    pattern = maximal.pop()
    _need(3 <= len(pattern) <= 12 and _connected(pattern), "die must be a connected 3..12-cell shape")
    us, vs = zip(*pattern)
    _need(max(us) - min(us) >= 1 and max(vs) - min(vs) >= 1, "line dies excluded")
    radius = max(max(u, v) for u, v in pattern)
    _need(all(min(b - t + 1, r - l + 1) >= 2 * radius + 3 for t, l, b, r in rects),
          "room too small to separate sockets and provide turning space")
    _need(any(clue != pattern for _, _, clue in occurrences), "no partial request")
    sources = [c for c, _, clue in occurrences if clue == pattern]
    return {"wall": wall, "floor": floor, "allowed": allowed, "rooms": rects,
            "corners": corners, "occurrences": occurrences, "pattern": pattern,
            "radius": radius, "sources": sources}


def _reach(allowed, pattern, sources):
    """Finite configuration space: anchor translation and guarded quarter-turns.

    The anchor may be empty. Include anchors just outside the canvas; only
    actual occupied die cells must stay on floor. Turning requires a complete
    (2*r+1)-square of floor about the anchor, including its center.
    """
    radius = max(max(abs(u), abs(v)) for u, v in pattern)
    h, w = allowed.shape
    ah, aw = h + 2 * radius, w + 2 * radius
    yy, xx = np.indices((ah, aw))
    yy, xx = yy - radius, xx - radius
    valid = np.ones((4, ah, aw), dtype=bool)
    for q in range(4):
        for u, v in pattern:
            dy, dx = _rotate((u, v), q)
            y, x = yy + dy, xx + dx
            inside = (y >= 0) & (y < h) & (x >= 0) & (x < w)
            valid[q] &= inside & allowed[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
    turn = np.zeros((ah, aw), dtype=bool)
    side = 2 * radius + 1
    if h >= side and w >= side:
        windows = np.lib.stride_tricks.sliding_window_view(allowed, (side, side))
        clear = windows.all(axis=(-1, -2))
        turn[2 * radius:2 * radius + clear.shape[0],
             2 * radius:2 * radius + clear.shape[1]] = clear
    seen = np.zeros_like(valid)
    queue = deque()
    for y, x, q in sources:
        y, x = y + radius, x + radius
        _need(0 <= y < ah and 0 <= x < aw and valid[q, y, x], "invalid source footprint")
        if not seen[q, y, x]:
            seen[q, y, x] = True
            queue.append((q, y, x))
    while queue:
        q, y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < ah and 0 <= nx < aw and valid[q, ny, nx] and not seen[q, ny, nx]:
                seen[q, ny, nx] = True
                queue.append((q, ny, nx))
        if turn[y, x]:
            for nq in ((q - 1) % 4, (q + 1) % 4):
                if valid[nq, y, x] and not seen[nq, y, x]:
                    seen[nq, y, x] = True
                    queue.append((nq, y, x))
    return seen, radius


def _is_reached(seen, radius, corner):
    y, x, q = corner
    return bool(seen[q, y + radius, x + radius])


def _analyze(inp):
    info = _read_scene(inp)
    seen, radius = _reach(info["allowed"], info["pattern"], info["sources"])
    out = inp.copy()
    reached, blocked = [], []
    for corner, color, clue in info["occurrences"]:
        if _is_reached(seen, radius, corner):
            for point in _positions(corner, info["pattern"]):
                _need(inp[point] in (info["floor"], color), "completion collision")
                out[point] = color
            if clue != info["pattern"]:
                reached.append(corner)
        else:
            _need(clue != info["pattern"], "full reference cannot be unreachable")
            blocked.append(corner)
    info["reached"] = reached
    info["blocked"] = blocked
    info["reachable_rooms"] = sum(_is_reached(seen, radius, _corners(rect)[0]) for rect in info["rooms"])
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer geometry, die, sources and every completion from this input only."""
    return _analyze(_grid(input_grid))[0]


def _sample_pattern(rng, radius, witness=False):
    domain = {(u, v) for u in range(radius + 1) for v in range(radius + 1)}
    for _ in range(100):
        count = int(rng.integers(3, min(12, len(domain)) + 1))
        points = {(0, 0)} if witness else {sorted(domain)[int(rng.integers(len(domain)))]}
        while len(points) < count:
            frontier = [p for p in sorted(domain - points)
                        if any(abs(p[0] - a) + abs(p[1] - b) == 1 for a, b in points)]
            points.add(frontier[int(rng.integers(len(frontier)))])
        us, vs = zip(*points)
        if max(us) - min(us) < 1 or max(vs) - min(vs) < 1:
            continue
        if max(max(u, v) for u, v in points) != radius:
            continue
        if witness and points == {(v, u) for u, v in points}:
            continue
        return frozenset(points)
    # A connected asymmetric staircase, translated within larger domains.
    fallback = {(0, 0), (0, 1), (1, 1)}
    for j in range(2, radius + 1):
        fallback.update(((j - 1, j), (j, j)))
    return frozenset(fallback)


def _adjacent(node, nodes):
    y, x = node
    return [p for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)) if p in nodes]


def _layout(rng, radius, room_count, extra_loops=False, exact_door=False):
    """Variable rooms on a partially occupied, randomly spaced packing lattice."""
    minimum = 2 * radius + 3
    shapes = [(a, b) for a in range(1, 4) for b in range(1, 4)
              if a * b >= room_count and a * minimum + a + 1 <= 30
              and b * minimum + b + 1 <= 30]
    nr, nc = shapes[int(rng.integers(len(shapes)))]

    def axis(n):
        gaps = [1 if exact_door else int(rng.integers(1, 4)) for _ in range(n - 1)]
        while n * minimum + sum(gaps) > 28:
            gaps[int(rng.integers(len(gaps)))] = 1
        sizes = [minimum] * n
        budget = 28 - sum(sizes) - sum(gaps)
        for j in rng.permutation(n):
            add = int(rng.integers(min(3, budget) + 1))
            sizes[int(j)] += add
            budget -= add
        used = sum(sizes) + sum(gaps)
        margin = int(rng.integers(1, 30 - used))
        starts = [margin]
        for j in range(n - 1):
            starts.append(starts[-1] + sizes[j] + gaps[j])
        return starts, sizes, used + margin + int(rng.integers(1, 31 - used - margin))

    ys, heights, h = axis(nr)
    xs, widths, w = axis(nc)
    all_nodes = {(y, x) for y in range(nr) for x in range(nc)}
    root = sorted(all_nodes)[int(rng.integers(len(all_nodes)))]
    order, tree, used = [root], [], {root}
    while len(used) < room_count:
        choices = [(a, b) for a in sorted(used) for b in _adjacent(a, all_nodes - used)]
        a, b = choices[int(rng.integers(len(choices)))]
        used.add(b)
        order.append(b)
        tree.append((a, b))
    rooms = {}
    allowed = np.zeros((h, w), dtype=bool)
    for y, x in order:
        sh = heights[y] if exact_door else int(rng.integers(minimum, heights[y] + 1))
        sw = widths[x] if exact_door else int(rng.integers(minimum, widths[x] + 1))
        t = ys[y] + (heights[y] - sh) // 2
        l = xs[x] + (widths[x] - sw) // 2
        rooms[(y, x)] = (t, l, t + sh - 1, l + sw - 1)
        allowed[t:t + sh, l:l + sw] = True
    # The first edge supplies a guaranteed accessible second room: either a
    # wide channel or the short door matched to the threading demonstration.
    # Cutting a later tree edge leaves a blocked region while floor connects.
    critical = int(rng.integers(1, len(tree)))
    side = {root}
    for index, (a, b) in enumerate(tree):
        if index != critical and a in side:
            side.add(b)
    edges = list(tree)
    candidates = [(a, b) for a in sorted(used) for b in _adjacent(a, used)
                  if a < b and (a, b) not in tree and (b, a) not in tree
                  and ((a in side) == (b in side))]
    for edge in candidates:
        if not exact_door and rng.random() < (0.9 if extra_loops else 0.35):
            edges.append(edge)
    profiles = []
    for index, (a, b) in enumerate(edges):
        ra, rb = rooms[a], rooms[b]
        horizontal = a[0] == b[0]
        if horizontal:
            if ra[1] > rb[1]:
                ra, rb = rb, ra
            low, high = max(ra[0], rb[0]) + 1, min(ra[2], rb[2]) - 1
            begin, end = ra[3] + 1, rb[1]
        else:
            if ra[0] > rb[0]:
                ra, rb = rb, ra
            low, high = max(ra[1], rb[1]) + 1, min(ra[3], rb[3]) - 1
            begin, end = ra[2] + 1, rb[0]
        capacity = min(4, high - low + 1)
        _need(capacity >= radius + 1, "insufficient door overlap")
        center = int(rng.integers(low, high + 1))
        profile = []
        for j in range(begin, end):
            if index == 0:
                width = 2 if exact_door else int(rng.integers(radius + 1, capacity + 1))
            else:
                width = int(rng.integers(1, capacity + 1))
            if index == critical and j == (begin + end - 1) // 2:
                width = 1
            lo = max(low, center - width + 1)
            hi = min(center, high - width + 1)
            start = int(rng.integers(lo, hi + 1))
            if index == 0 and not exact_door:
                # All slices contain a common full-footprint-wide band.
                start = min(max(center - (radius + 1) // 2, low), high - capacity + 1)
                width = capacity
            if horizontal:
                allowed[start:start + width, j] = True
            else:
                allowed[j, start:start + width] = True
            profile.append(width)
        profiles.append(tuple(profile))
    _need(set(_rooms(allowed)) == set(rooms.values()), "passages obscure room geometry")
    # Free margins follow the occupied network, including partially occupied
    # packing lattices. This retains compact strips as well as broad scenes.
    cells = np.argwhere(allowed)
    top, left = map(int, cells.min(axis=0))
    bottom, right = map(int, cells.max(axis=0) + 1)
    margins = []
    for length in (bottom - top, right - left):
        total = int(rng.integers(2, min(9, 30 - length) + 1))
        before = int(rng.integers(1, total))
        margins.append((before, total - before))
    allowed = np.pad(allowed[top:bottom, left:right], margins)
    dy, dx = margins[0][0] - top, margins[1][0] - left
    rooms = {node: (t + dy, l + dx, b + dy, r + dx)
             for node, (t, l, b, r) in rooms.items()}
    return allowed, list(rooms.values()), rooms[root], len(edges) - len(tree), profiles


def _reflected_countermodel(inp, info):
    """Isolate B's reflection/rotation ambiguity while keeping access fixed."""
    y, x, q = info["sources"][0]
    signs = ((1, 1), (1, -1), (-1, -1), (-1, 1))
    dy, dx = signs[q]
    source_points = _positions((y, x, q), info["pattern"])
    reflected = {(dy * (a - y), dx * (b - x)) for a, b in source_points}
    result = inp.copy()
    reached = set(info["reached"]) | set(info["sources"])
    for (a, b, o), color, _ in info["occurrences"]:
        if (a, b, o) in reached:
            sy, sx = signs[o]
            for u, v in reflected:
                result[a + sy * u, b + sx * v] = color
    return result


def _make_pair(rng, radius, room_count, roles, used, used_patterns, witness, loops):
    for _ in range(100):
        if witness:
            # A pedagogical shape whose narrow middle column permits a
            # sideways shift through a short width-two doorway. Pick its
            # orientation/reflection; other examples use random connected dies.
            base = {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)}
            q = int(rng.integers(4))
            offsets = [_rotate((u - 1, v - 1), q) for u, v in base]
            pattern = frozenset((u + 1, v + 1) for u, v in offsets)
            if rng.integers(2):
                pattern = frozenset((v, u) for u, v in pattern)
        else:
            pattern = _sample_pattern(rng, radius)
        if pattern in used_patterns:
            continue
        try:
            allowed, rects, source_rect, _, _ = _layout(rng, radius, room_count, loops, witness)
        except AssertionError:
            continue
        source_q = int(rng.integers(4))
        source = _corners(source_rect)[source_q]
        seen, r = _reach(allowed, pattern, [source])
        reachable = [rect for rect in rects if _is_reached(seen, r, _corners(rect)[0])]
        blocked = [rect for rect in rects if rect not in reachable]
        if len(reachable) < 2 or not blocked:
            continue
        wall, floor = roles
        accents = [v for v in range(10) if v not in roles]
        rng.shuffle(accents)
        palette = accents[:int(rng.integers(1, min(8, 2 * room_count) + 1))]
        selected = {source, _corners(source_rect)[(source_q + 1) % 4]}
        other = next(rect for rect in reachable if rect != source_rect)
        selected.add(_corners(other)[int(rng.integers(4))])
        blocked_witness = _corners(blocked[int(rng.integers(len(blocked)))])[int(rng.integers(4))]
        selected.add(blocked_witness)
        for rect in rects:
            for corner in _corners(rect):
                if rng.random() < float(rng.uniform(0.12, 0.55)):
                    selected.add(corner)
        inp = np.full(allowed.shape, wall, dtype=np.int8)
        inp[allowed] = floor
        for corner in sorted(selected):
            color = int(rng.choice(palette))
            if corner == source:
                clue = pattern
            elif corner == blocked_witness:
                clue = {sorted(pattern)[int(rng.integers(len(pattern)))]}
            elif witness and corner == _corners(source_rect)[(source_q + 1) % 4]:
                clue = {(1, 1)}
            elif corner in _corners(source_rect) and rng.random() < 0.12:
                clue = pattern
            else:
                n = int(rng.integers(1, len(pattern)))
                ordered = sorted(pattern)
                clue = {ordered[int(j)] for j in rng.permutation(len(pattern))[:n]}
            for point in _positions(corner, clue):
                inp[point] = color
        out, info = _analyze(inp)
        if not info["reached"] or not info["blocked"]:
            continue
        if witness and np.array_equal(_reflected_countermodel(inp, info), out):
            continue
        # Rigidly rotate/reflect the whole world; the quarter-turn convention
        # remains valid because the normalized die changes with the world.
        turns = int(rng.integers(4))
        flip = bool(rng.integers(2))
        inp, out = np.rot90(inp, turns), np.rot90(out, turns)
        if flip:
            inp, out = np.fliplr(inp), np.fliplr(out)
        inp, out = inp.copy(), out.copy()
        key = (inp.shape, inp.tobytes())
        if key in used:
            continue
        transformed_prediction, after = _analyze(inp)
        _need(np.array_equal(transformed_prediction, out), "world transform changed semantics")
        if after["pattern"] in used_patterns:
            continue
        if witness:
            if np.array_equal(_reflected_countermodel(inp, after), out):
                continue
        used.add(key)
        used_patterns.add(after["pattern"])
        return {"input": inp, "output": out}
    raise RuntimeError("bounded room/stencil construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(2))
    base_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    variable_roles = bool(rng.integers(2))
    puzzle, used, patterns = {"train": [], "test": []}, set(), set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            witness = split == "train" and index == 0
            if split == "train":
                radius = 2 if witness else int(rng.integers(1, 3))
                room_count = int(rng.integers(3, 5 if regime else 8))
            elif regime == 0:
                radius, room_count = 3, 4
            else:
                radius = int(rng.integers(1, 3))
                room_count = int(rng.integers(5, 9))
            roles = tuple(map(int, rng.choice(10, 2, replace=False))) if variable_roles else base_roles
            # Radius-one has just five admissible non-line patterns. Reuse
            # across inputs is legitimate; geometry, damage and palette vary.
            excluded = patterns if radius > 1 else set()
            pair = _make_pair(rng, radius, room_count, roles, used, excluded, witness,
                              split == "test" and regime == 1)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    footprint_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _need(inp.shape == out.shape, "shape preservation")
            key = inp.shape, inp.tobytes()
            _need(key not in seen, "duplicate input")
            seen.add(key)
            prediction, info = _analyze(inp)
            _need(np.array_equal(prediction, out), "incorrect access/completion or altered scenery")
            _need(info["reached"] and info["blocked"], "each example must expose access and blockage")
            _need(info["reachable_rooms"] >= 2, "must visit another room")
            source_rooms = [rect for rect in info["rooms"]
                            if any(c in _corners(rect) for c in info["sources"])]
            _need(any(not any(c in _corners(rect) for rect in source_rooms) for c in info["reached"]),
                  "must complete a request outside the reference rooms")
            changed = inp != out
            _need(changed.any() and np.all(inp[changed] == info["floor"]), "only missing floor cells change")
            _need(set(np.unique(inp)) == set(np.unique(out)), "palette must be preserved")
            _need(any(c in info["blocked"] and len(clue) == 1 for c, _, clue in info["occurrences"]),
                  "need blocked singleton to distinguish clue-sized movement")
            # Completing every marked corner discards A and must differ.
            all_completed = inp.copy()
            for corner, color, _ in info["occurrences"]:
                for point in _positions(corner, info["pattern"]):
                    all_completed[point] = color
            _need(not np.array_equal(all_completed, out), "A omission has no effect")
            _need(not np.array_equal(inp, out), "B omission has no effect")
            if split == "train" and not footprint_witness:
                us, vs = zip(*info["pattern"])
                box = {(u, v) for u in range(min(us), max(us) + 1)
                       for v in range(min(vs), max(vs) + 1)}
                box_seen, br = _reach(info["allowed"], box, info["sources"])
                footprint_witness = any(not _is_reached(box_seen, br, c) for c in info["reached"])
            infos[split].append(info)
    _need(footprint_witness, "training must distinguish actual shape from filled bounding box")
    _need(any(not np.array_equal(_reflected_countermodel(_grid(pair["input"]), info),
                                 _grid(pair["output"]))
              for pair, info in zip(puzzle["train"], infos["train"])),
          "training must distinguish quarter-turns from axis reflections")
    train_radius = max(info["radius"] for info in infos["train"])
    train_rooms = max(len(info["rooms"]) for info in infos["train"])
    _need(all(info["radius"] > train_radius for info in infos["test"])
          or all(len(info["rooms"]) > train_rooms for info in infos["test"]),
          "test must extend die span or room count")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for key in ("input", "output"):
                a = _grid(pair[key])
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _regression_test():
    # Hand-specified geometric contrast: a five-cell zigzag has a singleton
    # middle column, which lets it shift sideways while crossing a short
    # two-cell door. Its filled 3x3 bounding square cannot pass that door.
    # The next one-cell door blocks the zigzag too. Labels below specify
    # the expected accessible rooms, without calling the production solver.
    pattern = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)})
    allowed = np.zeros((9, 25), dtype=bool)
    for left in (1, 9, 17):
        allowed[1:8, left:left + 7] = True
    allowed[3:5, 8] = True
    allowed[4, 16] = True
    inp = np.zeros(allowed.shape, dtype=np.int8)
    inp[allowed] = 1
    requests = [((1, 1, 0), 2, pattern), ((1, 7, 1), 3, {(0, 0)}),
                ((1, 9, 0), 2, {(0, 0)}), ((1, 17, 0), 3, {(0, 0)})]
    for corner, color, clue in requests:
        for point in _positions(corner, clue):
            inp[point] = color
    expected = inp.copy()
    for corner, color, _ in requests[:3]:
        for point in _positions(corner, pattern):
            expected[point] = color
    for turns in range(4):
        for flip in (False, True):
            a, b = np.rot90(inp, turns), np.rot90(expected, turns)
            if flip:
                a, b = np.fliplr(a), np.fliplr(b)
            _need(np.array_equal(solve(a), b), "short-door zigzag regression")
    box = frozenset((u, v) for u in range(3) for v in range(3))
    seen, radius = _reach(allowed, box, [(1, 1, 0)])
    _need(not _is_reached(seen, radius, (1, 9, 0)), "bounding square unexpectedly passed")
    # A tiny retained clue is not the physical die. A point can traverse all
    # three rooms, whereas the reconstructed zigzag cannot reach the third.
    point_seen, radius = _reach(allowed, {(0, 0)}, [(1, 1, 0)])
    _need(_is_reached(point_seen, radius, (1, 17, 0)), "point-sized ablation")
    # Reused accents are local socket identities, not whole-color objects.
    _need(np.array_equal(solve(inp), expected), "reused-color regression")
    return {"input": inp, "output": expected}


def self_test():
    """Bounded invariants, counterexamples, reproducibility and 42-seed diversity."""
    _regression_test()
    features, signatures, patterns, shapes, palettes = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                prediction, info = _analyze(pair["input"])
                _need(np.array_equal(prediction, pair["output"]), "input-only prediction")
                _need(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "nested-list solve")
                patterns.add(tuple(sorted(info["pattern"])))
                shapes.add(pair["input"].shape)
                palettes.add((info["wall"], info["floor"]))
                features.append((len(info["rooms"]), info["radius"], len(info["pattern"]),
                                 len(info["sources"]), info["reachable_rooms"],
                                 len({color for _, color, _ in info["occurrences"]})))
    _need(len(signatures) == 42, "cross-seed puzzle variation")
    _need(len(patterns) >= 50 and len(shapes) >= 25 and len(palettes) >= 20, "structural variation")
    _need({f[1] for f in features} == {1, 2, 3}, "die spans")
    _need(len({f[0] for f in features}) >= 5 and len({f[2] for f in features}) >= 7, "counts and die sizes")
    _need(any(f[3] > 1 for f in features), "multiple references")
    _need(any(f[4] >= 3 for f in features), "longer access")
    _need(_fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0)), "determinism")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            case = generate_puzzle(900 + 10 * n + m, n, m)
            _need(len(case["train"]) == n and len(case["test"]) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    original = generate_puzzle(1)
    for mutation in ("wall", "blocked", "reached"):
        bad = {s: [{k: a.copy() for k, a in p.items()} for p in original[s]] for s in original}
        pair = bad["test"][0]
        info = _analyze(pair["input"])[1]
        if mutation == "wall":
            pair["output"][0, 0] = info["floor"]
        else:
            targets = info["blocked"] if mutation == "blocked" else info["reached"]
            corner, color, clue = next(o for o in info["occurrences"] if o[0] == targets[0])
            missing = sorted(_positions(corner, info["pattern"]) - _positions(corner, clue))[0]
            pair["output"][missing] = color if mutation == "blocked" else info["floor"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted " + mutation)
