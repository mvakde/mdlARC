"""Rolling contour stamps: finite-body exploration and deepest-ink superposition.

No files are read or written. All random choices belong to the local Generator.
"""
from collections import deque
import hashlib
import json

import numpy as np


_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=False)


def _neighbors(p, shape):
    for dr, dc in _DIR4:
        r, c = p[0] + dr, p[1] + dc
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            yield r, c


def _reachable(mask, start):
    result = np.zeros(mask.shape, dtype=bool)
    _require(bool(mask[start]), "starting footprint is illegal")
    result[start] = True
    queue = deque([start])
    while queue:
        for q in _neighbors(queue.popleft(), mask.shape):
            if mask[q] and not result[q]:
                result[q] = True
                queue.append(q)
    return result


def _erosion_depth(mask):
    """Eight-neighbor erosion; the exterior is outside, even on the canvas edge."""
    current = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int8)
    h, w = mask.shape
    d = 0
    while current.any():
        padded = np.pad(current, 1)
        inner = current.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                inner &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~inner] = d
        current = inner
        d += 1
    return result


def _decode(inp):
    """Recognize the exterior, floor, square frame and its inward diagonal key."""
    wall = int(inp[0, 0])
    _require(all(np.all(edge == wall) for edge in
                 (inp[0], inp[-1], inp[:, 0], inp[:, -1])), "exterior must be monochrome")
    counts = np.bincount(inp.ravel(), minlength=10)
    others = [c for c in range(10) if c != wall and counts[c]]
    _require(len(others) >= 2, "floor and stamp are required")
    neutral = max(others, key=lambda c: int(counts[c]))
    _require(all(counts[neutral] > counts[c] for c in others if c != neutral),
             "floor must be the unique most frequent non-wall color")
    marks = (inp != wall) & (inp != neutral)
    candidates = []
    for outer in others:
        if outer == neutral:
            continue
        points = np.argwhere(inp == outer)
        lo, hi = points.min(axis=0), points.max(axis=0) + 1
        a, b = map(int, lo)
        k, width = map(int, hi - lo)
        if k != width or k not in (3, 5, 7, 9):
            continue
        box = inp[a:a + k, b:b + k]
        frame = np.zeros((k, k), dtype=bool)
        frame[0] = frame[-1] = True
        frame[:, 0] = frame[:, -1] = True
        if not np.all(box[frame] == outer) or np.any(box == wall):
            continue
        outside = marks.copy()
        outside[a:a + k, b:b + k] = False
        if outside.any():
            continue
        rr, cc = np.indices((k, k))
        depth = np.minimum.reduce((rr, cc, k - 1 - rr, k - 1 - cc))
        cue = (box != neutral) & ~frame
        length = int(depth[cue].max()) + 1 if cue.any() else 1
        if length > 3:
            continue
        readings = set()
        for dr in (-1, 1):
            for dc in (-1, 1):
                r0 = 0 if dr == 1 else k - 1
                c0 = 0 if dc == 1 else k - 1
                key_positions = {(r0 + d * dr, c0 + d * dc) for d in range(1, length)}
                if not all(tuple(p) in key_positions for p in np.argwhere(cue)):
                    continue
                period = tuple(int(box[r0 + d * dr, c0 + d * dc]) for d in range(length))
                if period[-1] != neutral and period[0] == outer:
                    readings.add(period)
        if len(readings) == 1:
            candidates.append((wall, neutral, outer, (a, b), k, readings.pop()))
    _require(len(candidates) == 1, "expected one unambiguous framed contour stamp")
    wall, neutral, outer, start, k, period = candidates[0]
    floor = inp != wall
    cell_reach = _reachable(floor, start)
    _require(np.array_equal(cell_reach, floor), "the floor must be connected")
    valid = np.lib.stride_tricks.sliding_window_view(floor, (k, k)).all(axis=(-1, -2))
    reach = _reachable(valid, start)
    return {"wall": wall, "neutral": neutral, "outer": outer, "start": start,
            "k": k, "period": period, "floor": floor, "valid": valid, "reach": reach}


def _support(info):
    """Maximal local contour depth among all reachable square imprints."""
    k = info["k"]
    rr, cc = np.indices((k, k))
    stamp = np.minimum.reduce((rr, cc, k - 1 - rr, k - 1 - cc)).astype(np.int8)
    result = np.full(info["floor"].shape, -1, dtype=np.int8)
    for r, c in np.argwhere(info["reach"]):
        target = result[r:r + k, c:c + k]
        np.maximum(target, stamp, out=target)
    return result


def _render(inp, info, depth):
    out = inp.copy()
    out[info["floor"]] = info["neutral"]
    painted = depth >= 0
    period = np.asarray(info["period"], dtype=np.int8)
    out[painted] = period[depth[painted] % len(period)]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Explore by unit orthogonal translation, then combine all contour imprints."""
    inp = _grid(input_grid)
    info = _decode(inp)
    return _render(inp, info, _support(info))


def _check_support(info, actual):
    """Check imprints through center-set dilations, without iterating square stamps."""
    radius = info["k"] // 2
    centers = np.zeros(info["floor"].shape, dtype=bool)
    h, w = info["reach"].shape
    centers[radius:radius + h, radius:radius + w] = info["reach"]
    expected = np.full(centers.shape, -1, dtype=np.int8)
    for d in range(radius + 1):
        spread = radius - d
        expanded = np.lib.stride_tricks.sliding_window_view(
            np.pad(centers, spread), (2 * spread + 1, 2 * spread + 1)).any(axis=(-1, -2))
        expected[expanded] = d
    _require(np.array_equal(expected, actual), "contour support is inconsistent")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("counts must be integers")
        if not low <= int(value) <= high:
            raise ValueError("counts outside supported range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _layout(rng, k, anchor):
    q = k + (2 if anchor else int(rng.integers(4)))
    limit = (29 // (q + 1))
    minimum_count = (3 if k == 7 else 4) if anchor else 3
    options = [(nr, nc) for nr in range(1, limit + 1) for nc in range(1, limit + 1)
               if nr * nc >= minimum_count]
    nr, nc = options[int(rng.integers(len(options)))]

    def coordinates(n):
        gaps = np.ones(max(0, n - 1), dtype=int)
        budget = 28 - n * q - int(gaps.sum())
        for j in rng.permutation(len(gaps)):
            extra = int(rng.integers(min(3, budget) + 1))
            gaps[j] += extra
            budget -= extra
        return np.concatenate(([0], np.cumsum(q + gaps)))

    rows, cols = coordinates(nr), coordinates(nc)
    count = minimum_count if anchor else int(rng.integers(3, min(16, nr * nc) + 1))
    root = (int(rng.integers(nr)), int(rng.integers(nc)))
    nodes, occupied, edges = [root], {root}, []
    while len(nodes) < count:
        choices = [(a, b) for a in nodes for b in _neighbors(a, (nr, nc)) if b not in occupied]
        a, b = choices[int(rng.integers(len(choices)))]
        nodes.append(b)
        occupied.add(b)
        edges.append((a, b))
    if not anchor:
        extra = [(a, b) for a in nodes for b in _neighbors(a, (nr, nc))
                 if b in occupied and a < b and (a, b) not in edges and (b, a) not in edges]
        edges.extend(edge for edge in extra if rng.random() < 0.4)
    rooms = {}
    for node in nodes:
        rh, rw = (q, q) if node == root else tuple(map(int, rng.integers(k + int(anchor), q + 1, size=2)))
        rooms[node] = (int(rows[node[0]]), int(cols[node[1]]), rh, rw)
    shape = (max(r + rh for r, c, rh, rw in rooms.values()),
             max(c + rw for r, c, rh, rw in rooms.values()))
    floor = np.zeros(shape, dtype=bool)
    for r, c, rh, rw in rooms.values():
        floor[r:r + rh, c:c + rw] = True
    all_pass = not anchor and rng.random() < 0.3
    for i, (a, b) in enumerate(edges):
        ar, ac, ah, aw = rooms[a]
        br, bc, bh, bw = rooms[b]
        capacity = min(ah, bh) if ar == br else min(aw, bw)
        blocked = (k == 7 and i == len(edges) - 1) if anchor else bool(
            i != 0 and not all_pass and rng.random() < 0.45)
        width = int(rng.integers(1, k)) if blocked else int(rng.integers(k, capacity + 1))
        if anchor and (blocked or (k == 5 and i == 0)):
            width = 6
        offset = int(rng.integers(capacity - width + 1))
        if ar == br:
            if ac > bc:
                ac, bc, aw = bc, ac, bw
            floor[ar + offset:ar + offset + width, ac + aw:bc] = True
        else:
            if ar > br:
                ar, br, ah = br, ar, bh
            floor[ar + ah:br, ac + offset:ac + offset + width] = True
    r, c, rh, rw = rooms[root]
    start = (r + int(rng.integers(rh - k + 1)), c + int(rng.integers(rw - k + 1)))
    points = np.argwhere(floor)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    floor = floor[lo[0]:hi[0], lo[1]:hi[1]]
    start = (start[0] - int(lo[0]), start[1] - int(lo[1]))
    padding = []
    for extent in floor.shape:
        extra = int(rng.integers(31 - extent - 2))
        before = int(rng.integers(extra + 1))
        padding.append((1 + before, 1 + extra - before))
    floor = np.pad(floor, padding)
    start = (start[0] + padding[0][0], start[1] + padding[1][0])
    return floor, start


def _make_pair(rng, k, key_kind, anchor):
    floor, (r, c) = _layout(rng, k, anchor)
    wall, neutral, outer, ink, ink2 = map(int, rng.choice(10, 5, replace=False))
    if key_kind == "repeat":
        period = (outer, ink)
    elif key_kind == "gap":
        period = (outer, neutral, ink)
    else:
        length = int(rng.integers(2 if key_kind == "test" else 1, min(3, k // 2 + 1) + 1))
        period = (outer, ink, ink2)[:length]
        if length == 3 and rng.random() < 0.4:
            period = (outer, neutral, ink)
    inp = np.full(floor.shape, wall, dtype=np.int8)
    inp[floor] = neutral
    box = inp[r:r + k, c:c + k]
    box[0] = box[-1] = outer
    box[:, 0] = box[:, -1] = outer
    dr, dc = map(int, rng.choice((-1, 1), size=2))
    a, b = (0 if dr == 1 else k - 1), (0 if dc == 1 else k - 1)
    for d, color in enumerate(period):
        box[a + dr * d, b + dc * d] = color
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    inp = np.rot90(inp, turns)
    inp = np.array(np.fliplr(inp) if reflect else inp, dtype=np.int8, copy=True)
    return {"input": inp, "output": solve(inp)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    train_specs = [(7, "repeat", True), (5, "gap", True)]
    train_specs += [(int(rng.choice((5, 7))), "random", False) for _ in range(nt - 2)]
    train_specs = [train_specs[int(i)] for i in rng.permutation(nt)]
    # Either direction of size transfer, including mixed test sets, is possible.
    test_size = int(rng.choice((3, 9)))
    tests = [(test_size if i == 0 else int(rng.choice((3, 9))), "test", False) for i in range(ne)]
    result, seen = {"train": [], "test": []}, set()
    for split, specs in (("train", train_specs), ("test", tests)):
        for k, kind, anchor in specs:
            for _ in range(16):
                pair = _make_pair(rng, k, kind, anchor)
                signature = _signature(pair["input"])
                if signature not in seen:
                    break
            else:
                raise RuntimeError("bounded uniqueness search exhausted")
            seen.add(signature)
            result[split].append(pair)
    validate_generated_puzzle(result)
    return result


def _signature(a):
    return (tuple(a.shape), a.astype(np.int8, copy=False).tobytes())


def _pair_info(pair):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "shape changed")
    info = _decode(inp)
    depth = _support(info)
    _check_support(info, depth)
    _require(np.array_equal(out, _render(inp, info, depth)), "wrong contour imprint or preserved cells")
    _require(np.array_equal(inp == info["wall"], out == info["wall"]), "walls changed")
    _require(int(info["reach"].sum()) > 1, "stamp must be able to move")
    _require(not np.array_equal(inp, out), "identity example")
    info["support"] = depth
    return info


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    try:
        _counts(len(puzzle["train"]), len(puzzle["test"]))
    except ValueError as error:
        raise AssertionError(str(error)) from error
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            info = _pair_info(pair)
            signature = _signature(_grid(pair["input"]))
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            analyses[split].append(info)
    training = analyses["train"]
    _require({a["k"] for a in training} == {5, 7}, "training must demonstrate two body sizes")
    _require(any(a["k"] == 7 and len(a["period"]) == 2 for a in training), "missing repeated two-color key")
    _require(any(a["k"] == 5 and len(a["period"]) == 3 and a["period"][1] == a["neutral"]
                 for a in training), "missing meaningful neutral key entry")
    _require(all(a["k"] in (3, 9) for a in analyses["test"]), "missing held-out body-size shift")
    _require(all(len(a["period"]) >= 2 for a in analyses["test"]), "test must exercise contour ink")
    a_witnesses = b_witnesses = contour_witnesses = 0
    partial = complete = False
    for pair, a in zip(puzzle["train"], training):
        depth, floor, period = a["support"], a["floor"], np.asarray(a["period"])
        painted = depth >= 0
        # Both ablations are evaluated as output colors, not just geometry.
        point_flood = period[_erosion_depth(floor)[floor] % len(period)]
        actual_floor = np.full(int(floor.sum()), a["neutral"], dtype=np.int8)
        actual_floor[painted[floor]] = period[depth[painted] % len(period)]
        omit_a = np.full(int(painted.sum()), a["outer"])
        distinguishes_a = bool(np.any(period[depth[painted] % len(period)] != omit_a))
        teleport = dict(a, reach=a["valid"])
        without_b = _render(_grid(pair["input"]), teleport, _support(teleport))
        distinguishes_b = bool(np.any(without_b[floor] != actual_floor))
        contour_shortcut_fails = bool(np.any(actual_floor != point_flood))
        a_witnesses += int(distinguishes_a)
        b_witnesses += int(distinguishes_b)
        contour_witnesses += int(contour_shortcut_fails)
        partial |= bool(np.any(floor & ~painted))
        complete |= bool(np.array_equal(floor, painted))
    _require(a_witnesses >= 2 and b_witnesses >= 1 and contour_witnesses >= 2,
             "demonstrations must distinguish both parent ablations and whole-floor contours")
    _require(partial and complete, "training must show both excluded and fully explored floor")


def self_test():
    """42 puzzles, scalar statistics only; no saved grids, logs, or unbounded retries."""
    stats = {name: set() for name in ("puzzles", "shapes", "geometry", "sizes", "keys", "roles", "reach_counts")}
    partial = complete = branched = capped_witnesses = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _pair_info(pair)
                inp = pair["input"]
                digest.update(bytes(inp.shape) + inp.tobytes() + pair["output"].tobytes())
                stats["shapes"].add(inp.shape)
                stats["geometry"].add(hashlib.sha256(bytes(inp.shape) + a["floor"].tobytes()).digest())
                stats["sizes"].add(a["k"])
                stats["keys"].add((len(a["period"]), a["neutral"] in a["period"]))
                stats["roles"].add((a["wall"], a["neutral"], a["outer"]))
                stats["reach_counts"].add(int(a["reach"].sum()))
                painted = a["support"] >= 0
                partial += int(np.any(a["floor"] & ~painted))
                complete += int(np.array_equal(a["floor"], painted))
                branched += int(any(sum(bool(a["reach"][q]) for q in _neighbors(tuple(p), a["reach"].shape)) > 2
                                    for p in np.argwhere(a["reach"])))
                raw_depth = _erosion_depth(a["floor"])
                colors = np.asarray(a["period"])
                capped_witnesses += int(np.any(painted & (raw_depth > a["k"] // 2) &
                                             (colors[np.maximum(raw_depth, 0) % len(colors)] !=
                                              colors[np.maximum(a["support"], 0) % len(colors)])))
        stats["puzzles"].add(digest.digest())
    _require(len(stats["puzzles"]) == 42, "puzzles collapsed across seeds")
    _require(len(stats["shapes"]) >= 50 and len(stats["geometry"]) >= 150, "insufficient structural variation")
    _require(stats["sizes"] == {3, 5, 7, 9}, "size transfer coverage")
    _require({x[0] for x in stats["keys"]} == {1, 2, 3}, "key-length coverage")
    _require(len(stats["roles"]) >= 100 and len(stats["reach_counts"]) >= 40, "palette/geometry coverage")
    _require(partial >= 42 and complete >= 42 and branched >= 100 and capped_witnesses >= 42,
             "missing bottlenecks, broad exploration or contour saturation")
    # Hand-calculated straight channel: one seven-square, then a six-wide throat.
    fixture = np.full((11, 23), 8, dtype=np.int8)
    fixture[1:10, 1:10] = 0
    fixture[2:8, 10:13] = 0
    fixture[1:10, 13:22] = 0
    fixture[2:9, 2:9] = 2
    fixture[3:8, 3:8] = 0
    fixture[3, 3] = 4
    predicted = solve(fixture)
    expected = np.full_like(fixture, 8)
    expected[fixture != 8] = 0
    # All 3x3 placements in the first chamber are reachable. Their centers
    # fill rows/columns 4..6; best imprint depth is 3 minus Chebyshev distance.
    for r in range(1, 10):
        for c in range(1, 10):
            distance = max(4 - r, r - 6, 4 - c, c - 6, 0)
            expected[r, c] = (2, 4)[(3 - distance) % 2]
    _require(np.array_equal(predicted, expected), "hand-calculated throat fixture failed")
    # Widening just one row admits the same body; both rooms and the bridge paint.
    wide = fixture.copy()
    wide[8, 10:13] = 0
    _require(np.any(solve(wide)[1:10, 13:22] != 0), "width comparison is ineffective")
    # A neutral middle layer wins over shallower colored imprints. In a 9x9
    # chamber the five-square's reachable centers form the central 5x5 block.
    gap = np.full((11, 11), 8, dtype=np.int8)
    gap[1:10, 1:10] = 0
    gap[2:7, 2:7] = 2
    gap[3:6, 3:6] = 0
    gap[4, 4] = 4
    expected_gap = np.full_like(gap, 8)
    expected_gap[1:10, 1:10] = 2
    expected_gap[2:9, 2:9] = 0
    expected_gap[3:8, 3:8] = 4
    _require(np.array_equal(solve(gap), expected_gap), "neutral ink must win at the deepest layer")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(42, nt, ne)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    first, second = generate_puzzle(987, 2, 1), generate_puzzle(987, 2, 1)
    _require(all(np.array_equal(p[k], q[k]) for s in first for p, q in zip(first[s], second[s])
                 for k in ("input", "output")), "seed nondeterminism")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted changed scenery")
    return {**{k: len(v) for k, v in stats.items()}, "partial": partial, "complete": complete,
            "placement_branching": branched, "saturation_witnesses": capped_witnesses}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
