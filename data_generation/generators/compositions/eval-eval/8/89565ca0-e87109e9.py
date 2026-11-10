"""Compartment-powered relay beams: input-derived topology and finite-range turns.

Only the standard library and NumPy are used. The solver has no generator state.
"""
from collections import deque
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(np.all((g >= 0) & (g <= 9)), "invalid color")
    return g.astype(np.int8, copy=False)


def _signature(g):
    return g.shape, g.tobytes()


def _rooms(mask):
    """Bounded four-connected free regions, with a one-cell exterior collar."""
    free = np.pad(~mask, 1, constant_values=True)
    seen = np.zeros(free.shape, dtype=bool)
    result = []
    h, w = free.shape
    for rr, cc in np.argwhere(free):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        cells, outside = [], False
        while queue:
            y, x = queue.popleft()
            cells.append((y - 1, x - 1))
            outside |= y in (0, h - 1) or x in (0, w - 1)
            for dy, dx in DIRS:
                a, b = y + dy, x + dx
                if 0 <= a < h and 0 <= b < w and free[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    queue.append((a, b))
        if not outside:
            a = np.asarray(cells)
            top, left = a.min(axis=0)
            bottom, right = a.max(axis=0)
            _require(len(cells) == (bottom - top + 1) * (right - left + 1),
                     "elementary rooms must be rectangles")
            result.append(tuple(map(int, (top, left, bottom, right))))
    return result


def _walls(shape, rooms):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in rooms:
        mask[t, l:r + 1] = mask[b, l:r + 1] = True
        mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _euler_count(mask):
    edges = int(np.sum(mask[:, :-1] & mask[:, 1:]) + np.sum(mask[:-1] & mask[1:]))
    faces = int(np.sum(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]))
    return edges - int(mask.sum()) + 1 - faces


def _partition(rng, height, width, count, style):
    """Individual-room splits retain actual endpoints, including T-junctions."""
    _require(((height - 1) // 2) * ((width - 1) // 2) >= count, "insufficient room capacity")
    for _ in range(80):
        boxes = [(0, 0, height - 1, width - 1)]
        first_axis = None
        if style in ("strip", "grid"):
            factors = [(a, count // a) for a in range(1, count + 1)
                       if count % a == 0 and 2 * a < height and 2 * (count // a) < width
                       and (style == "grid" or a == 1 or a == count)]
            if factors:
                nr, nc = factors[int(rng.integers(len(factors)))]
                def offsets(n, span):
                    gaps = np.full(n, 2, dtype=int)
                    gaps += rng.multinomial(span - 2 * n, np.full(n, 1 / n))
                    return [0] + np.cumsum(gaps).tolist()
                ys, xs = offsets(nr, height - 1), offsets(nc, width - 1)
                boxes = [(ys[i], xs[j], ys[i + 1], xs[j + 1]) for i in range(nr) for j in range(nc)]
                return _walls((height, width), boxes)
        for step in range(count - 1):
            choices = []
            for i, (t, l, b, r) in enumerate(boxes):
                for axis, start, end in ((0, t, b), (1, l, r)):
                    if style == "partial" and step == 1 and axis == first_axis:
                        continue
                    for cut in range(start + 2, end - 1):
                        halves = [(t, l, cut, r), (cut, l, b, r)] if axis == 0 else [(t, l, b, cut), (t, cut, b, r)]
                        capacity = sum(((z[2] - z[0]) // 2) * ((z[3] - z[1]) // 2)
                                       for j, z in enumerate(boxes) if j != i)
                        capacity += sum(((z[2] - z[0]) // 2) * ((z[3] - z[1]) // 2) for z in halves)
                        if capacity >= count:
                            choices.append((i, axis, halves))
            if not choices:
                break
            i, axis, halves = choices[int(rng.integers(len(choices)))]
            if step == 0:
                first_axis = axis
            boxes[i:i + 1] = halves
        if len(boxes) != count:
            continue
        mask = _walls((height, width), boxes)
        if style == "partial" and count >= 3 and _lattice_count(mask) == count:
            continue
        return mask
    # A finite partial partition when a long enough axis is available.
    # This path is not a different semantic rule.
    if width >= 5 and height >= 2 * (count - 1) + 1 and count >= 3:
        cut = 2
        ys = [i * (height - 1) // (count - 1) for i in range(count)]
        boxes = [(ys[i], 0, ys[i + 1], cut) for i in range(count - 1)]
        boxes.append((0, cut, height - 1, width - 1))
        return _walls((height, width), boxes)
    raise ValueError("could not construct requested partition")


def _lattice_count(mask):
    hs = np.flatnonzero(np.any(mask[:, :-1] & mask[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(mask[:-1] & mask[1:], axis=0))
    return (len(hs) - 1) * (len(vs) - 1)


def _recover(field, color, scar):
    cells = np.argwhere(field == color)
    _require(len(cells) >= 4, "missing frame evidence")
    t, l = map(int, cells.min(axis=0))
    b, r = map(int, cells.max(axis=0))
    local = field[t:b + 1, l:r + 1]
    mask = local == color
    # Scars are isolated straight-wall substitutions, never background erasures.
    candidates = local == scar
    horizontal = np.zeros_like(mask)
    vertical = np.zeros_like(mask)
    horizontal[:, 1:-1] = mask[:, :-2] & mask[:, 2:]
    vertical[1:-1] = mask[:-2] & mask[2:]
    mask |= candidates & (horizontal | vertical)
    _require(mask[0].all() and mask[-1].all() and mask[:, 0].all() and mask[:, -1].all(),
             "outer boundary is not recoverable")
    _require(np.all(mask[[0, 0, -1, -1], [0, -1, 0, -1]]), "missing frame corners")
    rooms = _rooms(mask)
    _require(bool(rooms), "frame encloses no rooms")
    boundaries = [(a - 1, c - 1, d + 1, e + 1) for a, c, d, e in rooms]
    _require(np.array_equal(mask, _walls(mask.shape, boundaries)), "invalid partition walls")
    _require(_euler_count(mask) == len(rooms), "topology count disagreement")
    _require(np.all((local == scar) <= mask), "scar inside an open room")
    return {"color": int(color), "box": (t, l, b + 1, r + 1), "mask": mask,
            "count": len(rooms), "rooms": rooms, "damaged": bool(np.any(mask & (local == scar))),
            "lattice": _lattice_count(mask)}


def _decode(input_grid):
    g = _grid(input_grid)
    frame = int(g[0, 0])
    _require(np.all(g[0] == frame), "missing top instruction frame")
    bottoms = np.flatnonzero(np.all(g[1:] == frame, axis=1)) + 1
    _require(len(bottoms) > 0, "missing bottom instruction frame")
    bottom = int(bottoms[0])
    _require(2 <= bottom < len(g) - 1, "invalid instruction band")
    band = g[1:bottom]
    _require(np.all(band == band[0]), "inconsistent key rows")
    row = band[0]
    _require(row[0] == row[-1] == frame, "missing panel side walls")
    runs, c = [], 0
    while c < len(row):
        if row[c] == frame:
            c += 1
            continue
        a = c
        while c < len(row) and row[c] != frame:
            c += 1
        runs.append((a, c))
    _require(bool(runs) and all(b - a >= 3 for a, b in runs), "ambiguous panels")
    background = int(row[runs[0][0] + 1])
    turns = {}
    for a, b in runs:
        _require(np.all(row[a + 1:b - 1] == background), "inconsistent panel interiors")
        left, right = int(row[a]), int(row[b - 1])
        _require((left == background) != (right == background), "ambiguous side bar")
        color, side = (right, 1) if left == background else (left, -1)
        _require(color not in turns and color not in (frame, background), "conflicting turn key")
        turns[color] = side
    field = g[bottom + 1:]
    other = set(map(int, np.unique(field))) - {frame, background} - set(turns)
    _require(len(other) == 1, "unique unkeyed source color required")
    source = next(iter(other))
    cells = np.argwhere(field == source)
    r, c = map(int, cells.min(axis=0))
    a, b = map(int, cells.max(axis=0) + 1)
    k = a - r
    _require(k == b - c and len(cells) == k * k, "source must be a solid square")
    devices = [_recover(field, color, frame) for color in turns if np.any(field == color)]
    _require(bool(devices), "at least one frame required")
    for i, dev in enumerate(devices):
        t, l, b, e = dev["box"]
        _require(not (t < a and r < b and l < c + k and c < e), "source overlaps frame")
        for prev in devices[:i]:
            y, x, z, v = prev["box"]
            _require(b < y or z < t or e < x or v < l, "frames need a separating background gap")
    return {"field": field, "background": background, "source": source, "scar": frame,
            "source_box": (r, c, k), "turns": turns, "devices": devices,
            "crop": bottom + 1, "panel_widths": tuple(b - a + 2 for a, b in runs)}


def _trace(field, source_box, source, background, turns, devices, counts=None):
    """One-cell advances of a k-square; exhaustion precedes another collision.

    A remaining range of None means the initial unlimited emission. Collisions
    cost no paint and replace the range with the contacted device's room count.
    """
    h, w = field.shape
    sr, sc, k = source_box
    occupied = np.full((h, w), -1, dtype=int)
    for i, dev in enumerate(devices):
        t, l, b, r = dev["box"]
        occupied[t:b, l:r] = i
    output = field.copy()
    events, segments, ends, paths = [], [], [], []
    for branch in range(4):
        r, c, d, remaining = sr, sc, branch, None
        states, path, hits = set(), np.zeros((h, w), dtype=bool), 0
        while True:
            state = (r, c, d, remaining)
            _require(state not in states, "cyclic relay route")
            states.add(state)
            start, distance = (r, c), 0
            while True:
                if remaining == 0:
                    reason, hit = "exhausted", -1
                    break
                if (d == 0 and r == 0) or (d == 1 and c + k == w) or (d == 2 and r + k == h) or (d == 3 and c == 0):
                    reason, hit = "boundary", -1
                    break
                if d == 0:
                    ys, xs = np.full(k, r - 1), np.arange(c, c + k)
                elif d == 1:
                    ys, xs = np.arange(r, r + k), np.full(k, c + k)
                elif d == 2:
                    ys, xs = np.full(k, r + k), np.arange(c, c + k)
                else:
                    ys, xs = np.arange(r, r + k), np.full(k, c - 1)
                front = occupied[ys, xs]
                if np.any(front >= 0):
                    _require(np.all(front == front[0]) and front[0] >= 0, "partial or simultaneous frame collision")
                    hit, reason = int(front[0]), "collision"
                    break
                paint = field[ys, xs] == background
                output[ys[paint], xs[paint]] = source
                path[ys, xs] = True
                dy, dx = DIRS[d]
                r, c = r + dy, c + dx
                distance += 1
                if remaining is not None:
                    remaining -= 1
            segments.append((branch, start[0], start[1], d, distance, hit, reason))
            if hit < 0:
                ends.append((branch, reason, r, c, d))
                break
            dev = devices[hit]
            count = dev["count"] if counts is None else counts[hit]
            side = turns[dev["color"]]
            events.append((branch, hit, d, side, count, hits, r, c, distance))
            hits += 1
            d = (d + side) % 4
            remaining = int(count)
        paths.append(path)
    return output, {"events": events, "segments": segments, "ends": ends, "paths": paths}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive local keys, repaired room counts, thickness and all routes from input."""
    q = _decode(input_grid)
    return _trace(q["field"], q["source_box"], q["source"], q["background"],
                  q["turns"], q["devices"])[0]


def _make_header(rng, width, colors, turns):
    height = int(rng.integers(3, 6))
    sizes = np.full(len(colors), 5, dtype=int)
    sizes += rng.multinomial(width - 5 * len(colors), np.full(len(colors), 1 / len(colors)))
    header = np.zeros((height, width), dtype=np.int8)
    header[[0, -1]] = 8
    c = 0
    for color, size in zip(rng.permutation(colors), sizes):
        header[:, c] = header[:, c + size - 1] = 8
        header[1:-1, c + 1 if turns[int(color)] == -1 else c + size - 2] = color
        c += int(size)
    return header


def _device(rng, box, color, count, style):
    t, l, b, r = box
    mask = _partition(rng, b - t, r - l, count, style)
    _require(len(_rooms(mask)) == count and _euler_count(mask) == count, "constructed topology mismatch")
    return {"box": box, "color": color, "count": count, "mask": mask}


def _finish(rng, shape, source_box, devices, turns, forced_scar=None):
    field = np.zeros(shape, dtype=np.int8)
    sr, sc, k = source_box
    field[sr:sr + k, sc:sc + k] = 1
    for dev in devices:
        t, l, b, r = dev["box"]
        field[t:b, l:r][dev["mask"]] = dev["color"]
    output, _ = _trace(field, source_box, 1, 0, turns, devices)
    dirty = field.copy()
    for dev in devices:
        t, l, b, r = dev["box"]
        mask = dev["mask"]
        sites = []
        for y, x in np.argwhere(mask):
            y, x = int(y), int(x)
            # Keep all corners and junctions; each scar has opposite supports.
            horizontal = 0 < x < r - l - 1 and mask[y, x - 1] and mask[y, x + 1]
            vertical = 0 < y < b - t - 1 and mask[y - 1, x] and mask[y + 1, x]
            neighbors = sum(0 <= y + dy < mask.shape[0] and 0 <= x + dx < mask.shape[1] and mask[y + dy, x + dx]
                            for dy, dx in DIRS)
            if neighbors == 2 and (horizontal or vertical):
                sites.append((t + y, l + x))
        rng.shuffle(sites)
        if forced_scar in sites:
            sites.remove(forced_scar)
            sites.insert(0, forced_scar)
        amount = int(rng.integers(0, max(2, len(sites) // 8 + 1)))
        if forced_scar in sites:
            amount = max(1, amount)
        used = []
        for y, x in sites:
            if len(used) >= amount:
                break
            if any(abs(y - a) + abs(x - b) <= 1 for a, b in used):
                continue
            dirty[y, x] = output[y, x] = 8
            used.append((y, x))
    eligible = dirty == 0
    for dev in devices:
        t, l, b, r = dev["box"]
        eligible[t:b, l:r] = False
    # Background speckles stay background-equivalent to travel and are preserved.
    sites = np.argwhere(eligible)
    if len(sites):
        n = int(rng.integers(0, min(7, len(sites)) + 1))
        for y, x in sites[rng.choice(len(sites), size=n, replace=False)]:
            dirty[y, x] = output[y, x] = 8
    header = _make_header(rng, shape[1], list(turns), turns)
    _require(len(header) + shape[0] <= 30, "input too tall")
    palette = rng.permutation(10).astype(np.int8)
    pair = {"input": palette[np.vstack((header, dirty))], "output": palette[output]}
    q = _decode(pair["input"])
    actual = {d["color"]: d for d in q["devices"]}
    for d in devices:
        recovered = actual[int(palette[d["color"]])]
        _require(np.array_equal(recovered["mask"], d["mask"]), "scar recovery changes walls")
    _require(np.array_equal(solve(pair["input"]), pair["output"]), "input recovery changes route")
    return pair


def _relay_pair(rng, cap, k, rotation, reflected, depth=2, first_count=None, terminal_count=None):
    """Variable-geometry teaching/shift scene with two or three dependent relays."""
    for _ in range(100):
        minimum = k + 3 if depth == 3 else k + 2
        n1 = int(rng.integers(minimum, cap + 1)) if first_count is None else int(first_count)
        n2 = (int(rng.integers(max(3, k + 2 if depth == 3 else 3), cap + 1))
              if terminal_count is None else int(terminal_count))
        heights = list(map(int, rng.integers(5, 10, size=depth)))
        widths = list(map(int, rng.integers(5, 10, size=depth)))
        counts = [n1, n2] + ([int(rng.integers(1, cap + 1))] if depth == 3 else [])
        if any(((h - 1) // 2) * ((w - 1) // 2) < n for h, w, n in zip(heights, widths, counts)):
            continue
        gap1 = (n1 - 1 if first_count is not None else
                int(rng.integers(k + 2 if depth == 3 else k + 1, n1)))
        gap2 = int(rng.integers(k + 1, n2)) if depth == 3 else 0
        sc = int(rng.integers(0, 4))
        initial_gap = widths[1] + int(rng.integers(2, 5))
        xc = sc + initial_gap
        sr = gap1 + max(heights[1], heights[2] - k if depth == 3 else 0) + int(rng.integers(0, 3))
        boxes = [(sr - 1, xc + k, sr - 1 + heights[0], xc + k + widths[0]),
                 (sr - gap1 - heights[1], xc + k - widths[1], sr - gap1, xc + k)]
        if depth == 3:
            y = sr - gap1
            boxes.append((y + k - heights[2], xc + k + gap2, y + k, xc + k + gap2 + widths[2]))
        h = max(b for t, l, b, r in boxes) + int(rng.integers(1, 4))
        w = max(max(r for t, l, b, r in boxes), xc + k + n2 + 1) + int(rng.integers(1, 4))
        if h > 25 or w > 25:  # Either axis can become the input height on rotation.
            continue
        turns = {2: -1, 3: 1}
        if depth == 3:
            turns[4] = -1
        devices = []
        try:
            for i, (box, count) in enumerate(zip(boxes, counts)):
                style = "partial" if i < 2 else str(rng.choice(["partial", "strip", "grid"]))
                devices.append(_device(rng, box, 2 + i, count, style))
        except ValueError:
            continue
        # Transform clean wall masks and source together; chirality follows reflection.
        field = np.zeros((h, w), dtype=np.int8)
        field[sr:sr + k, sc:sc + k] = 1
        for d in devices:
            t, l, b, r = d["box"]
            field[t:b, l:r][d["mask"]] = d["color"]
        scar_mask = np.zeros((h, w), dtype=bool)
        scar_mask[sr, boxes[0][1]] = True
        # The initial struck face explicitly includes a repaired hole. Keep it
        # on a straight segment, rather than deleting a corner or T-junction.
        local_y = sr - boxes[0][0]
        if not (devices[0]["mask"][local_y - 1, 0]
                and devices[0]["mask"][local_y + 1, 0]
                and not devices[0]["mask"][local_y, 1]):
            continue
        if reflected:
            field, scar_mask = np.fliplr(field), np.fliplr(scar_mask)
            turns = {color: -side for color, side in turns.items()}
        field, scar_mask = np.rot90(field, rotation), np.rot90(scar_mask, rotation)
        transformed = []
        for d in devices:
            coords = np.argwhere(field == d["color"])
            t, l = map(int, coords.min(axis=0))
            b, r = map(int, coords.max(axis=0) + 1)
            transformed.append({"box": (t, l, b, r), "color": d["color"], "count": d["count"],
                                "mask": field[t:b, l:r] == d["color"]})
        sr, sc = map(int, np.argwhere(field == 1).min(axis=0))
        # Extra unused keys are locally visible and never change route semantics.
        available = [c for c in range(2, 8) if c not in turns]
        maximum = min(len(available), field.shape[1] // 5 - len(turns))
        for color in available[:int(rng.integers(0, maximum + 1))]:
            turns[color] = int(rng.choice([-1, 1]))
        pair = _finish(rng, field.shape, (sr, sc, k), transformed, turns,
                       tuple(map(int, np.argwhere(scar_mask)[0])))
        q, info = _analyze(pair)
        if max((e[5] + 1 for e in info["events"]), default=0) >= depth:
            return pair
    raise ValueError("exhausted relay scene construction")


def _clear(box, devices, source_box, shape):
    t, l, b, r = box
    sr, sc, k = source_box
    if not (0 <= t < b <= shape[0] and 0 <= l < r <= shape[1]):
        return False
    if t < sr + k and sr < b and l < sc + k and sc < r:
        return False
    return all(b < y or z < t or r < x or v < l for y, x, z, v in (d["box"] for d in devices))


def _random_pair(rng, cap):
    h, w = int(rng.integers(10, 26)), int(rng.integers(12, 31))
    k = int(rng.integers(1, 4))
    source_box = (int(rng.integers(h - k + 1)), int(rng.integers(w - k + 1)), k)
    key_count = int(rng.integers(2, min(6, w // 5) + 1))
    turns = {c: int(rng.choice([-1, 1])) for c in range(2, 2 + key_count)}
    target = int(rng.integers(1, min(5, key_count) + 1))
    devices = []
    field = np.zeros((h, w), dtype=np.int8)
    sr, sc, k = source_box
    field[sr:sr + k, sc:sc + k] = 1
    for _ in range(160):
        if len(devices) >= target:
            break
        _, info = _trace(field, source_box, 1, 0, turns, devices)
        segments = [s for s in info["segments"] if s[4] >= 2]
        if not segments:
            break
        _, y, x, d, length, _, _ = segments[int(rng.integers(len(segments)))]
        distance = int(rng.integers(1, length))
        bh, bw = int(rng.integers(max(3, k), min(10, h) + 1)), int(rng.integers(max(3, k), min(10, w) + 1))
        dy, dx = DIRS[d]
        y, x = y + dy * distance, x + dx * distance
        if d in (0, 2):
            left = x - int(rng.integers(bw - k + 1))
            top = y - bh if d == 0 else y + k
        else:
            top = y - int(rng.integers(bh - k + 1))
            left = x + k if d == 1 else x - bw
        box = (top, left, top + bh, left + bw)
        if not _clear(box, devices, source_box, field.shape):
            continue
        count = int(rng.integers(1, min(cap, ((bh - 1) // 2) * ((bw - 1) // 2)) + 1))
        color = next(c for c in turns if c not in {z["color"] for z in devices})
        try:
            dev = _device(rng, box, color, count, str(rng.choice(["strip", "grid", "partial"])))
            trial = devices + [dev]
            _trace(field, source_box, 1, 0, turns, trial)
        except ValueError:
            continue
        devices = trial
    if not devices:
        return None
    return _finish(rng, field.shape, source_box, devices, turns)


def _analyze(pair):
    q = _decode(pair["input"])
    output, info = _trace(q["field"], q["source_box"], q["source"], q["background"], q["turns"], q["devices"])
    _require(np.array_equal(output, _grid(pair["output"])), "incorrect relay output")
    protected = q["field"] != q["background"]
    _require(np.array_equal(output[protected], q["field"][protected]), "nonbackground cells changed")
    _require(output.shape == q["field"].shape, "incorrect instruction-band crop")
    return q, info


def _evidence(train):
    headings = {-1: set(), 1: set()}
    endpoint_ranges = set()
    features = {"recursive": False, "finite": False, "partial": False, "damage": False,
                "omit_a": False, "omit_b": False, "all_left": False,
                "reversed_key": False, "lattice_shortcut": False}
    for pair in train:
        q, info = _analyze(pair)
        for e in info["events"]:
            headings[e[3]].add(e[2])
            dev = q["devices"][e[1]]
            features["recursive"] |= e[5] >= 1
            features["partial"] |= dev["lattice"] > dev["count"]
            features["damage"] |= dev["damaged"]
        features["finite"] |= any(e[1] == "exhausted" for e in info["ends"])
        for branch, reason, *_ in info["ends"]:
            hits = [e for e in info["events"] if e[0] == branch]
            if reason == "exhausted" and hits:
                endpoint_ranges.add(hits[-1][4])
        arguments = (q["field"], q["source_box"], q["source"], q["background"])
        for name, table, counts in (
            ("omit_a", q["turns"], [1] * len(q["devices"])),
            ("omit_b", {c: 1 for c in q["turns"]}, None),
            ("all_left", {c: -1 for c in q["turns"]}, None),
            ("reversed_key", {c: -s for c, s in q["turns"].items()}, None),
            ("lattice_shortcut", q["turns"], [d["lattice"] for d in q["devices"]])):
            try:
                prediction = _trace(*arguments, table, q["devices"], counts)[0]
                features[name] |= not np.array_equal(prediction, pair["output"])
            except ValueError:
                features[name] = True
    features["turn_headings"] = all(len(v) >= 2 for v in headings.values())
    features["distinct_end_ranges"] = len(endpoint_ranges) >= 2
    return features


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, thicknesses = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            g = _grid(pair["input"])
            _grid(pair["output"])
            key = _signature(g)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            q, info = _analyze(pair)
            analyses[split].append((max(d["count"] for d in q["devices"]),
                                    max((e[5] + 1 for e in info["events"]), default=0)))
            if split == "train":
                thicknesses.add(q["source_box"][2])
    _require(len(thicknesses) >= 2, "training must expose thickness scaling")
    _require(all(_evidence(puzzle["train"]).values()), "missing teaching evidence")
    maxima = tuple(max(a[i] for a in analyses["train"]) for i in (0, 1))
    _require(all(count > maxima[0] or depth > maxima[1] for count, depth in analyses["test"]),
             "test lacks the intended range or relay-depth shift")


def _counts(num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer))
                 and low <= value <= high, name + " is illegal")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    cap, rotation = int(rng.integers(4, 7)), int(rng.integers(4))
    terminal_counts = rng.choice(np.arange(3, cap + 1), size=2, replace=False)
    shift = str(rng.choice(["range", "depth"]))
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            for _ in range(20):
                if split == "train" and i < 2:
                    pair = _relay_pair(rng, cap, i + 1, rotation, i == 1,
                                       terminal_count=int(terminal_counts[i]))
                elif split == "test":
                    if shift == "range":
                        pair = _relay_pair(rng, cap + 1, int(rng.choice([1, 2])), int(rng.integers(4)),
                                           bool(rng.integers(2)), first_count=cap + 1)
                    else:
                        test_k = int(rng.choice([1, 2])) if cap >= 5 else 1
                        pair = _relay_pair(rng, cap, test_k, int(rng.integers(4)),
                                           bool(rng.integers(2)), depth=3)
                else:
                    pair = _random_pair(rng, cap)
                    if pair is None:
                        pair = _relay_pair(rng, cap, int(rng.choice([1, 2])), int(rng.integers(4)), bool(rng.integers(2)))
                    _, info = _analyze(pair)
                    if any(e[5] >= 2 for e in info["events"]):
                        continue
                key = _signature(pair["input"])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise ValueError("could not produce a distinct input")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded 42-seed semantic/variation audit, plus all legal explicit counts."""
    records, signatures, shapes, topologies, palettes = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_signature(p["input"]) for s in puzzle for p in puzzle[s]))
        for split in puzzle:
            for pair in puzzle[split]:
                q, info = _analyze(pair)
                shapes.add(q["field"].shape)
                palettes.add((q["background"], q["source"], q["scar"]))
                counts = tuple(sorted(d["count"] for d in q["devices"]))
                records.append((len(q["devices"]), q["source_box"][2], counts,
                                max((e[5] + 1 for e in info["events"]), default=0)))
                for d in q["devices"]:
                    mask = d["mask"]
                    ys = sorted({x for r in d["rooms"] for x in (r[0] - 1, r[2] + 1)})
                    xs = sorted({x for r in d["rooms"] for x in (r[1] - 1, r[3] + 1)})
                    topologies.add(tuple(tuple(int(mask[y, x]) for x in xs) for y in ys)
                                   + tuple((ys.index(a - 1), xs.index(b - 1), ys.index(c + 1), xs.index(e + 1))
                                           for a, b, c, e in d["rooms"]))
                if seed < 3:
                    # Every role may be permuted, including zero. Geometry is unchanged.
                    perm = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
                    assert np.array_equal(solve(perm[pair["input"]]), perm[pair["output"]])
                    assert np.array_equal(solve(pair["input"].tolist()), pair["output"])
        if seed < 2:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(a[n], b[n]) for s in puzzle for a, b in zip(puzzle[s], repeat[s]) for n in a)
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(palettes) >= 40 and len(topologies) >= 20
    assert len({r[2] for r in records}) >= 20
    assert {1, 2, 3} <= {r[1] for r in records}
    assert {1, 2, 3, 4} <= {r[0] for r in records}
    assert any(r[3] >= 3 for r in records)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(1000 + nt * 10 + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Exact exhaustion: gap=count is NOT recharged; gap=count-1 IS recharged.
    field = np.zeros((16, 16), dtype=np.int8)
    source_box = (8, 1, 1)
    field[8, 1] = 1
    first = {"box": (7, 6, 12, 11), "color": 2, "count": 4}
    for gap, expected_hits in ((4, 1), (3, 2)):
        second = {"box": (8 - gap - 3, 2, 8 - gap, 6), "color": 3, "count": 2}
        _, info = _trace(field, source_box, 1, 0, {2: -1, 3: 1}, [first, second])
        assert sum(e[0] == 1 for e in info["events"]) == expected_hits
    # Exclude ambiguous partial-width faces and routes that recharge forever.
    negatives = [((4, 4, 2), [{"box": (1, 4, 3, 5), "color": 2, "count": 4}])]
    cycle = [{"box": box, "color": 2, "count": 4} for box in
             ((4, 7, 5, 8), (7, 6, 8, 7), (6, 3, 7, 4), (3, 4, 4, 5))]
    negatives.append(((4, 4, 1), cycle))
    for box, devices in negatives:
        try:
            _trace(np.zeros((10, 10), dtype=np.int8), box, 1, 0, {2: 1}, devices)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous or cyclic geometry accepted")
    return {"seeds": 42, "pairs": len(records), "field_shapes": len(shapes),
            "partition_topologies": len(topologies), "local_role_palettes": len(palettes)}


if __name__ == "__main__":
    print(self_test())
