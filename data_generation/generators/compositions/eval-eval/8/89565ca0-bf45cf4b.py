"""Echo wallpaper: recovered room-distance parity controls exact substitution.

Only standard-library modules and NumPy are used. No files, parent generators,
latent generation parameters, or demonstrations are consulted by solve.
"""
from collections import deque
import hashlib
import numbers

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid in colors 0..9")
    return a.astype(np.int8, copy=True)


def _counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError("example counts must be integers")
    a, b = int(num_train), int(num_test)
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    return a, b


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _bbox(mask):
    p = np.argwhere(mask)
    if not len(p):
        raise ValueError("empty visible object")
    t, l = map(int, p.min(axis=0))
    b, r = map(int, p.max(axis=0))
    return t, l, b, r


def _crop(a, box):
    t, l, b, r = box
    return a[t:b + 1, l:r + 1].copy()


def _separated(a, b):
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _transforms(a):
    return [f(np.rot90(a, k)) for k in range(4) for f in (lambda x: x, np.fliplr)]


def _asymmetric(a):
    return all(not np.array_equal(a, x) for x in _transforms(a)[1:])


def _connected8(mask):
    p = np.argwhere(mask)
    if not len(p):
        return False
    start = tuple(map(int, p[0]))
    seen, stack = {start}, [start]
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                q = y + dy, x + dx
                if (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                        and mask[q] and q not in seen):
                    seen.add(q)
                    stack.append(q)
    return len(seen) == int(mask.sum())


def _flood(walls):
    """Room labels from pixels; the intact outer border excludes the exterior."""
    labels = np.full(walls.shape, -1, dtype=np.int16)
    rooms = []
    for yy, xx in np.argwhere(~walls):
        y, x = int(yy), int(xx)
        if labels[y, x] >= 0:
            continue
        i = len(rooms)
        labels[y, x] = i
        cells, stack = [], [(y, x)]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for s, t in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= s < walls.shape[0] and 0 <= t < walls.shape[1]
                        and not walls[s, t] and labels[s, t] < 0):
                    labels[s, t] = i
                    stack.append((s, t))
        rooms.append(cells)
    return labels, rooms


def _pixel_graph(walls, labels, n):
    graph = [set() for _ in range(n)]
    interfaces = []
    for yy, xx in np.argwhere(walls[1:-1, 1:-1]):
        y, x = int(yy) + 1, int(xx) + 1
        for a, b in ((labels[y - 1, x], labels[y + 1, x]),
                     (labels[y, x - 1], labels[y, x + 1])):
            a, b = int(a), int(b)
            if a >= 0 and b >= 0 and a != b:
                graph[a].add(b)
                graph[b].add(a)
                interfaces.append((y, x, a, b))
    return graph, interfaces


def _distances(graph, root):
    distance = [-1] * len(graph)
    distance[root] = 0
    queue = deque([root])
    while queue:
        a = queue.popleft()
        for b in sorted(graph[a]):
            if distance[b] < 0:
                distance[b] = distance[a] + 1
                queue.append(b)
    if -1 in distance:
        raise ValueError("room adjacency must be connected")
    return np.asarray(distance, dtype=np.int16)


def _selection(labels, distances):
    selected = np.zeros(labels.shape, dtype=bool)
    floor = labels >= 0
    selected[floor] = distances[labels[floor]] % 2 == 0
    return selected[1:-1, 1:-1]


def _evaluate(walls, source):
    labels, rooms = _flood(walls)
    if labels[source] < 0:
        raise ValueError("source must be on floor")
    graph, interfaces = _pixel_graph(walls, labels, len(rooms))
    distances = _distances(graph, int(labels[source]))
    return labels, rooms, graph, interfaces, distances, _selection(labels, distances)


def _read_input(input_grid):
    """Require a unique interpretation from local geometry and color roles."""
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if counts.max() <= grid.size / 2:
        raise ValueError("background must be a strict majority")
    frequency = dict(zip(map(int, colors), map(int, counts)))
    candidates = []
    for w in map(int, colors):
        if w == bg:
            continue
        box = _bbox(grid == w)
        plan = _crop(grid, box)
        h, width = plan.shape
        if not (3 <= min(h, width) and max(h, width) <= 17 and max(h, width) >= 7):
            continue
        if not all(np.all(side == w) for side in (plan[0], plan[-1], plan[:, 0], plan[:, -1])):
            continue
        inside = set(map(int, np.unique(plan[1:-1, 1:-1]))) - {bg, w}
        if len(inside) != 2:
            continue
        singleton = [c for c in inside if frequency[c] == 1]
        if len(singleton) != 1:
            continue
        marker = singleton[0]
        noise = next(iter(inside - {marker}))
        if frequency[noise] < 2 or np.count_nonzero(plan == noise) != frequency[noise]:
            continue
        payload = set(map(int, colors)) - {bg, w, marker, noise}
        if not 2 <= len(payload) <= 4:
            continue
        stamp_box = _bbox(np.isin(grid, sorted(payload)))
        stamp = _crop(grid, stamp_box)
        if not _separated(box, stamp_box) or not all(2 <= n <= 5 for n in stamp.shape):
            continue
        if not set(map(int, np.unique(stamp))) <= payload | {bg}:
            continue
        visible = stamp != bg
        if not all(np.any(side) for side in (visible[0], visible[-1], visible[:, 0], visible[:, -1])):
            continue
        if not _connected8(visible):
            continue
        outside = np.ones(grid.shape, dtype=bool)
        t, l, b, r = box
        outside[t:b + 1, l:r + 1] = False
        t, l, b, r = stamp_box
        outside[t:b + 1, l:r + 1] = False
        if np.any(grid[outside] != bg):
            continue
        walls = plan == w
        repaired = []
        for yy, xx in np.argwhere(plan == noise):
            y, x = int(yy), int(xx)
            if not (0 < y < h - 1 and 0 < x < width - 1):
                break
            if ((plan[y - 1, x] == w and plan[y + 1, x] == w)
                    or (plan[y, x - 1] == w and plan[y, x + 1] == w)):
                walls[y, x] = True
                repaired.append((y, x))
        else:
            source = tuple(map(int, np.argwhere(plan == marker)[0]))
            try:
                labels, rooms, graph, interfaces, ds, mask = _evaluate(walls, source)
            except ValueError:
                continue
            if not 3 <= len(rooms) <= 10 or not repaired:
                continue
            # The intended family has rectangular elementary rooms, not open
            # corridors or nonrectangular remnants of incomplete reconstruction.
            rectangular = True
            for cells in rooms:
                p = np.asarray(cells)
                lo, hi = p.min(axis=0), p.max(axis=0)
                rectangular &= len(cells) == int(np.prod(hi - lo + 1))
            if not rectangular:
                continue
            if mask.shape[0] * stamp.shape[0] > 30 or mask.shape[1] * stamp.shape[1] > 30:
                continue
            candidates.append(dict(background=bg, wall=w, marker=marker, noise=noise,
                                   plan=plan, plan_box=box, stamp=stamp, stamp_box=stamp_box,
                                   walls=walls, labels=labels, rooms=rooms, graph=graph,
                                   interfaces=interfaces, distances=ds, source=source,
                                   repaired=repaired, mask=mask))
    if len(candidates) != 1:
        raise ValueError("input does not have one unambiguous floor-plan reading")
    return candidates[0]


def _expand(mask, stamp, background):
    """Cell indexing used for input-only prediction and validation."""
    sh, sw = stamp.shape
    r, c = np.indices((mask.shape[0] * sh, mask.shape[1] * sw))
    return np.where(mask[r // sh, c // sw], stamp[r % sh, c % sw], background).astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    return _expand(info["mask"], info["stamp"], info["background"])


def _write_blocks(mask, stamp, background):
    """Generator ground truth uses block writes rather than modulo indexing."""
    sh, sw = stamp.shape
    result = np.full((mask.shape[0] * sh, mask.shape[1] * sw), background, dtype=np.int8)
    for r, c in np.argwhere(mask):
        result[r * sh:(r + 1) * sh, c * sw:(c + 1) * sw] = stamp
    return result


def _walls_and_labels(shape, boxes):
    walls = np.zeros(shape, dtype=bool)
    labels = np.full(shape, -1, dtype=np.int16)
    for i, (t, l, b, r) in enumerate(boxes):
        walls[t, l:r + 1] = walls[b, l:r + 1] = True
        walls[t:b + 1, l] = walls[t:b + 1, r] = True
        labels[t + 1:b, l + 1:r] = i
    assert np.array_equal(labels < 0, walls)
    return walls, labels


def _box_graph(boxes):
    """Generation adjacency uses geometric shared boundary intervals."""
    graph = [set() for _ in boxes]
    for i, (t, l, b, r) in enumerate(boxes):
        for j in range(i):
            tt, ll, bb, rr = boxes[j]
            contact = ((r == ll or rr == l) and min(b, bb) - max(t, tt) >= 2)
            contact |= ((b == tt or bb == t) and min(r, rr) - max(l, ll) >= 2)
            if contact:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _capacity(box):
    t, l, b, r = box
    return ((b - t) // 2) * ((r - l) // 2)


def _offsets(n, span, rng):
    gaps = np.full(n, 2, dtype=int)
    for _ in range(span - 2 * n):
        gaps[int(rng.integers(n))] += 1
    return [0] + list(map(int, np.cumsum(gaps)))


def _partition(rng, shape, count, style):
    h, w = shape
    outer = (0, 0, h - 1, w - 1)
    if _capacity(outer) < count:
        return None
    if style in ("horizontal", "vertical", "grid"):
        factors = [(a, count // a) for a in range(1, count + 1)
                   if count % a == 0 and 2 * a < h and 2 * (count // a) < w
                   and (style == "grid" or (style == "horizontal" and a == 1)
                        or (style == "vertical" and a == count))]
        if not factors:
            return None
        nr, nc = factors[int(rng.integers(len(factors)))]
        ys, xs = _offsets(nr, h - 1, rng), _offsets(nc, w - 1, rng)
        return [(ys[a], xs[b], ys[a + 1], xs[b + 1]) for a in range(nr) for b in range(nc)]
    boxes = [outer]
    first_axis = None
    for step in range(count - 1):
        options = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, start, end in ((0, t, b), (1, l, r)):
                if step == 1 and axis == first_axis:
                    continue
                for cut in range(start + 2, end - 1):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    capacity = sum(_capacity(x) for j, x in enumerate(boxes) if j != i)
                    if capacity + sum(_capacity(x) for x in halves) >= count:
                        options.append((i, axis, halves))
        if not options:
            return None
        i, axis, halves = options[int(rng.integers(len(options)))]
        if step == 0:
            first_axis = axis
        boxes[i:i + 1] = halves
    return boxes


def _partial(walls):
    hs = np.flatnonzero(np.any(walls[:, :-1] & walls[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(walls[:-1, :] & walls[1:, :], axis=0))
    return not (all(walls[y].all() for y in hs) and all(walls[:, x].all() for x in vs))


def _lattice_mask(walls, source):
    hs = np.flatnonzero(np.any(walls[:, :-1] & walls[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(walls[:-1, :] & walls[1:, :], axis=0))
    extended = walls.copy()
    extended[hs, :] = True
    extended[:, vs] = True
    if extended[source]:
        return np.zeros((walls.shape[0] - 2, walls.shape[1] - 2), dtype=bool)
    # Staggered lines can become adjacent after extension. The alternative
    # counts full line coordinates crossed, including such adjacent lines.
    rr, cc = np.indices(walls.shape)
    dy = np.abs(np.searchsorted(hs, rr) - np.searchsorted(hs, source[0]))
    dx = np.abs(np.searchsorted(vs, cc) - np.searchsorted(vs, source[1]))
    return ((~extended) & ((dy + dx) % 2 == 0))[1:-1, 1:-1]


def _shape_options(count, stamp_shape, style):
    sh, sw = stamp_shape
    return [(h, w) for h in range(3, min(17, 30 // sh + 2) + 1)
            for w in range(3, min(17, 30 // sw + 2) + 1)
            if max(h, w) >= 7 and ((h - 1) // 2) * ((w - 1) // 2) >= count
            and (style != "horizontal" or (w >= 2 * count + 1 and h >= 4))
            and (style != "vertical" or (h >= 2 * count + 1 and w >= 4))
            and (style != "partial" or min(h, w) >= 5)]


def _stamp(rng, shape, bg, payload, witness):
    h, w = shape
    rr, cc = np.indices(shape)
    for _ in range(80):
        kind = "random" if witness else str(rng.choice(["random", "bands", "rings", "diagonal", "symmetric"]))
        if kind == "bands":
            labels = (rr if rng.integers(2) else cc) + int(rng.integers(len(payload)))
        elif kind == "rings":
            labels = np.minimum.reduce([rr, cc, h - 1 - rr, w - 1 - cc])
        elif kind == "diagonal":
            labels = rr + cc
        elif kind == "symmetric":
            labels = np.minimum(rr, h - 1 - rr) + np.minimum(cc, w - 1 - cc)
        else:
            labels = rng.integers(len(payload), size=shape)
        stamp = payload[labels % len(payload)].copy()
        if witness or rng.random() < .55:
            amount = int(rng.integers(1, max(2, stamp.size // 4 + 1)))
            stamp.flat[rng.choice(stamp.size, size=amount, replace=False)] = bg
        foreground = stamp != bg
        if (len(set(map(int, np.unique(stamp))) - {bg}) < 2
                or not all(np.any(side) for side in (foreground[0], foreground[-1], foreground[:, 0], foreground[:, -1]))
                or not _connected8(foreground)):
            continue
        if witness and (not _asymmetric(stamp) or not np.any(stamp == bg)):
            continue
        return stamp
    stamp = np.full(shape, payload[0], dtype=np.int8)
    stamp[0, 0] = payload[1]
    if witness:
        stamp[1, 0] = bg
        stamp[0, 1] = payload[-1]
        assert _asymmetric(stamp)
    return stamp


def _damage(rng, walls, labels, distance, source, bg, wall_color, marker, noise):
    graph, interfaces = _pixel_graph(walls, labels, len(distance))
    del graph
    # Only straight internal segments, away from corners and junctions.
    candidates = []
    for y, x, a, b in interfaces:
        neighbors = [walls[y - 1, x], walls[y + 1, x], walls[y, x - 1], walls[y, x + 1]]
        if sum(map(int, neighbors)) == 2:
            candidates.append((y, x, a, b))
    first = [p for p in candidates if distance[p[2]] % 2 != distance[p[3]] % 2]
    if not first:
        return None
    chosen = [first[int(rng.integers(len(first)))][:2]]
    target = int(rng.integers(2, 5))
    for i in rng.permutation(len(candidates)):
        y, x, _, _ = candidates[int(i)]
        if all(abs(y - a) + abs(x - b) > 1 for a, b in chosen):
            chosen.append((y, x))
            if len(chosen) >= target:
                break
    plan = np.where(walls, wall_color, bg).astype(np.int8)
    for p in chosen:
        plan[p] = noise
    # Floor smudges are ignored, and cannot be mistaken for erased walls.
    floor_sites = []
    for yy, xx in np.argwhere(~walls):
        y, x = int(yy), int(xx)
        if ((y, x) != source and not (walls[y - 1, x] and walls[y + 1, x])
                and not (walls[y, x - 1] and walls[y, x + 1])):
            floor_sites.append((y, x))
    amount = max(2 - len(chosen), int(rng.integers(0, 4)))
    if len(floor_sites) < amount:
        return None
    for i in rng.choice(len(floor_sites), size=amount, replace=False):
        plan[floor_sites[int(i)]] = noise
    plan[source] = marker
    if np.count_nonzero(plan == noise) < 2:
        return None
    return plan


def _compose(rng, plan, stamp, bg):
    ph, pw = plan.shape
    sh, sw = stamp.shape
    horizontal, reverse = bool(rng.integers(2)), bool(rng.integers(2))
    gap = int(rng.integers(1, 5))
    h = (max(ph, sh) if horizontal else ph + sh + gap) + int(rng.integers(0, 4))
    w = (pw + sw + gap if horizontal else max(pw, sw)) + int(rng.integers(0, 4))
    h, w = min(h, 30), min(w, 30)
    foreground = int(np.count_nonzero(plan != bg) + np.count_nonzero(stamp != bg))
    while h * w <= 2 * foreground:
        if h < 30 and (h <= w or w == 30):
            h += 1
        elif w < 30:
            w += 1
        else:
            raise ValueError("cannot make background evident")
    if horizontal:
        start = int(rng.integers(w - pw - sw - gap + 1))
        pc, sc = (start + sw + gap, start) if reverse else (start, start + pw + gap)
        pr, sr = int(rng.integers(h - ph + 1)), int(rng.integers(h - sh + 1))
    else:
        start = int(rng.integers(h - ph - sh - gap + 1))
        pr, sr = (start + sh + gap, start) if reverse else (start, start + ph + gap)
        pc, sc = int(rng.integers(w - pw + 1)), int(rng.integers(w - sw + 1))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[pr:pr + ph, pc:pc + pw] = plan
    grid[sr:sr + sh, sc:sc + sw] = stamp
    return grid


def _make_pair(rng, parameters, count, style, witness=None, previous_stamp=None, grow_stamp=False):
    palette = parameters["palette"] if parameters["stable"] else rng.permutation(10)
    bg, wall, marker, noise = map(int, palette[:4])
    k = 3 if parameters["stable"] else int(rng.integers(2, 5))
    payload = np.asarray(palette[4:4 + k], dtype=np.int8)
    stamp_options = [(h, w) for h in range(2, 6 if grow_stamp else 4)
                     for w in range(2, 6 if grow_stamp else 4)
                     if (not grow_stamp or max(h, w) >= 4)
                     and (not witness or h * w >= 6)
                     and (previous_stamp is None or (h, w) != previous_stamp)
                     and _shape_options(count, (h, w), style)]
    if not stamp_options:
        raise ValueError("no compatible plan and stamp dimensions")
    for _ in range(240):
        stamp_shape = stamp_options[int(rng.integers(len(stamp_options)))]
        shapes = _shape_options(count, stamp_shape, style)
        shape = shapes[int(rng.integers(len(shapes)))]
        boxes = _partition(rng, shape, count, style)
        if boxes is None:
            continue
        walls, labels = _walls_and_labels(shape, boxes)
        graph = _box_graph(boxes)
        pixel_graph, _ = _pixel_graph(walls, labels, count)
        assert pixel_graph == graph, "rectangle and pixel adjacency disagree"
        if style == "partial" and not _partial(walls):
            continue
        roots = [0, count - 1] if witness == "strip" else list(map(int, rng.permutation(count)))
        rng.shuffle(roots)
        picked = None
        for root in roots:
            distance = _distances(graph, root)
            mask = _selection(labels, distance)
            positions = np.argwhere(labels == root)
            source = tuple(map(int, positions[int(rng.integers(len(positions)))]))
            if witness == "partial":
                same = any(distance[a] % 2 == distance[b] % 2 for a in range(count) for b in graph[a])
                lattice = _lattice_mask(walls, source)
                if (max(distance) < 2 or not same or not _asymmetric(mask)
                        or not lattice.any() or np.array_equal(mask, lattice)):
                    continue
            picked = distance, mask, source
            break
        if picked is None:
            continue
        distance, mask, source = picked
        plan = _damage(rng, walls, labels, distance, source, bg, wall, marker, noise)
        if plan is None:
            continue
        stamp = _stamp(rng, stamp_shape, bg, payload, bool(witness))
        grid = _compose(rng, plan, stamp, bg)
        try:
            info = _read_input(grid)
        except ValueError:
            continue
        # Exact construction/recovery comparison, not merely equal room counts.
        if not np.array_equal(info["walls"], walls) or not np.array_equal(info["mask"], mask):
            continue
        output = _write_blocks(mask, stamp, bg)
        assert np.array_equal(solve(grid), output)
        return {"input": grid, "output": output}
    raise RuntimeError("bounded construction could not find a readable pair")


def _is_strip(info):
    walls = info["walls"]
    hs = np.flatnonzero(np.any(walls[:, :-1] & walls[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(walls[:-1, :] & walls[1:, :], axis=0))
    return len(hs) == 2 or len(vs) == 2


def _counterfactual_masks(info):
    labels, ds = info["labels"], info["distances"]
    root = int(labels[info["source"]])
    def select(values):
        result = np.zeros(labels.shape, dtype=bool)
        floor = labels >= 0
        result[floor] = np.asarray(values)[labels[floor]]
        return result[1:-1, 1:-1]
    return {
        "source_only": select(np.arange(len(ds)) == root),
        "all_rooms": select(np.ones(len(ds), dtype=bool)),
        "one_step": select(ds <= 1),
        "odd_distance": select(ds % 2 == 1),
        "unrepaired": _evaluate(info["plan"] == info["wall"], info["source"])[-1],
        "full_lattice": _lattice_mask(info["walls"], info["source"]),
    }


def _orientation_guard(information, puzzle):
    # Finite alternatives only; this is not a blind or independent solver.
    candidates = []
    for mi in range(8):
        for si in range(8):
            for info, pair in zip(information["train"], puzzle["train"]):
                mask, stamp = _transforms(info["mask"])[mi], _transforms(info["stamp"])[si]
                shape = mask.shape[0] * stamp.shape[0], mask.shape[1] * stamp.shape[1]
                if shape != np.asarray(pair["output"]).shape:
                    break
                if not np.array_equal(_expand(mask, stamp, info["background"]), pair["output"]):
                    break
            else:
                candidates.append((mi, si))
    assert (0, 0) in candidates
    for mi, si in candidates:
        for info, pair in zip(information["test"], puzzle["test"]):
            assert np.array_equal(_expand(_transforms(info["mask"])[mi],
                                         _transforms(info["stamp"])[si], info["background"]), pair["output"]), "orientation not established"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            key = _key(grid)
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _read_input(grid)
            assert np.array_equal(output, _expand(info["mask"], info["stamp"], info["background"])), "incorrect wallpaper"
            assert max(info["distances"]) >= 1
            assert info["mask"].any() and not info["mask"].all()
            assert not np.any(np.isin(output, [info["wall"], info["marker"], info["noise"]]))
            active = int(info["mask"].sum())
            for color in set(map(int, np.unique(info["stamp"]))) - {info["background"]}:
                assert np.count_nonzero(output == color) == active * np.count_nonzero(info["stamp"] == color)
            # Each claimed wall repair actually has opposing visible support.
            for y, x in info["repaired"]:
                assert info["labels"][y, x] == -1
            information[split].append(info)
    train, test = information["train"], information["test"]
    strip_witness = partial_witness = False
    rejected = set()
    for info in train:
        ds, graph = info["distances"], info["graph"]
        witness_stamp = _asymmetric(info["stamp"]) and np.any(info["stamp"] == info["background"])
        strip_witness |= (_is_strip(info) and 4 <= len(ds) <= 5 and max(ds) == len(ds) - 1 and witness_stamp)
        same = any(ds[a] % 2 == ds[b] % 2 for a in range(len(ds)) for b in graph[a])
        alternatives = _counterfactual_masks(info)
        partial_witness |= (_partial(info["walls"]) and max(ds) >= 2 and same
                            and _asymmetric(info["mask"]) and witness_stamp
                            and alternatives["full_lattice"].any()
                            and not np.array_equal(info["mask"], alternatives["full_lattice"]))
        rejected.update(name for name, mask in alternatives.items() if not np.array_equal(mask, info["mask"]))
        # B ablation keeps the output shape but erases the displayed fine detail.
        payload = info["stamp"][info["stamp"] != info["background"]]
        solid = np.full(info["stamp"].shape, np.bincount(payload).argmax(), dtype=np.int8)
        assert not np.array_equal(_expand(info["mask"], solid, info["background"]),
                                  _expand(info["mask"], info["stamp"], info["background"]))
    assert strip_witness and partial_witness, "missing topology or exact-copy teaching witness"
    assert rejected == {"source_only", "all_rooms", "one_step", "odd_distance", "unrepaired", "full_lattice"}
    assert len({i["stamp"].shape for i in train}) >= 2
    maximum = max(len(i["rooms"]) for i in train)
    assert maximum <= 6
    long_strip = all(_is_strip(i) and maximum < len(i["rooms"]) <= 8 for i in test)
    recombine = all(_partial(i["walls"]) and 7 <= len(i["rooms"]) <= 10 for i in test)
    stamp_grow = (max(max(i["stamp"].shape) for i in train) <= 3
                  and all(max(i["stamp"].shape) >= 4 for i in test))
    assert long_strip or recombine or stamp_grow, "missing declared structural train/test shift"
    _orientation_guard(information, puzzle)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    parameters = {"stable": bool(rng.integers(2)), "palette": rng.permutation(10),
                  "shift": str(rng.choice(["long_strip", "room_recombine", "stamp_grow"]))}
    puzzle, seen = {"train": [], "test": []}, set()
    previous = None
    for i in range(nt):
        witness = "strip" if i == 0 else "partial" if i == 1 else None
        count = int(rng.integers(4, 6) if i == 0 else rng.integers(5, 7) if i == 1 else rng.integers(3, 7))
        style = str(rng.choice(["horizontal", "vertical"])) if i == 0 else "partial" if i == 1 else str(rng.choice(["horizontal", "vertical", "grid", "partial", "partial"]))
        for _ in range(12):
            pair = _make_pair(rng, parameters, count, style, witness, previous)
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("duplicate inputs after bounded retries")
        seen.add(_key(pair["input"]))
        puzzle["train"].append(pair)
        if i == 0:
            previous = _read_input(pair["input"])["stamp"].shape
        elif i == 1:
            previous = None
    maximum = max(len(_read_input(p["input"])["rooms"]) for p in puzzle["train"])
    for _ in range(ne):
        shift = parameters["shift"]
        if shift == "long_strip":
            count, style = int(rng.integers(maximum + 1, 9)), str(rng.choice(["horizontal", "vertical"]))
        elif shift == "room_recombine":
            count, style = int(rng.integers(7, 11)), "partial"
        else:
            count, style = int(rng.integers(3, 7)), "partial"
        for _ in range(12):
            pair = _make_pair(rng, parameters, count, style, grow_stamp=shift == "stamp_grow")
            if _key(pair["input"]) not in seen:
                break
        else:
            raise RuntimeError("duplicate inputs after bounded retries")
        seen.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _topology_signature(info):
    walls = info["walls"]
    ys = np.flatnonzero(np.any(walls[:, :-1] & walls[:, 1:], axis=1))
    xs = np.flatnonzero(np.any(walls[:-1, :] & walls[1:, :], axis=0))
    # Normalize numeric spacing, while preserving segment endpoint order.
    rows = sorted(set(map(int, ys)) | {int(y + 1) for y in ys[:-1]})
    cols = sorted(set(map(int, xs)) | {int(x + 1) for x in xs[:-1]})
    return _key(walls[np.ix_(rows, cols)])


def self_test():
    """Bounded, sequential 42-seed audit; retains aggregate signatures only."""
    shapes, stamp_shapes, topologies, graphs, outputs, palettes = (set() for _ in range(6))
    fingerprints, counts, layouts = set(), set(), set()
    pairs = partial = floor_noise = holes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read_input(pair["input"])
                shapes.add(pair["input"].shape)
                stamp_shapes.add(info["stamp"].shape)
                topologies.add(_topology_signature(info))
                graphs.add((tuple(tuple(sorted(n)) for n in info["graph"]), tuple(map(int, info["distances"]))))
                outputs.add(pair["output"].shape)
                counts.add(len(info["rooms"]))
                palettes.add((info["background"], info["wall"], info["marker"], info["noise"]))
                a, b = info["plan_box"], info["stamp_box"]
                layouts.add("left" if a[3] < b[1] else "right" if b[3] < a[1] else "above" if a[2] < b[0] else "below")
                partial += int(_partial(info["walls"]))
                floor_noise += int(np.any((info["plan"] == info["noise"]) & ~info["walls"]))
                holes += int(np.any(info["stamp"] == info["background"]))
                pairs += 1
                for key in ("input", "output"):
                    digest.update(bytes(pair[key].shape))
                    digest.update(pair[key].tobytes())
                if seed < 3:
                    # Geometric and palette covariance are independent useful
                    # invariants of this spatial, role-based rule.
                    for transformed_input, transformed_output in zip(_transforms(pair["input"]), _transforms(pair["output"])):
                        assert np.array_equal(solve(transformed_input), transformed_output)
                    colors = np.asarray([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
                    assert np.array_equal(solve(colors[pair["input"]]), colors[pair["output"]])
        fingerprints.add(digest.hexdigest())
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(stamp_shapes) >= 10
    assert len(topologies) >= 40 and len(graphs) >= 40 and len(outputs) >= 30
    assert {3, 4, 5, 6, 7, 8, 9, 10} <= counts
    assert len(palettes) >= 20 and layouts == {"left", "right", "above", "below"}
    assert partial >= 70 and floor_noise >= 20 and holes >= 84
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(_key(x[k]) == _key(y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4))):
        validate_generated_puzzle(generate_puzzle(1000 + i, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for bad_seed in (-1, True, 0.5):
        try:
            generate_puzzle(bad_seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
    return {"seeds": 42, "pairs": pairs, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "stamp_shapes": len(stamp_shapes),
            "spacing_normalized_topologies": len(topologies), "rooted_graphs": len(graphs),
            "output_shapes": len(outputs), "room_counts": sorted(counts),
            "partial_pairs": partial, "pairs_with_floor_smudges": floor_noise,
            "pairs_with_stamp_holes": holes, "role_palettes": len(palettes),
            "layouts": sorted(layouts), "independent_solver": False, "blind_prediction": False}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
