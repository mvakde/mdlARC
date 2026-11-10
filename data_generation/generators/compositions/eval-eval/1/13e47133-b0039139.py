"""Census-weighted contours. Standard library and NumPy only."""
from collections import deque

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= x <= 30 for x in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return g


def _components(mask, diagonal=False):
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros_like(mask)
    groups = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        group, queue = [], deque([(r, c)])
        while queue:
            x, y = queue.popleft()
            group.append((x, y))
            for dr, dc in (DIR8 if diagonal else DIR4):
                a, b = x + dr, y + dc
                if 0 <= a < h and 0 <= b < w and mask[a, b] and not seen[a, b]:
                    seen[a, b] = True
                    queue.append((a, b))
        groups.append(group)
    return groups


def _masks(mask):
    result = []
    for group in _components(mask):
        room = np.zeros_like(mask, dtype=bool)
        rr, cc = zip(*group)
        room[rr, cc] = True
        result.append(room)
    return result


def _erosion_depth(mask, diagonal=True):
    """Repeated erosion, preserving the depth of the original room."""
    depth = np.full(mask.shape, -1, dtype=np.int16)
    current = mask.copy()
    h, w = mask.shape
    level = 0
    while current.any():
        p = np.pad(current, 1, constant_values=False)
        interior = current.copy()
        for dr, dc in (DIR8 if diagonal else DIR4):
            interior &= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[current & ~interior] = level
        current = interior
        level += 1
    return depth


def _distance_depth(mask):
    """A second internal depth check: BFS from eight-neighbor boundary cells."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                    and mask[r + dr, c + dc]) for dr, dc in DIR8):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR8:
            a, b = r + dr, c + dc
            if 0 <= a < h and 0 <= b < w and mask[a, b] and depth[a, b] < 0:
                depth[a, b] = int(depth[r, c]) + 1
                queue.append((a, b))
    return depth


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _paths(mask, depth):
    paths = []
    for r, c in np.argwhere(mask & (depth == 0)):
        r, c = int(r), int(c)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if (_inside(mask, r - dr, c) or _inside(mask, r, c - dc)
                        or not all(_inside(mask, x, y) for x, y in
                                   ((r + dr, c), (r, c + dc), (r + dr, c + dc)))):
                    continue
                n = 0
                while _inside(mask, r + n * dr, c + n * dc):
                    if depth[r + n * dr, c + n * dc] != n:
                        break
                    n += 1
                paths.append((r, c, dr, dc, n))
    return paths


def _decode_key(frame, mask, depth, neutral):
    cues = [tuple(map(int, p)) for p in np.argwhere(mask & (frame != neutral))]
    if not cues:
        return (), (), ()
    k = len(cues)
    if not 1 <= k <= 4 or len({int(frame[p]) for p in cues}) != k:
        raise ValueError("key must have 1..4 distinct explicit colors")
    if sorted(int(depth[p]) for p in cues) != list(range(k)):
        raise ValueError("key cues must occupy consecutive contour depths")
    candidates = []
    for r, c, dr, dc, capacity in _paths(mask, depth):
        points = tuple((r + d * dr, c + d * dc) for d in range(k))
        if capacity >= k and set(points) == set(cues):
            candidates.append((tuple(int(frame[p]) for p in points), points, (dr, dc)))
    if not candidates or len({a[0] for a in candidates}) != 1:
        raise ValueError("key has no unique ordered corner interpretation")
    return candidates[0][0], candidates[0][1], tuple(sorted({a[2] for a in candidates}))


def _solid_square(mask):
    return bool(np.any(mask[:-1, :-1] & mask[1:, :-1]
                       & mask[:-1, 1:] & mask[1:, 1:]))


def _parse_candidate(g, neutral, wall, box):
    a, b, c, d = box
    frame = g[a:b + 1, c:d + 1].copy()
    walls = frame == wall
    if len(_components(walls)) != 1 or _solid_square(walls):
        raise ValueError("wall network must be connected and one cell wide")
    outside = np.ones(g.shape, dtype=bool)
    outside[a:b + 1, c:d + 1] = False
    ink = outside & (g != neutral)
    # A neutral cell separates every control from the frame, including diagonally.
    if np.any(ink[max(0, a - 1):b + 2, max(0, c - 1):d + 2]):
        raise ValueError("control touches frame")
    counts, areas, objects = {}, {}, []
    for color in sorted(map(int, np.unique(g[ink]))):
        groups = _components(ink & (g == color))
        if color in (neutral, wall) or not 1 <= len(groups) <= 4:
            raise ValueError("invalid census color or component count")
        counts[color] = len(groups)
        areas[color] = sum(map(len, groups))
        objects.extend((color, group) for group in groups)
    if not counts or len(_components(ink, True)) != len(objects):
        raise ValueError("controls must be separated even diagonally")
    rooms, used = [], set()
    for mask in _masks(~walls):
        depth = _erosion_depth(mask)
        key, positions, directions = _decode_key(frame, mask, depth, neutral)
        if not set(key) <= set(counts):
            raise ValueError("key refers to a missing census color")
        used.update(key)
        rooms.append({"mask": mask, "depth": depth, "key": key,
                      "positions": positions, "directions": directions})
    if not rooms or used != set(counts):
        raise ValueError("census and key palettes must match")
    return {"neutral": neutral, "wall": wall, "box": box, "frame": frame,
            "walls": walls, "counts": counts, "areas": areas,
            "objects": objects, "rooms": rooms}


def _decode(input_grid):
    g = _grid(input_grid)
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    neutral = int(border[0])
    if not np.all(border == neutral):
        raise ValueError("input border must identify a uniform neutral canvas")
    candidates = []
    for wall in map(int, np.unique(g)):
        if wall == neutral:
            continue
        points = np.argwhere(g == wall)
        a, c = map(int, points.min(axis=0))
        b, d = map(int, points.max(axis=0))
        if min(b - a + 1, d - c + 1) < 5:
            continue
        if a < 1 or c < 1 or b >= g.shape[0] - 1 or d >= g.shape[1] - 1:
            continue
        if not (np.all(g[a, c:d + 1] == wall) and np.all(g[b, c:d + 1] == wall)
                and np.all(g[a:b + 1, c] == wall) and np.all(g[a:b + 1, d] == wall)):
            continue
        try:
            candidates.append(_parse_candidate(g, neutral, wall, (a, b, c, d)))
        except ValueError:
            continue
    if len(candidates) != 1:
        raise ValueError("input must have exactly one valid enclosing frame")
    return candidates[0]


def _render(info, counts=None, depth_mode="erosion", key_mode="local"):
    out = info["frame"].copy()
    counts = info["counts"] if counts is None else counts
    first_key = next(r["key"] for r in info["rooms"] if r["key"])
    for room in info["rooms"]:
        key = room["key"]
        if not key:
            out[room["mask"]] = info["neutral"]
            continue
        if key_mode == "sorted":
            key = tuple(sorted(key))
        elif key_mode == "global":
            key = first_key
        schedule = np.array([color for color in key for _ in range(counts[color])],
                            dtype=np.int8)
        mask = room["mask"]
        if depth_mode == "distance":
            depth = _distance_depth(mask)[mask]
        elif depth_mode == "manhattan":
            depth = _erosion_depth(mask, False)[mask]
        elif depth_mode == "box":
            rr, cc = np.nonzero(mask)
            depth = np.minimum.reduce((rr - rr.min(), rr.max() - rr,
                                       cc - cc.min(), cc.max() - cc))
        else:
            depth = room["depth"][mask]
        out[mask] = schedule[depth % len(schedule)]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, quantities, key orders, and contours from this input."""
    return _render(_decode(input_grid))


def _add_spur(walls, rng):
    h, w = walls.shape
    candidates = [tuple(map(int, p)) for p in np.argwhere(walls)]
    rng.shuffle(candidates)
    for r, c in candidates[:100]:
        directions = list(DIR4)
        rng.shuffle(directions)
        for dr, dc in directions:
            length = int(rng.integers(2, 6))
            cells = [(r + i * dr, c + i * dc) for i in range(1, length + 1)]
            if not all(1 <= x < h - 1 and 1 <= y < w - 1
                       and not walls[x, y] for x, y in cells):
                continue
            trial = walls.copy()
            for p in cells:
                trial[p] = True
            if not _solid_square(trial) and len(_components(~trial)) <= 6:
                walls[:] = trial
                return


def _jog(walls, rng):
    """Move a clear internal line interval; preserve the outer frame."""
    for _ in range(35):
        work = walls.T if rng.integers(2) else walls
        h, w = work.shape
        if min(h, w) < 9:
            return
        r = int(rng.integers(2, h - 5))
        end = min(h - 3, r + int(rng.integers(3, 8)))
        c = int(rng.integers(2, w - 2))
        shift = int(rng.choice((-1, 1)))
        lo, hi = sorted((c, c + shift))
        block = work[r - 1:end + 2, lo - 1:hi + 2]
        expect = np.zeros_like(block)
        expect[:, c - lo + 1] = True
        if not np.array_equal(block, expect):
            continue
        work[r + 1:end, c] = False
        work[r, lo:hi + 1] = work[end, lo:hi + 1] = True
        work[r:end + 1, c + shift] = True
        return


def _attach_box(walls, rng):
    """A wall-attached loop, with a small interior and a concave outer room."""
    turns = int(rng.integers(4))
    work = np.rot90(walls, turns).copy()
    h, w = work.shape
    if min(h, w) < 10:
        return
    far = int(rng.integers(4, min(8, h - 3)))
    width = int(rng.integers(4, min(9, w - 3)))
    left = int(rng.integers(2, w - width - 1))
    right = left + width
    if work[1:far + 2, left - 1:right + 2].any():
        return
    trial = work.copy()
    trial[:far + 1, left] = trial[:far + 1, right] = True
    trial[far, left:right + 1] = True
    if len(_components(~trial)) <= 6 and not _solid_square(trial):
        walls[:] = np.rot90(trial, -turns)


def _layout(rng, h, w, kind):
    walls = np.zeros((h, w), dtype=bool)
    walls[[0, -1], :] = True
    walls[:, [0, -1]] = True
    if kind == "first":
        side = int(rng.integers(4))
        work = np.rot90(walls, side).copy()
        col = int(rng.integers(5, work.shape[1] - 5))
        length = int(rng.integers(2, 5))
        work[:length + 1, col] = True
        return np.rot90(work, -side).copy()
    if kind == "second":
        shallow = int(rng.integers(3, 6))
        cut = h - 2 - shallow
        walls[cut] = True
        col = int(rng.integers(4, w - 4))
        walls[cut:, col] = True
        _jog(walls, rng)
        if rng.random() < .35:
            _add_spur(walls, rng)
        # Reflection keeps a large room and two shallow rooms without a fixed location.
        if rng.integers(2):
            walls = walls[::-1].copy()
        if rng.integers(2):
            walls = walls[:, ::-1].copy()
        return walls
    rectangles = [(1, h - 2, 1, w - 2)]
    target = int(rng.integers(1, 6))
    while len(rectangles) < target:
        choices = [(i, axis) for i, (a, b, c, d) in enumerate(rectangles)
                   for axis, length in ((0, b - a + 1), (1, d - c + 1)) if length >= 7]
        if not choices:
            break
        i, axis = choices[int(rng.integers(len(choices)))]
        a, b, c, d = rectangles.pop(i)
        if axis == 0:
            q = int(rng.integers(a + 3, b - 2))
            walls[q, c - 1:d + 2] = True
            rectangles.extend(((a, q - 1, c, d), (q + 1, b, c, d)))
        else:
            q = int(rng.integers(c + 3, d - 2))
            walls[a - 1:b + 2, q] = True
            rectangles.extend(((a, b, c, q - 1), (a, b, q + 1, d)))
    for _ in range(int(rng.integers(3))):
        _jog(walls, rng)
    for _ in range(int(rng.integers(3))):
        _add_spur(walls, rng)
    if rng.random() < .35:
        _attach_box(walls, rng)
    return walls


def _grow(h, w, area, rng):
    mask = np.zeros((h, w), dtype=bool)
    frontier = {(int(rng.integers(h)), int(rng.integers(w)))}
    for _ in range(area):
        choices = sorted(frontier)
        p = choices[int(rng.integers(len(choices)))]
        frontier.remove(p)
        mask[p] = True
        for dr, dc in DIR4:
            q = (p[0] + dr, p[1] + dc)
            if 0 <= q[0] < h and 0 <= q[1] < w and not mask[q]:
                frontier.add(q)
    rr, cc = np.nonzero(mask)
    return mask[rr.min():rr.max() + 1, cc.min():cc.max() + 1].copy()


def _control_shape(h, w, area, rng):
    if area == 8 and h >= 3 and w >= 3 and rng.random() < .4:
        ring = np.ones((3, 3), dtype=bool)
        ring[1, 1] = False
        return ring
    return _grow(h, w, area, rng)


def _controls(frame, neutral, wall, counts, rng, equal_area=False):
    objects = []
    for color, n in counts.items():
        if equal_area:
            sizes = [6] if n == 1 else [int(rng.integers(2, 5))]
            if n == 2:
                sizes.append(6 - sizes[0])
        else:
            sizes = list(map(int, rng.choice([1, 2, 3, 4, 6, 8, 9], n,
                             p=[.16, .16, .14, .18, .18, .10, .08])))
        objects.extend((color, area) for area in sizes)
    rng.shuffle(objects)
    fh, fw = frame.shape
    side = int(rng.integers(4))  # top, bottom, left, right reserved packing band
    margins = [1, 1, 1, 1]
    axial_frame = fh if side < 2 else fw
    margins[side] = int(rng.integers(5, min(8, 29 - axial_frame) + 1))
    band_h = margins[side] - 2
    needed = sum((area + band_h - 1) // band_h for _, area in objects) + len(objects) - 1
    cross_frame = fw if side < 2 else fh
    cross_low = max(cross_frame + 2, needed + 2)
    if cross_low > 30:
        raise ValueError("outside controls do not fit")
    cross_size = int(rng.integers(cross_low, 31))
    low_side, high_side = (2, 3) if side < 2 else (0, 1)
    margins[low_side] = int(rng.integers(1, cross_size - cross_frame))
    margins[high_side] = cross_size - cross_frame - margins[low_side]
    opposite = side ^ 1
    margins[opposite] = int(rng.integers(1, 31 - axial_frame - margins[side]))
    top, bottom, left, right = margins
    g = np.full((fh + top + bottom, fw + left + right), neutral, dtype=np.int8)
    g[top:top + fh, left:left + fw] = frame
    allowed = np.ones(g.shape, dtype=bool)
    allowed[0] = allowed[-1] = False
    allowed[:, 0] = allowed[:, -1] = False
    allowed[top - 1:top + fh + 1, left - 1:left + fw + 1] = False
    occupied = np.zeros_like(allowed)
    scattered = g.copy()
    # Try genuinely scattered positions, independently shaped and sized.
    fitted = True
    for color, area in sorted(objects, key=lambda x: -x[1]):
        found = False
        for _ in range(65):
            h = int(rng.integers(1, min(4, area) + 1))
            w = max((area + h - 1) // h, int(rng.integers(1, 4)))
            obj = _control_shape(h, w, area, rng)
            oh, ow = obj.shape
            r = int(rng.integers(1, g.shape[0] - oh))
            c = int(rng.integers(1, g.shape[1] - ow))
            points = [(r + x, c + y) for x, y in np.argwhere(obj)]
            if not all(allowed[x, y] for x, y in points):
                continue
            if any(occupied[max(0, x - 1):x + 2, max(0, y - 1):y + 2].any()
                   for x, y in points):
                continue
            for p in points:
                scattered[p] = color
                occupied[p] = True
            found = True
            break
        if not found:
            fitted = False
            break
    if fitted:
        return scattered
    # Guaranteed fit: independent variable boxes with neutral gaps on one side.
    widths = [(area + band_h - 1) // band_h for _, area in objects]
    gaps = [1] * (len(objects) - 1)
    padding = [0, 0]
    extra = cross_size - 2 - sum(widths) - sum(gaps)
    for _ in range(extra):
        index = int(rng.integers(len(widths) + len(gaps) + 2))
        if index < len(widths):
            widths[index] += 1
        elif index < len(widths) + len(gaps):
            gaps[index - len(widths)] += 1
        else:
            padding[index - len(widths) - len(gaps)] += 1
    offset = 1 + padding[0]
    for i, ((color, area), width) in enumerate(zip(objects, widths)):
        obj = _control_shape(band_h, width, area, rng)
        a = int(rng.integers(band_h - obj.shape[0] + 1))
        b = int(rng.integers(width - obj.shape[1] + 1))
        if side == 0:
            r, c, actual = 1 + a, offset + b, obj
        elif side == 1:
            r, c, actual = top + fh + 1 + a, offset + b, obj
        elif side == 2:
            r, c, actual = offset + b, 1 + a, obj.T
        else:
            r, c, actual = offset + b, left + fw + 1 + a, obj.T
        patch = g[r:r + actual.shape[0], c:c + actual.shape[1]]
        patch[actual] = color
        offset += width + (gaps[i] if i < len(gaps) else 0)
    return g


def _make_pair(rng, kind, neutral, wall, palette=None, full_order=None):
    for attempt in range(80):
        if kind == "first":
            h, w = map(int, rng.integers(16, 23, size=2))
        elif kind == "second":
            h, w = int(rng.integers(22, 25)), int(rng.integers(17, 24))
        else:
            low = 13 if kind == "test" else 9
            h, w = map(int, rng.integers(low, 25, size=2))
        walls = _layout(rng, h, w, kind)
        masks = _masks(~walls)
        depths = [_distance_depth(mask) for mask in masks]
        paths = [_paths(mask, depth) for mask, depth in zip(masks, depths)]
        capacities = [max((p[4] for p in ps), default=0) for ps in paths]
        big = max(range(len(masks)), key=lambda i: (int(depths[i].max()), capacities[i]))
        if kind in ("first", "second") and (capacities[big] < 3
                or depths[big].max() < (4 if kind == "first" else 5)):
            continue
        if kind == "second" and (len(masks) != 3 or not any(
                i != big and cap >= 2 for i, cap in enumerate(capacities))):
            continue
        if kind == "test" and (capacities[big] < 2 or depths[big].max() < 4):
            continue
        if capacities[big] < 1:
            continue
        if palette is None:
            k = int(rng.integers(2 if kind == "test" else 1,
                                 min(4, capacities[big]) + 1))
            colors = [c for c in range(10) if c not in (neutral, wall)]
            chosen = tuple(map(int, rng.choice(colors, k, replace=False)))
        else:
            chosen = tuple(palette)
        if len(chosen) > capacities[big]:
            continue
        values = ([2, 1, 1] if kind == "first" else [1, 2, 2]
                  if kind == "second" else list(map(int, rng.integers(1, 3, len(chosen)))))
        if kind == "test":
            values[int(rng.integers(len(values)))] = int(rng.integers(3, 5))
        counts = dict(zip(chosen, values))
        frame = np.full(walls.shape, neutral, dtype=np.int8)
        frame[walls] = wall
        rooms = []
        little = next((i for i, cap in enumerate(capacities) if i != big and cap >= 2), None)
        for i, (mask, depth, options) in enumerate(zip(masks, depths, paths)):
            if i == big:
                key = tuple(full_order) if full_order is not None else tuple(map(int, rng.permutation(chosen)))
            elif kind == "second":
                if i == little:
                    # Even a shallow room must visibly reject one global key.
                    leading = int(rng.choice([c for c in chosen if c != full_order[0]]))
                    following = int(rng.choice([c for c in chosen if c != leading]))
                    key = (leading, following)
                else:
                    key = ()
            elif not options or rng.random() < .22:
                key = ()
            else:
                k = int(rng.integers(1, min(len(chosen), capacities[i]) + 1))
                key = tuple(map(int, rng.choice(chosen, k, replace=False)))
            positions, directions = (), ()
            if key:
                valid = [p for p in options if p[4] >= len(key)]
                r, c, dr, dc, _ = valid[int(rng.integers(len(valid)))]
                positions = tuple((r + j * dr, c + j * dc) for j in range(len(key)))
                for p, color in zip(positions, key):
                    frame[p] = color
                directions = ((dr, dc),)
            rooms.append({"mask": mask, "depth": depth, "key": key,
                          "positions": positions, "directions": directions})
        info = {"frame": frame, "walls": walls, "neutral": neutral, "wall": wall,
                "counts": counts, "rooms": rooms}
        output = _render(info, depth_mode="distance")
        if kind == "first" and (np.array_equal(output, _render(info, depth_mode="box"))
                                or np.array_equal(output, _render(info, depth_mode="manhattan"))):
            continue
        if kind == "test" and np.array_equal(output, _render(
                info, counts={c: min(n, 2) for c, n in counts.items()})):
            continue
        try:
            input_grid = _controls(frame, neutral, wall, counts, rng,
                                   equal_area=kind in ("first", "second"))
            recovered = _decode(input_grid)
        except ValueError:
            continue
        _require(np.array_equal(output, _render(recovered)), "construction/solve mismatch")
        return {"input": input_grid, "output": output}, rooms[big]["key"]
    raise RuntimeError("bounded construction could not find a supported frame")


def _check_integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid_key(g):
    return tuple(g.shape), g.astype(np.int8).tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(np.asarray(p["input"])), _grid_key(np.asarray(p["output"])))
                       for p in puzzle[s]) for s in ("train", "test"))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent puzzle; all required conventions are visible."""
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _check_integer(num_train, "num_train", 2, 10)
    ne = _check_integer(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.permutation(10)))
    palette = colors[:3]
    first, order = _make_pair(rng, "first", colors[3], colors[4], palette)
    second_order = order[1:] + order[:1]
    second, _ = _make_pair(rng, "second", colors[5], colors[6], palette, second_order)
    puzzle, seen = {"train": [first, second], "test": []}, set()
    seen.update(_grid_key(p["input"]) for p in puzzle["train"])
    for split, count, kind in (("train", nt - 2, "extra"), ("test", ne, "test")):
        for _ in range(count):
            for attempt in range(20):
                neutral, wall = map(int, rng.choice(10, 2, replace=False))
                pair, _ = _make_pair(rng, kind, neutral, wall)
                if _grid_key(pair["input"]) not in seen:
                    seen.add(_grid_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not generate distinct inputs")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check returned data and family witnesses without generation metadata."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen, train, tests = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_grid_key(inp) not in seen, "duplicate input")
            seen.add(_grid_key(inp))
            info = _decode(inp)
            _require(1 <= len(info["rooms"]) <= 6, "generated room count outside subset")
            _require(out.shape == info["frame"].shape, "output must tight-crop the frame")
            _require(np.array_equal(out[info["walls"]], info["frame"][info["walls"]]), "wall changed")
            _require(np.array_equal(out, _render(info)), "incorrect weighted contours")
            for room in info["rooms"]:
                _require(np.array_equal(room["depth"], _distance_depth(room["mask"])), "depth disagreement")
                if not room["key"]:
                    _require(np.all(out[room["mask"]] == info["neutral"]), "unmarked room changed")
            info["output"] = out
            (train if split == "train" else tests).append(info)
    for role in ("neutral", "wall"):
        _require(len({a[role] for a in train}) >= 2, "training must vary visible role colors")
    _require(all(max(a["counts"].values()) <= 2 for a in train), "training width envelope")
    _require(all(max(a["counts"].values()) >= 3 for a in tests), "missing count transfer")
    _require(all(not np.array_equal(a["output"], _render(
        a, counts={c: min(n, 2) for c, n in a["counts"].items()})) for a in tests),
        "new test count does not affect answer")
    _require(any(set(a["counts"]) == set(b["counts"]) and a["counts"] != b["counts"]
                 and a["areas"] == b["areas"] for a in train for b in train),
             "missing equal-area/different-count census")
    _require(len({len(a["rooms"]) for a in train}) >= 2, "room-count variation missing")
    _require(any(not r["key"] for a in train for r in a["rooms"]), "unmarked-room witness missing")
    _require(any(len({r["key"] for r in a["rooms"] if r["key"]}) >= 2 for a in train),
             "shared census with different local keys missing")
    _require(sum(any(len(r["key"]) >= 3 and int(r["depth"].max()) >= sum(
        a["counts"][c] for c in r["key"]) for r in a["rooms"]) for a in train) >= 2,
        "two complete-cycle demonstrations needed")
    alternatives = {
        "omit B": lambda a: _render(a, counts={c: 1 for c in a["counts"]}),
        "pixel area": lambda a: _render(a, counts=a["areas"]),
        "total count": lambda a: _render(a, counts={c: sum(a["counts"].values()) for c in a["counts"]}),
        "omit A/box": lambda a: _render(a, depth_mode="box"),
        "Manhattan": lambda a: _render(a, depth_mode="manhattan"),
        "sorted key": lambda a: _render(a, key_mode="sorted"),
        "one global key": lambda a: _render(a, key_mode="global"),
    }
    for name, alternative in alternatives.items():
        _require(any(not np.array_equal(a["output"], alternative(a)) for a in train),
                 "training fails to distinguish " + name)
    _require(any(int(a["frame"][p]) != int(a["output"][p]) for a in train
                 for r in a["rooms"] for p in r["positions"]), "key-overwriting witness missing")


def self_test():
    """Bounded 42-seed diversity, count, determinism, and semantic-invariant checks."""
    shapes, geometry, palettes, widths, room_counts, keys, directions = (set() for _ in range(7))
    fingerprints, object_sizes, control_holes, multi_side = set(), set(), 0, 0
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        fingerprints.add(_puzzle_key(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _decode(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve disagrees")
                shapes.add(tuple(pair["input"].shape))
                geometry.add(_grid_key(a["walls"].astype(np.int8)))
                palettes.add((a["neutral"], a["wall"], tuple(a["counts"])))
                widths.update(a["counts"].values())
                room_counts.add(len(a["rooms"]))
                for r in a["rooms"]:
                    keys.add(r["key"])
                    directions.update(r["directions"])
                top, bottom, left, right = a["box"]
                sides = set()
                for _, obj in a["objects"]:
                    object_sizes.add(len(obj))
                    rr, cc = zip(*obj)
                    obj_mask = np.zeros((max(rr) - min(rr) + 3, max(cc) - min(cc) + 3), dtype=bool)
                    for r, c in obj:
                        obj_mask[r - min(rr) + 1, c - min(cc) + 1] = True
                    control_holes += len(_components(~obj_mask)) > 1
                    x, y = obj[0]
                    sides.add("top" if x < top else "bottom" if x > bottom
                              else "left" if y < left else "right")
                multi_side += len(sides) >= 2
    _require(len(fingerprints) == 42, "whole-puzzle diversity")
    _require(len(shapes) >= 40 and len(geometry) >= 100, "structural diversity")
    _require(len(palettes) >= 60 and len(keys) >= 80, "role/key diversity")
    _require(widths == {1, 2, 3, 4}, "width range missing")
    _require(len(room_counts) >= 5, "room count collapsed")
    _require(directions == {(-1, -1), (-1, 1), (1, -1), (1, 1)}, "corner orientations missing")
    _require(multi_side >= 10, "controls restricted to a fixed side")
    _require(max(object_sizes) >= 8 and control_holes >= 1, "counter shape/area diversity")
    for seed in (0, 17, 41):
        _require(_puzzle_key(puzzles[seed]) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(101 + nt * 7 + ne, nt, ne)
            _require(len(p["train"]) == nt and len(p["test"]) == ne, "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    # Check erosion islands, a genuine hole, and off-canvas boundaries directly.
    mask = np.ones((13, 15), dtype=bool)
    mask[4:9, 6:9] = False
    mask[6, :4] = False
    _require(np.array_equal(_erosion_depth(mask), _distance_depth(mask)), "hole/tip depth error")
    for shape in ((1, 1), (1, 9), (8, 1), (8, 12)):
        mask = np.ones(shape, dtype=bool)
        _require(np.array_equal(_erosion_depth(mask), _distance_depth(mask)), "boundary depth error")
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    corrupt["test"][0]["output"][0, 0] = (int(corrupt["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("changed output wall accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "wall_geometries": len(geometry), "palettes": len(palettes), "keys": len(keys),
            "widths": sorted(widths), "room_counts": sorted(room_counts),
            "multi_side_censuses": multi_side, "control_areas": sorted(object_sizes),
            "controls_with_holes": control_holes}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
