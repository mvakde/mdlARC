"""Phase-matched collection: a hub word repeats over geodesic distance layers.

No parent code, files, demonstrations, seed, or construction metadata is used by
solve. Structural colors are recovered anew from the frame and marked hub.
"""
from collections import deque
import json
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(np.all((0 <= a) & (a <= 9)), "Bad colors")
    return a.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _footprint(shape, pos):
    mask = np.zeros(shape, bool)
    r, c = pos
    mask[r:r+3, c:c+3] = True
    return mask


def _stem(pos, direction, length):
    r, c = pos
    dr, dc = STEPS[direction]
    # The collector center is one step behind its outward face.
    return [(r+1+(i+2)*dr, c+1+(i+2)*dc) for i in range(length+1)]


def _parse(grid):
    a = _grid(grid)
    h, w = a.shape
    _require(h >= 7 and w >= 7, "Scene is too small for marked collector")
    wall_color = int(a[0, 0])
    _require(np.all(a[0] == wall_color) and np.all(a[-1] == wall_color)
             and np.all(a[:, 0] == wall_color) and np.all(a[:, -1] == wall_color),
             "Outer boundary must be one wall color")
    candidates = []
    for color in np.unique(a):
        if color == wall_color or np.count_nonzero(a == color) != 8:
            continue
        rr, cc = np.nonzero(a == color)
        r, c = int(rr.min()), int(cc.min())
        if rr.max() != r+2 or cc.max() != c+2:
            continue
        patch = a[r:r+3, c:c+3]
        marker = int(patch[1, 1])
        if marker in (int(color), wall_color) or np.count_nonzero(a == marker) != 2:
            continue
        mask = np.ones((3, 3), bool)
        mask[1, 1] = False
        if not np.all(patch[mask] == color):
            continue
        mr, mc = np.nonzero(a == marker)
        cap = next((int(x), int(y)) for x, y in zip(mr, mc) if (x, y) != (r+1, c+1))
        for direction, (dr, dc) in enumerate(STEPS):
            delta_r, delta_c = cap[0]-r-1, cap[1]-c-1
            length = abs(delta_r)+abs(delta_c)-2
            if not 1 <= length <= 5 or (delta_r, delta_c) != ((length+2)*dr, (length+2)*dc):
                continue
            points = _stem((r, c), direction, length)
            values = [int(a[p]) for p in points[:-1]]
            if any(x in (wall_color, int(color), marker) for x in values):
                continue
            outside = ~(_footprint(a.shape, (r, c)) | (a == wall_color) | (a == marker))
            colors, counts = np.unique(a[outside], return_counts=True)
            if len(colors) < 2 or sum(counts == counts.max()) != 1:
                continue
            background = int(colors[np.argmax(counts)])
            if background in values:
                continue
            candidates.append(dict(pos=(r, c), hub=int(color), marker=marker,
                                   wall_color=wall_color, background=background,
                                   direction=direction, word=values, points=points))
    _require(len(candidates) == 1, "Missing or ambiguous marked collector and capped word")
    roles = candidates[0]
    protected = _footprint(a.shape, roles["pos"])
    for point in roles["points"]:
        protected[point] = True
    tokens = ~protected & ~np.isin(a, [roles[k] for k in ("background", "wall_color", "hub", "marker")])
    roles.update(protected=protected, tokens=tokens, wall=a == wall_color)
    return a, roles


def _distances(wall, pos):
    h, w = wall.shape
    result = np.full(wall.shape, -1, dtype=np.int16)
    r, c = pos
    queue = deque((r+i, c+j) for i in range(3) for j in range(3))
    for point in queue:
        result[point] = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc] and result[nr, nc] < 0:
                result[nr, nc] = result[r, c]+1
                queue.append((nr, nc))
    return result


def _wave_distances(wall, pos):
    """Separate array wavefront audit of graph distance, without a BFS queue."""
    seen = _footprint(wall.shape, pos)
    result = np.full(wall.shape, -1, dtype=np.int16)
    result[seen] = 0
    frontier = seen.copy()
    for depth in range(1, wall.size+1):
        spread = np.zeros(wall.shape, bool)
        spread[1:] |= frontier[:-1]
        spread[:-1] |= frontier[1:]
        spread[:, 1:] |= frontier[:, :-1]
        spread[:, :-1] |= frontier[:, 1:]
        frontier = spread & ~wall & ~seen
        if not frontier.any():
            break
        seen |= frontier
        result[frontier] = depth
    return result


def _direct_distances(shape, pos):
    rr, cc = np.indices(shape)
    r, c = pos
    return np.maximum(r-rr, 0)+np.maximum(rr-r-2, 0)+np.maximum(c-cc, 0)+np.maximum(cc-c-2, 0)


def _eligibility(a, roles, distance):
    word = np.asarray(roles["word"], dtype=np.int8)
    expected = word[(distance-1) % len(word)]
    return roles["tokens"] & (distance > 0) & (a == expected)


def _ordered(mask, distance, strict=True):
    cells = [tuple(map(int, p)) for p in np.argwhere(mask)]
    cells.sort(key=lambda p: (int(distance[p]), p))
    if strict and len(cells) > 9:
        _require(distance[cells[8]] < distance[cells[9]], "Tie across capacity frontier")
    return cells[:9]


def _transfer(a, roles, cells):
    out = a.copy()
    r, c = roles["pos"]
    for i, point in enumerate(cells):
        out[point] = roles["background"]
        out[r+i//3, c+i%3] = a[point]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a, roles = _parse(input_grid)
    distance = _distances(roles["wall"], roles["pos"])
    eligible = _eligibility(a, roles, distance)
    return _transfer(a, roles, _ordered(eligible, distance))


def _features(grid):
    a, roles = _parse(grid)
    distance = _distances(roles["wall"], roles["pos"])
    eligible = _eligibility(a, roles, distance)
    cells = _ordered(eligible, distance)
    selected = np.zeros(a.shape, bool)
    for point in cells:
        selected[point] = True
    retained = eligible & ~selected
    direct = _direct_distances(a.shape, roles["pos"])
    ignored_phase = _transfer(a, roles, _ordered(roles["tokens"] & (distance > 0), distance, strict=False))
    ignored_walls = _transfer(a, roles, _ordered(_eligibility(a, roles, direct), direct, strict=False))
    direct_phase = _eligibility(a, roles, direct) & (distance > 0)
    # Isolate the path-ranking contribution while retaining the correct phase filter.
    geometric_rank = _transfer(a, roles, _ordered(eligible, direct, strict=False))
    out = _transfer(a, roles, cells)
    stream = tuple(int(a[p]) for p in cells)
    repeating_prefix = tuple(roles["word"][i % len(roles["word"])] for i in range(len(cells)))
    return dict(roles=roles, distance=distance, eligible=eligible, selected=selected,
                output=out, shape=a.shape, word=tuple(roles["word"]), direction=roles["direction"],
                load=len(cells), eligible_count=int(eligible.sum()), tokens=int(roles["tokens"].sum()),
                mismatch=int((roles["tokens"] & (distance > 0) & ~eligible).sum()),
                unreachable=int((roles["tokens"] & (distance < 0)).sum()),
                wall_count=int(roles["wall"].sum()),
                max_distance=int(distance.max()),
                detour=int((distance-direct)[distance > 0].max()),
                phase_matters=not np.array_equal(out, ignored_phase),
                walls_matter=not np.array_equal(out, ignored_walls),
                phase_path_matters=not np.array_equal(eligible, direct_phase),
                path_rank_matters=not np.array_equal(out, geometric_rank),
                priority_reversal=bool(retained.any() and selected.any()
                                       and direct[selected].max() > direct[retained].min()),
                stream=stream, sparse_stream=stream != repeating_prefix,
                gap=int(distance[retained].min()-distance[selected].max()) if retained.any() else None)


def _rectangle(wall, bounds):
    r0, c0, r1, c1 = bounds
    wall[r0, c0:c1+1] = True
    wall[r1, c0:c1+1] = True
    wall[r0:r1+1, c0] = True
    wall[r0:r1+1, c1] = True


def _geometry(rng, shape, kind):
    h, w = shape
    wall = np.zeros(shape, bool)
    if kind == "maze":
        wall[:] = True
        nodes = [(r, c) for r in range(1, h-1, 2) for c in range(1, w-1, 2)]
        start = nodes[int(rng.integers(len(nodes)))]
        seen, stack = {start}, [start]
        wall[start] = False
        while stack:
            r, c = stack[-1]
            choices = [(r+2*dr, c+2*dc) for dr, dc in STEPS
                       if 1 <= r+2*dr < h-1 and 1 <= c+2*dc < w-1
                       and (r+2*dr, c+2*dc) not in seen]
            if not choices:
                stack.pop()
                continue
            nr, nc = choices[int(rng.integers(len(choices)))]
            wall[nr, nc] = wall[(nr+r)//2, (nc+c)//2] = False
            stack.append((nr, nc))
            seen.add((nr, nc))
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(1, h-3)), int(rng.integers(1, w-3))
            wall[r:min(h-1, r+int(rng.integers(2, 5))), c:min(w-1, c+int(rng.integers(2, 5)))] = False
    elif kind == "partitions":
        rooms = [(1, 1, h-2, w-2)]
        for _ in range(int(rng.integers(3, 9))):
            if not rooms:
                break
            idx = int(rng.integers(len(rooms)))
            r0, c0, r1, c1 = rooms.pop(idx)
            horizontal = (rng.random() < .5 and r1-r0 >= 5) or c1-c0 < 5
            if horizontal and r1-r0 >= 5:
                r = int(rng.integers(r0+2, r1-1))
                wall[r, c0:c1+1] = True
                for _ in range(int(rng.integers(0, 3))):
                    c = int(rng.integers(c0, c1+1))
                    wall[r, c:min(c1+1, c+int(rng.integers(1, 3)))] = False
                rooms.extend([(r0, c0, r-1, c1), (r+1, c0, r1, c1)])
            elif c1-c0 >= 5:
                c = int(rng.integers(c0+2, c1-1))
                wall[r0:r1+1, c] = True
                for _ in range(int(rng.integers(0, 3))):
                    r = int(rng.integers(r0, r1+1))
                    wall[r:min(r1+1, r+int(rng.integers(1, 3))), c] = False
                rooms.extend([(r0, c0, r1, c-1), (r0, c+1, r1, c1)])
    elif kind == "baffles":
        horizontal = bool(rng.integers(2))
        limit, span = (h, w) if horizontal else (w, h)
        choices = np.arange(3, limit-3)
        positions = sorted(map(int, rng.choice(choices, size=min(len(choices), int(rng.integers(2, 5))), replace=False)))
        for i, position in enumerate(positions):
            gap = int(rng.integers(2, max(3, span//3)))
            lo, hi = (1, span-1-gap) if i % 2 == 0 else (gap+1, span-1)
            if horizontal:
                wall[position, lo:hi] = True
            else:
                wall[lo:hi, position] = True
    elif kind in ("rooms", "turns"):
        for _ in range(int(rng.integers(2, 7))):
            r0, c0 = int(rng.integers(1, h-4)), int(rng.integers(1, w-4))
            r1, c1 = int(rng.integers(r0+2, h-1)), int(rng.integers(c0+2, w-1))
            if kind == "rooms":
                _rectangle(wall, (r0, c0, r1, c1))
                for _ in range(int(rng.integers(0, 4))):
                    if rng.random() < .5:
                        wall[r0 if rng.random() < .5 else r1, int(rng.integers(c0+1, c1))] = False
                    else:
                        wall[int(rng.integers(r0+1, r1)), c0 if rng.random() < .5 else c1] = False
            else:
                wall[r0, c0:c1+1] = True
                wall[r0:r1+1, c1] = True
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    return wall


def _closed_pocket(rng, wall, protected):
    h, w = wall.shape
    for _ in range(30):
        ph, pw = int(rng.integers(3, 7)), int(rng.integers(3, 7))
        r0, c0 = int(rng.integers(1, h-ph)), int(rng.integers(1, w-pw))
        if protected[r0:r0+ph, c0:c0+pw].any():
            continue
        wall[r0+1:r0+ph-1, c0+1:c0+pw-1] = False
        _rectangle(wall, (r0, c0, r0+ph-1, c0+pw-1))
        return


def _fallback_geometry(length, rng):
    """A variable U-detour and closed pocket, used after bounded random attempts."""
    h, w = int(rng.integers(24, 31)), int(rng.integers(23, 31))
    wall = np.zeros((h, w), bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    left = int(rng.integers(3, 5))
    right = left+7
    top = int(rng.integers(3, 5))
    bottom = h-3
    wall[top, left:right+1] = True
    wall[top:bottom+1, left] = True
    wall[top:bottom+1, right] = True
    pos, direction = (top+3, left+2), 1
    _rectangle(wall, (1, w-7, 5, w-2))
    wall[2:5, w-6:w-2] = False
    # Rotation happens before payload placement, so packing remains displayed row-major.
    turns = int(rng.integers(4))
    hub = _footprint(wall.shape, pos)
    cap = np.zeros(wall.shape, bool)
    cap[_stem(pos, direction, length)[-1]] = True
    wall, hub, cap = (np.rot90(x, turns).copy() for x in (wall, hub, cap))
    rr, cc = np.nonzero(hub)
    pos = (int(rr.min()), int(cc.min()))
    cr, cl = map(int, np.argwhere(cap)[0])
    delta = (cr-pos[0]-1, cl-pos[1]-1)
    direction = next(i for i, (dr, dc) in enumerate(STEPS)
                     if delta == ((length+2)*dr, (length+2)*dc))
    return wall, pos, direction


def _pick(rng, pool, count, style="dispersed", forced=()):
    available = {tuple(map(int, x)) for x in np.argwhere(pool)}
    chosen = list(dict.fromkeys(forced))
    if count > len(available) or not set(chosen) <= available or len(chosen) > count:
        return None
    available.difference_update(chosen)
    if not chosen and count:
        ordered = sorted(available)
        chosen.append(ordered[int(rng.integers(len(ordered)))])
        available.remove(chosen[0])
    while len(chosen) < count:
        ordered = sorted(available)
        if style == "dispersed":
            point = ordered[int(rng.integers(len(ordered)))]
        else:
            ar, ac = chosen[0]
            def score(p):
                dr, dc = abs(p[0]-ar), abs(p[1]-ac)
                return dr*dr+dc*dc if style == "compact" else dr*pool.shape[1]+dc
            scores = [score(p) for p in ordered]
            best = [p for p, s in zip(ordered, scores) if s == min(scores)]
            point = best[int(rng.integers(len(best)))]
        chosen.append(point)
        available.remove(point)
    return chosen


def _capacity_supply(rng, active, distance, direct, count, reversal, style):
    thresholds = np.unique(distance[active])
    for t in rng.permutation(thresholds):
        near, far = active & (distance <= t), active & (distance > t)
        if near.sum() < 9 or far.sum() < count-9:
            continue
        forced_near, forced_far = (), ()
        if reversal:
            m_near, m_far = int(direct[near].max()), int(direct[far].min())
            if m_near <= m_far:
                continue
            a = _pick(rng, near & (direct == m_near), 1)[0]
            b = _pick(rng, far & (direct == m_far), 1)[0]
            forced_near, forced_far = (a,), (b,)
        selected = _pick(rng, near, 9, style, forced_near)
        excess = _pick(rng, far, count-9, style, forced_far)
        return selected+excess
    return None


def _build_pair(rng, request, regime, testing):
    length = request["length"]
    colors = [int(x) for x in rng.permutation(10)]
    background, wall_color, hub, marker = colors[:4]
    palette = colors[4:4+int(rng.integers(2, 7))]
    if request["kind"] == "phase":
        word = [palette[0], palette[1], palette[0]]
    elif request["kind"] == "capacity":
        word = [palette[0], palette[1]]
    else:
        word = [int(x) for x in rng.choice(palette, size=length)]
        if length > 1 and len(set(word)) == 1 and rng.random() < .8:
            word[-1] = next(x for x in palette if x != word[0])
    kind = request["kind"]
    count = request["count"]
    for attempt in range(64):
        if attempt >= 48:
            wall, pos, direction = _fallback_geometry(length, rng)
        else:
            low, high = (18, 31) if testing and regime in ("route", "scale") else (12, 25) if regime == "scale" else (12, 31)
            shape = tuple(int(x) for x in rng.integers(low, high, size=2))
            directions = []
            r, c = int(rng.integers(1, shape[0]-3)), int(rng.integers(1, shape[1]-3))
            pos = (r, c)
            for d in range(4):
                points = _stem(pos, d, length)
                if all(1 <= rr < shape[0]-1 and 1 <= cc < shape[1]-1 for rr, cc in points):
                    directions.append(d)
            if not directions:
                continue
            direction = int(rng.choice(directions))
            modes = ("partitions", "maze", "rooms", "turns", "baffles") if testing and regime == "route" else ("open", "baffles", "partitions", "rooms", "turns", "maze")
            wall = _geometry(rng, shape, str(rng.choice(modes)))
        points = _stem(pos, direction, length)
        protected = _footprint(wall.shape, pos)
        for p in points:
            protected[p] = True
        wall[protected] = False
        if kind in ("phase", "capacity") or count == 0 or rng.random() < .5:
            _closed_pocket(rng, wall, protected)
        distance = _distances(wall, pos)
        direct = _direct_distances(wall.shape, pos)
        available = ~protected & ~wall
        active, inactive = available & (distance > 0), available & (distance < 0)
        if active.sum() < count+3 or (kind in ("phase", "capacity") or count == 0) and not inactive.any():
            continue
        style = str(rng.choice(("dispersed", "compact", "line")))
        if kind == "phase":
            first = _pick(rng, active & (distance == 4), 1)
            second = _pick(rng, active & (distance == 5), 1)
            if not first or not second or (active & (distance == 4)).sum() < 2:
                continue
            supply = _pick(rng, active, count, style, first+second)
        elif count > 9:
            supply = _capacity_supply(rng, active, distance, direct, count, kind == "capacity", style)
        else:
            supply = _pick(rng, active, count, style)
        if supply is None:
            continue
        used = np.zeros(wall.shape, bool)
        for p in supply:
            used[p] = True
        wrong_pool = active & ~used
        if kind == "phase":
            wrong_forced = _pick(rng, wrong_pool & (distance == 4), 1)
            if not wrong_forced:
                continue
            phase_difference = wrong_pool & (np.asarray(word)[(distance-1) % length]
                                             != np.asarray(word)[(direct-1) % length])
            if not phase_difference.any():
                continue
            route_forced = (wrong_forced if phase_difference[wrong_forced[0]]
                            else _pick(rng, phase_difference, 1))
            wrong_forced = list(dict.fromkeys(wrong_forced+route_forced))
        else:
            wrong_forced = ()
        wrong_count = int(rng.integers(max(1, len(wrong_forced)), min(7, int(wrong_pool.sum()))+1))
        wrong = _pick(rng, wrong_pool, wrong_count, style, wrong_forced)
        unreachable_min = 1 if kind in ("phase", "capacity") or count == 0 else 0
        unreachable_count = int(rng.integers(unreachable_min, min(8, int(inactive.sum()))+1))
        unreachable = _pick(rng, inactive, unreachable_count, str(rng.choice(("dispersed", "compact", "line"))))
        a = np.full(wall.shape, background, dtype=np.int8)
        a[wall] = wall_color
        r, c = pos
        a[r:r+3, c:c+3] = hub
        a[r+1, c+1] = marker
        for p, color in zip(points[:-1], word):
            a[p] = color
        a[points[-1]] = marker
        for p in supply:
            a[p] = word[(int(distance[p])-1) % length]
        for p in wrong:
            expected = word[(int(distance[p])-1) % length]
            if kind == "phase" and p == wrong_forced[0]:
                a[p] = word[1]
            elif kind == "phase" and p in route_forced:
                a[p] = word[(int(direct[p])-1) % length]
            else:
                a[p] = int(rng.choice([x for x in palette if x != expected]))
        for p in unreachable:
            a[p] = int(rng.choice(palette))
        # Labels are built from sampled eligible coordinates, never by solve.
        selected = sorted(supply, key=lambda p: (int(distance[p]), p))[:9]
        construction_roles = dict(pos=pos, background=background)
        out = _transfer(a, construction_roles, selected)
        try:
            f = _features(a)
            if not np.array_equal(f["output"], out):
                continue
            if kind == "capacity" and not (f["priority_reversal"] and f["path_rank_matters"]
                                           and f["walls_matter"] and f["sparse_stream"]):
                continue
            if kind == "phase" and not (f["phase_matters"] and f["phase_path_matters"]):
                continue
        except AssertionError:
            continue
        return dict(input=a, output=out)
    raise AssertionError("Bounded constructor and U-detour fallback exhausted")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not (_integer(num_train) and _integer(num_test) and 2 <= num_train <= 10
            and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Need 2..10 train, 1..4 test, and more train than test")
    rng = np.random.default_rng(int(seed))
    regime = str(rng.choice(("route", "word", "scale")))
    train_requests = [dict(kind="phase", length=3, count=int(rng.choice((2, 4)))),
                      dict(kind="capacity", length=2, count=int(rng.integers(10, 19)))]
    def random_request(testing):
        length = int(rng.integers(4, 6)) if testing and regime == "word" else int(rng.integers(1, 4 if regime == "word" else 6))
        draw = rng.random()
        count = 0 if draw < .06 else int(rng.integers(10, 27 if testing and regime == "scale" else 19)) if draw < .4 else int(rng.integers(1, 10))
        return dict(kind="random", length=length, count=count)
    train_requests.extend(random_request(False) for _ in range(num_train-2))
    rng.shuffle(train_requests)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, requests in (("train", train_requests),
                            ("test", [random_request(True) for _ in range(num_test)])):
        for request in requests:
            for _ in range(12):
                pair = _build_pair(rng, request, regime, split == "test")
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("Could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _whole_word_witness(grid, f):
    if len(f["word"]) != 3 or not (f["word"][0] == f["word"][2] != f["word"][1]):
        return False
    a = np.asarray(grid)
    d, selected = f["distance"], f["selected"]
    # At layer four the whole word says A; a truncated AB cycle says B.
    return bool((selected & (d == 4) & (a == f["word"][0])).any()
                and (f["roles"]["tokens"] & (d == 4) & (a == f["word"][1])).any()
                and (selected & (d == 5) & (a == f["word"][1])).any())


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Illegal counts")
    seen, training = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Wrong pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "Dimensions changed")
            _require(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            f = _features(a)
            roles, selected = f["roles"], f["selected"]
            _require(np.array_equal(f["distance"], _wave_distances(roles["wall"], roles["pos"])), "Path audit failed")
            _require(np.array_equal(f["output"], b), "Incorrect phase, capacity, or packing")
            _require(f["eligible_count"] <= 9 or f["gap"] > 0, "Capacity tie")
            _require(f["load"] == min(9, f["eligible_count"]), "Wrong transfer count")
            collector = _footprint(a.shape, roles["pos"])
            _require(np.array_equal(a[~collector & ~selected], b[~collector & ~selected]), "Unselected material changed")
            _require(np.all(b[selected] == roles["background"]), "Selected origins not erased")
            _require(int(roles["tokens"].sum()) == int((~roles["protected"] & ~np.isin(b, [roles[k] for k in ("background", "wall_color", "hub", "marker")])).sum())+f["load"], "Token conservation failed")
            if not f["load"]:
                _require(f["tokens"] > 0 and np.array_equal(a, b), "Empty collection must preserve visible tokens")
            if split == "train":
                training.append((a, f))
    _require(any(f["load"] in (2, 4) and _whole_word_witness(a, f) and f["unreachable"]
                 and f["phase_matters"] and f["phase_path_matters"] for a, f in training), "Need partial whole-word phase witness")
    _require(any(f["eligible_count"] > 9 and f["unreachable"] and f["priority_reversal"]
                 and f["path_rank_matters"] and f["walls_matter"] and f["sparse_stream"]
                 for _, f in training), "Need path/capacity/packing witness")


def self_test():
    """Bounded 42-seed checks of actual family features, plus negative checks."""
    records, signatures, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(pair[role]) for s in ("train", "test") for pair in puzzle[s] for role in ("input", "output"))
        signatures.add(signature)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                f = _features(a)
                _require(np.array_equal(solve(a.copy()), pair["output"]), "Input-only solve failed")
                records.append(f)
                mapping = {}
                normalized = tuple(mapping.setdefault(int(x), len(mapping)) for x in a.ravel())
                structures.add((a.shape, normalized))
    _require(len(signatures) == 42 and len(structures) >= 150, "Structural variation collapsed")
    for name, minimum in (("shape", 60), ("word", 50), ("load", 8),
                          ("eligible_count", 14), ("wall_count", 60), ("max_distance", 25)):
        _require(len({f[name] for f in records}) >= minimum, "Insufficient variation: "+name)
    _require({len(f["word"]) for f in records} == {1, 2, 3, 4, 5}, "Missing word lengths")
    _require({f["direction"] for f in records} == {0, 1, 2, 3}, "Missing directions")
    for role in ("background", "wall_color", "hub", "marker"):
        _require(len({f["roles"][role] for f in records}) == 10, "Missing role colors: "+role)
    _require(sum(f["priority_reversal"] for f in records) >= 42, "Missing path-ranking witnesses")
    _require(any(f["load"] == 0 for f in records), "Missing zero cases")
    _require(any(f["eligible_count"] == 9 for f in records), "Missing exact-capacity cases")
    _require(any(f["detour"] > 25 for f in records), "Missing long detours")
    # All legal count combinations, as well as count/type rejection.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(9031+nt*10+ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (3.0, 1), (4, True), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    p, q = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s]) for k in x), "Determinism failed")
    recoloring = np.random.default_rng(713).permutation(10)
    for split in p:
        for pair in p[split]:
            _require(np.array_equal(solve(recoloring[pair["input"]]), recoloring[pair["output"]]),
                     "Visible-role recoloring changed the rule")
    # Stress runs rarely need the fallback. Exercise its capacity and phase
    # geometry explicitly instead of leaving that escape path untested.
    fallback_rng = np.random.default_rng(651)
    for length in range(1, 6):
        wall, pos, direction = _fallback_geometry(length, fallback_rng)
        protected = _footprint(wall.shape, pos)
        for point in _stem(pos, direction, length):
            protected[point] = True
        wall[protected] = False
        distance = _distances(wall, pos)
        _require(np.array_equal(distance, _wave_distances(wall, pos)), "Fallback path audit failed")
        direct = _direct_distances(wall.shape, pos)
        active = ~wall & ~protected & (distance > 0)
        _require(_capacity_supply(fallback_rng, active, distance, direct, 26, True, "compact") is not None,
                 "Fallback cannot support largest excess request")
        _require((~wall & ~protected & (distance < 0)).any(), "Fallback lacks closed distraction")
        if length == 3:
            word = np.array([2, 3, 2])
            _require((active & (distance == 4)).sum() >= 2 and (active & (distance == 5)).any()
                     and (active & (word[(distance-1) % 3] != word[(direct-1) % 3])).any(),
                     "Fallback cannot support whole-word/geodesic-phase witness")
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    # The equal-distance capacity frontier is deliberately excluded, not given
    # an invisible queue or coordinate tie breaker.
    mask = np.ones((2, 5), bool)
    try:
        _ordered(mask, np.ones(mask.shape, dtype=int))
    except AssertionError:
        pass
    else:
        raise AssertionError("Capacity tie accepted")
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                normalized_structures=len(structures), shapes=len({f["shape"] for f in records}),
                words=len({f["word"] for f in records}),
                loads=sorted({f["load"] for f in records}),
                eligible_counts=sorted({f["eligible_count"] for f in records}),
                path_priority_witnesses=sum(f["priority_reversal"] for f in records),
                max_path=max(f["max_distance"] for f in records), fallback_cases=5)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
