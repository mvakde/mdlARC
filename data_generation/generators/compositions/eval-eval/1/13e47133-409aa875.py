"""Contour resonance: five-step arrow shots address cyclic erosion classes.

Only NumPy and the standard library are used. The solver reads one input and
does not access the constructor, seeds, files, or demonstration outputs.
"""
from collections import Counter, defaultdict, deque
import numpy as np


BACKGROUND, COLLISION, ARROW, DISTANCE = 0, 8, 9, 5
D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer ARC grid")
    return g.astype(np.int8, copy=False)


def _components(mask, neighbors=D4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, todo = [start], [start]
        while todo:
            r, c = todo.pop()
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    todo.append(q)
        comp = np.zeros(mask.shape, dtype=bool)
        rr, cc = zip(*cells)
        comp[rr, cc] = True
        result.append(comp)
    return result


def _depth(mask, neighbors=D8):
    """Repeated erosion; outside the canvas and other regions are exterior."""
    current = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int16)
    h, w = mask.shape
    level = 0
    while current.any():
        p = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in neighbors:
            interior &= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~interior] = level
        current = interior
        level += 1
    return result


def _bfs_depth(mask):
    """A second depth calculation for invariants, rather than rendering labels."""
    h, w = mask.shape
    result = np.full(mask.shape, -1, dtype=np.int16)
    todo = deque()
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w
                    and mask[r + dr, c + dc]) for dr, dc in D8):
            result[r, c] = 0
            todo.append((r, c))
    while todo:
        r, c = todo.popleft()
        for dr, dc in D8:
            nr, nc = r + dr, c + dc
            if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                    and result[nr, nc] < 0):
                result[nr, nc] = result[r, c] + 1
                todo.append((nr, nc))
    return result


def _corner_paths(mask, depth, row_direction=None):
    h, w = mask.shape
    def inside(r, c):
        return 0 <= r < h and 0 <= c < w and mask[r, c]
    paths = []
    for r, c in np.argwhere(mask & (depth == 0)):
        r, c = int(r), int(c)
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            if row_direction is not None and dr != row_direction:
                continue
            if inside(r - dr, c) or inside(r, c - dc):
                continue
            if not (inside(r + dr, c) and inside(r, c + dc)):
                continue
            path = []
            while inside(r + len(path) * dr, c + len(path) * dc):
                q = (r + len(path) * dr, c + len(path) * dc)
                if depth[q] != len(path):
                    break
                path.append(q)
            if path:
                paths.append(tuple(path))
    return paths


def _arrow_cells(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return tuple(sorted((r + a, c + b) for a, b in offsets))


def _decode(inp):
    wall_color = int(inp[0, 0])
    assert 1 <= wall_color <= 7, "Frame encodes a divider color in 1..7"
    wall = inp == wall_color
    assert wall[0].all() and wall[-1].all() and wall[:, 0].all() and wall[:, -1].all()
    assert len(_components(wall)) == 1, "All walls attach to the frame"
    assert not np.any(wall[:-1, :-1] & wall[1:, :-1] & wall[:-1, 1:] & wall[1:, 1:])
    assert not np.any(inp == COLLISION), "Collision color is output only"
    regions = _components(~wall)
    assert regions
    owner = np.full(inp.shape, -1, dtype=np.int16)
    depths, periods = [], []
    cue_mask = np.zeros(inp.shape, dtype=bool)
    for i, mask in enumerate(regions):
        owner[mask] = i
        depth = _depth(mask)
        cues = mask & (inp != BACKGROUND) & (inp != ARROW)
        assert cues.any(), "Each region has one visible diagonal key"
        positions = set(map(tuple, np.argwhere(cues).tolist()))
        length = int(depth[cues].max()) + 1
        assert 1 <= length <= 4
        possibilities = set()
        for path in _corner_paths(mask, depth):
            if len(path) >= length and set(path[:length]) == positions:
                possibilities.add(tuple(int(inp[q]) for q in path[:length]))
        assert len(possibilities) == 1, "Key must uniquely encode a period"
        period = next(iter(possibilities))
        assert all(c in range(1, 8) and c != wall_color for c in period)
        cue_mask |= cues
        depths.append(depth)
        periods.append(period)
    arrows = []
    for mask in _components(inp == ARROW, D8):
        cells = set(map(tuple, np.argwhere(mask).tolist()))
        assert len(cells) == 3, "Arrows are separated three-cell motifs"
        matches = [(tip, di, tuple(sorted(cells))) for tip in sorted(cells) for di in D8
                   if set(_arrow_cells(tip, di)) == cells]
        assert len(matches) == 1, "Arrow geometry must determine one direction"
        assert len({int(owner[q]) for q in cells}) == 1
        arrows.append(matches[0])
    assert arrows
    return {"wall_color": wall_color, "wall": wall, "regions": regions,
            "owner": owner, "depths": depths, "periods": periods,
            "cues": cue_mask, "arrows": arrows,
            "protected": wall | cue_mask | (inp == ARROW)}


def _shots(info, distance=DISTANCE, depths=None):
    depths = info["depths"] if depths is None else depths
    h, w = info["wall"].shape
    groups, endpoints = Counter(), []
    for tip, (dr, dc), _ in info["arrows"]:
        q = (tip[0] + distance * dr, tip[1] + distance * dc)
        assert 0 <= q[0] < h and 0 <= q[1] < w, "Shot must stay in bounds"
        region = int(info["owner"][q])
        assert region >= 0, "An endpoint must be in open space, not on a wall"
        phase = int(depths[region][q]) % len(info["periods"][region])
        group = (region, phase)
        groups[group] += 1
        endpoints.append((q, group))
    return groups, endpoints


def _apply(inp, info, distance=DISTANCE, depths=None, coordinate_collisions=False):
    depths = info["depths"] if depths is None else depths
    counts, endpoints = _shots(info, distance, depths)
    coordinate_counts = Counter(q for q, _ in endpoints)
    out = inp.copy()
    for (i, phase), count in counts.items():
        period = info["periods"][i]
        target = (info["regions"][i] & (depths[i] % len(period) == phase)
                  & ~info["protected"])
        collide = count >= 2
        if coordinate_collisions:
            collide = any(coordinate_counts[q] >= 2 for q, group in endpoints
                          if group == (i, phase))
        out[target] = COLLISION if collide else period[phase]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _grid(input_grid)
    return _apply(inp, _decode(inp))


def _box_depth(mask):
    rr, cc = np.nonzero(mask)
    result = np.full(mask.shape, -1, dtype=np.int16)
    result[rr, cc] = np.minimum.reduce((rr - rr.min(), rr.max() - rr,
                                      cc - cc.min(), cc.max() - cc))
    return result


def _features(inp, info=None):
    info = _decode(inp) if info is None else info
    counts, endpoints = _shots(info)
    actual = _apply(inp, info)
    full = inp.copy()
    point_only = inp.copy()
    for i, mask in enumerate(info["regions"]):
        free = mask & ~info["protected"]
        period = np.array(info["periods"][i])
        full[free] = period[info["depths"][i][free] % len(period)]
    coordinate_counts = Counter(q for q, _ in endpoints)
    for q, (i, phase) in endpoints:
        if not info["protected"][q]:
            point_only[q] = COLLISION if coordinate_counts[q] > 1 else info["periods"][i][phase]
    cross = any(info["owner"][tip] != group[0] and counts[group] <= 2
                for (tip, _, _), (_, group) in zip(info["arrows"], endpoints))
    repeated = any(len(set(info["depths"][i][info["regions"][i]
                         & ~info["protected"]
                         & (info["depths"][i] % len(info["periods"][i]) == phase)].tolist())) >= 2
                   for i, phase in counts)
    inactive = any(np.any(mask & ~info["protected"]
                         & (info["depths"][i] % len(info["periods"][i]) == phase))
                   for i, mask in enumerate(info["regions"])
                   for phase in range(len(info["periods"][i])) if (i, phase) not in counts)
    same_color_distinct_states = any(
        period[p] == period[q] and min(2, counts[i, p]) != min(2, counts[i, q])
        and all(np.any(info["regions"][i] & ~info["protected"]
                       & (info["depths"][i] % len(period) == phase)) for phase in (p, q))
        for i, period in enumerate(info["periods"])
        for p in range(len(period)) for q in range(p + 1, len(period)))
    return {"shape": inp.shape, "wall_color": info["wall_color"],
            "regions": len(info["regions"]), "arrows": len(info["arrows"]),
            "directions": {a[1] for a in info["arrows"]},
            "lengths": set(map(len, info["periods"])),
            "single": any(n == 1 for n in counts.values()),
            "collision": any(n >= 2 for n in counts.values()),
            "distinct_endpoint_collision": not np.array_equal(actual, _apply(inp, info, coordinate_collisions=True)),
            "cross": cross, "repeat": repeated, "inactive": inactive,
            "same_color_distinct_states": same_color_distinct_states,
            "omit_A_differs": not np.array_equal(actual, point_only),
            "omit_B_differs": not np.array_equal(actual, full),
            "box_differs": not np.array_equal(actual, _apply(inp, info, depths=[_box_depth(m) for m in info["regions"]])),
            "L1_differs": not np.array_equal(actual, _apply(inp, info, depths=[_depth(m, D4) for m in info["regions"]]))}


def _distance_models(train):
    parsed = [(np.asarray(p["input"]), np.asarray(p["output"]), _decode(np.asarray(p["input"])))
              for p in train]
    possibilities = []
    for distance in range(1, 30):
        try:
            if all(np.array_equal(_apply(inp, info, distance), out) for inp, out, info in parsed):
                possibilities.append(distance)
        except AssertionError:
            continue
    return possibilities


def _layout(rng, shape, target, spur_budget, box=False):
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    wall[0] = wall[-1] = True
    wall[:, 0] = wall[:, -1] = True
    rooms = [(1, h - 2, 1, w - 2)]
    while len(rooms) < target:
        choices = [(j, axis) for j, (a, b, c, d) in enumerate(rooms)
                   for axis, length in ((0, b - a + 1), (1, d - c + 1)) if length >= 13]
        if not choices:
            break
        j, axis = choices[int(rng.integers(len(choices)))]
        a, b, c, d = rooms.pop(j)
        if axis == 0:
            q = int(rng.integers(a + 6, b - 5))
            wall[q, c - 1:d + 2] = True
            rooms.extend(((a, q - 1, c, d), (q + 1, b, c, d)))
        else:
            q = int(rng.integers(c + 6, d - 5))
            wall[a - 1:b + 2, q] = True
            rooms.extend(((a, b, c, q - 1), (a, b, q + 1, d)))
    # Branches intrude into rooms without disconnecting them. Their location,
    # length, attachment side and occasional elbow are independently sampled.
    for _ in range(spur_budget):
        for _attempt in range(40):
            a, b, c, d = rooms[int(rng.integers(len(rooms)))]
            side = int(rng.integers(4))
            if side < 2:
                if d - c < 4 or b - a < 5:
                    continue
                col = int(rng.integers(c + 2, d - 1))
                length = int(rng.integers(2, min(6, b - a - 2) + 1))
                row, dr, dc = (a - 1, 1, 0) if side == 0 else (b + 1, -1, 0)
            else:
                if b - a < 4 or d - c < 5:
                    continue
                row = int(rng.integers(a + 2, b - 1))
                length = int(rng.integers(2, min(6, d - c - 2) + 1))
                col, dr, dc = (c - 1, 0, 1) if side == 2 else (d + 1, 0, -1)
            cells = [(row + k * dr, col + k * dc) for k in range(1, length + 1)]
            if rng.random() < .35:
                er, ec = cells[-1]
                turn = int(rng.choice((-1, 1)))
                cells += [(er + k * dc * turn, ec + k * dr * turn)
                          for k in range(1, int(rng.integers(2, 5)))]
            if any(not (a <= r <= b and c <= q <= d) or wall[r, q] for r, q in cells):
                continue
            candidate = wall.copy()
            for p in cells:
                candidate[p] = True
            if len(_components(~candidate)) != len(rooms):
                continue
            if np.any(candidate[:-1, :-1] & candidate[1:, :-1] & candidate[:-1, 1:] & candidate[1:, 1:]):
                continue
            wall = candidate
            break
    if box:
        # A U attached to one room wall creates an enclosed region and a hole
        # in its neighbor. No fixed central cross or four-room template.
        choices = [(a, b, c, d) for a, b, c, d in rooms if b - a >= 10 and d - c >= 9]
        for a, b, c, d in rng.permutation(choices).tolist() if choices else []:
            top = a + 2
            bottom = min(b - 2, top + int(rng.integers(4, 7)))
            far = c + int(rng.integers(3, min(7, d - c - 2)))
            candidate = wall.copy()
            candidate[top, c - 1:far + 1] = True
            candidate[bottom, c - 1:far + 1] = True
            candidate[top:bottom + 1, far] = True
            if len(_components(~candidate)) != len(rooms) + 1:
                continue
            if np.any(candidate[:-1, :-1] & candidate[1:, :-1] & candidate[:-1, 1:] & candidate[1:, 1:]):
                continue
            wall = candidate
            break
    if rng.integers(2):
        wall = wall[::-1].copy()
    if rng.integers(2):
        wall = wall[:, ::-1].copy()
    return wall


def _setup(rng, wall, wall_color, test, essential):
    regions = _components(~wall)
    depths = [_bfs_depth(mask) for mask in regions]
    paths = [_corner_paths(m, d, -1 if test else 1) for m, d in zip(regions, depths)]
    assert all(paths)
    capacity = [max(map(len, ps)) for ps in paths]
    order = sorted(range(len(regions)), key=lambda i: (-capacity[i], i))
    if essential:
        assert len(order) >= 2 and capacity[order[0]] >= 4 and capacity[order[1]] >= 2
    if test:
        assert capacity[order[0]] >= 2
    inp = np.zeros(wall.shape, dtype=np.int8)
    inp[wall] = wall_color
    owner = np.full(wall.shape, -1, dtype=np.int16)
    periods, cues = [], np.zeros(wall.shape, dtype=bool)
    palette = [c for c in range(1, 8) if c != wall_color]
    for i, mask in enumerate(regions):
        owner[mask] = i
        if essential and i == order[0]:
            length = 3
        elif essential and i == order[1]:
            length = 2
        elif test and i == order[0]:
            length = int(rng.integers(2, min(4, capacity[i]) + 1))
        else:
            length = int(rng.integers(1, min(4 if test else 3, capacity[i]) + 1))
        options = [p for p in paths[i] if len(p) >= length]
        path = options[int(rng.integers(len(options)))][:length]
        period = tuple(map(int, rng.choice(palette, length, replace=False)))
        # Cycle positions are distinct addresses even if their colors repeat.
        # The single-shot lesson deliberately exposes this distinction.
        if length >= 2 and ((essential and i == order[0]) or rng.random() < .25):
            period = period[:-1] + (period[0],)
        periods.append(period)
        for q, color in zip(path, period):
            inp[q] = color
            cues[q] = True
    info = {"wall": wall, "regions": regions, "depths": depths,
            "periods": periods, "owner": owner, "cues": cues,
            "protected": wall | cues, "arrows": [], "wall_color": wall_color}
    return inp, info


def _candidates(inp, info):
    h, w = inp.shape
    result = []
    for r, c in np.argwhere(inp == BACKGROUND):
        r, c = int(r), int(c)
        source = int(info["owner"][r, c])
        for dr, dc in D8:
            tip, di = (r, c), (dr, dc)
            endpoint = (r + DISTANCE * dr, c + DISTANCE * dc)
            if not (0 <= endpoint[0] < h and 0 <= endpoint[1] < w):
                continue
            i = int(info["owner"][endpoint])
            if i < 0:
                continue
            cells = _arrow_cells(tip, di)
            if any(not (0 <= a < h and 0 <= b < w) or inp[a, b] != BACKGROUND
                   or info["owner"][a, b] != source for a, b in cells):
                continue
            phase = int(info["depths"][i][endpoint]) % len(info["periods"][i])
            result.append((tip, di, cells, endpoint, (i, phase)))
    return result


def _compatible(candidate, selected):
    occupied = {p for obj in selected for p in obj[2]}
    return not any((r + dr, c + dc) in occupied for r, c in candidate[2]
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _select(rng, inp, info, kind, essential):
    candidates = _candidates(inp, info)
    assert candidates
    shuffled = [candidates[int(j)] for j in rng.permutation(len(candidates))]
    by_group = defaultdict(list)
    for obj in shuffled:
        by_group[obj[4]].append(obj)
    selected = []
    if kind == "single" and essential:
        # A deep endpoint whose periodic class contains repeated layers.
        options = [o for o in shuffled if len(info["periods"][o[4][0]]) == 3
                   and info["depths"][o[4][0]][o[3]] >= 3 and o[4][1] in (0, 2)]
        assert options
        selected.append(options[0])
    elif kind in ("collision", "mixed"):
        groups = list(by_group)
        rng.shuffle(groups)
        for group in groups:
            pool = by_group[group]
            for first in pool[:60]:
                second = next((o for o in pool if o[3] != first[3]
                               and _compatible(o, [first])), None)
                if second is not None:
                    selected = [first, second]
                    break
            if selected:
                break
        assert selected
    else:
        selected.append(shuffled[0])
    # A shot through a wall teaches target-region addressing and transparency.
    if essential:
        counts = Counter(o[4] for o in selected)
        cross = next((o for o in shuffled if info["owner"][o[0]] != o[4][0]
                      and o[4] not in counts and _compatible(o, selected)
                      and not (kind == "single" and o[4][0] == selected[0][4][0]
                               and o[4][1] == 2 - selected[0][4][1])), None)
        assert cross is not None
        selected.append(cross)
    desired = int(rng.integers(3, 7)) if essential else int(rng.integers(1, 11))
    desired = max(desired, len(selected))
    for obj in shuffled:
        if len(selected) >= desired:
            break
        if not _compatible(obj, selected):
            continue
        counts = Counter(o[4] for o in selected)
        if kind == "single" and obj[4] in counts:
            continue
        # Retain at least one unused phase of the three-entry teaching region.
        if essential and kind == "single":
            i = obj[4][0]
            used = {o[4][1] for o in selected if o[4][0] == i} | {obj[4][1]}
            if len(info["periods"][i]) == 3 and len(used) == 3:
                continue
            first_group = selected[0][4]
            if i == first_group[0] and obj[4][1] == 2 - first_group[1]:
                continue
        if essential and kind == "collision" and counts.get(obj[4], 0) >= 2:
            continue
        selected.append(obj)
    for tip, di, cells, _, _ in selected:
        for q in cells:
            inp[q] = ARROW
        info["arrows"].append((tip, di, cells))
    info["protected"] |= inp == ARROW
    return inp, info


def _make_pair(rng, wall_color, test=False, kind="mixed", essential=False):
    for attempt in range(60):
        if attempt == 59:
            shape, target, spurs, box = (28, 29), 2, 2, test
        else:
            low = 23 if essential else (19 if test else 12)
            high = 30 if test else 29
            shape = tuple(map(int, rng.integers(low, high + 1, size=2)))
            target = int(rng.integers(2, 4 if essential else (7 if test else 5)))
            spurs = int(rng.integers(1 if essential else (2 if test else 0), 6 if test else 4))
            box = bool((test or not essential) and rng.random() < .4)
        wall = _layout(rng, shape, target, spurs, box)
        try:
            inp, info = _setup(rng, wall, wall_color, test, essential)
            inp, info = _select(rng, inp, info, kind, essential)
            # Labels use the construction's geometry; solve separately decodes
            # only pixels. Neither code path uses a stored answer bank.
            out = _apply(inp, info)
            decoded = _decode(inp)
            assert np.array_equal(out, _apply(inp, decoded))
            f = _features(inp, decoded)
            assert f["omit_A_differs"] and f["omit_B_differs"]
            if essential and kind == "single":
                assert f["single"] and not f["collision"] and f["repeat"] and f["inactive"]
                assert f["box_differs"] and f["L1_differs"] and f["cross"]
                assert f["same_color_distinct_states"]
            if essential and kind == "collision":
                assert f["single"] and f["collision"] and f["distinct_endpoint_collision"]
            return {"input": inp, "output": out}
        except AssertionError:
            continue
    raise RuntimeError("Bounded contour-resonance construction failed")


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    for _attempt in range(12):
        puzzle, seen = {"train": [], "test": []}, set()
        wall_colors = list(map(int, rng.permutation(np.arange(1, 8))))
        kinds = ["single", "collision"] + [str(rng.choice(("single", "collision", "mixed")))
                                            for _ in range(num_train - 2)]
        specs = [(kinds[int(j)], int(j) < 2) for j in rng.permutation(num_train)]
        for number in range(num_train + num_test):
            test = number >= num_train
            kind, essential = (str(rng.choice(("single", "collision", "mixed"))), False) if test else specs[number]
            for _unique in range(8):
                pair = _make_pair(rng, wall_colors[number % 7], test, kind, essential)
                if _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("Unable to construct distinct inputs")
            seen.add(_key(pair["input"]))
            puzzle["test" if test else "train"].append(pair)
        if _distance_models(puzzle["train"]) == [DISTANCE]:
            validate_generated_puzzle(puzzle)
            return puzzle
    raise RuntimeError("Training did not uniquely expose the firing distance")


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, train_features = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _decode(inp)
            assert all(np.array_equal(d, _bfs_depth(m)) for m, d in zip(info["regions"], info["depths"]))
            assert np.array_equal(out, _apply(inp, info)), "Incorrect resonance output"
            assert np.array_equal(inp[info["protected"]], out[info["protected"]])
            assert not np.array_equal(inp, out)
            counts, _ = _shots(info)
            # Every unprotected contour class is uniform. Its state depends on
            # shot multiplicity, not shooter color, coordinate, or iteration order.
            for i, mask in enumerate(info["regions"]):
                for phase, color in enumerate(info["periods"][i]):
                    free = mask & ~info["protected"] & (info["depths"][i] % len(info["periods"][i]) == phase)
                    expected = BACKGROUND if counts[i, phase] == 0 else (color if counts[i, phase] == 1 else COLLISION)
                    assert np.all(out[free] == expected)
            reversed_info = dict(info, arrows=info["arrows"][::-1])
            assert np.array_equal(out, _apply(inp, reversed_info))
            f = _features(inp, info)
            assert f["omit_A_differs"] and f["omit_B_differs"]
            if split == "train":
                train_features.append(f)
    assert any(f["single"] and not f["collision"] and f["repeat"] and f["inactive"]
               and f["box_differs"] and f["L1_differs"] and f["cross"]
               and f["same_color_distinct_states"] for f in train_features)
    assert any(f["single"] and f["collision"] and f["distinct_endpoint_collision"] for f in train_features)
    assert {2, 3} <= set.union(*(f["lengths"] for f in train_features))
    assert _distance_models(puzzle["train"]) == [DISTANCE]


def self_test():
    """Bounded 42-seed audit plus regressions, count checks and mutations."""
    records, puzzles = [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(_puzzle_key(p))
        for split in ("train", "test"):
            for pair in p[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                records.append(_features(pair["input"], info))
        # Rotating and reflecting preserves all geometric semantics, including
        # key order, tip-relative shots and collision equivalence classes.
        if seed < 4:
            for pair in p["train"] + p["test"]:
                for transform in (np.rot90, np.fliplr):
                    assert np.array_equal(solve(transform(pair["input"])), transform(pair["output"]))
    assert len(puzzles) == 42
    assert len({f["shape"] for f in records}) >= 70
    assert len({f["regions"] for f in records}) >= 4
    assert len({f["arrows"] for f in records}) >= 8
    assert {f["wall_color"] for f in records} == set(range(1, 8))
    assert set.union(*(f["directions"] for f in records)) == set(D8)
    assert set.union(*(f["lengths"] for f in records)) == {1, 2, 3, 4}
    assert sum(f["distinct_endpoint_collision"] for f in records) >= 42
    assert sum(f["box_differs"] and f["L1_differs"] for f in records) >= 42
    assert sum(f["repeat"] for f in records) >= 42
    assert sum(f["cross"] for f in records) >= 42
    assert sum(f["same_color_distinct_states"] for f in records) >= 42
    _regressions()
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(617, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(generate_puzzle(0))
    original = generate_puzzle(3, 2, 1)
    broken = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in original[s]] for s in original}
    broken["test"][0]["output"][0, 0] = 0
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Changed frame accepted")
    return {"seeds": 42, "pairs": len(records), "shapes": len({f["shape"] for f in records}),
            "region_counts": sorted({f["regions"] for f in records}),
            "arrow_counts": sorted({f["arrows"] for f in records})}


def _regressions():
    # Two distinct endpoints address left-room depths 3 and 3. A third shot
    # passes through the internal wall and strikes another arrow. That arrow
    # still fires, making the right-room phase a collision as well.
    inp = np.zeros((19, 25), dtype=np.int8)
    inp[0] = inp[-1] = 7
    inp[:, 0] = inp[:, -1] = inp[:, 12] = 7
    for q, color in (((1, 1), 1), ((2, 2), 2), ((3, 3), 1),
                     ((1, 13), 4), ((2, 14), 5)):
        inp[q] = color
    for tip, di in (((8, 3), (0, 1)), ((12, 3), (0, 1)),
                    ((4, 10), (0, 1)), ((4, 15), (1, 0))):
        for q in _arrow_cells(tip, di):
            inp[q] = ARROW
    info = _decode(inp)
    counts, endpoints = _shots(info)
    assert counts == Counter({(0, 0): 2, (1, 0): 2})
    assert len({q for q, _ in endpoints}) == 4
    out = solve(inp)
    # Different coordinates and different depths with the same residue are
    # whole-class collisions. Repeated colors at phases 0 and 2 stay separate.
    assert out[1, 5] == COLLISION and out[4, 5] == COLLISION
    assert out[3, 5] == BACKGROUND and out[2, 5] == BACKGROUND
    assert np.array_equal(out[inp == ARROW], inp[inp == ARROW])
    assert out[3, 15] == COLLISION and out[2, 15] == BACKGROUND
    # Removing a shot changes exactly the corresponding unprotected class.
    fewer = inp.copy()
    for q in _arrow_cells((12, 3), (0, 1)):
        fewer[q] = BACKGROUND
    smaller = solve(fewer)
    assert smaller[1, 5] == 1 and smaller[4, 5] == 1
    assert smaller[3, 5] == BACKGROUND and smaller[3, 15] == COLLISION


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
