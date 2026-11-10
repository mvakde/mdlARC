"""Interference gates: simultaneous vector keys alter a collection network.

All solving conventions are fixed or encoded in the input. No parent module,
stored example, random state, or training pair is consulted by solve().
"""

from collections import Counter, deque
import json
import numpy as np


BACKGROUND, WALL, GATE = 0, 5, 6
DISTANCE, CAPACITY = 5, 9
VECTORS = ((-1, 0), (1, 0), (0, -1), (0, 1),
           (-1, -1), (-1, 1), (1, -1), (1, 1))
STEPS = VECTORS[:4]


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "Grid dimensions must be 1..30.")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grid cells must be integers 0..9.")
    return a.astype(np.int8, copy=True)


def _arrow(tip, vector):
    r, c = tip
    dr, dc = vector
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    cells = tuple(sorted((r + a, c + b) for a, b in offsets))
    endpoint = (r + DISTANCE * dr, c + DISTANCE * dc)
    return (tuple(tip), tuple(vector), cells, endpoint)


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if r or c) if diagonal else STEPS
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        for r, c in queue:
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        result.append(set(queue))
    return result


def _parse(grid):
    """Read collector, its swatch, arrow objects and endpoints from pixels."""
    candidates = []
    for color in range(10):
        if color in (BACKGROUND, WALL, GATE) or np.count_nonzero(grid == color) != 8:
            continue
        rr, cc = np.where(grid == color)
        r, c = int(rr.min()), int(cc.min())
        if rr.max() != r + 2 or cc.max() != c + 2:
            continue
        patch = grid[r:r + 3, c:c + 3]
        payload = int(patch[1, 1])
        ring = np.ones((3, 3), dtype=bool)
        ring[1, 1] = False
        if payload not in (BACKGROUND, WALL, GATE, color) and np.all(patch[ring] == color):
            candidates.append(((r, c), color, payload))
    _require(len(candidates) == 1, "Need a unique marked 3x3 collector.")
    pos, slot, payload = candidates[0]
    footprint = np.zeros(grid.shape, dtype=bool)
    r, c = pos
    footprint[r:r + 3, c:c + 3] = True
    arrow_mask = ~np.isin(grid, (BACKGROUND, WALL, GATE, slot, payload))
    objects = []
    for cells in _components(arrow_mask, diagonal=True):
        _require(len(cells) == 3 and len({int(grid[p]) for p in cells}) == 1,
                 "Arrows must be isolated monochrome triominoes.")
        matches = [_arrow(tip, vector) for tip in sorted(cells) for vector in VECTORS
                   if set(_arrow(tip, vector)[2]) == cells]
        _require(len(matches) == 1, "Arrow geometry must specify one tip/direction.")
        objects.append(matches[0])
    _require(objects, "Need at least one vector key.")
    _require(np.any(grid == GATE), "Need at least one gate.")
    for p in np.argwhere(grid == GATE):
        r, c = map(int, p)
        vertical = 0 < r < grid.shape[0] - 1 and grid[r - 1, c] == WALL and grid[r + 1, c] == WALL
        horizontal = 0 < c < grid.shape[1] - 1 and grid[r, c - 1] == WALL and grid[r, c + 1] == WALL
        _require(vertical or horizontal, "Each gate replaces a straight interior wall cell.")
    _require(all(0 <= o[3][0] < grid.shape[0] and 0 <= o[3][1] < grid.shape[1]
                 for o in objects), "Endpoints must remain in bounds.")
    return pos, slot, payload, footprint, objects


def _distances(blocked, footprint):
    distance = np.full(blocked.shape, -1, dtype=np.int32)
    queue = list(map(tuple, np.argwhere(footprint).tolist()))
    distance[footprint] = 0
    h, w = blocked.shape
    for r, c in queue:
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance


def _gate_depth(blocked, footprint, opening):
    """Minimum number of opened gate pixels on a route (a 0/1 BFS)."""
    depth = np.full(blocked.shape, blocked.size + 1, dtype=np.int32)
    starts = list(map(tuple, np.argwhere(footprint).tolist()))
    depth[footprint] = 0
    queue = deque(starts)
    h, w = blocked.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc]:
                weight = int(opening[nr, nc])
                cost = int(depth[r, c]) + weight
                if cost < depth[nr, nc]:
                    depth[nr, nc] = cost
                    (queue.append if weight else queue.appendleft)((nr, nc))
    depth[depth == blocked.size + 1] = -1
    return depth


def _selection(grid, payload, footprint, distance):
    candidates = (grid == payload) & ~footprint & (distance >= 0)
    ordered = np.sort(distance[candidates])
    if len(ordered) > CAPACITY:
        _require(ordered[8] < ordered[9], "A capacity tie must not straddle ninth/tenth.")
    threshold = int(ordered[min(8, len(ordered) - 1)]) if len(ordered) else -1
    return candidates & (distance <= threshold)


def _analyze(grid, gate_mode="single"):
    pos, slot, payload, footprint, objects = _parse(grid)
    counts = Counter(o[3] for o in objects)
    opening = np.zeros(grid.shape, dtype=bool)
    for p in map(tuple, np.argwhere(grid == GATE).tolist()):
        opening[p] = (counts[p] == 1 if gate_mode == "single" else
                      counts[p] > 0 if gate_mode == "any" else
                      gate_mode == "all")
    blocked = (grid == WALL) | ((grid == GATE) & ~opening)
    distance = _distances(blocked, footprint)
    return dict(pos=pos, slot=slot, payload=payload, footprint=footprint,
                objects=objects, counts=counts, opening=opening,
                blocked=blocked, distance=distance)


def _render_result(grid, info, selected):
    out = grid.copy()
    out[info["opening"]] = BACKGROUND
    out[selected] = BACKGROUND
    r, c = info["pos"]
    out[r:r + 3, c:c + 3] = info["slot"]
    for i in range(int(selected.sum())):
        out[r + i // 3, c + i % 3] = info["payload"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    info = _analyze(grid)
    selected = _selection(grid, info["payload"], info["footprint"], info["distance"])
    return _render_result(grid, info, selected)


def _manhattan(shape, pos):
    rr, cc = np.indices(shape)
    r, c = pos
    return np.maximum(r - rr, 0) + np.maximum(rr - r - 2, 0) + np.maximum(c - cc, 0) + np.maximum(cc - c - 2, 0)


def _choose(rng, items):
    return items[int(rng.integers(len(items)))]


def _network(rng, shape, cuts, serial_witness=False, teaching=False):
    """Recursive room cuts, varied apertures, plus independent rails/turns."""
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    rooms = [(0, 0, h - 1, w - 1)]
    lines = []
    for _ in range(cuts):
        choices = [(i, axis) for i, (r0, c0, r1, c1) in enumerate(rooms)
                   for axis in (0, 1) if (r1 - r0 if axis == 0 else c1 - c0) >= 12]
        if not choices:
            return None
        # Large rooms are preferred but the tree's orientation and branch vary.
        weights = np.array([(rooms[i][2] - rooms[i][0] + 1) *
                            (rooms[i][3] - rooms[i][1] + 1) for i, _ in choices], dtype=float)
        index, axis = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        r0, c0, r1, c1 = rooms.pop(index)
        if axis == 0:
            r = int(rng.integers(r0 + 6, r1 - 5))
            cells = [(r, c) for c in range(c0, c1 + 1)]
            rooms.extend(((r0, c0, r - 1, c1), (r + 1, c0, r1, c1)))
        else:
            c = int(rng.integers(c0 + 6, c1 - 5))
            cells = [(r, c) for r in range(r0, r1 + 1)]
            rooms.extend(((r0, c0, r1, c - 1), (r0, c + 1, r1, c1)))
        for p in cells:
            wall[p] = True
        lines.append(cells)
    # Internal Ls, short rails, and rectangular pockets add detours without
    # redefining the room tree. They are sampled separately from gate states.
    for r0, c0, r1, c1 in rooms:
        if rng.random() < .38 and r1 - r0 >= 4 and c1 - c0 >= 4:
            a = int(rng.integers(r0 + 1, r1 - 2))
            b = int(rng.integers(c0 + 1, c1 - 2))
            ar = int(rng.integers(a + 2, r1 + 1))
            bc = int(rng.integers(b + 2, c1 + 1))
            wall[a, b:bc + 1] = True
            if rng.random() < .75:
                wall[a:ar + 1, bc] = True
            if rng.random() < .25:
                wall[ar, b:bc + 1] = True
                wall[a:ar + 1, b] = True
    gates, states = [], []
    order = rng.permutation(len(lines))
    # Teach a jam in essential examples; ordinary scenes sample all event mixes.
    for logical_index, line_index in enumerate(order):
        cells = lines[int(line_index)]
        possible = [p for p in cells[2:-2]
                    if ((0 < p[0] < h - 1 and wall[p[0] - 1, p[1]] and wall[p[0] + 1, p[1]]) or
                        (0 < p[1] < w - 1 and wall[p[0], p[1] - 1] and wall[p[0], p[1] + 1]))]
        if not possible:
            return None
        gate = _choose(rng, possible)
        gates.append(gate)
        states.append(1 if logical_index == 0 else 2 if logical_index == 1 and teaching else
                      1 if serial_witness else int(rng.choice((0, 1, 2), p=(.3, .5, .2))))
        # A second aperture may provide redundancy or remain a missed gate.
        if rng.random() < .3:
            extra = [p for p in possible if max(abs(p[0] - gate[0]), abs(p[1] - gate[1])) >= 4]
            if extra:
                gates.append(_choose(rng, extra))
                states.append(0 if serial_witness and logical_index == 1 else int(rng.choice((0, 1), p=(.65, .35))))
    if serial_witness and 0 not in states:
        # Teach that an unhit gate stays shut, alongside single and jam events.
        extra = [p for line in lines for p in line[2:-2]
                 if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in gates)]
        if not extra:
            return None
        gates.append(_choose(rng, extra))
        states.append(0)
    return wall, gates, states


def _can_place(obj, grid, occupied):
    h, w = grid.shape
    return (0 <= obj[3][0] < h and 0 <= obj[3][1] < w and
            all(0 <= r < h and 0 <= c < w and grid[r, c] == BACKGROUND and
                not any((r + a, c + b) in occupied for a in (-1, 0, 1) for b in (-1, 0, 1))
                for r, c in obj[2]))


def _keys(rng, grid, gates, states, colors, triple=False, prefer_transparent=False):
    objects, occupied = [], set()
    for gate, state in zip(gates, states):
        want = 0 if state == 0 else 1 if state == 1 else 3 if triple == "force" or (triple and rng.random() < .5) else 2
        candidates = [_arrow((gate[0] - 5 * dr, gate[1] - 5 * dc), (dr, dc)) for dr, dc in VECTORS]
        options = [o for o in candidates if _can_place(o, grid, occupied)]
        # Exhaustively enumerate compatible subsets of at most three shooters.
        from itertools import combinations
        subsets = [subset for subset in combinations(options, want)
                   if all(max(abs(r - rr), abs(c - cc)) > 1
                          for i, o in enumerate(subset) for other in subset[:i]
                          for r, c in o[2] for rr, cc in other[2])]
        if not subsets:
            return None
        if prefer_transparent and want == 1:
            through = [subset for subset in subsets if any(
                grid[subset[0][0][0] + s * subset[0][1][0],
                     subset[0][0][1] + s * subset[0][1][1]] in (WALL, GATE)
                for s in range(1, DISTANCE))]
            if through:
                subsets = through
        for o in _choose(rng, subsets):
            objects.append(o)
            occupied.update(o[2])
    # Non-gate endpoints teach that a key is not a path or a universal impact.
    for _ in range(int(rng.integers(1, 5))):
        for _ in range(70):
            tip = (int(rng.integers(grid.shape[0])), int(rng.integers(grid.shape[1])))
            o = _arrow(tip, _choose(rng, VECTORS))
            if _can_place(o, grid, occupied) and o[3] not in gates:
                objects.append(o)
                occupied.update(o[2])
                break
    monochrome = rng.random() < .35
    base = int(_choose(rng, colors))
    for o in objects:
        color = base if monochrome else int(_choose(rng, colors))
        for p in o[2]:
            grid[p] = color
    return objects


def _pick(rng, mask, count, forced=(), style="scatter"):
    available = set(map(tuple, np.argwhere(mask).tolist()))
    picked = list(dict.fromkeys(forced))
    if count < len(picked) or count > len(available) or not set(picked) <= available:
        return None
    available.difference_update(picked)
    anchor = _choose(rng, sorted(available)) if available else (0, 0)
    while len(picked) < count:
        choices = sorted(available)
        if style != "scatter":
            ar, ac = anchor
            scores = [((r - ar) ** 2 + (c - ac) ** 2 if style == "cluster" else
                       abs(r - ar) * mask.shape[1] + abs(c - ac)) for r, c in choices]
            best = min(scores)
            choices = [p for p, score in zip(choices, scores) if score == best]
        p = _choose(rng, choices)
        picked.append(p)
        available.remove(p)
    return picked


def _place_payload(rng, grid, info, load, partial_witness=False, detour=False):
    d = info["distance"]
    pre = _analyze(grid, "none")["distance"]
    any_distance = _analyze(grid, "any")["distance"]
    free = grid == BACKGROUND
    active, inactive = free & (d >= 0), free & (d < 0)
    newly = active & (pre < 0)
    jam_release = inactive & (any_distance >= 0)
    if not newly.any() or (partial_witness and not jam_release.any()):
        return False
    if load == 0 and not inactive.any():
        return False
    if partial_witness:
        depth = _gate_depth(info["blocked"], info["footprint"], info["opening"])
        newly &= depth >= 2
        if not newly.any():
            return False
    style = str(rng.choice(("scatter", "cluster", "line")))
    if load <= 9:
        forced = [_choose(rng, list(map(tuple, np.argwhere(newly).tolist())))] if load else []
        supply = _pick(rng, active, load, forced, style)
        if supply is None:
            return False
    else:
        metric = _manhattan(grid.shape, info["pos"])
        supply = None
        thresholds = rng.permutation(np.unique(d[active]))
        for threshold in thresholds:
            near, far = active & (d <= threshold), active & (d > threshold)
            if near.sum() < 9 or far.sum() < load - 9 or not (near & newly).any():
                continue
            forced = [_choose(rng, list(map(tuple, np.argwhere(near & newly).tolist())))]
            retained_forced = []
            if detour:
                a = max(map(tuple, np.argwhere(near).tolist()), key=lambda p: int(metric[p]))
                candidates = far & (metric < metric[a])
                if not candidates.any():
                    continue
                b = _choose(rng, list(map(tuple, np.argwhere(candidates).tolist())))
                forced.append(a)
                retained_forced.append(b)
                # At least two collected cells touch: counting whole payload
                # components cannot substitute for counting transferred cells.
                adjacent = [(p, (p[0] + dr, p[1] + dc))
                            for p in map(tuple, np.argwhere(near).tolist())
                            for dr, dc in ((1, 0), (0, 1))
                            if p[0] + dr < grid.shape[0] and p[1] + dc < grid.shape[1]
                            and near[p[0] + dr, p[1] + dc]]
                if not adjacent:
                    continue
                forced.extend(_choose(rng, adjacent))
            collected = _pick(rng, near, 9, forced, style)
            retained = _pick(rng, far, load - 9, retained_forced, style)
            if collected is not None and retained is not None:
                supply = collected + retained
                break
        if supply is None:
            return False
    forced = [_choose(rng, list(map(tuple, np.argwhere(jam_release).tolist())))] if partial_witness else []
    distraction = _pick(rng, inactive, min(int(inactive.sum()), int(rng.integers(max(1, len(forced)), 9))), forced, style)
    if distraction is None:
        return False
    for p in supply + distraction:
        grid[p] = info["payload"]
    return True


def _make_pair(rng, load, witness, shifted):
    for _ in range(160):
        h, w = map(int, rng.integers(23 if shifted else 18, 31 if shifted else 28, 2))
        cuts = int(rng.integers(4, 7)) if shifted else 3 if witness == "partial" else int(rng.integers(2, 4))
        network = _network(rng, (h, w), cuts, serial_witness=witness == "partial",
                           teaching=witness in ("partial", "capacity"))
        if network is None:
            continue
        wall, gates, states = network
        grid = np.zeros((h, w), dtype=np.int8)
        grid[wall] = WALL
        for p in gates:
            grid[p] = GATE
        candidates = [(r, c) for r in range(h - 2) for c in range(w - 2)
                      if np.all(grid[r:r + 3, c:c + 3] == BACKGROUND)]
        if not candidates:
            continue
        pos = _choose(rng, candidates)
        palette = [int(c) for c in rng.permutation((1, 2, 3, 4, 7, 8, 9))]
        slot, payload = palette[:2]
        r, c = pos
        grid[r:r + 3, c:c + 3] = slot
        grid[r + 1, c + 1] = payload
        objects = _keys(rng, grid, gates, states, palette[2:],
                        triple="force" if witness == "capacity" else "mixed" if shifted else False,
                        prefer_transparent=witness == "capacity")
        if objects is None:
            continue
        info = _analyze(grid)
        if witness in ("partial", "capacity"):
            hits = [o for o in objects if o[3] in gates]
            if not (any(o[1][0] * o[1][1] == 0 for o in hits) and
                    any(o[1][0] * o[1][1] != 0 for o in hits)):
                continue
            if witness == "capacity" and not any(info["counts"][o[3]] == 1 and
                    any(grid[o[0][0] + step * o[1][0], o[0][1] + step * o[1][1]] in (WALL, GATE)
                        for step in range(1, DISTANCE)) for o in hits):
                continue
        if not _place_payload(rng, grid, info, load, partial_witness=witness == "partial", detour=witness == "capacity"):
            continue
        output = solve(grid)
        return {"input": grid, "output": output}
    raise RuntimeError("Bounded network construction could not meet the evidence obligations.")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test) and
            2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError("Require integer counts 2..10 train, 1..4 test, train > test.")


def _signature(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    specs = [(int(rng.choice((2, 4))), "partial"), (int(rng.integers(10, 19)), "capacity")]
    for _ in range(num_train - 2):
        specs.append((int(rng.integers(0, 19)), "ordinary"))
    specs = [specs[int(i)] for i in rng.permutation(len(specs))]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, requests in (("train", specs), ("test", [(int(rng.integers(0, 19)), "ordinary") for _ in range(num_test)])):
        for load, witness in requests:
            for _ in range(12):
                pair = _make_pair(rng, load, witness, shifted=split == "test")
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct input construction exhausted.")
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(input_grid):
    grid = _grid(input_grid)
    info = _analyze(grid)
    d, fp, payload = info["distance"], info["footprint"], info["payload"]
    selected = _selection(grid, payload, fp, d)
    supply = (grid == payload) & ~fp
    retained = supply & (d >= 0) & ~selected
    pre = _analyze(grid, "none")["distance"]
    alternative = _analyze(grid, "any")["distance"]
    geometric = _manhattan(grid.shape, info["pos"])
    gates = list(map(tuple, np.argwhere(grid == GATE).tolist()))
    depth = _gate_depth(info["blocked"], fp, info["opening"])
    hits = [o for o in info["objects"] if o[3] in gates]
    return dict(shape=list(grid.shape), collector=list(info["pos"]),
                slot=info["slot"], payload=payload, arrows=len(info["objects"]),
                arrow_colors=len({int(grid[o[2][0]]) for o in info["objects"]}),
                directions=[list(v) for v in sorted({o[1] for o in info["objects"]})],
                gates=len(gates), opened=int(info["opening"].sum()),
                jammed=sum(info["counts"][p] >= 2 for p in gates),
                missed=sum(info["counts"][p] == 0 for p in gates),
                max_convergence=max(info["counts"][p] for p in gates),
                nongate_shots=sum(o[3] not in gates for o in info["objects"]),
                cardinal_keys=sum(o[1][0] * o[1][1] == 0 for o in hits),
                diagonal_keys=sum(o[1][0] * o[1][1] != 0 for o in hits),
                through_wall_keys=sum(any(grid[o[0][0] + s * o[1][0], o[0][1] + s * o[1][1]] in (WALL, GATE)
                                          for s in range(1, DISTANCE)) for o in hits),
                transparent_open_keys=sum(info["counts"][o[3]] == 1 and
                                          any(grid[o[0][0] + s * o[1][0], o[0][1] + s * o[1][1]] in (WALL, GATE)
                                              for s in range(1, DISTANCE)) for o in hits),
                selected=int(selected.sum()), reachable=int((supply & (d >= 0)).sum()),
                excess=int(retained.sum()), unreachable=int((supply & (d < 0)).sum()),
                newly_collected=int((selected & (pre < 0)).sum()),
                serial_collected=int((selected & (depth >= 2)).sum()),
                touching_collected=bool(np.any(selected[1:] & selected[:-1]) or
                                        np.any(selected[:, 1:] & selected[:, :-1])),
                collision_release=int((supply & (d < 0) & (alternative >= 0)).sum()),
                path_beats_manhattan=bool(selected.any() and retained.any() and
                                          geometric[selected].max() > geometric[retained].min()),
                capacity_gap=int(d[retained].min() - d[selected].max()) if retained.any() else 0,
                max_distance=int(d.max()),
                open_components=len(_components(~info["blocked"])))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema.")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists.")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema.")
            grid, out = _grid(pair["input"]), _grid(pair["output"])
            _require(grid.shape == out.shape, "Dimensions must be preserved.")
            key = _signature(grid)
            _require(key not in seen, "Inputs must be distinct.")
            seen.add(key)
            info = _analyze(grid)
            selected = _selection(grid, info["payload"], info["footprint"], info["distance"])
            expected = _render_result(grid, info, selected)
            _require(np.array_equal(out, expected), "Incorrect endpoint/topology/capacity/prefix output.")
            changed = info["opening"] | selected | info["footprint"]
            _require(np.array_equal(grid[~changed], out[~changed]), "Unrelated cells changed.")
            _require(int(selected.sum()) <= 9, "Collector overflow.")
            _require(int(((grid == info["payload"]) & ~info["footprint"]).sum()) ==
                     int(((out == info["payload"]) & ~info["footprint"]).sum()) + int(selected.sum()),
                     "Payload removal must conserve collected cell count.")
            features[split].append(pair_features(grid))
    train = features["train"]
    _require(any(f["selected"] in (2, 4) and f["serial_collected"] and f["collision_release"]
                 and f["cardinal_keys"] and f["diagonal_keys"] and f["missed"] for f in train),
             "Training must expose partial prefix, serial gates, both arrow geometries and collision-blocked payload.")
    _require(any(f["excess"] and f["path_beats_manhattan"] and f["newly_collected"]
                 and f["transparent_open_keys"] and f["cardinal_keys"] and f["diagonal_keys"]
                 and f["max_convergence"] >= 3 and f["touching_collected"] for f in train),
             "Training must distinguish path ranking, transparent shots, triple jamming and capacity from simpler shortcuts.")
    _require(any(f["nongate_shots"] for f in train), "Training needs ignored non-gate endpoints.")


def self_test():
    """Bounded 42-seed structural audit plus semantic and mutation checks."""
    _semantic_fixtures()
    records, signatures, normalized = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_signature(p["input"]) for s in ("train", "test") for p in puzzle[s]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                before = grid.copy()
                _require(np.array_equal(solve(grid), pair["output"]), "Input-only solve mismatch.")
                _require(np.array_equal(grid, before), "Solver mutated its input.")
                info = _analyze(grid)
                selected = _selection(grid, info["payload"], info["footprint"], info["distance"])
                # Reverse the object ordering: simultaneous endpoints cannot depend on it.
                _require(Counter(o[3] for o in reversed(info["objects"])) == info["counts"], "Order-dependent impacts.")
                mapping = {}
                normalized.add((grid.shape, tuple(mapping.setdefault(int(x), len(mapping)) for x in grid.flat)))
                records.append(dict(pair_features(grid), split=split))
                # Removing A's gate operation changes a counter or selected supply
                # in every positive example, rather than only an isolated arrow panel.
                if selected.any():
                    closed = _analyze(grid, "none")
                    _require(np.any(selected & (closed["distance"] < 0)), "Arrow keys have no collection consequence.")
    _require(len(signatures) == 42 and len(normalized) >= 180, "Structural diversity collapsed.")
    for field, minimum in (("arrows", 8), ("gates", 5), ("selected", 8), ("open_components", 3), ("max_distance", 20)):
        _require(len({f[field] for f in records}) >= minimum, "Insufficient variation in " + field)
    _require(len({tuple(f["shape"]) for f in records}) >= 60, "Canvas variation collapsed.")
    _require({tuple(v) for f in records for v in f["directions"]} == set(VECTORS), "Missing compass direction.")
    _require(any(f["max_convergence"] >= 3 for f in records), "No triple convergence.")
    _require(any(f["selected"] == 0 for f in records), "No zero-load case.")
    for split in ("train", "test"):
        local = [f for f in records if f["split"] == split]
        _require(any(f["excess"] for f in local) and any(0 < f["selected"] < 9 for f in local), "Load variation absent.")
        _require(any(f["missed"] for f in local) and any(f["arrow_colors"] > 1 for f in local), "Gate/palette variation absent.")
    a, b = generate_puzzle(107, 2, 1), generate_puzzle(107, 2, 1)
    _require(all(np.array_equal(pa[k], pb[k]) for s in a for pa, pb in zip(a[s], b[s]) for k in pa), "Nondeterministic seed.")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(731, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted.")
    broken = generate_puzzle(123, 2, 1)
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted.")
    # Equal-distance capacity candidates are rejected, never ordered by queue.
    grid = np.zeros((7, 7), dtype=np.int8)
    fp = np.zeros(grid.shape, dtype=bool)
    fp[2:5, 2:5] = True
    d = _distances(np.zeros(grid.shape, dtype=bool), fp)
    grid[d == 1] = 9
    try:
        _selection(grid, 9, fp, d)
    except AssertionError:
        pass
    else:
        raise AssertionError("Capacity frontier tie accepted.")
    return {"seeds": 42, "pairs": len(records), "distinct_inputs_normalized": len(normalized),
            "shapes": len({tuple(f["shape"]) for f in records}),
            "semantic_verification": "generator-owned checks only"}


def _semantic_fixtures():
    """Small hand-specified cases with explicit expected changed coordinates."""
    base = np.zeros((17, 30), dtype=np.int8)
    base[:, 10] = WALL
    base[:, 20] = WALL
    base[8, 10] = base[5, 10] = base[8, 20] = GATE
    base[1:4, 1:4] = 2
    base[2, 2] = 9
    keys = [_arrow((8, 5), (0, 1)), _arrow((8, 15), (0, 1)),
            _arrow((8, 25), (0, -1)), _arrow((12, 7), (0, 1))]
    for o in keys:
        for p in o[2]:
            base[p] = 3
    collected = ((6, 2), (2, 12), (8, 18))
    for p in collected + ((8, 22),):
        base[p] = 9
    expected = base.copy()
    expected[8, 10] = 0
    for p in collected:
        expected[p] = 0
    expected[1:4, 1:4] = 2
    expected[1, 1:4] = 9
    _require(np.array_equal(solve(base), expected), "Single/jam/miss/ignored-shot fixture failed.")
    # A nongate endpoint ignores the wall crossed at (12,10); no ray is painted.
    _require(expected[12, 10] == WALL and expected[12, 12] == 0 and expected[8, 22] == 9,
             "Interior ray or disconnected supply changed.")
    info = _analyze(base, "any")
    any_selected = _selection(base, 9, info["footprint"], info["distance"])
    _require(any_selected[8, 22] and int(any_selected.sum()) == 4,
             "Collision ablation must change collection, not merely gate paint.")
    # Cardinal and diagonal keys, under all rotations, retain vector semantics.
    diagonal = base.copy()
    for p in keys[0][2]:
        diagonal[p] = 0
    for p in _arrow((3, 5), (1, 1))[2]:
        diagonal[p] = 3
    diagonal_expected = expected.copy()
    for p in keys[0][2]:
        diagonal_expected[p] = 0
    for p in _arrow((3, 5), (1, 1))[2]:
        diagonal_expected[p] = 3
    for grid, label in ((base, expected), (diagonal, diagonal_expected)):
        for turns in range(4):
            rotated = np.rot90(grid, turns).copy()
            wanted = np.rot90(label, turns).copy()
            pos, slot, _, _, _ = _parse(rotated)
            r, c = pos
            wanted[r:r + 3, c:c + 3] = slot
            wanted[r, c:c + 3] = 9
            _require(np.array_equal(solve(rotated), wanted), "Rotation must retain displayed prefix order.")
    triple = base.copy()
    extra = _arrow((13, 15), (-1, 1))
    wanted = expected.copy()
    for p in extra[2]:
        triple[p] = wanted[p] = 3
    _require(np.array_equal(solve(triple), wanted), "Three shots must jam rather than use odd/even parity.")
    # Capacity is a cell ranking: nine genuinely closer by route, including a
    # geometrically distant cell, precede a cell just across the gated wall.
    capacity = base.copy()
    capacity[(capacity == 9) & ~_parse(capacity)[3]] = 0
    near = tuple((4, c) for c in range(1, 9)) + ((16, 1),)
    far = ((3, 11), (8, 22))
    for p in near + far:
        capacity[p] = 9
    wanted = capacity.copy()
    wanted[8, 10] = 0
    for p in near:
        wanted[p] = 0
    wanted[1:4, 1:4] = 9
    _require(np.array_equal(solve(capacity), wanted), "Path/capacity fixture failed.")
    _require(pair_features(capacity)["path_beats_manhattan"], "Fixture must defeat geometric proximity.")


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
