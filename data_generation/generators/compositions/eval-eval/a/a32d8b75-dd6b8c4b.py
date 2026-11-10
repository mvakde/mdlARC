"""Stenciled watersheds: tile-defined, competing, capacity-limited collectors.

Only standard-library modules and NumPy are used. The solver reads the visible
blueprint and field; it never reads seeds, example labels, or construction state.
"""
from collections import deque
import numpy as np


STEPS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CORNERS = ((0, 0), (0, 1), (1, 1), (1, 0))
ARROWS = ((0, 1), (1, 2), (2, 1), (1, 0))
ALTERNATIVES = ("paint_only", "no_exchange", "rotate_tile", "solid_stencil",
                "no_rotation", "shared_pool", "manhattan", "claim_ties",
                "unlimited", "farthest", "columns")


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(name + " is outside its legal range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(np.all((a >= 0) & (a <= 9)), "Invalid colors")
    return a.astype(int, copy=False)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.uint8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = [start]
        for r, c in queue:
            for dr, dc in STEPS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        result.append(queue)
    return result


def _decode(input_grid):
    a = _grid(input_grid)
    floor = int(a[0, 0])
    _require(np.all(a[0] == floor), "Top row must identify floor")
    rows = [r for r in range(1, a.shape[0])
            if int(a[r, 0]) != floor and np.all(a[r] == a[r, 0])]
    _require(bool(rows), "Missing full-width divider")
    sep = rows[0]
    frame = int(a[sep, 0])
    header, board = a[:sep], a[sep + 1:].copy()
    _require(board.size > 0, "Empty field")
    boxes = []
    for comp in _components(header == frame):
        pts = np.array(comp)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0) + 1
        box = header[r0:r1, c0:c1]
        _require(min(box.shape) >= 3, "Incomplete diagram frame")
        border = np.ones(box.shape, bool)
        border[1:-1, 1:-1] = False
        _require(np.all(box[border] == frame) and not np.any(box[~border] == frame),
                 "Diagram frame must bound its complete interior")
        _require(len(comp) == int(border.sum()), "Stray frame cells")
        boxes.append((int(c0), int(r0), box[1:-1, 1:-1].copy()))
    boxes.sort(key=lambda x: x[0])
    _require(len(boxes) == 4, "Need four separate framed diagrams")
    tile, stencil, direction, corner = [b[2] for b in boxes]
    _require(tile.shape[0] == tile.shape[1] and 2 <= len(tile) <= 4, "Invalid complete tile")
    vals = set(map(int, np.unique(tile)))
    _require(len(vals) == 2 and floor in vals, "Tile must use floor and wall")
    wall = next(iter(vals - {floor}))
    vals = set(map(int, np.unique(stencil)))
    _require(len(vals - {floor}) == 1 and len(vals) <= 2, "Stencil must use floor and payload")
    payload = next(iter(vals - {floor}))
    _require(len({floor, wall, payload, frame}) == 4, "Role colors must be distinct")
    _require(direction.shape == corner.shape == (3, 3), "Keys require explicit 3x3 support")
    _require(set(map(int, np.unique(direction))) == {floor, payload}
             and set(map(int, np.unique(corner))) == {floor, payload}, "Invalid key colors")
    dp = set(map(tuple, np.argwhere(direction == payload)))
    cp = set(map(tuple, np.argwhere(corner == payload)))
    turns = [k for k, p in enumerate(ARROWS) if dp == {(1, 1), p}]
    anchors = [k for k, (r, c) in enumerate(CORNERS) if cp == {(2 * r, 2 * c)}]
    _require(len(turns) == len(anchors) == 1, "Invalid geometric markers")
    _require(set(map(int, np.unique(board))) <= {floor, wall, payload}, "Unknown field role")
    # There are no unframed instructions or hidden annotations.
    reconstructed = np.full(header.shape, floor, int)
    for c, r, interior in boxes:
        h, w = interior.shape
        reconstructed[r:r + h + 2, c:c + w + 2] = frame
        reconstructed[r + 1:r + h + 1, c + 1:c + w + 1] = interior
    _require(np.array_equal(header, reconstructed), "Stray blueprint cells")
    d = dict(board=board, tile=tile, stencil=stencil == payload,
             turn=turns[0], anchor=anchors[0], floor=floor, wall=wall,
             payload=payload, frame=frame, separator=sep)
    _placement(d, check_blank=True)
    return d


def _placement(d, check_blank=False, mode="normal"):
    s = d["stencil"]
    k = 0 if mode == "no_rotation" else d["turn"]
    s = np.rot90(s, -k)
    if mode == "solid_stencil":
        s = np.ones(s.shape, bool)
    n, (h, w) = len(d["tile"]), d["board"].shape
    mh, mw = n * s.shape[0], n * s.shape[1]
    _require(mh <= h and mw <= w, "Expanded stencil does not fit")
    ar, ac = CORNERS[d["anchor"]]
    top, left = ar * (h - mh), ac * (w - mw)
    origins = [(top + n * int(r), left + n * int(c)) for r, c in np.argwhere(s)]
    if check_blank:
        for r, c in origins:
            _require(np.all(d["board"][r:r + n, c:c + n] == d["floor"]),
                     "A future collector block must initially be blank floor")
    return origins


def _bfs(wall, starts):
    h, w = wall.shape
    blocked = wall.tolist()
    distance = [[-1] * w for _ in range(h)]
    q = deque()
    for r, c in starts:
        distance[r][c] = 0
        q.append((r, c))
    while q:
        r, c = q.popleft()
        nd = distance[r][c] + 1
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr][nc] and distance[nr][nc] < 0:
                distance[nr][nc] = nd
                q.append((nr, nc))
    return np.array(distance, dtype=int)


def _ownership(distances):
    scores = np.where(distances < 0, 10000, distances)
    minimum = scores.min(axis=0)
    owner = scores.argmin(axis=0).astype(int)
    owner[(scores == minimum).sum(axis=0) > 1] = -2
    owner[minimum == 10000] = -1
    return owner, np.where(minimum == 10000, -1, minimum)


def _world(d, mode="normal"):
    board = d["board"].copy()
    n, tile = len(d["tile"]), d["tile"]
    complemented = np.where(tile == d["floor"], d["wall"], d["floor"])
    if mode == "no_exchange":
        complemented = tile.copy()
    elif mode == "rotate_tile":
        complemented = np.rot90(complemented, -d["turn"])
    collectors, footprint = [], np.zeros(board.shape, bool)
    for r, c in _placement(d, mode=mode):
        board[r:r + n, c:c + n] = complemented
        footprint[r:r + n, c:c + n] = True
        slots = [(r + int(y), c + int(x)) for y, x in np.argwhere(complemented == d["floor"])]
        collectors.append(dict(origin=(r, c), slots=slots))
    distances = np.stack([_bfs(board == d["wall"], co["slots"]) for co in collectors])
    owner, minimum = _ownership(distances)
    return dict(board=board, collectors=collectors, footprint=footprint,
                distances=distances, owner=owner, minimum=minimum)


def _straight_distances(world):
    rr, cc = np.indices(world["board"].shape)
    ds = np.stack([np.minimum.reduce([np.abs(rr - r) + np.abs(cc - c) for r, c in co["slots"]])
                   for co in world["collectors"]])
    # This stronger alternative already knows true reachability; it substitutes
    # direct distance only for path length and inter-collector comparisons.
    return np.where(world["distances"] < 0, -1, ds)


def _select(cells, distance, capacity, strict=True, farthest=False):
    ordered = sorted(cells, key=lambda p: ((-1 if farthest else 1) * int(distance[p]), p[0], p[1]))
    if strict and len(ordered) > capacity:
        _require(distance[ordered[capacity - 1]] < distance[ordered[capacity]],
                 "Ambiguous distance tie at capacity frontier")
    return ordered[:capacity]


def _render(d, world, mode="normal", strict=True):
    out = world["board"].copy()
    loose = (out == d["payload"]) & ~world["footprint"]
    if mode == "paint_only":
        return out, [[] for _ in world["collectors"]]
    ds = _straight_distances(world) if mode == "manhattan" else world["distances"]
    owner, minimum = _ownership(ds)
    if mode == "claim_ties":
        owner[owner == -2] = np.where(ds < 0, 10000, ds).argmin(axis=0)[owner == -2]
    selected = []
    if mode == "shared_pool":
        cells = list(map(tuple, np.argwhere(loose & (minimum >= 0))))
        total = sum(len(co["slots"]) for co in world["collectors"])
        pool = _select(cells, minimum, total, strict=False)
        offset = 0
        for co in world["collectors"]:
            cap = len(co["slots"])
            selected.append(pool[offset:offset + cap])
            offset += cap
    else:
        for i, co in enumerate(world["collectors"]):
            cells = list(map(tuple, np.argwhere(loose & (owner == i))))
            if mode == "unlimited":
                selected.append(cells)
            else:
                selected.append(_select(cells, ds[i], len(co["slots"]), strict=strict,
                                        farthest=mode == "farthest"))
    for co, cells in zip(world["collectors"], selected):
        for p in cells:
            out[p] = d["floor"]
        slots = sorted(co["slots"], key=lambda p: (p[1], p[0])) if mode == "columns" else co["slots"]
        for p in slots[:len(cells)]:
            out[p] = d["payload"]
    return out, selected


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the output solely from this input's framed blueprint and field."""
    d = _decode(input_grid)
    return _render(d, _world(d))[0]


def _alternative(d, world, mode):
    try:
        if mode in {"no_exchange", "rotate_tile", "solid_stencil", "no_rotation"}:
            return _render(d, _world(d, mode), strict=False)[0]
        return _render(d, world, mode, strict=False)[0]
    except AssertionError:
        # An alternative that produces an out-of-bounds mosaic is also ruled out.
        return None


def _rejected(d, world, output):
    rejected = set()
    for mode in ALTERNATIVES:
        alt = _alternative(d, world, mode)
        if alt is None or not np.array_equal(alt, output):
            rejected.add(mode)
    return rejected


def _features(d, world, output=None):
    output, selected = _render(d, world)
    loose = d["board"] == d["payload"]
    owned = [int((loose & (world["owner"] == i)).sum()) for i in range(len(world["collectors"]))]
    capacities = [len(co["slots"]) for co in world["collectors"]]
    chosen = [len(x) for x in selected]
    gaps = []
    for i, cells in enumerate(selected):
        distances = sorted(map(int, world["distances"][i][loose & (world["owner"] == i)]))
        gaps.append(distances[capacities[i]] - distances[capacities[i] - 1]
                    if len(distances) > capacities[i] else 0)
    reachable = world["minimum"] >= 0
    return dict(shape=d["board"].shape, tile_size=len(d["tile"]),
                stencil_shape=d["stencil"].shape, turn=d["turn"], anchor=d["anchor"],
                collectors=len(capacities), capacity=capacities[0], owned=owned,
                collected=chosen, gaps=gaps, excess=sum(max(0, a - b) for a, b in zip(owned, capacities)),
                partial=any(0 < a < b for a, b in zip(chosen, capacities)),
                ties=int((loose & (world["owner"] == -2)).sum()),
                unreachable=int((loose & (world["owner"] == -1)).sum()),
                max_distance=int(world["minimum"][reachable].max()) if reachable.any() else 0,
                wall_count=int((d["board"] == d["wall"]).sum()),
                palette=(d["floor"], d["wall"], d["payload"], d["frame"]))


def _sample_tile(rng, n, witness):
    for _ in range(100):
        bits = rng.random((n, n)) < rng.uniform(.25, .75)
        count = int(bits.sum())
        if not 1 <= count < n * n:
            continue
        if witness:
            slots = list(map(tuple, np.argwhere(bits)))
            cols = sorted(slots, key=lambda p: (p[1], p[0]))
            if count < 2 or slots == cols or np.array_equal(bits, np.rot90(bits, 1)):
                continue
        return bits
    # A complete, asymmetric pattern with different row/column slot order.
    bits = np.zeros((n, n), bool)
    bits[0, 1] = bits[1, 0] = bits[-1, -1] = True
    return bits


def _sample_stencil(rng, sh, sw, witness):
    for _ in range(100):
        s = rng.random((sh, sw)) < rng.uniform(.2, .65)
        if not s.any() or not s[0].any() or not s[-1].any() or not s[:, 0].any() or not s[:, -1].any():
            continue
        if witness and (not 3 <= int(s.sum()) <= 7 or s.all() or np.array_equal(s, np.rot90(s, 2))):
            continue
        if not witness and int(s.sum()) > 10:
            continue
        return s
    s = np.zeros((sh, sw), bool)
    s[0, -1] = s[-1, 0] = s[-1, -1] = True
    return s


def _walls(rng, shape, reserved, intensity, preference):
    h, w = shape
    wall = np.zeros(shape, bool)
    if intensity == 0 and rng.random() < .45:
        return wall
    # Full cuts, partial cuts, turns, and pockets combine at variable offsets.
    for _ in range(int(rng.integers(2, 5 + intensity * 3))):
        horizontal = bool(rng.integers(2))
        if horizontal:
            r = int(rng.integers(1, h - 1))
            c0 = int(rng.integers(0, max(1, w - 3)))
            c1 = int(rng.integers(c0 + 2, w + 1))
            if rng.random() < .25:
                c0, c1 = 0, w
            wall[r, c0:c1] = True
            if rng.random() < .3:
                end = min(h, r + int(rng.integers(2, max(3, h // 2))))
                wall[r:end, c1 - 1] = True
        else:
            c = int(rng.integers(1, w - 1))
            r0 = int(rng.integers(0, max(1, h - 3)))
            r1 = int(rng.integers(r0 + 2, h + 1))
            if rng.random() < .25:
                r0, r1 = 0, h
            wall[r0:r1, c] = True
            if rng.random() < .3:
                wall[r1 - 1, c:min(w, c + int(rng.integers(2, 8)))] = True
    pocket_count = int(rng.integers(1, 3 + intensity))
    for _ in range(pocket_count * 6):
        if pocket_count == 0:
            break
        ph, pw = int(rng.integers(3, min(7, h) + 1)), int(rng.integers(3, min(8, w) + 1))
        r, c = int(rng.integers(h - ph + 1)), int(rng.integers(w - pw + 1))
        if reserved[r:r + ph, c:c + pw].any():
            continue
        wall[r, c:c + pw] = wall[r + ph - 1, c:c + pw] = True
        wall[r:r + ph, c] = wall[r:r + ph, c + pw - 1] = True
        # Preference changes topology frequency, not role interpretation.
        if preference == "openings" and rng.random() < .6:
            wall[r, c + int(rng.integers(1, pw - 1))] = False
        pocket_count -= 1
    wall[reserved] = False
    return wall


def _pick(rng, cells, count, style):
    if count <= 0:
        return []
    cells = list(cells)
    if style == 0:
        return [cells[int(i)] for i in rng.choice(len(cells), count, replace=False)]
    ar, ac = cells[int(rng.integers(len(cells)))]
    if style == 1:
        cells.sort(key=lambda p: (abs(p[0] - ar) + abs(p[1] - ac), p))
    else:
        horizontal = bool(rng.integers(2))
        cells.sort(key=lambda p: ((abs(p[0] - ar) * 100 + abs(p[1] - ac)) if horizontal
                                 else (abs(p[1] - ac) * 100 + abs(p[0] - ar)), p))
    return cells[:count]


def _supply_for(rng, cells, distance, capacity, excess=False):
    style = int(rng.integers(3))
    if not excess:
        count = int(rng.integers(0, min(capacity, len(cells)) + 1))
        return _pick(rng, cells, count, style)
    frontiers = sorted(set(int(distance[p]) for p in cells))
    usable = [t for t in frontiers if sum(distance[p] <= t for p in cells) >= capacity
              and sum(distance[p] > t for p in cells) >= 1]
    if not usable:
        return None
    threshold = int(rng.choice(usable))
    near = [p for p in cells if distance[p] <= threshold]
    far = [p for p in cells if distance[p] > threshold]
    selected = _pick(rng, near, capacity, style)
    retained = _pick(rng, far, int(rng.integers(1, min(6, len(far)) + 1)), style)
    return selected + retained


def _populate(rng, d, world, witness):
    allowed = (world["board"] == d["floor"]) & ~world["footprint"]
    basins = [list(map(tuple, np.argwhere(allowed & (world["owner"] == i))))
              for i in range(len(world["collectors"]))]
    capacity = len(world["collectors"][0]["slots"])
    tied = list(map(tuple, np.argwhere(allowed & (world["owner"] == -2))))
    inactive = list(map(tuple, np.argwhere(allowed & (world["owner"] == -1))))
    chosen, partial, excess = [], None, None
    if witness:
        if not tied or not inactive:
            return None
        # The tied token is claimed by this collector under the first-wins
        # alternative; keeping that collector partial exposes the difference.
        partial_options = sorted({int(np.where(world["distances"][:, p[0], p[1]] < 0, 10000,
                                               world["distances"][:, p[0], p[1]]).argmin()) for p in tied})
        partial_options = [i for i in partial_options if len(basins[i]) >= capacity - 1]
        choices = [(i, j) for i in partial_options for j in range(len(basins))
                   if i != j and len(basins[j]) >= capacity + 2]
        if not choices:
            return None
        partial, excess = choices[int(rng.integers(len(choices)))]
        slots = world["collectors"][partial]["slots"]
        colslots = sorted(slots, key=lambda p: (p[1], p[0]))
        cuts = [k for k in range(1, capacity) if set(slots[:k]) != set(colslots[:k])]
        if not cuts:
            return None
        cut = int(rng.choice(cuts))
        chosen += _pick(rng, basins[partial], cut, int(rng.integers(3)))
        load = _supply_for(rng, basins[excess], world["distances"][excess], capacity, True)
        if load is None:
            return None
        chosen += load
        tied = [p for p in tied if int(np.where(world["distances"][:, p[0], p[1]] < 0, 10000,
                                              world["distances"][:, p[0], p[1]]).argmin()) == partial]
        chosen += _pick(rng, tied, 1, 0)
        chosen += _pick(rng, inactive, int(rng.integers(1, min(5, len(inactive)) + 1)), int(rng.integers(3)))
    for i, basin in enumerate(basins):
        if i in (partial, excess):
            continue
        load = _supply_for(rng, basin, world["distances"][i], capacity,
                           excess=bool(len(basin) > capacity + 1 and rng.random() < .35))
        if load is None:
            return None
        chosen += load
    if not witness:
        chosen += _pick(rng, tied, int(rng.integers(0, min(4, len(tied)) + 1)), 0)
        chosen += _pick(rng, inactive, int(rng.integers(0, min(7, len(inactive)) + 1)), int(rng.integers(3)))
    board = d["board"].copy()
    for p in chosen:
        board[p] = d["payload"]
    return board


def _encode(rng, d):
    floor, frame, payload = d["floor"], d["frame"], d["payload"]
    direction = np.full((3, 3), floor, int)
    direction[1, 1] = direction[ARROWS[d["turn"]]] = payload
    corner = np.full((3, 3), floor, int)
    ar, ac = CORNERS[d["anchor"]]
    corner[2 * ar, 2 * ac] = payload
    cards = [d["tile"], np.where(d["stencil"], payload, floor), direction, corner]
    heights = [a.shape[0] + 2 for a in cards]
    sep = max(heights) + 2
    header = np.full((sep + 1, d["board"].shape[1]), floor, int)
    spare = header.shape[1] - sum(a.shape[1] + 2 for a in cards) - 3 - 2
    _require(spare >= 0, "Blueprint too wide")
    left = 1 + int(rng.integers(spare + 1))
    for card, height in zip(cards, heights):
        top = int(rng.integers(1, sep - height))
        width = card.shape[1] + 2
        header[top:top + height, left:left + width] = frame
        header[top + 1:top + height - 1, left + 1:left + width - 1] = card
        left += width + 1
    header[-1] = frame
    return np.concatenate((header, d["board"]), axis=0)


def _make_pair(rng, palette, preference, control, n, obligation=None, test=False):
    witness = obligation is not None
    floor, wall, payload, frame = map(int, palette)
    for attempt in range(400):
        sh = int(rng.integers(2, 5 if not test else 7))
        sw = int(rng.integers(2, min(6, 30 - n - 19) + 1))
        turn, anchor = control
        sep = max(n + 2, sh + 2, 5) + 2
        rh, rw = (sw, sh) if turn % 2 else (sh, sw)
        min_h, max_h = max(8, n * rh + 1), 30 - sep - 1
        min_w = max(n + sw + 19, n * rw + 1)
        if min_h > max_h or min_w > 30:
            continue
        h, w = int(rng.integers(min_h, max_h + 1)), int(rng.integers(min_w, 31))
        bits = _sample_tile(rng, n, witness)
        stencil = _sample_stencil(rng, sh, sw, witness)
        # Simpler examples can have one receiver while using the same stencil
        # convention. Witnesses retain several genuinely competing receivers.
        if not witness and rng.random() < .18:
            stencil = np.ones((1, 1), bool)
        d = dict(tile=np.where(bits, wall, floor), stencil=stencil,
                 turn=turn, anchor=anchor, floor=floor, wall=wall, payload=payload,
                 frame=frame, board=np.full((h, w), floor, int))
        reserved = np.zeros((h, w), bool)
        for r, c in _placement(d):
            reserved[r:r + n, c:c + n] = True
        intensity = 2 if witness or test else int(rng.integers(3))
        d["board"][_walls(rng, (h, w), reserved, intensity, preference)] = wall
        world = _world(d)
        board = _populate(rng, d, world, witness)
        if board is None:
            continue
        d["board"] = board
        world = _world(d)
        try:
            output, _ = _render(d, world)
        except AssertionError:
            continue
        if witness:
            required = {"paint_only", "no_exchange", "solid_stencil", "shared_pool",
                        "claim_ties", "unlimited", "farthest", "columns"}
            if turn % 2:
                required |= {"rotate_tile", "no_rotation"}
            if obligation == "paths":
                required.add("manhattan")
            if not required <= _rejected(d, world, output):
                continue
        inp = _encode(rng, d)
        recovered = _decode(inp)
        _require(np.array_equal(solve(inp), output), "Blueprint encoding changed the rule")
        return {"input": inp, "output": output}
    raise RuntimeError("Unable to construct the requested evidence within 400 attempts")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = rng.choice(10, 4, replace=False)
    vary_palette = bool(rng.random() < .7)
    preference = str(rng.choice(["closed", "openings", "segments"]))
    quarter, even = int(rng.choice([1, 3])), int(rng.choice([0, 2]))
    anchor = int(rng.integers(4))
    other = (anchor + int(rng.choice([1, 3]))) % 4
    controls = [(quarter, anchor), (even, other)]
    held_choices = [(k, a) for k in range(4) for a in range(4) if (k, a) not in controls]
    held = held_choices[int(rng.integers(len(held_choices)))]
    sizes = list(map(int, rng.permutation([2, 3])))
    specs = [(controls[0], sizes[0], "watershed"), (controls[1], sizes[1], "paths")]
    for _ in range(nt - 2):
        control = held_choices[int(rng.integers(len(held_choices)))]
        if control == held:
            control = controls[int(rng.integers(2))]
        specs.append((control, int(rng.integers(2, 5)), None))
    specs = [specs[int(i)] for i in rng.permutation(nt)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, requests in (("train", specs),
                            ("test", [(held, int(rng.integers(2, 5)), None) for _ in range(nq)])):
        for control, n, obligation in requests:
            for _ in range(12):
                local_palette = rng.choice(10, 4, replace=False) if vary_palette else palette
                pair = _make_pair(rng, local_palette, preference, control, n, obligation, split == "test")
                sig = _key(pair["input"])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Repeated duplicate input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "Invalid example counts")
    seen, rejected, stats, controls = set(), set(), {"train": [], "test": []}, {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            sig = _key(a)
            _require(sig not in seen, "Duplicate input")
            seen.add(sig)
            d = _decode(a)
            world = _world(d)
            predicted, selected = _render(d, world)
            _require(b.shape == d["board"].shape and np.array_equal(predicted, b), "Incorrect solved field")
            _require(np.array_equal(solve(a.copy()), b), "Input-only solver disagreement")
            _require(int((d["board"] == d["payload"]).sum()) == int((b == d["payload"]).sum()),
                     "Payload conservation failed")
            # Independently audit local storage addresses and preservation.
            removed = np.zeros(b.shape, bool)
            for co, chosen in zip(world["collectors"], selected):
                for p in chosen:
                    removed[p] = True
                occupied = [p for p in co["slots"] if b[p] == d["payload"]]
                _require(occupied == co["slots"][:len(chosen)], "Storage is not a masked row-major prefix")
            unchanged = ~world["footprint"] & ~removed
            _require(np.array_equal(b[unchanged], d["board"][unchanged]), "Unselected field cells changed")
            _require(np.all(b[world["board"] == d["wall"]] == d["wall"]), "A wall or baffle changed")
            f = _features(d, world)
            stats[split].append(f)
            controls[split].add((d["turn"], d["anchor"]))
            if split == "train":
                rejected |= _rejected(d, world, b)
    _require(set(ALTERNATIVES) <= rejected, "Training fails to distinguish: " + str(set(ALTERNATIVES) - rejected))
    tr = stats["train"]
    _require(any(f["partial"] and f["excess"] for f in tr), "Need partial and excess storage evidence")
    _require(any(f["ties"] and f["unreachable"] for f in tr), "Need watershed and disconnected supply evidence")
    _require(any(f["turn"] % 2 for f in tr), "Need informative quarter-turn training")
    _require(len({f["tile_size"] for f in tr}) >= 2, "Need more than one tile scale in training")
    _require(not (controls["train"] & controls["test"]), "Test control combination must be held out")


def _puzzle_key(p):
    return tuple((_key(pair["input"]), _key(pair["output"])) for s in ("train", "test") for pair in p[s])


def self_test():
    puzzles, features, structures = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ("train", "test"):
            for pair in p[split]:
                d = _decode(pair["input"])
                features.append(_features(d, _world(d)))
                mapping = {}
                structures.add((pair["input"].shape, tuple(mapping.setdefault(int(c), len(mapping))
                                                         for c in pair["input"].ravel())))
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "Seed diversity collapsed")
    _require(len(structures) >= 180, "Structural diversity collapsed")
    _require({f["tile_size"] for f in features} == {2, 3, 4}, "Tile scale coverage missing")
    _require({f["turn"] for f in features} == set(range(4)), "Rotation coverage missing")
    _require({f["anchor"] for f in features} == set(range(4)), "Anchor coverage missing")
    for field, minimum in (("shape", 25), ("stencil_shape", 12), ("capacity", 8),
                           ("collectors", 6), ("max_distance", 12), ("palette", 25)):
        _require(len({f[field] for f in features}) >= minimum, "Missing variation in " + field)
    _require(any(f["collectors"] == 1 for f in features), "Single-collector members missing")
    _require(any(f["wall_count"] == 0 for f in features), "Wall-free fields missing")
    _require(any(f["excess"] and f["ties"] and f["unreachable"] for f in features), "Integrated hard cases missing")
    _require(_puzzle_key(generate_puzzle(1234, 7, 3)) == _puzzle_key(generate_puzzle(1234, 7, 3)), "Replay failed")
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(2051 + nt, nt, nq)
        _require(len(p["train"]) == nt and len(p["test"]) == nq, "Requested counts ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("Illegal count accepted")
    # A label mutation must be rejected even if schema and dimensions are valid.
    damaged = {s: [{r: a.copy() for r, a in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"][0, 0] = (damaged["test"][0]["output"][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted label accepted")
    # A tiny graph exercises neutral ownership ties independently of sampling.
    wall = np.zeros((3, 7), bool)
    ds = np.stack([_bfs(wall, [(1, 0)]), _bfs(wall, [(1, 6)])])
    owner, _ = _ownership(ds)
    _require(np.all(owner[:, 3] == -2) and owner[1, 2] == 0 and owner[1, 4] == 1,
             "Watershed tie law failed")
    dist = np.array([[1, 2, 2, 3]])
    try:
        _select([(0, i) for i in range(4)], dist, 2)
    except AssertionError:
        pass
    else:
        raise AssertionError("Ambiguous capacity tie accepted")
    return {"seeds": 42, "pairs": len(features), "structures": len(structures),
            "shapes": len({f["shape"] for f in features}),
            "capacities": sorted({f["capacity"] for f in features}),
            "collector_counts": sorted({f["collectors"] for f in features}),
            "max_path": max(f["max_distance"] for f in features)}


if __name__ == "__main__":
    print(self_test())
