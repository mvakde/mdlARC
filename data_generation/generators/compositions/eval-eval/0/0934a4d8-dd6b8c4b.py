"""Veiled reservoirs: reflected terrain determines a collector's shape and capacity.

No parent code, file access, cached labels, or random state is used by solve.
Generation labels start from complete terrain; solving starts from visible pixels.
"""
from collections import deque
import hashlib
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 311413824, 263949175, 583866782, 799544826, 126613044,
              340779211, 614829191, 634414356, 330372033, 769061922,
              775926454, 374173303, 640387025, 921918524, 984995168,
              96640912, 584785506, 960105575, 621400201, 525282562,
              855791842, 899383575, 983521582, 650226130, 103686650,
              287461275, 832077916, 155723881, 276385074, 397303318,
              496612165, 198474446, 117113605, 302023094, 959891491,
              694767231, 665567667, 985914185, 68881488, 468528441, 75974072)


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
          "Grids must contain integer colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _window(shape, box):
    m = np.zeros(shape, dtype=bool)
    r, s, c, d = box
    m[r:s, c:d] = True
    return m


def _roles(a):
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    _need(len(np.unique(border)) == 1, "The outer boundary must identify the wall")
    wall = int(border[0])
    colors, counts = np.unique(a, return_counts=True)
    _need(len(colors) == 5, "Exactly five visible roles are required")
    singletons = [int(c) for c, n in zip(colors, counts) if n == 1 and c != wall]
    _need(len(singletons) == 1, "One singleton beacon must identify the active region")
    beacon = singletons[0]
    rectangles = []
    for color, count in zip(colors, counts):
        if int(color) in (wall, beacon) or count < 4:
            continue
        rr, cc = np.nonzero(a == color)
        r, s, c, d = int(rr.min()), int(rr.max())+1, int(cc.min()), int(cc.max())+1
        if (s-r)*(d-c) == count and min(s-r, d-c) >= 2:
            rectangles.append((int(color), (r, s, c, d)))
    _need(len(rectangles) == 1, "The veil must be the unique rectangular color")
    veil, box = rectangles[0]
    remaining = [int(c) for c in colors if int(c) not in (wall, beacon, veil)]
    remaining.sort(key=lambda c: int(np.count_nonzero(a == c)), reverse=True)
    floor, payload = remaining
    _need(np.count_nonzero(a == floor) > np.count_nonzero(a == payload) >= 2,
          "Floor must be more extensive than the payload")
    r, s, c, d = box
    _need(0 < r < s < a.shape[0] and 0 < c < d < a.shape[1], "Veil must be interior")
    return floor, wall, payload, veil, beacon, box


def _restore(a, roles):
    floor, wall, payload, veil, beacon, box = roles
    full = a.copy()
    full[(a == payload) | (a == beacon)] = floor
    full[a == veil] = -1
    h, w = a.shape
    # Inference is per symmetry orbit; no intact rectangular source is assumed.
    for r in range((h+1)//2):
        for c in range((w+1)//2):
            orbit = {(r, c), (h-1-r, c), (r, w-1-c), (h-1-r, w-1-c)}
            values = {int(full[x]) for x in orbit if full[x] >= 0}
            _need(len(values) == 1, "An erased orbit or inconsistent terrain has no unique repair")
            value = values.pop()
            _need(value in (floor, wall), "Terrain must consist of floor and wall")
            for x in orbit:
                full[x] = value
    return full


def _distances(floor_mask, starts):
    distance = np.full(floor_mask.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in np.argwhere(starts):
        distance[r, c] = 0
        queue.append((int(r), int(c)))
    h, w = floor_mask.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and floor_mask[nr, nc] and distance[nr, nc] < 0:
                distance[nr, nc] = distance[r, c]+1
                queue.append((nr, nc))
    return distance


def _wave_distances(floor_mask, starts):
    """Separate array wavefront audit of the queue implementation."""
    seen, frontier = starts.copy(), starts.copy()
    distance = np.full(floor_mask.shape, -1, dtype=np.int16)
    distance[seen] = 0
    for depth in range(1, floor_mask.size+1):
        spread = np.zeros(floor_mask.shape, dtype=bool)
        spread[1:] |= frontier[:-1]
        spread[:-1] |= frontier[1:]
        spread[:, 1:] |= frontier[:, :-1]
        spread[:, :-1] |= frontier[:, 1:]
        frontier = spread & floor_mask & ~seen
        if not frontier.any():
            break
        distance[frontier] = depth
        seen |= frontier
    return distance


def _storage(full, roles, a):
    floor, _, _, _, beacon, box = roles
    active = _distances(full == floor, a == beacon) >= 0
    slots = _window(a.shape, box) & (full == floor) & active
    _need(slots.any(), "The beacon must reach some reservoir floor")
    return slots, active


def _select(a, payload, distance, capacity):
    supply = (a == payload) & (distance >= 0)
    depths = np.sort(distance[supply])
    if len(depths) > capacity:
        _need(depths[capacity-1] < depths[capacity], "A distance tie straddles capacity")
    n = min(capacity, len(depths))
    selected = supply & (distance <= depths[n-1]) if n else np.zeros(a.shape, dtype=bool)
    return selected


def _transfer(a, full, roles, audit=False):
    floor, _, payload, veil, _, _ = roles
    slots, active = _storage(full, roles, a)
    distance = (_wave_distances if audit else _distances)(full == floor, slots)
    selected = _select(a, payload, distance, int(slots.sum()))
    out = a.copy()
    out[a == veil] = full[a == veil]
    out[selected] = floor
    destinations = np.flatnonzero(slots)[:int(selected.sum())]
    out.flat[destinations] = payload
    return out, slots, active, distance, selected


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    roles = _roles(a)
    full = _restore(a, roles)
    return _transfer(a, full, roles)[0]


def _terrain(rng, h, w, style, closed_axis):
    """Compose a quarter-network, then reflect terrain (never its overlays)."""
    bh, bw = (h+1)//2, (w+1)//2
    if style == "maze":
        base = np.zeros((bh, bw), dtype=bool)
        nodes = [(r, c) for r in range(1, bh, 2) for c in range(1, bw, 2)]
        start = nodes[int(rng.integers(len(nodes)))]
        seen, stack = {start}, [start]
        base[start] = True
        while stack:
            r, c = stack[-1]
            choices = [(r+2*dr, c+2*dc) for dr, dc in STEPS
                       if (r+2*dr, c+2*dc) in nodes and (r+2*dr, c+2*dc) not in seen]
            if not choices:
                stack.pop()
                continue
            nr, nc = choices[int(rng.integers(len(choices)))]
            base[(r+nr)//2, (c+nc)//2] = base[nr, nc] = True
            seen.add((nr, nc))
            stack.append((nr, nc))
        # Variable chambers and extra links coexist with the tree corridors.
        for _ in range(int(rng.integers(0, 4))):
            r, c = int(rng.integers(1, bh-2)), int(rng.integers(1, bw-2))
            base[r:r+int(rng.integers(2, 5)), c:c+int(rng.integers(2, 5))] = True
    else:
        base = np.ones((bh, bw), dtype=bool)
        # Partial dividers, turns, rings, and independently placed apertures.
        for _ in range(int(rng.integers(4, 10))):
            r, c = int(rng.integers(1, bh-2)), int(rng.integers(1, bw-2))
            s, d = int(rng.integers(r+1, bh)), int(rng.integers(c+1, bw))
            kind = int(rng.integers(4))
            if kind < 2:
                base[r:s+1, c] = False
                if kind == 1:
                    base[s, c:d+1] = False
            elif kind == 2:
                base[r, c:d+1] = False
            else:
                base[r:s+1, c] = base[r:s+1, d] = False
                base[r, c:d+1] = base[s, c:d+1] = False
            if rng.random() < .65:
                base[int(rng.integers(r, s+1)), c] = True
    base[0] = False
    base[:, 0] = False
    if closed_axis == 0:
        base[-1] = False
    elif closed_axis == 1:
        base[:, -1] = False
    # Some independent sealed pockets break whole-field connectivity as well.
    if rng.random() < .55 and min(bh, bw) >= 8:
        r, c = int(rng.integers(2, bh-3)), int(rng.integers(2, bw-3))
        base[r-1:r+2, c-1:c+2] = False
        base[r, c] = True
    rr = np.minimum(np.arange(h), h-1-np.arange(h))
    cc = np.minimum(np.arange(w), w-1-np.arange(w))
    return base[rr[:, None], cc[None, :]]


def _box(rng, shape, cross, target):
    h, w = shape
    rh = int(rng.integers(3 if target else 2, min(10 if target else 8, h-3)+1))
    rw = int(rng.integers(3 if target else 2, min(10 if target else 8, w-3)+1))
    if cross in (0, 1):
        # Cross exactly one seam; the perpendicular reflected copy survives.
        if cross == 0:
            rh = min(rh, h-3)
            rw = min(rw, (w-3)//2)
            r = max(1, min(h-rh-1, h//2-int(rng.integers(1, rh))))
            c = int(rng.integers(1, max(2, w//2-rw)))
        else:
            rh = min(rh, (h-3)//2)
            rw = min(rw, w-3)
            c = max(1, min(w-rw-1, w//2-int(rng.integers(1, rw))))
            r = int(rng.integers(1, max(2, h//2-rh)))
    else:
        rh, rw = min(rh, (h-3)//2), min(rw, (w-3)//2)
        r, c = int(rng.integers(1, max(2, h//2-rh))), int(rng.integers(1, max(2, w//2-rw)))
    if rng.random() < .5:
        r = h-r-rh
    if rng.random() < .5:
        c = w-c-rw
    return r, r+rh, c, c+rw


def _manhattan(slots):
    rr, cc = np.indices(slots.shape)
    distances = [np.abs(rr-int(r))+np.abs(cc-int(c)) for r, c in np.argwhere(slots)]
    return np.minimum.reduce(distances)


def _place_supply(rng, available, distance, slots, load, excess, reversal):
    k = int(slots.sum())
    cells = np.flatnonzero(available & (distance >= 0))
    if not excess:
        if len(cells) < load:
            return None
        return rng.choice(cells, load, replace=False).tolist()
    extra = load-k
    metric = _manhattan(slots)
    for a in rng.permutation(cells):
        near = cells[distance.flat[cells] <= distance.flat[a]]
        far = cells[distance.flat[cells] > distance.flat[a]]
        if len(near) < k or len(far) < extra:
            continue
        candidates = far[metric.flat[far] < metric.flat[a]] if reversal else far
        if not len(candidates):
            continue
        b = int(rng.choice(candidates))
        chosen = [int(a)] + rng.choice(near[near != a], k-1, replace=False).tolist()
        retained = [b] + rng.choice(far[far != b], extra-1, replace=False).tolist()
        return chosen+retained
    return None


def _make_pair(rng, colors, obligation, target, held_axis, seen):
    floor, wall, payload, veil, beacon = colors
    for attempt in range(700):
        h, w = int(rng.integers(15, 31)), int(rng.integers(15, 31))
        closed = (1-held_axis) if obligation == "partial" else (int(rng.integers(2)) if rng.random() < .85 else -1)
        style = "maze" if rng.random() < .65 or obligation == "excess" else "rooms"
        terrain = _terrain(rng, h, w, style, closed)
        cross = held_axis if target else (1-held_axis if obligation == "partial" else int(rng.choice([-1, 1-held_axis])))
        box = _box(rng, terrain.shape, cross, target)
        region = _window(terrain.shape, box)
        # Actual witnesses, not a geometry assumption, decide recoverability.
        if np.any(region & region[::-1] & region[:, ::-1] & region[::-1, ::-1]):
            continue
        if not np.any(region & ~terrain) or not np.any(region & terrain):
            continue
        provisional = np.where(terrain, floor, wall).astype(np.int8)
        provisional[region] = veil
        # Pick an active component from a visible neighbor of the window.
        adjacent = np.zeros(terrain.shape, dtype=bool)
        hidden_floor = region & terrain
        adjacent[1:] |= hidden_floor[:-1]
        adjacent[:-1] |= hidden_floor[1:]
        adjacent[:, 1:] |= hidden_floor[:, :-1]
        adjacent[:, :-1] |= hidden_floor[:, 1:]
        possible = np.flatnonzero(adjacent & terrain & ~region)
        if not len(possible):
            continue
        chosen_beacon = int(rng.choice(possible))
        provisional.flat[chosen_beacon] = beacon
        roles = (*colors, box)
        slots, active = _storage(np.where(terrain, floor, wall), roles, provisional)
        k = int(slots.sum())
        inactive_slots = int((region & terrain & ~active).sum())
        if not 3 <= k <= (28 if target else 20):
            continue
        if obligation == "partial" and inactive_slots == 0:
            continue
        available = terrain & ~region & (provisional != beacon)
        inactive = available & ~active
        if inactive.sum() < 2:
            continue
        distance = _distances(terrain, slots)
        if obligation == "partial":
            row_order = np.flatnonzero(slots)
            col_order = np.array(sorted(row_order, key=lambda i: (int(i)%w, int(i)//w)))
            all_order = np.flatnonzero(region & terrain)
            choices = [n for n in range(2, k) if set(row_order[:n]) != set(col_order[:n])
                       and set(row_order[:n]) != set(all_order[:n])]
            if not choices:
                continue
            load = int(rng.choice(choices))
            excess = False
        elif obligation == "excess":
            load, excess = k+int(rng.integers(2, 8)), True
        else:
            excess = rng.random() < .45
            load = k+int(rng.integers(1, 9)) if excess else int(rng.integers(0, k+1))
        supply = _place_supply(rng, available, distance, slots, load, excess, obligation == "excess")
        if supply is None:
            continue
        # Unreachable payload is variable and can be geometrically near the veil.
        n_distraction = int(rng.integers(2, min(14, int(inactive.sum()))+1))
        distractors = rng.choice(np.flatnonzero(inactive), n_distraction, replace=False).tolist()
        inp = provisional.copy()
        inp.flat[supply+distractors] = payload
        if _key(inp) in seen:
            continue
        full = np.where(terrain, floor, wall).astype(np.int8)
        try:
            observed = _roles(inp)
            if observed != roles:
                continue
            if not np.array_equal(_restore(inp, observed), full):
                continue
            output, _, _, _, selected = _transfer(inp, full, observed)
            # Opening the veil is a strong A-omission: its collection must differ
            # in the excess demonstration, beyond merely painting missing walls.
            if obligation == "excess":
                opened = full.copy()
                opened[region] = floor
                open_selected = _transfer(inp, opened, observed)[4]
                if np.array_equal(open_selected, selected):
                    continue
        except ValueError:
            continue
        return {"input": inp, "output": output}
    raise RuntimeError("Could not satisfy reservoir evidence after 700 bounded attempts")


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = tuple(int(x) for x in rng.choice(10, 5, replace=False))
    vary_colors = bool(rng.random() < .65)
    held_axis = int(rng.integers(2))
    requests = ["partial", "excess"]+["random"]*(num_train-2)
    rng.shuffle(requests)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, obligations in (("train", requests), ("test", ["random"]*num_test)):
        for obligation in obligations:
            local = tuple(int(x) for x in rng.choice(10, 5, replace=False)) if vary_colors else colors
            pair = _make_pair(rng, local, obligation, split == "test", held_axis, seen)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    roles = _roles(a)
    full = _restore(a, roles)
    out, slots, active, distance, selected = _transfer(a, full, roles)
    floor, wall, payload, veil, beacon, box = roles
    region = a == veil
    reachable = (a == payload) & (distance >= 0)
    retained = reachable & ~selected
    metric = _manhattan(slots)
    row_order = np.flatnonzero(slots)
    col_order = sorted(row_order, key=lambda i: (int(i)%a.shape[1], int(i)//a.shape[1]))
    all_order = np.flatnonzero(region & (full == floor))
    n = int(selected.sum())
    opened = full.copy()
    opened[region] = floor
    try:
        open_selected = _transfer(a, opened, roles)[4]
        open_changes_selection = not np.array_equal(selected, open_selected)
    except ValueError:  # A tied counterfactual is itself not a well-defined shortcut.
        open_changes_selection = True
    return out, roles, full, slots, distance, selected, {
        "shape": a.shape, "window": (box[1]-box[0], box[3]-box[2]),
        "colors": roles[:5], "capacity": int(slots.sum()), "collected": n,
        "excess": int(retained.sum()), "unreachable": int(((a == payload) & ~active).sum()),
        "inactive_slots": int((region & (full == floor) & ~active).sum()),
        "hidden_walls": int((region & (full == wall)).sum()),
        "max_path": int(distance.max()),
        "path_reversal": bool(selected.any() and retained.any() and metric[selected].max() > metric[retained].min()),
        "row_order_witness": 0 < n < int(slots.sum()) and set(row_order[:n]) != set(col_order[:n]),
        "beacon_order_witness": 0 < n < int(slots.sum()) and set(row_order[:n]) != set(all_order[:n]),
        "open_changes_selection": open_changes_selection,
        "cross_row_axis": bool(np.any(region & region[::-1])),
        "cross_col_axis": bool(np.any(region & region[:, ::-1])),
    }


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _need(_key(a) not in seen, "Duplicate input")
            seen.add(_key(a))
            expected, roles, full, slots, distance, selected, f = _features(a)
            _need(np.array_equal(expected, b), "Incorrect reservoir output")
            floor, wall, payload, veil, beacon, box = roles
            _need(np.array_equal(distance, _wave_distances(full == floor, slots)), "Distance audit failed")
            region = a == veil
            _need(f["hidden_walls"] > 0, "Veil must hide some walls")
            _need(np.count_nonzero(a == payload) == np.count_nonzero(b == payload), "Payload is not conserved")
            _need(np.array_equal(b[a == wall], a[a == wall]), "Visible walls changed")
            _need(np.array_equal(b[a == beacon], a[a == beacon]), "Beacon changed")
            _need(not np.any(b == veil), "Veil remained")
            _need(np.all(b[region & ~slots] == full[region & ~slots]), "Inactive storage or hidden wall changed")
            _need(np.all(b[(a == payload) & ~selected] == payload), "Unselected payload changed")
            features[split].append(f)
    tr = features["train"]
    _need(any(f["row_order_witness"] and f["beacon_order_witness"] and f["inactive_slots"] > 0 for f in tr),
          "Training needs partial row-major filling and inactive hidden floor")
    _need(any(f["excess"] > 0 and f["path_reversal"] and f["open_changes_selection"] for f in tr),
          "Training needs a path-sensitive saturated example and an A-omission witness")
    _need(any(f["unreachable"] > 0 for f in tr), "Training needs unreachable distraction")
    for row_axis in (True, False):
        field = "cross_row_axis" if row_axis else "cross_col_axis"
        if all(f[field] for f in features["test"]) and not any(f[field] for f in tr):
            break
    else:
        raise ValueError("Tests must cross an axis that no training window crosses")


def self_test():
    """42 exact host seeds, bounded streaming checks, no disk artifacts."""
    signatures, shapes, windows, palettes, capacities, paths = set(), set(), set(), set(), set(), set()
    cross_axes, inactive, reversals, zero, excess = set(), 0, 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                _need(np.array_equal(solve(a.copy()), b), "Input-only solve disagrees")
                digest.update(str(a.shape).encode()+a.tobytes()+b.tobytes())
                f = _features(a)[-1]
                shapes.add(f["shape"])
                windows.add(f["window"])
                palettes.add(f["colors"])
                capacities.add(f["capacity"])
                paths.add(f["max_path"])
                cross_axes.add((f["cross_row_axis"], f["cross_col_axis"]))
                inactive += f["inactive_slots"] > 0
                reversals += f["path_reversal"]
                zero += f["collected"] == 0
                excess += f["excess"] > 0
        signatures.add(digest.hexdigest())
    _need(len(signatures) == 42, "Seed diversity collapsed")
    _need(len(shapes) >= 60 and len(windows) >= 25 and len(palettes) >= 42, "Structural/palette variation collapsed")
    _need(len(capacities) >= 15 and len(paths) >= 25 and inactive >= 42 and reversals >= 42 and excess >= 42,
          "Reservoir topology, capacity, or path variation collapsed")
    _need({(True, False), (False, True), (False, False)} <= cross_axes, "Mask geometry collapsed")
    # Recolor each role without communicating that permutation to solve.
    p = generate_puzzle(17, 2, 1)
    permutation = np.array([7, 3, 6, 1, 9, 4, 0, 8, 5, 2], dtype=np.int8)
    recolored = {s: [{k: permutation[v] for k, v in pair.items()} for pair in p[s]] for s in p}
    validate_generated_puzzle(recolored)
    # Displayed row-major order is recomputed after geometric transformations.
    for transform in (lambda a: a.T.copy(), lambda a: a[::-1].copy()):
        for pair in p["train"]+p["test"]:
            a = transform(pair["input"])
            roles = _roles(a)
            expected = _transfer(a, _restore(a, roles), roles, audit=True)[0]
            _need(np.array_equal(solve(a), expected), "Transformed input failed")
    # Tie rejection, including a tie that would split a connected payload run.
    a = np.zeros((5, 7), dtype=np.int8)
    a[2, 1:6] = 4
    distances = np.full(a.shape, -1, dtype=np.int16)
    distances[2, 1:6] = [1, 2, 2, 3, 4]
    try:
        _select(a, 4, distances, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous capacity tie accepted")
    # Traversable payload can be selected through nearer payload cells. A cluster
    # is split at a strict frontier, and an unreachable cell is left untouched.
    a = np.zeros((3, 9), dtype=np.int8)
    terrain = np.zeros(a.shape, dtype=bool)
    terrain[1, 1:8] = True
    slots = np.zeros(a.shape, dtype=bool)
    slots[1, 1:3] = True
    a[1, 3:8] = 4
    a[0, 8] = 4
    selected = _select(a, 4, _distances(terrain, slots), 2)
    expected = np.zeros(a.shape, dtype=bool)
    expected[1, 3:5] = True
    _need(np.array_equal(selected, expected), "Payload traversability or cluster splitting failed")
    # Erasing every member of an orbit is invalid even when a rectangle exists.
    a = p["train"][0]["input"].copy()
    roles = _roles(a)
    region = a == roles[3]
    a[region | region[::-1] | region[:, ::-1] | region[::-1, ::-1]] = roles[3]
    try:
        _restore(a, roles)
    except ValueError:
        pass
    else:
        raise AssertionError("Fully erased symmetry orbit accepted")
    # Corrupted labels and a completely erased symmetry orbit must be rejected.
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1)%10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    first, second = generate_puzzle(31415, 3, 2), generate_puzzle(31415, 3, 2)
    _need(all(np.array_equal(p1[k], p2[k]) for s in first for p1, p2 in zip(first[s], second[s])
              for k in ("input", "output")), "Seed determinism failed")
    return {"seeds": 42, "pairs": 210, "shapes": len(shapes), "windows": len(windows),
            "role_palettes": len(palettes), "capacities": sorted(capacities),
            "path_lengths": len(paths), "inactive_storage_pairs": inactive,
            "path_reversal_pairs": reversals, "excess_pairs": excess, "zero_pairs": zero}
