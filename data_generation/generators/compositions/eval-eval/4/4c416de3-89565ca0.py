"""Corner-probe census: reflected incomplete stencils query enclosed rooms.

Only standard-library modules and NumPy are used. The decoder accepts pixels
alone; generation metadata and random state are never passed to solve().
"""

import hashlib
import numpy as np


DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected a 1..30 integer ARC grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        todo = [(y, x)]
        cells = []
        while todo:
            r, c = todo.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    todo.append((rr, cc))
        result.append(cells)
    return result


def _box(cells):
    a = np.asarray(cells)
    t, l = a.min(axis=0)
    b, r = a.max(axis=0)
    return tuple(map(int, (t, l, b, r)))


def _separate(a, b):
    return a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1]


def _corner(box, q):
    t, l, b, r = box
    return ((t, l), (t, r), (b, l), (b, r))[q]


def _positions(box, q, pattern):
    y, x = _corner(box, q)
    dy, dx = DIRS[q]
    return {(y + dy*u, x + dx*v) for u, v in pattern}


def _nearest(box, point):
    d = [(point[0]-y)**2 + (point[1]-x)**2 for y, x in
         (_corner(box, q) for q in range(4))]
    if d.count(min(d)) != 1:
        raise ValueError("Ambiguous corner association")
    return d.index(min(d))


def _regions(local_walls):
    """Label elementary free rooms inside an intact rectangular perimeter."""
    if not (local_walls[0].all() and local_walls[-1].all()
            and local_walls[:, 0].all() and local_walls[:, -1].all()):
        raise ValueError("Open outer boundary")
    labels = np.zeros(local_walls.shape, dtype=np.int16)
    rooms = _components(~local_walls)
    for i, cells in enumerate(rooms, 1):
        t, l, b, r = _box(cells)
        if len(cells) != (b-t+1)*(r-l+1):
            raise ValueError("Family rooms must be rectangular")
        for p in cells:
            labels[p] = i
    if not rooms:
        raise ValueError("Frame has no positive-area room")
    return labels, [_box(cells) for cells in rooms]


def _cycle_count(mask):
    """Euler check on the connected wall graph, discounting solid 2x2 faces."""
    edges = int(np.count_nonzero(mask[:, :-1] & mask[:, 1:]))
    edges += int(np.count_nonzero(mask[:-1, :] & mask[1:, :]))
    unit_faces = int(np.count_nonzero(mask[:-1, :-1] & mask[:-1, 1:]
                                     & mask[1:, :-1] & mask[1:, 1:]))
    return edges - int(mask.sum()) + 1 - unit_faces


def _repair(inp, wall, noise):
    original = inp == wall
    fixed = original.copy()
    h, w = inp.shape
    for y, x in np.argwhere(inp == noise):
        if (0 < x < w-1 and original[y, x-1] and original[y, x+1]) or (
                0 < y < h-1 and original[y-1, x] and original[y+1, x]):
            fixed[y, x] = True
    return fixed


def _decode(inp, rotation=False):
    """Recover roles, geometry, stencil and incidence entirely from the input."""
    inp = _grid(inp)
    frequencies = np.bincount(inp.ravel(), minlength=10)
    ranked = sorted(range(10), key=lambda c: (-int(frequencies[c]), c))
    background, wall = ranked[:2]
    if not (frequencies[background] > frequencies[wall] > frequencies[ranked[2]] > 0):
        raise ValueError("Background/wall ranks are not unique")
    components = _components(inp == wall)
    boxes = sorted(_box(cells) for cells in components)
    if not 2 <= len(boxes) <= 4:
        raise ValueError("Expected 2..4 complete separated frames")
    covered = np.zeros(inp.shape, dtype=bool)
    for i, (t, l, b, r) in enumerate(boxes):
        if min(b-t+1, r-l+1) < 7 or any(not _separate(boxes[i], a) for a in boxes[:i]):
            raise ValueError("Frames too small or not separated")
        local = inp[t:b+1, l:r+1] == wall
        if not (local[0].all() and local[-1].all() and local[:, 0].all() and local[:, -1].all()):
            raise ValueError("Outer frame must be intact")
        covered[t:b+1, l:r+1] = True
    exterior = set(map(int, np.unique(inp[~covered]))) - {background, wall}
    if len(exterior) != 1:
        raise ValueError("Noise must be uniquely visible outside the frames")
    noise = exterior.pop()
    walls = _repair(inp, wall, noise)
    records = []
    for box in boxes:
        t, l, b, r = box
        local = inp[t:b+1, l:r+1]
        labels, rooms = _regions(walls[t:b+1, l:r+1])
        accent = set(map(int, np.unique(local))) - {background, wall, noise}
        if len(accent) != 1:
            raise ValueError("Each frame needs one local accent color")
        color = accent.pop()
        points = {(int(y)+t, int(x)+l) for y, x in np.argwhere(local == color)}
        corners = {_nearest(box, p) for p in points}
        if len(corners) != 1:
            raise ValueError("Clue spans multiple corners")
        q = corners.pop()
        ay, ax = _corner(box, q)
        dy, dx = DIRS[q]
        clue = frozenset(((y-ay)*dy, (x-ax)*dx) for y, x in points)
        if any(not (1 <= u <= 5 and 1 <= v <= 5) for u, v in clue):
            raise ValueError("Probe outside the inward footprint")
        if rotation and q in (1, 2):
            clue = frozenset((v, u) for u, v in clue)
        records.append(dict(box=box, color=color, q=q, clue=clue, labels=labels,
                            rooms=rooms, total=len(rooms), points=points))
    maximal = {a["clue"] for a in records if all(b["clue"] <= a["clue"] for b in records)}
    if len(maximal) != 1:
        raise ValueError("No intact common stencil")
    pattern = maximal.pop()
    if not (3 <= len(pattern) <= 9 and len({u for u, v in pattern}) >= 2
            and len({v for u, v in pattern}) >= 2):
        raise ValueError("Invalid common probe constellation")
    full = 0
    for rec in records:
        box, q = rec["box"], rec["q"]
        local_pattern = frozenset((v, u) for u, v in pattern) if rotation and q in (1, 2) else pattern
        positions = _positions(box, q, local_pattern)
        t, l, b, r = box
        if any(not (t < y < b and l < x < r) or _nearest(box, (y, x)) != q for y, x in positions):
            raise ValueError("Virtual stencil has ambiguous or out-of-frame probes")
        selected = {int(rec["labels"][y-t, x-l]) for y, x in positions} - {0}
        observed = {int(rec["labels"][y-t, x-l]) for y, x in rec["points"]} - {0}
        pixel_count = sum(rec["labels"][y-t, x-l] != 0 for y, x in positions)
        if not selected or any(rec["labels"][y-t, x-l] == 0 for y, x in rec["points"]):
            raise ValueError("A clue must occupy a free room")
        rec.update(count=len(selected), visible_count=len(observed), pixels=int(pixel_count),
                   wall_hits=len(pattern)-int(pixel_count), virtual=positions)
        full += int(rec["clue"] == pattern and pixel_count == len(pattern))
    if not full:
        raise ValueError("No entirely visible intact reference")
    return dict(background=background, wall=wall, noise=noise, walls=walls,
                pattern=pattern, records=records, full=full)


def _answer(info, quantity="count", column_first=False):
    def key(rec):
        t, l, _, _ = rec["box"]
        return (rec[quantity], l, t) if column_first else (rec[quantity], t, l)
    ordered = sorted(info["records"], key=key)
    result = np.full((len(ordered), max(r[quantity] for r in ordered)), info["noise"], dtype=np.int8)
    for y, rec in enumerate(ordered):
        result[y, :rec[quantity]] = rec["color"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _answer(_decode(input_grid))


def _capacity(box):
    t, l, b, r = box
    return ((b-t)//2)*((r-l)//2)


def _partition(rng, h, w, count, protected):
    """Recursive room splitting, retaining actual divider endpoints."""
    boxes = [(0, 0, h-1, w-1)]
    for _ in range(count-1):
        choices = []
        for i, (t, l, b, r) in enumerate(boxes):
            for axis, lo, hi in ((0, t, b), (1, l, r)):
                for cut in range(lo+2, hi-1):
                    if any(t <= y <= b and l <= x <= r and (y if axis == 0 else x) == cut
                           for y, x in protected):
                        continue
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    if sum(_capacity(a) for j, a in enumerate(boxes) if j != i) + sum(map(_capacity, halves)) >= count:
                        choices.append((i, halves))
        if not choices:
            return None
        i, halves = choices[int(rng.integers(len(choices)))]
        boxes[i:i+1] = halves
    mask = np.zeros((h, w), dtype=bool)
    for t, l, b, r in boxes:
        mask[t, l:r+1] = mask[b, l:r+1] = True
        mask[t:b+1, l] = mask[t:b+1, r] = True
    return mask


def _stencil(rng, radius, witness):
    domain = [(u, v) for u in range(1, radius+1) for v in range(1, radius+1)]
    for _ in range(512):
        n = int(rng.integers(3 if radius < 5 else 4, min(9, len(domain))+1))
        points = frozenset(domain[int(i)] for i in rng.choice(len(domain), n, replace=False))
        us, vs = zip(*points)
        if min(us) != 1 or min(vs) != 1 or max(us) != radius or max(vs) != radius:
            continue
        if witness and points == frozenset((v, u) for u, v in points):
            continue
        return points
    # Finite geometric safety construction, never used as a solver dictionary.
    return frozenset({(1, 1), (1, radius), (radius, 1), (radius, 2)})


def _layout(rng, n, radius):
    minimum = 2*radius+3
    low = 2*minimum+1 if n >= 3 else minimum
    if low > 30:
        return None
    h, w = map(int, rng.integers(low, 31, 2))
    max_side = min(18, minimum+3) if n <= 2 else minimum+2
    # Independent positions and dimensions; retries are packing constraints,
    # not fixed quadrants or template scene layouts.
    for _ in range(50):
        boxes = []
        for i in range(n):
            for _ in range(80):
                bh = int(rng.integers(minimum, min(h, max_side)+1))
                bw = int(rng.integers(minimum, min(w, max_side)+1))
                t, l = int(rng.integers(h-bh+1)), int(rng.integers(w-bw+1))
                box = t, l, t+bh-1, l+bw-1
                if all(_separate(box, other) for other in boxes):
                    boxes.append(box)
                    break
            else:
                break
        if len(boxes) == n:
            return (h, w), boxes
    return None


def _lattice_count(rec):
    labels = rec["labels"]
    walls = labels == 0
    hs = np.flatnonzero(np.any(walls[:, :-1] & walls[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(walls[:-1, :] & walls[1:, :], axis=0))
    return (len(hs)-1)*(len(vs)-1)


def _lattice_census(rec):
    """Countermodel: extend every supported divider through open space."""
    walls = rec["labels"] == 0
    hs = np.flatnonzero(np.any(walls[:, :-1] & walls[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(walls[:-1, :] & walls[1:, :], axis=0))
    extended = walls.copy()
    extended[hs, :] = True
    extended[:, vs] = True
    if extended.all():
        return 0
    labels, _ = _regions(extended)
    t, l, _, _ = rec["box"]
    return len({int(labels[y-t, x-l]) for y, x in rec["virtual"]} - {0})


def _uncorrected_census(inp, wall, rec):
    """Countermodel: treat every noise interruption as ordinary free space."""
    t, l, b, r = rec["box"]
    walls = inp[t:b+1, l:r+1] == wall
    if np.array_equal(walls, rec["labels"] == 0):
        return rec["count"]
    labels = np.zeros(walls.shape, dtype=np.int16)
    # This countermodel can create L-shaped regions, so it deliberately does
    # not use the production parser's rectangular-room validity restriction.
    for i, cells in enumerate(_components(~walls), 1):
        for p in cells:
            labels[p] = i
    return len({int(labels[y-t, x-l]) for y, x in rec["virtual"]} - {0})


def _evidence(inp, info):
    rs = info["records"]
    counts = [r["count"] for r in rs]  # records already in reading order
    partial = any(_lattice_count(r) != r["total"] for r in rs)
    lattice_info = dict(info, records=[dict(r, lattice_census=_lattice_census(r)) for r in rs])
    lattice_diff = not np.array_equal(_answer(info), _answer(lattice_info, "lattice_census"))
    uncorrected_info = dict(info, records=[dict(r, uncorrected=_uncorrected_census(inp, info["wall"], r))
                                        for r in rs])
    repair_diff = not np.array_equal(_answer(info), _answer(uncorrected_info, "uncorrected"))
    tie_order = not np.array_equal(_answer(info), _answer(info, column_first=True))
    try:
        rotation_diff = not np.array_equal(_answer(info), _answer(_decode(inp, rotation=True)))
    except ValueError:
        rotation_diff = True
    return dict(A=any(r["visible_count"] < r["count"] for r in rs),
                B=any(r["pixels"] > r["count"] for r in rs),
                selective=any(r["total"] > r["count"] for r in rs),
                order=counts != sorted(counts), padding=min(counts) < max(counts),
                tie=len(counts) > len(set(counts)), tie_order=tie_order,
                rotation=rotation_diff, partial=partial,
                lattice=lattice_diff,
                repair_effect=repair_diff,
                damage=bool(np.any(info["walls"] & (inp == info["noise"]))),
                wall_hit=any(r["wall_hits"] for r in rs))


def _draw(rng, split, witness, stable_roles):
    radius = 3 if witness == 1 else int(rng.choice((3, 4))) if split == "train" else int(rng.choice((4, 5)))
    n = 3 if witness == 1 else int(rng.choice((2, 2, 3, 4)))
    pattern = _stencil(rng, radius, witness == 1)
    layout = _layout(rng, n, radius)
    if layout is None:
        return None
    shape, boxes = layout
    roles = stable_roles if stable_roles is not None else tuple(map(int, rng.choice(10, 3, replace=False)))
    background, wall, noise = roles
    palette = [c for c in range(10) if c not in roles]
    rng.shuffle(palette)
    colors = palette[:n]
    if not witness and rng.random() < .30:
        colors = [int(rng.choice(palette[:int(rng.integers(1, n+1))])) for _ in range(n)]
    orientations = [0, 1, 2] if witness == 1 else list(map(int, rng.integers(0, 4, n)))
    reference = 0 if witness == 1 else int(rng.integers(n))
    masks, labels, positions, selected = [], [], [], []
    clean = np.full(shape, background, dtype=np.int8)
    for i, (box, q) in enumerate(zip(boxes, orientations)):
        t, l, b, r = box
        points = _positions(box, q, pattern)
        protected = {(y-t, x-l) for y, x in points} if i == reference else set()
        count = int(rng.integers(3, 7)) if split == "train" else int(rng.integers(7, 11))
        local = _partition(rng, b-t+1, r-l+1, count, protected)
        if local is None:
            return None
        lab, _ = _regions(local)
        ids = {int(lab[y-t, x-l]) for y, x in points} - {0}
        if not ids:
            return None
        mask = np.zeros(shape, dtype=bool)
        mask[t:b+1, l:r+1] = local
        clean[mask] = wall
        masks.append(mask)
        labels.append(lab)
        positions.append(points)
        selected.append(ids)
    targets = [i for i in range(n) if i != reference and len(selected[i]) >= 2]
    if not targets:
        return None
    hidden_room_target = int(rng.choice(targets))
    inp = clean.copy()
    for i, (box, color, pts) in enumerate(zip(boxes, colors, positions)):
        t, l, _, _ = box
        available = sorted(p for p in pts if clean[p] != wall)
        if i == reference:
            keep = available
        elif i == hidden_room_target:
            room = int(rng.choice(sorted(selected[i])))
            choices = [p for p in available if labels[i][p[0]-t, p[1]-l] == room]
            amount = int(rng.integers(1, len(choices)+1))
            keep = [choices[int(j)] for j in rng.choice(len(choices), amount, replace=False)]
        elif len(available) == len(pattern) and rng.random() < .20:
            keep = available
        else:
            limit = len(available)-1 if len(available) == len(pattern) else len(available)
            amount = int(rng.integers(1, max(1, limit)+1))
            keep = [available[int(j)] for j in rng.choice(len(available), amount, replace=False)]
        for p in keep:
            inp[p] = color
    union = np.logical_or.reduce(masks)
    # Straight single-cell gaps retain immediate opposite support. Check that
    # they also retain one connected visible wall component per outer frame.
    for frame_id, (interior, mask) in enumerate(zip(boxes, masks)):
        candidates = []
        for y, x in np.argwhere(mask):
            if not (0 < y < shape[0]-1 and 0 < x < shape[1]-1):
                continue
            horizontal = mask[y, x-1] and mask[y, x+1]
            vertical = mask[y-1, x] and mask[y+1, x]
            t, l, b, r = interior
            if t < y < b and l < x < r and horizontal != vertical:
                candidates.append((int(y), int(x)))
        rng.shuffle(candidates)
        if witness:
            # Prefer holes between two selected rooms, making reconstruction
            # consequential for the census rather than only cosmetic damage.
            t, l, _, _ = interior
            lab, ids = labels[frame_id], selected[frame_id]
            def separates_selected(p):
                y, x = p[0]-t, p[1]-l
                pairs = ((int(lab[y-1, x]), int(lab[y+1, x])),
                         (int(lab[y, x-1]), int(lab[y, x+1])))
                return any(a != b and a in ids and b in ids for a, b in pairs)
            candidates.sort(key=lambda p: not separates_selected(p))
        desired = int(rng.integers(1, 4)) if witness else int(rng.integers(0, 4))
        damaged = 0
        for p in candidates:
            if damaged >= desired:
                break
            y, x = p
            visible = inp == wall
            supported = (visible[y, x-1] and visible[y, x+1]) or (visible[y-1, x] and visible[y+1, x])
            if not supported:
                continue
            inp[p] = noise
            if len(_components(inp == wall)) != n:
                inp[p] = wall
                continue
            damaged += 1
    covered = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        covered[t:b+1, l:r+1] = True
    exterior = np.argwhere(~covered)
    amount = min(len(exterior), int(rng.integers(3, 17)))
    for y, x in exterior[rng.choice(len(exterior), amount, replace=False)]:
        inp[y, x] = noise
    try:
        info = _decode(inp)
    except ValueError:
        return None
    if info["pattern"] != pattern or not np.array_equal(info["walls"], union):
        return None
    recovered = {r["box"]: r for r in info["records"]}
    if any(recovered[box]["count"] != len(ids) for box, ids in zip(boxes, selected)):
        return None
    ev = _evidence(inp, info)
    if not all(ev[k] for k in ("A", "B", "selective")):
        return None
    if witness == 1 and not all(ev[k] for k in ("order", "padding", "tie", "tie_order", "rotation")):
        return None
    if witness == 2 and not all(ev[k] for k in ("partial", "damage", "wall_hit", "lattice", "repair_effect")):
        return None
    # Labels are derived from the clean construction before decoder comparison.
    intended = dict(noise=noise, records=[dict(box=box, color=color, count=len(ids))
                                         for box, color, ids in zip(boxes, colors, selected)])
    return {"input": inp, "output": _answer(intended)}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    stable_roles = tuple(map(int, rng.choice(10, 3, replace=False))) if rng.random() < .45 else None
    puzzle = {"train": [], "test": []}
    seen, stencils = set(), set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            witness = i+1 if split == "train" and i < 2 else 0
            for _ in range(1500):
                pair = _draw(rng, split, witness, stable_roles)
                if pair is None or _key(pair["input"]) in seen:
                    continue
                pattern = _decode(pair["input"])["pattern"]
                if split == "train" and i == 1 and pattern in stencils:
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair["input"]))
                stencils.add(pattern)
                break
            else:
                raise RuntimeError("Bounded construction could not find a witness")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {"train", "test"}:
        raise ValueError("Puzzle must have exactly train/test keys")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("Splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(ne, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("Too few training examples")
    seen, evidence, stencils = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {"input", "output"}:
                raise ValueError("Pair must have exactly input/output keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            if _key(inp) in seen:
                raise ValueError("Repeated input")
            seen.add(_key(inp))
            info = _decode(inp)
            for rec in info["records"]:
                t, l, b, r = rec["box"]
                if _cycle_count(info["walls"][t:b+1, l:r+1]) != rec["total"]:
                    raise ValueError("Flood-fill rooms disagree with planar wall topology")
            if not np.array_equal(_answer(info), out):
                raise ValueError("Output differs from input-only census")
            lo, hi = (3, 6) if split == "train" else (7, 10)
            if any(not lo <= rec["total"] <= hi for rec in info["records"]):
                raise ValueError("Room-count train/test shift is missing")
            if np.array_equal(_answer(info, "visible_count"), out):
                raise ValueError("A's template recovery has no effect")
            if np.array_equal(_answer(info, "pixels"), out):
                raise ValueError("B's room deduplication has no effect")
            if np.array_equal(_answer(info, "total"), out):
                raise ValueError("Selective count is indistinguishable from total rooms")
            if split == "train":
                evidence.append(_evidence(inp, info))
                stencils.add(info["pattern"])
    if len(stencils) < 2:
        raise ValueError("Demonstrations need different instance-specific stencils")
    if not any(all(e[k] for k in ("order", "padding", "tie", "tie_order", "rotation")) for e in evidence):
        raise ValueError("Missing sorting/tie/reflection witness")
    if not any(all(e[k] for k in ("partial", "damage", "wall_hit", "lattice", "repair_effect")) for e in evidence):
        raise ValueError("Missing terminating-divider/repair/wall-hit witness")


def _signature(puzzle):
    return hashlib.sha256(b"".join(bytes(p[k].shape)+p[k].tobytes()
                                   for s in ("train", "test") for p in puzzle[s]
                                   for k in ("input", "output"))).hexdigest()


def _incidence_regression():
    """Hand-calculated example, rather than labels made by solve()."""
    inp = np.zeros((11, 21), dtype=np.int8)
    for t, l, b, r in ((1, 1, 9, 9), (1, 11, 9, 19)):
        inp[t, l:r+1] = inp[b, l:r+1] = 5
        inp[t:b+1, l] = inp[t:b+1, r] = 5
    # Left: one large upper room and two small lower rooms. Right: one
    # narrow right room and two rooms on the left of a terminating divider.
    inp[7, 1:10] = 5
    inp[7:10, 5] = 5
    inp[1:10, 17] = 5
    inp[3, 11:18] = 5
    for u, v in {(1, 1), (1, 3), (3, 1), (3, 3), (2, 3)}:
        inp[1+u, 1+v] = 1
    inp[2, 18] = 2
    inp[6, 17] = inp[0, 10] = 9
    expected = np.array([[1, 9, 9], [2, 2, 2]], dtype=np.int8)
    assert np.array_equal(solve(inp), expected)
    info = _decode(inp)
    assert [(r["total"], r["count"], r["pixels"], r["wall_hits"])
            for r in info["records"]] == [(3, 1, 5, 0), (3, 3, 4, 1)]
    assert not np.array_equal(_answer(info, "visible_count"), expected)
    assert not np.array_equal(_answer(info, "pixels"), expected)
    assert _lattice_census(info["records"][1]) == 4
    assert _uncorrected_census(inp, info["wall"], info["records"][1]) == 2
    # A deliberately open divider gap is not repaired as if it were noise.
    open_gap = inp.copy()
    open_gap[6, 17] = 0
    assert not _repair(open_gap, 5, 9)[6, 17]


def self_test():
    """Bounded 42-seed semantic, diversity, count and metamorphic checks."""
    _incidence_regression()
    shapes, patterns, signatures, counts, orientations, roles, room_sizes = set(), set(), set(), set(), set(), set(), set()
    records = []
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        signatures.add(_signature(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"].copy()), pair["output"])
                shapes.add(pair["input"].shape)
                patterns.add(info["pattern"])
                roles.add((info["background"], info["wall"], info["noise"]))
                counts.add(len(info["records"]))
                for rec in info["records"]:
                    orientations.add(rec["q"])
                    room_sizes.update((b-t+1, r-l+1) for t, l, b, r in rec["rooms"])
                records.append(info)
    assert len(signatures) == 42 and len(shapes) >= 30 and len(patterns) >= 50
    assert counts == {2, 3, 4} and orientations == {0, 1, 2, 3} and len(roles) >= 30
    assert {max(max(p) for p in pattern) for pattern in patterns} == {3, 4, 5}
    assert len(room_sizes) >= 30 and any(min(a) == 1 for a in room_sizes)
    assert any(i["full"] > 1 for i in records)
    assert any(len({r["color"] for r in i["records"]}) < len(i["records"]) for i in records)
    assert _signature(generate_puzzle(0)) == _signature(puzzles[0])
    # Every legal explicit count combination, including the maximum 14 pairs.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            generate_puzzle(42 + nt*5 + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
    for puzzle in puzzles[:3]:
        for pair in puzzle["train"] + puzzle["test"]:
            assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
            info = _decode(pair["input"])
            if len({r["count"] for r in info["records"]}) == len(info["records"]):
                for moved in (np.rot90(pair["input"]), pair["input"].T, np.fliplr(pair["input"])):
                    assert np.array_equal(solve(moved), pair["output"])
    # Wrong output colors and unrelated padding changes must be rejected.
    broken = {s: [dict(input=p["input"].tolist(), output=p["output"].tolist())
                  for p in puzzles[0][s]] for s in ("train", "test")}
    broken["test"][0]["output"][0][0] = (broken["test"][0]["output"][0][0]+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed answer accepted")
    return {"seeds": 42, "puzzles": len(signatures), "input_shapes": len(shapes),
            "stencils": len(patterns), "role_triples": len(roles), "room_sizes": len(room_sizes)}


if __name__ == "__main__":
    print(self_test())
