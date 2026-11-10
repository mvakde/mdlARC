"""Mirror-repaired cargo: recover a material stencil, then move it on a track.

Only the standard library and NumPy are used. No parent code or data is loaded.
All construction retries are bounded and all testing is in memory.
"""
from hashlib import sha256
import numpy as np


WALL, CENTER, GOAL, BRACE, ERASE = 0, 2, 3, 5, 9
FREE_COLORS = (1, 4, 6, 7, 8)
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _neighbors(p):
    return [(p[0] + dr, p[1] + dc) for dr, dc in DIRECTIONS]


def _connected(points):
    points = set(points)
    if not points:
        return False
    seen = {next(iter(points))}
    stack = list(seen)
    for p in stack:
        for q in _neighbors(p):
            if q in points and q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == len(points)


def _orbits(n):
    seen = set()
    for r in range(n):
        for c in range(n):
            if (r, c) not in seen:
                orbit = tuple(sorted({(r, c), (n-1-r, c), (r, n-1-c), (n-1-r, n-1-c)}))
                seen.update(orbit)
                yield orbit


def _read_input(input_grid):
    """Parse visible dock, reflection witnesses and the single painted track."""
    a = _grid(input_grid).copy()
    braces = set(map(tuple, np.argwhere(a == BRACE)))
    _require(len(braces) == 4, "four dock corner marks are required")
    r0, c0 = np.min(list(braces), axis=0)
    r1, c1 = np.max(list(braces), axis=0)
    _require(braces == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}, "dock corners are not a rectangle")
    n = int(r1-r0-1)
    _require(n in (5, 7, 9) and c1-c0-1 == n, "dock must enclose an odd square of side 5, 7 or 9")
    r0, c0 = int(r0+1), int(c0+1)
    radius = n//2
    start = (r0+radius, c0+radius)
    _require(list(map(tuple, np.argwhere(a == CENTER))) == [start], "one source center is required")
    goals = list(map(tuple, np.argwhere(a == GOAL)))
    _require(len(goals) == 1, "one track goal is required")
    goal = goals[0]
    source_mask = np.zeros(a.shape, dtype=bool)
    source_mask[r0:r0+n, c0:c0+n] = True
    exterior = set(map(int, np.unique(a[~source_mask]))) - {WALL, GOAL, BRACE}
    _require(len(exterior) == 2 and exterior <= set(FREE_COLORS), "floor and track must have two visible colors")
    candidates = []
    for color in sorted(exterior):
        points = set(map(tuple, np.argwhere((a == color) | (a == GOAL))))
        degree = {p: sum(q in points for q in _neighbors(p)) for p in points}
        ends = [p for p in points if degree[p] == 1]
        if len(ends) != 2 or goal not in ends or any(v not in (1, 2) for v in degree.values()):
            continue
        port = next(p for p in ends if p != goal)
        delta = (int(port[0]-start[0]), int(port[1]-start[1]))
        if sorted(map(abs, delta)) != [0, radius+1]:
            continue
        chain, previous, p = [], None, port
        while True:
            chain.append(p)
            if p == goal:
                break
            options = [q for q in _neighbors(p) if q in points and q != previous]
            if len(options) != 1 or len(chain) > len(points):
                break
            previous, p = p, options[0]
        if len(chain) != len(points) or chain[-1] != goal:
            continue
        dr, dc = delta[0]//(radius+1), delta[1]//(radius+1)
        path = [(start[0]+i*dr, start[1]+i*dc) for i in range(radius+1)] + chain
        if chain[1] != (port[0]+dr, port[1]+dc):
            continue
        candidates.append((color, path))
    _require(len(candidates) == 1, "track color or source exit is ambiguous")
    track, path = candidates[0]
    floor = next(iter(exterior - {track}))
    damaged = a[r0:r0+n, c0:c0+n].copy()
    _require(not (set(map(int, np.unique(damaged))) & {WALL, GOAL, BRACE, track}), "invalid source cell role")
    _require(np.any(damaged == ERASE), "cargo must contain erasures")
    clean = damaged.copy()
    diagonal_only = 0
    for orbit in _orbits(n):
        witnesses = {int(damaged[p]) for p in orbit if damaged[p] != ERASE}
        _require(len(witnesses) == 1, "reflection orbit is empty or contradictory")
        value = next(iter(witnesses))
        for r, c in orbit:
            if damaged[r, c] == ERASE:
                diagonal_only += int(damaged[n-1-r, c] == ERASE and damaged[r, n-1-c] == ERASE)
            clean[r, c] = value
    _require(clean[radius, radius] == CENTER, "center marker was damaged")
    solid = clean != floor
    _require(_connected(map(tuple, np.argwhere(solid))), "cargo material is disconnected")
    _require(np.any(clean == floor), "cargo needs transparent holes")
    _require(np.any(solid[0]) and np.any(solid[-1]) and np.any(solid[:, 0]) and np.any(solid[:, -1]), "dock is not the tight cargo bounding square")
    base = a.copy()
    base[source_mask] = floor
    return {"input": a, "base": base, "clean": clean, "damaged": damaged,
            "solid": solid, "floor": floor, "track": track, "path": path,
            "start": start, "goal": goal, "radius": radius, "box": (r0, c0, n),
            "diagonal_only": diagonal_only}


def _advance(base, path, mask):
    radius = mask.shape[0]//2
    offsets = np.argwhere(mask)-radius
    last = -1
    for i, (r, c) in enumerate(path):
        if not (radius <= r < base.shape[0]-radius and radius <= c < base.shape[1]-radius):
            break
        points = offsets + (r, c)
        values = base[points[:, 0], points[:, 1]]
        if np.any((values == WALL) | (values == BRACE)):
            break
        last = i
    _require(last >= 0, "initial cargo collides with scenery")
    return last


def _render(model, position):
    result = model["base"].copy()
    r, c = model["path"][position]
    radius = model["radius"]
    offsets = np.argwhere(model["solid"])
    result[offsets[:, 0]+r-radius, offsets[:, 1]+c-radius] = model["clean"][offsets[:, 0], offsets[:, 1]]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    model = _read_input(input_grid)
    return _render(model, _advance(model["base"], model["path"], model["solid"]))


def _turns(path):
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _route(rng, radius, harder):
    spacing = int(rng.integers(2*radius+2, 2*radius+5))
    limit = min(3, (29-2*(radius+1))//spacing+1)
    nr = nc = max(2, limit)
    if not harder and rng.random() < 0.35:
        nr = 2
    target = int(rng.integers(6, min(8, nr*nc)+1)) if harder else int(rng.integers(3, min(5, nr*nc)+1))
    for _ in range(64):
        walk = [(int(rng.integers(nr)), int(rng.integers(nc)))]
        while len(walk) < target:
            options = [q for q in _neighbors(walk[-1]) if 0 <= q[0] < nr and 0 <= q[1] < nc and q not in walk]
            if not options:
                break
            walk.append(options[int(rng.integers(len(options)))])
        turns = _turns(walk)
        if len(walk) == target and (turns >= 3 if harder else 1 <= turns <= 2):
            break
    else:
        # A finite geometric construction, not a fallback to a different rule.
        walk = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 1)] if harder else [(0, 0), (0, 1), (1, 1)]
    path = [tuple(spacing*x for x in walk[0])]
    for a, b in zip(walk, walk[1:]):
        dr, dc = b[0]-a[0], b[1]-a[1]
        for _ in range(spacing):
            path.append((path[-1][0]+dr, path[-1][1]+dc))
    points = np.asarray(path)
    low, high = points.min(axis=0), points.max(axis=0)
    minimum = high-low+2*(radius+1)+1
    padding = [int(rng.integers(0, min(3, 30-int(n))+1)) for n in minimum]
    before = [int(rng.integers(p+1)) for p in padding]
    offset = radius+1-low+before
    shape = tuple(map(int, minimum+padding))
    return shape, [tuple(map(int, p+offset)) for p in points]


def _cargo(rng, radius, floor, palette, heading, harder):
    n = 2*radius+1
    solid = np.zeros((n, n), dtype=bool)
    solid[radius, :] = solid[:, radius] = True
    probability = float(rng.uniform(0.25, 0.9))
    quarter = np.zeros((radius+1, radius+1), dtype=bool)
    quarter[0, :] = quarter[:, 0] = True
    for total in range(2, 2*radius+1):
        for r in range(1, radius+1):
            c = total-r
            if not 1 <= c <= radius or (r, c) == (radius, radius):
                continue
            quarter[r, c] = (r == c == 1) or ((quarter[r-1, c] or quarter[r, c-1]) and rng.random() < probability)
    for r in range(n):
        for c in range(n):
            solid[r, c] = quarter[abs(r-radius), abs(c-radius)]
    clean = np.full((n, n), floor, dtype=np.int8)
    orbits = list(_orbits(n))
    material = [o for o in orbits if solid[o[0]] and len(o) > 1]
    palette = list(palette)
    style = int(rng.integers(3))
    for j, orbit in enumerate(material):
        r, c = orbit[0]
        index = int(rng.integers(len(palette))) if style == 0 else ((r+c if style == 1 else max(abs(r-radius), abs(c-radius))) % len(palette))
        if j < len(palette):
            index = j
        for p in orbit:
            clean[p] = palette[index]
    clean[radius, radius] = CENTER
    damaged = clean.copy()
    # A lost leading material orbit guarantees a possible mechanical witness.
    dimension = 0 if heading[0] else 1
    sign = heading[dimension]
    four = [o for o in material if len(o) == 4]
    reach = max(max(abs(p[dimension]-radius) for p in o) for o in four)
    front = [o for o in four if max(abs(p[dimension]-radius) for p in o) == reach]
    chosen = front[int(rng.integers(len(front)))]
    witnesses = [p for p in chosen if (p[dimension]-radius)*sign < 0]
    survivor = witnesses[int(rng.integers(len(witnesses)))]
    protected = {chosen, tuple(sorted({(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)}))}
    for orbit in orbits:
        if len(orbit) == 1:
            continue
        if orbit == chosen:
            lost = [p for p in orbit if p != survivor]
        elif orbit in protected:
            witness = orbit[int(rng.integers(len(orbit)))]
            lost = [p for p in orbit if p != witness]
        else:
            count = int(rng.integers(0 if not harder else 1, len(orbit)))
            lost = [orbit[int(i)] for i in rng.permutation(len(orbit))[:count]]
        for p in lost:
            damaged[p] = ERASE
    return clean, damaged


def _first_hits(shape, path, mask):
    result = np.full(shape, len(path)+1, dtype=np.int16)
    offsets = np.argwhere(mask)-mask.shape[0]//2
    for i, p in enumerate(path):
        points = offsets+p
        y, x = points[:, 0], points[:, 1]
        result[y, x] = np.minimum(result[y, x], i)
    return result


def _make_pair(rng, radius, blocked, harder, seen):
    for _ in range(80):
        shape, path = _route(rng, radius, harder)
        colors = list(map(int, rng.permutation(FREE_COLORS)))
        floor, track = colors[:2]
        palette = colors[2:2+int(rng.integers(2, 4))]
        heading = (path[1][0]-path[0][0], path[1][1]-path[0][1])
        clean, damaged = _cargo(rng, radius, floor, palette, heading, harder)
        if np.array_equal(clean, np.rot90(clean)):
            continue
        solid = clean != floor
        visible = solid & (damaged != ERASE)
        unknown_void = (clean == floor) & (damaged == ERASE)
        full_hit = _first_hits(shape, path, solid)
        visible_hit = _first_hits(shape, path, visible)
        box_hit = _first_hits(shape, path, np.ones_like(solid))
        void_hit = _first_hits(shape, path, unknown_void)
        grid = np.full(shape, WALL, dtype=np.int8)
        for r, c in path:
            grid[r-radius:r+radius+1, c-radius:c+radius+1] = floor
        # Chambers widen independently of cargo and can extend outside its sweep.
        for index in rng.permutation(len(path))[:int(rng.integers(1, 5))]:
            r, c = path[int(index)]
            q = radius+int(rng.integers(1, 3))
            grid[max(0, r-q):min(shape[0], r+q+1), max(0, c-q):min(shape[1], c+q+1)] = floor
        forbidden = np.zeros(shape, dtype=bool)
        sr, sc = path[0]
        n = 2*radius+1
        forbidden[sr-radius:sr+radius+1, sc-radius:sc+radius+1] = True
        for p in path:
            forbidden[p] = True
        braces = [(sr+dr*(radius+1), sc+dc*(radius+1)) for dr in (-1, 1) for dc in (-1, 1)]
        for p in braces:
            forbidden[p] = True
        safe = (full_hit == len(path)+1) & ~forbidden & (box_hit > 1) & (box_hit < len(path)-1)
        useful_safe = safe & (void_hit < len(path))
        harmless = np.argwhere(useful_safe)
        if not len(harmless):
            continue
        p = harmless[int(rng.integers(len(harmless)))]
        rr, cc = map(int, p)
        patch = np.zeros(shape, dtype=bool)
        q = int(rng.integers(1, 4))
        patch[max(0, rr-q+1):rr+q, max(0, cc-q+1):cc+q] = True
        grid[safe & patch] = WALL
        if blocked:
            candidates = np.argwhere((full_hit >= radius+3) & (full_hit < len(path)-radius-3)
                                     & (visible_hit > full_hit) & ~forbidden)
            if not len(candidates):
                continue
            obstacle = tuple(map(int, candidates[int(rng.integers(len(candidates)))]))
            first = int(full_hit[obstacle])
            grid[obstacle] = WALL
            # Additional teeth after the first collision cannot change its order.
            later = np.argwhere((full_hit > first+2) & (full_hit < len(path)-1) & ~forbidden)
            for j in rng.permutation(len(later))[:int(rng.integers(0, 5))]:
                grid[tuple(later[int(j)])] = WALL
        for p in path[radius+1:]:
            grid[p] = track
        grid[path[-1]] = GOAL
        for p in braces:
            grid[p] = BRACE
        grid[sr-radius:sr+radius+1, sc-radius:sc+radius+1] = damaged
        signature = (shape, grid.tobytes())
        if signature in seen:
            continue
        model = _read_input(grid)
        destination = _advance(model["base"], path, solid)
        if (destination < len(path)-1) != blocked or destination <= 0:
            continue
        if blocked and _advance(model["base"], path, visible) <= destination:
            continue
        if not blocked and _advance(model["base"], path, solid | (damaged == ERASE)) >= destination:
            continue
        seen.add(signature)
        return {"input": grid, "output": _render(model, destination)}
    raise AssertionError("bounded cargo/track construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    size_shift = bool(rng.integers(2))
    training = [(int(rng.integers(2, 4)), False), (int(rng.integers(2, 4)), True)]
    training += [(int(rng.integers(2, 4)), bool(rng.integers(2))) for _ in range(num_train-2)]
    training = [training[int(j)] for j in rng.permutation(num_train)]
    puzzle, seen = {"train": [], "test": []}, set()
    for radius, blocked in training:
        puzzle["train"].append(_make_pair(rng, radius, blocked, False, seen))
    for _ in range(num_test):
        radius = 4 if size_shift else int(rng.integers(2, 4))
        puzzle["test"].append(_make_pair(rng, radius, bool(rng.integers(2)), not size_shift, seen))
    return puzzle


def _features(pair):
    model = _read_input(pair["input"])
    solid, damaged = model["solid"], model["damaged"]
    base, path = model["base"], model["path"]
    end = _advance(base, path, solid)
    return model, {"side": solid.shape[0], "turns": _turns(path), "steps": len(path)-1,
                   "travel": end, "success": end == len(path)-1,
                   "visible_travel": _advance(base, path, solid & (damaged != ERASE)),
                   "filled_travel": _advance(base, path, solid | (damaged == ERASE)),
                   "box_travel": _advance(base, path, np.ones_like(solid)),
                   "diagonal_only": model["diagonal_only"],
                   "lost_material": int(np.sum(solid & (damaged == ERASE))),
                   "lost_void": int(np.sum(~solid & (damaged == ERASE))),
                   "material": int(solid.sum()), "shape": base.shape,
                   "floor": model["floor"], "track": model["track"]}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, rows = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas shape changed")
            signature = (inp.shape, inp.astype(np.int8).tobytes())
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            model, row = _features(pair)
            _require(np.array_equal(_render(model, row["travel"]), out), "incorrect repair, motion or scenery preservation")
            _require(not np.any(out == ERASE), "erasure remains after movement")
            _require(row["travel"] > 0 and row["lost_material"] > 0 and row["lost_void"] > 0, "degenerate cargo example")
            _require(row["diagonal_only"] >= 2, "both folds must have missing direct witnesses")
            _require(np.array_equal(model["clean"], model["clean"][::-1]) and np.array_equal(model["clean"], model["clean"][:, ::-1]), "cargo lacks both symmetries")
            final = out == CENTER
            _require(int(final.sum()) == 1, "moving center was duplicated or lost")
            _require(int(out[model["goal"]]) == (CENTER if row["success"] else GOAL), "goal replacement disagrees with arrival")
            rows[split].append(row)
    tr = rows["train"]
    _require(any(r["success"] and r["box_travel"] < r["travel"] and r["filled_travel"] < r["travel"] for r in tr), "need a success through a recovered void")
    _require(any(not r["success"] and r["visible_travel"] > r["travel"] for r in tr), "need a collision caused by recovered material")
    _require(any(not np.array_equal(_read_input(p["input"])["clean"], np.rot90(_read_input(p["input"])["clean"])) for p in puzzle["train"]), "training must distinguish fixed orientation from turning the cargo")
    _require(all(r["side"] > max(v["side"] for v in tr) or r["turns"] > max(v["turns"] for v in tr) for r in rows["test"]), "test lacks its scale or bend shift")


def _fingerprint(puzzle):
    h = sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 seeds, every legal count, invariants and concrete negative mutations."""
    # Hand-specified geometry: the empty NE corner clears (2,14) at a bend.
    # A wall at (3,10) instead meets erased material when the center is (4,9),
    # so the last legal center is (4,8). These coordinates are not sampled.
    clean = np.array([[1, 1, 7, 1, 1], [1, 6, 6, 6, 1], [8, 7, 2, 7, 8],
                      [1, 6, 6, 6, 1], [1, 1, 7, 1, 1]], dtype=np.int8)
    damaged = clean.copy()
    for point in ((1, 1), (1, 3), (3, 1), (0, 0), (0, 4), (4, 0)):
        damaged[point] = ERASE
    fixture = np.ones((14, 17), dtype=np.int8)
    fixture[4, 7:13] = fixture[5:11, 12] = 4
    fixture[10, 12] = GOAL
    fixture[2, 14] = WALL
    for point in ((1, 1), (1, 7), (7, 1), (7, 7)):
        fixture[point] = BRACE
    fixture[2:7, 2:7] = damaged
    for blocked, destination in ((False, (10, 12)), (True, (4, 8))):
        inp = fixture.copy()
        if blocked:
            inp[3, 10] = WALL
        expected = inp.copy()
        expected[2:7, 2:7] = 1
        rr, cc = np.nonzero(clean != 1)
        expected[rr+destination[0]-2, cc+destination[1]-2] = clean[rr, cc]
        for rotation in range(4):
            _require(np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(expected, rotation)), "hand-specified footprint/rotation contrast failed")
    signatures, shapes, cargo_shapes, turns = set(), set(), set(), set()
    floors, tracks, headings, shifts, test_outcomes = set(), set(), set(), set(), set()
    success_void = blocked_material = total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "seed is not deterministic")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                model, f = _features(pair)
                total += 1
                shapes.add(f["shape"])
                cargo_shapes.add((model["solid"].shape, model["solid"].tobytes()))
                turns.add(f["turns"])
                floors.add(f["floor"])
                tracks.add(f["track"])
                p, q = model["path"][:2]
                headings.add((q[0]-p[0], q[1]-p[1]))
                success_void += f["success"] and f["filled_travel"] < f["travel"]
                blocked_material += not f["success"] and f["visible_travel"] > f["travel"]
                if split == "test":
                    shifts.add("size" if f["side"] == 9 else "bends")
                    test_outcomes.add(f["success"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "input-only solve disagrees")
                # The 180-degree mirror must suffice even when both direct mates are erased.
                d, clean = model["damaged"], model["clean"]
                for r, c in np.argwhere(d == ERASE):
                    witnesses = [d[-1-r, c], d[r, -1-c], d[-1-r, -1-c]]
                    values = {int(v) for v in witnesses if v != ERASE}
                    _require(values == {int(clean[r, c])}, "mirror witness mismatch")
    _require(len(signatures) == 42 and total == 210, "seed diversity collapsed")
    _require(len(shapes) >= 35 and len(cargo_shapes) >= 35, "geometry variation collapsed")
    _require({1, 2, 3, 4} <= turns, "bend spectrum missing")
    _require(floors == tracks == set(FREE_COLORS) and len(headings) == 4, "palette or heading coverage missing")
    _require(shifts == {"size", "bends"} and test_outcomes == {False, True}, "test coverage missing")
    _require(success_void >= 42 and blocked_material >= 42, "parent ablations lack witnesses")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt-1)+1):
            p = generate_puzzle(912, nt, ns)
            _require(len(p["train"]) == nt and len(p["test"]) == ns, "count was ignored")
            validate_generated_puzzle(p)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0)
    for kind in ("material", "visible_scenery", "center"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in sample[s]] for s in sample}
        pair = bad["test"][0]
        if kind == "material":
            point = tuple(np.argwhere(np.isin(pair["output"], FREE_COLORS))[0])
        elif kind == "visible_scenery":
            point = tuple(np.argwhere(pair["input"] == BRACE)[0])
        else:
            point = tuple(np.argwhere(pair["output"] == CENTER)[0])
        pair["output"][point] = (int(pair["output"][point])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted corrupt output")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds and every legal explicit count")
