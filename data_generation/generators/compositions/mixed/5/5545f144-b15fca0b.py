"""Tip-led waypoint journeys whose legs are unique shortest maze routes.

Only Python's standard library and NumPy are used. No files, generation state,
or examples are consulted by solve(). See task_spec.md for the exact convention.
"""
from collections import deque
import hashlib
import numbers

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ROUTE = 4  # A fixed family convention; this color is absent from all inputs.


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)),
             "grid cells must be integers in 0..9")
    return a


def _rot(point, k):
    r, c = point
    for _ in range(k % 4):
        r, c = c, -r
    return r, c


def _footprint(mask, tip, direction):
    return {(tip[0] + dr, tip[1] + dc)
            for dr, dc in (_rot(p, direction - 1) for p in mask)}


def _direction(a, b):
    delta = (b[0] - a[0], b[1] - a[1])
    _require(delta in DIRS, "route steps must be four-neighbor steps")
    return DIRS.index(delta)


def _components(cells):
    # Eight-connectivity recognizes the diagonal wings of a hollow arrow.
    left, result = set(cells), []
    while left:
        start = min(left)
        left.remove(start)
        component, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in left:
                        left.remove(q)
                        component.add(q)
                        stack.append(q)
        result.append(component)
    return result


def _identify_mover(cells):
    """Infer geometry, rather than selecting a stored shape by cell count."""
    matches = []
    for tip in sorted(cells):
        for d in range(4):
            mask = {_rot((r - tip[0], c - tip[1]), 1 - d) for r, c in cells}
            if (max(c for r, c in mask) == 0
                    and {p for p in mask if p[1] == 0} == {(0, 0)}
                    and max(abs(r) for r, c in mask) > 0
                    and {(-r, c) for r, c in mask} == mask):
                matches.append((tuple(sorted(mask)), tip, d))
    _require(len(matches) == 1, "mover must have one geometrically unique tip")
    return matches[0]


def _order(tip, waypoints):
    """A's order ignores walls: one occupied ray, nearest point on that ray."""
    remaining, points = set(waypoints), [tip]
    while remaining:
        r, c = points[-1]
        rays = {}
        for q in sorted(remaining):
            dr, dc = q[0] - r, q[1] - c
            if dr == 0 and dc:
                rays.setdefault(1 if dc > 0 else 3, []).append(q)
            elif dc == 0 and dr:
                rays.setdefault(2 if dr > 0 else 0, []).append(q)
        _require(len(rays) == 1, "waypoints need exactly one occupied ray at each arrival")
        candidates = next(iter(rays.values()))
        q = min(candidates, key=lambda p: abs(p[0] - r) + abs(p[1] - c))
        remaining.remove(q)
        points.append(q)
    return points


def _search(walls, start, goal, blocked=()):
    """BFS with shortest-path multiplicities saturated at two (never tie-break)."""
    h, w = walls.shape
    forbidden = set(blocked)
    dist, ways, parent = {start: 0}, {start: 1}, {}
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for dr, dc in DIRS:
            q = p[0] + dr, p[1] + dc
            if (not (0 <= q[0] < h and 0 <= q[1] < w)
                    or walls[q] or q in forbidden):
                continue
            nd = dist[p] + 1
            if q not in dist:
                dist[q], ways[q], parent[q] = nd, ways[p], p
                queue.append(q)
            elif dist[q] == nd:
                ways[q] = min(2, ways[q] + ways[p])
    if goal not in dist:
        return [], 0, dist
    path, q = [goal], goal
    while q != start:
        q = parent[q]
        path.append(q)
    return path[::-1], ways[goal], dist


def _legs(walls, points):
    result = []
    for i, (a, b) in enumerate(zip(points, points[1:])):
        # Later terminals cannot be crossed before their scheduled visit.
        path, count, _ = _search(walls, a, b, points[i + 2:])
        _require(count == 1, "every leg must have exactly one shortest path")
        result.append(path)
    return result


def _poses(points, legs):
    """Check both incoming and next-step dock poses; final keeps incoming.

    The graph routes the tip cursor, not the full body's swept volume. Heading
    may turn along the route; only terminal body footprints require clearance.
    """
    result = []
    for i, p in enumerate(points):
        headings = set()
        if i:
            headings.add(_direction(legs[i - 1][-2], p))
        if i < len(legs):
            headings.add(_direction(p, legs[i][1]))
        result.append(headings)
    return result


def _route_cells(points, legs):
    # Every terminal is a docking point, never route ink (even on a later revisit).
    return set().union(*(set(path[1:-1]) for path in legs)) - set(points)


def _decode(input_grid):
    a = _grid(input_grid)
    _require(ROUTE not in a, "route color 4 is reserved for outputs")
    wall = int(a[0, 0])
    _require(np.all(a[0] == wall) and np.all(a[-1] == wall)
             and np.all(a[:, 0] == wall) and np.all(a[:, -1] == wall),
             "a uniform wall frame must enclose the maze")
    values, counts = np.unique(a[a != wall], return_counts=True)
    _require(len(values) == 2 and counts[0] != counts[1], "need background and sparse mover ink")
    bg, ink = int(values[np.argmax(counts)]), int(values[np.argmin(counts)])
    components = _components(map(tuple, np.argwhere(a == ink)))
    movers = [c for c in components if len(c) > 1]
    _require(len(movers) == 1, "one separated mover is required")
    mask, tip, initial = _identify_mover(movers[0])
    waypoints = {next(iter(c)) for c in components if len(c) == 1}
    _require(len(waypoints) >= 2, "at least two waypoint terminals are required")
    points = _order(tip, waypoints)
    walls = a == wall
    legs = _legs(walls, points)
    _require(initial == _direction(legs[0][0], legs[0][1]),
             "initial mover must face the first route step")
    for i, headings in enumerate(_poses(points, legs)):
        for d in headings:
            footprint = _footprint(mask, points[i], d)
            _require(all(0 <= r < a.shape[0] and 0 <= c < a.shape[1]
                         and not walls[r, c] for r, c in footprint),
                     "arrival/departure mover collides with a wall or boundary")
            _require(all(max(abs(q[0] - r), abs(q[1] - c)) > 1
                         for q in points[i + 1:] for r, c in footprint),
                     "future terminals must stay separated from docked mover")
    final = _direction(legs[-1][-2], legs[-1][-1])
    return dict(grid=a, bg=bg, ink=ink, wall=wall, walls=walls, mask=mask,
                points=points, legs=legs, initial=initial, final=final,
                route=_route_cells(points, legs))


def _render(info):
    out = np.full(info["grid"].shape, info["bg"], dtype=np.int8)
    out[info["walls"]] = info["wall"]
    for p in info["route"]:
        out[p] = ROUTE
    # Final mover wins over route paint on its footprint, but never over walls.
    for p in _footprint(info["mask"], info["points"][-1], info["final"]):
        out[p] = info["ink"]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one input without a seed, training examples, caches, or latent state."""
    return _render(_decode(input_grid))


def _arrow(extension, solid, tails):
    cells = {(0, -i) for i in range(extension)}
    cells.update((r, -extension) for r in (range(-1, 2) if solid else (-1, 1)))
    for depth in range(1, tails + 1):
        cells.update((r, -extension - depth) for r in (-1, 1))
    return tuple(sorted(cells))


MASKS = tuple(_arrow(e, s, t) for e in (1, 2) for s in (False, True)
              for t in (0, 1) if e + t <= 2)


def _maze(rng, nr, nc, loops):
    """Random growing-tree maze, then add real competing corridors."""
    walls = np.ones((2 * nr + 1, 2 * nc + 1), dtype=bool)
    start = int(rng.integers(nr)), int(rng.integers(nc))
    active, visited = [start], {start}
    walls[2 * start[0] + 1, 2 * start[1] + 1] = False
    newest_probability = float(rng.uniform(0.55, 1.0))
    while active:
        index = len(active) - 1 if rng.random() < newest_probability else int(rng.integers(len(active)))
        r, c = active[index]
        choices = [(r + dr, c + dc) for dr, dc in DIRS
                   if 0 <= r + dr < nr and 0 <= c + dc < nc
                   and (r + dr, c + dc) not in visited]
        if not choices:
            active.pop(index)
            continue
        q = choices[int(rng.integers(len(choices)))]
        walls[r + q[0] + 1, c + q[1] + 1] = False
        walls[2 * q[0] + 1, 2 * q[1] + 1] = False
        active.append(q)
        visited.add(q)
    candidates = [(r, c) for r in range(1, walls.shape[0] - 1)
                  for c in range(1, walls.shape[1] - 1)
                  if (r % 2 != c % 2) and walls[r, c]]
    rng.shuffle(candidates)
    for p in candidates[:loops]:
        walls[p] = False
    return walls


def _sample_points(rng, shape, count):
    nodes = [(r, c) for r in range(3, shape[0] - 3, 2)
             for c in range(3, shape[1] - 3, 2)]
    for _ in range(24):
        points = [nodes[int(rng.integers(len(nodes)))]]
        while len(points) <= count:
            choices = []
            p = points[-1]
            for q in nodes:
                if q in points or (q[0] != p[0] and q[1] != p[1]):
                    continue
                if any(max(abs(q[0] - r), abs(q[1] - c)) < 4 for r, c in points):
                    continue
                try:
                    if _order(points[0], points[1:] + [q]) == points + [q]:
                        choices.append(q)
                except ValueError:
                    pass
            if not choices:
                break
            points.append(choices[int(rng.integers(len(choices)))])
        if len(points) == count + 1:
            return points
    raise ValueError("could not place a unique-ray terminal chain")


def _untie(rng, walls, points, protected):
    """Remove a competing corridor cell, never choose one path as a label tie rule.

    This edits candidate INPUT walls. Returned grids still have to pass exact
    path counting. A selected path is used only to preserve connectivity during
    construction; solve() never uses this repair procedure.
    """
    for _ in range(40):
        tied = None
        for i, (a, b) in enumerate(zip(points, points[1:])):
            future = points[i + 2:]
            path, count, ds = _search(walls, a, b, future)
            _require(count > 0, "candidate has a disconnected terminal leg")
            if count == 2:
                tied = i, path, ds
                break
        if tied is None:
            return
        i, path, ds = tied
        _, _, dt = _search(walls, path[-1], path[0], points[i + 2:])
        chosen = set(path)
        alternatives = sorted(p for p in ds if p in dt and
                              ds[p] + dt[p] == len(path) - 1
                              and p not in chosen and p not in protected)
        rng.shuffle(alternatives)
        edited = False
        for p in alternatives[:12]:
            walls[p] = True
            if all(_search(walls, a, b, points[j + 2:])[1] > 0
                   for j, (a, b) in enumerate(zip(points, points[1:]))):
                edited = True
                break
            walls[p] = False
        _require(edited, "cannot untie a shortest route without damaging docks")
    raise ValueError("candidate needs too many corridor repairs")


def _clear_docks(rng, walls, points, mask):
    """Carve docking masks and repair ties, then recheck the resulting INPUT."""
    walls = walls.copy()
    # A terminal must not be an unavoidable transit cell for an earlier leg.
    # Small local aprons supply bypasses while that terminal remains unvisited.
    for r, c in points:
        walls[r - 1:r + 2, c - 1:c + 2] = False
    for _ in range(8):
        # Only the CURRENT input-derived poses are protected. Keeping obsolete
        # poses from earlier repairs can permanently lock an avoidable tie open.
        protected = set(points)
        legs = []
        for i, (a, b) in enumerate(zip(points, points[1:])):
            path, count, _ = _search(walls, a, b, points[i + 2:])
            _require(count > 0, "candidate has a disconnected terminal leg")
            legs.append(path)
        changed = False
        for p, headings in zip(points, _poses(points, legs)):
            for d in headings:
                for r, c in _footprint(mask, p, d):
                    _require(0 < r < walls.shape[0] - 1 and 0 < c < walls.shape[1] - 1,
                             "docking mask exceeds interior")
                    if walls[r, c]:
                        walls[r, c] = False
                        changed = True
                    protected.add((r, c))
        _untie(rng, walls, points, protected)
        checked = _legs(walls, points)
        if _poses(points, checked) == _poses(points, legs):
            return walls, checked
        if not changed:
            # The repair can change headings without opening any new dock cells.
            # Continue until the actual poses are clear, not just the old poses.
            continue
    raise ValueError("docking geometry did not stabilize")


def _necessity(info):
    points, walls, legs = info["points"], info["walls"], info["legs"]
    direct, _, ds = _search(walls, points[0], points[-1])
    _, _, dt = _search(walls, points[-1], points[0])
    # An actual painted-route cell outside ALL unconstrained shortest paths
    # witnesses the contribution of the intermediate ordered terminals.
    distance = len(direct) - 1
    terminal_mask = _footprint(info["mask"], points[-1], info["final"])
    witness = any(ds[p] + dt[p] > distance for p in info["route"] - terminal_mask)
    detour = any(len(path) - 1 > abs(path[-1][0] - path[0][0])
                 + abs(path[-1][1] - path[0][1]) for path in legs)
    return witness, detour


def _longer_alternative(info):
    # Bounded check: at most twelve blocked-cell BFS calls, not every walk.
    for i, path in enumerate(info["legs"]):
        indices = np.linspace(1, len(path) - 2, min(4, len(path) - 2), dtype=int)
        for index in sorted(set(int(k) for k in indices)):
            other, _, _ = _search(info["walls"], path[0], path[-1],
                                  list(info["points"][i + 2:]) + [path[index]])
            if other and len(other) > len(path):
                return True
        if i >= 2:
            break
    return False


def _make_pair(rng, palette, mask, count, test=False, teach_shortest=False,
               need_rotation=False, excluded_mask=None):
    # Geometry is a visible candidate property, not a hidden command. A fixed
    # difficult mask must not monopolize all retries for a dense itinerary.
    # Cycle through every permitted geometry, with 24 maze candidates per mask.
    others = [m for m in MASKS if m != mask and m != excluded_mask]
    rng.shuffle(others)
    choices = ([mask] if mask != excluded_mask else []) + others
    _require(bool(choices), "no permitted mover geometry")
    for attempt in range(240):
        mask = choices[(attempt // 24) % len(choices)]
        try:
            low, high = (9, 15) if test else (7, 12)
            nr, nc = (int(rng.integers(low, high)) for _ in range(2))
            walls = _maze(rng, nr, nc, int(rng.integers(2, 7)))
            points = _sample_points(rng, walls.shape, count)
            walls, legs = _clear_docks(rng, walls, points, mask)
            initial = _direction(legs[0][0], legs[0][1])
            bg, wall, ink = palette
            a = np.full(walls.shape, bg, dtype=np.int8)
            a[walls] = wall
            for p in points[1:]:
                a[p] = ink
            for p in _footprint(mask, points[0], initial):
                a[p] = ink
            # Optional asymmetric wall padding changes margins/aspect ratios.
            pads = [(int(rng.integers(0, min(2, 30 - n) + 1)), 0) for n in a.shape]
            pads = [(int(rng.integers(k + 1)), 0) if k else (0, 0) for k, _ in pads]
            a = np.pad(a, pads, constant_values=wall)
            a = np.rot90(a, int(rng.integers(4))).copy()
            if rng.random() < 0.5:
                a = a[:, ::-1].copy()
            info = _decode(a)
            if not all(_necessity(info)):
                continue
            if need_rotation and info["initial"] == info["final"]:
                continue
            if teach_shortest and not _longer_alternative(info):
                continue
            return {"input": a, "output": _render(info)}
        except ValueError:
            continue
    raise RuntimeError("bounded construction exhausted; no unverified fallback")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, numbers.Integral) and not isinstance(seed, (bool, np.bool_))
             and int(seed) >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    palette = tuple(int(c) for c in rng.choice([c for c in range(10) if c != ROUTE], 3, replace=False))
    shared_palette = bool(rng.random() < 0.35)
    masks = rng.permutation(len(MASKS)).tolist()
    train_counts = [2, 3] + [4 if i == 2 else int(rng.integers(2, 5))
                             for i in range(2, num_train)]
    plans = [("train", n, MASKS[masks[i % len(masks)]], i == 0, i == 0)
             for i, n in enumerate(train_counts)]
    plans += [("test", int(rng.integers(max(train_counts) + 1, min(6, max(train_counts) + 2) + 1)),
               MASKS[int(rng.integers(len(MASKS)))], False, False) for _ in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    first_train_mask = None
    for split, count, mask, teach_shortest, need_rotation in plans:
        if not shared_palette:
            palette = tuple(int(c) for c in rng.choice([c for c in range(10) if c != ROUTE], 3, replace=False))
        for _ in range(8):
            pair = _make_pair(rng, palette, mask, count, split == "test", teach_shortest,
                              need_rotation, first_train_mask if split == "train"
                              and len(puzzle["train"]) == 1 else None)
            key = _fingerprint(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                if split == "train" and len(puzzle["train"]) == 1:
                    first_train_mask = _decode(pair["input"])["mask"]
                break
        else:
            raise RuntimeError("could not construct distinct inputs")
    # Shuffling demonstrations removes pair-index geometry/count shortcuts.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    """Check exact semantics, both-parent witnesses, docking, and teaching."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "output must retain the complete board")
            key = _fingerprint(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            info = _decode(a)
            _require(np.array_equal(b, _render(info)), "output disagrees with the input-only rule")
            _require(np.array_equal(a[info["walls"]], b[info["walls"]]), "walls changed")
            final = _footprint(info["mask"], info["points"][-1], info["final"])
            _require(set(map(tuple, np.argwhere(b == info["ink"]))) == final,
                     "output must contain exactly the actual rigid terminal mover")
            painted = info["route"] - final
            _require(set(map(tuple, np.argwhere(b == ROUTE))) == painted, "incorrect route paint")
            _require(all(_necessity(info)), "missing parent-A or parent-B counterfactual witness")
            _require(any(a[p] == info["bg"] and b[p] == info["bg"]
                         for p in map(tuple, np.argwhere(~info["walls"])) if p not in final),
                     "need off-route background")
            analyses[split].append(info)
    train = analyses["train"]
    _require(len({i["mask"] for i in train}) >= 2, "demonstrate two actual mover geometries")
    _require(len({len(i["legs"]) for i in train}) >= 2, "demonstrate variable waypoint counts")
    _require(any(i["initial"] != i["final"] for i in train), "demonstrate a final rotation")
    _require(any(_longer_alternative(i) for i in train), "demonstrate shortest versus longer route")
    _require(min(len(i["legs"]) for i in analyses["test"]) > max(len(i["legs"]) for i in train),
             "test must extend the demonstrated number of terminals")


def _puzzle_digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _itinerary_signature(points):
    """Remove translation, rotation and reflection for a diversity measurement."""
    candidates = []
    for reflection in (-1, 1):
        for k in range(4):
            candidates.append(tuple(_rot((r - points[0][0],
                                           reflection * (c - points[0][1])), k)
                                    for r, c in points))
    return min(candidates)


def self_test():
    """Bounded 42-seed sweep; no logs, traces, or puzzles are stored on disk."""
    shapes, masks, counts, palettes, directions, signatures = (set() for _ in range(6))
    same_facing = retraces = 0
    itineraries, lengths = set(), set()
    split_headings = {s: {"initial": set(), "final": set()} for s in ("train", "test")}
    maze_determined_heading = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _decode(pair["input"])
                before = pair["input"].copy()
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                _require(np.array_equal(before, pair["input"]), "solve mutated its input")
                shapes.add(pair["input"].shape)
                masks.add(info["mask"])
                counts.add(len(info["legs"]))
                palettes.add((info["bg"], info["wall"], info["ink"]))
                directions.add(info["final"])
                split_headings[split]["initial"].add(info["initial"])
                split_headings[split]["final"].add(info["final"])
                itineraries.add(_itinerary_signature(info["points"]))
                lengths.update(len(path) - 1 for path in info["legs"])
                a, b = info["points"][-2:]
                straight = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
                maze_determined_heading += DIRS[info["final"]] != straight
                same_facing += info["initial"] == info["final"]
                walked = [p for path in info["legs"] for p in path[1:-1]]
                retraces += len(walked) > len(set(walked))
    _require(len(signatures) == 42 and len(shapes) >= 25 and len(masks) == len(MASKS),
             "insufficient structural variation")
    _require({2, 3, 4, 5, 6} <= counts and len(palettes) >= 20 and directions == set(range(4)),
             "insufficient count/palette/direction variation")
    _require(same_facing > 0 and retraces > 0, "missing same-facing or revisited-route cases")
    _require(maze_determined_heading > 0,
             "no example distinguishes local maze heading from waypoint displacement")
    _require(all(v == set(range(4)) for split in split_headings.values() for v in split.values()),
             "initial/final headings must vary in each split")
    _require(_puzzle_digest(generate_puzzle(93)) == _puzzle_digest(generate_puzzle(93)),
             "seed is not deterministic")
    # BFS must detect a tie and an unreachable target, rather than invent a tie rule.
    square = np.zeros((3, 3), dtype=bool)
    _require(_search(square, (0, 0), (2, 2))[1] == 2, "shortest-path tie undetected")
    square[1, :] = True
    _require(_search(square, (0, 0), (2, 2))[1] == 0, "unreachable target accepted")
    try:
        _order((3, 3), {(3, 7), (7, 3)})
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous waypoint order accepted")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle = generate_puzzle(42, 2, 1)
    puzzle["test"][0]["output"][0, 0] = ROUTE
    try:
        validate_generated_puzzle(puzzle)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "mover_masks": len(masks), "waypoint_counts": sorted(counts),
            "palettes": len(palettes), "same_facing_pairs": int(same_facing),
            "retraced_pairs": int(retraces), "normalized_itineraries": len(itineraries),
            "leg_lengths": sorted(lengths),
            "maze_determined_final_heading_pairs": int(maze_determined_heading),
            "split_headings": {s: {k: sorted(v) for k, v in roles.items()}
                               for s, roles in split_headings.items()}}
