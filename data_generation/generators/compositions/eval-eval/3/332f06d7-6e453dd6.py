"""Hollow stencils punch gates while following a visible route.

Only standard-library modules and NumPy are used. No parent code or examples
are imported. See task_spec.md for the pixel-level rule.
"""
from __future__ import annotations

from collections import deque
import numpy as np

WALL, FLOOR, TRACK, HEAD, GOAL, GATE = range(6)
DIRECTIONS = ((0, 1), (-1, 0), (0, -1), (1, 0))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid colors")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        queue, component = deque([p]), [p]
        while queue:
            r, c = queue.popleft()
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    queue.append(q)
                    component.append(q)
        result.append(component)
    return result


def _holes(mask):
    h, w = mask.shape
    return [cells for cells in _components(~mask)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)]


def _projections(mask, head, selector="actual"):
    """Offsets of one-cell punches, relative to the object's head.

    Counterfactual selectors are used only by validation/self-tests.
    """
    h, w = mask.shape
    enclosed = np.zeros(mask.shape, dtype=bool)
    for cells in _holes(mask):
        rr, cc = np.array(cells).T
        enclosed[rr, cc] = True
    result = {}
    for dr, dc in DIRECTIONS:
        points = []
        if selector == "none":
            result[dr, dc] = points
            continue
        if selector == "all_holes":
            indices = np.flatnonzero(enclosed.any(axis=1 if dc else 0))
        else:
            eligible = ~mask if selector == "open_too" else enclosed
            if dc == 1:
                indices = np.flatnonzero(eligible[:, w - 2] & mask[:, w - 1])
            elif dc == -1:
                indices = np.flatnonzero(eligible[:, 1] & mask[:, 0])
            elif dr == -1:
                indices = np.flatnonzero(eligible[1, :] & mask[0, :])
            else:
                indices = np.flatnonzero(eligible[h - 2, :] & mask[h - 1, :])
        for index in indices:
            r, c = ((int(index), w if dc == 1 else -1) if dc else
                    (-1 if dr == -1 else h, int(index)))
            points.append((r - head[0], c - head[1]))
        result[dr, dc] = points
    return result


def _scene(input_grid):
    inp = _grid(input_grid)
    colors = set(map(int, np.unique(inp)))
    bodies = colors & {6, 7, 8, 9}
    _require(len(bodies) == 1, "one stencil color is required")
    color = bodies.pop()
    _require(np.count_nonzero(inp == HEAD) == 1, "one head is required")
    _require(np.count_nonzero(inp == GOAL) == 1, "one target is required")
    head = tuple(map(int, np.argwhere(inp == HEAD)[0]))
    goal = tuple(map(int, np.argwhere(inp == GOAL)[0]))
    fg = (inp == color) | (inp == HEAD)
    _require(len(_components(fg)) == 1, "stencil must be connected")
    cells = np.argwhere(fg)
    lo, hi = cells.min(axis=0), cells.max(axis=0) + 1
    mask = fg[lo[0]:hi[0], lo[1]:hi[1]].copy()
    _require(min(mask.shape) >= 3, "stencil is too small")
    local_head = tuple(map(int, np.array(head) - lo))
    offsets = cells - np.array(head)
    values = inp[tuple(cells.T)]
    route_cells = {tuple(map(int, p)) for p in np.argwhere((inp == TRACK) | (inp == HEAD) | (inp == GOAL))}
    neighbors = {p: [q for dr, dc in DIRECTIONS
                     if (q := (p[0] + dr, p[1] + dc)) in route_cells]
                 for p in route_cells}
    for p, adjacent in neighbors.items():
        _require(len(adjacent) == (1 if p in (head, goal) else 2), "guide must be an unbranched path")
    path, previous, p = [head], None, head
    while p != goal:
        following = [q for q in neighbors[p] if q != previous]
        _require(len(following) == 1, "ambiguous guide")
        previous, p = p, following[0]
        _require(p not in path, "guide contains a loop")
        path.append(p)
    _require(len(path) == len(route_cells), "disconnected guide")
    # The target is a head docking mark, so another part of the stencil must
    # not hide it while the head is still on an earlier guide cell.
    for p in path[:-1]:
        _require(not np.any(np.all(offsets + np.array(p) == np.array(goal), axis=1)),
                 "target would be covered before head arrival")
    terrain = inp.copy()
    terrain[fg] = FLOOR
    terrain[head] = TRACK
    return dict(input=inp, terrain=terrain, mask=mask, local_head=local_head,
                offsets=offsets, values=values, path=path, color=color,
                box=(tuple(map(int, lo)), tuple(map(int, hi))))


def _run(input_grid, selector="actual", ignore_collision=False):
    scene = _scene(input_grid)
    terrain = scene["terrain"].copy()
    profiles = _projections(scene["mask"], scene["local_head"], selector)
    path, offsets = scene["path"], scene["offsets"]
    cuts, collisions, index = [], [], 0
    for i, (a, b) in enumerate(zip(path, path[1:])):
        direction = b[0] - a[0], b[1] - a[1]
        for dr, dc in profiles[direction]:
            r, c = a[0] + dr, a[1] + dc
            if 0 <= r < terrain.shape[0] and 0 <= c < terrain.shape[1] and terrain[r, c] == GATE:
                terrain[r, c] = FLOOR
                cuts.append((r, c))
        proposed = offsets + np.array(b)
        invalid = [(int(r), int(c)) for r, c in proposed
                   if not (0 <= r < terrain.shape[0] and 0 <= c < terrain.shape[1])
                   or terrain[r, c] not in (FLOOR, TRACK, GOAL)]
        if invalid and not ignore_collision:
            collisions = invalid
            break
        index = i + 1
    destination = path[index]
    out = terrain.copy()
    placed = offsets + np.array(destination)
    _require(bool(np.all(placed >= 0)) and bool(np.all(placed < np.array(out.shape))),
             "final stencil is outside the canvas")
    out[tuple(placed.T)] = scene["values"]
    directions = [tuple(np.subtract(b, a)) for a, b in zip(path, path[1:])]
    traversed = directions[:index]
    return out, dict(scene, destination=destination, travel=index, cuts=cuts,
                     collisions=collisions, success=index == len(path) - 1,
                     bends=sum(a != b for a, b in zip(directions, directions[1:])),
                     traversed_bends=sum(a != b for a, b in zip(traversed, traversed[1:])),
                     profiles=profiles)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire result from the input's stencil, guide and terrain."""
    return _run(input_grid)[0]


def _make_stencil(rng, tail=0, flip=False):
    slots = int(rng.integers(3, 5))
    heights = np.ones(slots, dtype=int)
    extra = int(rng.integers(0, min(4, 12 - (2 * slots + 1)) + 1))
    heights += rng.multinomial(extra, np.full(slots, 1 / slots))
    h, w = int(heights.sum() + slots + 1), int(rng.integers(6, 11))
    mask = np.ones((h, w), dtype=bool)
    kinds = ["active", "decoy"] + ([str(rng.choice(["active", "decoy", "open"]))] if slots == 4 else []) + ["open"]
    row = 1
    for kind, size in zip(kinds, heights):
        for r in range(row, row + int(size)):
            if kind == "active":
                left = int(rng.integers(1, w - 1))
                mask[r, left:w - 1] = False
            elif kind == "decoy":
                right = int(rng.integers(1, w - 2))
                mask[r, 1:right + 1] = False
            else:
                mask[r, :w - 1] = False
        row += int(size) + 1
    # Some safely eroded back edges make outlines vary without replacing the
    # topological contrasts with different category-specific dimensions.
    original_holes = {p for cells in _holes(mask) for p in cells}
    for _ in range(int(rng.integers(5))):
        r = int(rng.integers(1, h - 1))
        trial = mask.copy()
        trial[r, 0] = False
        if len(_components(trial)) == 1 and {p for cells in _holes(trial) for p in cells} == original_holes:
            mask = trial
    if tail:
        mask = np.pad(mask, ((0, 0), (0, tail)), constant_values=False)
        mask[0, w:] = True
    if flip:
        mask = np.flipud(mask).copy()
    head = (0 if mask[0, -1] else h - 1, mask.shape[1] - 1)
    return mask, head


def _expand(nodes):
    path = [nodes[0]]
    for a, b in zip(nodes, nodes[1:]):
        dr, dc = np.sign(np.subtract(b, a)).astype(int)
        p = a
        while p != b:
            p = p[0] + int(dr), p[1] + int(dc)
            path.append(p)
    return path


def _route(rng, mask, head, test=False):
    h, w = mask.shape
    nr = int(rng.integers(2, min(3, 31 // (h + 1)) + 1))
    nc = int(rng.integers(2, min(3, 31 // (w + 1)) + 1))
    walks = []
    leading = {(0, 1), (-1 if head[0] == 0 else 1, 0)}
    def grow(path):
        count = len(path)
        directions = [tuple(np.subtract(b, a)) for a, b in zip(path, path[1:])]
        bends = sum(a != b for a, b in zip(directions, directions[1:]))
        if (test and count >= 4 and bends >= 2) or (not test and count in (2, 3) and directions[-1] in leading):
            walks.append(path[:])
        limit = nr * nc if test else 3
        if count == limit:
            return
        r, c = path[-1]
        for dr, dc in DIRECTIONS:
            q = r + dr, c + dc
            if 0 <= q[0] < nr and 0 <= q[1] < nc and q not in path:
                grow(path + [q])
    for r in range(nr):
        for c in range(nc - 1):
            grow([(r, c), (r, c + 1)])
    _require(bool(walks), "no route fits")
    walk = walks[int(rng.integers(len(walks)))]
    def coordinates(n, size):
        # Reserve a cell for a final head-leading docking step if needed.
        spare = 29 - (size + (n - 1) * (size + 1))
        total = int(rng.integers(spare + 1))
        gaps = size + 1 + rng.multinomial(total, np.full(n - 1, 1 / (n - 1)))
        return np.concatenate(([0], np.cumsum(gaps)))
    rr, cc = coordinates(nr, h), coordinates(nc, w)
    nodes = [(int(rr[r]) + head[0], int(cc[c]) + head[1]) for r, c in walk]
    path = _expand(nodes)
    last = tuple(np.subtract(path[-1], path[-2]))
    if test and last not in leading:
        reverse = -last[0], -last[1]
        d = next(d for d in DIRECTIONS if d in leading and d != reverse)
        path.append((path[-1][0] + d[0], path[-1][1] + d[1]))
    return path


def _draw(rng, mask, head, path, color):
    h, w = mask.shape
    lower = np.min(path, axis=0) - np.array(head)
    upper = np.max(path, axis=0) - np.array(head) + np.array((h, w))
    grid = np.zeros(tuple(upper - lower), dtype=np.int8)
    path = [tuple(map(int, np.array(p) - lower)) for p in path]
    for r, c in path:
        r, c = r - head[0], c - head[1]
        grid[r:r + h, c:c + w] = FLOOR
    for p in path:
        grid[p] = TRACK
    grid[path[-1]] = GOAL
    r0, c0 = path[0][0] - head[0], path[0][1] - head[1]
    view = grid[r0:r0 + h, c0:c0 + w]
    _require(sum(grid[r0 + r, c0 + c] in (TRACK, GOAL) for r, c in np.argwhere(mask)) == 1,
             "initial stencil obscures guide")
    view[mask] = color
    grid[path[0]] = HEAD
    padding = []
    for size in grid.shape:
        extra = int(rng.integers(31 - size))
        before = int(rng.integers(extra + 1))
        padding.append((before, extra - before))
    grid = np.pad(grid, padding, constant_values=WALL)
    path = [(r + padding[0][0], c + padding[1][0]) for r, c in path]
    return grid, path


def _put(grid, point, value=GATE):
    r, c = point
    if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] and grid[r, c] == FLOOR:
        grid[r, c] = value
        return True
    return False


def _finish(rng, grid):
    # Rotate/reflect all semantic marks together; solve infers offsets and
    # headings directly, without reversing this construction transform.
    if rng.integers(2):
        grid = np.flipud(grid)
    grid = np.rot90(grid, int(rng.integers(4))).copy()
    return {"input": grid, "output": solve(grid)}


def _blocked_contrast(rng, color):
    mask, head = _make_stencil(rng)
    path = _route(rng, mask, head)
    grid, path = _draw(rng, mask, head, path, color)
    east_length = next((i for i, p in enumerate(path) if p[0] != path[0][0]), len(path)) - 1
    t = int(rng.integers(2, east_length - 1))
    actual = _projections(mask, head)[0, 1]
    naive = _projections(mask, head, "open_too")[0, 1]
    all_holes = _projections(mask, head, "all_holes")[0, 1]
    opened = sorted(set(naive) - set(actual))
    recessed = sorted(set(all_holes) - set(actual))
    _require(actual and opened and recessed, "missing stencil contrasts")
    for pool in (actual, opened, recessed):
        dr, dc = pool[int(rng.integers(len(pool)))]
        _require(_put(grid, (path[t][0] + dr, path[t][1] + dc)), "gate did not fit")
    # An earlier passable gate makes first-obstacle/first-narrowing shortcuts
    # fail independently of the mixed blocking gate.
    dr, dc = actual[int(rng.integers(len(actual)))]
    _put(grid, (path[1][0] + dr, path[1][1] + dc))
    pair = _finish(rng, grid)
    _, info = _run(pair["input"])
    _require(not info["success"] and info["travel"] > 0 and len(info["cuts"]) >= 2,
             "blocked demonstration failed")
    return pair


def _turning_contrast(rng, color):
    mask, head = _make_stencil(rng, tail=int(rng.integers(1, 3)))
    h, w = mask.shape
    east = int(rng.integers(w + 1, 31 - w))
    north = int(rng.integers(h + 1, min(h + 5, 30 - h) + 1))
    start = head[0] + north, head[1]
    corner = start[0], start[1] + east
    path = _expand([start, corner, (corner[0] - north, corner[1])])
    grid, path = _draw(rng, mask, head, path, color)
    # This thin-walled enclosed cavity is recessed by the remote tail. Its
    # gate remains intact as the stencil turns away before its body hits it.
    east_decoys = _projections(mask, head, "all_holes")[0, 1]
    dr, dc = east_decoys[int(rng.integers(len(east_decoys)))]
    # Choose the first (top) closed compartment; those cells lie below the
    # head, so the subsequent northward tail never sweeps through this post.
    dr = min(p[0] for p in east_decoys)
    _require(_put(grid, (path[east - 1][0] + dr, path[east - 1][1] + dc)), "tail decoy did not fit")
    north_punches = _projections(mask, head)[-1, 0]
    _require(bool(north_punches), "north face must have an active cavity")
    t = east + int(rng.integers(1, north - 1))
    chosen = rng.choice(len(north_punches), size=int(rng.integers(1, len(north_punches) + 1)), replace=False)
    for j in chosen:
        dr, dc = north_punches[int(j)]
        _require(_put(grid, (path[t][0] + dr, path[t][1] + dc)), "turn gate did not fit")
    pair = _finish(rng, grid)
    _, info = _run(pair["input"])
    _require(info["success"] and info["cuts"] and np.any(pair["output"] == GATE),
             "turning demonstration failed")
    return pair


def _general_pair(rng, color, test):
    tail = int(rng.choice([0, 0, 0, 1, 2]))
    mask, head = _make_stencil(rng, tail, flip=bool(rng.integers(2)))
    path = _route(rng, mask, head, test)
    grid, path = _draw(rng, mask, head, path, color)
    scene = _scene(grid)
    lo, hi = scene["box"]
    profiles = _projections(mask, head)
    wanted_stop = bool(rng.integers(2))
    steps = list(range(1, len(path) - 1))
    rng.shuffle(steps)
    for t in steps[:int(rng.integers(1, 7))]:
        a, b = path[t], path[t + 1]
        d = b[0] - a[0], b[1] - a[1]
        pool = profiles[d]
        if not pool:
            continue
        dr, dc = pool[int(rng.integers(len(pool)))]
        for depth in range(int(rng.integers(1, 4))):
            point = a[0] + dr + depth * d[0], a[1] + dc + depth * d[1]
            if not (lo[0] <= point[0] < hi[0] and lo[1] <= point[1] < hi[1]):
                _put(grid, point)
    # First ensure a fully traversable baseline. Side encounters with a gate
    # intended for a later face are legitimate collisions, but must not erase
    # the intended behavioral test shift.
    if not _run(grid)[1]["success"]:
        grid[grid == GATE] = FLOOR
        if profiles[0, 1]:
            dr, dc = profiles[0, 1][int(rng.integers(len(profiles[0, 1])))]
            _put(grid, (path[1][0] + dr, path[1][1] + dc))
    if test and not _run(grid)[1]["cuts"]:
        # Every query should exercise the coupled rule. Find a gate that is
        # safely punched during the real route rather than leave a wholly
        # unobstructed long guide as the only test.
        candidates = []
        for t, (a, b) in enumerate(zip(path, path[1:])):
            d = b[0] - a[0], b[1] - a[1]
            for dr, dc in profiles[d]:
                point = a[0] + dr, a[1] + dc
                if not (lo[0] <= point[0] < hi[0] and lo[1] <= point[1] < hi[1]):
                    candidates.append(point)
        rng.shuffle(candidates)
        found = False
        for point in candidates[:128]:
            if not _put(grid, point):
                continue
            info = _run(grid)[1]
            if info["success"] and point in info["cuts"]:
                found = True
                break
            grid[point] = FLOOR
        if not found:
            return None
    if wanted_stop and test:
        directions = [tuple(np.subtract(b, a)) for a, b in zip(path, path[1:])]
        late = [t for t in range(1, len(path) - 1)
                if sum(a != b for a, b in zip(directions[:t], directions[1:t])) >= 2]
        rng.shuffle(late)
        added = False
        value = WALL if rng.random() < .35 else GATE
        for t in late:
            d = directions[t]
            frontier = []
            for r, c in np.argwhere(mask):
                q = int(r) + d[0], int(c) + d[1]
                if not (0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1] and mask[q]):
                    point = path[t][0] + q[0] - head[0], path[t][1] + q[1] - head[1]
                    if not (lo[0] <= point[0] < hi[0] and lo[1] <= point[1] < hi[1]):
                        frontier.append(point)
            rng.shuffle(frontier)
            for point in frontier:
                if not _put(grid, point, value):
                    continue
                info = _run(grid)[1]
                if not info["success"] and info["traversed_bends"] >= 2:
                    added = True
                    break
                grid[point] = FLOOR
            if added:
                break
        # If every late candidate is punched or encountered too early, retain
        # the clear-route sample. The semantic rule never acquires an exception.
    elif wanted_stop:
        east_length = next((i for i, p in enumerate(path) if p[0] != path[0][0]), len(path)) - 1
        t = int(rng.integers(1, max(2, east_length - tail - 1)))
        active_rows = {p[0] + head[0] for p in profiles[0, 1]}
        candidates = [r for r in range(mask.shape[0]) if r != head[0] and r not in active_rows and mask[r].any()]
        r = candidates[int(rng.integers(len(candidates)))]
        _put(grid, (path[t][0] + r - head[0], path[t][1] + mask.shape[1] - head[1]),
             WALL if rng.random() < .35 else GATE)
    if test and not _run(grid)[1]["cuts"]:
        return None
    return _finish(rng, grid)


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, "num_train")
    num_test = _count(num_test, 1, 4, "num_test")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    base_color = int(rng.integers(6, 10))
    variable_color = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            for _ in range(64):
                color = int(rng.integers(6, 10)) if variable_color else base_color
                if split == "train" and i == 0:
                    pair = _blocked_contrast(rng, color)
                elif split == "train" and i == 1:
                    pair = _turning_contrast(rng, color)
                else:
                    pair = _general_pair(rng, color, split == "test")
                if pair is None:
                    continue
                key = _signature(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_signature(puzzle):
    return tuple((_signature(p["input"]), _signature(p["output"]))
                 for split in ("train", "test") for p in puzzle[split])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "bad pair counts")
    seen, training, omissions = set(), [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = _signature(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            expected, info = _run(inp)
            _require(inp.shape == out.shape and np.array_equal(expected, out), "incorrect transformation")
            # These invariants inspect labels in addition to exact prediction.
            _require(np.array_equal(inp == WALL, out == WALL), "permanent terrain changed")
            _require(np.count_nonzero(out == HEAD) == 1, "head was lost or duplicated")
            _require(np.count_nonzero(inp == info["color"]) == np.count_nonzero(out == info["color"]),
                     "stencil mass changed")
            output_cells = np.argwhere((out == info["color"]) | (out == HEAD))
            relative = set(map(tuple, output_cells - np.array(info["destination"])))
            _require(relative == set(map(tuple, info["offsets"])), "stencil deformed")
            _require((not np.any(out == GOAL)) == info["success"], "target replacement is wrong")
            _require(len(info["cuts"]) == len(set(info["cuts"])), "a gate was cut twice")
            if not info["success"]:
                _require(bool(info["collisions"]), "stopped without a collision")
            if split == "train":
                training.append(info)
                _require(info["bends"] <= 1, "training shift schedule violated")
                for name, selector, ignore in (("B", "none", False), ("enclosure", "open_too", False),
                                              ("leading", "all_holes", False), ("A", "actual", True)):
                    other = _run(inp, selector, ignore)[0]
                    if not np.array_equal(other, out):
                        omissions.add(name)
            else:
                _require(info["traversed_bends"] >= 2, "test must traverse several bends")
                _require(info["cuts"] and not np.array_equal(_run(inp, "none")[0], out),
                         "test does not exercise cavity punching")
    _require(any(i["success"] and i["cuts"] for i in training), "no cutting success demonstrated")
    _require(any(not i["success"] and i["travel"] > 0 and i["cuts"] for i in training),
             "no moving partial-cut stop demonstrated")
    _require(omissions == {"A", "B", "enclosure", "leading"}, "training allows a parent-omitting shortcut")


def self_test():
    """Bounded 42-seed sampling, geometric regressions and mutation checks."""
    # Small, explicitly expected regressions do not use the random builders
    # or solve to construct their labels. The tail case also distinguishes
    # occupied-pixel collision from bounding-box collision.
    ring = np.ones((3, 3), dtype=bool)
    ring[1, 1] = False
    opened = ring.copy()
    opened[1, 0] = False
    tailed = np.pad(ring, ((0, 0), (0, 1)), constant_values=False)
    tailed[0, 3] = True
    for mask, head, stop, expected_cuts in (
            (ring, (0, 2), None, {(5, 4), (5, 5), (5, 6), (2, 6)}),
            (opened, (0, 2), (4, 3), set()),
            (tailed, (0, 3), None, {(2, 5)})):
        start, corner, goal = (4, head[1]), (4, 7), (1, 7)
        path = _expand([start, corner, goal])
        inp = np.full((8, 10), FLOOR, dtype=np.int8)
        for p in path:
            inp[p] = TRACK
        inp[goal] = GOAL
        view = inp[4:7, :mask.shape[1]]
        view[mask] = 6
        inp[start] = HEAD
        gates = {(5, 7), (2, 5)} if mask.shape[1] == 4 else {(5, 4), (5, 5), (5, 6), (2, 6)}
        for p in gates:
            inp[p] = GATE
        expected = inp.copy()
        expected[(inp == 6) | (inp == HEAD)] = FLOOR
        expected[start] = TRACK
        for p in expected_cuts:
            expected[p] = FLOOR
        destination = stop or goal
        r, c = destination[0] - head[0], destination[1] - head[1]
        expected[r:r + mask.shape[0], c:c + mask.shape[1]][mask] = 6
        expected[destination] = HEAD
        predicted, info = _run(inp)
        _require(np.array_equal(predicted, expected) and set(info["cuts"]) == expected_cuts,
                 "explicit closed/open/tail regression failed")
    # A visible guide can continue where the head fits but its body would
    # extend past the canvas. It must stop before that step, not crop itself.
    path = _expand([(1, 2), (1, 6), (6, 6), (6, 7)])
    inp = np.full((7, 10), FLOOR, dtype=np.int8)
    for p in path:
        inp[p] = TRACK
    inp[path[-1]] = GOAL
    inp[1:4, :3][ring] = 6
    inp[path[0]] = HEAD
    expected = inp.copy()
    expected[(inp == 6) | (inp == HEAD)] = FLOOR
    expected[path[0]] = TRACK
    expected[4:7, 4:7][ring] = 6
    expected[4, 6] = HEAD
    _require(np.array_equal(solve(inp), expected), "canvas-boundary stopping failed")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, structures, masks = [], set(), set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                out, info = _run(pair["input"])
                _require(np.array_equal(out, pair["output"]), "input-only prediction failed")
                normalized = pair["input"].copy()
                normalized[normalized >= 6] = 6
                structures.add(_signature(normalized))
                masks.add(_signature(info["mask"]))
                records.append((split, info))
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, "puzzle diversity collapsed")
    _require(len(structures) >= 150 and len(masks) >= 60, "geometry diversity collapsed")
    _require(len({i["input"].shape for _, i in records}) >= 40, "canvas shapes collapsed")
    _require({i["color"] for _, i in records} == {6, 7, 8, 9}, "body palette collapsed")
    _require({i["success"] for s, i in records if s == "test"} == {False, True}, "test outcomes collapsed")
    _require(max(i["bends"] for _, i in records) >= 3, "longer routes missing")
    _require(max(len(_holes(i["mask"])) for _, i in records) >= 3, "multiple cavities missing")
    _require(any(not i["success"] and any(i["terrain"][p] == WALL for p in i["collisions"])
                 for _, i in records), "permanent-wall stopping missing")
    # Equivariance and no mutation of an input are checked on both essential
    # demonstrations, including a projecting stop and a tail passing a peg.
    for pair in puzzles[0]["train"]:
        saved = pair["input"].copy()
        for turn in range(4):
            for reflect in (False, True):
                inp = np.rot90(saved, turn)
                expected = np.rot90(pair["output"], turn)
                if reflect:
                    inp, expected = np.fliplr(inp), np.fliplr(expected)
                _require(np.array_equal(solve(inp), expected), "orientation equivariance failed")
        _require(np.array_equal(pair["input"], saved), "solve mutated input")
    repeated = generate_puzzle(981, 10, 4)
    _require(_puzzle_signature(repeated) == _puzzle_signature(generate_puzzle(981, 10, 4)), "nondeterministic")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(12345, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    for mutation in ("label", "guide"):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        if mutation == "label":
            out = bad["test"][0]["output"]
            p = tuple(np.argwhere(out == HEAD)[0])
            out[p] = FLOOR
        else:
            inp = bad["test"][0]["input"]
            p = tuple(np.argwhere(inp == TRACK)[len(np.argwhere(inp == TRACK)) // 2])
            inp[p] = FLOOR
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted corruption")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
