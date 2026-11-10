"""Rigid rectangular brush routing from repeated colored registrations.

Only Python's standard library and NumPy are used. Coordinates in the route
search are top-left positions of the entire, untranslated-orientation brush.
"""

from collections import deque
import json

import numpy as np


BACKGROUND = 0
BARRIER = 8
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (
    0, 910854079, 92410550, 18086748, 534777501, 326845564, 611278470,
    205926725, 953190245, 667038287, 156489733, 965139485, 106129038,
    542440216, 223713227, 883303688, 351519758, 764196716, 223287089,
    549885518, 727116041, 696811354, 823775113, 182494103, 735288605,
    77634574, 460097717, 316044697, 61450007, 988023645, 655980398,
    588618396, 695464555, 164185804, 931552521, 282954314, 994765957,
    285104397, 164223462, 789305137, 702619472, 577356519,
)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return a.astype(int, copy=True)


def _components(mask):
    seen = set()
    result = []
    for rr, cc in np.argwhere(mask):
        start = (int(rr), int(cc))
        if start in seen:
            continue
        seen.add(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRS:
                p = (r + dr, c + dc)
                if (0 <= p[0] < mask.shape[0] and 0 <= p[1] < mask.shape[1]
                        and mask[p] and p not in seen):
                    seen.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _separated(a, b):
    """The two snapshot rectangles have at least a one-cell gap."""
    r, c, h, w = a
    y, x, j, k = b
    return r + h < y or y + j < r or c + w < x or x + k < c


def _pose_map(obstacles, h, w):
    """All rectangle positions whose complete footprints are barrier-free."""
    height, width = obstacles.shape
    if h > height or w > width:
        raise ValueError("brush larger than board")
    # Integral image: no dependency on a sampled route or generation metadata.
    sums = np.pad(obstacles.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return counts == 0


def _route(allowed, start, goals):
    """Return ((moves, bends), unique path or None, number of optima capped at 2).

    First compute exact remaining travel distances from every position and
    unvisited-goal subset. Only moves reducing that distance by one can occur
    on a shortest route. A layered dynamic program on those moves minimizes
    bends and counts all optimal routes, retaining the previous direction.
    Retracing remains legal, and no waypoint order or bend limit is assumed.
    """
    goals = tuple(goals)
    if len(set(goals)) != len(goals) or start in goals or not goals:
        raise ValueError("distinct destination poses required")
    height, width = allowed.shape
    if any(not (0 <= r < height and 0 <= c < width and allowed[r, c])
           for r, c in (start,) + goals):
        return None, None, 0
    size = height * width
    free = allowed.ravel().tolist()
    adjacent = [[] for _ in range(size)]
    for point in range(size):
        if not free[point]:
            continue
        r, c = divmod(point, width)
        for direction, (dr, dc) in enumerate(DIRS):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                other = nr * width + nc
                if free[other]:
                    adjacent[point].append((other, direction))
    start_point = start[0] * width + start[1]
    goal_points = [r * width + c for r, c in goals]
    infinity = 1000000
    distances = []
    for goal in goal_points:
        distance = [infinity] * size
        distance[goal] = 0
        queue = deque([goal])
        while queue:
            point = queue.popleft()
            next_distance = distance[point] + 1
            for other, _ in adjacent[point]:
                if distance[other] == infinity:
                    distance[other] = next_distance
                    queue.append(other)
        if distance[start_point] == infinity:
            return None, None, 0
        distances.append(distance)

    # Metric-closure subset DP gives exact remaining walk length: minimize
    # over all orders of the goals, with graph-shortest travel between them.
    # A segment may pass another goal; ordering its actual first visits shows
    # that this still equals the optimum in the original position graph.
    count = len(goals)
    complete = (1 << count) - 1
    tails = [[0] * (complete + 1) for _ in goals]
    for mask in range(1, complete + 1):
        for i in range(count):
            if mask & (1 << i):
                tails[i][mask] = tails[i][mask ^ (1 << i)]
            else:
                tails[i][mask] = min(
                    distances[i][goal_points[j]] + tails[j][mask ^ (1 << j)]
                    for j in range(count) if mask & (1 << j)
                )
    arrays = np.asarray(distances, dtype=int)
    remaining = np.zeros((complete + 1, size), dtype=int)
    for mask in range(1, complete + 1):
        remaining[mask] = infinity
        for i in range(count):
            if mask & (1 << i):
                np.minimum(remaining[mask], arrays[i] + tails[i][mask ^ (1 << i)],
                           out=remaining[mask])
    remaining = remaining.tolist()
    moves = remaining[complete][start_point]
    goal_bits = [0] * size
    for i, point in enumerate(goal_points):
        goal_bits[point] = 1 << i

    initial = (start_point, complete, 4)
    frontier = {initial: (0, 1)}  # state -> (fewest bends, optimal route count)
    previous = {initial: None}
    for steps_left in range(moves, 0, -1):
        following = {}
        for state, (bends, ways) in frontier.items():
            point, mask, direction = state
            for other, nd in adjacent[point]:
                new_mask = mask & ~goal_bits[other]
                if remaining[new_mask][other] != steps_left - 1:
                    continue
                new = (other, new_mask, nd)
                new_bends = bends + int(direction != 4 and direction != nd)
                old = following.get(new)
                if old is None or new_bends < old[0]:
                    following[new] = (new_bends, ways)
                    previous[new] = state
                elif new_bends == old[0]:
                    following[new] = (new_bends, min(2, old[1] + ways))
        assert following, "shortest-route distance must admit a next step"
        frontier = following
    best_bends = min(bends for bends, _ in frontier.values())
    total, end = 0, None
    for state, (bends, ways) in frontier.items():
        if bends == best_bends:
            total = min(2, total + ways)
            end = state
    best = (moves, best_bends)
    if total != 1:
        return best, None, total
    path = []
    while end is not None:
        path.append(divmod(end[0], width))
        end = previous[end]
    return best, tuple(reversed(path)), total


def _registrations(grid, source, markers, marker_colors):
    r0, c0, h, w = source
    keys = {(int(r), int(c)): int(grid[r, c]) for r, c in
            np.argwhere(np.isin(grid, tuple(marker_colors)))
            if not (r0 <= r < r0 + h and c0 <= c < c0 + w)}
    if not keys:
        raise ValueError("no destination registrations")
    mr, mc, color = markers[0]
    candidates = []
    for (r, c), value in sorted(keys.items()):
        top, left = r - mr, c - mc
        box = (top, left, h, w)
        if (value != color or top < 0 or left < 0
                or top + h > grid.shape[0] or left + w > grid.shape[1]
                or not _separated(box, source)):
            continue
        expected = {(top + y, left + x): v for y, x, v in markers}
        if any(keys.get(p) != v for p, v in expected.items()):
            continue
        inside = {p for p in keys if top <= p[0] < top + h and left <= p[1] < left + w}
        if inside != set(expected) or np.any(grid[top:top + h, left:left + w] == BARRIER):
            continue
        candidates.append((box, frozenset(expected)))
    by_key = {p: [candidate for candidate in candidates if p in candidate[1]] for p in keys}
    solutions = []

    def cover(used, chosen):
        if len(solutions) >= 2:
            return
        if len(used) == len(keys):
            solutions.append(tuple(sorted(box[:2] for box, _ in chosen)))
            return
        if len(chosen) == 3:
            return
        pivot = min(set(keys) - used, key=lambda p: (len(by_key[p]), p))
        for box, cells in by_key[pivot]:
            if cells & used or any(not _separated(box, old) for old, _ in chosen):
                continue
            cover(used | cells, chosen + [(box, cells)])

    cover(frozenset(), [])
    if len(solutions) != 1:
        raise ValueError("registration cover is missing or ambiguous")
    return solutions[0]


def _parse(grid):
    """Identify the complete source from geometry and locally visible roles."""
    answers = []
    for cells in _components((grid != BACKGROUND) & (grid != BARRIER)):
        r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
        h = max(r for r, _ in cells) - r0 + 1
        w = max(c for _, c in cells) - c0 + 1
        if h < 2 or w < 2 or h * w != len(cells):
            continue
        matrix = grid[r0:r0 + h, c0:c0 + w]
        colors, counts = np.unique(matrix, return_counts=True)
        body = int(colors[int(np.argmax(counts))])
        if int(counts.max()) <= matrix.size // 2 or np.count_nonzero(grid == body) != counts.max():
            continue
        markers = tuple((int(r), int(c), int(matrix[r, c])) for r, c in np.argwhere(matrix != body))
        marker_colors = {v for _, _, v in markers}
        if not (1 <= len(markers) <= 4 and 1 <= len(marker_colors) <= 2):
            continue
        if not set(map(int, np.unique(grid))) <= {BACKGROUND, BARRIER, body} | marker_colors:
            continue
        source = (r0, c0, h, w)
        try:
            goals = _registrations(grid, source, markers, marker_colors)
        except ValueError:
            continue
        answers.append({"source": source, "body": body, "markers": markers, "goals": goals})
    if len(answers) != 1:
        raise ValueError("one unambiguous filled brush is required")
    return answers[0]


def _paint(grid, info, path):
    output = grid.copy()
    h, w = info["source"][2:]
    for r, c in path:
        output[r:r + h, c:c + w] = info["body"]
    # Registration pins and source contents are permanent, not moving ink.
    output[grid != BACKGROUND] = grid[grid != BACKGROUND]
    return output


def _analyze(input_grid):
    grid = _grid(input_grid)
    info = _parse(grid)
    r, c, h, w = info["source"]
    cost, path, ways = _route(_pose_map(grid == BARRIER, h, w), (r, c), info["goals"])
    if ways != 1:
        raise ValueError("route must have a unique length-then-bend optimum")
    info.update(cost=cost, path=path)
    return grid, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all roles and the optimal footprint route from this input only."""
    grid, info = _analyze(input_grid)
    return _paint(grid, info, info["path"])


def _count(name, value, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _random_obstacles(rng, shape):
    mask = np.zeros(shape, dtype=bool)
    for _ in range(int(rng.integers(2, 8))):
        kind = int(rng.integers(4))
        if kind == 0:
            h, w = 1, int(rng.integers(1, 10))
        elif kind == 1:
            h, w = int(rng.integers(1, 10)), 1
        else:
            h, w = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        r = int(rng.integers(0, shape[0] - h + 1))
        c = int(rng.integers(0, shape[1] - w + 1))
        mask[r:r + h, c:c + w] = True
        if kind == 3:
            # Remove a corner to include irregular, connected L-shaped walls.
            mask[r:r + h - 1, c:c + w - 1] = False
    return mask


def _point_cost(obstacles, h, w, start, goals):
    # Same reference-coordinate bounds, but only the reference pixel collides.
    allowed = ~obstacles[:obstacles.shape[0] - h + 1, :obstacles.shape[1] - w + 1]
    return _route(allowed, start, goals)[0]


def _first_visits(path, goals):
    pending, order = set(goals), []
    for p in path:
        if p in pending:
            pending.remove(p)
            order.append(p)
    return tuple(order)


def _unique_nearest(allowed, start, goals):
    """Nearest destination in actual legal-position graph distance, without bends."""
    distances = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            p = (r + dr, c + dc)
            if (0 <= p[0] < allowed.shape[0] and 0 <= p[1] < allowed.shape[1]
                    and allowed[p] and p not in distances):
                distances[p] = distances[r, c] + 1
                queue.append(p)
    ranked = sorted((distances.get(p, 100000), p) for p in goals)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def _build_pair(rng, destinations, clearance_witness=False, ordering_witness=False,
                marker_mode=None, brush_regime="mixed"):
    for attempt in range(1800):
        height, width = map(int, rng.integers(16, 26, size=2))
        h, w = map(int, rng.integers(2, 6, size=2))
        if brush_regime == "elongated" or (brush_regime == "mixed" and rng.random() < 0.23):
            h, w = 2, int(rng.integers(6 if brush_regime == "elongated" else 3, 8))
            if rng.random() < 0.5:
                h, w = w, h
        obstacles = _random_obstacles(rng, (height, width))
        allowed = _pose_map(obstacles, h, w)
        positions = [tuple(map(int, p)) for p in np.argwhere(allowed)]
        rng.shuffle(positions)
        poses = []
        for p in positions:
            if all(_separated((*p, h, w), (*q, h, w)) for q in poses):
                poses.append(p)
                if len(poses) == destinations + 1:
                    break
        if len(poses) != destinations + 1:
            continue
        start, goals = poses[0], tuple(sorted(poses[1:]))
        cost, path, ways = _route(allowed, start, goals)
        if ways != 1 or cost[0] < 5:
            continue
        if clearance_witness:
            point = _point_cost(obstacles, h, w, start, goals)
            if point is None or cost[0] <= point[0]:
                continue
        if ordering_witness:
            nearest = _unique_nearest(allowed, start, goals)
            if nearest is None or _first_visits(path, goals)[0] == nearest:
                continue
        # A substantial trail must occur outside the observed poses.
        covered = np.zeros((height, width), dtype=bool)
        registered = covered.copy()
        for r, c in path:
            covered[r:r + h, c:c + w] = True
        for r, c in poses:
            registered[r:r + h, c:c + w] = True
        if np.count_nonzero(covered & ~registered) < 6:
            continue
        palette = [int(x) for x in rng.choice([1, 2, 3, 4, 5, 6, 7, 9], 3, replace=False)]
        body, loc_a, loc_b = palette
        if marker_mode == 0:
            marker_count, color_count = 2, 1
        elif marker_mode == 1:
            marker_count, color_count = 3, 2
        else:
            marker_count = int(rng.integers(1, min(4, (h * w - 1) // 2) + 1))
            color_count = int(rng.integers(1, min(2, marker_count) + 1))
        if 2 * marker_count >= h * w:
            continue
        marker_indices = rng.choice(h * w, size=marker_count, replace=False)
        colors = [loc_a] * marker_count
        if color_count == 2:
            colors[int(rng.integers(marker_count))] = loc_b
        markers = tuple(sorted((int(i) // w, int(i) % w, color)
                               for i, color in zip(marker_indices, colors)))
        grid = np.where(obstacles, BARRIER, BACKGROUND).astype(int)
        r0, c0 = start
        grid[r0:r0 + h, c0:c0 + w] = body
        for r, c in poses:
            for mr, mc, color in markers:
                grid[r + mr, c + mc] = color
        try:
            parsed = _parse(grid)
        except ValueError:
            continue
        if parsed["source"] != (*start, h, w) or parsed["goals"] != goals:
            continue
        return {"input": grid, "output": _paint(grid, parsed, path)}
    raise RuntimeError("bounded search could not construct a unique brush route")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if split == "train":
                destinations = (1, 3)[index] if index < 2 else int(rng.integers(1, 4))
            else:
                destinations = 3 if index == 0 else int(rng.integers(1, 4))
            for _ in range(30):
                pair = _build_pair(rng, destinations,
                                   clearance_witness=split == "train" and index == 0,
                                   ordering_witness=split == "train" and index == 1,
                                   marker_mode=index if split == "train" and index < 2 else None,
                                   brush_regime="ordinary" if split == "train" else
                                   "elongated" if index == 0 else "mixed")
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4 and len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, info = _analyze(pair["input"])
            output = _grid(pair["output"])
            assert grid.shape == output.shape
            key = (grid.shape, grid.tobytes())
            assert key not in seen
            seen.add(key)
            assert np.array_equal(output[grid != 0], grid[grid != 0])
            assert np.all((output == grid) | ((grid == 0) & (output == info["body"])))
            expected = _paint(grid, info, info["path"])
            assert np.array_equal(expected, output)
            r, c, h, w = info["source"]
            assert all(not np.any(grid[y:y + h, x:x + w] == BARRIER) for y, x in info["path"])
            snapshots = _paint(grid, info, ((r, c),) + info["goals"])
            assert np.count_nonzero(output != snapshots) >= 6
            assert info["cost"][0] == len(info["path"]) - 1
            infos[split].append(info)
    assert len(infos["train"][0]["goals"]) == 1
    assert len(infos["train"][1]["goals"]) == 3
    assert all(2 <= min(info["source"][2:]) <= max(info["source"][2:]) <= 5
               for info in infos["train"])
    assert len(infos["test"][0]["goals"]) == 3
    assert sorted(infos["test"][0]["source"][2:])[0] == 2
    assert sorted(infos["test"][0]["source"][2:])[1] in (6, 7)
    first = infos["train"][0]
    grid = np.asarray(puzzle["train"][0]["input"])
    r, c, h, w = first["source"]
    point = _point_cost(grid == BARRIER, h, w, (r, c), first["goals"])
    assert point is not None and point[0] < first["cost"][0]
    second = infos["train"][1]
    grid = np.asarray(puzzle["train"][1]["input"])
    r, c, h, w = second["source"]
    nearest = _unique_nearest(_pose_map(grid == BARRIER, h, w), (r, c), second["goals"])
    assert nearest is not None and _first_visits(second["path"], second["goals"])[0] != nearest


def _puzzle_key(puzzle):
    return tuple((np.asarray(p[field]).shape, np.asarray(p[field]).tobytes())
                 for s in ("train", "test") for p in puzzle[s] for field in ("input", "output"))


def self_test():
    """Bounded 42-seed variation audit, deterministic/count and ablation checks."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    shapes, brushes, signatures, palettes, bends, goals, marker_counts = (set() for _ in range(7))
    edge = irregular = offset = retrace = reading_order = nearest_first = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, info = _analyze(pair["input"])
                r, c, h, w = info["source"]
                shapes.add(grid.shape)
                brushes.add((h, w))
                signatures.add((h, w, info["markers"]))
                palettes.add((info["body"], tuple(sorted(v for _, _, v in info["markers"]))))
                bends.add(info["cost"][1])
                goals.add(len(info["goals"]))
                marker_counts.add(len(info["markers"]))
                edge += int(any(y == 0 or x == 0 or y + h == grid.shape[0] or x + w == grid.shape[1]
                                for y, x in ((r, c),) + info["goals"]))
                irregular += int(any(len({y for y, _ in part}) > 1 and len({x for _, x in part}) > 1
                                     for part in _components(grid == BARRIER)))
                offset += int(all(y != 0 or x != 0 for y, x, _ in info["markers"]))
                retrace += int(len(info["path"]) != len(set(info["path"])))
                visits = _first_visits(info["path"], info["goals"])
                reading_order += int(visits != tuple(sorted(info["goals"])))
                nearest = _unique_nearest(_pose_map(grid == BARRIER, h, w), (r, c), info["goals"])
                nearest_first += int(nearest is not None and visits[0] != nearest)
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(shapes) >= 30 and len(brushes) >= 12 and len(signatures) >= 120
    assert len(palettes) >= 60 and len(bends) >= 5
    assert goals == {1, 2, 3} and marker_counts == {1, 2, 3, 4}
    assert edge >= 10 and irregular >= 80 and offset >= 80
    assert reading_order >= 20 and nearest_first >= 42
    assert _puzzle_key(generate_puzzle(910854079)) == _puzzle_key(puzzles[1])
    # Known tiny routing cases exercise uniqueness and bend semantics directly.
    open_grid = np.ones((3, 5), dtype=bool)
    assert _route(open_grid, (0, 0), ((0, 4),))[:1] == ((4, 0),)
    assert _route(open_grid, (0, 0), ((2, 4),))[2] == 2
    open_grid[0, 1] = False
    cost, path, count = _route(open_grid, (0, 0), ((2, 4),))
    assert cost == (6, 1) and count == 1 and path[1] == (1, 0)
    # Rotation and pair-local palette changes must preserve the specified rule.
    example = puzzles[0]["test"][0]
    assert np.array_equal(solve(np.rot90(example["input"])), np.rot90(example["output"]))
    palette = np.arange(10)
    ordinary = [1, 2, 3, 4, 5, 6, 7, 9]
    palette[ordinary] = ordinary[::-1]
    assert np.array_equal(solve(palette[example["input"]]), palette[example["output"]])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(7000 + 10 * a + b, a, b)
            assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
            validate_generated_puzzle(puzzle)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "pairs": sum(len(p[s]) for p in puzzles for s in ("train", "test")),
            "board_shapes": len(shapes), "brush_shapes": len(brushes), "marker_signatures": len(signatures),
            "palettes": len(palettes), "bend_counts": sorted(bends), "edge_snapshots": edge,
            "irregular_barrier_pairs": irregular, "offset_locator_pairs": offset, "retracing_pairs": retrace,
            "reading_order_counterexamples": reading_order, "nearest_first_counterexamples": nearest_first}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
