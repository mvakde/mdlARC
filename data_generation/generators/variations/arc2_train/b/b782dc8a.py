"""Seeded alternating flood fill in varied four-connected mazes."""

import numpy as np


_SOURCE_INPUT_SIGNATURES = {
    (13, 14,
     "00080008000008"
     "88088808088808"
     "08000808080888"
     "08888808080000"
     "00000008088808"
     "88888808000808"
     "80000808880808"
     "88880800080800"
     "00081888880880"
     "88084100000080"
     "08081888888880"
     "08880800000000"
     "00000808888888"),
    (11, 24,
     "800000888888088808808880"
     "008880000008000808008080"
     "888080888808880808888080"
     "800080800800080800000080"
     "808880880808880880888880"
     "808000080808000080800000"
     "808888880808888883888880"
     "800000000800000032300080"
     "880888088808888883888080"
     "080808080008000080808080"
     "088808880888088088808880"),
    (15, 15,
     "880800800000000"
     "080888848888888"
     "080000438000008"
     "088888848880888"
     "000000800080800"
     "888880888080808"
     "000080008080808"
     "888088808080888"
     "008000808080000"
     "808880888088808"
     "800080000000808"
     "888080888888808"
     "008080800000008"
     "808880808888888"
     "800000808000000"),
}

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} outside [{low}, {high}]")


def _resolve_counts(seed, num_train, num_test):
    _count("num_train", num_train, 2, 10)
    _count("num_test", num_test, 1, 4)
    code = int(seed) % (1 << 63)
    if num_test is None:
        limit = 4 if num_train is None else min(4, int(num_train) - 1)
        test = 1 + code % limit
    else:
        test = int(num_test)
    if num_train is None:
        low = max(2, test + 1)
        train = low + (code // 4) % (11 - low)
    else:
        train = int(num_train)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _neighbors(y, x, height, width):
    for dy, dx in _DIRS:
        yy, xx = y + dy, x + dx
        if 0 <= yy < height and 0 <= xx < width:
            yield yy, xx


def _distances(walls, start):
    height, width = walls.shape
    result = np.full((height, width), -1, dtype=np.int16)
    if walls[start]:
        return result
    queue = [start]
    result[start] = 0
    head = 0
    while head < len(queue):
        y, x = queue[head]
        head += 1
        for yy, xx in _neighbors(y, x, height, width):
            if not walls[yy, xx] and result[yy, xx] < 0:
                result[yy, xx] = result[y, x] + 1
                queue.append((yy, xx))
    return result


def _components(walls):
    height, width = walls.shape
    labels = np.full((height, width), -1, dtype=np.int16)
    groups = []
    for y in range(height):
        for x in range(width):
            if walls[y, x] or labels[y, x] >= 0:
                continue
            label = len(groups)
            queue = [(y, x)]
            labels[y, x] = label
            head = 0
            while head < len(queue):
                cy, cx = queue[head]
                head += 1
                for yy, xx in _neighbors(cy, cx, height, width):
                    if not walls[yy, xx] and labels[yy, xx] < 0:
                        labels[yy, xx] = label
                        queue.append((yy, xx))
            groups.append(queue)
    return labels, groups


def _dimensions(rng, regime):
    # Both splits cover all aspect ratios and size bands; tests are only
    # somewhat more likely to use the larger band.
    band = int(rng.choice(3, p=(.42, .36, .22) if regime == "source_train"
                          else (.29, .38, .33)))
    aspect = int(rng.integers(3))
    if aspect == 0:
        lo, hi = ((10, 17), (16, 24), (22, 30))[band]
        h = int(rng.integers(lo, hi + 1))
        w = min(30, max(9, h + int(rng.integers(-3, 4))))
    else:
        lo, hi = ((9, 15), (12, 20), (17, 24))[band]
        short = int(rng.integers(lo, hi + 1))
        long = int(rng.integers(min(30, short + 4), min(30, short + 11) + 1))
        h, w = (short, long) if aspect == 1 else (long, short)
    return h, w


def _room_tree(rng, rows, cols, style):
    """Connect lattice rooms with either a winding DFS or a branching tree."""
    count = rows * cols
    if style == 0:
        start = int(rng.integers(count))
        visited = {start}
        stack = [start]
        selected = []
        while stack:
            a = stack[-1]
            y, x = divmod(a, cols)
            options = []
            for yy, xx in _neighbors(y, x, rows, cols):
                b = yy * cols + xx
                if b not in visited:
                    options.append(b)
            if not options:
                stack.pop()
                continue
            b = options[int(rng.integers(len(options)))]
            visited.add(b)
            selected.append((a, b))
            stack.append(b)
        return selected
    edges = []
    for y in range(rows):
        for x in range(cols):
            a = y * cols + x
            if y + 1 < rows:
                edges.append((a, a + cols))
            if x + 1 < cols:
                edges.append((a, a + 1))
    rng.shuffle(edges)
    parent = list(range(count))
    rank = [0] * count

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    selected = []
    for a, b in edges:
        aa, bb = root(a), root(b)
        if aa == bb:
            continue
        if rank[aa] < rank[bb]:
            aa, bb = bb, aa
        parent[bb] = aa
        rank[aa] += rank[aa] == rank[bb]
        selected.append((a, b))
    return selected


def _layout(rng, height, width, regime):
    for _ in range(28):
        oy, ox = int(rng.integers(2)), int(rng.integers(2))
        rows = (height - 1 - oy) // 2 + 1
        cols = (width - 1 - ox) // 2 + 1
        if rows < 4 or cols < 4:
            continue
        rooms = [(oy + 2 * y, ox + 2 * x)
                 for y in range(rows) for x in range(cols)]
        walls = np.ones((height, width), dtype=bool)
        for cell in rooms:
            walls[cell] = False
        tree = _room_tree(rng, rows, cols, int(rng.integers(2)))
        cut_limit = min(8, max(2, len(tree) // 13))
        cuts = set(int(x) for x in rng.choice(
            len(tree), size=int(rng.integers(1, cut_limit + 1)), replace=False))
        for index, (a, b) in enumerate(tree):
            if index not in cuts:
                ay, ax = rooms[a]
                by, bx = rooms[b]
                walls[(ay + by) // 2, (ax + bx) // 2] = False

        # Open some alternate passages only within an existing component.
        labels, groups = _components(walls)
        loop_rate = float(rng.uniform(.05, .39))
        for y in range(rows):
            for x in range(cols):
                a = y * cols + x
                for b in (a + 1, a + cols):
                    if b >= len(rooms) or (b == a + 1 and x + 1 == cols):
                        continue
                    ay, ax = rooms[a]
                    by, bx = rooms[b]
                    cy, cx = (ay + by) // 2, (ax + bx) // 2
                    if (walls[cy, cx] and labels[ay, ax] == labels[by, bx]
                            and rng.random() < loop_rate):
                        walls[cy, cx] = False

        # Irregular widths, holes, and edge exits keep the room lattice from
        # becoming a rigid template. A new cell may touch only one component.
        labels, groups = _components(walls)
        widen_rate = float(rng.uniform(.025, .22 if regime == "source_train" else .17))
        candidates = np.argwhere(walls)
        rng.shuffle(candidates)
        for y0, x0 in candidates:
            y, x = int(y0), int(x0)
            if rng.random() >= widen_rate:
                continue
            adjacent = {int(labels[yy, xx])
                        for yy, xx in _neighbors(y, x, height, width)
                        if labels[yy, xx] >= 0}
            if len(adjacent) == 1:
                walls[y, x] = False
                labels[y, x] = next(iter(adjacent))

        # An occasional uninterrupted partition is a simple member of the
        # family, rather than the compulsory boundary of every puzzle.
        if rng.random() < .09:
            if rng.random() < .5:
                walls[int(rng.integers(2, height - 2)), :] = True
            else:
                walls[:, int(rng.integers(2, width - 2))] = True

        labels, groups = _components(walls)
        eligible = [i for i, group in enumerate(groups)
                    if len(group) >= max(20, height * width // 17)
                    and sum(len(g) for j, g in enumerate(groups) if j != i)
                    >= max(8, height * width // 45)]
        if not eligible:
            continue
        if rng.random() < .65:
            chosen = max(eligible, key=lambda i: len(groups[i]))
        else:
            chosen = eligible[int(rng.integers(len(eligible)))]
        active_anchor = groups[chosen][0]

        # A tiny pocket can touch the active region diagonally while staying
        # disconnected in the four-neighbor graph.
        if rng.random() < .55:
            active = groups[chosen]
            ys = [cell[0] for cell in active]
            xs = [cell[1] for cell in active]
            pocket_candidates = []
            for y in range(height):
                for x in range(width):
                    if not walls[y, x]:
                        continue
                    if any(not walls[yy, xx]
                           for yy, xx in _neighbors(y, x, height, width)):
                        continue
                    if any(0 <= y + dy < height and 0 <= x + dx < width
                           and labels[y + dy, x + dx] == chosen
                           for dy in (-1, 1) for dx in (-1, 1)):
                        inside = min(ys) <= y <= max(ys) and min(xs) <= x <= max(xs)
                        pocket_candidates.append((inside, y, x))
            if pocket_candidates:
                inside_candidates = [p for p in pocket_candidates if p[0]]
                pool = inside_candidates if inside_candidates and rng.random() < .8 else pocket_candidates
                _, y, x = pool[int(rng.integers(len(pool)))]
                walls[y, x] = False

        labels, groups = _components(walls)
        chosen = int(labels[active_anchor])
        active = groups[chosen]
        by_degree = {2: [], 3: [], 4: []}
        for y, x in active:
            degree = sum(not walls[yy, xx]
                         for yy, xx in _neighbors(y, x, height, width))
            if degree in by_degree:
                by_degree[degree].append((y, x))
        options = [degree for degree, cells in by_degree.items() if cells]
        if not options:
            continue
        weights = {2: .17, 3: .46, 4: .37}
        probabilities = np.array([weights[d] for d in options], dtype=float)
        probabilities /= probabilities.sum()
        degree = int(rng.choice(options, p=probabilities))
        center = by_degree[degree][int(rng.integers(len(by_degree[degree])))]
        distances = _distances(walls, center)
        even = int(np.count_nonzero((distances >= 0) & (distances % 2 == 0))) - 1
        odd = int(np.count_nonzero((distances >= 0) & (distances % 2 == 1))) - degree
        if even >= 4 and odd >= 4:
            return walls, center
    return None


def _source_signature(grid):
    h, w = grid.shape
    return h, w, "".join(str(int(v)) for v in grid.flat)


def _generate_pair(rng, regime, wall_color, neutral_color, fill_pool):
    for _ in range(35):
        height, width = _dimensions(rng, regime)
        layout = _layout(rng, height, width, regime)
        if layout is None:
            continue
        walls, center = layout
        fill = [int(c) for c in rng.choice(fill_pool, size=2, replace=False)]
        if rng.random() < .5:
            fill.reverse()
        center_color, arm_color = fill
        grid = np.full((height, width), neutral_color, dtype=np.int8)
        grid[walls] = wall_color
        grid[center] = center_color
        for arm in _neighbors(*center, height, width):
            if not walls[arm]:
                grid[arm] = arm_color
        if _source_signature(grid) in _SOURCE_INPUT_SIGNATURES:
            continue
        distances = _distances(walls, center)
        output = grid.copy()
        output[(distances >= 0) & (distances % 2 == 0)] = center_color
        output[(distances >= 0) & (distances % 2 == 1)] = arm_color
        return grid, output
    raise ValueError("unable to construct a pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    palette = [int(x) for x in rng.permutation(10)]
    walls, neutrals, fill_pool = palette[:2], palette[2:4], palette[4:]
    roles = [(0, 0), (1, 1)]
    roles.extend((int(rng.integers(2)), int(rng.integers(2)))
                 for _ in range(train_count - 2))
    rng.shuffle(roles)
    roles.extend((int(rng.integers(2)), int(rng.integers(2)))
                 for _ in range(test_count))
    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=train_count + test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, ((wi, ni), child_seed) in enumerate(zip(roles, child_seeds)):
        pair_rng = np.random.default_rng(int(child_seed))
        regime = "source_train" if index < train_count else "target_test"
        for _ in range(10):
            input_grid, output_grid = _generate_pair(
                pair_rng, regime, walls[wi], neutrals[ni], fill_pool)
            key = _source_signature(input_grid)
            if key not in seen:
                break
        else:
            raise ValueError("could not make unique inputs")
        seen.add(key)
        puzzle["train" if index < train_count else "test"].append(
            {"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape, "dimensions changed")
    colors, counts = np.unique(input_grid, return_counts=True)
    _require(len(colors) == 4, "each input needs four distinct roles")
    _require(set(int(c) for c in np.unique(output_grid)) ==
             set(int(c) for c in colors), "output palette changed")
    singleton = [int(c) for c, n in zip(colors, counts) if n == 1]
    _require(len(singleton) == 1, "center is not a unique singleton")
    center_color = singleton[0]
    arms = [int(c) for c, n in zip(colors, counts) if 2 <= n <= 4]
    _require(len(arms) == 1, "arm color is not identifiable")
    arm_color = arms[0]
    residual = [int(c) for c in colors if c not in (center_color, arm_color)]
    changed = [bool(np.any(output_grid[input_grid == c] != c)) for c in residual]
    _require(sum(changed) == 1, "wall and neutral roles are not distinct")
    neutral_color = residual[changed.index(True)]
    wall_color = residual[changed.index(False)]
    point = np.argwhere(input_grid == center_color)[0]
    center = (int(point[0]), int(point[1]))
    height, width = input_grid.shape
    expected_arms = {(y, x) for y, x in _neighbors(*center, height, width)
                     if input_grid[y, x] != wall_color}
    actual_arms = {tuple(int(v) for v in point)
                   for point in np.argwhere(input_grid == arm_color)}
    _require(actual_arms == expected_arms, "arms are not the local open cross")
    _require(2 <= len(actual_arms) <= 4, "seed has too few arms")
    walls = input_grid == wall_color
    distances = _distances(walls, center)
    reachable = distances >= 0
    expected = input_grid.copy()
    expected[reachable & (distances % 2 == 0)] = center_color
    expected[reachable & (distances % 2 == 1)] = arm_color
    _require(np.array_equal(expected, output_grid), "incorrect four-neighbor parity flood")
    _require(np.all(output_grid[walls] == wall_color), "wall changed")
    _require(np.all(output_grid[~reachable] == input_grid[~reachable]),
             "inactive component changed")
    neutral = input_grid == neutral_color
    even = int(np.count_nonzero(neutral & reachable & (distances % 2 == 0)))
    odd = int(np.count_nonzero(neutral & reachable & (distances % 2 == 1)))
    inactive = int(np.count_nonzero(neutral & ~reachable))
    _require(even >= 4 and odd >= 4, "both colors need visible extension")
    _require(inactive >= 8, "inactive neutral region too small")
    labels, groups = _components(walls)
    active_label = int(labels[center])
    active = groups[active_label]
    ys = [p[0] for p in active]
    xs = [p[1] for p in active]
    inside = sum(1 for label, group in enumerate(groups) if label != active_label
                 for y, x in group
                 if min(ys) <= y <= max(ys) and min(xs) <= x <= max(xs))
    diagonal = sum(1 for y in range(height) for x in range(width)
                   if labels[y, x] >= 0 and labels[y, x] != active_label
                   and any(0 <= y + dy < height and 0 <= x + dx < width
                           and labels[y + dy, x + dx] == active_label
                           for dy in (-1, 1) for dx in (-1, 1)))
    missing = []
    cy, cx = center
    for name, dy, dx in (("up", -1, 0), ("down", 1, 0),
                         ("left", 0, -1), ("right", 0, 1)):
        y, x = cy + dy, cx + dx
        if not (0 <= y < height and 0 <= x < width) or walls[y, x]:
            missing.append(name)
    return {
        "wall": wall_color, "neutral": neutral_color,
        "center_color": center_color, "arm_color": arm_color,
        "arms": len(actual_arms), "missing": tuple(missing),
        "wall_fraction": float(np.mean(walls)), "active_size": len(active),
        "active_fraction": len(active) / max(1, int(np.count_nonzero(~walls))),
        "inactive_components": len(groups) - 1,
        "inside_inactive": inside, "diagonal_inactive": diagonal,
        "touches_edges": sum((any(y == 0 for y, _ in active),
                              any(y == height - 1 for y, _ in active),
                              any(x == 0 for _, x in active),
                              any(x == width - 1 for _, x in active))),
        "full_line": bool(np.any(np.all(walls[:, 1:-1], axis=0)) or
                          np.any(np.all(walls[1:-1, :], axis=1))),
    }


def _predict_from_training(train_pairs, test_input):
    wall_roles = set()
    neutral_roles = set()
    for pair in train_pairs:
        info = _analyze_pair(pair["input"], pair["output"])
        wall_roles.add(info["wall"])
        neutral_roles.add(info["neutral"])
    values, counts = np.unique(test_input, return_counts=True)
    center_candidates = [int(c) for c, n in zip(values, counts) if n == 1]
    arm_candidates = [int(c) for c, n in zip(values, counts) if 2 <= n <= 4]
    _require(len(center_candidates) == len(arm_candidates) == 1,
             "test cross is not identifiable")
    center_color, arm_color = center_candidates[0], arm_candidates[0]
    residual = {int(c) for c in values} - {center_color, arm_color}
    wall = residual & wall_roles
    neutral = residual & neutral_roles
    _require(len(wall) == len(neutral) == 1 and wall != neutral,
             "test structural colors were not demonstrated in training")
    wall_color = next(iter(wall))
    position = np.argwhere(test_input == center_color)[0]
    center = (int(position[0]), int(position[1]))
    distance = _distances(test_input == wall_color, center)
    predicted = test_input.copy()
    predicted[(distance >= 0) & (distance % 2 == 0)] = center_color
    predicted[(distance >= 0) & (distance % 2 == 1)] = arm_color
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle keys are invalid")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "sections must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4
             and len(train) > len(test), "pair counts are invalid")
    keys = set()
    infos = []
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair schema invalid")
        for grid in (pair["input"], pair["output"]):
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                     "grid is not a two-dimensional ndarray")
            h, w = grid.shape
            _require(1 <= h <= 30 and 1 <= w <= 30, "grid size invalid")
            _require(np.issubdtype(grid.dtype, np.integer), "grid dtype invalid")
            _require(bool(np.all((grid >= 0) & (grid <= 9))), "ARC color invalid")
        key = _source_signature(pair["input"])
        _require(key not in keys and key not in _SOURCE_INPUT_SIGNATURES,
                 "duplicate or source input")
        keys.add(key)
        infos.append(_analyze_pair(pair["input"], pair["output"]))
    wall_roles = {info["wall"] for info in infos[:len(train)]}
    neutral_roles = {info["neutral"] for info in infos[:len(train)]}
    _require(wall_roles.isdisjoint(neutral_roles), "wall and neutral conventions conflict")
    for info in infos[len(train):]:
        _require(info["wall"] in wall_roles and info["neutral"] in neutral_roles,
                 "test structural roles were not taught")
    for pair in test:
        _require(np.array_equal(_predict_from_training(train, pair["input"]),
                                pair["output"]), "test output not recoverable from training")


def _fingerprint(puzzle):
    return tuple((pair["input"].shape, pair["input"].tobytes(),
                  pair["output"].tobytes())
                 for section in ("train", "test") for pair in puzzle[section])


def _topology_metrics(input_grid, wall_color, center_color):
    point = np.argwhere(input_grid == center_color)[0]
    center = (int(point[0]), int(point[1]))
    active = {tuple(int(v) for v in row)
              for row in np.argwhere(_distances(input_grid == wall_color,
                                                  center) >= 0)}
    h, w = input_grid.shape
    adjacency = {p: [q for q in _neighbors(*p, h, w) if q in active]
                 for p in active}
    branch = sum(len(v) >= 3 for v in adjacency.values())
    bend = sum(len(v) == 2 and v[0][0] != v[1][0]
               and v[0][1] != v[1][1] for v in adjacency.values())
    loops = sum(len(v) for v in adjacency.values()) // 2 - len(active) + 1
    discovered = {}
    low = {}
    bridges = [0]

    def visit(p, parent):
        discovered[p] = len(discovered)
        low[p] = discovered[p]
        for q in adjacency[p]:
            if q == parent:
                continue
            if q not in discovered:
                visit(q, p)
                low[p] = min(low[p], low[q])
                bridges[0] += low[q] > discovered[p]
            else:
                low[p] = min(low[p], discovered[q])

    visit(center, None)
    return branch, bend, loops, bridges[0]


def self_test() -> None:
    automatic = generate_puzzle(17)
    validate_generated_puzzle(automatic)
    explicit = generate_puzzle(23, 6, 3)
    _require(_fingerprint(explicit) == _fingerprint(generate_puzzle(23, 6, 3)),
             "fixed seed is not deterministic")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (None, 1), (None, 4),
                                           (2, None), (10, None))):
        puzzle = generate_puzzle(100 + index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "train count not honored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "test count not honored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    fingerprints = set()
    train_counts, test_counts = set(), set()
    shapes, layouts, role_maps = set(), set(), set()
    arm_counts, missing_dirs = set(), set()
    wall_bins, active_bins, inactive_counts = set(), set(), set()
    inside_count = diagonal_count = full_line_count = edge_count = 0
    wide_count = tall_count = small_test = large_train = 0
    branch_count = bend_count = loop_count = bridge_count = 0
    changed_wall_puzzles = changed_neutral_puzzles = changed_fill_puzzles = 0
    changed_arm_puzzles = changed_shape_puzzles = novel_test_fill = 0
    pair_count = 0
    varying_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = [_analyze_pair(p["input"], p["output"])
                 for p in puzzle["train"] + puzzle["test"]]
        changed_wall_puzzles += len({i["wall"] for i in infos}) > 1
        changed_neutral_puzzles += len({i["neutral"] for i in infos}) > 1
        changed_fill_puzzles += len({
            frozenset((i["center_color"], i["arm_color"])) for i in infos}) > 1
        changed_arm_puzzles += len({i["arms"] for i in infos}) > 1
        changed_shape_puzzles += len({
            p["input"].shape for section in ("train", "test")
            for p in puzzle[section]}) > 1
        train_fill = {frozenset((i["center_color"], i["arm_color"]))
                      for i in infos[:len(puzzle["train"])]}
        novel_test_fill += any(
            frozenset((i["center_color"], i["arm_color"])) not in train_fill
            for i in infos[len(puzzle["train"]):])
        if len({(i["wall"], i["neutral"], i["center_color"], i["arm_color"])
                for i in infos}) > 1:
            varying_palettes += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                h, w = grid.shape
                info = _analyze_pair(grid, pair["output"])
                shapes.add((h, w))
                layouts.add((h, w, np.packbits(grid == info["wall"]).tobytes()))
                role_maps.add((info["wall"], info["neutral"],
                               info["center_color"], info["arm_color"]))
                arm_counts.add(info["arms"])
                missing_dirs.update(info["missing"])
                wall_bins.add(int(info["wall_fraction"] * 10))
                active_bins.add(int(info["active_fraction"] * 10))
                inactive_counts.add(info["inactive_components"])
                inside_count += info["inside_inactive"] > 0
                diagonal_count += info["diagonal_inactive"] > 0
                full_line_count += info["full_line"]
                edge_count += info["touches_edges"] == 4
                branch, bend, loops, bridges = _topology_metrics(
                    grid, info["wall"], info["center_color"])
                branch_count += branch > 0
                bend_count += bend > 0
                loop_count += loops > 0
                bridge_count += bridges > 0
                wide_count += w >= h + 4
                tall_count += h >= w + 4
                small_test += split == "test" and max(h, w) <= 17
                large_train += split == "train" and max(h, w) >= 24
                pair_count += 1
    _require(len(fingerprints) == 42, "cross-seed puzzles repeated")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "counts lack diversity")
    _require(len(shapes) >= 45 and len(layouts) >= 130, "geometry lacks diversity")
    _require(len(role_maps) >= 80 and varying_palettes >= 40, "colors lack diversity")
    _require(arm_counts == {2, 3, 4} and missing_dirs == {"up", "down", "left", "right"},
             "seed orientations lack diversity")
    _require(len(wall_bins) >= 3 and len(active_bins) >= 5
             and len(inactive_counts) >= 4, "maze density or components lack diversity")
    _require(inside_count >= 20 and diagonal_count >= 20,
             "enclosed and diagonal pockets lack coverage")
    _require(full_line_count < pair_count // 2, "straight dividers dominate")
    _require(wide_count >= 20 and tall_count >= 20 and small_test >= 5
             and large_train >= 10, "dimensions remain split-predictable")
    _require(edge_count >= 1, "no active region touches all four edges")
    _require(branch_count >= 100 and bend_count >= 100
             and loop_count >= 40 and bridge_count >= 100,
             "branches, bends, loops, or bottlenecks lack coverage")
    _require(changed_wall_puzzles >= 40 and changed_neutral_puzzles >= 40
             and changed_fill_puzzles >= 40 and changed_arm_puzzles >= 25
             and changed_shape_puzzles == 42 and novel_test_fill >= 15,
             "within-puzzle combinations lack coverage")
