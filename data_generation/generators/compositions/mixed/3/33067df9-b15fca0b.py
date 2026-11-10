"""Exact-span rectangle codes constrain a unique shortest orthogonal route.

Only the route is painted. Rectangle footprints are implicit obstacles; their
sparse code and the four-cell color key are copied unchanged. No file I/O,
example lookup, persistent generation state, or non-NumPy dependency is used.
"""
from collections import deque
import hashlib

import numpy as np


_HOST_SEEDS = (
    0, 506163566, 68684089, 134267110, 19281958, 418167275, 872884308,
    790705663, 842218342, 550166091, 153957332, 693906067, 168065505,
    759530223, 976213447, 241239784, 938795537, 705753779, 260472797,
    848821358, 531933295, 657619302, 691904030, 215500682, 384561436,
    413614957, 56870187, 473794988, 815651769, 222230441, 298816376,
    653774877, 792215012, 854769975, 808646173, 688894338, 218272893,
    248534934, 453391243, 534591369, 48737366, 770528139,
)


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return a.astype(np.int8, copy=False)


def _runs(coarse):
    """Maximal horizontal runs as (label, left, exclusive_right); 0 is empty."""
    rows = []
    for row in coarse:
        found, col = [], 0
        while col < len(row):
            color = int(row[col])
            if color == 0:
                col += 1
                continue
            end = col + 1
            while end < len(row) and int(row[end]) == color:
                end += 1
            found.append((color, col, end))
            col = end
        rows.append(found)
    return rows


def _rectangles(coarse):
    rows = _runs(coarse)
    rectangles = []
    for top, runs in enumerate(rows):
        before = set(rows[top - 1]) if top else set()
        for color, left, right in runs:
            if (color, left, right) in before:
                continue
            bottom = top + 1
            while bottom < len(rows) and (color, left, right) in rows[bottom]:
                bottom += 1
            rectangles.append((color, top, bottom, left, right))
    return rectangles


def _footprint(coarse, mode="exact"):
    """Render implicit obstacles at pitch two, core one, outer margin one.

    Other modes are deliberately wrong decoding rules for necessity checks.
    They are never used by solve to decide the intended answer.
    """
    height, width = coarse.shape
    mask = np.zeros((2 * height + 1, 2 * width + 1), dtype=bool)
    if mode == "raw":
        mask[1::2, 1::2] = coarse != 0
    elif mode == "no_vertical":
        for row, runs in enumerate(_runs(coarse)):
            for _, left, right in runs:
                mask[2 * row + 1, 2 * left + 1:2 * right] = True
    elif mode == "overlap":
        # Wrong: coalesce overlapping same-label runs, then take a bounding box.
        nodes = [(r, *run) for r, runs in enumerate(_runs(coarse)) for run in runs]
        parents = list(range(len(nodes)))

        def root(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i

        for i, (r, color, left, right) in enumerate(nodes):
            for j in range(i):
                rr, cc, ll, ee = nodes[j]
                if r == rr + 1 and color == cc and max(left, ll) < min(right, ee):
                    parents[root(i)] = root(j)
        groups = {}
        for i, node in enumerate(nodes):
            groups.setdefault(root(i), []).append(node)
        for group in groups.values():
            top, bottom = min(n[0] for n in group), max(n[0] for n in group) + 1
            left, right = min(n[2] for n in group), max(n[3] for n in group)
            mask[2 * top + 1:2 * bottom, 2 * left + 1:2 * right] = True
    elif mode == "exact":
        for _, top, bottom, left, right in _rectangles(coarse):
            mask[2 * top + 1:2 * bottom, 2 * left + 1:2 * right] = True
    else:
        raise ValueError("unknown decoding mode")
    return mask


def _bridge_footprint(coarse):
    """Cell-and-bridge check of rectangle rendering, without stack extraction."""
    mask = np.zeros((2 * coarse.shape[0] + 1, 2 * coarse.shape[1] + 1), bool)
    mask[1::2, 1::2] = coarse != 0
    rows = _runs(coarse)
    for row, runs in enumerate(rows):
        for color, left, right in runs:
            for col in range(left, right - 1):
                mask[2 * row + 1, 2 * col + 2] = True
            if row + 1 < len(rows) and (color, left, right) in rows[row + 1]:
                mask[2 * row + 2, 2 * left + 1:2 * right] = True
    return mask


def _parse(input_grid):
    grid = _grid(input_grid)
    height, width = grid.shape
    if height < 9 or width < 7 or height % 2 != 1 or width % 2 != 1:
        raise ValueError("expected a two-row key above an odd lattice board")
    roles = tuple(int(c) for c in grid[0, :4])
    if len(set(roles)) != 4:
        raise ValueError("the four role colors must be distinct")
    background, wall, terminal, route = roles
    if not (np.all(grid[0, 4:] == wall) and np.all(grid[1] == wall)):
        raise ValueError("malformed color key or separator")
    board = grid[2:]
    # Literal 0 can be a payload. Use internal labels 1..6, not literal colors.
    payloads = sorted(set(int(c) for c in np.unique(board)) - set(roles))
    coarse = np.zeros(board[1::2, 1::2].shape, dtype=np.int8)
    for label, color in enumerate(payloads, 1):
        positions = board == color
        allowed = np.zeros(board.shape, bool)
        allowed[1::2, 1::2] = True
        if np.any(positions & ~allowed):
            raise ValueError("rectangle code colors occur only at coarse sites")
        coarse[board[1::2, 1::2] == color] = label
    if not payloads or np.any(board == route):
        raise ValueError("need a rectangle code and no prepainted board route")
    terminals = np.flatnonzero(board.ravel() == terminal).tolist()
    if len(terminals) != 2:
        raise ValueError("expected exactly two board terminals")
    walls = board == wall
    mask = _footprint(coarse) | walls
    if mask.ravel()[terminals].any():
        raise ValueError("terminal lies in an obstacle footprint")
    return grid, board, roles, coarse, walls, mask, tuple(terminals)


def _bfs(mask, start):
    """All distances, shortest-path counts capped at two, and one parent tree."""
    height, width = mask.shape
    blocked = mask.ravel().tolist()
    distances, ways, parents = [-1] * len(blocked), [0] * len(blocked), [-1] * len(blocked)
    if blocked[start]:
        return distances, ways, parents
    distances[start], ways[start] = 0, 1
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        row, col = divmod(cell, width)
        neighbors = []
        if row:
            neighbors.append(cell - width)
        if row + 1 < height:
            neighbors.append(cell + width)
        if col:
            neighbors.append(cell - 1)
        if col + 1 < width:
            neighbors.append(cell + 1)
        distance = distances[cell] + 1
        for neighbor in neighbors:
            if blocked[neighbor]:
                continue
            if distances[neighbor] == -1:
                distances[neighbor] = distance
                ways[neighbor] = ways[cell]
                parents[neighbor] = cell
                queue.append(neighbor)
            elif distances[neighbor] == distance:
                ways[neighbor] = min(2, ways[neighbor] + ways[cell])
    return distances, ways, parents


def _path(parents, start, goal):
    result = [goal]
    while result[-1] != start:
        previous = parents[result[-1]]
        if previous < 0:
            raise ValueError("terminals are disconnected")
        result.append(previous)
    return result[::-1]


def _turns(path, width):
    differences = [b - a for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(differences, differences[1:]))


def _alternate(mask, start, goal, path):
    """Find a strictly longer simple alternative, by forbidding one route cell."""
    # Any distinct simple route omits some intermediate cell of the unique best.
    for cell in path[1:-1]:
        blocked = mask.copy()
        blocked.flat[cell] = True
        distances, _, parents = _bfs(blocked, start)
        if distances[goal] >= 0:
            assert distances[goal] > len(path) - 1
            return _path(parents, start, goal)
    return None


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode obstacles and paint the unique shortest route using input only."""
    grid, _, roles, _, _, mask, terminals = _parse(input_grid)
    start, goal = terminals
    distances, ways, parents = _bfs(mask, start)
    if distances[goal] < 2 or ways[goal] != 1:
        raise ValueError("expected a unique shortest route with an intermediate cell")
    path = _path(parents, start, goal)
    output = grid.copy()
    for cell in path[1:-1]:
        row, col = divmod(cell, mask.shape[1])
        if grid[row + 2, col] != roles[0]:
            raise ValueError("route crosses a non-background cell")
        output[row + 2, col] = roles[3]
    return output


def _sample_coarse(rng, rows, cols):
    coarse = np.zeros((rows, cols), np.int8)
    labels = int(rng.integers(1, 5))
    operations = int(rng.integers(5, min(19, rows + cols + 3)))
    for _ in range(operations):
        top, left = int(rng.integers(rows)), int(rng.integers(cols))
        bottom = int(rng.integers(top + 1, min(rows, top + 5) + 1))
        right = int(rng.integers(left + 1, cols + 1))
        coarse[top:bottom, left:right] = int(rng.integers(1, labels + 1))
    # Optional interrupted recurrence or blank band; both use the same decoder.
    if rng.random() < 0.22:
        coarse[int(rng.integers(rows))] = 0
    return coarse


def _sample_walls(rng, shape):
    walls = np.zeros(shape, bool)
    for _ in range(int(rng.integers(1, 5))):
        horizontal = bool(rng.integers(2))
        short, long = shape if horizontal else shape[::-1]
        position = 2 * int(rng.integers((short + 1) // 2))
        begin = int(rng.integers(long - 1))
        end = int(rng.integers(begin + 2, min(begin + 9, long) + 1))
        if horizontal:
            walls[position, begin:end] = True
        else:
            walls[begin:end, position] = True
    # Lines on an even row/column cannot erase a sparse code dot.
    return walls


def _make_pair(rng, roles, mode, test=False):
    """Select witnesses from random obstacle fields; no retained route template."""
    for _ in range(600):
        rows = int(rng.integers(4, 12 if test else 10))
        cols = int(rng.integers(4, 14 if test else 12))
        coarse = _sample_coarse(rng, rows, cols)
        if len(_rectangles(coarse)) < 2:
            continue
        if not any(right - left > 1 for rr in _runs(coarse) for _, left, right in rr):
            continue
        exact = _footprint(coarse)
        no_vertical = _footprint(coarse, "no_vertical")
        overmerged = _footprint(coarse, "overlap")
        if not np.any(exact & ~no_vertical):
            continue
        if mode != "exact" and not np.any(overmerged & ~exact):
            continue
        walls = _sample_walls(rng, exact.shape)
        mask, no_vertical, overmerged = exact | walls, no_vertical | walls, overmerged | walls
        raw = _footprint(coarse, "raw") | walls
        free = np.flatnonzero(~overmerged.ravel()).tolist()
        if len(free) < 2:
            continue
        starts = rng.choice(free, size=min(7, len(free)), replace=False).tolist()
        for start in starts:
            distances, ways, parents = _bfs(mask, start)
            no_dist, no_ways, _ = _bfs(no_vertical, start)
            sr, sc = divmod(start, mask.shape[1])
            goals = [cell for cell in free if ways[cell] == 1 and distances[cell] >= 8
                     and distances[cell] > abs(cell // mask.shape[1] - sr)
                     + abs(cell % mask.shape[1] - sc)]
            rng.shuffle(goals)
            wrong_search = raw_search = None
            for goal in goals[:40]:
                vertical_needed = no_dist[goal] < distances[goal]
                if (mode in ("exact", "both")) != vertical_needed:
                    continue
                if mode == "partial" and no_ways[goal] != 1:
                    continue
                path = _path(parents, start, goal)
                partial_needed = bool(overmerged.ravel()[path].any())
                if (mode in ("partial", "both")) != partial_needed:
                    continue
                if _turns(path, mask.shape[1]) < 2:
                    continue
                if partial_needed:
                    if wrong_search is None:
                        wrong_search = _bfs(overmerged, start)
                    wrong_dist, wrong_ways, _ = wrong_search
                    if wrong_dist[goal] <= distances[goal] or wrong_ways[goal] != 1:
                        continue
                if raw_search is None:
                    raw_search = _bfs(raw, start)
                if raw_search[0][goal] >= distances[goal]:
                    continue
                alternate = _alternate(mask, start, goal, path)
                if alternate is None:
                    continue
                background, wall, terminal, route = roles
                payload_colors = [c for c in range(10) if c not in roles]
                rng.shuffle(payload_colors)
                board = np.full(mask.shape, background, np.int8)
                for label in np.unique(coarse):
                    if label:
                        board[1::2, 1::2][coarse == label] = payload_colors[int(label) - 1]
                board[walls] = wall
                board.flat[start] = board.flat[goal] = terminal
                grid = np.full((board.shape[0] + 2, board.shape[1]), wall, np.int8)
                grid[0, :4] = roles
                grid[2:] = board
                return {"input": grid, "output": solve(grid)}
    raise RuntimeError("bounded search could not construct a " + mode + " witness")


def _key(array):
    a = np.asarray(array, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    roles = tuple(int(c) for c in rng.choice(10, size=4, replace=False))
    modes = ["exact", "partial"] + [str(rng.choice(("exact", "partial")))
                                    for _ in range(num_train - 2)]
    rng.shuffle(modes)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plan in (("train", modes), ("test", ["both"] * num_test)):
        for mode in plan:
            for _ in range(24):
                pair = _make_pair(rng, roles, mode, split == "test")
                signature = _key(pair["input"])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded distinct-input search exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_features(pair, check_output=True):
    grid, board, roles, coarse, walls, mask, terminals = _parse(pair["input"])
    start, goal = terminals
    width = mask.shape[1]
    distances, ways, parents = _bfs(mask, start)
    assert ways[goal] == 1 and distances[goal] >= 8
    path = _path(parents, start, goal)
    assert _turns(path, width) >= 2
    sr, sc = divmod(start, width)
    gr, gc = divmod(goal, width)
    assert distances[goal] > abs(sr - gr) + abs(sc - gc)
    assert np.array_equal(_footprint(coarse), _bridge_footprint(coarse))
    no_vertical = _footprint(coarse, "no_vertical") | walls
    no_dist, no_ways, _ = _bfs(no_vertical, start)
    vertical_needed = no_dist[goal] < distances[goal]
    wrong = _footprint(coarse, "overlap") | walls
    assert np.all(wrong[mask])
    partial_needed = bool(wrong.ravel()[path].any())
    if partial_needed:
        wrong_dist, wrong_ways, _ = _bfs(wrong, start)
        assert wrong_dist[goal] > distances[goal] and wrong_ways[goal] == 1
    if not vertical_needed:
        assert no_dist[goal] == distances[goal] and no_ways[goal] == 1
    raw_dist, _, _ = _bfs(_footprint(coarse, "raw") | walls, start)
    assert raw_dist[goal] < distances[goal]
    alternate = _alternate(mask, start, goal, path)
    assert alternate is not None and len(alternate) > len(path)
    assert np.any(walls) and len(_rectangles(coarse)) >= 2
    if check_output:
        output = _grid(pair["output"])
        assert output.shape == grid.shape
        expected = grid.copy()
        for cell in path[1:-1]:
            row, col = divmod(cell, width)
            assert board[row, col] == roles[0] and not mask[row, col]
            expected[row + 2, col] = roles[3]
        assert np.array_equal(expected, output)
        assert np.array_equal(solve(grid), output)
        assert np.count_nonzero(grid != output) == len(path) - 2
        assert np.all(output[grid != roles[0]] == grid[grid != roles[0]])
    rects = _rectangles(coarse)
    return {
        "shape": grid.shape, "coarse_shape": coarse.shape, "roles": roles,
        "rectangles": len(rects), "payload_colors": len(set(r[0] for r in rects)),
        "longest_run": max(r[4] - r[3] for r in rects),
        "deepest_stack": max(r[2] - r[1] for r in rects),
        "vertical_needed": vertical_needed, "partial_needed": partial_needed,
        "distance": distances[goal], "alternate_distance": len(alternate) - 1,
        "turns": _turns(path, width), "wall_cells": int(walls.sum()),
        "empty_rows": int(np.count_nonzero(np.all(coarse == 0, axis=1))),
        "empty_cols": int(np.count_nonzero(np.all(coarse == 0, axis=0))),
        "boundary_terminals": sum(r in (0, mask.shape[0] - 1) or c in (0, width - 1)
                                  for r, c in ((sr, sc), (gr, gc))),
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen, common_roles, train_kinds = set(), None, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            signature = _key(_grid(pair["input"]))
            assert signature not in seen
            seen.add(signature)
            features = _pair_features(pair)
            if common_roles is None:
                common_roles = features["roles"]
            assert features["roles"] == common_roles
            kind = features["vertical_needed"], features["partial_needed"]
            if split == "train":
                assert kind in ((True, False), (False, True))
                train_kinds.add(kind)
            else:
                assert kind == (True, True)
    assert train_kinds == {(True, False), (False, True)}


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded semantic, reproducibility, counterfactual, and diversity checks.

    Streams 42 puzzles; retains only digests and small aggregate sets. No files
    or execution traces are written. Returned metrics are JSON-serializable.
    """
    shapes, coarse_shapes, roles, digests, layouts = (set() for _ in range(5))
    rectangle_counts, lengths, depths, wall_counts, endpoints = (set() for _ in range(5))
    totals = {"pairs": 0, "exact_training": 0, "partial_training": 0,
              "combined_tests": 0, "empty_rows": 0, "empty_cols": 0}
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                features = _pair_features(pair)
                shapes.add(features["shape"])
                coarse_shapes.add(features["coarse_shape"])
                roles.add(features["roles"])
                rectangle_counts.add(features["rectangles"])
                lengths.add(features["longest_run"])
                depths.add(features["deepest_stack"])
                wall_counts.add(features["wall_cells"])
                endpoints.add(features["boundary_terminals"])
                totals["pairs"] += 1
                totals["empty_rows"] += features["empty_rows"]
                totals["empty_cols"] += features["empty_cols"]
                if split == "test":
                    totals["combined_tests"] += 1
                elif features["vertical_needed"]:
                    totals["exact_training"] += 1
                else:
                    totals["partial_training"] += 1
                grid, _, _, coarse, _, _, _ = _parse(pair["input"])
                layouts.add((coarse.shape, hashlib.sha256(coarse.tobytes()).hexdigest()))
                before = grid.copy()
                assert np.array_equal(solve(grid), pair["output"])
                assert np.array_equal(grid, before), "solve mutated its input"
        # One geometric metamorphism and one global palette permutation per seed.
        pair = puzzle["test"][0]
        mirrored_in, mirrored_out = pair["input"].copy(), pair["output"].copy()
        mirrored_in[2:] = mirrored_in[2:, ::-1]
        mirrored_out[2:] = mirrored_out[2:, ::-1]
        assert np.array_equal(solve(mirrored_in), mirrored_out)
        mirrored_in[2:] = pair["input"][2:][::-1]
        mirrored_out[2:] = pair["output"][2:][::-1]
        assert np.array_equal(solve(mirrored_in), mirrored_out)
        permutation = np.roll(np.arange(10, dtype=np.int8), seed % 9 + 1)
        assert np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]])
    assert len(digests) == 42 and len(roles) >= 30
    assert len(shapes) >= 25 and len(coarse_shapes) >= 25 and len(layouts) >= 180
    assert len(rectangle_counts) >= 12 and max(lengths) >= 8 and max(depths) >= 5
    assert len(wall_counts) >= 15 and len(endpoints) >= 2
    assert totals["empty_rows"] > 0 and totals["empty_cols"] > 0
    # Solver scope extends beyond the sampler: smallest/largest legal lattices,
    # literal zero as payload, one rectangle, and a straight unique route.
    for rows, cols in ((3, 3), (13, 14)):
        probe = np.full((2 * rows + 3, 2 * cols + 1), 9, np.int8)
        probe[0, :4] = (6, 9, 2, 4)
        probe[2] = 6
        probe[2, 0] = probe[2, -1] = 2
        probe[3::2, 1::2] = 0
        expected = probe.copy()
        expected[2, 1:-1] = 4
        assert np.array_equal(solve(probe.tolist()), expected)
        covered = probe.copy()
        covered[2, -1] = 6
        covered[4, 2] = 2
        try:
            solve(covered)
        except ValueError:
            pass
        else:
            raise AssertionError("covered terminal accepted")
    # A broad open corner gives tied shortest routes; isolated corners give none.
    for disconnected in (False, True):
        probe = np.full((9, 7), 9 if disconnected else 6, np.int8)
        probe[0] = probe[1] = 9
        probe[0, :4] = (6, 9, 2, 4)
        probe[7, 5] = 0
        probe[2, 0] = 2
        probe[8, 6] = 2
        try:
            solve(probe)
        except ValueError:
            pass
        else:
            raise AssertionError("tied or disconnected terminals accepted")
    first = generate_puzzle(123456, 5, 3)
    assert _puzzle_digest(first) == _puzzle_digest(generate_puzzle(123456, 5, 3))
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(91827, nt, ns)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, **totals, "distinct_puzzles": len(digests),
            "shapes": len(shapes), "coarse_shapes": len(coarse_shapes),
            "normalized_code_layouts": len(layouts), "role_palettes": len(roles),
            "rectangle_count_range": [min(rectangle_counts), max(rectangle_counts)],
            "longest_run_range": [min(lengths), max(lengths)],
            "deepest_stack_range": [min(depths), max(depths)],
            "wall_cell_count_range": [min(wall_counts), max(wall_counts)],
            "boundary_terminal_counts": sorted(endpoints)}
