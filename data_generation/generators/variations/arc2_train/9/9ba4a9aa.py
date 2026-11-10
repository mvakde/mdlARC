"""Generate checkerboard-to-tile dotted-route ARC puzzles."""

from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_SOURCE_INPUT_HASHES = {
    "963514825e4aced37a6d0c408dd1e2435596db4faf7869e358e3e9efd0028cdf",
    "7758c6d57d84a2959e690b431d9351549c9589771c06d22fa1492220edbb9441",
    "9279e8e0d528d5e9a497b74ff746498af274de91baf954bed6a481141fda9134",
    "9f89d99e1cfb4a3b828dc25db048f809c21baa53571e25942097365f1dbf5d98",
}


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = max(2, num_test + 1) + (seed // 4) % (10 - max(2, num_test + 1) + 1)
    elif num_train is None:
        minimum = max(2, num_test + 1)
        num_train = minimum + seed % (11 - minimum)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _inside(point: tuple[int, int], height: int, width: int) -> bool:
    return 0 <= point[0] < height and 0 <= point[1] < width


def _cells(top: int, left: int) -> set[tuple[int, int]]:
    return {(top + dr, left + dc) for dr in range(3) for dc in range(3)}


def _externals(top: int, left: int) -> tuple[tuple[int, int], ...]:
    return ((top - 1, left + 1), (top + 1, left + 3),
            (top + 3, left + 1), (top + 1, left - 1))


def _separated(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return (first[0] + 2 < second[0] or second[0] + 2 < first[0]
            or first[1] + 2 < second[1] or second[1] + 2 < first[1])


def _neighbors(point: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple((point[0] + 2 * dr, point[1] + 2 * dc) for dr, dc in _DIRECTIONS)


def _sample_marker(rng: np.random.Generator, height: int, width: int,
                   region: int | None) -> tuple[int, int]:
    if region is None:
        return (int(rng.integers(4, height - 6)), int(rng.integers(4, width - 6)))
    row_low = [0, 0, height - 6, height - 6][region]
    col_low = [0, width - 6, width - 6, 0][region]
    return (row_low + int(rng.integers(0, 4)),
            col_low + int(rng.integers(0, 4)))


def _grow_path(rng: np.random.Generator, start: tuple[int, int],
               outward: tuple[int, int], desired: int,
               height: int, width: int, blocked: set[tuple[int, int]],
               occupied: set[tuple[int, int]]) -> list[tuple[int, int]]:
    path = [start]
    direction = outward
    while len(path) < desired:
        options = []
        weights = []
        for dr, dc in _DIRECTIONS:
            nxt = (path[-1][0] + 2 * dr, path[-1][1] + 2 * dc)
            middle = (path[-1][0] + dr, path[-1][1] + dc)
            if not _inside(nxt, height, width) or nxt in blocked or nxt in occupied:
                continue
            if middle in blocked or middle in occupied:
                continue
            if any(nxt in _neighbors(old) for old in path[:-1]):
                continue
            if len(path) == 1 and (dr, dc) != outward:
                continue
            if len(path) > 1 and (dr, dc) == (-direction[0], -direction[1]):
                continue
            options.append((nxt, (dr, dc)))
            weights.append(2.0 if (dr, dc) == direction else 1.0)
        if not options:
            break
        choice = int(rng.choice(len(options), p=np.array(weights) / sum(weights)))
        nxt, direction = options[choice]
        path.append(nxt)
    return path


def _candidate_at_end(marker: tuple[int, int], path: list[tuple[int, int]],
                      height: int, width: int) -> tuple[int, int] | None:
    if len(path) < 4:
        return None
    last, before = path[-1], path[-2]
    dr = (last[0] - before[0]) // 2
    dc = (last[1] - before[1]) // 2
    tile = (last[0] + 2 * dr - 1, last[1] + 2 * dc - 1)
    if not (0 <= tile[0] <= height - 3 and 0 <= tile[1] <= width - 3):
        return None
    if not _separated(marker, tile):
        return None
    if any(dot in _cells(*tile) for dot in path):
        return None
    if {dot for dot in _externals(*tile) if dot in path} != {last}:
        return None
    return tile


def _candidate_options(rng: np.random.Generator, height: int, width: int,
                       marker: tuple[int, int], tiles: list[tuple[int, int]],
                       used: set[tuple[int, int]]) -> list[tuple[int, int]]:
    options = []
    for top in range(height - 2):
        for left in range(width - 2):
            tile = (top, left)
            if not all(_separated(tile, old) for old in [marker] + tiles):
                continue
            if _cells(*tile) & used or set(_externals(*tile)) & used:
                continue
            options.append(tile)
    rng.shuffle(options)
    mode = int(rng.integers(0, 4))
    if mode == 0:
        options.sort(key=lambda p: min(p[0], p[1], height - 3 - p[0], width - 3 - p[1]))
    elif mode == 1:
        options.sort(key=lambda p: min(abs(p[0] - old[0]) + abs(p[1] - old[1])
                                       for old in tiles))
    return options


def _decoy_path(rng: np.random.Generator, tile: tuple[int, int],
                height: int, width: int, blocked: set[tuple[int, int]],
                used: set[tuple[int, int]], desired: int) -> list[tuple[int, int]] | None:
    order = [int(i) for i in rng.permutation(4)]
    for side in order:
        start = _externals(*tile)[side]
        if not _inside(start, height, width) or start in blocked or start in used:
            continue
        outward = _DIRECTIONS[side]
        route = _grow_path(rng, start, outward, desired, height, width, blocked, used)
        if len(route) >= 2:
            return route
    return None


def _find_motifs(grid: np.ndarray, background: int) -> tuple[list[tuple], list[tuple]]:
    checkers, candidates = [], []
    height, width = grid.shape
    for row in range(height - 2):
        for col in range(width - 2):
            block = grid[row:row + 3, col:col + 3]
            a, b = int(block[0, 0]), int(block[0, 1])
            if a != b and a != background and b != background and all(
                int(block[y, x]) == (a if (x + y) % 2 == 0 else b)
                for y in range(3) for x in range(3)
            ):
                checkers.append((row, col, a, b))
            perimeter = int(block[0, 0])
            center = int(block[1, 1])
            if (perimeter != background and center != background and center != perimeter
                and all(int(block[y, x]) == perimeter for y in range(3)
                        for x in range(3) if (y, x) != (1, 1))):
                candidates.append((row, col, perimeter, center))
    return checkers, candidates


def _analyze_pair(grid: np.ndarray, output: np.ndarray) -> dict:
    counts = np.bincount(grid.ravel(), minlength=10)
    background = int(np.argmax(counts))
    _require(np.sum(counts == counts[background]) == 1, "background must be unique")
    checkers, candidates = _find_motifs(grid, background)
    _require(len(checkers) == 1, "expected one checkerboard")
    _require(3 <= len(candidates) <= 6, "expected three to six candidates")
    height, width = grid.shape
    marker = checkers[0][:2]
    motif_cells = _cells(*marker)
    for tile in candidates:
        motif_cells.update(_cells(*tile[:2]))
    marker_contacts = [p for p in _externals(*marker) if _inside(p, height, width)
                       and p not in motif_cells and int(grid[p]) != background]
    _require(len(marker_contacts) == 1, "checker needs one side-midpoint guide")
    start = marker_contacts[0]
    guide = int(grid[start])
    guide_nodes = {(int(r), int(c)) for r, c in np.argwhere(grid == guide)
                   if (int(r), int(c)) not in motif_cells
                   and (int(r) - start[0]) % 2 == 0
                   and (int(c) - start[1]) % 2 == 0}
    adjacency = {p: set() for p in guide_nodes}
    for point in guide_nodes:
        for neighbor in _neighbors(point):
            midpoint = ((point[0] + neighbor[0]) // 2,
                        (point[1] + neighbor[1]) // 2)
            if neighbor in guide_nodes and int(grid[midpoint]) == background:
                adjacency[point].add(neighbor)
    component, stack = {start}, [start]
    while stack:
        for neighbor in adjacency[stack.pop()]:
            if neighbor not in component:
                component.add(neighbor)
                stack.append(neighbor)
    _require(len(component) >= 4, "route is too short")
    _require(len(adjacency[start]) == 1, "checker route must start at an endpoint")
    _require(all(len(adjacency[p]) in (1, 2) for p in component),
             "checker-connected route branches")
    endpoints = [p for p in component if len(adjacency[p]) == 1]
    _require(len(endpoints) == 2, "route must have two endpoints")
    end = endpoints[0] if endpoints[1] == start else endpoints[1]
    order = [start]
    previous = None
    while order[-1] != end:
        options = adjacency[order[-1]] - ({previous} if previous is not None else set())
        _require(len(options) == 1, "route traversal is ambiguous")
        previous = order[-1]
        order.append(next(iter(options)))
    _require(len(order) == len(component), "route has a disconnected loop")
    route_hits = []
    contact_counts = []
    contact_details = []
    for index, tile in enumerate(candidates):
        externals = [(side, p) for side, p in enumerate(_externals(*tile[:2]))
                     if _inside(p, height, width) and p not in motif_cells
                     and int(grid[p]) != background]
        contact_counts.append(len(externals))
        for side, point in externals:
            contact_details.append((index, side, point))
            if point in component:
                route_hits.append((index, point))
    _require(all(count >= 1 for count in contact_counts),
             "every candidate needs a dotted approach")
    _require(len(route_hits) == 1 and route_hits[0][1] == end,
             "checker route must end at exactly one candidate side")
    target_index = route_hits[0][0]
    target = candidates[target_index]
    guide_decoy_contacts = sum(
        index != target_index and int(grid[point]) == guide
        and point not in component for index, _, point in contact_details)
    expected = grid[target[0]:target[0] + 3, target[1]:target[1] + 3]
    _require(output.shape == (3, 3) and np.array_equal(output, expected),
             "output must be the reached tile")
    steps = [((b[0] - a[0]) // 2, (b[1] - a[1]) // 2)
             for a, b in zip(order, order[1:])]
    bends = sum(a != b for a, b in zip(steps, steps[1:]))
    decoy_shapes = []
    for index, side, point in contact_details:
        if point == end:
            continue
        color = int(grid[point])
        seen, frontier = {point}, [point]
        while frontier:
            current = frontier.pop()
            for neighbor in _neighbors(current):
                midpoint = ((current[0] + neighbor[0]) // 2,
                            (current[1] + neighbor[1]) // 2)
                if (_inside(neighbor, height, width) and neighbor not in motif_cells
                    and int(grid[neighbor]) == color
                    and int(grid[midpoint]) == background and neighbor not in seen):
                    seen.add(neighbor)
                    frontier.append(neighbor)
        turning_nodes = 0
        for node in seen:
            neighbors = [q for q in _neighbors(node) if q in seen
                         and int(grid[(node[0] + q[0]) // 2,
                                      (node[1] + q[1]) // 2]) == background]
            if (len(neighbors) == 2 and neighbors[0][0] != neighbors[1][0]
                    and neighbors[0][1] != neighbors[1][1]):
                turning_nodes += 1
        decoy_shapes.append((index, side, len(seen), turning_nodes))
    _require(all(any(shape[0] == index and shape[2] >= 2
                     for shape in decoy_shapes)
                 for index in range(len(candidates)) if index != target_index),
             "each decoy needs a dotted path, not an isolated contact")
    return {"background": background, "marker": marker, "target": target[:2],
            "guide": guide, "candidate_count": len(candidates),
            "candidate_positions": tuple(t[:2] for t in candidates),
            "contact_counts": tuple(contact_counts),
            "decoy_shapes": tuple(decoy_shapes), "start": start, "end": end,
            "guide_decoy_contacts": guide_decoy_contacts,
            "bends": bends,
            "route_length": len(order), "shape": grid.shape,
            "checker_colors": checkers[0][2:],
            "candidate_colors": tuple(t[2:] for t in candidates)}


def _generate_pair(rng: np.random.Generator, regime: str,
                   marker_region: int | None, source_like: bool,
                   require_guide_decoy: bool) -> dict:
    for _attempt in range(280):
        height = int(rng.integers(18, 31))
        width = int(rng.integers(18, 31))
        marker = _sample_marker(rng, height, width, marker_region)
        marker_cells = _cells(*marker)
        contacts = [(side, point) for side, point in enumerate(_externals(*marker))
                    if _inside(point, height, width)]
        if not contacts:
            continue
        side, start = contacts[int(rng.integers(len(contacts)))]
        forbidden = marker_cells | (set(_externals(*marker)) - {start})
        if source_like:
            desired = int(rng.integers(6, 11) if regime == "source_train"
                          else rng.integers(12, 21))
            candidate_count = 4 if regime == "source_train" else 5
        else:
            desired = int(rng.integers(5, 17) if regime == "source_train"
                          else rng.integers(5, 21))
            candidate_count = int(rng.integers(3, 6) if regime == "source_train"
                                  else rng.integers(3, 7))
        route = _grow_path(rng, start, _DIRECTIONS[side], desired,
                           height, width, forbidden, set())
        target = _candidate_at_end(marker, route, height, width)
        if target is None:
            continue
        tiles = [target]
        used = set(route)
        paths = []
        success = True
        for _ in range(candidate_count - 1):
            options = _candidate_options(rng, height, width, marker, tiles, used)
            placed = False
            for tile in options[:100]:
                blocked = marker_cells | set().union(*(_cells(*p) for p in tiles + [tile]))
                path = _decoy_path(rng, tile, height, width, blocked,
                                   used | (set(_externals(*marker)) - {start}),
                                   int(rng.integers(2, 13)))
                if path is None:
                    continue
                tiles.append(tile)
                used.update(path)
                paths.append(path)
                placed = True
                break
            if not placed:
                success = False
                break
        if not success:
            continue
        if int(rng.integers(0, 5)) == 0:
            blocked = marker_cells | set().union(*(_cells(*p) for p in tiles))
            extra = _decoy_path(rng, target, height, width, blocked,
                                used | (set(_externals(*marker)) - {start}),
                                int(rng.integers(2, 9)))
            if extra is not None:
                paths.append(extra)
                used.update(extra)
        palette = [int(x) for x in rng.permutation(10)]
        background = palette[0]
        non_background = palette[1:]
        majority, minority, guide = [int(x) for x in rng.choice(
            non_background, size=3, replace=False)]
        pairs = [(a, b) for a in non_background for b in non_background if a != b]
        rng.shuffle(pairs)
        colors = pairs[:candidate_count]
        grid = np.full((height, width), background, dtype=np.int8)
        for point in route:
            grid[point] = guide
        safe_decoys = [index for index, path in enumerate(paths[:candidate_count - 1])
                       if not any(neighbor in route for point in path
                                  for neighbor in _neighbors(point))]
        if require_guide_decoy and not safe_decoys:
            continue
        guide_decoy_index = (int(rng.choice(safe_decoys))
                             if safe_decoys and (require_guide_decoy
                                                 or int(rng.integers(0, 10)) < 7)
                             else None)
        for index, path in enumerate(paths):
            if index == guide_decoy_index or (index in safe_decoys
                                              and int(rng.integers(0, 5)) == 0):
                decoy_color = guide
            else:
                decoy_color = int(rng.choice([c for c in non_background if c != guide]))
            for point in path:
                grid[point] = decoy_color
        top, left = marker
        for y in range(3):
            for x in range(3):
                grid[top + y, left + x] = majority if (x + y) % 2 == 0 else minority
        for tile, (perimeter, center) in zip(tiles, colors):
            top, left = tile
            grid[top:top + 3, left:left + 3] = perimeter
            grid[top + 1, left + 1] = center
        output = grid[target[0]:target[0] + 3, target[1]:target[1] + 3].copy()
        try:
            information = _analyze_pair(grid, output)
        except AssertionError:
            continue
        if require_guide_decoy and information["guide_decoy_contacts"] < 1:
            continue
        return {"input": grid, "output": output}
    raise ValueError("unable to construct a valid dotted-route pair")


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.astype(np.uint8).tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # All puzzles share the source rule. This presentation mode occasionally
    # recreates the source's held-out corner and longer-test progression.
    source_like = bool(rng.integers(0, 4) == 0) and train_count >= 3
    heldout = int(rng.integers(0, 4))
    corner_order = [c for c in range(4) if c != heldout]
    rng.shuffle(corner_order)
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if section == "train" and index < 2:
                region = corner_order[index]
            elif source_like:
                region = (corner_order[index % len(corner_order)]
                          if section == "train" else heldout)
            else:
                region = None if int(rng.integers(0, 10)) == 0 else int(rng.integers(0, 4))
            for _retry in range(40):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                pair = _generate_pair(pair_rng,
                                      "source_train" if section == "train" else "target_test",
                                      region, source_like,
                                      section == "train" and index == 0)
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise ValueError("could not make distinct pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            for role in ("input", "output"):
                grid = pair[role]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional ndarray")
                _require(all(1 <= size <= 30 for size in grid.shape),
                         "grid dimension outside ARC limits")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid color outside ARC range")
            input_grid, output = pair["input"], pair["output"]
            input_key, output_key = _grid_key(input_grid), _grid_key(output)
            _require(input_key not in seen_inputs and (input_key, output_key) not in seen_pairs,
                     "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            _require(hashlib.sha256(input_key[1]).hexdigest() not in _SOURCE_INPUT_HASHES
                     or input_grid.shape != (22, 22), "source input copied")
            _require(input_grid.shape != output.shape, "degenerate equal dimensions")
            info = _analyze_pair(input_grid, output)
            analyses[section].append(info)
    markers = [info["marker"] for info in analyses["train"]]
    _require(len(set(markers)) >= 2, "training marker placements must vary")
    # The only learned convention is the invariant: trace the unique route
    # from a checker and copy the reached tile. Every training pair exhibits it.
    _require(all(info["candidate_count"] >= 3 for info in analyses["train"]),
             "training must show decoy candidates")
    _require(any(info["guide_decoy_contacts"] >= 1 for info in analyses["train"]),
             "training must disprove guide-color contact matching")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k])
        for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    _require(_puzzles_equal(generate_puzzle(917), generate_puzzle(917)),
             "automatic generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(918, 7, 3), generate_puzzle(918, 7, 3)),
             "explicit generation is not deterministic")
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                       (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1200 + i, train, test)
        validate_generated_puzzle(puzzle)
        _require(train is None or len(puzzle["train"]) == train, "train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(1400, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count was accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    signatures, shapes, backgrounds, checker_positions = set(), set(), set(), set()
    bends, lengths, candidate_counts, color_reuse = set(), set(), set(), 0
    edge_tiles, even_grids, contacted_decoys, pair_total = 0, 0, 0, 0
    start_sides, target_sides, decoy_lengths, combinations_seen = set(), set(), set(), set()
    bent_decoys = adjacent_pairs = selected_extra = heldout_longer = 0
    within_shapes = within_lengths = 0
    guide_decoy_pairs = checker_tile_overlap = guide_tile_overlap = 0
    interior_markers = same_quadrant = opposite_quadrant = compact_grids = 0
    overlapping_regimes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_info = {section: [_analyze_pair(pair["input"], pair["output"])
                                 for pair in puzzle[section]] for section in ("train", "test")}
        within_shapes += len({info["shape"] for section in puzzle_info.values()
                              for info in section}) >= 2
        within_lengths += len({info["route_length"] for section in puzzle_info.values()
                               for info in section}) >= 2
        def region(info: dict) -> tuple[bool, bool]:
            row, col = info["marker"]
            height, width = info["shape"]
            return row + 1 >= height / 2, col + 1 >= width / 2
        train_regions = {region(info) for info in puzzle_info["train"]}
        test_regions = {region(info) for info in puzzle_info["test"]}
        overlapping_regimes += bool(
            {info["bends"] for info in puzzle_info["train"]}
            & {info["bends"] for info in puzzle_info["test"]})
        if len(test_regions) == 1 and test_regions.isdisjoint(train_regions):
            heldout_longer += (min(info["route_length"] for info in puzzle_info["test"])
                               > max(info["route_length"] for info in puzzle_info["train"]))
        for section in ("train", "test"):
            for pair, information in zip(puzzle[section], puzzle_info[section]):
                grid = pair["input"]
                info = information
                pair_total += 1
                signatures.add(_grid_key(grid))
                shapes.add(grid.shape)
                backgrounds.add(info["background"])
                checker_positions.add(info["marker"])
                bends.add(info["bends"])
                lengths.add(info["route_length"])
                candidate_counts.add(info["candidate_count"])
                even_grids += int(any(d % 2 == 0 for d in grid.shape))
                contacted_decoys += sum(c >= 1 for c in info["contact_counts"]) - 1
                edge_tiles += sum(r == 0 or c == 0 or r == grid.shape[0] - 3
                                  or c == grid.shape[1] - 3
                                  for r, c in info["candidate_positions"])
                roles = set(info["checker_colors"]) | {info["guide"]}
                color_reuse += int(any(color in roles for pair_colors in info["candidate_colors"]
                                       for color in pair_colors))
                guide_decoy_pairs += info["guide_decoy_contacts"] > 0
                start_sides.add(_externals(*info["marker"]).index(info["start"]))
                target_sides.add(_externals(*info["target"]).index(info["end"]))
                decoy_lengths.update(shape[2] for shape in info["decoy_shapes"])
                bent_decoys += any(shape[3] > 0 for shape in info["decoy_shapes"])
                selected_extra += (info["contact_counts"][info["candidate_positions"].index(
                    info["target"])] > 1)
                adjacent_pairs += any(
                    (abs(a[0] - b[0]) == 3 and abs(a[1] - b[1]) <= 2)
                    or (abs(a[1] - b[1]) == 3 and abs(a[0] - b[0]) <= 2)
                    for a, b in combinations(info["candidate_positions"], 2))
                all_tile_colors = {color for pair_colors in info["candidate_colors"]
                                   for color in pair_colors}
                checker_tile_overlap += bool(set(info["checker_colors"]) & all_tile_colors)
                guide_tile_overlap += info["guide"] in all_tile_colors
                height, width = grid.shape
                marker_row, marker_col = info["marker"]
                target_row, target_col = info["target"]
                interior_markers += (4 <= marker_row <= height - 7
                                     and 4 <= marker_col <= width - 7)
                compact_grids += min(height, width) <= 22
                marker_quadrant = (marker_row + 1 >= height / 2,
                                   marker_col + 1 >= width / 2)
                target_quadrant = (target_row + 1 >= height / 2,
                                   target_col + 1 >= width / 2)
                same_quadrant += marker_quadrant == target_quadrant
                opposite_quadrant += all(a != b for a, b in zip(
                    marker_quadrant, target_quadrant))
                combinations_seen.add((any(d % 2 == 0 for d in grid.shape),
                                       info["guide"] in all_tile_colors,
                                       info["candidate_count"] >= 5,
                                       info["bends"] >= 3))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(signatures) >= 200 and len(shapes) >= 25,
             "grid or size diversity is too low")
    _require(len(backgrounds) >= 8 and len(checker_positions) >= 20,
             "palette or placement diversity is too low")
    _require(len(bends) >= 5 and len(lengths) >= 9 and len(candidate_counts) >= 4,
             "route or candidate diversity is too low")
    _require(even_grids >= pair_total // 3 and edge_tiles >= 10,
             "even canvases or edge tiles are too rare")
    _require(color_reuse >= pair_total // 3, "color role reuse is too rare")
    _require(checker_tile_overlap >= pair_total // 4
             and guide_tile_overlap >= pair_total // 4,
             "checker/tile or guide/tile role reuse is too rare")
    _require(contacted_decoys >= pair_total * 2, "decoy approaches are too rare")
    _require(guide_decoy_pairs >= pair_total // 3,
             "same-guide decoy approaches are too rare")
    _require(start_sides == set(range(4)) and target_sides == set(range(4)),
             "route attachments lack directional diversity")
    _require(max(decoy_lengths) >= 8 and len(decoy_lengths) >= 7
             and bent_decoys >= 30, "decoy route geometry is too narrow")
    _require(adjacent_pairs >= 10 and selected_extra >= 5,
             "close candidates or extra selected contacts are too rare")
    _require(heldout_longer >= 3 and len(combinations_seen) >= 10,
             "source-like shifts or combined variation are too rare")
    _require(within_shapes == len(puzzles) and within_lengths >= 35,
             "within-puzzle shape or route variation is too low")
    _require(interior_markers >= 5 and compact_grids >= pair_total // 4,
             "interior markers or compact grids are too rare")
    _require(same_quadrant >= 10 and opposite_quadrant >= 10
             and overlapping_regimes >= 8,
             "target locations or train/test overlap are too narrow")
