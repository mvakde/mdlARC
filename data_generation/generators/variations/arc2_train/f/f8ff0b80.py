import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "000000000000", "000000011010", "000000011100", "000000000110",
        "000000001000", "000000000000", "000000000000", "000200000000",
        "002200000070", "000200007770", "000000000770", "000000000000",
    ),
    (
        "000000000000", "003300000000", "003300000800", "003330008800",
        "030000000080", "000000000000", "000022000000", "000222200000",
        "000020000000", "000000000000", "000000000000", "000000000000",
    ),
    (
        "000000000000", "000200000000", "002220001000", "022200001100",
        "000000001100", "000000000000", "000000000000", "000004440000",
        "000444440000", "000444440000", "000040000000", "000000000000",
    ),
    (
        "000000000000", "003000000000", "033000000000", "000000000000",
        "000000000000", "000011000000", "000111000000", "000000000000",
        "000000060000", "000000660000", "000006066000", "000000060000",
    ),
)
_SOURCE_OUTPUT_ROWS = (
    ("1", "7", "2"), ("3", "2", "8"), ("4", "2", "1"), ("6", "1", "3"),
)
_SOURCE_PAIRS = tuple(
    (
        np.array([[int(c) for c in row] for row in source_input], dtype=np.int8),
        np.array([[int(c) for c in row] for row in source_output], dtype=np.int8),
    )
    for source_input, source_output in zip(_SOURCE_INPUT_ROWS, _SOURCE_OUTPUT_ROWS)
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _resolve_counts(seed, num_train, num_test):
    def checked(name, value, lower, upper):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer or None")
        value = int(value)
        if not lower <= value <= upper:
            raise ValueError(f"{name} must be between {lower} and {upper}")
        return value

    train_count = checked("num_train", num_train, 2, 10)
    test_count = checked("num_test", num_test, 1, 4)
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum = test_count + 1
        train_count = minimum + (seed // 4) % (11 - minimum)
    elif train_count is None:
        minimum = test_count + 1
        train_count = minimum + seed % (11 - minimum)
    elif test_count is None:
        test_count = 1 + seed % min(4, train_count - 1)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def _bbox(grid, color):
    positions = np.argwhere(grid == color)
    return (
        int(positions[:, 0].min()), int(positions[:, 0].max()),
        int(positions[:, 1].min()), int(positions[:, 1].max()),
    )


def _box_area(box):
    return (box[1] - box[0] + 1) * (box[3] - box[2] + 1)


def _ranked_colors(input_grid):
    colors, counts = np.unique(input_grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    foreground = [int(c) for c in colors if int(c) != background]
    populations = {c: int(np.count_nonzero(input_grid == c)) for c in foreground}
    return background, populations, tuple(sorted(foreground, key=lambda c: -populations[c]))


def _pair_from_grid(input_grid):
    _, _, order = _ranked_colors(input_grid)
    return {
        "input": input_grid,
        "output": np.array(order, dtype=np.int8).reshape(3, 1),
    }


def _dimensions(rng, index):
    mode = (index + int(rng.integers(0, 4))) % 4
    if mode == 0:
        return int(rng.integers(8, 13)), int(rng.integers(8, 13))
    if mode == 1:
        return int(rng.integers(12, 23)), int(rng.integers(12, 23))
    if mode == 2:
        return int(rng.integers(23, 31)), int(rng.integers(23, 31))
    if rng.integers(0, 2):
        return int(rng.integers(8, 14)), int(rng.integers(21, 31))
    return int(rng.integers(21, 31)), int(rng.integers(8, 14))


def _zones(rng, height, width, layout):
    if layout == "horizontal":
        cut1 = width // 3
        cut2 = 2 * width // 3
        return [(0, height - 1, 0, cut1 - 1),
                (0, height - 1, cut1, cut2 - 1),
                (0, height - 1, cut2, width - 1)]
    if layout == "vertical":
        cut1 = height // 3
        cut2 = 2 * height // 3
        return [(0, cut1 - 1, 0, width - 1),
                (cut1, cut2 - 1, 0, width - 1),
                (cut2, height - 1, 0, width - 1)]
    if layout == "diagonal":
        cut1 = height // 3
        cut2 = 2 * height // 3
        span = max(3, 2 * width // 3)
        return [(0, cut1 - 1, 0, span - 1),
                (cut1, cut2 - 1, (width - span) // 2,
                 (width - span) // 2 + span - 1),
                (cut2, height - 1, width - span, width - 1)]
    row_cut = height // 2
    col_cut = width // 2
    quadrants = [
        (0, row_cut - 1, 0, col_cut - 1),
        (0, row_cut - 1, col_cut, width - 1),
        (row_cut, height - 1, 0, col_cut - 1),
        (row_cut, height - 1, col_cut, width - 1),
    ]
    omitted = int(rng.integers(0, 4))
    return [box for index, box in enumerate(quadrants) if index != omitted]


def _shape(rng, zone, count, height, width, diagonal=False, separate=False):
    r0, r1, c0, c1 = zone
    available = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    if rng.random() < 0.32:
        edge = [(r, c) for r, c in available
                if r in (0, height - 1) or c in (0, width - 1)]
        start_pool = edge if edge else available
    else:
        start_pool = available
    selected = {start_pool[int(rng.integers(0, len(start_pool)))]}
    growth_target = count - 1 if separate and count >= 4 else count
    while len(selected) < growth_target:
        candidates = []
        weights = []
        for point in available:
            if point in selected:
                continue
            r, c = point
            orthogonal = 0
            diagonal_neighbors = 0
            for sr, sc in selected:
                dr, dc = abs(r - sr), abs(c - sc)
                if dr + dc == 1:
                    orthogonal += 1
                elif dr == 1 and dc == 1:
                    diagonal_neighbors += 1
            if orthogonal or diagonal_neighbors:
                candidates.append(point)
                weights.append(
                    1 + (7 * diagonal_neighbors + orthogonal if diagonal
                         else 6 * orthogonal + diagonal_neighbors)
                )
        probabilities = np.array(weights, dtype=float)
        probabilities /= probabilities.sum()
        selected.add(candidates[int(rng.choice(len(candidates), p=probabilities))])
    if growth_target < count:
        separated = [
            point for point in available if point not in selected
            and min(max(abs(point[0] - r), abs(point[1] - c))
                    for r, c in selected) == 2
        ]
        if separated:
            selected.add(separated[int(rng.integers(0, len(separated)))])
        else:
            # A saturated small zone cannot hold a detached cell.
            return _shape(rng, zone, count, height, width, diagonal, False)
    return selected


def _inset_outer_edges(zone, height, width):
    r0, r1, c0, c1 = zone
    if r1 - r0 + 1 >= 3:
        r0 += int(r0 == 0)
        r1 -= int(r1 == height - 1)
    if c1 - c0 + 1 >= 3:
        c0 += int(c0 == 0)
        c1 -= int(c1 == width - 1)
    return r0, r1, c0, c1


def _place(grid, color, cells, background):
    for row, col in cells:
        _require(grid[row, col] == background,
                 "overlapping figures")
        grid[row, col] = color


def _special_training_pair(rng, colors, background):
    height = int(rng.integers(13, 25))
    width = int(rng.integers(13, 25))
    grid = np.full((height, width), background, dtype=np.int8)
    zones = _zones(rng, height, width, "corners")
    rng.shuffle(zones)
    sparse = {(i, i) for i in range(5)}
    sparse.update(((0, 1), (1, 0), (3, 4)))
    dense = {(r, c) for r in range(3) for c in range(3)}
    small_count = int(rng.integers(1, 6))
    local_small = _shape(rng, (0, 2, 0, 2), small_count, 3, 3,
                         diagonal=True)
    shapes = [sparse, dense, local_small]
    rng.shuffle(shapes)
    for color, zone, shape in zip(colors, zones, shapes):
        r0, r1, c0, c1 = zone
        side_r = max(r for r, _ in shape) + 1
        side_c = max(c for _, c in shape) + 1
        offset_r = int(rng.integers(r0, r1 - side_r + 2))
        offset_c = int(rng.integers(c0, c1 - side_c + 2))
        _place(grid, color, {(r + offset_r, c + offset_c) for r, c in shape},
               background)
    return _pair_from_grid(grid)


def _general_pair(rng, colors, background, index, forced_order=None):
    height, width = _dimensions(rng, index)
    layout = ("horizontal", "vertical", "corners", "diagonal")[
        int(rng.integers(0, 4))
    ]
    zones = _zones(rng, height, width, layout)
    rng.shuffle(zones)
    if rng.random() < 0.65:
        zones = [_inset_outer_edges(zone, height, width) for zone in zones]
    area_cap = min((r1 - r0 + 1) * (c1 - c0 + 1)
                   for r0, r1, c0, c1 in zones)
    cap = min(26, area_cap, max(6, height * width // 6))
    count_candidates = np.arange(1, cap + 1)
    if cap >= 18 and rng.random() < 0.43:
        largest = int(rng.integers(max(18, cap - 5), cap + 1))
        smaller = rng.choice(np.arange(1, largest), size=2, replace=False)
        counts = sorted((largest, int(smaller[0]), int(smaller[1])), reverse=True)
    else:
        counts = sorted((int(x) for x in rng.choice(count_candidates, size=3,
                                                     replace=False)), reverse=True)
    if forced_order is None:
        order = [int(c) for c in rng.permutation(colors)]
    else:
        order = list(forced_order)
    color_counts = dict(zip(order, counts))
    grid = np.full((height, width), background, dtype=np.int8)
    for color, zone in zip(colors, zones):
        diagonal = bool(rng.random() < 0.35)
        separate = bool(rng.random() < 0.08)
        cells = _shape(rng, zone, color_counts[color], height, width,
                       diagonal=diagonal, separate=separate)
        _place(grid, color, cells, background)
    return _pair_from_grid(grid)


def _nested_pair(rng, colors, background):
    height = int(rng.integers(15, 31))
    width = int(rng.integers(15, 31))
    grid = np.full((height, width), background, dtype=np.int8)
    side = int(rng.integers(5, 8))
    if rng.integers(0, 2):
        top = int(rng.integers(0, height - side + 1))
        left = int(rng.integers(0, max(1, width // 2 - side + 1)))
        third_zone = (0, height - 1, width // 2, width - 1)
    else:
        top = int(rng.integers(0, height - side + 1))
        left = int(rng.integers(width // 2, width - side + 1))
        third_zone = (0, height - 1, 0, width // 2 - 1)
    outer, inner, third = [int(c) for c in rng.permutation(colors)]
    perimeter = {
        (top + r, left + c) for r in range(side) for c in range(side)
        if r in (0, side - 1) or c in (0, side - 1)
    }
    _place(grid, outer, perimeter, background)
    inner_count = int(rng.integers(3, min(10, (side - 2) ** 2 + 1)))
    inner_cells = _shape(rng, (top + 1, top + side - 2,
                               left + 1, left + side - 2), inner_count,
                         height, width, diagonal=True)
    _place(grid, inner, inner_cells, background)
    third_limit = min(25, (third_zone[3] - third_zone[2] + 1)
                      * height // 5)
    third_options = [count for count in range(1, third_limit + 1)
                     if count not in (len(perimeter), inner_count)]
    third_count = third_options[int(rng.integers(0, len(third_options)))]
    if rng.random() < 0.65:
        third_zone = _inset_outer_edges(third_zone, height, width)
    third_cells = _shape(rng, third_zone, third_count, height, width,
                         separate=bool(rng.random() < 0.15))
    _place(grid, third, third_cells, background)
    return _pair_from_grid(grid)


def _choose_background(rng, train_pool, colors, previous=None):
    available = [c for c in train_pool if c not in colors and c != previous]
    if not available:
        available = [c for c in train_pool if c not in colors]
    return int(available[int(rng.integers(0, len(available)))])


def _test_colors(rng, core, novel, strategy):
    familiar_count = 3 - strategy
    familiar = list(rng.choice(core, size=familiar_count, replace=False))
    unseen = list(rng.choice(novel, size=strategy, replace=False))
    colors = [int(c) for c in familiar + unseen]
    rng.shuffle(colors)
    return colors


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(np.arange(10))]
    core = palette[:3]
    train_pool = palette[:6] + palette[9:]
    novel = palette[6:9]
    puzzle = {"train": [], "test": []}
    first_background = _choose_background(rng, train_pool, core)
    first = _special_training_pair(rng, core, first_background)
    puzzle["train"].append(first)
    first_order = tuple(int(c) for c in first["output"][:, 0])
    alternatives = [
        order for order in (
            (core[0], core[1], core[2]), (core[0], core[2], core[1]),
            (core[1], core[0], core[2]), (core[1], core[2], core[0]),
            (core[2], core[0], core[1]), (core[2], core[1], core[0]),
        ) if order != first_order
    ]
    second_order = alternatives[int(rng.integers(0, len(alternatives)))]
    background = _choose_background(rng, train_pool, core, first_background)
    puzzle["train"].append(_general_pair(rng, core, background, 1,
                                          forced_order=second_order))
    previous_background = background
    for index in range(2, train_count):
        core_count = int(rng.integers(0, 4))
        selected = list(rng.choice(core, size=core_count, replace=False))
        selected += list(rng.choice(palette[3:6], size=3 - core_count,
                                    replace=False))
        colors = [int(c) for c in selected]
        rng.shuffle(colors)
        background = _choose_background(rng, train_pool, colors,
                                        previous_background)
        if index % 4 == 3:
            pair = _nested_pair(rng, colors, background)
        else:
            pair = _general_pair(rng, colors, background, index)
        puzzle["train"].append(pair)
        previous_background = background
    for index in range(test_count):
        strategy = (seed + index) % 4
        colors = _test_colors(rng, core, novel, strategy)
        background = _choose_background(rng, train_pool, colors,
                                        previous_background)
        if (seed + index) % 5 == 0:
            pair = _nested_pair(rng, colors, background)
        else:
            pair = _general_pair(rng, colors, background, train_count + index)
        puzzle["test"].append(pair)
        previous_background = background
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair must contain exactly input and output")
    for role in ("input", "output"):
        grid = pair[role]
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                 "grids must be two-dimensional NumPy arrays")
        _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
        _require(all(1 <= n <= 30 for n in grid.shape), "grid size must be 1..30")
        _require(bool(np.all((grid >= 0) & (grid <= 9))), "colors must be 0..9")
    inp, out = pair["input"], pair["output"]
    _require(out.shape == (3, 1), "output must be a three-cell column")
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) == 4, "input must contain background and three colors")
    background, populations, order = _ranked_colors(inp)
    _require(int(counts[np.argmax(counts)]) > max(populations.values()),
             "background must be uniquely dominant")
    _require(len(set(populations.values())) == 3,
             "all foreground counts must be distinct")
    _require(tuple(int(c) for c in out[:, 0]) == order,
             "output must rank foreground colors by occupied-cell count")
    boxes = {color: _bbox(inp, color) for color in populations}
    return background, populations, order, boxes


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    analyses = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _analyze_pair(pair)
            input_key = _key(pair["input"])
            pair_key = input_key, _key(pair["output"])
            _require(input_key not in seen_inputs, "duplicate input")
            _require(pair_key not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not any(
                np.array_equal(pair["input"], source_input)
                and np.array_equal(pair["output"], source_output)
                for source_input, source_output in _SOURCE_PAIRS
            ), "source pair copied")
            analyses.append(analysis)
    first, second = analyses[:2]
    _require(set(first[1]) == set(second[1]),
             "first two demonstrations must reuse the same foreground colors")
    _require(first[2] != second[2],
             "training must change the rank of reused colors")
    first_counts = first[1]
    first_areas = {c: _box_area(first[3][c]) for c in first_counts}
    _require(any(
        first_counts[a] > first_counts[b] and first_areas[a] < first_areas[b]
        for a in first_counts for b in first_counts
    ), "training must refute bounding-box ranking")
    # The only learned convention is the transformation checked in _analyze_pair.
    # Each test input directly displays every foreground identity and its count.


def _puzzle_key(puzzle):
    return tuple(
        tuple((_key(pair["input"]), _key(pair["output"]))
              for pair in puzzle[split])
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts = set()
    test_counts = set()
    shapes = set()
    backgrounds = set()
    palettes = set()
    orders = set()
    structure = set()
    gap_one = 0
    edge_count = 0
    multi_piece = 0
    eight_piece = 0
    bbox_overlap = 0
    large_count = 0
    tiny_count = 0
    small_canvas = 0
    large_canvas = 0
    elongated_canvas = 0
    varying_background_puzzles = 0
    varying_layout_puzzles = 0
    changed_test_order_puzzles = 0
    unseen_ranks = set()
    novelty_counts = set()
    largest_test_positions = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)),
                 "generation must be deterministic")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = set()
        for pair in puzzle["train"]:
            train_colors.update(_analyze_pair(pair)[1])
        pair_backgrounds = set()
        pair_layouts = set()
        test_orders = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                background, populations, order, boxes = _analyze_pair(pair)
                expected = tuple(sorted(populations, key=lambda c: -populations[c]))
                _require(order == expected, "incorrect count order")
                _require(np.array_equal(pair["output"][:, 0], np.array(expected)),
                         "ground truth mismatch")
                pair_backgrounds.add(background)
                backgrounds.add(background)
                shapes.add(inp.shape)
                palettes.add(tuple(sorted(populations)))
                orders.add(order)
                small_canvas += int(inp.shape[0] <= 12 and inp.shape[1] <= 12)
                large_canvas += int(inp.shape[0] >= 23 and inp.shape[1] >= 23)
                elongated_canvas += int(
                    max(inp.shape) >= 2 * min(inp.shape)
                )
                counts = sorted(populations.values())
                gap_one += int(any(b - a == 1 for a, b in zip(counts, counts[1:])))
                tiny_count += int(min(counts) <= 2)
                large_count += int(max(counts) >= 18)
                edge_count += int(any(
                    np.any(inp[0] == c) or np.any(inp[-1] == c)
                    or np.any(inp[:, 0] == c) or np.any(inp[:, -1] == c)
                    for c in populations
                ))
                bbox_overlap += int(any(
                    max(boxes[a][0], boxes[b][0]) <= min(boxes[a][1], boxes[b][1])
                    and max(boxes[a][2], boxes[b][2]) <= min(boxes[a][3], boxes[b][3])
                    for a in populations for b in populations if a < b
                ))
                layout = tuple(sorted(
                    (boxes[c][0] * 3 // inp.shape[0],
                     boxes[c][2] * 3 // inp.shape[1])
                    for c in populations
                ))
                pair_layouts.add(layout)
                structure.add((
                    inp.shape,
                    tuple(sorted(populations.values())),
                    tuple(sorted(_box_area(boxes[c]) for c in populations)),
                    layout,
                ))
                for color in populations:
                    cells = set(zip(*np.nonzero(inp == color)))
                    frontier = [next(iter(cells))]
                    reached = set(frontier)
                    while frontier:
                        r, c = frontier.pop()
                        for neighbor in ((r - 1, c), (r + 1, c),
                                         (r, c - 1), (r, c + 1)):
                            if neighbor in cells and neighbor not in reached:
                                reached.add(neighbor)
                                frontier.append(neighbor)
                    multi_piece += int(len(reached) != len(cells))
                    frontier = [next(iter(cells))]
                    reached = set(frontier)
                    while frontier:
                        r, c = frontier.pop()
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                neighbor = (r + dr, c + dc)
                                if (dr or dc) and neighbor in cells and neighbor not in reached:
                                    reached.add(neighbor)
                                    frontier.append(neighbor)
                    eight_piece += int(len(reached) != len(cells))
                if split == "test":
                    test_orders.add(order)
                    novelty_counts.add(sum(c not in train_colors for c in populations))
                    top = min(populations, key=lambda c: boxes[c][0] + boxes[c][1])
                    bottom = max(populations, key=lambda c: boxes[c][0] + boxes[c][1])
                    if order[0] == top:
                        largest_test_positions.add("top")
                    if order[0] == bottom:
                        largest_test_positions.add("bottom")
                    if order[0] not in (top, bottom):
                        largest_test_positions.add("middle")
                    for rank, color in enumerate(order):
                        if color not in train_colors:
                            unseen_ranks.add(rank)
        varying_background_puzzles += int(len(pair_backgrounds) > 1)
        varying_layout_puzzles += int(len(pair_layouts) > 1)
        changed_test_order_puzzles += int(len(test_orders) > 1)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(shapes) >= 70 and len(structure) >= 150,
             "grid shapes or structural signatures lack diversity")
    _require(len(backgrounds) == 10 and len(palettes) >= 30 and len(orders) >= 40,
             "color assignment lacks diversity")
    _require(gap_one >= 20 and tiny_count >= 20 and large_count >= 20,
             "count difficulty lacks diversity")
    _require(edge_count >= 10 and multi_piece >= 20 and eight_piece >= 5
             and bbox_overlap >= 5,
             "figure geometry lacks diversity")
    _require(small_canvas >= 5 and large_canvas >= 5 and elongated_canvas >= 5,
             "canvas scale or aspect ratio lacks diversity")
    _require(varying_background_puzzles >= 30,
             "background role does not vary within enough puzzles")
    _require(varying_layout_puzzles >= 30
             and largest_test_positions == {"top", "middle", "bottom"},
             "location and size rank are too correlated")
    _require(changed_test_order_puzzles >= 1 and unseen_ranks == {0, 1, 2},
             "test palettes or output orders lack diversity")
    _require(novelty_counts == {0, 1, 2, 3},
             "test color novelty lacks diversity")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4),
                                    (2, None), (10, None)):
        puzzle = generate_puzzle(1001, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "ignored train count")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "ignored test count")
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(1001, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count request accepted")
    _require(_puzzle_key(generate_puzzle(1001)) != _puzzle_key(generate_puzzle(1002)),
             "different seeds should differ")
