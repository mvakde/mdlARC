"""Generate puzzles that turn colored cell counts into concentric layers."""

import numpy as np


_SOURCE_PAIRS = (
    (("777777", "777447", "777447", "677777", "667777", "777777"),
     ("444", "464", "444")),
    (("777777877", "777777877", "755778887", "755777777", "777777777",
      "777111777", "777171777", "777111777", "777777777"),
     ("11111", "18881", "18581", "18881", "11111")),
    (("7777777", "7888733", "7878777", "7888777", "7777797", "0077777",
      "0077777"),
     ("8888888", "8000008", "8033308", "8039308", "8033308", "8000008",
      "8888888")),
    (("77977222", "77977272", "57777777", "55787777", "57777077",
      "77177007", "71117700", "11111770"),
     ("11111111111", "10000000001", "10222222201", "10255555201",
      "10259995201", "10259895201", "10259995201", "10255555201",
      "10222222201", "10000000001", "11111111111")),
)


def _count_arg(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _pair_counts(seed, rng, num_train, num_test):
    train = _count_arg("num_train", num_train, 2, 10)
    test = _count_arg("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _normalize(cells):
    top = min(row for row, _ in cells)
    left = min(col for _, col in cells)
    return tuple(sorted((row - top, col - left) for row, col in cells))


def _orient(rng, cells):
    transformed = list(cells)
    for _ in range(int(rng.integers(4))):
        transformed = [(col, -row) for row, col in transformed]
    if int(rng.integers(2)):
        transformed = [(row, -col) for row, col in transformed]
    return _normalize(transformed)


def _polyomino(rng, area, mode):
    cells = {(0, 0)}
    tip = (0, 0)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while len(cells) < area:
        frontier = sorted({(r + dr, c + dc) for r, c in cells
                           for dr, dc in directions if (r + dr, c + dc) not in cells})
        if mode == "blob":
            scores = [sum((r + dr, c + dc) in cells for dr, dc in directions)
                      for r, c in frontier]
            best = max(scores)
            candidates = [point for point, score in zip(frontier, scores) if score == best]
        elif mode == "path":
            candidates = [(tip[0] + dr, tip[1] + dc) for dr, dc in directions
                          if (tip[0] + dr, tip[1] + dc) not in cells]
            if not candidates:
                candidates = frontier
        else:
            candidates = frontier
        tip = _choice(rng, candidates)
        cells.add(tip)
    return _orient(rng, cells)


def _compact(rng, area):
    width = int(np.ceil(np.sqrt(area)))
    cells = [(index // width, index % width) for index in range(area)]
    return _orient(rng, cells)


def _elbow(rng, area):
    horizontal = (area + 1) // 2
    vertical = area + 1 - horizontal
    cells = [(0, col) for col in range(horizontal)]
    cells += [(row, 0) for row in range(1, vertical)]
    return _orient(rng, cells)


def _branch(rng, area):
    if area < 5:
        return _polyomino(rng, area, "random")
    arms = [0, 0, 0, 0]
    for _ in range(area - 1):
        arms[int(rng.integers(4))] += 1
    cells = {(0, 0)}
    for length, (dr, dc) in zip(arms, ((1, 0), (-1, 0), (0, 1), (0, -1))):
        cells.update((step * dr, step * dc) for step in range(1, length + 1))
    return _orient(rng, cells)


def _ring(rng, area, open_ring=False):
    cells = {(r, c) for r in range(3) for c in range(3)
             if r in (0, 2) or c in (0, 2)}
    forbidden = {(1, 1)}
    if open_ring:
        cells.remove((0, 1))
        forbidden.add((0, 1))
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while len(cells) < area:
        frontier = sorted({(r + dr, c + dc) for r, c in cells
                           for dr, dc in directions
                           if (r + dr, c + dc) not in cells
                           and (r + dr, c + dc) not in forbidden})
        cells.add(_choice(rng, frontier))
    return _orient(rng, cells)


def _shape(rng, area, style):
    if style == "compact":
        return _compact(rng, area)
    if style == "elbow":
        return _elbow(rng, area)
    if style == "ring" and area >= 8:
        return _ring(rng, area)
    if style == "open" and area >= 7:
        return _ring(rng, area, True)
    if style == "branch":
        return _branch(rng, area)
    return _polyomino(rng, area, style)


def _areas_and_shapes(rng, count, lesson):
    if lesson == "extent":
        largest = int(rng.integers(12, 17))
        second = int(_choice(rng, (9, 11)))
        available = [value for value in range(1, second) if value not in (largest, second)]
        remainder = [int(v) for v in rng.choice(available, size=count - 2, replace=False)]
        areas = sorted([largest, second] + remainder, reverse=True)
        styles = ["compact", "elbow"] + [None] * (count - 2)
    elif lesson == "ring_open":
        available = [value for value in range(1, 21) if value not in (7, 8)]
        remainder = [int(v) for v in rng.choice(available, size=count - 2, replace=False)]
        areas = sorted([8, 7] + remainder, reverse=True)
        styles = [None] * count
        styles[areas.index(8)] = "ring"
        styles[areas.index(7)] = "open"
    else:
        upper = int(_choice(rng, (count + 4, max(14, count + 7), 23)))
        areas = sorted((int(v) for v in rng.choice(np.arange(1, upper + 1),
                                                    size=count, replace=False)), reverse=True)
        styles = [None] * count
    shape_styles = ("random", "blob", "path", "compact", "elbow", "branch", "ring", "open")
    shapes = []
    for area, style in zip(areas, styles):
        if style is None:
            style = _choice(rng, shape_styles)
        shapes.append(_shape(rng, area, style))
    return areas, shapes


def _select_colors(rng, palette, count, previous, overlap):
    if previous is None:
        return [int(v) for v in rng.choice(palette, size=count, replace=False)]
    prior = [int(v) for v in previous]
    outside = [int(v) for v in palette if v not in previous]
    minimum = max(0, count - len(outside))
    maximum = min(count, len(prior))
    if overlap == "none":
        shared_count = minimum
    elif overlap == "substantial":
        shared_count = maximum
    else:
        shared_count = (minimum + maximum) // 2
    shared = [int(v) for v in rng.choice(prior, size=shared_count, replace=False)]
    fresh = [int(v) for v in rng.choice(outside, size=count - shared_count, replace=False)]
    return shared + fresh


def _rank_colors(rng, colors, reverse_pair=None):
    ranked = [int(v) for v in rng.permutation(colors)]
    if reverse_pair is not None and all(color in ranked for color in reverse_pair):
        a, b, previous_a_before_b = reverse_pair
        if (ranked.index(a) < ranked.index(b)) == previous_a_before_b:
            ia, ib = ranked.index(a), ranked.index(b)
            ranked[ia], ranked[ib] = ranked[ib], ranked[ia]
    return ranked


def _dimension_candidates(shapes, total_area, output_side, density, relation,
                          aspect, outer_margin):
    minimum_height = max(max(r for r, _ in shape) + 1 for shape in shapes) + 2 * outer_margin
    minimum_width = max(max(c for _, c in shape) + 1 for shape in shapes) + 2 * outer_margin
    target_factor = {"crowded": 2.5, "mixed": 3.8, "spacious": 6.0}[density]
    target_area = total_area * target_factor
    candidates = []
    for height in range(max(3, minimum_height), 31):
        for width in range(max(3, minimum_width), 31):
            if height * width <= 2 * total_area:
                continue
            if relation == "output_larger" and not (height < output_side and width < output_side):
                continue
            if relation == "equal_axis" and not (height == output_side or width == output_side):
                continue
            if relation == "input_larger" and not (height > output_side or width > output_side):
                continue
            area_error = abs(np.log(height * width / target_area))
            aspect_error = abs(np.log((width / height) / aspect))
            candidates.append((area_error + 0.5 * aspect_error, height, width))
    candidates.sort()
    return [(height, width) for _, height, width in candidates[:100]]


def _touches(grid, rows, cols, background, gap):
    height, width = grid.shape
    for row, col in zip(rows, cols):
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                rr, cc = row + dr, col + dc
                if 0 <= rr < height and 0 <= cc < width and grid[rr, cc] != background:
                    return True
    return False


def _orthogonal_contact(grid, rows, cols, background):
    height, width = grid.shape
    for row, col in zip(rows, cols):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = row + dr, col + dc
            if 0 <= rr < height and 0 <= cc < width and grid[rr, cc] != background:
                return True
    return False


def _pack(rng, height, width, shapes, colors, background, density, outer_margin):
    require_contact = density == "crowded"
    gap = 1 if density == "spacious" else 0
    arrays = [np.array(shape, dtype=np.int16) for shape in shapes]
    for _ in range(5):
        grid = np.full((height, width), background, dtype=np.int8)
        success = True
        for index, (points, color) in enumerate(zip(arrays, colors)):
            shape_height = int(points[:, 0].max()) + 1
            shape_width = int(points[:, 1].max()) + 1
            if shape_height + 2 * outer_margin > height or shape_width + 2 * outer_margin > width:
                success = False
                break
            placed = False
            for _ in range(280):
                top = outer_margin + int(rng.integers(height - shape_height - 2 * outer_margin + 1))
                left = outer_margin + int(rng.integers(width - shape_width - 2 * outer_margin + 1))
                rows, cols = points[:, 0] + top, points[:, 1] + left
                if np.any(grid[rows, cols] != background):
                    continue
                if gap and _touches(grid, rows, cols, background, gap):
                    continue
                if require_contact and index == 1 and not _orthogonal_contact(
                    grid, rows, cols, background
                ):
                    continue
                grid[rows, cols] = color
                placed = True
                break
            if not placed:
                success = False
                break
        if success:
            return grid
    return None


def _build_output(ranked):
    side = 2 * len(ranked) - 1
    result = np.empty((side, side), dtype=np.int8)
    for index, color in enumerate(ranked):
        result[index:side - index, index:side - index] = color
    return result


def _generate_pair(rng, palette, count, previous, previous_background, lesson,
                   overlap, reverse_pair):
    colors = _select_colors(rng, palette, count, previous, overlap)
    ranked = _rank_colors(rng, colors, reverse_pair)
    available_backgrounds = [int(v) for v in palette if v not in colors]
    if previous_background in available_backgrounds and rng.random() < 0.35:
        background = int(previous_background)
    else:
        background = int(_choice(rng, available_backgrounds))
    areas, shapes = _areas_and_shapes(rng, count, lesson)
    output = _build_output(ranked)
    density = _choice(rng, ("crowded", "mixed", "spacious"))
    relation = _choice(rng, ("output_larger", "equal_axis", "input_larger", "free"))
    aspect = float(_choice(rng, (0.5, 0.7, 1.0, 1.4, 2.0)))
    outer_margin = int(_choice(rng, (0, 0, 1, 1, 2)))
    total_area = sum(areas)
    for candidate_relation in (relation, "free"):
        candidates = _dimension_candidates(shapes, total_area, output.shape[0],
                                            density, candidate_relation, aspect,
                                            outer_margin)
        if candidates:
            order = [int(v) for v in rng.permutation(min(len(candidates), 35))]
            order += list(range(35, min(len(candidates), 100)))
            for index in order:
                height, width = candidates[index]
                grid = _pack(rng, height, width, shapes, ranked, background,
                             density, outer_margin)
                if grid is not None:
                    return grid, output
    # A larger sparse canvas is a deterministic construction fallback.
    for height, width in ((24, 24), (30, 30)):
        grid = _pack(rng, height, width, shapes, ranked, background, "mixed",
                     outer_margin)
        if grid is not None:
            return grid, output
    raise RuntimeError("Could not place connected colored tokens")


def _rows(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _source_copy(inp, out):
    return (_rows(inp), _rows(out)) in _SOURCE_PAIRS


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _analyze(inp):
    colors, counts = np.unique(inp, return_counts=True)
    order = np.argsort(-counts)
    background = int(colors[order[0]])
    background_area = int(counts[order[0]])
    ranked = sorted(((int(color), int(count)) for color, count in zip(colors, counts)
                     if int(color) != background), key=lambda item: -item[1])
    return background, background_area, ranked


def _connected(mask):
    points = set(zip(*np.nonzero(mask)))
    if not points:
        return False
    visited = {next(iter(points))}
    frontier = list(visited)
    while frontier:
        row, col = frontier.pop()
        for point in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
            if point in points and point not in visited:
                visited.add(point)
                frontier.append(point)
    return len(visited) == len(points)


def _bbox_area(mask):
    rows, cols = np.nonzero(mask)
    return (int(rows.max() - rows.min()) + 1) * (int(cols.max() - cols.min()) + 1)


def _extent_witness(inp, ranked):
    if len(ranked) < 2:
        return False
    largest_area = ranked[0][1]
    largest_box = _bbox_area(inp == ranked[0][0])
    return any(area < largest_area and _bbox_area(inp == color) > largest_box
               for color, area in ranked[1:])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(seed, rng, num_train, num_test)
    # The counting and concentric-layer convention is shared. Literal colors are
    # pair-local roles, since the source changes their rank with their area.
    palette = [int(v) for v in rng.permutation(10)]
    train_counts = [int(rng.integers(2, 4)), int(rng.integers(4, 8))]
    train_counts += [int(rng.integers(2, 8)) for _ in range(train_count - 2)]
    train_counts = [int(v) for v in rng.permutation(train_counts)]
    test_counts = []
    for _ in range(test_count):
        mode = int(rng.integers(3))
        if mode == 0 and max(train_counts) < 9:
            count = int(rng.integers(max(train_counts) + 1, 10))
        elif mode == 1:
            count = int(_choice(rng, train_counts))
        else:
            count = int(rng.integers(2, 10))
        test_counts.append(count)
    child_seeds = [int(v) for v in rng.integers(0, 2**63 - 1,
                                               size=train_count + test_count)]
    puzzle = {"train": [], "test": []}
    previous = None
    previous_background = None
    first_rank = None
    seen_inputs = set()
    for pair_index, (section, count) in enumerate(
        [("train", count) for count in train_counts] +
        [("test", count) for count in test_counts]
    ):
        pair_rng = np.random.default_rng(child_seeds[pair_index])
        if section == "train" and pair_index == 0:
            lesson = "extent"
        elif section == "train" and pair_index == 1:
            lesson = _choice(pair_rng, ("ring_open", "ring_open", "generic"))
        else:
            lesson = _choice(pair_rng, ("generic", "generic", "ring_open", "extent"))
        overlap = _choice(pair_rng, ("none", "partial", "substantial"))
        reverse_pair = None
        if section == "train" and pair_index == 1 and first_rank is not None:
            if pair_rng.random() < 0.6:
                overlap = "substantial"
                reverse_pair = (first_rank[0], first_rank[-1], True)
        for _ in range(8):
            inp, out = _generate_pair(pair_rng, palette, count, previous,
                                      previous_background, lesson, overlap, reverse_pair)
            key = _grid_key(inp)
            if key not in seen_inputs and not _source_copy(inp, out):
                break
        else:
            raise RuntimeError("Could not make a unique non-source pair")
        puzzle[section].append({"input": inp, "output": out})
        seen_inputs.add(key)
        previous_background, _, rank_areas = _analyze(inp)
        previous = {color for color, _ in rank_areas}
        if pair_index == 0:
            first_rank = [color for color, _ in rank_areas]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    train_counts = []
    train_witness = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp_key, out_key = _grid_key(inp), _grid_key(out)
            assert inp_key not in seen_inputs and (inp_key, out_key) not in seen_pairs
            seen_inputs.add(inp_key)
            seen_pairs.add((inp_key, out_key))
            assert not _source_copy(inp, out)
            background, background_area, ranked = _analyze(inp)
            assert background_area > inp.size - background_area
            assert len(ranked) >= 2 and len(ranked) <= 9
            assert len({area for _, area in ranked}) == len(ranked)
            assert all(_connected(inp == color) for color, _ in ranked)
            assert background not in {color for color, _ in ranked}
            assert out.shape == (2 * len(ranked) - 1, 2 * len(ranked) - 1)
            assert np.array_equal(out, _build_output([color for color, _ in ranked]))
            assert set(int(v) for v in np.unique(out)) == {color for color, _ in ranked}
            assert not np.array_equal(inp, out) if inp.shape == out.shape else True
            if section == "train":
                train_counts.append(len(ranked))
                train_witness |= _extent_witness(inp, ranked)
    assert len(set(train_counts)) >= 2
    assert train_witness


def _independent_prediction(inp):
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    tokens = sorted(((int(n), int(v)) for v, n in zip(values, counts)
                     if int(v) != background), reverse=True)
    assert len({n for n, _ in tokens}) == len(tokens)
    side = 2 * len(tokens) - 1
    return np.array([[tokens[min(row, col, side - row - 1, side - col - 1)][1]
                      for col in range(side)] for row in range(side)], dtype=np.int8)


def _contact(inp, colors):
    height, width = inp.shape
    background = next(v for v in np.unique(inp) if int(v) not in colors)
    for row in range(height):
        for col in range(width):
            value = int(inp[row, col])
            if value == background:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = row + dr, col + dc
                if rr < height and cc < width and int(inp[rr, cc]) not in (value, background):
                    return True
    return False


def _has_hole(mask):
    rows, cols = np.nonzero(mask)
    box = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    height, width = box.shape
    exterior = {(r, c) for r in range(height) for c in range(width)
                if not box[r, c] and (r in (0, height - 1) or c in (0, width - 1))}
    frontier = list(exterior)
    while frontier:
        row, col = frontier.pop()
        for rr, cc in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
            if 0 <= rr < height and 0 <= cc < width and not box[rr, cc] and (rr, cc) not in exterior:
                exterior.add((rr, cc))
                frontier.append((rr, cc))
    return int((~box).sum()) > len(exterior)


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[section]) for section in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(101)
    assert _puzzle_key(automatic) == _puzzle_key(generate_puzzle(101))
    explicit = generate_puzzle(202, 7, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(202, 7, 3))
    assert _puzzle_key(explicit) != _puzzle_key(generate_puzzle(203, 7, 3))
    for seed, train, test in ((300, 2, 1), (301, 10, 1), (302, 10, 4),
                              (400, None, 1), (401, None, 4),
                              (402, 2, None), (403, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_train": 2.0}):
        try:
            generate_puzzle(500, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError: {kwargs}")
    puzzles = [generate_puzzle(seed) for seed in range(1000, 1042)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    dimensions = set()
    train_counts, test_counts = set(), set()
    holes = {"train": 0, "test": 0}
    contacts = {"train": 0, "test": 0}
    separated = {"train": 0, "test": 0}
    familiar_tests = 0
    extrapolating_tests = 0
    input_ge_output = 0
    output_gt_input = 0
    changing_backgrounds = 0
    positive_test_margins = 0
    close_profiles = 0
    wide_profiles = 0
    overlap_types = set()
    rank_reversals = 0
    hole_contact_tests = 0
    decreasing_train_order = 0
    normalized_shapes = set()
    max_area = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        first_two_train = [len(np.unique(pair["input"])) - 1
                           for pair in puzzle["train"][:2]]
        decreasing_train_order += first_two_train[0] > first_two_train[1]
        train_object_counts = {len(np.unique(pair["input"])) - 1 for pair in puzzle["train"]}
        backgrounds = set()
        prior_colors = None
        prior_train_rank = None
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                assert np.array_equal(out, _independent_prediction(inp))
                background, _, ranked = _analyze(inp)
                backgrounds.add(background)
                dimensions.add(inp.shape)
                (train_counts if section == "train" else test_counts).add(len(ranked))
                max_area = max(max_area, *(area for _, area in ranked))
                values = [area for _, area in ranked]
                close_profiles += any(a - b == 1 for a, b in zip(values, values[1:]))
                wide_profiles += values[0] - values[-1] >= 7
                has_hole = any(_has_hole(inp == color) for color, _ in ranked)
                if has_hole:
                    holes[section] += 1
                token_colors = {color for color, _ in ranked}
                has_contact = _contact(inp, token_colors)
                if has_contact:
                    contacts[section] += 1
                else:
                    separated[section] += 1
                if prior_colors is not None:
                    shared = len(token_colors & prior_colors)
                    if shared == 0:
                        overlap_types.add("none")
                    elif shared == min(len(token_colors), len(prior_colors)):
                        overlap_types.add("substantial")
                    else:
                        overlap_types.add("partial")
                prior_colors = token_colors
                current_rank = {color: index for index, (color, _) in enumerate(ranked)}
                if section == "train":
                    if prior_train_rank is not None:
                        shared = set(prior_train_rank) & set(current_rank)
                        rank_reversals += any(
                            (prior_train_rank[a] - prior_train_rank[b]) *
                            (current_rank[a] - current_rank[b]) < 0
                            for a in shared for b in shared if a < b)
                    prior_train_rank = current_rank
                for color, _ in ranked:
                    coordinates = np.argwhere(inp == color)
                    coordinates -= coordinates.min(axis=0)
                    normalized_shapes.add(tuple(sorted(map(tuple, coordinates.tolist()))))
                if section == "test":
                    familiar_tests += len(ranked) in train_object_counts
                    extrapolating_tests += len(ranked) > max(train_object_counts)
                    input_ge_output += inp.shape[0] >= out.shape[0] and inp.shape[1] >= out.shape[1]
                    output_gt_input += out.shape[0] > inp.shape[0] and out.shape[1] > inp.shape[1]
                    hole_contact_tests += has_hole and has_contact
                    rows, cols = np.nonzero(inp != background)
                    positive_test_margins += (
                        rows.min() > 0 and cols.min() > 0 and
                        rows.max() < inp.shape[0] - 1 and cols.max() < inp.shape[1] - 1)
        changing_backgrounds += len(backgrounds) > 1
    assert len(dimensions) >= 20
    assert train_counts.issuperset({2, 3, 4, 5, 6, 7})
    assert test_counts.issuperset({2, 3, 4, 5, 6, 7, 8, 9})
    assert all(holes[section] > 0 and contacts[section] > 0 and separated[section] > 0
               for section in ("train", "test"))
    assert familiar_tests > 0 and extrapolating_tests > 0
    assert input_ge_output > 0 and output_gt_input > 0
    assert changing_backgrounds > 0 and max_area >= 16
    assert positive_test_margins > 0 and hole_contact_tests > 0
    assert close_profiles > 0 and wide_profiles > 0
    assert overlap_types == {"none", "partial", "substantial"}
    assert rank_reversals > 0 and decreasing_train_order > 0
    assert len(normalized_shapes) >= 30
