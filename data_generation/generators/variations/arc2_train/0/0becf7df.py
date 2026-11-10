import numpy as np


_SOURCE_INPUTS = (
    ("1300000000", "2800001000", "0000111000", "0000111000", "0033331800", "0033208800", "0000208800", "0000200000", "0000200000", "0000000000"),
    ("4200000000", "3700004000", "0000034400", "0000032400", "0007732400", "0007332000", "0007002200", "0007700000", "0000000000", "0000000000"),
    ("9400000000", "7600099000", "0000079000", "0000040000", "0000740000", "0006670000", "0007660000", "0000000000", "0000000000", "0000000000"),
    ("8900000000", "2400099000", "0008889000", "0002889000", "0002420000", "0002240000", "0002440000", "0009440000", "0000040000", "0000000000"),
)
_SOURCE_SIGNATURES = {
    (10, 10, tuple(int(cell) for row in rows for cell in row))
    for rows in _SOURCE_INPUTS
}


def _signature(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(cell) for cell in grid.flat))


def _counts(seed, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"), (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f"{name} is outside its legal range")
    if num_test is None:
        num_test = 1 + (seed // 7) % (min(4, num_train - 1) if num_train is not None else 4)
    if num_train is None:
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _edge(first, second):
    return tuple(sorted((int(first), int(second))))


def _matching(legend):
    return frozenset(_edge(*row) for row in legend)


def _orient(rng, matching):
    rows = [list(edge) for edge in matching]
    if int(rng.integers(2)):
        rows.reverse()
    for row in rows:
        if int(rng.integers(2)):
            row.reverse()
    return np.asarray(rows, dtype=np.int8)


def _training_legends(rng, count, reserved):
    colors = [color for color in range(10) if color != reserved]
    first = rng.choice(colors, size=4, replace=False).tolist()
    rng.shuffle(first)
    matchings = [(tuple(first[:2]), tuple(first[2:]))]
    for index in range(1, count):
        mode = int(rng.integers(5))
        if index == 1 and mode <= 2:
            palette = first.copy()
            if mode == 0:
                matching = matchings[0]
            else:
                matching = ((palette[0], palette[2]), (palette[1], palette[3])) if mode == 1 else ((palette[0], palette[3]), (palette[1], palette[2]))
        elif mode == 0:
            matching = matchings[int(rng.integers(len(matchings)))]
        else:
            palette = rng.choice(colors, size=4, replace=False).tolist()
            rng.shuffle(palette)
            matching = (tuple(palette[:2]), tuple(palette[2:]))
        matchings.append(matching)
    return [_orient(rng, matching) for matching in matchings]


def _test_legends(rng, training, count):
    known = {_edge(*row) for legend in training for row in legend}
    prior_orders = {_edge(*row): tuple(int(color) for color in row) for legend in training for row in legend}
    seen_colors = {int(color) for legend in training for color in legend.flat}
    candidates = {kind: [] for kind in range(4)}
    for first in range(10):
        for second in range(first + 1, 10):
            for third in range(10):
                if third in (first, second):
                    continue
                for fourth in range(third + 1, 10):
                    if fourth in (first, second):
                        continue
                    matching = ((first, second), (third, fourth))
                    familiar = sum(_edge(*edge) in known for edge in matching)
                    has_new_color = any(color not in seen_colors for edge in matching for color in edge)
                    if has_new_color:
                        candidates[3].append(matching)
                    elif familiar == 1:
                        candidates[0].append(matching)
                    elif familiar == 2:
                        candidates[1].append(matching)
                    else:
                        candidates[2].append(matching)
    result = []
    for index in range(count):
        kind = (int(rng.integers(4)) + index) % 4
        if not candidates[kind]:
            kind = 3
        matching = candidates[kind][int(rng.integers(len(candidates[kind])))]
        reverse_edge = None
        if kind == 0:
            reverse_edge = next(_edge(*edge) for edge in matching if _edge(*edge) in known)
        legend = _orient(rng, matching)
        if reverse_edge is not None:
            for row in legend:
                if _edge(*row) == reverse_edge:
                    row[:] = prior_orders[reverse_edge][::-1]
        result.append(legend)
    return result


def _dimensions(rng, regime, force_demo, scale_shift):
    category = int(rng.choice(4, p=(0.18, 0.43, 0.22, 0.17)))
    if scale_shift and regime == "target_test" and int(rng.integers(3)) == 0:
        category = int(rng.choice((2, 3)))
    if force_demo and category == 0:
        category = 1
    if category == 0:
        return int(rng.integers(6, 10)), int(rng.integers(6, 10))
    if category == 1:
        return int(rng.integers(9, 17)), int(rng.integers(9, 17))
    if category == 2:
        return int(rng.integers(17, 31)), int(rng.integers(17, 31))
    short = int(rng.integers(6 if not force_demo else 9, 11))
    long = int(rng.integers(17, 31))
    return (short, long) if int(rng.integers(2)) else (long, short)


def _key_position(rng, height, width, fixed_corner, index):
    if fixed_corner is not None:
        kind = fixed_corner
    elif index == 0:
        kind = 0
    elif index == 1:
        kind = int(rng.choice((4, 5, 6, 7, 8)))
    else:
        kind = int(rng.integers(9))
    if kind < 4:
        return (0 if kind < 2 else height - 2, 0 if kind % 2 == 0 else width - 2)
    if kind == 4:
        return 0, int(rng.integers(2, width - 3))
    if kind == 5:
        return height - 2, int(rng.integers(2, width - 3))
    if kind == 6:
        return int(rng.integers(2, height - 3)), 0
    if kind == 7:
        return int(rng.integers(2, height - 3)), width - 2
    return int(rng.integers(2, height - 3)), int(rng.integers(2, width - 3))


def _shape(rng, height, width, force_demo):
    kind = int(rng.integers(6 if not force_demo else 5))
    span = int(rng.choice((3, 5, 8, 13)))
    local_height = min(height, max(2, int(rng.integers(2, span + 2))))
    local_width = min(width, max(2, int(rng.integers(2, span + 2))))
    if force_demo:
        local_height = max(3, local_height)
        local_width = max(3, local_width)
    if kind == 3:
        if int(rng.integers(2)):
            local_height = 1
        else:
            local_width = 1
    shape = np.zeros((local_height, local_width), dtype=bool)
    if kind == 0:
        target = max(2, int(rng.uniform(0.35, 0.9) * local_height * local_width))
        if force_demo:
            target = max(8, target)
        target = min(target, local_height * local_width)
        shape[int(rng.integers(local_height)), int(rng.integers(local_width))] = True
        while int(shape.sum()) < target:
            frontier = []
            for row in range(local_height):
                for column in range(local_width):
                    if not shape[row, column] and ((row and shape[row - 1, column]) or (column and shape[row, column - 1]) or (row + 1 < local_height and shape[row + 1, column]) or (column + 1 < local_width and shape[row, column + 1])):
                        frontier.append((row, column))
            row, column = frontier[int(rng.integers(len(frontier)))]
            shape[row, column] = True
    elif kind == 1:
        shape[:] = True
    elif kind == 2:
        if local_height < 3 or local_width < 3:
            shape[:] = True
        else:
            shape[0, :] = shape[-1, :] = True
            shape[:, 0] = shape[:, -1] = True
    elif kind == 3:
        shape[:] = True
    elif kind == 4:
        middle = local_width // 2
        shape[:, middle] = True
        for row in range(0, local_height, 2):
            if int(rng.integers(2)):
                shape[row, :middle + 1] = True
            else:
                shape[row, middle:] = True
    else:
        shape = np.ones((1, 1), dtype=bool)
    return shape


def _dilate(mask):
    expanded = mask.copy()
    height, width = mask.shape
    for row_shift in (-1, 0, 1):
        for column_shift in (-1, 0, 1):
            source_rows = slice(max(0, -row_shift), min(height, height - row_shift))
            source_columns = slice(max(0, -column_shift), min(width, width - column_shift))
            target_rows = slice(max(0, row_shift), min(height, height + row_shift))
            target_columns = slice(max(0, column_shift), min(width, width + column_shift))
            expanded[target_rows, target_columns] |= mask[source_rows, source_columns]
    return expanded


def _payload(rng, height, width, key_position, force_demo):
    key = np.zeros((height, width), dtype=bool)
    row, column = key_position
    key[row:row + 2, column:column + 2] = True
    forbidden = _dilate(key)
    payload = np.zeros_like(key)
    scatter = not force_demo and int(rng.integers(9)) == 0
    object_count = int(rng.integers(3, 9)) if scatter else int(rng.choice((1, 2, 3, 4), p=(0.36, 0.32, 0.22, 0.10)))
    limit = int(0.42 * height * width) - 4
    for object_index in range(object_count):
        for attempt in range(45):
            shape = np.ones((1, 1), dtype=bool) if scatter else _shape(rng, height - 2, width - 2, force_demo and object_index == 0)
            shape_height, shape_width = shape.shape
            if int(shape.sum()) + int(payload.sum()) > limit:
                continue
            top = int(rng.integers(height - shape_height + 1))
            left = int(rng.integers(width - shape_width + 1))
            occupied = shape & (forbidden | _dilate(payload))[top:top + shape_height, left:left + shape_width]
            if occupied.any():
                continue
            payload[top:top + shape_height, left:left + shape_width] |= shape
            break
    return payload


def _paint(rng, grid, payload, legend, force_demo):
    coordinates = np.argwhere(payload)
    palette = [int(color) for color in legend.flat]
    rng.shuffle(palette)
    active_count = 4 if force_demo else int(rng.choice((1, 2, 3, 4), p=(0.15, 0.23, 0.26, 0.36)))
    palette = palette[:active_count]
    style = int(rng.integers(5))
    if force_demo:
        ordering = np.lexsort((coordinates[:, 1], coordinates[:, 0])) if style % 2 else np.lexsort((coordinates[:, 0], coordinates[:, 1]))
        labels = np.empty(len(coordinates), dtype=int)
        for label, indices in enumerate(np.array_split(ordering, 4)):
            labels[indices] = label
    elif style == 0:
        labels = (coordinates[:, 0] + int(rng.integers(active_count))) % active_count
    elif style == 1:
        labels = (coordinates[:, 1] + int(rng.integers(active_count))) % active_count
    elif style == 2:
        labels = (coordinates[:, 0] + coordinates[:, 1]) % active_count
    elif style == 3:
        anchors = coordinates[rng.choice(len(coordinates), size=active_count, replace=True)]
        labels = np.argmin(np.abs(coordinates[:, None, :] - anchors[None, :, :]).sum(axis=2), axis=1)
    else:
        weights = rng.dirichlet(np.ones(active_count) * float(rng.choice((0.4, 1.0, 3.0))))
        labels = rng.choice(active_count, size=len(coordinates), p=weights)
    grid[coordinates[:, 0], coordinates[:, 1]] = np.asarray(palette, dtype=np.int8)[labels]


def _key_candidates(grid, background):
    height, width = grid.shape
    foreground = grid != background
    candidates = []
    for row in range(height - 1):
        for column in range(width - 1):
            block = grid[row:row + 2, column:column + 2]
            if len(set(int(color) for color in block.flat)) != 4 or not foreground[row:row + 2, column:column + 2].all():
                continue
            top, bottom = max(0, row - 1), min(height, row + 3)
            left, right = max(0, column - 1), min(width, column + 3)
            halo = foreground[top:bottom, left:right].copy()
            halo[row - top:row - top + 2, column - left:column - left + 2] = False
            if not halo.any():
                candidates.append((row, column))
    return candidates


def _solve_input(input_grid):
    counts = np.bincount(input_grid.flat, minlength=10)
    background = int(np.argmax(counts))
    candidates = _key_candidates(input_grid, background)
    assert len(candidates) == 1
    row, column = candidates[0]
    legend = input_grid[row:row + 2, column:column + 2]
    assert set(int(color) for color in np.unique(input_grid)) <= {background, *(int(color) for color in legend.flat)}
    output = input_grid.copy()
    outside = np.ones(input_grid.shape, dtype=bool)
    outside[row:row + 2, column:column + 2] = False
    for first, second in legend:
        first_cells = outside & (input_grid == first)
        second_cells = outside & (input_grid == second)
        output[first_cells] = second
        output[second_cells] = first
    return output, background, (row, column)


def _make_pair(rng, legend, regime, fixed_corner, index, scale_shift):
    force_demo = regime == "source_train" and index < 2
    palette = {int(color) for color in legend.flat}
    background = int(rng.choice([color for color in range(10) if color not in palette]))
    for attempt in range(100):
        height, width = _dimensions(rng, regime, force_demo, scale_shift)
        key_position = _key_position(rng, height, width, fixed_corner, index)
        payload = _payload(rng, height, width, key_position, force_demo)
        if int(payload.sum()) < (8 if force_demo else 1):
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        row, column = key_position
        input_grid[row:row + 2, column:column + 2] = legend
        _paint(rng, input_grid, payload, legend, force_demo)
        if force_demo and any(not np.any(payload & (input_grid == color)) for color in palette):
            continue
        if len(_key_candidates(input_grid, background)) != 1:
            continue
        output_grid, recovered_background, recovered_key = _solve_input(input_grid)
        if recovered_background != background or recovered_key != key_position or np.array_equal(input_grid, output_grid):
            continue
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not construct an unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    reserved = int(rng.integers(10))
    fixed_corner = int(rng.integers(4)) if int(rng.integers(3)) == 0 else None
    scale_shift = bool(rng.integers(2))
    training = _training_legends(rng, num_train, reserved)
    testing = _test_legends(rng, training, num_test)
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, legends in (("train", training), ("test", testing)):
        regime = "source_train" if section == "train" else "target_test"
        for index, legend in enumerate(legends):
            for attempt in range(20):
                pair = _make_pair(rng, legend, regime, fixed_corner, index, scale_shift)
                signature = _signature(pair["input"])
                if signature not in seen and signature not in _SOURCE_SIGNATURES:
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
            puzzle[section].append(pair)
            seen.add(signature)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert input_grid.shape == output_grid.shape
            signature = _signature(input_grid)
            assert signature not in seen_inputs and signature not in _SOURCE_SIGNATURES
            seen_inputs.add(signature)
            expected, background, key_position = _solve_input(input_grid)
            assert np.array_equal(output_grid, expected) and not np.array_equal(input_grid, output_grid)
            assert np.array_equal(input_grid == background, output_grid == background)
            if section == "train" and index < 2:
                row, column = key_position
                palette = input_grid[row:row + 2, column:column + 2]
                outside = input_grid.copy()
                outside[row:row + 2, column:column + 2] = background
                assert all(np.any(outside == color) for color in palette.flat)


def _same_puzzle(first, second):
    return all(len(first[section]) == len(second[section]) and all(np.array_equal(left[role], right[role]) for left, right in zip(first[section], second[section]) for role in ("input", "output")) for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert _same_puzzle(puzzles[17], generate_puzzle(17))
    assert not _same_puzzle(puzzles[17], puzzles[18])
    for num_train, num_test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2024 + (num_train or 0) + (num_test or 0), num_train, num_test)
        validate_generated_puzzle(puzzle)
        if num_train is not None:
            assert len(puzzle["train"]) == num_train
        if num_test is not None:
            assert len(puzzle["test"]) == num_test
    for num_train, num_test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(99, num_train, num_test)
        except ValueError:
            continue
        raise AssertionError("illegal counts were accepted")
    backgrounds, dimensions, locations, components, counts, color_counts = set(), set(), set(), set(), set(), set()
    masks = set()
    matching_repeats = False
    changed_partners = False
    moving_keys = varied_backgrounds = varied_dimensions = 0
    blank_margins = black_keys = large_canvas_small_drawing = 0
    familiar_histogram = [0, 0, 0]
    new_test_colors = 0
    shape_types = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        prior = []
        training_edges = set()
        training_colors = set()
        local_backgrounds, local_positions, local_dimensions = set(), set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                input_grid, output_grid = pair["input"], pair["output"]
                predicted, background, (row, column) = _solve_input(input_grid.copy())
                assert np.array_equal(predicted, output_grid)
                backgrounds.add(background)
                local_backgrounds.add(background)
                if row in (0, input_grid.shape[0] - 2) and column in (0, input_grid.shape[1] - 2):
                    local_positions.add(("corner", row == 0, column == 0))
                else:
                    local_positions.add(("other", row, column))
                local_dimensions.add(input_grid.shape)
                dimensions.add(input_grid.shape)
                locations.add((row == 0 or row == input_grid.shape[0] - 2, column == 0 or column == input_grid.shape[1] - 2))
                legend = input_grid[row:row + 2, column:column + 2]
                black_keys += int(0 in legend)
                payload = input_grid != background
                payload[row:row + 2, column:column + 2] = False
                counts.add(int(payload.sum()))
                large_canvas_small_drawing += int(input_grid.size >= 300 and int(payload.sum()) <= 13)
                color_counts.add(len(set(int(color) for color in input_grid[payload])))
                masks.add((input_grid.shape, tuple(bool(cell) for cell in payload.flat)))
                object_components = _components(payload)
                components.add(len(object_components))
                occupied_rows, occupied_columns = np.nonzero(payload)
                blank_margins += int(occupied_rows.min() > 0 and occupied_rows.max() < input_grid.shape[0] - 1 and occupied_columns.min() > 0 and occupied_columns.max() < input_grid.shape[1] - 1)
                for component in object_components:
                    rows = [cell[0] for cell in component]
                    columns = [cell[1] for cell in component]
                    extent_height = max(rows) - min(rows) + 1
                    extent_width = max(columns) - min(columns) + 1
                    if len(component) == 1:
                        shape_types.add("pixel")
                    elif extent_height == 1 or extent_width == 1:
                        shape_types.add("line")
                    elif len(component) == extent_height * extent_width:
                        shape_types.add("solid")
                    elif extent_height >= 3 and extent_width >= 3 and (min(rows) + 1, min(columns) + 1) not in component and all((min(rows), position) in component and (max(rows), position) in component for position in range(min(columns), max(columns) + 1)):
                        shape_types.add("hollow")
                    else:
                        shape_types.add("irregular")
                if section == "train":
                    matching = _matching(legend)
                    matching_repeats |= matching in prior
                    changed_partners |= any({color for edge in matching for color in edge} == {color for edge in old for color in edge} and matching != old for old in prior)
                    prior.append(matching)
                    training_edges.update(matching)
                    training_colors.update(int(color) for color in legend.flat)
                else:
                    matching = _matching(legend)
                    familiar_histogram[sum(edge in training_edges for edge in matching)] += 1
                    new_test_colors += int(any(int(color) not in training_colors for color in legend.flat))
        varied_backgrounds += int(len(local_backgrounds) > 1)
        varied_dimensions += int(len(local_dimensions) > 1)
        moving_keys += int(len(local_positions) > 1)
    assert len(backgrounds) == 10 and len(dimensions) >= 80
    assert len(locations) >= 3 and min(counts) <= 13 and max(counts) >= 35
    assert 1 in color_counts and 4 in color_counts
    assert 1 in components and max(components) >= 3
    assert varied_backgrounds >= 35 and varied_dimensions >= 35 and 5 <= moving_keys <= 41
    assert black_keys >= 20 and blank_margins >= 20 and large_canvas_small_drawing >= 5
    assert all(count >= 5 for count in familiar_histogram) and new_test_colors >= 5
    assert {"pixel", "line", "solid", "hollow", "irregular"} <= shape_types
    assert len(masks) >= 0.9 * sum(len(puzzle[section]) for puzzle in puzzles for section in ("train", "test"))
    assert matching_repeats and changed_partners


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    components = []
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            if not mask[row, column] or seen[row, column]:
                continue
            stack = [(row, column)]
            seen[row, column] = True
            component = set()
            while stack:
                current_row, current_column = stack.pop()
                component.add((current_row, current_column))
                for next_row, next_column in ((current_row - 1, current_column), (current_row + 1, current_column), (current_row, current_column - 1), (current_row, current_column + 1)):
                    if 0 <= next_row < mask.shape[0] and 0 <= next_column < mask.shape[1] and mask[next_row, next_column] and not seen[next_row, next_column]:
                        seen[next_row, next_column] = True
                        stack.append((next_row, next_column))
            components.append(component)
    return components
