"""Generate variations of the centered 3 by 3 marker expansion puzzle."""

import numpy as np


_SOURCE_POSITION_SETS = (
    ((1, 7), (2, 3), (5, 7), (7, 3)),
    ((1, 6), (4, 3), (7, 1)),
    ((1, 1), (2, 7), (4, 3), (7, 1), (7, 5)),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_optional_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_pair_counts(rng, seed, num_train, num_test):
    train = _validate_optional_count("num_train", num_train, 2, 10)
    test = _validate_optional_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = test + 1
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_colors(rng, train_count):
    # Fill is the sole learned color convention. Input roles remain local:
    # the dominant uniform field is background, and all isolated points mark
    # centers. Distinct pools make changing roles easy to recognize.
    colors = [int(x) for x in rng.permutation(10)]
    fill = colors[0]
    background_count = int(rng.choice([1, 2, 3], p=[0.40, 0.42, 0.18]))
    backgrounds = colors[1 : 1 + background_count]
    mode = int(rng.choice(3, p=[0.38, 0.27, 0.35]))
    marker_count = 1 if mode == 0 else int(rng.integers(2, min(3, train_count) + 1))
    markers = colors[1 + background_count : 1 + background_count + marker_count]
    return fill, backgrounds, markers, mode


def _sample_count(rng):
    roll = float(rng.random())
    if roll < 0.18:
        return 1
    if roll < 0.34:
        return 2
    if roll < 0.82:
        return int(rng.integers(3, 8))
    return int(rng.integers(8, 16))


def _sample_marker_counts(rng, train_n, test_n, color_mode, marker_colors):
    relation = int(rng.integers(4))
    if relation == 0 and rng.random() < 0.28:
        train = [3 + i % 2 for i in range(train_n)]
    else:
        train = [_sample_count(rng) for _ in range(train_n)]
    if color_mode == 2:
        train[0] = max(train[0], len(marker_colors))
    if relation == 0:  # familiar count extrapolation
        maximum = max(train)
        test = [
            int(rng.integers(maximum + 1, min(20, maximum + 5) + 1))
            for _ in range(test_n)
        ]
    elif relation == 1:  # count range overlaps, often with exact matches
        test = [
            train[int(rng.integers(len(train)))]
            if i == 0 else int(rng.integers(min(train), max(train) + 1))
            for i in range(test_n)
        ]
    elif relation == 2:  # smaller targets
        train = [max(2, value) for value in train]
        test = [int(rng.integers(1, min(train))) for _ in range(test_n)]
    else:  # unrestricted mixed counts
        test = [_sample_count(rng) for _ in range(test_n)]
    return train, test


def _axis_capacity(length, margin=1):
    return max(0, (length - 2 * margin + 2) // 3)


def _capacity(height, width, margin=1):
    return _axis_capacity(height, margin) * _axis_capacity(width, margin)


def _shape_in_category(height, width, category):
    if category == "tiny":
        return max(height, width) <= 8
    if category == "strip":
        return min(height, width) <= 5 and max(height, width) >= 9
    if category == "source":
        return 9 <= height <= 15 and 9 <= width <= 15
    if category == "large":
        return max(height, width) >= 20 and min(height, width) >= 6
    if category == "rectangle":
        return abs(height - width) >= 6 and min(height, width) >= 5
    if category == "limit":
        return height == 30 or width == 30
    return True


def _sample_shape(rng, count, used_shapes=()):
    if count == 1 and rng.random() < 0.22 and (3, 3) not in used_shapes:
        return 3, 3
    categories = ("tiny", "strip", "source", "large", "rectangle", "limit", "free")
    weights = (0.18, 0.16, 0.22, 0.16, 0.14, 0.08, 0.06)
    category = str(rng.choice(categories, p=weights))
    options = [
        (height, width)
        for height in range(3, 31)
        for width in range(3, 31)
        if _capacity(height, width) >= count
        and _shape_in_category(height, width, category)
        and (height, width) not in used_shapes
    ]
    if not options:
        options = [
            (height, width)
            for height in range(3, 31)
            for width in range(3, 31)
            if _capacity(height, width) >= count
            and (height, width) not in used_shapes
        ]
    return options[int(rng.integers(len(options)))]


def _positions_are_valid(positions, height, width):
    if not positions:
        return False
    if any(not (1 <= row < height - 1 and 1 <= col < width - 1)
           for row, col in positions):
        return False
    if len(set(positions)) != len(positions):
        return False
    return all(
        max(abs(row_a - row_b), abs(col_a - col_b)) >= 3
        for i, (row_a, col_a) in enumerate(positions)
        for row_b, col_b in positions[i + 1 :]
    )


def _spaced_axis(rng, low, high, count):
    """Choose separated coordinates with independently distributed slack."""
    slack = (high - low + 1) - (3 * count - 2)
    if slack < 0:
        raise ValueError("Axis is too short")
    gaps = rng.multinomial(slack, [1 / (count + 1)] * (count + 1))
    first = low + int(gaps[0])
    values = [first]
    for i in range(1, count):
        values.append(values[-1] + 3 + int(gaps[i]))
    return values


def _fits(candidate, positions):
    return all(
        max(abs(candidate[0] - row), abs(candidate[1] - col)) >= 3
        for row, col in positions
    )


def _lattice(rng, height, width, count, margin):
    row_low, row_high = margin, height - margin - 1
    col_low, col_high = margin, width - margin - 1
    rows = list(range(row_low, row_high + 1, 3))
    cols = list(range(col_low, col_high + 1, 3))
    pool = [(row, col) for row in rows for col in cols]
    selected = [pool[int(i)] for i in rng.permutation(len(pool))[:count]]
    # Move each point independently where possible; the regular packing is
    # only a guaranteed starting point, not the sole placement vocabulary.
    for index in rng.permutation(len(selected)):
        index = int(index)
        row, col = selected[index]
        moves = [(row + dr, col + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
        for raw in rng.permutation(len(moves)):
            candidate = moves[int(raw)]
            if (row_low <= candidate[0] <= row_high
                    and col_low <= candidate[1] <= col_high
                    and _fits(candidate, selected[:index] + selected[index + 1 :])):
                selected[index] = candidate
                break
    return selected


def _greedy_points(rng, height, width, count, margin, wide=False):
    candidates = [
        (row, col)
        for row in range(margin, height - margin)
        for col in range(margin, width - margin)
    ]
    if wide:
        positions = [candidates[int(rng.integers(len(candidates)))]]
        while len(positions) < count:
            available = [point for point in candidates if _fits(point, positions)]
            if not available:
                break
            distances = [
                min(max(abs(point[0] - row), abs(point[1] - col))
                    for row, col in positions)
                for point in available
            ]
            best = max(distances)
            choices = [point for point, distance in zip(available, distances)
                       if distance == best]
            positions.append(choices[int(rng.integers(len(choices)))])
        if len(positions) == count:
            return positions
    else:
        for _ in range(16):
            positions = []
            for index in rng.permutation(len(candidates)):
                candidate = candidates[int(index)]
                if _fits(candidate, positions):
                    positions.append(candidate)
                    if len(positions) == count:
                        return positions
    return _lattice(rng, height, width, count, margin)


def _compact_points(rng, height, width, count, margin):
    row_slots = _axis_capacity(height, margin)
    col_slots = _axis_capacity(width, margin)
    choices = [
        (rows, cols)
        for rows in range(1, row_slots + 1)
        for cols in range(1, col_slots + 1)
        if rows * cols >= count
    ]
    if not choices:
        return _lattice(rng, height, width, count, margin)
    smallest = min(rows * cols for rows, cols in choices)
    choices = [(rows, cols) for rows, cols in choices
               if rows * cols <= smallest + 2]
    rows_n, cols_n = choices[int(rng.integers(len(choices)))]
    row_start = int(rng.integers(margin, height - margin - 3 * (rows_n - 1)))
    col_start = int(rng.integers(margin, width - margin - 3 * (cols_n - 1)))
    pool = [
        (row_start + 3 * row, col_start + 3 * col)
        for row in range(rows_n)
        for col in range(cols_n)
    ]
    return [pool[int(i)] for i in rng.permutation(len(pool))[:count]]


def _sample_positions(rng, height, width, count):
    placement = str(rng.choice(("interior", "edge", "free"), p=(0.42, 0.38, 0.20)))
    margin = 1
    if placement == "interior" and _capacity(height, width, 2) >= count:
        margin = 2
        if rng.random() < 0.35 and _capacity(height, width, 3) >= count:
            margin = 3
    else:
        placement = "edge" if placement == "edge" else "free"
    row_slots = _axis_capacity(height, margin)
    col_slots = _axis_capacity(width, margin)
    styles = ["random", "compact", "wide"]
    weights = [0.24, 0.19, 0.17]
    if count >= 2 and col_slots >= count:
        styles.append("row")
        weights.append(0.17)
    if count >= 2 and row_slots >= count:
        styles.append("column")
        weights.append(0.17)
    if count >= 2 and min(row_slots, col_slots) >= count:
        styles += ["diagonal", "distinct"]
        weights += [0.12, 0.12]
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    style = str(rng.choice(styles, p=weights))
    row_low, row_high = margin, height - margin - 1
    col_low, col_high = margin, width - margin - 1
    if style == "row":
        row = int(rng.integers(row_low, row_high + 1))
        positions = [(row, col) for col in _spaced_axis(rng, col_low, col_high, count)]
    elif style == "column":
        col = int(rng.integers(col_low, col_high + 1))
        positions = [(row, col) for row in _spaced_axis(rng, row_low, row_high, count)]
    elif style in ("diagonal", "distinct"):
        rows = _spaced_axis(rng, row_low, row_high, count)
        cols = _spaced_axis(rng, col_low, col_high, count)
        if style == "diagonal":
            if rng.random() < 0.5:
                cols.reverse()
        else:
            cols = [cols[int(i)] for i in rng.permutation(count)]
        positions = list(zip(rows, cols))
    elif style == "compact":
        positions = _compact_points(rng, height, width, count, margin)
    else:
        positions = _greedy_points(rng, height, width, count, margin,
                                  wide=(style == "wide"))

    if placement == "edge":
        side = int(rng.integers(4))
        if side == 0:
            shift = 1 - min(row for row, _ in positions)
            positions = [(row + shift, col) for row, col in positions]
        elif side == 1:
            shift = height - 2 - max(row for row, _ in positions)
            positions = [(row + shift, col) for row, col in positions]
        elif side == 2:
            shift = 1 - min(col for _, col in positions)
            positions = [(row, col + shift) for row, col in positions]
        else:
            shift = width - 2 - max(col for _, col in positions)
            positions = [(row, col + shift) for row, col in positions]
    _require(_positions_are_valid(positions, height, width),
             "Placement failed to produce separate complete squares")
    return positions


def _construct_output(input_grid, background, fill):
    output = input_grid.copy()
    positions = np.argwhere(input_grid != background)
    for raw_row, raw_col in positions:
        row, col = int(raw_row), int(raw_col)
        center = int(input_grid[row, col])
        output[row - 1 : row + 2, col - 1 : col + 2] = fill
        output[row, col] = center
    return output


def _grid_signature(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _matches_source_pair(input_grid, output_grid):
    if input_grid.shape != (9, 9) or output_grid.shape != (9, 9):
        return False
    for positions in _SOURCE_POSITION_SETS:
        source_input = np.zeros((9, 9), dtype=np.int64)
        for row, col in positions:
            source_input[row, col] = 5
        if (np.array_equal(input_grid, source_input)
                and np.array_equal(output_grid,
                                   _construct_output(source_input, 0, 1))):
            return True
    return False


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_n, test_n = _resolve_pair_counts(rng, int(seed), num_train, num_test)
    fill, backgrounds, markers, color_mode = _sample_colors(rng, train_n)
    train_counts, test_counts = _sample_marker_counts(
        rng, train_n, test_n, color_mode, markers
    )
    common_shape = bool(rng.random() < 0.22)
    all_counts = train_counts + test_counts
    # A common tiny canvas with one marker has too few distinct placements
    # for a long puzzle, even though each individual pair would be valid.
    shared_shape = (
        _sample_shape(rng, max(max(all_counts), len(all_counts)))
        if common_shape else None
    )

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_shapes = set()
    for split, counts in (("train", train_counts), ("test", test_counts)):
        for index, count in enumerate(counts):
            for _ in range(40):
                if shared_shape is None:
                    shape = _sample_shape(rng, count, used_shapes)
                else:
                    shape = shared_shape
                height, width = shape
                background = (
                    backgrounds[index % len(backgrounds)]
                    if split == "train"
                    else backgrounds[int(rng.integers(len(backgrounds)))]
                )
                if color_mode == 0:
                    marker_colors = [markers[0]] * count
                elif color_mode == 1:
                    marker = (
                        markers[index % len(markers)]
                        if split == "train"
                        else markers[int(rng.integers(len(markers)))]
                    )
                    marker_colors = [marker] * count
                elif split == "train" and index == 0:
                    marker_colors = list(markers) + [
                        markers[int(rng.integers(len(markers)))]
                        for _ in range(count - len(markers))
                    ]
                else:
                    marker_colors = [
                        markers[int(rng.integers(len(markers)))]
                        for _ in range(count)
                    ]
                positions = _sample_positions(rng, height, width, count)
                input_grid = np.full(shape, background, dtype=np.int64)
                for (row, col), color in zip(positions, marker_colors):
                    input_grid[row, col] = color
                output_grid = _construct_output(input_grid, background, fill)
                signature = _grid_signature(input_grid)
                if signature in used_inputs or _matches_source_pair(
                    input_grid, output_grid
                ):
                    continue
                used_inputs.add(signature)
                used_shapes.add(shape)
                puzzle[split].append({"input": input_grid, "output": output_grid})
                break
            else:
                raise ValueError("Unable to construct distinct puzzle pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _input_roles(input_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    _require(len(values) >= 2, "Input needs background and marker colors")
    background_index = int(np.argmax(counts))
    _require(int(counts[background_index]) > int(sum(counts) - counts[background_index]),
             "Background must be the dominant uniform field")
    background = int(values[background_index])
    positions = [(int(row), int(col))
                 for row, col in np.argwhere(input_grid != background)]
    return background, positions


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must contain exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list),
             "Splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4
             and len(train) > len(test), "Invalid split counts")
    seen_inputs, seen_pairs = set(), set()
    fill = None
    trained_marker_colors = set()
    for split_name, pairs in (("train", train), ("test", test)):
        for index, pair in enumerate(pairs):
            label = f"{split_name}[{index}]"
            _require(isinstance(pair, dict)
                     and set(pair) == {"input", "output"}, f"{label}: invalid pair")
            input_grid, output_grid = pair["input"], pair["output"]
            for name, grid in (("input", input_grid), ("output", output_grid)):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label}.{name}: not a two-dimensional array")
                _require(all(1 <= side <= 30 for side in grid.shape)
                         and np.issubdtype(grid.dtype, np.integer)
                         and bool(np.all((grid >= 0) & (grid <= 9))),
                         f"{label}.{name}: invalid grid")
            _require(input_grid.shape == output_grid.shape,
                     f"{label}: dimensions changed")
            input_sig = _grid_signature(input_grid)
            pair_sig = input_sig, _grid_signature(output_grid)
            _require(input_sig not in seen_inputs and pair_sig not in seen_pairs,
                     f"{label}: duplicate")
            seen_inputs.add(input_sig)
            seen_pairs.add(pair_sig)
            _require(not _matches_source_pair(input_grid, output_grid),
                     f"{label}: exact source copy")

            background, positions = _input_roles(input_grid)
            _require(_positions_are_valid(positions, *input_grid.shape),
                     f"{label}: incomplete or overlapping footprints")
            input_colors = {int(color) for color in np.unique(input_grid)}
            output_colors = {int(color) for color in np.unique(output_grid)}
            added_colors = output_colors - input_colors
            if split_name == "train":
                _require(len(added_colors) == 1,
                         f"{label}: training does not reveal one fill color")
                learned = next(iter(added_colors))
                if fill is None:
                    fill = learned
                _require(learned == fill, f"{label}: inconsistent fill color")
                trained_marker_colors.update(input_colors - {background})
            else:
                _require(fill is not None, "Training did not reveal fill")
                _require(input_colors - {background} <= trained_marker_colors,
                         f"{label}: marker color absent from training")
            _require(fill not in input_colors,
                     f"{label}: fill conflicts with an input role")
            _require(output_colors == input_colors | {fill}
                     or output_colors == (input_colors - {background}) | {fill},
                     f"{label}: output has unexpected colors")
            expected = _construct_output(input_grid, background, fill)
            _require(np.array_equal(output_grid, expected),
                     f"{label}: incorrect 3 by 3 expansion")
            _require(int(np.count_nonzero(output_grid == fill)) == 8 * len(positions),
                     f"{label}: incorrect number of filled cells")
            for row, col in positions:
                _require(int(output_grid[row, col]) == int(input_grid[row, col]),
                         f"{label}: center color changed")


def _puzzle_fingerprint(puzzle):
    return tuple(
        (split, tuple((_grid_signature(pair["input"]),
                       _grid_signature(pair["output"]))
                      for pair in puzzle[split]))
        for split in ("train", "test")
    )


def _predict_from_training(train, test_input):
    # This deliberately uses only the visible training grids and test input.
    fill_colors = set()
    trained_markers = set()
    for pair in train:
        source, target = pair["input"], pair["output"]
        background, positions = _input_roles(source)
        trained_markers.update(int(source[row, col]) for row, col in positions)
        changes = target[source != target]
        _require(len(changes) > 0, "No training cells changed")
        fill_colors.update(int(value) for value in np.unique(changes))
    _require(len(fill_colors) == 1, "Training has ambiguous fill")
    fill = next(iter(fill_colors))
    background, positions = _input_roles(test_input)
    _require({int(test_input[row, col]) for row, col in positions} <= trained_markers,
             "Test marker role is unseen")
    prediction = test_input.copy()
    for row, col in positions:
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row or delta_col:
                    prediction[row + delta_row, col + delta_col] = fill
    return prediction


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    fingerprints = set()
    shapes = set()
    layouts = set()
    train_pair_counts, test_pair_counts = set(), set()
    marker_counts = set()
    relationship = set()
    features = set()
    within_shape_variation = 0
    within_background_variation = 0
    within_marker_variation = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_fingerprint(puzzle))
        train_pair_counts.add(len(puzzle["train"]))
        test_pair_counts.add(len(puzzle["test"]))
        pair_shapes = set()
        pair_backgrounds = set()
        pair_marker_palettes = set()
        split_counts = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                background, positions = _input_roles(source)
                marker_palette = frozenset(int(source[row, col])
                                           for row, col in positions)
                split_counts[split].append(len(positions))
                marker_counts.add(len(positions))
                pair_shapes.add(source.shape)
                pair_backgrounds.add(background)
                pair_marker_palettes.add(marker_palette)
                shapes.add(source.shape)
                layouts.add((source.shape, tuple(positions)))
                height, width = source.shape
                if (height, width) == (3, 3):
                    features.add("3x3")
                if min(height, width) <= 5 and max(height, width) >= 9:
                    features.add("strip")
                if height == 30 or width == 30:
                    features.add("axis30")
                if len(positions) >= 8:
                    features.add("eight")
                if len(marker_palette) >= 2 and split == "train":
                    features.add("mixed_markers_train")
                rows = [row for row, _ in positions]
                cols = [col for _, col in positions]
                if len(positions) >= 4 and len(set(rows)) == 1:
                    features.add("row_run")
                if len(positions) >= 4 and len(set(cols)) == 1:
                    features.add("column_run")
                if len(positions) >= 2 and len(set(rows)) == len(rows) and len(set(cols)) == len(cols):
                    features.add("no_shared_axis")
                edge = any(row in (1, height - 2) or col in (1, width - 2)
                           for row, col in positions)
                features.add("edge" if edge else "interior")
                # Reconstruct changed cells without using the output builder.
                expected_changes = set()
                for row, col in positions:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr or dc:
                                expected_changes.add((row + dr, col + dc))
                observed_changes = {tuple(int(x) for x in rc)
                                    for rc in np.argwhere(source != target)}
                _require(observed_changes == expected_changes,
                         "Independent footprint reconstruction failed")
                if split == "test":
                    _require(np.array_equal(
                        _predict_from_training(puzzle["train"], source), target
                    ), "Training-only prediction failed")
        within_shape_variation += len(pair_shapes) >= 2
        within_background_variation += len(pair_backgrounds) >= 2
        within_marker_variation += len(pair_marker_palettes) >= 2
        train_min, train_max = min(split_counts["train"]), max(split_counts["train"])
        test_min, test_max = min(split_counts["test"]), max(split_counts["test"])
        if test_min > train_max:
            relationship.add("extrapolate")
        if test_max < train_min:
            relationship.add("lower")
        if set(split_counts["train"]) & set(split_counts["test"]):
            relationship.add("overlap")
    _require(len(fingerprints) == 42 and len(shapes) >= 60
             and len(layouts) >= 180, "Insufficient structural diversity")
    _require(len(train_pair_counts) >= 4 and len(test_pair_counts) >= 3,
             "Automatic pair counts do not vary enough")
    _require({1, 2, 8} <= marker_counts and max(marker_counts) >= 12,
             "Marker count range is too narrow")
    _require({"extrapolate", "lower", "overlap"} <= relationship,
             "Count relationships do not vary")
    _require({"3x3", "strip", "axis30", "eight", "mixed_markers_train",
              "row_run", "column_run", "no_shared_axis", "edge", "interior"} <= features,
             "A reviewed placement, size, or color feature is absent")
    _require(within_shape_variation >= 15 and within_background_variation >= 8
             and within_marker_variation >= 8,
             "Within-puzzle variation is too narrow")

    for seed, train_n, test_n in ((1001, 2, 1), (1002, 10, 1),
                                  (1003, 10, 4), (1004, 5, 2),
                                  (1005, 8, 3)):
        puzzle = generate_puzzle(seed, train_n, test_n)
        _require(len(puzzle["train"]) == train_n and len(puzzle["test"]) == test_n,
                 "Explicit pair counts were not honored")
        validate_generated_puzzle(puzzle)
        _require(_puzzle_fingerprint(puzzle)
                 == _puzzle_fingerprint(generate_puzzle(seed, train_n, test_n)),
                 "Explicit generation is not deterministic")
    for seed, train_n, test_n in ((1101, None, 1), (1102, None, 4),
                                  (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train_n, test_n)
        _require(train_n is None or len(puzzle["train"]) == train_n,
                 "Partial train count was not honored")
        _require(test_n is None or len(puzzle["test"]) == test_n,
                 "Partial test count was not honored")
        validate_generated_puzzle(puzzle)
    for train_n, test_n in ((1, None), (11, None), (None, 0), (None, 5),
                            (2, 2), (3, 4), (4, 4), (2, 4), (2.0, 1)):
        try:
            generate_puzzle(1200, train_n, test_n)
        except ValueError:
            continue
        raise AssertionError("An illegal count request was accepted")
    _require(_puzzle_fingerprint(generate_puzzle(17))
             == _puzzle_fingerprint(generate_puzzle(17)),
             "Automatic generation is not deterministic")
    _require(len({_puzzle_fingerprint(generate_puzzle(seed, 6, 2))
                  for seed in range(200, 212)}) >= 11,
             "Different seeds did not produce distinct puzzles")
