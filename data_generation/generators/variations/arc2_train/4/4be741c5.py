import numpy as np


def _ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid_from_strings(rows):
    return np.asarray(
        [[ord(character) - ord("0") for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _grid_from_strings(
            (
                "2222222",
                "2222222",
                "2222222",
                "2888228",
                "8888888",
                "8888888",
                "8885588",
                "5855555",
                "5555555",
            )
        ),
        _grid_from_strings(("2", "8", "5")),
    ),
    (
        _grid_from_strings(
            (
                "4444444222222288",
                "4444444422228888",
                "4444444422228888",
                "4444444222228888",
                "4444442222222888",
                "4444442222228888",
                "4444444222228888",
                "4444444222222888",
                "4444422222222288",
                "4444422222222288",
                "4444222222222888",
                "4444222222228888",
                "4444422222228888",
                "4444442222222888",
            )
        ),
        _grid_from_strings(("428",)),
    ),
    (
        _grid_from_strings(
            (
                "666666666",
                "664466666",
                "644464644",
                "444444444",
                "444444444",
                "444444444",
                "422444224",
                "222222222",
                "232222233",
                "333333333",
                "333333333",
            )
        ),
        _grid_from_strings(("6", "4", "2", "3")),
    ),
    (
        _grid_from_strings(
            (
                "33332222211188",
                "33322222111888",
                "33333221118888",
                "33333221111888",
                "33332222211188",
                "33333222211118",
                "33322222111188",
                "33332222111888",
                "33332222111188",
                "33322222211188",
                "33322222211888",
                "33332222111188",
                "33333222111188",
                "33333322211188",
            )
        ),
        _grid_from_strings(("3218",)),
    ),
)


def _grid_signature(grid):
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _puzzle_signature(puzzle):
    return tuple(
        (
            split,
            _grid_signature(pair["input"]),
            _grid_signature(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def _puzzles_equal(first, second):
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False
    return True


def _is_source_pair(input_grid, output_grid):
    for source_input, source_output in _SOURCE_PAIRS:
        if (
            input_grid.shape == source_input.shape
            and output_grid.shape == source_output.shape
            and np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _collapse_line(line):
    result = [int(line[0])]
    for value in line[1:]:
        integer_value = int(value)
        if integer_value != result[-1]:
            result.append(integer_value)
    return tuple(result)


def _transition_positions(line):
    return np.flatnonzero(line[1:] != line[:-1]).astype(np.int64) + 1


def _analyze_input(grid):
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return None
    if grid.shape[0] == 0 or grid.shape[1] == 0:
        return None

    color_set = {int(value) for value in np.unique(grid)}
    candidates = []

    row_sequences = [_collapse_line(grid[row]) for row in range(grid.shape[0])]
    first_row_sequence = row_sequences[0]
    rows_valid = (
        len(first_row_sequence) >= 2
        and len(set(first_row_sequence)) == len(first_row_sequence)
        and set(first_row_sequence) == color_set
        and all(sequence == first_row_sequence for sequence in row_sequences)
    )
    if rows_valid:
        row_boundaries = np.stack(
            [_transition_positions(grid[row]) for row in range(grid.shape[0])],
            axis=0,
        )
        candidates.append(
            {
                "orientation": "horizontal",
                "sequence": first_row_sequence,
                "boundaries": row_boundaries,
            }
        )

    column_sequences = [
        _collapse_line(grid[:, column]) for column in range(grid.shape[1])
    ]
    first_column_sequence = column_sequences[0]
    columns_valid = (
        len(first_column_sequence) >= 2
        and len(set(first_column_sequence)) == len(first_column_sequence)
        and set(first_column_sequence) == color_set
        and all(
            sequence == first_column_sequence for sequence in column_sequences
        )
    )
    if columns_valid:
        column_boundaries = np.stack(
            [
                _transition_positions(grid[:, column])
                for column in range(grid.shape[1])
            ],
            axis=0,
        )
        candidates.append(
            {
                "orientation": "vertical",
                "sequence": first_column_sequence,
                "boundaries": column_boundaries,
            }
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _expected_output(analysis):
    values = np.asarray(analysis["sequence"], dtype=np.int8)
    if analysis["orientation"] == "horizontal":
        return values.reshape(1, -1)
    return values.reshape(-1, 1)


def _validate_count_argument(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng, seed_key, num_train, num_test):
    _validate_count_argument(num_train, "num_train", 2, 10)
    _validate_count_argument(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        # This deterministic cycle guarantees broad count diversity across
        # consecutive seeds while satisfying num_train > num_test.
        resolved_test = 1 + (seed_key % 4)
        quotient = seed_key // 4
        minimum_train = resolved_test + 1
        number_of_choices = 11 - minimum_train
        resolved_train = minimum_train + (quotient % number_of_choices)
    elif num_train is None:
        resolved_test = int(num_test)
        resolved_train = int(rng.integers(resolved_test + 1, 11))
    elif num_test is None:
        resolved_train = int(num_train)
        maximum_test = min(4, resolved_train - 1)
        resolved_test = int(rng.integers(1, maximum_test + 1))
    else:
        resolved_train = int(num_train)
        resolved_test = int(num_test)

    if resolved_train <= resolved_test:
        raise ValueError("num_train must be greater than num_test")

    return resolved_train, resolved_test


# The only puzzle-wide convention is the spatial compression rule.  The
# schedules below choose visible examples, not a hidden rule for the solver.
def _sample_specs(rng, train_count, test_count):
    mode = int(rng.integers(0, 3))  # shared axis, mixed, held-out recombination
    first_axis = ("horizontal", "vertical")[int(rng.integers(0, 2))]
    other_axis = "vertical" if first_axis == "horizontal" else "horizontal"
    first_count = int(rng.integers(2, 11))
    second_count = int(rng.integers(2, 10))
    if second_count >= first_count:
        second_count += 1

    if mode == 0:
        train = [(first_axis, first_count), (first_axis, second_count)]
        target = (first_axis, int(rng.integers(2, 11)))
    else:
        train = [(first_axis, first_count), (other_axis, second_count)]
        if mode == 2:
            target = ((first_axis, second_count), (other_axis, first_count))[
                int(rng.integers(0, 2))
            ]
        else:
            target = ((first_axis, other_axis)[int(rng.integers(0, 2))],
                      int(rng.integers(2, 11)))

    while len(train) < train_count:
        axis = first_axis if mode == 0 or int(rng.integers(0, 2)) == 0 else other_axis
        count = int(rng.integers(2, 11))
        if mode == 2 and (axis, count) == target:
            count = 2 + (count - 1) % 9
        train.append((axis, count))

    tests = [target]
    while len(tests) < test_count:
        axis = first_axis if mode == 0 or int(rng.integers(0, 2)) == 0 else other_axis
        if int(rng.integers(0, 3)) == 0:
            count = int(rng.choice([first_count, second_count]))
        else:
            count = int(rng.integers(2, 11))
        tests.append((axis, count))

    train = [train[int(i)] for i in rng.permutation(len(train))]
    return train, tests


def _sample_dimensions(rng, layer_count, forced_shape=None):
    # Shape is relative to the stack axis, so all three choices are available
    # for either orientation and both splits.
    shape = forced_shape
    if shape is None:
        shape = ("axis_long", "transverse_long", "square")[int(rng.integers(0, 3))]
    size_band = int(rng.integers(0, 3))
    ranges = ((5, 12), (13, 21), (22, 30))
    low, high = ranges[size_band]
    long_side = int(rng.integers(max(low, layer_count + 2), high + 1))
    if shape == "square":
        return long_side, long_side
    if shape == "axis_long":
        transverse = int(rng.integers(2, long_side))
        return long_side, transverse
    axis = int(rng.integers(layer_count + 1, long_side))
    return axis, long_side


def _initial_widths(rng, axis_length, layer_count, thickness_style):
    extra = axis_length - layer_count
    if thickness_style == "balanced":
        widths = np.full(layer_count, axis_length // layer_count, dtype=np.int64)
        for index in rng.permutation(layer_count)[:axis_length % layer_count]:
            widths[int(index)] += 1
    elif thickness_style == "dominant":
        widths = np.ones(layer_count, dtype=np.int64)
        dominant = int(rng.integers(0, layer_count))
        reserve = int(round(extra * float(rng.uniform(0.55, 0.85))))
        widths[dominant] += reserve
        if extra > reserve:
            widths += rng.multinomial(extra - reserve,
                                      np.full(layer_count, 1 / layer_count))
    else:  # one narrow band, possibly with other narrow bands
        widths = np.ones(layer_count, dtype=np.int64)
        narrow = int(rng.integers(0, layer_count))
        if layer_count > 1 and extra:
            weights = rng.dirichlet(np.ones(layer_count))
            weights[narrow] = 0
            weights /= weights.sum()
            widths += rng.multinomial(extra, weights)
    return widths


def _runs_overlap(old_cuts, new_cuts, axis_length):
    old = np.concatenate(([0], old_cuts, [axis_length]))
    new = np.concatenate(([0], new_cuts, [axis_length]))
    return bool(np.all(np.maximum(old[:-1], new[:-1]) <
                       np.minimum(old[1:], new[1:])))


def _boundary_profile(rng, axis_length, transverse, widths, style):
    cuts = np.cumsum(widths)[:-1].astype(np.int64)
    profile = np.empty((transverse, len(cuts)), dtype=np.int64)
    profile[0] = cuts
    if style == "localized" and transverse > 2:
        window = max(2, int(rng.integers(2, max(3, transverse // 2 + 1))))
        window_start = int(rng.integers(0, transverse - window + 1))
    else:
        window_start, window = 0, transverse
    favored = int(rng.integers(0, len(cuts)))
    slope_direction = (-1, 1)[int(rng.integers(0, 2))]
    phase_length = max(2, transverse // int(rng.integers(2, 5)))

    for position in range(1, transverse):
        previous = cuts.copy()
        if style == "straight":
            probability, moves = 0.0, 0
        elif style == "plateau":
            probability, moves = 0.18, 1
        elif style == "slope":
            probability, moves = 0.72, 1
        elif style == "wave":
            probability, moves = 0.78, 2
        elif style == "localized":
            probability, moves = (0.8 if window_start <= position < window_start + window
                                  else 0.0), 2
        else:
            probability, moves = 0.85, min(3, len(cuts))
        if style == "wave" and position % phase_length == 0:
            slope_direction *= -1
        if float(rng.random()) < probability:
            order = rng.permutation(len(cuts))
            if style in ("slope", "wave"):
                order = np.concatenate(([favored], order[order != favored]))
            for boundary in order[:moves]:
                boundary = int(boundary)
                if style in ("slope", "wave"):
                    directions = (slope_direction, -slope_direction)
                else:
                    first = (-1, 1)[int(rng.integers(0, 2))]
                    directions = (first, -first)
                for direction in directions:
                    proposed = cuts.copy()
                    proposed[boundary] += direction
                    if (proposed[0] < 1 or proposed[-1] >= axis_length
                            or np.any(proposed[1:] <= proposed[:-1])
                            or not _runs_overlap(previous, proposed, axis_length)):
                        continue
                    cuts = proposed
                    break
        profile[position] = cuts
    return profile


def _make_layer_grid(rng, orientation, layer_count, regime, forced_shape=None):
    axis_length, transverse = _sample_dimensions(rng, layer_count, forced_shape)
    thickness_style = ("balanced", "dominant", "narrow")[int(rng.integers(0, 3))]
    styles = ("straight", "plateau", "slope", "wave", "localized", "rough")
    probabilities = ((.15, .19, .17, .17, .15, .17) if regime == "source_train"
                     else (.07, .12, .19, .19, .20, .23))
    style = styles[int(rng.choice(len(styles), p=probabilities))]
    widths = _initial_widths(rng, axis_length, layer_count, thickness_style)
    profile = _boundary_profile(rng, axis_length, transverse, widths, style)
    colors = tuple(int(x) for x in rng.choice(10, size=layer_count, replace=False))
    stack = np.empty((transverse, axis_length), dtype=np.int8)
    for transverse_index, cuts in enumerate(profile):
        ends = np.concatenate(([0], cuts, [axis_length]))
        for layer_index in range(layer_count):
            stack[transverse_index, ends[layer_index]:ends[layer_index + 1]] = colors[layer_index]
    grid = stack if orientation == "horizontal" else stack.T.copy()
    return grid, colors


def _generate_pair(rng, orientation, layer_count, regime, forced_shape,
                   prior_sequences, used_inputs):
    for _ in range(32):
        grid, colors = _make_layer_grid(rng, orientation, layer_count, regime,
                                        forced_shape)
        # Repeating a color order with fresh geometry sometimes demonstrates
        # that thickness and boundary shape do not affect the answer.
        compatible = [sequence for sequence in prior_sequences if len(sequence) == layer_count]
        if compatible and float(rng.random()) < 0.22:
            replacement = compatible[int(rng.integers(0, len(compatible)))]
            mapping = np.empty(10, dtype=np.int8)
            for old, new in zip(colors, replacement):
                mapping[old] = new
            grid = mapping[grid]
        analysis = _analyze_input(grid)
        if analysis is None or analysis["orientation"] != orientation:
            continue
        output = _expected_output(analysis)
        signature = _grid_signature(grid)
        if signature in used_inputs or _is_source_pair(grid, output):
            continue
        return {"input": grid, "output": output}, analysis["sequence"]
    raise RuntimeError("failed to construct a distinct layer stack")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, int(seed), num_train, num_test)
    train_specs, test_specs = _sample_specs(rng, train_count, test_count)
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count,
                              dtype=np.int64)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    sequences = []
    pair_index = 0
    # One training example explicitly breaks the aspect-ratio shortcut.
    forced_index = int(rng.integers(0, train_count))
    forced_shape = ("square", "transverse_long")[int(rng.integers(0, 2))]
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for local_index, (orientation, layer_count) in enumerate(specs):
            pair_rng = np.random.default_rng(int(pair_seeds[pair_index]))
            pair_index += 1
            shape = forced_shape if split == "train" and local_index == forced_index else None
            pair, sequence = _generate_pair(
                pair_rng, orientation, layer_count,
                "source_train" if split == "train" else "target_test",
                shape, sequences, used_inputs,
            )
            puzzle[split].append(pair)
            sequences.append(sequence)
            used_inputs.add(_grid_signature(pair["input"]))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
            "puzzle must contain exactly train and test")
    _ensure(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
            "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _ensure(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
            "invalid pair counts")
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _ensure(isinstance(pair, dict) and set(pair) == {"input", "output"},
                    f"{split} {index}: malformed pair")
            for name in ("input", "output"):
                grid = pair[name]
                _ensure(isinstance(grid, np.ndarray) and grid.ndim == 2,
                        f"{split} {index}: {name} must be a 2D array")
                _ensure(all(1 <= d <= 30 for d in grid.shape),
                        f"{split} {index}: illegal grid size")
                _ensure(np.issubdtype(grid.dtype, np.integer),
                        f"{split} {index}: grid must contain integers")
                _ensure(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                        f"{split} {index}: illegal color")
            grid, output = pair["input"], pair["output"]
            input_key = _grid_signature(grid)
            pair_key = (input_key, _grid_signature(output))
            _ensure(input_key not in inputs and pair_key not in pairs,
                    "duplicate generated pair")
            inputs.add(input_key)
            pairs.add(pair_key)
            analysis = _analyze_input(grid)
            _ensure(analysis is not None, f"{split} {index}: no unique stacking axis")
            sequence, cuts = analysis["sequence"], analysis["boundaries"]
            _ensure(2 <= len(sequence) <= 10 and len(set(sequence)) == len(sequence),
                    f"{split} {index}: invalid layers")
            _ensure(set(sequence) == set(int(x) for x in np.unique(grid)),
                    f"{split} {index}: unrelated color")
            _ensure(cuts.shape[1] == len(sequence) - 1,
                    f"{split} {index}: invalid boundaries")
            axis_length = grid.shape[1] if analysis["orientation"] == "horizontal" else grid.shape[0]
            _ensure(all(_runs_overlap(cuts[i - 1], cuts[i], axis_length)
                        for i in range(1, len(cuts))),
                    f"{split} {index}: disconnected layer")
            _ensure(np.array_equal(output, _expected_output(analysis)),
                    f"{split} {index}: incorrect compression")
            _ensure(not np.array_equal(grid, output) and not _is_source_pair(grid, output),
                    f"{split} {index}: degenerate or source copy")
            analyses[split].append(analysis)
    train_axes = {analysis["orientation"] for analysis in analyses["train"]}
    test_axes = {analysis["orientation"] for analysis in analyses["test"]}
    _ensure(test_axes <= train_axes, "a test orientation is absent from training")
    _ensure(len({len(analysis["sequence"]) for analysis in analyses["train"]}) >= 2,
            "training must show that strip length follows layer count")
    _ensure(any((pair["input"].shape[1] <= pair["input"].shape[0]
                 if analysis["orientation"] == "horizontal"
                 else pair["input"].shape[0] <= pair["input"].shape[1])
                for pair, analysis in zip(puzzle["train"], analyses["train"])),
            "training must expose stacking axis beyond canvas aspect")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _ensure(len(train_counts) >= 4 and len(test_counts) >= 3,
            "automatic pair counts lack diversity")
    _ensure(len({_puzzle_signature(p) for p in puzzles}) >= 40,
            "different seeds repeat puzzles")
    orientations, counts, shapes, axis_shapes = set(), set(), set(), set()
    axis_count_categories = set()
    sizes, sequences, inputs, boundary_patterns = set(), set(), set(), set()
    same_axis_puzzles = mixed_axis_puzzles = smaller_test_puzzles = 0
    heldout_recombinations = 0
    within_count = within_size = within_shape = compact_large_puzzles = 0
    straight = variable = 0
    repeated_sequences = 0
    one_cell = dominant = balanced = 0
    long_plateau = gradual_slope = wavy = localized = rough = 0
    old_prefix = boundary_jump = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_axes = set()
        train_counts_here, sizes_here, shapes_here = set(), set(), set()
        size_bands_here = set()
        seen_sequences = set()
        train_categories = set()
        test_categories = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                analysis = _analyze_input(grid)
                _ensure(analysis is not None and np.array_equal(
                    pair["output"], _expected_output(analysis)),
                    "self-test disagrees with visible transects")
                axis = analysis["orientation"]
                count = len(analysis["sequence"])
                h, w = grid.shape
                shape = "square" if h == w else ("tall" if h > w else "wide")
                relative_shape = ("axis_long" if (w > h if axis == "horizontal" else h > w)
                                  else "square" if h == w else "transverse_long")
                orientations.add((split, axis))
                counts.add((split, count))
                axis_count_categories.add((split, axis, count))
                shapes.add((split, shape))
                axis_shapes.add((split, axis, relative_shape))
                sizes.add((h, w))
                inputs.add(_grid_signature(grid))
                sequences.add(analysis["sequence"])
                boundary_patterns.add(tuple(int(x) for x in analysis["boundaries"].ravel()))
                cuts = analysis["boundaries"]
                if np.all(np.ptp(cuts, axis=0) == 0):
                    straight += 1
                else:
                    variable += 1
                axis_length = w if axis == "horizontal" else h
                widths = np.diff(np.concatenate((
                    np.zeros((len(cuts), 1), dtype=np.int64), cuts,
                    np.full((len(cuts), 1), axis_length, dtype=np.int64),
                ), axis=1), axis=1)
                mean_widths = widths.mean(axis=0)
                one_cell += bool(np.any(widths == 1))
                dominant += bool(mean_widths.max() / axis_length >= .4)
                balanced += bool(mean_widths.max() / mean_widths.min() <= 1.5)
                steps = np.diff(cuts, axis=0)
                activity = np.any(steps != 0, axis=1)
                boundary_jump += bool(np.any(np.abs(steps) > 1))
                no_move_run = longest_no_move = 0
                for moving in activity:
                    no_move_run = 0 if moving else no_move_run + 1
                    longest_no_move = max(longest_no_move, no_move_run)
                long_plateau += bool(longest_no_move >= 3 and np.any(activity))
                wavy += any(np.any(steps[:, j] > 0) and np.any(steps[:, j] < 0)
                            and np.ptp(cuts[:, j]) >= 2 for j in range(cuts.shape[1]))
                localized += bool(np.count_nonzero(activity) >= 2 and
                                  np.ptp(np.flatnonzero(activity)) + 1
                                  <= max(3, int(.45 * len(activity))))
                rough += bool(len(activity) >= 4 and activity.mean() >= .5)
                gradual_slope += any(
                    np.all(steps[i:i + 3, j] == direction)
                    for j in range(cuts.shape[1])
                    for direction in (-1, 1)
                    for i in range(max(0, len(steps) - 2))
                )
                if len(cuts) >= 4 and cuts.shape[1] >= 3:
                    identity = np.eye(cuts.shape[1], dtype=np.int64)
                    old_prefix += all(np.array_equal(cuts[j], cuts[0] + identity[j - 1])
                                      for j in range(1, 4))
                if analysis["sequence"] in seen_sequences:
                    repeated_sequences += 1
                seen_sequences.add(analysis["sequence"])
                if split == "train":
                    train_axes.add(axis)
                    train_counts_here.add(count)
                    sizes_here.add(max(h, w))
                    shapes_here.add(shape)
                    size_bands_here.add("compact" if max(h, w) <= 12 else
                                        "large" if max(h, w) >= 22 else "medium")
                    train_categories.add((axis, count))
                else:
                    test_categories.add((axis, count))
        same_axis_puzzles += len(train_axes) == 1
        mixed_axis_puzzles += len(train_axes) == 2
        smaller_test_puzzles += min(len(_analyze_input(pair["input"])["sequence"])
                                    for pair in puzzle["test"]) < max(train_counts_here)
        heldout_recombinations += any(
            (axis, count) not in train_categories and axis in train_axes
            and count in train_counts_here for axis, count in test_categories
        )
        within_count += len(train_counts_here) > 1
        within_size += len(sizes_here) > 1
        within_shape += len(shapes_here) > 1
        compact_large_puzzles += "compact" in size_bands_here and "large" in size_bands_here
    _ensure(len(orientations) == 4 and len(counts) >= 16,
            "split/axis/count coverage too narrow")
    _ensure({("train", "horizontal", 4), ("test", "vertical", 4)} <= axis_count_categories
            and len(axis_count_categories) >= 32,
            "orientation and layer count combinations too narrow")
    _ensure({("train", "tall"), ("train", "wide"), ("train", "square"),
             ("test", "tall"), ("test", "wide"), ("test", "square")} <= shapes,
            "canvas shapes missing from a split")
    _ensure({("train", "horizontal", "transverse_long"),
             ("train", "vertical", "transverse_long"),
             ("test", "horizontal", "transverse_long"),
             ("test", "vertical", "transverse_long")} <= axis_shapes,
            "opposite-aspect stacks missing")
    _ensure(len(sizes) >= 60 and len(inputs) >= 180 and len(boundary_patterns) >= 100,
            "geometry diversity too narrow")
    _ensure(len(sequences) >= 150, "palette/order diversity too narrow")
    _ensure(same_axis_puzzles >= 3 and mixed_axis_puzzles >= 3 and smaller_test_puzzles >= 3,
            "puzzle-level curricula lack diversity")
    _ensure(heldout_recombinations >= 5 and compact_large_puzzles >= 15,
            "recombination or compact/large coverage too narrow")
    _ensure(within_count == 42 and within_size >= 25 and within_shape >= 20,
            "within-puzzle diversity too narrow")
    _ensure(straight >= 3 and variable >= 100 and repeated_sequences >= 2,
            "straight, jagged, or repeated-sequence cases absent")
    _ensure(one_cell >= 100 and dominant >= 40 and balanced >= 10,
            "layer thickness regimes too narrow")
    _ensure(long_plateau >= 30 and gradual_slope >= 10 and wavy >= 30
            and localized >= 10 and rough >= 40,
            "boundary geometry regimes too narrow")
    _ensure(old_prefix == 0 and boundary_jump == 0,
            "boundary construction artifact returned")
    _ensure(_puzzles_equal(generate_puzzle(918273, 8, 3),
                           generate_puzzle(918273, 8, 3)),
            "explicit generation is not deterministic")
    _ensure(_puzzles_equal(generate_puzzle(271828), generate_puzzle(271828)),
            "automatic generation is not deterministic")
    for seed, n_train, n_test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                                  (2001, None, 1), (2002, None, 4),
                                  (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, n_train, n_test)
        validate_generated_puzzle(puzzle)
        if n_train is not None:
            _ensure(len(puzzle["train"]) == n_train, "ignored training count")
        if n_test is not None:
            _ensure(len(puzzle["test"]) == n_test, "ignored test count")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(3001, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"illegal count accepted: {kwargs}")
