import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _coerce_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train_count = _coerce_count(num_train, "num_train", 2, 10)
    test_count = _coerce_count(num_test, "num_test", 1, 4)
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif train_count is None:
        train_count = test_count + 1 + seed % (10 - test_count)
    elif test_count is None:
        test_count = 1 + seed % min(4, train_count - 1)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


_DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_CARDINALS = ((0, 1), (0, -1), (1, 0), (-1, 0))
_SKEWS = ((1, 2), (1, -2), (-1, 2), (-1, -2),
          (2, 1), (2, -1), (-2, 1), (-2, -1))


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _schedule(rng, train_count, test_count, broad, subtype):
    upper = 7 if broad and subtype == "skew" else 9
    low = int(rng.integers(1, min(4, upper + 1)))
    high = int(rng.integers(low + 1, upper + 1))
    train_steps = [low, high if rng.random() >= 0.2 else low]
    for _ in range(train_count - 2):
        if rng.random() < 0.35:
            train_steps.append(_choice(rng, train_steps[:2]))
        else:
            train_steps.append(int(rng.integers(1, upper + 1)))
    rng.shuffle(train_steps)

    test_steps = []
    for _ in range(test_count):
        smallest, largest = min(train_steps), max(train_steps)
        modes = ["familiar", "random"]
        if largest - smallest > 1:
            modes.append("intermediate")
        if smallest > 1:
            modes.append("smaller")
        if largest < upper:
            modes.append("larger")
        mode = _choice(rng, modes)
        if mode == "familiar":
            step = _choice(rng, train_steps)
        elif mode == "intermediate":
            step = int(rng.integers(smallest + 1, largest))
        elif mode == "smaller":
            step = int(rng.integers(1, smallest))
        elif mode == "larger":
            step = int(rng.integers(largest + 1, upper + 1))
        else:
            step = int(rng.integers(1, upper + 1))
        test_steps.append(step)
    return train_steps, test_steps


def _directions(rng, train_count, test_count, broad, subtype):
    if broad:
        other = _CARDINALS if subtype == "cardinal" else _SKEWS
        shown = [_choice(rng, _DIAGONALS), _choice(rng, other)]
        directions = shown + [_choice(rng, shown) for _ in range(train_count - 2)]
        rng.shuffle(directions)
        tests = [_choice(rng, shown) if rng.random() < 0.6
                 else _choice(rng, other) for _ in range(test_count)]
    else:
        mode = int(rng.integers(3))
        first = _choice(rng, _DIAGONALS)
        if mode == 0:
            directions = [first] * train_count
            tests = [first] * test_count
        elif mode == 1:
            second = _choice(rng, tuple(x for x in _DIAGONALS if x != first))
            directions = [first, second] + [
                _choice(rng, (first, second)) for _ in range(train_count - 2)]
            rng.shuffle(directions)
            tests = [_choice(rng, (first, second)) for _ in range(test_count)]
        else:
            held_out = _choice(rng, tuple(x for x in _DIAGONALS if x != first))
            available = tuple(x for x in _DIAGONALS if x != held_out)
            directions = [first] + [
                _choice(rng, available) for _ in range(train_count - 1)]
            tests = [held_out] * test_count
    return directions, tests


def _make_pair(rng, step, direction, background, seed_color, continuation,
               forced_prefix=None):
    delta = (step * direction[0], step * direction[1])
    magnitude = max(abs(delta[0]), abs(delta[1]))
    maximum_prefix = min(5, 29 // magnitude)
    prefix_options = list(range(2, maximum_prefix + 1))
    weights = np.array([0.24, 0.40, 0.23, 0.13])[:len(prefix_options)]
    prefix = (forced_prefix if forced_prefix is not None else
              int(rng.choice(prefix_options, p=weights / weights.sum())))
    maximum_added = min(9, 29 // magnitude - prefix + 1)
    compact = magnitude <= 3 and rng.random() < 0.14
    added_target = int(rng.integers(1, min(maximum_added, 3) + 1)
                       if compact else rng.integers(1, maximum_added + 1))
    last_index = prefix - 1 + added_target
    active = [axis for axis in range(2) if delta[axis] != 0]
    anchor = _choice(rng, active)
    clipping_axis = _choice(rng, active)
    if len(active) == 2 and (compact or rng.random() < 0.25):
        clip_both = True
    else:
        clip_both = False

    dimensions = [0, 0]
    starts = [0, 0]
    for axis in range(2):
        component = abs(delta[axis])
        if not component:
            dimensions[axis] = int(rng.integers(5, 31))
            starts[axis] = int(rng.integers(dimensions[axis]))
            continue
        latest_start = 29 - last_index * component
        if axis == anchor or compact:
            latest_start = min(latest_start, component - 1)
        starts[axis] = int(rng.integers(latest_start + 1))
        last_position = starts[axis] + last_index * component
        next_position = last_position + component
        upper = min(30, next_position) if axis == clipping_axis or clip_both else 30
        dimensions[axis] = int(rng.integers(last_position + 1, upper + 1))

    height, width = dimensions
    input_grid = np.full((height, width), background, dtype=np.int8)

    def position(index):
        return tuple(
            starts[axis] + index * abs(delta[axis])
            if delta[axis] >= 0 else dimensions[axis] - 1 - starts[axis] - index * abs(delta[axis])
            for axis in range(2)
        )

    for index in range(prefix):
        input_grid[position(index)] = seed_color
    output_grid = input_grid.copy()
    index = prefix
    while True:
        row, col = position(index)
        if not (0 <= row < height and 0 <= col < width):
            break
        output_grid[row, col] = continuation
        index += 1
    _require(index - prefix == added_target, "construction did not clip as planned")
    return input_grid, output_grid


def _value_signature(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _puzzle_signature(puzzle):
    return tuple((split, tuple((_value_signature(pair["input"]),
                                _value_signature(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def _is_original_pair(input_grid, output_grid):
    if input_grid.shape != (15, 15) or output_grid.shape != (15, 15):
        return False
    originals = (
        (((0, 0), (4, 4), (8, 8)), ((12, 12),)),
        (((1, 1), (3, 3), (5, 5)), ((7, 7), (9, 9), (11, 11), (13, 13))),
        (((0, 0), (3, 3), (6, 6)), ((9, 9), (12, 12))),
    )
    for seeds, additions in originals:
        source_input = np.zeros((15, 15), dtype=np.int8)
        source_output = np.zeros((15, 15), dtype=np.int8)
        for position in seeds:
            source_input[position] = source_output[position] = 1
        for position in additions:
            source_output[position] = 2
        if np.array_equal(input_grid, source_input) and np.array_equal(output_grid, source_output):
            return True
    return False


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    continuation = int(rng.integers(10))
    colors = [color for color in range(10) if color != continuation]
    stable_palette = rng.random() < 0.5
    palette = tuple(int(color) for color in rng.choice(colors, size=2, replace=False))
    classic = rng.random() < 0.17
    if classic:
        stable_palette = True
    broad = not classic and rng.random() < 0.4
    subtype = _choice(rng, ("cardinal", "skew")) if broad else "diagonal"
    train_steps, test_steps = _schedule(rng, train_count, test_count, broad, subtype)
    train_directions, test_directions = _directions(rng, train_count, test_count, broad, subtype)
    if classic:
        train_directions = [(1, 1)] * train_count
        test_directions = [(1, 1)] * test_count
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, steps, directions in (("train", train_steps, train_directions),
                                     ("test", test_steps, test_directions)):
        for index, (step, direction) in enumerate(zip(steps, directions)):
            for attempt in range(512):
                if stable_palette:
                    background, seed_color = palette
                else:
                    background, seed_color = (int(color) for color in
                                              rng.choice(colors, size=2, replace=False))
                    if split == "train" and index == 1 and (background, seed_color) == palette:
                        continue
                    if split == "train" and index == 0:
                        palette = (background, seed_color)
                input_grid, output_grid = _make_pair(
                    rng, step, direction, background, seed_color, continuation,
                    forced_prefix=3 if classic else None)
                signature = _value_signature(input_grid)
                if signature not in seen_inputs and not _is_original_pair(input_grid, output_grid):
                    break
            else:
                raise ValueError("could not construct distinct pairs")
            seen_inputs.add(signature)
            puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _inside(position, shape):
    return all(0 <= position[axis] < shape[axis] for axis in range(2))


def _analyze_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    _require(input_grid.shape == output_grid.shape, "shape changed")
    values, counts = np.unique(input_grid, return_counts=True)
    _require(len(values) == 2, "input must have background and one seed color")
    seed_index = int(np.argmin(counts))
    _require(counts[seed_index] < counts[1 - seed_index], "background must dominate")
    seed_color = int(values[seed_index])
    background = int(values[1 - seed_index])
    coordinates = [tuple(map(int, position)) for position in np.argwhere(input_grid == seed_color)]
    _require(2 <= len(coordinates) <= 5, "prefix length must be 2 through 5")
    endpoints = []
    for first, second in ((min(coordinates), max(coordinates)),
                          (max(coordinates), min(coordinates))):
        delta = (second[0] - first[0], second[1] - first[1])
        if len(coordinates) > 2:
            delta = tuple(component // (len(coordinates) - 1) for component in delta)
        if delta == (0, 0):
            continue
        ordered = [tuple(first[axis] + term * delta[axis] for axis in range(2))
                   for term in range(len(coordinates))]
        if set(ordered) != set(coordinates):
            continue
        predecessor = tuple(first[axis] - delta[axis] for axis in range(2))
        successor = tuple(ordered[-1][axis] + delta[axis] for axis in range(2))
        if not _inside(predecessor, input_grid.shape) and _inside(successor, input_grid.shape):
            endpoints.append((ordered, delta))
    _require(len(endpoints) == 1, "seed vector or direction is ambiguous")
    ordered, delta = endpoints[0]
    unit = tuple(component // np.gcd(*[abs(x) for x in delta]) for component in delta)
    _require(unit in _DIAGONALS + _CARDINALS + _SKEWS,
             "unsupported vector geometry")
    _require(max(abs(x) for x in delta) <= 18, "displacement too large")
    new_colors = set(int(value) for value in np.unique(output_grid)) - set(map(int, values))
    _require(len(new_colors) == 1 and len(np.unique(output_grid)) == 3,
             "output must introduce one continuation color")
    continuation = new_colors.pop()
    expected = input_grid.copy()
    additions = []
    position = tuple(ordered[-1][axis] + delta[axis] for axis in range(2))
    while _inside(position, input_grid.shape):
        expected[position] = continuation
        additions.append(position)
        position = tuple(position[axis] + delta[axis] for axis in range(2))
    _require(additions and np.array_equal(expected, output_grid),
             "output must contain the complete boundary-clipped continuation")
    return {"background": background, "seed": seed_color, "continuation": continuation,
            "prefix": len(ordered), "step": int(np.gcd(*[abs(x) for x in delta])),
            "unit": unit, "delta": delta, "added_count": len(additions),
            "seed_coordinates": tuple(ordered), "added_coordinates": tuple(additions)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test only")
    _require(all(isinstance(puzzle[split], list) for split in ("train", "test")),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output only")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be two-dimensional arrays")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(all(1 <= size <= 30 for size in grid.shape), "invalid dimensions")
                _require(0 <= int(grid.min()) <= int(grid.max()) <= 9, "invalid colors")
            _require(pair["input"].dtype == pair["output"].dtype, "dtype changed")
            input_key = _value_signature(pair["input"])
            pair_key = (input_key, _value_signature(pair["output"]))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate pair or input")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not _is_original_pair(pair["input"], pair["output"]), "source copy")
            information[split].append(_analyze_pair(pair))
    continuation = information["train"][0]["continuation"]
    for item in information["train"] + information["test"]:
        _require(item["continuation"] == continuation, "learned color changed")
        _require(continuation not in (item["background"], item["seed"]),
                 "continuation color must be reserved from inputs")
    train_units = {item["unit"] for item in information["train"]}
    test_units = {item["unit"] for item in information["test"]}
    if any(unit not in _DIAGONALS for unit in test_units):
        _require(any(unit not in _DIAGONALS for unit in train_units),
                 "non-diagonal vector must be demonstrated in training")
    if any(unit in _SKEWS for unit in test_units):
        _require(any(unit in _SKEWS for unit in train_units),
                 "skew vectors must be demonstrated in training")
    if any(unit in _CARDINALS for unit in test_units):
        _require(any(unit in _CARDINALS for unit in train_units),
                 "cardinal vectors must be demonstrated in training")


def _predict_from_training(train_pairs, test_input):
    colors = set()
    for pair in train_pairs:
        colors.update(set(int(value) for value in np.unique(pair["output"])) -
                      set(int(value) for value in np.unique(pair["input"])))
    _require(len(colors) == 1, "training has no unique output color")
    continuation = colors.pop()
    values, counts = np.unique(test_input, return_counts=True)
    _require(len(values) == 2, "test has no seed role")
    points = [tuple(map(int, point)) for point in np.argwhere(test_input == values[np.argmin(counts)])]
    points.sort()
    for sequence in (points, points[::-1]):
        delta = tuple(sequence[1][axis] - sequence[0][axis] for axis in range(2))
        if all(tuple(sequence[0][axis] + index * delta[axis] for axis in range(2)) == point
               for index, point in enumerate(sequence)):
            predecessor = tuple(sequence[0][axis] - delta[axis] for axis in range(2))
            next_position = tuple(sequence[-1][axis] + delta[axis] for axis in range(2))
            if not _inside(predecessor, test_input.shape) and _inside(next_position, test_input.shape):
                result = test_input.copy()
                while _inside(next_position, result.shape):
                    result[next_position] = continuation
                    next_position = tuple(next_position[axis] + delta[axis] for axis in range(2))
                return result
    raise AssertionError("test input is ambiguous")


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    palettes, shapes, layouts, steps = set(), set(), set(), set()
    counts_by_split = {"train": set(), "test": set()}
    prefixes = set()
    units = set()
    same_step_different_length = 0
    palette_modes = set()
    direction_modes = set()
    schedule_modes = set()
    combinations = 0
    compact_puzzles = 0
    source_like_puzzles = 0
    multi_step_training = 0
    matched_vector_boundary_contrasts = 0
    palette_direction_combinations = 0
    vector_by_split = {"train": set(), "test": set()}
    sparse_three_by_split = {"train": 0, "test": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_signature(puzzle))
        details = {split: [_analyze_pair(pair) for pair in puzzle[split]]
                   for split in ("train", "test")}
        all_details = details["train"] + details["test"]
        train_steps = {item["step"] for item in details["train"]}
        test_steps = {item["step"] for item in details["test"]}
        schedule_modes.update("familiar" if step in train_steps else
                              "smaller" if step < min(train_steps) else
                              "larger" if step > max(train_steps) else "intermediate"
                              for step in test_steps)
        train_units = {item["unit"] for item in details["train"]}
        test_units = {item["unit"] for item in details["test"]}
        if len(train_steps) >= 3:
            multi_step_training += 1
        direction_modes.add("common" if len(train_units | test_units) == 1 else
                            "held_out" if test_units - train_units else "mixed")
        role_pairs = {(item["background"], item["seed"]) for item in all_details}
        palette_modes.add(len(role_pairs) > 1)
        if len(role_pairs) > 1 and len(train_units | test_units) > 1:
            palette_direction_combinations += 1
        if len(role_pairs) == 1 and train_units | test_units == {(1, 1)} \
                and all(item["prefix"] == 3 for item in all_details):
            source_like_puzzles += 1
        if any(len({item["added_count"] for item in all_details
                    if (item["delta"], item["prefix"]) == (example["delta"], example["prefix"])}) > 1
               for example in all_details):
            matched_vector_boundary_contrasts += 1
        for step in train_steps | test_steps:
            if len({item["added_count"] for item in all_details if item["step"] == step}) > 1:
                same_step_different_length += 1
                break
        if len({item["prefix"] for item in all_details}) > 1 and len(train_units | test_units) > 1 \
                and len({item["added_count"] for item in all_details}) > 1:
            combinations += 1
        for split in ("train", "test"):
            if any(max(pair["input"].shape) <= 6 for pair in puzzle[split]):
                compact_puzzles += 1
            for pair, item in zip(puzzle[split], details[split]):
                predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
                _require(np.array_equal(predicted, pair["output"]),
                         "training-derived prediction differs from ground truth")
                changed = {tuple(map(int, point)) for point in np.argwhere(pair["input"] != pair["output"])}
                _require(changed == set(item["added_coordinates"]), "only terms may change")
                _require(all(pair["output"][point] == item["seed"] for point in item["seed_coordinates"]),
                         "seed terms must be preserved")
                _require(not _is_original_pair(pair["input"], pair["output"]), "source copy")
                counts_by_split[split].add(min(item["added_count"], 3))
                prefixes.add(item["prefix"])
                units.add(item["unit"])
                vector_by_split[split].add(
                    "diagonal" if item["unit"] in _DIAGONALS else
                    "cardinal" if item["unit"] in _CARDINALS else "skew")
                if item["step"] in (8, 9) and item["prefix"] == 3:
                    sparse_three_by_split[split] += 1
                steps.add(item["step"])
                palettes.add((item["background"], item["seed"], item["continuation"]))
                shapes.add(pair["input"].shape)
                layouts.add((pair["input"].shape, item["seed_coordinates"], item["added_coordinates"]))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(signatures) >= 40 and len(palettes) >= 25, "puzzle/palette diversity")
    _require(len(shapes) >= 30 and len(layouts) >= 100, "shape/layout diversity")
    _require(steps == set(range(1, 10)), "spacing diversity")
    _require(prefixes == {2, 3, 4, 5}, "prefix diversity")
    _require(counts_by_split["train"] == {1, 2, 3} and counts_by_split["test"] == {1, 2, 3},
             "boundary length must vary in both splits")
    _require(all(mode in schedule_modes for mode in ("familiar", "smaller", "larger", "intermediate")),
             "spacing schedule diversity")
    _require(all(mode in direction_modes for mode in ("common", "mixed", "held_out")),
             "direction regime diversity")
    _require(all(mode in palette_modes for mode in (False, True)), "palette regime diversity")
    _require(any(unit in _CARDINALS for unit in units) and any(unit in _SKEWS for unit in units),
             "broader vector diversity")
    _require(same_step_different_length >= 6 and combinations >= 8,
             "insufficient independent within-puzzle variation")
    _require(multi_step_training >= 5 and matched_vector_boundary_contrasts >= 8,
             "insufficient varied schedules or direct boundary contrasts")
    _require(compact_puzzles >= 2 and source_like_puzzles >= 2,
             "compact and source-like regimes must remain available")
    _require(palette_direction_combinations >= 2,
             "color and direction variation should combine")
    for split in ("train", "test"):
        _require(vector_by_split[split] == {"diagonal", "cardinal", "skew"}
                 and sparse_three_by_split[split] >= 1,
                 "vector and sparse-spacing coverage must reach both splits")
    _require(_puzzle_signature(generate_puzzle(17)) == _puzzle_signature(generate_puzzle(17)),
             "automatic determinism")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (3, 1), (7, 2), (9, 4)):
        puzzle = generate_puzzle(1200 + train_count * 10 + test_count, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count),
                 "explicit counts not honored")
        _require(_puzzle_signature(puzzle) == _puzzle_signature(
            generate_puzzle(1200 + train_count * 10 + test_count, train_count, test_count)),
            "explicit determinism")
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(1300 + (train_count or 0) + (test_count or 0), train_count, test_count)
        _require(train_count is None or len(puzzle["train"]) == train_count,
                 "partial train count changed")
        _require(test_count is None or len(puzzle["test"]) == test_count,
                 "partial test count changed")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.0},
                   {"num_test": True}):
        try:
            generate_puzzle(1400, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
