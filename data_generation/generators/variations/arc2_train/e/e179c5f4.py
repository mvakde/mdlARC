"""Single-seed reflected zigzag ARC puzzle generator."""

import numpy as np


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _orientation(code):
    return code // 4, bool((code // 2) % 2), bool(code % 2)


def _make_pair(length, width, orientation, background, path, fill):
    axis, from_low, lane_low = _orientation(orientation)
    shape = (length, width) if axis == 0 else (width, length)
    source = np.full(shape, background, dtype=np.int8)
    target = np.full(shape, fill, dtype=np.int8)
    period = 2 * (width - 1)
    for step in range(length):
        phase = step % period
        lane = min(phase, period - phase)
        along = step if from_low else length - 1 - step
        across = lane if lane_low else width - 1 - lane
        cell = (along, across) if axis == 0 else (across, along)
        target[cell] = path
        if step == 0:
            source[cell] = path
    return {"input": source, "output": target}


def _length_options(width):
    period = 2 * (width - 1)
    return {
        "partial": list(range(width + 1, min(period, 30) + 1)),
        "return": [period + 1] if period + 1 <= 30 else [],
        "extended": list(range(period + 2, min(2 * period, 30) + 1)),
        "cycles": list(range(2 * period + 1, 31)),
    }


def _length(rng, width, full_reflection=False):
    options = _length_options(width)
    weights = {"partial": 4, "return": 2, "extended": 3, "cycles": 2}
    labels = [name for name in weights if options[name] and (not full_reflection or name != "partial")]
    probabilities = np.asarray([weights[name] for name in labels], dtype=float)
    probabilities /= probabilities.sum()
    label = labels[int(rng.choice(len(labels), p=probabilities))]
    return int(rng.choice(options[label]))


def _train_widths(rng, count):
    # The source-like 2, 3, 4 sequence remains a substantial regime.
    if rng.random() < 0.28:
        widths = [4, 2]
        if count >= 3:
            widths.append(3)
        while len(widths) < count:
            widths.append(int(rng.integers(2, 5)))
    else:
        anchor = int(rng.integers(3, 12))
        alternatives = [w for w in range(2, 13) if w != anchor]
        widths = [anchor, int(rng.choice(alternatives))]
        while len(widths) < count:
            widths.append(int(rng.integers(2, 13)))
    return widths


def _test_width(rng, train_widths):
    largest = max(train_widths)
    regimes = ["near", "wider", "shown", "narrower"]
    weights = np.asarray([3, 3, 3, 2], dtype=float)
    if largest <= 2:
        weights[3] = 0
    weights /= weights.sum()
    regime = regimes[int(rng.choice(4, p=weights))]
    if regime == "near":
        return largest + 1
    if regime == "wider":
        if rng.random() < 0.22:
            return 15
        return int(rng.integers(largest + 1, 16))
    if regime == "shown":
        return int(rng.choice(train_widths))
    return int(rng.integers(2, largest))


def _signature(grid):
    return grid.shape, tuple(int(cell) for cell in grid.flat)


def _source_copy(pair):
    source, target = pair["input"], pair["output"]
    if source.shape[0] != 10 or source.shape[1] not in (2, 3, 4, 5):
        return False
    original = _make_pair(10, source.shape[1], 1, 0, 1, 8)
    return np.array_equal(source, original["input"]) and np.array_equal(target, original["output"])


def _different_roles(rng, pool, first):
    eligible = [roles for roles in pool if roles[0] != first[0] and roles[1] != first[1]]
    return eligible[int(rng.integers(len(eligible)))]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, num_train, num_test)

    # Fill is learned from training. The marker, background, and travel
    # direction are visible in each pair's input.
    fill = int(rng.integers(10))
    palette = [color for color in range(10) if color != fill]
    role_pool = [(background, path) for background in palette for path in palette if background != path]
    simple = bool(rng.random() < 0.18)
    fixed_colors = simple or bool(rng.random() < 0.10)
    fixed_orientation = simple or bool(rng.random() < 0.10)
    total = train_count + test_count

    if fixed_colors:
        roles = [role_pool[int(rng.integers(len(role_pool)))]] * total
    else:
        first = role_pool[int(rng.integers(len(role_pool)))]
        roles = [first, _different_roles(rng, role_pool, first)]
        roles.extend(role_pool[int(rng.integers(len(role_pool)))] for _ in range(total - 2))

    if fixed_orientation:
        orientations = [int(rng.integers(8))] * total
    else:
        first = int(rng.integers(8))
        other_axis = list(range(4, 8)) if first < 4 else list(range(4))
        orientations = [first, int(rng.choice(other_axis))]
        orientations.extend(int(rng.integers(8)) for _ in range(total - 2))
        if rng.random() < 0.42:
            unused = [code for code in range(8) if code not in orientations[:train_count]]
            if unused:
                orientations[train_count] = int(rng.choice(unused))

    train_widths = _train_widths(rng, train_count)
    widths = train_widths + [_test_width(rng, train_widths) for _ in range(test_count)]
    lengths = [_length(rng, width, full_reflection=(i == 0)) for i, width in enumerate(widths)]
    used_inputs = set()
    used_geometry = set()
    sections = {"train": [], "test": []}

    for index, (width, initial_length, orientation, colors) in enumerate(zip(widths, lengths, orientations, roles)):
        background, path = colors
        # Finite fallback preserves legal counts even if a sampled input repeats.
        minimum = max(width + 1, 2 * (width - 1) + 1) if index == 0 else width + 1
        length_order = [initial_length] + [n for n in range(minimum, 31) if n != initial_length]
        orientation_order = [orientation] + [code for code in range(8) if code != orientation]
        selected = None
        for trial_orientation in orientation_order:
            for trial_length in length_order:
                pair = _make_pair(trial_length, width, trial_orientation, background, path, fill)
                signature = _signature(pair["input"])
                geometry = (trial_length, width, trial_orientation)
                if signature not in used_inputs and geometry not in used_geometry and not _source_copy(pair):
                    selected = pair
                    used_inputs.add(signature)
                    used_geometry.add(geometry)
                    break
            if selected is not None:
                break
        assert selected is not None
        section = "train" if index < train_count else "test"
        sections[section].append(selected)

    validate_generated_puzzle(sections)
    return sections


def _pair_features(pair):
    source, target = pair["input"], pair["output"]
    assert source.shape == target.shape and source.shape[0] != source.shape[1]
    axis = 0 if source.shape[0] > source.shape[1] else 1
    length, width = source.shape[axis], source.shape[1 - axis]
    assert width >= 2 and length > width
    values, counts = np.unique(source, return_counts=True)
    assert len(values) == 2
    singleton = np.flatnonzero(counts == 1)
    assert len(singleton) == 1
    path = int(values[int(singleton[0])])
    background = int(values[1 - int(singleton[0])])
    seed_row, seed_col = (int(n) for n in np.argwhere(source == path)[0])
    assert seed_row in (0, source.shape[0] - 1)
    assert seed_col in (0, source.shape[1] - 1)
    from_low = (seed_row if axis == 0 else seed_col) == 0
    lane_low = (seed_col if axis == 0 else seed_row) == 0
    code = 4 * axis + 2 * int(from_low) + int(lane_low)
    output_values = {int(v) for v in np.unique(target)}
    assert len(output_values) == 2 and path in output_values
    fill = next(color for color in output_values if color != path)
    assert len({background, path, fill}) == 3

    # Reconstruct from the input marker and the observed output fill.
    period = 2 * (width - 1)
    expected = np.full(source.shape, fill, dtype=np.int8)
    for step in range(length):
        phase = step % period
        lane = min(phase, period - phase)
        along = step if from_low else length - 1 - step
        across = lane if lane_low else width - 1 - lane
        cell = (along, across) if axis == 0 else (across, along)
        expected[cell] = path
    assert np.array_equal(target, expected)
    return length, width, code, background, path, fill


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    input_signatures, pair_signatures = set(), set()
    features = {"train": [], "test": []}
    for section, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            assert not np.array_equal(pair["input"], pair["output"])
            assert not _source_copy(pair)
            source_signature = _signature(pair["input"])
            full_signature = (source_signature, _signature(pair["output"]))
            assert source_signature not in input_signatures and full_signature not in pair_signatures
            input_signatures.add(source_signature)
            pair_signatures.add(full_signature)
            features[section].append(_pair_features(pair))

    training, testing = features["train"], features["test"]
    fills = {feature[5] for feature in training}
    assert len(fills) == 1 and all(feature[5] in fills for feature in testing)
    assert len({feature[1] for feature in training}) >= 2
    assert any(width >= 3 and length >= 2 * (width - 1) + 1 for length, width, *_ in training)
    train_geometry = {(f[0], f[1], f[2]) for f in training}
    assert all((f[0], f[1], f[2]) not in train_geometry for f in testing)


def _equal_puzzles(first, second):
    return all(
        len(first[section]) == len(second[section])
        and all(
            np.array_equal(a[name], b[name])
            for a, b in zip(first[section], second[section])
            for name in ("input", "output")
        )
        for section in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert all(_equal_puzzles(puzzle, generate_puzzle(seed)) for seed, puzzle in enumerate(puzzles))
    assert len({_signature(puzzle["test"][0]["input"]) for puzzle in puzzles}) >= 40
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3

    train_widths, test_widths, lengths, phases, orientations = set(), set(), set(), set(), set()
    varied_axes = varied_corners = varied_colors = 0
    wider = nearby = shown = narrower = combinations = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        training = [_pair_features(pair) for pair in puzzle["train"]]
        testing = [_pair_features(pair) for pair in puzzle["test"]]
        train_widths.update(f[1] for f in training)
        test_widths.update(f[1] for f in testing)
        lengths.update(f[0] for f in training + testing)
        orientations.update(f[2] for f in training + testing)
        varied_axes += len({f[2] // 4 for f in training}) > 1
        varied_corners += len({f[2] for f in training}) > 1
        varied_colors += len({(f[3], f[4]) for f in training}) > 1
        largest = max(f[1] for f in training)
        shown_widths = {f[1] for f in training}
        training_orientations = {f[2] for f in training}
        training_colors = {(f[3], f[4]) for f in training}
        for length, width, code, background, path, fill in testing:
            wider += width > largest
            nearby += width == largest + 1
            shown += width in shown_widths
            narrower += width < largest
            combinations += code not in training_orientations and (background, path) not in training_colors
        for length, width, *_ in training + testing:
            period = 2 * (width - 1)
            phase = "partial" if length <= period else "return" if length == period + 1 else "extended" if length <= 2 * period else "cycles"
            phases.add(phase)

    assert len(train_widths) >= 9 and len(test_widths) >= 12
    assert len(lengths) >= 23 and len(orientations) == 8 and len(phases) == 4
    assert varied_axes >= 15 and varied_corners >= 15 and varied_colors >= 15
    assert wider >= 10 and nearby >= 4 and shown >= 8 and narrower >= 8
    assert combinations >= 5

    for train, test in ((2, 1), (10, 1), (10, 4), (6, 3)):
        puzzle = generate_puzzle(13000 + train, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        validate_generated_puzzle(puzzle)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(14000 + (train or 0) + (test or 0), train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(900, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")

    corrupted = generate_puzzle(100)
    pair = corrupted["train"][0]
    fill = _pair_features(pair)[5]
    cell = tuple(int(n) for n in np.argwhere(pair["output"] == fill)[0])
    pair["output"][cell] = int(pair["input"][cell])
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted path accepted")
