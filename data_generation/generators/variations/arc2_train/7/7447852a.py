"""Periodic zigzag strips with selected opposite-side triangular bays."""

import numpy as np

_HEIGHTS = (3, 5)
_INTERVALS = (2, 3, 4)
_SOURCE_WIDTHS = (10, 15, 18, 25)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _seed_and_counts(seed, num_train, num_test):
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for value, name, lo, hi in ((num_train, "num_train", 2, 10),
                                (num_test, "num_test", 1, 4)):
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, np.integer))
                                  or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    seed = int(seed)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 3
        num_train = num_test + 1 + (seed // 3) % (10 - num_test)
    elif num_train is None:
        num_test = int(num_test)
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_train = int(num_train)
        num_test = 1 + seed % min(4, num_train - 1)
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return seed, num_train, num_test


def _canonical_input(height, length, side, phase, background, path):
    """The first visible boundary turn is at long-axis position phase."""
    grid = np.full((height, length), background, dtype=np.int8)
    period = 2 * (height - 1)
    for column in range(length):
        distance = abs((column - phase) % period - (height - 1))
        row = height - 1 - distance
        if side == 1:
            row = height - 1 - row
        grid[row, column] = path
    return grid


def _bay_cells(height, length, side, phase, turn_index):
    reach = height - 1
    center = phase + turn_index * reach
    turn_at_top = (side == 0) == (turn_index % 2 == 0)
    for column in range(max(0, center - reach + 1), min(length, center + reach)):
        distance = abs(column - center)
        if turn_at_top:
            for row in range(distance + 1, height):
                yield row, column
        else:
            for row in range(height - distance - 1):
                yield row, column


def _bay_indices(height, length, phase):
    # A turn just beyond the right edge can still have an in-bounds bay.
    reach = height - 1
    return range((length + reach - 2 - phase) // reach + 1)


def _paint(canonical, side, phase, interval, residue, highlight):
    result = canonical.copy()
    height, length = result.shape
    for index in _bay_indices(height, length, phase):
        if index % interval == residue:
            for row, column in _bay_cells(height, length, side, phase, index):
                result[row, column] = highlight
    return result


def _pair(height, length, vertical, side, phase, background, path,
          highlight, interval, residue):
    canonical = _canonical_input(height, length, side, phase, background, path)
    answer = _paint(canonical, side, phase, interval, residue, highlight)
    if vertical:
        canonical, answer = canonical.T.copy(), answer.T.copy()
    return {"input": canonical, "output": answer}


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _is_source_copy(info, pair):
    if (info["height"] != 3 or info["vertical"] or info["side"] != 0
            or info["phase"] != 0 or info["length"] not in _SOURCE_WIDTHS
            or (info["background"], info["path"], info["highlight"]) != (0, 2, 4)):
        return False
    source_input = _canonical_input(3, info["length"], 0, 0, 0, 2)
    source_output = _paint(source_input, 0, 0, 3, 0, 4)
    return np.array_equal(pair["input"], source_input) and np.array_equal(
        pair["output"], source_output)


def _choose_length(rng, height, phase, interval, residue, long_exemplar,
                   target, reference_length):
    first = phase + residue * (height - 1)
    second = phase + (residue + interval) * (height - 1)
    minimum = max(height + 4, first + 1, 10)
    if long_exemplar:
        minimum = max(minimum, second + 1)
    if long_exemplar:
        maximum = min(30, minimum + 5)
    elif not target:
        maximum = max(minimum, 28)
    elif rng.random() < 0.55:
        maximum = max(minimum, min(30, reference_length))
    else:
        maximum = 30
    return int(rng.integers(minimum, maximum + 1))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed, num_train, num_test = _seed_and_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    height = int(rng.choice(_HEIGHTS, p=(0.7, 0.3)))
    interval = int(rng.choice(_INTERVALS, p=(0.2, 0.6, 0.2)))
    residue = int(rng.integers(interval))
    highlight = int(rng.integers(10))

    first = tuple(int(rng.integers(2)) for _ in range(3))
    train_settings = [first, tuple(1 - value for value in first)]
    train_settings.extend(tuple(int(rng.integers(2)) for _ in range(3))
                          for _ in range(num_train - 2))
    test_settings = [tuple(int(rng.integers(2)) for _ in range(3))
                     for _ in range(num_test)]
    train_lengths = [_choose_length(rng, height, setting[2], interval, residue,
                                    index == 0, False, 30)
                     for index, setting in enumerate(train_settings)]
    test_lengths = [_choose_length(rng, height, setting[2], interval, residue,
                                   False, True, max(train_lengths))
                    for setting in test_settings]
    palettes = [(bg, path) for bg in range(10) for path in range(10)
                if bg != path and bg != highlight and path != highlight]
    rng.shuffle(palettes)

    puzzle = {"train": [], "test": []}
    palette_cursor = 0
    for section, settings_list, lengths in (("train", train_settings, train_lengths),
                                             ("test", test_settings, test_lengths)):
        for (vertical, side, phase), length in zip(settings_list, lengths):
            while True:
                background, path = palettes[palette_cursor]
                palette_cursor += 1
                pair = _pair(height, length, vertical, side, phase, background,
                             path, highlight, interval, residue)
                info = dict(height=height, length=length, vertical=bool(vertical),
                            side=side, phase=phase, background=background,
                            path=path, highlight=highlight)
                if not _is_source_copy(info, pair):
                    break
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} is not an ndarray")
    _require(grid.ndim == 2, f"{label} is not two-dimensional")
    _require(all(1 <= size <= 30 for size in grid.shape),
             f"{label} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer),
             f"{label} is not integer-valued")
    _require(0 <= int(grid.min()) and int(grid.max()) <= 9,
             f"{label} has an illegal color")


def _inspect_input(grid):
    shape = grid.shape
    _require(min(shape) in _HEIGHTS and max(shape) > min(shape),
             "input is not a supported strip")
    vertical = shape[0] > shape[1]
    canonical = grid.T if vertical else grid
    height, length = canonical.shape
    colors, counts = np.unique(canonical, return_counts=True)
    _require(len(colors) == 2, "input must have background and path")
    background = int(colors[int(np.argmax(counts))])
    path = int(colors[int(np.argmin(counts))])
    _require(int(np.max(counts)) == (height - 1) * length,
             "input does not have one path cell per position")
    matches = [(side, phase) for side in (0, 1) for phase in (0, 1)
               if np.array_equal(canonical, _canonical_input(
                   height, length, side, phase, background, path))]
    _require(len(matches) == 1, "input has no unique visible turn anchor")
    side, phase = matches[0]
    return dict(height=height, length=length, vertical=vertical, side=side,
                phase=phase, background=background, path=path)


def _inspect_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    _require(input_grid.shape == output_grid.shape,
             "input and output dimensions differ")
    info = _inspect_input(input_grid)
    changed = input_grid != output_grid
    _require(np.any(changed), "no bay was highlighted")
    new_colors = set(np.unique(output_grid).tolist()) - set(
        np.unique(input_grid).tolist())
    _require(len(new_colors) == 1, "output needs exactly one new color")
    highlight = int(next(iter(new_colors)))
    _require(np.all(output_grid[changed] == highlight),
             "changed cells do not share the highlight color")
    canonical_changed = changed.T if info["vertical"] else changed
    observed = []
    for index in _bay_indices(info["height"], info["length"], info["phase"]):
        cells = list(_bay_cells(info["height"], info["length"],
                                info["side"], info["phase"], index))
        flags = [bool(canonical_changed[row, col]) for row, col in cells]
        _require(all(flags) or not any(flags), "partially painted bay")
        if all(flags):
            observed.append(index)
    expected_changed = np.zeros_like(canonical_changed)
    for index in observed:
        for row, col in _bay_cells(info["height"], info["length"],
                                   info["side"], info["phase"], index):
            expected_changed[row, col] = True
    _require(np.array_equal(canonical_changed, expected_changed),
             "output changed cells outside complete selected bays")
    info["highlight"] = highlight
    info["observed"] = tuple(observed)
    return info


def _matching_schedules(train_infos):
    matches = []
    for interval in _INTERVALS:
        for residue in range(interval):
            if all(info["observed"] == tuple(
                index for index in _bay_indices(info["height"],
                                                 info["length"], info["phase"])
                if index % interval == residue
            ) for info in train_infos):
                matches.append((interval, residue))
    return matches


def _predict_from_train(train_pairs, test_input):
    """Infer from training grids and a test input, without its output or seed."""
    infos = [_inspect_pair(pair) for pair in train_pairs]
    schedule = _matching_schedules(infos)
    _require(len(schedule) == 1, "training does not identify the schedule")
    highlights = {info["highlight"] for info in infos}
    _require(len(highlights) == 1, "training has no shared output color")
    test_info = _inspect_input(test_input)
    canonical = test_input.T if test_info["vertical"] else test_input
    answer = _paint(canonical, test_info["side"], test_info["phase"],
                    *schedule[0], next(iter(highlights)))
    return answer.T.copy() if test_info["vertical"] else answer


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and
             isinstance(puzzle["test"], list), "sections must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and
             train_count > test_count, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            label = f"{section}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} has illegal keys")
            _check_grid(pair["input"], label + ".input")
            _check_grid(pair["output"], label + ".output")
            input_key, output_key = _grid_key(pair["input"]), _grid_key(pair["output"])
            _require(input_key not in seen_inputs, "duplicate input")
            _require((input_key, output_key) not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            info = _inspect_pair(pair)
            _require(not _is_source_copy(info, pair), "source pair copied")
            infos[section].append(info)
    train_infos, test_infos = infos["train"], infos["test"]
    _require(len({info["height"] for info in train_infos + test_infos}) == 1,
             "strip thickness must be shared within a puzzle")
    _require(len({info["highlight"] for info in train_infos + test_infos}) == 1,
             "highlight color must be shared within a puzzle")
    schedule = _matching_schedules(train_infos)
    _require(len(schedule) == 1, "training does not identify cadence and residue")
    interval, residue = schedule[0]
    _require(all(info["observed"] == tuple(
        index for index in _bay_indices(info["height"], info["length"],
                                         info["phase"])
        if index % interval == residue
    ) for info in test_infos), "test uses a different bay schedule")
    for key in ("vertical", "side", "phase"):
        _require({info[key] for info in train_infos} == {False, True},
                 f"training does not demonstrate both {key} variants")
    _require(any(len(info["observed"]) >= 2 for info in train_infos),
             "training does not show two selected bays")
    for pair in puzzle["test"]:
        prediction = _predict_from_train(puzzle["train"], pair["input"].copy())
        _require(np.array_equal(prediction, pair["output"]),
                 "test output not recoverable from training")


def _signature(puzzle):
    return tuple((section, name, _grid_key(pair[name]))
                 for section in ("train", "test")
                 for pair in puzzle[section] for name in ("input", "output"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_signature(p) for p in puzzles}) == 42,
             "automatic puzzles are not distinct")
    _require(len({len(p["train"]) for p in puzzles}) >= 4 and
             len({len(p["test"]) for p in puzzles}) >= 3,
             "automatic pair counts lack diversity")
    dimensions, schedules, combos, palettes = set(), set(), set(), set()
    shorter_tests = longer_tests = left_clips = right_clips = 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        _require(_signature(puzzle) == _signature(generate_puzzle(seed)),
                 "puzzle is not deterministic")
        train_infos = [_inspect_pair(pair) for pair in puzzle["train"]]
        test_infos = [_inspect_pair(pair) for pair in puzzle["test"]]
        schedules.add(_matching_schedules(train_infos)[0])
        maximum_train = max(info["length"] for info in train_infos)
        shorter_tests += any(info["length"] <= maximum_train for info in test_infos)
        longer_tests += any(info["length"] > maximum_train for info in test_infos)
        for info in train_infos + test_infos:
            dimensions.add((info["height"], info["length"], info["vertical"]))
            combos.add((info["vertical"], info["side"], info["phase"]))
            palettes.add((info["background"], info["path"], info["highlight"]))
            if 0 in info["observed"] and info["phase"] == 0:
                left_clips += 1
            if info["observed"] and info["length"] - 1 - (
                info["phase"] + info["observed"][-1] * (info["height"] - 1)
            ) < info["height"] - 1:
                right_clips += 1
    _require({h for h, _, _ in dimensions} == set(_HEIGHTS),
             "larger bay geometry absent")
    _require(len(dimensions) >= 45 and len(schedules) >= 7 and
             len(combos) == 8 and len(palettes) >= 70,
             "structural or palette diversity is too narrow")
    _require(shorter_tests >= 8 and longer_tests >= 8,
             "test length regimes are too narrow")
    _require(left_clips >= 10 and right_clips >= 10,
             "boundary clipping is too rare")
    _require(any(21 <= length <= 25 for _, length, _ in dimensions),
             "intermediate lengths are missing")
    for seed, train_count, test_count in (
        (103, 2, 1), (104, 10, 1), (105, 10, 4),
        (106, None, 1), (107, None, 4),
        (108, 2, None), (109, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "test count ignored")
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(211, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count was accepted")
