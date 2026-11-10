import numpy as np


# The three columns are the only code positions. Their output colors are
# learned once per puzzle; input colors merely locate markers.
_SOURCE_SEQUENCES = ((0, 1, 0), (2, 2, 2), (1, 2, 1), (2, 1, 0), (2, 0, 1))
_SOURCE_OUTPUT_COLORS = (2, 4, 3)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _validated_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _validated_count("num_train", num_train, 2, 10)
    test = _validated_count("num_test", num_test, 1, 4)
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _sample_height(rng, test=False):
    # Extra weight on the original three-row form, with shorter and
    # longer stacks. Tests also reach heights 9 and 10.
    choices = (2, 3, 3, 4, 5, 6, 7, 8, 9, 10) if test else (2, 3, 3, 4, 5, 6, 7, 8)
    return int(choices[int(rng.integers(len(choices)))])


def _sequence_of_kind(rng, height, kind):
    a, b, c = [int(x) for x in rng.permutation(3)]
    if kind == "constant":
        return (a,) * height
    if kind == "alternating":
        return tuple(a if row % 2 == 0 else b for row in range(height))
    if kind == "adjacent":
        sequence = [int(x) for x in rng.integers(0, 3, size=height)]
        place = int(rng.integers(height - 1))
        sequence[place] = a
        sequence[place + 1] = a
        sequence[(place + 2) % height] = b
        return tuple(sequence)
    if kind == "all_distinct" and height == 3:
        return (a, b, c)
    return tuple(int(x) for x in rng.integers(0, 3, size=height))


def _new_sequence(rng, height, used, kinds):
    for _ in range(80):
        kind = kinds[int(rng.integers(len(kinds)))]
        sequence = _sequence_of_kind(rng, height, kind)
        if sequence not in used and (
            "all_distinct" in kinds or height != 3 or len(set(sequence)) < 3
        ):
            return sequence
    # Bounded deterministic fallback for unlikely collisions.
    for number in range(3 ** height):
        digits, remaining = [], number
        for _ in range(height):
            digits.append(remaining % 3)
            remaining //= 3
        sequence = tuple(digits)
        if sequence not in used and (
            "all_distinct" in kinds or height != 3 or len(set(sequence)) < 3
        ):
            return sequence
    # A two-row pool has only nine members. If an unusually large puzzle
    # exhausts it, choose another legal height instead of rejecting a seed.
    for alternative in range(2, 11):
        if alternative == height:
            continue
        for number in range(3 ** alternative):
            digits, remaining = [], number
            for _ in range(alternative):
                digits.append(remaining % 3)
                remaining //= 3
            sequence = tuple(digits)
            if sequence not in used and (
                "all_distinct" in kinds or alternative != 3 or len(set(sequence)) < 3
            ):
                return sequence
    raise AssertionError("no unused marker sequence")


def _choose_sequences(rng, train_count, test_count):
    mode = int(rng.integers(3))
    a, b, c = [int(x) for x in rng.permutation(3)]
    if mode == 0:
        train = [(a, b, c)]
    elif mode == 1:
        train = [(a, b, c), (b, c, a)]
    else:
        # Two repeated layouts jointly teach all three column meanings.
        train = [(a, a, b), (c, c, a)]
    used = set(train)
    non_distinct = ("constant", "alternating", "adjacent", "mixed")
    while len(train) < train_count:
        height = _sample_height(rng)
        kinds = non_distinct if mode != 1 or height != 3 else non_distinct + ("all_distinct",)
        sequence = _new_sequence(rng, height, used, kinds)
        train.append(sequence)
        used.add(sequence)
    train = [train[int(i)] for i in rng.permutation(len(train))]
    train_heights = {len(sequence) for sequence in train}
    test = []
    for index in range(test_count):
        height = _sample_height(rng, test=True)
        if index == 0 and (len(train_heights) == 1 or rng.random() < 0.55):
            available = [h for h in range(2, 11) if h not in train_heights]
            if available:
                height = int(available[int(rng.integers(len(available)))])
        kinds = non_distinct + (("all_distinct",) if height == 3 else ())
        sequence = _new_sequence(rng, height, used, kinds)
        test.append(sequence)
        used.add(sequence)
    return train, test


def _is_exact_source_pair(input_grid, output_grid):
    if input_grid.shape != (3, 3) or output_grid.shape != (3, 3):
        return False
    for sequence in _SOURCE_SEQUENCES:
        source_input = np.zeros((3, 3), dtype=np.int8)
        source_output = np.empty((3, 3), dtype=np.int8)
        for row, column in enumerate(sequence):
            source_input[row, column] = 5
            source_output[row, :] = _SOURCE_OUTPUT_COLORS[column]
        if np.array_equal(input_grid, source_input) and np.array_equal(output_grid, source_output):
            return True
    return False


def _make_pair(rng, sequence, background, output_colors):
    height = len(sequence)
    input_grid = np.full((height, 3), background, dtype=np.int8)
    output_grid = np.empty((height, 3), dtype=np.int8)
    allowed_markers = [hue for hue in range(10) if hue != background]
    if rng.random() < 0.42:
        marker_hues = [allowed_markers[int(rng.integers(9))]] * height
    else:
        marker_hues = [allowed_markers[int(rng.integers(9))] for _ in sequence]
        if len(set(marker_hues)) == 1:
            marker_hues[-1] = allowed_markers[(allowed_markers.index(marker_hues[-1]) + 1) % 9]
    for row, column in enumerate(sequence):
        input_grid[row, column] = marker_hues[row]
        output_grid[row, :] = output_colors[column]
    if _is_exact_source_pair(input_grid, output_grid):
        replacement = next(hue for hue in allowed_markers if hue != int(input_grid[0, sequence[0]]))
        input_grid[0, sequence[0]] = replacement
    return {"input": input_grid, "output": output_grid}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    output_colors = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    train_sequences, test_sequences = _choose_sequences(rng, train_count, test_count)
    backgrounds = [int(x) for x in rng.permutation(10)]
    puzzle = {"train": [], "test": []}
    for index, sequence in enumerate(train_sequences + test_sequences):
        if index and index % 10 == 0:
            backgrounds = [int(x) for x in rng.permutation(10)]
        background = backgrounds[index % 10]
        pair = _make_pair(rng, sequence, background, output_colors)
        puzzle["train" if index < train_count else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _decode_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    height = grid.shape[0]
    _require(int(np.count_nonzero(grid == background)) == 2 * height,
             "background must occupy two cells per row")
    columns = []
    for row in grid:
        locations = np.flatnonzero(row != background)
        _require(len(locations) == 1, "each row needs one identifiable marker")
        columns.append(int(locations[0]))
    return background, tuple(columns)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "illegal pair counts")
    seen_inputs, seen_pairs, sequences = set(), set(), {"train": [], "test": []}
    backgrounds, heights = set(), set()
    learned = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grids must be 2D arrays")
                _require(all(1 <= n <= 30 for n in grid.shape), "illegal grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "grid cells must be integers")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "illegal ARC color")
            _require(inp.shape[1] == 3 and 2 <= inp.shape[0] <= 10 and out.shape == inp.shape,
                     "row stack must have three columns and equal output size")
            _require(not np.array_equal(inp, out), "unchanged pair")
            input_key, output_key = _grid_key(inp), _grid_key(out)
            _require(input_key not in seen_inputs, "duplicate input")
            _require((input_key, output_key) not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            _require(not _is_exact_source_pair(inp, out), "exact source pair copy")
            background, sequence = _decode_input(inp)
            backgrounds.add(background)
            heights.add(inp.shape[0])
            sequences[split].append(sequence)
            for row, column in enumerate(sequence):
                color = int(out[row, 0])
                _require(bool(np.all(out[row] == color)), "output row must be uniform")
                if split == "train":
                    if column in learned:
                        _require(learned[column] == color, "contradictory column code")
                    learned[column] = color
    _require(set(learned) == {0, 1, 2} and len(set(learned.values())) == 3,
             "training must teach three distinct column colors")
    for split in ("train", "test"):
        for pair, sequence in zip(puzzle[split], sequences[split]):
            expected = np.array([[learned[column]] * 3 for column in sequence], dtype=np.int8)
            _require(np.array_equal(pair["output"], expected), "row code output mismatch")
    _require(len(backgrounds) >= 2, "input appearance must vary within a puzzle")
    _require(len(heights) >= 2, "row count must vary within a puzzle")
    _require(len(set(sequences["train"] + sequences["test"])) == len(train) + len(test),
             "marker layouts must be distinct, including held-out layouts")


def _fingerprint(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _predict_from_training(train, test_input):
    # Uses only returned demonstrations and the queried input, never test labels.
    evidence = {}
    for pair in train:
        inp, out = pair["input"], pair["output"]
        values, counts = np.unique(inp, return_counts=True)
        background = int(values[int(np.argmax(counts))])
        for row in range(inp.shape[0]):
            column = int(np.flatnonzero(inp[row] != background)[0])
            color = int(out[row, 0])
            if column in evidence:
                _require(evidence[column] == color, "inconsistent training code")
            evidence[column] = color
    _require(set(evidence) == {0, 1, 2}, "incomplete training code")
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    return np.array([[evidence[int(np.flatnonzero(row != background)[0])]] * 3
                     for row in test_input], dtype=np.int8)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    categories = set()
    train_all_distinct_counts = set()
    train_heights, test_heights, backgrounds, output_maps = set(), set(), set(), set()
    layouts = set()
    varied_train_backgrounds = multi_hue_column_puzzles = novel_height_puzzles = 0
    varied_row_hue_pairs = overlap_pairs = 0
    overlap = varied_row_hues = novel_test_height = source_test_order = False
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        mapping = {}
        train_sequences = []
        train_backgrounds = set()
        column_marker_hues = {0: set(), 1: set(), 2: set()}
        for pair in puzzle["train"]:
            train_background, seq = _decode_input(pair["input"])
            train_backgrounds.add(train_background)
            train_sequences.append(seq)
            train_heights.add(len(seq))
            for row, column in enumerate(seq):
                mapping[column] = int(pair["output"][row, 0])
                column_marker_hues[column].add(int(pair["input"][row, column]))
        varied_train_backgrounds += len(train_backgrounds) > 1
        multi_hue_column_puzzles += any(len(hues) > 1 for hues in column_marker_hues.values())
        output_maps.add(tuple(mapping[col] for col in range(3)))
        train_all_distinct_counts.add(sum(len(s) == 3 and len(set(s)) == 3 for s in train_sequences))
        for pair in puzzle["train"] + puzzle["test"]:
            inp = pair["input"]
            bg, seq = _decode_input(inp)
            layouts.add(seq)
            backgrounds.add(bg)
            marker_hues = [int(inp[row, col]) for row, col in enumerate(seq)]
            varied_row_hue_pairs += len(set(marker_hues)) > 1
            overlap_pairs += bool(set(marker_hues + [bg]) & set(mapping.values()))
            varied_row_hues |= len(set(marker_hues)) > 1
            overlap |= bool(set(marker_hues + [bg]) & set(mapping.values()))
        puzzle_novel_height = False
        for pair in puzzle["test"]:
            inp = pair["input"]
            _, seq = _decode_input(inp)
            test_heights.add(len(seq))
            puzzle_novel_height |= len(seq) not in {len(s) for s in train_sequences}
            source_test_order |= seq == (2, 0, 1)
            if len(set(seq)) == 1:
                categories.add("constant")
            if len(seq) >= 3 and seq[0] == seq[2] and seq[0] != seq[1]:
                categories.add("alternating")
            if len(set(seq)) > 1 and any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
                categories.add("adjacent")
            if len(seq) == 3 and len(set(seq)) == 3:
                categories.add("all_distinct")
            prediction = _predict_from_training(puzzle["train"], inp.copy())
            _require(np.array_equal(prediction, pair["output"]), "test cannot be inferred from training")
        novel_height_puzzles += puzzle_novel_height
        novel_test_height |= puzzle_novel_height
    _require({0, 1, 2}.issubset(train_all_distinct_counts), "training syllabus did not vary")
    _require(categories == {"constant", "alternating", "adjacent", "all_distinct"}, "held-out patterns lack diversity")
    _require(len(train_heights) >= 6 and len(test_heights) >= 7, "row heights lack diversity")
    _require(len(backgrounds) >= 8 and len(output_maps) >= 30, "color roles lack diversity")
    _require(len(layouts) >= 100, "marker layouts lack structural diversity")
    _require(varied_train_backgrounds == 42 and multi_hue_column_puzzles >= 30,
             "training input appearance does not vary enough")
    _require(novel_height_puzzles >= 25 and varied_row_hue_pairs >= 100 and overlap_pairs >= 100,
             "height and input hue combinations lack diversity")
    _require(overlap and varied_row_hues and novel_test_height and source_test_order,
             "input appearance or test regime lacks requested variation")
    _require(len({_fingerprint(puzzle) for puzzle in puzzles}) >= 40, "cross-seed puzzles repeat")
    _require(len({len(puzzle["train"]) for puzzle in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(puzzle["test"]) for puzzle in puzzles}) >= 3, "test counts lack diversity")
    for seed, train, test in ((901, 2, 1), (902, 10, 4), (903, 10, 1),
                              (904, None, 1), (905, None, 4), (906, 2, None),
                              (907, 10, None), (908, 8, 2)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            _require(len(puzzle["train"]) == train, "train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "test count ignored")
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed, train, test)),
                 "generation is not deterministic")
        for pair in puzzle["test"]:
            _require(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"]),
                     "requested-count test cannot be inferred")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(123, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request was accepted")
    _require(_fingerprint(generate_puzzle(23)) == _fingerprint(generate_puzzle(23)),
             "automatic count generation is not deterministic")
