import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        ("75727", "77777", "77777"),
        ("75727", "77277", "77777"),
    ),
    (
        ("72727", "77777", "77777"),
        ("72727", "77577", "77777"),
    ),
    (
        ("25252", "77777", "77777"),
        ("25252", "75257", "77277"),
    ),
    (
        ("72757", "77777", "77777"),
        ("72757", "77577", "77777"),
    ),
    (
        (
            "552255522",
            "777777777",
            "777777777",
            "777777777",
            "777777777",
        ),
        (
            "552255522",
            "722552227",
            "775522577",
            "777225777",
            "777757777",
        ),
    ),
    (
        ("75757", "77777", "77777"),
        ("75757", "77277", "77777"),
    ),
    (
        ("5255552", "7777777", "7777777", "7777777"),
        ("5255552", "7252227", "7752577", "7772777"),
    ),
    (
        (
            "555555555557",
            "777777777777",
            "777777777777",
            "777777777777",
            "777777777777",
            "777777777777",
        ),
        (
            "555555555557",
            "722222222277",
            "775555555777",
            "777222227777",
            "777755577777",
            "777772777777",
        ),
    ),
)


TRAIN_LENGTHS = tuple(range(3, 20, 2))
TEST_LENGTHS = tuple(range(3, 24, 2))
STYLES = ("solid", "alternating", "runs", "irregular", "sparse", "gapped_alternating")


def _rows_to_array(rows):
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _validate_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed_key, num_train, num_test):
    train = _validate_count("num_train", num_train, 2, 10)
    test = _validate_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed_key % 4
        train = test + 1 + (seed_key // 4) % (10 - test)
    elif train is None:
        train = test + 1 + (seed_key // 17) % (10 - test)
    else:
        test = 1 + (seed_key // 97) % min(4, train - 1)
    return int(train), int(test)


def sample_puzzle_parameters(rng):
    palette = [int(v) for v in rng.permutation(10)]
    signals = (palette[0], palette[1])
    varying_background = bool(rng.integers(0, 3) == 0)
    backgrounds = tuple(palette[2:2 + int(rng.integers(2, 6))]) if varying_background else (palette[2],)
    return {"signals": signals, "backgrounds": backgrounds}


def _swap_signal_colors(values, signal_colors):
    a, b = signal_colors
    result = values.copy()
    result[values == a] = b
    result[values == b] = a
    return result


def _construct_output(input_grid, background, signal_colors):
    output = input_grid.copy()
    positions = np.flatnonzero(input_grid[0] != background)
    start, stop = int(positions[0]), int(positions[-1]) + 1
    sequence = input_grid[0, start:stop].copy()
    for row in range(1, (len(sequence) + 1) // 2):
        sequence = _swap_signal_colors(sequence[:-2], signal_colors)
        output[row, start + row:start + row + len(sequence)] = sequence
    return output


def _diagnostic_sequence(rng, length, background, signals):
    # Retained gaps and unequal left/right prefixes show both disputed operations.
    for _ in range(80):
        seq = rng.choice(np.array(signals, dtype=np.int8), size=length).astype(np.int8)
        max_gaps = min(4, length - 4)
        gap_count = int(rng.integers(1, max_gaps + 1))
        if gap_count > 1 and rng.integers(0, 2):
            first = int(rng.integers(1, length - 2 - gap_count + 1))
            gaps = np.arange(first, first + gap_count)
        else:
            gaps = rng.choice(np.arange(1, length - 2), size=gap_count, replace=False)
        seq[gaps] = background
        if set(signals) <= set(int(x) for x in seq) and not np.array_equal(seq[:-2], seq[1:-1]):
            return seq
    raise RuntimeError("diagnostic construction failed")


def _sequence(rng, length, style, background, signals):
    a, b = signals
    if style == "solid":
        return np.full(length, signals[int(rng.integers(0, 2))], dtype=np.int8)
    if style in ("alternating", "gapped_alternating"):
        phase = int(rng.integers(0, 2))
        seq = np.array([signals[(i + phase) % 2] for i in range(length)], dtype=np.int8)
    elif style == "runs":
        seq = np.empty(length, dtype=np.int8)
        phase = int(rng.integers(0, 2))
        i = 0
        while i < length:
            run = int(rng.integers(1, min(5, length - i) + 1))
            seq[i:i + run] = signals[phase]
            i += run
            phase = 1 - phase
    else:
        probability = 0.46 if style == "sparse" else 0.18
        seq = rng.choice(np.array([background, a, b], dtype=np.int8), size=length,
                         p=[probability, (1 - probability) / 2, (1 - probability) / 2]).astype(np.int8)
    if style == "gapped_alternating":
        count = int(rng.integers(1, min(4, length - 2) + 1))
        gaps = rng.choice(np.arange(1, length - 1), size=count, replace=False)
        seq[gaps] = background
    seq[0] = signals[int(rng.integers(0, 2))] if style in ("irregular", "sparse") else seq[0]
    seq[-1] = signals[int(rng.integers(0, 2))] if style in ("irregular", "sparse") else seq[-1]
    return seq


def _make_pair(rng, sequence, background, signals):
    length = len(sequence)
    left = int(rng.integers(0, min(8, 30 - length) + 1))
    remaining = 30 - length - left
    right = int(rng.integers(0, min(9, remaining) + 1))
    if length + left + right < 5:
        right += 5 - (length + left + right)
    height = max(3, (length + 1) // 2) + int(rng.integers(0, 6))
    inp = np.full((height, length + left + right), background, dtype=np.int8)
    inp[0, left:left + length] = sequence
    return {"input": inp, "output": _construct_output(inp, background, signals)}


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _is_source_pair(pair):
    key = _pair_key(pair)
    return any(key == (_grid_key(_rows_to_array(i)), _grid_key(_rows_to_array(o)))
               for i, o in _SOURCE_PAIR_ROWS)


def _pair_background(pair):
    inp = pair["input"]
    background = int(inp[1, 0])
    assert np.all(inp[1:] == background)
    return background


def _motif_key(pair, signals):
    background = _pair_background(pair)
    row = pair["input"][0]
    indices = np.flatnonzero(row != background)
    roles = {background: 0, signals[0]: 1, signals[1]: 2}
    return tuple(roles[int(v)] for v in row[int(indices[0]):int(indices[-1]) + 1])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed_key = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)
    params = sample_puzzle_parameters(rng)
    signals, backgrounds = params["signals"], params["backgrounds"]
    train = []
    test = []
    used_inputs = set()
    used_motifs = set()

    diagnostic_length = int(rng.choice(np.arange(7, 18, 2)))
    solid_length = int(rng.choice(np.arange(3, 14, 2)))
    if solid_length == diagnostic_length:
        solid_length = 3 if diagnostic_length != 3 else 5
    jobs = [("diagnostic", diagnostic_length), ("solid", solid_length)]
    for _ in range(train_count - 2):
        jobs.append((STYLES[int(rng.integers(0, len(STYLES)))], int(rng.choice(TRAIN_LENGTHS))))
    rng.shuffle(jobs)
    background_choices = list(rng.choice(backgrounds, size=train_count, replace=True))
    if len(backgrounds) > 1:
        background_choices[0] = backgrounds[0]
        background_choices[1] = backgrounds[1]
        rng.shuffle(background_choices)

    def add_pair(destination, style, length, background):
        if style == "diagnostic":
            alternatives = tuple(range(7, 18, 2))
        elif destination is train:
            alternatives = tuple(n for n in (range(3, 14, 2) if style == "solid" else TRAIN_LENGTHS)
                                 if style != "solid" or n != diagnostic_length)
        else:
            alternatives = TEST_LENGTHS[1:] if style in ("sparse", "gapped_alternating") else TEST_LENGTHS
        for attempt in range(500):
            trial_length = length if attempt < 20 else int(rng.choice(alternatives))
            if style == "diagnostic":
                seq = _diagnostic_sequence(rng, trial_length, background, signals)
            else:
                seq = _sequence(rng, trial_length, style, background, signals)
            pair = _make_pair(rng, seq, background, signals)
            motif = _motif_key(pair, signals)
            key = _grid_key(pair["input"])
            if key not in used_inputs and motif not in used_motifs and not _is_source_pair(pair):
                destination.append(pair)
                used_inputs.add(key)
                used_motifs.add(motif)
                return
        raise RuntimeError("unique pair construction failed")

    for (style, length), background in zip(jobs, background_choices):
        add_pair(train, style, length, int(background))

    test_styles = list(STYLES)
    rng.shuffle(test_styles)
    training_backgrounds = tuple(sorted({_pair_background(pair) for pair in train}))
    for index in range(test_count):
        style = test_styles[index]
        allowed_lengths = TEST_LENGTHS if style not in ("sparse", "gapped_alternating") else TEST_LENGTHS[1:]
        # Larger spans are common, while overlap with training remains frequent.
        weights = np.array([1 if n < 11 else 2 for n in allowed_lengths], dtype=float)
        weights /= weights.sum()
        length = int(rng.choice(allowed_lengths, p=weights))
        background = int(rng.choice(training_backgrounds))
        add_pair(test, style, length, background)

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_signals(train):
    inferred = None
    backgrounds = []
    for pair in train:
        background = _pair_background(pair)
        backgrounds.append(background)
        colors = set(int(v) for v in np.unique(pair["output"])) - {background}
        assert len(colors) == 2
        if inferred is None:
            inferred = colors
        assert colors == inferred
    assert inferred is not None
    assert not (set(backgrounds) & inferred)
    return tuple(sorted(inferred)), set(backgrounds)


def _inspect_pair(pair, signals):
    inp, out = pair["input"], pair["output"]
    background = _pair_background(pair)
    assert background not in signals
    assert inp.shape == out.shape
    assert set(int(v) for v in np.unique(inp)) <= {background, *signals}
    assert set(int(v) for v in np.unique(out)) == {background, *signals}
    positions = np.flatnonzero(inp[0] != background)
    assert len(positions) >= 2
    start, end = int(positions[0]), int(positions[-1])
    length = end - start + 1
    assert length >= 3 and length % 2 == 1
    assert int(inp[0, start]) in signals and int(inp[0, end]) in signals
    assert inp.shape[0] >= (length + 1) // 2
    # Direct calculation from the original seed independently checks recursive construction.
    expected = inp.copy()
    source = inp[0, start:end + 1]
    for level in range(1, (length + 1) // 2):
        prefix = source[:length - 2 * level]
        values = _swap_signal_colors(prefix, signals) if level % 2 else prefix
        expected[level, start + level:start + level + len(values)] = values
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    return {"background": background, "sequence": source, "length": length,
            "start": start, "right": inp.shape[1] - end - 1,
            "height": inp.shape[0], "width": inp.shape[1]}


def _training_coverage(train, signals, props):
    solid_demo = False
    prefix_gap_demo = False
    for pair, prop in zip(train, props):
        seq, bg = prop["sequence"], prop["background"]
        present = set(int(x) for x in seq) - {bg}
        if len(present) == 1 and set(signals) <= set(int(x) for x in np.unique(pair["output"])):
            solid_demo = True
        if (prop["length"] >= 7 and bg in seq[1:-2]
                and present == set(signals) and not np.array_equal(seq[:-2], seq[1:-1])):
            first_lower = pair["output"][1, prop["start"] + 1:prop["start"] + prop["length"] - 1]
            assert bg in first_lower
            prefix_gap_demo = True
    assert solid_demo and prefix_gap_demo
    assert len({p["length"] for p in props}) >= 2


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert not _is_source_pair(pair)
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    signals, training_backgrounds = _infer_signals(puzzle["train"])
    train_props = [_inspect_pair(pair, signals) for pair in puzzle["train"]]
    test_props = [_inspect_pair(pair, signals) for pair in puzzle["test"]]
    assert all(p["length"] in TRAIN_LENGTHS for p in train_props)
    assert all(p["length"] in TEST_LENGTHS for p in test_props)
    assert all(p["background"] in training_backgrounds for p in test_props)
    _training_coverage(puzzle["train"], signals, train_props)
    motifs = [_motif_key(pair, signals) for section in ("train", "test") for pair in puzzle[section]]
    assert len(set(motifs)) == len(motifs)


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(pair) for pair in puzzle[section]) for section in ("train", "test"))


def _predict_from_examples(train, test_input):
    signals, backgrounds = _infer_signals(train)
    background = int(test_input[1, 0])
    assert background in backgrounds
    assert background not in signals
    answer = test_input.copy()
    locations = np.flatnonzero(test_input[0] != background)
    start, end = int(locations[0]), int(locations[-1])
    original = test_input[0, start:end + 1]
    for depth in range(1, (len(original) + 1) // 2):
        prefix = original[:len(original) - 2 * depth]
        if depth % 2:
            values = np.where(prefix == signals[0], signals[1],
                              np.where(prefix == signals[1], signals[0], background))
        else:
            values = prefix
        answer[depth, start + depth:start + depth + len(values)] = values
    return answer


def self_test() -> None:
    for seed, train_count, test_count in ((123, None, None), (234, 7, 3),
                                          (345, 2, 1), (456, 10, 1), (567, 10, 4),
                                          (678, None, 1), (789, None, 4),
                                          (890, 2, None), (901, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, train_count, test_count))
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for bad in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(123, *bad)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    train_counts, test_counts, palettes, shapes, structures, puzzles = set(), set(), set(), set(), set(), set()
    train_lengths, test_lengths, lefts, rights = set(), set(), set(), set()
    diagnostic_positions, solid_positions = set(), set()
    long_train = short_test = solid_test = mixed_test = gap_test = adjacent_gap = retained_gap = 0
    varying_background = long_test = multi_gap_test = adjacent_gap_test = 0
    deep_gap_test = early_gap_test = combined_test = flush_left = flush_right = wide_right = 0
    multi_test_variety = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signals, bgs = _infer_signals(puzzle["train"])
        palettes.add((signals, tuple(sorted(bgs))))
        varying_background += len(bgs) > 1
        max_train = 0
        test_kinds = set()
        for section in ("train", "test"):
            for index, pair in enumerate(puzzle[section]):
                prop = _inspect_pair(pair, signals)
                seq = prop["sequence"]
                bg = prop["background"]
                shapes.add(pair["input"].shape)
                structures.add(_motif_key(pair, signals))
                assert np.array_equal(_predict_from_examples(puzzle["train"], pair["input"]), pair["output"])
                if section == "train":
                    train_lengths.add(prop["length"])
                    max_train = max(max_train, prop["length"])
                    long_train += prop["length"] >= 11
                    retained_gap += bg in seq[1:-2]
                    adjacent_gap += any(seq[i] == bg and seq[i + 1] == bg for i in range(1, len(seq) - 2))
                    if (prop["length"] >= 7 and bg in seq[1:-2]
                            and set(int(v) for v in seq) - {bg} == set(signals)
                            and not np.array_equal(seq[:-2], seq[1:-1])):
                        diagnostic_positions.add(index)
                    if bg not in seq and len(set(int(v) for v in seq)) == 1:
                        solid_positions.add(index)
                else:
                    test_lengths.add(prop["length"])
                    lefts.add(prop["start"])
                    rights.add(prop["right"])
                    present = set(int(v) for v in seq) - {bg}
                    solid_test += len(present) == 1 and bg not in seq
                    mixed_test += len(present) == 2 and bg not in seq
                    gap_test += bg in seq
                    gaps = [i for i, value in enumerate(seq) if value == bg]
                    long_test += prop["length"] >= 11
                    multi_gap_test += len(gaps) >= 2
                    adjacent_gap_test += any(b == a + 1 for a, b in zip(gaps, gaps[1:]))
                    deep_gap_test += any(i < prop["length"] - 4 for i in gaps)
                    early_gap_test += prop["length"] - 2 in gaps
                    combined_test += prop["length"] >= 11 and bool(gaps) and prop["start"] > 0
                    flush_left += prop["start"] == 0
                    flush_right += prop["right"] == 0
                    wide_right += prop["right"] >= 6
                    if gaps:
                        test_kinds.add("gapped")
                    elif len(present) == 1:
                        test_kinds.add("solid")
                    elif all(seq[i] != seq[i + 1] for i in range(len(seq) - 1)):
                        test_kinds.add("alternating")
                    else:
                        test_kinds.add("mixed")
        short_test += any(_inspect_pair(pair, signals)["length"] <= max_train for pair in puzzle["test"])
        multi_test_variety += len(puzzle["test"]) > 1 and len(test_kinds) > 1
    assert len(puzzles) >= 40 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 25 and len(shapes) >= 35 and len(structures) >= 150
    assert len(train_lengths) >= 7 and len(test_lengths) >= 9
    assert long_train >= 20 and short_test >= 10
    assert long_test >= 20 and multi_test_variety >= 15
    assert solid_test >= 5 and mixed_test >= 5 and gap_test >= 5
    assert retained_gap >= 42 and adjacent_gap >= 3
    assert multi_gap_test >= 5 and adjacent_gap_test >= 3
    assert deep_gap_test >= 5 and early_gap_test >= 3 and combined_test >= 5
    assert flush_left >= 3 and flush_right >= 3 and wide_right >= 5
    assert 0 in lefts and 0 in rights and max(lefts) >= 4 and max(rights) >= 6
    assert len(diagnostic_positions) >= 4 and len(solid_positions) >= 4
    assert varying_background >= 5
