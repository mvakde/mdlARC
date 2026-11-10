import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "000050000",
            "000050000",
            "500050000",
            "500050500",
            "500050500",
            "500050500",
            "500050500",
            "505050500",
            "505050500",
        ),
        (
            "000010000",
            "000010000",
            "000010000",
            "000010000",
            "000010000",
            "000010000",
            "000010000",
            "002010000",
            "002010000",
        ),
    ),
    (
        (
            "000000000",
            "005000000",
            "005000500",
            "505000500",
            "505000500",
            "505050500",
            "505050505",
            "505050505",
            "505050505",
        ),
        (
            "000000000",
            "001000000",
            "001000000",
            "001000000",
            "001000000",
            "001000000",
            "001000002",
            "001000002",
            "001000002",
        ),
    ),
    (
        (
            "000000000",
            "000000050",
            "000500050",
            "000500050",
            "000505050",
            "000505050",
            "000505050",
            "000505050",
            "050505050",
        ),
        (
            "000000000",
            "000000010",
            "000000010",
            "000000010",
            "000000010",
            "000000010",
            "000000010",
            "000000010",
            "020000010",
        ),
    ),
)


def _rows_signature(rows):
    return (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_PAIR_SIGNATURES = frozenset(
    (_rows_signature(input_rows), _rows_signature(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _grid_signature(grid):
    return (
        tuple(int(dimension) for dimension in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _puzzle_signature(puzzle):
    return tuple(
        tuple(
            (
                _grid_signature(pair["input"]),
                _grid_signature(pair["output"]),
            )
            for pair in puzzle[split]
        )
        for split in ("train", "test")
    )


def _checked_optional_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_optional_count(
        "num_train", num_train, 2, 10
    )
    num_test = _checked_optional_count(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum_train = max(2, num_test + 1)
        num_train = (
            minimum_train
            + (seed // 4) % (11 - minimum_train)
        )
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = (
            minimum_train
            + seed % (11 - minimum_train)
        )
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = [int(c) for c in rng.choice(10, size=2, replace=False)]
    return {"longest_output": colors[0], "shortest_output": colors[1]}


def _sample_lanes(rng, cross_size, count):
    # Insert one compulsory blank lane between consecutive bars. The other
    # blanks are distributed by the random subset, including at the margins.
    compressed = np.sort(rng.choice(cross_size - count + 1, size=count,
                                    replace=False))
    return tuple(int(x) for x in compressed + np.arange(count))


def _sample_lengths(rng, axis_size, count, regime):
    one_cell_probability = 0.23 if regime == "source_train" else 0.30
    if rng.random() < one_cell_probability:
        minimum = 1
    else:
        minimum = int(rng.integers(2, min(6, axis_size - 2) + 1))
    available_span = axis_size - minimum
    relationship = int(rng.integers(0, 3))
    if relationship == 0:
        span = int(rng.integers(2, min(4, available_span) + 1))
    elif relationship == 1 and available_span >= 8:
        span = int(rng.integers(8, available_span + 1))
    else:
        span = int(rng.integers(2, available_span + 1))
    maximum = minimum + span
    middle = rng.integers(minimum + 1, maximum, size=count - 2).astype(int)
    if count >= 4 and rng.random() < 0.35:
        middle[1] = middle[0]
    lengths = np.concatenate(([minimum, maximum], middle))
    return lengths


def _choose_extreme_lanes(rng, count, regime, pair_index):
    if regime == "source_train" and pair_index == 0:
        inside = rng.choice(np.arange(1, count - 1), size=2, replace=False)
        return int(inside[0]), int(inside[1])
    if regime == "source_train" and pair_index == 1:
        minimum_index = int(rng.choice([0, count - 1]))
        maximum_index = int(rng.choice([x for x in range(count)
                                         if x != minimum_index]))
        return minimum_index, maximum_index
    chosen = rng.choice(count, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _arrange_lengths(rng, lengths, minimum_index, maximum_index):
    count = len(lengths)
    arranged = np.empty(count, dtype=int)
    arranged[minimum_index] = lengths[0]
    arranged[maximum_index] = lengths[1]
    others = [i for i in range(count) if i not in (minimum_index, maximum_index)]
    for index, value in zip(others, rng.permutation(lengths[2:])):
        arranged[index] = value
    return tuple(int(x) for x in arranged)


def _sample_geometry(rng, regime, pair_index):
    if regime == "source_train" and pair_index == 0:
        orientation, side = "vertical", "high"
        count = int(rng.choice([4, 5]))
    else:
        if regime == "source_train" and pair_index == 1:
            count = int(rng.choice([3, 6, 7, 8]))
        elif regime == "target_test":
            count = int(rng.choice([3, 4, 5, 6, 7, 8],
                                   p=[.12, .18, .18, .20, .18, .14]))
        else:
            count = int(rng.integers(3, 9))
        # Familiar bottom-aligned bars remain in the mix; all other
        # directions use the identical length rule.
        orientation, side = (
            (("vertical", "high"), ("vertical", "low"),
             ("horizontal", "low"), ("horizontal", "high"))
            [int(rng.choice(4, p=[.33, .23, .22, .22]))]
        )

    axis_size = int(rng.integers(7, 27))
    if rng.random() < .08:
        axis_size = int(rng.integers(27, 31))
    cross_minimum = 2 * count - 1
    cross_size = int(rng.integers(cross_minimum, 31))
    raw_lengths = _sample_lengths(rng, axis_size, count, regime)
    # Keep the field color a strict majority even for the narrowest canvas.
    # This also leaves a simple, visible background cue for a new palette.
    if 2 * sum(raw_lengths) >= axis_size * cross_size:
        cross_size += 1
    lanes = _sample_lanes(rng, cross_size, count)
    min_index, max_index = _choose_extreme_lanes(rng, count, regime, pair_index)
    lengths = _arrange_lengths(rng, raw_lengths, min_index, max_index)
    largest = max(lengths)
    free = axis_size - largest
    if regime == "source_train" and pair_index == 0:
        offset = 0
    else:
        offset = int(rng.integers(0, free + 1))
    baseline = offset if side == "low" else axis_size - 1 - offset
    return orientation, side, axis_size, cross_size, lanes, lengths, baseline


def _render_pair(orientation, side, axis_size, cross_size, lanes,
                 lengths, baseline, background, foreground,
                 longest_color, shortest_color):
    canonical_input = np.full((axis_size, cross_size), background, dtype=np.int8)
    canonical_output = canonical_input.copy()
    minimum_index = int(np.argmin(lengths))
    maximum_index = int(np.argmax(lengths))
    for index, (lane, length) in enumerate(zip(lanes, lengths)):
        start = baseline if side == "low" else baseline - length + 1
        stop = start + length
        canonical_input[start:stop, lane] = foreground
        if index == minimum_index:
            canonical_output[start:stop, lane] = shortest_color
        elif index == maximum_index:
            canonical_output[start:stop, lane] = longest_color
    if orientation == "horizontal":
        return canonical_input.T.copy(), canonical_output.T.copy()
    return canonical_input, canonical_output


def generate_pair(rng: np.random.Generator, regime: str,
                  puzzle_parameters: dict, pair_index: int):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    geometry = _sample_geometry(rng, regime, pair_index)
    output_colors = {puzzle_parameters["longest_output"],
                     puzzle_parameters["shortest_output"]}
    available = [color for color in range(10) if color not in output_colors]
    background, foreground = (int(c) for c in rng.choice(available, size=2,
                                                         replace=False))
    return _render_pair(*geometry, background, foreground,
                        puzzle_parameters["longest_output"],
                        puzzle_parameters["shortest_output"])


def _parse_bar_input(input_grid):
    colors = [int(c) for c in np.unique(input_grid)]
    assert len(colors) == 2, "input needs one background and one bar color"
    candidates = []
    for background in colors:
        foreground = colors[1] if background == colors[0] else colors[0]
        for orientation in ("vertical", "horizontal"):
            mask = input_grid == foreground
            canonical = mask if orientation == "vertical" else mask.T
            lanes = np.flatnonzero(np.any(canonical, axis=0))
            if len(lanes) < 3 or np.any(np.diff(lanes) < 2):
                continue
            positions = [np.flatnonzero(canonical[:, lane]) for lane in lanes]
            if any(not np.array_equal(row, np.arange(row[0], row[-1] + 1))
                   for row in positions):
                continue
            lengths = tuple(len(row) for row in positions)
            if lengths.count(min(lengths)) != 1 or lengths.count(max(lengths)) != 1:
                continue
            if max(lengths) - min(lengths) < 2:
                continue
            starts = {int(row[0]) for row in positions}
            ends = {int(row[-1]) for row in positions}
            if len(starts) == 1:
                side, baseline = "low", next(iter(starts))
            elif len(ends) == 1:
                side, baseline = "high", next(iter(ends))
            else:
                continue
            candidates.append({
                "background": background, "foreground": foreground,
                "orientation": orientation, "side": side,
                "baseline": baseline,
                "lanes": tuple(int(x) for x in lanes),
                "lengths": lengths,
                "number": len(lanes),
                "minimum_index": int(np.argmin(lengths)),
                "maximum_index": int(np.argmax(lengths)),
            })
    assert len(candidates) == 1, "input needs one unambiguous bar decomposition"
    return candidates[0]


def _bar_masks(input_grid, info):
    canonical_shape = (input_grid.shape if info["orientation"] == "vertical"
                       else input_grid.T.shape)
    masks = []
    for lane, length in zip(info["lanes"], info["lengths"]):
        mask = np.zeros(canonical_shape, dtype=bool)
        start = (info["baseline"] if info["side"] == "low"
                 else info["baseline"] - length + 1)
        mask[start:start + length, lane] = True
        masks.append(mask if info["orientation"] == "vertical" else mask.T)
    return masks


def _expected_output(input_grid, info, longest_color, shortest_color):
    expected = np.full(input_grid.shape, info["background"], dtype=np.int8)
    masks = _bar_masks(input_grid, info)
    expected[masks[info["maximum_index"]]] = longest_color
    expected[masks[info["minimum_index"]]] = shortest_color
    return expected


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    info = _parse_bar_input(input_grid)
    masks = _bar_masks(input_grid, info)
    assert np.array_equal(input_grid == info["foreground"],
                          np.logical_or.reduce(masks))
    longest_values = np.unique(output_grid[masks[info["maximum_index"]]])
    shortest_values = np.unique(output_grid[masks[info["minimum_index"]]])
    assert len(longest_values) == len(shortest_values) == 1
    longest_color, shortest_color = int(longest_values[0]), int(shortest_values[0])
    assert len({info["background"], info["foreground"],
                longest_color, shortest_color}) == 4
    assert np.array_equal(output_grid, _expected_output(
        input_grid, info, longest_color, shortest_color)), "wrong bar transformation"
    return {**info, "longest_color": longest_color,
            "shortest_color": shortest_color}


def _prediction_from_training(train, test_input):
    roles = {(_analyze_pair(pair["input"], pair["output"])["longest_color"],
              _analyze_pair(pair["input"], pair["output"])["shortest_color"])
             for pair in train}
    assert len(roles) == 1, "training does not establish one output mapping"
    longest_color, shortest_color = next(iter(roles))
    return _expected_output(test_input, _parse_bar_input(test_input),
                            longest_color, shortest_color)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_pairs = set()
    first_background = None
    train_rank_positions = set()
    for regime, split, count in (("source_train", "train", num_train),
                                 ("target_test", "test", num_test)):
        for pair_index in range(count):
            for _ in range(256):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                input_grid, output_grid = generate_pair(
                    pair_rng, regime, parameters, pair_index)
                try:
                    info = _analyze_pair(input_grid, output_grid)
                except AssertionError:
                    continue
                if regime == "source_train" and pair_index == 1:
                    if info["background"] == first_background:
                        continue
                rank_position = (info["number"], info["minimum_index"],
                                 info["maximum_index"])
                if regime == "target_test" and rank_position in train_rank_positions:
                    continue
                input_signature = _grid_signature(input_grid)
                pair_signature = (input_signature, _grid_signature(output_grid))
                if (input_signature in seen_inputs or pair_signature in seen_pairs
                        or pair_signature in _SOURCE_PAIR_SIGNATURES):
                    continue
                seen_inputs.add(input_signature)
                seen_pairs.add(pair_signature)
                puzzle[split].append({"input": input_grid, "output": output_grid})
                if regime == "source_train":
                    train_rank_positions.add(rank_position)
                    if pair_index == 0:
                        first_background = info["background"]
                break
            else:
                raise RuntimeError("could not construct a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_generated_puzzle_impl(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    information = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_signature = _grid_signature(pair["input"])
            pair_signature = (input_signature, _grid_signature(pair["output"]))
            assert input_signature not in seen_inputs
            assert pair_signature not in seen_pairs
            assert pair_signature not in _SOURCE_PAIR_SIGNATURES
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            information[split].append(_analyze_pair(pair["input"], pair["output"]))
    train = information["train"]
    test = information["test"]
    learned = {(info["longest_color"], info["shortest_color"])
               for info in train}
    assert len(learned) == 1, "train output mapping must be consistent"
    assert all((info["longest_color"], info["shortest_color"]) in learned
               for info in test)
    assert train[0]["orientation"] == "vertical" and train[0]["side"] == "high"
    assert train[0]["baseline"] == puzzle["train"][0]["input"].shape[0] - 1
    assert any(0 < info["minimum_index"] < info["number"] - 1
               for info in train), "training needs an interior shortest bar"
    assert any(0 < info["maximum_index"] < info["number"] - 1
               for info in train), "training needs an interior longest bar"
    assert any(info["minimum_index"] in (0, info["number"] - 1)
               for info in train), "training needs a boundary shortest bar"
    train_positions = {(info["number"], info["minimum_index"],
                        info["maximum_index"]) for info in train}
    assert all((info["number"], info["minimum_index"],
                info["maximum_index"]) not in train_positions for info in test)
    assert train[0]["background"] != train[1]["background"]
    for pair in puzzle["test"]:
        assert np.array_equal(_prediction_from_training(puzzle["train"], pair["input"]),
                              pair["output"]), "test answer cannot be inferred"


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"invalid generated puzzle: {exc}") from exc


def self_test() -> None:
    train_counts, test_counts = set(), set()
    orientations, populations, shapes = set(), set(), set()
    rank_positions, palettes, layouts = set(), set(), set()
    one_cell = {"train": 0, "test": 0}
    longer_minimum = {"train": 0, "test": 0}
    ties = {"train": 0, "test": 0}
    close, wide, translated = 0, 0, 0
    within_orientation, within_baseline, within_population = 0, 0, 0
    within_background = 0
    interior_test_extrema, outer_test_intermediate = 0, 0
    combined = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {split: [_analyze_pair(p["input"], p["output"])
                         for p in puzzle[split]] for split in ("train", "test")}
        all_info = infos["train"] + infos["test"]
        within_orientation += len({(i["orientation"], i["side"])
                                   for i in all_info}) > 1
        within_baseline += len({(i["side"], i["baseline"])
                                for i in all_info}) > 1
        within_population += len({i["number"] for i in all_info}) > 1
        within_background += len({i["background"] for i in all_info}) > 1
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                inp, out = pair["input"], pair["output"]
                assert inp.shape == out.shape and not np.array_equal(inp, out)
                orientations.add((info["orientation"], info["side"]))
                populations.add(info["number"])
                shapes.add(inp.shape)
                palettes.add((info["background"], info["foreground"],
                              info["longest_color"], info["shortest_color"]))
                rank_positions.add((info["number"], info["minimum_index"],
                                    info["maximum_index"]))
                layouts.add((inp.shape, info["orientation"], info["side"],
                             info["lanes"], info["lengths"], info["baseline"]))
                signatures.add(_grid_signature(inp))
                minimum, maximum = min(info["lengths"]), max(info["lengths"])
                one_cell[split] += minimum == 1
                longer_minimum[split] += minimum > 1
                ties[split] += len(set(info["lengths"])) < info["number"]
                close += maximum - minimum <= 4
                wide += maximum - minimum >= 8
                axis_size = inp.shape[0] if info["orientation"] == "vertical" else inp.shape[1]
                translated += info["baseline"] not in (0, axis_size - 1)
                if split == "test":
                    interior_test_extrema += (0 < info["minimum_index"] < info["number"] - 1
                                              and 0 < info["maximum_index"] < info["number"] - 1)
                    outer_test_intermediate += (0 not in (info["minimum_index"], info["maximum_index"])
                                                or info["number"] - 1 not in
                                                (info["minimum_index"], info["maximum_index"]))
                    combined += (info["number"] >= 6 and
                                 info["orientation"] == "horizontal" and
                                 len(set(info["lengths"])) < info["number"] and
                                 info["baseline"] not in (0, axis_size - 1))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert populations == set(range(3, 9))
    assert len(orientations) == 4
    assert len(shapes) >= 80 and len(layouts) >= 150
    assert len(rank_positions) >= 30 and len(palettes) >= 80
    assert len(signatures) >= 150
    assert all(one_cell[s] > 0 and longer_minimum[s] > 0 and ties[s] > 0
               for s in ("train", "test"))
    assert close > 0 and wide > 0 and translated > 0
    assert interior_test_extrema > 0 and outer_test_intermediate > 0
    assert combined > 0
    assert min(within_orientation, within_baseline, within_population,
               within_background) >= 20
    assert _puzzle_signature(generate_puzzle(987654, 7, 3)) == \
           _puzzle_signature(generate_puzzle(987654, 7, 3))
    assert _puzzle_signature(generate_puzzle(17)) == _puzzle_signature(generate_puzzle(17))
    assert len({_puzzle_signature(generate_puzzle(3000 + seed, 4, 2))
                for seed in range(12)}) >= 10
    for seed, train_count, test_count in ((101, 2, 1), (102, 10, 1), (103, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for seed, requested_train, requested_test in ((201, None, 1), (202, None, 4),
                                                   (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, requested_train, requested_test)
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
        validate_generated_puzzle(puzzle)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 2, "num_test": 3},
                      {"num_train": 4.0, "num_test": 1}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {arguments}")
