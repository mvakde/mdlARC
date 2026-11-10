import numpy as np


_SOURCE_INPUTS = {
    (
        "44444444", "33633333", "44444444", "36333336",
        "44444444", "33333363", "44464444", "33333333",
    ),
    (
        "7272727272727", "7272727272727", "7272727272527",
        "7272727272757", "7572757275727", "7272727272727",
        "7272727272527", "7272757272727", "7572727252727",
        "7272727252727", "7272757272727", "7272727272725",
    ),
    (
        "11111111111", "00700000000", "11111117111",
        "00000000000", "11111111111", "00000000000",
        "11711117111", "00000070007", "11111111111",
        "00700000000", "11111111711",
    ),
    (
        "888888", "999999", "888828",
        "999999", "888888", "999999",
    ),
    (
        "464646464", "464646864", "262646464",
        "464646464", "464648464", "468646464",
        "464646484", "264646464", "464646464",
    ),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _choose_counts(rng, num_train, num_test):
    for value, name, low, high in (
        (num_train, "num_train", 2, 10),
        (num_test, "num_test", 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(
                f"{name} must be an integer from {low} to {high}"
            )

    if num_train is None:
        lower = (
            2 if num_test is None
            else max(2, int(num_test) + 1)
        )
        num_train = int(rng.integers(lower, 11))

    if num_test is None:
        upper = min(4, int(num_train) - 1)
        num_test = int(rng.integers(1, upper + 1))

    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    return int(num_train), int(num_test)


def _clean_grid(shape, orientation, first, second):
    height, width = shape

    if orientation == "H":
        stripe = np.where(
            np.arange(height) % 2 == 0, first, second
        ).astype(np.int8)
        return np.repeat(stripe[:, None], width, axis=1)

    stripe = np.where(
        np.arange(width) % 2 == 0, first, second
    ).astype(np.int8)
    return np.repeat(stripe[None, :], height, axis=0)


def _best_patterns(grid):
    best_score = grid.size + 1
    winners = []

    for orientation in ("H", "V"):
        if orientation == "H":
            even = grid[::2, :]
            odd = grid[1::2, :]
        else:
            even = grid[:, ::2]
            odd = grid[:, 1::2]

        even_counts = np.bincount(even.ravel(), minlength=10)
        odd_counts = np.bincount(odd.ravel(), minlength=10)

        for first in range(10):
            for second in range(10):
                if first == second:
                    continue

                score = int(
                    even.size - even_counts[first]
                    + odd.size - odd_counts[second]
                )

                if score < best_score:
                    best_score = score
                    winners = [(orientation, first, second)]
                elif score == best_score:
                    winners.append((orientation, first, second))

    return best_score, winners


def _rows(grid):
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _signature(grid):
    return (
        grid.shape,
        tuple(int(value) for value in grid.flat),
    )



def _output_pattern(output):
    found = []
    for direction in ("H", "V"):
        first = int(output[0, 0])
        second = int(output[1, 0] if direction == "H" else output[0, 1])
        if first != second and np.array_equal(
            output, _clean_grid(output.shape, direction, first, second)
        ):
            found.append((direction, first, second))
    return found


def _make_pair(rng, regime, shape=None):
    # Dimensions, direction, colors, and damage are local to each pair.
    if shape is None:
        shape = tuple(
            6 if rng.integers(10) == 0 else int(rng.integers(6, 19))
            for _ in range(2)
        )
    direction = "H" if rng.integers(2) == 0 else "V"
    palette = [int(color) for color in rng.permutation(10)]
    first, second = palette[:2]
    probabilities = ((0.66, 0.27, 0.07) if regime == "source_train"
                     else (0.47, 0.38, 0.15))
    noise_count = int(rng.choice((1, 2, 3), p=probabilities))
    noise_colors = palette[2:2 + noise_count]
    output = _clean_grid(shape, direction, first, second)
    height, width = shape
    upper = min(18, max(4, int(round(height * width * 0.13))))
    count = int(rng.integers(noise_count, upper + 1))
    if regime == "target_test" and count < upper and rng.integers(3) == 0:
        count += 1
    mode = int(rng.choice(3, p=(0.56, 0.26, 0.18)))
    line_length = width if direction == "H" else height
    line_count = height if direction == "H" else width
    locations = set()
    if mode and count >= 3:
        line = int(rng.integers(line_count))
        cluster_size = min(count, line_length - 1, int(rng.integers(3, 8)))
        if mode == 1:
            start = int(rng.integers(line_length - cluster_size + 1))
            positions = range(start, start + cluster_size)
        else:
            positions = rng.choice(line_length, size=cluster_size, replace=False)
        for position in positions:
            point = ((line, int(position)) if direction == "H"
                     else (int(position), line))
            locations.add(point)
    for flat in rng.permutation(height * width):
        if len(locations) == count:
            break
        locations.add(divmod(int(flat), width))
    colors = list(noise_colors)
    colors.extend(noise_colors[int(rng.integers(noise_count))]
                  for _ in range(count - noise_count))
    rng.shuffle(colors)
    input_grid = output.copy()
    for (row, col), color in zip(sorted(locations), colors):
        input_grid[row, col] = color
    return {"input": input_grid, "output": output}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    # The source has no hidden color map, direction, or other puzzle convention.
    # The shared convention is the demonstrated stripe-repair transformation.
    puzzle = {"train": [], "test": []}
    seen = set()
    for regime, split, count in (
        ("source_train", "train", num_train),
        ("target_test", "test", num_test),
    ):
        for _ in range(count):
            for _attempt in range(80):
                shared_shape = None
                if split == "test" and rng.integers(4) == 0:
                    source_pair = puzzle["train"][int(rng.integers(num_train))]
                    shared_shape = source_pair["input"].shape
                pair = _make_pair(rng, regime, shared_shape)
                signature = _signature(pair["input"])
                if signature in seen or _rows(pair["input"]) in _SOURCE_INPUTS:
                    continue
                score, winners = _best_patterns(pair["input"])
                actual = int(np.count_nonzero(pair["input"] != pair["output"]))
                if score == actual and winners == _output_pattern(pair["output"]):
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a unique stripe repair pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair):
    _require(type(pair) is dict and set(pair) == {"input", "output"}, "Bad pair structure")
    input_grid, output = pair["input"], pair["output"]
    for grid in (input_grid, output):
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "Grid must be a 2D NumPy array")
        _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, "Grid size out of range")
        _require(grid.dtype.kind in "iu", "Grid dtype must be integer")
        _require(bool(np.all((0 <= grid) & (grid <= 9))), "Grid colors out of range")
    _require(input_grid.shape == output.shape, "Dimensions changed")
    _require(min(input_grid.shape) >= 6, "Insufficient stripe evidence")
    patterns = _output_pattern(output)
    _require(len(patterns) == 1, "Output is not one alternating stripe pattern")
    changed = input_grid != output
    count = int(np.count_nonzero(changed))
    _require(1 <= count <= min(18, max(4, int(round(input_grid.size * 0.13)))),
             "Damage is not sparse")
    noise_colors = set(int(value) for value in np.unique(input_grid[changed]))
    _require(noise_colors.isdisjoint(set(patterns[0][1:])), "Damage must use foreign colors")
    _require(1 <= len(noise_colors) <= 3, "Unexpected noise palette")
    score, winners = _best_patterns(input_grid)
    _require(score == count and winners == patterns, "Stripe restoration is ambiguous")
    _require(_rows(input_grid) not in _SOURCE_INPUTS, "Copied a source pair")
    return patterns[0], len(noise_colors), count


def _predict_from_examples(train_pairs, test_input):
    for pair in train_pairs:
        _, winners = _best_patterns(pair["input"])
        _require(len(winners) == 1 and np.array_equal(
            _clean_grid(pair["input"].shape, *winners[0]), pair["output"]
        ), "Training examples do not establish the repair rule")
    _, winners = _best_patterns(test_input)
    _require(len(winners) == 1, "Test input has ambiguous stripes")
    return _clean_grid(test_input.shape, *winners[0])


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(type(puzzle) is dict and set(puzzle) == {"train", "test"}, "Bad puzzle structure")
    train, test = puzzle["train"], puzzle["test"]
    _require(type(train) is list and type(test) is list, "Train and test must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "Bad pair counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _validate_pair(pair)
            input_signature = _signature(pair["input"])
            pair_signature = input_signature, _signature(pair["output"])
            _require(input_signature not in seen_inputs, "Duplicate input")
            _require(pair_signature not in seen_pairs, "Duplicate pair")
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
    # A unique stripe restoration in each input makes all pair-level choices
    # visible; the training examples establish what transformation to perform.
    demonstrations = [{"input": pair["input"].copy(), "output": pair["output"].copy()}
                      for pair in train]
    for pair in test:
        predicted = _predict_from_examples(demonstrations, pair["input"].copy())
        _require(np.array_equal(predicted, pair["output"]), "Test output is not recoverable")


def _same_puzzle(first, second):
    if set(first) != set(second):
        return False
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if (not np.array_equal(a["input"], b["input"])
                    or not np.array_equal(a["output"], b["output"])):
                return False
    return True


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, palettes, damage_counts, noise_sizes, first_inputs = set(), set(), set(), set(), set()
    train_schedules, test_schedules = set(), set()
    square = even_rectangle = mixed_parity = small_rectangle = 0
    train_multicolor = test_multicolor = concentrated = role_crossovers = 0
    small_square = one_cell = old_cap_exceeded = combined_modes = 0
    overlapping_shapes = varied_noise_puzzles = compact_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        compact_puzzles += (len(puzzle["train"]), len(puzzle["test"])) == (2, 1)
        overlapping_shapes += bool(
            {pair["input"].shape for pair in puzzle["train"]}
            & {pair["input"].shape for pair in puzzle["test"]}
        )
        first_inputs.add(_signature(puzzle["train"][0]["input"]))
        if seed in (0, 1, 17, 41):
            _require(_same_puzzle(puzzle, generate_puzzle(seed)), "Nondeterministic puzzle")
        directions = {"train": [], "test": []}
        local_noise_sizes = set()
        prior_stripes, prior_noise = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pattern, noise_size, count = _validate_pair(pair)
                grid, output = pair["input"], pair["output"]
                score, winners = _best_patterns(grid)
                _require(score == count and len(winners) == 1 and np.array_equal(
                    _clean_grid(grid.shape, *winners[0]), output
                ), "Input-only repair failed")
                shapes.add(grid.shape)
                palettes.add(pattern[1:])
                damage_counts.add(count)
                noise_sizes.add(noise_size)
                local_noise_sizes.add(noise_size)
                height, width = grid.shape
                small_square += grid.shape == (6, 6)
                one_cell += count == 1
                square += height == width
                even_rectangle += height != width and height % 2 == width % 2 == 0
                mixed_parity += height % 2 != width % 2
                small_rectangle += height != width and min(height, width) <= 7
                changed = grid != output
                per_line = np.sum(changed, axis=1 if pattern[0] == "H" else 0)
                concentrated += bool(np.max(per_line) >= 3)
                line_length = width if pattern[0] == "H" else height
                old_cap_exceeded += bool(np.max(per_line) > max(1, line_length // 3))
                combined_modes += bool(noise_size >= 2 and np.max(per_line) >= 3)
                stripes = set(pattern[1:])
                noise = set(int(value) for value in np.unique(grid[changed]))
                role_crossovers += bool(stripes & prior_noise or noise & prior_stripes)
                prior_stripes |= stripes
                prior_noise |= noise
                directions[split].append(pattern[0])
                if split == "train":
                    train_multicolor += noise_size > 1
                else:
                    test_multicolor += noise_size > 1
        train_schedules.add(tuple(directions["train"][:2]))
        varied_noise_puzzles += len(local_noise_sizes) >= 2
        if len(directions["test"]) >= 2:
            test_schedules.add(tuple(directions["test"][:2]))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "Pair counts lack diversity")
    _require(len(first_inputs) >= 40 and len(shapes) >= 45 and len(palettes) >= 35,
             "Shape, palette, or puzzle diversity is too narrow")
    _require(len(damage_counts) >= 12 and noise_sizes == {1, 2, 3}, "Damage diversity missing")
    _require(len(train_schedules) == 4 and len(test_schedules) == 4,
             "Orientation schedules lack diversity")
    _require(min(square, even_rectangle, mixed_parity, small_rectangle) >= 5,
             "Shape regimes lack diversity")
    _require(train_multicolor >= 8 and test_multicolor >= 8 and concentrated >= 15
             and role_crossovers >= 15, "Damage or color role diversity missing")
    _require(compact_puzzles >= 1 and overlapping_shapes >= 5 and varied_noise_puzzles >= 20,
             "Within-puzzle counts, shapes, or damage palettes lack diversity")
    _require(small_square >= 1 and one_cell >= 1 and old_cap_exceeded >= 15
             and combined_modes >= 15, "Small or combined damage regimes missing")
    cases = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))
    for index, (requested_train, requested_test) in enumerate(cases):
        seed = 1000 + index
        puzzle = generate_puzzle(seed, requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        _require(_same_puzzle(puzzle, generate_puzzle(seed, requested_train, requested_test)),
                 "Requested counts are nondeterministic")
        _require(requested_train is None or len(puzzle["train"]) == requested_train,
                 "Requested train count ignored")
        _require(requested_test is None or len(puzzle["test"]) == requested_test,
                 "Requested test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (2, 3), (4, 4), (2.0, 1)):
        try:
            generate_puzzle(999, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts were accepted")
