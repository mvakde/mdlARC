import numpy as np


_SOURCE_ROWS = {
    tuple(int(c) for c in text)
    for text in (
        "3142789",
        "833373334",
        "43134",
        "754",
        "46060606064",
        "000000000",
    )
}
_TRAIN_WIDTHS = tuple(range(3, 18, 2))
_ALL_WIDTHS = tuple(range(3, 30, 2))


def _legal_counts(
    num_train: int | None,
    num_test: int | None,
) -> list[tuple[int, int]]:
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is None:
            continue
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")

    choices = [
        (train_count, test_count)
        for train_count in range(2, 11)
        for test_count in range(1, 5)
        if train_count > test_count
        and (num_train is None or train_count == num_train)
        and (num_test is None or test_count == num_test)
    ]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices


def _choose_colors(
    rng: np.random.Generator,
    bank: tuple[int, ...],
    center: int,
    width: int,
    style: str,
) -> list[int]:
    if style == "alternating":
        upper = 2
    elif style == "periodic":
        upper = min(4, len(bank), width)
    elif style == "blocks":
        upper = min(5, len(bank), width)
    else:
        upper = min(8, len(bank), width)
    if style not in ("alternating", "periodic") and rng.random() < 0.55:
        upper = min(upper, 3)
    count = int(rng.integers(2, upper + 1))
    others = [color for color in bank if color != center]
    selected = rng.choice(others, size=count - 1, replace=False)
    return [center] + [int(color) for color in selected]


def _block_flank(
    rng: np.random.Generator,
    length: int,
    colors: list[int],
) -> list[int]:
    result: list[int] = []
    previous = -1
    while len(result) < length:
        candidates = [color for color in colors if color != previous]
        if not candidates:
            candidates = colors
        color = int(rng.choice(candidates))
        remaining = length - len(result)
        run = int(rng.integers(1, min(remaining, 5) + 1))
        result.extend([color] * run)
        previous = color
    return result


def _make_row(
    rng: np.random.Generator,
    width: int,
    bank: tuple[int, ...],
    center: int,
    repeated: bool,
    informative: bool,
    regime: str,
) -> tuple[int, ...]:
    if repeated:
        styles = ["irregular", "mirror", "alternating", "periodic", "blocks", "solid"]
        weights = (
            [0.29, 0.17, 0.15, 0.14, 0.17, 0.08]
            if regime == "train"
            else [0.22, 0.20, 0.16, 0.15, 0.15, 0.12]
        )
        if width == 3:
            weights[2] = 0.0
            weights[3] = 0.0
            weights[1] = 0.0
        elif width == 5:
            weights[3] = 0.0
        if informative:
            weights[-1] = 0.0
    else:
        styles = ["irregular", "mirror", "blocks"]
        weights = [0.46, 0.29, 0.25]
    probabilities = np.array(weights, dtype=float)
    probabilities /= probabilities.sum()
    style = str(rng.choice(styles, p=probabilities))

    if style == "solid":
        return (center,) * width

    colors = _choose_colors(rng, bank, center, width, style)
    others = colors[1:]
    middle = width // 2

    if style == "irregular":
        row = [
            int(color)
            for color in rng.choice(others, size=width, replace=True)
        ]
        if not repeated and len(others) >= width - 1 and rng.random() < 0.38:
            distinct = rng.choice(others, size=width - 1, replace=False)
            row = [int(color) for color in distinct]
            row.insert(middle, center)
            return tuple(row)
        row[middle] = center
        if repeated:
            positions = [i for i in range(width) if i != middle]
            if informative:
                positions = [i for i in positions if i not in (0, width - 1)]
            method = int(rng.integers(4))
            if method == 0 and not informative:
                selected = [int(rng.choice([0, width - 1]))]
            elif method == 1:
                adjacent = [i for i in (middle - 1, middle + 1) if i in positions]
                selected = [int(rng.choice(adjacent or positions))]
            else:
                minimum = max(1, len(positions) // 2) if method == 3 else 1
                amount = int(rng.integers(minimum, len(positions) + 1))
                selected = [
                    int(i) for i in rng.choice(positions, size=amount, replace=False)
                ]
            for position in selected:
                row[position] = center
        return tuple(row)

    if style == "mirror":
        half = [
            int(color)
            for color in rng.choice(
                colors if repeated else others,
                size=middle,
                replace=True,
            )
        ]
        if repeated:
            half[int(rng.integers(middle))] = center
        return tuple(half + [center] + half[::-1])

    if style == "alternating":
        other = others[0]
        return tuple(
            center if index % 2 == middle % 2 else other
            for index in range(width)
        )

    if style == "periodic":
        period = int(rng.integers(2, min(4, middle) + 1))
        pattern = [
            int(color)
            for color in rng.choice(others, size=period, replace=True)
        ]
        pattern[middle % period] = center
        return tuple(pattern[index % period] for index in range(width))

    if repeated:
        maximum = min(width - (2 if informative else 1), max(2, width // 2 + 1))
        run = int(rng.integers(2, maximum + 1))
        low = max(0, middle - run + 1)
        high = min(middle, width - run)
        if informative:
            low = max(low, 1)
            high = min(high, width - run - 1)
        start = int(rng.integers(low, high + 1))
        left = _block_flank(rng, start, others)
        right = _block_flank(rng, width - start - run, others)
        return tuple(left + [center] * run + right)
    left = _block_flank(rng, middle, others)
    right = _block_flank(rng, middle, others)
    return tuple(left + [center] + right)


def _longest_run(row: tuple[int, ...]) -> int:
    best = current = 1
    for index in range(1, len(row)):
        current = current + 1 if row[index] == row[index - 1] else 1
        best = max(best, current)
    return best


def _shape_signature(row: tuple[int, ...]) -> tuple[int, int, int, bool, int]:
    center = row[len(row) // 2]
    return (
        len(row),
        row.count(center),
        len(set(row)),
        row == row[::-1],
        _longest_run(row),
    )


def _choose_test_width(
    rng: np.random.Generator,
    train_widths: list[int],
) -> int:
    choice = rng.random()
    if choice < 0.30:
        return int(rng.choice(train_widths))
    if choice < 0.50:
        candidates = [width for width in _ALL_WIDTHS if width < min(train_widths)]
        if not candidates:
            candidates = [3, 5, 7]
        return int(rng.choice(candidates))
    if choice < 0.82:
        candidates = [width for width in _ALL_WIDTHS if width > max(train_widths)]
        return int(rng.choice(candidates))
    return int(rng.choice(_ALL_WIDTHS))


def _pair(row: tuple[int, ...]) -> dict:
    center = row[len(row) // 2]
    return {
        "input": np.array([row], dtype=np.int8),
        "output": np.array([[center]], dtype=np.int8),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    choices = _legal_counts(num_train, num_test)
    train_count, test_count = choices[int(rng.integers(len(choices)))]

    bank_size = int(rng.integers(3, 11))
    bank = tuple(int(color) for color in rng.permutation(10)[:bank_size])
    source_shift = bool(rng.random() < 0.23)
    mono_color = int((seed // 3) % 10) if seed % 3 == 0 else None
    if mono_color is not None and mono_color not in bank:
        bank = bank[:-1] + (mono_color,)
    mono_in_train = bool(
        mono_color is not None
        and train_count >= 3
        and not source_shift
        and rng.random() < 0.5
    )
    anchor_colors = [int(color) for color in rng.choice(bank, size=2, replace=False)]

    if source_shift and train_count >= 3 and rng.random() < 0.45:
        train_widths = [5, 7, 9]
    else:
        first = int(rng.choice(_TRAIN_WIDTHS))
        second = int(rng.choice([width for width in _TRAIN_WIDTHS if width >= 5 and width != first]))
        train_widths = [first, second]
    while len(train_widths) < train_count:
        train_widths.append(int(rng.choice(_TRAIN_WIDTHS)))

    puzzle = {"train": [], "test": []}
    seen: set[tuple[int, ...]] = set()
    train_signatures: set[tuple[int, int, int, bool, int]] = set()
    for index, width in enumerate(train_widths):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        forced_mono = mono_in_train and index == 2
        center = (
            mono_color if forced_mono
            else anchor_colors[index] if index < 2
            else int(pair_rng.choice(bank))
        )
        repeated = False if source_shift else (
            forced_mono or index == 1 or (index >= 2 and pair_rng.random() < 0.58)
        )
        informative = index < 2
        for attempt in range(192):
            if forced_mono and attempt and attempt % 24 == 0:
                width = int(pair_rng.choice(_TRAIN_WIDTHS))
                train_widths[index] = width
            row = (
                (center,) * width if forced_mono
                else _make_row(
                    pair_rng, width, bank, center, repeated, informative, "train"
                )
            )
            if row in seen or row in _SOURCE_ROWS:
                continue
            if informative and (row[0] == center or row[-1] == center):
                continue
            break
        else:
            raise RuntimeError("Could not create a unique training row")
        seen.add(row)
        train_signatures.add(_shape_signature(row))
        puzzle["train"].append(_pair(row))

    for index in range(test_count):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        forced_mono = mono_color is not None and not mono_in_train and index == 0
        if source_shift and index == 0:
            width = int(pair_rng.choice([w for w in _ALL_WIDTHS if w > max(train_widths)]))
            repeated = True
        else:
            width = _choose_test_width(pair_rng, train_widths)
            repeated = bool(pair_rng.random() < 0.63)
        if forced_mono:
            repeated = True
        center = mono_color if forced_mono else int(pair_rng.choice(bank))
        for attempt in range(192):
            if attempt and attempt % 48 == 0:
                width = _choose_test_width(pair_rng, train_widths)
            row = (
                (center,) * width if forced_mono
                else _make_row(pair_rng, width, bank, center, repeated, False, "test")
            )
            if row in seen or row in _SOURCE_ROWS:
                continue
            if _shape_signature(row) in train_signatures:
                continue
            break
        else:
            raise RuntimeError("Could not create a new test configuration")
        seen.add(row)
        puzzle["test"].append(_pair(row))

    puzzle["train"] = [
        puzzle["train"][int(i)] for i in rng.permutation(train_count)
    ]
    puzzle["test"] = [
        puzzle["test"][int(i)] for i in rng.permutation(test_count)
    ]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs: set[tuple[int, ...]] = set()
    seen_pairs: set[tuple[tuple[int, ...], int]] = set()
    train_signatures: set[tuple[int, int, int, bool, int]] = set()
    train_widths: set[int] = set()
    train_centers: set[int] = set()
    informative: list[tuple[int, int]] = []
    test_signatures: list[tuple[int, int, int, bool, int]] = []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            input_grid = pair["input"]
            output_grid = pair["output"]
            assert input_grid.shape[0] == 1
            width = input_grid.shape[1]
            assert 3 <= width <= 29 and width % 2 == 1
            assert output_grid.shape == (1, 1)
            row = tuple(int(color) for color in input_grid[0])
            center = row[width // 2]
            assert int(output_grid[0, 0]) == center
            assert row not in seen_inputs and row not in _SOURCE_ROWS
            pair_key = (row, int(output_grid[0, 0]))
            assert pair_key not in seen_pairs
            seen_inputs.add(row)
            seen_pairs.add(pair_key)
            signature = _shape_signature(row)
            if regime == "train":
                assert width in _TRAIN_WIDTHS
                train_signatures.add(signature)
                train_widths.add(width)
                train_centers.add(center)
                if center != row[0] and center != row[-1]:
                    informative.append((width, center))
            else:
                test_signatures.append(signature)
    assert len(train_centers) >= 2
    assert len(train_widths) >= 2
    assert len(informative) >= 2
    assert len({color for _, color in informative}) >= 2
    assert len({width for width, _ in informative}) >= 2
    assert all(signature not in train_signatures for signature in test_signatures)


def _same_puzzles(left: dict, right: dict) -> bool:
    if set(left) != set(right):
        return False
    for regime in ("train", "test"):
        if len(left[regime]) != len(right[regime]):
            return False
        for first, second in zip(left[regime], right[regime]):
            for name in ("input", "output"):
                if first[name].dtype != second[name].dtype:
                    return False
                if not np.array_equal(first[name], second[name]):
                    return False
    return True


def self_test() -> None:
    train_counts: set[int] = set()
    test_counts: set[int] = set()
    train_widths: set[int] = set()
    test_widths: set[int] = set()
    palettes: set[tuple[int, ...]] = set()
    signatures: set[tuple[int, int, int, bool, int]] = set()
    puzzle_rows: set[tuple[tuple[int, ...], ...]] = set()
    features = {
        "train_multicolor_palindrome": 0,
        "test_multicolor_palindrome": 0,
        "train_repeated_center": 0,
        "train_unique_center": 0,
        "test_repeated_center": 0,
        "test_unique_center": 0,
        "short_repeated_test": 0,
        "short_matching_flanks": 0,
        "short_all_distinct": 0,
        "long_unique_test": 0,
        "long_sparse_mirrored_repeat": 0,
        "alternating": 0,
        "periodic": 0,
        "blocks": 0,
        "nonsymmetric_multicolor": 0,
        "center_at_edge": 0,
        "center_adjacent": 0,
        "center_majority": 0,
        "center_minority": 0,
        "center_tied": 0,
        "zero_mono": 0,
        "mono_widths": set(),
        "mono_colors": set(),
        "train_mono": 0,
        "test_mono": 0,
        "sparse_long_test": 0,
        "familiar_test": 0,
        "familiar_changed_frequency": 0,
        "familiar_changed_symmetry": 0,
        "long_extrapolation": 0,
        "short_extrapolation": 0,
        "source_shift": 0,
        "mixed_palettes": 0,
        "mixed_train_frequency": 0,
        "mixed_train_symmetry": 0,
    }
    for seed in range(1000, 1042):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzles(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        rows: list[tuple[int, ...]] = []
        pair_palettes: set[tuple[int, ...]] = set()
        training_widths = {
            int(pair["input"].shape[1]) for pair in puzzle["train"]
        }
        training_rows = [
            tuple(int(color) for color in pair["input"][0])
            for pair in puzzle["train"]
        ]
        testing_rows = [
            tuple(int(color) for color in pair["input"][0])
            for pair in puzzle["test"]
        ]
        if any(len(row) > max(training_widths) for row in testing_rows):
            features["long_extrapolation"] += 1
        if any(len(row) < min(training_widths) for row in testing_rows):
            features["short_extrapolation"] += 1
        if any(row.count(row[len(row) // 2]) == 1 for row in training_rows) and any(
            row.count(row[len(row) // 2]) > 1 for row in training_rows
        ):
            features["mixed_train_frequency"] += 1
        if any(row == row[::-1] for row in training_rows) and any(
            row != row[::-1] for row in training_rows
        ):
            features["mixed_train_symmetry"] += 1
        training_unique = True
        repeated_test = False
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                row = tuple(int(color) for color in pair["input"][0])
                output = pair["output"]
                center = row[len(row) // 2]
                assert output.shape == (1, 1)
                assert int(output[0, 0]) == center
                assert row not in _SOURCE_ROWS
                rows.append(row)
                palette = tuple(sorted(set(row)))
                palettes.add(palette)
                pair_palettes.add(palette)
                signatures.add(_shape_signature(row))
                if regime == "train":
                    train_widths.add(len(row))
                    training_unique &= row.count(center) == 1
                else:
                    test_widths.add(len(row))
                    repeated_test |= row.count(center) > 1
                    if len(row) in training_widths:
                        features["familiar_test"] += 1
                        peers = [item for item in training_rows if len(item) == len(row)]
                        if all(
                            (item.count(item[len(item) // 2]) > 1)
                            != (row.count(center) > 1)
                            for item in peers
                        ):
                            features["familiar_changed_frequency"] += 1
                        if all((item == item[::-1]) != (row == row[::-1]) for item in peers):
                            features["familiar_changed_symmetry"] += 1
                    if len(row) == 3 and row.count(center) > 1:
                        features["short_repeated_test"] += 1
                    if len(row) == 3 and row[0] == row[-1] and len(palette) > 1:
                        features["short_matching_flanks"] += 1
                    if len(row) == 3 and len(palette) == 3:
                        features["short_all_distinct"] += 1
                    if len(row) >= 11 and row.count(center) == 1:
                        features["long_unique_test"] += 1
                    if len(row) >= 11 and len(palette) <= 3:
                        features["sparse_long_test"] += 1
                    if (
                        len(row) >= 11 and len(palette) in (2, 3)
                        and row == row[::-1] and row.count(center) > 1
                    ):
                        features["long_sparse_mirrored_repeat"] += 1
                if row.count(center) == 1:
                    features[regime + "_unique_center"] += 1
                else:
                    features[regime + "_repeated_center"] += 1
                    if center in (row[0], row[-1]):
                        features["center_at_edge"] += 1
                    if center in (row[len(row) // 2 - 1], row[len(row) // 2 + 1]):
                        features["center_adjacent"] += 1
                if len(palette) > 1:
                    other_max = max(row.count(color) for color in palette if color != center)
                    if row.count(center) > other_max:
                        features["center_majority"] += 1
                    elif row.count(center) == other_max:
                        features["center_tied"] += 1
                    else:
                        features["center_minority"] += 1
                if len(palette) > 1 and row == row[::-1]:
                    features[regime + "_multicolor_palindrome"] += 1
                if len(palette) > 1 and row != row[::-1]:
                    features["nonsymmetric_multicolor"] += 1
                if len(palette) > 1 and all(row[i] == row[i % 2] for i in range(len(row))):
                    features["alternating"] += 1
                if len(palette) > 1 and len(row) >= 7 and any(
                    all(row[i] == row[i % period] for i in range(len(row)))
                    for period in (3, 4)
                ):
                    features["periodic"] += 1
                if len(palette) > 1 and _longest_run(row) >= 3:
                    features["blocks"] += 1
                if len(palette) == 1:
                    features["mono_widths"].add(len(row))
                    features["mono_colors"].add(center)
                    features[regime + "_mono"] += 1
                    if center == 0:
                        features["zero_mono"] += 1
        if training_unique and repeated_test:
            features["source_shift"] += 1
        if len(pair_palettes) >= 2:
            features["mixed_palettes"] += 1
        assert len(rows) == len(set(rows))
        puzzle_rows.add(tuple(rows))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_rows) >= 40
    assert len(train_widths) >= 7 and len(test_widths) >= 10
    assert len(palettes) >= 30 and len(signatures) >= 60
    for key in (
        "train_multicolor_palindrome",
        "test_multicolor_palindrome",
        "train_repeated_center",
        "train_unique_center",
        "test_repeated_center",
        "test_unique_center",
        "short_repeated_test",
        "short_matching_flanks",
        "short_all_distinct",
        "long_unique_test",
        "long_sparse_mirrored_repeat",
        "alternating",
        "periodic",
        "blocks",
        "nonsymmetric_multicolor",
        "center_at_edge",
        "center_adjacent",
        "center_majority",
        "center_minority",
        "center_tied",
        "zero_mono",
        "train_mono",
        "test_mono",
        "sparse_long_test",
        "familiar_test",
        "familiar_changed_frequency",
        "familiar_changed_symmetry",
        "long_extrapolation",
        "short_extrapolation",
        "source_shift",
        "mixed_train_frequency",
        "mixed_train_symmetry",
    ):
        assert features[key] > 0, key
    assert len(features["mono_widths"]) >= 3
    assert len(features["mono_colors"]) >= 8
    assert features["mixed_palettes"] >= 35

    for seed, train_count, test_count in (
        (57, 2, 1),
        (58, 10, 1),
        (59, 10, 4),
        (60, 5, 3),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        assert _same_puzzles(puzzle, generate_puzzle(seed, train_count, test_count))

    for seed, train_count, test_count in (
        (70, None, 1),
        (71, None, 4),
        (72, 2, None),
        (73, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        assert _same_puzzles(puzzle, generate_puzzle(seed, train_count, test_count))

    for train_count, test_count in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (4, 4), (2, 3),
    ):
        try:
            generate_puzzle(123, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid counts were accepted")

    invalid = generate_puzzle(901, 2, 1)
    old = int(invalid["test"][0]["output"][0, 0])
    invalid["test"][0]["output"] = np.array([[(old + 1) % 10]], dtype=np.int8)
    try:
        validate_generated_puzzle(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("Incorrect center output was accepted")
