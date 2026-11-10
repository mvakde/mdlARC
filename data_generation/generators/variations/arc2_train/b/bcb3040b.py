import numpy as np


# Comparing against these source inputs is stronger than merely checking that
# no complete generated input/output pair duplicates a source pair.
_SOURCE_INPUT_ROWS = {
    (
        "011010100000",
        "001001001100",
        "001000111110",
        "000000010110",
        "101001100011",
        "110100011000",
        "100100000000",
        "000010000000",
        "201110010012",
        "010000010100",
        "100010010111",
        "001001011010",
    ),
    (
        "2000000000001110",
        "0000000111110010",
        "0100000010010011",
        "0000000000111011",
        "0110110011110100",
        "0001111101101111",
        "0100000000101000",
        "0001100010000110",
        "1000100101010011",
        "1111110010110100",
        "0111010100010001",
        "0001000001000001",
        "0000000000110010",
        "1001010100001101",
        "0000010110000101",
        "0100101000001002",
    ),
    (
        "0111002001",
        "1010111011",
        "0000000010",
        "0100001110",
        "1110001001",
        "1111110010",
        "0110101010",
        "1000101101",
        "0111100111",
        "0101002110",
    ),
    (
        "110101011101111112",
        "110010100010001011",
        "011000010011010000",
        "111000011000101001",
        "010100101000000100",
        "110110110011101011",
        "100000000100110101",
        "110011110010011111",
        "001111100010111100",
        "111010000101001110",
        "000001100000010100",
        "001110110001101111",
        "111101101111101001",
        "110010101111110000",
        "100100011001101010",
        "111111101001110100",
        "110001010010010111",
        "201100000111101000",
    ),
}


def _integer_argument(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _integer_argument(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        num_test = _integer_argument(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = (
            num_test
            + 1
            + (seed // 4) % (10 - num_test)
        )
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + seed % maximum_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _grid_signature(grid):
    return grid.shape, tuple(int(value) for value in grid.ravel())


def _pair_signature(pair):
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _puzzle_signature(puzzle):
    return (
        tuple(_pair_signature(pair) for pair in puzzle["train"]),
        tuple(_pair_signature(pair) for pair in puzzle["test"]),
    )


_ORIENTATIONS = ("H", "V", "D", "A")


def _orientation_plan(rng, num_train, num_test):
    if num_train == 2:
        train = [str(rng.choice(("H", "V"))), str(rng.choice(("D", "A")))]
    else:
        train = [str(value) for value in rng.choice(_ORIENTATIONS, size=3, replace=False)]
        train.extend(str(value) for value in rng.choice(_ORIENTATIONS, size=num_train - 3))
    rng.shuffle(train)
    familiar = tuple(sorted(set(train)))
    withheld = tuple(value for value in _ORIENTATIONS if value not in familiar)
    if num_test == 1:
        pool = withheld if withheld and rng.random() < 0.52 else _ORIENTATIONS
        test = [str(rng.choice(pool))]
    else:
        test = [str(rng.choice(familiar))]
        test.append(str(rng.choice(withheld if withheld else _ORIENTATIONS)))
        test.extend(str(rng.choice(_ORIENTATIONS)) for _ in range(num_test - 2))
        rng.shuffle(test)
    return train, test


def _dimensions(rng, regime):
    # Both splits cover the full size range; tests give slightly more weight to extremes.
    weights = (0.18, 0.36, 0.20, 0.13, 0.13) if regime == "source_train" else (0.24, 0.20, 0.22, 0.17, 0.17)
    profile = int(rng.choice(5, p=weights))
    if profile == 0:
        return int(rng.integers(6, 13)), int(rng.integers(6, 13))
    if profile == 1:
        return int(rng.integers(10, 23)), int(rng.integers(10, 23))
    if profile == 2:
        return int(rng.integers(19, 31)), int(rng.integers(19, 31))
    if profile == 3:
        return int(rng.integers(6, 13)), int(rng.integers(20, 31))
    return int(rng.integers(20, 31)), int(rng.integers(6, 13))


def _offset(rng, low, high):
    if low == high:
        return low
    if rng.random() < 0.34:
        return low if rng.random() < 0.5 else high
    return int(rng.integers(low, high + 1))


def _segment_for_shape(rng, height, width, orientation):
    maximum = width if orientation == "H" else height if orientation == "V" else min(height, width)
    length_mode = int(rng.choice(3, p=(0.29, 0.40, 0.31)))
    if length_mode == 0:
        length = int(rng.integers(4, min(maximum, 8) + 1))
    elif length_mode == 1:
        length = int(rng.integers(4, maximum + 1))
    else:
        length = int(rng.integers(max(4, maximum - 2), maximum + 1))
    if orientation == "H":
        start = (int(rng.integers(height)), _offset(rng, 0, width - length))
        step = (0, 1)
    elif orientation == "V":
        start = (_offset(rng, 0, height - length), int(rng.integers(width)))
        step = (1, 0)
    elif orientation == "D":
        start = (_offset(rng, 0, height - length), _offset(rng, 0, width - length))
        step = (1, 1)
    else:
        start = (_offset(rng, 0, height - length), _offset(rng, length - 1, width - 1))
        step = (1, -1)
    return [(start[0] + i * step[0], start[1] + i * step[1]) for i in range(length)]


def _field(rng, height, width):
    mode = int(rng.choice(6, p=(0.33, 0.12, 0.12, 0.17, 0.17, 0.09)))
    if mode <= 2:
        low, high = ((0.20, 0.70), (0.035, 0.17), (0.79, 0.95))[mode]
        return rng.random((height, width)) < rng.uniform(low, high)
    if mode == 3:  # coherent patches with a small amount of scattered context
        field = rng.random((height, width)) < rng.uniform(0.025, 0.11)
        for _ in range(int(rng.integers(2, 6))):
            rh = int(rng.integers(2, max(3, height // 2 + 1)))
            rw = int(rng.integers(2, max(3, width // 2 + 1)))
            r = int(rng.integers(0, height - rh + 1))
            c = int(rng.integers(0, width - rw + 1))
            field[r:r + rh, c:c + rw] = True
        return field
    if mode == 4:  # horizontal or vertical bands
        field = rng.random((height, width)) < rng.uniform(0.025, 0.10)
        horizontal = bool(rng.integers(2))
        span = height if horizontal else width
        for _ in range(int(rng.integers(1, 4))):
            thickness = int(rng.integers(1, min(4, span) + 1))
            start = int(rng.integers(0, span - thickness + 1))
            if horizontal:
                field[start:start + thickness, :] = True
            else:
                field[:, start:start + thickness] = True
        return field
    field = rng.random((height, width)) < rng.uniform(0.16, 0.40)
    for _ in range(int(rng.integers(1, 4))):
        rh = int(rng.integers(2, max(3, height // 2 + 1)))
        rw = int(rng.integers(2, max(3, width // 2 + 1)))
        r = int(rng.integers(0, height - rh + 1))
        c = int(rng.integers(0, width - rw + 1))
        field[r:r + rh, c:c + rw] = True
    return field


def _set_crossings(rng, field, segment, regime):
    interior = segment[1:-1]
    count = len(interior)
    styles = ("natural", "isolated", "run", "mostly_foreground")
    weights = (0.38, 0.22, 0.23, 0.17)
    if regime == "target_test":
        styles += ("all_background", "all_foreground")
        weights = (0.29, 0.17, 0.19, 0.15, 0.10, 0.10)
    style = str(rng.choice(styles, p=weights))
    if style == "isolated":
        bits = np.zeros(count, dtype=bool)
        positions = list(range(count))
        rng.shuffle(positions)
        chosen = []
        for index in positions:
            if all(abs(index - other) > 1 for other in chosen):
                chosen.append(index)
            if len(chosen) >= min(3, max(1, count // 4)):
                break
        bits[chosen] = True
    elif style == "run":
        bits = np.zeros(count, dtype=bool)
        run = min(count - 1, max(2, int(round(count * rng.uniform(0.48, 0.83)))))
        start = int(rng.integers(0, count - run + 1))
        bits[start:start + run] = True
    elif style == "mostly_foreground":
        bits = np.ones(count, dtype=bool)
        indices = rng.choice(count, size=min(count - 1, max(1, count // 5)), replace=False)
        bits[indices] = False
    elif style == "all_background":
        bits = np.zeros(count, dtype=bool)
    elif style == "all_foreground":
        bits = np.ones(count, dtype=bool)
    else:
        bits = np.array([field[position] for position in interior], dtype=bool)
        if regime == "source_train":
            if not np.any(bits):
                bits[int(rng.integers(count))] = True
            if np.all(bits):
                bits[int(rng.integers(count))] = False
    for position, bit in zip(interior, bits):
        field[position] = bit


def _generate_pair(rng, regime, roles, orientation):
    height, width = _dimensions(rng, regime)
    segment = _segment_for_shape(rng, height, width, orientation)
    field = _field(rng, height, width)
    _set_crossings(rng, field, segment, regime)
    on_segment = np.zeros((height, width), dtype=bool)
    for position in segment:
        on_segment[position] = True
    outside = np.argwhere(~on_segment)
    chosen = rng.choice(len(outside), size=6, replace=False)
    for index in chosen[:3]:
        field[tuple(outside[int(index)])] = False
    for index in chosen[3:]:
        field[tuple(outside[int(index)])] = True
    background, foreground, line, intersection = roles
    inp = np.where(field, foreground, background).astype(np.int8)
    inp[segment[0]] = line
    inp[segment[-1]] = line
    out = inp.copy()
    for position in segment[1:-1]:
        out[position] = intersection if inp[position] == foreground else line
    return {"input": inp, "output": out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _integer_argument(seed, "seed") % (1 << 64)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(int(value) for value in rng.choice(10, size=4, replace=False))
    train_orientations, test_orientations = _orientation_plan(rng, num_train, num_test)
    pair_seeds = rng.integers(0, (1 << 63) - 1, size=num_train + num_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, (split, orientation) in enumerate(
        [("train", value) for value in train_orientations]
        + [("test", value) for value in test_orientations]
    ):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        regime = "source_train" if split == "train" else "target_test"
        for _ in range(8):
            pair = _generate_pair(pair_rng, regime, roles, orientation)
            key = _grid_signature(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError("could not construct unique pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _geometry(inp, line):
    endpoints = [tuple(map(int, point)) for point in np.argwhere(inp == line)]
    assert len(endpoints) == 2
    start, finish = endpoints
    dr, dc = finish[0] - start[0], finish[1] - start[1]
    if dr == 0 and dc:
        orientation = "H"
    elif dc == 0 and dr:
        orientation = "V"
    elif dr and abs(dr) == abs(dc):
        orientation = "D" if dr * dc > 0 else "A"
    else:
        raise AssertionError("markers must define an axial or 45-degree segment")
    distance = max(abs(dr), abs(dc))
    assert distance >= 3
    step = (dr // distance, dc // distance)
    segment = [(start[0] + i * step[0], start[1] + i * step[1]) for i in range(distance + 1)]
    assert segment[-1] == finish
    return orientation, segment


def _infer_roles(train_pair):
    inp, out = train_pair["input"], train_pair["output"]
    input_colors = {int(value) for value in np.unique(inp)}
    assert len(input_colors) == 3
    candidates = [color for color in input_colors if int(np.count_nonzero(inp == color)) == 2]
    assert len(candidates) == 1
    line = candidates[0]
    _, segment = _geometry(inp, line)
    new = {int(value) for value in np.unique(out)} - input_colors
    assert len(new) == 1
    intersection = next(iter(new))
    to_line = {int(inp[pos]) for pos in segment[1:-1] if int(out[pos]) == line}
    to_intersection = {int(inp[pos]) for pos in segment[1:-1] if int(out[pos]) == intersection}
    assert len(to_line) == len(to_intersection) == 1
    background, foreground = next(iter(to_line)), next(iter(to_intersection))
    assert len({background, foreground, line, intersection}) == 4
    assert input_colors == {background, foreground, line}
    return background, foreground, line, intersection


def _predict_from_training(train_pairs, test_input):
    roles = _infer_roles(train_pairs[0])
    for pair in train_pairs[1:]:
        assert _infer_roles(pair) == roles
    background, foreground, line, intersection = roles
    assert {int(value) for value in np.unique(test_input)} == {background, foreground, line}
    _, segment = _geometry(test_input, line)
    result = test_input.copy()
    for position in segment[1:-1]:
        old = int(test_input[position])
        assert old in (background, foreground)
        result[position] = line if old == background else intersection
    return result


def _inspect_pair(pair, roles):
    inp, out = pair["input"], pair["output"]
    background, foreground, line, intersection = roles
    assert inp.shape == out.shape
    assert {int(value) for value in np.unique(inp)} == {background, foreground, line}
    orientation, segment = _geometry(inp, line)
    assert int(np.count_nonzero(inp == line)) == 2
    assert int(np.count_nonzero(inp == background)) >= 3
    assert int(np.count_nonzero(inp == foreground)) >= 3
    expected = inp.copy()
    crossings = []
    for position in segment[1:-1]:
        old = int(inp[position])
        assert old in (background, foreground)
        crossings.append(old == foreground)
        expected[position] = intersection if old == foreground else line
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    height, width = inp.shape
    maximum = width if orientation == "H" else height if orientation == "V" else min(height, width)
    boundary = sum(r in (0, height - 1) or c in (0, width - 1) for r, c in (segment[0], segment[-1]))
    longest_run = 0
    current = 0
    for bit in crossings:
        current = current + 1 if bit else 0
        longest_run = max(longest_run, current)
    return {
        "orientation": orientation,
        "shape": (height, width),
        "length": len(segment),
        "maximum": maximum,
        "boundary_endpoints": boundary,
        "foreground_crossings": sum(crossings),
        "interior": len(crossings),
        "longest_run": longest_run,
        "density": int(np.count_nonzero(inp == foreground)) / (height * width - 2),
        "roles": roles,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_signature(pair["input"])
            pair_key = _pair_signature(pair)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            rows = tuple("".join(str(int(value)) for value in row) for row in pair["input"])
            assert rows not in _SOURCE_INPUT_ROWS
    roles = _infer_roles(puzzle["train"][0])
    train_info = []
    for pair in puzzle["train"]:
        assert _infer_roles(pair) == roles
        info = _inspect_pair(pair, roles)
        assert 0 < info["foreground_crossings"] < info["interior"]
        train_info.append(info)
    train_orientations = {item["orientation"] for item in train_info}
    assert len(train_orientations) >= 2
    assert train_orientations & {"H", "V"}
    assert train_orientations & {"D", "A"}
    test_info = []
    for pair in puzzle["test"]:
        info = _inspect_pair(pair, roles)
        predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])
        test_info.append(info)
    if len(test_info) >= 2 and len(train_orientations) < 4:
        assert any(item["orientation"] in train_orientations for item in test_info)
        assert any(item["orientation"] not in train_orientations for item in test_info)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(puzzle["train"]) for puzzle in puzzles}
    test_counts = {len(puzzle["test"]) for puzzle in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_puzzle_signature(puzzle) for puzzle in puzzles}) == 42
    palettes, shapes, layouts, inputs = set(), set(), set(), set()
    by_split = {"train": [], "test": []}
    mixed_test_puzzles = 0
    repeated_train_orientations = 0
    familiar_test_puzzles = 0
    withheld_test_puzzles = 0
    smaller_test_puzzles = 0
    larger_test_puzzles = 0
    diagonal_rectangles = 0
    band_cases = 0
    patch_cases = 0
    structured_crossing_cases = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        roles = _infer_roles(puzzle["train"][0])
        palettes.add(roles)
        train_orientations = {_inspect_pair(pair, roles)["orientation"] for pair in puzzle["train"]}
        test_orientations = {_inspect_pair(pair, roles)["orientation"] for pair in puzzle["test"]}
        repeated_train_orientations += len(train_orientations) < len(puzzle["train"])
        familiar_test_puzzles += bool(test_orientations & train_orientations)
        withheld_test_puzzles += bool(test_orientations - train_orientations)
        train_lengths = [_inspect_pair(pair, roles)["length"] for pair in puzzle["train"]]
        test_lengths = [_inspect_pair(pair, roles)["length"] for pair in puzzle["test"]]
        smaller_test_puzzles += min(test_lengths) < max(train_lengths)
        larger_test_puzzles += max(test_lengths) > max(train_lengths)
        if len(puzzle["test"]) >= 2 and len({_inspect_pair(pair, roles)["orientation"] for pair in puzzle["test"]}) >= 2:
            mixed_test_puzzles += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                item = _inspect_pair(pair, roles)
                by_split[split].append(item)
                shapes.add(item["shape"])
                layouts.add((item["shape"], item["orientation"], item["length"], _geometry(pair["input"], roles[2])[1][0]))
                inputs.add(_grid_signature(pair["input"]))
                height, width = item["shape"]
                diagonal_rectangles += item["orientation"] in ("D", "A") and height != width
                if height >= 10 and width >= 10:
                    foreground_mask = pair["input"] == roles[1]
                    structured = bool(
                        np.any(np.mean(foreground_mask, axis=0) >= 0.9)
                        or np.any(np.mean(foreground_mask, axis=1) >= 0.9)
                    )
                    band_cases += structured
                    patch_cases += any(
                        np.all(foreground_mask[row:row + 3, col:col + 3])
                        for row in range(height - 2)
                        for col in range(width - 2)
                    )
                    structured_crossing_cases += structured and 0 < item["foreground_crossings"] < item["interior"]
                if split == "train":
                    assert 0 < item["foreground_crossings"] < item["interior"]
                assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"].copy()), pair["output"])
        assert train_orientations & {"H", "V"} and train_orientations & {"D", "A"}
    assert len(palettes) >= 12 and len(shapes) >= 50 and len(layouts) >= 150 and len(inputs) >= 150
    assert mixed_test_puzzles >= 5
    assert repeated_train_orientations >= 5
    assert familiar_test_puzzles >= 5 and withheld_test_puzzles >= 5
    assert smaller_test_puzzles >= 5 and larger_test_puzzles >= 5
    assert diagonal_rectangles >= 20
    assert band_cases >= 5 and patch_cases >= 5 and structured_crossing_cases >= 5
    assert {item["orientation"] for item in by_split["train"]} == set(_ORIENTATIONS)
    assert {item["orientation"] for item in by_split["test"]} == set(_ORIENTATIONS)
    assert any(item["shape"][0] <= 12 and item["shape"][1] <= 12 for item in by_split["test"])
    assert any(min(item["shape"]) >= 23 for item in by_split["train"])
    assert any(item["length"] <= 7 for item in by_split["train"])
    assert any(item["length"] <= 7 for item in by_split["test"])
    assert any(item["length"] >= item["maximum"] - 2 for item in by_split["train"])
    assert any(item["length"] >= item["maximum"] - 2 for item in by_split["test"])
    assert any(max(item["shape"]) / min(item["shape"]) >= 2 for item in by_split["train"])
    assert any(max(item["shape"]) / min(item["shape"]) >= 2 for item in by_split["test"])
    assert any(item["boundary_endpoints"] == 0 for item in by_split["train"])
    assert any(item["boundary_endpoints"] == 2 for item in by_split["test"])
    assert any(item["density"] < 0.19 for split in by_split.values() for item in split)
    assert any(item["density"] > 0.75 for split in by_split.values() for item in split)
    assert any(item["foreground_crossings"] == 0 for item in by_split["test"])
    assert any(item["foreground_crossings"] == item["interior"] for item in by_split["test"])
    assert any(item["foreground_crossings"] == 1 for item in by_split["train"])
    assert any(item["longest_run"] >= 3 for item in by_split["train"])
    explicit_a = generate_puzzle(991, 7, 3)
    explicit_b = generate_puzzle(991, 7, 3)
    assert _puzzle_signature(explicit_a) == _puzzle_signature(explicit_b)
    assert len({_puzzle_signature(generate_puzzle(seed, 5, 2)) for seed in range(8)}) == 8
    for requested_train, requested_test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1200 + requested_train + requested_test, requested_train, requested_test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (requested_train, requested_test)
        validate_generated_puzzle(puzzle)
    for seed, requested_train, requested_test in ((1301, None, 1), (1302, None, 4), (1303, 2, None), (1304, 10, None)):
        puzzle = generate_puzzle(seed, requested_train, requested_test)
        assert requested_train is None or len(puzzle["train"]) == requested_train
        assert requested_test is None or len(puzzle["test"]) == requested_test
        validate_generated_puzzle(puzzle)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 3, "num_test": 3}, {"num_train": 2, "num_test": 4}):
        try:
            generate_puzzle(1400, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count arguments were accepted: {arguments}")
