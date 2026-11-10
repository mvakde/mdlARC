import numpy as np


# Exact inputs from the solved task. Reflection fixes their outputs uniquely.
_SOURCE_INPUT_SPECS = (
    (10, 5, 0, (2, 8)),
    (10, 3, 0, (2, 2, 3)),
    (10, 6, 0, (3, 5, 5)),
)
_SIDES = ("top", "left", "bottom", "right")
_ROTATIONS = {side: index for index, side in enumerate(_SIDES)}


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(token, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + token % 4
        train = max(test + 1, 2 + (token // 4) % 9)
    elif train is None:
        train = test + 1 + (token // 7) % (10 - test)
    else:
        test = 1 + (token // 11) % min(4, train - 1)
    return int(train), int(test)


def _grid_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _is_source_input(grid):
    for height, width, background, sequence in _SOURCE_INPUT_SPECS:
        if grid.shape != (height, width):
            continue
        if (np.all(grid[:len(sequence)] == np.asarray(sequence)[:, None])
                and np.all(grid[len(sequence):] == background)):
            return True
    return False


def _orientation_plan(rng, train_count, test_count):
    # Only the center-reflection rule is learned. Orientation is visible in
    # each input; these regimes vary how widely training demonstrates it.
    mode = int(rng.integers(0, 4))
    if mode == 0:
        return ["top"] * train_count, ["top"] * test_count
    if mode == 1:
        side = _SIDES[int(rng.integers(0, 4))]
        return [side] * train_count, [side] * test_count

    choices = [str(x) for x in rng.permutation(_SIDES)]
    if mode == 2:
        n = int(rng.integers(2, min(4, train_count) + 1))
        demonstrated = choices[:n]
        train = demonstrated + [demonstrated[int(rng.integers(n))]
                                for _ in range(train_count - n)]
        rng.shuffle(train)
        test = [demonstrated[int(rng.integers(n))] for _ in range(test_count)]
        return train, test

    vertical = [str(x) for x in rng.permutation(("top", "bottom"))]
    horizontal = [str(x) for x in rng.permutation(("left", "right"))]
    choices = [vertical[0], horizontal[0], vertical[1], horizontal[1]]
    train = choices[:2] + [choices[int(rng.integers(2))]
                           for _ in range(train_count - 2)]
    rng.shuffle(train)
    held_out = choices[2:]
    test = [held_out[int(rng.integers(2))] for _ in range(test_count)]
    return train, test


def _ranges(size_regime, split):
    if size_regime == "classic":
        return ((8, 14), (3, 5)) if split == "train" else ((10, 20), (6, 10))
    if size_regime == "overlap":
        return (5, 24), (2, 24)
    if size_regime == "narrow_test":
        return ((7, 30), (9, 28)) if split == "train" else ((3, 30), (2, 8))
    if size_regime == "shallow_test":
        return (7, 30), (2, 26) if split == "train" else (2, 26)
    return (3, 30), (2, 30)


def _dimensions(rng, size_regime, split, witness):
    length_range, span_range = _ranges(size_regime, split)
    low = max(length_range[0], 7 if witness else 3)
    length = int(rng.integers(low, length_range[1] + 1))
    span = int(rng.integers(span_range[0], span_range[1] + 1))
    if size_regime != "classic" and span_range[0] <= length <= span_range[1]:
        if int(rng.integers(0, 10)) == 0:
            span = length
    return length, span


def _depth(rng, length, size_regime, split, witness, first_test):
    maximum = min(14, (length - 1) // 2)
    if witness:
        high = min(maximum, 3 if size_regime == "classic" else 8)
        return int(rng.integers(3, high + 1))
    if size_regime == "classic" and split == "train":
        maximum = min(maximum, 3)
    if size_regime == "shallow_test" and split == "test":
        maximum = min(maximum, 2)
    if size_regime == "classic" and split == "test" and first_test:
        return int(rng.integers(4, maximum + 1))
    choice = int(rng.integers(0, 10))
    if choice < 2:
        return 1
    if choice < 4:
        return maximum
    if choice < 6:
        return int(rng.integers(1, min(maximum, 3) + 1))
    return int(rng.integers(1, maximum + 1))


def _stripe_word(rng, depth, background, witness):
    available = np.asarray([c for c in range(10) if c != background], dtype=np.int8)
    if witness:
        a, b = (int(v) for v in rng.choice(available, size=2, replace=False))
        splits = [n for n in range(1, depth) if n != depth - n]
        cut = splits[int(rng.integers(len(splits)))]
        return (a,) * cut + (b,) * (depth - cut)

    mode = int(rng.integers(0, 6))
    if depth == 1 or mode == 0:
        color = int(rng.choice(available))
        return (color,) * depth
    if mode == 1 and depth <= 9:
        return tuple(int(v) for v in rng.choice(available, size=depth, replace=False))
    if mode == 2:
        a, b = (int(v) for v in rng.choice(available, size=2, replace=False))
        return tuple((a, b)[i % 2] for i in range(depth))
    if mode == 3 and depth >= 3:
        a, b = (int(v) for v in rng.choice(available, size=2, replace=False))
        return (a, b, a) + tuple(int(v) for v in rng.choice((a, b), size=depth - 3))
    if mode == 4:
        run_count = int(rng.integers(2, min(depth, 6) + 1))
        cuts = sorted(int(v) for v in rng.choice(np.arange(1, depth),
                                                  size=run_count - 1, replace=False))
        lengths = np.diff((0, *cuts, depth))
        colors = []
        for _ in lengths:
            candidates = available if not colors else available[available != colors[-1]]
            colors.append(int(rng.choice(candidates)))
        return tuple(c for c, n in zip(colors, lengths) for _ in range(int(n)))
    palette_size = int(rng.integers(2, min(6, depth, 9) + 1))
    palette = rng.choice(available, size=palette_size, replace=False)
    return tuple(int(v) for v in rng.choice(palette, size=depth))


def _build_pair(side, length, span, background, sequence):
    canonical = np.full((length, span), background, dtype=np.int8)
    depth = len(sequence)
    canonical[:depth] = np.asarray(sequence, dtype=np.int8)[:, None]
    reflected = canonical.copy()
    reflected[-depth:] = canonical[:depth][::-1]
    rotation = _ROTATIONS[side]
    return {"input": np.rot90(canonical, rotation).copy(),
            "output": np.rot90(reflected, rotation).copy()}


def _sample_pair(rng, side, size_regime, split, witness, first_test,
                 background, seen):
    last = None
    for _ in range(32):
        length, span = _dimensions(rng, size_regime, split, witness)
        depth = _depth(rng, length, size_regime, split, witness, first_test)
        sequence = _stripe_word(rng, depth, background, witness)
        pair = _build_pair(side, length, span, background, sequence)
        key = _grid_key(pair["input"])
        if key not in seen and not _is_source_input(pair["input"]):
            seen.add(key)
            return pair
        last = sequence

    # Exhaustive, bounded geometry fallback preserves the chosen regime.
    length_range, span_range = _ranges(size_regime, split)
    for length in range(max(length_range[0], 2 * len(last) + 1), length_range[1] + 1):
        for span in range(span_range[0], span_range[1] + 1):
            pair = _build_pair(side, length, span, background, last)
            key = _grid_key(pair["input"])
            if key not in seen and not _is_source_input(pair["input"]):
                seen.add(key)
                return pair
    raise RuntimeError("no unique pair available")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    token = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(token, num_train, num_test)
    rng = np.random.default_rng(token)
    size_regime = ("classic", "overlap", "narrow_test", "shallow_test", "broad")[
        int(rng.integers(0, 5))]
    train_sides, test_sides = _orientation_plan(rng, train_count, test_count)

    witness_indices = set()
    for side in dict.fromkeys(train_sides):
        positions = [i for i, value in enumerate(train_sides) if value == side]
        witness_indices.add(positions[int(rng.integers(len(positions)))])

    # Each pair has its own visible colors. The second training background
    # differs deliberately so no color can act as a puzzle-wide marker.
    backgrounds = [int(rng.integers(0, 10))]
    other = [color for color in range(10) if color != backgrounds[0]]
    backgrounds.append(int(rng.choice(other)))
    backgrounds.extend(int(rng.integers(0, 10))
                       for _ in range(train_count + test_count - 2))

    seen = set()
    train = []
    test = []
    for index, side in enumerate(train_sides):
        train.append(_sample_pair(rng, side, size_regime, "train",
                                  index in witness_indices, False,
                                  backgrounds[index], seen))
    for index, side in enumerate(test_sides):
        test.append(_sample_pair(rng, side, size_regime, "test", False,
                                 index == 0, backgrounds[train_count + index], seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_input(grid):
    candidates = []
    for side, rotation in _ROTATIONS.items():
        canonical = np.rot90(grid, -rotation)
        length, span = canonical.shape
        if length < 3 or span < 2 or not np.all(canonical == canonical[:, :1]):
            continue
        rows = [int(v) for v in canonical[:, 0]]
        background = rows[-1]
        if rows[0] == background:
            continue
        depth = next((i for i, v in enumerate(rows) if v == background), length)
        if 2 * depth >= length or any(v == background for v in rows[:depth]):
            continue
        if any(v != background for v in rows[depth:]):
            continue
        candidates.append({"side": side, "length": length, "span": span,
                           "depth": depth, "gap": length - 2 * depth,
                           "background": background, "sequence": tuple(rows[:depth])})
    assert len(candidates) == 1
    return candidates[0]


def _expected_by_coordinates(grid, info):
    expected = grid.copy()
    height, width = grid.shape
    background = info["background"]
    for row in range(height):
        for col in range(width):
            if int(grid[row, col]) != background:
                if info["side"] in ("top", "bottom"):
                    expected[height - row - 1, col] = grid[row, col]
                else:
                    expected[row, width - col - 1] = grid[row, col]
    return expected


def _is_witness(sequence):
    if len(set(sequence)) < 2 or sequence == sequence[::-1]:
        return False
    runs = []
    for color in sequence:
        if not runs or runs[-1][0] != color:
            runs.append([color, 1])
        else:
            runs[-1][1] += 1
    return len(runs) >= 2 and len(set(run[1] for run in runs)) >= 2


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert not _is_source_input(inp)
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analysis = _inspect_input(inp)
            assert np.array_equal(out, _expected_by_coordinates(inp, analysis))
            info[split].append(analysis)

    train_sides = {item["side"] for item in info["train"]}
    test_sides = {item["side"] for item in info["test"]}
    assert len({item["background"] for item in info["train"]}) >= 2
    for side in train_sides:
        assert any(item["side"] == side and _is_witness(item["sequence"])
                   for item in info["train"])
    if not test_sides.issubset(train_sides):
        assert train_sides & {"top", "bottom"}
        assert train_sides & {"left", "right"}


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in automatic}) == 42
    assert _puzzle_key(generate_puzzle(19)) == _puzzle_key(generate_puzzle(19))
    assert _puzzle_key(generate_puzzle(123456789, 7, 3)) == _puzzle_key(
        generate_puzzle(123456789, 7, 3))
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3

    checked = list(automatic)
    for seed, ntrain, ntest in ((7001, 2, 1), (7002, 10, 1), (7003, 10, 4),
                                (7004, 6, 2), (7005, 9, 4),
                                (7101, None, 1), (7102, None, 4),
                                (7103, 2, None), (7104, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
        checked.append(puzzle)
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(8000, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {request}")

    orientations, backgrounds, dimensions, layouts, depths, gaps = (
        set() for _ in range(6))
    unseen_test_colors = familiar_test_colors = mixed_orientation = heldout = 0
    narrow_test = wide_test = overlapping_test = shallow_test = deep_test = 0
    single = mono = alternating = reused = all_thin = long_run = 0
    small = large = square = black_stripe = 0
    late_witness = 0
    first_test_geometry_by_background = {}
    for puzzle in checked:
        validate_generated_puzzle(puzzle)
        train_info = [_inspect_input(p["input"]) for p in puzzle["train"]]
        test_info = [_inspect_input(p["input"]) for p in puzzle["test"]]
        train_sides = {x["side"] for x in train_info}
        test_sides = {x["side"] for x in test_info}
        mixed_orientation += len(train_sides) > 1
        heldout += bool(test_sides - train_sides)
        late_witness += not _is_witness(train_info[0]["sequence"])
        train_colors = {v for x in train_info for v in x["sequence"]}
        test_colors = {v for x in test_info for v in x["sequence"]}
        unseen_test_colors += bool(test_colors - train_colors)
        familiar_test_colors += bool(test_colors & train_colors)
        train_spans = [x["span"] for x in train_info]
        train_depths = [x["depth"] for x in train_info]
        narrow_test += any(x["span"] < min(train_spans) for x in test_info)
        wide_test += any(x["span"] > max(train_spans) for x in test_info)
        overlapping_test += any(min(train_spans) <= x["span"] <= max(train_spans)
                                for x in test_info)
        shallow_test += any(x["depth"] < min(train_depths) for x in test_info)
        deep_test += any(x["depth"] > max(train_depths) for x in test_info)
        first = test_info[0]
        first_test_geometry_by_background.setdefault(first["background"], set()).add(
            (first["span"], first["depth"]))
        demonstrations = [(pair["input"].copy(), pair["output"].copy())
                          for pair in puzzle["train"]]
        assert all(np.array_equal(_expected_by_coordinates(inp, _inspect_input(inp)), out)
                   for inp, out in demonstrations)
        for pair in puzzle["test"]:
            test_input = pair["input"].copy()
            prediction = _expected_by_coordinates(test_input, _inspect_input(test_input))
            assert np.array_equal(prediction, pair["output"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _inspect_input(pair["input"])
                # This check uses cell coordinates, independent of the
                # canonical-grid construction used for ground truth.
                prediction = _expected_by_coordinates(pair["input"], analysis)
                assert np.array_equal(prediction, pair["output"])
                orientations.add(analysis["side"])
                backgrounds.add(analysis["background"])
                dimensions.add(tuple(pair["input"].shape))
                depths.add(analysis["depth"])
                gaps.add(analysis["gap"])
                word = analysis["sequence"]
                labels = {}
                signature = tuple(labels.setdefault(v, len(labels)) for v in word)
                layouts.add((analysis["side"], analysis["length"],
                             analysis["span"], signature))
                single += len(word) == 1
                mono += len(set(word)) == 1
                alternating += len(word) >= 4 and len(set(word)) == 2 and all(
                    word[i] == word[i - 2] for i in range(2, len(word)))
                reused += len(word) >= 3 and len(set(word)) > 1 and any(
                    word[i] == word[j] for i in range(len(word))
                    for j in range(i + 2, len(word)))
                run_lengths = []
                for color in word:
                    if not run_lengths or run_lengths[-1][0] != color:
                        run_lengths.append([color, 1])
                    else:
                        run_lengths[-1][1] += 1
                all_thin += len(word) >= 3 and len(run_lengths) == len(word)
                long_run += any(length >= 4 for _, length in run_lengths)
                small += analysis["length"] <= 6
                large += analysis["length"] >= 25
                square += pair["input"].shape[0] == pair["input"].shape[1]
                black_stripe += 0 in word and analysis["background"] != 0
    assert orientations == set(_SIDES) and backgrounds == set(range(10))
    assert len(dimensions) >= 60 and len(layouts) >= 100
    assert max(depths) >= 10 and min(gaps) == 1
    assert min(single, mono, alternating, reused) > 0
    assert min(all_thin, long_run, small, large, square, black_stripe) > 0
    assert min(unseen_test_colors, familiar_test_colors) >= 5
    assert min(mixed_orientation, heldout) >= 3
    assert min(narrow_test, wide_test, overlapping_test, shallow_test, deep_test) >= 3
    assert late_witness >= 3
    assert sum(len(values) > 1 for values in first_test_geometry_by_background.values()) >= 5
