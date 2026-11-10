import numpy as np


_SOURCE_SPECS = ((4, 8, 6), (3, 9, 6), (6, 2, 6))


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return value


def _optional_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _optional_count(num_train, "num_train", 2, 10)
    test = _optional_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        train = int(rng.choice(np.arange(2, 11), p=(0.18, 0.18, 0.16, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05)))
        test = int(rng.integers(1, min(4, train - 1) + 1))
        return train, test
    if train is None:
        choices = list(range(test + 1, 11))
        train = choices[int(rng.integers(len(choices)))]
    if test is None:
        choices = list(range(1, min(4, train - 1) + 1))
        test = choices[int(rng.integers(len(choices)))]
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _checkerboard_output(input_grid):
    top, bottom = int(input_grid[0, 0]), int(input_grid[1, 0])
    width = input_grid.shape[1]
    output = np.empty((2, width), dtype=np.int8)
    output[0, 0::2], output[1, 0::2] = top, bottom
    output[0, 1::2], output[1, 1::2] = bottom, top
    return output


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(value) for value in grid.flat)


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _is_source_pair(top, bottom, width):
    return (top, bottom, width) in _SOURCE_SPECS


def _widths(rng, train_count, test_count):
    # Shared widths reproduce the source regime; other draws cover repeated,
    # shorter, equal, and longer widths without coupling width to color.
    anchor = 6 if rng.random() < 0.28 else int(rng.integers(3, 31))
    shared = bool(rng.random() < 0.22)
    if shared:
        return [anchor] * train_count, [anchor] * test_count
    radius = int(rng.integers(2, 13))
    low, high = max(2, anchor - radius), min(30, anchor + radius)
    train_widths = []
    for _ in range(train_count):
        draw = rng.random()
        if draw < 0.25:
            width = anchor
        elif draw < 0.47 and train_widths:
            width = int(rng.choice(train_widths))
        else:
            width = int(rng.integers(low, high + 1))
        train_widths.append(width)
    if max(train_widths) < 4:
        train_widths[-1] = 4
    test_widths = []
    for _ in range(test_count):
        draw = rng.random()
        if draw < 0.32:
            width = int(rng.choice(train_widths))
        elif draw < 0.55 and max(train_widths) > 2:
            width = int(rng.integers(2, max(train_widths)))
        elif draw < 0.80 and max(train_widths) < 30:
            width = int(rng.integers(max(train_widths) + 1, 31))
        else:
            width = int(rng.integers(low, high + 1))
        test_widths.append(width)
    return train_widths, test_widths


def _palette(rng, train_count, total_count, shared_width):
    minimum = 2
    if shared_width:
        while minimum * (minimum - 1) < total_count + 3:
            minimum += 1
    draw = rng.random()
    if draw < 0.25:
        size = minimum
    elif draw < 0.55:
        size = min(10, minimum + 1)
    else:
        size = int(rng.integers(minimum, 11))
    colors = [int(v) for v in rng.permutation(np.arange(10))[:size]]
    if 0 not in colors and rng.random() < 0.4:
        colors[-1] = 0
    colors = tuple(colors)
    holdout = 0
    if size >= 3 and rng.random() < 0.7:
        holdout = int(rng.integers(1, min(3, size - 2) + 1))
    while holdout and (size - holdout) * (size - holdout - 1) < train_count:
        holdout -= 1
    return colors[:size - holdout], colors


def _choose_colors(rng, palette, previous, used_training_colors, is_test):
    pairs = [(a, b) for a in palette for b in palette if a < b]
    if is_test and rng.random() < 0.55:
        fresh = [p for p in pairs if set(p) - used_training_colors]
        if fresh:
            pairs = fresh
    elif previous is not None:
        draw = rng.random()
        if draw < 0.32 and tuple(sorted(previous)) in pairs:
            pairs = [tuple(sorted(previous))]
        elif draw < 0.68:
            shared = [p for p in pairs if len(set(p) & set(previous)) == 1]
            if shared:
                pairs = shared
        elif draw < 0.82:
            disjoint = [p for p in pairs if not set(p) & set(previous)]
            if disjoint:
                pairs = disjoint
    first, second = pairs[int(rng.integers(len(pairs)))]
    return (first, second) if rng.random() < 0.5 else (second, first)


def _make_pair(rng, target_width, preferred, palette, required_order, used_inputs):
    other_pairs = [(a, b) for a in palette for b in palette if a != b]
    if required_order is not None:
        other_pairs = [p for p in other_pairs if (p[0] < p[1]) == required_order]
    rng.shuffle(other_pairs)
    ordered_pairs = [preferred] + [p for p in other_pairs if p != preferred]
    other_widths = [w for w in range(3, 31) if w != target_width]
    rng.shuffle(other_widths)
    other_widths.sort(key=lambda w: abs(w - target_width))
    for width in [target_width] + other_widths:
        for top, bottom in ordered_pairs:
            if _is_source_pair(top, bottom, width):
                continue
            input_grid = np.empty((2, width), dtype=np.int8)
            input_grid[0, :], input_grid[1, :] = top, bottom
            if _grid_key(input_grid) in used_inputs:
                continue
            return {"input": input_grid, "output": _checkerboard_output(input_grid)}
    raise RuntimeError("no unique pair could be constructed")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    train_widths, test_widths = _widths(rng, train_count, test_count)
    train_palette, all_palette = _palette(
        rng, train_count, train_count + test_count,
        len(set(train_widths + test_widths)) == 1,
    )
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_training_colors = set()
    first_order = bool(rng.integers(2))
    first_pair = None
    previous = None
    for index, width in enumerate(train_widths):
        if index == 1 and rng.random() < 0.52:
            preferred = first_pair[::-1]
        else:
            preferred = _choose_colors(rng, train_palette, previous, used_training_colors, False)
        required_order = first_order if index == 0 else (not first_order if index == 1 else None)
        if required_order is not None and (preferred[0] < preferred[1]) != required_order:
            preferred = preferred[::-1]
        pair = _make_pair(rng, width, preferred, train_palette, required_order, used_inputs)
        puzzle["train"].append(pair)
        used_inputs.add(_grid_key(pair["input"]))
        previous = int(pair["input"][0, 0]), int(pair["input"][1, 0])
        if index == 0:
            first_pair = previous
        used_training_colors.update(previous)

    # Occasionally combine the supported shifts in one query. The extra
    # color is visible in the test input, so it introduces no hidden map.
    combined_preference = None
    extra_color = [c for c in range(10) if c not in used_training_colors]
    odd_longer = [w for w in range(max(train_widths) + 1, 31) if w % 2]
    if (len(set(train_widths + test_widths)) > 1 and extra_color
            and odd_longer and rng.random() < 0.25):
        test_widths[0] = int(rng.choice(odd_longer))
        fresh = int(rng.choice(extra_color))
        companion = int(rng.choice([c for c in all_palette if c != fresh]))
        combined_preference = (fresh, companion) if rng.random() < 0.5 else (companion, fresh)
        if fresh not in all_palette:
            all_palette = all_palette + (fresh,)

    for index, width in enumerate(test_widths):
        preferred = (combined_preference if index == 0 and combined_preference is not None
                     else _choose_colors(rng, all_palette, previous, used_training_colors, True))
        pair = _make_pair(rng, width, preferred, all_palette, None, used_inputs)
        puzzle["test"].append(pair)
        used_inputs.add(_grid_key(pair["input"]))
        previous = int(pair["input"][0, 0]), int(pair["input"][1, 0])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs = set(), set()
    train_orders, train_widths = set(), []
    saw_width_two = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            assert inp.shape[0] == 2 and inp.shape[1] >= 2 and out.shape == inp.shape
            top, bottom, width = int(inp[0, 0]), int(inp[1, 0]), inp.shape[1]
            saw_width_two |= width == 2
            assert top != bottom
            assert np.all(inp[0] == top) and np.all(inp[1] == bottom)
            assert np.array_equal(out, _checkerboard_output(inp))
            assert np.array_equal(out[:, 0], inp[:, 0])
            assert not np.array_equal(inp, out)
            assert not _is_source_pair(top, bottom, width)
            pair_key = _pair_key(pair)
            assert pair_key[0] not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(pair_key[0])
            seen_pairs.add(pair_key)
            if split == "train":
                train_orders.add(top < bottom)
                train_widths.append(width)
    assert train_orders == {False, True}, "training must demonstrate both color orders"
    if saw_width_two:
        assert max(train_widths) >= 4, "a width-2 training example needs a longer demonstration"


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split]) for split in ("train", "test"))


def _predict_from_examples(train, test_input):
    # This check sees only demonstrations and the query input. It expresses
    # the pattern with cell parity rather than the generator's column slices.
    for pair in train:
        inp, out = pair["input"], pair["output"]
        rows, columns = np.indices(inp.shape)
        predicted = np.where((rows + columns) % 2 == 0, inp[0, 0], inp[1, 0])
        assert np.array_equal(out, predicted)
    rows, columns = np.indices(test_input.shape)
    return np.where((rows + columns) % 2 == 0, test_input[0, 0], test_input[1, 0])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    widths, layouts = set(), set()
    equal_width = shorter = longer = reused_pair = shared_color = fresh_test = 0
    zero_top = zero_bottom = odd = even = width_two = 0
    same_pair_different_width = same_width_few_colors = fresh_odd_longer = 0
    test_smaller_top = test_larger_top = small_training_set = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train, test = puzzle["train"], puzzle["test"]
        training_widths = [pair["input"].shape[1] for pair in train]
        test_widths = [pair["input"].shape[1] for pair in test]
        small_training_set += len(train) <= 4
        equal_width += bool(set(training_widths) & set(test_widths))
        shorter += any(w < max(training_widths) for w in test_widths)
        longer += any(w > max(training_widths) for w in test_widths)
        layouts.add((tuple(training_widths), tuple(test_widths)))
        colors_in_train = {int(v) for pair in train for v in pair["input"][:, 0]}
        fresh_test += any(set(int(v) for v in pair["input"][:, 0]) - colors_in_train for pair in test)
        fresh_odd_longer += any(
            pair["input"].shape[1] % 2
            and pair["input"].shape[1] > max(training_widths)
            and set(int(v) for v in pair["input"][:, 0]) - colors_in_train
            for pair in test
        )
        test_smaller_top += any(int(pair["input"][0, 0]) < int(pair["input"][1, 0]) for pair in test)
        test_larger_top += any(int(pair["input"][0, 0]) > int(pair["input"][1, 0]) for pair in test)
        unordered = [tuple(sorted(int(v) for v in pair["input"][:, 0])) for pair in train + test]
        reused_pair += len(set(unordered)) < len(unordered)
        shared_color += any(len(set(a) & set(b)) == 1 for a, b in zip(unordered, unordered[1:]))
        same_pair_different_width += any(
            unordered[i] == unordered[j]
            and (train + test)[i]["input"].shape[1] != (train + test)[j]["input"].shape[1]
            for i in range(len(unordered)) for j in range(i + 1, len(unordered))
        )
        same_width_few_colors += (
            len(set(training_widths + test_widths)) == 1
            and len({int(v) for pair in train + test for v in pair["input"][:, 0]}) <= 4
        )
        for pair in train + test:
            inp = pair["input"]
            width = inp.shape[1]
            widths.add(width)
            odd += width % 2
            even += width % 2 == 0
            width_two += width == 2
            zero_top += int(inp[0, 0]) == 0
            zero_bottom += int(inp[1, 0]) == 0
        for pair in test:
            assert np.array_equal(_predict_from_examples(train, pair["input"]), pair["output"])
    assert len(widths) >= 20 and len(layouts) >= 25
    assert equal_width >= 4 and shorter >= 4 and longer >= 4
    assert reused_pair >= 10 and shared_color >= 4 and fresh_test >= 4
    assert same_pair_different_width >= 10 and same_width_few_colors >= 2
    assert fresh_odd_longer >= 3 and small_training_set >= 10
    assert test_smaller_top >= 10 and test_larger_top >= 10
    assert zero_top >= 2 and zero_bottom >= 2 and odd >= 10 and even >= 10
    assert width_two >= 1
    for seed, train_count, test_count in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for seed, train_count, test_count in ((2000, None, 1), (2001, None, 4), (2002, 2, None), (2003, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3.0}, {"num_test": True}):
        try:
            generate_puzzle(4000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {arguments}")
    assert _puzzle_key(generate_puzzle(123456, 7, 3)) == _puzzle_key(generate_puzzle(123456, 7, 3))
    assert _puzzle_key(generate_puzzle(123456)) != _puzzle_key(generate_puzzle(123457))
