import numpy as np


# Exact source grids are retained solely to reject accidental copies.
_SOURCE_ROWS = (
    (("00303302222000", "30003300222202", "33030002020202", "33300032200000"),
     ("0000005", "0000000", "0000050", "0005550")),
    (("33300302000222", "03333032000002", "00303330022022", "00030000020222"),
     ("0005000", "0000050", "5500000", "5500000")),
    (("03303032020002", "30300000000220", "03300330000202", "03303030222020"),
     ("0005050", "0505005", "5005000", "5000000")),
    (("00333332220000", "30000302000000", "30033000022220", "00000330002202"),
     ("0000000", "0555505", "0500005", "5550000")),
    (("30300330022022", "30003302000020", "03030300222022", "03003330200022"),
     ("0500500", "0555005", "5000500", "5055000")),
)


def _signature(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _pair_signature(pair):
    return (_signature(pair["input"]), _signature(pair["output"]))


_SOURCE_SIGNATURES = frozenset(
    (
        ((len(a), len(a[0])), tuple(int(c) for row in a for c in row)),
        ((len(b), len(b[0])), tuple(int(c) for row in b for c in row)),
    )
    for a, b in _SOURCE_ROWS
)


def _optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    num_train = _optional_count(num_train, "num_train", 2, 10)
    num_test = _optional_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        # Give concise, source-scale puzzles appreciable probability.
        num_train = int(rng.choice(np.arange(2, 11), p=[.15, .18, .20, .13, .10, .08, .06, .05, .05]))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        choices = np.arange(max(2, num_test + 1), 11)
        weights = 1 / np.sqrt(choices - num_test)
        num_train = int(rng.choice(choices, p=weights / weights.sum()))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _mask(rng, height, width, density, style=None):
    """Sample an exact-density binary mask from several geometric score fields."""
    if style is None:
        style = int(rng.integers(0, 7))
    rows, cols = np.indices((height, width))
    n = height * width
    if style == 0:  # independent cells
        score = rng.random((height, width))
    elif style == 1:  # separated small clusters
        score = np.zeros((height, width), dtype=float)
        for _ in range(int(rng.integers(2, 8))):
            cy = float(rng.uniform(0, height))
            cx = float(rng.uniform(0, width))
            ry = float(rng.uniform(.6, max(1.4, height / 3)))
            rx = float(rng.uniform(.6, max(1.4, width / 3)))
            score = np.maximum(score, np.exp(-((rows-cy)/ry)**2 - ((cols-cx)/rx)**2))
        score += rng.normal(0, .12, (height, width))
    elif style == 2:  # clean or broken stripes
        period = int(rng.integers(2, min(7, max(height, width) + 1)))
        slope = float(rng.choice([-1, 0, 0, 1]))
        axis = rows if bool(rng.integers(2)) else cols
        cross = cols if axis is rows else rows
        score = np.cos(2 * np.pi * (axis + slope * cross / 3 + rng.uniform(0, period)) / period)
        score += rng.normal(0, float(rng.uniform(.03, .42)), (height, width))
    elif style == 3:  # mixed sparse and dense patches
        gh = int(rng.integers(2, min(6, height + 1)))
        gw = int(rng.integers(2, min(6, width + 1)))
        coarse = rng.random((gh, gw))
        score = coarse[(rows * gh // height), (cols * gw // width)]
        score += rng.normal(0, .14, (height, width))
    elif style == 4:  # jagged edge or wedge
        slope = float(rng.uniform(-1.5, 1.5))
        edge = float(rng.uniform(-width / 3, width * 4 / 3))
        jitter = rng.normal(0, max(.4, width / 8), height)
        score = edge + slope * (rows - height / 2) + jitter[rows] - cols
        score += rng.normal(0, .4, (height, width))
    elif style == 5:  # ring, hollow block, or perforated patch
        cy = float(rng.uniform(0, height - 1))
        cx = float(rng.uniform(0, width - 1))
        ry = float(rng.uniform(1, max(1.5, height / 2)))
        rx = float(rng.uniform(1, max(1.5, width / 2)))
        radius = np.sqrt(((rows-cy)/ry)**2 + ((cols-cx)/rx)**2)
        score = -abs(radius - float(rng.uniform(.5, 1.2)))
        score += rng.normal(0, .12, (height, width))
    else:  # offset irregular near matches can retain local islands
        score = rng.random((height, width))
        for _ in range(2):
            score = (score + np.roll(score, 1, axis=int(rng.integers(2)))) / 2
        score += rng.normal(0, .12, (height, width))
    score += rng.random((height, width)) * 1e-6
    k = int(np.clip(round(float(density) * n), 0, n))
    result = np.zeros(n, dtype=bool)
    if k:
        result[np.argpartition(score.ravel(), -k)[-k:]] = True
    return result.reshape(height, width)


def _make_masks(rng, height, width, relation):
    n = height * width
    if relation in ("four_state", "independent"):
        left = _mask(rng, height, width, rng.uniform(.18, .82))
        right = _mask(rng, height, width, rng.uniform(.18, .82))
        if relation == "four_state":
            # A diagnostic training example has the complete NOR truth table.
            states = left.astype(np.int8) + 2 * right.astype(np.int8)
            minimum = 2 if n >= 8 else 1
            for state in range(4):
                while int(np.count_nonzero(states == state)) < minimum:
                    counts = np.bincount(states.ravel(), minlength=4)
                    donor = int(np.argmax(counts))
                    assert counts[donor] > minimum
                    choices = np.flatnonzero(states.ravel() == donor)
                    states.ravel()[int(rng.choice(choices))] = state
            left = (states & 1) != 0
            right = (states & 2) != 0
    elif relation == "disjoint":
        left = _mask(rng, height, width, rng.uniform(.18, .60))
        right = _mask(rng, height, width, rng.uniform(.18, .72)) & ~left
        if not right.any():
            right.flat[int(rng.choice(np.flatnonzero(~left)))] = True
    elif relation == "contained":
        outer = _mask(rng, height, width, rng.uniform(.25, .85))
        inner = outer & _mask(rng, height, width, rng.uniform(.25, .80))
        if not inner.any():
            inner.flat[int(rng.choice(np.flatnonzero(outer)))] = True
        left, right = (inner, outer) if bool(rng.integers(2)) else (outer, inner)
    elif relation == "identical":
        left = _mask(rng, height, width, rng.uniform(.08, .90))
        right = left.copy()
    elif relation == "near_match":
        left = _mask(rng, height, width, rng.uniform(.15, .85))
        if bool(rng.integers(2)):
            right = np.roll(left, int(rng.choice([-1, 1])), axis=int(rng.integers(2)))
        else:
            right = left.copy()
        edits = rng.choice(n, size=max(1, int(round(n * rng.uniform(.03, .17)))), replace=False)
        right.flat[edits] = ~right.flat[edits]
    elif relation == "empty":
        left = np.zeros((height, width), dtype=bool)
        right = _mask(rng, height, width, rng.uniform(.08, 1.0))
        if bool(rng.integers(2)):
            left, right = right, left
    elif relation == "full":
        left = np.ones((height, width), dtype=bool)
        right = _mask(rng, height, width, rng.uniform(.05, .88))
        if bool(rng.integers(2)):
            left, right = right, left
    elif relation == "both_empty":
        left = np.zeros((height, width), dtype=bool)
        right = left.copy()
    elif relation == "complementary":
        left = _mask(rng, height, width, rng.uniform(.15, .85))
        right = ~left
    else:
        raise ValueError("unknown mask relationship")
    return left, right


_RELATIONS = ("four_state", "independent", "disjoint", "contained",
              "identical", "near_match", "empty", "full", "both_empty", "complementary")
_TRAIN_WEIGHTS = np.array([14, 30, 10, 10, 8, 10, 6, 5, 3, 4], dtype=float)
_TEST_WEIGHTS = np.array([8, 25, 12, 11, 10, 13, 7, 6, 4, 4], dtype=float)


def _pair(rng, height, width, background, left_color, right_color, marker, relation):
    left_mask, right_mask = _make_masks(rng, height, width, relation)
    left = np.where(left_mask, left_color, background)
    right = np.where(right_mask, right_color, background)
    return {
        "input": np.concatenate((left, right), axis=1).astype(np.int8),
        "output": np.where(~(left_mask | right_mask), marker, background).astype(np.int8),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    background_count = int(rng.integers(2, min(4, num_train) + 1))
    foreground_count = int(rng.integers(2, min(4, 9 - background_count, 2 + 2 * (num_train - 2)) + 1))
    colors = [int(c) for c in rng.permutation(10)]
    marker = colors[0]
    backgrounds = colors[1:1 + background_count]
    foregrounds = colors[1 + background_count:1 + background_count + foreground_count]

    # Training exposes every background and foreground in the puzzle palette.
    train_backgrounds = backgrounds + [int(rng.choice(backgrounds)) for _ in range(num_train - background_count)]
    rng.shuffle(train_backgrounds)
    train_foregrounds = [(foregrounds[0], foregrounds[1]),
                         (foregrounds[1], foregrounds[0])]
    for _ in range(2, num_train):
        train_foregrounds.append(tuple(int(x) for x in rng.choice(foregrounds, size=2)))
    for index, color in enumerate(foregrounds[2:]):
        pair_index = 2 + index // 2
        side = index % 2
        pair_colors = list(train_foregrounds[pair_index])
        pair_colors[side] = color
        train_foregrounds[pair_index] = tuple(pair_colors)

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    used_shapes = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.bit_generator.random_raw()))
            if split == "train" and index == 0:
                height = int(pair_rng.integers(4, 15))
                width = int(pair_rng.integers(3, 14))
                relation = "four_state"
            else:
                height = int(pair_rng.integers(3, 19))
                width = int(pair_rng.integers(2, 16))
                if len(used_shapes) == 1 and (height, width) in used_shapes:
                    width = 2 + (width - 1) % 14
                weights = _TRAIN_WEIGHTS if split == "train" else _TEST_WEIGHTS
                relation = str(pair_rng.choice(_RELATIONS, p=weights / weights.sum()))
                if split == "train" and 2 <= index < 2 + (foreground_count - 1) // 2:
                    relation = "four_state"
            background = train_backgrounds[index] if split == "train" else int(pair_rng.choice(backgrounds))
            left_color, right_color = (train_foregrounds[index] if split == "train"
                                       else tuple(int(x) for x in pair_rng.choice(foregrounds, size=2)))
            for _ in range(12):
                pair = _pair(pair_rng, height, width, background, left_color, right_color, marker, relation)
                signature = _signature(pair["input"])
                if signature not in seen_inputs and _pair_signature(pair) not in _SOURCE_SIGNATURES:
                    break
                relation = "independent"
                height = 3 + (height - 2) % 16
            else:
                raise RuntimeError("unable to construct a distinct pair")
            seen_inputs.add(signature)
            used_shapes.add((height, width))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _states(left, right, background):
    return (left != background).astype(np.int8) + 2 * (right != background).astype(np.int8)


def _check_rule(pair, background, marker):
    source = pair["input"]
    target = pair["output"]
    height, total_width = source.shape
    assert total_width % 2 == 0
    width = total_width // 2
    assert target.shape == (height, width)
    left, right = source[:, :width], source[:, width:]
    assert np.count_nonzero(source == background) >= 1
    assert marker not in source
    assert np.unique(left[left != background]).size <= 1
    assert np.unique(right[right != background]).size <= 1
    assert set(int(v) for v in np.unique(target)) <= {background, marker}
    assert np.array_equal(target == marker, (left == background) & (right == background))
    assert np.array_equal(target == background, (left != background) | (right != background))
    return _states(left, right, background)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, shapes = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            _check_grid(pair["input"])
            _check_grid(pair["output"])
            inp = _signature(pair["input"])
            sig = _pair_signature(pair)
            assert inp not in seen_inputs and sig not in seen_pairs
            assert sig not in _SOURCE_SIGNATURES
            seen_inputs.add(inp)
            seen_pairs.add(sig)
            shapes.add(pair["output"].shape)
    assert len(shapes) >= 2

    # The first training pair identifies the background by its unique presence
    # in both panels, and consequently identifies the stable result color.
    first = puzzle["train"][0]
    width = first["output"].shape[1]
    assert first["input"].shape[1] == 2 * width
    left_colors = set(int(x) for x in np.unique(first["input"][:, :width]))
    right_colors = set(int(x) for x in np.unique(first["input"][:, width:]))
    assert len(left_colors) == len(right_colors) == 2
    shared = left_colors & right_colors
    assert len(shared) == 1
    first_background = next(iter(shared))
    output_colors = set(int(x) for x in np.unique(first["output"]))
    assert first_background in output_colors and len(output_colors) == 2
    marker = next(iter(output_colors - {first_background}))
    assert marker not in left_colors | right_colors
    assert set(int(s) for s in np.unique(_check_rule(first, first_background, marker))) == {0, 1, 2, 3}

    backgrounds, foregrounds = set(), set()
    demonstrated_states = set()
    for pair in puzzle["train"]:
        nonmarker_output = set(int(v) for v in np.unique(pair["output"])) - {marker}
        if nonmarker_output:
            assert len(nonmarker_output) == 1
            background = next(iter(nonmarker_output))
        else:
            input_colors = set(int(v) for v in np.unique(pair["input"]))
            assert len(input_colors) == 1
            background = next(iter(input_colors))
        states = _check_rule(pair, background, marker)
        backgrounds.add(background)
        foregrounds.update(int(v) for v in np.unique(pair["input"]) if int(v) != background)
        demonstrated_states.update(int(s) for s in np.unique(states))
    assert demonstrated_states == {0, 1, 2, 3}
    assert len(backgrounds) >= 2 and len(foregrounds) >= 2
    assert backgrounds.isdisjoint(foregrounds) and marker not in backgrounds | foregrounds

    for pair in puzzle["test"]:
        input_colors = set(int(v) for v in np.unique(pair["input"]))
        candidates = input_colors & backgrounds
        assert len(candidates) == 1
        background = next(iter(candidates))
        assert input_colors - {background} <= foregrounds
        _check_rule(pair, background, marker)


def _puzzles_equal(first, second):
    return all(
        np.array_equal(a[key], b[key])
        for split in ("train", "test")
        for a, b in zip(first[split], second[split])
        for key in ("input", "output")
    ) and all(len(first[s]) == len(second[s]) for s in ("train", "test"))


def _component_count(mask):
    remaining = set(zip(*np.nonzero(mask)))
    count = 0
    while remaining:
        count += 1
        frontier = [remaining.pop()]
        while frontier:
            row, col = frontier.pop()
            for point in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                if point in remaining:
                    remaining.remove(point)
                    frontier.append(point)
    return count


def _hole_count(mask):
    remaining = set(zip(*np.nonzero(~mask)))
    height, width = mask.shape
    count = 0
    while remaining:
        frontier = [remaining.pop()]
        touches_edge = False
        while frontier:
            row, col = frontier.pop()
            touches_edge |= row == 0 or col == 0 or row == height - 1 or col == width - 1
            for point in ((row-1, col), (row+1, col), (row, col-1), (row, col+1)):
                if point in remaining:
                    remaining.remove(point)
                    frontier.append(point)
        count += not touches_edge
    return count


def self_test() -> None:
    train_counts, test_counts = set(), set()
    all_shapes, all_layouts, all_palettes, puzzle_signatures = set(), set(), set(), set()
    relationships, components = set(), []
    test_comparisons = {"lower": 0, "similar": 0, "higher": 0}
    observed = {name: 0 for name in (
        "disjoint", "contained", "identical", "near_match", "one_empty",
        "one_full", "same_foreground", "both_exclusive_substantial", "tall_narrow",
        "test_in_train_range", "test_not_all_higher", "panel_holes",
    )}
    varying_shapes = varying_backgrounds = varying_foregrounds = 0
    all_marker = all_background = 0
    short_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        short_count += len(puzzle["train"]) <= 4
        puzzle_signatures.add(tuple(_pair_signature(p) for split in ("train", "test") for p in puzzle[split]))
        local_shapes, local_backgrounds, local_foregrounds = set(), set(), set()
        train_fractions, test_fractions = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                assert target.shape == (source.shape[0], source.shape[1] // 2)
                width = target.shape[1]
                left, right = source[:, :width], source[:, width:]
                if split == "train":
                    output_nonmarker = set(int(x) for x in np.unique(target))
                    # The stable marker is the non-input color when present.
                    novel = output_nonmarker - set(int(x) for x in np.unique(source))
                    if novel:
                        marker = next(iter(novel))
                backgrounds = set(int(x) for x in np.unique(source)) & set(int(x) for x in np.unique(target))
                if not backgrounds:  # both panels empty, output is entirely marker
                    backgrounds = set(int(x) for x in np.unique(source))
                assert len(backgrounds) == 1
                background = next(iter(backgrounds))
                expected = (left == background) & (right == background)
                assert np.array_equal(target == marker, expected)
                assert np.array_equal(target != marker, ~expected)
                states = _states(left, right, background)
                present = frozenset(int(x) for x in np.unique(states))
                relationships.add(present)
                left_on, right_on = left != background, right != background
                observed["disjoint"] += 3 not in present and left_on.any() and right_on.any()
                observed["contained"] += 3 in present and ((1 in present) != (2 in present))
                observed["identical"] += np.array_equal(left_on, right_on) and left_on.any() and (~left_on).any()
                observed["near_match"] += 0 < np.mean(left_on != right_on) <= .17 and left_on.any() and right_on.any()
                observed["one_empty"] += (not left_on.any()) != (not right_on.any())
                observed["one_full"] += left_on.all() != right_on.all()
                observed["same_foreground"] += (
                    len(set(int(x) for x in np.unique(left)) - {background}) == 1
                    and (set(int(x) for x in np.unique(left)) - {background})
                    == (set(int(x) for x in np.unique(right)) - {background})
                )
                observed["both_exclusive_substantial"] += (
                    np.mean(states == 1) >= .15 and np.mean(states == 2) >= .15
                )
                observed["tall_narrow"] += target.shape[0] >= 12 and width <= 5
                observed["panel_holes"] += (_hole_count(left_on) > 0) + (_hole_count(right_on) > 0)
                components.append(_component_count(expected))
                fraction = float(expected.mean())
                (train_fractions if split == "train" else test_fractions).append(fraction)
                all_marker += bool(expected.all())
                all_background += bool(not expected.any())
                local_shapes.add(target.shape)
                all_shapes.add(target.shape)
                local_backgrounds.add(background)
                local_foregrounds.add((
                    tuple(sorted(set(int(x) for x in np.unique(left)) - {background})),
                    tuple(sorted(set(int(x) for x in np.unique(right)) - {background})),
                ))
                all_palettes.add((background, tuple(sorted(set(int(x) for x in np.unique(source)) - {background})), marker))
                all_layouts.add((states.shape, tuple(int(x) for x in states.flat)))
        varying_shapes += len(local_shapes) >= 2
        varying_backgrounds += len(local_backgrounds) >= 2
        varying_foregrounds += len(local_foregrounds) >= 2
        middle = float(np.median(train_fractions))
        for fraction in test_fractions:
            category = "lower" if fraction < middle - .08 else "higher" if fraction > middle + .08 else "similar"
            test_comparisons[category] += 1
        observed["test_in_train_range"] += any(
            min(train_fractions) <= x <= max(train_fractions) for x in test_fractions
        )
        observed["test_not_all_higher"] += not all(
            x > max(train_fractions) for x in test_fractions
        )
    assert len(train_counts) >= 4 and len(test_counts) >= 3 and short_count >= 12
    assert len(all_shapes) >= 45 and len(all_layouts) >= 100 and len(all_palettes) >= 80
    assert len(puzzle_signatures) == 42
    assert varying_shapes == varying_backgrounds == 42 and varying_foregrounds >= 30
    assert {frozenset((0, 1, 2, 3)), frozenset((0, 3)), frozenset((0, 1, 2)), frozenset((0,))} <= relationships
    assert max(components) >= 5 and sum(c >= 2 for c in components) >= 80
    assert all_marker >= 1 and all_background >= 1
    assert min(test_comparisons.values()) >= 8
    assert observed["test_in_train_range"] >= 10
    assert observed["test_not_all_higher"] >= 20
    for name in ("disjoint", "contained", "identical", "near_match",
                 "one_empty", "one_full", "same_foreground", "tall_narrow"):
        assert observed[name] >= 5
    assert observed["both_exclusive_substantial"] >= 30
    assert observed["panel_holes"] >= 30

    for seed, train, test in ((5001, 2, 1), (5002, 10, 1), (5003, 10, 4),
                              (6001, None, 1), (6002, None, 4),
                              (6003, 2, None), (6004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        assert _puzzles_equal(puzzle, generate_puzzle(seed, train, test))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7001, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts were accepted")
