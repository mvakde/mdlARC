import numpy as np


# Exact source pairs are excluded by content, not by size or palette.
_SOURCE_ROWS = (
    (("777777777", "777777777", "777777777", "777777777", "277787777",
      "555555777", "555555777", "555555777", "555555777"),
     ("777777777", "777777777", "777777777", "777777777", "777727777",
      "555555777", "555555777", "555555777", "555555877")),
    (("7777777", "7727877", "7755577", "7755577", "7755577", "7755577", "7755577"),
     ("7777777", "7777277", "7755577", "7755577", "7755577", "7755577", "7755587")),
    (("77777777777", "77787727777", "77555555777", "77555555777",
      "77555555777", "77555555777", "77555555777", "77555555777",
      "77555555777", "77555555777", "77555555777"),
     ("77777777777", "77727777777", "77555555777", "77555555777",
      "77555555777", "77555555777", "77555555777", "77555555777",
      "77555555777", "77555555777", "78555555777")),
    (("7777777777777777777",) * 9 +
     ("7777777777877772777",) + ("7555555555555555557",) * 9,
     ("7777777777777777777",) * 9 +
     ("7777777777277777777",) + ("7555555555555555557",) * 8 +
     ("8555555555555555557",)),
)
_SOURCE_SIGNATURES = set(_SOURCE_ROWS)


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _source_copy(inp, out):
    return (_rows(inp), _rows(out)) in _SOURCE_SIGNATURES


def _key(grid):
    return grid.shape, grid.tobytes()


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _counts(seed, rng, train, test):
    train, test = _count(train, "num_train"), _count(test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be 2..10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be 1..4")
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + int(seed) % 4
        train = test + 1 + (int(seed) // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _sample_span(rng, train):
    # Span and depth use separate draws, so long lateral and short side legs combine.
    bands = ((5, 9), (10, 15), (16, 22)) if train else (
        (3, 9), (10, 17), (18, 26))
    probabilities = (0.52, 0.36, 0.12) if train else (0.48, 0.30, 0.22)
    low, high = bands[int(rng.choice(3, p=probabilities))]
    return int(rng.integers(low, high + 1))


def _sample_depth(rng, train):
    bands = ((2, 8), (9, 15), (16, 24)) if train else (
        (2, 8), (9, 17), (18, 26))
    probabilities = (0.53, 0.35, 0.12) if train else (0.48, 0.30, 0.22)
    low, high = bands[int(rng.choice(3, p=probabilities))]
    return int(rng.integers(low, high + 1))


def _make_pair(rng, train, colors, orientation, sign, exterior, core):
    background, block_color, pusher_color, token_color = colors
    reflect = bool(rng.random() < 0.5)
    if reflect != (orientation in ("right", "top")):
        sign = -sign
    span = _sample_span(rng, train)
    depth = _sample_depth(rng, train)
    if core:
        span = max(span, 5)

    # Canonical construction is bottom attached. The opposite lateral margin
    # may be zero, as in source train 1; the selected exit always has a lane.
    smallest_extra = max(0, 7 - span - 1)
    largest_extra = min(8, 30 - span - 1)
    extra = min(int(rng.integers(smallest_extra, largest_extra + 1)),
                int(rng.integers(smallest_extra, largest_extra + 1)))
    opposite = 0 if rng.random() < 0.32 else int(rng.integers(0, extra + 1))
    exit_extra = extra - opposite
    width = span + 1 + extra
    left = opposite if sign == 1 else 1 + exit_extra
    right = left + span - 1
    side = right + 1 if sign == 1 else left - 1

    smallest_front = max(0, 7 - depth - 1)
    largest_front = min(6, 30 - depth - 1)
    front = min(int(rng.integers(smallest_front, largest_front + 1)),
                int(rng.integers(smallest_front, largest_front + 1)))
    height = depth + 1 + front
    face = front
    block_top = front + 1

    if exterior:
        token_axis = side
        pusher_axis = int(rng.integers(left, right + 1))
    else:
        possible = [(p, t) for p in range(left, right + 1)
                    for t in range(left, right + 1)
                    if (t - p) * sign > 0
                    and (not core or abs(side - t) >= 3)]
        pusher_axis, token_axis = possible[int(rng.integers(len(possible)))]

    inp = np.full((height, width), background, dtype=np.int8)
    inp[block_top:, left:right + 1] = block_color
    inp[face, pusher_axis] = pusher_color
    inp[face, token_axis] = token_color
    out = inp.copy()
    out[face, pusher_axis] = background
    out[face, token_axis] = pusher_color
    out[height - 1, side] = token_color

    # Reflection changes handedness without changing the face-relative rule.
    if reflect:
        inp, out = np.fliplr(inp).copy(), np.fliplr(out).copy()
    rotations = {"bottom": 0, "right": 1, "top": 2, "left": 3}
    turns = rotations[orientation]
    if turns:
        inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
    return {"input": inp, "output": out}


def _assert_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= dimension <= 30 for dimension in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((grid >= 0) & (grid <= 9)))


def _analyze_pair(inp, out):
    _assert_grid(inp)
    _assert_grid(out)
    assert inp.shape == out.shape
    assert not np.array_equal(inp, out)
    height, width = inp.shape
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 4
    singletons = [int(c) for c, n in zip(colors, counts) if n == 1]
    large = [int(c) for c, n in zip(colors, counts) if n > 1]
    assert len(singletons) == len(large) == 2
    assert set(int(c) for c in np.unique(out)) == set(int(c) for c in colors)
    matches = []

    for block_color in large:
        cells = np.argwhere(inp == block_color)
        r0, c0 = [int(v) for v in cells.min(axis=0)]
        r1, c1 = [int(v) for v in cells.max(axis=0)]
        if r1 - r0 < 1 or c1 - c0 < 1:
            continue
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        background = next(c for c in large if c != block_color)
        for orientation in ("bottom", "top", "left", "right"):
            if orientation == "bottom":
                if r1 != height - 1 or r0 == 0:
                    continue
                span_low, span_high, strip, depth = c0, c1, r0 - 1, r1 - r0 + 1
                on_strip = lambda pos: pos[0] == strip
                axis = lambda pos: pos[1]
                destination = lambda side: (height - 1, side)
            elif orientation == "top":
                if r0 != 0 or r1 == height - 1:
                    continue
                span_low, span_high, strip, depth = c0, c1, r1 + 1, r1 - r0 + 1
                on_strip = lambda pos: pos[0] == strip
                axis = lambda pos: pos[1]
                destination = lambda side: (0, side)
            elif orientation == "left":
                if c0 != 0 or c1 == width - 1:
                    continue
                span_low, span_high, strip, depth = r0, r1, c1 + 1, c1 - c0 + 1
                on_strip = lambda pos: pos[1] == strip
                axis = lambda pos: pos[0]
                destination = lambda side: (side, 0)
            else:
                if c1 != width - 1 or c0 == 0:
                    continue
                span_low, span_high, strip, depth = r0, r1, c0 - 1, c1 - c0 + 1
                on_strip = lambda pos: pos[1] == strip
                axis = lambda pos: pos[0]
                destination = lambda side: (side, width - 1)

            positions = {color: tuple(int(v) for v in np.argwhere(inp == color)[0])
                         for color in singletons}
            for pusher_color in singletons:
                token_color = next(c for c in singletons if c != pusher_color)
                pusher_pos, token_pos = positions[pusher_color], positions[token_color]
                if not on_strip(pusher_pos) or not on_strip(token_pos):
                    continue
                pusher_axis, token_axis = axis(pusher_pos), axis(token_pos)
                if not span_low <= pusher_axis <= span_high or pusher_axis == token_axis:
                    continue
                sign = 1 if token_axis > pusher_axis else -1
                side = span_high + 1 if sign == 1 else span_low - 1
                if not 0 <= side < (width if orientation in ("bottom", "top") else height):
                    continue
                exterior = token_axis == side
                if not (span_low <= token_axis <= span_high or exterior):
                    continue
                target = destination(side)
                if int(inp[target]) != background:
                    continue
                expected = inp.copy()
                expected[pusher_pos] = background
                expected[token_pos] = pusher_color
                expected[target] = token_color
                if not np.array_equal(expected, out):
                    continue
                if orientation in ("bottom", "top"):
                    opposite_margin = c0 if sign == 1 else width - 1 - c1
                    exit_margin = width - 1 - c1 if sign == 1 else c0
                    front_margin = strip if orientation == "bottom" else height - 1 - strip
                else:
                    opposite_margin = r0 if sign == 1 else height - 1 - r1
                    exit_margin = height - 1 - r1 if sign == 1 else r0
                    front_margin = width - 1 - strip if orientation == "left" else strip
                matches.append({
                    "roles": (background, block_color, pusher_color, token_color),
                    "orientation": orientation, "sign": sign,
                    "shape": inp.shape, "box": (r0, r1, c0, c1),
                    "span": span_high - span_low + 1, "depth": depth,
                    "exterior": exterior, "lateral_leg": abs(side - token_axis),
                    "separation": abs(token_axis - pusher_axis),
                    "pusher_at_end": pusher_axis in (span_low, span_high),
                    "token_near_exit": abs(side - token_axis) <= 2,
                    "opposite_margin": opposite_margin,
                    "exit_margin": exit_margin, "front_margin": front_margin,
                })
    assert len(matches) == 1
    return matches[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _assert_grid(inp)
            _assert_grid(out)
            key = _key(inp)
            pair_key = (key, _key(out))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _source_copy(inp, out)
            info[split].append(_analyze_pair(inp, out))
    train, test = info["train"], info["test"]
    marker_roles = {(x["roles"][2], x["roles"][3]) for x in train + test}
    assert len(marker_roles) == 1
    assert {x["sign"] for x in train} == {-1, 1}
    assert all(not x["exterior"] and x["lateral_leg"] >= 3 for x in train[:2])
    trained_orientations = {x["orientation"] for x in train}
    assert all(x["orientation"] in trained_orientations for x in test)
    trained_exterior_signs = {x["sign"] for x in train if x["exterior"]}
    assert all(not x["exterior"] or x["sign"] in trained_exterior_signs for x in test)
    assert all(x["span"] >= 3 and x["depth"] >= 2 and x["exit_margin"] >= 1
               for x in train + test)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    pusher_color, token_color = [int(v) for v in rng.choice(10, size=2, replace=False)]
    remaining = [c for c in range(10) if c not in (pusher_color, token_color)]
    mixed = bool(rng.random() < 0.55)
    orientations = ("bottom", "top", "left", "right")
    if mixed:
        first_two = rng.choice(4, size=2, replace=False)
        train_orientations = [orientations[int(first_two[0])],
                              orientations[int(first_two[1])]]
        train_orientations += [orientations[int(rng.integers(4))]
                               for _ in range(train_count - 2)]
    else:
        orientation = orientations[int(rng.integers(4))]
        train_orientations = [orientation] * train_count
    first_sign = 1 if rng.random() < 0.5 else -1
    train_signs = [first_sign, -first_sign] + [
        1 if rng.random() < 0.5 else -1 for _ in range(train_count - 2)]
    test_first = 1 if rng.random() < 0.5 else -1
    test_signs = [test_first]
    if test_count >= 2:
        test_signs.append(-test_first)
    test_signs += [1 if rng.random() < 0.5 else -1
                   for _ in range(test_count - len(test_signs))]
    puzzle = {"train": [], "test": []}
    seen = set()
    exterior_signs = set()

    for split, count, signs in (("train", train_count, train_signs),
                                ("test", test_count, test_signs)):
        for index in range(count):
            sign = signs[index]
            if split == "train":
                orientation = train_orientations[index]
                exterior = index >= 2 and rng.random() < (0.28 if index == 2 else 0.10)
            else:
                orientation = train_orientations[int(rng.integers(train_count))]
                exterior = sign in exterior_signs and rng.random() < 0.08
            for _ in range(80):
                background, block_color = [
                    int(v) for v in rng.choice(remaining, size=2, replace=False)]
                pair = _make_pair(
                    rng, split == "train",
                    (background, block_color, pusher_color, token_color),
                    orientation, sign, exterior, split == "train" and index < 2)
                key = _key(pair["input"])
                if key not in seen and not _source_copy(pair["input"], pair["output"]):
                    break
            else:
                raise ValueError("could not generate a distinct pair")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train" and exterior:
                exterior_signs.add(sign)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    assert _puzzle_key(generate_puzzle(12345, 7, 3)) == _puzzle_key(
        generate_puzzle(12345, 7, 3))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(9000 + index, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
    for index, counts in enumerate(((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (2, 3))):
        try:
            generate_puzzle(10000 + index, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_info = []
    train_info = []
    test_info = []
    mixed_puzzles = 0
    single_orientation_puzzles = 0
    varying_block_colors = 0
    tests_opposite_first = 0
    two_direction_test_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_train = [_analyze_pair(pair["input"], pair["output"])
                       for pair in puzzle["train"]]
        local_test = [_analyze_pair(pair["input"], pair["output"])
                      for pair in puzzle["test"]]
        infos = local_train + local_test
        train_info.extend(local_train)
        test_info.extend(local_test)
        all_info.extend(infos)
        mixed_puzzles += len({x["orientation"] for x in infos}) > 1
        single_orientation_puzzles += len({x["orientation"] for x in infos}) == 1
        varying_block_colors += len({x["roles"][:2] for x in infos}) > 1
        tests_opposite_first += sum(x["sign"] != local_train[0]["sign"]
                                    for x in local_test)
        two_direction_test_puzzles += (len(local_test) >= 2 and
                                        {x["sign"] for x in local_test} == {-1, 1})
    assert mixed_puzzles >= 12
    assert single_orientation_puzzles >= 8
    assert varying_block_colors >= 35
    assert tests_opposite_first >= 20
    assert two_direction_test_puzzles >= 15
    assert {x["orientation"] for x in all_info} == {"top", "bottom", "left", "right"}
    assert {x["sign"] for x in all_info} == {-1, 1}
    assert {x["opposite_margin"] == 0 for x in all_info} == {False, True}
    assert {x["exterior"] for x in all_info} == {False, True}
    assert any(x["lateral_leg"] <= 2 and x["depth"] >= 12 for x in all_info)
    assert any(x["lateral_leg"] >= 8 and x["depth"] <= 8 for x in all_info)
    assert len({x["shape"] for x in all_info}) >= 40
    assert len({(x["shape"], x["box"], x["orientation"], x["separation"])
                for x in all_info}) >= 100
    for infos in (train_info, test_info):
        assert sum(x["opposite_margin"] == 0 for x in infos) >= 15
        assert sum(x["opposite_margin"] > 0 for x in infos) >= 15
        assert sum(x["separation"] == 1 for x in infos) >= 10
        assert sum(x["separation"] >= 5 for x in infos) >= 10
        assert sum(x["pusher_at_end"] for x in infos) >= 10
        assert sum(not x["pusher_at_end"] for x in infos) >= 10
        assert sum(x["token_near_exit"] for x in infos) >= 10
        assert sum(not x["token_near_exit"] for x in infos) >= 10
        assert sum(x["lateral_leg"] >= 8 and x["depth"] <= 8 for x in infos) >= 5
        assert sum(x["lateral_leg"] <= 2 and x["depth"] >= 12 for x in infos) >= 5
    assert 1 <= sum(x["exterior"] for x in train_info) < len(train_info) // 4
    assert sum(max(x["shape"]) <= 11 for x in test_info) >= 8
    assert sum(12 <= max(x["shape"]) <= 18 for x in test_info) >= 12
    assert sum(max(x["shape"]) >= 19 for x in test_info) >= 20
    assert sum(x["lateral_leg"] <= 2 for x in test_info) >= 10
    assert sum(x["lateral_leg"] >= 8 for x in test_info) >= 5
