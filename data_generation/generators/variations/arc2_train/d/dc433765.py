"""Generate aligned singleton-marker puzzles with one-step motion."""

import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _sign(x):
    return (x > 0) - (x < 0)


def _counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _supports(direction):
    dr, dc = direction
    if dr and dc:
        return [(dr, 0), (0, dc)]
    if dr:
        return [(dr, -1), (dr, 1)]
    return [(-1, dc), (1, dc)]


def _layout(rng, direction, distance, style):
    """Sample outside margins independently around a minimal aligned ray."""
    dr, dc = direction
    if style == "compact":
        compact_margin = 0 if rng.integers(3) else 2
        limits = ((0, compact_margin), (0, compact_margin))
    elif style == "roomy":
        limits = ((2, 10), (2, 10))
    elif style == "narrow":
        limits = ((0, 1), (0, 8)) if rng.integers(2) else ((0, 8), (0, 1))
    else:
        limits = ((0, 6), (0, 6))
    margins = []
    for axis, span in enumerate((abs(dr) * distance + 1, abs(dc) * distance + 1)):
        low, high = limits[axis]
        before = int(rng.integers(low, high + 1))
        after = int(rng.integers(low, high + 1))
        if span + before + after < 3:
            missing = 3 - span - before - after
            shift = int(rng.integers(missing + 1))
            before += shift
            after += missing - shift
        overflow = max(0, span + before + after - 30)
        reduction = min(after, overflow)
        after -= reduction
        before -= overflow - reduction
        margins.append((before, after))
    height = abs(dr) * distance + 1 + sum(margins[0])
    width = abs(dc) * distance + 1 + sum(margins[1])
    mover = (margins[0][0] + max(-dr, 0) * distance,
             margins[1][0] + max(-dc, 0) * distance)
    target = (mover[0] + dr * distance, mover[1] + dc * distance)
    return (height, width), mover, target


def _make_pair(shape, mover, target, background, mover_color, target_color):
    dr = _sign(target[0] - mover[0])
    dc = _sign(target[1] - mover[1])
    assert all(3 <= n <= 30 for n in shape)
    assert all(0 <= p < n for p, n in zip(mover, shape))
    assert all(0 <= p < n for p, n in zip(target, shape))
    assert max(abs(target[0] - mover[0]), abs(target[1] - mover[1])) >= 2
    inp = np.full(shape, background, dtype=np.int8)
    inp[mover] = mover_color
    inp[target] = target_color
    out = inp.copy()
    out[mover] = background
    out[mover[0] + dr, mover[1] + dc] = mover_color
    return {"input": inp, "output": out}


def _key(grid):
    return grid.shape, grid.tobytes()


def _analyze(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 3
    assert sorted(int(n) for n in counts) == [1, 1, inp.size - 2]
    background = int(colors[np.argmax(counts)])
    changed = np.argwhere(inp != out)
    assert len(changed) == 2
    old = [tuple(int(v) for v in p) for p in changed
           if inp[tuple(p)] != background and out[tuple(p)] == background]
    new = [tuple(int(v) for v in p) for p in changed
           if inp[tuple(p)] == background and out[tuple(p)] != background]
    assert len(old) == len(new) == 1
    mover, destination = old[0], new[0]
    mover_color = int(inp[mover])
    assert int(out[destination]) == mover_color
    target_color = next(int(c) for c in colors if c not in (background, mover_color))
    target = tuple(int(v) for v in np.argwhere(inp == target_color)[0])
    delta = target[0] - mover[0], target[1] - mover[1]
    direction = _sign(delta[0]), _sign(delta[1])
    assert direction in DIRECTIONS
    assert delta[0] == 0 or delta[1] == 0 or abs(delta[0]) == abs(delta[1])
    distance = max(abs(delta[0]), abs(delta[1]))
    assert distance >= 2
    assert destination == (mover[0] + direction[0], mover[1] + direction[1])
    expected = inp.copy()
    expected[mover] = background
    expected[destination] = mover_color
    assert np.array_equal(out, expected)
    return dict(background=background, mover_color=mover_color,
                target_color=target_color, mover=mover, target=target,
                direction=direction, distance=distance, shape=inp.shape)


def _predict(train_pairs, test_input):
    """Learn the color roles from train outputs; read geometry from the input."""
    training = [_analyze(p) for p in train_pairs]
    mover_color = training[0]["mover_color"]
    target_color = training[0]["target_color"]
    assert all(x["mover_color"] == mover_color and x["target_color"] == target_color
               for x in training)
    colors, counts = np.unique(test_input, return_counts=True)
    assert len(colors) == 3
    background = int(colors[np.argmax(counts)])
    assert background not in (mover_color, target_color)
    assert set(int(x) for x in colors) == {background, mover_color, target_color}
    mover = tuple(int(x) for x in np.argwhere(test_input == mover_color)[0])
    target = tuple(int(x) for x in np.argwhere(test_input == target_color)[0])
    dr, dc = _sign(target[0] - mover[0]), _sign(target[1] - mover[1])
    result = test_input.copy()
    result[mover] = background
    result[mover[0] + dr, mover[1] + dc] = mover_color
    return result


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _counts(rng, num_train, num_test)

    # Color roles remain identifiable through every example of this puzzle.
    while True:
        mover_color, target_color = (int(x) for x in rng.choice(10, 2, replace=False))
        if (mover_color, target_color) != (3, 4):
            break
    backgrounds = [c for c in range(10) if c not in (mover_color, target_color)]
    rng.shuffle(backgrounds)
    variable_background = bool(rng.integers(2))
    pool = backgrounds[:int(rng.integers(2, 5))] if variable_background else backgrounds[:1]

    holdout = DIRECTIONS[int(rng.integers(8))]
    support = _supports(holdout)
    rng.shuffle(support)
    train_directions = support + [support[0], tuple(-x for x in support[0])]
    other_directions = [d for d in DIRECTIONS if d != holdout]
    while len(train_directions) < num_train:
        train_directions.append(other_directions[int(rng.integers(len(other_directions)))])

    far = int(rng.integers(4, 13))
    styles = ("compact", "narrow", "medium", "roomy")
    anchored_contrast = bool(num_train >= 3 and rng.integers(2))
    if anchored_contrast:
        shape0, mover0, _ = _layout(rng, support[0], far, "medium")
        target0 = (mover0[0] + 2 * support[0][0], mover0[1] + 2 * support[0][1])
    else:
        shape0, mover0, target0 = _layout(rng, support[0], 2, "compact")
    layouts = [(shape0, mover0, target0)]
    layouts.append(_layout(rng, support[1], far,
                           "roomy" if rng.integers(2) else "narrow"))
    if num_train >= 3:
        if anchored_contrast:
            far_target = (mover0[0] + far * support[0][0],
                          mover0[1] + far * support[0][1])
            layouts.append((shape0, mover0, far_target))
        else:
            layouts.append(_layout(rng, support[0], int(rng.integers(3, 13)),
                                   styles[int(rng.integers(4))]))
    if num_train >= 4:
        layouts.append((shape0, target0, mover0))
    while len(layouts) < num_train:
        index = len(layouts)
        layouts.append(_layout(rng, train_directions[index],
                               int(rng.integers(2, 13)),
                               styles[int(rng.integers(4))]))

    def choose_background(index, training):
        if len(pool) == 1:
            return pool[0]
        if training and index < 2:
            return pool[index]
        return pool[int(rng.integers(len(pool)))]

    train, seen = [], set()
    for index, (shape, mover, target) in enumerate(layouts):
        background = choose_background(index, True)
        pair = _make_pair(shape, mover, target, background, mover_color, target_color)
        key = _key(pair["input"])
        if key in seen:
            direction = train_directions[index]
            for _ in range(60):
                shape, mover, target = _layout(rng, direction,
                    int(rng.integers(2, 13)), styles[int(rng.integers(4))])
                pair = _make_pair(shape, mover, target, background, mover_color, target_color)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise AssertionError("unique training layout unavailable")
        seen.add(key)
        train.append(pair)

    test = []
    for index in range(num_test):
        for _ in range(100):
            reverse = bool(index == 0 and rng.integers(5) == 0)
            if reverse:
                shape, mover, target = shape0, target0, mover0
            else:
                if index == 0:
                    direction = (holdout if rng.integers(4) else
                                 train_directions[int(rng.integers(num_train))])
                else:
                    direction = (holdout if rng.integers(2) else
                                 train_directions[int(rng.integers(num_train))])
                if rng.integers(5) < 2:
                    distance = _analyze(train[int(rng.integers(num_train))])["distance"]
                else:
                    distance = int(rng.integers(2, 15))
                shape, mover, target = _layout(
                    rng, direction, distance, styles[int(rng.integers(4))])
            background = choose_background(index, False)
            pair = _make_pair(shape, mover, target, background, mover_color, target_color)
            key = _key(pair["input"])
            if key not in seen:
                break
        else:
            raise AssertionError("unique test layout unavailable")
        seen.add(key)
        test.append(pair)

    rng.shuffle(train)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            input_key = _key(pair["input"])
            pair_key = input_key, _key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _analyze(pair)
            # Every source pair uses precisely (background, mover, target) = (0,3,4).
            assert (info["background"], info["mover_color"], info["target_color"]) != (0, 3, 4)
            infos[split].append(info)
    all_infos = infos["train"] + infos["test"]
    mover_color, target_color = all_infos[0]["mover_color"], all_infos[0]["target_color"]
    assert all(x["mover_color"] == mover_color and x["target_color"] == target_color
               for x in all_infos)
    assert len({x["direction"] for x in infos["train"]}) >= 2
    assert any(x["distance"] == 2 for x in infos["train"])
    assert any(x["distance"] >= 4 for x in infos["train"])
    assert all(x["distance"] <= 12 for x in infos["train"])
    assert all(x["distance"] <= 14 for x in infos["test"])
    for pair in puzzle["test"]:
        assert np.array_equal(_predict(puzzle["train"], pair["input"]), pair["output"])


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzle_key(generate_puzzle(76543, 8, 3)) == _puzzle_key(generate_puzzle(76543, 8, 3))
    assert _puzzle_key(generate_puzzle(76543, 8, 3)) != _puzzle_key(generate_puzzle(76544, 8, 3))

    shapes, palettes = set(), set()
    train_directions, test_directions = set(), set()
    train_distances, test_distances = set(), set()
    backgrounds, areas = set(), []
    small = large = reversals = anchored = shared_split = variable = fixed = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_info = [_analyze(p) for p in puzzle["train"]]
        test_info = [_analyze(p) for p in puzzle["test"]]
        infos = train_info + test_info
        palette = (infos[0]["mover_color"], infos[0]["target_color"], infos[0]["background"])
        palettes.add(palette)
        all_backgrounds = {x["background"] for x in infos}
        variable += len(all_backgrounds) > 1
        fixed += len(all_backgrounds) == 1
        for x in infos:
            shapes.add(x["shape"])
            backgrounds.add(x["background"])
            area = x["shape"][0] * x["shape"][1]
            areas.append(area)
            small += area <= 25
            large += area >= 150
        train_directions.update(x["direction"] for x in train_info)
        test_directions.update(x["direction"] for x in test_info)
        train_distances.update(x["distance"] for x in train_info)
        test_distances.update(x["distance"] for x in test_info)
        for a in train_info:
            for b in train_info + test_info:
                if a is b:
                    continue
                reversals += (a["mover"] == b["target"] and a["target"] == b["mover"]
                              and a["shape"] == b["shape"])
                anchored += (a["mover"] == b["mover"] and a["direction"] == b["direction"]
                             and a["distance"] != b["distance"])
        shared_split += bool(
            {(x["direction"], x["distance"]) for x in train_info}
            & {(x["direction"], x["distance"]) for x in test_info})
    assert len(shapes) >= 45 and len(palettes) >= 30
    assert train_directions == test_directions == set(DIRECTIONS)
    assert min(areas) == 9 and max(areas) >= 300
    assert small >= 15 and large >= 40
    assert min(train_distances) == 2 and max(test_distances) >= 13
    assert len(train_distances) >= 10 and len(test_distances) >= 10
    assert reversals >= 8 and anchored >= 3 and shared_split >= 3
    assert variable >= 8 and fixed >= 8 and len(backgrounds) == 10

    for seed, nt, nq in ((100, 2, 1), (101, 10, 1), (102, 10, 4)):
        p = generate_puzzle(seed, nt, nq)
        assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for seed, nt, nq in ((200, None, 1), (201, None, 4),
                         (202, 2, None), (203, 10, None)):
        p = generate_puzzle(seed, nt, nq)
        assert nt is None or len(p["train"]) == nt
        assert nq is None or len(p["test"]) == nq
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 2, "num_test": 3}, {"num_train": True}):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
