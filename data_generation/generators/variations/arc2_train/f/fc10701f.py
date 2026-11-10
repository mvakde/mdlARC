"""Compositional generator for the square-through-gates ARC family."""

import numpy as np


_SOURCE_INPUTS = tuple(
    tuple(line for line in block.split())
    for block in (
        """
666666666666
666666996666
666666996666
666600660066
666666666666
666600660066
666666666666
666666666666
666600660066
666666666666
666666776666
666666776666
""",
        """
66666666
66666666
66606066
66606066
67666669
66606066
66606066
66666666
""",
        """
6666666666666666
6666666666666666
6666666666666666
6666666666666666
6666666666666666
6060606066066666
6060606066066666
6666996666667766
6666996666667766
6060606066066666
6060606066066666
6666666666666666
6666666666666666
6666666666666666
6666666666666666
6666666666666666
""",
        """
66666666667777666666666666
66666666667777666666666666
66666666667777666666666666
66666666667777666666666666
66666666666666666666666666
66666666666666666666666666
66666666006666006666666666
66666666666666666666666666
66666666006666006666666666
66666666669999666666666666
66666666669999666666666666
66666666009999006666666666
66666666669999666666666666
66666666006666006666666666
66666666666666666666666666
66666666006666006666666666
66666666666666666666666666
66666666006666006666666666
66666666666666666666666666
66666666006666006666666666
66666666666666666666666666
66666666666666666666666666
66666666666666666666666666
66666666006666006666666666
66666666666666666666666666
66666666666666666666666666
""",
    )
)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def _is_source_input(grid):
    lines = tuple("".join(str(int(v)) for v in row) for row in grid)
    return lines in _SOURCE_INPUTS


def _choice(rng, values):
    return int(values[int(rng.integers(len(values)))])


def _positions(rng, positions, count):
    if count == 0:
        return set()
    if count >= 2 and len(positions) > count and rng.random() < 0.42:
        offset = int(rng.integers(0, len(positions) - count + 1))
        return set(positions[offset:offset + count])
    return {int(x) for x in rng.choice(positions, size=count, replace=False)}


def _local_colors(rng, roles, previous=None):
    pool = [c for c in range(10) if c not in roles]
    background_choices = [c for c in pool if previous is None or c != previous[0]]
    background = _choice(rng, background_choices)
    obstacle_choices = [
        c for c in pool
        if c != background and (previous is None or c != previous[1])
    ]
    obstacle = _choice(rng, obstacle_choices)
    return background, obstacle


def _paint_square(grid, vertical, axial, cross, side, color):
    if vertical:
        grid[axial:axial + side, cross:cross + side] = color
    else:
        grid[cross:cross + side, axial:axial + side] = color


def _paint_opening(grid, vertical, axial, cross, side, color):
    if vertical:
        grid[axial, cross:cross + side] = color
    else:
        grid[cross:cross + side, axial] = color


def _arm_extent(rng, room):
    roll = rng.random()
    if roll < 0.20:
        return room
    if roll < 0.34:
        return 1
    return int(rng.integers(1, room + 1))


def _paint_gate(grid, vertical, axial, cross, side, low, high, color):
    if vertical:
        grid[axial, cross - low:cross] = color
        grid[axial, cross + side:cross + side + high] = color
    else:
        grid[cross - low:cross, axial] = color
        grid[cross + side:cross + side + high, axial] = color


def _generate_pair(rng, roles, local_colors, regime, options):
    mover_color, goal_color, marker = roles
    background, obstacle = local_colors
    direction = options.get("direction")
    if direction is None:
        direction = ("up", "down", "left", "right")[int(rng.integers(4))]
    vertical = direction in ("up", "down")
    positive = direction in ("down", "right")

    side = options.get("side")
    if side is None:
        side = int(rng.integers(2, 5)) if options.get("teaching") else int(rng.integers(1, 6))
    min_gap = 3 if options.get("teaching") else 1
    gap = int(rng.integers(min_gap, min(13, 30 - 2 * side) + 1))
    base_length = 2 * side + gap

    if options.get("teaching"):
        edge_mode = "interior"
    else:
        edge_mode = ("mover", "goal", "interior", "both")[
            int(rng.choice(4, p=[0.25, 0.25, 0.40, 0.10]))
        ]
        if options.get("length_min", 0) >= 22 and edge_mode == "both":
            edge_mode = "mover" if rng.random() < 0.5 else "goal"
    minimum_spare = 2 if edge_mode == "interior" else 1 if edge_mode in ("mover", "goal") else 0
    if edge_mode == "both":
        axial_length = base_length
    else:
        minimum_length = max(8, base_length + minimum_spare, options.get("length_min", 0))
        if minimum_length > 30:
            minimum_length = base_length + minimum_spare
        if rng.random() < (0.42 if regime == "target_test" else 0.32):
            minimum_length = max(minimum_length, 21)
        axial_length = int(rng.integers(minimum_length, 31))
    spare = axial_length - base_length
    if edge_mode == "mover":
        lead, trail = 0, spare
    elif edge_mode == "goal":
        lead, trail = spare, 0
    elif edge_mode == "both":
        lead, trail = 0, 0
    else:
        lead = int(rng.integers(1, spare))
        trail = spare - lead

    if positive:
        mover_start = lead
        goal_start = lead + side + gap
    else:
        goal_start = trail
        mover_start = trail + side + gap

    cross_min = max(7, side + 2)
    if rng.random() < 0.43:
        cross_max = max(cross_min, 16)
    else:
        cross_max = 30
    cross_length = int(rng.integers(cross_min, cross_max + 1))
    if cross_length - side >= 7 and rng.random() < 0.58:
        cross_start = int(rng.integers(3, cross_length - side - 1))
    else:
        cross_start = int(rng.integers(1, cross_length - side))
    shape = (axial_length, cross_length) if vertical else (cross_length, axial_length)

    between = list(range(min(mover_start, goal_start) + side, max(mover_start, goal_start)))
    if positive:
        behind = list(range(0, mover_start))
        beyond = list(range(goal_start + side, axial_length))
    else:
        behind = list(range(mover_start + side, axial_length))
        beyond = list(range(0, goal_start))
    start_rows = list(range(mover_start, mover_start + side))
    goal_rows = list(range(goal_start, goal_start + side))

    if options.get("teaching"):
        crossed_count = int(rng.integers(1, min(3, len(between)) + 1))
        gates = _positions(rng, between, crossed_count)
        gates.add(_choice(rng, behind))
        gates.add(_choice(rng, beyond))
        gates.add(_choice(rng, start_rows))
        gates.add(_choice(rng, goal_rows))
    else:
        if regime == "target_test" and rng.random() < 0.16:
            crossed_count = 0
        elif rng.random() < 0.26:
            crossed_count = 1
        else:
            crossed_count = int(rng.integers(1, min(6, len(between)) + 1))
        gates = _positions(rng, between, crossed_count)
        if rng.random() >= (0.19 if regime == "target_test" else 0.13):
            for positions, probability in (
                (behind, 0.43), (beyond, 0.55),
                (start_rows, 0.34), (goal_rows, 0.42),
            ):
                if positions and rng.random() < probability:
                    if positions in (behind, beyond):
                        count = int(rng.integers(1, min(6, len(positions)) + 1))
                        gates.update(_positions(rng, positions, count))
                    else:
                        gates.add(_choice(rng, positions))
        if not gates:
            gates.add(_choice(rng, goal_rows))

    inp = np.full(shape, background, dtype=np.int8)
    prior_axial, prior_extents = -2, None
    band = rng.random() < 0.65
    for axial in sorted(gates):
        if band and axial == prior_axial + 1 and rng.random() < 0.80:
            low, high = prior_extents
        else:
            low = _arm_extent(rng, cross_start)
            high = _arm_extent(rng, cross_length - cross_start - side)
        _paint_gate(inp, vertical, axial, cross_start, side, low, high, obstacle)
        prior_axial, prior_extents = axial, (low, high)

    _paint_square(inp, vertical, mover_start, cross_start, side, mover_color)
    _paint_square(inp, vertical, goal_start, cross_start, side, goal_color)
    out = inp.copy()
    _paint_square(out, vertical, mover_start, cross_start, side, background)
    for axial in sorted(gates.intersection(between)):
        _paint_opening(out, vertical, axial, cross_start, side, marker)
    _paint_square(out, vertical, goal_start, cross_start, side, mover_color)
    return inp, out


def _square_info(grid, color):
    cells = np.argwhere(grid == color)
    assert len(cells) > 0
    low = cells.min(axis=0)
    high = cells.max(axis=0)
    side = int(high[0] - low[0] + 1)
    assert side == int(high[1] - low[1] + 1)
    assert len(cells) == side * side
    assert np.all(grid[low[0]:high[0] + 1, low[1]:high[1] + 1] == color)
    return int(low[0]), int(low[1]), side


def _analyze_pair(inp, out):
    assert inp.shape == out.shape
    input_colors = {int(c) for c in np.unique(inp)}
    output_colors = {int(c) for c in np.unique(out)}
    assert len(input_colors) == 4
    lost = input_colors - output_colors
    added = output_colors - input_colors
    assert len(lost) == 1 and len(added) <= 1
    goal = next(iter(lost))
    goal_replacements = {int(c) for c in np.unique(out[inp == goal])}
    assert len(goal_replacements) == 1
    mover = next(iter(goal_replacements))
    assert mover in input_colors and mover != goal
    start_replacements = {int(c) for c in np.unique(out[inp == mover])}
    assert len(start_replacements) == 1
    background = next(iter(start_replacements))
    assert background in input_colors - {mover, goal}
    obstacle_set = input_colors - {background, mover, goal}
    assert len(obstacle_set) == 1
    obstacle = next(iter(obstacle_set))
    marker = next(iter(added)) if added else None

    mr, mc, mover_side = _square_info(inp, mover)
    gr, gc, goal_side = _square_info(inp, goal)
    assert mover_side == goal_side
    side = mover_side
    assert 1 <= side <= 5
    assert (mr == gr) != (mc == gc)
    vertical = mc == gc
    cross = mc if vertical else mr
    mover_start = mr if vertical else mc
    goal_start = gr if vertical else gc
    direction = ("down" if goal_start > mover_start else "up") if vertical else (
        "right" if goal_start > mover_start else "left"
    )
    assert abs(goal_start - mover_start) > side
    axial_length = inp.shape[0] if vertical else inp.shape[1]
    cross_length = inp.shape[1] if vertical else inp.shape[0]
    assert cross >= 1 and cross + side < cross_length
    gates, arms = [], []
    for axial in range(axial_length):
        line = inp[axial, :] if vertical else inp[:, axial]
        positions = np.flatnonzero(line == obstacle)
        if not len(positions):
            continue
        low = positions[positions < cross]
        high = positions[positions >= cross + side]
        assert len(low) and len(high)
        assert len(low) + len(high) == len(positions)
        assert np.array_equal(low, np.arange(low[0], cross))
        assert np.array_equal(high, np.arange(cross + side, high[-1] + 1))
        gates.append(axial)
        arms.append((len(low), len(high)))
        opening = line[cross:cross + side]
        if mover_start <= axial < mover_start + side:
            color = mover
        elif goal_start <= axial < goal_start + side:
            color = goal
        else:
            color = background
        assert np.all(opening == color)
    assert gates

    between = set(range(min(mover_start, goal_start) + side, max(mover_start, goal_start)))
    crossed = [a for a in gates if a in between]
    start = [a for a in gates if mover_start <= a < mover_start + side]
    target = [a for a in gates if goal_start <= a < goal_start + side]
    if goal_start > mover_start:
        behind = [a for a in gates if a < mover_start]
        beyond = [a for a in gates if a >= goal_start + side]
    else:
        behind = [a for a in gates if a >= mover_start + side]
        beyond = [a for a in gates if a < goal_start]
    assert len(crossed) + len(start) + len(target) + len(behind) + len(beyond) == len(gates)
    assert (marker is not None) == bool(crossed)
    expected = inp.copy()
    _paint_square(expected, vertical, mover_start, cross, side, background)
    for axial in crossed:
        _paint_opening(expected, vertical, axial, cross, side, marker)
    _paint_square(expected, vertical, goal_start, cross, side, mover)
    assert np.array_equal(out, expected)
    return {
        "roles": (mover, goal, marker),
        "local": (background, obstacle),
        "direction": direction,
        "side": side,
        "shape": inp.shape,
        "axial_length": axial_length,
        "cross_length": cross_length,
        "mover_start": mover_start,
        "goal_start": goal_start,
        "cross_start": cross,
        "crossed": crossed,
        "start": start,
        "target": target,
        "behind": behind,
        "beyond": beyond,
        "gates": gates,
        "arms": arms,
        "edge": (
            mover_start == 0 or mover_start + side == axial_length,
            goal_start == 0 or goal_start + side == axial_length,
        ),
    }


def _validate_impl(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert inp.shape == out.shape
            input_key = _grid_key(inp)
            pair_key = input_key, _grid_key(out)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_input(inp)
            analyses[split].append(_analyze_pair(inp, out))

    train = analyses["train"]
    test = analyses["test"]
    assert all(info["crossed"] and info["roles"][2] is not None for info in train)
    assert any(info["start"] for info in train)
    assert any(info["target"] for info in train)
    assert any(info["behind"] for info in train)
    assert any(info["beyond"] for info in train)
    mover, goal, marker = train[0]["roles"]
    assert len({mover, goal, marker}) == 3
    assert all(info["roles"] == (mover, goal, marker) for info in train)
    for info in train + test:
        assert info["roles"][:2] == (mover, goal)
        assert info["roles"][2] in (None, marker)
        assert not ({mover, goal, marker} & set(info["local"]))
    assert train[0]["local"][0] != train[1]["local"][0]
    assert train[0]["local"][1] != train[1]["local"][1]
    assert train[0]["direction"] != train[1]["direction"]
    assert train[0]["side"] != train[1]["side"]


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed generated puzzle") from exc


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    colors = [int(c) for c in rng.choice(10, size=3, replace=False)]
    roles = tuple(colors)  # mover, target, traversal marker remain stable
    puzzle = {"train": [], "test": []}
    analyses = []
    seen = set()
    previous_colors = None

    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            options = {}
            if split == "train" and index == 0:
                options["teaching"] = True
            elif split == "train" and index == 1:
                first = analyses[0]
                options["direction"] = (
                    [d for d in ("up", "down", "left", "right") if d != first["direction"]]
                )[int(rng.integers(3))]
                options["side"] = (
                    [s for s in range(1, 6) if s != first["side"]]
                )[int(rng.integers(4))]
            elif split == "test" and index == 0:
                mode = int(rng.integers(3))
                if mode == 0:
                    example = analyses[int(rng.integers(len(analyses)))]
                    options["direction"] = example["direction"]
                    options["side"] = example["side"]
                elif mode == 1:
                    familiar = {(a["direction"], a["side"]) for a in analyses}
                    seen_directions = {a["direction"] for a in analyses}
                    seen_sides = {a["side"] for a in analyses}
                    all_novel = [
                        (d, s)
                        for d in ("up", "down", "left", "right")
                        for s in range(1, 6)
                        if (d, s) not in familiar
                    ]
                    novel = [
                        (d, s) for d, s in all_novel
                        if (d in seen_directions) != (s in seen_sides)
                    ]
                    if not novel:
                        novel = all_novel
                    if novel:
                        options["direction"], options["side"] = novel[
                            int(rng.integers(len(novel)))
                        ]
                else:
                    options["side"] = int(rng.integers(3, 6))
                    options["length_min"] = 22
            for attempt in range(30):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                local = _local_colors(child_rng, roles, previous_colors if split == "train" and index == 1 else None)
                regime = "source_train" if split == "train" else "target_test"
                inp, out = _generate_pair(child_rng, roles, local, regime, options)
                key = _grid_key(inp)
                if key not in seen and not _is_source_input(inp):
                    break
            else:
                raise ValueError("could not construct a distinct pair")
            puzzle[split].append({"input": inp, "output": out})
            seen.add(key)
            if split == "train":
                analyses.append(_analyze_pair(inp, out))
                if index == 0:
                    previous_colors = local
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple(
        tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split])
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert all(validate_generated_puzzle(puzzle) is None for puzzle in puzzles)
    assert _fingerprint(generate_puzzle(9173, 7, 3)) == _fingerprint(generate_puzzle(9173, 7, 3))
    assert _fingerprint(generate_puzzle(23)) == _fingerprint(generate_puzzle(23))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(100 + train_count + test_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(300 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": True, "num_test": 1},
    ):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")

    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_fingerprint(p) for p in puzzles}) >= 40
    records = [
        (split, _analyze_pair(pair["input"], pair["output"]))
        for p in puzzles for split in ("train", "test") for pair in p[split]
    ]
    assert {a["direction"] for _, a in records} == {"up", "down", "left", "right"}
    assert {a["side"] for _, a in records} == {1, 2, 3, 4, 5}
    assert len({a["shape"] for _, a in records}) >= 60
    assert len({
        (a["shape"], a["direction"], a["side"], tuple(a["gates"]),
         a["mover_start"], a["goal_start"], a["cross_start"])
        for _, a in records
    }) >= 160
    for split in ("train", "test"):
        selected = [a for part, a in records if part == split]
        assert {a["direction"] for a in selected} == {"up", "down", "left", "right"}
        assert any(a["edge"][0] for a in selected)
        assert any(a["edge"][1] for a in selected)
        assert any(not any(a["edge"]) for a in selected)
        assert any(a["behind"] for a in selected)
        assert any(a["beyond"] for a in selected)
        assert any(a["start"] for a in selected)
        assert any(a["target"] for a in selected)
        assert any(max(max(arm) for arm in a["arms"]) >= 5 for a in selected)
        assert any(any(low != high for low, high in a["arms"]) for a in selected)
        assert any(any(y == x + 1 and a["arms"][i] == a["arms"][i + 1]
                       for i, (x, y) in enumerate(zip(a["gates"], a["gates"][1:])))
                   for a in selected)
    tests = [a for split, a in records if split == "test"]
    assert {min(3, len(a["crossed"])) for a in tests} == {0, 1, 2, 3}
    assert any(not (a["behind"] or a["beyond"] or a["start"] or a["target"]) for a in tests)
    assert any(len(a["beyond"]) >= 3 for a in tests)
    assert any(a["axial_length"] >= 22 and a["side"] >= 3 for a in tests)
    assert any(a["axial_length"] <= 17 for a in tests)
