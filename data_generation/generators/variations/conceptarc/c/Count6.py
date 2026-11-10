"""Framed-cell counting puzzles with a fixed seven-cell tally."""

import numpy as np


SOURCE_KEYS = {
    ((12, 12), 0, 3, (1,), 2),
    ((7, 11), 0, 5, (7,), 5),
    ((15, 15), 0, 6, (8,), 3),
    ((11, 9), 0, 2, (2,), 4),
    ((6, 6), 0, 9, (4,), 2),
}
NON4 = (1, 2, 3, 5, 6, 7, 8, 9)


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None:
            integer = isinstance(value, (int, np.integer)) and not isinstance(
                value, (bool, np.bool_)
            )
            if not integer or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None:
        options = np.arange(max(2, (num_test or 0) + 1), 11)
        weights = np.array([1.5 if n <= 4 else 1.0 for n in options])
        num_train = int(rng.choice(options, p=weights / weights.sum()))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _count_plans(rng, train_size, test_size):
    source_like = train_size == 2 and rng.random() < 0.45
    if source_like:
        train = [2, 5]
        test = [int(rng.choice((3, 4))) for _ in range(test_size)]
        if test_size >= 2:
            test[:2] = [3, 4]
    else:
        first = int(rng.choice((2, 3, 4, 5)))
        second = int(rng.choice([n for n in range(1, 8) if n != first]))
        train = [first, second]
        remaining = [n for n in range(1, 8) if n not in train]
        rng.shuffle(remaining)
        train.extend(remaining[: min(train_size - 2, len(remaining))])
        while len(train) < train_size:
            train.append(int(rng.integers(1, 8)))
        test = [int(rng.choice((1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7)))
                for _ in range(test_size)]
        if train_size >= 3 and rng.random() < 0.19:
            train[int(rng.integers(2, train_size))] = 0
            test[int(rng.integers(test_size))] = 0
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test, source_like


def _size_plan(rng, size):
    bands = []
    while len(bands) < size:
        cycle = ["small", "medium", "large"]
        rng.shuffle(cycle)
        bands.extend(cycle)
    return bands[:size]


def _pattern(rng, count):
    if count == 0:
        return "empty"
    if count == 1:
        return "single"
    if count == 2:
        return str(rng.choice(("separated", "pair", "random")))
    choices = ["separated", "run", "L", "random"]
    if count >= 4:
        choices.append("block")
    return str(rng.choice(choices))


def _minimum_interior(count, pattern):
    if pattern == "separated":
        for side in (2, 3, 4):
            if (side * side + 1) // 2 >= count:
                return side
    return 2 if count <= 2 else 3


def _geometry(rng, band, count, pattern, inset):
    minimum = _minimum_interior(count, pattern) + (2 if inset else 0)
    if band == "small" and minimum == 2 and rng.random() < 0.28:
        lows, highs = (6, 6), (6, 6)
    elif band == "small":
        lows, highs = (6, 6), (10, 10)
    elif band == "medium":
        lows, highs = (9, 9), (17, 17)
    else:
        lows, highs = (17, 8), (29, 29)
        if bool(rng.integers(2)):
            lows = lows[::-1]
    height = int(rng.integers(max(lows[0], minimum + 4), highs[0] + 1))
    width = int(rng.integers(max(lows[1], minimum + 4), highs[1] + 1))
    outside_h = int(rng.integers(2, min(7, height - minimum - 2) + 1))
    outside_w = int(rng.integers(2, min(7, width - minimum - 2) + 1))
    top = int(rng.integers(1, outside_h))
    left = int(rng.integers(1, outside_w))
    bottom = height - (outside_h - top) - 1
    right = width - (outside_w - left) - 1
    return height, width, top, bottom, left, right


def _positions(rng, interior_h, interior_w, count, pattern):
    cells = [(r, c) for r in range(interior_h) for c in range(interior_w)]
    if count == 0:
        return []
    chosen = []
    if pattern == "separated":
        parity = int(rng.integers(2))
        pool = [(r, c) for r, c in cells if (r + c) % 2 == parity]
        if len(pool) < count:
            pool = [(r, c) for r, c in cells if (r + c) % 2 != parity]
        indices = rng.choice(len(pool), size=count, replace=False)
        return [pool[int(i)] for i in indices]
    if pattern in ("pair", "run"):
        length = 2 if pattern == "pair" else 3
        vertical = bool(rng.integers(2))
        max_r = interior_h - (length if vertical else 1) + 1
        max_c = interior_w - (1 if vertical else length) + 1
        r = int(rng.integers(max_r))
        c = int(rng.integers(max_c))
        chosen = [(r + k, c) if vertical else (r, c + k)
                  for k in range(length)]
    elif pattern in ("L", "block"):
        r = int(rng.integers(interior_h - 1))
        c = int(rng.integers(interior_w - 1))
        chosen = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
        if pattern == "L":
            chosen.pop(int(rng.integers(4)))
    elif pattern == "single":
        chosen = [cells[int(rng.integers(len(cells)))]]
    remaining = [cell for cell in cells if cell not in chosen]
    rng.shuffle(remaining)
    return chosen + remaining[:count - len(chosen)]


def _background_plan(rng, train_counts, test_size, source_like):
    train_size = len(train_counts)
    train = [0] * train_size
    test = [0] * test_size
    if source_like or rng.random() >= 0.38:
        return train, test
    candidates = [i for i, count in enumerate(train_counts) if 1 <= count <= 6]
    index = int(rng.choice(candidates))
    zero_index = int(rng.choice([i for i in range(train_size) if i != index]))
    train[index] = int(rng.choice(NON4))
    for i in range(train_size):
        if i not in (index, zero_index) and rng.random() < 0.28:
            train[i] = int(rng.choice(NON4))
    test[int(rng.integers(test_size))] = int(rng.choice(NON4))
    for i in range(test_size):
        if test[i] == 0 and rng.random() < 0.35:
            test[i] = int(rng.choice(NON4))
    return train, test


def _color_plan(rng, train_counts, test_counts, source_like):
    if source_like:
        train = ["plain"] * len(train_counts)
        test = [str(rng.choice(("plain", "same", "output")))
                for _ in test_counts]
        if len(test) >= 2:
            test[0], test[1] = "same", "output"
        return train, test
    train_options = ("plain", "plain", "plain", "same", "same",
                     "output", "output", "both")
    test_options = ("plain", "plain", "same", "output", "both")
    train = [str(rng.choice(train_options)) for _ in train_counts]
    test = [str(rng.choice(test_options)) for _ in test_counts]
    if rng.random() < 0.42:
        train_candidates = [i for i, n in enumerate(train_counts) if n >= 2]
        test_candidates = [i for i, n in enumerate(test_counts) if n >= 2]
        if train_candidates and test_candidates:
            train[int(rng.choice(train_candidates))] = "mixed"
            test[int(rng.choice(test_candidates))] = "mixed"
    return train, test


def _colors(rng, background, role, count):
    visible = [n for n in range(1, 10) if n != background]
    if role == "same":
        frame = int(rng.choice([n for n in visible if n != 4]))
        marks = [frame]
    elif role == "output":
        frame = int(rng.choice([n for n in visible if n != 4]))
        marks = [4]
    elif role == "both":
        frame, marks = 4, [4]
    elif role == "mixed":
        frame = int(rng.choice(visible))
        marks = list(map(int, rng.choice([n for n in visible if n != frame],
                                        size=min(count, 3),
                                        replace=False)))
    else:
        frame = int(rng.choice(visible))
        marks = [int(rng.choice([n for n in visible if n not in (4, frame)]))]
    return frame, marks


def _make_pair(rng, count, band, background, role):
    pattern = _pattern(rng, count)
    inset = role in ("same", "both") and count > 0
    height, width, top, bottom, left, right = _geometry(
        rng, band, count, pattern, inset
    )
    frame, mark_colors = _colors(rng, background, role, count)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[top, left:right + 1] = frame
    grid[bottom, left:right + 1] = frame
    grid[top:bottom + 1, left] = frame
    grid[top:bottom + 1, right] = frame
    interior_h, interior_w = bottom - top - 1, right - left - 1
    positions = _positions(rng, interior_h - 2 if inset else interior_h,
                           interior_w - 2 if inset else interior_w,
                           count, pattern)
    if inset:
        positions = [(r + 1, c + 1) for r, c in positions]
    if count:
        colors = [mark_colors[i % len(mark_colors)] for i in range(count)]
        rng.shuffle(colors)
        for (r, c), color in zip(positions, colors):
            grid[top + 1 + r, left + 1 + c] = color
    output = np.full((7, 1), background, dtype=np.int8)
    output[:count, 0] = 4
    return {"input": grid, "output": output}


def _signature(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _inspect_input(grid):
    height, width = grid.shape
    background = int(grid[0, 0])
    assert background != 4
    assert np.all(grid[0] == background)
    assert np.all(grid[-1] == background)
    assert np.all(grid[:, 0] == background)
    assert np.all(grid[:, -1] == background)
    occupied = np.argwhere(grid != background)
    assert len(occupied) >= 12
    top, left = np.min(occupied, axis=0)
    bottom, right = np.max(occupied, axis=0)
    top, bottom, left, right = map(int, (top, bottom, left, right))
    assert top >= 1 and left >= 1 and bottom < height - 1 and right < width - 1
    assert bottom - top >= 3 and right - left >= 3
    frame = int(grid[top, left])
    assert frame != background
    assert np.all(grid[top, left:right + 1] == frame)
    assert np.all(grid[bottom, left:right + 1] == frame)
    assert np.all(grid[top:bottom + 1, left] == frame)
    assert np.all(grid[top:bottom + 1, right] == frame)
    exterior = grid.copy()
    exterior[top:bottom + 1, left:right + 1] = background
    assert np.all(exterior == background)
    interior = grid[top + 1:bottom, left + 1:right]
    mask = interior != background
    count = int(np.count_nonzero(mask))
    assert 0 <= count <= 7 and count < interior.size
    marks = tuple(int(n) for n in np.unique(interior[mask]))
    vertical = bool(np.any(mask[:-1] & mask[1:]))
    horizontal = bool(np.any(mask[:, :-1] & mask[:, 1:]))
    block = bool(np.any(mask[:-1, :-1] & mask[1:, :-1] &
                        mask[:-1, 1:] & mask[1:, 1:]))
    run = bool(np.any(mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]) or
               np.any(mask[:-2] & mask[1:-1] & mask[2:]))
    l_shape = any(
        int(np.count_nonzero(mask[r:r + 2, c:c + 2])) >= 3
        for r in range(mask.shape[0] - 1)
        for c in range(mask.shape[1] - 1)
    )
    touches = bool(np.any(mask[0]) or np.any(mask[-1]) or
                   np.any(mask[:, 0]) or np.any(mask[:, -1]))
    if frame in marks:
        assert not touches
    return {
        "shape": (height, width),
        "bounds": (top, bottom, left, right),
        "margins": (top, height - 1 - bottom, left, width - 1 - right),
        "background": background,
        "frame": frame,
        "marks": marks,
        "count": count,
        "adjacent": vertical or horizontal,
        "block": block,
        "run": run,
        "L": l_shape,
        "touches": touches,
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_size, test_size = _resolve_counts(rng, num_train, num_test)
    train_counts, test_counts, source_like = _count_plans(rng, train_size, test_size)
    train_sizes = _size_plan(rng, train_size)
    test_sizes = _size_plan(rng, test_size)
    train_backgrounds, test_backgrounds = _background_plan(
        rng, train_counts, test_size, source_like
    )
    train_roles, test_roles = _color_plan(
        rng, train_counts, test_counts, source_like
    )
    puzzle = {"train": [], "test": []}
    used = set()
    for regime, counts, sizes, backgrounds, roles in (
        ("train", train_counts, train_sizes, train_backgrounds, train_roles),
        ("test", test_counts, test_sizes, test_backgrounds, test_roles),
    ):
        for count, size, background, role in zip(
            counts, sizes, backgrounds, roles
        ):
            for _ in range(12):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                pair = _make_pair(child, count, size, background, role)
                facts = _inspect_input(pair["input"])
                source_key = (facts["shape"], facts["background"],
                              facts["frame"], facts["marks"], facts["count"])
                key = _signature(pair["input"])
                if key not in used and source_key not in SOURCE_KEYS:
                    used.add(key)
                    puzzle[regime].append(pair)
                    break
            else:
                raise AssertionError("could not construct a unique pair")
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    facts = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _signature(pair["input"])
            pair_key = (key, _signature(pair["output"]))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            detail = _inspect_input(pair["input"])
            facts[regime].append(detail)
            source_key = (detail["shape"], detail["background"],
                          detail["frame"], detail["marks"], detail["count"])
            assert source_key not in SOURCE_KEYS
            expected = np.full((7, 1), detail["background"], dtype=np.int8)
            expected[:detail["count"], 0] = 4
            assert np.array_equal(pair["output"], expected)
    training, testing = facts["train"], facts["test"]
    assert len({d["count"] for d in training}) >= 2
    assert any(1 <= d["count"] <= 6 for d in training)
    assert any(d["background"] == 0 for d in training)
    if any(d["background"] != 0 for d in testing):
        assert any(d["background"] != 0 and 1 <= d["count"] <= 6
                   for d in training)
    if any(len(d["marks"]) > 1 for d in testing):
        assert any(len(d["marks"]) > 1 for d in training)
    assert len({d["shape"] for d in training}) >= 2


def _puzzle_signature(puzzle):
    return tuple(
        (regime, tuple((_signature(pair["input"]), _signature(pair["output"]))
                       for pair in puzzle[regime]))
        for regime in ("train", "test")
    )


def self_test() -> None:
    train_sizes, test_sizes, signatures = set(), set(), set()
    measured = {"train": [], "test": []}
    structures = set()
    source_like = 0
    within = {"three_counts": 0, "scale_mix": 0, "background_mix": 0,
              "count_overlap": 0, "mixed_both": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _puzzle_signature(puzzle)
        assert signature == _puzzle_signature(generate_puzzle(seed))
        signatures.add(signature)
        train_sizes.add(len(puzzle["train"]))
        test_sizes.add(len(puzzle["test"]))
        details = {"train": [], "test": []}
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                detail = _inspect_input(pair["input"])
                measured[regime].append(detail)
                details[regime].append(detail)
                predicted = np.full((7, 1), detail["background"], dtype=np.int8)
                predicted[:detail["count"], 0] = 4
                assert np.array_equal(predicted, pair["output"])
                top, bottom, left, right = detail["bounds"]
                inside = pair["input"][top + 1:bottom, left + 1:right]
                structures.add((inside.shape, tuple(
                    int(value) for value in (inside != detail["background"]).flat
                )))
        training, testing = details["train"], details["test"]
        within["three_counts"] += len({d["count"] for d in training}) >= 3
        within["scale_mix"] += (
            any(min(d["shape"]) <= 9 for d in training)
            and any(max(d["shape"]) >= 17 for d in training)
        )
        within["background_mix"] += len({d["background"] for d in training}) > 1
        within["count_overlap"] += bool(
            {d["count"] for d in training} & {d["count"] for d in testing}
        )
        within["mixed_both"] += (
            any(len(d["marks"]) > 1 for d in training)
            and any(len(d["marks"]) > 1 for d in testing)
        )
        if {_inspect_input(p["input"])["count"] for p in puzzle["train"]} == {2, 5} \
                and all(_inspect_input(p["input"])["count"] in (3, 4)
                        for p in puzzle["test"]):
            source_like += 1
    assert len(train_sizes) >= 4 and len(test_sizes) >= 3
    assert len(signatures) >= 40
    assert len(structures) >= 150
    assert within["three_counts"] >= 20
    assert within["scale_mix"] >= 20
    assert within["background_mix"] >= 5
    assert within["count_overlap"] >= 15
    assert within["mixed_both"] >= 5
    for regime in ("train", "test"):
        sample = measured[regime]
        assert {d["count"] for d in sample} >= set(range(1, 8))
        assert any(d["count"] == 0 for d in sample)
        assert len({d["shape"] for d in sample}) >= 35
        assert len({d["margins"] for d in sample}) >= 25
        assert any(d["shape"] == (6, 6) for d in sample)
        assert any(min(d["shape"]) == 6 for d in sample)
        assert any(max(d["shape"]) >= 17 for d in sample)
        assert any(max(d["margins"]) >= 2 for d in sample)
        assert any(d["adjacent"] for d in sample)
        assert any(not d["adjacent"] and d["count"] >= 2 for d in sample)
        assert any(d["run"] for d in sample)
        assert any(d["L"] for d in sample)
        assert any(d["block"] for d in sample)
        assert any(d["frame"] in d["marks"] for d in sample)
        assert any(4 in d["marks"] for d in sample)
        assert any(d["frame"] != 4 and d["marks"] and
                   4 not in d["marks"] and d["frame"] not in d["marks"]
                   for d in sample)
        assert any(len(d["marks"]) > 1 for d in sample)
        assert any(d["background"] != 0 for d in sample)
    assert source_like >= 1
    for ntrain, ntest in ((2, 1), (10, 1), (10, 4), (5, 2),
                          (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(12345, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        assert ntrain is None or len(puzzle["train"]) == ntrain
        assert ntest is None or len(puzzle["test"]) == ntest
        assert _puzzle_signature(puzzle) == _puzzle_signature(
            generate_puzzle(12345, ntrain, ntest)
        )
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid counts: {arguments}")
