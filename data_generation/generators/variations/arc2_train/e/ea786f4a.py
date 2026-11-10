"""Centered diagonal-completion ARC puzzles."""

import numpy as np


SIZES = tuple(range(3, 30, 2))
SOURCE = frozenset({(5, 2, 0), (3, 1, 0), (7, 3, 0), (11, 6, 0)})
RAYS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, rng, num_train, num_test):
    a = _count("num_train", num_train, 2, 10)
    b = _count("num_test", num_test, 1, 4)
    if a is not None and b is not None:
        if a <= b:
            raise ValueError("num_train must exceed num_test")
    elif a is None and b is None:
        b = 1 + seed % 4
        a = b + 1 + (seed // 4) % (10 - b)
    elif a is None:
        a = int(rng.integers(b + 1, 11))
    else:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    return a, b


def _pick(rng, options):
    assert len(options) > 0
    return options[int(rng.integers(len(options)))]


def _sizes(rng, a, b, mode):
    def fill_train(first, pool):
        train = list(first)
        for _ in range(a - 2):
            train.append(_pick(rng, [n for n in pool if train.count(n) < 4]))
        return train

    def fill_test(first, pool):
        test = list(first)
        for _ in range(b - len(test)):
            test.append(_pick(rng, [n for n in pool if n not in test]))
        return test

    if mode == "extrapolate":
        pool = SIZES[:8]
        first = [int(x) for x in rng.choice(pool, 2, replace=False)]
        train = fill_train(first, pool)
        pool = [n for n in SIZES if n > max(train)]
        test = fill_test([], pool)
    elif mode == "interpolate":
        lo = int(rng.integers(0, len(SIZES) - 2))
        hi = int(rng.integers(lo + 2, len(SIZES)))
        train = fill_train([SIZES[lo], SIZES[hi]], SIZES[lo:hi + 1])
        test = fill_test([_pick(rng, SIZES[lo + 1:hi])], SIZES)
    elif mode == "smaller":
        pool = SIZES[4:]
        first = [int(x) for x in rng.choice(pool, 2, replace=False)]
        train = fill_train(first, pool)
        pool = [n for n in SIZES if n < min(train)]
        test = fill_test([_pick(rng, pool)], SIZES)
    else:
        first = [int(x) for x in rng.choice(SIZES, 2, replace=False)]
        train = fill_train(first, SIZES)
        test = fill_test([_pick(rng, train)], SIZES)
    return train, test


def _colors(rng, train_sizes, test_sizes, variable, swap, overlap):
    marker_pool = [int(x) for x in rng.permutation(10)[:5]]
    field_pool = [int(x) for x in rng.permutation(10)[:6]]
    used = set()

    def choose(size, fields, markers, condition=None):
        candidates = [(f, m) for f in fields for m in markers
                      if f != m and (size, f, m) not in SOURCE
                      and (size, f, m) not in used
                      and (condition is None or condition(f, m))]
        f, m = _pick(rng, candidates)
        used.add((size, f, m))
        return f, m

    if swap:
        options = [(f, m) for f in marker_pool for m in marker_pool if f != m
                   and (train_sizes[0], f, m) not in SOURCE
                   and (train_sizes[1], m, f) not in SOURCE]
        first = _pick(rng, options)
        second = (first[1], first[0])
        used.add((train_sizes[0], *first))
        used.add((train_sizes[1], *second))
        field_pool = list(dict.fromkeys(field_pool + [first[0], second[0]]))
    else:
        first = choose(train_sizes[0], field_pool, marker_pool)
        if variable:
            second = choose(train_sizes[1], field_pool, marker_pool,
                            lambda f, m: m != first[1])
        else:
            second = choose(train_sizes[1], field_pool, [first[1]],
                            lambda f, m: f != first[0])
    train = [first, second]
    for size in train_sizes[2:]:
        train.append(choose(size, field_pool, marker_pool if variable else [first[1]]))

    train_fields = {f for f, _ in train}
    train_markers = {m for _, m in train}
    first_fields = (sorted(train_fields) if overlap
                    else [c for c in range(10) if c not in train_fields])
    first_markers = ([c for c in range(10) if c not in train_markers]
                     if variable else [first[1]])
    if overlap and not any(
        f != m and (test_sizes[0], f, m) not in SOURCE
        and (test_sizes[0], f, m) not in used
        for f in first_fields for m in first_markers
    ):
        first_fields = [c for c in range(10) if c not in train_fields]
    test = [choose(test_sizes[0], first_fields, first_markers)]
    for size in test_sizes[1:]:
        fields = list(range(10)) if overlap else [
            c for c in range(10) if c not in train_fields
        ]
        test.append(choose(size, fields, list(range(10)) if variable else [first[1]]))
    return train, test


def _pair(rng, size, field, marker, stub):
    inp = np.full((size, size), field, dtype=np.int8)
    out = np.full((size, size), field, dtype=np.int8)
    center = size // 2
    inp[center, center] = marker
    for row in range(size):
        out[row, row] = marker
        out[row, size - 1 - row] = marker
    if stub:
        limit = min(3, center)
        lengths = [int(rng.integers(0, limit + 1)) for _ in RAYS]
        if not any(lengths):
            lengths[int(rng.integers(4))] = 1
        if all(length == center for length in lengths):
            lengths[int(rng.integers(4))] = 0
        for (dr, dc), length in zip(RAYS, lengths):
            for step in range(1, length + 1):
                inp[center + dr * step, center + dc * step] = marker
    return {"input": inp, "output": out}


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(v) for v in grid.flat)


def _inspect(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= n <= 30 for n in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert bool(np.all((grid >= 0) & (grid <= 9)))
    assert inp.shape == out.shape
    size = inp.shape[0]
    assert size in SIZES and inp.shape[1] == size
    center = size // 2
    marker = int(inp[center, center])
    diagonal = np.eye(size, dtype=bool) | np.fliplr(np.eye(size, dtype=bool))
    fields = np.unique(inp[~diagonal])
    assert len(fields) == 1
    field = int(fields[0])
    assert field != marker
    assert bool(np.all((inp == field) | (inp == marker)))
    for dr, dc in RAYS:
        seen_field = False
        for step in range(1, center + 1):
            value = int(inp[center + dr * step, center + dc * step])
            if value == field:
                seen_field = True
            else:
                assert value == marker and not seen_field
    expected = np.full((size, size), field, dtype=np.int8)
    expected[diagonal] = marker
    assert np.array_equal(out, expected) and not np.array_equal(inp, out)
    stub = bool(np.count_nonzero(inp == marker) > 1)
    assert stub or (size, field, marker) not in SOURCE
    return size, field, marker, stub


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    records = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            records[split].append(_inspect(pair))
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    train, test = records["train"], records["test"]
    assert len({r[0] for r in train}) >= 2
    assert any(r[0] >= 5 for r in train)
    assert not train[0][3]
    if any(r[3] for r in test):
        assert any(r[3] for r in train)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    a, b = _counts(seed, rng, num_train, num_test)
    mode = _pick(rng, ("extrapolate", "interpolate", "smaller", "same_size"))
    variable = bool(rng.random() < 0.6)
    swap = variable and bool(rng.random() < 0.4)
    overlap = bool(rng.random() < 0.6)
    stubs = bool(rng.random() < 0.45)
    train_sizes, test_sizes = _sizes(rng, a, b, mode)
    if stubs and train_sizes[1] == 3:
        train_sizes[0], train_sizes[1] = train_sizes[1], train_sizes[0]
    train_colors, test_colors = _colors(
        rng, train_sizes, test_sizes, variable, swap, overlap
    )
    puzzle = {"train": [], "test": []}
    for i, (size, (field, marker)) in enumerate(zip(train_sizes, train_colors)):
        stub = stubs and (i == 1 or (i > 1 and size >= 5 and rng.random() < 0.35))
        puzzle["train"].append(_pair(rng, size, field, marker, stub))
    for size, (field, marker) in zip(test_sizes, test_colors):
        stub = stubs and size >= 5 and bool(rng.random() < 0.7)
        puzzle["test"].append(_pair(rng, size, field, marker, stub))
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _predict_from_demonstrations(train, test_input):
    for pair in train:
        inp, out = pair["input"], pair["output"]
        size = inp.shape[0]
        center = size // 2
        diagonal = np.eye(size, dtype=bool) | np.fliplr(np.eye(size, dtype=bool))
        assert np.all(out[diagonal] == inp[center, center])
        assert np.array_equal(out[~diagonal], inp[~diagonal])
    size = test_input.shape[0]
    predicted = test_input.copy()
    indices = np.arange(size)
    marker = test_input[size // 2, size // 2]
    predicted[indices, indices] = marker
    predicted[indices, size - 1 - indices] = marker
    return predicted


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            predicted = _predict_from_demonstrations(
                [{"input": p["input"].copy(), "output": p["output"].copy()}
                 for p in puzzle["train"]],
                pair["input"].copy(),
            )
            assert np.array_equal(predicted, pair["output"])
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for seed, a, b in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                       (2001, None, 1), (2002, None, 4), (2003, 2, None),
                       (2004, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    assert _puzzle_key(generate_puzzle(271828, 8, 4)) == _puzzle_key(generate_puzzle(271828, 8, 4))
    for kw in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
               {"num_test": 5}, {"num_train": 2, "num_test": 2},
               {"num_train": 3, "num_test": 4}, {"num_train": 3.0},
               {"num_train": True}):
        try:
            generate_puzzle(3001, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid counts accepted: {kw}")

    records = [{split: [_inspect(pair) for pair in puzzle[split]]
                for split in ("train", "test")} for puzzle in puzzles]
    all_sizes = {r[0] for item in records for split in ("train", "test")
                 for r in item[split]}
    assert set(SIZES).issubset(all_sizes)
    assert any(r[0] == 9 for item in records for r in item["train"])
    assert any(r[0] == 9 for item in records for r in item["test"])
    assert sum(len({r[2] for r in item["train"]}) > 1 for item in records) >= 8
    assert sum(len({r[2] for r in item["train"] + item["test"]}) == 1
               for item in records) >= 8
    assert sum(bool({r[1] for r in item["train"]} & {r[1] for r in item["test"]})
               for item in records) >= 8
    assert sum(any(r[3] for r in item["test"]) for item in records) >= 5
    assert sum(any(r[3] for r in item["train"]) for item in records) >= 5
    assert sum(any((f, m) in {(x[2], x[1]) for x in item["train"]}
                   for _, f, m, _ in item["train"]) for item in records) >= 5
    assert sum(any(r[0] in {x[0] for x in item["train"]} for r in item["test"])
               for item in records) >= 5
    assert sum(any(min(x[0] for x in item["train"]) < r[0] < max(x[0] for x in item["train"])
                   for r in item["test"]) for item in records) >= 5
    assert sum(all(r[0] > max(x[0] for x in item["train"]) for r in item["test"])
               for item in records) >= 3
    assert sum(any(r[0] < min(x[0] for x in item["train"]) for r in item["test"])
               for item in records) >= 5
    assert any(
        len({r[2] for r in item["train"]}) > 1
        and bool({r[1] for r in item["train"]} & {r[1] for r in item["test"]})
        and any(r[3] for r in item["test"])
        for item in records
    )
