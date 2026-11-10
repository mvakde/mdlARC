"""Recognize one of four fixed shapes and return its demonstrated color label."""
import numpy as np


SHAPES = ("011011100", "010111010", "101010101", "110101010")
# Exact source pairs are excluded, while source geometry remains the task.
SOURCE_LABELS = (3, 6, 2, 1)
SOURCE_INKS = ((1, 8), (5, 8), (2, 5, 8), (4, 5, 7))
AUTO_COUNTS = ((2, 1), (3, 1), (4, 1), (4, 2), (5, 1), (5, 2),
               (6, 3), (7, 2), (8, 3), (9, 4), (10, 1), (10, 4))


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"invalid {name}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = None if num_train is None else _integer("num_train", num_train, 2, 10)
    test = None if num_test is None else _integer("num_test", num_test, 1, 4)
    if train is None and test is None:
        train, test = AUTO_COUNTS[seed % len(AUTO_COUNTS)]
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _pair(shape, ink, label):
    mask = np.array([int(v) for v in SHAPES[shape]], dtype=np.int8).reshape(3, 3)
    return {"input": mask * ink, "output": np.array([[label]], dtype=np.int8)}


def _facts(pair):
    grid, output = pair["input"], pair["output"]
    assert grid.shape == (3, 3) and output.shape == (1, 1)
    colors = set(int(v) for v in grid.flat)
    assert 0 in colors and len(colors) == 2
    ink = next(v for v in colors if v != 0)
    mask = "".join("1" if v else "0" for v in grid.flat)
    assert mask in SHAPES, "input must use an exact original shape"
    return SHAPES.index(mask), ink, int(output[0, 0])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer("seed", seed, 0)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    active_count = min(4, max(2, train_count // 2))
    active = [int(v) for v in rng.choice(4, active_count, replace=False)]
    labels = {shape: int(label) for shape, label in
              zip(active, rng.choice(np.arange(1, 10), active_count, replace=False))}
    available = {shape: [ink for ink in range(1, 10)
                        if labels[shape] != SOURCE_LABELS[shape] or ink not in SOURCE_INKS[shape]]
                 for shape in active}
    # Shared ink across shapes makes color-to-label lookup insufficient.
    common = sorted(set.intersection(*(set(available[shape]) for shape in active)))
    common_ink = common[int(rng.integers(len(common)))]
    train = []
    schedule = [active[i % active_count] for i in range(train_count)]
    for i, shape in enumerate(schedule):
        choices = available[shape]
        ink = common_ink if i < active_count else choices[int(rng.integers(len(choices)))]
        choices.remove(ink)
        train.append(_pair(shape, ink, labels[shape]))
    rng.shuffle(train)
    # Choose from feasible unused shape/ink pairs, avoiding retry dead ends.
    candidates = [(shape, ink) for shape in active for ink in available[shape]]
    indices = rng.choice(len(candidates), test_count, replace=False)
    test = [_pair(*candidates[int(i)], labels[candidates[int(i)][0]]) for i in indices]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    mapping, inks = {}, {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert grid.size and int(grid.min()) >= 0 and int(grid.max()) <= 9
            shape, ink, label = _facts(pair)
            assert 1 <= ink <= 9 and 1 <= label <= 9
            assert label != SOURCE_LABELS[shape] or ink not in SOURCE_INKS[shape]
            key = tuple(int(v) for v in pair["input"].flat)
            assert key not in seen
            seen.add(key)
            if split == "train":
                assert shape not in mapping or mapping[shape] == label
                mapping[shape] = label
                inks.setdefault(shape, set()).add(ink)
            else:
                assert shape in mapping and mapping[shape] == label
                assert ink not in inks[shape]
    assert len(mapping) >= 2 and len(set(mapping.values())) == len(mapping)
    assert any(sum(ink in colors for colors in inks.values()) >= 2 for ink in range(1, 10))
    if len(puzzle["train"]) >= 3:
        assert any(len(colors) >= 2 for colors in inks.values())


def _key(puzzle):
    return tuple(tuple((tuple(int(v) for v in p["input"].flat), int(p["output"][0, 0]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert {_facts(pair)[0] for p in puzzles for split in p.values() for pair in split} == set(range(4))
    codebooks = set()
    for puzzle in puzzles:
        # Predict from normalized training masks alone, ignoring all input ink.
        lookup = {tuple(int(v != 0) for v in pair["input"].flat): int(pair["output"][0, 0])
                  for pair in puzzle["train"]}
        codebooks.add(tuple(sorted(lookup.items())))
        for pair in puzzle["test"]:
            assert lookup[tuple(int(v != 0) for v in pair["input"].flat)] == int(pair["output"][0, 0])
    assert len(codebooks) >= 30
    assert _key(generate_puzzle(129)) == _key(generate_puzzle(129))
    assert _key(generate_puzzle(129, 9, 4)) == _key(generate_puzzle(129, 9, 4))
    assert _key(generate_puzzle(12345)) != _key(generate_puzzle(12346))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(9000 + 10 * train + test, train, test)
            assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(9001, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for seed, train, test in ((0, 1, None), (0, 11, None), (0, None, 0), (0, None, 5),
                              (0, 2, 2), (0, 3, 4), (-1, None, None), (True, None, None)):
        try:
            generate_puzzle(seed, train, test)
        except ValueError:
            continue
        raise AssertionError("invalid arguments accepted")
