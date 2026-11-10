"""Solved ARC puzzles made from a solid square and a periodic lattice."""

import numpy as np


_CANVAS_SIZE = 15
_SIZES = (1, 2, 3, 4, 5, 6)
# These are the only source inputs whose outputs would also match when the
# background is zero. A different background changes the actual output.
_SOURCE_KEYS = frozenset({(2, 8), (3, 3), (4, 2), (5, 4)})


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _normalize_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _normalize_count("num_train", num_train, 2, 10)
    test = _normalize_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _sample_size_plan(
    rng: np.random.Generator, num_train: int, num_test: int
) -> tuple[list[int], list[int]]:
    """Pick distinct training geometries and distinct test geometries."""
    mode = int(rng.integers(0, 5))
    if 3 <= num_train <= 5 and num_test == 1:
        # Keep the source's simple next-size demonstration in the mix.
        return [2, 3, 4], [5]

    desired = min(6, max(2, num_test, num_train if num_train <= 3 else num_train - 1))
    if mode in (0, 1):
        targets = [size for size in (3, 4, 5, 6) if size - 1 >= min(desired, 5)]
        target = int(rng.choice(targets))
        available = list(range(1, target))
        count = min(desired, len(available))
        train_sizes = [target - 1]
        others = [size for size in available if size != target - 1]
        if count > 1:
            train_sizes.extend(int(x) for x in rng.choice(others, count - 1, replace=False))
        remaining = [size for size in _SIZES if size != target]
        test_sizes = [target]
        if num_test > 1:
            test_sizes.extend(int(x) for x in rng.choice(remaining, num_test - 1, replace=False))
    elif mode == 2:
        count = max(desired, num_test)
        train_sizes = [int(x) for x in rng.choice(_SIZES, count, replace=False)]
        test_sizes = [int(x) for x in rng.choice(train_sizes, num_test, replace=False)]
    elif mode == 3:
        targets = [size for size in (1, 2, 3, 4) if 6 - size >= min(desired, 5)]
        target = int(rng.choice(targets))
        available = [size for size in _SIZES if size > target]
        count = min(desired, len(available))
        train_sizes = [int(x) for x in rng.choice(available, count, replace=False)]
        remaining = [size for size in _SIZES if size != target]
        test_sizes = [target]
        if num_test > 1:
            test_sizes.extend(int(x) for x in rng.choice(remaining, num_test - 1, replace=False))
    else:
        train_sizes = [int(x) for x in rng.choice(_SIZES, desired, replace=False)]
        test_sizes = [int(x) for x in rng.choice(_SIZES, num_test, replace=False)]
    rng.shuffle(train_sizes)
    rng.shuffle(test_sizes)
    return train_sizes, test_sizes


def _source_copy(background: int, size: int, color: int) -> bool:
    return background == 0 and (size, color) in _SOURCE_KEYS


def _select_train_keys(
    rng: np.random.Generator, sizes: list[int], background: int, count: int
) -> list[tuple[int, int]]:
    palette = [color for color in range(10) if color != background]
    # Common colors make the deliberate color/size crosses possible without
    # accidentally reproducing a source pair in either position.
    common = [color for color in palette if all(not _source_copy(background, size, color) for size in sizes)]
    first, second = (int(x) for x in rng.choice(common, 2, replace=False))
    a, b = sizes[:2]
    if count == 2:
        keys = [(a, first), (b, second)]
    elif count == 3:
        keys = [(a, first), (b, second)]
        keys.append((sizes[2], first) if len(sizes) >= 3 else (a, second))
    elif len(sizes) >= 3:
        keys = [(a, first), (b, first), (sizes[2], second), (a, second)]
    else:
        keys = [(a, first), (b, first), (a, second), (b, second)]

    represented = {size for size, _ in keys}
    for size in sizes:
        if size not in represented:
            choices = [color for color in palette if not _source_copy(background, size, color)]
            keys.append((size, int(rng.choice(choices))))

    while len(keys) < count:
        size_counts = {size: sum(key[0] == size for key in keys) for size in sizes}
        least = min(size_counts.values())
        candidates = [
            (size, color)
            for size in sizes if size_counts[size] == least
            for color in palette
            if (size, color) not in keys and not _source_copy(background, size, color)
        ]
        keys.append(candidates[int(rng.integers(len(candidates)))])
    rng.shuffle(keys)
    return keys


def _select_test_keys(
    rng: np.random.Generator, sizes: list[int], background: int,
    train_keys: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    palette = [color for color in range(10) if color != background]
    trained = {color for _, color in train_keys}
    selected = []
    for size in sizes:
        available = [
            color for color in palette
            if (size, color) not in train_keys and not _source_copy(background, size, color)
        ]
        reused = [color for color in available if color in trained]
        novel = [color for color in available if color not in trained]
        prefer_reuse = bool(rng.integers(2))
        choices = (reused or novel) if prefer_reuse else (novel or reused)
        selected.append((size, int(rng.choice(choices))))
    return selected


def _construct_pair(size: int, color: int, background: int) -> dict:
    input_grid = np.full((size, size), color, dtype=np.int8)
    output_grid = np.full((_CANVAS_SIZE, _CANVAS_SIZE), background, dtype=np.int8)
    lines = np.arange(size, _CANVAS_SIZE, size + 1)
    output_grid[lines, :] = color
    output_grid[:, lines] = color
    return {"input": input_grid, "output": output_grid}


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    background = int(rng.integers(0, 10))
    train_sizes, test_sizes = _sample_size_plan(rng, train_count, test_count)
    train_keys = _select_train_keys(rng, train_sizes, background, train_count)
    test_keys = _select_test_keys(rng, test_sizes, background, train_keys)
    puzzle = {
        "train": [_construct_pair(size, color, background) for size, color in train_keys],
        "test": [_construct_pair(size, color, background) for size, color in test_keys],
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray, name: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{name} must be a NumPy array")
    _require(grid.ndim == 2, f"{name} must be two-dimensional")
    rows, columns = grid.shape
    _require(1 <= rows <= 30 and 1 <= columns <= 30, f"{name} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{name} must contain integers")
    _require(np.all((0 <= grid) & (grid <= 9)), f"{name} has an illegal ARC color")


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "illegal pair counts")
    backgrounds = set()
    seen_inputs, seen_pairs = set(), set()
    train_sizes, test_sizes, train_colors = [], [], set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            name = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{name} has invalid keys")
            input_grid, output_grid = pair["input"], pair["output"]
            _validate_grid(input_grid, f"{name} input")
            _validate_grid(output_grid, f"{name} output")
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, f"{name} repeats a pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            size = input_grid.shape[0]
            color = int(input_grid[0, 0])
            _require(input_grid.shape == (size, size) and size in _SIZES, f"{name} is not a supported square")
            _require(np.all(input_grid == color), f"{name} is not solid")
            _require(output_grid.shape == (_CANVAS_SIZE, _CANVAS_SIZE), f"{name} output must be 15 by 15")
            background = int(output_grid[0, 0])
            _require(background != color, f"{name} has no distinct background")
            backgrounds.add(background)
            _require(not _source_copy(background, size, color), f"{name} copies a source pair")
            for row in range(_CANVAS_SIZE):
                for column in range(_CANVAS_SIZE):
                    line = row % (size + 1) == size or column % (size + 1) == size
                    expected = color if line else background
                    _require(int(output_grid[row, column]) == expected, f"{name} violates the lattice at ({row}, {column})")
            if split == "train":
                train_sizes.append(size)
                train_colors.add(color)
            else:
                test_sizes.append(size)
    _require(len(backgrounds) == 1, "background changes within the puzzle")
    _require(len(set(train_sizes)) >= 2, "training must show two spacings")
    _require(len(train_colors) >= 2, "training must show two input colors")
    _require(len(set(test_sizes)) == len(test_sizes), "multi-test puzzles repeat a geometry")
    if train_count >= 4:
        train_keys = [(int(p["input"].shape[0]), int(p["input"][0, 0])) for p in puzzle["train"]]
        _require(any(sum(size == a for size, _ in train_keys) >= 2 for a, _ in train_keys), "training lacks a same-size color comparison")
        _require(any(sum(color == b for _, color in train_keys) >= 2 for _, b in train_keys), "training lacks a same-color size comparison")


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split]) for split in ("train", "test"))


def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # This check receives no test output or generation parameters.
    backgrounds = {int(pair["output"][0, 0]) for pair in train}
    _require(len(backgrounds) == 1, "training does not reveal one background")
    background = backgrounds.pop()
    size = test_input.shape[0]
    color = int(test_input[0, 0])
    side = train[0]["output"].shape[0]
    return np.array(
        [[color if row % (size + 1) == size or column % (size + 1) == size else background
          for column in range(side)] for row in range(side)], dtype=np.int8
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
            _require(np.array_equal(predicted, pair["output"]), "train-only prediction failed")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack range")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack range")
    _require(len({_puzzle_signature(p) for p in puzzles}) >= 40, "puzzles repeat")

    sizes = {split: {pair["input"].shape[0] for puzzle in puzzles for pair in puzzle[split]} for split in ("train", "test")}
    _require(sizes["train"] == set(_SIZES) and sizes["test"] == set(_SIZES), "size range collapsed")
    backgrounds = {int(puzzle["train"][0]["output"][0, 0]) for puzzle in puzzles}
    _require(len(backgrounds) >= 8, "background palette collapsed")
    _require(any(int(pair["input"][0, 0]) == 0 for puzzle in puzzles for pair in puzzle["train"] + puzzle["test"]), "zero never used as a line color")
    _require(any(len(puzzle["test"]) > 1 for puzzle in puzzles), "multi-test regime missing")
    _require(any(max(pair["input"].shape[0] for pair in p["test"]) > max(pair["input"].shape[0] for pair in p["train"]) for p in puzzles), "upward test shift missing")
    _require(any(min(pair["input"].shape[0] for pair in p["test"]) < min(pair["input"].shape[0] for pair in p["train"]) for p in puzzles), "downward test shift missing")
    _require(any({pair["input"].shape[0] for pair in p["test"]}.issubset({pair["input"].shape[0] for pair in p["train"]}) for p in puzzles), "within-range tests missing")
    _require(any((int(pair["input"][0, 0]) in {int(x["input"][0, 0]) for x in p["train"]}) for p in puzzles for pair in p["test"]), "reused test colors missing")
    _require(any((int(pair["input"][0, 0]) not in {int(x["input"][0, 0]) for x in p["train"]}) for p in puzzles for pair in p["test"]), "novel test colors missing")
    _require(any({pair["input"].shape[0] for pair in p["train"]} == {2, 3, 4} and [pair["input"].shape[0] for pair in p["test"]] == [5] for p in puzzles), "simple 2,3,4 to 5 regime missing")
    _require(all(len({pair["input"].shape[0] for pair in p["train"]}) >= 5 for p in puzzles if len(p["train"]) >= 6), "large training sets repeat too few spacings")
    _require(any(int(pair["input"][0, 0]) == 4 for p in puzzles for pair in p["test"]), "color 4 absent from tests")
    _require(any(int(p["train"][0]["output"][0, 0]) != 0 and any(int(pair["input"][0, 0]) == 0 for pair in p["train"] + p["test"]) for p in puzzles), "nonzero background and zero line never combine")

    for index, (train, test) in enumerate(((2, 1), (6, 3), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + index, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit counts ignored")
        validate_generated_puzzle(puzzle)
    for index, arguments in enumerate(({"num_train": None, "num_test": 1}, {"num_train": None, "num_test": 4}, {"num_train": 2, "num_test": None}, {"num_train": 10, "num_test": None})):
        puzzle = generate_puzzle(2000 + index, **arguments)
        validate_generated_puzzle(puzzle)
        for name, requested in arguments.items():
            if requested is not None:
                _require(len(puzzle["train" if name == "num_train" else "test"]) == requested, "partial count ignored")
    _require(_puzzle_signature(generate_puzzle(314159, 7, 3)) == _puzzle_signature(generate_puzzle(314159, 7, 3)), "explicit generation is nondeterministic")
    _require(_puzzle_signature(generate_puzzle(271828)) == _puzzle_signature(generate_puzzle(271828)), "automatic generation is nondeterministic")
    _require(_puzzle_signature(generate_puzzle(12345, 7, 3)) != _puzzle_signature(generate_puzzle(12346, 7, 3)), "neighboring seeds repeat")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": True}, {"num_test": False}):
        try:
            generate_puzzle(999, **arguments)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {arguments}")
