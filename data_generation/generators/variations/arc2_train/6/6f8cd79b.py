"""Generate blank canvases whose one-cell perimeter is recolored."""

import numpy as np


_SOURCE_SHAPES = {(5, 4), (4, 3), (5, 6), (3, 3), (7, 6)}
_SHAPES = tuple((h, w) for h in range(3, 31) for w in range(3, 31))


def _count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    lower, upper = (2, 10) if name == "num_train" else (1, 4)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    num_train = _count(num_train, "num_train")
    num_test = _count(num_test, "num_test")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        # Short sets of contrasting frames are most common; larger sets still
        # appear regularly and cover every legal test-count boundary.
        choices = (
            (2, 1), (3, 1), (3, 2), (4, 1), (4, 2),
            (4, 3), (5, 1), (5, 2), (5, 3), (5, 4),
            (3, 1), (4, 2), (5, 2), (5, 3),
            (6, 1), (6, 3), (7, 2), (8, 4), (9, 3), (10, 4),
        )
        return choices[seed % len(choices)]
    if num_train is None:
        choices = list(range(num_test + 1, 11))
        return choices[seed % len(choices)], num_test
    choices = list(range(1, min(4, num_train - 1) + 1))
    return num_train, choices[seed % len(choices)]


def _aspect(shape: tuple[int, int]) -> int:
    h, w = shape
    return (w > h) - (w < h)


def _shape_weight(shape: tuple[int, int]) -> float:
    h, w = shape
    weight = 1.0
    if h == 3 or w == 3:
        weight += 2.5
    if h == w:
        weight += 5.0
    if h >= 15 and w >= 15:
        weight += 1.0
    return weight


def _choose_shape(rng: np.random.Generator, candidates: list[tuple[int, int]]) -> tuple[int, int]:
    if not candidates:
        raise AssertionError("shape construction exhausted")
    weights = np.asarray([_shape_weight(shape) for shape in candidates], dtype=float)
    weights /= weights.sum()
    return candidates[int(rng.choice(len(candidates), p=weights))]


def _frame(input_grid: np.ndarray, border: int) -> np.ndarray:
    output = input_grid.copy()
    output[0, :] = border
    output[-1, :] = border
    output[:, 0] = border
    output[:, -1] = border
    return output


def _source_copy(shape: tuple[int, int], background: int, border: int) -> bool:
    return shape in _SOURCE_SHAPES and background == 0 and border == 8


def _pair(shape: tuple[int, int], background: int, border: int) -> dict:
    input_grid = np.full(shape, background, dtype=np.int8)
    return {"input": input_grid, "output": _frame(input_grid, border)}


def _candidate_shapes(used: set[tuple[int, int]], background: int, border: int) -> list[tuple[int, int]]:
    return [shape for shape in _SHAPES if shape not in used and not _source_copy(shape, background, border)]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(int(seed), num_train, num_test)

    # Every training output shows the shared border. Variable-background puzzles
    # show in two demonstrations that the input supplies the interior color.
    border = int(rng.integers(10))
    colors = [color for color in range(10) if color != border]
    variable_background = bool(rng.integers(2))
    fixed_background = int(rng.choice(colors))
    if variable_background:
        first_backgrounds = [int(value) for value in rng.choice(colors, size=2, replace=False)]
    else:
        first_backgrounds = [fixed_background, fixed_background]

    used: set[tuple[int, int]] = set()
    train: list[dict] = []
    for index in range(num_train):
        background = (
            first_backgrounds[index] if index < 2
            else int(rng.choice(colors)) if variable_background
            else fixed_background
        )
        candidates = _candidate_shapes(used, background, border)
        if index == 1:
            first_shape = train[0]["input"].shape
            candidates = [
                shape for shape in candidates
                if shape[0] != first_shape[0]
                and shape[1] != first_shape[1]
                and _aspect(shape) != _aspect(first_shape)
            ]
        shape = _choose_shape(rng, candidates)
        used.add(shape)
        train.append(_pair(shape, background, border))

    train_heights = {pair["input"].shape[0] for pair in train}
    train_widths = {pair["input"].shape[1] for pair in train}
    max_height, max_width = max(train_heights), max(train_widths)
    demonstrated_backgrounds = {int(pair["input"][0, 0]) for pair in train}

    test: list[dict] = []
    for _ in range(num_test):
        if variable_background:
            unseen = [color for color in colors if color not in demonstrated_backgrounds]
            palette = unseen if unseen and rng.random() < 0.4 else colors
            background = int(rng.choice(palette))
        else:
            background = fixed_background

        candidates = _candidate_shapes(used, background, border)
        regimes = {
            "height": [shape for shape in candidates if shape[0] > max_height and shape[1] <= max_width],
            "width": [shape for shape in candidates if shape[1] > max_width and shape[0] <= max_height],
            "both": [shape for shape in candidates if shape[0] > max_height and shape[1] > max_width],
            "neither": [shape for shape in candidates if shape[0] in train_heights and shape[1] in train_widths],
        }
        available = [name for name, shapes in regimes.items() if shapes]
        regime = available[int(rng.integers(len(available)))]
        shape = _choose_shape(rng, regimes[regime])
        used.add(shape)
        test.append(_pair(shape, background, border))

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)

    seen_inputs, seen_pairs = set(), set()
    borders = set()
    shapes = []
    for pairs in (train, test):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 3 <= grid.shape[0] <= 30 and 3 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert input_grid.shape == output_grid.shape
            background, border = int(input_grid[0, 0]), int(output_grid[0, 0])
            assert background != border and np.all(input_grid == background)
            assert np.array_equal(output_grid, _frame(input_grid, border))
            assert not _source_copy(input_grid.shape, background, border)
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            borders.add(border)
            shapes.append(input_grid.shape)

    assert len(borders) == 1
    train_shapes = [pair["input"].shape for pair in train]
    assert len({h for h, _ in train_shapes}) >= 2
    assert len({w for _, w in train_shapes}) >= 2
    assert len({_aspect(shape) for shape in train_shapes}) >= 2
    train_heights = {h for h, _ in train_shapes}
    train_widths = {w for _, w in train_shapes}
    max_height, max_width = max(train_heights), max(train_widths)
    train_backgrounds = {int(pair["input"][0, 0]) for pair in train}
    for pair in test:
        height, width = pair["input"].shape
        if len(train_backgrounds) == 1:
            assert int(pair["input"][0, 0]) in train_backgrounds
        assert (
            height > max_height or width > max_width
            or (height in train_heights and width in train_widths)
        )
    assert len(set(shapes)) == len(shapes)


def _fingerprint(puzzle: dict) -> tuple:
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    """Use displayed pairs and the blank test canvas alone to infer an answer."""
    learned = {int(pair["output"][0, 0]) for pair in train}
    assert len(learned) == 1
    border = learned.pop()
    assert all(np.all(pair["input"] == pair["input"][0, 0]) for pair in train)
    prediction = test_input.copy()
    height, width = prediction.shape
    for row in range(height):
        for column in range(width):
            if row in (0, height - 1) or column in (0, width - 1):
                prediction[row, column] = border
    return prediction


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(puzzle) for puzzle in puzzles}) >= 40
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert len({pair["input"].shape for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]}) >= 100
    assert len({int(puzzle["train"][0]["output"][0, 0]) for puzzle in puzzles}) >= 7
    assert len({
        (int(pair["input"][0, 0]), int(pair["output"][0, 0]))
        for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]
    }) >= 30

    regimes, aspects = set(), set()
    fixed, variable = 0, 0
    height_three = width_three = square = broad = 0
    height_thirty = width_thirty = 0
    same_first_aspect = opposite_first_aspect = unseen_test_background = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = puzzle["train"]
        backgrounds = {int(pair["input"][0, 0]) for pair in train}
        fixed += len(backgrounds) == 1
        variable += len(backgrounds) > 1
        heights = {pair["input"].shape[0] for pair in train}
        widths = {pair["input"].shape[1] for pair in train}
        first_aspect = _aspect(train[0]["input"].shape)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                h, w = inp.shape
                height_three += h == 3
                width_three += w == 3
                square += h == w
                broad += h >= 15 and w >= 15
                height_thirty += h == 30
                width_thirty += w == 30
                aspects.add(_aspect((h, w)))
                perimeter = np.zeros((h, w), dtype=bool)
                perimeter[0, :] = perimeter[-1, :] = True
                perimeter[:, 0] = perimeter[:, -1] = True
                assert np.all(out[perimeter] == out[0, 0])
                assert np.array_equal(out[~perimeter], inp[~perimeter])
            if split == "test":
                for pair in puzzle[split]:
                    h, w = pair["input"].shape
                    regimes.add((h > max(heights), w > max(widths)))
                    test_aspect = _aspect((h, w))
                    same_first_aspect += test_aspect == first_aspect
                    opposite_first_aspect += test_aspect * first_aspect == -1
                    unseen_test_background += int(pair["input"][0, 0]) not in backgrounds
                    assert np.array_equal(_predict_from_training(train, pair["input"].copy()), pair["output"])
    assert fixed >= 8 and variable >= 8
    assert regimes == {(False, False), (True, False), (False, True), (True, True)}
    assert aspects == {-1, 0, 1}
    assert height_three >= 10 and width_three >= 10
    assert square >= 8 and broad >= 25
    assert height_thirty >= 5 and width_thirty >= 5
    assert same_first_aspect >= 5 and opposite_first_aspect >= 5
    assert unseen_test_background >= 5
    assert sum(len(puzzle["train"]) <= 5 for puzzle in puzzles) > len(puzzles) // 2
    assert any(len(puzzle["train"]) >= 8 for puzzle in puzzles)

    # These reached examples guard against reviving whole-palette or
    # whole-shape bans under the guise of source-pair de-duplication.
    source_palette = generate_puzzle(2)
    assert any(
        int(pair["input"][0, 0]) == 0
        and int(pair["output"][0, 0]) == 8
        and pair["input"].shape not in _SOURCE_SHAPES
        for split in ("train", "test") for pair in source_palette[split]
    )
    recolored_source_shape = generate_puzzle(198)
    assert any(
        pair["input"].shape == (7, 6)
        and (int(pair["input"][0, 0]), int(pair["output"][0, 0])) != (0, 8)
        for split in ("train", "test") for pair in recolored_source_shape[split]
    )

    for seed, counts in ((314159, (2, 1)), (271828, (10, 1)), (161803, (10, 4)), (42, (6, 3))):
        puzzle = generate_puzzle(seed, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed, *counts))
    for index, kwargs in enumerate((
        {"num_train": None, "num_test": 1},
        {"num_train": None, "num_test": 4},
        {"num_train": 2, "num_test": None},
        {"num_train": 10, "num_test": None},
    )):
        puzzle = generate_puzzle(2000 + index, **kwargs)
        validate_generated_puzzle(puzzle)
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": True}, {"num_train": 3.0},
    ):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
