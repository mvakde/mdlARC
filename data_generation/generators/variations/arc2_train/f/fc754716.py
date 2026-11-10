"""Centered-cell to perimeter-frame ARC puzzle generator."""

import numpy as np


_ODD_SIZES = tuple(range(3, 30, 2))
_SOURCE_PAIR_SPECS = frozenset({
    (5, 7, 0, 1), (5, 3, 0, 3), (5, 5, 0, 6),
    (3, 3, 0, 2), (9, 7, 0, 8),
})
_SHIFT_MODES = ("source_like", "height_growth", "width_growth",
                "both_growth", "within_range")


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return result


def _normalize_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train_count = _normalize_count(num_train, "num_train", 2, 10)
    test_count = _normalize_count(num_test, "num_test", 1, 4)
    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must exceed num_test")
        return train_count, test_count
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif train_count is None:
        train_count = test_count + 1 + seed % (10 - test_count)
    else:
        test_count = 1 + seed % min(4, train_count - 1)
    return train_count, test_count


def _construct_pair(height: int, width: int, background: int, foreground: int) -> dict:
    inp = np.full((height, width), background, dtype=np.int8)
    inp[height // 2, width // 2] = foreground
    out = np.full((height, width), background, dtype=np.int8)
    out[0, :] = foreground
    out[-1, :] = foreground
    out[:, 0] = foreground
    out[:, -1] = foreground
    return {"input": inp, "output": out}


def _grid_signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        tuple((_grid_signature(pair["input"]), _grid_signature(pair["output"]))
              for pair in puzzle[regime])
        for regime in ("train", "test")
    )


def _roles(inp: np.ndarray) -> tuple[int, int]:
    height, width = inp.shape
    _ensure(height >= 3 and width >= 3 and height % 2 == width % 2 == 1,
            "input needs odd dimensions of at least 3 by 3")
    background = int(inp[0, 0])
    foreground = int(inp[height // 2, width // 2])
    _ensure(foreground != background, "the centered seed must differ from its background")
    expected = np.full(inp.shape, background, dtype=inp.dtype)
    expected[height // 2, width // 2] = foreground
    _ensure(np.array_equal(inp, expected),
            "input must have exactly one non-background cell at its center")
    return background, foreground


def _frame_from_roles(inp: np.ndarray, background: int, foreground: int) -> np.ndarray:
    expected = np.full(inp.shape, background, dtype=inp.dtype)
    expected[0, :] = foreground
    expected[-1, :] = foreground
    expected[:, 0] = foreground
    expected[:, -1] = foreground
    return expected


def _choose_size(rng: np.random.Generator, choices: tuple[int, ...]) -> int:
    return choices[int(rng.integers(len(choices)))]


def _train_shapes(rng: np.random.Generator, count: int, mode: str) -> list[tuple[int, int]]:
    if mode == "source_like":
        limit = 7
        first = (5, 7)
    else:
        limits = (9, 15, 23) if mode != "within_range" else (11, 19, 29)
        limit = _choose_size(rng, limits)
        roomy = tuple(size for size in _ODD_SIZES if 5 <= size <= limit)
        first = (_choose_size(rng, roomy), _choose_size(rng, roomy))
    available = tuple(size for size in _ODD_SIZES if size <= limit)
    shapes = [first]
    for index in range(1, count):
        if index >= 2 and rng.random() < 0.20:
            shape = shapes[int(rng.integers(len(shapes)))]
        else:
            shape = (_choose_size(rng, available), _choose_size(rng, available))
        if index == 1 and shape == first:
            alternatives = [(h, w) for h in available for w in available if (h, w) != first]
            shape = alternatives[int(rng.integers(len(alternatives)))]
        shapes.append(shape)
    return shapes


def _test_shapes(rng: np.random.Generator, count: int, mode: str,
                 train_shapes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    max_height = max(height for height, _ in train_shapes)
    max_width = max(width for _, width in train_shapes)
    low_heights = tuple(size for size in _ODD_SIZES if size <= max_height)
    low_widths = tuple(size for size in _ODD_SIZES if size <= max_width)
    high_heights = tuple(size for size in _ODD_SIZES if size > max_height)
    high_widths = tuple(size for size in _ODD_SIZES if size > max_width)
    shapes = []
    for index in range(count):
        if mode == "source_like" and index == 0:
            shape = (9, 7)
        elif mode == "height_growth":
            shape = (_choose_size(rng, high_heights), _choose_size(rng, low_widths))
        elif mode == "width_growth":
            shape = (_choose_size(rng, low_heights), _choose_size(rng, high_widths))
        elif mode == "both_growth":
            shape = (_choose_size(rng, high_heights), _choose_size(rng, high_widths))
        elif mode == "within_range":
            shape = (_choose_size(rng, low_heights), _choose_size(rng, low_widths))
        else:
            compact = tuple(size for size in _ODD_SIZES if size <= 15)
            shape = (_choose_size(rng, compact), _choose_size(rng, compact))
        shapes.append(shape)
    return shapes


def _pick_colors(rng: np.random.Generator, shape: tuple[int, int],
                 background: int, preferred: list[int],
                 used: set[tuple[int, int, int, int]],
                 forbidden: set[int] | None = None) -> tuple[int, int]:
    forbidden = forbidden or set()
    # A repeated shape could theoretically consume all nine foregrounds.
    # Other backgrounds provide a bounded fallback without changing geometry.
    alternate_backgrounds = [(background + offset) % 10 for offset in range(1, 10)]
    for candidate_background in [background] + alternate_backgrounds:
        candidates = [
            color for color in range(10)
            if color != candidate_background and color not in forbidden
            and (*shape, candidate_background, color) not in used
            and (*shape, candidate_background, color) not in _SOURCE_PAIR_SPECS
        ]
        if candidates:
            favored = [color for color in preferred if color in candidates]
            choices = favored if favored else candidates
            return candidate_background, choices[int(rng.integers(len(choices)))]
    raise AssertionError("all legal color assignments were consumed")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _normalize_seed(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)

    # The seed-to-frame rule is the only shared convention. The base palette
    # and dimension regime organize examples, without hiding any output rule.
    base_background = seed_value % 10
    vary_background = bool(rng.integers(2))
    mode = _SHIFT_MODES[int(rng.integers(len(_SHIFT_MODES)))]
    train_shapes = _train_shapes(rng, train_count, mode)
    test_shapes = _test_shapes(rng, test_count, mode, train_shapes)

    used: set[tuple[int, int, int, int]] = set()
    train_pairs = []
    train_foregrounds: list[int] = []
    for index, shape in enumerate(train_shapes):
        if index == 0 or not vary_background:
            background = base_background
        elif index == 1:
            alternatives = [color for color in range(10) if color != base_background]
            background = alternatives[int(rng.integers(len(alternatives)))]
        else:
            background = base_background if rng.random() < 0.5 else int(rng.integers(10))
        if index >= 2 and rng.random() < 0.45:
            preferred = train_foregrounds
        else:
            preferred = [color for color in range(10) if color not in train_foregrounds]
        forbidden = {train_foregrounds[0]} if index == 1 else set()
        background, foreground = _pick_colors(
            rng, shape, background, preferred, used, forbidden
        )
        used.add((*shape, background, foreground))
        train_foregrounds.append(foreground)
        train_pairs.append(_construct_pair(*shape, background, foreground))

    test_pairs = []
    for shape in test_shapes:
        background = (base_background if not vary_background or rng.random() < 0.5
                      else int(rng.integers(10)))
        if rng.random() < 0.50:
            preferred = train_foregrounds
        else:
            preferred = [color for color in range(10) if color not in train_foregrounds]
        background, foreground = _pick_colors(rng, shape, background, preferred, used)
        used.add((*shape, background, foreground))
        test_pairs.append(_construct_pair(*shape, background, foreground))

    train_pairs = [train_pairs[int(i)] for i in rng.permutation(train_count)]
    test_pairs = [test_pairs[int(i)] for i in rng.permutation(test_count)]
    puzzle = {"train": train_pairs, "test": test_pairs}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
            "puzzle must contain exactly train and test")
    _ensure(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
            "train and test must be lists")
    _ensure(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
            and len(puzzle["train"]) > len(puzzle["test"]), "invalid pair counts")
    seen_inputs = set()
    seen_pairs = set()
    train_shapes = set()
    train_foregrounds = set()
    has_roomy_training = False
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            _ensure(isinstance(pair, dict) and set(pair) == {"input", "output"},
                    "each pair must contain exactly input and output")
            for role in ("input", "output"):
                grid = pair[role]
                _ensure(isinstance(grid, np.ndarray) and grid.ndim == 2,
                        f"{role} must be a two-dimensional NumPy array")
                _ensure(all(1 <= size <= 30 for size in grid.shape),
                        f"{role} dimensions are out of bounds")
                _ensure(np.issubdtype(grid.dtype, np.integer)
                        and bool(np.all((grid >= 0) & (grid <= 9))),
                        f"{role} must contain integer ARC colors")
            inp, out = pair["input"], pair["output"]
            _ensure(inp.shape == out.shape, "the transformation must preserve dimensions")
            background, foreground = _roles(inp)
            _ensure(np.array_equal(out, _frame_from_roles(inp, background, foreground)),
                    "output must have the seed color on its full outer perimeter only")
            _ensure(not np.array_equal(inp, out), "input and output must differ")
            spec = (*inp.shape, background, foreground)
            _ensure(spec not in _SOURCE_PAIR_SPECS, "a pair copies the original task")
            input_key = _grid_signature(inp)
            pair_key = (input_key, _grid_signature(out))
            _ensure(input_key not in seen_inputs and pair_key not in seen_pairs,
                    "duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if regime == "train":
                train_shapes.add(inp.shape)
                train_foregrounds.add(foreground)
                has_roomy_training |= inp.shape[0] >= 5 and inp.shape[1] >= 5
    _ensure(has_roomy_training, "training needs a full two-dimensional interior")
    _ensure(len(train_shapes) >= 2, "training needs more than one grid shape")
    _ensure(len(train_foregrounds) >= 2, "training needs contrasting seed colors")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            background, foreground = _roles(inp)
            height, width = inp.shape
            border = np.zeros((height, width), dtype=bool)
            border[0, :] = border[-1, :] = True
            border[:, 0] = border[:, -1] = True
            _ensure(bool(np.all(out[border] == foreground))
                    and bool(np.all(out[~border] == background)),
                    "independent border-mask check failed")
    _ensure(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _ensure(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    _ensure(len({_puzzle_signature(p) for p in puzzles}) >= 40,
            "different seeds generated too few distinct puzzles")
    train_shapes = {pair["input"].shape for p in puzzles for pair in p["train"]}
    test_shapes = {pair["input"].shape for p in puzzles for pair in p["test"]}
    _ensure(len(train_shapes) >= 25 and len(test_shapes) >= 25,
            "height and width combinations lack diversity")
    _ensure(any(h > w for h, w in test_shapes) and any(h < w for h, w in test_shapes)
            and any(h == w for h, w in test_shapes), "test aspects lack diversity")
    _ensure(any(h == 9 and w == 7 for h, w in test_shapes)
            and any(w > 7 for _, w in test_shapes), "test shifts lack diversity")
    observed = {
        "height_growth": 0, "width_growth": 0, "both_growth": 0,
        "within_range": 0, "thin_test": 0, "roomy_test": 0,
        "varied_background": 0, "reused_test_color": 0,
        "new_test_color": 0, "same_size_new_color": 0,
        "same_color_new_size": 0, "source_like": 0,
    }
    for puzzle in puzzles:
        def spec(pair: dict) -> tuple[int, int, int, int]:
            inp = pair["input"]
            height, width = inp.shape
            return height, width, int(inp[0, 0]), int(inp[height // 2, width // 2])

        train = [spec(pair) for pair in puzzle["train"]]
        test = [spec(pair) for pair in puzzle["test"]]
        records = train + test
        maximum_height = max(item[0] for item in train)
        maximum_width = max(item[1] for item in train)
        observed["varied_background"] += len({item[2] for item in records}) > 1
        observed["reused_test_color"] += bool({item[3] for item in train}
                                               & {item[3] for item in test})
        observed["new_test_color"] += bool({item[3] for item in test}
                                            - {item[3] for item in train})
        observed["source_like"] += ((5, 7) in {item[:2] for item in train}
                                    and (9, 7) in {item[:2] for item in test})
        observed["same_size_new_color"] += any(
            first[:2] == second[:2] and first[3] != second[3]
            for index, first in enumerate(records) for second in records[index + 1:]
        )
        observed["same_color_new_size"] += any(
            first[:2] != second[:2] and first[3] == second[3]
            for index, first in enumerate(records) for second in records[index + 1:]
        )
        for height, width, _, _ in test:
            if height > maximum_height and width > maximum_width:
                observed["both_growth"] += 1
            elif height > maximum_height:
                observed["height_growth"] += 1
            elif width > maximum_width:
                observed["width_growth"] += 1
            else:
                observed["within_range"] += 1
            observed["thin_test"] += height == 3 or width == 3
            observed["roomy_test"] += height >= 5 and width >= 5
    for name, minimum in (
        ("height_growth", 5), ("width_growth", 5), ("both_growth", 5),
        ("within_range", 5), ("thin_test", 5), ("roomy_test", 20),
        ("varied_background", 5), ("reused_test_color", 5),
        ("new_test_color", 5), ("same_size_new_color", 5),
        ("same_color_new_size", 5), ("source_like", 2),
    ):
        _ensure(observed[name] >= minimum, f"review diversity missing: {name}")

    for seed, counts in ((1001, (7, 3)), (800, (2, 1)), (801, (10, 1)),
                         (802, (10, 4))):
        puzzle = generate_puzzle(seed, *counts)
        validate_generated_puzzle(puzzle)
        _ensure((len(puzzle["train"]), len(puzzle["test"])) == counts,
                "explicit counts were not honored")
        _ensure(_puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed, *counts)),
                "explicit generation is nondeterministic")
    for index, requested in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(900 + index, *requested)
        validate_generated_puzzle(puzzle)
        if requested[0] is not None:
            _ensure(len(puzzle["train"]) == requested[0], "partial train count changed")
        if requested[1] is not None:
            _ensure(len(puzzle["test"]) == requested[1], "partial test count changed")
    _ensure(_puzzle_signature(generate_puzzle(77)) == _puzzle_signature(generate_puzzle(77)),
            "automatic generation is nondeterministic")
    for arguments in ((1, None), (11, None), (None, 0), (None, 5),
                      (2, 2), (3, 4), (4, 4), (True, 1), (3, 1.5)):
        try:
            generate_puzzle(1000, *arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
