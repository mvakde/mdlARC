"""Blank-canvas ARC spirals with a convention learned from demonstrations."""

import hashlib

import numpy as np


_SOURCE_SIZES = frozenset((6, 8, 10, 13, 15, 18))
# SHA-256 of the six original foreground masks, in side-length order.
_SOURCE_MASK_HASHES = {
    6: "b8e07a1c2c3959900bdfb70a30737aaedbd7aaafbb37250863bf6803dac263c4",
    8: "04f73365b17c902b91b553b4b763890ee2f6c4789a68629ae7ed01276ef9a5dc",
    10: "82a3d367b32d73febc4906afce0173200cfc7a5c4bf1d47936f874b505242497",
    13: "100e2ad3a7658ab0e30441e9864fe6cc2f05eed1198a062724d5250015598901",
    15: "b1b94830e2ba6528ad8a56957cba152855d41c084cb1a6a19db9797ffb56947e",
    18: "5f17e9f28e72738927469b8af9ee39a581e6329b467a794763445fc18051aefd",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _seed_value(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return value


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _pair_counts(seed: int, rng: np.random.Generator,
                 num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _walk_mask(height: int, width: int) -> np.ndarray:
    """Construction used for generation: visit each run cell by cell."""
    mask = np.zeros((height, width), dtype=bool)
    row = column = 0
    mask[row, column] = True
    for _ in range(width - 1):
        column += 1
        mask[row, column] = True
    directions = ((1, 0), (0, -1), (-1, 0), (0, 1))
    for run in range(60):
        length = ((height if run % 2 == 0 else width)
                  - 1 - 2 * (run // 2))
        if length <= 0:
            break
        dr, dc = directions[run % 4]
        for _ in range(length):
            row += dr
            column += dc
            mask[row, column] = True
    return mask


def _slice_mask(height: int, width: int) -> np.ndarray:
    """Independent boundary construction used to inspect returned grids."""
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    for ring in range(30):
        vertical = height - 1 - 2 * ring
        if vertical <= 0:
            break
        if ring % 2 == 0:
            mask[ring + 1:height - ring, width - 1 - ring] = True
        else:
            mask[ring + 1:height - ring, ring - 1] = True
        horizontal = width - 1 - 2 * ring
        if horizontal <= 0:
            break
        if ring % 2 == 0:
            mask[height - 1 - ring, ring:width - 1 - ring] = True
        else:
            mask[ring + 1, ring:width - 1 - ring] = True
    return mask


def _oriented_mask(shape: tuple[int, int], orientation: int, builder) -> np.ndarray:
    turns = orientation % 4
    base_shape = shape[::-1] if turns % 2 else shape
    mask = builder(*base_shape)
    if orientation >= 4:
        mask = np.fliplr(mask)
    return np.rot90(mask, turns)


def _sample_scales(rng: np.random.Generator, train_count: int,
                   test_count: int, regime: int) -> tuple[list[int], list[int]]:
    if regime == 0:  # extrapolate beyond the training range
        if train_count <= 8 and rng.random() < 0.38:
            low_range, high_range, train_pool = range(5, 8), range(8, 13), range(5, 13)
        else:
            low_range, high_range, train_pool = range(5, 11), range(13, 20), range(5, 21)
    elif regime == 1:  # test smaller sizes
        low_range, high_range, train_pool = range(12, 18), range(21, 28), range(12, 31)
    elif regime == 2:  # unseen sizes interleaved with demonstrations
        low_range, high_range, train_pool = range(5, 11), range(22, 29), range(5, 31)
    else:  # test both sides of the demonstrated interval when possible
        low_range, high_range, train_pool = range(9, 15), range(18, 24), range(9, 24)
    low = int(rng.choice(list(low_range)))
    high_candidates = [v for v in high_range if v % 2 != low % 2]
    high = int(rng.choice(high_candidates))
    remaining = [v for v in train_pool if v not in (low, high)]
    extra = rng.choice(remaining, size=train_count - 2, replace=False).tolist()
    train = [low, high] + [int(v) for v in extra]
    rng.shuffle(train)
    used = set(train)
    minimum, maximum = min(train), max(train)
    if regime == 0:
        candidates = [v for v in range(maximum + 1, 31) if v not in used]
        test = rng.choice(candidates, size=test_count, replace=False).tolist()
    elif regime == 1:
        candidates = [v for v in range(5, minimum) if v not in used]
        test = rng.choice(candidates, size=test_count, replace=False).tolist()
    elif regime == 2:
        candidates = [v for v in range(minimum + 1, maximum) if v not in used]
        test = rng.choice(candidates, size=test_count, replace=False).tolist()
    else:
        below = [v for v in range(5, minimum) if v not in used]
        above = [v for v in range(maximum + 1, 31) if v not in used]
        if test_count == 1:
            test = [int(rng.choice(below if int(rng.integers(2)) else above))]
        else:
            test = [int(rng.choice(below)), int(rng.choice(above))]
            remaining = [v for v in range(5, 31) if v not in used and v not in test]
            test.extend(int(v) for v in rng.choice(
                remaining, size=test_count - 2, replace=False))
            rng.shuffle(test)
    return train, [int(v) for v in test]


def _sample_shapes(rng: np.random.Generator, scales: list[int],
                   train_count: int, rectangular: bool) -> list[tuple[int, int]]:
    flags = [False] * len(scales)
    if rectangular:
        possible = [i for i in range(train_count) if scales[i] >= 7]
        forced_rectangle = int(rng.choice(possible))
        forced_square = (forced_rectangle + 1) % train_count
        for i, scale in enumerate(scales):
            if scale >= 7 and i != forced_square:
                flags[i] = i == forced_rectangle or bool(rng.integers(2))
    shapes = []
    for scale, flag in zip(scales, flags):
        if flag:
            difference = int(rng.integers(1, min(8, scale - 5) + 1))
            shape = (scale, scale - difference)
            if bool(rng.integers(2)):
                shape = shape[::-1]
        else:
            shape = (scale, scale)
        shapes.append(shape)
    return shapes


def _allowed_backgrounds(shape: tuple[int, int], foreground: int,
                         orientation: int) -> list[int]:
    choices = [v for v in range(10) if v != foreground]
    if foreground == 3 and orientation == 0 and shape[0] == shape[1] \
            and shape[0] in _SOURCE_SIZES:
        choices.remove(0)
    return choices


def _sample_backgrounds(rng: np.random.Generator,
                        shapes: list[tuple[int, int]], train_count: int,
                        foreground: int, orientation: int,
                        mixed: bool) -> list[int]:
    allowed = [_allowed_backgrounds(shape, foreground, orientation)
               for shape in shapes]
    if not mixed:
        shared = set(allowed[0])
        for choices in allowed[1:]:
            shared.intersection_update(choices)
        background = int(rng.choice(sorted(shared)))
        return [background] * len(shapes)
    backgrounds = [int(rng.choice(choices)) for choices in allowed]
    second = [v for v in allowed[1] if v != backgrounds[0]]
    backgrounds[1] = int(rng.choice(second))
    return backgrounds


def _pair(shape: tuple[int, int], background: int,
          foreground: int, orientation: int) -> dict:
    input_grid = np.full(shape, background, dtype=np.int8)
    output_grid = input_grid.copy()
    output_grid[_oriented_mask(shape, orientation, _walk_mask)] = foreground
    return {"input": input_grid, "output": output_grid}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _seed_value(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _pair_counts(seed, rng, num_train, num_test)
    orientation = int(rng.integers(8))
    foreground = int(rng.integers(10))
    mixed = bool(rng.random() < 0.65)
    rectangular = bool(rng.random() < 0.58)
    regime = int(rng.integers(4))
    if seed % 11 == 0:
        orientation = 0
        mixed = False
        rectangular = False
    train_scales, test_scales = _sample_scales(rng, train_count, test_count, regime)
    shapes = _sample_shapes(rng, train_scales + test_scales, train_count, rectangular)
    backgrounds = _sample_backgrounds(
        rng, shapes, train_count, foreground, orientation, mixed)
    pairs = [_pair(shape, background, foreground, orientation)
             for shape, background in zip(shapes, backgrounds)]
    puzzle = {"train": pairs[:train_count], "test": pairs[train_count:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(tuple((_grid_signature(p["input"]), _grid_signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _source_copy(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    shape = input_grid.shape
    return (shape[0] == shape[1] and shape[0] in _SOURCE_SIZES
            and np.all(input_grid == 0)
            and np.array_equal(output_grid,
                               np.where(_slice_mask(*shape), 3, 0)))


def _validate(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "expected exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    foregrounds = set()
    train_orientations = set(range(8))
    train_shapes = []
    test_shapes = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label}: incorrect pair schema")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label}.{name}: expected 2D NumPy grid")
                _require(all(1 <= v <= 30 for v in grid.shape)
                         and np.issubdtype(grid.dtype, np.integer)
                         and bool(np.all((0 <= grid) & (grid <= 9))),
                         f"{label}.{name}: invalid dimensions, dtype, or color")
            inp, out = pair["input"], pair["output"]
            shape = inp.shape
            _require(shape == out.shape and min(shape) >= 5,
                     f"{label}: incompatible canvas")
            _require(np.unique(inp).size == 1, f"{label}: input is not blank")
            background = int(inp[0, 0])
            colors = {int(v) for v in np.unique(out)}
            _require(background in colors and len(colors) == 2,
                     f"{label}: output must retain background and add one color")
            foreground = next(iter(colors - {background}))
            foregrounds.add(foreground)
            candidates = {o for o in range(8) if np.array_equal(
                out != background, _oriented_mask(shape, o, _slice_mask))}
            _require(bool(candidates), f"{label}: spiral geometry is invalid")
            if split == "train":
                train_orientations.intersection_update(candidates)
                train_shapes.append(shape)
            else:
                test_shapes.append(shape)
            input_key = _grid_signature(inp)
            pair_key = (input_key, _grid_signature(out))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     f"{label}: duplicate input or pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not _source_copy(inp, out), f"{label}: exact source pair")
            _require(not np.array_equal(inp, out), f"{label}: unchanged output")
    _require(len(foregrounds) == 1, "foreground convention changes across pairs")
    _require(len(train_orientations) == 1,
             "training does not determine one orientation")
    orientation = next(iter(train_orientations))
    for pair in puzzle["test"]:
        shape = pair["input"].shape
        background = int(pair["input"][0, 0])
        _require(np.array_equal(pair["output"] != background,
                                _oriented_mask(shape, orientation, _slice_mask)),
                 "test orientation differs from training")
    train_scales = [max(shape) for shape in train_shapes]
    test_scales = [max(shape) for shape in test_shapes]
    _require(len(set(train_scales + test_scales)) == train_count + test_count,
             "pair scales must be distinct")
    _require({v % 2 for v in train_scales} == {0, 1},
             "training must show both scale parities")
    _require(len(set(train_shapes)) > 1, "training sizes do not vary")
    if any(h != w for h, w in test_shapes):
        _require(any(h != w for h, w in train_shapes),
                 "rectangular test has no rectangular demonstration")


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed puzzle") from exc


def _expect_value_error(seed: int, train: int | None, test: int | None) -> None:
    try:
        generate_puzzle(seed, train, test)
    except ValueError:
        return
    raise AssertionError("illegal count did not raise ValueError")


def _predict_from_demonstrations(train_pairs: list[dict],
                                 test_input: np.ndarray) -> np.ndarray:
    foregrounds = set()
    orientations = set(range(8))
    for pair in train_pairs:
        background = int(pair["input"][0, 0])
        foregrounds.update({int(v) for v in np.unique(pair["output"])}
                           - {background})
        orientations.intersection_update({
            option for option in range(8)
            if np.array_equal(pair["output"] != background,
                              _oriented_mask(pair["input"].shape,
                                             option, _slice_mask))
        })
    _require(len(foregrounds) == 1 and len(orientations) == 1,
             "training convention cannot be inferred")
    answer = test_input.copy()
    answer[_oriented_mask(test_input.shape, next(iter(orientations)),
                          _slice_mask)] = next(iter(foregrounds))
    return answer


def self_test() -> None:
    # The saved solved task is checked against the independent construction
    # with literal reference hashes, including the intentional even center.
    for size, expected_hash in _SOURCE_MASK_HASHES.items():
        mask = _slice_mask(size, size).astype(np.uint8)
        _require(hashlib.sha256(mask.tobytes()).hexdigest() == expected_hash,
                 f"original {size}x{size} output differs")
        _require(np.array_equal(mask, _walk_mask(size, size)),
                 "construction methods disagree on original size")
    for height in range(5, 31):
        for width in range(5, 31):
            _require(np.array_equal(_walk_mask(height, width),
                                    _slice_mask(height, width)),
                     f"construction methods disagree at {height}x{width}")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        demonstrations = [
            {"input": pair["input"].copy(), "output": pair["output"].copy()}
            for pair in puzzle["train"]
        ]
        for pair in puzzle["test"]:
            _require(np.array_equal(
                _predict_from_demonstrations(demonstrations,
                                             pair["input"].copy()),
                pair["output"]), "training-only prediction differs")
    _require(len({_puzzle_signature(p) for p in puzzles}) >= 40,
             "too few distinct puzzles")
    _require(len({len(p["train"]) for p in puzzles}) >= 4
             and len({len(p["test"]) for p in puzzles}) >= 3,
             "automatic pair counts lack diversity")
    orientations = set()
    foregrounds = set()
    scales = set()
    residues = set()
    structures = set()
    regimes = set()
    constant_backgrounds = 0
    mixed_backgrounds = 0
    rectangular_puzzles = 0
    rectangular_height_parities = set()
    rectangular_width_parities = set()
    combined_variations = 0
    shallow_training_puzzles = 0
    for puzzle in puzzles:
        pairs = puzzle["train"] + puzzle["test"]
        backgrounds = {int(p["input"][0, 0]) for p in pairs}
        if len(backgrounds) == 1:
            constant_backgrounds += 1
        else:
            mixed_backgrounds += 1
            _require(len({int(p["input"][0, 0]) for p in puzzle["train"]}) >= 2,
                     "mixed backgrounds absent from training")
        if any(p["input"].shape[0] != p["input"].shape[1] for p in pairs):
            rectangular_puzzles += 1
            _require(any(p["input"].shape[0] != p["input"].shape[1]
                         for p in puzzle["train"]), "unseen rectangular convention")
        foreground = next(iter({int(v) for v in np.unique(pairs[0]["output"])}
                               - {int(pairs[0]["input"][0, 0])}))
        foregrounds.add(foreground)
        first = puzzle["train"][0]
        shape = first["input"].shape
        bg = int(first["input"][0, 0])
        orientations.add(next(o for o in range(8) if np.array_equal(
            first["output"] != bg, _oriented_mask(shape, o, _slice_mask))))
        if len(backgrounds) > 1 and any(
                p["input"].shape[0] != p["input"].shape[1] for p in pairs
        ) and not np.array_equal(first["output"] != bg,
                                 _slice_mask(*shape)):
            combined_variations += 1
        train_scales = [max(p["input"].shape) for p in puzzle["train"]]
        test_scales = [max(p["input"].shape) for p in puzzle["test"]]
        if max(train_scales) < 13:
            shallow_training_puzzles += 1
        if min(test_scales) > max(train_scales):
            regimes.add("larger")
        elif max(test_scales) < min(train_scales):
            regimes.add("smaller")
        elif all(min(train_scales) < v < max(train_scales) for v in test_scales):
            regimes.add("interleaved")
        else:
            regimes.add("mixed")
        for pair in pairs:
            shape = pair["input"].shape
            scale = max(shape)
            scales.add(scale)
            residues.add(scale % 4)
            bg = int(pair["input"][0, 0])
            structures.add((shape, (pair["output"] != bg).tobytes()))
            if shape[0] != shape[1]:
                rectangular_height_parities.add(shape[0] % 2)
                rectangular_width_parities.add(shape[1] % 2)
    _require(len(orientations) == 8 and len(foregrounds) == 10,
             "orientation or foreground coverage is incomplete")
    _require(constant_backgrounds >= 3 and mixed_backgrounds >= 8,
             "background modes lack coverage")
    _require(rectangular_puzzles >= 8, "rectangular coverage is too narrow")
    _require(rectangular_height_parities == {0, 1}
             and rectangular_width_parities == {0, 1},
             "rectangular dimensions do not cover both parities")
    _require(combined_variations >= 5,
             "orientation, backgrounds, and shape seldom combine")
    _require(shallow_training_puzzles >= 1,
             "all training sets still require a deep spiral")
    _require(len(scales) >= 24 and residues == {0, 1, 2, 3}
             and len(structures) >= 100, "structural diversity is too narrow")
    _require(regimes == {"larger", "smaller", "interleaved", "mixed"},
             "train/test regimes lack coverage")
    original_palette = generate_puzzle(24)
    _require(any(int(pair["input"][0, 0]) == 0
                 and 3 in pair["output"]
                 and not _source_copy(pair["input"], pair["output"])
                 for pair in original_palette["train"] + original_palette["test"]),
             "the original palette assignment was excluded")
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (104, 5, 2), (105, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        _require(len(puzzle["train"]) == train and len(puzzle["test"]) == test,
                 "explicit counts ignored")
    for seed, train, test in ((201, None, 1), (202, None, 4),
                              (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _require((train is None or len(puzzle["train"]) == train)
                 and (test is None or len(puzzle["test"]) == test),
                 "partial count ignored")
    _require(_puzzle_signature(generate_puzzle(987654, 7, 3))
             == _puzzle_signature(generate_puzzle(987654, 7, 3)),
             "explicit puzzle is nondeterministic")
    _require(_puzzle_signature(generate_puzzle(7654321))
             == _puzzle_signature(generate_puzzle(7654321)),
             "automatic puzzle is nondeterministic")
    for seed, train, test in ((301, 1, None), (302, 11, None),
                              (303, None, 0), (304, None, 5),
                              (305, 2, 2), (306, 3, 4),
                              (307, 4, 4), (308, 2, 4),
                              (309, True, 1), (310, 3, False)):
        _expect_value_error(seed, train, test)
