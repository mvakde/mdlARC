"""Whole-tile stamping puzzles selected by the input's unique rarest color."""

import numpy as np


_SOURCE_INPUTS = (
    ((3, 2, 7), (2, 2, 7), (5, 5, 7)),
    ((8, 5, 5), (8, 8, 8), (5, 9, 9)),
    ((7, 1, 7), (1, 7, 7), (7, 1, 7)),
    ((1, 6, 6), (5, 1, 6), (5, 5, 5)),
    ((9, 9, 6), (3, 8, 8), (8, 3, 3)),
    ((4, 4, 2), (2, 2, 2), (2, 4, 2)),
    ((9, 7, 9), (9, 9, 7), (7, 9, 7)),
)
_SHAPES = tuple((h, w) for h in range(2, 6) for w in range(2, 6))
_RECTANGLES = tuple(shape for shape in _SHAPES if shape[0] != shape[1])
_MASK_STYLES = ("square", "row_plus", "symmetric", "empty_band", "cluster")


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _normalise_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a nonnegative integer")
    return result


def _normalise_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _normalise_count(num_train, "num_train", 2, 10)
    test = _normalise_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    return int(train), int(test)


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _minority(grid: np.ndarray) -> tuple[int, int]:
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) >= 2, "input must have at least two colors")
    smallest = int(counts.min())
    indices = np.flatnonzero(counts == smallest)
    _require(len(indices) == 1, "input must have a unique least-frequent color")
    return int(colors[int(indices[0])]), smallest


def _construct_output(tile: np.ndarray, fill: int) -> np.ndarray:
    selector, _ = _minority(tile)
    h, w = tile.shape
    output = np.full((h * h, w * w), fill, dtype=np.int8)
    for row, col in zip(*np.nonzero(tile == selector)):
        output[row * h:(row + 1) * h, col * w:(col + 1) * w] = tile
    return output


def _dihedral(mask: np.ndarray, code: int) -> np.ndarray:
    transformed = np.rot90(mask, code % 4)
    if code >= 4:
        transformed = np.fliplr(transformed)
    return transformed


def _connected(mask: np.ndarray) -> bool:
    cells = set(zip(*np.nonzero(mask)))
    if not cells:
        return False
    reached = {next(iter(cells))}
    fringe = list(reached)
    while fringe:
        row, col = fringe.pop()
        for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if neighbor in cells and neighbor not in reached:
                reached.add(neighbor)
                fringe.append(neighbor)
    return len(reached) == len(cells)


def _mask_matches(mask: np.ndarray, style: str) -> bool:
    h, w = mask.shape
    if style == "any":
        return True
    if style == "asymmetric":
        return all(not np.array_equal(mask, _dihedral(mask, code)) for code in range(1, 8))
    if style == "square":
        return any(np.all(mask[r:r + 2, c:c + 2]) for r in range(h - 1) for c in range(w - 1))
    if style == "row_plus":
        return bool(np.any(np.all(mask, axis=1)) and int(mask.sum()) == w + 1)
    if style == "symmetric":
        return any(np.array_equal(mask, _dihedral(mask, code)) for code in range(1, 8))
    if style == "empty_band":
        return bool(np.any(~np.any(mask, axis=0)) or np.any(~np.any(mask, axis=1)))
    if style == "cluster":
        return _connected(mask)
    raise ValueError(f"unknown mask style: {style}")


def _three_by_three_masks(count: int, style: str) -> list[np.ndarray]:
    candidates = []
    for bits in range(1 << 9):
        if bits.bit_count() != count:
            continue
        mask = np.asarray([(bits >> index) & 1 for index in range(9)], dtype=bool).reshape(3, 3)
        if _mask_matches(mask, style):
            candidates.append(mask)
    return candidates


def _sample_mask(rng: np.random.Generator, shape: tuple[int, int], count: int, style: str) -> np.ndarray:
    h, w = shape
    if shape == (3, 3):
        candidates = _three_by_three_masks(count, style)
        if not candidates:
            raise ValueError("no mask with the requested 3-by-3 geometry")
        return candidates[int(rng.integers(len(candidates)))].copy()

    if style != "any":
        raise ValueError("special mask styles are used only for 3-by-3 inputs")
    mask = np.zeros(shape, dtype=bool)
    mode = int(rng.integers(4))
    if mode == 1:
        first = (int(rng.integers(h)), int(rng.integers(w)))
        chosen = {first}
        while len(chosen) < count:
            frontier = {
                (nr, nc)
                for row, col in chosen
                for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in chosen
            }
            choices = sorted(frontier)
            chosen.add(choices[int(rng.integers(len(choices)))])
        for row, col in chosen:
            mask[row, col] = True
    else:
        allowed = np.arange(h * w)
        if mode == 2 and count <= (h - 1) * w:
            excluded_row = int(rng.integers(h))
            allowed = np.asarray([p for p in allowed if p // w != excluded_row])
        elif mode == 3 and count <= h * (w - 1):
            excluded_col = int(rng.integers(w))
            allowed = np.asarray([p for p in allowed if p % w != excluded_col])
        mask.flat[rng.choice(allowed, size=count, replace=False)] = True
    return mask


def _max_minority(shape: tuple[int, int]) -> int:
    return (shape[0] * shape[1] - 1) // 2


def _max_palette(shape: tuple[int, int], minority_count: int) -> int:
    cells = shape[0] * shape[1]
    return min(8, 1 + (cells - minority_count) // (minority_count + 1))


def _choose_palette(rng: np.random.Generator, shape: tuple[int, int], count: int,
                    minimum: int = 2, fixed: int | None = None) -> int:
    maximum = _max_palette(shape, count)
    if fixed is not None:
        if not minimum <= fixed <= maximum:
            raise ValueError("incompatible palette size")
        return fixed
    return int(rng.integers(minimum, maximum + 1))


def _is_source_pair(tile: np.ndarray, output: np.ndarray) -> bool:
    for values in _SOURCE_INPUTS:
        source = np.asarray(values, dtype=np.int8)
        if np.array_equal(tile, source) and np.array_equal(output, _construct_output(source, 0)):
            return True
    return False


def _make_input(rng: np.random.Generator, shape: tuple[int, int], count: int,
                palette: int, selector: int, required_common: tuple[int, ...],
                style: str, fill: int, used: set[tuple]) -> np.ndarray:
    required = tuple(dict.fromkeys(color for color in required_common if color != selector))
    _require(len(required) <= palette - 1, "too many required ordinary colors")
    candidates = [color for color in range(10) if color != selector and color not in required]

    def assemble(mask: np.ndarray, common: list[int], shuffle: bool) -> np.ndarray:
        common_count = len(common)
        counts = np.full(common_count, count + 1, dtype=int)
        extra = shape[0] * shape[1] - count - int(counts.sum())
        if shuffle:
            counts += rng.multinomial(extra, np.full(common_count, 1 / common_count))
        else:
            counts[0] += extra
        values = np.repeat(np.asarray(common, dtype=np.int8), counts)
        if shuffle:
            rng.shuffle(values)
        grid = np.empty(shape, dtype=np.int8)
        grid[mask] = selector
        grid[~mask] = values
        return grid

    for _ in range(96):
        mask = _sample_mask(rng, shape, count, style)
        remaining = rng.choice(candidates, size=palette - 1 - len(required), replace=False)
        common = list(required) + [int(x) for x in remaining]
        grid = assemble(mask, common, True)
        if _grid_key(grid) not in used and not _is_source_pair(grid, _construct_output(grid, fill)):
            return grid

    # A deterministic fallback avoids depending on a lucky retry when small
    # shape/color spaces have already been used within a large puzzle.
    if shape == (3, 3):
        masks = _three_by_three_masks(count, style)
    else:
        masks = []
        for offset in range(shape[0] * shape[1]):
            mask = np.zeros(shape, dtype=bool)
            mask.flat[(np.arange(count) + offset) % mask.size] = True
            masks.append(mask)
    for mask in masks:
        for offset in range(len(candidates)):
            ordered = candidates[offset:] + candidates[:offset]
            common = list(required) + ordered[:palette - 1 - len(required)]
            grid = assemble(mask, common, False)
            if _grid_key(grid) not in used and not _is_source_pair(grid, _construct_output(grid, fill)):
                return grid
    raise RuntimeError("could not construct a distinct input")


def _sample_shape(rng: np.random.Generator, rectangular: bool = False) -> tuple[int, int]:
    shapes = _RECTANGLES if rectangular else _SHAPES
    return shapes[int(rng.integers(len(shapes)))]


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _normalise_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # This output-only convention is shared by every pair and is visible in
    # every training output's uniform, unselected macro-blocks.
    fill = int(rng.integers(10))
    selector_order = [int(x) for x in rng.permutation(10)]
    source_like = seed % 5 == 0
    used: set[tuple] = set()
    train: list[dict] = []
    test: list[dict] = []

    def add(split: list[dict], shape: tuple[int, int], count: int, selector: int,
            style: str = "any", palette: int | None = None,
            minimum_palette: int = 2, required_common: tuple[int, ...] = ()) -> None:
        chosen_palette = _choose_palette(rng, shape, count, minimum_palette, palette)
        tile = _make_input(rng, shape, count, chosen_palette, selector,
                           required_common, style, fill, used)
        used.add(_grid_key(tile))
        split.append({"input": tile, "output": _construct_output(tile, fill)})

    # A 3-by-3 asymmetric mask settles direct macro-coordinate placement for
    # all dihedral alternatives. Other masks can freely be symmetric.
    witness_count = 3 if source_like else (2, 3, 4)[int(rng.integers(3))]
    add(train, (3, 3), witness_count, selector_order[0], "asymmetric")

    second_shape = (3, 3) if source_like else _sample_shape(rng, rectangular=True)
    common = [selector_order[0]]
    if seed % 2 == 0 and fill not in (selector_order[1], selector_order[0]):
        common.append(fill)
    add(train, second_shape, 1, selector_order[1], minimum_palette=3,
        required_common=tuple(common))

    for index in range(2, train_count):
        shape = (3, 3) if source_like else _sample_shape(rng)
        highest = min(3, _max_minority(shape)) if source_like else _max_minority(shape)
        count = int(rng.integers(1, highest + 1))
        add(train, shape, count, selector_order[index])

    ordinary_colors = sorted({int(color) for pair in train
                              for color in np.unique(pair["input"])
                              if int(color) != _minority(pair["input"])[0]})

    for index in range(test_count):
        special = index == 0 and seed % 5 in (0, 1)
        shape = (3, 3) if source_like or special else _sample_shape(rng)
        maximum = _max_minority(shape)
        if source_like or special:
            count = 4
            style = _MASK_STYLES[(seed // 5) % len(_MASK_STYLES)]
        else:
            style = "any"
            if index == 0 and seed % 3 == 0:
                count = 1
            elif index == 0 and seed % 3 == 1:
                matching = [int(_minority(pair["input"])[1]) for pair in train
                            if int(_minority(pair["input"])[1]) <= maximum]
                count = matching[int(rng.integers(len(matching)))]
            else:
                count = int(rng.integers(1, maximum + 1))
        selector = ordinary_colors[int(rng.integers(len(ordinary_colors)))]
        add(test, shape, count, selector, style,
            palette=2 if source_like else None)

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _fill_from_pair(pair: dict) -> int:
    tile, output = pair["input"], pair["output"]
    selector, _ = _minority(tile)
    h, w = tile.shape
    empty_positions = np.argwhere(tile != selector)
    _require(len(empty_positions) > 0, "training must show unused macro-blocks")
    row, col = (int(x) for x in empty_positions[0])
    block = output[row * h:(row + 1) * h, col * w:(col + 1) * w]
    _require(np.all(block == block[0, 0]), "unused block is not uniform")
    return int(block[0, 0])


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    all_pairs = puzzle["train"] + puzzle["test"]
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    for pair in all_pairs:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "each pair must contain exactly input and output")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                     "grids must be two-dimensional NumPy arrays")
            _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
            _require(all(1 <= size <= 30 for size in grid.shape), "grid size out of bounds")
            _require(0 <= int(grid.min()) and int(grid.max()) <= 9, "grid color out of bounds")
        tile, output = pair["input"], pair["output"]
        h, w = tile.shape
        _require((h, w) in _SHAPES and output.shape == (h * h, w * w),
                 "invalid self-copy dimensions")
        _require(tile.dtype == output.dtype, "input and output dtypes differ")
        _minority(tile)
        input_key = _grid_key(tile)
        pair_key = (input_key, _grid_key(output))
        _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                 "duplicate input or pair")
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
        _require(not _is_source_pair(tile, output), "exact source pair copied")

    fill = _fill_from_pair(puzzle["train"][0])
    for pair in all_pairs:
        tile, output = pair["input"], pair["output"]
        selector, _ = _minority(tile)
        h, w = tile.shape
        for row in range(h):
            for col in range(w):
                block = output[row * h:(row + 1) * h, col * w:(col + 1) * w]
                if int(tile[row, col]) == selector:
                    _require(np.array_equal(block, tile), "selected block is not an exact tile copy")
                else:
                    _require(np.all(block == fill), "empty blocks disagree on the puzzle fill")
        _require(set(int(x) for x in np.unique(output)) ==
                 set(int(x) for x in np.unique(tile)) | {fill},
                 "output palette violates the stamping rule")

    train_selectors = {_minority(pair["input"])[0] for pair in puzzle["train"]}
    _require(len(train_selectors) >= 2, "training must vary selector color")
    _require(any(len(np.unique(pair["input"])) >= 3 for pair in puzzle["train"]),
             "training must demonstrate a multicolor tile")
    ordinary_train = {int(color) for pair in puzzle["train"]
                      for color in np.unique(pair["input"])
                      if int(color) != _minority(pair["input"])[0]}
    for pair in puzzle["test"]:
        _require(_minority(pair["input"])[0] in ordinary_train,
                 "test selector role was not demonstrated as ordinary in training")

    train_masks = [pair["input"] == _minority(pair["input"])[0] for pair in puzzle["train"]]
    test_masks = [pair["input"] == _minority(pair["input"])[0] for pair in puzzle["test"]]
    for code in range(1, 8):
        indistinguishable = all(np.array_equal(mask, _dihedral(mask, code)) for mask in train_masks)
        if indistinguishable:
            _require(all(np.array_equal(mask, _dihedral(mask, code)) for mask in test_masks),
                     "training leaves a test-relevant mask orientation unresolved")

    if any(pair["input"].shape[0] != pair["input"].shape[1] for pair in puzzle["test"]):
        _require(any(pair["input"].shape[0] != pair["input"].shape[1]
                     for pair in puzzle["train"]),
                 "rectangular tests require a rectangular training witness")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((split, tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                                for pair in puzzle[split])) for split in ("train", "test"))


def _expect_value_error(**kwargs: int | None) -> None:
    try:
        generate_puzzle(123, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kwargs!r}")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42,
             "automatic seeds repeat whole puzzles")
    _require(len({len(puzzle["train"]) for puzzle in puzzles}) >= 4,
             "too few automatic train counts")
    _require(len({len(puzzle["test"]) for puzzle in puzzles}) >= 3,
             "too few automatic test counts")

    shapes: set[tuple[int, int]] = set()
    layouts: set[tuple] = set()
    palettes: set[tuple[int, ...]] = set()
    fills: set[int] = set()
    test_styles: set[str] = set()
    training_four = 0
    test_multicolor = 0
    five_color = 0
    zero_selector = 0
    zero_ordinary = 0
    fill_in_tile = 0
    mixed_dimensions = 0
    matched_count = 0
    smaller_count = 0
    larger_count = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        fill = _fill_from_pair(puzzle["train"][0])
        fills.add(fill)
        train_counts = {_minority(pair["input"])[1] for pair in puzzle["train"]}
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        if len(train_shapes) > 1:
            mixed_dimensions += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                tile, output = pair["input"], pair["output"]
                selector, count = _minority(tile)
                mask = tile == selector
                shapes.add(tile.shape)
                layouts.add((tile.shape, tuple(int(x) for x in mask.flat)))
                colors = tuple(int(x) for x in np.unique(tile))
                palettes.add(colors)
                five_color += len(colors) >= 5
                zero_selector += selector == 0
                zero_ordinary += 0 in colors and selector != 0
                fill_in_tile += fill in colors

                # An independent vectorized construction checks the block
                # placement against ground truth, including overlapping colors.
                h, w = tile.shape
                expanded_mask = np.repeat(np.repeat(mask, h, axis=0), w, axis=1)
                expected = np.where(expanded_mask, np.tile(tile, (h, w)), fill)
                _require(np.array_equal(output, expected), "self-test stamping mismatch")
                _require(not _is_source_pair(tile, output), "self-test source copy")
                if split == "train":
                    training_four += count == 4
                else:
                    test_multicolor += len(colors) >= 3
                    matched_count += count in train_counts
                    smaller_count += count < max(train_counts)
                    larger_count += count > max(train_counts)
                    if tile.shape == (3, 3) and count == 4:
                        test_styles.update(style for style in _MASK_STYLES if _mask_matches(mask, style))

        # Predict each test from copied demonstrations and its input. No seed,
        # generation metadata, or test output is used to construct a prediction.
        inferred_fill = _fill_from_pair(puzzle["train"][0])
        _require(all(_fill_from_pair(pair) == inferred_fill for pair in puzzle["train"]),
                 "inconsistent training fill")
        for pair in puzzle["test"]:
            tile = pair["input"]
            selector, _ = _minority(tile)
            h, w = tile.shape
            positions = np.repeat(np.repeat(tile == selector, h, axis=0), w, axis=1)
            prediction = np.where(positions, np.tile(tile, (h, w)), inferred_fill)
            _require(np.array_equal(prediction, pair["output"]),
                     "training-only prediction disagrees with test output")

    _require(len(shapes) >= 12 and (5, 5) in shapes and (2, 2) in shapes,
             "insufficient shape diversity")
    _require(mixed_dimensions >= 25, "too few puzzles vary dimensions within training")
    _require(len(layouts) >= 150 and len(palettes) >= 70,
             "insufficient layout or palette diversity")
    _require(len(fills) >= 8, "insufficient puzzle-level fill diversity")
    _require(set(_MASK_STYLES) <= test_styles, "test geometry styles are missing")
    _require(training_four > 0 and test_multicolor > 0 and five_color > 0,
             "copy-count or palette review cases are missing")
    _require(zero_selector > 0 and zero_ordinary > 0 and fill_in_tile > 0,
             "zero or overlapping fill review cases are missing")
    _require(matched_count > 0 and smaller_count > 0 and larger_count > 0,
             "train/test copy-count regimes are missing")

    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(2468 + train + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit pair counts were ignored")
        validate_generated_puzzle(puzzle)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(8642, train, test))
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 4, "num_test": 4}):
        _expect_value_error(**kwargs)
    _require(_puzzle_key(generate_puzzle(7777)) == _puzzle_key(generate_puzzle(7777)),
             "automatic generation is nondeterministic")
    _require(_puzzle_key(generate_puzzle(7777, 10, 4)) ==
             _puzzle_key(generate_puzzle(7777, 10, 4)),
             "explicit generation is nondeterministic")
