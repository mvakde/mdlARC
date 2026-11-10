"""Generate puzzles that append each grid's horizontal reflection."""

import numpy as np


_SOURCE_INPUTS = (
    ((1, 1, 1), (8, 1, 6), (6, 8, 8)),
    ((6, 8, 1), (6, 1, 1), (1, 1, 6)),
    ((1, 1, 1), (1, 6, 6), (6, 6, 6)),
    ((6, 6, 6), (1, 6, 1), (8, 8, 6)),
    ((6, 8, 6), (8, 6, 8), (1, 6, 1)),
)


def _signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(value) for value in grid.flat))


_SOURCE_SIGNATURES = {
    _signature(np.asarray(rows, dtype=np.int8)) for rows in _SOURCE_INPUTS
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng: np.random.Generator,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        train = int(rng.integers(2, 11))
        test = int(rng.integers(1, min(4, train - 1) + 1))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _colors(rng: np.random.Generator, pool: tuple[int, ...], size: int) -> tuple[int, ...]:
    indices = rng.choice(len(pool), size=min(size, len(pool)), replace=False)
    return tuple(pool[int(index)] for index in indices)


def _shape(rng: np.random.Generator, compact: bool,
           previous: list[tuple[int, int]], force_compact: bool) -> tuple[int, int]:
    if force_compact or (compact and rng.random() < 0.58):
        return 3, 3
    if previous and rng.random() < 0.27:
        return previous[int(rng.integers(len(previous)))]
    if rng.random() < 0.12:
        return 3, 3
    height_band = int(rng.choice(3, p=(0.35, 0.40, 0.25)))
    width_band = int(rng.choice(3, p=(0.38, 0.34, 0.28)))
    height_range = ((1, 6), (7, 18), (19, 30))[height_band]
    width_range = ((1, 5), (6, 10), (11, 15))[width_band]
    return (int(rng.integers(height_range[0], height_range[1] + 1)),
            int(rng.integers(width_range[0], width_range[1] + 1)))


def _asymmetric_count(rng: np.random.Generator, height: int,
                      width: int, witness: bool, all_palindromic: bool) -> int:
    if width == 1 or all_palindromic:
        return 0
    choices = (0, 1, max(1, height // 2), height)
    count = int(choices[int(rng.choice(4, p=(0.19, 0.27, 0.31, 0.23)))])
    return max(1, count) if witness else count


def _base_grid(rng: np.random.Generator, height: int, width: int,
               colors: tuple[int, ...], style: int) -> np.ndarray:
    if style == 0:  # independent cells
        return np.asarray(rng.choice(colors, size=(height, width)), dtype=np.int8)
    if style == 1:  # repeated rows
        template_count = min(height, int(rng.integers(1, 4)))
        templates = np.asarray(rng.choice(colors, size=(template_count, width)), dtype=np.int8)
        indices = rng.integers(0, template_count, size=height)
        return templates[indices].copy()
    if style == 2:  # horizontal color bands
        grid = np.empty((height, width), dtype=np.int8)
        band_height = int(rng.integers(1, min(6, height) + 1))
        for row in range(height):
            if row % band_height == 0:
                grid[row, :] = colors[int(rng.integers(len(colors)))]
            else:
                grid[row, :] = grid[row - 1, :]
        return grid
    if style == 3:  # vertical stripes
        stripe_width = int(rng.integers(1, min(5, width) + 1))
        stripe = [colors[int(rng.integers(len(colors)))]
                  for _ in range((width + stripe_width - 1) // stripe_width)]
        row = np.asarray([stripe[column // stripe_width] for column in range(width)], dtype=np.int8)
        grid = np.tile(row, (height, 1))
        if height > 2 and len(colors) > 1 and rng.random() < 0.45:
            grid[int(rng.integers(height)), int(rng.integers(width))] = colors[-1]
        return grid
    if style == 4:  # rectangles on a plain field
        grid = np.full((height, width), colors[0], dtype=np.int8)
        for _ in range(int(rng.integers(1, 6))):
            top = int(rng.integers(height))
            left = int(rng.integers(width))
            bottom = int(rng.integers(top + 1, height + 1))
            right = int(rng.integers(left + 1, width + 1))
            grid[top:bottom, left:right] = colors[int(rng.integers(len(colors)))]
        return grid
    if style == 5:  # sparse accent cells
        grid = np.full((height, width), colors[0], dtype=np.int8)
        if len(colors) > 1:
            count = max(1, int(height * width * float(rng.uniform(0.04, 0.24))))
            positions = rng.choice(height * width, size=count, replace=False)
            for position in positions:
                grid.flat[int(position)] = colors[int(rng.integers(1, len(colors)))]
        return grid
    if style == 6:  # short, source-like row vocabulary
        grid = np.empty((height, width), dtype=np.int8)
        for row_index in range(height):
            a, b, c = (colors[int(rng.integers(len(colors)))] for _ in range(3))
            pattern = ((a, a, a), (a, b, a), (a, a, b),
                       (a, b, b), (a, b, c))[int(rng.integers(5))]
            grid[row_index] = [pattern[column % 3] for column in range(width)]
        return grid
    # Periodic columns, sometimes with a changed row phase.
    grid = np.empty((height, width), dtype=np.int8)
    period = int(rng.integers(2, min(5, width + 1))) if width > 1 else 1
    pattern = [colors[int(rng.integers(len(colors)))] for _ in range(period)]
    for row_index in range(height):
        phase = int(rng.integers(period)) if rng.random() < 0.45 else 0
        grid[row_index] = [pattern[(column + phase) % period] for column in range(width)]
    return grid


def _impose_symmetry(rng: np.random.Generator, grid: np.ndarray,
                     asym_count: int, colors: tuple[int, ...]) -> np.ndarray:
    height, width = grid.shape
    asym_rows = set(int(i) for i in rng.choice(height, size=asym_count, replace=False)) if asym_count else set()
    for row_index in range(height):
        row = grid[row_index]
        if row_index not in asym_rows:
            for column in range(width // 2):
                row[-column - 1] = row[column]
        elif np.array_equal(row, row[::-1]):
            candidates = [color for color in colors if color != int(row[0])]
            if not candidates:
                candidates = [color for color in range(10) if color != int(row[0])]
            row[-1] = candidates[int(rng.integers(len(candidates)))]
    return grid


def _make_input(rng: np.random.Generator, shape: tuple[int, int],
                colors: tuple[int, ...], asym_count: int, style: int,
                required_color: int | None, require_all_colors: bool) -> np.ndarray:
    height, width = shape
    grid = _base_grid(rng, height, width, colors, style)
    if require_all_colors:
        half_width = (width + 1) // 2
        for index, color in enumerate(colors):
            grid[index // half_width, index % half_width] = color
    if required_color is not None:
        grid[0, 0] = required_color
    return _impose_symmetry(rng, grid, asym_count, colors)


def _transform(grid: np.ndarray) -> np.ndarray:
    return np.concatenate((grid, np.flip(grid, axis=1)), axis=1)


def _unique_fallback(pair_index: int, colors: tuple[int, ...],
                     required_color: int | None, witness: bool,
                     seen: set[tuple], require_all_colors: bool) -> np.ndarray:
    """Encode a pair in a distinct tall grid if a tiny pattern exhausted its options."""
    first = required_color if required_color is not None else colors[0]
    second = next((color for color in colors if color != first), (first + 1) % 10)
    for code in range(16):
        grid = np.full((16 + pair_index, 3), first, dtype=np.int8)
        for bit in range(4):
            if code & (1 << bit):
                grid[bit, 1] = second
        if require_all_colors:
            for index, color in enumerate(colors):
                grid[4 + index, 1] = color
        if witness:
            grid[-1, -1] = second
        if _signature(grid) not in seen:
            return grid
    raise AssertionError("distinct fallback construction failed")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)

    # Color relationships are content choices, not part of the output rule.
    palette_mode = int(rng.integers(3))  # common, varied, varied with unseen test color
    common_size = 3 if rng.random() < 0.44 else int(rng.integers(2, 6))
    common_palette = _colors(rng, tuple(range(10)), common_size)
    train_pool_size = int(rng.integers(3, 7))
    train_pool = _colors(rng, tuple(range(10)), train_pool_size)
    unseen_pool = tuple(color for color in range(10) if color not in train_pool)
    compact = bool(rng.random() < 0.38)
    original_shift = bool(rng.random() < 0.34)

    puzzle = {"train": [], "test": []}
    seen: set[tuple] = set()
    previous_shapes: list[tuple[int, int]] = []
    pair_seeds = [int(value) for value in rng.integers(0, 2**63, size=train_count + test_count)]
    for pair_index in range(train_count + test_count):
        pair_rng = np.random.default_rng(pair_seeds[pair_index])
        is_train = pair_index < train_count
        local_index = pair_index if is_train else pair_index - train_count
        split = "train" if is_train else "test"
        force_compact = compact and (pair_index in (0, 1, train_count))
        shape = _shape(pair_rng, compact, previous_shapes, force_compact)
        if is_train and local_index == 0 and shape[1] == 1:
            shape = (shape[0], 3)
        if is_train and local_index == 0 and shape[0] == 1:
            shape = (2, shape[1])
        if palette_mode == 0:
            colors = common_palette
            required_color = None
        elif is_train:
            pool = train_pool if palette_mode == 2 else tuple(range(10))
            size = int(pair_rng.integers(2 if local_index == 0 else 1, min(6, len(pool)) + 1))
            colors = _colors(pair_rng, pool, size)
            required_color = None
        elif palette_mode == 2:
            new_color = unseen_pool[int(pair_rng.integers(len(unseen_pool)))]
            other_pool = tuple(color for color in range(10) if color != new_color)
            size = int(pair_rng.integers(1, 6))
            colors = (new_color,) + _colors(pair_rng, other_pool, size - 1)
            required_color = new_color
        else:
            size = int(pair_rng.integers(1, 7))
            colors = _colors(pair_rng, tuple(range(10)), size)
            required_color = None
        if palette_mode == 0:
            available_per_row = (shape[1] + 1) // 2
            minimum_height = (len(colors) + available_per_row - 1) // available_per_row
            shape = (max(shape[0], minimum_height), shape[1])
        witness = is_train and local_index == 0
        asym_count = _asymmetric_count(pair_rng, shape[0], shape[1], witness,
                                       all_palindromic=(not is_train and original_shift))
        if len(colors) == 1:
            asym_count = 0
        style = int(pair_rng.integers(8))
        for attempt in range(80):
            grid = _make_input(pair_rng, shape, colors, asym_count, style,
                               required_color, palette_mode == 0)
            signature = _signature(grid)
            distinct_rule = (
                not witness or (
                    not np.array_equal(np.flip(grid, axis=1), grid)
                    and not np.array_equal(np.flip(grid, axis=1), np.flip(grid, axis=0))
                    and not np.array_equal(grid, np.flip(grid, axis=0))
                )
            )
            if signature not in seen and signature not in _SOURCE_SIGNATURES and distinct_rule:
                break
            if attempt == 39:
                shape = (min(30, shape[0] + 1), shape[1])
                asym_count = _asymmetric_count(pair_rng, shape[0], shape[1], witness,
                                               all_palindromic=(not is_train and original_shift))
                if len(colors) == 1:
                    asym_count = 0
        else:
            fallback_colors = colors
            if palette_mode == 2 and is_train and len(colors) == 1:
                fallback_colors = (colors[0], next(color for color in train_pool
                                                      if color != colors[0]))
            grid = _unique_fallback(pair_index, fallback_colors, required_color, witness,
                                    seen, palette_mode == 0)
            signature = _signature(grid)
            shape = grid.shape
        seen.add(signature)
        previous_shapes.append(shape)
        puzzle[split].append({"input": grid, "output": _transform(grid)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    witness_found = False
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label}: invalid pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label}.{name}: expected a 2D array")
                _require(np.issubdtype(grid.dtype, np.integer), f"{label}.{name}: noninteger array")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape),
                         f"{label}.{name}: invalid shape")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         f"{label}.{name}: invalid color")
            inp, out = pair["input"], pair["output"]
            height, width = inp.shape
            _require(width <= 15 and out.shape == (height, 2 * width),
                     f"{label}: wrong doubled dimensions")
            _require(np.array_equal(out[:, :width], inp) and
                     np.array_equal(out[:, width:], np.flip(inp, axis=1)),
                     f"{label}: incorrect mirror append")
            _require(set(int(x) for x in np.unique(inp)) == set(int(x) for x in np.unique(out)),
                     f"{label}: colors changed")
            inp_sig = _signature(inp)
            pair_sig = (inp_sig, _signature(out))
            _require(inp_sig not in seen_inputs and pair_sig not in seen_pairs,
                     f"{label}: duplicate pair")
            _require(inp_sig not in _SOURCE_SIGNATURES, f"{label}: copied source input")
            seen_inputs.add(inp_sig)
            seen_pairs.add(pair_sig)
            if split == "train" and np.any(inp != np.flip(inp, axis=1)):
                witness_found = True
    _require(witness_found, "training does not distinguish reflection from copying")
    for alternative in (
        lambda grid: grid,
        lambda grid: np.flip(grid, axis=0),
        lambda grid: np.flip(grid, axis=(0, 1)),
    ):
        _require(any(not np.array_equal(np.flip(pair["input"], axis=1),
                                        alternative(pair["input"]))
                     for pair in puzzle["train"]),
                 "training does not distinguish a competing append rule")


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_signature(pair["input"]), _signature(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def _features(puzzle: dict) -> dict:
    pairs = puzzle["train"] + puzzle["test"]
    inputs = [pair["input"] for pair in pairs]
    train_colors = set(int(x) for pair in puzzle["train"] for x in np.unique(pair["input"]))
    row_symmetry = [np.all(grid == grid[:, ::-1], axis=1) for grid in inputs]
    observed_palettes = [frozenset(int(x) for x in np.unique(grid)) for grid in inputs]
    bands = stripes = sparse = busy = compact_mixed = 0
    for grid, flags, palette in zip(inputs, row_symmetry, observed_palettes):
        height, width = grid.shape
        repeated = len({tuple(int(x) for x in row) for row in grid}) < height
        constant_rows = int(np.count_nonzero(np.all(grid == grid[:, :1], axis=1)))
        constant_columns = int(np.count_nonzero(np.all(grid == grid[:1, :], axis=0)))
        dominant = int(np.unique(grid, return_counts=True)[1].max()) / grid.size
        bands += constant_rows >= 2 and len(palette) >= 2 and repeated
        stripes += width >= 3 and constant_columns >= max(2, width // 2) and len(palette) >= 2
        sparse += dominant >= 0.8 and len(palette) >= 2
        busy += dominant <= 0.42 and len(palette) >= 3
        compact_mixed += (height, width) == (3, 3) and bool(np.any(flags)) and bool(np.any(~flags))
    return {
        "shapes": [grid.shape for grid in inputs],
        "heights": [grid.shape[0] for grid in inputs],
        "widths": [grid.shape[1] for grid in inputs],
        "color_counts": [np.unique(grid).size for grid in inputs],
        "all_pal_train": sum(bool(np.all(flags)) for flags in row_symmetry[:len(puzzle["train"])]),
        "one_asym_train": sum(int(np.count_nonzero(~flags)) == 1 for flags in row_symmetry[:len(puzzle["train"])]),
        "all_asym_train": sum(bool(np.all(~flags)) for flags in row_symmetry[:len(puzzle["train"])]),
        "pal_test": sum(bool(np.all(flags)) for flags in row_symmetry[len(puzzle["train"]):]),
        "asym_test": sum(bool(np.any(~flags)) for flags in row_symmetry[len(puzzle["train"]):]),
        "unseen_test": sum(bool(set(int(x) for x in np.unique(pair["input"])) - train_colors)
                           for pair in puzzle["test"]),
        "constant_rows": sum(int(np.count_nonzero(np.all(grid == grid[:, :1], axis=1)))
                             for grid in inputs),
        "repeated_rows": sum(len({tuple(int(x) for x in row) for row in grid}) < len(grid)
                             for grid in inputs),
        "shared_shape": len({grid.shape for grid in inputs}) < len(inputs),
        "cross_split_shape": bool({grid.shape for grid in inputs[:len(puzzle["train"])]} &
                                  {grid.shape for grid in inputs[len(puzzle["train"]):]}),
        "shared_palette": len(set(observed_palettes)) == 1,
        "varying_palette_size": len({len(colors) for colors in observed_palettes}) > 1,
        "all_pal_tests": all(bool(np.all(flags)) for flags in row_symmetry[len(puzzle["train"]):]),
        "test_asym": any(bool(np.any(~flags)) for flags in row_symmetry[len(puzzle["train"]):]),
        "one_color": sum(len(colors) == 1 for colors in observed_palettes),
        "source_scale_mixed": compact_mixed,
        "bands": bands,
        "stripes": stripes,
        "sparse": sparse,
        "busy": busy,
        "combined_asym_unseen": sum(
            bool(np.any(~flags)) and bool(set(colors) - train_colors)
            for colors, flags in zip(observed_palettes[len(puzzle["train"]):],
                                     row_symmetry[len(puzzle["train"]):])
        ),
    }


def self_test() -> None:
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (201, None, 1), (202, None, 4),
                              (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "requested train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "requested test count ignored")
        _require(_puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed, train, test)),
                 "generation is not deterministic")
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(301, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {request}")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = {_puzzle_signature(puzzle) for puzzle in puzzles}
    _require(len(signatures) == 42, "different seeds produced duplicate puzzles")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count lacks diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count lacks diversity")
    features = [_features(puzzle) for puzzle in puzzles]
    all_shapes = [shape for feature in features for shape in feature["shapes"]]
    _require(len(set(all_shapes)) >= 50, "shape diversity is too narrow")
    _require(sum(shape == (3, 3) for shape in all_shapes) >= 20, "few source-scale inputs")
    _require(sum(len(set(f["shapes"])) < len(f["shapes"]) for f in features) >= 10,
             "shape reuse is too rare")
    _require(max(height for f in features for height in f["heights"]) >= 25,
             "tall inputs missing")
    _require(max(width for f in features for width in f["widths"]) == 15,
             "wide inputs missing")
    _require(min(width for f in features for width in f["widths"]) == 1,
             "one-column inputs missing")
    for key in ("all_pal_train", "one_asym_train", "all_asym_train",
                "pal_test", "asym_test", "unseen_test", "constant_rows", "repeated_rows"):
        _require(sum(feature[key] for feature in features) > 0,
                 f"missing observed variation: {key}")
    _require(len({count for f in features for count in f["color_counts"]}) >= 5,
             "observed palette sizes lack diversity")
    for key, minimum in (("cross_split_shape", 5), ("shared_palette", 5),
                         ("varying_palette_size", 5), ("all_pal_tests", 5),
                         ("test_asym", 5), ("one_color", 5),
                         ("source_scale_mixed", 5), ("bands", 5),
                         ("stripes", 5), ("sparse", 5), ("busy", 5),
                         ("combined_asym_unseen", 3)):
        _require(sum(f[key] for f in features) >= minimum,
                 f"observed review variation is too rare: {key}")
