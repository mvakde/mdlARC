"""Generate solved self-substitution ARC puzzles."""

import numpy as np

_SOURCE_INPUTS = frozenset({
    (6, 6, 0, 6, 0, 0, 0, 6, 6),
    (4, 0, 4, 0, 0, 0, 0, 4, 0),
    (0, 0, 0, 0, 0, 2, 2, 0, 2),
    (2, 2, 2, 0, 0, 0, 0, 2, 2),
    (0, 7, 7, 7, 7, 7, 0, 7, 7),
    (7, 0, 7, 7, 0, 7, 7, 7, 0),
})
_SHAPES = ((2, 2), (2, 3), (3, 2), (3, 4), (4, 3),
           (4, 4), (4, 5), (5, 4), (5, 5))


def _check(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer_argument(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = None if num_train is None else _integer_argument(num_train, "num_train")
    test = None if num_test is None else _integer_argument(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
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


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_copy(grid):
    return grid.shape == (3, 3) and tuple(int(x) for x in grid.flat) in _SOURCE_INPUTS


def _transforms(mask):
    return [np.rot90(mask, k) for k in range(4)] + [
        np.rot90(np.fliplr(mask), k) for k in range(4)
    ]


def _fully_asymmetric(mask):
    return all(not np.array_equal(mask, variant) for variant in _transforms(mask)[1:])


def _make_output(input_grid, background):
    height, width = input_grid.shape
    output = np.full((height * height, width * width), background, dtype=np.int8)
    for row in range(height):
        for column in range(width):
            if input_grid[row, column] != background:
                output[row * height:(row + 1) * height,
                       column * width:(column + 1) * width] = input_grid
    return output


def _sample_mask(rng, shape, density, asymmetric, foreground, background, used):
    count = shape[0] * shape[1]
    if not 1 <= density < count:
        raise ValueError("density must leave both colors present")
    if count <= 9:
        candidates = []
        for code in range(1, 1 << count):
            if code.bit_count() != density:
                continue
            mask = np.array([(code >> i) & 1 for i in range(count)], dtype=np.int8).reshape(shape)
            grid = np.where(mask, foreground, background).astype(np.int8)
            if (not asymmetric or _fully_asymmetric(mask)) and _key(mask) not in used and not _source_copy(grid):
                candidates.append(mask)
        if not candidates:
            raise ValueError("no distinct motif at requested density")
        chosen = candidates[int(rng.integers(len(candidates)))]
    else:
        chosen = None
        for _ in range(64):
            positions = rng.choice(count, size=density, replace=False)
            mask = np.zeros(count, dtype=np.int8)
            mask[positions] = 1
            mask = mask.reshape(shape)
            if _key(mask) not in used:
                chosen = mask
                break
        if chosen is None:
            # Exhaust a bounded deterministic series after unlikely collisions.
            for start in range(count):
                mask = np.zeros(count, dtype=np.int8)
                mask[np.arange(start, start + density) % count] = 1
                mask = mask.reshape(shape)
                if _key(mask) not in used:
                    chosen = mask
                    break
        if chosen is None:
            raise ValueError("no distinct motif found")
    used.add(_key(chosen))
    return chosen


def _standard_densities(rng, seed, train_count, test_count):
    mode = seed % 5
    if mode == 0:  # Preserve the source's density shift as one regime.
        train = [3, 7] if train_count == 2 else [3, 5, 7]
        train += [int(rng.choice((3, 5, 7))) for _ in range(train_count - len(train))]
        test = [6] * test_count
    elif mode == 4:  # Fixed density, independently sampled layouts.
        density = int(rng.integers(3, 7))
        train, test = [density] * train_count, [density] * test_count
    else:
        anchor = int(rng.integers(3, 7))
        train = [anchor]
        test = [1 + seed % 8]
        if mode == 1:
            train += [int(rng.integers(1, 9)) for _ in range(train_count - 1)]
            test += [int(rng.integers(1, 9)) for _ in range(test_count - 1)]
        elif mode == 2:
            if test[0] == anchor:
                test[0] = test[0] % 8 + 1
            choices = [k for k in range(1, 9) if k != test[0]]
            train += [int(rng.choice(choices)) for _ in range(train_count - 1)]
            test *= test_count
        else:
            test[0] = anchor
            train += [int(rng.integers(1, 9)) for _ in range(train_count - 1)]
            test += [int(rng.integers(1, 9)) for _ in range(test_count - 1)]
    return train, test


def _shape_plans(rng, seed, train_count, test_count):
    if seed % 3 == 0:
        return [(3, 3)] * train_count, [(3, 3)] * test_count
    second = _SHAPES[(seed // 3) % len(_SHAPES)]
    train = [(3, 3), second]
    options = _SHAPES + ((3, 3),)
    train += [options[int(rng.integers(len(options)))] for _ in range(train_count - 2)]
    # Rectangular test cases require a rectangular demonstration.
    has_rectangle = any(height != width for height, width in train)
    allowed = options if has_rectangle else tuple(shape for shape in options if shape[0] == shape[1])
    if seed % 2 == 0:
        candidates = [shape for shape in allowed if shape not in train]
        first = candidates[int(rng.integers(len(candidates)))] if candidates else train[int(rng.integers(len(train)))]
    else:
        first = train[int(rng.integers(len(train)))]
    test = [first]
    for _ in range(test_count - 1):
        pool = allowed if seed % 2 == 0 else train
        test.append(pool[int(rng.integers(len(pool)))])
    return train, test


def _background_from_training(train_pairs):
    candidates = set(int(x) for x in np.unique(train_pairs[0]["input"]))
    for pair in train_pairs:
        candidates &= set(int(x) for x in np.unique(pair["input"]))
        candidates = {color for color in candidates
                      if np.array_equal(pair["output"], _make_output(pair["input"], color))}
    _check(len(candidates) == 1, "training does not identify one background")
    return candidates.pop()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer_argument(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    # The background is the sole learned puzzle-level convention.
    background = 0 if int(rng.integers(5)) < 3 else int(rng.integers(1, 10))
    colors = [color for color in range(10) if color != background]
    rng.shuffle(colors)
    if background != 0 and int(rng.integers(3)) == 0:
        colors.remove(0)
        colors.insert(0, 0)
    palette_size = int(rng.integers(1, min(4, train_count) + 1))
    palette = colors[:palette_size]
    train_colors = [palette[i % palette_size] for i in range(train_count)]
    rng.shuffle(train_colors)
    test_colors = []
    for _ in range(test_count):
        if int(rng.integers(4)) == 0:
            novel = [color for color in colors if color not in palette]
            test_colors.append(novel[int(rng.integers(len(novel)))])
        else:
            test_colors.append(palette[int(rng.integers(len(palette)))])
    train_shapes, test_shapes = _shape_plans(rng, seed, train_count, test_count)
    train_densities, test_densities = _standard_densities(rng, seed, train_count, test_count)
    used = set()
    train, test = [], []
    for split, shapes, densities, pair_colors in (
        (train, train_shapes, train_densities, train_colors),
        (test, test_shapes, test_densities, test_colors),
    ):
        for index, (shape, color) in enumerate(zip(shapes, pair_colors)):
            asymmetric = split is train and index == 0
            density = densities[index] if shape == (3, 3) else int(rng.integers(1, shape[0] * shape[1]))
            mask = _sample_mask(rng, shape, density, asymmetric, color, background, used)
            input_grid = np.where(mask, color, background).astype(np.int8)
            split.append({"input": input_grid, "output": _make_output(input_grid, background)})
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _check(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _check(grid.ndim == 2, f"{label} must be two-dimensional")
    height, width = grid.shape
    _check(1 <= height <= 30 and 1 <= width <= 30, f"{label} has illegal dimensions")
    _check(np.issubdtype(grid.dtype, np.integer), f"{label} must have integer cells")
    _check(np.all((0 <= grid) & (grid <= 9)), f"{label} has an illegal color")


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
           "invalid pair counts")
    for split_name in ("train", "test"):
        for index, pair in enumerate(puzzle[split_name]):
            label = f"{split_name}[{index}]"
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label} keys")
            _validate_grid(pair["input"], label + ".input")
            _validate_grid(pair["output"], label + ".output")
    background = _background_from_training(puzzle["train"])
    train_shapes = {pair["input"].shape for pair in puzzle["train"]}
    _check((3, 3) in train_shapes, "training needs a 3x3 orientation demonstration")
    _check(any(pair["input"].shape == (3, 3) and
               _fully_asymmetric(pair["input"] != background)
               for pair in puzzle["train"]), "training does not establish unchanged orientation")
    all_shapes = train_shapes | {pair["input"].shape for pair in puzzle["test"]}
    if all_shapes != {(3, 3)}:
        _check(len(train_shapes) >= 2, "variable-size tests need multiple training sizes")
    if any(height != width for height, width in all_shapes):
        _check(any(height != width for height, width in train_shapes),
               "rectangular tests need a rectangular training example")
    seen_inputs, seen_masks = set(), set()
    for split_name in ("train", "test"):
        for index, pair in enumerate(puzzle[split_name]):
            inp, out = pair["input"], pair["output"]
            label = f"{split_name}[{index}]"
            height, width = inp.shape
            _check(2 <= height <= 5 and 2 <= width <= 5, f"{label} input size")
            _check(out.shape == (height * height, width * width), f"{label} output size")
            values = set(int(x) for x in np.unique(inp))
            _check(len(values) == 2 and background in values, f"{label} color roles")
            foreground = (values - {background}).pop()
            mask = inp != background
            density = int(np.count_nonzero(mask))
            key, mask_key = _key(inp), _key(mask)
            _check(key not in seen_inputs, f"{label} duplicate input")
            _check(mask_key not in seen_masks, f"{label} duplicate motif")
            seen_inputs.add(key)
            seen_masks.add(mask_key)
            _check(not _source_copy(inp), f"{label} copies a source input")
            _check(set(int(x) for x in np.unique(out)) == values, f"{label} output palette")
            _check(int(np.count_nonzero(out == foreground)) == density * density,
                   f"{label} foreground count")
            blank = np.full(inp.shape, background, dtype=out.dtype)
            for row in range(height):
                for column in range(width):
                    block = out[row * height:(row + 1) * height,
                                column * width:(column + 1) * width]
                    expected = inp if mask[row, column] else blank
                    _check(np.array_equal(block, expected), f"{label} block substitution")


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[grid], b[grid]) for a, b in zip(first[split], second[split])
        for grid in ("input", "output")) for split in ("train", "test"))


def _component_count(mask):
    seen = set()
    components = 0
    for row, column in zip(*np.nonzero(mask)):
        if (row, column) in seen:
            continue
        components += 1
        stack = [(row, column)]
        seen.add((row, column))
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]
                        and mask[nr, nc] and (nr, nc) not in seen):
                    seen.add((nr, nc))
                    stack.append((nr, nc))
    return components


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack variety")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack variety")
    _check(_puzzles_equal(generate_puzzle(12345), generate_puzzle(12345)), "automatic determinism")
    _check(_puzzles_equal(generate_puzzle(23456, 8, 3), generate_puzzle(23456, 8, 3)),
           "explicit determinism")
    for seed, train, test in ((31001, 2, 1), (31002, 10, 1), (31003, 10, 4),
                              (32001, None, 1), (32002, None, 4),
                              (32003, 2, None), (32004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _check(train is None or len(puzzle["train"]) == train, "train count not honored")
        _check(test is None or len(puzzle["test"]) == test, "test count not honored")
    for index, arguments in enumerate((
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": 2.0, "num_test": 1}, {"num_train": True, "num_test": 1},
    )):
        try:
            generate_puzzle(33000 + index, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count request accepted: {arguments}")
    shapes, backgrounds, masks, signatures = set(), set(), set(), set()
    standard_train_densities, standard_test_densities = set(), set()
    novel_color = familiar_color = zero_foreground = 0
    variable_density_puzzles = fixed_density_puzzles = heldout_size = 0
    overlap_density = test_only_density = train_only_density = 0
    one_color_puzzles = multi_color_puzzles = familiar_size = 0
    connected = disconnected = symmetric = asymmetric = 0
    full_row = full_column = source_mask_reuses = combined_axes = 0
    source_masks = {tuple(int(value != 0) for value in source) for source in _SOURCE_INPUTS}
    for puzzle in puzzles:
        background = _background_from_training(puzzle["train"])
        backgrounds.add(background)
        train_colors = {(set(int(x) for x in np.unique(pair["input"])) - {background}).pop()
                        for pair in puzzle["train"]}
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        train_densities, test_densities = set(), set()
        one_color_puzzles += len(train_colors) == 1
        multi_color_puzzles += len(train_colors) > 1
        signatures.add(tuple(_key(pair["input"]) for split in ("train", "test")
                             for pair in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                shape = inp.shape
                shapes.add(shape)
                masks.add(_key(inp != background))
                density = int(np.count_nonzero(inp != background))
                foreground = (set(int(x) for x in np.unique(inp)) - {background}).pop()
                component_count = _component_count(inp != background)
                connected += component_count == 1
                disconnected += component_count > 1
                is_asymmetric = _fully_asymmetric(inp != background)
                asymmetric += is_asymmetric
                symmetric += not is_asymmetric
                full_row += bool(np.any(np.all(inp != background, axis=1)))
                full_column += bool(np.any(np.all(inp != background, axis=0)))
                source_mask_reuses += shape == (3, 3) and tuple(
                    int(value) for value in (inp != background).flat) in source_masks
                if split == "train":
                    train_densities.add(density)
                    if shape == (3, 3):
                        standard_train_densities.add(density)
                else:
                    test_densities.add(density)
                    novel_color += foreground not in train_colors
                    familiar_color += foreground in train_colors
                    zero_foreground += foreground == 0
                    heldout_size += shape not in train_shapes
                    familiar_size += shape in train_shapes
                    if shape == (3, 3):
                        standard_test_densities.add(density)
                # This check uses slices rather than the output constructor.
                h, w = shape
                for row in range(h):
                    for col in range(w):
                        block = out[row*h:(row+1)*h, col*w:(col+1)*w]
                        required = inp if inp[row, col] != background else np.full(shape, background)
                        _check(np.array_equal(block, required), "independent blockwise check")
        variable_density_puzzles += len(train_densities | test_densities) > 1
        fixed_density_puzzles += len(train_densities | test_densities) == 1
        overlap_density += bool(train_densities & test_densities)
        test_only_density += bool(test_densities - train_densities)
        train_only_density += bool(train_densities - test_densities)
        combined_axes += (background != 0 and len(train_shapes) > 1 and
                          len(train_densities | test_densities) > 1 and len(train_colors) > 1)
    _check(len(signatures) == 42 and len(masks) >= 90, "insufficient structural diversity")
    _check(set(_SHAPES) <= shapes, "size/aspect coverage missing")
    _check(len(backgrounds) >= 4 and 0 in backgrounds, "background diversity missing")
    _check(standard_train_densities == set(range(1, 9)), "3x3 train densities missing")
    _check(standard_test_densities == set(range(1, 9)), "3x3 test densities missing")
    _check(novel_color and familiar_color and zero_foreground, "color regimes missing")
    _check(variable_density_puzzles and fixed_density_puzzles and heldout_size,
           "density or shape regimes missing")
    _check(overlap_density and test_only_density and train_only_density,
           "train/test density regimes missing")
    _check(one_color_puzzles and multi_color_puzzles and familiar_size,
           "palette or familiar-size regimes missing")
    _check(connected and disconnected and symmetric and asymmetric and
           full_row and full_column, "motif geometry lacks breadth")
    _check(source_mask_reuses and combined_axes,
           "source-layout reuse or combined variation missing")
