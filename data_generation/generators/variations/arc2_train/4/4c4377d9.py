"""Generate whole-grid vertical reflection puzzles."""

import numpy as np


_SOURCE_INPUT_KEYS = {
    (3, 4, (9, 4, 9, 4, 9, 9, 4, 4, 4, 4, 4, 4)),
    (3, 4, (4, 1, 1, 4, 1, 1, 1, 1, 4, 4, 4, 1)),
    (3, 4, (3, 3, 5, 5, 3, 5, 5, 3, 5, 5, 3, 3)),
    (3, 4, (9, 9, 5, 9, 5, 5, 9, 9, 9, 5, 9, 9)),
    (3, 4, (4, 4, 9, 9, 4, 4, 4, 4, 4, 4, 9, 9)),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape[0], grid.shape[1], tuple(int(x) for x in grid.flat))


def _validate_optional_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    n_train = _validate_optional_count("num_train", num_train, 2, 10)
    n_test = _validate_optional_count("num_test", num_test, 1, 4)
    if n_train is None and n_test is None:
        n_test = int(rng.integers(1, 5))
        n_train = int(rng.integers(max(2, n_test + 1), 11))
    elif n_train is None:
        n_train = int(rng.integers(max(2, n_test + 1), 11))
    elif n_test is None:
        n_test = int(rng.integers(1, min(4, n_train - 1) + 1))
    if n_train <= n_test:
        raise ValueError("num_train must exceed num_test")
    return n_train, n_test


def _answer(grid):
    return np.concatenate((grid[::-1], grid), axis=0)


def _alternatives(grid):
    """Plausible interpretations of the training demonstrations."""
    return (
        np.concatenate((grid, grid), axis=0),
        np.concatenate((grid[:, ::-1], grid), axis=0),
        np.concatenate((grid[::-1, ::-1], grid), axis=0),
        np.concatenate((grid, grid[::-1]), axis=0),
    )


def _sample_shape(rng, minimum_height=1, minimum_width=1):
    # Small cases expose degeneracies; medium and large cases test scaling.
    tier = int(rng.choice(3, p=(0.20, 0.48, 0.32)))
    ranges = ((1, 5, 1, 7), (3, 11, 3, 19), (7, 16, 10, 31))
    lo_h, hi_h, lo_w, hi_w = ranges[tier]
    height = int(rng.integers(max(lo_h, minimum_height), hi_h))
    width = int(rng.integers(max(lo_w, minimum_width), hi_w))
    if rng.random() < 0.07 and minimum_height <= 3 and minimum_width <= 4:
        return 3, 4
    return height, width


def _sample_palette(rng, count, required_color=None):
    if required_color is None:
        return tuple(int(x) for x in rng.choice(10, size=count, replace=False))
    others = [x for x in range(10) if x != required_color]
    values = [required_color]
    if count > 1:
        values.extend(int(x) for x in rng.choice(others, size=count - 1, replace=False))
    rng.shuffle(values)
    return tuple(values)


def _sample_indices(rng, rows, cols, colors):
    mode = int(rng.integers(0, 6))
    if mode == 0:  # independent cells with varied density
        weights = rng.dirichlet(np.full(colors, float(rng.uniform(0.6, 2.0))))
        return rng.choice(colors, size=(rows, cols), p=weights).astype(np.int8)
    if mode == 1:  # bands and checker-like blocks
        rr, cc = np.indices((rows, cols))
        rb = int(rng.integers(1, min(rows, 5) + 1))
        cb = int(rng.integers(1, min(cols, 6) + 1))
        phase = int(rng.integers(0, colors))
        return ((rr // rb + cc // cb + phase) % colors).astype(np.int8)
    if mode == 2:  # scaled coarse pattern
        rs = int(rng.integers(1, min(rows, 4) + 1))
        cs = int(rng.integers(1, min(cols, 5) + 1))
        coarse = rng.integers(0, colors, size=((rows + rs - 1) // rs,
                                               (cols + cs - 1) // cs), dtype=np.int8)
        return np.repeat(np.repeat(coarse, rs, axis=0), cs, axis=1)[:rows, :cols].copy()
    if mode == 3:  # repeated rows, sometimes offset
        count = int(rng.integers(1, min(rows, 5) + 1))
        prototypes = rng.integers(0, colors, size=(count, cols), dtype=np.int8)
        result = prototypes[rng.integers(0, count, size=rows)].copy()
        if cols > 1 and rng.random() < 0.45:
            for r in range(rows):
                result[r] = np.roll(result[r], int(rng.integers(0, cols)))
        return result
    if mode == 4:  # overlapping filled rectangles, including at margins
        result = np.full((rows, cols), int(rng.integers(colors)), dtype=np.int8)
        for _ in range(int(rng.integers(1, 8))):
            r0 = int(rng.integers(rows))
            c0 = int(rng.integers(cols))
            r1 = int(rng.integers(r0 + 1, rows + 1))
            c1 = int(rng.integers(c0 + 1, cols + 1))
            result[r0:r1, c0:c1] = int(rng.integers(colors))
        return result
    # Sparse marks and occasional border: all cells remain part of the object.
    result = np.full((rows, cols), int(rng.integers(colors)), dtype=np.int8)
    marks = max(1, int(rows * cols * rng.uniform(0.05, 0.32)))
    for _ in range(marks):
        result[int(rng.integers(rows)), int(rng.integers(cols))] = int(rng.integers(colors))
    if min(rows, cols) >= 3 and rng.random() < 0.5:
        border = int(rng.integers(colors))
        result[0, :] = border
        result[-1, :] = border
        result[:, 0] = border
        result[:, -1] = border
    return result


def _sample_grid(rng, shape, palette, symmetry):
    height, width = shape
    colors = len(palette)
    if symmetry in ("symmetric", "near"):
        top = _sample_indices(rng, (height + 1) // 2, width, colors)
        tail = top[::-1] if height % 2 == 0 else top[-2::-1]
        indices = np.concatenate((top, tail), axis=0)
        if symmetry == "near":
            # Interior defect leaves identical endpoints but breaks reflection.
            row = int(rng.integers(1, height - 1))
            col = int(rng.integers(width))
            indices[row, col] = (indices[row, col] + 1) % colors
    else:
        indices = _sample_indices(rng, height, width, colors)
    return np.asarray(palette, dtype=np.int8)[indices]


def _acceptable(grid, symmetry):
    if symmetry == "symmetric":
        return bool(np.array_equal(grid, grid[::-1]))
    if symmetry == "near":
        return (np.array_equal(grid[0], grid[-1])
                and not np.array_equal(grid, grid[::-1]))
    if symmetry in ("asymmetric", "strong"):
        if np.array_equal(grid, grid[::-1]):
            return False
        if symmetry == "strong":
            upper = grid[::-1]
            return (not np.array_equal(upper, grid[:, ::-1])
                    and not np.array_equal(upper, grid[::-1, ::-1]))
    return True


def _fallback_grid(rng, shape, palette, symmetry):
    height, width = shape
    grid = np.full(shape, palette[0], dtype=np.int8)
    if symmetry == "symmetric":
        grid[0, 0] = palette[-1]
        grid[-1, 0] = palette[-1]
    elif symmetry == "near":
        grid[1, 0] = palette[-1]
    else:
        grid[0, 0] = palette[-1]
        if symmetry == "strong":
            grid[1, min(1, width - 1)] = palette[-1]
    return grid


def _symmetry_class(rng, section, index, classic, test_count):
    if classic:
        return "asymmetric" if section == "train" else "symmetric"
    if section == "train":
        if index == 0:
            return "strong"
        if index == 1:
            return "symmetric"
        return str(rng.choice(("asymmetric", "symmetric", "near"), p=(0.48, 0.27, 0.25)))
    if test_count >= 2:
        if index == 0:
            return "symmetric"
        if index == 1:
            return "asymmetric"
    return str(rng.choice(("asymmetric", "symmetric", "near"), p=(0.48, 0.32, 0.20)))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # These are construction regimes, never hidden conventions in the rule.
    classic = bool(rng.random() < 0.25)
    fixed_shape = bool(rng.random() < 0.24)
    fixed_palette = bool(rng.random() < 0.18)
    shared_shape = _sample_shape(rng, 3, 3) if fixed_shape else None
    shared_palette = _sample_palette(rng, 2) if fixed_palette else None
    puzzle = {"train": [], "test": []}
    seen = set()
    train_colors = set()

    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            symmetry = _symmetry_class(rng, section, index, classic, test_count)
            if section == "train" and index == 0:
                symmetry = "strong"
            min_height = 3 if symmetry == "strong" else (4 if symmetry == "near" else
                           (2 if symmetry == "asymmetric" else 1))
            min_width = 3 if symmetry == "strong" else 1
            shape = shared_shape or _sample_shape(rng, min_height, min_width)
            if shape[0] < min_height:
                shape = _sample_shape(rng, min_height, min_width)
            if shared_palette is not None:
                palette = shared_palette
            else:
                color_count = int(rng.choice((1, 2, 3, 4, 5), p=(0.07, 0.58, 0.22, 0.10, 0.03)))
                if symmetry != "symmetric" or (section == "train" and index == 0):
                    color_count = max(2, color_count)
                required = None
                if section == "test" and rng.random() < 0.55:
                    unseen = [x for x in range(10) if x not in train_colors]
                    if unseen:
                        required = int(rng.choice(unseen))
                palette = _sample_palette(rng, color_count, required)

            pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
            pair_rng = np.random.default_rng(pair_seed)
            for _ in range(96):
                grid = _sample_grid(pair_rng, shape, palette, symmetry)
                key = _grid_key(grid)
                if (key not in seen and key not in _SOURCE_INPUT_KEYS
                        and _acceptable(grid, symmetry)):
                    break
            else:
                # Deterministic construction for tiny, collision-prone grids.
                # Enlarge only this pair if a small shape is exhausted.
                for new_height in range(shape[0], 16):
                    for new_width in range(shape[1], 31):
                        candidate = _fallback_grid(
                            pair_rng, (new_height, new_width), palette, symmetry
                        )
                        candidate_key = _grid_key(candidate)
                        if (candidate_key not in seen
                                and candidate_key not in _SOURCE_INPUT_KEYS
                                and _acceptable(candidate, symmetry)):
                            grid, key = candidate, candidate_key
                            break
                    else:
                        continue
                    break
                else:
                    raise RuntimeError("Unable to construct a distinct pair")
            seen.add(key)
            if section == "train":
                train_colors.update(int(x) for x in np.unique(grid))
            puzzle[section].append({"input": grid.copy(), "output": _answer(grid).copy()})

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "Sections must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test,
             "Illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            label = f"{section}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} has invalid keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label} has invalid array")
                _require(all(1 <= dim <= 30 for dim in grid.shape),
                         f"{label} has invalid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer),
                         f"{label} has noninteger cells")
                _require(bool(np.all((grid >= 0) & (grid <= 9))),
                         f"{label} has invalid colors")
            h, w = inp.shape
            _require(h <= 15 and out.shape == (2 * h, w),
                     f"{label} has invalid size relationship")
            _require(out.dtype == inp.dtype and np.array_equal(out, _answer(inp)),
                     f"{label} violates reflection above unchanged input")
            _require(np.array_equal(np.unique(inp), np.unique(out)),
                     f"{label} changes colors")
            input_key, output_key = _grid_key(inp), _grid_key(out)
            pair_key = (input_key, output_key)
            _require(input_key not in _SOURCE_INPUT_KEYS, f"{label} copies source")
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     f"{label} is duplicated")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

    _require(any(not np.array_equal(p["input"], p["input"][::-1])
                 for p in puzzle["train"]), "Training must demonstrate row reversal")
    # An alternative that fits every training pair may only survive when its
    # test prediction is identical to the intended answer.
    for alternative_index in range(4):
        fits_train = all(np.array_equal(_alternatives(p["input"])[alternative_index],
                                        p["output"]) for p in puzzle["train"])
        if fits_train:
            _require(all(np.array_equal(_alternatives(p["input"])[alternative_index],
                                        _answer(p["input"])) for p in puzzle["test"]),
                     f"Training leaves alternative {alternative_index} ambiguous")


def _signature(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[section]) for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "Automatic counts lack diversity")
    _require(len({_signature(p) for p in puzzles}) >= 40,
             "Different seeds gave too few distinct puzzles")
    shapes, palettes, layouts = set(), set(), set()
    varying_shapes = varying_palettes = genuine_palettes = 0
    mixed_train = mixed_test = combined_axes = 0
    unseen_test_colors = familiar_test_sizes = new_test_sizes = 0
    taller_tests = shorter_tests = repeated_rows = 0
    symmetric_train = asymmetric_test = near = mono = multi = one_row = one_col = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        pairs = puzzle["train"] + puzzle["test"]
        local_shapes = {p["input"].shape for p in pairs}
        local_palettes = {tuple(int(x) for x in np.unique(p["input"])) for p in pairs}
        multicolor_palettes = {palette for palette in local_palettes if len(palette) >= 2}
        varying_shapes += len(local_shapes) > 1
        varying_palettes += len(local_palettes) > 1
        genuine_palettes += len(multicolor_palettes) > 1
        train_sym = [np.array_equal(p["input"], p["input"][::-1])
                     for p in puzzle["train"]]
        test_sym = [np.array_equal(p["input"], p["input"][::-1])
                    for p in puzzle["test"]]
        mixed_train += any(train_sym) and not all(train_sym)
        mixed_test += any(test_sym) and not all(test_sym)
        symmetric_train += any(train_sym)
        asymmetric_test += not all(test_sym)
        combined_axes += (len(local_shapes) > 1 and len(multicolor_palettes) > 1
                          and any(train_sym) and not all(train_sym)
                          and any(test_sym) and not all(test_sym))
        train_shapes = {p["input"].shape for p in puzzle["train"]}
        train_colors = {int(x) for p in puzzle["train"]
                        for x in np.unique(p["input"])}
        for pair in puzzle["test"]:
            shape = pair["input"].shape
            familiar_test_sizes += shape in train_shapes
            new_test_sizes += shape not in train_shapes
            unseen_test_colors += bool({int(x) for x in np.unique(pair["input"])}
                                       - train_colors)
            taller_tests += shape[0] > max(s[0] for s in train_shapes)
            shorter_tests += shape[0] < min(s[0] for s in train_shapes)
        for pair in pairs:
            inp = pair["input"]
            colors = np.unique(inp)
            shapes.add(inp.shape)
            palettes.add(tuple(int(x) for x in colors))
            normalized = np.searchsorted(colors, inp)
            layouts.add((inp.shape, tuple(int(x) for x in normalized.flat)))
            mono += len(colors) == 1
            multi += len(colors) >= 3
            one_row += inp.shape[0] == 1
            one_col += inp.shape[1] == 1
            repeated_rows += len({tuple(int(x) for x in row) for row in inp}) < inp.shape[0]
            near += (inp.shape[0] >= 4 and np.array_equal(inp[0], inp[-1])
                     and not np.array_equal(inp, inp[::-1]))
    _require(len(shapes) >= 35 and len(palettes) >= 25 and len(layouts) >= 150,
             "Insufficient cross-seed grid diversity")
    _require(varying_shapes >= 22 and varying_palettes >= 22,
             "Pair-level size or palette variation is too rare")
    _require(genuine_palettes >= 20 and unseen_test_colors >= 15,
             "Literal pair palettes lack variation or novel test colors")
    _require(familiar_test_sizes >= 15 and new_test_sizes >= 25
             and taller_tests >= 4 and shorter_tests >= 2,
             "Test sizes lack familiar and shifted examples")
    _require(mixed_train >= 15 and mixed_test >= 8 and symmetric_train >= 15
             and asymmetric_test >= 15, "Symmetry regimes lack diversity")
    _require(combined_axes >= 5 and repeated_rows >= 100,
             "Compatible variation axes rarely combine")
    _require(near >= 3 and mono >= 1 and multi >= 12 and one_row >= 1
             and one_col >= 1, "Motif or degenerate-size diversity is missing")

    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(5000 + i, nt, ns)
        validate_generated_puzzle(puzzle)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns),
                 "Explicit counts were not honored")
        _require(_signature(puzzle) == _signature(generate_puzzle(5000 + i, nt, ns)),
                 "Generation is nondeterministic")
    for i, (nt, ns) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(6000 + i, nt, ns)
        validate_generated_puzzle(puzzle)
        _require((nt is None or len(puzzle["train"]) == nt)
                 and (ns is None or len(puzzle["test"]) == ns),
                 "Partial counts were not honored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True, "num_test": 1},
                   {"num_train": 3, "num_test": 1.5}):
        try:
            generate_puzzle(7000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {kwargs}")
    for seed in (655, 688):
        validate_generated_puzzle(generate_puzzle(seed, 2, 1))
