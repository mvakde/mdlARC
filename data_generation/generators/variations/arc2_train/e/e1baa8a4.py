import numpy as np


# Exact source pairs are excluded; their dimensions and palettes remain valid.
_SOURCE_SPECS = (
    (((1, 2, 8), (5, 6, 3)), (5, 10), (3, 5, 6)),
    (((4, 5, 2), (1, 3, 2)), (4, 8), (5, 3, 6)),
    (((8, 7, 9, 8), (3, 1, 6, 4), (2, 4, 1, 5)),
     (4, 5, 6), (3, 5, 4, 3)),
    (((2, 8), (3, 5)), (9, 4), (4, 11)),
    (((8, 7, 4, 8), (3, 1, 2, 8), (4, 5, 3, 9),
      (2, 6, 1, 7), (1, 5, 2, 8)), (2, 3, 5, 2, 3), (4, 4, 5, 2)),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expand_grid(latent: np.ndarray, rows: tuple[int, ...],
                 columns: tuple[int, ...]) -> np.ndarray:
    return np.repeat(
        np.repeat(latent, np.asarray(rows), axis=0),
        np.asarray(columns), axis=1,
    ).astype(np.int8, copy=False)


def _grid_fingerprint(grid: np.ndarray) -> tuple:
    return (tuple(int(x) for x in grid.shape),
            tuple(int(x) for x in grid.ravel()))


def _pair_fingerprint(input_grid: np.ndarray,
                      output_grid: np.ndarray) -> tuple:
    return (_grid_fingerprint(input_grid), _grid_fingerprint(output_grid))


def _source_fingerprints() -> frozenset:
    pairs = set()
    for cells, rows, columns in _SOURCE_SPECS:
        output = np.asarray(cells, dtype=np.int8)
        pairs.add(_pair_fingerprint(_expand_grid(output, rows, columns), output))
    return frozenset(pairs)


_SOURCE_PAIR_FINGERPRINTS = _source_fingerprints()


def _line_starts(grid: np.ndarray, axis: int) -> list[int]:
    starts = [0]
    for index in range(1, grid.shape[axis]):
        current = grid[index, :] if axis == 0 else grid[:, index]
        previous = grid[index - 1, :] if axis == 0 else grid[:, index - 1]
        if not np.array_equal(current, previous):
            starts.append(index)
    return starts


def _run_lengths(grid: np.ndarray, axis: int) -> tuple[int, ...]:
    edges = _line_starts(grid, axis) + [grid.shape[axis]]
    return tuple(edges[i + 1] - edges[i] for i in range(len(edges) - 1))


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    return grid[np.ix_(_line_starts(grid, 0), _line_starts(grid, 1))]


def _observable(latent: np.ndarray) -> bool:
    return (len(_line_starts(latent, 0)) == latent.shape[0]
            and len(_line_starts(latent, 1)) == latent.shape[1])


def _coerce_count(name: str, value: int | None,
                  minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _coerce_count("num_train", num_train, 2, 10)
    test = _coerce_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if train is None and test is None:
        # Short puzzles occur frequently enough to remain visible in 42 seeds.
        train = int(rng.choice(np.arange(2, 11), p=[.11, .17, .17, .13, .11, .09, .08, .07, .07]))
        if train in (3, 4) and rng.random() < .65:
            test = 1
        else:
            test = int(rng.integers(1, min(4, train - 1) + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _choose_shapes(rng: np.random.Generator, train_count: int,
                   test_count: int) -> tuple[list[tuple[int, int]],
                                             list[tuple[int, int]]]:
    # Both axes are sampled separately. A second training shape gives even the
    # two-example puzzle evidence beyond a single output size.
    first = (int(rng.integers(2, 5)), int(rng.integers(2, 6)))
    second = (int(rng.integers(3, 7)), int(rng.integers(3, 7)))
    if second == first:
        second = (second[0], 6 if second[1] != 6 else 3)
    train = [first, second]
    for _ in range(2, train_count):
        train.append((int(rng.integers(2, 7)), int(rng.integers(2, 7))))

    max_h = max(h for h, _ in train)
    max_w = max(w for _, w in train)
    categories = ("same", "height", "width", "both", "smaller",
                  "transpose", "extreme", "free")
    test = []
    for _ in range(test_count):
        category = categories[int(rng.integers(len(categories)))]
        if category == "same":
            shape = train[int(rng.integers(len(train)))]
        elif category == "height":
            shape = (int(rng.integers(max_h + 1, 9)),
                     int(rng.integers(2, max_w + 1)))
        elif category == "width":
            shape = (int(rng.integers(2, max_h + 1)),
                     int(rng.integers(max_w + 1, 9)))
        elif category == "both":
            shape = (int(rng.integers(max_h + 1, 9)),
                     int(rng.integers(max_w + 1, 9)))
        elif category == "smaller":
            shape = (2, 2) if rng.random() < .55 else (
                int(rng.integers(2, max_h)), int(rng.integers(2, max_w)))
        elif category == "transpose":
            reference = train[int(rng.integers(len(train)))]
            shape = (reference[1], reference[0]) if reference[0] != reference[1] else (
                2, int(rng.integers(7, 9)))
        elif category == "extreme":
            shape = ((2, int(rng.integers(7, 9))) if rng.random() < .5
                     else (int(rng.integers(7, 9)), 2))
        else:
            shape = (int(rng.integers(2, 9)), int(rng.integers(2, 9)))
        test.append(shape)
    return train, test


def _sample_runs(rng: np.random.Generator, length: int,
                 must_expand: bool, must_vary: bool = False) -> tuple[int, ...]:
    if not must_expand:
        return (1,) * length
    modes = ("uniform", "ascending", "descending", "irregular",
             "wide", "mixed")
    for _ in range(80):
        mode = modes[int(rng.choice(len(modes), p=[.17, .12, .12, .20, .20, .19]))]
        if must_vary and mode == "uniform":
            mode = "wide"
        if mode == "uniform":
            value = int(rng.integers(2, min(7, 30 // length) + 1))
            values = [value] * length
        elif mode in ("ascending", "descending"):
            values = sorted(int(x) for x in rng.integers(1, 7, size=length))
            if mode == "descending":
                values.reverse()
        elif mode == "wide":
            values = [int(x) for x in rng.integers(1, 4, size=length)]
            wide_index = int(rng.integers(length))
            neighbors = [i for i in (wide_index - 1, wide_index + 1)
                         if 0 <= i < length]
            thin_index = neighbors[int(rng.integers(len(neighbors)))]
            values[wide_index] = int(rng.integers(9, 19))
            values[thin_index] = 1
        else:
            values = [int(x) for x in rng.integers(1, 8, size=length)]
            if mode == "irregular" and length >= 3:
                if not (any(a < b for a, b in zip(values, values[1:]))
                        and any(a > b for a, b in zip(values, values[1:]))):
                    continue
        if sum(values) > 30 or max(values) == 1:
            continue
        if must_vary and len(set(values)) == 1:
            continue
        return tuple(values)
    # This bounded fallback works at every permitted latent dimension.
    values = [1] * length
    values[0] = min(11, 30 - length + 1)
    if must_vary and len(set(values)) == 1:
        values[-1] = 2
    return tuple(values)


def _sample_latent(rng: np.random.Generator, height: int,
                   width: int, mode: str) -> np.ndarray:
    if mode in ("sparse", "checker"):
        color_count = 2 if mode == "checker" else int(rng.integers(2, 4))
    elif mode in ("repeat_row", "repeat_column"):
        color_count = int(rng.integers(2, min(6, height * width) + 1))
    else:
        color_count = int(rng.integers(2, min(9, height * width) + 1))
    colors = rng.choice(np.arange(10, dtype=np.int8), size=color_count,
                        replace=False)
    for _ in range(80):
        if mode == "checker":
            grid = np.fromfunction(lambda r, c: (r + c) % 2,
                                   (height, width), dtype=int).astype(np.int8)
            grid = colors[grid]
            if rng.random() < .45:
                grid[int(rng.integers(height)), int(rng.integers(width))] = colors[0]
        elif mode == "sparse":
            grid = np.full((height, width), colors[0], dtype=np.int8)
            count = max(2, int(rng.integers(2, max(3, height * width // 3 + 1))))
            for position in rng.choice(height * width,
                                       size=min(count, height * width), replace=False):
                grid.flat[int(position)] = colors[int(rng.integers(1, color_count))]
        else:
            grid = rng.choice(colors, size=(height, width)).astype(np.int8)
            if mode == "patch":
                top = int(rng.integers(height - 1))
                left = int(rng.integers(width - 1))
                bottom = int(rng.integers(top + 2, height + 1))
                right = int(rng.integers(left + 2, width + 1))
                grid[top:bottom, left:right] = colors[0]
            elif mode == "repeat_row" and height >= 3:
                a = int(rng.integers(height - 2))
                grid[a + 2] = grid[a]
            elif mode == "repeat_column" and width >= 3:
                a = int(rng.integers(width - 2))
                grid[:, a + 2] = grid[:, a]
            elif mode == "random" and height * width >= color_count:
                positions = rng.choice(height * width, size=color_count, replace=False)
                grid.flat[positions] = rng.permutation(colors)
        if _observable(grid) and len(np.unique(grid)) >= 2:
            return grid
    # Random and patterned draws very rarely exhaust the bounded attempts.
    # A checkerboard remains observable with any dimensions of at least two.
    base = np.fromfunction(lambda r, c: (r + c) % 2,
                           (height, width), dtype=int).astype(np.int8)
    return colors[base]


def _generate_pair(rng: np.random.Generator, regime: str,
                   shape: tuple[int, int], index: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    if regime == "source_train" and index == 0:
        expand_rows, expand_columns = True, True
    else:
        operation = int(rng.choice(3, p=[.68, .16, .16]))
        expand_rows, expand_columns = (True, True) if operation == 0 else (
            (True, False) if operation == 1 else (False, True))
    rows = _sample_runs(rng, height, expand_rows,
                        must_vary=(regime == "source_train" and index == 0))
    columns = _sample_runs(rng, width, expand_columns,
                           must_vary=(regime == "source_train" and index == 0))
    modes = ("random", "sparse", "checker", "patch",
             "repeat_row", "repeat_column")
    mode = modes[int(rng.choice(len(modes), p=[.31, .17, .13, .17, .11, .11]))]
    if mode == "repeat_row" and height < 3:
        mode = "patch"
    if mode == "repeat_column" and width < 3:
        mode = "patch"
    output = _sample_latent(rng, height, width, mode)
    return _expand_grid(output, rows, columns), output.copy()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    train_shapes, test_shapes = _choose_shapes(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for section, regime, shapes in (("train", "source_train", train_shapes),
                                    ("test", "target_test", test_shapes)):
        for index, shape in enumerate(shapes):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(100):
                input_grid, output_grid = _generate_pair(pair_rng, regime, shape, index)
                fingerprint = _grid_fingerprint(input_grid)
                pair_fingerprint = _pair_fingerprint(input_grid, output_grid)
                if fingerprint in seen_inputs or pair_fingerprint in _SOURCE_PAIR_FINGERPRINTS:
                    continue
                seen_inputs.add(fingerprint)
                puzzle[section].append({"input": input_grid, "output": output_grid})
                break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_shapes = set()
    train_both_nonuniform = False
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            label = f"{section}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must contain input and output")
            input_grid, output_grid = pair["input"], pair["output"]
            for name, grid in (("input", input_grid), ("output", output_grid)):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label}.{name} must be a 2D NumPy array")
                _require(all(1 <= side <= 30 for side in grid.shape),
                         f"{label}.{name} has invalid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer)
                         and bool(np.all((0 <= grid) & (grid <= 9))),
                         f"{label}.{name} has invalid colors or dtype")
            fingerprint = _grid_fingerprint(input_grid)
            pair_fingerprint = _pair_fingerprint(input_grid, output_grid)
            _require(fingerprint not in seen_inputs, "duplicate input")
            _require(pair_fingerprint not in seen_pairs, "duplicate pair")
            _require(pair_fingerprint not in _SOURCE_PAIR_FINGERPRINTS,
                     "source pair copied")
            seen_inputs.add(fingerprint)
            seen_pairs.add(pair_fingerprint)
            _require(output_grid.shape[0] >= 2 and output_grid.shape[1] >= 2,
                     f"{label} output must use both axes")
            _require(_observable(output_grid), f"{label} has hidden band boundaries")
            _require(np.array_equal(_compress_grid(input_grid), output_grid),
                     f"{label} is not exact whole-line compression")
            rows, columns = _run_lengths(input_grid, 0), _run_lengths(input_grid, 1)
            _require(np.array_equal(_expand_grid(output_grid, rows, columns), input_grid),
                     f"{label} is not a rectangular expansion")
            _require(input_grid.shape != output_grid.shape,
                     f"{label} has no compression")
            _require(np.array_equal(np.unique(input_grid), np.unique(output_grid)),
                     f"{label} changes colors")
            if section == "train":
                train_shapes.add(output_grid.shape)
                if (max(rows) > 1 and max(columns) > 1
                        and len(set(rows)) > 1 and len(set(columns)) > 1):
                    train_both_nonuniform = True
    _require(len(train_shapes) >= 2, "training lacks shape variation")
    _require(train_both_nonuniform,
             "training must demonstrate nonuniform compression on both axes")


def _independent_compression(grid: np.ndarray) -> np.ndarray:
    row_keep = np.r_[True, np.any(grid[1:] != grid[:-1], axis=1)]
    column_keep = np.r_[True, np.any(grid[:, 1:] != grid[:, :-1], axis=0)]
    return grid[row_keep][:, column_keep]


def _predict_from_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    # This check receives only demonstrations and an input; the test answer is
    # read by the caller only after prediction.
    for pair in train_pairs:
        _require(np.array_equal(_independent_compression(pair["input"]),
                                pair["output"]), "training does not show the rule")
    return _independent_compression(test_input)


def self_test() -> None:
    train_counts, test_counts, shapes, palettes = set(), set(), set(), set()
    input_shapes, layouts, row_patterns, column_patterns = set(), set(), set(), set()
    same_test, taller_test, wider_test, both_test, smaller_test = 0, 0, 0, 0, 0
    uniform_rows, uniform_columns = 0, 0
    one_axis, wide_thin, rich, sparse, repeated_line = 0, 0, 0, 0, 0
    short_puzzles = 0
    transposed_test, extreme_test, new_test_color = 0, 0, 0
    monochrome_column, checkerboard, partial_join = 0, 0, 0
    monotone_rows, monotone_columns, irregular_rows, irregular_columns = 0, 0, 0, 0
    pair_palette_variety, combined_variation = 0, 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        short_puzzles += len(puzzle["train"]) in (3, 4) and len(puzzle["test"]) == 1
        train_shapes = [pair["output"].shape for pair in puzzle["train"]]
        train_palettes = [tuple(int(x) for x in np.unique(pair["output"]))
                          for pair in puzzle["train"]]
        pair_palette_variety += len(set(train_palettes)) > 1
        max_h = max(h for h, _ in train_shapes)
        max_w = max(w for _, w in train_shapes)
        puzzle_fp = tuple(_pair_fingerprint(pair["input"], pair["output"])
                          for section in ("train", "test") for pair in puzzle[section])
        fingerprints.add(puzzle_fp)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                large, small = pair["input"], pair["output"]
                _require(np.array_equal(_independent_compression(large), small),
                         "independent compression failed")
                h, w = small.shape
                shapes.add((h, w))
                input_shapes.add(large.shape)
                palettes.add(tuple(int(x) for x in np.unique(small)))
                layouts.add(_grid_fingerprint(small))
                rows, columns = _run_lengths(large, 0), _run_lengths(large, 1)
                row_patterns.add(rows)
                column_patterns.add(columns)
                uniform_rows += len(set(rows)) == 1
                uniform_columns += len(set(columns)) == 1
                one_axis += (max(rows) == 1) != (max(columns) == 1)
                wide_thin += (any(min(a, b) == 1 and max(a, b) >= 9
                                  for a, b in zip(rows, rows[1:]))
                              or any(min(a, b) == 1 and max(a, b) >= 9
                                     for a, b in zip(columns, columns[1:])))
                rich += len(np.unique(small)) >= 6
                sparse += len(np.unique(small)) == 2
                monotone_rows += (len(set(rows)) > 1 and
                                  (list(rows) == sorted(rows)
                                   or list(rows) == sorted(rows, reverse=True)))
                monotone_columns += (len(set(columns)) > 1 and
                                     (list(columns) == sorted(columns)
                                      or list(columns) == sorted(columns, reverse=True)))
                irregular_rows += (any(a < b for a, b in zip(rows, rows[1:]))
                                   and any(a > b for a, b in zip(rows, rows[1:])))
                irregular_columns += (any(a < b for a, b in zip(columns, columns[1:]))
                                      and any(a > b for a, b in zip(columns, columns[1:])))
                monochrome_column += any(np.all(small[:, c] == small[0, c])
                                         for c in range(w))
                checkerboard += (len(np.unique(small)) == 2
                                 and all(small[r, c] != small[r, c + 1]
                                         for r in range(h) for c in range(w - 1))
                                 and all(small[r, c] != small[r + 1, c]
                                         for r in range(h - 1) for c in range(w)))
                joined = (any(np.any(small[r] == small[r + 1])
                              for r in range(h - 1))
                          or any(np.any(small[:, c] == small[:, c + 1])
                                 for c in range(w - 1)))
                partial_join += joined
                combined_variation += (joined and
                                       (max(rows) == 1 or max(columns) == 1)
                                       and len(np.unique(small)) >= 3)
                repeated_line += (any(np.array_equal(small[a], small[b])
                                      for a in range(h) for b in range(a + 2, h))
                                  or any(np.array_equal(small[:, a], small[:, b])
                                         for a in range(w) for b in range(a + 2, w)))
                if section == "test":
                    prediction = _predict_from_training(
                        [{"input": p["input"].copy(), "output": p["output"].copy()}
                         for p in puzzle["train"]], large.copy())
                    _require(np.array_equal(prediction, small),
                             "training-only prediction failed")
                    same_test += small.shape in train_shapes
                    taller_test += h > max_h and w <= max_w
                    wider_test += w > max_w and h <= max_h
                    both_test += h > max_h and w > max_w
                    smaller_test += any(h * w < th * tw for th, tw in train_shapes)
                    transposed_test += any((h, w) == (tw, th) and th != tw
                                           for th, tw in train_shapes)
                    extreme_test += max(h, w) / min(h, w) >= 3
                    exposed = set().union(*(set(palette) for palette in train_palettes))
                    new_test_color += bool(set(np.unique(small)) - exposed)
    for value, minimum, label in (
        (len(train_counts), 4, "train count"), (len(test_counts), 3, "test count"),
        (len(fingerprints), 40, "puzzle difference"),
        (len(shapes), 18, "logical shapes"), (len(input_shapes), 40, "input shapes"),
        (len(layouts), 100, "layouts"), (len(palettes), 30, "palettes"),
        (len(row_patterns), 25, "row schedules"),
        (len(column_patterns), 25, "column schedules"),
        (same_test, 1, "same-shape test"), (taller_test, 1, "taller test"),
        (wider_test, 1, "wider test"), (both_test, 1, "both-growing test"),
        (smaller_test, 1, "smaller test"),
        (uniform_rows, 1, "uniform row schedule"),
        (uniform_columns, 1, "uniform column schedule"),
        (one_axis, 1, "one-axis expansion"),
        (wide_thin, 1, "wide/thin adjacency"),
        (rich, 1, "rich palette"), (sparse, 1, "two colors"),
        (repeated_line, 1, "repeated nonadjacent line"),
        (short_puzzles, 1, "short puzzle"),
        (transposed_test, 1, "transposed test"),
        (extreme_test, 1, "extreme test aspect ratio"),
        (new_test_color, 1, "new test color"),
        (monochrome_column, 1, "monochrome column"),
        (checkerboard, 1, "checkerboard"),
        (partial_join, 1, "partial local join"),
        (monotone_rows, 1, "monotone row schedule"),
        (monotone_columns, 1, "monotone column schedule"),
        (irregular_rows, 1, "irregular row schedule"),
        (irregular_columns, 1, "irregular column schedule"),
        (pair_palette_variety, 1, "within-puzzle palette variation"),
        (combined_variation, 1, "combined independent variations"),
    ):
        _require(value >= minimum, f"insufficient {label}: {value}")
    a, b = generate_puzzle(73001), generate_puzzle(73001)
    _require(all(np.array_equal(x[key], y[key])
                 for section in ("train", "test")
                 for x, y in zip(a[section], b[section])
                 for key in ("input", "output")), "nondeterministic generation")
    a, b = generate_puzzle(73002, 7, 3), generate_puzzle(73002, 7, 3)
    _require(all(np.array_equal(x[key], y[key])
                 for section in ("train", "test")
                 for x, y in zip(a[section], b[section])
                 for key in ("input", "output")), "nondeterministic explicit counts")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (3, 1), (6, 2), (8, 3))):
        puzzle = generate_puzzle(81000 + index, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts not honored")
    for index, (train, test) in enumerate(((None, 1), (None, 4),
                                           (2, None), (10, None))):
        puzzle = generate_puzzle(82000 + index, train, test)
        _require(train is None or len(puzzle["train"]) == train,
                 "partial train count not honored")
        _require(test is None or len(puzzle["test"]) == test,
                 "partial test count not honored")
    for index, arguments in enumerate(({"num_train": 1}, {"num_train": 11},
                                        {"num_test": 0}, {"num_test": 5},
                                        {"num_train": 2, "num_test": 2},
                                        {"num_train": 3, "num_test": 4},
                                        {"num_train": True}, {"num_test": 1.5})):
        try:
            generate_puzzle(83000 + index, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
