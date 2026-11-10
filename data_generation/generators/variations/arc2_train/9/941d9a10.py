import numpy as np


_ALLOWED_TRAIN_BAND_COUNTS = ((5, 3), (3, 5), (3, 3))

_AUTO_COUNT_OPTIONS = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 3),
    (6, 4),
    (7, 1),
    (8, 2),
    (9, 3),
    (10, 4),
)

_SOURCE_PAIR_ROWS = (
    (
        (
            "0005000050",
            "5555555555",
            "0005000050",
            "5555555555",
            "0005000050",
            "0005000050",
            "5555555555",
            "0005000050",
            "5555555555",
            "0005000050",
        ),
        (
            "1115000050",
            "5555555555",
            "0005000050",
            "5555555555",
            "0005222250",
            "0005222250",
            "5555555555",
            "0005000050",
            "5555555555",
            "0005000053",
        ),
    ),
    (
        (
            "0050000500",
            "0050000500",
            "0050000500",
            "5555555555",
            "0050000500",
            "0050000500",
            "0050000500",
            "5555555555",
            "0050000500",
            "0050000500",
        ),
        (
            "1150000500",
            "1150000500",
            "1150000500",
            "5555555555",
            "0052222500",
            "0052222500",
            "0052222500",
            "5555555555",
            "0050000533",
            "0050000533",
        ),
    ),
    (
        (
            "0500505050",
            "0500505050",
            "0500505050",
            "5555555555",
            "0500505050",
            "0500505050",
            "5555555555",
            "0500505050",
            "0500505050",
            "0500505050",
        ),
        (
            "1500505050",
            "1500505050",
            "1500505050",
            "5555555555",
            "0500525050",
            "0500525050",
            "5555555555",
            "0500505053",
            "0500505053",
            "0500505053",
        ),
    ),
    (
        (
            "0505005050",
            "5555555555",
            "0505005050",
            "0505005050",
            "5555555555",
            "0505005050",
            "5555555555",
            "0505005050",
            "5555555555",
            "0505005050",
        ),
        (
            "1505005050",
            "5555555555",
            "0505005050",
            "0505005050",
            "5555555555",
            "0505225050",
            "5555555555",
            "0505005050",
            "5555555555",
            "0505005053",
        ),
    ),
)


def _rows_to_grid(rows):
    return np.asarray(
        [[ord(character) - ord("0") for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_rows_to_grid(input_rows), _rows_to_grid(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _coerce_count(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _coerce_count("num_train", num_train)
    num_test = _coerce_count("num_test", num_test)

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        return _AUTO_COUNT_OPTIONS[int(seed) % len(_AUTO_COUNT_OPTIONS)]

    if num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for this num_train")
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def sample_puzzle_parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    return {"accents": tuple(colors[:3]), "input_colors": tuple(colors[3:])}


_BAND_COUNTS = tuple((rows, cols)
                     for rows in (3, 5, 7, 9, 11, 13, 15)
                     for cols in (3, 5, 7, 9, 11, 13, 15))


def _composition(rng, count, total):
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total),
                                             size=count - 1, replace=False))
    edges = (0, *cuts, total)
    return tuple(edges[i + 1] - edges[i] for i in range(count))


def _sample_band_sizes(rng, count, dimension=None, spacing=None):
    if dimension is not None:
        assert dimension >= 2 * count - 1
        return _composition(rng, count, dimension - count + 1)

    if spacing is None:
        spacing = ("equal", "near", "varied", "skewed")[int(rng.integers(4))]
    maximum = (30 - count + 1) // count
    if spacing == "equal":
        size = int(rng.integers(1, maximum + 1))
        return (size,) * count
    if spacing == "near":
        base = int(rng.integers(1, maximum + 1))
        sizes = tuple(max(1, base + int(rng.integers(-1, 2)))
                      for _ in range(count))
        if sum(sizes) + count - 1 <= 30:
            return sizes
        return (base,) * count
    if spacing == "skewed":
        sizes = [int(rng.integers(1, 3)) for _ in range(count)]
        index = int(rng.integers(count))
        sizes[index] = int(rng.integers(4, 8))
        if sum(sizes) + count - 1 <= 30:
            return tuple(sizes)
    dimension = int(rng.integers(max(2 * count - 1, 8), 31))
    return _composition(rng, count, dimension - count + 1)


def _slices_and_dividers(sizes):
    bands, dividers = [], []
    cursor = 0
    for index, size in enumerate(sizes):
        bands.append((cursor, cursor + size))
        cursor += size
        if index + 1 < len(sizes):
            dividers.append(cursor)
            cursor += 1
    return tuple(bands), tuple(dividers), cursor


def generate_pair(rng, regime, puzzle_parameters, band_counts=None,
                  canvas=None, input_colors=None, spacing=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    if band_counts is None:
        band_counts = _BAND_COUNTS[int(rng.integers(len(_BAND_COUNTS)))]
    if band_counts not in _BAND_COUNTS:
        raise ValueError("invalid band counts")
    row_count, col_count = band_counts
    if canvas is not None:
        if (canvas[0] < 2 * row_count - 1 or
                canvas[1] < 2 * col_count - 1 or
                max(canvas) > 30):
            raise ValueError("canvas cannot hold these bands")
    if spacing is None:
        row_spacing = col_spacing = None
    else:
        row_spacing, col_spacing = spacing
    row_sizes = _sample_band_sizes(rng, row_count,
                                   None if canvas is None else canvas[0], row_spacing)
    col_sizes = _sample_band_sizes(rng, col_count,
                                   None if canvas is None else canvas[1], col_spacing)
    row_bands, divider_rows, height = _slices_and_dividers(row_sizes)
    col_bands, divider_cols, width = _slices_and_dividers(col_sizes)
    if input_colors is None:
        available = puzzle_parameters["input_colors"]
        chosen = rng.choice(available, size=2, replace=False)
        input_colors = (int(chosen[0]), int(chosen[1]))
    background, divider = input_colors
    if (background == divider or background in puzzle_parameters["accents"]
            or divider in puzzle_parameters["accents"]):
        raise ValueError("input colors must differ from accent colors")
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[list(divider_rows), :] = divider
    input_grid[:, list(divider_cols)] = divider
    output_grid = input_grid.copy()
    for accent, (ri, ci) in zip(
        puzzle_parameters["accents"],
        ((0, 0), (row_count // 2, col_count // 2),
         (row_count - 1, col_count - 1)),
    ):
        rs, re = row_bands[ri]
        cs, ce = col_bands[ci]
        output_grid[rs:re, cs:ce] = accent
    return input_grid, output_grid


def _array_value_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.reshape(-1)))


def _band_intervals(length, dividers):
    bands, start = [], 0
    for divider in dividers:
        divider = int(divider)
        assert start < divider
        bands.append((start, divider))
        start = divider + 1
    assert start < length
    bands.append((start, length))
    return tuple(bands)


def _input_structure(input_grid):
    colors = np.unique(input_grid)
    assert colors.size == 2
    candidates = []
    for raw in colors:
        color = int(raw)
        rows = np.flatnonzero(np.all(input_grid == color, axis=1))
        cols = np.flatnonzero(np.all(input_grid == color, axis=0))
        if len(rows) and len(cols):
            candidates.append((color, rows, cols))
    assert len(candidates) == 1
    divider, rows, cols = candidates[0]
    background = next(int(x) for x in colors if int(x) != divider)
    mask = np.zeros(input_grid.shape, dtype=bool)
    mask[rows, :] = True
    mask[:, cols] = True
    assert np.all(input_grid[mask] == divider)
    assert np.all(input_grid[~mask] == background)
    rb = _band_intervals(input_grid.shape[0], rows)
    cb = _band_intervals(input_grid.shape[1], cols)
    assert (len(rb), len(cb)) in _BAND_COUNTS
    return background, divider, rb, cb


def _inspect_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    background, divider, rb, cb = _input_structure(input_grid)
    targets = ((0, 0), (len(rb) // 2, len(cb) // 2),
               (len(rb) - 1, len(cb) - 1))
    expected = input_grid.copy()
    accents, areas = [], []
    for ri, ci in targets:
        rs, re = rb[ri]
        cs, ce = cb[ci]
        colors = np.unique(output_grid[rs:re, cs:ce])
        assert len(colors) == 1
        accent = int(colors[0])
        accents.append(accent)
        areas.append((re - rs) * (ce - cs))
        expected[rs:re, cs:ce] = accent
    assert len(set(accents)) == 3
    assert not ({background, divider} & set(accents))
    assert np.array_equal(expected, output_grid)
    assert set(int(x) for x in np.unique(output_grid)) == {
        background, divider, *accents}
    return {"accents": tuple(accents), "input_colors": (background, divider),
            "band_counts": (len(rb), len(cb)),
            "row_sizes": tuple(end - start for start, end in rb),
            "col_sizes": tuple(end - start for start, end in cb),
            "target_areas": tuple(areas), "shape": input_grid.shape}


def _is_source_pair(input_grid, output_grid):
    return any(np.array_equal(input_grid, a) and np.array_equal(output_grid, b)
               for a, b in _SOURCE_PAIRS)


def _predict_from_examples(train, test_input):
    learned = [_inspect_pair(pair["input"], pair["output"])["accents"]
               for pair in train]
    assert len(set(learned)) == 1
    _, _, rb, cb = _input_structure(test_input)
    predicted = test_input.copy()
    for accent, (ri, ci) in zip(learned[0],
                                ((0, 0), (len(rb)//2, len(cb)//2),
                                 (len(rb)-1, len(cb)-1))):
        rs, re = rb[ri]
        cs, ce = cb[ci]
        predicted[rs:re, cs:ce] = accent
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen_inputs, seen_pairs, infos = set(), set(), []
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert np.issubdtype(grid.dtype, np.integer)
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.all((grid >= 0) & (grid <= 9))
        key = _array_value_key(pair["input"])
        pair_key = (key, _array_value_key(pair["output"]))
        assert key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
        assert not _is_source_pair(pair["input"], pair["output"])
        infos.append(_inspect_pair(pair["input"], pair["output"]))
    accents = infos[0]["accents"]
    assert all(info["accents"] == accents for info in infos)
    train_infos, test_infos = infos[:nt], infos[nt:]
    assert len({info["band_counts"] for info in train_infos}) >= 2
    assert any(max(info["target_areas"]) > 1 for info in train_infos)
    if any(max(info["band_counts"]) > 5 for info in test_infos):
        assert any(info["band_counts"] == (5, 5) for info in train_infos)
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_examples(puzzle["train"], pair["input"]),
                              pair["output"])


def _training_and_tests(seed, rng, num_train, num_test):
    mode = int(seed) % 4
    compact = int(seed) % 3 == 0 and mode in (0, 1)
    if mode == 0:
        train = [(5, 3), (3, 5)]
        if num_train >= 3:
            train.append((3, 3))
        train_choices = ((3, 3), (5, 3), (3, 5))
        tests = [(5, 5)]
        test_choices = ((5, 5),)
    elif mode == 1:
        train = [(5, 5), (3, 5)]
        train_choices = ((3, 3), (5, 3), (3, 5), (5, 5)) if compact else _BAND_COUNTS
        tests = [(5, 3)]
        test_choices = ((3, 3), (5, 3), (3, 5), (5, 5))
    elif mode == 2:
        train = [(5, 5), (7, 3)]
        train_choices = _BAND_COUNTS
        tests = [((7, 7), (9, 3), (11, 3), (13, 3), (15, 3))
                 [(int(seed) // 4) % 5]]
        test_choices = ((7, 7), (7, 3), (3, 7), (7, 5), (5, 7),
                        (9, 3), (3, 9), (9, 9), (11, 3), (3, 11),
                        (13, 3), (3, 13), (15, 3), (3, 15))
    else:
        train = [(5, 5), (3, 7)]
        train_choices = _BAND_COUNTS
        tests = [((7, 3), (3, 7), (3, 9), (3, 11), (3, 13), (3, 15))
                 [(int(seed) // 4) % 6]]
        test_choices = ((7, 3), (3, 7), (7, 7), (7, 5), (5, 7),
                        (9, 3), (3, 9), (9, 9), (11, 3), (3, 11),
                        (13, 3), (3, 13), (15, 3), (3, 15))
    while len(train) < num_train:
        train.append(train_choices[int(rng.integers(len(train_choices)))])
    while len(tests) < num_test:
        tests.append(test_choices[int(rng.integers(len(test_choices)))])
    train = [train[int(i)] for i in rng.permutation(len(train))]
    return train, tests, compact


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    params = sample_puzzle_parameters(rng)
    train_counts, test_counts, compact = _training_and_tests(
        seed, rng, num_train, num_test)
    variable_colors = int(seed) % 2 == 1
    palette = params["input_colors"]
    first_colors = tuple(int(x) for x in rng.choice(palette, size=2, replace=False))
    puzzle = {"train": [], "test": []}
    seen = set()
    pair_index = 0
    for split, counts in (("train", train_counts), ("test", test_counts)):
        for band_counts in counts:
            for attempt in range(200):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                if variable_colors:
                    colors = tuple(int(x) for x in pair_rng.choice(
                        palette, size=2, replace=False))
                    if pair_index == 1 and colors == first_colors:
                        continue
                else:
                    colors = first_colors
                if pair_index == 0:
                    colors = first_colors
                spacing = tuple(("equal", "near", "varied", "skewed")[
                    int(pair_rng.integers(4))] for _ in range(2))
                input_grid, output_grid = generate_pair(
                    pair_rng, "source_train" if split == "train" else "target_test",
                    params, band_counts, canvas=(10, 10) if compact else None,
                    input_colors=colors, spacing=spacing)
                if (_array_value_key(input_grid) in seen or
                        _is_source_pair(input_grid, output_grid)):
                    continue
                info = _inspect_pair(input_grid, output_grid)
                if (split == "train" and pair_index == num_train - 1 and
                        not any(max(_inspect_pair(p["input"], p["output"])
                                    ["target_areas"]) > 1 for p in puzzle["train"]) and
                        max(info["target_areas"]) == 1):
                    continue
                seen.add(_array_value_key(input_grid))
                puzzle[split].append({"input": input_grid, "output": output_grid})
                pair_index += 1
                break
            else:
                raise RuntimeError("could not generate a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(left, right):
    return all(np.array_equal(a[k], b[k])
               for split in ("train", "test")
               for a, b in zip(left[split], right[split])
               for k in ("input", "output")) and all(
                   len(left[s]) == len(right[s]) for s in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(123456)
    explicit = generate_puzzle(654321, 7, 3)
    assert _puzzles_equal(automatic, generate_puzzle(123456))
    assert _puzzles_equal(explicit, generate_puzzle(654321, 7, 3))
    assert not _puzzles_equal(automatic, explicit)
    for i, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4),
                                  (None, 1), (None, 4),
                                  (2, None), (10, None))):
        p = generate_puzzle(200000 + i, nt, ne)
        if nt is not None:
            assert len(p["train"]) == nt
        if ne is not None:
            assert len(p["test"]) == ne
        validate_generated_puzzle(p)
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(300000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
    puzzles = [generate_puzzle(1000 + i) for i in range(42)]
    train_counts, test_counts, accents, shapes = set(), set(), set(), set()
    layouts, inputs, count_combos = set(), set(), set()
    compact_shared = singleton_train = regular_axes = regular_both = 0
    row_only_regular = col_only_regular = 0
    variable_puzzles = constant_puzzles = high_test_covered = 0
    source_shifts = familiar_tests = thin_strips = large_targets = 0
    wider_train = wider_test = 0
    near_axes = off_center = combined_variable_compact = 0
    combined_variable_singleton = combined_variable_high = 0
    for p in puzzles:
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        infos = [_inspect_pair(x["input"], x["output"])
                 for x in p["train"] + p["test"]]
        accents.add(infos[0]["accents"])
        palettes = {info["input_colors"] for info in infos}
        variable = len(palettes) > 1
        compact = all(info["shape"] == (10, 10) for info in infos)
        singleton = any(1 in info["target_areas"]
                        for info in infos[:len(p["train"])] )
        high = any(max(x["band_counts"]) > 5 for x in infos[len(p["train"]):])
        variable_puzzles += variable
        constant_puzzles += not variable
        compact_shared += compact
        singleton_train += singleton
        high_test_covered += high and any(x["band_counts"] == (5, 5)
                                         for x in infos[:len(p["train"])])
        source_shifts += (set(x["band_counts"] for x in infos[:len(p["train"])])
                          <= {(3, 3), (5, 3), (3, 5)} and
                          all(x["band_counts"] == (5, 5)
                              for x in infos[len(p["train"]):]))
        familiar_tests += any(x["band_counts"] in {(3, 3), (5, 3), (3, 5)}
                              for x in infos[len(p["train"]):])
        wider_train += any(max(x["band_counts"]) >= 9
                           for x in infos[:len(p["train"])])
        wider_test += any(max(x["band_counts"]) >= 9
                          for x in infos[len(p["train"]):])
        combined_variable_compact += variable and compact
        combined_variable_singleton += variable and singleton
        combined_variable_high += variable and high
        for info, pair in zip(infos, p["train"] + p["test"]):
            shapes.add(info["shape"])
            layouts.add((info["band_counts"], info["row_sizes"], info["col_sizes"]))
            count_combos.add(info["band_counts"])
            inputs.add(_array_value_key(pair["input"]))
            row_regular = len(set(info["row_sizes"])) == 1
            col_regular = len(set(info["col_sizes"])) == 1
            regular_axes += row_regular + col_regular
            regular_both += row_regular and col_regular
            row_only_regular += row_regular and not col_regular
            col_only_regular += col_regular and not row_regular
            near_axes += max(info["row_sizes"]) - min(info["row_sizes"]) <= 1
            near_axes += max(info["col_sizes"]) - min(info["col_sizes"]) <= 1
            indices = (0, len(info["row_sizes"]) // 2,
                       len(info["row_sizes"]) - 1)
            for ri, ci in zip(indices, (0, len(info["col_sizes"]) // 2,
                                         len(info["col_sizes"]) - 1)):
                h, w = info["row_sizes"][ri], info["col_sizes"][ci]
                thin_strips += min(h, w) == 1 and max(h, w) > 1
                large_targets += h * w >= 9
            mr = len(info["row_sizes"]) // 2
            mc = len(info["col_sizes"]) // 2
            row_center = (sum(info["row_sizes"][:mr]) + mr +
                          (info["row_sizes"][mr] - 1) / 2)
            col_center = (sum(info["col_sizes"][:mc]) + mc +
                          (info["col_sizes"][mc] - 1) / 2)
            off_center += (abs(row_center - (info["shape"][0] - 1) / 2) >= 2 or
                           abs(col_center - (info["shape"][1] - 1) / 2) >= 2)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(accents) >= 25 and len(shapes) >= 40
    assert len(layouts) >= 100 and len(inputs) >= 200
    assert len(count_combos) >= 7
    assert compact_shared >= 3 and singleton_train >= 8
    assert regular_axes >= 20 and regular_both >= 2
    assert row_only_regular >= 5 and col_only_regular >= 5
    assert variable_puzzles >= 15 and constant_puzzles >= 15
    assert high_test_covered >= 15
    assert source_shifts >= 8 and familiar_tests >= 8
    assert wider_train >= 5 and wider_test >= 5
    assert thin_strips >= 10 and large_targets >= 10
    assert near_axes >= 30 and off_center >= 10
    assert combined_variable_compact >= 1
    assert combined_variable_singleton >= 5 and combined_variable_high >= 5
