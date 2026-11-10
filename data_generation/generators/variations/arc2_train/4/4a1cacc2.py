import numpy as np


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_signature(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return (
        _grid_signature(input_grid),
        _grid_signature(output_grid),
    )


def _parse_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_ROWS = (
    (
        (
            "88888888",
            "88888888",
            "88888888",
            "88488888",
            "88888888",
            "88888888",
        ),
        (
            "88888888",
            "88888888",
            "88888888",
            "44488888",
            "44488888",
            "44488888",
        ),
    ),
    (
        (
            "888888",
            "888888",
            "888898",
            "888888",
            "888888",
            "888888",
        ),
        (
            "888899",
            "888899",
            "888899",
            "888888",
            "888888",
            "888888",
        ),
    ),
    (
        (
            "88888888",
            "88888888",
            "88888888",
            "88888888",
            "88688888",
            "88888888",
            "88888888",
            "88888888",
            "88888888",
            "88888888",
        ),
        (
            "66688888",
            "66688888",
            "66688888",
            "66688888",
            "66688888",
            "88888888",
            "88888888",
            "88888888",
            "88888888",
            "88888888",
        ),
    ),
    (
        (
            "8868",
            "8888",
            "8888",
            "8888",
        ),
        (
            "8866",
            "8888",
            "8888",
            "8888",
        ),
    ),
    (
        (
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888848",
            "8888888888",
            "8888888888",
        ),
        (
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888888",
            "8888888844",
            "8888888844",
            "8888888844",
        ),
    ),
)

_SOURCE_PAIR_SIGNATURES = frozenset(
    _pair_signature(_parse_grid(input_rows), _parse_grid(output_rows))
    for input_rows, output_rows in _SOURCE_ROWS
)

_AUTO_COUNT_OPTIONS = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 3),
    (6, 1),
    (6, 2),
    (7, 3),
    (7, 4),
    (8, 2),
    (8, 4),
    (9, 3),
    (10, 4),
)


def _normalise_integer(value, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _choose_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        num_train = _normalise_integer(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        num_test = _normalise_integer(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        return _AUTO_COUNT_OPTIONS[seed % len(_AUTO_COUNT_OPTIONS)]

    if num_train is None:
        legal_train_counts = tuple(
            range(max(2, num_test + 1), 11)
        )
        if not legal_train_counts:
            raise ValueError(
                "no legal num_train exists for the requested num_test"
            )
        num_train = legal_train_counts[
            seed % len(legal_train_counts)
        ]

    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError(
                "no legal num_test exists for the requested num_train"
            )
        num_test = 1 + seed % maximum_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _solve_input(input_grid: np.ndarray) -> np.ndarray:
    values, counts = np.unique(input_grid, return_counts=True)
    if len(values) != 2 or np.count_nonzero(counts == 1) != 1:
        raise ValueError("input needs one seed on a uniform field")
    seed = int(values[np.flatnonzero(counts == 1)[0]])
    row, column = (int(x) for x in np.argwhere(input_grid == seed)[0])
    height, width = input_grid.shape
    if row * 2 == height - 1 or column * 2 == width - 1:
        raise ValueError("the nearest boundary must be unique")
    row_start, row_stop = (0, row + 1) if row * 2 < height - 1 else (row, height)
    col_start, col_stop = (0, column + 1) if column * 2 < width - 1 else (column, width)
    output = input_grid.copy()
    output[row_start:row_stop, col_start:col_stop] = seed
    return output


def _sample_dimension(rng: np.random.Generator, minimum: int, cap: int = 30) -> int:
    # The two axes are drawn separately. The mixture retains small source-like
    # canvases while reaching the ARC limit in both splits.
    bands = ((minimum, min(3, cap), 0.09),
             (max(minimum, 4), min(13, cap), 0.43),
             (max(minimum, 14), min(24, cap), 0.36),
             (max(minimum, 25), cap, 0.12))
    legal = [(lo, hi, weight) for lo, hi, weight in bands if lo <= hi]
    weights = np.array([item[2] for item in legal], dtype=float)
    weights /= weights.sum()
    lo, hi, _ = legal[int(rng.choice(len(legal), p=weights))]
    return int(rng.integers(lo, hi + 1))


def _sample_shape(rng: np.random.Generator, geometry: str, cap: int = 30) -> tuple[int, int]:
    row_min = 2 if geometry in ("row_strip", "identity") else 4
    col_min = 2 if geometry in ("column_strip", "identity") else 4
    height = _sample_dimension(rng, row_min, cap)
    width = _sample_dimension(rng, col_min, cap)
    if int(rng.integers(0, 5)) == 0:
        common = _sample_dimension(rng, max(row_min, col_min), cap)
        height = width = common
    if geometry == "row_strip" and int(rng.integers(0, 3)) == 0:
        height = int(rng.integers(2, 4))
    if geometry == "column_strip" and int(rng.integers(0, 3)) == 0:
        width = int(rng.integers(2, 4))
    return height, width


def _axis_position(rng: np.random.Generator, length: int, toward_low: bool,
                   on_edge: bool) -> int:
    if on_edge:
        return 0 if toward_low else length - 1
    candidates = [coordinate for coordinate in range(length)
                  if coordinate != (0 if toward_low else length - 1)
                  and ((coordinate * 2 < length - 1) if toward_low
                       else (coordinate * 2 > length - 1))]
    if not candidates:
        raise ValueError("shape cannot support expanding geometry")
    draw = int(rng.integers(0, 5))
    if draw == 0:
        return candidates[0] if toward_low else candidates[-1]
    if draw == 1:
        return candidates[-1] if toward_low else candidates[0]
    return int(rng.choice(candidates))


def _make_pair(rng: np.random.Generator, orientation: str, geometry: str,
               colors: tuple[int, int], shape: tuple[int, int]) -> dict:
    height, width = shape
    row = _axis_position(rng, height, orientation[0] == "T",
                         geometry in ("row_strip", "identity"))
    column = _axis_position(rng, width, orientation[1] == "L",
                            geometry in ("column_strip", "identity"))
    background, seed = colors
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[row, column] = seed
    return {"input": input_grid, "output": _solve_input(input_grid)}


def _choose_orientations(rng: np.random.Generator, num_train: int,
                         num_test: int) -> tuple[list[str], list[str], bool]:
    corners = ("TL", "TR", "BL", "BR")
    if num_train == 2:
        # Two examples never have to establish a fourth, unobserved corner.
        if int(rng.integers(0, 3)) == 0:
            one = str(rng.choice(corners))
            return [one, one], [one] * num_test, False
        train = list(rng.choice(corners, size=2, replace=False))
        return train, [train[int(rng.integers(0, 2))] for _ in range(num_test)], False

    mode = int(rng.choice(4, p=(0.35, 0.29, 0.22, 0.14)))
    if mode == 0:  # rotate the omitted corner from puzzle to puzzle
        withheld = str(rng.choice(corners))
        base = [corner for corner in corners if corner != withheld]
        rng.shuffle(base)
        train = base + [str(rng.choice(base)) for _ in range(num_train - 3)]
        test = [withheld]
        test += [str(rng.choice(corners)) for _ in range(num_test - 1)]
        return train, test, True
    if mode == 1:  # train/test corner sets overlap
        count = min(num_train, int(rng.integers(2, 5)))
        base = list(rng.choice(corners, size=count, replace=False))
        train = base + [str(rng.choice(base)) for _ in range(num_train - count)]
        test = [str(rng.choice(base)) for _ in range(num_test)]
        return train, test, False
    if mode == 2:  # mixed test directions, when there is more than one test
        count = min(num_train, int(rng.integers(3, 5)))
        base = list(rng.choice(corners, size=count, replace=False))
        train = base + [str(rng.choice(base)) for _ in range(num_train - count)]
        if num_test >= 2:
            test = list(rng.choice(base, size=2, replace=False))
            test += [str(rng.choice(base)) for _ in range(num_test - 2)]
        else:
            test = [str(rng.choice(base))]
        return train, test, False
    one = str(rng.choice(corners))
    return [one] * num_train, [one] * num_test, False


def _sample_geometry(rng: np.random.Generator) -> str:
    return str(rng.choice(("rectangle", "row_strip", "column_strip", "identity"),
                          p=(0.64, 0.16, 0.16, 0.04)))


def _choose_colors(rng: np.random.Generator, mode: int,
                   index: int, train_colors: list[tuple[int, int]],
                   first_colors: tuple[int, int] | None = None) -> tuple[int, int]:
    if mode == 0 and first_colors is not None:
        return first_colors
    if mode == 2 and index == 1 and train_colors:
        background, seed = train_colors[0]
        return seed, background
    if mode == 1 and index == -1 and train_colors and int(rng.integers(0, 2)) == 0:
        seen = {seed for _, seed in train_colors}
        unseen = [color for color in range(10) if color not in seen]
        if unseen:
            seed = int(rng.choice(unseen))
            background = int(rng.choice([color for color in range(10) if color != seed]))
            return background, seed
    selected = rng.choice(10, size=2, replace=False)
    return int(selected[0]), int(selected[1])


def _unique_pair(rng: np.random.Generator, orientation: str, geometry: str,
                 colors: tuple[int, int], shape: tuple[int, int] | None,
                 seen_inputs: set, seen_pairs: set, cap: int) -> dict:
    for attempt in range(160):
        candidate_shape = shape if attempt < 20 and shape is not None else _sample_shape(rng, geometry, cap)
        pair = _make_pair(rng, orientation, geometry, colors, candidate_shape)
        input_signature = _grid_signature(pair["input"])
        pair_signature = _pair_signature(pair["input"], pair["output"])
        if (input_signature not in seen_inputs and pair_signature not in seen_pairs
                and pair_signature not in _SOURCE_PAIR_SIGNATURES):
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            return pair
    raise RuntimeError("could not construct a distinct pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _normalise_integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    num_train, num_test = _choose_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    train_orientations, test_orientations, held_out = _choose_orientations(rng, num_train, num_test)
    palette_mode = int(rng.choice(3, p=(0.18, 0.58, 0.24)))
    base_colors = _choose_colors(rng, palette_mode, 0, [])
    size_mode = int(rng.choice(4, p=(0.47, 0.19, 0.16, 0.18)))
    train_cap = 20 if size_mode == 3 else 30
    seen_inputs, seen_pairs = set(), set()
    train, test = [], []
    train_colors = []
    for index, orientation in enumerate(train_orientations):
        geometry = "rectangle" if index < (3 if held_out else 2) else _sample_geometry(rng)
        colors = _choose_colors(rng, palette_mode, index, train_colors, base_colors)
        if (palette_mode == 1 and index >= 2 and orientation != train_orientations[0]
                and int(rng.integers(0, 5)) == 0):
            seed_color = train_colors[0][1]
            colors = (int(rng.choice([color for color in range(10) if color != seed_color])), seed_color)
        pair = _unique_pair(rng, orientation, geometry, colors, None,
                            seen_inputs, seen_pairs, train_cap)
        train.append(pair)
        train_colors.append(colors)

    train_shapes = [pair["input"].shape for pair in train]
    for index, orientation in enumerate(test_orientations):
        geometry = "rectangle" if held_out and index == 0 else _sample_geometry(rng)
        colors = _choose_colors(rng, palette_mode, -1, train_colors, base_colors)
        shape = None
        if index == 0 and geometry == "rectangle":
            if size_mode == 1:
                shape = train_shapes[int(rng.integers(0, min(3, len(train_shapes))))]
            elif size_mode == 2:
                tall, wide = max(h for h, _ in train_shapes), max(w for _, w in train_shapes)
                shape = (int(rng.integers(4, max(5, tall))),
                         int(rng.integers(4, max(5, wide))))
            elif size_mode == 3:
                tall, wide = max(h for h, _ in train_shapes), max(w for _, w in train_shapes)
                shape = (int(rng.integers(tall + 1, 31)),
                         int(rng.integers(wide + 1, 31)))
        pair = _unique_pair(rng, orientation, geometry, colors, shape,
                            seen_inputs, seen_pairs, 30)
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyse_pair(pair: dict) -> dict:
    input_grid, output_grid = pair["input"], pair["output"]
    assert input_grid.shape == output_grid.shape
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2 and np.count_nonzero(counts == 1) == 1
    seed = int(values[np.flatnonzero(counts == 1)[0]])
    background = int(values[np.flatnonzero(counts != 1)[0]])
    row, column = (int(x) for x in np.argwhere(input_grid == seed)[0])
    height, width = input_grid.shape
    assert 2 <= height <= 30 and 2 <= width <= 30
    assert row * 2 != height - 1 and column * 2 != width - 1
    vertical = "T" if row * 2 < height - 1 else "B"
    horizontal = "L" if column * 2 < width - 1 else "R"
    r0, r1 = (0, row + 1) if vertical == "T" else (row, height)
    c0, c1 = (0, column + 1) if horizontal == "L" else (column, width)
    expected = input_grid.copy()
    expected[r0:r1, c0:c1] = seed
    assert np.array_equal(output_grid, expected), "incorrect nearest-boundary rectangle"
    assert set(int(x) for x in np.unique(output_grid)) == {background, seed}
    return {"background": background, "seed": seed, "orientation": vertical + horizontal,
            "row_span": r1-r0, "column_span": c1-c0, "row": row, "column": column}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs, pairs = set(), set()
    facts = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_signature = _grid_signature(pair["input"])
            pair_signature = _pair_signature(pair["input"], pair["output"])
            assert input_signature not in inputs and pair_signature not in pairs
            assert pair_signature not in _SOURCE_PAIR_SIGNATURES
            inputs.add(input_signature)
            pairs.add(pair_signature)
            facts[section].append(_analyse_pair(pair))
    assert any(np.any(pair["input"] != pair["output"]) for pair in train)
    train_corners = {item["orientation"] for item in facts["train"]}
    test_corners = {item["orientation"] for item in facts["test"]}
    if not test_corners <= train_corners:
        # Three expanding corners expose both independent axis choices.
        assert len(train_corners) >= 3
        assert len({item["orientation"][0] for item in facts["train"]}) == 2
        assert len({item["orientation"][1] for item in facts["train"]}) == 2
        for corner in train_corners:
            assert any(item["orientation"] == corner and item["row_span"] >= 2
                       and item["column_span"] >= 2 for item in facts["train"])
    if len(train) == 2:
        assert test_corners <= train_corners


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[section]) == len(second[section]) and
               all(np.array_equal(a[key], b[key]) for a, b in zip(first[section], second[section])
                   for key in ("input", "output")) for section in ("train", "test"))


def _predict_from_demonstrations(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # The predictor sees only returned demonstrations and the test input.
    observed = [set(), set()]
    for pair in train:
        source, target = pair["input"], pair["output"]
        values, counts = np.unique(source, return_counts=True)
        seed = int(values[np.argmin(counts)])
        filled = np.argwhere(target == seed)
        row, col = (int(x) for x in np.argwhere(source == seed)[0])
        observed[0].add("T" if np.min(filled[:, 0]) < row else "B" if np.max(filled[:, 0]) > row else
                        ("T" if row == 0 else "B"))
        observed[1].add("L" if np.min(filled[:, 1]) < col else "R" if np.max(filled[:, 1]) > col else
                        ("L" if col == 0 else "R"))
    values, counts = np.unique(test_input, return_counts=True)
    seed = int(values[np.argmin(counts)])
    row, col = (int(x) for x in np.argwhere(test_input == seed)[0])
    height, width = test_input.shape
    vertical = next(iter(observed[0])) if len(observed[0]) == 1 else ("T" if row * 2 < height - 1 else "B")
    horizontal = next(iter(observed[1])) if len(observed[1]) == 1 else ("L" if col * 2 < width - 1 else "R")
    r0, r1 = (0, row + 1) if vertical == "T" else (row, height)
    c0, c1 = (0, col + 1) if horizontal == "L" else (col, width)
    answer = test_input.copy()
    answer[r0:r1, c0:c1] = seed
    return answer


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rotated = {section: [{key: np.rot90(pair[key]) for key in ("input", "output")}
                         for pair in puzzles[2][section]] for section in ("train", "test")}
    validate_generated_puzzle(rotated)
    recolored = {section: [{key: pair[key].copy() for key in ("input", "output")}
                           for pair in puzzles[2][section]] for section in ("train", "test")}
    recolored["train"][0] = {key: ((grid.astype(np.int16) + 1) % 10).astype(np.int8)
                              for key, grid in recolored["train"][0].items()}
    validate_generated_puzzle(recolored)
    assert all(_puzzles_equal(generate_puzzle(seed), puzzle) for seed, puzzle in enumerate(puzzles))
    assert len({tuple(_pair_signature(pair["input"], pair["output"]) for section in ("train", "test")
                      for pair in puzzle[section]) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    orientations = {section: set() for section in ("train", "test")}
    shapes = {section: set() for section in ("train", "test")}
    strips = {section: 0 for section in ("train", "test")}
    mixed_colors = 0
    mixed_test_directions = 0
    held_out = 0
    matched_size = 0
    smaller_test = 0
    larger_test = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_facts = [_analyse_pair(pair) for pair in puzzle["train"]]
        test_facts = [_analyse_pair(pair) for pair in puzzle["test"]]
        mixed_colors += len({item["background"] for item in train_facts + test_facts}) > 1
        mixed_test_directions += len({item["orientation"] for item in test_facts}) > 1
        held_out += bool({item["orientation"] for item in test_facts} -
                         {item["orientation"] for item in train_facts})
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        test_shapes = {pair["input"].shape for pair in puzzle["test"]}
        matched_size += bool(train_shapes & test_shapes)
        smaller_test += any(h < max(x for x, _ in train_shapes) and w < max(y for _, y in train_shapes)
                            for h, w in test_shapes)
        larger_test += any(h > max(x for x, _ in train_shapes) and w > max(y for _, y in train_shapes)
                           for h, w in test_shapes)
        for section, local_facts in (("train", train_facts), ("test", test_facts)):
            for pair, item in zip(puzzle[section], local_facts):
                orientations[section].add(item["orientation"])
                shapes[section].add(pair["input"].shape)
                strips[section] += (item["row_span"] == 1) != (item["column_span"] == 1)
        for pair in puzzle["test"]:
            assert np.array_equal(_predict_from_demonstrations(puzzle["train"], pair["input"].copy()),
                                  pair["output"])
    assert orientations["train"] == orientations["test"] == {"TL", "TR", "BL", "BR"}
    assert len(shapes["train"]) >= 35 and len(shapes["test"]) >= 20
    assert strips["train"] > 0 and strips["test"] > 0
    assert mixed_colors >= 15 and mixed_test_directions > 0 and held_out > 0
    assert matched_size > 0 and smaller_test > 0 and larger_test > 0
    for seed, ntrain, ntest in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 5, 2)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        assert len(puzzle["train"]) == ntrain and len(puzzle["test"]) == ntest
    for seed, ntrain, ntest in ((201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
