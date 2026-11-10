"""Single-marker ARC puzzles with marker-aligned vertical stripes."""

import numpy as np


_COLORS = tuple(range(10))
_PROFILES = ("balanced", "wide", "tall")
_BANDS = {"small": (7, 9), "medium": (10, 19), "large": (20, 30)}
# (height, width, row, column, background, marker, stripe)
_SOURCE_PAIRS = frozenset({
    (5, 5, 2, 2, 0, 6, 4),
    (3, 3, 0, 1, 0, 2, 4),
    (9, 9, 4, 2, 0, 9, 4),
    (12, 12, 3, 5, 0, 3, 4),
})


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _validate_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _validate_count("num_train", num_train, 2, 10)
    test = _validate_count("num_test", num_test, 1, 4)
    selector = int(seed)
    if train is None and test is None:
        if selector % 8 == 0:
            train, test = 2, 1
        else:
            test = 1 + selector % 4
            minimum = max(2, test + 1)
            train = minimum + (selector // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + selector % (11 - minimum)
    elif test is None:
        test = 1 + selector % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _profile(height, width):
    if width >= height + 3:
        return "wide"
    if height >= width + 3:
        return "tall"
    return "balanced"


def _dimensions(rng, band, profile):
    low, high = _BANDS[band]
    # Propose the two sides independently, then condition only on the chosen
    # aspect class. The finite fallback is valid for every band.
    for _ in range(64):
        if profile == "wide":
            height = int(rng.integers(4, high + 1))
            width = int(rng.integers(low, high + 1))
        elif profile == "tall":
            height = int(rng.integers(low, high + 1))
            width = int(rng.integers(4, high + 1))
        else:
            height = int(rng.integers(low, high + 1))
            width = int(rng.integers(low, high + 1))
        if _profile(height, width) == profile:
            return height, width
    if profile == "wide":
        return 4, low
    if profile == "tall":
        return low, 4
    return low, low


def _marker_info(grid):
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 2, "input needs one marker and one background")
    background_index = int(np.argmax(counts))
    _require(int(counts[background_index]) == grid.size - 1,
             "input needs exactly one marker cell")
    background = int(colors[background_index])
    marker = int(colors[1 - background_index])
    row, column = (int(x) for x in np.argwhere(grid == marker)[0])
    return {
        "height": int(grid.shape[0]), "width": int(grid.shape[1]),
        "row": row, "column": column, "background": background,
        "marker": marker,
    }


def _make_pair(height, width, row, column, background, marker, stripe):
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[row, column] = marker
    output_grid = np.full((height, width), background, dtype=np.int8)
    output_grid[:row + 1, column % 2::2] = stripe
    output_grid[row + 1, column] = marker
    return {"input": input_grid, "output": output_grid}


def _signature(grid):
    return grid.shape, grid.tobytes()


def _choose_pair(
    rng, dimensions, row, preferred_column, background, marker, stripe,
    used_inputs,
):
    height, width = dimensions
    parity = preferred_column % 2
    columns = [preferred_column] + [
        c for c in range(parity, width, 2) if c != preferred_column
    ]
    rows = [row] + [r for r in range(height - 1) if r != row]
    markers = [marker] + [
        color for color in _COLORS
        if color not in (background, stripe, marker)
    ]
    for candidate_row in rows:
        for candidate_column in columns:
            for candidate_marker in markers:
                spec = (
                    height, width, candidate_row, candidate_column,
                    background, candidate_marker, stripe,
                )
                if spec in _SOURCE_PAIRS:
                    continue
                pair = _make_pair(*spec[:4], *spec[4:])
                if _signature(pair["input"]) not in used_inputs:
                    used_inputs.add(_signature(pair["input"]))
                    return pair
    raise AssertionError("no unique pair could be constructed")


def _test_row(rng, height, mode):
    if mode == 0:
        return 0
    if mode == 1:
        return height - 2
    if mode == 2:
        return int(rng.integers(1, max(2, height // 3)))
    if mode == 3:
        return int(rng.integers(max(1, 2 * height // 3), height - 1))
    return int(rng.integers(0, height - 1))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    stripe = int(rng.integers(0, 10))  # Learned once from the training outputs.
    available_backgrounds = [c for c in _COLORS if c != stripe]
    source_like = seed % 8 == 0 and train_count == 2
    extrapolate = seed % 4 == 0 and not source_like

    if source_like:
        sizes = [5, 9]
        rng.shuffle(sizes)
        train_dimensions = [(size, size) for size in sizes]
    else:
        profiles = list(rng.permutation(_PROFILES))
        while len(profiles) < train_count:
            profiles.append(_PROFILES[int(rng.integers(0, 3))])
        if extrapolate:
            bands = ["small", "medium"]
        else:
            bands = list(rng.choice(tuple(_BANDS), size=2, replace=False))
        while len(bands) < train_count:
            bands.append(
                ("small", "medium")[int(rng.integers(0, 2))]
                if extrapolate else tuple(_BANDS)[int(rng.integers(0, 3))]
            )
        train_dimensions = [
            _dimensions(rng, band, profile)
            for band, profile in zip(bands, profiles)
        ]

    backgrounds = list(rng.choice(available_backgrounds, size=2, replace=False))
    while len(backgrounds) < train_count:
        backgrounds.append(int(rng.choice(available_backgrounds)))
    first_marker = int(rng.choice(
        [c for c in _COLORS if c not in (stripe, int(backgrounds[0]))]
    ))
    second_marker = int(rng.choice(
        [c for c in _COLORS if c not in (stripe, int(backgrounds[1]), first_marker)]
    ))
    marker_colors = [first_marker, second_marker]
    while len(marker_colors) < train_count:
        background = int(backgrounds[len(marker_colors)])
        marker_colors.append(int(rng.choice(
            [c for c in _COLORS if c not in (stripe, background)]
        )))

    top_index = int(rng.integers(0, train_count))
    positive_index = (top_index + int(rng.integers(1, train_count))) % train_count
    even_index = int(rng.integers(0, train_count))
    odd_index = (even_index + int(rng.integers(1, train_count))) % train_count
    used_inputs = set()
    train = []
    for index, (height, width) in enumerate(train_dimensions):
        if index == top_index:
            row = 0
        elif index == positive_index:
            row = int(rng.integers(1, height - 1))
        else:
            row = int(rng.integers(0, height - 1))
        parity = 0 if index == even_index else (
            1 if index == odd_index else int(rng.integers(0, 2))
        )
        column = int(rng.choice(np.arange(parity, width, 2)))
        train.append(_choose_pair(
            rng, (height, width), row, column, int(backgrounds[index]),
            marker_colors[index], stripe, used_inputs,
        ))

    seen_train_markers = {_marker_info(pair["input"])["marker"] for pair in train}
    test = []
    for index in range(test_count):
        if source_like:
            side = int(rng.choice((12, 14, 16, 18, 20, 22, 24)))
            dimensions = (side, side)
        elif extrapolate and index == 0:
            profile = _PROFILES[int(rng.integers(0, 3))]
            dimensions = _dimensions(rng, "large", profile)
        elif seed % 4 == 1 and index == 0:
            dimensions = train_dimensions[int(rng.integers(0, train_count))]
        else:
            profile = _PROFILES[int(rng.integers(0, 3))]
            band = tuple(_BANDS)[int(rng.integers(0, 3))]
            dimensions = _dimensions(rng, band, profile)
        height, width = dimensions
        background = int(rng.choice(available_backgrounds))
        reused = [c for c in seen_train_markers if c not in (background, stripe)]
        novel = [
            c for c in _COLORS
            if c not in (background, stripe) and c not in seen_train_markers
        ]
        if (seed + index) % 3 == 0:
            palette = novel or reused
        elif (seed + index) % 3 == 1:
            palette = reused or novel
        else:
            palette = reused + novel
        marker = int(rng.choice(palette))
        row = _test_row(rng, height, (seed + 2 * index) % 5)
        edge_mode = (seed + 3 * index) % 7
        if edge_mode == 0:
            column = 0
        elif edge_mode == 1:
            column = width - 1
        else:
            column = int(rng.integers(0, width))
        test.append(_choose_pair(
            rng, dimensions, row, column, background, marker, stripe,
            used_inputs,
        ))

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list),
             "train and test must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    information = {"train": [], "test": []}
    stripes = set()
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output")
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2-D NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer),
                         "grid must use integers")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         "grid dimensions out of range")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid colors out of range")
            _require(input_grid.shape == output_grid.shape
                     and input_grid.dtype == output_grid.dtype,
                     "input and output shape or dtype mismatch")
            _require(input_grid.shape[0] >= 3 and input_grid.shape[1] >= 3,
                     "canvas too small to demonstrate the stripes")
            input_signature = _signature(input_grid)
            pair_signature = (input_signature, _signature(output_grid))
            _require(input_signature not in seen_inputs and pair_signature not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            info = _marker_info(input_grid)
            height, width = info["height"], info["width"]
            row, column = info["row"], info["column"]
            background, marker = info["background"], info["marker"]
            _require(row < height - 1, "marker cannot move down")
            stripe_candidates = set(int(c) for c in np.unique(output_grid))
            stripe_candidates.difference_update((background, marker))
            _require(len(stripe_candidates) == 1, "output needs one stripe color")
            stripe = stripe_candidates.pop()
            stripes.add(stripe)
            _require(stripe not in (background, marker), "color roles collide")
            for r in range(height):
                for c in range(width):
                    expected = (
                        stripe if r <= row and c % 2 == column % 2
                        else marker if (r, c) == (row + 1, column)
                        else background
                    )
                    _require(int(output_grid[r, c]) == expected,
                             "marker/stripe transformation is incorrect")
            _require(
                (height, width, row, column, background, marker, stripe)
                not in _SOURCE_PAIRS,
                "exact source pair copy",
            )
            info["stripe"] = stripe
            information[split].append(info)
    _require(len(stripes) == 1, "stripe color must be shared across the puzzle")
    train_info = information["train"]
    _require({item["column"] % 2 for item in train_info} == {0, 1},
             "training must show both stripe phases")
    _require(len({item["row"] for item in train_info}) >= 2,
             "training must show different stripe depths")
    _require(any(item["row"] == 0 for item in train_info),
             "training must show a top-row marker")
    _require(len({item["background"] for item in train_info}) >= 2,
             "training must show variable backgrounds")
    _require(len({item["marker"] for item in train_info}) >= 2,
             "training must show marker color preservation")
    _require(len({(item["height"], item["width"]) for item in train_info}) >= 2,
             "training must show distinct canvas sizes")
    if len(train_info) >= 3:
        _require({_profile(item["height"], item["width"]) for item in train_info}
                 == set(_PROFILES),
                 "training must show balanced, wide, and tall canvases")


def _puzzle_signature(puzzle):
    return tuple(
        (split, tuple(
            (_signature(pair["input"]), _signature(pair["output"]))
            for pair in puzzle[split]
        ))
        for split in ("train", "test")
    )


def _predict_from_training(train, test_input):
    # Reconstruct the only new output color from visible training grids.
    stripes = set()
    for pair in train:
        info = _marker_info(pair["input"])
        colors = set(int(c) for c in np.unique(pair["output"]))
        colors.difference_update((info["background"], info["marker"]))
        _require(len(colors) == 1, "training does not reveal the stripe color")
        stripes.update(colors)
    _require(len(stripes) == 1, "contradictory training stripe colors")
    stripe = next(iter(stripes))
    info = _marker_info(test_input)
    output = np.full(test_input.shape, info["background"], dtype=test_input.dtype)
    for r in range(info["row"] + 1):
        for c in range(info["width"]):
            if c % 2 == info["column"] % 2:
                output[r, c] = stripe
    output[info["row"] + 1, info["column"]] = info["marker"]
    return output


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, layouts, stripes, backgrounds = set(), set(), set(), set()
    test_rows, test_width_parities, test_profiles = set(), set(), set()
    rectangular_train = odd_test = overlapping = extrapolated = 0
    reused_test = novel_test = repeated_train = 0
    source_like = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_signature(puzzle))
        train_info = [_marker_info(pair["input"]) for pair in puzzle["train"]]
        test_info = [_marker_info(pair["input"]) for pair in puzzle["test"]]
        train_shapes = {(x["height"], x["width"]) for x in train_info}
        train_markers = {x["marker"] for x in train_info}
        if len(train_markers) < len(train_info):
            repeated_train += 1
        if all(x["height"] == x["width"] and x["height"] % 2
               for x in train_info) and all(
               x["height"] > max(y["height"] for y in train_info)
               and x["width"] > max(y["width"] for y in train_info)
               and x["width"] % 2 == 0 for x in test_info):
            source_like += 1
        if any(x["height"] != x["width"] for x in train_info):
            rectangular_train += 1
        if any(x["width"] % 2 for x in test_info):
            odd_test += 1
        if any((x["height"], x["width"]) in train_shapes for x in test_info):
            overlapping += 1
        if any(max(x["height"], x["width"]) >
               max(max(y["height"], y["width"]) for y in train_info)
               for x in test_info):
            extrapolated += 1
        for pair, info in zip(puzzle["train"] + puzzle["test"],
                              train_info + test_info):
            colors = set(int(c) for c in np.unique(pair["output"]))
            colors.difference_update((info["background"], info["marker"]))
            stripe = next(iter(colors))
            stripes.add(stripe)
            backgrounds.add(info["background"])
            shapes.add((info["height"], info["width"]))
            layouts.add((info["height"], info["width"], info["row"],
                         info["column"], info["background"]))
            for r in range(info["height"]):
                for c in range(info["width"]):
                    expected = (
                        stripe if r <= info["row"] and c % 2 == info["column"] % 2
                        else info["marker"] if (r, c) == (info["row"] + 1, info["column"])
                        else info["background"]
                    )
                    _require(int(pair["output"][r, c]) == expected,
                             "independent cellwise check failed")
        for pair, info in zip(puzzle["test"], test_info):
            _require(np.array_equal(
                _predict_from_training(puzzle["train"], pair["input"]),
                pair["output"],
            ), "training-only prediction failed")
            reused_test += info["marker"] in train_markers
            novel_test += info["marker"] not in train_markers
            test_width_parities.add(info["width"] % 2)
            test_profiles.add(_profile(info["height"], info["width"]))
            test_rows.add("top" if info["row"] == 0 else
                          "penultimate" if info["row"] == info["height"] - 2 else
                          "interior")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(signatures) == 42 and len(shapes) >= 35 and len(layouts) >= 150,
             "structural diversity is too narrow")
    _require(len(stripes) >= 8 and len(backgrounds) == 10,
             "color roles lack diversity")
    _require(rectangular_train >= 20 and odd_test >= 10 and overlapping >= 4
             and extrapolated >= 4 and source_like >= 1,
             "size and shape regimes lack diversity")
    _require(reused_test >= 8 and novel_test >= 8 and repeated_train >= 4,
             "marker color reuse and novelty lack diversity")
    _require(test_rows == {"top", "penultimate", "interior"}
             and test_width_parities == {0, 1}
             and test_profiles == set(_PROFILES),
             "test placement and aspect regimes lack diversity")
    for seed, train_count, test_count in (
        (2000, 2, 1), (2001, 10, 1), (2002, 10, 4),
        (3000, None, 1), (3001, None, 4),
        (3002, 2, None), (3003, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "train count not honored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "test count not honored")
    for seed, train_count, test_count in (
        (777, None, None), (778, 7, 3), (779, 2, 1),
    ):
        _require(_puzzle_signature(generate_puzzle(seed, train_count, test_count))
                 == _puzzle_signature(generate_puzzle(seed, train_count, test_count)),
                 "generation is not deterministic")
    _require(len({_puzzle_signature(generate_puzzle(1000 + i, 6, 3))
                  for i in range(12)}) >= 11,
             "different seeds repeat puzzles")
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": True},
        {"num_test": 2.0},
    ):
        try:
            generate_puzzle(4000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
