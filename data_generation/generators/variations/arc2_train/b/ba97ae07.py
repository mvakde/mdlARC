import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows_key(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )


def _source_pair_key(
    input_rows: tuple[str, ...],
    output_rows: tuple[str, ...],
) -> tuple:
    return (_rows_key(input_rows), _rows_key(output_rows))


_SOURCE_PAIR_KEYS = frozenset(
    {
        _source_pair_key(
            (
                "0010000",
                "0010000",
                "0010000",
                "7777777",
                "0010000",
                "0010000",
                "0010000",
                "0010000",
            ),
            (
                "0010000",
                "0010000",
                "0010000",
                "7717777",
                "0010000",
                "0010000",
                "0010000",
                "0010000",
            ),
        ),
        _source_pair_key(
            (
                "006600000",
                "006600000",
                "006600000",
                "111111111",
                "006600000",
                "006600000",
                "006600000",
            ),
            (
                "006600000",
                "006600000",
                "006600000",
                "116611111",
                "006600000",
                "006600000",
                "006600000",
            ),
        ),
        _source_pair_key(
            (
                "030000",
                "030000",
                "030000",
                "030000",
                "232222",
                "030000",
                "030000",
                "030000",
            ),
            (
                "030000",
                "030000",
                "030000",
                "030000",
                "222222",
                "030000",
                "030000",
                "030000",
            ),
        ),
        _source_pair_key(
            (
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "3338833333333",
                "3338833333333",
                "3338833333333",
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "0008800000000",
            ),
            (
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "3333333333333",
                "3333333333333",
                "3333333333333",
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "0008800000000",
                "0008800000000",
            ),
        ),
        _source_pair_key(
            (
                "004400",
                "004400",
                "554455",
                "554455",
                "004400",
                "004400",
                "004400",
                "004400",
                "004400",
                "004400",
                "004400",
            ),
            (
                "004400",
                "004400",
                "555555",
                "555555",
                "004400",
                "004400",
                "004400",
                "004400",
                "004400",
                "004400",
                "004400",
            ),
        ),
    }
)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return (_grid_key(input_grid), _grid_key(output_grid))


def _is_consecutive(indices: np.ndarray) -> bool:
    return bool(
        indices.size > 0
        and (
            indices.size == 1
            or np.all(np.diff(indices) == 1)
        )
    )


def _analyze_grid(grid: np.ndarray) -> dict:
    values = [int(x) for x in np.unique(grid)]
    _require(
        len(values) == 3,
        "a band grid must contain exactly three colors",
    )

    height, width = grid.shape
    candidates = []

    # Candidate in which the vertical band is on top.
    for vertical_color in values:
        vertical_columns = np.flatnonzero(
            np.all(grid == vertical_color, axis=0)
        )
        if not _is_consecutive(vertical_columns) or vertical_columns.size > width - 3 or 2 * vertical_columns.size >= width:
            continue

        outside_columns = np.ones(width, dtype=bool)
        outside_columns[vertical_columns] = False

        for horizontal_color in values:
            if horizontal_color == vertical_color:
                continue

            background = next(
                color
                for color in values
                if color not in (vertical_color, horizontal_color)
            )
            horizontal_rows = np.flatnonzero(
                np.all(
                    grid[:, outside_columns] == horizontal_color,
                    axis=1,
                )
            )
            if not _is_consecutive(horizontal_rows) or horizontal_rows.size > height - 3 or 2 * horizontal_rows.size >= height:
                continue

            v0 = int(vertical_columns[0])
            v1 = int(vertical_columns[-1]) + 1
            h0 = int(horizontal_rows[0])
            h1 = int(horizontal_rows[-1]) + 1

            expected = np.full(
                grid.shape,
                background,
                dtype=grid.dtype,
            )
            expected[:, v0:v1] = vertical_color
            expected[h0:h1, :] = horizontal_color
            expected[h0:h1, v0:v1] = vertical_color

            if np.array_equal(grid, expected):
                candidates.append(
                    {
                        "background": background,
                        "vertical_color": vertical_color,
                        "horizontal_color": horizontal_color,
                        "v0": v0,
                        "v1": v1,
                        "h0": h0,
                        "h1": h1,
                        "vertical_thickness": v1 - v0,
                        "horizontal_thickness": h1 - h0,
                        "top": "vertical",
                    }
                )

    # Candidate in which the horizontal band is on top.
    for horizontal_color in values:
        horizontal_rows = np.flatnonzero(
            np.all(grid == horizontal_color, axis=1)
        )
        if not _is_consecutive(horizontal_rows) or horizontal_rows.size > height - 3 or 2 * horizontal_rows.size >= height:
            continue

        outside_rows = np.ones(height, dtype=bool)
        outside_rows[horizontal_rows] = False

        for vertical_color in values:
            if vertical_color == horizontal_color:
                continue

            background = next(
                color
                for color in values
                if color not in (vertical_color, horizontal_color)
            )
            vertical_columns = np.flatnonzero(
                np.all(
                    grid[outside_rows, :] == vertical_color,
                    axis=0,
                )
            )
            if not _is_consecutive(vertical_columns) or vertical_columns.size > width - 3 or 2 * vertical_columns.size >= width:
                continue

            v0 = int(vertical_columns[0])
            v1 = int(vertical_columns[-1]) + 1
            h0 = int(horizontal_rows[0])
            h1 = int(horizontal_rows[-1]) + 1

            expected = np.full(
                grid.shape,
                background,
                dtype=grid.dtype,
            )
            expected[:, v0:v1] = vertical_color
            expected[h0:h1, :] = horizontal_color
            expected[h0:h1, v0:v1] = horizontal_color

            if np.array_equal(grid, expected):
                candidates.append(
                    {
                        "background": background,
                        "vertical_color": vertical_color,
                        "horizontal_color": horizontal_color,
                        "v0": v0,
                        "v1": v1,
                        "h0": h0,
                        "h1": h1,
                        "vertical_thickness": v1 - v0,
                        "horizontal_thickness": h1 - h0,
                        "top": "horizontal",
                    }
                )

    _require(
        len(candidates) == 1,
        "grid must have one unambiguous pair of crossing bands",
    )
    return candidates[0]


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    _require(
        input_grid.shape == output_grid.shape,
        "input and output dimensions must match",
    )

    input_info = _analyze_grid(input_grid)
    output_info = _analyze_grid(output_grid)

    structural_keys = (
        "background",
        "vertical_color",
        "horizontal_color",
        "v0",
        "v1",
        "h0",
        "h1",
        "vertical_thickness",
        "horizontal_thickness",
    )
    _require(
        all(
            input_info[key] == output_info[key]
            for key in structural_keys
        ),
        "output must preserve both bands, their roles, and positions",
    )
    _require(
        input_info["top"] != output_info["top"],
        "overlap order must reverse",
    )

    changed = input_grid != output_grid
    intersection = np.zeros(input_grid.shape, dtype=bool)
    intersection[
        input_info["h0"]:input_info["h1"],
        input_info["v0"]:input_info["v1"],
    ] = True

    _require(
        np.array_equal(changed, intersection),
        "exactly the intersection must change",
    )

    expected_color = (
        input_info["horizontal_color"]
        if input_info["top"] == "vertical"
        else input_info["vertical_color"]
    )
    _require(
        np.all(output_grid[intersection] == expected_color),
        "intersection must receive the formerly occluded band color",
    )

    return input_info


def _coerce_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not low <= result <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )
    return result


def _choose_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _coerce_count(
        num_train,
        "num_train",
        2,
        10,
    )
    test_count = _coerce_count(
        num_test,
        "num_test",
        1,
        4,
    )

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = (
            minimum_train
            + (seed // 4) % span
        )
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(
            rng.integers(minimum_train, 11)
        )
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError(
                "no legal num_test exists for this num_train"
            )
        test_count = int(
            rng.integers(1, maximum_test + 1)
        )

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count


def _random_palette(rng, background):
    colors = rng.choice([color for color in range(10) if color != background], size=2, replace=False)
    return (background, int(colors[0]), int(colors[1]))


def _sample_palettes(rng, train_count, test_count):
    first, second, third = (int(value) for value in rng.choice(10, size=3, replace=False))
    mode = int(rng.integers(0, 3))
    if mode == 0:
        initial = [(first, second, third), (first, second, third)]
        backgrounds = [first]
    elif mode == 1:
        available = [color for color in range(10) if color not in (first, second)]
        bands = rng.choice(available, size=2, replace=False)
        initial = [(first, int(bands[0]), int(bands[1])),
                   (second, int(bands[0]), int(bands[1]))]
        backgrounds = [first, second]
    else:
        initial = [(first, second, third), (second, first, third)]
        backgrounds = [first, second]

    train = initial[:]
    while len(train) < train_count:
        background = int(rng.choice(backgrounds))
        reusable = [palette for palette in train if palette[0] == background]
        if reusable and rng.random() < 0.35:
            train.append(reusable[int(rng.integers(len(reusable)))])
        else:
            train.append(_random_palette(rng, background))

    test = []
    for _ in range(test_count):
        background = int(rng.choice(backgrounds))
        reusable = [palette for palette in train if palette[0] == background]
        if rng.random() < 0.4:
            test.append(reusable[int(rng.integers(len(reusable)))])
        else:
            test.append(_random_palette(rng, background))
    return train, test


def _sample_thicknesses(rng, train_count, test_count):
    train = [(int(rng.integers(2, 5)), int(rng.integers(2, 5)))]
    second = ((1, 1) if rng.random() < 0.4 else
              (int(rng.integers(1, 5)), int(rng.integers(1, 5))))
    if second == train[0]:
        second = (1, 1)
    train.append(second)
    train.extend((int(rng.integers(1, 5)), int(rng.integers(1, 5)))
                 for _ in range(train_count - 2))
    test = [(int(rng.integers(1, 6)), int(rng.integers(1, 6)))
            for _ in range(test_count)]
    return train, test


def _sample_dimension(rng, regime, thickness):
    mode = int(rng.integers(0, 3))
    if regime == "source_train":
        low, high = ((6, 12), (10, 20), (16, 26))[mode]
    else:
        low, high = ((6, 8), (9, 19), (18, 29))[mode]
    return max(thickness + 3, 2 * thickness + 1, int(rng.integers(low, high)))


def _sample_start(rng, dimension, thickness):
    last = dimension - thickness
    if rng.random() < 0.13:
        return 0 if rng.random() < 0.5 else last
    return int(rng.integers(1, last))


def _generate_pair(rng, regime, palette, thickness, top):
    vertical_thickness, horizontal_thickness = thickness
    height = _sample_dimension(rng, regime, horizontal_thickness)
    width = _sample_dimension(rng, regime, vertical_thickness)
    v0 = _sample_start(rng, width, vertical_thickness)
    h0 = _sample_start(rng, height, horizontal_thickness)
    v1, h1 = v0 + vertical_thickness, h0 + horizontal_thickness
    background, vertical_color, horizontal_color = palette
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[:, v0:v1] = vertical_color
    input_grid[h0:h1, :] = horizontal_color
    if top == "vertical":
        input_grid[h0:h1, v0:v1] = vertical_color
    output_grid = input_grid.copy()
    output_grid[h0:h1, v0:v1] = horizontal_color if top == "vertical" else vertical_color
    return input_grid, output_grid


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(seed, rng, num_train, num_test)
    train_palettes, test_palettes = _sample_palettes(rng, train_count, test_count)
    train_thicknesses, test_thicknesses = _sample_thicknesses(rng, train_count, test_count)
    first_top = "vertical" if rng.random() < 0.5 else "horizontal"
    other_top = "horizontal" if first_top == "vertical" else "vertical"
    train_tops = [first_top, other_top] + [
        "vertical" if rng.random() < 0.5 else "horizontal"
        for _ in range(train_count - 2)
    ]
    test_tops = ["vertical" if rng.random() < 0.5 else "horizontal"
                 for _ in range(test_count)]
    if test_count > 1 and rng.random() < 0.7:
        test_tops[1] = "horizontal" if test_tops[0] == "vertical" else "vertical"
    train_specs = list(zip(train_palettes, train_thicknesses, train_tops))
    rng.shuffle(train_specs)

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for section, regime, specs in (
        ("train", "source_train", train_specs),
        ("test", "target_test", list(zip(test_palettes, test_thicknesses, test_tops))),
    ):
        for palette, thickness, top in specs:
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(20):
                input_grid, output_grid = _generate_pair(pair_rng, regime, palette, thickness, top)
                input_key = _grid_key(input_grid)
                if input_key not in seen_inputs and _pair_key(input_grid, output_grid) not in _SOURCE_PAIR_KEYS:
                    puzzle[section].append({"input": input_grid, "output": output_grid})
                    seen_inputs.add(input_key)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "sections must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D ndarray")
                _require(all(1 <= dimension <= 30 for dimension in grid.shape), "grid dimensions out of range")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "illegal ARC color")
            input_key = _grid_key(pair["input"])
            pair_key = _pair_key(pair["input"], pair["output"])
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            _require(pair_key not in _SOURCE_PAIR_KEYS, "source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            info = _analyze_pair(pair["input"], pair["output"])
            limit = 4 if section == "train" else 5
            _require(1 <= info["vertical_thickness"] <= limit
                     and 1 <= info["horizontal_thickness"] <= limit,
                     "band thickness outside regime")
            analyses[section].append(info)

    _require({info["top"] for info in analyses["train"]} == {"vertical", "horizontal"},
             "training must show both overlap orders")
    _require(any(info["vertical_thickness"] * info["horizontal_thickness"] >= 4
                 for info in analyses["train"]), "training must show a multi-cell crossing")
    train_backgrounds = {info["background"] for info in analyses["train"]}
    _require({info["background"] for info in analyses["test"]} <= train_backgrounds,
             "test background must be exposed in training")


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False

        for first_pair, second_pair in zip(
            first[section],
            second[section],
        ):
            if not np.array_equal(
                first_pair["input"],
                second_pair["input"],
            ):
                return False
            if not np.array_equal(
                first_pair["output"],
                second_pair["output"],
            ):
                return False

    return True


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            section,
            tuple(
                _pair_key(
                    pair["input"],
                    pair["output"],
                )
                for pair in puzzle[section]
            ),
        )
        for section in ("train", "test")
    )


def _predict_from_training(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    _require(len(train_pairs) >= 2, "prediction requires training examples")
    seen_tops = set()
    for pair in train_pairs:
        info = _analyze_grid(pair["input"])
        crossing = np.zeros(pair["input"].shape, dtype=bool)
        crossing[info["h0"]:info["h1"], info["v0"]:info["v1"]] = True
        other_color = (info["horizontal_color"] if info["top"] == "vertical"
                       else info["vertical_color"])
        expected = pair["input"].copy()
        expected[crossing] = other_color
        _require(np.array_equal(expected, pair["output"]), "training rule is inconsistent")
        seen_tops.add(info["top"])
    _require(seen_tops == {"vertical", "horizontal"}, "training lacks one direction")
    info = _analyze_grid(test_input)
    expected = test_input.copy()
    color = info["horizontal_color"] if info["top"] == "vertical" else info["vertical_color"]
    expected[info["h0"]:info["h1"], info["v0"]:info["v1"]] = color
    return expected


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42, "seed diversity collapsed")
    _require(len({len(puzzle["train"]) for puzzle in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(puzzle["test"]) for puzzle in puzzles}) >= 3, "test counts lack diversity")
    shapes, layouts, palettes, thicknesses = set(), set(), set(), set()
    train_thicknesses, test_thicknesses = set(), set()
    train_equal_large = test_thin = test_unequal = edge = 0
    variable_background = role_exchange = palette_reuse = both_test_orders = 0
    smaller = within = larger = 0
    train_2x2 = train_3x3 = train_1x1 = test_2x2 = test_width_6 = test_large = 0
    first_1x1 = first_multi = opposite_reused_palette = interior = one_cell_margin = 0
    test_tops = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_info = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["train"]]
        test_info = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["test"]]
        train_equal_large += any(info["vertical_thickness"] == info["horizontal_thickness"] >= 2
                                 for info in train_info)
        train_2x2 += any((info["vertical_thickness"], info["horizontal_thickness"]) == (2, 2)
                         for info in train_info)
        train_3x3 += any((info["vertical_thickness"], info["horizontal_thickness"]) == (3, 3)
                         for info in train_info)
        train_1x1 += any((info["vertical_thickness"], info["horizontal_thickness"]) == (1, 1)
                         for info in train_info)
        test_2x2 += any((info["vertical_thickness"], info["horizontal_thickness"]) == (2, 2)
                        for info in test_info)
        first_1x1 += (train_info[0]["vertical_thickness"], train_info[0]["horizontal_thickness"]) == (1, 1)
        first_multi += train_info[0]["vertical_thickness"] * train_info[0]["horizontal_thickness"] >= 4
        opposite_reused_palette += any(
            (a["vertical_color"], a["horizontal_color"]) == (b["vertical_color"], b["horizontal_color"])
            and a["top"] != b["top"] for a in train_info for b in train_info)
        test_thin += any(min(info["vertical_thickness"], info["horizontal_thickness"]) == 1
                         for info in test_info)
        test_unequal += any(info["vertical_thickness"] != info["horizontal_thickness"]
                            for info in test_info)
        variable_background += len({info["background"] for info in train_info}) > 1
        role_exchange += any(a["background"] in (b["vertical_color"], b["horizontal_color"])
                             for a in train_info for b in train_info if a is not b)
        palette_reuse += len({(info["vertical_color"], info["horizontal_color"])
                              for info in train_info}) < len(train_info)
        both_test_orders += len({info["top"] for info in test_info}) == 2
        test_tops.update(info["top"] for info in test_info)
        train_areas = [pair["input"].size for pair in puzzle["train"]]
        for pair, info in zip(puzzle["test"], test_info):
            prediction = _predict_from_training(
                [{"input": example["input"].copy(), "output": example["output"].copy()}
                 for example in puzzle["train"]], pair["input"].copy())
            _require(np.array_equal(prediction, pair["output"]), "test output not inferable")
            smaller += pair["input"].size < min(train_areas)
            within += min(train_areas) <= pair["input"].size <= max(train_areas)
            larger += pair["input"].size > max(train_areas)
            test_width_6 += pair["input"].shape[1] == 6
            test_large += max(pair["input"].shape) >= 27
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_pair(pair["input"], pair["output"])
                shape = pair["input"].shape
                shapes.add(shape)
                layouts.add((shape, info["v0"], info["v1"], info["h0"], info["h1"], info["top"]))
                palettes.add((info["background"], info["vertical_color"], info["horizontal_color"]))
                thicknesses.add((info["vertical_thickness"], info["horizontal_thickness"]))
                (train_thicknesses if section == "train" else test_thicknesses).add(
                    (info["vertical_thickness"], info["horizontal_thickness"]))
                edge += (info["v0"] == 0 or info["v1"] == shape[1]
                         or info["h0"] == 0 or info["h1"] == shape[0])
                interior += (0 < info["v0"] and info["v1"] < shape[1]
                             and 0 < info["h0"] and info["h1"] < shape[0])
                one_cell_margin += (info["v0"] == 1 or info["v1"] == shape[1] - 1
                                    or info["h0"] == 1 or info["h1"] == shape[0] - 1)
                crossing = np.zeros(shape, dtype=bool)
                crossing[info["h0"]:info["h1"], info["v0"]:info["v1"]] = True
                _require(np.array_equal(pair["input"] != pair["output"], crossing), "wrong changed cells")
    _require(len(shapes) >= 60 and len(layouts) >= 150 and len(palettes) >= 75,
             "structural diversity collapsed")
    _require(len(thicknesses) >= 15, "band thickness diversity collapsed")
    _require(len(train_thicknesses) >= 10 and len(test_thicknesses) >= 12,
             "regime thickness combinations collapsed")
    _require(all((train_equal_large, test_thin, test_unequal, edge, variable_background,
                  role_exchange, palette_reuse, both_test_orders, smaller, within, larger)),
             "reviewed variation axis missing")
    _require(all((train_2x2, train_3x3, train_1x1, test_2x2, test_width_6, test_large,
                  first_1x1, first_multi, opposite_reused_palette, interior, one_cell_margin)),
             "specific reviewed regime missing")
    _require(variable_background < 42 and edge < edge + interior,
             "stable-background or interior cases missing")
    _require(test_tops == {"vertical", "horizontal"}, "test direction fixed")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        seed = 9000 + train_count * 10 + test_count
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count),
                 "explicit counts ignored")
        _require(_puzzles_equal(puzzle, generate_puzzle(seed, train_count, test_count)),
                 "nondeterministic puzzle")
    for index, arguments in enumerate(({"num_test": 1}, {"num_test": 4},
                                       {"num_train": 2}, {"num_train": 10})):
        puzzle = generate_puzzle(12000 + index, **arguments)
        validate_generated_puzzle(puzzle)
        for name, value in arguments.items():
            section = "train" if name == "num_train" else "test"
            _require(len(puzzle[section]) == value, "partial count ignored")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(77, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
