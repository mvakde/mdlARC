import numpy as np


_QUADRANTS = ("ul", "ur", "ll", "lr")
_OFFSETS = {
    "ul": (0, 0),
    "ur": (0, 1),
    "ll": (1, 0),
    "lr": (1, 1),
}

_SOURCE_INPUT_ROWS = (
    (
        "0050000000",
        "0000000090",
        "0000000000",
        "0000000000",
        "0000880000",
        "0000880000",
        "0000000001",
        "0000000000",
        "0200000000",
        "0000000000",
    ),
    (
        "0000000070",
        "0000000000",
        "0000000000",
        "0600000000",
        "0000880000",
        "0000880000",
        "0000000000",
        "0000000000",
        "0400000000",
        "0000000009",
    ),
    (
        "0001000004",
        "0000000000",
        "0000088000",
        "0000088000",
        "0000000000",
        "0000000000",
        "0000000006",
        "0030000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000000000",
        "0000004000",
        "0000000000",
        "0300000000",
        "0000880000",
        "0000880000",
        "0000000000",
        "0000000000",
        "0000000070",
        "0060000000",
    ),
)


def _rows_key(rows):
    return (
        len(rows),
        len(rows[0]),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_INPUT_KEYS = frozenset(
    _rows_key(rows) for rows in _SOURCE_INPUT_ROWS
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid_key(grid):
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _checked_count(name, value, lower, upper):
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
        )

    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _checked_count(
        "num_train", num_train, 2, 10
    )
    num_test = _checked_count(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees substantial count
        # diversity for consecutive fixed seeds.
        num_test = 1 + seed % 4
        minimum_train = num_test + 1
        span = 11 - minimum_train
        num_train = (
            minimum_train + (seed // 4) % span
        )

    elif num_train is None:
        minimum_train = num_test + 1
        num_train = int(
            rng.integers(minimum_train, 11)
        )

    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = int(
            rng.integers(1, maximum_test + 1)
        )

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _sample_puzzle_parameters(rng):
    roles = rng.choice(10, size=2, replace=False)
    background, marker = (int(value) for value in roles)
    palette_mode = "reuse" if rng.random() < 0.45 else "change"
    role_mode = "variable" if rng.random() < 0.45 else "stable"
    shared_palette = None
    if palette_mode == "reuse":
        shared_palette = tuple(
            int(value) for value in rng.choice(
                [color for color in range(10)
                 if color not in (background, marker)],
                size=4,
                replace=False,
            )
        )
    return {
        "roles": (background, marker),
        "role_mode": role_mode,
        "palette_mode": palette_mode,
        "shared_palette": shared_palette,
    }


def _layout_mode(analysis):
    height, width = analysis["shape"]
    top, left = analysis["marker_top"]

    ideal_top = (height - 2) // 2
    ideal_left = (width - 2) // 2

    if (
        abs(top - ideal_top) <= 1
        and abs(left - ideal_left) <= 1
    ):
        return "central"

    if (
        top <= ideal_top - 2
        and left >= ideal_left + 1
    ):
        return "shifted_upper_right"

    return "other"


def _analyze_pair(input_grid, output_grid):
    _require(
        isinstance(input_grid, np.ndarray),
        "input must be a NumPy array",
    )
    _require(
        isinstance(output_grid, np.ndarray),
        "output must be a NumPy array",
    )
    _require(
        input_grid.ndim == 2 and output_grid.ndim == 2,
        "grids must be two-dimensional",
    )
    _require(
        input_grid.shape == output_grid.shape,
        "input and output dimensions must match",
    )
    _require(
        np.issubdtype(input_grid.dtype, np.integer),
        "input dtype must be integer",
    )
    _require(
        np.issubdtype(output_grid.dtype, np.integer),
        "output dtype must be integer",
    )
    _require(
        input_grid.size > 0 and output_grid.size > 0,
        "grids must be nonempty",
    )
    _require(
        np.all((0 <= input_grid) & (input_grid <= 9)),
        "input colors must be in 0..9",
    )
    _require(
        np.all((0 <= output_grid) & (output_grid <= 9)),
        "output colors must be in 0..9",
    )

    counts = np.bincount(
        input_grid.astype(
            np.int64, copy=False
        ).ravel(),
        minlength=10,
    )
    background = int(np.argmax(counts))

    _require(
        int(counts[background]) == input_grid.size - 8,
        "input must have four marker cells and four tokens",
    )

    nonbackground = [
        color
        for color in range(10)
        if color != background and counts[color] > 0
    ]
    _require(
        len(nonbackground) == 5,
        "input must have one marker color and four token colors",
    )

    marker_candidates = []

    for color in nonbackground:
        if counts[color] != 4:
            continue

        coordinates = np.argwhere(
            input_grid == color
        )
        rows = np.unique(coordinates[:, 0])
        columns = np.unique(coordinates[:, 1])

        if (
            len(rows) == 2
            and len(columns) == 2
            and rows[1] == rows[0] + 1
            and columns[1] == columns[0] + 1
        ):
            expected_coordinates = {
                (int(row), int(column))
                for row in rows
                for column in columns
            }
            actual_coordinates = {
                (int(row), int(column))
                for row, column in coordinates
            }

            if actual_coordinates == expected_coordinates:
                marker_candidates.append(
                    (
                        int(color),
                        int(rows[0]),
                        int(columns[0]),
                    )
                )

    _require(
        len(marker_candidates) == 1,
        "input must contain one solid 2-by-2 marker",
    )

    marker, top, left = marker_candidates[0]

    payload_by_quadrant = {}
    positions_by_quadrant = {}

    for color in nonbackground:
        if color == marker:
            continue

        _require(
            int(counts[color]) == 1,
            "each payload color must occur exactly once",
        )

        row, column = (
            int(value)
            for value in np.argwhere(
                input_grid == color
            )[0]
        )

        if row < top:
            vertical = "u"
        elif row > top + 1:
            vertical = "l"
        else:
            raise AssertionError(
                "a token lies in the marker's row span"
            )

        if column < left:
            horizontal = "l"
        elif column > left + 1:
            horizontal = "r"
        else:
            raise AssertionError(
                "a token lies in the marker's column span"
            )

        quadrant = vertical + horizontal

        _require(
            quadrant not in payload_by_quadrant,
            "two tokens occupy the same diagonal region",
        )

        payload_by_quadrant[quadrant] = int(color)
        positions_by_quadrant[quadrant] = (
            row,
            column,
        )

    _require(
        set(payload_by_quadrant) == set(_QUADRANTS),
        "all four diagonal regions must be represented",
    )

    expected_output = np.full(
        input_grid.shape,
        background,
        dtype=np.int8,
    )

    for quadrant in _QUADRANTS:
        row_offset, column_offset = _OFFSETS[
            quadrant
        ]
        expected_output[
            top + row_offset,
            left + column_offset,
        ] = payload_by_quadrant[quadrant]

    _require(
        np.array_equal(output_grid, expected_output),
        "output is not the quadrant-ordered replacement block",
    )
    _require(
        not np.array_equal(input_grid, output_grid),
        "input and output must differ",
    )

    color_order = tuple(
        payload_by_quadrant[quadrant]
        for quadrant in _QUADRANTS
    )
    position_order = tuple(
        positions_by_quadrant[quadrant]
        for quadrant in _QUADRANTS
    )
    shape = tuple(
        int(value) for value in input_grid.shape
    )

    return {
        "shape": shape,
        "background": background,
        "marker": marker,
        "marker_top": (top, left),
        "color_order": color_order,
        "palette": tuple(sorted(color_order)),
        "position_order": position_order,
        "layout_signature": (
            shape,
            (top, left),
            position_order,
        ),
    }


def _sample_shape(rng, training_anchor=False):
    # The object vocabulary stays fixed while the available empty space changes.
    kind = str(rng.choice(
        ("ordinary", "compact", "tall", "wide", "large"),
        p=(0.38, 0.18, 0.14, 0.14, 0.16),
    ))
    if kind == "ordinary":
        height = int(rng.integers(8, 17))
        width = int(rng.integers(8, 17))
    elif kind == "compact":
        height = int(rng.integers(4, 8))
        width = int(rng.integers(4, 8))
    elif kind == "tall":
        height = int(rng.integers(16, 31))
        width = int(rng.integers(4, 9))
    elif kind == "wide":
        height = int(rng.integers(4, 9))
        width = int(rng.integers(16, 31))
    else:
        height = int(rng.integers(20, 31))
        width = int(rng.integers(20, 31))

    # The second training example always has room for a visibly moved marker.
    if training_anchor:
        height = max(height, 9)
        width = max(width, 9)
    return height, width


def _sample_marker(rng, height, width, distance_style, force_noncentral):
    if distance_style in ("all_far", "one_near"):
        row_bounds = range(2, height - 3)
        column_bounds = range(2, width - 3)
    else:
        row_bounds = range(1, height - 2)
        column_bounds = range(1, width - 2)

    positions = [(row, column) for row in row_bounds
                 for column in column_bounds]
    center_row = (height - 2) // 2
    center_column = (width - 2) // 2
    if force_noncentral:
        positions = [position for position in positions
                     if abs(position[0] - center_row) > 1
                     or abs(position[1] - center_column) > 1]

    mode = str(rng.choice(
        ("central", "upper_left", "upper_right", "lower_left",
         "lower_right", "near_edge", "free"),
        p=(0.23, 0.11, 0.11, 0.11, 0.11, 0.14, 0.19),
    ))
    if mode == "central":
        preferred = [position for position in positions
                     if abs(position[0] - center_row) <= 1
                     and abs(position[1] - center_column) <= 1]
    elif mode == "near_edge":
        preferred = [position for position in positions
                     if position[0] in (1, height - 3)
                     or position[1] in (1, width - 3)]
    elif mode == "free":
        preferred = positions
    else:
        upper = mode.startswith("upper")
        left = mode.endswith("left")
        preferred = [position for position in positions
                     if (position[0] < center_row if upper
                         else position[0] > center_row)
                     and (position[1] < center_column if left
                          else position[1] > center_column)]
    choices = preferred or positions
    if not choices:
        raise ValueError("no legal marker placement")
    return choices[int(rng.integers(len(choices)))]


def _sample_distance(rng, maximum, near=None):
    if maximum == 1 or near is True:
        return 1
    if near is False:
        return int(rng.integers(2, maximum + 1))
    if rng.random() < 0.35:
        return 1
    return int(rng.integers(2, maximum + 1))


def _sample_positions(rng, shape, marker_top, style):
    height, width = shape
    top, left = marker_top
    chosen = int(rng.integers(4))
    near_set = set()
    if style == "several_near":
        number = int(rng.integers(2, 5))
        near_set = set(int(value) for value in rng.choice(
            4, size=number, replace=False))
    positions = []
    for index, quadrant in enumerate(_QUADRANTS):
        vertical_max = top if quadrant[0] == "u" else height - top - 2
        horizontal_max = left if quadrant[1] == "l" else width - left - 2
        if style == "all_far":
            vertical_near = horizontal_near = False
        elif style == "one_near":
            axis = int(rng.integers(3)) if index == chosen else -1
            vertical_near = axis in (0, 2)
            horizontal_near = axis in (1, 2)
        elif index in near_set:
            axis = int(rng.integers(3))
            vertical_near = True if axis in (0, 2) else None
            horizontal_near = True if axis in (1, 2) else None
        else:
            vertical_near = horizontal_near = None
        vertical = _sample_distance(rng, vertical_max, vertical_near)
        horizontal = _sample_distance(rng, horizontal_max, horizontal_near)
        row = top - vertical if quadrant[0] == "u" else top + 1 + vertical
        column = left - horizontal if quadrant[1] == "l" else left + 1 + horizontal
        positions.append((row, column))
    return tuple(positions)


def _generate_pair(rng, roles, palette, used_layouts, used_inputs,
                   first_train_order=None, training_anchor=False):
    background, marker = roles
    for _ in range(100):
        color_order = tuple(int(value) for value in rng.permutation(palette))
        if first_train_order is not None and any(
            current == previous for current, previous in
            zip(color_order, first_train_order)
        ):
            continue
        height, width = _sample_shape(rng, training_anchor)
        styles = ("all_far", "one_near", "several_near", "free")
        if min(height, width) >= 6:
            style = str(rng.choice(styles, p=(0.20, 0.18, 0.30, 0.32)))
        else:
            style = str(rng.choice(("several_near", "free")))
        top, left = _sample_marker(
            rng, height, width, style, training_anchor)
        positions = _sample_positions(rng, (height, width), (top, left), style)
        layout = ((height, width), (top, left), positions)
        if layout in used_layouts:
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        input_grid[top:top + 2, left:left + 2] = marker
        for color, (row, column) in zip(color_order, positions):
            input_grid[row, column] = color
        key = _grid_key(input_grid)
        if key in used_inputs or key in _SOURCE_INPUT_KEYS:
            continue
        output_grid = np.full((height, width), background, dtype=np.int8)
        for quadrant, color in zip(_QUADRANTS, color_order):
            row_offset, column_offset = _OFFSETS[quadrant]
            output_grid[top + row_offset, left + column_offset] = color
        used_layouts.add(layout)
        used_inputs.add(key)
        return input_grid, output_grid, color_order
    raise ValueError("unable to construct a distinct pair")


def _select_roles(rng, parameters, first_roles=None):
    if parameters["role_mode"] == "stable":
        return parameters["roles"]
    excluded = set(parameters["shared_palette"] or ())
    available = [color for color in range(10) if color not in excluded]
    choices = [(background, marker) for background in available
               for marker in available if background != marker
               and (first_roles is None or (
                   background != first_roles[0]
                   and marker != first_roles[1]))]
    return choices[int(rng.integers(len(choices)))]


def _select_palette(rng, parameters, roles, first_palette=None):
    if parameters["palette_mode"] == "reuse":
        return parameters["shared_palette"]
    available = [color for color in range(10) if color not in roles]
    for _ in range(50):
        palette = tuple(int(value) for value in rng.choice(
            available, size=4, replace=False))
        if first_palette is None or set(palette) != set(first_palette):
            return palette
    raise ValueError("unable to select a different payload palette")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    child_seeds = rng.integers(
        0, np.iinfo(np.uint64).max, size=train_count + test_count,
        dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    used_layouts = set()
    used_inputs = set()
    first_roles = first_palette = first_order = None
    for index in range(train_count + test_count):
        pair_rng = np.random.default_rng(child_seeds[index])
        second_train = index == 1
        roles = _select_roles(
            pair_rng, parameters,
            first_roles if second_train and parameters["role_mode"] == "variable"
            else None)
        palette = _select_palette(
            pair_rng, parameters, roles,
            first_palette if second_train and
            parameters["palette_mode"] == "change" else None)
        input_grid, output_grid, color_order = _generate_pair(
            pair_rng, roles, palette, used_layouts, used_inputs,
            first_order if second_train else None,
            training_anchor=second_train)
        if index == 0:
            first_roles = roles
            first_palette = palette
            first_order = color_order
        split = "train" if index < train_count else "test"
        puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and
             set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and
             isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "invalid pair counts")
    analyses = {"train": [], "test": []}
    input_keys, pair_keys, layouts = set(), set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and
                     set(pair) == {"input", "output"}, "invalid pair schema")
            input_grid, output_grid = pair["input"], pair["output"]
            analysis = _analyze_pair(input_grid, output_grid)
            height, width = analysis["shape"]
            _require(1 <= height <= 30 and 1 <= width <= 30,
                     "grid dimensions out of ARC bounds")
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            _require(input_key not in input_keys, "duplicate input")
            _require(pair_key not in pair_keys, "duplicate pair")
            _require(input_key not in _SOURCE_INPUT_KEYS,
                     "source pair was copied")
            layout = analysis["layout_signature"]
            _require(layout not in layouts, "duplicate geometric layout")
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            layouts.add(layout)
            analyses[split].append(analysis)
    train = analyses["train"]
    test = analyses["test"]
    _require(any(_layout_mode(item) != "central" for item in train),
             "training does not demonstrate a moved marker")
    for role_index in range(4):
        _require(len({item["color_order"][role_index]
                      for item in train}) >= 2,
                 "training does not demonstrate color-independent placement")
    first_roles = (train[0]["background"], train[0]["marker"])
    if any(item["background"] != first_roles[0] for item in test):
        _require(len({item["background"] for item in train}) >= 2,
                 "test background changes are absent from training")
    if any(item["marker"] != first_roles[1] for item in test):
        _require(len({item["marker"] for item in train}) >= 2,
                 "test marker changes are absent from training")
    if any(item["palette"] != train[0]["palette"] for item in test):
        _require(len({item["palette"] for item in train}) >= 2,
                 "test palette changes are absent from training")


def _puzzles_equal(first, second):
    if set(first) != set(second):
        return False
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            if not all(np.array_equal(pair_a[name], pair_b[name])
                       for name in ("input", "output")):
                return False
    return True


def _pair_features(analysis):
    height, width = analysis["shape"]
    top, left = analysis["marker_top"]
    near_count = 0
    far_count = 0
    for quadrant, (row, column) in zip(
        _QUADRANTS, analysis["position_order"]
    ):
        vertical = top - row if quadrant[0] == "u" else row - top - 1
        horizontal = left - column if quadrant[1] == "l" else column - left - 1
        near_count += int(vertical == 1 or horizontal == 1)
        far_count += int(vertical > 1 and horizontal > 1)
    return near_count, far_count, height, width


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(puzzle["train"]) for puzzle in puzzles}
    test_counts = {len(puzzle["test"]) for puzzle in puzzles}
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    shapes, layouts, orders, roles, keys = set(), set(), set(), set(), set()
    compact = stretched = large = four_by_four = 0
    noncentral_tests = central_tests = near_edge_tests = all_far_tests = 0
    close_tests = one_close_tests = 0
    marker_quadrants = set()
    close_quadrants = set()
    reuse_puzzles = changed_palette_puzzles = 0
    variable_role_puzzles = stable_role_puzzles = 0
    variable_role_reuse_puzzles = 0
    noncentral_train_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_analyze_pair(pair["input"], pair["output"])
                 for pair in puzzle["train"]]
        test = [_analyze_pair(pair["input"], pair["output"])
                for pair in puzzle["test"]]
        train_palettes = {item["palette"] for item in train}
        all_palettes = {item["palette"] for item in train + test}
        all_roles = {(item["background"], item["marker"])
                     for item in train + test}
        reuse_puzzles += int(len(all_palettes) == 1)
        changed_palette_puzzles += int(len(train_palettes) > 1)
        variable_role_puzzles += int(len({(item["background"], item["marker"])
                                          for item in train}) > 1)
        stable_role_puzzles += int(len(all_roles) == 1)
        variable_role_reuse_puzzles += int(
            len(all_roles) > 1 and len(all_palettes) == 1)
        noncentral_train_puzzles += int(any(
            _layout_mode(item) != "central" for item in train))
        _require(len({item["shape"] for item in train + test}) >= 2,
                 "a puzzle does not vary pair dimensions")
        for split, group in (("train", train), ("test", test)):
            for pair, analysis in zip(puzzle[split], group):
                near, far, height, width = _pair_features(analysis)
                compact += int(height <= 7 and width <= 7)
                four_by_four += int(height == 4 and width == 4)
                stretched += int(max(height, width) >= 20
                                 and min(height, width) <= 8)
                large += int(height >= 20 and width >= 20)
                if split == "test":
                    top, left = analysis["marker_top"]
                    center_row = (height - 2) // 2
                    center_column = (width - 2) // 2
                    central = _layout_mode(analysis) == "central"
                    central_tests += int(central)
                    noncentral_tests += int(not central)
                    near_edge_tests += int(
                        top in (1, height - 3)
                        or left in (1, width - 3))
                    if not central:
                        marker_quadrants.add((
                            "u" if top < center_row else "l",
                            "l" if left < center_column else "r",
                        ))
                    all_far_tests += int(far == 4)
                    close_tests += int(near >= 2)
                    one_close_tests += int(near == 1)
                    for quadrant, (row, column) in zip(
                        _QUADRANTS, analysis["position_order"]
                    ):
                        vertical = top - row if quadrant[0] == "u" else row - top - 1
                        horizontal = left - column if quadrant[1] == "l" else column - left - 1
                        if vertical == 1 or horizontal == 1:
                            close_quadrants.add(quadrant)
                shapes.add(analysis["shape"])
                layouts.add(analysis["layout_signature"])
                orders.add(analysis["color_order"])
                roles.add((analysis["background"], analysis["marker"]))
                keys.add(_grid_key(pair["input"]))
    _require(len(shapes) >= 55 and len(layouts) >= 150
             and len(orders) >= 100 and len(roles) >= 15
             and len(keys) >= 150, "insufficient cross-seed diversity")
    _require(compact >= 8 and stretched >= 8 and large >= 8
             and four_by_four >= 1,
             "grid scale regimes lack coverage")
    _require(noncentral_tests >= 10 and central_tests >= 10
             and near_edge_tests >= 5 and len(marker_quadrants) == 4,
             "test marker locations lack coverage")
    _require(all_far_tests >= 5 and close_tests >= 10
             and one_close_tests >= 5 and len(close_quadrants) == 4,
             "test token distances lack coverage")
    _require(reuse_puzzles >= 5 and changed_palette_puzzles >= 5
             and variable_role_puzzles >= 5 and stable_role_puzzles >= 5
             and variable_role_reuse_puzzles >= 3,
             "color-role regimes lack coverage")
    _require(noncentral_train_puzzles == 42,
             "training marker coverage is incomplete")
    for index, (train_count, test_count) in enumerate(
        ((2, 1), (10, 1), (10, 4), (7, 3))
    ):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        _require(len(puzzle["train"]) == train_count and
                 len(puzzle["test"]) == test_count,
                 "explicit counts were not honored")
    for seed, train_count, test_count in (
        (2000, None, 1), (2001, None, 4),
        (2002, 2, None), (2003, 10, None)
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require((train_count is None or len(puzzle["train"]) == train_count)
                 and (test_count is None or len(puzzle["test"]) == test_count),
                 "partial count was not honored")
    _require(_puzzles_equal(generate_puzzle(314159, 8, 3),
                            generate_puzzle(314159, 8, 3)),
             "generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(2718),
                                generate_puzzle(2719)),
             "different seeds generated the same puzzle")
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}
    ):
        try:
            generate_puzzle(99, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts did not raise ValueError")
