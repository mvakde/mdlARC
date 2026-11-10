import numpy as np


_TRAIN_SHAPES = ((5, 3), (4, 4), (3, 5))


def _grid_key(grid):
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _grid_from_rows(rows):
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _make_source_pair_keys():
    raw_pairs = (
        (
            (
                "000000000000",
                "001000000000",
                "001100000000",
                "000100000000",
                "001110000000",
                "000010000000",
                "000000000000",
                "000000000000",
                "000000000000",
                "000000000000",
                "000000000000",
            ),
            ("100", "110", "010", "111", "001"),
        ),
        (
            (
                "000000000000",
                "000000000000",
                "000022200000",
                "000002000000",
                "000222000000",
                "000202000000",
                "000000000000",
                "000000000000",
                "000000000000",
                "000000000000",
            ),
            ("0222", "0020", "2220", "2020"),
        ),
        (
            (
                "000000000000",
                "000000000000",
                "000000000000",
                "000080800000",
                "000888800000",
                "000000880000",
                "000000000000",
                "000000000000",
                "000000000000",
                "000000000000",
                "000000000000",
                "000000000000",
            ),
            ("08080", "88880", "00088"),
        ),
        (
            (
                "000000000000",
                "000000000000",
                "000000000000",
                "000000000000",
                "000066660000",
                "000060000000",
                "006060000000",
                "006666000000",
                "000000000000",
                "000000000000",
                "000000000000",
                "000000000000",
            ),
            ("006666", "006000", "606000", "666600"),
        ),
    )

    return frozenset(
        (
            _grid_key(_grid_from_rows(input_rows)),
            _grid_key(_grid_from_rows(output_rows)),
        )
        for input_rows, output_rows in raw_pairs
    )


_SOURCE_PAIR_KEYS = _make_source_pair_keys()


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _normalized_optional_count(name, value, minimum, maximum):
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(seed, num_train, num_test):
    train_count = _normalized_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _normalized_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        train_count = (
            minimum_train
            + (seed // 4) % (11 - minimum_train)
        )

    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = (
            minimum_train
            + seed % (11 - minimum_train)
        )

    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed % maximum_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count



def sample_puzzle_parameters(rng):
    # The crop operation is the sole puzzle-wide convention. This profile only
    # changes the frequency of locally sampled sizes and shape complexity.
    return {"profile": int(rng.integers(0, 3))}


def _is_connected(mask):
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    height, width = mask.shape
    start = tuple(int(v) for v in points[0])
    visited = {start}
    stack = [start]
    while stack:
        row, column = stack.pop()
        for neighbor in ((row - 1, column), (row + 1, column),
                         (row, column - 1), (row, column + 1)):
            r, c = neighbor
            if (0 <= r < height and 0 <= c < width and mask[r, c]
                    and neighbor not in visited):
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == int(mask.sum())


def _bounds(mask):
    points = np.argwhere(mask)
    if not len(points):
        return None
    low = points.min(axis=0)
    high = points.max(axis=0)
    return int(low[0]), int(high[0]) + 1, int(low[1]), int(high[1]) + 1


def _role_candidates(grid):
    colors = np.unique(grid)
    if len(colors) != 2:
        return []
    candidates = []
    for foreground in colors:
        mask = grid == foreground
        if not _is_connected(mask):
            continue
        top, bottom, left, right = _bounds(mask)
        if (top, bottom, left, right) == (0, grid.shape[0], 0, grid.shape[1]):
            continue
        candidates.append((int(foreground), int(colors[colors != foreground][0]),
                           (top, bottom, left, right)))
    return candidates


def _boundary_gaps(mask):
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True
    return int(np.count_nonzero(boundary & ~mask))


def _enclosed_holes(mask):
    background = ~mask
    height, width = mask.shape
    visited = set()
    holes = 0
    for row, column in np.argwhere(background):
        start = int(row), int(column)
        if start in visited:
            continue
        visited.add(start)
        stack = [start]
        touches_edge = False
        while stack:
            r, c = stack.pop()
            touches_edge |= r in (0, height - 1) or c in (0, width - 1)
            for rr, cc in ((r - 1, c), (r + 1, c),
                           (r, c - 1), (r, c + 1)):
                neighbor = rr, cc
                if (0 <= rr < height and 0 <= cc < width
                        and background[neighbor] and neighbor not in visited):
                    visited.add(neighbor)
                    stack.append(neighbor)
        holes += not touches_edge
    return holes


def _dimension(rng, profile, regime):
    if profile == 0:
        # A genuinely modest puzzle profile remains in the mixture.
        return int(rng.integers(2, 9 if regime == "source_train" else 11))
    if profile == 1:
        weights = (0.36, 0.44, 0.20)
    else:
        weights = (0.23, 0.39, 0.38)
    band = int(rng.choice(3, p=weights))
    low, high = ((1, 5), (6, 14), (15, 27))[band]
    return int(rng.integers(low, high + 1))


def _backbone(rng, height, width):
    mask = np.zeros((height, width), dtype=bool)
    steps = [0] * (height - 1) + [1] * (width - 1)
    rng.shuffle(steps)
    row = column = 0
    mask[0, 0] = True
    for step in steps:
        if step == 0:
            row += 1
        else:
            column += 1
        mask[row, column] = True
    if int(rng.integers(2)):
        mask = mask[::-1, :]
    if int(rng.integers(2)):
        mask = mask[:, ::-1]
    return mask.copy()


def _grow(rng, mask, target):
    height, width = mask.shape
    frontier = set()
    def add_neighbors(row, column):
        for r, c in ((row - 1, column), (row + 1, column),
                     (row, column - 1), (row, column + 1)):
            if 0 <= r < height and 0 <= c < width and not mask[r, c]:
                frontier.add((r, c))
    for row, column in np.argwhere(mask):
        add_neighbors(int(row), int(column))
    while int(mask.sum()) < target and frontier:
        choices = sorted(frontier)
        selected = choices[int(rng.integers(len(choices)))]
        frontier.remove(selected)
        mask[selected] = True
        add_neighbors(*selected)
    return mask


def _sample_mask(rng, height, width, mode):
    if mode == "solid":
        return np.ones((height, width), dtype=bool)
    if mode == "holes" and height >= 4 and width >= 4:
        mask = np.ones((height, width), dtype=bool)
        sites = [(r, c) for r in range(1, height - 1, 2)
                 for c in range(1, width - 1, 2)]
        rng.shuffle(sites)
        count = min(len(sites), max(2, int(round(len(sites) *
                                                  float(rng.uniform(.3, .9))))))
        for cell in sites[:count]:
            mask[cell] = False
        return mask
    if mode == "ring" and height >= 3 and width >= 3:
        mask = np.zeros((height, width), dtype=bool)
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        if float(rng.random()) < .55:
            target = int(round(height * width * float(rng.uniform(.35, .65))))
            _grow(rng, mask, target)
        return mask
    mask = _backbone(rng, height, width)
    if mode in ("branch", "dense"):
        density = (float(rng.uniform(.32, .68)) if mode == "branch"
                   else float(rng.uniform(.72, .96)))
        target = max(int(mask.sum()), int(round(height * width * density)))
        _grow(rng, mask, target)
    return mask


def _axis_padding(rng, object_size):
    available = 30 - object_size
    kind = int(rng.choice(4, p=(.18, .32, .34, .16)))
    if kind == 0:
        extra = 0
    elif kind == 1:
        extra = int(rng.integers(1, min(3, available) + 1))
    elif kind == 2:
        extra = int(rng.integers(1, min(12, available) + 1))
    else:
        extra = int(rng.integers(max(1, available - 3), available + 1))
    first = int(rng.integers(extra + 1))
    return first, extra - first


def _sample_pair(rng, regime, parameters, pair_index, train_count,
                 previous_pairs):
    for _ in range(350):
        profile = parameters["profile"]
        framed_dense = False
        if regime == "source_train" and pair_index == 0:
            height = int(rng.integers(4, 10))
            width = int(rng.integers(4, 10))
            mode = "branch" if float(rng.random()) < .5 else "path"
        elif regime == "source_train" and train_count == 2 and pair_index == 1:
            height = _dimension(rng, profile, regime)
            width = _dimension(rng, profile, regime)
            mode = "branch" if float(rng.random()) < .5 else "path"
        else:
            framed_dense = profile != 0 and float(rng.random()) < .075
            if framed_dense:
                height = int(rng.integers(12, 29))
                width = int(rng.integers(12, 29))
                mode = "dense"
            else:
                height = _dimension(rng, profile, regime)
                width = _dimension(rng, profile, regime)
                if profile == 0:
                    mode = str(rng.choice(
                        ("path", "branch", "solid", "horizontal", "vertical",
                         "singleton", "ring"),
                        p=(.30, .30, .16, .07, .07, .04, .06),
                    ))
                else:
                    mode = str(rng.choice(
                        ("path", "branch", "dense", "holes", "ring", "solid",
                         "horizontal", "vertical", "singleton"),
                        p=(.17, .23, .15, .13, .10, .10, .045, .045, .03),
                    ))
        if mode == "horizontal":
            height = 1
        elif mode == "vertical":
            width = 1
        elif mode == "singleton":
            height = width = 1
        if regime == "source_train" and pair_index == 1:
            first_shape = previous_pairs[0]["output"].shape
            if height == first_shape[0] or width == first_shape[1]:
                continue
        mask = _sample_mask(rng, height, width, mode)
        if regime == "source_train" and (pair_index == 0 or train_count == 2):
            if int(np.count_nonzero(~mask)) < 2 or _boundary_gaps(mask) < 2:
                continue
        if framed_dense:
            top = bottom = left = right = 1
        else:
            top, bottom = _axis_padding(rng, height)
            left, right = _axis_padding(rng, width)
        if top + bottom + left + right == 0:
            bottom = 1
        background = int(rng.integers(10))
        if regime == "source_train" and pair_index == 1:
            prior_background = _role_candidates(previous_pairs[0]["input"])[0][1]
            if background == prior_background:
                background = (background + int(rng.integers(1, 10))) % 10
        prior_foregrounds = [
            _role_candidates(pair["input"])[0][0] for pair in previous_pairs
        ]
        reused = [color for color in prior_foregrounds if color != background]
        if reused and float(rng.random()) < .32:
            foreground = int(reused[int(rng.integers(len(reused)))])
        else:
            foreground = int(rng.integers(9))
            if foreground >= background:
                foreground += 1
        grid = np.full((top + height + bottom, left + width + right),
                       background, dtype=np.int8)
        view = grid[top:top + height, left:left + width]
        view[mask] = foreground
        candidates = _role_candidates(grid)
        if len(candidates) != 1 or candidates[0][0] != foreground:
            continue
        bounds = candidates[0][2]
        output = grid[bounds[0]:bounds[1], bounds[2]:bounds[3]].copy()
        if output.shape != (height, width):
            continue
        return {"input": grid, "output": output}
    raise RuntimeError("could not construct a clear, unique pair")


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} is not a NumPy array")
    _require(grid.ndim == 2, f"{label} is not two-dimensional")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
             f"{label} dimensions are outside 1..30")
    _require(np.issubdtype(grid.dtype, np.integer),
             f"{label} does not have an integer dtype")
    _require(bool(np.all((grid >= 0) & (grid <= 9))),
             f"{label} has a color outside 0..9")


def _validate_pair_transformation(pair, regime):
    grid, output = pair["input"], pair["output"]
    candidates = _role_candidates(grid)
    _require(len(candidates) == 1,
             "foreground and background roles are not uniquely visible")
    foreground, background, bounds = candidates[0]
    top, bottom, left, right = bounds
    _require(np.array_equal(output, grid[top:bottom, left:right]),
             "output is not the exact tight crop")
    _require(set(int(x) for x in np.unique(output)) <= {foreground, background},
             "output changed a color")
    _require(not np.array_equal(grid, output), "input/output pair is degenerate")
    _require(regime in ("source_train", "target_test"), "unknown pair regime")
    return foreground, background, bounds


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list)
             and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "illegal pair counts")
    inputs, pairs = set(), set()
    train_heights, train_widths, train_backgrounds = set(), set(), set()
    retained_background = False
    for split, regime in (("train", "source_train"),
                          ("test", "target_test")):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            _validate_grid(pair["input"], f"{split}[{index}].input")
            _validate_grid(pair["output"], f"{split}[{index}].output")
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            _require(input_key not in inputs and pair_key not in pairs,
                     "duplicate input or pair")
            _require(pair_key not in _SOURCE_PAIR_KEYS, "source pair copied")
            inputs.add(input_key)
            pairs.add(pair_key)
            foreground, background, _ = _validate_pair_transformation(pair, regime)
            if split == "train":
                height, width = pair["output"].shape
                train_heights.add(height)
                train_widths.add(width)
                train_backgrounds.add(background)
                mask = pair["output"] == foreground
                retained_background |= bool(np.any(~mask) and _boundary_gaps(mask))
    _require(len(train_heights) >= 2 and len(train_widths) >= 2,
             "training does not vary both crop dimensions")
    _require(len(train_backgrounds) >= 2,
             "training does not demonstrate variable background roles")
    _require(retained_background,
             "training never demonstrates retained background in a crop")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    parameters = sample_puzzle_parameters(rng)
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seed_index = 0
    for split, regime, count in (("train", "source_train", train_count),
                                 ("test", "target_test", test_count)):
        for pair_index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[seed_index]))
            seed_index += 1
            for _ in range(100):
                previous = puzzle["train"] + puzzle["test"]
                pair = _sample_pair(pair_rng, regime, parameters, pair_index,
                                    train_count, previous)
                key = _grid_key(pair["input"])
                if key not in seen_inputs and _pair_key(pair) not in _SOURCE_PAIR_KEYS:
                    break
            else:
                raise RuntimeError("could not construct a unique non-source pair")
            seen_inputs.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if not np.array_equal(a["input"], b["input"]):
                return False
            if not np.array_equal(a["output"], b["output"]):
                return False
    return True


def _expect_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected ValueError was not raised")


def _predict_from_visible_pairs(training_pairs, test_input):
    # This helper receives only displayed demonstrations and a test input.
    for pair in training_pairs:
        choices = _role_candidates(pair["input"])
        _require(len(choices) == 1, "ambiguous training input")
        top, bottom, left, right = choices[0][2]
        _require(np.array_equal(pair["output"],
                                pair["input"][top:bottom, left:right]),
                 "training does not demonstrate cropping")
    choices = _role_candidates(test_input)
    _require(len(choices) == 1, "test input has ambiguous color roles")
    top, bottom, left, right = choices[0][2]
    return test_input[top:bottom, left:right].copy()


def self_test() -> None:
    automatic = generate_puzzle(20260827)
    explicit = generate_puzzle(314159, 7, 3)
    _require(len(explicit["train"]) == 7 and len(explicit["test"]) == 3,
             "explicit counts not honored")
    _require(_puzzles_equal(automatic, generate_puzzle(20260827))
             and _puzzles_equal(explicit, generate_puzzle(314159, 7, 3)),
             "generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(1001, 6, 2),
                                generate_puzzle(1002, 6, 2)),
             "different seeds produced the same puzzle")
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None),
                                    (10, None))):
        puzzle = generate_puzzle(8000 + index, *counts)
        validate_generated_puzzle(puzzle)
        if counts[0] is not None:
            _require(len(puzzle["train"]) == counts[0], "train count lost")
        if counts[1] is not None:
            _require(len(puzzle["test"]) == counts[1], "test count lost")
    for callback in (lambda: generate_puzzle(1, num_train=1),
                     lambda: generate_puzzle(1, num_train=11),
                     lambda: generate_puzzle(1, num_test=0),
                     lambda: generate_puzzle(1, num_test=5),
                     lambda: generate_puzzle(1, 2, 2),
                     lambda: generate_puzzle(1, 3, 4),
                     lambda: generate_puzzle(1, 4, 4),
                     lambda: generate_puzzle(-1)):
        _expect_value_error(callback)
    train_counts, test_counts = set(), set()
    input_shapes, output_shapes, mask_keys = set(), set(), set()
    foregrounds, backgrounds = set(), set()
    regimes = set()
    coverage = {name: 0 for name in (
        "zero_foreground", "zero_background", "repeated_foreground",
        "test_palette_overlap", "tiny", "large", "tall", "wide", "square",
        "singleton", "line", "solid", "multiple_holes", "boundary_gaps",
        "edge", "corner", "spanning", "asymmetric", "canvas_30",
        "foreground_majority", "low_scale_puzzle", "zero_with_gaps",
        "large_with_background", "majority_at_edge", "thin_dense_frame")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_areas = [p["output"].size for p in puzzle["train"]]
        train_colors = [_role_candidates(p["input"])[0][0]
                        for p in puzzle["train"]]
        test_colors = [_role_candidates(p["input"])[0][0]
                       for p in puzzle["test"]]
        coverage["repeated_foreground"] += (
            len(set(train_colors + test_colors)) < len(train_colors + test_colors))
        coverage["test_palette_overlap"] += bool(set(train_colors) & set(test_colors))
        coverage["low_scale_puzzle"] += max(
            p["output"].size for p in puzzle["train"] + puzzle["test"]) <= 100
        for pair in puzzle["train"] + puzzle["test"]:
            grid, output = pair["input"], pair["output"]
            candidates = _role_candidates(grid)
            _require(len(candidates) == 1, "ambiguous input")
            foreground, background, bounds = candidates[0]
            _require(np.array_equal(output, grid[bounds[0]:bounds[1],
                                                    bounds[2]:bounds[3]]),
                     "independent pair crop check failed")
            foregrounds.add(foreground)
            backgrounds.add(background)
            input_shapes.add(grid.shape)
            output_shapes.add(output.shape)
            mask = output == foreground
            mask_keys.add((mask.shape, tuple(bool(x) for x in mask.flat)))
            top, bottom, left, right = bounds
            margins = (top, grid.shape[0] - bottom,
                       left, grid.shape[1] - right)
            majority = int(np.count_nonzero(grid == foreground)) > grid.size / 2
            edge = any(margin == 0 for margin in margins)
            coverage["zero_foreground"] += foreground == 0
            coverage["zero_background"] += background == 0
            coverage["tiny"] += output.size <= 9
            coverage["large"] += output.size >= 100
            coverage["tall"] += output.shape[0] > output.shape[1]
            coverage["wide"] += output.shape[0] < output.shape[1]
            coverage["square"] += output.shape[0] == output.shape[1]
            coverage["singleton"] += output.shape == (1, 1)
            coverage["line"] += output.size > 1 and 1 in output.shape
            coverage["solid"] += output.size > 1 and bool(np.all(mask))
            coverage["multiple_holes"] += _enclosed_holes(mask) >= 2
            coverage["boundary_gaps"] += _boundary_gaps(mask) >= 2
            coverage["edge"] += edge
            coverage["corner"] += (margins[0] == 0 or margins[1] == 0) and (
                margins[2] == 0 or margins[3] == 0)
            coverage["spanning"] += (margins[0] == margins[1] == 0) or (
                margins[2] == margins[3] == 0)
            coverage["asymmetric"] += max(margins) - min(margins) >= 6
            coverage["canvas_30"] += 30 in grid.shape
            coverage["foreground_majority"] += majority
            coverage["zero_with_gaps"] += foreground == 0 and _boundary_gaps(mask) >= 2
            coverage["large_with_background"] += output.size >= 100 and bool(np.any(~mask))
            coverage["majority_at_edge"] += majority and edge
            coverage["thin_dense_frame"] += (
                output.size >= 100 and float(mask.mean()) >= .70
                and margins == (1, 1, 1, 1))
        for pair in puzzle["test"]:
            predicted = _predict_from_visible_pairs(
                [{"input": p["input"].copy(), "output": p["output"].copy()}
                 for p in puzzle["train"]], pair["input"].copy())
            _require(np.array_equal(predicted, pair["output"]),
                     "displayed evidence did not predict the test output")
            area = pair["output"].size
            regimes.add("larger" if area > max(train_areas) else
                        "smaller" if area < min(train_areas) else "comparable")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic counts lack range")
    _require(len(input_shapes) >= 35 and len(output_shapes) >= 35,
             "grid dimensions lack range")
    _require(len(mask_keys) >= 100, "object geometries lack range")
    _require(len(foregrounds) == 10 and len(backgrounds) == 10,
             "palette roles lack range")
    _require(regimes == {"larger", "smaller", "comparable"},
             "test size regimes lack range")
    for name, minimum in (
        ("zero_foreground", 10), ("zero_background", 10),
        ("repeated_foreground", 10), ("test_palette_overlap", 10),
        ("tiny", 10), ("large", 20), ("tall", 40), ("wide", 40),
        ("square", 10), ("singleton", 3), ("line", 10),
        ("solid", 20), ("multiple_holes", 10), ("boundary_gaps", 50),
        ("edge", 50), ("corner", 20), ("spanning", 20),
        ("asymmetric", 30), ("canvas_30", 5),
        ("foreground_majority", 10), ("low_scale_puzzle", 4),
        ("zero_with_gaps", 4), ("large_with_background", 10),
        ("majority_at_edge", 5), ("thin_dense_frame", 3)):
        _require(coverage[name] >= minimum,
                 f"42-seed review coverage too low for {name}")
