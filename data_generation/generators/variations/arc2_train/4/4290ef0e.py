import numpy as np


_SIDES = ("top", "right", "bottom", "left")

# Compare actual source inputs; their canvas dimensions remain available.
_SOURCE_INPUTS = {
    (
        "888888888888888888", "888888888888888888",
        "881118111888888888", "881888881888888888",
        "881888881888888888", "888888888888888888",
        "881888881888808888", "881888881888888888",
        "881118111888888888", "888888888888888888",
        "888888888888888888", "844488888888888888",
        "848488822822888888", "844488828882888888",
        "888888888888888888", "888888828882888888",
        "888888822822888888", "888888888888888888",
    ),
    (
        "44444444444444444", "44444444444433344",
        "44411411444434344", "44414441444433344",
        "44444444444444444", "44414441444444444",
        "44411411444444444", "44444444446644466",
        "44444444446444446", "44444444444444444",
        "44444444444444444", "44444444444444444",
        "44444444446444446",
    ),
    (
        "333333333333333333", "222333333333333333",
        "332331113331113883", "332331333333313833",
        "333331333333313333", "332333333333333833",
        "332333377733333883", "222333373733333333",
        "333331377733313333", "333331333333313333",
        "333331113331113333", "333333333333333333",
        "333333333333333333", "333443333333443333",
        "333433333333343333", "333333333333333333",
        "333333333333333363", "333333333333333333",
    ),
    (
        "1111111111111411111", "1111111111111111111",
        "1111111111111411111", "1111111888111411111",
        "1111111818111411111", "1111111888111411111",
        "1111111111111444441", "1111111111111111111",
        "1113313311111111111", "1113111311111111111",
        "1111111111111111111", "1113111311111111111",
        "1113313311111111111", "1111111111111166111",
        "1111111111111161111", "1111222111222111111",
        "1111211111112111111", "1111211111112111111",
        "1111111111111161111",
    ),
}


def _is_source_input(grid: np.ndarray) -> bool:
    return tuple("".join(str(int(cell)) for cell in row) for row in grid) in _SOURCE_INPUTS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _frame_cells(size: int, arm: int) -> tuple[tuple[int, int], ...]:
    cells = []
    for row in range(size):
        for col in range(size):
            on_horizontal_side = row == 0 or row == size - 1
            on_vertical_side = col == 0 or col == size - 1
            near_horizontal_corner = col < arm or col >= size - arm
            near_vertical_corner = row < arm or row >= size - arm

            if (
                on_horizontal_side
                and near_horizontal_corner
            ) or (
                on_vertical_side
                and near_vertical_corner
            ):
                cells.append((row, col))

    return tuple(cells)


def _visible_cells(
    size: int,
    arm: int,
    top: int,
    left: int,
    height: int,
    width: int,
) -> frozenset[tuple[int, int]]:
    return frozenset(
        (top + row, left + col)
        for row, col in _frame_cells(size, arm)
        if 0 <= top + row < height and 0 <= left + col < width
    )


def _matching_placements(
    mask: frozenset[tuple[int, int]],
    size: int,
    arm: int,
    height: int,
    width: int,
) -> tuple[tuple[int, int], ...]:
    if not mask:
        return ()

    anchor = min(mask)
    matches = set()

    # In a valid placement, the lexicographically first visible cell must
    # correspond to one of the frame's colored cells.
    for pattern_row, pattern_col in _frame_cells(size, arm):
        top = anchor[0] - pattern_row
        left = anchor[1] - pattern_col

        if _visible_cells(
            size,
            arm,
            top,
            left,
            height,
            width,
        ) == mask:
            matches.add((top, left))

    return tuple(sorted(matches))


def _background_of(grid: np.ndarray) -> int:
    counts = np.bincount(
        grid.ravel().astype(np.int64),
        minlength=10,
    )
    maximum = int(counts.max())
    winners = np.flatnonzero(counts == maximum)

    _require(
        len(winners) == 1,
        "input background is not uniquely identifiable",
    )
    return int(winners[0])


def _decode_input(grid: np.ndarray) -> dict:
    height, width = grid.shape
    background = _background_of(grid)

    colors = [
        int(color)
        for color in np.unique(grid)
        if int(color) != background
    ]
    _require(colors, "input contains no motifs")

    masks = {
        color: frozenset(zip(*np.where(grid == color)))
        for color in colors
    }

    one_cell_colors = [
        color
        for color in colors
        if len(masks[color]) == 1
    ]
    _require(
        len(one_cell_colors) <= 1,
        "input has more than one singleton candidate",
    )

    singleton = one_cell_colors[0] if one_cell_colors else None
    frame_colors = [
        color
        for color in colors
        if color != singleton
    ]

    frame_count = len(frame_colors)
    _require(frame_count >= 1, "input contains no frame")

    largest = 2 * frame_count + 1
    _require(largest <= 29, "too many frame colors")

    sizes = tuple(range(3, largest + 1, 2))
    candidates = {}

    for color in frame_colors:
        semantic_candidates = []

        for size in sizes:
            for arm in range(1, (size + 1) // 2 + 1):
                placements = _matching_placements(
                    masks[color],
                    size,
                    arm,
                    height,
                    width,
                )
                if placements:
                    semantic_candidates.append(
                        (size, arm, placements)
                    )

        _require(
            semantic_candidates,
            f"color {color} is not a recoverable frame",
        )
        candidates[color] = semantic_candidates

    search_order = sorted(
        frame_colors,
        key=lambda color: (len(candidates[color]), color),
    )
    solutions = []

    def search(
        index: int,
        used_sizes: set[int],
        assignment: dict,
    ) -> None:
        if len(solutions) >= 2:
            return

        if index == len(search_order):
            if len(used_sizes) == len(sizes):
                solutions.append(dict(assignment))
            return

        color = search_order[index]

        for size, arm, placements in candidates[color]:
            if size in used_sizes:
                continue

            assignment[size] = {
                "color": color,
                "arm": arm,
                "placements": placements,
            }
            search(
                index + 1,
                used_sizes | {size},
                assignment,
            )
            del assignment[size]

    search(0, set(), {})

    _require(
        len(solutions) == 1,
        "input does not have exactly one size/arm reconstruction",
    )

    return {
        "background": background,
        "singleton": singleton,
        "frames": solutions[0],
        "height": height,
        "width": width,
    }


def _construct_output(decoded: dict) -> np.ndarray:
    sizes = sorted(decoded["frames"])
    largest = sizes[-1]

    output = np.full(
        (largest, largest),
        decoded["background"],
        dtype=np.int8,
    )

    for size in sizes:
        frame = decoded["frames"][size]
        offset = (largest - size) // 2

        for row, col in _frame_cells(size, frame["arm"]):
            output[offset + row, offset + col] = frame["color"]

    if decoded["singleton"] is not None:
        center = largest // 2
        output[center, center] = decoded["singleton"]

    return output


def _placement_sides(
    top: int,
    left: int,
    size: int,
    height: int,
    width: int,
) -> frozenset[str]:
    sides = []

    if top < 0:
        sides.append("top")
    if left + size > width:
        sides.append("right")
    if top + size > height:
        sides.append("bottom")
    if left < 0:
        sides.append("left")

    return frozenset(sides)


def _analyze_pair(pair: dict) -> dict:
    input_grid = pair["input"]
    output_grid = pair["output"]

    decoded = _decode_input(input_grid)
    expected = _construct_output(decoded)

    _require(
        np.array_equal(output_grid, expected),
        "output is not the uniquely reconstructed concentric assembly",
    )

    profiles = {}
    for size, frame in decoded["frames"].items():
        profiles[size] = frozenset(
            _placement_sides(
                top,
                left,
                size,
                decoded["height"],
                decoded["width"],
            )
            for top, left in frame["placements"]
        )

    return {
        "background": decoded["background"],
        "singleton": decoded["singleton"],
        "frames": decoded["frames"],
        "profiles": profiles,
        "largest": expected.shape[0],
        "input_shape": input_grid.shape,
        "input_colors": frozenset(
            int(color)
            for color in np.unique(input_grid)
        ),
    }


def _certain_side_set(
    options: frozenset[frozenset[str]],
) -> set[str] | None:
    if len(options) != 1:
        return None
    return set(next(iter(options)))


def _clipping_summary(info: dict) -> tuple[int, set[str]]:
    clipped_count = 0
    distinct_sides = set()

    for options in info["profiles"].values():
        side_set = _certain_side_set(options)
        if side_set:
            clipped_count += 1
            distinct_sides.update(side_set)

    return clipped_count, distinct_sides


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    # Palette order is a shared preference, not a color-to-size mapping.
    return {"palette": tuple(int(x) for x in rng.permutation(10))}


def _select_colors(
    rng: np.random.Generator,
    parameters: dict,
    frame_count: int,
    has_center: bool,
) -> tuple[int, tuple[int, ...], int | None]:
    if rng.random() < 0.22:
        colors = list(parameters["palette"])
    else:
        colors = [int(x) for x in rng.permutation(10)]
    background = colors[0]
    frames = tuple(colors[1:frame_count + 1])
    singleton = colors[frame_count + 1] if has_center else None
    return background, frames, singleton


def _sample_arms(
    rng: np.random.Generator, sizes: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        int(rng.integers(1, (size + 1) // 2 + 1))
        for size in sizes
    )


def _sample_dimensions(
    rng: np.random.Generator, largest: int
) -> tuple[int, int]:
    minimum = max(largest, 5)
    if rng.random() < 0.28:
        short = int(rng.integers(minimum, min(30, minimum + 6) + 1))
        long = int(rng.integers(max(short, 22), 31))
        return (short, long) if rng.random() < 0.5 else (long, short)
    return (
        int(rng.integers(minimum, 31)),
        int(rng.integers(minimum, 31)),
    )


def _clip_styles(
    rng: np.random.Generator,
    sizes: tuple[int, ...],
    arms: tuple[int, ...],
    complexity: str,
    regime: str,
) -> tuple[str, ...]:
    if complexity == "clear":
        return ("full",) * len(sizes)

    chance = 0.42 if complexity == "mixed" else 0.72
    styles = []
    for size, arm in zip(sizes, arms):
        if size == 3 and arm == 1:
            styles.append("full")
            continue
        if rng.random() >= chance:
            styles.append("full")
            continue
        possible = ["shallow", "deep", "corner"]
        weights = (0.35, 0.34, 0.31) if regime == "source_train" else (0.22, 0.36, 0.42)
        styles.append(str(rng.choice(possible, p=weights)))

    if all(style == "full" for style in styles):
        eligible = [i for i, size in enumerate(sizes) if size >= 5]
        if not eligible:
            eligible = [0]
        styles[int(rng.choice(eligible))] = "shallow"
    return tuple(styles)


def _sample_translation(
    rng: np.random.Generator,
    height: int,
    width: int,
    size: int,
    arm: int,
    style: str,
) -> tuple[int, int]:
    if style == "full":
        return (
            int(rng.integers(0, height - size + 1)),
            int(rng.integers(0, width - size + 1)),
        )

    if style == "corner":
        vertical = str(rng.choice(("top", "bottom")))
        horizontal = str(rng.choice(("left", "right")))
        sides = (vertical, horizontal)
    else:
        sides = (str(rng.choice(_SIDES)),)

    def depth() -> int:
        if style == "shallow":
            high = min(size - 2, max(1, arm))
            return int(rng.integers(1, high + 1))
        if style == "deep":
            low = min(size - 2, max(1, arm + 1))
            return int(rng.integers(low, size - 1))
        return int(rng.integers(1, size - 1))

    top = int(rng.integers(0, height - size + 1))
    left = int(rng.integers(0, width - size + 1))
    for side in sides:
        amount = depth()
        if side == "top":
            top = -amount
        elif side == "bottom":
            top = height - size + amount
        elif side == "left":
            left = -amount
        else:
            left = width - size + amount
    return top, left


def _build_scattered_input(
    rng: np.random.Generator,
    height: int,
    width: int,
    background: int,
    sizes: tuple[int, ...],
    arms: tuple[int, ...],
    frame_colors: tuple[int, ...],
    singleton: int | None,
    styles: tuple[str, ...],
) -> np.ndarray:
    order = sorted(
        range(len(sizes)),
        key=lambda i: (styles[i] == "full", -sizes[i]),
    )
    occupied: set[tuple[int, int]] = set()
    placements = {}
    for index in order:
        accepted = False
        for _ in range(150):
            top, left = _sample_translation(
                rng, height, width, sizes[index], arms[index], styles[index]
            )
            cells = _visible_cells(
                sizes[index], arms[index], top, left, height, width
            )
            if len(cells) < 4 or not cells.isdisjoint(occupied):
                continue
            placements[index] = cells
            occupied.update(cells)
            accepted = True
            break
        if not accepted:
            raise ValueError("could not place scattered motifs")

    center_cell = None
    if singleton is not None:
        for flat in rng.permutation(height * width):
            cell = (int(flat) // width, int(flat) % width)
            if cell in occupied:
                continue
            if any(
                (cell[0] + dr, cell[1] + dc) in occupied
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            ):
                continue
            center_cell = cell
            break
        if center_cell is None:
            raise ValueError("could not place singleton")

    grid = np.full((height, width), background, dtype=np.int8)
    for index, cells in placements.items():
        for row, col in cells:
            grid[row, col] = frame_colors[index]
    if center_cell is not None:
        grid[center_cell] = singleton
    return grid


def _sample_frame_count(
    rng: np.random.Generator,
    has_center: bool,
    regime: str,
    band: str | None,
) -> int:
    maximum = 8 if has_center else 9
    if band == "small":
        choices = (1, 2, 3, 4)
    elif band == "large":
        choices = (4, 5, 6, 7)
    else:
        bands = ("small", "middle", "large")
        probabilities = (0.25, 0.48, 0.27) if regime == "source_train" else (0.20, 0.42, 0.38)
        selected = str(rng.choice(bands, p=probabilities))
        if selected == "small":
            choices = (1, 2, 3)
        elif selected == "middle":
            choices = (3, 4, 5, 6)
        else:
            choices = (5, 6, 7, 8, 9)
    choices = tuple(count for count in choices if count <= maximum)
    return int(rng.choice(choices))


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    parameters: dict,
    has_center: bool,
    complexity: str,
    band: str | None = None,
) -> dict:
    for _ in range(90):
        frame_count = _sample_frame_count(rng, has_center, regime, band)
        sizes = tuple(range(3, 2 * frame_count + 2, 2))
        height, width = _sample_dimensions(rng, sizes[-1])
        background, frame_colors, singleton = _select_colors(
            rng, parameters, frame_count, has_center
        )
        arms = _sample_arms(rng, sizes)
        styles = _clip_styles(rng, sizes, arms, complexity, regime)
        try:
            input_grid = _build_scattered_input(
                rng, height, width, background, sizes, arms,
                frame_colors, singleton, styles,
            )
            if _is_source_input(input_grid):
                continue
            intended = {
                "background": background,
                "singleton": singleton,
                "frames": {
                    size: {"color": frame_colors[i], "arm": arms[i]}
                    for i, size in enumerate(sizes)
                },
            }
            pair = {"input": input_grid, "output": _construct_output(intended)}
            _analyze_pair(pair)
            return pair
        except (AssertionError, ValueError):
            continue
    raise ValueError("unable to generate an unambiguous pair")


def _validate_optional_count(value: int | None, name: str, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} is outside its legal range")


def _resolve_counts(
    seed: int, num_train: int | None, num_test: int | None
) -> tuple[int, int]:
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)
    seed_code = int(seed)
    if num_train is None and num_test is None:
        test_count = 1 + seed_code % 4
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + (seed_code // 4) % (11 - minimum_train)
    elif num_train is None:
        test_count = int(num_test)
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + seed_code % (11 - minimum_train)
    elif num_test is None:
        train_count = int(num_train)
        test_count = 1 + seed_code % min(4, train_count - 1)
    else:
        train_count, test_count = int(num_train), int(num_test)
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)

    train_states = [True, False] + [bool(rng.integers(0, 2)) for _ in range(train_count - 2)]
    rng.shuffle(train_states)
    train_complexities = ["clear", "mixed"] + [
        str(rng.choice(("clear", "mixed", "heavy"), p=(0.20, 0.42, 0.38)))
        for _ in range(train_count - 2)
    ]
    rng.shuffle(train_complexities)
    train_bands = ["small", "large"] + [None] * (train_count - 2)
    rng.shuffle(train_bands)
    test_states = [bool(rng.integers(0, 2)) for _ in range(test_count)]
    if test_count >= 2:
        test_states[:2] = [True, False]
        rng.shuffle(test_states)

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train":
                has_center = train_states[index]
                complexity = train_complexities[index]
                band = train_bands[index]
                regime = "source_train"
            else:
                has_center = test_states[index]
                if index == 0:
                    complexity = str(rng.choice(("mixed", "heavy")))
                else:
                    complexity = str(rng.choice(
                        ("clear", "mixed", "heavy"), p=(0.08, 0.35, 0.57)
                    ))
                band = None
                regime = "target_test"
            for _ in range(8):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                try:
                    pair = _generate_pair(
                        np.random.default_rng(child_seed), regime, parameters,
                        has_center, complexity, band,
                    )
                except ValueError:
                    continue
                key = _grid_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("could not generate distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(value) for value in grid.ravel())


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")

    seen_inputs = set()
    seen_pairs = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be 2D NumPy arrays")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         "grid dimensions outside 1..30")
                _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC color")
            input_key = _grid_key(pair["input"])
            output_key = _grid_key(pair["output"])
            _require(input_key not in seen_inputs, "duplicate input")
            _require((input_key, output_key) not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            _require(not _is_source_input(pair["input"]), "source input copied")
            _require(not np.array_equal(pair["input"], pair["output"]),
                     "unchanged output")
            info = _analyze_pair(pair)
            _require(len(info["frames"]) + 1 + (info["singleton"] is not None) <= 10,
                     "palette roles collide")
            _require(set(info["frames"]) == set(range(3, info["largest"] + 1, 2)),
                     "frame sizes are not consecutive odd sizes")
            _require(pair["output"].shape == (info["largest"], info["largest"]),
                     "incorrect output crop")
            infos[split].append(info)

    train_infos, test_infos = infos["train"], infos["test"]
    _require({info["singleton"] is not None for info in train_infos} == {False, True},
             "training must demonstrate center presence and absence")
    _require(any(_clipping_summary(info)[0] == 0 for info in train_infos),
             "training lacks a complete assembly")
    _require(any(_clipping_summary(info)[0] > 0 for info in train_infos),
             "training lacks clipped motifs")
    _require(any(_clipping_summary(info)[0] > 0 for info in test_infos),
             "test lacks the clipped-frame shift")
    # The only shared convention is the reconstruction rule. Literal colors,
    # dimensions, arms, clipping and center state are read from each input.
    for info in test_infos:
        _require(info["largest"] >= 3, "test lacks a frame")


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != set(second):
        return False
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for left, right in zip(first[split], second[split]):
            if not np.array_equal(left["input"], right["input"]):
                return False
            if not np.array_equal(left["output"], right["output"]):
                return False
    return True


def _expect_value_error(call) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    automatic = generate_puzzle(17)
    explicit = generate_puzzle(23, 6, 3)
    validate_generated_puzzle(automatic)
    validate_generated_puzzle(explicit)
    _require(_puzzles_equal(generate_puzzle(12345, 7, 3),
                           generate_puzzle(12345, 7, 3)),
             "generation is not deterministic")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(300 + train_count + test_count,
                                 train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit counts ignored")
    for seed, train_count, test_count in (
        (401, None, 1), (402, None, 4), (403, 2, None), (404, 10, None)
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "partial train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "partial test count ignored")
    for call in (
        lambda: generate_puzzle(0, num_train=1),
        lambda: generate_puzzle(0, num_train=11),
        lambda: generate_puzzle(0, num_test=0),
        lambda: generate_puzzle(0, num_test=5),
        lambda: generate_puzzle(0, 2, 2),
        lambda: generate_puzzle(0, 3, 4),
    ):
        _expect_value_error(call)

    train_counts, test_counts, puzzles = set(), set(), set()
    frame_counts, output_sizes, arms, backgrounds = set(), set(), set(), set()
    shapes, orientations, clip_sides = set(), set(), set()
    center_states, hard_clips, deep_clips = set(), 0, 0
    within_count, within_background, within_shape, within_center = 0, 0, 0, 0
    total_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprint = []
        local_counts, local_backgrounds, local_shapes, local_centers = set(), set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                total_pairs += 1
                info = _analyze_pair(pair)
                fingerprint.append((_grid_key(pair["input"]), _grid_key(pair["output"])))
                count = len(info["frames"])
                frame_counts.add(count)
                output_sizes.add(info["largest"])
                backgrounds.add(info["background"])
                shapes.add(info["input_shape"])
                center = info["singleton"] is not None
                center_states.add(center)
                local_counts.add(count)
                local_backgrounds.add(info["background"])
                local_shapes.add(info["input_shape"])
                local_centers.add(center)
                height, width = info["input_shape"]
                orientations.add("wide" if width > height else "tall" if height > width else "square")
                for size, frame in info["frames"].items():
                    arms.add((size, frame["arm"]))
                    mask = np.argwhere(pair["input"] == frame["color"])
                    if mask[:, 0].max() - mask[:, 0].min() + 1 < size and \
                       mask[:, 1].max() - mask[:, 1].min() + 1 < size:
                        hard_clips += 1
                    for sides in info["profiles"][size]:
                        clip_sides.add(tuple(sorted(sides)))
                        if len(sides) == 1 and len(mask) < len(_frame_cells(size, frame["arm"])) - size:
                            deep_clips += 1
        puzzles.add(tuple(fingerprint))
        within_count += len(local_counts) > 1
        within_background += len(local_backgrounds) > 1
        within_shape += len(local_shapes) > 1
        within_center += len(local_centers) > 1

    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(puzzles) == 42, "puzzles repeat across seeds")
    _require(frame_counts >= {1, 2, 3, 4, 5, 6, 7},
             "frame-count range is too narrow")
    _require(len(shapes) >= 45 and len(arms) >= 25,
             "shape or arm diversity is too low")
    _require(len(backgrounds) == 10 and center_states == {False, True},
             "color or center diversity is too low")
    _require(orientations == {"wide", "tall", "square"},
             "aspect-ratio diversity is too low")
    _require(hard_clips >= 10 and deep_clips >= 10 and len(clip_sides) >= 9,
             "clipping diversity is too low")
    _require(min(within_count, within_background, within_shape, within_center) >= 35,
             "within-puzzle diversity is too low")
    _require(total_pairs >= 150, "too few measured pairs")
