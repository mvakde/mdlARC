import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalize_requested_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return value


def _choose_pair_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_requested_count(
        "num_train", num_train, 2, 10
    )
    test_count = _normalize_requested_count(
        "num_test", num_test, 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif train_count is None:
        minimum_train = max(2, int(test_count) + 1)
        if minimum_train > 10:
            raise ValueError("no legal num_train exists for num_test")
        train_count = int(rng.integers(minimum_train, 11))
    else:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("could not select a legal pair-count combination")

    return int(train_count), int(test_count)


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = rng.choice(np.arange(10), size=3, replace=False)
    return {
        "background": int(colors[0]),
        "foreground": int(colors[1]),
        "contrast": int(colors[2]),
    }


def _construct_output(
    input_grid: np.ndarray,
    background: int,
    foreground: int,
    contrast: int,
) -> np.ndarray:
    foreground_mask = input_grid == foreground
    touches_left = bool(np.any(foreground_mask[:, 0]))
    touches_right = bool(np.any(foreground_mask[:, -1]))

    _require(
        touches_left != touches_right,
        "foreground must touch exactly one lateral boundary",
    )

    reflected_mask = np.fliplr(foreground_mask)
    reflected_negative = np.where(
        reflected_mask,
        background,
        contrast,
    ).astype(input_grid.dtype, copy=False)

    if touches_left:
        return np.concatenate(
            (reflected_negative, input_grid.copy()),
            axis=1,
        )

    return np.concatenate(
        (input_grid.copy(), reflected_negative),
        axis=1,
    )


def _make_mask(
    rng: np.random.Generator,
    height: int,
    width: int,
    side: str,
    density_band: str,
) -> np.ndarray:
    """Build geometry with independent row, contact, density, and motif choices.

    Columns are temporarily measured inward from the contacted edge.  This
    makes the two directions exactly symmetric without constraining their
    independently sampled shapes.
    """
    _require(side in {"left", "right"}, "invalid contact side")
    mask = np.zeros((height, width), dtype=bool)

    # A true one-pixel boundary pattern is a useful extreme of the rule.
    if density_band == "sparse" and rng.random() < 0.16:
        mask[int(rng.integers(height)), 0] = True
        return mask if side == "left" else np.fliplr(mask).copy()

    row_mode = int(rng.integers(4))
    if row_mode == 0:
        active = np.ones(height, dtype=bool)
    else:
        portion = float(rng.uniform(0.25, 1.0))
        count = max(1, min(height, int(round(height * portion))))
        active = np.zeros(height, dtype=bool)
        if row_mode == 1:
            start = int(rng.integers(0, height - count + 1))
            active[start : start + count] = True
        elif row_mode == 2:
            active[rng.choice(height, size=count, replace=False)] = True
        else:
            # Two blocks make mixed long and short vertical gaps possible.
            start = int(rng.integers(0, height - count + 1))
            active[start : start + count] = True
            swaps = int(rng.integers(1, max(2, count // 2 + 1)))
            for _ in range(swaps):
                on = np.flatnonzero(active)
                off = np.flatnonzero(~active)
                if not len(off):
                    break
                active[int(rng.choice(on))] = False
                active[int(rng.choice(off))] = True

    contact_mode = int(rng.integers(4))
    if contact_mode == 3:
        active[:] = True
    available = np.flatnonzero(active)
    contact = np.zeros(height, dtype=bool)
    if contact_mode == 0:
        contact[int(rng.choice(available))] = True
    elif contact_mode == 1:
        count = int(rng.integers(1, len(available) + 1))
        contact[rng.choice(available, size=count, replace=False)] = True
    elif contact_mode == 2:
        first = int(rng.choice(available))
        run = int(rng.integers(1, max(2, height // 2 + 1)))
        contact[first : min(height, first + run)] = active[
            first : min(height, first + run)
        ]
        if not np.any(contact):
            contact[first] = True
    else:
        contact[:] = True

    density_ranges = {
        "sparse": (0.02, 0.20),
        "medium": (0.18, 0.52),
        "dense": (0.55, 0.97),
    }
    low, high = density_ranges[density_band]
    fill_probability = float(rng.uniform(low, high))
    motif = int(rng.integers(4))

    # Interior random cells can coexist with runs, stepped boundaries, or
    # detached patches.  The final column is always the untouched edge.
    if width > 2:
        mask[:, 1 : width - 1] = (
            rng.random((height, width - 2)) < fill_probability
        ) & active[:, None]

    if motif in (0, 3):
        # Runs on neighboring contacted rows form boundary-anchored clusters.
        for row in np.flatnonzero(contact):
            if rng.random() < 0.75:
                length = int(rng.integers(1, width))
                mask[int(row), :length] = True
        if motif == 3 and height > 1:
            # A changing run width makes a stepped or bent contour.
            start = int(rng.integers(height))
            span = int(rng.integers(2, min(height, 7) + 1))
            length = int(rng.integers(1, width))
            for row in range(start, min(height, start + span)):
                if active[row]:
                    mask[row, :length] = True
                length = max(1, min(width - 1, length + int(rng.integers(-1, 2))))
    elif motif == 1:
        # A bounded vertical walk gives four-connected turns when rows align.
        col = int(rng.integers(1, width - 1))
        for row in range(height):
            if active[row] and rng.random() < 0.8:
                mask[row, col] = True
                if col > 1 and rng.random() < 0.35:
                    mask[row, col - 1] = True
            col = max(1, min(width - 2, col + int(rng.integers(-1, 2))))
    else:
        # Small detached islands are independent of the boundary contact.
        for _ in range(int(rng.integers(1, 4))):
            row = int(rng.integers(height))
            col = int(rng.integers(1, width - 1))
            row_end = min(height, row + int(rng.integers(1, 4)))
            col_end = min(width - 1, col + int(rng.integers(1, 3)))
            mask[row:row_end, col:col_end] |= active[row:row_end, None]

    mask[:, 0] = contact
    mask[:, -1] = False
    mask[~active] = False

    # Repetition is optional.  Copy only between rows with the same contact
    # state so the separately chosen contact geometry remains intact.
    if rng.random() < 0.30:
        for state in (False, True):
            candidates = np.flatnonzero(active & (contact == state))
            if len(candidates) >= 2:
                source, target = rng.choice(candidates, size=2, replace=False)
                if np.any(mask[int(source)]):
                    mask[int(target)] = mask[int(source)]
                break

    _require(np.any(mask[:, 0]), "missing boundary contact")
    _require(not np.any(mask[:, -1]), "opposite edge reached")
    return mask if side == "left" else np.fliplr(mask).copy()


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    side: str,
    density_band: str,
) -> tuple[np.ndarray, np.ndarray]:
    background = int(puzzle_parameters["background"])
    foreground = int(puzzle_parameters["foreground"])
    contrast = int(puzzle_parameters["contrast"])

    if regime == "source_train":
        height = int(rng.integers(3, 11))
        width = int(rng.integers(3, 7))
    elif regime == "target_test":
        height_band = int(rng.integers(3))
        width_band = int(rng.integers(3))
        height_limits = ((3, 10), (11, 17), (18, 30))
        width_limits = ((3, 6), (7, 10), (11, 15))
        height_low, height_high = height_limits[height_band]
        width_low, width_high = width_limits[width_band]
        height = int(rng.integers(height_low, height_high + 1))
        width = int(rng.integers(width_low, width_high + 1))
    else:
        raise ValueError(f"unknown regime: {regime}")

    mask = _make_mask(rng, height, width, side, density_band)

    input_grid = np.full(
        mask.shape,
        background,
        dtype=np.int8,
    )
    input_grid[mask] = foreground

    output_grid = _construct_output(
        input_grid,
        background,
        foreground,
        contrast,
    )

    return input_grid, output_grid


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return _grid_key(input_grid), _grid_key(output_grid)


def _source_task_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    source_rows = [
        (
            "020",
            "022",
            "002",
        ),
        (
            "200000",
            "222000",
            "002000",
            "022200",
            "002200",
            "220000",
            "222220",
            "200000",
        ),
        (
            "220",
            "200",
            "220",
        ),
        (
            "00000",
            "00002",
            "00020",
            "00202",
            "02222",
            "00002",
            "00220",
            "00002",
            "00022",
            "00000",
        ),
        (
            "000000",
            "000002",
            "000020",
            "000222",
            "000022",
            "022200",
            "000222",
            "000002",
            "000022",
            "000022",
            "000000",
            "000000",
        ),
    ]

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for rows in source_rows:
        input_grid = np.array(
            [[int(character) for character in row] for row in rows],
            dtype=np.int8,
        )
        output_grid = _construct_output(
            input_grid,
            background=0,
            foreground=2,
            contrast=8,
        )
        pairs.append((input_grid, output_grid))

    return pairs


def _is_exact_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for source_input, source_output in _source_task_pairs():
        if (
            input_grid.shape == source_input.shape
            and output_grid.shape == source_output.shape
            and np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_pair_counts(
        rng,
        num_train,
        num_test,
    )
    puzzle_parameters = sample_puzzle_parameters(rng)

    if rng.random() < 0.65:
        train_sides = ["left", "right"]
        train_sides.extend(
            "left" if rng.random() < 0.5 else "right"
            for _ in range(train_count - 2)
        )
    else:
        common_side = "left" if rng.random() < 0.5 else "right"
        train_sides = [common_side] * train_count
    train_sides = [train_sides[int(i)] for i in rng.permutation(train_count)]

    # These are coverage targets, shuffled independently of pair position and
    # contact direction.  Pair geometry itself remains independently sampled.
    train_bands = ["sparse", "dense"]
    train_bands.extend(
        ("sparse", "medium", "dense")[int(rng.integers(3))]
        for _ in range(train_count - 2)
    )
    train_bands = [train_bands[int(i)] for i in rng.permutation(train_count)]

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()

    def append_unique_pair(
        split: str,
        regime: str,
        side: str,
        density_band: str,
    ) -> None:
        child_seed = int(
            rng.integers(0, np.iinfo(np.int64).max)
        )
        pair_rng = np.random.default_rng(child_seed)

        for _ in range(256):
            input_grid, output_grid = generate_pair(
                pair_rng,
                regime,
                puzzle_parameters,
                side,
                density_band,
            )
            input_key = _grid_key(input_grid)
            pair_key = _pair_key(input_grid, output_grid)

            if split == "train":
                foreground_fraction = float(np.mean(
                    input_grid == puzzle_parameters["foreground"]
                ))
                if density_band == "sparse" and foreground_fraction > 0.25:
                    continue
                if density_band == "dense" and foreground_fraction <= 0.52:
                    continue

            if input_key in seen_inputs:
                continue
            if pair_key in seen_pairs:
                continue
            if _is_exact_source_pair(input_grid, output_grid):
                continue

            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            puzzle[split].append(
                {
                    "input": input_grid,
                    "output": output_grid,
                }
            )
            return

        raise ValueError("unable to generate a unique valid pair")

    for side, density_band in zip(train_sides, train_bands):
        append_unique_pair(
            split="train",
            regime="source_train",
            side=side,
            density_band=density_band,
        )

    for _ in range(test_count):
        if len(set(train_sides)) == 1:
            side = train_sides[0]
        else:
            side = "left" if rng.random() < 0.5 else "right"
        append_unique_pair(
            split="test",
            regime="target_test",
            side=side,
            density_band=("sparse", "medium", "dense")[
                int(rng.integers(3))
            ],
        )

    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_puzzle_rule(
    puzzle: dict,
) -> tuple[int, int, int, list[str]]:
    all_pairs = puzzle["train"] + puzzle["test"]
    _require(bool(all_pairs), "puzzle contains no pairs")

    input_palette = {
        int(value)
        for value in np.unique(all_pairs[0]["input"])
    }
    _require(
        len(input_palette) == 2,
        "inputs must share a binary palette",
    )

    for pair in all_pairs:
        palette = {
            int(value)
            for value in np.unique(pair["input"])
        }
        _require(
            palette == input_palette,
            "input color roles are inconsistent",
        )

    contrast_color: int | None = None
    for pair in all_pairs:
        output_palette = {
            int(value)
            for value in np.unique(pair["output"])
        }
        extras = output_palette - input_palette
        _require(
            len(extras) == 1,
            "each output must introduce one contrast color",
        )
        current_contrast = next(iter(extras))
        _require(
            output_palette
            == input_palette | {current_contrast},
            "output palette is inconsistent",
        )

        if contrast_color is None:
            contrast_color = current_contrast
        else:
            _require(
                contrast_color == current_contrast,
                "contrast color changes between pairs",
            )

    _require(
        contrast_color is not None,
        "contrast color could not be inferred",
    )

    palette_values = sorted(input_palette)
    candidates: list[tuple[int, int, int, list[str]]] = []

    for foreground in palette_values:
        background = (
            palette_values[1]
            if foreground == palette_values[0]
            else palette_values[0]
        )
        directions: list[str] = []
        valid = True

        for pair in all_pairs:
            input_grid = pair["input"]
            foreground_mask = input_grid == foreground
            touches_left = bool(np.any(foreground_mask[:, 0]))
            touches_right = bool(np.any(foreground_mask[:, -1]))

            if touches_left == touches_right:
                valid = False
                break

            try:
                expected = _construct_output(
                    input_grid,
                    background,
                    foreground,
                    contrast_color,
                )
            except AssertionError:
                valid = False
                break

            if not np.array_equal(expected, pair["output"]):
                valid = False
                break

            directions.append(
                "left" if touches_left else "right"
            )

        if valid:
            candidates.append(
                (
                    background,
                    foreground,
                    contrast_color,
                    directions,
                )
            )

    _require(
        len(candidates) == 1,
        "puzzle color roles are ambiguous or inconsistent",
    )
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict), "puzzle must be a dictionary")
    _require(
        set(puzzle.keys()) == {"train", "test"},
        "puzzle must contain exactly train and test",
    )
    _require(
        isinstance(puzzle["train"], list),
        "train must be a list",
    )
    _require(
        isinstance(puzzle["test"], list),
        "test must be a list",
    )

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    _require(2 <= train_count <= 10, "illegal train count")
    _require(1 <= test_count <= 4, "illegal test count")
    _require(
        train_count > test_count,
        "train count must exceed test count",
    )

    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()

    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(
                isinstance(pair, dict),
                "each pair must be a dictionary",
            )
            _require(
                set(pair.keys()) == {"input", "output"},
                "pair must contain exactly input and output",
            )

            for grid_name in ("input", "output"):
                grid = pair[grid_name]
                _require(
                    isinstance(grid, np.ndarray),
                    f"{grid_name} must be a NumPy array",
                )
                _require(
                    grid.ndim == 2,
                    f"{grid_name} must be two-dimensional",
                )
                height, width = grid.shape
                _require(
                    1 <= height <= 30 and 1 <= width <= 30,
                    f"{grid_name} dimensions are out of range",
                )
                _require(
                    np.issubdtype(grid.dtype, np.integer),
                    f"{grid_name} must use an integer dtype",
                )
                _require(
                    int(grid.min()) >= 0 and int(grid.max()) <= 9,
                    f"{grid_name} contains an invalid color",
                )

            input_key = _grid_key(pair["input"])
            pair_key = _pair_key(
                pair["input"],
                pair["output"],
            )
            _require(
                input_key not in seen_inputs,
                "duplicate input grid",
            )
            _require(
                pair_key not in seen_pairs,
                "duplicate input/output pair",
            )
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            _require(
                not _is_exact_source_pair(
                    pair["input"],
                    pair["output"],
                ),
                "generated pair exactly duplicates a source pair",
            )

    background, foreground, contrast, directions = (
        _infer_puzzle_rule(puzzle)
    )

    train_only = {
        "train": puzzle["train"],
        "test": [],
    }
    (
        train_background,
        train_foreground,
        train_contrast,
        train_directions,
    ) = _infer_puzzle_rule(train_only)

    _require(
        (
            train_background,
            train_foreground,
            train_contrast,
        )
        == (background, foreground, contrast),
        "training pairs do not independently reveal color roles",
    )

    all_pairs = puzzle["train"] + puzzle["test"]
    for pair in all_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]
        height, width = input_grid.shape

        _require(
            output_grid.shape == (height, 2 * width),
            "output must preserve height and double width",
        )

        expected = _construct_output(
            input_grid,
            background,
            foreground,
            contrast,
        )
        _require(
            np.array_equal(output_grid, expected),
            "pair violates the reflected-complement rule",
        )
        _require(
            not np.array_equal(input_grid, output_grid),
            "input and output must not be identical",
        )

    for pair in puzzle["train"]:
        height, width = pair["input"].shape
        _require(
            3 <= height <= 10,
            "training height is outside source-train range",
        )
        _require(
            3 <= width <= 6,
            "training width is outside source-train range",
        )

    train_fractions = [
        float(np.mean(pair["input"] == foreground))
        for pair in puzzle["train"]
    ]
    _require(
        min(train_fractions) <= 0.25 and max(train_fractions) > 0.52,
        "training must show sparse and foreground-majority examples",
    )

    _require(
        set(train_directions) in ({"left"}, {"right"}, {"left", "right"}),
        "invalid training directions",
    )
    test_directions = directions[train_count:]

    for pair, direction in zip(
        puzzle["test"],
        test_directions,
    ):
        input_grid = pair["input"]
        height, width = input_grid.shape

        _require(3 <= width <= 15, "test width is outside its regime")
        _require(3 <= height <= 30, "test height is outside its regime")
        if len(set(train_directions)) == 1:
            _require(
                direction == train_directions[0],
                "a one-sided training set cannot transfer direction",
            )


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for first_pair, second_pair in zip(
            first[split],
            second[split],
        ):
            for grid_name in ("input", "output"):
                first_grid = first_pair[grid_name]
                second_grid = second_pair[grid_name]
                if first_grid.dtype != second_grid.dtype:
                    return False
                if not np.array_equal(first_grid, second_grid):
                    return False

    return True


def _puzzle_signature(puzzle: dict) -> tuple:
    signature: list[object] = [
        len(puzzle["train"]),
        len(puzzle["test"]),
    ]
    for split in ("train", "test"):
        signature.append(split)
        for pair in puzzle[split]:
            signature.append(_grid_key(pair["input"]))
            signature.append(_grid_key(pair["output"]))
    return tuple(signature)


def _component_count(mask: np.ndarray) -> int:
    remaining = mask.copy()
    components = 0
    height, width = mask.shape
    while np.any(remaining):
        row, col = (int(value) for value in np.argwhere(remaining)[0])
        remaining[row, col] = False
        frontier = [(row, col)]
        components += 1
        while frontier:
            current_row, current_col = frontier.pop()
            for next_row, next_col in (
                (current_row - 1, current_col),
                (current_row + 1, current_col),
                (current_row, current_col - 1),
                (current_row, current_col + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and remaining[next_row, next_col]
                ):
                    remaining[next_row, next_col] = False
                    frontier.append((next_row, next_col))
    return components


def self_test() -> None:
    automatic_a = generate_puzzle(271828)
    automatic_b = generate_puzzle(271828)
    _require(
        _puzzles_equal(automatic_a, automatic_b),
        "automatic-count generation is not deterministic",
    )
    validate_generated_puzzle(automatic_a)

    explicit_a = generate_puzzle(
        314159,
        num_train=7,
        num_test=3,
    )
    explicit_b = generate_puzzle(
        314159,
        num_train=7,
        num_test=3,
    )
    _require(
        _puzzles_equal(explicit_a, explicit_b),
        "explicit-count generation is not deterministic",
    )
    _require(
        len(explicit_a["train"]) == 7
        and len(explicit_a["test"]) == 3,
        "explicit counts were not used exactly",
    )
    validate_generated_puzzle(explicit_a)

    for seed, train_count, test_count in (
        (101, 3, 1),
        (102, 6, 2),
        (103, 9, 4),
    ):
        puzzle = generate_puzzle(
            seed,
            num_train=train_count,
            num_test=test_count,
        )
        _require(
            len(puzzle["train"]) == train_count,
            "wrong explicit train count",
        )
        _require(
            len(puzzle["test"]) == test_count,
            "wrong explicit test count",
        )
        validate_generated_puzzle(puzzle)

    for seed, train_count, test_count in (
        (201, 2, 1),
        (202, 10, 1),
        (203, 10, 4),
    ):
        puzzle = generate_puzzle(
            seed,
            num_train=train_count,
            num_test=test_count,
        )
        _require(
            len(puzzle["train"]) == train_count
            and len(puzzle["test"]) == test_count,
            "legal boundary counts failed",
        )
        validate_generated_puzzle(puzzle)

    partial_cases = (
        (301, None, 1),
        (302, None, 4),
        (303, 2, None),
        (304, 10, None),
    )
    for seed, requested_train, requested_test in partial_cases:
        puzzle = generate_puzzle(
            seed,
            num_train=requested_train,
            num_test=requested_test,
        )
        if requested_train is not None:
            _require(
                len(puzzle["train"]) == requested_train,
                "partial num_train was not preserved",
            )
        if requested_test is not None:
            _require(
                len(puzzle["test"]) == requested_test,
                "partial num_test was not preserved",
            )
        validate_generated_puzzle(puzzle)

    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
        {"num_train": True, "num_test": 1},
        {"num_train": 3, "num_test": False},
    )
    for kwargs in illegal_cases:
        raised = False
        try:
            generate_puzzle(400, **kwargs)
        except ValueError:
            raised = True
        _require(raised, f"illegal counts were accepted: {kwargs}")

    observed_train_counts: set[int] = set()
    observed_test_counts: set[int] = set()
    observed_palettes: set[tuple[int, int, int]] = set()
    observed_shapes: set[tuple[int, int]] = set()
    observed_layouts: set[tuple] = set()
    observed_puzzles: set[tuple] = set()
    train_side_modes: set[tuple[str, ...]] = set()
    test_sides: set[str] = set()
    test_size_relation: set[str] = set()
    test_width_bands: set[str] = set()
    test_height_bands: set[str] = set()
    test_empty_modes: set[str] = set()
    test_repeat_modes: set[bool] = set()
    component_modes: set[str] = set()
    single_pixels = 0
    majority_masks = 0
    full_contacts = 0
    one_row_contacts = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)

        observed_train_counts.add(len(puzzle["train"]))
        observed_test_counts.add(len(puzzle["test"]))
        observed_puzzles.add(_puzzle_signature(puzzle))

        background, foreground, contrast, directions = (
            _infer_puzzle_rule(puzzle)
        )
        observed_palettes.add(
            (background, foreground, contrast)
        )

        train_count = len(puzzle["train"])
        train_side_modes.add(tuple(sorted(set(directions[:train_count]))))
        test_sides.update(directions[train_count:])
        maximum_train_height = max(
            pair["input"].shape[0] for pair in puzzle["train"]
        )
        train_fractions = [
            float(np.mean(pair["input"] == foreground))
            for pair in puzzle["train"]
        ]
        _require(
            min(train_fractions) <= 0.25 and max(train_fractions) > 0.52,
            "training does not show both sparse and majority masks",
        )

        for pair in puzzle["test"]:
            input_grid = pair["input"]
            height, width = input_grid.shape
            test_size_relation.add(
                "familiar" if height <= maximum_train_height else "taller"
            )
            test_width_bands.add(
                "familiar" if width <= 6 else "wider"
            )
            test_height_bands.add(
                "small" if height <= 10 else
                "medium" if height <= 17 else "large"
            )
            test_mask = input_grid == foreground
            empty = int(np.sum(~np.any(test_mask, axis=1)))
            test_empty_modes.add("none" if empty == 0 else
                                 "many" if empty >= 3 else "some")
            rows = [tuple(row) for row in test_mask if np.any(row)]
            test_repeat_modes.add(len(rows) != len(set(rows)))
            components = _component_count(test_mask)
            component_modes.add("few" if components <= 3 else "many")

        for pair in puzzle["train"] + puzzle["test"]:
            input_grid = pair["input"]
            output_grid = pair["output"]
            height, width = input_grid.shape
            mask = input_grid == foreground
            touches_left = bool(np.any(mask[:, 0]))
            original = output_grid[:, width:] if touches_left else output_grid[:, :width]
            extension = output_grid[:, :width] if touches_left else output_grid[:, width:]
            _require(np.array_equal(original, input_grid), "input half changed")
            for row in range(height):
                for col in range(width):
                    expected_cell = (
                        background if mask[row, width - 1 - col]
                        else contrast
                    )
                    _require(
                        int(extension[row, col]) == expected_cell,
                        "self-test found an incorrect reflected cell",
                    )

            fraction = float(np.mean(mask))
            single_pixels += int(mask.sum() == 1)
            majority_masks += int(fraction > 0.5)
            contact_count = int(np.sum(mask[:, 0 if touches_left else -1]))
            full_contacts += int(contact_count == height)
            one_row_contacts += int(contact_count == 1)

            observed_shapes.add(
                (
                    int(input_grid.shape[0]),
                    int(input_grid.shape[1]),
                )
            )
            normalized_mask = (
                input_grid == foreground
            ).astype(np.uint8)
            observed_layouts.add(
                (
                    tuple(int(value) for value in input_grid.shape),
                    normalized_mask.tobytes(),
                )
            )

    _require(
        len(observed_train_counts) >= 4,
        "42 seeds did not produce four train counts",
    )
    _require(
        len(observed_test_counts) >= 3,
        "42 seeds did not produce three test counts",
    )
    _require(
        len(observed_palettes) >= 20,
        "puzzle-level color mappings lack diversity",
    )
    _require(
        len(observed_shapes) >= 12,
        "grid dimensions lack structural diversity",
    )
    _require(
        len(observed_layouts) >= 100,
        "foreground layouts lack substantial diversity",
    )
    _require(
        len(observed_puzzles) >= 40,
        "different seeds usually produced identical puzzles",
    )
    _require(
        {("left",), ("right",), ("left", "right")} <= train_side_modes,
        "training side modes lack coverage",
    )
    _require(test_sides == {"left", "right"}, "test sides lack coverage")
    _require(test_size_relation == {"familiar", "taller"},
             "test heights lack transfer range")
    _require(test_width_bands == {"familiar", "wider"},
             "test widths lack transfer range")
    _require(test_height_bands == {"small", "medium", "large"},
             "test height bands lack range")
    _require({"none", "many"} <= test_empty_modes,
             "test row activity lacks range")
    _require(test_repeat_modes == {False, True},
             "test row repetition lacks range")
    _require(component_modes == {"few", "many"},
             "test connectivity lacks range")
    _require(single_pixels > 0, "no single-pixel masks observed")
    _require(majority_masks > 0, "no foreground-majority masks observed")
    _require(full_contacts > 0 and one_row_contacts > 0,
             "boundary contact range is incomplete")
