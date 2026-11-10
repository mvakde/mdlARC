import numpy as np


_TALLY_SITES = (
    (0, 0),
    (0, 2),
    (1, 1),
    (2, 0),
    (2, 2),
)

_SOURCE_ROWS = (
    (
        ("000000", "022000", "022000", "000000", "002200", "002200"),
        ("101", "000", "000"),
    ),
    (
        (
            "0000220",
            "0220220",
            "0220000",
            "0000022",
            "2200022",
            "2202200",
            "0002200",
        ),
        ("101", "010", "101"),
    ),
    (
        (
            "0022022",
            "0022022",
            "2200000",
            "2202200",
            "0002200",
            "0000000",
            "0000000",
        ),
        ("101", "010", "100"),
    ),
    (
        ("000", "022", "022"),
        ("100", "000", "000"),
    ),
    (
        ("00000", "02200", "02200", "00022", "00022"),
        ("101", "000", "000"),
    ),
    (
        (
            "0000000",
            "0220000",
            "0220220",
            "0000220",
            "0022000",
            "0022000",
            "0000000",
        ),
        ("101", "010", "000"),
    ),
    (
        (
            "0000220",
            "0000220",
            "0220000",
            "0220220",
            "0000220",
            "0220000",
            "0220000",
        ),
        ("101", "010", "100"),
    ),
    (
        ("22000", "22000", "00000", "00000", "00000"),
        ("100", "000", "000"),
    ),
    (
        (
            "0000000",
            "2202200",
            "2202200",
            "0000022",
            "0022022",
            "0022000",
            "0000000",
        ),
        ("101", "010", "100"),
    ),
    (
        ("000220", "220220", "220000", "002200", "002200", "000000"),
        ("101", "010", "000"),
    ),
    (
        (
            "2202200",
            "2202200",
            "0000022",
            "0220022",
            "0220000",
            "0000220",
            "0000220",
        ),
        ("101", "010", "101"),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _array_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _rows_signature(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_PAIR_SIGNATURES = frozenset(
    (_rows_signature(input_rows), _rows_signature(output_rows))
    for input_rows, output_rows in _SOURCE_ROWS
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    result: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue

            stack = [(row, col)]
            seen[row, col] = True
            component: list[tuple[int, int]] = []

            while stack:
                current_row, current_col = stack.pop()
                component.append((current_row, current_col))

                for delta_row, delta_col in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col

                    if not (
                        0 <= next_row < height
                        and 0 <= next_col < width
                    ):
                        continue

                    if (
                        mask[next_row, next_col]
                        and not seen[next_row, next_col]
                    ):
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))

            result.append(component)

    return result


def _square_positions_for_color(
    grid: np.ndarray,
    color: int,
) -> tuple[tuple[int, int], ...] | None:
    components = _components(grid == color)

    if not 1 <= len(components) <= 5:
        return None

    positions: list[tuple[int, int]] = []

    for component in components:
        if len(component) != 4:
            return None

        rows = sorted({row for row, _ in component})
        cols = sorted({col for _, col in component})

        if len(rows) != 2 or len(cols) != 2:
            return None
        if rows[1] != rows[0] + 1 or cols[1] != cols[0] + 1:
            return None

        expected = {
            (row, col)
            for row in rows
            for col in cols
        }
        if set(component) != expected:
            return None

        positions.append((rows[0], cols[0]))

    return tuple(sorted(positions))


def _infer_input_roles(
    grid: np.ndarray,
) -> list[tuple[int, tuple[int, ...], tuple[tuple[int, int], ...]]]:
    colors = [int(value) for value in np.unique(grid)]
    candidates = []
    if not 2 <= len(colors) <= 6:
        return candidates
    if len(colors) == 2:
        for object_color in colors:
            positions = _square_positions_for_color(grid, object_color)
            if positions is not None:
                candidates.append((next(color for color in colors
                                        if color != object_color),
                                   (object_color,), positions))
        return candidates
    for background_color in colors:
        components = _components(grid != background_color)
        if not 1 <= len(components) <= 5:
            continue
        positions = []
        for component in components:
            rows = {row for row, _ in component}
            cols = {col for _, col in component}
            if (len(component) != 4 or len(rows) != 2 or len(cols) != 2 or
                max(rows)-min(rows) != 1 or max(cols)-min(cols) != 1 or
                len({int(grid[row, col]) for row, col in component}) != 1):
                break
            positions.append((min(rows), min(cols)))
        else:
            candidates.append((background_color,
                               tuple(color for color in colors if color != background_color),
                               tuple(sorted(positions))))

    return candidates


def _make_output(
    count: int,
    background_color: int,
    marker_color: int,
) -> np.ndarray:
    output = np.full(
        (3, 3),
        background_color,
        dtype=np.int8,
    )

    for row, col in _TALLY_SITES[:count]:
        output[row, col] = marker_color

    return output


def _positions_are_compatible(
    first: tuple[int, int],
    second: tuple[int, int],
) -> bool:
    row_distance = abs(first[0] - second[0])
    col_distance = abs(first[1] - second[1])

    # These conditions reject overlap and orthogonal edge contact.
    # A (2, 2) displacement remains legal diagonal contact.
    conflict = (
        (row_distance <= 2 and col_distance <= 1)
        or (col_distance <= 2 and row_distance <= 1)
    )
    return not conflict


def _extend_positions(
    ordered_candidates: list[tuple[int, int]],
    selected: list[tuple[int, int]],
    target_count: int,
    search_budget: list[int],
    forbid_diagonal: bool = False,
) -> tuple[tuple[int, int], ...] | None:
    if len(selected) == target_count:
        return tuple(selected)

    if len(selected) + len(ordered_candidates) < target_count:
        return None

    for index, position in enumerate(ordered_candidates):
        search_budget[0] -= 1
        if search_budget[0] < 0:
            return None

        if not all(
            _positions_are_compatible(position, other)
            and (not forbid_diagonal or
                 abs(position[0] - other[0]) != 2 or
                 abs(position[1] - other[1]) != 2)
            for other in selected
        ):
            continue

        selected.append(position)
        result = _extend_positions(
            ordered_candidates[index + 1 :],
            selected,
            target_count,
            search_budget,
            forbid_diagonal,
        )

        if result is not None:
            return result

        selected.pop()

    return None


def _sample_positions(
    rng: np.random.Generator,
    height: int,
    width: int,
    count: int,
    arrangement: str,
    interior: bool,
) -> tuple[tuple[int, int], ...] | None:
    candidates = [
        (row, col)
        for row in range(height - 1)
        for col in range(width - 1)
        if not interior or (1 <= row <= height - 3 and
                            1 <= col <= width - 3)
    ]
    if not candidates:
        return None

    if arrangement == "diagonal":
        diagonal_seed_pairs = []

        for index, first in enumerate(candidates):
            for second in candidates[index + 1 :]:
                if (
                    abs(first[0] - second[0]) == 2
                    and abs(first[1] - second[1]) == 2
                ):
                    diagonal_seed_pairs.append((first, second))

        if not diagonal_seed_pairs:
            return None

        for pair_index in rng.permutation(len(diagonal_seed_pairs))[:32]:
            first, second = diagonal_seed_pairs[int(pair_index)]

            remaining = [
                position
                for position in candidates
                if position != first and position != second
            ]

            center_row = (first[0] + second[0]) / 2
            center_col = (first[1] + second[1]) / 2
            jitter = rng.random(len(remaining))
            ordered = [p for _, p in sorted(
                (max(abs(p[0]-center_row), abs(p[1]-center_col))
                 + float(jitter[i]) * 2, p)
                for i, p in enumerate(remaining))]

            result = _extend_positions(
                ordered,
                [first, second],
                count,
                [10000],
            )
            if result is not None:
                return tuple(sorted(result))

        return None

    for _ in range(8):
        ordering = rng.permutation(len(candidates))
        ordered = [
            candidates[int(index)]
            for index in ordering
        ]
        if arrangement == "cluster":
            anchor = ordered[0]
            jitter = rng.random(len(ordered))
            ordered = [p for _, p in sorted(
                (max(abs(p[0]-anchor[0]), abs(p[1]-anchor[1]))
                 + 2 * float(jitter[i]), p)
                for i, p in enumerate(ordered))]

        result = _extend_positions(
            ordered,
            [],
            count,
            [10000],
            arrangement == "separated",
        )
        if result is not None:
            return tuple(sorted(result))

    return None


def _has_diagonal_touch(
    positions: tuple[tuple[int, int], ...],
) -> bool:
    for index, first in enumerate(positions):
        for second in positions[index + 1 :]:
            if (
                abs(first[0] - second[0]) == 2
                and abs(first[1] - second[1]) == 2
            ):
                return True

    return False


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    return {"marker": int(rng.integers(0, 10))}


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    object_count: int,
    background_color: int,
    object_colors: tuple[int, ...],
    arrangement: str,
    size_mode: str,
    interior: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError(f"unknown pair regime: {regime}")
    minimum = (3, 4, 5, 6, 6)[object_count - 1]
    for _ in range(80):
        if size_mode == "compact":
            height = int(rng.integers(minimum, minimum + 4))
            width = int(rng.integers(minimum, minimum + 4))
            if object_count == 5 and rng.integers(0, 2) == 0:
                height, width = ((6, 7) if rng.integers(0, 2) == 0
                                 else (7, 6))
        elif size_mode == "roomy":
            height = int(rng.integers(9, 17))
            width = int(rng.integers(9, 17))
        elif size_mode == "tall":
            height = int(rng.integers(10, 19))
            width = int(rng.integers(max(5, minimum), 10))
        elif size_mode == "wide":
            height = int(rng.integers(max(5, minimum), 10))
            width = int(rng.integers(10, 19))
        else:
            height = int(rng.integers(max(6, minimum), 13))
            width = int(rng.integers(max(6, minimum), 13))
        if size_mode in ("compact", "medium", "roomy") and rng.integers(0, 4) == 0:
            width = height
        positions = _sample_positions(
            rng,
            height,
            width,
            object_count,
            arrangement,
            interior,
        )
        if positions is None:
            continue

        input_grid = np.full(
            (height, width),
            background_color,
            dtype=np.int8,
        )

        colors_for_squares = list(object_colors)
        colors_for_squares.extend(
            int(rng.choice(object_colors))
            for _ in range(object_count-len(object_colors))
        )
        rng.shuffle(colors_for_squares)
        for (row, col), object_color in zip(positions, colors_for_squares):
            input_grid[row : row + 2, col : col + 2] = (
                object_color
            )

        # Reject any rare arrangement whose complement could itself be
        # mistaken for a collection of solid 2x2 objects.
        if len(_infer_input_roles(input_grid)) != 1:
            continue

        output_grid = _make_output(
            object_count,
            background_color,
            puzzle_parameters["marker"],
        )
        return input_grid, output_grid

    raise ValueError("could not construct a valid pair")


def _valid_count_value(
    value: object,
    lower: int,
    upper: int,
) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and lower <= int(value) <= upper
    )


def _resolve_pair_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None and not _valid_count_value(
        num_train, 2, 10
    ):
        raise ValueError("num_train must be between 2 and 10")

    if num_test is not None and not _valid_count_value(
        num_test, 1, 4
    ):
        raise ValueError("num_test must be between 1 and 4")

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))

    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))

    else:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    return num_train, num_test


def _plan_object_counts(
    rng: np.random.Generator,
    num_train: int,
    num_test: int,
) -> tuple[list[int], list[int]]:
    withheld_five = num_train >= 4 and bool(rng.integers(0, 3) == 0)
    pool = [1, 2, 3, 4] if withheld_five else [1, 2, 3, 4, 5]
    order = [pool[int(i)] for i in rng.permutation(len(pool))]
    train_counts = order[:min(num_train, len(pool))]
    train_counts.extend(int(rng.choice(pool)) for _ in range(num_train-len(train_counts)))
    if withheld_five:
        test_counts = [5]
        test_counts.extend(int(rng.choice(pool)) for _ in range(num_test-1))
    else:
        test_counts = [int(rng.choice(train_counts)) for _ in range(num_test)]
    return train_counts, test_counts


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if (
        not isinstance(seed, (int, np.integer))
        or isinstance(seed, (bool, np.bool_))
    ):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))

    puzzle_parameters = sample_puzzle_parameters(rng)

    resolved_train, resolved_test = _resolve_pair_counts(
        rng,
        num_train,
        num_test,
    )

    train_counts, test_counts = _plan_object_counts(
        rng,
        resolved_train,
        resolved_test,
    )
    dense_index = int(rng.choice(
        [i for i, count in enumerate(train_counts) if count >= 2]
    ))
    interior_index = int(rng.choice(
        [i for i in range(resolved_train) if i != dense_index]
    ))
    multicolor_enabled = resolved_train >= 3 and bool(rng.integers(0, 2))
    multicolor_index = (int(rng.choice(
        [i for i, count in enumerate(train_counts) if count >= 2]
    )) if multicolor_enabled else -1)

    puzzle = {
        "train": [],
        "test": [],
    }

    seen_inputs = set()
    used_pair_seeds = set()
    used_train_roles = []

    specifications = (
        ("train", "source_train", train_counts),
        ("test", "target_test", test_counts),
    )

    for split, regime, object_counts in specifications:
        for pair_index, object_count in enumerate(object_counts):
            for _ in range(256):
                pair_seed = int(
                    rng.integers(0, np.iinfo(np.int64).max)
                )
                if pair_seed in used_pair_seeds:
                    continue
                used_pair_seeds.add(pair_seed)

                pair_rng = np.random.default_rng(pair_seed)
                available = [c for c in range(10) if c != puzzle_parameters["marker"]]
                if split == "train" and pair_index == 1:
                    backgrounds = [c for c in available if c != used_train_roles[0][0]]
                    objects = [c for c in available if c != used_train_roles[0][1]]
                else:
                    backgrounds = available
                    objects = available
                background = int(pair_rng.choice(backgrounds))
                object_options = [c for c in objects if c != background]
                object_color = int(pair_rng.choice(object_options))
                use_multiple_colors = (
                    object_count >= 2 and multicolor_enabled and
                    ((split == "train" and
                      (pair_index == multicolor_index or pair_rng.integers(0, 5) == 0))
                     or (split == "test" and pair_rng.integers(0, 2) == 0))
                )
                object_colors = [object_color]
                if use_multiple_colors:
                    extras = [c for c in available
                              if c != background and c != object_color]
                    extra_count = min(object_count-1, int(pair_rng.integers(1, 3)))
                    object_colors.extend(int(c) for c in pair_rng.choice(
                        extras, size=extra_count, replace=False))

                if split == "train" and pair_index == dense_index:
                    arrangement = "diagonal"
                else:
                    arrangement = str(pair_rng.choice(
                        ["diagonal", "separated", "cluster", "random"]
                    ))
                if split == "train" and pair_index == interior_index:
                    size_mode = str(pair_rng.choice(["medium", "roomy", "tall", "wide"]))
                    interior = True
                else:
                    size_mode = str(pair_rng.choice(
                        ["compact", "compact", "medium", "roomy", "tall", "wide"]
                    ))
                    interior = bool(pair_rng.integers(0, 2))
                    if size_mode == "compact" and object_count >= 4:
                        interior = False
                if object_count == 1 and arrangement == "diagonal":
                    arrangement = "separated"
                input_grid, output_grid = generate_pair(
                    pair_rng,
                    regime,
                    puzzle_parameters,
                    object_count,
                    background,
                    tuple(object_colors),
                    arrangement,
                    size_mode,
                    interior,
                )

                input_signature = _array_signature(input_grid)
                pair_signature = (
                    input_signature,
                    _array_signature(output_grid),
                )

                if input_signature in seen_inputs:
                    continue
                if pair_signature in _SOURCE_PAIR_SIGNATURES:
                    continue

                seen_inputs.add(input_signature)
                if split == "train":
                    used_train_roles.append((background, object_color))
                puzzle[split].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                break
            else:
                raise ValueError(
                    "could not construct a unique non-source pair"
                )

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _require(isinstance(puzzle, dict), "puzzle must be a dict")
        _require(
            len(puzzle) == 2
            and set(puzzle) == {"train", "test"},
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

        train_length = len(puzzle["train"])
        test_length = len(puzzle["test"])

        _require(2 <= train_length <= 10, "illegal train length")
        _require(1 <= test_length <= 4, "illegal test length")
        _require(
            train_length > test_length,
            "train length must exceed test length",
        )

        seen_inputs = set()
        seen_pairs = set()
        puzzle_markers = []
        train_backgrounds = set()
        train_objects = set()
        train_diagonal = False
        test_diagonal = False
        train_multicolor = False
        test_multicolor = False
        split_counts: dict[str, list[int]] = {
            "train": [],
            "test": [],
        }

        for split in ("train", "test"):
            for pair in puzzle[split]:
                _require(
                    isinstance(pair, dict),
                    "each pair must be a dict",
                )
                _require(
                    len(pair) == 2
                    and set(pair) == {"input", "output"},
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
                    _require(
                        1 <= grid.shape[0] <= 30
                        and 1 <= grid.shape[1] <= 30,
                        f"{grid_name} dimensions are invalid",
                    )
                    _require(
                        np.issubdtype(grid.dtype, np.integer),
                        f"{grid_name} must use an integer dtype",
                    )
                    _require(
                        grid.size > 0
                        and int(grid.min()) >= 0
                        and int(grid.max()) <= 9,
                        f"{grid_name} has an invalid color",
                    )

                input_signature = _array_signature(pair["input"])
                pair_signature = (
                    input_signature,
                    _array_signature(pair["output"]),
                )

                _require(
                    input_signature not in seen_inputs,
                    "duplicate input grid",
                )
                _require(
                    pair_signature not in seen_pairs,
                    "duplicate input/output pair",
                )
                _require(
                    pair_signature not in _SOURCE_PAIR_SIGNATURES,
                    "generated pair exactly duplicates a source pair",
                )

                seen_inputs.add(input_signature)
                seen_pairs.add(pair_signature)

                role_candidates = _infer_input_roles(pair["input"])
                _require(
                    len(role_candidates) == 1,
                    "input object/background roles are ambiguous",
                )

                background_color, object_color, positions = (
                    role_candidates[0]
                )
                object_count = len(positions)

                output_grid = pair["output"]
                _require(
                    output_grid.shape == (3, 3),
                    "every output must be 3x3",
                )

                output_colors = {
                    int(value)
                    for value in np.unique(output_grid)
                }
                _require(
                    len(output_colors) == 2
                    and background_color in output_colors,
                    "output must use the shared background and one marker",
                )

                marker_color = next(
                    color
                    for color in output_colors
                    if color != background_color
                )
                _require(
                    marker_color not in object_color,
                    "semantic colors must be distinct",
                )

                expected_output = _make_output(
                    object_count,
                    background_color,
                    marker_color,
                )
                _require(
                    np.array_equal(output_grid, expected_output),
                    "output does not encode the object count correctly",
                )

                puzzle_markers.append(marker_color)
                split_counts[split].append(object_count)

                if split == "train":
                    train_backgrounds.add(background_color)
                    train_objects.add(object_color)
                    train_diagonal |= _has_diagonal_touch(positions)
                    train_multicolor |= len(object_color) > 1
                else:
                    test_diagonal |= _has_diagonal_touch(positions)
                    test_multicolor |= len(object_color) > 1

        _require(
            len(set(puzzle_markers)) == 1,
            "output marker must stay fixed within a puzzle",
        )
        _require(
            len(train_backgrounds) >= 2 and len(train_objects) >= 2,
            "training must demonstrate pair-level input color changes",
        )
        _require(
            train_diagonal,
            "training must demonstrate diagonal object separation",
        )
        _require(
            not test_diagonal or train_diagonal,
            "diagonal contact in test is not demonstrated in training",
        )
        _require(
            not test_multicolor or train_multicolor,
            "multiple square colors in test are not demonstrated in training",
        )
        train_counts = set(split_counts["train"])
        for count in split_counts["test"]:
            _require(count in train_counts or (count == 5 and
                     {1, 2, 3, 4}.issubset(train_counts)),
                     "test count lacks a demonstrated output prefix")

    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError("malformed generated puzzle") from exc


def _puzzles_equal(first: dict, second: dict) -> bool:
    if (
        not isinstance(first, dict)
        or not isinstance(second, dict)
        or set(first) != set(second)
    ):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for first_pair, second_pair in zip(
            first[split],
            second[split],
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


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            tuple(
                (
                    _array_signature(pair["input"]),
                    _array_signature(pair["output"]),
                )
                for pair in puzzle[split]
            ),
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    train_lengths, test_lengths = set(), set()
    shapes, layouts, palettes, fingerprints = set(), set(), set(), set()
    test_counts, test_diagonal, test_separated = set(), 0, 0
    test_interior, roomy_interior, rectangles = 0, 0, 0
    withheld, smaller_test, color_varying = 0, 0, 0
    combined = 0
    tall, wide, five_nonseven, six_seven_five = 0, 0, 0, 0
    novel_background, novel_object, asymmetric = 0, 0, 0
    clustered, dispersed = 0, 0
    multicolor_train_puzzles, multicolor_test_puzzles = 0, 0

    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        train_lengths.add(len(puzzle["train"]))
        test_lengths.add(len(puzzle["test"]))
        fingerprints.add(_puzzle_fingerprint(puzzle))
        train_backgrounds, train_objects, train_markers = set(), set(), set()
        train_counts = set()
        train_areas = []
        train_has_multicolor = False
        for pair in puzzle["train"]:
            background, object_color, positions = _infer_input_roles(pair["input"])[0]
            train_backgrounds.add(background)
            train_objects.add(object_color)
            train_counts.add(len(positions))
            train_areas.append(pair["input"].size)
            train_has_multicolor |= len(object_color) > 1
            train_markers.add(next(int(c) for c in np.unique(pair["output"])
                                   if int(c) != background))
        _require(len(train_markers) == 1, "marker changed within training")
        color_varying += len(train_backgrounds) >= 2 and len(train_objects) >= 2
        marker = next(iter(train_markers))
        multicolor_train_puzzles += train_has_multicolor
        test_has_multicolor = False
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, out = pair["input"], pair["output"]
                background, object_color, positions = _infer_input_roles(grid)[0]
                count = len(positions)
                _require(int(np.count_nonzero(np.isin(grid, object_color))) == 4*count,
                         "object area is inconsistent with square count")
                predicted = np.full((3, 3), background, dtype=np.int8)
                for row, col in ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))[:count]:
                    predicted[row, col] = marker
                _require(np.array_equal(predicted, out),
                         "train-derived marker and counted squares do not predict output")
                shapes.add(tuple(grid.shape))
                layouts.add((tuple(grid.shape), tuple(positions)))
                palettes.add((background, object_color, marker))
                if split == "test":
                    test_has_multicolor |= len(object_color) > 1
                    test_counts.add(count)
                    diagonal = _has_diagonal_touch(positions)
                    interior = all(0 < row and row+2 < grid.shape[0] and
                                   0 < col and col+2 < grid.shape[1]
                                   for row, col in positions)
                    test_diagonal += diagonal
                    test_separated += not diagonal and count >= 2
                    test_interior += interior
                    roomy_interior += interior and min(grid.shape) >= 9
                    rectangles += grid.shape[0] != grid.shape[1]
                    tall += grid.shape[0] >= grid.shape[1] + 3
                    wide += grid.shape[1] >= grid.shape[0] + 3
                    five_nonseven += count == 5 and grid.shape != (7, 7)
                    six_seven_five += count == 5 and sorted(grid.shape) == [6, 7]
                    novel_background += background not in train_backgrounds
                    novel_object += object_color not in train_objects
                    asymmetric += interior and (
                        min(r for r, c in positions) !=
                        grid.shape[0]-2-max(r for r, c in positions) or
                        min(c for r, c in positions) !=
                        grid.shape[1]-2-max(c for r, c in positions))
                    if count >= 3:
                        span = max(max(abs(r-r2), abs(c-c2))
                                   for r,c in positions for r2,c2 in positions)
                        clustered += span <= 5
                        dispersed += span >= 7
                    combined += interior and not diagonal and count >= 3 and \
                                grid.shape[0] != grid.shape[1]
                    if count not in train_counts:
                        _require(count == 5 and {1,2,3,4}.issubset(train_counts),
                                 "unshown test count lacks prefix evidence")
                        withheld += 1
        smaller_test += any(p["input"].size < max(train_areas) for p in puzzle["test"])
        _require(not test_has_multicolor or train_has_multicolor,
                 "test has multiple colors without training example")
        multicolor_test_puzzles += test_has_multicolor

    _require(len(train_lengths) >= 4 and len(test_lengths) >= 3,
             "automatic pair counts have insufficient variation")
    _require(len(fingerprints) >= 40 and len(layouts) >= 100,
             "insufficient puzzle or layout diversity")
    _require(len(shapes) >= 25 and len(palettes) >= 50,
             "insufficient dimension or pair-color diversity")
    _require(test_counts == {1, 2, 3, 4, 5}, "test counts do not cover 1-5")
    _require(test_diagonal >= 10 and test_separated >= 10,
             "test layouts lack contact and separation regimes")
    _require(test_interior >= 10 and roomy_interior >= 5 and rectangles >= 20,
             "roomy, interior, or rectangular tests are underrepresented")
    _require(tall >= 5 and wide >= 5 and asymmetric >= 5,
             "aspect ratios or asymmetric margins are underrepresented")
    _require(five_nonseven >= 5 and six_seven_five >= 1,
             "five-square canvas variety is underrepresented")
    _require(novel_background >= 5 and novel_object >= 5,
             "test input colors rarely differ from training colors")
    _require(clustered >= 5 and dispersed >= 5,
             "clustered or dispersed layouts are underrepresented")
    _require(multicolor_train_puzzles >= 5 and multicolor_test_puzzles >= 3,
             "multicolor squares are underrepresented")
    _require(withheld >= 2 and smaller_test >= 10 and combined >= 2,
             "missing withheld count or combined test variation")
    _require(color_varying == 42, "input roles must vary within each puzzle")

    _require(_puzzles_equal(generate_puzzle(90210, 7, 3),
                            generate_puzzle(90210, 7, 3)),
             "fixed-seed generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(90210, 7, 3),
                                generate_puzzle(90211, 7, 3)),
             "different seeds produced the same puzzle")
    for seed, ntrain, ntest in ((100, 2, 1), (101, 10, 1), (102, 10, 4)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (ntrain, ntest), "explicit pair count was not honored")
    for seed, ntrain, ntest in ((200, None, 1), (201, None, 4),
                                (202, 2, None), (203, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(puzzle)
        _require(ntrain is None or len(puzzle["train"]) == ntrain,
                 "partial train count was not honored")
        _require(ntest is None or len(puzzle["test"]) == ntest,
                 "partial test count was not honored")
    for ntrain, ntest in ((1, None), (11, None), (None, 0), (None, 5),
                          (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(300, ntrain, ntest)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts were accepted")
