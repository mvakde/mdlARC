import numpy as np


_AUTO_COUNTS = (
    (2, 1),
    (3, 2),
    (4, 1),
    (5, 3),
    (6, 2),
    (7, 4),
    (8, 3),
    (9, 1),
    (10, 4),
    (6, 1),
    (8, 2),
    (10, 3),
)


_SOURCE_TEXT = (
    (
        (
            "0000000011",
            "0000000011",
            "0000000311",
            "0000000311",
            "0000000311",
            "0000030311",
            "0000030311",
            "0000230311",
            "0000230311",
            "0222233311",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000311",
            "0000000311",
            "0000000311",
            "0000031311",
            "0000031311",
            "0000231311",
            "0000231311",
            "0222233311",
        ),
    ),
    (
        (
            "0030000000",
            "0030000001",
            "0030000111",
            "0030000111",
            "0030001111",
            "0030031111",
            "0030035115",
            "0031135115",
            "0031135115",
            "0033335555",
        ),
        (
            "0030000000",
            "0030000000",
            "0030000000",
            "0030000000",
            "0031111111",
            "0031131111",
            "0031135115",
            "0031135115",
            "0031135115",
            "0033335555",
        ),
    ),
    (
        (
            "0000000000",
            "9911000099",
            "0991000990",
            "0099000900",
            "0009009900",
            "0000099000",
            "0000040000",
            "2004040000",
            "2024040000",
            "2224440000",
        ),
        (
            "0000000000",
            "9900000099",
            "0990000990",
            "0099000900",
            "0009009900",
            "0000099000",
            "0000040000",
            "2004140000",
            "2124140000",
            "2224440000",
        ),
    ),
    (
        (
            "1100000000",
            "9911000000",
            "0991000000",
            "0099900005",
            "0000999005",
            "0000000505",
            "3000000005",
            "3300005505",
            "3330055505",
            "3333555505",
        ),
        (
            "0000000000",
            "9900000000",
            "0990000000",
            "0099900005",
            "0000999005",
            "0000000505",
            "3000000005",
            "3300005515",
            "3331155515",
            "3333555515",
        ),
    ),
)


def _text_grid(lines: tuple[str, ...]) -> np.ndarray:
    return np.asarray(
        [[int(character) for character in row] for row in lines],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_text_grid(input_lines), _text_grid(output_lines))
    for input_lines, output_lines in _SOURCE_TEXT
)


def _component(
    mask: np.ndarray,
    start: tuple[int, int],
) -> np.ndarray:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    stack = [start]
    seen[start] = True

    while stack:
        row, column = stack.pop()

        for next_row, next_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                0 <= next_row < height
                and 0 <= next_column < width
                and mask[next_row, next_column]
                and not seen[next_row, next_column]
            ):
                seen[next_row, next_column] = True
                stack.append((next_row, next_column))

    return seen


def _component_sizes(mask: np.ndarray) -> list[int]:
    unseen = mask.copy()
    sizes: list[int] = []

    while np.any(unseen):
        row, column = np.argwhere(unseen)[0]
        component = _component(
            unseen,
            (int(row), int(column)),
        )
        sizes.append(int(np.count_nonzero(component)))
        unseen[component] = False

    return sorted(sizes)


def _liquid_component(
    grid: np.ndarray,
    liquid: int,
) -> np.ndarray:
    liquid_cells = np.argwhere(grid == liquid)

    if liquid_cells.size == 0:
        raise ValueError("the liquid must be present")

    free = (grid == 0) | (grid == liquid)
    start = tuple(int(value) for value in liquid_cells[0])
    component = _component(free, start)

    if np.any((grid == liquid) & ~component):
        raise ValueError(
            "all liquid cells must occupy one free-space component"
        )

    return component


def _settle_liquid(
    grid: np.ndarray,
    liquid: int,
) -> np.ndarray:
    component = _liquid_component(grid, liquid)
    remaining = int(np.count_nonzero(grid == liquid))
    fill_rows: list[int] = []

    for row in range(grid.shape[0] - 1, -1, -1):
        capacity = int(np.count_nonzero(component[row]))

        if capacity == 0:
            continue

        if remaining < capacity:
            raise ValueError(
                "liquid volume would partially fill a horizontal level"
            )

        fill_rows.append(row)
        remaining -= capacity

        if remaining == 0:
            break

    if remaining != 0:
        raise ValueError(
            "liquid volume exceeds its enclosed component"
        )

    output = grid.copy()
    output[output == liquid] = 0

    selected_rows = np.isin(
        np.arange(grid.shape[0])[:, None],
        fill_rows,
    )
    output[component & selected_rows] = liquid
    return output


def _boundary_of(cavity: np.ndarray) -> np.ndarray:
    wall = np.zeros_like(cavity, dtype=bool)

    wall[1:, :] |= cavity[:-1, :]
    wall[:-1, :] |= cavity[1:, :]
    wall[:, 1:] |= cavity[:, :-1]
    wall[:, :-1] |= cavity[:, 1:]

    wall &= ~cavity
    return wall


def _paint_terrain(
    cavity: np.ndarray,
    terrain_colors: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    grid = np.zeros(cavity.shape, dtype=np.int8)
    coordinates = np.argwhere(_boundary_of(cavity))

    if len(coordinates) < len(terrain_colors):
        raise ValueError("not enough terrain cells")

    ordered_colors = np.asarray(
        terrain_colors,
        dtype=np.int8,
    )[rng.permutation(len(terrain_colors))]

    if bool(rng.integers(0, 2)):
        coordinates = coordinates[::-1]

    count = len(coordinates)

    for index, (row, column) in enumerate(coordinates):
        band = min(
            len(terrain_colors) - 1,
            index * len(terrain_colors) // count,
        )
        grid[int(row), int(column)] = ordered_colors[band]

    return grid


def _put_centered_interval(
    mask: np.ndarray,
    row: int,
    center: float,
    width: int,
    lower_column: int = 1,
    upper_column: int = 8,
) -> tuple[int, int]:
    width = int(width)
    left = int(round(float(center) - (width - 1) / 2.0))
    left = max(
        lower_column,
        min(left, upper_column - width + 1),
    )
    right = left + width - 1
    mask[int(row), left:right + 1] = True
    return left, right


def _well_cavity(
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, str]:
    cavity = np.zeros((10, 10), dtype=bool)

    top = int(rng.integers(1, 3))
    bottom = int(rng.integers(8, 10))
    row_count = bottom - top + 1

    top_width = int(rng.integers(4, 6))
    bottom_width = int(rng.integers(1, 3))

    start_center = int(rng.integers(2, 8))
    end_center = int(
        np.clip(
            start_center + int(rng.integers(-2, 3)),
            1,
            8,
        )
    )

    for index, row in enumerate(range(top, bottom + 1)):
        if index < 2:
            width = top_width
        elif index >= row_count - 2:
            width = bottom_width
        else:
            width = 3

        if index >= row_count - 2:
            center = float(end_center)
        else:
            center = (
                start_center
                + (end_center - start_center)
                * index
                / max(1, row_count - 3)
            )

        _put_centered_interval(
            cavity,
            row,
            center,
            width,
        )

    return cavity, bottom - 1, "compact"


def _basin_cavity(
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, str]:
    cavity = np.zeros((10, 10), dtype=bool)

    top = int(rng.integers(0, 2))
    bottom = int(rng.integers(8, 10))
    basin_width = int(rng.integers(6, 8))

    maximum_left = 8 - basin_width + 1
    left = int(rng.integers(1, maximum_left + 1))

    basin_center = left + (basin_width - 1) / 2.0
    upper_center = float(
        np.clip(
            basin_center + int(rng.integers(-1, 2)),
            3,
            6,
        )
    )

    row_count = bottom - top + 1

    for index, row in enumerate(range(top, bottom + 1)):
        center = (
            upper_center
            + (basin_center - upper_center)
            * index
            / max(1, row_count - 1)
        )

        if index <= 2:
            width = 5
        elif row >= bottom - 2:
            width = basin_width
        else:
            width = 3

        _put_centered_interval(
            cavity,
            row,
            center,
            width,
        )

    divider = left + int(
        rng.integers(2, basin_width - 2)
    )
    cavity[bottom - 1:bottom + 1, divider] = False

    return cavity, bottom - 1, "spread"


def _funnel_cavity(
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, str]:
    cavity = np.zeros((10, 10), dtype=bool)

    top = 1
    bottom = 9
    basin_width = 6
    left = int(rng.integers(1, 4))

    basin_center = left + 2.5
    start_center = 7.0 if basin_center < 5 else 2.0

    for index, row in enumerate(range(top, bottom + 1)):
        center = (
            start_center
            + (basin_center - start_center)
            * index
            / (bottom - top)
        )

        if index <= 1:
            width = 4
        elif row >= bottom - 2:
            width = basin_width
        else:
            width = 2 + int((row + left) % 2 == 0)

        _put_centered_interval(
            cavity,
            row,
            center,
            width,
        )

    divider = left + int(rng.integers(2, 4))
    cavity[bottom - 1:bottom + 1, divider] = False

    return cavity, bottom - 1, "vertical"


def _target_cavity(
    rng: np.random.Generator,
    variant: int,
) -> tuple[np.ndarray, int, str]:
    cavity = np.zeros((10, 10), dtype=bool)

    top_width = 3 + int(rng.integers(0, 2))
    cavity[0, 0:top_width] = True
    cavity[1, 0:4] = True

    shelf_left = 1 + int(
        (variant + int(rng.integers(0, 3))) % 3
    )
    gap = 3 + int(rng.integers(0, 2))
    well_left = min(shelf_left + gap, 8)

    for row in range(2, 7):
        fraction = (row - 2) / 4.0
        center = (
            1.5
            + ((shelf_left + 1.0) - 1.5)
            * fraction
        )
        width = 3 if row in (2, 3, 6) else 2

        _put_centered_interval(
            cavity,
            row,
            center,
            width,
            lower_column=0,
            upper_column=9,
        )

    cavity[7, shelf_left:well_left + 2] = True

    # Two-cell shelf pocket.
    cavity[8, shelf_left:shelf_left + 2] = True

    # Three-cell boundary-backed well: one cell above two bottom cells.
    cavity[8, well_left] = True
    cavity[9, well_left:well_left + 2] = True

    return cavity, 8, "corner"


def _choose_source(
    cavity: np.ndarray,
    cutoff: int,
    volume: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    all_coordinates = np.argwhere(cavity)
    top = int(all_coordinates[:, 0].min())

    if mode == "corner":
        eligible = np.argwhere(
            cavity
            & (np.arange(10)[:, None] <= 1)
        )
        mandatory = np.asarray([0, 0], dtype=int)
        remaining = eligible[
            ~np.all(eligible == mandatory, axis=1)
        ]

        indices = rng.choice(
            len(remaining),
            size=volume - 1,
            replace=False,
        )
        return np.vstack(
            [mandatory, remaining[indices]]
        )

    preferred_limit = min(
        cutoff - 1,
        top + (
            3
            if mode in ("spread", "vertical")
            else 1
        ),
    )

    eligible = np.argwhere(
        cavity
        & (
            np.arange(10)[:, None]
            <= preferred_limit
        )
    )

    if len(eligible) < volume:
        eligible = np.argwhere(
            cavity
            & (np.arange(10)[:, None] < cutoff)
        )

    if len(eligible) < volume:
        raise ValueError(
            "cavity has insufficient source space"
        )

    if mode == "compact":
        anchor = eligible[
            int(rng.integers(0, len(eligible)))
        ]
        jitter = rng.random(len(eligible)) * 0.01
        scores = (
            np.abs(eligible[:, 0] - anchor[0])
            + np.abs(eligible[:, 1] - anchor[1])
            + jitter
        )
        return eligible[np.argsort(scores)[:volume]]

    if mode == "vertical":
        anchor_column = int(
            eligible[
                int(rng.integers(0, len(eligible))),
                1,
            ]
        )
        jitter = rng.random(len(eligible)) * 0.01
        scores = (
            np.abs(eligible[:, 1] - anchor_column)
            * 2.0
            + eligible[:, 0] * 0.2
            + jitter
        )
        return eligible[np.argsort(scores)[:volume]]

    indices = rng.choice(
        len(eligible),
        size=volume,
        replace=False,
    )
    return eligible[indices]


def _pair_from_cavity(
    cavity: np.ndarray,
    cutoff: int,
    source_mode: str,
    rng: np.random.Generator,
    puzzle_parameters: dict,
) -> tuple[np.ndarray, np.ndarray]:
    liquid = puzzle_parameters["liquid"]
    terrain_colors = puzzle_parameters["terrain_colors"]

    target = (
        cavity
        & (np.arange(10)[:, None] >= cutoff)
    )
    volume = int(np.count_nonzero(target))

    source = _choose_source(
        cavity,
        cutoff,
        volume,
        source_mode,
        rng,
    )

    input_grid = _paint_terrain(
        cavity,
        terrain_colors,
        rng,
    )
    input_grid[source[:, 0], source[:, 1]] = liquid

    output_grid = _settle_liquid(
        input_grid,
        liquid,
    )

    expected = input_grid.copy()
    expected[expected == liquid] = 0
    expected[target] = liquid

    if not np.array_equal(output_grid, expected):
        raise ValueError(
            "constructed cavity did not settle as intended"
        )

    return (
        input_grid.astype(np.int8, copy=False),
        output_grid.astype(np.int8, copy=False),
    )


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    variant: int,
) -> tuple[np.ndarray, np.ndarray]:
    if regime == "source_train":
        kind = variant % 3

        if kind == 0:
            cavity, cutoff, source_mode = _well_cavity(rng)
        elif kind == 1:
            cavity, cutoff, source_mode = _basin_cavity(rng)
        else:
            cavity, cutoff, source_mode = _funnel_cavity(rng)

    elif regime == "target_test":
        cavity, cutoff, source_mode = _target_cavity(
            rng,
            variant,
        )

    else:
        raise ValueError("unknown pair regime")

    return _pair_from_cavity(
        cavity,
        cutoff,
        source_mode,
        rng,
        puzzle_parameters,
    )


def _is_integer_argument(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None and (
        not _is_integer_argument(num_train)
        or not 2 <= int(num_train) <= 10
    ):
        raise ValueError(
            "num_train must be between 2 and 10"
        )

    if num_test is not None and (
        not _is_integer_argument(num_test)
        or not 1 <= int(num_test) <= 4
    ):
        raise ValueError(
            "num_test must be between 1 and 4"
        )

    if num_train is None and num_test is None:
        train_count, test_count = _AUTO_COUNTS[
            seed % len(_AUTO_COUNTS)
        ]

    elif num_train is None:
        test_count = int(num_test)
        train_count = int(
            rng.integers(
                max(2, test_count + 1),
                11,
            )
        )

    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)

        if maximum_test < 1:
            raise ValueError(
                "no legal test count for num_train"
            )

        test_count = int(
            rng.integers(1, maximum_test + 1)
        )

    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_integer_argument(seed):
        raise ValueError("seed must be an integer")

    seed = int(seed)
    rng = np.random.default_rng(seed)

    train_count, test_count = _resolve_counts(
        seed,
        rng,
        num_train,
        num_test,
    )

    sampled_colors = rng.choice(
        np.arange(1, 10),
        size=4,
        replace=False,
    )

    puzzle_parameters = {
        "liquid": int(sampled_colors[0]),
        "terrain_colors": tuple(
            int(color)
            for color in sampled_colors[1:]
        ),
    }

    puzzle = {
        "train": [],
        "test": [],
    }

    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()

    for regime, count, split in (
        ("source_train", train_count, "train"),
        ("target_test", test_count, "test"),
    ):
        for variant in range(count):
            for _attempt in range(100):
                pair_seed = int(
                    rng.integers(
                        0,
                        2**63,
                        dtype=np.int64,
                    )
                )
                pair_rng = np.random.default_rng(
                    pair_seed
                )

                input_grid, output_grid = _generate_pair(
                    pair_rng,
                    regime,
                    puzzle_parameters,
                    variant,
                )

                input_key = (
                    input_grid.shape,
                    input_grid.tobytes(),
                )
                pair_key = (
                    input_key,
                    output_grid.tobytes(),
                )

                if (
                    input_key not in seen_inputs
                    and pair_key not in seen_pairs
                ):
                    seen_inputs.add(input_key)
                    seen_pairs.add(pair_key)

                    puzzle[split].append(
                        {
                            "input": input_grid,
                            "output": output_grid,
                        }
                    )
                    break
            else:
                raise RuntimeError(
                    "could not create a distinct pair"
                )

    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_pair_liquid(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> int:
    changed = input_grid != output_grid
    assert np.any(changed)

    changed_values = np.unique(
        np.concatenate(
            (
                input_grid[changed],
                output_grid[changed],
            )
        )
    )
    nonzero = changed_values[changed_values != 0]

    assert len(nonzero) == 1
    liquid = int(nonzero[0])

    valid_change = (
        ((input_grid == liquid) & (output_grid == 0))
        | ((input_grid == 0) & (output_grid == liquid))
    )
    assert np.all(~changed | valid_change)

    assert int(
        np.count_nonzero(input_grid == liquid)
    ) == int(
        np.count_nonzero(output_grid == liquid)
    )

    return liquid


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    liquid = _infer_pair_liquid(
        input_grid,
        output_grid,
    )

    try:
        expected = _settle_liquid(
            input_grid,
            liquid,
        )
        cavity = _liquid_component(
            input_grid,
            liquid,
        )
    except ValueError as error:
        raise AssertionError(str(error)) from error

    assert np.array_equal(output_grid, expected)

    source = np.argwhere(input_grid == liquid)
    target = np.argwhere(output_grid == liquid)

    assert len(source) > 0
    assert int(source[:, 0].max()) < int(
        target[:, 0].min()
    )
    assert (
        float(target[:, 0].mean())
        > float(source[:, 0].mean()) + 2.0
    )

    # Every settled cell is supported by liquid, terrain, or the
    # bottom canvas boundary.
    for row, column in target:
        if int(row) + 1 < output_grid.shape[0]:
            assert output_grid[
                int(row) + 1,
                int(column),
            ] != 0

    terrain = frozenset(
        int(value)
        for value in np.unique(input_grid)
        if int(value) not in (0, liquid)
    )

    assert len(terrain) == 3
    assert terrain == frozenset(
        int(value)
        for value in np.unique(output_grid)
        if int(value) not in (0, liquid)
    )

    return {
        "liquid": liquid,
        "terrain": terrain,
        "volume": len(source),
        "source": source,
        "target": target,
        "cavity": cavity,
        "output_components": _component_sizes(
            output_grid == liquid
        ),
    }


def _is_target_signature(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    analysis: dict,
) -> bool:
    del output_grid

    liquid = analysis["liquid"]
    source = analysis["source"]
    target = analysis["target"]
    cavity = analysis["cavity"]

    if (
        int(np.count_nonzero(input_grid == liquid))
        != 5
    ):
        return False

    if input_grid[0, 0] != liquid:
        return False

    if int(source[:, 0].max()) > 1:
        return False

    if analysis["output_components"] != [2, 3]:
        return False

    if set(int(row) for row in target[:, 0]) != {
        8,
        9,
    }:
        return False

    if int(np.count_nonzero(cavity[8])) != 3:
        return False

    if int(np.count_nonzero(cavity[9])) != 2:
        return False

    return True


def _assert_source_train(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    analysis: dict,
) -> None:
    liquid = analysis["liquid"]

    assert input_grid[0, 0] != liquid
    assert int(analysis["source"][:, 0].max()) <= 4
    assert int(analysis["target"][:, 0].min()) >= 6
    assert not _is_target_signature(
        input_grid,
        output_grid,
        analysis,
    )


def _assert_target_test(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    analysis: dict,
) -> None:
    assert _is_target_signature(
        input_grid,
        output_grid,
        analysis,
    )

    liquid = analysis["liquid"]
    unseen = output_grid == liquid
    components: list[np.ndarray] = []

    while np.any(unseen):
        row, column = np.argwhere(unseen)[0]
        component = _component(
            unseen,
            (int(row), int(column)),
        )
        components.append(component)
        unseen[component] = False

    shelf = next(
        component
        for component in components
        if np.count_nonzero(component) == 2
    )
    well = next(
        component
        for component in components
        if np.count_nonzero(component) == 3
    )

    shelf_cells = np.argwhere(shelf)
    assert np.all(shelf_cells[:, 0] == 8)

    # The shelf is terrain-supported rather than bottom-supported.
    assert all(
        output_grid[9, int(column)]
        not in (0, liquid)
        for _, column in shelf_cells
    )

    # The other component is a three-cell boundary well.
    assert int(np.count_nonzero(well[9])) == 2
    assert int(np.count_nonzero(well[8])) == 1

    cavity = analysis["cavity"]
    assert all(np.any(cavity[row]) for row in range(10))
    assert int(np.count_nonzero(cavity[7])) >= 5

    # The upper channel has a net rightward descent. A half-cell
    # backward centroid step is allowed when an integer-width channel
    # changes between even and odd widths.
    centers = [
        float(np.argwhere(cavity[row])[:, 0].mean())
        for row in range(7)
    ]
    assert centers[-1] >= centers[0]
    assert all(
        centers[index + 1] >= centers[index] - 0.51
        for index in range(6)
    )


def validate_generated_puzzle(
    puzzle: dict,
) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}

    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()

    analyses = {
        "train": [],
        "test": [],
    }

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert grid.dtype.kind in "iu"
                assert np.all((0 <= grid) & (grid <= 9))

            assert (
                input_grid.shape
                == output_grid.shape
                == (10, 10)
            )

            input_key = (
                input_grid.shape,
                input_grid.tobytes(),
            )
            pair_key = (
                input_key,
                output_grid.tobytes(),
            )

            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs

            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            assert all(
                not (
                    np.array_equal(
                        input_grid,
                        source_input,
                    )
                    and np.array_equal(
                        output_grid,
                        source_output,
                    )
                )
                for source_input, source_output
                in _SOURCE_PAIRS
            )

            analysis = _analyze_pair(
                input_grid,
                output_grid,
            )
            analyses[split].append(analysis)

            if split == "train":
                _assert_source_train(
                    input_grid,
                    output_grid,
                    analysis,
                )
            else:
                _assert_target_test(
                    input_grid,
                    output_grid,
                    analysis,
                )

    all_analyses = (
        analyses["train"] + analyses["test"]
    )

    liquid_colors = {
        analysis["liquid"]
        for analysis in all_analyses
    }
    terrain_palettes = {
        analysis["terrain"]
        for analysis in all_analyses
    }

    assert len(liquid_colors) == 1
    assert len(terrain_palettes) == 1

    liquid = next(iter(liquid_colors))
    terrain = next(iter(terrain_palettes))

    assert liquid not in terrain
    assert 0 not in terrain

    # Training demonstrates that volume is pair-specific rather than a
    # fixed puzzle constant.
    assert len(
        {
            analysis["volume"]
            for analysis in analyses["train"]
        }
    ) >= 2

    # Training includes both a single receiving well and a divided
    # receiving basin.
    assert any(
        analysis["output_components"]
        == [analysis["volume"]]
        for analysis in analyses["train"]
    )
    assert any(
        len(analysis["output_components"]) >= 2
        for analysis in analyses["train"]
    )

    # At least two genuinely different terrain geometries occur in
    # training.
    train_terrain_masks = {
        (
            (pair["input"] != 0)
            & (pair["input"] != liquid)
        ).tobytes()
        for pair in puzzle["train"]
    }
    assert len(train_terrain_masks) >= 2

    # Every terrain color used in tests has already been shown to be
    # stationary in every training example.
    for color in terrain:
        assert all(
            np.any(pair["input"] == color)
            and np.array_equal(
                pair["input"] == color,
                pair["output"] == color,
            )
            for pair in puzzle["train"]
        )


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first) != set(second):
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


def _puzzle_signature(puzzle: dict) -> tuple:
    pieces: list[object] = []

    for split in ("train", "test"):
        pieces.append(len(puzzle[split]))

        for pair in puzzle[split]:
            pieces.append(pair["input"].tobytes())
            pieces.append(pair["output"].tobytes())

    return tuple(pieces)


def self_test() -> None:
    automatic = generate_puzzle(17)
    validate_generated_puzzle(automatic)
    assert _puzzles_equal(
        automatic,
        generate_puzzle(17),
    )

    explicit = generate_puzzle(
        12345,
        num_train=7,
        num_test=3,
    )
    validate_generated_puzzle(explicit)

    assert _puzzles_equal(
        explicit,
        generate_puzzle(
            12345,
            num_train=7,
            num_test=3,
        ),
    )
    assert not _puzzles_equal(
        explicit,
        generate_puzzle(
            12346,
            num_train=7,
            num_test=3,
        ),
    )

    for train_count, test_count in (
        (2, 1),
        (10, 1),
        (10, 4),
    ):
        puzzle = generate_puzzle(
            300 + train_count * 10 + test_count,
            num_train=train_count,
            num_test=test_count,
        )
        validate_generated_puzzle(puzzle)
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count

    partial_cases = (
        (401, None, 1),
        (402, None, 4),
        (403, 2, None),
        (404, 10, None),
    )

    for seed, train_count, test_count in partial_cases:
        puzzle = generate_puzzle(
            seed,
            num_train=train_count,
            num_test=test_count,
        )
        validate_generated_puzzle(puzzle)

        if train_count is not None:
            assert len(puzzle["train"]) == train_count

        if test_count is not None:
            assert len(puzzle["test"]) == test_count

    illegal_cases = (
        (0, 1, None),
        (0, 11, None),
        (0, None, 0),
        (0, None, 5),
        (0, 2, 2),
        (0, 3, 4),
        (0, 4, 4),
    )

    for seed, train_count, test_count in illegal_cases:
        try:
            generate_puzzle(
                seed,
                num_train=train_count,
                num_test=test_count,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal count combination was accepted: "
                f"{train_count}, {test_count}"
            )

    puzzle_signatures: set[tuple] = set()
    layout_signatures: set[bytes] = set()
    train_counts: set[int] = set()
    test_counts: set[int] = set()
    liquid_colors: set[int] = set()
    terrain_palettes: set[frozenset[int]] = set()

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)

        puzzle_signatures.add(
            _puzzle_signature(puzzle)
        )
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))

        first_train = puzzle["train"][0]
        analysis = _analyze_pair(
            first_train["input"],
            first_train["output"],
        )

        liquid = analysis["liquid"]
        liquid_colors.add(liquid)
        terrain_palettes.add(analysis["terrain"])

        test_input = puzzle["test"][0]["input"]
        normalized_layout = np.where(
            test_input == liquid,
            2,
            np.where(test_input == 0, 0, 1),
        ).astype(np.int8)

        layout_signatures.add(
            normalized_layout.tobytes()
        )

    assert len(puzzle_signatures) >= 36
    assert len(layout_signatures) >= 6
    assert len(liquid_colors) >= 5
    assert len(terrain_palettes) >= 12
    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
