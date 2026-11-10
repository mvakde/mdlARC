import numpy as np


_MIN_BANDS = 3
_MAX_BANDS = 9

# Semantic order:
# background, divider, upper-left, upper-right,
# lower-left, lower-right, interior
_SOURCE_ROLES = (0, 3, 2, 4, 1, 8, 7)

# Compartment-size descriptions of the three source training pairs
# and the source test pair.
_SOURCE_LAYOUTS = (
    ((3, 6, 4), (3, 3, 4)),
    ((4, 2, 4, 6), (3, 2, 5)),
    ((2, 4, 2, 2, 4), (3, 3, 3, 2, 1)),
    ((3, 3, 2, 4), (2, 3, 2, 5)),
)
def _checked_optional_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
        )
    return result


def _checked_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")

    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a nonnegative integer")
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _checked_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is None and test_count is None:
        # This construction guarantees broad count diversity over
        # consecutive seeds while remaining deterministic.
        test_count = 1 + seed % 4
        lower = max(2, test_count + 1)
        train_count = lower + (seed // 4) % (11 - lower)

    elif train_count is None:
        lower = max(2, test_count + 1)
        train_count = lower + seed % (11 - lower)

    elif test_count is None:
        upper = min(4, train_count - 1)
        if upper < 1:
            raise ValueError("no legal test count for num_train")
        test_count = 1 + seed % upper

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _geometry_from_sizes(
    sizes: tuple[int, ...],
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[int, ...],
    int,
]:
    bands = []
    separators = []
    position = 0

    for index, raw_size in enumerate(sizes):
        size = int(raw_size)
        assert size >= 1

        bands.append((position, position + size))
        position += size

        if index + 1 < len(sizes):
            separators.append(position)
            position += 1

    return tuple(bands), tuple(separators), position


def _make_lattice_input(
    row_sizes: tuple[int, ...],
    col_sizes: tuple[int, ...],
    background: int,
    divider: int,
) -> tuple[np.ndarray, dict]:
    row_bands, row_separators, height = _geometry_from_sizes(
        row_sizes
    )
    col_bands, col_separators, width = _geometry_from_sizes(
        col_sizes
    )

    grid = np.full(
        (height, width),
        int(background),
        dtype=np.int8,
    )
    grid[np.asarray(row_separators, dtype=int), :] = int(divider)
    grid[:, np.asarray(col_separators, dtype=int)] = int(divider)

    separator_mask = np.zeros(grid.shape, dtype=bool)
    separator_mask[
        np.asarray(row_separators, dtype=int), :
    ] = True
    separator_mask[
        :, np.asarray(col_separators, dtype=int)
    ] = True

    info = {
        "background": int(background),
        "divider": int(divider),
        "row_bands": row_bands,
        "col_bands": col_bands,
        "row_separators": row_separators,
        "col_separators": col_separators,
        "separator_mask": separator_mask,
        "row_sizes": tuple(int(x) for x in row_sizes),
        "col_sizes": tuple(int(x) for x in col_sizes),
        "lattice": (len(row_sizes), len(col_sizes)),
    }
    return grid, info


def _apply_rule(
    input_grid: np.ndarray,
    info: dict,
    roles: tuple[int, ...],
) -> np.ndarray:
    (
        background,
        divider,
        upper_left,
        upper_right,
        lower_left,
        lower_right,
        interior,
    ) = (int(x) for x in roles)

    assert background == info["background"]
    assert divider == info["divider"]

    output = np.array(input_grid, copy=True)
    last_row = len(info["row_bands"]) - 1
    last_col = len(info["col_bands"]) - 1

    for row_index, (row_start, row_stop) in enumerate(
        info["row_bands"]
    ):
        for col_index, (col_start, col_stop) in enumerate(
            info["col_bands"]
        ):
            color = background

            if row_index == 0 and col_index == 0:
                color = upper_left
            elif row_index == 0 and col_index == last_col:
                color = upper_right
            elif row_index == last_row and col_index == 0:
                color = lower_left
            elif (
                row_index == last_row
                and col_index == last_col
            ):
                color = lower_right
            elif (
                0 < row_index < last_row
                and 0 < col_index < last_col
            ):
                color = interior

            output[
                row_start:row_stop,
                col_start:col_stop,
            ] = color

    return output


def _construct_pair(
    row_sizes: tuple[int, ...],
    col_sizes: tuple[int, ...],
    roles: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    input_grid, info = _make_lattice_input(
        row_sizes,
        col_sizes,
        roles[0],
        roles[1],
    )
    output_grid = _apply_rule(input_grid, info, roles)
    return input_grid, output_grid


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return (
        _grid_key(input_grid),
        _grid_key(output_grid),
    )


def _is_exact_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for row_sizes, col_sizes in _SOURCE_LAYOUTS:
        source_input, source_output = _construct_pair(
            row_sizes,
            col_sizes,
            _SOURCE_ROLES,
        )
        if (
            np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _sample_band_sizes(
    rng: np.random.Generator,
    count: int,
    regime: str,
    force_equal: bool = False,
) -> tuple[int, ...]:
    """Compose nonempty bands under the 30-cell canvas limit."""
    assert _MIN_BANDS <= count <= _MAX_BANDS
    budget = 30 - (count - 1)
    mode = 1 if force_equal else int(rng.choice(6, p=(.40, .12, .12, .10, .12, .14)))

    if mode == 1:  # Regular, equal-width bands.
        size = int(rng.integers(1, budget // count + 1))
        sizes = [size] * count
    elif mode == 2:  # Mirror symmetry, including unequal mirrors.
        sizes = [1] * count
        for index in range((count + 1) // 2):
            partner = count - 1 - index
            multiplicity = 1 if index == partner else 2
            available = budget - sum(sizes)
            increment = int(rng.integers(0, min(5, available // multiplicity) + 1))
            sizes[index] += increment
            if partner != index:
                sizes[partner] += increment
    elif mode == 3:  # Repeated alternating sizes.
        options = [
            (a, b)
            for a in range(1, 8)
            for b in range(1, 8)
            if sum(a if index % 2 == 0 else b for index in range(count)) <= budget
        ]
        a, b = options[int(rng.integers(len(options)))]
        sizes = [a if index % 2 == 0 else b for index in range(count)]
    elif mode == 4:  # No thin compartments.
        minimum = 2 * count
        total = int(rng.integers(minimum, budget + 1))
        weights = rng.dirichlet(np.ones(count))
        sizes = list(2 + rng.multinomial(total - minimum, weights))
    elif mode == 5:  # A large, strictly interior compartment.
        sizes = [1] * count
        index = int(rng.integers(1, count - 1))
        sizes[index] = int(rng.integers(6, min(14, budget - count + 1) + 1))
        for other in rng.permutation(count):
            if other != index:
                sizes[int(other)] += int(rng.integers(0, min(2, budget - sum(sizes)) + 1))
    else:  # Free composition, sometimes using nearly all 30 cells.
        minimum = count
        if regime == "target_test" and rng.random() < .35:
            low_total = max(minimum, budget - 4)
        else:
            low_total = minimum
        total = int(rng.integers(low_total, budget + 1))
        weights = rng.dirichlet(np.ones(count))
        sizes = list(1 + rng.multinomial(total - minimum, weights))

    result = tuple(int(size) for size in sizes)
    assert len(result) == count
    assert min(result) >= 1
    assert sum(result) <= budget
    return result


def _sample_lattice(rng: np.random.Generator) -> tuple[int, int]:
    return (
        int(rng.integers(_MIN_BANDS, _MAX_BANDS + 1)),
        int(rng.integers(_MIN_BANDS, _MAX_BANDS + 1)),
    )


def _test_lattice(
    rng: np.random.Generator,
    train_lattices: list[tuple[int, int]],
) -> tuple[int, int]:
    mode = int(rng.integers(5))
    if mode == 0:
        return (3, 3)  # Sometimes simpler than all demonstrations.
    if mode == 1:
        return (
            train_lattices[int(rng.integers(len(train_lattices)))][0],
            train_lattices[int(rng.integers(len(train_lattices)))][1],
        )
    if mode in (2, 3):
        rows, cols = _sample_lattice(rng)
        if mode == 2:
            rows = train_lattices[int(rng.integers(len(train_lattices)))][0]
        else:
            cols = train_lattices[int(rng.integers(len(train_lattices)))][1]
        return rows, cols
    return _sample_lattice(rng)


def _generate_unique_pair(
    rng: np.random.Generator,
    regime: str,
    lattice: tuple[int, int],
    roles: tuple[int, ...],
    used_input_keys: set,
    layout: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
) -> dict:
    row_count, col_count = lattice

    for attempt in range(128):
        if layout is None or attempt:
            regular = rng.random() < .06
            row_sizes = _sample_band_sizes(rng, row_count, regime, regular)
            col_sizes = _sample_band_sizes(rng, col_count, regime, regular)
        else:
            row_sizes, col_sizes = layout

        input_grid, output_grid = _construct_pair(
            row_sizes,
            col_sizes,
            roles,
        )
        input_key = _grid_key(input_grid)

        if input_key in used_input_keys:
            continue
        if _is_exact_source_pair(input_grid, output_grid):
            continue

        used_input_keys.add(input_key)
        return {
            "input": input_grid,
            "output": output_grid,
        }

    # Guaranteed bounded fallback for extreme count requests or collisions.
    def variants(count: int) -> list[tuple[int, ...]]:
        base = (1,) * count
        result = [base]
        for extra in range(1, 4):
            for index in range(count):
                result.append(tuple(1 + extra if i == index else 1 for i in range(count)))
        return result

    for row_sizes in variants(row_count):
        for col_sizes in variants(col_count):
            input_grid, output_grid = _construct_pair(row_sizes, col_sizes, roles)
            input_key = _grid_key(input_grid)
            if input_key in used_input_keys or _is_exact_source_pair(input_grid, output_grid):
                continue
            used_input_keys.add(input_key)
            return {"input": input_grid, "output": output_grid}
    raise ValueError("could not generate enough distinct legal pairs")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _checked_seed(seed)
    rng = np.random.default_rng(seed_value)

    train_count, test_count = _resolve_counts(
        seed_value,
        num_train,
        num_test,
    )

    # Only the five fill roles must be learned from the training outputs.
    # Background and divider colors are visible in each local input.
    palette = tuple(int(x) for x in rng.permutation(10))
    fill_roles = palette[:5]
    available = palette[5:]
    first_local = available[:2]
    varied_local = bool(rng.integers(2))
    local_options = [(a, b) for a in available for b in available if a != b]

    first_rows = int(rng.integers(3, 8))
    first_cols = int(rng.integers(3, 8))
    if first_cols == first_rows:
        first_cols = 3 + (first_cols - 2) % 5
    first_lattice = (first_rows, first_cols)
    train_schedule = [first_lattice, (first_cols, first_rows)]
    train_schedule.extend(_sample_lattice(rng) for _ in range(train_count - 2))

    test_schedule = []
    for _ in range(test_count):
        candidate = _test_lattice(rng, train_schedule)
        while candidate in test_schedule:
            # There are 49 legal count combinations and at most four tests.
            candidate = ((candidate[0] - 2) % 7 + 3, candidate[1])
        test_schedule.append(candidate)

    complete_schedule = (
        [("source_train", lattice) for lattice in train_schedule]
        + [("target_test", lattice) for lattice in test_schedule]
    )

    # Give every pair an independent deterministic random stream.
    child_seeds = rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=len(complete_schedule),
        dtype=np.int64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }
    used_input_keys = set()

    first_layout = None
    for index, (
        regime,
        lattice,
    ), child_seed in zip(
        range(len(complete_schedule)),
        complete_schedule,
        child_seeds,
    ):
        pair_rng = np.random.default_rng(
            int(child_seed)
        )
        if index == 0:
            local = first_local
        elif index == 1 and varied_local:
            choices = [option for option in local_options if option != first_local]
            local = choices[int(pair_rng.integers(len(choices)))]
        elif varied_local:
            local = local_options[int(pair_rng.integers(len(local_options)))]
        else:
            local = first_local
        roles = (*local, *fill_roles)

        layout = None
        if index == 1 and first_layout is not None:
            row_sizes, col_sizes = first_layout
            new_rows = tuple(reversed(col_sizes)) if pair_rng.random() < .5 else col_sizes
            new_cols = tuple(reversed(row_sizes)) if pair_rng.random() < .5 else row_sizes
            layout = (new_rows, new_cols)
        pair = _generate_unique_pair(
            pair_rng,
            regime,
            lattice,
            roles,
            used_input_keys,
            layout,
        )
        if index == 0:
            info = _analyze_input(pair["input"])
            first_layout = (info["row_sizes"], info["col_sizes"])

        split = (
            "train"
            if regime == "source_train"
            else "test"
        )
        puzzle[split].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _bands_from_separators(
    length: int,
    separators: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    separators = tuple(
        int(x) for x in separators
    )

    assert len(separators) >= 2
    assert all(
        0 < value < length - 1
        for value in separators
    )

    bands = []
    start = 0

    for separator in separators:
        # This also rejects adjacent divider lines because
        # they would create an empty compartment.
        assert separator > start
        bands.append((start, separator))
        start = separator + 1

    assert start < length
    bands.append((start, length))
    return tuple(bands)


def _analyze_input(grid: np.ndarray) -> dict:
    colors = tuple(
        int(x) for x in np.unique(grid)
    )
    assert len(colors) == 2

    candidates = []
    for color in colors:
        rows = tuple(
            int(x)
            for x in np.flatnonzero(
                np.all(grid == color, axis=1)
            )
        )
        cols = tuple(
            int(x)
            for x in np.flatnonzero(
                np.all(grid == color, axis=0)
            )
        )

        if len(rows) >= 2 and len(cols) >= 2:
            candidates.append(
                (color, rows, cols)
            )

    # Exactly one input color must form both complete
    # horizontal and complete vertical divider lines.
    assert len(candidates) == 1
    (
        divider,
        row_separators,
        col_separators,
    ) = candidates[0]

    background = (
        colors[0]
        if colors[1] == divider
        else colors[1]
    )

    row_bands = _bands_from_separators(
        grid.shape[0],
        row_separators,
    )
    col_bands = _bands_from_separators(
        grid.shape[1],
        col_separators,
    )

    assert len(row_bands) >= 3
    assert len(col_bands) >= 3

    separator_mask = np.zeros(
        grid.shape,
        dtype=bool,
    )
    separator_mask[
        np.asarray(row_separators, dtype=int), :
    ] = True
    separator_mask[
        :, np.asarray(col_separators, dtype=int)
    ] = True

    # Proves that there are no partial dividers, noise,
    # extra objects, or incorrectly colored cells.
    assert np.all(
        grid[separator_mask] == divider
    )
    assert np.all(
        grid[~separator_mask] == background
    )

    row_sizes = tuple(
        stop - start
        for start, stop in row_bands
    )
    col_sizes = tuple(
        stop - start
        for start, stop in col_bands
    )

    return {
        "background": int(background),
        "divider": int(divider),
        "row_bands": row_bands,
        "col_bands": col_bands,
        "row_separators": row_separators,
        "col_separators": col_separators,
        "separator_mask": separator_mask,
        "row_sizes": row_sizes,
        "col_sizes": col_sizes,
        "lattice": (
            len(row_bands),
            len(col_bands),
        ),
    }


def _derive_roles(
    info: dict,
    output_grid: np.ndarray,
) -> tuple[int, ...]:
    assert (
        output_grid.shape
        == info["separator_mask"].shape
    )

    def component_value(
        row_index: int,
        col_index: int,
    ) -> int:
        row_start, row_stop = (
            info["row_bands"][row_index]
        )
        col_start, col_stop = (
            info["col_bands"][col_index]
        )
        block = output_grid[
            row_start:row_stop,
            col_start:col_stop,
        ]

        value = int(block[0, 0])
        assert np.all(block == value)
        return value

    roles = (
        info["background"],
        info["divider"],
        component_value(0, 0),
        component_value(0, -1),
        component_value(-1, 0),
        component_value(-1, -1),
        component_value(1, 1),
    )

    # All seven semantic roles must have distinct colors.
    assert len(set(roles)) == 7
    return roles


def _assert_grid_valid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30
    assert 1 <= grid.shape[1] <= 30
    assert np.issubdtype(
        grid.dtype,
        np.integer,
    )
    assert int(np.min(grid)) >= 0
    assert int(np.max(grid)) <= 9


def validate_generated_puzzle(
    puzzle: dict,
) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {
        "train",
        "test",
    }
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert (
        len(puzzle["train"])
        > len(puzzle["test"])
    )

    records = []
    input_keys = set()
    pair_keys = set()

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {
                "input",
                "output",
            }

            input_grid = pair["input"]
            output_grid = pair["output"]

            _assert_grid_valid(input_grid)
            _assert_grid_valid(output_grid)

            input_key = _grid_key(input_grid)
            pair_key = _pair_key(
                input_grid,
                output_grid,
            )

            assert input_key not in input_keys
            assert pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)

            info = _analyze_input(input_grid)

            assert all(size >= 1 for size in info["row_sizes"])
            assert all(size >= 1 for size in info["col_sizes"])
            assert all(
                _MIN_BANDS <= count <= _MAX_BANDS
                for count in info["lattice"]
            )
            assert not _is_exact_source_pair(
                input_grid,
                output_grid,
            )

            records.append(
                (split, pair, info)
            )

    train_records = [
        record
        for record in records
        if record[0] == "train"
    ]
    test_records = [
        record
        for record in records
        if record[0] == "test"
    ]

    train_lattices = [
        record[2]["lattice"]
        for record in train_records
    ]

    assert train_lattices[0][0] != train_lattices[0][1]
    assert train_lattices[1] == train_lattices[0][::-1]
    test_lattices = [record[2]["lattice"] for record in test_records]
    assert len(set(test_lattices)) == len(test_lattices)

    # Learn the complete puzzle-level convention from a
    # returned training output.
    learned_roles = _derive_roles(
        train_records[0][2],
        train_records[0][1]["output"],
    )
    fill_roles = learned_roles[2:]
    train_locals = {
        (record[2]["background"], record[2]["divider"])
        for record in train_records
    }
    all_locals = {
        (record[2]["background"], record[2]["divider"])
        for record in records
    }
    if len(all_locals) > 1:
        assert len(train_locals) > 1

    for _, pair, info in records:
        local_roles = (info["background"], info["divider"], *fill_roles)
        observed_roles = _derive_roles(
            info,
            pair["output"],
        )
        assert observed_roles == local_roles

        expected = _apply_rule(
            pair["input"],
            info,
            local_roles,
        )
        assert np.array_equal(
            pair["output"],
            expected,
        )
        assert not np.array_equal(
            pair["input"],
            pair["output"],
        )

        assert {
            int(x)
            for x in np.unique(pair["output"])
        } == set(local_roles)

    # Every training output explicitly demonstrates all five
    # shared fill roles needed for solving any test layout.
    for _, pair, info in train_records:
        visible_colors = {
            int(x)
            for x in np.unique(pair["output"])
        }
        assert set(fill_roles).issubset(visible_colors)
        assert {info["background"], info["divider"]}.issubset(visible_colors)


def _puzzles_equal(
    left: dict,
    right: dict,
) -> bool:
    if not isinstance(left, dict):
        return False
    if not isinstance(right, dict):
        return False

    if set(left.keys()) != {"train", "test"}:
        return False
    if set(right.keys()) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(left[split]) != len(right[split]):
            return False

        for left_pair, right_pair in zip(
            left[split],
            right[split],
        ):
            if not np.array_equal(
                left_pair["input"],
                right_pair["input"],
            ):
                return False
            if not np.array_equal(
                left_pair["output"],
                right_pair["output"],
            ):
                return False

    return True


def _puzzle_signature(
    puzzle: dict,
) -> tuple:
    return tuple(
        (
            split,
            _pair_key(
                pair["input"],
                pair["output"],
            ),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def _self_check_task_rule(
    puzzle: dict,
) -> tuple[int, ...]:
    first_pair = puzzle["train"][0]
    first_info = _analyze_input(
        first_pair["input"]
    )
    roles = _derive_roles(
        first_info,
        first_pair["output"],
    )[2:]
    assert len(set(roles)) == 5

    for split in ("train", "test"):
        for pair in puzzle[split]:
            info = _analyze_input(pair["input"])
            local = (info["background"], info["divider"], *roles)

            expected = _apply_rule(
                pair["input"],
                info,
                local,
            )
            assert np.array_equal(
                expected,
                pair["output"],
            )
            assert _derive_roles(
                info,
                pair["output"],
            ) == local

            # The input alone identifies both copied colors, while
            # the demonstrations identify every fill color.
            assert np.all(
                pair["output"][info["separator_mask"]]
                == info["divider"]
            )

            assert not _is_exact_source_pair(
                pair["input"],
                pair["output"],
            )

    return roles


def self_test() -> None:
    # Automatic count generation.
    for seed in (0, 7, 19, 41):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _self_check_task_rule(puzzle)

    # Explicit counts, including all required boundaries.
    explicit_cases = (
        (2, 1),
        (10, 1),
        (10, 4),
        (7, 3),
    )
    for index, (
        train_count,
        test_count,
    ) in enumerate(explicit_cases):
        puzzle = generate_puzzle(
            100 + index,
            num_train=train_count,
            num_test=test_count,
        )

        assert (
            len(puzzle["train"])
            == train_count
        )
        assert (
            len(puzzle["test"])
            == test_count
        )

        validate_generated_puzzle(puzzle)
        _self_check_task_rule(puzzle)

    # Partial count specifications.
    partial_cases = (
        (211, None, 1),
        (212, None, 4),
        (213, 2, None),
        (214, 10, None),
    )
    for (
        seed,
        requested_train,
        requested_test,
    ) in partial_cases:
        puzzle = generate_puzzle(
            seed,
            num_train=requested_train,
            num_test=requested_test,
        )

        if requested_train is not None:
            assert (
                len(puzzle["train"])
                == requested_train
            )
        if requested_test is not None:
            assert (
                len(puzzle["test"])
                == requested_test
            )

        validate_generated_puzzle(puzzle)
        _self_check_task_rule(puzzle)

    # Whole-puzzle determinism.
    deterministic_a = generate_puzzle(
        12345,
        num_train=7,
        num_test=3,
    )
    deterministic_b = generate_puzzle(
        12345,
        num_train=7,
        num_test=3,
    )
    assert _puzzles_equal(
        deterministic_a,
        deterministic_b,
    )

    # Different seeds should nearly always produce different
    # complete puzzles.
    different_seed_signatures = {
        _puzzle_signature(
            generate_puzzle(
                3000 + index,
                num_train=6,
                num_test=2,
            )
        )
        for index in range(6)
    }
    assert len(different_seed_signatures) >= 5

    # Required illegal count cases.
    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    )

    for kwargs in illegal_cases:
        raised = False
        try:
            generate_puzzle(99, **kwargs)
        except ValueError:
            raised = True
        assert raised

    # Measure observed grids, not sampler internals, across 42 seeds.
    train_counts, test_counts = set(), set()
    fill_assignments, input_keys, layouts, dimensions = set(), set(), set(), set()
    train_lattices, test_lattices = set(), set()
    requested_lattices = set()
    equal_both = min_two = strong = near_full_test = 0
    rectangular = simpler_test = varied_local = stable_local = 0
    shifted_rows = shifted_cols = shifted_both = familiar = 0
    combined_min_two_large = 0
    multi_test_lattices = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fill_assignments.add(_self_check_task_rule(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        test_info = [_analyze_input(pair["input"]) for pair in puzzle["test"]]
        train_rows = {info["lattice"][0] for info in train_info}
        train_cols = {info["lattice"][1] for info in train_info}
        if len({info["lattice"] for info in test_info}) > 1:
            multi_test_lattices += 1
        if any(
            a * b < min(x * y for x, y in (info["lattice"] for info in train_info))
            for a, b in (info["lattice"] for info in test_info)
        ):
            simpler_test += 1
        local_colors = {
            (info["background"], info["divider"])
            for info in train_info + test_info
        }
        if len(local_colors) > 1:
            varied_local += 1
        else:
            stable_local += 1

        for split, pairs, infos in (
            ("train", puzzle["train"], train_info),
            ("test", puzzle["test"], test_info),
        ):
            for pair, info in zip(pairs, infos):
                row_sizes, col_sizes = info["row_sizes"], info["col_sizes"]
                rows, cols = info["lattice"]
                input_keys.add(_grid_key(pair["input"]))
                layouts.add((row_sizes, col_sizes))
                dimensions.add(tuple(int(x) for x in pair["input"].shape))
                (train_lattices if split == "train" else test_lattices).add((rows, cols))
                if (rows, cols) in ((3, 4), (5, 3), (3, 6), (6, 3)):
                    requested_lattices.add((rows, cols))
                if rows != cols:
                    rectangular += 1
                if len(set(row_sizes)) == len(set(col_sizes)) == 1:
                    equal_both += 1
                has_min_two = min(row_sizes + col_sizes) >= 2
                if has_min_two:
                    min_two += 1
                if any(
                    value >= 6 and value >= max(sizes[0], sizes[-1]) + 3
                    for sizes in (row_sizes, col_sizes)
                    for value in sizes[1:-1]
                ):
                    strong += 1
                is_large = max(pair["input"].shape) >= 28
                if split == "test":
                    near_full_test += int(is_large)
                    shifted_rows += int(rows not in train_rows and cols in train_cols)
                    shifted_cols += int(rows in train_rows and cols not in train_cols)
                    shifted_both += int(rows not in train_rows and cols not in train_cols)
                    familiar += int(rows in train_rows and cols in train_cols)
                if has_min_two and is_large:
                    combined_min_two_large += 1

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fill_assignments) >= 35
    assert len(input_keys) >= 300 and len(layouts) >= 250
    assert len(dimensions) >= 100
    assert len(train_lattices) >= 30 and len(test_lattices) >= 25
    assert requested_lattices == {(3, 4), (5, 3), (3, 6), (6, 3)}
    assert rectangular >= 150
    assert equal_both >= 5 and min_two >= 30 and strong >= 40
    assert near_full_test >= 10 and combined_min_two_large >= 10
    assert simpler_test >= 5 and multi_test_lattices >= 15
    assert varied_local >= 10 and stable_local >= 10
    assert min(shifted_rows, shifted_cols, shifted_both, familiar) >= 3
