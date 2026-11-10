import numpy as np


# None of these shapes can exactly duplicate a source pair.
_SOURCE_PAIR_SHAPES = {(5, 13), (21, 22), (29, 29)}


def _validate_optional_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train = _validate_optional_count("num_train", num_train, 2, 10)
    test = _validate_optional_count("num_test", num_test, 1, 4)

    if train is None and test is None:
        test = 1 + seed % 4
        minimum_train = max(2, test + 1)
        train = minimum_train + (seed // 4) % (11 - minimum_train)
    elif train is None:
        minimum_train = max(2, test + 1)
        train = minimum_train + seed % (11 - minimum_train)
    elif test is None:
        maximum_test = min(4, train - 1)
        test = 1 + seed % maximum_test

    if train <= test:
        raise ValueError("num_train must be greater than num_test")

    return int(train), int(test)


def _sample_puzzle_parameters(
    rng: np.random.Generator,
    total_pairs: int,
) -> dict:
    palette = [int(x) for x in rng.permutation(10)]
    role_shifts = [
        int(x) for x in rng.permutation(np.arange(1, 10))
    ]
    palette_start = int(rng.integers(0, 10))

    # Every pair gets a distinct ordered (structure, padding) assignment.
    # Colors are still identified geometrically inside each grid.
    role_pairs = []
    for serial in range(total_pairs):
        layer, position = divmod(serial, 10)
        structure_index = (palette_start + position) % 10
        padding_index = (
            structure_index + role_shifts[layer]
        ) % 10
        role_pairs.append(
            (
                palette[structure_index],
                palette[padding_index],
            )
        )

    return {
        "role_pairs": tuple(role_pairs),
        "noise_cap": int(rng.integers(2, 4)),
        "train_period_cap": int(rng.integers(3, 5)),
    }


def _period_candidates(
    sequence: np.ndarray,
    structure: int,
    accent: int,
) -> list[tuple[int, np.ndarray, int]]:
    cross_size, length = sequence.shape
    assert length >= 4
    assert set(int(x) for x in np.unique(sequence)) == {
        structure,
        accent,
    }

    candidates = []
    error_limit = max(1, sequence.size // 12)

    # A candidate must provide at least four observations of every phase.
    for period in range(1, length // 4 + 1):
        template = np.empty((cross_size, period), dtype=np.int8)
        error_count = 0
        supported = True

        for cross_index in range(cross_size):
            for phase in range(period):
                values = sequence[
                    cross_index,
                    phase:length:period,
                ]
                structure_count = int(
                    np.count_nonzero(values == structure)
                )
                accent_count = int(
                    np.count_nonzero(values == accent)
                )

                # Strict majority, with at most one outlier for this
                # phase and cross-sectional coordinate.
                if (
                    structure_count == accent_count
                    or min(structure_count, accent_count) > 1
                ):
                    supported = False
                    break

                if structure_count > accent_count:
                    template[cross_index, phase] = structure
                else:
                    template[cross_index, phase] = accent

                error_count += min(
                    structure_count,
                    accent_count,
                )

            if not supported:
                break

        if not supported:
            continue
        if not 1 <= error_count <= error_limit:
            continue
        if set(int(x) for x in np.unique(template)) != {
            structure,
            accent,
        }:
            continue

        repaired = template[:, np.arange(length) % period]
        candidates.append(
            (
                period,
                repaired.astype(np.int8, copy=False),
                error_count,
            )
        )

    return candidates


def _recover_normalized(
    sequence: np.ndarray,
    structure: int,
    accent: int,
) -> tuple[np.ndarray, int, tuple[int, ...]]:
    candidates = _period_candidates(
        sequence,
        structure,
        accent,
    )
    assert candidates, "motif has no strongly supported period"

    candidates.sort(key=lambda item: item[0])
    repaired = candidates[0][1]

    # A longer redundant period is acceptable only if it produces the
    # same correction as the shortest supported period.
    assert all(
        np.array_equal(repaired, candidate[1])
        for candidate in candidates
    )

    return (
        repaired,
        int(candidates[0][0]),
        tuple(int(candidate[0]) for candidate in candidates),
    )


def _corrupt_motif(
    clean: np.ndarray,
    period: int,
    structure: int,
    accent: int,
    rng: np.random.Generator,
    noise_cap: int,
) -> np.ndarray:
    cross_size, length = clean.shape
    noisy = clean.copy()

    maximum_errors = min(
        noise_cap,
        max(1, clean.size // 18),
        cross_size * period,
    )
    error_count = int(rng.integers(1, maximum_errors + 1))

    # Selecting distinct (cross coordinate, phase) groups guarantees
    # no template cell receives more than one corruption.
    groups = rng.choice(
        cross_size * period,
        size=error_count,
        replace=False,
    )

    for group in np.atleast_1d(groups):
        cross_index, phase = divmod(int(group), period)
        positions = np.arange(phase, length, period)
        position = int(
            positions[int(rng.integers(0, len(positions)))]
        )

        if noisy[cross_index, position] == structure:
            noisy[cross_index, position] = accent
        else:
            noisy[cross_index, position] = structure

    return noisy


def _sample_motif(
    rng: np.random.Generator,
    cross_size: int,
    length: int,
    period: int,
    structure: int,
    accent: int,
    noise_cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(500):
        bits = rng.integers(
            0,
            2,
            size=(cross_size, period),
            dtype=np.int8,
        )
        if len(np.unique(bits)) != 2:
            continue

        template = np.where(
            bits == 0,
            structure,
            accent,
        ).astype(np.int8)

        clean = template[:, np.arange(length) % period]
        noisy = _corrupt_motif(
            clean,
            period,
            structure,
            accent,
            rng,
            noise_cap,
        )

        try:
            repaired, recovered_period, _ = _recover_normalized(
                noisy,
                structure,
                accent,
            )
        except AssertionError:
            continue

        if (
            recovered_period == period
            and np.array_equal(repaired, clean)
        ):
            return noisy, clean

    raise RuntimeError(
        "could not sample an unambiguous periodic motif"
    )


def _sample_periods(
    rng: np.random.Generator,
    length: int,
    count: int,
    cap: int,
) -> list[int]:
    maximum = min(cap, length // 4)
    available = list(range(2, maximum + 1))
    partial = [
        period
        for period in available
        if length % period != 0
    ]

    if not partial:
        raise RuntimeError(
            "layout does not admit a partial repetition"
        )
    if count >= 2 and len(available) < 2:
        raise RuntimeError(
            "layout does not admit distinct periods"
        )

    # Every pair contains at least one partial final repetition.
    periods = [int(rng.choice(partial))]

    # Multi-compartment pairs visibly demonstrate independent periods.
    if count >= 2:
        second_choices = [
            period
            for period in available
            if period != periods[0]
        ]
        periods.append(int(rng.choice(second_choices)))

    while len(periods) < count:
        periods.append(int(rng.choice(available)))

    rng.shuffle(periods)
    return periods


def _build_pair(
    orientation: str,
    crosses: list[int],
    length: int,
    periods: list[int],
    structure: int,
    padding: int,
    accents: list[int],
    rng: np.random.Generator,
    noise_cap: int,
) -> dict:
    if orientation == "horizontal":
        height = 1 + sum(cross + 3 for cross in crosses)
        width = length + 4
    else:
        height = length + 4
        width = 1 + sum(cross + 3 for cross in crosses)

    clean_grid = np.full(
        (height, width),
        padding,
        dtype=np.int8,
    )
    noisy_grid = clean_grid.copy()

    dividers = [0]
    divider = 0
    for cross in crosses:
        divider += cross + 3
        dividers.append(divider)

    if orientation == "horizontal":
        clean_grid[:, 0] = structure
        clean_grid[:, -1] = structure
        noisy_grid[:, 0] = structure
        noisy_grid[:, -1] = structure
        clean_grid[dividers, :] = structure
        noisy_grid[dividers, :] = structure
    else:
        clean_grid[0, :] = structure
        clean_grid[-1, :] = structure
        noisy_grid[0, :] = structure
        noisy_grid[-1, :] = structure
        clean_grid[:, dividers] = structure
        noisy_grid[:, dividers] = structure

    for index, (cross, period, accent) in enumerate(
        zip(crosses, periods, accents)
    ):
        noisy_motif, clean_motif = _sample_motif(
            rng,
            cross,
            length,
            period,
            structure,
            accent,
            noise_cap,
        )

        first = dividers[index]
        last = dividers[index + 1]

        if orientation == "horizontal":
            rectangle = (
                slice(first + 2, last - 1),
                slice(2, width - 2),
            )
            noisy_grid[rectangle] = noisy_motif
            clean_grid[rectangle] = clean_motif
        else:
            rectangle = (
                slice(2, height - 2),
                slice(first + 2, last - 1),
            )
            noisy_grid[rectangle] = noisy_motif.T
            clean_grid[rectangle] = clean_motif.T

    return {
        "input": noisy_grid,
        "output": clean_grid,
    }


def _parse_structure(
    grid: np.ndarray,
    orientation: str,
) -> dict:
    height, width = grid.shape
    structure = int(grid[0, 0])

    assert np.all(grid[0, :] == structure)
    assert np.all(grid[-1, :] == structure)
    assert np.all(grid[:, 0] == structure)
    assert np.all(grid[:, -1] == structure)

    full_rows = np.flatnonzero(
        np.all(grid == structure, axis=1)
    ).tolist()
    full_columns = np.flatnonzero(
        np.all(grid == structure, axis=0)
    ).tolist()

    if orientation == "horizontal":
        assert full_columns == [0, width - 1]
        dividers = full_rows
        assert dividers[0] == 0
        assert dividers[-1] == height - 1
    else:
        assert full_rows == [0, height - 1]
        dividers = full_columns
        assert dividers[0] == 0
        assert dividers[-1] == width - 1

    assert len(dividers) >= 2
    assert np.all(np.diff(dividers) >= 4)

    structural_mask = np.zeros_like(grid, dtype=bool)
    motif_mask = np.zeros_like(grid, dtype=bool)
    rectangles = []

    if orientation == "horizontal":
        structural_mask[:, 0] = True
        structural_mask[:, -1] = True
        structural_mask[dividers, :] = True

        for first, last in zip(
            dividers[:-1],
            dividers[1:],
        ):
            rectangle = (
                slice(first + 2, last - 1),
                slice(2, width - 2),
            )
            rectangles.append(rectangle)
            motif_mask[rectangle] = True
    else:
        structural_mask[0, :] = True
        structural_mask[-1, :] = True
        structural_mask[:, dividers] = True

        for first, last in zip(
            dividers[:-1],
            dividers[1:],
        ):
            rectangle = (
                slice(2, height - 2),
                slice(first + 2, last - 1),
            )
            rectangles.append(rectangle)
            motif_mask[rectangle] = True

    assert np.all(grid[structural_mask] == structure)

    padding_mask = ~(structural_mask | motif_mask)
    padding_values = np.unique(grid[padding_mask])
    assert len(padding_values) == 1

    padding = int(padding_values[0])
    assert padding != structure

    if orientation == "horizontal":
        length = width - 4
        crosses = [
            rectangle[0].stop - rectangle[0].start
            for rectangle in rectangles
        ]
    else:
        length = height - 4
        crosses = [
            rectangle[1].stop - rectangle[1].start
            for rectangle in rectangles
        ]

    assert all(length > cross >= 1 for cross in crosses)

    return {
        "structure": structure,
        "padding": padding,
        "dividers": tuple(dividers),
        "rectangles": tuple(rectangles),
        "structural_mask": structural_mask,
        "motif_mask": motif_mask,
        "padding_mask": padding_mask,
        "length": length,
        "crosses": tuple(crosses),
    }


def _analyze_pair(
    pair: dict,
    orientation: str,
) -> dict:
    input_grid = pair["input"]
    output_grid = pair["output"]

    assert input_grid.shape == output_grid.shape
    parsed = _parse_structure(input_grid, orientation)

    # Frames, dividers, and padding must be copied unchanged.
    assert np.array_equal(
        input_grid[~parsed["motif_mask"]],
        output_grid[~parsed["motif_mask"]],
    )

    structure = parsed["structure"]
    padding = parsed["padding"]

    periods = []
    accents = []
    errors = []
    supported_periods = []

    for rectangle in parsed["rectangles"]:
        input_motif = input_grid[rectangle]
        output_motif = output_grid[rectangle]

        # Normalize both orientations to cross_size x long_length.
        if orientation == "vertical":
            input_motif = input_motif.T
            output_motif = output_motif.T

        nonstructural_values = [
            int(value)
            for value in np.unique(input_motif)
            if int(value) != structure
        ]
        assert len(nonstructural_values) == 1

        accent = nonstructural_values[0]
        assert accent != padding
        assert set(int(x) for x in np.unique(input_motif)) == {
            structure,
            accent,
        }
        assert set(int(x) for x in np.unique(output_motif)) == {
            structure,
            accent,
        }

        repaired, period, candidate_periods = (
            _recover_normalized(
                input_motif,
                structure,
                accent,
            )
        )
        assert np.array_equal(output_motif, repaired)

        difference = input_motif != output_motif
        error_count = int(np.count_nonzero(difference))
        assert 1 <= error_count <= max(
            1,
            input_motif.size // 12,
        )

        # Every changed cell must be an exact structure/accent swap.
        assert np.all(
            (
                (input_motif[difference] == structure)
                & (output_motif[difference] == accent)
            )
            | (
                (input_motif[difference] == accent)
                & (output_motif[difference] == structure)
            )
        )
        assert input_motif.shape[1] // period >= 4

        periods.append(period)
        accents.append(accent)
        errors.append(error_count)
        supported_periods.append(candidate_periods)

    assert len(set(accents)) == len(accents)
    assert not np.array_equal(input_grid, output_grid)

    return {
        "orientation": orientation,
        "shape": tuple(int(x) for x in input_grid.shape),
        "structure": structure,
        "padding": padding,
        "length": int(parsed["length"]),
        "crosses": parsed["crosses"],
        "compartments": len(parsed["rectangles"]),
        "periods": tuple(periods),
        "accents": tuple(accents),
        "errors": tuple(errors),
        "supported_periods": tuple(supported_periods),
    }


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    parameters: dict,
    serial: int,
    force_single: bool = False,
    force_multi: bool = False,
) -> dict:
    structure, padding = parameters["role_pairs"][serial]

    if regime == "source_train":
        orientation = "horizontal"
    else:
        orientation = "vertical"

    for _ in range(300):
        if regime == "source_train":
            if force_single:
                compartment_count = 1
            elif force_multi:
                compartment_count = int(rng.integers(3, 5))
            else:
                compartment_count = int(rng.integers(1, 5))

            crosses = [
                int(x)
                for x in rng.integers(
                    1,
                    4,
                    size=compartment_count,
                )
            ]
            length = int(rng.integers(14, 19))
            period_cap = parameters["train_period_cap"]
        else:
            compartment_count = int(rng.integers(2, 5))

            # Every test has unequal widths and includes a width larger
            # than anything permitted in training.
            crosses = [4, int(rng.integers(2, 4))]
            while len(crosses) < compartment_count:
                crosses.append(int(rng.integers(2, 5)))
            rng.shuffle(crosses)

            length = int(rng.integers(20, 27))
            period_cap = min(
                6,
                parameters["train_period_cap"] + 2,
            )

        if orientation == "horizontal":
            shape = (
                1 + sum(cross + 3 for cross in crosses),
                length + 4,
            )
        else:
            shape = (
                length + 4,
                1 + sum(cross + 3 for cross in crosses),
            )

        if shape in _SOURCE_PAIR_SHAPES:
            continue
        if max(shape) > 30:
            continue

        try:
            periods = _sample_periods(
                rng,
                length,
                compartment_count,
                period_cap,
            )
        except RuntimeError:
            continue

        available_accents = [
            color
            for color in range(10)
            if color not in (structure, padding)
        ]
        accents = [
            int(x)
            for x in rng.choice(
                available_accents,
                size=compartment_count,
                replace=False,
            )
        ]

        try:
            pair = _build_pair(
                orientation,
                crosses,
                length,
                periods,
                structure,
                padding,
                accents,
                rng,
                parameters["noise_cap"],
            )
            descriptor = _analyze_pair(
                pair,
                orientation,
            )
        except (AssertionError, RuntimeError):
            continue

        if descriptor["periods"] != tuple(periods):
            continue

        return pair

    raise RuntimeError("could not generate a valid pair")


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys = set()
    pair_keys = set()
    descriptors = {
        "train": [],
        "test": [],
    }

    for section, orientation in (
        ("train", "horizontal"),
        ("test", "vertical"),
    ):
        for pair in puzzle[section]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(
                    grid.dtype,
                    np.integer,
                )
                assert int(grid.min()) >= 0
                assert int(grid.max()) <= 9

            assert (
                pair["input"].shape
                == pair["output"].shape
            )
            assert (
                tuple(pair["input"].shape)
                not in _SOURCE_PAIR_SHAPES
            )

            input_key = _grid_key(pair["input"])
            pair_key = (
                input_key,
                _grid_key(pair["output"]),
            )

            assert input_key not in input_keys
            assert pair_key not in pair_keys

            input_keys.add(input_key)
            pair_keys.add(pair_key)

            descriptors[section].append(
                _analyze_pair(pair, orientation)
            )

    train_descriptors = descriptors["train"]
    test_descriptors = descriptors["test"]

    # Source-training regime: horizontal, shorter lanes, cross sizes 1–3.
    assert all(
        14 <= descriptor["length"] <= 18
        for descriptor in train_descriptors
    )
    assert all(
        1 <= descriptor["compartments"] <= 4
        for descriptor in train_descriptors
    )
    assert all(
        all(
            1 <= cross <= 3
            for cross in descriptor["crosses"]
        )
        for descriptor in train_descriptors
    )

    # The first two generated training pairs guarantee these exposure
    # conditions even at the minimum train count.
    assert any(
        descriptor["compartments"] == 1
        for descriptor in train_descriptors
    )
    assert any(
        descriptor["compartments"] >= 3
        for descriptor in train_descriptors
    )
    assert any(
        descriptor["compartments"] >= 2
        and len(set(descriptor["periods"])) >= 2
        for descriptor in train_descriptors
    )
    assert len(
        {
            descriptor["structure"]
            for descriptor in train_descriptors
        }
    ) >= 2
    assert len(
        {
            descriptor["padding"]
            for descriptor in train_descriptors
        }
    ) >= 2

    # Target-test regime: rotated vertical lanes, longer motifs, and
    # unequal cross sections containing width four.
    assert all(
        20 <= descriptor["length"] <= 26
        for descriptor in test_descriptors
    )
    assert all(
        2 <= descriptor["compartments"] <= 4
        for descriptor in test_descriptors
    )
    assert all(
        all(
            2 <= cross <= 4
            for cross in descriptor["crosses"]
        )
        for descriptor in test_descriptors
    )
    assert all(
        max(descriptor["crosses"]) == 4
        for descriptor in test_descriptors
    )
    assert all(
        len(set(descriptor["crosses"])) >= 2
        for descriptor in test_descriptors
    )
    assert all(
        len(set(descriptor["periods"])) >= 2
        for descriptor in test_descriptors
    )

    assert min(
        descriptor["length"]
        for descriptor in test_descriptors
    ) > max(
        descriptor["length"]
        for descriptor in train_descriptors
    )

    # Every pair includes a motif ending in a partial repetition.
    assert all(
        any(
            descriptor["length"] % period != 0
            for period in descriptor["periods"]
        )
        for descriptor in (
            train_descriptors + test_descriptors
        )
    )

    # Literal role assignments vary, while their geometric semantics
    # and correction rule remain invariant.
    train_roles = {
        (
            descriptor["structure"],
            descriptor["padding"],
        )
        for descriptor in train_descriptors
    }
    test_roles = {
        (
            descriptor["structure"],
            descriptor["padding"],
        )
        for descriptor in test_descriptors
    }
    all_roles = [
        (
            descriptor["structure"],
            descriptor["padding"],
        )
        for descriptor in (
            train_descriptors + test_descriptors
        )
    ]

    assert len(set(all_roles)) == len(all_roles)
    assert train_roles.isdisjoint(test_roles)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    train_count, test_count = _resolve_counts(
        seed,
        num_train,
        num_test,
    )

    rng = np.random.default_rng(seed)
    parameters = _sample_puzzle_parameters(
        rng,
        train_count + test_count,
    )

    # Each pair receives an independent deterministic generator.
    pair_seeds = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=train_count + test_count,
        dtype=np.uint64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }

    for index in range(train_count):
        pair_rng = np.random.default_rng(
            pair_seeds[index]
        )
        puzzle["train"].append(
            _generate_pair(
                pair_rng,
                "source_train",
                parameters,
                index,
                force_single=(index == 0),
                force_multi=(index == 1),
            )
        )

    for index in range(test_count):
        serial = train_count + index
        pair_rng = np.random.default_rng(
            pair_seeds[serial]
        )
        puzzle["test"].append(
            _generate_pair(
                pair_rng,
                "target_test",
                parameters,
                serial,
            )
        )

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False

        for pair_a, pair_b in zip(
            first[section],
            second[section],
        ):
            for name in ("input", "output"):
                if not np.array_equal(
                    pair_a[name],
                    pair_b[name],
                ):
                    return False

    return True


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        (
            section,
            _grid_key(pair["input"]),
            _grid_key(pair["output"]),
        )
        for section in ("train", "test")
        for pair in puzzle[section]
    )


def self_test() -> None:
    train_counts = set()
    test_counts = set()
    puzzle_signatures = set()
    grid_signatures = set()
    shape_signatures = set()
    layout_signatures = set()
    palette_signatures = set()

    # Required 42-seed diversity run.
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_signatures.add(
            _puzzle_signature(puzzle)
        )

        first_train = _analyze_pair(
            puzzle["train"][0],
            "horizontal",
        )
        palette_signatures.add(
            (
                first_train["structure"],
                first_train["padding"],
            )
        )

        for section, orientation in (
            ("train", "horizontal"),
            ("test", "vertical"),
        ):
            for pair in puzzle[section]:
                descriptor = _analyze_pair(
                    pair,
                    orientation,
                )

                grid_signatures.add(
                    _grid_key(pair["input"])
                )
                shape_signatures.add(
                    descriptor["shape"]
                )
                layout_signatures.add(
                    (
                        orientation,
                        descriptor["shape"],
                        descriptor["crosses"],
                        descriptor["periods"],
                        descriptor["errors"],
                    )
                )

                assert all(
                    len(period_choices) >= 1
                    for period_choices
                    in descriptor["supported_periods"]
                )

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40
    assert len(grid_signatures) >= 120
    assert len(shape_signatures) >= 30
    assert len(layout_signatures) >= 100
    assert len(palette_signatures) >= 15

    # Explicit counts and whole-puzzle determinism.
    requested = generate_puzzle(8001, 7, 3)
    validate_generated_puzzle(requested)

    repeated = generate_puzzle(8001, 7, 3)
    assert _puzzles_equal(requested, repeated)

    different = generate_puzzle(8002, 7, 3)
    assert not _puzzles_equal(requested, different)

    # Legal boundary cases.
    for train_count, test_count in (
        (2, 1),
        (10, 1),
        (10, 4),
    ):
        puzzle = generate_puzzle(
            9000 + train_count + test_count,
            train_count,
            test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    # Partial count specifications.
    partial_cases = (
        (9101, None, 1),
        (9102, None, 4),
        (9103, 2, None),
        (9104, 10, None),
    )
    for (
        case_seed,
        train_request,
        test_request,
    ) in partial_cases:
        puzzle = generate_puzzle(
            case_seed,
            train_request,
            test_request,
        )

        if train_request is not None:
            assert (
                len(puzzle["train"])
                == train_request
            )
        if test_request is not None:
            assert (
                len(puzzle["test"])
                == test_request
            )

        validate_generated_puzzle(puzzle)

    # Required illegal count checks.
    illegal_requests = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    )

    for request in illegal_requests:
        try:
            generate_puzzle(9200, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal request did not raise ValueError: "
                f"{request}"
            )
