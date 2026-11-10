import numpy as np


# Exact source contents, used only to exclude copies (source sizes remain legal).
_SOURCE_PAIRS = ((('3333333333333', '3222222222223', '3213131333123', '3222222222223', '3333333333333'),
  ('3333333333333', '3222222222223', '3213131313123', '3222222222223', '3333333333333')),
 (('4444444444444444444444',
   '4222222222222222222224',
   '4233433433433433443324',
   '4233433433433433443324',
   '4222222222222222222224',
   '4444444444444444444444',
   '4222222222222222222224',
   '4214141411141414141424',
   '4241414141414141414124',
   '4222222222222222222224',
   '4444444444444444444444',
   '4222222222222222222224',
   '4284488488848884888424',
   '4288448884888488848824',
   '4222222222222222222224',
   '4444444444444444444444',
   '4222222222222222222224',
   '4294949494949494949924',
   '4294949499949494949424',
   '4222222222222222222224',
   '4444444444444444444444'),
  ('4444444444444444444444',
   '4222222222222222222224',
   '4233433433433433433424',
   '4233433433433433433424',
   '4222222222222222222224',
   '4444444444444444444444',
   '4222222222222222222224',
   '4214141414141414141424',
   '4241414141414141414124',
   '4222222222222222222224',
   '4444444444444444444444',
   '4222222222222222222224',
   '4284888488848884888424',
   '4288848884888488848824',
   '4222222222222222222224',
   '4444444444444444444444',
   '4222222222222222222224',
   '4294949494949494949424',
   '4294949494949494949424',
   '4222222222222222222224',
   '4444444444444444444444')),
 (('88888888888888888888888888888',
   '83333338333333833333383333338',
   '83888838318883834488383888838',
   '83822838381883834444383888838',
   '83822838388183838884383899838',
   '83888838388813834444383898838',
   '83822838388183834888383888838',
   '83822838381883834444383898838',
   '83888838318883838884383899838',
   '83822838381883834444383898838',
   '83822838388183834888383888838',
   '83888838388813834444383898838',
   '83822838318183838884383899838',
   '83822838381883834844383898838',
   '83888838318883834888383888838',
   '83822838381883834444383898838',
   '83822838388183838884383899838',
   '83888838388813834444383898838',
   '83822838388183834888383888838',
   '83822838381883834444383898838',
   '83888838318883838884383899838',
   '83822838381883834444383898838',
   '83822838388183834888383888838',
   '83888838388183834844383898838',
   '83882838388183838884383899838',
   '83822838381883834444383899838',
   '83888838318883834888383888838',
   '83333338333333833333383333338',
   '88888888888888888888888888888'),
  ('88888888888888888888888888888',
   '83333338333333833333383333338',
   '83888838318883834888383888838',
   '83822838381883834444383898838',
   '83822838388183838884383899838',
   '83888838388813834444383898838',
   '83822838388183834888383888838',
   '83822838381883834444383898838',
   '83888838318883838884383899838',
   '83822838381883834444383898838',
   '83822838388183834888383888838',
   '83888838388813834444383898838',
   '83822838388183838884383899838',
   '83822838381883834444383898838',
   '83888838318883834888383888838',
   '83822838381883834444383898838',
   '83822838388183838884383899838',
   '83888838388813834444383898838',
   '83822838388183834888383888838',
   '83822838381883834444383898838',
   '83888838318883838884383899838',
   '83822838381883834444383898838',
   '83822838388183834888383888838',
   '83888838388813834444383898838',
   '83822838388183838884383899838',
   '83822838381883834444383898838',
   '83888838318883834888383888838',
   '83333338333333833333383333338',
   '88888888888888888888888888888')))


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

    count_rng = np.random.default_rng(seed)
    if train is None and test is None:
        test = int(count_rng.integers(1, 5))
        train = int(count_rng.integers(test + 1, 11))
    elif train is None:
        train = int(count_rng.integers(test + 1, 11))
    elif test is None:
        test = int(count_rng.integers(1, min(4, train - 1) + 1))

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
        "noise_cap": int(rng.integers(2, 6)),
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
    # The source includes four flips in a 2-by-18 motif (over one twelfth).
    error_limit = max(1, sequence.size // 8)

    # Consider even three-copy competitors, although construction supplies four.
    for period in range(1, length // 3 + 1):
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

                # Compare all strict-majority explanations. Construction uses
                # at most one flip per phase cell, but a solver must not need
                # that hidden constraint to rule out a competing period.
                if structure_count == accent_count:
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
    # Random compositional units, followed by a finite deterministic fallback.
    # The fallback enumerates binary row words; it is not a layout bank.
    for attempt in range(128 + (1 << period) - 2):
        if attempt < 128:
            bits = rng.integers(0, 2, size=(cross_size, period), dtype=np.int8)
        else:
            word = attempt - 127
            row = np.array([(word >> i) & 1 for i in range(period)], dtype=np.int8)
            bits = np.stack([np.roll(row, i % period) for i in range(cross_size)])
        if len(np.unique(bits)) != 2:
            continue
        template = np.where(bits == 0, structure, accent).astype(np.int8)
        clean = template[:, np.arange(length) % period]
        if attempt < 128:
            noisy = _corrupt_motif(clean, period, structure, accent, rng, noise_cap)
        else:
            noisy = clean.copy()
            noisy[0, 0] = accent if clean[0, 0] == structure else structure
        try:
            repaired, recovered_period, _ = _recover_normalized(noisy, structure, accent)
        except AssertionError:
            continue
        if recovered_period == period and np.array_equal(repaired, clean):
            return noisy, clean
    raise AssertionError("finite motif construction exhausted")


def _sample_periods(
    rng: np.random.Generator,
    length: int,
    count: int,
    cap: int,
) -> list[int]:
    available = list(range(2, min(cap, length // 4) + 1))
    periods = [int(rng.choice(available)) for _ in range(count)]
    if count >= 2:
        # Demonstrate that neighboring lanes can have different units.
        periods[1] = int(rng.choice([p for p in available if p != periods[0]]))
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
            input_motif.size // 8,
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


def _orientation(grid: np.ndarray) -> str:
    structure = grid[0, 0]
    if np.count_nonzero(np.all(grid == structure, axis=1)) > 2:
        return "horizontal"
    if np.count_nonzero(np.all(grid == structure, axis=0)) > 2:
        return "vertical"
    assert grid.shape[0] != grid.shape[1], "single frame must have a long axis"
    return "horizontal" if grid.shape[1] > grid.shape[0] else "vertical"


def _orientation_plan(rng, seed, train_count, test_count):
    # Exactly one seed residue in fourteen has homogeneous splits. Cycling
    # its four regimes covers H/H, V/V, H/V, and V/H without count dependence.
    if seed % 14 == 0:
        train_axis, test_axis = (("horizontal", "horizontal"),
                                 ("vertical", "vertical"),
                                 ("horizontal", "vertical"),
                                 ("vertical", "horizontal"))[(seed // 14) % 4]
        return [train_axis] * train_count + [test_axis] * test_count
    axes = []
    for count in (train_count, test_count):
        choices = rng.integers(0, 2, size=count)
        if count >= 2 and len(set(choices)) == 1:
            choices[int(rng.integers(count))] ^= 1
        axes.extend("vertical" if value else "horizontal" for value in choices)
    return axes


def _sample_crosses(rng, count, length):
    # A bounded composition of the available cross-axis space. One structural
    # line and two padding cells per lane leave at most 29 - 3*count motif cells.
    max_cross = min(10, length - 1)
    budget = min(29 - 3 * count, count * max_cross)
    # Keep some equal-height arrays, as in the source, alongside unequal ones.
    if count > 1 and int(rng.integers(4)) == 0:
        return [int(rng.integers(1, budget // count + 1))] * count
    minimum = count + (1 if count > 1 else 0)
    total = int(rng.integers(minimum, budget + 1))
    crosses = np.ones(count, dtype=int)
    for _ in range(total - count):
        choices = np.flatnonzero(crosses < max_cross)
        crosses[int(rng.choice(choices))] += 1
    # Unequal widths whenever there is more than one lane. Retain the full
    # attainable range of overall dimensions, including 30-by-30 grids.
    if count > 1 and len(set(crosses)) == 1:
        donor, receiver = rng.choice(count, size=2, replace=False)
        if crosses[receiver] < max_cross:
            crosses[donor] -= 1
            crosses[receiver] += 1
        else:
            crosses[donor] -= 1
    rng.shuffle(crosses)
    return [int(cross) for cross in crosses]


def _generate_pair(rng, regime, parameters, serial, orientation, count, previous):
    assert regime in ("source_train", "target_test")
    structure, padding = parameters["role_pairs"][serial]
    # Both regimes admit both orientations and all frame counts. Tests favor
    # longer lanes; periods and geometry remain visible in each input.
    minimum_length = (8 if count == 1 else 12) if regime == "source_train" else 16
    lengths = list(range(minimum_length, 27))
    previous_heights = set()
    if previous is not None:
        lengths = [n for n in lengths if n != previous["length"]]
        previous_heights = ({previous["length"] + 4}
                            if previous["orientation"] == "vertical" else
                            {c + 4 for c in previous["crosses"]})
        if orientation == "vertical" and len(previous_heights) == 1:
            lengths = [n for n in lengths if n + 4 not in previous_heights]
    length = int(rng.choice(lengths))
    crosses = _sample_crosses(rng, count, length)
    if previous is not None and (
        len(set(previous["crosses"])) == 1 and set(previous["crosses"]) == set(crosses)
        or orientation == "horizontal" and len(previous_heights) == 1
        and previous_heights == {c + 4 for c in crosses}
    ):
        # The first two training pairs must expose different thicknesses even
        # if both happened to draw uniform arrays of the same thickness.
        crosses[0] += 1 if crosses[0] == 1 else -1
    periods = _sample_periods(rng, length, count, 6)
    accents = [int(c) for c in rng.choice(
        [c for c in range(10) if c not in (structure, padding)],
        size=count, replace=False)]
    pair = _build_pair(orientation, crosses, length, periods, structure,
                       padding, accents, rng, parameters["noise_cap"])
    # Only exact source contents are excluded; recoloring this extraordinarily
    # rare collision keeps the sampled size and layout legal.
    if _source_key(pair) in _SOURCE_PAIRS:
        pair = {role: np.where(grid == accents[0], accents[-1],
                              np.where(grid == accents[-1], accents[0], grid))
                for role, grid in pair.items()} if count > 1 else {
                    role: np.where(grid == accents[0],
                                   next(c for c in range(10) if c not in
                                        (structure, padding, accents[0])), grid)
                    for role, grid in pair.items()}
    return pair


def _source_key(pair):
    return tuple(tuple("".join(str(int(c)) for c in row) for row in pair[role])
                 for role in ("input", "output"))


def _predict(train_pairs, test_input):
    # Only demonstrations and the query input enter this prediction path.
    # The demonstrations establish geometric roles, binary flips, and copying.
    for pair in train_pairs:
        _analyze_pair(pair, _orientation(pair["input"]))
    axis = _orientation(test_input)
    parsed = _parse_structure(test_input, axis)
    answer = test_input.copy()
    for rectangle in parsed["rectangles"]:
        motif = test_input[rectangle]
        if axis == "vertical":
            motif = motif.T
        accent, = [int(c) for c in np.unique(motif) if c != parsed["structure"]]
        repaired, _, _ = _recover_normalized(motif, parsed["structure"], accent)
        answer[rectangle] = repaired.T if axis == "vertical" else repaired
    return answer


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

    for section in ("train", "test"):
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
            assert _source_key(pair) not in _SOURCE_PAIRS

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
                _analyze_pair(pair, _orientation(pair["input"]))
            )

    train_descriptors = descriptors["train"]
    test_descriptors = descriptors["test"]

    all_descriptors = train_descriptors + test_descriptors
    for section, descriptions in descriptors.items():
        for d in descriptions:
            assert 1 <= d["compartments"] <= 7
            assert all(1 <= c <= 10 for c in d["crosses"])
            assert (8 if d["compartments"] == 1 else 12) <= d["length"] <= 26
            if section == "test":
                assert d["length"] >= 16
            if d["compartments"] > 1:
                assert len(set(d["periods"])) >= 2

    # Count and thickness vary even for the smallest legal training set.
    assert len({d["compartments"] for d in train_descriptors}) >= 2
    assert len({c for d in train_descriptors for c in d["crosses"]}) >= 2
    assert len({d["length"] for d in train_descriptors}) >= 2
    assert len({h for d in train_descriptors for h in (
        [d["length"] + 4] if d["orientation"] == "vertical" else
        [c + 4 for c in d["crosses"]])}) >= 2
    assert any(len(set(d["periods"])) >= 2 for d in train_descriptors)
    assert len({d["structure"] for d in train_descriptors}) >= 2
    assert len({d["padding"] for d in train_descriptors}) >= 2
    roles = [(d["structure"], d["padding"]) for d in all_descriptors]
    assert len(set(roles)) == len(roles)

    train_axes = {d["orientation"] for d in train_descriptors}
    test_axes = {d["orientation"] for d in test_descriptors}
    # Minimal regimes have two homogeneous splits. Otherwise both splits mix
    # axes whenever they contain more than one example.
    if len(train_axes) > 1:
        assert len(test_axes) == min(2, len(test_descriptors))
    else:
        assert len(test_axes) == 1

    for pair in puzzle["test"]:
        predicted = _predict(puzzle["train"], pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


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

    orientations = _orientation_plan(rng, seed, train_count, test_count)
    # Shuffled cycles give all seven counts useful probability and guarantee
    # variation among the first two training pairs, without fixed first layouts.
    counts = []
    while len(counts) < train_count + test_count:
        cycle = [int(c) for c in rng.permutation(np.arange(1, 8))]
        if counts and cycle[0] == counts[-1]:
            cycle[0], cycle[1] = cycle[1], cycle[0]
        counts.extend(cycle)
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                             size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    first_geometry = None
    for serial in range(train_count + test_count):
        split = "train" if serial < train_count else "test"
        pair = _generate_pair(
            np.random.default_rng(pair_seeds[serial]),
            "source_train" if split == "train" else "target_test",
            parameters, serial, orientations[serial], counts[serial],
            first_geometry if serial == 1 else None)
        puzzle[split].append(pair)
        if serial == 0:
            first_geometry = _parse_structure(pair["input"], orientations[serial])
            first_geometry["orientation"] = orientations[serial]

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

    minimal = 0
    frame_counts, thicknesses, lengths, periods_seen = set(), set(), set(), set()
    axis_counts = {"horizontal": set(), "vertical": set()}
    partial_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_signatures.add(_puzzle_signature(puzzle))
        axes = {section: [] for section in ("train", "test")}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                d = _analyze_pair(pair, _orientation(pair["input"]))
                axes[section].append(d["orientation"])
                frame_counts.add(d["compartments"])
                axis_counts[d["orientation"]].add(d["compartments"])
                thicknesses.update(d["crosses"])
                lengths.add(d["length"])
                periods_seen.update(d["periods"])
                partial_pairs += any(d["length"] % p for p in d["periods"])
                grid_signatures.add(_grid_key(pair["input"]))
                shape_signatures.add(d["shape"])
                layout_signatures.add((d["orientation"], d["shape"], d["crosses"], d["periods"]))
                palette_signatures.add((d["structure"], d["padding"]))
        minimal += len(set(axes["train"])) == len(set(axes["test"])) == 1
    assert minimal == 3  # Measured from grids, not sampled metadata.
    assert frame_counts == set(range(1, 8))
    assert all(counts == frame_counts for counts in axis_counts.values())
    assert len(thicknesses) >= 8 and len(lengths) >= 15
    assert periods_seen == set(range(2, 7))
    assert 20 <= partial_pairs < len(grid_signatures)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) == 42
    assert len(grid_signatures) >= 120 and len(shape_signatures) >= 60
    assert len(layout_signatures) >= 100 and len(palette_signatures) >= 30

    observed_regimes = set()
    for seed in (0, 14, 28, 42):
        puzzle = generate_puzzle(seed, 4, 2)
        observed_regimes.add(tuple(tuple(sorted({_orientation(p["input"]) for p in puzzle[s]}))
                                  for s in ("train", "test")))
    assert len(observed_regimes) == 4
    # Source evidence exercises the same repair logic, without constraining sizes.
    for source_input, source_output in _SOURCE_PAIRS:
        pair = {role: np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
                for role, rows in (("input", source_input), ("output", source_output))}
        assert np.array_equal(_predict([], pair["input"]), pair["output"])

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
