import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _grid((
            "888888888",
            "818188888",
            "888188811",
            "888888888",
            "888888888",
            "222222222",
            "888888888",
            "888888888",
            "888888888",
        )),
        _grid((
            "888888888",
            "888888888",
            "888888888",
            "888888888",
            "888888811",
            "282822222",
            "888888888",
            "888188888",
            "818188888",
        )),
    ),
    (
        _grid((
            "8818888",
            "8888888",
            "8888888",
            "2222222",
            "8888888",
            "8888888",
            "8888888",
        )),
        _grid((
            "8888888",
            "8888888",
            "8888888",
            "2282222",
            "8888888",
            "8888888",
            "8818888",
        )),
    ),
    (
        _grid((
            "88888888818",
            "81118881888",
            "81888881888",
            "81888881888",
            "88888888888",
            "88888888888",
            "22222222222",
            "88888888888",
            "88888888888",
            "88888888888",
            "88888888888",
        )),
        _grid((
            "88888888888",
            "88888888888",
            "88888888888",
            "88888888888",
            "88888888888",
            "81118888888",
            "21222228282",
            "81888888888",
            "88888881888",
            "88888881888",
            "88888881818",
        )),
    ),
    (
        _grid((
            "888888888888881888",
            "888188111888881888",
            "888188818881881881",
            "888188818811881888",
            "888188818881881888",
            "881118818881881888",
            "888888888888881888",
            "888888888888888888",
            "222222222222222222",
            "888888888888888888",
            "888888888888888888",
            "888888888888888888",
            "888888888888888888",
            "888888888888888888",
            "888888888888888888",
            "888888888888888888",
            "888888888888888888",
            "888888888888888888",
        )),
        _grid((
            "888888888888888888",
            "888888888888888888",
            "888888888888888888",
            "888188888888888888",
            "888188888888888888",
            "888188888888888888",
            "888188888881888888",
            "881118111811888888",
            "222222212221228228",
            "888888818881888888",
            "888888818888888888",
            "888888818888881888",
            "888888888888881888",
            "888888888888881888",
            "888888888888881888",
            "888888888888881888",
            "888888888888881888",
            "888888888888881881",
        )),
    ),
)


def _grid_key(grid: np.ndarray) -> tuple:
    canonical = np.asarray(grid, dtype=np.int8)
    return tuple(canonical.shape), canonical.tobytes()


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return any(
        input_grid.shape == source_input.shape
        and output_grid.shape == source_output.shape
        and np.array_equal(input_grid, source_input)
        and np.array_equal(output_grid, source_output)
        for source_input, source_output in _SOURCE_PAIRS
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []

    for row in range(height):
        for column in range(width):
            if not mask[row, column] or seen[row, column]:
                continue

            stack = [(row, column)]
            seen[row, column] = True
            cells = []

            while stack:
                current_row, current_column = stack.pop()
                cells.append((current_row, current_column))

                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    if 0 <= next_row < height and 0 <= next_column < width:
                        if (
                            mask[next_row, next_column]
                            and not seen[next_row, next_column]
                        ):
                            seen[next_row, next_column] = True
                            stack.append((next_row, next_column))

            result.append(np.asarray(cells, dtype=np.int16))

    return result


def _analyze_input(grid: np.ndarray) -> dict:
    values, counts = np.unique(grid, return_counts=True)
    _require(
        len(values) == 3,
        "each input must contain exactly three role colors",
    )

    background = int(values[int(np.argmax(counts))])
    divider_candidates = []

    for row in range(grid.shape[0]):
        color = int(grid[row, 0])
        if color != background and np.all(grid[row, :] == color):
            divider_candidates.append((0, row, color))

    for column in range(grid.shape[1]):
        color = int(grid[0, column])
        if color != background and np.all(grid[:, column] == color):
            divider_candidates.append((1, column, color))

    _require(
        len(divider_candidates) == 1,
        "input must have one complete non-background divider",
    )
    axis, divider, membrane = divider_candidates[0]

    payload_values = [
        int(value)
        for value in values
        if int(value) not in (background, membrane)
    ]
    _require(len(payload_values) == 1, "payload color must be unique")
    payload = payload_values[0]

    expected_membrane_cells = grid.shape[1 - axis]
    _require(
        int(np.count_nonzero(grid == membrane))
        == expected_membrane_cells,
        "membrane color may occur only on the complete divider",
    )

    payload_cells = np.argwhere(grid == payload)
    _require(len(payload_cells) > 0, "input must contain payload")

    axial_coordinates = payload_cells[:, axis]
    if np.all(axial_coordinates < divider):
        sign = 1
    elif np.all(axial_coordinates > divider):
        sign = -1
    else:
        raise AssertionError(
            "all payload must occupy exactly one side of the divider"
        )

    return {
        "background": background,
        "membrane": membrane,
        "payload": payload,
        "axis": int(axis),
        "divider": int(divider),
        "sign": int(sign),
    }


def _classify_component(
    component: np.ndarray,
    axis: int,
    sign: int,
) -> dict:
    levels = sorted(
        (int(value) for value in np.unique(component[:, axis])),
        reverse=(sign > 0),
    )

    expected_levels = list(
        range(levels[0], levels[-1] - sign, -sign)
    )
    _require(
        levels == expected_levels,
        "a connected payload must occupy consecutive axial slices",
    )

    widths = tuple(
        int(np.count_nonzero(component[:, axis] == level))
        for level in levels
    )
    first_wide = next(
        (
            index
            for index, width in enumerate(widths)
            if width >= 2
        ),
        None,
    )

    if first_wide is None:
        outcome = "pass"
    elif first_wide == 0:
        outcome = "blocked"
    else:
        outcome = "partial"

    return {
        "slice_widths": widths,
        "first_wide": first_wide,
        "outcome": outcome,
        "axial_length": len(levels),
    }


def _simulate_component(
    component: np.ndarray,
    info: dict,
    shape: tuple[int, int],
) -> dict:
    axis = info["axis"]
    sign = info["sign"]
    divider = info["divider"]

    delta = np.zeros(2, dtype=np.int16)
    delta[axis] = sign

    current = component.copy()
    footprint = {
        tuple(int(value) for value in cell)
        for cell in current
    }
    punctures: set[tuple[int, int]] = set()
    stop_reason = None

    for _ in range(shape[axis] + 2):
        candidate = current + delta

        inside = (
            np.all(candidate[:, 0] >= 0)
            and np.all(candidate[:, 0] < shape[0])
            and np.all(candidate[:, 1] >= 0)
            and np.all(candidate[:, 1] < shape[1])
        )
        if not inside:
            stop_reason = "boundary"
            break

        membrane_load = int(
            np.count_nonzero(candidate[:, axis] == divider)
        )
        if membrane_load >= 2:
            stop_reason = "membrane"
            break

        current = candidate
        footprint.update(
            tuple(int(value) for value in cell)
            for cell in current
        )

        for cell in current[current[:, axis] == divider]:
            punctures.add(
                tuple(int(value) for value in cell)
            )

    _require(
        stop_reason is not None,
        "component simulation did not terminate",
    )

    return {
        "final": current,
        "footprint": footprint,
        "punctures": punctures,
        "stop_reason": stop_reason,
    }


def _solve_and_describe(
    input_grid: np.ndarray,
) -> tuple[np.ndarray, dict, list[dict]]:
    info = _analyze_input(input_grid)
    components = _components(
        input_grid == info["payload"]
    )
    _require(len(components) > 0, "payload segmentation failed")

    occupied_trajectories: set[tuple[int, int]] = set()
    all_punctures: set[tuple[int, int]] = set()
    final_cells: set[tuple[int, int]] = set()

    details = []
    simulations = []

    for component in components:
        classification = _classify_component(
            component,
            info["axis"],
            info["sign"],
        )
        simulation = _simulate_component(
            component,
            info,
            input_grid.shape,
        )

        _require(
            occupied_trajectories.isdisjoint(
                simulation["footprint"]
            ),
            "different objects may not have intersecting paths",
        )
        occupied_trajectories.update(
            simulation["footprint"]
        )

        final_set = {
            tuple(int(value) for value in cell)
            for cell in simulation["final"]
        }
        _require(
            final_cells.isdisjoint(final_set),
            "final payload placements may not overlap",
        )
        final_cells.update(final_set)

        all_punctures.update(simulation["punctures"])

        detail = dict(classification)
        detail.update(simulation)
        details.append(detail)
        simulations.append(simulation)

    output = input_grid.copy()
    output[
        input_grid == info["payload"]
    ] = info["background"]

    for row, column in all_punctures:
        output[row, column] = info["background"]

    for simulation in simulations:
        for row, column in simulation["final"]:
            output[int(row), int(column)] = info["payload"]

    return output, info, details


def _solve_grid(input_grid: np.ndarray) -> np.ndarray:
    return _solve_and_describe(input_grid)[0]


def _check_detail_semantics(
    info: dict,
    details: list[dict],
    shape: tuple[int, int],
) -> None:
    axis = info["axis"]
    sign = info["sign"]
    divider = info["divider"]

    for detail in details:
        final = detail["final"]
        if sign > 0:
            final_leading = int(np.max(final[:, axis]))
        else:
            final_leading = int(np.min(final[:, axis]))

        on_membrane = int(
            np.count_nonzero(final[:, axis] == divider)
        )

        if detail["outcome"] == "pass":
            far_boundary = shape[axis] - 1 if sign > 0 else 0
            _require(
                final_leading == far_boundary,
                "a narrow object must reach the far boundary",
            )
            _require(
                detail["stop_reason"] == "boundary",
                "passing object must stop at the boundary",
            )
            _require(
                len(detail["punctures"]) >= 1,
                "passing object must puncture the membrane",
            )
            _require(
                bool(np.all((final[:, axis] - divider) * sign > 0)),
                "passing object must clear the membrane completely",
            )
        else:
            first_wide = int(detail["first_wide"])
            broad_level = (
                final_leading - sign * first_wide
            )
            _require(
                broad_level == divider - sign,
                "first broad slice must stop before the membrane",
            )
            _require(
                detail["stop_reason"] == "membrane",
                "broad object must stop at the membrane",
            )

            expected_load = (
                0
                if detail["outcome"] == "blocked"
                else 1
            )
            _require(
                on_membrane == expected_load,
                "final penetration depth is inconsistent",
            )


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = tuple(int(value) for value in rng.choice(10, size=3, replace=False))
    return {
        "colors": colors,
        "orientation": int(rng.integers(0, 4)),
        "vary_colors": bool(rng.integers(0, 2)),
        "vary_orientation": bool(rng.integers(0, 2)),
    }


def _pair_convention(
    rng: np.random.Generator,
    parameters: dict,
    split: str,
    index: int,
) -> tuple[tuple[int, int, int], int]:
    colors = parameters["colors"]
    orientation = parameters["orientation"]
    if parameters["vary_colors"]:
        if split == "train" and index == 1:
            colors = (colors[1], colors[2], colors[0])
        elif split != "train" or index > 1:
            colors = tuple(int(value) for value in rng.choice(10, size=3, replace=False))
    if parameters["vary_orientation"]:
        if split == "train" and index == 1:
            orientation = (orientation + int(rng.integers(1, 4))) % 4
        elif split != "train" or index > 1:
            orientation = int(rng.integers(0, 4))
    return colors, orientation


def _orient_grid(grid: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 0:
        return grid.copy()
    if orientation == 1:
        return np.flipud(grid).copy()
    if orientation == 2:
        return grid.T.copy()
    _require(orientation == 3, "invalid orientation")
    return np.fliplr(grid.T).copy()


def _training_plan(rng: np.random.Generator, count: int) -> list[list[str]]:
    plan = [[] for _ in range(count)]
    required = ["pass", "blocked", "partial"]
    rng.shuffle(required)
    destinations = list(int(value) for value in rng.permutation(count))
    for index, outcome in enumerate(required):
        plan[destinations[index % count]].append(outcome)
    names = ("pass", "blocked", "partial")
    for outcomes in plan:
        target = int(rng.integers(max(1, len(outcomes)), 4))
        while len(outcomes) < target:
            outcomes.append(names[int(rng.integers(0, 3))])
        rng.shuffle(outcomes)
    return plan


def _test_outcomes(rng: np.random.Generator, scale: int, first: bool) -> list[str]:
    limits = ((1, 2), (2, 4), (3, 6))[scale]
    count = int(rng.integers(limits[0], limits[1] + 1))
    names = ("pass", "blocked", "partial")
    outcomes = [names[int(rng.integers(0, 3))] for _ in range(count)]
    if first and count >= 2:
        outcomes[0] = "pass"
        outcomes[1] = names[int(rng.integers(1, 3))]
    rng.shuffle(outcomes)
    return outcomes


def _sample_profile(
    rng: np.random.Generator,
    outcome: str,
    max_length: int,
    max_width: int,
    min_length: int = 1,
) -> list[tuple[int, ...]]:
    minimum = max(min_length, 2 if outcome == "partial" else 1)
    length = int(rng.integers(minimum, max_length + 1))
    if outcome == "pass":
        return [(0,) for _ in range(length)]

    first_wide = 0 if outcome == "blocked" else int(rng.integers(1, length))
    profile: list[tuple[int, ...]] = [(0,) for _ in range(first_wide)]
    width = int(rng.integers(2, max_width + 1))
    left = int(rng.integers(1 - width, 1))
    profile.append(tuple(range(left, left + width)))
    previous_left, previous_right = left, left + width - 1

    for _ in range(first_wide + 1, length):
        width = int(rng.integers(1, max_width + 1))
        candidates = [
            new_left
            for new_left in range(-3, 4)
            if new_left <= previous_right
            and new_left + width - 1 >= previous_left
            and new_left + width - 1 <= 4
        ]
        left = candidates[int(rng.integers(0, len(candidates)))]
        profile.append(tuple(range(left, left + width)))
        previous_left, previous_right = left, left + width - 1
    return profile


def _profile_span(profile: list[tuple[int, ...]]) -> tuple[int, int]:
    return min(min(level) for level in profile), max(max(level) for level in profile)


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    parameters: dict,
    index: int,
    outcomes: list[str],
    scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    _require(regime in ("source_train", "target_test"), "unknown regime")
    if regime == "source_train":
        height_bounds = (7, 17)
        width_bounds = (7, 18)
        max_length = 4
    else:
        height_bounds = ((10, 17), (18, 23), (24, 30))[scale]
        width_bounds = ((8, 17), (14, 24), (18, 30))[scale]
        max_length = (5, 7, 8)[scale]

    number = len(outcomes)
    max_slice_width = 5 if number <= 2 else (4 if number <= 4 else 3)
    for _ in range(120):
        profiles = [
            _sample_profile(
                rng, outcome, max_length, max_slice_width,
                5 if regime == "target_test" and index == 0 and profile_index == 0 else 1,
            )
            for profile_index, outcome in enumerate(outcomes)
        ]
        spans = [_profile_span(profile) for profile in profiles]
        widths = [right - left + 1 for left, right in spans]
        minimum_width = sum(widths) + number - 1
        minimum_occupied = max(
            len(profile) + (outcome == "blocked")
            for outcome, profile in zip(outcomes, profiles)
        )
        minimum_receiving = max(
            [2]
            + [len(profile) for outcome, profile in zip(outcomes, profiles) if outcome == "pass"]
            + [
                next(i for i, level in enumerate(profile) if len(level) >= 2)
                for outcome, profile in zip(outcomes, profiles) if outcome == "partial"
            ]
        )
        minimum_height = minimum_occupied + 1 + minimum_receiving
        if minimum_width <= width_bounds[1] and minimum_height <= height_bounds[1]:
            break
    else:
        raise ValueError("could not fit connected profiles in the requested regime")

    width = int(rng.integers(max(width_bounds[0], minimum_width), width_bounds[1] + 1))
    height = int(rng.integers(max(height_bounds[0], minimum_height), height_bounds[1] + 1))
    occupied = int(rng.integers(minimum_occupied, height - minimum_receiving))
    divider = occupied
    extra = width - minimum_width
    allocations = rng.multinomial(extra, np.ones(number + 1) / (number + 1))
    left_margin = int(allocations[0])
    centers = []
    cursor = left_margin
    for profile_index, (left, right) in enumerate(spans):
        centers.append(cursor - left)
        cursor += right - left + 1
        if profile_index < number - 1:
            cursor += 1 + int(allocations[profile_index + 1])

    colors, orientation = _pair_convention(
        rng, parameters, "train" if regime == "source_train" else "test", index
    )
    background, membrane, payload = colors
    grid = np.full((height, width), background, dtype=np.int8)
    grid[divider, :] = membrane

    for outcome, profile, center in zip(outcomes, profiles, centers):
        min_gap = 1 if outcome == "blocked" else 0
        max_gap = occupied - len(profile)
        gap = int(rng.integers(min_gap, max_gap + 1))
        leading = divider - 1 - gap
        for slice_index, offsets in enumerate(profile):
            row = leading - slice_index
            for offset in offsets:
                column = center + offset
                _require(grid[row, column] == background, "objects overlap")
                grid[row, column] = payload

    input_grid = _orient_grid(grid, orientation)
    output_grid = _solve_grid(input_grid)
    return input_grid, output_grid


def _validate_requested_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_requested_count(num_train, "num_train", 2, 10)
    num_test = _validate_requested_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        seed_number = abs(int(seed))
        num_test = 1 + seed_number % 4
        num_train = num_test + 1 + (seed_number // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    training_plan = _training_plan(rng, train_count)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_pairs = set()

    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if split == "train":
                outcomes = training_plan[index]
                scale = 0
                regime = "source_train"
            else:
                scale = int(rng.integers(0, 3))
                outcomes = _test_outcomes(rng, scale, index == 0)
                regime = "target_test"
            for _ in range(30):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                input_grid, output_grid = _generate_pair(
                    pair_rng, regime, parameters, index, outcomes, scale
                )
                input_key = _grid_key(input_grid)
                pair_key = (input_key, _grid_key(output_grid))
                if input_key not in seen_inputs and pair_key not in seen_pairs and not _is_source_pair(input_grid, output_grid):
                    break
            else:
                raise ValueError("could not produce a unique pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            puzzle[split].append({"input": input_grid, "output": output_grid})

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10, "invalid training count")
    _require(1 <= len(puzzle["test"]) <= 4, "invalid test count")
    _require(len(puzzle["train"]) > len(puzzle["test"]), "training count must exceed test count")
    seen_inputs = set()
    seen_pairs = set()
    training_outcomes = set()
    test_has_shift = False

    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            input_grid, output_grid = pair["input"], pair["output"]
            for name, grid in (("input", input_grid), ("output", output_grid)):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, f"invalid {name} array")
                _require(all(1 <= size <= 30 for size in grid.shape), f"invalid {name} dimensions")
                _require(np.issubdtype(grid.dtype, np.integer), f"invalid {name} dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), f"invalid {name} colors")
            _require(input_grid.shape == output_grid.shape, "grid dimensions changed")
            _require(not np.array_equal(input_grid, output_grid), "unchanged pair")
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate grid or pair")
            _require(not _is_source_pair(input_grid, output_grid), "copied source pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            expected, info, details = _solve_and_describe(input_grid)
            _require(np.array_equal(output_grid, expected), "incorrect output")
            _check_detail_semantics(info, details, input_grid.shape)
            role_colors = {info["background"], info["membrane"], info["payload"]}
            _require(set(int(value) for value in np.unique(output_grid)) == role_colors, "output lost a role color")
            axial = input_grid.shape[info["axis"]]
            transverse = input_grid.shape[1 - info["axis"]]
            if split == "train":
                _require(7 <= axial <= 17 and 7 <= transverse <= 18, "training size out of range")
                _require(1 <= len(details) <= 3, "training object count out of range")
                _require(all(detail["axial_length"] <= 4 for detail in details), "training object too long")
                training_outcomes.update(detail["outcome"] for detail in details)
            else:
                _require(10 <= axial <= 30 and 8 <= transverse <= 30, "test size out of range")
                _require(1 <= len(details) <= 6, "test object count out of range")
                _require(all(detail["axial_length"] <= 8 for detail in details), "test object too long")
                test_has_shift |= any(detail["axial_length"] >= 5 for detail in details)
    _require(training_outcomes == {"pass", "blocked", "partial"}, "training omits a rule outcome")
    _require(test_has_shift, "test lacks a length generalization")


def _analytic_output(input_grid: np.ndarray) -> np.ndarray:
    info = _analyze_input(input_grid)
    axis, sign, divider = info["axis"], info["sign"], info["divider"]
    output = input_grid.copy()
    output[input_grid == info["payload"]] = info["background"]
    for cells in _components(input_grid == info["payload"]):
        levels = sorted((int(value) for value in np.unique(cells[:, axis])), reverse=(sign > 0))
        first_wide = next((index for index, level in enumerate(levels)
                           if np.count_nonzero(cells[:, axis] == level) >= 2), None)
        leading = levels[0]
        if first_wide is None:
            boundary = input_grid.shape[axis] - 1 if sign > 0 else 0
            displacement = boundary - leading
        else:
            displacement = divider - sign - (leading - sign * first_wide)
        _require(displacement * sign >= 0, "analytic move points backward")
        shifted = cells.copy()
        shifted[:, axis] += displacement
        for cell in cells:
            start = int(cell[axis])
            end = start + displacement
            if min(start, end) <= divider <= max(start, end):
                puncture = cell.copy()
                puncture[axis] = divider
                output[tuple(puncture)] = info["background"]
        for cell in shifted:
            output[tuple(cell)] = info["payload"]
    return output


def _puzzles_equal(first: dict, second: dict) -> bool:
    if first.keys() != second.keys():
        return False
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if not np.array_equal(a["input"], b["input"]) or not np.array_equal(a["output"], b["output"]):
                return False
    return True


def _normalized_layout_key(grid: np.ndarray) -> tuple:
    info = _analyze_input(grid)
    normalized = np.empty(grid.shape, dtype=np.int8)
    for color, role in ((info["background"], 0), (info["membrane"], 1), (info["payload"], 2)):
        normalized[grid == color] = role
    return _grid_key(normalized)


def self_test() -> None:
    for source_input, source_output in _SOURCE_PAIRS:
        _require(np.array_equal(_solve_grid(source_input), source_output), "source rule regression")
        _require(np.array_equal(_analytic_output(source_input), source_output), "source analytic regression")
    for seed, train_count, test_count in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                                          (104, None, 1), (105, None, 4),
                                          (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "test count ignored")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(108, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    a = generate_puzzle(12345, 6, 3)
    b = generate_puzzle(12345, 6, 3)
    _require(_puzzles_equal(a, b), "same seed changed")
    _require(not _puzzles_equal(a, generate_puzzle(12346, 6, 3)), "different seed repeated")

    train_counts, test_counts = set(), set()
    palettes, orientations, dimensions, layouts = set(), set(), set(), set()
    gaps, broad_widths, test_sizes, test_outcomes = set(), set(), set(), set()
    divider_positions, travel_distances = set(), set()
    variable_palette_puzzles = variable_orientation_puzzles = 0
    fixed_palette_puzzles = fixed_orientation_puzzles = 0
    both_variable_puzzles = stepped_broad_objects = edge_objects = 0
    single_block_or_partial = mixed_pass_block = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_palettes, pair_orientations = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze_input(grid)
                _, _, details = _solve_and_describe(grid)
                _require(np.array_equal(_analytic_output(grid), pair["output"]), "analytic prediction mismatch")
                palettes.add((info["background"], info["membrane"], info["payload"]))
                pair_palettes.add((info["background"], info["membrane"], info["payload"]))
                orientation = (info["axis"], info["sign"])
                orientations.add(orientation)
                pair_orientations.add(orientation)
                dimensions.add(grid.shape)
                layouts.add(_normalized_layout_key(grid))
                divider_positions.add(info["divider"])
                if split == "test":
                    axial = grid.shape[info["axis"]]
                    test_sizes.add("small" if axial <= 17 else "moderate" if axial <= 23 else "large")
                    test_outcomes.add(tuple(sorted(set(detail["outcome"] for detail in details))))
                else:
                    outcomes = [detail["outcome"] for detail in details]
                    single_block_or_partial += len(outcomes) == 1 and outcomes[0] in ("blocked", "partial")
                    mixed_pass_block += "pass" in outcomes and "blocked" in outcomes
                for component in _components(grid == info["payload"]):
                    detail = _classify_component(component, info["axis"], info["sign"])
                    leading = (int(np.max(component[:, info["axis"]])) if info["sign"] > 0
                               else int(np.min(component[:, info["axis"]])))
                    gaps.add(abs(info["divider"] - leading) - 1)
                    broad_widths.update(width for width in detail["slice_widths"] if width >= 2)
                    simulation = _simulate_component(component, info, grid.shape)
                    final_leading = (int(np.max(simulation["final"][:, info["axis"]])) if info["sign"] > 0
                                     else int(np.min(simulation["final"][:, info["axis"]])))
                    travel_distances.add(abs(final_leading - leading))
                    transverse = 1 - info["axis"]
                    edge_objects += bool(np.any(component[:, transverse] == 0) or
                                         np.any(component[:, transverse] == grid.shape[transverse] - 1))
                    broad_sections = set()
                    for level in np.unique(component[:, info["axis"]]):
                        positions = component[component[:, info["axis"]] == level, transverse]
                        if len(positions) >= 2:
                            broad_sections.add((int(np.min(positions)), int(np.max(positions))))
                    stepped_broad_objects += len(broad_sections) >= 2
        variable_palette_puzzles += len(pair_palettes) > 1
        variable_orientation_puzzles += len(pair_orientations) > 1
        fixed_palette_puzzles += len(pair_palettes) == 1
        fixed_orientation_puzzles += len(pair_orientations) == 1
        both_variable_puzzles += len(pair_palettes) > 1 and len(pair_orientations) > 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "pair counts lack diversity")
    _require(len(palettes) >= 12 and len(orientations) == 4, "conventions lack diversity")
    _require(len(dimensions) >= 25 and len(layouts) >= 150, "geometry lacks diversity")
    _require(max(gaps) >= 4 and len(gaps) >= 5, "starting distances lack diversity")
    _require(len(divider_positions) >= 10 and len(travel_distances) >= 12, "movement geometry lacks diversity")
    _require(max(broad_widths) >= 4, "broad slices never exceed three cells")
    _require(stepped_broad_objects >= 20 and edge_objects >= 20, "object geometry lacks diversity")
    _require(test_sizes == {"small", "moderate", "large"}, "test scales lack diversity")
    _require(len(test_outcomes) >= 5, "test outcome combinations lack diversity")
    _require(variable_palette_puzzles >= 5 and fixed_palette_puzzles >= 5, "palette modes lack diversity")
    _require(variable_orientation_puzzles >= 5 and fixed_orientation_puzzles >= 5, "orientation modes lack diversity")
    _require(both_variable_puzzles >= 3, "palette and orientation rarely combine")
    _require(single_block_or_partial >= 5 and mixed_pass_block >= 5, "training outcome combinations lack diversity")
