import numpy as np


_SOURCE_INPUT_TEXTS = frozenset(
    {
        (
            "444477777/477477777/777777777/777777777/777777117/"
            "777777117/777777777/777774774/777774444"
        ),
        (
            "77777444/72777474/72777777/72777777/"
            "77777777/77777777/47477777/44477777"
        ),
        (
            "777774444/777774774/787777777/788777777/787777777/"
            "777777777/777777777/477477777/444477777"
        ),
        (
            "77777444/77777474/77777777/77777777/"
            "77377777/77777777/74747777/74447777"
        ),
    }
)


def _check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _checked_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        # This schedule guarantees broad count diversity over consecutive seeds.
        code = abs(int(seed))
        num_test = 1 + code % 4
        baseline_train = 2 + (code // 4) % 9
        num_train = max(num_test + 1, baseline_train)
    elif num_train is None:
        lower = max(2, int(num_test) + 1)
        num_train = int(rng.integers(lower, 11))
    elif num_test is None:
        upper = min(4, int(num_train) - 1)
        num_test = int(rng.integers(1, upper + 1))

    if not (
        2 <= int(num_train) <= 10
        and 1 <= int(num_test) <= 4
        and int(num_train) > int(num_test)
    ):
        raise ValueError(
            "counts must satisfy 2 <= num_train <= 10, "
            "1 <= num_test <= 4, and num_train > num_test"
        )

    return int(num_train), int(num_test)


def _sample_puzzle_parameters(
    rng: np.random.Generator,
    num_train: int,
    num_test: int,
) -> dict:
    colors = tuple(int(v) for v in rng.permutation(10))
    return {
        "colors": colors,
        "vary_palette": bool(rng.integers(0, 2)),
        "vary_axis": bool(rng.integers(0, 2)),
        "vary_direction": bool(rng.integers(0, 2)),
        "base_axis": int(rng.integers(0, 2)),
        "base_direction": -1 if int(rng.integers(0, 2)) else 1,
        "wide_aperture": int(rng.integers(3, 5)) if num_train >= 3 else None,
        "canonical_test": bool(rng.integers(0, 4) == 0),
        "num_train": num_train,
    }


def _lane_pair(
    rng: np.random.Generator, transverse_size: int, gateway_length: int,
    symmetric: bool,
) -> tuple[int, int]:
    candidates = []
    for low in range(transverse_size - gateway_length + 1):
        for high in range(low + gateway_length + 1,
                          transverse_size - gateway_length + 1):
            is_symmetric = high == transverse_size - gateway_length - low
            if is_symmetric == symmetric:
                candidates.append((low, high))
    if not candidates:
        raise ValueError("no separated gateway lanes")
    return candidates[int(rng.integers(len(candidates)))]


def _payload_shape(
    rng: np.random.Generator, width: int, max_height: int,
    force_asymmetric: bool = False,
    force_longitudinal: bool = False,
    force_singleton: bool = False,
) -> np.ndarray:
    if force_singleton:
        return np.ones((1, 1), dtype=np.int8)
    if width == 1:
        return np.ones((int(rng.integers(1, max_height + 1)), 1), dtype=np.int8)
    for _ in range(100):
        height = int(rng.integers(2 if force_longitudinal else 1,
                                  max_height + 1))
        shape = np.zeros((height, width), dtype=np.int8)
        # A growing orthogonally connected object spans the whole aperture.
        t, x = int(rng.integers(height)), int(rng.integers(width))
        shape[t, x] = 1
        for _ in range(height * width * 4):
            occupied = np.argwhere(shape)
            t, x = occupied[int(rng.integers(len(occupied)))]
            neighbors = [(int(t) + dt, int(x) + dx)
                         for dt, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            neighbors = [(a, b) for a, b in neighbors
                         if 0 <= a < height and 0 <= b < width]
            if neighbors:
                a, b = neighbors[int(rng.integers(len(neighbors)))]
                shape[a, b] = 1
            if int(rng.integers(4)) == 0:
                break
        if np.count_nonzero(shape[:, 0]) == 0 or np.count_nonzero(shape[:, -1]) == 0:
            continue
        if np.count_nonzero(shape[0]) == 0 or np.count_nonzero(shape[-1]) == 0:
            continue
        if force_asymmetric and np.array_equal(shape, shape[:, ::-1]):
            continue
        if force_longitudinal and np.array_equal(shape, shape[::-1]):
            continue
        if force_asymmetric and force_longitudinal and np.array_equal(
            shape, shape[::-1, ::-1]
        ):
            continue
        return shape
    # Deterministic connected fallback with both kinds of asymmetry.
    shape = np.ones((min(3, max_height), width), dtype=np.int8)
    shape[0, -1] = 0
    return shape


def _render_pair(
    *,
    longitudinal_size: int,
    transverse_size: int,
    axis: int,
    direction: int,
    background: int,
    gateway: int,
    payload_color: int,
    aperture_width: int,
    min_side_start: int,
    max_side_start: int,
    payload_shape: np.ndarray,
    payload_longitudinal_start: int,
) -> tuple[np.ndarray, np.ndarray]:
    gateway_length = aperture_width + 2
    canonical_input = np.full(
        (longitudinal_size, transverse_size),
        background,
        dtype=np.int8,
    )

    # Minimum-side U: boundary bar plus two inward-pointing legs.
    canonical_input[
        0,
        min_side_start:min_side_start + gateway_length,
    ] = gateway
    canonical_input[1, min_side_start] = gateway
    canonical_input[
        1,
        min_side_start + gateway_length - 1,
    ] = gateway

    # Maximum-side U, facing the minimum-side U.
    canonical_input[
        longitudinal_size - 1,
        max_side_start:max_side_start + gateway_length,
    ] = gateway
    canonical_input[
        longitudinal_size - 2,
        max_side_start,
    ] = gateway
    canonical_input[
        longitudinal_size - 2,
        max_side_start + gateway_length - 1,
    ] = gateway

    if direction == 1:
        entry_start = max_side_start
        exit_start = min_side_start
    else:
        entry_start = min_side_start
        exit_start = max_side_start

    entry_aperture_start = entry_start + 1
    payload_local_cells = np.argwhere(payload_shape != 0)

    for local_t, local_x in payload_local_cells:
        canonical_input[
            payload_longitudinal_start + int(local_t),
            entry_aperture_start + int(local_x),
        ] = payload_color

    canonical_output = canonical_input.copy()
    canonical_output[canonical_output == payload_color] = background

    payload_height = payload_shape.shape[0]
    exit_aperture_start = exit_start + 1

    for local_t, local_x in payload_local_cells:
        if direction == 1:
            output_t = (
                longitudinal_size
                - payload_height
                + int(local_t)
            )
        else:
            output_t = int(local_t)

        output_x = exit_aperture_start + int(local_x)

        _check(
            canonical_output[output_t, output_x] == background,
            "payload destination collides with a gateway",
        )
        canonical_output[output_t, output_x] = payload_color

    if axis == 0:
        return canonical_input.copy(), canonical_output.copy()

    return canonical_input.T.copy(), canonical_output.T.copy()


def _generate_pair_candidate(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    pair_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError(f"unknown pair regime: {regime}")
    colors = puzzle_parameters["colors"]
    if puzzle_parameters["vary_palette"]:
        role_index = pair_index + (puzzle_parameters["num_train"] if regime == "target_test" else 0)
        background = colors[role_index % 10]
        gateway = colors[(role_index + 3) % 10]
    else:
        background, gateway = colors[:2]
    available_colors = [color for color in colors if color not in (background, gateway)]
    if regime == "source_train" and pair_index < 2:
        payload_color = available_colors[pair_index]
    elif regime == "target_test" and bool(rng.integers(2)):
        payload_color = available_colors[0]
    else:
        payload_color = available_colors[int(rng.integers(len(available_colors)))]
    axis = puzzle_parameters["base_axis"]
    direction = puzzle_parameters["base_direction"]
    if puzzle_parameters["vary_axis"] and pair_index % 2 == 1:
        axis = 1 - axis
    if puzzle_parameters["vary_direction"] and pair_index % 2 == 1:
        direction = -direction
    if regime == "target_test":
        if puzzle_parameters["vary_axis"] and bool(rng.integers(2)):
            axis = 1 - axis
        if puzzle_parameters["vary_direction"] and bool(rng.integers(2)):
            direction = -direction

    canonical = regime == "target_test" and pair_index == 0 and puzzle_parameters["canonical_test"]
    if canonical:
        aperture_width = 1
        longitudinal_size = transverse_size = 8
        symmetric = False
        payload_shape = _payload_shape(rng, 1, 1, force_singleton=True)
        payload_color = next(v for v in available_colors if v not in colors[2:4])
    else:
        if regime == "source_train" and pair_index == 0:
            aperture_width = 2
        elif regime == "source_train" and pair_index == 1:
            aperture_width = 1
        elif regime == "source_train" and pair_index == 2:
            aperture_width = puzzle_parameters["wide_aperture"]
        else:
            supported = [1, 2]
            if puzzle_parameters["wide_aperture"] is not None:
                supported.append(puzzle_parameters["wide_aperture"])
            aperture_width = int(rng.choice(supported))
        gateway_length = aperture_width + 2
        longitudinal_size = int(rng.integers(8, 17))
        symmetric = (False if regime == "target_test" and pair_index == 0
                     else bool(rng.integers(2)))
        transverse_size = int(rng.integers(2 * gateway_length + (1 if symmetric else 2),
                                          min(30, 2 * gateway_length + 12) + 1))
        max_height = min(5, longitudinal_size - 2)
        payload_shape = _payload_shape(
            rng, aperture_width, max_height,
            force_asymmetric=(regime == "source_train" and pair_index == 0),
            force_longitudinal=(regime == "source_train" and pair_index == 0),
            force_singleton=(aperture_width == 1 and bool(rng.integers(3) == 0)),
        )
    gateway_length = aperture_width + 2
    low_start, high_start = _lane_pair(rng, transverse_size, gateway_length, symmetric)
    entry_low = bool(rng.integers(2))
    entry_start = low_start if entry_low else high_start
    exit_start = high_start if entry_low else low_start

    if direction == 1:
        min_side_start = exit_start
        max_side_start = entry_start
    else:
        min_side_start = entry_start
        max_side_start = exit_start

    payload_height = int(payload_shape.shape[0])
    latest_start = longitudinal_size - payload_height - 1
    if latest_start < 1:
        raise ValueError("canvas is too short for the payload")
    payload_longitudinal_start = int(rng.integers(1, latest_start + 1))

    return _render_pair(
        longitudinal_size=longitudinal_size,
        transverse_size=transverse_size,
        axis=axis,
        direction=direction,
        background=background,
        gateway=gateway,
        payload_color=payload_color,
        aperture_width=aperture_width,
        min_side_start=min_side_start,
        max_side_start=max_side_start,
        payload_shape=payload_shape,
        payload_longitudinal_start=payload_longitudinal_start,
    )


def _array_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(v) for v in grid.shape),
        tuple(int(v) for v in grid.ravel()),
    )


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _array_key(input_grid), _array_key(output_grid)


def _grid_text(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(
        rng,
        seed,
        num_train,
        num_test,
    )

    puzzle_parameters = _sample_puzzle_parameters(
        rng,
        num_train,
        num_test,
    )

    total_pairs = num_train + num_test
    pair_seeds = rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=total_pairs,
        dtype=np.int64,
    )

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_pairs = set()
    seed_offset = 0

    for section, regime, count in (
        ("train", "source_train", num_train),
        ("test", "target_test", num_test),
    ):
        for pair_index in range(count):
            pair_rng = np.random.default_rng(
                int(pair_seeds[seed_offset])
            )
            seed_offset += 1

            accepted = None
            for _ in range(200):
                input_grid, output_grid = _generate_pair_candidate(
                    pair_rng,
                    regime,
                    puzzle_parameters,
                    pair_index,
                )
                input_key = _array_key(input_grid)
                complete_key = _pair_key(input_grid, output_grid)

                if input_key in used_inputs:
                    continue
                if complete_key in used_pairs:
                    continue
                if _grid_text(input_grid) in _SOURCE_INPUT_TEXTS:
                    continue

                accepted = {
                    "input": input_grid,
                    "output": output_grid,
                }
                used_inputs.add(input_key)
                used_pairs.add(complete_key)
                break

            if accepted is None:
                raise ValueError(
                    "could not generate enough distinct valid pairs"
                )

            puzzle[section].append(accepted)

    validate_generated_puzzle(puzzle)
    return puzzle


def _contiguous_boundary_block(
    boundary: np.ndarray,
) -> tuple[int, int] | None:
    indices = np.flatnonzero(boundary)
    if len(indices) < 3:
        return None
    if not np.all(np.diff(indices) == 1):
        return None
    return int(indices[0]), int(len(indices))


def _detect_gate_geometry(
    grid: np.ndarray,
    candidate_color: int,
) -> dict | None:
    mask = grid == candidate_color
    detections = []

    for axis in (0, 1):
        view = mask if axis == 0 else mask.T
        longitudinal_size, transverse_size = view.shape

        if longitudinal_size < 4:
            continue

        minimum_block = _contiguous_boundary_block(view[0])
        maximum_block = _contiguous_boundary_block(view[-1])
        if minimum_block is None or maximum_block is None:
            continue

        min_start, min_length = minimum_block
        max_start, max_length = maximum_block
        if min_length != max_length:
            continue

        gateway_length = min_length
        expected_min_inner = np.array(
            [min_start, min_start + gateway_length - 1],
            dtype=int,
        )
        expected_max_inner = np.array(
            [max_start, max_start + gateway_length - 1],
            dtype=int,
        )

        if not np.array_equal(
            np.flatnonzero(view[1]),
            expected_min_inner,
        ):
            continue
        if not np.array_equal(
            np.flatnonzero(view[-2]),
            expected_max_inner,
        ):
            continue

        expected = np.zeros_like(view, dtype=bool)
        expected[
            0,
            min_start:min_start + gateway_length,
        ] = True
        expected[1, min_start] = True
        expected[
            1,
            min_start + gateway_length - 1,
        ] = True

        expected[
            -1,
            max_start:max_start + gateway_length,
        ] = True
        expected[-2, max_start] = True
        expected[
            -2,
            max_start + gateway_length - 1,
        ] = True

        if not np.array_equal(view, expected):
            continue

        detections.append(
            {
                "axis": axis,
                "longitudinal_size": longitudinal_size,
                "transverse_size": transverse_size,
                "min_side_start": min_start,
                "max_side_start": max_start,
                "gateway_length": gateway_length,
                "aperture": gateway_length - 2,
            }
        )

    if len(detections) != 1:
        return None
    return detections[0]


def _is_connected(mask: np.ndarray) -> bool:
    coordinates = {
        (int(r), int(c))
        for r, c in np.argwhere(mask)
    }
    if not coordinates:
        return False

    stack = [next(iter(coordinates))]
    reached = set()

    while stack:
        cell = stack.pop()
        if cell in reached:
            continue
        reached.add(cell)
        r, c = cell

        for neighbor in (
            (r - 1, c),
            (r + 1, c),
            (r, c - 1),
            (r, c + 1),
        ):
            if neighbor in coordinates and neighbor not in reached:
                stack.append(neighbor)

    return reached == coordinates


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    _check(
        input_grid.shape == output_grid.shape,
        "input and output dimensions must match",
    )

    input_colors = [int(v) for v in np.unique(input_grid)]
    output_colors = [int(v) for v in np.unique(output_grid)]
    _check(
        len(input_colors) == 3,
        "each input must contain exactly three semantic colors",
    )
    _check(
        set(input_colors) == set(output_colors),
        "input and output must use the same colors",
    )

    gate_detections = []
    for color in input_colors:
        geometry = _detect_gate_geometry(input_grid, color)
        if geometry is not None:
            gate_detections.append((color, geometry))

    _check(
        len(gate_detections) == 1,
        "there must be exactly one detectable gateway color",
    )
    gateway_color, geometry = gate_detections[0]

    remaining_colors = [
        color for color in input_colors if color != gateway_color
    ]
    color_counts = {
        color: int(np.count_nonzero(input_grid == color))
        for color in remaining_colors
    }
    background_color = max(
        remaining_colors,
        key=lambda color: color_counts[color],
    )
    payload_color = next(
        color
        for color in remaining_colors
        if color != background_color
    )

    _check(
        color_counts[background_color] > color_counts[payload_color],
        "background must be distinguishable by extent",
    )

    axis = geometry["axis"]
    input_view = input_grid if axis == 0 else input_grid.T
    output_view = output_grid if axis == 0 else output_grid.T

    longitudinal_size = geometry["longitudinal_size"]
    transverse_size = geometry["transverse_size"]
    min_side_start = geometry["min_side_start"]
    max_side_start = geometry["max_side_start"]
    gateway_length = geometry["gateway_length"]
    aperture = geometry["aperture"]

    _check(aperture >= 1, "gateway aperture must be nonempty")

    separated = (
        min_side_start + gateway_length < max_side_start
        or max_side_start + gateway_length < min_side_start
    )
    _check(
        separated,
        "the two gateway spans must occupy distinct lanes",
    )

    payload_mask = input_view == payload_color
    _check(
        _is_connected(payload_mask),
        "payload must be a single connected object",
    )

    payload_t, payload_x = np.where(payload_mask)
    payload_count = int(len(payload_t))
    _check(payload_count > 0, "payload cannot be empty")

    t_min = int(payload_t.min())
    t_max = int(payload_t.max())
    x_min = int(payload_x.min())
    x_max = int(payload_x.max())

    _check(
        x_max - x_min + 1 == aperture,
        "payload transverse width must equal aperture width",
    )
    _check(
        t_min >= 1 and t_max <= longitudinal_size - 2,
        "input payload must lie inside the portal-facing corridor",
    )

    minimum_aperture_start = min_side_start + 1
    maximum_aperture_start = max_side_start + 1
    matches_minimum = x_min == minimum_aperture_start
    matches_maximum = x_min == maximum_aperture_start

    _check(
        matches_minimum != matches_maximum,
        "payload must align with exactly one gateway aperture",
    )

    if matches_minimum:
        entry_side = 0
        direction = -1
        entry_start = min_side_start
        exit_start = max_side_start
    else:
        entry_side = 1
        direction = 1
        entry_start = max_side_start
        exit_start = min_side_start

    entry_aperture_start = entry_start + 1
    exit_aperture_start = exit_start + 1

    expected_view = input_view.copy()
    expected_view[expected_view == payload_color] = background_color

    destination_cells = []
    for t, x in zip(payload_t, payload_x):
        t = int(t)
        x = int(x)

        if direction == 1:
            output_t = longitudinal_size - 1 - (t_max - t)
        else:
            output_t = t - t_min

        aperture_offset = x - entry_aperture_start
        output_x = exit_aperture_start + aperture_offset

        _check(
            expected_view[output_t, output_x] == background_color,
            "transformed payload collides with fixed structure",
        )
        destination_cells.append((output_t, output_x))

    _check(
        len(set(destination_cells)) == payload_count,
        "payload mapping must be one-to-one",
    )

    for output_t, output_x in destination_cells:
        expected_view[output_t, output_x] = payload_color

    _check(
        np.array_equal(expected_view, output_view),
        "output does not implement the portal transformation",
    )
    _check(
        not np.array_equal(input_grid, output_grid),
        "payload relocation must change the grid",
    )

    output_t_values, _ = np.where(output_view == payload_color)
    output_t_min = int(output_t_values.min())
    output_t_max = int(output_t_values.max())

    if direction == 1:
        _check(
            output_t_max == longitudinal_size - 1,
            "payload must finish flush with the maximum boundary",
        )
    else:
        _check(
            output_t_min == 0,
            "payload must finish flush with the minimum boundary",
        )

    normalized_pattern = tuple(
        sorted(
            (
                int(t) - t_min,
                int(x) - x_min,
            )
            for t, x in zip(payload_t, payload_x)
        )
    )
    reflected_pattern = tuple(
        sorted(
            (
                local_t,
                aperture - 1 - local_x,
            )
            for local_t, local_x in normalized_pattern
        )
    )
    reversed_longitudinal_pattern = tuple(
        sorted((t_max - t, int(x) - x_min) for t, x in zip(payload_t, payload_x))
    )
    halfturn_pattern = tuple(
        sorted((t_max - t, aperture - 1 - (int(x) - x_min))
               for t, x in zip(payload_t, payload_x))
    )

    return {
        "background": background_color,
        "gateway": gateway_color,
        "payload": payload_color,
        "axis": axis,
        "direction": direction,
        "entry_side": entry_side,
        "entry_start": entry_start,
        "exit_start": exit_start,
        "lane_delta": exit_start - entry_start,
        "min_side_start": min_side_start,
        "max_side_start": max_side_start,
        "gateway_length": gateway_length,
        "aperture": aperture,
        "longitudinal_size": longitudinal_size,
        "transverse_size": transverse_size,
        "input_shape": tuple(int(v) for v in input_grid.shape),
        "payload_count": payload_count,
        "payload_height": t_max - t_min + 1,
        "payload_t_min": t_min,
        "payload_t_max": t_max,
        "output_t_min": output_t_min,
        "output_t_max": output_t_max,
        "payload_pattern": normalized_pattern,
        "transversely_asymmetric": (
            normalized_pattern != reflected_pattern
        ),
        "longitudinally_asymmetric": (
            normalized_pattern != reversed_longitudinal_pattern
        ),
        "halfturn_asymmetric": (
            normalized_pattern != halfturn_pattern
        ),
        "symmetric_lanes": (
            max_side_start == transverse_size - gateway_length - min_side_start
        ),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict), "puzzle must be a dictionary")
    _check(
        set(puzzle.keys()) == {"train", "test"},
        "puzzle must contain exactly train and test",
    )
    _check(
        isinstance(puzzle["train"], list),
        "train must be a list",
    )
    _check(
        isinstance(puzzle["test"], list),
        "test must be a list",
    )

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    _check(2 <= train_count <= 10, "invalid train count")
    _check(1 <= test_count <= 4, "invalid test count")
    _check(train_count > test_count, "train count must exceed test count")

    input_keys = set()
    pair_keys = set()

    for section in ("train", "test"):
        for pair in puzzle[section]:
            _check(isinstance(pair, dict), "pair must be a dictionary")
            _check(
                set(pair.keys()) == {"input", "output"},
                "pair must contain exactly input and output",
            )

            for grid_name in ("input", "output"):
                grid = pair[grid_name]
                _check(
                    isinstance(grid, np.ndarray),
                    f"{grid_name} must be a NumPy array",
                )
                _check(grid.ndim == 2, f"{grid_name} must be 2D")
                rows, columns = grid.shape
                _check(
                    1 <= rows <= 30 and 1 <= columns <= 30,
                    f"{grid_name} dimensions are out of range",
                )
                _check(
                    np.issubdtype(grid.dtype, np.integer),
                    f"{grid_name} must use an integer dtype",
                )
                _check(
                    int(grid.min()) >= 0 and int(grid.max()) <= 9,
                    f"{grid_name} contains an invalid color",
                )

            input_key = _array_key(pair["input"])
            complete_key = _pair_key(
                pair["input"],
                pair["output"],
            )
            _check(
                input_key not in input_keys,
                "duplicate input grid",
            )
            _check(
                complete_key not in pair_keys,
                "duplicate input/output pair",
            )
            input_keys.add(input_key)
            pair_keys.add(complete_key)

            _check(
                _grid_text(pair["input"]) not in _SOURCE_INPUT_TEXTS,
                "generated input exactly duplicates a source input",
            )

    train_analyses = [
        _analyze_pair(pair["input"], pair["output"])
        for pair in puzzle["train"]
    ]
    test_analyses = [
        _analyze_pair(pair["input"], pair["output"])
        for pair in puzzle["test"]
    ]
    all_analyses = train_analyses + test_analyses

    # All output checks above are reconstructed from the returned grids.
    train_widths = {a["aperture"] for a in train_analyses}
    _check({1, 2} <= train_widths,
           "training must demonstrate one- and two-cell apertures")
    _check(any(a["aperture"] == 2 and a["transversely_asymmetric"]
               and a["longitudinally_asymmetric"]
               and a["halfturn_asymmetric"] for a in train_analyses),
           "training must distinguish translation and retained row order")
    _check({a["aperture"] for a in test_analyses} <= train_widths,
           "a test aperture width lacks a training demonstration")

    train_axes = {a["axis"] for a in train_analyses}
    train_directions = {a["direction"] for a in train_analyses}
    _check({a["axis"] for a in test_analyses} <= train_axes,
           "test orientation is absent from training")
    _check({a["direction"] for a in test_analyses} <= train_directions,
           "test travel direction is absent from training")

    train_roles = {(a["background"], a["gateway"])
                   for a in train_analyses}
    test_roles = {(a["background"], a["gateway"])
                  for a in test_analyses}
    if len(train_roles) == 1:
        _check(test_roles <= train_roles,
               "test palette variation is absent from training")

    for analysis in all_analyses:
        _check(1 <= analysis["aperture"] <= 4,
               "unsupported gateway aperture")
        _check(analysis["payload_t_min"] >= 1,
               "payload overlaps a boundary bar")
        _check(analysis["payload_t_max"] <= analysis["longitudinal_size"] - 2,
               "payload overlaps a boundary bar")
        _check(analysis["payload_count"] >= 1,
               "payload is empty")

    # Every target set exercises the source's shifted-gateway generalization;
    # further tests may isolate symmetry, size, color, or motif changes.
    _check(any(not a["symmetric_lanes"] for a in test_analyses),
           "target set lacks a shifted gateway arrangement")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            section,
            tuple(
                _pair_key(pair["input"], pair["output"])
                for pair in puzzle[section]
            ),
        )
        for section in ("train", "test")
    )


def _expect_value_error(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> None:
    try:
        generate_puzzle(seed, num_train, num_test)
    except ValueError:
        return
    raise AssertionError(
        "illegal count combination did not raise ValueError"
    )


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    explicit = []
    for seed, train_count, test_count in (
        (1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
        (1004, 6, 2), (1005, 8, 3),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _check((len(puzzle["train"]), len(puzzle["test"])) ==
               (train_count, test_count), "explicit counts ignored")
        explicit.append(puzzle)
    partial = []
    for seed, train_count, test_count in (
        (1101, None, 1), (1102, None, 4),
        (1103, 2, None), (1104, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            _check(len(puzzle["train"]) == train_count, "train count ignored")
        if test_count is not None:
            _check(len(puzzle["test"]) == test_count, "test count ignored")
        partial.append(puzzle)
    for train_count, test_count in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (3, 4), (4, 4), (2, 3),
    ):
        _expect_value_error(1301, train_count, test_count)
    _check(_puzzle_key(generate_puzzle(1201)) ==
           _puzzle_key(generate_puzzle(1201)), "automatic nondeterminism")
    _check(_puzzle_key(generate_puzzle(1202, 7, 3)) ==
           _puzzle_key(generate_puzzle(1202, 7, 3)), "explicit nondeterminism")
    _check(len({_puzzle_key(p) for p in automatic}) >= 38,
           "different seeds lack puzzle diversity")

    train_counts = {len(p["train"]) for p in automatic}
    test_counts = {len(p["test"]) for p in automatic}
    _check(len(train_counts) >= 4 and len(test_counts) >= 3,
           "automatic pair counts lack diversity")

    shapes, layouts, palettes = set(), set(), set()
    train_info, test_info = [], []
    mixed_axis = mixed_direction = mixed_palette = 0
    reused_test_color = novel_test_color = 0
    for puzzle in automatic + explicit + partial:
        validate_generated_puzzle(puzzle)
        train = [_analyze_pair(p["input"], p["output"])
                 for p in puzzle["train"]]
        test = [_analyze_pair(p["input"], p["output"])
                for p in puzzle["test"]]
        train_info.extend(train)
        test_info.extend(test)
        all_info = train + test
        mixed_axis += len({a["axis"] for a in train}) > 1
        mixed_direction += len({a["direction"] for a in train}) > 1
        mixed_palette += len({(a["background"], a["gateway"])
                              for a in train}) > 1
        train_colors = {a["payload"] for a in train}
        reused_test_color += any(a["payload"] in train_colors for a in test)
        novel_test_color += any(a["payload"] not in train_colors for a in test)
        for a in all_info:
            shapes.add(a["input_shape"])
            palettes.add((a["background"], a["gateway"], a["payload"]))
            layouts.add((a["input_shape"], a["axis"], a["direction"],
                         a["aperture"], a["min_side_start"],
                         a["max_side_start"], a["payload_t_min"],
                         a["payload_pattern"]))
    _check(len(shapes) >= 35 and len(layouts) >= 150 and len(palettes) >= 40,
           "insufficient structural or color diversity")
    _check(mixed_axis >= 5 and mixed_direction >= 5 and mixed_palette >= 5,
           "within-puzzle conventions lack variation")
    _check(reused_test_color >= 5 and novel_test_color >= 5,
           "test payload colors are rigidly split")
    _check({a["axis"] for a in train_info} == {0, 1} and
           {a["direction"] for a in train_info} == {-1, 1},
           "orientation or travel diversity is missing")
    _check({a["aperture"] for a in train_info} >= {1, 2, 3, 4},
           "wider apertures are missing")
    _check(any(a["payload_count"] == 1 for a in train_info) and
           any(a["payload_count"] > 1 for a in test_info),
           "singleton and larger-payload regimes are rigid")
    _check(any(a["symmetric_lanes"] for a in test_info) and
           any(not a["symmetric_lanes"] for a in train_info),
           "gateway symmetry is rigidly split")
    _check(any(a["aperture"] == 2 and a["transversely_asymmetric"]
               for a in test_info), "asymmetric width-two tests missing")
    _check(any(a["payload_t_min"] == 1 for a in train_info) and
           any(a["payload_t_max"] == a["longitudinal_size"] - 2
               for a in train_info), "near-edge source positions missing")
    _check(any(a["aperture"] >= 2 and not a["symmetric_lanes"]
               and a["transversely_asymmetric"] for a in test_info),
           "combined test variations missing")
