import numpy as np


_DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
_PROJECTILE_WIDTHS = (2, 4, 6, 8)

_SOURCE_PAIR_ROWS = (
    (
        (
            "7777777777",
            "7777777777",
            "7779999777",
            "7779999777",
            "7777337777",
            "7777337777",
            "7777887777",
            "7777887777",
            "7777887777",
            "7777887777",
        ),
        (
            "7777337777",
            "7777337777",
            "7799779977",
            "7799779977",
            "7777777777",
            "7777777777",
            "7777887777",
            "7777887777",
            "7777887777",
            "7777887777",
        ),
    ),
    (
        (
            "7776666777",
            "7776666777",
            "7777667777",
            "7777117777",
            "7777117777",
            "7777117777",
            "7788888877",
            "7788888877",
            "7788888877",
            "7788888877",
        ),
        (
            "7776666777",
            "7776666777",
            "7777667777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7888778887",
            "7888118887",
            "7888118887",
            "7888118887",
        ),
    ),
    (
        (
            "7777777777",
            "7777777777",
            "7777777772",
            "7477777722",
            "7477771122",
            "7477771122",
            "7477777722",
            "7777777772",
            "7777777777",
            "7777777777",
        ),
        (
            "7777777777",
            "7777777777",
            "7477777772",
            "7477777722",
            "1177777722",
            "1177777722",
            "7477777722",
            "7477777772",
            "7777777777",
            "7777777777",
        ),
    ),
    (
        (
            "2777777777",
            "2277777788",
            "2222997788",
            "2222997788",
            "2277777788",
            "2777777777",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
        ),
        (
            "2777777788",
            "2277777788",
            "2222777799",
            "2222777799",
            "2277777788",
            "2777777788",
            "7777777777",
            "7777777777",
            "7777777777",
            "7777777777",
        ),
    ),
)


def _grid_key(grid):
    normalized = np.ascontiguousarray(grid.astype(np.int8, copy=False))
    return tuple(grid.shape), normalized.tobytes()


def _pair_key(input_grid, output_grid):
    return _grid_key(input_grid), _grid_key(output_grid)


def _array_rows(grid):
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _matches_source_pair(input_grid, output_grid):
    if tuple(input_grid.shape) != (10, 10):
        return False
    if tuple(output_grid.shape) != (10, 10):
        return False
    rows = (_array_rows(input_grid), _array_rows(output_grid))
    return rows in _SOURCE_PAIR_ROWS


def _axis_sizes(shape, direction):
    rows, cols = shape
    if direction in ("LEFT", "RIGHT"):
        return cols, rows
    return rows, cols


def _to_rc(u, v, axis_length, direction):
    if direction == "RIGHT":
        return v, u
    if direction == "LEFT":
        return v, axis_length - 1 - u
    if direction == "DOWN":
        return u, v
    if direction == "UP":
        return axis_length - 1 - u, v
    raise ValueError("Unknown direction")


def _uv_coordinates(grid, color, direction):
    rows, cols = np.where(grid == color)
    if direction == "RIGHT":
        return cols.astype(int), rows.astype(int)
    if direction == "LEFT":
        return (grid.shape[1] - 1 - cols).astype(int), rows.astype(int)
    if direction == "DOWN":
        return rows.astype(int), cols.astype(int)
    if direction == "UP":
        return (grid.shape[0] - 1 - rows).astype(int), cols.astype(int)
    raise ValueError("Unknown direction")


def _rectangle_bounds(u_values, v_values):
    if len(u_values) == 0:
        return None

    u0 = int(np.min(u_values))
    u1 = int(np.max(u_values))
    v0 = int(np.min(v_values))
    v1 = int(np.max(v_values))

    expected_size = (u1 - u0 + 1) * (v1 - v0 + 1)
    if len(u_values) != expected_size:
        return None

    occupied = set(zip(u_values.tolist(), v_values.tolist()))
    for u in range(u0, u1 + 1):
        for v in range(v0, v1 + 1):
            if (u, v) not in occupied:
                return None

    return u0, u1, v0, v1


def _compressed_profile(profile):
    result = []
    for value in profile:
        if not result or result[-1] != value:
            result.append(value)
    return result


def _launcher_kind(profile, projectile_width):
    compressed = _compressed_profile(profile)
    if not compressed or compressed[-1] < projectile_width:
        return None
    if any(value % 2 or value < projectile_width for value in compressed):
        return None
    if any(a <= b for a, b in zip(compressed, compressed[1:])):
        return None
    return ("block" if len(compressed) == 1 else
            "shoulder" if len(compressed) == 2 else "taper")


def _try_interpret_roles(
    input_grid,
    background,
    direction,
    launcher_color,
    projectile_color,
    barrier_color,
):
    axis_length, transverse_length = _axis_sizes(
        input_grid.shape, direction
    )

    launcher_u, launcher_v = _uv_coordinates(
        input_grid, launcher_color, direction
    )
    projectile_u, projectile_v = _uv_coordinates(
        input_grid, projectile_color, direction
    )
    barrier_u, barrier_v = _uv_coordinates(
        input_grid, barrier_color, direction
    )

    if len(launcher_u) == 0 or int(np.min(launcher_u)) != 0:
        return None

    projectile_bounds = _rectangle_bounds(projectile_u, projectile_v)
    barrier_bounds = _rectangle_bounds(barrier_u, barrier_v)
    if projectile_bounds is None or barrier_bounds is None:
        return None

    pu0, pu1, pv0, pv1 = projectile_bounds
    bu0, bu1, bv0, bv1 = barrier_bounds

    launcher_end = int(np.max(launcher_u))
    if pu0 != launcher_end + 1:
        return None

    launcher_axis_values = np.unique(launcher_u)
    if not np.array_equal(
        launcher_axis_values,
        np.arange(launcher_end + 1),
    ):
        return None

    projectile_length = pu1 - pu0 + 1
    projectile_width = pv1 - pv0 + 1
    if projectile_length < 1 or projectile_length > 5:
        return None
    if projectile_width not in _PROJECTILE_WIDTHS:
        return None
    if projectile_width % 2 != 0:
        return None

    center_twice = pv0 + pv1 + 1
    if center_twice % 2 != 0:
        return None
    center = center_twice // 2

    launcher_profile = []
    for u in range(launcher_end + 1):
        values = np.sort(launcher_v[launcher_u == u])
        if len(values) == 0:
            return None
        if not np.array_equal(
            values,
            np.arange(int(values[0]), int(values[-1]) + 1),
        ):
            return None
        if int(values[0]) + int(values[-1]) + 1 != 2 * center:
            return None
        launcher_profile.append(int(len(values)))

    launcher_kind = _launcher_kind(
        launcher_profile, projectile_width
    )
    if launcher_kind is None:
        return None
    if launcher_profile[-1] < projectile_width:
        return None

    if bu0 < pu1 + 1:
        return None

    barrier_span = bv1 - bv0 + 1
    barrier_thickness = bu1 - bu0 + 1

    if barrier_span % 2 != 0:
        return None
    if barrier_span <= projectile_width:
        return None
    if barrier_span <= barrier_thickness:
        return None
    if bv0 + bv1 + 1 != 2 * center:
        return None

    shift = projectile_width // 2
    if bv0 - shift < 0:
        return None
    if bv1 + shift >= transverse_length:
        return None

    destination_start = axis_length - projectile_length
    if pu1 >= destination_start:
        return None

    gap = bu0 - pu1 - 1
    tail = axis_length - 1 - bu1
    if gap < 0 or tail < 0:
        return None

    return {
        "shape": tuple(input_grid.shape),
        "background": int(background),
        "direction": direction,
        "axis_length": int(axis_length),
        "transverse_length": int(transverse_length),
        "launcher_color": int(launcher_color),
        "projectile_color": int(projectile_color),
        "barrier_color": int(barrier_color),
        "launcher_profile": tuple(launcher_profile),
        "launcher_kind": launcher_kind,
        "projectile_bounds": (
            int(pu0), int(pu1), int(pv0), int(pv1)
        ),
        "barrier_bounds": (
            int(bu0), int(bu1), int(bv0), int(bv1)
        ),
        "projectile_length": int(projectile_length),
        "projectile_width": int(projectile_width),
        "barrier_span": int(barrier_span),
        "barrier_thickness": int(barrier_thickness),
        "center": int(center),
        "shift": int(shift),
        "gap": int(gap),
        "tail": int(tail),
        "destination_start": int(destination_start),
    }


def _interpret_input(input_grid):
    assert isinstance(input_grid, np.ndarray)
    assert input_grid.ndim == 2
    assert np.issubdtype(input_grid.dtype, np.integer)
    assert input_grid.size > 0
    assert int(np.min(input_grid)) >= 0
    assert int(np.max(input_grid)) <= 9

    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 4

    largest_index = int(np.argmax(counts))
    assert int(np.sum(counts == counts[largest_index])) == 1

    background = int(values[largest_index])
    assert all(
        int(counts[largest_index]) > int(counts[index])
        for index in range(len(counts))
        if index != largest_index
    )

    object_colors = [
        int(value) for value in values if int(value) != background
    ]

    candidates = []
    for direction in _DIRECTIONS:
        for launcher_color in object_colors:
            for projectile_color in object_colors:
                if projectile_color == launcher_color:
                    continue

                remaining = [
                    color
                    for color in object_colors
                    if color not in (launcher_color, projectile_color)
                ]
                if len(remaining) != 1:
                    continue

                candidate = _try_interpret_roles(
                    input_grid=input_grid,
                    background=background,
                    direction=direction,
                    launcher_color=launcher_color,
                    projectile_color=projectile_color,
                    barrier_color=remaining[0],
                )
                if candidate is not None:
                    candidates.append(candidate)

    assert len(candidates) == 1, (
        "Input scene must have one unambiguous role interpretation"
    )
    return candidates[0]


def _paint_rectangle_uv(
    grid,
    direction,
    axis_length,
    u0,
    u1,
    v0,
    v1,
    color,
    background,
):
    for u in range(u0, u1 + 1):
        for v in range(v0, v1 + 1):
            row, col = _to_rc(u, v, axis_length, direction)
            assert int(grid[row, col]) == int(background)
            grid[row, col] = color


def _construct_output(input_grid, analysis):
    background = analysis["background"]
    direction = analysis["direction"]
    axis_length = analysis["axis_length"]
    launcher_color = analysis["launcher_color"]
    projectile_color = analysis["projectile_color"]
    barrier_color = analysis["barrier_color"]

    pu0, pu1, pv0, pv1 = analysis["projectile_bounds"]
    bu0, bu1, bv0, bv1 = analysis["barrier_bounds"]

    projectile_length = pu1 - pu0 + 1
    destination_start = axis_length - projectile_length
    shift = analysis["shift"]
    center = analysis["center"]

    output_grid = np.full(
        input_grid.shape, background, dtype=np.int8
    )

    output_grid[input_grid == launcher_color] = launcher_color

    _paint_rectangle_uv(
        output_grid,
        direction,
        axis_length,
        bu0,
        bu1,
        bv0 - shift,
        center - shift - 1,
        barrier_color,
        background,
    )
    _paint_rectangle_uv(
        output_grid,
        direction,
        axis_length,
        bu0,
        bu1,
        center + shift,
        bv1 + shift,
        barrier_color,
        background,
    )
    _paint_rectangle_uv(
        output_grid,
        direction,
        axis_length,
        destination_start,
        axis_length - 1,
        pv0,
        pv1,
        projectile_color,
        background,
    )

    return output_grid


def _derive_output_from_input(input_grid):
    analysis = _interpret_input(input_grid)
    return _construct_output(input_grid, analysis)


def _analyze_pair(input_grid, output_grid, regime):
    assert isinstance(input_grid, np.ndarray)
    assert isinstance(output_grid, np.ndarray)
    assert input_grid.ndim == 2
    assert output_grid.ndim == 2
    assert input_grid.shape == output_grid.shape

    analysis = _interpret_input(input_grid)
    expected_output = _construct_output(input_grid, analysis)
    assert np.array_equal(output_grid, expected_output)

    assert set(np.unique(input_grid).tolist()) == set(
        np.unique(output_grid).tolist()
    )

    input_counts = np.bincount(
        input_grid.astype(np.int64).ravel(), minlength=10
    )
    output_counts = np.bincount(
        output_grid.astype(np.int64).ravel(), minlength=10
    )
    assert np.array_equal(input_counts, output_counts)
    assert not np.array_equal(input_grid, output_grid)

    assert regime in ("source_train", "target_test")

    result = dict(analysis)
    result["barrier_at_edge"] = analysis["tail"] == 0
    result["barrier_clear_of_destination"] = (
        analysis["barrier_bounds"][1]
        < analysis["destination_start"]
    )
    bv0, bv1 = analysis["barrier_bounds"][2:]
    result["split_touches_side"] = (
        bv0 - analysis["shift"] == 0
        or bv1 + analysis["shift"] == analysis["transverse_length"] - 1
    )
    result["launcher_touches_corner"] = any(
        width == 2 * analysis["center"]
        or width == 2 * (analysis["transverse_length"] - analysis["center"])
        for width in analysis["launcher_profile"]
    )
    result["wider_tip"] = (
        analysis["launcher_profile"][-1] > analysis["projectile_width"]
    )
    return result


def _check_transformation_masks(input_grid, output_grid, analysis):
    """Check the three color footprints without using the output painter."""
    direction = analysis["direction"]
    axis_length = analysis["axis_length"]
    pu0, pu1, pv0, pv1 = analysis["projectile_bounds"]
    bu0, bu1, bv0, bv1 = analysis["barrier_bounds"]
    shift = analysis["shift"]
    center = analysis["center"]

    def positions(color):
        u, v = _uv_coordinates(output_grid, color, direction)
        return set(zip(u.tolist(), v.tolist()))

    launcher = analysis["launcher_color"]
    assert np.array_equal(input_grid == launcher, output_grid == launcher)
    expected_projectile = {
        (u, v)
        for u in range(axis_length - (pu1 - pu0 + 1), axis_length)
        for v in range(pv0, pv1 + 1)
    }
    assert positions(analysis["projectile_color"]) == expected_projectile
    expected_barrier = {
        (u, v)
        for u in range(bu0, bu1 + 1)
        for v in list(range(bv0 - shift, center - shift))
        + list(range(center + shift, bv1 + shift + 1))
    }
    assert positions(analysis["barrier_color"]) == expected_barrier
    assert int(np.count_nonzero(output_grid == analysis["background"])) == (
        output_grid.size
        - int(np.count_nonzero(output_grid == launcher))
        - len(expected_projectile)
        - len(expected_barrier)
    )


def _layout_key(analysis):
    return (
        analysis["direction"],
        analysis["shape"],
        analysis["projectile_bounds"],
        analysis["barrier_bounds"],
        analysis["launcher_profile"],
    )


def _validate_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < low or int(value) > high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )


def _resolve_counts(rng, seed, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        resolved_test = 1 + seed % 4
        minimum_train = max(2, resolved_test + 1)
        span = 10 - minimum_train + 1
        resolved_train = minimum_train + (seed // 4) % span
    elif num_train is None:
        resolved_test = int(num_test)
        minimum_train = max(2, resolved_test + 1)
        resolved_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        resolved_train = int(num_train)
        maximum_test = min(4, resolved_train - 1)
        resolved_test = int(rng.integers(1, maximum_test + 1))
    else:
        resolved_train = int(num_train)
        resolved_test = int(num_test)

    if resolved_train <= resolved_test:
        raise ValueError("num_train must be greater than num_test")

    return resolved_train, resolved_test


def _sample_puzzle_parameters(rng):
    return {
        "background": int(rng.integers(0, 10)),
        "projectile_width": int(rng.choice(_PROJECTILE_WIDTHS)),
        "mix_background": bool(rng.integers(0, 2)),
        "mix_width": bool(rng.integers(0, 2)),
    }


def _launcher_profile(rng, motif, projectile_width, half_barrier):
    # Widths are centered on the projectile axis. A level is one cell per side.
    if motif == "block":
        levels = [int(rng.integers(0, min(half_barrier, 3) + 1))]
    elif motif == "shoulder":
        tip = int(rng.integers(0, min(half_barrier - 1, 3) + 1))
        base = int(rng.integers(tip + 1, half_barrier + 1))
        levels = [base, tip]
    elif motif == "taper":
        tip = int(rng.integers(0, min(half_barrier - 2, 2) + 1))
        base = int(rng.integers(tip + 2, half_barrier + 1))
        middle = int(rng.integers(tip + 1, base))
        levels = [base, middle, tip]
        if base - tip >= 3 and bool(rng.integers(0, 2)):
            remaining = [level for level in range(tip + 1, base) if level != middle]
            if remaining:
                levels = sorted([base, middle, int(rng.choice(remaining)), tip], reverse=True)
    else:
        raise ValueError("Unknown launcher motif")

    profile = []
    for level in levels:
        profile.extend([projectile_width + 2 * level] * int(rng.integers(1, 4)))
    return profile


def _generate_pair_once(
    rng,
    regime,
    background,
    projectile_width,
    direction,
    motif,
    tail_mode,
):
    shift = projectile_width // 2

    available_colors = [
        color for color in range(10) if color != background
    ]
    role_colors = rng.choice(
        np.array(available_colors, dtype=int),
        size=3,
        replace=False,
    )
    launcher_color = int(role_colors[0])
    projectile_color = int(role_colors[1])
    barrier_color = int(role_colors[2])

    half_barrier = int(rng.integers(shift + 1, min(shift + 5, (26 - projectile_width) // 2) + 1))
    barrier_span = 2 * half_barrier

    expanded_span = barrier_span + projectile_width
    low_margin = int(rng.integers(0, min(4, 30 - expanded_span) + 1))
    high_margin = int(rng.integers(0, min(4, 30 - expanded_span - low_margin) + 1))
    transverse_length = (
        low_margin
        + expanded_span
        + high_margin
    )
    center = low_margin + half_barrier + shift

    profile = _launcher_profile(
        rng, motif, projectile_width, half_barrier
    )
    launcher_depth = len(profile)

    projectile_length = int(rng.integers(1, 6))
    max_thickness = min(5, barrier_span - 1)
    barrier_thickness = int(rng.integers(1, max_thickness + 1))
    gap = int(rng.integers(0, 9))

    if tail_mode == "edge":
        tail = 0
    elif tail_mode == "clear":
        tail = projectile_length + int(rng.integers(0, 6))
    elif tail_mode == "overlap":
        if projectile_length == 1:
            raise AssertionError("overlap needs a multi-cell projectile")
        else:
            tail = int(rng.integers(1, projectile_length))
    else:
        raise ValueError("Unknown tail mode")

    movement_margin = gap + barrier_thickness + tail
    if movement_margin < projectile_length:
        gap += projectile_length - movement_margin

    axis_length = (
        launcher_depth
        + projectile_length
        + gap
        + barrier_thickness
        + tail
    )
    if axis_length < 8:
        gap += 8 - axis_length
        axis_length = 8

    assert axis_length <= 30
    assert transverse_length <= 30

    projectile_u0 = launcher_depth
    projectile_u1 = projectile_u0 + projectile_length - 1
    projectile_v0 = center - shift
    projectile_v1 = center + shift - 1

    barrier_u0 = projectile_u1 + 1 + gap
    barrier_u1 = barrier_u0 + barrier_thickness - 1
    barrier_v0 = center - half_barrier
    barrier_v1 = center + half_barrier - 1

    if direction in ("LEFT", "RIGHT"):
        shape = (transverse_length, axis_length)
    else:
        shape = (axis_length, transverse_length)

    input_grid = np.full(shape, background, dtype=np.int8)

    for u, width in enumerate(profile):
        v0 = center - width // 2
        v1 = center + width // 2 - 1
        _paint_rectangle_uv(
            input_grid,
            direction,
            axis_length,
            u,
            u,
            v0,
            v1,
            launcher_color,
            background,
        )

    _paint_rectangle_uv(
        input_grid,
        direction,
        axis_length,
        projectile_u0,
        projectile_u1,
        projectile_v0,
        projectile_v1,
        projectile_color,
        background,
    )
    _paint_rectangle_uv(
        input_grid,
        direction,
        axis_length,
        barrier_u0,
        barrier_u1,
        barrier_v0,
        barrier_v1,
        barrier_color,
        background,
    )

    analysis = _interpret_input(input_grid)
    assert analysis["direction"] == direction
    assert analysis["launcher_kind"] == motif
    assert analysis["projectile_width"] == projectile_width

    output_grid = _construct_output(input_grid, analysis)
    _analyze_pair(input_grid, output_grid, regime)
    return input_grid, output_grid


def _make_unique_pair(
    pair_rng,
    regime,
    background,
    projectile_width,
    direction,
    motif,
    tail_mode,
    seen_inputs,
    seen_pairs,
    seen_layouts,
):
    for _ in range(400):
        try:
            input_grid, output_grid = _generate_pair_once(
                pair_rng,
                regime,
                background,
                projectile_width,
                direction,
                motif,
                tail_mode,
            )
            analysis = _analyze_pair(
                input_grid, output_grid, regime
            )
            _check_transformation_masks(input_grid, output_grid, analysis)
        except AssertionError:
            continue

        if analysis["background"] != background:
            continue
        if analysis["projectile_width"] != projectile_width:
            continue
        if _matches_source_pair(input_grid, output_grid):
            continue

        input_key = _grid_key(input_grid)
        pair_key = _pair_key(input_grid, output_grid)
        layout_key = _layout_key(analysis)

        if input_key in seen_inputs:
            continue
        if pair_key in seen_pairs:
            continue
        if layout_key in seen_layouts:
            continue

        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
        seen_layouts.add(layout_key)

        return {
            "input": input_grid,
            "output": output_grid,
        }

    raise ValueError("Unable to generate a unique valid pair")


def _pair_schedule(rng, train_count, test_count, parameters):
    held_out = (str(rng.choice(_DIRECTIONS)) if bool(rng.integers(0, 2))
                else None)
    available = [d for d in _DIRECTIONS if d != held_out]
    vertical = [d for d in available if d in ("UP", "DOWN")]
    horizontal = [d for d in available if d in ("LEFT", "RIGHT")]
    train_directions = [str(rng.choice(vertical)), str(rng.choice(horizontal))]
    train_directions.extend(str(rng.choice(available)) for _ in range(train_count - 2))
    rng.shuffle(train_directions)

    if held_out is None:
        test_directions = [str(rng.choice(train_directions)) for _ in range(test_count)]
    else:
        test_directions = [held_out]
        test_directions.extend(str(rng.choice(_DIRECTIONS)) for _ in range(test_count - 1))
    if test_count > 1 and len(set(test_directions)) == 1:
        alternatives = [d for d in _DIRECTIONS if d != test_directions[0]]
        test_directions[-1] = str(rng.choice(alternatives))
    rng.shuffle(test_directions)

    train_motifs = ["block", str(rng.choice(("shoulder", "taper")))]
    if train_count >= 3:
        train_motifs[1] = "shoulder"
        train_motifs.append("taper")
    train_motifs.extend(str(rng.choice(("block", "shoulder", "taper")))
                        for _ in range(train_count - len(train_motifs)))
    rng.shuffle(train_motifs)
    test_motifs = [str(rng.choice(train_motifs)) for _ in range(test_count)]
    if test_count > 1 and len(set(test_motifs)) == 1 and len(set(train_motifs)) > 1:
        test_motifs[-1] = str(rng.choice([m for m in ("block", "shoulder", "taper")
                                           if m in train_motifs and m != test_motifs[0]]))

    train_tails = ["edge", "clear"]
    if train_count >= 3:
        train_tails.append("overlap")
    train_tails.extend(str(rng.choice(("edge", "clear", "overlap")))
                       for _ in range(train_count - len(train_tails)))
    rng.shuffle(train_tails)
    test_tails = [str(rng.choice(train_tails)) for _ in range(test_count)]
    if test_count > 1 and len(set(test_tails)) == 1 and len(set(train_tails)) > 1:
        test_tails[-1] = str(rng.choice([t for t in ("edge", "clear", "overlap")
                                        if t in train_tails and t != test_tails[0]]))

    background = parameters["background"]
    width = parameters["projectile_width"]
    backgrounds = [background] * (train_count + test_count)
    widths = [width] * (train_count + test_count)
    if parameters["mix_background"]:
        backgrounds[1] = int(rng.choice([c for c in range(10) if c != background]))
        for i in range(2, len(backgrounds)):
            backgrounds[i] = int(rng.integers(0, 10))
    if parameters["mix_width"]:
        widths[1] = int(rng.choice([w for w in _PROJECTILE_WIDTHS if w != width]))
        for i in range(2, len(widths)):
            widths[i] = int(rng.choice(_PROJECTILE_WIDTHS))

    return [
        (direction, motif, tail, backgrounds[i], widths[i])
        for i, (direction, motif, tail) in enumerate(
            zip(train_directions + test_directions,
                train_motifs + test_motifs,
                train_tails + test_tails)
        )
    ]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")

    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    rng = np.random.default_rng(seed)
    resolved_train, resolved_test = _resolve_counts(
        rng, seed, num_train, num_test
    )
    puzzle_parameters = _sample_puzzle_parameters(rng)

    schedule = _pair_schedule(
        rng, resolved_train, resolved_test, puzzle_parameters
    )

    pair_seed_count = resolved_train + resolved_test
    pair_seeds = rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=pair_seed_count,
        dtype=np.int64,
    )

    seen_inputs = set()
    seen_pairs = set()
    seen_layouts = set()

    train = []
    for index in range(resolved_train):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        train.append(
            _make_unique_pair(
                pair_rng=pair_rng,
                regime="source_train",
                direction=schedule[index][0],
                motif=schedule[index][1],
                tail_mode=schedule[index][2],
                background=schedule[index][3],
                projectile_width=schedule[index][4],
                seen_inputs=seen_inputs,
                seen_pairs=seen_pairs,
                seen_layouts=seen_layouts,
            )
        )

    test = []
    for index in range(resolved_test):
        pair_seed = int(pair_seeds[resolved_train + index])
        pair_rng = np.random.default_rng(pair_seed)
        spec = schedule[resolved_train + index]
        test.append(
            _make_unique_pair(
                pair_rng=pair_rng,
                regime="target_test",
                direction=spec[0],
                motif=spec[1],
                tail_mode=spec[2],
                background=spec[3],
                projectile_width=spec[4],
                seen_inputs=seen_inputs,
                seen_pairs=seen_pairs,
                seen_layouts=seen_layouts,
            )
        )

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    num_train = len(puzzle["train"])
    num_test = len(puzzle["test"])
    assert 2 <= num_train <= 10
    assert 1 <= num_test <= 4
    assert num_train > num_test

    seen_inputs = set()
    seen_pairs = set()
    seen_layouts = set()
    train_analyses = []
    test_analyses = []

    for split_name, regime in (
        ("train", "source_train"),
        ("test", "target_test"),
    ):
        for pair in puzzle[split_name]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                rows, cols = grid.shape
                assert 1 <= rows <= 30
                assert 1 <= cols <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert grid.size > 0
                assert int(np.min(grid)) >= 0
                assert int(np.max(grid)) <= 9

            assert input_grid.shape == output_grid.shape

            input_key = _grid_key(input_grid)
            pair_key = _pair_key(input_grid, output_grid)
            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs
            assert not _matches_source_pair(input_grid, output_grid)

            analysis = _analyze_pair(
                input_grid, output_grid, regime
            )
            layout_key = _layout_key(analysis)
            assert layout_key not in seen_layouts

            independently_derived = _derive_output_from_input(input_grid)
            assert np.array_equal(
                independently_derived, output_grid
            )

            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            seen_layouts.add(layout_key)

            if split_name == "train":
                train_analyses.append(analysis)
            else:
                test_analyses.append(analysis)

    all_analyses = train_analyses + test_analyses

    train_directions = {
        analysis["direction"] for analysis in train_analyses
    }
    assert bool(train_directions.intersection({"UP", "DOWN"}))
    assert bool(train_directions.intersection({"LEFT", "RIGHT"}))

    train_launcher_kinds = {
        analysis["launcher_kind"] for analysis in train_analyses
    }
    assert "block" in train_launcher_kinds
    assert bool(train_launcher_kinds.intersection({"shoulder", "taper"}))
    if num_train >= 3:
        assert {"block", "shoulder", "taper"}.issubset(train_launcher_kinds)

    assert any(
        analysis["barrier_at_edge"]
        for analysis in train_analyses
    )
    assert any(
        analysis["barrier_clear_of_destination"]
        for analysis in train_analyses
    )

    for analysis in test_analyses:
        assert analysis["launcher_kind"] in train_launcher_kinds

    # Literal colors and projectile width are read from each input, so they
    # may differ even between two demonstrations of the same rule.
    for analysis in all_analyses:
        assert analysis["projectile_width"] == (
            analysis["projectile_bounds"][3]
            - analysis["projectile_bounds"][2] + 1
        )

    train_layouts = {
        _layout_key(analysis) for analysis in train_analyses
    }
    assert len(train_layouts) == num_train


def _puzzle_key(puzzle):
    result = []
    for split_name in ("train", "test"):
        result.append(split_name)
        for pair in puzzle[split_name]:
            result.append(_pair_key(pair["input"], pair["output"]))
    return tuple(result)


def _same_puzzle(first, second):
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split_name in ("train", "test"):
        if len(first[split_name]) != len(second[split_name]):
            return False
        for pair_a, pair_b in zip(
            first[split_name], second[split_name]
        ):
            if not np.array_equal(
                pair_a["input"], pair_b["input"]
            ):
                return False
            if not np.array_equal(
                pair_a["output"], pair_b["output"]
            ):
                return False
    return True


def _assert_value_error(seed, num_train=None, num_test=None):
    try:
        generate_puzzle(
            seed,
            num_train=num_train,
            num_test=num_test,
        )
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def self_test() -> None:
    for source_input, source_output in _SOURCE_PAIR_ROWS:
        original_input = np.array(
            [[int(cell) for cell in row] for row in source_input], dtype=np.int8
        )
        original_output = np.array(
            [[int(cell) for cell in row] for row in source_output], dtype=np.int8
        )
        source_analysis = _interpret_input(original_input)
        assert np.array_equal(_derive_output_from_input(original_input),
                              original_output)
        _check_transformation_masks(original_input, original_output,
                                    source_analysis)

    automatic = generate_puzzle(314159)
    assert _same_puzzle(automatic, generate_puzzle(314159))
    assert not _same_puzzle(generate_puzzle(9001, 4, 1),
                            generate_puzzle(9002, 4, 1))
    explicit = generate_puzzle(271828, 6, 3)
    assert (len(explicit["train"]), len(explicit["test"])) == (6, 3)

    for index, (train_count, test_count) in enumerate(
        ((2, 1), (10, 1), (10, 4))
    ):
        puzzle = generate_puzzle(7000 + index, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)

    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(8000 + index, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count

    for index, (train_count, test_count) in enumerate(
        ((1, None), (11, None), (None, 0), (None, 5),
         (2, 2), (3, 4))
    ):
        _assert_value_error(index + 1, train_count, test_count)

    train_counts, test_counts, shapes, layouts, palettes = (
        set(), set(), set(), set(), set()
    )
    train_directions, test_directions, kinds = set(), set(), set()
    widths, lengths, thicknesses, gaps = set(), set(), set(), set()
    fixed_background, mixed_background = 0, 0
    fixed_width, mixed_width = 0, 0
    within_train_background, within_train_width = 0, 0
    wider_tips = corners = split_edges = positive_margins = 0
    zero_gaps = edge_barriers = clear_barriers = overlap_barriers = 0
    test_edge = test_nonedge = 0
    combined = 0
    varied_test_directions = varied_test_kinds = varied_test_endpoints = 0
    puzzles = []

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        split_analyses = {}
        for split in ("train", "test"):
            analyses = []
            for pair in puzzle[split]:
                input_grid, output_grid = pair["input"], pair["output"]
                analysis = _analyze_pair(input_grid, output_grid,
                                         "source_train" if split == "train" else "target_test")
                analyses.append(analysis)
                assert np.array_equal(_derive_output_from_input(input_grid.copy()),
                                      output_grid)
                assert not _matches_source_pair(input_grid, output_grid)
                shapes.add(tuple(input_grid.shape))
                layouts.add(_layout_key(analysis))
                palettes.add((analysis["background"], analysis["launcher_color"],
                              analysis["projectile_color"], analysis["barrier_color"]))
                kinds.add(analysis["launcher_kind"])
                widths.add(analysis["projectile_width"])
                lengths.add(analysis["projectile_length"])
                thicknesses.add(analysis["barrier_thickness"])
                gaps.add(analysis["gap"])
                wider_tips += analysis["wider_tip"]
                corners += analysis["launcher_touches_corner"]
                split_edges += analysis["split_touches_side"]
                positive_margins += not analysis["split_touches_side"]
                zero_gaps += analysis["gap"] == 0
                edge_barriers += analysis["tail"] == 0
                clear_barriers += analysis["barrier_clear_of_destination"]
                overlap_barriers += (not analysis["barrier_clear_of_destination"]
                                     and analysis["tail"] > 0)
                combined += analysis["wider_tip"] and analysis["split_touches_side"]
                if split == "train":
                    train_directions.add(analysis["direction"])
                else:
                    test_directions.add(analysis["direction"])
                    test_edge += analysis["tail"] == 0
                    test_nonedge += analysis["tail"] > 0
            split_analyses[split] = analyses
        all_analyses = split_analyses["train"] + split_analyses["test"]
        all_backgrounds = {a["background"] for a in all_analyses}
        all_widths = {a["projectile_width"] for a in all_analyses}
        fixed_background += len(all_backgrounds) == 1
        mixed_background += len(all_backgrounds) > 1
        fixed_width += len(all_widths) == 1
        mixed_width += len(all_widths) > 1
        within_train_background += len({a["background"] for a in split_analyses["train"]}) > 1
        within_train_width += len({a["projectile_width"] for a in split_analyses["train"]}) > 1
        tests = split_analyses["test"]
        varied_test_directions += len({a["direction"] for a in tests}) > 1
        varied_test_kinds += len({a["launcher_kind"] for a in tests}) > 1
        varied_test_endpoints += len({a["tail"] == 0 for a in tests}) > 1

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len(shapes) >= 25 and len(layouts) >= 100 and len(palettes) >= 60
    assert train_directions == test_directions == set(_DIRECTIONS)
    assert kinds == {"block", "shoulder", "taper"}
    assert widths == set(_PROJECTILE_WIDTHS)
    assert lengths == {1, 2, 3, 4, 5}
    assert len(thicknesses) >= 4 and len(gaps) >= 7
    assert min(fixed_background, mixed_background, fixed_width, mixed_width) >= 5
    assert min(within_train_background, within_train_width) >= 5
    assert min(wider_tips, corners, split_edges, positive_margins, combined) >= 3
    assert min(zero_gaps, edge_barriers, clear_barriers, overlap_barriers) >= 3
    assert min(test_edge, test_nonedge) >= 3
    assert min(varied_test_directions, varied_test_kinds, varied_test_endpoints) >= 2
