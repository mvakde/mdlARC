import numpy as np


_DIRECTIONS = (
    "left_to_right",
    "right_to_left",
    "top_to_bottom",
    "bottom_to_top",
)

_SOURCE_INPUT_ROWS = {
    (
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0002700000",
        "0002770000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000000000",
        "0000000000",
        "0000000000",
        "0002200000",
        "0006600000",
        "0000600000",
        "0000660000",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0008880000",
        "0000080000",
        "0000020000",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000000000",
        "0000000000",
        "0000000000",
        "0000420000",
        "0044420000",
        "0000420000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000000000",
        "0000000000",
        "0000000000",
        "0000100000",
        "0000112000",
        "0000012000",
        "0000000000",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.reshape(-1)),
    )


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            tuple(
                (_grid_key(pair["input"]), _grid_key(pair["output"]))
                for pair in puzzle[split]
            ),
        )
        for split in ("train", "test")
    )


def _same_puzzle(a: dict, b: dict) -> bool:
    if set(a) != {"train", "test"} or set(b) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(a[split]) != len(b[split]):
            return False

        for pair_a, pair_b in zip(a[split], b[split]):
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False

    return True


def _matches_source_input(grid: np.ndarray) -> bool:
    if grid.shape != (10, 10):
        return False

    rows = tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )
    return rows in _SOURCE_INPUT_ROWS


def _is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(int(x) for x in rc) for rc in np.argwhere(mask)]
    if not cells:
        return False

    cell_set = set(cells)
    seen = {cells[0]}
    stack = [cells[0]]

    while stack:
        row, column = stack.pop()
        neighbors = (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        )
        for neighbor in neighbors:
            if neighbor in cell_set and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)

    return len(seen) == len(cell_set)


def _consecutive(values: set[int]) -> bool:
    if not values:
        return False

    ordered = sorted(values)
    return ordered == list(range(ordered[0], ordered[-1] + 1))


def _reflect_mask(
    payload_mask: np.ndarray,
    direction: str,
    axis_sum: int,
) -> np.ndarray | None:
    height, width = payload_mask.shape
    result = payload_mask.copy()

    for row_value, column_value in np.argwhere(payload_mask):
        row = int(row_value)
        column = int(column_value)

        if direction in ("left_to_right", "right_to_left"):
            reflected_row = row
            reflected_column = axis_sum - column
        else:
            reflected_row = axis_sum - row
            reflected_column = column

        if not (
            0 <= reflected_row < height
            and 0 <= reflected_column < width
        ):
            return None

        result[reflected_row, reflected_column] = True

    return result


def _candidate_directions(
    payload_mask: np.ndarray,
    guide_mask: np.ndarray,
) -> list[tuple[str, int]]:
    payload_positions = np.argwhere(payload_mask)
    guide_positions = np.argwhere(guide_mask)
    candidates: list[tuple[str, int]] = []

    # A vertical guide column can indicate left-to-right or right-to-left.
    if np.all(guide_positions[:, 1] == guide_positions[0, 1]):
        guide_column = int(guide_positions[0, 1])
        guide_rows = {int(row) for row in guide_positions[:, 0]}

        if _consecutive(guide_rows):
            if np.all(payload_positions[:, 1] < guide_column):
                boundary_rows = {
                    int(row)
                    for row, column in payload_positions
                    if int(column) == guide_column - 1
                }
                if boundary_rows == guide_rows:
                    candidates.append(
                        ("left_to_right", 2 * guide_column - 1)
                    )

            if np.all(payload_positions[:, 1] > guide_column):
                boundary_rows = {
                    int(row)
                    for row, column in payload_positions
                    if int(column) == guide_column + 1
                }
                if boundary_rows == guide_rows:
                    candidates.append(
                        ("right_to_left", 2 * guide_column + 1)
                    )

    # A horizontal guide row can indicate top-to-bottom or bottom-to-top.
    if np.all(guide_positions[:, 0] == guide_positions[0, 0]):
        guide_row = int(guide_positions[0, 0])
        guide_columns = {int(column) for column in guide_positions[:, 1]}

        if _consecutive(guide_columns):
            if np.all(payload_positions[:, 0] < guide_row):
                boundary_columns = {
                    int(column)
                    for row, column in payload_positions
                    if int(row) == guide_row - 1
                }
                if boundary_columns == guide_columns:
                    candidates.append(
                        ("top_to_bottom", 2 * guide_row - 1)
                    )

            if np.all(payload_positions[:, 0] > guide_row):
                boundary_columns = {
                    int(column)
                    for row, column in payload_positions
                    if int(row) == guide_row + 1
                }
                if boundary_columns == guide_columns:
                    candidates.append(
                        ("bottom_to_top", 2 * guide_row + 1)
                    )

    return [
        candidate
        for candidate in candidates
        if _reflect_mask(
            payload_mask,
            candidate[0],
            candidate[1],
        ) is not None
    ]


def _shape_profiles(
    payload_mask: np.ndarray,
    direction: str,
) -> tuple[
    int,
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
]:
    positions = [
        (int(row), int(column))
        for row, column in np.argwhere(payload_mask)
    ]
    by_depth: dict[int, set[int]] = {}

    if direction == "left_to_right":
        boundary = max(column for _, column in positions)
        for row, column in positions:
            by_depth.setdefault(boundary - column, set()).add(row)

    elif direction == "right_to_left":
        boundary = min(column for _, column in positions)
        for row, column in positions:
            by_depth.setdefault(column - boundary, set()).add(row)

    elif direction == "top_to_bottom":
        boundary = max(row for row, _ in positions)
        for row, column in positions:
            by_depth.setdefault(boundary - row, set()).add(column)

    else:
        boundary = min(row for row, _ in positions)
        for row, column in positions:
            by_depth.setdefault(row - boundary, set()).add(column)

    depth = max(by_depth) + 1
    _require(
        set(by_depth) == set(range(depth)),
        "payload has an empty depth layer",
    )

    profiles = tuple(
        tuple(sorted(by_depth[layer]))
        for layer in range(depth)
    )
    tangent_minimum = min(
        tangent
        for profile in profiles
        for tangent in profile
    )
    normalized_profiles = tuple(
        tuple(tangent - tangent_minimum for tangent in profile)
        for profile in profiles
    )

    return depth, profiles, normalized_profiles


def _tangent_asymmetric(
    profiles: tuple[tuple[int, ...], ...],
) -> bool:
    cells = {
        (depth, tangent)
        for depth, profile in enumerate(profiles)
        for tangent in profile
    }
    tangent_minimum = min(tangent for _, tangent in cells)
    tangent_maximum = max(tangent for _, tangent in cells)

    reflected = {
        (depth, tangent_minimum + tangent_maximum - tangent)
        for depth, tangent in cells
    }
    return reflected != cells


def _distinguishes_alternatives(input_grid: np.ndarray, output_grid: np.ndarray, payload: int, guide: int) -> bool:
    source = input_grid == payload
    direction, axis_sum = _candidate_directions(source, input_grid == guide)[0]
    positions = [(int(r), int(c)) for r, c in np.argwhere(source)]
    normal_vertical = direction in ("top_to_bottom", "bottom_to_top")
    tangent_values = [c if normal_vertical else r for r, c in positions]
    tangent_sum = min(tangent_values) + max(tangent_values)
    turn = source.copy()
    shift = source.copy()
    step = 1 if direction in ("left_to_right", "top_to_bottom") else -1
    for r, c in positions:
        if normal_vertical:
            turn[axis_sum - r, tangent_sum - c] = True
            shift[r + step, c] = True
        else:
            turn[tangent_sum - r, axis_sum - c] = True
            shift[r, c + step] = True
    target = output_grid == payload
    return not np.array_equal(turn, target) and not np.array_equal(shift, target)


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    _require(
        input_grid.shape == output_grid.shape,
        "input and output dimensions differ",
    )

    input_values, input_counts = np.unique(
        input_grid,
        return_counts=True,
    )
    output_values = np.unique(output_grid)

    _require(
        len(input_values) == 3,
        "input must use exactly three colors",
    )
    _require(
        len(output_values) == 2,
        "output must use exactly two colors",
    )

    maximum_count = int(input_counts.max())
    background_candidates = [
        int(value)
        for value, count in zip(input_values, input_counts)
        if int(count) == maximum_count
    ]
    _require(
        len(background_candidates) == 1,
        "input background is not uniquely identifiable",
    )
    background = background_candidates[0]

    input_non_background = {
        int(value)
        for value in input_values
        if int(value) != background
    }
    output_colors = {int(value) for value in output_values}
    shared_colors = input_non_background & output_colors

    _require(
        len(shared_colors) == 1,
        "payload color is not uniquely identified across input and output",
    )

    payload = next(iter(shared_colors))
    guide = next(iter(input_non_background - {payload}))
    field = next(iter(output_colors - {payload}))

    _require(
        ({int(value) for value in input_values} & output_colors)
        == {payload},
        "background or guide improperly survives in the output",
    )
    _require(
        len({background, guide, field, payload}) == 4,
        "semantic color roles are not distinct",
    )

    payload_mask = input_grid == payload
    guide_mask = input_grid == guide

    _require(
        _is_connected(payload_mask),
        "payload is not one connected object",
    )

    # The role assignment must also be recoverable from the input alone.
    role_candidates = []
    for possible_payload in input_non_background:
        possible_guide = next(iter(input_non_background - {possible_payload}))
        mask = input_grid == possible_payload
        if not _is_connected(mask):
            continue
        for candidate in _candidate_directions(mask, input_grid == possible_guide):
            role_candidates.append((possible_payload, possible_guide, candidate))
    _require(len(role_candidates) == 1, "input does not identify payload and guide uniquely")
    _require(role_candidates[0][0] == payload, "input role inference disagrees with output")
    candidates = _candidate_directions(payload_mask, guide_mask)
    _require(
        len(candidates) == 1,
        "guide does not determine one unique fitted reflection axis",
    )

    direction, axis_sum = candidates[0]
    expected_payload = _reflect_mask(
        payload_mask,
        direction,
        axis_sum,
    )
    _require(
        expected_payload is not None,
        "reflected payload does not fit",
    )

    _require(
        np.array_equal(output_grid == payload, expected_payload),
        "output payload is not the source/reflection union",
    )
    _require(
        np.all(output_grid[~expected_payload] == field),
        "non-shape output cells are not uniformly field-colored",
    )
    _require(
        not np.array_equal(input_grid, output_grid),
        "input and output are degenerately identical",
    )

    depth, profiles, normalized_profiles = _shape_profiles(
        payload_mask,
        direction,
    )

    return {
        "background": background,
        "guide": guide,
        "field": field,
        "payload": payload,
        "direction": direction,
        "depth": depth,
        "profiles": profiles,
        "normalized_profiles": normalized_profiles,
        "payload_size": int(payload_mask.sum()),
        "guide_size": int(guide_mask.sum()),
        "height": int(input_grid.shape[0]),
        "width": int(input_grid.shape[1]),
        "edge_touch": bool(np.any(expected_payload[0]) or np.any(expected_payload[-1]) or np.any(expected_payload[:, 0]) or np.any(expected_payload[:, -1])),
        "split_layer": any(len(profile) != profile[-1] - profile[0] + 1 for profile in profiles),
    }


def _checked_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}"
        )

    return result


def _resolve_counts(
    seed_value: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count(
        "num_train",
        num_train,
        2,
        10,
    )
    test_count = _checked_count(
        "num_test",
        num_test,
        1,
        4,
    )

    if train_count is None and test_count is None:
        test_count = 1 + seed_value % 4
        minimum_train = max(2, test_count + 1)
        span = 10 - minimum_train + 1
        train_count = (
            minimum_train
            + (seed_value // 4) % span
        )

    elif train_count is None:
        minimum_train = max(2, int(test_count) + 1)
        span = 10 - minimum_train + 1
        train_count = minimum_train + seed_value % span

    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed_value % maximum_test

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return int(train_count), int(test_count)


def _sample_half_shape(rng: np.random.Generator, regime: str, require_asym: bool = False) -> set[tuple[int, int]]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    for _ in range(240):
        depth = int(rng.integers(2, 7 if regime == "source_train" else 9))
        mode = int(rng.integers(0, 6))
        boundary_length = int(rng.integers(1, min(9, 14 - depth)))
        cells = {(0, tangent) for tangent in range(boundary_length)}
        previous = set(range(boundary_length))
        if mode == 5:  # compact L and T motifs remain in the family
            depth = int(rng.integers(2, 4))
            if rng.random() < .5:
                cells = {(0, 0), (0, 1)} | {(layer, 0) for layer in range(1, depth)}
            else:
                cells = {(0, 0), (0, 1), (0, 2)} | {(layer, 1) for layer in range(1, depth)}
        elif mode == 0:  # familiar rectangles and bars
            for layer in range(1, depth):
                cells.update((layer, tangent) for tangent in previous)
        elif mode == 3 and boundary_length >= 3 and depth >= 3:
            cells = {(layer, tangent) for layer in range(depth) for tangent in range(boundary_length)}
            attempts = int(rng.integers(1, min(7, (depth - 1) * (boundary_length - 2) + 1)))
            for _hole in range(attempts):
                location = (int(rng.integers(1, depth)), int(rng.integers(1, boundary_length - 1)))
                candidate = cells - {location}
                local = np.zeros((depth, boundary_length), dtype=bool)
                for d, t in candidate:
                    local[d, t] = True
                if _is_connected(local):
                    cells = candidate
        else:
            for layer in range(1, depth):
                if mode == 4:  # thin bending path
                    center = int(rng.choice(sorted(previous))) + int(rng.integers(-1, 2))
                    current = {center}
                    if rng.random() < .35:
                        current.add(center + int(rng.choice((-1, 1))))
                elif mode == 2:  # growing branches, sometimes separated within a layer
                    anchors = rng.choice(sorted(previous), size=min(len(previous), int(rng.integers(1, 4))), replace=False)
                    current = {int(v) for v in anchors}
                    for _growth in range(int(rng.integers(0, 5))):
                        base = int(rng.choice(sorted(current)))
                        current.add(base + int(rng.choice((-1, 1))))
                else:  # overlapping, shifting strips
                    length = int(rng.integers(1, min(9, boundary_length + 4)))
                    anchor = int(rng.choice(sorted(previous)))
                    start = anchor - int(rng.integers(0, length))
                    current = set(range(start, start + length))
                cells.update((layer, tangent) for tangent in current)
                previous = current
        t_min = min(t for _, t in cells)
        cells = {(d, t - t_min) for d, t in cells}
        span = max(t for _, t in cells) + 1
        if span > 13 or len(cells) < depth + 1:
            continue
        local = np.zeros((depth, span), dtype=bool)
        for d, t in cells:
            local[d, t] = True
        if not _is_connected(local):
            continue
        profiles = tuple(tuple(int(t) for t in np.flatnonzero(local[d])) for d in range(depth))
        if require_asym and not _tangent_asymmetric(profiles):
            continue
        return cells
    # A small asymmetric L is valid for either split and for the witness pair.
    return {(0, 0), (0, 1), (1, 0)}


def _sample_dimensions(rng: np.random.Generator, normal_min: int, tangent_min: int) -> tuple[int, int]:
    # Normal dimension needs room for the complete two-sided reflection.
    mode = int(rng.integers(0, 5))
    if mode == 0:
        normal = normal_min + int(rng.integers(0, min(4, 31 - normal_min)))
        tangent = tangent_min + int(rng.integers(0, min(4, 31 - tangent_min)))
    elif mode == 1:
        normal = int(rng.integers(normal_min, min(21, 31))) if normal_min < 21 else normal_min
        tangent = int(rng.integers(tangent_min, min(21, 31)))
    elif mode == 2:
        normal = int(rng.integers(max(normal_min, 20), 31))
        tangent = int(rng.integers(max(tangent_min, 20), 31))
    elif mode == 3:
        normal = int(rng.integers(normal_min, min(19, 31))) if normal_min < 19 else normal_min
        tangent = int(rng.integers(max(tangent_min, 20), 31))
    else:
        normal = int(rng.integers(max(normal_min, 20), 31))
        tangent = int(rng.integers(tangent_min, min(19, 31))) if tangent_min < 19 else tangent_min
    return normal, tangent


def _render_pair(rng: np.random.Generator, regime: str, field: int, background: int, guide: int,
                 payload_color: int, direction: str, require_asym: bool = False) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        cells = _sample_half_shape(rng, regime, require_asym)
        depth = max(d for d, _ in cells) + 1
        span = max(t for _, t in cells) + 1
        normal_min = 2 * depth + 1
        tangent_min = span + 1
        if normal_min > 30 or tangent_min > 30:
            continue
        normal_size, tangent_size = _sample_dimensions(rng, normal_min, tangent_min)
        horizontal = direction in ("left_to_right", "right_to_left")
        height, width = (tangent_size, normal_size) if horizontal else (normal_size, tangent_size)
        # Endpoints are included, so the completed object can reach an edge or corner.
        block = int(rng.integers(0, normal_size - 2 * depth + 1))
        tangent_offset = int(rng.integers(0, tangent_size - span + 1))
        inp = np.full((height, width), background, dtype=np.int8)
        out = np.full((height, width), field, dtype=np.int8)
        if direction == "left_to_right":
            boundary = block + depth - 1
            guide_coordinate = boundary + 1
            axis_sum = 2 * boundary + 1
            locate = lambda d, t: (tangent_offset + t, boundary - d)
            guide_locate = lambda t: (tangent_offset + t, guide_coordinate)
        elif direction == "right_to_left":
            boundary = block + depth
            guide_coordinate = boundary - 1
            axis_sum = 2 * boundary - 1
            locate = lambda d, t: (tangent_offset + t, boundary + d)
            guide_locate = lambda t: (tangent_offset + t, guide_coordinate)
        elif direction == "top_to_bottom":
            boundary = block + depth - 1
            guide_coordinate = boundary + 1
            axis_sum = 2 * boundary + 1
            locate = lambda d, t: (boundary - d, tangent_offset + t)
            guide_locate = lambda t: (guide_coordinate, tangent_offset + t)
        else:
            boundary = block + depth
            guide_coordinate = boundary - 1
            axis_sum = 2 * boundary - 1
            locate = lambda d, t: (boundary + d, tangent_offset + t)
            guide_locate = lambda t: (guide_coordinate, tangent_offset + t)
        for d, t in cells:
            row, col = locate(d, t)
            inp[row, col] = payload_color
            out[row, col] = payload_color
            if horizontal:
                out[row, axis_sum - col] = payload_color
            else:
                out[axis_sum - row, col] = payload_color
            if d == 0:
                gr, gc = guide_locate(t)
                inp[gr, gc] = guide
        if _matches_source_input(inp):
            continue
        try:
            info = _analyze_pair(inp, out)
        except AssertionError:
            continue
        if info["direction"] == direction:
            return inp, out
    # Deterministic construction if repeated random proposals were unsuitable.
    inp = np.full((9, 9), background, dtype=np.int8)
    out = np.full((9, 9), field, dtype=np.int8)
    cells = ((0, 0), (0, 1), (1, 0))
    if direction == "left_to_right":
        place = lambda d, t: (3 + t, 3 - d)
        guide_place = lambda t: (3 + t, 4)
        mirror = lambda r, c: (r, 7 - c)
    elif direction == "right_to_left":
        place = lambda d, t: (3 + t, 5 + d)
        guide_place = lambda t: (3 + t, 4)
        mirror = lambda r, c: (r, 9 - c)
    elif direction == "top_to_bottom":
        place = lambda d, t: (3 - d, 3 + t)
        guide_place = lambda t: (4, 3 + t)
        mirror = lambda r, c: (7 - r, c)
    else:
        place = lambda d, t: (5 + d, 3 + t)
        guide_place = lambda t: (4, 3 + t)
        mirror = lambda r, c: (9 - r, c)
    for d, t in cells:
        r, c = place(d, t)
        inp[r, c] = payload_color
        out[r, c] = payload_color
        mr, mc = mirror(r, c)
        out[mr, mc] = payload_color
        if d == 0:
            gr, gc = guide_place(t)
            inp[gr, gc] = guide
    _analyze_pair(inp, out)
    return inp, out


def _direction_plan(rng: np.random.Generator, train_count: int, test_count: int) -> tuple[list[str], list[str]]:
    mode = int(rng.integers(0, 5))
    order = [_DIRECTIONS[int(i)] for i in rng.permutation(4)]
    if mode == 0:  # uniform-direction puzzles
        train = [order[0]] * train_count
    elif mode == 1:  # two demonstrated directions
        train = [order[i % 2] for i in range(train_count)]
    else:
        count = min(train_count, int(rng.integers(2, 5)))
        train = [order[i % count] for i in range(train_count)]
    train = [train[int(i)] for i in rng.permutation(train_count)]
    if mode == 2 and len(set(train)) < 4:  # geometrically inferable held-out direction
        unseen = [d for d in _DIRECTIONS if d not in train]
        test = [unseen[0]] + [train[int(rng.integers(0, train_count))] for _ in range(test_count - 1)]
    elif mode == 0:
        test = [train[0]] * test_count
    else:
        test = [order[int(rng.integers(0, 4))] for _ in range(test_count)]
    return train, test


def _choose_roles(rng: np.random.Generator, field: int, base_background: int,
                  base_guide: int, mode: int, index: int) -> tuple[int, int]:
    available = [c for c in range(10) if c != field]
    if mode in (1, 3):
        backgrounds = [c for c in available if c != base_guide]
        background = backgrounds[(index + int(rng.integers(0, len(backgrounds)))) % len(backgrounds)]
    else:
        background = base_background
    if mode in (2, 3):
        guides = [c for c in available if c != background]
        guide = guides[(index + int(rng.integers(0, len(guides)))) % len(guides)]
    else:
        guide = base_guide
        if background == guide:
            background = next(c for c in available if c != guide)
    return background, guide


def _new_pair(rng: np.random.Generator, regime: str, field: int, background: int,
              guide: int, payload: int, direction: str, seen: set[tuple],
              require_asym: bool = False) -> dict:
    for _ in range(30):
        child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        inp, out = _render_pair(child_rng, regime, field, background, guide, payload, direction, require_asym)
        key = _grid_key(inp)
        if key not in seen:
            seen.add(key)
            return {"input": inp, "output": out}
    raise RuntimeError("could not avoid a duplicate input")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    palette = [int(c) for c in rng.permutation(10)]
    base_background, base_guide, field = palette[:3]
    role_mode = int(rng.integers(0, 4))
    train_directions, test_directions = _direction_plan(rng, train_count, test_count)
    seen: set[tuple] = set()
    train, test = [], []
    train_payloads: list[int] = []
    for index, direction in enumerate(train_directions):
        background, guide = _choose_roles(rng, field, base_background, base_guide, role_mode, index)
        candidates = [c for c in range(10) if c not in (background, guide, field)]
        if index == 0:
            payload = int(rng.choice(candidates))
        elif index == 1:
            payload = int(rng.choice([c for c in candidates if c != train_payloads[0]]))
        else:
            payload = int(rng.choice(candidates))
        train_payloads.append(payload)
        train.append(_new_pair(rng, "source_train", field, background, guide, payload,
                               direction, seen, require_asym=(index == 0)))
    for index, direction in enumerate(test_directions):
        background, guide = _choose_roles(rng, field, base_background, base_guide, role_mode, train_count + index)
        candidates = [c for c in range(10) if c not in (background, guide, field)]
        familiar = [c for c in candidates if c in train_payloads]
        novel = [c for c in candidates if c not in train_payloads]
        pool = familiar if rng.random() < .5 and familiar else novel if novel else candidates
        payload = int(rng.choice(pool))
        test.append(_new_pair(rng, "target_test", field, background, guide, payload,
                              direction, seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    _require(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
             and len(puzzle["train"]) > len(puzzle["test"]), "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid is not a 2D ndarray")
                _require(all(1 <= n <= 30 for n in grid.shape), "illegal grid size")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype is not integral")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "illegal cell color")
            _require(not _matches_source_input(inp), "source-task input copied")
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            _require(key not in seen_inputs and pair_key not in seen_pairs, "duplicate input or pair")
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analyses[split].append(_analyze_pair(inp, out))
    all_info = analyses["train"] + analyses["test"]
    _require(len({a["field"] for a in all_info}) == 1, "output field changes within puzzle")
    _require(len({a["payload"] for a in analyses["train"]}) >= 2,
             "training does not demonstrate payload color independence")
    _require(any(_distinguishes_alternatives(pair["input"], pair["output"], a["payload"], a["guide"])
                 for pair, a in zip(puzzle["train"], analyses["train"])),
             "training does not distinguish reflection from rotation or translation")
    # All directions are identified by the input geometry, including any held-out direction.
    _require(all(a["depth"] >= 2 for a in all_info), "payload lacks a second depth layer")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    for seed, tr, te in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 7, 3),
                         (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, tr, te)
        _require(tr is None or len(puzzle["train"]) == tr, "train count ignored")
        _require(te is None or len(puzzle["test"]) == te, "test count ignored")
    _require(_same_puzzle(generate_puzzle(777, 8, 3), generate_puzzle(777, 8, 3)), "nondeterministic explicit puzzle")
    _require(_same_puzzle(generate_puzzle(778), generate_puzzle(778)), "nondeterministic automatic puzzle")
    _require(not _same_puzzle(generate_puzzle(779), generate_puzzle(780)), "seeds do not vary")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                      {"num_train": True}, {"num_train": 2.5}):
        try:
            generate_puzzle(900, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, "puzzle diversity is low")
    infos = [_analyze_pair(pair["input"], pair["output"])
             for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]]
    _require(len({(a["height"], a["width"]) for a in infos}) >= 45, "dimensions lack diversity")
    _require(len({(a["direction"], a["normalized_profiles"]) for a in infos}) >= 70,
             "shape layouts lack diversity")
    _require({a["direction"] for a in infos} == set(_DIRECTIONS), "directions lack diversity")
    _require(any(a["guide_size"] >= 5 for a in infos), "long guides absent")
    _require(any(a["split_layer"] for a in infos), "branched shapes absent")
    _require(any(a["edge_touch"] for a in infos), "edge placements absent")
    _require(any(a["height"] >= 25 or a["width"] >= 25 for a in infos), "large canvases absent")
    _require(any(a["height"] <= 7 or a["width"] <= 7 for a in infos), "compact canvases absent")
    _require(any(len({a["background"] for a in [_analyze_pair(pair["input"], pair["output"])
                    for split in ("train", "test") for pair in p[split]]}) > 1 for p in puzzles),
             "within-puzzle background variation absent")
    _require(any(len({a["guide"] for a in [_analyze_pair(pair["input"], pair["output"])
                    for split in ("train", "test") for pair in p[split]]}) > 1 for p in puzzles),
             "within-puzzle guide variation absent")
    _require(any(any(a["direction"] not in {b["direction"] for b in
                [_analyze_pair(pair["input"], pair["output"]) for pair in p["train"]]}
                for a in [_analyze_pair(pair["input"], pair["output"]) for pair in p["test"]])
                for p in puzzles), "held-out directions absent")
