import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

_SOURCE_PAIR_ROWS = (
    (
        (
            "7727777",
            "7722277",
            "7777277",
            "2277222",
            "7277777",
            "7222277",
            "7777277",
        ),
        (
            "3325555",
            "3322255",
            "3333255",
            "2233222",
            "5233333",
            "5222233",
            "5555233",
        ),
    ),
    (
        (
            "2777777",
            "2777777",
            "2777777",
            "2227777",
            "7727777",
            "7722277",
            "7777222",
        ),
        (
            "2333333",
            "2333333",
            "2333333",
            "2223333",
            "5523333",
            "5522233",
            "5555222",
        ),
    ),
    (
        (
            "77777277",
            "77777277",
            "22227277",
            "77727277",
            "77727277",
            "22727227",
            "72727722",
            "72727777",
        ),
        (
            "55555233",
            "55555233",
            "22225233",
            "33325233",
            "33325233",
            "22325223",
            "52325522",
            "52325555",
        ),
    ),
    (
        (
            "777777777",
            "777777777",
            "222222227",
            "777777727",
            "222222722",
            "777772777",
            "222272277",
            "777277222",
            "777277777",
        ),
        (
            "333333333",
            "333333333",
            "222222223",
            "555555523",
            "222222522",
            "333332555",
            "222232255",
            "555233222",
            "555233333",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


def _grid_signature(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int8)
    return tuple(int(x) for x in normalized.shape), normalized.tobytes(order="C")


def _pair_signature(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _grid_signature(input_grid), _grid_signature(output_grid)


def _is_exact_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for source_input_rows, source_output_rows in _SOURCE_PAIR_ROWS:
        source_input = _rows_to_grid(source_input_rows)
        source_output = _rows_to_grid(source_output_rows)
        if (
            input_grid.shape == source_input.shape
            and output_grid.shape == source_output.shape
            and np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _components(mask: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    height, width = mask.shape
    labels = np.full((height, width), -1, dtype=np.int16)
    components: list[list[tuple[int, int]]] = []

    for start_row in range(height):
        for start_col in range(width):
            if not bool(mask[start_row, start_col]):
                continue
            if int(labels[start_row, start_col]) >= 0:
                continue

            component_id = len(components)
            labels[start_row, start_col] = component_id
            stack = [(start_row, start_col)]
            coordinates: list[tuple[int, int]] = []

            while stack:
                row, col = stack.pop()
                coordinates.append((row, col))

                for delta_row, delta_col in _NEIGHBORS4:
                    next_row = row + delta_row
                    next_col = col + delta_col
                    if not (
                        0 <= next_row < height
                        and 0 <= next_col < width
                    ):
                        continue
                    if not bool(mask[next_row, next_col]):
                        continue
                    if int(labels[next_row, next_col]) >= 0:
                        continue
                    labels[next_row, next_col] = component_id
                    stack.append((next_row, next_col))

            components.append(coordinates)

    return labels, components


def _longest_horizontal_run(labels: np.ndarray, component_id: int) -> int:
    height, width = labels.shape
    best = 0

    for row in range(height):
        run = 0
        for col in range(width):
            if int(labels[row, col]) == component_id:
                run += 1
                best = max(best, run)
            else:
                run = 0

    return best


def _geometry_info(
    input_grid: np.ndarray,
    field_color: int,
    divider_color: int,
) -> dict:
    height, width = input_grid.shape
    valid_input = (
        (input_grid == field_color)
        | (input_grid == divider_color)
    )
    _require(np.all(valid_input), "input has an unexpected color")

    wall_mask = input_grid == divider_color
    _require(np.any(wall_mask), "input must contain at least one divider")
    _require(
        not bool(wall_mask[height - 1, 0]),
        "the lower-left cell must be a field cell",
    )

    wall_labels, wall_components = _components(wall_mask)
    endpoint_pairs: list[
        tuple[tuple[int, int], tuple[int, int]]
    ] = []

    for component_id, component in enumerate(wall_components):
        _require(len(component) >= 2, "a divider path is too short")
        endpoints: list[tuple[int, int]] = []
        boundary_cells: list[tuple[int, int]] = []

        for row, col in component:
            degree = 0
            for delta_row, delta_col in _NEIGHBORS4:
                next_row = row + delta_row
                next_col = col + delta_col
                if (
                    0 <= next_row < height
                    and 0 <= next_col < width
                    and int(wall_labels[next_row, next_col]) == component_id
                ):
                    degree += 1

            _require(
                degree in (1, 2),
                "a divider branches, cycles, or touches itself orthogonally",
            )
            if degree == 1:
                endpoints.append((row, col))

            if (
                row == 0
                or row == height - 1
                or col == 0
                or col == width - 1
            ):
                boundary_cells.append((row, col))

        _require(
            len(endpoints) == 2,
            "every divider must be one simple unbranched path",
        )
        _require(
            all(row in (0, height - 1) or col in (0, width - 1)
                for row, col in endpoints),
            "divider endpoints must lie on the boundary",
        )
        endpoint_pairs.append((endpoints[0], endpoints[1]))

    # Different dividers may not even touch diagonally.
    for row in range(height):
        for col in range(width):
            component_id = int(wall_labels[row, col])
            if component_id < 0:
                continue
            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row == 0 and delta_col == 0:
                        continue
                    next_row = row + delta_row
                    next_col = col + delta_col
                    if not (
                        0 <= next_row < height
                        and 0 <= next_col < width
                    ):
                        continue
                    other_id = int(wall_labels[next_row, next_col])
                    _require(
                        other_id < 0 or other_id == component_id,
                        "distinct dividers touch each other",
                    )

    region_labels, regions = _components(~wall_mask)
    divider_count = len(wall_components)
    region_count = len(regions)

    _require(
        region_count == divider_count + 1,
        "dividers must partition the field into one extra region each",
    )

    edges: list[tuple[int, int]] = []
    adjacency: list[list[int]] = [[] for _ in range(region_count)]

    for component in wall_components:
        neighboring_regions: set[int] = set()

        for row, col in component:
            for delta_row, delta_col in _NEIGHBORS4:
                next_row = row + delta_row
                next_col = col + delta_col
                if not (
                    0 <= next_row < height
                    and 0 <= next_col < width
                ):
                    continue
                region_id = int(region_labels[next_row, next_col])
                if region_id >= 0:
                    neighboring_regions.add(region_id)

        _require(
            len(neighboring_regions) == 2,
            "each divider must separate exactly two field regions",
        )
        first, second = sorted(neighboring_regions)
        edges.append((first, second))
        adjacency[first].append(second)
        adjacency[second].append(first)

    root_region = int(region_labels[height - 1, 0])
    _require(root_region >= 0, "lower-left anchor region is missing")

    depths = [-1] * region_count
    depths[root_region] = 0
    queue = [root_region]
    queue_index = 0

    while queue_index < len(queue):
        region_id = queue[queue_index]
        queue_index += 1

        for neighbor in adjacency[region_id]:
            if depths[neighbor] < 0:
                depths[neighbor] = depths[region_id] + 1
                queue.append(neighbor)
            else:
                _require(
                    (depths[neighbor] - depths[region_id]) % 2 != 0,
                    "the region graph is not parity colorable",
                )

    _require(
        all(depth >= 0 for depth in depths),
        "the region-adjacency graph is disconnected",
    )
    _require(
        len(edges) == region_count - 1,
        "the region-adjacency graph must be a tree",
    )

    for first, second in edges:
        _require(
            abs(depths[first] - depths[second]) == 1,
            "a divider edge has inconsistent crossing depth",
        )

    has_left_to_right = False
    has_top_to_bottom = False

    for first_endpoint, second_endpoint in endpoint_pairs:
        first_row, first_col = first_endpoint
        second_row, second_col = second_endpoint

        if (
            (first_col == 0 and second_col == width - 1)
            or (second_col == 0 and first_col == width - 1)
        ):
            has_left_to_right = True

        if (
            (first_row == 0 and second_row == height - 1)
            or (second_row == 0 and first_row == height - 1)
        ):
            has_top_to_bottom = True

    longest_horizontal = 0
    for component_id in range(divider_count):
        longest_horizontal = max(
            longest_horizontal,
            _longest_horizontal_run(wall_labels, component_id),
        )

    turn_counts = []
    boundary_counts = []
    corner_endpoint_count = 0
    for component, endpoints in zip(wall_components, endpoint_pairs):
        cells = set(component)
        current, previous = endpoints[0], None
        ordered = [current]
        while current != endpoints[1]:
            choices = [
                (current[0] + dr, current[1] + dc)
                for dr, dc in _NEIGHBORS4
                if (current[0] + dr, current[1] + dc) in cells
                and (current[0] + dr, current[1] + dc) != previous
            ]
            _require(len(choices) == 1, "divider path traversal is ambiguous")
            previous, current = current, choices[0]
            ordered.append(current)
        directions = [
            (b[0] - a[0], b[1] - a[1])
            for a, b in zip(ordered, ordered[1:])
        ]
        turn_counts.append(sum(a != b for a, b in zip(directions, directions[1:])))
        boundary_counts.append(sum(
            row in (0, height - 1) or col in (0, width - 1)
            for row, col in component
        ))
        corner_endpoint_count += sum(
            row in (0, height - 1) and col in (0, width - 1)
            for row, col in endpoints
        )

    return {
        "height": height,
        "width": width,
        "divider_count": divider_count,
        "region_count": region_count,
        "region_labels": region_labels,
        "depths": tuple(depths),
        "root_region": root_region,
        "max_depth": max(depths),
        "has_left_to_right": has_left_to_right,
        "has_top_to_bottom": has_top_to_bottom,
        "longest_horizontal": longest_horizontal,
        "turn_counts": tuple(turn_counts),
        "boundary_counts": tuple(boundary_counts),
        "corner_endpoint_count": corner_endpoint_count,
        "max_region_degree": max(len(neighbors) for neighbors in adjacency),
    }


def _construct_output(
    input_grid: np.ndarray,
    field_color: int,
    divider_color: int,
    anchor_fill: int,
    alternate_fill: int,
) -> tuple[np.ndarray, dict]:
    info = _geometry_info(input_grid, field_color, divider_color)
    region_labels = info["region_labels"]
    depths = info["depths"]

    output_grid = np.full(
        input_grid.shape,
        divider_color,
        dtype=np.int8,
    )

    for region_id, depth in enumerate(depths):
        color = anchor_fill if depth % 2 == 0 else alternate_fill
        output_grid[region_labels == region_id] = color

    return output_grid, info


def _spaced_values(rng, low, high, count, gap=2):
    if count == 0:
        return []
    adjusted = high - (gap - 1) * (count - 1)
    _require(adjusted - low + 1 >= count, "insufficient space for dividers")
    picks = np.sort(rng.choice(np.arange(low, adjusted + 1), count, replace=False))
    return [int(x) for x in picks + np.arange(count) * (gap - 1)]


def _parts(rng, total, count):
    if count == 1:
        return [total]
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), count - 1, replace=False))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _corner_path(rng, height, width, corner, reach_row, reach_col, boundary=False):
    """A monotone staircase between adjacent sides of a corner box."""
    steps = int(rng.integers(1, min(4, reach_row, reach_col) + 1))
    horizontal = _parts(rng, reach_col, steps)
    vertical = _parts(rng, reach_row, steps)
    row, col = reach_row, 0
    path = [(row, col)]
    for dx, dy in zip(horizontal, vertical):
        for _ in range(dx):
            col += 1
            path.append((row, col))
        for _ in range(dy):
            row -= 1
            path.append((row, col))
    if boundary:
        # These are terminal stretches along the outer border. They may
        # make a corner an endpoint, as in the original second training pair.
        start_extension = (reach_row if int(rng.integers(3)) == 0
                           else int(rng.integers(1, reach_row + 1)))
        end_extension = int(rng.integers(0, min(3, width - reach_col)))
        path = [(r, 0) for r in range(reach_row - start_extension, reach_row)] + path
        path += [(0, c) for c in range(reach_col + 1, reach_col + end_extension + 1)]
    if corner in ("TR", "BR"):
        path = [(r, width - 1 - c) for r, c in path]
    if corner in ("BL", "BR"):
        path = [(height - 1 - r, c) for r, c in path]
    return path


def _through_path(rng, height, width, base, low, high, vertical=False, simple=False):
    """An x-monotone crossing with independent turn sites and row levels."""
    span = height if vertical else width
    max_turns = min(4, (span - 3) // 3)
    turns = 0 if simple else int(rng.integers(0, max_turns + 1))
    turn_columns = _spaced_values(rng, 2, span - 3, turns, gap=3)
    level = base
    points = [(level, 0)]
    column = 0
    for next_column in turn_columns + [span - 1]:
        for c in range(column + 1, next_column + 1):
            points.append((level, c))
        column = next_column
        if next_column == span - 1:
            break
        options = [r for r in range(low, high + 1) if r != level]
        if not options:
            continue
        new_level = int(rng.choice(options))
        step = 1 if new_level > level else -1
        for r in range(level + step, new_level + step, step):
            points.append((r, column))
        level = new_level
    if vertical:
        points = [(c, r) for r, c in points]
    return points


def _parallel_candidate(rng, height, width, count, vertical=False, simple=False):
    thickness, span = (width, height) if vertical else (height, width)
    bases = _spaced_values(rng, 2, thickness - 3, count, gap=4)
    paths = []
    for i, base in enumerate(bases):
        left_gap = base - bases[i - 1] if i else None
        right_gap = bases[i + 1] - base if i + 1 < count else None
        low = max(1, base - (left_gap - 2) // 2) if left_gap else max(1, base - 4)
        high = min(thickness - 2, base + (right_gap - 2) // 2) if right_gap else min(thickness - 2, base + 4)
        paths.append(_through_path(rng, height, width, base, low, high, vertical, simple))
    return paths


def _star_candidate(rng, height, width, count, include_bl=True, boundary=True):
    corners = ["TL", "TR", "BR", "BL"] if include_bl else ["TL", "TR", "BR"]
    chosen = [corners[int(i)] for i in rng.choice(len(corners), count, replace=False)]
    max_row = min(8, (height - 4) // 2)
    max_col = min(8, (width - 4) // 2)
    _require(max_row >= 2 and max_col >= 2, "star needs room")
    paths = []
    for corner in chosen:
        reach_row = int(rng.integers(2, max_row + 1))
        reach_col = int(rng.integers(2, max_col + 1))
        paths.append(_corner_path(rng, height, width, corner, reach_row, reach_col,
                                  boundary and corner != "BL" and int(rng.integers(3)) == 0))
    return paths


def _mixed_candidate(rng, height, width, count, anchor_demo=False):
    _require(count in (2, 3, 4, 5), "mixed count is out of range")
    middle = int(rng.integers(4, height - 4))
    # A crossing band plus cuts in its upper and lower field components.
    band = _through_path(rng, height, width, middle, middle, middle,
                         simple=anchor_demo or int(rng.integers(3)) == 0)
    if anchor_demo:
        chosen = ["BR"]
    else:
        corners = ["TL", "TR", "BR", "BL"]
        chosen = [corners[int(i)] for i in rng.choice(4, count - 1, replace=False)]
    paths = [band]
    max_col = min(8, (width - 4) // 2)
    for corner in chosen:
        max_row = middle - 2 if corner in ("TL", "TR") else height - middle - 3
        max_row = min(8, max_row)
        _require(max_row >= 2 and max_col >= 2, "corner cut needs room")
        row = int(rng.integers(2, max_row + 1))
        col = int(rng.integers(2, max_col + 1))
        paths.append(_corner_path(rng, height, width, corner, row, col,
                                  not anchor_demo and corner != "BL" and int(rng.integers(3)) == 0))
    return paths


def _u_candidate(rng, height, width):
    """A boundary-to-same-boundary cut with an open rectangular detour."""
    orientation = int(rng.integers(4))
    across, inward = (width, height) if orientation >= 2 else (height, width)
    first = int(rng.integers(1, across - 4))
    second = int(rng.integers(first + 3, across - 1))
    depth = int(rng.integers(2, inward - 1))
    path = [(first, c) for c in range(depth + 1)]
    path.extend((r, depth) for r in range(first + 1, second + 1))
    path.extend((second, c) for c in range(depth - 1, -1, -1))
    if orientation in (1, 3):
        path = [(r, inward - 1 - c) for r, c in path]
    if orientation >= 2:
        path = [(c, r) for r, c in path]
    return [path]


def _anchor_candidate(rng, height, width):
    # In either layout, TL, TR, and BR are alternate while BL is anchor.
    if height >= 12 and width >= 12 and int(rng.integers(2)):
        return _star_candidate(rng, height, width, 3, include_bl=False, boundary=False)
    return _mixed_candidate(rng, height, width, 2, anchor_demo=True)


def _sample_dimensions(rng, split_style, regime, anchor_demo=False):
    if split_style == "large_test":
        low, high = (10, 18) if regime == "source_train" else (19, 30)
    elif split_style == "small_test":
        low, high = (16, 30) if regime == "source_train" else (7, 15)
    else:
        low, high = (7, 30) if regime == "source_train" else (7, 30)
    if anchor_demo:
        low = max(low, 11)
    height = int(rng.integers(low, high + 1))
    width = int(rng.integers(low, high + 1))
    if not anchor_demo and high == 30:
        if int(rng.integers(12)) == 0:
            height = 30
        if int(rng.integers(12)) == 0:
            width = 30
    # An elongated rectangle is a real part of the distribution.
    if not anchor_demo and int(rng.integers(4)) == 0:
        if int(rng.integers(2)):
            height = int(rng.integers(low, min(high, low + 5) + 1))
        else:
            width = int(rng.integers(low, min(high, low + 5) + 1))
    return height, width


def _candidate(rng, height, width, pair_index, regime, split_style):
    if regime == "source_train" and pair_index == 0:
        return _anchor_candidate(rng, height, width)
    if regime == "source_train" and pair_index == 1:
        axis_vertical = bool(rng.integers(2))
        return _parallel_candidate(rng, height, width, 1, axis_vertical, simple=True)
    styles = ["parallel", "star", "mixed", "same_side"]
    style = styles[int(rng.choice(4, p=[0.42, 0.23, 0.25, 0.10]))]
    if style == "star" and (height < 8 or width < 8):
        style = "parallel"
    if style == "mixed" and (height < 9 or width < 8):
        style = "parallel"
    if style == "parallel":
        vertical = split_style == "vertical_test" and regime == "target_test"
        if not vertical:
            vertical = bool(rng.integers(2))
        thickness = width if vertical else height
        maximum = min(7, 1 + (thickness - 5) // 4)
        count = int(rng.integers(1, maximum + 1))
        return _parallel_candidate(rng, height, width, count, vertical)
    if style == "star":
        count = int(rng.integers(2, 5))
        return _star_candidate(rng, height, width, count)
    if style == "same_side":
        return _u_candidate(rng, height, width)
    count = int(rng.integers(2, 6))
    return _mixed_candidate(rng, height, width, count)


def _choose_input_colors(rng, fills, used_colors, fixed_colors):
    if fixed_colors is not None:
        return fixed_colors
    options = [c for c in range(10) if c not in fills]
    pairs = [(a, b) for a in options for b in options if a != b and (a, b) not in used_colors]
    choice = pairs[int(rng.integers(len(pairs)))]
    return choice


def _generate_unique_pair(rng, regime, pair_index, split_style, fills,
                          used_colors, fixed_colors, used_inputs):
    for _ in range(160):
        try:
            anchor_demo = regime == "source_train" and pair_index == 0
            height, width = _sample_dimensions(rng, split_style, regime, anchor_demo)
            paths = _candidate(rng, height, width, pair_index, regime, split_style)
            field, divider = _choose_input_colors(rng, fills, used_colors, fixed_colors)
            input_grid = np.full((height, width), field, dtype=np.int8)
            for path in paths:
                for row, col in path:
                    input_grid[row, col] = divider
            output_grid, info = _construct_output(input_grid, field, divider, *fills)
            if anchor_demo:
                for row, col in ((0, 0), (0, width - 1), (height - 1, width - 1)):
                    _require(input_grid[row, col] == field and output_grid[row, col] == fills[1],
                             "anchor demonstration does not distinguish other corners")
            if regime == "source_train" and pair_index == 1:
                _require(info["divider_count"] == 1 and info["max_depth"] == 1,
                         "simple training demonstration is invalid")
            if _is_exact_source_pair(input_grid, output_grid):
                continue
            signature = _grid_signature(input_grid)
            if signature in used_inputs:
                continue
            used_inputs.add(signature)
            if fixed_colors is None:
                used_colors.add((field, divider))
            return {"input": input_grid, "output": output_grid}
        except AssertionError:
            continue
    raise ValueError("could not generate a unique valid pair")


def _validate_count_value(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train_count = _validate_count_value("num_train", num_train, 2, 10)
    test_count = _validate_count_value("num_test", num_test, 1, 4)
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    fills = tuple(int(x) for x in rng.choice(10, 2, replace=False))
    split_style = ("large_test", "shared", "small_test", "vertical_test")[int(rng.choice(4, p=[.3, .35, .15, .2]))]
    fixed_colors = None
    if int(rng.integers(5)) == 0:
        options = [c for c in range(10) if c not in fills]
        chosen = rng.choice(options, 2, replace=False)
        fixed_colors = (int(chosen[0]), int(chosen[1]))
    used_inputs = set()
    used_colors = set()
    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=train_count + test_count, dtype=np.int64)
    train, test = [], []
    for index in range(train_count + test_count):
        local_rng = np.random.default_rng(int(child_seeds[index]))
        regime = "source_train" if index < train_count else "target_test"
        pair_index = index if regime == "source_train" else index - train_count
        pair = _generate_unique_pair(local_rng, regime, pair_index, split_style,
                                     fills, used_colors, fixed_colors, used_inputs)
        (train if regime == "source_train" else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_roles(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    field = int(input_grid[-1, 0])
    input_colors = set(int(x) for x in np.unique(input_grid))
    _require(len(input_colors) == 2 and field in input_colors, "input must have field and divider")
    divider = next(iter(input_colors - {field}))
    _require(np.array_equal(input_grid == divider, output_grid == divider),
             "divider is not preserved exactly")
    output_colors = set(int(x) for x in np.unique(output_grid))
    _require(len(output_colors) == 3, "output must have two fills and divider")
    anchor = int(output_grid[-1, 0])
    alternate_set = output_colors - {divider, anchor}
    _require(len(alternate_set) == 1, "alternate fill is not identifiable")
    alternate = next(iter(alternate_set))
    _require(len({field, divider, anchor, alternate}) == 4, "color roles overlap")
    return field, divider, anchor, alternate


def _check_pair(pair, fills):
    field, divider, anchor, alternate = _pair_roles(pair)
    _require((anchor, alternate) == fills, "output fills or phase differ across pairs")
    expected, info = _construct_output(pair["input"], field, divider, *fills)
    _require(np.array_equal(expected, pair["output"]), "incorrect region parity coloring")
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle schema is invalid")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "puzzle splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "pair counts are invalid")
    seen_inputs, seen_pairs = set(), set()
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair schema is invalid")
        for grid in (pair["input"], pair["output"]):
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                     "grid must be a two-dimensional NumPy array")
            _require(np.issubdtype(grid.dtype, np.integer), "grid colors must be integers")
            _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                     "grid dimensions are invalid")
            _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                     "grid colors are invalid")
        _require(pair["input"].shape == pair["output"].shape,
                 "input/output dimensions differ")
        input_sig = _grid_signature(pair["input"])
        pair_sig = _pair_signature(pair["input"], pair["output"])
        _require(input_sig not in seen_inputs and pair_sig not in seen_pairs,
                 "duplicate grid or pair")
        seen_inputs.add(input_sig)
        seen_pairs.add(pair_sig)
        _require(not _is_exact_source_pair(pair["input"], pair["output"]),
                 "source pair was copied")
    fills = _pair_roles(train[0])[2:]
    train_infos = [_check_pair(pair, fills) for pair in train]
    test_infos = [_check_pair(pair, fills) for pair in test]
    _require(train_infos[1]["divider_count"] == 1,
             "training must include a simple crossing")
    # Every competing corner phase is contradicted by the first demonstration.
    for row, col in ((0, 0), (0, -1), (-1, -1)):
        first = train[0]
        _require(int(first["input"][row, col]) != _pair_roles(first)[1]
                 and int(first["output"][row, col]) == fills[1],
                 "training does not establish lower-left anchoring")
    _require(all(info["region_count"] >= 2 for info in train_infos + test_infos),
             "a divider failed to separate regions")


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    for input_rows, output_rows in _SOURCE_PAIR_ROWS:
        source_input, source_output = _rows_to_grid(input_rows), _rows_to_grid(output_rows)
        actual, _ = _construct_output(source_input, 7, 2, 5, 3)
        _require(np.array_equal(actual, source_output), "source regression failed")
    first = generate_puzzle(991, 6, 3)
    _require(_puzzles_equal(first, generate_puzzle(991, 6, 3)), "determinism failed")
    _require(not _puzzles_equal(first, generate_puzzle(992, 6, 3)), "cross-seed variation failed")
    for seed, a, b in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                       (1101, None, 1), (1102, None, 4), (1103, 2, None),
                       (1104, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        if a is not None:
            _require(len(puzzle["train"]) == a, "explicit train count ignored")
        if b is not None:
            _require(len(puzzle["test"]) == b, "explicit test count ignored")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1200, **kwargs)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    counts_train, counts_test, shapes, layouts, palettes = set(), set(), set(), set(), set()
    turns, dividers, degrees, boundary, corner_ends, vertical = set(), set(), set(), 0, 0, 0
    variable_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        fills = _pair_roles(puzzle["train"][0])[2:]
        palettes.add(fills)
        input_roles = set()
        for pair in puzzle["train"] + puzzle["test"]:
            field, divider, _, _ = _pair_roles(pair)
            input_roles.add((field, divider))
            _, info = _construct_output(pair["input"], field, divider, *fills)
            shapes.add(pair["input"].shape)
            layouts.add((pair["input"].shape,
                         (pair["input"] == divider).tobytes()))
            turns.update(info["turn_counts"])
            dividers.add(info["divider_count"])
            degrees.add(info["max_region_degree"])
            boundary += sum(x > 2 for x in info["boundary_counts"])
            corner_ends += info["corner_endpoint_count"]
            vertical += info["has_top_to_bottom"]
        variable_palettes += len(input_roles) > 1
    _require(len(counts_train) >= 4 and len(counts_test) >= 3, "count diversity failed")
    _require(len(shapes) >= 35 and len(layouts) >= 150 and len(palettes) >= 12,
             "size, structure, or fill palette diversity failed")
    _require(max(turns) >= 5 and len(turns) >= 5, "turn diversity failed")
    _require(max(dividers) >= 5 and 1 in dividers, "divider count diversity failed")
    _require(max(degrees) >= 3, "branching region graph absent")
    _require(boundary > 0 and corner_ends > 0, "boundary configurations absent")
    _require(vertical > 0 and variable_palettes >= 15, "orientation or input palette variation absent")
