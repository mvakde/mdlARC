import numpy as np
from heapq import heappop, heappush


_SOURCE_PAIR_ROWS = {
    (
        (
            "1111111111",
            "1111211111",
            "1111211111",
            "1188888811",
            "1111111111",
            "1212111111",
            "1111111111",
            "1111111111",
            "1111111111",
            "1111111111",
        ),
        (
            "1111111111",
            "1111211111",
            "1222211111",
            "1288888811",
            "1211111111",
            "1222111111",
            "1111111111",
            "1111111111",
            "1111111111",
            "1111111111",
        ),
    ),
    (
        (
            "1111111111",
            "1121111441",
            "1111111111",
            "1121111111",
            "1111111141",
            "1111111111",
            "1112221111",
            "1111111111",
            "8881111131",
            "1111111111",
        ),
        (
            "1111111111",
            "1121111441",
            "1121111141",
            "1121111141",
            "1121111141",
            "1121111111",
            "1122221111",
            "1111111111",
            "8881111131",
            "1111111111",
        ),
    ),
    (
        (
            "111111111111",
            "111919111111",
            "119111111111",
            "111111111112",
            "118888888811",
            "111116131112",
            "118888888811",
            "161111111111",
            "116211111111",
            "511171111111",
            "111111111111",
            "141411111111",
        ),
        (
            "111111111111",
            "119999111111",
            "119111111111",
            "111111111112",
            "118888888812",
            "166666131112",
            "168888888812",
            "161111111112",
            "166222222222",
            "511171111111",
            "111111111111",
            "144411111111",
        ),
    ),
    (
        (
            "1111111111",
            "1211211141",
            "1111181111",
            "1111212111",
            "0111111188",
            "1151511111",
            "1111111171",
            "1311111111",
        ),
        (
            "1111111111",
            "1222211141",
            "1111281111",
            "1111222111",
            "0111111188",
            "1155511111",
            "1111111171",
            "1311111111",
        ),
    ),
    (
        (
            "111111111111",
            "141111411171",
            "111118888881",
            "111111113111",
            "111611188888",
            "111111111111",
            "111111311111",
            "122111111111",
            "111111313111",
            "112111111111",
            "111122111191",
            "111111111111",
        ),
        (
            "111111111111",
            "144444411171",
            "111118888881",
            "111111333111",
            "111611388888",
            "111111311111",
            "111111311111",
            "122111311111",
            "112111333111",
            "112111111111",
            "112222111191",
            "111111111111",
        ),
    ),
}

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _neighbors(r, c, height, width):
    for dr, dc in _DIRS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < height and 0 <= cc < width:
            yield rr, cc


def _components(mask):
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    result = []

    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue

        stack = [(r, c)]
        seen[r, c] = True
        component = []

        while stack:
            x, y = stack.pop()
            component.append((x, y))

            for xx, yy in _neighbors(x, y, height, width):
                if mask[xx, yy] and not seen[xx, yy]:
                    seen[xx, yy] = True
                    stack.append((xx, yy))

        result.append(tuple(sorted(component)))

    return result


def _line(a, b):
    r1, c1 = a
    r2, c2 = b

    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        return [(r1, c) for c in range(c1, c2 + step, step)]

    if c1 == c2:
        step = 1 if r2 >= r1 else -1
        return [(r, c1) for r in range(r1, r2 + step, step)]

    return None


def _trace_keys(keys):
    sequence = [keys[0]]

    for a, b in zip(keys, keys[1:]):
        segment = _line(a, b)
        if segment is None:
            return None
        sequence.extend(segment[1:])

    if len(set(sequence)) != len(sequence):
        return None

    height = max(r for r, _ in sequence) + 1
    width = max(c for _, c in sequence) + 1
    cells = set(sequence)
    edge_count = 0

    for r, c in cells:
        edge_count += sum(
            (rr, cc) in cells
            for rr, cc in _neighbors(r, c, height, width)
        )

    if edge_count // 2 != len(sequence) - 1:
        return None

    return tuple(sequence)


def _bend_count(sequence):
    if len(sequence) < 3:
        return 0

    directions = [
        (
            sequence[i + 1][0] - sequence[i][0],
            sequence[i + 1][1] - sequence[i][1],
        )
        for i in range(len(sequence) - 1)
    ]

    return sum(a != b for a, b in zip(directions, directions[1:]))


def _canonical_path(sequence):
    sequence = tuple(sequence)
    reverse = tuple(reversed(sequence))
    return min(sequence, reverse)


def _reconstruct_color(input_grid, color, background, limit=None):
    """Find a unique least-added, least-bent simple route through all fragments.

    A visible fragment is traversed from one endpoint to the other. Search over
    blank cells between fragments, so a turn need not be visible in the input.
    ``limit`` is only an upper bound used by validation; it is never a route hint.
    """
    height, width = input_grid.shape
    fragments = []
    endpoint_owner = {}
    required = int(np.count_nonzero(input_grid == color))
    for component in _components(input_grid == color):
        mask = np.zeros(input_grid.shape, dtype=bool)
        for r, c in component:
            mask[r, c] = True
        ordered = _ordered_path(mask)
        if ordered is None:
            return None
        indices = tuple(r * width + c for r, c in ordered)
        owner = len(fragments)
        fragments.append(indices)
        endpoint_owner[indices[0]] = owner
        endpoint_owner[indices[-1]] = owner
    if len(fragments) <= 1:
        return tuple(sorted(_components(input_grid == color)[0]))
    if len(fragments) > 5:
        return None

    neighbors = []
    for index in range(height * width):
        r, c = divmod(index, width)
        neighbors.append(tuple(rr * width + cc for rr, cc in
                               _neighbors(r, c, height, width)))
    values = input_grid.ravel()
    all_mask = (1 << len(fragments)) - 1
    queue = []
    serial = 0

    def append_cells(path, visited, cells, added, bends, direction):
        for point in cells:
            if visited & (1 << point):
                return None
            previous = path[-1] if path else None
            if previous is not None:
                adjacent = sum(bool(visited & (1 << other)) for other in neighbors[point])
                if adjacent != 1 or previous not in neighbors[point]:
                    return None
                new_direction = (point // width - previous // width,
                                 point % width - previous % width)
                if direction is not None and new_direction != direction:
                    bends += 1
                direction = new_direction
            if int(values[point]) == background:
                added += 1
            visited |= 1 << point
            path += (point,)
        return path, visited, added, bends, direction

    for owner, fragment in enumerate(fragments):
        for oriented in (fragment, fragment[::-1]):
            if oriented is not fragment and oriented == fragment:
                continue
            result = append_cells((), 0, oriented, 0, 0, None)
            if result is None:
                continue
            path, visited, added, bends, direction = result
            heappush(queue, (added, bends, serial, 1 << owner,
                             path, visited, direction))
            serial += 1

    best_cost = None
    best_paths = set()
    expanded = 0
    while queue:
        added, bends, _, mask, path, visited, direction = heappop(queue)
        cost = (added, bends)
        if limit is not None and cost > limit:
            break
        if best_cost is not None and cost > best_cost:
            break
        if mask == all_mask:
            if len(path) != required + added:
                continue
            candidate = min(path, path[::-1])
            if best_cost is None:
                best_cost = cost
            best_paths.add(candidate)
            if len(best_paths) > 1:
                return None
            continue
        expanded += 1
        if expanded > 120000:
            return None
        current = path[-1]
        remaining = [i for i in range(len(fragments)) if not mask & (1 << i)]
        closest = min(abs(current // width - endpoint // width) +
                      abs(current % width - endpoint % width)
                      for i in remaining for endpoint in
                      (fragments[i][0], fragments[i][-1]))
        if limit is not None and added + max(0, closest - 1) > limit[0]:
            continue
        for point in neighbors[current]:
            if visited & (1 << point):
                continue
            value = int(values[point])
            if value == background:
                cells = (point,)
                new_mask = mask
            elif value == color and point in endpoint_owner:
                owner = endpoint_owner[point]
                if mask & (1 << owner):
                    continue
                fragment = fragments[owner]
                cells = fragment if point == fragment[0] else fragment[::-1]
                new_mask = mask | (1 << owner)
            else:
                continue
            result = append_cells(path, visited, cells, added, bends, direction)
            if result is None:
                continue
            new_path, new_visited, new_added, new_bends, new_dir = result
            if limit is not None and (new_added, new_bends) > limit:
                continue
            heappush(queue, (new_added, new_bends, serial, new_mask,
                             new_path, new_visited, new_dir))
            serial += 1
    if len(best_paths) != 1:
        return None
    return tuple(divmod(index, width) for index in next(iter(best_paths)))


def _ordered_path(mask):
    cells = {
        tuple(map(int, coordinate))
        for coordinate in np.argwhere(mask)
    }

    if not cells:
        return None

    height, width = mask.shape
    degrees = {
        cell: sum(
            neighbor in cells
            for neighbor in _neighbors(
                cell[0], cell[1], height, width
            )
        )
        for cell in cells
    }

    if len(cells) == 1:
        return (next(iter(cells)),)

    endpoints = sorted(
        cell for cell, degree in degrees.items()
        if degree == 1
    )

    if len(endpoints) != 2:
        return None
    if any(degree not in (1, 2) for degree in degrees.values()):
        return None
    if sum(degrees.values()) // 2 != len(cells) - 1:
        return None

    sequence = [endpoints[0]]
    previous = None
    current = endpoints[0]

    while current != endpoints[1]:
        next_cells = sorted(
            neighbor
            for neighbor in _neighbors(
                current[0], current[1], height, width
            )
            if neighbor in cells and neighbor != previous
        )

        if len(next_cells) != 1:
            return None

        previous, current = current, next_cells[0]
        sequence.append(current)

    if len(sequence) != len(cells):
        return None

    return tuple(sequence)


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _grid_rows(grid):
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _active_colors(input_grid, background, barrier):
    return [
        color
        for color in sorted(
            set(map(int, np.unique(input_grid)))
            - {background, barrier}
        )
        if len(_components(input_grid == color)) > 1
    ]


def _singleton_colors(input_grid, background, barrier):
    return [
        color
        for color in sorted(
            set(map(int, np.unique(input_grid)))
            - {background, barrier}
        )
        if len(_components(input_grid == color)) == 1
    ]


def _is_line_component(component):
    if len(component) == 1:
        return True

    return (
        len({r for r, _ in component}) == 1
        or len({c for _, c in component}) == 1
    )


def _infer_background(pairs):
    return _background_of(pairs[0]["input"])


def _background_of(input_grid):
    counts = np.bincount(input_grid.ravel(), minlength=10)
    modes = np.flatnonzero(counts == counts.max())
    assert len(modes) == 1
    return int(modes[0])


def _infer_barrier(train_pairs, background):
    candidates = []
    backgrounds = {_background_of(pair["input"]) for pair in train_pairs}

    for color in range(10):
        if color in backgrounds:
            continue

        valid = True

        saw_long = False
        saw_any = False
        for pair in train_pairs:
            input_grid = pair["input"]
            output_grid = pair["output"]

            if not np.any(input_grid == color):
                continue
            saw_any = True

            if not np.array_equal(
                input_grid == color,
                output_grid == color,
            ):
                valid = False
                break

            components = _components(input_grid == color)
            saw_long |= any(len(component) >= 4 for component in components)
            if (
                not components
                or any(
                    not _is_line_component(component)
                    for component in components
                )
            ):
                valid = False
                break

        if valid and saw_any and saw_long:
            candidates.append(color)

    assert len(candidates) == 1
    return candidates[0]


def _has_forced_detour(
    input_grid,
    output_grid,
    background,
    barrier,
):
    barrier_cells = set(map(tuple, np.argwhere(input_grid == barrier)))

    for color in _active_colors(
        input_grid,
        background,
        barrier,
    ):
        sequence = _ordered_path(output_grid == color)
        if sequence is None or len(sequence) < 2:
            continue

        direct = _line(sequence[0], sequence[-1])

        if (
            direct is not None
            and any(cell in barrier_cells for cell in direct)
            and set(direct) != set(sequence)
        ):
            return True

    return False


def _assert_pair_rule(
    input_grid,
    output_grid,
    background,
    barrier,
):
    assert input_grid.shape == output_grid.shape

    # Every original non-background cell is immutable.
    assert np.array_equal(
        output_grid[input_grid != background],
        input_grid[input_grid != background],
    )

    assert np.array_equal(
        input_grid == barrier,
        output_grid == barrier,
    )
    assert not np.any(
        (input_grid == background)
        & (output_grid == barrier)
    )
    assert not np.array_equal(input_grid, output_grid)

    input_colors = set(map(int, np.unique(input_grid)))
    output_colors = set(map(int, np.unique(output_grid)))
    assert output_colors.issubset(input_colors)

    active_count = 0

    for color in sorted(input_colors - {background, barrier}):
        components = _components(input_grid == color)
        assert components

        if len(components) == 1:
            assert np.array_equal(
                input_grid == color,
                output_grid == color,
            )
            continue

        active_count += 1
        ordered = _ordered_path(output_grid == color)
        assert ordered is not None
        inferred = _reconstruct_color(
            input_grid,
            color,
            background,
            (
                int(np.count_nonzero(output_grid == color))
                - int(np.count_nonzero(input_grid == color)),
                _bend_count(ordered),
            ),
        )
        assert inferred is not None

        actual_cells = set(
            map(tuple, np.argwhere(output_grid == color))
        )
        assert set(inferred) == actual_cells

        original_cells = set(
            map(tuple, np.argwhere(input_grid == color))
        )
        assert original_cells.issubset(set(ordered))

        assert np.any(
            (input_grid == background)
            & (output_grid == color)
        )

    assert active_count >= 1


def _sample_puzzle_parameters(rng):
    palette = [int(color) for color in rng.permutation(10)]
    ordinary = palette[3:]

    return {
        "backgrounds": palette[:2],
        "barrier": palette[2],
        # This color is singleton-only in training and active in tests.
        "switch": ordinary[0],
        # This color is active in one training pair and inert in another.
        "flex": ordinary[1],
        # Both anchors are visibly demonstrated in training before tests.
        "anchor_a": ordinary[2],
        "anchor_b": ordinary[3],
        "ordinary": ordinary,
        "switch_mode": bool(rng.integers(0, 2)),
    }


def _add_barrier(
    rng,
    input_grid,
    output_grid,
    barrier,
    background,
    min_length,
    max_length,
):
    height, width = input_grid.shape

    for _ in range(300):
        horizontal = bool(rng.integers(0, 2))
        limit = width if horizontal else height
        highest_length = min(max_length, limit)

        if highest_length < min_length:
            continue

        length = int(
            rng.integers(min_length, highest_length + 1)
        )

        if horizontal:
            r = int(rng.integers(0, height))
            c = int(rng.integers(0, width - length + 1))
            cells = [
                (r, x)
                for x in range(c, c + length)
            ]
        else:
            c = int(rng.integers(0, width))
            r = int(rng.integers(0, height - length + 1))
            cells = [
                (x, c)
                for x in range(r, r + length)
            ]

        cell_set = set(cells)

        if any(
            int(output_grid[r, c]) != background
            for r, c in cells
        ):
            continue

        touches_existing_barrier = False

        for r, c in cells:
            for rr, cc in _neighbors(
                r, c, height, width
            ):
                if (
                    int(input_grid[rr, cc]) == barrier
                    and (rr, cc) not in cell_set
                ):
                    touches_existing_barrier = True
                    break

            if touches_existing_barrier:
                break

        if touches_existing_barrier:
            continue

        for r, c in cells:
            input_grid[r, c] = barrier
            output_grid[r, c] = barrier

        return True

    return False


def _detour_geometry(rng, height, width, regime, turns=None):
    for _ in range(200):
        transpose = bool(rng.integers(0, 2))
        h, w = (
            (width, height)
            if transpose
            else (height, width)
        )

        minimum_length = 4 if regime == "source_train" else 2
        maximum_length = min(7, w - 2)

        if maximum_length < minimum_length or h < 8:
            continue

        length = int(
            rng.integers(
                minimum_length,
                maximum_length + 1,
            )
        )

        barrier_row = int(rng.integers(3, h - 3))
        barrier_start = int(
            rng.integers(1, w - length)
        )
        barrier_end = barrier_start + length - 1

        top = int(rng.integers(0, barrier_row - 2))
        bottom = int(
            rng.integers(barrier_row + 3, h)
        )
        route_column = int(
            rng.integers(
                barrier_start,
                barrier_end + 1,
            )
        )

        left_distance = route_column - barrier_start + 1
        right_distance = barrier_end - route_column + 1
        side_column = (barrier_start - 1 if left_distance < right_distance
                       else barrier_end + 1)
        if left_distance == right_distance:
            side_column = (barrier_start - 1 if bool(rng.integers(0, 2))
                           else barrier_end + 1)

        barrier_cells = [
            (barrier_row, c)
            for c in range(
                barrier_start,
                barrier_end + 1,
            )
        ]

        if turns == 3 or (turns is None and bool(rng.integers(0, 2))):
            keys = [
                (top, route_column),
                (barrier_row - 1, route_column),
                (barrier_row - 1, side_column),
                (bottom, side_column),
                (bottom, route_column),
            ]
        else:
            keys = [
                (top, route_column),
                (barrier_row - 1, route_column),
                (barrier_row - 1, side_column),
                (barrier_row + 1, side_column),
                (barrier_row + 1, route_column),
                (bottom, route_column),
            ]

        path = _trace_keys(keys)
        if path is None:
            continue

        if transpose:
            barrier_cells = [
                (c, r) for r, c in barrier_cells
            ]
            keys = [(c, r) for r, c in keys]
            path = tuple((c, r) for r, c in path)

        return (
            tuple(barrier_cells),
            tuple(keys),
            tuple(path),
        )

    return None


def _sample_polyline(rng, output_grid, background):
    height, width = output_grid.shape

    for _ in range(500):
        segment_count = int(rng.choice(np.array([1, 1, 2, 2, 3, 4])))
        start = (
            int(rng.integers(0, height)),
            int(rng.integers(0, width)),
        )
        direction = int(rng.integers(0, 4))

        keys = [start]
        r, c = start

        for segment_index in range(segment_count):
            if segment_index:
                if direction < 2:
                    direction = int(
                        rng.choice(np.array([2, 3]))
                    )
                else:
                    direction = int(
                        rng.choice(np.array([0, 1]))
                    )

            steps = int(
                rng.integers(
                    3 if segment_count == 1 else 2,
                    7 if segment_count == 1 else 5,
                )
            )

            dr, dc = _DIRS[direction]
            r += dr * steps
            c += dc * steps
            keys.append((r, c))

        if any(
            not (0 <= r < height and 0 <= c < width)
            for r, c in keys
        ):
            continue

        path = _trace_keys(keys)

        if path is None:
            continue

        if any(
            int(output_grid[r, c]) != background
            for r, c in path
        ):
            continue

        return tuple(path)

    return None


def _sample_checkpoints(rng, path, shape):
    length = len(path)
    if length < 4:
        return None
    visibility = rng.random()
    if visibility < 0.12:
        marked = {0, length - 1}
    elif visibility < 0.25:
        gap = int(rng.integers(1, length - 1))
        marked = set(range(length)) - {gap}
    else:
        marked = None
    # Runs at the ends make both terminals observable; middle fragments vary
    # independently of bend visibility.
    if marked is None:
        front = int(rng.integers(1, min(5, max(2, length // 3)) + 1))
        back = int(rng.integers(1, min(5, max(2, length // 3)) + 1))
        if front + back >= length - 1:
            front = back = 1
        marked = set(range(front)) | set(range(length - back, length))
        available = list(range(front + 2, length - back - 2))
        rng.shuffle(available)
        for center in available[:int(rng.integers(0, 3))]:
            run = int(rng.integers(1, 4))
            for index in range(center, min(center + run, length - back - 1)):
                marked.add(index)

    turns = []
    for index in range(1, length - 1):
        before = (path[index][0] - path[index - 1][0],
                  path[index][1] - path[index - 1][1])
        after = (path[index + 1][0] - path[index][0],
                 path[index + 1][1] - path[index][1])
        if before != after:
            turns.append(index)
    if visibility >= 0.25:
        for index in turns:
            if rng.random() < 0.36:
                marked.add(index)
    if visibility >= 0.25 and turns and rng.random() < 0.65 and all(index in marked for index in turns):
        removable = [index for index in turns if index not in (0, length - 1)]
        marked.remove(removable[int(rng.integers(0, len(removable)))])

    mask = np.zeros(shape, dtype=bool)
    for index in marked:
        mask[path[index]] = True
    components = _components(mask)
    if not 2 <= len(components) <= 5:
        return None
    for component in components:
        fragment_mask = np.zeros(shape, dtype=bool)
        for r, c in component:
            fragment_mask[r, c] = True
        if _ordered_path(fragment_mask) is None:
            return None
    return tuple(path[index] for index in sorted(marked))


def _choose_active_colors(rng, regime, index, parameters):
    switch = parameters["switch"]
    flex = parameters["flex"]
    anchor_a = parameters["anchor_a"]
    anchor_b = parameters["anchor_b"]

    if regime == "source_train":
        if index == 0:
            active = [flex]
            return active

        if index == 1:
            return [anchor_a]

        desired = int(rng.integers(1, 5))
        pool = [color for color in parameters["ordinary"]
                if not parameters["switch_mode"] or color != switch]
        rng.shuffle(pool)
        return pool[:desired]

    desired = int(rng.integers(1, 5))
    pool = [flex, anchor_a, anchor_b, switch]
    rng.shuffle(pool)
    if parameters["switch_mode"] and rng.random() < 0.55:
        pool.remove(switch)
        pool.insert(0, switch)
    return pool[:desired]


def _place_singletons(
    rng,
    input_grid,
    output_grid,
    colors,
    background,
):
    for color in colors:
        free = np.argwhere(output_grid == background)

        if len(free) == 0:
            return False

        r, c = map(
            int,
            free[int(rng.integers(0, len(free)))],
        )
        cells = [(r, c)]
        if rng.random() < 0.35:
            directions = list(_DIRS)
            rng.shuffle(directions)
            for dr, dc in directions:
                length = int(rng.integers(2, 4))
                candidate = [(r + dr * k, c + dc * k) for k in range(length)]
                if all(0 <= rr < input_grid.shape[0] and
                       0 <= cc < input_grid.shape[1] and
                       int(output_grid[rr, cc]) == background
                       for rr, cc in candidate):
                    cells = candidate
                    break
        for rr, cc in cells:
            input_grid[rr, cc] = color
            output_grid[rr, cc] = color

    return True


def _build_pair_once(rng, regime, index, parameters):
    if regime == "source_train":
        height = int(rng.integers(10, 14))
        width = int(rng.integers(10, 14))
    else:
        height = int(rng.integers(8, 16))
        width = int(rng.integers(8, 16))

    if regime == "source_train" and index < 2:
        background = parameters["backgrounds"][index]
    else:
        background = parameters["backgrounds"][int(rng.integers(0, 2))]
    barrier = parameters["barrier"]

    input_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )
    output_grid = input_grid.copy()

    active_colors = _choose_active_colors(
        rng,
        regime,
        index,
        parameters,
    )
    routes = []
    special_detour = ((regime == "source_train" and index == 0)
                      or rng.random() < (0.20 if regime == "source_train" else 0.30))
    if special_detour:
        geometry = _detour_geometry(
            rng, height, width, regime,
            4 if regime == "source_train" and index == 0 else None,
        )
        if geometry is None:
            return None
        barrier_cells, _, path = geometry
        for r, c in barrier_cells:
            input_grid[r, c] = barrier
            output_grid[r, c] = barrier
        color = active_colors[0]
        if any(int(output_grid[r, c]) != background for r, c in path):
            return None
        for r, c in path:
            output_grid[r, c] = color
        routes.append((color, path))

    detour_routes = int(special_detour)
    if (special_detour and len(active_colors) >= 2
            and min(height, width) >= 12 and rng.random() < 0.55):
        for _ in range(40):
            second = _detour_geometry(rng, height, width, regime, 3)
            if second is None:
                break
            second_barrier, _, second_path = second
            if any(int(output_grid[r, c]) != background
                   for r, c in second_barrier + second_path):
                continue
            if any(int(input_grid[rr, cc]) == barrier
                   for r, c in second_barrier
                   for rr, cc in _neighbors(r, c, height, width)):
                continue
            for r, c in second_barrier:
                input_grid[r, c] = barrier
                output_grid[r, c] = barrier
            for r, c in second_path:
                output_grid[r, c] = active_colors[1]
            routes.append((active_colors[1], second_path))
            detour_routes = 2
            break

    wanted_barriers = int(rng.integers(0, 4))
    if regime == "source_train" and index == 0:
        wanted_barriers = max(1, wanted_barriers)
    if regime == "source_train" and index == 1:
        wanted_barriers = max(1, wanted_barriers)
    wanted_barriers = max(detour_routes, wanted_barriers)
    for _ in range(wanted_barriers - detour_routes):
        if not _add_barrier(
            rng, input_grid, output_grid, barrier, background, 1, 7,
        ):
            return None
    route_start = detour_routes
    for color in active_colors[route_start:]:
        path = _sample_polyline(rng, output_grid, background)
        if path is None:
            return None
        for r, c in path:
            output_grid[r, c] = color
        routes.append((color, path))

    ordinary = parameters["ordinary"]
    if regime == "source_train":
        singleton_colors = []
        if index == 1:
            singleton_colors.append(parameters["flex"])
            if parameters["switch_mode"]:
                singleton_colors.append(parameters["switch"])
        elif parameters["switch_mode"] and index > 1 and rng.random() < 0.55:
            singleton_colors.append(parameters["switch"])
        remaining = [color for color in ordinary
                     if color not in active_colors and color not in singleton_colors]
        rng.shuffle(remaining)
        extra = int(rng.integers(0, min(4, len(remaining)) + 1))
        singleton_colors.extend(remaining[:extra])
    else:
        remaining = [color for color in ordinary if color not in active_colors]
        rng.shuffle(remaining)
        singleton_colors = remaining[:int(rng.integers(0, min(4, len(remaining)) + 1))]
    if not _place_singletons(
        rng, input_grid, output_grid, singleton_colors, background,
    ):
        return None

    # Resample the visible fragments without changing the latent completed
    # routes.  Each accepted mask is checked against an independent search.
    base_input = input_grid.copy()
    for _ in range(16):
        input_grid = base_input.copy()
        possible = True
        for color, path in routes:
            checkpoints = _sample_checkpoints(rng, path, input_grid.shape)
            if checkpoints is None:
                possible = False
                break
            for r, c in checkpoints:
                input_grid[r, c] = color
        if not possible:
            continue
        counts = np.bincount(input_grid.ravel(), minlength=10)
        if np.count_nonzero(counts == counts.max()) != 1 or int(np.argmax(counts)) != background:
            continue
        try:
            _assert_pair_rule(input_grid, output_grid, background, barrier)
        except AssertionError:
            continue
        return input_grid.astype(np.int8), output_grid.astype(np.int8)
    return None


def _generate_pair(
    rng,
    regime,
    index,
    parameters,
):
    for _ in range(600):
        result = _build_pair_once(
            rng,
            regime,
            index,
            parameters,
        )
        if result is not None:
            return result

    raise ValueError(
        "unable to construct an unambiguous pair"
    )


def _validate_requested(name, value, low, high):
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(
            f"{name} must be an integer or None"
        )

    value = int(value)

    if not low <= value <= high:
        raise ValueError(
            f"{name} is outside its legal range"
        )

    return value


def _resolve_counts(
    seed,
    rng,
    num_train,
    num_test,
):
    num_train = _validate_requested(
        "num_train",
        num_train,
        2,
        10,
    )
    num_test = _validate_requested(
        "num_test",
        num_test,
        1,
        4,
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = (
            num_test
            + 1
            + (
                (seed // 4 + 2 * seed)
                % (10 - num_test)
            )
        )
    elif num_train is None:
        num_train = int(
            rng.integers(
                max(2, num_test + 1),
                11,
            )
        )
    elif num_test is None:
        num_test = int(
            rng.integers(
                1,
                min(4, num_train - 1) + 1,
            )
        )

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError(
            "pair counts must satisfy "
            "2 <= train <= 10, "
            "1 <= test <= 4, and train > test"
        )

    return num_train, num_test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError(
            "seed must be a nonnegative integer"
        )

    seed = int(seed)

    if seed < 0:
        raise ValueError(
            "seed must be a nonnegative integer"
        )

    rng = np.random.default_rng(seed)

    num_train, num_test = _resolve_counts(
        seed,
        rng,
        num_train,
        num_test,
    )

    parameters = _sample_puzzle_parameters(rng)

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs = set()

    generation_groups = (
        (
            "source_train",
            num_train,
            puzzle["train"],
        ),
        (
            "target_test",
            num_test,
            puzzle["test"],
        ),
    )

    for regime, count, destination in generation_groups:
        for index in range(count):
            for _ in range(100):
                child_seed = int(
                    rng.integers(
                        0,
                        np.iinfo(np.int64).max,
                    )
                )
                pair_rng = np.random.default_rng(
                    child_seed
                )

                input_grid, output_grid = _generate_pair(
                    pair_rng,
                    regime,
                    index,
                    parameters,
                )

                key = _grid_key(input_grid)

                if key not in seen_inputs:
                    seen_inputs.add(key)
                    destination.append(
                        {
                            "input": input_grid,
                            "output": output_grid,
                        }
                    )
                    break
            else:
                raise ValueError(
                    "unable to make all puzzle inputs distinct"
                )

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    all_pairs = puzzle["train"] + puzzle["test"]
    input_keys = set()
    pair_keys = set()

    for pair in all_pairs:
        assert isinstance(pair, dict)
        assert set(pair) == {"input", "output"}

        for grid in (
            pair["input"],
            pair["output"],
        ):
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30
            assert 1 <= grid.shape[1] <= 30
            assert np.issubdtype(
                grid.dtype,
                np.integer,
            )
            assert np.all((0 <= grid) & (grid <= 9))

        input_key = _grid_key(pair["input"])
        pair_key = (
            input_key,
            _grid_key(pair["output"]),
        )

        assert input_key not in input_keys
        assert pair_key not in pair_keys

        input_keys.add(input_key)
        pair_keys.add(pair_key)

        source_signature = (
            _grid_rows(pair["input"]),
            _grid_rows(pair["output"]),
        )
        assert source_signature not in _SOURCE_PAIR_ROWS

    # Infer puzzle conventions exclusively from training evidence.
    train_backgrounds = {_background_of(pair["input"]) for pair in puzzle["train"]}
    assert len(train_backgrounds) == 2
    assert _background_of(puzzle["train"][0]["input"]) != _background_of(puzzle["train"][1]["input"])
    assert all(_background_of(pair["input"]) in train_backgrounds
               for pair in puzzle["test"])

    barrier = _infer_barrier(
        puzzle["train"],
        next(iter(train_backgrounds)),
    )
    assert all(np.any(pair["input"] == barrier) for pair in puzzle["train"][:2])

    for pair in all_pairs:
        background = _background_of(pair["input"])
        _assert_pair_rule(
            pair["input"],
            pair["output"],
            background,
            barrier,
        )

        barrier_components = _components(pair["input"] == barrier)
        assert all(
            _is_line_component(component)
            for component in barrier_components
        )

    train_active_by_color = {
        color: [] for color in range(10)
    }
    train_inert_by_color = {
        color: [] for color in range(10)
    }

    for pair in puzzle["train"]:
        input_grid = pair["input"]
        height, width = input_grid.shape
        background = _background_of(input_grid)

        assert 10 <= height <= 13
        assert 10 <= width <= 13

        barrier_components = _components(
            input_grid == barrier
        )
        assert 0 <= len(barrier_components) <= 3

        active = _active_colors(
            input_grid,
            background,
            barrier,
        )
        assert 1 <= len(active) <= 4

        assert 0 <= len(_singleton_colors(input_grid, background, barrier)) <= 7

        ordinary_colors = (
            set(map(int, np.unique(input_grid)))
            - {background, barrier}
        )

        for color in ordinary_colors:
            if color in active:
                train_active_by_color[color].append(
                    pair
                )
            else:
                train_inert_by_color[color].append(
                    pair
                )

    # Training proves that activity depends on component geometry,
    # rather than on a permanently active literal color.
    assert any(
        train_active_by_color[color]
        and train_inert_by_color[color]
        for color in range(10)
        if color not in train_backgrounds and color != barrier
    )

    # Training visibly establishes obstacle avoidance.
    assert any(
        _has_forced_detour(
            pair["input"],
            pair["output"],
            _background_of(pair["input"]),
            barrier,
        )
        for pair in puzzle["train"]
    )

    for pair in puzzle["test"]:
        input_grid = pair["input"]
        height, width = input_grid.shape
        background = _background_of(input_grid)

        assert 8 <= height <= 15
        assert 8 <= width <= 15

        barrier_components = _components(
            input_grid == barrier
        )
        assert 0 <= len(barrier_components) <= 3

        active = set(
            _active_colors(
                input_grid,
                background,
                barrier,
            )
        )
        assert 1 <= len(active) <= 4
        assert 0 <= len(_singleton_colors(input_grid, background, barrier)) <= 7


def _puzzles_equal(a, b):
    if set(a) != set(b):
        return False

    for split in ("train", "test"):
        if len(a[split]) != len(b[split]):
            return False

        for pair_a, pair_b in zip(
            a[split],
            b[split],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False

            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _puzzle_key(puzzle):
    return tuple(
        (
            _grid_key(pair["input"]),
            _grid_key(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def self_test() -> None:
    automatic_puzzles = []
    train_counts = set()
    test_counts = set()
    backgrounds = set()
    role_pairs = set()
    shapes = set()
    all_inputs = set()
    active_counts = set()

    total_pairs = 0
    forced_detours = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        automatic_puzzles.append(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))

        background = _infer_background(
            puzzle["train"]
        )
        barrier = _infer_barrier(
            puzzle["train"],
            background,
        )

        backgrounds.add(background)
        role_pairs.add((background, barrier))

        for pair in puzzle["train"] + puzzle["test"]:
            pair_background = _background_of(pair["input"])
            backgrounds.add(pair_background)
            role_pairs.add((pair_background, barrier))
            _assert_pair_rule(
                pair["input"],
                pair["output"],
                pair_background,
                barrier,
            )

            shapes.add(pair["input"].shape)
            all_inputs.add(
                _grid_key(pair["input"])
            )
            active_counts.add(
                len(
                    _active_colors(
                        pair["input"],
                        pair_background,
                        barrier,
                    )
                )
            )

            total_pairs += 1
            forced_detours += int(
                _has_forced_detour(
                    pair["input"],
                    pair["output"],
                    pair_background,
                    barrier,
                )
            )

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3

    assert len(
        {
            _puzzle_key(puzzle)
            for puzzle in automatic_puzzles
        }
    ) >= 40

    assert len(backgrounds) >= 7
    assert len(role_pairs) >= 18
    assert len(shapes) >= 12
    assert len(all_inputs) >= int(total_pairs * 0.9)
    assert active_counts == {1, 2, 3, 4}
    assert forced_detours >= 2 * len(automatic_puzzles)

    review = {
        "bends": set(), "fragments": set(), "barriers": set(),
        "inert": set(), "hidden": 0, "marked": 0, "long_run": 0,
        "sparse": 0, "nearly_complete": 0, "isolated_barrier": 0,
        "edge_barrier": 0, "edge_two": 0, "inert_run": 0,
        "familiar_test": 0, "switch_test": 0, "large_test": 0,
        "two_obstacles": 0,
    }
    varied_barriers = varied_inert = varied_active = 0
    for puzzle in automatic_puzzles:
        barrier = _infer_barrier(puzzle["train"], _infer_background(puzzle["train"]))
        train_active = set()
        train_inert = set()
        for pair in puzzle["train"]:
            bg = _background_of(pair["input"])
            train_active.update(_active_colors(pair["input"], bg, barrier))
            train_inert.update(_singleton_colors(pair["input"], bg, barrier))
        train_areas = [pair["input"].size for pair in puzzle["train"]]
        local_barriers, local_inert, local_active = set(), set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg = _background_of(inp)
                barrier_parts = _components(inp == barrier)
                active = _active_colors(inp, bg, barrier)
                inert = _singleton_colors(inp, bg, barrier)
                local_barriers.add(len(barrier_parts))
                local_inert.add(len(inert))
                local_active.add(len(active))
                review["barriers"].add(len(barrier_parts))
                review["inert"].add(len(inert))
                review["isolated_barrier"] += int(any(len(part) == 1 for part in barrier_parts))
                edge_parts = [
                    part for part in barrier_parts
                    if any(r in (0, inp.shape[0] - 1) or
                           c in (0, inp.shape[1] - 1) for r, c in part)
                ]
                review["edge_barrier"] += int(bool(edge_parts))
                review["edge_two"] += int(any(len(part) == 2 for part in edge_parts))
                review["inert_run"] += int(any(np.count_nonzero(inp == color) > 1
                                                for color in inert))
                if split == "test":
                    review["familiar_test"] += int(bool(set(active) & train_active))
                    review["switch_test"] += int(bool(set(active) & (train_inert - train_active)))
                    review["large_test"] += int(inp.size >= max(train_areas))
                obstacle_owners = []
                for color in active:
                    path = _ordered_path(out == color)
                    bends = _bend_count(path)
                    parts = _components(inp == color)
                    review["bends"].add(bends)
                    review["fragments"].add(len(parts))
                    review["long_run"] += int(max(map(len, parts)) >= 3)
                    visible = np.count_nonzero(inp == color) / len(path)
                    review["sparse"] += int(visible <= 0.3)
                    review["nearly_complete"] += int(visible >= 0.85)
                    for index in range(1, len(path) - 1):
                        before = (path[index][0] - path[index - 1][0],
                                  path[index][1] - path[index - 1][1])
                        after = (path[index + 1][0] - path[index][0],
                                 path[index + 1][1] - path[index][1])
                        if before != after:
                            review["hidden"] += int(inp[path[index]] == bg)
                            review["marked"] += int(inp[path[index]] == color)
                    excess = len(path) - 1 - (
                        abs(path[0][0] - path[-1][0]) +
                        abs(path[0][1] - path[-1][1])
                    )
                    if excess > 0 and barrier_parts:
                        distances = [
                            min(abs(r - rr) + abs(c - cc)
                                for r, c in path for rr, cc in part)
                            for part in barrier_parts
                        ]
                        closest = int(np.argmin(distances))
                        if distances[closest] == 1:
                            obstacle_owners.append(closest)
                review["two_obstacles"] += int(len(set(obstacle_owners)) >= 2)
        varied_barriers += int(len(local_barriers) >= 2)
        varied_inert += int(len(local_inert) >= 2)
        varied_active += int(len(local_active) >= 2)
    assert review["bends"] == {0, 1, 2, 3, 4}
    assert {2, 3, 4}.issubset(review["fragments"])
    assert review["barriers"] == {0, 1, 2, 3}
    assert {0, 1, 2, 3, 4}.issubset(review["inert"])
    assert review["hidden"] >= 80 and review["marked"] >= 60
    assert review["long_run"] >= 50
    assert review["sparse"] >= 3 and review["nearly_complete"] >= 20
    assert review["isolated_barrier"] >= 10
    assert review["edge_barrier"] >= 20 and review["edge_two"] >= 2
    assert review["inert_run"] >= 20
    assert review["familiar_test"] >= 10 and review["switch_test"] >= 5
    assert review["large_test"] >= 3 and review["two_obstacles"] >= 1
    assert varied_barriers >= 30 and varied_inert >= 30 and varied_active >= 30

    # Prediction receives only demonstrated pairs and a copied test input.
    for puzzle in automatic_puzzles:
        demonstrations = [
            {"input": np.array(pair["input"], copy=True),
             "output": np.array(pair["output"], copy=True)}
            for pair in puzzle["train"]
        ]
        learned_barrier = _infer_barrier(demonstrations,
                                         _infer_background(demonstrations))
        for pair in puzzle["test"]:
            unseen_input = np.array(pair["input"], copy=True)
            bg = _background_of(unseen_input)
            predicted = unseen_input.copy()
            for color in _active_colors(unseen_input, bg, learned_barrier):
                route = _reconstruct_color(unseen_input, color, bg, (30, 8))
                assert route is not None
                for r, c in route:
                    predicted[r, c] = color
            assert np.array_equal(predicted, pair["output"])

    explicit = generate_puzzle(
        314159,
        num_train=6,
        num_test=2,
    )
    validate_generated_puzzle(explicit)

    repeated = generate_puzzle(
        314159,
        num_train=6,
        num_test=2,
    )
    assert _puzzles_equal(explicit, repeated)

    different = generate_puzzle(
        314160,
        num_train=6,
        num_test=2,
    )
    assert not _puzzles_equal(explicit, different)

    for train_count, test_count in (
        (2, 1),
        (10, 1),
        (10, 4),
    ):
        puzzle = generate_puzzle(
            8000 + train_count * 10 + test_count,
            train_count,
            test_count,
        )
        validate_generated_puzzle(puzzle)
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count

    partial_cases = (
        (None, 1),
        (None, 4),
        (2, None),
        (10, None),
    )

    for index, (train_count, test_count) in enumerate(
        partial_cases
    ):
        puzzle = generate_puzzle(
            9000 + index,
            train_count,
            test_count,
        )
        validate_generated_puzzle(puzzle)

        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count

    illegal_cases = (
        (1, None),
        (11, None),
        (None, 0),
        (None, 5),
        (2, 2),
        (4, 4),
        (3, 4),
    )

    for train_count, test_count in illegal_cases:
        raised = False

        try:
            generate_puzzle(
                17,
                train_count,
                test_count,
            )
        except ValueError:
            raised = True

        assert raised
