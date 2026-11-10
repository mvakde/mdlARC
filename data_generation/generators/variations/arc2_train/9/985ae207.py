import hashlib
import numpy as np


_DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_DIRECTION_NAMES = tuple(_DIRECTIONS)


def _checked_count(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed_value, rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train)
    num_test = _checked_count("num_test", num_test)

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = 1 + seed_value % 4
        quotient = seed_value // 4
        num_train = num_test + 1 + quotient % (10 - num_test)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = int(rng.integers(low, 11))
    elif num_test is None:
        high = min(4, num_train - 1)
        if high < 1:
            raise ValueError("no legal num_test exists for num_train")
        num_test = int(rng.integers(1, high + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def sample_puzzle_parameters(rng):
    # The only puzzle-wide convention is the fixed 3-cell tile phase. Color
    # roles are visible locally and are deliberately resampled for each pair.
    return {"tile_pitch": 3}


def _shuffle_connections(rng, directions, targets):
    order = rng.permutation(len(directions))
    shuffled_directions = [
        directions[int(index)] for index in order
    ]
    shuffled_targets = [targets[int(index)] for index in order]
    return shuffled_directions, shuffled_targets


def _pair_spec(rng, regime, puzzle_parameters, pair_index):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")

    # Both regimes span small and large scenes. Test is biased toward more
    # connections, but need not contain a prescribed direction or fan-in.
    if regime == "source_train":
        height = int(rng.integers(17, 31))
        width = int(rng.integers(17, 31))
        signal_count = int(rng.integers(2, 5))
    else:
        height = int(rng.integers(19, 31))
        width = int(rng.integers(19, 31))
        signal_count = int(rng.integers(2, 6))

    # Sparse scenes and shared receivers both occur. The first training pair
    # has two distinct matches and both repetition axes for clear evidence.
    if regime == "source_train" and pair_index == 0:
        signal_count = max(2, signal_count)
        receiver_count = min(signal_count, int(rng.integers(2, 4)))
    else:
        receiver_count = int(rng.integers(1, min(signal_count, 4) + 1))

    colors = rng.permutation(10)
    background = int(colors[0])
    carrier = int(colors[1])
    receiver_colors = tuple(int(x) for x in colors[2 : 2 + receiver_count])

    directions = [
        _DIRECTION_NAMES[int(rng.integers(0, 4))]
        for _ in range(signal_count)
    ]
    if regime == "source_train" and pair_index == 0:
        directions[0] = ("left", "right")[int(rng.integers(0, 2))]
        directions[1] = ("up", "down")[int(rng.integers(0, 2))]

    targets = list(range(receiver_count))
    while len(targets) < signal_count:
        targets.append(int(rng.integers(0, receiver_count)))
    directions, targets = _shuffle_connections(rng, directions, targets)
    styles = tuple(
        ("horizontal", "vertical", "compact", "random")[
            int(rng.integers(0, 4))
        ]
        for _ in range(receiver_count)
    )

    return {
        "height": height,
        "width": width,
        "background": background,
        "carrier": carrier,
        "receiver_colors": receiver_colors,
        "directions": directions,
        "targets": targets,
        "styles": styles,
        "regime": regime,
    }


def _rectangle_size_options(height, width, directions, style, regime):
    horizontal_signals = sum(
        direction in ("left", "right") for direction in directions
    )
    vertical_signals = sum(
        direction in ("up", "down") for direction in directions
    )

    min_height = max(
        2,
        4 * (horizontal_signals - 1) + 1
        if horizontal_signals > 1
        else 2,
    )
    min_width = max(
        2,
        4 * (vertical_signals - 1) + 1
        if vertical_signals > 1
        else 2,
    )

    options = []

    for rect_height in range(min_height, height - 1):
        for rect_width in range(min_width, width - 1):
            area = rect_height * rect_width
            if area < 8 or area > max(110, height * width // 4):
                continue

            if style == "horizontal" and rect_width < rect_height + 2:
                continue
            if style == "vertical" and rect_height < rect_width + 2:
                continue
            if style == "compact" and abs(rect_height - rect_width) > 2:
                continue

            row_low = 4 if "down" in directions else 0
            row_high = (
                height - rect_height - 4
                if "up" in directions
                else height - rect_height
            )
            col_low = 4 if "right" in directions else 0
            col_high = (
                width - rect_width - 4
                if "left" in directions
                else width - rect_width
            )

            if row_low <= row_high and col_low <= col_high:
                options.append((rect_height, rect_width))

    return options


def _rectangles_too_close(first, second):
    first_r0, first_c0, first_r1, first_c1 = first
    second_r0, second_c0, second_r1, second_c1 = second

    separated = (
        first_r1 + 1 < second_r0
        or second_r1 + 1 < first_r0
        or first_c1 + 1 < second_c0
        or second_c1 + 1 < first_c0
    )
    return not separated


def _place_rectangles(
    rng,
    height,
    width,
    directions_by_receiver,
    size_options,
):
    rectangle_count = len(directions_by_receiver)
    rectangles = [None] * rectangle_count

    order = sorted(
        range(rectangle_count),
        key=lambda index: (
            len(size_options[index]),
            -len(directions_by_receiver[index]),
        ),
    )

    for receiver_index in order:
        directions = directions_by_receiver[receiver_index]
        options = size_options[receiver_index]
        placed = False

        for _ in range(240):
            option = options[int(rng.integers(0, len(options)))]
            rect_height, rect_width = option

            row_low = 4 if "down" in directions else 0
            row_high = (
                height - rect_height - 4
                if "up" in directions
                else height - rect_height
            )
            col_low = 4 if "right" in directions else 0
            col_high = (
                width - rect_width - 4
                if "left" in directions
                else width - rect_width
            )

            if row_low > row_high or col_low > col_high:
                continue

            r0 = int(rng.integers(row_low, row_high + 1))
            c0 = int(rng.integers(col_low, col_high + 1))
            candidate = (
                r0,
                c0,
                r0 + rect_height - 1,
                c0 + rect_width - 1,
            )

            if any(
                other is not None
                and _rectangles_too_close(candidate, other)
                for other in rectangles
            ):
                continue

            rectangles[receiver_index] = candidate
            placed = True
            break

        if not placed:
            return None

    return rectangles


def _rect_cells(rectangle):
    r0, c0, r1, c1 = rectangle
    return {
        (row, col)
        for row in range(r0, r1 + 1)
        for col in range(c0, c1 + 1)
    }


def _patch_cells(center, height, width):
    row, col = center
    assert 1 <= row < height - 1
    assert 1 <= col < width - 1
    return {
        (rr, cc)
        for rr in range(row - 1, row + 2)
        for cc in range(col - 1, col + 2)
    }


def _buffer_cells(center, radius, height, width):
    row, col = center
    return {
        (rr, cc)
        for rr in range(max(0, row - radius), min(height, row + radius + 1))
        for cc in range(max(0, col - radius), min(width, col + radius + 1))
    }


def _signal_candidates(
    height,
    width,
    rectangle,
    direction,
):
    r0, c0, r1, c1 = rectangle
    dr, dc = _DIRECTIONS[direction]
    endpoints = []

    if direction in ("left", "right"):
        for endpoint_row in range(max(1, r0), min(height - 2, r1) + 1):
            for inset in range(min(2, c1 - c0) + 1):
                endpoint_col = c1 - inset if direction == "left" else c0 + inset
                endpoints.append((endpoint_row, endpoint_col))
    else:
        for endpoint_col in range(max(1, c0), min(width - 2, c1) + 1):
            for inset in range(min(2, r1 - r0) + 1):
                endpoint_row = r1 - inset if direction == "up" else r0 + inset
                endpoints.append((endpoint_row, endpoint_col))

    candidates = []

    for endpoint in endpoints:
        endpoint_row, endpoint_col = endpoint

        if direction == "right":
            max_steps = (endpoint_col - 1) // 3
        elif direction == "left":
            max_steps = (width - 2 - endpoint_col) // 3
        elif direction == "down":
            max_steps = (endpoint_row - 1) // 3
        else:
            max_steps = (height - 2 - endpoint_row) // 3

        for steps in range(1, max_steps + 1):
            source = (
                endpoint_row - 3 * steps * dr,
                endpoint_col - 3 * steps * dc,
            )

            if not (
                1 <= source[0] < height - 1
                and 1 <= source[1] < width - 1
            ):
                continue

            centers = [
                (
                    source[0] + 3 * offset * dr,
                    source[1] + 3 * offset * dc,
                )
                for offset in range(steps + 1)
            ]

            path = set()
            valid = True
            for center in centers:
                if not (
                    1 <= center[0] < height - 1
                    and 1 <= center[1] < width - 1
                ):
                    valid = False
                    break
                path.update(_patch_cells(center, height, width))

            if not valid:
                continue

            candidates.append(
                {
                    "source": source,
                    "endpoint": endpoint,
                    "centers": centers,
                    "path": path,
                    "source_patch": _patch_cells(
                        source, height, width
                    ),
                    "endpoint_patch": _patch_cells(
                        endpoint, height, width
                    ),
                }
            )

    return candidates


def _draw_tile(grid, center, carrier, label):
    row, col = center
    grid[row - 1 : row + 2, col - 1 : col + 2] = carrier
    grid[row, col] = label


def _generate_scene(rng, spec):
    height = spec["height"]
    width = spec["width"]
    background = spec["background"]
    carrier = spec["carrier"]
    receiver_colors = spec["receiver_colors"]
    directions = spec["directions"]
    targets = spec["targets"]
    styles = spec["styles"]
    regime = spec["regime"]

    receiver_count = len(receiver_colors)
    directions_by_receiver = [[] for _ in range(receiver_count)]
    for direction, target in zip(directions, targets):
        directions_by_receiver[target].append(direction)

    size_options = [
        _rectangle_size_options(
            height,
            width,
            directions_by_receiver[index],
            styles[index],
            regime,
        )
        for index in range(receiver_count)
    ]

    if any(not options for options in size_options):
        raise RuntimeError("no feasible receiver geometry")

    for _ in range(1400):
        rectangles = _place_rectangles(
            rng,
            height,
            width,
            directions_by_receiver,
            size_options,
        )
        if rectangles is None:
            continue

        rectangle_sets = [_rect_cells(rect) for rect in rectangles]
        rectangle_union = set().union(*rectangle_sets)

        candidate_lists = []
        feasible = True
        for direction, target in zip(directions, targets):
            candidates = _signal_candidates(
                height,
                width,
                rectangles[target],
                direction,
            )
            if not candidates:
                feasible = False
                break
            candidate_lists.append(candidates)

        if not feasible:
            continue

        signal_order = sorted(
            range(len(directions)),
            key=lambda index: len(candidate_lists[index]),
        )

        records = [None] * len(directions)
        used_paths = set()
        source_patches = []
        failed = False

        for signal_index in signal_order:
            target = targets[signal_index]
            candidates = candidate_lists[signal_index]
            order = rng.permutation(len(candidates))
            chosen = None

            for candidate_index in order:
                candidate = candidates[int(candidate_index)]
                path = candidate["path"]

                if path & used_paths:
                    continue

                bad_intersection = False
                for rectangle_index, cells in enumerate(rectangle_sets):
                    intersection = path & cells
                    if rectangle_index == target:
                        # With an inset landing, the preceding tile's edge
                        # may cover one receiver column or row. Its center
                        # is still outside, so the first-center rule holds.
                        continue
                    elif intersection:
                        bad_intersection = True
                        break

                if bad_intersection:
                    continue

                source_buffer = _buffer_cells(
                    candidate["source"], 2, height, width
                )
                if candidate["source_patch"] & rectangle_union:
                    continue
                if any(
                    source_buffer & previous_patch
                    for previous_patch in source_patches
                ):
                    continue

                chosen = candidate
                break

            if chosen is None:
                failed = True
                break

            records[signal_index] = chosen
            used_paths.update(chosen["path"])
            source_patches.append(chosen["source_patch"])

        if failed:
            continue

        input_grid = np.full(
            (height, width), background, dtype=np.int8
        )

        for rectangle, color in zip(rectangles, receiver_colors):
            r0, c0, r1, c1 = rectangle
            input_grid[r0 : r1 + 1, c0 : c1 + 1] = color

        for signal_index, record in enumerate(records):
            label = receiver_colors[targets[signal_index]]
            _draw_tile(
                input_grid,
                record["source"],
                carrier,
                label,
            )

        values, counts = np.unique(input_grid, return_counts=True)
        background_count = int(
            counts[np.where(values == background)[0][0]]
        )
        other_counts = [
            int(count)
            for value, count in zip(values, counts)
            if int(value) != background
        ]
        if other_counts and background_count <= max(other_counts):
            continue

        output_grid = input_grid.copy()
        for signal_index, record in enumerate(records):
            label = receiver_colors[targets[signal_index]]
            for center in record["centers"]:
                _draw_tile(
                    output_grid,
                    center,
                    carrier,
                    label,
                )

        try:
            info = _analyze_pair(input_grid, output_grid)
        except AssertionError:
            continue

        if len(info["signals"]) != len(directions):
            continue
        if len(info["rectangles"]) != receiver_count:
            continue

        return input_grid, output_grid

    raise RuntimeError("could not construct a nonconflicting scene")


def generate_pair(rng, regime, puzzle_parameters, pair_index):
    spec = _pair_spec(
        rng,
        regime,
        puzzle_parameters,
        pair_index,
    )
    return _generate_scene(rng, spec)


def _color_components(grid, color):
    coordinates = {
        (int(row), int(col))
        for row, col in zip(*np.where(grid == color))
    }
    components = []

    while coordinates:
        start = coordinates.pop()
        stack = [start]
        component = {start}

        while stack:
            row, col = stack.pop()
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in coordinates:
                    coordinates.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def _route_candidates(signal_center, rectangle):
    row, col = signal_center
    r0, c0, r1, c1 = rectangle
    candidates = []

    if r0 <= row <= r1 and col < c0:
        endpoint_col = c0 + (col - c0) % 3
        if endpoint_col <= c1:
            candidates.append(("right", (row, endpoint_col)))

    if r0 <= row <= r1 and col > c1:
        endpoint_col = c1 - (c1 - col) % 3
        if endpoint_col >= c0:
            candidates.append(("left", (row, endpoint_col)))

    if c0 <= col <= c1 and row < r0:
        endpoint_row = r0 + (row - r0) % 3
        if endpoint_row <= r1:
            candidates.append(("down", (endpoint_row, col)))

    if c0 <= col <= c1 and row > r1:
        endpoint_row = r1 - (r1 - row) % 3
        if endpoint_row >= r0:
            candidates.append(("up", (endpoint_row, col)))

    return candidates


def _analyze_pair(input_grid, output_grid):
    assert isinstance(input_grid, np.ndarray)
    assert isinstance(output_grid, np.ndarray)
    assert input_grid.ndim == 2
    assert output_grid.ndim == 2
    assert input_grid.shape == output_grid.shape

    height, width = input_grid.shape
    values, counts = np.unique(input_grid, return_counts=True)
    assert np.sum(counts == np.max(counts)) == 1
    background = int(values[int(np.argmax(counts))])

    signal_candidates = []

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            patch = input_grid[
                row - 1 : row + 2,
                col - 1 : col + 2,
            ]
            flat = patch.reshape(-1)
            outer = np.concatenate((flat[:4], flat[5:]))
            carrier = int(outer[0])
            center = int(flat[4])

            if (
                np.all(outer == carrier)
                and carrier != background
                and center != background
                and center != carrier
            ):
                signal_candidates.append(
                    {
                        "center": (row, col),
                        "label": center,
                        "carrier": carrier,
                    }
                )

    assert signal_candidates
    carriers = {signal["carrier"] for signal in signal_candidates}
    assert len(carriers) == 1
    carrier = next(iter(carriers))

    used_signal_cells = set()
    for signal in signal_candidates:
        cells = _patch_cells(signal["center"], height, width)
        assert not cells & used_signal_cells
        used_signal_cells.update(cells)

    labels = {signal["label"] for signal in signal_candidates}
    present_colors = {int(value) for value in np.unique(input_grid)}
    assert present_colors == {background, carrier} | labels

    centers_by_label = {
        label: {
            signal["center"]
            for signal in signal_candidates
            if signal["label"] == label
        }
        for label in labels
    }

    rectangles_by_label = {}
    rectangle_info = []

    for label in labels:
        rectangular_components = []
        singleton_centers = set()

        for component in _color_components(input_grid, label):
            rows = [cell[0] for cell in component]
            cols = [cell[1] for cell in component]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            rect_height = r1 - r0 + 1
            rect_width = c1 - c0 + 1
            filled_rectangle = (
                len(component) == rect_height * rect_width
            )

            if (
                filled_rectangle
                and rect_height >= 2
                and rect_width >= 2
                and len(component) >= 8
            ):
                rectangular_components.append((r0, c0, r1, c1))
            else:
                assert len(component) == 1
                singleton_centers.update(component)

        assert len(rectangular_components) == 1
        assert singleton_centers == centers_by_label[label]

        rectangle = rectangular_components[0]
        rectangles_by_label[label] = rectangle
        rectangle_info.append(
            {
                "label": label,
                "bbox": rectangle,
                "height": rectangle[2] - rectangle[0] + 1,
                "width": rectangle[3] - rectangle[1] + 1,
            }
        )

    reconstructed_input = np.full(
        input_grid.shape, background, dtype=input_grid.dtype
    )

    for label, rectangle in rectangles_by_label.items():
        r0, c0, r1, c1 = rectangle
        reconstructed_input[r0 : r1 + 1, c0 : c1 + 1] = label

    for signal in signal_candidates:
        _draw_tile(
            reconstructed_input,
            signal["center"],
            carrier,
            signal["label"],
        )

    assert np.array_equal(reconstructed_input, input_grid)

    expected_output = input_grid.copy()
    used_route_cells = set()
    analyzed_signals = []

    for signal in sorted(
        signal_candidates, key=lambda item: item["center"]
    ):
        center = signal["center"]
        label = signal["label"]
        rectangle = rectangles_by_label[label]
        route_options = _route_candidates(center, rectangle)

        assert len(route_options) == 1
        direction, endpoint = route_options[0]
        dr, dc = _DIRECTIONS[direction]

        row_distance = abs(endpoint[0] - center[0])
        col_distance = abs(endpoint[1] - center[1])
        total_distance = row_distance + col_distance

        assert total_distance % 3 == 0
        steps = total_distance // 3
        assert steps >= 1

        centers = [
            (
                center[0] + 3 * offset * dr,
                center[1] + 3 * offset * dc,
            )
            for offset in range(steps + 1)
        ]
        assert centers[-1] == endpoint

        route_cells = set()
        for tile_center in centers:
            route_cells.update(
                _patch_cells(tile_center, height, width)
            )

        assert not route_cells & used_route_cells
        used_route_cells.update(route_cells)

        for tile_center in centers:
            _draw_tile(
                expected_output,
                tile_center,
                carrier,
                label,
            )

        analyzed_signals.append(
            {
                "center": center,
                "label": label,
                "direction": direction,
                "endpoint": endpoint,
                "steps": steps,
                "inset": (
                    endpoint[1] - rectangle[1]
                    if direction == "right" else
                    rectangle[3] - endpoint[1]
                    if direction == "left" else
                    endpoint[0] - rectangle[0]
                    if direction == "down" else
                    rectangle[2] - endpoint[0]
                ),
            }
        )

    assert np.array_equal(expected_output, output_grid)
    assert np.any(input_grid != output_grid)

    rectangle_info.sort(key=lambda item: item["label"])

    return {
        "background": background,
        "carrier": carrier,
        "signals": analyzed_signals,
        "rectangles": rectangle_info,
        "labels": labels,
    }


def _grid_signature(grid):
    return (grid.shape, grid.dtype.str, grid.tobytes())


_SOURCE_PAIR_DIGESTS = {
    "6bbd6261f9748b2f02d08d741e6475879a6accbaf6bffb821d22a772af59d8a1",
    "53672a028be912d1485314393ba2f385a99bdf8089bc41640f1370d12b99f915",
    "8341bf7bb4cb5a036a01a69fb060c61832874c4110e51d128b7a14dbd6d08967",
    "2072083bdba9c49cf0e3eef6efd89e5ebc28b620ed7e492408930770e8ddfe48",
}


def _source_pair_digest(input_grid, output_grid):
    canonical_input = np.asarray(input_grid, dtype=np.int8)
    canonical_output = np.asarray(output_grid, dtype=np.int8)
    return hashlib.sha256(
        bytes(canonical_input.shape)
        + canonical_input.tobytes()
        + canonical_output.tobytes()
    ).hexdigest()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    input_signatures = set()
    pair_signatures = set()
    train_info = []
    test_info = []

    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict)
            assert set(pair) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                rows, cols = grid.shape
                assert 1 <= rows <= 30
                assert 1 <= cols <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))

            assert input_grid.shape == output_grid.shape

            input_signature = _grid_signature(input_grid)
            pair_signature = (
                input_signature,
                _grid_signature(output_grid),
            )
            assert input_signature not in input_signatures
            assert pair_signature not in pair_signatures
            assert _source_pair_digest(input_grid, output_grid) not in _SOURCE_PAIR_DIGESTS
            input_signatures.add(input_signature)
            pair_signatures.add(pair_signature)

            info = _analyze_pair(input_grid, output_grid)
            if regime == "train":
                train_info.append(info)
            else:
                test_info.append(info)

    # Geometry and color identities are pair-local. The first demonstration
    # establishes matching by color and both horizontal and vertical motion.
    first = train_info[0]
    assert len(first["labels"]) >= 2
    first_directions = {s["direction"] for s in first["signals"]}
    assert first_directions & {"left", "right"}
    assert first_directions & {"up", "down"}
    train_signals = [s for info in train_info for s in info["signals"]]
    test_signals = [s for info in test_info for s in info["signals"]]
    assert any(s["steps"] >= 2 for s in train_signals)
    assert {s["inset"] for s in test_signals}.issubset(
        {s["inset"] for s in train_signals}
    )

    for info in train_info + test_info:
        assert 1 <= len(info["rectangles"]) <= 4
        assert 2 <= len(info["signals"]) <= 5
        assert info["background"] != info["carrier"]
        assert info["background"] not in info["labels"]
        assert info["carrier"] not in info["labels"]
        assert all(1 <= s["steps"] for s in info["signals"])


def _puzzles_equal(first, second):
    if set(first) != {"train", "test"}:
        return False
    if set(second) != {"train", "test"}:
        return False

    for regime in ("train", "test"):
        if len(first[regime]) != len(second[regime]):
            return False
        for first_pair, second_pair in zip(
            first[regime], second[regime]
        ):
            if not np.array_equal(
                first_pair["input"], second_pair["input"]
            ):
                return False
            if not np.array_equal(
                first_pair["output"], second_pair["output"]
            ):
                return False

    return True


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = int(seed)
    if seed_value < 0:
        seed_value %= 1 << 64

    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(
        seed_value, rng, num_train, num_test
    )
    puzzle_parameters = sample_puzzle_parameters(rng)

    # A rare random draw can omit a landing phase from demonstrations while
    # using it in test. Resample the complete scene set in that case.
    for _ in range(60):
        puzzle = {"train": [], "test": []}
        seen_inputs = set()
        complete = True

        for regime, count in (
            ("source_train", train_count),
            ("target_test", test_count),
        ):
            destination = "train" if regime == "source_train" else "test"

            for pair_index in range(count):
                pair = None
                for _ in range(60):
                    pair_seed = int(rng.integers(0, (1 << 63) - 1))
                    pair_rng = np.random.default_rng(pair_seed)
                    try:
                        input_grid, output_grid = generate_pair(
                            pair_rng, regime, puzzle_parameters, pair_index
                        )
                    except RuntimeError:
                        continue
                    signature = _grid_signature(input_grid)
                    if signature in seen_inputs:
                        continue
                    pair = {"input": input_grid, "output": output_grid}
                    seen_inputs.add(signature)
                    break
                if pair is None:
                    complete = False
                    break
                puzzle[destination].append(pair)
            if not complete:
                break

        if not complete:
            continue

        train_info = [
            _analyze_pair(pair["input"], pair["output"])
            for pair in puzzle["train"]
        ]
        test_info = [
            _analyze_pair(pair["input"], pair["output"])
            for pair in puzzle["test"]
        ]
        train_signals = [s for info in train_info for s in info["signals"]]
        test_signals = [s for info in test_info for s in info["signals"]]
        if not any(s["steps"] >= 2 for s in train_signals):
            continue
        if not {s["inset"] for s in test_signals}.issubset(
            {s["inset"] for s in train_signals}
        ):
            continue

        validate_generated_puzzle(puzzle)
        return puzzle

    raise RuntimeError("could not construct a coherent puzzle")


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)

    repeated = generate_puzzle(314159, 6, 3)
    assert _puzzles_equal(repeated, generate_puzzle(314159, 6, 3))
    validate_generated_puzzle(repeated)

    for seed, train_count, test_count in (
        (11001, 2, 1), (11002, 10, 1), (11003, 10, 4),
        (12001, None, 1), (12002, None, 4),
        (12003, 2, None), (12004, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count

    for arguments in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    ):
        try:
            generate_puzzle(13000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")

    train_counts = {len(p["train"]) for p in automatic}
    test_counts = {len(p["test"]) for p in automatic}
    shapes = set()
    rectangle_shapes = set()
    structural_layouts = set()
    signal_counts = set()
    receiver_counts = set()
    backgrounds = set()
    carriers = set()
    steps = set()
    insets = set()
    direction_sets = set()
    all_inputs = set()
    long_receiver = edge_receiver = 0
    same_side_share = mixed_side_share = 0
    within_size_change = within_count_change = within_color_change = 0
    short_inset = long_inset = 0

    for puzzle in automatic:
        infos = []
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                input_grid, output_grid = pair["input"], pair["output"]
                info = _analyze_pair(input_grid, output_grid)
                infos.append(info)
                height, width = input_grid.shape
                shapes.add((height, width))
                signal_counts.add(len(info["signals"]))
                receiver_counts.add(len(info["rectangles"]))
                backgrounds.add(info["background"])
                carriers.add(info["carrier"])
                direction_sets.add(frozenset(
                    signal["direction"] for signal in info["signals"]
                ))
                all_inputs.add(_grid_signature(input_grid))
                assert _source_pair_digest(input_grid, output_grid) not in _SOURCE_PAIR_DIGESTS

                layout = (
                    input_grid.shape,
                    tuple(sorted((s["direction"], s["steps"],
                                  s["center"]) for s in info["signals"])),
                    tuple(sorted(r["bbox"] for r in info["rectangles"])),
                )
                structural_layouts.add(layout)

                by_label = {}
                for signal in info["signals"]:
                    steps.add(signal["steps"])
                    insets.add(signal["inset"])
                    by_label.setdefault(signal["label"], []).append(
                        signal["direction"]
                    )
                    if signal["inset"] and signal["steps"] == 1:
                        short_inset += 1
                    if signal["inset"] and signal["steps"] >= 3:
                        long_inset += 1
                for directions in by_label.values():
                    if len(directions) >= 2:
                        if len(set(directions)) == 1:
                            same_side_share += 1
                        else:
                            mixed_side_share += 1

                for rectangle in info["rectangles"]:
                    rectangle_shapes.add((rectangle["height"], rectangle["width"]))
                    r0, c0, r1, c1 = rectangle["bbox"]
                    if rectangle["height"] >= 15 or rectangle["width"] >= 15:
                        long_receiver += 1
                    if r0 == 0 or c0 == 0 or r1 == height - 1 or c1 == width - 1:
                        edge_receiver += 1

        pairs = puzzle["train"] + puzzle["test"]
        within_size_change += len({pair["input"].shape for pair in pairs}) > 1
        within_count_change += len({len(info["signals"]) for info in infos}) > 1
        within_color_change += len({info["carrier"] for info in infos}) > 1

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 70 and len(rectangle_shapes) >= 60
    assert len(structural_layouts) >= 200
    assert signal_counts == {2, 3, 4, 5}
    assert receiver_counts == {1, 2, 3, 4}
    assert len(backgrounds) == 10 and len(carriers) == 10
    assert 1 in steps and max(steps) >= 5
    assert insets == {0, 1, 2}
    assert len(direction_sets) >= 8
    assert long_receiver >= 10 and edge_receiver >= 10
    assert same_side_share >= 5 and mixed_side_share >= 5
    assert within_size_change >= 35
    assert within_count_change >= 20
    assert within_color_change >= 35
    assert short_inset > 0 and long_inset > 0
    assert len(all_inputs) >= 250
