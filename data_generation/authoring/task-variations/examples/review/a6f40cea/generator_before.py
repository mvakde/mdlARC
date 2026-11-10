import numpy as np


_SOURCE_OUTPUT_SHAPES = {
    (4, 7),
    (13, 14),
    (7, 8),
    (8, 10),
}


def _integer_argument(value, name):
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _resolve_counts(seed_key, rng, num_train, num_test):
    if num_train is not None:
        num_train = _integer_argument(num_train, "num_train")
        if not 2 <= num_train <= 10:
            raise ValueError("num_train must be between 2 and 10")

    if num_test is not None:
        num_test = _integer_argument(num_test, "num_test")
        if not 1 <= num_test <= 4:
            raise ValueError("num_test must be between 1 and 4")

    if num_train is None and num_test is None:
        num_test = 1 + seed_key % 4
        low = max(2, num_test + 1)
        span = 11 - low
        num_train = low + (seed_key // 4) % span
    elif num_train is None:
        low = max(2, num_test + 1)
        if low > 10:
            raise ValueError("no legal num_train exceeds num_test")
        num_train = int(rng.integers(low, 11))
    elif num_test is None:
        high = min(4, num_train - 1)
        if high < 1:
            raise ValueError("no legal num_test is below num_train")
        num_test = int(rng.integers(1, high + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _sample_puzzle_parameters(rng):
    colors = [int(x) for x in rng.permutation(10)]
    return {
        "background": colors[0],
        "frame": colors[1],
        "fill": colors[2],
        "solid": (colors[3], colors[4]),
        "alternating": (colors[5], colors[6]),
        "distractor": (colors[7], colors[8]),
    }


def _randint_inclusive(rng, low, high):
    if low > high:
        raise RuntimeError("empty integer sampling interval")
    return int(rng.integers(low, high + 1))


def _perimeter(bbox):
    r0, r1, c0, c1 = bbox
    cells = set()

    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))

    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))

    return cells


def _rect_area(bbox):
    r0, r1, c0, c1 = bbox
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }


def _inside_closed_main(cell, main):
    r, c = cell
    top, bottom, left, right = main
    return top <= r <= bottom and left <= c <= right


def _inside_strict_main(cell, main):
    r, c = cell
    top, bottom, left, right = main
    return top < r < bottom and left < c < right


def _sample_canvas(rng, regime):
    while True:
        if regime == "source_train":
            out_h = int(rng.integers(5, 12))
            out_w = int(rng.integers(6, 14))
            margin = 4
        else:
            out_h = int(rng.integers(8, 13))
            out_w = int(rng.integers(8, 15))
            margin = 6

        if (out_h, out_w) in _SOURCE_OUTPUT_SHAPES:
            continue

        min_h = out_h + 2 + 2 * margin
        min_w = out_w + 2 + 2 * margin
        if min_h > 30 or min_w > 30:
            continue

        height = _randint_inclusive(rng, min_h, 30)
        width = _randint_inclusive(rng, min_w, 30)

        top_max = height - (out_h + 2) - margin
        left_max = width - (out_w + 2) - margin
        top = _randint_inclusive(rng, margin, top_max)
        left = _randint_inclusive(rng, margin, left_max)

        bottom = top + out_h + 1
        right = left + out_w + 1
        return height, width, (top, bottom, left, right)


def _sample_corner_rectangle(rng, corner, main, height, width):
    top, bottom, left, right = main

    inner_r = _randint_inclusive(rng, top + 2, bottom - 2)
    inner_c = _randint_inclusive(rng, left + 2, right - 2)

    if corner == 0:  # northwest
        outer_r = _randint_inclusive(rng, 1, top - 2)
        outer_c = _randint_inclusive(rng, 1, left - 2)
        return outer_r, inner_r, outer_c, inner_c

    if corner == 1:  # northeast
        outer_r = _randint_inclusive(rng, 1, top - 2)
        outer_c = _randint_inclusive(rng, right + 2, width - 2)
        return outer_r, inner_r, inner_c, outer_c

    if corner == 2:  # southwest
        outer_r = _randint_inclusive(rng, bottom + 2, height - 2)
        outer_c = _randint_inclusive(rng, 1, left - 2)
        return inner_r, outer_r, outer_c, inner_c

    if corner == 3:  # southeast
        outer_r = _randint_inclusive(rng, bottom + 2, height - 2)
        outer_c = _randint_inclusive(rng, right + 2, width - 2)
        return inner_r, outer_r, inner_c, outer_c

    raise RuntimeError("unknown corner")


def _touch_orthogonally(first, second):
    for r, c in first:
        if (
            (r, c) in second
            or (r - 1, c) in second
            or (r + 1, c) in second
            or (r, c - 1) in second
            or (r, c + 1) in second
        ):
            return True
    return False


def _sample_rectangles(rng, specs, main, height, width):
    count = len(specs)

    for _ in range(6000):
        corners = [
            int(x)
            for x in rng.choice(4, size=count, replace=False)
        ]
        rectangles = []
        occupied = set()
        alternating_visible = []
        failed = False

        for spec, corner in zip(specs, corners):
            bbox = _sample_corner_rectangle(
                rng, corner, main, height, width
            )
            full = _perimeter(bbox)
            visible = {
                p for p in full if not _inside_closed_main(p, main)
            }
            hidden = {
                p for p in full if _inside_strict_main(p, main)
            }

            if not visible or len(hidden) < 3 or full & occupied:
                failed = True
                break

            if spec["kind"] == "alternating":
                if {
                    (r + c) & 1 for r, c in hidden
                } != {0, 1}:
                    failed = True
                    break

                if any(
                    _touch_orthogonally(visible, old)
                    for old in alternating_visible
                ):
                    failed = True
                    break

            descriptor = dict(spec)
            descriptor["bbox"] = bbox
            descriptor["phase"] = int(rng.integers(0, 2))
            descriptor["corner"] = corner
            rectangles.append(descriptor)
            occupied.update(full)

            if spec["kind"] == "alternating":
                alternating_visible.append(visible)

        if not failed and len(rectangles) == count:
            return rectangles

    return None


def _sample_distractors(rng, main, height, width, rectangles):
    blocked = set()
    for rectangle in rectangles:
        blocked.update(_perimeter(rectangle["bbox"]))

    selected = []
    selected_area = set()
    top, bottom, left, right = main

    for _ in range(4000):
        if len(selected) == 2:
            return selected

        box_h = int(rng.integers(3, 6))
        box_w = int(rng.integers(3, 6))
        r0 = _randint_inclusive(rng, 1, height - box_h - 1)
        c0 = _randint_inclusive(rng, 1, width - box_w - 1)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1

        is_separate = (
            r1 <= top - 2
            or r0 >= bottom + 2
            or c1 <= left - 2
            or c0 >= right + 2
        )
        if not is_separate:
            continue

        bbox = (r0, r1, c0, c1)
        area = _rect_area(bbox)
        if area & blocked or area & selected_area:
            continue

        if _touch_orthogonally(area, selected_area):
            continue

        selected.append(bbox)
        selected_area.update(area)

    return None


def _descriptor_color(descriptor, r, c):
    if descriptor["kind"] == "solid":
        return descriptor["color"]

    first, second = descriptor["palette"]
    if ((r + c + descriptor["phase"]) & 1) == 0:
        return first
    return second


def _render_pair(
    height,
    width,
    main,
    rectangles,
    distractors,
    fill_color,
    parameters,
):
    background = parameters["background"]
    frame = parameters["frame"]
    grid = np.full((height, width), background, dtype=np.int8)

    for descriptor in rectangles:
        for r, c in _perimeter(descriptor["bbox"]):
            grid[r, c] = _descriptor_color(descriptor, r, c)

    for color, bbox in zip(parameters["distractor"], distractors):
        for r, c in _perimeter(bbox):
            grid[r, c] = color

    top, bottom, left, right = main
    grid[top + 1:bottom, left + 1:right] = fill_color
    grid[top, left:right + 1] = frame
    grid[bottom, left:right + 1] = frame
    grid[top + 1:bottom, left] = frame
    grid[top + 1:bottom, right] = frame

    output = np.full(
        (bottom - top - 1, right - left - 1),
        fill_color,
        dtype=np.int8,
    )

    for descriptor in rectangles:
        for r, c in _perimeter(descriptor["bbox"]):
            if _inside_strict_main((r, c), main):
                output[r - top - 1, c - left - 1] = (
                    _descriptor_color(descriptor, r, c)
                )

    return grid, output


def _generate_pair(
    rng,
    regime,
    parameters,
    forced_mode=None,
    forced_fill=None,
):
    if regime not in {"source_train", "target_test"}:
        raise ValueError("unknown pair regime")

    if regime == "source_train":
        mode = forced_mode
        if mode is None:
            mode = "solid" if int(rng.integers(0, 2)) == 0 else "alternating"

        fill_color = forced_fill
        if fill_color is None:
            fill_color = (
                parameters["background"]
                if int(rng.integers(0, 2)) == 0
                else parameters["fill"]
            )

        if mode == "solid":
            specs = [
                {"kind": "solid", "color": parameters["solid"][0]},
                {"kind": "solid", "color": parameters["solid"][1]},
            ]
        elif mode == "alternating":
            specs = [
                {
                    "kind": "alternating",
                    "palette": parameters["alternating"],
                },
                {
                    "kind": "alternating",
                    "palette": parameters["alternating"],
                },
            ]
        else:
            raise ValueError("unknown training mode")
    else:
        fill_color = parameters["fill"]
        specs = [
            {"kind": "solid", "color": parameters["solid"][0]},
            {"kind": "solid", "color": parameters["solid"][1]},
            {
                "kind": "alternating",
                "palette": parameters["alternating"],
            },
        ]

    for _ in range(300):
        height, width, main = _sample_canvas(rng, regime)
        rectangles = _sample_rectangles(
            rng, specs, main, height, width
        )
        if rectangles is None:
            continue

        if regime == "target_test":
            distractors = _sample_distractors(
                rng, main, height, width, rectangles
            )
            if distractors is None:
                continue
        else:
            distractors = []

        grid, output = _render_pair(
            height,
            width,
            main,
            rectangles,
            distractors,
            fill_color,
            parameters,
        )

        expected_output_colors = {fill_color}
        if regime == "target_test" or mode == "solid":
            expected_output_colors.update(parameters["solid"])
        if regime == "target_test" or mode == "alternating":
            expected_output_colors.update(parameters["alternating"])

        if set(int(x) for x in np.unique(output)) != expected_output_colors:
            continue

        return grid, output

    raise RuntimeError("could not construct a valid pair")


def _array_key(array):
    return (
        tuple(int(x) for x in array.shape),
        tuple(int(x) for x in array.ravel()),
    )


def _pair_key(pair):
    return _array_key(pair["input"]), _array_key(pair["output"])


def _puzzle_key(puzzle):
    return (
        tuple(_pair_key(pair) for pair in puzzle["train"]),
        tuple(_pair_key(pair) for pair in puzzle["test"]),
    )


def _next_pair_rng(master_rng):
    seed = int(
        master_rng.integers(
            0,
            np.iinfo(np.int64).max,
            dtype=np.int64,
        )
    )
    return np.random.default_rng(seed)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _integer_argument(seed, "seed")
    seed_key = seed % (1 << 64)
    master_rng = np.random.default_rng(seed_key)

    num_train, num_test = _resolve_counts(
        seed_key, master_rng, num_train, num_test
    )
    parameters = _sample_puzzle_parameters(master_rng)

    train = []
    test = []
    seen_inputs = set()

    alternating_demo = 1 if num_train == 2 else 2

    for index in range(num_train):
        if index == 0:
            forced_mode = "solid"
            forced_fill = parameters["background"]
        elif num_train >= 3 and index == 1:
            forced_mode = "solid"
            forced_fill = parameters["fill"]
        elif index == alternating_demo:
            forced_mode = "alternating"
            forced_fill = (
                parameters["fill"]
                if num_train == 2
                else parameters["background"]
            )
        else:
            forced_mode = None
            forced_fill = None

        for _ in range(100):
            pair_rng = _next_pair_rng(master_rng)
            input_grid, output_grid = _generate_pair(
                pair_rng,
                "source_train",
                parameters,
                forced_mode=forced_mode,
                forced_fill=forced_fill,
            )
            key = _array_key(input_grid)
            if key not in seen_inputs:
                seen_inputs.add(key)
                train.append(
                    {"input": input_grid, "output": output_grid}
                )
                break
        else:
            raise RuntimeError("could not generate a unique training pair")

    for _ in range(num_test):
        for _ in range(100):
            pair_rng = _next_pair_rng(master_rng)
            input_grid, output_grid = _generate_pair(
                pair_rng,
                "target_test",
                parameters,
            )
            key = _array_key(input_grid)
            if key not in seen_inputs:
                seen_inputs.add(key)
                test.append(
                    {"input": input_grid, "output": output_grid}
                )
                break
        else:
            raise RuntimeError("could not generate a unique test pair")

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _locate_main(grid, output):
    height, width = grid.shape
    out_h, out_w = output.shape
    frame_h = out_h + 2
    frame_w = out_w + 2
    background = int(grid[0, 0])
    candidates = []

    for top in range(height - frame_h + 1):
        bottom = top + frame_h - 1
        for left in range(width - frame_w + 1):
            right = left + frame_w - 1
            frame_color = int(grid[top, left])

            if frame_color == background:
                continue

            border = []
            border.extend((top, c) for c in range(left, right + 1))
            border.extend(
                (bottom, c) for c in range(left, right + 1)
            )
            border.extend(
                (r, left) for r in range(top + 1, bottom)
            )
            border.extend(
                (r, right) for r in range(top + 1, bottom)
            )

            if not all(
                int(grid[r, c]) == frame_color for r, c in border
            ):
                continue

            interior = grid[top + 1:bottom, left + 1:right]
            fill = int(interior[0, 0])
            if not np.all(interior == fill):
                continue

            candidates.append(
                {
                    "bbox": (top, bottom, left, right),
                    "frame": frame_color,
                    "fill": fill,
                }
            )

    assert len(candidates) == 1, "main frame is not uniquely identifiable"
    return candidates[0]


def _corner_label(bbox, main_bbox):
    r0, r1, c0, c1 = bbox
    top, bottom, left, right = main_bbox

    inside_rows = [r for r in (r0, r1) if top < r < bottom]
    outside_rows = [
        r for r in (r0, r1) if r < top or r > bottom
    ]
    inside_cols = [c for c in (c0, c1) if left < c < right]
    outside_cols = [
        c for c in (c0, c1) if c < left or c > right
    ]

    assert len(inside_rows) == 1
    assert len(outside_rows) == 1
    assert len(inside_cols) == 1
    assert len(outside_cols) == 1

    vertical = "N" if outside_rows[0] < top else "S"
    horizontal = "W" if outside_cols[0] < left else "E"
    return vertical + horizontal


def _extract_solid_rectangle(grid, color, main_bbox):
    positions = {
        (int(r), int(c))
        for r, c in np.argwhere(grid == color)
    }
    assert positions, "solid outline color is absent"

    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    bbox = (min(rows), max(rows), min(cols), max(cols))
    full = _perimeter(bbox)
    visible = {
        p for p in full if not _inside_closed_main(p, main_bbox)
    }
    hidden = {
        p for p in full if _inside_strict_main(p, main_bbox)
    }

    assert positions == visible
    assert hidden
    corner = _corner_label(bbox, main_bbox)

    assignments = {p: color for p in hidden}
    descriptor = {
        "kind": "solid",
        "bbox": bbox,
        "corner": corner,
        "color": color,
    }
    return descriptor, assignments


def _connected_components(positions):
    remaining = set(positions)
    components = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}

        while stack:
            r, c = stack.pop()
            for neighbor in (
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def _extract_alternating_rectangles(
    grid,
    palette,
    main_bbox,
    expected_count,
):
    positions = {
        (int(r), int(c))
        for r, c in np.argwhere(
            (grid == palette[0]) | (grid == palette[1])
        )
    }
    assert positions, "alternating outline palette is absent"

    components = _connected_components(positions)
    assert len(components) == expected_count

    extracted = []
    for component in components:
        rows = [r for r, _ in component]
        cols = [c for _, c in component]
        bbox = (min(rows), max(rows), min(cols), max(cols))
        full = _perimeter(bbox)
        visible = {
            p for p in full
            if not _inside_closed_main(p, main_bbox)
        }
        hidden = {
            p for p in full
            if _inside_strict_main(p, main_bbox)
        }

        assert component == visible
        assert hidden

        parity_map = {}
        for r, c in component:
            parity = (r + c) & 1
            value = int(grid[r, c])
            if parity in parity_map:
                assert parity_map[parity] == value
            else:
                parity_map[parity] = value

        assert set(parity_map) == {0, 1}
        assert set(parity_map.values()) == set(palette)
        assert parity_map[0] != parity_map[1]

        assignments = {
            p: parity_map[(p[0] + p[1]) & 1]
            for p in hidden
        }
        assert set(assignments.values()) == set(palette)

        descriptor = {
            "kind": "alternating",
            "bbox": bbox,
            "corner": _corner_label(bbox, main_bbox),
            "palette": tuple(sorted(palette)),
            "even_color": parity_map[0],
        }
        extracted.append((descriptor, assignments))

    extracted.sort(key=lambda item: item[0]["bbox"])
    return extracted


def _extract_distractor(grid, color, main_bbox, background):
    positions = {
        (int(r), int(c))
        for r, c in np.argwhere(grid == color)
    }
    assert positions, "distractor color is absent"

    rows = [r for r, _ in positions]
    cols = [c for _, c in positions]
    bbox = (min(rows), max(rows), min(cols), max(cols))
    r0, r1, c0, c1 = bbox
    assert 3 <= r1 - r0 + 1 <= 5
    assert 3 <= c1 - c0 + 1 <= 5
    assert positions == _perimeter(bbox)

    top, bottom, left, right = main_bbox
    assert (
        r1 <= top - 2
        or r0 >= bottom + 2
        or c1 <= left - 2
        or c0 >= right + 2
    )

    interior = grid[r0 + 1:r1, c0 + 1:c1]
    assert np.all(interior == background)
    return bbox


def _validate_pair_structure(
    pair,
    main,
    roles,
    mode,
    expected_outline_count,
    split,
):
    grid = pair["input"]
    output = pair["output"]
    background = roles["background"]
    frame = roles["frame"]
    fill = main["fill"]
    main_bbox = main["bbox"]

    assert int(grid[0, 0]) == background
    assert int(grid[0, -1]) == background
    assert int(grid[-1, 0]) == background
    assert int(grid[-1, -1]) == background
    assert main["frame"] == frame

    descriptors = []
    assignment_groups = []

    if mode in {"solid", "mixed"}:
        for color in sorted(roles["solid"]):
            descriptor, assignments = _extract_solid_rectangle(
                grid, color, main_bbox
            )
            descriptors.append(descriptor)
            assignment_groups.append(assignments)

    if mode in {"alternating", "mixed"}:
        alternating_count = 2 if mode == "alternating" else 1
        extracted = _extract_alternating_rectangles(
            grid,
            tuple(sorted(roles["alternating"])),
            main_bbox,
            alternating_count,
        )
        for descriptor, assignments in extracted:
            descriptors.append(descriptor)
            assignment_groups.append(assignments)

    assert len(descriptors) == expected_outline_count

    if split == "train":
        active = (
            set(roles["solid"])
            if mode == "solid"
            else set(roles["alternating"])
        )
        expected_input_colors = {
            background,
            frame,
            fill,
        } | active
        expected_output_colors = {fill} | active
        distractor_boxes = []
    else:
        expected_input_colors = {
            background,
            frame,
            fill,
        } | set(roles["solid"]) | set(roles["alternating"]) | set(
            roles["distractor"]
        )
        expected_output_colors = {
            fill
        } | set(roles["solid"]) | set(roles["alternating"])
        distractor_boxes = [
            _extract_distractor(
                grid, color, main_bbox, background
            )
            for color in sorted(roles["distractor"])
        ]

    assert set(int(x) for x in np.unique(grid)) == expected_input_colors
    assert (
        set(int(x) for x in np.unique(output))
        == expected_output_colors
    )

    reconstructed = np.full(output.shape, fill, dtype=np.int8)
    used_hidden_cells = set()
    top, bottom, left, right = main_bbox

    for assignments in assignment_groups:
        for (r, c), color in assignments.items():
            assert top < r < bottom and left < c < right
            assert (r, c) not in used_hidden_cells
            used_hidden_cells.add((r, c))
            reconstructed[r - top - 1, c - left - 1] = color

    assert np.array_equal(reconstructed, output)

    return {
        "split": split,
        "mode": mode,
        "main": main,
        "input_shape": tuple(grid.shape),
        "output_shape": tuple(output.shape),
        "rectangles": descriptors,
        "distractors": distractor_boxes,
    }


def _inspect_structure(puzzle):
    train = puzzle["train"]
    test = puzzle["test"]

    train_mains = [
        _locate_main(pair["input"], pair["output"])
        for pair in train
    ]
    test_mains = [
        _locate_main(pair["input"], pair["output"])
        for pair in test
    ]

    background = int(train[0]["input"][0, 0])
    frame = train_mains[0]["frame"]

    assert train_mains[0]["fill"] == background
    special_fill = train_mains[1]["fill"]
    assert special_fill != background

    for pair, main in zip(train, train_mains):
        assert int(pair["input"][0, 0]) == background
        assert main["frame"] == frame
        assert main["fill"] in {background, special_fill}

    for pair, main in zip(test, test_mains):
        assert int(pair["input"][0, 0]) == background
        assert main["frame"] == frame
        assert main["fill"] == special_fill

    solid_colors = (
        set(int(x) for x in np.unique(train[0]["output"]))
        - {train_mains[0]["fill"]}
    )
    assert len(solid_colors) == 2

    alternating_demo = 1 if len(train) == 2 else 2
    alternating_colors = (
        set(
            int(x)
            for x in np.unique(
                train[alternating_demo]["output"]
            )
        )
        - {train_mains[alternating_demo]["fill"]}
    )
    assert len(alternating_colors) == 2
    assert solid_colors.isdisjoint(alternating_colors)

    core_colors = {
        background,
        frame,
        special_fill,
    } | solid_colors | alternating_colors
    assert len(core_colors) == 7

    distractor_colors = (
        set(int(x) for x in np.unique(test[0]["input"]))
        - core_colors
    )
    assert len(distractor_colors) == 2

    for pair in test:
        current_extra = (
            set(int(x) for x in np.unique(pair["input"]))
            - core_colors
        )
        assert current_extra == distractor_colors

    assert len(core_colors | distractor_colors) == 9

    roles = {
        "background": background,
        "frame": frame,
        "fill": special_fill,
        "solid": frozenset(solid_colors),
        "alternating": frozenset(alternating_colors),
        "distractor": frozenset(distractor_colors),
    }

    summaries = []
    modes = []

    for pair, main in zip(train, train_mains):
        nonfill = (
            set(int(x) for x in np.unique(pair["output"]))
            - {main["fill"]}
        )
        if nonfill == solid_colors:
            mode = "solid"
        elif nonfill == alternating_colors:
            mode = "alternating"
        else:
            raise AssertionError("training outline regime is inconsistent")

        modes.append(mode)
        summaries.append(
            _validate_pair_structure(
                pair,
                main,
                roles,
                mode,
                expected_outline_count=2,
                split="train",
            )
        )

    assert modes[0] == "solid"
    if len(train) == 2:
        assert modes[1] == "alternating"
    else:
        assert modes[1] == "solid"
        assert modes[2] == "alternating"
        assert train_mains[2]["fill"] == background

    assert train_mains[1]["fill"] == special_fill

    for pair, main in zip(test, test_mains):
        summaries.append(
            _validate_pair_structure(
                pair,
                main,
                roles,
                mode="mixed",
                expected_outline_count=3,
                split="test",
            )
        )

    return roles, summaries


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

    input_keys = set()
    pair_keys = set()

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0
                assert int(grid.max()) <= 9

            # Every original source output has one of these dimensions.
            # Avoiding all four is a sufficient exact non-duplication check.
            assert tuple(pair["output"].shape) not in _SOURCE_OUTPUT_SHAPES

            input_key = _array_key(pair["input"])
            pair_key = _pair_key(pair)
            assert input_key not in input_keys
            assert pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)

    _inspect_structure(puzzle)


def _puzzles_equal(first, second):
    return _puzzle_key(first) == _puzzle_key(second)


def self_test() -> None:
    automatic = generate_puzzle(12345)
    validate_generated_puzzle(automatic)

    explicit = generate_puzzle(54321, num_train=6, num_test=3)
    validate_generated_puzzle(explicit)

    same_first = generate_puzzle(777, num_train=5, num_test=2)
    same_second = generate_puzzle(777, num_train=5, num_test=2)
    assert _puzzles_equal(same_first, same_second)

    auto_first = generate_puzzle(778)
    auto_second = generate_puzzle(778)
    assert _puzzles_equal(auto_first, auto_second)

    different = [
        generate_puzzle(seed, num_train=5, num_test=2)
        for seed in range(8)
    ]
    assert len({_puzzle_key(p) for p in different}) >= 7

    boundary_cases = [
        generate_puzzle(9001, num_train=2, num_test=1),
        generate_puzzle(9002, num_train=10, num_test=1),
        generate_puzzle(9003, num_train=10, num_test=4),
    ]
    for puzzle in boundary_cases:
        validate_generated_puzzle(puzzle)

    partial_cases = [
        generate_puzzle(9101, num_train=None, num_test=1),
        generate_puzzle(9102, num_train=None, num_test=4),
        generate_puzzle(9103, num_train=2, num_test=None),
        generate_puzzle(9104, num_train=10, num_test=None),
    ]
    for puzzle in partial_cases:
        validate_generated_puzzle(puzzle)

    illegal_arguments = [
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    ]
    for arguments in illegal_arguments:
        raised = False
        try:
            generate_puzzle(999, **arguments)
        except ValueError:
            raised = True
        assert raised

    train_counts = set()
    test_counts = set()
    puzzle_keys = set()
    palette_signatures = set()
    input_shapes = set()
    output_shapes = set()
    pair_keys = set()
    layout_signatures = set()
    observed_phase_relations = set()

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        roles, summaries = _inspect_structure(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))

        palette_signatures.add(
            (
                roles["background"],
                roles["frame"],
                roles["fill"],
                tuple(sorted(roles["solid"])),
                tuple(sorted(roles["alternating"])),
                tuple(sorted(roles["distractor"])),
            )
        )

        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_shapes.add(tuple(pair["input"].shape))
                output_shapes.add(tuple(pair["output"].shape))
                pair_keys.add(_pair_key(pair))

                assert pair["input"].dtype == np.int8
                assert pair["output"].dtype == np.int8
                assert tuple(pair["output"].shape) not in (
                    _SOURCE_OUTPUT_SHAPES
                )

        for summary in summaries:
            rectangle_signature = tuple(
                sorted(
                    (
                        descriptor["kind"],
                        descriptor["corner"],
                        descriptor["bbox"][1]
                        - descriptor["bbox"][0]
                        + 1,
                        descriptor["bbox"][3]
                        - descriptor["bbox"][2]
                        + 1,
                    )
                    for descriptor in summary["rectangles"]
                )
            )
            layout_signatures.add(
                (
                    summary["split"],
                    summary["mode"],
                    summary["input_shape"],
                    summary["output_shape"],
                    summary["main"]["bbox"],
                    rectangle_signature,
                )
            )

            for descriptor in summary["rectangles"]:
                if descriptor["kind"] == "alternating":
                    observed_phase_relations.add(
                        descriptor["even_color"]
                        == min(roles["alternating"])
                    )

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(puzzle_keys) >= 40
    assert len(palette_signatures) >= 35
    assert len(input_shapes) >= 25
    assert len(output_shapes) >= 20
    assert len(pair_keys) >= 200
    assert len(layout_signatures) >= 150
    assert observed_phase_relations == {False, True}
