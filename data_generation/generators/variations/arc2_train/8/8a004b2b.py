import numpy as np


_SOURCE_OUTPUT_ROWS = (
    (
        "4000004",
        "0223300",
        "0223300",
        "0338800",
        "0338800",
        "0000000",
        "4000004",
    ),
    (
        "40000000000004",
        "00000000000000",
        "03300000022000",
        "03300000022000",
        "01111001111000",
        "01111001111000",
        "00011111100000",
        "00011111100000",
        "40000000000004",
    ),
    (
        "40000000004",
        "00000000000",
        "00000000000",
        "02221111110",
        "02221111110",
        "02221111110",
        "00003331110",
        "00003331110",
        "00003331110",
        "00000000000",
        "40000000004",
    ),
    (
        "400000000000000004",
        "000000000000000000",
        "000088880000333300",
        "000088880000333300",
        "000088880000333300",
        "000088880000333300",
        "000011111111111100",
        "000011111111111100",
        "000011111111111100",
        "000011111111111100",
        "000000001111000000",
        "000000001111000000",
        "000000001111000000",
        "400000001111000004",
    ),
)


def _grid_signature(grid: np.ndarray) -> tuple:
    return grid.shape, grid.dtype.str, grid.tobytes()


def _pair_signature(pair: dict) -> tuple:
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _grid_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _matches_a_source_output(pair: dict) -> bool:
    # Rejecting every original output is stronger than merely rejecting every
    # original complete pair.
    return _grid_rows(pair["output"]) in _SOURCE_OUTPUT_ROWS


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    start = tuple(int(x) for x in points[0])
    seen = {start}
    stack = [start]
    height, width = mask.shape

    while stack:
        row, col = stack.pop()
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_row, new_col = row + drow, col + dcol
            point = new_row, new_col
            if (
                0 <= new_row < height
                and 0 <= new_col < width
                and bool(mask[new_row, new_col])
                and point not in seen
            ):
                seen.add(point)
                stack.append(point)

    return len(seen) == len(points)


def _has_branch(mask: np.ndarray) -> bool:
    height, width = mask.shape

    for row_value, col_value in np.argwhere(mask):
        row, col = int(row_value), int(col_value)
        degree = 0
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            new_row, new_col = row + drow, col + dcol
            if (
                0 <= new_row < height
                and 0 <= new_col < width
                and bool(mask[new_row, new_col])
            ):
                degree += 1
        if degree >= 3:
            return True

    return False


def _sample_blueprint(
    rng: np.random.Generator,
    height: int,
    width: int,
    active_colors: tuple[int, ...],
    background: int,
) -> np.ndarray:
    """Sample a bounded pixel blueprint, including holes and separate islands."""
    mode = int(rng.integers(0, 6))
    if mode == 0:
        mask = np.ones((height, width), dtype=bool)
    elif mode == 1 and min(height, width) >= 3:
        mask = np.zeros((height, width), dtype=bool)
        mask[[0, -1], :] = True
        mask[:, [0, -1]] = True
    elif mode == 2:
        mask = np.zeros((height, width), dtype=bool)
        if width >= height:
            mask[int(rng.integers(height)), :] = True
        else:
            mask[:, int(rng.integers(width))] = True
        mask[int(rng.integers(height)), 0] = True
        mask[int(rng.integers(height)), -1] = True
        mask[0, int(rng.integers(width))] = True
        mask[-1, int(rng.integers(width))] = True
    else:
        density = float(rng.uniform(0.30, 0.88))
        mask = rng.random((height, width)) < density
        mask[int(rng.integers(height)), 0] = True
        mask[int(rng.integers(height)), -1] = True
        mask[0, int(rng.integers(width))] = True
        mask[-1, int(rng.integers(width))] = True

    positions = [tuple(map(int, point)) for point in np.argwhere(mask)]
    if len(positions) < max(3, len(active_colors)):
        raise ValueError("blueprint has too few occupied cells")

    # Some patterns have no singleton payload at all; others are deliberately
    # dominated by one color, as in the source's 3x5 blueprint.
    balanced = len(positions) >= 2 * len(active_colors) and bool(rng.integers(2))
    colors = list(active_colors) * (2 if balanced else 1)
    dominant = int(rng.integers(len(active_colors)))
    while len(colors) < len(positions):
        if rng.random() < 0.55:
            colors.append(active_colors[dominant])
        else:
            colors.append(active_colors[int(rng.integers(len(active_colors)))])
    rng.shuffle(colors)
    blueprint = np.full((height, width), background, dtype=np.int8)
    for (row, col), color in zip(positions, colors):
        blueprint[row, col] = color
    return blueprint


def _pattern_box_hits_marker(
    top: int,
    left: int,
    pattern_height: int,
    pattern_width: int,
    window_height: int,
    window_width: int,
) -> bool:
    corners = (
        (0, 0),
        (0, window_width - 1),
        (window_height - 1, 0),
        (window_height - 1, window_width - 1),
    )
    return any(
        top <= row < top + pattern_height
        and left <= col < left + pattern_width
        for row, col in corners
    )


def _full_alignment_solutions(
    output: np.ndarray,
    blueprint: np.ndarray,
    background: int,
    marker: int,
) -> list[tuple[int, int, int]]:
    window_height, window_width = output.shape
    blueprint_height, blueprint_width = blueprint.shape
    cells = [
        (int(row), int(col), int(blueprint[row, col]))
        for row, col in np.argwhere(blueprint != background)
    ]
    visible_payload = int(
        np.count_nonzero((output != background) & (output != marker))
    )
    solutions = []

    max_scale = min(
        window_height // blueprint_height,
        window_width // blueprint_width,
    )
    for scale in range(1, max_scale + 1):
        if visible_payload != len(cells) * scale * scale:
            continue

        pattern_height = blueprint_height * scale
        pattern_width = blueprint_width * scale

        for top in range(window_height - pattern_height + 1):
            for left in range(window_width - pattern_width + 1):
                if _pattern_box_hits_marker(
                    top,
                    left,
                    pattern_height,
                    pattern_width,
                    window_height,
                    window_width,
                ):
                    continue

                matches = True
                for row, col, color in cells:
                    block = output[
                        top + row * scale : top + (row + 1) * scale,
                        left + col * scale : left + (col + 1) * scale,
                    ]
                    if not bool(np.all(block == color)):
                        matches = False
                        break

                # The payload count proves that there are no unexplained
                # colored cells outside these disjoint blocks.
                if matches:
                    solutions.append((scale, top, left))

    return solutions


def _partial_alignment_solutions(
    crop: np.ndarray,
    blueprint: np.ndarray,
    background: int,
    marker: int,
) -> list[tuple[int, int, int, tuple[tuple[int, int], ...]]]:
    window_height, window_width = crop.shape
    blueprint_height, blueprint_width = blueprint.shape
    cells = [
        (int(row), int(col), int(blueprint[row, col]))
        for row, col in np.argwhere(blueprint != background)
    ]
    visible_payload = int(
        np.count_nonzero((crop != background) & (crop != marker))
    )
    solutions = []

    max_scale = min(
        window_height // blueprint_height,
        window_width // blueprint_width,
    )
    for scale in range(1, max_scale + 1):
        if visible_payload % (scale * scale) != 0:
            continue

        pattern_height = blueprint_height * scale
        pattern_width = blueprint_width * scale

        for top in range(window_height - pattern_height + 1):
            for left in range(window_width - pattern_width + 1):
                if _pattern_box_hits_marker(
                    top,
                    left,
                    pattern_height,
                    pattern_width,
                    window_height,
                    window_width,
                ):
                    continue

                revealed = []
                matches = True

                for row, col, color in cells:
                    block = crop[
                        top + row * scale : top + (row + 1) * scale,
                        left + col * scale : left + (col + 1) * scale,
                    ]
                    if bool(np.all(block == color)):
                        revealed.append((row, col))
                    elif not bool(np.all(block == background)):
                        matches = False
                        break

                if (
                    matches
                    and 0 < len(revealed) < len(cells)
                    and len(revealed) * scale * scale == visible_payload
                ):
                    solutions.append(
                        (scale, top, left, tuple(revealed))
                    )

    return solutions


def _analyze_marker_candidate(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    background: int,
    marker: int,
    bounds: tuple[int, int, int, int],
) -> dict:
    top, bottom, left, right = bounds
    assert output_grid.shape == (
        bottom - top + 1,
        right - left + 1,
    )

    crop = input_grid[top : bottom + 1, left : right + 1]
    corner_positions = (
        (0, 0),
        (0, crop.shape[1] - 1),
        (crop.shape[0] - 1, 0),
        (crop.shape[0] - 1, crop.shape[1] - 1),
    )
    assert all(
        int(crop[row, col]) == marker
        for row, col in corner_positions
    )
    assert all(
        int(output_grid[row, col]) == marker
        for row, col in corner_positions
    )
    assert int(np.count_nonzero(input_grid == marker)) == 4
    assert int(np.count_nonzero(output_grid == marker)) == 4

    outside = np.ones(input_grid.shape, dtype=bool)
    outside[top : bottom + 1, left : right + 1] = False
    external_points = np.argwhere(
        outside & (input_grid != background)
    )
    assert len(external_points) > 0

    blueprint_top = int(external_points[:, 0].min())
    blueprint_bottom = int(external_points[:, 0].max())
    blueprint_left = int(external_points[:, 1].min())
    blueprint_right = int(external_points[:, 1].max())
    blueprint = input_grid[
        blueprint_top : blueprint_bottom + 1,
        blueprint_left : blueprint_right + 1,
    ].copy()

    assert 2 <= blueprint.shape[0] <= 6
    assert 2 <= blueprint.shape[1] <= 6

    mask = blueprint != background
    assert (
        bool(mask[0].any())
        and bool(mask[-1].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )
    assert marker not in {
        int(value) for value in np.unique(blueprint)
    }
    assert outside[blueprint_top : blueprint_bottom + 1,
                   blueprint_left : blueprint_right + 1].all()
    assert int(np.count_nonzero(outside & (input_grid != background))) == int(np.count_nonzero(mask))

    full_solutions = _full_alignment_solutions(
        output_grid,
        blueprint,
        background,
        marker,
    )
    partial_solutions = _partial_alignment_solutions(
        crop,
        blueprint,
        background,
        marker,
    )
    assert len(full_solutions) == 1
    assert len(partial_solutions) == 1
    assert full_solutions[0] == partial_solutions[0][:3]
    assert not np.array_equal(crop, output_grid)

    return {
        "background": background,
        "marker": marker,
        "bounds": bounds,
        "blueprint": blueprint,
        "blueprint_bounds": (
            blueprint_top,
            blueprint_bottom,
            blueprint_left,
            blueprint_right,
        ),
        "payload_colors": frozenset(
            int(value)
            for value in np.unique(blueprint)
            if int(value) != background
        ),
        "scale": full_solutions[0][0],
        "offset": full_solutions[0][1:],
        "revealed": partial_solutions[0][3],
    }


def _analyze_pair(pair: dict) -> dict:
    input_grid = pair["input"]
    output_grid = pair["output"]

    values = np.unique(input_grid)
    analyses = []

    for marker_value in values:
        marker = int(marker_value)

        points = np.argwhere(input_grid == marker)
        if len(points) != 4:
            continue

        rows = sorted({int(point[0]) for point in points})
        cols = sorted({int(point[1]) for point in points})
        if len(rows) != 2 or len(cols) != 2:
            continue

        expected = {
            (rows[0], cols[0]),
            (rows[0], cols[1]),
            (rows[1], cols[0]),
            (rows[1], cols[1]),
        }
        actual = {
            (int(point[0]), int(point[1]))
            for point in points
        }
        if actual != expected:
            continue

        for background_value in values:
            background = int(background_value)
            if background == marker:
                continue
            try:
                analysis = _analyze_marker_candidate(
                    input_grid, output_grid, background, marker,
                    (rows[0], rows[1], cols[0], cols[1]),
                )
            except AssertionError:
                continue
            analyses.append(analysis)

    assert len(analyses) == 1
    return analyses[0]


def _render_full_output(
    blueprint: np.ndarray,
    scale: int,
    background: int,
    marker: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> np.ndarray:
    height = (
        top_margin + blueprint.shape[0] * scale + bottom_margin
    )
    width = (
        left_margin + blueprint.shape[1] * scale + right_margin
    )
    output = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )
    output[0, 0] = marker
    output[0, -1] = marker
    output[-1, 0] = marker
    output[-1, -1] = marker

    for row_value, col_value in np.argwhere(
        blueprint != background
    ):
        row, col = int(row_value), int(col_value)
        color = int(blueprint[row, col])
        output[
            top_margin + row * scale :
            top_margin + (row + 1) * scale,
            left_margin + col * scale :
            left_margin + (col + 1) * scale,
        ] = color

    return output


def _render_partial_crop(
    blueprint: np.ndarray,
    revealed_indices: list[int],
    scale: int,
    background: int,
    marker: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> np.ndarray:
    height = (
        top_margin + blueprint.shape[0] * scale + bottom_margin
    )
    width = (
        left_margin + blueprint.shape[1] * scale + right_margin
    )
    crop = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )
    crop[0, 0] = marker
    crop[0, -1] = marker
    crop[-1, 0] = marker
    crop[-1, -1] = marker

    occupied = [
        (int(row), int(col))
        for row, col in np.argwhere(blueprint != background)
    ]
    for index in revealed_indices:
        row, col = occupied[index]
        color = int(blueprint[row, col])
        crop[
            top_margin + row * scale :
            top_margin + (row + 1) * scale,
            left_margin + col * scale :
            left_margin + (col + 1) * scale,
        ] = color

    return crop


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    roles: dict,
    scale: int,
) -> dict:
    background, marker = roles["background"], roles["marker"]
    available = [c for c in range(10) if c not in (background, marker)]
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")

    for _ in range(600):
        try:
            height = int(rng.integers(2, 6))
            width = int(rng.integers(2, 7))
            if regime == "target_test" and rng.random() < 0.45:
                height = int(rng.integers(3, 6))
                width = int(rng.integers(3, 7))
            margins = [int(rng.integers(0, 6)), int(rng.integers(0, 5)),
                       int(rng.integers(0, 6)), int(rng.integers(0, 5))]
            out_h = margins[0] + height * scale + margins[1]
            out_w = margins[2] + width * scale + margins[3]
            if out_h > 29 or out_w > 29 or min(out_h, out_w) < 5:
                continue
            if _pattern_box_hits_marker(margins[0], margins[2],
                                        height * scale, width * scale,
                                        out_h, out_w):
                continue

            color_count = int(rng.integers(2, min(4, height * width) + 1))
            colors = tuple(int(c) for c in rng.choice(available, color_count, replace=False))
            blueprint = _sample_blueprint(rng, height, width, colors, background)
            output = _render_full_output(blueprint, scale, background, marker, *margins)
            if len(_full_alignment_solutions(output, blueprint, background, marker)) != 1:
                continue

            occupied = [tuple(map(int, p)) for p in np.argwhere(blueprint != background)]
            limit = min(len(occupied) - 1, 7)
            if rng.random() < 0.10:
                revealed_count = 1
            else:
                revealed_count = int(rng.integers(2, limit + 1))
            if revealed_count >= 2 and rng.random() < 0.7:
                first = int(rng.integers(len(occupied)))
                distances = [abs(p[0] - occupied[first][0]) + abs(p[1] - occupied[first][1])
                             for p in occupied]
                second = int(np.argmax(distances))
                remaining = [i for i in range(len(occupied)) if i not in (first, second)]
                extras = rng.choice(remaining, revealed_count - 2, replace=False)
                revealed = [first, second] + [int(i) for i in extras]
            else:
                revealed = [int(i) for i in rng.choice(len(occupied), revealed_count, replace=False)]
            crop = _render_partial_crop(blueprint, revealed, scale, background, marker, *margins)
            solutions = _partial_alignment_solutions(crop, blueprint, background, marker)
            if len(solutions) != 1 or solutions[0][:3] != (scale, margins[0], margins[2]):
                continue

            side = int(rng.integers(4))
            gap = int(rng.integers(1, 5))
            if side < 2:
                btop = -height - gap if side == 0 else out_h + gap
                bleft = int(rng.integers(-width + 1, out_w))
            else:
                bleft = -width - gap if side == 2 else out_w + gap
                btop = int(rng.integers(-height + 1, out_h))
            min_r, min_c = min(0, btop), min(0, bleft)
            max_r, max_c = max(out_h, btop + height), max(out_w, bleft + width)
            padding = [0 if rng.random() < 0.48 else int(rng.integers(1, 4)) for _ in range(4)]
            canvas_h = max_r - min_r + padding[0] + padding[1]
            canvas_w = max_c - min_c + padding[2] + padding[3]
            if canvas_h > 30 or canvas_w > 30:
                continue
            win_r, win_c = padding[0] - min_r, padding[2] - min_c
            bp_r, bp_c = win_r + btop, win_c + bleft
            input_grid = np.full((canvas_h, canvas_w), background, dtype=np.int8)
            input_grid[win_r:win_r + out_h, win_c:win_c + out_w] = crop
            input_grid[bp_r:bp_r + height, bp_c:bp_c + width] = blueprint
            pair = {"input": input_grid, "output": output}
            if _matches_a_source_output(pair):
                continue
            analysis = _analyze_pair(pair)
            if analysis["scale"] == scale:
                return pair
        except (AssertionError, ValueError):
            continue
    raise ValueError("unable to construct an unambiguous pair")


def _validate_count_value(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not low <= int(value) <= high:
        raise ValueError(f"{name} is outside its legal range")


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count_value("num_train", num_train, 2, 10)
    _validate_count_value("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        # This deterministic cycle guarantees broad count coverage over
        # consecutive fixed seeds.
        test_count = 1 + (int(seed) % 4)
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = (
            minimum_train + ((int(seed) // 4) % span)
        )
    elif num_train is None:
        test_count = int(num_test)
        minimum_train = max(2, test_count + 1)
        train_count = int(
            rng.integers(minimum_train, 11)
        )
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        test_count = int(
            rng.integers(1, maximum_test + 1)
        )
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    background = 0 if rng.random() < 0.35 else int(rng.integers(1, 10))
    marker = int(rng.choice([c for c in range(10) if c != background]))
    return {"background": background, "marker": marker,
            "vary_roles": bool(rng.integers(2))}


def _pair_roles(rng: np.random.Generator, parameters: dict, index: int) -> dict:
    if not parameters["vary_roles"] or index == 0:
        return {"background": parameters["background"],
                "marker": parameters["marker"]}
    # Literal color roles are local and identifiable from the grid geometry.
    background = int(rng.integers(10))
    marker = int(rng.choice([c for c in range(10) if c != background]))
    if index == 1 and (background, marker) == (
        parameters["background"], parameters["marker"]
    ):
        marker = (marker + 1) % 10
        if marker == background:
            marker = (marker + 1) % 10
    return {"background": background, "marker": marker}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(
        seed,
        rng,
        num_train,
        num_test,
    )
    puzzle_parameters = _sample_puzzle_parameters(rng)

    puzzle = {
        "train": [],
        "test": [],
    }
    used_inputs = set()
    train_scales = []
    for index in range(train_count + test_count):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        regime = "source_train" if index < train_count else "target_test"
        if regime == "source_train":
            scale = int(pair_rng.choice((2, 3, 4), p=(0.48, 0.38, 0.14)))
            if index == 1 and scale == train_scales[0]:
                scale = 3 if scale == 2 else 2
            train_scales.append(scale)
        else:
            # Larger factors are more common in tests, but familiar scales
            # also occur, so scale itself is never a split convention.
            scale = int(pair_rng.choice((2, 3, 4, 5), p=(0.16, 0.21, 0.43, 0.20)))
        roles = _pair_roles(pair_rng, puzzle_parameters, index)
        for _ in range(8):
            pair = _generate_pair(pair_rng, regime, roles, scale)
            signature = _grid_signature(pair["input"])
            if signature not in used_inputs:
                used_inputs.add(signature)
                puzzle["train" if index < train_count else "test"].append(pair)
                break
        else:
            raise ValueError("could not find a distinct pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    all_pairs = puzzle["train"] + puzzle["test"]
    input_signatures = set()
    pair_signatures = set()
    analyses = []

    for pair in all_pairs:
        assert isinstance(pair, dict)
        assert set(pair) == {"input", "output"}

        for key in ("input", "output"):
            grid = pair[key]
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30
            assert 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert bool(np.all((0 <= grid) & (grid <= 9)))

        input_signature = _grid_signature(pair["input"])
        pair_signature = _pair_signature(pair)
        assert input_signature not in input_signatures
        assert pair_signature not in pair_signatures
        input_signatures.add(input_signature)
        pair_signatures.add(pair_signature)

        analysis = _analyze_pair(pair)
        analyses.append(analysis)

        assert not _matches_a_source_output(pair)

    train_analyses = analyses[:train_count]
    test_analyses = analyses[train_count:]

    assert len({item["scale"] for item in train_analyses}) >= 2
    for item in analyses:
        assert item["background"] != item["marker"]
        assert 2 <= item["scale"] <= 5
        assert len(item["revealed"]) >= 1
        assert len(item["revealed"]) < int(np.count_nonzero(
            item["blueprint"] != item["background"]
        ))
    # Each pair independently displays its background, four matching crop
    # markers, literal blueprint colors, and reference blocks. No hidden
    # literal color mapping or fixed placement is required by a test.


def _puzzles_equal(first: dict, second: dict) -> bool:
    if (
        set(first) != {"train", "test"}
        or set(second) != {"train", "test"}
    ):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
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


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        _pair_signature(pair)
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    puzzle_signatures, motif_signatures, layout_signatures = set(), set(), set()
    train_scales, test_scales, sides = set(), set(), set()
    properties = {name: 0 for name in (
        "full", "dense", "wide", "repeat", "no_singleton", "disconnected",
        "branch", "one_reference", "multiple_references", "separated_references",
        "large_top_margin", "large_left_margin", "background_zero",
        "edge_window", "inset_window", "nonbelow", "new_test_color",
        "within_role_change", "mixed_test_scale", "combined"
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_signatures.add(_puzzle_signature(puzzle))
        pairs = puzzle["train"] + puzzle["test"]
        analyses = [_analyze_pair(pair) for pair in pairs]
        train = analyses[:len(puzzle["train"])]
        tests = analyses[len(puzzle["train"]):]
        train_scales.update(a["scale"] for a in train)
        test_scales.update(a["scale"] for a in tests)
        if len({(a["background"], a["marker"]) for a in train}) > 1:
            properties["within_role_change"] += 1
        if len({a["scale"] for a in tests}) > 1:
            properties["mixed_test_scale"] += 1
        train_colors = set().union(*(a["payload_colors"] for a in train))
        if any(not a["payload_colors"].issubset(train_colors) for a in tests):
            properties["new_test_color"] += 1
        for pair, a in zip(pairs, analyses):
            bp, bg = a["blueprint"], a["background"]
            mask = bp != bg
            n = int(mask.sum())
            counts = [int(np.count_nonzero(bp == c)) for c in a["payload_colors"]]
            top, bottom, left, right = a["bounds"]
            bt, bb, bl, br = a["blueprint_bounds"]
            side = "above" if bb < top else "below" if bt > bottom else "left" if br < left else "right"
            sides.add(side)
            motif_signatures.add((mask.shape, tuple(bool(v) for v in mask.ravel())))
            layout_signatures.add((pair["input"].shape, pair["output"].shape,
                                   a["bounds"], a["blueprint_bounds"], a["offset"], a["scale"]))
            properties["full"] += n == bp.size
            properties["dense"] += n / bp.size >= .75
            properties["wide"] += bp.shape[1] >= 5
            properties["repeat"] += max(counts) >= 4
            properties["no_singleton"] += min(counts) >= 2
            properties["disconnected"] += not _is_connected(mask)
            properties["branch"] += _has_branch(mask)
            properties["one_reference"] += len(a["revealed"]) == 1
            properties["multiple_references"] += len(a["revealed"]) >= 2
            properties["separated_references"] += any(
                abs(x[0]-y[0]) + abs(x[1]-y[1]) >= 2
                for x in a["revealed"] for y in a["revealed"])
            properties["large_top_margin"] += a["offset"][0] >= 3
            properties["large_left_margin"] += a["offset"][1] >= 4
            properties["background_zero"] += bg == 0
            properties["edge_window"] += (top == 0 or left == 0 or bottom == pair["input"].shape[0]-1
                                           or right == pair["input"].shape[1]-1)
            properties["inset_window"] += (top > 0 and left > 0 and bottom < pair["input"].shape[0]-1
                                            and right < pair["input"].shape[1]-1)
            properties["nonbelow"] += side != "below"
            properties["combined"] += side != "below" and min(counts) >= 2 and len(a["revealed"]) >= 2
            assert not _matches_a_source_output(pair)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) == 42
    assert len(motif_signatures) >= 30 and len(layout_signatures) >= 100
    assert train_scales >= {2, 3, 4} and test_scales >= {2, 3, 4, 5}
    assert sides == {"above", "below", "left", "right"}
    assert all(properties[name] > 0 for name in properties)

    a = generate_puzzle(314159, 6, 2)
    b = generate_puzzle(314159, 6, 2)
    assert _puzzles_equal(a, b)
    assert not _puzzles_equal(a, generate_puzzle(314160, 6, 2))
    for seed, nt, ns in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4)):
        p = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(p)
        assert (len(p["train"]), len(p["test"])) == (nt, ns)
    for seed, nt, ns in ((2001, None, 1), (2002, None, 4),
                         (2003, 2, None), (2004, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        validate_generated_puzzle(p)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {kwargs}")
