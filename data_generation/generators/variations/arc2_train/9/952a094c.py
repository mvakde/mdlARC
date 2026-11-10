import numpy as np


_SOURCE_PAIR_TEXTS = frozenset([
    (
        "0000000000\n"
        "0000000000\n"
        "0077777700\n"
        "0072008700\n"
        "0070000700\n"
        "0070000700\n"
        "0079003700\n"
        "0077777700\n"
        "0000000000\n"
        "0000000000",
        "0000000000\n"
        "0300000090\n"
        "0077777700\n"
        "0070000700\n"
        "0070000700\n"
        "0070000700\n"
        "0070000700\n"
        "0077777700\n"
        "0800000020\n"
        "0000000000",
    ),
    (
        "0000000000\n"
        "0000000000\n"
        "0008888000\n"
        "0008438000\n"
        "0008008000\n"
        "0008268000\n"
        "0008888000\n"
        "0000000000\n"
        "0000000000\n"
        "0000000000",
        "0000000000\n"
        "0060000200\n"
        "0008888000\n"
        "0008008000\n"
        "0008008000\n"
        "0008008000\n"
        "0008888000\n"
        "0030000400\n"
        "0000000000\n"
        "0000000000",
    ),
    (
        "0000000000\n"
        "0111111000\n"
        "0120051000\n"
        "0100001000\n"
        "0160031000\n"
        "0111111000\n"
        "0000000000\n"
        "0000000000\n"
        "0000000000\n"
        "0000000000",
        "3000000600\n"
        "0111111000\n"
        "0100001000\n"
        "0100001000\n"
        "0100001000\n"
        "0111111000\n"
        "5000000200\n"
        "0000000000\n"
        "0000000000\n"
        "0000000000",
    ),
    (
        "0000000000\n"
        "0000000000\n"
        "0000000000\n"
        "0033333300\n"
        "0036004300\n"
        "0030000300\n"
        "0038002300\n"
        "0033333300\n"
        "0000000000\n"
        "0000000000",
        "0000000000\n"
        "0000000000\n"
        "0200000080\n"
        "0033333300\n"
        "0030000300\n"
        "0030000300\n"
        "0030000300\n"
        "0033333300\n"
        "0400000060\n"
        "0000000000",
    ),
])


def _normalize_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _normalize_count(num_train, "num_train")
    num_test = _normalize_count(num_test, "num_test")

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if (
        num_train is not None
        and num_test is not None
        and num_train <= num_test
    ):
        raise ValueError("num_train must be greater than num_test")

    if num_train is None and num_test is None:
        # This deterministic schedule guarantees broad count diversity over
        # consecutive seeds while always producing a legal combination.
        num_test = 1 + seed % 4
        lowest_train = num_test + 1
        number_of_choices = 11 - lowest_train
        num_train = lowest_train + (seed // 4) % number_of_choices
    elif num_train is None:
        lowest_train = max(2, num_test + 1)
        num_train = int(rng.integers(lowest_train, 11))
    elif num_test is None:
        highest_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, highest_test + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("the requested pair counts cannot form a legal puzzle")

    return num_train, num_test


def _sample_puzzle_parameters(rng, num_train, num_test):
    """Choose color regimes and a canvas regime before constructing pairs.

    Color is never a hidden transformation parameter: every pair visibly
    identifies its background, frame, and four payloads. These modes only
    control which kinds of role reuse a whole puzzle demonstrates.
    """
    total = num_train + num_test
    color_mode = ("stable", "varied", "transfer")[int(rng.integers(0, 3))]
    background_mode = ("zero", "nonzero", "changing")[int(rng.integers(0, 3))]
    fixed_background = (
        0 if background_mode == "zero" else
        int(rng.integers(1, 10)) if background_mode == "nonzero" else None
    )
    palette = [int(x) for x in rng.permutation(10)]
    if fixed_background is not None:
        palette.remove(fixed_background)
    # Some puzzles reserve a literal hue for the test. Its behavior is still
    # determined by geometry and the demonstrated corner correspondence.
    novel_hue = palette[-1] if rng.integers(0, 4) == 0 else None
    train_palette = [c for c in palette if c != novel_hue]

    anchor = palette[0]
    if color_mode == "stable":
        frames = [anchor] * total
    elif color_mode == "transfer":
        frame_pool = [c for c in train_palette if c != anchor]
        train_frames = [frame_pool[i % len(frame_pool)] for i in range(num_train)]
        frames = train_frames + [anchor] * num_test
    else:
        # The first two demonstrations actually use different frame colors.
        frames = [train_palette[0], train_palette[1]]
        frames.extend(int(rng.choice(train_palette)) for _ in range(num_train - 2))
        for _ in range(num_test):
            unseen = [c for c in palette if c not in frames[:num_train]]
            if novel_hue is not None and len(frames) == num_train and rng.integers(0, 2):
                frames.append(novel_hue)
            else:
                pool = unseen if unseen and rng.integers(0, 2) else palette
                frames.append(int(rng.choice(pool)))

    backgrounds = []
    for index, frame in enumerate(frames):
        if fixed_background is not None:
            background = fixed_background
        else:
            excluded = {frame}
            if color_mode == "transfer" and index < num_train:
                excluded.add(anchor)
            if novel_hue is not None:
                excluded.add(novel_hue)
            pool = [c for c in range(10) if c not in excluded]
            if index == 1:
                pool = [c for c in pool if c != backgrounds[0]]
            background = int(rng.choice(pool))
        backgrounds.append(background)

    roles = []
    first_anchor_corner = int(rng.integers(0, 4))
    for index, (frame, background) in enumerate(zip(frames, backgrounds)):
        required = []
        if color_mode == "transfer":
            if index < num_train:
                required.append(anchor)
            else:
                eligible = [c for c in frames[:num_train]
                            if c not in (frame, background)]
                if eligible:
                    required.append(int(rng.choice(eligible)))
        if novel_hue is not None and index == num_train and frame != novel_hue:
            required.append(novel_hue)
        pool = [c for c in range(10)
                if c not in (frame, background, *required)
                and (index >= num_train or c != novel_hue)]
        marker_count = 4 - len(required)
        markers = [int(c) for c in rng.choice(pool, size=marker_count, replace=False)]
        markers.extend(required)
        if anchor in required and index < num_train:
            corner = (first_anchor_corner + index) % 4
            others = iter(int(c) for c in rng.permutation([c for c in markers if c != anchor]))
            markers = [anchor if k == corner else next(others) for k in range(4)]
        else:
            markers = [int(c) for c in rng.permutation(markers)]
        roles.append((background, frame, tuple(markers)))

    return {
        "roles": tuple(roles),
        "shared_canvas": bool(rng.integers(0, 4) == 0),
    }


def _build_pair(
    height,
    width,
    top,
    left,
    frame_height,
    frame_width,
    background,
    frame_color,
    markers,
):
    bottom = top + frame_height - 1
    right = left + frame_width - 1

    input_grid = np.full((height, width), background, dtype=np.int8)

    input_grid[top, left:right + 1] = frame_color
    input_grid[bottom, left:right + 1] = frame_color
    input_grid[top:bottom + 1, left] = frame_color
    input_grid[top:bottom + 1, right] = frame_color

    inner_positions = (
        (top + 1, left + 1),
        (top + 1, right - 1),
        (bottom - 1, left + 1),
        (bottom - 1, right - 1),
    )
    for position, color in zip(inner_positions, markers):
        input_grid[position] = color

    output_grid = input_grid.copy()
    for position in inner_positions:
        output_grid[position] = background

    # Output exterior order is top-left, top-right, bottom-left,
    # bottom-right. It receives input BR, BL, TR, TL respectively.
    exterior_positions = (
        (top - 1, left - 1),
        (top - 1, right + 1),
        (bottom + 1, left - 1),
        (bottom + 1, right + 1),
    )
    rotated_markers = (
        markers[3],
        markers[2],
        markers[1],
        markers[0],
    )
    for position, color in zip(exterior_positions, rotated_markers):
        output_grid[position] = color

    return input_grid, output_grid


def _sample_frame_shape(rng, category):
    if category == "ordinary":
        return int(rng.integers(5, 7)), int(rng.integers(5, 7))
    if category == "small":
        return 4, 4
    if category == "long_wide":
        return int(rng.integers(4, 7)), int(rng.integers(10, 19))
    if category == "long_tall":
        return int(rng.integers(10, 19)), int(rng.integers(4, 7))
    if category == "large":
        return int(rng.integers(8, 19)), int(rng.integers(8, 19))
    return int(rng.integers(4, 19)), int(rng.integers(4, 19))


def _sample_shapes(rng, num_train, num_test, source_canvas):
    categories = ("ordinary", "small", "long_wide", "long_tall", "large", "broad")
    if source_canvas:
        # A few compact puzzles retain the source's common 10x10 canvas,
        # with both a familiar frame and an adjacent-marker 4x4 frame.
        chosen = ["ordinary", "small"]
        chosen.extend(("ordinary", "small")[int(rng.integers(0, 2))]
                      for _ in range(num_train - 2))
        chosen.append("ordinary")
        return tuple(_sample_frame_shape(rng, category) for category in chosen)
    # An ordinary demonstration anchors the familiar source layout. The
    # second reveals that shape does not control the corner operation.
    train_categories = ["ordinary", categories[1 + int(rng.integers(0, 5))]]
    train_categories.extend(
        categories[int(rng.integers(0, len(categories)))]
        for _ in range(num_train - 2)
    )
    test_categories = []
    for _ in range(num_test):
        pool = [c for c in categories if not test_categories or c != test_categories[-1]]
        test_categories.append(pool[int(rng.integers(0, len(pool)))])
    return tuple(_sample_frame_shape(rng, category)
                 for category in train_categories + test_categories)


def _sample_canvas(rng, frame_height, frame_width):
    height = frame_height + 2 + int(rng.integers(0, min(12, 28 - frame_height) + 1))
    width = frame_width + 2 + int(rng.integers(0, min(12, 28 - frame_width) + 1))
    return height, width


def _generate_pair(rng, shape, canvas, role):
    frame_height, frame_width = shape
    height, width = canvas
    top = int(rng.integers(1, height - frame_height))
    left = int(rng.integers(1, width - frame_width))
    background, frame_color, markers = role
    return _build_pair(height, width, top, left, frame_height, frame_width,
                       background, frame_color, markers)


def _array_key(grid):
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _grid_text(grid):
    return "\n".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(input_grid, output_grid):
    return (
        _grid_text(input_grid),
        _grid_text(output_grid),
    ) in _SOURCE_PAIR_TEXTS


def _analyze_pair(input_grid, output_grid):
    assert isinstance(input_grid, np.ndarray)
    assert isinstance(output_grid, np.ndarray)
    assert input_grid.ndim == output_grid.ndim == 2
    assert input_grid.shape == output_grid.shape
    assert all(1 <= size <= 30 for size in input_grid.shape)
    assert np.issubdtype(input_grid.dtype, np.integer)
    assert np.issubdtype(output_grid.dtype, np.integer)
    assert bool(np.all((0 <= input_grid) & (input_grid <= 9)))
    assert bool(np.all((0 <= output_grid) & (output_grid <= 9)))

    background = int(input_grid[0, 0])
    input_colors = {int(color) for color in np.unique(input_grid)}
    assert len(input_colors) == 6
    assert background in input_colors

    frame_candidates = []
    for color in input_colors - {background}:
        rows, columns = np.where(input_grid == color)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(columns.min()), int(columns.max())
        if bottom - top + 1 < 4 or right - left + 1 < 4:
            continue
        expected = np.zeros(input_grid.shape, dtype=bool)
        expected[top, left:right + 1] = True
        expected[bottom, left:right + 1] = True
        expected[top:bottom + 1, left] = True
        expected[top:bottom + 1, right] = True
        if np.array_equal(input_grid == color, expected):
            frame_candidates.append((color, top, bottom, left, right))
    assert len(frame_candidates) == 1
    frame_color, top, bottom, left, right = frame_candidates[0]
    height, width = input_grid.shape
    margins = (top, height - bottom - 1, left, width - right - 1)
    assert min(margins) >= 1

    inner = ((top + 1, left + 1), (top + 1, right - 1),
             (bottom - 1, left + 1), (bottom - 1, right - 1))
    assert len(set(inner)) == 4
    markers = tuple(int(input_grid[position]) for position in inner)
    assert len(set(markers)) == 4
    assert background not in markers and frame_color not in markers

    expected_input = np.full(input_grid.shape, background, dtype=np.int8)
    expected_input[top, left:right + 1] = frame_color
    expected_input[bottom, left:right + 1] = frame_color
    expected_input[top:bottom + 1, left] = frame_color
    expected_input[top:bottom + 1, right] = frame_color
    for position, color in zip(inner, markers):
        expected_input[position] = color
    assert np.array_equal(input_grid, expected_input)

    expected_output = expected_input.copy()
    for position in inner:
        expected_output[position] = background
    exterior = ((top - 1, left - 1), (top - 1, right + 1),
                (bottom + 1, left - 1), (bottom + 1, right + 1))
    for position, color in zip(exterior, markers[::-1]):
        expected_output[position] = color
    assert np.array_equal(output_grid, expected_output)
    assert int(np.count_nonzero(input_grid != output_grid)) == 8

    return {
        "background": background,
        "frame_color": frame_color,
        "markers": markers,
        "frame_height": bottom - top + 1,
        "frame_width": right - left + 1,
        "top_margin": margins[0],
        "bottom_margin": margins[1],
        "left_margin": margins[2],
        "right_margin": margins[3],
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    num_train, num_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test
    seen_inputs = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            key = _array_key(input_grid)
            assert key not in seen_inputs
            seen_inputs.add(key)
            assert not _is_source_pair(input_grid, output_grid)
            _analyze_pair(input_grid, output_grid)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng, num_train, num_test)
    source_canvas = num_train <= 4 and num_test == 1 and seed % 3 == 0
    shapes = _sample_shapes(rng, num_train, num_test, source_canvas)
    if source_canvas:
        canvases = ((10, 10),) * len(shapes)
    elif parameters["shared_canvas"]:
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)
        canvas = (
            max_height + 2 + int(rng.integers(0, min(8, 28 - max_height) + 1)),
            max_width + 2 + int(rng.integers(0, min(8, 28 - max_width) + 1)),
        )
        canvases = (canvas,) * len(shapes)
    else:
        canvases = tuple(_sample_canvas(rng, *shape) for shape in shapes)

    nonce = int(rng.integers(0, 2**32, dtype=np.uint64))
    sequences = np.random.SeedSequence([nonce, num_train, num_test]).spawn(len(shapes))
    pair_rngs = [np.random.default_rng(sequence) for sequence in sequences]
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for index, (shape, canvas, role, pair_rng) in enumerate(zip(
        shapes, canvases, parameters["roles"], pair_rngs
    )):
        for _ in range(512):
            input_grid, output_grid = _generate_pair(pair_rng, shape, canvas, role)
            key = _array_key(input_grid)
            if key in seen_inputs or _is_source_pair(input_grid, output_grid):
                continue
            seen_inputs.add(key)
            split = "train" if index < num_train else "test"
            puzzle[split].append({"input": input_grid, "output": output_grid})
            break
        else:
            raise RuntimeError("could not construct a unique valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(
        (
            regime,
            tuple(
                (
                    _array_key(pair["input"]),
                    _array_key(pair["output"]),
                )
                for pair in puzzle[regime]
            ),
        )
        for regime in ("train", "test")
    )


def _same_puzzle(first, second):
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for regime in ("train", "test"):
        if len(first[regime]) != len(second[regime]):
            return False
        for pair_a, pair_b in zip(first[regime], second[regime]):
            for grid_name in ("input", "output"):
                grid_a = pair_a[grid_name]
                grid_b = pair_b[grid_name]
                if grid_a.dtype != grid_b.dtype:
                    return False
                if not np.array_equal(grid_a, grid_b):
                    return False
    return True


def self_test() -> None:
    train_counts, test_counts, puzzle_keys = set(), set(), set()
    grid_shapes, frame_shapes, test_frame_shapes = set(), set(), set()
    margins = {"train": {"high": 0, "low": 0, "center": 0},
               "test": {"high": 0, "low": 0, "center": 0}}
    one_cell_sides = set()
    backgrounds, frame_colors = set(), set()
    stable_frame_puzzles = varied_frame_puzzles = 0
    changing_background_puzzles = shared_canvas_puzzles = varied_canvas_puzzles = 0
    shared_ten_by_ten_puzzles = 0
    varied_test_shape_puzzles = new_test_hue_puzzles = transfer_puzzles = 0
    small_tests = long_tests = large_tests = combined_pairs = 0
    compact_puzzles = rich_puzzles = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        if len(puzzle["train"]) <= 4 and len(puzzle["test"]) == 1:
            compact_puzzles += 1
        if len(puzzle["train"]) >= 7 and len(puzzle["test"]) >= 2:
            rich_puzzles += 1
        analyses = {split: [_analyze_pair(p["input"], p["output"])
                            for p in puzzle[split]] for split in ("train", "test")}
        all_analyses = analyses["train"] + analyses["test"]
        all_pairs = puzzle["train"] + puzzle["test"]
        frame_set = {a["frame_color"] for a in all_analyses}
        stable_frame_puzzles += len(frame_set) == 1
        varied_frame_puzzles += len(frame_set) > 1
        changing_background_puzzles += len({a["background"] for a in all_analyses}) > 1
        canvas_set = {p["input"].shape for p in all_pairs}
        shared_canvas_puzzles += len(canvas_set) == 1
        varied_canvas_puzzles += len(canvas_set) > 1
        shared_ten_by_ten_puzzles += canvas_set == {(10, 10)}
        test_shapes = {(a["frame_height"], a["frame_width"])
                       for a in analyses["test"]}
        varied_test_shape_puzzles += len(test_shapes) > 1
        train_palette = set(int(c) for p in puzzle["train"]
                            for c in np.unique(p["input"]))
        new_test_hue_puzzles += any(
            int(c) not in train_palette for p in puzzle["test"]
            for c in np.unique(p["input"])
        )
        train_frames = {a["frame_color"] for a in analyses["train"]}
        train_markers = {c for a in analyses["train"] for c in a["markers"]}
        transfer_puzzles += any(
            a["frame_color"] in train_markers
            and a["frame_color"] not in train_frames
            for a in analyses["test"]
        )

        for split in ("train", "test"):
            for pair, analysis in zip(puzzle[split], analyses[split]):
                shape = (analysis["frame_height"], analysis["frame_width"])
                grid_shapes.add(pair["input"].shape)
                frame_shapes.add(shape)
                backgrounds.add(analysis["background"])
                frame_colors.add(analysis["frame_color"])
                if split == "test":
                    test_frame_shapes.add(shape)
                    small_tests += shape == (4, 4)
                    long_tests += min(shape) <= 6 and max(shape) >= 10
                    large_tests += min(shape) >= 8
                top, bottom = analysis["top_margin"], analysis["bottom_margin"]
                relation = "high" if top < bottom else "low" if top > bottom else "center"
                margins[split][relation] += 1
                for name in ("top", "bottom", "left", "right"):
                    if analysis[name + "_margin"] == 1:
                        one_cell_sides.add(name)
                combined_pairs += (
                    analysis["background"] != 0
                    and min(shape) >= 8
                    and abs(top - bottom) >= 3
                    and analysis["left_margin"] != analysis["right_margin"]
                )

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) >= 40
    assert len(grid_shapes) >= 25 and len(frame_shapes) >= 35
    assert len(test_frame_shapes) >= 15
    assert min(small_tests, long_tests, large_tests) > 0
    assert all(min(counts.values()) > 0 for counts in margins.values())
    assert one_cell_sides == {"top", "bottom", "left", "right"}
    assert len(backgrounds) >= 7 and len(frame_colors) >= 7
    assert min(stable_frame_puzzles, varied_frame_puzzles) > 0
    assert min(changing_background_puzzles, shared_canvas_puzzles,
               varied_canvas_puzzles, varied_test_shape_puzzles,
               new_test_hue_puzzles, transfer_puzzles) > 0
    assert shared_ten_by_ten_puzzles > 0
    assert min(compact_puzzles, rich_puzzles, combined_pairs) > 0

    for seed, train_count, test_count in ((101, 2, 1), (102, 10, 1),
                                           (103, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for seed, train_count, test_count in ((201, None, 1), (202, None, 4),
                                           (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    assert _same_puzzle(generate_puzzle(12345, 7, 3),
                        generate_puzzle(12345, 7, 3))
    assert len({_puzzle_key(generate_puzzle(seed, 6, 2))
                for seed in range(1000, 1010)}) >= 9
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count request was accepted: {request}")
