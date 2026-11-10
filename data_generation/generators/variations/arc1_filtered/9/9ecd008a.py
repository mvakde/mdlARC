"""Mirrored mosaics with one rectangular patch replaced by an alien color."""

import numpy as np


_SOURCE_PAIRS = tuple(
    (
        tuple(tuple(int(ch) for ch in row) for row in image.split()),
        tuple(tuple(int(ch) for ch in row) for row in crop.split()),
    )
    for image, crop in (
        (
            """2135111881115312 1257178888717521 3544182992814453
            5744889229884475 1118441111448111 1788000991748871
            1829000331119281 8892000113912988 8892193113912988
            1829111331119281 1788471991748871 1118441111448111
            5744889229884475 3544182992814453 1257178888717521
            2135111881115312""",
            "471 111 193",
        ),
        (
            """3331776666771333 3313776116773133 3188669779668813
            1385617997165831 7766335115336677 7761331111331677
            6697516116157966 6179111441119716 6179000441119716
            6697000116157966 7761000111331677 7766335115336677
            1385617997165831 3188669779668813 3313776116773133
            3331776666771333""",
            "111 516 331",
        ),
        (
            """9353395555933539 3936955885596393 5333556666553335
            3636586666856363 3955552112555593 9558581661858559
            5566219339126655 5866163993000685 5866163993000685
            5566219339000655 9558581661858559 3955552112555593
            3636586666856363 5333556666553335 3936955885596393
            9353395555933539""",
            "616 616 126",
        ),
        (
            """4899665115669984 8699671551769968 9952515555152599
            9922155995512299 6651145225411566 6000442772445176
            5000529559255515 1000275995729551 1559275995729551
            5155529559255515 6715442772445176 6651145225411566
            9922155995512299 9952515555152599 8699671551769968
            4899665115669984""",
            "715 155 559",
        ),
    )
)


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _mirror(quadrant, height, width):
    right = quadrant[:, :width // 2][:, ::-1]
    top = np.concatenate((quadrant, right), axis=1)
    bottom = top[:height // 2, :][::-1, :]
    return np.concatenate((top, bottom), axis=0)


def _paint_design(rng, height, width, colors, style):
    qh, qw = (height + 1) // 2, (width + 1) // 2
    base = int(rng.choice(colors))
    quadrant = np.full((qh, qw), base, dtype=np.int8)
    if style == "mosaic":
        weights = (np.ones(len(colors)) / len(colors)
                   if rng.random() < 0.5 else rng.dirichlet(np.ones(len(colors))))
        quadrant[:, :] = rng.choice(colors, size=(qh, qw), p=weights)
    elif style == "bands":
        for axis, length in ((0, qh), (1, qw)):
            for _ in range(int(rng.integers(2, 6))):
                start = int(rng.integers(length))
                stop = min(length, start + int(rng.integers(1, max(3, length // 2 + 1))))
                color = int(rng.choice(colors))
                if axis == 0:
                    quadrant[start:stop, :] = color
                else:
                    quadrant[:, start:stop] = color
    elif style == "blocks":
        bh = int(rng.integers(2, min(6, qh) + 1))
        bw = int(rng.integers(2, min(6, qw) + 1))
        blocks = rng.choice(colors, size=(bh, bw))
        rows = np.minimum(bh - 1, np.arange(qh) * bh // qh)
        cols = np.minimum(bw - 1, np.arange(qw) * bw // qw)
        quadrant[:, :] = blocks[rows[:, None], cols[None, :]]
    elif style == "diagonals":
        period = int(rng.integers(2, 6))
        row = np.arange(qh)[:, None]
        col = np.arange(qw)[None, :]
        quadrant[(row + col + int(rng.integers(period))) % period == 0] = int(rng.choice(colors))
        quadrant[(row - col + int(rng.integers(period))) % period == 0] = int(rng.choice(colors))

    if style == "sparse":
        strokes = int(rng.integers(2, 7))
    elif style == "mosaic":
        strokes = int(rng.integers(2, 9))
    else:
        strokes = int(rng.integers(4, 12))
    for _ in range(strokes):
        r = int(rng.integers(qh))
        c = int(rng.integers(qw))
        h = int(rng.integers(1, min(6, qh - r) + 1))
        w = int(rng.integers(1, min(6, qw - c) + 1))
        color = int(rng.choice(colors))
        kind = int(rng.integers(5))
        if kind == 0:
            quadrant[r:r + h, c:c + w] = color
        elif kind == 1:
            quadrant[r:r + h, c] = color
        elif kind == 2:
            quadrant[r, c:c + w] = color
        elif kind == 3:
            quadrant[r, c:c + w] = color
            quadrant[r + h - 1, c:c + w] = color
            quadrant[r:r + h, c] = color
            quadrant[r:r + h, c + w - 1] = color
        else:
            quadrant[r, c] = color

    # Each chosen design color is observable outside a recoverable erasure.
    spots = rng.choice(qh * qw, size=len(colors), replace=False)
    quadrant.flat[spots] = colors
    return _mirror(quadrant, height, width)


def _needs(mask):
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    need_top_bottom = any(mask[r, width - 1 - c] for r, c in zip(rows, cols))
    need_left_right = any(mask[height - 1 - r, c] for r, c in zip(rows, cols))
    return bool(need_top_bottom), bool(need_left_right)


def _valid_positions(height, width, crop_h, crop_w):
    choices = []
    for r0 in range(height - crop_h + 1):
        for c0 in range(width - crop_w + 1):
            mask = np.zeros((height, width), dtype=bool)
            mask[r0:r0 + crop_h, c0:c0 + crop_w] = True
            rows, cols = np.nonzero(mask)
            if all(
                not all(mask[rr, cc] for rr in (r, height - 1 - r)
                        for cc in (c, width - 1 - c))
                for r, c in zip(rows, cols)
            ):
                choices.append((r0, c0, *_needs(mask)))
    return choices


def _choose_position(rng, height, width, crop_h, crop_w, category):
    choices = _valid_positions(height, width, crop_h, crop_w)
    pools = []
    for r, c, need_tb, need_lr in choices:
        edge_y = r == 0 or r + crop_h == height
        edge_x = c == 0 or c + crop_w == width
        near = (abs(2 * c + crop_w - width) <= crop_w + 2 or
                abs(2 * r + crop_h - height) <= crop_h + 2)
        matched = {
            "vertical": need_tb and not need_lr,
            "horizontal": need_lr and not need_tb,
            "ordinary": not need_tb and not need_lr and not edge_x and not edge_y,
            "left_edge": c == 0,
            "right_edge": c + crop_w == width,
            "top_edge": r == 0,
            "bottom_edge": r + crop_h == height,
            "corner": edge_x and edge_y,
            "near": near and not need_tb and not need_lr,
            "random": True,
        }[category]
        if matched:
            pools.append((r, c))
    if not pools and category in ("ordinary", "near"):
        pools = [(r, c) for r, c, need_tb, need_lr in choices
                 if not need_tb and not need_lr]
    if not pools:
        raise AssertionError(f"no recoverable position for {category}")
    return pools[int(rng.integers(len(pools)))]


def _dimensions(rng, classic=False):
    if classic:
        return 16, 16, 3, 3
    choice = int(rng.integers(4))
    if choice == 0:
        height = width = 16
    elif choice == 1:
        height = width = int(rng.integers(11, 27))
    else:
        height = int(rng.integers(11, 27))
        width = int(rng.integers(11, 27))
        if height == width:
            width = 11 if width != 11 else 12
    if rng.random() < 0.65:
        crop_h = crop_w = int(rng.integers(2, 6))
    else:
        crop_h = int(rng.integers(2, 6))
        crop_w = int(rng.integers(2, 6))
        if crop_h == crop_w:
            crop_w = 2 if crop_w != 2 else 3
    return height, width, crop_h, crop_w


def _infer_patch(puzzle_input):
    """Find the alien rectangle and recover it using only the visible input."""
    height, width = puzzle_input.shape
    candidates = []
    for marker in np.unique(puzzle_input):
        marker = int(marker)
        rows, cols = np.nonzero(puzzle_input == marker)
        r0, c0 = int(rows.min()), int(cols.min())
        crop_h = int(rows.max()) - r0 + 1
        crop_w = int(cols.max()) - c0 + 1
        if not (2 <= crop_h <= 5 and 2 <= crop_w <= 5):
            continue
        if len(rows) != crop_h * crop_w:
            continue
        if not np.all(puzzle_input[r0:r0 + crop_h, c0:c0 + crop_w] == marker):
            continue
        restored = puzzle_input.copy()
        possible = True
        for r in range(r0, r0 + crop_h):
            for c in range(c0, c0 + crop_w):
                copies = {
                    int(puzzle_input[rr, cc])
                    for rr in (r, height - 1 - r)
                    for cc in (c, width - 1 - c)
                    if puzzle_input[rr, cc] != marker
                }
                if len(copies) != 1:
                    possible = False
                    break
                restored[r, c] = copies.pop()
            if not possible:
                break
        if possible and np.array_equal(restored, restored[::-1, :]) and np.array_equal(restored, restored[:, ::-1]):
            candidates.append(((r0, c0), restored[r0:r0 + crop_h, c0:c0 + crop_w].copy(), marker, restored))
    assert len(candidates) == 1, f"expected one recoverable alien rectangle; got {len(candidates)}"
    return candidates[0]


def _palette(rng, marker, shared_palette, forbidden=None, required=None):
    if shared_palette is not None:
        return shared_palette
    available = np.array([c for c in range(10) if c != marker and c != forbidden],
                         dtype=np.int8)
    count = int(rng.integers(2, len(available) + 1))
    if required is not None:
        rest = rng.choice(available[available != required], size=count - 1,
                          replace=False)
        return np.concatenate((np.array([required], dtype=np.int8), rest))
    if marker != 0 and 0 in available and rng.random() < 0.4:
        rest = rng.choice(available[available != 0], size=count - 1,
                          replace=False)
        return np.concatenate((np.array([0], dtype=np.int8), rest))
    return rng.choice(available, size=count, replace=False)


def _make_pair(rng, dimensions, category, marker, colors, colorful=False):
    height, width, crop_h, crop_w = dimensions
    r0, c0 = _choose_position(rng, height, width, crop_h, crop_w, category)
    styles = ("mosaic", "bands", "blocks", "boxes", "sparse", "diagonals")
    style = styles[int(rng.integers(len(styles)))]
    rich = len(colors) >= 7 and crop_h * crop_w >= 16 and rng.random() < 0.45
    for attempt in range(40):
        design_style = "mosaic" if rich or attempt >= 25 else style
        completed = _paint_design(rng, height, width, colors, design_style)
        output = completed[r0:r0 + crop_h, c0:c0 + crop_w].copy()
        if len(np.unique(output)) < (7 if rich else 2 if colorful else 1):
            continue
        puzzle_input = completed.copy()
        puzzle_input[r0:r0 + crop_h, c0:c0 + crop_w] = marker
        try:
            _, recovered, inferred_marker, _ = _infer_patch(puzzle_input)
        except AssertionError:
            continue
        if inferred_marker == marker and np.array_equal(output, recovered):
            return {"input": puzzle_input, "output": output}
    raise AssertionError("bounded construction could not make a unique recoverable crop")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    total = train_count + test_count

    # These are puzzle conventions. A varying marker is taught by two distinct
    # training markers; the patch shape, rather than its numeric color, defines it.
    stable_colors = bool(rng.integers(2))
    stable_marker = bool(rng.integers(2))
    classic = rng.random() < 0.3
    shared_palette = None
    if stable_colors:
        size = int(rng.integers(3, 8))
        shared_palette = rng.choice(np.arange(10, dtype=np.int8), size=size, replace=False)
        marker_pool = [c for c in range(10) if c not in shared_palette]
    else:
        marker_pool = list(range(10))
    if stable_marker:
        marker = marker_pool[int(rng.integers(len(marker_pool)))]
        markers = [marker] * total
    else:
        markers = [int(rng.choice(marker_pool)) for _ in range(total)]
        choices = [c for c in marker_pool if c != markers[0]]
        markers[1] = choices[int(rng.integers(len(choices)))]
    novel_test_color = None
    if not stable_colors and rng.random() < 0.4:
        candidates = [c for c in range(10) if c != markers[train_count]]
        novel_test_color = candidates[int(rng.integers(len(candidates)))]

    if classic:
        train_dims = [_dimensions(rng, classic=True) for _ in range(train_count)]
        test_dims = [_dimensions(rng, classic=True) for _ in range(test_count)]
    else:
        train_dims = [(16, 16, 3, 3)]
        second = _dimensions(rng)
        if second[2] == second[3]:
            second = (second[0], second[1], second[2],
                      2 if second[2] != 2 else 3)
        train_dims.append(second)
        train_dims.extend(_dimensions(rng) for _ in range(train_count - 2))
        rng.shuffle(train_dims)
        test_dims = [_dimensions(rng) for _ in range(test_count)]

    source_like = classic and rng.random() < 0.45
    train_categories = ["horizontal", "ordinary" if source_like else "vertical"]
    category_options = ("ordinary", "vertical", "horizontal", "left_edge",
                        "right_edge", "top_edge", "bottom_edge", "corner", "near")
    train_categories.extend(
        category_options[int(rng.integers(len(category_options)))]
        for _ in range(train_count - 2)
    )
    rng.shuffle(train_categories)
    colorful_index = int(rng.integers(train_count))

    train = []
    used_inputs = set()
    for index, (dimensions, category) in enumerate(zip(train_dims, train_categories)):
        marker = markers[index]
        colors = _palette(rng, marker, shared_palette,
                          forbidden=novel_test_color)
        for _ in range(8):
            pair = _make_pair(rng, dimensions, category, marker, colors,
                              colorful=index == colorful_index)
            if _grid_key(pair["input"]) not in used_inputs:
                break
        else:
            raise AssertionError("could not make a distinct training input")
        used_inputs.add(_grid_key(pair["input"]))
        train.append(pair)

    trained_vertical = any(_needs(pair["input"] == _infer_patch(pair["input"])[2])[0] for pair in train)
    test_options = ("ordinary", "horizontal", "left_edge", "right_edge",
                    "top_edge", "bottom_edge", "corner", "near")
    if trained_vertical:
        test_options += ("vertical", "vertical")
    test = []
    for index, dimensions in enumerate(test_dims):
        category = test_options[int(rng.integers(len(test_options)))]
        marker = markers[train_count + index]
        colors = _palette(rng, marker, shared_palette,
                          required=novel_test_color if index == 0 else None)
        colorful = rng.random() < 0.8
        for _ in range(8):
            pair = _make_pair(rng, dimensions, category, marker, colors,
                              colorful=colorful)
            if _grid_key(pair["input"]) not in used_inputs:
                break
        else:
            raise AssertionError("could not make a distinct test input")
        used_inputs.add(_grid_key(pair["input"]))
        test.append(pair)

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= size <= 30 for size in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    source_keys = {
        ((16, 16), tuple(x for row in image for x in row),
         (3, 3), tuple(x for row in crop for x in row))
        for image, crop in _SOURCE_PAIRS
    }
    seen = set()
    train_needs = []
    test_needs = []
    train_markers = []
    test_markers = []
    train_outputs = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            puzzle_input, output = pair["input"], pair["output"]
            _check_grid(puzzle_input)
            _check_grid(output)
            input_key = _grid_key(puzzle_input)
            output_key = _grid_key(output)
            assert input_key not in seen
            seen.add(input_key)
            assert (input_key[0], input_key[1], output_key[0], output_key[1]) not in source_keys
            (r0, c0), recovered, marker, completed = _infer_patch(puzzle_input)
            assert output.shape == recovered.shape
            assert np.array_equal(output, recovered)
            assert np.array_equal(completed[r0:r0 + output.shape[0],
                                            c0:c0 + output.shape[1]], output)
            assert marker not in completed
            mask = puzzle_input == marker
            needs = _needs(mask)
            if split == "train":
                train_needs.append(needs)
                train_markers.append(marker)
                train_outputs.append(output)
            else:
                test_needs.append(needs)
                test_markers.append(marker)
    assert any(need_lr for _, need_lr in train_needs)
    assert any(len(np.unique(out)) >= 2 for out in train_outputs)
    if any(need_tb for need_tb, _ in test_needs):
        assert any(need_tb for need_tb, _ in train_needs)
    if any(need_lr for _, need_lr in test_needs):
        assert any(need_lr for _, need_lr in train_needs)
    if len(set(train_markers)) == 1:
        assert all(marker == train_markers[0] for marker in test_markers)
    if any(pair["output"].shape[0] != pair["output"].shape[1]
           for pair in puzzle["test"]):
        assert any(pair["output"].shape[0] != pair["output"].shape[1]
                   for pair in puzzle["train"])


def _same_puzzle(left, right):
    return all(
        len(left[split]) == len(right[split]) and all(
            np.array_equal(a[role], b[role])
            for a, b in zip(left[split], right[split])
            for role in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    input_keys, shape_keys, output_shapes = set(), set(), set()
    markers, output_color_counts = set(), set()
    train_vertical = test_vertical = edge_tests = symmetric_outputs = 0
    varying_palette_puzzles = varying_marker_puzzles = mixed_shape_puzzles = 0
    varying_position_puzzles = new_later_color_puzzles = new_test_color_puzzles = 0
    stable_palette_vary_marker = vary_palette_stable_marker = 0
    classic_pairs = aspect_pairs = rectangular_crops = zero_design_pairs = 0
    nonzero_markers = zero_markers = test_left_edge = test_right_edge = test_corner = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_palettes, pair_markers, pair_shapes, pair_positions = set(), set(), set(), set()
        palette_order = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                (top, left), predicted, marker, completed = _infer_patch(inp)
                assert np.array_equal(predicted, out)
                assert np.array_equal(completed, completed[::-1, :])
                assert np.array_equal(completed, completed[:, ::-1])
                palette = tuple(sorted(int(c) for c in np.unique(completed)))
                pair_palettes.add(palette)
                palette_order.append(set(palette))
                pair_markers.add(marker)
                pair_shapes.add((inp.shape, out.shape))
                pair_positions.add((top, left))
                input_keys.add(_grid_key(inp))
                shape_keys.add(inp.shape)
                output_shapes.add(out.shape)
                markers.add(marker)
                output_color_counts.add(len(np.unique(out)))
                classic_pairs += inp.shape == (16, 16) and out.shape == (3, 3)
                aspect_pairs += inp.shape[0] != inp.shape[1]
                rectangular_crops += out.shape[0] != out.shape[1]
                zero_design_pairs += 0 in palette
                nonzero_markers += marker != 0
                zero_markers += marker == 0
                rows, cols = np.nonzero(inp == marker)
                assert len(rows) == out.size
                for row, col in zip(rows, cols):
                    copies = {int(inp[rr, cc])
                              for rr in (int(row), inp.shape[0] - 1 - int(row))
                              for cc in (int(col), inp.shape[1] - 1 - int(col))
                              if inp[rr, cc] != marker}
                    assert len(copies) == 1
                    assert int(out[int(row) - top, int(col) - left]) == copies.pop()
                need_tb, _ = _needs(inp == marker)
                if split == "train" and need_tb:
                    train_vertical += 1
                if split == "test":
                    test_vertical += int(need_tb)
                    edge_tests += int(np.any((inp == marker)[0]) or np.any((inp == marker)[-1]) or
                                      np.any((inp == marker)[:, 0]) or np.any((inp == marker)[:, -1]))
                    test_left_edge += left == 0
                    test_right_edge += left + out.shape[1] == inp.shape[1]
                    test_corner += (top in (0, inp.shape[0] - out.shape[0]) and
                                    left in (0, inp.shape[1] - out.shape[1]))
                symmetric_outputs += int(np.array_equal(out, out[::-1, :]) or
                                         np.array_equal(out, out[:, ::-1]))
        varying_palette_puzzles += len(pair_palettes) > 1
        varying_marker_puzzles += len(pair_markers) > 1
        mixed_shape_puzzles += len(pair_shapes) > 1
        varying_position_puzzles += len(pair_positions) > 1
        new_later_color_puzzles += any(palette - palette_order[0]
                                       for palette in palette_order[1:])
        train_colors = set().union(*palette_order[:len(puzzle["train"])])
        new_test_color_puzzles += any(palette - train_colors
                                      for palette in palette_order[len(puzzle["train"]):])
        stable_palette_vary_marker += len(pair_palettes) == 1 and len(pair_markers) > 1
        vary_palette_stable_marker += len(pair_palettes) > 1 and len(pair_markers) == 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(input_keys) >= 200 and len(shape_keys) >= 20 and len(output_shapes) >= 8
    assert len(markers) >= 8 and min(output_color_counts) == 1 and max(output_color_counts) >= 7
    assert train_vertical >= 20 and test_vertical >= 5 and edge_tests >= 10
    assert symmetric_outputs >= 20
    assert varying_palette_puzzles >= 8 and varying_marker_puzzles >= 8
    assert mixed_shape_puzzles >= 15
    assert varying_position_puzzles == 42
    assert new_later_color_puzzles >= 5 and new_test_color_puzzles >= 2
    assert stable_palette_vary_marker >= 3 and vary_palette_stable_marker >= 3
    assert classic_pairs >= 50 and aspect_pairs >= 20 and rectangular_crops >= 20
    assert zero_design_pairs >= 30 and nonzero_markers >= 30 and zero_markers >= 5
    assert test_left_edge >= 5 and test_right_edge >= 5 and test_corner >= 5
    for i, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        p = generate_puzzle(1000 + i, train_count, test_count)
        validate_generated_puzzle(p)
        assert len(p["train"]) == train_count and len(p["test"]) == test_count
        assert _same_puzzle(p, generate_puzzle(1000 + i, train_count, test_count))
    for i, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        validate_generated_puzzle(generate_puzzle(2000 + i, *counts))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(3000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
    broken = generate_puzzle(4000, 3, 2)
    bad = {split: [{role: p[role].copy() for role in ("input", "output")}
                   for p in broken[split]] for split in ("train", "test")}
    inp = bad["test"][0]["input"]
    marker = _infer_patch(inp)[2]
    spot = np.argwhere(inp != marker)[0]
    r, c = int(spot[0]), int(spot[1])
    inp[r, c] = (int(inp[r, c]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a broken reflection")
