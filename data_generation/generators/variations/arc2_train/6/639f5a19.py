"""Variations of the five-role rectangle decoration task."""

import numpy as np


_SOURCE_FOOTPRINTS = {
    ((1, 5, 10, 12), (14, 1, 8, 12)),
    ((1, 2, 8, 8), (10, 7, 10, 12)),
    ((1, 3, 10, 6), (4, 12, 14, 10), (15, 3, 6, 6)),
}


def _valid_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    low, high = (2, 10) if name == "num_train" else (1, 4)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _valid_count(num_train, "num_train")
    num_test = _valid_count(num_test, "num_test")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _fits(height, width, count, style):
    if height < 6 or width < 6:
        return False
    if style == "vertical":
        return height >= 7 * count - 1
    if style == "horizontal":
        return width >= 7 * count - 1
    return ((height + 1) // 7) * ((width + 1) // 7) >= count


def _canvas_and_area(rng, count, style):
    shapes = [(h, w) for h in range(12, 31) for w in range(12, 31)
              if _fits(h - 1, w - 1, count, style)]
    assert shapes
    squares = [shape for shape in shapes if shape[0] == shape[1]]
    if (23, 23) in shapes and rng.random() < 0.09:
        shape = (23, 23)
    elif squares and rng.random() < 0.23:
        shape = squares[int(rng.integers(len(squares)))]
    else:
        shape = shapes[int(rng.integers(len(shapes)))]
    height, width = shape
    margins = [(top, bottom, left, right)
               for top in range(5) for bottom in range(5)
               for left in range(5) for right in range(5)
               if (top or left) and _fits(height - top - bottom,
                                           width - left - right, count, style)]
    top, bottom, left, right = margins[int(rng.integers(len(margins)))]
    return shape, (top, left, height - top - bottom, width - left - right)


def _partition(rng, area, count, style):
    if count == 1:
        return [area]
    row, col, height, width = area
    options = []
    axes = (style,) if style != "free" else ("vertical", "horizontal")
    for axis in axes:
        length = height if axis == "vertical" else width
        for first_count in range(1, count):
            for gap in range(1, 5):
                for first_length in range(6, length - gap - 5):
                    second_length = length - gap - first_length
                    if axis == "vertical":
                        valid = (_fits(first_length, width, first_count, style)
                                 and _fits(second_length, width, count - first_count, style))
                    else:
                        valid = (_fits(height, first_length, first_count, style)
                                 and _fits(height, second_length, count - first_count, style))
                    if valid:
                        options.append((axis, first_count, gap, first_length))
    assert options
    axis, first_count, gap, first_length = options[int(rng.integers(len(options)))]
    if axis == "vertical":
        first = (row, col, first_length, width)
        second = (row + first_length + gap, col, height - first_length - gap, width)
    else:
        first = (row, col, height, first_length)
        second = (row, col + first_length + gap, height, width - first_length - gap)
    return (_partition(rng, first, first_count, style)
            + _partition(rng, second, count - first_count, style))


def _even_size(rng, extent):
    sizes = list(range(6, extent - extent % 2 + 1, 2))
    return sizes[-1] if rng.random() < 0.25 else sizes[int(rng.integers(len(sizes)))]


def _offset(rng, slack):
    if slack == 0:
        return 0
    if rng.random() < 0.55:
        return 0 if rng.random() < 0.5 else slack
    return int(rng.integers(slack + 1))


def _rectangles(rng, area, count, style):
    rectangles = []
    for row, col, sh, sw in _partition(rng, area, count, style):
        previous = rectangles[int(rng.integers(len(rectangles)))] if rectangles else None
        if sh >= 14 and sw >= 14 and rng.random() < 0.22:
            square_sizes = list(range(14, min(20, sh, sw) + 1, 2))
            height = width = square_sizes[int(rng.integers(len(square_sizes)))]
        elif (previous is not None and previous[2] <= sh and previous[3] <= sw
                and rng.random() < 0.22):
            height, width = previous[2:4]
        else:
            height, width = _even_size(rng, sh), _even_size(rng, sw)
        rectangles.append((row + _offset(rng, sh - height),
                           col + _offset(rng, sw - width), height, width))
    return rectangles


def _decorate(shape, rectangles, background, sources, roles):
    inp = np.full(shape, background, dtype=np.int8)
    out = np.full(shape, background, dtype=np.int8)
    for (row, col, height, width), source in zip(rectangles, sources):
        inp[row:row + height, col:col + width] = source
        rm, cm = row + height // 2, col + width // 2
        out[row:rm, col:cm] = roles[0]
        out[row:rm, cm:col + width] = roles[1]
        out[rm:row + height, col:cm] = roles[2]
        out[rm:row + height, cm:col + width] = roles[3]
        out[row + 2:row + height - 2, col + 2:col + width - 2] = roles[4]
    return {"input": inp, "output": out}


def _object_count(rng, split, index, style):
    if split == "train" and index == 0:
        if style == "free":
            return int(rng.choice((2, 3, 4), p=(0.65, 0.25, 0.10)))
        return int(rng.choice((2, 3), p=(0.8, 0.2)))
    if split == "train" and style != "free":
        return int(rng.choice((1, 2, 3, 4), p=(0.24, 0.50, 0.20, 0.06)))
    weights = ((0.18, 0.30, 0.27, 0.17, 0.08) if split == "train"
               else (0.12, 0.24, 0.31, 0.22, 0.11))
    return int(rng.choice((1, 2, 3, 4, 5), p=weights))


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _assert_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9


def _inspect_input(inp):
    background = int(inp[0, 0])
    assert int(np.count_nonzero(inp == background)) > inp.size // 2
    visited = np.zeros(inp.shape, dtype=bool)
    rectangles = []
    for start_row in range(inp.shape[0]):
        for start_col in range(inp.shape[1]):
            if visited[start_row, start_col] or int(inp[start_row, start_col]) == background:
                continue
            color = int(inp[start_row, start_col])
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            cells = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for nr, nc in ((row - 1, col), (row + 1, col),
                               (row, col - 1), (row, col + 1)):
                    if (0 <= nr < inp.shape[0] and 0 <= nc < inp.shape[1]
                            and not visited[nr, nc] and int(inp[nr, nc]) != background):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            r0 = min(row for row, _ in cells)
            r1 = max(row for row, _ in cells) + 1
            c0 = min(col for _, col in cells)
            c1 = max(col for _, col in cells) + 1
            height, width = r1 - r0, c1 - c0
            assert len(cells) == height * width
            assert np.all(inp[r0:r1, c0:c1] == color)
            assert height >= 6 and width >= 6 and height % 2 == 0 and width % 2 == 0
            rectangles.append((r0, c0, height, width, color))
    assert 1 <= len(rectangles) <= 5
    rectangles.sort()
    for index, a in enumerate(rectangles):
        for b in rectangles[index + 1:]:
            assert (a[0] + a[2] < b[0] or b[0] + b[2] < a[0]
                    or a[1] + a[3] < b[1] or b[1] + b[3] < a[1])
    return background, rectangles


def _roles_from_pair(pair, rectangles):
    row, col, height, width, _ = rectangles[0]
    out = pair["output"]
    return (int(out[row, col]), int(out[row, col + width - 1]),
            int(out[row + height - 1, col]),
            int(out[row + height - 1, col + width - 1]),
            int(out[row + 2, col + 2]))


def _expected_output(inp, background, rectangles, roles):
    out = np.full(inp.shape, background, dtype=np.int8)
    for row, col, height, width, _ in rectangles:
        rm, cm = row + height // 2, col + width // 2
        out[row:rm, col:cm] = roles[0]
        out[row:rm, cm:col + width] = roles[1]
        out[rm:row + height, col:cm] = roles[2]
        out[rm:row + height, cm:col + width] = roles[3]
        out[row + 2:row + height - 2, col + 2:col + width - 2] = roles[4]
    return out


def _is_source_copy(inp, background, rectangles, roles):
    footprints = tuple((r, c, h, w) for r, c, h, w, _ in rectangles)
    return (inp.shape == (23, 23) and background == 0
            and all(color == 8 for *_, color in rectangles)
            and roles == (6, 1, 2, 3, 4)
            and footprints in _SOURCE_FOOTPRINTS)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _assert_grid(inp)
            _assert_grid(out)
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            records[split].append((_inspect_input(inp), pair))
    first_info, first_pair = records["train"][0]
    roles = _roles_from_pair(first_pair, first_info[1])
    assert len(set(roles)) == 5
    assert any(len(info[1]) >= 2 for info, _ in records["train"])
    backgrounds = {"train": set(), "test": set()}
    sources = {"train": set(), "test": set()}
    within_pair_source_demo = False
    for split in ("train", "test"):
        for (background, rectangles), pair in records[split]:
            backgrounds[split].add(background)
            pair_sources = {color for *_, color in rectangles}
            sources[split].update(pair_sources)
            if split == "train" and len(pair_sources) >= 2:
                within_pair_source_demo = True
            assert background not in roles and not (pair_sources & set(roles))
            assert np.array_equal(pair["output"], _expected_output(
                pair["input"], background, rectangles, roles))
            assert not _is_source_copy(pair["input"], background, rectangles, roles)
            assert set(int(v) for v in np.unique(pair["output"])) == {background, *roles}
    if backgrounds["test"] - backgrounds["train"]:
        assert len(backgrounds["train"]) >= 2
    if sources["test"] - sources["train"]:
        assert len(sources["train"]) >= 2 and within_pair_source_demo


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    palette = [int(value) for value in rng.permutation(10)]
    roles = tuple(palette[:5])
    neutrals = palette[5:]
    variable_inputs = bool(rng.random() < 0.55)
    train_style = ("vertical", "horizontal", "free")[int(rng.integers(3))]
    puzzle = {"train": [], "test": []}
    seen = set()
    first_background = None
    first_source = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            style = train_style if split == "train" else "free"
            object_count = _object_count(rng, split, index, style)
            for attempt in range(30):
                shape, area = _canvas_and_area(rng, object_count, style)
                rectangles = _rectangles(rng, area, object_count, style)
                if variable_inputs:
                    background_options = [v for v in neutrals
                                          if split != "train" or index != 1 or v != first_background]
                    background = int(rng.choice(background_options))
                    source_options = [v for v in neutrals if v != background]
                    if split == "train" and index == 1:
                        distinct = [v for v in source_options if v != first_source]
                        if distinct:
                            source_options = distinct
                    sources = [int(rng.choice(source_options)) for _ in rectangles]
                    if split == "train" and index == 0:
                        sources[1] = int(rng.choice([v for v in source_options if v != sources[0]]))
                else:
                    background = neutrals[0]
                    sources = [neutrals[1]] * object_count
                pair = _decorate(shape, rectangles, background, sources, roles)
                key = _grid_key(pair["input"])
                labeled_rects = sorted((*rect, source)
                                       for rect, source in zip(rectangles, sources))
                if (int(np.count_nonzero(pair["input"] == background)) > pair["input"].size // 2
                        and key not in seen and not _is_source_copy(
                        pair["input"], background, labeled_rects, roles)):
                    break
            else:
                raise ValueError("could not construct distinct pairs")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train" and index == 0:
                first_background, first_source = background, sources[0]
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[field], y[field])
        for x, y in zip(a[split], b[split]) for field in ("input", "output")
    ) for split in ("train", "test"))


def _axis(rectangles):
    if len(rectangles) < 2:
        return "single"
    by_row = sorted(rectangles)
    by_col = sorted(rectangles, key=lambda item: item[1])
    vertical = all(a[0] + a[2] < b[0] for a, b in zip(by_row, by_row[1:]))
    horizontal = all(a[1] + a[3] < b[1] for a, b in zip(by_col, by_col[1:]))
    if vertical:
        return "vertical"
    if horizontal:
        return "horizontal"
    return "staggered"


def self_test() -> None:
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(713 + (train or 0) + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for bad in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                {"num_test": 5}, {"num_train": 2, "num_test": 2},
                {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(927, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {bad}")
    assert _puzzles_equal(generate_puzzle(2431, 7, 3), generate_puzzle(2431, 7, 3))
    train_counts, test_counts, structures, palettes = set(), set(), set(), set()
    shapes, object_counts = set(), {"train": set(), "test": set()}
    axes, train_regimes = set(), set()
    variable_background_seeds = variable_source_seeds = multicolor_seeds = 0
    edge_touch = repeated_size = long_thin = large_square = 0
    one_gap = roomy = square_canvas = source_canvas = 0
    horizontal_thin = vertical_thin = 0
    test_lower = test_match = test_higher = 0
    unseen_test_size = repeated_test_layout = combined_variation = 0
    within_count_changes = 0
    three_layouts, tall_sides = set(), set()
    train_six = train_fourteen = test_six = test_fourteen = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        info_by_split = {split: [_inspect_input(pair["input"]) for pair in puzzle[split]]
                         for split in ("train", "test")}
        roles = _roles_from_pair(puzzle["train"][0], info_by_split["train"][0][1])
        palettes.add(roles)
        all_info = [item for split in ("train", "test") for item in info_by_split[split]]
        backgrounds = {background for background, _ in all_info}
        source_colors = {color for _, rects in all_info for *_, color in rects}
        variable_background_seeds += len(backgrounds) > 1
        variable_source_seeds += len(source_colors) > 1
        multicolor_seeds += any(len({r[4] for r in rects}) > 1
                                for _, rects in info_by_split["train"])
        train_object_counts = {len(rects) for _, rects in info_by_split["train"]}
        train_max = max(train_object_counts)
        training_sizes = {(r[2], r[3]) for _, rr in info_by_split["train"] for r in rr}
        training_layouts = {_axis(rr) for _, rr in info_by_split["train"]}
        within_count_changes += len({len(rr) for _, rr in all_info}) > 1
        for _, rects in info_by_split["test"]:
            test_lower += len(rects) < train_max
            test_match += len(rects) in train_object_counts
            test_higher += len(rects) > train_max
            unseen_test_size += any((r[2], r[3]) not in training_sizes for r in rects)
            repeated_test_layout += _axis(rects) in training_layouts
        current_axes = {_axis(rects) for _, rects in info_by_split["train"]} - {"single"}
        if current_axes == {"vertical"}:
            train_regimes.add("all_vertical")
        elif current_axes == {"horizontal"}:
            train_regimes.add("all_horizontal")
        else:
            train_regimes.add("mixed")
        structure = []
        for split in ("train", "test"):
            for (background, rects), pair in zip(info_by_split[split], puzzle[split]):
                inp = pair["input"]
                assert np.array_equal(pair["output"], _expected_output(
                    inp, background, rects, roles))
                shapes.add(inp.shape)
                square_canvas += inp.shape[0] == inp.shape[1]
                source_canvas += inp.shape == (23, 23)
                object_counts[split].add(len(rects))
                axes.add(_axis(rects))
                structure.append((split, inp.shape,
                                  tuple((r, c, h, w) for r, c, h, w, _ in rects)))
                sizes = [(h, w) for _, _, h, w, _ in rects]
                combined_variation += (len(rects) >= 3 and inp.shape[0] == inp.shape[1]
                                       and len({r[4] for r in rects}) >= 2)
                if len(rects) == 3:
                    three_layouts.add(_axis(rects))
                    tall = max(rects, key=lambda rectangle: rectangle[2])
                    other = [rectangle for rectangle in rects if rectangle is not tall]
                    tx = tall[1] + tall[3] / 2
                    ty = tall[0] + tall[2] / 2
                    ox = sum(r[1] + r[3] / 2 for r in other) / 2
                    oy = sum(r[0] + r[2] / 2 for r in other) / 2
                    if abs(tx - ox) >= abs(ty - oy):
                        tall_sides.add("left" if tx < ox else "right")
                    else:
                        tall_sides.add("top" if ty < oy else "bottom")
                repeated_size += len(set(sizes)) < len(sizes)
                long_thin += any((h == 6 and w >= 18) or (w == 6 and h >= 18)
                                 for h, w in sizes)
                horizontal_thin += any(h == 6 and w >= 18 for h, w in sizes)
                vertical_thin += any(w == 6 and h >= 18 for h, w in sizes)
                large_square += any(h == w and h >= 14 for h, w in sizes)
                edge_touch += any(r == 0 or c == 0 or r + h == inp.shape[0]
                                  or c + w == inp.shape[1] for r, c, h, w, _ in rects)
                roomy += any(r >= 3 and c >= 3 and inp.shape[0] - r - h >= 3
                            and inp.shape[1] - c - w >= 3
                            for r, c, h, w, _ in rects)
                for i, a in enumerate(rects):
                    for b in rects[i + 1:]:
                        row_overlap = max(a[0], b[0]) < min(a[0] + a[2], b[0] + b[2])
                        col_overlap = max(a[1], b[1]) < min(a[1] + a[3], b[1] + b[3])
                        row_gap = max(b[0] - a[0] - a[2], a[0] - b[0] - b[2])
                        col_gap = max(b[1] - a[1] - a[3], a[1] - b[1] - b[3])
                        if (row_overlap and col_gap == 1) or (col_overlap and row_gap == 1):
                            one_gap += 1
                if split == "train":
                    train_six += any(6 in size for size in sizes)
                    train_fourteen += any(14 in size for size in sizes)
                else:
                    test_six += any(6 in size for size in sizes)
                    test_fourteen += any(14 in size for size in sizes)
        structures.add(tuple(structure))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(structures) == 42 and len(palettes) >= 30 and len(shapes) >= 50
    assert object_counts["train"] >= {1, 2, 3, 4}
    assert object_counts["test"] >= {1, 2, 3, 4, 5}
    assert {"all_vertical", "all_horizontal", "mixed"} <= train_regimes
    assert {"vertical", "horizontal", "staggered"} <= axes
    assert variable_background_seeds >= 8 and variable_source_seeds >= 8
    assert multicolor_seeds >= 8
    assert min(edge_touch, repeated_size, long_thin, large_square) > 0
    assert min(one_gap, roomy, square_canvas, source_canvas) > 0
    assert min(horizontal_thin, vertical_thin, test_lower, test_match, test_higher) > 0
    assert min(unseen_test_size, repeated_test_layout, combined_variation) > 0
    assert within_count_changes == 42
    assert {"vertical", "horizontal", "staggered"} <= three_layouts
    assert {"left", "right", "top", "bottom"} <= tall_sides
    assert min(train_six, train_fourteen, test_six, test_fourteen) > 0
