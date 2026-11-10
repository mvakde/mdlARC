import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "00000000000",
        "02222000000",
        "02222000000",
        "02222000000",
        "02222000000",
        "00000000000",
        "00000000000",
        "00111111000",
        "00111111000",
        "00111111000",
        "00000000000",
        "00000000000",
    ),
    (
        "000000000000000",
        "002222200011100",
        "002222200011100",
        "002222200011100",
        "002222200000000",
        "002222200000000",
        "002222200000000",
        "000000000000000",
        "000000003333333",
        "000000003333333",
        "000000003333333",
        "000000003333333",
        "000000003333333",
    ),
    (
        "0000000000000",
        "0000002222000",
        "3333002222000",
        "3333002222000",
        "3333002222000",
        "3333002222000",
        "0000002222000",
        "0000000000000",
        "0011111111000",
        "0011111111000",
        "0011111111000",
        "0011111111000",
    ),
    (
        "0111110000000",
        "0111110011100",
        "0111110011100",
        "0111110011100",
        "0000000000000",
        "0022222200000",
        "0022222200000",
        "0022222203333",
        "0022222203333",
        "0022222203333",
        "0022222203333",
        "0000000003333",
        "0000000003333",
        "0000000003333",
    ),
)


def _row_signature(rows):
    return (
        len(rows),
        len(rows[0]),
        tuple(int(value) for row in rows for value in row),
    )


_SOURCE_INPUT_SIGNATURES = {
    _row_signature(rows) for rows in _SOURCE_INPUT_ROWS
}


def _grid_signature(grid):
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _extract_rectangles(grid, background):
    """Extract maximal four-connected, same-colored rectangular objects."""
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    rectangles = []

    for start_row in range(height):
        for start_col in range(width):
            if visited[start_row, start_col]:
                continue

            color = int(grid[start_row, start_col])
            if color == background:
                continue

            visited[start_row, start_col] = True
            stack = [(start_row, start_col)]
            cells = []

            while stack:
                row, col = stack.pop()
                cells.append((row, col))

                for next_row, next_col in (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                ):
                    if not (
                        0 <= next_row < height
                        and 0 <= next_col < width
                    ):
                        continue
                    if visited[next_row, next_col]:
                        continue
                    if int(grid[next_row, next_col]) != color:
                        continue

                    visited[next_row, next_col] = True
                    stack.append((next_row, next_col))

            row0 = min(row for row, _ in cells)
            row1 = max(row for row, _ in cells)
            col0 = min(col for _, col in cells)
            col1 = max(col for _, col in cells)

            rectangle_height = row1 - row0 + 1
            rectangle_width = col1 - col0 + 1

            assert len(cells) == rectangle_height * rectangle_width
            assert np.all(
                grid[row0 : row1 + 1, col0 : col1 + 1] == color
            )
            assert rectangle_height >= 3
            assert rectangle_width >= 3

            rectangles.append((row0, col0, row1, col1, color))

    rectangles.sort()
    return rectangles


def _make_output(grid, background, marker):
    output = grid.copy()

    for row0, col0, row1, col1, _ in _extract_rectangles(
        grid, background
    ):
        output[row0 + 1 : row1, col0 + 1 : col1] = marker

    return output


def _overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _edge_adjacent(a, b):
    if a[2] + 1 == b[0] or b[2] + 1 == a[0]:
        return max(a[1], b[1]) <= min(a[3], b[3])
    if a[3] + 1 == b[1] or b[3] + 1 == a[1]:
        return max(a[0], b[0]) <= min(a[2], b[2])
    return False


def _corner_background(grid, background):
    corners = (grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1])
    counts = np.bincount(np.asarray(corners, dtype=np.int64), minlength=10)
    return counts[background] >= 2 and counts[background] > max(
        counts[color] for color in range(10) if color != background
    )


def _infer_background(grid):
    """The surrounding color is the unique corner mode, even when not a majority."""
    corners = (grid[0, 0], grid[0, -1], grid[-1, 0], grid[-1, -1])
    counts = np.bincount(np.asarray(corners, dtype=np.int64), minlength=10)
    background = int(np.argmax(counts))
    assert counts[background] >= 2
    assert np.count_nonzero(counts == counts[background]) == 1
    return background


def _sample_dimension(rng, regime):
    # Heights and widths are independent. Both regimes include source scale.
    band = int(rng.choice(3, p=(.42, .36, .22) if regime == "train"
                          else (.44, .34, .22)))
    bounds = ((11, 17), (17, 23), (23, 31))[band]
    return int(rng.integers(*bounds))


def _sample_extent(rng, limit):
    maximum = min(limit, 12)
    if maximum <= 5 or rng.random() < .48:
        return int(rng.integers(3, min(maximum, 5) + 1))
    return int(rng.integers(6, maximum + 1))


def _sample_box(rng, height, width):
    h = _sample_extent(rng, height)
    w = _sample_extent(rng, width)
    def position(span):
        if rng.random() < .38:
            return int(rng.choice((0, span)))
        return int(rng.integers(0, span + 1))
    row = position(height - h)
    col = position(width - w)
    return (row, col, row + h - 1, col + w - 1)


def _related_box(rng, anchor, height, width, relation):
    h = _sample_extent(rng, height)
    w = _sample_extent(rng, width)
    direction = int(rng.integers(4))
    gap = 0 if relation == "edge" else 1 if relation == "gap" else 0
    if relation == "diagonal":
        row = anchor[2] + 1 if direction // 2 == 0 else anchor[0] - h
        col = anchor[3] + 1 if direction % 2 == 0 else anchor[1] - w
    elif direction < 2:
        row = anchor[2] + 1 + gap if direction == 0 else anchor[0] - h - gap
        low = max(0, anchor[1] - w + 1)
        high = min(width - w, anchor[3])
        if high < low:
            return None
        col = int(rng.integers(low, high + 1))
    else:
        col = anchor[3] + 1 + gap if direction == 2 else anchor[1] - w - gap
        low = max(0, anchor[0] - h + 1)
        high = min(height - h, anchor[2])
        if high < low:
            return None
        row = int(rng.integers(low, high + 1))
    if not (0 <= row <= height - h and 0 <= col <= width - w):
        return None
    return (row, col, row + h - 1, col + w - 1)


def _dense_boxes(rng, height, width, count, relation):
    # Random bands span most of the shorter axis while a side channel keeps
    # the background spatially identifiable. Band widths and offsets vary.
    horizontal = bool(rng.integers(2))
    primary, secondary = (height, width) if horizontal else (width, height)
    gaps = [1 if relation == "repeat" else 0 if relation == "edge"
            else int(rng.integers(0, 3))]
    gaps.extend(int(rng.integers(0, 3)) for _ in range(count - 2))
    spare = primary - sum(gaps) - 3 * count
    if spare < 0 or secondary < 5:
        return None
    lengths = [3] * count
    for slot in rng.integers(0, count, size=spare):
        lengths[int(slot)] += 1
    side = int(rng.integers(2))
    boxes = []
    start = 0
    for index, length in enumerate(lengths):
        far_margin = int(rng.integers(0, min(3, secondary - 4) + 1))
        near_margin = int(rng.integers(1, min(3, secondary - far_margin - 3) + 1))
        if side == 0:
            begin = near_margin
            end = secondary - far_margin - 1
        else:
            begin = far_margin
            end = secondary - near_margin - 1
        box = (start, begin, start + length - 1, end)
        boxes.append(box if horizontal else (begin, start, end, start + length - 1))
        if index < count - 1:
            start += length + gaps[index]
    if relation == "edge" and not _edge_adjacent(boxes[0], boxes[1]):
        return None
    return boxes


def _sample_boxes(rng, height, width, count, relation):
    if rng.random() < .22:
        boxes = _dense_boxes(rng, height, width, count, relation)
        if boxes is not None:
            return boxes
    boxes = [_sample_box(rng, height, width)]
    for index in range(1, count):
        found = False
        for _ in range(140):
            chosen_relation = relation if index == 1 else str(
                rng.choice(("free", "edge", "gap", "diagonal"),
                           p=(.48, .17, .25, .10))
            )
            anchor = boxes[0] if index == 1 else boxes[int(rng.integers(index))]
            candidate = (_sample_box(rng, height, width)
                         if chosen_relation in ("free", "repeat")
                         else _related_box(rng, anchor, height, width, chosen_relation))
            if candidate is None or any(_overlap(candidate, box) for box in boxes):
                continue
            if index == 1 and relation == "edge" and not _edge_adjacent(candidate, anchor):
                continue
            if index == 1 and relation == "repeat" and _edge_adjacent(candidate, anchor):
                continue
            boxes.append(candidate)
            found = True
            break
        if not found:
            return None
    return boxes


def _color_boxes(rng, boxes, background, marker, reserved, regime,
                 relation, forced_color=None):
    palette = [color for color in range(10)
               if color not in (background, marker)
               and (regime == "test" or color != reserved)]
    colors = []
    for index, box in enumerate(boxes):
        forbidden = {colors[old] for old in range(index)
                     if _edge_adjacent(box, boxes[old])}
        if index == 0 and forced_color is not None:
            if forced_color in forbidden or forced_color not in palette:
                return None
            color = forced_color
        elif index == 1 and relation == "repeat":
            color = colors[0]
            if color in forbidden:
                return None
        else:
            options = [color for color in palette if color not in forbidden]
            if index == 1 and relation == "edge":
                options = [color for color in options if color != colors[0]]
            if not options:
                return None
            repeat_options = [color for color in colors if color in options]
            color = (int(rng.choice(repeat_options))
                     if repeat_options and rng.random() < .38
                     else int(rng.choice(options)))
        colors.append(color)
    return colors


def _generate_pair(rng, marker, reserved, regime, relation,
                   forced_background=None, forced_color=None):
    for _ in range(450):
        height = _sample_dimension(rng, regime)
        width = _sample_dimension(rng, regime)
        count = int(rng.integers(2, 5 if regime == "train" else 6))
        background = (forced_background if forced_background is not None
                      else int(rng.choice([color for color in range(10)
                                           if color != marker and
                                           not (regime == "test" and
                                                forced_color == reserved and color == reserved)])))
        if background == forced_color:
            continue
        boxes = _sample_boxes(rng, height, width, count, relation)
        if boxes is None:
            continue
        colors = _color_boxes(rng, boxes, background, marker, reserved,
                              regime, relation, forced_color)
        if colors is None:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        for (row0, col0, row1, col1), color in zip(boxes, colors):
            grid[row0:row1 + 1, col0:col1 + 1] = color
        if not _corner_background(grid, background):
            continue
        try:
            rectangles = _extract_rectangles(grid, background)
        except AssertionError:
            continue
        if len(rectangles) != count:
            continue
        if relation == "repeat" and len(set(colors)) == count:
            continue
        output = _make_output(grid, background, marker)
        return {"input": grid, "output": output}
    raise RuntimeError("unable to construct a valid rectangle scene")


def _validate_count(name, value, minimum, maximum):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not minimum <= int(value) <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _resolve_counts(rng, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(test + 1, 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _analyze_pair(input_grid, output_grid):
    background = _infer_background(input_grid)
    rectangles = _extract_rectangles(input_grid, background)
    assert 2 <= len(rectangles) <= 5
    changed = input_grid != output_grid
    assert np.any(changed)
    marker_values = np.unique(output_grid[changed])
    assert len(marker_values) == 1
    marker = int(marker_values[0])
    assert marker != background and not np.any(input_grid == marker)
    assert np.array_equal(output_grid, _make_output(input_grid, background, marker))
    return background, marker, rectangles


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs = set()
    pairs = set()
    markers = set()
    analyses = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert source.shape == target.shape
            signature = _grid_signature(source)
            pair_signature = (signature, _grid_signature(target))
            assert signature not in inputs and pair_signature not in pairs
            assert signature not in _SOURCE_INPUT_SIGNATURES
            inputs.add(signature)
            pairs.add(pair_signature)
            result = _analyze_pair(source, target)
            analyses[regime].append(result)
            markers.add(result[1])
            count = len(result[2])
            assert 2 <= count <= (4 if regime == "train" else 5)
            assert 11 <= source.shape[0] <= 30 and 11 <= source.shape[1] <= 30
    assert len(markers) == 1
    train_rectangles = [rectangle for _, _, rects in analyses["train"]
                        for rectangle in rects]
    train_backgrounds = {background for background, _, _ in analyses["train"]}
    train_colors = {rectangle[4] for rectangle in train_rectangles}
    assert len({rectangle[4] for rectangle in train_rectangles}) >= 2
    assert len(train_backgrounds) >= 2
    assert train_backgrounds & train_colors
    assert any(len({rect[4] for rect in rects}) < len(rects)
               for _, _, rects in analyses["train"])
    assert any(rects[i][4] != rects[j][4] and _edge_adjacent(rects[i], rects[j])
               for _, _, rects in analyses["train"]
               for i in range(len(rects)) for j in range(i + 1, len(rects)))
    # Marker is the only learned convention. Multiple differently colored
    # training interiors demonstrate that it applies regardless of color.
    assert len({int(pair["output"][pair["input"] != pair["output"]][0])
                for pair in puzzle["train"]}) == 1
    assert any(rectangle[4] not in train_colors
               for _, _, rects in analyses["test"] for rectangle in rects)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    palette = [int(color) for color in rng.permutation(10)]
    marker, reserved = palette[:2]
    demonstrations = ("edge", "repeat") if rng.integers(2) else ("repeat", "edge")
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    used = set()
    first_background = None
    first_object = None
    for regime, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(pair_seeds[
                index if regime == "train" else train_count + index])
            if regime == "train" and index < 2:
                relation = demonstrations[index]
            else:
                relation = str(pair_rng.choice(
                    ("free", "edge", "gap", "diagonal"), p=(.35, .22, .32, .11)))
            for _ in range(20):
                pair = _generate_pair(
                    pair_rng, marker, reserved, regime, relation,
                    forced_background=(palette[2] if regime == "train" and index == 0
                                       else first_object if regime == "train" and index == 1
                                       else None),
                    forced_color=(first_background if regime == "train" and index == 1
                                  else reserved if regime == "test" and index == 0 else None),
                )
                signature = _grid_signature(pair["input"])
                if signature not in used and signature not in _SOURCE_INPUT_SIGNATURES:
                    break
            else:
                raise RuntimeError("unable to produce unique pair")
            if regime == "train" and index == 0:
                first_background = _infer_background(pair["input"])
                first_object = _extract_rectangles(pair["input"], first_background)[0][4]
            used.add(signature)
            puzzle[regime].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple((_grid_signature(pair["input"]), _grid_signature(pair["output"]))
                 for regime in ("train", "test") for pair in puzzle[regime])


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    observations = {"small_train": 0, "small_test": 0, "large_train": 0,
                    "large_test": 0, "dense": 0, "test_fewer": 0,
                    "test_same": 0, "new_test_color": 0, "role_flip": 0,
                    "test_separated": 0, "test_adjacent": 0,
                    "thin": 0, "long": 0, "boundary": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = set()
        train_backgrounds = set()
        train_objects = []
        for pair in puzzle["train"]:
            background, _, rects = _analyze_pair(pair["input"], pair["output"])
            train_backgrounds.add(background)
            train_colors.update(rect[4] for rect in rects)
            train_objects.append(len(rects))
        test_objects = []
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                source = pair["input"]
                background, marker, rects = _analyze_pair(source, pair["output"])
                shapes.add((regime, source.shape))
                layouts.add((regime, source.shape, tuple(rect[:4] for rect in rects)))
                palettes.add((background, marker, tuple(sorted({r[4] for r in rects}))))
                small = source.shape[0] <= 16 and source.shape[1] <= 16
                large = source.shape[0] >= 23 or source.shape[1] >= 23
                if small:
                    observations["small_" + regime] += 1
                if large:
                    observations["large_" + regime] += 1
                observations["dense"] += int(np.count_nonzero(source != background) >= source.size / 2)
                observations["thin"] += int(any(r[2] - r[0] == 2 or r[3] - r[1] == 2 for r in rects))
                observations["long"] += int(any(r[2] - r[0] >= 6 or r[3] - r[1] >= 6 for r in rects))
                observations["boundary"] += int(any(r[0] == 0 or r[1] == 0 or r[2] == source.shape[0]-1 or r[3] == source.shape[1]-1 for r in rects))
                if regime == "test":
                    test_objects.append(len(rects))
                    observations["new_test_color"] += int(any(r[4] not in train_colors for r in rects))
                    observations["test_adjacent"] += int(any(_edge_adjacent(rects[i], rects[j]) for i in range(len(rects)) for j in range(i+1, len(rects))))
                    observations["test_separated"] += int(all(not _edge_adjacent(rects[i], rects[j]) for i in range(len(rects)) for j in range(i+1, len(rects))))
        observations["test_fewer"] += int(any(t < max(train_objects) for t in test_objects))
        observations["test_same"] += int(any(t in train_objects for t in test_objects))
        observations["role_flip"] += int(any(color in train_backgrounds for color in train_colors))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 80 and len(layouts) >= 150 and len(palettes) >= 100
    assert all(value > 0 for value in observations.values()), observations
    assert _puzzle_signature(generate_puzzle(314159)) == _puzzle_signature(generate_puzzle(314159))
    assert _puzzle_signature(generate_puzzle(0)) != _puzzle_signature(generate_puzzle(1))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None), (7, 3)):
        puzzle = generate_puzzle(5000 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(123, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
