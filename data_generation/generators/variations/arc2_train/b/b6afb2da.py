import numpy as np


# Exact original inputs, including the solved source test. Their shapes and
# contents, rather than their palette or dimensions alone, are excluded.
_SOURCE_INPUT_ROWS = {
    ("5555550000", "5555550000", "5555550000", "5555550000", "5555550000",
     "0000000000", "0000555555", "0000555555", "0000555555", "0000555555"),
    ("0000000000", "0000000000", "0555500000", "0555500000", "0555500000",
     "0555505555", "0000005555", "0000005555", "0000005555", "0000005555"),
    ("0555500000", "0555500000", "0555500000", "0555500000", "0555500000",
     "0555500000", "0000000000", "0000555555", "0000555555", "0000555555"),
}


def _checked_count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = None if num_train is None else _checked_count("num_train", num_train, 2, 10)
    test = None if num_test is None else _checked_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        sampled_train = int(rng.integers(4, 11))
        sampled_test = int(rng.integers(1, min(4, sampled_train - 1) + 1))
        phase = seed % 8
        if phase in (0, 1):
            train, test = 2, 1
        elif phase == 2:
            train, test = 3, 1 + (seed // 8) % 2
        else:
            train, test = sampled_train, sampled_test
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _key(grid):
    return grid.shape, tuple(int(value) for value in grid.ravel())


def _puzzle_key(puzzle):
    return tuple((section, tuple((_key(p["input"]), _key(p["output"]))
                                 for p in puzzle[section])) for section in ("train", "test"))


def _source_copy(grid):
    if grid.shape != (10, 10):
        return False
    return tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUT_ROWS


def _sample_size(rng):
    if rng.random() < 0.10:
        return (3, 3)
    high = 13 if rng.random() < 0.16 else 11
    return int(rng.integers(3, high)), int(rng.integers(3, high))


def _new_size(rng, familiar):
    for _ in range(32):
        size = _sample_size(rng)
        if size not in familiar:
            return size
    for height in range(3, 13):
        for width in range(3, 13):
            if (height, width) not in familiar:
                return height, width
    raise AssertionError("no unused rectangle dimension")


def _sizes(rng, count, regime, familiar, second_train=False):
    if regime == "target_test":
        # Relative to the demonstrated sizes, tests use zero, one, or two
        # novel dimensions among their first two objects.
        novelty = int(rng.integers(0, min(2, count) + 1))
        flags = [True] * novelty + [False] * (count - novelty)
        rng.shuffle(flags)
        sizes = [(_new_size(rng, familiar) if flag else familiar[int(rng.integers(len(familiar)))])
                 for flag in flags]
        return sizes

    sizes = [_sample_size(rng) for _ in range(count)]
    if count >= 2 and rng.random() < 0.30:
        sizes[int(rng.integers(1, count))] = sizes[0]
    if second_train and set(sizes).issubset(set(familiar)):
        sizes[0] = _new_size(rng, familiar)
    return sizes


def _separated(a, b):
    return (a[1] + 1 < b[0] or b[1] + 1 < a[0]
            or a[3] + 1 < b[2] or b[3] + 1 < a[2])


def _avoids_corners(box, shape):
    top, bottom, left, right = box
    rows, cols = shape
    return not ((top == 0 or bottom == rows - 1)
                and (left == 0 or right == cols - 1))


def _orientation(boxes):
    if len(boxes) != 2:
        return "other"
    a, b = boxes
    row_overlap = not (a[1] < b[0] or b[1] < a[0])
    col_overlap = not (a[3] < b[2] or b[3] < a[2])
    if row_overlap and not col_overlap:
        return "horizontal"
    if col_overlap and not row_overlap:
        return "vertical"
    return "diagonal"


def _place_oriented(rng, sizes, orientation):
    if rng.random() < 0.5:
        sizes = sizes[::-1]
    if orientation == "vertical":
        transposed, boxes = _place_oriented(rng, [(w, h) for h, w in sizes], "horizontal")
        return (transposed[1], transposed[0]), [(l, r, t, b) for t, b, l, r in boxes]

    (h1, w1), (h2, w2) = sizes
    min_width = w1 + w2 + 1
    min_height = max(h1, h2) + 2
    for _ in range(80):
        width = int(rng.integers(min_width, min(30, min_width + 12) + 1))
        height = int(rng.integers(min_height, min(30, min_height + 12) + 1))
        extra = width - min_width
        margins = rng.multinomial(extra, [1 / 3] * 3)
        x1 = int(margins[0])
        x2 = x1 + w1 + 1 + int(margins[1])
        y1 = int(rng.integers(0, height - h1 + 1))
        y2 = int(rng.integers(0, height - h2 + 1))
        boxes = [(y1, y1 + h1 - 1, x1, x1 + w1 - 1),
                 (y2, y2 + h2 - 1, x2, x2 + w2 - 1)]
        if (_orientation(boxes) == "horizontal"
                and all(_avoids_corners(box, (height, width)) for box in boxes)):
            return (height, width), boxes
    width = min_width + 2
    height = max(h1, h2) + 2
    return (height, width), [(1, h1, 1, w1), (1, h2, w1 + 2, w1 + w2 + 1)]


def _place_free(rng, sizes):
    count = len(sizes)
    min_height = max(max(h for h, _ in sizes) + 2, 5 + 3 * count)
    min_width = max(max(w for _, w in sizes) + 2, 5 + 3 * count)
    for canvas_try in range(24):
        upper_h = min(30, min_height + (10 if canvas_try < 8 else 22))
        upper_w = min(30, min_width + (10 if canvas_try < 8 else 22))
        height = int(rng.integers(min_height, upper_h + 1))
        width = int(rng.integers(min_width, upper_w + 1))
        order = list(rng.permutation(count))
        boxes = [None] * count
        for index in order:
            h, w = sizes[index]
            for _ in range(75):
                # Independent border and interior choices yield flush objects,
                # generous padding, aligned positions, and staggered positions.
                y = int(rng.integers(0, height - h + 1))
                x = int(rng.integers(0, width - w + 1))
                if rng.random() < 0.22:
                    y = 0 if rng.random() < 0.5 else height - h
                if rng.random() < 0.22:
                    x = 0 if rng.random() < 0.5 else width - w
                box = (y, y + h - 1, x, x + w - 1)
                if (_avoids_corners(box, (height, width))
                        and all(other is None or _separated(box, other) for other in boxes)):
                    boxes[index] = box
                    break
            if boxes[index] is None:
                break
        if all(box is not None for box in boxes):
            return (height, width), boxes

    # Every requested size is at most 12, so four 14-by-14 slots fit in 30.
    slots = [(1, 1), (1, 16), (16, 1), (16, 16)]
    rng.shuffle(slots)
    boxes = []
    for (h, w), (top, left) in zip(sizes, slots):
        boxes.append((top, top + h - 1, left, left + w - 1))
    return (30, 30), boxes


def _paint(shape, boxes, background, roles):
    corner, edge, fill = roles
    result = np.full(shape, background, dtype=np.int8)
    for top, bottom, left, right in boxes:
        result[top:bottom + 1, left:right + 1] = fill
        result[top, left:right + 1] = edge
        result[bottom, left:right + 1] = edge
        result[top:bottom + 1, left] = edge
        result[top:bottom + 1, right] = edge
        result[top, left] = result[top, right] = corner
        result[bottom, left] = result[bottom, right] = corner
    return result


def _palette(rng):
    colors = [int(v) for v in rng.permutation(10)]
    mode_roll = rng.random()
    mode = "simple" if mode_roll < 0.30 else ("overlap" if mode_roll < 0.68 else "varied")
    roles = tuple(colors[:3])
    clean_bg, clean_src = colors[3:5]
    if mode == "overlap":
        second_bg, second_src = roles[0], roles[1]
    elif mode == "varied":
        second_bg, second_src = colors[5:7]
    else:
        second_bg, second_src = clean_bg, clean_src
    second_alt = next(c for c in colors if c not in (second_bg, second_src))
    return dict(mode=mode, roles=roles, clean_bg=clean_bg, clean_src=clean_src,
                second_bg=second_bg, second_src=second_src, second_alt=second_alt)


def _pair_colors(rng, palette, section, index, count):
    mode = palette["mode"]
    if mode == "simple" or (section == "train" and index == 0):
        bg, src = palette["clean_bg"], palette["clean_src"]
        sources = [src] * count
    elif section == "train" and index == 1:
        bg, src = palette["second_bg"], palette["second_src"]
        sources = [src] * count
        sources[-1] = palette["second_alt"]
    else:
        bg = int(rng.integers(0, 10))
        available = [c for c in range(10) if c != bg]
        src = available[int(rng.integers(len(available)))]
        sources = [src] * count
        if count >= 2 and rng.random() < 0.38:
            alternatives = [c for c in available if c != src]
            sources[int(rng.integers(count))] = alternatives[int(rng.integers(len(alternatives)))]
    return bg, sources


def _make_pair(rng, sizes, colors, roles, orientation):
    bg, sources = colors
    if orientation in ("horizontal", "vertical") and len(sizes) == 2:
        shape, boxes = _place_oriented(rng, sizes, orientation)
    else:
        shape, boxes = _place_free(rng, sizes)
    input_grid = np.full(shape, bg, dtype=np.int8)
    for box, color in zip(boxes, sources):
        top, bottom, left, right = box
        input_grid[top:bottom + 1, left:right + 1] = color
    return {"input": input_grid, "output": _paint(shape, boxes, bg, roles)}


def _components(grid, background):
    foreground = grid != background
    seen = np.zeros(grid.shape, dtype=bool)
    boxes = []
    for row, col in np.argwhere(foreground):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        color = int(grid[row, col])
        stack = [(row, col)]
        seen[row, col] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1]:
                    if foreground[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
        ys, xs = zip(*cells)
        top, bottom, left, right = min(ys), max(ys), min(xs), max(xs)
        assert bottom - top + 1 >= 3 and right - left + 1 >= 3
        assert len(cells) == (bottom - top + 1) * (right - left + 1)
        assert np.all(grid[top:bottom + 1, left:right + 1] == color)
        boxes.append((top, bottom, left, right))
    boxes.sort()
    assert 1 <= len(boxes) <= 4
    for i, first in enumerate(boxes):
        assert _avoids_corners(first, grid.shape)
        for second in boxes[i + 1:]:
            assert _separated(first, second)
    return boxes


def _inspect(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    assert input_grid.shape == output_grid.shape
    h, w = input_grid.shape
    bg = int(input_grid[0, 0])
    assert all(int(input_grid[y, x]) == bg for y, x in
               ((0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    boxes = _components(input_grid, bg)
    assert np.any(input_grid == bg)
    top, _, left, _ = boxes[0]
    roles = (int(output_grid[top, left]), int(output_grid[top, left + 1]),
             int(output_grid[top + 1, left + 1]))
    assert len(set(roles)) == 3
    assert np.array_equal(output_grid, _paint(input_grid.shape, boxes, bg, roles))
    assert not np.array_equal(input_grid, output_grid)
    sizes = [(b - t + 1, r - l + 1) for t, b, l, r in boxes]
    assert all(3 <= height <= 12 and 3 <= width <= 12 for height, width in sizes)
    sources = [int(input_grid[t, l]) for t, _, l, _ in boxes]
    return dict(background=bg, sources=sources, roles=roles, boxes=boxes,
                sizes=sizes, orientation=_orientation(boxes),
                border=any(t == 0 or l == 0 or b == h - 1 or r == w - 1
                           for t, b, l, r in boxes))


def _predict_from_training(train, test_input):
    # This path receives only visible demonstrations and the test input.
    observations = [_inspect(pair)["roles"] for pair in train]
    assert all(roles == observations[0] for roles in observations)
    bg = int(test_input[0, 0])
    boxes = _components(test_input, bg)
    return _paint(test_input.shape, boxes, bg, observations[0])


def _cellwise_expected(input_grid, roles):
    # A separate, cellwise reconstruction for self_test, rather than a call
    # to the slice-based routine that constructs ground truth.
    bg = int(input_grid[0, 0])
    result = np.full(input_grid.shape, bg, dtype=np.int8)
    for top, bottom, left, right in _components(input_grid, bg):
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                if row in (top, bottom) and col in (left, right):
                    role = 0
                elif row in (top, bottom) or col in (left, right):
                    role = 1
                else:
                    role = 2
                result[row, col] = roles[role]
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    palette = _palette(rng)
    orientation_roll = rng.random()
    orientation = ("horizontal" if orientation_roll < 0.18 else
                   "vertical" if orientation_roll < 0.36 else "free")
    max_count = 2 if orientation != "free" else 4
    object_counts = [2]
    if palette["mode"] == "simple":
        object_counts.append(int(rng.integers(1, max_count + 1)))
    else:
        object_counts.append(int(rng.integers(2, max_count + 1)))
    object_counts += [int(rng.integers(1, max_count + 1))
                      for _ in range(train_count + test_count - 2)]
    if len(set(object_counts)) == 1:
        object_counts[train_count] = 1

    pair_entropy = rng.integers(0, 2**32, size=4, dtype=np.uint32)
    streams = np.random.SeedSequence(pair_entropy.tolist()).spawn(train_count + test_count)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    familiar = []
    for global_index, (section, index) in enumerate(
            [("train", i) for i in range(train_count)]
            + [("test", i) for i in range(test_count)]):
        pair_rng = np.random.default_rng(streams[global_index])
        count = object_counts[global_index]
        sizes = _sizes(pair_rng, count, "source_train" if section == "train" else "target_test",
                       familiar, second_train=(section == "train" and index == 1))
        colors = _pair_colors(pair_rng, palette, section, index, count)
        for _ in range(12):
            pair = _make_pair(pair_rng, sizes, colors, palette["roles"], orientation)
            key = _key(pair["input"])
            if key not in used_inputs and not _source_copy(pair["input"]):
                break
        else:
            raise AssertionError("could not construct a unique pair")
        used_inputs.add(key)
        puzzle[section].append(pair)
        if section == "train":
            familiar.extend(sizes)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in inputs and pair_key not in pairs
            assert not _source_copy(pair["input"])
            inputs.add(input_key)
            pairs.add(pair_key)
            analyses[section].append(_inspect(pair))

    roles = analyses["train"][0]["roles"]
    assert all(a["roles"] == roles for section in analyses for a in analyses[section])
    assert len(analyses["train"][0]["boxes"]) == 2
    first = analyses["train"][0]
    assert first["background"] not in roles and all(c not in roles for c in first["sources"])
    train_geometries = {tuple(sorted(a["sizes"])) for a in analyses["train"]}
    assert len(train_geometries) >= 2
    all_counts = [len(a["boxes"]) for section in analyses for a in analyses[section]]
    assert len(set(all_counts)) >= 2
    if any(len(set(a["sources"])) > 1 for a in analyses["test"]):
        assert any(len(set(a["sources"])) > 1 for a in analyses["train"])
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(10_000, 10_042)]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert sum(len(p["train"]) <= 3 for p in puzzles) >= 8
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    grid_shapes, sizes, object_counts, orientations, role_maps = set(), set(), set(), set(), set()
    single = triple = same_size = three_square = large_square = mixed = overlap = 0
    border = internal = 0
    varied_background = varied_source = same_horizontal = same_vertical = mixed_orientation = 0
    two_object_test_novelty = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        role_maps.add(_inspect(puzzle["train"][0])["roles"])
        training = [_inspect(pair) for pair in puzzle["train"]]
        familiar = {size for analysis in training for size in analysis["sizes"]}
        varied_background += len({a["background"] for a in training}) > 1
        varied_source += len({source for a in training for source in a["sources"]}) > 1
        all_analyses = training + [_inspect(pair) for pair in puzzle["test"]]
        pair_orientations = [a["orientation"] for a in all_analyses if a["orientation"] != "other"]
        same_horizontal += len(pair_orientations) >= 2 and set(pair_orientations) == {"horizontal"}
        same_vertical += len(pair_orientations) >= 2 and set(pair_orientations) == {"vertical"}
        mixed_orientation += len(set(pair_orientations)) > 1
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a = _inspect(pair)
                assert np.array_equal(_cellwise_expected(pair["input"], a["roles"]),
                                      pair["output"])
                grid_shapes.add(pair["input"].shape)
                sizes.update(a["sizes"])
                object_counts.add(len(a["boxes"]))
                orientations.add(a["orientation"])
                single += len(a["boxes"]) == 1
                triple += len(a["boxes"]) >= 3
                same_size += len(a["sizes"]) >= 2 and len(set(a["sizes"])) < len(a["sizes"])
                three_square += (3, 3) in a["sizes"]
                large_square += any(h == w and h >= 5 for h, w in a["sizes"])
                mixed += len(set(a["sources"])) > 1
                overlap += a["background"] in a["roles"] or any(c in a["roles"] for c in a["sources"])
                border += a["border"]
                internal += not a["border"]
                if section == "test" and len(a["sizes"]) == 2:
                    two_object_test_novelty.add(sum(size not in familiar for size in a["sizes"]))
    assert len(grid_shapes) >= 35 and len(sizes) >= 45
    assert object_counts == {1, 2, 3, 4}
    assert {"horizontal", "vertical", "diagonal"}.issubset(orientations)
    assert len(role_maps) >= 20
    assert min(single, triple, same_size, three_square, large_square, mixed, overlap,
               border, internal) > 0
    assert varied_background >= 8 and varied_source >= 8
    assert same_horizontal > 0 and same_vertical > 0 and mixed_orientation > 0
    assert two_object_test_novelty == {0, 1, 2}
    assert _puzzle_key(generate_puzzle(431)) == _puzzle_key(generate_puzzle(431))
    assert _puzzle_key(generate_puzzle(432, 7, 3)) == _puzzle_key(generate_puzzle(432, 7, 3))
    for seed, train, test in ((501, 2, 1), (502, 10, 1), (503, 10, 4),
                              (504, None, 1), (505, None, 4),
                              (506, 2, None), (507, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(700, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {arguments}")
