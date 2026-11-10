import numpy as np


_SIZES = (12, 14, 16, 18, 20, 23, 25, 27, 30)
_SOURCE_FIRST_ROWS = {
    "9399287887829939", "9961896666981699", "2276899119986722",
    "9965967777695699", "7781982662891877",
}
_LAYOUTS = ("general", "same_half", "opposite", "axis", "edge", "hard")
_MOTIFS = ("busy", "bands", "nested", "rays", "blocks")


def _orbits(size):
    seen, result = set(), []
    for row in range(size):
        for col in range(size):
            if (row, col) in seen:
                continue
            orbit = tuple(sorted({
                (row, col), (col, size - 1 - row),
                (size - 1 - row, size - 1 - col), (size - 1 - col, row),
                (row, size - 1 - col), (size - 1 - row, col),
                (col, row), (size - 1 - col, size - 1 - row),
            }))
            seen.update(orbit)
            result.append(orbit)
    return result


def _latent(rng, size, palette, orbits, motif):
    count = len(palette)
    period = int(rng.integers(1, 4))
    phase = int(rng.integers(count))
    centers = [
        (int(rng.integers(0, size // 2 + 1)),
         int(rng.integers(0, size // 2 + 1)))
        for _ in range(count)
    ]
    labels = []
    for orbit in orbits:
        row, col = orbit[0]
        row_depth = min(row, size - 1 - row)
        col_depth = min(col, size - 1 - col)
        low, high = sorted((row_depth, col_depth))
        if motif == "busy":
            label = int(rng.integers(count)) if rng.random() < 0.7 else (
                low + high // 2 + phase) % count
        elif motif == "bands":
            label = (low // period + phase) % count
        elif motif == "nested":
            label = ((low + high // 3) // period + phase) % count
        elif motif == "rays":
            label = (int(count * (low + 0.5) / (high + 1)) + phase) % count
        else:
            label = min(range(count), key=lambda index: (
                (low - centers[index][0]) ** 2
                + (high - centers[index][1]) ** 2,
                index,
            ))
        if motif != "busy" and rng.random() < 0.045:
            label = int(rng.integers(count))
        labels.append(label)
    missing = set(range(count)) - set(labels)
    if missing:
        chosen = rng.choice(len(orbits), size=len(missing), replace=False)
        for index, label in zip(chosen, sorted(missing)):
            labels[int(index)] = label
    output = np.empty((size, size), dtype=np.int8)
    for orbit, label in zip(orbits, labels):
        for row, col in orbit:
            output[row, col] = palette[label]
    return output


def _separated(first, second):
    row_a, col_a, height_a, width_a = first
    row_b, col_b, height_b, width_b = second
    return (row_a + height_a < row_b or row_b + height_b < row_a
            or col_a + width_a < col_b or col_b + width_b < col_a)


def _coverage(size, rectangles):
    mask = np.zeros((size, size), dtype=bool)
    for row, col, height, width in rectangles:
        mask[row:row + height, col:col + width] = True
    return mask


def _recoverable(mask, orbits):
    return all(any(not mask[row, col] for row, col in orbit) for orbit in orbits)


def _axial_hidden(mask):
    size = mask.shape[0]
    rows, cols = np.indices(mask.shape)
    off_axes = (rows != size - 1 - rows) & (cols != size - 1 - cols)
    return int(np.count_nonzero(
        mask & np.flipud(mask) & np.fliplr(mask) & off_axes
    ))


def _shape(rng, maximum, kind):
    height = int(rng.integers(2, maximum + 1))
    if kind == "square":
        return height, height
    widths = [width for width in range(2, maximum + 1)
              if kind != "nonsquare" or width != height]
    return height, int(rng.choice(widths))


def _position(rng, size, height, width, constraint=None):
    rows = range(size - height + 1)
    cols = range(size - width + 1)
    if constraint is not None:
        axis, side = constraint
        if axis == "row":
            rows = [row for row in rows if
                    (row + height <= size // 2 if side == 0
                     else row >= (size + 1) // 2)]
        else:
            cols = [col for col in cols if
                    (col + width <= size // 2 if side == 0
                     else col >= (size + 1) // 2)]
    if not rows or not cols:
        return None
    return int(rng.choice(rows)), int(rng.choice(cols)), height, width


def _hard_rectangles(rng, size, orbits):
    for _ in range(200):
        first_height = int(rng.integers(3, 5))
        second_height = int(rng.integers(3, 5))
        first_width = int(rng.integers(3, 6))
        second_width = int(rng.integers(3, 6))
        top = int(rng.integers(0, max(1, size // 2 - first_height)))
        bottom = size - top - second_height + int(rng.integers(-1, 2))
        first_col = size // 2 - first_width // 2 + int(rng.integers(-1, 2))
        second_col = size // 2 - second_width // 2 + int(rng.integers(-1, 2))
        rectangles = ((top, first_col, first_height, first_width),
                      (bottom, second_col, second_height, second_width))
        if not all(0 <= row <= size - height and 0 <= col <= size - width
                   for row, col, height, width in rectangles):
            continue
        if not _separated(*rectangles):
            continue
        mask = _coverage(size, rectangles)
        if _axial_hidden(mask) and _recoverable(mask, orbits):
            return rectangles
    raise AssertionError("No full-orbit layout found")


def _rectangles_for_pair(rng, size, orbits, layout, shapes):
    if layout == "hard":
        return _hard_rectangles(rng, size, orbits)
    maximum = min(7, size // 3 + 1)
    for _ in range(2000):
        if shapes == "two_squares":
            kinds = ("square", "square")
        elif shapes == "mixed":
            kinds = ("square", "nonsquare")
        elif shapes == "two_nonsquares":
            kinds = ("nonsquare", "nonsquare")
        else:
            kinds = ("any", "any")
        dimensions = [_shape(rng, maximum, kind) for kind in kinds]
        axis = "row" if rng.random() < 0.5 else "col"
        side = int(rng.integers(2))
        if layout == "same_half":
            constraints = ((axis, side), (axis, side))
        elif layout == "opposite":
            constraints = ((axis, side), (axis, 1 - side))
        else:
            constraints = (None, None)
        positions = [_position(rng, size, height, width, constraint)
                     for (height, width), constraint in zip(dimensions, constraints)]
        if None in positions:
            continue
        if layout == "axis":
            index = int(rng.integers(2))
            height, width = dimensions[index]
            if axis == "row":
                row = size // 2 - height // 2 + int(rng.integers(-1, 2))
                positions[index] = (row, positions[index][1], height, width)
            else:
                col = size // 2 - width // 2 + int(rng.integers(-1, 2))
                positions[index] = (positions[index][0], col, height, width)
        elif layout == "edge":
            index = int(rng.integers(2))
            row, col, height, width = positions[index]
            edge = int(rng.integers(4))
            positions[index] = ((0 if edge == 0 else size - height if edge == 1 else row),
                                (0 if edge == 2 else size - width if edge == 3 else col),
                                height, width)
        first, second = positions
        if not all(0 <= row <= size - height and 0 <= col <= size - width
                   for row, col, height, width in positions):
            continue
        if not _separated(first, second):
            continue
        mask = _coverage(size, positions)
        if np.count_nonzero(mask) > size * size * 0.3:
            continue
        if not _axial_hidden(mask) and _recoverable(mask, orbits):
            return tuple(positions)
    raise AssertionError("No recoverable rectangle layout found")


def _counts(rng, num_train, num_test):
    for name, count, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if count is not None and (isinstance(count, (bool, np.bool_))
                                  or not isinstance(count, (int, np.integer))
                                  or not low <= count <= high):
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _make_pair(rng, size, mask_color, palette, layout, shapes, motif):
    orbits = _orbits(size)
    rectangles = _rectangles_for_pair(rng, size, orbits, layout, shapes)
    output = _latent(rng, size, palette, orbits, motif)
    input_grid = output.copy()
    mask = _coverage(size, rectangles)
    input_grid[mask] = mask_color
    return {"input": input_grid, "output": output}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    mask_color = int(rng.integers(10))
    available = np.array([color for color in range(10) if color != mask_color])
    base_size = int(rng.choice(_SIZES))
    shared_size = bool(rng.random() < 0.4)
    simple_plan = bool(rng.random() < 0.2)
    shared_size |= simple_plan
    test_choices = _LAYOUTS[:-1] if simple_plan else _LAYOUTS
    test_modes = [str(rng.choice(test_choices)) for _ in range(num_test)]
    second_mode = "hard" if not simple_plan and (
        "hard" in test_modes or rng.random() < 0.5
    ) else str(rng.choice(_LAYOUTS[:-1]))
    train_modes = ["opposite", second_mode] + [
        str(rng.choice(test_choices)) for _ in range(num_train - 2)
    ]
    puzzle = {"train": [], "test": []}
    for regime, modes in (("train", train_modes), ("test", test_modes)):
        for index, mode in enumerate(modes):
            child = np.random.default_rng(int(rng.integers(0, 2**63)))
            size = base_size if shared_size or (regime == "train" and index == 0) else int(child.choice(_SIZES))
            color_count = int(child.integers(3, 10))
            palette = tuple(int(color) for color in child.choice(available, size=color_count, replace=False))
            shapes = str(child.choice(("any", "two_squares", "mixed", "two_nonsquares")))
            motif = str(child.choice(_MOTIFS))
            pair = _make_pair(child, size, mask_color, palette, mode, shapes, motif)
            if size == 16 and "".join(str(int(value)) for value in pair["input"][0]) in _SOURCE_FIRST_ROWS:
                pair = _make_pair(child, size, mask_color, palette, mode, shapes, motif)
            puzzle[regime].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _rectangles(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    rectangles = []
    while unseen:
        start = unseen.pop()
        component, stack = {start}, [start]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        top = min(row for row, _ in component)
        bottom = max(row for row, _ in component) + 1
        left = min(col for _, col in component)
        right = max(col for _, col in component) + 1
        assert len(component) == (bottom - top) * (right - left)
        rectangles.append((top, left, bottom - top, right - left))
    return sorted(rectangles)


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    input_grid, output = pair["input"], pair["output"]
    for grid in (input_grid, output):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    size = output.shape[0]
    assert input_grid.shape == output.shape == (size, size)
    assert np.array_equal(output, np.rot90(output))
    assert np.array_equal(output, np.fliplr(output))
    input_colors = set(map(int, np.unique(input_grid)))
    output_colors = set(map(int, np.unique(output)))
    missing = input_colors - output_colors
    assert len(missing) == 1 and output_colors <= input_colors
    mask_color = missing.pop()
    mask = input_grid == mask_color
    assert np.array_equal(input_grid[~mask], output[~mask])
    assert np.all(input_grid[mask] != output[mask])
    rectangles = _rectangles(mask)
    assert len(rectangles) == 2 and _separated(*rectangles)
    for row, col, height, width in rectangles:
        assert height >= 2 and width >= 2
        assert row + height <= size and col + width <= size
    for orbit in _orbits(size):
        visible = {int(input_grid[row, col]) for row, col in orbit if not mask[row, col]}
        assert visible == {int(output[orbit[0]])}
    if size % 2:
        assert not mask[size // 2, size // 2]
    if size == 16:
        assert "".join(str(int(value)) for value in input_grid[0]) not in _SOURCE_FIRST_ROWS
    return mask_color, mask, rectangles, output_colors


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)
    masks, seen_inputs, seen_pairs = set(), set(), set()
    hard_train, hard_test = False, False
    for regime, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            mask_color, mask, _, _ = _inspect_pair(pair)
            masks.add(mask_color)
            input_key = (pair["input"].shape, pair["input"].tobytes())
            pair_key = (input_key, pair["output"].tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            if regime == "train":
                hard_train |= _axial_hidden(mask) > 0
            else:
                hard_test |= _axial_hidden(mask) > 0
    assert len(masks) == 1
    assert not hard_test or hard_train
    for pair in test:
        assert np.array_equal(_predict(train, pair["input"]), pair["output"])


def _same_puzzle(first, second):
    return all(len(first[regime]) == len(second[regime]) and all(
        np.array_equal(pair_a["input"], pair_b["input"])
        and np.array_equal(pair_a["output"], pair_b["output"])
        for pair_a, pair_b in zip(first[regime], second[regime]))
        for regime in ("train", "test"))


def _reconstruct(input_grid, mask_color):
    result = input_grid.copy()
    for orbit in _orbits(input_grid.shape[0]):
        visible = {int(input_grid[row, col]) for row, col in orbit
                   if int(input_grid[row, col]) != mask_color}
        assert len(visible) == 1
        for row, col in orbit:
            result[row, col] = next(iter(visible))
    return result


def _predict(train_pairs, test_input):
    markers = []
    for pair in train_pairs:
        changed = pair["input"] != pair["output"]
        colors = set(map(int, np.unique(pair["input"][changed])))
        assert len(colors) == 1
        markers.append(colors.pop())
    assert len(set(markers)) == 1
    return _reconstruct(test_input, markers[0])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts, sizes, colors = set(), set(), set(), set()
    layouts, palettes, fingerprints = set(), set(), set()
    shapes, sectors, hard_splits, smooth = set(), set(), set(), []
    dimensions, areas, test_areas, placements, combinations = (
        set(), set(), set(), set(), set()
    )
    within_size, within_palette, within_area, within_layout = 0, 0, 0, 0
    simple_puzzles, hard_test_pairs, boundary_pairs, pair_count = 0, 0, 0, 0
    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        for pair in puzzle["test"]:
            train_copy = [{"input": item["input"].copy(),
                           "output": item["output"].copy()}
                          for item in puzzle["train"]]
            assert np.array_equal(
                _predict(train_copy, pair["input"].copy()), pair["output"]
            )
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_sizes, puzzle_palettes, puzzle_areas, puzzle_layouts = (
            set(), set(), set(), set()
        )
        puzzle_hard = False
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                mask_color, mask, rectangles, output_colors = _inspect_pair(pair)
                assert np.array_equal(_reconstruct(pair["input"], mask_color), pair["output"])
                size = pair["input"].shape[0]
                sizes.add(size)
                colors.add(mask_color)
                palettes.add(tuple(sorted(output_colors)))
                puzzle_sizes.add(size)
                puzzle_palettes.add(tuple(sorted(output_colors)))
                layouts.add((size, tuple(rectangles)))
                pair_count += 1
                rectangle_areas = [height * width for _, _, height, width in rectangles]
                area = sum(rectangle_areas)
                areas.add(area)
                puzzle_areas.add(area)
                if regime == "test":
                    test_areas.add(area)
                for _, _, height, width in rectangles:
                    dimensions.add((height, width))
                shapes.add(tuple(sorted("square" if height == width else "rect"
                                        for _, _, height, width in rectangles)))
                sectors.add(tuple((row < size // 2, col < size // 2)
                                  for row, col, _, _ in rectangles))
                hard = _axial_hidden(mask) > 0
                puzzle_hard |= hard
                hard_splits.add((regime, hard))
                hard_test_pairs += regime == "test" and hard
                top = all(row + height <= size // 2
                          for row, _, height, _ in rectangles)
                bottom = all(row >= (size + 1) // 2
                             for row, _, _, _ in rectangles)
                left = all(col + width <= size // 2
                           for _, col, _, width in rectangles)
                right = all(col >= (size + 1) // 2
                            for _, col, _, _ in rectangles)
                same_half = top or bottom or left or right
                first, second = rectangles
                opposite = (
                    (first[0] + first[2] <= size // 2
                     and second[0] >= (size + 1) // 2)
                    or (second[0] + second[2] <= size // 2
                        and first[0] >= (size + 1) // 2)
                    or (first[1] + first[3] <= size // 2
                        and second[1] >= (size + 1) // 2)
                    or (second[1] + second[3] <= size // 2
                        and first[1] >= (size + 1) // 2)
                )
                edge = any(row == 0 or col == 0 or row + height == size
                           or col + width == size
                           for row, col, height, width in rectangles)
                axis = any(
                    (row <= (size - 1) // 2 < row + height
                     and row <= size // 2 < row + height)
                    or (col <= (size - 1) // 2 < col + width
                        and col <= size // 2 < col + width)
                    for row, col, height, width in rectangles
                )
                placements.update((('same_half', same_half),
                                   ('opposite', opposite), ('edge', edge),
                                   ('axis', axis)))
                puzzle_layouts.add((hard, same_half, edge, axis))
                combinations.add((regime, hard, size % 2, sum(
                    height == width for _, _, height, width in rectangles
                )))
                output = pair["output"]
                boundary_pairs += all(
                    len(np.unique(output[row:row + height,
                                         col:col + width])) >= 2
                    for row, col, height, width in rectangles
                )
                smooth.append((np.mean(output[:, 1:] == output[:, :-1])
                               + np.mean(output[1:, :] == output[:-1, :])) / 2)
                fingerprints.add((pair["input"].shape, pair["input"].tobytes()))
        within_size += len(puzzle_sizes) > 1
        within_palette += len(puzzle_palettes) > 1
        within_area += len(puzzle_areas) > 1
        within_layout += len(puzzle_layouts) > 1
        simple_puzzles += not puzzle_hard
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(sizes) >= 7 and any(size % 2 for size in sizes) and max(sizes) == 30
    assert len(colors) >= 7 and len(palettes) >= 25
    assert len(layouts) >= 100 and len(fingerprints) >= 150
    assert len(shapes) == 3 and len(sectors) >= 8
    assert ("train", True) in hard_splits and ("test", True) in hard_splits
    assert within_size >= 8 and within_palette >= 30
    assert within_area >= 30 and within_layout >= 30
    assert min(height for height, _ in dimensions) == 2
    assert max(height for height, _ in dimensions) == 7
    assert min(width for _, width in dimensions) == 2
    assert max(width for _, width in dimensions) == 7
    assert len(areas) >= 40 and len(test_areas) >= 20
    assert all((name, value) in placements for name in
               ("same_half", "opposite", "edge", "axis") for value in (False, True))
    assert simple_puzzles >= 5 and hard_test_pairs >= 5
    assert boundary_pairs >= pair_count // 2
    assert ("test", True, 1, 2) in combinations
    assert min(smooth) < .3 and max(smooth) > .45
    for train_count, test_count in ((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2027, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        assert _same_puzzle(puzzle, generate_puzzle(2027, train_count, test_count))
    for train_count, test_count in ((1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (4, 4), (2, 3)):
        try:
            generate_puzzle(99, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts were accepted")
