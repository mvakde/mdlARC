"""Generate exact, untranslated-orientation copies of a visible ARC tile."""

import numpy as np


_SOURCE_PAIR_STRINGS = (
    (
        "0000005555" "0666605555" "0886805555" "0688800000" "0000000000"
        "0000555500" "0000555500" "0000555500" "0000000000" "0000000000",
        "0000006666" "0666608868" "0886806888" "0688800000" "0000000000"
        "0000666600" "0000886800" "0000688800" "0000000000" "0000000000",
    ),
    (
        "0000000000" "0222000000" "0244000000" "0444000000" "0000005550"
        "0000005550" "0000005550" "0055500000" "0055500000" "0055500000",
        "0000000000" "0222000000" "0244000000" "0444000000" "0000002220"
        "0000002440" "0000004440" "0022200000" "0024400000" "0044400000",
    ),
    (
        "0444000000" "0144005550" "0141005550" "0111005550" "0000005550"
        "0055500000" "0055500555" "0055500555" "0055500555" "0000000555",
        "0444000000" "0144004440" "0141001440" "0111001410" "0000001110"
        "0044400000" "0014400444" "0014100141" "0011100141" "0000000111",
    ),
)

_DIMENSIONS = tuple(
    (height, width)
    for height in range(2, 6)
    for width in range(2, 6)
    if height * width >= 6
)


def _validate_supplied_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _validate_supplied_count(num_train, "num_train", 2, 10)
    num_test = _validate_supplied_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        seed_index = int(seed)
        num_test = 1 + seed_index % 4
        low = max(2, num_test + 1)
        num_train = low + (seed_index // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _is_asymmetric(pattern):
    transforms = [np.flipud(pattern), np.fliplr(pattern), np.rot90(pattern, 2)]
    if pattern.shape[0] == pattern.shape[1]:
        transforms += [np.rot90(pattern), np.rot90(pattern, 3), pattern.T,
                       np.fliplr(pattern.T)]
    return all(not np.array_equal(pattern, moved) for moved in transforms)


def _make_pattern(rng, height, width, colors, force_asymmetric=False):
    """Compose several coherent motifs and irregular ones from the same two colors."""
    rows, cols = np.indices((height, width))
    for _ in range(80):
        mode = int(rng.integers(0, 7))
        if mode == 0:  # independent pixels with a varying color balance
            probability = float(rng.uniform(0.22, 0.78))
            bits = (rng.random((height, width)) < probability).astype(np.int8)
        elif mode == 1:  # diagonal wedge, with either slope
            diagonal = rows + (cols if int(rng.integers(0, 2)) else width - 1 - cols)
            threshold = int(rng.integers(1, height + width - 2))
            bits = (diagonal >= threshold).astype(np.int8)
        elif mode == 2:  # horizontal or vertical band
            length = height if int(rng.integers(0, 2)) else width
            thickness = int(rng.integers(1, length))
            start = int(rng.integers(0, length - thickness + 1))
            axis = rows if length == height and (height != width or int(rng.integers(0, 2))) else cols
            bits = ((axis >= start) & (axis < start + thickness)).astype(np.int8)
        elif mode == 3:  # a corner block, potentially symmetric
            patch_h = int(rng.integers(1, height + 1))
            patch_w = int(rng.integers(1, width + 1))
            top = 0 if int(rng.integers(0, 2)) else height - patch_h
            left = 0 if int(rng.integers(0, 2)) else width - patch_w
            bits = np.zeros((height, width), dtype=np.int8)
            bits[top:top + patch_h, left:left + patch_w] = 1
        elif mode == 4:  # one connected, grown patch
            bits = np.zeros((height, width), dtype=np.int8)
            row, col = int(rng.integers(height)), int(rng.integers(width))
            bits[row, col] = 1
            target = int(rng.integers(2, height * width))
            while int(bits.sum()) < target:
                frontier = []
                for r in range(height):
                    for c in range(width):
                        if bits[r, c]:
                            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                                if 0 <= nr < height and 0 <= nc < width and not bits[nr, nc]:
                                    frontier.append((nr, nc))
                if not frontier:
                    break
                nr, nc = frontier[int(rng.integers(len(frontier)))]
                bits[nr, nc] = 1
        elif mode == 5:  # sparse accents
            bits = np.zeros((height, width), dtype=np.int8)
            count = int(rng.integers(1, min(4, height * width - 1) + 1))
            indices = rng.choice(height * width, size=count, replace=False)
            bits.flat[indices] = 1
        else:  # two opposing triangular fields
            pivot = int(rng.integers(1, height + width - 2))
            bits = ((rows + cols) <= pivot).astype(np.int8)
            if int(rng.integers(0, 2)):
                bits = np.fliplr(bits)
        ones = int(bits.sum())
        if not 1 <= ones < height * width:
            continue
        if int(rng.integers(0, 2)):
            bits = 1 - bits
        if force_asymmetric and not _is_asymmetric(bits):
            continue
        return np.where(bits == 0, colors[0], colors[1]).astype(np.int8)
    bits = np.zeros((height, width), dtype=np.int8)
    bits[0, 0] = 1
    return np.where(bits == 0, colors[0], colors[1]).astype(np.int8)


def _separated(first, second, height, width):
    ar, ac = first
    br, bc = second
    return (ar + height < br or br + height < ar or
            ac + width < bc or bc + width < ac)


def _pad_to_background_majority(rng, height, width, positions, tile_area):
    occupied = len(positions) * tile_area
    while height * width <= 2 * occupied:
        if height <= width and height < 30:
            height += 1
        elif width < 30:
            width += 1
        elif height < 30:
            height += 1
        else:
            raise ValueError("cannot fit a dominant background")
    return height, width, positions


def _line_layout(rng, height, width, count, vertical):
    length = height if vertical else width
    gap = int(rng.integers(1, 4))
    gap = min(gap, max(1, (30 - count * length) // max(1, count - 1)))
    offsets = [0]
    for _ in range(count - 1):
        offsets.append(offsets[-1] + length + gap)
    # Shift individual objects along the minor axis, without using slots.
    minor = width if vertical else height
    variation = int(rng.integers(0, 4))
    across = [int(rng.integers(0, variation + 1)) for _ in range(count)]
    leading = int(rng.integers(0, 3))
    cross_leading = int(rng.integers(0, 3))
    positions = [(offset + leading, offset_other + cross_leading)
                 if vertical else (offset_other + cross_leading, offset + leading)
                 for offset, offset_other in zip(offsets, across)]
    canvas_height = max(r + height for r, _ in positions) + int(rng.integers(0, 3))
    canvas_width = max(c + width for _, c in positions) + int(rng.integers(0, 3))
    if canvas_height > 30 or canvas_width > 30:
        # A full row/column of five large tiles fits only with tight margins.
        positions = [(r - (leading if vertical else cross_leading),
                      c - (cross_leading if vertical else leading)) for r, c in positions]
        canvas_height = max(r + height for r, _ in positions)
        canvas_width = max(c + width for _, c in positions)
    return _pad_to_background_majority(rng, canvas_height, canvas_width,
                                       positions, height * width)


def _compact_layout(rng, height, width, count):
    rows = 2 if count <= 4 else 3
    cols = (count + rows - 1) // rows
    if int(rng.integers(0, 2)):
        rows, cols = cols, rows
    gap_r = int(rng.integers(1, 4))
    gap_c = int(rng.integers(1, 4))
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    order = rng.permutation(len(slots))[:count]
    top = int(rng.integers(0, 3))
    left = int(rng.integers(0, 3))
    positions = []
    for index in order:
        r, c = slots[int(index)]
        positions.append((top + r * (height + gap_r + 2) + int(rng.integers(0, 3)),
                          left + c * (width + gap_c + 2) + int(rng.integers(0, 3))))
    canvas_height = max(r + height for r, _ in positions) + int(rng.integers(0, 3))
    canvas_width = max(c + width for _, c in positions) + int(rng.integers(0, 3))
    if canvas_height > 30 or canvas_width > 30:
        return _free_layout(rng, height, width, count)
    return _pad_to_background_majority(rng, canvas_height, canvas_width,
                                       positions, height * width)


def _free_layout(rng, height, width, count):
    rows = int(np.ceil(np.sqrt(count)))
    cols = (count + rows - 1) // rows
    if int(rng.integers(0, 2)):
        rows, cols = cols, rows
    canvas_height = min(30, rows * (height + 2) + int(rng.integers(1, 8)))
    canvas_width = min(30, cols * (width + 2) + int(rng.integers(1, 8)))
    canvas_height, canvas_width, _ = _pad_to_background_majority(
        rng, canvas_height, canvas_width, [], 0)
    candidates = [(r, c) for r in range(canvas_height - height + 1)
                  for c in range(canvas_width - width + 1)]
    for _ in range(70):
        order = rng.permutation(len(candidates))
        positions = []
        for index in order:
            candidate = candidates[int(index)]
            if all(_separated(candidate, other, height, width) for other in positions):
                positions.append(candidate)
                if len(positions) == count:
                    return _pad_to_background_majority(
                        rng, canvas_height, canvas_width, positions, height * width)
    # A long row always fits the supported five-by-five maximum and avoids
    # a recursive fallback between compact and free layouts.
    return _line_layout(rng, height, width, count, vertical=False)


def _crowded_layout(rng):
    # Four 4x3 objects occupy 48 of 100 cells, as in the source test.
    upper = int(rng.integers(0, 2))
    left_a = int(rng.integers(0, 3))
    left_b = int(rng.integers(0, 3))
    right_a = int(rng.integers(left_a + 4, 8))
    right_b = int(rng.integers(left_b + 4, 8))
    positions = [(upper, left_a), (upper, right_a),
                 (upper + 5, left_b), (upper + 5, right_b)]
    return 10, 10, positions


def _sample_layout(rng, height, width, count, regime, crowded=False):
    if crowded:
        return _crowded_layout(rng)
    style = int(rng.integers(0, 10))
    if style < 2:
        return _line_layout(rng, height, width, count, vertical=False)
    if style < 4:
        return _line_layout(rng, height, width, count, vertical=True)
    if style < 6:
        return _compact_layout(rng, height, width, count)
    return _free_layout(rng, height, width, count)


def _choose_dimensions(rng, index, num_train, mode, base_dimension, factor):
    if mode == "original":
        if index == 0:
            return 3, 4
        if index == 1:
            return 3, 3
        if index == num_train:
            return 4, 3
    if mode == "quiet":
        if index < num_train:
            return base_dimension
        if factor == "height":
            options = [d for d in _DIMENSIONS if d[1] == base_dimension[1] and d[0] != base_dimension[0]]
        elif factor == "width":
            options = [d for d in _DIMENSIONS if d[0] == base_dimension[0] and d[1] != base_dimension[1]]
        else:
            return base_dimension
        return options[int(rng.integers(len(options)))] if options else base_dimension
    if index == 0:
        return base_dimension
    choices = [d for d in _DIMENSIONS if d != base_dimension] if index == 1 else _DIMENSIONS
    return choices[int(rng.integers(len(choices)))]


def _choose_blank_count(rng, index, num_train, mode, factor):
    if mode == "original":
        if index < 2:
            return 2
        if index == num_train:
            return 3
    if mode == "quiet":
        if index < num_train or factor != "count":
            return 2
        return int(rng.choice([1, 3, 4]))
    if index == 0:
        return 2
    if index == 1:
        return int(rng.choice([1, 3]))
    return int(rng.integers(1, 5))


def _sample_roles(rng, palette, shared_colors, varying, index):
    if not varying or index == 0:
        return int(palette[0]), int(palette[1])
    if index == 1:
        return int(palette[4]), int(palette[5])
    allowed = [int(color) for color in palette if int(color) not in shared_colors]
    selected = rng.choice(allowed, size=2, replace=False)
    return int(selected[0]), int(selected[1])


def _sample_content_colors(rng, background, marker, shared_colors, use_shared):
    if use_shared or rng.random() < 0.22:
        colors = list(shared_colors)
    else:
        allowed = [color for color in range(10) if color not in (background, marker)]
        colors = [int(color) for color in rng.choice(allowed, size=2, replace=False)]
    if int(rng.integers(0, 2)):
        colors.reverse()
    return tuple(colors)


def _generate_pair(rng, regime, dimensions, blank_count, background, marker,
                   content_colors, asymmetric=False, crowded=False):
    height, width = dimensions
    canvas_height, canvas_width, positions = _sample_layout(
        rng, height, width, blank_count + 1, regime, crowded=crowded)
    pattern = _make_pattern(rng, height, width, content_colors,
                            force_asymmetric=asymmetric)
    source_index = int(rng.integers(len(positions)))
    input_grid = np.full((canvas_height, canvas_width), background, dtype=np.int8)
    for row, col in positions:
        input_grid[row:row + height, col:col + width] = marker
    row, col = positions[source_index]
    input_grid[row:row + height, col:col + width] = pattern
    output_grid = input_grid.copy()
    for index, (row, col) in enumerate(positions):
        if index != source_index:
            output_grid[row:row + height, col:col + width] = pattern
    return input_grid, output_grid


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    palette = rng.permutation(10)
    shared_colors = (int(palette[2]), int(palette[3]))
    varying_roles = int(seed) % 7 in (2, 4)
    mode_number = int(seed) % 11
    mode = "original" if mode_number == 0 else "quiet" if mode_number in (1, 2) else "mixed"
    base_dimension = _DIMENSIONS[int(rng.integers(len(_DIMENSIONS)))]
    factor = ("height", "width", "count")[int(rng.integers(0, 3))]
    recur_palette = int(seed) % 3 != 0
    train, test = [], []
    training_patterns = set()
    maximum_seed = np.iinfo(np.int64).max
    for index in range(num_train + num_test):
        pair_rng = np.random.default_rng(int(rng.integers(0, maximum_seed)))
        regime = "source_train" if index < num_train else "target_test"
        dimensions = _choose_dimensions(pair_rng, index, num_train, mode,
                                        base_dimension, factor)
        blank_count = _choose_blank_count(pair_rng, index, num_train, mode, factor)
        background, marker = _sample_roles(pair_rng, palette, shared_colors,
                                            varying_roles, index)
        use_shared = index == 0 or (recur_palette and index == num_train)
        content_colors = _sample_content_colors(pair_rng, background, marker,
                                                 shared_colors, use_shared)
        crowded = (regime == "target_test" and dimensions == (4, 3)
                   and blank_count == 3 and (mode == "original" or pair_rng.random() < 0.4))
        for attempt in range(12):
            input_grid, output_grid = _generate_pair(
                pair_rng, regime, dimensions, blank_count, background, marker,
                content_colors, asymmetric=index == 0, crowded=crowded)
            if index < num_train:
                analysis = _analyze_pair(input_grid, output_grid, regime)
                tile = analysis["template"]
                first = int(tile.flat[0])
                key = (tile.shape, (tile != first).astype(np.int8).tobytes())
                if index == 1 and key in training_patterns:
                    continue
                training_patterns.add(key)
            break
        else:
            # A different shape guarantees a different demonstrated motif
            # if twelve same-shape draws happen to repeat the first one.
            dimensions = next(shape for shape in _DIMENSIONS if shape != dimensions)
            input_grid, output_grid = _generate_pair(
                pair_rng, regime, dimensions, blank_count, background, marker,
                content_colors, asymmetric=False, crowded=False)
            tile = _analyze_pair(input_grid, output_grid, regime)["template"]
            training_patterns.add(
                (tile.shape, (tile != int(tile.flat[0])).astype(np.int8).tobytes()))
        pair = {"input": input_grid, "output": output_grid}
        (train if regime == "source_train" else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _connected_components(mask):
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    for start_row in range(height):
        for start_col in range(width):
            if not mask[start_row, start_col] or visited[start_row, start_col]:
                continue
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            coordinates = []
            while stack:
                row, col = stack.pop()
                coordinates.append((row, col))
                for nr, nc in ((row - 1, col), (row + 1, col),
                               (row, col - 1), (row, col + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coordinates)
    return components


def _analyze_pair(input_grid, output_grid, regime):
    assert regime in ("source_train", "target_test")
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 4
    background = int(values[int(np.argmax(counts))])
    assert int(max(counts)) > input_grid.size // 2
    rectangles = []
    for coordinates in _connected_components(input_grid != background):
        rows = [r for r, _ in coordinates]
        cols = [c for _, c in coordinates]
        top, left = min(rows), min(cols)
        bottom, right = max(rows) + 1, max(cols) + 1
        view = input_grid[top:bottom, left:right].copy()
        assert view.size == len(coordinates)
        assert np.all(view != background)
        colors = tuple(int(color) for color in np.unique(view))
        rectangles.append((top, left, view, colors))
    sources = [rect for rect in rectangles if len(rect[3]) == 2]
    blanks = [rect for rect in rectangles if len(rect[3]) == 1]
    assert len(sources) == 1 and 1 <= len(blanks) <= 4
    assert len(rectangles) == len(blanks) + 1
    top, left, template, content_colors = sources[0]
    height, width = template.shape
    assert (height, width) in _DIMENSIONS
    marker_colors = {rect[3][0] for rect in blanks}
    assert len(marker_colors) == 1
    marker = next(iter(marker_colors))
    assert len({background, marker, *content_colors}) == 4
    expected = input_grid.copy()
    for row, col, view, colors in blanks:
        assert view.shape == template.shape
        assert colors == (marker,)
        expected[row:row + height, col:col + width] = template
    assert np.array_equal(output_grid, expected)
    assert set(int(v) for v in np.unique(output_grid)) == {background, *content_colors}
    positions = [(rect[0], rect[1]) for rect in rectangles]
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert _separated(positions[i], positions[j], height, width)
    return {
        "background": background,
        "marker": marker,
        "content_colors": frozenset(content_colors),
        "template": template,
        "template_position": (top, left),
        "placeholder_positions": tuple((r, c) for r, c, _, _ in blanks),
        "dimensions": (height, width),
    }


def _canonical_grid_key(grid):
    canonical = grid.astype(np.int8, copy=False)
    return canonical.shape, canonical.tobytes()


def _is_source_pair(input_grid, output_grid):
    if input_grid.shape != (10, 10) or output_grid.shape != (10, 10):
        return False
    key = ("".join(str(int(x)) for x in input_grid.flat),
           "".join(str(int(x)) for x in output_grid.flat))
    return key in _SOURCE_PAIR_STRINGS


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    input_keys, pair_keys = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        regime = "source_train" if split == "train" else "target_test"
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= edge <= 30 for edge in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert input_grid.shape == output_grid.shape
            assert not np.array_equal(input_grid, output_grid)
            input_key = _canonical_grid_key(input_grid)
            pair_key = (input_key, _canonical_grid_key(output_grid))
            assert input_key not in input_keys and pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            assert not _is_source_pair(input_grid, output_grid)
            analyses[split].append(_analyze_pair(input_grid, output_grid, regime))
    # A varying background or marker is demonstrated by at least two training
    # assignments; the roles remain visible in each individual input.
    train_roles = {(a["background"], a["marker"]) for a in analyses["train"]}
    test_roles = {(a["background"], a["marker"]) for a in analyses["test"]}
    if not test_roles.issubset(train_roles):
        assert len(train_roles) >= 2
    assert any(len(a["placeholder_positions"]) >= 2 for a in analyses["train"])
    assert any(_is_asymmetric(a["template"]) for a in analyses["train"])
    normalized = set()
    for analysis in analyses["train"]:
        tile = analysis["template"]
        normalized.add((tile.shape, (tile != int(tile.flat[0])).astype(np.int8).tobytes()))
    assert len(normalized) >= 2


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[role], b[role])
        for a, b in zip(first[s], second[s]) for role in ("input", "output"))
        for s in ("train", "test"))


def _expect_value_error(seed, num_train=None, num_test=None):
    try:
        generate_puzzle(seed, num_train=num_train, num_test=num_test)
    except ValueError:
        return
    raise AssertionError("invalid counts were accepted")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(90210, 8, 4), generate_puzzle(90210, 8, 4))
    assert _puzzles_equal(generate_puzzle(12345), generate_puzzle(12345))
    for index, (num_train, num_test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(11000 + index, num_train, num_test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (num_train, num_test)
    for index, (num_train, num_test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(12000 + index, num_train, num_test)
        assert num_train is None or len(puzzle["train"]) == num_train
        assert num_test is None or len(puzzle["test"]) == num_test
    for index, counts in enumerate(((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (4, 4), (2, 4))):
        _expect_value_error(index + 1, *counts)
    dimensions, blanks, shapes, patterns, layouts, palettes = set(), set(), set(), set(), set(), set()
    portrait_train = square_or_wide_test = fewer_test = crowded = repeated_palette = varying_roles = 0
    within_dimension_changes = within_count_changes = symmetric = 0
    original_like = one_factor = combined = 0
    border_contact = one_cell_gap = rows_or_columns = 0
    compact = dispersed = coherent = intricate = 0
    for puzzle in puzzles:
        all_analyses = [_analyze_pair(p["input"], p["output"],
                                       "source_train" if split == "train" else "target_test")
                        for split in ("train", "test") for p in puzzle[split]]
        train = all_analyses[:len(puzzle["train"])]
        test = all_analyses[len(puzzle["train"]):]
        within_dimension_changes += len({a["dimensions"] for a in all_analyses}) > 1
        within_count_changes += len({len(a["placeholder_positions"]) for a in all_analyses}) > 1
        portrait_train += any(a["dimensions"][0] > a["dimensions"][1] for a in train)
        square_or_wide_test += any(a["dimensions"][0] <= a["dimensions"][1] for a in test)
        fewer_test += any(len(a["placeholder_positions"]) <= max(len(t["placeholder_positions"]) for t in train)
                          for a in test)
        varying_roles += len({(a["background"], a["marker"]) for a in train}) > 1
        repeated_palette += any(a["content_colors"] == b["content_colors"] and
                                not np.array_equal(a["template"], b["template"])
                                for a in train for b in test)
        original_like += (train[0]["dimensions"] == (3, 4) and
                          train[1]["dimensions"] == (3, 3) and
                          test[0]["dimensions"] == (4, 3) and
                          len(train[0]["placeholder_positions"]) == 2 and
                          len(train[1]["placeholder_positions"]) == 2 and
                          len(test[0]["placeholder_positions"]) == 3)
        first_train = train[0]
        first_test = test[0]
        first_features = (*first_train["dimensions"],
                          len(first_train["placeholder_positions"]),
                          first_train["background"], first_train["marker"])
        test_features = (*first_test["dimensions"],
                         len(first_test["placeholder_positions"]),
                         first_test["background"], first_test["marker"])
        changes = sum(a != b for a, b in zip(first_features, test_features))
        one_factor += changes == 1
        combined += changes >= 3
        palettes.update((a["background"], a["marker"]) for a in all_analyses)
        for pair, analysis in zip(puzzle["train"] + puzzle["test"], all_analyses):
            dimensions.add(analysis["dimensions"])
            blanks.add(len(analysis["placeholder_positions"]))
            shapes.add(pair["input"].shape)
            symmetric += not _is_asymmetric(analysis["template"])
            crowded += (pair["input"].shape == (10, 10) and
                        len(analysis["placeholder_positions"]) == 3 and
                        analysis["dimensions"] == (4, 3))
            tile = analysis["template"]
            motif_components = sum(
                len(_connected_components(tile == color))
                for color in np.unique(tile)
            )
            coherent += motif_components == 2
            intricate += motif_components >= 4
            patterns.add((tile.shape, (tile != int(tile.flat[0])).astype(np.int8).tobytes()))
            anchor = [(*analysis["template_position"], 1)] + [(*p, 0) for p in analysis["placeholder_positions"]]
            height, width = analysis["dimensions"]
            positions = [(r, c) for r, c, _ in anchor]
            rows_or_columns += len({r for r, _ in positions}) == 1 or len({c for _, c in positions}) == 1
            border_contact += any(r == 0 or c == 0 or r + height == pair["input"].shape[0]
                                  or c + width == pair["input"].shape[1] for r, c in positions)
            fraction = len(positions) * height * width / pair["input"].size
            compact += fraction >= 0.45
            dispersed += fraction < 0.25
            for i, (ar, ac) in enumerate(positions):
                if any(((ar + height + 1 == br or br + height + 1 == ar)
                        and max(ac, bc) < min(ac + width, bc + width))
                       or ((ac + width + 1 == bc or bc + width + 1 == ac)
                           and max(ar, br) < min(ar + height, br + height))
                       for br, bc in positions[i + 1:]):
                    one_cell_gap += 1
                    break
            min_row = min(p[0] for p in anchor)
            min_col = min(p[1] for p in anchor)
            layouts.add((analysis["dimensions"], tuple(sorted((r - min_row, c - min_col, kind)
                                                            for r, c, kind in anchor))))
    assert len(dimensions) >= 10 and blanks == {1, 2, 3, 4}
    assert len(shapes) >= 35 and len(layouts) >= 90 and len(patterns) >= 100
    assert len(palettes) >= 20 and symmetric >= 10
    assert portrait_train >= 10 and square_or_wide_test >= 12
    assert fewer_test >= 20 and crowded >= 1
    assert repeated_palette >= 15 and varying_roles >= 8
    assert within_dimension_changes >= 25 and within_count_changes >= 25
    assert original_like >= 3 and one_factor >= 5 and combined >= 15
    assert border_contact >= 80 and one_cell_gap >= 15
    assert rows_or_columns >= 10 and compact >= 3 and dispersed >= 50
    assert coherent >= 100 and intricate >= 10
