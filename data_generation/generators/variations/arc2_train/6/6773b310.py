import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "00080608006",
        "00080008060",
        "06080608000",
        "88888888888",
        "00080008000",
        "00080608000",
        "60080008060",
        "88888888888",
        "00080008600",
        "06080008006",
        "00086008000",
    ),
    (
        "60080008000",
        "00080068006",
        "00080008000",
        "88888888888",
        "60080008000",
        "00080008060",
        "00080068600",
        "88888888888",
        "00080008000",
        "60080008000",
        "06080608006",
    ),
    (
        "00080008006",
        "00680008600",
        "00080608000",
        "88888888888",
        "00080608000",
        "60080068000",
        "00080008060",
        "88888888888",
        "00080008000",
        "00680008600",
        "00080608000",
    ),
    (
        "00080008000",
        "60080608006",
        "00680008060",
        "88888888888",
        "06080068000",
        "00080008000",
        "06080008600",
        "88888888888",
        "00680008000",
        "00080008060",
        "00086008000",
    ),
    (
        "00080008606",
        "06080068000",
        "00080008000",
        "88888888888",
        "00080008060",
        "00680608000",
        "00086008006",
        "88888888888",
        "00680008000",
        "60080008060",
        "00080608000",
    ),
)

_SOURCE_OUTPUT_ROWS = (
    ("011", "000", "001"),
    ("000", "001", "100"),
    ("001", "010", "000"),
    ("101", "100", "000"),
    ("001", "011", "100"),
)

_SOURCE_PAIR_ROW_KEYS = frozenset(
    zip(_SOURCE_INPUT_ROWS, _SOURCE_OUTPUT_ROWS)
)

def _validate_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _optional_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_pair_counts(seed, num_train, num_test):
    train = _optional_count(num_train, "num_train", 2, 10)
    test = _optional_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _array_value_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _row_strings(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _geometry(rng):
    if rng.random() < 0.16:
        return (3, 3, 3), (3, 3, 3)
    rows = int(rng.integers(2, 5))
    cols = int(rng.integers(2, 5))
    if rng.random() < 0.34:
        height = int(rng.integers(3, 7))
        width = int(rng.integers(3, 7))
        return (height,) * rows, (width,) * cols
    return (tuple(int(v) for v in rng.integers(3, 7, size=rows)),
            tuple(int(v) for v in rng.integers(3, 7, size=cols)))


def _palette(rng, indicator):
    colors = [v for v in range(10) if v != indicator]
    return tuple(int(v) for v in rng.choice(colors, size=3, replace=False))


def _active_count(rng, size, regime, early):
    if early:
        return int(rng.integers(2, min(size - 2, 4) + 1))
    mode = int(rng.integers(10))
    if mode == 0:
        return 0
    if mode == 1:
        return size
    if mode == 2:
        return 1
    if mode == 3:
        return size - 1
    if regime == "target_test" and mode == 4:
        return min(size, 4 + int(rng.integers(3)))
    return int(rng.integers(2, size - 1))


def _layout(rng, rows, cols, count):
    layout = np.zeros((rows, cols), dtype=bool)
    if count == 0:
        return layout
    if count == rows * cols:
        layout[:] = True
        return layout
    mode = int(rng.integers(5))
    if mode == 0 and count <= cols:
        row = int(rng.integers(rows))
        layout[row, rng.choice(cols, size=count, replace=False)] = True
    elif mode == 1 and count <= rows:
        col = int(rng.integers(cols))
        layout[rng.choice(rows, size=count, replace=False), col] = True
    elif mode == 2 and rows * cols - count <= cols:
        layout[:] = True
        row = int(rng.integers(rows))
        layout[row, rng.choice(cols, size=rows * cols - count, replace=False)] = False
    else:
        layout.flat[rng.choice(rows * cols, size=count, replace=False)] = True
    return layout


def _bounds(lengths):
    starts, lines = [], []
    cursor = 0
    for i, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length
        if i + 1 < len(lengths):
            lines.append(cursor)
            cursor += 1
    return starts, lines, cursor


def _make_pair(rng, heights, widths, layout, palette, indicator):
    background, divider, marker = palette
    row_starts, row_lines, height = _bounds(heights)
    col_starts, col_lines, width = _bounds(widths)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[row_lines, :] = divider
    grid[:, col_lines] = divider
    for row, top in enumerate(row_starts):
        for col, left in enumerate(col_starts):
            h, w = heights[row], widths[col]
            first = int(rng.integers(h * w))
            grid[top + first // w, left + first % w] = marker
            if layout[row, col]:
                candidates = [index for index in range(h * w)
                              if abs(index // w - first // w) + abs(index % w - first % w) > 1]
                second = int(rng.choice(candidates))
                grid[top + second // w, left + second % w] = marker
    output = np.full(layout.shape, background, dtype=np.int8)
    output[layout] = indicator
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_pair_counts(seed, num_train, num_test)
    indicator = int(rng.integers(10))
    fixed_palette = bool(rng.random() < 0.52)
    fixed_geometry = bool(rng.random() < 0.25)
    common_palette = _palette(rng, indicator)
    common_geometry = _geometry(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_layouts, train_geometries, train_palettes = [], [], []
    for regime, amount in (("train", train_count), ("test", test_count)):
        for index in range(amount):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            geometry = common_geometry if fixed_geometry else _geometry(pair_rng)
            if not fixed_geometry and regime == "train" and index == 1:
                for _ in range(8):
                    if geometry != train_geometries[0]:
                        break
                    geometry = _geometry(pair_rng)
            heights, widths = geometry
            rows, cols = len(heights), len(widths)
            palette = common_palette if fixed_palette else _palette(pair_rng, indicator)
            if not fixed_palette and regime == "train" and index == 1:
                for _ in range(8):
                    if palette != train_palettes[0]:
                        break
                    palette = _palette(pair_rng, indicator)
            count = _active_count(pair_rng, rows * cols,
                                  "source_train" if regime == "train" else "target_test",
                                  regime == "train" and index < 2)
            layout = _layout(pair_rng, rows, cols, count)
            if regime == "train" and index >= 2 and pair_rng.random() < 0.20:
                matches = [old for geom, old in zip(train_geometries, train_layouts) if geom == geometry]
                if matches:
                    layout = matches[int(pair_rng.integers(len(matches)))].copy()
            if regime == "test" and pair_rng.random() < 0.16:
                matches = [old for geom, old in zip(train_geometries, train_layouts) if geom == geometry]
                if matches:
                    layout = matches[int(pair_rng.integers(len(matches)))].copy()
            if regime == "train":
                train_layouts.append(layout.copy())
                train_geometries.append(geometry)
                train_palettes.append(palette)
            for _ in range(12):
                pair = _make_pair(pair_rng, heights, widths, layout, palette, indicator)
                key = _array_value_key(pair["input"])
                if key not in seen_inputs:
                    break
            else:
                raise AssertionError("could not construct a unique input")
            seen_inputs.add(key)
            puzzle[regime].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _spans(lines, extent):
    edges = [-1] + list(lines) + [extent]
    return [(edges[i] + 1, edges[i + 1]) for i in range(len(edges) - 1)]


def _analyze_input(grid):
    height, width = grid.shape
    rows = [r for r in range(height) if np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(width) if np.all(grid[:, c] == grid[0, c])]
    assert 1 <= len(rows) <= 3 and 1 <= len(cols) <= 3
    divider = int(grid[rows[0], cols[0]])
    mask = np.zeros(grid.shape, dtype=bool)
    mask[rows, :] = True
    mask[:, cols] = True
    assert np.all(grid[mask] == divider)
    assert np.all(grid[~mask] != divider)
    values, frequencies = np.unique(grid[~mask], return_counts=True)
    assert len(values) == 2
    order = np.argsort(frequencies)
    marker, background = int(values[order[0]]), int(values[order[1]])
    assert frequencies[order[1]] > frequencies[order[0]]
    row_spans, col_spans = _spans(rows, height), _spans(cols, width)
    assert all(3 <= end - start <= 6 for start, end in row_spans + col_spans)
    counts = np.zeros((len(row_spans), len(col_spans)), dtype=np.int8)
    for r, (top, bottom) in enumerate(row_spans):
        for c, (left, right) in enumerate(col_spans):
            block = grid[top:bottom, left:right]
            assert np.all((block == background) | (block == marker))
            positions = np.argwhere(block == marker)
            assert len(positions) in (1, 2)
            if len(positions) == 2:
                assert int(np.abs(positions[0] - positions[1]).sum()) > 1
            counts[r, c] = len(positions)
    return background, divider, marker, counts, row_spans, col_spans


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    indicator = None
    inspected = []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer) and not np.issubdtype(grid.dtype, np.bool_)
                assert all(1 <= side <= 30 for side in grid.shape)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            ikey, okey = _array_value_key(inp), _array_value_key(out)
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            assert (_row_strings(inp), _row_strings(out)) not in _SOURCE_PAIR_ROW_KEYS
            background, divider, marker, counts, row_spans, col_spans = _analyze_input(inp)
            assert out.shape == counts.shape
            assert background != divider and background != marker and divider != marker
            nonbackground = set(int(v) for v in np.unique(out)) - {background}
            assert len(nonbackground) <= 1
            if nonbackground:
                found = nonbackground.pop()
                assert found not in (background, divider, marker)
                if indicator is None:
                    indicator = found
                else:
                    assert indicator == found
            inspected.append((regime, background, divider, marker, counts, out))
    assert indicator is not None
    train_evidence = 0
    for regime, background, divider, marker, counts, out in inspected:
        assert indicator not in (background, divider, marker)
        expected = np.full(counts.shape, background, dtype=np.int8)
        expected[counts == 2] = indicator
        assert np.array_equal(out, expected)
        if regime == "train" and np.any(counts == 1) and np.any(counts == 2):
            train_evidence += 1
    assert train_evidence >= 2


def _puzzle_value_key(puzzle):
    return tuple((_array_value_key(pair["input"]), _array_value_key(pair["output"]))
                 for regime in ("train", "test") for pair in puzzle[regime])


def self_test() -> None:
    trains, tests, shapes, sizes, counts, schemes, layouts = set(), set(), set(), set(), set(), set(), set()
    variable_geometry = variable_palette = repeated_map = originals = 0
    unequal = single_only = double_only = concentrated = novel_test_count = novel_test_shape = combined = 0
    keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_value_key(puzzle) == _puzzle_value_key(generate_puzzle(seed))
        keys.add(_puzzle_value_key(puzzle))
        trains.add(len(puzzle["train"]))
        tests.add(len(puzzle["test"]))
        geometries, palettes, local_maps = set(), set(), {}
        train_counts, train_shapes = set(), set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                bg, div, mark, local_counts, row_spans, col_spans = _analyze_input(inp)
                unequal += len({end - start for start, end in row_spans}) > 1 or len({end - start for start, end in col_spans}) > 1
                geometries.add((inp.shape, out.shape))
                palettes.add((bg, div, mark))
                schemes.add((bg, div, mark))
                shapes.add(out.shape)
                sizes.add(inp.shape)
                active = tuple(int(v) for v in (local_counts == 2).flat)
                map_key = (out.shape, active)
                marker_mask = inp == mark
                if map_key in local_maps and not np.array_equal(marker_mask, local_maps[map_key]):
                    repeated_map += 1
                local_maps[map_key] = marker_mask
                layouts.add(map_key)
                active_count = int(np.count_nonzero(local_counts == 2))
                counts.add(active_count)
                single_only += active_count == 0
                double_only += active_count == local_counts.size
                if active_count:
                    concentrated += (np.max(np.sum(local_counts == 2, axis=0)) == active_count
                                     or np.max(np.sum(local_counts == 2, axis=1)) == active_count)
                if regime == "train":
                    train_counts.add((local_counts.size, active_count))
                    train_shapes.add(out.shape)
                else:
                    novel_test_count += (local_counts.size, active_count) not in train_counts
                    novel_test_shape += out.shape not in train_shapes
                originals += inp.shape == (11, 11) and out.shape == (3, 3)
        variable_geometry += len(geometries) > 1
        variable_palette += len(palettes) > 1
        combined += len(geometries) > 1 and len(palettes) > 1
    assert len(trains) >= 4 and len(tests) >= 3 and len(keys) == 42
    assert len(shapes) >= 6 and len(sizes) >= 25 and len(layouts) >= 100
    assert len(counts) >= 10 and len(schemes) >= 35
    assert variable_geometry >= 15 and variable_palette >= 10 and originals >= 5
    assert repeated_map >= 2
    assert unequal >= 30 and single_only >= 5 and double_only >= 5
    assert concentrated >= 15 and novel_test_count >= 10 and novel_test_shape >= 5
    assert combined >= 5
    for train, test in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2718, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(17, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError((train, test))
