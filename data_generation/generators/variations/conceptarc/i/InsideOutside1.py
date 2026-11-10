"""Synthetic instances of counting markers outside hollow frames."""
import numpy as np

_SOURCE_ROWS = (
    ("0000000000", "0000000000", "0022222201", "0020000200",
     "0020010200", "0020000200", "0020000200", "0022222200",
     "1000000000", "0000010000"),
    ("0000000000", "0007777000", "0007607000", "0007067000",
     "0007007000", "0607607000", "0007007000", "0007777000",
     "0000000000", "0000060000"),
    ("0000000000", "0033333300", "0030000300", "0030900300",
     "9030000309", "0030000300", "0030000300", "0033333300",
     "0000000000", "0000090000"),
    ("0000000050", "0333333305", "0300000300", "0300000300",
     "0300050305", "0300000300", "0305000300", "0300000305",
     "0300000300", "0333333300"),
    ("00000300", "11111000", "10001003", "10301000",
     "10001000", "11111030", "00000000", "00011110",
     "00010013", "30010010", "00010010", "00011110"),
    ("000005", "044440", "040040", "045040", "044440", "000000"),
)
_SOURCE_KEYS = {(len(rows), len(rows[0]), "".join(rows)) for rows in _SOURCE_ROWS}


def _key(grid):
    return grid.shape[0], grid.shape[1], "".join(str(int(x)) for x in grid.flat)


def _rectangles(mask):
    remaining = set(zip(*np.where(mask)))
    if not remaining:
        return None
    result = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            row, col = stack.pop()
            for point in ((row - 1, col), (row + 1, col),
                          (row, col - 1), (row, col + 1)):
                if point in remaining:
                    remaining.remove(point)
                    component.add(point)
                    stack.append(point)
        r0, r1 = min(p[0] for p in component), max(p[0] for p in component) + 1
        c0, c1 = min(p[1] for p in component), max(p[1] for p in component) + 1
        if r1 - r0 < 4 or c1 - c0 < 4:
            return None
        border = {(r, c) for r in range(r0, r1) for c in range(c0, c1)
                  if r in (r0, r1 - 1) or c in (c0, c1 - 1)}
        if component != border:
            return None
        result.append((r0, c0, r1, c1))
    return tuple(sorted(result))


def _inspect(grid):
    """Infer color roles and count from the input pixels."""
    assert np.any(grid == 0)
    colors = [int(x) for x in np.unique(grid) if x != 0]
    assert len(colors) == 2
    candidates = [(color, _rectangles(grid == color)) for color in colors]
    candidates = [(color, boxes) for color, boxes in candidates if boxes]
    assert len(candidates) == 1
    frame_color, boxes = candidates[0]
    marker_color = next(color for color in colors if color != frame_color)
    markers = np.argwhere(grid == marker_color)
    inside = sum(any(r0 < int(r) < r1 - 1 and c0 < int(c) < c1 - 1
                     for r0, c0, r1, c1 in boxes) for r, c in markers)
    outside = len(markers) - inside
    assert outside >= 1
    return frame_color, marker_color, boxes, inside, outside


def _int(rng, low, high):
    return int(rng.integers(low, high + 1))


def _witness_geometry(rng):
    h, w = _int(rng, 4, 8), _int(rng, 4, 8)
    r0, c0 = _int(rng, 1, 9 - h), _int(rng, 1, 9 - w)
    return 10, 10, ((r0, c0, r0 + h, c0 + w),)


def _single_geometry(rng, regime):
    if regime == "source_train":
        height, width = ((10, 10) if rng.random() < 0.35
                         else (_int(rng, 8, 16), _int(rng, 8, 16)))
    else:
        draw = rng.random()
        height, width = ((6, 6) if draw < 0.20 else
                         (12, 8) if draw < 0.36 else
                         (_int(rng, 7, 22), _int(rng, 7, 22)))
    fh, fw = _int(rng, 4, height - 1), _int(rng, 4, width - 1)
    placement = rng.random()
    if placement < 0.27 and height - fh >= 2 and width - fw >= 2:
        r0, c0 = _int(rng, 1, height - fh - 1), _int(rng, 1, width - fw - 1)
    elif placement < 0.54:
        side = _int(rng, 0, 3)
        r0 = (0 if side == 0 else height - fh if side == 1
              else _int(rng, 0, height - fh))
        c0 = (0 if side == 2 else width - fw if side == 3
              else _int(rng, 0, width - fw))
    else:
        r0, c0 = _int(rng, 0, height - fh), _int(rng, 0, width - fw)
    return height, width, ((r0, c0, r0 + fh, c0 + fw),)


def _bands(rng, count):
    maximum = 5 if count == 4 else 6
    lengths = [_int(rng, 4, maximum) for _ in range(count)]
    gaps = [_int(rng, 1, 2 if count >= 3 else 3)
            for _ in range(count - 1)]
    before, after = _int(rng, 0, 1), _int(rng, 0, 1)
    starts = []
    cursor = before
    for index, length in enumerate(lengths):
        starts.append(cursor)
        cursor += length
        if index < count - 1:
            cursor += gaps[index]
    return cursor + after, tuple(zip(starts, lengths))


def _multi_geometry(rng, regime, count):
    arrangement = str(rng.choice(("vertical", "horizontal", "diagonal")))
    long_dimension, row_bands = _bands(rng, count)
    if arrangement == "diagonal":
        width, col_bands = _bands(rng, count)
        if rng.random() < 0.5:
            col_bands = col_bands[::-1]
        height = long_dimension
    else:
        short_limit = 17 if regime == "source_train" else 22
        short = _int(rng, 6, short_limit)
        height, width = ((long_dimension, short) if arrangement == "vertical"
                         else (short, long_dimension))
    boxes = []
    for index in range(count):
        if arrangement == "diagonal":
            row_start, row_size = row_bands[index]
            col_start, col_size = col_bands[index]
        elif arrangement == "vertical":
            row_start, row_size = row_bands[index]
            col_start, col_size = 0, width
        else:
            row_start, row_size = 0, height
            col_start, col_size = row_bands[index]
        fh, fw = _int(rng, 4, row_size), _int(rng, 4, col_size)
        r0 = row_start + _int(rng, 0, row_size - fh)
        c0 = col_start + _int(rng, 0, col_size - fw)
        boxes.append((r0, c0, r0 + fh, c0 + fw))
    return height, width, tuple(sorted(boxes))


def _sample_points(rng, pool, count):
    if count == 0:
        return []
    choices = rng.choice(len(pool), size=count, replace=False)
    return [pool[int(index)] for index in np.atleast_1d(choices)]


def _make_pair(rng, geometry, colors, outside_target, inside_target,
               witness_side=None):
    height, width, boxes = geometry
    frame_color, marker_color = colors
    grid = np.zeros((height, width), dtype=np.int8)
    for r0, c0, r1, c1 in boxes:
        grid[r0, c0:c1] = frame_color
        grid[r1 - 1, c0:c1] = frame_color
        grid[r0:r1, c0] = frame_color
        grid[r0:r1, c1 - 1] = frame_color
    inside, outside = [], []
    for row in range(height):
        for col in range(width):
            if grid[row, col] != 0:
                continue
            enclosed = any(r0 < row < r1 - 1 and c0 < col < c1 - 1
                           for r0, c0, r1, c1 in boxes)
            (inside if enclosed else outside).append((row, col))
    assert len(inside) >= inside_target and len(outside) >= outside_target
    if witness_side is not None:
        r0, c0, r1, c1 = boxes[0]
        outside = [point for point in outside if
                   (point[0] < r0 and c0 <= point[1] < c1 if witness_side == 0 else
                    point[0] >= r1 and c0 <= point[1] < c1 if witness_side == 1 else
                    point[1] < c0 and r0 <= point[0] < r1 if witness_side == 2 else
                    point[1] >= c1 and r0 <= point[0] < r1)]
        assert len(outside) >= outside_target
    for row, col in _sample_points(rng, inside, inside_target):
        grid[row, col] = marker_color
    for row, col in _sample_points(rng, outside, outside_target):
        grid[row, col] = marker_color
    measured = _inspect(grid)[4]
    return {"input": grid, "output": np.zeros((measured, 1), dtype=np.int8)}


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError("Invalid pair count")
    legal = [(tr, te) for tr in range(2, 11) for te in range(1, 5)
             if tr > te and (num_train is None or tr == num_train)
             and (num_test is None or te == num_test)]
    if not legal:
        raise ValueError("Train count must exceed test count")
    if num_train is None and num_test is None:
        te = _int(rng, 1, 4)
        return _int(rng, te + 1, min(10, te + 4)), te
    return legal[_int(rng, 0, len(legal) - 1)]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    # Background and counting rule are shared. Visible shape gives each
    # pair's color roles, so the literal palette is free to change.
    witness_colors = tuple(int(x) for x in rng.choice(np.arange(1, 10), 2,
                                                     replace=False))
    witness_side = _int(rng, 0, 3)
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=train_count + test_count, dtype=np.uint64)
    result = {"train": [], "test": []}
    seen_inputs, seen_layouts = set(), set()
    for index, pair_seed in enumerate(pair_seeds):
        pair_rng = np.random.default_rng(int(pair_seed))
        train = index < train_count
        local_index = index if train else index - train_count
        for _ in range(120):
            if train and local_index < 2:
                geometry = _witness_geometry(pair_rng)
                colors = witness_colors
                outside_count, inside_count = 2 + local_index, 2 - local_index
                side = witness_side
            else:
                regime = "source_train" if train else "target_test"
                probabilities = ((0.54, 0.34, 0.12) if train else
                                 (0.43, 0.34, 0.18, 0.05))
                frame_count = int(pair_rng.choice(
                    np.arange(1, len(probabilities) + 1), p=probabilities))
                geometry = (_single_geometry(pair_rng, regime) if frame_count == 1
                            else _multi_geometry(pair_rng, regime, frame_count))
                colors = tuple(int(x) for x in pair_rng.choice(
                    np.arange(1, 10), 2, replace=False))
                if frozenset(colors) == frozenset(witness_colors):
                    colors = (colors[0], next(c for c in range(1, 10)
                                                 if c not in colors))
                outside_count = (2 if train and local_index == 2 else
                                 _int(pair_rng, 1, 8))
                inside_count = (3 if train and local_index == 2 else
                                _int(pair_rng, 0, min(5, frame_count * 4)))
                side = None
            height, width, boxes = geometry
            border_cells = sum(2 * (r1 - r0 + c1 - c0) - 4
                               for r0, c0, r1, c1 in boxes)
            interior_cells = sum((r1 - r0 - 2) * (c1 - c0 - 2)
                                 for r0, c0, r1, c1 in boxes)
            exterior_cells = height * width - border_cells - interior_cells
            if interior_cells < inside_count or exterior_cells < outside_count:
                continue
            pair = _make_pair(pair_rng, geometry, colors, outside_count,
                              inside_count, side)
            key = _key(pair["input"])
            layout = (pair["input"].shape, boxes)
            if key not in seen_inputs and key not in _SOURCE_KEYS and layout not in seen_layouts:
                break
        else:
            raise RuntimeError("Could not construct a distinct valid pair")
        seen_inputs.add(key)
        seen_layouts.add(layout)
        result["train" if train else "test"].append(pair)
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    tr, te = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= tr <= 10 and 1 <= te <= 4 and tr > te
    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    facts, palettes = [], set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            grid, output = pair["input"], pair["output"]
            frame, marker, boxes, inside, outside = _inspect(grid)
            assert 1 <= len(boxes) <= 4
            assert output.shape == (outside, 1) and np.all(output == 0)
            assert not np.array_equal(grid, output)
            key, output_key = _key(grid), _key(output)
            assert key not in seen_inputs and (key, output_key) not in seen_pairs
            assert key not in _SOURCE_KEYS
            seen_inputs.add(key)
            seen_pairs.add((key, output_key))
            layout = (grid.shape, boxes)
            assert layout not in seen_layouts
            seen_layouts.add(layout)
            palettes.add(frozenset((frame, marker)))
            if split == "train":
                facts.append((grid.shape, frame, marker, boxes, inside, outside))
    assert len(palettes) >= 2
    first, second = facts[:2]
    assert first[0] == second[0] == (10, 10)
    assert len(first[3]) == len(second[3]) == 1
    assert first[1:3] == second[1:3]
    assert first[4:] == (2, 2) and second[4:] == (1, 3)
    for fact in (first, second):
        r0, c0, r1, c1 = fact[3][0]
        assert r0 > 0 and c0 > 0 and r1 < 10 and c1 < 10
    witness_sides = []
    for pair, fact in zip(puzzle["train"][:2], (first, second)):
        r0, c0, r1, c1 = fact[3][0]
        marker = fact[2]
        sides = set()
        for row, col in np.argwhere(pair["input"] == marker):
            row, col = int(row), int(col)
            if r0 < row < r1 - 1 and c0 < col < c1 - 1:
                continue
            if row < r0 and c0 <= col < c1:
                sides.add("top")
            elif row >= r1 and c0 <= col < c1:
                sides.add("bottom")
            elif col < c0 and r0 <= row < r1:
                sides.add("left")
            elif col >= c1 and r0 <= row < r1:
                sides.add("right")
            else:
                raise AssertionError("Witness marker is off the selected side")
        assert len(sides) == 1
        witness_sides.append(next(iter(sides)))
    assert witness_sides[0] == witness_sides[1]
    if tr >= 3:
        assert facts[2][4:] == (3, 2)


def _same_puzzle(left, right):
    return all(len(left[split]) == len(right[split]) and
               all(np.array_equal(a[field], b[field])
                   for a, b in zip(left[split], right[split])
                   for field in ("input", "output"))
               for split in ("train", "test"))


def _orientation(boxes):
    if len(boxes) == 1:
        return "single"
    rows = all(a[2] <= b[0] or b[2] <= a[0]
               for index, a in enumerate(boxes) for b in boxes[index + 1:])
    cols = all(a[3] <= b[1] or b[3] <= a[1]
               for index, a in enumerate(boxes) for b in boxes[index + 1:])
    return ("diagonal" if rows and cols else "vertical" if rows else
            "horizontal" if cols else "other")


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, layouts, palettes, heights = set(), set(), set(), set()
    train_frame_counts, test_frame_counts = set(), set()
    train_orientations, test_orientations = set(), set()
    frame_heights, frame_widths = set(), set()
    split_heights = {"train": set(), "test": set()}
    train_edge = test_inset = zero_inside = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signature = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                frame, marker, boxes, inside, outside = _inspect(grid)
                assert output.shape == (outside, 1)
                shapes.add(grid.shape)
                layouts.add((grid.shape, boxes))
                palettes.add(frozenset((frame, marker)))
                heights.add(outside)
                split_heights[split].add(outside)
                for r0, c0, r1, c1 in boxes:
                    frame_heights.add(r1 - r0)
                    frame_widths.add(c1 - c0)
                signature.append((grid.shape, grid.tobytes(), output.shape))
                touches = any(r0 == 0 or c0 == 0 or r1 == grid.shape[0]
                              or c1 == grid.shape[1] for r0, c0, r1, c1 in boxes)
                if split == "train":
                    train_frame_counts.add(len(boxes))
                    train_orientations.add(_orientation(boxes))
                    train_edge += touches
                else:
                    test_frame_counts.add(len(boxes))
                    test_orientations.add(_orientation(boxes))
                    test_inset += not touches
                zero_inside += inside == 0
        signatures.add(tuple(signature))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 30 and len(layouts) >= 180
    assert len(palettes) >= 15 and len(signatures) == 42
    assert set(range(1, 6)).issubset(heights) and max(heights) >= 6
    assert {1, 2, 3}.issubset(train_frame_counts)
    assert {1, 2, 3, 4}.issubset(test_frame_counts)
    assert {"vertical", "horizontal", "diagonal"}.issubset(train_orientations)
    assert {"vertical", "horizontal", "diagonal"}.issubset(test_orientations)
    assert min(frame_heights) == min(frame_widths) == 4
    assert max(frame_heights) >= 10 and max(frame_widths) >= 10
    assert {2, 3}.issubset(split_heights["test"])
    assert {1, 4, 5}.issubset(split_heights["train"])
    assert train_edge >= 10 and test_inset >= 10 and zero_inside >= 10
    for tr, te in ((2, 1), (10, 1), (10, 4),
                   (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(100 + (tr or 0) + (te or 0), tr, te)
        validate_generated_puzzle(puzzle)
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4), (True, 1), (2, False)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError((tr, te))
