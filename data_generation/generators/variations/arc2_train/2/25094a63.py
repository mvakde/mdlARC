"""Dense camouflage fields with conspicuous solid rectangles."""

import hashlib
import numpy as np


_MIN_RECT_SIDE = 4
_MIN_RECT_AREA = 20
_SOURCE_INPUT_DIGESTS = frozenset((
    "f2766a9287100dd85c417801675d4f9c77a06e262538d25dc74caf88f2295394",
    "93023f8fbd3e6703f4b1acfa4d0d5f961dee56e4068452d11d70472e9649a773",
    "19657a3a1e172dc99c59cf2734b08aaf15bfd32abe6e3328918c271646bfd419",
))


def _validate_count(value, name, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng, num_train, num_test):
    _validate_count(num_train, "num_train", 2, 10)
    _validate_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, int(num_test) + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _large_uniform_rectangle_mask(grid):
    """Visible rectangle footprints, including only fully supported cores."""
    height, width = grid.shape
    result = np.zeros(grid.shape, dtype=bool)
    for color in np.unique(grid):
        for top in range(height - _MIN_RECT_SIDE + 1):
            common = np.ones(width, dtype=bool)
            for bottom in range(top, height):
                common &= grid[bottom] == color
                if not np.any(common):
                    break
                rect_height = bottom - top + 1
                if rect_height < _MIN_RECT_SIDE:
                    continue
                needed_width = max(
                    _MIN_RECT_SIDE,
                    (_MIN_RECT_AREA + rect_height - 1) // rect_height,
                )
                padded = np.concatenate(([False], common, [False]))
                changes = np.flatnonzero(padded[1:] != padded[:-1])
                for start, stop in zip(changes[::2], changes[1::2]):
                    if stop - start >= needed_width:
                        result[top:bottom + 1, start:stop] = True
    return result


def _component_boxes(mask):
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            r0 = min(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            rh = max(r for r, _ in cells) - r0 + 1
            rw = max(c for _, c in cells) - c0 + 1
            if len(cells) != rh * rw or not np.all(mask[r0:r0 + rh, c0:c0 + rw]):
                raise AssertionError("changed component is not a filled rectangle")
            boxes.append((r0, c0, rh, rw))
    return sorted(boxes)


def _boxes_have_gap(first, second):
    r, c, h, w = first
    rr, cc, hh, ww = second
    return r + h + 1 <= rr or rr + hh + 1 <= r or c + w + 1 <= cc or cc + ww + 1 <= c


def _place_rectangles(rng, canvas_shape, specs):
    height, width = canvas_shape
    order = sorted(range(len(specs)), key=lambda i: specs[i][0] * specs[i][1], reverse=True)
    for _ in range(24):
        placed = [None] * len(specs)
        for index in order:
            rh, rw = specs[index]
            if rh > height or rw > width:
                break
            candidates = [(r, c, rh, rw)
                          for r in range(height - rh + 1)
                          for c in range(width - rw + 1)]
            chosen = None
            for choice in rng.permutation(len(candidates)):
                candidate = candidates[int(choice)]
                if all(old is None or _boxes_have_gap(candidate, old) for old in placed):
                    chosen = candidate
                    break
            if chosen is None:
                break
            placed[index] = chosen
        if all(box is not None for box in placed):
            return placed
    return None


def _sample_canvas(rng, style):
    draw = float(rng.random())
    if style == 0 and draw < 0.45:
        return int(rng.integers(26, 31)), int(rng.integers(26, 31))
    if draw < 0.32:
        short = int(rng.integers(12, 19))
        long = int(rng.integers(23, 31))
        return (short, long) if rng.random() < 0.5 else (long, short)
    return int(rng.integers(15, 31)), int(rng.integers(15, 31))


def _sample_rectangle_count(rng, canvas_shape, split, index, style):
    h, w = canvas_shape
    capacity = min(7, max(2, (h * w) // 110 + 1))
    if min(h, w) < 16:
        capacity = min(capacity, 4)
    if index == 0 and split == "train":
        return int(rng.integers(2, min(4, capacity) + 1))
    if index == 0 and split == "test" and style == 1:
        return int(rng.integers(2, capacity + 1))
    if style == 0:
        weights = np.array([1, 3, 4, 2, 1, 0.7, 0.4], dtype=float)
    else:
        weights = np.array([2, 3, 3, 2, 1.6, 1.2, 0.8], dtype=float)
    choices = np.arange(1, capacity + 1)
    probabilities = weights[:capacity] / weights[:capacity].sum()
    return int(rng.choice(choices, p=probabilities))


def _sample_specs(rng, count, canvas_shape, split, style):
    h, w = canvas_shape
    short_cap = 8 if count <= 3 else 7 if count <= 5 else 6
    long_cap = 12 if count <= 2 else 10 if count <= 4 else 8
    short_cap = min(short_cap, h - 2, w - 2)
    long_cap = min(long_cap, max(h, w) - 2)
    if split == "train" and style == 1:
        long_cap = min(long_cap, 9)
    specs = []
    common = None
    if count > 1 and rng.random() < 0.22:
        side = int(rng.integers(5, min(8, short_cap) + 1))
        common = (side, side)
    for index in range(count):
        if common is not None and index < 2:
            rh, rw = common
        else:
            square = rng.random() < 0.28
            if square:
                side = int(rng.integers(5, short_cap + 1))
                rh = rw = side
            else:
                minimum = 5 if split == "train" and style == 1 else 4
                short = int(rng.integers(minimum, short_cap + 1))
                if short == 4:
                    long = int(rng.integers(5, long_cap + 1))
                else:
                    long = int(rng.integers(short, long_cap + 1))
                rh, rw = (short, long) if rng.random() < 0.5 else (long, short)
        if split == "test" and style == 1 and index < 2:
            rh, rw = (4, 5) if index == 0 else (7, 10)
            if rng.random() < 0.5:
                rh, rw = rw, rh
        if rh > h - 1 or rw > w - 1:
            rh, rw = min(rh, h - 1), min(rw, w - 1)
        specs.append((rh, rw))
    return specs


def _choose_palette(rng, available, fill_colors, size):
    chosen = list(dict.fromkeys(fill_colors))
    remaining = [c for c in available if c not in chosen]
    rng.shuffle(remaining)
    chosen += remaining[:max(0, size - len(chosen))]
    rng.shuffle(chosen)
    return tuple(chosen)


def _sample_fills(rng, palette, count, forced=None):
    fills = []
    if forced is not None:
        fills.append(int(forced))
    if count > len(fills) and rng.random() < 0.12:
        fills.extend([fills[0] if fills else int(rng.choice(palette))] * (count - len(fills)))
        return fills
    while len(fills) < count:
        if fills and rng.random() < 0.4:
            fills.append(int(rng.choice(fills)))
        else:
            fills.append(int(rng.choice(palette)))
    return fills


def _add_texture(grid, forbidden, palette, rng, style):
    h, w = grid.shape
    trials = int(rng.integers(0, 3)) if style == 0 else int(rng.integers(1, 7))
    for _ in range(trials):
        ph = int(rng.integers(1, 4))
        pw = int(rng.integers(1, 4))
        row = int(rng.integers(0, h - ph + 1))
        col = int(rng.integers(0, w - pw + 1))
        if not np.any(forbidden[row:row + ph, col:col + pw]):
            grid[row:row + ph, col:col + pw] = int(rng.choice(palette))


def _add_attachments(grid, forbidden, boxes, fills, rng, style):
    h, w = grid.shape
    for (row, col, rh, rw), color in zip(boxes, fills):
        if rng.random() < (0.38 if style == 0 else 0.72):
            sides = []
            if row > 0:
                sides.append([(row - 1, c) for c in range(col, col + rw)])
            if row + rh < h:
                sides.append([(row + rh, c) for c in range(col, col + rw)])
            if col > 0:
                sides.append([(r, col - 1) for r in range(row, row + rh)])
            if col + rw < w:
                sides.append([(r, col + rw) for r in range(row, row + rh)])
            if sides:
                edge = sides[int(rng.integers(len(sides)))]
                length = int(rng.integers(1, min(4, len(edge)) + 1))
                start = int(rng.integers(0, len(edge) - length + 1))
                for point in edge[start:start + length]:
                    if not forbidden[point]:
                        grid[point] = color


def _construct_pair(rng, canvas_shape, boxes, fills, palette, highlight, style):
    h, w = canvas_shape
    palette_array = np.asarray(palette, dtype=np.int8)
    mask = np.zeros(canvas_shape, dtype=bool)
    for row, col, rh, rw in boxes:
        mask[row:row + rh, col:col + rw] = True
    for _ in range(80):
        weights = rng.dirichlet(np.full(len(palette), 2.0 if style == 0 else 1.1))
        if float(weights.max()) > 0.72:
            continue
        grid = rng.choice(palette_array, size=canvas_shape, p=weights).astype(np.int8)
        _add_texture(grid, mask, palette, rng, style)
        for (row, col, rh, rw), color in zip(boxes, fills):
            grid[row:row + rh, col:col + rw] = color
        _add_attachments(grid, mask, boxes, fills, rng, style)
        if len(np.unique(grid)) != len(palette):
            continue
        if any(not np.any(grid[~mask] == color) for color in set(fills)):
            continue
        if not np.array_equal(_large_uniform_rectangle_mask(grid), mask):
            continue
        output = grid.copy()
        output[mask] = highlight
        return grid, output
    raise ValueError("unable to make an unambiguous camouflage field")


def _grid_key(grid):
    array = np.asarray(grid, dtype=np.uint8)
    return (tuple(array.shape), array.tobytes())


def _analyze_pair(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    changed = input_grid != output_grid
    assert np.any(changed)
    changed_colors = np.unique(output_grid[changed])
    assert len(changed_colors) == 1
    highlight = int(changed_colors[0])
    palette = frozenset(int(c) for c in np.unique(input_grid))
    assert 3 <= len(palette) <= 7
    assert highlight not in palette
    assert np.array_equal(_large_uniform_rectangle_mask(input_grid), changed)
    assert np.array_equal(output_grid[~changed], input_grid[~changed])
    boxes = _component_boxes(changed)
    assert 1 <= len(boxes) <= 7
    rectangles = []
    for row, col, rh, rw in boxes:
        assert rh >= _MIN_RECT_SIDE and rw >= _MIN_RECT_SIDE
        assert rh * rw >= _MIN_RECT_AREA
        patch = input_grid[row:row + rh, col:col + rw]
        assert len(np.unique(patch)) == 1
        fill = int(patch[0, 0])
        assert np.any(input_grid[~changed] == fill)
        rectangles.append({"box": (row, col, rh, rw), "fill": fill})
    for index, first in enumerate(boxes):
        for second in boxes[index + 1:]:
            assert _boxes_have_gap(first, second)
    return {"highlight": highlight, "palette": palette, "rectangles": rectangles}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    highlight = int(rng.integers(0, 10))
    available = tuple(c for c in range(10) if c != highlight)
    style = int(rng.integers(0, 3))  # simple, shifted, or mixed
    anchors = rng.choice(np.asarray(available), size=2, replace=False)
    if style == 0:
        stable_size = int(rng.integers(3, 6))
        stable_palette = _choose_palette(rng, available, anchors, stable_size)
    else:
        stable_palette = ()
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    for split, count, offset in (("train", train_count, 0), ("test", test_count, train_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[offset + index]))
            forced = int(anchors[index]) if split == "train" and index < 2 else None
            for _ in range(80):
                canvas = _sample_canvas(pair_rng, style)
                object_count = _sample_rectangle_count(pair_rng, canvas, split, index, style)
                specs = _sample_specs(pair_rng, object_count, canvas, split, style)
                boxes = _place_rectangles(pair_rng, canvas, specs)
                if boxes is None:
                    continue
                if style == 0:
                    palette = stable_palette
                else:
                    palette_size = int(pair_rng.integers(3, 8))
                    provisional = [forced] if forced is not None else []
                    palette = _choose_palette(pair_rng, available, provisional, palette_size)
                fills = _sample_fills(pair_rng, palette, object_count, forced)
                # Local palette size is chosen independently; repeated target colors
                # may require fewer distinct colors, never more than this palette.
                if any(color not in palette for color in fills):
                    continue
                try:
                    input_grid, output_grid = _construct_pair(
                        pair_rng, canvas, boxes, fills, palette, highlight, style
                    )
                except ValueError:
                    continue
                pair = {"input": input_grid, "output": output_grid}
                if _grid_key(input_grid) in {
                    _grid_key(p["input"])
                    for s in ("train", "test") for p in puzzle[s]
                }:
                    continue
                puzzle[split].append(pair)
                break
            else:
                raise ValueError("unable to construct a valid pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert hashlib.sha256(input_key[1]).hexdigest() not in _SOURCE_INPUT_DIGESTS or input_key[0] != (30, 30)
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    highlights = {a["highlight"] for group in analyses.values() for a in group}
    assert len(highlights) == 1
    train_fills = {r["fill"] for a in analyses["train"] for r in a["rectangles"]}
    assert len(train_fills) >= 2
    assert len(analyses["train"][0]["rectangles"]) >= 2
    # Every input supplies the geometry; training supplies the output-only color.
    learned = next(iter(highlights))
    for pair in puzzle["test"]:
        predicted = pair["input"].copy()
        predicted[_large_uniform_rectangle_mask(pair["input"])] = learned
        assert np.array_equal(predicted, pair["output"])


def _puzzle_key(puzzle):
    return tuple(
        tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split])
        for split in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    counts = set()
    sizes = set()
    shapes = set()
    palettes = set()
    repeated_colors = 0
    unequal_repeats = 0
    one_color_grids = 0
    equal_size_grids = 0
    small_multiple = 0
    large_multiple = 0
    edge_pairs = 0
    small_canvases = 0
    elongated_canvases = 0
    combined_pairs = 0
    variable_counts = 0
    variable_palettes = 0
    stable_palettes = 0
    source_like_shifts = 0
    shared_ranges = 0
    fewer_test_objects = 0
    novel_test_fills = 0
    square_sizes = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local_counts = set()
        local_palettes = set()
        train_areas = []
        test_areas = []
        train_counts = []
        test_counts = []
        train_fills = set()
        test_fills = set()
        learned = {
            int(pair["output"][pair["input"] != pair["output"]][0])
            for pair in puzzle["train"]
        }
        assert len(learned) == 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair["input"], pair["output"])
                rectangles = analysis["rectangles"]
                counts.add(len(rectangles))
                local_counts.add(len(rectangles))
                shapes.add(pair["input"].shape)
                palettes.add(tuple(sorted(analysis["palette"])))
                local_palettes.add(tuple(sorted(analysis["palette"])))
                fills = [r["fill"] for r in rectangles]
                repeated_colors += len(set(fills)) < len(fills)
                unequal_repeats += len(set(fills)) < len(fills) and len({
                    tuple(sorted(r["box"][2:])) for r in rectangles
                }) > 1
                one_color_grids += len(fills) >= 2 and len(set(fills)) == 1
                equal_size_grids += len(rectangles) >= 2 and len({
                    tuple(sorted(r["box"][2:])) for r in rectangles
                }) < len(rectangles)
                areas = [r["box"][2] * r["box"][3] for r in rectangles]
                small_multiple += sum(a <= 30 for a in areas) >= 2
                large_multiple += sum(a >= 56 for a in areas) >= 2
                h, w = pair["input"].shape
                small_canvases += h * w <= 400
                elongated_canvases += max(h, w) / min(h, w) >= 1.5
                edge_pairs += any(
                    r == 0 or c == 0 or r + rh == h or c + rw == w
                    for r, c, rh, rw in (item["box"] for item in rectangles)
                )
                combined_pairs += (
                    len(set(fills)) < len(fills)
                    and len(analysis["palette"]) >= 6
                    and h * w <= 500
                )
                if split == "train":
                    train_areas.extend(areas)
                    train_counts.append(len(rectangles))
                    train_fills.update(fills)
                else:
                    test_areas.extend(areas)
                    test_counts.append(len(rectangles))
                    test_fills.update(fills)
                for rect in rectangles:
                    rh, rw = rect["box"][2:]
                    sizes.add(tuple(sorted((rh, rw))))
                    if rh == rw:
                        square_sizes.add(rh)
                predicted = pair["input"].copy()
                predicted[_large_uniform_rectangle_mask(pair["input"])] = next(iter(learned))
                assert np.array_equal(predicted, pair["output"])
        variable_counts += len(local_counts) > 1
        variable_palettes += len(local_palettes) > 1
        stable_palettes += len(local_palettes) == 1
        source_like_shifts += (
            min(test_areas) < min(train_areas)
            and max(test_areas) > max(train_areas)
        )
        shared_ranges += (
            min(test_areas) >= min(train_areas)
            and max(test_areas) <= max(train_areas)
        )
        fewer_test_objects += min(test_counts) < min(train_counts)
        novel_test_fills += bool(test_fills - train_fills)
    assert {1, 2, 3, 4, 5} <= counts
    assert {5, 7, 8} <= square_sizes
    assert (4, 5) in sizes and (7, 10) in sizes
    assert len(shapes) >= 30 and len(palettes) >= 20 and len(sizes) >= 20
    assert repeated_colors >= 10 and one_color_grids >= 5
    assert unequal_repeats >= 10 and equal_size_grids >= 10
    assert small_multiple >= 5 and large_multiple >= 5
    assert edge_pairs >= 30 and small_canvases >= 20 and elongated_canvases >= 20
    assert combined_pairs >= 5
    assert variable_counts >= 30 and variable_palettes >= 20 and stable_palettes >= 2
    assert source_like_shifts >= 5 and shared_ranges >= 5
    assert fewer_test_objects >= 2 and novel_test_fills >= 2
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9000 + train_count * 10 + test_count, train_count, test_count)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(10000 + index, train_count, test_count)
        validate_generated_puzzle(p)
        assert train_count is None or len(p["train"]) == train_count
        assert test_count is None or len(p["test"]) == test_count
    explicit = generate_puzzle(8128, 7, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(8128, 7, 3))
    assert _puzzle_key(explicit) != _puzzle_key(generate_puzzle(8129, 7, 3))
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(123, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
