import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "2200000000",
            "2200000000",
            "2200000222",
            "2200000222",
            "0000000222",
            "0002200222",
            "0002200000",
            "0002200000",
            "0002202222",
            "0000002222",
        ),
        (
            "2200000000",
            "2200000000",
            "2299999222",
            "2299999222",
            "0000000222",
            "0002299222",
            "0002200000",
            "0002200000",
            "0002292222",
            "0000002222",
        ),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "2220000000",
            "2220000220",
            "2220000220",
            "0000000220",
            "0002200220",
            "0002200220",
            "0002200000",
            "0002200000",
        ),
        (
            "0000000000",
            "0000000000",
            "2220000000",
            "2229999220",
            "2229999220",
            "0000000220",
            "0002299220",
            "0002299220",
            "0002200000",
            "0002200000",
        ),
    ),
    (
        (
            "0000002222",
            "2222002222",
            "2222000000",
            "2222022200",
            "0000022200",
            "0000022200",
            "0000022202",
            "2222022202",
            "2222022202",
            "2222000002",
        ),
        (
            "0000002222",
            "2222992222",
            "2222000000",
            "2222922200",
            "0000022200",
            "0000022200",
            "0000022292",
            "2222922292",
            "2222922292",
            "2222000002",
        ),
    ),
    (
        (
            "2220000000",
            "2220002222",
            "0000002222",
            "0222002222",
            "0222000000",
            "0222022220",
            "0222022220",
            "0222000000",
            "0000000000",
            "0000000000",
        ),
        (
            "2220000000",
            "2229992222",
            "0000002222",
            "0222992222",
            "0222000000",
            "0222922220",
            "0222922220",
            "0222000000",
            "0000000000",
            "0000000000",
        ),
    ),
)


def _coerce_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    num_train = _coerce_count(num_train, "num_train", 2, 10)
    num_test = _coerce_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        options = list(range(max(2, num_test + 1), 11))
        num_train = options[(seed // 4) % len(options)]
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _palette_choices(rng):
    values = [int(v) for v in rng.permutation(10)]
    connector = values[0]
    palettes = [(values[1], values[2])]
    if rng.random() < 0.6:
        palettes.append((values[3], values[4]))
    return connector, palettes


def _separated(first, second):
    a, b, c, d = first
    e, f, g, h = second
    return d + 1 < g or h + 1 < c or b + 1 < e or f + 1 < a


def _normalize(rectangles, rng, fixed_size=None):
    min_y = min(r[0] for r in rectangles)
    max_y = max(r[1] for r in rectangles)
    min_x = min(r[2] for r in rectangles)
    max_x = max(r[3] for r in rectangles)
    span_h, span_w = max_y - min_y + 1, max_x - min_x + 1
    if fixed_size is None:
        top, left = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        bottom, right = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        height, width = span_h + top + bottom, span_w + left + right
    else:
        height, width = fixed_size
        if span_h > height or span_w > width:
            return None
        top = int(rng.integers(0, height - span_h + 1))
        left = int(rng.integers(0, width - span_w + 1))
    if height > 30 or width > 30:
        return None
    placed = [(a - min_y + top, b - min_y + top,
               c - min_x + left, d - min_x + left) for a, b, c, d in rectangles]
    return height, width, placed


def _make_grid(layout, background, object_color):
    height, width, rectangles = layout
    grid = np.full((height, width), background, dtype=np.int8)
    for top, bottom, left, right in rectangles:
        assert np.all(grid[top:bottom + 1, left:right + 1] == background)
        grid[top:bottom + 1, left:right + 1] = object_color
    return grid


def _blocked_layout(rng):
    # The middle object interrupts only part of the left/right shared span.
    for _ in range(80):
        left_width, middle_width, right_width = (int(rng.integers(1, 6)) for _ in range(3))
        first_gap, second_gap = (int(rng.integers(1, 7)) for _ in range(2))
        left_height, right_height = (int(rng.integers(4, 9)) for _ in range(2))
        right_top = int(rng.integers(-2, 3))
        shared_top = max(0, right_top)
        shared_bottom = min(left_height - 1, right_top + right_height - 1)
        if shared_bottom - shared_top + 1 < 3:
            continue
        middle_top = int(rng.integers(shared_top, shared_bottom))
        middle_bottom = int(rng.integers(middle_top, shared_bottom))
        middle_left = left_width + first_gap
        right_left = middle_left + middle_width + second_gap
        rectangles = [
            (0, left_height - 1, 0, left_width - 1),
            (middle_top, middle_bottom, middle_left, middle_left + middle_width - 1),
            (right_top, right_top + right_height - 1, right_left, right_left + right_width - 1),
        ]
        layout = _normalize(rectangles, rng)
        if layout is not None:
            return layout
    raise AssertionError("blocked layout construction failed")


def _random_layout(rng, count, fixed_size=None, unchanged=False):
    for _ in range(90):
        width = int(rng.integers(1, 7))
        height = int(rng.integers(2, 9))
        rectangles = [(0, height - 1, 0, width - 1)]
        complete = True
        for _ in range(count - 1):
            added = False
            for _ in range(100):
                anchor = rectangles[int(rng.integers(len(rectangles)))]
                h = int(rng.integers(2, 9))
                w = int(rng.integers(1, 7))
                stack = unchanged or rng.random() < 0.28
                if stack:
                    left = int(rng.integers(anchor[2] - w + 1, anchor[3] + 1))
                    gap = int(rng.integers(1, 5))
                    if rng.random() < 0.5:
                        top = anchor[1] + gap + 1
                    else:
                        top = anchor[0] - gap - h
                else:
                    top = int(rng.integers(anchor[0] - h + 1, anchor[1] + 1))
                    gap = int(rng.integers(1, 9))
                    if rng.random() < 0.5:
                        left = anchor[3] + gap + 1
                    else:
                        left = anchor[2] - gap - w
                candidate = (top, top + h - 1, left, left + w - 1)
                if not all(_separated(candidate, old) for old in rectangles):
                    continue
                trial = rectangles + [candidate]
                if (max(r[1] for r in trial) - min(r[0] for r in trial) >= 29 or
                        max(r[3] for r in trial) - min(r[2] for r in trial) >= 29):
                    continue
                rectangles.append(candidate)
                added = True
                break
            if not added:
                complete = False
                break
        if complete:
            layout = _normalize(rectangles, rng, fixed_size)
            if layout is not None:
                return layout
    raise AssertionError("random layout construction failed")


def _compact_layout(rng):
    # Source-scale canvas with varying widths, heights, gaps, and vertical offsets.
    for _ in range(100):
        wa, wb = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        gap = int(rng.integers(1, 6))
        if wa + gap + wb > 10:
            continue
        ha, hb = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        y_a = int(rng.integers(0, 11 - ha))
        y_b = int(rng.integers(0, 11 - hb))
        if max(y_a, y_b) > min(y_a + ha - 1, y_b + hb - 1):
            continue
        x_a = int(rng.integers(0, 11 - wa - gap - wb))
        x_b = x_a + wa + gap
        return (10, 10, [(y_a, y_a + ha - 1, x_a, x_a + wa - 1),
                         (y_b, y_b + hb - 1, x_b, x_b + wb - 1)])
    raise AssertionError("compact layout construction failed")


def _rectangles(grid, object_color):
    mask = grid == object_color
    seen = np.zeros(mask.shape, dtype=bool)
    found = []
    height, width = mask.shape
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= yy < height and 0 <= xx < width and mask[yy, xx] and not seen[yy, xx]:
                        seen[yy, xx] = True
                        stack.append((yy, xx))
            top = min(y for y, _ in cells)
            bottom = max(y for y, _ in cells)
            left = min(x for _, x in cells)
            right = max(x for _, x in cells)
            assert len(cells) == (bottom - top + 1) * (right - left + 1)
            assert np.all(mask[top:bottom + 1, left:right + 1])
            found.append((top, bottom, left, right))
    return found


def _clear_span_output(grid, object_color, connector):
    result = grid.copy()
    rectangles = _rectangles(grid, object_color)
    for index, first in enumerate(rectangles):
        for second in rectangles[index + 1:]:
            if first[3] < second[2]:
                left, right = first, second
            elif second[3] < first[2]:
                left, right = second, first
            else:
                continue
            top = max(left[0], right[0])
            bottom = min(left[1], right[1])
            if top > bottom or right[2] - left[3] <= 1:
                continue
            strip = grid[top:bottom + 1, left[3] + 1:right[2]]
            if np.all(strip != object_color):
                result[top:bottom + 1, left[3] + 1:right[2]] = connector
    return result


def _rowwise_output(grid, object_color, connector):
    result = grid.copy()
    for row in range(grid.shape[0]):
        columns = np.flatnonzero(grid[row] == object_color)
        for a, b in zip(columns, columns[1:]):
            if b - a > 1:
                result[row, a + 1:b] = connector
    return result


def _cellwise_output(grid, object_color, connector):
    """Independent cell-by-cell formulation used for prediction checks."""
    rectangles = _rectangles(grid, object_color)
    result = grid.copy()
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[row, col] == object_color:
                continue
            for first in rectangles:
                for second in rectangles:
                    if first is second or not (first[3] < col < second[2]):
                        continue
                    if not (first[0] <= row <= first[1] and second[0] <= row <= second[1]):
                        continue
                    top = max(first[0], second[0])
                    bottom = min(first[1], second[1])
                    if np.all(grid[top:bottom + 1, first[3] + 1:second[2]] != object_color):
                        result[row, col] = connector
                        break
                if result[row, col] == connector:
                    break
    return result


def _predict_from_training(training, test_input):
    palettes = {}
    connector = None
    for pair in training:
        before, after = pair["input"], pair["output"]
        changed = before != after
        if not np.any(changed):
            continue
        background = int(np.unique(before[changed])[0])
        new_color = int(np.unique(after[changed])[0])
        if connector is None:
            connector = new_color
        assert connector == new_color
        colors = frozenset(int(c) for c in np.unique(before))
        object_color = next(c for c in colors if c != background)
        palettes[colors] = (background, object_color)
    assert connector is not None
    colors = frozenset(int(c) for c in np.unique(test_input))
    assert colors in palettes
    return _cellwise_output(test_input, palettes[colors][1], connector)


def _row_run_max(grid, object_color):
    maximum = 0
    for row in grid:
        columns = np.flatnonzero(row == object_color)
        if columns.size:
            maximum = max(maximum, 1 + int(np.count_nonzero(np.diff(columns) > 1)))
    return maximum


def _signature(grid):
    return grid.shape, grid.tobytes()


def _source_copy(input_grid, output_grid):
    if input_grid.shape != (10, 10):
        return False
    rows = (tuple("".join(str(int(c)) for c in row) for row in input_grid),
            tuple("".join(str(int(c)) for c in row) for row in output_grid))
    return rows in _SOURCE_PAIR_ROWS


def _make_pair(rng, connector, palette, kind, count=None):
    background, object_color = palette
    for _ in range(100):
        if kind == "blocked":
            layout = _blocked_layout(rng)
        elif kind == "compact":
            layout = _compact_layout(rng)
        elif kind == "unchanged":
            layout = _random_layout(rng, count or int(rng.integers(2, 4)), unchanged=True)
        else:
            layout = _random_layout(rng, count or int(rng.integers(2, 7)))
        input_grid = _make_grid(layout, background, object_color)
        output_grid = _clear_span_output(input_grid, object_color, connector)
        changed = not np.array_equal(input_grid, output_grid)
        if kind == "unchanged" and changed:
            continue
        if kind != "unchanged" and not changed:
            continue
        if kind == "blocked" and np.array_equal(output_grid, _rowwise_output(input_grid, object_color, connector)):
            continue
        return {"input": input_grid, "output": output_grid}
    raise AssertionError("pair construction failed")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    connector, palettes = _palette_choices(rng)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, total in (("train", num_train), ("test", num_test)):
        for index in range(total):
            if split == "train" and index == 0:
                kind = "blocked"
            elif split == "test" and index == 0 and seed % 7 == 0:
                kind = "compact"
            elif (split == "test" or index >= len(palettes)) and rng.random() < 0.12:
                kind = "unchanged"
            else:
                kind = "random"
            if split == "train" and index < len(palettes):
                palette = palettes[index]
            else:
                palette = palettes[int(rng.integers(len(palettes)))]
            for _ in range(100):
                if kind == "random":
                    count = int(rng.integers(2, 7 if split == "train" else 8))
                else:
                    count = None
                pair = _make_pair(rng, connector, palette, kind, count)
                key = _signature(pair["input"])
                if key in used or _source_copy(pair["input"], pair["output"]):
                    continue
                used.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError("unique pair construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert 0 <= int(np.min(grid)) and int(np.max(grid)) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    palettes = {}
    connector = None
    seen_inputs, seen_pairs = set(), set()
    blocked_demo = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            before, after = pair["input"], pair["output"]
            _check_grid(before)
            _check_grid(after)
            assert before.shape == after.shape
            key = _signature(before)
            pair_key = (key, _signature(after))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            assert not _source_copy(before, after)
            colors = [int(c) for c in np.unique(before)]
            assert len(colors) == 2
            changed = before != after
            if np.any(changed):
                bg_values = np.unique(before[changed])
                new_values = np.unique(after[changed])
                assert len(bg_values) == 1 and len(new_values) == 1
                background, new_color = int(bg_values[0]), int(new_values[0])
                assert new_color not in colors
                if connector is None:
                    connector = new_color
                assert connector == new_color
                object_color = next(c for c in colors if c != background)
                if split == "train":
                    previous = palettes.get(frozenset(colors))
                    assert previous is None or previous == (background, object_color)
                    palettes[frozenset(colors)] = (background, object_color)
            else:
                assert np.array_equal(before, after)
    assert connector is not None and palettes
    for split in ("train", "test"):
        for pair in puzzle[split]:
            before, after = pair["input"], pair["output"]
            palette = palettes.get(frozenset(int(c) for c in np.unique(before)))
            assert palette is not None
            background, object_color = palette
            assert connector not in (background, object_color)
            rectangles = _rectangles(before, object_color)
            assert 2 <= len(rectangles) <= 7
            for index, first in enumerate(rectangles):
                for second in rectangles[index + 1:]:
                    assert _separated(first, second)
            expected = _clear_span_output(before, object_color, connector)
            assert np.array_equal(after, expected)
            assert np.all(after[before == object_color] == object_color)
            assert np.all(after[before != after] == connector)
            if split == "train" and not np.array_equal(after, _rowwise_output(before, object_color, connector)):
                blocked_demo = True
    assert blocked_demo


def _puzzles_equal(first, second):
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if any(not np.array_equal(a[k], b[k]) for k in ("input", "output")):
                return False
    return True


def self_test() -> None:
    train_counts, test_counts, palettes, sizes, object_counts = set(), set(), set(), set(), set()
    layouts, max_runs, gap_lengths = set(), set(), set()
    stacked = edge = short_wide = thin_tall = unchanged = blocked = compact = 0
    pair_total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training_copy = [
            {"input": pair["input"].copy(), "output": pair["output"].copy()}
            for pair in puzzle["train"]
        ]
        for pair in puzzle["test"]:
            prediction = _predict_from_training(training_copy, pair["input"].copy())
            assert np.array_equal(prediction, pair["output"])
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        learned_palettes = {}
        for pair in puzzle["train"]:
            before, after = pair["input"], pair["output"]
            if np.any(before != after):
                background = int(np.unique(before[before != after])[0])
                colors = [int(c) for c in np.unique(before)]
                learned_palettes[frozenset(colors)] = (
                    background, next(c for c in colors if c != background))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                pair_total += 1
                before, after = pair["input"], pair["output"]
                sizes.add(before.shape)
                if before.shape == (10, 10):
                    compact += 1
                colors = [int(c) for c in np.unique(before)]
                if np.any(before != after):
                    background = int(np.unique(before[before != after])[0])
                    object_color = next(c for c in colors if c != background)
                    connector = int(np.unique(after[before != after])[0])
                    palettes.add((background, object_color, connector))
                else:
                    unchanged += 1
                    object_color = learned_palettes[frozenset(colors)][1]
                rectangles = _rectangles(before, object_color)
                object_counts.add(len(rectangles))
                mask = before == object_color
                layouts.add((before.shape, mask.tobytes()))
                max_runs.add(_row_run_max(before, object_color))
                edge += int(any(r[0] == 0 or r[1] == before.shape[0] - 1 or
                                r[2] == 0 or r[3] == before.shape[1] - 1 for r in rectangles))
                short_wide += int(any(r[3] - r[2] + 1 >= 4 and r[1] - r[0] + 1 <= 3 for r in rectangles))
                thin_tall += int(any(r[3] - r[2] + 1 <= 2 and r[1] - r[0] + 1 >= 5 for r in rectangles))
                for i, a in enumerate(rectangles):
                    for b in rectangles[i + 1:]:
                        if a[2] <= b[3] and b[2] <= a[3] and (a[1] < b[0] or b[1] < a[0]):
                            stacked += 1
                        if a[3] < b[2] or b[3] < a[2]:
                            left, right = (a, b) if a[3] < b[2] else (b, a)
                            if max(a[0], b[0]) <= min(a[1], b[1]):
                                gap_lengths.add(right[2] - left[3] - 1)
                if split == "train" and np.any(before != after) and not np.array_equal(after, _rowwise_output(before, object_color, connector)):
                    blocked += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 20 and len(sizes) >= 25
    assert len(layouts) >= int(0.75 * pair_total)
    assert {2, 3, 4, 5}.issubset(object_counts)
    assert max(max_runs) >= 3 and max(gap_lengths) >= 5
    assert stacked >= 10 and edge >= 20 and short_wide >= 10 and thin_tall >= 10
    assert unchanged >= 1 and blocked >= 42 and compact >= 6
    assert _puzzles_equal(generate_puzzle(777), generate_puzzle(777))
    assert _puzzles_equal(generate_puzzle(778, 7, 3), generate_puzzle(778, 7, 3))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + index, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
    for index, kwargs in enumerate(({"num_test": 1}, {"num_test": 4},
                                    {"num_train": 2}, {"num_train": 10})):
        validate_generated_puzzle(generate_puzzle(1100 + index, **kwargs))
    for index, kwargs in enumerate(({"num_train": 1}, {"num_train": 11},
                                    {"num_test": 0}, {"num_test": 5},
                                    {"num_train": 2, "num_test": 2},
                                    {"num_train": 3, "num_test": 4})):
        try:
            generate_puzzle(1200 + index, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
