import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "00000000000000000000",
        "01111111110000000000",
        "01111111110000000000",
        "01111111110000000000",
        "00000000000000000000",
        "00000007777777777700",
        "00000007777777777700",
        "00000007777777777700",
    ),
    (
        "000000000000000000000000000000",
        "444444444444444444444444444440",
        "444444444444444444444444444440",
        "444444444444444444444444444440",
        "000000000000000000000000000000",
        "000000000000888888888888800000",
        "000000000000888888888888800000",
        "000000000000888888888888800000",
    ),
    (
        "00000000000000000000",
        "00555555500000000000",
        "00555555500000000000",
        "00555555500000000000",
        "00000000000000000000",
        "04444444444444000000",
        "04444444444444000000",
        "04444444444444000000",
        "00000000000000088888",
        "00000000000000088888",
        "00000000000000088888",
    ),
)


def _coerce_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _coerce_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _coerce_count("num_train", num_train, 2, 10)
    num_test = _coerce_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(value) for value in grid.flat)


def _is_source_input(grid):
    rows = tuple("".join(str(int(v)) for v in row) for row in grid)
    return rows in _SOURCE_INPUT_ROWS


def _components(grid, color):
    h, w = grid.shape
    marked = np.zeros((h, w), dtype=bool)
    result = []
    for y in range(h):
        for x in range(w):
            if marked[y, x] or grid[y, x] != color:
                continue
            marked[y, x] = True
            todo = [(y, x)]
            cells = []
            while todo:
                cy, cx = todo.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx),
                               (cy, cx - 1), (cy, cx + 1)):
                    if (0 <= ny < h and 0 <= nx < w and
                            not marked[ny, nx] and grid[ny, nx] == color):
                        marked[ny, nx] = True
                        todo.append((ny, nx))
            result.append(cells)
    return result


def _bars_for_background(grid, background):
    if not np.any(grid == background):
        return None
    bars = []
    for color in np.unique(grid):
        color = int(color)
        if color == background:
            continue
        for cells in _components(grid, color):
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            y, x = min(ys), min(xs)
            height, width = max(ys) - y + 1, max(xs) - x + 1
            if height * width != len(cells):
                return None
            if not np.all(grid[y:y + height, x:x + width] == color):
                return None
            if height == 3 and width >= 3 and width % 2:
                axis = "S" if width == 3 else "H"
            elif width == 3 and height >= 3 and height % 2:
                axis = "V"
            else:
                return None
            bars.append((y, x, height, width, color, axis))
    return sorted(bars) if bars else None


def _infer_bars(grid):
    candidates = []
    for color in np.unique(grid):
        bars = _bars_for_background(grid, int(color))
        if bars is not None:
            candidates.append((int(color), bars))
    assert len(candidates) == 1, "background or bars are ambiguous"
    return candidates[0]


def _make_output(input_grid, background, bars):
    output = input_grid.copy()
    for y, x, height, width, _color, axis in bars:
        if axis == "V":
            output[y + 1:y + height:2, x + 1] = background
        else:
            output[y + 1, x + 1:x + width:2] = background
    return output


def _dimension(rng, minimum):
    band = int(rng.integers(0, 3))
    low, high = ((3, 10), (11, 20), (21, 30))[band]
    low = max(low, minimum)
    if low > high:
        low, high = minimum, 30
    return int(rng.integers(low, high + 1))


def _odd_length(rng, maximum, minimum=3):
    choices = list(range(minimum if minimum % 2 else minimum + 1,
                         maximum + 1, 2))
    if not choices:
        return None
    return int(choices[int(rng.integers(len(choices)))])


def _colors(rng, background, count, allowed_colors=None):
    available = ([v for v in range(10) if v != background]
                 if allowed_colors is None else allowed_colors)
    if len(available) == 1:
        return [int(available[0])] * count
    mode = int(rng.integers(0, 3))
    if mode == 0 or count == 1:
        color = int(rng.choice(available))
        return [color] * count
    if mode == 1 and count >= 3:
        chosen = [int(v) for v in rng.choice(available, size=2, replace=False)]
        result = [chosen[i % 2] for i in range(count)]
        rng.shuffle(result)
        return result
    return [int(v) for v in rng.choice(available, size=count, replace=False)]


def _orientations(rng, profile, count, demonstrate=False):
    if profile == "horizontal":
        return ["H"] * count
    if profile == "vertical":
        return ["V"] * count
    if demonstrate:
        result = ["H", "V"]
        result.extend("H" if rng.random() < 0.5 else "V"
                      for _ in range(count - 2))
        return result
    horizontal_probability = 0.2 if profile == "vertical_mix" else 0.5
    return ["H" if rng.random() < horizontal_probability else "V"
            for _ in range(count)]


def _too_close_same_color(rect, placed):
    y, x, bh, bw, color = rect
    for py, px, ph, pw, pc in placed:
        if color != pc:
            continue
        if (y <= py + ph and py <= y + bh and
                x <= px + pw and px <= x + bw):
            return True
    return False


def _place_bars(rng, height, width, colors, axes, lengths, margin_mode):
    grid = np.full((height, width), -1, dtype=np.int8)
    placed = []
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    for ordinal, index in enumerate(order):
        axis, length, color = axes[index], lengths[index], colors[index]
        bh, bw = (3, length) if axis == "H" else (length, 3)
        margin = 2 if margin_mode == "interior" else 0
        options = []
        for y in range(margin, height - bh - margin + 1):
            for x in range(margin, width - bw - margin + 1):
                if np.any(grid[y:y + bh, x:x + bw] != -1):
                    continue
                rect = (y, x, bh, bw, color)
                if _too_close_same_color(rect, placed):
                    continue
                if (ordinal == 0 and margin_mode == "edge" and
                        not (y == 0 or x == 0 or y + bh == height or x + bw == width)):
                    continue
                options.append(rect)
        if not options:
            return None
        if placed and rng.random() < 0.55:
            py, px, ph, pw, _ = placed[-1]
            kinds = (
                lambda r: r[1] == px and
                (r[0] + r[2] <= py or py + ph <= r[0]),
                lambda r: r[0] == py and
                (r[1] + r[3] <= px or px + pw <= r[1]),
                lambda r: max(r[0], py) < min(r[0] + r[2], py + ph) and
                (r[1] + r[3] <= px or px + pw <= r[1]),
                lambda r: r[0] == py + ph or py == r[0] + r[2],
            )
            predicate = kinds[int(rng.integers(len(kinds)))]
            favored = [rect for rect in options if predicate(rect)]
            if favored:
                options = favored
        rect = options[int(rng.integers(len(options)))]
        y, x, bh, bw, _ = rect
        grid[y:y + bh, x:x + bw] = color
        placed.append(rect)
    return np.where(grid == -1, 0, grid), placed


def _generate_pair(rng, background, count, profile, demonstrate=False,
                   train_length_cap=None, forced_long=False,
                   allowed_colors=None, first_evidence=False):
    axes = _orientations(rng, profile, count, demonstrate)
    for _ in range(96):
        minimum = (3, 5, 7, 9, 11)[count - 1]
        if demonstrate:
            minimum = max(minimum, 10)
        height = _dimension(rng, minimum)
        width = _dimension(rng, minimum)
        if forced_long:
            if axes[0] == "H":
                width = max(width, int(rng.integers(19, 31)))
            else:
                height = max(height, int(rng.integers(19, 31)))
        colors = _colors(rng, background, count, allowed_colors)
        repeat_lengths = count >= 2 and rng.random() < 0.40
        shared = None
        if repeat_lengths:
            shared = _odd_length(
                rng, min(height, width, train_length_cap or 30),
                5 if first_evidence else 3)
        lengths = []
        for index, axis in enumerate(axes):
            maximum = width if axis == "H" else height
            if train_length_cap is not None:
                maximum = min(maximum, train_length_cap)
            if forced_long and index == 0:
                length = _odd_length(rng, maximum, 17)
            elif repeat_lengths and index < 2:
                length = shared
            else:
                minimum_length = 5 if (
                    (demonstrate and index < 2) or
                    (first_evidence and index == 0)) else 3
                length = _odd_length(rng, maximum, minimum_length)
            if length is None:
                break
            lengths.append(length)
        if len(lengths) != count:
            continue
        mode = ("interior", "edge", "free")[int(rng.integers(3))]
        result = _place_bars(rng, height, width, colors, axes, lengths, mode)
        if result is None:
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        for y, x, bh, bw, color in result[1]:
            input_grid[y:y + bh, x:x + bw] = color
        if _is_source_input(input_grid):
            continue
        try:
            found_background, bars = _infer_bars(input_grid)
        except AssertionError:
            continue
        if found_background != background or len(bars) != count:
            continue
        if demonstrate and not {"H", "V"} <= {bar[5] for bar in bars}:
            continue
        output_grid = _make_output(input_grid, background, bars)
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not place an unambiguous set of bars")


def _source_style_pair(rng, background, split, allowed_colors):
    """Keep the original sparse two-to-staggered-three progression as one mode."""
    count = 2 if split == "train" else 3
    height = 8 if split == "train" else 11
    for _ in range(128):
        width = int(rng.integers(15 if split == "train" else 18, 31))
        colors = _colors(rng, background, count, allowed_colors)
        if split == "train":
            tops = ((0, 4), (0, 5), (1, 5))[int(rng.integers(3))]
            lengths = [_odd_length(rng, width, 5) for _ in range(2)]
            xs = [int(rng.integers(width - length + 1)) for length in lengths]
        else:
            top = int(rng.integers(2))
            tops = (top, top + 4, top + 7)
            first = _odd_length(rng, width, 5)
            second = _odd_length(rng, min(17, width - 6), 5)
            third = _odd_length(rng, min(17, width - second - 1), 5)
            if third is None:
                continue
            lengths = [first, second, third]
            slack = width - second - third - 1
            left = int(rng.integers(slack + 1))
            gap = int(rng.integers(1, slack - left + 2))
            if rng.random() < 0.5:
                x2, x3 = left, left + second + gap
            else:
                x2, x3 = left + third + gap, left
            xs = [int(rng.integers(width - first + 1)), x2, x3]
        grid = np.full((height, width), background, dtype=np.int8)
        for y, x, length, color in zip(tops, xs, lengths, colors):
            grid[y:y + 3, x:x + length] = color
        if _is_source_input(grid):
            continue
        try:
            found_background, bars = _infer_bars(grid)
        except AssertionError:
            continue
        if found_background == background and len(bars) == count:
            return {"input": grid, "output": _make_output(grid, background, bars)}
    raise RuntimeError("could not construct a source-style scene")


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_axes = set()
    test_axes = set()
    holes = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            _validate_grid(source)
            _validate_grid(target)
            assert source.shape == target.shape
            assert not _is_source_input(source)
            key = _grid_key(source)
            assert key not in seen
            seen.add(key)
            background, bars = _infer_bars(source)
            assert 1 <= len(bars) <= 5
            assert np.array_equal(target, _make_output(source, background, bars))
            assert not np.array_equal(source, target)
            for y, x, height, width, color, axis in bars:
                assert color != background
                assert (height == 3 and width % 2 == 1) or (
                    width == 3 and height % 2 == 1)
                if axis == "H":
                    axis_set = {"H"}
                    holes += (width - 1) // 2 if split == "train" else 0
                elif axis == "V":
                    axis_set = {"V"}
                    holes += (height - 1) // 2 if split == "train" else 0
                else:
                    axis_set = set()
                    holes += 1 if split == "train" else 0
                if split == "train":
                    train_axes.update(axis_set)
                else:
                    test_axes.update(axis_set)
    assert test_axes <= train_axes
    assert holes >= 2


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _coerce_seed(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The bar rule has no hidden phase or color map. These parameters govern
    # coherent presentation and training coverage, not the answer.
    profile = ("horizontal", "vertical", "balanced_mix", "vertical_mix")[
        int(rng.integers(4))]
    arrangement = ("progression", "equal", "free")[int(rng.integers(3))]
    source_mode = rng.random() < 0.18
    if source_mode:
        profile = "horizontal"
        arrangement = "progression"
    background_style = ("fixed_two", "stable", "local")[
        int(rng.choice(3, p=[0.15, 0.25, 0.60]))]
    base_background = int(rng.integers(10))
    fixed_foreground = int(rng.choice(
        [c for c in range(10) if c != base_background]))
    long_test = arrangement == "free" and rng.random() < 0.45
    puzzle = {"train": [], "test": []}
    seen = set()
    shared_count = int(rng.integers(1, 5))
    if profile in ("balanced_mix", "vertical_mix"):
        shared_count = max(shared_count, 2)
    for index in range(num_train + num_test):
        split = "train" if index < num_train else "test"
        local_index = index if split == "train" else index - num_train
        if arrangement == "progression" and local_index == 0:
            count = 2 if split == "train" else 3
        elif arrangement == "equal" and local_index == 0:
            count = shared_count
        else:
            count = int(rng.integers(1, 6))
        if split == "train" and index == 0 and profile in (
                "balanced_mix", "vertical_mix"):
            count = max(count, 2)
        if split == "test" and local_index == 0 and long_test:
            count = 1
        if background_style != "local" or index == 0:
            background = base_background
        else:
            alternatives = [c for c in range(10) if c != base_background]
            background = int(rng.choice(alternatives))
        demonstrate = split == "train" and index == 0 and profile in (
            "balanced_mix", "vertical_mix")
        for _ in range(32):
            allowed_colors = (
                [fixed_foreground] if background_style == "fixed_two" else None)
            if source_mode and local_index == 0:
                pair = _source_style_pair(rng, background, split, allowed_colors)
            else:
                pair = _generate_pair(
                    rng, background, count, profile, demonstrate,
                    13 if split == "train" and long_test else None,
                    split == "test" and local_index == 0 and long_test,
                    allowed_colors, split == "train" and index == 0)
            key = _grid_key(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("could not generate a distinct input")
        seen.add(key)
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    train_counts, test_counts = set(), set()
    structures, sizes, backgrounds, object_counts = set(), set(), set(), set()
    orientations, one_color, repeated_lengths = set(), 0, 0
    coverage = {}

    def count(name, condition):
        coverage[name] = coverage.get(name, 0) + int(bool(condition))

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_backgrounds = set()
        puzzle_bar_counts = set()
        puzzle_shapes = set()
        puzzle_palettes = set()
        largest_train, largest_test = 0, 0
        train_bar_counts, test_bar_counts = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                background, bars = _infer_bars(source)
                puzzle_backgrounds.add(background)
                puzzle_bar_counts.add(len(bars))
                puzzle_shapes.add(source.shape)
                puzzle_palettes.add(tuple(int(v) for v in np.unique(source)))
                (train_bar_counts if split == "train" else test_bar_counts).add(
                    len(bars))
                expected_changes = set()
                for y, x, height, width, _color, axis in bars:
                    if axis == "V":
                        expected_changes.update((row, x + 1)
                                                for row in range(y + 1, y + height, 2))
                    else:
                        expected_changes.update((y + 1, col)
                                                for col in range(x + 1, x + width, 2))
                actual_changes = {tuple(int(v) for v in cell)
                                  for cell in np.argwhere(source != target)}
                assert actual_changes == expected_changes
                assert all(target[y, x] == background for y, x in actual_changes)
                structures.add((source.shape, tuple((y, x, h, w, a)
                                                    for y, x, h, w, _, a in bars)))
                sizes.add(source.shape)
                backgrounds.add(background)
                object_counts.add(len(bars))
                orientations.update(bar[5] for bar in bars)
                one_color += len({bar[4] for bar in bars}) == 1
                repeated_lengths += len({max(bar[2], bar[3]) for bar in bars}) < len(bars)
                lengths = [max(bar[2], bar[3]) for bar in bars]
                if split == "train":
                    largest_train = max(largest_train, *lengths)
                else:
                    largest_test = max(largest_test, *lengths)
                    count("long_test_bar", max(lengths) > 17)
                axes = {bar[5] for bar in bars}
                h, w = source.shape
                count("single_bar", len(bars) == 1)
                count("multi_bar_one_color", len(bars) >= 2 and
                      len({bar[4] for bar in bars}) == 1)
                count("repeated_length", len(set(lengths)) < len(lengths))
                count("unequal_length", len(set(lengths)) > 1)
                count("mixed_axes", {"H", "V"} <= axes)
                count("square_bar", "S" in axes)
                count("small_canvas", h <= 10 and w <= 10)
                count("square_canvas", h == w)
                count("tall_canvas", h >= 1.5 * w)
                count("wide_canvas", w >= 1.5 * h)
                count("edge_contact", any(y == 0 or x == 0 or y + bh == h or
                      x + bw == w for y, x, bh, bw, _, _ in bars))
                count("generous_margin", all(y >= 2 and x >= 2 and
                      y + bh <= h - 2 and x + bw <= w - 2
                      for y, x, bh, bw, _, _ in bars))
                count("background_nonmajority",
                      np.count_nonzero(source == background) <
                      max(np.count_nonzero(source == v)
                          for v in np.unique(source) if v != background))
                count("black_foreground", any(bar[4] == 0 for bar in bars))
                side = partial = shared_left = stagger = False
                for i, a in enumerate(bars):
                    for b in bars[i + 1:]:
                        ay, ax, ah, aw = a[:4]
                        by, bx, bh, bw = b[:4]
                        overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
                        separate_columns = ax + aw <= bx or bx + bw <= ax
                        side |= overlap > 0 and separate_columns
                        partial |= 0 < overlap < min(ah, bh) and separate_columns
                        shared_left |= ax == bx and (
                            ay + ah <= by or by + bh <= ay)
                        stagger |= (ay + ah == by or by + bh == ay) and (
                            separate_columns)
                count("side_by_side", side)
                count("partial_row_overlap", partial)
                count("shared_left", shared_left)
                count("dense_stagger", stagger)
        count("varying_backgrounds", len(puzzle_backgrounds) > 1)
        count("stable_background", len(puzzle_backgrounds) == 1)
        count("fixed_palette", len(puzzle_palettes) == 1)
        count("within_count_variation", len(puzzle_bar_counts) > 1)
        count("within_shape_variation", len(puzzle_shapes) > 1)
        count("equal_train_test_count", bool(train_bar_counts & test_bar_counts))
        count("test_longer_than_train", largest_test > largest_train)
        first_train = _infer_bars(puzzle["train"][0]["input"])[1]
        first_test = _infer_bars(puzzle["test"][0]["input"])[1]
        count("source_style_progression",
              puzzle["train"][0]["input"].shape[0] == 8 and
              puzzle["test"][0]["input"].shape[0] == 11 and
              tuple(bar[0] for bar in first_train) in (
                  (0, 4), (0, 5), (1, 5)) and
              tuple(bar[0] for bar in first_test) in (
                  (0, 4, 7), (1, 5, 8)))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(structures) >= 180 and len(sizes) >= 80
    assert len(backgrounds) >= 7 and len(object_counts) >= 4
    assert {"H", "V"} <= orientations
    assert one_color >= 15 and repeated_lengths >= 10
    minimums = {
        "single_bar": 10, "multi_bar_one_color": 20,
        "repeated_length": 20, "unequal_length": 20,
        "mixed_axes": 10, "square_bar": 10,
        "small_canvas": 5, "square_canvas": 5,
        "tall_canvas": 10, "wide_canvas": 10,
        "edge_contact": 20, "generous_margin": 10,
        "background_nonmajority": 2, "black_foreground": 5,
        "side_by_side": 20, "partial_row_overlap": 10,
        "shared_left": 10, "dense_stagger": 5,
        "long_test_bar": 3, "test_longer_than_train": 2,
        "varying_backgrounds": 5, "stable_background": 5,
        "fixed_palette": 2, "within_count_variation": 20,
        "within_shape_variation": 20, "equal_train_test_count": 10,
        "source_style_progression": 1,
    }
    assert all(coverage.get(name, 0) >= minimum
               for name, minimum in minimums.items()), coverage
    assert _signature(generate_puzzle(123456, 7, 3)) == _signature(
        generate_puzzle(123456, 7, 3))
    assert _signature(generate_puzzle(0)) != _signature(generate_puzzle(1))
    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                             (7, 3), (None, 1), (None, 4),
                                             (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + offset, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(4000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count was accepted")
