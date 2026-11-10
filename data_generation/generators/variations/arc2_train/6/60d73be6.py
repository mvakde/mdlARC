"""Generate completion puzzles by reflecting one populated quadrant in a cross."""

import numpy as np


_SOURCE_INPUT_ROWS = (
    ("777777771727797", "777777771747977", "777777771377777",
     "777777771774775", "111111111111111", "777777771777777",
     "777777771777777", "777777771777777", "777777771777777",
     "777777771777777"),
    ("17776777777", "22776777777", "04776777777", "74776777777",
     "78776777777", "66666666666", "77776777777", "77776777777",
     "77776777777", "77776777777", "77776777777"),
    ("77738071", "77737767", "77730270", "77737777", "33333333",
     "77737777", "77737777", "77737777"),
    ("77797747777777777", "77797047777777777", "67077747777777777",
     "87777547777777777", "79783747777777777", "67727747777777777",
     "44444444444444444", "77777747777777777", "77777747777777777",
     "77777747777777777", "77777747777777777"),
)


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(value) for value in grid.flat)


def _matches_source_input(grid: np.ndarray) -> bool:
    for rows in _SOURCE_INPUT_ROWS:
        if grid.shape == (len(rows), len(rows[0])):
            if tuple("".join(str(int(x)) for x in row) for row in grid) == rows:
                return True
    return False


def _analyze_input(grid: np.ndarray) -> dict:
    height, width = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) >= 3
    assert np.count_nonzero(counts == counts.max()) == 1
    background = int(colors[np.argmax(counts)])
    candidates = []
    for color in colors:
        color = int(color)
        if color == background:
            continue
        full_rows = np.flatnonzero(np.all(grid == color, axis=1))
        full_columns = np.flatnonzero(np.all(grid == color, axis=0))
        if len(full_rows) and len(full_columns):
            candidates.append((color, full_rows, full_columns))
    assert len(candidates) == 1
    axis_color, rows, columns = candidates[0]
    assert len(rows) == len(columns) == 1
    axis_row, axis_col = int(rows[0]), int(columns[0])
    assert 1 < axis_row < height - 2
    assert 1 < axis_col < width - 2
    axis_mask = np.zeros(grid.shape, dtype=bool)
    axis_mask[axis_row, :] = True
    axis_mask[:, axis_col] = True
    assert np.all(grid[axis_mask] == axis_color)
    assert not np.any(grid[~axis_mask] == axis_color)
    accent_mask = (grid != background) & ~axis_mask
    ys, xs = np.nonzero(accent_mask)
    assert len(ys) >= 2
    quadrants = ((ys < axis_row) * 2 + (xs < axis_col)).astype(int)
    assert len(np.unique(quadrants)) == 1
    quadrant = (0 if ys[0] < axis_row else 2) + (0 if xs[0] < axis_col else 1)
    assert np.all(grid[~axis_mask & ~accent_mask] == background)
    return {
        "height": height, "width": width, "background": background,
        "axis_color": axis_color, "axis_row": axis_row, "axis_col": axis_col,
        "accent_mask": accent_mask, "accent_colors": tuple(int(v) for v in np.unique(grid[accent_mask])),
        "ys": ys, "xs": xs, "quadrant": quadrant,
    }


def _complete_reflection(grid: np.ndarray, info: dict) -> np.ndarray:
    result = grid.copy()
    for row, col in zip(info["ys"], info["xs"]):
        row, col = int(row), int(col)
        for target_row in (row, 2 * info["axis_row"] - row):
            for target_col in (col, 2 * info["axis_col"] - col):
                if 0 <= target_row < grid.shape[0] and 0 <= target_col < grid.shape[1]:
                    result[target_row, target_col] = grid[row, col]
    return result


def _clipping(info: dict) -> tuple[bool, bool]:
    vertical = any(not 0 <= 2 * info["axis_row"] - int(y) < info["height"] for y in info["ys"])
    horizontal = any(not 0 <= 2 * info["axis_col"] - int(x) < info["width"] for x in info["xs"])
    return vertical, horizontal


def _visible_both(info: dict) -> bool:
    return any(
        0 <= 2 * info["axis_row"] - int(y) < info["height"]
        and 0 <= 2 * info["axis_col"] - int(x) < info["width"]
        for y, x in zip(info["ys"], info["xs"])
    )


def _choose_counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
                raise ValueError(f"{name} is outside its legal integer range")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _side_lengths(rng, clipped: bool, regime: str) -> tuple[int, int]:
    # The active side and the opposite side independently control the
    # position of an axis. Actual clipping is confirmed from colored cells.
    upper = 11 if regime == "source_train" else 13
    active = int(rng.integers(4, upper))
    if clipped:
        opposite = int(rng.integers(3, active))
    else:
        opposite = int(rng.integers(active, min(upper + 4, 30 - active)))
    return active, opposite


def _distance_interval(rng, active: int, opposite: int, clipped: bool) -> tuple[int, int]:
    if clipped:
        near = int(rng.integers(1, min(opposite, active - 1) + 1))
        far = int(rng.integers(max(opposite + 1, near + 1), active + 1))
    else:
        near = int(rng.integers(1, active))
        far = int(rng.integers(near + 1, active + 1))
    return near, far


def _pattern_cells(rng, y_interval, x_interval, mode: int) -> set[tuple[int, int]]:
    ya, yb = y_interval
    xa, xb = x_interval
    cells = {(ya, xa)}
    area = (yb - ya + 1) * (xb - xa + 1)
    if mode == 0:  # sparse independent points
        count = int(rng.integers(3, min(area, 16) + 1))
        choices = [(y, x) for y in range(ya, yb + 1) for x in range(xa, xb + 1)]
        for index in rng.choice(len(choices), size=min(count, len(choices)), replace=False):
            cells.add(choices[int(index)])
    elif mode == 1:  # short strokes
        for _ in range(int(rng.integers(2, 5))):
            if int(rng.integers(2)):
                y = int(rng.integers(ya, yb + 1))
                x0 = int(rng.integers(xa, xb))
                x1 = int(rng.integers(x0 + 1, xb + 1))
                cells.update((y, x) for x in range(x0, x1 + 1))
            else:
                x = int(rng.integers(xa, xb + 1))
                y0 = int(rng.integers(ya, yb))
                y1 = int(rng.integers(y0 + 1, yb + 1))
                cells.update((y, x) for y in range(y0, y1 + 1))
    elif mode == 2:  # a solid block, sometimes with an isolated point
        y0 = int(rng.integers(ya, yb))
        x0 = int(rng.integers(xa, xb))
        y1 = int(rng.integers(y0 + 1, min(yb, y0 + 3) + 1))
        x1 = int(rng.integers(x0 + 1, min(xb, x0 + 3) + 1))
        cells.update((y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1))
    else:  # connected walk with optional isolated cells
        y, x = int(rng.integers(ya, yb + 1)), int(rng.integers(xa, xb + 1))
        for _ in range(int(rng.integers(7, 19))):
            cells.add((y, x))
            dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            y = min(yb, max(ya, y + dy))
            x = min(xb, max(xa, x + dx))
        if mode == 4:
            for _ in range(int(rng.integers(2, 6))):
                cells.add((int(rng.integers(ya, yb + 1)), int(rng.integers(xa, xb + 1))))
    return cells


def _make_pair(rng, regime: str, color_order: tuple[int, ...], background: int,
               previous_axis: int | None) -> tuple[np.ndarray, np.ndarray, int]:
    quadrant = int(rng.integers(4))
    upper = quadrant in (0, 1)
    left = quadrant in (0, 2)
    # Each axis has its own clipping choice. Tests lean toward two-axis
    # clipping but still sample all four cases and overlapping sizes.
    clip_probability = 0.43 if regime == "source_train" else 0.55
    want_vertical = bool(rng.random() < clip_probability)
    want_horizontal = bool(rng.random() < clip_probability)
    active_y, opposite_y = _side_lengths(rng, want_vertical, regime)
    active_x, opposite_x = _side_lengths(rng, want_horizontal, regime)
    height = active_y + opposite_y + 1
    width = active_x + opposite_x + 1
    axis_row = active_y if upper else opposite_y
    axis_col = active_x if left else opposite_x
    near_y, far_y = _distance_interval(rng, active_y, opposite_y, want_vertical)
    near_x, far_x = _distance_interval(rng, active_x, opposite_x, want_horizontal)
    mode = int(rng.integers(5))
    cells = _pattern_cells(rng, (near_y, far_y), (near_x, far_x), mode)
    # Retain a near-axis witness and independently place the farthest row
    # and column witnesses. These are motif bounds, not quadrant corners.
    cells.update(((near_y, near_x),
                  (far_y, int(rng.integers(near_x, far_x + 1))),
                  (int(rng.integers(near_y, far_y + 1)), far_x)))
    colors = [c for c in color_order if c != background]
    if previous_axis is not None and previous_axis != background and rng.random() < 0.36:
        axis_color = previous_axis
    else:
        axis_color = int(rng.choice(colors))
    candidates = [c for c in colors if c != axis_color]
    accent_count = int(rng.integers(1, min(8, len(cells), len(candidates)) + 1))
    accents = [int(c) for c in rng.choice(candidates, size=accent_count, replace=False)]
    assignments = list(accents)
    assignments += [accents[int(rng.integers(accent_count))] for _ in range(len(cells) - accent_count)]
    rng.shuffle(assignments)
    grid = np.full((height, width), background, dtype=np.int8)
    grid[axis_row, :] = axis_color
    grid[:, axis_col] = axis_color
    for (dy, dx), color in zip(sorted(cells), assignments):
        y = axis_row - dy if upper else axis_row + dy
        x = axis_col - dx if left else axis_col + dx
        grid[y, x] = color
    info = _analyze_input(grid)
    assert _clipping(info) == (want_vertical, want_horizontal)
    assert _visible_both(info)
    return grid, _complete_reflection(grid, info), axis_color


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(seed, rng, num_train, num_test)
    color_order = tuple(int(c) for c in rng.permutation(10))
    shared_background = bool(rng.random() < 0.52)
    background_order = tuple(int(c) for c in rng.permutation(10))
    puzzle = {"train": [], "test": []}
    seen = set()
    previous_axis = None
    serial = 0
    for regime, count, section in (("source_train", num_train, "train"),
                                   ("target_test", num_test, "test")):
        for _ in range(count):
            background = background_order[0] if shared_background else background_order[serial % 10]
            # Independent child draws keep neighboring pairs distinct. The
            # bounded fallback only handles an exceedingly rare exact copy.
            for attempt in range(12):
                child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                inp, out, axis_color = _make_pair(child, regime, color_order,
                                                  background, previous_axis)
                if _grid_key(inp) not in seen and not _matches_source_input(inp):
                    break
            else:
                raise AssertionError("could not construct a unique pair")
            puzzle[section].append({"input": inp, "output": out})
            seen.add(_grid_key(inp))
            previous_axis = axis_color
            serial += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert not _matches_source_input(inp)
            info = _analyze_input(inp)
            assert 1 <= len(info["accent_colors"]) <= 8
            assert _visible_both(info)
            assert np.array_equal(out, _complete_reflection(inp, info))
            assert not np.array_equal(inp, out)
            assert np.array_equal(out[info["axis_row"], :], inp[info["axis_row"], :])
            assert np.array_equal(out[:, info["axis_col"]], inp[:, info["axis_col"]])
            assert np.array_equal(out[info["accent_mask"]], inp[info["accent_mask"]])
    # No hidden palette or direction convention is used: each input has a
    # locally identifiable cross, background, and single source quadrant.
    # Every training output includes a cell mirrored across both axes.


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[section]) == len(second[section]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[section], second[section])) for section in ("train", "test"))


def self_test() -> None:
    def independently_complete(inp, info):
        expected = inp.copy()
        upper = info["quadrant"] in (0, 1)
        left = info["quadrant"] in (0, 2)
        ay, ax = info["axis_row"], info["axis_col"]
        for y in range(info["height"]):
            if y == ay:
                continue
            source_y = ay - abs(y - ay) if upper else ay + abs(y - ay)
            if not 0 <= source_y < info["height"]:
                continue
            for x in range(info["width"]):
                if x == ax:
                    continue
                source_x = ax - abs(x - ax) if left else ax + abs(x - ax)
                if 0 <= source_x < info["width"]:
                    expected[y, x] = inp[source_y, source_x]
        return expected

    assert _puzzles_equal(generate_puzzle(314159, 6, 3), generate_puzzle(314159, 6, 3))
    assert not _puzzles_equal(generate_puzzle(100), generate_puzzle(101))
    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                            (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(5000 + offset, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}, {"num_train": True}):
        try:
            generate_puzzle(7000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {args}")
    train_counts, test_counts, quadrants, clip_modes = set(), set(), set(), set()
    shapes, layouts, palettes, modes_by_split = set(), set(), set(), {"train": set(), "test": set()}
    margins, backgrounds, color_counts, first_inputs = set(), set(), set(), set()
    distinct_per_puzzle = 0
    near_blank = far_blank = no_corner = no_margin = 0
    runs = blocks = repeated = small_seed = large_seed = 0
    square = tall = wide = 0
    shared_background = varied_background = simpler_test = largest_train = 0
    mixed_quadrants = mixed_clipping = axis_reuse = axis_change = 0
    margin_sides = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        first_inputs.add(_grid_key(puzzle["train"][0]["input"]))
        local = []
        local_infos = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                info = _analyze_input(pair["input"])
                assert np.array_equal(pair["output"], independently_complete(pair["input"], info))
                local_infos[section].append(info)
                quadrants.add(info["quadrant"])
                mode = _clipping(info)
                clip_modes.add(mode)
                modes_by_split[section].add(mode)
                shapes.add((info["height"], info["width"]))
                layouts.add((info["height"], info["width"], info["axis_row"],
                             info["axis_col"], info["quadrant"]))
                palettes.add((info["background"], info["axis_color"]))
                backgrounds.add(info["background"])
                color_counts.add(len(info["accent_colors"]))
                margins.add((info["axis_row"] - min(info["ys"]),
                             max(info["ys"]) - info["axis_row"],
                             info["axis_col"] - min(info["xs"]),
                             max(info["xs"]) - info["axis_col"]))
                upper = info["quadrant"] in (0, 1)
                left = info["quadrant"] in (0, 2)
                active_y = info["axis_row"] if upper else info["height"] - info["axis_row"] - 1
                active_x = info["axis_col"] if left else info["width"] - info["axis_col"] - 1
                opposite_y = info["height"] - info["axis_row"] - 1 if upper else info["axis_row"]
                opposite_x = info["width"] - info["axis_col"] - 1 if left else info["axis_col"]
                dy = np.abs(info["ys"] - info["axis_row"])
                dx = np.abs(info["xs"] - info["axis_col"])
                near_blank += bool(dy.min() > 1 or dx.min() > 1)
                far_blank += bool(dy.max() < active_y or dx.max() < active_x)
                no_margin += bool(dy.max() >= opposite_y and dx.max() >= opposite_x)
                if opposite_y > dy.max():
                    margin_sides.add("bottom" if upper else "top")
                if opposite_x > dx.max():
                    margin_sides.add("right" if left else "left")
                y_edge = (0, info["axis_row"] - 1) if upper else (info["axis_row"] + 1, info["height"] - 1)
                x_edge = (0, info["axis_col"] - 1) if left else (info["axis_col"] + 1, info["width"] - 1)
                no_corner += not any(info["accent_mask"][y, x] for y in y_edge for x in x_edge)
                accent = info["accent_mask"]
                blocks += bool(np.any(accent[:-1, :-1] & accent[1:, :-1] & accent[:-1, 1:] & accent[1:, 1:]))
                runs += bool(np.any(accent[:, :-2] & accent[:, 1:-1] & accent[:, 2:])
                             or np.any(accent[:-2, :] & accent[1:-1, :] & accent[2:, :]))
                repeated += len(info["ys"]) > len(info["accent_colors"])
                small_seed += len(info["ys"]) <= 6
                large_seed += len(info["ys"]) >= 12
                square += info["height"] == info["width"]
                tall += info["height"] > info["width"]
                wide += info["height"] < info["width"]
                local.append((info["quadrant"], mode, pair["input"].shape,
                              info["background"], len(info["accent_colors"])))
        distinct_per_puzzle += len(set(local)) > 1
        all_infos = local_infos["train"] + local_infos["test"]
        background_count = len({info["background"] for info in all_infos})
        shared_background += background_count == 1
        varied_background += background_count > 1
        simpler_test += max(len(info["accent_colors"]) for info in local_infos["test"]) < max(
            len(info["accent_colors"]) for info in local_infos["train"])
        largest_train += max(info["height"] * info["width"] for info in local_infos["train"]) > max(
            info["height"] * info["width"] for info in local_infos["test"])
        mixed_quadrants += len({info["quadrant"] for info in all_infos}) > 1
        mixed_clipping += len({_clipping(info) for info in all_infos}) > 1
        axis_reuse += len({info["axis_color"] for info in all_infos}) < len(all_infos)
        axis_change += len({info["axis_color"] for info in all_infos}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert quadrants == {0, 1, 2, 3}
    assert clip_modes == {(False, False), (False, True), (True, False), (True, True)}
    assert modes_by_split["test"] == clip_modes
    assert len(shapes) >= 70 and len(layouts) >= 120
    assert len(palettes) >= 30 and len(backgrounds) >= 8
    assert len(color_counts) >= 6 and len(margins) >= 50
    assert len(first_inputs) == 42 and distinct_per_puzzle >= 35
    assert min(near_blank, far_blank, no_corner, no_margin) >= 20
    assert margin_sides == {"top", "bottom", "left", "right"}
    assert min(runs, blocks, repeated, small_seed, large_seed) >= 15
    assert square >= 5 and tall >= 40 and wide >= 40
    assert min(shared_background, varied_background, simpler_test, largest_train) >= 8
    assert min(mixed_quadrants, mixed_clipping, axis_reuse, axis_change) >= 25
