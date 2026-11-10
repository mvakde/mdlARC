"""Synthetic puzzles for the colored-level and nearest-perimeter ARC rule."""

import numpy as np


# Exact input configurations of the three original examples.  The dimensions
# and colors themselves remain available in new combinations.
_SOURCE_INPUTS = {
    (10, 10, 0, ((2, 6, 1), (7, 5, 4))),
    (10, 10, 0, ((2, 2, 6), (7, 7, 7))),
    (10, 10, 0, ((2, 4, 2), (7, 6, 8))),
}


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def _count(value, name: str):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train")
    test = _count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be 2..10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be 1..4")
    if train is None and test is None:
        test = 1 + int(seed) % 4
        train = test + 1 + (int(seed) // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _dimensions(rng, count, axis, forbidden_dimensions=None):
    minimum = max(5, 3 * count - 2)
    forbidden_dimensions = forbidden_dimensions or set()
    for _ in range(100):
        long_size = int(rng.integers(minimum, 31))
        cross_size = int(rng.integers(3, 31))
        shape = (long_size, cross_size) if axis == 0 else (cross_size, long_size)
        if shape not in forbidden_dimensions:
            return shape
    for long_size in range(minimum, 31):
        for cross_size in range(3, 31):
            shape = (long_size, cross_size) if axis == 0 else (cross_size, long_size)
            if shape not in forbidden_dimensions:
                return shape
    raise AssertionError("no legal shape")


def _levels(rng, length, count):
    if count == 1:
        return (int(rng.integers(1, length - 1)),)
    gaps = []
    remaining = length - 1
    for index in range(count - 1):
        remaining_gaps = count - index - 2
        choices = list(range(3, remaining - 3 * remaining_gaps + 1, 2))
        gap = choices[int(rng.integers(len(choices)))]
        gaps.append(gap)
        remaining -= gap
    # Boundary levels occur often enough to be learned, without becoming a
    # permanent convention.  All other margins are also possible.
    placement = int(rng.integers(4))
    if placement == 0:
        start = 0
    elif placement == 1:
        start = remaining
    else:
        start = int(rng.integers(remaining + 1))
    result = [start]
    for gap in gaps:
        result.append(result[-1] + gap)
    return tuple(result)


def _off_center(levels, length):
    if len(levels) != 2:
        return False
    return abs((levels[0] + levels[1] + 1) // 2 - length // 2) >= 2


def _geometry(rng, count, axis, forbidden=None, dimensions=None,
              forbidden_dimensions=None, off_center=False):
    forbidden = forbidden or set()
    for _ in range(256):
        shape = dimensions if dimensions is not None else _dimensions(
            rng, count, axis, forbidden_dimensions
        )
        length = shape[axis]
        if length < max(5, 3 * count - 2):
            continue
        levels = _levels(rng, length, count)
        geometry = (shape, levels)
        if geometry in forbidden or (off_center and not _off_center(levels, length)):
            continue
        return geometry
    # The normal path is stochastic, but a finite fallback makes requested
    # count boundaries deterministic and total.
    shapes = ([dimensions] if dimensions is not None else [
        ((a, b) if axis == 0 else (b, a))
        for a in range(max(5, 3 * count - 2), 31) for b in range(3, 31)
    ])
    for shape in shapes:
        if shape is None or shape in (forbidden_dimensions or set()):
            continue
        length = shape[axis]
        if length < max(5, 3 * count - 2):
            continue
        for start in range(length):
            for gap in range(3, length, 2):
                levels = tuple(start + gap * i for i in range(count))
                if levels[-1] >= length:
                    continue
                geometry = (shape, levels)
                if geometry not in forbidden and (not off_center or _off_center(levels, length)):
                    return geometry
    raise AssertionError("no legal geometry")


def _columns(rng, count, cross_size):
    if count == 1:
        return (int(rng.integers(cross_size)),)
    mode = int(rng.integers(5))
    if mode == 0:
        return (int(rng.integers(cross_size)),) * count
    if mode == 1:
        first = int(rng.integers(cross_size - 1))
        last = int(rng.integers(first + 1, cross_size))
    elif mode == 2:
        first = int(rng.integers(1, cross_size))
        last = int(rng.integers(first))
    elif mode == 3:
        first, last = 0, cross_size - 1
    else:
        first = int(rng.integers(cross_size))
        last = int(rng.integers(cross_size))
    middle = [int(rng.integers(cross_size)) for _ in range(count - 2)]
    return tuple([first] + middle + [last])


def _draw(shape, levels, columns, colors, background, axis):
    inp = np.full(shape, background, dtype=np.int8)
    out = np.full(shape, background, dtype=np.int8)
    for level, column, color in zip(levels, columns, colors):
        if axis == 0:
            inp[level, column] = color
            out[level, :] = color
        else:
            inp[column, level] = color
            out[:, level] = color
    for position in range(shape[axis]):
        nearest = min(range(len(levels)), key=lambda i: abs(position - levels[i]))
        color = colors[nearest]
        if axis == 0:
            out[position, 0] = color
            out[position, -1] = color
        else:
            out[0, position] = color
            out[-1, position] = color
    if axis == 0:
        out[0, :] = colors[0]
        out[-1, :] = colors[-1]
    else:
        out[:, 0] = colors[0]
        out[:, -1] = colors[-1]
    return {"input": inp, "output": out}


def _background_and_colors(rng, count, mode, palette, previous_background=None,
                           familiar=()):
    if mode == 0:
        background = 0
        choices = [color for color in palette if color != background]
    elif mode == 1:
        backgrounds = palette[:2]
        if previous_background is not None:
            backgrounds = [v for v in backgrounds if v != previous_background]
        background = int(backgrounds[int(rng.integers(len(backgrounds)))])
        choices = [color for color in palette if color != background]
    else:
        backgrounds = [v for v in range(10) if v != previous_background]
        background = backgrounds[int(rng.integers(len(backgrounds)))]
        choices = [v for v in range(10) if v != background]
    picked = [int(v) for v in rng.choice(choices, size=count, replace=False)]
    # Reuse and role exchange are possible in the same puzzle as fresh colors.
    usable_familiar = [int(v) for v in familiar if v != background]
    if usable_familiar and int(rng.integers(3)) == 0:
        preferred = usable_familiar[int(rng.integers(len(usable_familiar)))]
        if preferred in picked:
            earlier = picked.index(preferred)
            picked[earlier], picked[-1] = picked[-1], picked[earlier]
        else:
            picked[-1] = preferred
    return background, tuple(picked)


def _input_signature(grid):
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    markers = tuple(sorted(
        (int(r), int(c), int(grid[r, c]))
        for r, c in np.argwhere(grid != background)
    ))
    return (int(grid.shape[0]), int(grid.shape[1]), background, markers)


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    axis = int(rng.integers(2))  # one demonstrated convention per puzzle
    palette_mode = int(rng.integers(3))
    palette = [int(v) for v in rng.permutation(np.arange(10))[:7]]
    if palette_mode == 0:
        palette = [int(v) for v in rng.permutation(np.arange(1, 10))[:4]]
    profile = int(rng.integers(4))

    train_marker_counts = [2]
    if profile == 0:
        train_marker_counts.append(2)
    elif profile == 1 or (profile == 3 and train_count == 2):
        train_marker_counts.append(3)
    else:
        train_marker_counts.append(4)
    for index in range(2, train_count):
        if profile == 0:
            count = 2
        elif profile == 3 and index == 2:
            count = 1
        else:
            count = int(rng.choice([1, 2, 2, 2, 3, 3, 4]))
        train_marker_counts.append(count)

    train_geometries = []
    for index, count in enumerate(train_marker_counts):
        compatible = [g for g in train_geometries if len(g[1]) == count]
        if index >= 2 and compatible and int(rng.integers(5)) == 0:
            geometry = compatible[int(rng.integers(len(compatible)))]
        else:
            geometry = _geometry(
                rng, count, axis, forbidden=set(train_geometries),
                off_center=(index == 0)
            )
        train_geometries.append(geometry)

    test_geometries = []
    available_counts = sorted(set(train_marker_counts))
    for _ in range(test_count):
        count = 2 if int(rng.integers(5)) < 3 else available_counts[
            int(rng.integers(len(available_counts)))
        ]
        matching = [g for g in train_geometries if len(g[1]) == count]
        mode = int(rng.integers(5))  # exact, familiar size, or new size
        if mode == 0:
            geometry = matching[int(rng.integers(len(matching)))]
        elif mode in (1, 2):
            candidates = [g[0] for g in train_geometries
                          if g[0][axis] >= max(5, 3 * count - 2)]
            shape = candidates[int(rng.integers(len(candidates)))] if candidates else None
            try:
                geometry = _geometry(
                    rng, count, axis, forbidden=set(train_geometries + test_geometries),
                    dimensions=shape,
                    forbidden_dimensions={g[0] for g in train_geometries} if shape is None else None,
                )
            except AssertionError:
                geometry = _geometry(
                    rng, count, axis, forbidden=set(train_geometries + test_geometries),
                    forbidden_dimensions={g[0] for g in train_geometries},
                )
        else:
            geometry = _geometry(
                rng, count, axis, forbidden=set(train_geometries + test_geometries),
                forbidden_dimensions={g[0] for g in train_geometries},
            )
        test_geometries.append(geometry)

    seen_inputs = set()
    train, test = [], []
    familiar = []
    first_background = None
    for split, geometries in ((train, train_geometries), (test, test_geometries)):
        for index, (shape, levels) in enumerate(geometries):
            for attempt in range(80):
                previous = first_background if split is train and index == 1 and palette_mode else None
                background, marker_colors = _background_and_colors(
                    rng, len(levels), palette_mode, palette, previous, familiar
                )
                columns = _columns(rng, len(levels), shape[1 - axis])
                pair = _draw(shape, levels, columns, marker_colors, background, axis)
                key = _grid_key(pair["input"])
                if key not in seen_inputs and _input_signature(pair["input"]) not in _SOURCE_INPUTS:
                    break
            else:
                raise AssertionError("unable to produce a unique pair")
            seen_inputs.add(key)
            split.append(pair)
            familiar.extend(marker_colors)
            if split is train and index == 0:
                first_background = background

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_input(grid, axis):
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    points = np.argwhere(grid != background)
    if not 1 <= len(points) <= 4:
        raise ValueError("expected one to four singleton markers")
    markers = sorted((int(p[axis]), int(p[1 - axis]), int(grid[p[0], p[1]]))
                     for p in points)
    levels = tuple(m[0] for m in markers)
    marker_colors = tuple(m[2] for m in markers)
    if len(set(levels)) != len(levels) or len(set(marker_colors)) != len(markers):
        raise ValueError("marker levels and colors must be distinct")
    if any(b - a < 3 or (b - a) % 2 == 0 for a, b in zip(levels, levels[1:])):
        raise ValueError("consecutive marker gaps must be odd and at least three")
    if grid.shape[axis] < max(5, 3 * len(markers) - 2) or grid.shape[1-axis] < 3:
        raise ValueError("insufficient grid size")
    return background, levels, marker_colors


def _predict_cellwise(grid, axis):
    background, levels, colors = _inspect_input(grid, axis)
    result = np.empty_like(grid)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            position = (r, c)[axis]
            if position in levels or r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1):
                distances = [abs(position - level) for level in levels]
                if distances.count(min(distances)) != 1:
                    raise ValueError("ambiguous perimeter tie")
                result[r, c] = colors[distances.index(min(distances))]
            else:
                result[r, c] = background
    return result


def _inferred_axis(train):
    possible = []
    for axis in (0, 1):
        try:
            if all(np.array_equal(_predict_cellwise(pair["input"], axis), pair["output"])
                   for pair in train):
                possible.append(axis)
        except ValueError:
            pass
    if len(possible) != 1:
        raise AssertionError("training does not establish a unique expansion axis")
    return possible[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "expected train and test only")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")),
             "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    seen = set()
    for split in (train, test):
        for pair in split:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain input and output only")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be two-dimensional NumPy arrays")
                _require(all(1 <= n <= 30 for n in grid.shape), "grid size out of range")
                _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC color")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape and not np.array_equal(inp, out),
                     "invalid output size or unchanged pair")
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(_input_signature(inp) not in _SOURCE_INPUTS,
                     "source input copied")
    axis = _inferred_axis(train)
    train_geometries = set()
    train_counts = set()
    for pair in train:
        inp = pair["input"]
        background, levels, colors = _inspect_input(inp, axis)
        _require(len(set(colors + (background,))) == len(colors) + 1,
                 "background and marker colors overlap")
        train_geometries.add((inp.shape, levels))
        train_counts.add(len(levels))
    _require(len(train_geometries) >= 2, "training repeats one geometry")
    _require(2 in train_counts, "training needs an ordinary two-marker example")
    _require(any(len(_inspect_input(p["input"], axis)[1]) == 2 and
                 _off_center(_inspect_input(p["input"], axis)[1], p["input"].shape[axis])
                 for p in train), "training does not distinguish the border rule")
    for pair in train + test:
        try:
            background, levels, colors = _inspect_input(pair["input"], axis)
            prediction = _predict_cellwise(pair["input"], axis)
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
        _require(len(set(colors + (background,))) == len(colors) + 1,
                 "background and marker colors overlap")
        _require(np.array_equal(prediction, pair["output"]),
                 "output violates the nearest-level construction")
    for pair in test:
        count = len(_inspect_input(pair["input"], axis)[1])
        _require(count in train_counts, "test marker count not demonstrated")
    if any(count > 2 for count in train_counts):
        _require(any(len(_inspect_input(p["input"], axis)[1]) > 2 for p in train),
                 "multi-marker partition not demonstrated")


def _same_puzzle(first, second):
    return all(len(first[k]) == len(second[k]) and all(
        np.array_equal(a[role], b[role])
        for a, b in zip(first[k], second[k]) for role in ("input", "output")
    ) for k in ("train", "test"))


def _expect_value_error(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    _require(_same_puzzle(generate_puzzle(12345), generate_puzzle(12345)),
             "automatic generation is nondeterministic")
    _require(_same_puzzle(generate_puzzle(54321, 7, 3), generate_puzzle(54321, 7, 3)),
             "explicit generation is nondeterministic")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(7000 + train_count + test_count, train_count, test_count)
        validate_generated_puzzle(puzzle)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit count ignored")
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(8000 + (train_count or 0) + (test_count or 0),
                                 train_count, test_count)
        validate_generated_puzzle(puzzle)
        _require(train_count is None or len(puzzle["train"]) == train_count,
                 "partial train count ignored")
        _require(test_count is None or len(puzzle["test"]) == test_count,
                 "partial test count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (4, 4), (True, 1)):
        _expect_value_error(lambda a=args: generate_puzzle(1, a[0], a[1]))

    observations = []
    train_counts, test_counts, axes, shapes, geometries = set(), set(), set(), set(), set()
    marker_counts, backgrounds, gaps = set(), set(), set()
    aligned_train = separated_test = edge_levels = zero_markers = 0
    new_test_geometry = new_test_shape = same_shape_new_levels = repeat_test_geometry = 0
    variable_background_puzzles = multi_training_puzzles = off_center_puzzles = 0
    template_success = template_total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        axis = _inferred_axis(puzzle["train"])
        axes.add(axis)
        train_records = []
        for pair in puzzle["train"]:
            inp = pair["input"]
            bg, levels, colors = _inspect_input(inp, axis)
            points = np.argwhere(inp != bg)
            columns = tuple(int(p[1-axis]) for p in points)
            train_records.append((inp.shape, levels, colors, columns, bg, pair["output"]))
        train_geometry = {(r[0], r[1]) for r in train_records}
        train_shapes = {r[0] for r in train_records}
        variable_background_puzzles += len({r[4] for r in train_records}) > 1
        multi_training_puzzles += any(len(r[1]) > 2 for r in train_records)
        off_center_puzzles += any(len(r[1]) == 2 and _off_center(r[1], r[0][axis])
                                  for r in train_records)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg, levels, colors = _inspect_input(inp, axis)
                points = np.argwhere(inp != bg)
                columns = tuple(int(p[1-axis]) for p in points)
                shapes.add(inp.shape)
                geometries.add((axis, inp.shape, levels))
                marker_counts.add(len(levels))
                backgrounds.add(bg)
                gaps.update(b-a for a, b in zip(levels, levels[1:]))
                edge_levels += levels[0] == 0 or levels[-1] == inp.shape[axis]-1
                zero_markers += 0 in colors
                if split == "train":
                    aligned_train += len(set(columns)) == 1 and len(columns) > 1
                else:
                    separated_test += len(set(columns)) > 1
                    geometry = (inp.shape, levels)
                    new_test_geometry += geometry not in train_geometry
                    new_test_shape += inp.shape not in train_shapes
                    same_shape_new_levels += inp.shape in train_shapes and geometry not in train_geometry
                    repeat_test_geometry += geometry in train_geometry
                    # This deliberately weak shortcut copies a training output
                    # of the same dimensions and merely substitutes colors.
                    template_total += 1
                    for shape, old_levels, old_colors, _, old_bg, old_out in train_records:
                        if shape != inp.shape or len(old_colors) != len(colors):
                            continue
                        copied = old_out.copy()
                        for old, new in zip(old_colors, colors):
                            copied[old_out == old] = new
                        copied[old_out == old_bg] = bg
                        template_success += np.array_equal(copied, out)
                        break
                # Prediction uses only copied training demonstrations and the
                # queried input; it has no seed or test label access.
                inferred = _inferred_axis(puzzle["train"])
                _require(np.array_equal(_predict_cellwise(inp.copy(), inferred), out),
                         "independent cellwise prediction failed")
                observations.append((axis, inp.shape, levels, colors, columns, bg))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity collapsed")
    _require(axes == {0, 1}, "orientation diversity collapsed")
    _require(len(shapes) >= 60 and len(geometries) >= 150, "geometry diversity collapsed")
    _require(marker_counts == {1, 2, 3, 4}, "marker count diversity collapsed")
    _require(len(backgrounds) >= 8 and len(gaps) >= 6, "color or spacing collapsed")
    _require(aligned_train >= 5 and separated_test >= 5 and edge_levels >= 5,
             "column or edge variation collapsed")
    _require(zero_markers >= 5 and variable_background_puzzles >= 5,
             "background variation collapsed")
    _require(multi_training_puzzles >= 5 and off_center_puzzles == 42,
             "training does not demonstrate the expanded rule")
    _require(new_test_geometry >= 15 and new_test_shape >= 8 and
             same_shape_new_levels >= 5 and repeat_test_geometry >= 3,
             "test geometry regimes collapsed")
    _require(template_success < template_total // 2,
             "dimension-only output copying solves too many tests")
    _require(not _same_puzzle(generate_puzzle(0), generate_puzzle(1)),
             "different seeds produced the same puzzle")
