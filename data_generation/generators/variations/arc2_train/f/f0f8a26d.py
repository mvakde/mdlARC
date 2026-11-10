"""Generate ARC puzzles in which each separate odd bar turns about its midpoint."""

import numpy as np


_SOURCE_INPUT_ROWS = (
    ("757777777", "757775777", "757577777", "777577577", "777577577",
     "777777577", "555557577", "777777577", "777777777"),
    ("777777777", "772227277", "777777277", "772777277", "772777777",
     "772777777", "777772227", "722277777", "777777777"),
    ("77777777777", "77777777777", "78777888887", "78787777777",
     "78777787777", "77777787888", "77777787777", "77777787777",
     "88888787777", "77777787888", "77777787777"),
    ("7777777777777", "7777777777777", "9999977999997", "7777777777777",
     "7777777777999", "9999999997777", "7777777779777", "7777779779777",
     "7777977779777", "7777977777777", "9997977797777", "7777977797999",
     "7777977797777"),
)
_SOURCE_INPUTS = tuple(np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
                       for rows in _SOURCE_INPUT_ROWS)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return int(value)


def _counts(rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        train = int(rng.integers(2, 11))
        test = int(rng.integers(1, min(4, train - 1) + 1))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _cells(row, col, length, orientation):
    half = length // 2
    if orientation == "H":
        return frozenset((row, c) for c in range(col - half, col + half + 1))
    if orientation == "V":
        return frozenset((r, col) for r in range(row - half, row + half + 1))
    return frozenset(((row, col),))


def _turned(row, col, length, orientation):
    return _cells(row, col, length, {"H": "V", "V": "H", "S": "S"}[orientation])


def _separate(cells, occupied):
    if cells & occupied:
        return False
    return all((r + dr, c + dc) not in occupied for r, c in cells
               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))


def _specs(rng, height, width, regime, index):
    cap = min(height, width, 11)
    options = [length for length in (1, 3, 5, 7, 9, 11) if length <= cap]
    area = height * width
    maximum = min(11 if regime == "target_test" else 9, max(3, area // 12))
    minimum = 3 if regime == "source_train" and index == 0 else 1
    count = int(rng.integers(minimum, maximum + 1))
    inventory = int(rng.integers(0, 6))
    orientation_mode = int(rng.integers(0, 5))
    one_direction = "H" if int(rng.integers(0, 2)) == 0 else "V"
    specs = []
    if regime == "source_train" and index == 0:
        specs = [(3 if cap < 5 else int(rng.choice((3, 5))), "H"),
                 (3 if cap < 5 else int(rng.choice((3, 5))), "V"), (1, "S")]
    weights = np.array((0.15, 0.32, 0.25, 0.15, 0.09, 0.04), dtype=float)
    if regime == "target_test":
        weights = np.array((0.11, 0.27, 0.25, 0.18, 0.13, 0.06), dtype=float)
    while len(specs) < count:
        if inventory == 0:  # all short, frequently without singletons
            length = 3
        elif inventory == 1:
            length = int(rng.choice([length for length in options if length >= 3]))
        else:
            probabilities = weights[:len(options)] / sum(weights[:len(options)])
            length = int(rng.choice(options, p=probabilities))
        orientation = "S" if length == 1 else (one_direction if orientation_mode == 0
                                                else ("H" if int(rng.integers(0, 2)) == 0 else "V"))
        specs.append((length, orientation))
    if all(orientation == "S" for _, orientation in specs):
        specs[-1] = (3, one_direction)
    return specs


def _dimensions(rng, regime):
    if regime == "source_train":
        height = int(rng.integers(8, 16))
        width = int(rng.integers(8, 16))
    elif int(rng.integers(0, 5)) < 2:
        height = int(rng.integers(13, 21))
        width = int(rng.integers(13, 21))
    else:
        height = int(rng.integers(7, 16))
        width = int(rng.integers(7, 16))
    if int(rng.integers(0, 4)) == 0:
        width = height
    return height, width


def _background(rng, conventions, regime, index):
    if conventions["background_mode"] == "fixed" or (regime == "source_train" and index == 0):
        return conventions["base_background"]
    if regime == "source_train" and index == 1:
        return int(rng.choice([color for color in range(10) if color != conventions["base_background"]]))
    return int(rng.integers(0, 10))


def _colors(rng, conventions, background, regime, index, training_colors, count):
    available = [color for color in range(10) if color != background]
    style = conventions["color_style"]
    if style == "shared":
        color = conventions["shared_color"]
        if color == background:
            # The shared-color convention excludes its foreground from the background pool.
            raise AssertionError("shared color became background")
        return [color] * count
    if style == "varied":
        if regime == "source_train":
            preferred = conventions["train_palette"][index % len(conventions["train_palette"])]
            color = preferred if preferred != background else available[(index + 1) % len(available)]
            if index > 0 and len(training_colors) < 9 and color in training_colors:
                color = next(candidate for candidate in available if candidate not in training_colors)
        elif int(rng.integers(0, 3)) == 0:
            novel = [color for color in available if color not in training_colors]
            color = int(rng.choice(novel if novel else available))
        else:
            reusable = [color for color in training_colors if color in available]
            color = int(rng.choice(reusable if reusable else available))
        return [color] * count
    # Multiple object colors are a puzzle-wide visible convention, first taught in training.
    number = min(count, int(rng.integers(2, 4)))
    if regime == "target_test" and int(rng.integers(0, 2)) == 0:
        reusable = [color for color in training_colors if color in available]
        palette = list(rng.choice(reusable if len(reusable) >= number else available,
                                  size=number, replace=False))
    else:
        palette = list(rng.choice(available, size=number, replace=False))
    colors = [int(palette[i % number]) for i in range(count)]
    rng.shuffle(colors)
    return colors


def _placement_weight(layout, height, width, row, col, input_cells, placed, overlap):
    edge = any(r in (0, height - 1) or c in (0, width - 1) for r, c in input_cells)
    if layout == "interior":
        return 3.0 if not edge else 0.12
    if layout == "edge":
        return 4.0 if edge else 0.45
    if layout == "aligned":
        return 5.0 if any(row == bar["row"] or col == bar["col"] for bar in placed) else 0.7
    if layout == "dense" and placed:
        distance = min(abs(row - bar["row"]) + abs(col - bar["col"]) for bar in placed)
        return 1.0 / (1.0 + distance / 4.0) * (4.0 if overlap else 1.0)
    if layout == "sparse" and placed:
        distance = min(abs(row - bar["row"]) + abs(col - bar["col"]) for bar in placed)
        return 0.4 + distance / 4.0
    return 1.0


def _place(rng, height, width, background, specs, colors, layout, collision_mode):
    order = sorted(range(len(specs)), key=lambda i: (-specs[i][0], float(rng.random())))
    placed = []
    occupied = set()
    output_colors = {}
    for index in order:
        length, orientation = specs[index]
        color = colors[index]
        half = length // 2
        candidates = []
        weights = []
        for row in range(half, height - half):
            for col in range(half, width - half):
                input_cells = _cells(row, col, length, orientation)
                if not _separate(input_cells, occupied):
                    continue
                output_cells = _turned(row, col, length, orientation)
                overlaps = output_cells & output_colors.keys()
                if (collision_mode == "clear" and overlaps) or len(overlaps) > 2:
                    continue
                if any(output_colors[cell] != color for cell in overlaps):
                    continue
                if any(not (bar["output_cells"] - output_cells -
                            set().union(*(other["output_cells"] for other in placed if other is not bar)))
                       for bar in placed):
                    continue
                weight = _placement_weight(layout, height, width, row, col,
                                           input_cells, placed, len(overlaps))
                if collision_mode == "mild" and overlaps:
                    weight *= 5.0
                candidates.append((row, col, input_cells, output_cells))
                weights.append(weight)
        if not candidates:
            return None
        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()
        row, col, input_cells, output_cells = candidates[int(rng.choice(len(candidates), p=probabilities))]
        placed.append({"row": row, "col": col, "length": length, "orientation": orientation,
                       "color": color, "input_cells": input_cells, "output_cells": output_cells})
        occupied.update(input_cells)
        for cell in output_cells:
            output_colors[cell] = color
    if len(occupied) >= height * width / 2:
        return None
    if any(not (bar["output_cells"] - set().union(*(other["output_cells"]
           for other in placed if other is not bar))) for bar in placed):
        return None
    source = np.full((height, width), background, dtype=np.int8)
    target = np.full((height, width), background, dtype=np.int8)
    for bar in placed:
        for row, col in bar["input_cells"]:
            source[row, col] = bar["color"]
        for row, col in bar["output_cells"]:
            target[row, col] = bar["color"]
    if np.array_equal(source, target):
        return None
    return source, target


def _pair(rng, regime, index, conventions, training_colors):
    for attempt in range(24):
        height, width = _dimensions(rng, regime)
        background = _background(rng, conventions, regime, index)
        if conventions["color_style"] == "shared" and background == conventions["shared_color"]:
            background = conventions["base_background"]
        specs = _specs(rng, height, width, regime, index)
        colors = _colors(rng, conventions, background, regime, index, training_colors, len(specs))
        layout = ("interior", "edge", "sparse", "dense", "aligned")[int(rng.integers(0, 5))]
        collision = "mild" if int(rng.integers(0, 4)) == 0 else "clear"
        pair = _place(rng, height, width, background, specs, colors, layout, collision)
        if pair is not None and not any(np.array_equal(pair[0], source) for source in _SOURCE_INPUTS):
            return {"input": pair[0], "output": pair[1]}
    raise RuntimeError("could not construct a separated bar layout")


def _array_key(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    base_background = int(rng.integers(0, 10))
    color_style = ("shared", "varied", "multi")[int(rng.choice(3, p=(0.25, 0.5, 0.25)))]
    available = [color for color in range(10) if color != base_background]
    conventions = {
        "base_background": base_background,
        "background_mode": "fixed" if int(rng.integers(0, 2)) == 0 else "vary",
        "color_style": color_style,
        "shared_color": int(rng.choice(available)),
        "train_palette": tuple(int(value) for value in rng.permutation(available)),
    }
    puzzle = {"train": [], "test": []}
    seen = set()
    training_colors = set()
    for regime, number in (("source_train", train_count), ("target_test", test_count)):
        section = "train" if regime == "source_train" else "test"
        for index in range(number):
            for _ in range(8):
                pair = _pair(rng, regime, index, conventions, training_colors)
                key = _array_key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[section].append(pair)
                    if section == "train":
                        background = _infer_background(pair["input"])
                        training_colors.update(int(color) for color in np.unique(pair["input"])
                                               if color != background)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    _check(counts[index] > grid.size / 2, "background must be uniquely dominant")
    return int(colors[index])


def _components(grid, background):
    remaining = {(int(r), int(c)) for r, c in np.argwhere(grid != background)}
    _check(bool(remaining), "input has no bars")
    components = []
    height, width = grid.shape
    while remaining:
        start = remaining.pop()
        stack = [start]
        cells = {start}
        while stack:
            row, col = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (row + dr, col + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    cells.add(neighbor)
                    stack.append(neighbor)
        rows = {row for row, _ in cells}
        cols = {col for _, col in cells}
        length = len(cells)
        _check(length % 2 == 1, "bar length must be odd")
        colors = {int(grid[row, col]) for row, col in cells}
        _check(len(colors) == 1, "each input bar has one color")
        if length == 1:
            orientation = "S"
            row, col = next(iter(cells))
        elif len(rows) == 1:
            orientation = "H"
            row = next(iter(rows))
            col = (min(cols) + max(cols)) // 2
        elif len(cols) == 1:
            orientation = "V"
            col = next(iter(cols))
            row = (min(rows) + max(rows)) // 2
        else:
            raise AssertionError("input component is not a straight bar")
        _check(cells == _cells(row, col, length, orientation), "input bar has gaps")
        half = length // 2
        _check(half <= row < height - half and half <= col < width - half,
               "rotated bar leaves the canvas")
        components.append((row, col, length, orientation, next(iter(colors))))
    return components


def _expected(grid, background, components):
    output = np.full(grid.shape, background, dtype=np.int8)
    colored = {}
    footprints = []
    for row, col, length, orientation, color in components:
        footprint = _turned(row, col, length, orientation)
        footprints.append(footprint)
        for cell in footprint:
            _check(cell not in colored or colored[cell] == color,
                   "different-colored output bars collide")
            colored[cell] = color
    for index, footprint in enumerate(footprints):
        others = set().union(*(other for j, other in enumerate(footprints) if j != index))
        _check(bool(footprint - others), "a rotated bar is completely hidden")
    for (row, col), color in colored.items():
        output[row, col] = color
    return output


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
           and len(puzzle["train"]) > len(puzzle["test"]), "pair counts")
    seen = set()
    records = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
                _check(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _check(all(1 <= dimension <= 30 for dimension in grid.shape), "grid dimensions")
                _check(int(grid.min()) >= 0 and int(grid.max()) <= 9, "ARC colors")
            _check(source.shape == target.shape and not np.array_equal(source, target), "dimension or identity")
            key = _array_key(source)
            _check(key not in seen, "duplicate input or pair")
            seen.add(key)
            _check(not any(np.array_equal(source, original) for original in _SOURCE_INPUTS), "source copy")
            background = _infer_background(source)
            components = _components(source, background)
            expected = _expected(source, background, components)
            _check(np.array_equal(target, expected), "wrong local rotation")
            _check(set(int(color) for color in np.unique(target)) ==
                   set(int(color) for color in np.unique(source)), "colors changed")
            records[section].append((background, components, source.shape))
    trained = [component for _, components, _ in records["train"] for component in components]
    _check(any(component[3] == "H" for component in trained) and
           any(component[3] == "V" for component in trained), "training misses a rotation direction")
    # A singleton is unchanged. It is explicitly taught whenever a test contains one.
    if any(component[3] == "S" for _, components, _ in records["test"] for component in components):
        _check(any(component[3] == "S" for component in trained), "singleton not taught")
    multi_test = any(len({component[4] for component in components}) > 1
                     for _, components, _ in records["test"])
    if multi_test:
        _check(any(len({component[4] for component in components}) > 1
                   for _, components, _ in records["train"]), "multiple colors not taught")


def _equal(first, second):
    return all(len(first[section]) == len(second[section]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[section], second[section])) for section in ("train", "test"))


def self_test() -> None:
    seen = set()
    train_counts, test_counts, shapes, backgrounds, lengths, bar_counts = set(), set(), set(), set(), set(), set()
    no_singleton = one_orientation = multicolor = interior = edge = overlap = 0
    smaller_tests = larger_tests = reused_colors = novel_colors = 0
    all_short = several_long = wide = tall = aligned = adjacent = 0
    all_threes_without_singleton = clear_short_train = 0
    test_count_not_larger = test_length_not_larger = 0
    within_shape = within_count = within_background = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _check(_equal(puzzle, generate_puzzle(seed)), "nondeterminism")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        seen.add(tuple(_array_key(pair["input"]) for section in ("train", "test")
                       for pair in puzzle[section]))
        train_sizes, train_bars, train_colors = [], [], set()
        train_lengths = []
        puzzle_shapes, puzzle_counts, puzzle_backgrounds = set(), set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                source, target = pair["input"], pair["output"]
                background = _infer_background(source)
                components = _components(source, background)
                footprints = [_turned(*component[:4]) for component in components]
                # This reconstruction uses the visible input, not generator metadata.
                _check(np.array_equal(target, _expected(source, background, components)), "rotation")
                shapes.add(source.shape)
                backgrounds.add(background)
                lengths.update(component[2] for component in components)
                bar_counts.add(len(components))
                puzzle_shapes.add(source.shape)
                puzzle_counts.add(len(components))
                puzzle_backgrounds.add(background)
                no_singleton += not any(component[3] == "S" for component in components)
                all_short += max(component[2] for component in components) <= 3
                several_long += sum(component[2] >= 5 for component in components) >= 2
                all_threes_without_singleton += all(component[2] == 3 for component in components)
                orientations = {component[3] for component in components if component[3] != "S"}
                one_orientation += len(orientations) == 1
                multicolor += len({component[4] for component in components}) > 1
                wide += source.shape[1] > source.shape[0]
                tall += source.shape[0] > source.shape[1]
                aligned += any(a[0] == b[0] or a[1] == b[1]
                               for i, a in enumerate(components) for b in components[i + 1:])
                adjacent += any(abs(r - rr) + abs(c - cc) == 1
                                for i, first in enumerate(footprints)
                                for second in footprints[i + 1:]
                                for r, c in first for rr, cc in second)
                edge += any(row in (0, source.shape[0]-1) or col in (0, source.shape[1]-1)
                            for row, col in np.argwhere(source != background))
                interior += not any(row in (0, source.shape[0]-1) or col in (0, source.shape[1]-1)
                                    for row, col in np.argwhere(source != background))
                overlap += sum(map(len, footprints)) > len(set().union(*footprints))
                if section == "train":
                    train_sizes.append(source.size)
                    train_bars.append(len(components))
                    train_lengths.append(max(component[2] for component in components))
                    train_colors.update(component[4] for component in components)
                    clear_short_train += (max(component[2] for component in components) <= 3
                                          and sum(map(len, footprints)) == len(set().union(*footprints)))
                else:
                    smaller_tests += source.size <= max(train_sizes) and len(components) <= max(train_bars)
                    larger_tests += source.size > max(train_sizes) or len(components) > max(train_bars)
                    test_count_not_larger += len(components) <= max(train_bars)
                    test_length_not_larger += max(component[2] for component in components) <= max(train_lengths)
                    test_colors = {component[4] for component in components}
                    reused_colors += bool(test_colors & train_colors)
                    novel_colors += bool(test_colors - train_colors)
        within_shape += len(puzzle_shapes) > 1
        within_count += len(puzzle_counts) > 1
        within_background += len(puzzle_backgrounds) > 1
    _check(len(seen) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _check(len(shapes) >= 30 and len(backgrounds) >= 6 and len(lengths) >= 5
           and len(bar_counts) >= 7, "structural diversity")
    _check(no_singleton >= 40 and one_orientation >= 25 and multicolor >= 10
           and interior >= 20 and edge >= 20 and 5 <= overlap < 100,
           "inventory, color, layout, or overlap diversity")
    _check(all_short >= 20 and several_long >= 20 and all_threes_without_singleton >= 10
           and clear_short_train >= 15 and wide >= 20 and tall >= 20
           and aligned >= 20 and adjacent >= 20, "structural variation")
    _check(smaller_tests >= 10 and larger_tests >= 10 and test_count_not_larger >= 10
           and test_length_not_larger >= 10 and reused_colors >= 10 and novel_colors >= 5,
           "test regime or color reuse variation")
    _check(within_shape >= 30 and within_count >= 30 and within_background >= 5,
           "within-puzzle variation")
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(101, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(101, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
