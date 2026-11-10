"""Generate five-column histograms of eight-connected monochrome objects."""

import numpy as np


_SOURCE_PAIRS = {
    (
        ("7777777799", "7767777799", "7666777977", "7767779777",
         "7777777777", "7777777777", "2727777677", "7277776667",
         "2727777677", "7777777777"),
        ("77776", "72796"),
    ),
    (
        ("7777777755", "7777777775", "7755577777", "7757577777",
         "7755577775", "7777777777", "7777777877", "7777778887",
         "9977788888", "7777777777"),
        ("57777", "57777", "57897"),
    ),
    (
        ("8887778799", "8777777777", "8777779799", "7787779799",
         "7777777777", "8777722799", "7777772777", "8766677757",
         "7767677555", "7766677757"),
        ("77877", "77897", "77897", "77897", "52896"),
    ),
    (
        ("2272778722", "2277777777", "7772722788", "2227772787",
         "7777777787", "8888888887", "7777777787", "7767777887",
         "6666727777", "7767772227"),
        ("72777", "72777", "72777", "72777", "72877", "72876"),
    ),
}

_DIR8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
              if dr or dc)
_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _seed_value(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _count_argument(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _pair_counts(seed, num_train, num_test):
    train = _count_argument("num_train", num_train, 2, 10)
    test = _count_argument("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if test is None:
        test = 1 + seed % (min(4, train - 1) if train is not None else 4)
    if train is None:
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    return train, test


def _background(grid):
    values, frequencies = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(frequencies))])


def _components(grid, color, diagonals=True):
    rows, cols = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    directions = _DIR8 if diagonals else _DIR4
    for row in range(rows):
        for col in range(cols):
            if seen[row, col] or int(grid[row, col]) != color:
                continue
            seen[row, col] = True
            stack = [(row, col)]
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols and
                            not seen[nr, nc] and int(grid[nr, nc]) == color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(size)
    return result


def _counts(grid, order):
    return np.array([len(_components(grid, color)) for color in order], dtype=np.int64)


def _output(counts, background, order):
    height = int(max(counts))
    if height < 1:
        raise ValueError("at least one foreground object is required")
    result = np.full((height, 5), background, dtype=np.int8)
    for col, (color, count) in enumerate(zip(order, counts)):
        if count:
            result[height - int(count):, col] = color
    return result


def _normalize(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _shape(rng, size, kind=None):
    if size == 1:
        return ((0, 0),)
    if kind == "diagonal" or (size == 2 and rng.random() < 0.45):
        return ((0, 0), (1, 1))
    if kind == "ring":
        side = 3 if size == 8 else 4
        return _normalize({(r, c) for r in range(side) for c in range(side)
                           if r in (0, side - 1) or c in (0, side - 1)})
    if kind == "bridge" and size >= 8:
        cells = {(0, 0), (0, 1), (1, 0), (1, 1),
                 (2, 2), (2, 3), (3, 2), (3, 3)}
        extension = [(2, 4), (3, 4), (4, 2), (4, 3), (4, 4),
                     (3, 5), (4, 5), (5, 3), (5, 4), (5, 5)]
        cells.update(extension[:size - 8])
        return _normalize(cells)
    if size in (8, 12) and rng.random() < 0.18:
        side = 3 if size == 8 else 4
        return _normalize({(r, c) for r in range(side) for c in range(side)
                           if r in (0, side - 1) or c in (0, side - 1)})
    cells = {(0, 0)}
    style = int(rng.integers(0, 3))
    head = (0, 0)
    while len(cells) < size:
        anchor = head if style == 1 else tuple(cells)[int(rng.integers(0, len(cells)))]
        directions = _DIR4 if style == 0 else _DIR8
        dr, dc = directions[int(rng.integers(0, len(directions)))]
        new = (anchor[0] + dr, anchor[1] + dc)
        if new not in cells:
            cells.add(new)
            head = new
    return _normalize(cells)


def _orient(shape, rng):
    points = list(shape)
    if rng.random() < 0.5:
        points = [(r, -c) for r, c in points]
    for _ in range(int(rng.integers(0, 4))):
        points = [(c, -r) for r, c in points]
    return _normalize(points)


def _dimensions(rng, attempt, previous):
    if attempt >= 5:
        return (28 + int(rng.integers(0, 3)), 28 + int(rng.integers(0, 3)))
    for _ in range(12):
        aspect = int(rng.integers(0, 3))
        if aspect == 0:
            h = int(rng.integers(12, 25))
            w = max(11, min(29, h + int(rng.integers(-4, 5))))
        elif aspect == 1:
            h, w = int(rng.integers(11, 18)), int(rng.integers(19, 30))
        else:
            h, w = int(rng.integers(19, 30)), int(rng.integers(11, 18))
        if (h, w) not in previous:
            return h, w
    return h, w


def _choose_sizes(rng, counts, special):
    objects = []
    for index, count in enumerate(counts):
        for item in range(int(count)):
            hint = special.get((index, item))
            if hint is None:
                size = int(rng.choice((1, 1, 1, 2, 2, 3, 4, 5, 7, 9, 12)))
                kind = None
            else:
                size, kind = hint
            objects.append([index, size, kind, hint is not None])
    return objects


def _place(rng, shape, order, background, objects):
    height, width = shape
    grid = np.full(shape, background, dtype=np.int8)
    objects.sort(key=lambda obj: obj[1], reverse=True)
    for index, size, kind, _forced in objects:
        color = order[index]
        placed = False
        for attempt in range(320):
            if attempt % 32 == 0:
                motif = _orient(_shape(rng, size, kind), rng)
                mh = max(r for r, _ in motif) + 1
                mw = max(c for _, c in motif) + 1
            if mh > height or mw > width:
                continue
            top = int(rng.integers(0, height - mh + 1))
            left = int(rng.integers(0, width - mw + 1))
            cells = [(top + r, left + c) for r, c in motif]
            if any(grid[r, c] != background for r, c in cells):
                continue
            if any(grid[nr, nc] == color
                   for r, c in cells
                   for dr, dc in _DIR8
                   for nr, nc in ((r + dr, c + dc),)
                   if 0 <= nr < height and 0 <= nc < width):
                continue
            for r, c in cells:
                grid[r, c] = color
            placed = True
            break
        if not placed:
            return None
    return grid


def _plan_counts(rng, active, height, mode):
    counts = np.zeros(5, dtype=np.int64)
    leader = int(rng.choice(active))
    for index in active:
        counts[index] = int(rng.integers(1, height + 1))
    counts[leader] = height
    special = {}
    others = [index for index in active if index != leader]
    if mode == "tie" and others:
        counts[int(rng.choice(others))] = height
    if mode in ("inversion", "equal_area") and others and height >= 2:
        other = int(rng.choice(others))
        counts[other] = 1
        for item in range(height):
            special[(leader, item)] = (1, None)
        big = int(rng.integers(max(9, height + 3), 19)) if mode == "inversion" else height
        special[(other, 0)] = (big, "bridge" if big >= 8 else None)
    return counts, special


def _make_pair(rng, order, background, active, height, mode, previous_shapes,
               require_diagonal=False, large=False):
    counts, special = _plan_counts(rng, active, height, mode)
    if large and not any(size >= 9 for size, _ in special.values()):
        choices = [(index, 0) for index in active if (index, 0) not in special]
        if choices:
            special[choices[int(rng.integers(0, len(choices)))]] = (
                int(rng.integers(9, 19)), "bridge")
    if require_diagonal and not any(kind in ("diagonal", "bridge")
                                    for _, kind in special.values()):
        choices = [(index, item) for index in active
                   for item in range(int(counts[index]))
                   if (index, item) not in special]
        if choices:
            special[choices[int(rng.integers(0, len(choices)))]] = (2, "diagonal")
    if rng.random() < 0.35:
        choices = [(index, item) for index in active
                   for item in range(int(counts[index]))
                   if (index, item) not in special]
        if choices:
            special[choices[int(rng.integers(0, len(choices)))]] = (8, "ring")
    for attempt in range(8):
        shape = _dimensions(rng, attempt, previous_shapes)
        if shape[0] * shape[1] < 10 * int(np.sum(counts)):
            continue
        objects = _choose_sizes(rng, counts, special)
        maximum_ink = int(shape[0] * shape[1] * 0.38)
        while sum(obj[1] for obj in objects) > maximum_ink:
            shrinkable = [obj for obj in objects if not obj[3] and obj[1] > 1]
            if not shrinkable:
                break
            shrinkable[int(rng.integers(0, len(shrinkable)))][1] -= 1
        if sum(obj[1] for obj in objects) > maximum_ink:
            continue
        grid = _place(rng, shape, order, background, objects)
        if grid is None:
            continue
        actual = _counts(grid, order)
        if not np.array_equal(actual, counts):
            continue
        if require_diagonal and not _diagonal_evidence(grid, order):
            continue
        return {"input": grid, "output": _output(actual, background, order)}
    raise RuntimeError("could not place the requested components")


def _diagonal_evidence(grid, order):
    return any(len(_components(grid, color, False)) > len(_components(grid, color))
               for color in order)


def _has_ring(grid, background):
    for row in range(grid.shape[0] - 2):
        for col in range(grid.shape[1] - 2):
            block = grid[row:row + 3, col:col + 3]
            edge = (block[0, 0], block[0, 1], block[0, 2], block[1, 0],
                    block[1, 2], block[2, 0], block[2, 1], block[2, 2])
            if int(edge[0]) != background and all(x == edge[0] for x in edge):
                if block[1, 1] != edge[0]:
                    return True
    return False


def _training_subsets(rng, count):
    permutation = [int(x) for x in rng.permutation(5)]
    cut = int(rng.integers(2, 4))
    first = set(permutation[:cut])
    second = set(permutation[cut:])
    if rng.random() < 0.55:
        first.add(int(rng.choice(list(second))))
    if rng.random() < 0.55:
        second.add(int(rng.choice(list(first))))
    subsets = [first, second]
    while len(subsets) < count:
        size = int(rng.integers(1, 6))
        subsets.append(set(int(x) for x in rng.choice(5, size=size, replace=False)))
    rng.shuffle(subsets)
    return [tuple(sorted(s)) for s in subsets]


def _rows(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _signature(grid):
    return (grid.shape, grid.tobytes())


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _seed_value(seed)
    train_count, test_count = _pair_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Only this foreground-to-column convention is sampled for the whole puzzle.
    palette = [int(x) for x in rng.permutation(10)]
    order = tuple(palette[:5])
    backgrounds = tuple(palette[5:])
    subsets = _training_subsets(rng, train_count)
    low = int(rng.integers(2, 4))
    high = int(rng.integers(4, 7))
    heights = [low, high] + [int(rng.integers(1, 7)) for _ in range(train_count - 2)]
    rng.shuffle(heights)
    contrast_index = int(np.argmax(heights))
    if len(subsets[contrast_index]) < 2:
        subsets[contrast_index] = tuple(sorted(set(subsets[contrast_index]) |
                                               set(int(x) for x in rng.choice(5, 2, replace=False))))
    puzzle = {"train": [], "test": []}
    shapes = set()
    inputs = set()
    for i in range(train_count + test_count):
        training = i < train_count
        if training:
            active = subsets[i]
            height = heights[i]
            mode = "inversion" if i == contrast_index else (
                "tie" if rng.random() < 0.20 else "general")
        else:
            size = int(rng.integers(1, 6))
            active = tuple(sorted(int(x) for x in rng.choice(5, size=size, replace=False)))
            if rng.random() < 0.45:
                height = int(rng.integers(max(heights) + 1, min(10, max(heights) + 4)))
            else:
                height = int(rng.integers(1, max(heights) + 1))
            mode = ("general" if len(active) < 2 or height < 2 else
                    ("general", "tie", "inversion", "equal_area")[int(rng.integers(0, 4))])
        background = int(rng.choice(backgrounds))
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        for _ in range(5):
            pair = _make_pair(pair_rng, order, background, active, height, mode, shapes,
                              require_diagonal=training or pair_rng.random() < 0.7,
                              large=(not training and (height >= 6 or pair_rng.random() < 0.45)))
            signature = _signature(pair["input"])
            if signature in inputs or (_rows(pair["input"]), _rows(pair["output"])) in _SOURCE_PAIRS:
                continue
            puzzle["train" if training else "test"].append(pair)
            shapes.add(pair["input"].shape)
            inputs.add(signature)
            break
        else:
            raise RuntimeError("failed to make a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_order(train):
    columns = [set() for _ in range(5)]
    for pair in train:
        background = _background(pair["input"])
        output = pair["output"]
        for col in range(5):
            columns[col].update(int(x) for x in np.unique(output[:, col]) if int(x) != background)
    _require(all(len(colors) == 1 for colors in columns),
             "training must reveal every column color")
    order = tuple(next(iter(colors)) for colors in columns)
    _require(len(set(order)) == 5, "column colors must differ")
    return order


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    seen = set()
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair must have input and output")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                     np.issubdtype(grid.dtype, np.integer), "grid must be a 2D integer array")
            _require(all(1 <= size <= 30 for size in grid.shape) and
                     int(np.min(grid)) >= 0 and int(np.max(grid)) <= 9,
                     "invalid grid shape or colors")
        input_grid = pair["input"]
        key = _signature(input_grid)
        _require(key not in seen, "duplicate input")
        seen.add(key)
        _require((_rows(input_grid), _rows(pair["output"])) not in _SOURCE_PAIRS,
                 "source pair was copied")
    order = _infer_order(train)
    backgrounds = set()
    for pair in train + test:
        grid, output = pair["input"], pair["output"]
        background = _background(grid)
        backgrounds.add(background)
        ink = int(np.count_nonzero(grid != background))
        _require(ink > 0 and ink <= grid.size * 0.42,
                 "background must be unmistakable")
        _require(background not in order, "background conflicts with a category")
        _require(set(int(x) for x in np.unique(grid)) <= set(order) | {background},
                 "input contains an unknown foreground color")
        counts = _counts(grid, order)
        _require(np.array_equal(output, _output(counts, background, order)),
                 "incorrect object-count histogram")
        _require(output.shape[1] == 5, "output must have five columns")
    training_heights = [pair["output"].shape[0] for pair in train]
    _require(len(set(training_heights)) >= 2, "training must show variable height")
    _require(any(np.count_nonzero(_counts(pair["input"], order)) < 5 for pair in train),
             "training must show blank reserved columns")
    _require(any(len(set(int(x) for x in _counts(pair["input"], order) if x)) > 1
                 for pair in train), "training must demonstrate bottom alignment")
    _require(any(_diagonal_evidence(pair["input"], order) for pair in train),
             "training must demonstrate diagonal connectivity")
    _require(any(max(_components(pair["input"], color) or [0]) >= 8
                 for pair in train for color in order),
             "training must separate area from object count")
    for pair in test:
        _require(set(int(x) for x in np.unique(pair["input"])) - {_background(pair["input"])}
                 <= set(order), "test has an unseen category")


def _predict_from_training(train, test_input):
    """Reconstruct the visible column legend and count only the test input."""
    order = _infer_order(train)
    background = _background(test_input)
    predicted_counts = []
    for color in order:
        remaining = {(int(r), int(c)) for r, c in np.argwhere(test_input == color)}
        number = 0
        while remaining:
            number += 1
            stack = [remaining.pop()]
            while stack:
                r, c = stack.pop()
                for dr, dc in _DIR8:
                    point = (r + dr, c + dc)
                    if point in remaining:
                        remaining.remove(point)
                        stack.append(point)
        predicted_counts.append(number)
    height = max(predicted_counts)
    result = np.full((height, 5), background, dtype=np.int8)
    for col, count in enumerate(predicted_counts):
        result[height - count:, col] = order[col]
    return result


def self_test() -> None:
    explicit = generate_puzzle(314159, 7, 3)
    repeat = generate_puzzle(314159, 7, 3)
    _require(all(np.array_equal(a[key], b[key]) for regime in ("train", "test")
                 for a, b in zip(explicit[regime], repeat[regime])
                 for key in ("input", "output")), "same seed must reproduce the puzzle")
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(5100 + train_count + test_count, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit counts were ignored")
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(5200 + (train_count or 0) +
                                                   (test_count or 0), train_count, test_count))
    for index, kwargs in enumerate(({"num_train": 1}, {"num_train": 11},
                                     {"num_test": 0}, {"num_test": 5},
                                     {"num_train": 2, "num_test": 2},
                                     {"num_train": 2, "num_test": 4},
                                     {"num_train": 4, "num_test": 4})):
        try:
            generate_puzzle(5300 + index, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    train_counts, test_counts = set(), set()
    orders, sizes, backgrounds, shapes, signatures = set(), set(), set(), set(), set()
    test_active, test_heights, ties, inversions, equal_areas, large, diagonal = (
        set(), set(), 0, 0, 0, 0, 0)
    extrapolated = ordinary = below = varied_size_puzzles = varied_background_puzzles = 0
    edge_blanks = interior_blanks = large_and_high = sparse = crowded = rings = 0
    leaders, aspect_types = set(), set()
    for seed in range(4200, 4242):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        order = _infer_order(puzzle["train"])
        orders.add(order)
        local_shapes = set()
        local_backgrounds = set()
        maximum_train = max(pair["output"].shape[0] for pair in puzzle["train"])
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                grid = pair["input"]
                counts = _counts(grid, order)
                background = _background(grid)
                _require(np.array_equal(pair["output"],
                                        _predict_from_training(puzzle["train"], grid)),
                         "training-only prediction differs from ground truth")
                _require(np.array_equal(pair["output"], _output(counts, background, order)),
                         "independent count check failed")
                sizes.add(grid.shape)
                local_shapes.add(grid.shape)
                backgrounds.add(background)
                local_backgrounds.add(background)
                shapes.add((grid.shape, tuple(sorted(int(x) for x in counts))))
                signatures.add(_signature(grid))
                ratio = grid.shape[1] / grid.shape[0]
                aspect_types.add("wide" if ratio >= 1.4 else
                                 "tall" if ratio <= 1 / 1.4 else "near_square")
                if regime == "test":
                    test_active.add(int(np.count_nonzero(counts)))
                    test_heights.add(int(max(counts)))
                    areas = [int(np.count_nonzero(grid == color)) for color in order]
                    fill = sum(areas) / grid.size
                    sparse += fill <= 0.08
                    crowded += fill >= 0.18
                    rings += _has_ring(grid, background)
                    blanks = set(range(5)) - set(int(i) for i in np.flatnonzero(counts))
                    edge_blanks += bool(blanks & {0, 4})
                    interior_blanks += bool(blanks & {1, 2, 3})
                    leaders.update(int(i) for i in np.flatnonzero(counts == max(counts)))
                    if sum(int(c == max(counts)) for c in counts) >= 2:
                        ties += 1
                    if any(counts[a] > counts[b] and areas[a] < areas[b]
                           for a in range(5) for b in range(5)):
                        inversions += 1
                    if any(counts[a] != counts[b] and areas[a] == areas[b] and
                           areas[a] > 0 for a in range(5) for b in range(5)):
                        equal_areas += 1
                    if any(max(_components(grid, color) or [0]) >= 9 for color in order):
                        large += 1
                        if max(counts) >= 6:
                            large_and_high += 1
                    if _diagonal_evidence(grid, order):
                        diagonal += 1
                    if max(counts) > maximum_train:
                        extrapolated += 1
                    else:
                        ordinary += 1
                        below += max(counts) < maximum_train
        if len(local_shapes) >= 2:
            varied_size_puzzles += 1
        if len(local_backgrounds) >= 2:
            varied_background_puzzles += 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack variety")
    _require(len(orders) >= 30 and len(backgrounds) >= 8,
             "palette conventions lack variety")
    _require(len(sizes) >= 35 and varied_size_puzzles == 42,
             "canvas dimensions lack variety")
    _require(len(aspect_types) == 3 and varied_background_puzzles >= 30,
             "aspect ratios or within-puzzle backgrounds lack variety")
    _require(len(shapes) >= 90 and len(signatures) >= 150,
             "structural layouts lack variety")
    _require(test_active == {1, 2, 3, 4, 5} and len(test_heights) >= 7,
             "test subsets or count ranges lack variety")
    _require(ties >= 5 and inversions >= 5 and equal_areas >= 3,
             "test count/area contrasts are missing")
    _require(large >= 8 and diagonal >= 15,
             "large or diagonally joined test objects are missing")
    _require(large_and_high >= 5 and rings >= 5 and sparse >= 5 and crowded >= 5,
             "test shape, hole, or density combinations are missing")
    _require(edge_blanks >= 5 and interior_blanks >= 5 and leaders == set(range(5)),
             "empty columns or count leaders do not move")
    _require(extrapolated >= 8 and ordinary >= 8 and below >= 5,
             "test heights lack both regimes")
