"""Synthetic puzzles for preserving the uniquely most frequent input color."""

import numpy as np


_SOURCE_PAIRS = {
    (((3, 3), (2, 2, 2, 8, 8, 2, 2, 2, 2)), ((3, 3), (2, 2, 2, 5, 5, 2, 2, 2, 2))),
    (((3, 3), (1, 1, 1, 8, 1, 3, 8, 2, 2)), ((3, 3), (1, 1, 1, 5, 1, 5, 5, 5, 5))),
    (((3, 3), (3, 3, 8, 4, 4, 4, 8, 1, 1)), ((3, 3), (5, 5, 5, 4, 4, 4, 5, 5, 5))),
    (((3, 3), (2, 2, 2, 2, 1, 8, 2, 8, 8)), ((3, 3), (2, 2, 2, 2, 5, 5, 2, 5, 5))),
    (((3, 3), (1, 3, 2, 3, 3, 2, 1, 3, 2)), ((3, 3), (5, 3, 5, 3, 3, 5, 5, 3, 5))),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid):
    return tuple(int(x) for x in grid.shape), tuple(int(x) for x in grid.flat)


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _checked_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _mode(grid):
    colors, counts = np.unique(grid, return_counts=True)
    maximum = int(counts.max())
    _require(int(np.count_nonzero(counts == maximum)) == 1, "input has tied modes")
    return int(colors[int(np.argmax(counts))])


def _output(grid, marker):
    mode = _mode(grid)
    return np.where(grid == mode, grid, marker).astype(np.int8)


def _shape(rng, regime, first=False):
    if first and float(rng.random()) < 0.45:
        return 3, 3
    p = float(rng.random())
    if p < (0.28 if regime == "target_test" else 0.24):
        return 3, 3
    if p < 0.38:
        side = int(rng.integers(4, 10))
        return side, side
    if p < 0.90:
        return int(rng.integers(3, 10)), int(rng.integers(3, 10))
    return int(rng.integers(9, 14)), int(rng.integers(9, 14))


def _bounded_composition(rng, total, parts, cap):
    _require(parts >= 1 and parts <= total <= parts * cap, "invalid composition")
    counts = np.ones(parts, dtype=np.int32)
    for _ in range(total - parts):
        eligible = np.flatnonzero(counts < cap)
        j = int(eligible[int(rng.integers(eligible.size))])
        counts[j] += 1
    return [int(x) for x in counts]


def _color_counts(rng, area, palette_size, regime, force_close=False):
    k = palette_size
    low = (area + k - 1 + k - 1) // k
    high = area - k + 1
    _require(low <= high, "shape cannot hold palette")
    style = "close" if force_close else ("close", "middle", "majority")[int(rng.integers(3))]
    if style == "close" and k >= 3:
        close_high = min(area // 2, (area - k + 3) // 2, high)
        if low <= close_high:
            top = int(rng.integers(low, close_high + 1))
            second = top - 1
            tail = _bounded_composition(rng, area - top - second, k - 2, top - 1)
            rng.shuffle(tail)
            return [top, second] + tail
    if style == "majority":
        majority_low = max(low, area // 2 + 1)
        if majority_low <= high:
            top = int(rng.integers(majority_low, high + 1))
            return [top] + _bounded_composition(rng, area - top, k - 1, top - 1)
    if k == 2 and style == "close":
        top = low
    else:
        top = int(rng.integers(low, high + 1))
    return [top] + _bounded_composition(rng, area - top, k - 1, top - 1)


def _arrange(rng, shape, colors, counts, layout):
    height, width = shape
    all_positions = [(r, c) for r in range(height) for c in range(width)]
    if layout == "random":
        order = [all_positions[int(j)] for j in rng.permutation(len(all_positions))]
    elif layout in ("rows", "columns"):
        order = all_positions if layout == "rows" else [(r, c) for c in range(width) for r in range(height)]
        if bool(rng.integers(2)):
            order = list(reversed(order))
    elif layout == "blocks":
        anchor = (int(rng.integers(height)), int(rng.integers(width)))
        jitter = rng.random((height, width))
        order = sorted(all_positions, key=lambda p: (abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]), float(jitter[p])))
    else:  # A minority island, then scatter the dominant around it.
        minority = 1 + int(np.argmax(counts[1:]))
        anchor = (int(rng.integers(height)), int(rng.integers(width)))
        jitter = rng.random((height, width))
        compact = sorted(all_positions, key=lambda p: (abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]), float(jitter[p])))
        island = compact[:counts[minority]]
        grid = np.full(shape, -1, dtype=np.int8)
        for p in island:
            grid[p] = colors[minority]
        rest = [p for p in all_positions if grid[p] < 0]
        rng.shuffle(rest)
        # Checkerboard preference breaks the mode into smaller patches.
        rest.sort(key=lambda p: (p[0] + p[1]) % 2)
        for p in rest[:counts[0]]:
            grid[p] = colors[0]
        remaining = rest[counts[0]:]
        other = [i for i in range(1, len(colors)) if i != minority]
        rng.shuffle(other)
        flat_colors = [colors[i] for i in other for _ in range(counts[i])]
        rng.shuffle(flat_colors)
        for p, color in zip(remaining, flat_colors):
            grid[p] = color
        _require(bool(np.all(grid >= 0)), "incomplete island layout")
        return grid
    grid = np.empty(shape, dtype=np.int8)
    start = 0
    for color, count in zip(colors, counts):
        for p in order[start:start + count]:
            grid[p] = color
        start += count
    return grid


def _make_pair(rng, marker, mode, palette, shape, regime, force_close=False):
    other = [c for c in palette if c != mode]
    rng.shuffle(other)
    colors = [mode] + other
    counts = _color_counts(rng, shape[0] * shape[1], len(colors), regime, force_close)
    layout = ("random", "rows", "columns", "blocks", "islands")[int(rng.integers(5))]
    grid = _arrange(rng, shape, colors, counts, layout)
    return {"input": grid, "output": _output(grid, marker)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    marker = int(rng.integers(10))
    available = [c for c in range(10) if c != marker]
    rng.shuffle(available)
    pool_size = int(rng.integers(3, min(7, train_count + 2) + 1))
    pool = available[:pool_size]
    first, second = pool[:2]
    extras = pool[2:]
    assignments = [[] for _ in range(train_count)]
    for i, color in enumerate(extras):
        assignments[i % train_count].append(color)
    training = []
    seen = set()
    for i in range(train_count):
        mode = first if i == 0 else second if i == 1 else int(pool[int(rng.integers(pool_size))])
        mandatory = [first, second] + assignments[i]
        if mode not in mandatory:
            mandatory.append(mode)
        optional = [c for c in pool if c not in mandatory]
        rng.shuffle(optional)
        # The first example is a compact, strict-plurality demonstration.
        if i == 0:
            palette = mandatory[:]
            if len(palette) < 3:
                palette.append(optional.pop())
            if optional and bool(rng.integers(2)):
                palette.append(optional.pop())
            shape = _shape(rng, "source_train", first=True)
            force_close = True
        else:
            shape = _shape(rng, "source_train")
            maximum = min(pool_size, shape[0] * shape[1] - 2)
            desired = int(rng.integers(len(mandatory), maximum + 1))
            palette = mandatory + optional[:desired - len(mandatory)]
            force_close = False
        for attempt in range(8):
            pair = _make_pair(rng, marker, mode, palette, shape, "source_train", force_close)
            key = _pair_key(pair)
            if key not in seen and key not in _SOURCE_PAIRS:
                break
        else:
            raise AssertionError("could not make distinct training pair")
        training.append(pair)
        seen.add(key)
    training_modes = {_mode(p["input"]) for p in training}
    promotable = [c for c in pool if c not in training_modes]
    tests = []
    for i in range(test_count):
        if i == 0 and test_count >= 2:
            mode = int(pool[int(rng.integers(pool_size))]) if not training_modes else int(sorted(training_modes)[int(rng.integers(len(training_modes)))])
        elif i == 1 and promotable:
            mode = int(promotable[int(rng.integers(len(promotable)))])
        elif promotable and bool(rng.integers(2)):
            mode = int(promotable[int(rng.integers(len(promotable)))])
        else:
            mode = int(sorted(training_modes)[int(rng.integers(len(training_modes)))])
        shape = _shape(rng, "target_test")
        maximum = min(pool_size, shape[0] * shape[1] - 2)
        palette_size = int(rng.integers(2, maximum + 1))
        optional = [c for c in pool if c != mode]
        rng.shuffle(optional)
        palette = [mode] + optional[:palette_size - 1]
        for attempt in range(8):
            pair = _make_pair(rng, marker, mode, palette, shape, "target_test")
            key = _pair_key(pair)
            if key not in seen and key not in _SOURCE_PAIRS:
                break
        else:
            raise AssertionError("could not make distinct test pair")
        tests.append(pair)
        seen.add(key)
    puzzle = {"train": training, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "invalid pair counts")
    markers = set()
    seen_inputs = set()
    train_modes = []
    train_colors = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be 2D ndarray")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                _require(all(1 <= n <= 30 for n in grid.shape), "invalid grid shape")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid color")
            _require(inp.shape == out.shape, "dimensions changed")
            key = _pair_key(pair)
            _require(key[0] not in seen_inputs, "duplicate input")
            _require(key not in _SOURCE_PAIRS, "source pair copied")
            seen_inputs.add(key[0])
            colors, counts = np.unique(inp, return_counts=True)
            _require(len(colors) >= 2, "no minority")
            maximum = int(counts.max())
            _require(int(np.count_nonzero(counts == maximum)) == 1, "tied modes")
            mode = int(colors[int(np.argmax(counts))])
            _require(bool(np.all(out[inp == mode] == mode)), "mode not preserved")
            changed = out[inp != mode]
            _require(changed.size > 0 and np.unique(changed).size == 1, "minorities lack one marker")
            marker = int(changed.flat[0])
            _require(marker not in colors, "marker appears in input")
            markers.add(marker)
            if split == "train":
                train_modes.append(mode)
                train_colors.append(set(int(x) for x in colors))
    _require(len(markers) == 1, "marker differs across puzzle")
    _require(len(set(train_modes)) >= 2, "training has only one mode")
    _require(any(2 * int(np.unique(p["input"], return_counts=True)[1].max()) <= p["input"].size for p in train), "training lacks a plurality example")
    _require(any(mode in colors and other_mode != mode for mode in set(train_modes) for colors, other_mode in zip(train_colors, train_modes)), "training lacks color role reversal")
    known_colors = set().union(*train_colors)
    for pair in test:
        _require(set(int(x) for x in np.unique(pair["input"])).issubset(known_colors), "test has undemonstrated input colors")


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(_pair_key(x) == _pair_key(y) for x, y in zip(a[s], b[s])) for s in ("train", "test"))


def _component_sizes(mask):
    remaining = set(zip(*np.nonzero(mask)))
    sizes = []
    while remaining:
        pending = [remaining.pop()]
        size = 0
        while pending:
            row, col = pending.pop()
            size += 1
            for point in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if point in remaining:
                    remaining.remove(point)
                    pending.append(point)
        sizes.append(size)
    return sizes


def _infer_from_training(training, test_input):
    markers = set()
    for pair in training:
        inp, out = pair["input"], pair["output"]
        mode = _mode(inp)
        markers.update(int(x) for x in np.unique(out[inp != mode]))
    _require(len(markers) == 1, "training cannot identify one marker")
    marker = markers.pop()
    mode = _mode(test_input)
    return np.where(test_input == mode, test_input, marker)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes = {s: set() for s in ("train", "test")}
    palettes = {s: set() for s in ("train", "test")}
    modes = {s: set() for s in ("train", "test")}
    margins = {s: set() for s in ("train", "test")}
    markers = set()
    structures = set()
    train_counts, test_counts = set(), set()
    overlap = promoted = repeated_mode = 0
    close_two = close_multi = majority = 0
    rows = columns = blocks = larger_minority_patch = 0
    close_three_by_three = 0
    first_shapes = set()
    first_count_patterns = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        marker = int(puzzle["train"][0]["output"][puzzle["train"][0]["input"] != _mode(puzzle["train"][0]["input"])][0])
        markers.add(marker)
        train_mode_set = {_mode(p["input"]) for p in puzzle["train"]}
        train_minority_set = set().union(*(set(int(x) for x in np.unique(p["input"])) - {_mode(p["input"])} for p in puzzle["train"]))
        train_shapes = {p["input"].shape for p in puzzle["train"]}
        first_input = puzzle["train"][0]["input"]
        first_shapes.add(first_input.shape)
        first_count_patterns.add(tuple(sorted((int(x) for x in np.unique(first_input, return_counts=True)[1]), reverse=True)))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                colors, counts = np.unique(inp, return_counts=True)
                order = np.sort(counts)[::-1]
                mode = int(colors[int(np.argmax(counts))])
                _require(np.array_equal(out, np.where(inp == mode, inp, marker)), "independent transform mismatch")
                shapes[split].add(tuple(inp.shape))
                palettes[split].add(len(colors))
                modes[split].add(mode)
                margins[split].add(int(order[0] - order[1]))
                relabel = {}
                signature = []
                for x in inp.flat:
                    value = int(x)
                    if value not in relabel:
                        relabel[value] = len(relabel)
                    signature.append(relabel[value])
                structures.add((tuple(inp.shape), tuple(signature)))
                if any(np.all(inp[r, :] == inp[r, 0]) for r in range(inp.shape[0])):
                    rows += 1
                if any(np.all(inp[:, c] == inp[0, c]) for c in range(inp.shape[1])):
                    columns += 1
                if any(np.all(inp[r:r+2, c:c+2] == inp[r, c]) for r in range(inp.shape[0]-1) for c in range(inp.shape[1]-1)):
                    blocks += 1
                mode_parts = _component_sizes(inp == mode)
                minority_parts = [max(_component_sizes(inp == int(color)), default=0) for color in colors if int(color) != mode]
                if len(mode_parts) >= 2 and max(minority_parts) > max(mode_parts):
                    larger_minority_patch += 1
                if split == "test":
                    only_train = [{"input": q["input"].copy(), "output": q["output"].copy()} for q in puzzle["train"]]
                    predicted = _infer_from_training(only_train, inp.copy())
                    _require(np.array_equal(predicted, out), "training-only test prediction failed")
                    if int(order[0]) > inp.size // 2:
                        majority += 1
                    if int(order[0] - order[1]) == 1 and len(colors) == 2:
                        close_two += 1
                        if inp.shape == (3, 3):
                            close_three_by_three += 1
                    if int(order[0] - order[1]) == 1 and len(colors) >= 3:
                        close_multi += 1
                    if mode in train_mode_set:
                        repeated_mode += 1
                    if mode in train_minority_set - train_mode_set:
                        promoted += 1
        if any(p["input"].shape in train_shapes for p in puzzle["test"]):
            overlap += 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity collapsed")
    _require(len(markers) >= 7, "marker diversity collapsed")
    _require(all(len(shapes[s]) >= 12 for s in shapes), "shape diversity collapsed")
    _require(all({2, 3, 4}.issubset(palettes[s]) for s in palettes), "palette diversity collapsed")
    _require(all(len(modes[s]) >= 7 for s in modes), "color roles collapsed")
    _require(all(len(margins[s]) >= 5 for s in margins), "frequency margins collapsed")
    _require(len(structures) >= 200, "structure diversity collapsed")
    _require(len(first_shapes) >= 8 and len(first_count_patterns) >= 8, "first training pattern collapsed")
    _require(rows >= 10 and columns >= 10 and blocks >= 20 and larger_minority_patch >= 5, "spatial arrangement diversity collapsed")
    _require(overlap >= 6 and repeated_mode >= 8 and promoted >= 5, "train/test color or size shift collapsed")
    _require(close_two >= 2 and close_multi >= 3 and majority >= 5 and close_three_by_three >= 1, "test frequency spectrum collapsed")
    for i, (tr, te) in enumerate(((2, 1), (10, 1), (10, 4), (6, 2))):
        p = generate_puzzle(1000 + i, tr, te)
        _require((len(p["train"]), len(p["test"])) == (tr, te), "explicit counts ignored")
    for i, (tr, te) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + i, tr, te)
        _require((tr is None or len(p["train"]) == tr) and (te is None or len(p["test"]) == te), "partial counts ignored")
    _require(_equal_puzzles(generate_puzzle(987654, 7, 3), generate_puzzle(987654, 7, 3)), "nondeterministic explicit puzzle")
    _require(_equal_puzzles(generate_puzzle(314159), generate_puzzle(314159)), "nondeterministic automatic puzzle")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 2, "num_test": 4}, {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(123, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {kwargs}")
