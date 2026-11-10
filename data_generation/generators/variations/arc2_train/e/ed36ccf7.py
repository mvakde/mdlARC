import numpy as np


_SOURCE_INPUTS = (
    (0, 0, 9, 0, 0, 9, 9, 9, 9),
    (6, 6, 6, 0, 0, 0, 6, 6, 0),
    (2, 0, 2, 0, 0, 2, 0, 2, 2),
    (9, 0, 0, 9, 9, 9, 9, 9, 9),
    (0, 0, 0, 5, 0, 0, 0, 5, 5),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _source_keys():
    result = set()
    for flat in _SOURCE_INPUTS:
        inp = np.array(flat, dtype=np.int8).reshape(3, 3)
        result.add((_key(inp), _key(np.rot90(inp))))
    return result


def _dihedral(grid):
    for turns in range(4):
        rotated = np.rot90(grid, turns)
        yield rotated
        yield np.fliplr(rotated)


def _family(mask):
    return min(_key(item) for item in _dihedral(mask))


def _direction_revealing(grid):
    target = np.rot90(grid)
    alternatives = [grid, np.rot90(grid, 2), np.rot90(grid, 3)]
    alternatives.extend(np.fliplr(np.rot90(grid, turns)) for turns in range(4))
    return all(not np.array_equal(target, other) for other in alternatives)


def _connected(mask):
    points = set(zip(*np.nonzero(mask)))
    if not points:
        return False
    stack = [next(iter(points))]
    visited = set(stack)
    while stack:
        r, c = stack.pop()
        for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if neighbor in points and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
    return len(visited) == len(points)


def _validate_count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _validate_count("num_train", num_train, 2, 10)
    test = _validate_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sizes(rng, train_count, test_count):
    mode = int(rng.choice(4, p=[0.40, 0.29, 0.17, 0.14]))
    if mode == 0:
        return [3] * train_count, [3] * test_count
    if mode == 1:
        train = [int(rng.choice((3, 4, 5))) for _ in range(train_count)]
        test = [int(rng.choice((3, 4, 5))) for _ in range(test_count)]
        train[0] = 3
        train[1] = int(rng.choice((4, 5)))
        return train, test
    if mode == 2:
        return [3] * train_count, [int(rng.choice((4, 5))) for _ in range(test_count)]
    return ([int(rng.choice((4, 5))) for _ in range(train_count)],
            [int(rng.choice((4, 5))) for _ in range(test_count)])


def _count_in_band(rng, size, band):
    total = size * size
    if band == "sparse":
        lower, upper = 1, max(2, int(round(total * 0.34)))
    elif band == "medium":
        lower, upper = max(2, int(round(total * 0.34))), min(total - 2, int(round(total * 0.68)))
    else:
        lower, upper = min(total - 2, int(round(total * 0.64))), total - 1
    return int(rng.integers(lower, upper + 1))


def _density_counts(rng, train_sizes, test_sizes):
    mode = int(rng.choice(4, p=[0.22, 0.19, 0.27, 0.32]))
    train = []
    test = []
    for index, size in enumerate(train_sizes):
        if mode == 0:
            count = (5 if index % 2 == 0 else 7) if size == 3 else _count_in_band(rng, size, "dense")
        elif mode == 1:
            if index == 0:
                count = 3 if size == 3 else max(3, int(round(size * size * 0.27)))
            else:
                count = _count_in_band(rng, size, "sparse")
        elif mode == 2:
            count = _count_in_band(rng, size, ("medium", "sparse", "dense")[index % 3])
        else:
            count = int(rng.integers(1, size * size))
        train.append(count)
    for index, size in enumerate(test_sizes):
        if mode == 0:
            count = 3 if size == 3 else _count_in_band(rng, size, "sparse")
        elif mode == 1:
            count = _count_in_band(rng, size, "dense")
        elif mode == 2:
            count = _count_in_band(rng, size, ("medium", "dense", "sparse")[index % 3])
        else:
            count = int(rng.integers(1, size * size))
        test.append(count)
    return train, test


def _connected_random_mask(rng, size, count):
    start = (int(rng.integers(size)), int(rng.integers(size)))
    points = {start}
    while len(points) < count:
        frontier = set()
        for r, c in points:
            for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= point[0] < size and 0 <= point[1] < size and point not in points:
                    frontier.add(point)
        points.add(sorted(frontier)[int(rng.integers(len(frontier)))])
    mask = np.zeros((size, size), dtype=np.int8)
    for r, c in points:
        mask[r, c] = 1
    return mask


def _rotational_masks(size):
    remaining = {(r, c) for r in range(size) for c in range(size)}
    orbits = []
    while remaining:
        r, c = min(remaining)
        orbit = {(r, c), (size - 1 - c, r), (size - 1 - r, size - 1 - c), (c, size - 1 - r)}
        remaining -= orbit
        orbits.append(orbit)
    for code in range(1, (1 << len(orbits)) - 1):
        mask = np.zeros((size, size), dtype=np.int8)
        for index, orbit in enumerate(orbits):
            if code & (1 << index):
                for r, c in orbit:
                    mask[r, c] = 1
        yield mask


def _mask_candidates(rng, size, count, unchanged):
    if unchanged:
        masks = list(_rotational_masks(size))
        order = rng.permutation(len(masks))
        return sorted((masks[int(i)] for i in order), key=lambda mask: abs(int(mask.sum()) - count))
    if size == 3:
        masks = []
        for code in range(1, 511):
            if code.bit_count() == count:
                masks.append(np.array([(code >> bit) & 1 for bit in range(9)], dtype=np.int8).reshape(3, 3))
        return [masks[int(i)] for i in rng.permutation(len(masks))]
    masks = []
    for index in range(240):
        if index % 3 == 0:
            masks.append(_connected_random_mask(rng, size, count))
        else:
            mask = np.zeros(size * size, dtype=np.int8)
            mask[rng.choice(size * size, size=count, replace=False)] = 1
            masks.append(mask.reshape(size, size))
    return masks


def _choose_mask(rng, size, count, used_families, require_direction, unchanged):
    if size == 3 and not require_direction and not unchanged and rng.random() < 0.6:
        motifs = {
            3: [((1, 1, 1), (0, 0, 0), (0, 0, 0))],
            5: [((1, 1, 1), (0, 1, 0), (0, 1, 0)),
                ((1, 1, 1), (1, 0, 0), (1, 0, 0))],
        }
        options = motifs.get(count, ())
        if options:
            order = rng.permutation(len(options))
            for index in order:
                mask = np.array(options[int(index)], dtype=np.int8)
                if _family(mask) not in used_families:
                    return np.rot90(mask, int(rng.integers(4))).copy()
    if require_direction:
        count_options = [count] + [c for c in range(1, size * size) if c != count]
        count_options.sort(key=lambda c: abs(c - count))
    else:
        count_options = [count]
    want_connected = bool(rng.integers(2))
    want_reflection = rng.random() < 0.28
    for target_count in count_options:
        candidates = _mask_candidates(rng, size, target_count, unchanged)
        best = None
        best_score = -1
        for mask in candidates:
            if require_direction and not _direction_revealing(mask):
                continue
            family = _family(mask)
            reflective = any(np.array_equal(mask, np.fliplr(np.rot90(mask, k))) for k in range(4))
            score = (8 if family not in used_families else 0)
            score += 2 if _connected(mask) == want_connected else 0
            score += 1 if reflective == want_reflection else 0
            if score > best_score:
                best, best_score = mask, score
                if score == 11:
                    break
        if best is not None:
            return best
    raise RuntimeError("No usable binary pattern")


def _colorize(rng, mask, background, primary, palette, multicolor):
    grid = np.full(mask.shape, background, dtype=np.int8)
    positions = np.argwhere(mask != 0)
    if multicolor and len(positions) >= 2:
        extras = [int(c) for c in palette if int(c) not in (background, primary)]
        rng.shuffle(extras)
        number = min(len(positions), 2 + int(rng.integers(2)), 1 + len(extras))
        colors = [primary] + extras[:number - 1]
        rng.shuffle(positions)
        for index, (r, c) in enumerate(positions):
            value = colors[index] if index < len(colors) else colors[int(rng.integers(len(colors)))]
            grid[r, c] = value
    else:
        grid[mask != 0] = primary
    return grid


def _make_pair(rng, size, desired_count, background, primary, palette, multicolor,
               used_inputs, used_families, source_keys, require_direction, unchanged):
    if multicolor:
        desired_count = max(2, desired_count)
    for attempt in range(18):
        mask = _choose_mask(rng, size, desired_count, used_families,
                            require_direction, unchanged)
        grid = _colorize(rng, mask, background, primary, palette, multicolor and not unchanged)
        output = np.rot90(grid).copy()
        key = _key(grid)
        if key in used_inputs or (key, _key(output)) in source_keys:
            used_families.add(_family(mask))
            continue
        used_inputs.add(key)
        used_families.add(_family(mask))
        return {"input": grid, "output": output}
    raise RuntimeError("Unable to construct a distinct non-source pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    train_sizes, test_sizes = _sizes(rng, train_count, test_count)
    train_counts, test_counts = _density_counts(rng, train_sizes, test_sizes)
    train_pool = [0] + [int(x) for x in rng.choice(np.arange(1, 10), size=4, replace=False)]
    rng.shuffle(train_pool)
    backgrounds = [train_pool[int(i) % len(train_pool)] for i in rng.permutation(train_count)]
    multi_mode = rng.random() < 0.32
    multi_train = [multi_mode and (i == 0 or rng.random() < 0.38) for i in range(train_count)]
    multi_test = [multi_mode and (i == 0 or rng.random() < 0.38) for i in range(test_count)]
    ambiguous_slots = set(int(x) for x in rng.choice(
        np.arange(1, train_count), size=min((train_count - 1) // 3, train_count - 1), replace=False))
    used_inputs, used_families = set(), set()
    source_keys = _source_keys()
    train, test = [], []
    train_primaries = []
    for index, (size, count) in enumerate(zip(train_sizes, train_counts)):
        background = backgrounds[index]
        primary = int(rng.choice([color for color in train_pool if color != background]))
        pair = _make_pair(rng, size, count, background, primary, train_pool,
                          multi_train[index], used_inputs, used_families, source_keys,
                          require_direction=index not in ambiguous_slots, unchanged=False)
        train.append(pair)
        train_primaries.append(primary)
    train_colors = {int(color) for pair in train for color in np.unique(pair["input"])}
    for index, (size, count) in enumerate(zip(test_sizes, test_counts)):
        held_out = (index % 2 == 1) if test_count > 1 else bool(rng.integers(2))
        if held_out:
            primary = int(rng.choice([color for color in range(10) if color not in train_colors]))
        else:
            primary = int(rng.choice(train_primaries))
        background = int(rng.choice([color for color in train_pool if color != primary]))
        unchanged = rng.random() < 0.12 and not multi_test[index]
        pair = _make_pair(rng, size, count, background, primary, train_pool,
                          multi_test[index], used_inputs, used_families, source_keys,
                          require_direction=False, unchanged=unchanged)
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be an ndarray")
    _require(grid.ndim == 2, f"{label} must be two dimensional")
    _require(all(1 <= side <= 30 for side in grid.shape), f"{label} dimensions are invalid")
    _require(np.issubdtype(grid.dtype, np.integer) and grid.dtype != np.bool_,
             f"{label} must have an integer dtype")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} has an invalid color")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[part], list) for part in ("train", "test")), "Invalid pair lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "Invalid pair counts")
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} has invalid keys")
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp, label + ".input")
            _validate_grid(out, label + ".output")
            _require(inp.shape == out.shape and inp.shape[0] == inp.shape[1]
                     and inp.shape[0] in (3, 4, 5), f"{label} has invalid size")
            _require(2 <= np.unique(inp).size <= 4, f"{label} has invalid palette")
            _require(np.array_equal(out, np.rot90(inp)), f"{label} violates CCW rotation")
            input_key, pair_key = _key(inp), (_key(inp), _key(out))
            _require(input_key not in inputs and pair_key not in pairs, f"{label} is duplicated")
            _require(pair_key not in _source_keys(), f"{label} copies a source pair")
            inputs.add(input_key)
            pairs.add(pair_key)
    _require(any(_direction_revealing(pair["input"]) for pair in train),
             "Training never reveals the rotation direction")
    ambiguous = sum(not _direction_revealing(pair["input"]) for pair in train)
    _require(ambiguous <= (len(train) - 1) // 3, "Too many ambiguous training pairs")
    if any(np.unique(pair["input"]).size > 2 for pair in test):
        _require(any(np.unique(pair["input"]).size > 2 for pair in train),
                 "Test multicolor case lacks a training demonstration")


def _signature(puzzle, structural=False):
    entries = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            grid = pair["input"]
            if structural:
                colors = {int(color): index for index, color in enumerate(np.unique(grid))}
                grid = np.array([colors[int(color)] for color in grid.flat], dtype=np.int8).reshape(grid.shape)
            entries.append(_key(grid))
    return tuple(entries)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                independent = np.empty_like(inp)
                size = len(inp)
                for r in range(size):
                    for c in range(size):
                        independent[size - 1 - c, r] = inp[r, c]
                _require(np.array_equal(out, independent), "Coordinate rotation mismatch")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "Too few train counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "Too few test counts")
    _require(_signature(puzzles[17]) == _signature(generate_puzzle(17)), "Nondeterministic result")
    _require(len({_signature(p) for p in puzzles}) >= 40, "Too few distinct puzzles")
    _require(len({_signature(p, True) for p in puzzles}) >= 38, "Too few structural puzzles")
    for counts in ((2, 1), (10, 1), (10, 4), (7, 3), (None, 1),
                   (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(7401, *counts)
        validate_generated_puzzle(puzzle)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            _require(requested is None or actual == requested, "Requested count ignored")
        _require(_signature(puzzle) == _signature(generate_puzzle(7401, *counts)),
                 "Explicit count call is nondeterministic")
    for counts in ((1, 1), (11, 1), (2, 0), (2, 5), (2, 2), (4, 4)):
        try:
            generate_puzzle(7401, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count request accepted")
    grids = [pair["input"] for p in puzzles for split in ("train", "test") for pair in p[split]]
    _require({g.shape[0] for g in grids} == {3, 4, 5}, "Size regimes missing")
    _require({len(np.unique(g)) for g in grids} == {2, 3, 4}, "Palette regimes missing")
    _require(any(int(np.count_nonzero(g == 0)) in (1, 2) for g in grids), "Sparse zero marks missing")
    _require(any(np.array_equal(g, np.rot90(g)) for g in grids), "Unchanged cases missing")
    _require(any(not _connected(g == color) for g in grids for color in np.unique(g)),
             "Separated configurations missing")
    _require(all(len({min(np.unique(pair["input"], return_counts=True)[1])
                      for split in ("train", "test") for pair in puzzle[split]}) >= 2
                 for puzzle in puzzles), "A puzzle lacks visible density variation")
    _require({int(min(np.unique(g, return_counts=True)[1])) for g in grids if g.shape == (3, 3)}
             == {1, 2, 3, 4}, "Small-grid density range collapsed")
    _require(sum(all(pair["input"].shape == (3, 3)
                     for split in ("train", "test") for pair in puzzle[split])
                 for puzzle in puzzles) >= 5, "Source-size puzzles missing")
    _require(sum(all(pair["input"].shape == (3, 3) for pair in puzzle["train"])
                 and all(pair["input"].shape[0] > 3 for pair in puzzle["test"])
                 for puzzle in puzzles) >= 3, "Held-out larger tests missing")
    _require(sum(any(len(np.unique(pair["input"])) > 2 for pair in puzzle["train"])
                 and any(len(np.unique(pair["input"])) > 2 for pair in puzzle["test"])
                 for puzzle in puzzles) >= 5, "Multicolor train/test demonstrations missing")
    _require(any(0 in grid and np.count_nonzero(grid == 0) <
                 max(np.unique(grid, return_counts=True)[1]) for grid in grids),
             "Zero never appears as a mark color")
    _require(any(np.array_equal(pair["input"], np.rot90(pair["input"]))
                 for puzzle in puzzles for pair in puzzle["test"]),
             "Unchanged test rotations missing")
    _require(any(any(np.array_equal(grid == color,
                                     np.fliplr(np.rot90(grid == color, turns)))
                     for turns in range(4))
                 for grid in grids for color in np.unique(grid)),
             "Reflection-symmetric arrangements missing")
    training_colors = [{int(color) for pair in puzzle["train"]
                        for color in np.unique(pair["input"])} for puzzle in puzzles]
    _require(any(set(int(color) for color in np.unique(pair["input"])) <= colors
                 for puzzle, colors in zip(puzzles, training_colors) for pair in puzzle["test"]),
             "No test palette reuses training colors")
    _require(any(not set(int(color) for color in np.unique(pair["input"])) <= colors
                 for puzzle, colors in zip(puzzles, training_colors) for pair in puzzle["test"]),
             "No test palette contains a held-out color")
    motifs = [np.array(rows, dtype=np.int8) for rows in (
        ((1, 1, 1), (0, 0, 0), (0, 0, 0)),
        ((1, 1, 1), (0, 1, 0), (0, 1, 0)),
        ((1, 1, 1), (1, 0, 0), (1, 0, 0)),
    )]
    observed_families = {_family(grid == color) for grid in grids if grid.shape == (3, 3)
                         for color in np.unique(grid)}
    _require(all(_family(motif) in observed_families for motif in motifs),
             "A requested bar, T, or corner family was not sampled")
