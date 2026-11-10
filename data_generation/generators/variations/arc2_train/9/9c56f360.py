"""Independent edge-run gravity puzzles with visible, per-line collisions."""

import numpy as np


_SOURCE = (
    (("00080083", "08008003", "88080083", "88000003", "00088008", "80000000", "00088800"),
     ("00080083", "08008300", "88080083", "88300000", "00088008", "80000000", "00088800")),
    (("000800", "008008", "800008", "008080", "000033", "808033", "080880"),
     ("000800", "008008", "800008", "008080", "330000", "808330", "080880")),
    (("00008888", "00080833", "80080033", "88000033", "88008808", "00080803", "08000003", "00088083", "80088808"),
     ("00008888", "00080833", "80083300", "88330000", "88008808", "00080830", "08300000", "00088083", "80088808")),
    (("088888808", "888008808", "088088008", "080000033", "080800033", "800000033", "008088033", "088800000", "080808880"),
     ("088888808", "888008808", "088088008", "083300000", "080833000", "833000000", "008088330", "088800000", "080808880")),
)


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _key(grid):
    return grid.shape, tuple(map(int, grid.ravel()))


def _source_copy(a, b):
    for first, second in _SOURCE:
        if a.shape != (len(first), len(first[0])):
            continue
        if all("".join(map(str, a[r])) == row for r, row in enumerate(first)) and all(
            "".join(map(str, b[r])) == row for r, row in enumerate(second)
        ):
            return True
    return False


def _groups(lengths):
    groups = []
    begin = None
    for i, length in enumerate(lengths):
        if length and begin is None:
            begin = i
        if not length and begin is not None:
            groups.append((begin, i))
            begin = None
    if begin is not None:
        groups.append((begin, len(lengths)))
    return groups


def _inspect(grid, colors):
    background, obstacle, movable = colors
    _check(set(map(int, np.unique(grid))) == set(colors), "three distinct roles required")
    height, width = grid.shape
    lengths = np.zeros(height, dtype=np.int16)
    stops = {}
    for row in range(height):
        columns = np.flatnonzero(grid[row] == movable)
        if not len(columns):
            continue
        start = int(columns[0])
        _check(np.array_equal(columns, np.arange(start, width)), "movable run is not edge attached")
        _check(1 <= width - start < width, "bad movable run length")
        lengths[row] = width - start
        walls = np.flatnonzero(grid[row, :start] == obstacle)
        stops[row] = int(walls[-1] + 1) if len(walls) else 0
    _check(bool(stops), "no movable run")
    _check(int(np.count_nonzero(grid == background)) > int(np.count_nonzero(grid == obstacle)),
           "background must be visually prevalent")
    return lengths, stops


def _fall(grid, colors):
    background, _, movable = colors
    lengths, stops = _inspect(grid, colors)
    result = grid.copy()
    width = grid.shape[1]
    for row, length in enumerate(lengths):
        if length:
            result[row, width - length:] = background
            result[row, stops[row]:stops[row] + length] = movable
    return result


def _orient(grid, rotation):
    return np.rot90(grid, rotation).copy()


def _candidates(a, b):
    if a.shape != b.shape or np.array_equal(a, b):
        return set()
    found = set()
    colors = tuple(map(int, np.unique(a)))
    if len(colors) != 3 or set(map(int, np.unique(b))) != set(colors):
        return found
    for rotation in range(4):
        x = _orient(a, -rotation)
        y = _orient(b, -rotation)
        for movable in colors:
            for background in colors:
                if background == movable:
                    continue
                obstacle = next(c for c in colors if c not in (movable, background))
                try:
                    predicted = _fall(x, (background, obstacle, movable))
                except AssertionError:
                    continue
                if np.array_equal(predicted, y):
                    found.add((background, obstacle, movable, rotation))
    return found


def _predict_from_train(conventions, test_input):
    predictions = {}
    for background, obstacle, movable, rotation in conventions:
        if set(map(int, np.unique(test_input))) != {background, obstacle, movable}:
            continue
        canonical = _orient(test_input, -rotation)
        try:
            predicted = _orient(_fall(canonical, (background, obstacle, movable)), rotation)
        except AssertionError:
            continue
        if np.array_equal(test_input, predicted):
            continue
        predictions[_key(predicted)] = predicted
    _check(len(predictions) == 1, "training and test input do not determine one output")
    return next(iter(predictions.values()))


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} through {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _lengths(rng, height, max_width, lesson, four_stop):
    capacity = height
    if lesson == 0 or four_stop:
        group_count = 1
        active = int(rng.integers(4 if four_stop else 3, min(capacity - 1, 7) + 1))
    elif lesson == 1:
        group_count = 2 if capacity >= max_width + 2 else 1
        active = int(rng.integers(max(max_width, group_count + 1), min(capacity - group_count, 9) + 1))
    else:
        group_count = int(rng.integers(1, min(3, (capacity + 1) // 2) + 1))
        upper = min(capacity - group_count, 10)
        active = int(rng.integers(group_count, upper + 1))
    group_heights = [1] * group_count
    for _ in range(active - group_count):
        group_heights[int(rng.integers(group_count))] += 1
    rng.shuffle(group_heights)
    gaps = [0] * (group_count + 1)
    for i in range(1, group_count):
        gaps[i] = 1
    extra = capacity - active - (group_count - 1)
    for _ in range(extra):
        gaps[int(rng.integers(group_count + 1))] += 1
    lengths = np.zeros(height, dtype=np.int16)
    cursor = gaps[0]
    for i, group_height in enumerate(group_heights):
        lengths[cursor:cursor + group_height] = rng.integers(
            1, max_width + 1, size=group_height, dtype=np.int16
        )
        cursor += group_height + gaps[i + 1]
    active_rows = list(map(int, np.flatnonzero(lengths)))
    if lesson == 0:
        lengths[active_rows[:3]] = rng.integers(1, min(max_width, 2) + 1, size=3)
    if lesson == 1:
        for width, row in enumerate(rng.permutation(active_rows)[:max_width], 1):
            lengths[int(row)] = width
    if four_stop:
        start = _groups(lengths)[0][0]
        lengths[start:start + 4] = 2
    return lengths


def _stops(rng, lengths, width, lesson, four_stop):
    rows = list(map(int, np.flatnonzero(lengths)))
    stops = {}
    for row in rows:
        source_start = width - int(lengths[row])
        mode = float(rng.random())
        if mode < 0.17:
            stop = 0
        elif mode < 0.34:
            stop = source_start
        else:
            stop = int(rng.integers(1, source_start))
        stops[row] = stop
    if lesson == 0:
        rows = rows[:3]
        stops[rows[0]] = width - int(lengths[rows[0]])
        stops[rows[1]] = 0
        source_start = width - int(lengths[rows[2]])
        stops[rows[2]] = int(rng.integers(3, source_start))
    if four_stop:
        rows = rows[:4]
        source_start = width - 2
        choices = rng.choice(np.arange(source_start + 1), size=4, replace=False)
        for row, stop in zip(rows, choices):
            stops[row] = int(stop)
    if all(stops[r] == width - int(lengths[r]) for r in stops):
        stops[rows[0]] = 0
    return stops


def _obstacles(rng, height, width, lengths, stops, density, lesson):
    allowed = np.ones((height, width), dtype=bool)
    forced = np.zeros((height, width), dtype=bool)
    for row in np.flatnonzero(lengths):
        row = int(row)
        start = width - int(lengths[row])
        stop = stops[row]
        allowed[row, stop:] = False
        if stop:
            allowed[row, stop - 1] = True
            forced[row, stop - 1] = True
    if lesson == 0:
        row = int(np.flatnonzero(lengths)[2])
        forced[row, int(rng.integers(0, stops[row] - 1))] = True
    else:
        possibilities = [r for r in stops if stops[r] >= 3]
        if possibilities and rng.random() < 0.55:
            row = int(rng.choice(possibilities))
            forced[row, int(rng.integers(0, stops[row] - 1))] = True
    score = rng.random((height, width))
    motifs = int(rng.integers(4, 11))
    for _ in range(motifs):
        row = int(rng.integers(height))
        col = int(rng.integers(width))
        size = int(rng.integers(2, 6))
        kind = int(rng.integers(4))
        for step in range(size):
            for thick in range(2 if kind == 2 else 1):
                rr = row + (step if kind in (1, 2, 3) else thick)
                cc = col + (step if kind == 0 else thick if kind == 2 else step // 2 if kind == 3 else 0)
                if rr < height and cc < width:
                    score[rr, cc] += float(rng.uniform(1.5, 3.0))
    target = max(int(round(density * height * width)), int(np.count_nonzero(forced)))
    target = min(target, int(np.count_nonzero(allowed)))
    score[~allowed] = -1
    score[forced] = 100
    flat = np.argsort(score.ravel())[-target:]
    obstacles = np.zeros((height, width), dtype=bool)
    obstacles.ravel()[flat] = True
    return obstacles


def _pair(rng, colors, rotation, max_width, split, lesson):
    if lesson == 0:
        width = int(rng.integers(max(8, max_width + 4), 17))
    else:
        width = int(rng.integers(max(7, max_width + 3), 21))
    height = int(rng.integers(7, 19))
    if split == "test" and rng.random() < 0.30:
        width = int(rng.integers(max(10, max_width + 4), 22))
        height = int(rng.integers(10, 20))
    four_stop = lesson is None and rng.random() < (0.16 if split == "test" else 0.10)
    lengths = _lengths(rng, height, max_width, lesson, four_stop)
    stops = _stops(rng, lengths, width, lesson, four_stop)
    density_band = int(rng.integers(3))
    low, high = ((0.08, 0.17), (0.17, 0.28), (0.28, 0.40))[density_band]
    density = float(rng.uniform(low, high))
    obstacles = _obstacles(rng, height, width, lengths, stops, density, lesson)
    background, obstacle, movable = colors
    grid = np.full((height, width), background, dtype=np.int8)
    grid[obstacles] = obstacle
    for row in np.flatnonzero(lengths):
        grid[row, width - int(lengths[row]):] = movable
    output = _fall(grid, colors)
    _check(not np.array_equal(grid, output), "no visible motion")
    return {"input": _orient(grid, rotation), "output": _orient(output, rotation)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    max_width = int(rng.integers(2, 5))
    movable = int(rng.integers(10))
    remaining = [c for c in range(10) if c != movable]
    rng.shuffle(remaining)
    palette_count = 2 if rng.random() < 0.42 else 1
    palettes = [(remaining[2 * i], remaining[2 * i + 1], movable) for i in range(palette_count)]
    rotation_count = 1 if rng.random() < 0.45 else int(rng.integers(2, min(4, train_count) + 1))
    rotations = list(map(int, rng.choice(4, size=rotation_count, replace=False)))
    combos = [(palettes[i % palette_count], rotations[i % rotation_count])
              for i in range(max(palette_count, rotation_count))]
    while len(combos) < train_count:
        combos.append((palettes[int(rng.integers(palette_count))],
                       rotations[int(rng.integers(rotation_count))]))
    rng.shuffle(combos)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            colors, rotation = combos[index] if split == "train" else combos[int(rng.integers(train_count))]
            lesson = index if split == "train" and index < 2 else None
            for _ in range(40):
                child_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                try:
                    pair = _pair(child_rng, colors, rotation, max_width, split, lesson)
                except AssertionError:
                    continue
                key = _key(pair["input"])
                if key in seen or _source_copy(pair["input"], pair["output"]):
                    continue
                if _candidates(pair["input"], pair["output"]) != {(*colors, rotation)}:
                    continue
                if split == "test":
                    learned = [next(iter(_candidates(p["input"], p["output"])))
                               for p in puzzle["train"]]
                    try:
                        predicted = _predict_from_train(learned, pair["input"])
                    except AssertionError:
                        continue
                    if not np.array_equal(predicted, pair["output"]):
                        continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise ValueError("could not construct an unambiguous distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "bad split schema")
    _check(2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
           and len(puzzle["train"]) > len(puzzle["test"]), "bad pair counts")
    seen_inputs = set()
    seen_pairs = set()
    conventions = []
    widths_train = set()
    evidence = {"stationary": False, "boundary": False, "nearest": False, "shear": False}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "bad grid type")
                _check(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
                _check(all(1 <= d <= 30 for d in grid.shape), "bad grid size")
                _check(bool(np.all((0 <= grid) & (grid <= 9))), "illegal ARC color")
            _check(a.shape == b.shape and not np.array_equal(a, b), "grid must visibly transform in place")
            input_key = _key(a)
            pair_key = (input_key, _key(b))
            _check(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _check(not _source_copy(a, b), "source pair copied")
            candidates = _candidates(a, b)
            _check(len(candidates) == 1, "pair does not establish a unique local rule")
            convention = next(iter(candidates))
            if split == "train":
                conventions.append(convention)
                canonical = _orient(a, -convention[3])
                lengths, stops = _inspect(canonical, convention[:3])
                widths_train.update(map(int, lengths[lengths > 0]))
                for row in np.flatnonzero(lengths):
                    row = int(row)
                    source_start = canonical.shape[1] - int(lengths[row])
                    evidence["stationary"] |= stops[row] == source_start
                    evidence["boundary"] |= stops[row] == 0
                    evidence["nearest"] |= int(np.count_nonzero(canonical[row, :source_start] == convention[1])) >= 2
                    if row + 1 < len(lengths) and lengths[row + 1]:
                        evidence["shear"] |= stops[row] != stops[row + 1]
            else:
                _check(convention in conventions, "test color and direction combination absent from training")
                prediction = _predict_from_train(conventions, a)
                _check(np.array_equal(prediction, b), "test answer is not inferable from training and input")
    max_width = max(widths_train)
    _check(set(range(1, max_width + 1)).issubset(widths_train), "training did not show each run width")
    _check(all(evidence.values()), "training lacks collision, boundary, or scanline evidence")


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(first[s], second[s]) for k in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(6200, 6242)]
    _check(_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3)), "explicit determinism")
    _check(_equal(generate_puzzle(12346), generate_puzzle(12346)), "automatic determinism")
    for seed, train, test in ((11, 2, 1), (12, 10, 1), (13, 10, 4),
                              (14, None, 1), (15, None, 4), (16, 2, None), (17, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _check(train is None or len(puzzle["train"]) == train, "explicit train count")
        _check(test is None or len(puzzle["test"]) == test, "explicit test count")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(9, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "train counts collapsed")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "test counts collapsed")
    normalized = set()
    shapes = set()
    density = {"train": [], "test": []}
    widths = {"train": set(), "test": set()}
    group_heights = {"train": set(), "test": set()}
    directions = set()
    within_rotations = 0
    within_palettes = 0
    single_groups = 0
    separated = {"train": 0, "test": 0}
    stationary_test = 0
    far_test = 0
    same_stop_test = 0
    multiple_walls = {"train": 0, "test": 0}
    four_stops = {"train": 0, "test": 0}
    simple_test = 0
    rich_test = 0
    smaller_test = 0
    within_shapes = 0
    within_density = 0
    patch_pairs = {"train": 0, "test": 0}
    irregular_pairs = {"train": 0, "test": 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_conventions = [next(iter(_candidates(p["input"], p["output"]))) for p in puzzle["train"]]
        within_rotations += len({c[3] for c in train_conventions}) > 1
        within_palettes += len({c[:3] for c in train_conventions}) > 1
        within_shapes += len({p["input"].shape for s in puzzle.values() for p in s}) > 1
        puzzle_density = []
        train_areas = [p["input"].size for p in puzzle["train"]]
        test_areas = [p["input"].size for p in puzzle["test"]]
        smaller_test += min(test_areas) < max(train_areas)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a, b = pair["input"], pair["output"]
                convention = next(iter(_candidates(a, b)))
                directions.add(convention[3])
                canonical = _orient(a, -convention[3])
                lengths, stops = _inspect(canonical, convention[:3])
                groups = _groups(lengths)
                group_heights[split].update(end - start for start, end in groups)
                separated[split] += len(groups) > 1
                single_groups += any(end - start == 1 for start, end in groups)
                widths[split].update(map(int, lengths[lengths > 0]))
                obstacle_fraction = float(np.mean(a == convention[1]))
                density[split].append(obstacle_fraction)
                puzzle_density.append(obstacle_fraction)
                shapes.add(a.shape)
                remap = {convention[0]: 0, convention[1]: 1, convention[2]: 2}
                normalized.add((a.shape, tuple(remap[int(x)] for x in a.ravel())))
                for row in np.flatnonzero(lengths):
                    row = int(row)
                    start = canonical.shape[1] - int(lengths[row])
                    multiple_walls[split] += int(np.count_nonzero(canonical[row, :start] == convention[1])) >= 2
                    if split == "test":
                        stationary_test += stops[row] == start
                        far_test += stops[row] == 0
                        if row + 1 < len(lengths) and lengths[row + 1]:
                            same_stop_test += stops[row] == stops[row + 1]
                if split == "test":
                    simple_test += int(np.count_nonzero(lengths) <= 3)
                    rich_test += int(len(groups) >= 2 or np.count_nonzero(lengths) >= 5)
                walls = canonical == convention[1]
                if (np.any(walls[:, 1:] & walls[:, :-1])
                        and np.any(walls[1:] & walls[:-1])
                        and np.any(walls[:-1, :-1] & walls[1:, :-1]
                                   & walls[:-1, 1:] & walls[1:, 1:])):
                    patch_pairs[split] += 1
                irregular_pairs[split] += any(
                    lengths[r] != lengths[r + 1]
                    for r in range(len(lengths) - 1)
                    if lengths[r] and lengths[r + 1]
                )
                four_stops[split] += any(
                    len({stops[r + j] for j in range(4)}) == 4
                    for r in range(len(lengths) - 3) if np.all(lengths[r:r + 4] == 2)
                )
                _check(np.array_equal(_orient(_fall(canonical, convention[:3]), convention[3]), b),
                       "independent scanline result disagrees")
        within_density += max(puzzle_density) - min(puzzle_density) >= 0.12
    _check(len(shapes) >= 35 and len(normalized) >= 180, "structural diversity collapsed")
    _check(directions == {0, 1, 2, 3}, "direction coverage incomplete")
    _check(within_rotations >= 12 and within_palettes >= 8, "pair-level conventions lack coverage")
    _check(within_shapes >= 35 and within_density >= 25, "within-puzzle geometry or density collapsed")
    _check(smaller_test >= 20 and simple_test >= 15 and rich_test >= 25,
           "test complexity remains scripted")
    _check(widths["train"] == {1, 2, 3, 4} and widths["test"] == {1, 2, 3, 4}, "run width coverage incomplete")
    _check(single_groups >= 10 and all(separated[s] >= 10 for s in separated), "group geometry collapsed")
    _check(all(min(density[s]) < 0.17 and max(density[s]) > 0.27 for s in density), "density split cue remains")
    _check(stationary_test >= 8 and far_test >= 8 and same_stop_test >= 3, "test stopping modes absent")
    _check(all(multiple_walls[s] >= 15 for s in multiple_walls), "nearest-obstacle cases absent")
    _check(all(patch_pairs[s] >= 25 and irregular_pairs[s] >= 15
               and four_stops[s] >= 1 for s in ("train", "test")),
           "patch, silhouette, or four-stop examples absent")
