"""One-marker expanding cones with a learned periodic trail convention."""

import numpy as np


_SOURCE_WIDTHS = frozenset((5, 7, 13, 17))
_PERIODS = (3, 4, 5)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _integer(value, name: str, minimum: int | None = None,
             maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is out of range")
    return value


def _counts(seed: int, num_train: int | None,
            num_test: int | None) -> tuple[int, int]:
    train = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    test = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if train is None and test is None:
        phase = seed % 18
        test = 4 if phase == 17 else (1 if phase < 11 else (2 if phase < 15 else 3))
        train = test + 1 + (seed // 3) % 3
        if seed % 13 == 12:
            train = 7 + (seed // 13) % 4
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _choice(rng: np.random.Generator, values: list[int]) -> int:
    return values[int(rng.integers(len(values)))]


def _sample_widths(rng: np.random.Generator, train_count: int,
                   test_count: int, centered: bool,
                   extrapolate: bool) -> tuple[list[int], list[int]]:
    pool = list(range(5, 30, 2)) if centered else list(range(5, 31))
    if extrapolate:
        limit = (29 - 2 * test_count) if centered else (30 - test_count)
        train_pool = [n for n in pool if n <= limit]
    else:
        train_pool = pool.copy()
    anchors = [n for n in train_pool if (n in (17, 19, 21, 23) if centered
                                        else n in (18, 20, 22, 24))]
    anchor = _choice(rng, anchors)
    selected = [anchor]
    small = [n for n in train_pool if 5 <= n <= 10 and n != anchor]
    if train_count >= 2 and rng.random() < 0.75:
        selected.append(_choice(rng, small))
    remaining = [n for n in train_pool if n not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[:train_count - len(selected)])
    _require(len(selected) == train_count, "width selection failed")
    if extrapolate:
        test_pool = [n for n in pool if n > max(selected)]
        rng.shuffle(test_pool)
        tests = test_pool[:test_count]
    else:
        test_pool = [n for n in pool if n not in selected and n < max(selected)]
        first = _choice(rng, test_pool)
        remaining_test = [n for n in pool if n not in selected and n != first]
        rng.shuffle(remaining_test)
        tests = [first] + remaining_test[:test_count - 1]
    _require(len(tests) == test_count, "test width selection failed")
    rng.shuffle(selected)
    rng.shuffle(tests)
    return selected, tests


def _marker_position(rng: np.random.Generator, width: int,
                     centered: bool, force_offset: bool) -> int:
    if centered:
        return width // 2
    low = 3 if width >= 10 else 2
    high = width - 1 - low
    choices = list(range(low, high + 1))
    if force_offset:
        offset = [m for m in choices if m != width // 2 and
                  abs((width - 1) / 2 - m) >= 2]
        if not offset:
            offset = [m for m in choices if m != width // 2]
        if offset:
            choices = offset
    return _choice(rng, choices)


def _construct_pair(width: int, marker: int, background: int,
                    frontier: int, trail: int, period: int,
                    slant: int, rotation: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((1, width), background, dtype=np.int8)
    inp[0, marker] = frontier
    rows = np.arange(width)[:, None]
    columns = np.arange(width)[None, :]
    left = marker - rows
    right = marker + rows
    inside = (left < columns) & (columns < right)
    lattice = (columns - slant * rows - marker) % period == 0
    frontiers = (columns == left) | (columns == right)
    out = np.full((width, width), background, dtype=np.int8)
    out[inside & lattice] = trail
    out[frontiers] = frontier
    return np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()


def _input_roles(inp: np.ndarray) -> tuple[int, int]:
    values, counts = np.unique(inp, return_counts=True)
    _require(len(values) == 2 and sorted(int(x) for x in counts) == [1, inp.size - 1],
             "input must have one marker and a uniform background")
    marker = int(values[int(np.argmin(counts))])
    background = int(values[int(np.argmax(counts))])
    return background, marker


def _canonical_marker(inp: np.ndarray, rotation: int) -> int | None:
    unrotated = np.rot90(inp, -rotation)
    if unrotated.shape != (1, inp.size):
        return None
    background, marker_color = _input_roles(inp)
    positions = np.flatnonzero(unrotated[0] == marker_color)
    if len(positions) != 1 or np.any(unrotated[0, np.arange(inp.size) != positions[0]] != background):
        return None
    return int(positions[0])


def _trail_color(train: list[dict]) -> int:
    extras: set[int] = set()
    for pair in train:
        background, marker = _input_roles(pair["input"])
        extra = {int(x) for x in np.unique(pair["output"])} - {background, marker}
        _require(len(extra) <= 1, "an output has multiple trail colors")
        extras.update(extra)
    _require(len(extras) == 1, "training must establish one shared trail color")
    return next(iter(extras))


def _matching_models(train: list[dict], trail: int) -> list[tuple[int, int, int]]:
    matches = []
    for rotation in range(4):
        # Include nearby ungenerated periods when checking identifiability.
        for period in range(2, 8):
            for slant in (-1, 1):
                good = True
                for pair in train:
                    inp = pair["input"]
                    marker = _canonical_marker(inp, rotation)
                    if marker is None:
                        good = False
                        break
                    background, frontier = _input_roles(inp)
                    if trail in (background, frontier):
                        good = False
                        break
                    _, expected = _construct_pair(inp.size, marker, background,
                                                   frontier, trail, period, slant, rotation)
                    if not np.array_equal(pair["output"], expected):
                        good = False
                        break
                if good:
                    matches.append((rotation, period, slant))
    return matches


def _source_copy(pair: dict) -> bool:
    inp, out = pair["input"], pair["output"]
    if inp.shape[0] != 1 or inp.shape[1] not in _SOURCE_WIDTHS:
        return False
    width = inp.shape[1]
    source_inp, source_out = _construct_pair(width, width // 2, 0, 2, 1, 4, 1, 0)
    return np.array_equal(inp, source_inp) and np.array_equal(out, source_out)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    baseline = rng.random() < 0.40 and train_count + test_count <= 13
    if baseline:
        rotation, period, slant, centered = 0, 4, 1, True
    else:
        rotation = int(rng.integers(4))
        period = _choice(rng, list(_PERIODS))
        slant = _choice(rng, [-1, 1])
        centered = bool(rng.random() < 0.30) and train_count + test_count <= 13
    trail = int(rng.integers(10))
    extrapolate = bool(rng.random() < 0.30)
    train_widths, test_widths = _sample_widths(rng, train_count, test_count,
                                               centered, extrapolate)
    if baseline and train_count >= 3 and rng.random() < 0.25 and 3 not in train_widths:
        small_index = int(np.argmin(train_widths))
        train_widths[small_index] = 3
    pairs = {"train": [], "test": []}
    first_colors = None
    for split, widths in (("train", train_widths), ("test", test_widths)):
        for index, width in enumerate(widths):
            force_offset = not centered and ((split == "train" and width % 2 == 0 and width >= 18)
                                                   or (split == "test" and index == 0))
            marker = _marker_position(rng, width, centered, force_offset)
            choices = [c for c in range(10) if c != trail]
            if first_colors is not None and split == "train" and index == 1:
                background = _choice(rng, [c for c in choices if c != first_colors[0]])
                frontier = _choice(rng, [c for c in choices if c not in (background, first_colors[1])])
            else:
                background = _choice(rng, choices)
                frontier = _choice(rng, [c for c in choices if c != background])
            if (background, frontier, trail) == (0, 2, 1) and rotation == 0 and period == 4 and slant == 1 and width in _SOURCE_WIDTHS and marker == width // 2:
                background = _choice(rng, [c for c in choices if c not in (0, frontier)])
            if first_colors is None:
                first_colors = (background, frontier)
            inp, out = _construct_pair(width, marker, background, frontier,
                                       trail, period, slant, rotation)
            pairs[split].append({"input": inp, "output": out})
    validate_generated_puzzle(pairs)
    return pairs


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    seen_inputs = set()
    seen_pairs = set()
    seen_widths = set()
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair keys must be input and output")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2 and
                     np.issubdtype(grid.dtype, np.integer), "grids must be 2D integer arrays")
            _require(all(1 <= dimension <= 30 for dimension in grid.shape) and
                     bool(np.all((0 <= grid) & (grid <= 9))), "grid bounds violated")
        inp, out = pair["input"], pair["output"]
        _require(min(inp.shape) == 1 and 3 <= max(inp.shape) <= 30,
                 "input must be a marker strip")
        width = inp.size
        _require(out.shape == (width, width), "output must be a matching square")
        _input_roles(inp)
        _require(_grid_key(inp) not in seen_inputs, "duplicate input")
        _require((_grid_key(inp), _grid_key(out)) not in seen_pairs, "duplicate pair")
        _require(width not in seen_widths, "widths must be distinct within a puzzle")
        _require(not _source_copy(pair), "exact source pair copied")
        seen_inputs.add(_grid_key(inp))
        seen_pairs.add((_grid_key(inp), _grid_key(out)))
        seen_widths.add(width)
    trail = _trail_color(train)
    models = _matching_models(train, trail)
    _require(len(models) == 1, "training does not uniquely establish direction, period, and slant")
    rotation, period, slant = models[0]
    train_markers = [_canonical_marker(pair["input"], rotation) for pair in train]
    train_widths = [pair["input"].size for pair in train]
    trained_offcenter = any(w % 2 == 0 or m != w // 2
                            for w, m in zip(train_widths, train_markers))
    trained_even = any(w % 2 == 0 for w in train_widths)
    cadence = False
    post_clip = False
    precedence = False
    for pair in train + test:
        inp, out = pair["input"], pair["output"]
        background, frontier = _input_roles(inp)
        _require(trail not in (background, frontier), "pair colors must be distinct")
        marker = _canonical_marker(inp, rotation)
        _require(marker is not None, "pair input orientation differs from training")
        width = inp.size
        _require(1 < marker < width - 2 or width == 3,
                 "marker too close to an edge")
        _, expected = _construct_pair(width, marker, background, frontier,
                                       trail, period, slant, rotation)
        _require(np.array_equal(out, expected), "output violates cone and trail rule")
        if any(pair is test_pair for test_pair in test):
            if width % 2 == 0:
                _require(trained_even, "even test width absent from training")
            if width % 2 == 0 or marker != width // 2:
                _require(trained_offcenter, "offset test marker absent from training")
    for pair in train:
        inp = pair["input"]
        width = inp.size
        marker = _canonical_marker(inp, rotation)
        background, frontier = _input_roles(inp)
        canonical = np.rot90(pair["output"], -rotation)
        for row in range(width):
            columns = np.flatnonzero(canonical[row] == trail)
            if len(columns) >= 2 and period in np.diff(columns):
                cadence = True
            if row > max(marker, width - 1 - marker) and len(columns):
                post_clip = True
        if width >= 5 and canonical[1, marker + slant] == frontier:
            precedence = True
    _require(cadence and post_clip and precedence,
             "training must show cadence, trails after clipping, and frontier precedence")
    _require(len({int(_input_roles(p["input"])[0]) for p in train}) >= 2 and
             len({int(_input_roles(p["input"])[1]) for p in train}) >= 2,
             "training must vary both input-visible colors")


def _cellwise_check(pair: dict, model: tuple[int, int, int], trail: int) -> None:
    rotation, period, slant = model
    inp = np.rot90(pair["input"], -rotation)
    out = np.rot90(pair["output"], -rotation)
    background, frontier = _input_roles(inp)
    marker = int(np.flatnonzero(inp[0] == frontier)[0])
    width = inp.size
    for row in range(width):
        for column in range(width):
            if column == marker - row or column == marker + row:
                expected = frontier
            elif marker - row < column < marker + row and (column - slant * row - marker) % period == 0:
                expected = trail
            else:
                expected = background
            _require(int(out[row, column]) == expected, "independent cellwise check failed")


def _equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[role], b[role])
                   for a, b in zip(first[split], second[split])
                   for role in ("input", "output"))
               for split in ("train", "test"))


def _expect_value_error(**kwargs) -> None:
    try:
        generate_puzzle(9001, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError for invalid counts")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    models = []
    geometries = set()
    extractions = 0
    mixed_sizes = 0
    offset_puzzles = 0
    color_variation = 0
    baseline_puzzles = 0
    width_three_puzzles = 0
    offset_tests = 0
    single_tests = 0
    small_training_sets = 0
    background_colors = set()
    frontier_colors = set()
    trail_colors = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        trail = _trail_color(puzzle["train"])
        model, = _matching_models(puzzle["train"], trail)
        models.append(model)
        trail_colors.add(trail)
        train_widths = [p["input"].size for p in puzzle["train"]]
        test_widths = [p["input"].size for p in puzzle["test"]]
        single_tests += len(test_widths) == 1
        small_training_sets += len(train_widths) <= 5
        width_three_puzzles += 3 in train_widths
        extractions += min(test_widths) > max(train_widths)
        mixed_sizes += min(test_widths) < max(train_widths)
        centered = all(p["input"].size % 2 == 1 and
                       _canonical_marker(p["input"], model[0]) == p["input"].size // 2
                       for p in puzzle["train"] + puzzle["test"])
        baseline_puzzles += centered and model == (0, 4, 1)
        offset_tests += any(p["input"].size % 2 == 0 or
                            _canonical_marker(p["input"], model[0]) != p["input"].size // 2
                            for p in puzzle["test"])
        if len({_input_roles(p["input"])[0] for p in puzzle["train"]}) > 1 and len({_input_roles(p["input"])[1] for p in puzzle["train"]}) > 1:
            color_variation += 1
        if any(w % 2 == 0 or _canonical_marker(p["input"], model[0]) != w // 2
               for p in puzzle["train"] for w in (p["input"].size,)):
            offset_puzzles += 1
        for pair in puzzle["train"] + puzzle["test"]:
            _cellwise_check(pair, model, trail)
            _require(not _source_copy(pair), "source pair copied")
            background, frontier = _input_roles(pair["input"])
            background_colors.add(background)
            frontier_colors.add(frontier)
            width = pair["input"].size
            marker = _canonical_marker(pair["input"], model[0])
            geometries.add((width, marker, model))
    _require(len({len(p["train"]) for p in puzzles}) >= 4 and
             len({len(p["test"]) for p in puzzles}) >= 3,
             "automatic counts lack diversity")
    _require(len(set(models)) >= 7 and len({m[0] for m in models}) == 4 and
             len({m[1] for m in models}) == 3 and len({m[2] for m in models}) == 2,
             "puzzle conventions lack diversity")
    _require(extractions >= 4 and mixed_sizes >= 12 and offset_puzzles >= 8,
             "size and marker regimes lack diversity")
    _require(baseline_puzzles >= 8 and offset_tests >= 8 and
             1 <= width_three_puzzles <= 6,
             "baseline, offset tests, or boundary cases lack coverage")
    _require(single_tests >= 20 and small_training_sets >= 25 and
             min(len(background_colors), len(frontier_colors), len(trail_colors)) >= 7,
             "pair counts or color roles lack diversity")
    _require(color_variation == 42 and len(geometries) >= 100,
             "pair variation lacks diversity")
    signatures = {
        tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                    for pair in puzzle[split])
              for split in ("train", "test"))
        for puzzle in puzzles
    }
    _require(len(signatures) >= 40, "different seeds repeat whole puzzles")
    _require(_equal(puzzles[17], generate_puzzle(17)), "automatic generation is not deterministic")
    for seed, train, test in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                              (1010, 6, 3), (2000, None, 1), (2001, None, 4),
                              (2002, 2, None), (2003, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _require(train is None or len(puzzle["train"]) == train, "train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "test count ignored")
        _require(_equal(puzzle, generate_puzzle(seed, train, test)), "explicit generation is not deterministic")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4},
                   {"num_train": 2.5, "num_test": 1},
                   {"num_train": True, "num_test": 1}):
        _expect_value_error(**kwargs)
