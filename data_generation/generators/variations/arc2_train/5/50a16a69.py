"""Generate periodic-texture completion puzzles with a fixed left phase advance."""

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _grid_from_strings(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(cell) for cell in row] for row in rows], dtype=np.int8)


_SOURCE_INPUTS = (
    _grid_from_strings((
        "63576357635888888888", "57635763576888888888",
        "63576357635888888888", "57635763576888888888",
        "63576357635888888888", "57635763576888888888",
        "63576357635888888888", "57635763576888888888",
        "63576357635888888888", "57635763576888888888",
        "63576357635888888888", "88888888888888888888",
        "88888888888888888888", "88888888888888888888",
        "88888888888888888888", "88888888888888888888",
        "88888888888888888888", "88888888888888888888",
        "88888888888888888888", "88888888888888888888",
    )),
    _grid_from_strings((
        "52525251", "25252521", "52525251", "25252521",
        "52525251", "25252521", "52525251", "11111111",
    )),
    _grid_from_strings((
        "763763764", "637637634", "763763764",
        "637637634", "763763764", "637637634",
        "763763764", "637637634", "444444444",
    )),
    _grid_from_strings((
        "686863", "868683", "686863", "868683",
        "686863", "333333",
    )),
)


def _is_source_input(grid: np.ndarray) -> bool:
    return any(grid.shape == source.shape and np.array_equal(grid, source)
               for source in _SOURCE_INPUTS)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _optional_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _optional_count(num_train, "num_train", 2, 10)
    test = _optional_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _primitive_period(items: list | np.ndarray) -> int:
    length = len(items)
    for period in range(1, length + 1):
        if all(items[i] == items[i % period] for i in range(length)):
            return period
    raise AssertionError("A finite sequence always has a period")


def _row_phases(rng: np.random.Generator, horizontal: int,
                vertical: int) -> np.ndarray:
    if vertical == 1:
        return np.array([0], dtype=np.int64)
    if vertical == 2:
        return np.array([0, int(rng.integers(1, horizontal))], dtype=np.int64)
    # A primitive phase word can revisit a row; its whole sequence has period 3 or 4.
    for _ in range(32):
        tail = rng.integers(0, horizontal, size=vertical - 1)
        phases = np.concatenate(([0], tail)).astype(np.int64)
        if len(set(int(x) for x in phases)) >= min(3, horizontal, vertical) and _primitive_period(
            phases.tolist()
        ) == vertical:
            return phases
    if vertical == 3:
        return np.array([0, 1, 2 if horizontal >= 3 else 1], dtype=np.int64)
    return np.array([0, 1, 2, 1] if horizontal >= 3 else [0, 1, 1, 0],
                    dtype=np.int64)


def _sample_cycle(rng: np.random.Generator, period: int, filler: int,
                  repeated_color: bool) -> np.ndarray:
    count = period - int(repeated_color)
    available = [color for color in range(10) if color != filler]
    colors = [int(x) for x in rng.choice(available, size=count, replace=False)]
    if repeated_color:
        colors.append(int(rng.choice(colors)))
    cycle = np.array(colors, dtype=np.int8)
    rng.shuffle(cycle)
    return cycle


def _dimensions(rng: np.random.Generator, period: int, row_period: int,
                regime: str, source_like: bool) -> tuple[int, int, int, int]:
    min_width = 2 * period
    min_height = max(4, 2 * row_period)
    if source_like:
        canvas = int(rng.integers(5, 8))
        return canvas, canvas, canvas - 1, canvas - 1
    if rng.random() < (0.26 if regime == "source_train" else 0.16):
        low = max(min_width, min_height)
        side = int(rng.integers(low, 27))
        depth = int(rng.integers(1, min(10, 30 - side) + 1))
        return side + depth, side + depth, side, side
    upper = 25 if regime == "source_train" else 29
    seed_width = int(rng.integers(min_width, upper + 1))
    seed_height = int(rng.integers(min_height, upper + 1))
    right = int(rng.integers(1, min(12, 30 - seed_width, seed_width + 2) + 1))
    bottom = int(rng.integers(1, min(12, 30 - seed_height, seed_height + 2) + 1))
    return seed_height + bottom, seed_width + right, seed_height, seed_width


def _construct_pair(rng: np.random.Generator, regime: str, period: int,
                    row_period: int, repeated_color: bool, source_like: bool,
                    forbidden_fillers: set[int]) -> dict:
    filler_options = [x for x in range(10) if x not in forbidden_fillers]
    filler = int(rng.choice(filler_options))
    cycle = _sample_cycle(rng, period, filler, repeated_color)
    phases = _row_phases(rng, period, row_period)
    height, width, seed_height, seed_width = _dimensions(
        rng, period, row_period, regime, source_like
    )
    rows = np.arange(height, dtype=np.int64)[:, None]
    columns = np.arange(width, dtype=np.int64)[None, :]
    texture = cycle[(columns + phases[rows % row_period] + 1) % period]
    input_grid = np.full((height, width), filler, dtype=np.int8)
    input_grid[:seed_height, :seed_width] = cycle[
        (columns[:, :seed_width] + phases[rows[:seed_height] % row_period])
        % period
    ]
    return {"input": input_grid, "output": texture.astype(np.int8, copy=False)}


def _unique_pair(rng: np.random.Generator, regime: str, period: int,
                 row_period: int, repeated_color: bool, source_like: bool,
                 forbidden_fillers: set[int], seen: set) -> dict:
    for _ in range(128):
        child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        pair = _construct_pair(
            np.random.default_rng(child_seed), regime, period, row_period,
            repeated_color, source_like, forbidden_fillers
        )
        key = _grid_key(pair["input"])
        if key not in seen and not _is_source_input(pair["input"]):
            seen.add(key)
            return pair
    raise ValueError("Unable to make a distinct non-source pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)

    # The only puzzle-wide convention is the source's fixed one-cell left advance.
    # Pair palettes, row cycles, and dimensions are local and visible in each input.
    periods = [2, int(rng.integers(3, 5))]
    periods.extend(int(x) for x in rng.integers(2, 5, size=train_count - 2))
    rng.shuffle(periods)
    row_periods = [2, int(rng.choice([1, 3, 4]))]
    row_periods.extend(int(x) for x in rng.integers(1, 5, size=train_count - 2))
    rng.shuffle(row_periods)
    puzzle = {"train": [], "test": []}
    seen = set()
    training_fillers = set()
    repeat_witnesses = set()

    for index, (period, row_period) in enumerate(zip(periods, row_periods)):
        repeated = period >= 3 and rng.random() < 0.30
        pair = _unique_pair(
            rng, "source_train", period, row_period, repeated, False,
            training_fillers if index == 1 else set(), seen
        )
        puzzle["train"].append(pair)
        filler = int(pair["input"][-1, -1])
        training_fillers.add(filler)
        if repeated:
            repeat_witnesses.add(period)

    train_period_set = sorted(set(periods))
    train_row_set = sorted(set(row_periods))
    for _ in range(test_count):
        source_like = rng.random() < 0.22
        if source_like:
            period, row_period = 2, 2
        else:
            period = int(rng.choice(train_period_set))
            row_period = int(rng.choice(train_row_set))
        repeated = period in repeat_witnesses and rng.random() < 0.35
        pair = _unique_pair(
            rng, "target_test", period, row_period, repeated, source_like,
            set(), seen
        )
        puzzle["test"].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair: dict) -> dict:
    input_grid, output_grid = pair["input"], pair["output"]
    _require(input_grid.shape == output_grid.shape,
             "Input and output dimensions must agree")
    height, width = input_grid.shape
    filler = int(input_grid[-1, -1])
    seed_height = next((r for r in range(height)
                        if int(input_grid[r, 0]) == filler), height)
    seed_width = next((c for c in range(width)
                       if int(input_grid[0, c]) == filler), width)
    _require(1 <= seed_height < height and 1 <= seed_width < width,
             "An upper-left seed and both filler regions are required")
    seed = input_grid[:seed_height, :seed_width]
    _require(np.all(seed != filler)
             and np.all(input_grid[seed_height:, :] == filler)
             and np.all(input_grid[:seed_height, seed_width:] == filler),
             "Filler must form a uniform right-and-bottom L")

    first_row = seed[0].tolist()
    period = _primitive_period(first_row)
    _require(2 <= period <= 4 and seed_width >= 2 * period,
             "Seed must expose two cycles of primitive period 2–4")
    cycle = seed[0, :period].astype(np.int8, copy=True)
    positions = np.arange(seed_width) % period
    phases = []
    for row in seed:
        matching = [shift for shift in range(period)
                    if np.array_equal(row, cycle[(positions + shift) % period])]
        _require(len(matching) == 1,
                 "Every seed row must have one inferable phase of the same cycle")
        phases.append(matching[0])
    row_period = _primitive_period(phases)
    _require(1 <= row_period <= 4 and seed_height >= max(4, 2 * row_period),
             "Seed must expose two cycles of a row period 1–4")
    _require(len(set(cycle.tolist())) >= 2,
             "Texture needs at least two colors")
    texture_colors = set(int(x) for x in cycle)
    _require(filler not in texture_colors,
             "Filler cannot be a texture color in the same pair")

    rows = np.arange(height, dtype=np.int64)[:, None]
    columns = np.arange(width, dtype=np.int64)[None, :]
    expected = cycle[(columns + np.array(phases[:row_period])[
        rows % row_period] + 1) % period]
    _require(np.array_equal(output_grid, expected),
             "Output must advance every row one cell left, then tile")
    _require(set(int(x) for x in np.unique(output_grid)) == texture_colors,
             "Output must contain only the completed texture")
    _require(not np.array_equal(input_grid, output_grid),
             "Replacing filler must change the grid")
    return {
        "height": height, "width": width, "seed_height": seed_height,
        "seed_width": seed_width, "bottom": height - seed_height,
        "right": width - seed_width, "filler": filler,
        "period": period, "row_period": row_period,
        "texture_colors": texture_colors,
        "repeated_color": len(texture_colors) < period,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list)
             and isinstance(puzzle["test"], list), "Sections must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4
             and len(train) > len(test), "Illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    metadata = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Each pair must have exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "Grids must be two-dimensional NumPy arrays")
                _require(all(1 <= side <= 30 for side in grid.shape)
                         and np.issubdtype(grid.dtype, np.integer)
                         and int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "Grid size, dtype, or color is invalid")
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "Duplicate input or pair")
            _require(not _is_source_input(pair["input"]), "Exact source copy")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            metadata[section].append(_analyze_pair(pair))

    train_meta, test_meta = metadata["train"], metadata["test"]
    train_periods = {m["period"] for m in train_meta}
    train_row_periods = {m["row_period"] for m in train_meta}
    _require(2 in train_periods and bool(train_periods & {3, 4}),
             "Training needs a checkerboard and an asymmetric direction witness")
    _require(2 in train_row_periods, "Training needs source-like alternating rows")
    _require(len({m["filler"] for m in train_meta}) >= 2,
             "Training must show that filler is a local spatial role")
    _require(len({(m["height"], m["width"], m["seed_height"],
                   m["seed_width"], m["period"], m["row_period"])
                  for m in train_meta}) >= 2,
             "Training layouts must vary")
    for m in test_meta:
        _require(m["period"] in train_periods
                 and m["row_period"] in train_row_periods,
                 "Test periods must be demonstrated in training")
        if m["repeated_color"]:
            _require(any(t["period"] == m["period"] and t["repeated_color"]
                         for t in train_meta),
                     "A repeated-color test cycle needs a training witness")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k])
                   for a, b in zip(first[s], second[s])
                   for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    _require(_puzzles_equal(generate_puzzle(12345), generate_puzzle(12345)),
             "Automatic generation must be deterministic")
    _require(_puzzles_equal(generate_puzzle(54321, 7, 3),
                            generate_puzzle(54321, 7, 3)),
             "Explicit generation must be deterministic")
    for i, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                       (None, 1), (None, 4),
                                       (2, None), (10, None))):
        puzzle = generate_puzzle(20000 + i, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "Train count not honored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "Test count not honored")
    for i, args in enumerate(({"num_train": 1}, {"num_train": 11},
                              {"num_test": 0}, {"num_test": 5},
                              {"num_train": 2, "num_test": 2},
                              {"num_train": 3, "num_test": 4},
                              {"num_train": True, "num_test": 1},
                              {"num_train": 4, "num_test": False})):
        try:
            generate_puzzle(21000 + i, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count request was accepted")

    train_counts, test_counts, signatures = set(), set(), set()
    periods, row_periods, test_periods, test_row_periods = set(), set(), set(), set()
    fillers, dimensions, layouts, depths = set(), set(), set(), set()
    repeated_train = repeated_test = rectangular = unequal_depth = 0
    small_test = overlapping_test = exchange = combined = 0
    no_common_texture = different_seed_sizes = partial_test = 0
    pairwise_exchange = no_exchange = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_pair_key(p) for s in ("train", "test")
                             for p in puzzle[s]))
        tm = [_analyze_pair(p) for p in puzzle["train"]]
        em = [_analyze_pair(p) for p in puzzle["test"]]
        train_texture = set().union(*(m["texture_colors"] for m in tm))
        train_fillers = {m["filler"] for m in tm}
        no_common_texture += not set.intersection(
            *(m["texture_colors"] for m in tm)
        )
        different_seed_sizes += len({
            (m["seed_height"], m["seed_width"]) for m in tm
        }) >= 2
        largest_train = max(max(m["height"], m["width"]) for m in tm)
        for m in tm + em:
            dimensions.add((m["height"], m["width"]))
            layouts.add((m["height"], m["width"], m["seed_height"],
                         m["seed_width"], m["period"], m["row_period"]))
            depths.add((m["bottom"], m["right"]))
            fillers.add(m["filler"])
            rectangular += m["height"] != m["width"] or m["seed_height"] != m["seed_width"]
            unequal_depth += m["bottom"] != m["right"]
            combined += (m["repeated_color"] and m["row_period"] >= 3
                         and m["bottom"] != m["right"])
        for m in tm:
            periods.add(m["period"])
            row_periods.add(m["row_period"])
            repeated_train += m["repeated_color"]
        for m in em:
            test_periods.add(m["period"])
            test_row_periods.add(m["row_period"])
            repeated_test += m["repeated_color"]
            small_test += (m["height"] == m["width"]
                           and 5 <= m["height"] <= 7 and m["period"] == 2)
            overlapping_test += max(m["height"], m["width"]) >= largest_train
            exchange += (m["filler"] in train_texture
                         and bool(m["texture_colors"] & train_fillers))
            pairwise_exchange += any(
                m["filler"] in t["texture_colors"]
                and t["filler"] in m["texture_colors"] for t in tm
            )
            no_exchange += m["filler"] not in train_texture
            partial_test += (m["seed_width"] % m["period"] != 0
                             or m["width"] % m["period"] != 0)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "Automatic pair counts need broad coverage")
    _require(len(signatures) >= 40 and len(layouts) >= 80
             and len(dimensions) >= 35 and len(depths) >= 20,
             "Cross-seed structural diversity is too narrow")
    _require(periods == {2, 3, 4} and test_periods == {2, 3, 4}
             and row_periods == {1, 2, 3, 4}
             and test_row_periods == {1, 2, 3, 4},
             "Horizontal and vertical periods need full coverage")
    _require(repeated_train >= 3 and repeated_test >= 1 and combined >= 1,
             "Repeated-color cycles and their combinations need coverage")
    _require(rectangular >= 20 and unequal_depth >= 20
             and small_test >= 3 and overlapping_test >= 1 and exchange >= 1,
             "Shape, depth, small test, size overlap, or role exchange absent")
    _require(no_common_texture >= 20 and different_seed_sizes >= 20
             and pairwise_exchange >= 1 and no_exchange >= 1
             and partial_test >= 10,
             "Independent palettes, seed evidence, or partial cycles absent")
    _require(len(fillers) == 10, "All ARC colors should occur as filler")
