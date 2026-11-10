import numpy as np


_SOURCE_INPUT_ROWS = (
    ("080000000800", "808000008080", "000800080008", "000080800000", "000008000000"),
    ("04000400", "00404000", "00040000"),
    ("10000000", "01000001", "00100010", "00010100", "00001000"),
    ("00030000", "00303000", "03000300", "30000030"),
    ("006000006", "060600060", "600060600", "000006000"),
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


_SOURCE_INPUT_KEYS = frozenset(
    (len(rows), len(rows[0]), tuple(int(ch) for row in rows for ch in row))
    for rows in _SOURCE_INPUT_ROWS
)


def _is_source_input(grid):
    return (grid.shape[0], grid.shape[1], _grid_key(grid)[1]) in _SOURCE_INPUT_KEYS


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + int(seed) % 4
        num_train = num_test + 1 + (int(seed) // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _walk_signature(rows):
    rows = tuple(int(x) for x in rows)
    top = max(rows)
    variants = (rows, rows[::-1], tuple(top - x for x in rows),
                tuple(top - x for x in rows[::-1]))
    return min(variants)


def _walk_features(rows):
    counts = np.bincount(rows)
    max_gap = 0
    for row in range(len(counts)):
        columns = np.flatnonzero(rows == row)
        if len(columns) > 1:
            max_gap = max(max_gap, int(np.diff(columns).max()) - 1)
    steps = np.diff(rows)
    turns = int(np.count_nonzero(steps[1:] != steps[:-1]))
    return int(np.count_nonzero(counts)), int(counts.max()), max_gap, turns


def _reflect_walk(rng, rows):
    rows = np.asarray(rows, dtype=np.int64)
    rows = rows - int(rows.min())
    if int(rng.integers(2)):
        rows = rows[::-1].copy()
    if int(rng.integers(2)):
        rows = int(rows.max()) - rows
    return rows


def _make_walk(rng, mode):
    if mode == "simple":
        depth = int(rng.choice((2, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9)))
        if depth >= 3 and rng.random() < 0.08:
            returning = 1
        else:
            returning = int(rng.integers(2, depth + 1))
        rows = list(range(depth + 1)) + list(
            range(depth - 1, depth - returning - 1, -1)
        )
        return _reflect_walk(rng, rows)
    if mode not in ("fold", "extended"):
        raise ValueError("unknown walk mode")

    for _ in range(160):
        ceiling = int(rng.integers(3, 12))
        length = int(rng.integers(11, 22)) if mode == "fold" else int(rng.integers(18, 29))
        position = int(rng.integers(ceiling))
        direction = 1 if int(rng.integers(2)) else -1
        turn_chance = float(rng.uniform(0.07, 0.30))
        rows = [position]
        for _ in range(length - 1):
            if position + direction < 0 or position + direction >= ceiling or rng.random() < turn_chance:
                direction = -direction
            position += direction
            rows.append(position)
        rows = np.asarray(rows, dtype=np.int64)
        rows -= int(rows.min())
        active, intersections, gap, turns = _walk_features(rows)
        minimum = 3 if mode == "fold" else 4
        maximum = 4 if mode == "fold" else 6
        if active >= 3 and gap >= 3 and minimum <= intersections <= maximum and turns >= (2 if mode == "fold" else 3):
            return _reflect_walk(rng, rows)

    # A finite fallback keeps all legal count requests productive.
    if mode == "fold":
        rows = (0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4)
    else:
        rows = (0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1,
                2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3)
    return _reflect_walk(rng, rows)


def _embed_walk(rng, rows, background, trace):
    active_height = int(rows.max()) + 1
    length = len(rows)
    extra_height = min(int(rng.choice(6, p=(0.40, 0.30, 0.19, 0.07, 0.03, 0.01))), 30 - active_height)
    extra_width = min(int(rng.choice(7, p=(0.14, 0.22, 0.24, 0.18, 0.12, 0.07, 0.03))), 30 - length)
    top = int(rng.integers(extra_height + 1))
    left = int(rng.integers(extra_width + 1))
    grid = np.full((active_height + extra_height, length + extra_width), background, dtype=np.int8)
    grid[top + rows, left + np.arange(length)] = trace
    return grid


def _fill_spans(input_grid, background, fill):
    result = input_grid.copy()
    for row in range(input_grid.shape[0]):
        columns = np.flatnonzero(input_grid[row] != background)
        if len(columns) >= 2:
            between = result[row, int(columns[0]) + 1:int(columns[-1])]
            between[between == background] = fill
    return result


def _color_plan(rng, num_train, num_test):
    fill = 2 if rng.random() < 0.70 else int(rng.choice([0, 1, 3, 4, 5, 6, 7, 8, 9]))
    available = np.array([x for x in range(10) if x != fill], dtype=np.int8)
    rng.shuffle(available)
    background_count = int(rng.choice((1, 2, 3), p=(0.56, 0.32, 0.12)))
    backgrounds = tuple(int(x) for x in available[:background_count])
    trace_count = min(num_train, int(rng.integers(2, 4)))
    trace_palette = tuple(int(x) for x in available[background_count:background_count + trace_count])
    unseen_palette = tuple(int(x) for x in available[background_count + trace_count:])
    train_backgrounds = [backgrounds[0]]
    if background_count > 1:
        train_backgrounds.append(backgrounds[1])
    else:
        train_backgrounds.append(backgrounds[0])
    while len(train_backgrounds) < num_train:
        train_backgrounds.append(int(rng.choice(backgrounds)))
    rng.shuffle(train_backgrounds)
    test_backgrounds = [int(rng.choice(backgrounds)) for _ in range(num_test)]
    train_traces = [trace_palette[0], trace_palette[1]]
    while len(train_traces) < num_train:
        train_traces.append(int(rng.choice(trace_palette)))
    rng.shuffle(train_traces)
    test_traces = [int(rng.choice(trace_palette if rng.random() < 0.55 else unseen_palette))
                   for _ in range(num_test)]
    return fill, train_backgrounds, test_backgrounds, train_traces, test_traces


def _make_pair(rng, mode, background, trace, fill):
    rows = _make_walk(rng, mode)
    input_grid = _embed_walk(rng, rows, background, trace)
    return {"input": input_grid, "output": _fill_spans(input_grid, background, fill)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    fill, train_backgrounds, test_backgrounds, train_traces, test_traces = _color_plan(rng, num_train, num_test)
    train_modes = ["simple", "fold"]
    train_modes.extend(str(rng.choice(("simple", "fold", "extended"), p=(0.30, 0.50, 0.20)))
                       for _ in range(num_train - 2))
    rng.shuffle(train_modes)
    test_modes = [str(rng.choice(("simple", "fold", "extended"), p=(0.35, 0.45, 0.20)))
                  for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_walks = set()
    for split, modes, backgrounds, traces in (
        ("train", train_modes, train_backgrounds, train_traces),
        ("test", test_modes, test_backgrounds, test_traces),
    ):
        for mode, background, trace in zip(modes, backgrounds, traces):
            pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
            for _ in range(200):
                pair = _make_pair(pair_rng, mode, background, trace, fill)
                key = _grid_key(pair["input"])
                mask = pair["input"] == trace
                columns = np.flatnonzero(mask.any(axis=0))
                walk = np.argmax(mask[:, columns], axis=0)
                walk = _walk_signature(walk - int(walk.min()))
                if key not in used_inputs and walk not in used_walks and not _is_source_input(pair["input"]):
                    used_inputs.add(key)
                    used_walks.add(walk)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a distinct diagonal walk")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must have two dimensions")
    _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30, f"{label} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must use integer cells")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{label} has illegal colors")


def _analyze_pair(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape, "input and output shapes differ")
    colors, counts = np.unique(input_grid, return_counts=True)
    _require(len(colors) == 2 and counts[0] != counts[1], "input needs an identifiable background and one trace")
    background = int(colors[int(np.argmax(counts))])
    trace = int(colors[int(np.argmin(counts))])
    mask = input_grid == trace
    columns = np.flatnonzero(mask.any(axis=0))
    _require(len(columns) >= 5 and np.array_equal(columns, np.arange(columns[0], columns[-1] + 1)),
             "trace must occupy at least five consecutive columns")
    _require(np.all(mask[:, columns].sum(axis=0) == 1), "trace needs one cell per occupied column")
    rows = np.argmax(mask[:, columns], axis=0).astype(np.int64)
    _require(np.all(np.abs(np.diff(rows)) == 1), "trace steps must be diagonal units")
    active, intersections, gap, turns = _walk_features(rows - int(rows.min()))
    _require(active >= 3 and intersections >= 2, "trace needs a multirow fill span")
    changed = input_grid != output_grid
    _require(np.any(changed), "pair must visibly demonstrate filling")
    inserted = np.unique(output_grid[changed])
    _require(len(inserted) == 1, "only one fill color may be inserted")
    fill = int(inserted[0])
    _require(fill not in (background, trace), "fill must differ from input colors")
    _require(np.array_equal(output_grid, _fill_spans(input_grid, background, fill)),
             "output violates the row span rule")
    _require(not _is_source_input(input_grid), "generated pair copies a source pair")
    return {"background": background, "trace": trace, "fill": fill,
            "walk": _walk_signature(rows - int(rows.min())), "length": len(rows),
            "active_height": active, "intersections": intersections,
            "gap": gap, "turns": turns, "changed": int(changed.sum()),
            "shape": input_grid.shape}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle needs train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    analyses = {"train": [], "test": []}
    inputs, pairs, walks = set(), set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{split}[{index}] has wrong pair schema")
            input_grid, output_grid = pair["input"], pair["output"]
            _validate_grid(input_grid, f"{split}[{index}].input")
            _validate_grid(output_grid, f"{split}[{index}].output")
            information = _analyze_pair(input_grid, output_grid)
            input_key, output_key = _grid_key(input_grid), _grid_key(output_grid)
            _require(input_key not in inputs and (input_key, output_key) not in pairs,
                     "duplicate input or pair")
            _require(information["walk"] not in walks, "duplicate trace geometry within puzzle")
            inputs.add(input_key)
            pairs.add((input_key, output_key))
            walks.add(information["walk"])
            analyses[split].append(information)
    all_information = analyses["train"] + analyses["test"]
    _require(len({item["fill"] for item in all_information}) == 1, "fill color changes within puzzle")
    _require(len({item["trace"] for item in analyses["train"]}) >= 2,
             "training needs multiple trace colors")
    _require(any(item["intersections"] == 2 for item in analyses["train"]),
             "training needs a clear two-intersection trace")
    _require(any(item["intersections"] >= 3 and item["gap"] >= 3 for item in analyses["train"]),
             "training needs a substantial multi-intersection span")
    # A majority background is visible locally in every test, and each training
    # output reveals the same inserted color, including non-red conventions.
    for item in analyses["test"]:
        _require(item["background"] != item["fill"] and item["trace"] != item["fill"],
                 "test colors obscure the demonstrated fill convention")


def _puzzle_signature(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, fills, backgrounds = set(), set(), set(), set()
    shapes, walks, lengths, active_heights = set(), set(), set(), set()
    signatures = set()
    mixed_backgrounds = shared_backgrounds = familiar_tests = unseen_tests = 0
    simple_tests = folded_tests = deeper_train_than_test = 0
    tight_three = horizontal_tight = tall_active = long_walk = four_plus = five_plus = 0
    small_changes = uneven_rows = asymmetric_arms = blank_three = combined_regimes = 0
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_signature(puzzle))
        information = {split: [_analyze_pair(pair["input"], pair["output"])
                               for pair in puzzle[split]] for split in ("train", "test")}
        train_fills = {item["fill"] for item in information["train"]}
        _require(len(train_fills) == 1, "training fill convention is unclear")
        learned_fill = next(iter(train_fills))
        fills.add(learned_fill)
        train_traces = {item["trace"] for item in information["train"]}
        all_items = information["train"] + information["test"]
        background_set = {item["background"] for item in all_items}
        mixed_backgrounds += len(background_set) > 1
        shared_backgrounds += len(background_set) == 1
        combined_regimes += (len(background_set) > 1 and learned_fill != 2 and
                             any(item["trace"] in train_traces and item["turns"] >= 3
                                 for item in information["test"]))
        deeper_train_than_test += max(x["intersections"] for x in information["train"]) > max(
            x["intersections"] for x in information["test"])
        for item in information["test"]:
            familiar_tests += item["trace"] in train_traces
            unseen_tests += item["trace"] not in train_traces
            simple_tests += item["intersections"] == 2
            folded_tests += item["intersections"] >= 3
        for split in ("train", "test"):
            for pair, item in zip(puzzle[split], information[split]):
                grid = pair["input"]
                pair_count += 1
                # This prediction uses only observed training changes and the
                # test input's locally identifiable majority color.
                observed = np.unique(grid, return_counts=True)
                predicted_background = int(observed[0][np.argmax(observed[1])])
                prediction = _fill_spans(grid, predicted_background, learned_fill)
                _require(np.array_equal(prediction, pair["output"]), "training-based prediction failed")
                backgrounds.add(item["background"])
                shapes.add(item["shape"])
                walks.add(item["walk"])
                lengths.add(item["length"])
                active_heights.add(item["active_height"])
                tight_three += item["shape"][0] == 3
                horizontal_tight += item["shape"][1] == item["length"]
                tall_active += item["active_height"] >= 7
                long_walk += item["length"] >= 18
                four_plus += item["intersections"] >= 4
                five_plus += item["intersections"] >= 5
                small_changes += item["changed"] <= 2
                blank_three += item["shape"][0] - item["active_height"] >= 3
                trace_mask = grid == item["trace"]
                trace_columns = np.flatnonzero(trace_mask.any(axis=0))
                trace_rows = np.argmax(trace_mask[:, trace_columns], axis=0)
                steps = np.diff(trace_rows)
                turn_columns = np.flatnonzero(steps[1:] != steps[:-1]) + 1
                run_lengths = np.diff(np.r_[0, turn_columns, len(steps)])
                asymmetric_arms += len(run_lengths) > 1 and run_lengths[0] != run_lengths[-1]
                for row in range(grid.shape[0]):
                    crossings = np.flatnonzero(trace_mask[row])
                    if len(crossings) >= 3 and len(set(np.diff(crossings))) > 1:
                        uneven_rows += 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic pair counts lack diversity")
    _require(len(signatures) == 42, "seed changes failed to change puzzles")
    _require(2 in fills and len(fills) >= 3 and len(backgrounds) >= 6, "color roles lack diversity")
    _require(len(shapes) >= 30 and len(walks) >= 90 and len(lengths) >= 15,
             "shape and trace diversity is too narrow")
    _require(len(active_heights) >= 6 and tight_three > 0 and horizontal_tight > 0,
             "tight and tall canvas options are missing")
    _require(mixed_backgrounds >= 5 and shared_backgrounds >= 5,
             "background regimes are not both represented")
    _require(familiar_tests >= 8 and unseen_tests >= 8 and simple_tests >= 8 and folded_tests >= 8,
             "test color or geometry regimes lack coverage")
    _require(deeper_train_than_test >= 4 and tall_active >= 8 and long_walk >= 8,
             "difficulty or extent is too narrow")
    _require(four_plus >= 8 and five_plus > 0, "multi-intersection traces are missing")
    _require(five_plus < pair_count // 4, "beyond-source folds are too frequent")
    _require(uneven_rows >= 20 and asymmetric_arms >= 20,
             "turn positions and crossing distances lack variation")
    _require(blank_three < pair_count // 4 and combined_regimes > 0,
             "padding or combined color/geometry regimes lack coverage")
    _require(small_changes < len(walks) // 5, "weak demonstrations are too frequent")
    same = generate_puzzle(123456, 6, 3)
    _require(_puzzle_signature(same) == _puzzle_signature(generate_puzzle(123456, 6, 3)),
             "generation is nondeterministic")
    _require(_puzzle_signature(same) != _puzzle_signature(generate_puzzle(123457, 6, 3)),
             "neighboring seeds produced the same puzzle")
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + index, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                 "explicit counts were ignored")
        validate_generated_puzzle(puzzle)
    for index, (train, test) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + index, train, test)
        _require(train is None or len(puzzle["train"]) == train, "partial train count was ignored")
        _require(test is None or len(puzzle["test"]) == test, "partial test count was ignored")
        validate_generated_puzzle(puzzle)
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(4000, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count request accepted: {request}")
