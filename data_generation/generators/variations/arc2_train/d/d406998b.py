import numpy as np


_SOURCE_INPUTS = frozenset({
    ("0050050500050", "5000505005005", "0505000050500"),
    ("050500505000", "500050050050", "005005000505"),
    ("00500505050500", "50005000505005", "05050050000050"),
    ("5050050005", "0500500500", "0005005050"),
    ("00050005050050500", "50500500500500050", "05005050005005005"),
})
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))
_OTHER_PATHS = ("two_row", "runs", "periodic", "irregular")
_ALL_PATHS = ("hop", "flat") + _OTHER_PATHS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, rng: np.random.Generator,
                    num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        train = int(rng.integers(test + 1, 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _training_widths(rng: np.random.Generator, count: int) -> list[int]:
    widths = [int(rng.choice(np.arange(8, 20, 2))),
              int(rng.choice(np.arange(9, 20, 2)))]
    widths.extend(int(v) for v in rng.integers(6, 25, size=count - 2))
    rng.shuffle(widths)
    return widths


def _test_widths(seed: int, rng: np.random.Generator,
                 train_widths: list[int], count: int) -> list[int]:
    low, high = min(train_widths), max(train_widths)
    choices = {"short": list(range(5, low)),
               "within": list(range(low, high + 1)),
               "long": list(range(high + 1, 31))}
    regimes = ("short", "within", "long")
    widths: list[int] = []
    for index in range(count):
        candidates = [v for v in choices[regimes[(seed + index) % 3]] if v not in widths]
        if not candidates:
            candidates = choices[regimes[(seed + index) % 3]]
        if index == 1:
            opposite = [v for v in candidates if v % 2 != widths[0] % 2]
            if opposite:
                candidates = opposite
        else:
            parity = [v for v in candidates if v % 2 == (seed + index) % 2]
            if parity:
                candidates = parity
        widths.append(int(rng.choice(candidates)))
    return widths


def _marker_rows(rng: np.random.Generator, width: int,
                 height: int, mode: str) -> np.ndarray:
    if mode == "hop":
        rows = np.empty(width, dtype=np.int64)
        first = min(width, height)
        rows[:first] = rng.permutation(height)[:first]
        for col in range(first, width):
            possible = [row for row in range(height) if row != rows[col - 1]]
            rows[col] = int(rng.choice(possible))
        return rows
    if mode == "flat":
        return np.full(width, int(rng.integers(height)), dtype=np.int64)
    if mode == "two_row":
        two = rng.choice(height, size=2, replace=False)
        rows = rng.choice(two, size=width).astype(np.int64)
        rows[0], rows[1] = int(two[0]), int(two[1])
        return rows
    if mode == "runs":
        rows = np.empty(width, dtype=np.int64)
        current = int(rng.integers(height))
        col = 0
        while col < width:
            length = 3 if col == 0 else int(rng.integers(2, 7))
            stop = min(width, col + length)
            rows[col:stop] = current
            col = stop
            if col < width:
                current = (current + int(rng.integers(1, height))) % height
        return rows
    if mode == "periodic":
        period = int(rng.integers(3, min(6, width) + 1))
        template = rng.integers(0, height, size=period, dtype=np.int64)
        template[1] = (template[0] + 1) % height
        return np.resize(template, width)
    if mode == "irregular":
        return rng.integers(0, height, size=width, dtype=np.int64)
    raise AssertionError("unknown path mode")


def _digit_rows(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _make_pair(rng: np.random.Generator, width: int, height: int,
               mode: str, background: int, marker: int, highlight: int) -> dict:
    rows = _marker_rows(rng, width, height, mode)
    columns = np.arange(width)
    source = np.full((height, width), background, dtype=np.int8)
    source[rows, columns] = marker
    if _digit_rows(source) in _SOURCE_INPUTS:
        rows = columns % height
        source[:] = background
        source[rows, columns] = marker
    target = source.copy()
    chosen = np.arange(width - 1, -1, -2)
    target[rows[chosen], chosen] = highlight
    return {"input": source, "output": target}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    highlight = int(rng.integers(10))
    pool = rng.permutation([color for color in range(10) if color != highlight])
    total = train_count + test_count
    roles = [(int(pool[i % 9]), int(pool[(i + 1 + i // 9) % 9]))
             for i in range(total)]
    train_widths = _training_widths(rng, train_count)
    test_widths = _test_widths(seed, rng, train_widths, test_count)
    train_heights = [3, int(rng.integers(4, 9))]
    train_heights.extend(int(v) for v in rng.integers(3, 9, size=train_count - 2))
    test_heights = [int(v) for v in rng.integers(3, 9, size=test_count)]
    train_modes = ["hop", "flat"]
    train_modes.extend(str(rng.choice(_ALL_PATHS)) for _ in range(train_count - 2))
    test_modes = [str(v) for v in rng.permutation(_ALL_PATHS)[:test_count]]
    pairs = []
    for index, (width, height, mode) in enumerate(zip(
            train_widths + test_widths,
            train_heights + test_heights,
            train_modes + test_modes)):
        child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        background, marker = roles[index]
        pairs.append(_make_pair(child, width, height, mode,
                                background, marker, highlight))
    puzzle = {"train": pairs[:train_count], "test": pairs[train_count:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.tobytes())


def _analyze(pair: dict) -> dict:
    source, target = pair["input"], pair["output"]
    _require(source.shape == target.shape, "input and output shapes differ")
    height, width = source.shape
    _require(3 <= height <= 8 and 5 <= width <= 30, "unsupported grid size")
    colors, counts = np.unique(source, return_counts=True)
    _require(len(colors) == 2, "input must have background and marker colors")
    background = int(colors[int(np.argmax(counts))])
    marker = int(colors[int(np.argmin(counts))])
    _require(int(np.sum(source == marker)) == width, "marker count must equal width")
    _require(np.all(np.sum(source == marker, axis=0) == 1),
             "each column must contain one marker")
    new_colors = set(int(v) for v in np.unique(target)) - {background, marker}
    _require(len(new_colors) == 1, "output must introduce one highlight")
    highlight = next(iter(new_colors))
    rows = np.argmax(source == marker, axis=0)
    expected = source.copy()
    chosen = np.arange(width - 1, -1, -2)
    expected[rows[chosen], chosen] = highlight
    _require(np.array_equal(target, expected),
             "right-anchored alternating recoloring failed")
    return {"height": height, "width": width, "background": background,
            "marker": marker, "highlight": highlight,
            "rows": tuple(int(v) for v in rows)}


def _solve_from_examples(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    highlights = []
    for pair in train:
        before = set(int(v) for v in np.unique(pair["input"]))
        added = set(int(v) for v in np.unique(pair["output"])) - before
        _require(len(added) == 1, "ambiguous training highlight")
        highlights.append(next(iter(added)))
    _require(len(set(highlights)) == 1, "training highlight is inconsistent")
    colors, counts = np.unique(test_input, return_counts=True)
    _require(len(colors) == 2 and counts[0] != counts[1], "ambiguous input roles")
    marker = int(colors[int(np.argmin(counts))])
    rows = np.argmax(test_input == marker, axis=0)
    chosen = np.arange(test_input.shape[1] - 1, -1, -2)
    answer = test_input.copy()
    answer[rows[chosen], chosen] = highlights[0]
    return answer


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    analyses = {"train": [], "test": []}
    inputs: set[tuple] = set()
    pairs: set[tuple] = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integral")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         "grid dimensions out of ARC bounds")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid color out of ARC bounds")
            inp = _signature(pair["input"])
            both = (inp, _signature(pair["output"]))
            _require(inp not in inputs and both not in pairs, "duplicate pair or input")
            inputs.add(inp)
            pairs.add(both)
            _require(_digit_rows(pair["input"]) not in _SOURCE_INPUTS,
                     "source input copied")
            analyses[split].append(_analyze(pair))
    training = analyses["train"]
    all_pairs = training + analyses["test"]
    _require(len({a["highlight"] for a in all_pairs}) == 1,
             "highlight changes within the puzzle")
    _require({a["width"] % 2 for a in training} == {0, 1},
             "training must show both right-edge phases")
    _require(any(a["height"] == 3 for a in training)
             and any(a["height"] >= 4 for a in training),
             "training must show height transfer")
    _require(len({a["background"] for a in training}) >= 2
             and len({a["marker"] for a in training}) >= 2,
             "training must vary visible input color roles")
    _require(any(all(x != y for x, y in zip(a["rows"], a["rows"][1:]))
                 for a in training), "training needs a hopping path")
    _require(any(len(set(a["rows"])) == 1 for a in training),
             "training needs a contrasting flat path")
    for pair in test:
        _require(np.array_equal(_solve_from_examples(train, pair["input"]), pair["output"]),
                 "test is not recoverable from training and input")


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_signature(p["input"]), _signature(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    for seed, train_count, test_count in ((73, None, None), (8675309, 7, 3),
                                          (1000, 2, 1), (1001, 10, 1),
                                          (1002, 10, 4), (1100, None, 1),
                                          (1101, None, 4), (1102, 2, None),
                                          (1103, 10, None)):
        first = generate_puzzle(seed, train_count, test_count)
        second = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(first)
        _require(_puzzle_signature(first) == _puzzle_signature(second),
                 "generation is not deterministic")
        if train_count is not None:
            _require(len(first["train"]) == train_count, "train count changed")
        if test_count is not None:
            _require(len(first["test"]) == test_count, "test count changed")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 2, "num_test": 3}, {"num_train": 4, "num_test": 4},
                      {"num_train": 2.0}, {"num_train": True}):
        try:
            generate_puzzle(1200, **arguments)
        except ValueError:
            continue
        else:
            raise AssertionError(f"invalid counts accepted: {arguments}")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42,
             "cross-seed puzzles repeat")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts collapsed")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts collapsed")
    observed = []
    relationships = set()
    coverage = {name: 0 for name in (
        "even_test", "short_test", "within_test", "long_test",
        "one_row", "two_row", "hopping", "long_run", "periodic",
        "irregular", "tall_hopping", "three_row_flat", "tall_two_row",
        "tall_irregular", "multiple_test_relations")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_analyze(pair) for pair in puzzle["train"]]
        low, high = min(a["width"] for a in train), max(a["width"] for a in train)
        puzzle_test_relations = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair)
                observed.append(a)
                rows, width, height = a["rows"], a["width"], a["height"]
                row_span = len(set(rows))
                hopping = all(x != y for x, y in zip(rows, rows[1:]))
                run = 1
                longest = 1
                for x, y in zip(rows, rows[1:]):
                    run = run + 1 if x == y else 1
                    longest = max(longest, run)
                periodic = row_span >= 2 and any(
                    all(rows[i] == rows[i % period] for i in range(width))
                    for period in range(2, min(6, width)))
                irregular = row_span >= 3 and not hopping and not periodic and longest < 5
                for name, condition in (
                    ("one_row", row_span == 1), ("two_row", row_span == 2),
                    ("hopping", hopping), ("long_run", longest >= 5),
                    ("periodic", periodic), ("irregular", irregular),
                    ("tall_hopping", height >= 4 and hopping),
                    ("three_row_flat", height == 3 and row_span == 1),
                    ("tall_two_row", height >= 4 and row_span == 2),
                    ("tall_irregular", height >= 4 and irregular)):
                    coverage[name] += bool(condition)
                if split == "test":
                    relation = "short" if width < low else "long" if width > high else "within"
                    relationships.add(relation)
                    puzzle_test_relations.add(relation)
                    coverage[relation + "_test"] += 1
                    coverage["even_test"] += width % 2 == 0
        coverage["multiple_test_relations"] += len(puzzle_test_relations) > 1
    _require(relationships == {"short", "within", "long"},
             "test width relationships collapsed")
    _require({a["width"] % 2 for a in observed} == {0, 1}, "width parity collapsed")
    _require(len({a["width"] for a in observed}) >= 20, "width range too narrow")
    _require({a["height"] for a in observed} == set(range(3, 9)),
             "height range too narrow")
    _require({a["highlight"] for a in observed} == set(range(10)),
             "highlight assignments too narrow")
    _require({len(set(a["rows"])) for a in observed}.issuperset({1, 2, 3}),
             "marker-row spans too narrow")
    _require(len({(a["height"], a["width"], a["rows"]) for a in observed}) >= 250,
             "structural layouts too repetitive")
    for name, minimum in (("even_test", 10), ("short_test", 5),
                          ("within_test", 5), ("long_test", 5),
                          ("one_row", 20), ("two_row", 10),
                          ("hopping", 20), ("long_run", 20),
                          ("periodic", 10), ("irregular", 10),
                          ("tall_hopping", 5), ("three_row_flat", 1),
                          ("tall_two_row", 5), ("tall_irregular", 5),
                          ("multiple_test_relations", 5)):
        _require(coverage[name] >= minimum, f"insufficient {name} coverage")
