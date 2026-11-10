"""Solved ARC strips built from an indexed spine of two-cell dominoes."""

import numpy as np

_SOURCE = ((13, True), (11, False), (11, True),
           (14, False), (10, True), (17, False))


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(rng, num_train, num_test):
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt is None and ne is None:
        options = [(a, b) for a in range(2, 11) for b in range(1, 5) if a > b]
        return options[int(rng.integers(len(options)))]
    if nt is None:
        nt = int(rng.integers(max(2, ne + 1), 11))
    if ne is None:
        ne = int(rng.integers(1, min(4, nt - 1) + 1))
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    return nt, ne


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _pair_key(pair):
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _make_sides(rng, length, first_side, runs):
    # A cut repeats the preceding side; all other positions alternate.
    cuts = set()
    if runs:
        cut_count = int(rng.integers(1, min(3, length // 4) + 1))
        cuts = set(int(x) for x in rng.choice(np.arange(1, length),
                                               size=cut_count, replace=False))
    sides = [bool(first_side)]
    for index in range(1, length):
        sides.append(sides[-1] if index in cuts else not sides[-1])
    return tuple(sides)


def _build(length, sides, orientation, background, structure, highlight, period, offset):
    shape = (3, length) if orientation == "H" else (length, 3)
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for index, first_side in enumerate(sides):
        spine = (1, index) if orientation == "H" else (index, 1)
        tooth = ((0 if first_side else 2), index) if orientation == "H" else (index, (0 if first_side else 2))
        inp[spine] = inp[tooth] = structure
        out[spine] = out[tooth] = highlight if index % period == offset else structure
    return {"input": inp, "output": out}


def _source_keys():
    keys = set()
    for length, first in _SOURCE:
        sides = tuple(first if i % 2 == 0 else not first for i in range(length))
        keys.add(_pair_key(_build(length, sides, "H", 0, 4, 6, 3, 0)))
    return keys


_SOURCE_KEYS = _source_keys()


def _grid_check(grid, label):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
             f"{label}: expected a two-dimensional NumPy array")
    _require(all(1 <= x <= 30 for x in grid.shape), f"{label}: invalid shape")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label}: expected integer cells")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label}: invalid color")


def _input_geometry(inp, label):
    height, width = inp.shape
    if height == 3 and width >= 7:
        orientation, length = "H", width
        spine = inp[1, :]
        near, far = inp[0, :], inp[2, :]
    elif width == 3 and height >= 7:
        orientation, length = "V", height
        spine = inp[:, 1]
        near, far = inp[:, 0], inp[:, 2]
    else:
        raise AssertionError(f"{label}: expected a three-cell-wide spine strip")
    structure = int(spine[0])
    _require(bool(np.all(spine == structure)), f"{label}: spine is not uniform")
    colors = {int(x) for x in np.unique(inp)}
    _require(len(colors) == 2 and structure in colors, f"{label}: input roles are invalid")
    background = next(iter(colors - {structure}))
    sides = tuple(bool(x == structure) for x in near)
    _require(all((int(a == structure) + int(b == structure)) == 1
                 for a, b in zip(near, far)), f"{label}: expected one tooth per spine cell")
    _require(set(sides) == {False, True}, f"{label}: teeth occupy only one side")
    runs = any(sides[i] == sides[i - 1] for i in range(1, length))
    run_length = 1
    for i in range(1, length):
        run_length = run_length + 1 if sides[i] == sides[i - 1] else 1
        _require(run_length <= 4, f"{label}: same-side run is too long")
    return orientation, length, background, structure, sides, runs


def _analyze(pair, label):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             f"{label}: invalid pair schema")
    inp, out = pair["input"], pair["output"]
    _grid_check(inp, f"{label}.input")
    _grid_check(out, f"{label}.output")
    _require(inp.shape == out.shape, f"{label}: changed dimensions")
    orientation, length, background, structure, sides, runs = _input_geometry(inp, label)
    changed = inp != out
    selected = tuple(int(i) for i in np.flatnonzero(changed[1, :] if orientation == "H" else changed[:, 1]))
    _require(len(selected) >= 3, f"{label}: expected three selected dominoes")
    period = selected[1] - selected[0]
    offset = selected[0]
    _require(period in (3, 4, 5) and 0 <= offset < period,
             f"{label}: unsupported period or offset")
    _require(selected == tuple(range(offset, length, period)),
             f"{label}: selected positions do not continue periodically")
    highlight = int(out[1, offset] if orientation == "H" else out[offset, 1])
    _require(highlight not in (background, structure), f"{label}: highlight is not output-only")
    expected = _build(length, sides, orientation, background, structure,
                      highlight, period, offset)["output"]
    _require(np.array_equal(out, expected), f"{label}: incorrect domino recoloring")
    _require({int(x) for x in np.unique(out)} == {background, structure, highlight},
             f"{label}: incorrect output roles")
    _require(_pair_key(pair) not in _SOURCE_KEYS, f"{label}: exact source copy")
    return {"orientation": orientation, "length": length, "background": background,
            "structure": structure, "highlight": highlight, "period": period,
            "offset": offset, "first_side": sides[0], "runs": runs,
            "sides": sides}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            item = _analyze(pair, f"{split}[{index}]")
            info[split].append(item)
            pair_key = _pair_key(pair)
            _require(pair_key[0] not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(pair_key[0])
            seen_pairs.add(pair_key)
    all_info = info["train"] + info["test"]
    conventions = {(x["highlight"], x["period"], x["offset"]) for x in all_info}
    _require(len(conventions) == 1, "highlight or periodic rule changed within puzzle")
    train = info["train"]
    test = info["test"]
    _require(all(x["length"] <= 26 for x in train), "training strip too long")
    _require(len({x["length"] for x in train}) >= 2, "training lacks length variation")
    _require(len({x["length"] % x["period"] for x in train}) >= 2,
             "training lacks different partial endings")
    _require({x["first_side"] for x in train} == {False, True},
             "training lacks both starting tooth sides")
    trained_orientations = {x["orientation"] for x in train}
    _require(all(x["orientation"] in trained_orientations for x in test),
             "test orientation absent from training")
    if any(x["runs"] for x in test):
        _require(any(x["runs"] for x in train), "test runs absent from training")
    if any(not x["runs"] for x in test):
        _require(any(not x["runs"] for x in train), "test alternation absent from training")


def _train_lengths(rng, count, minimum, period):
    pool = list(range(minimum, 27))
    first = int(rng.choice(pool))
    second_pool = [n for n in pool if n % period != first % period]
    second = int(rng.choice(second_pool))
    remaining = [n for n in pool if n not in (first, second)]
    extras = [int(x) for x in rng.choice(remaining, size=count - 2, replace=False)]
    return [first, second] + extras


def _test_length(rng, train_lengths, minimum):
    low, high = min(train_lengths), max(train_lengths)
    modes = []
    if minimum < low:
        modes.append((minimum, low - 1))
    modes.append((low, high))
    if high < 30:
        modes.append((high + 1, 30))
    start, stop = modes[int(rng.integers(len(modes)))]
    return int(rng.integers(start, stop + 1))


def _colors(rng, highlight):
    available = [c for c in range(10) if c != highlight]
    two = rng.choice(available, size=2, replace=False)
    return int(two[0]), int(two[1])


def _choose_rule(rng):
    # Most puzzles retain the source rule; other seeds teach a related
    # constant period and offset through three visible repetitions.
    if rng.random() < 0.5:
        return 3, 0
    period = int(rng.choice((3, 4, 5), p=(0.2, 0.5, 0.3)))
    offset = int(rng.integers(period))
    return period, offset


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    nt, ne = _counts(rng, num_train, num_test)
    period, offset = _choose_rule(rng)
    highlight = int(rng.integers(10))
    minimum = max(7, offset + 2 * period + 1)
    train_lengths = _train_lengths(rng, nt, minimum, period)
    orientation_mode = int(rng.choice(3, p=(0.45, 0.15, 0.40)))
    run_mode = bool(rng.integers(2))
    varied_colors = bool(rng.random() < 0.75)
    first_colors = _colors(rng, highlight)
    second_colors = first_colors
    if varied_colors:
        if rng.random() < 0.5:
            second_colors = first_colors[::-1]
        else:
            for _ in range(20):
                second_colors = _colors(rng, highlight)
                if second_colors != first_colors:
                    break
    train_specs = []
    for index, length in enumerate(train_lengths):
        if orientation_mode == 0:
            orientation = "H"
        elif orientation_mode == 1:
            orientation = "V"
        else:
            orientation = ("H", "V")[index] if index < 2 else ("H" if rng.integers(2) else "V")
        runs = run_mode and (index == 1 or (index > 1 and bool(rng.integers(2))))
        colors = first_colors if index == 0 else second_colors if index == 1 else (
            _colors(rng, highlight) if varied_colors else first_colors)
        first_side = True if index == 0 else False if index == 1 else bool(rng.integers(2))
        train_specs.append((length, orientation, runs, first_side, colors))
    rng.shuffle(train_specs)
    puzzle = {"train": [], "test": []}
    used_inputs = set()

    def add(split, length, orientation, runs, first_side, colors):
        # Preserve the demonstrated side, orientation, and length even if a
        # palette collides with an earlier input or an exact source pair.
        alternate_colors = [(a, b) for a in range(10) if a != highlight
                            for b in range(10) if b not in (a, highlight)
                            and (a, b) != colors]
        for candidate_colors in [colors] + alternate_colors:
            candidate_sides = _make_sides(rng, length, first_side, runs)
            candidate = _build(length, candidate_sides, orientation,
                               candidate_colors[0], candidate_colors[1],
                               highlight, period, offset)
            key = _pair_key(candidate)
            if key[0] not in used_inputs and key not in _SOURCE_KEYS:
                used_inputs.add(key[0])
                puzzle[split].append(candidate)
                return
        raise AssertionError("unable to construct a distinct pair")

    for length, orientation, runs, first_side, colors in train_specs:
        add("train", length, orientation, runs, first_side, colors)
    train_info = [_analyze(pair, "generated train") for pair in puzzle["train"]]
    orientations = [x["orientation"] for x in train_info]
    morphologies = [x["runs"] for x in train_info]
    for _ in range(ne):
        length = _test_length(rng, train_lengths, minimum)
        orientation = orientations[int(rng.integers(len(orientations)))]
        runs = morphologies[int(rng.integers(len(morphologies)))]
        colors = _colors(rng, highlight) if varied_colors else first_colors
        add("test", length, orientation, runs, bool(rng.integers(2)), colors)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple(tuple(_pair_key(p) for p in puzzle[s]) for s in ("train", "test"))


def _predict_from_examples(train, test_input):
    # Uses only visible training pairs and a copied test input.
    examples = [_analyze(p, "prediction example") for p in train]
    rules = {(x["period"], x["offset"], x["highlight"]) for x in examples}
    _require(len(rules) == 1, "inconsistent prediction examples")
    period, offset, highlight = next(iter(rules))
    orientation, length, background, structure, sides, _ = _input_geometry(test_input, "prediction input")
    return _build(length, sides, orientation, background, structure,
                  highlight, period, offset)["output"]


def self_test() -> None:
    observations = []
    signatures = set()
    train_counts, test_counts = set(), set()
    test_length_regimes = set()
    orientation_regimes = set()
    varied_color_puzzles = 0
    swapped_color_puzzles = 0
    fixed_color_puzzles = 0
    source_role_seen = False
    source_width_phase_seen = False
    one_test_count = 0
    old_signature_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        sig = _signature(puzzle)
        signatures.add(sig)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        _require(sig == _signature(generate_puzzle(seed)), "non-deterministic puzzle")
        train_copy = [{"input": p["input"].copy(), "output": p["output"].copy()}
                      for p in puzzle["train"]]
        for pair in puzzle["test"]:
            predicted = _predict_from_examples(train_copy, pair["input"].copy())
            _require(np.array_equal(predicted, pair["output"]), "test cannot be predicted from training")
        train_info = [_analyze(p, "diversity train") for p in puzzle["train"]]
        test_info = [_analyze(p, "diversity test") for p in puzzle["test"]]
        observations.append(train_info + test_info)
        low = min(x["length"] for x in train_info)
        high = max(x["length"] for x in train_info)
        test_length_regimes.update("shorter" if x["length"] < low else
                                   "longer" if x["length"] > high else "within"
                                   for x in test_info)
        orientation_regimes.add(tuple(sorted({x["orientation"] for x in train_info})))
        role_pairs = {(x["background"], x["structure"]) for x in train_info}
        varied_color_puzzles += len(role_pairs) > 1
        fixed_color_puzzles += len(role_pairs) == 1
        swapped_color_puzzles += any((b, a) in role_pairs for a, b in role_pairs)
        source_role_seen |= any(x["background"] == 0 and x["structure"] == 4
                                for x in train_info + test_info)
        source_width_phase_seen |= any(x["orientation"] == "H" and x["length"] == 17
                                       and not x["first_side"] for x in test_info)
        if len(test_info) == 1:
            one_test_count += 1
            x = test_info[0]
            old_signature_count += x["length"] >= 20 and not x["first_side"] and x["length"] % 3 == 2
    _require(len(signatures) >= 40, "too few distinct puzzles")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic counts lack diversity")
    flat = [x for group in observations for x in group]
    _require({x["orientation"] for x in flat} == {"H", "V"}, "orientation diversity missing")
    _require(orientation_regimes == {("H",), ("V",), ("H", "V")},
             "horizontal, vertical, and mixed puzzles are not all represented")
    _require({x["runs"] for x in flat} == {False, True}, "tooth-run diversity missing")
    _require({x["period"] for x in flat} == {3, 4, 5}, "period diversity missing")
    _require(len({(x["period"], x["offset"]) for x in flat}) >= 8,
             "period and offset combinations are too narrow")
    _require(len({x["length"] for x in flat}) >= 20, "length diversity missing")
    _require({x["length"] % 3 for x in flat if x["period"] == 3} == {0, 1, 2},
             "ending diversity missing")
    _require(test_length_regimes == {"shorter", "within", "longer"},
             "test length regimes are incomplete")
    _require(varied_color_puzzles >= 15 and swapped_color_puzzles >= 5 and
             fixed_color_puzzles >= 2, "within-puzzle role variation is too narrow")
    _require(source_role_seen and source_width_phase_seen,
             "source role or width/phase configuration is blanket-excluded")
    _require(old_signature_count < one_test_count, "one-test signature is prescribed")
    _require(any(x["orientation"] == "V" and x["runs"] for x in flat),
             "combined orientation and run variation missing")
    for index, (nt, ne) in enumerate(((2, 1), (10, 1), (10, 4),
                                      (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + index, nt, ne)
        validate_generated_puzzle(puzzle)
        _require((nt is None or len(puzzle["train"]) == nt) and
                 (ne is None or len(puzzle["test"]) == ne), "requested count ignored")
        _require(_signature(puzzle) == _signature(generate_puzzle(1000 + index, nt, ne)),
                 "requested counts are non-deterministic")
    for nt, ne in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(7, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
