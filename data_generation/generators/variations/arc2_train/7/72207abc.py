import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_integer(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (not _is_integer(value) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif num_train is None:
        test = int(num_test)
        train = max(2, test + 1) + seed % (11 - max(2, test + 1))
    elif num_test is None:
        train = int(num_train)
        test = 1 + seed % min(4, train - 1)
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _triangular(index: int) -> int:
    return index * (index + 1) // 2


def _signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_signature(pair: dict) -> tuple:
    return (_signature(pair["input"]), _signature(pair["output"]))


def _make_pair(
    length: int,
    breadth: int,
    lane_index: int,
    start: int,
    background: int,
    colors: tuple[int, ...],
    prefix_count: int,
    vertical: bool,
) -> dict:
    shape = (length, breadth) if vertical else (breadth, length)
    source = np.full(shape, background, dtype=np.int8)
    target = source.copy()
    index = 0
    while start + _triangular(index) < length:
        position = start + _triangular(index)
        cell = (position, lane_index) if vertical else (lane_index, position)
        value = colors[index % len(colors)]
        target[cell] = value
        if index < prefix_count:
            source[cell] = value
        index += 1
    return {"input": source, "output": target}


_SOURCE_PAIRS = frozenset(
    _pair_signature(_make_pair(length, 3, 1, 0, 0, colors, len(colors), False))
    for length, colors in (
        (19, (6, 8, 1)),
        (29, (2, 3)),
        (19, (1, 2)),
        (29, (2, 1, 8)),
    )
)


def _pick_pair(
    rng: np.random.Generator,
    period: int,
    background: int,
    vertical: bool,
    used_inputs: set[tuple],
    *,
    source_like: bool = False,
    long_prefix: bool = False,
    output_count_range: tuple[int, int] | None = None,
    familiar_length: int | None = None,
) -> dict:
    prefix_count = 4 if long_prefix else period
    minimum_output_count = max(prefix_count + 2, 5)
    for attempt in range(100):
        if source_like:
            breadth, lane_index, start = 3, 1, 0
        else:
            breadth = int(rng.integers(1, 7))
            lane_index = int(rng.integers(0, breadth))
            if familiar_length is None:
                start = int(rng.integers(0, 5))
            else:
                maximum_start = min(4, familiar_length - _triangular(minimum_output_count - 1) - 1)
                if maximum_start < 0:
                    raise ValueError("Familiar lane is too short for its seed")
                start = int(rng.integers(0, maximum_start + 1))
        if familiar_length is None:
            low, high = output_count_range or (minimum_output_count, 8)
            low = max(low, minimum_output_count)
            feasible = [n for n in range(low, high + 1) if start + _triangular(n - 1) < 30]
            if not feasible:
                continue
            output_count = int(feasible[int(rng.integers(0, len(feasible)))])
            last = start + _triangular(output_count - 1)
            next_position = start + _triangular(output_count)
            largest_tail = min(29 - last, next_position - last - 1)
            tail = int(rng.integers(0, largest_tail + 1))
            length = last + 1 + tail
        else:
            length = familiar_length
            if start + _triangular(minimum_output_count - 1) >= length:
                continue
        choices = [color for color in range(10) if color != background]
        colors = tuple(int(x) for x in rng.choice(choices, size=period, replace=False))
        pair = _make_pair(length, breadth, lane_index, start, background, colors, prefix_count, vertical)
        if _signature(pair["input"]) not in used_inputs and _pair_signature(pair) not in _SOURCE_PAIRS:
            used_inputs.add(_signature(pair["input"]))
            return pair
    raise ValueError("Could not construct a unique pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    # The rule is shared by the puzzle. Background and geometry are visible
    # properties of each pair; no hidden color map or direction is sampled.
    first_background = 0 if seed % 3 == 0 else int(rng.integers(0, 10))
    second_background = int(rng.choice([x for x in range(10) if x != first_background]))
    recombination = seed % 4 == 0
    first_period = 3 if recombination or seed % 2 == 0 else 2
    second_period = 2 if first_period == 3 else 3
    first_long_prefix = first_period == 2 and seed % 5 == 1
    second_long_prefix = second_period == 2 and seed % 5 == 2
    used_inputs: set[tuple] = set()
    train = [
        _pick_pair(
            rng, first_period, first_background, False, used_inputs,
            source_like=seed % 3 == 0,
            long_prefix=first_long_prefix,
            output_count_range=(5, 6) if recombination else None,
        ),
        _pick_pair(
            rng, second_period, second_background,
            vertical=(not recombination and seed % 2 == 0),
            used_inputs=used_inputs,
            long_prefix=second_long_prefix,
            output_count_range=(7, 8) if recombination else None,
        ),
    ]
    for index in range(2, train_count):
        period = int(rng.integers(2, 6))
        background = int(rng.integers(0, 10))
        vertical = bool(rng.integers(0, 2))
        long_prefix = period == 2 and bool(rng.integers(0, 4) == 0)
        train.append(_pick_pair(rng, period, background, vertical, used_inputs,
                                long_prefix=long_prefix))

    train_info = [_inspect_pair(pair, "training pair") for pair in train]
    seen_directions = [info["vertical"] for info in train_info]
    long_prefix_demonstrated = any(info["prefix_count"] > info["period"] for info in train_info)
    test = []
    for index in range(test_count):
        mode = seed % 4
        if mode == 0:
            period = 3
            vertical = False
            length = None
            count_range = (7, 8)
        elif mode == 1:
            period = int(rng.integers(4, 6))
            vertical = seen_directions[int(rng.integers(0, len(seen_directions)))]
            length = None
            count_range = None
        elif mode == 2:
            period = second_period
            vertical = seen_directions[int(rng.integers(0, len(seen_directions)))]
            length = train_info[0]["length"]
            count_range = None
        else:
            period = int(rng.integers(2, 6))
            vertical = seen_directions[int(rng.integers(0, len(seen_directions)))]
            length = None
            count_range = None
        background = int(rng.integers(0, 10))
        long_prefix = (period == 2 and long_prefix_demonstrated
                       and (length is None or length >= 16)
                       and bool(rng.integers(0, 3) == 0))
        test.append(_pick_pair(rng, period, background, vertical, used_inputs,
                               long_prefix=long_prefix,
                               output_count_range=count_range,
                               familiar_length=length))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: object, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} is not an array")
    _require(grid.ndim == 2, f"{label} is not two-dimensional")
    _require(all(1 <= size <= 30 for size in grid.shape), f"{label} has invalid dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} is not integer")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{label} has invalid colors")


def _inspect_pair(pair: dict, label: str) -> dict:
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label} schema")
    source, target = pair["input"], pair["output"]
    _validate_grid(source, label + ".input")
    _validate_grid(target, label + ".output")
    _require(source.shape == target.shape and source.dtype == target.dtype, f"{label} shape or dtype")
    vertical = source.shape[0] > source.shape[1]
    length = source.shape[0] if vertical else source.shape[1]
    breadth = source.shape[1] if vertical else source.shape[0]
    _require(11 <= length <= 30 and 1 <= breadth <= 6, f"{label} lane geometry")
    counts = np.bincount(source.reshape(-1), minlength=10)
    background = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[background]) == 1, f"{label} ambiguous background")
    active = np.argwhere(source != background)
    _require(len(active) >= 2, f"{label} insufficient seed")
    lane_index = int(active[0, 1] if vertical else active[0, 0])
    _require(all(int(cell[1] if vertical else cell[0]) == lane_index for cell in active), f"{label} multiple lanes")
    lane = source[:, lane_index] if vertical else source[lane_index, :]
    positions = [int(x) for x in np.flatnonzero(lane != background)]
    start = positions[0]
    _require(positions == [start + _triangular(i) for i in range(len(positions))], f"{label} seed gaps")
    visible = tuple(int(lane[pos]) for pos in positions)
    possible_periods = [p for p in range(1, len(visible) + 1)
                        if all(visible[i] == visible[i % p] for i in range(len(visible)))]
    period = possible_periods[0]
    _require(2 <= period <= 5, f"{label} invalid color period")
    colors = visible[:period]
    _require(len(set(colors)) == period, f"{label} repeated seed colors")
    _require(len(visible) == period or (period == 2 and len(visible) == 4), f"{label} unsupported prefix")
    expected = _make_pair(length, breadth, lane_index, start, background, colors,
                          len(visible), vertical)
    _require(np.array_equal(source, expected["input"]), f"{label} malformed input")
    _require(np.array_equal(target, expected["output"]), f"{label} wrong continuation")
    output_count = sum(start + _triangular(i) < length for i in range(9))
    _require(output_count >= len(visible) + 2, f"{label} too few additions")
    _require(_pair_signature(pair) not in _SOURCE_PAIRS, f"{label} copies source")
    return {"vertical": vertical, "length": length, "breadth": breadth,
            "lane_index": lane_index, "start": start, "background": background,
            "period": period, "prefix_count": len(visible), "output_count": output_count,
            "tail": length - 1 - start - _triangular(output_count - 1)}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    train_info, test_info = [], []
    input_signatures, pair_signatures = set(), set()
    for section, pairs, destination in (("train", train, train_info), ("test", test, test_info)):
        for index, pair in enumerate(pairs):
            destination.append(_inspect_pair(pair, f"{section}[{index}]"))
            input_signature = _signature(pair["input"])
            pair_signature = _pair_signature(pair)
            _require(input_signature not in input_signatures, "duplicate input")
            _require(pair_signature not in pair_signatures, "duplicate pair")
            input_signatures.add(input_signature)
            pair_signatures.add(pair_signature)
    _require({2, 3}.issubset({info["period"] for info in train_info}), "train must demonstrate two and three colors")
    train_orientations = {info["vertical"] for info in train_info}
    _require(all(info["vertical"] in train_orientations for info in test_info), "unseen test direction")
    if any(info["prefix_count"] > info["period"] for info in test_info):
        _require(any(info["prefix_count"] > info["period"] for info in train_info), "unseen long prefix")
    _require(len({info["background"] for info in train_info}) >= 2, "training backgrounds do not vary")


def _equal_puzzles(a: dict, b: dict) -> bool:
    return (all(len(a[section]) == len(b[section]) for section in ("train", "test"))
            and all(np.array_equal(pa[key], pb[key]) for section in ("train", "test")
               for pa, pb in zip(a[section], b[section]) for key in ("input", "output"))
            )


def _independent_rule_check(pair: dict) -> None:
    before, after = pair["input"], pair["output"]
    background = int(np.bincount(before.reshape(-1), minlength=10).argmax())
    vertical = before.shape[0] > before.shape[1]
    cells = np.argwhere(before != background)
    lane_index = int(cells[0, 1] if vertical else cells[0, 0])
    before_lane = before[:, lane_index] if vertical else before[lane_index, :]
    after_lane = after[:, lane_index] if vertical else after[lane_index, :]
    old_positions = np.flatnonzero(before_lane != background).tolist()
    new_positions = np.flatnonzero(after_lane != background).tolist()
    old_colors = [int(before_lane[position]) for position in old_positions]
    new_colors = [int(after_lane[position]) for position in new_positions]
    period = next(p for p in range(2, len(old_colors) + 1)
                  if all(old_colors[j] == old_colors[j % p] for j in range(len(old_colors))))
    length = len(before_lane)
    expected_positions = []
    index = 0
    while old_positions[0] + index * (index + 1) // 2 < length:
        expected_positions.append(old_positions[0] + index * (index + 1) // 2)
        index += 1
    _require(old_positions == expected_positions[:len(old_positions)], "independent seed positions")
    _require(new_positions == expected_positions, "independent output positions")
    _require(new_colors == [old_colors[j % period] for j in range(len(new_positions))],
             "independent cyclic colors")
    if vertical:
        _require(all(np.array_equal(before[:, i], after[:, i]) for i in range(before.shape[1])
                     if i != lane_index), "independent unchanged cells")
    else:
        _require(all(np.array_equal(before[i], after[i]) for i in range(before.shape[0])
                     if i != lane_index), "independent unchanged cells")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos = [[_inspect_pair(pair, "self test") for section in ("train", "test")
              for pair in puzzle[section]] for puzzle in puzzles]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                _independent_rule_check(pair)
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _require(len({(i["length"], i["breadth"], i["start"], i["lane_index"],
                   i["period"], i["output_count"]) for group in infos for i in group}) >= 100,
             "structural diversity")
    _require({2, 3, 4, 5}.issubset({i["period"] for group in infos for i in group}), "cycle diversity")
    _require({True, False}.issubset({i["vertical"] for group in infos for i in group}), "direction diversity")
    _require(any(i["breadth"] == 1 for group in infos for i in group), "one-lane grids absent")
    _require(any(i["start"] > 0 for group in infos for i in group), "leading margin absent")
    _require(any(i["background"] != 0 for group in infos for i in group), "nonzero background absent")
    _require(any(0 in pair["input"] and info["background"] != 0
                 for puzzle, group in zip(puzzles, infos)
                 for pair, info in zip(puzzle["train"] + puzzle["test"], group)),
             "zero marker absent")
    _require(any(i["prefix_count"] > i["period"] for group in infos for i in group), "long prefix absent")
    _require(any(i["tail"] == 0 for group in infos for i in group)
             and any(i["tail"] >= 3 for group in infos for i in group), "boundary variety absent")
    _require(any(i["length"] == 30 for group in infos for i in group), "maximum length absent")
    _require(sum(len({i["background"] for i in group[:len(puzzle["train"])]}) > 1
                 for puzzle, group in zip(puzzles, infos)) == 42, "within-puzzle background diversity")
    _require(sum(len({i["vertical"] for i in group}) > 1 for group in infos) >= 15,
             "within-puzzle orientation diversity")
    _require(all(len({i["length"] for i in group}) > 1 and
                 len({i["period"] for i in group}) > 1 for group in infos),
             "within-puzzle length or period diversity")
    _require(any(i["vertical"] and i["background"] != 0 and i["start"] > 0
                 for group in infos for i in group), "combined vertical variation absent")
    _require(any(i["period"] == 5 and i["background"] != 0 and i["start"] > 0
                 for group in infos for i in group), "combined cycle variation absent")
    _require(any(not i["vertical"] and i["breadth"] == 3 and i["lane_index"] == 1
                 and i["start"] == 0 and i["background"] == 0 and i["period"] in (2, 3)
                 for group in infos for i in group), "source-like examples absent")
    _require(_equal_puzzles(generate_puzzle(137), generate_puzzle(137)), "nondeterministic")
    _require(_equal_puzzles(generate_puzzle(99173, 7, 3), generate_puzzle(99173, 7, 3)),
             "nondeterministic explicit counts")
    _require(not _equal_puzzles(generate_puzzle(137), generate_puzzle(138)), "seed difference absent")
    for seed, train, test in ((8100, 2, 1), (8101, 10, 1), (8102, 10, 4), (8103, 6, 2)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "explicit count")
    for seed, train, test in ((8200, None, 1), (8201, None, 4),
                              (8202, 2, None), (8203, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _require(train is None or len(puzzle["train"]) == train, "partial train count")
        _require(test is None or len(puzzle["test"]) == test, "partial test count")
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(8300, **request)
        except ValueError:
            continue
        raise AssertionError(f"illegal count accepted: {request}")
