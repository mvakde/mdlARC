import numpy as np


_ORIENTATIONS = ("top", "bottom", "left", "right")
_MODES = ("exact", "fit", "one", "two")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rows(lines: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(cell) for cell in line] for line in lines], dtype=np.int8)


_SOURCE_EXAMPLES = tuple(
    (_rows(before), _rows(after))
    for before, after in (
        (
            ("00700000", "00700000", "00700000", "00700000",
             "00700000", "00000000", "00000000"),
            ("78787870", "78787800", "78787000", "08780000",
             "00700000", "00000000", "00000000"),
        ),
        (
            ("0007000", "0007000", "0007000", "0007000", "0000000"),
            ("8787878", "0787870", "0087800", "0007000", "0000000"),
        ),
        (
            ("000007000", "000007000", "000007000", "000007000",
             "000007000", "000007000", "000007000", "000000000",
             "000000000"),
            ("878787878", "878787878", "078787878", "008787878",
             "000787870", "000087800", "000007000", "000000000",
             "000000000"),
        ),
    )
)


def _optional_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(rng: np.random.Generator, num_train: int | None,
            num_test: int | None) -> tuple[int, int]:
    train = _optional_count("num_train", num_train, 2, 10)
    test = _optional_count("num_test", num_test, 1, 4)
    choices = [(a, b) for a in range(2, 11) for b in range(1, 5)
               if a > b and (train is None or a == train)
               and (test is None or b == test)]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _mode(length: int, cross: int, position: int) -> str:
    radius = length - 1
    left, right = position, cross - 1 - position
    if left == radius and right == radius:
        return "exact"
    if left >= radius and right >= radius:
        return "fit"
    if left < radius and right < radius:
        return "two"
    return "one"


# This finite geometry catalogue contains possible line shapes, not puzzle
# templates. Length, width, position, padding, direction and colors are sampled
# separately for each pair.
_GEOMETRIES = {
    mode: tuple(
        (length, cross, position)
        for length in range(2, 23)
        for cross in range(3, 31)
        for position in range(1, cross - 1)
        if _mode(length, cross, position) == mode
    )
    for mode in _MODES
}


def _pick_band(rng: np.random.Generator, values: list[int],
               bands: tuple[tuple[int, int], ...]) -> int:
    available = [sorted(set(v for v in values if low <= v <= high))
                 for low, high in bands]
    available = [group for group in available if group]
    if not available:
        raise ValueError("No feasible geometry")
    group = available[int(rng.integers(len(available)))]
    return group[int(rng.integers(len(group)))]


def _sample_geometry(rng: np.random.Generator, mode: str, *,
                     phase: bool = False, short: bool = False,
                     length_relation: tuple[int, int] | None = None,
                     cross_relation: tuple[int, int] | None = None,
                     preferred_side: str | None = None,
                     source_small: bool = False) -> dict:
    candidates = list(_GEOMETRIES[mode])
    if phase:
        candidates = [(length, cross, position)
                      for length, cross, position in candidates
                      if length >= 3 and max(position, cross - 1 - position) >= 2]
    elif not short:
        candidates = [item for item in candidates if item[0] >= 3]
    if short:
        candidates = [item for item in candidates if item[0] == 2]
    if source_small:
        candidates = [item for item in candidates if item[0] <= 6]
    if length_relation is not None:
        narrowed = [item for item in candidates
                    if length_relation[0] <= item[0] <= length_relation[1]]
        if narrowed:
            candidates = narrowed

    length = _pick_band(rng, [item[0] for item in candidates],
                        ((2, 2), (3, 6), (7, 12), (13, 22)))
    candidates = [item for item in candidates if item[0] == length]
    if cross_relation is not None:
        narrowed = [item for item in candidates
                    if cross_relation[0] <= item[1] <= cross_relation[1]]
        if narrowed:
            candidates = narrowed
    cross = _pick_band(rng, [item[1] for item in candidates],
                       ((3, 9), (10, 19), (20, 30)))
    positions = [item[2] for item in candidates if item[1] == cross]
    groups = {
        "near_left": [p for p in positions if p <= 2],
        "negative": [p for p in positions if 2 * p < cross - 1],
        "center": [p for p in positions if abs(2 * p - (cross - 1)) <= 1],
        "positive": [p for p in positions if 2 * p > cross - 1],
        "near_right": [p for p in positions if cross - 1 - p <= 2],
    }
    if preferred_side and groups[preferred_side]:
        group = groups[preferred_side]
    else:
        nonempty = [group for group in groups.values() if group]
        group = nonempty[int(rng.integers(len(nonempty)))]
    position = group[int(rng.integers(len(group)))]

    padding = _pick_band(rng, list(range(1, 31 - length)),
                         ((1, 3), (4, 9), (10, 28)))
    return {"A": length + padding, "C": cross, "L": length, "p": position}


def _orient(grid: np.ndarray, direction: str) -> np.ndarray:
    if direction == "top":
        return grid.copy()
    if direction == "bottom":
        return np.flipud(grid).copy()
    if direction == "left":
        return grid.T.copy()
    if direction == "right":
        return np.fliplr(grid.T).copy()
    raise ValueError("Unknown direction")


def _make_pair(config: dict, colors: tuple[int, int, int],
               direction: str) -> dict:
    axis, cross, length, position = (config[key] for key in ("A", "C", "L", "p"))
    background, spine, accent = colors
    before = np.full((axis, cross), background, dtype=np.int8)
    after = before.copy()
    before[:length, position] = spine
    for a in range(length):
        radius = length - 1 - a
        for p in range(max(0, position - radius),
                       min(cross - 1, position + radius) + 1):
            after[a, p] = spine if (p - position) % 2 == 0 else accent
    return {"input": _orient(before, direction),
            "output": _orient(after, direction)}


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.tobytes())


def _source_pair(pair: dict) -> bool:
    return any(np.array_equal(pair["input"], source_in)
               and np.array_equal(pair["output"], source_out)
               for source_in, source_out in _SOURCE_EXAMPLES)


def _describe(grid: np.ndarray) -> tuple[dict, int, int]:
    values, counts = np.unique(grid, return_counts=True)
    _check(len(values) == 2, "Input needs exactly two colors")
    background_index = int(np.argmax(counts))
    _check(counts[background_index] > counts[1 - background_index],
           "Background must be unambiguous")
    background = int(values[background_index])
    spine = int(values[1 - background_index])
    coordinates = np.argwhere(grid == spine)
    _check(len(coordinates) >= 2, "Spine is too short")
    rows, columns = coordinates[:, 0], coordinates[:, 1]
    height, width = grid.shape
    if np.all(columns == columns[0]):
        ordered = np.sort(rows)
        _check(np.array_equal(ordered, np.arange(ordered[0], ordered[-1] + 1)),
               "Spine is not contiguous")
        position = int(columns[0])
        _check(0 < position < width - 1, "Spine lies on lateral edge")
        if ordered[0] == 0 and ordered[-1] < height - 1:
            direction = "top"
        elif ordered[-1] == height - 1 and ordered[0] > 0:
            direction = "bottom"
        else:
            raise AssertionError("Spine must touch one axial edge")
        axis, cross = height, width
    elif np.all(rows == rows[0]):
        ordered = np.sort(columns)
        _check(np.array_equal(ordered, np.arange(ordered[0], ordered[-1] + 1)),
               "Spine is not contiguous")
        position = int(rows[0])
        _check(0 < position < height - 1, "Spine lies on lateral edge")
        if ordered[0] == 0 and ordered[-1] < width - 1:
            direction = "left"
        elif ordered[-1] == width - 1 and ordered[0] > 0:
            direction = "right"
        else:
            raise AssertionError("Spine must touch one axial edge")
        axis, cross = width, height
    else:
        raise AssertionError("Foreground must be a straight spine")
    length = len(coordinates)
    _check(length < axis, "The free tip needs background beyond it")
    return ({"A": axis, "C": cross, "L": length, "p": position,
             "direction": direction, "mode": _mode(length, cross, position)},
            background, spine)


def _accent_from_pair(pair: dict) -> int:
    old = set(int(v) for v in np.unique(pair["input"]))
    new = set(int(v) for v in np.unique(pair["output"])) - old
    _check(len(new) == 1, "Output must introduce one accent")
    return new.pop()


def _choose_palette(rng: np.random.Generator, accent: int,
                    previous: tuple[int, int] | None = None) -> tuple[int, int, int]:
    other = [color for color in range(10) if color != accent]
    choices = [(a, b) for a in other for b in other if a != b
               and (previous is None or (a != previous[0] and b != previous[1]))]
    background, spine = choices[int(rng.integers(len(choices)))]
    return background, spine, accent


def _test_relation(rng: np.random.Generator, maximum: int,
                   limit: int) -> tuple[int, int]:
    choice = int(rng.integers(3))
    if choice == 0:
        return 2, max(2, maximum - 1)
    if choice == 1:
        return maximum, maximum
    return min(maximum + 1, limit), limit


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    accent = int(rng.integers(10))
    source_like = bool(rng.random() < 0.22)
    mixed_palette = not source_like and bool(rng.random() < 0.55)
    mixed_direction = not source_like and bool(rng.random() < 0.58)

    base_direction = _ORIENTATIONS[int(rng.integers(4))]
    train_directions = [base_direction]
    if mixed_direction:
        alternatives = [d for d in _ORIENTATIONS if d != base_direction]
        train_directions.append(alternatives[int(rng.integers(3))])
    else:
        train_directions.append(base_direction)
    for _ in range(2, train_count):
        if mixed_direction and rng.random() < 0.7:
            choices = train_directions[:2]
        elif mixed_direction:
            choices = _ORIENTATIONS
        else:
            choices = (base_direction,)
        train_directions.append(choices[int(rng.integers(len(choices)))])
    distinct_train_directions = set(train_directions)
    test_directions = []
    for _ in range(test_count):
        novel = [direction for direction in _ORIENTATIONS
                 if direction not in distinct_train_directions]
        choices = (novel if novel and len(distinct_train_directions) >= 2
                   and rng.random() < 0.3
                   else sorted(distinct_train_directions))
        test_directions.append(choices[int(rng.integers(len(choices)))])

    fixed_colors = _choose_palette(rng, accent)
    train_colors = [fixed_colors]
    for index in range(1, train_count):
        if mixed_palette:
            previous = fixed_colors[:2] if index == 1 else None
            train_colors.append(_choose_palette(rng, accent, previous))
        else:
            train_colors.append(fixed_colors)
    test_colors = [
        _choose_palette(rng, accent) if mixed_palette else fixed_colors
        for _ in range(test_count)
    ]

    if source_like:
        train_modes = ["exact", "one"]
    else:
        first = _MODES[int(rng.integers(4))]
        second_choices = [mode for mode in _MODES if mode != first]
        train_modes = [first, second_choices[int(rng.integers(3))]]
    train_modes += [_MODES[int(rng.integers(4))]
                    for _ in range(2, train_count)]

    puzzle = {"train": [], "test": []}
    used_inputs: set[tuple] = set()
    used_layouts: set[tuple] = set()
    train_descriptors: list[dict] = []

    def append_pair(split: str, mode: str, direction: str,
                    colors: tuple[int, int, int], *, phase: bool = False,
                    short: bool = False, length_relation=None,
                    cross_relation=None, preferred_side=None,
                    source_small=False) -> None:
        for attempt in range(160):
            current_mode = mode if attempt < 120 else _MODES[(attempt - 120) % 4]
            config = _sample_geometry(
                rng, current_mode, phase=phase, short=short and current_mode == "fit",
                length_relation=length_relation, cross_relation=cross_relation,
                preferred_side=preferred_side, source_small=source_small,
            )
            layout = (direction, config["L"], config["p"], config["C"])
            if layout in used_layouts:
                continue
            pair = _make_pair(config, colors, direction)
            if _key(pair["input"]) in used_inputs or _source_pair(pair):
                continue
            used_layouts.add(layout)
            used_inputs.add(_key(pair["input"]))
            puzzle[split].append(pair)
            if split == "train":
                train_descriptors.append(config)
            return
        raise RuntimeError("Unable to find a distinct legal fan layout")

    for index in range(train_count):
        mode = train_modes[index]
        short = (index > 0 and mode == "fit" and rng.random() < 0.28)
        append_pair(
            "train", mode, train_directions[index], train_colors[index],
            phase=index == 0, short=short,
            source_small=source_like and index < 2,
        )

    max_length = max(config["L"] for config in train_descriptors)
    max_cross = max(config["C"] for config in train_descriptors)
    for index in range(test_count):
        mode = "two" if source_like else _MODES[int(rng.integers(4))]
        length_relation = ((max_length + 1, 22) if source_like
                           else _test_relation(rng, max_length, 22))
        cross_relation = ((max_cross + 1, 30) if source_like
                          else _test_relation(rng, max_cross, 30))
        append_pair(
            "test", mode, test_directions[index], test_colors[index],
            length_relation=length_relation, cross_relation=cross_relation,
            preferred_side="positive" if source_like else None,
        )

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
           "Puzzle requires exactly train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
           "Splits must be lists")
    _check(2 <= len(puzzle["train"]) <= 10, "Illegal train count")
    _check(1 <= len(puzzle["test"]) <= 4, "Illegal test count")
    _check(len(puzzle["train"]) > len(puzzle["test"]), "Train must exceed test")

    inputs: set[tuple] = set()
    layouts: set[tuple] = set()
    descriptors = {"train": [], "test": []}
    train_accents = set()
    phase_seen = False

    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"},
                   "Invalid pair schema")
            for name in ("input", "output"):
                grid = pair[name]
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2,
                       "Grid must be a two-dimensional array")
                _check(np.issubdtype(grid.dtype, np.integer),
                       "Grid cells must be integers")
                _check(all(1 <= size <= 30 for size in grid.shape),
                       "Grid dimension outside 1..30")
                _check(bool(np.all((grid >= 0) & (grid <= 9))),
                       "Grid color outside 0..9")
            _check(pair["input"].shape == pair["output"].shape,
                   "Shape must be preserved")
            _check(not np.array_equal(pair["input"], pair["output"]),
                   "Fan must visibly change input")
            _check(not _source_pair(pair), "Source pair was copied")
            input_key = _key(pair["input"])
            _check(input_key not in inputs, "Duplicate input")
            inputs.add(input_key)

            desc, background, spine = _describe(pair["input"])
            layout = (desc["direction"], desc["L"], desc["p"], desc["C"])
            _check(layout not in layouts,
                   "Repeated geometry with only padding or recoloring changed")
            layouts.add(layout)
            accent = _accent_from_pair(pair)
            _check(accent not in (background, spine), "Accent collides with input")
            _check(set(int(v) for v in np.unique(pair["output"]))
                   == {background, spine, accent}, "Output palette is wrong")
            expected = _make_pair(desc, (background, spine, accent),
                                  desc["direction"])["output"]
            _check(np.array_equal(expected, pair["output"]),
                   "Triangular widening or stripe parity is wrong")
            descriptors[split].append(desc)
            if split == "train":
                train_accents.add(accent)
                if desc["L"] == 2:
                    _check(phase_seen, "Length-two fan precedes parity evidence")
                phase_seen |= (desc["L"] >= 3
                               and max(desc["p"], desc["C"] - 1 - desc["p"]) >= 2)

    _check(len(train_accents) == 1, "Training accents disagree")
    accent = next(iter(train_accents))
    _check(phase_seen, "Training does not establish odd and even stripe phases")
    _check(len({item["mode"] for item in descriptors["train"]}) >= 2,
           "Training repeats one clipping regime")
    train_directions = {item["direction"] for item in descriptors["train"]}
    for pair, desc in zip(puzzle["test"], descriptors["test"]):
        _check(_accent_from_pair(pair) == accent,
               "Test accent is not recoverable from training")
        _check(desc["direction"] in train_directions or len(train_directions) >= 2,
               "Novel test direction needs multiple training directions")


def _signature(puzzle: dict) -> tuple:
    return tuple((split, tuple((_key(pair["input"]), _key(pair["output"]))
                               for pair in puzzle[split]))
                 for split in ("train", "test"))


def _predict(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    accents = {_accent_from_pair(pair) for pair in train}
    _check(len(accents) == 1, "Cannot learn a single accent")
    desc, background, spine = _describe(test_input)
    return _make_pair(desc, (background, spine, next(iter(accents))),
                      desc["direction"])["output"]


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            _check(np.array_equal(_predict(puzzle["train"], pair["input"]),
                                  pair["output"]), "Training-only prediction failed")
    _check(len({len(p["train"]) for p in puzzles}) >= 4,
           "Too few train count variants")
    _check(len({len(p["test"]) for p in puzzles}) >= 3,
           "Too few test count variants")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (1010, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        _check((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
               "Explicit counts were ignored")
        validate_generated_puzzle(puzzle)
    for seed, train, test in ((1101, None, 1), (1102, None, 4),
                              (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _check(train is None or len(puzzle["train"]) == train,
               "Partial train count ignored")
        _check(test is None or len(puzzle["test"]) == test,
               "Partial test count ignored")
    _check(_signature(generate_puzzle(123456, 7, 3))
           == _signature(generate_puzzle(123456, 7, 3)),
           "Generation is nondeterministic")
    _check(len({_signature(generate_puzzle(2000 + i, 6, 2))
                for i in range(10)}) >= 9, "Seeds lack diversity")
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(3000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")

    all_descriptors = []
    all_shapes = set()
    all_palettes = set()
    all_layouts = set()
    structural_combinations = set()
    mixed_directions = mixed_palettes = short_trains = 0
    fixed_directions = fixed_palettes = 0
    two_sided_train = source_like = combined = novel_test_direction = 0
    train_modes = set()
    test_modes = set()
    test_positions = set()
    scale_relations = {axis: set() for axis in ("A", "C", "L")}
    near_edge_test = asymmetric_test = long_padding = large_canvas = 0
    for puzzle in puzzles:
        train_descriptions = [_describe(pair["input"])[0] for pair in puzzle["train"]]
        test_descriptions = [_describe(pair["input"])[0] for pair in puzzle["test"]]
        directions = {d["direction"] for d in train_descriptions}
        palettes = {_describe(pair["input"])[1:] for pair in puzzle["train"]}
        if len(directions) > 1:
            mixed_directions += 1
        else:
            fixed_directions += 1
        if len(palettes) > 1:
            mixed_palettes += 1
        else:
            fixed_palettes += 1
        if any(d["L"] == 2 for d in train_descriptions):
            short_trains += 1
        if any(d["mode"] == "two" for d in train_descriptions):
            two_sided_train += 1
            if len(directions) > 1 and len(palettes) > 1:
                combined += 1
        if (train_descriptions[0]["mode"] == "exact"
                and train_descriptions[1]["mode"] == "one"
                and len(directions) == len(palettes) == 1
                and all(d["mode"] == "two" for d in test_descriptions)):
            source_like += 1
        if any(d["direction"] not in directions for d in test_descriptions):
            novel_test_direction += 1
        for desc in test_descriptions:
            near_edge_test += min(desc["p"], desc["C"] - 1 - desc["p"]) <= 2
            asymmetric_test += abs(2 * desc["p"] - (desc["C"] - 1)) >= 5
            for axis in ("A", "C", "L"):
                maximum = max(item[axis] for item in train_descriptions)
                scale_relations[axis].add(
                    "smaller" if desc[axis] < maximum else
                    "larger" if desc[axis] > maximum else "similar"
                )
        for split in ("train", "test"):
            for pair in puzzle[split]:
                desc, background, spine = _describe(pair["input"])
                all_descriptors.append(desc)
                all_shapes.add(pair["input"].shape)
                all_palettes.add((background, spine, _accent_from_pair(pair)))
                all_layouts.add((desc["direction"], desc["L"],
                                 desc["C"], desc["p"]))
                structural_combinations.add((desc["mode"], desc["direction"],
                                             desc["L"] // 5, desc["C"] // 5))
                long_padding += desc["A"] - desc["L"] >= 8
                large_canvas += max(pair["input"].shape) >= 27
                if split == "train":
                    train_modes.add(desc["mode"])
                if split == "test":
                    test_modes.add(desc["mode"])
                    test_positions.add("negative" if 2 * desc["p"] < desc["C"] - 1
                                       else "positive" if 2 * desc["p"] > desc["C"] - 1
                                       else "center")
    _check(len(all_shapes) >= 30 and len(all_palettes) >= 50,
           "Weak shape or palette diversity")
    _check({d["direction"] for d in all_descriptors} == set(_ORIENTATIONS),
           "Missing direction")
    _check(mixed_directions >= 5 and mixed_palettes >= 5,
           "Missing within-puzzle direction or palette variation")
    _check(fixed_directions >= 5 and fixed_palettes >= 5,
           "Missing simple fixed-convention puzzles")
    _check(two_sided_train >= 5 and source_like >= 1 and combined >= 1,
           "Missing clipping or combined variation regimes")
    _check(novel_test_direction >= 1,
           "Missing a novel test direction after multiple training directions")
    _check(short_trains >= 1, "No short training fan observed")
    _check(train_modes == set(_MODES), "Missing training clipping regime")
    _check(test_modes == set(_MODES), "Missing test clipping regime")
    _check(test_positions == {"negative", "positive", "center"},
           "Missing test lateral position")
    _check(max(max(shape) for shape in all_shapes) >= 27,
           "No canvas approaches thirty cells")
    _check(near_edge_test >= 5 and asymmetric_test >= 5,
           "Missing near-edge or asymmetric test spines")
    _check(long_padding >= 10 and large_canvas >= 10,
           "Missing long blank space or near-thirty canvases")
    _check(all(values == {"smaller", "similar", "larger"}
               for values in scale_relations.values()),
           "Test scales do not span smaller, similar and larger")
    _check(len(all_layouts) >= 150 and len(structural_combinations) >= 100,
           "Insufficient structural combinations")
