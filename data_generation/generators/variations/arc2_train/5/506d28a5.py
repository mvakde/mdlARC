"""Generate aligned two-panel Boolean-OR ARC puzzles."""

import numpy as np


_SOURCE_ROWS = (
    ("00000/02220/02000/22202/44444/11001/11011/10101/01011", "33003/33333/33303/33333"),
    ("22222/00020/02022/22222/44444/01100/11010/11000/00111", "33333/33030/33033/33333"),
    ("02002/02202/00022/00220/44444/10100/10010/00001/01110", "33303/33333/00033/03330"),
    ("02200/22220/00220/00220/44444/10010/11101/00111/11100", "33330/33333/00333/33330"),
    ("02222/00020/02000/22202/44444/00100/00001/10011/00001", "03333/00033/33033/33303"),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_key(text: str) -> tuple:
    rows = text.split("/")
    return ((len(rows), len(rows[0])), tuple(int(x) for row in rows for x in row))


_SOURCE_PAIRS = frozenset((_source_key(a), _source_key(b)) for a, b in _SOURCE_ROWS)
_SOURCE_INPUTS = frozenset(a for a, _ in _SOURCE_PAIRS)


def _count_argument(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count_argument("num_train", num_train)
    test = _count_argument("num_test", num_test)
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test")
    if train is None and test is None:
        test = seed % 4 + 1
        choices = list(range(max(2, test + 1), 11))
        train = choices[(seed // 4) % len(choices)]
    elif train is None:
        choices = list(range(max(2, test + 1), 11))
        train = choices[seed % len(choices)]
    elif test is None:
        choices = list(range(1, min(4, train - 1) + 1))
        test = choices[seed % len(choices)]
    return train, test


def _spatial_order(rng: np.random.Generator, height: int, width: int, style: int) -> np.ndarray:
    rows, cols = np.indices((height, width))
    rows = rows.ravel().astype(float)
    cols = cols.ravel().astype(float)
    size = height * width
    if style == 0:  # isolated, irregular points
        return rng.permutation(size)
    if style == 1:  # compact block
        ar = int(rng.integers(height))
        ac = int(rng.integers(width))
        score = (rows - ar) ** 2 + (cols - ac) ** 2
    elif style == 2:  # horizontal or vertical bar
        if int(rng.integers(2)):
            score = np.abs(rows - int(rng.integers(height))) * (width + 1) + rng.random(size)
        else:
            score = np.abs(cols - int(rng.integers(width))) * (height + 1) + rng.random(size)
    elif style == 3:  # rectangular block
        r0, r1 = sorted(int(x) for x in rng.integers(height, size=2))
        c0, c1 = sorted(int(x) for x in rng.integers(width, size=2))
        distance = np.maximum(r0 - rows, 0) + np.maximum(rows - r1, 0)
        distance += np.maximum(c0 - cols, 0) + np.maximum(cols - c1, 0)
        score = distance * (size + 1) + rng.random(size)
    elif style == 4:  # ring or hole
        ar = float(rng.integers(height))
        ac = float(rng.integers(width))
        radius = float(rng.uniform(1.1, max(1.2, min(height, width) / 2)))
        score = np.abs(np.maximum(np.abs(rows - ar), np.abs(cols - ac)) - radius)
    elif style == 5:  # several separated pieces
        anchors = rng.integers(0, [height, width], size=(int(rng.integers(2, 5)), 2))
        score = np.min(
            np.abs(rows[:, None] - anchors[None, :, 0])
            + np.abs(cols[:, None] - anchors[None, :, 1]), axis=1
        )
    elif style == 6:  # diagonal strip
        slope = float(rng.choice((-1.0, -0.5, 0.5, 1.0)))
        center = float(rng.integers(height))
        score = np.abs(rows - center - slope * (cols - (width - 1) / 2))
    elif style == 7:  # checkerboard islands
        score = ((rows + cols + int(rng.integers(2))) % 2) * 3.0
    elif style == 8:  # border with a hole
        edge = np.minimum.reduce((rows, cols, height - 1 - rows, width - 1 - cols))
        score = edge if int(rng.integers(2)) else -edge
    else:  # a filled side or top/bottom strip
        direction = int(rng.integers(4))
        score = (rows, -rows, cols, -cols)[direction]
    return np.argsort(score + rng.random(size) * 0.15, kind="stable")


def _choose_cells(
    rng: np.random.Generator, allowed: np.ndarray, count: int, style: int
) -> np.ndarray:
    height, width = allowed.shape
    result = np.zeros((height, width), dtype=bool)
    if count:
        order = _spatial_order(rng, height, width, style)
        available = order[allowed.ravel()[order]]
        _require(len(available) >= count, "mask allocation exceeded available cells")
        result.ravel()[available[:count]] = True
    return result


def _density_count(rng: np.random.Generator, size: int, regime: str) -> int:
    probabilities = (0.20, 0.60, 0.20) if regime == "source_train" else (0.36, 0.28, 0.36)
    band = int(rng.choice(3, p=probabilities))
    bounds = ((0.06, 0.24), (0.28, 0.58), (0.62, 0.84))[band]
    low = max(1, int(np.ceil(size * bounds[0])))
    high = min(size - 1, max(low, int(np.floor(size * bounds[1]))))
    return int(rng.integers(low, high + 1))


_RELATIONS = (
    "partial", "disjoint", "upper_inside_lower", "lower_inside_upper",
    "identical", "empty_upper", "empty_lower", "full_union", "lower_feature",
)


def _masks(
    rng: np.random.Generator, height: int, width: int, relation: str, regime: str
) -> tuple[np.ndarray, np.ndarray]:
    size = height * width
    upper_style = int(rng.integers(10))
    lower_style = int(rng.integers(10))
    upper_count = _density_count(rng, size, regime)
    lower_count = _density_count(rng, size, regime)
    if relation == "partial":
        upper_count = min(size - 2, max(2, upper_count))
        lower_count = min(size - 2, max(2, lower_count))
        low_overlap = max(1, upper_count + lower_count - size + 1)
        high_overlap = min(upper_count - 1, lower_count - 1)
        overlap = int(rng.integers(low_overlap, high_overlap + 1))
    elif relation == "lower_feature":
        minimum_extra = max(3, int(np.ceil(size * 0.08)))
        upper_count = min(size - minimum_extra - 1, max(2, upper_count))
        overlap = int(rng.integers(1, upper_count))
        maximum_extra = min(size - upper_count - 1, max(minimum_extra, int(size * 0.24)))
        extra = int(rng.integers(minimum_extra, maximum_extra + 1))
        lower_count = overlap + extra
    elif relation == "disjoint":
        upper_count = min(size - 2, upper_count)
        lower_count = min(size - upper_count - 1, lower_count)
        overlap = 0
    elif relation == "upper_inside_lower":
        lower_count = max(2, lower_count)
        upper_count = min(lower_count - 1, upper_count)
        overlap = upper_count
    elif relation == "lower_inside_upper":
        upper_count = max(2, upper_count)
        lower_count = min(upper_count - 1, lower_count)
        overlap = lower_count
    elif relation == "identical":
        lower_count = upper_count
        overlap = upper_count
    elif relation == "empty_upper":
        upper_count = 0
        overlap = 0
    elif relation == "empty_lower":
        lower_count = 0
        overlap = 0
    elif relation == "full_union":
        upper_count = min(size - 1, max(1, upper_count))
        overlap = int(rng.integers(0, min(upper_count, size - upper_count) ))
        lower_count = size - upper_count + overlap
    else:
        raise ValueError("unknown mask relationship")
    all_cells = np.ones((height, width), dtype=bool)
    upper = _choose_cells(rng, all_cells, upper_count, upper_style)
    order = _spatial_order(rng, height, width, lower_style)
    in_upper = order[upper.ravel()[order]]
    outside_upper = order[~upper.ravel()[order]]
    lower = np.zeros((height, width), dtype=bool)
    lower.ravel()[in_upper[:overlap]] = True
    lower.ravel()[outside_upper[:lower_count - overlap]] = True
    _require(int(lower.sum()) == lower_count, "lower mask count was not realized")
    return upper, lower


def _pair_palette(
    rng: np.random.Generator, result: int, base: tuple[int, ...],
    index: int, classic: bool, previous: tuple[int, ...] | None
) -> tuple[int, int, int, int]:
    if classic:
        return (0, 2, 1, 4)
    if index == 0:
        return base
    mode = index % 4 if index < 4 else int(rng.integers(4))
    if mode == 1:
        palette = (base[0], base[2], base[1], base[3])
    elif mode == 2:
        palette = (base[3], base[1], base[2], base[0])
    elif mode == 3:
        palette = tuple(int(x) for x in rng.choice([c for c in range(10) if c != result], 4, replace=False))
    else:
        palette = base
    if palette == previous:
        palette = (palette[3], palette[1], palette[2], palette[0])
    return palette


def _pair_dimensions(
    rng: np.random.Generator, index: int, classic: bool,
    first_dimensions: tuple[int, int] | None
) -> tuple[int, int]:
    if classic:
        return 4, 5
    height = int(rng.integers(3, 15))
    width = int(rng.integers(3, 17))
    if index == 1 and (height, width) == first_dimensions:
        width = 3 + (width - 2) % 14
    return height, width


def _make_pair(
    rng: np.random.Generator, height: int, width: int,
    colors: tuple[int, int, int, int], result: int,
    relation: str, regime: str
) -> dict:
    background, upper_color, lower_color, divider = colors
    upper_mask, lower_mask = _masks(rng, height, width, relation, regime)
    upper = np.full((height, width), background, dtype=np.int8)
    lower = upper.copy()
    upper[upper_mask] = upper_color
    lower[lower_mask] = lower_color
    separator = np.full((1, width), divider, dtype=np.int8)
    output = np.full((height, width), background, dtype=np.int8)
    output[upper_mask | lower_mask] = result
    return {"input": np.vstack((upper, separator, lower)), "output": output}


def generate_puzzle(
    seed: int, num_train: int | None = None, num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    # Periodic easy cases ensure that each 42-seed run includes several
    # source-like puzzles, independently of the nine-way relationship cycle.
    classic = seed % 11 == 3
    result = 3 if classic else int(rng.integers(10))
    base = (0, 2, 1, 4) if classic else tuple(
        int(x) for x in rng.choice([c for c in range(10) if c != result], 4, replace=False)
    )
    puzzle = {"train": [], "test": []}
    seen = set()
    first_dimensions = None
    previous_palette = None
    index = 0
    for section, count, regime in (
        ("train", train_count, "source_train"),
        ("test", test_count, "target_test"),
    ):
        for section_index in range(count):
            relation = (
                "partial" if section == "train" and section_index == 0
                else _RELATIONS[(seed * (7 if section == "train" else 5) + section_index) % len(_RELATIONS)]
            )
            for attempt in range(30):
                height, width = _pair_dimensions(rng, index, classic, first_dimensions)
                colors = _pair_palette(rng, result, base, index, classic, previous_palette)
                pair = _make_pair(rng, height, width, colors, result, relation, regime)
                input_key = _key(pair["input"])
                pair_key = (input_key, _key(pair["output"]))
                if input_key in seen or input_key in _SOURCE_INPUTS or pair_key in _SOURCE_PAIRS:
                    continue
                seen.add(input_key)
                puzzle[section].append(pair)
                if index == 0:
                    first_dimensions = (height, width)
                previous_palette = colors
                break
            else:
                raise RuntimeError("could not generate a distinct pair")
            index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: object, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    _require(all(1 <= size <= 30 for size in grid.shape), f"{label} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} must contain integers")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{label} has illegal colors")


def _inspect_pair(pair: dict) -> dict:
    input_grid, output = pair["input"], pair["output"]
    height, width = output.shape
    _require(input_grid.shape == (2 * height + 1, width), "panels must match output dimensions")
    upper = input_grid[:height]
    separator = input_grid[height]
    lower = input_grid[height + 1:]
    divider = int(separator[0])
    _require(np.all(separator == divider), "separator must be a solid row")
    upper_colors = set(int(x) for x in np.unique(upper))
    lower_colors = set(int(x) for x in np.unique(lower))
    _require(1 <= len(upper_colors) <= 2 and 1 <= len(lower_colors) <= 2,
             "each panel must contain at most one foreground")
    common = upper_colors & lower_colors
    _require(len(common) == 1, "panels must expose one shared background")
    background = next(iter(common))
    _require(background in upper_colors and background in lower_colors,
             "background must be present in both panels")
    _require(divider not in upper_colors | lower_colors, "divider color is not unique")
    upper_mask = upper != background
    lower_mask = lower != background
    union = upper_mask | lower_mask
    _require(bool(union.any()), "both masks cannot be empty")
    _require(not bool(upper_mask.all()) and not bool(lower_mask.all()),
             "a solid foreground panel would conceal its background")
    result_colors = set(int(x) for x in np.unique(output[union]))
    _require(len(result_colors) == 1, "union cells must have one result color")
    result = next(iter(result_colors))
    _require(result not in set(int(x) for x in np.unique(input_grid)),
             "result must have a distinct color role")
    _require(np.all(output[~union] == background), "background-only cells changed")
    cases = (
        int(np.count_nonzero(~upper_mask & ~lower_mask)),
        int(np.count_nonzero(upper_mask & ~lower_mask)),
        int(np.count_nonzero(~upper_mask & lower_mask)),
        int(np.count_nonzero(upper_mask & lower_mask)),
    )
    return {
        "height": height, "width": width, "background": background,
        "result": result, "divider": divider,
        "upper_color": next(iter(upper_colors - {background}), None),
        "lower_color": next(iter(lower_colors - {background}), None),
        "upper_mask": upper_mask, "lower_mask": lower_mask,
        "cases": cases,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    infos = {"train": [], "test": []}
    seen_inputs = set()
    seen_pairs = set()
    for section in ("train", "test"):
        for index, pair in enumerate(puzzle[section]):
            label = f"{section}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must contain exactly input and output")
            _validate_grid(pair["input"], label + ".input")
            _validate_grid(pair["output"], label + ".output")
            input_key, output_key = _key(pair["input"]), _key(pair["output"])
            pair_key = (input_key, output_key)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            _require(input_key not in _SOURCE_INPUTS and pair_key not in _SOURCE_PAIRS,
                     "a source pair was copied")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            infos[section].append(_inspect_pair(pair))
    result_colors = {info["result"] for section in infos.values() for info in section}
    _require(len(result_colors) == 1, "result color must be a puzzle-level convention")
    covered_cases = [sum(info["cases"][case] for info in infos["train"]) for case in range(4)]
    _require(all(covered_cases), "training must show all four Boolean cases")
    _require(all(all(info["cases"]) for info in infos["train"][:1]),
             "first training example must be a complete OR demonstration")
    # A test background is inferred from its two aligned panels; the only learned
    # color is the result color, which is visible in the training outputs.
    for info in infos["test"]:
        _require(info["result"] == infos["train"][0]["result"],
                 "test output uses an unseen result color")
    for pair in puzzle["test"]:
        prediction = _predict_from_demonstrations(puzzle["train"], pair["input"].copy())
        _require(np.array_equal(prediction, pair["output"]),
                 "training examples and test input do not determine the test output")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((section, tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[section]))
                 for section in ("train", "test"))


def _predict_from_demonstrations(train: list, test_input: np.ndarray) -> np.ndarray:
    # Only training examples and the test input are consulted here.
    learned = set()
    for pair in train:
        height = pair["output"].shape[0]
        upper = pair["input"][:height]
        lower = pair["input"][height + 1:]
        background = (set(int(x) for x in np.unique(upper))
                      & set(int(x) for x in np.unique(lower))).pop()
        learned.update(int(x) for x in np.unique(pair["output"]) if x != background)
    _require(len(learned) == 1, "training does not determine a unique result color")
    result = next(iter(learned))
    height = (test_input.shape[0] - 1) // 2
    upper = test_input[:height]
    lower = test_input[height + 1:]
    background = (set(int(x) for x in np.unique(upper))
                  & set(int(x) for x in np.unique(lower))).pop()
    output = np.full(upper.shape, background, dtype=np.int8)
    output[(upper != background) | (lower != background)] = result
    return output


def _relation(info: dict) -> str:
    background, upper_only, lower_only, overlap = info["cases"]
    if upper_only + overlap == 0:
        return "empty_upper"
    if lower_only + overlap == 0:
        return "empty_lower"
    if upper_only == lower_only == 0:
        return "identical"
    if background == 0:
        return "full_union"
    if overlap == 0:
        return "disjoint"
    if upper_only == 0:
        return "upper_inside_lower"
    if lower_only == 0:
        return "lower_inside_upper"
    return "partial"


def _components(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    for r, c in zip(*np.where(mask)):
        if seen[r, c]:
            continue
        count += 1
        stack = [(int(r), int(c))]
        seen[r, c] = True
        while stack:
            row, col = stack.pop()
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _expect_bad_counts(seed: int, train: int | None, test: int | None) -> None:
    try:
        generate_puzzle(seed, train, test)
    except ValueError:
        return
    raise AssertionError("illegal count request did not raise ValueError")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["test"]:
            prediction = _predict_from_demonstrations(puzzle["train"], pair["input"].copy())
            _require(np.array_equal(prediction, pair["output"]),
                     "train-only OR prediction failed")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None), (123456, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "train count not honored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "test count not honored")
    for seed, train, test in ((3001, 1, None), (3002, 11, None),
                              (3003, None, 0), (3004, None, 5),
                              (3005, 2, 2), (3006, 3, 4),
                              (3007, 4, 4), (3010, 2.0, 1), (3011, 3, True)):
        _expect_bad_counts(seed, train, test)
    _require(_puzzle_key(generate_puzzle(654321, 8, 3)) ==
             _puzzle_key(generate_puzzle(654321, 8, 3)), "explicit generation is nondeterministic")
    _require(_puzzle_key(generate_puzzle(31)) == _puzzle_key(generate_puzzle(31)),
             "automatic generation is nondeterministic")
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, "cross-seed puzzles repeat")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "too few training counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "too few test counts")
    infos = [(section, _inspect_pair(pair), puzzle_index)
             for puzzle_index, puzzle in enumerate(puzzles)
             for section in ("train", "test") for pair in puzzle[section]]
    train_infos = [info for section, info, _ in infos if section == "train"]
    test_infos = [info for section, info, _ in infos if section == "test"]
    dimensions = {(info["height"], info["width"]) for _, info, _ in infos}
    role_palettes = {(info["background"], info["upper_color"], info["lower_color"],
                      info["divider"], info["result"]) for _, info, _ in infos}
    relationships = {_relation(info) for info in test_infos}
    _require(len(dimensions) >= 35, "dimensions lack cross-seed variety")
    _require(len(role_palettes) >= 80, "input color roles lack variety")
    _require({"partial", "disjoint", "upper_inside_lower", "lower_inside_upper",
              "identical", "empty_upper", "empty_lower", "full_union"} <= relationships,
             "test masks omit a requested relationship")
    _require(any(info["cases"][2] >= 3 for info in test_infos),
             "no test has several lower-only cells")
    _require(any(_components(info["lower_mask"] & ~info["upper_mask"]) >= 2
                 for info in test_infos), "lower-only pieces never separate")
    _require(any(info["cases"][2] >= 3 and
                 _components(info["lower_mask"] & ~info["upper_mask"]) == 1
                 for info in test_infos), "coherent lower-only shapes are missing")
    for section_infos in (train_infos, test_infos):
        for mask_name in ("upper_mask", "lower_mask"):
            densities = [float(info[mask_name].mean()) for info in section_infos]
            _require(any(value <= 0.25 for value in densities), "sparse mask missing")
            _require(any(value >= 0.60 for value in densities), "dense mask missing")
    within_dimensions = 0
    within_palettes = 0
    for puzzle in puzzles:
        local = [_inspect_pair(pair) for section in ("train", "test") for pair in puzzle[section]]
        within_dimensions += len({(x["height"], x["width"]) for x in local}) > 1
        within_palettes += len({(x["background"], x["upper_color"], x["lower_color"], x["divider"])
                                for x in local}) > 1
    _require(within_dimensions >= 30 and within_palettes >= 30,
             "pair-level dimensions or colors rarely vary")
    density_similar = 0
    density_above = 0
    density_below = 0
    visible_swaps = 0
    classic_puzzles = 0
    mixed_combinations = set()
    for puzzle in puzzles:
        training = [_inspect_pair(pair) for pair in puzzle["train"]]
        testing = [_inspect_pair(pair) for pair in puzzle["test"]]
        training_lower = [float(info["lower_mask"].mean()) for info in training]
        first, second = training[:2]
        visible_swaps += (
            first["background"] == second["background"]
            and first["divider"] == second["divider"]
            and first["upper_color"] == second["lower_color"]
            and first["lower_color"] == second["upper_color"]
        )
        classic_puzzles += all(
            info["height"] == 4 and info["width"] == 5
            and (info["background"], info["divider"], info["result"]) == (0, 4, 3)
            and info["upper_color"] in (None, 2)
            and info["lower_color"] in (None, 1)
            for info in training + testing
        )
        for info in testing:
            density = float(info["lower_mask"].mean())
            density_similar += min(abs(density - value) for value in training_lower) <= 0.10
            density_above += density > max(training_lower)
            density_below += density < min(training_lower)
            mixed_combinations.add((
                _relation(info), info["height"] * info["width"] >= 100,
                density >= 0.60, info["background"] != first["background"],
            ))
    _require(density_similar >= 10 and density_above >= 4 and density_below >= 4,
             "lower density relationships are still locked to one shift")
    _require(visible_swaps >= 10, "foreground color swaps are missing")
    _require(classic_puzzles >= 3, "source-like fixed-palette easy cases are missing")
    _require(len(mixed_combinations) >= 20,
             "size, relationship, density, and palette variation rarely combine")
    _require(any(info["height"] == 4 and info["width"] == 5 for _, info, _ in infos),
             "source-like small panels absent")
    _require(any(info["height"] >= 11 and info["width"] >= 11 for _, info, _ in infos),
             "large panels absent")
