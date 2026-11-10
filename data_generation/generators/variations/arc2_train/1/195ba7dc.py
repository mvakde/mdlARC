import numpy as np


def _rows_to_array(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = (
    (_rows_to_array(("7077072770000", "7007002000700", "0770002007700", "0777702700070", "7070772777777")),
     _rows_to_array(("111101", "100100", "011100", "111110", "111111"))),
    (_rows_to_array(("0777072770707", "0007072077707", "7070002770000", "7770002770077", "0770772777007")),
     _rows_to_array(("111101", "011101", "111000", "111011", "111011"))),
    (_rows_to_array(("7700702077777", "7000772700777", "0700702000000", "7707772707000", "7707702777000")),
     _rows_to_array(("111111", "100111", "010010", "111111", "111110"))),
    (_rows_to_array(("0770772700007", "7000072700770", "7077072700700", "0700002707070", "7707702070070")),
     _rows_to_array(("111011", "100111", "101101", "111010", "110110"))),
    (_rows_to_array(("7700002077007", "0700002700707", "7770072077707", "0000002777700", "0077702077077")),
     _rows_to_array(("111001", "110101", "111101", "111100", "011111"))),
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _validate_optional_count(value: int | None, name: str, lower: int, upper: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


def _resolve_counts(rng: np.random.Generator, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train_count = _validate_optional_count(num_train, "num_train", 2, 10)
    test_count = _validate_optional_count(num_test, "num_test", 1, 4)
    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must exceed num_test")
        return train_count, test_count
    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(test_count + 1, 11))
    elif train_count is None:
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    else:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    return train_count, test_count


def _sample_convention(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.permutation(10)]
    background, base_marker, base_separator = colors[:3]
    result_kind = int(rng.choice(3, p=[0.68, 0.22, 0.10]))
    result = (colors[3], base_marker, base_separator)[result_kind]
    orientation_mode = str(rng.choice(
        ["vertical", "horizontal", "mixed", "v_to_h", "h_to_v"],
        p=[0.27, 0.20, 0.29, 0.12, 0.12],
    ))
    shift_mode = str(rng.choice(["steady", "up", "down"], p=[0.46, 0.27, 0.27]))
    if shift_mode == "up":
        base_overlap = float(rng.uniform(0.08, 0.56))
    elif shift_mode == "down":
        base_overlap = float(rng.uniform(0.44, 0.92))
    else:
        base_overlap = float(rng.uniform(0.12, 0.88))
    return {
        "background": background,
        "base_marker": base_marker,
        "base_separator": base_separator,
        "result": result,
        "variable_marker": bool(rng.random() < 0.68),
        "orientation_mode": orientation_mode,
        "shift_mode": shift_mode,
        "base_overlap": base_overlap,
    }


def _orientation(mode: str, split: str, index: int) -> str:
    if mode == "vertical":
        return "vertical"
    if mode == "horizontal":
        return "horizontal"
    if mode == "v_to_h":
        return "vertical" if split == "train" else "horizontal"
    if mode == "h_to_v":
        return "horizontal" if split == "train" else "vertical"
    return "vertical" if (index + (split == "test")) % 2 == 0 else "horizontal"


def _sample_shape(rng: np.random.Generator, orientation: str, first: bool) -> tuple[int, int]:
    if first:
        return int(rng.integers(4, 10)), int(rng.integers(4, 10))
    category = str(rng.choice(
        ["small", "regular", "tall", "wide", "large"],
        p=[0.16, 0.32, 0.18, 0.18, 0.16],
    ))
    if orientation == "vertical":
        ranges = {
            "small": ((2, 4), (2, 4)),
            "regular": ((4, 10), (4, 9)),
            "tall": ((11, 30), (2, 7)),
            "wide": ((2, 7), (9, 14)),
            "large": ((10, 30), (8, 14)),
        }
    else:
        ranges = {
            "small": ((2, 4), (2, 4)),
            "regular": ((4, 9), (4, 10)),
            "tall": ((9, 14), (2, 7)),
            "wide": ((2, 7), (11, 30)),
            "large": ((8, 14), (10, 30)),
        }
    (lo_h, hi_h), (lo_w, hi_w) = ranges[category]
    return int(rng.integers(lo_h, hi_h + 1)), int(rng.integers(lo_w, hi_w + 1))


def _spatial_scores(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    style = str(rng.choice(
        ["scatter", "islands", "connected", "bars", "motifs", "holes", "band"],
        p=[0.23, 0.21, 0.17, 0.15, 0.11, 0.06, 0.07],
    ))
    noise = rng.random((height, width))
    rows, cols = np.indices((height, width))
    if style == "scatter":
        return noise
    if style == "islands":
        score = np.zeros((height, width), dtype=float)
        for _ in range(int(rng.integers(2, 6))):
            cy = int(rng.integers(0, height))
            cx = int(rng.integers(0, width))
            ry = float(rng.uniform(1.0, max(1.5, height / 3)))
            rx = float(rng.uniform(1.0, max(1.5, width / 3)))
            patch = 1.0 - ((rows - cy) / ry) ** 2 - ((cols - cx) / rx) ** 2
            score = np.maximum(score, patch)
        return score + 0.22 * noise
    if style == "connected":
        score = 0.2 * noise
        y, x = int(rng.integers(height)), int(rng.integers(width))
        for step in range(max(5, height * width // 2)):
            score[y, x] += 1.0 + 0.002 * step
            dy, dx = ((1, 0), (-1, 0), (0, 1), (0, -1))[int(rng.integers(4))]
            y = min(height - 1, max(0, y + dy))
            x = min(width - 1, max(0, x + dx))
        return score
    if style == "bars":
        score = 0.28 * noise
        for _ in range(int(rng.integers(2, 6))):
            if bool(rng.integers(2)):
                y = int(rng.integers(height))
                x0 = int(rng.integers(width))
                length = int(rng.integers(1, width + 1))
                score[y, x0:min(width, x0 + length)] += float(rng.uniform(0.8, 1.5))
            else:
                x = int(rng.integers(width))
                y0 = int(rng.integers(height))
                length = int(rng.integers(1, height + 1))
                score[y0:min(height, y0 + length), x] += float(rng.uniform(0.8, 1.5))
        return score
    if style == "motifs":
        score = 0.23 * noise
        mh, mw = min(height, 3), min(width, 3)
        motif = rng.random((mh, mw)) < float(rng.uniform(0.35, 0.7))
        if not np.any(motif):
            motif[0, 0] = True
        for _ in range(int(rng.integers(2, 7))):
            y = int(rng.integers(0, height - mh + 1))
            x = int(rng.integers(0, width - mw + 1))
            score[y:y + mh, x:x + mw] += motif * float(rng.uniform(0.8, 1.2))
        return score
    if style == "holes":
        score = 0.45 + 0.35 * noise
        for _ in range(int(rng.integers(1, 4))):
            cy, cx = int(rng.integers(height)), int(rng.integers(width))
            radius = float(rng.uniform(1.0, max(1.5, min(height, width) / 3)))
            hole = ((rows - cy) ** 2 + (cols - cx) ** 2) <= radius ** 2
            score[hole] -= float(rng.uniform(0.65, 1.1))
        return score
    direction = int(rng.integers(3))
    base = (rows / max(1, height - 1), cols / max(1, width - 1),
            rows / max(1, height - 1) + cols / max(1, width - 1))[direction]
    if bool(rng.integers(2)):
        base = -base
    return base + 0.18 * noise


def _top_mask(scores: np.ndarray, count: int, allowed: np.ndarray | None = None) -> np.ndarray:
    result = np.zeros(scores.shape, dtype=bool)
    positions = np.flatnonzero(allowed.ravel()) if allowed is not None else np.arange(scores.size)
    if count:
        chosen = positions[np.argsort(scores.ravel()[positions], kind="stable")[-count:]]
        result.ravel()[chosen] = True
    return result


def _density_profile(rng: np.random.Generator, informative: bool) -> tuple[float, float, str]:
    if informative:
        return float(rng.uniform(0.38, 0.62)), float(rng.uniform(0.38, 0.62)), "informative"
    profile = str(rng.choice(
        ["sparse", "medium", "dense", "left_sparse", "right_sparse", "near_identical", "unrestricted"],
        p=[0.16, 0.21, 0.17, 0.13, 0.13, 0.09, 0.11],
    ))
    if profile == "sparse":
        return float(rng.uniform(0.03, 0.23)), float(rng.uniform(0.03, 0.23)), profile
    if profile == "medium":
        return float(rng.uniform(0.25, 0.64)), float(rng.uniform(0.25, 0.64)), profile
    if profile == "dense":
        return float(rng.uniform(0.66, 0.94)), float(rng.uniform(0.66, 0.94)), profile
    if profile == "left_sparse":
        return float(rng.uniform(0.03, 0.24)), float(rng.uniform(0.65, 0.94)), profile
    if profile == "right_sparse":
        return float(rng.uniform(0.65, 0.94)), float(rng.uniform(0.03, 0.24)), profile
    if profile == "near_identical":
        p = float(rng.uniform(0.25, 0.78))
        return p, float(np.clip(p + rng.uniform(-0.09, 0.09), 0.03, 0.97)), profile
    return float(rng.uniform(0.04, 0.96)), float(rng.uniform(0.04, 0.96)), profile


def _make_masks(rng: np.random.Generator, height: int, width: int,
                overlap_bias: float, special: str | None, informative: bool) -> tuple[np.ndarray, np.ndarray]:
    size = height * width
    left_density, right_density, profile = _density_profile(rng, informative)
    left_count = int(np.clip(round(left_density * size), 0, size))
    right_count = int(np.clip(round(right_density * size), 0, size))
    left_score = _spatial_scores(rng, height, width)
    right_score = _spatial_scores(rng, height, width)
    left = _top_mask(left_score, left_count)
    if special == "empty_union":
        return np.zeros_like(left), np.zeros_like(left)
    if special == "empty_operand":
        if bool(rng.integers(2)):
            return left, np.zeros_like(left)
        return np.zeros_like(left), _top_mask(right_score, right_count)
    if special == "identical":
        return left, left.copy()
    if special == "subset":
        if bool(rng.integers(2)):
            return left, _top_mask(right_score, int(rng.integers(1, max(2, left_count + 1))), left)
        right = _top_mask(right_score, right_count)
        return _top_mask(left_score, int(rng.integers(1, max(2, right_count + 1))), right), right
    if special == "full_union":
        extra = _top_mask(right_score, int(rng.integers(0, left_count + 1)), left)
        return left, (~left) | extra
    if special == "disjoint":
        right_count = min(right_count, size - left_count)
        return left, _top_mask(right_score, right_count, ~left)

    low = max(0, left_count + right_count - size)
    high = min(left_count, right_count)
    if informative:
        low = max(low, 1, left_count + right_count - size + 1)
        high = min(high, left_count - 1, right_count - 1)
    if profile == "near_identical":
        overlap_bias = max(overlap_bias, 0.86)
    overlap_count = int(np.clip(round(low + overlap_bias * (high - low)), low, high))
    right = _top_mask(right_score, overlap_count, left)
    right |= _top_mask(right_score, right_count - overlap_count, ~left)
    return left, right


def _pair_colors(rng: np.random.Generator, convention: dict, informative: bool) -> tuple[int, int, int]:
    background = convention["background"]
    base_marker = convention["base_marker"]
    if informative or not convention["variable_marker"] or rng.random() < 0.32:
        left_marker = right_marker = base_marker
    else:
        allowed = [color for color in range(10) if color != background]
        left_marker = int(rng.choice(allowed))
        right_marker = int(rng.choice(allowed)) if rng.random() < 0.42 else left_marker
    separator_options = [color for color in range(10)
                         if color not in (background, left_marker, right_marker)]
    if convention["base_separator"] in separator_options and rng.random() < 0.48:
        separator = convention["base_separator"]
    else:
        separator = int(rng.choice(separator_options))
    return left_marker, right_marker, separator


def _generate_pair(rng: np.random.Generator, convention: dict, shape: tuple[int, int],
                   orientation: str, split: str, informative: bool, special: str | None) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    bias = convention["base_overlap"]
    if split == "test":
        bias += {"up": 0.42, "down": -0.42, "steady": 0.0}[convention["shift_mode"]]
    bias = float(np.clip(bias + rng.uniform(-0.17, 0.17), 0.0, 1.0))
    left, right = _make_masks(rng, height, width, bias, special, informative)
    background, result = convention["background"], convention["result"]
    left_marker, right_marker, separator = _pair_colors(rng, convention, informative)
    first = np.where(left, left_marker, background).astype(np.int8)
    second = np.where(right, right_marker, background).astype(np.int8)
    if orientation == "vertical":
        line = np.full((height, 1), separator, dtype=np.int8)
        input_grid = np.concatenate((first, line, second), axis=1)
    else:
        line = np.full((1, width), separator, dtype=np.int8)
        input_grid = np.concatenate((first, line, second), axis=0)
    output_grid = np.where(left | right, result, background).astype(np.int8)
    return input_grid, output_grid


def _array_key(array: np.ndarray) -> tuple:
    return (tuple(int(v) for v in array.shape), tuple(int(v) for v in array.ravel()))


def _is_source_duplicate(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return any(np.array_equal(input_grid, source_input) and np.array_equal(output_grid, source_output)
               for source_input, source_output in _SOURCE_PAIRS)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    convention = _sample_convention(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_shapes = []
    special_kinds = ("disjoint", "identical", "subset", "empty_operand", "full_union", "empty_union")
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            informative = split == "train" and index == 0
            orientation = _orientation(convention["orientation_mode"], split, index)
            for attempt in range(40):
                child_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                if split == "test" and child_rng.random() < 0.26:
                    eligible = [shape for shape in train_shapes
                                if (orientation == "vertical" and shape[1] <= 14) or
                                   (orientation == "horizontal" and shape[0] <= 14)]
                    shape = eligible[int(child_rng.integers(len(eligible)))] if eligible else _sample_shape(child_rng, orientation, False)
                else:
                    shape = _sample_shape(child_rng, orientation, informative)
                if split == "train" and index == 1 and shape == train_shapes[0]:
                    continue
                special = None
                if not informative and child_rng.random() < (0.30 if split == "train" else 0.36):
                    special = str(child_rng.choice(special_kinds))
                input_grid, output_grid = _generate_pair(child_rng, convention, shape, orientation,
                                                         split, informative, special)
                key = _array_key(input_grid)
                if key in seen_inputs or _is_source_duplicate(input_grid, output_grid):
                    continue
                seen_inputs.add(key)
                puzzle[split].append({"input": input_grid, "output": output_grid})
                if split == "train":
                    train_shapes.append(shape)
                break
            else:
                raise RuntimeError("unable to construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _split_panels(pair: dict) -> tuple[str, np.ndarray, np.ndarray, int]:
    input_grid, output_grid = pair["input"], pair["output"]
    oh, ow = output_grid.shape
    if input_grid.shape == (oh, 2 * ow + 1):
        first, line, second = input_grid[:, :ow], input_grid[:, ow], input_grid[:, ow + 1:]
        orientation = "vertical"
    elif input_grid.shape == (2 * oh + 1, ow):
        first, line, second = input_grid[:oh, :], input_grid[oh, :], input_grid[oh + 1:, :]
        orientation = "horizontal"
    else:
        raise AssertionError("input must contain two equal aligned panels and a one-cell separator")
    _require(np.all(line == line[0]), "separator must be a uniform complete line")
    separator = int(line[0])
    _require(not np.any(first == separator) and not np.any(second == separator),
             "separator ink must be distinct from both panels")
    return orientation, first, second, separator


def _inspect_pair(pair: dict, background: int, result: int) -> dict:
    orientation, first, second, separator = _split_panels(pair)
    _require(separator != background, "separator must differ from background")
    for panel in (first, second):
        _require(len(set(int(x) for x in np.unique(panel)) - {background}) <= 1,
                 "each panel may have only one foreground color")
    left, right = first != background, second != background
    expected = np.where(left | right, result, background)
    _require(np.array_equal(pair["output"], expected), "output is not aligned cellwise OR")
    states = left.astype(np.uint8) + 2 * right.astype(np.uint8)
    counts = tuple(int(np.count_nonzero(states == state)) for state in range(4))
    return {"orientation": orientation, "shape": first.shape, "separator": separator,
            "counts": counts, "states": states,
            "markers": (tuple(int(x) for x in np.unique(first) if x != background),
                        tuple(int(x) for x in np.unique(second) if x != background))}


def _infer_convention(train: list) -> tuple[int, int]:
    candidates = []
    for background in range(10):
        for result in range(10):
            if background == result:
                continue
            try:
                information = [_inspect_pair(pair, background, result) for pair in train]
            except AssertionError:
                continue
            observed = {state for info in information for state, count in enumerate(info["counts"]) if count}
            if observed == {0, 1, 2, 3}:
                candidates.append((background, result))
    _require(len(candidates) == 1, "training must uniquely establish background, result color, and all OR states")
    return candidates[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "split types")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "pair counts")
    seen_inputs = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, f"{name} array")
                _require(all(1 <= side <= 30 for side in grid.shape), f"{name} bounds")
                _require(np.issubdtype(grid.dtype, np.integer), f"{name} integer dtype")
                _require(np.all((grid >= 0) & (grid <= 9)), f"{name} ARC colors")
            key = _array_key(pair["input"])
            _require(key not in seen_inputs, "duplicate input")
            seen_inputs.add(key)
            _require(not _is_source_duplicate(pair["input"], pair["output"]), "source pair copy")
            _split_panels(pair)
    background, result = _infer_convention(puzzle["train"])
    train_info = [_inspect_pair(pair, background, result) for pair in puzzle["train"]]
    test_info = [_inspect_pair(pair, background, result) for pair in puzzle["test"]]
    _require(all(count > 0 for count in train_info[0]["counts"]), "first demonstration must be informative")
    _require(len({info["shape"] for info in train_info + test_info}) >= 2,
             "pairs must vary panel dimensions")


def _fingerprint(puzzle: dict) -> tuple:
    return tuple(tuple((_array_key(pair["input"]), _array_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(314159)
    _require(_fingerprint(automatic) == _fingerprint(generate_puzzle(314159)), "automatic determinism")
    explicit = generate_puzzle(271828, 7, 3)
    _require(len(explicit["train"]) == 7 and len(explicit["test"]) == 3, "explicit counts")
    _require(_fingerprint(explicit) == _fingerprint(generate_puzzle(271828, 7, 3)), "explicit determinism")
    for index, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(5000 + index, ntrain, ntest)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest), "boundary counts")
    for index, (ntrain, ntest) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(6000 + index, ntrain, ntest)
        _require(ntrain is None or len(puzzle["train"]) == ntrain, "partial train count")
        _require(ntest is None or len(puzzle["test"]) == ntest, "partial test count")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(7000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "cross-seed difference")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    shapes, orientations, backgrounds, results, separators = set(), set(), set(), set(), set()
    sparse = dense = asymmetric = disjoint = identical = subset = empty_operand = full_union = empty_union = 0
    near_identical = one_sparse_one_dense = narrow = tall = wide = nonbanded = 0
    reused_marker = reused_separator = horizontal_special = large_sparse = panel_color_difference = 0
    varying_separators = varying_markers = 0
    mixed = horizontal_only = vertical_only = 0
    overlap_directions = set()
    shifts = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        background, result = _infer_convention(puzzle["train"])
        backgrounds.add(background)
        results.add(result)
        info = {split: [_inspect_pair(pair, background, result) for pair in puzzle[split]]
                for split in ("train", "test")}
        used = {x["orientation"] for split in info.values() for x in split}
        mixed += used == {"vertical", "horizontal"}
        horizontal_only += used == {"horizontal"}
        vertical_only += used == {"vertical"}
        all_info = info["train"] + info["test"]
        varying_separators += len({x["separator"] for x in all_info}) > 1
        varying_markers += len({x["markers"] for x in all_info}) > 1
        train_orientations = {x["orientation"] for x in info["train"]}
        test_orientations = {x["orientation"] for x in info["test"]}
        if len(train_orientations) == len(test_orientations) == 1 and train_orientations != test_orientations:
            shifts.add((next(iter(train_orientations)), next(iter(test_orientations))))
        train_overlap = np.mean([x["counts"][3] / np.prod(x["shape"]) for x in info["train"]])
        test_overlap = np.mean([x["counts"][3] / np.prod(x["shape"]) for x in info["test"]])
        overlap_directions.add("up" if test_overlap > train_overlap + 0.08 else
                               "down" if train_overlap > test_overlap + 0.08 else "steady")
        for split in ("train", "test"):
            for pair, item in zip(puzzle[split], info[split]):
                shapes.add(item["shape"])
                orientations.add(item["orientation"])
                separators.add(item["separator"])
                n0, n1, n2, n3 = item["counts"]
                area = sum(item["counts"])
                sparse += (area - n0) / area < 0.36
                dense += (area - n0) / area > 0.84
                one_sparse_one_dense += min(n1 + n3, n2 + n3) / area <= 0.25 and max(n1 + n3, n2 + n3) / area >= 0.65
                asymmetric += max(n1, n2) >= 3 * max(1, min(n1, n2)) and max(n1, n2) >= 3
                disjoint += n3 == 0 and n1 > 0 and n2 > 0
                identical += n1 == 0 and n2 == 0 and n3 > 0
                subset += (n1 == 0 or n2 == 0) and n3 > 0 and (n1 + n2) > 0
                empty_operand += (n1 + n3 == 0 or n2 + n3 == 0) and (n1 + n2 + n3) > 0
                full_union += n0 == 0
                empty_union += n0 == area
                near_identical += n3 / max(1, area - n0) >= 0.85 and n3 > 0
                height, width = item["shape"]
                narrow += min(height, width) <= 3 and max(height, width) >= 8
                tall += height >= 2 * width and height >= 10
                wide += width >= 2 * height and width >= 10
                output = pair["output"]
                nonbanded += area >= 30 and len({tuple(row) for row in output}) > 3 and len({tuple(col) for col in output.T}) > 3
                reused_marker += result in item["markers"][0] or result in item["markers"][1]
                reused_separator += result == item["separator"]
                horizontal_special += item["orientation"] == "horizontal" and any(count == 0 for count in item["counts"])
                large_sparse += area >= 100 and (area - n0) / area < 0.36
                panel_color_difference += bool(item["markers"][0] and item["markers"][1]
                                               and item["markers"][0] != item["markers"][1])
                # A separate reconstruction makes mask union explicit in this test.
                _, a, b, _ = _split_panels(pair)
                calculated = np.where((a != background) | (b != background), result, background)
                _require(np.array_equal(calculated, pair["output"]), "independent OR reconstruction")
    _require(len(shapes) >= 65 and orientations == {"vertical", "horizontal"}, "shape/orientation diversity")
    _require(mixed >= 5 and horizontal_only >= 3 and vertical_only >= 3, "orientation modes")
    _require(len(backgrounds) >= 8 and len(results) >= 8 and len(separators) >= 8, "palette diversity")
    _require(all(x >= 2 for x in (sparse, dense, asymmetric, disjoint, identical,
                                  subset, empty_operand, full_union, empty_union)), "mask diversity")
    _require(all(x >= 2 for x in (near_identical, one_sparse_one_dense, narrow, tall, wide,
                                  nonbanded, reused_marker, reused_separator, horizontal_special,
                                  large_sparse, panel_color_difference)), "composed variation diversity")
    _require(varying_separators >= 5 and varying_markers >= 5, "within-puzzle palette variation")
    _require(shifts == {("vertical", "horizontal"), ("horizontal", "vertical")},
             "bidirectional orientation shifts")
    _require(overlap_directions == {"up", "down", "steady"}, "overlap shift diversity")
