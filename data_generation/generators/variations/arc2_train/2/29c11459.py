"""Odd-span endpoint propagation, with a learned midpoint color."""

import numpy as np


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _source_grid(rows):
    return np.asarray([[int(c) for c in row] for row in rows], dtype=np.int8)


_SOURCE_ROWS = (
    (("00000000000", "00000000000", "00000000000", "30000000007", "00000000000"),
     ("00000000000", "00000000000", "00000000000", "33333577777", "00000000000")),
    (("00000000000", "10000000002", "00000000000", "00000000000", "00000000000"),
     ("00000000000", "11111522222", "00000000000", "00000000000", "00000000000")),
    (("00000000000", "40000000008", "00000000000", "00000000000", "60000000009"),
     ("00000000000", "44444588888", "00000000000", "00000000000", "66666599999")),
)
_SOURCE_PAIRS = tuple((_source_grid(a), _source_grid(b)) for a, b in _SOURCE_ROWS)


def _is_source_pair(inp, out):
    return any(np.array_equal(inp, a) and np.array_equal(out, b)
               for a, b in _SOURCE_PAIRS)


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _sample_length(rng, minimum=3):
    # The two dimensions are sampled independently, so thin and broad grids occur.
    bucket = int(rng.choice(4, p=[0.22, 0.38, 0.32, 0.08]))
    ranges = ((3, 7), (8, 17), (18, 29), (30, 30))
    low, high = ranges[bucket]
    if high < minimum:
        low, high = minimum, max(minimum, 17)
    return int(rng.integers(max(low, minimum), high + 1))


def _sample_line_count(rng, lane_count, regime, forced=None):
    maximum = min(6, lane_count - 1)
    if forced is not None:
        if forced > maximum:
            raise ValueError("Forced line count exceeds available lanes")
        return forced
    choices = np.arange(1, maximum + 1)
    if regime == "source_train":
        weights = np.asarray([0.47 if n == 1 else 0.28 if n == 2 else 0.25 / (maximum - 2)
                              for n in choices], dtype=float) if maximum > 2 else np.asarray([0.6, 0.4][:maximum], dtype=float)
    else:
        weights = np.asarray([0.20 if n == 1 else 0.31 if n == 2 else 0.49 / (maximum - 2)
                              for n in choices], dtype=float) if maximum > 2 else np.asarray([0.38, 0.62][:maximum], dtype=float)
    weights /= weights.sum()
    return int(rng.choice(choices, p=weights))


def _sample_lanes(rng, lane_count, count, full_lines=False):
    if full_lines and count == 2:
        # Odd separation makes the two full-length lines unambiguous when rotated.
        candidates = [(a, b) for a in range(lane_count) for b in range(a + 1, lane_count)
                      if (b - a) % 2 == 1]
        return tuple(candidates[int(rng.integers(len(candidates)))])
    choice = float(rng.random())
    if count == 1:
        if choice < 0.30:
            return (int(rng.choice([0, lane_count - 1])),)
        return (int(rng.integers(lane_count)),)
    if choice < 0.22:
        first = int(rng.integers(0, lane_count - count + 1))
        return tuple(range(first, first + count))
    if choice < 0.43:
        inner = rng.choice(np.arange(1, lane_count - 1), size=count - 2,
                           replace=False) if count > 2 else []
        return tuple(sorted((0, lane_count - 1, *[int(x) for x in inner])))
    return tuple(sorted(int(x) for x in rng.choice(lane_count, size=count, replace=False)))


def _sample_span(rng, axis_length, long_span=False, internal=False, full=False):
    maximum = (axis_length - 1) // 2
    if internal:
        maximum = min(maximum, (axis_length - 2) // 2)
    minimum = 2 if long_span else 1
    if maximum < minimum:
        raise ValueError("Axis too short for requested span")
    if full:
        half = maximum
        if 2 * half + 1 != axis_length:
            raise ValueError("Full span requires odd axis length")
    elif not long_span and rng.random() < 0.17:
        half = 1
    elif not internal and axis_length % 2 and rng.random() < 0.23:
        half = maximum
    else:
        half = int(rng.integers(minimum, maximum + 1))
    span_length = 2 * half + 1
    margin = axis_length - span_length
    start = int(rng.integers(margin + 1))
    return start, start + span_length - 1


def _endpoint_colors(rng, background, separator, history):
    legal = [c for c in range(10) if c not in (background, separator)]
    reusable = [pair for pair in history if pair[0] in legal and pair[1] in legal]
    if reusable and rng.random() < 0.48:
        a, b = reusable[int(rng.integers(len(reusable)))]
        return (b, a) if rng.random() < 0.35 else (a, b)
    values = rng.choice(legal, size=2, replace=False)
    return int(values[0]), int(values[1])


def _make_pair(rng, orientation, separator, background, regime, history,
               long_span=False, internal=False, full_lines=False):
    axis_length = _sample_length(rng, 6 if long_span and internal else
                                 5 if long_span or full_lines else 3)
    if full_lines and axis_length % 2 == 0:
        axis_length = axis_length - 1 if axis_length > 5 else axis_length + 1
    lane_count = _sample_length(rng, 3)
    count = _sample_line_count(rng, lane_count, regime, 2 if full_lines else None)
    lanes = _sample_lanes(rng, lane_count, count, full_lines)
    shape = (lane_count, axis_length) if orientation == "horizontal" else (axis_length, lane_count)
    inp = np.full(shape, background, dtype=np.int8)
    local_history = list(history)
    for lane in lanes:
        start, end = _sample_span(rng, axis_length, long_span, internal, full_lines)
        a, b = _endpoint_colors(rng, background, separator, local_history)
        if orientation == "horizontal":
            inp[lane, start], inp[lane, end] = a, b
        else:
            inp[start, lane], inp[end, lane] = a, b
        local_history.append((a, b))
    out = _predict(inp, background, orientation, separator)
    return inp, out, local_history[len(history):]


def _lines(inp, background, orientation):
    view = inp if orientation == "horizontal" else inp.T
    lines = []
    blank = 0
    for lane, row in enumerate(view):
        positions = np.flatnonzero(row != background)
        if len(positions) == 0:
            blank += 1
            continue
        if len(positions) != 2:
            return None
        a, b = (int(x) for x in positions)
        if b - a < 2 or (b - a) % 2 or int(row[a]) == int(row[b]):
            return None
        lines.append((lane, a, b, int(row[a]), int(row[b]), (b - a) // 2))
    if not lines or not blank:
        return None
    return tuple(lines)


def _predict(inp, background, orientation, separator):
    lines = _lines(inp, background, orientation)
    if lines is None:
        raise ValueError("Invalid endpoint geometry")
    out = inp.copy()
    view = out if orientation == "horizontal" else out.T
    for lane, a, b, left, right, half in lines:
        middle = a + half
        view[lane, a:middle] = left
        view[lane, middle] = separator
        view[lane, middle + 1:b + 1] = right
    return out


def _background(inp):
    colors, counts = np.unique(inp, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _inspect_pair(inp, out):
    background = _background(inp)
    candidates = [(orientation, lines) for orientation in ("horizontal", "vertical")
                  if (lines := _lines(inp, background, orientation)) is not None]
    _require(len(candidates) == 1, "Endpoint orientation is ambiguous or invalid")
    orientation, lines = candidates[0]
    _require(all(int(np.count_nonzero(inp == background)) > int(np.count_nonzero(inp == c))
                 for c in np.unique(inp) if c != background), "Background is not dominant")
    view = out if orientation == "horizontal" else out.T
    separators = {int(view[lane, a + half]) for lane, a, b, left, right, half in lines}
    _require(len(separators) == 1, "Separator changes within a pair")
    separator = next(iter(separators))
    _require(separator != background, "Separator equals background")
    _require(all(separator not in (left, right) for _, _, _, left, right, _ in lines),
             "Separator is used as an endpoint")
    _require(np.array_equal(_predict(inp, background, orientation, separator), out),
             "Output violates midpoint propagation or changes outer background")
    _require(not np.array_equal(inp, out), "Input and output are identical")
    return {"background": background, "separator": separator,
            "orientation": orientation, "lines": lines}


def _grid_key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    separator = int(rng.integers(10))
    orientation_mode = str(rng.choice(["horizontal", "vertical", "mixed"], p=[0.30, 0.29, 0.41]))
    fixed_background = int(rng.choice([c for c in range(10) if c != separator])) if rng.random() < 0.16 else None
    classic = rng.random() < 0.17
    if orientation_mode == "mixed":
        first = str(rng.choice(["horizontal", "vertical"]))
        other = "vertical" if first == "horizontal" else "horizontal"
        train_orientations = [first, other] + [str(rng.choice([first, other])) for _ in range(train_count - 2)]
        test_orientations = [str(rng.choice([first, other])) for _ in range(test_count)]
    else:
        train_orientations = [orientation_mode] * train_count
        test_orientations = [orientation_mode] * test_count
    pair_seeds = np.random.SeedSequence([int(x) for x in rng.integers(0, 2**32, 4,
                                                                      dtype=np.uint64)]).spawn(train_count + test_count)
    puzzle = {"train": [], "test": []}
    seen = set()
    history = []
    for index, (regime, orientation) in enumerate(
            [("source_train", x) for x in train_orientations] +
            [("target_test", x) for x in test_orientations]):
        pair_rng = np.random.default_rng(pair_seeds[index])
        background = fixed_background if fixed_background is not None else int(pair_rng.choice(
            [c for c in range(10) if c != separator]))
        training_index = index if regime == "source_train" else -1
        need_demo = training_index == 0 or (orientation_mode == "mixed" and training_index == 1)
        full_lines = classic and index == train_count
        for _ in range(96):
            inp, out, used_colors = _make_pair(pair_rng, orientation, separator, background,
                                               regime, history, long_span=need_demo,
                                               internal=need_demo, full_lines=full_lines)
            key = _grid_key(inp)
            if key in seen or _is_source_pair(inp, out):
                continue
            try:
                info = _inspect_pair(inp, out)
            except AssertionError:
                continue
            if info["orientation"] != orientation:
                continue
            puzzle[regime.replace("source_train", "train").replace("target_test", "test")].append(
                {"input": inp, "output": out})
            seen.add(key)
            history.extend(used_colors)
            break
        else:
            raise RuntimeError("Unable to construct an unambiguous distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "Splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "Illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Pair needs exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "Grid must be a 2D NumPy array")
                _require(all(1 <= d <= 30 for d in grid.shape), "Grid size outside ARC limits")
                _require(np.issubdtype(grid.dtype, np.integer), "Grid dtype is not integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "Illegal ARC color")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape, "Input/output shape differs")
            ikey = _grid_key(inp)
            pkey = ikey, _grid_key(out)
            _require(ikey not in seen_inputs and pkey not in seen_pairs, "Duplicate pair")
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            _require(not _is_source_pair(inp, out), "Exact source pair copied")
            infos[split].append(_inspect_pair(inp, out))
    separators = {info["separator"] for split in infos.values() for info in split}
    _require(len(separators) == 1, "Separator is not puzzle-wide")
    separator = next(iter(separators))
    train_orientations = {info["orientation"] for info in infos["train"]}
    _require({info["orientation"] for info in infos["test"]} <= train_orientations,
             "Test orientation is not demonstrated in training")
    for orientation in train_orientations:
        _require(any(info["orientation"] == orientation and
                     any(half >= 2 and (a > 0 or b < (pair["input"].shape[1]
                         if orientation == "horizontal" else pair["input"].shape[0]) - 1)
                         for _, a, b, _, _, half in info["lines"])
                     for pair, info in zip(puzzle["train"], infos["train"])),
                 "Training lacks an internal, extending span for an orientation")
    # Predict each test output from training pairs and the test input only.
    learned = {info["separator"] for info in infos["train"]}
    _require(len(learned) == 1, "Training does not establish the separator")
    learned_separator = next(iter(learned))
    for pair in puzzle["test"]:
        inp = pair["input"]
        bg = _background(inp)
        options = [axis for axis in ("horizontal", "vertical") if _lines(inp, bg, axis) is not None]
        _require(len(options) == 1, "Test input has ambiguous orientation")
        predicted = _predict(inp, bg, options[0], learned_separator)
        _require(np.array_equal(predicted, pair["output"]),
                 "Test output is not recoverable from training")


def _puzzles_equal(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        np.array_equal(x[name], y[name]) for x, y in zip(a[split], b[split])
        for name in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    _require(_puzzles_equal(generate_puzzle(314159, 7, 3),
                           generate_puzzle(314159, 7, 3)), "Nondeterministic generation")
    for seed, train, test in ((1201, 2, 1), (1202, 10, 1), (1203, 10, 4),
                              (1204, 6, 2), (1301, None, 1), (1302, None, 4),
                              (1303, 2, None), (1304, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _require(train is None or len(puzzle["train"]) == train, "Training count ignored")
        _require(test is None or len(puzzle["test"]) == test, "Test count ignored")
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 2, "num_test": 3}, {"num_train": 4, "num_test": 4},
                    {"num_train": True}, {"num_test": 2.5}):
        try:
            generate_puzzle(99173, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal request accepted: {request}")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = {"train_counts": set(), "test_counts": set(), "modes": set(),
               "backgrounds": set(), "separators": set(), "endpoint_colors": set(),
               "shapes": set(), "layouts": set(), "signatures": set()}
    observations = {name: 0 for name in ("train_multi", "test_single", "test_three",
                    "adjacent", "edge", "middle", "unequal_gaps", "span_three",
                    "full_span", "internal", "asymmetric", "reused_within",
                    "reused_across", "reversed", "black_endpoint", "black_separator",
                    "nonblack_background", "compact", "large", "tall", "wide",
                    "combined", "background_varies", "background_fixed",
                    "line_count_varies", "horizontal_to_vertical", "vertical_to_horizontal",
                    "two_full_test_lines", "same_colors_new_geometry", "dimension_30")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        metrics["train_counts"].add(len(puzzle["train"]))
        metrics["test_counts"].add(len(puzzle["test"]))
        orientations = set()
        train_axes, test_axes = set(), set()
        puzzle_backgrounds, puzzle_line_counts = set(), set()
        prior_colors = []
        puzzle_signature = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                info = _inspect_pair(inp, out)
                axis = info["orientation"]
                orientations.add(axis)
                (train_axes if split == "train" else test_axes).add(axis)
                lines = info["lines"]
                background = info["background"]
                puzzle_backgrounds.add(background)
                puzzle_line_counts.add(len(lines))
                metrics["backgrounds"].add(background)
                metrics["separators"].add(info["separator"])
                metrics["endpoint_colors"].update(c for line in lines for c in line[3:5])
                metrics["shapes"].add(inp.shape)
                metrics["layouts"].add((axis, inp.shape,
                    tuple((line[0], line[1], line[2]) for line in lines)))
                puzzle_signature.append(_grid_key(inp))
                extent = inp.shape[1] if axis == "horizontal" else inp.shape[0]
                lanes = inp.shape[0] if axis == "horizontal" else inp.shape[1]
                lane_ids = [line[0] for line in lines]
                colors = [(line[3], line[4]) for line in lines]
                for row in range(inp.shape[0]):
                    for col in range(inp.shape[1]):
                        lane, pos = (row, col) if axis == "horizontal" else (col, row)
                        expected = int(inp[row, col])
                        for line_lane, a, b, left, right, half in lines:
                            if lane == line_lane and a <= pos <= b:
                                midpoint = a + half
                                expected = (left if pos < midpoint else
                                            info["separator"] if pos == midpoint else right)
                                break
                        _require(int(out[row, col]) == expected,
                                 "Independent per-cell transformation check failed")
                if split == "train" and len(lines) > 1: observations["train_multi"] += 1
                if split == "test" and len(lines) == 1: observations["test_single"] += 1
                if split == "test" and len(lines) >= 3: observations["test_three"] += 1
                if split == "test" and len(lines) == 2 and all(
                        line[1] == 0 and line[2] == extent-1 for line in lines):
                    observations["two_full_test_lines"] += 1
                if any(b-a == 1 for a, b in zip(lane_ids, lane_ids[1:])): observations["adjacent"] += 1
                if 0 in lane_ids or lanes-1 in lane_ids: observations["edge"] += 1
                if any(0 < n < lanes-1 for n in lane_ids): observations["middle"] += 1
                if len(lane_ids) >= 3 and len(set(b-a for a,b in zip(lane_ids,lane_ids[1:]))) > 1:
                    observations["unequal_gaps"] += 1
                for lane, a, b, left, right, half in lines:
                    if half == 1: observations["span_three"] += 1
                    if a == 0 and b == extent-1: observations["full_span"] += 1
                    if a > 0 or b < extent-1: observations["internal"] += 1
                    if a != extent-1-b: observations["asymmetric"] += 1
                    if left == 0 or right == 0: observations["black_endpoint"] += 1
                if len(colors) != len(set(colors)): observations["reused_within"] += 1
                if any(colors[i] == colors[j] and
                       (lines[i][1], lines[i][2]) != (lines[j][1], lines[j][2])
                       for i in range(len(lines)) for j in range(i+1, len(lines))):
                    observations["same_colors_new_geometry"] += 1
                if any(c in prior_colors for c in colors): observations["reused_across"] += 1
                if any((b,a) in prior_colors for a,b in colors): observations["reversed"] += 1
                prior_colors.extend(colors)
                if info["separator"] == 0: observations["black_separator"] += 1
                if background != 0: observations["nonblack_background"] += 1
                if min(inp.shape) <= 6: observations["compact"] += 1
                if max(inp.shape) >= 27: observations["large"] += 1
                if 30 in inp.shape: observations["dimension_30"] += 1
                if inp.shape[0] >= 2 * inp.shape[1]: observations["tall"] += 1
                if inp.shape[1] >= 2 * inp.shape[0]: observations["wide"] += 1
                if len(lines) >= 2 and background != 0 and any(l[5] == 1 for l in lines):
                    observations["combined"] += 1
        metrics["modes"].add("mixed" if len(orientations) == 2 else next(iter(orientations)))
        observations["background_varies" if len(puzzle_backgrounds) > 1 else "background_fixed"] += 1
        if len(puzzle_line_counts) > 1: observations["line_count_varies"] += 1
        if "horizontal" in train_axes and "vertical" in test_axes:
            observations["horizontal_to_vertical"] += 1
        if "vertical" in train_axes and "horizontal" in test_axes:
            observations["vertical_to_horizontal"] += 1
        metrics["signatures"].add(tuple(puzzle_signature))
    _require(len(metrics["train_counts"]) >= 4 and len(metrics["test_counts"]) >= 3,
             "Automatic counts lack diversity")
    _require(metrics["modes"] == {"horizontal", "vertical", "mixed"},
             "Orientation regimes lack diversity")
    _require(len(metrics["backgrounds"]) == 10 and len(metrics["endpoint_colors"]) == 10,
             "Color roles do not span the ARC palette")
    _require(len(metrics["separators"]) >= 8, "Separator palette lacks diversity")
    _require(len(metrics["shapes"]) >= 80 and len(metrics["layouts"]) >= 150 and
             len(metrics["signatures"]) == 42, "Structural diversity is too low")
    _require(all(value > 0 for value in observations.values()),
             f"Review feature absent: {observations}")
