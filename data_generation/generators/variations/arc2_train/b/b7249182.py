"""Generate midpoint-box puzzles from two differently colored point markers."""
import numpy as np

_SOURCE_INPUT_SPECS = (
    ((10, 13), ((6, 1, 3), (6, 12, 1))),
    ((14, 10), ((1, 4, 2), (10, 4, 8))),
    ((10, 18), ((5, 3, 5), (5, 16, 8))),
    ((19, 9), ((1, 3, 7), (16, 3, 6))),
)
_DEPTHS = (4, 6, 8, 10)
_SPANS = (3, 5, 7, 9)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _info(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "invalid input")
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) == 3, "input needs background and two marker colors")
    background_index = int(np.argmax(counts))
    background = int(values[background_index])
    _require(int(counts[background_index]) == grid.size - 2, "input needs two points")
    _require(all(int(n) == 1 for i, n in enumerate(counts) if i != background_index),
             "marker colors must occur once")
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != background)]
    same_row = markers[0][0] == markers[1][0]
    same_col = markers[0][1] == markers[1][1]
    _require(same_row != same_col, "markers must share one axis")
    orientation = "horizontal" if same_row else "vertical"
    axis = 1 if same_row else 0
    markers.sort(key=lambda point: point[axis])
    distance = markers[1][axis] - markers[0][axis]
    _require(distance >= 7 and distance % 2 == 1, "invalid separation")
    return {"background": background, "markers": tuple(markers),
            "orientation": orientation, "separation": distance}


def _construct_output(input_grid, depth, span):
    info = _info(input_grid)
    _require(depth >= 4 and depth % 2 == 0, "invalid axial depth")
    _require(span >= 3 and span % 2 == 1, "invalid transverse span")
    first, second = info["markers"]
    horizontal = info["orientation"] == "horizontal"
    axis, cross = (1, 0) if horizontal else (0, 1)
    distance = info["separation"]
    _require(distance >= depth + 3, "connectors need a visible interior cell")
    start = (first[axis] + second[axis] - depth + 1) // 2
    side = first[cross] - span // 2
    _require(side >= 0 and side + span <= input_grid.shape[cross], "box exceeds canvas")
    _require(start > first[axis] and start + depth - 1 < second[axis], "box exceeds interval")
    output = input_grid.copy()
    color0, color1 = first[2], second[2]
    if horizontal:
        row = first[0]
        top, bottom = side, side + span - 1
        for edge in (top, bottom):
            output[edge, start:start + depth // 2] = color0
            output[edge, start + depth // 2:start + depth] = color1
        output[top:bottom + 1, start] = color0
        output[top:bottom + 1, start + depth - 1] = color1
        output[row, first[1]:start + 1] = color0
        output[row, start + depth - 1:second[1] + 1] = color1
    else:
        col = first[1]
        left, right = side, side + span - 1
        output[start, left:right + 1] = color0
        output[start + depth - 1, left:right + 1] = color1
        for edge in (left, right):
            output[start:start + depth // 2, edge] = color0
            output[start + depth // 2:start + depth, edge] = color1
        output[first[0]:start + 1, col] = color0
        output[start + depth - 1:second[0] + 1, col] = color1
    return output


def _fingerprint(grid):
    return grid.shape, grid.tobytes()


def _matches_source(grid):
    for shape, markers in _SOURCE_INPUT_SPECS:
        if grid.shape == shape:
            source = np.zeros(shape, dtype=np.int8)
            for row, col, color in markers:
                source[row, col] = color
            if np.array_equal(grid, source):
                return True
    return False


def _odds(low, high):
    return list(range(low + (low + 1) % 2, high + 1, 2))


def _dimensions(rng, distance, span):
    minimum = distance + 1
    mode = int(rng.integers(0, 5))
    if mode == 0:  # tight transverse fit
        across = int(rng.integers(span, min(30, span + 3) + 1))
        along = int(rng.integers(minimum, min(30, minimum + 5) + 1))
    elif mode == 1:  # square
        along = across = int(rng.integers(max(minimum, span), 31))
    elif mode == 2 and minimum < 30:  # across longer than along
        along = int(rng.integers(minimum, min(29, minimum + 6) + 1))
        across = int(rng.integers(max(span, along + 1), 31))
    elif mode == 3:  # along longer than across
        across = int(rng.integers(span, min(29, max(span, minimum - 1)) + 1))
        along = int(rng.integers(max(minimum, across + 1), 31))
    else:
        along = int(rng.integers(minimum, 31))
        across = int(rng.integers(span, 31))
    return along, across


def _sample_pair(rng, depth, span, orientation, distance, palette):
    along, across = _dimensions(rng, distance, span)
    shape = (across, along) if orientation == "horizontal" else (along, across)
    bg, color0, color1 = (int(v) for v in rng.choice(palette, 3, replace=False))
    grid = np.full(shape, bg, dtype=np.int8)
    end_margin = along - distance - 1
    if int(rng.integers(0, 4)) == 0:
        first_axis = 0 if int(rng.integers(0, 2)) == 0 else end_margin
    else:
        first_axis = int(rng.integers(0, end_margin + 1))
    cross_min, cross_max = span // 2, across - span // 2 - 1
    if int(rng.integers(0, 4)) == 0:
        cross_axis = cross_min if int(rng.integers(0, 2)) == 0 else cross_max
    else:
        cross_axis = int(rng.integers(cross_min, cross_max + 1))
    if orientation == "horizontal":
        grid[cross_axis, first_axis] = color0
        grid[cross_axis, first_axis + distance] = color1
    else:
        grid[first_axis, cross_axis] = color0
        grid[first_axis + distance, cross_axis] = color1
    return {"input": grid, "output": _construct_output(grid, depth, span)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    # Geometry is learned once; background and marker colors are visible per pair.
    depth = int(rng.choice(_DEPTHS, p=(.35, .28, .23, .14)))
    span = int(rng.choice(_SPANS, p=(.22, .38, .25, .15)))
    palette = tuple(int(x) for x in rng.choice(10, int(rng.integers(3, 11)), replace=False))
    orientation_mode = int(rng.integers(0, 5))
    if orientation_mode == 0:
        train_orientations = ["horizontal", "vertical"]
        train_orientations += ["horizontal" if int(rng.integers(0, 2)) == 0
                               else "vertical" for _ in range(train_count - 2)]
        rng.shuffle(train_orientations)
    else:
        orientation = "horizontal" if orientation_mode in (1, 3) else "vertical"
        train_orientations = [orientation] * train_count
    minimum_distance = depth + 3
    train_options = _odds(minimum_distance, min(23, minimum_distance + 10))
    chosen_train = [int(rng.choice(train_options)) for _ in range(train_count)]
    if len(set(chosen_train)) == 1:
        chosen_train[-1] = train_options[(train_options.index(chosen_train[0]) + 1)
                                           % len(train_options)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if section == "train":
                orientation, distance = train_orientations[index], chosen_train[index]
            else:
                if orientation_mode == 2 and index == 0:
                    orientation = "horizontal"  # vertical-to-horizontal transfer
                elif orientation_mode == 3:
                    orientation = "horizontal"  # same-axis horizontal puzzle
                elif orientation_mode == 4:
                    orientation = "vertical"  # same-axis vertical puzzle
                else:
                    orientation = "horizontal" if int(rng.integers(0, 2)) == 0 else "vertical"
                shift = int(rng.integers(0, 3))
                if shift == 0:
                    options = _odds(max(chosen_train) + 2, 27)
                elif shift == 1:
                    options = chosen_train
                else:
                    options = _odds(minimum_distance, max(minimum_distance, min(chosen_train)))
                distance = int(rng.choice(options))
            for _ in range(100):
                pair = _sample_pair(rng, depth, span, orientation, distance, palette)
                fp = _fingerprint(pair["input"])
                if fp not in seen and not _matches_source(pair["input"]):
                    seen.add(fp)
                    puzzle[section].append(pair)
                    break
            else:
                raise ValueError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _matching_conventions(pair):
    """Fit box geometry to visible examples without generation metadata."""
    info = _info(pair["input"])
    cross_size = (pair["input"].shape[0] if info["orientation"] == "horizontal"
                  else pair["input"].shape[1])
    matches = set()
    for depth in range(4, min(12, info["separation"] - 3) + 1, 2):
        for span in range(3, min(11, cross_size) + 1, 2):
            try:
                candidate = _construct_output(pair["input"], depth, span)
            except AssertionError:
                continue
            if np.array_equal(candidate, pair["output"]):
                matches.add((depth, span))
    return matches


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    train_conventions = None
    train_distances = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= n <= 30 for n in grid.shape), "invalid dimensions")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), "invalid ARC color")
            grid, output = pair["input"], pair["output"]
            _require(grid.shape == output.shape and not np.array_equal(grid, output),
                     "output must change content and preserve shape")
            _require(not _matches_source(grid), "source pair copied")
            input_fp = _fingerprint(grid)
            pair_fp = (input_fp, _fingerprint(output))
            _require(input_fp not in seen_inputs and pair_fp not in seen_pairs,
                     "duplicate generated pair")
            seen_inputs.add(input_fp)
            seen_pairs.add(pair_fp)
            info = _info(grid)
            matches = _matching_conventions(pair)
            _require(len(matches) == 1, "output must identify one valid box rule")
            if section == "train":
                train_distances.add(info["separation"])
                train_conventions = matches if train_conventions is None else train_conventions & matches
            else:
                _require(matches == train_conventions,
                         "test output disagrees with geometry learned from training")
    _require(len(train_conventions) == 1, "training must reveal one shared geometry")
    _require(len(train_distances) >= 2, "training needs varied marker spacing")


def _geometry_seen_in_output(pair):
    """Measure the visible box away from the marker axis, not from metadata."""
    grid, output = pair["input"], pair["output"]
    info = _info(grid)
    horizontal = info["orientation"] == "horizontal"
    axis_line = info["markers"][0][0 if horizontal else 1]
    non_background = np.argwhere(output != info["background"])
    off_axis = [(int(r), int(c)) for r, c in non_background
                if (r != axis_line if horizontal else c != axis_line)]
    _require(bool(off_axis), "box must extend away from the marker axis")
    axial = [c if horizontal else r for r, c in off_axis]
    transverse = [r if horizontal else c for r, c in off_axis]
    return max(axial) - min(axial) + 1, max(transverse) - min(transverse) + 1


def _pixelwise_prediction(grid, depth, span):
    """A second, cellwise formulation used only to test ground truth."""
    info = _info(grid)
    first, second = info["markers"]
    horizontal = info["orientation"] == "horizontal"
    axis, cross = (1, 0) if horizontal else (0, 1)
    start = (first[axis] + second[axis] - depth + 1) // 2
    low = first[cross] - span // 2
    result = grid.copy()
    for along in range(first[axis], second[axis] + 1):
        for across in range(low, low + span):
            if start <= along < start + depth:
                if across not in (low, low + span - 1) and along not in (start, start + depth - 1):
                    continue
                color = first[2] if along < start + depth // 2 else second[2]
            elif across == first[cross]:
                color = first[2] if along < start else second[2]
            else:
                continue
            if horizontal:
                result[across, along] = color
            else:
                result[along, across] = color
    return result


def self_test() -> None:
    first = generate_puzzle(314159)
    second = generate_puzzle(314159)
    _require(all(np.array_equal(a[k], b[k]) for section in ("train", "test")
                 for a, b in zip(first[section], second[section]) for k in ("input", "output")),
             "same seed must reproduce complete puzzle")
    cases = [(2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
             (2, None), (10, None), (7, 3)]
    for index, (train, test) in enumerate(cases):
        puzzle = generate_puzzle(4000 + index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "train count not honored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "test count not honored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 3), (3, 4)):
        try:
            generate_puzzle(6000, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal count was accepted")
    corpus = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in corpus:
        inferred = {_geometry_seen_in_output(pair) for pair in puzzle["train"]}
        _require(len(inferred) == 1, "training outputs disagree on box geometry")
        geometry = inferred.pop()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                _require(np.array_equal(_pixelwise_prediction(pair["input"], *geometry),
                                        pair["output"]),
                         "cellwise prediction disagrees with ground truth")
    _require(len({len(p["train"]) for p in corpus}) >= 4, "train count collapse")
    _require(len({len(p["test"]) for p in corpus}) >= 3, "test count collapse")
    _require(len({tuple(_fingerprint(q["input"]) for part in ("train", "test")
                        for q in p[part]) for p in corpus}) == 42, "seed diversity collapse")
    geometries, shapes, layouts, backgrounds, palettes = set(), set(), set(), set(), set()
    orientation_regimes, distance_regimes = set(), set()
    edge_test = short_test = horizontal_long = cross_aspect = 0
    square = tight = varying_backgrounds = reused_color = 0
    same_orientation_puzzles = vertical_to_horizontal = 0
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        train_infos = [_info(pair["input"]) for pair in puzzle["train"]]
        test_infos = [_info(pair["input"]) for pair in puzzle["test"]]
        orientation_regimes.add((frozenset(x["orientation"] for x in train_infos),
                                 frozenset(x["orientation"] for x in test_infos)))
        if (len({x["orientation"] for x in train_infos}) == 1 and
                {x["orientation"] for x in train_infos} ==
                {x["orientation"] for x in test_infos}):
            same_orientation_puzzles += 1
        if ({x["orientation"] for x in train_infos} == {"vertical"} and
                any(x["orientation"] == "horizontal" for x in test_infos)):
            vertical_to_horizontal += 1
        train_distances = {x["separation"] for x in train_infos}
        all_infos = train_infos + test_infos
        if len({x["background"] for x in all_infos}) > 1:
            varying_backgrounds += 1
        train_colors = {marker[2] for x in train_infos for marker in x["markers"]}
        if any(marker[2] in train_colors for x in test_infos for marker in x["markers"]):
            reused_color += 1
        for part in ("train", "test"):
            for pair in puzzle[part]:
                grid = pair["input"]
                info = _info(grid)
                geometry = next(iter(_matching_conventions(pair)))
                geometries.add(geometry)
                shapes.add(grid.shape)
                backgrounds.add(info["background"])
                palettes.add((info["background"], info["markers"][0][2],
                              info["markers"][1][2]))
                layouts.add((part, grid.shape, info["orientation"],
                             tuple(point[:2] for point in info["markers"])))
                h, w = grid.shape
                if h == w:
                    square += 1
                if (h if info["orientation"] == "horizontal" else w) == geometry[1]:
                    tight += 1
                if (info["orientation"] == "horizontal" and h > w) or (
                    info["orientation"] == "vertical" and w > h):
                    cross_aspect += 1
                if part == "test":
                    distance_regimes.add("long" if info["separation"] > max(train_distances)
                                         else "familiar" if info["separation"] in train_distances
                                         else "short")
                    axis = 1 if info["orientation"] == "horizontal" else 0
                    if info["markers"][0][axis] == 0 or info["markers"][1][axis] == grid.shape[axis] - 1:
                        edge_test += 1
                    if info["separation"] == geometry[0] + 3:
                        short_test += 1
                    if info["orientation"] == "horizontal" and info["separation"] > max(train_distances):
                        horizontal_long += 1
    _require(len(geometries) >= 8 and (4, 5) in geometries, "geometry diversity collapse")
    _require(len(shapes) >= 60 and len(layouts) >= 150, "layout diversity collapse")
    _require(len(backgrounds) >= 8 and len(palettes) >= 60, "palette diversity collapse")
    _require(len(orientation_regimes) >= 5, "orientation regime collapse")
    _require(same_orientation_puzzles > 0 and vertical_to_horizontal > 0,
             "same-axis or vertical-to-horizontal transfer missing")
    _require(distance_regimes == {"long", "familiar", "short"}, "distance regime collapse")
    _require(edge_test > 0 and short_test > 0 and horizontal_long > 0 and cross_aspect > 0,
             "required edge, short-arm, horizontal-long or crossed-aspect case absent")
    _require(square > 0 and tight > 0 and varying_backgrounds >= 20 and reused_color >= 20,
             "square, tight or within-puzzle palette diversity absent")
