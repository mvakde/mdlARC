"""Generate puzzles about a uniquely aligned marker pair and its branches."""

import numpy as np


_SOURCE_SIGNATURES = {
    ((14, 10), ((7, 4), (9, 2), (9, 7), (11, 5))),
    ((16, 12), ((2, 5), (6, 2), (7, 10), (10, 8), (12, 5))),
    ((14, 15), ((1, 4), (3, 6), (5, 0), (6, 8), (8, 1), (10, 2), (12, 4))),
    ((19, 13), ((3, 8), (6, 4), (6, 12), (9, 10), (11, 7))),
    ((14, 17), ((3, 6), (6, 4), (8, 1), (8, 12), (10, 10), (12, 8))),
}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
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


def _grid_key(grid):
    copy = np.ascontiguousarray(grid, dtype=np.int16)
    return (tuple(copy.shape), copy.tobytes())


def _source_input_keys():
    keys = set()
    for shape, points in _SOURCE_SIGNATURES:
        grid = np.zeros(shape, dtype=np.int8)
        for row, column in points:
            grid[row, column] = 2
        keys.add(_grid_key(grid))
    return frozenset(keys)


_SOURCE_INPUT_KEYS = _source_input_keys()


def _branch_count(rng, regime, first_train=False):
    if first_train:
        return int(rng.integers(2, 7))
    probabilities = (
        (0.05, 0.14, 0.17, 0.22, 0.21, 0.14, 0.07)
        if regime == "source_train" else
        (0.05, 0.17, 0.17, 0.20, 0.19, 0.14, 0.08)
    )
    return int(rng.choice(np.arange(1, 8), p=probabilities))


def _span(rng, axis_size, branch_count, mode):
    ranges = {"compact": (0.30, 0.50), "middle": (0.50, 0.78),
              "long": (0.78, 1.0)}
    low, high = ranges[mode]
    proportion = float(rng.uniform(low, high))
    return min(axis_size - 1, max(branch_count + 1,
                                  int(round((axis_size - 1) * proportion))))


def _pick_cross_positions(rng, trunk, cross_size, negative_count, positive_count):
    positions = []
    for count, side in ((negative_count, -1), (positive_count, 1)):
        if count == 0:
            continue
        available = list(range(0, trunk)) if side < 0 else list(range(trunk + 1, cross_size))
        chosen = []
        if int(rng.integers(0, 4)) == 0:
            adjacent = trunk + side
            chosen.append(adjacent)
            available.remove(adjacent)
        if count > len(chosen):
            chosen.extend(int(x) for x in rng.choice(available, count - len(chosen),
                                                      replace=False))
        positions.extend(chosen)
    return positions


def _make_pair(rng, regime, orientation, branch_count, span_mode, colors,
               min_width=None, max_width=None, landscape=False, balanced=False):
    # Work in trunk-axis (t), perpendicular-axis (u) coordinates. The
    # final transpose is solely an orientation choice.
    if orientation == "horizontal":
        axis_size = int(rng.integers(max(10, branch_count + 3, min_width or 0),
                                     min(30, max_width or 30) + 1))
        cross_size = int(rng.integers(max(9, branch_count + 2),
                                      min(30, axis_size - 1) + 1 if landscape else 31))
    else:
        axis_size = int(rng.integers(max(10, branch_count + 3), 31))
        cross_size = int(rng.integers(max(9, branch_count + 2, min_width or 0),
                                      min(30, max_width or 30) + 1))
    length = _span(rng, axis_size, branch_count, span_mode)
    last_start = axis_size - length - 1
    if int(rng.integers(0, 3)) == 0:
        start = 0 if int(rng.integers(0, 2)) == 0 else last_start
    else:
        start = int(rng.integers(0, last_start + 1))
    end = start + length

    if balanced:
        negative_count = branch_count // 2
    elif branch_count == 1 or float(rng.random()) < 0.41:
        negative_count = branch_count if int(rng.integers(0, 2)) == 0 else 0
    else:
        negative_count = int(rng.integers(1, branch_count))
    positive_count = branch_count - negative_count
    trunk_u = int(rng.integers(negative_count,
                               cross_size - positive_count))
    branch_u = _pick_cross_positions(rng, trunk_u, cross_size,
                                     negative_count, positive_count)

    branch_t = [int(x) for x in rng.choice(np.arange(start + 1, end),
                                           branch_count, replace=False)]
    if branch_count >= 2 and float(rng.random()) < 0.55:
        anchors = (start + 1, end - 1) if length > 2 else (start + 1,)
        anchor = anchors[int(rng.integers(0, len(anchors)))]
        if anchor in branch_t:
            at = branch_t.index(anchor)
            branch_t[0], branch_t[at] = branch_t[at], branch_t[0]
        else:
            branch_t[0] = anchor

    background, marker, connector = colors
    input_axis = np.full((axis_size, cross_size), background, dtype=np.int8)
    input_axis[start, trunk_u] = marker
    input_axis[end, trunk_u] = marker
    for t, u in zip(branch_t, branch_u):
        input_axis[t, u] = marker
    output_axis = input_axis.copy()
    output_axis[start + 1:end, trunk_u] = connector
    for t, u in zip(branch_t, branch_u):
        output_axis[t, min(u, trunk_u) + 1:max(u, trunk_u)] = connector
    if orientation == "horizontal":
        return input_axis.T.copy(), output_axis.T.copy()
    return input_axis, output_axis


def _pair_colors(rng, connector, base, mode, index):
    if mode == "fixed":
        return (*base, connector)
    if mode == "swap" or index == 1:
        roles = base if index % 2 == 0 else base[::-1]
        return (*roles, connector)
    if index == 0:
        return (*base, connector)
    available = [color for color in range(10) if color != connector]
    roles = [int(x) for x in rng.choice(available, 2, replace=False)]
    return (roles[0], roles[1], connector)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)

    # The connector is learned from all training outputs. Input roles are
    # visible from each input, and may be recolored or exchanged by pair.
    connector = int(rng.integers(0, 10))
    base = tuple(int(x) for x in rng.choice(
        [c for c in range(10) if c != connector], 2, replace=False))
    color_mode = str(rng.choice(("fixed", "swap", "varied"),
                                p=(0.22, 0.38, 0.40)))
    familiar_layout = float(rng.random()) < 0.24

    first_orientation = ("horizontal" if familiar_layout else
                         str(rng.choice(("horizontal", "vertical"))))
    if familiar_layout or float(rng.random()) < 0.72:
        second_orientation = ("vertical" if first_orientation == "horizontal"
                              else "horizontal")
    else:
        second_orientation = first_orientation
    train_orientations = [first_orientation, second_orientation]
    train_orientations.extend(str(rng.choice(("horizontal", "vertical")))
                              for _ in range(train_count - 2))
    known_orientations = tuple(sorted(set(train_orientations)))
    test_orientations = []
    if len(known_orientations) == 2 and test_count >= 2 and float(rng.random()) < 0.6:
        test_orientations.extend(known_orientations)
    test_orientations.extend(str(rng.choice(known_orientations))
                             for _ in range(test_count - len(test_orientations)))
    rng.shuffle(test_orientations)
    if familiar_layout:
        test_orientations[0] = "horizontal"

    puzzle = {"train": [], "test": []}
    used = set()
    index = 0
    widest_train = 0
    for split, orientations in (("train", train_orientations),
                                ("test", test_orientations)):
        regime = "source_train" if split == "train" else "target_test"
        for pair_index, orientation in enumerate(orientations):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            colors = _pair_colors(rng, connector, base, color_mode, index)
            index += 1
            familiar_test = familiar_layout and split == "test" and pair_index == 0
            for _ in range(40):
                branches = (4 if familiar_test else
                            _branch_count(pair_rng, regime,
                                          first_train=(split == "train" and index == 1)))
                mode = (str(pair_rng.choice(("middle", "long"))) if familiar_test else
                        str(pair_rng.choice(("compact", "middle", "long"),
                                            p=((0.26, 0.43, 0.31) if split == "train"
                                               else (0.32, 0.24, 0.44)))))
                input_grid, output_grid = _make_pair(
                    pair_rng, regime, orientation, branches, mode, colors,
                    min_width=(widest_train + 2 if familiar_test else None),
                    max_width=(22 if familiar_layout and split == "train" else None),
                    landscape=familiar_test, balanced=familiar_test)
                key = _grid_key(input_grid)
                if key in used:
                    continue
                if key in _SOURCE_INPUT_KEYS:
                    continue
                used.add(key)
                puzzle[split].append({"input": input_grid, "output": output_grid})
                if split == "train":
                    widest_train = max(widest_train, input_grid.shape[1])
                break
            else:
                raise RuntimeError("unique pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid, label):
    _require(isinstance(grid, np.ndarray), f"{label} must be an array")
    _require(grid.ndim == 2, f"{label} must be 2D")
    _require(all(1 <= n <= 30 for n in grid.shape), f"{label} has invalid shape")
    _require(np.issubdtype(grid.dtype, np.integer), f"{label} has noninteger cells")
    _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} has invalid color")


def _input_geometry(input_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    _require(len(values) == 2, "input must have exactly two colors")
    bg_index = int(np.argmax(counts))
    _require(int(counts[bg_index]) > int(counts[1 - bg_index]),
             "background is not uniquely dominant")
    background = int(values[bg_index])
    marker = int(values[1 - bg_index])
    points = [tuple(int(v) for v in xy) for xy in np.argwhere(input_grid == marker)]
    _require(3 <= len(points) <= 9, "marker count is outside the family")
    aligned = [(a, b) for i, a in enumerate(points) for b in points[i + 1:]
               if a[0] == b[0] or a[1] == b[1]]
    _require(len(aligned) == 1, "endpoints are not the unique aligned pair")
    a, b = aligned[0]
    horizontal = a[0] == b[0]
    if horizontal:
        axis_size = input_grid.shape[1]
        start, end = sorted((a[1], b[1]))
        trunk_u = a[0]
        branches = [(col, row) for row, col in points if (row, col) not in (a, b)]
    else:
        axis_size = input_grid.shape[0]
        start, end = sorted((a[0], b[0]))
        trunk_u = a[1]
        branches = [(row, col) for row, col in points if (row, col) not in (a, b)]
    _require(end - start - 1 >= len(branches), "too few interior projections")
    _require(all(start < t < end and u != trunk_u for t, u in branches),
             "branch misses strict trunk interior")
    side_counts = (sum(u < trunk_u for _, u in branches),
                   sum(u > trunk_u for _, u in branches))
    return {"background": background, "marker": marker,
            "orientation": "horizontal" if horizontal else "vertical",
            "branches": branches, "branch_count": len(branches),
            "side_counts": side_counts, "start": start, "end": end,
            "trunk_u": trunk_u, "axis_size": axis_size,
            "shape": tuple(input_grid.shape),
            "edge": start == 0 or end == axis_size - 1,
            "span_ratio": (end - start) / (axis_size - 1),
            "adjacent": any(abs(u - trunk_u) == 1 for _, u in branches),
            "near_endpoint": any(t in (start + 1, end - 1) for t, _ in branches),
            "near_middle": any(abs(t - (start + end) / 2) <= 1 for t, _ in branches)}


def _reconstruct(input_grid, connector, geometry):
    result = input_grid.copy()
    # A transpose gives a single, orientation-independent implementation
    # that operates in the same axis coordinates as _input_geometry.
    axis = result.T if geometry["orientation"] == "horizontal" else result
    start, end = geometry["start"], geometry["end"]
    trunk_u = geometry["trunk_u"]
    axis[start + 1:end, trunk_u] = connector
    for t, u in geometry["branches"]:
        axis[t, min(u, trunk_u) + 1:max(u, trunk_u)] = connector
    return result


def _analyze_pair(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape, "input/output shapes differ")
    geometry = _input_geometry(input_grid)
    input_colors = {geometry["background"], geometry["marker"]}
    output_colors = {int(c) for c in np.unique(output_grid)}
    new = output_colors - input_colors
    _require(len(new) == 1 and len(output_colors) == 3,
             "output must introduce one distinct connector color")
    connector = next(iter(new))
    _require(np.array_equal(output_grid,
                            _reconstruct(input_grid, connector, geometry)),
             "output disagrees with unique-pair construction")
    _require(not np.array_equal(input_grid, output_grid), "unchanged pair")
    geometry["connector"] = connector
    geometry["colors"] = (geometry["background"], geometry["marker"], connector)
    return geometry


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "invalid pair counts")
    infos = {"train": [], "test": []}
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            input_grid, output_grid = pair["input"], pair["output"]
            _validate_grid(input_grid, f"{split}[{index}].input")
            _validate_grid(output_grid, f"{split}[{index}].output")
            input_key, output_key = _grid_key(input_grid), _grid_key(output_grid)
            _require(input_key not in inputs, "duplicate input")
            _require((input_key, output_key) not in pairs, "duplicate pair")
            inputs.add(input_key)
            pairs.add((input_key, output_key))
            info = _analyze_pair(input_grid, output_grid)
            _require(input_key not in _SOURCE_INPUT_KEYS, "exact source pair")
            infos[split].append(info)
    connectors = {info["connector"] for split in infos for info in infos[split]}
    _require(len(connectors) == 1, "connector convention varies within puzzle")
    trained_orientations = {info["orientation"] for info in infos["train"]}
    _require(all(info["orientation"] in trained_orientations
                 for info in infos["test"]),
             "test orientation is not demonstrated in training")
    _require(any(info["branch_count"] >= 2 for info in infos["train"]),
             "training lacks a multiple-branch example")


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[key], b[key])
                   for a, b in zip(first[split], second[split])
                   for key in ("input", "output"))
               for split in ("train", "test"))


def _predict_from_training(train, test_input):
    learned = {_analyze_pair(pair["input"], pair["output"])["connector"]
               for pair in train}
    _require(len(learned) == 1, "training does not establish a connector")
    return _reconstruct(test_input, next(iter(learned)),
                        _input_geometry(test_input))


def self_test() -> None:
    source_palette_pairs = []
    for index, orientation in enumerate(("horizontal", "vertical", "horizontal")):
        source_palette_pairs.append(_make_pair(
            np.random.default_rng(900 + index),
            "source_train" if index < 2 else "target_test",
            orientation, 4, "long", (0, 2, 3), min_width=23))
    source_palette_puzzle = {
        "train": [{"input": a, "output": b} for a, b in source_palette_pairs[:2]],
        "test": [{"input": a, "output": b} for a, b in source_palette_pairs[2:]],
    }
    validate_generated_puzzle(source_palette_puzzle)
    source_input = np.zeros((14, 10), dtype=np.int8)
    for row, column in ((7, 4), (9, 2), (9, 7), (11, 5)):
        source_input[row, column] = 2
    source_output = _reconstruct(source_input, 3, _input_geometry(source_input))
    source_palette_puzzle["test"][0] = {"input": source_input,
                                         "output": source_output}
    try:
        validate_generated_puzzle(source_palette_puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError("exact source pair accepted")

    for seed, train, test in ((501, 2, 1), (502, 10, 1), (503, 10, 4),
                              (504, None, 1), (505, None, 4),
                              (506, 2, None), (507, 10, None),
                              (508, 6, 3)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "test count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(600, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_puzzles_equal(generate_puzzle(777, 5, 2),
                            generate_puzzle(777, 5, 2)), "nondeterminism")
    counts = {"train": set(), "test": set()}
    shapes, structural, palettes = set(), set(), set()
    train_branches, test_branches = set(), set()
    test_orientations, test_sides = set(), set()
    test_aspects, test_width_relation = set(), set()
    span_bins, edge_count, adjacent_count = set(), 0, 0
    pair_recolors, swaps, mixed_test = 0, 0, 0
    near_endpoint, near_middle, varied_arms = 0, 0, 0
    varied_training_sides, varied_spans = 0, 0
    familiar_tests, combined_tests = 0, 0
    orientation_aspects = set()
    puzzle_keys = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts["train"].add(len(puzzle["train"]))
        counts["test"].add(len(puzzle["test"]))
        infos = {split: [_analyze_pair(p["input"], p["output"])
                         for p in puzzle[split]] for split in ("train", "test")}
        _require(all(np.array_equal(_predict_from_training(puzzle["train"],
                                                        p["input"]), p["output"])
                     for p in puzzle["test"]), "test not recoverable from training")
        puzzle_keys.add(tuple(_grid_key(p["input"]) for split in puzzle
                              for p in puzzle[split]))
        palettes_here = [x["colors"] for split in infos for x in infos[split]]
        palettes.update(palettes_here)
        pair_recolors += len(set(palettes_here)) > 1
        swaps += any(a[0] == b[1] and a[1] == b[0]
                     for a in palettes_here for b in palettes_here if a != b)
        mixed_test += len({x["orientation"] for x in infos["test"]}) > 1
        varied_training_sides += len({min(x["side_counts"]) == 0
                                      for x in infos["train"]}) > 1
        varied_spans += (any(x["span_ratio"] < 0.5 for split in infos
                             for x in infos[split]) and
                         any(x["span_ratio"] >= 0.7 for split in infos
                             for x in infos[split]))
        widest_train = max(x["shape"][1] for x in infos["train"])
        for split in ("train", "test"):
            for x in infos[split]:
                shapes.add(x["shape"])
                structural.add((x["orientation"], x["shape"], x["start"],
                                x["end"], x["trunk_u"], tuple(x["branches"])))
                span_bins.add("compact" if x["span_ratio"] < 0.5 else
                              "long" if x["span_ratio"] >= 0.7 else "middle")
                edge_count += x["edge"]
                adjacent_count += x["adjacent"]
                near_endpoint += x["near_endpoint"]
                near_middle += x["near_middle"]
                varied_arms += len({abs(u - x["trunk_u"])
                                    for _, u in x["branches"]}) >= 2
                if split == "train":
                    train_branches.add(x["branch_count"])
                else:
                    test_branches.add(x["branch_count"])
                    test_orientations.add(x["orientation"])
                    test_sides.add("one" if min(x["side_counts"]) == 0 else "both")
                    rows, cols = x["shape"]
                    aspect = ("portrait" if rows > cols else
                              "landscape" if cols > rows else "square")
                    test_aspects.add(aspect)
                    orientation_aspects.add((x["orientation"], aspect))
                    test_width_relation.add(cols <= widest_train)
                    familiar_tests += (x["orientation"] == "horizontal" and
                                       aspect == "landscape" and
                                       x["branch_count"] == 4 and
                                       x["side_counts"] == (2, 2) and
                                       cols > widest_train)
                    combined_tests += (x["orientation"] == "vertical" and
                                       aspect == "portrait" and
                                       min(x["side_counts"]) == 0 and
                                       x["branch_count"] <= 3 and
                                       cols <= widest_train)
    _require(len(counts["train"]) >= 4 and len(counts["test"]) >= 3,
             "automatic counts lack diversity")
    _require(len(puzzle_keys) == 42 and len(shapes) >= 75
             and len(structural) >= 250 and len(palettes) >= 35,
             "structural or palette diversity is too narrow")
    _require(5 in train_branches and {2, 3, 5}.issubset(test_branches),
             "branch densities lack coverage")
    _require(test_orientations == {"horizontal", "vertical"}
             and test_sides == {"one", "both"}
             and {"portrait", "landscape"}.issubset(test_aspects)
             and test_width_relation == {False, True},
             "test layouts lack review regimes")
    _require(span_bins == {"compact", "middle", "long"}
             and edge_count >= 10 and adjacent_count >= 10,
             "trunk and arm diversity is too narrow")
    _require(pair_recolors >= 15 and swaps >= 10 and mixed_test >= 3,
             "within-puzzle color or orientation variation is too narrow")
    _require(near_endpoint >= 30 and near_middle >= 30 and varied_arms >= 50
             and varied_training_sides >= 10 and varied_spans >= 10,
             "branch placement or within-puzzle geometry is too narrow")
    _require(familiar_tests >= 2 and combined_tests >= 1 and
             {("horizontal", "portrait"), ("horizontal", "landscape"),
              ("vertical", "portrait"), ("vertical", "landscape")}.issubset(
                  orientation_aspects),
             "familiar and combined test regimes lack coverage")
