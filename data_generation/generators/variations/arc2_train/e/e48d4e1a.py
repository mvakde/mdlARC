import numpy as np


_CORNERS = ("TL", "TR", "BL", "BR")
_AXES = ("vertical", "horizontal")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    tr = None if num_train is None else _count("num_train", num_train, 2, 10)
    te = None if num_test is None else _count("num_test", num_test, 1, 4)
    if tr is None and te is None:
        te = int(rng.integers(1, 5))
        tr = int(rng.integers(max(2, te + 1), 11))
    elif tr is None:
        tr = int(rng.integers(max(2, te + 1), 11))
    elif te is None:
        te = int(rng.integers(1, min(4, tr - 1) + 1))
    if tr <= te:
        raise ValueError("num_train must exceed num_test")
    return tr, te


def _corner_data(height, width, corner):
    if corner == "TL":
        return 0, 0, 1, 1
    if corner == "TR":
        return 0, width - 1, 1, -1
    if corner == "BL":
        return height - 1, 0, -1, 1
    if corner == "BR":
        return height - 1, width - 1, -1, -1
    raise ValueError("unknown corner")


def _marker_cells(height, width, corner, axis, length):
    r, c, dr, dc = _corner_data(height, width, corner)
    if axis == "vertical":
        return [(r + i * dr, c) for i in range(length)]
    if axis == "horizontal":
        return [(r, c + i * dc) for i in range(length)]
    raise ValueError("unknown marker axis")


def _valid_intersections(height, width, corner, axis, length):
    marker = _marker_cells(height, width, corner, axis, length)
    _, _, dr, dc = _corner_data(height, width, corner)
    return [
        (r, c)
        for r in range(height)
        for c in range(width)
        if 0 <= r + dr * length < height
        and 0 <= c + dc * length < width
        and all(mr != r and mc != c for mr, mc in marker)
    ]


def _build_pair(height, width, background, marker_color, structural_color,
                corner, marker_axis, marker_length, intersection_row,
                intersection_col):
    inp = np.full((height, width), background, dtype=np.int8)
    inp[intersection_row, :] = structural_color
    inp[:, intersection_col] = structural_color
    for r, c in _marker_cells(height, width, corner, marker_axis, marker_length):
        inp[r, c] = marker_color
    _, _, dr, dc = _corner_data(height, width, corner)
    out = np.full((height, width), background, dtype=np.int8)
    out[intersection_row + dr * marker_length, :] = structural_color
    out[:, intersection_col + dc * marker_length] = structural_color
    return inp, out


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(x) for x in grid.flat)


def _pair_key(inp, out):
    return _grid_key(inp), _grid_key(out)


def _source_pair_keys():
    descriptors = ((6, 3, 4, 6), (4, 3, 3, 3), (3, 1, 2, 4),
                   (2, 2, 6, 3), (8, 2, 3, 5))
    return {
        _pair_key(*_build_pair(10, 10, 0, 5, color, "TR", "vertical",
                              length, row, col))
        for color, length, row, col in descriptors
    }


def _shape(rng, length, source_like=False):
    if length <= 3 and (source_like or rng.random() < 0.13):
        return 10, 10
    minimum = max(6, 2 * length + 2)
    def dimension():
        band = int(rng.integers(0, 3))
        if band == 0:
            lo, hi = minimum, max(minimum, 10)
        elif band == 1:
            lo, hi = max(minimum, 11), max(minimum, 20)
        else:
            lo, hi = max(minimum, 21), 30
        return int(rng.integers(lo, hi + 1))
    return dimension(), dimension()


def _palette(rng, base, mode=None, seen_colors=None, seen_cross=None):
    if mode is None:
        mode = int(rng.integers(0, 5))
    if mode == 0:
        return base
    if mode == 1:
        return base[2], base[0], base[1]
    if mode == 2:
        return base[1], base[2], base[0]
    if mode == 3:
        colors = [int(x) for x in rng.permutation(10)]
        if seen_cross is True:
            colors[2] = base[2]
            colors[:2] = [x for x in colors[:2] if x != base[2]]
            if len(colors[:2]) < 2:
                colors[:2] = [x for x in range(10) if x != base[2]][:2]
        elif seen_cross is False:
            unused = [x for x in range(10) if x not in seen_colors]
            colors[2] = unused[int(rng.integers(0, len(unused)))]
            colors[:2] = [x for x in colors if x != colors[2]][:2]
        return tuple(colors[:3])
    colors = [int(x) for x in rng.permutation(10)]
    return tuple(colors[:3])


def _make_pair(rng, spec, used_inputs, source_keys, edge_preference=False):
    h, w = spec["shape"]
    corner, axis, length = spec["corner"], spec["axis"], spec["length"]
    choices = _valid_intersections(h, w, corner, axis, length)
    if spec["position"] is not None:
        position = spec["position"]
        choices = [position] + [p for p in choices if p != position]
    else:
        if edge_preference:
            _, _, dr, dc = _corner_data(h, w, corner)
            edge = [p for p in choices if p[0] + dr * length in (0, h - 1)
                    or p[1] + dc * length in (0, w - 1)]
            if edge:
                choices = edge
        choices = [choices[int(i)] for i in rng.permutation(len(choices))]
    _require(bool(choices), "no legal intersection for sampled shape")
    for row, col in choices:
        inp, out = _build_pair(h, w, *spec["palette"], corner, axis, length,
                               row, col)
        key = _grid_key(inp)
        if key not in used_inputs and _pair_key(inp, out) not in source_keys:
            used_inputs.add(key)
            return {"input": inp, "output": out}, (row, col)
    raise AssertionError("no unused legal input")


def _spec(shape, palette, corner, axis, length, position=None):
    return {"shape": shape, "palette": palette, "corner": corner,
            "axis": axis, "length": length, "position": position}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    source_like = bool(rng.random() < 0.24)
    base = tuple(int(x) for x in rng.permutation(10)[:3])
    primary = "TR" if source_like else _CORNERS[int(rng.integers(0, 4))]
    axis = "vertical" if source_like else _AXES[int(rng.integers(0, 2))]
    other_corners = [c for c in _CORNERS if c != primary]
    other_corners = [other_corners[int(i)] for i in rng.permutation(3)]
    secondary = other_corners[0]
    controlled_shape = (10, 10) if source_like else _shape(rng, 3)
    primary_lengths = (1, 2, 3) if train_count >= 3 else (1, 2)
    shared = set(_valid_intersections(*controlled_shape, primary, axis,
                                      max(primary_lengths)))
    if train_count >= 3 and not source_like:
        comparison_corner = secondary if train_count == 3 else primary
        shared &= set(_valid_intersections(*controlled_shape, comparison_corner,
                                           axis, 3 if train_count == 3 else 2))
    if train_count >= 4 and not source_like:
        shared &= set(_valid_intersections(*controlled_shape, secondary, axis, 2))
    if train_count >= 6 and not source_like:
        opposite_axis = _AXES[1 - _AXES.index(axis)]
        shared &= set(_valid_intersections(*controlled_shape, secondary,
                                           opposite_axis, 2))
    _require(bool(shared), "controlled geometry has no shared position")
    shared_position = sorted(shared)[int(rng.integers(0, len(shared)))]

    specs = [_spec(controlled_shape, base, primary, axis, 1, shared_position),
             _spec(controlled_shape, base, primary, axis, 2, shared_position)]
    if source_like:
        if train_count >= 3:
            specs.append(_spec((10, 10), base, primary, axis, 3,
                               shared_position))
        cross_pool = [base[2]] + [c for c in range(10) if c not in base][:3]
        while len(specs) < train_count:
            color = cross_pool[int(rng.integers(0, len(cross_pool)))]
            palette = (base[0], base[1], color)
            length = int(rng.integers(1, 4))
            specs.append(_spec((10, 10), palette, primary, axis, length))
    else:
        if train_count >= 3:
            specs.append(_spec(controlled_shape, base,
                               secondary if train_count == 3 else primary,
                               axis, 3, shared_position))
        if train_count >= 4:
            specs.append(_spec(controlled_shape, base, secondary, axis, 2,
                               shared_position))
        if train_count >= 5:
            specs.append(_spec(controlled_shape, base, secondary, axis, 2))
        if train_count >= 6:
            specs.append(_spec(controlled_shape, base, secondary,
                               _AXES[1 - _AXES.index(axis)], 2, shared_position))
        if train_count >= 7:
            specs.append(_spec(_shape(rng, 2), _palette(rng, base),
                               other_corners[1], _AXES[int(rng.integers(0, 2))], 2))
        if train_count >= 8 and rng.random() < 0.5:
            specs.append(_spec(_shape(rng, 3), _palette(rng, base),
                               other_corners[2], _AXES[int(rng.integers(0, 2))], 3))
        while len(specs) < train_count:
            length = int(rng.integers(1, 6))
            specs.append(_spec(_shape(rng, length), _palette(rng, base),
                               _CORNERS[int(rng.integers(0, 4))],
                               _AXES[int(rng.integers(0, 2))], length))

    source_keys = _source_pair_keys()
    used_inputs = set()
    train = []
    train_info = []
    for i, spec in enumerate(specs):
        pair, position = _make_pair(rng, spec, used_inputs, source_keys,
                                    edge_preference=i >= 4 and rng.random() < 0.25)
        train.append(pair)
        train_info.append((spec, position))

    train_corners = {s["corner"] for s, _ in train_info}
    train_cross_colors = {s["palette"][2] for s, _ in train_info}
    train_shapes = {s["shape"] for s, _ in train_info}
    test_lengths = ([1, 2] if train_count == 2 else
                    [1, 2, 3] if source_like else [1, 2, 3, 4, 5, 6])
    if test_count >= 3 and train_count >= 3 and not source_like:
        lengths = [1, 2, 3]
        if rng.random() < 0.6:
            lengths[int(rng.integers(0, 3))] = int(rng.integers(4, 7))
        while len(lengths) < test_count:
            lengths.append(int(rng.integers(4, 7)))
        lengths = [lengths[int(i)] for i in rng.permutation(len(lengths))]
    else:
        lengths = [test_lengths[int(i)] for i in
                   rng.choice(len(test_lengths), size=test_count,
                              replace=test_count > len(test_lengths))]
    tests = []
    for i, length in enumerate(lengths):
        if len(train_corners) >= 3 and len(train_corners) < 4 and rng.random() < 0.25:
            corner = next(c for c in _CORNERS if c not in train_corners)
        else:
            choices = sorted(train_corners)
            corner = choices[int(rng.integers(0, len(choices)))]
        if train_count < 6:
            test_axis = axis
        else:
            test_axis = _AXES[int(rng.integers(0, 2))]
        if source_like:
            shape = (10, 10)
        elif rng.random() < 0.35:
            compatible = [shape for shape in train_shapes
                          if min(shape) >= 2 * length + 2]
            shape = compatible[int(rng.integers(0, len(compatible)))] if compatible else _shape(rng, length)
        else:
            shape = _shape(rng, length)
        if test_count >= 2:
            seen = i == 0
        else:
            seen = bool(rng.integers(0, 2))
        if source_like:
            available = sorted(train_cross_colors) if seen else [
                c for c in range(10) if c not in train_cross_colors
                and c not in base[:2]]
            color = available[int(rng.integers(0, len(available)))]
            palette = (base[0], base[1], color)
        else:
            palette = _palette(rng, base, mode=3, seen_colors=train_cross_colors,
                               seen_cross=seen)
        spec = _spec(shape, palette, corner, test_axis, length)
        pair, _ = _make_pair(rng, spec, used_inputs, source_keys,
                              edge_preference=bool(rng.random() < 0.3))
        tests.append(pair)
    puzzle = {"train": train, "test": tests}
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(inp, out):
    _require(inp.shape == out.shape, "input/output shape changed")
    h, w = inp.shape
    colors = [int(c) for c in np.unique(inp)]
    _require(len(colors) == 3, "input must have three distinct roles")
    crosses = []
    for color in colors:
        mask = inp == color
        rows = np.flatnonzero(np.all(mask, axis=1))
        cols = np.flatnonzero(np.all(mask, axis=0))
        if len(rows) == len(cols) == 1:
            expected = np.zeros((h, w), dtype=bool)
            expected[int(rows[0]), :] = True
            expected[:, int(cols[0])] = True
            if np.array_equal(mask, expected):
                crosses.append((color, int(rows[0]), int(cols[0])))
    _require(len(crosses) == 1, "input must contain one complete cross")
    cross, row, col = crosses[0]
    output_colors = [int(c) for c in np.unique(out)]
    _require(len(output_colors) == 2 and cross in output_colors,
             "output must preserve cross color and one background")
    background = next(c for c in output_colors if c != cross)
    _require(background in colors, "background changed")
    marker = next(c for c in colors if c not in (cross, background))
    length = int(np.count_nonzero(inp == marker))
    _require(1 <= length <= 6, "marker length outside supported range")
    matches = []
    for corner in _CORNERS:
        for axis in _AXES:
            cells = _marker_cells(h, w, corner, axis, length)
            if all(0 <= r < h and 0 <= c < w for r, c in cells):
                mask = np.zeros((h, w), dtype=bool)
                for r, c in cells:
                    mask[r, c] = True
                if np.array_equal(inp == marker, mask):
                    matches.append((corner, axis))
    _require(bool(matches), "marker is not a contiguous corner run")
    corners = {c for c, _ in matches}
    _require(len(corners) == 1, "marker corner is ambiguous")
    corner = next(iter(corners))
    axis_options = {a for c, a in matches if c == corner}
    _, _, dr, dc = _corner_data(h, w, corner)
    new_row, new_col = row + dr * length, col + dc * length
    _require(0 <= new_row < h and 0 <= new_col < w,
             "displaced intersection is out of bounds")
    expected_input, expected_output = _build_pair(
        h, w, background, marker, cross, corner, sorted(axis_options)[0],
        length, row, col)
    _require(np.array_equal(inp, expected_input), "cross and marker overlap or noise exists")
    _require(np.array_equal(out, expected_output), "incorrect relocated cross")
    return {"shape": (h, w), "background": background, "marker": marker,
            "cross": cross, "corner": corner, "axes": axis_options,
            "length": length, "row": row, "col": col,
            "new_row": new_row, "new_col": new_col, "dr": dr, "dc": dc}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")),
             "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    used_inputs, used_pairs = set(), set()
    analyses = {"train": [], "test": []}
    source = _source_pair_keys()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be 2D NumPy arrays")
                _require(all(1 <= d <= 30 for d in grid.shape),
                         "grid dimensions out of ARC bounds")
                _require(np.issubdtype(grid.dtype, np.integer)
                         and int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid colors or dtype invalid")
            ik, pk = _grid_key(inp), _pair_key(inp, out)
            _require(ik not in used_inputs and pk not in used_pairs,
                     "duplicate input or pair")
            _require(pk not in source, "exact source pair copied")
            used_inputs.add(ik)
            used_pairs.add(pk)
            analyses[split].append(_analyze_pair(inp, out))
    train, tests = analyses["train"], analyses["test"]
    train_lengths = {a["length"] for a in train}
    _require({1, 2}.issubset(train_lengths), "training must demonstrate lengths 1 and 2")
    if nt >= 3:
        _require(3 in train_lengths, "training must demonstrate length 3")
    train_corners = {a["corner"] for a in train}
    row_signs = {a["dr"] for a in train}
    col_signs = {a["dc"] for a in train}
    for test in tests:
        _require(test["corner"] in train_corners or
                 (test["dr"] in row_signs and test["dc"] in col_signs),
                 "test corner's row and column signs not demonstrated")
        _require(test["length"] <= 3 or {1, 2, 3}.issubset(train_lengths),
                 "long test length lacks unary progression in training")
    _require(all(a["length"] <= 6 for a in train + tests),
             "unsupported marker length")


def _puzzles_equal(a, b):
    return all(len(a[k]) == len(b[k]) and all(
        np.array_equal(x["input"], y["input"]) and
        np.array_equal(x["output"], y["output"])
        for x, y in zip(a[k], b[k])) for k in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(73)
    _require(_puzzles_equal(a, generate_puzzle(73)), "automatic generation is not deterministic")
    explicit = generate_puzzle(314159, 7, 3)
    _require(_puzzles_equal(explicit, generate_puzzle(314159, 7, 3)),
             "explicit generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(2025, 6, 2),
                                generate_puzzle(2026, 6, 2)), "seeds collided")
    for i, (tr, te) in enumerate(((2, 1), (10, 1), (10, 4),
                                  (None, 1), (None, 4),
                                  (2, None), (10, None))):
        p = generate_puzzle(5000 + i, tr, te)
        validate_generated_puzzle(p)
        _require((tr is None or len(p["train"]) == tr) and
                 (te is None or len(p["test"]) == te), "requested counts ignored")
    for i, (tr, te) in enumerate(((1, None), (11, None), (None, 0),
                                  (None, 5), (2, 2), (3, 3),
                                  (3, 4), (2.5, 1), (True, 1))):
        try:
            generate_puzzle(6000 + i, tr, te)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    train_counts, test_counts, corners, axes, lengths, test_lengths, shapes = (
        set(), set(), set(), set(), set(), set(), set())
    small = large = changed_corner = changed_palette = new_shape = 0
    seen_cross = unseen_cross = edge = controlled_length = controlled_corner = 0
    two_edges = role_swaps = longer_tests = mixed_test_lengths = 0
    unseen_corner = marker_fixed_position_changed = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        train_counts.add(len(p["train"]))
        test_counts.add(len(p["test"]))
        ts = [_analyze_pair(x["input"], x["output"]) for x in p["train"]]
        es = [_analyze_pair(x["input"], x["output"]) for x in p["test"]]
        aa = ts + es
        corners.update(x["corner"] for x in aa)
        axes.update(y for x in aa if x["length"] > 1 for y in x["axes"])
        lengths.update(x["length"] for x in aa)
        test_lengths.update(x["length"] for x in es)
        shapes.update(x["shape"] for x in aa)
        small += any(min(x["shape"]) < 10 for x in aa)
        large += any(max(x["shape"]) > 20 for x in aa)
        changed_corner += len({x["corner"] for x in aa}) > 1
        changed_palette += len({(x["background"], x["marker"], x["cross"])
                                for x in aa}) > 1
        new_shape += any(x["shape"] not in {y["shape"] for y in ts} for x in es)
        seen_cross += any(x["cross"] in {y["cross"] for y in ts} for x in es)
        unseen_cross += any(x["cross"] not in {y["cross"] for y in ts} for x in es)
        edge += any(x["new_row"] in (0, x["shape"][0] - 1) or
                    x["new_col"] in (0, x["shape"][1] - 1) for x in aa)
        controlled_length += (ts[0]["shape"], ts[0]["row"], ts[0]["col"],
                              ts[0]["corner"]) == (ts[1]["shape"], ts[1]["row"],
                                                   ts[1]["col"], ts[1]["corner"])
        mixed_test_lengths += len({x["length"] for x in es}) > 1
        longer_tests += any(x["length"] > 3 for x in es)
        unseen_corner += any(x["corner"] not in {y["corner"] for y in ts}
                             for x in es)
        two_edges += any(
            {next(iter(x["axes"])) for x in ts if x["corner"] == corner
             and x["length"] > 1} == set(_AXES) for corner in _CORNERS)
        role_swaps += any(
            (x["cross"] in (y["background"], y["marker"]) or
             y["cross"] in (x["background"], x["marker"]))
            for i, x in enumerate(aa) for y in aa[i + 1:])
        marker_fixed_position_changed += any(
            x["shape"] == y["shape"] and x["corner"] == y["corner"]
            and x["length"] == y["length"] and x["axes"] == y["axes"]
            and (x["row"], x["col"]) != (y["row"], y["col"])
            for i, x in enumerate(ts) for y in ts[i + 1:])
        if len(ts) >= 4:
            controlled_corner += (ts[1]["shape"], ts[1]["row"], ts[1]["col"],
                                  ts[1]["length"]) == (ts[3]["shape"], ts[3]["row"],
                                                       ts[3]["col"], ts[3]["length"])
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic counts lack diversity")
    _require(corners == set(_CORNERS) and axes == set(_AXES)
             and lengths == set(range(1, 7)) and test_lengths == set(range(1, 7)),
             "corner, axis, or length coverage missing")
    _require(len(shapes) >= 30 and small >= 5 and large >= 5,
             "scale diversity missing")
    _require(changed_corner >= 15 and changed_palette >= 35
             and new_shape >= 10 and seen_cross >= 10 and unseen_cross >= 10,
             "within-puzzle variation missing")
    _require(edge >= 10 and controlled_length == 42 and controlled_corner >= 15,
             "boundary or controlled contrast missing")
    _require(two_edges >= 10 and role_swaps >= 10 and longer_tests >= 10
             and mixed_test_lengths >= 5 and unseen_corner >= 1
             and marker_fixed_position_changed >= 10,
             "marker-edge, palette, test, or position contrast missing")
    damaged = {"train": list(explicit["train"]), "test": list(explicit["test"])}
    damaged["test"][0] = dict(damaged["test"][0])
    damaged["test"][0]["output"] = damaged["test"][0]["output"].copy()
    damaged["test"][0]["output"][0, 0] = 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted invalid output")
