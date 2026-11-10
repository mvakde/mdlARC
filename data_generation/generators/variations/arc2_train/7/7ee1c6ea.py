import numpy as np


_SOURCE_PAIR_SIGNATURES = {
    (
        (
            "0086060808",
            "8555555550",
            "0508866058",
            "6566680658",
            "0566860658",
            "6588868058",
            "6568686858",
            "0560688858",
            "8555555556",
            "8880886066",
        ),
        (
            "0086060808",
            "8555555550",
            "0506688058",
            "6588860858",
            "0588680858",
            "6566686058",
            "6586868658",
            "0580866658",
            "8555555556",
            "8880886066",
        ),
    ),
    (
        (
            "9400490099",
            "4994990090",
            "0055555509",
            "9459099504",
            "4450040544",
            "9454909500",
            "0950400504",
            "0455555544",
            "9099409000",
            "9990949900",
        ),
        (
            "9400490099",
            "4994990090",
            "0055555509",
            "9454044504",
            "4450090544",
            "9459404500",
            "0950900504",
            "0455555544",
            "9099409000",
            "9990949900",
        ),
    ),
    (
        (
            "0033332020",
            "3555555553",
            "3532222052",
            "0503032252",
            "3520232253",
            "3533023353",
            "3533303252",
            "0530333053",
            "0555555553",
            "2033323230",
        ),
        (
            "0033332020",
            "3555555553",
            "3523333052",
            "0502023352",
            "3530323353",
            "3522032253",
            "3522202352",
            "0520222053",
            "0555555553",
            "2033323230",
        ),
    ),
    (
        (
            "7011700777",
            "1555555557",
            "1500101757",
            "0571701751",
            "7577017151",
            "7501707751",
            "1577111150",
            "0517777057",
            "0555555550",
            "0171070077",
        ),
        (
            "7011700777",
            "1555555557",
            "1500707157",
            "0517107151",
            "7511071751",
            "7507101151",
            "1511777750",
            "0571111057",
            "0555555550",
            "0171070077",
        ),
    ),
}


def _validate_optional_count(value, name, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _resolve_counts(rng, seed_value, num_train, num_test):
    _validate_optional_count(num_train, "num_train", 2, 10)
    _validate_optional_count(num_test, "num_test", 1, 4)

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        # This schedule guarantees count diversity for consecutive fixed seeds.
        num_test = 1 + seed_value % 4
        legal_train_counts = tuple(range(num_test + 1, 11))
        index = (seed_value // 4) % len(legal_train_counts)
        num_train = legal_train_counts[index]
        return num_train, num_test

    if num_train is None:
        low = max(2, num_test + 1)
        num_train = int(rng.integers(low, 11))
        return num_train, num_test

    maximum_test = min(4, num_train - 1)
    num_test = int(rng.integers(1, maximum_test + 1))
    return num_train, num_test


def _grid_signature(grid):
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_signature(input_grid, output_grid):
    return (_grid_signature(input_grid), _grid_signature(output_grid))


def _rows_signature(grid):
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _find_unique_frame(grid):
    candidates = []
    for raw_color in np.unique(grid):
        color = int(raw_color)
        positions = np.argwhere(grid == color)
        r0, c0 = positions.min(axis=0)
        r1, c1 = positions.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        outline = np.zeros(grid.shape, dtype=bool)
        outline[r0, c0:c1 + 1] = True
        outline[r1, c0:c1 + 1] = True
        outline[r0:r1 + 1, c0] = True
        outline[r0:r1 + 1, c1] = True
        if np.array_equal(grid == color, outline):
            candidates.append((color, (r0, r1, c0, c1), outline))
    assert len(candidates) == 1, "exactly one complete dedicated frame required"
    color, bbox, outline = candidates[0]
    return {"color": color, "bbox": bbox, "mask": outline,
            "shape": (bbox[1] - bbox[0] + 1, bbox[3] - bbox[2] + 1)}


def _interior_mask(shape, bbox):
    r0, r1, c0, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 + 1:r1, c0 + 1:c1] = True
    return mask


def _analyze_pair(input_grid, output_grid, neutral=None):
    assert input_grid.shape == output_grid.shape
    frame = _find_unique_frame(input_grid)
    other_frame = _find_unique_frame(output_grid)
    assert frame["color"] == other_frame["color"]
    assert frame["bbox"] == other_frame["bbox"]
    interior = _interior_mask(input_grid.shape, frame["bbox"])
    assert np.array_equal(input_grid[~interior], output_grid[~interior])
    source, target = input_grid[interior], output_grid[interior]
    assert np.any(source != target)
    if neutral is None:
        unchanged = {int(x) for x in np.unique(source[source == target])}
        assert len(unchanged) == 1
        neutral = next(iter(unchanged))
        assert np.count_nonzero(source == neutral) >= 2
    else:
        assert np.all(source[source == target] == neutral)
    payloads = tuple(sorted(int(x) for x in np.unique(source) if x != neutral))
    assert len(payloads) == 2
    first, second = payloads
    assert np.count_nonzero(source == first) >= 2
    assert np.count_nonzero(source == second) >= 2
    assert np.all(target[source == first] == second)
    assert np.all(target[source == second] == first)
    assert np.all(target[source == neutral] == neutral)
    assert frame["color"] not in (neutral, first, second)
    assert set(map(int, np.unique(input_grid))) <= {frame["color"], neutral, first, second}
    assert set(map(int, np.unique(output_grid))) <= {frame["color"], neutral, first, second}
    exterior = ~(frame["mask"] | interior)
    r0, r1, c0, c1 = frame["bbox"]
    return {
        "frame_color": frame["color"], "neutral_color": neutral,
        "payloads": payloads, "bbox": frame["bbox"],
        "frame_shape": frame["shape"],
        "margins": (r0, input_grid.shape[0] - r1 - 1,
                    c0, input_grid.shape[1] - c1 - 1),
        "exterior_payloads": sum(np.any(input_grid[exterior] == color)
                                 for color in payloads),
        "interior_neutral": int(np.count_nonzero(source == neutral)),
        "payload_counts": tuple(int(np.count_nonzero(source == color))
                                for color in payloads),
    }


def _infer_neutral(train_pairs):
    neutrals = set()
    for pair in train_pairs:
        frame = _find_unique_frame(pair["input"])
        interior = _interior_mask(pair["input"].shape, frame["bbox"])
        unchanged = pair["input"][interior][
            pair["input"][interior] == pair["output"][interior]]
        assert len(unchanged) >= 2
        colors = {int(x) for x in np.unique(unchanged)}
        assert len(colors) == 1
        neutrals.update(colors)
    assert len(neutrals) == 1
    return next(iter(neutrals))


def _predict_output(train_pairs, test_input):
    neutral = _infer_neutral(train_pairs)
    frame = _find_unique_frame(test_input)
    interior = _interior_mask(test_input.shape, frame["bbox"])
    colors = sorted(int(x) for x in np.unique(test_input[interior])
                    if x != neutral)
    assert len(colors) == 2
    result = test_input.copy()
    result[interior & (test_input == colors[0])] = colors[1]
    result[interior & (test_input == colors[1])] = colors[0]
    return result


def _train_payload_schedule(rng, universe, count):
    palette_size = int(rng.integers(2, min(6, 2 * count) + 1))
    pool = [int(x) for x in rng.permutation(universe)[:palette_size]]
    unused = pool.copy()
    schedule = []
    for _ in range(count):
        if unused:
            first = unused.pop(0)
        else:
            first = int(rng.choice(pool))
        if unused:
            second = unused.pop(0)
        else:
            second = int(rng.choice([x for x in pool if x != first]))
        schedule.append(tuple(sorted((first, second))))
    rng.shuffle(schedule)
    return schedule


def _test_payload_schedule(rng, universe, train_pairs, count, classic):
    familiar = sorted({color for pair in train_pairs for color in pair})
    new = sorted(set(universe) - set(familiar))
    categories = ["familiar", "mixed", "new"]
    sampled = [str(rng.choice(categories)) for _ in range(count)]
    if classic:
        sampled[0] = "new"
    if count > 1 and len(set(sampled)) == 1:
        sampled[1] = str(rng.choice([x for x in categories if x != sampled[0]]))
    result = []
    for index in range(count):
        category = sampled[index]
        if count == 1 and len(familiar) == 2 and category == "familiar":
            category = "mixed"
        if category == "familiar":
            pair = rng.choice(familiar, size=2, replace=False)
        elif category == "mixed":
            pair = (rng.choice(familiar), rng.choice(new))
        else:
            pair = rng.choice(new, size=2, replace=False)
        result.append(tuple(sorted(int(x) for x in pair)))
    return result


def _frame_shape(rng):
    roll = rng.random()
    if roll < 0.16:
        side = int(rng.choice((6, 8)))
        return (side, side)
    if roll < 0.42:
        short = int(rng.integers(5, 8))
        long = int(rng.integers(10, 18))
        return (short, long) if rng.random() < 0.5 else (long, short)
    return (int(rng.integers(5, 17)), int(rng.integers(5, 17)))


def _margins(rng, frame_shape, classic):
    fh, fw = frame_shape
    if (fh <= 8 and fw <= 8 and rng.random() < (0.55 if classic else 0.13)):
        top = int(rng.integers(0, 11 - fh))
        left = int(rng.integers(0, 11 - fw))
        return top, 10 - fh - top, left, 10 - fw - left
    choices = np.array((0, 1, 2, 3, 4, 5, 7, 9, 12, 15))
    probabilities = np.array((.18, .16, .15, .13, .11, .09,
                              .07, .05, .04, .02))
    result = []
    for extent in frame_shape:
        budget = 30 - extent
        for _ in range(40):
            a, b = rng.choice(choices, size=2, p=probabilities)
            if a + b <= budget:
                result.append((int(a), int(b)))
                break
        else:
            result.append((0, min(2, budget)))
    (top, bottom), (left, right) = result
    return top, bottom, left, right


def _interior_values(rng, shape, colors, mode, neutral_required):
    height, width = shape
    neutral, first, second = colors
    rr, cc = np.indices(shape)
    if mode == "noise":
        probabilities = rng.dirichlet((1.4, 1.2, 1.2))
        values = rng.choice(colors, size=shape, p=probabilities)
    elif mode == "clusters":
        centers = rng.integers((0, 0), shape, size=(4, 2))
        roles = rng.choice(colors, size=4, replace=True)
        roles[:3] = colors
        distance = np.stack([(rr - r) ** 2 + (cc - c) ** 2
                             for r, c in centers])
        values = roles[np.argmin(distance, axis=0)]
        noise = rng.random(shape) < .06
        values[noise] = rng.choice(colors, size=int(noise.sum()))
    elif mode == "bars":
        axis = int(rng.integers(0, 2))
        stripe = (rr if axis == 0 else cc) // int(rng.integers(1, 4))
        roles = rng.permutation(colors)
        values = roles[(stripe + int(rng.integers(0, 3))) % 3]
    elif mode == "checker":
        values = np.where((rr + cc + int(rng.integers(0, 2))) % 2,
                          first, second)
        values[rng.random(shape) < .14] = neutral
    elif mode == "mostly_neutral":
        values = rng.choice(colors, size=shape, p=(.78, .11, .11))
    elif mode == "unequal":
        if rng.random() < .5:
            first, second = second, first
        values = rng.choice((neutral, first, second), size=shape,
                            p=(.16, .74, .10))
    elif mode == "no_neutral":
        probability = rng.uniform(.25, .75)
        values = rng.choice((first, second), size=shape,
                            p=(probability, 1 - probability))
    else:
        raise ValueError("unknown interior layout")
    values = np.asarray(values, dtype=np.int8)
    flat = values.ravel()
    requirements = ((first, 2), (second, 2))
    if neutral_required:
        requirements += ((neutral, 2),)
    for color, minimum in requirements:
        missing = max(0, minimum - int(np.count_nonzero(flat == color)))
        if missing:
            eligible = np.flatnonzero(flat != color)
            selected = rng.choice(eligible, size=missing, replace=False)
            flat[selected] = color
    return values


def _fill_exterior(rng, grid, exterior, colors, mode, bbox):
    neutral, first, second = colors
    count = int(np.count_nonzero(exterior))
    if not count:
        return
    if mode == "none":
        grid[exterior] = neutral
    elif mode == "one":
        chosen = first if rng.random() < .5 else second
        grid[exterior] = rng.choice((neutral, chosen), size=count,
                                    p=(.68, .32))
    elif mode == "noise":
        weights = rng.dirichlet((1.3, 1.1, 1.1))
        grid[exterior] = rng.choice(colors, size=count, p=weights)
    elif mode == "clusters":
        grid[exterior] = neutral
        locations = np.argwhere(exterior)
        r0, r1, c0, c1 = bbox
        distance = np.maximum.reduce((
            np.maximum(r0 - locations[:, 0], 0),
            np.maximum(locations[:, 0] - r1, 0),
            np.maximum(c0 - locations[:, 1], 0),
            np.maximum(locations[:, 1] - c1, 0),
        ))
        near_frame = locations[distance <= 2]
        if len(near_frame) == 0:
            near_frame = locations
        for color in (first, second):
            center = near_frame[int(rng.integers(0, len(near_frame)))]
            radius = int(rng.integers(2, 5))
            nearby = (abs(locations[:, 0] - center[0]) <= radius) & (
                abs(locations[:, 1] - center[1]) <= radius)
            spots = locations[nearby]
            grid[spots[:, 0], spots[:, 1]] = color
    else:
        raise ValueError("unknown exterior layout")
    if mode in ("noise", "clusters") and count >= 2:
        locations = np.argwhere(exterior)
        for color in (first, second):
            if not np.any(grid[exterior] == color):
                spot = locations[int(rng.integers(0, count))]
                grid[spot[0], spot[1]] = color


def _generate_pair(rng, frame_shape, payloads, marker, neutral,
                   regime, classic, full_cue, forbidden_inputs):
    colors = (neutral, *payloads)
    for _ in range(80):
        top, bottom, left, right = _margins(rng, frame_shape, classic)
        fh, fw = frame_shape
        height, width = top + fh + bottom, left + fw + right
        bbox = (top, top + fh - 1, left, left + fw - 1)
        frame_mask = np.zeros((height, width), dtype=bool)
        r0, r1, c0, c1 = bbox
        frame_mask[r0, c0:c1 + 1] = True
        frame_mask[r1, c0:c1 + 1] = True
        frame_mask[r0:r1 + 1, c0] = True
        frame_mask[r0:r1 + 1, c1] = True
        interior = _interior_mask((height, width), bbox)
        exterior = ~(frame_mask | interior)
        grid = np.full((height, width), neutral, dtype=np.int8)
        if classic:
            inner_mode = "noise"
        else:
            modes = ("noise", "clusters", "bars", "checker",
                     "mostly_neutral", "unequal")
            inner_mode = str(rng.choice(modes))
            if regime == "target_test" and rng.random() < .13:
                inner_mode = "no_neutral"
        grid[r0 + 1:r1, c0 + 1:c1] = _interior_values(
            rng, (fh - 2, fw - 2), colors, inner_mode,
            regime == "source_train")
        if full_cue or classic:
            outer_mode = "noise" if rng.random() < .6 else "clusters"
        else:
            outer_mode = str(rng.choice(
                ("noise", "clusters", "one", "none"),
                p=(.32, .28, .22, .18)))
        _fill_exterior(rng, grid, exterior, colors, outer_mode, bbox)
        if full_cue and int(np.count_nonzero(exterior)) >= 2:
            for color in payloads:
                if not np.any(grid[exterior] == color):
                    positions = np.argwhere(exterior)
                    spot = positions[int(rng.integers(0, len(positions)))]
                    grid[spot[0], spot[1]] = color
        grid[frame_mask] = marker
        output = grid.copy()
        output[interior & (grid == payloads[0])] = payloads[1]
        output[interior & (grid == payloads[1])] = payloads[0]
        signature = (_rows_signature(grid), _rows_signature(output))
        if _grid_signature(grid) in forbidden_inputs or signature in _SOURCE_PAIR_SIGNATURES:
            continue
        try:
            info = _analyze_pair(grid, output, neutral)
        except AssertionError:
            continue
        if full_cue and info["exterior_payloads"] != 2:
            continue
        return grid, output
    raise RuntimeError("could not construct an unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, seed, num_train, num_test)
    permutation = [int(x) for x in rng.permutation(10)]
    neutral, base_marker = permutation[:2]
    universe = tuple(permutation[2:])
    classic = rng.random() < .18
    variable_markers = not classic and rng.random() < .5
    train_payloads = _train_payload_schedule(rng, universe, num_train)
    test_payloads = _test_payload_schedule(
        rng, universe, train_payloads, num_test, classic)
    if classic:
        train_shapes = [(8, 8), (6, 6)]
    else:
        train_shapes = [_frame_shape(rng)]
        for _ in range(50):
            candidate = _frame_shape(rng)
            if candidate != train_shapes[0]:
                train_shapes.append(candidate)
                break
        else:
            train_shapes.append((6, 8) if train_shapes[0] != (6, 8) else (8, 6))
    train_shapes.extend(_frame_shape(rng) for _ in range(2, num_train))
    train_markers = []
    for index, payload in enumerate(train_payloads):
        candidates = [x for x in (base_marker, *universe) if x not in payload]
        if not variable_markers or index == 0:
            marker = base_marker
        elif index == 1:
            marker = int(rng.choice([x for x in candidates if x != base_marker]))
        else:
            marker = int(rng.choice(candidates))
        train_markers.append(marker)
    puzzle = {"train": [], "test": []}
    forbidden_inputs = set()
    pair_seeds = rng.integers(0, 2**62, size=num_train + num_test,
                              dtype=np.int64)
    for index, (payload, shape, marker) in enumerate(zip(
            train_payloads, train_shapes, train_markers)):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        grid, output = _generate_pair(
            pair_rng, shape, payload, marker, neutral, "source_train",
            classic, index == 0, forbidden_inputs)
        forbidden_inputs.add(_grid_signature(grid))
        puzzle["train"].append({"input": grid, "output": output})
    train_marker_set = set(train_markers)
    for index, payload in enumerate(test_payloads):
        pair_rng = np.random.default_rng(int(pair_seeds[num_train + index]))
        if classic or rng.random() < .47:
            shape = train_shapes[int(rng.integers(0, len(train_shapes)))]
        else:
            for _ in range(40):
                shape = _frame_shape(rng)
                if shape not in train_shapes:
                    break
            else:
                shape = (5, 17)
        if variable_markers:
            candidates = [x for x in (base_marker, *universe) if x not in payload]
            novel = [x for x in candidates if x not in train_marker_set]
            if novel and rng.random() < .45:
                marker = int(rng.choice(novel))
            else:
                marker = int(rng.choice([x for x in train_marker_set if x in candidates]))
        else:
            marker = base_marker
        grid, output = _generate_pair(
            pair_rng, shape, payload, marker, neutral, "target_test",
            classic, False, forbidden_inputs)
        forbidden_inputs.add(_grid_signature(grid))
        puzzle["test"].append({"input": grid, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= extent <= 30 for extent in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_signature = _grid_signature(pair["input"])
            pair_signature = _pair_signature(pair["input"], pair["output"])
            assert input_signature not in inputs and pair_signature not in pairs
            inputs.add(input_signature)
            pairs.add(pair_signature)
            assert (_rows_signature(pair["input"]),
                    _rows_signature(pair["output"])) not in _SOURCE_PAIR_SIGNATURES
    for pair in puzzle["train"]:
        analyses["train"].append(_analyze_pair(pair["input"], pair["output"]))
    neutral = _infer_neutral(puzzle["train"])
    assert {x["neutral_color"] for x in analyses["train"]} == {neutral}
    assert analyses["train"][0]["exterior_payloads"] == 2
    assert len({x["frame_shape"] for x in analyses["train"]}) >= 2
    train_markers = {x["frame_color"] for x in analyses["train"]}
    for pair in puzzle["test"]:
        info = _analyze_pair(pair["input"], pair["output"], neutral)
        analyses["test"].append(info)
        assert np.array_equal(_predict_output(puzzle["train"], pair["input"]),
                              pair["output"])
        if info["frame_color"] not in train_markers:
            assert len(train_markers) >= 2


def _puzzles_equal(first, second):
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for a, b in zip(first[split], second[split]):
            if any(not np.array_equal(a[key], b[key]) for key in ("input", "output")):
                return False
    return True


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(17, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(314159), generate_puzzle(314159))
    assert _puzzles_equal(generate_puzzle(271828, 6, 3),
                          generate_puzzle(271828, 6, 3))
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                    (None, 1), (None, 4),
                                    (2, None), (10, None))):
        puzzle = generate_puzzle(4000 + index, *counts)
        if counts[0] is not None:
            assert len(puzzle["train"]) == counts[0]
        if counts[1] is not None:
            assert len(puzzle["test"]) == counts[1]
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True}, {"num_test": 1.5}):
        _expect_value_error(**kwargs)
    train_counts, test_counts = set(), set()
    canvas_shapes, frame_shapes, layouts = set(), set(), set()
    palette_sizes, test_categories, exterior_cues = set(), set(), set()
    neutral_absent = edge = source_canvas = novel_shape = 0
    marker_changes = novel_markers = unequal = 0
    corner = elongated = large_canvas = source_frame = 0
    mostly_neutral = clustered = checkerlike = 0
    cross_axis = equal_axis = changing_test_pairs = changing_test_shapes = 0
    changing_cues = all_new_new_shape = 0
    role_assignments = set()
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_analyze_pair(p["input"], p["output"])
                      for p in puzzle["train"]]
        neutral = train_info[0]["neutral_color"]
        test_info = [_analyze_pair(p["input"], p["output"], neutral)
                     for p in puzzle["test"]]
        train_palette = {c for info in train_info for c in info["payloads"]}
        train_markers = {info["frame_color"] for info in train_info}
        train_shapes = {info["frame_shape"] for info in train_info}
        role_assignments.add((neutral, tuple(sorted(train_markers))))
        palette_sizes.add(len(train_palette))
        marker_changes += len(train_markers) > 1
        changing_test_pairs += len({x["payloads"] for x in test_info}) > 1
        changing_test_shapes += len({x["frame_shape"] for x in test_info}) > 1
        changing_cues += len({x["exterior_payloads"]
                              for x in train_info + test_info}) > 1
        cross_axis += any(
            (a[0] - b[0]) * (a[1] - b[1]) < 0
            for a in train_shapes for b in train_shapes)
        equal_axis += any(
            (a[0] == b[0] and a[1] != b[1]) or
            (a[1] == b[1] and a[0] != b[0])
            for a in train_shapes for b in train_shapes)
        for split, info_list in (("train", train_info), ("test", test_info)):
            for pair, info in zip(puzzle[split], info_list):
                total += 1
                grid = pair["input"]
                canvas_shapes.add(grid.shape)
                frame_shapes.add(info["frame_shape"])
                layouts.add((grid.shape, info["bbox"], info["frame_shape"]))
                edge += 0 in info["margins"]
                margins = info["margins"]
                corner += ((margins[0] == 0 or margins[1] == 0) and
                           (margins[2] == 0 or margins[3] == 0))
                source_canvas += grid.shape == (10, 10)
                large_canvas += max(grid.shape) >= 25
                h, w = info["frame_shape"]
                elongated += min(h, w) <= 6 and max(h, w) >= 10
                source_frame += info["frame_shape"] in ((6, 6), (8, 8))
                exterior_cues.add(info["exterior_payloads"])
                a, b = info["payload_counts"]
                unequal += max(a, b) >= 3 * min(a, b)
                r0, r1, c0, c1 = info["bbox"]
                interior = grid[r0 + 1:r1, c0 + 1:c1]
                mostly_neutral += np.mean(interior == neutral) >= .65
                matching = (np.count_nonzero(interior[:, 1:] == interior[:, :-1]) +
                            np.count_nonzero(interior[1:, :] == interior[:-1, :]))
                neighboring = (interior.shape[0] * (interior.shape[1] - 1) +
                               (interior.shape[0] - 1) * interior.shape[1])
                clustered += matching / neighboring >= .7
                checkerlike += matching / neighboring <= .2
                if split == "test":
                    neutral_absent += info["interior_neutral"] == 0
                    novel_shape += info["frame_shape"] not in train_shapes
                    novel_markers += info["frame_color"] not in train_markers
                    familiar = sum(c in train_palette for c in info["payloads"])
                    test_categories.add(familiar)
                    all_new_new_shape += (familiar == 0 and
                                          info["frame_shape"] not in train_shapes)
                    assert np.array_equal(_predict_output(puzzle["train"], grid),
                                          pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(canvas_shapes) >= 40 and len(frame_shapes) >= 40
    assert len(layouts) >= total - 20 and len(role_assignments) >= 20
    assert len(palette_sizes) >= 4 and test_categories == {0, 1, 2}
    assert exterior_cues == {0, 1, 2}
    assert edge >= 30 and source_canvas >= 5 and novel_shape >= 10
    assert marker_changes >= 8 and novel_markers >= 5
    assert neutral_absent >= 2 and unequal >= 15
    assert corner >= 10 and elongated >= 20 and large_canvas >= 10
    assert source_frame >= 20 and mostly_neutral >= 10
    assert clustered >= 10 and checkerlike >= 10
    assert cross_axis >= 15 and equal_axis >= 10
    assert changing_test_pairs >= 10 and changing_test_shapes >= 10
    assert changing_cues >= 10 and all_new_new_shape >= 5
