import numpy as np


def _digit_grid(rows):
    return np.array(
        [[ord(ch) - ord("0") for ch in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_digit_grid(input_rows), _digit_grid(output_rows))
    for input_rows, output_rows in (
        (
            (
                "7777777",
                "7877777",
                "7872777",
                "7872777",
                "7872777",
                "7872777",
            ),
            (
                "7777777",
                "7877777",
                "7872777",
                "7872777",
                "7872777",
                "7872757",
            ),
        ),
        (
            (
                "7777777",
                "7777777",
                "7777777",
                "7778777",
                "7878777",
                "7878777",
            ),
            (
                "7777777",
                "7777757",
                "7777757",
                "7778757",
                "7878757",
                "7878757",
            ),
        ),
        (
            (
                "77777777777",
                "77777777777",
                "77777277777",
                "77787277777",
                "77787277777",
                "77787278777",
                "78787278777",
                "78787278777",
            ),
            (
                "77777777777",
                "77777777777",
                "77777277777",
                "77787277777",
                "77787277757",
                "77787278757",
                "78787278757",
                "78787278757",
            ),
        ),
        (
            (
                "777777777777777",
                "777777777772777",
                "787777777772777",
                "787777777772777",
                "787777787772777",
                "787777787772777",
                "787277787772777",
                "787278787772777",
                "787278787272777",
            ),
            (
                "777777777777777",
                "777777777772777",
                "787777777772777",
                "787777777772777",
                "787777787772777",
                "787777787772777",
                "787277787772777",
                "787278787772757",
                "787278787272757",
            ),
        ),
    )
)


def _normalize_seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        low = max(2, num_test + 1)
        num_train = low + seed % (11 - low)
    elif num_test is None:
        high = min(4, num_train - 1)
        num_test = 1 + seed % high

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _sample_puzzle_parameters(rng, seed):
    positive, negative, result = (int(x) for x in rng.choice(10, 3, replace=False))
    backgrounds = [c for c in range(10) if c not in (positive, negative, result)]
    rng.shuffle(backgrounds)
    return {
        "positive": positive, "negative": negative, "result": result,
        "backgrounds": backgrounds, "curriculum": seed % 5,
        "orientation_mode": (seed // 5) % 3,
        "growth": int(rng.integers(0, 4)),
        "reverse": bool(rng.integers(0, 2)),
    }


def _bounded_composition(total, parts, cap, rng):
    assert parts >= 1 and parts <= total <= parts * cap
    values = []
    for index in range(parts - 1):
        after = parts - index - 1
        low = max(1, total - cap * after)
        high = min(cap, total - after)
        value = int(rng.integers(low, high + 1))
        values.append(value)
        total -= value
    values.append(total)
    return values


def _sample_signed_lengths(rng, count, negatives, cap):
    positives = count - negatives
    if negatives == 0:
        result = int(rng.integers(count + 1, cap + 1))
        terms = [(1, v) for v in _bounded_composition(result, count, cap, rng)]
    else:
        low = max(negatives, positives - cap)
        high = min(negatives * cap, positives * cap - 1)
        assert low <= high
        negative_total = int(rng.integers(low, high + 1))
        result_low = max(1, positives - negative_total)
        result_high = min(cap, positives * cap - negative_total)
        if result_low == 1 and rng.random() < 0.22:
            result = 1
        elif result_high == cap and rng.random() < 0.18:
            result = cap
        else:
            result = int(rng.integers(result_low, result_high + 1))
        positive_lengths = _bounded_composition(negative_total + result, positives, cap, rng)
        negative_lengths = _bounded_composition(negative_total, negatives, cap, rng)
        terms = [(1, v) for v in positive_lengths] + [(-1, v) for v in negative_lengths]
    order = rng.permutation(count)
    terms = [terms[int(i)] for i in order]
    if negatives and rng.random() < 0.5:
        equal_pairs = [(i, j) for i, (s, v) in enumerate(terms)
                       for j, (ss, vv) in enumerate(terms)
                       if s == 1 and ss == -1 and v == vv]
        if equal_pairs:
            i, j = equal_pairs[int(rng.integers(len(equal_pairs)))]
            term = terms.pop(j)
            if j < i:
                i -= 1
            terms.insert(i + 1, term)
    return terms, result


def _transform(grid, turns, reverse):
    grid = np.rot90(grid, turns)
    return np.fliplr(grid) if reverse else grid


def _untransform(grid, turns, reverse):
    if reverse:
        grid = np.fliplr(grid)
    return np.rot90(grid, -turns)


def _choose_count(rng, low, high):
    return int(rng.integers(low, high + 1))


def _pair_plan(rng, curriculum, split, index, train_info):
    if split == "train":
        if index == 0:
            if curriculum == 4:
                count = _choose_count(rng, 3, 5)
            else:
                count = _choose_count(rng, 2, 4)
            negatives = 0
        elif index == 1:
            bounds = ((2, 4), (4, 7), (3, 5), (3, 6), (5, 8))[curriculum]
            count = _choose_count(rng, *bounds)
            negatives = 2 if curriculum in (1, 4) and count >= 3 else 1
        else:
            bounds = ((2, 5), (3, 8), (2, 5), (3, 6), (4, 10))[curriculum]
            count = _choose_count(rng, *bounds)
            negatives = _choose_count(rng, 0, 1 if curriculum == 3 else count - 1)
    else:
        if curriculum == 0:  # original long, tall, signed shift
            count = _choose_count(rng, 6, 12)
            negatives = _choose_count(rng, 2, count - 1)
        elif curriculum == 1:  # overlap, including reordered expressions
            count = _choose_count(rng, 2, 8)
            negatives = _choose_count(rng, 0, count - 1)
        elif curriculum == 2:  # more terms without greater depth
            count = _choose_count(rng, 6, 12)
            negatives = _choose_count(rng, 2, count - 1)
        elif curriculum == 3:  # more minus terms at familiar sizes
            count = _choose_count(rng, 3, min(6, train_info["max_count"]))
            negatives = _choose_count(rng, 2, count - 1)
        else:  # small, sometimes all positive, after richer training
            count = _choose_count(rng, 2, 5)
            negatives = 0 if index == 0 else _choose_count(rng, 0, count - 1)
    if split == "train" and curriculum in (0, 2):
        depth_range = (5, 12) if curriculum == 0 else (5, 24)
    elif split == "test" and curriculum == 0:
        depth_range = (14, 30)
    elif split == "test" and curriculum == 2:
        depth_range = (3, max(3, train_info["max_depth"]))
    else:
        depth_range = ((3, 6), (7, 16), (17, 30))[int(rng.integers(0, 3))]
    return count, negatives, depth_range


def _orientation(parameters, split, index, rng):
    mode = parameters["orientation_mode"]
    if mode == 0:
        turns = parameters["growth"]
        reverse = parameters["reverse"]
    elif mode == 1:
        turns = (parameters["growth"] + index + (0 if split == "train" else 2)) % 4
        reverse = bool(rng.integers(0, 2))
    else:
        turns = (parameters["growth"] + (0 if split == "train" else 1 + index)) % 4
        reverse = parameters["reverse"] if split == "train" else bool(rng.integers(0, 2))
    return turns, reverse


def _make_pair(rng, parameters, count, negatives, depth_range, thickness, turns, reverse,
               background, override=None):
    if override is not None:
        terms = list(override)
        count = len(terms)
        negatives = sum(s == -1 for s, _ in terms)
        minimum_cap = max(thickness + 1, max(v for _, v in terms), sum(s * v for s, v in terms))
    else:
        positives = count - negatives
        minimum_cap = max(thickness + 1,
                          count + thickness if negatives == 0 else 2,
                          (negatives + positives) // positives,
                          (positives + negatives) // (negatives + 1))
    minimum_depth = max(3, minimum_cap)
    depth_low = max(depth_range[0], minimum_depth)
    depth_high = max(depth_low, depth_range[1])
    depth = _choose_count(rng, depth_low, min(30, depth_high))
    below = _choose_count(rng, 0, min(3, depth - minimum_cap))
    cap = depth - below
    if override is None:
        if negatives == 0:
            result = _choose_count(rng, count + thickness, cap)
            lengths = [thickness + 1] + _bounded_composition(
                result - thickness - 1, count - 1, cap, rng)
            terms = [(1, lengths[int(i)]) for i in rng.permutation(count)]
        else:
            for _ in range(80):
                terms, result = _sample_signed_lengths(rng, count, negatives, cap)
                if len({v for _, v in terms}) >= 2 and max(v for _, v in terms) > thickness:
                    break
            else:
                raise RuntimeError("could not obtain recognizable bar lengths")
    else:
        result = sum(s * v for s, v in terms)
        assert 1 <= result <= cap and len({v for _, v in terms}) >= 2
        order = rng.permutation(count)
        terms = [terms[int(i)] for i in order]
        if terms == list(override):
            terms = terms[1:] + terms[:1]
    pitches = [p for p in range(thickness + 1, 6) if count * p + thickness <= 30]
    if not pitches:
        raise RuntimeError("expression does not fit")
    pitch = int(rng.choice(pitches))
    max_lead = min(pitch - 1, 3, 30 - count * pitch - thickness)
    lead = _choose_count(rng, 0, max_lead)
    trail = _choose_count(rng, 0, min(4, 30 - lead - count * pitch - thickness))
    width = lead + count * pitch + thickness + trail
    canonical = np.full((depth, width), background, dtype=np.int8)
    baseline = depth - 1 - below
    for slot, (sign, length) in enumerate(terms):
        col = lead + slot * pitch
        color = parameters["positive"] if sign == 1 else parameters["negative"]
        canonical[baseline - length + 1:baseline + 1, col:col + thickness] = color
    target = canonical.copy()
    result_col = lead + count * pitch
    target[baseline - result + 1:baseline + 1,
           result_col:result_col + thickness] = parameters["result"]
    return {"input": _transform(canonical, turns, reverse).copy(),
            "output": _transform(target, turns, reverse).copy()}


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _matches_source_pair(inp, out):
    return any(np.array_equal(inp, a) and np.array_equal(out, b) for a, b in _SOURCE_PAIRS)


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _canonical_candidate(grid, turns, reverse, background):
    canonical = _untransform(grid, turns, reverse)
    height, width = canonical.shape
    mask = canonical != background
    occupied = np.any(mask, axis=0)
    starts, ends = [], []
    col = 0
    while col < width:
        if occupied[col]:
            start = col
            while col < width and occupied[col]:
                col += 1
            starts.append(start)
            ends.append(col)
        else:
            col += 1
    count = len(starts)
    if count < 2 or count > 12:
        return None
    thickness = ends[0] - starts[0]
    if not 1 <= thickness <= 3 or any(e - s != thickness for s, e in zip(starts, ends)):
        return None
    pitch = starts[1] - starts[0]
    if not thickness + 1 <= pitch <= 5 or any(starts[i] - starts[i-1] != pitch for i in range(1, count)):
        return None
    if starts[0] >= pitch or starts[-1] + pitch + thickness > width:
        return None
    if width - (starts[-1] + pitch + thickness) > 4:
        return None
    baselines, lengths, colors = [], [], []
    for start in starts:
        block = canonical[:, start:start + thickness]
        rows = np.flatnonzero(np.any(block != background, axis=1))
        if not len(rows) or not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            return None
        if not np.all(block[rows] != background):
            return None
        values = np.unique(block[rows])
        if len(values) != 1:
            return None
        baselines.append(int(rows[-1]))
        lengths.append(int(len(rows)))
        colors.append(int(values[0]))
    if len(set(baselines)) != 1 or height - 1 - baselines[0] > 3:
        return None
    if len(set(lengths)) < 2 or max(lengths) <= thickness:
        return None
    result_col = starts[-1] + pitch
    if np.any(canonical[:, result_col:result_col + thickness] != background):
        return None
    return {"canonical": canonical, "turns": turns, "reverse": reverse,
            "background": background, "count": count, "thickness": thickness,
            "pitch": pitch, "lead": starts[0], "trail": width-result_col-thickness,
            "baseline": baselines[0], "lengths": tuple(lengths), "colors": tuple(colors),
            "result_col": result_col, "depth": height, "width": width}


def _parse_input(grid):
    background = _background(grid)
    candidates = [_canonical_candidate(grid, turns, reverse, background)
                  for turns in range(4) for reverse in (False, True)]
    candidates = [c for c in candidates if c is not None]
    assert len(candidates) == 1, f"input has {len(candidates)} geometric interpretations"
    return candidates[0]


def _parse_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    info = _parse_input(inp)
    canonical_out = _untransform(out, info["turns"], info["reverse"])
    left = info["result_col"]
    right = left + info["thickness"]
    outside = np.ones(info["canonical"].shape, dtype=bool)
    outside[:, left:right] = False
    assert np.array_equal(canonical_out[outside], info["canonical"][outside])
    block = canonical_out[:, left:right]
    rows = np.flatnonzero(np.any(block != info["background"], axis=1))
    assert len(rows) >= 1
    assert np.array_equal(rows, np.arange(info["baseline"] - len(rows) + 1, info["baseline"] + 1))
    assert np.all(block[rows] != info["background"])
    values = np.unique(block[rows])
    assert len(values) == 1
    assert np.all(block[:rows[0]] == info["background"])
    assert np.all(block[rows[-1]+1:] == info["background"])
    info["result_length"] = int(len(rows))
    info["result_color"] = int(values[0])
    assert info["result_color"] != info["background"]
    return info


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= d <= 30 for d in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _inspect_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    parsed = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            _check_grid(pair["input"]); _check_grid(pair["output"])
            key = _grid_key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert not _matches_source_pair(pair["input"], pair["output"])
            parsed[split].append(_parse_pair(pair))
    train = parsed["train"]
    result_colors = {x["result_color"] for x in train}
    operand_palette = {color for x in train for color in x["colors"]}
    assert len(result_colors) == 1 and len(operand_palette) == 2
    result_color = next(iter(result_colors))
    assert result_color not in operand_palette
    assert all(x["result_color"] == result_color for x in parsed["test"])
    assert all(x["background"] not in operand_palette | {result_color}
               for x in train + parsed["test"])
    palette = tuple(operand_palette)
    mappings = []
    for positive in palette:
        if all(sum((v if c == positive else -v) for c, v in zip(x["colors"], x["lengths"]))
               == x["result_length"] for x in train):
            mappings.append(positive)
    assert len(mappings) == 1
    positive = mappings[0]
    negative = next(c for c in palette if c != positive)
    assert any(all(c == positive for c in x["colors"]) for x in train)
    assert any(negative in x["colors"] for x in train)
    assert {x["thickness"] for x in train} >= {1, 2}
    for x in train + parsed["test"]:
        assert set(x["colors"]) <= operand_palette
        expected = sum(v if c == positive else -v for c, v in zip(x["colors"], x["lengths"]))
        assert 1 <= expected <= x["baseline"] + 1
        assert expected == x["result_length"]
    return {"positive": positive, "negative": negative, "result": result_color, "parsed": parsed}


def validate_generated_puzzle(puzzle: dict) -> None:
    _inspect_puzzle(puzzle)


def _fresh_pair(rng, parameters, count, negatives, depth_range, thickness, turns, reverse,
                background, seen, override=None):
    for _ in range(100):
        pair = _make_pair(rng, parameters, count, negatives, depth_range, thickness,
                          turns, reverse, background, override)
        key = _grid_key(pair["input"])
        if key in seen or _matches_source_pair(pair["input"], pair["output"]):
            continue
        try:
            _parse_pair(pair)
        except AssertionError:
            continue
        seen.add(key)
        return pair
    raise RuntimeError("could not create an unambiguous fresh pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng, seed)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_info = {"max_depth": 3, "max_count": 2}
    # The two mandatory demonstrations can appear in either order.
    demo_order = [0, 1] if bool(rng.integers(0, 2)) else [1, 0]
    backgrounds = parameters["backgrounds"]
    for split, total in (("train", train_count), ("test", test_count)):
        for index in range(total):
            logical_index = demo_order[index] if split == "train" and index < 2 else index
            count, negatives, depth_range = _pair_plan(
                rng, parameters["curriculum"], split, logical_index, train_info)
            override = None
            if split == "test" and index == 0 and parameters["curriculum"] == 1:
                source_info = _parse_input(puzzle["train"][demo_order.index(1)]["input"])
                override = [(1 if c == parameters["positive"] else -1, v)
                            for c, v in zip(source_info["colors"], source_info["lengths"])]
                count = len(override)
                negatives = sum(s == -1 for s, _ in override)
            thickness = 1 if split == "train" and logical_index == 0 else (
                2 if split == "train" and logical_index == 1 else
                _choose_count(rng, 1, 3))
            if count * (thickness + 1) + thickness > 30:
                thickness = 1
            if override is not None:
                thickness = min(thickness, max(v for _, v in override) - 1)
            turns, reverse = _orientation(parameters, split, index, rng)
            background = backgrounds[(index if split == "train" else train_count + index) % len(backgrounds)]
            pair = _fresh_pair(rng, parameters, count, negatives, depth_range, thickness,
                               turns, reverse, background, seen, override)
            puzzle[split].append(pair)
            if split == "train":
                train_info["max_depth"] = max(train_info["max_depth"],
                                               _parse_input(pair["input"])["depth"])
                train_info["max_count"] = max(train_info["max_count"], count)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, layouts, palettes = set(), set(), set(), set(), set()
    widths, depths, counts, negatives, pitches, thicknesses, orientations = set(), set(), set(), set(), set(), set(), set()
    majority_negative = shallow_many = equal_adjacent = negative_prefix = near_cancel = full_height = 0
    overlap = multiple_train_negative = all_positive_test = translated = varied_background = 0
    permutations = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        inspection = _inspect_puzzle(puzzle)
        train_counts.add(len(puzzle["train"])); test_counts.add(len(puzzle["test"]))
        palettes.add((inspection["positive"], inspection["negative"], inspection["result"]))
        train = inspection["parsed"]["train"]
        test = inspection["parsed"]["test"]
        overlap += any(x["count"] == y["count"] for x in train for y in test)
        multiple_train_negative += any(x["colors"].count(inspection["negative"]) >= 2 for x in train)
        all_positive_test += any(all(c == inspection["positive"] for c in x["colors"]) for x in test)
        varied_background += len({x["background"] for x in train + test}) > 1
        multisets = [sorted(zip(x["colors"], x["lengths"])) for x in train]
        permutations += any(sorted(zip(x["colors"], x["lengths"])) in multisets for x in test)
        for split in ("train", "test"):
            for pair, x in zip(puzzle[split], inspection["parsed"][split]):
                _check_grid(pair["input"]); _check_grid(pair["output"])
                shapes.add(pair["input"].shape)
                widths.add(x["width"]); depths.add(x["depth"]); counts.add(x["count"])
                pitches.add(x["pitch"]); thicknesses.add(x["thickness"])
                orientations.add((x["turns"], x["reverse"]))
                q = x["colors"].count(inspection["negative"])
                negatives.add(q)
                majority_negative += q > x["count"] - q
                shallow_many += x["depth"] <= 5 and x["count"] >= 6
                equal_adjacent += any(x["colors"][i] != x["colors"][i+1] and
                                      x["lengths"][i] == x["lengths"][i+1]
                                      for i in range(x["count"]-1))
                running = 0
                for c, v in zip(x["colors"], x["lengths"]):
                    running += v if c == inspection["positive"] else -v
                    negative_prefix += running < 0
                near_cancel += x["result_length"] == 1
                full_height += x["result_length"] == x["baseline"] + 1
                translated += x["baseline"] < x["depth"] - 1 or x["lead"] != 1
                rebuilt = x["canonical"].copy()
                rebuilt[x["baseline"]-x["result_length"]+1:x["baseline"]+1,
                        x["result_col"]:x["result_col"]+x["thickness"]] = inspection["result"]
                assert np.array_equal(_transform(rebuilt, x["turns"], x["reverse"]), pair["output"])
                layouts.add((x["depth"], x["width"], x["count"], x["pitch"], x["thickness"],
                             x["lead"], x["trail"], x["turns"], x["reverse"],
                             tuple(x["lengths"]), tuple(x["colors"])))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 20 and len(shapes) >= 40 and len(layouts) >= 150
    assert max(depths) >= 28 and max(counts) >= 9 and min(depths) <= 5
    assert len(pitches) >= 3 and thicknesses == {1, 2, 3}
    assert len(orientations) >= 8 and majority_negative > 0 and shallow_many > 0
    assert equal_adjacent > 0 and negative_prefix > 0 and near_cancel > 0
    assert full_height > 0 and overlap > 0 and multiple_train_negative > 0
    assert all_positive_test > 0 and translated > 0 and varied_background >= 40
    assert permutations > 0
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    assert not _puzzles_equal(generate_puzzle(0), generate_puzzle(1))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(20000 + 10*train_count + test_count, train_count, test_count)
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for seed, a, b in ((30101, None, 1), (30104, None, 4), (30201, 2, None), (31001, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(77, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
