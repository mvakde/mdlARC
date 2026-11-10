"""Generate edge-segment projection puzzles from the source ARC rule."""

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


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
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = max(2, num_test + 1)
        num_train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


_CANDIDATES = {}


def _interval_candidates(q, relation, full):
    key = (q, relation, full)
    if key not in _CANDIDATES:
        choices = []
        for long_start in range(q):
            for long_end in range(long_start + 2, q + 1):
                longer = (long_start, long_end)
                if (long_start == 0 and long_end == q) != full:
                    continue
                long_length = long_end - long_start
                for short_start in range(q):
                    for short_end in range(short_start + 1, q + 1):
                        short_length = short_end - short_start
                        if short_length >= long_length:
                            continue
                        overlap = min(long_end, short_end) - max(long_start, short_start)
                        if overlap <= 0:
                            continue
                        contained = long_start <= short_start and short_end <= long_end
                        if contained != (relation == "contained"):
                            continue
                        choices.append((longer, (short_start, short_end), overlap))
        _CANDIDATES[key] = choices
    return _CANDIDATES[key]


def _choose_intervals(rng, q, relation, full):
    choices = _interval_candidates(q, relation, full)
    _require(bool(choices), "interval category is impossible")
    # Conditional preferences expose small objects, near ties, and narrow
    # intersections without binding them to side, orientation, or size.
    preferences = []
    if rng.random() < 0.13:
        preferences.append("one")
    if rng.random() < 0.32:
        preferences.append("near")
    if rng.random() < 0.21:
        preferences.append("narrow")
    selected = choices
    for preference in preferences:
        if preference == "one":
            narrowed = [c for c in selected if c[1][1] - c[1][0] == 1]
        elif preference == "near":
            narrowed = [c for c in selected if c[0][1] - c[0][0] - (c[1][1] - c[1][0]) == 1]
        else:
            narrowed = [c for c in selected if c[2] == 1]
        if narrowed:
            selected = narrowed
    longer, shorter, _ = selected[int(rng.integers(len(selected)))]
    return longer, shorter


def _build_pair(p, q, separator_index, first, second, colors, orientation):
    background, separator, first_endpoint, second_endpoint, highlight = colors
    grid = np.full((p, q), background, dtype=np.int8)
    grid[separator_index, :] = separator
    grid[0, first[0]:first[1]] = first_endpoint
    grid[-1, second[0]:second[1]] = second_endpoint
    output = grid.copy()
    overlap = slice(max(first[0], second[0]), min(first[1], second[1]))
    if first[1] - first[0] > second[1] - second[0]:
        output[1:separator_index, overlap] = highlight
    else:
        output[separator_index + 1:p - 1, overlap] = highlight
    if orientation == "vertical":
        return {"input": grid.T.copy(), "output": output.T.copy()}
    return {"input": grid, "output": output}


def _sample_colors(rng, highlight, first_endpoint, second_endpoint):
    excluded = (highlight, first_endpoint, second_endpoint)
    available = [color for color in range(10) if color not in excluded]
    if 0 not in excluded and 2 not in excluded and rng.random() < 0.14:
        background, separator = 0, 2
    else:
        picked = rng.choice(available, size=2, replace=False)
        background, separator = int(picked[0]), int(picked[1])
    return background, separator, first_endpoint, second_endpoint, highlight


def _training_profiles(rng, count):
    if count == 2:
        profiles = [("first", "partial", False), ("second", "partial", False)]
    else:
        profiles = [("first", "partial", False), ("second", "partial", False)]
        if count == 3:
            side = "first" if rng.random() < 0.5 else "second"
            profiles.append((side, "contained", True))
        else:
            profiles.extend([("first", "contained", True), ("second", "contained", False)])
            while len(profiles) < count:
                side = "first" if rng.random() < 0.5 else "second"
                relation = "partial" if rng.random() < 0.5 else "contained"
                full = relation == "contained" and rng.random() < 0.27
                profiles.append((side, relation, full))
    rng.shuffle(profiles)
    return profiles


def _orientations(rng, num_train, num_test):
    first = "horizontal" if rng.random() < 0.5 else "vertical"
    opposite = "vertical" if first == "horizontal" else "horizontal"
    mode = int(rng.integers(5))
    if mode in (0, 1):
        return [first] * num_train, [first] * num_test
    if mode == 2:
        train = [first, opposite] + [first if rng.random() < 0.5 else opposite for _ in range(num_train - 2)]
        rng.shuffle(train)
        test = [first if rng.random() < 0.5 else opposite for _ in range(num_test)]
        return train, test
    if mode == 3:
        return [first] * num_train, [opposite] * num_test
    train = [first, opposite] + [first if rng.random() < 0.5 else opposite for _ in range(num_train - 2)]
    rng.shuffle(train)
    return train, [opposite] * num_test


def _sample_dimensions(rng, regime, prior=None):
    p_high = 21 if regime == "train" else 23
    q_high = 13 if regime == "train" else 14
    p_choices = [v for v in range(8, p_high) if prior is None or v != prior[0]]
    q_choices = [v for v in range(4, q_high) if prior is None or v != prior[1]]
    return int(rng.choice(p_choices)), int(rng.choice(q_choices))


def _sample_separator(rng, p):
    center = p // 2
    if rng.random() < 0.24:
        return center
    choices = [i for i in range(3, p - 3) if i != center]
    return int(rng.choice(choices if choices else [center]))


def _make_pair(rng, profile, orientation, endpoint, second_endpoint, highlight, regime, prior=None):
    longer_side, relation, full = profile
    p, q = _sample_dimensions(rng, regime, prior)
    separator_index = _sample_separator(rng, p)
    longer, shorter = _choose_intervals(rng, q, relation, full)
    first, second = (longer, shorter) if longer_side == "first" else (shorter, longer)
    colors = _sample_colors(rng, highlight, endpoint, second_endpoint)
    return _build_pair(p, q, separator_index, first, second, colors, orientation)


def _boundary_interval(line, background):
    indices = np.flatnonzero(line != background)
    _require(indices.size > 0, "boundary segment is missing")
    _require(np.all(np.diff(indices) == 1), "boundary segment is disconnected")
    color = int(line[indices[0]])
    _require(np.all(line[indices] == color), "boundary segment has mixed colors")
    return (int(indices[0]), int(indices[-1]) + 1), color


def _parse_input(input_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    _require(values.size in (3, 4), "input must have three or four semantic colors")
    _require(np.count_nonzero(counts == counts.max()) == 1, "background is not identifiable")
    background = int(values[np.argmax(counts)])
    candidates = []
    for orientation in ("horizontal", "vertical"):
        canonical = input_grid if orientation == "horizontal" else input_grid.T
        p, q = canonical.shape
        if p < 8 or q < 4:
            continue
        try:
            first, first_color = _boundary_interval(canonical[0], background)
            second, second_color = _boundary_interval(canonical[-1], background)
        except AssertionError:
            continue
        for separator_index in range(3, p - 3):
            line = canonical[separator_index]
            if not np.all(line == line[0]):
                continue
            separator = int(line[0])
            if separator in (background, first_color, second_color):
                continue
            expected = np.full(canonical.shape, background, dtype=input_grid.dtype)
            expected[separator_index, :] = separator
            expected[0, first[0]:first[1]] = first_color
            expected[-1, second[0]:second[1]] = second_color
            if not np.array_equal(canonical, expected):
                continue
            length_first = first[1] - first[0]
            length_second = second[1] - second[0]
            if length_first == length_second:
                continue
            overlap = (max(first[0], second[0]), min(first[1], second[1]))
            if overlap[0] >= overlap[1]:
                continue
            longer = "first" if length_first > length_second else "second"
            long_interval, short_interval = (first, second) if longer == "first" else (second, first)
            relation = "contained" if long_interval[0] <= short_interval[0] and short_interval[1] <= long_interval[1] else "partial"
            candidates.append({
                "orientation": orientation, "p": p, "q": q,
                "separator_index": separator_index, "background": background,
                "separator_color": separator, "endpoint_color": first_color,
                "second_endpoint_color": second_color,
                "first_interval": first, "second_interval": second,
                "first_length": length_first, "second_length": length_second,
                "overlap": overlap, "longer": longer, "relation": relation,
                "full": long_interval == (0, q),
            })
    _require(len(candidates) == 1, "input does not have a unique structural reading")
    return candidates[0]


def _expected_output(input_grid, info, highlight):
    canonical = input_grid.copy() if info["orientation"] == "horizontal" else input_grid.T.copy()
    start, end = info["overlap"]
    separator = info["separator_index"]
    if info["longer"] == "first":
        canonical[1:separator, start:end] = highlight
    else:
        canonical[separator + 1:info["p"] - 1, start:end] = highlight
    return canonical if info["orientation"] == "horizontal" else canonical.T.copy()


_SOURCE_PAIRS = (
    ("01100/00000/00000/00000/00000/00000/22222/00000/00000/00000/00000/00000/11110", "01100/00000/00000/00000/00000/00000/22222/04400/04400/04400/04400/04400/11110"),
    ("06666/00000/00000/00000/00000/00000/22222/00000/00000/00000/00000/00000/00666", "06666/00444/00444/00444/00444/00444/22222/00000/00000/00000/00000/00000/00666"),
    ("000330/000000/000000/000000/000000/000000/222222/000000/000000/000000/000000/000000/033333", "000330/000000/000000/000000/000000/000000/222222/000440/000440/000440/000440/000440/033333"),
    ("0333/0000/0000/0000/2222/0000/0000/0000/3300", "0333/0400/0400/0400/2222/0000/0000/0000/3300"),
    ("77777/00000/00000/00000/00000/22222/00000/00000/00000/00000/07700", "77777/04400/04400/04400/04400/22222/00000/00000/00000/00000/07700"),
)


def _source_array(rows):
    return np.array([[int(c) for c in row] for row in rows.split("/")], dtype=np.int8)


def _is_source_duplicate(input_grid, output_grid):
    return any(np.array_equal(input_grid, _source_array(a)) and np.array_equal(output_grid, _source_array(b)) for a, b in _SOURCE_PAIRS)


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"} and len(puzzle) == 2, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "regimes must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    metadata = {"train": [], "test": []}
    highlight = None
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"} and len(pair) == 2, "invalid pair keys")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a two-dimensional ndarray")
                _require(all(1 <= s <= 30 for s in grid.shape), "grid size out of range")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must contain integers")
                _require(np.all((0 <= grid) & (grid <= 9)), "ARC color out of range")
            input_grid, output_grid = pair["input"], pair["output"]
            _require(input_grid.shape == output_grid.shape, "output shape changed")
            input_key = _key(input_grid)
            pair_key = input_key, _key(output_grid)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair or input")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _require(not _is_source_duplicate(input_grid, output_grid), "exact source copy")
            info = _parse_input(input_grid)
            changed = input_grid != output_grid
            _require(np.any(changed), "empty fill")
            _require(np.all(input_grid[changed] == info["background"]), "nonbackground cell changed")
            new_colors = np.unique(output_grid[changed])
            _require(new_colors.size == 1, "multiple highlight colors")
            color = int(new_colors[0])
            _require(color not in np.unique(input_grid), "highlight occurs in input")
            if highlight is None:
                highlight = color
            _require(color == highlight, "highlight changed within puzzle")
            _require(np.array_equal(output_grid, _expected_output(input_grid, info, color)), "length-and-overlap rule violated")
            metadata[regime].append(info)
    _require(all(highlight not in np.unique(pair["input"]) for regime in ("train", "test") for pair in puzzle[regime]), "highlight appears in an input")
    train = metadata["train"]
    train_profiles = {(m["longer"], m["relation"]) for m in train}
    _require({("first", "partial"), ("second", "partial")} <= train_profiles, "training must show partial overlaps and both longer sides")
    if ntrain >= 3:
        _require(any(m["relation"] == "contained" for m in train), "training must show containment")
        _require(any(m["full"] for m in train), "training must show a full boundary")
    if ntrain >= 4:
        _require(len(train_profiles) == 4, "training must show all side/relation combinations")
        _require(any(m["endpoint_color"] != m["second_endpoint_color"] for m in train), "two-color edges must be demonstrated")
    if ntrain == 2:
        _require(all(m["relation"] == "partial" for m in metadata["test"]), "two-example training cannot support containment tests")
    if any(m["endpoint_color"] != m["second_endpoint_color"] for m in metadata["test"]):
        _require(any(m["endpoint_color"] != m["second_endpoint_color"] for m in train), "two-color test edges lack a demonstration")
    _require(len({m["endpoint_color"] for m in train}) >= 2, "training must vary edge colors")
    _require(len({m["p"] for m in train}) >= 2 and len({m["q"] for m in train}) >= 2, "training sizes must vary")
    # Every output convention needed at test time is visible in the training
    # outputs; orientations and role colors are read directly from each input.
    _require(all(np.any(pair["input"] != pair["output"]) for pair in puzzle["train"]), "training does not demonstrate highlight")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    highlight = int(rng.integers(10))
    available = [c for c in range(10) if c != highlight]
    train_palette = [int(c) for c in rng.choice(available, size=3, replace=False)]
    profiles = _training_profiles(rng, num_train)
    train_orientations, test_orientations = _orientations(rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    used = set()
    train_endpoints = []
    two_color_train_index = int(rng.integers(num_train)) if num_train >= 4 else None
    prior_size = None
    for index, profile in enumerate(profiles):
        if index < 2:
            endpoint = train_palette[index]
        else:
            endpoint = train_palette[int(rng.integers(3))]
        train_endpoints.append(endpoint)
        if index == two_color_train_index:
            second_endpoint = int(rng.choice([c for c in available if c != endpoint]))
        else:
            second_endpoint = endpoint
        for _ in range(40):
            pair = _make_pair(rng, profile, train_orientations[index], endpoint, second_endpoint, highlight, "train", prior_size if index == 1 else None)
            if _key(pair["input"]) not in used and not _is_source_duplicate(pair["input"], pair["output"]):
                break
        else:
            raise AssertionError("could not construct a unique training pair")
        used.add(_key(pair["input"]))
        puzzle["train"].append(pair)
        if index == 0:
            info = _parse_input(pair["input"])
            prior_size = info["p"], info["q"]
    unseen_endpoints = [c for c in available if c not in train_endpoints]
    for index in range(num_test):
        side = "first" if rng.random() < 0.5 else "second"
        relation = "partial" if num_train == 2 or rng.random() < 0.55 else "contained"
        full = relation == "contained" and rng.random() < 0.27
        profile = side, relation, full
        if rng.random() < 0.5:
            endpoint = int(rng.choice(train_endpoints))
        else:
            endpoint = int(rng.choice(unseen_endpoints))
        if num_train >= 4 and rng.random() < 0.34:
            second_endpoint = int(rng.choice([c for c in available if c != endpoint]))
        else:
            second_endpoint = endpoint
        for _ in range(40):
            pair = _make_pair(rng, profile, test_orientations[index], endpoint, second_endpoint, highlight, "test")
            if _key(pair["input"]) not in used and not _is_source_duplicate(pair["input"], pair["output"]):
                break
        else:
            raise AssertionError("could not construct a unique test pair")
        used.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _same_puzzle(a, b):
    return all(np.array_equal(pa[name], pb[name]) for regime in ("train", "test") for pa, pb in zip(a[regime], b[regime]) for name in ("input", "output")) and all(len(a[r]) == len(b[r]) for r in ("train", "test"))


def self_test() -> None:
    for args in ((73, None, None), (101, 3, 1), (102, 6, 2), (103, 9, 4), (121, 2, 1), (122, 10, 1), (123, 10, 4), (1301, None, 1), (1302, None, 4), (1303, 2, None), (1304, 10, None)):
        puzzle = generate_puzzle(*args)
        validate_generated_puzzle(puzzle)
        if args[1] is not None:
            _require(len(puzzle["train"]) == args[1], "train count ignored")
        if args[2] is not None:
            _require(len(puzzle["test"]) == args[2], "test count ignored")
    _require(_same_puzzle(generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3)), "nondeterministic puzzle")
    _require(not _same_puzzle(generate_puzzle(900, 6, 2), generate_puzzle(901, 6, 2)), "seed collision")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}, {"num_train": 2.0, "num_test": 1}):
        try:
            generate_puzzle(1400, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"invalid counts accepted: {kwargs}")
    train_counts, test_counts = set(), set()
    fingerprints, layouts, shapes = set(), set(), set()
    sides, relations, combinations = set(), set(), set()
    full_train = full_test = short_one = narrow = near = different_train = different_test = 0
    mixed_orientation = flipped_orientation = reused_test_color = novel_test_color = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple((regime, _key(pair["input"]), _key(pair["output"])) for regime in ("train", "test") for pair in puzzle[regime]))
        infos = {regime: [_parse_input(pair["input"]) for pair in puzzle[regime]] for regime in ("train", "test")}
        train_orientations = {m["orientation"] for m in infos["train"]}
        if len(train_orientations) == 2:
            mixed_orientation += 1
        if any(m["orientation"] not in train_orientations for m in infos["test"]):
            flipped_orientation += 1
        train_colors = {m["endpoint_color"] for m in infos["train"]}
        for m in infos["test"]:
            reused_test_color += m["endpoint_color"] in train_colors
            novel_test_color += m["endpoint_color"] not in train_colors
        for regime in ("train", "test"):
            for pair, m in zip(puzzle[regime], infos[regime]):
                changed = pair["output"] != pair["input"]
                observed_highlight = int(np.unique(pair["output"][changed])[0])
                _require(np.array_equal(pair["output"], _expected_output(pair["input"], m, observed_highlight)), "self-test output mismatch")
                sides.add(m["longer"])
                relations.add(m["relation"])
                combinations.add((m["longer"], m["relation"]))
                shapes.add((m["orientation"], m["p"], m["q"]))
                layouts.add((m["orientation"], m["p"], m["q"], m["separator_index"], m["first_interval"], m["second_interval"]))
                full_train += regime == "train" and m["full"]
                full_test += regime == "test" and m["full"]
                different_train += regime == "train" and m["endpoint_color"] != m["second_endpoint_color"]
                different_test += regime == "test" and m["endpoint_color"] != m["second_endpoint_color"]
                short_one += min(m["first_length"], m["second_length"]) == 1
                narrow += m["overlap"][1] - m["overlap"][0] == 1
                near += abs(m["first_length"] - m["second_length"]) == 1
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity too low")
    _require(len(fingerprints) >= 40 and len(layouts) >= 180 and len(shapes) >= 50, "structural diversity too low")
    _require(len(sides) == 2 and len(relations) == 2 and len(combinations) == 4, "factor combinations missing")
    _require(full_train >= 15 and full_test >= 2, "full-boundary coverage too low")
    _require(short_one >= 5 and narrow >= 15 and near >= 20, "small-geometry coverage too low")
    _require(mixed_orientation >= 5 and flipped_orientation >= 3, "orientation diversity too low")
    _require(reused_test_color >= 5 and novel_test_color >= 5, "test color regimes missing")
    _require(different_train >= 15 and different_test >= 5, "two-color edge coverage too low")
