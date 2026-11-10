"""Horizontal carrier and marker stalk ARC family."""

import numpy as np


_SOURCE_INPUT_ROWS = (
    ("000000", "000000", "000000", "181111", "000000", "000000", "000000", "000000", "000000"),
    ("0000000", "0000000", "0000000", "0000000", "0000000", "1112111", "0000000", "0000000", "0000000", "0000000"),
    ("000000000", "000000000", "000000000", "000000000", "000000000", "000000000", "000000000", "181118121", "000000000"),
    ("0000000000", "0000000000", "0000000000", "0000000000", "0000000000", "0000000000", "0000000000", "0000000000", "0000000000", "1121118111", "0000000000", "0000000000", "0000000000"),
    ("00000000000", "00000000000", "00000000000", "00000000000", "00000000000", "00000000000", "12181118121", "00000000000", "00000000000"),
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_pairs():
    pairs = set()
    for rows in _SOURCE_INPUT_ROWS:
        inp = np.array([[int(x) for x in row] for row in rows], dtype=np.int8)
        out = inp.copy()
        line = next(r for r in range(inp.shape[0]) if np.any(inp[r] != 0))
        for c in range(inp.shape[1]):
            marker = int(inp[line, c])
            if marker in (8, 2):
                distance = 3 if marker == 8 else 4
                out[line - distance + 1:line, c] = 1
                out[line - distance, c] = marker
        pairs.add((_key(inp), _key(out)))
    return pairs


_SOURCE_PAIRS = _source_pairs()


def _count_arg(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(rng, num_train, num_test):
    train = _count_arg("num_train", num_train, 2, 10)
    test = _count_arg("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _parameters(rng):
    colors = [int(x) for x in rng.choice(10, size=4, replace=False)]
    if int(rng.integers(4)) == 0:
        distances = (3, 4)
    else:
        distances = tuple(sorted(int(x) for x in rng.choice(np.arange(2, 7), size=2, replace=False)))
    return {
        "short": colors[2], "tall": colors[3],
        "background": colors[0], "carrier": colors[1],
        "distances": {colors[2]: distances[0], colors[3]: distances[1]},
        "palette_mode": int(rng.integers(4)),
    }


def _local_palette(rng, parameters, index):
    background = parameters["background"]
    carrier = parameters["carrier"]
    mode = parameters["palette_mode"]
    pool = [x for x in range(10) if x not in (parameters["short"], parameters["tall"])]
    if index and mode in (1, 3):
        choices = [x for x in pool if x != background and (index != 1 or x != parameters["background"])]
        background = int(rng.choice(choices))
    if index and mode in (2, 3):
        choices = [x for x in pool if x != background and (index != 1 or x != parameters["carrier"])]
        carrier = int(rng.choice(choices))
    elif carrier == background:
        carrier = next(x for x in pool if x != background)
    return background, carrier


def _training_roles(rng, count):
    if count == 2:
        choice = int(rng.integers(4))
        if choice == 0:
            plans = [("short",), ("tall", "short", "tall")]
        elif choice == 1:
            plans = [("tall",), ("short", "tall", "short")]
        elif choice == 2:
            plans = [("short", "tall"), ("tall", "short", "short")]
        else:
            plans = [("short", "short", "tall"), ("tall", "short")]
    else:
        repeated = "short" if int(rng.integers(2)) == 0 else "tall"
        plans = [("short",), ("tall",), ("short", "tall", repeated)]
        for _ in range(count - 3):
            n = int(rng.integers(1, 4))
            plans.append(tuple("short" if int(rng.integers(2)) == 0 else "tall" for _ in range(n)))
    rng.shuffle(plans)
    return plans


def _test_roles(rng, train_plans, count):
    max_train = max(len(plan) for plan in train_plans)
    modes = ["increase", "familiar"] if count >= 2 else ["increase" if int(rng.integers(2)) else "familiar"]
    modes += ["increase" if int(rng.integers(2)) else "familiar" for _ in range(count - len(modes))]
    rng.shuffle(modes)
    plans = []
    for mode in modes:
        if mode == "increase":
            n = int(rng.integers(max_train + 1, 9))
        else:
            n = int(rng.integers(1, max_train + 1))
        plans.append(tuple("short" if int(rng.integers(2)) == 0 else "tall" for _ in range(n)))
    return plans


def _positions(rng, width, count):
    mode = int(rng.integers(4))
    if mode == 0:
        compressed = np.sort(rng.choice(width - count + 1, size=count, replace=False))
        return tuple(int(x) for x in compressed + np.arange(count))
    if mode == 1 and count >= 2:
        start = int(rng.integers(width - 1))
        chosen = {start, start + 1}
    elif mode == 2:
        chosen = {0 if int(rng.integers(2)) == 0 else width - 1}
    elif mode == 3 and count >= 3:
        length = int(rng.integers(2, min(4, count) + 1))
        start = int(rng.integers(width - length + 1))
        chosen = set(range(start, start + length))
    else:
        chosen = set()
    remaining = [x for x in range(width) if x not in chosen]
    chosen.update(int(x) for x in rng.choice(remaining, size=count - len(chosen), replace=False))
    return tuple(sorted(chosen))


def _make_pair(rng, roles, parameters, index, regime):
    marker_count = len(roles)
    width = int(rng.integers(max(5, 2 * marker_count + 1), (22 if regime == "source_train" else 24) + 1))
    height = int(rng.integers(7, 25))
    background, carrier = _local_palette(rng, parameters, index)
    colors = [parameters[role] for role in roles]
    maximum_distance = max(parameters["distances"][color] for color in colors)
    max_below = min(height - maximum_distance - 1, 8)
    below = int(rng.integers(min(max_below, 4) + 1))
    if max_below > 4 and int(rng.integers(5)) == 0:
        below = int(rng.integers(5, max_below + 1))
    line = height - below - 1
    columns = _positions(rng, width, marker_count)
    role_order = rng.permutation(marker_count)
    inp = np.full((height, width), background, dtype=np.int8)
    inp[line, :] = carrier
    out = inp.copy()
    for column, role_index in zip(columns, role_order):
        marker = colors[int(role_index)]
        distance = parameters["distances"][marker]
        inp[line, column] = marker
        out[line, column] = marker
        out[line - distance + 1:line, column] = carrier
        out[line - distance, column] = marker
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, num_train, num_test)
    parameters = _parameters(rng)
    train_plans = _training_roles(rng, train_count)
    test_plans = _test_roles(rng, train_plans, test_count)
    entropy = rng.integers(0, np.iinfo(np.uint32).max, size=(train_count + test_count, 4), dtype=np.uint32)
    puzzle = {"train": [], "test": []}
    used = set()
    for index, (regime, roles) in enumerate(
        [("source_train", x) for x in train_plans] + [("target_test", x) for x in test_plans]
    ):
        pair_rng = np.random.default_rng(entropy[index])
        for _ in range(80):
            pair = _make_pair(pair_rng, roles, parameters, index, regime)
            key = _key(pair["input"])
            if key not in used and (_key(pair["input"]), _key(pair["output"])) not in _SOURCE_PAIRS:
                break
        else:
            raise RuntimeError("could not make a unique pair")
        used.add(key)
        puzzle["train" if regime == "source_train" else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a two-dimensional NumPy array")
    _require(all(1 <= x <= 30 for x in grid.shape), "grid dimensions outside 1..30")
    _require(np.issubdtype(grid.dtype, np.integer) and grid.dtype != np.dtype(np.bool_), "grid must be integer")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "grid color outside 0..9")


def _parse_input(grid):
    height, width = grid.shape
    _require(height >= 7 and width >= 5, "carrier canvas is too small")
    background = int(grid[0, 0])
    rows = np.flatnonzero(np.any(grid != background, axis=1))
    _require(len(rows) == 1, "input needs exactly one carrier row")
    line = int(rows[0])
    _require(line >= 2, "carrier needs headroom")
    baseline = grid[line]
    colors, counts = np.unique(baseline, return_counts=True)
    carrier = int(colors[int(np.argmax(counts))])
    _require(carrier != background, "carrier and background must differ")
    columns = tuple(int(x) for x in np.flatnonzero(baseline != carrier))
    _require(1 <= len(columns) <= 8, "invalid marker count")
    _require(width >= 2 * len(columns) + 1, "carrier must remain the clear majority of the line")
    marker_colors = {int(baseline[c]) for c in columns}
    _require(background not in marker_colors and len(marker_colors) <= 2, "invalid marker palette")
    return {"background": background, "carrier": carrier, "line": line,
            "columns": columns, "marker_colors": marker_colors, "height": height, "width": width}


def _distance_in_output(inp, out, info, column):
    line = info["line"]
    carrier = info["carrier"]
    marker = int(inp[line, column])
    distance = 1
    while line - distance >= 0 and int(out[line - distance, column]) == carrier:
        distance += 1
    _require(line - distance >= 0 and int(out[line - distance, column]) == marker,
             "stalk must end in a copy of its marker")
    _require(2 <= distance <= 6, "stalk distance outside supported range")
    return distance


def _expected(inp, info, mapping):
    out = inp.copy()
    for column in info["columns"]:
        marker = int(inp[info["line"], column])
        _require(marker in mapping, "test uses an unseen marker")
        distance = mapping[marker]
        _require(info["line"] >= distance, "marker has insufficient headroom")
        out[info["line"] - distance + 1:info["line"], column] = info["carrier"]
        out[info["line"] - distance, column] = marker
    return out


def _describe(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest, "invalid pair counts")
    infos = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            _require(inp.shape == out.shape and not np.array_equal(inp, out), "invalid output dimensions or unchanged output")
            pair_key = (_key(inp), _key(out))
            _require(pair_key[0] not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            _require(pair_key not in _SOURCE_PAIRS, "exact source pair copy")
            seen_inputs.add(pair_key[0])
            seen_pairs.add(pair_key)
            info = _parse_input(inp)
            _require(info["width"] <= (22 if split == "train" else 24) and info["height"] <= 24,
                     "pair outside construction regime")
            _require(len(info["columns"]) <= (3 if split == "train" else 8), "invalid regime marker count")
            infos[split].append(info)
    mapping = {}
    for pair, info in zip(puzzle["train"], infos["train"]):
        for column in info["columns"]:
            marker = int(pair["input"][info["line"], column])
            distance = _distance_in_output(pair["input"], pair["output"], info, column)
            if marker in mapping:
                _require(mapping[marker] == distance, "marker distance changes within puzzle")
            mapping[marker] = distance
    _require(len(mapping) == 2 and len(set(mapping.values())) == 2, "train must teach two distinct stalk distances")
    _require(any(len(info["marker_colors"]) == 2 for info in infos["train"]), "train needs a mixed-marker demonstration")
    for split in ("train", "test"):
        for pair, info in zip(puzzle[split], infos[split]):
            _require(info["marker_colors"].issubset(mapping), "unseen marker color")
            _require(info["background"] not in mapping and info["carrier"] not in mapping,
                     "local background or carrier overlaps a marker identity")
            _require(np.array_equal(pair["output"], _expected(pair["input"], info, mapping)),
                     "output violates the carrier stalk rule")
    if ntest >= 2:
        max_train = max(len(x["columns"]) for x in infos["train"])
        counts = [len(x["columns"]) for x in infos["test"]]
        _require(any(x > max_train for x in counts) and any(x <= max_train for x in counts),
                 "tests need both increased and familiar marker counts")
    return infos, mapping


def validate_generated_puzzle(puzzle: dict) -> None:
    _describe(puzzle)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(x["train"]) for x in puzzles}) >= 4, "train counts lack diversity")
    _require(len({len(x["test"]) for x in puzzles}) >= 3, "test counts lack diversity")
    _require(len({tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in x[s]) for x in puzzles}) >= 40,
             "puzzles lack diversity")
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 7, 3),
                              (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        _require(train is None or len(puzzle["train"]) == train, "train count request ignored")
        _require(test is None or len(puzzle["test"]) == test, "test count request ignored")
    for seed, train, test in ((300, None, None), (301, 8, 3)):
        a, b = generate_puzzle(seed, train, test), generate_puzzle(seed, train, test)
        _require(all(np.array_equal(x[role], y[role]) for split in ("train", "test")
                     for x, y in zip(a[split], b[split]) for role in ("input", "output")),
                 "generation is nondeterministic")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                 {"num_train": 2.0, "num_test": 1}, {"num_train": 4, "num_test": True}):
        try:
            generate_puzzle(400, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count request accepted: {args}")
    all_infos = [_describe(puzzle) for puzzle in puzzles]
    shapes = {tuple(pair["input"].shape) for puzzle in puzzles for split in ("train", "test") for pair in puzzle[split]}
    _require(len(shapes) >= 50, "grid sizes lack diversity")
    distances = {tuple(sorted(mapping.values())) for _, mapping in all_infos}
    _require(len(distances) >= 6 and (3, 4) in distances, "stalk distances lack diversity")
    marker_counts = {len(info["columns"]) for infos, _ in all_infos for split in ("train", "test") for info in infos[split]}
    _require(set(range(1, 6)).issubset(marker_counts), "marker counts lack diversity")
    _require(any(info["line"] == info["height"] - 1 for infos, _ in all_infos for split in ("train", "test") for info in infos[split]),
             "bottom carriers were not sampled")
    _require(any(any(b - a == 1 for a, b in zip(info["columns"], info["columns"][1:]))
                 for infos, _ in all_infos for split in ("train", "test") for info in infos[split]),
             "touching markers were not sampled")
    _require(any(any(c in (0, info["width"] - 1) for c in info["columns"])
                 for infos, _ in all_infos for split in ("train", "test") for info in infos[split]),
             "edge markers were not sampled")
    _require(any(len({info["background"] for split in ("train", "test") for info in infos[split]}) > 1
                 for infos, _ in all_infos), "background colors never vary within puzzles")
    _require(any(len({info["carrier"] for split in ("train", "test") for info in infos[split]}) > 1
                 for infos, _ in all_infos), "carrier colors never vary within puzzles")
    train_counts = {len(info["columns"]) for infos, _ in all_infos for info in infos["train"]}
    test_counts = {len(info["columns"]) for infos, _ in all_infos for info in infos["test"]}
    _require(train_counts == {1, 2, 3} and set(range(1, 9)).issubset(test_counts),
             "train or test marker-count regimes lack coverage")
    _require(all(any(len(info["marker_colors"]) == 2 for info in infos["train"])
                 for infos, _ in all_infos), "a puzzle lacks mixed training evidence")
    _require(all(any(len(info["columns"]) > len(info["marker_colors"]) for info in infos["train"])
                 for infos, _ in all_infos), "a puzzle lacks repeated-role training evidence")
    _require(any(all(len({info[name] for split in ("train", "test") for info in infos[split]}) == 1
                     for name in ("background", "carrier")) for infos, _ in all_infos),
             "constant-palette puzzles are absent")
    _require(any(all(len({info[name] for split in ("train", "test") for info in infos[split]}) > 1
                     for name in ("background", "carrier")) for infos, _ in all_infos),
             "puzzles varying both local palette roles are absent")
    _require(any(any(len(info["columns"]) >= 3 and len(set(np.diff(info["columns"]))) > 1
                     for info in infos[split]) for infos, _ in all_infos for split in ("train", "test")),
             "uneven marker clusters are absent")
    _require(any(any(info["width"] >= 17 and len(info["columns"]) >= 6 for info in infos["test"])
                 for infos, _ in all_infos), "large grids with proportionate marker counts are absent")
    _require(any(any(info["width"] <= 8 and info["height"] <= 10 for info in infos[split])
                 for infos, _ in all_infos for split in ("train", "test")),
             "compact examples are absent")
