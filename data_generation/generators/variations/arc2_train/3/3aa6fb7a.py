import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))
_SOURCE_PAIRS = (
    (("0000880", "0000080", "0080000", "0088000", "0000000", "0000800", "0008800"),
     ("0000880", "0000180", "0081000", "0088000", "0000000", "0001800", "0008800")),
    (("0000000", "0800000", "0880000", "0000880", "0000080", "0000000", "0000000"),
     ("0000000", "0810000", "0880000", "0000880", "0000180", "0000000", "0000000")),
    (("0000088", "8800008", "8000000", "0008000", "0008800", "0800000", "8800000"),
     ("0000088", "8800018", "8100000", "0008100", "0008800", "1800000", "8800000")),
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _seed(seed):
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(seed)


def _counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None and num_test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif num_train is None:
        test = int(num_test)
        train = int(rng.integers(max(2, test + 1), 11))
    elif num_test is None:
        train = int(num_train)
        test = int(rng.integers(1, min(4, train - 1) + 1))
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return (tuple(grid.shape), tuple(int(value) for value in grid.flat))


def _source_copy(inp, out):
    rows_in = tuple("".join(str(int(v)) for v in row) for row in inp)
    rows_out = tuple("".join(str(int(v)) for v in row) for row in out)
    return (rows_in, rows_out) in _SOURCE_PAIRS


def _grid(grid, name):
    _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
             f"{name} must be a two-dimensional NumPy array")
    h, w = grid.shape
    _require(1 <= h <= 30 and 1 <= w <= 30, f"{name} has illegal dimensions")
    _require(np.issubdtype(grid.dtype, np.integer), f"{name} must have integer cells")
    _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, f"{name} has illegal colors")


def _components(inp, background):
    pending = {(int(r), int(c)) for r, c in np.argwhere(inp != background)}
    found = []
    while pending:
        start = pending.pop()
        group = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in pending:
                    pending.remove(p)
                    group.add(p)
                    stack.append(p)
        found.append(group)
    return found


def _analyze_pair(inp, out):
    _grid(inp, "input")
    _grid(out, "output")
    _require(inp.shape == out.shape, "output dimensions changed")
    colors, counts = np.unique(inp, return_counts=True)
    _require(len(colors) >= 2, "input has no L")
    largest = int(np.argmax(counts))
    _require(int(counts[largest]) > max((int(v) for i, v in enumerate(counts) if i != largest), default=0),
             "background is not a unique majority color")
    bg = int(colors[largest])
    groups = _components(inp, bg)
    _require(len(groups) >= 1, "input has no components")
    expected_missing = []
    boxes = []
    orientations = []
    foreground_colors = []
    for group in groups:
        _require(len(group) == 3, "foreground component is not a triomino")
        group_colors = {int(inp[p]) for p in group}
        _require(len(group_colors) == 1, "an L must be monochrome")
        r = min(p[0] for p in group)
        c = min(p[1] for p in group)
        _require(max(p[0] for p in group) == r + 1 and max(p[1] for p in group) == c + 1,
                 "an L must fit a 2 by 2 box")
        missing = [(i, (r + dr, c + dc)) for i, (dr, dc) in enumerate(_CORNERS)
                   if (r + dr, c + dc) not in group]
        _require(len(missing) == 1 and int(inp[missing[0][1]]) == bg,
                 "an L must have one background corner")
        orientations.append(missing[0][0])
        expected_missing.append(missing[0][1])
        boxes.append((r, c))
        foreground_colors.append(next(iter(group_colors)))
    for i, (r, c) in enumerate(boxes):
        for rr, cc in boxes[i + 1:]:
            _require(abs(r - rr) >= 2 or abs(c - cc) >= 2, "L boxes overlap")
    _require(len(set(expected_missing)) == len(groups), "two Ls share a missing corner")
    changes = np.argwhere(inp != out)
    _require(len(changes) == len(groups), "must add exactly one cell per L")
    new_colors = {int(out[tuple(p)]) for p in changes}
    _require(len(new_colors) == 1, "completion color varies inside a pair")
    completion = next(iter(new_colors))
    _require(completion not in set(int(v) for v in colors), "completion color must be absent from input")
    expected = inp.copy()
    for p in expected_missing:
        expected[p] = completion
    _require(np.array_equal(out, expected), "output is not exact L completion")
    h, w = inp.shape
    edge = sum(r == 0 or c == 0 or r == h - 2 or c == w - 2 for r, c in boxes)
    close = sum(max(abs(r - rr), abs(c - cc)) == 2
                for i, (r, c) in enumerate(boxes) for rr, cc in boxes[i + 1:])
    return {"background": bg, "completion": completion,
            "foreground_colors": tuple(foreground_colors), "count": len(groups),
            "orientations": tuple(orientations), "edge": edge,
            "interior": len(groups) - edge, "close": close, "boxes": tuple(boxes)}


def _predict_from_training(train, test_input):
    # This uses only demonstrations and the test input, never a test output or seed.
    learned = set()
    for pair in train:
        a, b = pair["input"], pair["output"]
        learned.update(int(b[tuple(p)]) for p in np.argwhere(a != b))
    _require(len(learned) == 1, "training does not identify a completion color")
    completion = next(iter(learned))
    colors, counts = np.unique(test_input, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    prediction = test_input.copy()
    for component in _components(test_input, background):
        r = min(p[0] for p in component)
        c = min(p[1] for p in component)
        missing = [(r + dr, c + dc) for dr, dc in _CORNERS
                   if (r + dr, c + dc) not in component]
        _require(len(component) == 3 and len(missing) == 1, "test object is ambiguous")
        prediction[missing[0]] = completion
    return prediction


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    ntrain, ntest = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest,
             "invalid pair counts")
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must contain only input and output")
            inp, out = pair["input"], pair["output"]
            info = _analyze_pair(inp, out)
            analyses[section].append(info)
            ik, ok = _key(inp), _key(out)
            _require(ik not in inputs and (ik, ok) not in pairs, "duplicate pair or input")
            _require(not _source_copy(inp, out), "exact source pair copied")
            inputs.add(ik)
            pairs.add((ik, ok))
    all_info = analyses["train"] + analyses["test"]
    completion = analyses["train"][0]["completion"]
    _require(all(x["completion"] == completion for x in all_info),
             "puzzle completion color is inconsistent")
    _require(all(completion not in set(int(v) for v in np.unique(pair["input"]))
                 for section in ("train", "test") for pair in puzzle[section]),
             "completion color appears in an input")
    _require(len({x["background"] for x in analyses["train"]}) >= 2,
             "training must demonstrate changing backgrounds")
    _require(len({c for x in analyses["train"] for c in x["foreground_colors"]}) >= 2,
             "training must demonstrate changing L colors")
    _require(any(len(set(x["foreground_colors"])) >= 2 for x in analyses["train"]),
             "training must demonstrate multicolored scenes")
    seen = {o for x in analyses["train"] for o in x["orientations"]}
    _require(len(seen) >= 3, "training does not establish rotation invariance")
    _require(seen == {0, 1, 2, 3} or len(seen) == 3,
             "invalid training orientation coverage")
    if len(seen) == 3:
        unseen_corner = next(iter(set(range(4)) - seen))
        _require(any(unseen_corner in x["orientations"] for x in analyses["test"]),
                 "target test omits the held-out rotation")
    for pair in puzzle["test"]:
        _require(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"]),
                 "test output cannot be predicted from training and test input")


def _orientation_specs(rng, counts, choices):
    slots = [(i, j) for i, count in enumerate(counts) for j in range(count)]
    order = rng.permutation(len(slots))
    specs = [[-1] * count for count in counts]
    for value, k in zip(choices, order):
        i, j = slots[int(k)]
        specs[i][j] = int(value)
    for k in order[len(choices):]:
        i, j = slots[int(k)]
        specs[i][j] = int(rng.choice(choices))
    return specs


def _dimensions(rng, count):
    category = int(rng.choice(4, p=(0.15, 0.20, 0.45, 0.20)))
    if category == 0 and count <= 3:
        if rng.random() < 0.25:
            h, w = 7, 7
        else:
            h, w = int(rng.integers(3, 7)), int(rng.integers(3, 7))
    elif category == 1:
        short = int(rng.integers(3 if count <= 7 else 5, 7))
        long = int(rng.integers(max(9, min(30, 2 * count + 4)), 31))
        h, w = (short, long) if rng.integers(2) else (long, short)
    elif category == 3:
        if rng.random() < 0.08:
            h, w = 30, 30
        else:
            h, w = int(rng.integers(19, 31)), int(rng.integers(19, 31))
    else:
        low = max(7, 3 * int(np.ceil(np.sqrt(count))))
        h, w = int(rng.integers(low, 19)), int(rng.integers(low, 19))
    return h, w


def _edge(pos, h, w):
    r, c = pos
    return r == 0 or c == 0 or r == h - 2 or c == w - 2


def _placement(rng, h, w, orientations, style):
    count = len(orientations)
    all_positions = [(r, c) for r in range(h - 1) for c in range(w - 1)]
    if style == "edge":
        requirements = ["edge"] * count
    elif style == "interior":
        requirements = ["interior"] * count
    elif style == "mixed" and count >= 2:
        requirements = ["edge", "interior"] + ["any"] * (count - 2)
    else:
        requirements = ["any"] * count
    for _ in range(12):
        selected = []
        occupied = set()
        order = rng.permutation(count)
        cluster = bool(rng.integers(2))
        for j in range(count):
            orientation = orientations[int(order[j])]
            requirement = requirements[j]
            candidates = [p for p in all_positions
                          if requirement == "any" or (requirement == "edge") == _edge(p, h, w)]
            rng.shuffle(candidates)
            if cluster and selected:
                candidates.sort(key=lambda p: min(abs(p[0] - q[0][0]) + abs(p[1] - q[0][1])
                                                  for q in selected))
            chosen = None
            for r, c in candidates:
                if any(abs(r - q[0][0]) < 2 and abs(c - q[0][1]) < 2 for q in selected):
                    continue
                cells = {(r + dr, c + dc) for i, (dr, dc) in enumerate(_CORNERS)
                         if i != orientation}
                if any(p in occupied or any(n in occupied for n in
                       ((p[0] - 1, p[1]), (p[0] + 1, p[1]),
                        (p[0], p[1] - 1), (p[0], p[1] + 1))) for p in cells):
                    continue
                chosen = ((r, c), orientation, cells)
                break
            if chosen is None:
                break
            selected.append(chosen)
            occupied.update(chosen[2])
        if len(selected) == count and h * w - len(occupied) > len(occupied):
            return selected
    return None


def _colors(rng, completion, count, background_exclude, foreground_exclude, force_multi):
    bg_options = [x for x in range(10) if x != completion and x != background_exclude]
    background = int(rng.choice(bg_options))
    options = [x for x in range(10)
               if x not in (completion, background, foreground_exclude)]
    ncolors = (int(rng.integers(2, min(count, 4) + 1)) if count >= 2 and
               (force_multi or rng.random() < 0.40) else 1)
    chosen = [int(x) for x in rng.choice(options, size=ncolors, replace=False)]
    assignments = [chosen[i % ncolors] for i in range(count)]
    rng.shuffle(assignments)
    return background, assignments


def _make_pair(rng, completion, orientations, background_exclude=None,
               foreground_exclude=None, force_multi=False):
    count = len(orientations)
    style = str(rng.choice(("edge", "interior", "mixed", "free")))
    for attempt in range(60):
        h, w = _dimensions(rng, count) if attempt < 45 else (30, 30)
        placed = _placement(rng, h, w, orientations, style)
        if placed is not None:
            break
    else:
        raise RuntimeError("could not place L objects")
    background, object_colors = _colors(rng, completion, count, background_exclude,
                                         foreground_exclude, force_multi)
    inp = np.full((h, w), background, dtype=np.int8)
    out = inp.copy()
    for ((r, c), orientation, cells), color in zip(placed, object_colors):
        for p in cells:
            inp[p] = color
            out[p] = color
        dr, dc = _CORNERS[orientation]
        out[r + dr, c + dc] = completion
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _counts(seed, rng, num_train, num_test)
    completion = int(rng.integers(0, 10))
    held_out = int(rng.integers(0, 4)) if rng.random() < 0.55 else None
    train_choices = [i for i in range(4) if i != held_out]
    train_counts = [int(rng.integers(1, 11)) for _ in range(ntrain)]
    if sum(train_counts) < len(train_choices):
        train_counts[int(rng.integers(ntrain))] += len(train_choices) - sum(train_counts)
    if max(train_counts) < 2:
        train_counts[int(rng.integers(ntrain))] = 2
    test_counts = [int(rng.integers(1, 11)) for _ in range(ntest)]
    train_specs = _orientation_specs(rng, train_counts, train_choices)
    test_specs = [[int(rng.integers(4)) for _ in range(count)] for count in test_counts]
    if held_out is not None:
        index = int(rng.integers(ntest))
        test_specs[index][int(rng.integers(test_counts[index]))] = held_out
    force_multi_index = next(i for i, count in enumerate(train_counts) if count >= 2)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_bg = None
    first_fg = None
    for section, specs in (("train", train_specs), ("test", test_specs)):
        for i, orientations in enumerate(specs):
            for _ in range(20):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(pair_seed)
                pair = _make_pair(pair_rng, completion, orientations,
                                  first_bg if section == "train" and i == 1 else None,
                                  first_fg if section == "train" and i == 1 else None,
                                  section == "train" and i == force_multi_index)
                key = _key(pair["input"])
                if key not in seen and not _source_copy(pair["input"], pair["output"]):
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
            seen.add(key)
            puzzle[section].append(pair)
            if section == "train" and i == 0:
                info = _analyze_pair(pair["input"], pair["output"])
                first_bg = info["background"]
                first_fg = info["foreground_colors"][0]
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "too few train counts")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "too few test counts")
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, "too few distinct puzzles")
    shapes, layouts, colors = set(), set(), set()
    singleton = repeated = many = multi = close = small = large = narrow = 0
    heldout = familiar = 0
    held_corners = set()
    completion_roles, background_roles, foreground_roles = set(), set(), set()
    split_coverage = {"train": {"singleton": 0, "many": 0, "close": 0,
                                 "edge": 0, "interior": 0, "mixed": 0},
                      "test": {"singleton": 0, "many": 0, "close": 0,
                               "edge": 0, "interior": 0, "mixed": 0}}
    all_edge = all_interior = mixed = 0
    lower_or_equal = higher = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_infos = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test_infos = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        train_orientations = {o for x in train_infos for o in x["orientations"]}
        if len(train_orientations) == 3:
            heldout += 1
            held_corners.update(set(range(4)) - train_orientations)
            _require(any(o not in train_orientations for x in test_infos for o in x["orientations"]),
                     "held-out rotation never tested")
        else:
            familiar += 1
        for x in test_infos:
            lower_or_equal += x["count"] <= max(t["count"] for t in train_infos)
            higher += x["count"] > max(t["count"] for t in train_infos)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                x = _analyze_pair(inp, out)
                h, w = inp.shape
                shapes.add((h, w))
                layouts.add((inp.shape, tuple(sorted(x["boxes"])), tuple(sorted(x["orientations"]))))
                colors.add((x["background"], x["completion"]))
                completion_roles.add(x["completion"])
                background_roles.add(x["background"])
                foreground_roles.update(x["foreground_colors"])
                singleton += x["count"] == 1
                repeated += len(set(x["orientations"])) < x["count"]
                many += x["count"] >= 7
                multi += len(set(x["foreground_colors"])) >= 2
                close += x["close"] > 0
                small += h <= 6 and w <= 6
                large += h >= 19 and w >= 19
                narrow += min(h, w) <= 6 and max(h, w) >= 9
                all_edge += x["edge"] == x["count"]
                all_interior += x["interior"] == x["count"]
                mixed += x["edge"] > 0 and x["interior"] > 0
                split_coverage[section]["singleton"] += x["count"] == 1
                split_coverage[section]["many"] += x["count"] >= 7
                split_coverage[section]["close"] += x["close"] > 0
                split_coverage[section]["edge"] += x["edge"] == x["count"]
                split_coverage[section]["interior"] += x["interior"] == x["count"]
                split_coverage[section]["mixed"] += x["edge"] > 0 and x["interior"] > 0
                _require(len(np.argwhere(inp != out)) == x["count"], "incorrect addition count")
        for pair in puzzle["test"]:
            _require(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"]),
                     "train-only prediction failed")
    _require(len(shapes) >= 35 and len(layouts) >= 150 and len(colors) >= 20,
             "insufficient structural or palette diversity")
    _require(all(v > 0 for v in (singleton, repeated, many, multi, close, small, large,
                                 narrow, heldout, familiar, all_edge, all_interior,
                                 mixed, lower_or_equal, higher)), "missing review diversity axis")
    _require(held_corners == set(range(4)), "held-out corner does not vary")
    _require(all(len(roles) == 10 for roles in
                 (completion_roles, background_roles, foreground_roles)),
             "ARC colors do not occur in all supported roles")
    _require(all(value > 0 for values in split_coverage.values() for value in values.values()),
             "one split lacks a supported geometry or placement regime")
    for i, (a, b) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1),
                                (None, 4), (2, None), (10, None))):
        p = generate_puzzle(7000 + i, a, b)
        validate_generated_puzzle(p)
        _require((a is None or len(p["train"]) == a) and (b is None or len(p["test"]) == b),
                 "requested pair counts not honored")
        _require(_puzzle_key(p) == _puzzle_key(generate_puzzle(7000 + i, a, b)),
                 "generation is not deterministic")
    _require(_puzzle_key(generate_puzzle(7000)) != _puzzle_key(generate_puzzle(7001)),
             "different seeds gave identical puzzles")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4),
                 (2.5, 1), (3, True)):
        try:
            generate_puzzle(10000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
    staggered = np.zeros((7, 7), dtype=np.int8)
    for (r, c), missing in (((1, 1), 3), ((3, 3), 0)):
        for index, (dr, dc) in enumerate(_CORNERS):
            if index != missing:
                staggered[r + dr, c + dc] = 8
    completed = staggered.copy()
    completed[2, 2] = 1
    completed[3, 3] = 1
    _require(_analyze_pair(staggered, completed)["close"] == 1,
             "valid diagonally staggered Ls were rejected")
    damaged = generate_puzzle(11000, 2, 1)
    damaged["test"][0]["output"] = damaged["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an unchanged output")
