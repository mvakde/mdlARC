"""Solved ARC puzzles using a clipped, direction-colored diagonal stencil."""

import numpy as np


_DIRECTIONS = (("nw", -1, -1), ("ne", -1, 1),
               ("sw", 1, -1), ("se", 1, 1))
_KINDS = ("interior", "top_edge", "bottom_edge", "left_edge",
          "right_edge", "top_left_corner", "top_right_corner",
          "bottom_left_corner", "bottom_right_corner")
_EDGES = _KINDS[1:5]
_CORNERS = _KINDS[5:]
_SOURCE = (
    (((0, 0, 2, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
     ((0, 0, 0, 0, 0), (0, 8, 0, 7, 0), (0, 0, 0, 0, 0))),
    (((0, 0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 0, 2)),
     ((0, 0, 0, 0, 0), (0, 0, 0, 3, 0), (0, 0, 0, 0, 0))),
    (((0, 0, 0, 0, 0), (0, 0, 0, 2, 0), (0, 0, 0, 0, 0)),
     ((0, 0, 3, 0, 6), (0, 0, 0, 0, 0), (0, 0, 8, 0, 7))),
    (((0, 0, 0, 0, 0), (0, 2, 0, 0, 0), (0, 0, 0, 0, 0)),
     ((3, 0, 6, 0, 0), (0, 0, 0, 0, 0), (8, 0, 7, 0, 0))),
    (((0, 0, 0, 0, 0), (0, 0, 0, 0, 2), (0, 0, 0, 0, 0)),
     ((0, 0, 0, 3, 0), (0, 0, 0, 0, 0), (0, 0, 0, 8, 0))),
)
_SOURCE_KEYS = frozenset(
    ((len(inp), len(inp[0]), tuple(v for row in inp for v in row)),
     (len(out), len(out[0]), tuple(v for row in out for v in row)))
    for inp, out in _SOURCE
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in {lower}..{upper}")
    return value


def _counts(rng, seed, num_train, num_test):
    num_train = _count("num_train", num_train, 2, 10)
    num_test = _count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        minimum = max(2, num_test + 1)
        num_train = minimum + (seed // 4) % (11 - minimum)
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _category(kind):
    if kind == "interior":
        return 4
    if kind in _EDGES:
        return 2
    return 1


def _train_kinds(rng, count):
    # Both regimes reveal all four role colors and demonstrate clipping.
    if int(rng.integers(0, 5)) < 3:
        first = ["interior", str(rng.choice(_KINDS[1:]))]
    elif int(rng.integers(0, 2)):
        first = ["top_edge", "bottom_edge"]
    else:
        first = ["left_edge", "right_edge"]
    if count >= 3 and "interior" not in first:
        first.append(str(rng.choice(("top_edge", "bottom_edge",
                                     "left_edge", "right_edge"))))
    kinds = list(first)
    while len(kinds) < count:
        order = [str(_KINDS[int(i)]) for i in rng.permutation(len(_KINDS))]
        kinds.extend(kind for kind in order if kind not in kinds or kind not in _CORNERS)
    kinds = kinds[:count]
    rng.shuffle(kinds)
    return kinds


def _test_kinds(rng, count, train_kinds):
    classes = (("interior",), _EDGES, _CORNERS)
    categories = [int(rng.choice(3, p=(0.28, 0.47, 0.25)))]
    if count >= 2:
        categories.append(int(rng.choice([i for i in range(3) if i != categories[0]])))
    while len(categories) < count:
        categories.append(int(rng.choice(3, p=(0.28, 0.47, 0.25))))
    result = []
    for i, category in enumerate(categories):
        candidates = list(classes[category])
        if i == 0:
            unseen = [kind for kind in candidates if kind not in train_kinds]
            if unseen:
                candidates = unseen
        if category == 1:
            # The source-like right edge remains a somewhat common target.
            candidates += ["right_edge"] if "right_edge" in candidates else []
        result.append(str(rng.choice(candidates)))
    return result


def _min_dimensions(kind):
    return (3 if kind == "interior" or kind in ("left_edge", "right_edge") else 2,
            3 if kind == "interior" or kind in ("top_edge", "bottom_edge") else 2)


def _sample_dimensions(rng, kind):
    minimum_h, minimum_w = _min_dimensions(kind)
    def side(minimum):
        if int(rng.integers(0, 10)) < 7:
            return int(rng.integers(minimum, 7))
        return int(rng.integers(7, 15))
    return side(minimum_h), side(minimum_w)


def _position(rng, height, width, kind):
    if kind == "interior":
        return int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1))
    if kind == "top_edge":
        return 0, int(rng.integers(1, width - 1))
    if kind == "bottom_edge":
        return height - 1, int(rng.integers(1, width - 1))
    if kind == "left_edge":
        return int(rng.integers(1, height - 1)), 0
    if kind == "right_edge":
        return int(rng.integers(1, height - 1)), width - 1
    return (0 if kind.startswith("top") else height - 1,
            0 if "left" in kind else width - 1)


def _destinations(height, width, row, col):
    return {name: (row + dr, col + dc) for name, dr, dc in _DIRECTIONS
            if 0 <= row + dr < height and 0 <= col + dc < width}


def _kind_of(height, width, row, col):
    if row == 0 and col == 0:
        return "top_left_corner"
    if row == 0 and col == width - 1:
        return "top_right_corner"
    if row == height - 1 and col == 0:
        return "bottom_left_corner"
    if row == height - 1 and col == width - 1:
        return "bottom_right_corner"
    if row == 0:
        return "top_edge"
    if row == height - 1:
        return "bottom_edge"
    if col == 0:
        return "left_edge"
    if col == width - 1:
        return "right_edge"
    return "interior"


def _key(grid):
    return grid.shape[0], grid.shape[1], tuple(int(x) for x in grid.flat)


def _make_pair(rng, dimensions, kind, background, marker, colors):
    height, width = dimensions
    row, col = _position(rng, height, width, kind)
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    inp[row, col] = marker
    for direction, (target_row, target_col) in _destinations(height, width, row, col).items():
        out[target_row, target_col] = colors[direction]
    return {"input": inp, "output": out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    role_colors = [int(x) for x in rng.choice(10, size=4, replace=False)]
    colors = {name: color for (name, _, _), color in zip(_DIRECTIONS, role_colors)}
    local_colors = [color for color in range(10) if color not in role_colors]
    base_local = tuple(int(x) for x in rng.choice(local_colors, size=2, replace=False))
    stable_local = bool(rng.integers(0, 3) == 0)
    train_kinds = _train_kinds(rng, train_count)
    test_kinds = _test_kinds(rng, test_count, train_kinds)
    kinds = train_kinds + test_kinds

    # Some puzzles keep one canvas, some reuse it for a subset of pairs.
    dimension_mode = int(rng.integers(0, 3))
    minimum_h = max(_min_dimensions(kind)[0] for kind in kinds)
    minimum_w = max(_min_dimensions(kind)[1] for kind in kinds)
    shared_dimensions = _sample_dimensions(rng, "interior")
    shared_dimensions = (max(shared_dimensions[0], minimum_h),
                         max(shared_dimensions[1], minimum_w))
    if dimension_mode == 1 and shared_dimensions[0] * shared_dimensions[1] < len(kinds):
        shared_dimensions = (max(shared_dimensions[0], 4), max(shared_dimensions[1], 4))
    puzzle = {"train": [], "test": []}
    used = set()
    local_choices = [(a, b) for a in local_colors for b in local_colors if a != b]

    for index, kind in enumerate(kinds):
        if stable_local or index == 0:
            background, marker = base_local
        else:
            choices = [choice for choice in local_choices
                       if index != 1 or choice != base_local]
            background, marker = choices[int(rng.integers(len(choices)))]
        pair_rng = np.random.default_rng(
            np.random.SeedSequence(rng.integers(0, 2**32, size=4, dtype=np.uint32)))
        pair = None
        for attempt in range(160):
            if dimension_mode == 1 or (dimension_mode == 2 and index % 2 == 0):
                dimensions = shared_dimensions
            else:
                dimensions = _sample_dimensions(pair_rng, kind)
            if attempt >= 32:
                dimensions = _sample_dimensions(pair_rng, kind)
            candidate = _make_pair(pair_rng, dimensions, kind, background, marker, colors)
            key = _key(candidate["input"])
            if key in used or (key, _key(candidate["output"])) in _SOURCE_KEYS:
                continue
            pair = candidate
            used.add(key)
            break
        if pair is None:
            raise RuntimeError("could not construct a unique pair")
        puzzle["train" if index < train_count else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[k], list) for k in ("train", "test")),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    learned, all_roles = {}, {}
    train_kinds, test_kinds = [], []
    train_counts, test_counts = [], []
    locals_seen, dimensions = [], []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label}: malformed pair")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{label}: grid must be a 2D array")
                _require(np.issubdtype(grid.dtype, np.integer), f"{label}: noninteger grid")
                _require(all(2 <= side <= 30 for side in grid.shape), f"{label}: invalid size")
                _require(bool(np.all((0 <= grid) & (grid <= 9))), f"{label}: invalid color")
            _require(inp.shape == out.shape, f"{label}: shape changed")
            input_key, output_key = _key(inp), _key(out)
            _require(input_key not in seen_inputs, f"{label}: duplicate input")
            _require((input_key, output_key) not in seen_pairs, f"{label}: duplicate pair")
            _require((input_key, output_key) not in _SOURCE_KEYS, f"{label}: source copy")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            values, counts = np.unique(inp, return_counts=True)
            _require(len(values) == 2 and sorted(int(x) for x in counts) == [1, inp.size - 1],
                     f"{label}: expected one marker on uniform background")
            background = int(values[np.argmax(counts)])
            marker = int(values[np.argmin(counts)])
            row, col = (int(x) for x in np.argwhere(inp == marker)[0])
            destinations = _destinations(*inp.shape, row, col)
            _require(bool(destinations), f"{label}: no in-bounds diagonals")
            expected = set(destinations.values())
            actual = {(int(r), int(c)) for r, c in np.argwhere(out != background)}
            _require(actual == expected, f"{label}: diagonal geometry is wrong")
            _require(int(out[row, col]) == background, f"{label}: marker not erased")
            for direction, (r, c) in destinations.items():
                color = int(out[r, c])
                _require(color not in (background, marker), f"{label}: role color collision")
                if direction in all_roles:
                    _require(all_roles[direction] == color, f"{label}: changed direction color")
                all_roles[direction] = color
                if split == "train":
                    learned[direction] = color
            kind = _kind_of(*inp.shape, row, col)
            (train_kinds if split == "train" else test_kinds).append(kind)
            (train_counts if split == "train" else test_counts).append(len(destinations))
            locals_seen.append((background, marker))
            dimensions.append(inp.shape)
    _require(set(learned) == {name for name, _, _ in _DIRECTIONS},
             "training does not reveal all four directions")
    _require(len(set(learned.values())) == 4, "direction roles are not distinct")
    _require(all(set(local).isdisjoint(learned.values()) for local in locals_seen),
             "a local background or marker collides with a direction role")
    _require(any(count < 4 for count in train_counts), "training does not show clipping")
    _require(all_roles == learned, "test changes the training map")
    _require(all(count in (1, 2, 4) for count in test_counts), "invalid test geometry")
    _require(test_count < 2 or len(set(test_counts)) >= 2,
             "multiple tests should mix clipping outcomes")
    return {"train_kinds": train_kinds, "test_kinds": test_kinds,
            "train_counts": train_counts, "test_counts": test_counts,
            "local_colors": locals_seen, "dimensions": dimensions,
            "mapping": learned}


def validate_generated_puzzle(puzzle: dict) -> None:
    _analyze(puzzle)


def _infer(train, test_input):
    mapping = {}
    for pair in train:
        inp, out = pair["input"], pair["output"]
        values, counts = np.unique(inp, return_counts=True)
        background = int(values[np.argmax(counts)])
        marker = int(values[np.argmin(counts)])
        row, col = (int(x) for x in np.argwhere(inp == marker)[0])
        for direction, (r, c) in _destinations(*inp.shape, row, col).items():
            color = int(out[r, c])
            if direction in mapping:
                _require(mapping[direction] == color, "contradictory training map")
            mapping[direction] = color
    _require(len(mapping) == 4, "incomplete training map")
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[np.argmax(counts)])
    marker = int(values[np.argmin(counts)])
    row, col = (int(x) for x in np.argwhere(test_input == marker)[0])
    predicted = np.full(test_input.shape, background, dtype=np.int8)
    for direction, (r, c) in _destinations(*test_input.shape, row, col).items():
        predicted[r, c] = mapping[direction]
    return predicted


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    infos = [_analyze(puzzle) for puzzle in puzzles]
    train_counts = {len(p["train"]) for p in puzzles}
    test_counts = {len(p["test"]) for p in puzzles}
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len({_key(p["train"][0]["input"]) for p in puzzles}) >= 35,
             "structural diversity")
    all_dimensions = {shape for info in infos for shape in info["dimensions"]}
    _require(len(all_dimensions) >= 20, "dimension diversity")
    all_test_counts = {count for info in infos for count in info["test_counts"]}
    _require(all_test_counts == {1, 2, 4}, "test clipping diversity")
    _require(any("right_edge" in info["train_kinds"] for info in infos),
             "no right-edge training")
    _require(any("right_edge" in info["test_kinds"] for info in infos),
             "no source-like test")
    _require(any(4 not in info["train_counts"] for info in infos),
             "no edge-only training")
    _require(any(len(set(info["local_colors"])) > 1 for info in infos),
             "no local recoloring")
    _require(any(len(set(info["local_colors"])) == 1 for info in infos),
             "no stable local palette")
    _require(any(len(set(info["dimensions"])) == 1 for info in infos),
             "no fixed canvas puzzles")
    _require(any(2 in shape for shape in all_dimensions), "no two-cell side")
    _require((3, 5) in all_dimensions, "the source-sized canvas is never sampled")
    _require(any(4 not in info["train_counts"] and 4 in info["test_counts"]
                 for info in infos), "no edge-only training with an interior test")
    _require(any(info["train_kinds"][0] != "interior" for info in infos),
             "training always starts with an interior example")
    _require(any(info["test_kinds"][0] not in info["train_kinds"] for info in infos),
             "tests never hold out a position kind")
    _require(any("right_edge" in info["train_kinds"] and
                 any(kind == "left_edge" or kind in _CORNERS
                     for kind in info["test_kinds"]) for info in infos),
             "no rotated right-edge demonstration")
    _require(any(set(info["dimensions"][len(puzzle["train"]):]) &
                 set(info["dimensions"][:len(puzzle["train"])])
                 for puzzle, info in zip(puzzles, infos)),
             "test canvases never match training dimensions")
    _require(all(len(puzzle["test"]) == 1 or len(set(info["test_counts"])) >= 2
                 for puzzle, info in zip(puzzles, infos)),
             "a multi-test puzzle repeats one clipping outcome")
    for puzzle in puzzles:
        for pair in puzzle["test"]:
            _require(np.array_equal(_infer(puzzle["train"], pair["input"]), pair["output"]),
                     "training-derived test prediction differs")
    for seed, nt, ns in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (1004, 7, 3)):
        puzzle = generate_puzzle(seed, nt, ns)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, ns),
                 "explicit counts ignored")
        validate_generated_puzzle(puzzle)
    for seed, nt, ns in ((2001, None, 1), (2002, None, 4),
                         (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, nt, ns)
        _require((nt is None or len(puzzle["train"]) == nt) and
                 (ns is None or len(puzzle["test"]) == ns), "partial count ignored")
        validate_generated_puzzle(puzzle)
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4},
                    {"num_train": 2.5}, {"num_test": True}):
        try:
            generate_puzzle(4001, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {request}")
    for seed, nt, ns in ((3001, None, None), (3002, 8, 3)):
        first = generate_puzzle(seed, nt, ns)
        second = generate_puzzle(seed, nt, ns)
        _require(all(np.array_equal(a[role], b[role])
                     for split in ("train", "test")
                     for a, b in zip(first[split], second[split])
                     for role in ("input", "output")), "nondeterministic puzzle")
