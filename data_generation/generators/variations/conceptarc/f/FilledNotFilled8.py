"""Two-stroke completion and down-left parallelogram-side extraction."""

import numpy as np


_DIRECTIONS = {
    "right_fill": (1, 1),
    "left_fill": (1, -1),
    "horizontal_fill": (0, 1),
    "vertical_fill": (1, 0),
    "left_reduce": (1, -1),
}
_FILL_KINDS = tuple(k for k in _DIRECTIONS if k != "left_reduce")
_SOURCE_SIGNATURES = {
    ("right_fill", 0, 6, 13, 13, 6, 3, 0, 2),
    ("left_reduce", 0, 3, 9, 12, 5, 2, 0, 1),
    ("left_fill", 0, 5, 10, 10, 4, 5, 0, 0),
    ("left_reduce", 0, 7, 15, 11, 7, 3, 5, 0),
    ("horizontal_fill", 0, 3, 11, 14, 8, 5, 3, 2),
}


def _masks(shape, start, direction, length, displacement):
    """Rasterize two lattice sides and their closed convex hull."""
    p = np.array(start, dtype=int)
    u = np.array(direction, dtype=int)
    d = np.array(displacement, dtype=int)
    corners = [p, p + (length - 1) * u,
               p + (length - 1) * u + d, p + d]
    sparse = np.zeros(shape, dtype=bool)
    for i in range(length):
        for origin in (p, p + d):
            r, c = origin + i * u
            sparse[int(r), int(c)] = True
    rr, cc = np.indices(shape)
    filled = np.ones(shape, dtype=bool)
    orientation = int(u[0] * d[1] - u[1] * d[0])
    assert orientation != 0
    for a, b in zip(corners, corners[1:] + corners[:1]):
        cross = (b[0] - a[0]) * (cc - a[1]) - (b[1] - a[1]) * (rr - a[0])
        filled &= orientation * cross >= 0
    return sparse, filled


def _margins(rng, extent):
    first = int(rng.integers(2, min(6, 30 - extent - 2) + 1))
    second = int(rng.integers(2, min(6, 30 - extent - first) + 1))
    return first, second


def _construct(rng, kind, background, foreground, length=None, sheared=None,
               separation=None, lateral=None):
    if length is None:
        length = int(rng.integers(3, 11))
    if separation is None:
        separation = int(rng.integers(2, 9))
    direction = _DIRECTIONS[kind]
    if kind == "left_reduce":
        displacement = (separation, separation)
    elif kind in ("right_fill", "left_fill"):
        if sheared is None:
            sheared = bool(rng.integers(0, 2))
        if lateral is None:
            lateral = separation
            if sheared:
                choices = [x for x in range(max(1, separation - 4),
                                            min(9, separation + 4) + 1)
                           if x != separation]
                lateral = int(rng.choice(choices))
        displacement = (separation, -direction[1] * lateral)
    elif kind == "horizontal_fill":
        displacement = (separation, 0)
    else:
        displacement = (0, separation)
    u = np.array(direction)
    d = np.array(displacement)
    vertices = np.array([(0, 0), (length - 1) * u,
                         (length - 1) * u + d, d])
    low = vertices.min(axis=0)
    extent = vertices.max(axis=0) - low + 1
    top, bottom = _margins(rng, int(extent[0]))
    left, right = _margins(rng, int(extent[1]))
    shape = (int(top + extent[0] + bottom), int(left + extent[1] + right))
    start = (int(top - low[0]), int(left - low[1]))
    sparse, filled = _masks(shape, start, direction, length, displacement)
    assert sparse.sum() < filled.sum() < filled.size / 2
    before, after = (filled, sparse) if kind == "left_reduce" else (sparse, filled)
    return {"input": np.where(before, foreground, background).astype(np.int8),
            "output": np.where(after, foreground, background).astype(np.int8)}


def _sparse_candidate(mask, kind):
    u = _DIRECTIONS[kind]
    points = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if kind == "horizontal_fill":
        key = lambda p: p[0]
        step = lambda p: p[1]
    elif kind == "vertical_fill":
        key = lambda p: p[1]
        step = lambda p: p[0]
    else:
        key = lambda p: p[1] - u[1] * p[0]
        step = lambda p: p[0]
    groups = {}
    for p in points:
        groups.setdefault(key(p), []).append(p)
    if len(groups) != 2:
        return None
    segments = []
    for group in groups.values():
        group.sort(key=step)
        if len(group) < 3:
            return None
        start = group[0]
        if any(p != (start[0] + i * u[0], start[1] + i * u[1])
               for i, p in enumerate(group)):
            return None
        segments.append((start, len(group)))
    segments.sort(key=lambda item: item[0][1] if kind == "vertical_fill"
                  else item[0][0])
    (p, length), (q, other_length) = segments
    dr, dc = q[0] - p[0], q[1] - p[1]
    if length != other_length or length > 10:
        return None
    if kind in ("right_fill", "left_fill"):
        lateral = -u[1] * dc
        if not (2 <= dr <= 8 and 1 <= lateral <= 9 and abs(lateral - dr) <= 4):
            return None
        separation = dr
    elif kind == "horizontal_fill":
        if not (2 <= dr <= 8 and dc == 0):
            return None
        lateral, separation = 0, dr
    else:
        if not (dr == 0 and 2 <= dc <= 8):
            return None
        lateral, separation = 0, dc
    sparse, filled = _masks(mask.shape, p, u, length, (dr, dc))
    if not np.array_equal(mask, sparse) or filled.sum() <= sparse.sum():
        return None
    rows, cols = np.nonzero(sparse)
    return filled, {"kind": kind, "length": length, "separation": separation,
                    "lateral": lateral, "sheared": kind in ("right_fill", "left_fill")
                    and lateral != separation, "top": int(rows.min()),
                    "left": int(cols.min())}


def _reduction_candidate(mask):
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    top = int(rows.min())
    top_cols = np.flatnonzero(mask[top])
    if len(top_cols) != 1:
        return None
    start_col = int(top_cols[0])
    length = start_col - int(cols.min()) + 1
    separation = int(cols.max()) - start_col
    if not (3 <= length <= 10 and 2 <= separation <= 8):
        return None
    if top + separation + length - 1 >= mask.shape[0]:
        return None
    sparse, filled = _masks(mask.shape, (top, start_col), (1, -1), length,
                            (separation, separation))
    if not np.array_equal(mask, filled) or sparse.sum() >= filled.sum():
        return None
    return sparse, {"kind": "left_reduce", "length": length,
                    "separation": separation, "lateral": separation,
                    "sheared": False, "top": top, "left": int(cols.min())}


def _inspect_pair(pair):
    before, after = pair["input"], pair["output"]
    assert before.shape == after.shape
    colors = np.unique(before)
    assert len(colors) == 2
    background = int(before[0, 0])
    assert all(int(before[r, c]) == background and int(after[r, c]) == background
               for r in (0, -1) for c in (0, -1))
    foreground = int(colors[0] if colors[1] == background else colors[1])
    assert set(map(int, np.unique(after))) == {background, foreground}
    mask = before == foreground
    expected = after == foreground
    candidates = []
    for kind in _FILL_KINDS:
        candidate = _sparse_candidate(mask, kind)
        if candidate is not None and np.array_equal(candidate[0], expected):
            candidates.append(candidate[1])
    candidate = _reduction_candidate(mask)
    if candidate is not None and np.array_equal(candidate[0], expected):
        candidates.append(candidate[1])
    assert len(candidates) == 1, "pair has no unique shape transformation"
    return background, foreground, candidates[0]


def _source_signature(pair, analysis):
    background, foreground, info = analysis
    return (info["kind"], background, foreground, *pair["input"].shape,
            info["length"], info["separation"], info["top"], info["left"])


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    # Puzzle-wide policies guide the schedule; the transformation has no hidden key.
    schedule = int(rng.integers(0, 3))
    background_policy = int(rng.integers(0, 3))
    reduction_scale = int(rng.integers(0, 3))
    first_fill = "right_fill" if schedule == 0 else str(rng.choice(_FILL_KINDS))
    first_shear = (first_fill in ("right_fill", "left_fill")
                   and schedule != 0 and bool(rng.integers(0, 2)))
    train_specs = [(first_fill, first_shear), ("left_reduce", False)]
    if num_train >= 4:
        alternatives = [k for k in _FILL_KINDS if k != first_fill]
        rng.shuffle(alternatives)
        second = alternatives[0]
        train_specs.append((second, second in ("right_fill", "left_fill")
                            and bool(rng.integers(0, 2))))
    while len(train_specs) < num_train:
        kind = str(rng.choice((*_FILL_KINDS, "left_reduce")))
        shear = kind in ("right_fill", "left_fill") and bool(rng.integers(0, 2))
        train_specs.append((kind, shear))
    rng.shuffle(train_specs)

    puzzle = {"train": [], "test": []}
    seen_inputs, seen_shapes = set(), set()
    train_foregrounds = set()
    train_reduction_lengths = []
    train_vertical = any(k == "vertical_fill" for k, _ in train_specs)
    train_shear = any(k in ("right_fill", "left_fill") and s for k, s in train_specs)

    def colors(split, index):
        if background_policy == 0:
            background = 0
        elif background_policy == 1 and split == "train" and index == 0:
            background = 0
        elif background_policy == 1 and split == "train" and index == 1:
            background = int(rng.integers(1, 10))
        elif background_policy == 1 and bool(rng.integers(0, 2)):
            background = 0
        else:
            background = int(rng.integers(1, 10))
        available = [c for c in range(10) if c != background]
        if split == "test" and bool(rng.integers(0, 2)):
            new = [c for c in available if c not in train_foregrounds]
            if new:
                available = new
        return background, int(rng.choice(available))

    def add(split, kind, shear, length=None):
        index = len(puzzle[split])
        background, foreground = colors(split, index)
        for _ in range(100):
            child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _construct(child, kind, background, foreground, length, shear)
            analysis = _inspect_pair(pair)
            info = analysis[2]
            shape_key = (info["kind"], info["length"], info["separation"],
                         info["lateral"])
            input_key = (pair["input"].shape, pair["input"].tobytes())
            if (shape_key in seen_shapes or input_key in seen_inputs
                    or _source_signature(pair, analysis) in _SOURCE_SIGNATURES):
                continue
            puzzle[split].append(pair)
            seen_shapes.add(shape_key)
            seen_inputs.add(input_key)
            if split == "train":
                train_foregrounds.add(foreground)
                if kind == "left_reduce":
                    train_reduction_lengths.append(info["length"])
            return
        # Exhaust the finite geometry space before declaring a legal request impossible.
        lengths = ([length] + [x for x in range(3, 11) if x != length]
                   if length is not None else list(range(3, 11)))
        alternatives = []
        for candidate_length in lengths:
            for separation in range(2, 9):
                laterals = ([x for x in range(max(1, separation - 4),
                                               min(9, separation + 4) + 1)
                             if x != separation] if shear else
                            [separation if kind in ("right_fill", "left_fill", "left_reduce")
                             else 0])
                for lateral in laterals:
                    signature = (kind, candidate_length, separation, lateral)
                    if signature not in seen_shapes:
                        alternatives.append((candidate_length, separation, lateral))
        rng.shuffle(alternatives)
        for candidate_length, separation, lateral in alternatives:
            child = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            pair = _construct(child, kind, background, foreground,
                              candidate_length, shear, separation, lateral)
            analysis = _inspect_pair(pair)
            info = analysis[2]
            input_key = (pair["input"].shape, pair["input"].tobytes())
            if (input_key in seen_inputs or
                    _source_signature(pair, analysis) in _SOURCE_SIGNATURES):
                continue
            puzzle[split].append(pair)
            seen_shapes.add((kind, candidate_length, separation, lateral))
            seen_inputs.add(input_key)
            if split == "train":
                train_foregrounds.add(foreground)
                if kind == "left_reduce":
                    train_reduction_lengths.append(candidate_length)
            return
        raise RuntimeError("No unused geometry remains")

    first_reduction = True
    for kind, shear in train_specs:
        length = int(rng.integers(4, 8)) if kind == "left_reduce" and first_reduction else None
        add("train", kind, shear, length)
        if kind == "left_reduce":
            first_reduction = False

    test_specs = []
    if num_test >= 2 and bool(rng.integers(0, 4) != 0):
        test_specs.append(("left_reduce", False))
    if train_vertical and len(test_specs) < num_test and bool(rng.integers(0, 2)):
        test_specs.append(("vertical_fill", False))
    while len(test_specs) < num_test:
        options = ["right_fill", "left_fill", "horizontal_fill", "left_reduce"]
        if train_vertical:
            options.append("vertical_fill")
        kind = str(rng.choice(options))
        shear = kind in ("right_fill", "left_fill") and train_shear and bool(rng.integers(0, 2))
        test_specs.append((kind, shear))
    rng.shuffle(test_specs)
    maximum = max(train_reduction_lengths)
    for kind, shear in test_specs:
        length = None
        if kind == "left_reduce":
            if reduction_scale == 0:
                choices = list(range(3, maximum))
            elif reduction_scale == 1:
                choices = [maximum]
            else:
                choices = list(range(maximum + 1, 11))
            if not choices:
                choices = list(range(3, 11))
            length = int(rng.choice(choices))
        add("test", kind, shear, length)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, silhouettes = set(), set(), set()
    descriptions = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            before, after = pair["input"], pair["output"]
            assert before.shape == after.shape and not np.array_equal(before, after)
            input_key = (before.shape, before.tobytes())
            pair_key = (input_key, after.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analysis = _inspect_pair(pair)
            assert _source_signature(pair, analysis) not in _SOURCE_SIGNATURES
            info = analysis[2]
            silhouette = (info["kind"], info["length"], info["separation"],
                          info["lateral"])
            assert silhouette not in silhouettes
            silhouettes.add(silhouette)
            descriptions[split].append(info)
    train = descriptions["train"]
    test = descriptions["test"]
    assert any(d["kind"] != "left_reduce" for d in train)
    assert any(d["kind"] == "left_reduce" for d in train)
    if any(d["kind"] == "vertical_fill" for d in test):
        assert any(d["kind"] == "vertical_fill" for d in train)
    if any(d["sheared"] for d in test):
        assert any(d["sheared"] for d in train)


def _puzzle_key(puzzle):
    return tuple(tuple((p["input"].shape, p["input"].tobytes(),
                        p["output"].tobytes()) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles, train_counts, test_counts = set(), set(), set()
    kinds, shapes, silhouettes, backgrounds, foregrounds = set(), set(), set(), set(), set()
    novel_tests = vertical_tests = sheared_tests = 0
    reductions = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        key = _puzzle_key(puzzle)
        assert key == _puzzle_key(generate_puzzle(seed))
        puzzles.add(key)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_colors = {_inspect_pair(p)[1] for p in puzzle["train"]}
        train_lengths = [_inspect_pair(p)[2]["length"] for p in puzzle["train"]
                         if _inspect_pair(p)[2]["kind"] == "left_reduce"]
        for split in ("train", "test"):
            for pair in puzzle[split]:
                bg, fg, info = _inspect_pair(pair)
                kinds.add(info["kind"])
                shapes.add(pair["input"].shape)
                silhouettes.add((info["kind"], info["length"], info["separation"],
                                 info["lateral"]))
                backgrounds.add(bg)
                foregrounds.add(fg)
                if split == "test":
                    novel_tests += fg not in train_colors
                    vertical_tests += info["kind"] == "vertical_fill"
                    sheared_tests += info["sheared"]
                    if info["kind"] == "left_reduce":
                        maximum = max(train_lengths)
                        reductions.add("smaller" if info["length"] < maximum else
                                       "equal" if info["length"] == maximum else "larger")
    assert len(puzzles) == 42 and len(train_counts) >= 4 and len(test_counts) >= 3
    assert kinds == set(_DIRECTIONS) and len(shapes) >= 35 and len(silhouettes) >= 75
    assert len(backgrounds) >= 5 and len(foregrounds) >= 8
    assert novel_tests >= 5 and vertical_tests >= 1 and sheared_tests >= 1
    assert reductions == {"smaller", "equal", "larger"}
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                        (2, None), (10, None), (5, 3)):
        puzzle = generate_puzzle(12345, train, test)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(12345, train, test))
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (2, 4), (True, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count was accepted")
