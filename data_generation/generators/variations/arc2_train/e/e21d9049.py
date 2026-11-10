"""Generate complete two-axis cyclic extension puzzles."""

import numpy as np


_SOURCE_INPUT_ROWS = (
    ("00000000000000", "00000000000000", "00000000000000",
     "00000000000000", "00000000400000", "00000000800000",
     "00000002384000", "00000000200000") + ("00000000000000",) * 7,
    ("00000000000", "00000000000", "00800000000", "00300000000",
     "83200000000") + ("00000000000",) * 7,
    ("00000000000000000000",) * 5 +
    ("00000040000000000000", "00000423100000000000",
     "00000030000000000000", "00000010000000000000") +
    ("00000000000000000000",) * 11,
)
_SOURCE_INPUTS = tuple(
    np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    for rows in _SOURCE_INPUT_ROWS
)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _pick(rng, values):
    return values[int(rng.integers(0, len(values)))]


def _palette(rng, background, period, reference=None, overlap="free"):
    available = [c for c in range(10) if c != background]
    if reference is None or overlap == "free":
        return tuple(int(x) for x in rng.choice(available, size=period, replace=False))
    reference = set(reference) & set(available)
    outside = set(available) - reference
    if overlap == "exact" and len(reference) == period:
        return tuple(sorted(reference))
    minimum = max(0, period - len(outside))
    maximum = min(period, len(reference))
    if overlap == "little":
        matching = min(maximum, max(minimum, int(rng.integers(0, 2))))
    else:
        lower = max(minimum, 1)
        upper = min(maximum, period - 1)
        matching = int(rng.integers(lower, upper + 1)) if lower <= upper else minimum
    chosen = list(rng.choice(sorted(reference), size=matching, replace=False)) if matching else []
    chosen += list(rng.choice(sorted(outside), size=period - matching, replace=False))
    return tuple(int(x) for x in chosen)


def _indices(rng, horizontal_period, vertical_period, kind):
    ends_h = (0, horizontal_period - 1)
    ends_v = (0, vertical_period - 1)
    inside_h = tuple(range(1, horizontal_period - 1))
    inside_v = tuple(range(1, vertical_period - 1))
    choices = {
        "cross": [(i, j) for i in inside_h for j in inside_v],
        "L": [(i, j) for i in ends_h for j in ends_v],
        "T": ([(i, j) for i in ends_h for j in inside_v]
              + [(i, j) for i in inside_h for j in ends_v]),
    }
    if kind == "same":
        choices = {shape: [(i, j) for i, j in pairs if i == j]
                   for shape, pairs in choices.items()}
    elif kind == "rotation":
        choices = {shape: [(i, j) for i, j in pairs if i != j]
                   for shape, pairs in choices.items()}
    shapes = [shape for shape in ("cross", "cross", "L", "L", "T", "T")
              if choices[shape]]
    shape = _pick(rng, shapes)
    return _pick(rng, choices[shape])


def _nonrotational_word(rng, horizontal_word, shared_color, vertical_index):
    period = len(horizontal_word)
    other = [c for c in horizontal_word if c != shared_color]
    other = [int(c) for c in rng.permutation(other)]
    word = other[:vertical_index] + [shared_color] + other[vertical_index:]
    rotations = {tuple(horizontal_word[s:] + horizontal_word[:s]) for s in range(period)}
    if tuple(word) in rotations:
        movable = [i for i in range(period) if i != vertical_index]
        word[movable[0]], word[movable[1]] = word[movable[1]], word[movable[0]]
    _require(tuple(word) not in rotations, "independent word became a rotation")
    return tuple(word)


def _dimension(rng, period, regime):
    # Short tails and long extrapolations coexist, independently on the two axes.
    ranges = ((period + 1, min(30, period + 3)),
              (period + 4, min(30, period + 10)),
              (period + 11, 30))
    valid = [bounds for bounds in ranges if bounds[0] <= bounds[1]]
    if regime == "target_test" and len(valid) == 3:
        valid.append(valid[-1])
    low, high = _pick(rng, valid)
    return int(rng.integers(low, high + 1))


def _start(rng, length, period):
    last = length - period
    choice = int(rng.integers(0, 5))
    if choice == 0:
        return 0
    if choice == 1:
        return last
    return int(rng.integers(0, last + 1))


def _construct_output(height, width, background, row, column,
                      horizontal_start, vertical_start,
                      horizontal_word, vertical_word):
    output = np.full((height, width), background, dtype=np.int8)
    for x in range(width):
        output[row, x] = horizontal_word[(x - horizontal_start) % len(horizontal_word)]
    for y in range(height):
        output[y, column] = vertical_word[(y - vertical_start) % len(vertical_word)]
    return output


def _generate_pair(rng, regime, background, kind, reference=None,
                   overlap="free", fixed_period=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    if fixed_period is not None:
        horizontal_period = fixed_period
    elif overlap == "exact" and reference is not None:
        horizontal_period = len(reference)
    else:
        horizontal_period = int(rng.integers(3, 9))
    vertical_period = horizontal_period
    if kind == "unequal":
        vertical_period = _pick(rng, [p for p in range(3, 9) if p != horizontal_period])
    horizontal_index, vertical_index = _indices(
        rng, horizontal_period, vertical_period, kind)
    horizontal_palette = _palette(rng, background, horizontal_period, reference, overlap)
    horizontal_word = tuple(int(x) for x in rng.permutation(horizontal_palette))
    shared_color = horizontal_word[horizontal_index]
    if kind == "same":
        vertical_word = horizontal_word
    elif kind == "rotation":
        offset = (horizontal_index - vertical_index) % horizontal_period
        vertical_word = tuple(horizontal_word[(i + offset) % horizontal_period]
                              for i in range(horizontal_period))
    elif kind == "permuted":
        vertical_word = _nonrotational_word(rng, horizontal_word, shared_color,
                                           vertical_index)
    elif kind == "unequal":
        horizontal_others = set(horizontal_palette) - {shared_color}
        outside = set(range(10)) - set(horizontal_palette) - {background}
        lowest_overlap = max(1, horizontal_period + vertical_period - 9)
        highest_overlap = min(horizontal_period - 1, vertical_period - 1)
        overlap_count = int(rng.integers(lowest_overlap, highest_overlap + 1))
        shared_others = list(rng.choice(sorted(horizontal_others),
                                        size=overlap_count - 1, replace=False))
        new_colors = list(rng.choice(sorted(outside),
                                         size=vertical_period - overlap_count,
                                         replace=False))
        others = [int(c) for c in rng.permutation(shared_others + new_colors)]
        vertical_word = tuple(others[:vertical_index] + [shared_color]
                              + others[vertical_index:])
    else:
        raise ValueError("unknown word relationship")
    if int(rng.integers(0, 5)) == 0:
        height = width = _dimension(rng, max(horizontal_period, vertical_period), regime)
    else:
        height = _dimension(rng, vertical_period, regime)
        width = _dimension(rng, horizontal_period, regime)
    horizontal_start = _start(rng, width, horizontal_period)
    vertical_start = _start(rng, height, vertical_period)
    row = vertical_start + vertical_index
    column = horizontal_start + horizontal_index
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[row, horizontal_start:horizontal_start + horizontal_period] = horizontal_word
    input_grid[vertical_start:vertical_start + vertical_period, column] = vertical_word
    output_grid = _construct_output(height, width, background, row, column,
                                    horizontal_start, vertical_start,
                                    horizontal_word, vertical_word)
    return {"input": input_grid, "output": output_grid}


def _array_key(grid):
    return (grid.shape, grid.tobytes())


def _analyze_pair(input_grid, output_grid):
    _require(input_grid.shape == output_grid.shape, "dimensions differ")
    height, width = input_grid.shape
    counts = np.bincount(input_grid.reshape(-1), minlength=10)
    _require(np.count_nonzero(counts == counts.max()) == 1,
             "background is not uniquely identifiable")
    background = int(np.argmax(counts))
    mask = input_grid != background
    rows = np.flatnonzero(mask.sum(axis=1) > 1)
    columns = np.flatnonzero(mask.sum(axis=0) > 1)
    _require(len(rows) == len(columns) == 1, "expected one row and one column")
    row, column = int(rows[0]), int(columns[0])
    horizontal_cells = np.flatnonzero(mask[row])
    vertical_cells = np.flatnonzero(mask[:, column])
    hp, vp = len(horizontal_cells), len(vertical_cells)
    _require(3 <= hp <= 8 and 3 <= vp <= 8, "invalid cycle length")
    hs, vs = int(horizontal_cells[0]), int(vertical_cells[0])
    _require(np.array_equal(horizontal_cells, np.arange(hs, hs + hp)) and
             np.array_equal(vertical_cells, np.arange(vs, vs + vp)),
             "fragments are not contiguous")
    expected_mask = np.zeros_like(mask)
    expected_mask[row, hs:hs + hp] = True
    expected_mask[vs:vs + vp, column] = True
    _require(np.array_equal(mask, expected_mask), "input has extra signal cells")
    horizontal_word = tuple(int(c) for c in input_grid[row, hs:hs + hp])
    vertical_word = tuple(int(c) for c in input_grid[vs:vs + vp, column])
    _require(len(set(horizontal_word)) == hp and len(set(vertical_word)) == vp,
             "each visible word must be a primitive cycle of distinct colors")
    hi, vi = column - hs, row - vs
    _require(horizontal_word[hi] == vertical_word[vi], "intersection disagrees")
    expected = _construct_output(height, width, background, row, column,
                                 hs, vs, horizontal_word, vertical_word)
    _require(np.array_equal(expected, output_grid), "wrong periodic extension")
    _require(not np.array_equal(input_grid, output_grid), "no extension visible")
    _require(np.all(output_grid[~((np.indices(mask.shape)[0] == row) |
                                   (np.indices(mask.shape)[1] == column))] == background),
             "off-axis cells changed")
    rotations = {horizontal_word[s:] + horizontal_word[:s] for s in range(hp)}
    if hp != vp:
        kind = "unequal"
    elif horizontal_word == vertical_word:
        kind = "same"
    elif vertical_word in rotations:
        kind = "rotation"
    else:
        kind = "permuted"
    shape = "L" if hi in (0, hp - 1) and vi in (0, vp - 1) else (
        "cross" if 0 < hi < hp - 1 and 0 < vi < vp - 1 else "T")
    return {"background": background, "height": height, "width": width,
            "horizontal_period": hp, "vertical_period": vp,
            "horizontal_row": row, "vertical_column": column,
            "horizontal_start": hs, "vertical_start": vs,
            "horizontal_index": hi, "vertical_index": vi,
            "horizontal_word": horizontal_word, "vertical_word": vertical_word,
            "horizontal_palette": frozenset(horizontal_word),
            "vertical_palette": frozenset(vertical_word),
            "kind": kind, "shape": shape}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    input_keys = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be two-dimensional arrays")
                _require(all(1 <= dim <= 30 for dim in grid.shape), "grid size invalid")
                _require(np.issubdtype(grid.dtype, np.integer) and
                         bool(np.all((grid >= 0) & (grid <= 9))), "invalid ARC colors")
            key = _array_key(pair["input"])
            _require(key not in input_keys, "duplicate input")
            input_keys.add(key)
            _require(not any(pair["input"].shape == original.shape and
                             np.array_equal(pair["input"], original)
                             for original in _SOURCE_INPUTS), "source input copied")
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))
    training = analyses["train"]
    _require(any(a["kind"] in ("permuted", "unequal") for a in training),
             "training must demonstrate independent axis readings")
    if any(a["kind"] == "unequal" for a in analyses["test"]):
        _require(any(a["kind"] == "unequal" for a in training),
                 "unequal periods in test require a training example")
    if len({a["background"] for a in analyses["test"] + training}) > 1:
        _require(len({a["background"] for a in training}) > 1,
                 "background changes need a training demonstration")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # The only global conventions are the fixed reading directions and whether
    # a puzzle varies its easily observed background between pairs.
    varying_background = bool(rng.integers(0, 2))
    shared_background = int(rng.integers(0, 10))
    fixed_period = int(rng.integers(3, 9)) if int(rng.integers(0, 5)) == 0 else None
    demonstrate_unequal = fixed_period is None and bool(rng.integers(0, 4) == 0)
    mandatory = ["permuted"] + (["unequal"] if demonstrate_unequal else [])
    train_kinds = mandatory + [
        _pick(rng, ["same", "same", "rotation", "permuted", "permuted"]
              + (["unequal"] if demonstrate_unequal else []))
        for _ in range(num_train - len(mandatory))
    ]
    train_kinds = [train_kinds[int(i)] for i in rng.permutation(num_train)]
    puzzle = {"train": [], "test": []}
    used = set()
    training_backgrounds = []
    for split, kinds in (("train", train_kinds),
                         ("test", [_pick(rng, ["same", "same", "rotation",
                                            "permuted", "permuted"]
                                           + (["unequal"] if demonstrate_unequal else []))
                                   for _ in range(num_test)])):
        for index, kind in enumerate(kinds):
            accepted = None
            for _ in range(128):
                if varying_background:
                    if split == "train" and index == 1:
                        background = _pick(rng, [c for c in range(10)
                                                 if c != training_backgrounds[0]])
                    else:
                        background = int(rng.integers(0, 10))
                else:
                    background = shared_background
                reference = None
                overlap = "free"
                if split == "test" and puzzle["train"]:
                    reference_pair = _pick(rng, puzzle["train"])
                    reference_analysis = _analyze_pair(reference_pair["input"],
                                                       reference_pair["output"])
                    reference = reference_analysis["horizontal_palette"]
                    overlap = _pick(rng, ["exact", "partial", "little", "free"])
                    if overlap == "exact" and kind == "unequal":
                        overlap = "partial"
                    if overlap == "exact" and background in reference:
                        if varying_background:
                            background = _pick(rng, [c for c in range(10)
                                                     if c not in reference])
                        else:
                            overlap = "partial"
                regime = "source_train" if split == "train" else "target_test"
                pair = _generate_pair(rng, regime, background, kind, reference,
                                      overlap, fixed_period)
                if _array_key(pair["input"]) not in used:
                    accepted = pair
                    break
            if accepted is None:
                raise ValueError("unable to construct a distinct input")
            used.add(_array_key(accepted["input"]))
            puzzle[split].append(accepted)
            if split == "train":
                training_backgrounds.append(background)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and
               all(np.array_equal(a[k], b[k])
                   for a, b in zip(first[s], second[s])
                   for k in ("input", "output"))
               for s in ("train", "test"))


def self_test() -> None:
    a = generate_puzzle(101)
    b = generate_puzzle(101)
    _require(_puzzles_equal(a, b), "generation is not deterministic")
    for seed, counts in ((202, (2, 1)), (203, (10, 4)), (204, (7, 3))):
        puzzle = generate_puzzle(seed, *counts)
        _require((len(puzzle["train"]), len(puzzle["test"])) == counts,
                 "explicit counts not honored")
    for args in ({"num_train": 1}, {"num_test": 5},
                 {"num_train": 2, "num_test": 2}):
        try:
            generate_puzzle(10, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    seen_kinds, seen_shapes, seen_periods, seen_backgrounds = set(), set(), set(), set()
    l_corners = set()
    train_square = test_rectangular = test_twenty = 0
    test_edge = test_short_tail = test_long_extension = 0
    varied_training_background = varied_training_shape = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training_analyses = [_analyze_pair(p["input"], p["output"])
                             for p in puzzle["train"]]
        varied_training_background += len({a["background"] for a in training_analyses}) > 1
        varied_training_shape += len({a["shape"] for a in training_analyses}) > 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair["input"], pair["output"])
                seen_kinds.add(a["kind"])
                seen_shapes.add(a["shape"])
                seen_periods.add((a["horizontal_period"], a["vertical_period"]))
                seen_backgrounds.add(a["background"])
                if a["shape"] == "L":
                    l_corners.add((a["horizontal_index"] == 0,
                                   a["vertical_index"] == 0))
                if split == "train":
                    train_square += a["height"] == a["width"]
                else:
                    test_rectangular += a["height"] != a["width"]
                    test_twenty += (a["height"], a["width"]) == (20, 20)
                    test_edge += (a["horizontal_row"] in (0, a["height"] - 1) or
                                  a["vertical_column"] in (0, a["width"] - 1))
                    test_short_tail += (a["width"] - a["horizontal_period"] <= 3 or
                                        a["height"] - a["vertical_period"] <= 3)
                    test_long_extension += (a["width"] - a["horizontal_period"] >= 10 or
                                            a["height"] - a["vertical_period"] >= 10)
    _require(seen_kinds == {"same", "rotation", "permuted", "unequal"},
             "word relationships insufficiently varied")
    _require(seen_shapes == {"cross", "L", "T"}, "intersection shapes insufficiently varied")
    _require(len(seen_periods) >= 12 and len(seen_backgrounds) >= 8,
             "period or background diversity insufficient")
    _require(len(l_corners) == 4 and train_square > 0 and test_rectangular > 0
             and test_twenty > 0, "intersection or dimension diversity insufficient")
    _require(test_edge > 0 and test_short_tail > 0 and test_long_extension > 0,
             "test completion geometry insufficiently varied")
    _require(varied_training_background > 0 and varied_training_shape > 0,
             "within-puzzle diversity insufficient")
