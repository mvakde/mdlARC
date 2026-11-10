"""Whole-puzzle generator for counting cells inside a hollow rectangle."""

import numpy as np

_SOURCE_INPUTS = (
    ("0000000", "0000000", "0333000", "0303000", "0333000", "0000000", "0000000"),
    ("0000000", "0333300", "0300300", "0300300", "0333300", "0000000", "0000000"),
    ("0000000", "0333330", "0300030", "0300030", "0333330", "0000000", "0000000"),
    ("0333000", "0303000", "0303000", "0303000", "0303000", "0333000", "0000000"),
    ("0000000", "0333330", "0300030", "0333330", "0000000", "0000000", "0000000"),
    ("0333300", "0300300", "0300300", "0300300", "0300300", "0333300", "0000000"),
)
_TRAIN_SHAPES = tuple((h, w) for h in range(1, 21) for w in range(1, 21)
                      if h * w <= 20)
_TEST_SHAPES = tuple((h, w) for h in range(1, 27) for w in range(1, 27)
                     if h * w <= 30)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= value <= high):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None:
        lower = 2 if num_test is None else max(2, int(num_test) + 1)
        num_train = int(rng.integers(lower, 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, int(num_train) - 1) + 1))
    return int(num_train), int(num_test)


def _sample_colors(rng):
    background = int(rng.integers(10))
    frame = int(rng.integers(9))
    if frame >= background:
        frame += 1
    return background, frame


def _shape_category(shape):
    if shape[0] == shape[1]:
        return "square"
    return "strip" if min(shape) == 1 else "rectangle"


def _sample_shape(rng, regime, excluded=(), minimum_area=1):
    pool = _TRAIN_SHAPES if regime == "source_train" else _TEST_SHAPES
    candidates = [s for s in pool if s not in excluded
                  and s[0] * s[1] >= minimum_area]
    if not candidates:
        raise ValueError("No eligible interior shape")
    categories = ("source", "square", "strip", "rectangle", "all")
    probabilities = ((0.18, 0.13, 0.24, 0.32, 0.13)
                     if regime == "source_train" else
                     (0.12, 0.12, 0.28, 0.35, 0.13))
    category = categories[int(rng.choice(len(categories), p=probabilities))]
    if category == "source":
        preferred = [s for s in candidates if max(s) <= 3]
    elif category == "all":
        preferred = candidates
    else:
        preferred = [s for s in candidates if _shape_category(s) == category]
    choices = preferred or candidates
    return choices[int(rng.integers(len(choices)))]


def _training_shapes(rng, count):
    # Two output lengths and, when possible, equal-area unlike rectangles.
    small = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 3))
    shapes = [small[int(rng.integers(len(small)))]]
    area = (6, 8, 10, 12, 15, 18, 20)[int(rng.integers(7))]
    factors = [(h, area // h) for h in range(2, int(area ** 0.5) + 1)
               if area % h == 0]
    first = factors[int(rng.integers(len(factors)))]
    alternatives = [s for s in [(1, area)] + factors if s != first]
    second = alternatives[int(rng.integers(len(alternatives)))]
    for shape in (first, second):
        shapes.append(shape[::-1] if rng.integers(2) else shape)
    shapes = shapes[:count]
    while len(shapes) < count:
        shapes.append(_sample_shape(rng, "source_train", shapes))
    rng.shuffle(shapes)
    return shapes


def _test_shapes(rng, count, train_shapes):
    shapes = []
    train_max = max(h * w for h, w in train_shapes)
    if rng.random() < 0.72:
        shapes.append(_sample_shape(rng, "target_test", minimum_area=train_max + 1))
    while len(shapes) < count:
        if rng.random() < 0.27:
            shared = [s for s in train_shapes if s not in shapes]
            if shared:
                shapes.append(shared[int(rng.integers(len(shared)))])
                continue
        shapes.append(_sample_shape(rng, "target_test", shapes))
    rng.shuffle(shapes)
    return shapes


def _padding_total(rng, capacity, minimum=0):
    if capacity == minimum:
        return minimum
    bands = ((minimum, min(capacity, 2)),
             (max(minimum, 3), min(capacity, 9)),
             (max(minimum, 10), capacity))
    available = [(lo, hi) for lo, hi in bands if lo <= hi]
    lo, hi = available[int(rng.integers(len(available)))]
    return int(rng.integers(lo, hi + 1))


def _axis_margins(rng, frame_size, start_zero=False, end_zero=False,
                  both_positive=False):
    if start_zero and end_zero:
        return 0, 0
    total = _padding_total(rng, 30 - frame_size,
                           minimum=2 if both_positive else 0)
    if start_zero:
        return 0, total
    if end_zero:
        return total, 0
    lo = 1 if both_positive else 0
    before = int(rng.integers(lo, total - lo + 1))
    return before, total - before


def _sample_margins(rng, shape, mode):
    fh, fw = shape[0] + 2, shape[1] + 2
    if mode == "tight":
        return 0, 0, 0, 0
    if mode == "interior":
        top, bottom = _axis_margins(rng, fh, both_positive=True)
        left, right = _axis_margins(rng, fw, both_positive=True)
    elif mode == "edge":
        side = int(rng.integers(4))
        top, bottom = _axis_margins(rng, fh, side == 0, side == 1)
        left, right = _axis_margins(rng, fw, side == 2, side == 3)
    elif mode == "corner":
        vertical, horizontal = int(rng.integers(2)), int(rng.integers(2))
        top, bottom = _axis_margins(rng, fh, vertical == 0, vertical == 1)
        left, right = _axis_margins(rng, fw, horizontal == 0, horizontal == 1)
    else:
        top, bottom = _axis_margins(rng, fh)
        left, right = _axis_margins(rng, fw)
    return top, bottom, left, right


def _position_modes(rng, count, regime):
    modes = ["interior", "edge"] if regime == "source_train" else []
    choices = ("tight", "edge", "corner", "interior", "asymmetric")
    weights = (0.10, 0.27, 0.18, 0.27, 0.18)
    while len(modes) < count:
        modes.append(choices[int(rng.choice(len(choices), p=weights))])
    rng.shuffle(modes)
    return modes


def _make_pair(rng, shape, colors, mode):
    height, width = shape
    background, frame = colors
    top, bottom, left, right = _sample_margins(rng, shape, mode)
    fh, fw = height + 2, width + 2
    grid = np.full((top + fh + bottom, left + fw + right),
                   background, dtype=np.int8)
    grid[top:top + fh, left] = frame
    grid[top:top + fh, left + fw - 1] = frame
    grid[top, left:left + fw] = frame
    grid[top + fh - 1, left:left + fw] = frame
    return {"input": grid,
            "output": np.full((1, height * width), background, dtype=np.int8)}


def _grid_key(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _is_source_input(grid):
    if grid.shape != (7, 7):
        return False
    return tuple("".join(str(int(v)) for v in row) for row in grid) in _SOURCE_INPUTS


def _infer_input(grid):
    colors = np.unique(grid)
    _require(len(colors) == 2, "Input must contain exactly two colors")
    matches = []
    for candidate in colors:
        rows, columns = np.where(grid == candidate)
        top, bottom = int(rows.min()), int(rows.max())
        left, right = int(columns.min()), int(columns.max())
        if bottom - top < 2 or right - left < 2:
            continue
        ring = np.zeros(grid.shape, dtype=bool)
        ring[top:bottom + 1, left] = True
        ring[top:bottom + 1, right] = True
        ring[top, left:right + 1] = True
        ring[bottom, left:right + 1] = True
        if np.array_equal(grid == candidate, ring):
            background = int(colors[0] if colors[1] == candidate else colors[1])
            matches.append((background, int(candidate),
                            (bottom - top - 1, right - left - 1),
                            (top, bottom, left, right)))
    _require(len(matches) == 1,
             "Input must contain exactly one complete hollow frame")
    return matches[0]


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "Pair must contain exactly input and output")
    for grid in (pair["input"], pair["output"]):
        _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                 "Grids must be two-dimensional NumPy arrays")
        _require(all(1 <= size <= 30 for size in grid.shape),
                 "Grid dimensions must be from 1 through 30")
        _require(np.issubdtype(grid.dtype, np.integer),
                 "Grid dtype must be integer")
        _require(bool(np.all((grid >= 0) & (grid <= 9))),
                 "Grid colors must be from 0 through 9")
    background, frame, shape, bounds = _infer_input(pair["input"])
    _require(pair["output"].shape == (1, shape[0] * shape[1])
             and bool(np.all(pair["output"] == background)),
             "Output must be one background-color row of enclosed area")
    return background, frame, shape, bounds


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list)
             and isinstance(puzzle["test"], list),
             "Train and test must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns,
             "Illegal pair counts")
    seen_inputs, train_areas, train_shapes = set(), set(), []
    train_colors, test_colors = [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            background, frame, shape, _ = _analyze_pair(pair)
            key = _grid_key(pair["input"])
            _require(key not in seen_inputs, "Duplicate input")
            seen_inputs.add(key)
            _require(not _is_source_input(pair["input"]),
                     "Generated pair copies a source pair")
            if split == "train":
                _require(shape in _TRAIN_SHAPES,
                         "Training interior exceeds the supported regime")
                train_shapes.append(shape)
                train_areas.add(shape[0] * shape[1])
                train_colors.append((background, frame))
            else:
                _require(shape in _TEST_SHAPES,
                         "Test interior exceeds the supported regime")
                test_colors.append((background, frame))
    _require(len(train_areas) >= 2,
             "Training must show two different enclosed areas")
    if nt >= 3:
        equal_area = any(a[0] * a[1] == b[0] * b[1]
                         and sorted(a) != sorted(b)
                         for i, a in enumerate(train_shapes)
                         for b in train_shapes[i + 1:])
        _require(equal_area,
                 "Training must show unlike factorizations of one area")
    if any(colors != train_colors[0] for colors in test_colors):
        role_swap = any((frame, background) in train_colors
                        for background, frame in train_colors)
        _require(role_swap,
                 "Training must demonstrate changing colors and a role swap")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    base_colors = _sample_colors(rng)
    varied_colors = bool(rng.random() < 0.60)
    train_shapes = _training_shapes(rng, num_train)
    test_shapes = _test_shapes(rng, num_test, train_shapes)
    train_positions = _position_modes(rng, num_train, "source_train")
    test_positions = _position_modes(rng, num_test, "target_test")
    train_colors = [base_colors] * num_train
    test_colors = [base_colors] * num_test
    if varied_colors:
        train_colors = [base_colors, base_colors[::-1]]
        train_colors.extend(_sample_colors(rng) for _ in range(num_train - 2))
        rng.shuffle(train_colors)
        test_colors = [_sample_colors(rng) for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, shapes, colors, modes in (
        ("train", train_shapes, train_colors, train_positions),
        ("test", test_shapes, test_colors, test_positions),
    ):
        for shape, palette, mode in zip(shapes, colors, modes):
            fallback_modes = {"tight": "edge", "edge": "corner",
                              "corner": "edge", "interior": "asymmetric",
                              "asymmetric": "interior"}
            for attempt in range(128):
                pair_rng = np.random.default_rng(
                    int(rng.integers(0, 2**63, dtype=np.int64)))
                chosen_mode = mode if attempt < 8 else fallback_modes[mode]
                pair = _make_pair(pair_rng, shape, palette, chosen_mode)
                key = _grid_key(pair["input"])
                if key not in seen_inputs and not _is_source_input(pair["input"]):
                    seen_inputs.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not construct a unique frame input")
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, sizes, colors = set(), set(), set(), set(), set()
    touched_train, touched_test, signatures = set(), set(), set()
    changed_palettes = bigger_tests = equal_area_puzzles = 0
    first_test_areas = set()
    shape_canvas_repeats = 0
    source_palette_pairs = 0
    combined_variations = 0
    observed = {split: {key: 0 for key in
                         ("horizontal_strip", "vertical_strip", "square",
                          "large_frame", "tight", "corner", "interior")}
                for split in ("train", "test")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _signature(puzzle)
        assert signature == _signature(generate_puzzle(seed))
        signatures.add(signature)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        training = [_infer_input(p["input"]) for p in puzzle["train"]]
        assert all(p["output"].shape == (1, info[2][0] * info[2][1])
                   and np.all(p["output"] == info[0])
                   for p, info in zip(puzzle["train"], training))
        max_train_area = max(s[0] * s[1] for _, _, s, _ in training)
        equal_area_puzzles += len(puzzle["train"]) >= 3
        palettes = {(bg, fg) for bg, fg, _, _ in training}
        changed_palettes += len(palettes) > 1
        first_shape = _infer_input(puzzle["test"][0]["input"])[2]
        first_test_areas.add(first_shape[0] * first_shape[1])
        canvases_by_shape = {}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                bg, fg, shape, (top, bottom, left, right) = _infer_input(pair["input"])
                prediction = np.full((1, shape[0] * shape[1]), bg, dtype=np.int8)
                assert np.array_equal(prediction, pair["output"])
                assert not _is_source_input(pair["input"])
                shapes.add(shape)
                sizes.add(pair["input"].shape)
                colors.add((bg, fg))
                source_palette_pairs += (bg, fg) == (0, 3)
                canvases_by_shape.setdefault(shape, set()).add(pair["input"].shape)
                metrics = observed[split]
                metrics["horizontal_strip"] += shape[0] == 1 and shape[1] >= 6
                metrics["vertical_strip"] += shape[1] == 1 and shape[0] >= 6
                metrics["square"] += shape[0] == shape[1] and shape[0] >= 2
                metrics["large_frame"] += shape[0] + 2 > 7 or shape[1] + 2 > 6
                margins = (top, pair["input"].shape[0] - bottom - 1,
                           left, pair["input"].shape[1] - right - 1)
                metrics["tight"] += all(value == 0 for value in margins)
                metrics["interior"] += all(value > 0 for value in margins)
                metrics["corner"] += ((margins[0] == 0 or margins[1] == 0)
                                      and (margins[2] == 0 or margins[3] == 0))
                touched = touched_train if split == "train" else touched_test
                if top == 0:
                    touched.add("top")
                if bottom == pair["input"].shape[0] - 1:
                    touched.add("bottom")
                if left == 0:
                    touched.add("left")
                if right == pair["input"].shape[1] - 1:
                    touched.add("right")
                if split == "test" and shape[0] * shape[1] > max_train_area:
                    bigger_tests += 1
                if (split == "test"
                        and (shape[0] + 2 > 7 or shape[1] + 2 > 6)
                        and 0 in margins and len(palettes) > 1):
                    combined_variations += 1
        shape_canvas_repeats += any(len(canvas_set) > 1
                                    for canvas_set in canvases_by_shape.values())
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(sizes) >= 50 and len(colors) >= 25
    assert touched_train == touched_test == {"top", "bottom", "left", "right"}
    assert changed_palettes >= 12 and bigger_tests >= 12 and equal_area_puzzles >= 15
    assert len(first_test_areas) >= 10 and shape_canvas_repeats >= 8
    assert source_palette_pairs >= 1 and combined_variations >= 5
    for split in ("train", "test"):
        assert all(observed[split][key] >= 3 for key in observed[split])
    for tr, te in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(101, tr, te)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (tr, te)
        assert _signature(puzzle) == _signature(generate_puzzle(101, tr, te))
    for tr, te in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(202, tr, te)
        validate_generated_puzzle(puzzle)
        assert tr is None or len(puzzle["train"]) == tr
        assert te is None or len(puzzle["test"]) == te
        assert _signature(puzzle) == _signature(generate_puzzle(202, tr, te))
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (4, 4), (2, 4), (True, 1)):
        try:
            generate_puzzle(303, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid pair counts accepted: {(tr, te)}")
