"""Boundary-guide Cartesian products for the reviewed ARC task."""

import numpy as np


_SOURCE_TEXT = (
    (("0505500500", "0000000000", "0000000005", "0000000000", "0000000005",
      "0000000000", "0000000000", "0000000005", "0000000000", "0000000000"),
     ("0505500500", "0000000000", "0202200205", "0000000000", "0202200205",
      "0000000000", "0000000000", "0202200205", "0000000000", "0000000000")),
    (("5005000500", "0000000000", "0000000000", "0000000005", "0000000000",
      "0000000000", "0000000000", "0000000005", "0000000000", "0000000000"),
     ("5005000500", "0000000000", "0000000000", "2002000205", "0000000000",
      "0000000000", "0000000000", "2002000205", "0000000000", "0000000000")),
    (("0055050550", "0000000000", "0000000005", "0000000005", "0000000000",
      "0000000000", "0000000005", "0000000000", "0000000005", "0000000000"),
     ("0055050550", "0000000000", "0022020225", "0022020225", "0000000000",
      "0000000000", "0022020225", "0000000000", "0022020225", "0000000000")),
    (("5055005050", "0000000000", "0000000005", "0000000005", "0000000000",
      "0000000005", "0000000000", "0000000005", "0000000000", "0000000005"),
     ("5055005050", "0000000000", "2022002025", "2022002025", "0000000000",
      "2022002025", "0000000000", "2022002025", "0000000000", "2022002025")),
)


def _text_grid(rows):
    return np.array([[int(ch) for ch in row] for row in rows], dtype=np.int8)


_SOURCE_PAIRS = tuple((_text_grid(a), _text_grid(b)) for a, b in _SOURCE_TEXT)


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _is_source_pair(inp, out):
    return any(np.array_equal(inp, a) and np.array_equal(out, b)
               for a, b in _SOURCE_PAIRS)


def _count_argument(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _count_argument(num_train, "num_train", 2, 10)
    num_test = _count_argument(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _local_palette(inp):
    values, counts = np.unique(inp, return_counts=True)
    if len(values) != 2 or counts[0] == counts[1]:
        return None
    bg_index = int(np.argmax(counts))
    return int(values[bg_index]), int(values[1 - bg_index])


def _interpretations(inp, product):
    """Infer complete outputs from only one input and the trained product color."""
    palette = _local_palette(inp)
    if palette is None or product in palette:
        return []
    background, signal = palette
    height, width = inp.shape
    signal_cells = set(map(tuple, np.argwhere(inp == signal)))
    interpretations = []
    for edge_row in (0, height - 1):
        for edge_col in (0, width - 1):
            if inp[edge_row, edge_col] != background:
                continue
            columns = tuple(c for c in range(width) if inp[edge_row, c] == signal)
            rows = tuple(r for r in range(height) if inp[r, edge_col] == signal)
            if not columns or not rows:
                continue
            expected_signals = ({(edge_row, c) for c in columns}
                                | {(r, edge_col) for r in rows})
            if signal_cells != expected_signals:
                continue
            out = inp.copy()
            out[np.ix_(rows, columns)] = product
            interpretations.append(((edge_row, edge_col), rows, columns, out))
    return interpretations


def _unique_prediction(inp, product):
    interpretations = _interpretations(inp, product)
    if not interpretations:
        return None
    answer = interpretations[0][3]
    if any(not np.array_equal(answer, candidate[3])
           for candidate in interpretations[1:]):
        return None
    return answer


def _sample_shape(rng, shape_mode, split, fixed_shape, diagnostic):
    if shape_mode == "constant":
        return fixed_shape
    if shape_mode == "small_large":
        low, high = (3, 16) if split == "train" else (12, 30)
    elif shape_mode == "large_small":
        low, high = (12, 30) if split == "train" else (3, 16)
    else:
        low, high = 3, 30
    if diagnostic:
        low = max(low, 5)
    if rng.random() < 0.28 and not diagnostic:
        short = int(rng.integers(3, 7))
        long = int(rng.integers(max(9, low), high + 1))
        return (short, long) if rng.random() < 0.5 else (long, short)
    if rng.random() < 0.10:
        side = int(rng.integers(low, high + 1))
        return side, side
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _sample_count(rng, capacity, forced=None):
    if forced == "single":
        return 1
    if forced == "full":
        return capacity
    if forced == "near":
        return max(1, capacity - 1)
    mode = int(rng.choice(6, p=(0.12, 0.10, 0.13, 0.22, 0.20, 0.23)))
    if mode == 0:
        return 1
    if mode == 1:
        return capacity
    if mode == 2:
        return max(1, capacity - 1)
    if mode == 3:
        return int(rng.integers(1, max(2, capacity // 3 + 1)))
    if mode == 4:
        return int(rng.integers(max(1, capacity // 3), max(2, 2 * capacity // 3 + 1)))
    return int(rng.integers(1, capacity + 1))


def _sample_positions(rng, length, corner, forced=None, diagnostic=False):
    available = [p for p in range(length) if p != corner]
    capacity = len(available)
    if diagnostic:
        # An internal hole on each axis refutes bounding-rectangle filling.
        count = int(rng.integers(2, capacity))
        for _ in range(30):
            chosen = sorted(int(p) for p in rng.choice(available, count, replace=False))
            if chosen[-1] - chosen[0] + 1 > count:
                return chosen
        return [available[0], available[-1]]
    count = _sample_count(rng, capacity, forced)
    if count == capacity:
        return available
    far = length - 1 if corner == 0 else 0
    include_far = bool(rng.integers(0, 2))
    style = int(rng.integers(0, 5))
    if style == 0 and count > 1:
        runs = [available[start:start + count]
                for start in range(capacity - count + 1)]
        runs = [run for run in runs if (far in run) == include_far]
        if runs:
            return sorted(runs[int(rng.integers(len(runs)))])
    pool = [p for p in available if p != far]
    needed = count - int(include_far)
    if needed > len(pool):
        include_far = True
        needed = count - 1
    if style == 1 and needed >= 3:
        run_length = int(rng.integers(2, needed))
        start = int(rng.integers(len(pool) - run_length + 1))
        run = pool[start:start + run_length]
        remaining = [p for p in pool if p not in run]
        extra = list(rng.choice(remaining, needed - run_length, replace=False))
        chosen = run + extra
    else:
        chosen = list(rng.choice(pool, needed, replace=False))
    if include_far:
        chosen.append(far)
    return sorted(int(p) for p in chosen)


def _make_pair(rng, shape, corner, palette, product, diagnostic=False,
               forced_axis=None):
    height, width = shape
    edge_row = 0 if corner[0] == 0 else height - 1
    edge_col = 0 if corner[1] == 0 else width - 1
    background, signal = palette
    row_force = forced_axis if bool(rng.integers(0, 2)) else None
    col_force = forced_axis if row_force is None else None
    rows = _sample_positions(rng, height, edge_row, row_force, diagnostic)
    columns = _sample_positions(rng, width, edge_col, col_force, diagnostic)
    inp = np.full((height, width), background, dtype=np.int8)
    inp[edge_row, columns] = signal
    inp[rows, edge_col] = signal
    out = inp.copy()
    out[np.ix_(rows, columns)] = product
    return inp, out


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    product = int(rng.integers(10))
    other_colors = [c for c in range(10) if c != product]
    fixed_palette = tuple(int(c) for c in rng.choice(other_colors, 2, replace=False))
    palette_mode = "mixed" if rng.random() < 0.70 else "constant"
    corner_mode = str(rng.choice(("mixed", "shift", "constant"),
                                 p=(0.55, 0.23, 0.22)))
    shape_mode = str(rng.choice(("overlap", "small_large", "large_small", "constant"),
                                p=(0.43, 0.20, 0.20, 0.17)))
    fixed_shape = (int(rng.integers(5, 31)), int(rng.integers(5, 31)))
    fixed_corner = (int(rng.integers(2)), int(rng.integers(2)))
    corners = [(a, b) for a in range(2) for b in range(2)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if corner_mode == "constant" or (corner_mode == "shift" and split == "train"):
                corner = fixed_corner
            elif corner_mode == "shift":
                alternatives = [c for c in corners if c != fixed_corner]
                corner = alternatives[int(rng.integers(3))]
            elif split == "train" and index == 0:
                corner = fixed_corner
            elif split == "train" and index == 1:
                alternatives = [c for c in corners if c != fixed_corner]
                corner = alternatives[int(rng.integers(3))]
            else:
                corner = corners[int(rng.integers(4))]
            if palette_mode == "constant" or (split == "train" and index == 0):
                palette = fixed_palette
            else:
                palette = tuple(int(c) for c in rng.choice(other_colors, 2, replace=False))
                if palette_mode == "mixed" and split == "train" and index == 1:
                    if palette == fixed_palette:
                        palette = fixed_palette[::-1]
            diagnostic = split == "train" and index == 0
            forced_axis = None
            if split == "test" and index == 0:
                forced_axis = str(rng.choice(("single", "full", "near", "free")))
                if forced_axis == "free":
                    forced_axis = None
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for attempt in range(150):
                shape = _sample_shape(pair_rng, shape_mode, split, fixed_shape, diagnostic)
                inp, out = _make_pair(pair_rng, shape, corner, palette, product,
                                      diagnostic, forced_axis)
                if (_grid_key(inp) not in seen and not _is_source_pair(inp, out)
                        and _unique_prediction(inp, product) is not None):
                    break
            else:
                # Deterministic, separated markers on a distinct sufficiently large canvas.
                shape = (12 + index, 13 + index)
                inp, out = _make_pair(pair_rng, shape, corner, palette, product,
                                      diagnostic=True)
                if _grid_key(inp) in seen or _unique_prediction(inp, product) is None:
                    raise ValueError("unable to construct an unambiguous unique pair")
            seen.add(_grid_key(inp))
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _has_internal_gap(positions):
    return len(positions) >= 2 and positions[-1] - positions[0] + 1 > len(positions)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    products = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            key = _grid_key(inp)
            assert key not in seen_inputs
            seen_inputs.add(key)
            assert not _is_source_pair(inp, out)
            palette = _local_palette(inp)
            assert palette is not None
            assert int(np.count_nonzero(inp == palette[0])) > int(np.count_nonzero(inp == palette[1]))
            new_colors = set(map(int, np.unique(out))) - set(palette)
            assert len(new_colors) == 1
            product = new_colors.pop()
            products.add(product)
            assert not np.array_equal(inp, out)
            assert np.array_equal(inp[inp == palette[1]], out[inp == palette[1]])
            predicted = _unique_prediction(inp, product)
            assert predicted is not None and np.array_equal(predicted, out)
    assert len(products) == 1
    product = next(iter(products))
    # The first demonstration includes gaps on both axes, so rectangle filling
    # and copying a contiguous span disagree with its observed output.
    first_input = puzzle["train"][0]["input"]
    meanings = _interpretations(first_input, product)
    assert meanings and any(_has_internal_gap(rows) and _has_internal_gap(cols)
                            for _, rows, cols, _ in meanings)


def _puzzles_equal(left, right):
    return all(len(left[s]) == len(right[s]) and all(
        np.array_equal(a[role], b[role])
        for a, b in zip(left[s], right[s]) for role in ("input", "output"))
        for s in ("train", "test"))


def _solve_from_training(train, test_input):
    """Self-test oracle with no seed, generated test label, or hidden settings."""
    learned = []
    for example in train:
        new = set(map(int, np.unique(example["output"]))) - set(
            map(int, np.unique(example["input"])))
        assert len(new) == 1
        learned.append(new.pop())
    assert len(set(learned)) == 1
    product = learned[0]
    values, sizes = np.unique(test_input, return_counts=True)
    assert len(values) == 2 and sizes[0] != sizes[1]
    background = int(values[int(np.argmax(sizes))])
    marker = int(values[int(np.argmin(sizes))])
    points = set(map(tuple, np.argwhere(test_input == marker)))
    answers = []
    for edge_row in (0, test_input.shape[0] - 1):
        for edge_col in (0, test_input.shape[1] - 1):
            if test_input[edge_row, edge_col] != background:
                continue
            rows = [row for row, col in points if col == edge_col]
            columns = [col for row, col in points if row == edge_row]
            if not rows or not columns:
                continue
            if points != ({(row, edge_col) for row in rows}
                          | {(edge_row, col) for col in columns}):
                continue
            answer = test_input.copy()
            for row in rows:
                for col in columns:
                    answer[row, col] = product
            answers.append(answer)
    assert answers and all(np.array_equal(answers[0], value) for value in answers)
    return answers[0]


def self_test() -> None:
    ambiguous = _text_grid(("500", "000", "005"))
    assert _unique_prediction(ambiguous, 2) is None
    unique_singleton = _text_grid(("00500", "00005", "00000"))
    assert np.array_equal(_unique_prediction(unique_singleton, 2),
                          _text_grid(("00500", "00205", "00000")))
    puzzles = [generate_puzzle(20000 + seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        training_copies = [{role: pair[role].copy() for role in ("input", "output")}
                           for pair in puzzle["train"]]
        for pair in puzzle["test"]:
            prediction = _solve_from_training(training_copies, pair["input"].copy())
            assert np.array_equal(prediction, pair["output"])
    source_train = [{"input": a.copy(), "output": b.copy()}
                    for a, b in _SOURCE_PAIRS[:3]]
    assert np.array_equal(_solve_from_training(source_train, _SOURCE_PAIRS[3][0]),
                          _SOURCE_PAIRS[3][1])
    assert _puzzles_equal(generate_puzzle(314159, 7, 3),
                          generate_puzzle(314159, 7, 3))
    assert _puzzles_equal(generate_puzzle(271828), generate_puzzle(271828))
    assert not _puzzles_equal(generate_puzzle(1, 6, 2), generate_puzzle(2, 6, 2))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(310000 + (train or 0) * 10 + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(99, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted illegal pair counts")
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    shapes, layouts, colors, counts = set(), set(), set(), set()
    corner_modes, palette_modes = set(), set()
    pattern_counts = {"train": {"single": 0, "full": 0, "near": 0,
                                 "equal_other": 0, "unequal_large": 0,
                                 "narrow": 0, "axis_30": 0},
                      "test": {"single": 0, "full": 0, "near": 0,
                               "equal_other": 0, "unequal_large": 0,
                               "narrow": 0, "axis_30": 0}}
    split_features = {"train": [], "test": []}
    combined_changes = 0
    for puzzle in puzzles:
        corners, palettes = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                product = (set(map(int, np.unique(out))) - set(map(int, np.unique(inp)))).pop()
                meanings = _interpretations(inp, product)
                assert meanings and _unique_prediction(inp, product) is not None
                corner, rows, cols, _ = meanings[0]
                corner = (int(corner[0] == 0), int(corner[1] == 0))
                corners.append(corner)
                palettes.append(_local_palette(inp))
                shapes.add(inp.shape)
                layouts.add((inp.shape, tuple(rows), tuple(cols), corner))
                colors.add(tuple(sorted((int(v) for v in np.unique(inp)))))
                counts.add((len(rows), len(cols)))
                pcounts = pattern_counts[split]
                pcounts["single"] += int(1 in (len(rows), len(cols)))
                pcounts["full"] += int(len(rows) == inp.shape[0] - 1
                                         or len(cols) == inp.shape[1] - 1)
                pcounts["near"] += int(len(rows) >= inp.shape[0] - 2
                                         or len(cols) >= inp.shape[1] - 2)
                pcounts["equal_other"] += int(len(rows) == len(cols) != 5)
                pcounts["unequal_large"] += int(min(len(rows), len(cols)) >= 6
                                                  and len(rows) != len(cols))
                pcounts["narrow"] += int(min(inp.shape) <= 6 and max(inp.shape) >= 12)
                pcounts["axis_30"] += int(30 in inp.shape)
                far_row = (inp.shape[0] - 1) if corner[0] else 0
                far_col = (inp.shape[1] - 1) if corner[1] else 0
                split_features[split].append((inp.shape, len(rows), len(cols),
                                              far_row in rows, far_col in cols))
        corner_modes.add((len(set(corners[:len(puzzle["train"])])), len(set(corners))))
        palette_modes.add(len(set(palettes)))
        if len(set(corners)) > 1 and len(set(palettes)) > 1 and len({
                pair["input"].shape for split in ("train", "test")
                for pair in puzzle[split]}) > 1:
            combined_changes += 1
    assert len(shapes) >= 80 and len(layouts) >= 170 and len(colors) >= 15
    assert len(counts) >= 35 and combined_changes >= 12
    assert (1, 1) in corner_modes and any(a > 1 for a, _ in corner_modes)
    assert any(a == 1 and b > 1 for a, b in corner_modes)
    assert 1 in palette_modes and any(v > 1 for v in palette_modes)
    for split in ("train", "test"):
        features = split_features[split]
        assert all(value > 0 for value in pattern_counts[split].values())
        assert any(min(shape) <= 4 for shape, *_ in features)
        assert any(max(shape) >= 28 for shape, *_ in features)
        assert any(a == 1 or b == 1 for _, a, b, *_ in features)
        assert any(a == shape[0] - 1 or b == shape[1] - 1
                   for shape, a, b, *_ in features)
        assert {int(row_far) + int(col_far) for _, _, _, row_far, col_far in features} == {0, 1, 2}
