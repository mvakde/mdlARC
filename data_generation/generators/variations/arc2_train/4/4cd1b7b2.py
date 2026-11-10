import numpy as np


_SOURCE_PAIRS = (
    ("0423/4102/0340/3014", "1423/4132/2341/3214"),
    ("1034/0021/2140/0312", "1234/3421/2143/4312"),
    ("3021/1000/4302/0143", "3421/1234/4312/2143"),
    ("0123/0310/3041/0402", "4123/2314/3241/1432"),
)


def _count_parameter(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _pair_counts(seed, num_train, num_test):
    train = _count_parameter(num_train, "num_train", 2, 10)
    test = _count_parameter(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _latin_square(rng, order):
    if order == 4 and int(rng.integers(2)):
        base = np.array([[i ^ j for j in range(4)] for i in range(4)], dtype=np.int8)
    else:
        base = np.array([[(i + j) % order for j in range(order)] for i in range(order)], dtype=np.int8)
    rows = rng.permutation(order)
    columns = rng.permutation(order)
    values = rng.permutation(order)
    return values[base[np.ix_(rows, columns)]]


def _completions(grid, symbols, blank, limit=2):
    order = grid.shape[0]
    symbols = tuple(sorted(int(x) for x in symbols))
    symbol_set = set(symbols)
    work = grid.copy()
    rows = [set() for _ in range(order)]
    columns = [set() for _ in range(order)]
    for r in range(order):
        for c in range(order):
            value = int(grid[r, c])
            if value == blank:
                continue
            if value not in symbol_set or value in rows[r] or value in columns[c]:
                return 0, None
            rows[r].add(value)
            columns[c].add(value)
    count = 0
    first = None

    def search():
        nonlocal count, first
        if count >= limit:
            return
        best = None
        choices = None
        for r in range(order):
            for c in range(order):
                if int(work[r, c]) != blank:
                    continue
                available = symbol_set - rows[r] - columns[c]
                if not available:
                    return
                if choices is None or len(available) < len(choices):
                    best = (r, c)
                    choices = available
                    if len(choices) == 1:
                        break
            if choices is not None and len(choices) == 1:
                break
        if best is None:
            count += 1
            if first is None:
                first = work.copy()
            return
        r, c = best
        for value in sorted(choices):
            work[r, c] = value
            rows[r].add(value)
            columns[c].add(value)
            search()
            columns[c].remove(value)
            rows[r].remove(value)
            work[r, c] = blank
            if count >= limit:
                return

    search()
    return count, first


def _visible_symbols(square, mask, order):
    return len(set(int(x) for x in square[~mask])) == order


def _unique_mask(square, mask, blank, symbols):
    if not _visible_symbols(square, mask, len(symbols)):
        return False
    grid = square.copy()
    grid[mask] = blank
    count, answer = _completions(grid, symbols, blank)
    return count == 1 and np.array_equal(answer, square)


def _initial_mask(rng, square, kind, axis, blank, symbols):
    order = square.shape[0]
    mask = np.zeros((order, order), dtype=bool)
    if kind == "classic":
        candidates = []
        for focus in range(4):
            others = [r for r in range(4) if r != focus]
            for retained in range(4):
                for a in range(4):
                    for b in range(4):
                        for c in range(4):
                            trial = np.zeros((4, 4), dtype=bool)
                            trial[focus, :] = True
                            trial[focus, retained] = False
                            for row, column in zip(others, (a, b, c)):
                                trial[row, column] = True
                            if tuple(sorted(int(x) for x in np.sum(trial, axis=0))) != (1, 1, 2, 2):
                                continue
                            if axis == "column":
                                trial = trial.T.copy()
                            if _unique_mask(square, trial, blank, symbols):
                                candidates.append(trial)
        if candidates:
            return candidates[int(rng.integers(len(candidates)))]
    elif kind == "line":
        index = int(rng.integers(order))
        if axis == "row":
            mask[index, :] = True
        else:
            mask[:, index] = True
    elif kind == "rectangle":
        candidates = []
        for r1 in range(order):
            for r2 in range(r1 + 1, order):
                for c1 in range(order):
                    for c2 in range(c1 + 1, order):
                        trial = mask.copy()
                        trial[np.ix_((r1, r2), (c1, c2))] = True
                        if _unique_mask(square, trial, blank, symbols):
                            candidates.append(trial)
        if candidates:
            mask = candidates[int(rng.integers(len(candidates)))]
    return mask


def _mask_target(rng, order, kind):
    if kind == "classic":
        return 6
    if kind == "few":
        return int(rng.integers(1, min(4, order) + 1))
    if kind == "rectangle":
        return int(rng.integers(4, min(order * 2 + 1, order * order)))
    if kind == "line":
        return int(rng.integers(order, min(order * 2 + 2, order * order)))
    if kind == "sparse":
        return int(rng.integers(order * 2, min(order * 3 + 1, order * order)))
    return int(rng.integers(3, min(order * 2 + 4, order * order)))


def _mask(rng, square, blank, symbols, kind, axis, local_blank):
    order = square.shape[0]
    target = _mask_target(rng, order, kind)
    if local_blank:
        target = max(2, target)
    best = None
    best_distance = 100
    for _ in range(6 if kind == "sparse" else 3):
        start_kind = kind if kind in ("line", "rectangle", "classic") else "random"
        mask = _initial_mask(rng, square, start_kind, axis, blank, symbols)
        if not _unique_mask(square, mask, blank, symbols):
            mask[:] = False
        if local_blank and not np.any(np.sum(mask, axis=0) >= 2) and not np.any(np.sum(mask, axis=1) >= 2):
            line = int(rng.integers(order))
            positions = rng.choice(order, size=2, replace=False)
            trial = mask.copy()
            if axis == "row":
                trial[line, positions] = True
            else:
                trial[positions, line] = True
            if _unique_mask(square, trial, blank, symbols):
                mask = trial
        preference = rng.random((order, order))
        if axis == "row":
            focus = int(rng.integers(order))
            preference[focus, :] -= 0.35
        elif axis == "column":
            focus = int(rng.integers(order))
            preference[:, focus] -= 0.35
        if kind == "cluster":
            center_r, center_c = rng.integers(order, size=2)
            for r in range(order):
                for c in range(order):
                    preference[r, c] += 0.08 * (abs(r - center_r) + abs(c - center_c))
        for flat in np.argsort(preference, axis=None):
            if int(np.sum(mask)) >= target:
                break
            r, c = divmod(int(flat), order)
            if mask[r, c]:
                continue
            mask[r, c] = True
            if not _unique_mask(square, mask, blank, symbols):
                mask[r, c] = False
        count = int(np.sum(mask))
        repeat = bool(np.any(np.sum(mask, axis=0) >= 2) or np.any(np.sum(mask, axis=1) >= 2))
        if count and (not local_blank or repeat):
            distance = abs(target - count)
            if distance < best_distance:
                best, best_distance = mask.copy(), distance
            if distance == 0:
                break
    if best is None:
        raise RuntimeError("unable to construct a recoverable mask")
    return best


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_copy(inp, out):
    left = "/".join("".join(str(int(x)) for x in row) for row in inp)
    right = "/".join("".join(str(int(x)) for x in row) for row in out)
    return (left, right) in _SOURCE_PAIRS


def _repeated_color(grid):
    candidates = []
    for value in set(int(x) for x in grid.flat):
        if any(np.sum(grid[r, :] == value) > 1 for r in range(grid.shape[0])) or any(
            np.sum(grid[:, c] == value) > 1 for c in range(grid.shape[1])
        ):
            candidates.append(value)
    return candidates


def _predict_from_train(train, test_input):
    blanks = {int(pair["input"][pair["input"] != pair["output"]][0]) for pair in train}
    if len(blanks) == 1:
        blank = next(iter(blanks))
    else:
        repeated = _repeated_color(test_input)
        if len(repeated) != 1:
            return None
        blank = repeated[0]
    symbols = set(int(x) for x in test_input.flat) - {blank}
    if len(symbols) != test_input.shape[0]:
        return None
    count, answer = _completions(test_input, symbols, blank)
    return answer if count == 1 else None


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _count_parameter(seed, "seed", 0, 2**63 - 1)
    train_count, test_count = _pair_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    order_mode = seed % 5
    local_blank = (seed // 5) % 4 == 0
    shared_blank = int(rng.integers(10))
    shift = (seed // 3) % 4
    if order_mode in (0, 1):
        train_orders = [4] * train_count
        test_orders = [4] * test_count
    elif order_mode == 2:
        order = (3, 5, 6)[int(rng.integers(3))]
        train_orders = [order] * train_count
        test_orders = [order] * test_count
    elif order_mode == 3:
        train_orders = [int(rng.integers(3, 7)) for _ in range(train_count)]
        test_orders = [int(rng.integers(3, 7)) for _ in range(test_count)]
    else:
        high = int(rng.integers(4, 7))
        train_orders = [int(rng.integers(3, high)) for _ in range(train_count)]
        test_orders = [high] + [int(rng.integers(3, 7)) for _ in range(test_count - 1)]
    row_focus = int(rng.integers(train_count))
    column_focus = int(rng.integers(train_count))
    train_plan = []
    test_plan = []
    kinds = ("few", "random", "random", "cluster", "rectangle", "line", "sparse", "classic")
    for index in range(train_count):
        axis = ("row", "column", "row", "column")[shift]
        if shift == 2:
            axis = ("row", "column")[int(rng.integers(2))]
        if shift == 3:
            axis = ("row", "column", "neutral")[int(rng.integers(3))]
        kind = kinds[int(rng.integers(len(kinds)))]
        if (shift == 0 and index == row_focus) or (shift == 1 and index == column_focus):
            kind = "classic"
        train_plan.append((kind, axis))
    for index in range(test_count):
        axis = ("column", "row", train_plan[index % train_count][1], "neutral")[shift]
        if shift == 3:
            axis = ("row", "column", "neutral")[int(rng.integers(3))]
        kind = kinds[int(rng.integers(len(kinds)))]
        if index == 0 and shift in (0, 1):
            kind = "classic"
        test_plan.append((kind, axis))
    if train_count == 2 and all(k == "few" for k, _ in train_plan):
        train_plan[1] = ("rectangle", train_plan[1][1])
    pairs = {"train": [], "test": []}
    used = set()
    train_blanks = []
    for section, orders, plans in (("train", train_orders, train_plan), ("test", test_orders, test_plan)):
        for index, (order, (kind, axis)) in enumerate(zip(orders, plans)):
            if kind == "classic" and order != 4:
                kind = "line"
            for attempt in range(32):
                if local_blank:
                    options = [color for color in range(10) if section != "train" or index == 0 or color != train_blanks[0]]
                    blank = int(options[int(rng.integers(len(options)))])
                else:
                    blank = shared_blank
                available = [color for color in range(10) if color != blank]
                symbols = np.asarray(rng.choice(available, size=order, replace=False), dtype=np.int8)
                square = symbols[_latin_square(rng, order)]
                mask = _mask(rng, square, blank, symbols, kind, axis, local_blank)
                inp = square.copy()
                inp[mask] = blank
                key = _grid_key(inp)
                if key in used or _source_copy(inp, square):
                    continue
                if section == "test" and _predict_from_train(pairs["train"], inp) is None:
                    continue
                used.add(key)
                pairs[section].append({"input": inp, "output": square})
                if section == "train":
                    train_blanks.append(blank)
                break
            else:
                raise RuntimeError("could not make a distinct pair")
    validate_generated_puzzle(pairs)
    return pairs


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    train_blanks = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert 0 <= int(np.min(grid)) and int(np.max(grid)) <= 9
            assert inp.shape == out.shape
            order = inp.shape[0]
            assert 3 <= order <= 6 and inp.shape == (order, order)
            symbols = set(int(x) for x in out.flat)
            assert len(symbols) == order
            for line in list(out) + list(out.T):
                assert set(int(x) for x in line) == symbols
            changed = inp != out
            assert np.any(changed)
            blanks = set(int(x) for x in inp[changed])
            assert len(blanks) == 1
            blank = next(iter(blanks))
            assert blank not in symbols
            assert np.array_equal(changed, inp == blank)
            assert set(int(x) for x in inp.flat) == symbols | {blank}
            assert np.array_equal(inp[~changed], out[~changed])
            count, solved = _completions(inp, symbols, blank)
            assert count == 1 and np.array_equal(solved, out)
            ikey, okey = _grid_key(inp), _grid_key(out)
            assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
            assert not _source_copy(inp, out)
            if section == "train":
                train_blanks.add(blank)
    if len(train_blanks) > 1:
        for pair in puzzle["train"] + puzzle["test"]:
            changed = pair["input"] != pair["output"]
            blank = int(pair["input"][changed][0])
            assert _repeated_color(pair["input"]) == [blank]
    else:
        shared = next(iter(train_blanks))
        for pair in puzzle["test"]:
            changed = pair["input"] != pair["output"]
            assert int(pair["input"][changed][0]) == shared
    for pair in puzzle["test"]:
        assert np.array_equal(_predict_from_train(puzzle["train"], pair["input"].copy()), pair["output"])


def _line_passes(inp, blank):
    mask = inp == blank
    passes = 0
    while np.any(mask):
        row_counts = np.sum(mask, axis=1)
        column_counts = np.sum(mask, axis=0)
        removable = mask & ((row_counts[:, None] == 1) | (column_counts[None, :] == 1))
        if not np.any(removable):
            return None
        mask[removable] = False
        passes += 1
    return passes


def _no_initial_line_single(mask):
    rows = np.sum(mask, axis=1)
    columns = np.sum(mask, axis=0)
    return not (np.any(rows == 1) or np.any(columns == 1))


def _puzzles_equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[s], second[s])
    ) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, sizes, blanks, masks, palettes, structures = (set() for _ in range(7))
    coverage = {key: 0 for key in (
        "missing_row", "missing_column", "given_row", "given_column", "rectangle",
        "four_eleven", "row_to_column", "column_to_row", "row_to_row",
        "column_to_column", "heldout_larger", "combined", "classic_train_row",
        "classic_test_column", "no_classic_train_row", "multiple_classic_train_row",
        "nonfirst_classic_train_row",
    )}
    per_puzzle_palette_variation = 0
    per_puzzle_size_variation = 0
    local_role_puzzles = 0
    full_lines = 0
    few = 0
    dense = 0
    intersection = 0
    long_chains = 0
    signatures = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_palettes, pair_sizes, pair_blanks = set(), set(), set()
        split_focus = {"train": set(), "test": set()}
        train_sizes, test_sizes = [], []
        classic_row_positions = []
        puzzle_intersection = False
        signature = []
        for section, index, pair in [
            (section, index, pair)
            for section in ("train", "test")
            for index, pair in enumerate(puzzle[section])
        ]:
            inp, out = pair["input"], pair["output"]
            blank = int(inp[inp != out][0])
            mask = inp == blank
            order = len(inp)
            holes = int(np.sum(mask))
            row_counts = np.sum(mask, axis=1)
            column_counts = np.sum(mask, axis=0)
            coverage["missing_row"] += int(np.any(row_counts == order))
            coverage["missing_column"] += int(np.any(column_counts == order))
            coverage["given_row"] += int(np.any(row_counts == 0))
            coverage["given_column"] += int(np.any(column_counts == 0))
            coverage["four_eleven"] += int(order == 4 and holes >= 11)
            coverage["rectangle"] += int(
                order == 4 and holes == 4
                and tuple(sorted(int(x) for x in row_counts)) == (0, 0, 2, 2)
                and tuple(sorted(int(x) for x in column_counts)) == (0, 0, 2, 2)
            )
            if order == 4 and holes == 6:
                if section == "train" and tuple(sorted(int(x) for x in row_counts)) == (1, 1, 1, 3):
                    coverage["classic_train_row"] += 1
                    classic_row_positions.append(index)
                if section == "test" and tuple(sorted(int(x) for x in column_counts)) == (1, 1, 1, 3):
                    coverage["classic_test_column"] += 1
            if int(np.max(row_counts)) > int(np.max(column_counts)):
                split_focus[section].add("row")
            if int(np.max(column_counts)) > int(np.max(row_counts)):
                split_focus[section].add("column")
            (train_sizes if section == "train" else test_sizes).append(order)
            sizes.add(order)
            blanks.add(holes)
            masks.add((order, tuple(int(x) for x in mask.flat)))
            palette = (blank, tuple(sorted(int(x) for x in set(out.flat))))
            palettes.add(palette)
            pair_palettes.add(palette[1])
            pair_sizes.add(order)
            pair_blanks.add(blank)
            structures.add((order, tuple(int(x) for x in out.flat)))
            signature.append((_grid_key(inp), _grid_key(out)))
            full_lines += int(np.any(np.sum(mask, axis=0) == order) or np.any(np.sum(mask, axis=1) == order))
            few += int(holes <= 3)
            dense += int(holes >= 9)
            passes = _line_passes(inp, blank)
            no_line_single = _no_initial_line_single(mask)
            intersection += int(no_line_single)
            puzzle_intersection |= no_line_single
            long_chains += int(passes is not None and passes >= 3)
        per_puzzle_palette_variation += len(pair_palettes) > 1
        per_puzzle_size_variation += len(pair_sizes) > 1
        local_role_puzzles += len(pair_blanks) > 1
        coverage["row_to_column"] += "row" in split_focus["train"] and "column" in split_focus["test"]
        coverage["column_to_row"] += "column" in split_focus["train"] and "row" in split_focus["test"]
        coverage["row_to_row"] += "row" in split_focus["train"] and "row" in split_focus["test"]
        coverage["column_to_column"] += "column" in split_focus["train"] and "column" in split_focus["test"]
        coverage["heldout_larger"] += max(test_sizes) > max(train_sizes)
        coverage["combined"] += len(pair_sizes) > 1 and len(pair_palettes) > 1 and puzzle_intersection
        coverage["no_classic_train_row"] += not classic_row_positions
        coverage["multiple_classic_train_row"] += len(classic_row_positions) > 1
        coverage["nonfirst_classic_train_row"] += any(index > 0 for index in classic_row_positions)
        signatures.add(tuple(signature))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert sizes == {3, 4, 5, 6}
    assert len(blanks) >= 9 and len(masks) >= 120 and len(palettes) >= 40
    assert len(structures) >= 150 and len(signatures) == 42
    assert per_puzzle_palette_variation >= 20 and per_puzzle_size_variation >= 8
    assert local_role_puzzles >= 5 and full_lines >= 15 and few >= 8 and dense >= 10
    assert intersection >= 8 and long_chains >= 2
    assert coverage["rectangle"] >= 1 and coverage["four_eleven"] >= 3
    assert coverage["classic_train_row"] >= 3 and coverage["classic_test_column"] >= 3
    assert coverage["no_classic_train_row"] >= 5
    assert coverage["multiple_classic_train_row"] >= 1
    assert coverage["nonfirst_classic_train_row"] >= 1
    assert all(coverage[key] >= 5 for key in (
        "missing_row", "missing_column", "given_row", "given_column",
        "row_to_column", "column_to_row", "row_to_row", "column_to_column",
        "heldout_larger", "combined",
    ))
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4), (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    first = generate_puzzle(314159, 7, 3)
    second = generate_puzzle(314159, 7, 3)
    assert _puzzles_equal(first, second)
    assert not _puzzles_equal(first, generate_puzzle(314160, 7, 3))
    for arguments in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(3001, *arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
