import numpy as np

TOP, BOTTOM, LEFT, RIGHT = range(4)
SIDES = (TOP, BOTTOM, LEFT, RIGHT)
SOURCE_ROWS = (
    ("101010", "010108", "101010", "010101", "101010"),
    ("1010101", "0101010", "1010101", "0101010", "8010101", "0101010", "1010101"),
    ("01010101", "10101010", "01010101", "10801010"),
    ("080", "101", "010"),
    ("10101", "01010", "80101", "01010", "10101"),
)
SOURCE_KEYS = frozenset(
    (len(rows), len(rows[0]), tuple(int(c) for row in rows for c in row))
    for rows in SOURCE_ROWS
)


def _key(grid):
    return (grid.shape[0], grid.shape[1], tuple(int(v) for v in grid.flat))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    key = abs(seed)
    if train is None and test is None:
        test = 1 + key % 4
        low = max(2, test + 1)
        train = low + (key // 4) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + key % (11 - low)
    elif test is None:
        test = 1 + key % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _opposite(side):
    return (BOTTOM, TOP, RIGHT, LEFT)[side]


def _location(height, width, side, offset):
    if side == TOP:
        return 0, offset
    if side == BOTTOM:
        return height - 1, offset
    if side == LEFT:
        return offset, 0
    return offset, width - 1


def _axis_length(height, width, side):
    return height if side in (TOP, BOTTOM) else width


def _orth_length(height, width, side):
    return width if side in (TOP, BOTTOM) else height


def _make_input(height, width, side, offset, selected, other, marker):
    row, col = _location(height, width, side, offset)
    rr, cc = np.indices((height, width))
    grid = np.where((rr + cc) % 2 == (row + col) % 2, selected, other).astype(np.int8)
    grid[row, col] = marker
    return grid


def _transform(grid, selected, other, marker, annotation, side):
    row, col = (int(x) for x in np.argwhere(grid == marker)[0])
    output = np.empty_like(grid)
    for index in range(_orth_length(*grid.shape, side)):
        line = grid[:, index] if side in (TOP, BOTTOM) else grid[index, :]
        ordinary = int(np.count_nonzero(line == selected))
        others = int(np.count_nonzero(line == other))
        marked = index == (col if side in (TOP, BOTTOM) else row)
        if marked:
            assert others - ordinary in (0, 1)
            packed = [marker] + [selected] * ordinary + [annotation] * ordinary + [other] * (others - ordinary)
        else:
            packed = [selected] * ordinary + [other] * others
        if side in (BOTTOM, RIGHT):
            packed.reverse()
        if side in (TOP, BOTTOM):
            output[:, index] = packed
        else:
            output[index, :] = packed
    return output


def _side_plan(rng, num_train, num_test):
    if num_train == 2:
        first = int(rng.integers(4))
        train = [first, _opposite(first)]
    else:
        count = int(rng.integers(2, min(4, num_train) + 1))
        if count == 2:
            first = int(rng.integers(4))
            pool = [first, _opposite(first)]
        else:
            pool = [int(x) for x in rng.permutation(4)[:count]]
        train = pool + [int(rng.choice(pool)) for _ in range(num_train - count)]
    rng.shuffle(train)
    allowed = list(set(train))
    rng.shuffle(allowed)
    if num_train >= 6 and len(allowed) == 3 and rng.random() < 0.3:
        allowed.append(next(side for side in SIDES if side not in allowed))
    test = [int(x) for x in rng.permutation(allowed)[:min(num_test, len(allowed))]]
    while len(test) < num_test:
        test.append(int(rng.choice(allowed)))
    rng.shuffle(test)
    return train, test


def _dimension(rng, split):
    bands = ((3, 8), (9, 16), (17, 26)) if split == "train" else ((3, 8), (9, 17), (18, 29))
    probabilities = (0.34, 0.38, 0.28) if split == "train" else (0.25, 0.36, 0.39)
    low, high = bands[int(rng.choice(3, p=probabilities))]
    return int(rng.integers(low, high + 1))


def _geometry(rng, side, split, parity=None, large_floor=None):
    height = _dimension(rng, split)
    width = height if rng.random() < 0.29 else _dimension(rng, split)
    if large_floor is not None:
        height = int(rng.integers(max(3, large_floor[0]), 31))
        width = int(rng.integers(max(3, large_floor[1]), 31))
    if parity is not None:
        if side in (TOP, BOTTOM):
            if height % 2 != parity:
                height += 1 if height < 30 else -1
        elif width % 2 != parity:
            width += 1 if width < 30 else -1
    orth = _orth_length(height, width, side)
    if rng.random() < (0.17 if split == "train" else 0.38):
        offset = int(rng.choice((1, orth - 2)))
    else:
        offset = int(rng.integers(1, orth - 1))
    return height, width, side, offset


def _palette(rng, annotation, base, mode, index):
    if mode == "fixed":
        return base
    if mode == "swap":
        return (base[1], base[0], base[2]) if index % 2 else base
    colors = [color for color in range(10) if color != annotation]
    chosen = tuple(int(x) for x in rng.choice(colors, size=3, replace=False))
    if index == 1 and chosen == base:
        return (chosen[1], chosen[0], chosen[2])
    return chosen


def _make_pair(config, palette, annotation):
    height, width, side, offset = config
    selected, other, marker = palette
    grid = _make_input(height, width, side, offset, selected, other, marker)
    return {"input": grid, "output": _transform(grid, selected, other, marker, annotation, side)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    colors = tuple(int(x) for x in rng.permutation(10)[:4])
    base, annotation = colors[:3], colors[3]
    mode = str(rng.choice(("fixed", "swap", "varied"), p=(0.43, 0.24, 0.33)))
    train_sides, test_sides = _side_plan(rng, num_train, num_test)
    parity_plan = [None] * num_train
    even_index, odd_index = (int(x) for x in rng.choice(num_train, size=2, replace=False))
    parity_plan[even_index], parity_plan[odd_index] = 0, 1
    train, test, used = [], [], set()
    for index, side in enumerate(train_sides):
        for _ in range(120):
            config = _geometry(rng, side, "train", parity_plan[index])
            pair = _make_pair(config, _palette(rng, annotation, base, mode, index), annotation)
            key = _key(pair["input"])
            if key not in used and key not in SOURCE_KEYS:
                train.append(pair)
                used.add(key)
                break
        else:
            raise RuntimeError("could not build unique training pair")
    largest = (max(p["input"].shape[0] for p in train), max(p["input"].shape[1] for p in train))
    large_test = int(rng.integers(num_test)) if rng.random() < 0.43 else None
    for index, side in enumerate(test_sides):
        for _ in range(120):
            config = _geometry(rng, side, "test", large_floor=largest if index == large_test else None)
            pair = _make_pair(config, _palette(rng, annotation, base, mode, num_train + index), annotation)
            key = _key(pair["input"])
            if key not in used and key not in SOURCE_KEYS:
                test.append(pair)
                used.add(key)
                break
        else:
            raise RuntimeError("could not build unique test pair")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    grid, output = pair["input"], pair["output"]
    height, width = grid.shape
    assert height >= 3 and width >= 3 and output.shape == grid.shape
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3
    candidates = []
    for value, count in zip(values, counts):
        if count != 1:
            continue
        row, col = (int(x) for x in np.argwhere(grid == value)[0])
        if (row in (0, height - 1)) != (col in (0, width - 1)):
            candidates.append((int(value), row, col))
    assert len(candidates) == 1
    marker, row, col = candidates[0]
    side = TOP if row == 0 else BOTTOM if row == height - 1 else LEFT if col == 0 else RIGHT
    rr, cc = np.indices(grid.shape)
    phase = (rr + cc) % 2 == (row + col) % 2
    same = phase & (grid != marker)
    selected_values, other_values = np.unique(grid[same]), np.unique(grid[~phase])
    assert len(selected_values) == len(other_values) == 1
    selected, other = int(selected_values[0]), int(other_values[0])
    assert len({selected, other, marker}) == 3
    assert np.all(grid[same] == selected) and np.all(grid[~phase] == other)
    extra = set(int(x) for x in np.unique(output)) - set(int(x) for x in values)
    assert len(extra) == 1
    annotation = next(iter(extra))
    assert set(int(x) for x in np.unique(output)) == {selected, other, marker, annotation}
    assert output[row, col] == marker and np.count_nonzero(output == marker) == 1
    for index in range(_orth_length(height, width, side)):
        source = grid[:, index] if side in (TOP, BOTTOM) else grid[index, :]
        result = output[:, index] if side in (TOP, BOTTOM) else output[index, :]
        if side in (BOTTOM, RIGHT):
            source, result = source[::-1], result[::-1]
        ordinary = int(np.count_nonzero(source == selected))
        others = int(np.count_nonzero(source == other))
        if index == (col if side in (TOP, BOTTOM) else row):
            assert others - ordinary in (0, 1)
            expected = [marker] + [selected] * ordinary + [annotation] * ordinary + [other] * (others - ordinary)
        else:
            expected = [selected] * ordinary + [other] * others
        assert np.array_equal(result, np.asarray(expected))
    assert not np.array_equal(grid, output)
    return {"side": side, "height": height, "width": width,
            "offset": col if side in (TOP, BOTTOM) else row,
            "length": _axis_length(height, width, side),
            "annotation": annotation, "selected": selected, "other": other, "marker": marker}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    metas = {"train": [], "test": []}
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dim <= 30 for dim in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].dtype == pair["output"].dtype
            key = _key(pair["input"])
            assert key not in seen and key not in SOURCE_KEYS
            seen.add(key)
            metas[split].append(_inspect_pair(pair))
    train, test = metas["train"], metas["test"]
    assert len({m["annotation"] for m in train + test}) == 1
    assert len({m["side"] for m in train}) >= 2
    assert {m["length"] % 2 for m in train} == {0, 1}
    seen_sides = {m["side"] for m in train}
    for m in test:
        if m["side"] not in seen_sides:
            assert train_count >= 6 and len(seen_sides) >= 3
    if test_count >= 2:
        assert len({m["side"] for m in test}) >= 2


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    def fails(*args):
        try:
            generate_puzzle(*args)
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    for args in ((1, 1, None), (1, 11, None), (1, None, 0), (1, None, 5), (1, 2, 2), (1, 3, 4)):
        fails(*args)
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(500 + (train or 0) + (test or 0), train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    seen, layouts, palettes, train_counts, test_counts = set(), set(), set(), set(), set()
    train_parities, test_parities, test_sides = set(), set(), set()
    small_train = rectangular_test = near_corner_test = larger_test = varied_palette = 0
    square_test = centered_test = first_test_later = mixed_tests = multi_tests = 0
    one_axis = repeated_side = four_sides = unseen_side = 0
    fixed_palette = varied_marker = combined_test = offset_one = offset_far = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        seen.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_meta = [_inspect_pair(p) for p in puzzle["train"]]
        test_meta = [_inspect_pair(p) for p in puzzle["test"]]
        train_sides = {m["side"] for m in train_meta}
        train_palettes = {(m["selected"], m["other"], m["marker"]) for m in train_meta}
        varied_palette += len(train_palettes) > 1
        fixed_palette += len(train_palettes) == 1
        varied_marker += len({m["marker"] for m in train_meta}) > 1
        small_train += any(m["height"] <= 8 and m["width"] <= 8 for m in train_meta)
        rectangular_test += any(m["height"] != m["width"] for m in test_meta)
        square_test += any(m["height"] == m["width"] for m in test_meta)
        near_corner_test += any(m["offset"] in (1, _orth_length(m["height"], m["width"], m["side"]) - 2) for m in test_meta)
        centered_test += any(2 * m["offset"] == _orth_length(m["height"], m["width"], m["side"]) - 1 for m in test_meta)
        offset_one += any(m["offset"] == 1 for m in test_meta)
        offset_far += any(m["offset"] == _orth_length(m["height"], m["width"], m["side"]) - 2 for m in test_meta)
        larger_test += any(m["height"] >= max(t["height"] for t in train_meta) and m["width"] >= max(t["width"] for t in train_meta) for m in test_meta)
        first_test_later += test_meta[0]["side"] != train_meta[0]["side"]
        multi_tests += len(test_meta) > 1
        mixed_tests += len(test_meta) > 1 and len({m["side"] for m in test_meta}) > 1
        one_axis += len({m["side"] // 2 for m in train_meta}) == 1
        repeated_side += len(train_sides) < len(train_meta)
        four_sides += len(train_sides) == 4
        unseen_side += any(m["side"] not in train_sides for m in test_meta)
        combined_test += any(
            m["length"] % 2 == 0 and m["height"] != m["width"]
            and m["offset"] in (1, _orth_length(m["height"], m["width"], m["side"]) - 2)
            for m in test_meta
        )
        for m in train_meta + test_meta:
            layouts.add((m["height"], m["width"], m["side"], m["offset"]))
            palettes.add((m["selected"], m["other"], m["marker"], m["annotation"]))
        train_parities.update(m["length"] % 2 for m in train_meta)
        test_parities.update(m["length"] % 2 for m in test_meta)
        test_sides.update(m["side"] for m in test_meta)
    assert len(seen) == 42 and len(layouts) >= 130 and len(palettes) >= 35
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert train_parities == test_parities == {0, 1} and test_sides == set(SIDES)
    assert small_train >= 4 and rectangular_test >= 8 and near_corner_test >= 8
    assert larger_test >= 5 and varied_palette >= 15
    assert square_test >= 5 and centered_test >= 2 and offset_one >= 5 and offset_far >= 5
    assert first_test_later >= 10 and mixed_tests == multi_tests
    assert one_axis >= 4 and repeated_side >= 10 and four_sides >= 4 and unseen_side >= 1
    assert fixed_palette >= 10 and varied_marker >= 5 and combined_test >= 4
