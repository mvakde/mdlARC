import numpy as np


def _checked_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low = num_test + 1
        return low + (seed // 4) % (11 - low), num_test
    if num_train is None:
        low = num_test + 1
        return low + seed % (11 - low), num_test
    return num_train, 1 + seed % min(4, num_train - 1)


def _grid_key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


_SOURCE_TEXT = (
    """004000000
004000000
004000000
004000000
334333333
004000000
004000000
004000000
004000000
888888888
004000000
004000000""",
    """00100803000
00100803000
00100803000
22122823222
00100803000
00100803000
00100803000
00100803000
00100803000
55155853555
00100803000
00100803000""",
    """00300001000
77377771777
00300001000
00300001000
00300001000
00300001000
22222221222
00300001000
00300001000
00300001000
88888888888
00300001000""",
    """00080000600
33383333633
00080000600
00080000600
00080000600
00080000600
55585555655
00080000600
00080000600
00080000600""",
    """0030020700400
0030020700400
0030020700400
6666626766466
0030020700400
0030020700400
1111121711411
0030020700400
0030020700400
8888888788488
0030020700400
0030020700400""",
)
_SOURCE_INPUT_KEYS = frozenset(
    _grid_key(np.array([[int(c) for c in row] for row in text.splitlines()], dtype=np.int8))
    for text in _SOURCE_TEXT
)


def _compress_grid(grid):
    rows = np.ones(grid.shape[0], dtype=bool)
    rows[1:] = np.any(grid[1:] != grid[:-1], axis=1)
    reduced = grid[rows]
    columns = np.ones(reduced.shape[1], dtype=bool)
    columns[1:] = np.any(reduced[:, 1:] != reduced[:, :-1], axis=0)
    return reduced[:, columns].copy()


def _axis_runs(grid, axis):
    vectors = grid if axis == 0 else grid.T
    result = []
    length = 1
    for index in range(1, len(vectors)):
        if np.array_equal(vectors[index], vectors[index - 1]):
            length += 1
        else:
            result.append(length)
            length = 1
    result.append(length)
    return tuple(result)


def _sample_dimension(line_count, rng, require_repeat=False):
    maximum = 30 - (2 * line_count + 1)
    minimum = int(require_repeat)
    ranges = ((0, 3, 3), (4, 9, 4), (10, 17, 3), (18, maximum, 2))
    available = [
        (max(low, minimum), min(high, maximum), weight)
        for low, high, weight in ranges
        if max(low, minimum) <= min(high, maximum)
    ]
    weights = np.array([entry[2] for entry in available], dtype=float)
    choice = available[int(rng.choice(len(available), p=weights / weights.sum()))]
    extra = int(rng.integers(choice[0], choice[1] + 1))
    return 2 * line_count + 1 + extra


def _sample_spacing(line_count, dimension, rng):
    parts = line_count + 1
    extra = dimension - (2 * line_count + 1)
    spacing = np.ones(parts, dtype=np.int64)
    if extra:
        mode = int(rng.integers(0, 3))
        if mode == 0:
            spacing += rng.multinomial(extra, np.ones(parts) / parts)
        else:
            bulk = int(rng.integers((extra + 1) // 2, extra + 1))
            if mode == 1 or parts == 2:
                target = int(rng.choice((0, parts - 1)))
            else:
                target = int(rng.integers(1, parts - 1))
            spacing += rng.multinomial(extra - bulk, np.ones(parts) / parts)
            spacing[target] += bulk
    runs = []
    for index, length in enumerate(spacing):
        runs.append(int(length))
        if index < line_count:
            runs.append(1)
    return tuple(runs)


def _line_colors(rng, background, h, v, mode):
    palette = np.array([color for color in range(10) if color != background], dtype=np.int8)
    total = h + v
    if mode == 0 and total <= 9:
        colors = rng.choice(palette, size=total, replace=False).astype(np.int8)
    else:
        colors = rng.choice(palette, size=total, replace=True).astype(np.int8)
        if mode in (1, 3) and (h >= 2 or v >= 2):
            if h >= 2 and (v < 2 or rng.random() < 0.5):
                colors[1] = colors[0]
            else:
                colors[h + 1] = colors[h]
        if mode in (2, 3):
            colors[h] = colors[0]
        if total > 2 and len(set(int(x) for x in colors)) == 1:
            colors[-1] = int(rng.choice(palette[palette != colors[0]]))
    return colors[:h], colors[h:]


def _crossing_bits(rng, h, v, mode):
    if mode == 0:
        return np.zeros((h, v), dtype=np.int8)
    if mode == 1:
        return np.ones((h, v), dtype=np.int8)
    if mode == 2:
        return np.repeat(rng.integers(0, 2, size=(h, 1), dtype=np.int8), v, axis=1)
    if mode == 3:
        return np.repeat(rng.integers(0, 2, size=(1, v), dtype=np.int8), h, axis=0)
    return rng.integers(0, 2, size=(h, v), dtype=np.int8)


def _make_pair(rng, h, v, background, color_mode, crossing_mode,
               both_repeat=False, no_op_axis=None, visible_mix=False):
    horizontal, vertical = _line_colors(rng, background, h, v, color_mode)
    canonical = np.full((2 * h + 1, 2 * v + 1), background, dtype=np.int8)
    for j, color in enumerate(vertical):
        canonical[:, 2 * j + 1] = color
    for i, color in enumerate(horizontal):
        canonical[2 * i + 1, :] = color
    bits = _crossing_bits(rng, h, v, crossing_mode)
    if visible_mix:
        for _ in range(16):
            bits = _crossing_bits(rng, h, v, 4)
            if (np.all(np.any(bits == 0, axis=0))
                    and np.all(np.any(bits == 1, axis=0))
                    and np.all(np.any(bits == 0, axis=1))
                    and np.all(np.any(bits == 1, axis=1))):
                break
        else:
            bits = np.fromfunction(
                lambda i, j: (i + j + int(rng.integers(0, 2))) % 2,
                (h, v), dtype=int,
            ).astype(np.int8)
    for i in range(h):
        for j in range(v):
            canonical[2 * i + 1, 2 * j + 1] = (
                vertical[j] if bits[i, j] else horizontal[i]
            )

    height = (2 * h + 1 if no_op_axis == 0 else
              _sample_dimension(h, rng, require_repeat=both_repeat or no_op_axis == 1))
    width = (2 * v + 1 if no_op_axis == 1 else
             _sample_dimension(v, rng, require_repeat=both_repeat or no_op_axis == 0))
    if height == 2 * h + 1 and width == 2 * v + 1:
        if rng.random() < 0.5:
            height += 1
        else:
            width += 1
    row_runs = _sample_spacing(h, height, rng)
    column_runs = _sample_spacing(v, width, rng)
    expanded = np.repeat(np.repeat(canonical, row_runs, axis=0), column_runs, axis=1)
    return {"input": expanded, "output": _compress_grid(expanded)}


def _test_inventories(rng, train_inventories, num_test):
    horizontal = [item[0] for item in train_inventories]
    vertical = [item[1] for item in train_inventories]
    maximum_h, maximum_v = max(horizontal), max(vertical)
    minimum_h, minimum_v = min(horizontal), min(vertical)
    modes = list(int(x) for x in rng.permutation(5))
    result = []
    for mode in modes[:num_test]:
        if mode == 0:
            target = (int(rng.choice(horizontal)), maximum_v + 1)
        elif mode == 1:
            target = (maximum_h + 1, int(rng.choice(vertical)))
        elif mode == 2:
            target = train_inventories[int(rng.integers(len(train_inventories)))]
        elif mode == 3:
            target = (max(1, minimum_h - 1), max(1, minimum_v - 1))
        else:
            target = (maximum_h + 1, maximum_v + 1)
        if target in result:
            for candidate in (
                (maximum_h + 1, maximum_v + 1),
                (maximum_h + 1, 1),
                (1, maximum_v + 1),
                (1, 1),
                (6, 6),
            ):
                if candidate not in result:
                    target = candidate
                    break
        result.append(target)
    return result


def _fallback_pair(h, v, background, serial, rotation):
    palette = [color for color in range(10) if color != background]
    colors = [palette[(rotation + index) % 9] for index in range(h + v)]
    canonical = np.full((2 * h + 1, 2 * v + 1), background, dtype=np.int8)
    for j in range(v):
        canonical[:, 2 * j + 1] = colors[h + j]
    for i in range(h):
        canonical[2 * i + 1, :] = colors[i]
        for j in range(v):
            if (i + j) % 2:
                canonical[2 * i + 1, 2 * j + 1] = colors[h + j]
    row_runs = [serial + 2] + [1] * (2 * h)
    column_runs = [2] + [1] * (2 * v)
    expanded = np.repeat(
        np.repeat(canonical, row_runs, axis=0), column_runs, axis=1
    )
    return {"input": expanded, "output": _compress_grid(expanded)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # The only learned convention is literal compression. Background policy
    # may stay fixed or vary within a puzzle; every pair identifies its own.
    base_background = int(rng.integers(0, 10))
    varying_background = bool(rng.integers(0, 2))
    anchor = (int(rng.integers(2, 5)), int(rng.integers(2, 5)))
    train_inventories = [anchor]
    for _ in range(1, num_train):
        candidate = (int(rng.integers(1, 6)), int(rng.integers(1, 6)))
        if candidate == anchor:
            candidate = (1 if anchor[0] != 1 else 5, candidate[1])
        train_inventories.append(candidate)
    test_inventories = _test_inventories(rng, train_inventories, num_test)

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for split, inventories in (("train", train_inventories), ("test", test_inventories)):
        for index, (h, v) in enumerate(inventories):
            if not varying_background:
                background = base_background
            elif split == "train" and index == 0:
                background = base_background
            elif split == "train" and index == 1:
                background = (base_background + int(rng.integers(1, 10))) % 10
            else:
                background = int(rng.integers(0, 10))
            anchor_pair = split == "train" and index == 0
            for _ in range(96):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                no_op_axis = None
                if not anchor_pair and pair_rng.random() < 0.18:
                    no_op_axis = int(pair_rng.integers(0, 2))
                pair = _make_pair(
                    pair_rng, h, v, background,
                    color_mode=0 if anchor_pair else int(pair_rng.integers(0, 4)),
                    crossing_mode=4 if anchor_pair else int(pair_rng.integers(0, 5)),
                    both_repeat=anchor_pair, no_op_axis=no_op_axis,
                    visible_mix=anchor_pair,
                )
                key = _grid_key(pair["input"])
                if key not in used_inputs and key not in _SOURCE_INPUT_KEYS:
                    used_inputs.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                for serial in range(17):
                    for rotation in range(9):
                        pair = _fallback_pair(h, v, background, serial, rotation)
                        key = _grid_key(pair["input"])
                        if key not in used_inputs and key not in _SOURCE_INPUT_KEYS:
                            used_inputs.add(key)
                            puzzle[split].append(pair)
                            break
                    else:
                        continue
                    break
                else:
                    raise ValueError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= side <= 30 for side in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    source, output = pair["input"], pair["output"]
    _validate_grid(source)
    _validate_grid(output)
    assert np.array_equal(_compress_grid(source), output)
    assert source.shape != output.shape
    height, width = output.shape
    assert height >= 3 and width >= 3 and height % 2 == width % 2 == 1
    h, v = (height - 1) // 2, (width - 1) // 2
    background = int(output[0, 0])
    assert np.all(output[0::2, 0::2] == background)
    horizontal = tuple(int(output[2 * i + 1, 0]) for i in range(h))
    vertical = tuple(int(output[0, 2 * j + 1]) for j in range(v))
    assert all(color != background for color in horizontal + vertical)
    for i, color in enumerate(horizontal):
        assert np.all(output[2 * i + 1, 0::2] == color)
    for j, color in enumerate(vertical):
        assert np.all(output[0::2, 2 * j + 1] == color)
    ownership = []
    for i in range(h):
        row = []
        for j in range(v):
            color = int(output[2 * i + 1, 2 * j + 1])
            assert color in (horizontal[i], vertical[j])
            row.append(-1 if horizontal[i] == vertical[j] else
                       (0 if color == horizontal[i] else 1))
        ownership.append(tuple(row))
    row_runs, column_runs = _axis_runs(source, 0), _axis_runs(source, 1)
    assert len(row_runs) == height and len(column_runs) == width
    assert all(row_runs[i] == 1 for i in range(1, height, 2))
    assert all(column_runs[j] == 1 for j in range(1, width, 2))
    expanded = np.repeat(np.repeat(output, row_runs, axis=0), column_runs, axis=1)
    assert np.array_equal(expanded, source)
    return {
        "h": h, "v": v, "background": background,
        "horizontal": horizontal, "vertical": vertical,
        "ownership": tuple(ownership), "row_runs": row_runs,
        "column_runs": column_runs,
    }


def _training_rule_visible(analyses):
    for analysis in analyses:
        values = {bit for row in analysis["ownership"] for bit in row}
        if (analysis["h"] >= 2 and analysis["v"] >= 2
                and max(analysis["row_runs"]) > 1
                and max(analysis["column_runs"]) > 1
                and {0, 1} <= values
                and len(set((analysis["background"],) +
                            analysis["horizontal"] + analysis["vertical"]))
                == 1 + analysis["h"] + analysis["v"]):
            return True
    return False


def _predict_from_demonstrations(train_pairs, test_input):
    # This uses only copied demonstrations and an input, not generation state.
    for pair in train_pairs:
        grid = pair["input"]
        row_indices = [0]
        for i in range(1, grid.shape[0]):
            if not np.array_equal(grid[i], grid[i - 1]):
                row_indices.append(i)
        reduced = grid[row_indices]
        column_indices = [0]
        for j in range(1, reduced.shape[1]):
            if not np.array_equal(reduced[:, j], reduced[:, j - 1]):
                column_indices.append(j)
        assert np.array_equal(reduced[:, column_indices], pair["output"])
    row_indices = [0]
    for i in range(1, test_input.shape[0]):
        if not np.array_equal(test_input[i], test_input[i - 1]):
            row_indices.append(i)
    reduced = test_input[row_indices]
    column_indices = [0]
    for j in range(1, reduced.shape[1]):
        if not np.array_equal(reduced[:, j], reduced[:, j - 1]):
            column_indices.append(j)
    return reduced[:, column_indices]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            analysis = _analyze_pair(pair)
            analyses[split].append(analysis)
            key = _grid_key(pair["input"])
            assert key not in used and key not in _SOURCE_INPUT_KEYS
            used.add(key)
    assert _training_rule_visible(analyses["train"])
    assert len({(a["h"], a["v"]) for a in analyses["train"]}) >= 2
    assert len({(a["h"], a["v"]) for a in analyses["test"]}) == len(puzzle["test"])
    train_backgrounds = {a["background"] for a in analyses["train"]}
    test_backgrounds = {a["background"] for a in analyses["test"]}
    if not test_backgrounds <= train_backgrounds:
        assert len(train_backgrounds) >= 2
    copied_train = [
        {"input": pair["input"].copy(), "output": pair["output"].copy()}
        for pair in puzzle["train"]
    ]
    for pair in puzzle["test"]:
        predicted = _predict_from_demonstrations(copied_train, pair["input"].copy())
        assert np.array_equal(predicted, pair["output"])


def _puzzle_key(puzzle):
    return tuple(
        tuple((_grid_key(p["input"]), _grid_key(p["output"])) for p in puzzle[split])
        for split in ("train", "test")
    )


def _require_value_error(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    fallback_keys = set()
    for serial in range(17):
        fallback = _fallback_pair(6, 6, 1, serial, 0)
        _analyze_pair(fallback)
        fallback_keys.add(_grid_key(fallback["input"]))
    assert len(fallback_keys) == 17
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (201, None, 1), (202, None, 4),
                              (203, 2, None), (204, 10, None),
                              (777, 6, 3)):
        first = generate_puzzle(seed, train, test)
        second = generate_puzzle(seed, train, test)
        assert _puzzle_key(first) == _puzzle_key(second)
        if train is not None:
            assert len(first["train"]) == train
        if test is not None:
            assert len(first["test"]) == test
        validate_generated_puzzle(first)
    for callback in (
        lambda: generate_puzzle(-1), lambda: generate_puzzle(0, 1),
        lambda: generate_puzzle(0, 11), lambda: generate_puzzle(0, num_test=0),
        lambda: generate_puzzle(0, num_test=5), lambda: generate_puzzle(0, 2, 2),
        lambda: generate_puzzle(0, 2, 4), lambda: generate_puzzle(0, 2.0, 1),
    ):
        _require_value_error(callback)

    train_counts, test_counts, puzzle_keys = set(), set(), set()
    input_shapes, output_shapes, inventories, layouts = set(), set(), set(), set()
    backgrounds = set()
    all_horizontal = all_vertical = mixed = 0
    parallel_reuse = cross_reuse = no_op = both_shrink = 0
    near = large = rectangular = border_bulk = interior_bulk = 0
    varying_background_puzzles = varied_test_inventories = 0
    added_h = added_v = same_inventory = fewer = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_keys.add(_puzzle_key(puzzle))
        train_analysis = [_analyze_pair(p) for p in puzzle["train"]]
        test_analysis = [_analyze_pair(p) for p in puzzle["test"]]
        if len({a["background"] for a in train_analysis + test_analysis}) > 1:
            varying_background_puzzles += 1
        if len({(a["h"], a["v"]) for a in test_analysis}) > 1:
            varied_test_inventories += 1
        train_inventory = {(a["h"], a["v"]) for a in train_analysis}
        max_h = max(a["h"] for a in train_analysis)
        max_v = max(a["v"] for a in train_analysis)
        min_h = min(a["h"] for a in train_analysis)
        min_v = min(a["v"] for a in train_analysis)
        for a in test_analysis:
            added_h += a["h"] > max_h
            added_v += a["v"] > max_v
            same_inventory += (a["h"], a["v"]) in train_inventory
            fewer += a["h"] < min_h or a["v"] < min_v
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_pair(pair)
                grid = pair["input"]
                input_shapes.add(grid.shape)
                output_shapes.add(pair["output"].shape)
                inventories.add((a["h"], a["v"]))
                layouts.add((a["h"], a["v"], a["row_runs"], a["column_runs"]))
                backgrounds.add(a["background"])
                bits = [bit for row in a["ownership"] for bit in row]
                if bits and all(bit == 0 for bit in bits):
                    all_horizontal += 1
                if bits and all(bit == 1 for bit in bits):
                    all_vertical += 1
                if 0 in bits and 1 in bits:
                    mixed += 1
                parallel_reuse += (len(set(a["horizontal"])) < a["h"] or
                                   len(set(a["vertical"])) < a["v"])
                cross_reuse += bool(set(a["horizontal"]) & set(a["vertical"]))
                row_shrink = grid.shape[0] > pair["output"].shape[0]
                column_shrink = grid.shape[1] > pair["output"].shape[1]
                no_op += row_shrink != column_shrink
                both_shrink += row_shrink and column_shrink
                near += max(grid.shape) <= 9
                large += max(grid.shape) >= 24
                rectangular += max(grid.shape) >= 2 * min(grid.shape)
                for runs in (a["row_runs"], a["column_runs"]):
                    spaces = runs[::2]
                    if max(spaces) >= 5 and max(spaces) >= sum(spaces) / 2:
                        location = spaces.index(max(spaces))
                        border_bulk += location in (0, len(spaces) - 1)
                        interior_bulk += location not in (0, len(spaces) - 1)
                assert np.array_equal(_compress_grid(grid), pair["output"])
                assert _grid_key(grid) not in _SOURCE_INPUT_KEYS
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) == 42 and len(backgrounds) >= 8
    assert len(input_shapes) >= 35 and len(output_shapes) >= 15
    assert len(inventories) >= 20 and len(layouts) >= 150
    assert all_horizontal >= 3 and all_vertical >= 3 and mixed >= 20
    assert parallel_reuse >= 10 and cross_reuse >= 10
    assert no_op >= 8 and both_shrink >= 50
    assert near >= 3 and large >= 20 and rectangular >= 10
    assert border_bulk >= 10 and interior_bulk >= 5
    assert varying_background_puzzles >= 10 and varied_test_inventories >= 8
    assert added_h >= 3 and added_v >= 3 and same_inventory >= 3 and fewer >= 3
    assert _puzzle_key(generate_puzzle(500, 5, 2)) != _puzzle_key(generate_puzzle(501, 5, 2))
