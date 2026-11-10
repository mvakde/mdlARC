import numpy as np


_SOURCE_PAIRS = {
    (("111150000", "100150660", "100150660", "111150000"), ("1111", "1661", "1661", "1111")),
    (("111152200", "100152200", "100150000", "111150000"), ("1111", "1001", "1001", "1111")),
    (("111150000", "100050777", "101150700", "101050707"), ("1111", "1777", "1711", "1717")),
    (("000152200", "100052200", "110050220", "111050220"), ("0001", "1000", "1100", "1110")),
    (("110050033", "100150330", "100150330", "110050033"), ("1133", "1331", "1331", "1133")),
    (("111153300", "100153300", "100153000", "100150000"), ("1111", "1001", "1001", "1001")),
    (("000152220", "100050222", "110050022", "111050002"), ("2221", "1222", "1122", "1112")),
    (("111152000", "011052222", "011052000", "000050000"), ("1111", "0110", "0110", "0000")),
    (("110050033", "100150330", "000153330", "011153000"), ("1133", "1331", "3331", "3111")),
}


def _count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _array_key(grid):
    return grid.shape, grid.tobytes()


def _mask_key(mask):
    return mask.shape, mask.tobytes()


def _source_copy(pair):
    def rows(grid):
        return tuple("".join(str(int(cell)) for cell in row) for row in grid)
    return (rows(pair["input"]), rows(pair["output"])) in _SOURCE_PAIRS


def _shape(rng):
    return int(rng.integers(4, 11)), int(rng.integers(4, 13))


def _sample_base(rng, height, width, style=None):
    area = height * width
    rows, columns = np.indices((height, width))
    for _ in range(32):
        choice = int(rng.integers(0, 7)) if style is None else style
        if choice == 0:
            hole_height = int(rng.integers(2, height))
            hole_width = int(rng.integers(2, width))
            top = int(rng.integers(0, height - hole_height + 1))
            left = int(rng.integers(0, width - hole_width + 1))
            mask = np.ones((height, width), dtype=bool)
            mask[top:top + hole_height, left:left + hole_width] = False
            if rng.random() < 0.45:
                options = np.flatnonzero(mask)
                mask.reshape(-1)[int(rng.choice(options))] = False
        elif choice == 1:
            start = int(rng.integers(1, width))
            slope = int(rng.choice((-1, 1)))
            steps = np.clip(start + slope * np.arange(height), 1, width - 1)
            mask = columns < steps[:, None]
            if rng.random() < 0.5:
                mask = np.fliplr(mask)
        elif choice == 2:
            positions = np.empty(height, dtype=int)
            positions[0] = int(rng.integers(1, width))
            for index in range(1, height):
                positions[index] = np.clip(positions[index - 1] + int(rng.integers(-2, 3)), 1, width - 1)
            mask = columns < positions[:, None]
            if rng.random() < 0.5:
                mask = np.fliplr(mask)
            if rng.random() < 0.5:
                mask = np.flipud(mask)
        else:
            density = float(rng.uniform(0.30, 0.70))
            count = int(np.clip(round(area * density), 3, area - 4))
            jitter = rng.random((height, width))
            if choice == 3:
                anchor_row = int(rng.integers(0, height))
                anchor_column = int(rng.integers(0, width))
                scores = abs(rows - anchor_row) + abs(columns - anchor_column) + jitter
            elif choice == 4:
                if rng.random() < 0.5:
                    scores = abs(rows - int(rng.integers(0, height))) + jitter
                else:
                    scores = abs(columns - int(rng.integers(0, width))) + jitter
            elif choice == 5:
                diagonal = rows + int(rng.choice((-1, 1))) * columns
                scores = abs(diagonal - int(rng.choice(diagonal.reshape(-1)))) + jitter
            else:
                depth = np.minimum.reduce((rows, height - 1 - rows, columns, width - 1 - columns))
                scores = abs(depth - int(rng.choice(depth.reshape(-1)))) + jitter
            mask = np.zeros((height, width), dtype=bool)
            mask.reshape(-1)[np.argsort(scores.reshape(-1), kind="stable")[:count]] = True
        if 3 <= int(np.count_nonzero(mask)) <= area - 4:
            return mask
    mask = np.zeros((height, width), dtype=bool)
    mask.reshape(-1)[rng.choice(area, size=area // 2, replace=False)] = True
    return mask


def _near_miss(rng, base, one_cell=False, preserve_count=False):
    fitting_positions = np.flatnonzero(~base)
    base_positions = np.flatnonzero(base)
    maximum = min(4, len(base_positions), len(fitting_positions) - 2)
    overlaps = 1 if one_cell or maximum == 1 or rng.random() < 0.55 else int(rng.integers(2, maximum + 1))
    candidate = (~base).copy().reshape(-1)
    candidate[rng.choice(fitting_positions, size=overlaps, replace=False)] = False
    candidate[rng.choice(base_positions, size=overlaps, replace=False)] = True
    if not preserve_count and len(fitting_positions) - overlaps >= 4 and rng.random() < 0.28:
        remaining = np.flatnonzero(candidate & ~base.reshape(-1))
        extra = min(int(rng.integers(1, 3)), len(remaining) - 2)
        candidate[rng.choice(remaining, size=extra, replace=False)] = False
    return candidate.reshape(base.shape)


def _reuse_base(rng, base, candidate):
    result = base.copy().reshape(-1)
    result[int(rng.choice(np.flatnonzero(candidate)))] = True
    result[int(rng.choice(np.flatnonzero(base)))] = False
    return result.reshape(base.shape)


def _pair(base, candidate, colors):
    background, separator, base_color, candidate_color = colors
    height, width = base.shape
    left = np.full((height, width), background, dtype=np.int8)
    right = left.copy()
    left[base] = base_color
    right[candidate] = candidate_color
    divider = np.full((height, 1), separator, dtype=np.int8)
    output = left.copy()
    if not np.any(base & candidate):
        output[candidate] = candidate_color
    return {"input": np.concatenate((left, divider, right), axis=1), "output": output}


def _schedule(rng, count, test=False):
    accepted = int(rng.integers(0, 2)) if test and count == 1 else (count + 1) // 2
    outcomes = [False] * accepted + [True] * (count - accepted)
    return [outcomes[int(index)] for index in rng.permutation(count)]


def _specs(rng, train_count, test_count, fixed_shape, mixed_dimensions):
    anchor_shape = _shape(rng) if mixed_dimensions else fixed_shape
    anchor_base = _sample_base(rng, *anchor_shape, style=int(rng.integers(0, 3)))
    complement = ~anchor_base
    train = [(anchor_base, complement, True),
             (anchor_base.copy(), _near_miss(rng, anchor_base, one_cell=True, preserve_count=True), True)]
    remaining = _schedule(rng, train_count - 2) if train_count > 2 else []
    if train_count >= 4:
        train.append((_reuse_base(rng, anchor_base, complement), complement.copy(), True))
        remaining.remove(True)
    for collision in remaining:
        shape = _shape(rng) if mixed_dimensions else fixed_shape
        base = _sample_base(rng, *shape)
        train.append((base, _near_miss(rng, base) if collision else ~base, False))
    train = [train[int(index)] for index in rng.permutation(len(train))]
    test = []
    for collision in _schedule(rng, test_count, test=True):
        shape = _shape(rng) if mixed_dimensions else fixed_shape
        if mixed_dimensions and shape == anchor_shape:
            shape = (shape[0], 4 + (shape[1] - 3) % 9)
        base = _sample_base(rng, *shape)
        test.append((base, _near_miss(rng, base, one_cell=True) if collision else ~base, False))
    return train, test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    fixed_shape = _shape(rng)
    mixed_dimensions = seed % 4 != 0
    mixed_roles = seed % 3 != 0
    fixed_roles = tuple(int(value) for value in rng.permutation(10)[:3])
    train_specs, test_specs = _specs(rng, train_count, test_count, fixed_shape, mixed_dimensions)
    anchor_roles = tuple(int(value) for value in rng.permutation(10)[:3]) if mixed_roles else fixed_roles
    anchor_colors = [value for value in rng.permutation(10) if value not in anchor_roles]
    anchor_color = None
    for color in anchor_colors:
        if all(not _source_copy(_pair(base, candidate, (*anchor_roles, int(color))))
               for base, candidate, matched in train_specs if matched):
            anchor_color = int(color)
            break
    assert anchor_color is not None
    seen = set()
    seen_layouts = set()
    used_train_candidate_colors = set()
    puzzle = {"train": [], "test": []}
    for split, specs in (("train", train_specs), ("test", test_specs)):
        for index, (base, candidate, matched) in enumerate(specs):
            for _ in range(64):
                if matched:
                    roles = anchor_roles
                    candidate_color = anchor_color
                else:
                    roles = tuple(int(value) for value in rng.permutation(10)[:3]) if mixed_roles else fixed_roles
                    available = [value for value in range(10) if value not in roles]
                    fresh = [value for value in available if value not in used_train_candidate_colors]
                    if split == "test" and index == 0 and seed % 5 == 1 and fresh:
                        candidate_color = int(rng.choice(fresh))
                    else:
                        candidate_color = int(rng.choice(available))
                pair = _pair(base, candidate, (*roles, candidate_color))
                key = _array_key(pair["input"])
                layout = (_mask_key(base), _mask_key(candidate))
                if key not in seen and layout not in seen_layouts and not _source_copy(pair):
                    seen.add(key)
                    seen_layouts.add(layout)
                    puzzle[split].append(pair)
                    if split == "train":
                        used_train_candidate_colors.add(candidate_color)
                    break
                base = _sample_base(rng, *base.shape)
                candidate = _near_miss(rng, base, one_cell=True) if np.any(base & candidate) else ~base
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    height, input_width = input_grid.shape
    assert input_width >= 9 and input_width % 2 == 1
    width = (input_width - 1) // 2
    assert 4 <= height <= 10 and 4 <= width <= 12
    assert output_grid.shape == (height, width)
    left, divider, right = input_grid[:, :width], input_grid[:, width], input_grid[:, width + 1:]
    assert np.all(divider == divider[0])
    left_colors = set(int(value) for value in np.unique(left))
    right_colors = set(int(value) for value in np.unique(right))
    assert len(left_colors) == len(right_colors) == 2
    shared = left_colors & right_colors
    assert len(shared) == 1
    background = shared.pop()
    base_color = (left_colors - {background}).pop()
    candidate_color = (right_colors - {background}).pop()
    assert len({background, base_color, candidate_color, int(divider[0])}) == 4
    base, candidate = left != background, right != background
    overlap = int(np.count_nonzero(base & candidate))
    fitting = int(np.count_nonzero(~base & candidate))
    assert int(np.count_nonzero(base)) >= 3
    assert int(np.count_nonzero(~base)) >= 4
    assert int(np.count_nonzero(candidate)) >= 3
    expected = left.copy()
    if overlap == 0:
        assert np.array_equal(candidate, ~base)
        expected[candidate] = candidate_color
    else:
        assert fitting >= 2
        assert int(np.count_nonzero(base)) + int(np.count_nonzero(candidate)) <= height * width
    assert np.array_equal(output_grid, expected)
    return {"height": height, "width": width, "background": background,
            "separator": int(divider[0]), "base_color": base_color,
            "candidate_color": candidate_color, "base": base, "candidate": candidate,
            "overlap": overlap, "fitting": fitting}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    details = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _array_key(pair["input"])
            pair_key = (input_key, _array_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert not _source_copy(pair)
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            details[split].append(_inspect(pair))
    train, test = details["train"], details["test"]
    assert any(item["overlap"] == 0 for item in train)
    assert any(item["overlap"] > 0 and item["fitting"] >= 2 for item in train)
    assert any(item["overlap"] == 1 for item in train)
    if len(test) >= 2:
        assert any(item["overlap"] == 0 for item in test)
        assert any(item["overlap"] > 0 for item in test)
    assert all(item["overlap"] in (0, 1) for item in test)
    train_layouts = {(_mask_key(item["base"]), _mask_key(item["candidate"])) for item in train}
    assert all((_mask_key(item["base"]), _mask_key(item["candidate"])) not in train_layouts for item in test)


def _puzzle_key(puzzle):
    return tuple((split, tuple((_array_key(pair["input"]), _array_key(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    groups = []
    while unseen:
        origin = unseen.pop()
        group = {origin}
        pending = [origin]
        while pending:
            row, column = pending.pop()
            for neighbor in ((row - 1, column), (row + 1, column),
                             (row, column - 1), (row, column + 1)):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    group.add(neighbor)
                    pending.append(neighbor)
        groups.append(group)
    return groups


def _prediction_from_input(input_grid):
    width = (input_grid.shape[1] - 1) // 2
    left, right = input_grid[:, :width], input_grid[:, width + 1:]
    background = (set(np.unique(left)) & set(np.unique(right))).pop()
    candidate_color = (set(np.unique(right)) - {background}).pop()
    base = left != background
    candidate = right != background
    answer = left.copy()
    if not np.any(base & candidate):
        answer[candidate] = candidate_color
    return answer


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    cases = [(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None), (7, 3)]
    for index, (train_count, test_count) in enumerate(cases):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(1000 + index, train_count, test_count))
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(99, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    all_details = []
    mixed_dimensions = 0
    mixed_roles = 0
    combined_variation = 0
    constant_both = 0
    matched_comparisons = 0
    reused_candidates = 0
    fresh_test_colors = 0
    rich_train_palettes = 0
    discarded_separate_pieces = 0
    frame_holes = 0
    stepped_rows = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_inspect(pair) for pair in puzzle["train"]]
        test = [_inspect(pair) for pair in puzzle["test"]]
        details = train + test
        all_details.extend(details)
        shapes = {(item["height"], item["width"]) for item in details}
        roles = {(item["background"], item["separator"], item["base_color"]) for item in details}
        mixed_dimensions += len(shapes) > 1
        mixed_roles += len(roles) > 1
        combined_variation += len(shapes) > 1 and len(roles) > 1
        constant_both += len(shapes) == 1 and len(roles) == 1
        rich_train_palettes += len({item["candidate_color"] for item in train}) >= 4
        fresh_test_colors += bool({item["candidate_color"] for item in test} -
                                  {item["candidate_color"] for item in train})
        matched_comparisons += any(
            _mask_key(first["base"]) == _mask_key(second["base"])
            and first["background"] == second["background"]
            and first["separator"] == second["separator"]
            and first["base_color"] == second["base_color"]
            and first["candidate_color"] == second["candidate_color"]
            and first["overlap"] == 0 and second["overlap"] == 1
            for first in train for second in train
        )
        reused_candidates += any(
            _mask_key(first["candidate"]) == _mask_key(second["candidate"])
            and _mask_key(first["base"]) != _mask_key(second["base"])
            for first in train for second in train
        )
        for pair in puzzle["test"]:
            assert np.array_equal(_prediction_from_input(pair["input"].copy()), pair["output"])
        for item in details:
            base, candidate = item["base"], item["candidate"]
            if item["overlap"]:
                groups = _components(candidate)
                discarded_separate_pieces += any(
                    any(base[row, column] for row, column in group)
                    for group in groups
                ) and any(
                    all(not base[row, column] for row, column in group)
                    for group in groups
                )
            frame_holes += (np.all(base[0]) and np.all(base[-1]) and
                            np.all(base[:, 0]) and np.all(base[:, -1]) and
                            np.any(~base[1:-1, 1:-1]))
            width = base.shape[1]
            row_counts = [int(np.count_nonzero(row)) for row in base]
            prefix = all(np.array_equal(row, np.arange(width) < row.sum()) for row in base)
            suffix = all(np.array_equal(row, np.arange(width) >= width - row.sum()) for row in base)
            stepped_rows += (prefix or suffix) and len(set(row_counts)) >= 3
    assert len({(item["height"], item["width"]) for item in all_details}) >= 20
    assert len({(item["background"], item["separator"], item["base_color"]) for item in all_details}) >= 20
    assert mixed_dimensions >= 20 and mixed_roles >= 20 and combined_variation >= 15
    assert constant_both >= 3
    assert matched_comparisons == 42 and reused_candidates >= 10
    assert fresh_test_colors >= 15 and rich_train_palettes >= 10
    assert discarded_separate_pieces >= 20
    assert frame_holes >= 5 and stepped_rows >= 30
    assert sum(item["overlap"] == 0 for item in all_details) > 40
    assert sum(item["overlap"] == 1 for item in all_details) > 40
    assert sum(item["overlap"] > 1 for item in all_details) > 5
