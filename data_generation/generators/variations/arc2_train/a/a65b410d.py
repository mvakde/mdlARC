import numpy as np


# (height, width, marker row, marker length); only the exact source coloring
# and left alignment are excluded, not these useful canvas geometries.
_SOURCE_GEOMETRIES = frozenset(
    {(8, 9, 3, 3), (7, 7, 3, 2), (7, 9, 2, 4), (9, 9, 2, 5)}
)
_SOURCE_ROLES = (0, 2, 3, 1)
_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))


def _counts(seed, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not low <= int(value) <= high:
                raise ValueError(f"{name} must be in {low}..{high}")
    if num_train is None and num_test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if num_train is None:
        test = int(num_test)
        low = max(2, test + 1)
        return low + seed % (11 - low), test
    if num_test is None:
        train = int(num_train)
        return train, 1 + seed % min(4, train - 1)
    train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return (grid.shape, grid.tobytes())


def _read_marker(grid):
    height, width = grid.shape
    assert height >= 4 and width >= 4
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2 and counts[0] != counts[1]
    marker = int(values[np.argmin(counts)])
    background = int(values[np.argmax(counts)])
    locations = np.argwhere(grid == marker)
    rows = np.unique(locations[:, 0])
    assert len(rows) == 1
    row = int(rows[0])
    columns = np.sort(locations[:, 1])
    length = len(columns)
    assert 1 <= length < width
    if np.array_equal(columns, np.arange(length)):
        anchor = "left"
    else:
        assert np.array_equal(columns, np.arange(width - length, width))
        anchor = "right"
    return background, marker, row, length, anchor


def _paint(height, width, row, length, anchor, roles):
    background, marker, upper, lower = roles
    source = np.full((height, width), background, dtype=np.int8)
    target = source.copy()

    def fill(grid, at_row, amount, color):
        if anchor == "left":
            grid[at_row, :amount] = color
        else:
            grid[at_row, width - amount:] = color

    fill(source, row, length, marker)
    fill(target, row, length, marker)
    for at_row in range(row):
        fill(target, at_row, min(width, length + row - at_row), upper)
    for at_row in range(row + 1, min(height, row + length)):
        fill(target, at_row, length - (at_row - row), lower)
    return source, target


def _length_plan(rng, train_count, test_count, shift):
    if shift == 0:  # The source-style length extrapolation remains present.
        train = [int(rng.integers(3, 6)) for _ in range(train_count)]
        test = [int(rng.integers(6, 10)) for _ in range(test_count)]
    elif shift == 1:  # Reuse a learned length on a new canvas.
        train = [int(rng.integers(3, 9)) for _ in range(train_count)]
        test = [train[i % train_count] for i in range(test_count)]
    elif shift == 2:  # The largest segment belongs to training.
        train = [int(rng.integers(8, 10))]
        train += [int(rng.integers(3, 8)) for _ in range(train_count - 1)]
        test = [int(rng.integers(2, 8)) for _ in range(test_count)]
    else:  # Shared sizes and a less predictable ordering.
        train = [int(rng.integers(3, 9)) for _ in range(train_count)]
        test = [train[int(rng.integers(0, train_count))]]
        test += [int(rng.integers(1, 10)) for _ in range(test_count - 1)]
    # A one-cell marker is a useful no-lower-color boundary example, but it
    # never replaces either of the two rule-teaching training examples.
    if train_count >= 4 and shift != 0 and int(rng.integers(0, 3)) == 0:
        train[-1] = 1
    return train, test


def _geometry(rng, length, kind, boundary_variant=0):
    if kind == "teaching":
        row = int(rng.integers(2, 7))
        below = int(rng.integers(max(2, length - 1), 12))
    elif kind == "boundary":
        if boundary_variant == 0:
            row = 0
            below = int(rng.integers(max(3, length - 1), 15))
        elif boundary_variant == 1:
            row = 1
            below = int(rng.integers(max(2, length - 1), 15))
        elif boundary_variant == 2:
            row = int(rng.integers(3, 13))
            below = length - 1  # The one-cell tip touches the bottom.
        else:
            row = int(rng.integers(3, 13))
            below = int(rng.integers(0, max(1, length - 1)))
    else:
        row = int(rng.integers(0, 13))
        below = int(rng.integers(0, 16))
        if int(rng.integers(0, 4)) == 0:
            below = max(0, length - 1 + int(rng.integers(-2, 3)))

    if length == 1 and row == 0:
        row = 1
    below = max(below, 3 - row)
    height = row + below + 1
    assert 4 <= height <= 30

    # Margin is sampled after row and length, so clipping and right-side
    # background occur at both small and large marker sizes.
    max_margin = 30 - length
    band = int(rng.integers(0, 3))
    if band == 0 and row >= 1:
        margin = int(rng.integers(1, min(row, max_margin) + 1))
    elif band == 1:
        low = min(max_margin, row + 1)
        high = min(max_margin, row + 6)
        margin = int(rng.integers(low, high + 1))
    else:
        low = min(max_margin, row + 7)
        margin = int(rng.integers(low, max_margin + 1))
    width = length + margin
    width = max(4, width)
    return height, width, row


def _source_copy(info, roles):
    background, marker, row, length, anchor, height, width = info
    return (anchor == "left" and (height, width, row, length) in _SOURCE_GEOMETRIES
            and roles == _SOURCE_ROLES)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)

    # The two output-only roles and the orientation convention are learned
    # from training. In variable-color puzzles, each pair's background and
    # marker are visible in its own input.
    upper, lower = (int(x) for x in rng.choice(10, size=2, replace=False))
    available = [x for x in range(10) if x not in (upper, lower)]
    style = seed % 5
    variable_colors = style in (1, 4)
    mixed_anchors = style in (3, 4)
    default_anchor = "right" if style == 2 else "left"
    palette_order = [int(x) for x in rng.permutation(available)]
    fixed_colors = (palette_order[0], palette_order[1])
    train_lengths, test_lengths = _length_plan(rng, train_count, test_count, seed % 4)
    if seed % 17 == 0 and default_anchor == "left":
        train_lengths[0] = 3

    puzzle = {"train": [], "test": []}
    seen_geometry = set()
    seen_inputs = set()

    for split, lengths in (("train", train_lengths), ("test", test_lengths)):
        for index, length in enumerate(lengths):
            if mixed_anchors:
                anchor = ("left", "right")[index % 2] if split == "train" else (
                    "left", "right")[int(rng.integers(0, 2))]
            else:
                anchor = default_anchor

            if variable_colors:
                if split == "train" and index < 2:
                    background, marker = palette_order[2 * index:2 * index + 2]
                else:
                    background, marker = (int(x) for x in rng.choice(available, size=2, replace=False))
            else:
                background, marker = fixed_colors
            roles = (background, marker, upper, lower)

            if split == "train" and (index == 0 or (index == 1 and
                                                 (variable_colors or mixed_anchors))):
                kind = "teaching"
            elif split == "train" and index == 1:
                kind = "boundary"
            elif int(rng.integers(0, 4)) == 0:
                kind = "boundary"
            else:
                kind = "ordinary"

            for attempt in range(512):
                variant = (seed + index + attempt // 24) % 4
                if (attempt == 0 and split == "train" and index == 0
                        and seed % 17 == 0 and anchor == "left"):
                    height, width, row = 8, 9, 3
                else:
                    height, width, row = _geometry(rng, length, kind, variant)
                geometry = (height, width, row, length, anchor)
                if geometry in seen_geometry:
                    continue
                source, target = _paint(height, width, row, length, anchor, roles)
                info = (background, marker, row, length, anchor, height, width)
                if _source_copy(info, roles) or _key(source) in seen_inputs:
                    continue
                seen_geometry.add(geometry)
                seen_inputs.add(_key(source))
                puzzle[split].append({"input": source, "output": target})
                break
            else:
                raise ValueError("could not construct a distinct geometry")

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs, seen_pairs, geometries = set(), set(), set()
    records = {"train": [], "test": []}
    observed_upper, observed_lower = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 4 <= grid.shape[0] <= 30 and 4 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert source.shape == target.shape and not np.array_equal(source, target)
            input_key = _key(source)
            pair_key = (input_key, _key(target))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            background, marker, row, length, anchor = _read_marker(source)
            height, width = source.shape
            geometry = (height, width, row, length, anchor)
            assert geometry not in geometries
            geometries.add(geometry)
            info = (background, marker, row, length, anchor, height, width)
            assert np.array_equal(source[row], target[row])
            if split == "train" and row > 0:
                observed_upper.add(int(target[0, 0 if anchor == "left" else -1]))
            if split == "train" and row + 1 < height and length > 1:
                observed_lower.add(int(target[row + 1, 0 if anchor == "left" else -1]))
            records[split].append((pair, info))

    assert len(observed_upper) == 1 and len(observed_lower) == 1
    upper, lower = next(iter(observed_upper)), next(iter(observed_lower))
    assert upper != lower

    train_colors = []
    train_anchors = set()
    teaching_count = 0
    for split in ("train", "test"):
        for pair, info in records[split]:
            background, marker, row, length, anchor, height, width = info
            roles = (background, marker, upper, lower)
            assert len(set(roles)) == 4
            assert not _source_copy(info, roles)
            expected_source, expected_target = _paint(height, width, row, length, anchor, roles)
            assert np.array_equal(pair["input"], expected_source)
            assert np.array_equal(pair["output"], expected_target)
            if split == "train":
                train_colors.append((background, marker))
                train_anchors.add(anchor)
                if row >= 2 and length >= 3 and height - row - 1 >= 2:
                    teaching_count += 1
            else:
                assert anchor in train_anchors

    assert teaching_count >= 1
    backgrounds = {x[0] for x in train_colors}
    markers = {x[1] for x in train_colors}
    for _, info in records["test"]:
        background, marker = info[:2]
        if len(backgrounds) == 1 and len(markers) == 1:
            assert (background, marker) == train_colors[0]
        else:
            assert len(backgrounds) >= 2 and len(markers) >= 2


def _puzzle_signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[split])
                 for split in ("train", "test"))


def _predict_from_training(train_pairs, test_input):
    """Infer the two output colors from demonstrations, then solve an input."""
    upper_values, lower_values, taught_anchors = set(), set(), set()
    for pair in train_pairs:
        _, _, row, length, anchor = _read_marker(pair["input"])
        edge = 0 if anchor == "left" else -1
        taught_anchors.add(anchor)
        if row > 0:
            upper_values.add(int(pair["output"][row - 1, edge]))
        if row + 1 < pair["input"].shape[0] and length > 1:
            lower_values.add(int(pair["output"][row + 1, edge]))
    assert len(upper_values) == len(lower_values) == 1
    upper, lower = next(iter(upper_values)), next(iter(lower_values))
    _, _, marker_row, length, anchor = _read_marker(test_input)
    assert anchor in taught_anchors
    result = test_input.copy()
    height, width = result.shape
    for row in range(height):
        if row < marker_row:
            size, color = min(width, length + marker_row - row), upper
        elif row > marker_row:
            size, color = max(0, length - (row - marker_row)), lower
        else:
            continue
        if size:
            if anchor == "left":
                result[row, :size] = color
            else:
                result[row, width - size:] = color
    return result


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    styles, shifts, shapes, layouts = set(), set(), set(), set()
    boundary = set()
    saturation = {"train": set(), "test": set()}
    overlap = set()
    source_like_geometry = False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training_copies = [
            {"input": pair["input"].copy(), "output": pair["output"].copy()}
            for pair in puzzle["train"]
        ]
        for pair in puzzle["test"]:
            prediction = _predict_from_training(training_copies, pair["input"].copy())
            assert np.array_equal(prediction, pair["output"])
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_signature(puzzle))
        styles.add(seed % 5)
        shifts.add(seed % 4)
        train_lengths, test_lengths = [], []
        train_heights, test_heights = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                bg, marker, row, length, anchor = _read_marker(pair["input"])
                height, width = pair["input"].shape
                shapes.add((height, width))
                layouts.add((height, width, row, length, anchor))
                saturation[split].add(length + row >= width and row > 0)
                source_like_geometry |= (height, width, row, length) in _SOURCE_GEOMETRIES
                if row == 0:
                    boundary.add("absent upper")
                if row == 1:
                    boundary.add("shallow upper")
                if height - row - 1 == length - 1:
                    boundary.add("tip on edge")
                if height - row - 1 < length - 1:
                    boundary.add("cropped lower")
                if row >= 5 and length >= 6 and height >= 15:
                    boundary.add("long marker down in large grid")
                if split == "train":
                    train_lengths.append(length)
                    train_heights.append(height)
                else:
                    test_lengths.append(length)
                    test_heights.append(height)
        overlap.add((bool(set(train_lengths) & set(test_lengths)),
                     max(train_lengths) > max(test_lengths),
                     max(test_lengths) > max(train_lengths),
                     bool(set(train_heights) & set(test_heights))))

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(shapes) >= 80 and len(layouts) >= 150
    assert styles == set(range(5)) and shifts == set(range(4))
    assert saturation["train"] == {False, True}
    assert saturation["test"] == {False, True}
    assert {"absent upper", "shallow upper", "tip on edge", "cropped lower",
            "long marker down in large grid"} <= boundary
    assert any(x[0] for x in overlap) and any(x[1] for x in overlap)
    assert any(x[2] for x in overlap) and any(x[3] for x in overlap)
    assert source_like_geometry

    for train, test in ((2, 1), (10, 1), (10, 4), (3, 2), (7, 3)):
        puzzle = generate_puzzle(200 + train + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        validate_generated_puzzle(puzzle)
    for seed, train, test in ((301, None, 1), (302, None, 4),
                              (303, 2, None), (304, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        validate_generated_puzzle(puzzle)
    assert _puzzle_signature(generate_puzzle(98765, 7, 3)) == _puzzle_signature(
        generate_puzzle(98765, 7, 3))
    assert len({_puzzle_signature(generate_puzzle(seed, 6, 2))
                for seed in range(500, 510)}) == 10
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(999, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
