import numpy as np


def _count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    low, high = (2, 10) if name == "num_train" else (1, 4)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train, num_test = _count(num_train, "num_train"), _count(num_test, "num_test")
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


def _cells(row, col):
    return ((row, col), (row, col + 1), (row, col + 2),
            (row + 1, col), (row + 1, col + 2))


def _arches(grid, background):
    height, width = grid.shape
    found = []
    for row in range(height - 1):
        for col in range(width - 2):
            color = int(grid[row, col])
            if color == background:
                continue
            if (grid[row, col + 1] == color and grid[row, col + 2] == color
                    and grid[row + 1, col] == color and grid[row + 1, col + 2] == color
                    and grid[row + 1, col + 1] == background):
                found.append((row, col, color))
    return found


def _draw(height, width, background, anchors, colors):
    grid = np.full((height, width), background, dtype=np.int8)
    occupied = set()
    for (row, col), color in zip(anchors, colors):
        cells = _cells(row, col)
        if occupied.intersection(cells):
            raise ValueError("overlapping arches")
        occupied.update(cells)
        for cell in cells:
            grid[cell] = color
    return grid


def _output(before, background, marker):
    after = before.copy()
    for _, col, _ in _arches(before, background):
        after[-1, col + 1] = marker
    return after


def _dimensions(rng, count, regime):
    aspect = int(rng.integers(0, 4))
    limit = 13 if regime == "train" else 20
    min_width = max(5, count + 3)
    if aspect == 0:
        height = int(rng.integers(5, 11))
        width = int(rng.integers(min_width, max(min_width + 1, 12)))
    elif aspect == 1:
        height = int(rng.integers(5, 12))
        width = int(rng.integers(max(min_width, height + 2, 11), limit + 5))
    elif aspect == 2:
        width = int(rng.integers(min_width, 13))
        height = int(rng.integers(max(10, width + 2), limit + 5))
    else:
        side = int(rng.integers(max(7, min_width), limit + 1))
        height = side
        width = side + int(rng.integers(-1, 2))
    return height, width


def _weight(style, row, col, anchors):
    if not anchors:
        return 1.0
    distances = [(abs(row - old_row), abs(col - old_col))
                 for old_row, old_col in anchors]
    if style == 1 and any(dr <= 2 and dc <= 4 for dr, dc in distances):
        return 18.0
    if style == 2 and any(dr == 0 for dr, _ in distances):
        return 24.0
    if style == 3 and any(dr >= 2 and dc <= 3 for dr, dc in distances):
        return 20.0
    if style == 4:
        if any(dr <= 2 and dc <= 4 for dr, dc in distances):
            return 12.0
        if any(dr >= 3 and dc >= 3 for dr, dc in distances):
            return 7.0
    return 1.0


def _layout(rng, height, width, background, colors):
    count = len(colors)
    coordinates = [(row, col) for row in range(height - 2)
                   for col in range(width - 2)]
    style = int(rng.integers(0, 6))
    if style == 5 and count >= 3 and height >= count + 2 and width >= 3 * count:
        row = int(rng.integers(0, height - count - 1))
        col = int(rng.integers(0, width - 3 * count + 1))
        anchors = [(row + index, col + 3 * index) for index in range(count)]
        grid = _draw(height, width, background, anchors, colors)
        if {(r, c) for r, c, _ in _arches(grid, background)} == set(anchors):
            return grid
    for _ in range(80):
        anchors, occupied, centers = [], set(), set()
        for _ in range(count):
            candidates, weights = [], []
            for row, col in coordinates:
                if col + 1 in centers or occupied.intersection(_cells(row, col)):
                    continue
                candidates.append((row, col))
                weights.append(_weight(style, row, col, anchors))
            if not candidates:
                break
            weights = np.asarray(weights, dtype=float)
            anchor = candidates[int(rng.choice(len(candidates), p=weights / weights.sum()))]
            anchors.append(anchor)
            centers.add(anchor[1] + 1)
            occupied.update(_cells(*anchor))
        if len(anchors) != count:
            continue
        grid = _draw(height, width, background, anchors, colors)
        if {(r, c) for r, c, _ in _arches(grid, background)} == set(anchors):
            return grid
    # Bounded fallback for crowded small grids; all columns remain distinct.
    fallback_height = max(height, 4 * count + 2)
    fallback_width = max(width, count + 3)
    anchors = [(4 * index, index) for index in range(count)]
    return _draw(fallback_height, fallback_width, background, anchors, colors)


def _pair(rng, marker, regime, index):
    if regime == "train" and index == 0:
        count = 1
    elif regime == "train" and index == 1:
        count = int(rng.integers(2, 5))
    elif regime == "test":
        count = int(rng.choice([1, 2, 3, 3, 4, 5, 6]))
    else:
        count = int(rng.integers(1, 7))
    background = int(rng.choice([color for color in range(10) if color != marker]))
    palette = [color for color in range(10) if color not in (marker, background)]
    distinct = (2 if regime == "train" and index == 1 else
                int(rng.integers(1, min(5, count) + 1)))
    chosen = [int(color) for color in rng.choice(palette, size=distinct, replace=False)]
    colors = chosen + [chosen[int(rng.integers(distinct))] for _ in range(count - distinct)]
    rng.shuffle(colors)
    height, width = _dimensions(rng, count, regime)
    before = _layout(rng, height, width, background, colors)
    return before, _output(before, background, marker)


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(value) for value in grid.ravel()))


def _pair_key(before, after):
    return (_grid_key(before), _grid_key(after))


_SOURCE_ROWS = (
    (("03330", "03030", "00000", "00000", "00000"),
     ("03330", "03030", "00000", "00000", "00400")),
    (("06660000", "06060000", "00000666", "00000606", "00000000",
      "00000000", "00000000", "00000000"),
     ("06660000", "06060000", "00000666", "00000606", "00000000",
      "00000000", "00000000", "00400040")),
    (("0000000", "0888000", "0808666", "0000606", "0000000"),
     ("0000000", "0888000", "0808666", "0000606", "0040040")),
    (("05550000000", "05050888000", "00000808333", "00000000303",
      "00000000000", "00000000000", "00000000000"),
     ("05550000000", "05050888000", "00000808333", "00000000303",
      "00000000000", "00000000000", "00400040040")),
)


def _rows_key(rows):
    return ((len(rows), len(rows[0])),
            tuple(int(character) for row in rows for character in row))


_SOURCE_KEYS = {(_rows_key(before), _rows_key(after)) for before, after in _SOURCE_ROWS}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    marker = int(rng.integers(0, 10))
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for regime, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(100):
                before, after = _pair(pair_rng, marker, regime, index)
                input_key, pair_key = _grid_key(before), _pair_key(before, after)
                if (input_key not in seen_inputs and pair_key not in seen_pairs
                        and pair_key not in _SOURCE_KEYS):
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            puzzle[regime].append({"input": before, "output": after})
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    before, after = pair["input"], pair["output"]
    assert before.shape == after.shape
    background = int(before[-1, 0])
    assert np.all(before[-1] == background)
    arches = _arches(before, background)
    assert arches
    cells = [cell for row, col, _ in arches for cell in _cells(row, col)]
    assert len(cells) == len(set(cells))
    assert set(map(tuple, np.argwhere(before != background))) == set(cells)
    centers = [col + 1 for _, col, _ in arches]
    assert len(centers) == len(set(centers))
    changed = set(map(tuple, np.argwhere(before != after)))
    assert changed == {(before.shape[0] - 1, center) for center in centers}
    markers = {int(after[cell]) for cell in changed}
    assert len(markers) == 1
    marker = markers.pop()
    assert marker != background and not np.any(before == marker)
    expected = before.copy()
    expected[-1, centers] = marker
    assert np.array_equal(after, expected)
    return background, marker, arches


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, markers = set(), set(), set()
    train_counts = []
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(
                pair["input"], pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            _, marker, arches = _inspect_pair(pair)
            markers.add(marker)
            if regime == "train":
                train_counts.append(len(arches))
    assert len(markers) == 1
    assert train_counts[0] == 1 and train_counts[1] >= 2
    second = puzzle["train"][1]["input"]
    second_background = int(second[-1, 0])
    assert len({color for _, _, color in _arches(second, second_background)}) >= 2


def _puzzle_key(puzzle):
    return tuple((regime, tuple(_pair_key(pair["input"], pair["output"])
                                  for pair in puzzle[regime]))
                 for regime in ("train", "test"))


def _predict_from_training(train, test_input):
    learned = {int(pair["output"][tuple(cell)]) for pair in train
               for cell in np.argwhere(pair["input"] != pair["output"])}
    assert len(learned) == 1
    marker = learned.pop()
    background = int(test_input[-1, 0])
    assert np.all(test_input[-1] == background)
    prediction = test_input.copy()
    for row in range(test_input.shape[0] - 1):
        for col in range(test_input.shape[1] - 2):
            color = int(test_input[row, col])
            if (color != background
                    and np.all(test_input[row, col:col + 3] == color)
                    and test_input[row + 1, col] == color
                    and test_input[row + 1, col + 1] == background
                    and test_input[row + 1, col + 2] == color):
                prediction[-1, col + 1] = marker
    return prediction


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    roles, dimensions, layouts, aspect_classes, color_counts, combinations = (
        set(), set(), set(), set(), set(), set())
    train_many = test_few = test_many = same_color_multi = 0
    mixed_background_puzzles = touching = same_row = reversed_order = zigzag = 0
    review = {name: 0 for name in (
        "staircase", "irregular_gaps", "nonperiodic_colors", "same_color_touch",
        "same_row_same_color_touch", "touch_and_isolated", "count_increase",
        "new_test_background", "compact_test", "test_count_varies")}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        pair_backgrounds = set()
        train_backgrounds = {int(pair["input"][-1, 0]) for pair in puzzle["train"]}
        test_backgrounds = {int(pair["input"][-1, 0]) for pair in puzzle["test"]}
        review["new_test_background"] += bool(test_backgrounds - train_backgrounds)
        train_arch_counts = [len(_inspect_pair(pair)[2]) for pair in puzzle["train"]]
        test_arch_counts = [len(_inspect_pair(pair)[2]) for pair in puzzle["test"]]
        review["count_increase"] += max(test_arch_counts) > max(train_arch_counts)
        review["test_count_varies"] += len(set(test_arch_counts)) > 1
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                background, marker, arches = _inspect_pair(pair)
                if regime == "test":
                    assert np.array_equal(_predict_from_training(
                        puzzle["train"], pair["input"]), pair["output"])
                pair_backgrounds.add(background)
                roles.add((background, marker))
                shape = pair["input"].shape
                dimensions.add(shape)
                layouts.add((shape, tuple(sorted((r, c) for r, c, _ in arches))))
                height, width = shape
                aspect = "square" if abs(width - height) <= 1 else (
                    "wide" if width > height else "tall")
                aspect_classes.add((regime, aspect))
                distinct = len({color for _, _, color in arches})
                color_counts.add((regime, distinct))
                combinations.add((regime, aspect, len(arches), distinct))
                if regime == "train" and len(arches) >= 3:
                    train_many += 1
                if regime == "test":
                    test_few += len(arches) <= 2
                    test_many += len(arches) >= 3
                    review["compact_test"] += height <= 10 and width <= 12
                if len(arches) >= 2 and distinct == 1:
                    same_color_multi += 1
                ordered = sorted(arches, key=lambda item: item[1])
                rows = [item[0] for item in ordered]
                columns = [item[1] for item in ordered]
                same_row += len(rows) >= 2 and len(set(rows)) == 1
                reversed_order += len(rows) >= 2 and rows[0] > rows[-1]
                zigzag += len(rows) >= 3 and any(
                    (rows[i] - rows[i - 1]) * (rows[i + 1] - rows[i]) < 0
                    for i in range(1, len(rows) - 1))
                touching += any(
                    abs(first[0] - second[0]) <= 2
                    and abs(first[1] - second[1]) <= 3
                    for i, first in enumerate(arches) for second in arches[i + 1:])
                if regime == "test" and len(arches) >= 2:
                    gaps = [right - left for left, right in zip(columns, columns[1:])]
                    review["irregular_gaps"] += len(gaps) >= 2 and len(set(gaps)) > 1
                    review["staircase"] += len(arches) >= 3 and all(
                        ordered[i + 1][0] == ordered[i][0] + 1
                        and ordered[i + 1][1] == ordered[i][1] + 3
                        for i in range(len(ordered) - 1))
                    sequence = [item[2] for item in ordered]
                    review["nonperiodic_colors"] += (
                        len(sequence) >= 4 and 1 < len(set(sequence)) < len(sequence)
                        and not any(all(sequence[i] == sequence[i % period]
                                        for i in range(len(sequence)))
                                    for period in range(1, len(sequence) // 2 + 1)))
                    adjacent = []
                    for i, first in enumerate(arches):
                        first_cells = set(_cells(first[0], first[1]))
                        for second in arches[i + 1:]:
                            second_cells = set(_cells(second[0], second[1]))
                            if any((r + dr, c + dc) in second_cells
                                   for r, c in first_cells
                                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                                adjacent.append((first, second))
                    review["same_color_touch"] += any(
                        first[2] == second[2] for first, second in adjacent)
                    review["same_row_same_color_touch"] += any(
                        first[2] == second[2] and first[0] == second[0]
                        for first, second in adjacent)
                    touched = {item for group in adjacent for item in group}
                    review["touch_and_isolated"] += bool(adjacent) and len(touched) < len(arches)
        mixed_background_puzzles += len(pair_backgrounds) >= 2
    assert len(roles) >= 15 and len(dimensions) >= 35 and len(layouts) >= 100
    assert train_many >= 8 and test_few >= 5 and test_many >= 15
    assert mixed_background_puzzles >= 25
    assert same_color_multi >= 8 and touching >= 8
    assert same_row >= 3 and reversed_order >= 5 and zigzag >= 5
    assert all(("test", aspect) in aspect_classes
               for aspect in ("square", "wide", "tall"))
    assert all(("test", count) in color_counts for count in (1, 2, 3))
    assert len(combinations) >= 25
    assert review["staircase"] >= 1 and review["irregular_gaps"] >= 10
    assert review["nonperiodic_colors"] >= 5
    assert review["same_color_touch"] >= 5
    assert review["same_row_same_color_touch"] >= 1
    assert review["touch_and_isolated"] >= 5
    assert review["count_increase"] >= 3 and review["new_test_background"] >= 10
    assert review["compact_test"] >= 5 and review["test_count_varies"] >= 10
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (6, 3)):
        seed = 1000 + train_count * 10 + test_count
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, train_count, test_count))
    for index, (train_count, test_count) in enumerate(
            ((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1200 + index, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
    assert _puzzle_key(generate_puzzle(919, 6, 3)) != _puzzle_key(generate_puzzle(920, 6, 3))
