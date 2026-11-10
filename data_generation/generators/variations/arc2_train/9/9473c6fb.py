import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "7797777777",
            "7777677777",
            "7777777477",
            "7797777777",
            "7777677777",
            "7797777777",
        ),
        (
            "7727777777",
            "7777877777",
            "7777777577",
            "7727777777",
            "7777877777",
            "7757777777",
        ),
    ),
    (
        (
            "67777777",
            "77777977",
            "77777777",
            "77677777",
            "77779777",
            "77777717",
            "77776777",
            "77177777",
        ),
        (
            "27777777",
            "77777877",
            "77777777",
            "77577777",
            "77772777",
            "77777787",
            "77775777",
            "77277777",
        ),
    ),
    (
        (
            "7777777777",
            "9777777777",
            "7767797171",
            "7777777777",
            "7177979767",
            "7777777777",
        ),
        (
            "7777777777",
            "2777777777",
            "7757787275",
            "7777777777",
            "7877275787",
            "7777777777",
        ),
    ),
    (
        (
            "777777777777",
            "677777774777",
            "772777777777",
            "777777777277",
            "777777777777",
            "777772777707",
            "747777777777",
            "777777777777",
            "777677707776",
        ),
        (
            "777777777777",
            "277777772777",
            "775777777777",
            "777777777877",
            "777777777777",
            "777778777757",
            "787777777777",
            "777777777777",
            "777277757772",
        ),
    ),
)


def _rows_to_grid(rows):
    return np.array(
        [[ord(character) - ord("0") for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_rows_to_grid(input_rows), _rows_to_grid(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _validated_requested_count(value, name, low, high):
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(seed, num_train, num_test):
    train_count = _validated_requested_count(
        num_train, "num_train", 2, 10
    )
    test_count = _validated_requested_count(
        num_test, "num_test", 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = test_count + 1
        train_count = minimum_train + (
            (seed // 4) % (11 - minimum_train)
        )
    elif train_count is None:
        minimum_train = test_count + 1
        train_count = minimum_train + seed % (11 - minimum_train)
    else:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed % maximum_test

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("The resolved train/test counts are illegal")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    """Only the three output phases are a shared color convention."""
    return {"phase_colors": tuple(int(x) for x in rng.choice(10, 3, replace=False))}


def _background(grid):
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    largest = int(counts.max())
    assert largest > grid.size // 2 and np.count_nonzero(counts == largest) == 1
    return int(np.argmax(counts))


def _ordered(coordinates):
    coordinates = np.asarray(coordinates, dtype=np.int64)
    assert coordinates.ndim == 2 and coordinates.shape[1] == 2
    rows = np.unique(coordinates[:, 0]).size
    columns = np.unique(coordinates[:, 1]).size
    assert rows != columns
    if rows > columns:
        return coordinates[np.lexsort((coordinates[:, 1], coordinates[:, 0]))], "row"
    return coordinates[np.lexsort((-coordinates[:, 0], coordinates[:, 1]))], "column"


def _adjacent(a, b):
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


def _make_coordinates(rng, orientation, mode, regime):
    """Choose occupied levels first, then distribute cells among them."""
    if mode == "isolated":
        marker_count = int(rng.integers(6, 15 if regime == "source_train" else 16))
        extras = 0
    else:
        marker_count = int(rng.integers(6, 19 if regime == "source_train" else 23))
        extras = int(rng.integers(1, min(4, marker_count - 4) + 1)) if mode == "tie" else 0
    primary_count = marker_count - extras

    if mode == "isolated":
        minimum = 2 * primary_count - 1
        primary_size = int(rng.integers(minimum, min(30, minimum + 7) + 1))
        parity = int(rng.integers(0, 2))
        candidates = list(range(parity, primary_size, 2))
        if len(candidates) < primary_count:
            candidates = list(range(0, primary_size, 2))
        primary_levels = sorted(int(x) for x in rng.choice(candidates, primary_count, replace=False))
    else:
        primary_size = int(rng.integers(primary_count, min(30, primary_count + 12) + 1))
        primary_levels = sorted(int(x) for x in rng.choice(primary_size, primary_count, replace=False))

    secondary_size = int(rng.integers(4, 25 if regime == "source_train" else 29))
    maximum_secondary_levels = min(secondary_size, primary_count - 1, max(3, primary_count // 2 + 2))
    secondary_count = int(rng.integers(2, maximum_secondary_levels + 1))
    if mode == "tie":
        first = int(rng.integers(0, secondary_size - 1))
        adjacent_levels = [first, first + 1]
        other_levels = [x for x in range(secondary_size) if x not in adjacent_levels]
        secondary_pool = adjacent_levels + [int(x) for x in rng.choice(other_levels, secondary_count - 2, replace=False)]
    else:
        secondary_pool = [int(x) for x in rng.choice(secondary_size, secondary_count, replace=False)]

    assignments = secondary_pool + [int(x) for x in rng.choice(secondary_pool, primary_count - secondary_count)]
    rng.shuffle(assignments)
    primary_secondary = [(primary_levels[i], assignments[i]) for i in range(primary_count)]
    touch_pair = None
    if mode == "tie":
        base_index = assignments.index(adjacent_levels[0])
        original = primary_secondary[base_index]
        second = (original[0], adjacent_levels[1])
        primary_secondary.append(second)
        touch_pair = (original, second)
        occupied = set(primary_secondary)
        possibilities = [(p, s) for p in primary_levels for s in secondary_pool if (p, s) not in occupied]
        rng.shuffle(possibilities)
        primary_secondary.extend(possibilities[:extras - 1])

    if orientation == "row":
        coordinates = [(p, s) for p, s in primary_secondary]
        height, width = primary_size, secondary_size
        if touch_pair is not None:
            touch_pair = ((touch_pair[0][0], touch_pair[0][1]), (touch_pair[1][0], touch_pair[1][1]))
    else:
        coordinates = [(s, p) for p, s in primary_secondary]
        height, width = secondary_size, primary_size
        if touch_pair is not None:
            touch_pair = ((touch_pair[0][1], touch_pair[0][0]), (touch_pair[1][1], touch_pair[1][0]))
    assert len(coordinates) == marker_count and len(set(coordinates)) == marker_count
    assert height * width > 2 * marker_count
    return height, width, coordinates, touch_pair


def _paint_pair(rng, height, width, coordinates, touch_pair, phase_colors):
    ordered, _ = _ordered(coordinates)
    marker_count = len(ordered)
    background = int(rng.choice([x for x in range(10) if x not in phase_colors]))
    palette_size = int(rng.integers(1, min(7, marker_count - 1) + 1))
    palette = [int(x) for x in rng.choice([x for x in range(10) if x != background], palette_size, replace=False)]
    colors = [palette[int(rng.integers(palette_size))] for _ in range(marker_count)]

    if touch_pair is None:
        first = int(rng.integers(marker_count))
        eligible = [i for i in range(marker_count) if i % 3 != first % 3]
        second = int(rng.choice(eligible))
    else:
        positions = {tuple(map(int, point)): i for i, point in enumerate(ordered)}
        first, second = (positions[touch_pair[0]], positions[touch_pair[1]])
    colors[first] = colors[second] = palette[0]
    other_positions = [i for i in range(marker_count) if i not in (first, second)]
    rng.shuffle(other_positions)
    for color, position in zip(palette[1:], other_positions):
        colors[position] = color

    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = input_grid.copy()
    for index, (point, color) in enumerate(zip(ordered, colors)):
        row, column = map(int, point)
        input_grid[row, column] = color
        output_grid[row, column] = phase_colors[index % 3]
    return {"input": input_grid, "output": output_grid}


def _grid_signature(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _layout_signature(grid):
    background = _background(grid)
    return grid.shape, (grid != background).tobytes()


def _pair_exactly_matches_source(input_grid, output_grid):
    return any(np.array_equal(input_grid, a) and np.array_equal(output_grid, b) for a, b in _SOURCE_PAIRS)


def _train_plans(rng, count):
    if count == 2:
        tied_axis = "row" if int(rng.integers(2)) == 0 else "column"
        plans = [("row", "tie" if tied_axis == "row" else "isolated"),
                 ("column", "tie" if tied_axis == "column" else "isolated")]
    else:
        plans = [("row", "tie"), ("column", "tie"),
                 ("row" if int(rng.integers(2)) == 0 else "column", "isolated")]
        for _ in range(count - 3):
            axis = "row" if int(rng.integers(2)) == 0 else "column"
            mode = ("isolated", "loose", "tie")[int(rng.integers(3))]
            plans.append((axis, mode))
    rng.shuffle(plans)
    return plans


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    phase_colors = sample_puzzle_parameters(rng)["phase_colors"]
    plans = _train_plans(rng, train_count)
    calibrated_ties = {axis for axis, mode in plans if mode == "tie"}

    # Pair streams advance independently and reproducibly.
    child_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                               size=train_count + test_count, dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_layouts = set(), set()
    for section, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(child_seeds[len(puzzle["train"]) + len(puzzle["test"])]))
            if section == "train":
                orientation, mode = plans[index]
            else:
                orientation = "row" if int(pair_rng.integers(2)) == 0 else "column"
                possibilities = ("isolated", "loose", "tie") if orientation in calibrated_ties else ("isolated", "loose")
                mode = possibilities[int(pair_rng.integers(len(possibilities)))]
            for _ in range(40):
                height, width, coordinates, touch_pair = _make_coordinates(
                    pair_rng, orientation, mode, "source_train" if section == "train" else "target_test")
                pair = _paint_pair(pair_rng, height, width, coordinates, touch_pair, phase_colors)
                input_grid, output_grid = pair["input"], pair["output"]
                signature = _grid_signature(input_grid)
                layout = _layout_signature(input_grid)
                if (signature not in seen_inputs and layout not in seen_layouts
                        and not _pair_exactly_matches_source(input_grid, output_grid)):
                    break
            else:
                raise RuntimeError("Could not construct a distinct pair")
            seen_inputs.add(signature)
            seen_layouts.add(layout)
            puzzle[section].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _facts(pair, phase_colors):
    input_grid, output_grid = pair["input"], pair["output"]
    background = _background(input_grid)
    assert background not in phase_colors
    assert _background(output_grid) == background
    coordinates = np.argwhere(input_grid != background)
    assert len(coordinates) >= 6 and len(coordinates) * 2 < input_grid.size
    ordered, orientation = _ordered(coordinates)
    assert np.array_equal(input_grid == background, output_grid == background)
    for index, (row, column) in enumerate(ordered):
        assert int(output_grid[row, column]) == phase_colors[index % 3]
    assert not np.array_equal(input_grid, output_grid)
    primary = 0 if orientation == "row" else 1
    ties = len(coordinates) - np.unique(coordinates[:, primary]).size
    touching = []
    for i, point in enumerate(coordinates):
        for other in coordinates[i + 1:]:
            if _adjacent(point, other):
                touching.append((tuple(map(int, point)), tuple(map(int, other))))
    same_color_touch = any(input_grid[a] == input_grid[b] and output_grid[a] != output_grid[b]
                           for a, b in touching)
    palette = set(int(x) for x in input_grid[input_grid != background])
    witness = any(input_grid[tuple(a)] == input_grid[tuple(b)] and i % 3 != j % 3
                  for i, a in enumerate(ordered) for j, b in enumerate(ordered) if i < j)
    return {"orientation": orientation, "ties": ties, "touching": bool(touching),
            "same_color_touch": same_color_touch, "palette_size": len(palette),
            "witness": witness, "markers": len(coordinates), "background": background,
            "shape": input_grid.shape}


def _untied_training_pair(train_pairs):
    for pair in train_pairs:
        background = _background(pair["input"])
        ordered, axis = _ordered(np.argwhere(pair["input"] != background))
        primary = 0 if axis == "row" else 1
        if np.unique(ordered[:, primary]).size == len(ordered):
            return pair
    raise AssertionError("Training needs an untied phase example")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs, seen_layouts = set(), set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            input_signature = _grid_signature(pair["input"])
            pair_signature = (input_signature, _grid_signature(pair["output"]))
            layout_signature = _layout_signature(pair["input"])
            assert input_signature not in seen_inputs and pair_signature not in seen_pairs
            assert layout_signature not in seen_layouts
            assert not _pair_exactly_matches_source(pair["input"], pair["output"])
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            seen_layouts.add(layout_signature)

    first = _untied_training_pair(puzzle["train"])
    background = _background(first["input"])
    ordered, _ = _ordered(np.argwhere(first["input"] != background))
    assert len(ordered) >= 3
    phases = tuple(int(first["output"][tuple(point)]) for point in ordered[:3])
    assert len(set(phases)) == 3
    train_facts = [_facts(pair, phases) for pair in puzzle["train"]]
    test_facts = [_facts(pair, phases) for pair in puzzle["test"]]
    assert {fact["orientation"] for fact in train_facts} == {"row", "column"}
    assert all(fact["witness"] for fact in train_facts)
    assert any(fact["ties"] == 0 for fact in train_facts)
    assert any(fact["same_color_touch"] for fact in train_facts)
    trained_ties = {fact["orientation"] for fact in train_facts if fact["ties"] > 0}
    assert all(fact["orientation"] in trained_ties for fact in test_facts if fact["ties"] > 0)
    if len(train_facts) >= 3:
        assert trained_ties == {"row", "column"}


def _puzzle_signature(puzzle):
    return tuple((_grid_signature(pair["input"]), _grid_signature(pair["output"]))
                 for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzle_signature(generate_puzzle(123456, 7, 3)) == _puzzle_signature(generate_puzzle(123456, 7, 3))
    assert len({_puzzle_signature(generate_puzzle(seed, 6, 2)) for seed in range(12)}) >= 10
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(700 + seed, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
    for seed, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(800 + seed, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(900, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count request accepted")

    conventions, layouts, shapes, markers = set(), set(), set(), set()
    palettes, tied, untied, isolated, touching, test_smaller, test_larger = set(), 0, 0, 0, 0, 0, 0
    row_tests = column_tests = 0
    wide_rows = tall_columns = 0
    test_tie_values = set()
    row_tie_tests = column_tie_tests = untied_column_tests = 0
    first_two_different = phase_overlap = 0
    varied_backgrounds = varied_palettes = mixed_test_axes = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        first = _untied_training_pair(puzzle["train"])
        bg = _background(first["input"])
        ordered, _ = _ordered(np.argwhere(first["input"] != bg))
        phases = tuple(int(first["output"][tuple(point)]) for point in ordered[:3])
        conventions.add(phases)
        train_shapes = [pair["input"].shape for pair in puzzle["train"]]
        train_markers = [_facts(pair, phases)["markers"] for pair in puzzle["train"]]
        local_backgrounds, local_palettes, local_test_axes = set(), set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                fact = _facts(pair, phases)
                grid = pair["input"]
                local_backgrounds.add(fact["background"])
                local_palettes.add(fact["palette_size"])
                layouts.add(_layout_signature(grid))
                shapes.add(grid.shape)
                markers.add(fact["markers"])
                palettes.add(fact["palette_size"])
                tied += fact["ties"] > 0
                untied += fact["ties"] == 0
                isolated += not fact["touching"]
                touching += fact["touching"]
                ordered, _ = _ordered(np.argwhere(grid != fact["background"]))
                first_two_different += grid[tuple(ordered[0])] != grid[tuple(ordered[1])]
                phase_overlap += any(color in phases for color in np.unique(grid[grid != fact["background"]]))
                wide_rows += fact["orientation"] == "row" and grid.shape[1] > grid.shape[0]
                tall_columns += fact["orientation"] == "column" and grid.shape[0] > grid.shape[1]
                if section == "test":
                    local_test_axes.add(fact["orientation"])
                    test_tie_values.add(fact["ties"])
                    row_tie_tests += fact["orientation"] == "row" and fact["ties"] > 0
                    column_tie_tests += fact["orientation"] == "column" and fact["ties"] > 0
                    untied_column_tests += fact["orientation"] == "column" and fact["ties"] == 0
                    row_tests += fact["orientation"] == "row"
                    column_tests += fact["orientation"] == "column"
                    test_smaller += fact["markers"] <= max(train_markers) and grid.size <= max(h*w for h, w in train_shapes)
                    test_larger += fact["markers"] > max(train_markers) or grid.size > max(h*w for h, w in train_shapes)
        varied_backgrounds += len(local_backgrounds) > 1
        varied_palettes += len(local_palettes) > 1
        mixed_test_axes += len(local_test_axes) > 1
    assert len(conventions) >= 25 and len(layouts) >= 200 and len(shapes) >= 60
    assert len(markers) >= 14 and len(palettes) >= 6
    assert tied >= 40 and untied >= 50 and isolated >= 30 and touching >= 30
    assert row_tests >= 10 and column_tests >= 10
    assert test_smaller >= 10 and test_larger >= 10
    assert wide_rows >= 15 and tall_columns >= 15
    assert min(palettes) == 1 and max(palettes) == 7
    assert len(test_tie_values) >= 4 and 0 in test_tie_values
    assert row_tie_tests >= 2 and column_tie_tests >= 2 and untied_column_tests >= 10
    assert first_two_different >= 30 and phase_overlap >= 40
    assert varied_backgrounds >= 35 and varied_palettes >= 35 and mixed_test_axes >= 5
