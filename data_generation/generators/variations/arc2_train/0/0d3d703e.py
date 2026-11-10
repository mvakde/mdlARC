import numpy as np


_LEGAL_COUNTS = tuple(
    (train_count, test_count)
    for test_count in range(1, 5)
    for train_count in range(test_count + 1, 11)
)

_SOURCE_ROWS = (
    ("586", "192"),
    ("238", "649"),
    ("942", "836"),
    ("312", "456"),
    ("813", "954"),
)


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(value) for value in grid.flat))


_SOURCE_PAIRS = frozenset(
    (
        _grid_key(np.tile(np.array([int(value) for value in source], dtype=np.int8), (3, 1))),
        _grid_key(np.tile(np.array([int(value) for value in target], dtype=np.int8), (3, 1))),
    )
    for source, target in _SOURCE_ROWS
)


def _integer_argument(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _counts(seed: int, rng: np.random.Generator, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train_count = _integer_argument("num_train", num_train, 2, 10)
    test_count = _integer_argument("num_test", num_test, 1, 4)
    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must exceed num_test")
        return train_count, test_count
    if train_count is None and test_count is None:
        return _LEGAL_COUNTS[seed % len(_LEGAL_COUNTS)]
    if train_count is None:
        return int(rng.integers(test_count + 1, 11)), test_count
    return train_count, int(rng.integers(1, min(4, train_count - 1) + 1))


def _partition(rng: np.random.Generator, length: int, parts: int) -> np.ndarray:
    if parts == 1:
        return np.array([0, length], dtype=int)
    cuts = np.sort(rng.choice(np.arange(1, length), size=parts - 1, replace=False))
    return np.concatenate(([0], cuts, [length]))


def _band_grid(rng: np.random.Generator, colors: tuple[int, ...], orientation: str, compact: bool) -> np.ndarray:
    color_count = len(colors)
    if compact:
        band_count = 3
        axis_length = 3
        other_length = 3
    else:
        band_count = int(rng.integers(color_count, min(7, color_count + 3) + 1))
        if rng.random() < 0.16:
            axis_length = band_count
            other_length = int(rng.integers(2, 8))
        else:
            axis_length = 30 if rng.random() < 0.035 else int(rng.integers(band_count + 1, 25))
            other_length = 30 if rng.random() < 0.035 else int(rng.integers(2, 25))
    sequence = [int(color) for color in rng.permutation(colors)]
    while len(sequence) < band_count:
        available = [color for color in colors if color != sequence[-1]]
        sequence.append(int(rng.choice(available)))
    boundaries = _partition(rng, axis_length, band_count)
    stripe = np.empty(axis_length, dtype=np.int8)
    for index, color in enumerate(sequence):
        stripe[boundaries[index]:boundaries[index + 1]] = color
    if orientation == "vertical":
        return np.tile(stripe, (other_length, 1))
    return np.tile(stripe[:, None], (1, other_length))


def _block_grid(rng: np.random.Generator, colors: tuple[int, ...]) -> np.ndarray:
    height = int(rng.integers(5, 25))
    width = int(rng.integers(5, 25))
    row_count = int(rng.integers(2, 5))
    column_count = int(rng.integers(2, 5))
    if row_count * column_count < len(colors):
        row_count = 3
        column_count = 3
    row_edges = _partition(rng, height, row_count)
    column_edges = _partition(rng, width, column_count)
    tile_count = row_count * column_count
    labels = rng.choice(colors, size=tile_count).astype(np.int8)
    anchor_tiles = rng.choice(tile_count, size=len(colors), replace=False)
    for tile_index, color in zip(anchor_tiles, rng.permutation(colors)):
        labels[int(tile_index)] = int(color)
    grid = np.empty((height, width), dtype=np.int8)
    for row in range(row_count):
        for column in range(column_count):
            grid[row_edges[row]:row_edges[row + 1], column_edges[column]:column_edges[column + 1]] = labels[row * column_count + column]
    return grid


def _irregular_grid(rng: np.random.Generator, colors: tuple[int, ...]) -> np.ndarray:
    height = int(rng.integers(5, 25))
    width = int(rng.integers(5, 25))
    site_indices = rng.choice(height * width, size=len(colors), replace=False)
    rows, columns = np.indices((height, width))
    distances = np.empty((len(colors), height, width), dtype=float)
    for index, site in enumerate(site_indices):
        site_row, site_column = divmod(int(site), width)
        distances[index] = (
            np.abs(rows - site_row) + np.abs(columns - site_column)
            + rng.random((height, width)) * 0.35
        )
    return np.asarray(colors, dtype=np.int8)[np.argmin(distances, axis=0)]


def _fallback_grid(colors: tuple[int, ...], index: int) -> np.ndarray:
    height = 16 + index
    width = 30
    stripe = np.asarray([colors[column % len(colors)] for column in range(width)], dtype=np.int8)
    return np.tile(stripe, (height, 1))


def _new_pair(
    rng: np.random.Generator,
    colors: tuple[int, ...],
    layout: str,
    orientation: str,
    compact: bool,
    partner_lut: np.ndarray,
    seen_inputs: set[tuple],
) -> dict:
    for _ in range(18):
        if layout == "bands":
            input_grid = _band_grid(rng, colors, orientation, compact)
        elif layout == "blocks":
            input_grid = _block_grid(rng, colors)
        else:
            input_grid = _irregular_grid(rng, colors)
        output_grid = partner_lut[input_grid]
        input_key = _grid_key(input_grid)
        if input_key not in seen_inputs and (input_key, _grid_key(output_grid)) not in _SOURCE_PAIRS:
            seen_inputs.add(input_key)
            return {"input": input_grid, "output": output_grid}
    input_grid = _fallback_grid(colors, len(seen_inputs))
    output_grid = partner_lut[input_grid]
    input_key = _grid_key(input_grid)
    assert input_key not in seen_inputs
    seen_inputs.add(input_key)
    return {"input": input_grid, "output": output_grid}


def _color_sets(active_colors: tuple[int, ...], maximum: int) -> list[tuple[int, ...]]:
    sets = []
    for mask in range(1 << len(active_colors)):
        selected = tuple(active_colors[index] for index in range(len(active_colors)) if mask & (1 << index))
        if 2 <= len(selected) <= maximum:
            sets.append(selected)
    return sets


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed_value = int(seed)
    rng = np.random.default_rng(seed_value)
    train_count, test_count = _counts(seed_value, rng, num_train, num_test)

    tier = int(rng.choice(3, p=(0.12, 0.43, 0.45)))
    maximum_pairs = 3 if tier == 0 else min(5, train_count + 2)
    pair_count = int(rng.integers(2, maximum_pairs + 1))
    shuffled_palette = tuple(int(color) for color in rng.permutation(10))
    active_colors = shuffled_palette[:2 * pair_count]
    partner_lut = np.arange(10, dtype=np.int8)
    for index in range(0, len(active_colors), 2):
        first, second = active_colors[index:index + 2]
        partner_lut[first] = second
        partner_lut[second] = first

    orientation_style = int(rng.integers(0, 3))
    shift_test_orientation = bool(rng.random() < 0.27)
    base_count = min(train_count, max(2, (len(active_colors) + 2) // 3))
    shuffled_active = tuple(int(color) for color in rng.permutation(active_colors))
    mandatory_groups = [tuple(shuffled_active[index::base_count]) for index in range(base_count)]
    maximum_colors = 3 if tier == 0 else 5
    all_sets = _color_sets(active_colors, maximum_colors)
    seen_inputs: set[tuple] = set()
    training_sets: set[frozenset[int]] = set()
    train: list[dict] = []

    def orientation(index: int, split: str) -> str:
        if orientation_style == 0:
            return "horizontal" if split == "test" and shift_test_orientation else "vertical"
        if orientation_style == 1:
            return "vertical" if split == "test" and shift_test_orientation else "horizontal"
        if split == "train" and index < 2:
            return ("vertical", "horizontal")[index]
        return ("vertical", "horizontal")[int(rng.integers(0, 2))]

    def layout(index: int, split: str) -> str:
        if tier < 2:
            return "bands"
        if split == "train" and index == 0:
            return "bands"
        if split == "train" and index == (2 if train_count >= 3 else 1):
            return ("blocks", "irregular")[int(rng.integers(0, 2))]
        if split == "train" and index == 1:
            return "bands"
        choice = int(rng.choice(3, p=(0.50, 0.25, 0.25)))
        return ("bands", "blocks", "irregular")[choice]

    for index in range(train_count):
        if index < base_count:
            colors = mandatory_groups[index]
        else:
            colors = all_sets[int(rng.integers(0, len(all_sets)))]
        pair = _new_pair(
            rng, colors, layout(index, "train"), orientation(index, "train"),
            tier == 0, partner_lut, seen_inputs,
        )
        train.append(pair)
        training_sets.add(frozenset(int(color) for color in np.unique(pair["input"])))

    familiar_sets = [colors for colors in all_sets if frozenset(colors) in training_sets]
    novel_sets = [colors for colors in all_sets if frozenset(colors) not in training_sets]
    training_sizes = {len(colors) for colors in training_sets}
    test: list[dict] = []
    for index in range(test_count):
        if index == 0 and novel_sets:
            pool = novel_sets
            if len(training_sizes) == 1:
                varied_sizes = [colors for colors in pool if len(colors) not in training_sizes]
                if varied_sizes:
                    pool = varied_sizes
        else:
            selection = int(rng.integers(0, 3))
            pool = (familiar_sets, novel_sets, all_sets)[selection]
            if not pool:
                pool = all_sets
        colors = pool[int(rng.integers(0, len(pool)))]
        test.append(_new_pair(
            rng, colors, layout(index, "test"), orientation(index, "test"),
            tier == 0, partner_lut, seen_inputs,
        ))

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    mapping: dict[int, int] = {}
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert input_grid.shape == output_grid.shape
            assert not np.array_equal(input_grid, output_grid)
            assert 2 <= len(np.unique(input_grid)) <= 5
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIRS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert len(np.unique(output_grid)) == len(np.unique(input_grid))
            for color in np.unique(input_grid):
                source = int(color)
                targets = np.unique(output_grid[input_grid == source])
                assert len(targets) == 1
                target = int(targets[0])
                assert source != target
                if split == "train":
                    assert source not in mapping or mapping[source] == target
                    mapping[source] = target
                else:
                    assert source in mapping and mapping[source] == target

    assert len(mapping) >= 4 and len(mapping) % 2 == 0
    assert all(partner in mapping and mapping[partner] == color for color, partner in mapping.items())
    for pair in puzzle["test"]:
        prediction = np.empty_like(pair["input"])
        for color in np.unique(pair["input"]):
            prediction[pair["input"] == color] = mapping[int(color)]
        assert np.array_equal(prediction, pair["output"])


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(left[role], right[role])
            for left, right in zip(first[split], second[split])
            for role in ("input", "output")
        )
        for split in ("train", "test")
    )


def _layout_features(grid: np.ndarray) -> tuple[str, bool, bool]:
    vertical = bool(np.all(grid == grid[0:1, :]))
    horizontal = bool(np.all(grid == grid[:, 0:1]))
    if not vertical and not horizontal:
        return "patch", False, False
    stripe = grid[0] if vertical else grid[:, 0]
    starts = np.flatnonzero(np.r_[True, stripe[1:] != stripe[:-1]])
    lengths = np.diff(np.r_[starts, len(stripe)])
    labels = [int(stripe[start]) for start in starts]
    repeated = len(labels) > len(set(labels))
    return ("vertical" if vertical else "horizontal"), len(set(lengths)) > 1, repeated


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = set()
    mappings = set()
    palettes = set()
    shapes = set()
    counts = set()
    input_color_counts = set()
    orientations = set()
    pair_together = False
    unequal_bands = False
    repeated_bands = False
    familiar_tests = False
    novel_tests = False
    mixed_training = False
    same_orientation = False
    zero_seen = False
    seven_seen = False
    within_color_count = False
    within_shape = False
    active_pair_counts = set()
    compact_puzzles = False
    same_whole_orientation = False
    block_patterns = False
    irregular_patterns = False
    test_orientation_shift = False
    test_size_shift = False
    retained_position = False
    new_position = False

    for seed, puzzle in enumerate(puzzles):
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        signatures.add(tuple(_grid_key(pair["input"]) for split in ("train", "test") for pair in puzzle[split]))
        learned = {}
        train_sets = set()
        train_orientations = set()
        puzzle_shapes = set()
        puzzle_color_counts = set()
        for pair in puzzle["train"]:
            source, target = pair["input"], pair["output"]
            train_sets.add(frozenset(int(color) for color in np.unique(source)))
            for input_value, output_value in zip(source.flat, target.flat):
                color, partner = int(input_value), int(output_value)
                assert color not in learned or learned[color] == partner
                learned[color] = partner
            train_orientations.add(_layout_features(source)[0])
        mappings.add(tuple(sorted(learned.items())))
        palettes.add(tuple(sorted(learned)))
        active_pair_counts.add(len(learned) // 2)
        pair_together |= any(
            any(learned[color] in set(int(value) for value in np.unique(pair["input"])) for color in np.unique(pair["input"]))
            for pair in puzzle["train"]
        )
        mixed_training |= "vertical" in train_orientations and "horizontal" in train_orientations
        same_orientation |= len(train_orientations) == 1 and "patch" not in train_orientations
        training_positions: dict[int, set[int]] = {}
        for pair in puzzle["train"]:
            source = pair["input"]
            orientation = _layout_features(source)[0]
            if orientation == "patch":
                continue
            stripe = source[0] if orientation == "vertical" else source[:, 0]
            starts = np.flatnonzero(np.r_[True, stripe[1:] != stripe[:-1]])
            for position, start in enumerate(starts):
                training_positions.setdefault(int(stripe[start]), set()).add(position)
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        puzzle_orientations = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                colors = frozenset(int(color) for color in np.unique(source))
                zero_seen |= 0 in colors
                seven_seen |= 7 in colors
                shapes.add(source.shape)
                puzzle_shapes.add(source.shape)
                input_color_counts.add(len(colors))
                puzzle_color_counts.add(len(colors))
                orientation, unequal, repeated = _layout_features(source)
                orientations.add(orientation)
                puzzle_orientations.add(orientation)
                unequal_bands |= unequal
                repeated_bands |= repeated
                if orientation == "patch":
                    row_changes = sum(
                        bool(np.any(source[row] != source[row - 1]))
                        for row in range(1, source.shape[0])
                    )
                    column_changes = sum(
                        bool(np.any(source[:, column] != source[:, column - 1]))
                        for column in range(1, source.shape[1])
                    )
                    block_patterns |= 1 <= row_changes <= 3 and 1 <= column_changes <= 3
                    irregular_patterns |= row_changes > 3 or column_changes > 3
                prediction = np.empty_like(source)
                for color in colors:
                    assert color in learned
                    prediction[source == color] = learned[color]
                assert np.array_equal(prediction, target)
                if split == "test":
                    familiar_tests |= colors in train_sets
                    novel_tests |= colors not in train_sets
                    test_orientation_shift |= orientation not in train_orientations
                    test_size_shift |= source.shape not in train_shapes
                    if orientation != "patch":
                        stripe = source[0] if orientation == "vertical" else source[:, 0]
                        starts = np.flatnonzero(np.r_[True, stripe[1:] != stripe[:-1]])
                        for position, start in enumerate(starts):
                            color = int(stripe[start])
                            retained_position |= position in training_positions.get(color, set())
                            new_position |= color in training_positions and position not in training_positions[color]
        within_color_count |= len(puzzle_color_counts) > 1
        within_shape |= len(puzzle_shapes) > 1
        compact_puzzles |= puzzle_shapes == {(3, 3)}
        same_whole_orientation |= len(puzzle_orientations) == 1 and "patch" not in puzzle_orientations

    assert len({count[0] for count in counts}) >= 4
    assert len({count[1] for count in counts}) >= 3
    assert len(signatures) >= 40
    assert len(mappings) >= 35 and len(palettes) >= 12
    assert active_pair_counts == {2, 3, 4, 5}
    assert len(shapes) >= 40
    assert input_color_counts == {2, 3, 4, 5}
    assert orientations == {"vertical", "horizontal", "patch"}
    assert all((pair_together, unequal_bands, repeated_bands, familiar_tests, novel_tests,
                mixed_training, same_orientation, zero_seen, seven_seen,
                within_color_count, within_shape, compact_puzzles, same_whole_orientation,
                block_patterns, irregular_patterns, test_orientation_shift, test_size_shift,
                retained_position, new_position))

    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 6, 2)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        assert _puzzles_equal(puzzle, generate_puzzle(seed, train_count, test_count))
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(2000, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(3000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {arguments}")
