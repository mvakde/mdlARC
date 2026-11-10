import numpy as np


_SNAKE_PATH = (
    (0, 0), (0, 1), (0, 2),
    (1, 2), (1, 1), (1, 0),
    (2, 0), (2, 1), (2, 2),
)

_SOURCE_PAIR_ROWS = (
    (("0000000000", "0040000002", "0000008000", "0000000000",
      "0000600000", "9000000000", "0000000050", "0300000000",
      "0000000000", "0000000000"), ("934", "586", "200")),
    (("0000000000", "0000000000", "0100600000", "0000000000",
      "3000000000", "0000007000", "0000000000", "0000000009",
      "0080000000", "0000000000"), ("318", "976", "000")),
    (("0000000003", "0050090000", "0000000010", "0400000000",
      "0000000500", "0000300000", "0000000000", "0000000000",
      "2000001000", "0000000000"), ("245", "193", "513")),
    (("0000000000", "0000509000", "0000000001", "0200000000",
      "0000000600", "0003000000", "0000000000", "5000000000",
      "0000040000", "0090000000"), ("529", "453", "961")),
)


def _rows_to_array(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(cell) for cell in row] for row in rows], dtype=np.int8)


def _array_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(cell) for cell in grid.flat)


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return any(
        np.array_equal(input_grid, _rows_to_array(source_input))
        and np.array_equal(output_grid, _rows_to_array(source_output))
        for source_input, source_output in _SOURCE_PAIR_ROWS
    )


def _background(input_grid: np.ndarray) -> int:
    return int(np.bincount(input_grid.ravel(), minlength=10).argmax())


def _extract_sequence(input_grid: np.ndarray, background: int) -> list[int]:
    sequence = []
    for column in range(input_grid.shape[1]):
        markers = input_grid[:, column]
        markers = markers[markers != background]
        if markers.size:
            sequence.append(int(markers[0]))
    return sequence


def _make_output(sequence: list[int] | np.ndarray, background: int) -> np.ndarray:
    output = np.full((3, 3), background, dtype=np.int8)
    for color, (row, column) in zip(sequence, _SNAKE_PATH):
        output[row, column] = int(color)
    return output


def _normalize_count(name: str, value: int | None, lower: int, upper: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    num_train = _normalize_count("num_train", num_train, 2, 10)
    num_test = _normalize_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _sample_width(rng: np.random.Generator, count: int,
                  require_gap: bool) -> int:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        extra = int(rng.integers(0, 4))
    elif mode == 1:
        extra = int(rng.integers(4, 10))
    else:
        extra = int(rng.integers(10, 22))
    return min(30, count + max(extra, int(require_gap)))


def _sample_height(rng: np.random.Generator, scrambled: bool) -> int:
    mode = int(rng.integers(0, 3))
    bounds = ((3, 7), (7, 15), (15, 25))[mode]
    return max(4 if scrambled else 3, int(rng.integers(*bounds)))


def _sample_columns(rng: np.random.Generator, width: int, count: int,
                    require_gap: bool) -> np.ndarray:
    for _ in range(80):
        mode = int(rng.integers(0, 4))
        if mode == 0:
            start = int(rng.integers(0, width - count + 1))
            columns = np.arange(start, start + count)
        elif mode == 1 or width - count < 3:
            columns = np.sort(rng.choice(width, size=count, replace=False))
        elif mode == 2:
            # Both margins can be blank when the canvas permits them.
            interior = np.arange(1, width - 1)
            if len(interior) < count:
                continue
            columns = np.sort(rng.choice(interior, size=count, replace=False))
        else:
            # A short run with some independently placed columns.
            run = int(rng.integers(2, min(count, 5) + 1)) if count >= 2 else 1
            start = int(rng.integers(0, width - run + 1))
            fixed = np.arange(start, start + run)
            remaining = np.setdiff1d(np.arange(width), fixed)
            columns = np.sort(np.concatenate((fixed, rng.choice(
                remaining, size=count - run, replace=False))))
        if not require_gap or np.any(np.diff(columns) > 1):
            return columns.astype(np.int16)
    columns = np.arange(count, dtype=np.int16)
    columns[-1] += 1
    return columns


def _sample_rows(rng: np.random.Generator, height: int, count: int,
                 scrambled: bool) -> np.ndarray:
    mode = 4 if scrambled else int(rng.integers(0, 6))
    if mode == 0:
        rows = np.full(count, int(rng.integers(0, height)))
    elif mode == 1 and count > 1:
        choices = rng.choice(height, size=2, replace=False)
        rows = rng.choice(choices, size=count)
        rows[:2] = choices
        rng.shuffle(rows)
    elif mode == 2:
        band = min(height, int(rng.integers(3, 6)))
        start = int(rng.integers(0, height - band + 1))
        rows = rng.integers(start, start + band, size=count)
    elif mode == 3:
        rows = np.rint(np.linspace(0, height - 1, count)).astype(int)
        if bool(rng.integers(0, 2)):
            rows = rows[::-1]
    elif mode == 4:
        rows = rng.integers(0, height, size=count)
    else:
        rows = np.array([0 if i % 2 == 0 else height - 1
                         for i in range(count)])
        if count > 2:
            rows[int(rng.integers(0, count))] = int(rng.integers(0, height))
    return rows.astype(np.int16)


def _sample_sequence(rng: np.random.Generator, colors: np.ndarray,
                     count: int, strong: bool) -> np.ndarray:
    if strong:
        for _ in range(100):
            sequence = rng.choice(colors, size=count, replace=True)
            if (len(set(sequence)) >= 4 and sequence[0] != sequence[2]
                    and sequence[3] != sequence[5]
                    and sequence[6] != sequence[8]):
                return sequence.astype(np.int8)
        # The color pool has at least four members, so this is always feasible.
        sequence = np.resize(rng.permutation(colors), count)
        if sequence[3] == sequence[5]:
            sequence[5] = next(color for color in colors if color != sequence[3])
        return sequence.astype(np.int8)

    mode = int(rng.integers(0, 4))
    if mode == 0 and len(colors) >= count:
        sequence = rng.choice(colors, size=count, replace=False)
    elif mode == 1:
        pool_size = int(rng.integers(1, min(3, len(colors)) + 1))
        pool = rng.choice(colors, size=pool_size, replace=False)
        sequence = rng.choice(pool, size=count, replace=True)
    elif mode == 2 and len(colors) >= 2:
        pool_size = int(rng.integers(2, min(6, len(colors)) + 1))
        pool = rng.choice(colors, size=pool_size, replace=False)
        weights = np.ones(pool_size, dtype=float)
        weights[0] = float(rng.integers(4, 10))
        weights /= weights.sum()
        sequence = rng.choice(pool, size=count, replace=True, p=weights)
    else:
        sequence = rng.choice(colors, size=count, replace=True)
    return sequence.astype(np.int8)


def _generate_pair(rng: np.random.Generator, count: int, background: int,
                   colors: np.ndarray, strong: bool = False,
                   require_gap: bool = False,
                   novel_color: int | None = None,
                   reused_columns: tuple[int, np.ndarray] | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(100):
        if reused_columns is not None:
            width, columns = reused_columns
        else:
            width = (10 if count == 9 and rng.random() < 0.28
                     else _sample_width(rng, count, require_gap))
            columns = _sample_columns(rng, width, count, require_gap)
        height = _sample_height(rng, strong)
        rows = _sample_rows(rng, height, count, strong)
        sequence = _sample_sequence(rng, colors, count, strong)
        if novel_color is not None:
            sequence[int(rng.integers(0, count))] = novel_color
        if strong:
            order = np.lexsort((columns, rows))
            if np.array_equal(sequence[order], sequence):
                continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        input_grid[rows, columns] = sequence
        return input_grid, _make_output(sequence, background)
    raise RuntimeError("could not construct a row-order counterexample")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)

    # The family has no hidden palette map: marker colors are copied literally.
    # This pool only shapes the examples; each input reveals its own background.
    base_palette = rng.choice(10, size=int(rng.integers(5, 9)), replace=False)
    fixed_background = int(rng.integers(0, 10)) if rng.random() < 0.25 else None

    train_specs = [(9, True, True), (int(rng.integers(4, 9)), False, True)]
    train_specs.extend((int(rng.integers(1, 10)), False, False)
                       for _ in range(num_train - 2))
    rng.shuffle(train_specs)
    test_specs = [int(rng.integers(4, 9))]
    for _ in range(num_test - 1):
        band = int(rng.integers(0, 10))
        if band == 0:
            test_specs.append(int(rng.integers(1, 4)))
        elif band <= 6:
            test_specs.append(int(rng.integers(4, 9)))
        else:
            test_specs.append(9)

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_colors = set()
    train_shapes = set()
    train_heights = set()
    train_widths = set()
    previous_full_columns = []
    first_train_background = None
    for split_name, specifications in (("train", train_specs), ("test", test_specs)):
        for index, specification in enumerate(specifications):
            if split_name == "train":
                count, strong, require_gap = specification
            else:
                count, strong, require_gap = specification, False, False
            child_seed = rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
            pair_rng = np.random.default_rng(child_seed)
            for _ in range(120):
                background = (fixed_background if fixed_background is not None
                              else int(pair_rng.integers(0, 10)))
                if (fixed_background is None and split_name == "train" and index == 1
                        and background == first_train_background):
                    background = (background + 1 + int(pair_rng.integers(0, 9))) % 10
                colors = base_palette[base_palette != background]
                novel = None
                if split_name == "test" and pair_rng.random() < 0.42:
                    unseen = [color for color in range(10)
                              if color != background and color not in train_colors]
                    if unseen:
                        novel = int(pair_rng.choice(unseen))
                reused_columns = None
                if count == 9 and not strong and previous_full_columns:
                    if pair_rng.random() < 0.42:
                        prior = previous_full_columns[int(pair_rng.integers(
                            0, len(previous_full_columns)))]
                        reused_columns = (prior[0], prior[1].copy())
                input_grid, output_grid = _generate_pair(
                    pair_rng, count, background, colors, strong,
                    require_gap, novel, reused_columns)
                input_key = _array_key(input_grid)
                if input_key in seen_inputs or _is_source_pair(input_grid, output_grid):
                    continue
                if split_name == "train" and index == num_train - 1:
                    if (len(train_heights | {input_grid.shape[0]}) < 2
                            or len(train_widths | {input_grid.shape[1]}) < 2):
                        continue
                seen_inputs.add(input_key)
                puzzle[split_name].append({"input": input_grid, "output": output_grid})
                if count == 9:
                    occupied = np.flatnonzero(np.any(input_grid != background, axis=0))
                    previous_full_columns.append((input_grid.shape[1], occupied))
                if split_name == "train":
                    train_shapes.add(input_grid.shape)
                    train_heights.add(input_grid.shape[0])
                    train_widths.add(input_grid.shape[1])
                    train_colors.update(_extract_sequence(input_grid, background))
                    if first_train_background is None:
                        first_train_background = background
                break
            else:
                raise RuntimeError("could not generate a unique pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def _independent_prediction(input_grid: np.ndarray) -> np.ndarray:
    background = int(np.bincount(input_grid.ravel(), minlength=10).argmax())
    locations = np.argwhere(input_grid != background)
    ordered = sorted((int(column), int(input_grid[row, column]))
                     for row, column in locations)
    result = np.full((3, 3), background, dtype=np.int8)
    for index, (_, color) in enumerate(ordered):
        row = index // 3
        column = index % 3 if row % 2 == 0 else 2 - index % 3
        result[row, column] = color
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs = set()
    seen_pairs = set()
    full_train = False
    padded_train = False
    gap_train = False
    row_counterexample = False
    visible_snake = False
    partial_test = False
    train_shapes = set()
    train_heights = set()
    train_widths = set()
    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert output_grid.shape == (3, 3)
            background = _background(input_grid)
            background_count = int(np.count_nonzero(input_grid == background))
            assert background_count > input_grid.size // 2
            locations = np.argwhere(input_grid != background)
            count = len(locations)
            assert 1 <= count <= 9
            columns = locations[:, 1]
            assert len(set(int(column) for column in columns)) == count
            assert count <= input_grid.shape[1]
            assert np.array_equal(output_grid, _independent_prediction(input_grid))
            assert not _is_source_pair(input_grid, output_grid)
            input_key = _array_key(input_grid)
            pair_key = (input_key, _array_key(output_grid))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            sequence = _extract_sequence(input_grid, background)
            assert len(sequence) == count
            assert set(sequence).isdisjoint({background})
            occupied = np.sort(columns)
            if split_name == "train":
                train_shapes.add(input_grid.shape)
                train_heights.add(input_grid.shape[0])
                train_widths.add(input_grid.shape[1])
                full_train |= count == 9
                padded_train |= count < 9
                gap_train |= np.any(np.diff(occupied) > 1)
                row_order = [int(value) for value in input_grid[input_grid != background]]
                row_counterexample |= row_order != sequence
                visible_snake |= (count == 9 and sequence[0] != sequence[2]
                                  and sequence[3] != sequence[5]
                                  and sequence[6] != sequence[8])
            else:
                partial_test |= count < 9
    assert full_train and padded_train and gap_train
    assert row_counterexample and visible_snake
    assert partial_test and len(train_shapes) >= 2
    assert len(train_heights) >= 2 and len(train_widths) >= 2


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((split, tuple((_array_key(pair["input"]), _array_key(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    original = [(_rows_to_array(source_input), _rows_to_array(source_output))
                for source_input, source_output in _SOURCE_PAIR_ROWS]
    for input_grid, output_grid in original:
        assert np.array_equal(_independent_prediction(input_grid), output_grid)

    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(7000 + 10 * train_count + test_count,
                                 train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(
            7000 + 10 * train_count + test_count, train_count, test_count))
    for offset, args in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(8100 + offset, *args)
        if args[0] is not None:
            assert len(puzzle["train"]) == args[0]
        if args[1] is not None:
            assert len(puzzle["test"]) == args[1]
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(99173, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {args}")

    puzzles = [generate_puzzle(seed) for seed in range(4200, 4242)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    shapes, widths, heights, layouts, backgrounds = set(), set(), set(), set(), set()
    counts, color_counts, test_counts = set(), set(), set()
    new_test_color = 0
    one_row = two_rows = multiple_gaps = contiguous = zero_marker = 0
    combined = 0
    all_distinct = heavy_repeat = uneven = 0
    wide_sparse = narrow = large_margins = 0
    monotone = compact_band = scrambled = adjacent = 0
    repeated_full = different_full = same_train_test_full = 0
    varied_background = varied_train_heights = varied_train_widths = 0
    tiny_test = full_test = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_colors = set()
        puzzle_backgrounds = set()
        puzzle_train_heights = set()
        puzzle_train_widths = set()
        full_columns = {"train": [], "test": []}
        for pair in puzzle["train"]:
            grid = pair["input"]
            train_colors.update(_extract_sequence(grid, _background(grid)))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                background = _background(grid)
                markers = np.argwhere(grid != background)
                count = len(markers)
                columns = np.sort(markers[:, 1])
                sequence = [int(grid[row, column]) for row, column in
                            sorted(markers, key=lambda position: position[1])]
                shapes.add(grid.shape)
                heights.add(grid.shape[0])
                widths.add(grid.shape[1])
                layouts.add((grid.shape, tuple(map(tuple, markers))))
                backgrounds.add(background)
                counts.add(count)
                color_counts.add(len(set(sequence)))
                puzzle_backgrounds.add(background)
                if split == "train":
                    puzzle_train_heights.add(grid.shape[0])
                    puzzle_train_widths.add(grid.shape[1])
                if count == 9:
                    full_columns[split].append((grid.shape[1], tuple(columns)))
                one_row += len(set(markers[:, 0])) == 1
                two_rows += len(set(markers[:, 0])) == 2
                multiple_gaps += int(np.count_nonzero(np.diff(columns) > 1) >= 2)
                contiguous += int(count > 1 and np.all(np.diff(columns) == 1))
                zero_marker += int(background != 0 and 0 in sequence)
                combined += int(background != 0 and count < 9 and
                                np.count_nonzero(np.diff(columns) > 1) >= 2)
                all_distinct += int(count >= 4 and len(set(sequence)) == count)
                heavy_repeat += int(count >= 6 and len(set(sequence)) <= 2)
                uneven += int(count >= 6 and len(set(sequence)) >= 2 and
                              max(sequence.count(color) for color in set(sequence)) >= 4)
                wide_sparse += int(grid.shape[1] - count >= 10)
                narrow += int(grid.shape[1] == count)
                large_margins += int(columns[0] >= 3 and
                                     grid.shape[1] - 1 - columns[-1] >= 3)
                row_values = [int(markers[markers[:, 1] == column][0, 0])
                              for column in columns]
                differences = np.diff(row_values)
                monotone += int(count >= 4 and len(set(row_values)) >= 3 and
                                (np.all(differences >= 0) or np.all(differences <= 0)))
                compact_band += int(max(row_values) - min(row_values) <= 3 and
                                    len(set(row_values)) >= 3)
                scrambled += int(count >= 4 and np.any(differences > 0) and
                                 np.any(differences < 0))
                adjacent += int(any(row_values[index] == row_values[index + 1]
                                    and columns[index + 1] == columns[index] + 1
                                    for index in range(count - 1)))
                if split == "test":
                    test_counts.add(count)
                    new_test_color += bool(set(sequence) - train_colors)
                    tiny_test += count <= 3
                    full_test += count == 9
                assert np.array_equal(pair["output"], _independent_prediction(grid))
        varied_background += len(puzzle_backgrounds) > 1
        varied_train_heights += len(puzzle_train_heights) > 1
        varied_train_widths += len(puzzle_train_widths) > 1
        train_full = full_columns["train"]
        all_full = train_full + full_columns["test"]
        repeated_full += len(set(all_full)) < len(all_full)
        different_full += len(set(train_full)) > 1
        same_train_test_full += bool(set(train_full) & set(full_columns["test"]))
    assert len(shapes) >= 45 and len(widths) >= 20 and len(heights) >= 18
    assert len(layouts) >= 200 and len(backgrounds) == 10
    assert {1, 2, 3, 4, 5, 6, 7, 8, 9} <= counts
    assert {4, 5, 6, 7, 8, 9} <= test_counts
    assert {1, 2, 3, 4, 5, 6} <= color_counts
    assert min(one_row, two_rows, multiple_gaps, contiguous,
               zero_marker, combined, new_test_color) >= 3
    assert min(all_distinct, heavy_repeat, uneven, wide_sparse, narrow,
               large_margins, monotone, compact_band, scrambled, adjacent,
               tiny_test, full_test) >= 1
    assert repeated_full >= 1 and different_full >= 1
    assert same_train_test_full >= 1
    assert varied_background >= 8
    assert varied_train_heights == varied_train_widths == 42
