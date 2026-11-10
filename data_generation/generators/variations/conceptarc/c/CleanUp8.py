import numpy as np


_LAYOUTS = tuple(
    (size * count, size)
    for size in range(2, 6)
    for count in range(3 if size > 2 else 4, min(9, 30 // size) + 1)
)

_SOURCE_INPUT_ROWS = (
    (10, """
2200000000
6200300100
0016000000
0011006006
0000330000
0300330100
0000034400
0100704600
7000000055
0000300075
"""),
    (10, """
0004000443
0600000033
0000007700
0004006700
4000660000
0100680000
0055000060
0015006000
9900000001
8900440000
"""),
    (10, """
3300030009
3303000000
0022000060
0022009000
0000650060
0440550000
0000007700
0600004760
0600400066
0000000066
"""),
    (14, """
44000003000000
34000700000000
00770000000000
00470606000300
60004900000000
00009900009000
00000022000000
90060022000090
00000000177000
00000000110000
90300200003300
00000000069300
00900000000059
00000090000055
"""),
    (12, """
115100000000
111105000020
312300030000
111100000000
000044440000
250034255000
000044440000
005044340030
200050002888
000000038382
003025008288
000000005888
"""),
)
_SOURCE_INPUT_KEYS = {
    (height, "".join(rows.split()))
    for height, rows in _SOURCE_INPUT_ROWS
}


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.flat),
    )


def _source_key(grid: np.ndarray) -> tuple:
    return (
        grid.shape[0],
        "".join(str(int(value)) for value in grid.flat),
    )


def _check_requested_count(value, minimum: int, maximum: int, name: str) -> None:
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not minimum <= int(value) <= maximum
    ):
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")


def _choose_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _check_requested_count(num_train, 2, 10, "num_train")
    _check_requested_count(num_test, 1, 4, "num_test")

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    return num_train, num_test


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    return {
        "shared_background": bool(rng.integers(0, 2)),
        "background": int(rng.integers(0, 10)),
    }


def _sample_layout(rng: np.random.Generator, excluded=()) -> tuple[int, int]:
    choices = [layout for layout in _LAYOUTS if layout not in excluded]
    return choices[int(rng.integers(len(choices)))]


def _target_column(block_row: int, block_count: int, direction: int) -> int:
    return block_row if direction == 1 else block_count - 1 - block_row


def _make_output(
    height: int,
    square_size: int,
    direction: int,
    colors: tuple[int, ...],
    background: int,
) -> np.ndarray:
    output = np.full((height, height), background, dtype=np.int8)
    block_count = height // square_size

    for block_row, color in enumerate(colors):
        block_col = _target_column(block_row, block_count, direction)
        row = block_row * square_size
        col = block_col * square_size
        output[row:row + square_size, col:col + square_size] = color

    return output


def _sample_colors(
    rng: np.random.Generator,
    count: int,
    background: int,
    excluded_color: int | None = None,
) -> tuple[int, ...]:
    available = np.array([
        color for color in range(10)
        if color != background and color != excluded_color
    ])
    if rng.random() < 0.38:
        pool_size = int(rng.integers(2, count))
        pool = rng.choice(available, size=pool_size, replace=False)
        colors = rng.choice(pool, size=count, replace=True)
        if len(set(int(color) for color in colors)) == count:
            colors[-1] = colors[0]
    else:
        colors = rng.choice(available, size=count, replace=len(available) < count)
    return tuple(int(color) for color in colors)


def _damage_positions(
    rng: np.random.Generator, size: int, count: int
) -> np.ndarray:
    positions = np.arange(size * size)
    if size == 2 or rng.random() < 0.35:
        return rng.choice(positions, size=count, replace=False)
    pattern = int(rng.integers(0, 3))
    if pattern == 0:
        candidates = np.array(
            [p for p in positions if p // size in (0, size - 1)
             or p % size in (0, size - 1)]
        )
        front = rng.permutation(candidates)
        back = rng.permutation(np.setdiff1d(positions, candidates))
        return np.concatenate((front, back))[:count]
    anchor = int(rng.choice(positions))
    row, col = divmod(anchor, size)
    jitter = rng.random(size * size)
    distances = np.array(
        [abs(p // size - row) + abs(p % size - col) for p in positions]
    )
    if pattern == 2:
        distances = np.array(
            [max(abs(p // size - row), abs(p % size - col)) for p in positions]
        )
    return np.argsort(distances + jitter * 0.4)[:count]


def _make_noisy_input(
    rng: np.random.Generator,
    output: np.ndarray,
    square_size: int,
    direction: int,
    colors: tuple[int, ...],
    background: int,
    regime: str,
) -> np.ndarray:
    height = output.shape[0]
    block_count = height // square_size
    input_grid = output.copy()

    available = np.array([color for color in range(10) if color != background])
    if rng.random() < 0.58:
        palette_size = int(rng.integers(2, 5))
    else:
        palette_size = int(rng.integers(5, 10))
    distractors = rng.choice(available, size=palette_size, replace=False)
    erasure_rate = (0.0, 0.28, 0.72)[int(rng.choice(3, p=[0.62, 0.25, 0.13]))]
    affected_count = int(rng.integers(1, block_count + 1))
    affected = set(
        int(i) for i in rng.choice(block_count, size=affected_count, replace=False)
    )

    for block_row, color in enumerate(colors):
        if block_row not in affected:
            continue

        maximum = (square_size * square_size - 1) // 2
        corruption_count = int(rng.integers(1, maximum + 1))

        block_col = _target_column(block_row, block_count, direction)
        positions = _damage_positions(rng, square_size, corruption_count)
        for position in positions:
            position = int(position)
            row = block_row * square_size + position // square_size
            col = block_col * square_size + position % square_size
            if rng.random() < erasure_rate:
                input_grid[row, col] = background
            else:
                choices = distractors[distractors != color]
                input_grid[row, col] = int(rng.choice(choices))

    outside_blocks = [
        (block_row, block_col)
        for block_row in range(block_count)
        for block_col in range(block_count)
        if block_col != _target_column(block_row, block_count, direction)
    ]
    density = rng.uniform(0.06, 0.20 if regime == "source_train" else 0.23)
    noise_count = max(2, int(round(density * len(outside_blocks) * square_size**2)))
    decoy = None
    if square_size == 2 and rng.random() < 0.25:
        decoy = int(rng.integers(len(outside_blocks)))
        noise_count = max(noise_count, 3)
    capacity = min(2, square_size * square_size // 2)
    slots = np.repeat(np.arange(len(outside_blocks)), capacity)
    if decoy is not None:
        slots = slots[slots != decoy]
    remaining = min(len(slots), noise_count - (3 if decoy is not None else 0))
    selected_slots = rng.choice(slots, size=remaining, replace=False)
    counts = np.bincount(
        selected_slots, minlength=len(outside_blocks)
    )
    if decoy is not None:
        counts[decoy] = 3

    for block_index, count in enumerate(counts):
        if count == 0:
            continue

        block_row, block_col = outside_blocks[block_index]
        positions = rng.choice(
            square_size * square_size,
            size=int(count),
            replace=False,
        )
        repeated = int(rng.choice(distractors)) if (count >= 2 and rng.random() < 0.5) else None
        for position in positions:
            position = int(position)
            row = block_row * square_size + position // square_size
            col = block_col * square_size + position % square_size
            input_grid[row, col] = repeated if repeated is not None else int(rng.choice(distractors))

    return input_grid


def _recover_candidates(input_grid: np.ndarray) -> list[tuple]:
    height, width = input_grid.shape
    if height != width:
        return []

    background = int(np.argmax(np.bincount(input_grid.ravel(), minlength=10)))
    candidates = []
    for square_size in range(2, min(10, height // 3) + 1):
        if height % square_size != 0:
            continue

        block_count = height // square_size
        for direction in (1, -1):
            colors = []
            false_squares = 0
            valid = True
            for block_row in range(block_count):
                target_col = _target_column(block_row, block_count, direction)
                for block_col in range(block_count):
                    row = block_row * square_size
                    col = block_col * square_size
                    block = input_grid[row:row + square_size, col:col + square_size]
                    counts = np.bincount(block.ravel(), minlength=10)
                    foreground = counts.copy()
                    foreground[background] = 0
                    color = int(np.argmax(foreground))
                    occupied = square_size * square_size - int(counts[background])
                    if block_col == target_col:
                        if foreground[color] <= square_size * square_size // 2:
                            valid = False
                            break
                        colors.append(color)
                    elif occupied > square_size * square_size // 2:
                        if square_size != 2 or occupied != 3:
                            valid = False
                            break
                        false_squares += 1
                        if false_squares > 1:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                colors = tuple(colors)
                output = _make_output(height, square_size, direction, colors, background)
                candidates.append((output, square_size, direction, colors, background))

    return candidates


def _output_layout(output: np.ndarray) -> tuple[int, int, tuple[int, ...], int]:
    height, width = output.shape
    if height != width:
        raise AssertionError("Output must be square")

    background = int(np.argmax(np.bincount(output.ravel(), minlength=10)))
    matches = []
    for square_size in range(2, min(10, height // 3) + 1):
        if height % square_size != 0:
            continue

        block_count = height // square_size
        for direction in (1, -1):
            colors = []
            valid = True

            for block_row in range(block_count):
                target_col = _target_column(
                    block_row, block_count, direction
                )
                for block_col in range(block_count):
                    row = block_row * square_size
                    col = block_col * square_size
                    block = output[
                        row:row + square_size,
                        col:col + square_size,
                    ]

                    if block_col == target_col:
                        color = int(block[0, 0])
                        if color == background or not np.all(block == color):
                            valid = False
                            break
                        colors.append(color)
                    elif np.any(block != background):
                        valid = False
                        break

                if not valid:
                    break

            if valid:
                matches.append(
                    (square_size, direction, tuple(colors), background)
                )

    if len(matches) != 1:
        raise AssertionError("Output lacks one unique diagonal square layout")
    return matches[0]


def _generate_pair(
    rng: np.random.Generator,
    layout: tuple[int, int],
    direction: int,
    colors: tuple[int, ...],
    background: int,
    regime: str,
    seen_inputs: set,
) -> dict:
    height, square_size = layout
    output = _make_output(height, square_size, direction, colors, background)

    for _ in range(40):
        input_grid = _make_noisy_input(
            rng, output, square_size, direction, colors, background, regime
        )
        input_key = _grid_key(input_grid)
        if input_key in seen_inputs:
            continue
        if _source_key(input_grid) in _SOURCE_INPUT_KEYS:
            continue

        candidates = _recover_candidates(input_grid)
        if (
            len(candidates) == 1
            and np.array_equal(candidates[0][0], output)
        ):
            seen_inputs.add(input_key)
            return {"input": input_grid, "output": output}

    raise RuntimeError("Could not produce a uniquely recoverable pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _choose_counts(rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    total = num_train + num_test
    reserved_color = int(rng.integers(0, 10)) if rng.random() < 0.5 else None

    if parameters["shared_background"]:
        backgrounds = [parameters["background"]] * total
    else:
        backgrounds = [parameters["background"]]
        alternatives = [c for c in range(10) if c != backgrounds[0]]
        backgrounds.append(alternatives[int(rng.integers(len(alternatives)))])
        backgrounds.extend(int(rng.integers(0, 10)) for _ in range(total - 2))

    train_layouts = [_sample_layout(rng)]
    different_sizes = [x for x in _LAYOUTS if x[1] != train_layouts[0][1]]
    train_layouts.append(different_sizes[int(rng.integers(len(different_sizes)))])
    train_layouts.extend(_sample_layout(rng) for _ in range(num_train - 2))
    test_layouts = []
    for _ in range(num_test):
        if rng.random() < 0.42:
            test_layouts.append(train_layouts[int(rng.integers(num_train))])
        else:
            test_layouts.append(_sample_layout(rng))

    pair_seeds = rng.integers(
        0, np.iinfo(np.int64).max, size=total, dtype=np.int64
    )
    first_direction = 1 if rng.random() < 0.5 else -1
    if rng.random() < 0.38:
        train_directions = [first_direction] * num_train
    else:
        train_directions = [first_direction, -first_direction]
        train_directions.extend(
            1 if rng.random() < 0.5 else -1 for _ in range(num_train - 2)
        )
    test_directions = [
        1 if rng.random() < 0.5 else -1 for _ in range(num_test)
    ]
    seen_inputs = set()
    puzzle = {"train": [], "test": []}
    pair_directions = train_directions + test_directions
    for index, layout in enumerate(train_layouts + test_layouts):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        count = layout[0] // layout[1]
        background = backgrounds[index]
        colors = _sample_colors(
            pair_rng, count, background,
            reserved_color if index < num_train else None,
        )
        if (
            index >= num_train
            and reserved_color is not None
            and reserved_color != background
            and pair_rng.random() < 0.75
        ):
            colors = list(colors)
            colors[int(pair_rng.integers(count))] = reserved_color
            colors = tuple(colors)
        direction = pair_directions[index]
        section = "train" if index < num_train else "test"
        regime = "source_train" if section == "train" else "target_test"
        pair = _generate_pair(
            pair_rng, layout, direction, colors, background, regime, seen_inputs
        )
        puzzle[section].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    seen_inputs = set()
    seen_pairs = set()
    train_layouts = []
    train_backgrounds = []
    test_backgrounds = []
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output = pair["input"], pair["output"]
            for grid in (input_grid, output):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert input_grid.shape == output.shape
            assert not np.array_equal(input_grid, output)
            assert _source_key(input_grid) not in _SOURCE_INPUT_KEYS
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            size, direction, colors, background = _output_layout(output)
            height = output.shape[0]
            count = height // size
            assert (height, size) in _LAYOUTS
            input_background = int(np.argmax(np.bincount(input_grid.ravel(), minlength=10)))
            assert input_background == background
            candidates = _recover_candidates(input_grid)
            assert len(candidates) == 1
            assert np.array_equal(candidates[0][0], output)
            assert candidates[0][1:] == (size, direction, colors, background)

            damaged = 0
            stray = 0
            false_squares = 0
            for block_row in range(count):
                target_col = _target_column(block_row, count, direction)
                for block_col in range(count):
                    row, col = block_row * size, block_col * size
                    block = input_grid[row:row + size, col:col + size]
                    if block_col == target_col:
                        matching = int(np.count_nonzero(block == colors[block_row]))
                        assert matching > size * size // 2
                        damaged += matching < size * size
                    else:
                        occupied = int(np.count_nonzero(block != background))
                        stray += occupied
                        if occupied > size * size // 2:
                            assert size == 2 and occupied == 3
                            false_squares += 1
                        elif size == 2:
                            assert occupied <= 2
            assert damaged >= 1 and stray >= 2
            assert false_squares <= 1
            if section == "train":
                train_layouts.append((height, size))
                train_backgrounds.append(background)
            else:
                test_backgrounds.append(background)
    assert train_layouts[0][1] != train_layouts[1][1]
    if len(set(train_backgrounds)) == 1:
        assert set(test_backgrounds) == set(train_backgrounds)


def _same_puzzles(first: dict, second: dict) -> bool:
    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False
        for left, right in zip(first[section], second[section]):
            for key in ("input", "output"):
                if not np.array_equal(left[key], right[key]):
                    return False
    return True


def self_test() -> None:
    train_counts, test_counts = set(), set()
    train_layouts, test_layouts = set(), set()
    sizes, lengths, backgrounds = set(), set(), set()
    directions, first_directions = set(), set()
    one_slope = both_slopes = variable_background = test_changes_slope = 0
    near_clean = almost_all = all_damaged = 0
    two_strays = decoys = repeated_colors = novel_test_colors = 0
    no_erasure = with_erasure = small_palette = broad_palette = 0
    familiar_test = smaller_test = 0
    source_like = same_color_two = zero_object = zero_noise = 0
    palette_overlap = edge_damage = interior_damage = adjacent_damage = 0
    varied_train_damage = 0
    signatures = set()

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_features = [_output_layout(pair["output"]) for pair in puzzle["train"]]
        train_dirs = {feature[1] for feature in train_features}
        one_slope += len(train_dirs) == 1
        both_slopes += len(train_dirs) == 2
        test_changes_slope += any(
            _output_layout(pair["output"])[1] not in train_dirs
            for pair in puzzle["test"]
        )
        first_directions.add(train_features[0][1])
        train_bgs = {feature[3] for feature in train_features}
        variable_background += len(train_bgs) > 1
        train_colors = {color for feature in train_features for color in feature[2]}
        train_heights = {pair["output"].shape[0] for pair in puzzle["train"]}
        train_damage_counts = []
        signature = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                size, direction, colors, bg = _output_layout(out)
                height = out.shape[0]
                count = height // size
                assert len(_recover_candidates(inp)) == 1
                assert np.array_equal(_recover_candidates(inp)[0][0], out)
                assert _source_key(inp) not in _SOURCE_INPUT_KEYS
                signature.append((section, _grid_key(inp), _grid_key(out)))
                (train_layouts if section == "train" else test_layouts).add((height, size))
                sizes.add(size)
                lengths.add(count)
                backgrounds.add(bg)
                directions.add(direction)
                repeated_colors += len(set(colors)) < len(colors)
                if section == "test":
                    novel_test_colors += bool(set(colors) - train_colors)
                    familiar_test += (height, size) in {
                        (p["output"].shape[0], _output_layout(p["output"])[0])
                        for p in puzzle["train"]
                    }
                    smaller_test += height < min(train_heights)
                damaged = 0
                erasures = 0
                corruption_colors = set()
                outside_colors = set()
                max_outside = 0
                damaged_masks = []
                for r in range(count):
                    tc = _target_column(r, count, direction)
                    for c in range(count):
                        block = inp[r*size:(r+1)*size, c*size:(c+1)*size]
                        if c == tc:
                            wrong_mask = block != colors[r]
                            wrong = block[wrong_mask]
                            damaged_masks.append(wrong_mask)
                            damaged += len(wrong) > 0
                            erasures += int(np.count_nonzero(wrong == bg))
                            corruption_colors.update(int(v) for v in wrong if v != bg)
                        else:
                            vals = block[block != bg]
                            outside_colors.update(int(v) for v in vals)
                            max_outside = max(max_outside, len(vals))
                            same_color_two += (
                                size == 2 and len(vals) == 2 and vals[0] == vals[1]
                            )
                if section == "train":
                    train_damage_counts.append(damaged)
                near_clean += damaged == 1
                almost_all += damaged == count - 1
                all_damaged += damaged == count
                source_like += size == 2 and count == 5 and damaged in (4, 5)
                two_strays += max_outside >= 2
                decoys += max_outside == 3 and size == 2
                no_erasure += erasures == 0
                with_erasure += erasures > 0
                zero_object += bg != 0 and 0 in colors
                zero_noise += bg != 0 and 0 in outside_colors
                palette_overlap += bool(corruption_colors & outside_colors)
                palette_count = len(outside_colors | corruption_colors)
                small_palette += palette_count <= 4
                broad_palette += palette_count >= 5
                if size >= 3:
                    edge_damage += any(
                        any(r in (0, size - 1) or c in (0, size - 1)
                            for r, c in np.argwhere(mask))
                        for mask in damaged_masks
                    )
                    interior_damage += any(
                        any(0 < r < size - 1 and 0 < c < size - 1
                            for r, c in np.argwhere(mask))
                        for mask in damaged_masks
                    )
                    adjacent_damage += any(
                        any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                            for i, a in enumerate(np.argwhere(mask))
                            for b in np.argwhere(mask)[i + 1:])
                        for mask in damaged_masks
                    )
        varied_train_damage += len(set(train_damage_counts)) > 1
        signatures.add(tuple(signature))

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(train_layouts) >= 12 and len(test_layouts) >= 12
    assert sizes == {2, 3, 4, 5} and len(lengths) >= 6
    assert len(backgrounds) >= 5 and directions == {-1, 1}
    assert first_directions == {-1, 1} and one_slope > 0 and both_slopes > 0
    assert test_changes_slope > 0
    assert variable_background > 0 and near_clean > 0 and almost_all > 0
    assert all_damaged > 0 and two_strays > 0 and decoys > 0
    assert source_like > 0 and same_color_two > 0
    assert zero_object > 0 and zero_noise > 0 and palette_overlap > 0
    assert edge_damage > 0 and interior_damage > 0 and adjacent_damage > 0
    assert varied_train_damage > 0
    assert repeated_colors > 0 and novel_test_colors > 0
    assert no_erasure > 0 and with_erasure > 0
    assert small_palette > 0 and broad_palette > 0
    assert familiar_test > 0 and smaller_test > 0
    assert len(signatures) >= 40

    for seed, tr, te in (
        (100, 2, 1), (101, 10, 1), (102, 10, 4),
        (103, None, 1), (104, None, 4),
        (105, 2, None), (106, 10, None),
    ):
        puzzle = generate_puzzle(seed, tr, te)
        validate_generated_puzzle(puzzle)
        if tr is not None:
            assert len(puzzle["train"]) == tr
        if te is not None:
            assert len(puzzle["test"]) == te
        assert _same_puzzles(puzzle, generate_puzzle(seed, tr, te))

    assert not _same_puzzles(generate_puzzle(0), generate_puzzle(1))
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3)):
        try:
            generate_puzzle(123, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts did not raise ValueError")
