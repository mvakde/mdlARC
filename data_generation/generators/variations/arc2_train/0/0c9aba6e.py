import numpy as np


_SOURCE_EXAMPLES = {
    (("0002", "2000", "0222", "0002", "2020", "0220", "7777", "6066", "6006", "0666", "6000", "6006", "0060"),
     ("0800", "0880", "8000", "0880", "0800", "8008")),
    (("2202", "2022", "2200", "0202", "0220", "2002", "7777", "6066", "0600", "0000", "0006", "6600", "6060"),
     ("0000", "0000", "0088", "8080", "0008", "0800")),
    (("2200", "0220", "2200", "2000", "0002", "2200", "7777", "6666", "6066", "6600", "0000", "6600", "0060"),
     ("0000", "0000", "0088", "0888", "0080", "0008")),
    (("0220", "2000", "0202", "2222", "0020", "0022", "7777", "0660", "0000", "6666", "6606", "0666", "0060"),
     ("8008", "0888", "0000", "0000", "8000", "8800")),
    (("0002", "0220", "2002", "0220", "2022", "0002", "7777", "6606", "6660", "0000", "6606", "6060", "0066"),
     ("0080", "0008", "0880", "0000", "0800", "8800")),
}


def _resolve_counts(rng, num_train, num_test):
    for name, value, minimum, maximum in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not minimum <= int(value) <= maximum
        ):
            raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train = int(num_train)
        num_test = int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _is_source_example(pair):
    rows = lambda grid: tuple("".join(str(int(value)) for value in row) for row in grid)
    return (rows(pair["input"]), rows(pair["output"])) in _SOURCE_EXAMPLES


def _layout(seed, rng, split, index):
    mode = seed % 4
    if mode == 0:
        return "stacked"
    if mode == 1:
        return "side"
    if mode == 2:
        return "stacked" if index % 2 == 0 else "side"
    train_layout = "stacked" if bool(rng.integers(0, 2)) else "side"
    return train_layout if split == "train" else ("side" if train_layout == "stacked" else "stacked")


def _sample_shape(rng, layout, both_layouts):
    maximum_height = 14 if layout == "stacked" or both_layouts else 30
    maximum_width = 14 if layout == "side" or both_layouts else 30
    category = int(rng.integers(0, 7))
    if category == 0:
        return int(rng.integers(2, 5)), int(rng.integers(2, 5))
    if category == 1:
        return int(rng.integers(4, 9)), int(rng.integers(3, 10))
    if category == 2:
        size = int(rng.integers(3, min(maximum_height, maximum_width, 12) + 1))
        return size, size
    if category == 3:
        return int(rng.integers(2, 6)), int(rng.integers(10, maximum_width + 1))
    if category == 4:
        return int(rng.integers(10, maximum_height + 1)), int(rng.integers(2, 6))
    if category == 5:
        return int(rng.integers(maximum_height - 2, maximum_height + 1)), int(rng.integers(maximum_width - 2, maximum_width + 1))
    return int(rng.integers(2, maximum_height + 1)), int(rng.integers(2, maximum_width + 1))


def _anchor_states(rng, shape):
    size = shape[0] * shape[1]
    states = np.concatenate((np.arange(4, dtype=np.int8), rng.integers(0, 4, size=size - 4, dtype=np.int8)))
    rng.shuffle(states)
    return states.reshape(shape)


def _choose_density(rng, split):
    category = int(rng.choice(3, p=(0.28, 0.44, 0.28) if split == "train" else (0.34, 0.32, 0.34)))
    low, high = ((0.04, 0.26), (0.30, 0.70), (0.74, 0.96))[category]
    return float(rng.uniform(low, high))


def _structured_mask(rng, shape, density):
    height, width = shape
    size = height * width
    target = max(1, min(size - 1, int(round(density * size))))
    mode = int(rng.integers(0, 7))
    mask = np.zeros(shape, dtype=bool)
    if mode == 0:
        mask.flat[rng.choice(size, size=target, replace=False)] = True
    elif mode == 1:
        rectangle_height = int(rng.integers(1, height + 1))
        rectangle_width = max(1, min(width, int(round(target / rectangle_height))))
        row = int(rng.integers(0, height - rectangle_height + 1))
        column = int(rng.integers(0, width - rectangle_width + 1))
        mask[row:row + rectangle_height, column:column + rectangle_width] = True
    elif mode == 2:
        if bool(rng.integers(0, 2)):
            mask[int(rng.integers(0, height)), :] = True
        else:
            mask[:, int(rng.integers(0, width))] = True
    elif mode == 3:
        period = int(rng.integers(2, 5))
        offset = int(rng.integers(0, period))
        if bool(rng.integers(0, 2)):
            mask[offset::period, :] = True
        else:
            mask[:, offset::period] = True
    elif mode == 4:
        top = int(rng.integers(0, max(1, height - 1)))
        left = int(rng.integers(0, max(1, width - 1)))
        bottom = int(rng.integers(top + 1, height + 1))
        right = int(rng.integers(left + 1, width + 1))
        mask[top:bottom, left] = True
        mask[top:bottom, right - 1] = True
        mask[top, left:right] = True
        mask[bottom - 1, left:right] = True
    else:
        start = (int(rng.integers(0, height)), int(rng.integers(0, width)))
        cells = {start}
        while len(cells) < target:
            row, column = tuple(cells)[int(rng.integers(0, len(cells)))]
            neighbors = [(row + 1, column), (row - 1, column), (row, column + 1), (row, column - 1)]
            candidates = [(r, c) for r, c in neighbors if 0 <= r < height and 0 <= c < width and (r, c) not in cells]
            if candidates:
                cells.add(candidates[int(rng.integers(0, len(candidates)))])
            elif len(cells) < target and all(not (0 <= r < height and 0 <= c < width and (r, c) not in cells) for point in cells for r, c in ((point[0] + 1, point[1]), (point[0] - 1, point[1]), (point[0], point[1] + 1), (point[0], point[1] - 1))):
                break
        for row, column in cells:
            mask[row, column] = True
        if mode == 6:
            additions = max(1, target // 5)
            mask.flat[rng.choice(size, size=additions, replace=False)] = True
    return mask


def _general_states(rng, shape, split):
    size = shape[0] * shape[1]
    first_count = int(round(size * _choose_density(rng, split)))
    second_count = int(round(size * _choose_density(rng, split)))
    first_count = max(1, min(size - 1, first_count))
    second_count = max(1, min(size - 1, second_count))
    minimum_overlap = max(0, first_count + second_count - size)
    maximum_overlap = min(first_count, second_count)
    overlap = int(rng.integers(minimum_overlap, maximum_overlap + 1))
    counts = [size - first_count - second_count + overlap, first_count - overlap, second_count - overlap, overlap]
    states = np.repeat(np.arange(4, dtype=np.int8), counts)
    rng.shuffle(states)
    return states.reshape(shape)


def _structured_states(rng, shape, split):
    if bool(rng.integers(0, 2)):
        first = _structured_mask(rng, shape, _choose_density(rng, split))
        second = _structured_mask(rng, shape, _choose_density(rng, split))
        return first.astype(np.int8) + 2 * second.astype(np.int8)
    uncovered = _structured_mask(rng, shape, _choose_density(rng, split))
    states = np.zeros(shape, dtype=np.int8)
    weights = rng.dirichlet(np.array([1.3, 1.3, 1.3]))
    states[~uncovered] = rng.choice(np.array([1, 2, 3], dtype=np.int8), size=int((~uncovered).sum()), p=weights)
    return states


def _boundary_states(rng, shape, mode):
    first = _structured_mask(rng, shape, float(rng.uniform(0.22, 0.78)))
    if not first.any():
        first.flat[int(rng.integers(0, first.size))] = True
    if first.all():
        first.flat[int(rng.integers(0, first.size))] = False
    second = np.zeros(shape, dtype=bool)
    if mode == "identical":
        second = first.copy()
    elif mode == "disjoint":
        available = np.flatnonzero(~first)
        count = max(1, int(round(len(available) * float(rng.uniform(0.25, 0.85)))))
        second.flat[rng.choice(available, size=count, replace=False)] = True
    elif mode == "contained":
        available = np.flatnonzero(first)
        count = max(1, len(available) // 2)
        second.flat[rng.choice(available, size=count, replace=False)] = True
    elif mode == "empty":
        if bool(rng.integers(0, 2)):
            second = first.copy()
            first[:] = False
        else:
            second, first = first, second
    elif mode == "full":
        if bool(rng.integers(0, 2)):
            first[:] = True
        else:
            second[:] = True
    elif mode == "full_union":
        second = ~first
        second |= first & (rng.random(shape) < float(rng.uniform(0.1, 0.8)))
    elif mode == "both_empty":
        first[:] = False
    return first.astype(np.int8) + 2 * second.astype(np.int8)


def _make_pair(states, layout, colors):
    background, first_color, second_color, divider, result = colors
    first = np.where((states & 1) != 0, first_color, background)
    second = np.where((states & 2) != 0, second_color, background)
    if layout == "stacked":
        input_grid = np.vstack((first, np.full((1, states.shape[1]), divider), second))
    else:
        input_grid = np.hstack((first, np.full((states.shape[0], 1), divider), second))
    output_grid = np.where(states == 0, result, background)
    return {"input": input_grid.astype(np.int8), "output": output_grid.astype(np.int8)}


def _pair_colors(rng, anchor_colors, fixed, anchor_or_boundary):
    if fixed or anchor_or_boundary:
        return anchor_colors
    result = anchor_colors[-1]
    if bool(rng.integers(0, 2)):
        local = np.array(anchor_colors[:4], dtype=np.int8)
        rng.shuffle(local)
        if tuple(int(value) for value in local) == anchor_colors[:4]:
            local = np.roll(local, 1)
    else:
        local = rng.choice(np.array([color for color in range(10) if color != result]), size=4, replace=False)
    return tuple(int(value) for value in local) + (result,)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    anchor_colors = tuple(int(value) for value in rng.choice(10, size=5, replace=False))
    fixed_colors = seed % 7 == 0 or seed % 20 == 0 or seed % 13 == 0
    fixed_shape = seed % 5 == 0 or seed % 13 == 0
    repeated_output = seed % 13 == 0 and train_count >= 2
    both_layouts = seed % 4 in (2, 3)
    transfer_layout = "stacked" if bool(rng.integers(0, 2)) else "side"
    common_shape = (6, 4) if seed % 20 == 0 else None
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    first_states = None
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if seed % 4 == 3:
                layout = transfer_layout if split == "train" else ("side" if transfer_layout == "stacked" else "stacked")
            else:
                layout = _layout(seed, rng, split, index)
            boundary_mode = None
            if split == "test" and index == 0 and seed % 3 == 0:
                boundary_mode = ("identical", "disjoint", "contained", "empty", "full", "full_union", "both_empty")[seed % 7]
            elif split == "train" and index > 0 and not (repeated_output and index == 1) and rng.random() < 0.20:
                boundary_mode = ("identical", "disjoint", "contained", "empty", "full", "full_union", "both_empty")[int(rng.integers(0, 7))]
            if common_shape is None or not fixed_shape:
                shape = _sample_shape(rng, layout, both_layouts)
                if fixed_shape:
                    common_shape = shape
            else:
                shape = common_shape
            if repeated_output and split == "train" and index == 1:
                shape = first_states.shape
            colors = _pair_colors(rng, anchor_colors, fixed_colors, index == 0 and split == "train" or boundary_mode is not None or repeated_output and split == "train" and index == 1)
            accepted = None
            for attempt in range(30):
                if split == "train" and index == 0:
                    states = _anchor_states(rng, shape)
                elif repeated_output and split == "train" and index == 1:
                    states = first_states.copy()
                    positions = np.flatnonzero(states != 0)
                    position = int(positions[attempt % len(positions)])
                    states.flat[position] = 1 + int(states.flat[position]) % 3
                elif boundary_mode is not None and attempt < 8:
                    states = _boundary_states(rng, shape, boundary_mode)
                elif (split == "test" and seed % 3 == 2) or rng.random() < 0.48:
                    states = _structured_states(rng, shape, split)
                else:
                    states = _general_states(rng, shape, split)
                if (boundary_mode is None or attempt >= 8) and not (repeated_output and split == "train" and index == 1) and any(
                    int(((states & bit) != 0).sum()) in (0, states.size) for bit in (1, 2)
                ):
                    continue
                candidate = _make_pair(states, layout, colors)
                key = _grid_key(candidate["input"])
                if key in seen_inputs or _is_source_example(candidate):
                    continue
                accepted = candidate
                seen_inputs.add(key)
                break
            if accepted is None:
                for code in range(256):
                    states = np.zeros(shape, dtype=np.int8)
                    quotient = code
                    for position in range(4):
                        states.flat[position] = quotient % 4
                        quotient //= 4
                    state_counts = np.bincount(states.flat, minlength=4)
                    if split == "train" and index == 0 and not np.all(state_counts > 0):
                        continue
                    if any(int(((states & bit) != 0).sum()) in (0, states.size) for bit in (1, 2)):
                        continue
                    candidate = _make_pair(states, layout, colors)
                    key = _grid_key(candidate["input"])
                    if key not in seen_inputs and not _is_source_example(candidate):
                        accepted = candidate
                        seen_inputs.add(key)
                        break
            if accepted is None:
                raise RuntimeError("could not construct a distinct pair")
            if split == "train" and index == 0:
                first_states = states.copy()
            puzzle[split].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair, learned=None):
    input_grid, output_grid = pair["input"], pair["output"]
    height, width = output_grid.shape
    if input_grid.shape == (2 * height + 1, width):
        layout = "stacked"
        first, separator, second = input_grid[:height], input_grid[height:height + 1], input_grid[height + 1:]
    else:
        assert input_grid.shape == (height, 2 * width + 1)
        layout = "side"
        first, separator, second = input_grid[:, :width], input_grid[:, width:width + 1], input_grid[:, width + 1:]
    separator_colors = set(int(value) for value in np.unique(separator))
    assert len(separator_colors) == 1
    divider = next(iter(separator_colors))
    first_colors = set(int(value) for value in np.unique(first))
    second_colors = set(int(value) for value in np.unique(second))
    output_colors = set(int(value) for value in np.unique(output_grid))
    assert divider not in first_colors | second_colors
    if learned is None:
        shared = first_colors & second_colors & output_colors
        assert len(shared) == 1
        background = next(iter(shared))
        remaining = output_colors - {background}
        assert len(remaining) == 1
        result = next(iter(remaining))
    else:
        result = learned[4]
        assert result not in first_colors | second_colors | {divider}
        other_output = output_colors - {result}
        if other_output:
            assert len(other_output) == 1
            background = next(iter(other_output))
        else:
            shared = first_colors & second_colors
            assert len(shared) == 1
            background = next(iter(shared))
    assert result != background and divider != background
    assert first_colors & second_colors <= {background}
    assert len(first_colors - {background}) <= 1
    assert len(second_colors - {background}) <= 1
    assert output_colors <= {background, result}
    first_foreground = next(iter(first_colors - {background}), None)
    second_foreground = next(iter(second_colors - {background}), None)
    if first_foreground is not None and second_foreground is not None:
        assert first_foreground != second_foreground
    if learned is not None and (len(first_colors) == 1 or len(second_colors) == 1):
        assert background == learned[0] and divider == learned[3]
        assert first_foreground is None or first_foreground == learned[1]
        assert second_foreground is None or second_foreground == learned[2]
    first_occupied = first != background
    second_occupied = second != background
    states = first_occupied.astype(np.int8) + 2 * second_occupied.astype(np.int8)
    expected = np.where(states == 0, result, background)
    assert np.array_equal(output_grid, expected)
    return {"layout": layout, "shape": (height, width), "colors": (background, first_foreground, second_foreground, divider, result), "states": states}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs = set()
    seen_pairs = set()
    learned = None
    training_evidence = np.zeros(4, dtype=bool)
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_example(pair)
            analysis = _analyze_pair(pair, learned)
            if learned is None:
                assert split == "train" and index == 0
                learned = analysis["colors"]
                assert len(set(learned)) == 5
                counts = np.bincount(analysis["states"].flat, minlength=4)
                assert np.all(counts > 0)
            assert analysis["colors"][4] == learned[4]
            if split == "train":
                training_evidence |= np.bincount(analysis["states"].flat, minlength=4) > 0
    assert np.all(training_evidence)


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(left["input"], right["input"]) and np.array_equal(left["output"], right["output"])
        for left, right in zip(first[split], second[split])
    ) for split in ("train", "test"))


def self_test() -> None:
    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, None, 1), (1005, None, 4), (1006, 2, None), (1007, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed, train_count, test_count))
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(1008, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts accepted")
    corpus = [generate_puzzle(seed) for seed in range(42)]
    assert len({_grid_key(puzzle["train"][0]["input"]) for puzzle in corpus}) >= 40
    assert len({len(puzzle["train"]) for puzzle in corpus}) >= 4
    assert len({len(puzzle["test"]) for puzzle in corpus}) >= 3
    layouts, shapes, backgrounds, state_signatures, modes = set(), set(), set(), set(), set()
    layout_modes, train_densities, test_densities = set(), set(), set()
    variable_shape_puzzles = variable_color_puzzles = variable_divider_puzzles = 0
    variable_foreground_puzzles = repeated_outputs = transfer_puzzles = 0
    foreground_swaps = role_reuses = both_exclusive = subtle_exclusive = 0
    near_stacked = near_side = alignment_discriminators = coherent_runs = 0
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        analyses = []
        learned = None
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair, learned)
                if learned is None:
                    learned = analysis["colors"]
                analyses.append(analysis)
                layouts.add(analysis["layout"])
                shapes.add(analysis["shape"])
                backgrounds.add(analysis["colors"][0])
                states = analysis["states"]
                state_signatures.add((states.shape, tuple(int(value) for value in states.flat)))
                counts = np.bincount(states.flat, minlength=4)
                first = int(counts[1] + counts[3])
                second = int(counts[2] + counts[3])
                first_density = first / states.size
                second_density = second / states.size
                density_set = train_densities if split == "train" else test_densities
                if 0.30 <= first_density <= 0.70 and 0.30 <= second_density <= 0.70:
                    density_set.add("moderate/moderate")
                if first_density <= 0.25 and second_density <= 0.25:
                    density_set.add("sparse/sparse")
                if first_density >= 0.60 and second_density >= 0.60:
                    density_set.add("dense/dense")
                if min(first_density, second_density) <= 0.25 and max(first_density, second_density) >= 0.60:
                    density_set.add("sparse/dense")
                both_exclusive += counts[1] >= 2 and counts[2] >= 2
                subtle_exclusive += min(counts[1], counts[2]) <= 1
                height, width = states.shape
                near_stacked += analysis["layout"] == "stacked" and height >= 12 and width >= 25
                near_side += analysis["layout"] == "side" and height >= 25 and width >= 12
                first_empty = (states & 1) == 0
                second_empty = (states & 2) == 0
                alignment_discriminators += not np.array_equal(states == 0, first_empty & np.fliplr(second_empty))
                coherent_runs += bool(np.any((states[:, :-1] == 0) & (states[:, 1:] == 0)) or np.any((states[:-1] == 0) & (states[1:] == 0)))
                if first == second == 0:
                    modes.add("both_empty")
                if int(counts[0]) == 0:
                    modes.add("full_union")
                if int(counts[1]) == int(counts[2]) == 0 and first > 0:
                    modes.add("identical")
                if int(counts[3]) == 0 and first > 0 and second > 0:
                    modes.add("disjoint")
                if first == 0 or second == 0:
                    modes.add("empty_panel")
                if first == states.size or second == states.size:
                    modes.add("full_panel")
                if first > 0 and second > 0 and (counts[1] == 0 or counts[2] == 0):
                    modes.add("contained")
                assert np.array_equal(pair["output"] == learned[4], states == 0)
        variable_shape_puzzles += len({item["shape"] for item in analyses}) > 1
        variable_color_puzzles += len({item["colors"][0] for item in analyses}) > 1
        variable_divider_puzzles += len({item["colors"][3] for item in analyses}) > 1
        variable_foreground_puzzles += len({item["colors"][1:3] for item in analyses}) > 1
        train_layouts = {item["layout"] for item in analyses[:len(puzzle["train"])]}
        test_layouts = {item["layout"] for item in analyses[len(puzzle["train"]):]}
        layout_modes.add(tuple(sorted({item["layout"] for item in analyses})))
        transfer_puzzles += not test_layouts <= train_layouts
        for first_analysis in analyses:
            for second_analysis in analyses:
                if first_analysis is second_analysis:
                    continue
                first_colors = first_analysis["colors"]
                second_colors = second_analysis["colors"]
                foreground_swaps += first_colors[1] is not None and first_colors[2] is not None and first_colors[1] == second_colors[2] and first_colors[2] == second_colors[1]
                role_reuses += any(color in second_colors[0::3] for color in first_colors[1:3] if color is not None)
        outputs = [_grid_key(pair["output"]) for split in ("train", "test") for pair in puzzle[split]]
        repeated_outputs += len(outputs) != len(set(outputs))
    assert layouts == {"stacked", "side"}
    assert layout_modes == {("stacked",), ("side",), ("side", "stacked")}
    assert len(shapes) >= 30 and len(backgrounds) >= 8 and len(state_signatures) >= 200
    assert variable_shape_puzzles >= 20 and variable_color_puzzles >= 15
    assert variable_divider_puzzles >= 15 and variable_foreground_puzzles >= 15
    assert transfer_puzzles >= 5 and foreground_swaps > 0 and role_reuses > 0
    assert near_stacked > 0 and near_side > 0
    assert both_exclusive > 20 and subtle_exclusive > 20
    assert alignment_discriminators > 50 and coherent_runs > 50
    assert {"moderate/moderate", "sparse/sparse", "dense/dense", "sparse/dense"} <= train_densities
    assert {"moderate/moderate", "sparse/sparse", "dense/dense", "sparse/dense"} <= test_densities
    assert repeated_outputs >= 2
    assert {"both_empty", "full_union", "identical", "disjoint", "empty_panel", "full_panel", "contained"} <= modes
