"""Generate aligned, two-panel mask-union ARC puzzles."""

import numpy as np


_SOURCE_INPUT_ROWS = frozenset({
    ("404330", "400300", "004300"), ("440300", "000003", "400000"),
    ("000030", "400000", "004330"), ("004030", "044303", "440003"),
    ("440330", "400300", "000003"), ("004030", "040333", "400300"),
    ("044300", "400330", "040300"),
})


def _validate_optional_count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _validate_optional_count("num_train", num_train, 2, 10)
    num_test = _validate_optional_count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return num_train, num_test
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    return num_train, num_test


def _orientation(input_grid):
    height, width = input_grid.shape
    if height % 2 and width % 2 == 0:
        return "horizontal"
    if width % 2 and height % 2 == 0:
        return "vertical"
    raise AssertionError("input dimensions must identify one equal-panel split")


def _split(input_grid):
    if _orientation(input_grid) == "horizontal":
        width = input_grid.shape[1] // 2
        return input_grid[:, :width], input_grid[:, width:]
    height = input_grid.shape[0] // 2
    return input_grid[:height, :], input_grid[height:, :]


def _local_masks(input_grid):
    first, second = _split(input_grid)
    first_colors = set(int(x) for x in np.unique(first))
    second_colors = set(int(x) for x in np.unique(second))
    shared = first_colors & second_colors
    assert len(shared) == 1
    background = next(iter(shared))
    assert len(first_colors - {background}) <= 1
    assert len(second_colors - {background}) <= 1
    return background, first != background, second != background


def _sample_dimensions(rng, orientation):
    # Odd unsplit and even doubled axes identify orientation from an input alone.
    odds = np.arange(1, 30, 2)
    odd_weights = np.array([7, 12, 12, 9, 7, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2], float)
    odd_weights /= odd_weights.sum()
    shorts = np.arange(1, 16)
    short_weights = np.array([5, 8, 10, 10, 9, 8, 7, 6, 5, 4, 3, 3, 3, 3, 3], float)
    short_weights /= short_weights.sum()
    for _ in range(20):
        odd = int(rng.choice(odds, p=odd_weights))
        short = int(rng.choice(shorts, p=short_weights))
        height, width = (odd, short) if orientation == "horizontal" else (short, odd)
        if height * width >= 4:
            return height, width
    return (3, 3)


def _mask_scores(rng, height, width, style):
    rows, cols = np.indices((height, width))
    jitter = rng.random((height, width)) * 0.18
    if style == "scatter":
        return rng.random((height, width))
    if style == "bar":
        if rng.integers(2):
            return abs(rows - rng.uniform(0, height - 1)) + jitter
        return abs(cols - rng.uniform(0, width - 1)) + jitter
    if style == "block":
        row, col = rng.uniform(0, height - 1), rng.uniform(0, width - 1)
        rr = rng.uniform(0.5, max(1.0, height / 3))
        cr = rng.uniform(0.5, max(1.0, width / 3))
        return np.maximum(abs(rows - row) / rr, abs(cols - col) / cr) + jitter
    if style == "line":
        angle = rng.uniform(0, 2 * np.pi)
        row, col = rng.uniform(0, height - 1), rng.uniform(0, width - 1)
        return abs((rows - row) * np.cos(angle) + (cols - col) * np.sin(angle)) + jitter
    if style == "diagonal":
        slope = float(rng.choice([-1, 1]))
        offset = rng.uniform(-0.4, 0.4)
        return abs(rows / max(1, height - 1) - slope * cols / max(1, width - 1)
                   - offset - (slope < 0)) + jitter
    if style == "clusters":
        score = np.full((height, width), np.inf)
        for _ in range(int(rng.integers(2, 5))):
            row, col = rng.uniform(0, height - 1), rng.uniform(0, width - 1)
            score = np.minimum(score, (rows - row) ** 2 + (cols - col) ** 2)
        return score + jitter
    if style == "holes":
        distance = np.full((height, width), np.inf)
        for _ in range(int(rng.integers(1, 4))):
            row, col = rng.uniform(0, height - 1), rng.uniform(0, width - 1)
            distance = np.minimum(distance, (rows - row) ** 2 + (cols - col) ** 2)
        return -distance + jitter
    noise = rng.random((height, width))
    return (noise + np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
            + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)) / 5 + jitter


def _sample_mask(rng, height, width, regime):
    styles = ("scatter", "bar", "block", "line", "diagonal", "clusters", "holes", "irregular")
    weights = ([22, 14, 14, 10, 10, 12, 8, 10] if regime == "source_train"
               else [12, 12, 12, 12, 12, 14, 12, 14])
    style = str(rng.choice(styles, p=np.array(weights) / sum(weights)))
    bands = ((0.02, 0.16), (0.16, 0.34), (0.34, 0.57), (0.57, 0.80), (0.80, 0.97))
    band_weights = ([21, 25, 29, 18, 7] if regime == "source_train"
                    else [29, 25, 25, 14, 7])
    low, high = bands[int(rng.choice(len(bands), p=np.array(band_weights) / sum(band_weights)))]
    count = height * width
    active_count = min(count - 1, max(1, int(round(rng.uniform(low, high) * count))))
    mask = np.zeros(count, dtype=bool)
    scores = _mask_scores(rng, height, width, style)
    mask[np.argsort(scores.ravel(), kind="stable")[:active_count]] = True
    return mask.reshape(height, width)


def _make_masks(rng, height, width, regime, guarantee_truth_table):
    first = _sample_mask(rng, height, width, regime)
    second = _sample_mask(rng, height, width, regime)
    # Most masks are placed independently; others deliberately vary intersection.
    relation = int(rng.choice(7, p=[0.60, 0.09, 0.09, 0.06, 0.07, 0.06, 0.03]))
    if relation == 1:
        second &= ~first
    elif relation == 2:
        second &= first
    elif relation == 3:
        second = first.copy()
    elif relation == 4:
        (first if rng.integers(2) else second)[:] = False
    elif relation == 5:
        second = ~first
    elif relation == 6:
        first[:] = False
        second[:] = False
    if guarantee_truth_table:
        positions = rng.choice(height * width, size=4, replace=False)
        for index, (one, two) in zip(positions, ((0, 0), (1, 0), (0, 1), (1, 1))):
            first.ravel()[index] = bool(one)
            second.ravel()[index] = bool(two)
    for mask in (first, second):
        if np.all(mask):
            mask.ravel()[int(rng.integers(height * width))] = False
    return first, second


def _pair_palette(rng, base, color_mode, pair_index):
    background, first, second, result = base
    if pair_index == 0 or color_mode == 0:
        return background, first, second, result
    if color_mode == 1 and pair_index == 1 and rng.random() < 0.45:
        return background, second, first, result
    available = [color for color in range(10) if color != result]
    if color_mode == 2:
        choices = ([color for color in available if color != background]
                   if pair_index == 1 else available)
        background = int(rng.choice(choices))
    choices = [color for color in available if color != background]
    first, second = (int(x) for x in rng.choice(choices, size=2, replace=False))
    if color_mode == 1 and pair_index == 1 and (first, second) == base[1:3]:
        first, second = second, first
    return background, first, second, result


def _make_pair(rng, regime, orientation, dimensions, palette, guarantee_truth_table):
    height, width = dimensions
    background, first_color, second_color, result = palette
    first_mask, second_mask = _make_masks(rng, height, width, regime, guarantee_truth_table)
    first = np.where(first_mask, first_color, background).astype(np.int8)
    second = np.where(second_mask, second_color, background).astype(np.int8)
    input_grid = (np.concatenate((first, second), axis=1) if orientation == "horizontal"
                  else np.concatenate((first, second), axis=0))
    output_grid = np.where(first_mask | second_mask, result, background).astype(np.int8)
    return {"input": input_grid, "output": output_grid}


def _array_key(grid):
    return (grid.shape, grid.tobytes())


def _grid_rows(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _training_result(train_pairs):
    result_colors = set()
    observed_states = set()
    for pair in train_pairs:
        background, first, second = _local_masks(pair["input"])
        state = first.astype(np.int8) + 2 * second.astype(np.int8)
        observed_states.update(int(x) for x in np.unique(state))
        result_colors.update(int(x) for x in np.unique(pair["output"][state != 0]))
        assert np.all(pair["output"][state == 0] == background)
    assert observed_states == {0, 1, 2, 3}
    assert len(result_colors) == 1
    return next(iter(result_colors))


def _predict_from_training(train_pairs, test_input):
    result = _training_result(train_pairs)
    background, first, second = _local_masks(test_input)
    assert result != background
    return np.where(first | second, result, background).astype(np.int8)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_grid, output_grid = pair["input"], pair["output"]
            input_key = _array_key(input_grid)
            pair_key = (input_key, _array_key(output_grid))
            assert input_key not in inputs and pair_key not in pairs
            inputs.add(input_key)
            pairs.add(pair_key)
            assert _grid_rows(input_grid) not in _SOURCE_INPUT_ROWS
            first_panel, second_panel = _split(input_grid)
            assert first_panel.shape == second_panel.shape == output_grid.shape
            background, first, second = _local_masks(input_grid)
            assert np.any(~first) and np.any(~second)
            if split == "train":
                assert np.all(output_grid[~(first | second)] == background)
    result = _training_result(puzzle["train"])
    for split in ("train", "test"):
        for pair in puzzle[split]:
            input_grid, output_grid = pair["input"], pair["output"]
            background, first, second = _local_masks(input_grid)
            assert result not in set(int(x) for x in np.unique(input_grid))
            assert np.array_equal(output_grid, np.where(first | second, result, background))
            if split == "test":
                assert np.array_equal(output_grid, _predict_from_training(puzzle["train"], input_grid))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    base = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    color_mode = int(rng.choice(3, p=[0.35, 0.30, 0.35]))
    layout_mode = int(rng.choice(3, p=[0.35, 0.35, 0.30]))
    classic = rng.random() < 0.12
    fixed_dimensions = None
    if classic:
        color_mode = 0
        layout_mode = 0
        fixed_dimensions = (int(rng.choice((3, 5, 7))), int(rng.integers(3, 8)))
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    first_dimensions = None
    for index in range(num_train + num_test):
        split = "train" if index < num_train else "test"
        regime = "source_train" if split == "train" else "target_test"
        if layout_mode == 0:
            orientation = "horizontal"
        elif layout_mode == 1:
            orientation = "vertical"
        elif index < 2:
            orientation = ("horizontal", "vertical")[index]
        else:
            orientation = str(rng.choice(("horizontal", "vertical")))
        if fixed_dimensions is not None:
            dimensions = fixed_dimensions
        else:
            for _ in range(30):
                dimensions = _sample_dimensions(rng, orientation)
                if index != 1 or dimensions != first_dimensions:
                    break
        if index == 0:
            first_dimensions = dimensions
        palette = _pair_palette(rng, base, color_mode, index)
        for _ in range(100):
            pair = _make_pair(rng, regime, orientation, dimensions, palette, index == 0)
            key = _array_key(pair["input"])
            if key not in seen_inputs and _grid_rows(pair["input"]) not in _SOURCE_INPUT_ROWS:
                seen_inputs.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError("could not construct a distinct pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple(tuple((_array_key(pair["input"]), _array_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert len({_fingerprint(p) for p in automatic}) == 42
    assert _fingerprint(generate_puzzle(123456, 7, 3)) == _fingerprint(generate_puzzle(123456, 7, 3))
    assert _fingerprint(generate_puzzle(37)) == _fingerprint(generate_puzzle(37))
    for seed, train_count, test_count in ((5001, 2, 1), (5002, 10, 1), (5003, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for seed, train_count, test_count in ((6001, None, 1), (6002, None, 4),
                                          (6003, 2, None), (6004, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (2.0, 1), (True, 1)):
        try:
            generate_puzzle(7000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    orientations, dimensions, palette_roles, state_patterns = set(), set(), set(), set()
    train_density, test_density, density_directions = [], [], set()
    relation_counts = {split: {name: 0 for name in ("disjoint", "contained", "identical",
                                                    "blank_panel", "blank_union", "full_union",
                                                    "low_overlap", "high_overlap", "thin", "large")}
                       for split in ("train", "test")}
    varying_dimensions = varying_backgrounds = mixed_orientations = swapped_colors = 0
    for puzzle in automatic:
        local_dimensions, local_backgrounds, local_orientations = set(), set(), set()
        foreground_pairs = []
        split_densities = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_grid, output_grid = pair["input"], pair["output"]
                background, first, second = _local_masks(input_grid)
                state = first.astype(np.int8) + 2 * second.astype(np.int8)
                active_fraction = float(np.count_nonzero(state)) / state.size
                counts = np.bincount(state.ravel(), minlength=4)
                categories = relation_counts[split]
                categories["disjoint"] += counts[3] == 0
                categories["contained"] += counts[1] == 0 or counts[2] == 0
                categories["identical"] += counts[1] == counts[2] == 0
                categories["blank_panel"] += not np.any(first) or not np.any(second)
                categories["blank_union"] += counts[0] == state.size
                categories["full_union"] += counts[0] == 0
                categories["thin"] += min(state.shape) == 1
                categories["large"] += max(state.shape) >= 20
                if state.size > counts[0]:
                    fraction = counts[3] / (state.size - counts[0])
                    categories["low_overlap"] += fraction <= 0.2
                    categories["high_overlap"] += fraction >= 0.6
                panels = _split(input_grid)
                foreground_pairs.append(tuple(next(iter(set(int(x) for x in np.unique(panel)) - {background}), None)
                                              for panel in panels))
                split_densities[split].append(active_fraction)
                (train_density if split == "train" else test_density).append(active_fraction)
                orientations.add(_orientation(input_grid))
                local_orientations.add(_orientation(input_grid))
                dimensions.add(output_grid.shape)
                local_dimensions.add(output_grid.shape)
                local_backgrounds.add(background)
                palette_roles.add((background, tuple(sorted(int(x) for x in np.unique(input_grid))),
                                   tuple(sorted(int(x) for x in np.unique(output_grid)))))
                state_patterns.add((state.shape, state.tobytes()))
                assert np.array_equal(output_grid, _predict_from_training(puzzle["train"], input_grid))
                assert _grid_rows(input_grid) not in _SOURCE_INPUT_ROWS
        varying_dimensions += len(local_dimensions) > 1
        varying_backgrounds += len(local_backgrounds) > 1
        mixed_orientations += len(local_orientations) > 1
        swapped_colors += any(pair == foreground_pairs[0][::-1] for pair in foreground_pairs[1:]
                              if None not in pair and None not in foreground_pairs[0])
        density_directions.add(int(np.sign(np.mean(split_densities["test"])
                                          - np.mean(split_densities["train"]))))
    assert orientations == {"horizontal", "vertical"}
    assert len(dimensions) >= 35 and len(state_patterns) >= 180
    assert len(palette_roles) >= 30
    assert varying_dimensions >= 30 and varying_backgrounds >= 8 and mixed_orientations >= 6
    assert min(train_density) <= 0.15 and max(train_density) >= 0.85
    assert min(test_density) <= 0.15 and max(test_density) >= 0.85
    assert -1 in density_directions and 1 in density_directions
    assert swapped_colors >= 2
    for split in ("train", "test"):
        assert all(count > 0 for count in relation_counts[split].values())
    corrupted = generate_puzzle(123456, 7, 3)
    old = int(corrupted["train"][0]["output"][0, 0])
    corrupted["train"][0]["output"][0, 0] = 9 if old != 9 else 8
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
