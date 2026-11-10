import numpy as np


def _rows(*rows: str) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _rows(
            "03000300030",
            "00000000000",
            "00030000300",
            "00000000000",
            "03000000000",
            "55555555555",
            "00000000007",
            "70000000700",
            "00707000000",
            "00000000000",
            "70000070007",
        ),
        _rows(
            "03000300037",
            "70000000700",
            "00737000300",
            "00000000000",
            "73000070007",
        ),
    ),
    (
        _rows(
            "00000800000",
            "00000000000",
            "00080008000",
            "00000000000",
            "08000000080",
            "55555555555",
            "00000000000",
            "00001010000",
            "00000000000",
            "00100000100",
            "10000000001",
        ),
        _rows(
            "00000800000",
            "00001010000",
            "00080008000",
            "00100000100",
            "18000000081",
        ),
    ),
    (
        _rows(
            "00000000000",
            "01011010010",
            "00000010000",
            "00101001000",
            "00000000001",
            "55555555555",
            "20000200002",
            "00000000000",
            "00000000000",
            "00000000000",
            "20002000200",
        ),
        _rows(
            "20000200002",
            "01011010010",
            "00000010000",
            "00101001000",
            "20002000201",
        ),
    ),
    (
        _rows(
            "00000000007",
            "00770000700",
            "00000070000",
            "07070000007",
            "00000070000",
            "55555555555",
            "60006060000",
            "00000000006",
            "60006000006",
            "00000000060",
            "60060000000",
        ),
        _rows(
            "60006060007",
            "00770000706",
            "60006070006",
            "07070000067",
            "60060070000",
        ),
    ),
)


def _validate_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)

    if num_train is not None and num_test is not None:
        resolved_train = int(num_train)
        resolved_test = int(num_test)

        if resolved_train <= resolved_test:
            raise ValueError("num_train must be greater than num_test")

        return resolved_train, resolved_test

    if num_train is None and num_test is None:
        seed_value = int(seed)
        resolved_test = 1 + seed_value % 4
        minimum_train = max(2, resolved_test + 1)
        possible_train_counts = 11 - minimum_train
        resolved_train = (
            minimum_train
            + (seed_value // 4) % possible_train_counts
        )
        return resolved_train, resolved_test

    if num_train is None:
        resolved_test = int(num_test)
        minimum_train = max(2, resolved_test + 1)
        resolved_train = int(rng.integers(minimum_train, 11))
        return resolved_train, resolved_test

    resolved_train = int(num_train)
    maximum_test = min(4, resolved_train - 1)
    resolved_test = int(rng.integers(1, maximum_test + 1))
    return resolved_train, resolved_test


def _grid_key(grid: np.ndarray) -> tuple:
    canonical = np.ascontiguousarray(grid, dtype=np.int8)
    return canonical.shape, canonical.tobytes()


def _is_source_pair(pair: dict) -> bool:
    return any(
        np.array_equal(pair["input"], source_input)
        and np.array_equal(pair["output"], source_output)
        for source_input, source_output in _SOURCE_PAIRS
    )


def _dimensions(rng: np.random.Generator, orientation: str, shift_size: str | None) -> tuple[int, int]:
    # Coordinates are canonicalized to the horizontal, stacked view. A
    # vertical example is its transpose after the two panels are composed.
    if shift_size == "train":
        return int(rng.integers(3, 7)), int(rng.integers(4, 11))
    if shift_size == "test":
        return int(rng.integers(10, 15)), int(rng.integers(16, 31))
    kind = int(rng.integers(5))
    if kind == 0:  # compact
        height, width = int(rng.integers(3, 6)), int(rng.integers(4, 7))
    elif kind == 1:  # ordinary
        height, width = int(rng.integers(4, 11)), int(rng.integers(7, 15))
    elif kind == 2:  # wide
        height, width = int(rng.integers(3, 8)), int(rng.integers(15, 31))
    elif kind == 3:  # tall
        height, width = int(rng.integers(10, 15)), int(rng.integers(4, 13))
    else:  # near the input size limit
        height, width = int(rng.integers(12, 15)), int(rng.integers(24, 31))
    if orientation == "vertical":
        # Transposition turns a wide canonical panel into a tall visible one.
        # The canonical height must be <=14 for a 2*height+1 input.
        return height, width
    return height, width


def _add_mark(mask: np.ndarray, other: np.ndarray, rng: np.random.Generator) -> None:
    vacancies = np.argwhere(~mask & ~other)
    if len(vacancies):
        row, col = vacancies[int(rng.integers(len(vacancies)))]
        mask[row, col] = True


def _add_shape(
    upper: np.ndarray,
    lower: np.ndarray,
    rng: np.random.Generator,
    kind: int,
) -> None:
    height, width = upper.shape
    if kind == 0:  # short horizontal stroke
        length = min(width, int(rng.integers(2, 6)))
        row = int(rng.integers(height))
        col = int(rng.integers(width - length + 1))
        cells = [(row, col + offset) for offset in range(length)]
    elif kind == 1:  # short vertical stroke
        length = min(height, int(rng.integers(2, 5)))
        row = int(rng.integers(height - length + 1))
        col = int(rng.integers(width))
        cells = [(row + offset, col) for offset in range(length)]
    elif kind == 2:  # compact cluster
        row = int(rng.integers(height - 1))
        col = int(rng.integers(width - 1))
        cells = [(row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1)]
    else:  # contiguous composite with complementary contributions
        length = min(width, int(rng.integers(4, 8)))
        row = int(rng.integers(height))
        col = int(rng.integers(width - length + 1))
        for offset in range(length):
            upper[row, col + offset] = False
            lower[row, col + offset] = False
        for offset in range(length):
            (upper if offset % 2 == 0 else lower)[row, col + offset] = True
        return
    target = upper if rng.random() < 0.5 else lower
    other = lower if target is upper else upper
    for row, col in cells:
        other[row, col] = False
        target[row, col] = True


def _sample_masks(
    rng: np.random.Generator,
    height: int,
    width: int,
    regime: str,
    diagnostic: bool,
    multicolor: bool,
    dense_shift: bool,
) -> tuple[np.ndarray, np.ndarray]:
    upper = np.zeros((height, width), dtype=bool)
    lower = np.zeros((height, width), dtype=bool)
    if not dense_shift and not diagnostic and not multicolor and rng.random() < 0.18:
        # A distinct sparse regime preserves the source's isolated-dot cases.
        # Place points without same-layer edge adjacency or aligned collisions.
        for mask, other in ((upper, lower), (lower, upper)):
            count = int(rng.integers(1, min(4, height + 1)))
            for _ in range(count):
                safe = ~mask & ~other
                safe[1:] &= ~mask[:-1]
                safe[:-1] &= ~mask[1:]
                safe[:, 1:] &= ~mask[:, :-1]
                safe[:, :-1] &= ~mask[:, 1:]
                vacancies = np.argwhere(safe)
                if not len(vacancies):
                    break
                row, col = vacancies[int(rng.integers(len(vacancies)))]
                mask[row, col] = True
        return upper, lower
    density = (0 if regime == "source_train" else 2) if dense_shift else int(rng.choice(3, p=[0.30, 0.42, 0.28]))
    # Training and test draw from overlapping distributions. The denser tail
    # remains a little more likely in tests, as in the source example.
    if regime == "target_test" and rng.random() < 0.35:
        density = 2
    state_weights = (
        [0.30, 0.25, 0.25, 0.20],
        [0.12, 0.20, 0.20, 0.48],
        [0.03, 0.10, 0.10, 0.77],
    )[density]
    states = list(int(v) for v in rng.choice(4, size=height, p=state_weights))
    if dense_shift and regime == "target_test":
        states = [3] * height
    if diagnostic:
        # The first demonstration collectively establishes all essential
        # layer relationships, even when only two training pairs are asked for.
        for row, state in enumerate(rng.permutation([1, 2, 3])):
            states[row] = int(state)
    for row, state in enumerate(states):
        if state == 0:
            continue
        capacity = min(width - 1, 3 if density == 0 else 5 if density == 1 else 8)
        minimum_count = 3 if dense_shift and regime == "target_test" else 1
        upper_count = int(rng.integers(minimum_count, capacity + 1)) if state & 1 else 0
        lower_count = int(rng.integers(minimum_count, capacity + 1)) if state & 2 else 0
        if upper_count + lower_count > width:
            excess = upper_count + lower_count - width
            if upper_count >= lower_count:
                upper_count -= excess
            else:
                lower_count -= excess
        columns = rng.choice(width, size=upper_count + lower_count, replace=False)
        upper[row, columns[:upper_count]] = True
        lower[row, columns[upper_count:]] = True
    # Geometry is independent of palette and orientation. Some pairs remain
    # sparse dots; others have strokes, clusters, or joined composite pieces.
    if rng.random() < 0.65:
        _add_shape(upper, lower, rng, int(rng.integers(4)))
    if rng.random() < 0.20:
        _add_shape(upper, lower, rng, int(rng.integers(4)))
    minimum = 2 if multicolor else 1
    for mask, other in ((upper, lower), (lower, upper)):
        while int(mask.sum()) < minimum:
            _add_mark(mask, other, rng)
    if diagnostic:
        # Shapes can cross a diagnostic row. Restore three unambiguous row
        # relationships after adding motifs, regardless of the random draw.
        diagnostic_rows = rng.choice(height, size=3, replace=False)
        for row, state in zip(diagnostic_rows, (2, 1, 3)):
            upper[row] = False
            lower[row] = False
            columns = rng.choice(width, size=minimum * (2 if state == 3 else 1),
                                 replace=False)
            if state & 1:
                upper[row, columns[:minimum]] = True
            if state & 2:
                lower[row, columns[-minimum:]] = True
    return upper, lower


def _paint_panel(
    mask: np.ndarray,
    colors: tuple[int, ...],
    background: int,
    rng: np.random.Generator,
) -> np.ndarray:
    panel = np.full(mask.shape, background, dtype=np.int8)
    positions = np.argwhere(mask)
    rng.shuffle(positions)
    for index, (row, col) in enumerate(positions):
        color = colors[index] if index < len(colors) else colors[int(rng.integers(len(colors)))]
        panel[row, col] = color
    return panel


def _make_pair(
    rng: np.random.Generator,
    regime: str,
    orientation: str,
    height: int,
    width: int,
    background: int,
    divider: int,
    upper_colors: tuple[int, ...],
    lower_colors: tuple[int, ...],
    diagnostic: bool,
    dense_shift: bool,
) -> dict:
    upper_mask, lower_mask = _sample_masks(
        rng, height, width, regime, diagnostic,
        len(upper_colors) > 1 or len(lower_colors) > 1, dense_shift,
    )
    upper = _paint_panel(upper_mask, upper_colors, background, rng)
    lower = _paint_panel(lower_mask, lower_colors, background, rng)
    input_grid = np.full((2 * height + 1, width), background, dtype=np.int8)
    input_grid[:height] = upper
    input_grid[height] = divider
    input_grid[height + 1:] = lower
    output = upper.copy()
    output[lower_mask] = lower[lower_mask]
    if orientation == "vertical":
        input_grid = input_grid.T.copy()
        output = output.T.copy()
    return {"input": input_grid, "output": output}


def _palette_schedule(
    rng: np.random.Generator,
    count_train: int,
    count_test: int,
    stable: bool,
    multi_flags: list[bool],
) -> list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]:
    order = [int(v) for v in rng.permutation(10)]
    if stable:
        multi = multi_flags[0]
        payload = order[:4] if multi else order[:2]
        bg, divider = order[-2:]
        upper = tuple(payload[:2]) if multi else (payload[0],)
        lower = tuple(payload[2:]) if multi else (payload[1],)
        return [(bg, divider, upper, lower)] * (count_train + count_test)
    # Restrict training payloads to five digits, leaving genuinely unseen
    # payload digits available for test examples. Structural colors remain
    # independently sampled from the other visible colors in each pair.
    train_pool = order[:5]
    reserved = order[5:8]
    schedule = []
    prior_bg = prior_divider = None
    first_lower = None
    for index in range(count_train + count_test):
        is_test = index >= count_train
        multi = multi_flags[index]
        colors = [int(v) for v in rng.choice(train_pool, size=4 if multi else 2, replace=False)]
        if index == 1 and first_lower is not None:
            colors[0] = first_lower
            remaining = [v for v in train_pool if v != first_lower]
            others = [int(v) for v in rng.choice(remaining, size=len(colors) - 1, replace=False)]
            colors[1:] = others
        if is_test and index == count_train:
            colors[0] = reserved[0]
        if index == 0:
            first_lower = colors[2] if multi else colors[1]
        upper = tuple(colors[:2]) if multi else (colors[0],)
        lower = tuple(colors[2:]) if multi else (colors[1],)
        eligible = [v for v in order if v not in colors]
        bg_options = [v for v in eligible if v != prior_bg] or eligible
        bg = int(rng.choice(bg_options))
        divider_options = [v for v in eligible if v != bg and v != prior_divider]
        divider = int(rng.choice(divider_options))
        schedule.append((bg, divider, upper, lower))
        prior_bg, prior_divider = bg, divider
    return schedule


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    count_train, count_test = _resolve_counts(seed, rng, num_train, num_test)
    stable = bool(rng.random() < 0.23)
    multi = bool(rng.random() < 0.25)
    mixed = bool(rng.random() < 0.42)
    dense_shift = bool(rng.random() < 0.18)
    multi_flags = [multi] * (count_train + count_test) if stable else [False] * (count_train + count_test)
    if multi and not stable:
        multi_flags[0] = True
        multi_flags[count_train] = True
        for index in range(1, count_train + count_test):
            if index != count_train:
                multi_flags[index] = bool(rng.random() < 0.45)
    palette = _palette_schedule(rng, count_train, count_test, stable, multi_flags)
    orientations = ["horizontal"] * (count_train + count_test)
    if mixed:
        orientations[1] = "vertical"
        for index in range(2, len(orientations)):
            orientations[index] = "vertical" if rng.random() < 0.5 else "horizontal"
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=len(orientations), dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_geometry = None
    for index, orientation in enumerate(orientations):
        split = "train" if index < count_train else "test"
        pair_rng = np.random.default_rng(pair_seeds[index])
        bg, divider, upper, lower = palette[index]
        for attempt in range(100):
            shift_size = ("train" if split == "train" else "test") if dense_shift else None
            height, width = _dimensions(pair_rng, orientation, shift_size)
            # Pair dimensions are intentionally free, even when colors are
            # stable. Two training pairs show at least two sizes.
            if index == 1 and (height, width) == first_geometry:
                continue
            pair = _make_pair(pair_rng, "source_train" if split == "train" else "target_test",
                              orientation, height, width, bg, divider, upper, lower,
                              diagnostic=index == 0, dense_shift=dense_shift)
            key = _grid_key(pair["input"])
            if key not in seen and not _is_source_pair(pair):
                break
        else:
            raise RuntimeError("could not construct a unique pair")
        if index == 0:
            first_geometry = height, width
        seen.add(key)
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair: dict) -> dict:
    input_grid, output = pair["input"], pair["output"]
    if input_grid.shape[0] == 2 * output.shape[0] + 1 and input_grid.shape[1] == output.shape[1]:
        orientation = "horizontal"
        canonical_input, canonical_output = input_grid, output
    else:
        assert input_grid.shape[1] == 2 * output.shape[1] + 1
        assert input_grid.shape[0] == output.shape[0]
        orientation = "vertical"
        canonical_input, canonical_output = input_grid.T, output.T
    height, width = canonical_output.shape
    assert canonical_input.shape == (2 * height + 1, width)
    divider_values = np.unique(canonical_input[height])
    assert len(divider_values) == 1
    divider = int(divider_values[0])
    upper = canonical_input[:height]
    lower = canonical_input[height + 1:]
    upper_colors = {int(v) for v in np.unique(upper)}
    lower_colors = {int(v) for v in np.unique(lower)}
    shared = upper_colors & lower_colors
    assert len(shared) == 1
    bg = next(iter(shared))
    assert divider not in upper_colors | lower_colors
    upper_payload = upper_colors - {bg}
    lower_payload = lower_colors - {bg}
    assert 1 <= len(upper_payload) <= 2
    assert 1 <= len(lower_payload) <= 2
    upper_mask = upper != bg
    lower_mask = lower != bg
    assert not np.any(upper_mask & lower_mask)
    assert np.any(~upper_mask) and np.any(~lower_mask)
    expected = upper.copy()
    expected[lower_mask] = lower[lower_mask]
    assert np.array_equal(canonical_output, expected)
    upper_rows = np.any(upper_mask, axis=1)
    lower_rows = np.any(lower_mask, axis=1)
    return {
        "orientation": orientation, "height": height, "width": width,
        "background": bg, "divider": divider,
        "upper_colors": upper_payload, "lower_colors": lower_payload,
        "upper_mask": upper_mask, "lower_mask": lower_mask,
        "row_states": set(int(u) + 2 * int(l) for u, l in zip(upper_rows, lower_rows)),
        "foreground_count": int(upper_mask.sum() + lower_mask.sum()),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(pair)
            analyses[split].append(_analyze_pair(pair))
    train, test = analyses["train"], analyses["test"]
    assert {1, 2, 3} <= set().union(*(item["row_states"] for item in train))
    assert len({(item["height"], item["width"]) for item in train}) >= 2
    train_orientations = {item["orientation"] for item in train}
    assert {item["orientation"] for item in test} <= train_orientations
    if any(len(item["upper_colors"]) > 1 or len(item["lower_colors"]) > 1 for item in test):
        assert any(len(item["upper_colors"]) > 1 or len(item["lower_colors"]) > 1 for item in train)


def _puzzles_equal(left: dict, right: dict) -> bool:
    return all(
        len(left[split]) == len(right[split])
        and all(np.array_equal(a[key], b[key])
                for a, b in zip(left[split], right[split]) for key in ("input", "output"))
        for split in ("train", "test")
    )


def _predict_from_demonstrations(train_pairs: list, test_input: np.ndarray) -> np.ndarray:
    # This uses only the returned demonstrations and test input. It derives
    # the shared background from each pair's panels, then overlays all
    # non-background cells. No generator state or test output is consulted.
    assert train_pairs
    for pair in train_pairs:
        _analyze_pair(pair)
    shape = test_input.shape
    if shape[0] % 2 == 1 and shape[0] > shape[1] / 2:
        candidates = ["horizontal", "vertical"]
    else:
        candidates = ["vertical", "horizontal"]
    for orientation in candidates:
        grid = test_input if orientation == "horizontal" else test_input.T
        if grid.shape[0] % 2 != 1:
            continue
        h = (grid.shape[0] - 1) // 2
        upper, divider, lower = grid[:h], grid[h], grid[h + 1:]
        if upper.shape != lower.shape or len(set(int(v) for v in divider)) != 1:
            continue
        shared = set(int(v) for v in np.unique(upper)) & set(int(v) for v in np.unique(lower))
        if len(shared) != 1 or int(divider[0]) in shared:
            continue
        bg = next(iter(shared))
        output = upper.copy()
        output[lower != bg] = lower[lower != bg]
        return output if orientation == "horizontal" else output.T.copy()
    raise AssertionError("could not identify the separator and background")


def self_test() -> None:
    automatic = generate_puzzle(12345)
    explicit = generate_puzzle(90210, 7, 3)
    assert _puzzles_equal(automatic, generate_puzzle(12345))
    assert _puzzles_equal(explicit, generate_puzzle(90210, 7, 3))
    assert not _puzzles_equal(generate_puzzle(111, 6, 2), generate_puzzle(112, 6, 2))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(3000 + train_count + test_count, train_count, test_count)
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
    for index, arguments in enumerate((
        {"num_train": None, "num_test": 1}, {"num_train": None, "num_test": 4},
        {"num_train": 2, "num_test": None}, {"num_train": 10, "num_test": None},
    )):
        puzzle = generate_puzzle(4000 + index, **arguments)
        for key, requested in (("train", arguments["num_train"]), ("test", arguments["num_test"])):
            if requested is not None:
                assert len(puzzle[key]) == requested
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(17, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
    train_counts, test_counts, geometries, palettes, layouts = set(), set(), set(), set(), set()
    orient, multicolor, stable, varied, strokes, sparse_tests, dense_train = set(), 0, 0, 0, 0, 0, 0
    reused_roles, fresh_tests, within_size, within_color = 0, 0, 0, 0
    dense_tests = point_pairs = compact = near_limit = vertical_multi = joined = 0
    multi_single = background_varies = divider_varies = 0
    max_layer_row = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        all_pairs = puzzle["train"] + puzzle["test"]
        features = [_analyze_pair(pair) for pair in all_pairs]
        train_features = features[:len(puzzle["train"])]
        test_features = features[len(puzzle["train"]):]
        within_size += len({(f["height"], f["width"]) for f in features}) > 1
        within_color += len({(f["background"], f["divider"]) for f in features}) > 1
        background_varies += len({f["background"] for f in features}) > 1
        divider_varies += len({f["divider"] for f in features}) > 1
        multi_single += (any(len(f["upper_colors"]) > 1 for f in features)
                         and any(len(f["upper_colors"]) == 1 for f in features))
        stable += len({(f["background"], f["divider"],
                        tuple(sorted(f["upper_colors"])), tuple(sorted(f["lower_colors"]))) for f in features}) == 1
        varied += len({(f["background"], f["divider"]) for f in features}) > 1
        train_payload = set().union(*(f["upper_colors"] | f["lower_colors"] for f in train_features))
        fresh_tests += any(bool((f["upper_colors"] | f["lower_colors"]) - train_payload) for f in test_features)
        reused_roles += bool(set().union(*(f["lower_colors"] for f in train_features)) &
                             set().union(*(f["upper_colors"] for f in train_features)))
        sparse_tests += any(f["foreground_count"] < max(t["foreground_count"] for t in train_features)
                            for f in test_features)
        dense_train += any(f["foreground_count"] >= min(t["foreground_count"] for t in test_features)
                           for f in train_features)
        dense_tests += min(f["foreground_count"] for f in test_features) > max(
            f["foreground_count"] for f in train_features)
        for pair, feature in zip(all_pairs, features):
            orient.add(feature["orientation"])
            multicolor += len(feature["upper_colors"]) > 1 or len(feature["lower_colors"]) > 1
            vertical_multi += feature["orientation"] == "vertical" and len(feature["upper_colors"]) > 1
            compact += feature["height"] <= 5 and feature["width"] <= 6
            near_limit += max(pair["input"].shape) >= 27
            geometries.add((feature["height"], feature["width"], feature["orientation"]))
            palettes.add((feature["background"], feature["divider"],
                          tuple(sorted(feature["upper_colors"])), tuple(sorted(feature["lower_colors"]))))
            layouts.add((feature["height"], feature["width"], feature["orientation"],
                         feature["upper_mask"].tobytes(), feature["lower_mask"].tobytes()))
            output = _predict_from_demonstrations(puzzle["train"], pair["input"])
            assert np.array_equal(output, pair["output"])
            upper, lower = feature["upper_mask"], feature["lower_mask"]
            adjacent = bool(np.any(upper[:, 1:] & upper[:, :-1]) or np.any(lower[:, 1:] & lower[:, :-1])
                            or np.any(upper[1:] & upper[:-1]) or np.any(lower[1:] & lower[:-1]))
            strokes += adjacent
            point_pairs += not adjacent
            joined += bool(np.any(upper[:, 1:] & lower[:, :-1]) or np.any(lower[:, 1:] & upper[:, :-1])
                           or np.any(upper[1:] & lower[:-1]) or np.any(lower[1:] & upper[:-1]))
            max_layer_row = max(max_layer_row, int(upper.sum(axis=1).max()), int(lower.sum(axis=1).max()))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(geometries) >= 60 and len(palettes) >= 50 and len(layouts) >= 300
    assert orient == {"horizontal", "vertical"}
    assert multicolor >= 20 and stable >= 3 and varied >= 20 and strokes >= 60
    assert sparse_tests >= 5 and dense_train >= 5 and dense_tests >= 5
    assert reused_roles >= 20 and fresh_tests >= 15
    assert within_size >= 40 and within_color >= 20
    assert background_varies >= 20 and divider_varies >= 20 and multi_single >= 3
    assert compact >= 20 and near_limit >= 20 and vertical_multi >= 3
    assert joined >= 80 and point_pairs >= 20 and max_layer_row >= 5
