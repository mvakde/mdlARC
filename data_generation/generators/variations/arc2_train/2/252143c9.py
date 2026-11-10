import numpy as np


_CORNERS = (
    (-1, -1),  # upper-left
    (-1, 1),   # upper-right
    (1, -1),   # lower-left
    (1, 1),    # lower-right
)

_TRAIN_SIZES = (7, 9, 11)
_TEST_SIZES = (13, 15, 17, 19)

_SOURCE_PAIRS = (
    (
        (
            "73737770770",
            "77373777707",
            "37377700000",
            "73737770707",
            "77373777070",
            "77777077777",
            "22727777797",
            "77272797979",
            "72727797997",
            "77727777979",
            "72772779779",
        ),
        (
            "77777777770",
            "77777777707",
            "77777777077",
            "77777770777",
            "77777707777",
            "77777077777",
            "77777777777",
            "77777777777",
            "77777777777",
            "77777777777",
            "77777777777",
        ),
    ),
    (
        (
            "0707477",
            "0777747",
            "7077447",
            "7775777",
            "5577717",
            "7557117",
            "7757177",
        ),
        (
            "7777777",
            "7777777",
            "7777777",
            "7775777",
            "7757777",
            "7577777",
            "5777777",
        ),
    ),
    (
        (
            "8777777777767",
            "7787877766776",
            "7878787776777",
            "7778777767667",
            "8787877676776",
            "7877787666767",
            "7777774777777",
            "7447477727727",
            "4777777772277",
            "7474477727777",
            "7744477272227",
            "7477777772772",
            "7777477277277",
        ),
        (
            "7777777777777",
            "7777777777777",
            "7777777777777",
            "7777777777777",
            "7777777777777",
            "7777777777777",
            "7777774777777",
            "7777747777777",
            "7777477777777",
            "7774777777777",
            "7747777777777",
            "7477777777777",
            "4777777777777",
        ),
    ),
)


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_count_value(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return result


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_count_value("num_train", num_train, 2, 10)
    test_count = _validate_count_value("num_test", num_test, 1, 4)
    seed_key = abs(int(seed))

    if train_count is None and test_count is None:
        test_count = 1 + seed_key % 4
        train_span = 10 - test_count
        train_count = test_count + 1 + (seed_key // 4) % train_span
    elif train_count is None:
        train_span = 10 - test_count
        train_count = test_count + 1 + seed_key % train_span
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed_key % maximum_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    # These are presentation preferences, not hidden facts needed to solve a pair.
    pool_size = int(rng.choice((2, 3, 4), p=(0.28, 0.18, 0.54)))
    return {
        "anchor_background": int(rng.integers(0, 10)),
        "style": int(rng.integers(0, 3)),
        "corner_pool": tuple(int(x) for x in rng.permutation(4)[:pool_size]),
    }


def _sector_cells(size: int, corner: int) -> list[tuple[int, int]]:
    middle = size // 2
    row_step, col_step = _CORNERS[corner]
    rows = range(0, middle) if row_step < 0 else range(middle + 1, size)
    cols = range(0, middle) if col_step < 0 else range(middle + 1, size)
    return [(row, col) for row in rows for col in cols]


def _diagonal_cells(size: int, corner: int, include_center: bool = True) -> list[tuple[int, int]]:
    middle = size // 2
    row_step, col_step = _CORNERS[corner]
    return [
        (middle + distance * row_step, middle + distance * col_step)
        for distance in range(0 if include_center else 1, middle + 1)
    ]


def _grid_matches_rows(grid: np.ndarray, rows: tuple[str, ...]) -> bool:
    if grid.shape != (len(rows), len(rows[0])):
        return False
    return all(
        "".join(str(int(value)) for value in grid[row_index]) == expected
        for row_index, expected in enumerate(rows)
    )


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return any(
        _grid_matches_rows(input_grid, source_input)
        and _grid_matches_rows(output_grid, source_output)
        for source_input, source_output in _SOURCE_PAIRS
    )


def _grid_key(grid: np.ndarray) -> tuple:
    canonical = np.asarray(grid, dtype=np.uint8)
    return canonical.shape, canonical.tobytes()


def _choose_size(rng: np.random.Generator, regime: str, index: int, first_size: int | None) -> int:
    all_sizes = list(range(5, 30, 2))
    if regime == "source_train" and index == 0:
        choices = list(range(5, 14, 2))
    elif regime == "target_test" and rng.random() < 0.30:
        choices = list(range(13, 30, 2))
    else:
        choices = all_sizes
    if regime == "source_train" and index == 1:
        choices = [size for size in choices if size != first_size]
    return int(rng.choice(choices))


def _draw_support(rng: np.random.Generator, middle: int) -> int:
    roll = float(rng.random())
    if roll < 0.16:
        return 0
    if roll < 0.32:
        return middle
    return int(rng.integers(0, middle + 1))


def _choose_supports(
    rng: np.random.Generator,
    middle: int,
    active_corners: list[int],
    target_corner: int,
    mode: str,
) -> dict[int, int]:
    supports = {corner: _draw_support(rng, middle) for corner in active_corners}
    rivals = [corner for corner in active_corners if corner != target_corner]
    if mode == "weak":
        target = int(rng.integers(0, middle))
        rival = int(rng.choice(rivals))
        supports[target_corner] = target
        supports[rival] = int(rng.integers(target + 1, middle + 1))
    elif mode == "strong":
        target = int(rng.integers(1, middle + 1))
        supports[target_corner] = target
        for rival in rivals:
            supports[rival] = int(rng.integers(0, target))
    elif mode == "tie":
        target = _draw_support(rng, middle)
        supports[target_corner] = target
        supports[int(rng.choice(rivals))] = target
    elif mode == "zero":
        supports[target_corner] = 0
    elif mode == "full":
        supports[target_corner] = middle
    elif mode != "random":
        raise ValueError(f"unknown support mode: {mode}")
    return supports


def _off_diagonal_count(
    rng: np.random.Generator,
    capacity: int,
    style: int,
    regime: str,
) -> int:
    weights = ((0.58, 0.32, 0.10), (0.34, 0.44, 0.22), (0.18, 0.40, 0.42))[style]
    if regime == "target_test":
        weights = (max(0.08, weights[0] - 0.08), weights[1], weights[2] + min(0.08, weights[0] - 0.08))
    category = int(rng.choice(3, p=np.array(weights) / sum(weights)))
    bounds = ((0.0, 0.16), (0.16, 0.44), (0.44, 0.90))[category]
    fraction = float(rng.uniform(*bounds))
    return min(capacity, int(round(capacity * fraction)))


def _shape_cells(
    rng: np.random.Generator,
    candidates: list[tuple[int, int]],
    count: int,
    size: int,
    corner: int,
) -> list[tuple[int, int]]:
    if count == 0:
        return []
    if count >= len(candidates):
        return candidates[:]
    mode = int(rng.integers(0, 6))
    if mode == 0:
        indices = rng.choice(len(candidates), size=count, replace=False)
        return [candidates[int(index)] for index in np.atleast_1d(indices)]
    if mode == 2:
        available = set(candidates)
        runs = []
        for row, col in candidates:
            for row_step, col_step in ((0, 1), (1, 0)):
                run = [(row + step * row_step, col + step * col_step) for step in range(count)]
                if all(cell in available for cell in run):
                    runs.append(run)
        if runs:
            return runs[int(rng.integers(0, len(runs)))]
    middle = size // 2
    pivot = candidates[int(rng.integers(0, len(candidates)))]
    horizontal = bool(rng.integers(0, 2))
    row_step, col_step = _CORNERS[corner]
    scored = []
    for row, col in candidates:
        if mode == 1:  # compact patch
            score = abs(row - pivot[0]) + abs(col - pivot[1])
        elif mode == 2:  # narrow stripe
            score = 2.5 * (abs(row - pivot[0]) if horizontal else abs(col - pivot[1])) + 0.25 * (abs(col - pivot[1]) if horizontal else abs(row - pivot[0]))
        elif mode == 3:  # outer edge or corner
            score = (row if row_step < 0 else size - 1 - row) + (col if col_step < 0 else size - 1 - col)
        elif mode == 4:  # near the center
            score = abs(row - middle) + abs(col - middle)
        else:  # patch mixed with dispersed cells
            score = min(abs(row - pivot[0]) + abs(col - pivot[1]), float(rng.uniform(0, middle + 2)))
        scored.append((score + float(rng.uniform(0, 0.85)), row, col))
    scored.sort()
    return [(row, col) for _, row, col in scored[:count]]


def _make_instance(
    rng: np.random.Generator,
    size: int,
    background: int,
    target_corner: int,
    target_color: int,
    active_count: int,
    support_mode: str,
    style: int,
    regime: str,
) -> tuple[np.ndarray, np.ndarray]:
    middle = size // 2
    rival_corners = [corner for corner in range(4) if corner != target_corner]
    chosen = rng.choice(rival_corners, size=active_count - 1, replace=False)
    active_corners = [target_corner] + [int(corner) for corner in np.atleast_1d(chosen)]
    other_colors = [color for color in range(10) if color not in (background, target_color)]
    sampled_colors = rng.choice(other_colors, size=active_count - 1, replace=False)
    corner_colors = {target_corner: target_color}
    corner_colors.update(zip(active_corners[1:], (int(color) for color in sampled_colors)))
    supports = _choose_supports(rng, middle, active_corners, target_corner, support_mode)
    off_counts = {}
    for corner in active_corners:
        capacity = middle * middle - middle
        count = _off_diagonal_count(rng, capacity, style, regime)
        if supports[corner] == 0:
            count = max(1, count)
        off_counts[corner] = count

    # Keep the background visibly dominant even for dense, asymmetric clouds.
    maximum_colored = int(0.45 * size * size)
    excess = 1 + sum(supports.values()) + sum(off_counts.values()) - maximum_colored
    while excess > 0:
        reducible = [
            corner for corner in active_corners
            if off_counts[corner] > (1 if supports[corner] == 0 else 0)
        ]
        if not reducible:
            break
        corner = int(rng.choice(reducible))
        off_counts[corner] -= 1
        excess -= 1
    if excess > 0:
        raise RuntimeError("could not retain a dominant background")

    input_grid = np.full((size, size), background, dtype=np.int8)
    for corner in active_corners:
        diagonal = _diagonal_cells(size, corner, include_center=False)
        diagonal_indices = rng.choice(middle, size=supports[corner], replace=False)
        selected_diagonal = [diagonal[int(index)] for index in np.atleast_1d(diagonal_indices)] if supports[corner] else []
        diagonal_set = set(diagonal)
        off_diagonal = [cell for cell in _sector_cells(size, corner) if cell not in diagonal_set]
        selected_off = _shape_cells(rng, off_diagonal, off_counts[corner], size, corner)
        for row, col in selected_diagonal + selected_off:
            input_grid[row, col] = corner_colors[corner]
    input_grid[middle, middle] = target_color

    output_grid = np.full((size, size), background, dtype=np.int8)
    for row, col in _diagonal_cells(size, target_corner):
        output_grid[row, col] = target_color
    return input_grid, output_grid


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _check(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _check(grid.ndim == 2, f"{label} must be two-dimensional")
    height, width = grid.shape
    _check(1 <= height <= 30 and 1 <= width <= 30, f"{label} size is out of range")
    _check(np.issubdtype(grid.dtype, np.integer), f"{label} must contain integers")
    _check(bool(np.all((grid >= 0) & (grid <= 9))), f"{label} contains an invalid color")


def _infer_from_input(input_grid: np.ndarray) -> tuple[np.ndarray, dict]:
    height, width = input_grid.shape
    _check(height == width and height % 2 == 1 and height >= 5, "inputs must be odd squares of size at least 5")
    colors, counts = np.unique(input_grid, return_counts=True)
    winner = np.flatnonzero(counts == counts.max())
    _check(len(winner) == 1, "background must be uniquely dominant")
    background = int(colors[int(winner[0])])
    middle = height // 2
    target_color = int(input_grid[middle, middle])
    _check(target_color != background, "center must contain a signal color")
    signals = [int(color) for color in colors if int(color) != background]
    _check(2 <= len(signals) <= 4, "input must contain target and one to three distractors")
    sector_by_color = {}
    supports = {}
    populations = {}
    off_counts = {}
    for color in signals:
        positions = np.argwhere(input_grid == color)
        sectors = set()
        off_count = 0
        noncenter = 0
        for row_value, col_value in positions:
            row, col = int(row_value), int(col_value)
            if (row, col) == (middle, middle):
                continue
            _check(row != middle and col != middle, "signal lies on a center axis")
            sector = _CORNERS.index((-1 if row < middle else 1, -1 if col < middle else 1))
            sectors.add(sector)
            noncenter += 1
            if abs(row - middle) != abs(col - middle):
                off_count += 1
        _check(noncenter >= 1 and len(sectors) == 1, "every color needs one unambiguous quadrant")
        corner = next(iter(sectors))
        sector_by_color[color] = corner
        populations[color] = noncenter
        off_counts[color] = off_count
        supports[color] = sum(int(input_grid[row, col]) == color for row, col in _diagonal_cells(height, corner, False))
    _check(len(set(sector_by_color.values())) == len(signals), "clouds must occupy distinct quadrants")
    _check(target_color in sector_by_color, "center color needs directional cloud evidence")
    _check(int(np.count_nonzero(input_grid == background)) > (height * height) // 2, "background must occupy a majority")
    target_corner = sector_by_color[target_color]
    output = np.full(input_grid.shape, background, dtype=np.int8)
    for row, col in _diagonal_cells(height, target_corner):
        output[row, col] = target_color
    rival_best = max(supports[color] for color in signals if color != target_color)
    comparison = "strong" if supports[target_color] > rival_best else "tie" if supports[target_color] == rival_best else "weak"
    metadata = {
        "size": height, "background": background, "target_color": target_color,
        "target_corner": target_corner, "signals": len(signals),
        "supports": supports, "target_support": supports[target_color],
        "support_comparison": comparison, "noise": sum(off_counts.values()),
        "populations": populations, "off_counts": off_counts,
    }
    return output, metadata


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _check(input_grid.shape == output_grid.shape, "input and output dimensions differ")
    expected, metadata = _infer_from_input(input_grid)
    _check(np.array_equal(output_grid, expected), "output is not the center-color diagonal")
    _check(not np.array_equal(input_grid, output_grid), "input and output are identical")
    return metadata


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = sample_puzzle_parameters(rng)
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    train_info = []
    for split, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[len(seen_inputs)]))
            for attempt in range(48):
                if split == "train" and index == 0:
                    background = parameters["anchor_background"]
                elif split == "train" and index == 1:
                    background = int(pair_rng.choice([color for color in range(10) if color != train_info[0]["background"]]))
                else:
                    background = int(pair_rng.integers(0, 10))
                size = _choose_size(pair_rng, regime, index, train_info[0]["size"] if train_info else None)
                if split == "train" and index == 1:
                    target_corner = int(pair_rng.choice([corner for corner in parameters["corner_pool"] if corner != train_info[0]["target_corner"]]))
                else:
                    target_corner = int(pair_rng.choice(parameters["corner_pool"]))
                if split == "train" and index == 1:
                    candidates = [color for color in range(10) if color not in (background, train_info[0]["target_color"])]
                    target_color = int(pair_rng.choice(candidates))
                elif split == "test" and pair_rng.random() < 0.48:
                    familiar = [item["target_color"] for item in train_info if item["target_color"] != background]
                    target_color = int(pair_rng.choice(familiar))
                else:
                    target_color = int(pair_rng.choice([color for color in range(10) if color != background]))
                active_count = 4 if split == "train" and index == 0 else int(pair_rng.choice((2, 3, 4), p=(0.25, 0.30, 0.45)))
                support_mode = "weak" if split == "train" and index == 0 else str(pair_rng.choice(("random", "weak", "strong", "tie", "zero", "full"), p=(0.20, 0.18, 0.18, 0.14, 0.15, 0.15)))
                input_grid, output_grid = _make_instance(
                    pair_rng, size, background, target_corner, target_color,
                    active_count, support_mode, parameters["style"], regime,
                )
                key = _grid_key(input_grid)
                if key not in seen_inputs and not _is_source_pair(input_grid, output_grid):
                    break
            else:
                # Exhaust a small deterministic size ladder if random redraws collide.
                fallback_sizes = list(range(5, 14, 2)) if split == "train" and index == 0 else list(range(5, 30, 2))
                if split == "train" and index == 1:
                    fallback_sizes = [value for value in fallback_sizes if value != train_info[0]["size"]]
                background = parameters["anchor_background"] if split == "train" and index == 0 else next(
                    color for color in range(10)
                    if split != "train" or index != 1 or color != train_info[0]["background"]
                )
                target_corner = next(
                    corner for corner in parameters["corner_pool"]
                    if split != "train" or index != 1 or corner != train_info[0]["target_corner"]
                )
                target_color = next(
                    color for color in range(10)
                    if color != background and (split != "train" or index != 1 or color != train_info[0]["target_color"])
                )
                for size in fallback_sizes:
                    input_grid, output_grid = _make_instance(
                        pair_rng, size, background, target_corner, target_color,
                        4 if split == "train" and index == 0 else 2,
                        "weak", parameters["style"], regime,
                    )
                    key = _grid_key(input_grid)
                    if key not in seen_inputs and not _is_source_pair(input_grid, output_grid):
                        break
                else:
                    raise RuntimeError("could not construct a distinct pair")
            seen_inputs.add(key)
            puzzle[split].append({"input": input_grid, "output": output_grid})
            if split == "train":
                train_info.append({"background": background, "size": size, "target_corner": target_corner, "target_color": target_color})
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_generated_puzzle_impl(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle needs train and test")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    metadata = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair structure")
            input_grid, output_grid = pair["input"], pair["output"]
            _validate_grid(input_grid, f"{split}[{index}].input")
            _validate_grid(output_grid, f"{split}[{index}].output")
            input_key = _grid_key(input_grid)
            pair_key = (input_key, _grid_key(output_grid))
            _check(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair or input")
            _check(not _is_source_pair(input_grid, output_grid), "source pair copied exactly")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            metadata[split].append(_analyze_pair(input_grid, output_grid))
    train = metadata["train"]
    _check(len({item["target_corner"] for item in train}) >= 2, "training needs multiple directions")
    _check(len({item["target_color"] for item in train}) >= 2, "training needs multiple target colors")
    _check(len({item["background"] for item in train}) >= 2, "training needs background variation")
    _check(len({item["size"] for item in train}) >= 2, "training needs size variation")
    _check(train[0]["signals"] == 4, "first training pair should show four quadrants")
    _check(train[0]["support_comparison"] == "weak", "training should refute maximum-support selection")
    for pair in puzzle["test"]:
        predicted = _solve_from_demonstrations(puzzle["train"], pair["input"])
        _check(np.array_equal(predicted, pair["output"]), "test cannot be solved from demonstrations")


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(f"malformed generated puzzle: {exc}") from exc


def _solve_one_input(grid: np.ndarray) -> np.ndarray:
    # Separate reconstruction used by the whole-puzzle prediction check.
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    middle = grid.shape[0] // 2
    color = int(grid[middle, middle])
    positions = np.argwhere(grid == color)
    directions = {
        (-1 if int(row) < middle else 1, -1 if int(col) < middle else 1)
        for row, col in positions
        if (int(row), int(col)) != (middle, middle)
    }
    _check(len(directions) == 1, "center color has ambiguous direction")
    row_step, col_step = next(iter(directions))
    result = np.full(grid.shape, background, dtype=np.int8)
    for distance in range(middle + 1):
        result[middle + distance * row_step, middle + distance * col_step] = color
    return result


def _solve_from_demonstrations(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    for pair in train_pairs:
        _check(np.array_equal(_solve_one_input(pair["input"]), pair["output"]), "training does not teach the center-color ray")
    return _solve_one_input(test_input)


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[split], second[split])
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    for index, (source_input, source_output) in enumerate(_SOURCE_PAIRS):
        input_grid = np.array([[int(char) for char in row] for row in source_input], dtype=np.int8)
        output_grid = np.array([[int(char) for char in row] for row in source_output], dtype=np.int8)
        center = input_grid.shape[0] // 2
        _check(int(input_grid[center, center]) == (0, 5, 4)[index], "source center role changed")
        _check(int(np.bincount(input_grid.ravel()).argmax()) == 7, "source background role changed")
        if index < 2:
            _check(np.array_equal(_solve_one_input(input_grid), output_grid), "source training rule mismatch")
        if index == 0:
            _, source_features = _infer_from_input(input_grid)
            _check(source_features["target_support"] == 4, "source target support changed")
            _check(max(value for color, value in source_features["supports"].items() if color != 0) == 3, "source rival support changed")
    a = generate_puzzle(12345, 6, 3)
    b = generate_puzzle(12345, 6, 3)
    _check(_puzzles_equal(a, b), "same seed must be deterministic")
    _check(not _puzzles_equal(a, generate_puzzle(12346, 6, 3)), "different seeds did not differ")
    rotated = {
        "train": a["train"],
        "test": [dict(pair) for pair in a["test"]],
    }
    rotated["test"][0] = {
        "input": np.rot90(a["test"][0]["input"]).copy(),
        "output": np.rot90(a["test"][0]["output"]).copy(),
    }
    validate_generated_puzzle(rotated)
    for index, (trains, tests) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + index, trains, tests)
        _check((len(puzzle["train"]), len(puzzle["test"])) == (trains, tests), "explicit counts changed")
    for index, arguments in enumerate((
        {"num_train": None, "num_test": 1}, {"num_train": None, "num_test": 4},
        {"num_train": 2, "num_test": None}, {"num_train": 10, "num_test": None},
    )):
        puzzle = generate_puzzle(3000 + index, **arguments)
        if arguments["num_train"] is not None:
            _check(len(puzzle["train"]) == arguments["num_train"], "partial train count changed")
        if arguments["num_test"] is not None:
            _check(len(puzzle["test"]) == arguments["num_test"], "partial test count changed")
    for index, arguments in enumerate((
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": True, "num_test": 1}, {"num_train": 4, "num_test": 1.5},
    )):
        try:
            generate_puzzle(4000 + index, **arguments)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {arguments}")
    train_counts, test_counts = set(), set()
    sizes, backgrounds, corners, comparisons, signal_counts = set(), set(), set(), set(), set()
    target_support_ratios = set()
    background_varied, mixed_test_corners = 0, 0
    two_corner_puzzles = 0
    tests_not_train_two = unseen_test_corner = familiar_test_color = 0
    test_smaller = test_comparable = test_larger = 0
    test_less_noise = test_more_noise = uneven_cloud_puzzles = 0
    fingerprints = set()
    noise_profiles = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signature = []
        train_meta, test_meta = [], []
        for split, output_list in (("train", train_meta), ("test", test_meta)):
            for pair in puzzle[split]:
                expected, item = _infer_from_input(pair["input"])
                _check(np.array_equal(expected, pair["output"]), "self-test pair mismatch")
                output_list.append(item)
                signature.append(_grid_key(pair["input"]))
                sizes.add(item["size"])
                backgrounds.add(item["background"])
                corners.add(item["target_corner"])
                comparisons.add(item["support_comparison"])
                signal_counts.add(item["signals"])
                target_support_ratios.add((item["target_support"] == 0, item["target_support"] == item["size"] // 2))
        fingerprints.add(tuple(signature))
        noise_profiles.add(tuple(item["noise"] for item in train_meta + test_meta))
        background_varied += len({item["background"] for item in train_meta + test_meta}) > 1
        mixed_test_corners += len({item["target_corner"] for item in test_meta}) > 1
        two_corner_puzzles += len({item["target_corner"] for item in train_meta + test_meta}) == 2
        tests_not_train_two += any(item["target_corner"] != train_meta[1]["target_corner"] for item in test_meta)
        unseen_test_corner += any(item["target_corner"] not in {entry["target_corner"] for entry in train_meta} for item in test_meta)
        familiar_test_color += any(item["target_color"] in {entry["target_color"] for entry in train_meta} for item in test_meta)
        largest_train = max(item["size"] for item in train_meta)
        train_sizes = {item["size"] for item in train_meta}
        test_smaller += any(item["size"] < largest_train for item in test_meta)
        test_comparable += any(item["size"] in train_sizes for item in test_meta)
        test_larger += any(item["size"] > largest_train for item in test_meta)
        largest_train_noise = max(item["noise"] for item in train_meta)
        test_less_noise += any(item["noise"] <= largest_train_noise for item in test_meta)
        test_more_noise += any(item["noise"] > largest_train_noise for item in test_meta)
        uneven_cloud_puzzles += any(
            max(item["populations"].values()) >= 3 * min(item["populations"].values())
            for item in train_meta + test_meta
        )
        for pair in puzzle["test"]:
            _check(np.array_equal(_solve_from_demonstrations(puzzle["train"], pair["input"]), pair["output"]), "prediction failed")
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "automatic pair counts lack diversity")
    _check(len(fingerprints) >= 40 and len(noise_profiles) >= 20, "layouts lack diversity")
    _check(len(sizes) >= 10 and {5, 21, 29}.issubset(sizes), "odd sizes lack coverage")
    _check(len(backgrounds) == 10 and corners == {0, 1, 2, 3}, "palette or direction lacks coverage")
    _check(comparisons == {"weak", "tie", "strong"}, "diagonal support classes lack coverage")
    _check({2, 3, 4}.issubset(signal_counts), "cloud counts lack coverage")
    _check((True, False) in target_support_ratios and (False, True) in target_support_ratios, "support boundaries missing")
    _check(background_varied >= 35 and mixed_test_corners >= 4, "within-puzzle variation is too narrow")
    _check(two_corner_puzzles >= 3 and tests_not_train_two >= 20 and unseen_test_corner >= 2, "direction combinations lack diversity")
    _check(familiar_test_color >= 15, "familiar target colors are missing in tests")
    _check(min(test_smaller, test_comparable, test_larger) >= 4, "size relations lack diversity")
    _check(min(test_less_noise, test_more_noise) >= 4, "clutter relations lack diversity")
    _check(uneven_cloud_puzzles >= 20, "cloud populations are too uniform")
