import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "07777777",
            "07777777",
            "07777777",
            "08877777",
            "07777777",
            "08777777",
            "07777777",
            "08777777",
            "07777777",
            "07777777",
            "08777777",
            "07777777",
            "08877777",
            "07777777",
            "07777777",
            "08777777",
        ),
        (
            "07777777",
            "07777777",
            "07777777",
            "07777700",
            "07777777",
            "08877777",
            "07777777",
            "08877777",
            "07777777",
            "07777777",
            "08877777",
            "07777777",
            "07777700",
            "07777777",
            "07777777",
            "08877777",
        ),
    ),
    (
        (
            "0000000000000000",
            "7787787777787877",
            "7777787777777777",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
        ),
        (
            "0000000000000000",
            "7787777777787877",
            "7787777777787877",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "7777707777777777",
            "7777707777777777",
        ),
    ),
    (
        (
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "8777878777777778",
            "8787878787778778",
            "0000000000000000",
        ),
        (
            "0777070777777770",
            "0777070777777770",
            "7777777777777777",
            "7777777777777777",
            "7777777777777777",
            "7787777787778777",
            "7787777787778777",
            "0000000000000000",
        ),
    ),
)


def _rows_to_array(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = tuple(
    (_rows_to_array(input_rows), _rows_to_array(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _validate_count_value(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return value


def _resolve_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_count_value(num_train, "num_train", 2, 10)
    num_test = _validate_count_value(num_test, "num_test", 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        # This schedule guarantees broad count diversity for consecutive seeds.
        num_test = 1 + (seed % 4)
        span = 10 - num_test
        num_train = num_test + 1 + ((seed // 4) % span)
        return num_train, num_test

    if num_train is None:
        assert num_test is not None
        num_train = int(rng.integers(num_test + 1, 11))
        return num_train, num_test

    assert num_test is None
    largest_legal_test_count = min(4, num_train - 1)
    num_test = int(rng.integers(1, largest_legal_test_count + 1))
    return num_train, num_test


def _coordinate(
    side: str,
    depth_index: int,
    lane: int,
    height: int,
    width: int,
) -> tuple[int, int]:
    if side == "top":
        return depth_index, lane
    if side == "bottom":
        return height - 1 - depth_index, lane
    if side == "left":
        return lane, depth_index
    if side == "right":
        return lane, width - 1 - depth_index
    raise AssertionError(f"Unknown edge side: {side}")


def _analyze_input(grid: np.ndarray) -> dict:
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 3, "Each input must contain exactly three colors")

    largest_count = int(np.max(counts))
    _require(
        int(np.count_nonzero(counts == largest_count)) == 1,
        "The background color must be uniquely identifiable by area",
    )
    background = int(colors[int(np.argmax(counts))])

    edges = (
        ("top", grid[0, :]),
        ("bottom", grid[-1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, -1]),
    )

    control_candidates: list[tuple[str, int]] = []
    for side, edge in edges:
        edge_color = int(edge[0])
        if edge_color != background and bool(np.all(edge == edge_color)):
            control_candidates.append((side, edge_color))

    _require(
        len(control_candidates) == 1,
        "There must be exactly one non-background monochrome control edge",
    )
    side, control = control_candidates[0]

    accent_candidates = {
        int(color) for color in colors
    } - {background, control}
    _require(
        len(accent_candidates) == 1,
        "The accent color must be uniquely identifiable",
    )
    accent = accent_candidates.pop()

    height, width = grid.shape
    if side in ("top", "bottom"):
        depth = height
        edge_length = width
    else:
        depth = width
        edge_length = height

    _require(
        depth >= 5,
        "Near-edge and opposite-edge two-cell regions must be disjoint",
    )

    runs: list[tuple[int, int]] = []

    for lane in range(edge_length):
        first_row, first_col = _coordinate(
            side, 1, lane, height, width
        )
        first_value = int(grid[first_row, first_col])

        if first_value == background:
            for depth_index in range(1, depth):
                row, col = _coordinate(
                    side, depth_index, lane, height, width
                )
                _require(
                    int(grid[row, col]) == background,
                    "An empty lane may contain only background",
                )
            continue

        _require(
            first_value == accent,
            "An occupied lane must start with the accent color",
        )

        run_length = 0
        depth_index = 1
        while depth_index < depth:
            row, col = _coordinate(
                side, depth_index, lane, height, width
            )
            if int(grid[row, col]) != accent:
                break
            run_length += 1
            depth_index += 1

        _require(
            run_length in (1, 2),
            "Each accent bar must have length one or two",
        )

        for remaining_depth in range(1 + run_length, depth):
            row, col = _coordinate(
                side, remaining_depth, lane, height, width
            )
            _require(
                int(grid[row, col]) == background,
                "Accent bars must be the only non-background lane contents",
            )

        runs.append((lane, run_length))

    _require(runs, "Every pair must contain occupied lanes")

    return {
        "background": background,
        "control": control,
        "accent": accent,
        "side": side,
        "depth": depth,
        "edge_length": edge_length,
        "runs": tuple(runs),
    }


def _apply_rule(grid: np.ndarray, analysis: dict) -> np.ndarray:
    output = grid.copy()

    background = analysis["background"]
    control = analysis["control"]
    accent = analysis["accent"]
    side = analysis["side"]
    depth = analysis["depth"]
    height, width = grid.shape

    for lane, run_length in analysis["runs"]:
        if run_length == 1:
            row, col = _coordinate(side, 2, lane, height, width)
            output[row, col] = accent
        else:
            for depth_index in (1, 2):
                row, col = _coordinate(
                    side, depth_index, lane, height, width
                )
                output[row, col] = background

            for depth_index in (depth - 2, depth - 1):
                row, col = _coordinate(
                    side, depth_index, lane, height, width
                )
                output[row, col] = control

    return output


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = rng.choice(10, size=3, replace=False)
    return {
        "colors": tuple(int(color) for color in colors),
        "stable_palette": bool(rng.integers(0, 2)),
    }


def _sample_dimensions(rng: np.random.Generator) -> tuple[int, int]:
    shape = int(rng.integers(0, 5))
    if shape == 0:
        size = int(rng.integers(6, 19))
        return size, size
    short = int(rng.integers(6, 16))
    long = int(rng.integers(short + 1, min(27, short + 14)))
    return (long, short) if shape in (1, 2) else (short, long)


def _choose_sides(
    rng: np.random.Generator, train_count: int, test_count: int
) -> tuple[list[str], list[str]]:
    sides = ("top", "bottom", "left", "right")
    mode = int(rng.integers(0, 10))
    if mode == 0:  # An occasional source-like held-out direction.
        train = ["left", "top"] + [("left", "top")[int(rng.integers(0, 2))]
                                   for _ in range(train_count - 2)]
        return train, ["bottom"] * test_count
    if mode <= 2:  # One direction throughout a puzzle.
        side = sides[int(rng.integers(0, 4))]
        return [side] * train_count, [side] * test_count
    if mode <= 5:  # A direction absent from training.
        held_out = sides[int(rng.integers(0, 4))]
        available = [side for side in sides if side != held_out]
        train = [available[int(rng.integers(0, 3))]
                 for _ in range(train_count)]
        return train, [held_out] * test_count
    train = [sides[int(rng.integers(0, 4))]
             for _ in range(train_count)]
    test = [sides[int(rng.integers(0, 4))]
            for _ in range(test_count)]
    if len(set(train + test)) == 1:
        test[0] = sides[(sides.index(test[0]) + 1) % 4]
    return train, test


def _sample_lanes(
    rng: np.random.Generator, edge_length: int, minimum: int
) -> tuple[list[int], str]:
    style = ("spaced", "alternating", "cluster_equal",
             "cluster_mixed", "varied_gaps", "random")[
                 int(rng.integers(0, 6))]
    n = edge_length
    if style == "spaced":
        parity = int(rng.integers(0, 2))
        candidates = list(range(parity, n, 2))
        size = int(rng.integers(minimum, len(candidates) + 1))
        lanes = sorted(int(x) for x in rng.choice(candidates, size, False))
    elif style == "alternating":
        parity = int(rng.integers(0, 2))
        lanes = list(range(parity, n, 2))
        if len(lanes) > minimum and bool(rng.integers(0, 2)):
            drop = int(rng.integers(1, min(3, len(lanes) - minimum) + 1))
            removed = set(int(x) for x in rng.choice(lanes, drop, False))
            lanes = [x for x in lanes if x not in removed]
    elif style.startswith("cluster"):
        length = int(rng.integers(max(2, minimum), min(n, 8) + 1))
        start = int(rng.integers(0, n - length + 1))
        lanes = list(range(start, start + length))
        available = [x for x in range(n) if x not in lanes]
        extra = int(rng.integers(0, min(len(available), 4) + 1))
        if extra:
            lanes += [int(x) for x in rng.choice(available, extra, False)]
        lanes.sort()
    else:
        density = int(rng.integers(0, 4))
        if density == 0:
            lower, upper = minimum, min(n, max(minimum, 3))
        elif density == 1:
            lower, upper = max(minimum, 2), max(minimum, n // 2)
        elif density == 2:
            lower, upper = max(minimum, n // 3), max(minimum, (3 * n) // 4)
        else:
            lower, upper = max(minimum, (2 * n) // 3), n
        size = int(rng.integers(lower, upper + 1))
        if style == "varied_gaps" and n >= 8 and size >= 3:
            for _ in range(30):
                trial = sorted(int(x) for x in rng.choice(n, size, False))
                gaps = [b - a - 1 for a, b in zip(trial, trial[1:])]
                if len(set(gaps)) >= 2:
                    lanes = trial
                    break
            else:
                lanes = trial
        else:
            lanes = sorted(int(x) for x in rng.choice(n, size, False))
    return lanes, style


def _sample_lengths(
    rng: np.random.Generator, lanes: list[int], style: str,
    require_both: bool
) -> dict[int, int]:
    count = len(lanes)
    if require_both:
        two_count = int(rng.integers(1, count))
    else:
        mixture = int(rng.integers(0, 6))
        if mixture == 0:
            two_count = 0
        elif mixture == 1:
            two_count = count
        elif mixture == 2:
            two_count = int(rng.integers(0, max(1, count // 3) + 1))
        elif mixture == 3:
            two_count = int(rng.integers(count // 3, (2 * count) // 3 + 1))
        elif mixture == 4:
            two_count = int(rng.integers((2 * count) // 3, count + 1))
        else:
            two_count = int(rng.integers(0, count + 1))
    chosen = set(int(x) for x in rng.choice(lanes, two_count, False))
    lengths = {lane: (2 if lane in chosen else 1) for lane in lanes}
    if style == "cluster_equal" and not require_both:
        # A run of neighbors of the same length, without changing the count.
        blocks = []
        block = [lanes[0]]
        for lane in lanes[1:]:
            if lane == block[-1] + 1:
                block.append(lane)
            else:
                blocks.append(block)
                block = [lane]
        blocks.append(block)
        largest = max(blocks, key=len)
        if len(largest) >= 2:
            value = 2 if bool(rng.integers(0, 2)) else 1
            for lane in largest:
                lengths[lane] = value
    if style == "cluster_mixed" and len(lanes) >= 2:
        for a, b in zip(lanes, lanes[1:]):
            if b == a + 1:
                lengths[a], lengths[b] = 1, 2
                break
    if require_both and set(lengths.values()) != {1, 2}:
        lengths[lanes[-1]] = 2 if lengths[lanes[0]] == 1 else 1
    return lengths


def _sample_colors(
    rng: np.random.Generator, parameters: dict, index: int
) -> tuple[int, int, int]:
    base = parameters["colors"]
    if parameters["stable_palette"] or index == 0:
        return base
    colors = tuple(int(x) for x in rng.choice(10, 3, False))
    if index == 1 and colors == base:
        colors = (base[1], base[2], base[0])
    return colors


def _generate_pair(
    rng: np.random.Generator, regime: str, side: str,
    shape: tuple[int, int], colors: tuple[int, int, int],
    require_both: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("Unknown regime")
    height, width = shape
    edge_length = width if side in ("top", "bottom") else height
    lanes, style = _sample_lanes(rng, edge_length, 2 if require_both else 1)
    lengths = _sample_lengths(rng, lanes, style, require_both)
    background, control, accent = colors
    grid = np.full((height, width), background, dtype=np.int8)
    if side == "top":
        grid[0, :] = control
    elif side == "bottom":
        grid[-1, :] = control
    elif side == "left":
        grid[:, 0] = control
    else:
        grid[:, -1] = control
    for lane, length in lengths.items():
        for depth_index in range(1, length + 1):
            row, col = _coordinate(side, depth_index, lane, height, width)
            grid[row, col] = accent
    return grid, _apply_rule(grid, _analyze_input(grid))


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int16)
    return normalized.shape, normalized.tobytes()


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _grid_key(input_grid), _grid_key(output_grid)


def _is_exact_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return any(np.array_equal(input_grid, original_input)
               and np.array_equal(output_grid, original_output)
               for original_input, original_output in _SOURCE_PAIRS)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    train_sides, test_sides = _choose_sides(rng, train_count, test_count)
    sides = train_sides + test_sides
    total = len(sides)
    shapes = [_sample_dimensions(rng) for _ in range(total)]
    # Same-size side changes make direction visibly independent of size.
    different = [i for i in range(1, total) if sides[i] != sides[0]]
    if different and rng.random() < 0.45:
        shapes[different[int(rng.integers(0, len(different)))]] = shapes[0]
    evidence_index = int(rng.integers(0, train_count))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=total,
                              dtype=np.int64)
    puzzle = {"train": [], "test": []}
    used_inputs: set[tuple] = set()
    for index in range(total):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        colors = _sample_colors(pair_rng, parameters, index)
        split = "train" if index < train_count else "test"
        regime = "source_train" if split == "train" else "target_test"
        for attempt in range(80):
            grid, output = _generate_pair(
                pair_rng, regime, sides[index], shapes[index], colors,
                require_both=index == evidence_index,
            )
            if _grid_key(grid) not in used_inputs and not _is_exact_source_pair(grid, output):
                break
        else:
            raise RuntimeError("Could not construct a distinct pair")
        used_inputs.add(_grid_key(grid))
        puzzle[split].append({"input": grid, "output": output})
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")
    height, width = grid.shape
    _require(1 <= height <= 30 and 1 <= width <= 30,
             f"{label} dimensions must be within 1..30")
    _require(np.issubdtype(grid.dtype, np.integer),
             f"{label} must have an integer dtype")
    _require(int(np.min(grid)) >= 0 and int(np.max(grid)) <= 9,
             f"{label} colors must be within 0..9")


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "A puzzle must have exactly train and test lists")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "Train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "Invalid pair counts")
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    training_states: set[int] = set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{label} must have exactly input and output")
            grid, output = pair["input"], pair["output"]
            _validate_grid(grid, label + ".input")
            _validate_grid(output, label + ".output")
            _require(grid.shape == output.shape, f"{label} changed dimensions")
            _require(not np.array_equal(grid, output), f"{label} did not change")
            input_key, pair_key = _grid_key(grid), _pair_key(grid, output)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                     f"{label} duplicated another pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analysis = _analyze_input(grid)
            _require(np.array_equal(output, _apply_rule(grid, analysis)),
                     f"{label} does not obey the lane rule")
            _require(set(int(x) for x in np.unique(output)).issubset(
                {analysis["background"], analysis["control"], analysis["accent"]}),
                f"{label} used an unrelated color")
            _require(not _is_exact_source_pair(grid, output),
                     f"{label} copied a source pair")
            if split == "train":
                training_states.update(length for _, length in analysis["runs"])
    _require(training_states == {1, 2},
             "Training must demonstrate growth and transfer")
    for index, pair in enumerate(puzzle["test"]):
        _require(np.array_equal(
            _predict_from_training(puzzle["train"], pair["input"]),
            pair["output"],
        ), f"test[{index}] cannot be recovered from training")


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a["input"], b["input"])
        and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[split], second[split]))
        for split in ("train", "test"))


def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    # Derive the two local output templates from demonstrated pairs only.
    templates: dict[int, tuple[int, int, int, int]] = {}
    for pair in train:
        grid, output = pair["input"], pair["output"]
        info = _analyze_input(grid)
        height, width = grid.shape
        roles = (info["background"], info["control"], info["accent"])
        for lane, length in info["runs"]:
            positions = (1, 2, info["depth"] - 2, info["depth"] - 1)
            pattern = tuple(roles.index(int(output[_coordinate(
                info["side"], depth, lane, height, width)]))
                for depth in positions)
            if length in templates:
                _require(templates[length] == pattern,
                         "Training pairs imply contradictory bar outcomes")
            templates[length] = pattern
    _require(set(templates) == {1, 2}, "Training lacks a bar outcome")
    info = _analyze_input(test_input)
    roles = (info["background"], info["control"], info["accent"])
    predicted = test_input.copy()
    height, width = predicted.shape
    for lane, length in info["runs"]:
        for depth, role_index in zip(
            (1, 2, info["depth"] - 2, info["depth"] - 1), templates[length]
        ):
            predicted[_coordinate(info["side"], depth, lane,
                                  height, width)] = roles[role_index]
    return predicted


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(puzzle["train"]) for puzzle in puzzles}
    test_counts = {len(puzzle["test"]) for puzzle in puzzles}
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    signatures: set[tuple] = set()
    shapes: set[tuple] = set()
    sides: set[str] = set()
    endpoint_counts: set[int] = set()
    state_sets: set[tuple[int, ...]] = set()
    ratios: set[tuple[int, int]] = set()
    side_shapes: set[tuple[str, str]] = set()
    endpoint_states: set[int] = set()
    same_side = held_out = same_size_side_change = palette_change = 0
    stable_palette = one_lane = all_two_test = 0
    spaced = alternating = varied_gaps = adjacent_equal = adjacent_mixed = 0
    ratio_test_above = ratio_test_below = 0
    sparse_tests = dense_tests = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_info = [_analyze_input(p["input"]) for p in puzzle["train"]]
        test_info = [_analyze_input(p["input"]) for p in puzzle["test"]]
        train_sides = {info["side"] for info in train_info}
        if any(info["side"] in train_sides for info in test_info):
            same_side += 1
        if any(info["side"] not in train_sides for info in test_info):
            held_out += 1
        all_pairs = puzzle["train"] + puzzle["test"]
        all_info = train_info + test_info
        roles = set()
        for pair, info in zip(all_pairs, all_info):
            predicted = _predict_from_training(puzzle["train"], pair["input"])
            assert np.array_equal(predicted, pair["output"])
            roles.add((info["background"], info["control"], info["accent"]))
            h, w = pair["input"].shape
            shapes.add((h, w))
            sides.add(info["side"])
            side_shapes.add((info["side"],
                             "square" if h == w else
                             "portrait" if h > w else "landscape"))
            runs = info["runs"]
            signatures.add((info["side"], h, w, runs))
            one_lane += len(runs) == 1
            gaps = [b - a - 1 for (a, _), (b, _) in zip(runs, runs[1:])]
            spaced += len(runs) >= 2 and all(gap > 0 for gap in gaps)
            alternating += len(runs) >= 3 and all(gap == 1 for gap in gaps)
            varied_gaps += len(set(gaps)) >= 2
            adjacent_equal += any(b == a + 1 and x == y
                                  for (a, x), (b, y) in zip(runs, runs[1:]))
            adjacent_mixed += any(b == a + 1 and x != y
                                  for (a, x), (b, y) in zip(runs, runs[1:]))
            endpoint_states.update(length for lane, length in runs
                                   if lane in (0, info["edge_length"] - 1))
            endpoint_counts.add(sum(lane in (0, info["edge_length"] - 1)
                                    for lane, _ in runs))
            state_sets.add(tuple(sorted({length for _, length in runs})))
            ratios.add((sum(length == 2 for _, length in runs), len(runs)))
        palette_change += len(roles) > 1
        stable_palette += len(roles) == 1
        same_size_side_change += any(
            a["input"].shape == b["input"].shape
            and ia["side"] != ib["side"]
            for i, (a, ia) in enumerate(zip(all_pairs, all_info))
            for b, ib in list(zip(all_pairs, all_info))[i + 1:])
        max_train = max(len(info["runs"]) for info in train_info)
        sparse_tests += any(len(info["runs"]) <= max_train for info in test_info)
        dense_tests += any(len(info["runs"]) > max_train for info in test_info)
        all_two_test += any(
            {length for _, length in info["runs"]} == {2}
            and info["accent"] not in pair["output"]
            for pair, info in zip(puzzle["test"], test_info)
        )
        maximum_train_ratio = max(
            sum(length == 2 for _, length in info["runs"]) / len(info["runs"])
            for info in train_info
        )
        ratio_test_above += any(
            sum(length == 2 for _, length in info["runs"]) / len(info["runs"])
            > maximum_train_ratio for info in test_info
        )
        ratio_test_below += any(
            sum(length == 2 for _, length in info["runs"]) / len(info["runs"])
            < maximum_train_ratio for info in test_info
        )
    assert sides == {"top", "bottom", "left", "right"}
    assert len(side_shapes) == 12
    assert endpoint_counts == {0, 1, 2}
    assert endpoint_states == {1, 2}
    assert {(1,), (2,), (1, 2)}.issubset(state_sets)
    assert min(count for _, count in ratios) == 1 and one_lane >= 5
    assert len(signatures) >= 170 and len(shapes) >= 40
    assert same_side >= 6 and held_out >= 6
    assert same_size_side_change >= 3 and palette_change >= 8
    assert stable_palette >= 8 and all_two_test >= 3
    assert sparse_tests >= 6 and dense_tests >= 6
    assert ratio_test_above >= 3 and ratio_test_below >= 6
    assert min(spaced, alternating, varied_gaps,
               adjacent_equal, adjacent_mixed) >= 10
    for index, (train_count, test_count) in enumerate(
        ((2, 1), (10, 1), (10, 4), (6, 2),
         (None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    assert _puzzles_equal(generate_puzzle(8675309, 7, 3),
                          generate_puzzle(8675309, 7, 3))
    assert not _puzzles_equal(generate_puzzle(30001, 7, 3),
                              generate_puzzle(30002, 7, 3))
    for index, arguments in enumerate((
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": True}, {"num_test": False},
    )):
        try:
            generate_puzzle(4000 + index, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {arguments}")
