import numpy as np


_HEIGHT = 7
_BLOCK_SIZE = 2
_MIN_WIDTH = 5
_MAX_WIDTH = 17
_MIN_BLOCKS = 2
_MAX_BLOCKS = 9


_SOURCE_ROW_PAIRS = (
    (
        (
            "0000000",
            "0000022",
            "2200022",
            "2200000",
            "0022000",
            "0022000",
            "0000000",
        ),
        (
            "0000000",
            "0000088",
            "8800088",
            "8800000",
            "0022000",
            "0022000",
            "0000000",
        ),
    ),
    (
        (
            "00000000000022",
            "02202200000022",
            "02202200022000",
            "00000000022000",
            "00022002200000",
            "00022002200000",
            "00000000000000",
        ),
        (
            "00000000000088",
            "02202200000088",
            "02202200022000",
            "00000000022000",
            "00088008800000",
            "00088008800000",
            "00000000000000",
        ),
    ),
    (
        (
            "000000220000000",
            "000000220000022",
            "002200000000022",
            "002200002200000",
            "000002202200000",
            "220002200002200",
            "220000000002200",
        ),
        (
            "000000220000000",
            "000000220000088",
            "002200000000088",
            "002200008800000",
            "000008808800000",
            "880008800002200",
            "880000000002200",
        ),
    ),
    (
        (
            "00000",
            "22000",
            "22000",
            "00000",
            "00022",
            "00022",
            "00000",
        ),
        (
            "00000",
            "22000",
            "22000",
            "00000",
            "00088",
            "00088",
            "00000",
        ),
    ),
    (
        (
            "22000000000220000",
            "22002200000220000",
            "00002202200000000",
            "02200002200002200",
            "02200000002202200",
            "00000022002200022",
            "00000022000000022",
        ),
        (
            "88000000000880000",
            "88008800000880000",
            "00008808800000000",
            "02200008800002200",
            "02200000002202200",
            "00000022002200088",
            "00000022000000088",
        ),
    ),
    (
        (
            "000000000000000",
            "000000022000022",
            "002200022000022",
            "002200000000000",
            "000022000002200",
            "220022002202200",
            "220000002200000",
        ),
        (
            "000000000000000",
            "000000022000088",
            "002200022000088",
            "002200000000000",
            "000088000002200",
            "880088008802200",
            "880000008800000",
        ),
    ),
)


def _rows_key(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_PAIR_KEYS = frozenset(
    (_rows_key(input_rows), _rows_key(output_rows))
    for input_rows, output_rows in _SOURCE_ROW_PAIRS
)


def _array_key(array: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in array.shape),
        tuple(int(value) for value in array.reshape(-1)),
    )


def _pair_key(pair: dict) -> tuple:
    return _array_key(pair["input"]), _array_key(pair["output"])


def _validate_count_argument(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> None:
    if value is None:
        return

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not minimum <= int(value) <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )


def _resolve_pair_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count_argument("num_train", num_train, 2, 10)
    _validate_count_argument("num_test", num_test, 1, 4)

    train_count = None if num_train is None else int(num_train)
    test_count = None if num_test is None else int(num_test)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        span = 10 - minimum_train + 1
        train_count = minimum_train + (seed // 4) % span
        return train_count, test_count

    ticket = int(rng.integers(0, 2**31))

    if train_count is None:
        minimum_train = max(2, test_count + 1)
        span = 10 - minimum_train + 1
        train_count = minimum_train + ticket % span
    else:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + ticket % maximum_test

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("the requested pair counts cannot form a legal puzzle")

    return train_count, test_count


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = [int(value) for value in rng.permutation(10)]
    accent = colors[0]
    variable_palette = bool(int(rng.integers(0, 2)))
    if variable_palette:
        role_pairs = [(colors[i], colors[j]) for i, j in ((1, 2), (3, 4), (5, 6), (2, 4))]
    else:
        role_pairs = [(colors[1], colors[2])]
    shape_mode = int(rng.integers(0, 3))
    shapes = ((2, 2),) if shape_mode == 0 else (((2, 2), (2, 3)) if shape_mode == 1 else ((2, 2), (3, 2), (2, 3)))
    return {"accent": accent, "roles": role_pairs, "shapes": shapes}


def _compatible(first: tuple[int, int], second: tuple[int, int], shape: tuple[int, int]) -> bool:
    row_a, col_a = first
    row_b, col_b = second
    block_h, block_w = shape
    col_distance = abs(col_a - col_b)
    row_distance = abs(row_a - row_b)
    if col_distance < block_w:
        return row_distance > block_h
    if col_distance == block_w:
        return row_distance >= block_h
    return True


def _corner_contact(positions: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    block_h, block_w = shape
    return any(
        abs(a[0] - b[0]) == block_h and abs(a[1] - b[1]) == block_w
        for i, a in enumerate(positions) for b in positions[i + 1:]
    )


def _row_separable(positions: list[tuple[int, int]]) -> bool:
    count = len(positions)
    changed = [row for i, (row, _) in enumerate(positions) if (count - 1 - i) % 2 == 0]
    retained = [row for i, (row, _) in enumerate(positions) if (count - 1 - i) % 2 == 1]
    return max(changed) < min(retained) or max(retained) < min(changed)


def _sample_dimensions(rng: np.random.Generator, count: int, shape: tuple[int, int], regime: str) -> tuple[int, int]:
    block_h, block_w = shape
    height_max = 16 if regime == "source_train" else 20
    height = int(rng.integers(max(6, 2 * block_h + 1), height_max + 1))
    if int(rng.integers(0, 6)) == 0:
        height = max(7, 2 * block_h + 1)
    minimum_width = max(5, block_w * count - count // 4)
    if regime == "target_test":
        minimum_width = max(minimum_width, 11)
    maximum_width = min(30, max(minimum_width + 2, 4 * count + 7))
    width = int(rng.integers(minimum_width, maximum_width + 1))
    height = max(height, (2 * count * block_h * block_w) // width + 1)
    return height, width


def _sample_positions(
    rng: np.random.Generator,
    height: int,
    width: int,
    count: int,
    shape: tuple[int, int],
    require_spread: bool,
    require_nonseparable: bool,
    prefer_corner: bool,
) -> list[tuple[int, int]]:
    block_h, block_w = shape
    row_options = height - block_h + 1
    column_options = width - block_w + 1
    best = None
    best_score = -1
    for attempt in range(180):
        columns = sorted(int(x) for x in rng.choice(column_options, count, replace=False))
        positions = []
        valid = True
        for column in columns:
            available = [
                row for row in range(row_options)
                if all(_compatible((row, column), previous, shape) for previous in positions)
            ]
            if not available:
                valid = False
                break
            row = available[int(rng.integers(len(available)))]
            positions.append((row, column))
        if not valid:
            continue
        spread = len({row for row, _ in positions})
        nonseparable = not _row_separable(positions)
        corner = _corner_contact(positions, shape)
        score = spread + 2 * nonseparable + 2 * corner
        if score > best_score:
            best, best_score = positions, score
        if require_spread and spread < min(5, count, row_options):
            continue
        if require_nonseparable and not nonseparable:
            continue
        if prefer_corner and not corner:
            continue
        return positions
    if best is not None:
        return best
    # Every legal sampled width fits this staggered two-row construction.
    fallback = [
        ((index % 2) * (block_h + 1),
         (index // 2) * (block_w + 1) + index % 2)
        for index in range(count)
    ]
    assert all(row + block_h <= height and col + block_w <= width for row, col in fallback)
    return fallback


def _apply_rule(input_grid: np.ndarray, blocks: list[tuple[int, int]], shape: tuple[int, int], accent: int) -> np.ndarray:
    output = input_grid.copy()
    block_h, block_w = shape
    for index in range(len(blocks) - 1, -1, -2):
        row, column = blocks[index]
        output[row:row + block_h, column:column + block_w] = accent
    return output


def _build_pair(
    rng: np.random.Generator,
    regime: str,
    count: int,
    shape: tuple[int, int],
    roles: tuple[int, int],
    accent: int,
    forced_signature: tuple[int, int] | None = None,
) -> dict:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")
    for _ in range(20):
        height, width = _sample_dimensions(rng, count, shape, regime)
        if forced_signature is not None:
            width, count = forced_signature
            height = max(height, (2 * count * shape[0] * shape[1]) // width + 1)
        if width - shape[1] + 1 < count:
            continue
        positions = _sample_positions(
            rng, height, width, count, shape,
            require_spread=(count >= 5 and int(rng.integers(0, 4)) != 0),
            require_nonseparable=(count >= 5 and int(rng.integers(0, 5)) != 0),
            prefer_corner=(count >= 3 and int(rng.integers(0, 3)) == 0),
        )
        if regime == "target_test" and len({row for row, _ in positions}) < 3:
            continue
        background, foreground = roles
        input_grid = np.full((height, width), background, dtype=np.int8)
        for row, column in positions:
            input_grid[row:row + shape[0], column:column + shape[1]] = foreground
        output_grid = _apply_rule(input_grid, positions, shape, accent)
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not build pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_pair_counts(int(seed), rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)
    accent = parameters["accent"]
    shapes = parameters["shapes"]
    role_pool = parameters["roles"]
    train_pairs = []
    test_pairs = []
    seen_inputs = set()

    train_shapes = [shapes[i % len(shapes)] for i in range(train_count)]
    rng.shuffle(train_shapes)
    first_count = int(rng.integers(6, 10))
    second_candidates = [n for n in range(3, 10) if n % 2 != first_count % 2]
    second_count = second_candidates[int(rng.integers(len(second_candidates)))]
    train_counts = [first_count, second_count]
    train_counts.extend(int(rng.integers(2, 13 if shape[1] == 2 else 11)) for shape in train_shapes[2:])
    train_roles = [role_pool[i % len(role_pool)] for i in range(train_count)]
    rng.shuffle(train_roles)

    def unique_pair(regime, count, shape, roles, forced_signature=None):
        for _ in range(30):
            pair = _build_pair(rng, regime, count, shape, roles, accent, forced_signature)
            key = _array_key(pair["input"])
            if key in seen_inputs or _pair_key(pair) in _SOURCE_PAIR_KEYS:
                continue
            seen_inputs.add(key)
            return pair
        raise RuntimeError("could not produce a unique pair")

    for count, shape, roles in zip(train_counts, train_shapes, train_roles):
        train_pairs.append(unique_pair("source_train", count, shape, roles))

    for test_index in range(test_count):
        matching_train = [
            pair for pair in train_pairs
            if _analyze_pair(pair["input"], pair["output"])["count"] >= 5
        ]
        anchor = matching_train[int(rng.integers(len(matching_train)))]
        anchor_info = _analyze_pair(anchor["input"], anchor["output"])
        shape = anchor_info["shape"] if int(rng.integers(0, 3)) == 0 else train_shapes[int(rng.integers(train_count))]
        roles = train_roles[int(rng.integers(train_count))]
        familiar = (int(seed) + test_index) % 2 == 0
        if familiar:
            candidates = [
                pair for pair in train_pairs
                if _analyze_pair(pair["input"], pair["output"])["shape"] == shape
                and _analyze_pair(pair["input"], pair["output"])["count"] >= 5
            ]
            if candidates:
                chosen = candidates[int(rng.integers(len(candidates)))]
                info = _analyze_pair(chosen["input"], chosen["output"])
                forced = (info["width"], info["count"])
                count = info["count"]
            else:
                familiar = False
        if not familiar:
            count = int(rng.integers(5, 13 if shape[1] == 2 else 11))
            forced = None
        for _ in range(25):
            pair = unique_pair("target_test", count, shape, roles, forced)
            info = _analyze_pair(pair["input"], pair["output"])
            signature = (info["width"], info["count"])
            train_signatures = {
                (_analyze_pair(p["input"], p["output"])["width"], _analyze_pair(p["input"], p["output"])["count"])
                for p in train_pairs
            }
            if familiar or signature not in train_signatures:
                test_pairs.append(pair)
                break
            seen_inputs.remove(_array_key(pair["input"]))
        else:
            raise RuntimeError("could not sample test signature")

    puzzle = {"train": train_pairs, "test": test_pairs}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert 0 <= int(np.min(grid)) and int(np.max(grid)) <= 9


def _extract_blocks(grid: np.ndarray, background: int, foreground: int) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    assert set(int(x) for x in np.unique(grid)) == {background, foreground}
    mask = grid == foreground
    visited = np.zeros(grid.shape, dtype=bool)
    blocks = []
    shape = None
    for start_row, start_col in np.argwhere(mask):
        start_row, start_col = int(start_row), int(start_col)
        if visited[start_row, start_col]:
            continue
        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and mask[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        rows = [row for row, _ in cells]
        cols = [col for _, col in cells]
        top, left = min(rows), min(cols)
        block_shape = (max(rows) - top + 1, max(cols) - left + 1)
        assert block_shape in ((2, 2), (2, 3), (3, 2))
        assert len(cells) == block_shape[0] * block_shape[1]
        assert np.all(mask[top:top + block_shape[0], left:left + block_shape[1]])
        if shape is None:
            shape = block_shape
        assert block_shape == shape
        blocks.append((top, left))
    blocks.sort(key=lambda item: item[1])
    assert 2 <= len(blocks) <= 12
    assert len({col for _, col in blocks}) == len(blocks)
    assert sum(shape[0] * shape[1] for _ in blocks) == int(np.count_nonzero(mask))
    return blocks, shape


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    assert input_grid.shape == output_grid.shape
    _validate_grid(input_grid)
    _validate_grid(output_grid)
    changed = input_grid != output_grid
    assert np.any(changed)
    assert len(set(int(x) for x in np.unique(input_grid))) == 2
    old = set(int(x) for x in np.unique(input_grid[changed]))
    new = set(int(x) for x in np.unique(output_grid[changed]))
    assert len(old) == len(new) == 1
    foreground = next(iter(old))
    accent = next(iter(new))
    assert accent not in set(int(x) for x in np.unique(input_grid))
    background = next(iter(set(int(x) for x in np.unique(input_grid)) - {foreground}))
    assert int(np.count_nonzero(input_grid == background)) > int(np.count_nonzero(input_grid == foreground))
    blocks, shape = _extract_blocks(input_grid, background, foreground)
    assert np.array_equal(output_grid, _apply_rule(input_grid, blocks, shape, accent))
    assert set(int(x) for x in np.unique(output_grid)) == {background, foreground, accent}
    return {"background": background, "foreground": foreground, "accent": accent,
            "blocks": blocks, "shape": shape, "count": len(blocks),
            "height": input_grid.shape[0], "width": input_grid.shape[1]}


def _predict_from_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    accents = []
    for pair in train_pairs:
        mask = pair["input"] != pair["output"]
        accents.extend(int(x) for x in np.unique(pair["output"][mask]))
    assert accents and len(set(accents)) == 1
    accent = accents[0]
    colors, counts = np.unique(test_input, return_counts=True)
    assert len(colors) == 2
    background = int(colors[int(np.argmax(counts))])
    foreground = int(next(color for color in colors if color != background))
    blocks, shape = _extract_blocks(test_input, background, foreground)
    return _apply_rule(test_input, blocks, shape, accent)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            info = _analyze_pair(pair["input"], pair["output"])
            key = _array_key(pair["input"])
            assert key not in seen_inputs
            assert _pair_key(pair) not in _SOURCE_PAIR_KEYS
            seen_inputs.add(key)
            infos[split].append(info)
    assert len({info["accent"] for split in infos for info in infos[split]}) == 1
    assert {info["count"] % 2 for info in infos["train"]} == {0, 1}
    assert any(info["count"] >= 5 for info in infos["train"])
    trained_shapes = {info["shape"] for info in infos["train"]}
    trained_roles = {(info["background"], info["foreground"]) for info in infos["train"]}
    assert all(6 <= info["height"] <= 16 for info in infos["train"])
    for pair, info in zip(puzzle["test"], infos["test"]):
        assert info["shape"] in trained_shapes
        assert (info["background"], info["foreground"]) in trained_roles
        assert 6 <= info["height"] <= 20
        assert info["count"] >= 5
        assert len({row for row, _ in info["blocks"]}) >= 3
        assert np.array_equal(_predict_from_training(puzzle["train"], pair["input"]), pair["output"])


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[name], b[name]) and a[name].dtype == b[name].dtype
        for a, b in zip(first[split], second[split]) for name in ("input", "output")
    ) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert _puzzles_equal(generate_puzzle(17), generate_puzzle(17))
    assert not _puzzles_equal(generate_puzzle(17), generate_puzzle(18))
    assert _puzzles_equal(generate_puzzle(7001, 6, 3), generate_puzzle(7001, 6, 3))
    for seed, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4)), 8000):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for seed, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None)), 8100):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for index, arguments in enumerate((
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 2, "num_test": 3},
        {"num_train": 4, "num_test": 4},
    )):
        try:
            generate_puzzle(8200 + index, **arguments)
        except ValueError:
            continue
        raise AssertionError(f"illegal counts accepted: {arguments}")
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    infos = [(_analyze_pair(pair["input"], pair["output"]), split, index)
             for index, puzzle in enumerate(puzzles) for split in ("train", "test") for pair in puzzle[split]]
    assert len({i["height"] for i, _, _ in infos}) >= 10
    assert len({i["width"] for i, _, _ in infos}) >= 16
    assert len({i["count"] for i, _, _ in infos}) >= 9
    assert len({i["shape"] for i, _, _ in infos}) == 3
    assert len({i["accent"] for i, _, _ in infos}) >= 8
    assert sum(_corner_contact(i["blocks"], i["shape"]) for i, _, _ in infos) >= 20
    assert sum(len({r for r, _ in i["blocks"]}) >= 6 for i, _, _ in infos) >= 15
    dense = [i for i, _, _ in infos if i["count"] >= 5]
    assert sum(not _row_separable(i["blocks"]) for i in dense) >= len(dense) * 0.75
    assert any(len({(i["background"], i["foreground"]) for i, _, ix in infos if ix == index}) > 1 for index in range(42))
    assert any(len({i["height"] for i, _, ix in infos if ix == index}) > 1 for index in range(42))
    assert len({(i["height"], i["width"], i["count"], i["shape"], tuple(i["blocks"])) for i, _, _ in infos}) >= 180
    train_signatures = {
        index: {(i["width"], i["count"]) for i, split, ix in infos if ix == index and split == "train"}
        for index in range(42)
    }
    familiar = sum((i["width"], i["count"]) in train_signatures[index]
                   for i, split, index in infos if split == "test")
    novel = sum((i["width"], i["count"]) not in train_signatures[index]
                for i, split, index in infos if split == "test")
    assert familiar >= 10 and novel >= 10
