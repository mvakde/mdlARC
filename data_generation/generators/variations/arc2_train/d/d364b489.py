from __future__ import annotations

import numpy as np


_DIRECTIONS = (
    ("up", -1, 0),
    ("down", 1, 0),
    ("left", 0, -1),
    ("right", 0, 1),
)

_SOURCE_PAIR_ROWS = (
    (
        ("0000010000", "0000000000", "1000000000", "0000000001",
         "0000000000", "0000010000", "0000000000", "0000000000",
         "0010000000", "0000000001"),
        ("0000716000", "2000080000", "1600000002", "8000000071",
         "0000020008", "0000716000", "0000080000", "0020000000",
         "0716000002", "0080000071"),
    ),
    (
        ("0000000000", "0000001000", "0000000000", "0000000001",
         "0000000000", "0001000000", "0000000000", "0000000100",
         "0000000000", "0100000000"),
        ("0000002000", "0000071600", "0000008002", "0000000071",
         "0002000008", "0071600000", "0008000200", "0000007160",
         "0200000800", "7160000000"),
    ),
    (
        ("0100000001", "0000000000", "0000000100", "0001000000",
         "0000000000", "0000000000", "1000000100", "0000000000",
         "0000000000", "0000100000"),
        ("7160000071", "0800000208", "0002007160", "0071600800",
         "0008000000", "2000000200", "1600007160", "8000000800",
         "0000200000", "0007160000"),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_count(name: str, value: int | None, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must exceed num_test")
        return int(num_train), int(num_test)
    if num_train is None and num_test is None:
        test = int(rng.integers(1, 5))
        return int(rng.integers(test + 1, 11)), test
    if num_train is not None:
        return int(num_train), int(rng.integers(1, min(4, num_train - 1) + 1))
    return int(rng.integers(max(2, num_test + 1), 11)), int(num_test)


def _array_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(v) for v in grid.flat)


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    if input_grid.shape != (10, 10) or output_grid.shape != (10, 10):
        return False
    rows = (
        tuple("".join(str(int(v)) for v in row) for row in input_grid),
        tuple("".join(str(int(v)) for v in row) for row in output_grid),
    )
    return rows in _SOURCE_PAIR_ROWS


def _sample_dimensions(rng: np.random.Generator, regime: str) -> tuple[int, int]:
    probabilities = (
        (0.20, 0.43, 0.21, 0.16)
        if regime == "source_train"
        else (0.23, 0.30, 0.24, 0.23)
    )
    category = int(rng.choice(4, p=probabilities))
    if category == 0:
        return int(rng.integers(3, 8)), int(rng.integers(3, 8))
    if category == 1:
        return int(rng.integers(7, 19)), int(rng.integers(7, 19))
    if category == 2:
        return int(rng.integers(18, 31)), int(rng.integers(18, 31))
    short, long = int(rng.integers(3, 10)), int(rng.integers(18, 31))
    return (short, long) if rng.integers(2) else (long, short)


def _sample_count(rng: np.random.Generator, height: int, width: int) -> int:
    band = int(rng.choice(4, p=(0.16, 0.31, 0.34, 0.19)))
    if band == 0:
        count = 1
    elif band == 1:
        count = int(rng.integers(2, 5))
    elif band == 2:
        count = int(rng.integers(5, 10))
    else:
        count = int(rng.integers(10, 24))
    return min(count, max(1, (height * width) // 5))


def _corners(height: int, width: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]


def _can_add(cell: tuple[int, int], markers: list[tuple[int, int]]) -> bool:
    row, col = cell
    return all(abs(row - r) + abs(col - c) >= 3 for r, c in markers)


def _sample_positions(
    rng: np.random.Generator,
    height: int,
    width: int,
    desired: int,
    require_interior: bool,
) -> list[tuple[int, int]]:
    mode = str(rng.choice(
        ("scattered", "edge", "interior", "cluster", "row", "column"),
        p=(0.29, 0.19, 0.12, 0.17, 0.115, 0.115),
    ))
    corner_goal = min(
        desired,
        int(rng.choice((0, 1, 2, 3), p=(0.43, 0.36, 0.18, 0.03))),
    )
    if mode == "interior":
        corner_goal = 0

    markers: list[tuple[int, int]] = []
    if require_interior:
        markers.append((
            int(rng.integers(1, height - 1)),
            int(rng.integers(1, width - 1)),
        ))

    corners = _corners(height, width)
    for index in rng.permutation(4):
        corner = corners[int(index)]
        if len([cell for cell in markers if cell in corners]) >= corner_goal:
            break
        if _can_add(corner, markers):
            markers.append(corner)

    focus_row = int(rng.integers(height))
    focus_col = int(rng.integers(width))
    candidates = [
        (row, col)
        for row in range(height)
        for col in range(width)
        if (row, col) not in corners
        and (
            mode != "interior"
            or (0 < row < height - 1 and 0 < col < width - 1)
        )
    ]
    jitter = rng.random(len(candidates))

    def priority(index: int) -> float:
        row, col = candidates[index]
        noise = float(jitter[index])
        if mode == "edge":
            return (
                0 if row in (0, height - 1) or col in (0, width - 1) else 2
            ) + noise
        if mode == "cluster":
            return abs(row - focus_row) + abs(col - focus_col) + 2 * noise
        if mode == "row":
            return 2 * abs(row - focus_row) + noise
        if mode == "column":
            return 2 * abs(col - focus_col) + noise
        return noise

    for index in sorted(range(len(candidates)), key=priority):
        if len(markers) >= desired:
            break
        cell = candidates[index]
        if _can_add(cell, markers):
            markers.append(cell)
    return sorted(markers)


def _stamp(
    input_grid: np.ndarray,
    markers: list[tuple[int, int]],
    arm_colors: dict[str, int],
) -> np.ndarray:
    output = input_grid.copy()
    height, width = input_grid.shape
    for row, col in markers:
        for direction, delta_row, delta_col in _DIRECTIONS:
            arm_row, arm_col = row + delta_row, col + delta_col
            if 0 <= arm_row < height and 0 <= arm_col < width:
                output[arm_row, arm_col] = arm_colors[direction]
    return output


def _make_pair(
    rng: np.random.Generator,
    regime: str,
    background: int,
    center_pool: tuple[int, ...],
    arm_colors: dict[str, int],
    anchor: bool,
) -> dict:
    if anchor:
        height, width = int(rng.integers(9, 19)), int(rng.integers(9, 19))
        desired = max(len(center_pool), int(rng.integers(4, 10)))
    else:
        height, width = _sample_dimensions(rng, regime)
        desired = _sample_count(rng, height, width)
    markers = _sample_positions(rng, height, width, desired, anchor)
    if anchor and len(markers) < len(center_pool):
        # A roomy deterministic fallback keeps legal count requests total.
        height, width = 17, 19
        markers = _sample_positions(rng, height, width, len(center_pool) + 3, True)
    _require(bool(markers), "position construction produced no marker")

    input_grid = np.full((height, width), background, dtype=np.int8)
    if anchor:
        colors = list(center_pool)
        colors.extend(int(rng.choice(center_pool)) for _ in range(len(markers) - len(colors)))
        rng.shuffle(colors)
    else:
        number_colors = int(rng.integers(1, min(len(center_pool), len(markers)) + 1))
        chosen = [int(v) for v in rng.choice(center_pool, size=number_colors, replace=False)]
        colors = chosen + [
            chosen[int(rng.integers(len(chosen)))]
            for _ in range(len(markers) - number_colors)
        ]
        rng.shuffle(colors)
    for (row, col), color in zip(markers, colors):
        input_grid[row, col] = color
    return {"input": input_grid, "output": _stamp(input_grid, markers, arm_colors)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    palette = [int(v) for v in rng.permutation(10)]
    arm_colors = dict(zip((name for name, _, _ in _DIRECTIONS), palette[:4]))
    backgrounds = (palette[4], palette[5])
    pool_size = int(rng.integers(2, 5))
    center_pool = tuple(palette[6:6 + pool_size])

    puzzle: dict[str, list[dict]] = {"train": [], "test": []}
    used: set[tuple] = set()
    for section, regime, count in (
        ("train", "source_train", train_count),
        ("test", "target_test", test_count),
    ):
        for index in range(count):
            for attempt in range(16):
                child_rng = np.random.default_rng(
                    rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
                )
                background = (
                    backgrounds[index]
                    if section == "train" and index < 2
                    else backgrounds[int(rng.integers(2))]
                )
                pair = _make_pair(
                    child_rng,
                    regime,
                    background,
                    center_pool,
                    arm_colors,
                    section == "train" and index == 0,
                )
                key = _array_key(pair["input"])
                if key not in used and not _is_source_pair(pair["input"], pair["output"]):
                    used.add(key)
                    puzzle[section].append(pair)
                    break
            else:
                raise ValueError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_input(grid: np.ndarray) -> dict:
    values, counts = np.unique(grid, return_counts=True)
    _require(len(values) >= 2, "input needs background and markers")
    background = int(values[int(np.argmax(counts))])
    _require(
        int(np.max(counts)) > grid.size // 2,
        "background must occupy more than half the input",
    )
    markers = [(int(r), int(c)) for r, c in np.argwhere(grid != background)]
    for index, cell in enumerate(markers):
        _require(_can_add(cell, markers[:index]), "marker crosses overlap")
    return {
        "background": background,
        "markers": markers,
        "centers": {int(grid[r, c]) for r, c in markers},
    }


def _learn_arms(train: list[dict], infos: list[dict]) -> dict[str, int]:
    seen = {name: set() for name, _, _ in _DIRECTIONS}
    full_cross = False
    for pair, info in zip(train, infos):
        output = pair["output"]
        height, width = output.shape
        for row, col in info["markers"]:
            _require(output[row, col] == pair["input"][row, col], "center was changed")
            full_cross |= 0 < row < height - 1 and 0 < col < width - 1
            for name, delta_row, delta_col in _DIRECTIONS:
                arm_row, arm_col = row + delta_row, col + delta_col
                if 0 <= arm_row < height and 0 <= arm_col < width:
                    seen[name].add(int(output[arm_row, arm_col]))
    _require(full_cross, "training needs at least one complete cross")
    _require(all(len(values) == 1 for values in seen.values()),
             "training must identify one stable color per direction")
    arms = {name: next(iter(values)) for name, values in seen.items()}
    _require(len(set(arms.values())) == 4, "direction colors must be distinct")
    return arms


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "illegal pair counts")
    used_inputs: set[tuple] = set()
    used_pairs: set[tuple] = set()
    infos = []
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair needs exactly input and output")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                     f"{name} must be a two-dimensional ndarray")
            height, width = grid.shape
            _require(1 <= height <= 30 and 1 <= width <= 30,
                     f"{name} dimensions are illegal")
            _require(np.issubdtype(grid.dtype, np.integer),
                     f"{name} must have integer dtype")
            _require(bool(np.all((0 <= grid) & (grid <= 9))),
                     f"{name} must use ARC colors")
        input_grid, output_grid = pair["input"], pair["output"]
        _require(input_grid.shape == output_grid.shape, "output dimensions changed")
        _require(not np.array_equal(input_grid, output_grid), "cross made no change")
        input_key, output_key = _array_key(input_grid), _array_key(output_grid)
        _require(input_key not in used_inputs, "duplicate input")
        _require((input_key, output_key) not in used_pairs, "duplicate pair")
        used_inputs.add(input_key)
        used_pairs.add((input_key, output_key))
        _require(not _is_source_pair(input_grid, output_grid), "source pair copied")
        infos.append(_analyze_input(input_grid))

    train_infos, test_infos = infos[:len(train)], infos[len(train):]
    _require(len({info["background"] for info in train_infos}) >= 2,
             "training must demonstrate varying backgrounds")
    _require(any(len(info["centers"]) >= 2 for info in train_infos),
             "training must demonstrate independent center colors in one grid")
    seen_centers = set().union(*(info["centers"] for info in train_infos))
    seen_backgrounds = {info["background"] for info in train_infos}
    _require(all(info["centers"] <= seen_centers for info in test_infos),
             "test contains an undemonstrated marker color")
    _require(all(info["background"] in seen_backgrounds for info in test_infos),
             "test contains an undemonstrated background color")

    arms = _learn_arms(train, train_infos)
    input_colors = set().union(*(
        info["centers"] | {info["background"]} for info in infos
    ))
    _require(not (set(arms.values()) & input_colors),
             "input roles collide with output arm colors")
    for pair, info in zip(train + test, infos):
        expected = _stamp(pair["input"], info["markers"], arms)
        _require(np.array_equal(expected, pair["output"]),
                 "output is not independent clipped directional stamping")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (section, _array_key(pair["input"]), _array_key(pair["output"]))
        for section in ("train", "test")
        for pair in puzzle[section]
    )


def _check_pair_cellwise(pair: dict, arms: dict[str, int]) -> None:
    """Check the local rule without using the generation-time stamping routine."""
    source, result = pair["input"], pair["output"]
    background = int(np.bincount(source.ravel(), minlength=10).argmax())
    height, width = source.shape
    for row in range(height):
        for col in range(width):
            if int(source[row, col]) != background:
                expected = int(source[row, col])
            else:
                touching = []
                for name, delta_row, delta_col in _DIRECTIONS:
                    center_row, center_col = row - delta_row, col - delta_col
                    if (
                        0 <= center_row < height
                        and 0 <= center_col < width
                        and int(source[center_row, center_col]) != background
                    ):
                        touching.append(arms[name])
                _require(len(touching) <= 1, "two centers compete for an arm cell")
                expected = touching[0] if touching else background
            _require(int(result[row, col]) == expected,
                     "cellwise cross rule differs from output")


def self_test() -> None:
    train_counts, test_counts, palettes, shapes = set(), set(), set(), set()
    layouts, marker_counts = set(), {"train": set(), "test": set()}
    corners = {"train": set(), "test": set()}
    repeated_rows = {"train": 0, "test": 0}
    repeated_columns = {"train": 0, "test": 0}
    one_marker = {"train": 0, "test": 0}
    dense = {"train": 0, "test": 0}
    interior_only = {"train": 0, "test": 0}
    clipped_only_test = 0
    background_changes = 0
    multi_center_grids = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_infos = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        arms = _learn_arms(puzzle["train"], train_infos)
        palettes.add(tuple(arms[name] for name, _, _ in _DIRECTIONS))
        background_changes += len({info["background"] for info in train_infos}) >= 2
        layout = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                _check_pair_cellwise(pair, arms)
                input_grid = pair["input"]
                info = _analyze_input(input_grid)
                height, width = input_grid.shape
                markers = info["markers"]
                shapes.add((height, width))
                marker_counts[section].add(len(markers))
                layout.append(((height, width), tuple(markers)))
                corner_count = len(set(markers) & set(_corners(height, width)))
                corners[section].add(corner_count)
                repeated_rows[section] += len({r for r, _ in markers}) < len(markers)
                repeated_columns[section] += len({c for _, c in markers}) < len(markers)
                one_marker[section] += len(markers) == 1
                dense[section] += len(markers) >= 10
                interior_only[section] += all(
                    0 < r < height - 1 and 0 < c < width - 1 for r, c in markers
                )
                multi_center_grids += len(info["centers"]) >= 2
                if section == "test":
                    clipped_only_test += all(
                        r in (0, height - 1) or c in (0, width - 1)
                        for r, c in markers
                    )
                    # The prediction uses only copied training demonstrations
                    # and this test input; the expected output is compared afterward.
                    predicted = _stamp(input_grid.copy(), markers, arms)
                    _require(np.array_equal(predicted, pair["output"]),
                             "training-derived test prediction failed")
        layouts.add(tuple(layout))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(palettes) >= 30 and len(shapes) >= 80 and len(layouts) == 42,
             "palettes, dimensions, or layouts lack diversity")
    _require(any(h == 30 or w == 30 for h, w in shapes),
             "ARC maximum dimension was not sampled")
    _require(any(min(h, w) <= 6 and max(h, w) >= 18 for h, w in shapes),
             "pronounced rectangles were not sampled")
    _require(background_changes == 42 and multi_center_grids >= 42,
             "training input-role variation is missing")
    for section in ("train", "test"):
        _require({0, 1, 2} <= corners[section],
                 f"{section} corner count variation is missing")
        _require(repeated_rows[section] >= 4 and repeated_columns[section] >= 4,
                 f"{section} repeated row or column variation is missing")
        _require(one_marker[section] >= 2 and dense[section] >= 2,
                 f"{section} sparse or dense cases are missing")
        _require(interior_only[section] >= 2,
                 f"{section} interior-only cases are missing")
        _require(len(marker_counts[section]) >= 10,
                 f"{section} marker counts lack range")
    _require(clipped_only_test >= 2, "clipped-only tests are missing")

    for seed, train_count, test_count in (
        (101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 6, 3)
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require((len(puzzle["train"]), len(puzzle["test"])) ==
                 (train_count, test_count), "explicit counts ignored")
        validate_generated_puzzle(puzzle)
    for seed, train_count, test_count in (
        (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None)
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        _require(train_count is None or len(puzzle["train"]) == train_count,
                 "partial train count ignored")
        _require(test_count is None or len(puzzle["test"]) == test_count,
                 "partial test count ignored")
        validate_generated_puzzle(puzzle)
    _require(_puzzle_key(generate_puzzle(999, 7, 3)) ==
             _puzzle_key(generate_puzzle(999, 7, 3)), "not deterministic")
    _require(_puzzle_key(generate_puzzle(1000, 7, 3)) !=
             _puzzle_key(generate_puzzle(1001, 7, 3)), "seeds did not differ")
    for counts in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(300, **counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
