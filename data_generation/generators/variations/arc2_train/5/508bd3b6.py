import numpy as np


_SIDES = ("top", "right", "bottom", "left")

# Rejecting these inputs is stronger than merely rejecting the complete
# corresponding source input/output pairs.
_SOURCE_INPUTS = {
    """000000000000
000000000000
800000000000
080000000000
008000000000
000000000000
000000000000
000000000000
000000000000
222222222222
222222222222
222222222222""",
    """000000000022
000000000022
000000000022
000000000022
000000000022
000000000022
000000000022
000000000022
000000000022
000000000022
000800000022
008000000022""",
    """220000000000
220000000000
220000000000
220000000000
220000000000
220000000000
220000000000
220000000000
220000000000
220000800000
220000080000
220000008000""",
    """000800002222
000080002222
000000002222
000000002222
000000002222
000000002222
000000002222
000000002222
000000002222
000000002222
000000002222
000000002222""",
}


def _grid_text(grid: np.ndarray) -> str:
    return "\n".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_signature(pair: dict) -> tuple:
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _puzzle_signature(puzzle: dict) -> tuple:
    return (
        tuple(_pair_signature(pair) for pair in puzzle["train"]),
        tuple(_pair_signature(pair) for pair in puzzle["test"]),
    )


def _inside(row: int, col: int, height: int, width: int) -> bool:
    return 0 <= row < height and 0 <= col < width


def _barrier_mask(
    height: int,
    width: int,
    side: str,
    thickness: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)

    if side == "top":
        mask[:thickness, :] = True
    elif side == "right":
        mask[:, width - thickness:] = True
    elif side == "bottom":
        mask[height - thickness:, :] = True
    elif side == "left":
        mask[:, :thickness] = True
    else:
        raise ValueError("Unknown barrier side")

    return mask


def _coerce_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}, inclusive"
        )
    return value


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _coerce_count(num_train, "num_train", 2, 10)
    num_test = _coerce_count(num_test, "num_test", 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_train = int(rng.choice(
            np.arange(2, 11),
            p=[.22, .20, .16, .12, .09, .07, .06, .05, .03],
        ))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
        return num_train, num_test

    if num_train is None:
        assert num_test is not None
        num_train = int(rng.integers(num_test + 1, 11))
        return num_train, num_test

    maximum_test = min(4, num_train - 1)
    if maximum_test < 1:
        raise ValueError("No legal test count for the requested train count")

    num_test = int(rng.integers(1, maximum_test + 1))
    return num_train, num_test


def _sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    # The only convention invisible in a fresh input is its output color.
    # All three input roles can be identified from their geometry.
    return {"trace": int(rng.integers(0, 10))}


def _ray_direction(side: str, parallel_sign: int) -> tuple[int, int]:
    if parallel_sign not in (-1, 1):
        raise ValueError("parallel_sign must be -1 or 1")

    if side == "top":
        return -1, parallel_sign
    if side == "right":
        return parallel_sign, 1
    if side == "bottom":
        return 1, parallel_sign
    if side == "left":
        return parallel_sign, -1

    raise ValueError("Unknown barrier side")


def _reflected_direction(
    side: str,
    direction: tuple[int, int],
) -> tuple[int, int]:
    row_step, col_step = direction

    if side in ("top", "bottom"):
        return -row_step, col_step
    if side in ("left", "right"):
        return row_step, -col_step

    raise ValueError("Unknown barrier side")


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    side: str,
    parallel_sign: int,
    near_corner: bool = False,
) -> dict:
    if regime not in ("source_train", "target_test"):
        raise ValueError("Unknown generation regime")

    trace = puzzle_parameters["trace"]

    direction = _ray_direction(side, parallel_sign)
    reflected = _reflected_direction(side, direction)
    row_step, col_step = direction

    for _ in range(12000):
        size_mode = int(rng.integers(0, 5))
        if size_mode == 0:
            if int(rng.integers(0, 2)):
                height, width = 12, 12
            else:
                height, width = (int(rng.integers(10, 16)),
                                 int(rng.integers(10, 16)))
        elif size_mode == 1:
            height, width = (int(rng.integers(8, 13)),
                             int(rng.integers(8, 13)))
        elif size_mode == 2:
            height = int(rng.integers(12, 30))
            width = int(np.clip(height + int(rng.integers(-2, 3)), 8, 30))
        elif size_mode == 3:
            height, width = (int(rng.integers(9, 16)),
                             int(rng.integers(21, 31)))
            if int(rng.integers(0, 2)):
                height, width = width, height
        else:
            height, width = (int(rng.integers(22, 31)),
                             int(rng.integers(22, 31)))
        normal_size = height if side in ("top", "bottom") else width
        thickness = int(rng.integers(1, min(6, normal_size - 4) + 1))
        cue_length = int(rng.integers(2, 5))

        barrier_cells = _barrier_mask(
            height, width, side, thickness
        )

        forward_row = int(rng.integers(0, height))
        forward_col = int(rng.integers(0, width))

        cue_cells = [
            (
                forward_row - offset * row_step,
                forward_col - offset * col_step,
            )
            for offset in range(cue_length)
        ]

        if any(
            not _inside(row, col, height, width)
            for row, col in cue_cells
        ):
            continue

        if any(
            bool(barrier_cells[row, col])
            for row, col in cue_cells
        ):
            continue

        cue_set = set(cue_cells)

        incoming = []
        current_row, current_col = forward_row, forward_col
        hit_barrier = False

        for _step in range(height + width + 2):
            next_row = current_row + row_step
            next_col = current_col + col_step

            if not _inside(next_row, next_col, height, width):
                break

            if barrier_cells[next_row, next_col]:
                hit_barrier = True
                break

            if (next_row, next_col) in cue_set:
                break

            incoming.append((next_row, next_col))
            current_row, current_col = next_row, next_col

        if not hit_barrier or len(incoming) < 1:
            continue

        bounce_row, bounce_col = current_row, current_col
        reflected_row_step, reflected_col_step = reflected

        reflected_path = []
        current_row = bounce_row + reflected_row_step
        current_col = bounce_col + reflected_col_step
        valid_reflection = True

        while _inside(current_row, current_col, height, width):
            if barrier_cells[current_row, current_col]:
                valid_reflection = False
                break
            if (current_row, current_col) in cue_set:
                valid_reflection = False
                break

            reflected_path.append((current_row, current_col))
            current_row += reflected_row_step
            current_col += reflected_col_step

        if not valid_reflection or len(reflected_path) < 1:
            continue
        if near_corner and min(len(incoming), len(reflected_path)) != 1:
            continue
        if not near_corner and min(len(incoming), len(reflected_path)) < 2:
            continue

        complete_path = incoming + reflected_path
        if len(set(complete_path)) != len(complete_path):
            continue

        background, barrier, cue_color = (
            int(value) for value in rng.choice(
                [color for color in range(10) if color != trace],
                size=3, replace=False
            )
        )
        input_grid = np.full(
            (height, width), background, dtype=np.int8
        )
        input_grid[barrier_cells] = barrier

        for row, col in cue_cells:
            input_grid[row, col] = cue_color

        if any(
            int(input_grid[row, col]) != background
            for row, col in complete_path
        ):
            continue

        output_grid = input_grid.copy()
        for row, col in complete_path:
            output_grid[row, col] = trace

        if np.array_equal(input_grid, output_grid):
            continue

        if _grid_text(input_grid) in _SOURCE_INPUTS:
            continue

        return {
            "input": input_grid,
            "output": output_grid,
        }

    raise RuntimeError("Unable to construct a valid ray-reflection pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(
        rng, num_train, num_test
    )
    parameters = _sample_puzzle_parameters(rng)
    first = int(rng.integers(0, 4))
    second = (first + (1 if int(rng.integers(0, 2)) else 3)) % 4
    train_plan = [
        (_SIDES[first], int(rng.choice((-1, 1)))),
        (_SIDES[second], int(rng.choice((-1, 1)))),
    ]
    mode = int(rng.integers(0, 3))
    if num_train >= 3:
        if mode == 0:
            train_plan.append((train_plan[0][0], -train_plan[0][1]))
        elif mode == 1:
            train_plan.append((_SIDES[(first + 2) % 4], int(rng.choice((-1, 1)))))
        else:
            train_plan.append((_SIDES[int(rng.integers(0, 4))], int(rng.choice((-1, 1)))))
    if num_train >= 4 and mode == 1:
        train_plan.append((_SIDES[(second + 2) % 4], int(rng.choice((-1, 1)))))
    while len(train_plan) < num_train:
        train_plan.append((_SIDES[int(rng.integers(0, 4))], int(rng.choice((-1, 1)))))

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_pairs = set()

    def make_unique_pair(
        regime: str,
        side: str,
        parallel_sign: int,
        near_corner: bool,
    ) -> dict:
        for _ in range(200):
            child_seed = int(
                rng.integers(0, np.iinfo(np.int64).max)
            )
            pair_rng = np.random.default_rng(child_seed)
            pair = _generate_pair(
                pair_rng,
                regime,
                parameters,
                side,
                parallel_sign,
                near_corner,
            )

            input_signature = _grid_signature(pair["input"])
            pair_signature = _pair_signature(pair)

            if input_signature in used_inputs:
                continue
            if pair_signature in used_pairs:
                continue
            if regime == "source_train" and len(puzzle["train"]) == 1:
                first_analysis = _analyze_pair(
                    puzzle["train"][0]["input"],
                    puzzle["train"][0]["output"],
                )
                new_analysis = _analyze_pair(
                    pair["input"], pair["output"]
                )
                if all(
                    first_analysis[role] == new_analysis[role]
                    for role in ("background", "barrier", "cue")
                ):
                    continue

            used_inputs.add(input_signature)
            used_pairs.add(pair_signature)
            return pair

        raise RuntimeError("Unable to generate a unique puzzle pair")

    for index, (side, parallel_sign) in enumerate(train_plan):
        puzzle["train"].append(
            make_unique_pair(
                "source_train",
                side,
                parallel_sign,
                index >= 2 and bool(rng.random() < 0.16),
            )
        )
    for _ in range(num_test):
        choice = int(rng.integers(0, 3))
        side, parallel_sign = train_plan[int(rng.integers(0, len(train_plan)))]
        if choice == 1:
            parallel_sign = -parallel_sign
        elif choice == 2:
            unseen = [candidate for candidate in _SIDES
                      if candidate not in {item[0] for item in train_plan}]
            side = (unseen[int(rng.integers(0, len(unseen)))] if unseen
                    else _SIDES[int(rng.integers(0, 4))])
            parallel_sign = int(rng.choice((-1, 1)))
        puzzle["test"].append(
            make_unique_pair(
                "target_test",
                side,
                parallel_sign,
                bool(rng.random() < 0.24),
            )
        )

    validate_generated_puzzle(puzzle)
    return puzzle


def _find_barrier(grid: np.ndarray) -> tuple[int, str, int]:
    height, width = grid.shape
    candidates = []

    for color_value in np.unique(grid):
        color = int(color_value)
        actual = grid == color

        for side in _SIDES:
            maximum = (
                height - 1
                if side in ("top", "bottom")
                else width - 1
            )

            for thickness in range(1, maximum + 1):
                expected = _barrier_mask(
                    height, width, side, thickness
                )
                if np.array_equal(actual, expected):
                    candidates.append((color, side, thickness))

    assert len(candidates) == 1
    return candidates[0]


def _diagonal_segment(
    grid: np.ndarray,
    color: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    coordinates = np.argwhere(grid == color)
    count = len(coordinates)

    if not 2 <= count <= 4:
        return None

    order = np.argsort(coordinates[:, 0], kind="stable")
    coordinates = coordinates[order]

    row_differences = np.diff(coordinates[:, 0])
    col_differences = np.diff(coordinates[:, 1])

    if not np.all(row_differences == 1):
        return None
    if not np.all(col_differences == col_differences[0]):
        return None

    col_step = int(col_differences[0])
    if col_step not in (-1, 1):
        return None

    cells = [
        (int(row), int(col))
        for row, col in coordinates
    ]
    return (1, col_step), cells


def _direction_toward_barrier(
    side: str,
    axis: tuple[int, int],
) -> tuple[int, int]:
    candidates = (
        axis,
        (-axis[0], -axis[1]),
    )

    if side == "top":
        selected = [d for d in candidates if d[0] == -1]
    elif side == "bottom":
        selected = [d for d in candidates if d[0] == 1]
    elif side == "left":
        selected = [d for d in candidates if d[1] == -1]
    elif side == "right":
        selected = [d for d in candidates if d[1] == 1]
    else:
        selected = []

    assert len(selected) == 1
    return selected[0]


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    assert input_grid.shape == output_grid.shape
    height, width = input_grid.shape

    input_colors = {int(x) for x in np.unique(input_grid)}
    output_colors = {int(x) for x in np.unique(output_grid)}

    assert len(input_colors) == 3
    assert len(output_colors) == 4
    assert input_colors.issubset(output_colors)

    barrier, side, thickness = _find_barrier(input_grid)

    non_barrier_colors = input_colors - {barrier}
    cue_candidates = []

    for color in non_barrier_colors:
        segment = _diagonal_segment(input_grid, color)
        if segment is not None:
            cue_candidates.append((color, segment))

    assert len(cue_candidates) == 1
    cue_color, (axis, cue_cells) = cue_candidates[0]

    remaining = non_barrier_colors - {cue_color}
    assert len(remaining) == 1
    background = next(iter(remaining))

    difference = input_grid != output_grid
    assert np.any(difference)
    assert np.all(input_grid[difference] == background)

    difference_colors = {
        int(x) for x in np.unique(output_grid[difference])
    }
    assert len(difference_colors) == 1
    trace = next(iter(difference_colors))
    assert trace not in input_colors

    direction = _direction_toward_barrier(side, axis)
    row_step, col_step = direction
    cue_set = set(cue_cells)

    forward_candidates = []
    for row, col in cue_cells:
        next_cell = (row + row_step, col + col_step)
        if next_cell not in cue_set:
            forward_candidates.append((row, col))

    assert len(forward_candidates) == 1
    current_row, current_col = forward_candidates[0]

    incoming = []
    hit_barrier = False

    for _step in range(height + width + 2):
        next_row = current_row + row_step
        next_col = current_col + col_step

        if not _inside(next_row, next_col, height, width):
            break

        next_value = int(input_grid[next_row, next_col])
        if next_value == barrier:
            hit_barrier = True
            break

        assert next_value == background
        incoming.append((next_row, next_col))
        current_row, current_col = next_row, next_col

    assert hit_barrier
    assert len(incoming) >= 1

    bounce_row, bounce_col = current_row, current_col
    reflected = _reflected_direction(side, direction)
    reflected_row_step, reflected_col_step = reflected

    reflected_path = []
    current_row = bounce_row + reflected_row_step
    current_col = bounce_col + reflected_col_step

    while _inside(current_row, current_col, height, width):
        assert int(input_grid[current_row, current_col]) == background
        reflected_path.append((current_row, current_col))
        current_row += reflected_row_step
        current_col += reflected_col_step

    assert len(reflected_path) >= 1

    complete_path = incoming + reflected_path
    assert len(set(complete_path)) == len(complete_path)

    expected = input_grid.copy()
    for row, col in complete_path:
        expected[row, col] = trace

    assert np.array_equal(output_grid, expected)

    parallel_sign = (
        row_step
        if side in ("left", "right")
        else col_step
    )

    return {
        "background": background,
        "barrier": barrier,
        "cue": cue_color,
        "trace": trace,
        "side": side,
        "thickness": thickness,
        "parallel_sign": parallel_sign,
        "cue_length": len(cue_cells),
        "incoming_length": len(incoming),
        "reflected_length": len(reflected_path),
    }


def _validate_regime_structure(
    train_analyses: list[dict],
    test_analyses: list[dict],
) -> None:
    assert train_analyses
    assert test_analyses
    all_analyses = train_analyses + test_analyses
    trace = train_analyses[0]["trace"]
    assert all(analysis["trace"] == trace for analysis in all_analyses)
    assert all(trace not in (analysis["background"], analysis["barrier"],
                             analysis["cue"]) for analysis in all_analyses)
    assert len({(analysis["background"], analysis["barrier"], analysis["cue"])
                for analysis in train_analyses}) >= 2
    # Two clear demonstrations establish reflection for each stripe axis.
    assert any(analysis["side"] in ("top", "bottom")
               and min(analysis["incoming_length"], analysis["reflected_length"]) >= 2
               for analysis in train_analyses)
    assert any(analysis["side"] in ("left", "right")
               and min(analysis["incoming_length"], analysis["reflected_length"]) >= 2
               for analysis in train_analyses)
    assert all(1 <= analysis["thickness"] <= 6 and
               2 <= analysis["cue_length"] <= 4
               for analysis in all_analyses)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])

    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    seen_inputs = set()
    seen_pairs = set()
    train_analyses = []
    test_analyses = []

    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2

                height, width = grid.shape
                assert 1 <= height <= 30
                assert 1 <= width <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0
                assert int(grid.max()) <= 9

            assert input_grid.shape == output_grid.shape
            assert _grid_text(input_grid) not in _SOURCE_INPUTS

            input_signature = _grid_signature(input_grid)
            pair_signature = _pair_signature(pair)

            assert input_signature not in seen_inputs
            assert pair_signature not in seen_pairs
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)

            analysis = _analyze_pair(input_grid, output_grid)
            if split_name == "train":
                train_analyses.append(analysis)
            else:
                test_analyses.append(analysis)

    _validate_regime_structure(
        train_analyses,
        test_analyses,
    )


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_signature(p) for p in automatic}) == 42
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert _puzzle_signature(generate_puzzle(78123)) == _puzzle_signature(
        generate_puzzle(78123))
    assert _puzzle_signature(generate_puzzle(90210, 7, 3)) == _puzzle_signature(
        generate_puzzle(90210, 7, 3))

    additional = []
    for index, counts in enumerate(((2, 1), (10, 1), (10, 4), (6, 2))):
        puzzle = generate_puzzle(10000 + index, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        additional.append(puzzle)
    for index, counts in enumerate(((None, 1), (None, 4),
                                    (2, None), (10, None))):
        puzzle = generate_puzzle(11000 + index, *counts)
        assert counts[0] is None or len(puzzle["train"]) == counts[0]
        assert counts[1] is None or len(puzzle["test"]) == counts[1]
        additional.append(puzzle)
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4), (3.0, 1), (True, 1)):
        try:
            generate_puzzle(123, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count request accepted")

    analyses = []
    shape_set, layout_set = set(), set()
    for puzzle in automatic + additional:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair["input"], pair["output"])
                analyses.append((split, analysis))
                shape_set.add(pair["input"].shape)
                normalized = np.empty(pair["input"].shape, dtype=np.int8)
                for role, color in enumerate((
                    analysis["background"], analysis["barrier"], analysis["cue"]
                )):
                    normalized[pair["input"] == color] = role
                layout_set.add(_grid_signature(normalized))
    assert len(shape_set) >= 35
    assert len(layout_set) >= 100
    assert {a["side"] for _, a in analyses} == set(_SIDES)
    assert {a["cue_length"] for _, a in analyses} == {2, 3, 4}
    assert len({a["thickness"] for _, a in analyses}) >= 5
    assert len({a["trace"] for _, a in analyses}) >= 7
    assert any(min(a["incoming_length"], a["reflected_length"]) == 1
               for _, a in analyses)
    assert any(a["thickness"] >= 3 for split, a in analyses if split == "train")
    assert any(a["thickness"] <= 2 for split, a in analyses if split == "test")
    assert any(pair["input"].shape == (12, 12)
               for puzzle in automatic for split in ("train", "test")
               for pair in puzzle[split])
    assert any(max(pair["input"].shape) >= 28
               for puzzle in automatic for pair in puzzle["train"])
    assert any(max(pair["input"].shape) >= 28
               for puzzle in automatic for pair in puzzle["test"])
    assert any(max(pair["input"].shape) / min(pair["input"].shape) >= 1.5
               for puzzle in automatic for pair in puzzle["train"])
    assert any(max(pair["input"].shape) / min(pair["input"].shape) >= 1.5
               for puzzle in automatic for pair in puzzle["test"])
    all_four, both_signs = False, False
    unseen_test, familiar_test = False, False
    equal_thickness, thinner_test = False, False
    for puzzle in automatic:
        train = [_analyze_pair(pair["input"], pair["output"])
                 for pair in puzzle["train"]]
        test = [_analyze_pair(pair["input"], pair["output"])
                for pair in puzzle["test"]]
        sides = {a["side"] for a in train}
        configurations = {(a["side"], a["parallel_sign"]) for a in train}
        all_four |= sides == set(_SIDES)
        both_signs |= any((side, -1) in configurations and
                          (side, 1) in configurations for side in sides)
        for analysis in test:
            unseen_test |= analysis["side"] not in sides
            familiar_test |= ((analysis["side"], analysis["parallel_sign"])
                              in configurations)
            equal_thickness |= analysis["thickness"] in {
                a["thickness"] for a in train}
            thinner_test |= analysis["thickness"] < max(
                a["thickness"] for a in train)
    assert all((all_four, both_signs, unseen_test, familiar_test,
                equal_thickness, thinner_test))
