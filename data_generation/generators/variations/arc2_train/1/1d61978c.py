import numpy as np


_AUTO_COUNT_PAIRS = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 1),
    (5, 2),
    (6, 3),
    (7, 2),
    (8, 4),
    (9, 3),
    (10, 4),
    (10, 1),
    (6, 2),
)


# Rejecting an exact source input is stronger than merely rejecting an exact
# source input/output pair.
_SOURCE_INPUT_KEYS = {
    (
        "7777577777577777"
        "7777757777757777"
        "7777775777775777"
        "7777777577777577"
        "7777777757777757"
        "7777777775777775"
        "7777777777577777"
        "7777777757757777"
        "7777777577775777"
        "7777775777777577"
        "7777757777777777"
        "7777577777777777"
        "7775777777777777"
        "7757777777777777"
        "7577777777777777"
        "5777777777777777"
    ),
    (
        "7777777777777777"
        "7777777777777777"
        "7777777577777775"
        "7775777757777757"
        "7777577777777577"
        "7757757777775777"
        "7775775777757777"
        "7777577777577777"
        "7777757775777777"
        "7777775777777777"
        "7777777577777757"
        "7777757757777577"
        "7777577775775777"
        "7775777777757777"
        "7777777777577777"
        "7777777777777777"
    ),
    (
        "7777777777777777"
        "7777777777777777"
        "7777777777777777"
        "7777777777777777"
        "7777777777777777"
        "7777777777577777"
        "7777777775777777"
        "7777777577777777"
        "7777777757777777"
        "7777777775777777"
        "7777777777777777"
        "7777777777777777"
        "7777777777777777"
        "7777777777777777"
        "7777777777777777"
        "7777777777777777"
    ),
}


def _coerce_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    num_train = _coerce_count(num_train, "num_train", 2, 10)
    num_test = _coerce_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        return _AUTO_COUNT_PAIRS[seed % len(_AUTO_COUNT_PAIRS)]

    if num_train is None:
        minimum_train = max(2, num_test + 1)
        choices = tuple(range(minimum_train, 11))
        if not choices:
            raise ValueError("no legal num_train exists for num_test")
        num_train = choices[seed % len(choices)]

    if num_test is None:
        maximum_test = min(4, num_train - 1)
        choices = tuple(range(1, maximum_test + 1))
        if not choices:
            raise ValueError("no legal num_test exists for num_train")
        num_test = choices[seed % len(choices)]

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _sample_puzzle_parameters(rng):
    majority_color, minority_color = (
        int(value) for value in rng.choice(10, size=2, replace=False)
    )
    # Only the meaning of the two output colors has to be learned globally.
    # The background and marker are identifiable afresh in every input.
    fixed_shape = (
        _sample_shape(rng, witness=True)
        if rng.random() < 0.35 else None
    )
    fixed_input_colors = (
        _sample_input_colors(rng, majority_color, minority_color)
        if rng.random() < 0.35 else None
    )
    return {
        "majority_color": majority_color,
        "minority_color": minority_color,
        "fixed_shape": fixed_shape,
        "fixed_input_colors": fixed_input_colors,
    }


def _array_key(grid):
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _is_source_input(grid):
    if grid.shape != (16, 16):
        return False
    content = "".join(str(int(value)) for value in grid.ravel())
    return content in _SOURCE_INPUT_KEYS


def _analyze_input(input_grid):
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 2

    background_index = int(np.argmax(counts))
    marker_index = 1 - background_index

    background = int(values[background_index])
    marker = int(values[marker_index])
    assert int(counts[background_index]) > int(counts[marker_index])

    marker_mask = input_grid == marker
    height, width = input_grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components = []

    marker_positions = np.argwhere(marker_mask)
    for row_value, col_value in marker_positions:
        start_row = int(row_value)
        start_col = int(col_value)
        if seen[start_row, start_col]:
            continue

        stack = [(start_row, start_col)]
        seen[start_row, start_col] = True
        component = []

        while stack:
            row, col = stack.pop()
            component.append((row, col))

            for row_delta in (-1, 0, 1):
                for col_delta in (-1, 0, 1):
                    if row_delta == 0 and col_delta == 0:
                        continue
                    next_row = row + row_delta
                    next_col = col + col_delta
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if marker_mask[next_row, next_col] and not seen[
                        next_row, next_col
                    ]:
                        seen[next_row, next_col] = True
                        stack.append((next_row, next_col))

        component.sort()
        assert len(component) >= 2

        rows = [row for row, _ in component]
        consecutive_rows = (
            len(set(rows)) == len(rows)
            and max(rows) - min(rows) + 1 == len(rows)
        )
        is_down_right = (
            consecutive_rows
            and len({row - col for row, col in component}) == 1
        )
        is_down_left = (
            consecutive_rows
            and len({row + col for row, col in component}) == 1
        )

        # Every component must be exactly one unambiguous 45-degree run.
        assert is_down_right != is_down_left
        orientation = "dr" if is_down_right else "dl"

        components.append(
            {
                "orientation": orientation,
                "cells": tuple(component),
            }
        )

    components.sort(key=lambda component: component["cells"])
    assert len(components) >= 2

    totals = {"dr": 0, "dl": 0}
    for component in components:
        totals[component["orientation"]] += len(component["cells"])

    assert totals["dr"] > 0
    assert totals["dl"] > 0
    assert totals["dr"] != totals["dl"]

    majority = "dr" if totals["dr"] > totals["dl"] else "dl"

    return {
        "background": background,
        "marker": marker,
        "marker_mask": marker_mask,
        "components": components,
        "totals": totals,
        "majority": majority,
    }


def _solve_input(input_grid, majority_color, minority_color):
    analysis = _analyze_input(input_grid)
    output_grid = np.full_like(input_grid, analysis["background"])

    for component in analysis["components"]:
        color = (
            majority_color
            if component["orientation"] == analysis["majority"]
            else minority_color
        )
        for row, col in component["cells"]:
            output_grid[row, col] = color

    return output_grid


def _sample_shape(rng, witness=False):
    if witness:
        if rng.random() < 0.28:
            short = int(rng.integers(14, 20))
            long = int(rng.integers(21, 31))
            return (short, long) if rng.random() < 0.5 else (long, short)
        return (
            int(rng.integers(16, 31)),
            int(rng.integers(16, 31)),
        )

    style = int(rng.integers(0, 4))
    if style == 0:
        return (
            int(rng.integers(6, 11)),
            int(rng.integers(6, 11)),
        )
    if style == 1:
        return (
            int(rng.integers(11, 21)),
            int(rng.integers(11, 21)),
        )
    if style == 2:
        return (
            int(rng.integers(21, 31)),
            int(rng.integers(21, 31)),
        )
    short = int(rng.integers(7, 14))
    long = int(rng.integers(21, 31))
    return (short, long) if rng.random() < 0.5 else (long, short)


def _sample_input_colors(rng, majority_color, minority_color):
    available = [
        color for color in range(10)
        if color not in (majority_color, minority_color)
    ]
    indices = rng.choice(len(available), size=2, replace=False)
    return (available[int(indices[0])], available[int(indices[1])])


def _candidate_segments(height, width, orientation, length):
    candidates = []
    for start_row in range(height - length + 1):
        if orientation == "dr":
            columns = range(width - length + 1)
            step = 1
        else:
            columns = range(length - 1, width)
            step = -1
        for start_col in columns:
            candidates.append(tuple(
                (start_row + offset, start_col + step * offset)
                for offset in range(length)
            ))
    return candidates


def _touches_edge(cells, height, width):
    return any(
        row in (0, height - 1) or col in (0, width - 1)
        for row, col in cells
    )


def _touches_corner(cells, height, width):
    return any(
        row in (0, height - 1) and col in (0, width - 1)
        for row, col in cells
    )


def _place_specs(rng, height, width, specs, anchor_style):
    candidate_lists = [
        _candidate_segments(height, width, orientation, length)
        for orientation, length in specs
    ]
    if not all(candidate_lists):
        raise RuntimeError("a run does not fit the canvas")

    anchor = int(rng.integers(0, len(specs)))
    if anchor_style == "edge":
        candidate_lists[anchor] = [
            cells for cells in candidate_lists[anchor]
            if _touches_edge(cells, height, width)
        ]
    elif anchor_style == "corner":
        candidate_lists[anchor] = [
            cells for cells in candidate_lists[anchor]
            if _touches_corner(cells, height, width)
        ]
    if not all(candidate_lists):
        raise RuntimeError("the requested anchor does not fit")

    # A one-cell moat makes eight-connected components exactly the runs.
    for _ in range(36):
        blocked = np.zeros((height, width), dtype=bool)
        placed = [None] * len(specs)
        remainder = [index for index in range(len(specs)) if index != anchor]
        random_ties = rng.random(len(specs))
        remainder.sort(
            key=lambda index: (-specs[index][1], random_ties[index])
        )
        order = [anchor] + remainder
        success = True
        for index in order:
            candidates = candidate_lists[index]
            chosen = None
            for choice in rng.permutation(len(candidates)):
                cells = candidates[int(choice)]
                if all(not blocked[row, col] for row, col in cells):
                    chosen = cells
                    break
            if chosen is None:
                success = False
                break
            placed[index] = chosen
            for row, col in chosen:
                blocked[
                    max(0, row - 1):min(height, row + 2),
                    max(0, col - 1):min(width, col + 2),
                ] = True
        if success:
            return placed
    raise RuntimeError("could not separate all runs")


def _specs_from_lengths(rng, majority_orientation, winners, losers):
    loser_orientation = "dl" if majority_orientation == "dr" else "dr"
    specs = (
        [(majority_orientation, int(length)) for length in winners]
        + [(loser_orientation, int(length)) for length in losers]
    )
    return [specs[int(index)] for index in rng.permutation(len(specs))]


def _sample_witness_specs(rng, majority_orientation, kind, max_length):
    # Two complementary demonstrations defeat longest-run and run-count
    # shortcuts even when the puzzle has only two training examples.
    for _ in range(1000):
        if kind == "witness_longest":
            winner_count = int(rng.integers(2, 5))
            loser_count = int(rng.integers(1, min(3, winner_count)))
            winners = rng.integers(
                2, max_length, size=winner_count
            ).tolist()
            losers = rng.integers(
                2, max_length + 1, size=loser_count
            ).tolist()
            if max(losers) <= max(winners):
                continue
        elif kind == "witness_count":
            winner_count = int(rng.integers(2, 4))
            loser_count = int(rng.integers(
                winner_count + 1, min(6, winner_count + 3)
            ))
            winners = rng.integers(
                4, max_length + 1, size=winner_count
            ).tolist()
            losers = rng.integers(
                2, max(4, max_length - 2), size=loser_count
            ).tolist()
        else:
            raise ValueError("unknown witness kind")
        if sum(winners) > sum(losers):
            return _specs_from_lengths(
                rng, majority_orientation, winners, losers
            )
    raise RuntimeError("could not sample training witness")


def _sample_generic_specs(rng, majority_orientation, height, width, mode):
    short_side = min(height, width)
    max_length = min(14, short_side - 2)
    if short_side <= 8:
        max_runs = 3
    elif short_side <= 11:
        max_runs = 4
    elif short_side <= 15:
        max_runs = 6
    else:
        max_runs = 8

    for _ in range(900):
        if mode == "sparse":
            winners, losers = [3], [2]
        else:
            if mode == "two":
                run_count = 2
            elif mode == "dense":
                run_count = int(rng.integers(6, max_runs + 1)) if max_runs >= 6 else max_runs
            else:
                run_count = int(rng.integers(3, min(6, max_runs + 1)))
            winner_count = int(rng.integers(
                max(1, run_count - 4), min(4, run_count - 1) + 1
            ))
            loser_count = run_count - winner_count
            winners = rng.integers(
                2, max_length + 1, size=winner_count
            ).tolist()
            losers = rng.integers(
                2, max_length + 1, size=loser_count
            ).tolist()
        margin = sum(winners) - sum(losers)
        if margin <= 0:
            continue
        if mode != "sparse":
            margin_style = int(rng.integers(0, 3))
            if margin_style == 0 and margin != 1:
                continue
            if margin_style == 1 and margin < 3:
                continue
        if sum(winners) + sum(losers) > min(
            68, int(0.22 * height * width)
        ):
            continue
        return _specs_from_lengths(
            rng, majority_orientation, winners, losers
        )
    raise RuntimeError("could not sample valid run lengths")


def _generate_pair(rng, regime, parameters, desired_majority, mode):
    majority_color = parameters["majority_color"]
    minority_color = parameters["minority_color"]
    witness = mode in ("witness_longest", "witness_count")
    for _ in range(120):
        shape = parameters["fixed_shape"]
        if shape is None:
            shape = _sample_shape(rng, witness=witness or mode == "aggregate")
        height, width = shape
        if mode == "aggregate":
            specs = _sample_witness_specs(
                rng, desired_majority, "witness_longest",
                min(12, min(height, width) - 2),
            )
        elif witness:
            specs = _sample_witness_specs(
                rng, desired_majority, mode,
                min(12, min(height, width) - 2),
            )
        else:
            try:
                specs = _sample_generic_specs(
                    rng, desired_majority, height, width, mode
                )
            except RuntimeError:
                continue
        anchor_style = ("any", "edge", "corner")[
            int(rng.choice(3, p=[0.40, 0.35, 0.25]))
        ]
        try:
            placed = _place_specs(
                rng, height, width, specs, anchor_style
            )
        except RuntimeError:
            continue

        colors = parameters["fixed_input_colors"]
        if colors is None:
            colors = _sample_input_colors(
                rng, majority_color, minority_color
            )
        background, marker = colors
        input_grid = np.full((height, width), background, dtype=np.int8)
        for segment in placed:
            for row, col in segment:
                input_grid[row, col] = marker
        try:
            analysis = _analyze_input(input_grid)
        except AssertionError:
            continue
        if analysis["majority"] != desired_majority:
            continue
        output_grid = _solve_input(
            input_grid, majority_color, minority_color
        )
        return {"input": input_grid, "output": output_grid}
    # A deterministic roomy construction covers unusually unlucky rejection
    # sequences. The same witnessed rule and color convention still apply.
    if mode in ("witness_longest", "aggregate"):
        runs = [
            ("win", 4, 2, 2), ("win", 4, 2, 12),
            ("lose", 7, 12, 25),
        ]
    elif mode == "witness_count":
        runs = [
            ("win", 8, 1, 1), ("win", 8, 1, 12),
            ("lose", 3, 13, 28), ("lose", 3, 13, 18),
            ("lose", 3, 21, 28), ("lose", 3, 21, 18),
        ]
    else:
        runs = [
            ("win", 3, 2, 2), ("lose", 2, 20, 25),
        ]
    colors = parameters["fixed_input_colors"]
    if colors is None:
        colors = _sample_input_colors(
            rng, majority_color, minority_color
        )
    background, marker = colors
    input_grid = np.full((30, 30), background, dtype=np.int8)
    for role, length, row, col in runs:
        slope = "dr" if role == "win" else "dl"
        for offset in range(length):
            x = col + offset * (1 if slope == "dr" else -1)
            input_grid[row + offset, x] = marker
    # Mirroring positions as well as directions keeps the fallback separated.
    if desired_majority == "dl":
        input_grid = np.fliplr(input_grid).copy()
    output_grid = _solve_input(
        input_grid, majority_color, minority_color
    )
    return {"input": input_grid, "output": output_grid}


def _next_child_rng(master_rng, used_child_seeds):
    for _ in range(1000):
        child_seed = int(master_rng.integers(
            0, np.iinfo(np.uint64).max, dtype=np.uint64
        ))
        if child_seed not in used_child_seeds:
            used_child_seeds.add(child_seed)
            return np.random.default_rng(child_seed)
    raise RuntimeError("could not derive a unique pair seed")


def _check_pair_transformation(input_grid, output_grid):
    assert input_grid.shape == output_grid.shape
    analysis = _analyze_input(input_grid)
    background = analysis["background"]
    marker = analysis["marker"]
    marker_mask = analysis["marker_mask"]
    assert np.all(output_grid[~marker_mask] == background)
    majority_component = next(
        component for component in analysis["components"]
        if component["orientation"] == analysis["majority"]
    )
    minority_component = next(
        component for component in analysis["components"]
        if component["orientation"] != analysis["majority"]
    )
    majority_color = int(output_grid[majority_component["cells"][0]])
    minority_color = int(output_grid[minority_component["cells"][0]])
    assert len({
        background, marker, majority_color, minority_color
    }) == 4
    for component in analysis["components"]:
        color = (
            majority_color
            if component["orientation"] == analysis["majority"]
            else minority_color
        )
        for row, col in component["cells"]:
            assert int(output_grid[row, col]) == color
    assert set(int(v) for v in np.unique(output_grid)) == {
        background, majority_color, minority_color
    }
    return analysis, (
        background, marker, majority_color, minority_color
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng)

    first_orientation = (
        "dr" if int(rng.integers(0, 2)) == 0 else "dl"
    )
    other_orientation = (
        "dl" if first_orientation == "dr" else "dr"
    )
    train_plan = [
        ("witness_longest", first_orientation),
        ("witness_count", other_orientation),
    ]
    for _ in range(num_train - 2):
        orientation = "dr" if rng.random() < 0.5 else "dl"
        mode = ("two", "moderate", "dense")[
            int(rng.choice(3, p=[0.18, 0.47, 0.35]))
        ]
        train_plan.append((mode, orientation))
    train_plan = [
        train_plan[int(index)]
        for index in rng.permutation(len(train_plan))
    ]

    test_plan = []
    for index in range(num_test):
        orientation = (
            first_orientation if index % 2 == 0 else other_orientation
        )
        mode = ("sparse", "two", "moderate", "dense", "aggregate")[
            int(rng.choice(5, p=[0.18, 0.16, 0.25, 0.25, 0.16]))
        ]
        test_plan.append((mode, orientation))
    test_plan = [
        test_plan[int(index)]
        for index in rng.permutation(len(test_plan))
    ]

    puzzle = {"train": [], "test": []}
    used_child_seeds = set()
    seen_inputs = set()
    for split, regime, plan in (
        ("train", "source_train", train_plan),
        ("test", "target_test", test_plan),
    ):
        for mode, majority in plan:
            for _ in range(100):
                child_rng = _next_child_rng(rng, used_child_seeds)
                try:
                    pair = _generate_pair(
                        child_rng, regime, parameters, majority, mode
                    )
                except RuntimeError:
                    continue
                key = _array_key(pair["input"])
                if key in seen_inputs or _is_source_input(pair["input"]):
                    continue
                seen_inputs.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("could not generate a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_profile(analysis):
    winner = analysis["majority"]
    winning = [
        len(component["cells"]) for component in analysis["components"]
        if component["orientation"] == winner
    ]
    losing = [
        len(component["cells"]) for component in analysis["components"]
        if component["orientation"] != winner
    ]
    return winning, losing


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs = set()
    seen_pairs = set()
    output_convention = None
    train_orientations = set()
    longest_witness = False
    count_witness = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert input_grid.shape == output_grid.shape
            input_key = _array_key(input_grid)
            pair_key = (input_key, _array_key(output_grid))
            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_input(input_grid)
            analysis, convention = _check_pair_transformation(
                input_grid, output_grid
            )
            assert 2 <= len(analysis["components"]) <= 9
            assert all(
                2 <= len(component["cells"]) <= 14
                for component in analysis["components"]
            )
            if output_convention is None:
                output_convention = convention[2:]
            else:
                assert output_convention == convention[2:]
            if split == "train":
                train_orientations.add(analysis["majority"])
                winning, losing = _pair_profile(analysis)
                longest_witness |= max(losing) > max(winning)
                count_witness |= len(winning) < len(losing)

    assert train_orientations == {"dr", "dl"}
    assert longest_witness
    assert count_witness
    assert output_convention is not None
    for pair in puzzle["test"]:
        predicted = _solve_input(
            pair["input"], output_convention[0], output_convention[1]
        )
        assert np.array_equal(predicted, pair["output"])


def _puzzles_equal(first, second):
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False
    return True


def self_test() -> None:
    explicit = generate_puzzle(314159, 7, 3)
    assert len(explicit["train"]) == 7
    assert len(explicit["test"]) == 3
    assert _puzzles_equal(explicit, generate_puzzle(314159, 7, 3))
    assert not _puzzles_equal(
        generate_puzzle(2718, 6, 2),
        generate_puzzle(2719, 6, 2),
    )
    for seed, train_count, test_count in (
        (8001, 2, 1), (8002, 10, 1), (8003, 10, 4),
        (8101, None, 1), (8102, None, 4),
        (8103, 2, None), (8104, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for request in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 2.5}, {"num_test": True},
    ):
        try:
            generate_puzzle(9000, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {request}")

    train_counts, test_counts = set(), set()
    shapes, palettes, profiles = set(), set(), set()
    output_roles = set()
    test_runs, margins = set(), set()
    compact = large = rectangular = long_run = False
    edge = corner = False
    variable_shapes = variable_inputs = 0
    stable_shapes = stable_inputs = 0
    fewer_two_vs_many = collective_test = 0
    sparse_tests = dense_tests = 0
    different_two_run_tests = three_or_more_tests = 0
    test_close = test_wide = 0
    dense_to_sparse = sparse_to_dense = similar_complexity = 0
    mixed_test_regimes = 0
    large_sparse_short = small_several = rectangular_long = 0
    both_stable = both_vary = 0
    train_four_in_family = test_multiple_each_slope = 0
    restored_lengths = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        pair_shapes, input_palettes = set(), set()
        train_max = max(
            len(_analyze_input(pair["input"])["components"])
            for pair in puzzle["train"]
        )
        train_min = min(
            len(_analyze_input(pair["input"])["components"])
            for pair in puzzle["train"]
        )
        test_run_counts = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                analysis, convention = _check_pair_transformation(
                    grid, pair["output"]
                )
                winning, losing = _pair_profile(analysis)
                profile = (
                    tuple(sorted(winning)), tuple(sorted(losing))
                )
                profiles.add(profile)
                shapes.add(grid.shape)
                palettes.add(convention)
                output_roles.add(convention[2:])
                pair_shapes.add(grid.shape)
                input_palettes.add(convention[:2])
                h, w = grid.shape
                compact |= h <= 10 and w <= 10
                large |= h >= 27 and w >= 27
                rectangular |= max(h, w) >= 2 * min(h, w)
                long_run |= any(
                    len(c["cells"]) >= 10
                    for c in analysis["components"]
                )
                large_sparse_short += (
                    h >= 24 and w >= 24
                    and len(analysis["components"]) <= 3
                    and max(winning + losing) <= 5
                )
                small_several += (
                    h <= 10 and w <= 10
                    and len(analysis["components"]) >= 3
                )
                rectangular_long += (
                    max(h, w) >= 2 * min(h, w)
                    and max(winning + losing) >= 8
                )
                margins.add(sum(winning) - sum(losing))
                restored_lengths += any(
                    8 <= length <= 10
                    for length in winning + losing
                )
                for component in analysis["components"]:
                    cells = component["cells"]
                    edge |= _touches_edge(cells, h, w)
                    corner |= _touches_corner(cells, h, w)
                if split == "train":
                    fewer_two_vs_many += (
                        len(winning) == 2 and len(losing) >= 3
                    )
                    train_four_in_family += max(
                        len(winning), len(losing)
                    ) >= 4
                else:
                    run_count = len(analysis["components"])
                    test_run_counts.append(run_count)
                    test_runs.add(run_count)
                    sparse_tests += run_count == 2 and sum(
                        winning + losing
                    ) == 5
                    dense_tests += run_count >= 5
                    different_two_run_tests += (
                        run_count == 2
                        and sorted(winning + losing) != [2, 3]
                    )
                    three_or_more_tests += run_count >= 3
                    test_close += sum(winning) - sum(losing) == 1
                    test_wide += sum(winning) - sum(losing) >= 5
                    collective_test += max(losing) > max(winning)
                    test_multiple_each_slope += (
                        len(winning) >= 2 and len(losing) >= 2
                    )
                    assert run_count >= 2
        variable_shapes += len(pair_shapes) > 1
        variable_inputs += len(input_palettes) > 1
        stable_shapes += len(pair_shapes) == 1
        stable_inputs += len(input_palettes) == 1
        both_stable += (
            len(pair_shapes) == 1 and len(input_palettes) == 1
        )
        both_vary += (
            len(pair_shapes) > 1 and len(input_palettes) > 1
        )
        dense_to_sparse += train_max >= 5 and min(test_run_counts) <= 2
        sparse_to_dense += train_min <= 3 and max(test_run_counts) >= 5
        similar_complexity += any(
            train_min <= count <= train_max
            for count in test_run_counts
        )
        mixed_test_regimes += len(set(test_run_counts)) >= 2

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(palettes) >= 50
    assert len(output_roles) >= 20
    assert len(profiles) >= 35 and len(test_runs) >= 4
    assert len(margins) >= 8
    assert compact and large and rectangular and long_run
    assert edge and corner
    assert variable_shapes >= 20 and variable_inputs >= 20
    assert stable_shapes >= 3 and stable_inputs >= 3
    assert both_stable >= 1 and both_vary >= 10
    assert fewer_two_vs_many >= 5
    assert sparse_tests >= 3 and dense_tests >= 5
    assert collective_test >= 3
    assert different_two_run_tests >= 3 and three_or_more_tests >= 15
    assert test_close >= 5 and test_wide >= 10
    assert dense_to_sparse >= 5 and sparse_to_dense >= 5
    assert similar_complexity >= 10 and mixed_test_regimes >= 5
    assert large_sparse_short >= 1 and small_several >= 5
    assert rectangular_long >= 2
    assert train_four_in_family >= 10
    assert test_multiple_each_slope >= 5
    assert restored_lengths >= 30
