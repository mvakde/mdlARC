import numpy as np


_SOURCE_INPUT_ROWS = {
    tuple(block.splitlines())
    for block in (
        """7777777777777777
7777777777777777
7777777777777777
7777777777777777
7777777777777777
7777737777777777
7777773777777777
7777777377777777
7777777737777777
7777777773777777
7777779777377777
7777777977737777
7777771797773777
7777717779777377
7777177777977737
7444444444497773""",
        """7777777777777777
7777777777777777
7777777777777777
7777771777777777
7777777177777777
7777777717777777
7777777771777777
7777777777177777
7777777787717777
7777777877771777
7777778777777177
7777787277777717
7777877727777771
7778777772777777
7787777777277777
7877777777727777""",
        """7777777777777777
7777777777777777
7774777777777777
7777477777777777
7777747777777777
7777774777777777
7777777477777777
7777777747777777
7777797774777777
7777779777477777
7777777977747777
7777771797774777
7777713779777477
7777177377977747
7771777737797774
7888888888888888""",
        """7777777777777777
7777777777777777
7777777777777777
7777777777777777
7777777777777777
7777777777777777
7777777777577777
7777777775777777
7777777757777777
7777777577777777
7777775717777777
7777754771777777
7777577477177777
7775777747717777
7757777774771777
7573333333333337""",
    )
}


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed_value


def _normalize_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(
    seed_value: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count(num_train, "num_train")
    test_count = _normalize_count(num_test, "num_test")

    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + seed_value % 4
        minimum_train = test_count + 1
        span = 11 - minimum_train
        train_count = minimum_train + (seed_value // 4) % span
    elif train_count is None:
        assert test_count is not None
        minimum_train = test_count + 1
        span = 11 - minimum_train
        train_count = minimum_train + (seed_value // 4) % span
    else:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed_value % maximum_test

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("the requested pair-count combination is impossible")

    return train_count, test_count


def _sample_puzzle_parameters(
    rng: np.random.Generator,
    train_count: int,
) -> dict:
    if rng.random() < 0.48:
        pool_size = 1
    else:
        pool_size = int(rng.integers(2, min(4, train_count) + 1))
    pool = rng.choice(np.arange(10), size=pool_size, replace=False)
    return {"background_pool": [int(value) for value in pool]}


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


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _analyze_input(grid: np.ndarray) -> dict:
    counts = np.bincount(grid.ravel().astype(np.int64, copy=False), minlength=10)
    background = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[background]) == 1
    colors = [c for c in range(10) if c != background and counts[c] > 0]
    assert 3 <= len(colors) <= 5
    assert int(counts[background]) > sum(int(counts[c]) for c in colors)
    traces = []
    for color in colors:
        coordinates = np.argwhere(grid == color)
        rows, columns = coordinates[:, 0], coordinates[:, 1]
        length = int(len(rows))
        if np.all(rows == rows[0]):
            assert length >= 3
            assert np.array_equal(np.sort(columns), np.arange(columns.min(), columns.max() + 1))
            direction = "h"
            start_row = int(rows[0])
            start_column = int(columns.min())
            end_row = start_row
        else:
            assert length >= 2
            order = np.argsort(rows)
            sorted_rows, sorted_columns = rows[order], columns[order]
            assert np.array_equal(sorted_rows, np.arange(sorted_rows[0], sorted_rows[0] + length))
            steps = np.diff(sorted_columns)
            if np.all(steps == 1):
                direction = "dr"
            elif np.all(steps == -1):
                direction = "dl"
            else:
                raise AssertionError("foreground color is not a straight diagonal")
            start_row = int(sorted_rows[0])
            start_column = int(sorted_columns[0])
            end_row = int(sorted_rows[-1])
        traces.append({"color": color, "start_row": start_row,
                       "start_column": start_column, "end_row": end_row,
                       "direction": direction, "length": length})
    traces.sort(key=lambda trace: trace["start_row"])
    starts = [trace["start_row"] for trace in traces]
    assert len(set(starts)) == len(starts)
    height = grid.shape[0]
    assert starts[0] <= max(6, height // 3)
    assert starts[-1] >= height // 2
    assert sum(trace["direction"] == "h" for trace in traces) <= 1
    return {"background": background, "traces": traces}


def _expected_output(analysis: dict) -> np.ndarray:
    output = np.full((5, 3), analysis["background"], dtype=np.int8)
    for offset, trace in enumerate(analysis["traces"]):
        output[5 - len(analysis["traces"]) + offset, :] = trace["color"]
    return output


def _directions(rng: np.random.Generator, count: int,
                first: str | None = None, horizontal_rank: int | None = None) -> list[str]:
    directions = ["dl" if rng.integers(2) == 0 else "dr" for _ in range(count)]
    if horizontal_rank is not None:
        directions[horizontal_rank] = "h"
    if first is not None:
        assert horizontal_rank != 0
        directions[0] = first
    return directions


def _sample_start_rows(rng: np.random.Generator, height: int,
                       directions: list[str], style: str) -> list[int]:
    count = len(directions)
    if style == "long":
        first_max = min(4, height - 10)
    else:
        first_max = min(6, height // 3)
    first = int(rng.integers(1, first_max + 1))
    if directions[-1] == "h":
        last = height - 1
    else:
        lowest = max(height // 2, height - 6, first + count - 1)
        last = int(rng.integers(lowest, height - 3))
    candidates = np.arange(first + 1, last)
    assert len(candidates) >= count - 2
    if count > 2:
        if style == "long":
            weights = np.exp(-(candidates - first) / 6.0)
            weights /= weights.sum()
            middle = rng.choice(candidates, size=count - 2, replace=False, p=weights)
        else:
            middle = rng.choice(candidates, size=count - 2, replace=False)
        rows = [first] + sorted(int(x) for x in middle) + [last]
    else:
        rows = [first, last]
    return rows


def _target_lengths(rng: np.random.Generator, height: int, width: int,
                    starts: list[int], directions: list[str], style: str) -> list[int]:
    lengths = []
    for rank, (row, direction) in enumerate(zip(starts, directions)):
        if direction == "h":
            if style == "long":
                low, high = 8, min(width, 21)
            else:
                low, high = 3, min(width, 9)
        else:
            maximum = min(width, height - row)
            if style == "long":
                low = 9 if rank == 0 else 4
                high = min(maximum, 16 if rank == 0 else 13)
            else:
                low = 2 if rng.random() < 0.12 else 3
                high = min(maximum, 7)
        low = min(low, high)
        lengths.append(int(rng.integers(low, high + 1)))
    return lengths


def _segment_candidates(rng: np.random.Generator, height: int, width: int,
                        start: int, direction: str, target: int) -> list[tuple]:
    minimum = 3 if direction == "h" else 2
    candidates = []
    for length in range(target, minimum - 1, -1):
        if direction == "h":
            first_columns = np.arange(width - length + 1)
        elif direction == "dr":
            first_columns = np.arange(width - length + 1)
        else:
            first_columns = np.arange(length - 1, width)
        for column in rng.permutation(first_columns):
            column = int(column)
            cells = tuple((start if direction == "h" else start + step,
                           column + (step if direction == "dr" else -step if direction == "dl" else step))
                          for step in range(length))
            candidates.append(cells)
    return candidates


def _place_traces(rng: np.random.Generator, height: int, width: int,
                  starts: list[int], directions: list[str], targets: list[int]) -> list[tuple]:
    choices = [_segment_candidates(rng, height, width, row, direction, target)
               for row, direction, target in zip(starts, directions, targets)]
    order = sorted(range(len(starts)), key=lambda i: (
        1 if directions[i] == "h" else 0,
        0 if i == 0 else 1,
        -targets[i], len(choices[i])))
    placed = [None] * len(starts)
    occupied = set()
    nodes = 0

    def backtrack(depth: int) -> bool:
        nonlocal nodes
        if depth == len(order):
            return True
        if nodes >= 30000:
            return False
        index = order[depth]
        for cells in choices[index]:
            nodes += 1
            if not occupied.isdisjoint(cells):
                continue
            placed[index] = cells
            occupied.update(cells)
            if backtrack(depth + 1):
                return True
            occupied.difference_update(cells)
        return False

    if backtrack(0):
        return placed
    # A guaranteed, short disjoint construction for any permitted start rows.
    diagonal_index = 0
    fallback = []
    for row, direction in zip(starts, directions):
        if direction == "h":
            fallback.append(tuple((row, col) for col in range(width - 3, width)))
        else:
            column = 2 * diagonal_index + (1 if direction == "dl" else 0)
            step = -1 if direction == "dl" else 1
            fallback.append(((row, column), (row + 1, column + step)))
            diagonal_index += 1
    assert len(set(cell for cells in fallback for cell in cells)) == sum(map(len, fallback))
    return fallback


def _sample_palette(rng: np.random.Generator, background: int,
                    count: int, forced_color: int | None = None) -> list[int]:
    available = [color for color in range(10) if color != background and color != forced_color]
    colors = [] if forced_color is None else [forced_color]
    colors.extend(int(x) for x in rng.choice(available, size=count - len(colors), replace=False))
    rng.shuffle(colors)
    return colors


def _generate_pair(rng: np.random.Generator, spec: dict) -> dict:
    style = spec["style"]
    if style == "long":
        if rng.random() < 0.24:
            height, width = 16, 16
        else:
            height = int(rng.integers(15, 27))
            width = int(rng.integers(14, 28))
    else:
        height = int(rng.integers(11, 25))
        width = int(rng.integers(11, 26))
    starts = _sample_start_rows(rng, height, spec["directions"], style)
    targets = _target_lengths(rng, height, width, starts, spec["directions"], style)
    segments = _place_traces(rng, height, width, starts, spec["directions"], targets)
    grid = np.full((height, width), spec["background"], dtype=np.int8)
    for color, cells in zip(spec["colors"], segments):
        for row, column in cells:
            grid[row, column] = color
    output = _expected_output(_analyze_input(grid))
    return {"input": grid, "output": output}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    parameters = _sample_puzzle_parameters(rng, train_count)
    backgrounds = parameters["background_pool"]
    source_shift = bool(rng.random() < 0.28)
    early_horizontal = not source_shift and bool(rng.random() < 0.62)

    train_counts = [3, 4] + ([5] if train_count >= 3 else [])
    train_counts += [int(rng.integers(3, 6)) for _ in range(train_count - len(train_counts))]
    rng.shuffle(train_counts)
    train_specs = []
    for index, count in enumerate(train_counts):
        background = backgrounds[index] if index < len(backgrounds) else int(rng.choice(backgrounds))
        if index == 0:
            horizontal = count - 1 if source_shift else (1 if early_horizontal else count - 1)
            directions = _directions(rng, count, "dr" if source_shift else "dl", horizontal)
            if source_shift:
                directions[1] = "dl"
            elif count >= 3:
                directions[2 if horizontal != 2 else count - 1] = "dr"
        elif index == 1:
            horizontal = count - 1 if rng.random() < 0.55 else None
            directions = _directions(rng, count, "dr", horizontal)
            if not source_shift:
                directions[1] = "dl" if directions[1] != "h" else "h"
        else:
            horizontal = None
            if rng.random() < 0.62:
                horizontal = int(rng.integers(0, count)) if early_horizontal else count - 1
            directions = _directions(rng, count,
                                     first="dr" if source_shift else None,
                                     horizontal_rank=horizontal)
        forced = None
        if len(backgrounds) > 1 and index < 2:
            forced = backgrounds[1 - index]
        train_specs.append({"count": count, "background": background,
                            "directions": directions, "style": "long" if index == 0 else
                            "short" if index == 1 else ("long" if rng.random() < 0.53 else "short"),
                            "colors": _sample_palette(rng, background, count, forced)})

    # Anchor pairs expose both diagonal directions and the horizontal vocabulary.
    observed_directions = {direction for spec in train_specs for direction in spec["directions"]}
    if "dl" not in observed_directions:
        train_specs[1]["directions"][1] = "dl"
    if "dr" not in observed_directions:
        train_specs[1]["directions"][1] = "dr"
    if "h" not in observed_directions:
        train_specs[1]["directions"][-1] = "h"
    early_ranks = [i for spec in train_specs for i, d in enumerate(spec["directions"][:-1]) if d == "h"]
    bottom_seen = any(spec["directions"][-1] == "h" for spec in train_specs)

    test_specs = []
    for index in range(test_count):
        count = 4 if source_shift and index == 0 else int(rng.choice(train_counts))
        if source_shift and index == 0:
            directions = ["dl", "dr", "dr", "h"]
        else:
            horizontal = None
            if rng.random() < 0.57:
                possible = []
                if bottom_seen:
                    possible.append(count - 1)
                possible += [rank for rank in early_ranks if rank < count - 1]
                if possible:
                    horizontal = int(rng.choice(possible))
            first = None if horizontal == 0 else ("dr" if index == 0 and not source_shift else None)
            directions = _directions(rng, count, first, horizontal)
        background = int(rng.choice(backgrounds))
        colors = _sample_palette(rng, background, count)
        if index == 0 and rng.random() < 0.52:
            compatible = [spec for spec in train_specs if spec["count"] == count]
            if compatible:
                exemplar = compatible[int(rng.integers(len(compatible)))]
                background = exemplar["background"]
                colors = list(exemplar["colors"])
                rng.shuffle(colors)
                if colors == exemplar["colors"]:
                    colors = colors[1:] + colors[:1]
        test_specs.append({"count": count, "background": background,
                           "directions": directions,
                           "style": "long" if (source_shift and index == 0) or rng.random() < 0.55 else "short",
                           "colors": colors})

    used_inputs = set()
    puzzle = {"train": [], "test": []}
    for section, specs in (("train", train_specs), ("test", test_specs)):
        for spec in specs:
            for _ in range(24):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair = _generate_pair(np.random.default_rng(pair_seed), spec)
                signature = _grid_signature(pair["input"])
                if signature not in used_inputs and _rows_signature(pair["input"]) not in _SOURCE_INPUT_ROWS:
                    used_inputs.add(signature)
                    puzzle[section].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a new input")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= size <= 30 for size in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_grid, output_grid = pair["input"], pair["output"]
            signature, pair_signature = _grid_signature(input_grid), _pair_signature(pair)
            assert signature not in seen_inputs and pair_signature not in seen_pairs
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
            assert _rows_signature(input_grid) not in _SOURCE_INPUT_ROWS
            assert output_grid.shape == (5, 3) and input_grid.shape != output_grid.shape
            analysis = _analyze_input(input_grid)
            assert np.array_equal(output_grid, _expected_output(analysis))
            analyses[section].append(analysis)

    train_counts = {len(a["traces"]) for a in analyses["train"]}
    assert {3, 4}.issubset(train_counts)
    if len(puzzle["train"]) >= 3:
        assert 5 in train_counts
    assert {len(a["traces"]) for a in analyses["test"]}.issubset(train_counts)
    train_backgrounds = {a["background"] for a in analyses["train"]}
    assert {a["background"] for a in analyses["test"]}.issubset(train_backgrounds)
    train_directions = {t["direction"] for a in analyses["train"] for t in a["traces"]}
    assert train_directions == {"dl", "dr", "h"}
    train_early_h = {i for a in analyses["train"] for i, t in enumerate(a["traces"][:-1])
                     if t["direction"] == "h"}
    train_bottom_h = any(a["traces"][-1]["direction"] == "h" for a in analyses["train"])
    for analysis in analyses["test"]:
        for rank, trace in enumerate(analysis["traces"]):
            if trace["direction"] == "h":
                assert (rank in train_early_h) if rank < len(analysis["traces"]) - 1 else train_bottom_h
    if len(train_backgrounds) > 1:
        assert any(a["background"] in {t["color"] for other in analyses["train"]
                    if other is not a for t in other["traces"]}
                   for a in analyses["train"])
    assert any(t["length"] >= 9 and t["direction"] != "h"
               for a in analyses["train"] for t in a["traces"])
    assert any(t["length"] <= 7 and t["direction"] != "h"
               for a in analyses["train"] for t in a["traces"])


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[section]) == len(second[section]) and
               all(np.array_equal(a[role], b[role]) for a, b in zip(first[section], second[section])
                   for role in ("input", "output")) for section in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    coverage = {
        "constant_background": 0, "variable_background": 0,
        "source_shift": 0, "palette_reuse": 0,
        "source_size": 0, "long_diagonal": 0, "long_horizontal": 0,
        "short_diagonal": 0, "overlap_two": 0, "overlap_three": 0,
        "variable_source_long": 0, "variable_early_test": 0,
        "test_five_no_horizontal": 0, "test_early_horizontal": 0,
        "later_longer": 0,
    }
    direction_orders = set()
    geometric_patterns = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_analyses = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        test_analyses = [_analyze_input(pair["input"]) for pair in puzzle["test"]]
        backgrounds = {a["background"] for a in train_analyses + test_analyses}
        variable = len(backgrounds) > 1
        coverage["variable_background" if variable else "constant_background"] += 1
        assert len({pair["input"].shape for section in ("train", "test")
                    for pair in puzzle[section]}) >= 2
        if all(a["traces"][0]["direction"] == "dr" for a in train_analyses) and any(
                len(a["traces"]) == 4 and a["traces"][0]["direction"] == "dl"
                and a["traces"][-1]["direction"] == "h" for a in test_analyses):
            coverage["source_shift"] += 1
        if any(frozenset(t["color"] for t in train_a["traces"]) ==
               frozenset(t["color"] for t in test_a["traces"]) and
               tuple(t["color"] for t in train_a["traces"]) !=
               tuple(t["color"] for t in test_a["traces"])
               for train_a in train_analyses for test_a in test_analyses):
            coverage["palette_reuse"] += 1
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                analysis = _analyze_input(grid)
                traces = analysis["traces"]
                direction_orders.add(tuple(t["direction"] for t in traces))
                geometric_patterns.add((grid.shape, tuple(
                    (t["start_row"], t["start_column"], t["length"], t["direction"])
                    for t in traces)))
                if grid.shape == (16, 16):
                    coverage["source_size"] += 1
                    if variable and any(t["direction"] != "h" and t["length"] >= 10
                                        for t in traces):
                        coverage["variable_source_long"] += 1
                if any(t["direction"] != "h" and t["length"] > traces[0]["length"]
                       for t in traces[1:]):
                    coverage["later_longer"] += 1
                overlap = max(sum(t["direction"] != "h" and
                                  t["start_row"] <= row <= t["end_row"]
                                  for t in traces) for row in range(grid.shape[0]))
                if overlap >= 2:
                    coverage["overlap_two"] += 1
                if overlap >= 3:
                    coverage["overlap_three"] += 1
                for trace in traces:
                    if trace["direction"] == "h" and trace["length"] >= 10:
                        coverage["long_horizontal"] += 1
                    if trace["direction"] != "h" and trace["length"] >= 10:
                        coverage["long_diagonal"] += 1
                    if trace["direction"] != "h" and trace["length"] == 2:
                        coverage["short_diagonal"] += 1
                if section == "test":
                    if len(traces) == 5 and all(t["direction"] != "h" for t in traces):
                        coverage["test_five_no_horizontal"] += 1
                    if any(t["direction"] == "h" and rank < len(traces) - 1
                           for rank, t in enumerate(traces)):
                        coverage["test_early_horizontal"] += 1
                        if variable:
                            coverage["variable_early_test"] += 1
                counts = np.bincount(grid.ravel(), minlength=10)
                background = int(np.argmax(counts))
                first_rows = [(int(np.argwhere(grid == color)[:, 0].min()), color)
                              for color in range(10) if color != background and counts[color] > 0]
                first_rows.sort()
                expected = np.full((5, 3), background, dtype=np.int8)
                for index, (_, color) in enumerate(first_rows):
                    expected[5 - len(first_rows) + index] = color
                assert np.array_equal(pair["output"], expected)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _puzzles_equal(generate_puzzle(123456), generate_puzzle(123456))
    assert _puzzles_equal(generate_puzzle(654321, 7, 3), generate_puzzle(654321, 7, 3))
    assert not _puzzles_equal(generate_puzzle(4001, 6, 2), generate_puzzle(4002, 6, 2))
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 6, 2),
                              (2001, None, 1), (2002, None, 4), (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                   {"num_train": 2.0, "num_test": 1}, {"num_train": 3, "num_test": 1.0}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    analyses = [(section, pair, _analyze_input(pair["input"])) for puzzle in puzzles
                for section in ("train", "test") for pair in puzzle[section]]
    assert len({_grid_signature(pair["input"]) for _, pair, _ in analyses}) >= int(0.95 * len(analyses))
    assert len({pair["input"].shape for _, pair, _ in analyses}) >= 45
    assert len({a["background"] for _, _, a in analyses}) >= 7
    assert len({tuple(t["direction"] for t in a["traces"]) for _, _, a in analyses}) >= 14
    assert any(pair["input"].shape == (16, 16) for _, pair, _ in analyses)
    assert any(t["length"] >= 10 and t["direction"] != "h" for _, _, a in analyses for t in a["traces"])
    assert any(t["direction"] == "h" and i < len(a["traces"]) - 1
               for _, _, a in analyses for i, t in enumerate(a["traces"]))
    assert {len(a["traces"]) for section, _, a in analyses if section == "test"} == {3, 4, 5}
    assert {a["traces"][0]["direction"] for section, _, a in analyses if section == "test"} >= {"dl", "dr"}
    assert {a["traces"][0]["direction"] for section, _, a in analyses if section == "train"} >= {"dl", "dr"}
    assert any(all(t["direction"] != "h" for t in a["traces"])
               for section, _, a in analyses if section == "test")
    assert any(len({a["background"] for a in [_analyze_input(pair["input"])
               for section in ("train", "test") for pair in puzzle[section]]}) > 1 for puzzle in puzzles)
    assert len(direction_orders) >= 40 and len(geometric_patterns) >= 200
    for key, minimum in {
        "constant_background": 10, "variable_background": 10,
        "source_shift": 5, "palette_reuse": 10, "source_size": 15,
        "long_diagonal": 80, "long_horizontal": 30, "short_diagonal": 1,
        "overlap_two": 100, "overlap_three": 30,
        "variable_source_long": 5, "variable_early_test": 2,
        "test_five_no_horizontal": 5, "test_early_horizontal": 5,
        "later_longer": 10,
    }.items():
        assert coverage[key] >= minimum, (key, coverage[key])
