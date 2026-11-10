import numpy as np


# Exact source input/output pairs have these identifying parameters.
_SOURCE_SIGNATURES = {
    (8, 4, 9, 3, 2, 1),
    (5, 7, 8, 2, 2, 1),
    (7, 2, 4, 3, 4, 1),
    (7, 5, 6, 3, 3, 1),
}


def _plain_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        if not _plain_int(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _plain_int(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _construct_output(input_grid, row, col, trace):
    result = input_grid.copy()
    marker = int(input_grid[row, col])
    result[:, col] = trace
    if 2 * col == input_grid.shape[1] - 1:
        result[0, col:] = trace
        result[-1, : col + 1] = trace
    else:
        result[0, : col + 1] = trace
        result[-1, col:] = trace
    result[row, col] = marker
    return result


def _make_pair(size, row, col, background, marker, trace):
    input_grid = np.full((size, size), background, dtype=np.int8)
    input_grid[row, col] = marker
    return {
        "input": input_grid,
        "output": _construct_output(input_grid, row, col, trace),
    }


def _middle_rows(size):
    return {(size - 1) // 2, size // 2}


def _pick_geometry(rng, sizes, branch, used, row_rule=None, near=False):
    """Choose without rejection; branch is 0 (center), -1, or +1."""
    for size_index in rng.permutation(len(sizes)):
        size = int(sizes[int(size_index)])
        if branch == 0 and size % 2 == 0:
            continue
        cols = [
            col for col in range(1, size - 1)
            if (0 if 2 * col == size - 1 else
                -1 if 2 * col < size - 1 else 1) == branch
        ]
        if near and cols:
            distance = min(abs(2 * col - (size - 1)) for col in cols)
            cols = [col for col in cols if abs(2 * col - (size - 1)) == distance]
        rows = [row for row in range(1, size - 1)
                if row_rule is None or row_rule(size, row)]
        candidates = [(size, row, col) for row in rows for col in cols
                      if (size, row, col) not in used]
        if candidates:
            chosen = candidates[int(rng.integers(len(candidates)))]
            used.add(chosen)
            return chosen
    raise RuntimeError("no unused geometry satisfies the plan")


def _grid_key(grid):
    return grid.shape, tuple(int(value) for value in grid.ravel())


def _pair_facts(pair):
    source = pair["input"]
    result = pair["output"]
    assert source.shape == result.shape
    size, width = source.shape
    assert size == width and size >= 5
    colors, counts = np.unique(source, return_counts=True)
    assert len(colors) == 2
    singleton = np.flatnonzero(counts == 1)
    assert len(singleton) == 1
    marker = int(colors[int(singleton[0])])
    background = int(colors[1 - int(singleton[0])])
    assert int(counts[1 - int(singleton[0])]) == size * size - 1
    position = np.argwhere(source == marker)
    assert position.shape == (1, 2)
    row, col = (int(value) for value in position[0])
    assert 0 < row < size - 1 and 0 < col < size - 1
    extra = set(int(v) for v in np.unique(result)) - {background, marker}
    assert len(extra) == 1
    trace = extra.pop()
    assert set(int(v) for v in np.unique(result)) == {background, marker, trace}
    centered = 2 * col == size - 1
    side = 0 if centered else (-1 if 2 * col < size - 1 else 1)

    # Rebuild the expected grid cell by cell, independently of construction.
    for r in range(size):
        for c in range(size):
            expected = background
            if c == col or (r == 0 and
                            ((centered and c >= col) or
                             (not centered and c <= col))) or (
                                r == size - 1 and
                                ((centered and c <= col) or
                                 (not centered and c >= col))):
                expected = trace
            if (r, c) == (row, col):
                expected = marker
            assert int(result[r, c]) == expected
    return {
        "size": size, "row": row, "col": col, "centered": centered,
        "side": side, "background": background, "marker": marker,
        "trace": trace,
    }


def _prediction_from_training(train, test_input):
    """Infer the color and both arm directions using training grids alone."""
    trace_colors = set()
    top_left_by_centered = {}
    for pair in train:
        source, result = pair["input"], pair["output"]
        colors, counts = np.unique(source, return_counts=True)
        marker = int(colors[int(np.flatnonzero(counts == 1)[0])])
        row, col = (int(v) for v in np.argwhere(source == marker)[0])
        centered = 2 * col == source.shape[1] - 1
        extra = set(int(v) for v in np.unique(result)) - set(
            int(v) for v in colors
        )
        assert len(extra) == 1
        trace = extra.pop()
        trace_colors.add(trace)
        top_left = int(result[0, 0]) == trace
        if centered in top_left_by_centered:
            assert top_left_by_centered[centered] == top_left
        top_left_by_centered[centered] = top_left
    assert len(trace_colors) == 1
    assert top_left_by_centered == {True: False, False: True}

    colors, counts = np.unique(test_input, return_counts=True)
    marker = int(colors[int(np.flatnonzero(counts == 1)[0])])
    row, col = (int(v) for v in np.argwhere(test_input == marker)[0])
    centered = 2 * col == test_input.shape[1] - 1
    trace = next(iter(trace_colors))
    answer = test_input.copy()
    answer[:, col] = trace
    if top_left_by_centered[centered]:
        answer[0, : col + 1] = trace
        answer[-1, col:] = trace
    else:
        answer[0, col:] = trace
        answer[-1, : col + 1] = trace
    answer[row, col] = marker
    return answer


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _plain_int(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # Trace is the only learned puzzle-level convention. Input colors are
    # independent pair-level roles, even when a palette is deliberately reused.
    trace = int(rng.integers(10))
    source_like = num_train >= 3 and int(rng.integers(4)) == 0
    if source_like:
        core_size = int(rng.choice(np.arange(5, 12, 2)))
    else:
        core_size = int(rng.choice(np.arange(5, 30, 2)))
    core_row = int(rng.integers(1, core_size - 1))
    first_side = -1 if int(rng.integers(2)) == 0 else 1
    displaced_cols = [col for col in range(1, core_size - 1)
                      if (2 * col < core_size - 1 if first_side < 0
                          else 2 * col > core_size - 1)]
    first_col = int(rng.choice(displaced_cols))

    # This same-size, same-row contrast makes size and vertical position
    # unable to explain the change in arm direction, including at (2, 1).
    train_records = [
        (core_size, core_row, core_size // 2),
        (core_size, core_row, first_col),
    ]
    used = set(train_records)

    if num_train >= 3:
        if source_like:
            extra_sizes = list(range(15, 30))
        else:
            mode = int(rng.integers(3))
            extra_sizes = (
                list(range(5, core_size)) if mode == 0 else
                [core_size] if mode == 1 else
                list(range(core_size + 1, 31))
            )
            if not extra_sizes:
                extra_sizes = list(range(5, 31))
        train_records.append(_pick_geometry(
            rng, extra_sizes, -first_side, used,
            row_rule=lambda size, row: row != core_row,
        ))

    if num_train >= 4:
        train_records.append(_pick_geometry(
            rng, list(range(5, 30, 2)), 0, used,
            row_rule=lambda size, row: row != core_row,
        ))

    while len(train_records) < num_train:
        branch = int(rng.choice([-1, 0, 1]))
        sizes = list(range(5, 30, 2)) if branch == 0 else list(range(5, 31))
        train_records.append(_pick_geometry(rng, sizes, branch, used))

    test_records = []
    for index in range(num_test):
        if source_like and index == 0:
            # Source-style recombination survives in some puzzles, while
            # the core training contrast removes its old size confound.
            size = train_records[2][0]
            if size % 2 == 0:
                size = min(29, size + 1)
            branch, sizes, near = 0, [size], False
        else:
            possible = [0, first_side] if num_train == 2 else [0, -1, 1]
            branch = int(rng.choice(possible))
            scale = int(rng.integers(4))
            if scale == 0:
                sizes = [core_size]
            elif scale == 1:
                sizes = list(range(5, 12))
            elif scale == 2:
                sizes = list(range(12, 22))
            else:
                sizes = list(range(22, 31))
            if branch == 0:
                sizes = [size for size in sizes if size % 2]
            near = branch != 0 and int(rng.integers(2)) == 0
        # With two training examples, keep the row value demonstrated in
        # both branches. Larger training sets demonstrate row variation.
        row_rule = (
            (lambda size, row: row == core_row)
            if num_train == 2 else None
        )
        valid_sizes = [size for size in sizes if
                       (branch != 0 or size % 2) and
                       (num_train != 2 or core_row < size - 1)]
        if not valid_sizes:
            valid_sizes = [size for size in range(5, 31)
                           if (branch != 0 or size % 2) and
                           (num_train != 2 or core_row < size - 1)]
        try:
            record = _pick_geometry(
                rng, valid_sizes, branch, used, row_rule=row_rule, near=near,
            )
        except RuntimeError:
            fallback_sizes = [size for size in range(5, 31)
                              if (branch != 0 or size % 2) and
                              (num_train != 2 or core_row < size - 1)]
            record = _pick_geometry(
                rng, fallback_sizes, branch, used, row_rule=row_rule,
            )
        test_records.append(record)

    rng.shuffle(train_records)
    rng.shuffle(test_records)

    palettes = [(bg, marker) for bg in range(10) for marker in range(10)
                if bg != marker and bg != trace and marker != trace]
    rng.shuffle(palettes)
    total = num_train + num_test
    preferred = palettes[:total]
    palette_mode = int(rng.integers(5))
    if palette_mode in (1, 3) and num_train >= 2:
        preferred[1] = preferred[0]
    if palette_mode in (2, 3):
        preferred[num_train] = preferred[0]
    if palette_mode == 4:
        preferred[num_train] = (preferred[0][1], preferred[0][0])

    seen_inputs = set()
    materialized = []
    for index, (size, row, col) in enumerate(train_records + test_records):
        for background, marker in [preferred[index]] + palettes:
            signature = (size, background, marker, row, col, trace)
            input_signature = (size, background, marker, row, col)
            if signature not in _SOURCE_SIGNATURES and input_signature not in seen_inputs:
                seen_inputs.add(input_signature)
                materialized.append(_make_pair(
                    size, row, col, background, marker, trace,
                ))
                break
        else:
            raise RuntimeError("no valid input palette")

    puzzle = {
        "train": materialized[:num_train],
        "test": materialized[num_train:],
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs, seen_pairs = set(), set()
    facts = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            item = _pair_facts(pair)
            assert (item["size"], item["background"], item["marker"],
                    item["row"], item["col"], item["trace"]) not in _SOURCE_SIGNATURES
            facts[split].append(item)

    all_facts = facts["train"] + facts["test"]
    assert len({item["trace"] for item in all_facts}) == 1
    training = facts["train"]
    assert any(item["centered"] for item in training)
    assert any(not item["centered"] for item in training)
    # A direct controlled contrast rules out both size and row as the arm
    # selector in every puzzle, even with the minimum number of examples.
    assert any(
        a["centered"] and not b["centered"] and
        a["size"] == b["size"] and a["row"] == b["row"]
        for a in training for b in training
    )
    sides = {item["side"] for item in training}
    if len(training) >= 3:
        assert {-1, 1} <= sides
        assert len({item["row"] for item in training}) >= 2
    assert all(item["side"] in sides for item in facts["test"])
    if len(training) == 2:
        assert all(item["row"] == training[0]["row"] for item in facts["test"])
    for pair in puzzle["test"]:
        assert np.array_equal(
            _prediction_from_training(puzzle["train"], pair["input"]),
            pair["output"],
        )


def _puzzle_key(puzzle):
    return tuple(
        (split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                      for p in puzzle[split]))
        for split in ("train", "test")
    )


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(999, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert len({_puzzle_key(p) for p in automatic}) == 42

    structures, layouts, sizes, traces = set(), set(), set(), set()
    test_branches, test_size_bands = set(), set()
    size_relations, palette_modes, offcenter_training = set(), set(), 0
    even_tests, near_even_tests, same_size_tests, new_size_tests = 0, 0, 0, 0
    source_like_tests = known_center_tests = inverted_tests = 0
    offcenter_center_training = offcenter_tests = 0
    even_middle_rows = set()
    for puzzle in automatic:
        validate_generated_puzzle(puzzle)
        train = [_pair_facts(pair) for pair in puzzle["train"]]
        test = [_pair_facts(pair) for pair in puzzle["test"]]
        all_facts = train + test
        structures.add(tuple(
            (split, tuple((f["size"], f["row"], f["col"])
                          for f in (train if split == "train" else test)))
            for split in ("train", "test")
        ))
        for f in all_facts:
            layouts.add((f["size"], f["row"], f["col"]))
            sizes.add(f["size"])
            traces.add(f["trace"])
        offcenter_training += any(
            f["row"] not in _middle_rows(f["size"]) for f in train
        )
        offcenter_center_training += any(
            f["centered"] and f["row"] not in _middle_rows(f["size"])
            for f in train
        )
        centered_sizes = [f["size"] for f in train if f["centered"]]
        displaced_sizes = [f["size"] for f in train if not f["centered"]]
        size_relations.update(
            (-1 if c < d else 0 if c == d else 1)
            for c in centered_sizes for d in displaced_sizes
        )
        train_palettes = {(f["background"], f["marker"]) for f in train}
        if len(train_palettes) < len(train):
            palette_modes.add("within_train")
        if any((f["background"], f["marker"]) in train_palettes for f in test):
            palette_modes.add("train_test")
        if any((f["marker"], f["background"]) in train_palettes for f in test):
            palette_modes.add("swapped")
        if all((f["background"], f["marker"]) not in train_palettes for f in test):
            palette_modes.add("novel")
        trained_sizes = {f["size"] for f in train}
        for f in test:
            test_branches.add(f["side"])
            test_size_bands.add("small" if f["size"] <= 11 else
                                "medium" if f["size"] <= 21 else "large")
            even_tests += f["size"] % 2 == 0
            near_even_tests += (
                f["size"] % 2 == 0 and abs(2 * f["col"] - (f["size"] - 1)) == 1
            )
            same_size_tests += f["size"] in trained_sizes
            new_size_tests += f["size"] not in trained_sizes
            offcenter_tests += f["row"] not in _middle_rows(f["size"])
            known_center_tests += f["centered"] and f["size"] in centered_sizes
            source_like_tests += (
                f["centered"] and f["size"] > min(centered_sizes)
                and f["size"] in displaced_sizes
                and f["size"] not in centered_sizes
            )
            inverted_tests += not f["centered"] and f["size"] in centered_sizes
        for f in all_facts:
            if f["size"] % 2 == 0 and f["row"] in _middle_rows(f["size"]):
                even_middle_rows.add(
                    "upper" if f["row"] == (f["size"] - 1) // 2 else "lower"
                )
    assert len(structures) >= 38 and len(layouts) >= 80
    assert len(sizes) >= 20 and len(traces) >= 7
    assert test_branches == {-1, 0, 1}
    assert test_size_bands == {"small", "medium", "large"}
    assert size_relations == {-1, 0, 1}
    assert palette_modes == {"within_train", "train_test", "swapped", "novel"}
    assert offcenter_training >= 25
    assert min(even_tests, near_even_tests, same_size_tests, new_size_tests) > 0
    assert min(source_like_tests, known_center_tests, inverted_tests) > 0
    assert offcenter_center_training > 0 and offcenter_tests > 0
    assert even_middle_rows == {"upper", "lower"}

    for index, (train_count, test_count) in enumerate(
        [(2, 1), (10, 1), (10, 4), (6, 3)]
    ):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (
            train_count, test_count
        )
        validate_generated_puzzle(puzzle)
    for index, (train_count, test_count) in enumerate(
        [(None, 1), (None, 4), (2, None), (10, None)]
    ):
        puzzle = generate_puzzle(2000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for seed, counts in [(314159, (7, 3)), (271828, (None, None))]:
        assert _puzzle_key(generate_puzzle(seed, *counts)) == _puzzle_key(
            generate_puzzle(seed, *counts)
        )
    assert len({_puzzle_key(generate_puzzle(seed, 6, 2))
                for seed in range(20, 32)}) >= 11
    _expect_value_error(num_train=1)
    _expect_value_error(num_train=11)
    _expect_value_error(num_test=0)
    _expect_value_error(num_test=5)
    _expect_value_error(num_train=2, num_test=2)
    _expect_value_error(num_train=3, num_test=3)
    _expect_value_error(num_train=3, num_test=4)
