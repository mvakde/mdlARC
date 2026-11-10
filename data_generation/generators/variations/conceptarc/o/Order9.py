"""Synthetic ARC puzzles that sort full-width bands by decreasing length."""

import numpy as np


# Exact input profiles of the five original pairs. Their output is determined
# by the sorting rule, so excluding these inputs excludes exact source pairs.
_SOURCE_PROFILES = {
    (3, (1, 4, 2, 5)),
    (2, (3, 5, 1, 4)),
    (5, (4, 2, 1, 3)),
    (7, (5, 3, 1)),
    (2, (3, 1, 4, 5)),
}


def _grid(
    prefix: int,
    lengths: tuple[int, ...] | list[int],
    breadth: int,
    background: int,
    foreground: int,
    axis: str,
) -> np.ndarray:
    line = [background] * prefix
    for index, length in enumerate(lengths):
        line.extend([foreground] * int(length))
        if index + 1 < len(lengths):
            line.append(background)
    array = np.repeat(np.asarray(line, dtype=np.int8)[:, None], breadth, axis=1)
    return array if axis == "row" else array.T.copy()


def _parse(line: list[int], background: int, foreground: int):
    position = 0
    while position < len(line) and line[position] == background:
        position += 1
    prefix = position
    lengths = []
    while position < len(line):
        start = position
        while position < len(line) and line[position] == foreground:
            position += 1
        if position == start:
            return None
        lengths.append(position - start)
        if position == len(line):
            break
        if line[position] != background:
            return None
        position += 1
        if position == len(line) or line[position] != foreground:
            return None
    if len(lengths) < 2 or sum(length >= 2 for length in lengths) < 2:
        return None
    return prefix, tuple(lengths)


def _decode(grid: np.ndarray):
    """Infer the axis and both color roles from the visible input alone."""
    row_bands = bool(np.all(grid == grid[:, :1]))
    column_bands = bool(np.all(grid == grid[:1, :]))
    assert row_bands != column_bands
    axis = "row" if row_bands else "column"
    line = (grid[:, 0] if row_bands else grid[0, :]).tolist()
    colors = {int(value) for value in np.unique(grid)}
    assert len(colors) == 2
    interpretations = []
    for background in colors:
        foreground = next(iter(colors - {background}))
        parsed = _parse(line, background, foreground)
        if parsed is not None:
            interpretations.append((background, foreground, *parsed))
    assert len(interpretations) == 1
    background, foreground, prefix, lengths = interpretations[0]
    return axis, background, foreground, prefix, lengths


def _choose_lengths(
    rng: np.random.Generator,
    count: int,
    prefix: int,
    minimum_unique: int,
) -> tuple[int, ...]:
    # Most bands have at least two cells along the sorting axis. A smaller
    # fraction have singleton bands; gamma weights also admit dominant bands.
    base = 2 if prefix == 0 or rng.random() < 0.74 else 1
    # Distinct runs should be common, while ties remain a regular special
    # case. Reserve enough room before sampling a fully distinct profile.
    desired_unique = count if rng.random() < 0.65 else minimum_unique
    required_extra = max(4, desired_unique * (desired_unique - 1) // 2)
    minimum = max(8, prefix + count * base + count - 1 + required_extra)
    if minimum > 30:
        desired_unique = minimum_unique
        required_extra = max(4, desired_unique * (desired_unique - 1) // 2)
        minimum = max(8, prefix + count * base + count - 1 + required_extra)
    if rng.random() < 0.3:
        axis_length = int(rng.integers(minimum, min(30, minimum + 5) + 1))
    else:
        axis_length = int(rng.integers(minimum, 31))
    extra = axis_length - prefix - (count - 1) - count * base
    for _ in range(80):
        concentration = 0.55 if rng.random() < 0.22 else 2.7
        weights = rng.dirichlet(np.full(count, concentration))
        lengths = tuple((base + rng.multinomial(extra, weights)).tolist())
        if (
            len(set(lengths)) >= desired_unique
            and sum(length >= 2 for length in lengths) >= 2
        ):
            return lengths
    # A bounded fallback is useful for very short canvases and rare RNG tails.
    values = [base] * count
    if base == 1 and desired_unique == 2:
        values[:2] = [2, 3]
        used = 3
    else:
        for index in range(desired_unique):
            values[index] += index
        used = desired_unique * (desired_unique - 1) // 2
    values[desired_unique - 1] += extra - used
    assert len(set(values)) >= desired_unique
    assert sum(value >= 2 for value in values) >= 2
    return tuple(values)


def _ordered(
    rng: np.random.Generator, lengths: tuple[int, ...], identity: bool
) -> tuple[int, ...]:
    descending = tuple(sorted(lengths, reverse=True))
    if identity:
        return descending
    if rng.random() < 0.28:
        order = list(descending)
        choices = [i for i in range(len(order) - 1) if order[i] != order[i + 1]]
        index = int(rng.choice(choices))
        order[index], order[index + 1] = order[index + 1], order[index]
        return tuple(order)
    for _ in range(30):
        order = tuple(int(value) for value in rng.permutation(lengths))
        if order != descending:
            return order
    return descending[1:] + descending[:1]


def _make_pair(
    rng: np.random.Generator,
    axis: str,
    count: int,
    demonstration: bool,
) -> dict:
    prefix = int(rng.integers(1 if demonstration else 0, 8))
    lengths = _choose_lengths(rng, count, prefix, 3 if demonstration else 2)
    identity = False if demonstration else rng.random() < 0.18
    order = _ordered(rng, lengths, identity)
    breadth = int(rng.integers(1, 13))
    # Color roles are visible from the run structure, so palettes can vary
    # between pairs. Zero is permitted in either role.
    background = 0 if rng.random() < 0.4 else int(rng.integers(1, 10))
    if background != 0 and rng.random() < 0.22:
        foreground = 0
    else:
        colors = [color for color in range(10) if color != background]
        foreground = int(rng.choice(colors))
    return {
        "input": _grid(prefix, order, breadth, background, foreground, axis),
        "output": _grid(
            prefix, sorted(order, reverse=True), breadth,
            background, foreground, axis,
        ),
    }


def _key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(value) for value in grid.flat)


def _source_copy(pair: dict, decoded: tuple) -> bool:
    axis, background, foreground, prefix, lengths = decoded
    return (
        axis == "row"
        and pair["input"].shape == (18, 3)
        and background == 0
        and foreground == 3
        and (prefix, lengths) in _SOURCE_PROFILES
    )


def _validate_count(name: str, value: int | None, low: int, high: int) -> None:
    if value is not None and (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    choices = [
        (train, test)
        for train in range(2, 11)
        for test in range(1, 5)
        if train > test
        and (num_train is None or train == num_train)
        and (num_test is None or test == num_test)
    ]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    # Give small and large demonstration sets comparable opportunities even
    # though large train counts admit more legal train/test combinations.
    train_options = sorted({train for train, _ in choices})
    train_count = int(rng.choice(train_options))
    test_options = [test for train, test in choices if train == train_count]
    test_count = int(rng.choice(test_options))

    # A puzzle either stays with horizontal bands or teaches both axes in its
    # first two demonstrations. All tests use an axis seen in training.
    mixed_axes = bool(rng.random() < 0.42)
    introduction = ["row", "column"] if mixed_axes else ["row", "row"]
    if mixed_axes and rng.random() < 0.5:
        introduction.reverse()

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_shapes = set()
    seen_targets = set()
    first_palette = None
    first_count = None
    for split, amount in (("train", train_count), ("test", test_count)):
        for index in range(amount):
            demonstration = split == "train" and index < 2
            for _ in range(300):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                axis = (
                    introduction[index]
                    if demonstration else
                    ("column" if mixed_axes and pair_rng.random() < 0.5 else "row")
                )
                count_options = (3, 4, 5) if demonstration else (2, 3, 4, 5)
                count = int(pair_rng.choice(count_options))
                if demonstration and index == 1 and count == first_count:
                    continue
                pair = _make_pair(pair_rng, axis, count, demonstration)
                decoded = _decode(pair["input"])
                actual_axis, background, foreground, prefix, lengths = decoded
                palette = (background, foreground)
                if demonstration and index == 1 and palette == first_palette:
                    continue
                input_key = _key(pair["input"])
                shape = pair["input"].shape
                target = (actual_axis, prefix, tuple(sorted(lengths, reverse=True)))
                if (
                    input_key in seen_inputs
                    or shape in seen_shapes
                    or target in seen_targets
                    or _source_copy(pair, decoded)
                ):
                    continue
                seen_inputs.add(input_key)
                seen_shapes.add(shape)
                seen_targets.add(target)
                puzzle[split].append(pair)
                if demonstration and index == 0:
                    first_palette, first_count = palette, count
                break
            else:
                raise RuntimeError("could not construct a distinct band pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs = set()
    seen_pairs = set()
    seen_shapes = set()
    seen_targets = set()
    training_axes = set()
    introductions = []
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            input_grid, output_grid = pair["input"], pair["output"]
            assert input_grid.shape == output_grid.shape
            axis, background, foreground, prefix, lengths = _decode(input_grid)
            assert {
                int(color) for color in np.unique(output_grid)
            } == {background, foreground}
            breadth = input_grid.shape[1] if axis == "row" else input_grid.shape[0]
            expected = _grid(
                prefix, sorted(lengths, reverse=True), breadth,
                background, foreground, axis,
            )
            assert np.array_equal(output_grid, expected)
            assert not _source_copy(pair, (axis, background, foreground, prefix, lengths))
            input_key = _key(input_grid)
            pair_key = (input_key, _key(output_grid))
            target = (axis, prefix, tuple(sorted(lengths, reverse=True)))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert input_grid.shape not in seen_shapes and target not in seen_targets
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            seen_shapes.add(input_grid.shape)
            seen_targets.add(target)
            if split == "train":
                training_axes.add(axis)
                if index < 2:
                    assert prefix >= 1 and len(set(lengths)) >= 3
                    assert not np.array_equal(input_grid, output_grid)
                    introductions.append((axis, len(lengths), (background, foreground)))
            else:
                assert axis in training_axes
    assert introductions[0][1] != introductions[1][1]
    assert introductions[0][2] != introductions[1][2]
    assert training_axes in ({"row"}, {"row", "column"})
    if "column" in training_axes:
        assert {item[0] for item in introductions} == {"row", "column"}


def _same_puzzle(left: dict, right: dict) -> bool:
    return all(
        len(left[split]) == len(right[split])
        and all(
            np.array_equal(a[role], b[role]) and a[role].dtype == b[role].dtype
            for a, b in zip(left[split], right[split])
            for role in ("input", "output")
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts, puzzles = set(), set(), set()
    shapes, palettes, profiles, axes = set(), set(), set(), set()
    run_counts, prefixes = set(), set()
    singleton = no_singleton = equal_runs = dominant = close = 0
    train_identity = test_identity = foreground_zero = mixed = 0
    positions = {3: {"min": set(), "max": set()}, 4: {"min": set(), "max": set()}}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_key = []
        local_axes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                axis, bg, fg, prefix, lengths = _decode(inp)
                # Recompute the semantic answer from observed run lengths.
                ordered = tuple(sorted(lengths, reverse=True))
                breadth = inp.shape[1] if axis == "row" else inp.shape[0]
                assert np.array_equal(out, _grid(prefix, ordered, breadth, bg, fg, axis))
                assert _key(inp) != _key(out) or lengths == ordered
                assert not _source_copy(pair, (axis, bg, fg, prefix, lengths))
                shapes.add(inp.shape)
                palettes.add((bg, fg))
                profiles.add((axis, prefix, lengths))
                axes.add(axis)
                local_axes.add(axis)
                run_counts.add(len(lengths))
                prefixes.add(prefix)
                singleton += 1 in lengths
                no_singleton += min(lengths) >= 2
                equal_runs += len(set(lengths)) < len(lengths)
                largest = sorted(lengths, reverse=True)
                dominant += largest[0] >= 2 * largest[1]
                close += max(lengths) - min(lengths) <= 2
                foreground_zero += fg == 0
                if len(lengths) in positions and len(set(lengths)) == len(lengths):
                    positions[len(lengths)]["min"].add(lengths.index(min(lengths)))
                    positions[len(lengths)]["max"].add(lengths.index(max(lengths)))
                if lengths == ordered:
                    if split == "train":
                        train_identity += 1
                    else:
                        test_identity += 1
                puzzle_key.append((_key(inp), _key(out)))
        mixed += len(local_axes) == 2
        puzzles.add(tuple(puzzle_key))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzles) >= 40 and len(shapes) >= 80
    assert len(palettes) >= 30 and len(profiles) >= 100
    assert axes == {"row", "column"} and mixed >= 8
    assert run_counts == {2, 3, 4, 5} and set(range(8)).issubset(prefixes)
    assert min(singleton, no_singleton, equal_runs, dominant, close) > 0
    assert train_identity > 0 and test_identity > 0 and foreground_zero > 0
    for count in (3, 4):
        assert positions[count]["min"] == set(range(count))
        assert positions[count]["max"] == set(range(count))

    cases = [(2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)]
    for index, (train, test) in enumerate(cases):
        puzzle = generate_puzzle(1000 + index, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        assert _same_puzzle(puzzle, generate_puzzle(1000 + index, train, test))
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (2, 4)):
        try:
            generate_puzzle(2000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
