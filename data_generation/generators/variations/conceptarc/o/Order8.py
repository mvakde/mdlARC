"""Vertical color-run ARC puzzle generator."""

import numpy as np


_SOURCE_PAIRS = {
    ("/".join(["00000000"] * 3 + ["00066000"] * 2 + ["00044000"] * 2
              + ["00033000"] * 2 + ["00022000"] * 2), "6432"),
    ("/".join(["00999900"] * 4 + ["00444400"] * 4
              + ["00777700"] * 3 + ["00333300"] * 3), "9473"),
    ("/".join(["000000"] * 2 + ["008800"] * 2 + ["007700"] * 2
              + ["003300"] * 2 + ["001100"] * 2 + ["005500"] * 2), "87315"),
    ("/".join(["00000000"] * 2 + [
        "00070000", "00777000", "00070000", "00010000", "00111000", "00010000",
        "00060000", "00666000", "00060000", "00040000", "00444000", "00040000",
    ]), "7164"),
    ("/".join(["000000"] * 5 + [
        "000200", "000400", "000300", "000800", "000200", "000400", "000900",
    ]), "2438249"),
}


def _grid_key(inp: np.ndarray, out: np.ndarray) -> tuple[str, str]:
    def encode(grid: np.ndarray) -> str:
        return "/".join("".join(str(int(v)) for v in row) for row in grid)
    return encode(inp), encode(out)


def _decode_input(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int]], list[tuple[int, int]]]:
    """Infer background; inspect connected monochrome rows and color runs."""
    uniform = [int(row[0]) for row in grid if np.all(row == row[0])]
    assert uniform and len(set(uniform)) == 1
    background = uniform[0]
    occupied = np.flatnonzero(np.any(grid != background, axis=1))
    assert occupied.size >= 3
    first, last = int(occupied[0]), int(occupied[-1])
    assert np.array_equal(occupied, np.arange(first, last + 1))
    colors, spans = [], []
    for row_index in occupied:
        row = grid[int(row_index)]
        columns = np.flatnonzero(row != background)
        assert 0 < columns.size < grid.shape[1]
        left, right = int(columns[0]), int(columns[-1]) + 1
        assert 0 < left and right < grid.shape[1]
        assert np.array_equal(columns, np.arange(left, right))
        unique = np.unique(row[columns])
        assert unique.size == 1 and int(unique[0]) != background
        if spans:
            assert max(left, spans[-1][0]) < min(right, spans[-1][1])
        spans.append((left, right))
        colors.append(int(unique[0]))
    stages, start, current = [], first, colors[0]
    for offset, color in enumerate(colors[1:], 1):
        if color != current:
            stages.append((current, start, first + offset))
            start, current = first + offset, color
    stages.append((current, start, last + 1))
    assert 3 <= len(stages) <= 8
    return background, stages, spans


def _solve(grid: np.ndarray) -> np.ndarray:
    return np.array([[color for color, _, _ in _decode_input(grid)[1]]], dtype=np.int8)


def _validate_count(value: int | None, name: str, low: int, high: int) -> None:
    if value is not None and (isinstance(value, (bool, np.bool_))
                              or not isinstance(value, (int, np.integer))
                              or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _stage_colors(rng: np.random.Generator, options: tuple[int, ...],
                  count: int, recurrence: bool) -> list[int]:
    repeats = int(rng.integers(1, min(3, count - 2) + 1)) if recurrence else 0
    repeat_positions = set(int(v) for v in rng.choice(
        np.arange(2, count), size=repeats, replace=False
    )) if repeats else set()
    colors, unused = [], list(options)
    for index in range(count):
        if index in repeat_positions:
            choices = [v for v in colors[:-1] if v != colors[-1]]
            colors.append(int(rng.choice(choices)))
        else:
            assert unused
            color = int(rng.choice(unused))
            unused.remove(color)
            colors.append(color)
    return colors


def _profiles(rng: np.random.Generator, mode: str, count: int,
              max_width: int, require_multirow: bool) -> list[tuple[list[int], int]]:
    """Return row widths and horizontal movement for each stage."""
    max_height = 3 if count >= 7 else 4
    profiles = []
    for _ in range(count):
        kind = mode if mode != "mixed" else str(rng.choice(
            ["band", "cross", "singleton", "taper", "step"]
        ))
        motion = 0
        if kind == "singleton":
            widths = [1]
        elif kind == "cross":
            choices = [[1, 3, 1]]
            if max_width >= 4:
                choices += [[1, 4, 1], [2, 4, 1]]
            if max_width >= 5:
                choices += [[1, 5, 2], [2, 5, 1]]
            widths = list(choices[int(rng.integers(len(choices)))])
        elif kind == "taper":
            height = int(rng.integers(2, min(max_height, max_width) + 1))
            width = int(rng.integers(height, max_width + 1))
            widths = list(range(width - height + 1, width + 1))
            if rng.random() < .5:
                widths.reverse()
        else:
            height = int(rng.integers(1 if kind == "band" else 2, max_height + 1))
            width = int(rng.integers(1 if kind == "band" else 2, max_width + 1))
            widths = [width] * height
            if kind == "step":
                motion = int(rng.choice([-1, 1]))
        profiles.append((widths, motion))
    if require_multirow and all(len(widths) == 1 for widths, _ in profiles):
        index = int(rng.integers(count))
        profiles[index] = ([int(rng.integers(2, max_width + 1))] * 2, 0)
    return profiles


def _place_span(rng: np.random.Generator, width: int, span_width: int,
                previous: tuple[int, int] | None, shift: int) -> tuple[int, int]:
    choices = [left for left in range(1, width - span_width)
               if previous is None or max(left, previous[0]) < min(left + span_width, previous[1])]
    assert choices
    if previous is None:
        left = int(rng.choice(choices))
    else:
        target = (previous[0] + previous[1] - 1) / 2 + shift
        scores = [abs(left + (span_width - 1) / 2 - target) for left in choices]
        best = [left for left, score in zip(choices, scores) if score == min(scores)]
        left = int(rng.choice(best))
    return left, left + span_width


def _make_pair(rng: np.random.Generator, background: int, options: tuple[int, ...],
               count: int, recurrence: bool, mode: str, force_top: bool = False,
               force_bottom: bool = False, require_multirow: bool = False,
               classic_bands: bool = False
               ) -> tuple[np.ndarray, np.ndarray]:
    colors = _stage_colors(rng, options, count, recurrence)
    width = int(rng.integers(5, 20))
    if classic_bands:
        band_width = int(rng.integers(2, min(6, width - 2) + 1))
        profiles = [([band_width] * int(rng.integers(2, 5)), 0)
                    for _ in range(count)]
    else:
        profiles = _profiles(rng, mode, count, min(6, width - 2), require_multirow)
    occupied_height = sum(len(profile) for profile, _ in profiles)
    assert occupied_height <= 24
    padding_budget = min(5, 30 - occupied_height)
    top = int(rng.integers(0, padding_budget + 1))
    bottom = int(rng.integers(0, padding_budget - top + 1))
    if force_top and top == 0:
        top = 1
    if force_bottom and bottom == 0:
        bottom = 1
    if top + bottom > 30 - occupied_height:
        top, bottom = 1, 1
    if top == bottom == 0:
        top, bottom = (1, 0) if rng.random() < .5 else (0, 1)
    grid = np.full((top + occupied_height + bottom, width), background, dtype=np.int8)
    drift = 0 if classic_bands else int(rng.choice([-2, -1, 0, 1, 2]))
    previous, row = None, top
    for color, (profile, motion) in zip(colors, profiles):
        for within_stage, span_width in enumerate(profile):
            shift = (drift if classic_bands else drift + int(rng.integers(-1, 2))) if within_stage == 0 else motion
            previous = _place_span(rng, width, span_width, previous, shift)
            grid[row, previous[0]:previous[1]] = color
            row += 1
    return grid, _solve(grid)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    _validate_count(num_train, "num_train", 2, 10)
    _validate_count(num_test, "num_test", 1, 4)
    if num_test is None:
        num_test = int(rng.integers(1, (4 if num_train is None else min(4, num_train - 1)) + 1))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    # Background is the sole puzzle-wide visual convention. Stage colors are
    # visible in each input; no palette lookup is needed for new test colors.
    source_like = bool(rng.random() < .14)
    background = 0 if source_like or rng.random() < .72 else int(rng.integers(1, 10))
    available = tuple(v for v in range(1, 10) if v != background)
    training_colors = tuple(int(v) for v in rng.choice(
        available, size=int(rng.integers(6, 8)), replace=False
    ))
    recurrence_demo = int(rng.integers(num_train))
    distinct_demo = int(rng.integers(num_train - 1))
    if distinct_demo >= recurrence_demo:
        distinct_demo += 1
    multirow_demo = int(rng.integers(num_train))
    top_demo = int(rng.integers(num_train))
    bottom_demo = int(rng.integers(num_train))

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for regime, pair_count in (("train", num_train), ("test", num_test)):
        for index in range(pair_count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for _ in range(32):
                if regime == "train":
                    count = int(pair_rng.integers(3, 7))
                    recurrence = index == recurrence_demo or (
                        index != distinct_demo and pair_rng.random() < .35
                    )
                    if index == distinct_demo:
                        recurrence = False
                    mode = str(pair_rng.choice(
                        ["band", "cross", "singleton", "mixed"], p=[.34, .17, .13, .36]
                    ))
                    if source_like:
                        mode = "band"
                    if index == multirow_demo and mode == "singleton":
                        mode = "mixed"
                    pair = _make_pair(pair_rng, background, training_colors, count,
                                      recurrence, mode, index == top_demo,
                                      index == bottom_demo, index == multirow_demo,
                                      source_like)
                else:
                    count = int(pair_rng.integers(3, 9))
                    recurrence = bool(pair_rng.random() < .48)
                    mode = str(pair_rng.choice(
                        ["band", "cross", "singleton", "mixed"], p=[.20, .22, .20, .38]
                    ))
                    if source_like and index == 0:
                        mode = str(pair_rng.choice(["cross", "singleton", "mixed"]))
                    pair = _make_pair(pair_rng, background, available, count,
                                      recurrence, mode)
                inp, out = pair
                key = (inp.shape, inp.tobytes())
                if key not in seen_inputs and _grid_key(inp, out) not in _SOURCE_PAIRS:
                    break
            else:
                raise RuntimeError("Could not construct a distinct pair")
            seen_inputs.add(key)
            puzzle[regime].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    backgrounds, seen_inputs, seen_pairs = set(), set(), set()
    train_sequences = []
    train_multirow = train_top = train_bottom = False
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            input_key = (inp.shape, inp.tobytes())
            pair_key = (input_key, out.shape, out.tobytes())
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _grid_key(inp, out) not in _SOURCE_PAIRS
            background, stages, _ = _decode_input(inp)
            backgrounds.add(background)
            assert out.shape == (1, len(stages))
            assert np.array_equal(out, np.array([[s[0] for s in stages]], dtype=int))
            assert background not in out
            if regime == "train":
                train_sequences.append(tuple(int(v) for v in out[0]))
                train_multirow |= any(end - start > 1 for _, start, end in stages)
                train_top |= bool(np.all(inp[0] == background))
                train_bottom |= bool(np.all(inp[-1] == background))
    assert len(backgrounds) == 1
    assert train_multirow and train_top and train_bottom
    assert any(len(set(seq)) == len(seq) for seq in train_sequences)
    assert any(len(set(seq)) < len(seq) for seq in train_sequences)


def _fingerprint(puzzle: dict) -> tuple:
    return tuple(tuple((pair["input"].shape, pair["input"].tobytes(),
                        pair["output"].shape, pair["output"].tobytes())
                       for pair in puzzle[regime]) for regime in ("train", "test"))


def _predict_from_demonstrations(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    """Use only returned training pairs and a test input, with no generator state."""
    backgrounds = set()
    for pair in train:
        full_rows = [int(row[0]) for row in pair["input"] if np.all(row == row[0])]
        assert full_rows and len(set(full_rows)) == 1
        backgrounds.add(full_rows[0])
    assert len(backgrounds) == 1
    background = backgrounds.pop()
    result = []
    for row in test_input:
        foreground = [int(v) for v in row if v != background]
        if foreground:
            assert len(set(foreground)) == 1
            if not result or foreground[0] != result[-1]:
                result.append(foreground[0])
    return np.array([result], dtype=int)


def self_test() -> None:
    fingerprints = set()
    train_counts, test_counts = set(), set()
    backgrounds, dimensions, structures = set(), set(), set()
    train_lengths, test_lengths = set(), set()
    train_profiles, test_profiles = set(), set()
    novel_tests = repeated_tests = distinct_tests = 0
    single_train = cross_train = variable_width_train = drifting_pairs = 0
    bottom_padded_tests = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = _fingerprint(puzzle)
        fingerprints.add(fingerprint)
        if seed < 8:
            assert fingerprint == _fingerprint(generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        training_palette = {int(v) for pair in puzzle["train"] for v in pair["output"].flat}
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                background, stages, spans = _decode_input(inp)
                backgrounds.add(background)
                dimensions.add(inp.shape)
                assert np.array_equal(out, _predict_from_demonstrations(puzzle["train"], inp))
                assert _grid_key(inp, out) not in _SOURCE_PAIRS
                first = stages[0][1]
                profile = tuple(tuple(spans[row - first][1] - spans[row - first][0]
                                      for row in range(start, end))
                                for _, start, end in stages)
                structures.add((inp.shape, profile, tuple(spans)))
                widths = {right - left for left, right in spans}
                if max(left for left, _ in spans) >= min(right for _, right in spans):
                    drifting_pairs += 1
                if regime == "train":
                    train_lengths.add(len(stages))
                    train_profiles.add(profile)
                    single_train += all(p == (1,) for p in profile)
                    cross_train += all(len(p) == 3 and p[1] > p[0] and p[1] > p[2]
                                       for p in profile)
                    variable_width_train += len(widths) > 1
                else:
                    test_lengths.add(len(stages))
                    test_profiles.add(profile)
                    colors = tuple(int(v) for v in out[0])
                    repeated_tests += len(set(colors)) < len(colors)
                    distinct_tests += len(set(colors)) == len(colors)
                    novel_tests += bool(set(colors) - training_palette)
                    bottom_padded_tests += bool(np.all(inp[-1] == background))
    assert len(fingerprints) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 4 and len(dimensions) >= 35
    assert len(structures) >= 150
    assert train_lengths == {3, 4, 5, 6}
    assert test_lengths == {3, 4, 5, 6, 7, 8}
    assert len(train_profiles) >= 80 and len(test_profiles) >= 50
    assert novel_tests >= 10 and repeated_tests >= 10 and distinct_tests >= 10
    assert single_train >= 3 and cross_train >= 3 and variable_width_train >= 20
    assert drifting_pairs >= 10 and bottom_padded_tests >= 10

    requests = [(2, 1), (10, 1), (10, 4), (None, 1),
                (None, 4), (2, None), (10, None)]
    for index, (train_count, test_count) in enumerate(requests):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(
            generate_puzzle(1000 + index, train_count, test_count))
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in [(1, None), (11, None), (None, 0),
                                    (None, 5), (2, 2), (3, 4), (4, 4)]:
        try:
            generate_puzzle(2000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal pair counts did not raise ValueError")
