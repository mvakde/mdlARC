"""Reflected one-cell path puzzles with sparse additions and overwrites."""

import hashlib
import numpy as np


SOURCE_INPUT_DIGESTS = {
    "42d6771518ab23d3686a4a1dd3f6073a0839e92bef946f33388cd271562fb75a",
    "9c03b59227b933510a5d9120d3f5eaa7cc3478828ed60dbbe53c47419a6d6255",
    "a932bbd25d700e73596aa0b9a252c86bf9defa09c595e0e2cb6af41dd2d1edf2",
    "42b781cccbb149798057befb66ebf28da73e064fc8dbb7a9d74157be869f538e",
    "c67e2842885b05c9a9c003a51ec4c601e97c6904dd1215fc92323dd2b9cac8a3",
}


def _resolve_counts(seed, rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _positions(length, width, start, span):
    steps = np.arange(length)
    phase = steps % (2 * span)
    distance = np.minimum(phase, 2 * span - phase)
    return start + (1 if start == 0 else -1) * distance


def _clean_grid(length, width, horizontal, background, foreground, start, span):
    shape = (width, length) if horizontal else (length, width)
    grid = np.full(shape, background, dtype=np.int8)
    positions = _positions(length, width, start, span)
    steps = np.arange(length)
    if horizontal:
        grid[positions, steps] = foreground
    else:
        grid[steps, positions] = foreground
    return grid


def _recover(input_grid, background):
    horizontal = input_grid.shape[1] > input_grid.shape[0]
    length = max(input_grid.shape)
    width = min(input_grid.shape)
    foregrounds = [int(x) for x in np.unique(input_grid) if x != background]
    candidates = []
    for foreground in foregrounds:
        for start in (0, width - 1):
            for span in range(2, width):
                candidate = _clean_grid(
                    length, width, horizontal, background, foreground, start, span
                )
                cost = int(np.count_nonzero(input_grid != candidate))
                candidates.append((cost, foreground, start, span, candidate))
    candidates.sort(key=lambda item: item[:4])
    if not candidates:
        return None, 0
    gap = candidates[1][0] - candidates[0][0] if len(candidates) > 1 else length
    return candidates[0][4], gap


def _point(horizontal, step, transverse):
    return (transverse, step) if horizontal else (step, transverse)


def _choose_extra_steps(rng, length, width, start, span, positions, count, style):
    """Choose distractors on the background, including repeated margin phases."""
    choices = []
    used = set()
    if style == "margin" and span < width - 1 and count >= 2:
        margin = width - 1 if start == 0 else 0
        phase = int(rng.integers(0, 2 * span))
        repeated = list(range(phase, length, 2 * span))
        if len(repeated) < 2:
            repeated = list(range(0, length, 2 * span))
        for step in repeated[:min(3, count)]:
            used.add((step, margin))
            choices.append((step, margin))
    for _ in range(count - len(choices)):
        local_style = str(rng.choice(
            ("random", "neighbor", "turn", "distant")
            if style == "mixed" else (style,)
        ))
        eligible = []
        for step in range(length):
            for transverse in range(width):
                if transverse == positions[step] or (step, transverse) in used:
                    continue
                distance = abs(transverse - positions[step])
                if local_style == "neighbor" and distance != 1:
                    continue
                if local_style == "distant" and distance < 2:
                    continue
                if local_style == "turn" and not (
                    step % span in (0, 1, span - 1) and distance == 1
                ):
                    continue
                if local_style == "margin" and not (
                    span < width - 1
                    and transverse == (width - 1 if start == 0 else 0)
                ):
                    continue
                eligible.append((step, transverse))
        if not eligible:
            eligible = [
                (step, transverse)
                for step in range(length)
                for transverse in range(width)
                if transverse != positions[step] and (step, transverse) not in used
            ]
        chosen = eligible[int(rng.integers(len(eligible)))]
        used.add(chosen)
        choices.append(chosen)
    return choices


def _input_digest(grid):
    canonical = np.asarray(grid, dtype=np.int8)
    return hashlib.sha256(
        str(canonical.shape).encode() + canonical.tobytes()
    ).hexdigest()


def _pair_signature(pair):
    return (
        pair["input"].shape, pair["input"].tobytes(),
        pair["output"].shape, pair["output"].tobytes(),
    )


def _make_pair(rng, background, foreground, noise_color, extra_mode,
               missing_mode, seen_inputs, forced_horizontal):
    for attempt in range(160):
        width = int(rng.integers(4, 10))
        span = int(rng.integers(2, width))
        length = int(rng.integers(max(12, 2 * span + 3), 31))
        horizontal = (bool(rng.integers(0, 2)) if forced_horizontal is None
                      else forced_horizontal)
        start = 0 if rng.integers(0, 2) == 0 else width - 1
        output = _clean_grid(
            length, width, horizontal, background, foreground, start, span
        )
        input_grid = output.copy()
        positions = _positions(length, width, start, span)
        if missing_mode == "none":
            erased_count = overwritten_count = 0
        elif missing_mode == "erase":
            erased_count = int(rng.integers(1, 4))
            overwritten_count = 0
        elif missing_mode == "overwrite":
            erased_count = 0
            overwritten_count = int(rng.integers(1, 4))
        else:
            erased_count = int(rng.integers(1, 3))
            overwritten_count = int(rng.integers(1, 3))
        missing_steps = rng.choice(
            length, size=erased_count + overwritten_count, replace=False
        )
        for step in missing_steps[:erased_count]:
            input_grid[_point(horizontal, int(step), int(positions[step]))] = background
        for step in missing_steps[erased_count:]:
            input_grid[_point(horizontal, int(step), int(positions[step]))] = noise_color

        extra_count = int(rng.integers(2, 11))
        if attempt >= 100:
            extra_count = min(extra_count, 5)
        style = str(rng.choice(("random", "neighbor", "turn", "distant", "margin", "mixed")))
        extras = _choose_extra_steps(
            rng, length, width, start, span, positions, extra_count, style
        )
        if extra_mode == "same":
            colors = [foreground] * extra_count
        elif extra_mode == "distinct":
            colors = [noise_color] * extra_count
        else:
            colors = [foreground, noise_color] + [
                foreground if rng.integers(0, 2) else noise_color
                for _ in range(extra_count - 2)
            ]
            rng.shuffle(colors)
        for (step, transverse), color in zip(extras, colors):
            input_grid[_point(horizontal, step, transverse)] = color
        signature = (input_grid.shape, input_grid.tobytes())
        if signature in seen_inputs or _input_digest(input_grid) in SOURCE_INPUT_DIGESTS:
            continue
        recovered, gap = _recover(input_grid, background)
        if gap >= 2 and np.array_equal(recovered, output):
            seen_inputs.add(signature)
            return {"input": input_grid, "output": output}
    raise RuntimeError("Could not construct an unambiguous reflected path")


def _sample_background(rng):
    return 0 if rng.random() < 0.45 else int(rng.integers(1, 10))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(seed, rng, num_train, num_test)
    fixed_background = bool(rng.integers(0, 2))
    common_background = _sample_background(rng)
    extra_plans = ["same", "distinct"]
    missing_plans = ["none", str(rng.choice(("erase", "overwrite", "mixed")))]
    rng.shuffle(extra_plans)
    rng.shuffle(missing_plans)
    plans = list(zip(extra_plans, missing_plans))
    for _ in range(num_train - 2):
        plans.append((
            str(rng.choice(("same", "distinct", "mixed"))),
            str(rng.choice(("none", "erase", "overwrite", "mixed"))),
        ))
    for _ in range(num_test):
        plans.append((
            str(rng.choice(("same", "distinct", "mixed"))),
            str(rng.choice(("none", "erase", "overwrite", "mixed"))),
        ))

    orientation_regime = int(rng.integers(0, 6))
    if orientation_regime == 0:
        orientations = [True] * (num_train + num_test)
    elif orientation_regime == 1:
        orientations = [False] * (num_train + num_test)
    elif orientation_regime == 2:
        orientations = [True] * num_train + [False] * num_test
    elif orientation_regime == 3:
        orientations = [False] * num_train + [True] * num_test
    elif orientation_regime == 4:
        first = bool(rng.integers(0, 2))
        orientations = [first, not first] + [
            bool(rng.integers(0, 2)) for _ in range(num_train + num_test - 2)
        ]
    else:
        orientations = [None] * (num_train + num_test)

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    previous_foreground = None
    previous_noise = None
    first_background = None
    first_foreground = None
    for index, (extra_mode, missing_mode) in enumerate(plans):
        if fixed_background:
            background = common_background
        elif index == 1:
            options = [color for color in range(10) if color != first_background]
            background = int(rng.choice(options))
        else:
            background = _sample_background(rng)
        if index == 0:
            first_background = background
        options = [color for color in range(10) if color != background]
        preferred = [color for color in (previous_noise, previous_foreground)
                     if color in options]
        if preferred and rng.random() < 0.45:
            foreground = int(rng.choice(preferred))
        else:
            foreground = int(rng.choice(options))
        if index == 1 and foreground == first_foreground:
            foreground = int(rng.choice([color for color in options if color != first_foreground]))
        if index == 0:
            first_foreground = foreground
        noise_color = int(rng.choice(
            [color for color in range(10) if color not in (background, foreground)]
        ))
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        pair = _make_pair(
            pair_rng, background, foreground, noise_color,
            extra_mode, missing_mode, seen_inputs, orientations[index]
        )
        destination = "train" if index < num_train else "test"
        puzzle[destination].append(pair)
        previous_foreground, previous_noise = foreground, noise_color
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    input_grid, output = pair["input"], pair["output"]
    for grid in (input_grid, output):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert all(1 <= n <= 30 for n in grid.shape)
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((grid >= 0) & (grid <= 9))
    assert input_grid.shape == output.shape
    width, length = min(output.shape), max(output.shape)
    assert 4 <= width <= 9 and max(12, 2 * 2 + 3) <= length <= 30
    horizontal = output.shape[1] > output.shape[0]
    colors, counts = np.unique(output, return_counts=True)
    assert len(colors) == 2 and int(np.min(counts)) == length
    background = int(colors[np.argmax(counts)])
    foreground = int(colors[np.argmin(counts)])
    matches = [
        (start, span)
        for start in (0, width - 1)
        for span in range(2, width)
        if np.array_equal(
            output, _clean_grid(length, width, horizontal,
                                background, foreground, start, span)
        )
    ]
    assert len(matches) == 1
    start, span = matches[0]
    assert length >= max(12, 2 * span + 3)
    extra_same = int(np.count_nonzero(
        (output == background) & (input_grid == foreground)
    ))
    extra_distinct = int(np.count_nonzero(
        (output == background) & (input_grid != background)
        & (input_grid != foreground)
    ))
    erased = int(np.count_nonzero(
        (output == foreground) & (input_grid == background)
    ))
    overwritten = int(np.count_nonzero(
        (output == foreground) & (input_grid != background)
        & (input_grid != foreground)
    ))
    assert 2 <= extra_same + extra_distinct <= 10
    assert erased <= 3 and overwritten <= 3 and erased + overwritten <= 4
    assert len(set(int(x) for x in np.unique(input_grid)) -
               {background, foreground}) <= 1
    assert not np.array_equal(input_grid, output)
    assert _input_digest(input_grid) not in SOURCE_INPUT_DIGESTS
    recovered, gap = _recover(input_grid, background)
    assert gap >= 2 and np.array_equal(recovered, output)
    return {
        "background": background, "foreground": foreground,
        "horizontal": horizontal, "width": width, "length": length,
        "start": start, "span": span, "margin": width - 1 - span,
        "extra_same": extra_same, "extra_distinct": extra_distinct,
        "erased": erased, "overwritten": overwritten,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    info = {"train": [], "test": []}
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            info[regime].append(_inspect_pair(pair))
            signature = _pair_signature(pair)
            assert signature[:2] not in seen_inputs and signature not in seen_pairs
            seen_inputs.add(signature[:2])
            seen_pairs.add(signature)
    training = info["train"]
    assert any(x["extra_same"] for x in training)
    assert any(x["extra_distinct"] for x in training)
    assert any(x["erased"] + x["overwritten"] for x in training)
    assert any(x["erased"] + x["overwritten"] == 0 for x in training)


def _puzzle_signature(puzzle):
    return tuple(
        (regime, _pair_signature(pair))
        for regime in ("train", "test") for pair in puzzle[regime]
    )


def self_test() -> None:
    from collections import Counter

    measurements = {name: set() for name in (
        "train_counts", "test_counts", "widths", "lengths", "spans",
        "margins", "backgrounds", "foregrounds", "starts", "orientations",
        "noise_modes", "missing_modes", "extra_counts", "end_phases",
    )}
    puzzles = set()
    coverage = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_signature(puzzle))
        measurements["train_counts"].add(len(puzzle["train"]))
        measurements["test_counts"].add(len(puzzle["test"]))
        train_info = [_inspect_pair(pair) for pair in puzzle["train"]]
        test_info = [_inspect_pair(pair) for pair in puzzle["test"]]
        train_orientations = {x["horizontal"] for x in train_info}
        test_orientations = {x["horizontal"] for x in test_info}
        coverage["mixed_training"] += len(train_orientations) == 2
        coverage["same_orientation"] += (
            train_orientations == test_orientations
            and len(train_orientations) == 1
        )
        coverage["horizontal_to_vertical"] += (
            train_orientations == {True} and test_orientations == {False}
        )
        coverage["vertical_to_horizontal"] += (
            train_orientations == {False} and test_orientations == {True}
        )
        coverage["changing_background"] += len(
            {x["background"] for x in train_info}
        ) > 1
        coverage["varying_width"] += len(
            {x["width"] for x in train_info + test_info}
        ) > 1
        prior_foregrounds, prior_noise = set(), set()
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                x = _inspect_pair(pair)
                input_grid, output = pair["input"], pair["output"]
                coverage["additions_only"] += x["erased"] + x["overwritten"] == 0
                coverage["erasures"] += x["erased"] > 0
                coverage["colored_overwrites"] += x["overwritten"] > 0
                coverage["source_dimensions"] += (x["width"], x["length"]) in {
                    (6, 21), (6, 24), (6, 18), (6, 15), (6, 19)
                }
                coverage["combined_changes"] += (
                    x["margin"] > 0 and x["width"] != 6
                    and x["erased"] + x["overwritten"] > 0
                    and x["extra_same"] + x["extra_distinct"] >= 5
                )
                coverage["reused_path_color"] += x["foreground"] in prior_foregrounds
                coverage["noise_becomes_path"] += x["foreground"] in prior_noise
                prior_foregrounds.add(x["foreground"])
                prior_noise.update(
                    int(color) for color in np.unique(input_grid)
                    if color not in (x["background"], x["foreground"])
                )
                positions = _positions(
                    x["length"], x["width"], x["start"], x["span"]
                )
                margin_steps = []
                for step in range(x["length"]):
                    for transverse in range(x["width"]):
                        row, column = _point(x["horizontal"], step, transverse)
                        if output[row, column] != x["background"] or input_grid[row, column] == x["background"]:
                            continue
                        distance = abs(transverse - int(positions[step]))
                        coverage["neighbor_decoys"] += distance == 1
                        coverage["distant_decoys"] += distance >= 2
                        coverage["turn_decoys"] += (
                            step % x["span"] in (0, 1, x["span"] - 1)
                            and distance == 1
                        )
                        if x["margin"] and transverse == (
                            x["width"] - 1 if x["start"] == 0 else 0
                        ):
                            margin_steps.append(step)
                coverage["repeated_margin_decoys"] += any(
                    later - earlier == 2 * x["span"]
                    for earlier in margin_steps for later in margin_steps
                )
                for key, value in (
                    ("widths", x["width"]), ("lengths", x["length"]),
                    ("spans", x["span"]), ("margins", x["margin"]),
                    ("backgrounds", x["background"]),
                    ("foregrounds", x["foreground"]),
                    ("starts", x["start"]),
                    ("orientations", x["horizontal"]),
                    ("extra_counts", x["extra_same"] + x["extra_distinct"]),
                    ("noise_modes", (x["extra_same"] > 0, x["extra_distinct"] > 0)),
                    ("missing_modes", (x["erased"] > 0, x["overwritten"] > 0)),
                    ("end_phases", (x["length"] - 1) % (2 * x["span"])),
                ):
                    measurements[key].add(value)
    assert len(puzzles) >= 40
    assert len(measurements["train_counts"]) >= 4
    assert len(measurements["test_counts"]) >= 3
    assert measurements["widths"] == set(range(4, 10))
    assert len(measurements["lengths"]) >= 15
    assert len(measurements["spans"]) >= 6
    assert len(measurements["margins"]) >= 5
    assert len(measurements["backgrounds"]) >= 7
    assert len(measurements["foregrounds"]) >= 8
    assert len(measurements["starts"]) >= 2
    assert len(measurements["orientations"]) == 2
    assert len(measurements["noise_modes"]) == 3
    assert len(measurements["missing_modes"]) == 4
    assert 10 in measurements["extra_counts"]
    assert len(measurements["end_phases"]) >= 10
    assert coverage["mixed_training"] >= 5
    assert coverage["same_orientation"] >= 5
    assert coverage["horizontal_to_vertical"] >= 2
    assert coverage["vertical_to_horizontal"] >= 2
    assert coverage["changing_background"] >= 5
    assert coverage["varying_width"] >= 35
    assert coverage["additions_only"] >= 20
    assert coverage["erasures"] >= 20
    assert coverage["colored_overwrites"] >= 20
    assert coverage["source_dimensions"] >= 1
    assert coverage["combined_changes"] >= 10
    assert coverage["reused_path_color"] >= 10
    assert coverage["noise_becomes_path"] >= 10
    assert coverage["neighbor_decoys"] >= 20
    assert coverage["distant_decoys"] >= 20
    assert coverage["turn_decoys"] >= 10
    assert coverage["repeated_margin_decoys"] >= 5
    for seed, a, b in (
        (100, 2, 1), (101, 10, 1), (102, 10, 4),
        (103, None, 1), (104, None, 4),
        (105, 2, None), (106, 10, None), (107, 7, 3),
    ):
        puzzle = generate_puzzle(seed, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for args in ((7, None, None), (7, 6, 2)):
        assert _puzzle_signature(generate_puzzle(*args)) == _puzzle_signature(generate_puzzle(*args))
    for a, b in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (3, 4)):
        try:
            generate_puzzle(41, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted illegal counts: {a}, {b}")
