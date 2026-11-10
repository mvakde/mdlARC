"""Solved ARC puzzles with whole-grid horizontal duplication."""

import numpy as np


_SOURCE_INPUTS = frozenset(
    tuple(tuple(int(ch) for ch in row) for row in rows)
    for rows in (
        ("30003", "03030", "00300"),
        ("202", "202", "222"),
        ("444", "404", "404", "404", "404", "444"),
        ("808", "000", "080", "000", "808"),
        ("66666", "66666"),
        ("222222", "222222", "211112", "211112", "222222", "222222"),
    )
)


def _signature(grid):
    return tuple(tuple(int(v) for v in row) for row in grid)


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _counts(rng, num_train, num_test):
    for value, low, high, name in (
        (num_train, 2, 10, "num_train"),
        (num_test, 1, 4, "num_test"),
    ):
        if value is not None and (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be an integer from {low} to {high}")
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _dimension(rng, challenge, regime, axis):
    # The bands overlap. Challenge and regime change weights, not legality.
    bands = ((2, 5), (5, 10), (10, 15)) if axis == "width" else (
        (2, 6), (6, 13), (13, 24)
    )
    weights = np.array(
        ((0.48, 0.40, 0.12), (0.32, 0.43, 0.25), (0.22, 0.36, 0.42))[challenge]
    )
    if regime == "test":
        weights *= np.array((0.8, 1.0, 1.3))
        weights /= weights.sum()
    low, high = bands[int(rng.choice(3, p=weights))]
    return int(rng.integers(low, high + 1))


def _has_horizontal_period(grid):
    width = grid.shape[1]
    return any(
        width % period == 0
        and np.array_equal(grid, np.tile(grid[:, :period], (1, width // period)))
        for period in range(1, width)
    )


def _pattern(rng, height, width, style):
    mask = np.zeros((height, width), dtype=bool)
    if style == "noise":
        mask = rng.random((height, width)) < rng.uniform(0.22, 0.72)
    elif style == "sparse":
        count = int(rng.integers(1, max(3, height * width // 5 + 1)))
        mask.flat[rng.choice(height * width, size=count, replace=False)] = True
    elif style == "diagonal":
        slope = int(rng.choice((-1, 1)))
        offset = int(rng.integers(-height + 1, width))
        for row in range(height):
            col = offset + slope * row
            if 0 <= col < width:
                mask[row, col] = True
                if rng.random() < 0.20 and col + 1 < width:
                    mask[row, col + 1] = True
    elif style == "stripes":
        spacing = int(rng.integers(2, 5))
        phase = int(rng.integers(spacing))
        if rng.random() < 0.5:
            mask[:, phase::spacing] = True
        else:
            mask[phase::spacing, :] = True
    elif style == "frame":
        top = int(rng.integers(0, max(1, height // 3)))
        left = int(rng.integers(0, max(1, width // 3)))
        bottom = int(rng.integers(max(top + 1, height * 2 // 3), height + 1))
        right = int(rng.integers(max(left + 1, width * 2 // 3), width + 1))
        mask[top:bottom, left] = True
        mask[top:bottom, right - 1] = True
        mask[top, left:right] = True
        mask[bottom - 1, left:right] = True
    elif style == "blocks":
        for _ in range(int(rng.integers(1, 5))):
            top = int(rng.integers(height))
            left = int(rng.integers(width))
            bottom = int(rng.integers(top + 1, height + 1))
            right = int(rng.integers(left + 1, width + 1))
            mask[top:bottom, left:right] = True
    else:  # local motifs, often disconnected
        for _ in range(int(rng.integers(2, 7))):
            row = int(rng.integers(height))
            col = int(rng.integers(width))
            mask[row, col] = True
            if rng.random() < 0.6:
                mask[row, min(width - 1, col + 1)] = True
            if rng.random() < 0.6:
                mask[min(height - 1, row + 1), col] = True
    return mask


def _palette(rng, count, favorite=None):
    if count == 1:
        if favorite is not None and rng.random() < 0.65:
            return (int(favorite),)
        return (int(rng.integers(0, 10)),)
    background = 0 if rng.random() < 0.62 else int(rng.integers(1, 10))
    available = [c for c in range(10) if c != background]
    foreground = []
    if favorite is not None and favorite != background and rng.random() < 0.7:
        foreground.append(int(favorite))
        available.remove(int(favorite))
    foreground.extend(
        int(c) for c in rng.choice(
            available, size=count - 1 - len(foreground), replace=False
        )
    )
    return (background, *foreground)


def _make_grid(rng, challenge, regime, favorite=None, anchor=False):
    height = _dimension(rng, challenge, regime, "height")
    width = _dimension(rng, challenge, regime, "width")
    if anchor:
        height = max(3, height)
        width = max(4, width)
    style = str(rng.choice(
        ("noise", "sparse", "diagonal", "stripes", "frame", "blocks", "motifs")
    ))
    symmetric = not anchor and rng.random() < (0.45 if challenge == 0 else 0.31)
    if anchor:
        style = str(rng.choice(("noise", "blocks", "motifs")))
    solid_probability = 0.15 if regime == "test" else 0.11
    if not anchor and rng.random() < solid_probability:
        count = 1
    else:
        count = int(rng.choice((2, 3, 4), p=(0.62, 0.29, 0.09)))
    palette = _palette(rng, count, favorite)
    if count == 1:
        return np.full((height, width), palette[0], dtype=np.int8)

    mask = _pattern(rng, height, width, style)
    if symmetric:
        for col in range(width // 2):
            mask[:, width - 1 - col] = mask[:, col]
    if rng.random() < 0.29:
        mask[int(rng.integers(height)), :] = False
    if rng.random() < 0.26:
        mask[:, int(rng.integers(width))] = False
    if width >= 4 and rng.random() < 0.30:
        left = int(rng.integers(0, min(3, width - 2)))
        right = left if symmetric else int(rng.integers(0, min(3, width - left - 1)))
        if not symmetric and left == right:
            right = (right + 1) % min(3, width - left - 1)
        mask[:, :left] = False
        if right:
            mask[:, width - right:] = False
    if not mask.any():
        mask[int(rng.integers(height)), width // 2] = True
    if mask.all():
        mask[0, 0] = False
    grid = np.full((height, width), palette[0], dtype=np.int8)
    grid[mask] = rng.choice(palette[1:], size=int(mask.sum()))
    if symmetric:
        for col in range(width // 2):
            grid[:, width - 1 - col] = grid[:, col]
    return grid


def _new_grid(
    rng, challenge, regime, seen, favorite=None, anchor=False,
    other_shapes=(), other_palettes=(),
):
    for _ in range(240):
        grid = _make_grid(rng, challenge, regime, favorite, anchor)
        signature = _signature(grid)
        if signature in seen or signature in _SOURCE_INPUTS:
            continue
        if other_shapes and grid.shape in other_shapes:
            continue
        if frozenset(int(c) for c in np.unique(grid)) in other_palettes:
            continue
        if anchor and (
            len(np.unique(grid)) < 2
            or np.array_equal(grid, grid[:, ::-1])
            or _has_horizontal_period(grid)
        ):
            continue
        seen.add(signature)
        return grid
    # Deterministic fallback for exceptionally unlucky bounded sampling.
    grid = np.zeros((3, 15), dtype=np.int8)
    for code in range(1, 1000):
        grid[0, 0] = code % 9 + 1
        grid[1, 2] = (code // 9) % 9 + 1
        grid[2, 14] = (code // 81) % 9 + 1
        signature = _signature(grid)
        if (
            signature not in seen
            and signature not in _SOURCE_INPUTS
            and grid.shape not in other_shapes
            and frozenset(int(c) for c in np.unique(grid)) not in other_palettes
        ):
            seen.add(signature)
            return grid.copy()
    raise RuntimeError("Could not construct a distinct grid")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, num_train, num_test)
    challenge = int(rng.integers(0, 3))
    seen = set()
    pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
    training = [_new_grid(pair_rng, challenge, "train", seen, anchor=True)]
    for index in range(1, train_count):
        colors = [int(c) for grid in training for c in np.unique(grid) if c != 0]
        favorite = int(rng.choice(colors)) if colors and rng.random() < 0.45 else None
        shapes = (training[0].shape,) if index == 1 else ()
        palettes = (
            (frozenset(int(c) for c in np.unique(training[0])),)
            if index == 1 else ()
        )
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        training.append(
            _new_grid(
                pair_rng, challenge, "train", seen, favorite,
                other_shapes=shapes, other_palettes=palettes,
            )
        )
    rng.shuffle(training)
    training_colors = [int(c) for grid in training for c in np.unique(grid) if c != 0]
    tests = []
    for _ in range(test_count):
        favorite = int(rng.choice(training_colors)) if training_colors and rng.random() < 0.68 else None
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
        tests.append(_new_grid(pair_rng, challenge, "test", seen, favorite))
    puzzle = {
        "train": [
            {"input": grid, "output": np.concatenate((grid, grid), axis=1)}
            for grid in training
        ],
        "test": [
            {"input": grid, "output": np.concatenate((grid, grid), axis=1)}
            for grid in tests
        ],
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Puzzle keys")
    _check(all(isinstance(puzzle[k], list) for k in ("train", "test")), "Pair lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "Pair counts")
    seen_inputs = set()
    seen_pairs = set()
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Pair keys")
            for key in ("input", "output"):
                grid = pair[key]
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "Grid array and rank")
                _check(all(1 <= n <= 30 for n in grid.shape), "Grid size")
                _check(grid.dtype.kind in "iu", "Integer dtype")
                _check(bool(np.all((grid >= 0) & (grid <= 9))), "Color range")
            grid, output = pair["input"], pair["output"]
            height, width = grid.shape
            _check(width <= 15, "Input width")
            _check(output.shape == (height, 2 * width), "Output dimensions")
            _check(np.array_equal(output[:, :width], grid), "First copy")
            _check(np.array_equal(output[:, width:], grid), "Second copy")
            signature = _signature(grid)
            pair_signature = (signature, _signature(output))
            _check(signature not in seen_inputs, "Duplicate input")
            _check(pair_signature not in seen_pairs, "Duplicate pair")
            _check(signature not in _SOURCE_INPUTS, "Source copy")
            seen_inputs.add(signature)
            seen_pairs.add(pair_signature)
    train = [pair["input"] for pair in puzzle["train"]]
    _check(len({grid.shape for grid in train}) >= 2, "Training shape diversity")
    _check(
        len({frozenset(int(c) for c in np.unique(grid)) for grid in train}) >= 2,
        "Training palette diversity",
    )
    _check(any(
        len(np.unique(grid)) >= 2
        and not np.array_equal(grid, grid[:, ::-1])
        and not _has_horizontal_period(grid)
        for grid in train
    ), "Training must demonstrate literal, nonperiodic copying")


def _puzzle_signature(puzzle):
    return tuple(
        tuple((_signature(p["input"]), _signature(p["output"])) for p in puzzle[k])
        for k in ("train", "test")
    )


def _component_count(grid):
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    remaining = set(zip(*np.nonzero(grid != background)))
    components = 0
    while remaining:
        components += 1
        frontier = [remaining.pop()]
        while frontier:
            row, col = frontier.pop()
            for neighbor in ((row - 1, col), (row + 1, col),
                             (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    frontier.append(neighbor)
    return components


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    train_shapes, test_shapes, structures = set(), set(), set()
    small_tests = large_tests = solid_train = multicolor_train = 0
    asymmetric_test = symmetric_test = blank_test = 0
    observed = {
        name: 0 for name in (
            "nonzero_train", "nonzero_test", "multicolor_test", "solid_test",
            "zero_row_train", "zero_row_test", "zero_col_train", "zero_col_test",
            "unequal_train", "unequal_test", "portrait_train", "landscape_train",
            "portrait_test", "landscape_test", "reused_test", "novel_test",
            "small_symmetric_test", "large_asymmetric_multicolor_test",
            "disconnected_train", "disconnected_test",
        )
    }
    anchor_positions = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        demonstrated_colors = {
            int(c) for p in puzzle["train"] for c in np.unique(p["input"]) if c != 0
        }
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_signature(puzzle))
        anchor_positions.update(
            i for i, p in enumerate(puzzle["train"])
            if len(np.unique(p["input"])) >= 2
            and not np.array_equal(p["input"], p["input"][:, ::-1])
            and not _has_horizontal_period(p["input"])
        )
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                grid, out = pair["input"], pair["output"]
                _check(np.array_equal(out, np.tile(grid, (1, 2))), "Independent whole-grid rule")
                structures.add(_signature(grid))
                colors, counts = np.unique(grid, return_counts=True)
                palette = set(int(c) for c in colors)
                background = int(colors[int(np.argmax(counts))])
                left = next(
                    (i for i in range(grid.shape[1])
                     if not np.all(grid[:, i] == background)), grid.shape[1]
                )
                right = next(
                    (i for i in range(grid.shape[1])
                     if not np.all(grid[:, grid.shape[1] - i - 1] == background)),
                    grid.shape[1],
                )
                suffix = "train" if regime == "train" else "test"
                observed[f"nonzero_{suffix}"] += 0 not in palette
                observed[f"zero_row_{suffix}"] += bool(
                    np.any(np.all(grid == 0, axis=1))
                )
                observed[f"zero_col_{suffix}"] += bool(
                    np.any(np.all(grid == 0, axis=0))
                )
                observed[f"unequal_{suffix}"] += (
                    left != right and min(left, right) < grid.shape[1]
                )
                observed[f"portrait_{suffix}"] += grid.shape[0] > grid.shape[1]
                observed[f"landscape_{suffix}"] += grid.shape[1] > grid.shape[0]
                observed[f"disconnected_{suffix}"] += _component_count(grid) >= 2
                if regime == "train":
                    train_shapes.add(grid.shape)
                    solid_train += len(np.unique(grid)) == 1
                    multicolor_train += len(np.unique(grid)) >= 3
                else:
                    test_shapes.add(grid.shape)
                    small_tests += grid.shape[1] <= 5
                    large_tests += grid.shape[1] >= 10
                    asymmetric_test += not np.array_equal(grid, grid[:, ::-1])
                    symmetric_test += np.array_equal(grid, grid[:, ::-1])
                    blank_test += bool(
                        np.any(np.all(grid == 0, axis=0))
                        or np.any(np.all(grid == 0, axis=1))
                    )
                    observed["solid_test"] += len(colors) == 1
                    observed["multicolor_test"] += len(colors) >= 3
                    nonzero = palette - {0}
                    observed["reused_test"] += bool(nonzero & demonstrated_colors)
                    observed["novel_test"] += bool(nonzero - demonstrated_colors)
                    observed["small_symmetric_test"] += (
                        grid.shape[1] <= 6
                        and np.array_equal(grid, grid[:, ::-1])
                        and len(colors) == 2 and 0 in palette
                    )
                    observed["large_asymmetric_multicolor_test"] += (
                        grid.shape[1] >= 10
                        and len(colors) >= 3
                        and not np.array_equal(grid, grid[:, ::-1])
                    )
    _check(len(train_counts) >= 4 and len(test_counts) >= 3, "Count diversity")
    _check(len(signatures) >= 40, "Seed diversity")
    _check(
        len(train_shapes) >= 25 and len(test_shapes) >= 20 and len(structures) >= 150,
        "Structural diversity",
    )
    _check(small_tests >= 5 and large_tests >= 5, "Overlapping small and large tests")
    _check(solid_train >= 4 and multicolor_train >= 8, "Training palette diversity")
    _check(
        asymmetric_test >= 8 and symmetric_test >= 5 and blank_test >= 5,
        "Test layout diversity",
    )
    _check(len(anchor_positions) >= 3, "Vary order of asymmetric evidence")
    _check(
        {width for _, width in train_shapes}
        & {width for _, width in test_shapes}
        >= {3, 5, 10},
        "Train/test width overlap",
    )
    _check(all(value >= 3 for value in observed.values()), "Review feature coverage")
    for seed in (0, 7, 41):
        for counts in ((None, None), (6, 3), (None, 4), (2, None)):
            a = generate_puzzle(seed, *counts)
            b = generate_puzzle(seed, *counts)
            _check(_puzzle_signature(a) == _puzzle_signature(b), "Determinism")
    for counts in (
        (2, 1), (10, 1), (10, 4), (None, 1),
        (None, 4), (2, None), (10, None),
    ):
        puzzle = generate_puzzle(91, *counts)
        validate_generated_puzzle(puzzle)
        _check(
            all(want is None or len(puzzle[k]) == want
                for k, want in zip(("train", "test"), counts)),
            "Requested counts",
        )
    for counts in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (4, 4), (3, 4), (2, 4),
    ):
        try:
            generate_puzzle(93, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
