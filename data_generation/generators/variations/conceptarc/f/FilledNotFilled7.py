"""Generate ARC puzzles that toggle the interiors of line-bounded spans."""

import numpy as np


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _key(grid):
    return grid.shape, grid.astype(np.uint8, copy=False).tobytes()


def _runs(indices):
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1) + 1
    return [(int(g[0]), int(g[-1])) for g in np.split(indices, cuts)]


def _draw(height, width, spans, background, foreground, orientation):
    """Draw horizontal spans and transpose for a vertical pair."""
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for top, bottom, left, right, mode in spans:
        inp[top, left:right + 1] = foreground
        inp[bottom, left:right + 1] = foreground
        if mode == "solid":
            inp[top + 1:bottom, left:right + 1] = foreground
            out[top, left:right + 1] = foreground
            out[bottom, left:right + 1] = foreground
        else:
            out[top:bottom + 1, left:right + 1] = foreground
    if orientation == "v":
        return inp.T.copy(), out.T.copy()
    return inp, out


def _source_keys():
    # Compare exact original pairs; their dimensions and layouts remain legal.
    specs = (
        (6, 14, 2, 4, 0, 13, "empty", 2),
        (6, 18, 1, 4, 4, 15, "empty", 3),
        (6, 15, 1, 4, 3, 12, "solid", 7),
        (10, 15, 3, 7, 4, 9, "empty", 5),
        (9, 13, 4, 6, 2, 6, "solid", 4),
        (6, 15, 0, 5, 0, 14, "solid", 9),
    )
    pairs = set()
    for height, width, top, bottom, left, right, mode, color in specs:
        inp, out = _draw(
            height, width, [(top, bottom, left, right, mode)], 0, color, "h"
        )
        pairs.add((_key(inp), _key(out)))
    return pairs


_SOURCE_KEYS = _source_keys()


def _parse_oriented(inp, background, vertical):
    grid = inp.T if vertical else inp
    colors = np.unique(grid[grid != background])
    if len(colors) != 1:
        return None
    foreground = int(colors[0])
    bands = _runs(np.flatnonzero(np.any(grid != background, axis=0)))
    if not 1 <= len(bands) <= 2:
        return None
    if any(bands[i + 1][0] - bands[i][1] < 3 for i in range(len(bands) - 1)):
        return None
    reconstruction = np.full(grid.shape, background, dtype=np.int8)
    result = reconstruction.copy()
    spans = []
    for left, right in bands:
        rows = np.flatnonzero(np.any(grid[:, left:right + 1] != background, axis=1))
        top, bottom = int(rows[0]), int(rows[-1])
        depth, length = bottom - top - 1, right - left + 1
        if depth < 1 or length < 4 or length <= depth + 2:
            return None
        if not np.all(grid[top, left:right + 1] == foreground):
            return None
        if not np.all(grid[bottom, left:right + 1] == foreground):
            return None
        interior = grid[top + 1:bottom, left:right + 1]
        if np.all(interior == background):
            mode = "empty"
        elif np.all(interior == foreground):
            mode = "solid"
        else:
            return None
        reconstruction[top, left:right + 1] = foreground
        reconstruction[bottom, left:right + 1] = foreground
        if mode == "solid":
            reconstruction[top + 1:bottom, left:right + 1] = foreground
            result[top, left:right + 1] = foreground
            result[bottom, left:right + 1] = foreground
        else:
            result[top:bottom + 1, left:right + 1] = foreground
        spans.append((top, bottom, left, right, mode))
    if not np.array_equal(grid, reconstruction):
        return None
    if vertical:
        result = result.T.copy()
    return result, spans, foreground


def _solve_input(inp, background):
    candidates = []
    for vertical in (False, True):
        parsed = _parse_oriented(inp, background, vertical)
        if parsed is not None:
            output, spans, foreground = parsed
            candidates.append((output, spans, foreground, "v" if vertical else "h"))
    assert len(candidates) == 1, "span orientation must be visible"
    return candidates[0]


def _single(rng, background, foreground, orientation, mode, regime, edge, anchor):
    if regime == "test" and rng.random() < 0.3:
        depth = int(rng.integers(7, 13))
    else:
        depth = int(rng.integers(1, 8 if regime == "train" else 10))
    length = int(rng.integers(depth + 3, (28 if anchor else 30) + 1))
    if edge == "full_canvas":
        left_margin = right_margin = 0
    else:
        available = min(12, 30 - length)
        if anchor:
            extra_width = int(rng.integers(2, available + 1))
            left_margin = int(rng.integers(1, extra_width))
        else:
            extra_width = int(rng.integers(0, available + 1))
            left_margin = int(rng.integers(0, extra_width + 1))
        right_margin = extra_width - left_margin
    width = left_margin + length + right_margin
    span_height = depth + 2
    if edge in ("full_y", "full_canvas"):
        top_margin = bottom_margin = 0
    elif edge == "upper":
        top_margin = 0
        bottom_margin = int(rng.integers(1, min(12, 30 - span_height) + 1))
    elif edge == "lower":
        top_margin = int(rng.integers(1, min(12, 30 - span_height) + 1))
        bottom_margin = 0
    else:
        extra_height = int(rng.integers(2, min(14, 30 - span_height) + 1))
        top_margin = int(rng.integers(1, extra_height))
        bottom_margin = extra_height - top_margin
    height = top_margin + span_height + bottom_margin
    span = (
        top_margin, top_margin + span_height - 1,
        left_margin, left_margin + length - 1, mode
    )
    return _draw(height, width, [span], background, foreground, orientation)


def _multiple(rng, background, foreground, orientation, modes, edge):
    lengths = [int(rng.integers(5, 12)) for _ in range(2)]
    depths = [int(rng.integers(1, min(8, length - 2))) for length in lengths]
    gap = int(rng.integers(2, 5))
    minimum_width = sum(lengths) + gap
    extra_width = int(rng.integers(0, min(6, 30 - minimum_width) + 1))
    left_margin = int(rng.integers(0, extra_width + 1))
    width = minimum_width + extra_width
    lefts = (left_margin, left_margin + lengths[0] + gap)
    maximum_height = max(depths) + 2
    height = int(rng.integers(maximum_height + 1, min(30, maximum_height + 12) + 1))
    spans = []
    for index, (left, length, depth, mode) in enumerate(
        zip(lefts, lengths, depths, modes)
    ):
        span_height = depth + 2
        if index == 0 and edge == "upper":
            top = 0
        elif index == 0 and edge == "lower":
            top = height - span_height
        else:
            top = int(rng.integers(0, height - span_height + 1))
        spans.append((top, top + span_height - 1, left, left + length - 1, mode))
    return _draw(height, width, spans, background, foreground, orientation)


def _fallback(ordinal, variant, background, foreground, orientation,
              modes, multiple, edge):
    """Finite, indexed layouts if random construction ever repeats a grid."""
    if multiple:
        height, width = 9 + ordinal, 16 + variant
        first_top = 0 if edge == "upper" else 1
        second_top = height - 5 if edge == "lower" else height - 6
        spans = [
            (first_top, first_top + 3, 0, 5, modes[0]),
            (second_top, second_top + 4, 9, 15, modes[1]),
        ]
    else:
        depth = 2 + variant % 3
        length = 10 + variant
        if edge == "full_canvas":
            height, width, top, left = depth + 2, length, 0, 0
        else:
            height = depth + 5 + ordinal
            width = length + 2
            top = 0 if edge in ("upper", "full_y") else (
                height - depth - 2 if edge == "lower" else 1
            )
            if edge == "full_y":
                height = depth + 2
            left = 1
        spans = [(top, top + depth + 1, left, left + length - 1, modes[0])]
    return _draw(height, width, spans, background, foreground, orientation)


def _signature(puzzle):
    return tuple(
        (regime, _key(pair["input"]), _key(pair["output"]))
        for regime in ("train", "test") for pair in puzzle[regime]
    )


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    anchor = puzzle["train"][0]
    assert isinstance(anchor, dict) and set(anchor) == {"input", "output"}
    first = anchor["input"]
    assert isinstance(first, np.ndarray) and first.ndim == 2
    background = int(first[0, 0])
    assert first[0, -1] == first[-1, 0] == first[-1, -1] == background
    seen_inputs, seen_pairs = set(), set()
    train_modes, train_orientations = set(), set()
    train_multi = test_multi = False
    for regime in ("train", "test"):
        for pair in puzzle[regime]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            input_key, pair_key = _key(inp), (_key(inp), _key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            expected, spans, foreground, orientation = _solve_input(inp, background)
            assert np.array_equal(out, expected)
            assert foreground != background and not np.array_equal(inp, out)
            if regime == "train":
                train_modes.update(span[4] for span in spans)
                train_orientations.add(orientation)
                train_multi |= len(spans) == 2
            else:
                test_multi |= len(spans) == 2
                assert orientation in {"h", "v"}
    expected, spans, _, _ = _solve_input(first, background)
    assert np.array_equal(anchor["output"], expected)
    assert len(spans) == 1 and spans[0][4] == "empty"
    assert train_modes == {"empty", "solid"}
    assert train_multi or not test_multi
    test_orientations = {
        _solve_input(pair["input"], background)[3] for pair in puzzle["test"]
    }
    assert test_orientations <= train_orientations


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    background = int(rng.integers(0, 10))
    orientation_regime = str(
        rng.choice(("h", "v", "mixed"), p=(0.35, 0.30, 0.35))
    )
    first_orientation = (
        str(rng.choice(("h", "v"))) if orientation_regime == "mixed"
        else orientation_regime
    )
    second_orientation = (
        ("v" if first_orientation == "h" else "h")
        if orientation_regime == "mixed" else first_orientation
    )
    multi_enabled = bool(rng.random() < 0.38)
    colors = [color for color in range(10) if color != background]
    previous_foreground = None
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for regime, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            anchor = regime == "train" and index == 0
            if previous_foreground is not None and rng.random() < 0.38:
                foreground = previous_foreground
            else:
                foreground = int(rng.choice(colors))
            previous_foreground = foreground
            if anchor:
                orientation, modes, multiple, edge = (
                    first_orientation, ["empty"], False, "none"
                )
            elif regime == "train" and index == 1:
                orientation = second_orientation
                multiple = multi_enabled
                modes = ["solid", "empty"] if multiple else ["solid"]
                edge = str(rng.choice(
                    ("none", "upper", "lower"), p=(0.40, 0.30, 0.30)
                ))
            else:
                orientation = (
                    str(rng.choice(("h", "v"))) if orientation_regime == "mixed"
                    else first_orientation
                )
                multiple = multi_enabled and (
                    regime == "test" and index == 0 or rng.random() < 0.35
                )
                first_mode = str(rng.choice(("empty", "solid")))
                modes = [first_mode]
                if multiple:
                    modes.append(
                        ("solid" if first_mode == "empty" else "empty")
                        if rng.random() < 0.65
                        else str(rng.choice(("empty", "solid")))
                    )
                edge_choices = (
                    ("none", "upper", "lower") if multiple else
                    ("none", "upper", "lower", "full_y", "full_canvas")
                )
                probabilities = (
                    (0.40, 0.30, 0.30) if multiple else
                    ((0.30, 0.23, 0.23, 0.12, 0.12) if regime == "train"
                     else (0.18, 0.25, 0.25, 0.14, 0.18))
                )
                edge = str(rng.choice(edge_choices, p=probabilities))
            for _ in range(80):
                if multiple:
                    inp, out = _multiple(
                        rng, background, foreground, orientation, modes, edge
                    )
                else:
                    inp, out = _single(
                        rng, background, foreground, orientation,
                        modes[0], regime, edge, anchor
                    )
                key = _key(inp)
                if key not in seen_inputs and (key, _key(out)) not in _SOURCE_KEYS:
                    break
            else:
                for variant in range(15):
                    inp, out = _fallback(
                        len(seen_inputs), variant, background, foreground,
                        orientation, modes, multiple, edge
                    )
                    key = _key(inp)
                    if key not in seen_inputs and (key, _key(out)) not in _SOURCE_KEYS:
                        break
                else:
                    raise RuntimeError("could not construct a unique pair")
            seen_inputs.add(key)
            puzzle[regime].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    signatures, shapes, structures, foregrounds = set(), set(), set(), set()
    train_counts, test_counts, backgrounds = set(), set(), set()
    orientations, modes, edge_contacts, multi_modes = set(), set(), set(), set()
    train_depths, test_depths = set(), set()
    mixed_puzzles = multi_puzzles = zero_foreground = reused_foreground = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        background = int(puzzle["train"][0]["input"][0, 0])
        backgrounds.add(background)
        local_orientations, local_foregrounds = set(), []
        local_multi = False
        for regime in ("train", "test"):
            for pair in puzzle[regime]:
                inp, out = pair["input"], pair["output"]
                expected, spans, foreground, orientation = _solve_input(
                    inp.copy(), background
                )
                assert np.array_equal(out, expected)
                assert (_key(inp), _key(out)) not in _SOURCE_KEYS
                local_orientations.add(orientation)
                local_foregrounds.append(foreground)
                shapes.add(inp.shape)
                foregrounds.add(foreground)
                orientations.add(orientation)
                if foreground == 0:
                    zero_foreground += 1
                local_multi |= len(spans) == 2
                if len(spans) == 2:
                    multi_modes.add(tuple(span[4] for span in spans))
                oriented = inp.T if orientation == "v" else inp
                for top, bottom, left, right, mode in spans:
                    depth = bottom - top - 1
                    modes.add(mode)
                    (train_depths if regime == "train" else test_depths).add(depth)
                    structures.add(
                        (orientation, inp.shape, top, bottom, left, right, mode)
                    )
                    edge_contacts.add((
                        orientation, mode, top == 0,
                        bottom == oriented.shape[0] - 1
                    ))
        mixed_puzzles += len(local_orientations) == 2
        multi_puzzles += local_multi
        reused_foreground += len(set(local_foregrounds)) < len(local_foregrounds)
    assert len(signatures) == 42
    assert len(shapes) >= 70 and len(structures) >= 180
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(backgrounds) >= 8 and len(foregrounds) == 10
    assert orientations == {"h", "v"} and modes == {"empty", "solid"}
    assert len(train_depths) >= 6 and len(test_depths) >= 9
    assert train_depths & test_depths >= {1, 2, 3, 4}
    assert mixed_puzzles >= 6 and multi_puzzles >= 6
    assert zero_foreground >= 5 and reused_foreground >= 10
    assert ("empty", "solid") in multi_modes or ("solid", "empty") in multi_modes
    assert any(upper and not lower for _, _, upper, lower in edge_contacts)
    assert any(lower and not upper for _, _, upper, lower in edge_contacts)
    assert any(upper and lower for _, _, upper, lower in edge_contacts)
    for index, (requested_train, requested_test) in enumerate((
        (2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)
    )):
        puzzle = generate_puzzle(100 + index, requested_train, requested_test)
        validate_generated_puzzle(puzzle)
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
        assert _signature(puzzle) == _signature(
            generate_puzzle(100 + index, requested_train, requested_test)
        )
    for requested_train, requested_test in (
        (1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)
    ):
        try:
            generate_puzzle(200, requested_train, requested_test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
