"""Generate puzzles where one colored object slides intact to an edge."""

import numpy as np


COMMANDS = {6: "top", 8: "right", 4: "bottom", 3: "left"}
COLORS = tuple(COMMANDS)
BACKGROUNDS = (0, 1, 2, 5, 7, 9)
SOURCE_STARTS = frozenset({
    (6, 3, 4), (6, 4, 2), (8, 5, 2), (4, 2, 3),
    (8, 3, 3), (8, 2, 2), (3, 5, 3), (4, 4, 4),
})


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    test = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _schedule(rng, train_count, test_count):
    if train_count == 2:
        taught = [int(x) for x in rng.choice(COLORS, 2, replace=False)]
        held_out = None
    else:
        mode = int(rng.integers(0, 3 if train_count >= 4 else 2))
        if mode == 0:
            count = int(rng.integers(2, min(4, train_count) + 1))
            taught = [int(x) for x in rng.choice(COLORS, count, replace=False)]
            held_out = None
        elif mode == 1:
            held_out = int(rng.choice(COLORS))
            taught = [color for color in COLORS if color != held_out]
        else:
            taught = list(COLORS)
            held_out = None
    train_colors = taught + [int(rng.choice(taught)) for _ in range(train_count - len(taught))]
    rng.shuffle(train_colors)
    if held_out is None:
        test_colors = [int(rng.choice(taught)) for _ in range(test_count)]
    else:
        test_colors = [held_out]
        test_colors += [int(rng.choice(taught + [held_out])) for _ in range(test_count - 1)]
        rng.shuffle(test_colors)
    return train_colors, test_colors


def _shape(rng):
    kind = int(rng.choice(3, p=[0.32, 0.51, 0.17]))
    if kind == 0:
        height = width = 3
    else:
        height = int(rng.integers(3, 9))
        width = int(rng.integers(3, 9))
        if kind == 1 and height == width == 3:
            width = 4
    mask = np.zeros((height, width), dtype=bool)
    if kind < 2:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
    else:
        mask[:, -1 if int(rng.integers(0, 2)) else 0] = True
        mask[-1 if int(rng.integers(0, 2)) else 0, :] = True
    return mask


def _canvas(rng, mask):
    mh, mw = mask.shape
    category = int(rng.choice(5, p=[0.20, 0.21, 0.25, 0.25, 0.09]))
    if category == 0:
        return 10, 10
    if category == 1:
        low_h, high_h, low_w, high_w = 7, 11, 7, 11
    elif category == 2:
        low_h, high_h, low_w, high_w = 11, 18, 11, 18
    elif category == 3:
        low_h, high_h, low_w, high_w = 18, 27, 18, 27
    elif int(rng.integers(0, 2)):
        low_h, high_h, low_w, high_w = 8, 13, 18, 31
    else:
        low_h, high_h, low_w, high_w = 18, 31, 8, 13
    height = int(rng.integers(max(low_h, mh + 2), high_h))
    width = int(rng.integers(max(low_w, mw + 2), high_w))
    return height, width


def _destination(direction, row, col, canvas, mask):
    height, width = canvas
    mh, mw = mask.shape
    return {
        "top": (0, col),
        "right": (row, width - mw),
        "bottom": (height - mh, col),
        "left": (row, 0),
    }[direction]


def _draw(canvas, background, color, mask, row, col):
    grid = np.full(canvas, background, dtype=np.int64)
    mh, mw = mask.shape
    grid[row:row + mh, col:col + mw][mask] = color
    return grid


def _position(rng, direction, canvas, mask, allow_still):
    height, width = canvas
    mh, mw = mask.shape
    if direction in ("top", "bottom"):
        maximum = height - mh
        perpendicular = int(rng.integers(0, width - mw + 1))
    else:
        maximum = width - mw
        perpendicular = int(rng.integers(0, height - mh + 1))
    category = int(rng.integers(0, 4))
    if allow_still and int(rng.integers(0, 12)) == 0:
        distance = 0
    elif category == 0:
        distance = 1
    elif category == 1:
        distance = maximum
    else:
        distance = int(rng.integers(1, maximum + 1))
    axis = maximum - distance if direction in ("bottom", "right") else distance
    return (axis, perpendicular) if direction in ("top", "bottom") else (perpendicular, axis)


def _source_copy(grid, color, row, col, mask, background):
    return (grid.shape == (10, 10) and background == 0
            and mask.shape == (3, 3) and int(mask.sum()) == 8
            and (color, row, col) in SOURCE_STARTS)


def _make_pair(rng, color, allow_still, used_inputs, different_from=None, match=None):
    direction = COMMANDS[color]
    if match is not None:
        original = match["input"]
        background, _, mask, row, col = _extract(original)
        canvas = original.shape
        end_row, end_col = _destination(direction, row, col, canvas, mask)
        input_grid = _draw(canvas, background, color, mask, row, col)
        key = (input_grid.shape, input_grid.tobytes())
        if ((end_row, end_col) != (row, col)
                and not _source_copy(input_grid, color, row, col, mask, background)
                and key not in used_inputs):
            used_inputs.add(key)
            return {"input": input_grid,
                    "output": _draw(canvas, background, color, mask, end_row, end_col)}
    for _ in range(80):
        mask = _shape(rng)
        canvas = _canvas(rng, mask)
        if different_from is not None and (canvas, mask.shape, mask.tobytes()) == different_from:
            continue
        background = int(rng.choice(BACKGROUNDS, p=[0.52, 0.096, 0.096, 0.096, 0.096, 0.096]))
        row, col = _position(rng, direction, canvas, mask, allow_still)
        input_grid = _draw(canvas, background, color, mask, row, col)
        if _source_copy(input_grid, color, row, col, mask, background):
            continue
        key = (input_grid.shape, input_grid.tobytes())
        if key in used_inputs:
            continue
        used_inputs.add(key)
        output_row, output_col = _destination(direction, row, col, canvas, mask)
        output_grid = _draw(canvas, background, color, mask, output_row, output_col)
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not construct a distinct pair")


def _extract(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    height, width = grid.shape
    assert 1 <= height <= 30 and 1 <= width <= 30
    assert np.all((0 <= grid) & (grid <= 9))
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2
    background = int(colors[int(np.argmax(counts))])
    foreground = int(colors[int(np.argmin(counts))])
    assert background in BACKGROUNDS and foreground in COLORS
    ys, xs = np.where(grid == foreground)
    row, col = int(ys.min()), int(xs.min())
    bottom, right = int(ys.max()) + 1, int(xs.max()) + 1
    mask = grid[row:bottom, col:right] == foreground
    mh, mw = mask.shape
    assert 3 <= mh <= 8 and 3 <= mw <= 8
    assert height >= mh + 2 and width >= mw + 2
    border = np.zeros(mask.shape, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    hollow = np.array_equal(mask, border)
    elbow = any(
        np.array_equal(mask, np.logical_or(
            np.arange(mh)[:, None] == end_row,
            np.arange(mw)[None, :] == side_col,
        ))
        for end_row in (0, mh - 1)
        for side_col in (0, mw - 1)
    )
    assert hollow or elbow
    assert np.count_nonzero(grid == foreground) == int(mask.sum())
    return background, foreground, mask, row, col


def _observed_direction(input_grid, output_grid):
    bg, color, mask, row, col = _extract(input_grid)
    out_bg, out_color, out_mask, out_row, out_col = _extract(output_grid)
    assert input_grid.shape == output_grid.shape
    assert bg == out_bg and color == out_color and np.array_equal(mask, out_mask)
    if (row, col) == (out_row, out_col):
        assert np.array_equal(input_grid, output_grid)
        return None
    candidates = [direction for direction in ("top", "right", "bottom", "left")
                  if _destination(direction, row, col, input_grid.shape, mask) == (out_row, out_col)]
    assert len(candidates) == 1
    return candidates[0]


def _predict_from_training(train, test_input):
    observed = {}
    for pair in train:
        color = _extract(pair["input"])[1]
        direction = _observed_direction(pair["input"], pair["output"])
        if direction is not None:
            assert color not in observed or observed[color] == direction
            observed[color] = direction
    assert len(set(observed.values())) == len(observed)
    bg, color, mask, row, col = _extract(test_input)
    if color in observed:
        direction = observed[color]
    else:
        assert len(observed) == 3
        direction = (set(COMMANDS.values()) - set(observed.values())).pop()
    end_row, end_col = _destination(direction, row, col, test_input.shape, mask)
    return _draw(test_input.shape, bg, color, mask, end_row, end_col)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    train_colors, test_colors = _schedule(rng, train_count, test_count)
    pair_rngs = [np.random.default_rng(int(x)) for x in rng.integers(0, 2**63 - 1, size=train_count + test_count)]
    used_inputs = set()
    train = []
    first_geometry = None
    repeated_first_geometry = False
    for index, color in enumerate(train_colors):
        match = (train[0] if index == 1 and train_count >= 3
                 and color != train_colors[0] and int(rng.integers(0, 4)) == 0
                 else None)
        different_from = first_geometry if (
            (index == 1 and match is None) or
            (index == 2 and repeated_first_geometry)
        ) else None
        pair = _make_pair(pair_rngs[index], color, False, used_inputs,
                          different_from, match=match)
        train.append(pair)
        if index == 0:
            _, _, mask, _, _ = _extract(pair["input"])
            first_geometry = (pair["input"].shape, mask.shape, mask.tobytes())
        elif index == 1:
            _, _, mask, _, _ = _extract(pair["input"])
            repeated_first_geometry = (
                pair["input"].shape, mask.shape, mask.tobytes()
            ) == first_geometry
    test = []
    for index, color in enumerate(test_colors):
        test.append(_make_pair(pair_rngs[train_count + index], color,
                               color in train_colors, used_inputs))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    taught = {}
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            bg, color, mask, row, col = _extract(input_grid)
            direction = _observed_direction(input_grid, output_grid)
            assert direction is None or direction == COMMANDS[color]
            expected_row, expected_col = _destination(COMMANDS[color], row, col, input_grid.shape, mask)
            assert np.array_equal(output_grid, _draw(input_grid.shape, bg, color, mask, expected_row, expected_col))
            if direction is None:
                assert split == "test" and color in taught
            if split == "train":
                assert direction is not None
                assert color not in taught or taught[color] == direction
                taught[color] = direction
            assert not _source_copy(input_grid, color, row, col, mask, bg)
            key = (input_grid.shape, input_grid.tobytes())
            assert key not in seen
            seen.add(key)
    assert len(taught) >= 2
    assert len({(pair["input"].shape, _extract(pair["input"])[2].tobytes()) for pair in train}) >= 2
    for pair in test:
        assert np.array_equal(_predict_from_training(train, pair["input"]), pair["output"])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    automatic_puzzles = list(puzzles)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple((pair["input"].shape, pair["input"].tobytes()) for split in ("train", "test") for pair in p[split]) for p in puzzles}) == 42
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        p = generate_puzzle(20000 + seed, *counts)
        assert (len(p["train"]), len(p["test"])) == counts
        puzzles.append(p)
    for seed, counts in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(21000 + seed, *counts)
        assert counts[0] is None or len(p["train"]) == counts[0]
        assert counts[1] is None or len(p["test"]) == counts[1]
        puzzles.append(p)
    for seed, counts in enumerate(((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (2, 1.0))):
        try:
            generate_puzzle(22000 + seed, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    a, b = generate_puzzle(91357, 7, 3), generate_puzzle(91357, 7, 3)
    assert all(np.array_equal(x[side], y[side]) for split in ("train", "test") for x, y in zip(a[split], b[split]) for side in ("input", "output"))
    puzzles.append(a)
    sizes, shapes, backgrounds, positions, distances = set(), set(), set(), set(), set()
    train_colors, test_colors, heldouts, identities = set(), set(), set(), 0
    rectangular = asymmetric = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        taught = {_extract(pair["input"])[1] for pair in puzzle["train"]}
        train_colors.update(taught)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg, color, mask, row, col = _extract(inp)
                _, _, _, end_row, end_col = _extract(out)
                sizes.add(inp.shape)
                shapes.add((mask.shape, mask.tobytes()))
                backgrounds.add(bg)
                positions.add((row, col))
                distances.add(abs(end_row - row) + abs(end_col - col))
                rectangular += mask.shape[0] != mask.shape[1]
                asymmetric += not np.array_equal(mask, np.flip(mask, axis=0)) or not np.array_equal(mask, np.flip(mask, axis=1))
                if split == "test":
                    test_colors.add(color)
                    heldouts.add(color not in taught)
                    identities += np.array_equal(inp, out)
                    assert np.array_equal(_predict_from_training(puzzle["train"], inp), out)
    assert train_colors == test_colors == set(COLORS)
    assert heldouts == {False, True}
    assert len(sizes) >= 40 and len(shapes) >= 25 and len(positions) >= 40
    assert backgrounds == set(BACKGROUNDS)
    assert 0 in distances and max(distances) >= 15 and len(distances) >= 15
    assert rectangular >= 30 and asymmetric >= 10 and identities >= 1

    # Review-specific properties are counted from returned grids, using only
    # the 42 automatic puzzles rather than boundary-count test cases.
    measured = {name: 0 for name in (
        "two_train", "novel_two_train", "all_four", "heldout", "canvas_variety",
        "shape_variety", "background_variety", "exact_color_contrast",
        "source_canvas", "rectangular_canvas", "nonzero_background",
        "square3", "other_hollow", "elbow", "shape_shift", "near",
        "opposite", "long", "identity", "combined",
    )}
    for puzzle in automatic_puzzles:
        pairs = puzzle["train"] + puzzle["test"]
        taught = {_extract(pair["input"])[1] for pair in puzzle["train"]}
        measured["two_train"] += len(puzzle["train"]) == 2
        measured["all_four"] += len(taught) == 4
        measured["heldout"] += any(_extract(pair["input"])[1] not in taught for pair in puzzle["test"])
        measured["canvas_variety"] += len({pair["input"].shape for pair in pairs}) > 1
        measured["background_variety"] += len({_extract(pair["input"])[0] for pair in pairs}) > 1
        train_masks = {(_extract(pair["input"])[2].shape,
                        _extract(pair["input"])[2].tobytes()) for pair in puzzle["train"]}
        all_masks = {(_extract(pair["input"])[2].shape,
                      _extract(pair["input"])[2].tobytes()) for pair in pairs}
        measured["shape_variety"] += len(all_masks) > 1
        same_layout = {}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                bg, color, mask, row, col = _extract(inp)
                _, _, _, end_row, end_col = _extract(out)
                height, width = inp.shape
                mh, mw = mask.shape
                distance = abs(end_row - row) + abs(end_col - col)
                outline = int(mask.sum()) == 2 * (mh + mw) - 4
                square3 = outline and (mh, mw) == (3, 3)
                elbow = not outline
                measured["source_canvas"] += inp.shape == (10, 10)
                measured["rectangular_canvas"] += height != width
                measured["nonzero_background"] += bg != 0
                measured["square3"] += square3
                measured["other_hollow"] += outline and not square3
                measured["elbow"] += elbow
                measured["near"] += distance == 1
                measured["opposite"] += distance == (
                    height - mh if COMMANDS[color] in ("top", "bottom") else width - mw
                )
                measured["long"] += distance >= 10
                measured["identity"] += distance == 0
                measured["combined"] += bg != 0 and height != width and elbow
                layout = (inp.shape, bg, mask.shape, mask.tobytes(), row, col)
                same_layout.setdefault(layout, set()).add(color)
                if split == "test":
                    measured["novel_two_train"] += len(puzzle["train"]) == 2 and color not in taught
                    measured["shape_shift"] += (mask.shape, mask.tobytes()) not in train_masks
        measured["exact_color_contrast"] += any(len(colors) > 1 for colors in same_layout.values())
    assert measured["two_train"] >= 1 and measured["novel_two_train"] == 0
    assert measured["all_four"] >= 5 and measured["heldout"] >= 5
    assert measured["canvas_variety"] == measured["shape_variety"] == 42
    assert measured["background_variety"] >= 30 and measured["exact_color_contrast"] >= 1
    assert measured["source_canvas"] >= 20 and measured["rectangular_canvas"] >= 50
    assert measured["nonzero_background"] >= 50
    assert measured["square3"] >= 30 and measured["other_hollow"] >= 30 and measured["elbow"] >= 10
    assert measured["shape_shift"] >= 15 and measured["combined"] >= 5
    assert measured["near"] >= 20 and measured["opposite"] >= 20 and measured["long"] >= 10
    assert measured["identity"] >= 1
