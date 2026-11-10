"""Solved ARC puzzles with border signals projected onto a solid panel."""

import numpy as np


_SIDES = ("top", "bottom", "left", "right")
_SOURCE_INPUT_ROWS = {
    ("0000700000", "0000000000", "0000000000", "6008880000",
     "0008880000", "0008880002", "0008880000", "3008880000",
     "0000000000", "0000010000"),
    ("0000900000", "0000000000", "0000000000", "0000880000",
     "0000880000", "0000880000", "6000880000", "0000000000",
     "0000000000", "0000040000"),
    ("0004000000", "0000000000", "0000000000", "0008880006",
     "3008880000", "0008880000", "2008880000", "0008880002",
     "0000000000", "0007000000"),
    ("0006020000", "0000000000", "0000000000", "9088880000",
     "0088880007", "3088880000", "4088880000", "0000000000",
     "0000000000", "0000600000"),
}


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _counts(seed, num_train, num_test):
    if num_train is not None:
        if not _integer(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 to 10")
        num_train = int(num_train)
    if num_test is not None:
        if not _integer(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 to 4")
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


def _key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _rows(grid):
    return tuple("".join(str(int(v)) for v in row) for row in grid)


def _dimension(rng, regime):
    # Height and width are drawn independently, including small and large panels.
    band = float(rng.random())
    if band < 0.30:
        low, high = 2, 4
    elif band < (0.68 if regime == "source_train" else 0.61):
        low, high = 5, 9
    elif band < (0.91 if regime == "source_train" else 0.82):
        low, high = 10, 16
    else:
        low, high = 17, 26
    return int(rng.integers(low, high + 1))


def _margins(rng, panel_length):
    capacity = 30 - panel_length
    assert capacity >= 4
    if rng.random() < 0.50:
        upper = min(capacity, 8)
    else:
        upper = capacity
    extra = int(rng.integers(4, upper + 1))
    before = int(rng.integers(2, extra - 1))
    return before, extra - before


def _signal_count(rng, perimeter, minimum, regime):
    draw = float(rng.random())
    if draw < (0.24 if regime == "source_train" else 0.28):
        low, high = 1, min(2, perimeter)
    elif draw < (0.68 if regime == "source_train" else 0.58):
        low, high = 3, min(8, perimeter)
    else:
        low = min(perimeter, max(5, (perimeter + 2) // 3))
        high = perimeter
    low = max(minimum, low)
    high = max(minimum, high)
    return int(rng.integers(low, high + 1))


def _destinations(panel_height, panel_width):
    return {
        "top": [(0, c) for c in range(panel_width)],
        "bottom": [(panel_height - 1, c) for c in range(panel_width)],
        "left": [(r, 0) for r in range(panel_height)],
        "right": [(r, panel_width - 1) for r in range(panel_height)],
    }


def _sites(rng, panel_height, panel_width, required, count):
    candidates = _destinations(panel_height, panel_width)
    requirements = [side for side in _SIDES for _ in range(required.get(side, 0))]
    requirements.sort(key=lambda side: len(candidates[side]))
    selected = []
    used = set()

    def place(index):
        if index == len(requirements):
            return True
        side = requirements[index]
        for choice in rng.permutation(len(candidates[side])):
            destination = candidates[side][int(choice)]
            if destination in used:
                continue
            used.add(destination)
            selected.append((side, destination))
            if place(index + 1):
                return True
            selected.pop()
            used.remove(destination)
        return False

    assert place(0)
    all_sites = [
        (side, destination)
        for side in _SIDES
        for destination in candidates[side]
        if destination not in used
    ]
    rng.shuffle(all_sites)
    for side, destination in all_sites:
        if len(selected) == count:
            break
        if destination not in used:
            selected.append((side, destination))
            used.add(destination)
    assert len(selected) == count
    rng.shuffle(selected)
    return selected


def _role_colors(rng, previous):
    choices = [
        (background, panel)
        for background in range(10)
        for panel in range(10)
        if background != panel
        and (previous is None or (
            background != previous[0] and panel != previous[1]
        ))
    ]
    return choices[int(rng.integers(len(choices)))]


def _colors(rng, background, panel, count, minimum_distinct):
    available = [c for c in range(10) if c not in (background, panel)]
    maximum = min(8, count)
    distinct = int(rng.integers(minimum_distinct, maximum + 1))
    chosen = [int(c) for c in rng.choice(available, size=distinct, replace=False)]
    colors = chosen + [chosen[int(rng.integers(distinct))] for _ in range(count - distinct)]
    rng.shuffle(colors)
    return colors


def _pair(rng, regime, required, rich_palette, previous_roles, previous_shape):
    panel_height = _dimension(rng, regime)
    panel_width = _dimension(rng, regime)
    if previous_shape == (panel_height, panel_width):
        panel_width = 2 + (panel_width - 1) % 25
    top, bottom_margin = _margins(rng, panel_height)
    left, right_margin = _margins(rng, panel_width)
    height = top + panel_height + bottom_margin
    width = left + panel_width + right_margin
    background, panel = _role_colors(rng, previous_roles)
    perimeter = 2 * panel_height + 2 * panel_width - 4
    minimum = sum(required.values())
    if rich_palette:
        minimum = max(3, minimum)
    count = _signal_count(rng, perimeter, minimum, regime)
    sites = _sites(rng, panel_height, panel_width, required, count)
    colors = _colors(rng, background, panel, count, 3 if rich_palette else 1)

    input_grid = np.full((height, width), background, dtype=np.int64)
    input_grid[top:top + panel_height, left:left + panel_width] = panel
    output_grid = input_grid.copy()
    for (side, (relative_row, relative_column)), color in zip(sites, colors):
        row = top + relative_row
        column = left + relative_column
        if side == "top":
            signal = (0, column)
        elif side == "bottom":
            signal = (height - 1, column)
        elif side == "left":
            signal = (row, 0)
        else:
            signal = (row, width - 1)
        input_grid[signal] = color
        output_grid[signal] = color
        output_grid[row, column] = color
    return {"input": input_grid, "output": output_grid}


def _analyze(pair):
    source = pair["input"]
    output = pair["output"]
    assert source.shape == output.shape
    height, width = source.shape
    assert height >= 6 and width >= 6

    # Corners are exterior background even when the panel is the largest region.
    background = int(source[0, 0])
    assert all(int(source[r, c]) == background for r, c in (
        (0, width - 1), (height - 1, 0), (height - 1, width - 1)
    ))
    interior = source[1:-1, 1:-1]
    occupied = np.argwhere(interior != background)
    assert len(occupied) >= 4
    top = int(occupied[:, 0].min()) + 1
    bottom = int(occupied[:, 0].max()) + 1
    left = int(occupied[:, 1].min()) + 1
    right = int(occupied[:, 1].max()) + 1
    assert top >= 2 and left >= 2
    assert bottom <= height - 3 and right <= width - 3
    panel = int(source[top, left])
    assert panel != background
    assert len(occupied) == (bottom - top + 1) * (right - left + 1)
    assert np.all(source[top:bottom + 1, left:right + 1] == panel)

    expected = source.copy()
    destinations = {}
    side_counts = {side: 0 for side in _SIDES}
    signal_colors = set()
    occupied_sides = []
    for row, column in np.argwhere(source != background):
        row, column = int(row), int(column)
        if top <= row <= bottom and left <= column <= right:
            continue
        assert int(row == 0) + int(row == height - 1) + int(column == 0) + int(column == width - 1) == 1
        color = int(source[row, column])
        assert color not in (background, panel)
        if row == 0:
            side, destination = "top", (top, column)
            assert left <= column <= right
        elif row == height - 1:
            side, destination = "bottom", (bottom, column)
            assert left <= column <= right
        elif column == 0:
            side, destination = "left", (row, left)
            assert top <= row <= bottom
        else:
            side, destination = "right", (row, right)
            assert top <= row <= bottom
        assert destination not in destinations or destinations[destination] == color
        destinations[destination] = color
        expected[destination] = color
        side_counts[side] += 1
        signal_colors.add(color)
        occupied_sides.append(side)
    assert occupied_sides
    assert np.array_equal(expected, output)
    assert not np.array_equal(source, output)
    return {
        "background": background,
        "panel": panel,
        "panel_shape": (bottom - top + 1, right - left + 1),
        "panel_bbox": (top, left, bottom, right),
        "canvas_shape": (height, width),
        "side_counts": side_counts,
        "signal_count": len(occupied_sides),
        "signal_colors": signal_colors,
        "destination_count": len(destinations),
        "panel_fraction": ((bottom - top + 1) * (right - left + 1)) / (height * width),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4
    assert train_count > test_count
    analyses = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            input_key = _key(pair["input"])
            pair_key = (input_key, _key(pair["output"]))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _rows(pair["input"]) not in _SOURCE_INPUT_ROWS
            analyses[split].append(_analyze(pair))

    train = analyses["train"]
    all_pairs = train + analyses["test"]
    assert len({x["panel_shape"] for x in all_pairs}) >= 2
    assert len({x["background"] for x in all_pairs}) >= 2
    assert len({x["panel"] for x in all_pairs}) >= 2
    assert all(any(x["side_counts"][s] for x in train) for s in _SIDES)
    assert any(max(x["side_counts"].values()) >= 2 for x in train)
    assert any(len(x["signal_colors"]) >= 3 for x in train)
    training_structures = {
        (x["panel_shape"], tuple(x["side_counts"].values()), len(x["signal_colors"]))
        for x in train
    }
    for x in analyses["test"]:
        assert (x["panel_shape"], tuple(x["side_counts"].values()),
                len(x["signal_colors"])) not in training_structures


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # The sole puzzle-wide convention is geometric inward projection with literal
    # color copying. Arrange training evidence without fixing its pair order.
    requirements = [{} for _ in range(train_count)]
    sides = [str(_SIDES[int(i)]) for i in rng.permutation(4)]
    positions = [int(i) for i in rng.permutation(train_count)]
    for index, side in enumerate(sides):
        slot = positions[index % min(4, train_count)]
        requirements[slot][side] = 1
    rich_slot = positions[0]
    repeated_side = sides[0]
    requirements[rich_slot][repeated_side] += 1

    puzzle = {"train": [], "test": []}
    seen = set()
    previous_roles = None
    previous_shape = None
    training_structures = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            regime = "source_train" if split == "train" else "target_test"
            needed = requirements[index] if split == "train" else {}
            for _ in range(30):
                pair = _pair(rng, regime, needed, split == "train" and index == rich_slot,
                             previous_roles, previous_shape)
                key = _key(pair["input"])
                if key in seen or _rows(pair["input"]) in _SOURCE_INPUT_ROWS:
                    continue
                analysis = _analyze(pair)
                structure = (analysis["panel_shape"],
                             tuple(analysis["side_counts"].values()),
                             len(analysis["signal_colors"]))
                if split == "test" and structure in training_structures:
                    continue
                break
            else:
                raise RuntimeError("Could not build a distinct valid pair")
            puzzle[split].append(pair)
            seen.add(key)
            previous_roles = (analysis["background"], analysis["panel"])
            previous_shape = analysis["panel_shape"]
            if split == "train":
                training_structures.add(structure)
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple((split, _key(pair["input"]), _key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def self_test() -> None:
    train_counts, test_counts, signatures = set(), set(), set()
    panel_shapes, canvas_shapes, side_patterns = set(), set(), set()
    backgrounds, panels, signal_counts, palettes = set(), set(), set(), set()
    aspects, fractions, margin_tuples = set(), [], set()
    rare = {name: 0 for name in (
        "single_side", "one_signal", "two_signals", "color_repeat", "all_distinct",
        "zero_background", "eight_panel", "large_panel", "large_canvas",
        "test_sparse", "test_smaller", "test_unseen_color", "same_aspect",
        "reversed_aspect", "corner_destination", "noncorner_destination",
        "small_canvas", "wide_canvas", "tall_canvas", "adjacent_signals",
        "same_color_adjacent", "test_single_side", "test_all_sides",
        "test_simple_palette", "test_all_distinct", "training_rich_palette",
        "source_role_assignment", "one_color_many", "panel_dominates",
        "large_dense", "small_sparse",
    )}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train = [_analyze(pair) for pair in puzzle["train"]]
        test = [_analyze(pair) for pair in puzzle["test"]]
        train_colors = set().union(*(x["signal_colors"] for x in train))
        train_min_area = min(x["panel_shape"][0] * x["panel_shape"][1] for x in train)
        train_max_signals = max(x["signal_count"] for x in train)
        train_aspects = {int(np.sign(h - w)) for h, w in (x["panel_shape"] for x in train)}
        for x in test:
            rare["test_sparse"] += x["signal_count"] < train_max_signals
            rare["test_smaller"] += x["panel_shape"][0] * x["panel_shape"][1] < train_min_area
            rare["test_unseen_color"] += bool(x["signal_colors"] - train_colors)
            aspect = int(np.sign(x["panel_shape"][0] - x["panel_shape"][1]))
            rare["same_aspect"] += aspect in train_aspects
            rare["reversed_aspect"] += any(aspect == -a and a != 0 for a in train_aspects)
            rare["test_single_side"] += sum(v > 0 for v in x["side_counts"].values()) == 1
            rare["test_all_sides"] += all(x["side_counts"].values())
            rare["test_simple_palette"] += len(x["signal_colors"]) <= 2
            rare["test_all_distinct"] += x["signal_count"] == len(x["signal_colors"])
        rare["training_rich_palette"] += any(len(x["signal_colors"]) >= 5 for x in train)
        for split, analyses in (("train", train), ("test", test)):
            for pair, x in zip(puzzle[split], analyses):
                h, w = x["panel_shape"]
                panel_shapes.add((h, w))
                canvas_shapes.add(x["canvas_shape"])
                side_patterns.add(tuple(x["side_counts"].values()))
                backgrounds.add(x["background"])
                panels.add(x["panel"])
                signal_counts.add(x["signal_count"])
                palettes.add(len(x["signal_colors"]))
                aspects.add(int(np.sign(h - w)))
                fractions.append(x["panel_fraction"])
                canvas_height, canvas_width = x["canvas_shape"]
                top, left, bottom, right = x["panel_bbox"]
                margin_tuples.add((top, canvas_height - bottom - 1,
                                   left, canvas_width - right - 1))
                rare["single_side"] += sum(v > 0 for v in x["side_counts"].values()) == 1
                rare["one_signal"] += x["signal_count"] == 1
                rare["two_signals"] += x["signal_count"] == 2
                rare["color_repeat"] += x["signal_count"] > len(x["signal_colors"])
                rare["all_distinct"] += x["signal_count"] == len(x["signal_colors"])
                rare["zero_background"] += x["background"] == 0
                rare["eight_panel"] += x["panel"] == 8
                rare["large_panel"] += x["panel_fraction"] > 0.45
                rare["large_canvas"] += max(x["canvas_shape"]) >= 27
                rare["small_canvas"] += canvas_height <= 10 and canvas_width <= 10
                rare["wide_canvas"] += canvas_width >= 2 * canvas_height
                rare["tall_canvas"] += canvas_height >= 2 * canvas_width
                rare["source_role_assignment"] += (x["background"], x["panel"]) == (0, 8)
                rare["one_color_many"] += len(x["signal_colors"]) == 1 and x["signal_count"] >= 7
                rare["panel_dominates"] += (
                    x["panel_shape"][0] * x["panel_shape"][1]
                    > canvas_height * canvas_width - x["panel_shape"][0] * x["panel_shape"][1] - x["signal_count"]
                )
                rare["large_dense"] += x["panel_fraction"] > 0.45 and x["signal_count"] >= 10
                rare["small_sparse"] += (x["panel_shape"][0] <= 4 and x["panel_shape"][1] <= 4
                                          and x["signal_count"] <= 2)
                source = pair["input"]
                lines = (source[0, left:right + 1], source[-1, left:right + 1],
                         source[top:bottom + 1, 0], source[top:bottom + 1, -1])
                rare["adjacent_signals"] += any(
                    np.any((line[:-1] != x["background"]) & (line[1:] != x["background"]))
                    for line in lines
                )
                rare["same_color_adjacent"] += any(
                    np.any((line[:-1] == line[1:]) & (line[:-1] != x["background"]))
                    for line in lines
                )
                changed = np.argwhere(pair["input"] != pair["output"])
                rare["corner_destination"] += any(
                    int(r) in (top, bottom) and int(c) in (left, right)
                    for r, c in changed)
                rare["noncorner_destination"] += any(
                    not (int(r) in (top, bottom) and int(c) in (left, right))
                    for r, c in changed)
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(panel_shapes) >= 55 and len(canvas_shapes) >= 70
    assert len(side_patterns) >= 25 and len(signal_counts) >= 12
    assert len(backgrounds) == 10 and len(panels) == 10
    assert len(palettes) >= 7 and aspects == {-1, 0, 1}
    assert len(margin_tuples) >= 100
    assert min(fractions) < 0.08 and max(fractions) > 0.45
    assert all(value > 0 for value in rare.values())
    assert _signature(generate_puzzle(314159)) == _signature(generate_puzzle(314159))
    assert _signature(generate_puzzle(271828, 7, 3)) == _signature(generate_puzzle(271828, 7, 3))
    assert _signature(generate_puzzle(1001, 6, 2)) != _signature(generate_puzzle(1002, 6, 2))
    for index, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(5000 + index, nt, ns)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ns)
        validate_generated_puzzle(puzzle)
    for index, (nt, ns) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(6000 + index, nt, ns)
        assert nt is None or len(puzzle["train"]) == nt
        assert ns is None or len(puzzle["test"]) == ns
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(7000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair count was accepted")
