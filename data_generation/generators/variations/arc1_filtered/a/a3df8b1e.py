import numpy as np


_CORNERS = ("TL", "TR", "BL", "BR")


def _resolve_counts(seed, rng, num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
    if num_train is None and num_test is None:
        num_train = 2 + int(seed) % 9
        cap = min(4, num_train - 1)
        num_test = 1 + (int(seed) // 9 + int(seed)) % cap
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _palette(rng):
    if rng.random() < 0.28:
        return int(rng.integers(1, 10)), 0
    colors = rng.choice(10, size=2, replace=False)
    return int(colors[0]), int(colors[1])


def _dimensions(axis, travel, breadth):
    return (travel, breadth) if axis == "V" else (breadth, travel)


def _corner_cell(corner, height, width):
    return (0 if corner[0] == "T" else height - 1,
            0 if corner[1] == "L" else width - 1)


def _make_pair(axis, corner, travel, breadth, palette):
    height, width = _dimensions(axis, travel, breadth)
    background, foreground = palette
    inp = np.full((height, width), background, dtype=np.int8)
    out = np.full((height, width), background, dtype=np.int8)
    seed_row, seed_col = _corner_cell(corner, height, width)
    inp[seed_row, seed_col] = foreground
    period = 2 * (breadth - 1)
    for step in range(travel):
        phase = step % period
        side = min(phase, period - phase)
        if axis == "V":
            row = seed_row + step * (1 if corner[0] == "T" else -1)
            col = side if corner[1] == "L" else width - 1 - side
        else:
            col = seed_col + step * (1 if corner[1] == "L" else -1)
            row = side if corner[0] == "T" else height - 1 - side
        out[row, col] = foreground
    return {"input": inp, "output": out}


def _category_travels(breadth, category):
    if category == "short":
        return range(2, breadth)
    if category == "turn":
        return range(breadth + 1, min(2 * breadth - 1, 31))
    if category == "return":
        return range(2 * breadth - 1, min(4 * breadth - 3, 31))
    if category == "multi":
        return range(4 * (breadth - 1) + 1, 31)
    return range(2, 31)


def _source_copy(axis, corner, travel, breadth, palette):
    return (axis == "V" and corner == "BL" and travel == 10
            and breadth in (2, 3, 4, 5) and palette == (0, 1))


def _choose_pair(rng, axis, corner, palette, breadth_options, category,
                 used_inputs, fixed_travel=None, excluded_travels=()):
    for chosen_category in (category, "any"):
        for breadth in breadth_options:
            travels = ([fixed_travel] if fixed_travel is not None
                       else list(_category_travels(breadth, chosen_category)))
            candidates = []
            for travel in travels:
                if not 2 <= travel <= 30 or travel == breadth:
                    continue
                if travel in excluded_travels:
                    continue
                if _source_copy(axis, corner, travel, breadth, palette):
                    continue
                shape = _dimensions(axis, travel, breadth)
                key = (shape, corner, palette)
                if key not in used_inputs:
                    candidates.append((travel, key))
            if candidates:
                travel, key = candidates[int(rng.integers(len(candidates)))]
                used_inputs.add(key)
                return _make_pair(axis, corner, travel, breadth, palette)
    raise AssertionError("No unused dimensions in requested regime")


def _breadth_options(rng, preferred, low=2, high=24):
    others = [x for x in range(low, high + 1) if x != preferred]
    rng.shuffle(others)
    return [preferred] + others


def _broad_breadth(rng):
    branch = rng.random()
    if branch < 0.08:
        return 2
    if branch < 0.68:
        return int(rng.integers(3, 9))
    if branch < 0.91:
        return int(rng.integers(9, 15))
    return int(rng.integers(15, 21))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)

    # A puzzle's corner-to-axis convention is sampled once. Test corners occur
    # in training, and each corner fixes the inward reflection direction.
    mode = str(rng.choice(("vertical", "horizontal", "mixed"), p=(0.67, 0.16, 0.17)))
    ladder = mode == "vertical" and rng.random() < 0.23
    stable_palette = ladder or rng.random() < 0.38
    shared_palette = _palette(rng) if stable_palette else None
    if ladder and shared_palette == (0, 1):
        shared_palette = (1, 0)

    corners = list(_CORNERS)
    rng.shuffle(corners)
    if mode == "mixed":
        axis_by_corner = {corners[0]: "V", corners[1]: "H"}
        for corner in corners[2:]:
            axis_by_corner[corner] = str(rng.choice(("V", "H")))
    else:
        axis_by_corner = {corner: ("V" if mode == "vertical" else "H")
                          for corner in corners}

    if ladder:
        train_corners = ["BL"] * train_count
    else:
        train_corners = corners[:2]
        for index in range(2, train_count):
            train_corners.append(corners[int(rng.integers(min(index + 1, 4)))])

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_specs = []
    first_palette = None
    for index, corner in enumerate(train_corners):
        axis = axis_by_corner[corner]
        palette = shared_palette if stable_palette else _palette(rng)
        if not stable_palette and index == 1 and palette == first_palette:
            palette = (palette[1], palette[0])
        if index == 0:
            first_palette = palette

        if ladder:
            preferred = (3, 4, 2)[index] if index < 3 else min(20, index + 2)
            options = _breadth_options(rng, preferred, 2, 20)
            pair = _choose_pair(rng, axis, corner, palette, options, "any",
                                used_inputs, fixed_travel=10)
        else:
            if index == 0:
                preferred = int(rng.integers(3, 8))
                category = "return"
            elif index == 1:
                preferred = int(rng.integers(4, 11))
                category = "return" if mode == "mixed" else "turn"
            else:
                preferred = _broad_breadth(rng)
                category = str(rng.choice(("short", "turn", "return", "multi"),
                                          p=(0.23, 0.26, 0.31, 0.20)))
            options = _breadth_options(rng, preferred)
            pair = _choose_pair(rng, axis, corner, palette, options, category,
                                used_inputs)
        puzzle["train"].append(pair)
        breadth = pair["input"].shape[1 if axis == "V" else 0]
        travel = pair["input"].shape[0 if axis == "V" else 1]
        train_specs.append((corner, axis, breadth, travel))

    demonstrated_corners = sorted(set(train_corners))
    for index in range(test_count):
        corner = str(rng.choice(demonstrated_corners))
        axis = axis_by_corner[corner]
        palette = shared_palette if stable_palette else _palette(rng)
        same_axis = [breadth for _, a, breadth, _ in train_specs if a == axis]
        if ladder:
            preferred = min(30, max(same_axis) + 1 + index)
            breadth_options = [preferred] + list(range(preferred + 1, 31))
            pair = _choose_pair(rng, axis, corner, palette, breadth_options,
                                "any", used_inputs, fixed_travel=10)
        else:
            relation = (int(seed) + index) % 3 if index == 0 else int(rng.integers(3))
            if relation == 0:
                preferred = int(rng.choice(same_axis))
                breadth_options = [preferred] + [b for b in sorted(set(same_axis)) if b != preferred]
            elif relation == 1:
                low = max(same_axis) + 1
                breadth_options = list(range(low, 27))
            else:
                high = max(same_axis) - 1
                breadth_options = list(range(high, 1, -1))
                if not breadth_options:
                    breadth_options = [2]
            category = str(rng.choice(("short", "turn", "return", "multi"),
                                      p=(0.29, 0.27, 0.27, 0.17)))
            excluded = (tuple(travel for _, a, b, travel in train_specs
                              if a == axis and b == breadth_options[0])
                        if relation == 0 else ())
            pair = _choose_pair(rng, axis, corner, palette, breadth_options,
                                category, used_inputs, excluded_travels=excluded)
        puzzle["test"].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _analyze_pair(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 2 and sorted(counts.tolist()) == [1, inp.size - 1]
    foreground = int(values[int(np.argmin(counts))])
    background = int(values[int(np.argmax(counts))])
    positions = np.argwhere(inp == foreground)
    assert len(positions) == 1
    row, col = (int(x) for x in positions[0])
    height, width = inp.shape
    assert row in (0, height - 1) and col in (0, width - 1)
    corner = ("T" if row == 0 else "B") + ("L" if col == 0 else "R")
    valid_axes = []
    for axis in ("V", "H"):
        travel = height if axis == "V" else width
        breadth = width if axis == "V" else height
        if breadth < 2 or travel < 2:
            continue
        expected = np.full(inp.shape, background, dtype=np.int64)
        side = 0
        direction = 1
        for step in range(travel):
            if axis == "V":
                r = row + step * (1 if row == 0 else -1)
                c = side if col == 0 else width - 1 - side
            else:
                c = col + step * (1 if col == 0 else -1)
                r = side if row == 0 else height - 1 - side
            expected[r, c] = foreground
            if side == breadth - 1:
                direction = -1
            elif side == 0:
                direction = 1
            side += direction
        if np.array_equal(out, expected):
            valid_axes.append(axis)
    assert valid_axes
    return corner, valid_axes, (background, foreground)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs, seen_pairs = set(), set()
    training_axes = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            corner, axes, palette = _analyze_pair(pair)
            assert len(axes) == 1
            axis = axes[0]
            height, width = pair["input"].shape
            breadth = width if axis == "V" else height
            travel = height if axis == "V" else width
            assert breadth >= 2 and travel >= 2
            assert not _source_copy(axis, corner, travel, breadth, palette)
            inp_key = _grid_key(pair["input"])
            pair_key = (inp_key, _grid_key(pair["output"]))
            assert inp_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
            if split == "train":
                if corner in training_axes:
                    assert training_axes[corner] == axis
                training_axes[corner] = axis
            else:
                assert corner in training_axes and training_axes[corner] == axis

    # Each demonstrated axis contains a visible return to the seeded side.
    for axis in set(training_axes.values()):
        assert any(
            (pair["input"].shape[0 if axis == "V" else 1]
             > 2 * (pair["input"].shape[1 if axis == "V" else 0] - 1))
            for pair in puzzle["train"]
            if _analyze_pair(pair)[1] == [axis]
        )


def _same_puzzle(left, right):
    return all(
        len(left[split]) == len(right[split]) and all(
            np.array_equal(a[role], b[role])
            for a, b in zip(left[split], right[split])
            for role in ("input", "output")
        ) for split in ("train", "test")
    )


def _predict_from_training(train_pairs, test_input):
    by_corner = {}
    for pair in train_pairs:
        corner, axes, _ = _analyze_pair(pair)
        assert len(axes) == 1
        assert corner not in by_corner or by_corner[corner] == axes[0]
        by_corner[corner] = axes[0]
    values, counts = np.unique(test_input, return_counts=True)
    foreground = int(values[int(np.argmin(counts))])
    background = int(values[int(np.argmax(counts))])
    row, col = (int(x) for x in np.argwhere(test_input == foreground)[0])
    corner = ("T" if row == 0 else "B") + ("L" if col == 0 else "R")
    axis = by_corner[corner]
    height, width = test_input.shape
    travel = height if axis == "V" else width
    breadth = width if axis == "V" else height
    return _make_pair(axis, corner, travel, breadth, (background, foreground))["output"]


def self_test() -> None:
    stats = {name: set() for name in ("train_counts", "test_counts", "shapes",
                                     "palettes", "corners", "axes", "visibility",
                                     "structures")}
    stable_palettes = varied_palettes = mixed_axes = same_width = narrower = wider = 0
    common_height = foreground_zero = nonzero_background = width_two = 0
    wide_training = varied_corners = varied_visibility = varied_dimensions = 0
    combined_symmetry_palette = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        validate_generated_puzzle(puzzle)
        stats["train_counts"].add(len(puzzle["train"]))
        stats["test_counts"].add(len(puzzle["test"]))
        train_palettes, train_axes, train_corners = set(), set(), set()
        puzzle_visibilities = set()
        puzzle_shapes = set()
        train_widths = [p["input"].shape[1] for p in puzzle["train"]]
        test_widths = [p["input"].shape[1] for p in puzzle["test"]]
        if all(p["input"].shape[0] == 10 for split in puzzle.values() for p in split):
            common_height += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                corner, axes, palette = _analyze_pair(pair)
                axis = axes[0]
                height, width = pair["input"].shape
                travel, breadth = ((height, width) if axis == "V" else (width, height))
                if split == "train":
                    train_palettes.add(palette)
                    train_axes.add(axis)
                    train_corners.add(corner)
                    width_two += width == 2
                    wide_training += width >= 5
                foreground_zero += palette[1] == 0
                nonzero_background += palette[0] != 0
                puzzle_shapes.add((height, width))
                stats["shapes"].add((height, width))
                stats["palettes"].add(palette)
                stats["corners"].add(corner)
                stats["axes"].add(axis)
                visibility = ("short" if travel < breadth else
                              "turn" if travel < 2 * breadth - 1 else
                              "return" if travel <= 4 * (breadth - 1) else "multi")
                stats["visibility"].add(visibility)
                puzzle_visibilities.add(visibility)
                stats["structures"].add((height, width, corner, axis,
                                         tuple(np.argwhere(pair["output"] == palette[1]).flat)))
        stable_palettes += len(train_palettes) == 1
        varied_palettes += len(train_palettes) > 1
        mixed_axes += len(train_axes) > 1
        varied_corners += len(train_corners) > 1
        varied_visibility += len(puzzle_visibilities) > 1
        varied_dimensions += len(puzzle_shapes) > 1
        combined_symmetry_palette += len(train_axes) > 1 and len(train_palettes) > 1
        same_width += any(
            test["input"].shape[1] in train_widths
            and all(test["input"].shape[0] != train["input"].shape[0]
                    for train in puzzle["train"]
                    if train["input"].shape[1] == test["input"].shape[1])
            for test in puzzle["test"]
        )
        narrower += min(test_widths) < max(train_widths)
        wider += max(test_widths) > max(train_widths)
        for test_pair in puzzle["test"]:
            training = [{key: value.copy() for key, value in pair.items()}
                        for pair in puzzle["train"]]
            prediction = _predict_from_training(training, test_pair["input"].copy())
            assert np.array_equal(prediction, test_pair["output"])

    assert len(stats["train_counts"]) >= 4 and len(stats["test_counts"]) >= 3
    assert len(stats["shapes"]) >= 90 and len(stats["structures"]) >= 130
    assert len(stats["palettes"]) >= 20 and stats["corners"] == set(_CORNERS)
    assert stats["axes"] == {"V", "H"} and len(stats["visibility"]) == 4
    assert stable_palettes >= 4 and varied_palettes >= 10 and mixed_axes >= 2
    assert varied_corners >= 10 and varied_visibility >= 20 and varied_dimensions >= 35
    assert combined_symmetry_palette >= 1
    assert same_width >= 4 and narrower >= 4 and wider >= 4
    assert common_height >= 1 and foreground_zero >= 10 and nonzero_background >= 100
    assert width_two < 60 and wide_training >= 80

    for seed, requested_train, requested_test in ((100, 2, 1), (101, 10, 1),
                                                   (102, 10, 4), (103, None, 1),
                                                   (104, None, 4), (105, 2, None),
                                                   (106, 10, None)):
        puzzle = generate_puzzle(seed, requested_train, requested_test)
        assert _same_puzzle(puzzle, generate_puzzle(seed, requested_train, requested_test))
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
        validate_generated_puzzle(puzzle)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(777, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Illegal counts accepted: {arguments}")
    broken = generate_puzzle(2026, 2, 1)
    test = broken["test"][0]
    corner, axes, palette = _analyze_pair(test)
    row, col = _corner_cell(corner, *test["input"].shape)
    test["output"][row, col] = palette[0]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupt output was accepted")
