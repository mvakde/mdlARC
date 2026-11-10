import numpy as np


_DIRECTIONS = ("T", "B", "L", "R")
_SOURCE_DESCRIPTORS = {
    (22, 19, 8, ((8, 8, ("L", "R", "T")), (14, 2, ("B",)))),
    (18, 17, 2, ((2, 2, ("B", "R")),)),
    (22, 22, 3, ((5, 6, ("L", "T")), (11, 12, ("B", "R")))),
    (24, 24, 4, ((6, 6, ("B", "L", "T")), (12, 13, ("B", "T")))),
}


def _pattern(side):
    pattern = np.ones((side, side), dtype=bool)
    pattern[1:-1, 1:-1] = False
    return pattern


def _grid_key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _resolve_counts(seed, num_train, num_test):
    def checked(value, lower, upper):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("count must be an integer or None")
        value = int(value)
        if not lower <= value <= upper:
            raise ValueError("count is outside its legal range")
        return value

    train = checked(num_train, 2, 10)
    test = checked(num_test, 1, 4)
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


def _analyze_input(grid):
    """Recover roles, sources, and commands solely from a displayed input."""
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    height, width = grid.shape
    assert 1 <= height <= 30 and 1 <= width <= 30
    background = int(grid[0, 0])
    assert int(grid[0, -1]) == int(grid[-1, 0]) == int(grid[-1, -1]) == background
    perimeter = np.zeros(grid.shape, dtype=bool)
    perimeter[0, :] = perimeter[-1, :] = True
    perimeter[:, 0] = perimeter[:, -1] = True
    edge_colors = {int(x) for x in grid[perimeter]} - {background}
    assert len(edge_colors) == 1
    guide = edge_colors.pop()
    assert not np.any((grid == guide) & ~perimeter)
    contents = {int(x) for x in np.unique(grid)} - {background, guide}
    assert contents

    possibilities = []
    for side in (3, 4, 5):
        if side + 2 > min(height, width):
            continue
        pattern = _pattern(side)
        anchors = []
        for row in range(1, height - side):
            for col in range(1, width - side):
                block = grid[row:row + side, col:col + side]
                color = int(block[0, 0])
                if color in contents and np.array_equal(block == color, pattern) and np.all(block[~pattern] == background):
                    anchors.append((row, col, color))
        if not anchors:
            continue
        covered = np.zeros(grid.shape, dtype=bool)
        for row, col, color in anchors:
            area = covered[row:row + side, col:col + side]
            if np.any(area & pattern):
                break
            area |= pattern
        else:
            if np.array_equal(covered, np.isin(grid, list(contents))):
                possibilities.append((side, anchors))
    assert len(possibilities) == 1
    side, anchors = possibilities[0]
    assert len({row for row, _, _ in anchors}) == len(anchors)
    assert len({col for _, col, _ in anchors}) == len(anchors)

    directions = {}
    guide_mask = np.zeros(grid.shape, dtype=bool)
    for row, col, color in anchors:
        found = set()
        for direction, cells in (
            ("T", grid[0, col:col + side]),
            ("B", grid[-1, col:col + side]),
            ("L", grid[row:row + side, 0]),
            ("R", grid[row:row + side, -1]),
        ):
            if np.all(cells == guide):
                found.add(direction)
                if direction == "T":
                    guide_mask[0, col:col + side] = True
                elif direction == "B":
                    guide_mask[-1, col:col + side] = True
                elif direction == "L":
                    guide_mask[row:row + side, 0] = True
                else:
                    guide_mask[row:row + side, -1] = True
        directions[(row, col)] = found
    assert np.array_equal(grid == guide, guide_mask)
    assert any(directions.values())
    return {
        "height": height, "width": width, "side": side,
        "background": background, "guide": guide,
        "anchors": sorted(anchors), "directions": directions,
    }


def _paint(assigned, row, col, color, background, side):
    pattern = _pattern(side)
    height, width = assigned.shape
    for dr in range(side):
        rr = row + dr
        if not 0 <= rr < height:
            continue
        for dc in range(side):
            cc = col + dc
            if not 0 <= cc < width:
                continue
            value = color if pattern[dr, dc] else background
            assert assigned[rr, cc] in (-1, value)
            assigned[rr, cc] = value


def _expected_output(grid, analysis=None):
    if analysis is None:
        analysis = _analyze_input(grid)
    height, width = grid.shape
    side = analysis["side"]
    background = analysis["background"]
    assigned = np.full(grid.shape, -1, dtype=np.int16)
    for row, col, color in analysis["anchors"]:
        _paint(assigned, row, col, color, background, side)
    for row, col, color in analysis["anchors"]:
        for direction in analysis["directions"][(row, col)]:
            if direction == "T":
                origins = ((rr, col) for rr in range(row - side, -side, -side))
            elif direction == "B":
                origins = ((rr, col) for rr in range(row + side, height, side))
            elif direction == "L":
                origins = ((row, cc) for cc in range(col - side, -side, -side))
            else:
                origins = ((row, cc) for cc in range(col + side, width, side))
            for rr, cc in origins:
                _paint(assigned, rr, cc, color, background, side)
    output = np.full(grid.shape, background, dtype=np.int8)
    occupied = assigned >= 0
    output[occupied] = assigned[occupied]
    return output


def _has_partial_copy(analysis):
    side = analysis["side"]
    height, width = analysis["height"], analysis["width"]
    for row, col, _ in analysis["anchors"]:
        directions = analysis["directions"][(row, col)]
        if (("T" in directions and row % side) or
            ("B" in directions and (height - row) % side) or
            ("L" in directions and col % side) or
            ("R" in directions and (width - col) % side)):
            return True
    return False


def _source_copy(analysis):
    if analysis["side"] != 4 or analysis["background"] != 0 or analysis["guide"] != 1:
        return False
    colors = {color for _, _, color in analysis["anchors"]}
    if len(colors) != 1:
        return False
    descriptor = (analysis["height"], analysis["width"], next(iter(colors)),
                  tuple((row, col, tuple(sorted(analysis["directions"][(row, col)])))
                        for row, col, _ in analysis["anchors"]))
    return descriptor in _SOURCE_DESCRIPTORS


def _build_input(height, width, side, background, guide, anchors, direction_sets):
    grid = np.full((height, width), background, dtype=np.int8)
    pattern = _pattern(side)
    for row, col, color in anchors:
        grid[row:row + side, col:col + side] = np.where(pattern, color, background)
    for (row, col, _), directions in zip(anchors, direction_sets):
        if "T" in directions:
            grid[0, col:col + side] = guide
        if "B" in directions:
            grid[-1, col:col + side] = guide
        if "L" in directions:
            grid[row:row + side, 0] = guide
        if "R" in directions:
            grid[row:row + side, -1] = guide
    return grid


def _axis_positions(rng, length, side, count):
    gap = side + 1
    slack = length - side - 2 - (count - 1) * gap
    assert slack >= 0
    if count == 1 and int(rng.integers(4)) == 0:
        return [1 if int(rng.integers(2)) == 0 else length - side - 1]
    offsets = sorted(int(x) for x in rng.integers(0, slack + 1, size=count))
    if int(rng.integers(4)) == 0:
        offsets[0] = 0
    return [1 + index * gap + offset for index, offset in enumerate(offsets)]


def _sample_pair(rng, side, direction_sets, colors, regime, aligned=False, reuse=None):
    count = len(direction_sets)
    minimum = max(8, side + 2 + (count - 1) * (side + 1))
    if minimum > 30:
        raise ValueError("too many motifs for the grid")
    for _ in range(600):
        if int(rng.integers(4)) == 0:
            height = int(rng.integers(minimum, min(30, minimum + 5) + 1))
            width = int(rng.integers(max(minimum, 22), 31))
        elif int(rng.integers(4)) == 0:
            height = int(rng.integers(max(minimum, 22), 31))
            width = int(rng.integers(minimum, min(30, minimum + 5) + 1))
        else:
            lower = max(minimum, 10 if regime == "source_train" else 11)
            height = int(rng.integers(lower, 31))
            width = int(rng.integers(lower, 31))
        rows = _axis_positions(rng, height, side, count)
        cols = _axis_positions(rng, width, side, count)
        rng.shuffle(cols)
        anchors = [(rows[i], cols[i], colors[i]) for i in range(count)]
        if reuse is not None:
            background, guide = reuse
        else:
            palette = [int(x) for x in rng.permutation(10) if int(x) not in colors]
            background, guide = palette[:2]
        grid = _build_input(height, width, side, background, guide, anchors, direction_sets)
        try:
            analysis = _analyze_input(grid)
            output = _expected_output(grid, analysis)
        except AssertionError:
            continue
        if aligned and _has_partial_copy(analysis):
            continue
        if _source_copy(analysis) or np.array_equal(grid, output):
            continue
        return {"input": grid, "output": output}
    raise ValueError("could not construct an unambiguous pair")


def _directions_for_many(rng, count, primary):
    choices = ("T", "B") if primary == "vertical" else ("L", "R")
    directions = []
    for index in range(count):
        if index == count - 1 and count >= 3 and int(rng.integers(3)) == 0:
            directions.append(set())
        else:
            selected = {d for d in choices if int(rng.integers(2))}
            directions.append(selected or {choices[int(rng.integers(2))]})
    return directions


def _random_plan(rng, count, allow_unguided, test=False):
    if count == 1:
        if test and int(rng.integers(5)) == 0:
            size = 4
        else:
            size = int(rng.integers(1, 5))
        selected = set(str(x) for x in rng.choice(_DIRECTIONS, size=size, replace=False))
        return [selected], None
    primary = "vertical" if int(rng.integers(2)) == 0 else "horizontal"
    directions = _directions_for_many(rng, count, primary)
    if not allow_unguided:
        for index, item in enumerate(directions):
            if not item:
                item.add(("T" if primary == "vertical" else "L"))
    return directions, primary


def _sample_with_plan(rng, side, directions, primary, colors, regime, aligned=False, reuse=None):
    # Retry the geometry for rare plans whose continuation strips can meet.
    if primary is None or len(directions) == 1:
        return _sample_pair(rng, side, directions, colors, regime, aligned, reuse)
    for _ in range(8):
        try:
            return _sample_pair(rng, side, directions, colors, regime, aligned, reuse)
        except ValueError:
            pass
    raise ValueError("multi-source plan did not fit")


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[key], b[key])
        for a, b in zip(first[split], second[split]) for key in ("input", "output"))
        for split in ("train", "test"))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    analyses = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            source, target = pair["input"], pair["output"]
            assert source.shape == target.shape and not np.array_equal(source, target)
            analysis = _analyze_input(source)
            assert np.array_equal(target, _expected_output(source, analysis))
            assert analysis["guide"] not in set(int(x) for x in np.unique(target))
            assert not _source_copy(analysis)
            key = _grid_key(source)
            pair_key = key, _grid_key(target)
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)
            analyses[split].append(analysis)
    train = analyses["train"]
    test = analyses["test"]
    train_sides = {a["side"] for a in train}
    assert all(a["side"] in train_sides for a in test)
    all_train_directions = set().union(*(set().union(*a["directions"].values()) for a in train))
    assert all_train_directions == set(_DIRECTIONS)
    train_unguided = any(not directions for a in train for directions in a["directions"].values())
    train_multicolor = any(len({color for _, _, color in a["anchors"]}) > 1 for a in train)
    for a in test:
        assert train_unguided or all(a["directions"].values())
        assert train_multicolor or len({color for _, _, color in a["anchors"]}) == 1
    assert any(len(a["anchors"]) >= 2 for a in train)
    assert any(_has_partial_copy(a) for a in train)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # Size and the availability of multicolor/unchanged sources are learned
    # from training. Literal colors and geometry are visible pair-level roles.
    side = (3, 4, 5)[int(rng.choice(3, p=[0.15, 0.7, 0.15]))]
    enable_multicolor = bool(rng.integers(2))
    enable_unguided = bool(rng.integers(2))
    pair_rngs = [np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                 for _ in range(train_count + test_count)]
    puzzle = {"train": [], "test": []}
    used = set()
    train_content_colors = set()

    # One variable single-source demonstration establishes composition and
    # phase. A multi-source demonstration establishes independent matching.
    missing = _DIRECTIONS[int(rng.integers(4))]
    single_directions = [set(_DIRECTIONS) - {missing}]
    if int(rng.integers(5)) == 0:
        single_directions = [set(_DIRECTIONS)]
    multi_count = 3 if enable_unguided or int(rng.integers(3)) == 0 else 2
    primary = "vertical" if missing in ("T", "B") else "horizontal"
    multi_directions = _directions_for_many(rng, multi_count, primary)
    multi_directions[0].add(missing)
    if enable_unguided:
        multi_directions[-1] = set()
    elif not multi_directions[-1]:
        multi_directions[-1].add("T" if primary == "vertical" else "L")
    plans = [(single_directions, None, False, False),
             (multi_directions, primary, enable_multicolor, False)]
    for _ in range(train_count - 2):
        aligned = int(rng.integers(3)) == 0
        if aligned:
            directions = [{_DIRECTIONS[int(rng.integers(4))]}]
            primary = None
            count = 1
        else:
            count = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
            directions, primary = _random_plan(rng, count, enable_unguided)
        plans.append((directions, primary,
                      enable_multicolor and count > 1 and bool(rng.integers(2)), aligned))
    rng.shuffle(plans)

    for index, (directions, primary, multicolor, aligned) in enumerate(plans):
        local_rng = pair_rngs[index]
        for _ in range(20):
            available = [int(x) for x in local_rng.permutation(10)]
            colors = available[:len(directions)] if multicolor else [available[0]] * len(directions)
            try:
                pair = _sample_with_plan(local_rng, side, directions, primary, colors,
                                         "source_train", aligned)
            except ValueError:
                continue
            key = _grid_key(pair["input"])
            if key not in used:
                break
        else:
            raise ValueError("could not make a unique training pair")
        used.add(key)
        puzzle["train"].append(pair)
        train_content_colors.update(color for _, _, color in _analyze_input(pair["input"])["anchors"])

    for index in range(test_count):
        local_rng = pair_rngs[train_count + index]
        # Source-like multi-bar cases coexist with single-source, horizontal,
        # and four-sided tests; the seed does not encode a fixed test layout.
        combined_case = index == 0 and enable_unguided and enable_multicolor and seed % 3 == 1
        if combined_case:
            count = 3
            primary = "vertical" if int(local_rng.integers(2)) == 0 else "horizontal"
            directions = _directions_for_many(local_rng, count, primary)
            directions[-1] = set()
            aligned = False
        elif index == 0 and seed % 4 == 0:
            count = 1
            directions = [{_DIRECTIONS[int(local_rng.integers(4))]}]
            primary = None
            aligned = True
        elif int(local_rng.integers(5)) == 0:
            count = 2
            directions = [{"T", "B", "L"}, {"T", "B"}]
            primary = "vertical"
            aligned = False
        else:
            count = int(local_rng.choice([1, 2, 3], p=[0.25, 0.5, 0.25]))
            directions, primary = _random_plan(local_rng, count, enable_unguided, True)
            aligned = False
        for _ in range(30):
            multicolor = combined_case or (enable_multicolor and count > 1 and bool(local_rng.integers(2)))
            palette = [int(x) for x in local_rng.permutation(10)]
            if int(local_rng.integers(2)) == 0:
                palette[0] = int(local_rng.choice(sorted(train_content_colors)))
            colors = palette[:count] if multicolor else [palette[0]] * count
            if multicolor and len(set(colors)) != count:
                continue
            try:
                pair = _sample_with_plan(local_rng, side, directions, primary, colors,
                                         "target_test", aligned)
            except ValueError:
                continue
            key = _grid_key(pair["input"])
            if key not in used:
                break
        else:
            raise ValueError("could not make a unique test pair")
        used.add(key)
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    for args in ((201, 2, 1), (202, 10, 1), (203, 10, 4),
                 (301, None, 1), (302, None, 4), (303, 2, None), (304, 10, None)):
        puzzle = generate_puzzle(*args)
        validate_generated_puzzle(puzzle)
        if args[1] is not None:
            assert len(puzzle["train"]) == args[1]
        if args[2] is not None:
            assert len(puzzle["test"]) == args[2]
    for args in ((0, 1, None), (0, 11, None), (0, None, 0), (0, None, 5),
                 (0, 2, 2), (0, 3, 4)):
        try:
            generate_puzzle(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert _puzzles_equal(generate_puzzle(91, 6, 3), generate_puzzle(91, 6, 3))
    assert not _puzzles_equal(generate_puzzle(101), generate_puzzle(102))
    train_counts, test_counts, shapes, layouts, sizes = set(), set(), set(), set(), set()
    split_shapes = {"train": set(), "test": set()}
    measurements = {}
    def record(split, name, condition):
        if condition:
            key = (split, name)
            measurements[key] = measurements.get(key, 0) + 1
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        training_colors = {
            color for pair in puzzle["train"]
            for _, _, color in _analyze_input(pair["input"])["anchors"]
        }
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze_input(pair["input"])
                shapes.add(pair["input"].shape)
                split_shapes[split].add(pair["input"].shape)
                sizes.add(a["side"])
                layouts.add((a["height"], a["width"], a["side"],
                             tuple((r, c, tuple(sorted(a["directions"][(r, c)])))
                                   for r, c, _ in a["anchors"])))
                directions = list(a["directions"].values())
                bars = {d: sum(d in commands for commands in directions) for d in _DIRECTIONS}
                record(split, "single", len(directions) == 1)
                record(split, "three_plus", len(directions) >= 3)
                record(split, "unguided", any(not d for d in directions))
                record(split, "multicolor", len({color for _, _, color in a["anchors"]}) > 1)
                record(split, "top_bottom", any({"T", "B"} <= d for d in directions))
                record(split, "left_right", any({"L", "R"} <= d for d in directions))
                record(split, "four_sided", any(len(d) == 4 for d in directions))
                record(split, "multiple_top_bottom", max(bars["T"], bars["B"]) >= 2)
                record(split, "multiple_left_right", max(bars["L"], bars["R"]) >= 2)
                record(split, "aligned", not _has_partial_copy(a))
                record(split, "partial", _has_partial_copy(a))
                record(split, "edge_near", any(
                    row == 1 or col == 1 or row == a["height"] - a["side"] - 1
                    or col == a["width"] - a["side"] - 1
                    for row, col, _ in a["anchors"]))
                record(split, "nonzero_background", a["background"] != 0)
                record(split, "nonone_guide", a["guide"] != 1)
                record(split, "reused_color", split == "test" and any(
                    color in training_colors for _, _, color in a["anchors"]))
                record(split, "long_aspect", abs(a["height"] - a["width"]) >= 8)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 40 and len(layouts) >= 100 and sizes == {3, 4, 5}
    assert split_shapes["train"] & split_shapes["test"]
    minimums = {
        "single": (20, 10), "three_plus": (20, 5),
        "unguided": (10, 2), "multicolor": (10, 5),
        "top_bottom": (20, 5), "left_right": (20, 5),
        "four_sided": (5, 3), "multiple_top_bottom": (10, 5),
        "multiple_left_right": (10, 5), "aligned": (20, 5),
        "partial": (100, 50), "edge_near": (50, 20),
        "nonzero_background": (100, 40), "nonone_guide": (100, 40),
        "long_aspect": (40, 20),
    }
    for name, (train_min, test_min) in minimums.items():
        assert measurements.get(("train", name), 0) >= train_min
        assert measurements.get(("test", name), 0) >= test_min
    assert measurements.get(("test", "reused_color"), 0) >= 20
