import numpy as np


_SOURCE_PAIRS = (
    (
        ("0201000000", "0000020100", "0000000000", "5555555555", "0000000000", "0200000002", "0000100000", "0000000000", "0010002010", "0000000000"),
        ("0201000100", "0200020100", "0200020000", "5555555555", "0200002002", "0200002002", "0000102000", "0000102000", "0010102010", "0010100010"),
    ),
    (
        ("0010000000", "0000001000", "0200000002", "0000000000", "0000000000", "5555555555", "0000000000", "0000000000", "0100020010", "0000000000"),
        ("0010001000", "0000001000", "0200000002", "0200000002", "0200000002", "5555555555", "0000020000", "0000020000", "0100020010", "0100000010"),
    ),
    (
        ("0000000000", "0201000020", "0000001000", "0000000000", "5555555555", "0000000000", "0000000000", "0010020000", "2000000000", "0000000010"),
        ("0001001000", "0201001020", "0200001020", "0200000020", "5555555555", "2000020000", "2000020000", "2010020000", "2010000000", "0010000010"),
    ),
)

_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))


def _grid_key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _is_source_pair(input_grid, output_grid):
    for source_input, source_output in _SOURCE_PAIRS:
        if input_grid.shape != (len(source_input), len(source_input[0])):
            continue
        actual_input = tuple("".join(str(int(v)) for v in row) for row in input_grid)
        actual_output = tuple("".join(str(int(v)) for v in row) for row in output_grid)
        if (actual_input, actual_output) == (source_input, source_output):
            return True
    return False


def _analyze_input(grid):
    values, counts = np.unique(grid, return_counts=True)
    assert 3 <= len(values) <= 4
    assert np.count_nonzero(counts == max(counts)) == 1
    background = int(values[np.argmax(counts)])
    candidates = []
    for row in range(grid.shape[0]):
        color = int(grid[row, 0])
        if color != background and np.all(grid[row, :] == color):
            candidates.append(("horizontal", row, color))
    for column in range(grid.shape[1]):
        color = int(grid[0, column])
        if color != background and np.all(grid[:, column] == color):
            candidates.append(("vertical", column, color))
    assert len(candidates) == 1
    orientation, separator_index, separator = candidates[0]
    perpendicular = grid.shape[0] if orientation == "horizontal" else grid.shape[1]
    parallel = grid.shape[1] if orientation == "horizontal" else grid.shape[0]
    assert perpendicular >= 5 and parallel >= 5
    assert 1 <= separator_index <= perpendicular - 2
    assert np.count_nonzero(grid == separator) == parallel
    seed_colors = frozenset(int(v) for v in values if int(v) not in (background, separator))
    assert 1 <= len(seed_colors) <= 2
    return {"background": background, "separator": separator,
            "seed_colors": seed_colors, "orientation": orientation,
            "separator_index": int(separator_index), "perpendicular": perpendicular,
            "parallel": parallel}


def _ray_positions(position, separator_index, perpendicular, grows_inward):
    if position < separator_index:
        endpoint = separator_index - 1 if grows_inward else 0
    else:
        endpoint = separator_index + 1 if grows_inward else perpendicular - 1
    return range(min(position, endpoint), max(position, endpoint) + 1)


def _ray_cells(grid, analysis, inward, outward):
    result = {inward: set(), outward: set()}
    horizontal = analysis["orientation"] == "horizontal"
    for color, grows_inward in ((inward, True), (outward, False)):
        for row, column in np.argwhere(grid == color):
            position = int(row if horizontal else column)
            lane = int(column if horizontal else row)
            for location in _ray_positions(position, analysis["separator_index"],
                                           analysis["perpendicular"], grows_inward):
                cell = (location, lane) if horizontal else (lane, location)
                result[color].add(cell)
    assert result[inward].isdisjoint(result[outward])
    return result


def _solve_with_roles(grid, analysis, inward, outward):
    rays = _ray_cells(grid, analysis, inward, outward)
    result = grid.copy()
    for color, cells in rays.items():
        for row, column in cells:
            assert int(grid[row, column]) in (analysis["background"], color)
            result[row, column] = color
    return result


def _normalize_optional_count(value, name, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    train_count = _normalize_optional_count(num_train, "num_train", 2, 10)
    test_count = _normalize_optional_count(num_test, "num_test", 1, 4)
    if train_count is None and test_count is None:
        train_count, test_count = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif train_count is None:
        train_count = int(rng.integers(max(2, test_count + 1), 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")
    return train_count, test_count


def _sample_length(rng, minimum):
    band = int(rng.choice(3, p=(0.25, 0.45, 0.30)))
    ranges = ((5, 8), (9, 18), (19, 30))
    low, high = ranges[band]
    return int(rng.integers(max(low, minimum), high + 1)) if high >= minimum else minimum


def _choose_mode(rng, split, index):
    if split == "train" and index < 2:
        return "demo"
    modes = ("general", "one_role", "one_side", "merge", "shared", "uneven")
    probabilities = ((0.31, 0.10, 0.10, 0.17, 0.17, 0.15)
                     if split == "train" else
                     (0.12, 0.28, 0.19, 0.14, 0.17, 0.10))
    return str(rng.choice(modes, p=probabilities))


def _try_add(placements, role, position, lane, separator_index, perpendicular):
    if any(p == position and l == lane for _, p, l in placements):
        return False
    proposed = set(_ray_positions(position, separator_index, perpendicular,
                                  role == "inward"))
    for old_role, old_position, old_lane in placements:
        if old_lane == lane and old_role != role:
            old_ray = set(_ray_positions(old_position, separator_index,
                                         perpendicular, old_role == "inward"))
            if proposed.intersection(old_ray):
                return False
    placements.append((role, position, lane))
    return True


def _add_random(rng, placements, role, side, separator_index, perpendicular, parallel):
    positions = (list(range(separator_index)) if side == "before" else
                 list(range(separator_index + 1, perpendicular)))
    destination = ((separator_index - 1 if side == "before" else separator_index + 1)
                   if role == "inward" else
                   (0 if side == "before" else perpendicular - 1))
    for _ in range(60):
        lane = int(rng.integers(0, parallel))
        choice = int(rng.integers(0, 5))
        if choice == 0:
            position = destination
        elif choice == 1:
            position = positions[0] if destination != positions[0] else positions[-1]
        else:
            position = int(rng.choice(positions))
        if _try_add(placements, role, position, lane, separator_index, perpendicular):
            return True
    return False


def _make_placements(rng, mode, separator_index, perpendicular, parallel):
    sides = {"before": list(range(separator_index)),
             "after": list(range(separator_index + 1, perpendicular))}
    placements = []
    cap = min(12, max(3, ((perpendicular - 1) * parallel) // 9))

    if mode == "demo":
        for side in ("before", "after"):
            lanes = [int(v) for v in rng.choice(parallel, size=2, replace=False)]
            for role, lane in zip(("inward", "outward"), lanes):
                positions = sides[side]
                destination = (positions[-1] if (role == "inward") == (side == "before")
                               else positions[0])
                possible = [p for p in positions if p != destination]
                position = int(rng.choice(possible))
                assert _try_add(placements, role, position, lane,
                                separator_index, perpendicular)
        return placements

    if mode == "shared":
        possible_sides = [side for side in sides if len(sides[side]) >= 4]
        if not possible_sides:
            return None
        side = str(rng.choice(possible_sides))
        positions = sides[side]
        lane = int(rng.integers(0, parallel))
        if side == "before":
            outward = int(rng.choice(positions[1:-2]))
            inward = int(rng.choice([p for p in positions if outward < p < positions[-1]]))
        else:
            inward = int(rng.choice(positions[1:-2]))
            outward = int(rng.choice([p for p in positions if inward < p < positions[-1]]))
        assert _try_add(placements, "inward", inward, lane, separator_index, perpendicular)
        assert _try_add(placements, "outward", outward, lane, separator_index, perpendicular)
        extra = int(rng.integers(0, min(3, cap - 1)))
    elif mode == "merge":
        possible_sides = [side for side in sides if len(sides[side]) >= 3]
        if not possible_sides:
            return None
        side = str(rng.choice(possible_sides))
        role = str(rng.choice(("inward", "outward")))
        lane = int(rng.integers(0, parallel))
        positions = [int(v) for v in rng.choice(sides[side], size=2, replace=False)]
        for position in positions:
            assert _try_add(placements, role, position, lane, separator_index, perpendicular)
        extra = int(rng.integers(0, min(3, cap - 1)))
    elif mode == "one_role":
        role = str(rng.choice(("inward", "outward")))
        count = int(rng.integers(1, min(5, cap) + 1))
        for _ in range(count):
            if not _add_random(rng, placements, role, str(rng.choice(tuple(sides))),
                               separator_index, perpendicular, parallel):
                return None
        extra = 0
    elif mode == "one_side":
        side = str(rng.choice(tuple(sides)))
        count = int(rng.integers(2, min(6, cap) + 1))
        roles = ["inward", "outward"] + [str(rng.choice(("inward", "outward")))
                                         for _ in range(count - 2)]
        for role in roles:
            if not _add_random(rng, placements, role, side,
                               separator_index, perpendicular, parallel):
                return None
        extra = 0
    elif mode == "uneven":
        dominant = str(rng.choice(("inward", "outward")))
        other = "outward" if dominant == "inward" else "inward"
        count = int(rng.integers(3, min(7, cap) + 1))
        for role in [other] + [dominant] * (count - 1):
            if not _add_random(rng, placements, role, str(rng.choice(tuple(sides))),
                               separator_index, perpendicular, parallel):
                return None
        extra = 0
    else:
        count = int(rng.integers(2, min(8, cap) + 1))
        for _ in range(count):
            role = str(rng.choice(("inward", "outward")))
            side = str(rng.choice(tuple(sides)))
            if not _add_random(rng, placements, role, side,
                               separator_index, perpendicular, parallel):
                return None
        extra = 0

    for _ in range(extra):
        role = str(rng.choice(("inward", "outward")))
        side = str(rng.choice(tuple(sides)))
        _add_random(rng, placements, role, side,
                    separator_index, perpendicular, parallel)
    return placements


def _sample_pair(rng, split, index, orientation, inward, outward,
                 first_analysis, train_separator_indices):
    mode = _choose_mode(rng, split, index)
    minimum_perpendicular = 8 if mode == "demo" else 9 if mode == "shared" else 7 if mode == "merge" else 5
    for _ in range(120):
        perpendicular = _sample_length(rng, minimum_perpendicular)
        parallel = _sample_length(rng, 6 if mode == "demo" else 5)
        if mode == "demo":
            choices = list(range(2, perpendicular - 2))
            if index == 1:
                filtered = [p for p in choices if p not in train_separator_indices]
                if filtered:
                    choices = filtered
        else:
            choices = list(range(1, perpendicular - 1))
            if split == "test":
                if int(rng.integers(0, 4)) == 0:
                    choices = [1, perpendicular - 2]
                elif int(rng.integers(0, 2)) == 0:
                    familiar = [p for p in choices if p in train_separator_indices]
                    if familiar:
                        choices = familiar
                else:
                    novel = [p for p in choices if p not in train_separator_indices]
                    if novel:
                        choices = novel
        separator_index = int(rng.choice(choices))
        colors = [c for c in range(10) if c not in (inward, outward)]
        if first_analysis is not None and index == 1 and split == "train":
            backgrounds = [c for c in colors if c != first_analysis["background"]]
        else:
            backgrounds = colors
        background = int(rng.choice(backgrounds))
        separators = [c for c in colors if c != background]
        if first_analysis is not None and index == 1 and split == "train":
            separators = [c for c in separators if c != first_analysis["separator"]]
        separator = int(rng.choice(separators))
        shape = ((perpendicular, parallel) if orientation == "horizontal" else
                 (parallel, perpendicular))
        if first_analysis is not None and index == 1 and split == "train" and shape == first_analysis["shape"]:
            continue
        placements = _make_placements(rng, mode, separator_index, perpendicular, parallel)
        if placements is None:
            continue
        grid = np.full(shape, background, dtype=np.int8)
        if orientation == "horizontal":
            grid[separator_index, :] = separator
        else:
            grid[:, separator_index] = separator
        for role, position, lane in placements:
            color = inward if role == "inward" else outward
            if orientation == "horizontal":
                grid[position, lane] = color
            else:
                grid[lane, position] = color
        try:
            analysis = _analyze_input(grid)
            output = _solve_with_roles(grid, analysis, inward, outward)
        except AssertionError:
            continue
        if np.array_equal(grid, output) or _is_source_pair(grid, output):
            continue
        if mode == "demo":
            if not all(np.any((grid == analysis["background"]) & (output == color))
                       for color in (inward, outward)):
                continue
        analysis["shape"] = shape
        return {"input": grid, "output": output}, analysis
    raise ValueError("could not construct a valid pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    inward, outward = (int(v) for v in rng.choice(10, size=2, replace=False))
    base_orientation = str(rng.choice(("horizontal", "vertical")))
    orientation_regime = int(rng.choice(3, p=(0.45, 0.30, 0.25)))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    first_analysis = None
    train_separator_indices = set()
    for serial, pair_seed in enumerate(pair_seeds):
        split = "train" if serial < train_count else "test"
        index = serial if split == "train" else serial - train_count
        opposite = "vertical" if base_orientation == "horizontal" else "horizontal"
        if orientation_regime == 0:
            orientation = base_orientation
        elif split == "train":
            if orientation_regime == 1 and index == 1:
                orientation = opposite
            elif orientation_regime == 1 and index > 1:
                orientation = str(rng.choice(("horizontal", "vertical")))
            else:
                orientation = base_orientation
        elif index == 0:
            orientation = opposite
        else:
            orientation = str(rng.choice(("horizontal", "vertical")))
        pair_rng = np.random.default_rng(int(pair_seed))
        for _ in range(100):
            pair, analysis = _sample_pair(pair_rng, split, index, orientation,
                                          inward, outward, first_analysis,
                                          train_separator_indices)
            key = _grid_key(pair["input"])
            if key not in seen:
                break
        else:
            raise ValueError("could not construct distinct pair inputs")
        seen.add(key)
        puzzle[split].append(pair)
        if serial == 0:
            first_analysis = analysis
        if split == "train":
            train_separator_indices.add(analysis["separator_index"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _assignments_that_fit(training_pairs, analyses, colors):
    first, second = sorted(colors)
    assignments = []
    for inward, outward in ((first, second), (second, first)):
        try:
            fits = all(np.array_equal(pair["output"],
                                      _solve_with_roles(pair["input"], analysis, inward, outward))
                       for pair, analysis in zip(training_pairs, analyses))
        except AssertionError:
            fits = False
        if fits:
            assignments.append((inward, outward))
    return assignments


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs, seen_pairs = set(), set()
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert not _is_source_pair(inp, out)
            records.append((split, pair, _analyze_input(inp)))

    first = records[0][2]
    second = records[1][2]
    assert first["seed_colors"] == second["seed_colors"] and len(first["seed_colors"]) == 2
    assert records[0][1]["input"].shape != records[1][1]["input"].shape
    assert first["background"] != second["background"]
    assert first["separator"] != second["separator"]
    colors = first["seed_colors"]
    for split, pair, analysis in records:
        assert analysis["seed_colors"].issubset(colors)
        assert analysis["background"] not in colors
        assert analysis["separator"] not in colors
        assert np.array_equal(pair["output"][pair["input"] != analysis["background"]],
                              pair["input"][pair["input"] != analysis["background"]])
        assert not np.array_equal(pair["input"], pair["output"])
        if split == "train" and any(pair is demo for demo in puzzle["train"][:2]):
            perpendicular = analysis["perpendicular"]
            assert 2 <= analysis["separator_index"] <= perpendicular - 3
            for color in colors:
                coords = np.argwhere(pair["input"] == color)
                positions = coords[:, 0] if analysis["orientation"] == "horizontal" else coords[:, 1]
                assert np.any(positions < analysis["separator_index"])
                assert np.any(positions > analysis["separator_index"])

    training_analyses = [analysis for split, _, analysis in records if split == "train"]
    assignments = _assignments_that_fit(puzzle["train"], training_analyses, colors)
    assert len(assignments) == 1
    inward, outward = assignments[0]
    for _, pair, analysis in records:
        expected = _solve_with_roles(pair["input"], analysis, inward, outward)
        assert np.array_equal(pair["output"], expected)
    for pair, analysis in zip(puzzle["train"][:2], training_analyses[:2]):
        for color in colors:
            assert np.any((pair["input"] == analysis["background"]) &
                          (pair["output"] == color))


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[name], b[name])
                   for a, b in zip(first[split], second[split])
                   for name in ("input", "output"))
               for split in ("train", "test"))


def _expect_value_error(**kwargs):
    try:
        generate_puzzle(99991, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({tuple(_grid_key(pair["input"]) for split in ("train", "test")
                      for pair in puzzle[split]) for puzzle in puzzles}) >= 40
    orientations, shapes, palette_pairs, separator_positions = set(), set(), set(), set()
    one_role_tests = one_side_tests = shared_pairs = merged_pairs = mixed_puzzles = 0
    new_test_orientation_puzzles = unequal_pairs = 0
    zero_extensions = short_extensions = long_extensions = 0
    familiar_tests = novel_tests = thin_regions = 0
    minimum_dimension, maximum_dimension = 30, 1
    role_assignments = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        analyses = [_analyze_input(pair["input"])
                    for split in ("train", "test") for pair in puzzle[split]]
        pairs = puzzle["train"] + puzzle["test"]
        inferred = _assignments_that_fit(puzzle["train"],
                                         analyses[:len(puzzle["train"])],
                                         analyses[0]["seed_colors"])[0]
        role_assignments.add(inferred)
        train_positions = {a["separator_index"] for a in analyses[:len(puzzle["train"])]}
        if len({a["orientation"] for a in analyses}) > 1:
            mixed_puzzles += 1
        training_orientations = {a["orientation"] for a in analyses[:len(puzzle["train"])]}
        if any(a["orientation"] not in training_orientations
               for a in analyses[len(puzzle["train"]):]):
            new_test_orientation_puzzles += 1
        for i, (pair, analysis) in enumerate(zip(pairs, analyses)):
            inp, out = pair["input"], pair["output"]
            orientations.add(analysis["orientation"])
            shapes.add(inp.shape)
            palette_pairs.add((analysis["background"], analysis["separator"]))
            separator_positions.add((analysis["perpendicular"], analysis["separator_index"]))
            minimum_dimension = min(minimum_dimension, *inp.shape)
            maximum_dimension = max(maximum_dimension, *inp.shape)
            if min(analysis["separator_index"],
                   analysis["perpendicular"] - 1 - analysis["separator_index"]) == 1:
                thin_regions += 1
            near = min(analysis["separator_index"],
                       analysis["perpendicular"] - 1 - analysis["separator_index"])
            far = max(analysis["separator_index"],
                      analysis["perpendicular"] - 1 - analysis["separator_index"])
            unequal_pairs += far >= 3 * near
            for color, grows_inward in ((inferred[0], True), (inferred[1], False)):
                for row, column in np.argwhere(inp == color):
                    position = int(row if analysis["orientation"] == "horizontal" else column)
                    extension = len(_ray_positions(position, analysis["separator_index"],
                                                   analysis["perpendicular"], grows_inward)) - 1
                    zero_extensions += extension == 0
                    short_extensions += extension == 1
                    long_extensions += extension >= 10
            if i >= len(puzzle["train"]):
                one_role_tests += len(analysis["seed_colors"]) == 1
                positions = np.argwhere(np.isin(inp, tuple(analysis["seed_colors"])))
                axis = 0 if analysis["orientation"] == "horizontal" else 1
                one_side_tests += (np.all(positions[:, axis] < analysis["separator_index"]) or
                                   np.all(positions[:, axis] > analysis["separator_index"]))
                if analysis["separator_index"] in train_positions:
                    familiar_tests += 1
                else:
                    novel_tests += 1
            first, second = sorted(analyses[0]["seed_colors"])
            horizontal = analysis["orientation"] == "horizontal"
            for lane in range(analysis["parallel"]):
                coordinates = [(int(pos), int(inp[pos, lane] if horizontal else inp[lane, pos]))
                               for pos in range(analysis["perpendicular"])
                               if int(inp[pos, lane] if horizontal else inp[lane, pos]) in (first, second)]
                for side in ("before", "after"):
                    located = [(p, c) for p, c in coordinates
                               if (p < analysis["separator_index"]) == (side == "before")]
                    if len({c for _, c in located}) == 2:
                        shared_pairs += 1
                    if any(sum(c == color for _, c in located) >= 2 for color in (first, second)):
                        merged_pairs += 1
            assert np.array_equal(out, _solve_with_roles(inp, analysis, *
                                   _assignments_that_fit(puzzle["train"], analyses[:len(puzzle["train"])],
                                                         analyses[0]["seed_colors"])[0]))
    assert orientations == {"horizontal", "vertical"}
    assert minimum_dimension <= 6 and maximum_dimension >= 27
    assert len(shapes) >= 50 and len(palette_pairs) >= 25
    assert len(separator_positions) >= 35
    assert len(role_assignments) >= 25
    assert mixed_puzzles >= 10
    assert new_test_orientation_puzzles >= 5
    assert unequal_pairs >= 50
    assert one_role_tests >= 8 and one_side_tests >= 5
    assert shared_pairs >= 8 and merged_pairs >= 8
    assert familiar_tests >= 5 and novel_tests >= 5 and thin_regions >= 5
    assert zero_extensions >= 30 and short_extensions >= 30 and long_extensions >= 20
    assert _puzzles_equal(generate_puzzle(7001, 7, 3), generate_puzzle(7001, 7, 3))
    assert _puzzles_equal(generate_puzzle(7002), generate_puzzle(7002))
    assert not _puzzles_equal(generate_puzzle(7003), generate_puzzle(7004))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(8000 + 10 * train_count + test_count,
                                 train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(8100 + (train_count or 0) + (test_count or 0),
                                 train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for bad in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                {"num_test": 5}, {"num_train": 2, "num_test": 2},
                {"num_train": 3, "num_test": 4}, {"num_train": 2.0, "num_test": 1}):
        _expect_value_error(**bad)
