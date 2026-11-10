import numpy as np


_DIRECTIONS = {
    "NW": ((0, 0), (-1, -1)),
    "NE": ((0, 1), (-1, 1)),
    "SW": ((1, 0), (1, -1)),
    "SE": ((1, 1), (1, 1)),
}


def _rows_to_array(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _rows_to_array((
            "0000000000",
            "0009900000",
            "0000900000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0009000000",
            "0009900000",
            "0000000000",
            "0000000000",
        )),
        _rows_to_array((
            "0000000000",
            "0009900009",
            "0000900090",
            "0090000900",
            "0900009000",
            "9000090000",
            "0009000000",
            "0009900000",
            "0000000000",
            "0000000000",
        )),
    ),
    (
        _rows_to_array((
            "0000000000",
            "0000000000",
            "0700000000",
            "0770000000",
            "0000007700",
            "0000000700",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        )),
        _rows_to_array((
            "0000700000",
            "0007000000",
            "0700000000",
            "0770000000",
            "0000007700",
            "0000000700",
            "0000070000",
            "0000700000",
            "0007000000",
            "0070000000",
        )),
    ),
    (
        _rows_to_array((
            "0000000000",
            "0000000000",
            "0000800000",
            "0008800000",
            "0000000880",
            "0000000800",
            "0088000000",
            "0008000000",
            "0000000000",
            "0000000000",
        )),
        _rows_to_array((
            "0800000000",
            "0080000000",
            "0000800000",
            "0008800000",
            "0000000880",
            "0000000800",
            "0088000008",
            "0008000000",
            "0800000000",
            "8000000000",
        )),
    ),
)


def _validate_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> None:
    if value is None:
        return

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("no legal train/test count combination exists")

    return num_train, num_test


def _choice(rng: np.random.Generator, values) -> int:
    return int(values[int(rng.integers(len(values)))])


def _geometry(height, width, top, left, orientation):
    missing_relative, direction = _DIRECTIONS[orientation]
    missing = (top + missing_relative[0], left + missing_relative[1])
    block = {(top, left), (top, left + 1), (top + 1, left), (top + 1, left + 1)}
    ray = []
    row, column = missing[0] + direction[0], missing[1] + direction[1]
    while 0 <= row < height and 0 <= column < width:
        ray.append((row, column))
        row += direction[0]
        column += direction[1]
    return {"orientation": orientation, "top": top, "left": left,
            "missing": missing, "block": block, "filled": block - {missing},
            "ray": tuple(ray)}


def _array_key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.uint8).tobytes()


def _is_source_pair(input_grid, output_grid):
    return any(np.array_equal(input_grid, original_input) and
               np.array_equal(output_grid, original_output)
               for original_input, original_output in _SOURCE_PAIRS)


def _analyze_input(input_grid):
    """Recover the Ls from pixels, without using generation-time placement data."""
    colors, counts = np.unique(input_grid, return_counts=True)
    assert len(colors) >= 2
    background_index = int(np.argmax(counts))
    background = int(colors[background_index])
    assert counts[background_index] > sum(counts) - counts[background_index]
    background_mask = input_grid == background
    height, width = input_grid.shape
    coverage = np.zeros((height, width), dtype=np.int16)
    markers = []
    for color in colors:
        color = int(color)
        if color == background:
            continue
        mask = input_grid == color
        local_count = (mask[:-1, :-1].astype(np.int16) +
                       mask[:-1, 1:] + mask[1:, :-1] + mask[1:, 1:])
        local_background = (background_mask[:-1, :-1].astype(np.int16) +
                            background_mask[:-1, 1:] +
                            background_mask[1:, :-1] +
                            background_mask[1:, 1:])
        tops, lefts = np.nonzero((local_count == 3) & (local_background == 1))
        for top, left in zip(tops, lefts):
            top, left = int(top), int(left)
            local = mask[top:top + 2, left:left + 2]
            gap = tuple(int(v) for v in np.argwhere(~local)[0])
            orientation = next(name for name, (corner, _) in _DIRECTIONS.items()
                               if corner == gap)
            marker = _geometry(height, width, top, left, orientation)
            marker["color"] = color
            markers.append(marker)
            for row, column in marker["filled"]:
                coverage[row, column] += 1
    assert markers
    assert np.all(coverage[~background_mask] == 1)
    assert np.all(coverage[background_mask] == 0)
    for index, marker in enumerate(markers):
        assert marker["ray"]
        assert input_grid[marker["missing"]] == background
        last_row, last_column = marker["ray"][-1]
        assert last_row in (0, height - 1) or last_column in (0, width - 1)
        for other in markers[index + 1:]:
            assert marker["block"].isdisjoint(other["block"])
            assert set(marker["ray"]).isdisjoint(other["block"])
            assert set(other["ray"]).isdisjoint(marker["block"])
            if marker["color"] != other["color"]:
                assert set(marker["ray"]).isdisjoint(other["ray"])
    expected = input_grid.copy()
    for marker in markers:
        for row, column in marker["ray"]:
            expected[row, column] = marker["color"]
    return {"background": background, "markers": markers,
            "expected_output": expected}


def _minimum_chebyshev(a, b):
    return min(max(abs(ra - rb), abs(ca - cb))
               for ra, ca in a for rb, cb in b)


def _compatible(candidate, placed):
    for other in placed:
        if not candidate["block"].isdisjoint(other["block"]):
            return False
        if not set(candidate["ray"]).isdisjoint(other["block"]):
            return False
        if not set(other["ray"]).isdisjoint(candidate["block"]):
            return False
        if (candidate["color"] != other["color"] and
                not set(candidate["ray"]).isdisjoint(other["ray"])):
            return False
    return True


def _sample_shape(rng, train_shapes, split):
    if split == "test" and train_shapes and rng.random() < 0.17:
        return train_shapes[int(rng.integers(len(train_shapes)))]
    draw = rng.random()
    if draw < 0.14:
        return 10, 10
    if draw < 0.32:
        return int(rng.integers(22, 31)), int(rng.integers(22, 31))
    return int(rng.integers(8, 23)), int(rng.integers(8, 23))


def _sample_count(rng, height, width):
    capacity = min(8, max(2, height * width // 25))
    return int(rng.integers(1, capacity + 1))


def _place_markers(rng, height, width, orientations, colors, background, mode):
    placed = []
    grid = np.full((height, width), background, dtype=np.int8)
    for index, (orientation, color) in enumerate(zip(orientations, colors)):
        candidates = []
        for top in range(height - 1):
            for left in range(width - 1):
                candidate = _geometry(height, width, top, left, orientation)
                if not candidate["ray"]:
                    continue
                candidate["color"] = color
                candidates.append(candidate)
        if not candidates:
            return None
        # Short rays and long rays coexist; the profile is chosen per attempt.
        profile = int(rng.integers(5))
        if index == 0 and profile == 0:
            preferred = [c for c in candidates if len(c["ray"]) == 1]
        elif index == 1 and placed and len(placed[0]["ray"]) == 1:
            preferred = [c for c in candidates if len(c["ray"]) >= 4]
        else:
            preferred = candidates
        if not preferred:
            preferred = candidates
        choices = [preferred[int(i)] for i in rng.permutation(len(preferred))]
        for candidate in choices:
            if not _compatible(candidate, placed):
                continue
            if index == 1 and mode == "close" and _minimum_chebyshev(
                    candidate["filled"], placed[0]["filled"]) > 2:
                continue
            if index == 1 and mode == "cross" and not (
                    candidate["color"] == placed[0]["color"] and
                    set(candidate["ray"]) & set(placed[0]["ray"])):
                continue
            for row, column in candidate["filled"]:
                grid[row, column] = color
            try:
                analysis = _analyze_input(grid)
                valid = len(analysis["markers"]) == index + 1
                valid = valid and { (m["top"], m["left"], m["color"])
                                    for m in analysis["markers"] } == {
                                    (m["top"], m["left"], m["color"])
                                    for m in placed + [candidate] }
            except (AssertionError, StopIteration):
                valid = False
            if valid:
                placed.append(candidate)
                break
            for row, column in candidate["filled"]:
                grid[row, column] = background
        else:
            return None
    output = _analyze_input(grid)["expected_output"]
    return grid, output


def _sample_colors(rng, count, background, anchor, mixed, novel_pool):
    available = [c for c in range(10) if c != background and
                 (novel_pool is None or c not in novel_pool)]
    if not available:
        available = [c for c in range(10) if c != background]
    if mixed and len(available) < 2:
        available += [_choice(rng, [c for c in range(10)
                                    if c != background and c not in available])]
    first = anchor if anchor in available else _choice(rng, available)
    colors = [first]
    for _ in range(1, count):
        if mixed and rng.random() < 0.65:
            choices = [c for c in available if c != colors[0]]
            colors.append(_choice(rng, choices or available))
        else:
            colors.append(first)
    if mixed and len(set(colors)) == 1:
        choices = [c for c in available if c != first]
        colors[-1] = _choice(rng, choices)
    return colors


def _make_pair(rng, shape, orientations, colors, background):
    height, width = shape
    possible_cross = (len(colors) >= 2 and colors[0] == colors[1] and
                      (orientations[0] in ("NW", "SE")) !=
                      (orientations[1] in ("NW", "SE")))
    draw = rng.random()
    mode = "cross" if possible_cross and draw < 0.22 else (
        "close" if len(colors) >= 2 and draw < 0.45 else "free")
    for attempt in range(22):
        active_mode = mode if attempt < 12 else "free"
        result = _place_markers(rng, height, width, orientations,
                                colors, background, active_mode)
        if result is not None:
            return result
    raise ValueError("could not place a unique L layout")


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    input_keys, pair_keys = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert input_grid.shape == output_grid.shape
            key = _array_key(input_grid)
            pair_key = (key, _array_key(output_grid))
            assert key not in input_keys and pair_key not in pair_keys
            input_keys.add(key)
            pair_keys.add(pair_key)
            analysis = _analyze_input(input_grid)
            assert np.array_equal(output_grid, analysis["expected_output"])
            assert not np.array_equal(input_grid, output_grid)
            assert not _is_source_pair(input_grid, output_grid)
            analyses[split].append(analysis)
    train_orientations = {marker["orientation"]
                          for a in analyses["train"] for marker in a["markers"]}
    assert len(train_orientations) >= 2
    train_mixed = any(len({m["color"] for m in a["markers"]}) > 1
                      for a in analyses["train"])
    test_mixed = any(len({m["color"] for m in a["markers"]}) > 1
                     for a in analyses["test"])
    assert not test_mixed or train_mixed
    # Both splits use the same blank-gap diagonal rule, inferred from pixels.


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    # Only the geometric rule and semantic color roles are puzzle conventions.
    base_background = int(rng.integers(10))
    anchor = _choice(rng, [c for c in range(10) if c != base_background])
    palette_mode = _choice(rng, (0, 1, 2))  # shared, flexible, novel test
    mixed_puzzle = rng.random() < 0.47
    source_like = rng.random() < 0.26
    if source_like:
        train_pool = ["NE", "SW"]
    else:
        pool_size = int(rng.integers(2, 5))
        train_pool = [str(x) for x in rng.permutation(list(_DIRECTIONS))[:pool_size]]
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_shapes = []
    train_colors = set()
    train_specs = []
    for index in range(num_train):
        shape = _sample_shape(rng, train_shapes, "train")
        count = _sample_count(rng, *shape)
        if index == 0:
            count = max(count, 2)
        train_specs.append([shape, count])
        train_shapes.append(shape)
    if sum(spec[1] for spec in train_specs) < len(train_pool):
        train_specs[-1][1] += len(train_pool) - sum(spec[1] for spec in train_specs)
    forced = list(train_pool)
    first_two = forced[:2]
    forced = forced[2:]
    for index, (shape, count) in enumerate(train_specs):
        orientations = []
        if index == 0:
            orientations.extend(first_two)
        while forced and len(orientations) < count:
            orientations.append(forced.pop(0))
        while len(orientations) < count:
            orientations.append(str(train_pool[int(rng.integers(len(train_pool)))]))
        rng.shuffle(orientations)
        background = base_background if rng.random() < 0.53 else _choice(
            rng, [c for c in range(10) if c != anchor])
        mixed = mixed_puzzle and index == 0 or (
            mixed_puzzle and count >= 2 and rng.random() < 0.35)
        pair_anchor = anchor if palette_mode == 0 or rng.random() < 0.55 else _choice(
            rng, [c for c in range(10) if c != background])
        colors = _sample_colors(rng, count, background, pair_anchor, mixed, None)
        for retry in range(30):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            try:
                input_grid, output_grid = _make_pair(
                    pair_rng, shape, orientations, colors, background)
            except ValueError:
                continue
            if _array_key(input_grid) not in used_inputs and not _is_source_pair(
                    input_grid, output_grid):
                break
        else:
            raise ValueError("could not generate distinct training pair")
        used_inputs.add(_array_key(input_grid))
        train_colors.update(colors)
        puzzle["train"].append({"input": input_grid, "output": output_grid})
    seen_train_orientations = {m["orientation"] for pair in puzzle["train"]
                               for m in _analyze_input(pair["input"])["markers"]}
    for index in range(num_test):
        shape = _sample_shape(rng, train_shapes, "test")
        count = _sample_count(rng, *shape)
        unseen = [d for d in _DIRECTIONS if d not in seen_train_orientations]
        if unseen and rng.random() < 0.50:
            orientations = [str(unseen[int(rng.integers(len(unseen)))])]
        else:
            orientations = [str(train_pool[int(rng.integers(len(train_pool)))])]
        while len(orientations) < count:
            orientations.append(str(list(_DIRECTIONS)[int(rng.integers(4))]))
        background = base_background if rng.random() < 0.53 else _choice(
            rng, [c for c in range(10) if c != anchor])
        novel = train_colors if palette_mode == 2 else None
        mixed = mixed_puzzle and count >= 2 and rng.random() < 0.43
        pair_anchor = anchor if palette_mode == 0 or rng.random() < 0.55 else _choice(
            rng, [c for c in range(10) if c != background])
        colors = _sample_colors(rng, count, background, pair_anchor, mixed, novel)
        for retry in range(30):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            try:
                input_grid, output_grid = _make_pair(
                    pair_rng, shape, orientations, colors, background)
            except ValueError:
                continue
            if _array_key(input_grid) not in used_inputs and not _is_source_pair(
                    input_grid, output_grid):
                break
        else:
            raise ValueError("could not generate distinct test pair")
        used_inputs.add(_array_key(input_grid))
        puzzle["test"].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[role], b[role])
        for a, b in zip(first[split], second[split])
        for role in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    for original_input, original_output in _SOURCE_PAIRS:
        assert np.array_equal(_analyze_input(original_input)["expected_output"],
                              original_output)
    assert 1 in [len(m["ray"]) for m in _analyze_input(
        _SOURCE_PAIRS[2][0])["markers"]]
    automatic = generate_puzzle(123456789)
    assert _puzzles_equal(automatic, generate_puzzle(123456789))
    explicit = generate_puzzle(987654321, 6, 3)
    assert _puzzles_equal(explicit, generate_puzzle(987654321, 6, 3))
    assert not _puzzles_equal(explicit, generate_puzzle(987654322, 6, 3))
    for index, (a, b) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(2000 + index, a, b)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (a, b)
        validate_generated_puzzle(puzzle)
    for index, (a, b) in enumerate(((None, 1), (None, 4),
                                     (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + index, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
        validate_generated_puzzle(puzzle)
    for a, b in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (True, 1), (4.0, 1)):
        try:
            generate_puzzle(4000, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    metrics = {name: set() for name in ("train_count", "test_count", "shape",
               "layout", "background", "orientation", "marker_count",
               "ray_length", "palette")}
    flags = {name: 0 for name in ("train_all_four", "test_familiar",
             "test_heldout", "single_train", "multi_train", "single_test",
             "large_count", "short_ray", "close", "cross", "multicolor",
             "shared_color", "variable_background", "small_test",
             "equal_size", "ten_square", "large_canvas")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        metrics["train_count"].add(len(puzzle["train"]))
        metrics["test_count"].add(len(puzzle["test"]))
        train_analyses = [_analyze_input(p["input"]) for p in puzzle["train"]]
        test_analyses = [_analyze_input(p["input"]) for p in puzzle["test"]]
        train_dirs = {m["orientation"] for a in train_analyses for m in a["markers"]}
        test_dirs = {m["orientation"] for a in test_analyses for m in a["markers"]}
        train_colors = {m["color"] for a in train_analyses for m in a["markers"]}
        test_colors = {m["color"] for a in test_analyses for m in a["markers"]}
        train_shapes = {p["input"].shape for p in puzzle["train"]}
        test_shapes = {p["input"].shape for p in puzzle["test"]}
        flags["train_all_four"] += len(train_dirs) == 4
        flags["test_familiar"] += bool(test_dirs & train_dirs)
        flags["test_heldout"] += bool(test_dirs - train_dirs)
        flags["single_train"] += any(len(a["markers"]) == 1 for a in train_analyses)
        flags["multi_train"] += any(len(a["markers"]) >= 3 for a in train_analyses)
        flags["single_test"] += any(len(a["markers"]) == 1 for a in test_analyses)
        flags["large_count"] += any(len(a["markers"]) > 4 for a in train_analyses + test_analyses)
        flags["shared_color"] += bool(train_colors & test_colors)
        flags["variable_background"] += len({a["background"] for a in train_analyses + test_analyses}) > 1
        flags["small_test"] += min(h * w for h, w in test_shapes) < max(h * w for h, w in train_shapes)
        flags["equal_size"] += bool(train_shapes & test_shapes)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                analysis = _analyze_input(grid)
                assert np.array_equal(pair["output"], analysis["expected_output"])
                metrics["shape"].add(grid.shape)
                metrics["layout"].add((grid.shape, np.packbits(grid != analysis["background"]).tobytes()))
                metrics["background"].add(analysis["background"])
                metrics["marker_count"].add(len(analysis["markers"]))
                metrics["palette"].add(tuple(sorted(set(int(x) for x in grid.flat))))
                flags["ten_square"] += grid.shape == (10, 10)
                flags["large_canvas"] += max(grid.shape) >= 25
                flags["multicolor"] += len({m["color"] for m in analysis["markers"]}) > 1
                markers = analysis["markers"]
                for m in markers:
                    metrics["orientation"].add(m["orientation"])
                    metrics["ray_length"].add(len(m["ray"]))
                    flags["short_ray"] += len(m["ray"]) == 1
                for i, m in enumerate(markers):
                    for other in markers[i + 1:]:
                        flags["close"] += _minimum_chebyshev(m["filled"], other["filled"]) <= 2
                        flags["cross"] += bool(set(m["ray"]) & set(other["ray"]))
    assert len(metrics["train_count"]) >= 4 and len(metrics["test_count"]) >= 3
    assert len(metrics["shape"]) >= 35 and len(metrics["layout"]) >= 100
    assert len(metrics["background"]) >= 8 and len(metrics["palette"]) >= 20
    assert metrics["orientation"] == set(_DIRECTIONS)
    assert max(metrics["marker_count"]) > 4 and 1 in metrics["marker_count"]
    assert 1 in metrics["ray_length"] and max(metrics["ray_length"]) >= 10
    assert all(value > 0 for value in flags.values())
