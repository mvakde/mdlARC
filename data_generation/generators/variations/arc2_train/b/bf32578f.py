import numpy as np


def _parse_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(cell) for cell in row] for row in rows], dtype=np.int8)


_SOURCE_RAW = (
    (("888000", "800000", "800000", "800000", "888000", "000000"),
     ("000000", "088880", "088880", "088880", "000000", "000000")),
    (("0000000000", "0000000000", "0000000000", "0000700000",
      "0007000000", "0070000000", "0070000000", "0007000000",
      "0000700000", "0000000000"),
     ("0000000000", "0000000000", "0000000000", "0000000000",
      "0000770000", "0007777000", "0007777000", "0000770000",
      "0000000000", "0000000000")),
    (("000000", "006000", "060000", "600000", "060000", "006000"),
     ("000000", "000000", "006600", "066660", "006600", "000000")),
    (("00000000", "00010000", "01100000", "10000000", "10000000",
      "01100000", "00010000", "00000000"),
     ("00000000", "00000000", "00011000", "01111110", "01111110",
      "00011000", "00000000", "00000000")),
)
_SOURCE_PAIRS = tuple(
    (_parse_grid(input_rows), _parse_grid(output_rows))
    for input_rows, output_rows in _SOURCE_RAW
)


def _grid_signature(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_signature(pair: dict) -> tuple:
    return (_grid_signature(pair["input"]), _grid_signature(pair["output"]))


def _matches_source_pair(pair: dict) -> bool:
    return any(
        np.array_equal(pair["input"], original_input)
        and np.array_equal(pair["output"], original_output)
        for original_input, original_output in _SOURCE_PAIRS
    )


def _orthogonal_boundary(filled: np.ndarray) -> np.ndarray:
    up = np.zeros_like(filled)
    down = np.zeros_like(filled)
    left = np.zeros_like(filled)
    right = np.zeros_like(filled)
    up[1:] = filled[:-1]
    down[:-1] = filled[1:]
    left[:, 1:] = filled[:, :-1]
    right[:, :-1] = filled[:, 1:]
    return filled & ~(up & down & left & right)


def _flood_exterior(boundary: np.ndarray) -> np.ndarray:
    height, width = boundary.shape
    exterior = np.zeros_like(boundary)
    stack: list[tuple[int, int]] = []

    def add(row: int, column: int) -> None:
        if not boundary[row, column] and not exterior[row, column]:
            exterior[row, column] = True
            stack.append((row, column))

    for column in range(width):
        add(0, column)
        add(height - 1, column)
    for row in range(height):
        add(row, 0)
        add(row, width - 1)
    while stack:
        row, column = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, column + dc
            if 0 <= nr < height and 0 <= nc < width:
                add(nr, nc)
    return exterior


def _is_connected(mask: np.ndarray, diagonal: bool) -> bool:
    locations = np.argwhere(mask)
    if not len(locations):
        return False
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    if diagonal:
        directions += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    height, width = mask.shape
    start = tuple(int(x) for x in locations[0])
    seen = {start}
    stack = [start]
    while stack:
        row, column = stack.pop()
        for dr, dc in directions:
            neighbor = (row + dr, column + dc)
            nr, nc = neighbor
            if (0 <= nr < height and 0 <= nc < width
                    and mask[neighbor] and neighbor not in seen):
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == int(mask.sum())


def _infer_colors(grid: np.ndarray) -> tuple[int, int]:
    values, counts = np.unique(grid, return_counts=True)
    if len(values) != 2 or counts[0] == counts[1]:
        raise ValueError("Input requires two colors and a dominant background.")
    background_index = int(np.argmax(counts))
    return int(values[background_index]), int(values[1 - background_index])


def _solve_grid(input_grid: np.ndarray) -> np.ndarray:
    background, foreground = _infer_colors(input_grid)
    half = input_grid == foreground
    rows, columns = np.nonzero(half)
    if not len(rows):
        raise ValueError("Missing half-outline.")
    seam = int(columns.max())
    reflected = 2 * seam + 1 - columns
    if reflected.min() < 0 or reflected.max() >= input_grid.shape[1]:
        raise ValueError("The reflected half does not fit.")
    outline = half.copy()
    outline[rows, reflected] = True
    interior = ~outline & ~_flood_exterior(outline)
    result = np.full_like(input_grid, background)
    result[interior] = foreground
    return result


def _predict_from_visible_rows(input_grid: np.ndarray) -> np.ndarray:
    """Independent row-fill prediction using only the visible input."""
    background, foreground = _infer_colors(input_grid)
    half = input_grid == foreground
    seam = int(np.argwhere(half)[:, 1].max())
    filled = np.zeros_like(half)
    for row in range(half.shape[0]):
        columns = np.flatnonzero(half[row])
        if len(columns):
            left = int(columns.min())
            filled[row, left:2 * seam + 2 - left] = True
    interior = filled & ~_orthogonal_boundary(filled)
    result = np.full_like(input_grid, background)
    result[interior] = foreground
    return result


def _checked_count(name: str, value: int | None, lower: int, upper: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None.")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}.")
    return value


def _resolve_counts(seed: int, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is not None and test is not None and train <= test:
        raise ValueError("num_train must exceed num_test.")
    key = int(seed)
    if train is None and test is None:
        test = 1 + key % 4
        train = test + 1 + (key // 4) % (10 - test)
    elif train is None:
        train = test + 1 + key % (10 - test)
    elif test is None:
        test = 1 + key % min(4, train - 1)
    return train, test


def _sample_color_schedule(rng: np.random.Generator, train_count: int,
                           test_count: int) -> tuple[list[tuple[int, int]],
                                                     list[tuple[int, int]]]:
    colors = [int(x) for x in rng.permutation(10)]
    varied_background = bool(rng.integers(0, 2))
    train = [(colors[0], colors[1]),
             (colors[2] if varied_background else colors[0], colors[3])]
    for _ in range(2, train_count):
        background = int(rng.integers(0, 10)) if varied_background else colors[0]
        choices = [color for color in range(10) if color != background]
        train.append((background, int(rng.choice(choices))))

    foregrounds = {foreground for _, foreground in train}
    test = []
    reuse_mode = bool(rng.integers(0, 2))
    for index in range(test_count):
        background = int(rng.integers(0, 10)) if varied_background else colors[0]
        if varied_background and index == 0 and int(rng.integers(0, 3)) == 0:
            background = train[0][1]
            foreground = train[0][0]
        else:
            eligible = [color for color in range(10) if color != background]
            if index == 0:
                preferred = [color for color in eligible
                             if (color in foregrounds) == reuse_mode]
                if preferred:
                    eligible = preferred
            foreground = int(rng.choice(eligible))
        test.append((background, foreground))
    return train, test


def _shoulder(rng: np.random.Generator, start: int, peak: int,
              stages: int) -> list[int]:
    middle = sorted(int(x) for x in rng.integers(start, peak + 1,
                                                 size=max(0, stages - 1)))
    return [start] + middle + [peak]


def _sample_profile(rng: np.random.Generator, kind: str,
                    scale: str) -> list[int]:
    if scale == "compact":
        radius = int(rng.integers(1, 4))
        stage_limit, plateau_limit = 2, 2
    elif scale == "tall":
        radius = int(rng.integers(1, 3))
        stage_limit, plateau_limit = 3, 4
    elif scale == "wide":
        radius = int(rng.integers(4, 9))
        stage_limit, plateau_limit = 2, 3
    else:
        radius = int(rng.integers(1, 9))
        stage_limit, plateau_limit = 5, 4

    if kind == "rectangle":
        lengths = {"compact": (3, 9), "tall": (12, 22),
                   "wide": (3, 8), "general": (3, 16)}
        low, high = lengths[scale]
        return [radius] * int(rng.integers(low, high))

    plateau = int(rng.integers(1, plateau_limit + 1))
    if kind == "stepped":
        upper = [0, radius]
        lower = [radius, 0]
        if radius >= 3 and bool(rng.integers(0, 2)):
            upper.insert(1, int(rng.integers(1, radius)))
        if radius >= 3 and bool(rng.integers(0, 2)):
            lower.insert(1, int(rng.integers(1, radius)))
    else:
        upper_start = 0
        lower_end = 0
        if kind in ("asymmetric", "waist") and radius > 1:
            upper_start = int(rng.integers(0, min(3, radius)))
            lower_end = int(rng.integers(0, min(3, radius)))
        upper = _shoulder(rng, upper_start, radius,
                          int(rng.integers(1, stage_limit + 1)))
        lower = _shoulder(rng, lower_end, radius,
                          int(rng.integers(1, stage_limit + 1)))[::-1]
        if kind == "taper" and bool(rng.integers(0, 2)):
            lower = upper[::-1]

    if kind == "waist":
        valley = max(1, radius - int(rng.integers(1, min(4, radius + 1))))
        profile = (upper[:-1] + [radius] * plateau
                   + [valley] * int(rng.integers(1, 3))
                   + [radius] * int(rng.integers(1, 3)) + lower[1:])
    else:
        profile = upper[:-1] + [radius] * plateau + lower[1:]
    if kind == "asymmetric" and profile == profile[::-1]:
        profile.insert(1, profile[1])
    if scale == "tall":
        target = int(rng.integers(12, 21))
        while len(profile) < target:
            index = int(rng.integers(1, len(profile)))
            profile.insert(index, profile[index - 1])
    return profile


def _sample_margins(rng: np.random.Generator, extent: int,
                    scale: str) -> tuple[int, int]:
    cap = {"compact": 3, "tall": 3, "wide": 4, "general": 6}[scale]
    maximum = 10 if scale == "compact" else 30
    choices = [(a, b) for a in range(cap + 1) for b in range(cap + 1)
               if 6 <= extent + a + b <= maximum]
    if not choices:
        raise ValueError("No valid margins for sampled geometry.")
    # Zero margins have extra weight, including two-edge contact.
    weights = np.array([1 + 2 * (a == 0) + 2 * (b == 0)
                        for a, b in choices], dtype=float)
    weights /= weights.sum()
    return choices[int(rng.choice(len(choices), p=weights))]


def _render_pair(rng: np.random.Generator, profile: list[int],
                 background: int, foreground: int,
                 scale: str) -> tuple[np.ndarray, np.ndarray]:
    radius = max(profile)
    top, bottom = _sample_margins(rng, len(profile), scale)
    left, right = _sample_margins(rng, 2 * radius + 2, scale)
    height = top + len(profile) + bottom
    width = left + 2 * radius + 2 + right
    seam = left + radius
    filled = np.zeros((height, width), dtype=bool)
    for offset, row_radius in enumerate(profile):
        filled[top + offset,
               seam - row_radius:seam + row_radius + 2] = True
    boundary = _orthogonal_boundary(filled)
    half = boundary.copy()
    half[:, seam + 1:] = False
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[half] = foreground
    output_grid = np.full((height, width), background, dtype=np.int8)
    output_grid[filled & ~boundary] = foreground
    return input_grid, output_grid


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    assert input_grid.shape == output_grid.shape
    assert not np.array_equal(input_grid, output_grid)
    background, foreground = _infer_colors(input_grid)
    assert set(int(x) for x in np.unique(output_grid)) == {background, foreground}
    half = input_grid == foreground
    rows, columns = np.nonzero(half)
    seam = int(columns.max())
    reflected = 2 * seam + 1 - columns
    assert reflected.min() >= 0 and reflected.max() < input_grid.shape[1]
    boundary = half.copy()
    boundary[rows, reflected] = True
    exterior = _flood_exterior(boundary)
    interior = ~boundary & ~exterior
    assert interior.any() and _is_connected(interior, diagonal=False)
    expected = np.full_like(input_grid, background)
    expected[interior] = foreground
    assert np.array_equal(output_grid, expected)
    assert np.array_equal(output_grid, _predict_from_visible_rows(input_grid))
    filled = boundary | interior
    assert np.array_equal(boundary, _orthogonal_boundary(filled))
    assert _is_connected(half, diagonal=True)
    assert _is_connected(boundary, diagonal=True)
    occupied = np.flatnonzero(filled.any(axis=1))
    assert len(occupied) >= 3 and np.all(np.diff(occupied) == 1)
    assert np.array_equal(occupied, np.flatnonzero(half.any(axis=1)))
    profile = []
    for row in occupied:
        full_columns = np.flatnonzero(filled[row])
        half_columns = np.flatnonzero(half[row])
        assert len(full_columns) and np.all(np.diff(full_columns) == 1)
        assert len(half_columns) and np.all(np.diff(half_columns) == 1)
        assert int(full_columns[0] + full_columns[-1]) == 2 * seam + 1
        assert int(half_columns[-1]) <= seam
        profile.append(seam - int(full_columns[0]))
    assert half[int(occupied[0]), seam] and half[int(occupied[-1]), seam]
    return {"background": background, "foreground": foreground,
            "profile": tuple(profile), "top": int(occupied[0]),
            "bottom": input_grid.shape[0] - int(occupied[-1]) - 1,
            "left": seam - max(profile),
            "right": input_grid.shape[1] - (seam + max(profile) + 2),
            "height": input_grid.shape[0], "width": input_grid.shape[1]}


def _profile_features(profile: tuple[int, ...]) -> dict:
    changes = [profile[i + 1] - profile[i] for i in range(len(profile) - 1)]
    return {"rectangle": len(set(profile)) == 1,
            "asymmetric": profile != profile[::-1],
            "stepped": any(abs(change) >= 2 for change in changes),
            "waist": any(profile[i] < profile[i - 1] and
                         profile[i] < profile[i + 1]
                         for i in range(1, len(profile) - 1)),
            "pointed": profile[0] == 0 or profile[-1] == 0}


def _validate_generated_puzzle_impl(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs: set[tuple] = set()
    pairs: set[tuple] = set()
    analyses: dict[str, list[dict]] = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            input_signature = _grid_signature(pair["input"])
            pair_signature = _pair_signature(pair)
            assert input_signature not in inputs and pair_signature not in pairs
            inputs.add(input_signature)
            pairs.add(pair_signature)
            assert not _matches_source_pair(pair)
            analyses[split].append(_analyze_pair(pair["input"], pair["output"]))

    train = analyses["train"]
    assert len({item["foreground"] for item in train}) >= 2
    assert any(_profile_features(item["profile"])["rectangle"] for item in train)
    assert any(not _profile_features(item["profile"])["rectangle"] for item in train)
    # The only shared convention is the demonstrated left-half reflection rule.
    # Color roles are visible separately in each input, including every test input.
    for item in train + analyses["test"]:
        assert 6 <= item["height"] <= 30 and 6 <= item["width"] <= 30
        assert item["left"] >= 0 and item["right"] >= 0


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError("Malformed generated puzzle.") from error


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    train_colors, test_colors = _sample_color_schedule(rng, train_count, test_count)
    contour_kinds = ("taper", "stepped", "asymmetric", "waist")
    first_train_kinds = ["rectangle", str(rng.choice(contour_kinds))]
    rng.shuffle(first_train_kinds)
    train_kinds = first_train_kinds + [
        str(rng.choice(("rectangle",) + contour_kinds))
        for _ in range(train_count - 2)]
    test_kinds = [str(rng.choice(("rectangle",) + contour_kinds,
                                 p=(0.2, 0.2, 0.2, 0.2, 0.2)))
                  for _ in range(test_count)]
    puzzle: dict[str, list[dict]] = {"train": [], "test": []}
    used_inputs: set[tuple] = set()
    for split, colors, kinds in (("train", train_colors, train_kinds),
                                 ("test", test_colors, test_kinds)):
        for (background, foreground), kind in zip(colors, kinds):
            accepted = None
            for _ in range(128):
                scale_weights = ((0.32, 0.24, 0.13, 0.31)
                                 if split == "train" else
                                 (0.22, 0.17, 0.32, 0.29))
                scale = str(rng.choice(("compact", "tall", "wide", "general"),
                                       p=scale_weights))
                try:
                    profile = _sample_profile(rng, kind, scale)
                    input_grid, output_grid = _render_pair(
                        rng, profile, background, foreground, scale)
                    candidate = {"input": input_grid, "output": output_grid}
                    signature = _grid_signature(input_grid)
                    if signature in used_inputs or _matches_source_pair(candidate):
                        continue
                    analysis = _analyze_pair(input_grid, output_grid)
                    if kind == "asymmetric" and not _profile_features(
                            analysis["profile"])["asymmetric"]:
                        continue
                    if kind == "waist" and not _profile_features(
                            analysis["profile"])["waist"]:
                        continue
                    accepted = candidate
                    used_inputs.add(signature)
                    break
                except (AssertionError, ValueError):
                    continue
            if accepted is None:
                raise ValueError("Unable to construct a valid distinct pair.")
            puzzle[split].append(accepted)
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[split]) == len(second[split]) and
        all(np.array_equal(a[name], b[name]) and a[name].dtype == b[name].dtype
            for a, b in zip(first[split], second[split])
            for name in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    assert len({tuple(_pair_signature(pair) for split in ("train", "test")
                      for pair in puzzle[split]) for puzzle in puzzles}) >= 40
    profiles: set[tuple[int, ...]] = set()
    layouts: set[tuple] = set()
    dimensions: set[tuple[int, int]] = set()
    colors: set[tuple[int, int]] = set()
    feature_counts = {name: 0 for name in
                      ("rectangle", "asymmetric", "stepped", "waist", "pointed")}
    edge_sides = {name: 0 for name in ("top", "bottom", "left", "right")}
    edge_count = compact_count = tall_count = wide_count = 0
    changed_background_puzzles = reused_foreground_puzzles = 0
    simple_test_after_complex_train = 0
    wider_test_puzzles = swapped_color_puzzles = 0
    first_train_nonrectangles = test_rectangles = 0
    flat_ended_count = repeated_rows_count = horizontal_runs_count = 0
    compact_test_count = large_test_count = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_analyze_pair(pair["input"], pair["output"])
                 for pair in puzzle["train"]]
        test = [_analyze_pair(pair["input"], pair["output"])
                for pair in puzzle["test"]]
        first_train_nonrectangles += int(not _profile_features(
            train[0]["profile"])["rectangle"])
        test_rectangles += sum(_profile_features(item["profile"])["rectangle"]
                               for item in test)
        wider_test_puzzles += int(max(max(item["profile"]) for item in test)
                                  > max(max(item["profile"]) for item in train))
        swapped_color_puzzles += int(any(
            (test_item["background"], test_item["foreground"])
            == (train_item["foreground"], train_item["background"])
            for train_item in train for test_item in test))
        if len({item["background"] for item in train + test}) > 1:
            changed_background_puzzles += 1
        if {item["foreground"] for item in train} & {
                item["foreground"] for item in test}:
            reused_foreground_puzzles += 1
        if any(_profile_features(item["profile"])["asymmetric"] or
               _profile_features(item["profile"])["stepped"] for item in train):
            if any(_profile_features(item["profile"])["rectangle"] or
                   (max(item["profile"]) <= 2 and
                    not _profile_features(item["profile"])["rectangle"])
                   for item in test):
                simple_test_after_complex_train += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                item = _analyze_pair(pair["input"], pair["output"])
                assert np.array_equal(pair["output"],
                                      _predict_from_visible_rows(pair["input"]))
                profile = item["profile"]
                profiles.add(profile)
                dimensions.add((item["height"], item["width"]))
                colors.add((item["background"], item["foreground"]))
                layouts.add((profile, item["top"], item["left"],
                             item["height"], item["width"]))
                for name, present in _profile_features(profile).items():
                    feature_counts[name] += int(present)
                edge_count += int(min(item["top"], item["bottom"],
                                      item["left"], item["right"]) == 0)
                for side in edge_sides:
                    edge_sides[side] += int(item[side] == 0)
                compact_count += int(item["height"] <= 10 and item["width"] <= 10)
                tall_count += int(item["height"] >= 1.5 * item["width"])
                wide_count += int(item["width"] >= 1.5 * item["height"])
                flat_ended_count += int(profile[0] > 0 and profile[-1] > 0)
                repeated_rows_count += int(any(
                    profile[index] == profile[index + 1]
                    for index in range(len(profile) - 1)))
                foreground_mask = pair["input"] == item["foreground"]
                horizontal_runs_count += int(any(
                    foreground_mask[row].sum() >= 2
                    for row in range(foreground_mask.shape[0])))
                if split == "test":
                    compact_test_count += int(item["height"] <= 10
                                              and item["width"] <= 10)
                    large_test_count += int(item["height"] >= 18
                                            or item["width"] >= 18)
    assert len(profiles) >= 45 and len(layouts) >= 100
    assert len(dimensions) >= 35 and len(colors) >= 20
    assert all(count >= 6 for count in feature_counts.values())
    assert edge_count >= 20 and compact_count >= 12
    assert tall_count >= 8 and wide_count >= 8
    assert min(edge_sides.values()) >= 4
    assert flat_ended_count >= 8 and repeated_rows_count >= 10
    assert horizontal_runs_count >= 10
    assert compact_test_count >= 3 and large_test_count >= 3
    assert changed_background_puzzles >= 8
    assert reused_foreground_puzzles >= 8
    assert swapped_color_puzzles >= 1
    assert simple_test_after_complex_train >= 3
    assert wider_test_puzzles >= 3
    assert first_train_nonrectangles >= 4 and test_rectangles >= 3
    assert _puzzles_equal(generate_puzzle(314159), generate_puzzle(314159))
    assert _puzzles_equal(generate_puzzle(271828, 7, 3),
                          generate_puzzle(271828, 7, 3))
    assert not _puzzles_equal(generate_puzzle(271828, 7, 3),
                              generate_puzzle(271829, 7, 3))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(1000 + train_count + test_count,
                                 train_count, test_count)
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for train_count, test_count in ((None, 1), (None, 4),
                                    (2, None), (10, None)):
        puzzle = generate_puzzle(2000 + (train_count or 0) + (test_count or 0),
                                 train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for arguments in ({"num_train": 1}, {"num_train": 11},
                      {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": True}, {"num_test": 1.5}):
        try:
            generate_puzzle(9999, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {arguments}.")
