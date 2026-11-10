import numpy as np


# A profile records the visible height or width of copies on opposite sides.
# The value 2 also permits larger margins, but copies remain exactly 2x2.
_AXIS_PROFILES = ((0, 2), (1, 1), (2, 0), (1, 2), (2, 1), (2, 2))
_FULL = (2, 2, 2, 2)
_PROFILES = tuple(
    (top, bottom, left, right)
    for top, bottom in _AXIS_PROFILES
    for left, right in _AXIS_PROFILES
)
_SINGLE_PARTIAL = tuple(
    profile for profile in _PROFILES
    if sorted(profile) == [1, 2, 2, 2]
)
_MIXED = tuple(
    profile for profile in _PROFILES
    if sum(value < 2 for value in profile) >= 2
)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _normalize_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed


def _choose_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _normalize_count(num_train, "num_train", 2, 10)
    test = _normalize_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    _require(int(np.max(counts)) == grid.size - 4, "input must have one uniform background")
    return background


def _locate_seed(grid: np.ndarray) -> tuple[int, int, np.ndarray, int]:
    background = _background(grid)
    cells = np.argwhere(grid != background)
    _require(cells.shape == (4, 2), "input must have exactly four seed cells")
    row, col = (int(cells[:, axis].min()) for axis in (0, 1))
    expected = {(row + dr, col + dc) for dr in range(2) for dc in range(2)}
    actual = {(int(r), int(c)) for r, c in cells}
    _require(actual == expected, "seed must be a complete 2x2 square")
    seed = grid[row:row + 2, col:col + 2].copy()
    _require(len(set(int(x) for x in seed.flat)) == 4, "seed colors must be distinct")
    return row, col, seed, background


def _construct_output(input_grid: np.ndarray) -> np.ndarray:
    row, col, seed, _ = _locate_seed(input_grid)
    height, width = input_grid.shape
    output = input_grid.copy()
    # Each cell travels to the region diagonally opposite its own corner.
    for source_row in range(2):
        for source_col in range(2):
            dest_row = row + (2 if source_row == 0 else -2)
            dest_col = col + (2 if source_col == 0 else -2)
            r0, r1 = max(0, dest_row), min(height, dest_row + 2)
            c0, c1 = max(0, dest_col), min(width, dest_col + 2)
            if r0 < r1 and c0 < c1:
                output[r0:r1, c0:c1] = seed[source_row, source_col]
    return output


def _visibility(height: int, width: int, row: int, col: int) -> tuple[int, int, int, int]:
    return (
        min(2, row),
        min(2, height - row - 2),
        min(2, col),
        min(2, width - col - 2),
    )


def _check_rule(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple[int, int, int, tuple]:
    """Check every destination cell independently of the array-slice constructor."""
    row, col, seed, background = _locate_seed(input_grid)
    height, width = input_grid.shape
    expected = np.full((height, width), background, dtype=np.int8)
    expected[row:row + 2, col:col + 2] = seed
    regions = (
        (row - 2, col - 2, int(seed[1, 1])),
        (row - 2, col + 2, int(seed[1, 0])),
        (row + 2, col - 2, int(seed[0, 1])),
        (row + 2, col + 2, int(seed[0, 0])),
    )
    for r0, c0, color in regions:
        for dr in range(2):
            for dc in range(2):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < height and 0 <= c < width:
                    expected[r, c] = color
    _require(np.array_equal(output_grid, expected), "opposite-corner copies or clipping are wrong")
    _require(not np.array_equal(input_grid, output_grid), "pair must show a visible copy")
    return row, col, background, _visibility(height, width, row, col)


def _array_from_rows(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[int(char) for char in line] for line in rows], dtype=np.int8)


def _source_pairs() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    return (
        (
            _array_from_rows(("000000", "046000", "021000", "000000", "000000", "000000")),
            _array_from_rows(("100220", "046000", "021000", "600440", "600440", "000000")),
        ),
        (
            _array_from_rows(("000000", "000000", "009300", "007800", "000000", "000000")),
            _array_from_rows(("880077", "880077", "009300", "007800", "330099", "330099")),
        ),
        (
            _array_from_rows(("000000", "000000", "003600", "005200", "000000", "000000")),
            _array_from_rows(("220055", "220055", "003600", "005200", "660033", "660033")),
        ),
        (
            _array_from_rows(("000000", "000000", "000000", "003100", "002500", "000000")),
            _array_from_rows(("000000", "550022", "550022", "003100", "002500", "110033")),
        ),
    )


def _array_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(x) for x in grid.flat)


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _array_key(input_grid), _array_key(output_grid)


def _source_keys() -> set[tuple]:
    return {_pair_key(a, b) for a, b in _source_pairs()}


def sample_puzzle_parameters(rng: np.random.Generator, train_count: int) -> dict:
    """The geometric rule is fixed; only local backgrounds and presentation vary."""
    base_background = 0 if rng.random() < 0.55 else int(rng.integers(1, 10))
    alternatives = [color for color in range(10) if color != base_background]
    alternate_background = int(rng.choice(alternatives))
    return {
        "base_background": base_background,
        "alternate_background": alternate_background,
        "vary_background": train_count >= 4 and bool(rng.random() < 0.65),
    }


def _actual_margin(rng: np.random.Generator, nominal: int, compact: bool) -> int:
    if nominal < 2:
        return nominal
    if compact or rng.random() < 0.23:
        return 2
    return int(rng.integers(3, 8))


def _geometry(
    rng: np.random.Generator, profile: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    compact = bool(rng.random() < 0.20)
    top, bottom, left, right = (
        _actual_margin(rng, nominal, compact) for nominal in profile
    )
    height, width = top + 2 + bottom, left + 2 + right
    return height, width, top, left


def _make_pair(
    rng: np.random.Generator,
    profile_options: tuple[tuple[int, int, int, int], ...],
    background: int,
    used_geometry: set[tuple],
    seen_inputs: set[tuple],
    forbidden: set[tuple],
    required_new_shape: tuple[int, int] | None = None,
) -> dict:
    # Several profiles and margin realizations are available even at legal
    # count boundaries. The deterministic scan is a last resort.
    for attempt in range(256):
        profile = profile_options[int(rng.integers(len(profile_options)))]
        geometry = _geometry(rng, profile)
        height, width, row, col = geometry
        if geometry in used_geometry or (required_new_shape == (height, width)):
            continue
        palette = rng.choice([c for c in range(10) if c != background], 4, replace=False)
        input_grid = np.full((height, width), background, dtype=np.int8)
        input_grid[row:row + 2, col:col + 2] = palette.reshape(2, 2)
        output_grid = _construct_output(input_grid)
        input_key = _array_key(input_grid)
        if input_key in seen_inputs or _pair_key(input_grid, output_grid) in forbidden:
            continue
        used_geometry.add(geometry)
        seen_inputs.add(input_key)
        return {"input": input_grid, "output": output_grid}
    for profile in profile_options:
        for large_margin in range(2, 8):
            margins = tuple(large_margin if x == 2 else x for x in profile)
            top, bottom, left, right = margins
            geometry = (top + bottom + 2, left + right + 2, top, left)
            height, width, row, col = geometry
            if geometry in used_geometry or required_new_shape == (height, width):
                continue
            palette = rng.choice([c for c in range(10) if c != background], 4, replace=False)
            input_grid = np.full((height, width), background, dtype=np.int8)
            input_grid[row:row + 2, col:col + 2] = palette.reshape(2, 2)
            output_grid = _construct_output(input_grid)
            if _array_key(input_grid) in seen_inputs or _pair_key(input_grid, output_grid) in forbidden:
                continue
            used_geometry.add(geometry)
            seen_inputs.add(_array_key(input_grid))
            return {"input": input_grid, "output": output_grid}
    raise ValueError("no unused geometry remained for the requested profile")


def _training_profiles(rng: np.random.Generator, count: int) -> list[tuple]:
    partial = _SINGLE_PARTIAL[int(rng.integers(len(_SINGLE_PARTIAL)))]
    profiles = [_FULL, partial]
    pool = tuple(p for p in _PROFILES if p not in (_FULL, partial, (1, 1, 1, 1)))
    if count > 2:
        chosen = rng.choice(len(pool), size=count - 2, replace=False)
        profiles.extend(pool[int(index)] for index in chosen)
    return profiles


def _test_profile_options(
    mode: int,
    train_profiles: list[tuple],
) -> tuple[tuple, ...]:
    if mode == 0:
        # Source-like tests may be complete or repeat a demonstrated clipping.
        return (_FULL, train_profiles[1])
    if mode == 1:
        # A different side or a zero-visibility version of a familiar side.
        options = tuple(p for p in _PROFILES if p != _FULL and p not in train_profiles
                        and sum(x < 2 for x in p) == 1)
        return options or tuple(p for p in _PROFILES if p != _FULL)
    options = tuple(p for p in _MIXED if p not in train_profiles)
    return options or _MIXED


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    train_count, test_count = _choose_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    parameters = sample_puzzle_parameters(rng, train_count)
    train_profiles = _training_profiles(rng, train_count)
    mode = seed % 3
    test_options = _test_profile_options(mode, train_profiles)
    forbidden = _source_keys()
    used_geometry: set[tuple] = set()
    seen_inputs: set[tuple] = set()
    puzzle = {"train": [], "test": []}

    backgrounds = [parameters["base_background"]] * (train_count + test_count)
    if parameters["vary_background"]:
        backgrounds[1] = parameters["alternate_background"]
        for index in range(2, len(backgrounds)):
            backgrounds[index] = (
                parameters["base_background"]
                if rng.random() < 0.5 else parameters["alternate_background"]
            )

    first_shape = None
    for index, profile in enumerate(train_profiles):
        child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        pair = _make_pair(
            child, (profile,), backgrounds[index], used_geometry, seen_inputs,
            forbidden, first_shape if index == 1 else None,
        )
        puzzle["train"].append(pair)
        if index == 0:
            first_shape = pair["input"].shape

    for index in range(test_count):
        child = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        options = ((1, 1, 1, 1),) if mode == 2 and seed % 9 == 2 and index == 0 else test_options
        pair = _make_pair(
            child, options, backgrounds[train_count + index],
            used_geometry, seen_inputs, forbidden,
        )
        puzzle["test"].append(pair)

    order = rng.permutation(train_count)
    puzzle["train"] = [puzzle["train"][int(index)] for index in order]

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")

    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    seen_geometry: set[tuple] = set()
    train_shapes: set[tuple] = set()
    has_full = has_partial = False
    forbidden = _source_keys()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have input and output")
            for key in ("input", "output"):
                grid = pair[key]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be 2D NumPy arrays")
                h, w = grid.shape
                _require(4 <= h <= 16 and 4 <= w <= 16, "canvas is outside generated range")
                _require(np.issubdtype(grid.dtype, np.integer), "grids must use integer dtype")
                _require(0 <= int(grid.min()) and int(grid.max()) <= 9,
                         "grids must use ARC colors")
            input_grid, output_grid = pair["input"], pair["output"]
            _require(input_grid.shape == output_grid.shape, "output must keep canvas size")
            row, col, background, profile = _check_rule(input_grid, output_grid)
            _require(len(np.unique(input_grid)) == 5, "input needs background and four colors")
            _require(set(int(x) for x in np.unique(output_grid))
                     == set(int(x) for x in np.unique(input_grid)),
                     "output must preserve its local palette")
            geometry = (*input_grid.shape, row, col)
            _require(geometry not in seen_geometry, "repeated structural layout within puzzle")
            seen_geometry.add(geometry)
            ikey, pkey = _array_key(input_grid), _pair_key(input_grid, output_grid)
            _require(ikey not in seen_inputs and pkey not in seen_pairs,
                     "duplicate pair within puzzle")
            _require(pkey not in forbidden, "generated pair duplicates a source pair")
            seen_inputs.add(ikey)
            seen_pairs.add(pkey)
            if split == "train":
                train_shapes.add(input_grid.shape)
                has_full |= profile == _FULL
                has_partial |= 1 in profile
    _require(has_full and has_partial,
             "training must show four complete copies and a partially clipped copy")
    _require(len(train_shapes) >= 2, "training must vary canvas dimensions")


def _puzzles_equal(first: dict, second: dict) -> bool:
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for left, right in zip(first[split], second[split]):
            for key in ("input", "output"):
                if not np.array_equal(left[key], right[key]):
                    return False
    return True


def self_test() -> None:
    for source_input, source_output in _source_pairs():
        _require(np.array_equal(_construct_output(source_input), source_output),
                 "rule does not reproduce the source task")

    # Hand-derived 4x4 case: every copy is reduced to one cell.
    compact = _array_from_rows(("0000", "0120", "0340", "0000"))
    expected = _array_from_rows(("4003", "0120", "0340", "2001"))
    _require(np.array_equal(_construct_output(compact), expected),
             "four-sided 4x4 clipping failed")
    for side in range(4):
        for visible in range(3):
            h = w = 8
            row, col = 3, 3
            if side == 0:
                row = visible
            elif side == 1:
                row = h - 2 - visible
            elif side == 2:
                col = visible
            else:
                col = w - 2 - visible
            grid = np.zeros((h, w), dtype=np.int8)
            grid[row:row + 2, col:col + 2] = ((1, 2), (3, 4))
            output = _construct_output(grid)
            _require(_check_rule(grid, output)[3][side] == visible,
                     "zero/one/two-cell side visibility failed")

    cases = (
        (1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
        (1003, None, 1), (1004, None, 4),
        (1005, 2, None), (1006, 10, None), (1007, 7, 3),
    )
    automatic = [generate_puzzle(seed) for seed in range(42)]
    requested = []
    for seed, nt, ns in cases:
        puzzle = generate_puzzle(seed, nt, ns)
        _require(nt is None or len(puzzle["train"]) == nt, "num_train not honored")
        _require(ns is None or len(puzzle["test"]) == ns, "num_test not honored")
        requested.append(puzzle)
    for seed, nt, ns in ((2000, None, None), (2001, 7, 3), (2002, None, 4)):
        _require(_puzzles_equal(generate_puzzle(seed, nt, ns),
                               generate_puzzle(seed, nt, ns)),
                 "generation is not deterministic")
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 2, "num_test": 4}, {"num_train": 2.0},
        {"num_test": True},
    ):
        try:
            generate_puzzle(3000, **arguments)
        except ValueError:
            continue
        raise AssertionError(f"illegal count accepted: {arguments}")

    all_puzzles = automatic + requested
    for puzzle in all_puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                _check_rule(pair["input"], pair["output"])
    _require(len({len(p["train"]) for p in automatic}) >= 4, "few train counts")
    _require(len({len(p["test"]) for p in automatic}) >= 3, "few test counts")

    shapes, profiles, layouts, palettes, backgrounds = set(), set(), set(), set(), set()
    side_levels = [set() for _ in range(4)]
    single_side = [0] * 4
    train_full = train_clipped = test_full = test_clipped = 0
    compact_four_clipped = 0
    varying_background_puzzles = 0
    varying_shape_puzzles = 0
    seen_test_profiles = unseen_test_profiles = 0
    offcenter_full = 0
    wide = tall = compact_five = 0
    zero_as_seed = 0
    for puzzle in automatic:
        puzzle_backgrounds = set()
        puzzle_shapes = set()
        training_profiles = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                row, col, seed_grid, background = _locate_seed(grid)
                height, width = grid.shape
                profile = _visibility(height, width, row, col)
                shapes.add(grid.shape)
                profiles.add(profile)
                layouts.add((*grid.shape, row, col))
                palettes.add(tuple(int(x) for x in seed_grid.flat))
                backgrounds.add(background)
                puzzle_backgrounds.add(background)
                puzzle_shapes.add(grid.shape)
                zero_as_seed += background != 0 and 0 in seed_grid
                for axis, level in enumerate(profile):
                    side_levels[axis].add(level)
                if sum(level < 2 for level in profile) == 1:
                    single_side[next(axis for axis, level in enumerate(profile) if level < 2)] += 1
                wide += width >= height + 3
                tall += height >= width + 3
                compact_five += grid.shape == (5, 5)
                if profile == _FULL:
                    centered_rows = {(height - 2) // 2, (height - 1) // 2}
                    centered_cols = {(width - 2) // 2, (width - 1) // 2}
                    offcenter_full += row not in centered_rows or col not in centered_cols
                if split == "train":
                    training_profiles.add(profile)
                    train_full += profile == _FULL
                    train_clipped += profile != _FULL
                else:
                    seen_test_profiles += profile in training_profiles
                    unseen_test_profiles += profile not in training_profiles
                    test_full += profile == _FULL
                    test_clipped += profile != _FULL
                compact_four_clipped += grid.shape == (4, 4) and profile == (1, 1, 1, 1)
        varying_background_puzzles += len(puzzle_backgrounds) > 1
        varying_shape_puzzles += len(puzzle_shapes) > 1
    _require(len(shapes) >= 35 and len(layouts) >= 120 and len(profiles) >= 20,
             "insufficient structural diversity")
    _require(all(levels == {0, 1, 2} for levels in side_levels),
             "some side lacks zero/one/two-cell visibility")
    _require(all(count >= 5 for count in single_side),
             "top/right/bottom/left-only clipping is missing")
    _require(train_full > 0 and train_clipped > 0 and test_full > 0 and test_clipped > 0,
             "train or test exposure types collapsed")
    _require(seen_test_profiles >= 5 and unseen_test_profiles >= 5,
             "tests lack both familiar and novel clipping profiles")
    _require(offcenter_full >= 10 and wide >= 10 and tall >= 10,
             "off-center full examples or rectangular canvases are scarce")
    _require(compact_four_clipped >= 1, "4x4 all-clipped cases missing")
    _require(compact_five >= 1, "5x5 cases missing")
    _require(varying_shape_puzzles == len(automatic),
             "some puzzle does not vary canvas shape")
    _require(len(backgrounds) >= 5 and varying_background_puzzles >= 3 and zero_as_seed >= 3,
             "background and zero-as-seed variation is insufficient")
    _require(len(palettes) >= 100, "corner-color roles are insufficiently varied")
