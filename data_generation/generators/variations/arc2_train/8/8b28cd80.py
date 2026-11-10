"""Solved examples of the positional square-spiral ARC family."""

from itertools import combinations

import numpy as np


_POSITIONS = tuple((r, c) for r in range(3) for c in range(3))
_SIZES = (9, 13, 17)
_SOURCE_SPECS = (
    ((1, 0), 5), ((1, 2), 8), ((0, 2), 7), ((0, 1), 3),
    ((1, 1), 4), ((2, 0), 3), ((2, 2), 6),
)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _normalize_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _normalize_count("num_train", num_train, 2, 10)
    test = _normalize_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + (seed // 5 + test) % (10 - test)
    elif test is None:
        test = 1 + (seed // 7 + train) % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _transform(grid: np.ndarray, orientation: int) -> np.ndarray:
    turned = np.rot90(grid, orientation % 4)
    return np.fliplr(turned) if orientation >= 4 else turned


def _base_mask(position: tuple[int, int], size: int) -> np.ndarray:
    spacing = (size - 1) // 2
    anchor_row, anchor_col = spacing * position[0], spacing * position[1]
    rows, cols = np.indices((size, size))
    distance = np.maximum(abs(rows - anchor_row), abs(cols - anchor_col))
    mask = distance % 2 == 0
    for k in range(1, size + 1):
        row, col = anchor_row - k, anchor_col - k + 1
        if 0 <= row < size and 0 <= col < size:
            mask[row, col] = not bool(mask[row, col])
    return mask


def _mask(position: tuple[int, int], size: int, orientation: int) -> np.ndarray:
    # Find the canonical input coordinate whose transformed marker is at position.
    for canonical in _POSITIONS:
        probe = np.zeros((3, 3), dtype=np.int8)
        probe[canonical] = 1
        if bool(_transform(probe, orientation)[position]):
            return _transform(_base_mask(canonical, size), orientation)
    raise AssertionError("invalid position transform")


def _output(position: tuple[int, int], background: int, marker: int,
            size: int, orientation: int) -> np.ndarray:
    grid = np.full((size, size), background, dtype=np.int8)
    grid[_mask(position, size, orientation)] = marker
    return grid


def _pair(position: tuple[int, int], background: int, marker: int,
          size: int, orientation: int) -> dict:
    grid = np.full((3, 3), background, dtype=np.int8)
    grid[position] = marker
    return {"input": grid, "output": _output(position, background, marker, size, orientation)}


def _has_axis_contrasts(positions: list[tuple[int, int]] | tuple) -> bool:
    pairs = list(combinations(set(positions), 2))
    return (
        any(a[0] == b[0] and a[1] != b[1] for a, b in pairs)
        and any(a[1] == b[1] and a[0] != b[0] for a, b in pairs)
    )


def _permutation(rng: np.random.Generator, values: list) -> list:
    order = rng.permutation(len(values))
    return [values[int(index)] for index in order]


def _position_plan(rng: np.random.Generator, train_count: int,
                   test_count: int) -> tuple[list, list]:
    if train_count == 2:
        boundary = [p for p in _POSITIONS if p != (1, 1)]
        train = [(1, 1), boundary[int(rng.integers(len(boundary)))]]
        train = _permutation(rng, train)
        return train, [train[int(rng.integers(2))]]

    modes = ["seen", "source_row", "row", "column", "feature"]
    if train_count > 6:
        modes = [mode for mode in modes if mode not in ("source_row", "row", "column")]
    if train_count > 8:
        modes = ["seen"]
    mode = modes[int(rng.integers(len(modes)))]
    target = None
    if mode == "source_row":
        withheld = [p for p in _POSITIONS if p[0] == 2]
    elif mode == "row":
        row = int(rng.integers(3))
        withheld = [p for p in _POSITIONS if p[0] == row]
    elif mode == "column":
        col = int(rng.integers(3))
        withheld = [p for p in _POSITIONS if p[1] == col]
    elif mode == "feature":
        target = _POSITIONS[int(rng.integers(9))]
        withheld = [target]
    else:
        withheld = []

    pool = [p for p in _POSITIONS if p not in withheld]
    triples = [triple for triple in combinations(pool, 3) if _has_axis_contrasts(triple)]
    if mode != "feature" or target != (1, 1):
        centered = [triple for triple in triples if (1, 1) in triple]
        if centered:
            triples = centered
    # A three-position L demonstrates independent row and column movement.
    initial = list(triples[int(rng.integers(len(triples)))])
    remaining = _permutation(rng, [p for p in pool if p not in initial])
    train = initial + remaining[:train_count - 3]
    if len(train) < train_count:
        train.extend(_permutation(rng, list(train))[:train_count - len(train)])
    train = _permutation(rng, train)

    if mode == "seen":
        tests = _permutation(rng, list(dict.fromkeys(train)))[:test_count]
    elif mode == "feature":
        tests = [target]
        others = _permutation(rng, [p for p in _POSITIONS if p != target])
        tests.extend(others[:test_count - 1])
    else:
        tests = _permutation(rng, withheld)[:test_count]
        if len(tests) < test_count:
            other = _permutation(rng, [p for p in _POSITIONS if p not in tests])
            tests.extend(other[:test_count - len(tests)])
    return train, _permutation(rng, tests)


def _input_roles(grid: np.ndarray) -> tuple[int, int, tuple[int, int]]:
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2 and sorted(int(n) for n in counts) == [1, 8]
    marker = int(colors[int(np.argmin(counts))])
    background = int(colors[int(np.argmax(counts))])
    row, col = np.argwhere(grid == marker)[0]
    return background, marker, (int(row), int(col))


def _is_source_copy(pair: dict) -> bool:
    grid, output = pair["input"], pair["output"]
    if output.shape != (9, 9):
        return False
    for position, marker in _SOURCE_SPECS:
        source = np.zeros((3, 3), dtype=np.int8)
        source[position] = marker
        if np.array_equal(grid, source):
            return np.array_equal(output, _output(position, 0, marker, 9, 0))
    return False


def _choose_color_pair(rng: np.random.Generator, position: tuple[int, int],
                       size: int, orientation: int, used: set,
                       backgrounds: list[int], markers: list[int]) -> tuple[int, int]:
    choices = []
    for background in backgrounds:
        for marker in markers:
            if background == marker or (position, background, marker) in used:
                continue
            pair = _pair(position, background, marker, size, orientation)
            if not _is_source_copy(pair):
                choices.append((background, marker))
    assert choices, "no admissible color assignment"
    return choices[int(rng.integers(len(choices)))]


def _color_plan(rng: np.random.Generator, train_positions: list,
                test_positions: list, size: int, orientation: int) -> tuple[list, list]:
    mode = ("constant", "mixed", "swap")[int(rng.integers(3))]
    all_colors = list(range(10))
    used: set = set()
    train_colors = []
    test_colors = []
    if mode == "constant":
        fixed = int(rng.integers(10))
        backgrounds = [fixed] * (len(train_positions) + len(test_positions))
    elif mode == "swap":
        first = 0 if int(rng.integers(2)) == 0 else int(rng.integers(10))
        second_options = [
            color for color in all_colors
            if color != first
            and not _is_source_copy(
                _pair(train_positions[0], first, color, size, orientation)
            )
            and not _is_source_copy(
                _pair(train_positions[1], color, first, size, orientation)
            )
        ]
        second = second_options[int(rng.integers(len(second_options)))]
        backgrounds = [first, second]
        backgrounds.extend(
            int(rng.integers(10))
            for _ in range(len(train_positions) + len(test_positions) - 2)
        )
    else:
        first = 0 if int(rng.integers(2)) == 0 else int(rng.integers(10))
        second = [c for c in all_colors if c != first][int(rng.integers(9))]
        backgrounds = [first, second]
        backgrounds.extend(
            int(rng.integers(10))
            for _ in range(len(train_positions) + len(test_positions) - 2)
        )

    for index, position in enumerate(train_positions):
        background = backgrounds[index]
        if mode == "swap" and index < 2:
            desired_marker = backgrounds[1 - index]
            candidates = [desired_marker]
        elif index == 1 and mode == "constant":
            candidates = [c for c in all_colors if c != train_colors[0][1]]
        else:
            candidates = all_colors
        background, marker = _choose_color_pair(
            rng, position, size, orientation, used, [background], candidates
        )
        used.add((position, background, marker))
        train_colors.append((background, marker))

    seen = {marker for _, marker in train_colors}
    for offset, position in enumerate(test_positions):
        background = backgrounds[len(train_positions) + offset]
        if len(test_positions) >= 2 and offset == 0:
            candidates = sorted(seen)
        elif len(test_positions) >= 2 and offset == 1:
            candidates = [c for c in all_colors if c not in seen] or all_colors
        else:
            candidates = all_colors
        try:
            background, marker = _choose_color_pair(
                rng, position, size, orientation, used, [background], candidates
            )
        except AssertionError:
            # A requested new/seen marker can be exhausted by the fixed
            # background, prior inputs, or an exact source-pair exclusion.
            background, marker = _choose_color_pair(
                rng, position, size, orientation, used, [background], all_colors
            )
        used.add((position, background, marker))
        test_colors.append((background, marker))
    return train_colors, test_colors


def _matching_orientations(train: list) -> list[int]:
    size = train[0]["output"].shape[0]
    matching = []
    for orientation in range(8):
        if all(np.array_equal(
            pair["output"],
            _output(_input_roles(pair["input"])[2],
                    _input_roles(pair["input"])[0],
                    _input_roles(pair["input"])[1], size, orientation)
        ) for pair in train):
            matching.append(orientation)
    return matching


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    size = (9, 9, 9, 9, 13, 13, 17)[int(rng.integers(7))]
    orientation = int(rng.integers(8))
    train_positions, test_positions = _position_plan(rng, train_count, test_count)
    train_colors, test_colors = _color_plan(
        rng, train_positions, test_positions, size, orientation
    )
    puzzle = {
        "train": [
            _pair(position, *colors, size, orientation)
            for position, colors in zip(train_positions, train_colors)
        ],
        "test": [
            _pair(position, *colors, size, orientation)
            for position, colors in zip(test_positions, test_colors)
        ],
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    input_keys = set()
    pair_keys = set()
    size = None
    positions = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == (3, 3)
            assert out.shape[0] == out.shape[1] and out.shape[0] in _SIZES
            if size is None:
                size = out.shape[0]
            assert out.shape == (size, size)
            background, marker, position = _input_roles(inp)
            positions[split].append(position)
            assert set(int(x) for x in np.unique(out)) == {background, marker}
            assert int(out[((size - 1) // 2 * position[0],
                            (size - 1) // 2 * position[1])]) == marker
            assert not _is_source_copy(pair)
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in input_keys and pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)
    assert (1, 1) in positions["train"] or _has_axis_contrasts(positions["train"])
    if len(train) >= 3:
        assert _has_axis_contrasts(positions["train"])
    else:
        assert set(positions["test"]).issubset(set(positions["train"]))
    orientations = _matching_orientations(train)
    assert len(orientations) == 1, (
        "training must identify one consistent spiral convention"
    )
    # Predict from training pairs and the test input alone. Every convention
    # compatible with training must give the same answer on every test.
    for pair in test:
        background, marker, position = _input_roles(pair["input"])
        predictions = [
            _output(position, background, marker, size, orientation)
            for orientation in orientations
        ]
        assert all(np.array_equal(predictions[0], item) for item in predictions)
        assert np.array_equal(pair["output"], predictions[0])
    for pair in train:
        background, marker, position = _input_roles(pair["input"])
        assert np.array_equal(
            pair["output"], _output(position, background, marker, size, orientations[0])
        )


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
              for pair in puzzle[split])
        for split in ("train", "test")
    )


def self_test() -> None:
    # Exact source grids independently copied from source_task.txt.
    source_outputs = (
        ("555550505", "000050505", "555050505", "505050505", "505050505",
         "005050505", "555050505", "000050505", "555550505"),
        ("808080888", "808080800", "808080808", "808080808", "808080808",
         "808080800", "808080888", "808080000", "808088888"),
        ("707070707", "707070700", "707070777", "707070000", "707077777",
         "707000000", "707777777", "700000000", "777777777"),
        ("303030303", "303000303", "303333303", "300000003", "333333333",
         "000000000", "333333333", "000000000", "333333333"),
        ("404444444", "404000004", "404044404", "404040404", "404040404",
         "404000404", "404444404", "400000004", "444444444"),
        ("333333333", "000000003", "333333303", "000000303", "333330303",
         "000030303", "333030303", "303030303", "303030303"),
        ("606666666", "606000000", "606066666", "606060000", "606060666",
         "606060600", "606060606", "606060606", "606060606"),
    )
    for (position, marker), rows in zip(_SOURCE_SPECS, source_outputs):
        expected = np.array([[int(char) for char in row] for row in rows], dtype=np.int8)
        assert np.array_equal(_output(position, 0, marker, 9, 0), expected)

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    all_positions = set()
    masks = set()
    sizes = set()
    orientations = set()
    zero_backgrounds = 0
    variable_backgrounds = 0
    role_swaps = 0
    held_out = 0
    seen = 0
    bottom_middle = 0
    heldout_center = 0
    heldout_corner = 0
    heldout_edge = 0
    bottom_row_shift = 0
    column_shift = 0
    seen_marker = 0
    new_marker = 0
    constant_backgrounds = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        size = puzzle["train"][0]["output"].shape[0]
        sizes.add(size)
        orientations.update(_matching_orientations(puzzle["train"]))
        train_roles = [_input_roles(pair["input"]) for pair in puzzle["train"]]
        test_roles = [_input_roles(pair["input"]) for pair in puzzle["test"]]
        roles = train_roles + test_roles
        all_positions.update(role[2] for role in roles)
        backgrounds = {role[0] for role in roles}
        zero_backgrounds += 0 in backgrounds
        variable_backgrounds += len(backgrounds) > 1
        constant_backgrounds += len(backgrounds) == 1
        role_swaps += any(a[0] == b[1] and a[1] == b[0]
                          for a, b in combinations(train_roles, 2))
        train_positions = {role[2] for role in train_roles}
        held_out += any(role[2] not in train_positions for role in test_roles)
        seen += any(role[2] in train_positions for role in test_roles)
        bottom_middle += (2, 1) in {role[2] for role in roles}
        withheld = {role[2] for role in test_roles} - train_positions
        heldout_center += (1, 1) in withheld
        heldout_corner += any(p in ((0, 0), (0, 2), (2, 0), (2, 2))
                              for p in withheld)
        heldout_edge += any(p in ((0, 1), (1, 0), (1, 2), (2, 1))
                            for p in withheld)
        bottom_row_shift += bool(withheld) and all(p[0] == 2 for p in withheld) and all(
            p[0] != 2 for p in train_positions
        )
        column_shift += bool(withheld) and len({p[1] for p in withheld}) == 1 and all(
            p[1] != next(iter(withheld))[1] for p in train_positions
        )
        markers = {role[1] for role in train_roles}
        seen_marker += any(role[1] in markers for role in test_roles)
        new_marker += any(role[1] not in markers for role in test_roles)
        assert len({role[2] for role in test_roles}) == len(test_roles)
        if len(train_roles) < 10:
            assert len(train_positions) == len(train_roles)
        for pair, (_, marker, _) in zip(puzzle["train"] + puzzle["test"], roles):
            masks.add((size, tuple(int(x == marker) for x in pair["output"].flat)))
    assert all_positions == set(_POSITIONS)
    assert sizes == set(_SIZES)
    assert len(orientations) == 8
    assert len(masks) >= 50
    assert zero_backgrounds >= 5 and variable_backgrounds >= 8
    assert constant_backgrounds >= 5 and role_swaps >= 2
    assert held_out >= 5 and seen >= 5 and bottom_middle >= 5
    assert heldout_center and heldout_corner and heldout_edge
    assert bottom_row_shift and column_shift
    assert seen_marker >= 5 and new_marker >= 5

    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(314159 + train_count, train_count, test_count)
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(6001, train_count, test_count))
    assert _puzzle_key(generate_puzzle(271828, 8, 4)) == _puzzle_key(
        generate_puzzle(271828, 8, 4)
    )
    for request in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": 2.0, "num_test": 1},
    ):
        try:
            generate_puzzle(7000, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count accepted: {request}")
