import numpy as np


_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))

_AUTO_COUNT_PAIRS = (
    (2, 1),
    (3, 1),
    (4, 2),
    (5, 3),
    (6, 4),
    (7, 1),
    (8, 2),
    (9, 3),
    (10, 4),
)


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.reshape(-1)),
    )


_SOURCE_PAIRS_TEXT = (
    (
        (
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000200",
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000000",
            "00800000600",
            "00000000000",
            "00000000000",
        ),
        (
            "00000000000",
            "00000000000",
            "00000000000",
            "00000000200",
            "00000000000",
            "00000020000",
            "00000500000",
            "00008060000",
            "00000000000",
            "00800000600",
            "00000000000",
            "00000000000",
        ),
    ),
    (
        (
            "000000000000",
            "000200000100",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000300000000",
            "000000000000",
            "000000000000",
        ),
        (
            "000000000000",
            "000200000100",
            "000000000000",
            "000002010000",
            "000000500000",
            "000003000000",
            "000000000000",
            "000300000000",
            "000000000000",
            "000000000000",
        ),
    ),
    (
        (
            "00000000000000",
            "00000000000000",
            "00030000000800",
            "00000000000000",
            "00000000000000",
            "00000000000000",
            "00000000000000",
            "00000000000000",
            "00000000000000",
            "00000000000000",
            "00090000000000",
            "00000000000000",
        ),
        (
            "00000000000000",
            "00000000000000",
            "00030000000800",
            "00000000000000",
            "00000000000000",
            "00000030800000",
            "00000005000000",
            "00000090000000",
            "00000000000000",
            "00000000000000",
            "00090000000000",
            "00000000000000",
        ),
    ),
)


def _text_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIR_SIGNATURES = frozenset(
    (
        _grid_signature(_text_grid(input_rows)),
        _grid_signature(_text_grid(output_rows)),
    )
    for input_rows, output_rows in _SOURCE_PAIRS_TEXT
)


def _checked_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count("num_train", num_train, 2, 10)
    test_count = _checked_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        return _AUTO_COUNT_PAIRS[seed % len(_AUTO_COUNT_PAIRS)]

    if train_count is None:
        assert test_count is not None
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    assert train_count is not None
    assert test_count is not None

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


_SIDES = tuple(range(5, 30, 2))
_SMALL_SIDES = (5, 7, 9, 11)


def _extra(rng, maximum):
    """Choose independently for each axis, including tight and roomy crops."""
    if maximum == 0:
        return 0
    band = int(rng.integers(0, 3))
    upper = (min(1, maximum), min(8, maximum), maximum)[band]
    return int(rng.integers(0, upper + 1))


def _boxes(rng, sides):
    if len(sides) == 1:
        side = sides[0]
        if side <= 7 and int(rng.integers(0, 4)) == 0:
            extra_height = int(rng.integers(10, 31 - side))
            extra_width = int(rng.integers(10, 31 - side))
        else:
            extra_height, extra_width = (_extra(rng, 30 - side) for _ in range(2))
        height, width = side + extra_height, side + extra_width
        top = int(rng.integers(0, height - side + 1))
        left = int(rng.integers(0, width - side + 1))
        return (height, width), [(top, left, side)]
    a, b = sides
    horizontal = bool(rng.integers(0, 2))
    gap = int(rng.integers(1, 5))
    if horizontal:
        needed_height, needed_width = max(a, b), a + gap + b
    else:
        needed_height, needed_width = a + gap + b, max(a, b)
    if max(needed_height, needed_width) > 30:
        raise ValueError("two squares do not fit")
    height = needed_height + _extra(rng, 30 - needed_height)
    width = needed_width + _extra(rng, 30 - needed_width)
    row = int(rng.integers(0, height - needed_height + 1))
    col = int(rng.integers(0, width - needed_width + 1))
    if horizontal:
        first = (row + int(rng.integers(0, needed_height - a + 1)), col, a)
        second = (row + int(rng.integers(0, needed_height - b + 1)),
                  col + a + gap, b)
    else:
        first = (row, col + int(rng.integers(0, needed_width - a + 1)), a)
        second = (row + a + gap,
                  col + int(rng.integers(0, needed_width - b + 1)), b)
    return (height, width), [first, second]


def _candidate_groups(coordinates):
    from itertools import combinations
    points = [tuple(map(int, point)) for point in coordinates]
    candidates = []
    for triple in combinations(points, 3):
        rows, cols = [p[0] for p in triple], [p[1] for p in triple]
        top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
        side = bottom - top + 1
        corners = ((top, left), (top, right), (bottom, left), (bottom, right))
        if side >= 5 and side % 2 == 1 and side == right - left + 1 \
                and set(triple).issubset(corners):
            candidates.append((top, left, side,
                               tuple(corners.index(p) for p in triple),
                               frozenset(triple)))
    return candidates


def _analyze_pair(before, after):
    assert before.shape == after.shape
    values, counts = np.unique(before, return_counts=True)
    background = int(values[np.argmax(counts)])
    coordinates = np.argwhere(before != background)
    assert len(coordinates) in (3, 6)
    groups = _candidate_groups(coordinates)
    assert len(groups) == len(coordinates) // 3, "ambiguous corner grouping"
    assert len(set().union(*(g[4] for g in groups))) == len(coordinates)
    expected = before.copy()
    center_colors = set()
    details = []
    for top, left, side, present, points in groups:
        center_row, center_col = top + side // 2, left + side // 2
        center_color = int(after[center_row, center_col])
        assert center_color != background and not np.any(before == center_color)
        center_colors.add(center_color)
        assert int(before[center_row, center_col]) == background
        for corner in present:
            row_bit, col_bit = _CORNERS[corner]
            color = int(before[top + row_bit * (side - 1),
                               left + col_bit * (side - 1)])
            miniature = (center_row + 2 * row_bit - 1,
                         center_col + 2 * col_bit - 1)
            assert int(before[miniature]) == background
            expected[miniature] = color
        expected[center_row, center_col] = center_color
        details.append({"side": side,
                        "missing": next(c for c in range(4) if c not in present),
                        "corners": frozenset(present),
                        "bounds": (top, left, side),
                        "marker_colors": tuple(int(before[p]) for p in points)})
    assert len(center_colors) == 1
    assert np.array_equal(expected, after)
    assert int(np.count_nonzero(before != after)) == 4 * len(groups)
    return {"background": background, "center_color": next(iter(center_colors)),
            "groups": details}


def _draw_pair(rng, sides, omissions, background, center_color, distinct_first):
    shape, boxes = _boxes(rng, sides)
    before = np.full(shape, background, dtype=np.int8)
    after = before.copy()
    available = [c for c in range(10) if c not in (background, center_color)]
    for index, (top, left, side) in enumerate(boxes):
        present = [c for c in range(4) if c != omissions[index]]
        if distinct_first and index == 0:
            colors = [int(c) for c in rng.choice(available, size=3, replace=False)]
        else:
            n_colors = int(rng.choice((1, 2, 3), p=(0.16, 0.39, 0.45)))
            palette = [int(c) for c in rng.choice(available, size=n_colors,
                                                 replace=False)]
            colors = palette[:]
            while len(colors) < 3:
                colors.append(int(rng.choice(palette)))
            rng.shuffle(colors)
        center_row, center_col = top + side // 2, left + side // 2
        for corner, color in zip(present, colors):
            row_bit, col_bit = _CORNERS[corner]
            outer = (top + row_bit * (side - 1), left + col_bit * (side - 1))
            miniature = (center_row + 2 * row_bit - 1,
                         center_col + 2 * col_bit - 1)
            before[outer] = color
            after[outer] = color
            after[miniature] = color
        after[center_row, center_col] = center_color
    return before, after


def _new_pair(rng, sides, omissions, background, center_color, distinct_first, used):
    for _ in range(160):
        before, after = _draw_pair(rng, sides, omissions, background,
                                   center_color, distinct_first)
        signature = _grid_signature(before)
        pair_signature = (signature, _grid_signature(after))
        if signature in used or pair_signature in _SOURCE_PAIR_SIGNATURES:
            continue
        try:
            _analyze_pair(before, after)
        except AssertionError:
            continue
        used.add(signature)
        return {"input": before, "output": after}
    raise ValueError("could not construct an unambiguous fresh pair")


def _choose_sides(rng, seed, train_count, test_count):
    regime = int(rng.integers(0, 4))  # overlap, larger, smaller, unrestricted
    if seed % 11 == 0:
        train = [5, 7]
        regime = 1
        pool = _SIDES
    else:
        pool = _SIDES
        if regime == 1:
            pool = tuple(s for s in _SIDES if s <= 21)
        elif regime == 2:
            pool = tuple(s for s in _SIDES if s >= 11)
        train = [int(rng.choice(pool))]
        train.append(int(rng.choice([s for s in pool if s != train[0]])))
    while len(train) < train_count:
        train.append(int(rng.choice(pool)))
    lower, upper = min(train), max(train)
    if regime == 1:
        choices = [s for s in _SIDES if s > upper]
    elif regime == 2:
        choices = [s for s in _SIDES if s < lower]
    elif regime == 0:
        choices = [s for s in _SIDES if lower <= s <= upper]
    else:
        choices = list(_SIDES)
    if not choices:
        choices = list(_SIDES)
    test = [int(rng.choice(choices)) for _ in range(test_count)]
    if seed % 11 == 0:
        test[0] = 9
    return train, test


def _choose_omissions(rng, train_count, test_count):
    mode = int(rng.integers(0, 4))
    if mode == 0:
        train = [int(rng.integers(0, 4))] * train_count
    else:
        first = int(rng.integers(0, 4))
        second = int(rng.choice([c for c in range(4) if c != first]))
        train = [first, second]
        while len(train) < train_count:
            train.append(int(rng.integers(0, 4)))
        rng.shuffle(train)
    demonstrated = set().union(*(set(range(4)) - {m} for m in train))
    eligible = [m for m in range(4) if (set(range(4)) - {m}).issubset(demonstrated)]
    if mode == 0:
        eligible = [train[0]]
    test = [int(rng.choice(eligible)) for _ in range(test_count)]
    return train, test


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    center_color = int(rng.integers(0, 10))
    backgrounds = [c for c in range(10) if c != center_color]
    first_backgrounds = [int(c) for c in rng.choice(backgrounds, size=2,
                                                    replace=False)]
    train_sides, test_sides = _choose_sides(rng, seed, train_count, test_count)
    train_missing, test_missing = _choose_omissions(rng, train_count, test_count)
    multi = train_count >= 3 and seed % 6 == 0
    used = set()
    train = []
    for index, side in enumerate(train_sides):
        sides, omissions = [side], [train_missing[index]]
        if multi and index == train_count - 1:
            sides = [int(rng.choice(_SMALL_SIDES)), int(rng.choice(_SMALL_SIDES))]
            omissions.append(int(rng.integers(0, 4)))
        background = (first_backgrounds[index] if index < 2
                      else int(rng.choice(backgrounds)))
        train.append(_new_pair(rng, sides, omissions, background, center_color,
                               index == 0, used))
    test = []
    for index, side in enumerate(test_sides):
        sides, omissions = [side], [test_missing[index]]
        if multi and bool(rng.integers(0, 2)):
            sides = [int(rng.choice(_SMALL_SIDES)), int(rng.choice(_SMALL_SIDES))]
            omissions.append(int(rng.choice(test_missing)))
        test.append(_new_pair(rng, sides, omissions, int(rng.choice(backgrounds)),
                              center_color, False, used))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used_inputs, used_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            before, after = pair["input"], pair["output"]
            for grid in (before, after):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            input_signature = _grid_signature(before)
            pair_signature = (input_signature, _grid_signature(after))
            assert input_signature not in used_inputs
            assert pair_signature not in used_pairs
            assert pair_signature not in _SOURCE_PAIR_SIGNATURES
            used_inputs.add(input_signature)
            used_pairs.add(pair_signature)
            analyses[split].append(_analyze_pair(before, after))
    centers = {a["center_color"] for split in analyses.values() for a in split}
    assert len(centers) == 1
    center = next(iter(centers))
    assert all(a["background"] != center for split in analyses.values() for a in split)
    train_positions = set().union(*(g["corners"] for a in analyses["train"] for g in a["groups"]))
    test_positions = set().union(*(g["corners"] for a in analyses["test"] for g in a["groups"]))
    assert test_positions.issubset(train_positions)
    if any(len(a["groups"]) == 2 for a in analyses["test"]):
        assert any(len(a["groups"]) == 2 for a in analyses["train"])
    assert len({a["background"] for a in analyses["train"]}) >= 2
    assert len({g["side"] for a in analyses["train"] for g in a["groups"]}) >= 2


def _puzzle_signature(puzzle):
    return tuple(tuple((_grid_signature(p["input"]), _grid_signature(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    from itertools import combinations
    for train, test in ((2, 1), (10, 1), (10, 4), (6, 3)):
        puzzle = generate_puzzle(900 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for arguments in ({"num_test": 1}, {"num_test": 4},
                      {"num_train": 2}, {"num_train": 10}):
        validate_generated_puzzle(generate_puzzle(707, **arguments))
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(123, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid count request: {arguments}")
    signatures, counts, sides, omissions, palettes = set(), set(), set(), set(), set()
    relations, marker_counts, margins = set(), set(), set()
    mixed_backgrounds = repeated_markers = multi_train = multi_test = 0
    tight = roomy_small = adjacent = novel_test = varied_test = 0
    three_orientations = one_orientation = 0
    source_like = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(seed))
        signatures.add(_puzzle_signature(puzzle))
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        info = {split: [_analyze_pair(p["input"], p["output"]) for p in puzzle[split]]
                for split in ("train", "test")}
        train_sizes = [g["side"] for a in info["train"] for g in a["groups"]]
        test_sizes = [g["side"] for a in info["test"] for g in a["groups"]]
        train_missing = {g["missing"] for a in info["train"] for g in a["groups"]}
        test_missing = {g["missing"] for a in info["test"] for g in a["groups"]}
        adjacent += any(a + b != 3 for a, b in combinations(train_missing, 2))
        novel_test += bool(test_missing - train_missing)
        varied_test += len(test_missing) > 1
        three_orientations += len(train_missing) >= 3
        one_orientation += len(train_missing | test_missing) == 1
        source_like += 7 in train_sizes and 9 in test_sizes
        relations.add("larger" if min(test_sizes) > max(train_sizes) else
                      "smaller" if max(test_sizes) < min(train_sizes) else "overlap")
        mixed_backgrounds += len({a["background"] for a in info["train"]}) > 1
        multi_train += any(len(a["groups"]) == 2 for a in info["train"])
        multi_test += any(len(a["groups"]) == 2 for a in info["test"])
        for split in ("train", "test"):
            for pair, a in zip(puzzle[split], info[split]):
                marker_counts.add(len(a["groups"]))
                palettes.add((a["background"], a["center_color"]))
                for g in a["groups"]:
                    sides.add(g["side"])
                    omissions.add(g["missing"])
                    repeated_markers += len(set(g["marker_colors"])) < 3
                    top, left, side = g["bounds"]
                    margins.add((top, left, pair["input"].shape[0] - top - side,
                                 pair["input"].shape[1] - left - side))
                    if len(a["groups"]) == 1:
                        extra_h = pair["input"].shape[0] - side
                        extra_w = pair["input"].shape[1] - side
                        tight += extra_h + extra_w <= 1
                        roomy_small += side <= 7 and extra_h >= 10 and extra_w >= 10
    assert len(signatures) == 42
    assert len({c[0] for c in counts}) >= 4 and len({c[1] for c in counts}) >= 3
    assert sides == set(_SIDES) and omissions == set(range(4))
    assert relations == {"larger", "smaller", "overlap"}
    assert mixed_backgrounds == 42 and repeated_markers > 0
    assert marker_counts == {1, 2} and multi_train > 0 and multi_test > 0
    assert len(palettes) >= 15 and len(margins) >= 50
    assert tight > 0 and roomy_small > 0 and source_like > 0
    assert min(adjacent, novel_test, varied_test, three_orientations, one_orientation) > 0
