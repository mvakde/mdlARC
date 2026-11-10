import numpy as np


_CORNERS = ("UL", "UR", "LL", "LR")

_SOURCE_PAIR_SIGNATURES = {
    (
        (
            "888888888846",
            "888888888812",
            "000020000088",
            "020000002088",
            "000000000088",
            "200000220088",
            "000000002088",
            "000200000088",
            "000000000088",
            "020000020088",
            "020000000088",
            "000020000288",
        ),
        (
            "888888888846",
            "888888888812",
            "000040000088",
            "040000006088",
            "000000000088",
            "400000660088",
            "000000006088",
            "000100000088",
            "000000000088",
            "010000020088",
            "010000000088",
            "000010000288",
        ),
    ),
    (
        (
            "1488888888",
            "3288888888",
            "8800001000",
            "8801000000",
            "8800000101",
            "8800000000",
            "8800100100",
            "8800000100",
            "8801001000",
            "8800000001",
        ),
        (
            "1488888888",
            "3288888888",
            "8800004000",
            "8801000000",
            "8800000404",
            "8800000000",
            "8800300200",
            "8800000200",
            "8803002000",
            "8800000002",
        ),
    ),
    (
        (
            "01000000000088",
            "00010001010088",
            "10010000000088",
            "00000000001088",
            "00000000100088",
            "00001000000088",
            "01000000000088",
            "10000100100088",
            "00000000000088",
            "00001000001088",
            "00000001100088",
            "01000000000088",
            "88888888888831",
            "88888888888874",
        ),
        (
            "03000000000088",
            "00030001010088",
            "30030000000088",
            "00000000001088",
            "00000000100088",
            "00003000000088",
            "07000000000088",
            "70000700400088",
            "00000000000088",
            "00007000004088",
            "00000004400088",
            "07000000000088",
            "88888888888831",
            "88888888888874",
        ),
    ),
}


def _rows_signature(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _grid_signature(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _layout_slices(
    corner: str,
    height: int,
    width: int,
) -> tuple:
    if corner == "UL":
        field = (slice(2, height), slice(2, width))
        key = (slice(0, 2), slice(0, 2))
        horizontal = (slice(0, 2), slice(2, width))
        vertical = (slice(2, height), slice(0, 2))
    elif corner == "UR":
        field = (slice(2, height), slice(0, width - 2))
        key = (slice(0, 2), slice(width - 2, width))
        horizontal = (slice(0, 2), slice(0, width - 2))
        vertical = (slice(2, height), slice(width - 2, width))
    elif corner == "LL":
        field = (slice(0, height - 2), slice(2, width))
        key = (slice(height - 2, height), slice(0, 2))
        horizontal = (slice(height - 2, height), slice(2, width))
        vertical = (slice(0, height - 2), slice(0, 2))
    elif corner == "LR":
        field = (slice(0, height - 2), slice(0, width - 2))
        key = (
            slice(height - 2, height),
            slice(width - 2, width),
        )
        horizontal = (
            slice(height - 2, height),
            slice(0, width - 2),
        )
        vertical = (
            slice(0, height - 2),
            slice(width - 2, width),
        )
    else:
        raise ValueError("unknown corner")

    return field, key, horizontal, vertical


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    assert input_grid.ndim == 2
    assert output_grid.ndim == 2
    assert input_grid.shape == output_grid.shape

    height, width = input_grid.shape
    assert height == width

    side = height - 2
    assert 4 <= side <= 28
    assert side % 2 == 0

    candidates = []

    for corner in _CORNERS:
        (
            field_slice,
            key_slice,
            horizontal_slice,
            vertical_slice,
        ) = _layout_slices(corner, height, width)

        horizontal = input_grid[horizontal_slice]
        vertical = input_grid[vertical_slice]
        separator = int(horizontal.flat[0])

        if not np.all(horizontal == separator):
            continue
        if not np.all(vertical == separator):
            continue

        key = input_grid[key_slice]
        key_values = tuple(int(value) for value in key.ravel())
        key_set = set(key_values)

        if len(key_set) != 4:
            continue
        if separator in key_set:
            continue

        field = input_grid[field_slice]
        field_set = set(
            int(value) for value in np.unique(field)
        )

        marker_candidates = field_set & key_set
        background_candidates = field_set - key_set

        if len(field_set) != 2:
            continue
        if len(marker_candidates) != 1:
            continue
        if len(background_candidates) != 1:
            continue

        marker = next(iter(marker_candidates))
        background = next(iter(background_candidates))

        if background == separator:
            continue
        if background in key_set:
            continue
        if len(set(int(v) for v in np.unique(input_grid))) != 6:
            continue

        candidates.append(
            {
                "corner": corner,
                "side": side,
                "field_slice": field_slice,
                "key_slice": key_slice,
                "key": key.copy(),
                "key_values": key_values,
                "key_set": frozenset(key_set),
                "marker": marker,
                "background": background,
                "separator": separator,
            }
        )

    assert len(candidates) == 1, (
        "input must have exactly one valid field/key layout"
    )

    info = candidates[0]
    field = input_grid[info["field_slice"]]

    expected = input_grid.copy()
    expected_field = expected[info["field_slice"]]

    half = side // 2
    quadrant_counts = []

    for quadrant_row in range(2):
        row_slice = slice(
            quadrant_row * half,
            (quadrant_row + 1) * half,
        )

        for quadrant_col in range(2):
            col_slice = slice(
                quadrant_col * half,
                (quadrant_col + 1) * half,
            )

            source_block = field[row_slice, col_slice]
            marker_mask = source_block == info["marker"]
            quadrant_counts.append(
                int(np.count_nonzero(marker_mask))
            )

            target_block = expected_field[row_slice, col_slice]
            target_block[marker_mask] = info["key"][
                quadrant_row,
                quadrant_col,
            ]

    assert np.array_equal(output_grid, expected), (
        "output does not follow quadrant recoloring"
    )
    assert len(set(int(v) for v in np.unique(output_grid))) == 6

    marker_positions = np.argwhere(
        info["key"] == info["marker"]
    )
    assert marker_positions.shape == (1, 2)

    info["marker_position"] = int(
        marker_positions[0, 0] * 2
        + marker_positions[0, 1]
    )
    info["quadrant_counts"] = tuple(quadrant_counts)
    info["marker_count"] = sum(quadrant_counts)

    return info


def _validate_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer or None")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)

    if not low <= value <= high:
        raise ValueError(
            f"{name} must be between {low} and {high}"
        )

    return value


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _validate_count(
        "num_train",
        num_train,
        2,
        10,
    )
    num_test = _validate_count(
        "num_test",
        num_test,
        1,
        4,
    )

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(
            rng.integers(num_test + 1, 11)
        )
    elif num_train is None:
        num_train = int(
            rng.integers(num_test + 1, 11)
        )
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = int(
            rng.integers(1, maximum_test + 1)
        )

    if num_train <= num_test:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return num_train, num_test


def _new_roles(
    rng: np.random.Generator,
    base: tuple[int, int],
    colors: tuple[int, ...],
) -> tuple[int, int]:
    choices = [
        (a, b) for a in colors for b in colors
        if a != b and (a, b) != base
    ]
    return choices[int(rng.integers(len(choices)))]


def _new_key(
    rng: np.random.Generator,
    background: int,
    separator: int,
    forbidden: int | None = None,
    required: int | None = None,
    different_from: frozenset[int] | None = None,
) -> np.ndarray:
    available = [
        color for color in range(10)
        if color not in (background, separator, forbidden)
    ]
    if required is not None:
        assert required in available
        available.remove(required)
        selected = [required]
        selected.extend(int(v) for v in rng.choice(available, 3, replace=False))
    else:
        selected = [int(v) for v in rng.choice(available, 4, replace=False)]

    if different_from is not None and frozenset(selected) == different_from:
        for replacement in available:
            if replacement not in selected:
                selected[0] = replacement
                break

    return np.asarray(selected, dtype=np.int8)[rng.permutation(4)].reshape(2, 2)


def _choose_marker_mask(
    rng: np.random.Generator,
    side: int,
    marker_position: int,
    full_coverage: bool,
    crossing: bool,
    style: str,
) -> np.ndarray:
    half = side // 2
    mask = np.zeros((side, side), dtype=bool)
    crossing_cells: list[tuple[int, int]] = []

    if crossing:
        if bool(rng.integers(2)):
            column = int(rng.integers(side))
            crossing_cells = [(half - 1, column), (half, column)]
        else:
            row = int(rng.integers(side))
            crossing_cells = [(row, half - 1), (row, half)]

    def quadrant(row: int, column: int) -> int:
        return 2 * (row >= half) + (column >= half)

    selected = {quadrant(row, col) for row, col in crossing_cells}
    if full_coverage:
        selected = set(range(4))
    else:
        wanted = int(rng.choice((1, 2, 3, 4), p=(0.20, 0.27, 0.25, 0.28)))
        wanted = max(wanted, len(selected))
        remaining = [q for q in range(4) if q not in selected]
        rng.shuffle(remaining)
        selected.update(remaining[:wanted - len(selected)])
        if selected == {marker_position}:
            selected = {int(rng.choice([q for q in range(4) if q != marker_position]))}

    allowed = np.zeros_like(mask)
    for q in selected:
        allowed[(q // 2) * half:(q // 2 + 1) * half,
                (q % 2) * half:(q % 2 + 1) * half] = True

    density_band = int(rng.choice(3, p=(0.32, 0.43, 0.25)))
    low, high = ((0.025, 0.075), (0.08, 0.17), (0.18, 0.30))[density_band]
    target = max(4, int(round(side * side * rng.uniform(low, high))))
    occupied_capacity = int(np.count_nonzero(allowed))
    target = min(target, max(4, int(occupied_capacity * 0.65)), side * side - 1)

    for row, column in crossing_cells:
        mask[row, column] = True
    for q in selected:
        region = mask[(q // 2) * half:(q // 2 + 1) * half,
                      (q % 2) * half:(q % 2 + 1) * half]
        if not np.any(region):
            row = (q // 2) * half + int(rng.integers(half))
            col = (q % 2) * half + int(rng.integers(half))
            mask[row, col] = True

    q_list = sorted(selected)
    alpha = 0.3 if bool(rng.integers(2)) else 1.4
    q_weights = rng.dirichlet(np.full(len(q_list), alpha))
    weight_by_q = np.zeros(4)
    for q, weight in zip(q_list, q_weights):
        weight_by_q[q] = weight

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    direction = directions[int(rng.integers(4))]
    head = tuple(int(v) for v in np.argwhere(mask)[0])

    while int(np.count_nonzero(mask)) < target:
        candidate = None
        if style in ("cluster", "stroke", "mixed"):
            chance = 0.64 if style != "mixed" else 0.43
            if rng.random() < chance:
                if style == "stroke":
                    anchors = [head]
                    offsets = [direction]
                else:
                    points = np.argwhere(mask)
                    anchors = [tuple(int(v) for v in points[int(rng.integers(len(points)))])]
                    offsets = list(directions)
                    rng.shuffle(offsets)
                for anchor in anchors:
                    for dr, dc in offsets:
                        row, col = anchor[0] + dr, anchor[1] + dc
                        if (0 <= row < side and 0 <= col < side
                                and allowed[row, col] and not mask[row, col]):
                            candidate = (row, col)
                            break
                    if candidate is not None:
                        break

        if candidate is None:
            options = np.argwhere(allowed & ~mask)
            if style == "isolated":
                isolated = []
                for row, col in options:
                    if all(
                        not (0 <= row + dr < side and 0 <= col + dc < side
                             and mask[row + dr, col + dc])
                        for dr, dc in directions
                    ):
                        isolated.append((int(row), int(col)))
                if isolated:
                    options = np.asarray(isolated)
            weights = np.asarray([
                weight_by_q[quadrant(int(row), int(col))]
                for row, col in options
            ])
            weights /= weights.sum()
            row, col = options[int(rng.choice(len(options), p=weights))]
            candidate = (int(row), int(col))

        mask[candidate] = True
        head = candidate
        if style == "stroke" and rng.random() < 0.18:
            direction = directions[int(rng.integers(4))]

    return mask


def _generate_pair(
    rng: np.random.Generator,
    corner: str,
    side: int,
    background: int,
    separator: int,
    key: np.ndarray,
    marker_position: int,
    full_coverage: bool,
    crossing: bool,
    style: str,
) -> dict:
    size = side + 2
    field_slice, key_slice, _, _ = _layout_slices(corner, size, size)
    input_grid = np.full((size, size), separator, dtype=np.int8)
    input_grid[field_slice] = background
    input_grid[key_slice] = key

    mask = _choose_marker_mask(
        rng, side, marker_position, full_coverage, crossing, style
    )
    marker = int(key.ravel()[marker_position])
    input_grid[field_slice][mask] = marker
    output_grid = input_grid.copy()
    output_field = output_grid[field_slice]
    half = side // 2
    for q in range(4):
        rows = slice((q // 2) * half, (q // 2 + 1) * half)
        cols = slice((q % 2) * half, (q % 2 + 1) * half)
        output_field[rows, cols][mask[rows, cols]] = key.ravel()[q]
    return {"input": input_grid, "output": output_grid}


def _is_source_pair(pair: dict) -> bool:
    return (
        _rows_signature(pair["input"]),
        _rows_signature(pair["output"]),
    ) in _SOURCE_PAIR_SIGNATURES


def _test_side(
    rng: np.random.Generator,
    train_sides: list[int],
) -> int:
    relation = int(rng.integers(4))
    if relation == 0:
        options = list(range(max(train_sides) + 2, 29, 2))
    elif relation == 1:
        options = list(set(train_sides))
    elif relation == 2:
        options = list(range(4, min(train_sides), 2))
    else:
        options = list(range(4, 29, 2))
    if not options:
        options = list(range(4, 29, 2))
    return int(rng.choice(options))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)

    novel_color = int(rng.integers(10)) if rng.random() < 0.46 else None
    role_colors = tuple(c for c in range(10) if c != novel_color)
    base_roles = tuple(int(v) for v in rng.choice(role_colors, 2, replace=False))
    role_variation = rng.random() < 0.53

    source_like = rng.random() < 0.28
    if source_like:
        first_corners = ["UR", "UL"]
    else:
        first_corners = [str(v) for v in rng.choice(_CORNERS, 2, replace=False)]

    marker_slots = [int(v) for v in rng.choice(4, 3, replace=False)]
    if num_train == 2 or rng.random() < 0.55:
        marker_slots = marker_slots[:2]
    train_corners = first_corners + [str(rng.choice(_CORNERS)) for _ in range(num_train - 2)]
    train_positions = marker_slots[:2] + [int(rng.choice(marker_slots)) for _ in range(num_train - 2)]

    size_choices = list(range(6, 25, 2))
    first_sides = [int(v) for v in rng.choice(size_choices, 2, replace=False)]
    train_sides = first_sides + [int(rng.choice(size_choices)) for _ in range(num_train - 2)]

    train = []
    seen_inputs = set()
    first_palette = None
    for index in range(num_train):
        if index == 0 or not role_variation:
            background, separator = base_roles
        elif index == 1 or rng.random() < 0.48:
            background, separator = _new_roles(rng, base_roles, role_colors)
        else:
            background, separator = base_roles

        key = _new_key(
            rng, background, separator, forbidden=novel_color,
            different_from=first_palette if index == 1 else None,
        )
        if index == 0:
            first_palette = frozenset(int(v) for v in key.ravel())
        style = str(rng.choice(("scatter", "isolated", "cluster", "stroke", "mixed")))
        for _ in range(20):
            pair = _generate_pair(
                rng, train_corners[index], train_sides[index],
                background, separator, key, train_positions[index],
                index == 0, index == 0 or rng.random() < 0.34, style,
            )
            signature = _grid_signature(pair["input"])
            if signature not in seen_inputs and not _is_source_pair(pair):
                break
        else:
            raise RuntimeError("could not make a distinct training pair")
        seen_inputs.add(signature)
        train.append(pair)

    heldout_corners = [c for c in _CORNERS if c not in train_corners]
    unseen_slots = [p for p in range(4) if p not in train_positions]
    test = []
    for index in range(num_test):
        if index == 0 and source_like:
            corner = "LR"
        elif heldout_corners and rng.random() < 0.55:
            corner = str(rng.choice(heldout_corners))
        else:
            corner = str(rng.choice(train_corners))

        if unseen_slots and rng.random() < 0.53:
            marker_position = int(rng.choice(unseen_slots))
        else:
            marker_position = int(rng.choice(train_positions))

        if role_variation and rng.random() < 0.55:
            background, separator = _new_roles(rng, base_roles, role_colors)
        else:
            background, separator = base_roles
        key = _new_key(
            rng, background, separator,
            required=novel_color if index == 0 else None,
        )
        side = _test_side(rng, train_sides)
        style = str(rng.choice(("scatter", "isolated", "cluster", "stroke", "mixed")))
        for _ in range(20):
            pair = _generate_pair(
                rng, corner, side, background, separator, key,
                marker_position, False, rng.random() < 0.39, style,
            )
            signature = _grid_signature(pair["input"])
            if signature not in seen_inputs and not _is_source_pair(pair):
                break
        else:
            raise RuntimeError("could not make a distinct test pair")
        seen_inputs.add(signature)
        test.append(pair)

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _mask_features(input_grid: np.ndarray, info: dict) -> dict:
    mask = input_grid[info["field_slice"]] == info["marker"]
    side = info["side"]
    half = side // 2
    horizontal_cross = bool(np.any(mask[half - 1] & mask[half]))
    vertical_cross = bool(np.any(mask[:, half - 1] & mask[:, half]))
    adjacent = 0
    isolated = 0
    for row, col in np.argwhere(mask):
        neighbors = sum(
            0 <= row + dr < side and 0 <= col + dc < side
            and mask[row + dr, col + dc]
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
        )
        if neighbors == 0:
            isolated += 1
        adjacent += neighbors
    return {
        "crossing": horizontal_cross or vertical_cross,
        "adjacent_edges": adjacent // 2,
        "isolated": isolated,
        "occupied_quadrants": sum(v > 0 for v in info["quadrant_counts"]),
        "dominance": max(info["quadrant_counts"]) / info["marker_count"],
        "density": info["marker_count"] / (side * side),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].dtype == pair["output"].dtype
            input_signature = _grid_signature(pair["input"])
            pair_signature = (input_signature, _grid_signature(pair["output"]))
            assert input_signature not in seen_inputs
            assert pair_signature not in seen_pairs
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            assert not _is_source_pair(pair)
            info = _analyze_pair(pair["input"], pair["output"])
            assert not np.array_equal(pair["input"], pair["output"])
            analyses[split].append(info)

    train_info = analyses["train"]
    assert len({i["corner"] for i in train_info}) >= 2
    assert len({i["side"] for i in train_info}) >= 2
    assert len({i["marker_position"] for i in train_info}) >= 2
    assert len({i["key_set"] for i in train_info}) >= 2
    assert any(all(v > 0 for v in i["quadrant_counts"]) for i in train_info)
    assert any(
        _mask_features(pair["input"], info)["crossing"]
        for pair, info in zip(puzzle["train"], train_info)
    )
    # Every test color and marker are shown in its own key and field;
    # the demonstrated global quadrant rule has no hidden color map.
    for info in analyses["test"]:
        assert info["marker"] in info["key_set"]


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(
        len(first[split]) == len(second[split])
        and all(
            np.array_equal(a["input"], b["input"])
            and np.array_equal(a["output"], b["output"])
            for a, b in zip(first[split], second[split])
        )
        for split in ("train", "test")
    )


def self_test() -> None:
    a = generate_puzzle(87001, 7, 3)
    b = generate_puzzle(87001, 7, 3)
    assert _puzzles_equal(a, b)
    assert not _puzzles_equal(a, generate_puzzle(87002, 7, 3))

    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(88000 + index, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(89000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for index, (train_count, test_count) in enumerate(
        ((1, None), (11, None), (None, 0), (None, 5),
         (2, 2), (3, 3), (4, 4), (2, 4))
    ):
        try:
            generate_puzzle(90000 + index, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal pair counts did not raise ValueError")

    train_counts, test_counts = set(), set()
    palettes, sizes, corners, marker_slots = set(), set(), set(), set()
    puzzle_signatures = set()
    size_relations, corner_relations, slot_relations = set(), set(), set()
    role_changes = 0
    new_test_colors = 0
    empty_quadrants = 0
    lopsided = 0
    crossings = 0
    isolated = 0
    clustered = 0
    densities = set()
    combined = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_signatures.add(tuple(
            _grid_signature(pair["input"])
            for split in ("train", "test") for pair in puzzle[split]
        ))
        info = {
            split: [_analyze_pair(p["input"], p["output"]) for p in puzzle[split]]
            for split in ("train", "test")
        }
        train_key_union = set().union(*(i["key_set"] for i in info["train"]))
        if len({(i["background"], i["separator"]) for i in info["train"]}) > 1:
            role_changes += 1
        if any(set(i["key_set"]) - train_key_union for i in info["test"]):
            new_test_colors += 1
        train_corners = {i["corner"] for i in info["train"]}
        train_slots = {i["marker_position"] for i in info["train"]}
        for i in info["test"]:
            if i["side"] > max(t["side"] for t in info["train"]):
                size_relations.add("larger")
            if i["side"] < min(t["side"] for t in info["train"]):
                size_relations.add("smaller")
            if i["side"] in {t["side"] for t in info["train"]}:
                size_relations.add("equal")
            corner_relations.add("repeat" if i["corner"] in train_corners else "heldout")
            slot_relations.add("repeat" if i["marker_position"] in train_slots else "heldout")
        for split in ("train", "test"):
            for pair, i in zip(puzzle[split], info[split]):
                palettes.add(i["key_set"])
                sizes.add(i["side"])
                corners.add(i["corner"])
                marker_slots.add(i["marker_position"])
                features = _mask_features(pair["input"], i)
                empty_quadrants += features["occupied_quadrants"] < 4
                lopsided += features["dominance"] >= 0.55
                crossings += features["crossing"]
                isolated += features["isolated"] > 0
                clustered += features["adjacent_edges"] >= 2
                densities.add(int(features["density"] * 10))
                combined += (
                    i["corner"] == "LL"
                    and features["occupied_quadrants"] < 4
                    and features["adjacent_edges"] >= 2
                )

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_signatures) >= 40
    assert len(palettes) >= 30 and len(sizes) >= 10
    assert corners == set(_CORNERS) and marker_slots == set(range(4))
    assert size_relations == {"larger", "smaller", "equal"}
    assert corner_relations == slot_relations == {"repeat", "heldout"}
    assert role_changes >= 8 and new_test_colors >= 8
    assert empty_quadrants >= 15 and lopsided >= 15
    assert crossings >= 20 and isolated >= 20 and clustered >= 20
    assert len(densities) >= 4 and combined >= 2
