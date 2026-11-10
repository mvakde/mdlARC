from __future__ import annotations

import numpy as np


_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))

_ORIGINAL_ROLE_TUPLE = (7, 9, 6, 2, 5)
_SOURCE_INPUTS = {
    ("775777777", "777777777", "777777775", "777775777", "999999999",
     "777772777", "777776762", "777776777", "772677777"),
    ("7777577", "7577777", "7777777", "9999999", "7777777",
     "7267677", "7777277"),
    ("577777757", "777777777", "775577777", "777777777", "999999999",
     "777677777", "772277767", "776777767", "267777727"),
    ("77777775757", "77777777777", "77777775777", "77777777777",
     "77757777775", "99999999999", "77727777662", "77767777777",
     "77777772667", "77777777777", "77766662726"),
}


def _is_plain_int(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if not _is_plain_int(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)

    if num_test is not None:
        if not _is_plain_int(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_train = int(rng.choice((2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10)))
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    else:
        upper = min(4, num_train - 1)
        num_test = 1 + int(rng.integers(upper))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError("the requested pair counts cannot form a legal puzzle")

    return num_train, num_test


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    roles = (_ORIGINAL_ROLE_TUPLE if rng.random() < 0.10 else
             tuple(int(x) for x in rng.choice(10, size=5, replace=False)))

    return {
        "background": roles[0],
        "separator": roles[1],
        "guide": roles[2],
        "guide_marker": roles[3],
        "other_marker": roles[4],
        "orientation": int(rng.integers(2)),
        "guide_side": -1 if int(rng.integers(2)) == 0 else 1,
        "palette_mode": str(rng.choice(("constant", "partial", "new"),
                                        p=(0.48, 0.32, 0.20))),
        "orientation_mode": str(rng.choice(("stable", "mixed", "shift"),
                                            p=(0.37, 0.38, 0.25))),
        "side_mode": str(rng.choice(("stable", "mixed", "shift"),
                                     p=(0.38, 0.44, 0.18))),
    }


def _candidate_compatible(candidate: tuple, chosen: list[tuple]) -> bool:
    marker, guide, _, _ = candidate
    footprint = {marker, *guide}

    for old_marker, old_guide, _, _ in chosen:
        old_footprint = {old_marker, *old_guide}

        if footprint & old_footprint:
            return False

        # Separate guide components may not join or branch.
        if any(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            for a in guide
            for b in old_guide
        ):
            return False

        # A marker may touch only its own guide.
        if any(
            abs(marker[0] - b[0]) + abs(marker[1] - b[1]) == 1
            for b in old_guide
        ):
            return False

        if any(
            abs(old_marker[0] - a[0]) + abs(old_marker[1] - a[1]) == 1
            for a in guide
        ):
            return False

    return True


def _place_objects(
    rng: np.random.Generator,
    depth: int,
    span: int,
    specifications: list[tuple[int, int]],
    focus: str = "uniform",
) -> list[tuple]:
    all_candidates = []

    for direction_index, length in specifications:
        dr, dc = _DIRECTIONS[direction_index]
        candidates = []
        weights = []

        for row in range(depth):
            for col in range(span):
                end_row = row + length * dr
                end_col = col + length * dc

                if not (0 <= end_row < depth and 0 <= end_col < span):
                    continue

                guide = tuple(
                    (row + k * dr, col + k * dc)
                    for k in range(1, length + 1)
                )
                candidates.append(
                    ((row, col), guide, direction_index, length)
                )
                favored = (
                    (focus == "separator" and row == 0)
                    or (focus == "far_edge" and row == depth - 1)
                    or (focus == "side_edge" and col in (0, span - 1))
                    or (focus == "interior" and 0 < row < depth - 1
                        and 1 < col < span - 2)
                )
                weights.append(4.0 if favored else 1.0)

        if not candidates:
            raise RuntimeError("no legal placement for guide")
        priority = -np.log(rng.random(len(candidates))) / np.asarray(weights)
        order = np.argsort(priority)
        all_candidates.append(
            [candidates[int(index)] for index in order]
        )

    object_order = sorted(
        range(len(specifications)),
        key=lambda index: (
            -specifications[index][1],
            len(all_candidates[index]),
        ),
    )

    selected: dict[int, tuple] = {}
    node_count = 0

    def search(position: int) -> bool:
        nonlocal node_count

        if position == len(object_order):
            return True

        index = object_order[position]

        for candidate in all_candidates[index]:
            node_count += 1
            if node_count > 20_000:
                return False

            if _candidate_compatible(
                candidate, list(selected.values())
            ):
                selected[index] = candidate

                if search(position + 1):
                    return True

                del selected[index]

        return False

    if not search(0):
        raise RuntimeError(
            "could not construct a non-ambiguous guide layout"
        )

    return [selected[index] for index in range(len(specifications))]


def _reflect(
    position: tuple[int, int],
    separator_index: int,
    orientation: int,
) -> tuple[int, int]:
    row, col = position

    if orientation == 0:
        return 2 * separator_index - row, col

    return row, 2 * separator_index - col


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    params: dict,
    required: tuple[tuple[int, int], ...] | list[tuple[int, int]] = (),
    geometry: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    if geometry is None:
        depth = int(rng.integers(3, 7 if regime == "source_train" else 9))
        span = int(rng.integers(7, 16 if regime == "source_train" else 19))
        if rng.random() < 0.26:
            span = 2 * depth + 1
    else:
        depth, span = geometry
    count = int(rng.integers(2, 5 if regime == "source_train" else 7))
    count = max(count, len(required))
    focus = str(rng.choice(("separator", "far_edge", "side_edge", "interior", "uniform")))
    for _ in range(40):
        dominant = int(rng.integers(4)) if rng.random() < 0.48 else None
        directions = [int(rng.integers(4)) if dominant is None or rng.random() < 0.32
                      else dominant for _ in range(count)]
        lengths = [int(rng.integers(1, min(2, depth - 1 if direction < 2 else span - 1) + 1))
                   for direction in directions]
        if regime == "target_test":
            mode = str(rng.choice(("short", "one_long", "many_long"),
                                  p=(0.34, 0.43, 0.23)))
            eligible = [i for i, direction in enumerate(directions)
                        if (depth - 1 if direction < 2 else span - 1) >= 3]
            if mode != "short" and eligible:
                rng.shuffle(eligible)
                number = 1 if mode == "one_long" else min(len(eligible), int(rng.integers(2, 4)))
                for index in eligible[:number]:
                    extent = depth - 1 if directions[index] < 2 else span - 1
                    lengths[index] = int(rng.integers(3, min(5, extent) + 1))
        for index, (direction, length) in enumerate(required):
            directions[index], lengths[index] = direction, length
        permutation = rng.permutation(count)
        specifications = [(directions[int(i)], lengths[int(i)]) for i in permutation]
        try:
            objects = _place_objects(rng, depth, span, specifications, focus)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError("could not construct an unambiguous pair")

    background = params["background"]
    separator = params["separator"]
    guide_color = params["guide"]
    guide_marker = params["guide_marker"]
    other_marker = params["other_marker"]
    guide_side = params["guide_side"]

    height = 2 * depth + 1
    input_grid = np.full(
        (height, span), background, dtype=np.int8
    )
    input_grid[depth, :] = separator

    def guide_global(
        cell: tuple[int, int],
    ) -> tuple[int, int]:
        panel_row, col = cell

        if guide_side == 1:
            return depth + 1 + panel_row, col

        return depth - 1 - panel_row, col

    def other_global(
        cell: tuple[int, int],
    ) -> tuple[int, int]:
        panel_row, col = cell

        if guide_side == 1:
            return depth - 1 - panel_row, col

        return depth + 1 + panel_row, col

    for marker, guide, _, _ in objects:
        input_grid[guide_global(marker)] = guide_marker
        input_grid[other_global(marker)] = other_marker

        for cell in guide:
            input_grid[guide_global(cell)] = guide_color

    output_grid = input_grid.copy()

    for marker, guide, _, _ in objects:
        output_grid[guide_global(marker)] = background
        output_grid[other_global(marker)] = background

        for cell in guide:
            output_grid[guide_global(cell)] = background

        endpoint = guide[-1]
        output_grid[guide_global(endpoint)] = guide_marker
        output_grid[other_global(endpoint)] = other_marker

    if params["orientation"] == 1:
        input_grid = np.ascontiguousarray(input_grid.T)
        output_grid = np.ascontiguousarray(output_grid.T)

    return input_grid, output_grid


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, tuple(int(value) for value in grid.ravel())


def _pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return _grid_key(input_grid), _grid_key(output_grid)


def _find_separator_and_background(
    grid: np.ndarray,
) -> tuple[int, int, int, int]:
    values, counts = np.unique(grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    candidates = []

    for row in range(grid.shape[0]):
        color = int(grid[row, 0])

        if color != background and np.all(grid[row, :] == color):
            candidates.append((0, row, color))

    for col in range(grid.shape[1]):
        color = int(grid[0, col])

        if color != background and np.all(grid[:, col] == color):
            candidates.append((1, col, color))

    assert len(candidates) == 1
    orientation, index, separator = candidates[0]

    if orientation == 0:
        assert index == grid.shape[0] - 1 - index
    else:
        assert index == grid.shape[1] - 1 - index

    return orientation, index, background, separator


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    regime: str | None = None,
) -> dict:
    assert input_grid.shape == output_grid.shape

    (
        orientation,
        separator_index,
        background,
        separator,
    ) = _find_separator_and_background(input_grid)

    (
        out_orientation,
        out_index,
        out_background,
        out_separator,
    ) = _find_separator_and_background(output_grid)

    assert (
        out_orientation,
        out_index,
        out_background,
        out_separator,
    ) == (
        orientation,
        separator_index,
        background,
        separator,
    )

    input_colors = {int(value) for value in np.unique(input_grid)}
    output_colors = {int(value) for value in np.unique(output_grid)}

    assert len(input_colors) == 5
    assert len(output_colors) == 4
    assert background in input_colors
    assert separator in input_colors

    marker_colors = output_colors - {background, separator}
    assert len(marker_colors) == 2

    residual = (
        input_colors
        - {background, separator}
        - marker_colors
    )
    assert len(residual) == 1

    guide_color = next(iter(residual))
    assert guide_color not in output_colors

    def axis_value(position: tuple[int, int]) -> int:
        if orientation == 0:
            return position[0]
        return position[1]

    guide_positions = [
        tuple(int(value) for value in position)
        for position in np.argwhere(input_grid == guide_color)
    ]
    assert guide_positions

    guide_signs = {
        int(np.sign(axis_value(position) - separator_index))
        for position in guide_positions
    }
    assert len(guide_signs) == 1
    assert 0 not in guide_signs

    guide_side = next(iter(guide_signs))
    side_by_color = {}
    positions_by_color = {}

    for color in marker_colors:
        positions = {
            tuple(int(value) for value in position)
            for position in np.argwhere(input_grid == color)
        }
        assert positions

        signs = {
            int(np.sign(axis_value(position) - separator_index))
            for position in positions
        }
        assert len(signs) == 1
        assert 0 not in signs

        positions_by_color[color] = positions
        side_by_color[color] = next(iter(signs))

    matches = [
        color
        for color in marker_colors
        if side_by_color[color] == guide_side
    ]
    assert len(matches) == 1

    guide_marker = matches[0]
    other_marker = next(
        color
        for color in marker_colors
        if color != guide_marker
    )

    assert side_by_color[other_marker] == -guide_side
    assert len(positions_by_color[guide_marker]) == len(
        positions_by_color[other_marker]
    )

    reflected_markers = {
        _reflect(position, separator_index, orientation)
        for position in positions_by_color[guide_marker]
    }
    assert reflected_markers == positions_by_color[other_marker]

    guide_set = set(guide_positions)
    used_guide_cells = set()
    guide_owners = {}
    records = []
    expected = input_grid.copy()

    for marker in sorted(positions_by_color[guide_marker]):
        neighbors = []

        for dr, dc in _DIRECTIONS:
            neighbor = marker[0] + dr, marker[1] + dc

            if neighbor in guide_set:
                neighbors.append((neighbor, (dr, dc)))

        assert len(neighbors) == 1
        _, direction = neighbors[0]

        trace = []
        k = 1

        while (
            marker[0] + k * direction[0],
            marker[1] + k * direction[1],
        ) in guide_set:
            trace.append(
                (
                    marker[0] + k * direction[0],
                    marker[1] + k * direction[1],
                )
            )
            k += 1

        assert trace
        assert not (set(trace) & used_guide_cells)
        used_guide_cells.update(trace)
        for cell in trace:
            guide_owners[cell] = marker

        counterpart = _reflect(
            marker, separator_index, orientation
        )

        expected[marker] = background
        expected[counterpart] = background

        for cell in trace:
            expected[cell] = background

        endpoint = trace[-1]
        expected[endpoint] = guide_marker
        expected[
            _reflect(endpoint, separator_index, orientation)
        ] = other_marker

        records.append(
            (marker, direction, len(trace), endpoint)
        )

    assert used_guide_cells == guide_set
    for cell in guide_set:
        for dr, dc in _DIRECTIONS:
            neighbor = cell[0] + dr, cell[1] + dc
            if neighbor in guide_set:
                assert guide_owners[neighbor] == guide_owners[cell]
    assert np.array_equal(expected, output_grid)
    assert not np.array_equal(input_grid, output_grid)

    if orientation == 0:
        depth = separator_index
        span = input_grid.shape[1]
    else:
        depth = separator_index
        span = input_grid.shape[0]

    count = len(records)
    lengths = tuple(record[2] for record in records)
    directions = tuple(record[1] for record in records)

    if regime == "source_train":
        assert 3 <= depth <= 6
        assert 7 <= span <= 15
        assert 2 <= count <= 4
        assert all(1 <= length <= 2 for length in lengths)

    elif regime == "target_test":
        assert 3 <= depth <= 8
        assert 7 <= span <= 18
        assert 2 <= count <= 6
        assert all(1 <= length <= 5 for length in lengths)

    elif regime is not None:
        raise AssertionError("unknown validation regime")

    return {
        "orientation": orientation,
        "separator_index": separator_index,
        "background": background,
        "separator": separator,
        "guide": guide_color,
        "guide_marker": guide_marker,
        "other_marker": other_marker,
        "guide_side": guide_side,
        "depth": depth,
        "span": span,
        "count": count,
        "lengths": lengths,
        "directions": directions,
        "records": tuple(records),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys = set()
    pair_keys = set()
    analyses = {"train": [], "test": []}

    for split, regime in (
        ("train", "source_train"),
        ("test", "target_test"),
    ):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))

            input_key = _grid_key(input_grid)
            current_pair_key = _pair_key(
                input_grid, output_grid
            )
            source_rows = tuple("".join(str(int(value)) for value in row)
                                for row in input_grid)
            assert source_rows not in _SOURCE_INPUTS

            assert input_key not in input_keys
            assert current_pair_key not in pair_keys

            input_keys.add(input_key)
            pair_keys.add(current_pair_key)

            analyses[split].append(
                _analyze_pair(
                    input_grid, output_grid, regime
                )
            )

    training_directions = set()
    training_lengths = set()

    for analysis in analyses["train"]:
        training_directions.update(analysis["directions"])
        training_lengths.update(analysis["lengths"])

    # A multi-cell guide makes the distal endpoint learnable; both axes
    # demonstrate that the rule is geometric rather than color-specific.
    assert any(direction[0] for direction in training_directions)
    assert any(direction[1] for direction in training_directions)
    assert 2 in training_lengths


def _pair_parameters(rng, base, split, pair_index):
    params = base.copy()
    mode = base["palette_mode"]
    if mode != "constant" and pair_index:
        if mode == "new":
            roles = [int(value) for value in rng.choice(10, 5, replace=False)]
        else:
            roles = [base[key] for key in ("background", "separator", "guide",
                                               "guide_marker", "other_marker")]
            indices = [int(rng.integers(5))]
            indices.extend(i for i in range(5) if i not in indices and rng.random() < 0.3)
            for index in indices:
                roles[index] = int(rng.choice([color for color in range(10)
                                                if color not in roles]))
        for key, color in zip(("background", "separator", "guide",
                               "guide_marker", "other_marker"), roles):
            params[key] = color
    for field, mode_field in (("orientation", "orientation_mode"),
                              ("guide_side", "side_mode")):
        choice = base[mode_field]
        if choice == "mixed":
            params[field] = int(rng.integers(2)) if field == "orientation" else (
                -1 if rng.integers(2) == 0 else 1)
        elif choice == "shift" and split == "test":
            params[field] = 1 - base[field] if field == "orientation" else -base[field]
    return params


def _geometry(rng, split, train_geometries):
    if split == "train":
        depth = int(rng.integers(3, 7))
        span = int(rng.integers(7, 16))
        if rng.random() < 0.27:
            span = 2 * depth + 1
        return depth, span
    relation = str(rng.choice(("larger", "same", "smaller", "free"),
                              p=(0.38, 0.22, 0.18, 0.22)))
    reference = train_geometries[int(rng.integers(len(train_geometries)))]
    if relation == "same":
        return reference
    if relation == "smaller":
        return (max(3, reference[0] - int(rng.integers(0, 3))),
                max(7, reference[1] - int(rng.integers(1, 4))))
    if relation == "larger":
        depth = min(8, max(d for d, _ in train_geometries) + int(rng.integers(0, 3)))
        span = min(18, max(s for _, s in train_geometries) + int(rng.integers(0, 4)))
        if (depth, span) == (max(d for d, _ in train_geometries),
                             max(s for _, s in train_geometries)):
            span = min(18, span + 1)
        return depth, span
    return int(rng.integers(3, 9)), int(rng.integers(7, 19))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_plain_int(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    base = _sample_puzzle_parameters(rng)
    required = [[] for _ in range(train_count)]
    required[int(rng.integers(train_count))].append((int(rng.choice((0, 1))), 2))
    required[int(rng.integers(train_count))].append((int(rng.choice((2, 3))),
                                                     int(rng.integers(1, 3))))
    puzzle = {"train": [], "test": []}
    train_geometries = []
    seen_inputs = set()
    pair_index = 0
    for split, count, regime in (("train", train_count, "source_train"),
                                 ("test", test_count, "target_test")):
        for index in range(count):
            for _ in range(120):
                child_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(child_seed)
                geometry = _geometry(pair_rng, split, train_geometries)
                params = _pair_parameters(pair_rng, base, split, pair_index)
                pair_required = required[index] if split == "train" else ()
                if params["orientation"] == 1:
                    # Construction uses transposed panel coordinates. Keep
                    # the training demonstrations on both visible axes.
                    pair_required = [((direction + 2) % 4, length)
                                     for direction, length in pair_required]
                try:
                    input_grid, output_grid = _generate_pair(
                        pair_rng, regime, params, pair_required, geometry)
                    _analyze_pair(input_grid, output_grid, regime)
                except (RuntimeError, AssertionError):
                    continue
                input_key = _grid_key(input_grid)
                source_rows = tuple("".join(str(int(value)) for value in row)
                                    for row in input_grid)
                if input_key in seen_inputs or source_rows in _SOURCE_INPUTS:
                    continue
                break
            else:
                raise ValueError("unable to construct unique, unambiguous pairs")
            seen_inputs.add(input_key)
            puzzle[split].append({"input": input_grid, "output": output_grid})
            if split == "train":
                train_geometries.append(geometry)
            pair_index += 1
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != set(second):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for first_pair, second_pair in zip(
            first[split], second[split]
        ):
            if not np.array_equal(
                first_pair["input"],
                second_pair["input"],
            ):
                return False

            if not np.array_equal(
                first_pair["output"],
                second_pair["output"],
            ):
                return False

    return True


def _expect_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return

    raise AssertionError("expected ValueError")


def _expect_assertion(function) -> None:
    try:
        function()
    except AssertionError:
        return

    raise AssertionError("expected AssertionError")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts = {len(puzzle["train"]) for puzzle in puzzles}
    test_counts = {len(puzzle["test"]) for puzzle in puzzles}
    shapes, layouts, role_tuples, directions = set(), set(), set(), set()
    train_lengths, test_lengths, train_objects, test_objects = set(), set(), set(), set()
    rectangular = square = short_train = short_test = long_test = 0
    mixed_orientation = mixed_side = mixed_palette = 0
    same_size = smaller_test = larger_test = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        analyses = {
            split: [_analyze_pair(pair["input"], pair["output"])
                    for pair in puzzle[split]]
            for split in ("train", "test")
        }
        all_analyses = analyses["train"] + analyses["test"]
        mixed_orientation += len({a["orientation"] for a in all_analyses}) > 1
        mixed_side += len({a["guide_side"] for a in all_analyses}) > 1
        mixed_palette += len({tuple(a[k] for k in ("background", "separator", "guide",
                                                       "guide_marker", "other_marker"))
                              for a in all_analyses}) > 1
        short_train += any(max(a["lengths"]) == 1 for a in analyses["train"])
        short_test += any(max(a["lengths"]) <= 2 for a in analyses["test"])
        long_test += any(max(a["lengths"]) >= 3 for a in analyses["test"])
        train_areas = [a["depth"] * a["span"] for a in analyses["train"]]
        test_areas = [a["depth"] * a["span"] for a in analyses["test"]]
        same_size += any(area in train_areas for area in test_areas)
        smaller_test += any(area < max(train_areas) for area in test_areas)
        larger_test += any(area > max(train_areas) for area in test_areas)
        for split in ("train", "test"):
            for a in analyses[split]:
                shapes.add((a["depth"], a["span"], a["orientation"]))
                role_tuples.add(tuple(a[k] for k in ("background", "separator", "guide",
                                                    "guide_marker", "other_marker")))
                layouts.add((a["depth"], a["span"], a["orientation"], a["guide_side"],
                             a["records"]))
                directions.update(a["directions"])
                (train_lengths if split == "train" else test_lengths).update(a["lengths"])
                (train_objects if split == "train" else test_objects).add(a["count"])
                rectangular += a["span"] != 2 * a["depth"] + 1
                square += a["span"] == 2 * a["depth"] + 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 20 and len(layouts) >= 200 and len(role_tuples) >= 40
    assert rectangular >= 100 and square >= 20
    assert train_lengths == {1, 2} and {1, 2, 3, 4}.issubset(test_lengths)
    assert train_objects == {2, 3, 4} and {2, 3, 4, 5, 6}.issubset(test_objects)
    assert directions == set(_DIRECTIONS)
    assert short_train >= 8 and short_test >= 8 and long_test >= 8
    assert mixed_orientation >= 8 and mixed_side >= 8 and mixed_palette >= 8
    assert same_size >= 3 and smaller_test >= 6 and larger_test >= 6
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        puzzle = generate_puzzle(100 + index, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4),
                                                        (2, None), (10, None))):
        puzzle = generate_puzzle(200 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    assert _puzzles_equal(generate_puzzle(777, 7, 3), generate_puzzle(777, 7, 3))
    assert not _puzzles_equal(generate_puzzle(778, 7, 3), generate_puzzle(779, 7, 3))
    for call in (lambda: generate_puzzle(1, num_train=1),
                 lambda: generate_puzzle(1, num_train=11),
                 lambda: generate_puzzle(1, num_test=0),
                 lambda: generate_puzzle(1, num_test=5),
                 lambda: generate_puzzle(1, 2, 2),
                 lambda: generate_puzzle(1, 3, 4)):
        _expect_value_error(call)
    good = puzzles[0]
    bad = {split: [{key: value.copy() for key, value in pair.items()}
                   for pair in good[split]] for split in ("train", "test")}
    first = _analyze_pair(bad["train"][0]["input"], bad["train"][0]["output"])
    cell = ((first["separator_index"], 0) if first["orientation"] == 0
            else (0, first["separator_index"]))
    bad["train"][0]["output"][cell] = first["background"]
    _expect_assertion(lambda: validate_generated_puzzle(bad))
