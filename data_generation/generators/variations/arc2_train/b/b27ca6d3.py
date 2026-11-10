import numpy as np


_SOURCE_INPUT_SIGNATURES = {
    (
        "0000000022000200",
        "2000000000000200",
        "0000000000000000",
        "0000000020000000",
        "0020000000200202",
        "0000000000000002",
        "0000000000200000",
        "0200000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0200000000000000",
        "0000000000000020",
        "0020000000200000",
    ),
    (
        "000000000000000000",
        "000000000000000000",
        "002000020000000000",
        "000000020000000000",
        "000000000000020000",
        "000000000000000000",
        "000000000000000000",
        "000000000000000000",
        "000000000000000002",
        "000000000000000000",
        "000000000000000000",
        "000020002000000000",
        "200000000000000000",
        "000000000000000000",
        "200000000000000002",
    ),
    (
        "0000000000000002",
        "0000200000000000",
        "0000000000000000",
        "0000000000000202",
        "0000000022000000",
        "0000000000002000",
        "0000000000000000",
        "0022000000000000",
        "0000000000000020",
        "0000002000000020",
        "2000000200000000",
        "0000000002020000",
        "0020200002000000",
        "0200002200020200",
        "0002000000000002",
        "0000000200000000",
        "0000000020000000",
    ),
}


def _checked_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    return num_train, num_test


def _neighbors(
    row: int,
    column: int,
    height: int,
    width: int,
):
    if row > 0:
        yield row - 1, column
    if row + 1 < height:
        yield row + 1, column
    if column > 0:
        yield row, column - 1
    if column + 1 < width:
        yield row, column + 1


def _components(
    grid: np.ndarray,
    marker: int,
) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    result = []

    for row in range(height):
        for column in range(width):
            if seen[row, column] or grid[row, column] != marker:
                continue

            seen[row, column] = True
            stack = [(row, column)]
            component = []

            while stack:
                current_row, current_column = stack.pop()
                component.append((current_row, current_column))

                for next_row, next_column in _neighbors(
                    current_row,
                    current_column,
                    height,
                    width,
                ):
                    if (
                        not seen[next_row, next_column]
                        and grid[next_row, next_column] == marker
                    ):
                        seen[next_row, next_column] = True
                        stack.append((next_row, next_column))

            result.append(sorted(component))

    return result


def _profile(
    grid: np.ndarray,
    marker: int,
) -> dict:
    singles = []
    dominoes = []
    invalid = []

    for component in _components(grid, marker):
        if len(component) == 1:
            singles.append(component[0])
        elif len(component) == 2:
            first, second = component
            distance = (
                abs(first[0] - second[0])
                + abs(first[1] - second[1])
            )
            if distance == 1:
                dominoes.append(component)
            else:
                invalid.append(component)
        else:
            invalid.append(component)

    return {
        "singles": singles,
        "dominoes": dominoes,
        "large": invalid,
    }


def _orientation(component: list[tuple[int, int]]) -> str:
    return "H" if component[0][0] == component[1][0] else "V"


def _touches_boundary(
    component: list[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    return any(
        row in (0, height - 1) or column in (0, width - 1)
        for row, column in component
    )


def _domino_is_interior(
    component: list[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    height, width = shape
    rows = [point[0] for point in component]
    columns = [point[1] for point in component]

    return (
        min(rows) > 0
        and max(rows) < height - 1
        and min(columns) > 0
        and max(columns) < width - 1
    )


def _expanded_box(
    component: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = shape
    rows = [point[0] for point in component]
    columns = [point[1] for point in component]

    return (
        max(0, min(rows) - 1),
        min(height - 1, max(rows) + 1),
        max(0, min(columns) - 1),
        min(width - 1, max(columns) + 1),
    )


def _expanded_boxes_overlap(
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    first_r0, first_r1, first_c0, first_c1 = _expanded_box(
        first, shape
    )
    second_r0, second_r1, second_c0, second_c1 = _expanded_box(
        second, shape
    )

    rows_overlap = max(first_r0, second_r0) <= min(
        first_r1, second_r1
    )
    columns_overlap = max(first_c0, second_c0) <= min(
        first_c1, second_c1
    )
    return rows_overlap and columns_overlap


def _halos_overlap_on_background(
    grid: np.ndarray,
    background: int,
    dominoes: list[list[tuple[int, int]]],
) -> bool:
    halo_sets = []

    for component in dominoes:
        row0, row1, column0, column1 = _expanded_box(
            component, grid.shape
        )
        halo_sets.append(
            {
                (row, column)
                for row in range(row0, row1 + 1)
                for column in range(column0, column1 + 1)
                if grid[row, column] == background
            }
        )

    return any(
        halo_sets[first_index].intersection(halo_sets[second_index])
        for first_index in range(len(halo_sets))
        for second_index in range(first_index + 1, len(halo_sets))
    )


def _can_place(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    background: int,
) -> bool:
    height, width = grid.shape
    cell_set = set(cells)

    if len(cell_set) != len(cells):
        return False

    for row, column in cells:
        if not (
            0 <= row < height
            and 0 <= column < width
            and grid[row, column] == background
        ):
            return False

        for next_row, next_column in _neighbors(
            row, column, height, width
        ):
            if (
                (next_row, next_column) not in cell_set
                and grid[next_row, next_column] != background
            ):
                return False

    return True


def _place_cells(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    background: int,
    marker: int,
) -> None:
    if not _can_place(grid, cells, background):
        raise RuntimeError("planned marker placement was not feasible")

    for row, column in cells:
        grid[row, column] = marker


def _domino_candidates(
    height: int,
    width: int,
    orientation: str,
    interior: bool = False,
    boundary: bool = False,
) -> list[list[tuple[int, int]]]:
    candidates = []

    if orientation == "H":
        for row in range(height):
            for column in range(width - 1):
                cells = [
                    (row, column),
                    (row, column + 1),
                ]
                is_interior = (
                    0 < row < height - 1
                    and 0 < column
                    and column + 1 < width - 1
                )
                if (
                    (not interior or is_interior)
                    and (
                        not boundary
                        or _touches_boundary(cells, height, width)
                    )
                ):
                    candidates.append(cells)

    elif orientation == "V":
        for row in range(height - 1):
            for column in range(width):
                cells = [
                    (row, column),
                    (row + 1, column),
                ]
                is_interior = (
                    0 < row
                    and row + 1 < height - 1
                    and 0 < column < width - 1
                )
                if (
                    (not interior or is_interior)
                    and (
                        not boundary
                        or _touches_boundary(cells, height, width)
                    )
                ):
                    candidates.append(cells)
    else:
        raise ValueError("orientation must be 'H' or 'V'")

    return candidates


def _place_random_domino(
    rng: np.random.Generator,
    grid: np.ndarray,
    background: int,
    marker: int,
    orientation: str,
    interior: bool = False,
    boundary: bool = False,
    avoid_halo_overlap: bool = False,
) -> list[tuple[int, int]]:
    candidates = _domino_candidates(
        grid.shape[0],
        grid.shape[1],
        orientation,
        interior=interior,
        boundary=boundary,
    )

    existing_dominoes = _profile(grid, marker)["dominoes"]

    for index in rng.permutation(len(candidates)):
        cells = candidates[int(index)]

        if not _can_place(grid, cells, background):
            continue

        if avoid_halo_overlap and any(
            _expanded_boxes_overlap(cells, existing, grid.shape)
            for existing in existing_dominoes
        ):
            continue

        _place_cells(grid, cells, background, marker)
        return cells

    raise RuntimeError("could not place a requested domino")


def _place_singletons(
    rng: np.random.Generator,
    grid: np.ndarray,
    background: int,
    marker: int,
    count: int,
) -> None:
    height, width = grid.shape
    placed = 0

    for flat_index in rng.permutation(height * width):
        row, column = divmod(int(flat_index), width)

        if _can_place(grid, [(row, column)], background):
            grid[row, column] = marker
            placed += 1

            if placed == count:
                return

    raise RuntimeError(
        "could not place all requested singleton distractors"
    )


def _apply_rule(
    input_grid: np.ndarray,
    background: int,
    marker: int,
    accent: int,
) -> np.ndarray:
    output_grid = input_grid.copy()

    for component in _components(input_grid, marker):
        if len(component) != 2:
            continue

        first, second = component
        distance = (
            abs(first[0] - second[0])
            + abs(first[1] - second[1])
        )
        if distance != 1:
            continue

        row0, row1, column0, column1 = _expanded_box(
            component, input_grid.shape
        )

        for row in range(row0, row1 + 1):
            for column in range(column0, column1 + 1):
                if input_grid[row, column] == background:
                    output_grid[row, column] = accent

    return output_grid


def _relationship(first, second, shape):
    """Classify the interaction of two expanded domino rectangles."""
    a0, a1, a2, a3 = _expanded_box(first, shape)
    b0, b1, b2, b3 = _expanded_box(second, shape)
    row_gap = max(b0 - a1, a0 - b1, 0)
    col_gap = max(b2 - a3, a2 - b3, 0)
    if row_gap == col_gap == 0:
        # Distinct components can occupy some of the intersection. Require
        # at least one actual background cell in the common halo.
        occupied = set(first) | set(second)
        for row in range(max(a0, b0), min(a1, b1) + 1):
            for col in range(max(a2, b2), min(a3, b3) + 1):
                if (row, col) not in occupied:
                    return "overlap"
    if max(row_gap, col_gap) <= 1:
        return "touching"
    return "disjoint"


def _choose_domino(rng, grid, background, marker, orientation,
                    existing, relation=None, location=None):
    candidates = _domino_candidates(*grid.shape, orientation)
    for index in rng.permutation(len(candidates)):
        cells = candidates[int(index)]
        if not _can_place(grid, cells, background):
            continue
        if location == "interior" and not _domino_is_interior(cells, grid.shape):
            continue
        if location == "edge" and not _touches_boundary(cells, *grid.shape):
            continue
        if location == "corner" and not any(
            r in (0, grid.shape[0] - 1) and c in (0, grid.shape[1] - 1)
            for r, c in cells
        ):
            continue
        if relation is not None and not any(
            _relationship(cells, other, grid.shape) == relation
            for other in existing
        ):
            continue
        _place_cells(grid, cells, background, marker)
        return cells
    raise RuntimeError("no feasible domino placement")


def _place_large_component(rng, grid, background, marker):
    # This optional negative example is a connected three-cell path.
    height, width = grid.shape
    shapes = (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 0)),
        ((0, 0), (0, 1), (1, 1)),
        ((0, 1), (1, 0), (1, 1)),
    )
    for _ in range(200):
        shape = shapes[int(rng.integers(len(shapes)))]
        row = int(rng.integers(height - 2))
        col = int(rng.integers(width - 2))
        cells = [(row + dr, col + dc) for dr, dc in shape]
        if _can_place(grid, cells, background):
            _place_cells(grid, cells, background, marker)
            return
    raise RuntimeError("no feasible larger component")


def _pair_dimensions(rng):
    # Independent axes produce square, wide, and tall canvases.
    bands = ((10, 13), (13, 20), (20, 27), (27, 31))
    def axis():
        low, high = bands[int(rng.integers(4))]
        return int(rng.integers(low, high))
    return axis(), axis()


def _make_pair(rng, parameters, regime, required_orientation,
               interaction, large_example, palette):
    background, marker = palette
    accent = parameters["accent"]
    for _ in range(80):
        height, width = _pair_dimensions(rng)
        area = height * width
        maximum = min(7, max(2, area // 65))
        if regime == "source_train":
            count = int(rng.integers(2, maximum + 1)) if interaction else int(
                rng.integers(1, maximum + 1)
            )
        else:
            if rng.random() < 0.055:
                count = 0
            elif interaction:
                count = int(rng.integers(2, maximum + 1))
            else:
                count = int(rng.integers(1, maximum + 1))
        if count < 2:
            interaction = None
        grid = np.full((height, width), background, dtype=np.int8)
        dominos = []
        try:
            if count:
                first_orientation = required_orientation or (
                    "H" if rng.random() < 0.5 else "V"
                )
                first_location = "interior" if regime == "source_train" else rng.choice(
                    ("interior", "interior", "edge", "corner")
                )
                dominos.append(_choose_domino(
                    rng, grid, background, marker, first_orientation,
                    dominos, location=first_location,
                ))
            for index in range(1, count):
                orientation = "H" if rng.random() < 0.5 else "V"
                relation = interaction if index == 1 else (
                    rng.choice(("overlap", "touching", "disjoint", None))
                    if rng.random() < 0.55 else None
                )
                location = None if relation else rng.choice(
                    (None, None, None, "edge", "corner", "interior")
                )
                dominos.append(_choose_domino(
                    rng, grid, background, marker, orientation, dominos,
                    relation=relation, location=location,
                ))
            if large_example:
                _place_large_component(rng, grid, background, marker)
            # Counts and density vary independently of split and size.
            density = float(rng.uniform(0.012, 0.105))
            singleton_count = max(1, min(55, int(round(area * density))))
            if count == 0:
                singleton_count = max(2, singleton_count)
            # Sometimes a singleton falls inside a halo and must be retained.
            if dominos and rng.random() < 0.5:
                halo_cells = []
                for domino in dominos:
                    r0, r1, c0, c1 = _expanded_box(domino, grid.shape)
                    halo_cells.extend((r, c) for r in range(r0, r1 + 1)
                                      for c in range(c0, c1 + 1))
                for index in rng.permutation(len(halo_cells)):
                    cell = halo_cells[int(index)]
                    if _can_place(grid, [cell], background):
                        _place_cells(grid, [cell], background, marker)
                        singleton_count -= 1
                        break
            if singleton_count:
                _place_singletons(rng, grid, background, marker, singleton_count)
            if interaction == "overlap" and not _halos_overlap_on_background(
                grid, background, dominos[:2]
            ):
                raise RuntimeError("planned halo overlap was obscured")
            if np.count_nonzero(grid == marker) >= area // 3:
                raise RuntimeError("markers not sparse enough")
            output = _apply_rule(grid, background, marker, accent)
            if count and np.array_equal(grid, output):
                raise RuntimeError("halo did not change a background cell")
            return grid, output
        except RuntimeError:
            continue
    raise RuntimeError("could not construct pair")


def _grid_value_key(grid):
    return grid.shape, tuple(int(value) for value in grid.ravel())


def _row_signature(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _infer_roles(pair):
    values, counts = np.unique(pair["input"], return_counts=True)
    assert len(values) == 2
    background = int(values[np.argmax(counts)])
    marker = int(values[np.argmin(counts)])
    assert int(max(counts)) > int(min(counts))
    return background, marker


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    accent = int(rng.integers(10))
    available = [color for color in range(10) if color != accent]
    rng.shuffle(available)
    base_palette = (available[0], available[1])
    parameters = {"accent": accent}
    puzzle = {"train": [], "test": []}
    used_inputs = set()

    # The accent is the only learned literal color. Input roles are identified
    # by abundance in each pair, allowing background/marker changes and swaps.
    palettes = []
    for index in range(train_count + test_count):
        if index == 0:
            palettes.append(base_palette)
        elif index == 1 and rng.random() < 0.4:
            palettes.append(base_palette[::-1])
        elif rng.random() < 0.28:
            palettes.append(base_palette)
        else:
            chosen = rng.choice(available, size=2, replace=False)
            palettes.append((int(chosen[0]), int(chosen[1])))

    required = [None] * train_count
    first_orientation = "H" if rng.random() < 0.5 else "V"
    required[0] = first_orientation
    required[1] = "V" if first_orientation == "H" else "H"
    modes = ["overlap", "touching", "disjoint"]
    train_modes = [modes[int(rng.integers(3))] if rng.random() < 0.8 else None
                   for _ in range(train_count)]
    test_modes = [modes[int(rng.integers(3))] if rng.random() < 0.8 else None
                  for _ in range(test_count)]
    if "overlap" in test_modes and "overlap" not in train_modes:
        train_modes[int(rng.integers(train_count))] = "overlap"
    # A larger negative component is introduced in training first.
    large_rule_probe = rng.random() < 0.35
    large_training_index = int(rng.integers(train_count)) if large_rule_probe else -1

    for split, pair_count in (("train", train_count), ("test", test_count)):
        for index in range(pair_count):
            global_index = index if split == "train" else train_count + index
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                large = (index == large_training_index if split == "train"
                         else large_rule_probe and rng.random() < 0.4)
                input_grid, output_grid = _make_pair(
                    pair_rng, parameters,
                    "source_train" if split == "train" else "target_test",
                    required[index] if split == "train" else None,
                    train_modes[index] if split == "train" else test_modes[index],
                    large, palettes[global_index],
                )
                key = _grid_value_key(input_grid)
                if key not in used_inputs:
                    used_inputs.add(key)
                    break
            else:
                raise RuntimeError("could not generate a unique pair")
            puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    all_pairs = puzzle["train"] + puzzle["test"]
    assert all(isinstance(p, dict) and set(p) == {"input", "output"}
               for p in all_pairs)
    train_accents = set()
    seen = set()
    train_orientations = set()
    train_has_large = False
    test_has_large = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert inp.shape == out.shape
            key = _grid_value_key(inp)
            assert key not in seen
            seen.add(key)
            assert _row_signature(inp) not in _SOURCE_INPUT_SIGNATURES
            background, marker = _infer_roles(pair)
            profile = _profile(inp, marker)
            assert profile["singles"]
            assert np.count_nonzero(inp == background) > 2 * np.count_nonzero(inp == marker)
            output_colors = {int(value) for value in np.unique(out)}
            assert output_colors.issubset(set(range(10)))
            extras = output_colors - {background, marker}
            assert len(extras) <= 1
            if split == "train":
                assert profile["dominoes"] and extras
                train_accents.update(extras)
                train_orientations.update(_orientation(d) for d in profile["dominoes"])
                train_has_large |= bool(profile["large"])
            else:
                test_has_large |= bool(profile["large"])
                if not profile["dominoes"]:
                    assert np.array_equal(inp, out)
            assert np.array_equal(inp[inp == marker], out[inp == marker])
            if extras:
                accent = next(iter(extras))
                assert np.array_equal(out, _apply_rule(inp, background, marker, accent))
            else:
                assert np.array_equal(inp, out)
    assert len(train_accents) == 1
    accent = next(iter(train_accents))
    assert train_orientations == {"H", "V"}
    assert not test_has_large or train_has_large
    for pair in all_pairs:
        background, marker = _infer_roles(pair)
        assert np.array_equal(pair["output"],
                              _apply_rule(pair["input"], background, marker, accent))


def _puzzles_equal(left, right):
    return all(len(left[s]) == len(right[s]) and all(
        np.array_equal(a[k], b[k]) for a, b in zip(left[s], right[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(314159), generate_puzzle(314159))
    for train, test in ((2, 1), (5, 3), (10, 4)):
        puzzle = generate_puzzle(train * 100 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for kwargs in ({"num_train": 2}, {"num_test": 4}, {"num_train": 10}):
        validate_generated_puzzle(generate_puzzle(888, **kwargs))
    for kwargs in ({"num_train": 1}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(5, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    shapes, counts, relations, colors = set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        counts.add((len(puzzle["train"]), len(puzzle["test"])))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                background, marker = _infer_roles(pair)
                profile = _profile(grid, marker)
                shapes.add(grid.shape)
                colors.add((background, marker))
                dominoes = profile["dominoes"]
                for i in range(len(dominoes)):
                    for j in range(i + 1, len(dominoes)):
                        relations.add((split, _relationship(dominoes[i], dominoes[j], grid.shape)))
    assert len(shapes) >= 45 and len(counts) >= 10 and len(colors) >= 20
    assert all((split, relation) in relations for split in ("train", "test")
               for relation in ("overlap", "touching", "disjoint"))
