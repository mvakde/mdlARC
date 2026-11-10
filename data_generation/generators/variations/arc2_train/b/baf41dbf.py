import numpy as np


_SIDES = ("top", "right", "bottom", "left")


def _check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _optional_count(
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
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _optional_count(num_train, "num_train", 2, 10)
    num_test = _optional_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        low_train = max(2, num_test + 1)
        num_train = low_train + (seed // 4) % (11 - low_train)
    elif num_train is None:
        low_train = max(2, num_test + 1)
        num_train = low_train + (seed + 17 * num_test) % (11 - low_train)
    elif num_test is None:
        high_test = min(4, num_train - 1)
        num_test = 1 + (seed + 11 * num_train) % high_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _sample_puzzle_parameters(rng: np.random.Generator) -> dict:
    colors = [int(x) for x in rng.choice(10, size=3, replace=False)]
    alternate = [
        int(x) for x in rng.choice(
            [color for color in range(10) if color not in colors],
            size=3,
            replace=False,
        )
    ]
    return {
        "palettes": [tuple(colors), tuple(alternate)],
        "switch_palette": bool(rng.random() < 0.35),
        "fixed_anchor": (
            _SIDES[int(rng.integers(0, 4))]
            if rng.random() < 0.35 else None
        ),
        "train_delay": bool(rng.random() < 0.65),
        "double_delay_test": bool(rng.random() < 0.45),
    }


def _draw_structure(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    horizontal_dividers: list[int],
    vertical_dividers: list[int],
    color: int,
) -> None:
    top, bottom, left, right = bounds

    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color

    for row in horizontal_dividers:
        grid[row, left : right + 1] = color
    for col in vertical_dividers:
        grid[top : bottom + 1, col] = color


def _sample_lines(
    rng: np.random.Generator,
    low: int,
    high: int,
    count: int,
) -> list[int]:
    dimension = high - low + 1
    _check(
        0 <= count <= (dimension - 3) // 2,
        "invalid divider count",
    )
    slack = dimension - 2 * count - 3
    offsets = sorted(
        int(rng.integers(0, slack + 1))
        for _ in range(count)
    )
    return [
        low + 2 + 2 * index + offset
        for index, offset in enumerate(offsets)
    ]


def _variable_line_count(
    rng: np.random.Generator,
    dimension: int,
) -> int:
    maximum = min(4, (dimension - 3) // 2)
    _check(maximum >= 1, "frame is too small for a divider")
    return int(rng.integers(1, maximum + 1))


def _generate_canonical_pair(
    rng: np.random.Generator,
    regime: str,
    index: int,
    total_in_regime: int,
    puzzle_parameters: dict,
    allowed_sides: set[str],
    palette: tuple[int, int, int],
    force_double_delay: bool = False,
    allow_multiple_delay: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    background, structure, marker = palette

    for _ in range(300):
        if regime == "source_train":
            if index == 0 or (index == 1 and total_in_regime > 2):
                side_count = 2
            elif index in (1, 2):
                side_count = 3
            else:
                side_count = int(rng.integers(2, 4))
        elif regime == "target_test":
            side_count = int(rng.integers(2, 4))
        else:
            raise ValueError("unknown pair regime")

        force_one_delay = (
            regime == "source_train"
            and index == 1
            and puzzle_parameters["train_delay"]
        )
        adjacent_pairs = [
            (first, second)
            for first in sorted(allowed_sides)
            for second in sorted(allowed_sides)
            if first < second
            and (
                (first in ("top", "bottom"))
                != (second in ("top", "bottom"))
            )
        ]
        if force_double_delay or force_one_delay:
            if not adjacent_pairs:
                continue
            pair = adjacent_pairs[int(rng.integers(0, len(adjacent_pairs)))]
            expanded = set(pair)
            remaining = sorted(allowed_sides - expanded)
            if side_count == 3:
                expanded.add(remaining[int(rng.integers(0, len(remaining)))])
        else:
            expanded = set(
                str(side)
                for side in rng.choice(
                    sorted(allowed_sides),
                    size=side_count,
                    replace=False,
                )
            )
            pair = None

        if regime == "source_train" and index == 0 and rng.random() < 0.25:
            old_height = old_width = 3
        else:
            minimum = (
                5 if regime == "source_train"
                and (index == 2 or (index == 1 and total_in_regime == 2))
                else 3
            )
            old_height = int(rng.integers(minimum, 15))
            old_width = int(rng.integers(minimum, 15))

        if regime == "source_train" and index == 0:
            divider_mode = "none"
        elif regime == "source_train" and index == 1 and total_in_regime > 2:
            divider_mode = ("horizontal", "vertical")[
                int(rng.integers(0, 2))
            ]
        elif regime == "source_train" and (
            index == 2 or (index == 1 and total_in_regime == 2)
        ):
            divider_mode = "both"
        else:
            divider_mode = ("none", "horizontal", "vertical", "both")[
                int(rng.integers(0, 4))
            ]

        max_horizontal = min(4, (old_height - 3) // 2)
        max_vertical = min(4, (old_width - 3) // 2)
        horizontal_count = (
            _variable_line_count(rng, old_height)
            if divider_mode in ("horizontal", "both")
            and max_horizontal > 0 else 0
        )
        vertical_count = (
            _variable_line_count(rng, old_width)
            if divider_mode in ("vertical", "both")
            and max_vertical > 0 else 0
        )
        if (
            regime == "source_train"
            and divider_mode == "both"
            and index in (1, 2)
            and rng.random() < 0.5
        ):
            horizontal_count = 1
            vertical_count = 1
        if divider_mode == "both" and (
            horizontal_count == 0 or vertical_count == 0
        ):
            continue
        if divider_mode in ("horizontal", "vertical") and (
            horizontal_count + vertical_count == 0
        ):
            continue

        distances = {
            side: int(rng.integers(1, 9 if regime == "source_train" else 10))
            if side in expanded else 0
            for side in _SIDES
        }
        if force_double_delay:
            distances[pair[0]] = max(2, distances[pair[0]])
            distances[pair[1]] = max(2, distances[pair[1]])
        margins = {
            side: int(rng.integers(1 if side in expanded else 0, 4))
            for side in _SIDES
        }
        final_top = margins["top"]
        final_left = margins["left"]
        old_top = final_top + distances["top"]
        old_left = final_left + distances["left"]
        old_bottom = old_top + old_height - 1
        old_right = old_left + old_width - 1
        final_bottom = old_bottom + distances["bottom"]
        final_right = old_right + distances["right"]
        height = final_bottom + 1 + margins["bottom"]
        width = final_right + 1 + margins["right"]
        if height > 30 or width > 30:
            continue

        old_bounds = (old_top, old_bottom, old_left, old_right)
        final_bounds = (final_top, final_bottom, final_left, final_right)
        horizontal_dividers = _sample_lines(
            rng, old_top, old_bottom, horizontal_count
        )
        vertical_dividers = _sample_lines(
            rng, old_left, old_right, vertical_count
        )
        input_grid = np.full(
            (height, width), background, dtype=np.int8
        )
        _draw_structure(
            input_grid, old_bounds, horizontal_dividers,
            vertical_dividers, structure,
        )

        dependencies: dict[str, str] = {}
        if force_double_delay:
            dependencies[pair[0]] = pair[1]
            dependencies[pair[1]] = pair[0]
        eligible = [
            side for side in sorted(expanded)
            if any(
                distances[orthogonal] >= 2
                for orthogonal in sorted(expanded)
                if (side in ("top", "bottom"))
                != (orthogonal in ("top", "bottom"))
            )
        ]
        if force_one_delay and not eligible:
            continue
        if force_one_delay:
            forced = eligible[int(rng.integers(0, len(eligible)))]
            orthogonals = [
                side for side in sorted(expanded)
                if distances[side] >= 2
                and (side in ("top", "bottom"))
                != (forced in ("top", "bottom"))
            ]
            dependencies[forced] = orthogonals[
                int(rng.integers(0, len(orthogonals)))
            ]
        for side in sorted(expanded):
            if side in dependencies or side not in eligible:
                continue
            if regime == "source_train" and index == 0:
                continue
            if (
                regime == "source_train"
                and not puzzle_parameters["train_delay"]
            ):
                continue
            probability = 0.25 if regime == "source_train" else 0.35
            if rng.random() < probability:
                orthogonals = [
                    other for other in sorted(expanded)
                    if distances[other] >= 2
                    and (side in ("top", "bottom"))
                    != (other in ("top", "bottom"))
                ]
                dependencies[side] = orthogonals[
                    int(rng.integers(0, len(orthogonals)))
                ]
        if not allow_multiple_delay and len(dependencies) > 1:
            retained = sorted(dependencies)[
                int(rng.integers(0, len(dependencies)))
            ]
            dependencies = {retained: dependencies[retained]}

        marker_positions = []
        for side in sorted(expanded):
            if side in ("left", "right"):
                if side in dependencies:
                    dependency = dependencies[side]
                    coordinates = (
                        range(final_top + 1, old_top)
                        if dependency == "top"
                        else range(old_bottom + 1, final_bottom)
                    )
                else:
                    coordinates = range(old_top + 1, old_bottom)
                coordinate = list(coordinates)
                if not coordinate:
                    break
                row = coordinate[int(rng.integers(0, len(coordinate)))]
                col = (
                    final_left - 1 if side == "left"
                    else final_right + 1
                )
            else:
                if side in dependencies:
                    dependency = dependencies[side]
                    coordinates = (
                        range(final_left + 1, old_left)
                        if dependency == "left"
                        else range(old_right + 1, final_right)
                    )
                else:
                    coordinates = range(old_left + 1, old_right)
                coordinate = list(coordinates)
                if not coordinate:
                    break
                col = coordinate[int(rng.integers(0, len(coordinate)))]
                row = (
                    final_top - 1 if side == "top"
                    else final_bottom + 1
                )
            marker_positions.append((row, col))
        if len(marker_positions) != len(expanded):
            continue
        if len(set(marker_positions)) != len(marker_positions):
            continue
        for row, col in marker_positions:
            if input_grid[row, col] != background:
                break
            input_grid[row, col] = marker
        else:
            output_grid = input_grid.copy()
            output_grid[input_grid == structure] = background
            _draw_structure(
                output_grid, final_bounds, horizontal_dividers,
                vertical_dividers, structure,
            )
            try:
                info = _analyze_pair(input_grid, output_grid)
            except AssertionError:
                continue
            if set(info["expanded"]) == expanded:
                return input_grid, output_grid

    raise ValueError("could not construct an unambiguous pair")


def _parse_scaffold(
    grid: np.ndarray,
    structure: int,
) -> tuple[
    tuple[int, int, int, int],
    list[int],
    list[int],
]:
    locations = np.argwhere(grid == structure)
    _check(len(locations) > 0, "missing structural scaffold")

    top = int(locations[:, 0].min())
    bottom = int(locations[:, 0].max())
    left = int(locations[:, 1].min())
    right = int(locations[:, 1].max())

    _check(
        bottom - top + 1 >= 3,
        "scaffold is too short",
    )
    _check(
        right - left + 1 >= 3,
        "scaffold is too narrow",
    )

    horizontal_dividers = [
        row
        for row in range(top + 1, bottom)
        if np.all(grid[row, left : right + 1] == structure)
    ]
    vertical_dividers = [
        col
        for col in range(left + 1, right)
        if np.all(grid[top : bottom + 1, col] == structure)
    ]

    for coordinates in (
        [top, *horizontal_dividers, bottom],
        [left, *vertical_dividers, right],
    ):
        _check(
            all(
                following - preceding >= 2
                for preceding, following in zip(
                    coordinates, coordinates[1:]
                )
            ),
            "structural lines leave no background compartment",
        )

    expected = np.zeros(grid.shape, dtype=bool)

    expected[top, left : right + 1] = True
    expected[bottom, left : right + 1] = True
    expected[top : bottom + 1, left] = True
    expected[top : bottom + 1, right] = True

    for row in horizontal_dividers:
        expected[row, left : right + 1] = True
    for col in vertical_dividers:
        expected[top : bottom + 1, col] = True

    _check(
        np.array_equal(expected, grid == structure),
        "structure is not exactly a framed scaffold with full dividers",
    )

    return (
        (top, bottom, left, right),
        horizontal_dividers,
        vertical_dividers,
    )


def _infer_marker_solution(
    markers: list[tuple[int, int]],
    old_bounds: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[
    tuple[int, int, int, int],
    dict[str, tuple[int, int]],
]:
    old_top, old_bottom, old_left, old_right = old_bounds

    solutions: list[
        tuple[
            tuple[int, int, int, int],
            tuple[str, ...],
        ]
    ] = []

    def visit(
        index: int,
        used: set[str],
        assigned: list[str],
    ) -> None:
        if index == len(markers):
            values = {
                side: markers[i]
                for i, side in enumerate(assigned)
            }

            final_top = (
                values["top"][0] + 1
                if "top" in values
                else old_top
            )
            final_right = (
                values["right"][1] - 1
                if "right" in values
                else old_right
            )
            final_bottom = (
                values["bottom"][0] - 1
                if "bottom" in values
                else old_bottom
            )
            final_left = (
                values["left"][1] + 1
                if "left" in values
                else old_left
            )

            bounds = (
                final_top,
                final_bottom,
                final_left,
                final_right,
            )

            if not (
                0 <= final_top <= old_top
                and old_bottom <= final_bottom < shape[0]
                and 0 <= final_left <= old_left
                and old_right <= final_right < shape[1]
            ):
                return

            for i, side in enumerate(assigned):
                row, col = markers[i]

                if side == "top":
                    valid = (
                        row == final_top - 1
                        and final_left <= col <= final_right
                    )
                elif side == "right":
                    valid = (
                        col == final_right + 1
                        and final_top <= row <= final_bottom
                    )
                elif side == "bottom":
                    valid = (
                        row == final_bottom + 1
                        and final_left <= col <= final_right
                    )
                else:
                    valid = (
                        col == final_left - 1
                        and final_top <= row <= final_bottom
                    )

                if not valid:
                    return

            solutions.append((bounds, tuple(assigned)))
            return

        row, col = markers[index]

        for side in _SIDES:
            if side in used:
                continue

            if side == "top" and not row + 1 < old_top:
                continue
            if side == "right" and not col - 1 > old_right:
                continue
            if side == "bottom" and not row - 1 > old_bottom:
                continue
            if side == "left" and not col + 1 < old_left:
                continue

            used.add(side)
            assigned.append(side)
            visit(index + 1, used, assigned)
            assigned.pop()
            used.remove(side)

    visit(0, set(), [])

    _check(
        len(solutions) == 1,
        "marker-to-side assignment is not unique",
    )

    final_bounds, assigned = solutions[0]
    marker_by_side = {
        side: markers[i]
        for i, side in enumerate(assigned)
    }

    return final_bounds, marker_by_side


def _analyze_with_roles(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    background: int,
    structure: int,
    marker: int,
) -> dict:
    _check(
        input_grid.shape == output_grid.shape,
        "input/output dimensions changed",
    )

    _check(
        set(int(x) for x in np.unique(input_grid))
        == {background, structure, marker},
        "input does not use exactly the three semantic colors",
    )
    _check(
        set(int(x) for x in np.unique(output_grid))
        == {background, structure, marker},
        "output does not preserve the three semantic colors",
    )

    (
        old_bounds,
        horizontal_dividers,
        vertical_dividers,
    ) = _parse_scaffold(input_grid, structure)

    marker_array = np.argwhere(input_grid == marker)
    _check(
        2 <= len(marker_array) <= 3,
        "there must be two or three markers",
    )

    markers = sorted(
        (int(row), int(col))
        for row, col in marker_array
    )

    height, width = input_grid.shape

    for row, col in markers:
        for neighbor_row in range(
            max(0, row - 1),
            min(height, row + 2),
        ):
            for neighbor_col in range(
                max(0, col - 1),
                min(width, col + 2),
            ):
                if (
                    neighbor_row == row
                    and neighbor_col == col
                ):
                    continue

                _check(
                    int(
                        input_grid[
                            neighbor_row,
                            neighbor_col,
                        ]
                    )
                    not in (structure, marker),
                    "markers must be isolated from markers and structure",
                )

    final_bounds, marker_by_side = _infer_marker_solution(
        markers,
        old_bounds,
        input_grid.shape,
    )

    expected = input_grid.copy()
    expected[input_grid == structure] = background

    _draw_structure(
        expected,
        final_bounds,
        horizontal_dividers,
        vertical_dividers,
        structure,
    )

    _check(
        np.array_equal(expected, output_grid),
        "incorrect transformed output",
    )
    _check(
        not np.array_equal(input_grid, output_grid),
        "degenerate unchanged pair",
    )

    return {
        "roles": (background, structure, marker),
        "shape": tuple(int(x) for x in input_grid.shape),
        "old_bounds": old_bounds,
        "final_bounds": final_bounds,
        "horizontal_dividers": tuple(horizontal_dividers),
        "vertical_dividers": tuple(vertical_dividers),
        "marker_by_side": marker_by_side,
        "expanded": frozenset(marker_by_side),
    }


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    input_colors = [
        int(x)
        for x in np.unique(input_grid)
    ]
    output_colors = [
        int(x)
        for x in np.unique(output_grid)
    ]

    _check(
        len(input_colors) == 3,
        "input must contain exactly three colors",
    )
    _check(
        set(input_colors) == set(output_colors),
        "input/output color sets differ",
    )

    successful: list[dict] = []

    for background in input_colors:
        for structure in input_colors:
            if structure == background:
                continue

            remaining = [
                color
                for color in input_colors
                if color not in (background, structure)
            ]
            marker = remaining[0]

            try:
                result = _analyze_with_roles(
                    input_grid,
                    output_grid,
                    background,
                    structure,
                    marker,
                )
            except AssertionError:
                continue

            successful.append(result)

    _check(
        len(successful) == 1,
        "semantic color roles are not uniquely recoverable",
    )

    return successful[0]


def _marker_aligned_with_old(
    side: str,
    marker: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> bool:
    top, bottom, left, right = bounds
    row, col = marker

    if side in ("left", "right"):
        return top <= row <= bottom
    return left <= col <= right


def _validate_regimes(
    train_infos: list[dict],
    test_infos: list[dict],
) -> None:
    _check(bool(train_infos) and bool(test_infos), "empty puzzle split")
    first = train_infos[0]
    _check(
        len(first["expanded"]) == 2
        and not first["horizontal_dividers"]
        and not first["vertical_dividers"]
        and all(
            _marker_aligned_with_old(
                side, marker, first["old_bounds"]
            )
            for side, marker in first["marker_by_side"].items()
        ),
        "first training pair must demonstrate plain expansion",
    )

    if len(train_infos) == 2:
        divider_example = train_infos[1]
    else:
        divider_example = train_infos[2]
        second = train_infos[1]
        _check(
            len(second["expanded"]) == 2
            and (
                bool(second["horizontal_dividers"])
                != bool(second["vertical_dividers"])
            ),
            "second training pair must show one divider orientation",
        )
    _check(
        len(divider_example["expanded"]) == 3
        and bool(divider_example["horizontal_dividers"])
        and bool(divider_example["vertical_dividers"]),
        "training lacks a mixed-divider three-side example",
    )

    demonstrated_sides = set().union(
        *(info["expanded"] for info in train_infos)
    )
    demonstrated_palettes = {
        info["roles"] for info in train_infos
    }
    training_has_delay = any(
        not _marker_aligned_with_old(
            side, marker, info["old_bounds"]
        )
        for info in train_infos
        for side, marker in info["marker_by_side"].items()
    )

    for info in train_infos + test_infos:
        _check(
            2 <= len(info["expanded"]) <= 3,
            "pairs require two or three distinct marked sides",
        )
    for info in test_infos:
        _check(
            set(info["expanded"]) <= demonstrated_sides,
            "test uses a side never expanded in training",
        )
        _check(
            info["roles"] in demonstrated_palettes,
            "test palette was not demonstrated in training",
        )
        delayed_count = sum(
            not _marker_aligned_with_old(
                side, marker, info["old_bounds"]
            )
            for side, marker in info["marker_by_side"].items()
        )
        _check(
            delayed_count < 2 or training_has_delay,
            "two delayed test markers need training precedent",
        )

def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.uint8)
    return (
        tuple(int(x) for x in grid.shape),
        normalized.tobytes(),
    )


def _grid_text(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


_SOURCE_PAIR_KEYS = {
    (
        "000000000000000/000000000000000/000033300000000/"
        "060030300000000/000033300000000/000000000000000/"
        "000000000000000/000000000000000/000000000000000/"
        "000006000000000/000000000000000",
        "000000000000000/000000000000000/003333300000000/"
        "063000300000000/003000300000000/003000300000000/"
        "003000300000000/003000300000000/003333300000000/"
        "000006000000000/000000000000000",
    ),
    (
        "00000000000000000/03333330000000000/"
        "03030030000600000/03030030000000000/"
        "03333330000000000/00000000000000000/"
        "00006000000000000/00000000000000000",
        "00000000000000000/03333333333000000/"
        "03030000003600000/03030000003000000/"
        "03030000003000000/03333333333000000/"
        "00006000000000000/00000000000000000",
    ),
    (
        "000000000000000/000000000000000/000000000000000/"
        "000033333300000/000030030300600/000033333300000/"
        "060030030300000/000033333300000/000000000000000/"
        "000000000000000/000000000000000/000000000000000/"
        "000000000000000/000000000000000/000000060000000",
        "000000000000000/000000000000000/000000000000000/"
        "003333333333000/003000030003600/003333333333000/"
        "063000030003000/003000030003000/003000030003000/"
        "003000030003000/003000030003000/003000030003000/"
        "003000030003000/003333333333000/000000060000000",
    ),
    (
        "00000000000000000/00000000000000000/"
        "00000333000000000/00000303000000000/"
        "06000303000000000/00000333000000000/"
        "00000303000000000/00000333000000000/"
        "00000000000000000/00000000000000000/"
        "00000000000000000/00000000000006000/"
        "00000060000000000/00000000000000000/"
        "00000000000000000/00000000000000000",
        "00000000000000000/00000000000000000/"
        "00333333333330000/00300000000030000/"
        "06300000000030000/00333333333330000/"
        "00300000000030000/00300000000030000/"
        "00300000000030000/00300000000030000/"
        "00300000000030000/00333333333336000/"
        "00000060000000000/00000000000000000/"
        "00000000000000000/00000000000000000",
    ),
}


def _is_exact_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    key = (
        _grid_text(input_grid),
        _grid_text(output_grid),
    )
    return key in _SOURCE_PAIR_KEYS


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(
        isinstance(puzzle, dict),
        "puzzle must be a dictionary",
    )
    _check(
        set(puzzle.keys()) == {"train", "test"},
        "wrong puzzle keys",
    )
    _check(
        isinstance(puzzle["train"], list),
        "train must be a list",
    )
    _check(
        isinstance(puzzle["test"], list),
        "test must be a list",
    )

    _check(
        2 <= len(puzzle["train"]) <= 10,
        "invalid train count",
    )
    _check(
        1 <= len(puzzle["test"]) <= 4,
        "invalid test count",
    )
    _check(
        len(puzzle["train"]) > len(puzzle["test"]),
        "train must exceed test",
    )

    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    infos: dict[str, list[dict]] = {
        "train": [],
        "test": [],
    }

    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(
                isinstance(pair, dict),
                "pair must be a dictionary",
            )
            _check(
                set(pair.keys()) == {"input", "output"},
                "wrong pair keys",
            )

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                _check(
                    isinstance(grid, np.ndarray),
                    "grid must be a NumPy array",
                )
                _check(
                    grid.ndim == 2,
                    "grid must be two-dimensional",
                )
                _check(
                    1 <= grid.shape[0] <= 30
                    and 1 <= grid.shape[1] <= 30,
                    "grid dimensions out of range",
                )
                _check(
                    np.issubdtype(grid.dtype, np.integer),
                    "grid dtype is not integer",
                )
                _check(
                    bool(np.all((grid >= 0) & (grid <= 9))),
                    "grid color outside 0 through 9",
                )

            input_key = _grid_key(input_grid)
            pair_key = (
                input_key,
                _grid_key(output_grid),
            )

            _check(
                input_key not in seen_inputs,
                "duplicate input grid",
            )
            _check(
                pair_key not in seen_pairs,
                "duplicate input/output pair",
            )

            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            _check(
                not _is_exact_source_pair(
                    input_grid,
                    output_grid,
                ),
                "generated pair exactly duplicates a source pair",
            )

            infos[split].append(
                _analyze_pair(input_grid, output_grid)
            )

    _validate_regimes(
        infos["train"],
        infos["test"],
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _validate_seed(seed)
    num_train, num_test = _resolve_counts(
        seed,
        num_train,
        num_test,
    )

    rng = np.random.default_rng(seed)
    puzzle_parameters = _sample_puzzle_parameters(rng)

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs: set[tuple] = set()
    training_has_delay = False
    anchor = puzzle_parameters["fixed_anchor"]
    train_allowed = set(_SIDES) - ({anchor} if anchor else set())
    trained_sides: set[str] = set()

    for split, regime, count in (
        ("train", "source_train", num_train),
        ("test", "target_test", num_test),
    ):
        for index in range(count):
            if puzzle_parameters["switch_palette"]:
                palette_index = (
                    index if split == "train" and index < 2
                    else int(rng.integers(0, 2))
                )
            else:
                palette_index = 0
            palette = puzzle_parameters["palettes"][palette_index]
            allowed_sides = (
                train_allowed if split == "train" else trained_sides
            )
            pair_seed = int(
                rng.integers(
                    0,
                    np.iinfo(np.int64).max,
                )
            )
            pair_rng = np.random.default_rng(pair_seed)

            for _ in range(100):
                input_grid, output_grid = (
                    _generate_canonical_pair(
                        pair_rng,
                        regime,
                        index,
                        count,
                        puzzle_parameters,
                        allowed_sides,
                        palette,
                        split == "test"
                        and index == 0
                        and puzzle_parameters["train_delay"]
                        and puzzle_parameters["double_delay_test"],
                        split == "train" or training_has_delay,
                    )
                )

                input_key = _grid_key(input_grid)

                if input_key in seen_inputs:
                    continue
                if _is_exact_source_pair(
                    input_grid,
                    output_grid,
                ):
                    continue

                seen_inputs.add(input_key)
                if split == "train":
                    info = _analyze_pair(input_grid, output_grid)
                    trained_sides.update(info["expanded"])
                    training_has_delay |= any(
                        not _marker_aligned_with_old(
                            side, position, info["old_bounds"]
                        )
                        for side, position in info["marker_by_side"].items()
                    )
                puzzle[split].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                break
            else:
                raise ValueError(
                    "could not generate enough distinct valid pairs"
                )

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            tuple(
                (
                    _grid_key(pair["input"]),
                    _grid_key(pair["output"]),
                )
                for pair in puzzle[split]
            ),
        )
        for split in ("train", "test")
    )


def _predict_from_training(
    train_pairs: list[dict],
    test_input: np.ndarray,
) -> np.ndarray:
    palettes = {
        _analyze_pair(pair["input"], pair["output"])["roles"]
        for pair in train_pairs
    }
    candidates = [
        roles for roles in palettes
        if set(roles)
        == set(int(color) for color in np.unique(test_input))
    ]
    _check(
        len(candidates) == 1,
        "training does not uniquely identify test color roles",
    )
    background, structure, marker = candidates[0]
    old_bounds, horizontal, vertical = _parse_scaffold(
        test_input, structure
    )
    markers = [
        (int(row), int(col))
        for row, col in np.argwhere(test_input == marker)
    ]
    final_bounds, _ = _infer_marker_solution(
        markers, old_bounds, test_input.shape
    )
    prediction = test_input.copy()
    prediction[test_input == structure] = background
    _draw_structure(
        prediction, final_bounds, horizontal, vertical, structure
    )
    return prediction


def self_test() -> None:
    automatic = generate_puzzle(314159)
    _check(
        _puzzle_key(automatic) == _puzzle_key(generate_puzzle(314159)),
        "automatic call is not deterministic",
    )
    explicit = generate_puzzle(271828, 7, 3)
    _check(
        _puzzle_key(explicit)
        == _puzzle_key(generate_puzzle(271828, 7, 3)),
        "explicit call is not deterministic",
    )
    _check(
        (len(explicit["train"]), len(explicit["test"])) == (7, 3),
        "explicit counts were not honored",
    )

    for offset, (train_count, test_count) in enumerate(
        ((2, 1), (10, 1), (10, 4))
    ):
        puzzle = generate_puzzle(5000 + offset, train_count, test_count)
        _check(
            (len(puzzle["train"]), len(puzzle["test"]))
            == (train_count, test_count),
            "boundary counts were not honored",
        )
    for seed, train_count, test_count in (
        (6100, None, 1),
        (6101, None, 4),
        (6102, 2, None),
        (6103, 10, None),
    ):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            _check(len(puzzle["train"]) == train_count, "partial train")
        if test_count is not None:
            _check(len(puzzle["test"]) == test_count, "partial test")
    for arguments in (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 2, "num_test": 3},
        {"num_train": 4, "num_test": 4},
    ):
        try:
            generate_puzzle(7000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")

    train_counts: set[int] = set()
    test_counts: set[int] = set()
    roles: set[tuple] = set()
    shapes: set[tuple] = set()
    side_sets: set[tuple] = set()
    signatures: set[tuple] = set()
    fingerprints: set[tuple] = set()
    measures = {
        name: 0
        for name in (
            "palette_switch", "side_change", "no_fixed_side",
            "train_delay", "test_direct", "test_one_delay",
            "test_multi_delay", "train_one_move", "train_large_move",
            "train_small_frame", "test_small_frame", "train_edge",
            "test_edge", "divider_alignment", "many_dividers",
            "test_no_divider", "test_mixed_dividers",
            "test_compact",
        )
    }
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        infos = {
            split: [
                _analyze_pair(pair["input"], pair["output"])
                for pair in puzzle[split]
            ]
            for split in ("train", "test")
        }
        all_infos = infos["train"] + infos["test"]
        measures["palette_switch"] += (
            len({info["roles"] for info in infos["train"]}) > 1
        )
        measures["side_change"] += (
            len({
                tuple(sorted(info["expanded"]))
                for info in all_infos
            }) > 1
        )
        measures["no_fixed_side"] += not set.intersection(
            *(set(_SIDES) - set(info["expanded"]) for info in all_infos)
        )
        measures["train_delay"] += any(
            not _marker_aligned_with_old(
                side, marker, info["old_bounds"]
            )
            for info in infos["train"]
            for side, marker in info["marker_by_side"].items()
        )
        max_train_area = max(
            info["shape"][0] * info["shape"][1]
            for info in infos["train"]
        )
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], infos[split]):
                old = info["old_bounds"]
                final = info["final_bounds"]
                height, width = info["shape"]
                roles.add(info["roles"])
                shapes.add(info["shape"])
                side_sets.add(tuple(sorted(info["expanded"])))
                delays = sum(
                    not _marker_aligned_with_old(side, marker, old)
                    for side, marker in info["marker_by_side"].items()
                )
                moves = (
                    old[0] - final[0], final[3] - old[3],
                    final[1] - old[1], old[2] - final[2],
                )
                signatures.add((
                    info["shape"], old, final,
                    len(info["horizontal_dividers"]),
                    len(info["vertical_dividers"]),
                    tuple(sorted(info["expanded"])), delays,
                ))
                prefix = "train" if split == "train" else "test"
                measures[prefix + "_small_frame"] += (
                    old[1] - old[0] + 1 == 3
                    or old[3] - old[2] + 1 == 3
                )
                measures[prefix + "_edge"] += (
                    final[0] == 0 or final[2] == 0
                    or final[1] == height - 1
                    or final[3] == width - 1
                )
                measures["divider_alignment"] += any(
                    (
                        side in ("left", "right")
                        and marker[0] in info["horizontal_dividers"]
                    ) or (
                        side in ("top", "bottom")
                        and marker[1] in info["vertical_dividers"]
                    )
                    for side, marker in info["marker_by_side"].items()
                )
                measures["many_dividers"] += (
                    len(info["horizontal_dividers"]) >= 3
                    or len(info["vertical_dividers"]) >= 3
                )
                if split == "train":
                    measures["train_one_move"] += 1 in moves
                    measures["train_large_move"] += any(
                        distance >= 6 for distance in moves
                    )
                else:
                    _check(
                        np.array_equal(
                            _predict_from_training(
                                puzzle["train"], pair["input"].copy()
                            ),
                            pair["output"],
                        ),
                        "test output not inferable from training",
                    )
                    measures["test_direct"] += delays == 0
                    measures["test_one_delay"] += delays == 1
                    measures["test_multi_delay"] += delays >= 2
                    measures["test_no_divider"] += (
                        not info["horizontal_dividers"]
                        and not info["vertical_dividers"]
                    )
                    measures["test_mixed_dividers"] += (
                        bool(info["horizontal_dividers"])
                        and bool(info["vertical_dividers"])
                    )
                    measures["test_compact"] += (
                        height * width <= max_train_area
                    )

    for condition, message in (
        (len(train_counts) >= 4, "train counts lack diversity"),
        (len(test_counts) >= 3, "test counts lack diversity"),
        (len(fingerprints) >= 40, "puzzles repeat across seeds"),
        (len(roles) >= 15, "color roles lack diversity"),
        (len(shapes) >= 35, "canvas shapes lack diversity"),
        (len(side_sets) >= 8, "marked side sets lack diversity"),
        (len(signatures) >= 250, "layouts lack diversity"),
        (measures["palette_switch"] >= 6, "palette switching absent"),
        (measures["side_change"] >= 30, "within-puzzle side sets fixed"),
        (measures["no_fixed_side"] >= 8, "unmarked side always fixed"),
        (measures["train_delay"] >= 15, "training delays absent"),
        (measures["test_direct"] >= 12, "direct-marker tests absent"),
        (measures["test_one_delay"] >= 10, "one-delay tests absent"),
        (measures["test_multi_delay"] >= 8, "multi-delay tests absent"),
        (measures["train_one_move"] >= 20, "one-cell moves absent"),
        (measures["train_large_move"] >= 35, "large moves absent"),
        (measures["train_small_frame"] >= 8, "small training frames absent"),
        (measures["test_small_frame"] >= 4, "small test frames absent"),
        (measures["train_edge"] >= 20, "edge-touching train frames absent"),
        (measures["test_edge"] >= 8, "edge-touching test frames absent"),
        (measures["divider_alignment"] >= 15, "marker-divider alignment absent"),
        (measures["many_dividers"] >= 20, "several dividers absent"),
        (measures["test_no_divider"] >= 12, "divider-free tests absent"),
        (measures["test_mixed_dividers"] >= 5, "mixed-divider tests absent"),
        (measures["test_compact"] >= 15, "compact tests absent"),
    ):
        _check(condition, message)
