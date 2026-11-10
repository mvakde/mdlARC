import hashlib

import numpy as np


_SOURCE_HASHES = {
    "8e8a4a14a8a4ad6468f18a30ab8b0d01bb8283adca4ede05ec902c16680548e3",
    "75f6ebf9a5a7fb7122c308b6b8366f968a9cd897840737e1876674e3d5a89f48",
    "db48d806831b7a322999492d420e0aa41147037e077a232c959952fc1e6c3d12",
    "3d228773a8d091ea7cabb39dd4851e36bae2b073566ad37220dc378ff141c13a",
}


def _is_integer_count(value) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    if num_train is not None:
        if not _is_integer_count(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)

    if num_test is not None:
        if not _is_integer_count(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)

    if num_train is None and num_test is None:
        num_train = int(rng.integers(2, 11))
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        if minimum_train > 10:
            raise ValueError("no legal num_train is greater than num_test")
        num_train = int(rng.integers(minimum_train, 11))

    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test is less than num_train")
        num_test = int(rng.integers(1, maximum_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _connected_eight_neighbor(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    start = tuple(int(x) for x in points[0])
    seen = {start}
    stack = [start]
    height, width = mask.shape

    while stack:
        row, col = stack.pop()
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                new_row = row + drow
                new_col = col + dcol
                neighbor = (new_row, new_col)
                if (
                    0 <= new_row < height
                    and 0 <= new_col < width
                    and mask[new_row, new_col]
                    and neighbor not in seen
                ):
                    seen.add(neighbor)
                    stack.append(neighbor)

    return len(seen) == int(mask.sum())


def _d4_matrices(matrix: np.ndarray) -> list[np.ndarray]:
    results: list[np.ndarray] = []

    for turns in range(4):
        rotated = np.rot90(matrix, turns)
        for candidate in (rotated, np.fliplr(rotated)):
            candidate = np.asarray(candidate)
            if not any(np.array_equal(candidate, old) for old in results):
                results.append(candidate.copy())

    return results


def _role_signature(
    prototype: np.ndarray,
    background: int,
    connector: int,
    endpoint_a: int,
    endpoint_b: int,
) -> tuple[int, ...]:
    role_grid = np.empty((3, 3), dtype=np.int8)
    role_grid[prototype == background] = 0
    role_grid[prototype == connector] = 1
    role_grid[prototype == endpoint_a] = 2
    role_grid[prototype == endpoint_b] = 3

    return min(
        tuple(int(value) for value in transformed.ravel())
        for transformed in _d4_matrices(role_grid)
    )


def _build_motif_catalog() -> tuple[np.ndarray, ...]:
    catalog: list[np.ndarray] = []
    signatures: set[tuple[int, ...]] = set()

    symmetric_orbits = (
        ((0, 1), (1, 0)),
        ((0, 2), (2, 0)),
        ((1, 2), (2, 1)),
        ((1, 1),),
    )

    for bits in range(16):
        connector_mask = np.zeros((3, 3), dtype=bool)

        for orbit_index, cells in enumerate(symmetric_orbits):
            if (bits >> orbit_index) & 1:
                for row, col in cells:
                    connector_mask[row, col] = True

        connector_count = int(connector_mask.sum())
        active = connector_mask.copy()
        active[0, 0] = True
        active[2, 2] = True

        if not 1 <= connector_count <= 7:
            continue
        if not _connected_eight_neighbor(active):
            continue

        role_grid = np.zeros((3, 3), dtype=np.int8)
        role_grid[connector_mask] = 1
        role_grid[0, 0] = 2
        role_grid[2, 2] = 3

        signature = min(
            tuple(int(value) for value in transformed.ravel())
            for transformed in _d4_matrices(role_grid)
        )

        if signature not in signatures:
            signatures.add(signature)
            catalog.append(connector_mask.copy())

    if len(catalog) < 3:
        raise AssertionError("motif catalog is unexpectedly small")

    return tuple(catalog)


_MOTIF_CATALOG = _build_motif_catalog()


def _four_neighbor_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros_like(mask)
    components: list[list[tuple[int, int]]] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue

            visited[row, col] = True
            stack = [(row, col)]
            cells: list[tuple[int, int]] = []

            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))

                for drow, dcol in (
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                ):
                    new_row = current_row + drow
                    new_col = current_col + dcol

                    if (
                        0 <= new_row < height
                        and 0 <= new_col < width
                        and mask[new_row, new_col]
                        and not visited[new_row, new_col]
                    ):
                        visited[new_row, new_col] = True
                        stack.append((new_row, new_col))

            components.append(cells)

    return components


def _find_prototype(grid: np.ndarray) -> dict:
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 4

    maximum_count = int(counts.max())
    assert int(np.count_nonzero(counts == maximum_count)) == 1

    background = int(values[int(np.argmax(counts))])
    nonbackground_colors = sorted(
        int(value) for value in values if int(value) != background
    )

    height, width = grid.shape
    candidates = []

    for connector in nonbackground_colors:
        endpoints = [
            color
            for color in nonbackground_colors
            if color != connector
        ]
        endpoint_a, endpoint_b = endpoints

        total_connector = int(np.count_nonzero(grid == connector))
        if not 1 <= total_connector <= 7:
            continue

        if int(np.count_nonzero(grid == endpoint_a)) <= 1:
            continue
        if int(np.count_nonzero(grid == endpoint_b)) <= 1:
            continue

        for top in range(height - 2):
            for left in range(width - 2):
                subgrid = grid[top:top + 3, left:left + 3]

                if int(np.count_nonzero(subgrid == connector)) != total_connector:
                    continue
                if int(np.count_nonzero(subgrid == endpoint_a)) != 1:
                    continue
                if int(np.count_nonzero(subgrid == endpoint_b)) != 1:
                    continue

                position_a = tuple(
                    int(x) for x in np.argwhere(subgrid == endpoint_a)[0]
                )
                position_b = tuple(
                    int(x) for x in np.argwhere(subgrid == endpoint_b)[0]
                )

                if position_a not in (
                    (0, 0),
                    (0, 2),
                    (2, 0),
                    (2, 2),
                ):
                    continue

                if position_b != (
                    2 - position_a[0],
                    2 - position_a[1],
                ):
                    continue

                if not _connected_eight_neighbor(subgrid != background):
                    continue

                compatible = [
                    transformed
                    for transformed in _d4_matrices(subgrid)
                    if int(transformed[0, 0]) == endpoint_a
                    and int(transformed[2, 2]) == endpoint_b
                ]

                compatible_masks = {
                    tuple(
                        bool(value)
                        for value in (transformed == connector).ravel()
                    )
                    for transformed in compatible
                }

                if len(compatible) == 0 or len(compatible_masks) != 1:
                    continue

                candidates.append(
                    (
                        top,
                        left,
                        connector,
                        endpoint_a,
                        endpoint_b,
                        subgrid.copy(),
                    )
                )

    assert len(candidates) == 1

    (
        top,
        left,
        connector,
        endpoint_a,
        endpoint_b,
        prototype,
    ) = candidates[0]

    return {
        "background": background,
        "connector": connector,
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "top": top,
        "left": left,
        "prototype": prototype,
        "role_signature": _role_signature(
            prototype,
            background,
            connector,
            endpoint_a,
            endpoint_b,
        ),
    }


def _solid_endpoint_squares(
    grid: np.ndarray,
    color: int,
    prototype_top: int,
    prototype_left: int,
) -> list[tuple[int, int, int]]:
    mask = (grid == color).copy()
    mask[
        prototype_top:prototype_top + 3,
        prototype_left:prototype_left + 3,
    ] = False

    squares: list[tuple[int, int, int]] = []

    for cells in _four_neighbor_components(mask):
        rows = [row for row, _ in cells]
        cols = [col for _, col in cells]

        top = min(rows)
        bottom = max(rows)
        left = min(cols)
        right = max(cols)
        side = bottom - top + 1

        assert side == right - left + 1
        assert len(cells) == side * side
        assert np.all(mask[top:bottom + 1, left:right + 1])
        assert 1 <= side <= 9

        squares.append((top, left, side))

    squares.sort()
    return squares


def _corner_index(row_index: int, col_index: int) -> int:
    mapping = {
        (0, 0): 0,
        (0, 2): 1,
        (2, 2): 2,
        (2, 0): 3,
    }
    assert (row_index, col_index) in mapping
    return mapping[(row_index, col_index)]


def _analyze_input(grid: np.ndarray) -> dict:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2

    prototype_info = _find_prototype(grid)

    background = prototype_info["background"]
    connector = prototype_info["connector"]
    endpoint_a = prototype_info["endpoint_a"]
    endpoint_b = prototype_info["endpoint_b"]
    prototype_top = prototype_info["top"]
    prototype_left = prototype_info["left"]
    prototype = prototype_info["prototype"]

    prototype_a_position = tuple(
        int(x) for x in np.argwhere(prototype == endpoint_a)[0]
    )
    prototype_a_corner = _corner_index(*prototype_a_position)

    squares_a = _solid_endpoint_squares(
        grid,
        endpoint_a,
        prototype_top,
        prototype_left,
    )
    squares_b = _solid_endpoint_squares(
        grid,
        endpoint_b,
        prototype_top,
        prototype_left,
    )

    assert len(squares_a) == len(squares_b)
    assert len(squares_a) >= 1

    adjacency_a: list[list[int]] = [[] for _ in squares_a]
    adjacency_b: list[list[int]] = [[] for _ in squares_b]

    for index_a, (row_a, col_a, side_a) in enumerate(squares_a):
        for index_b, (row_b, col_b, side_b) in enumerate(squares_b):
            if (
                side_a == side_b
                and abs(row_a - row_b) == 2 * side_a
                and abs(col_a - col_b) == 2 * side_a
            ):
                adjacency_a[index_a].append(index_b)
                adjacency_b[index_b].append(index_a)

    assert all(len(neighbors) == 1 for neighbors in adjacency_a)
    assert all(len(neighbors) == 1 for neighbors in adjacency_b)
    assert len({neighbors[0] for neighbors in adjacency_a}) == len(squares_b)

    expected = grid.copy()
    occupied = np.zeros(grid.shape, dtype=bool)
    occupied[
        prototype_top:prototype_top + 3,
        prototype_left:prototype_left + 3,
    ] = True

    transformed_prototypes = _d4_matrices(prototype)
    copies = []

    for index_a, (row_a, col_a, side) in enumerate(squares_a):
        index_b = adjacency_a[index_a][0]
        row_b, col_b, side_b = squares_b[index_b]
        assert side == side_b

        top = min(row_a, row_b)
        left = min(col_a, col_b)
        extent = 3 * side

        assert top + extent <= grid.shape[0]
        assert left + extent <= grid.shape[1]
        assert not np.any(
            occupied[top:top + extent, left:left + extent]
        )
        occupied[top:top + extent, left:left + extent] = True

        endpoint_a_row = (row_a - top) // side
        endpoint_a_col = (col_a - left) // side
        endpoint_b_row = (row_b - top) // side
        endpoint_b_col = (col_b - left) // side

        assert (endpoint_a_row, endpoint_a_col) in (
            (0, 0),
            (0, 2),
            (2, 0),
            (2, 2),
        )
        assert (endpoint_b_row, endpoint_b_col) == (
            2 - endpoint_a_row,
            2 - endpoint_a_col,
        )

        compatible_masks = {
            tuple(
                bool(value)
                for value in (transformed == connector).ravel()
            )
            for transformed in transformed_prototypes
            if int(
                transformed[endpoint_a_row, endpoint_a_col]
            ) == endpoint_a
            and int(
                transformed[endpoint_b_row, endpoint_b_col]
            ) == endpoint_b
        }

        assert len(compatible_masks) == 1

        connector_mask = np.array(
            next(iter(compatible_masks)),
            dtype=bool,
        ).reshape(3, 3)

        for mask_row in range(3):
            for mask_col in range(3):
                if not connector_mask[mask_row, mask_col]:
                    continue

                block = expected[
                    top + mask_row * side:
                    top + (mask_row + 1) * side,
                    left + mask_col * side:
                    left + (mask_col + 1) * side,
                ]
                assert np.all(block == background)
                block[:] = connector

        absolute_corner = _corner_index(
            endpoint_a_row,
            endpoint_a_col,
        )
        relative_orientation = (
            absolute_corner - prototype_a_corner
        ) % 4

        copies.append(
            {
                "top": top,
                "left": left,
                "scale": side,
                "orientation": relative_orientation,
                "connector_mask": tuple(
                    bool(value) for value in connector_mask.ravel()
                ),
            }
        )

    prototype_nonbackground = int(
        np.count_nonzero(prototype != background)
    )
    expected_input_nonbackground = (
        prototype_nonbackground
        + sum(
            2 * copy["scale"] * copy["scale"]
            for copy in copies
        )
    )
    assert (
        int(np.count_nonzero(grid != background))
        == expected_input_nonbackground
    )

    connector_count = int(
        np.count_nonzero(prototype == connector)
    )
    expected_change_count = connector_count * sum(
        copy["scale"] ** 2 for copy in copies
    )
    assert (
        int(np.count_nonzero(expected != grid))
        == expected_change_count
    )

    copies.sort(key=lambda item: (item["top"], item["left"]))

    return {
        **prototype_info,
        "expected": expected,
        "copies": copies,
        "scales": tuple(
            sorted(copy["scale"] for copy in copies)
        ),
        "orientations": tuple(
            copy["orientation"] for copy in copies
        ),
    }


def _rectangles_separated(
    first: tuple[int, int, int],
    second: tuple[int, int, int],
    gap: int = 0,
) -> bool:
    row_1, col_1, side_1 = first
    row_2, col_2, side_2 = second
    return (
        row_1 + side_1 + gap <= row_2
        or row_2 + side_2 + gap <= row_1
        or col_1 + side_1 + gap <= col_2
        or col_2 + side_2 + gap <= col_1
    )


def _place_rectangles(
    rng: np.random.Generator,
    height: int,
    width: int,
    sizes: list[int],
    gap: int,
) -> list[tuple[int, int]] | None:
    order = sorted(
        range(len(sizes)),
        key=lambda index: (-sizes[index], index),
    )

    for _ in range(12):
        placed: dict[int, tuple[int, int, int]] = {}
        failed = False

        for index in order:
            side = sizes[index]
            candidates = []

            for top in range(height - side + 1):
                for left in range(width - side + 1):
                    candidate = (top, left, side)
                    if all(
                        _rectangles_separated(candidate, old, gap)
                        for old in placed.values()
                    ):
                        candidates.append(candidate)

            if not candidates:
                failed = True
                break

            if gap == 0 and placed and rng.random() < 0.65:
                touching = [candidate for candidate in candidates
                            if any(_boxes_touch(candidate, old)
                                   for old in placed.values())]
                if touching:
                    candidates = touching
            placed[index] = candidates[int(rng.integers(len(candidates)))]

        if not failed:
            return [
                (placed[index][0], placed[index][1])
                for index in range(len(sizes))
            ]

    return None


def _boxes_touch(first: tuple[int, int, int],
                 second: tuple[int, int, int]) -> bool:
    r, c, s = first
    q, d, t = second
    vertical = (r + s == q or q + t == r) and max(c, d) < min(c + s, d + t)
    horizontal = (c + s == d or d + t == c) and max(r, q) < min(r + s, q + t)
    return vertical or horizontal


def _draw_role_assignment(
    rng: np.random.Generator,
    palette: np.ndarray,
    previous: tuple[int, int, int, int] | None,
    continuity: float,
) -> tuple[int, int, int, int]:
    if previous is not None and rng.random() < continuity:
        if rng.random() < 0.5:
            fixed = previous[:2]
        else:
            fixed = previous[:1]
        remaining = [int(v) for v in palette if int(v) not in fixed]
        rng.shuffle(remaining)
        return tuple(fixed) + tuple(remaining[:4 - len(fixed)])
    selection = rng.choice(palette, size=4, replace=False)
    return tuple(int(v) for v in selection)


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    scales: tuple[int, ...],
    orientations: tuple[int, ...],
    colors: tuple[int, int, int, int],
    motif_mask: np.ndarray,
    gap: int,
    size_bias: int,
    minimum_dimension: int,
    maximum_dimension: int,
    preferred_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    assert regime in ("source_train", "target_test")
    assert len(scales) == len(orientations)

    background, connector, endpoint_a, endpoint_b = (
        int(value) for value in colors
    )

    canonical = np.full(
        (3, 3),
        background,
        dtype=np.int8,
    )
    canonical[motif_mask] = connector
    canonical[0, 0] = endpoint_a
    canonical[2, 2] = endpoint_b

    expected_signature = _role_signature(
        canonical,
        background,
        connector,
        min(endpoint_a, endpoint_b),
        max(endpoint_a, endpoint_b),
    )

    rectangle_sizes = [3] + [
        3 * int(scale) for scale in scales
    ]
    area = sum(side * side for side in rectangle_sizes)
    dimension_low = min(maximum_dimension, max(
        max(rectangle_sizes),
        min(26, int(np.ceil(np.sqrt(area * 1.25))) + gap + size_bias),
        minimum_dimension,
    ))

    for attempt in range(180):
        if preferred_shape is not None and attempt < 50:
            height, width = preferred_shape
            current_gap = gap if attempt < 20 else 0
        elif attempt < 130:
            height = int(rng.integers(dimension_low, maximum_dimension + 1))
            width = int(rng.integers(dimension_low, maximum_dimension + 1))
            current_gap = gap
        else:
            height = width = maximum_dimension
            current_gap = 0

        placements = _place_rectangles(
            rng,
            height,
            width,
            rectangle_sizes,
            current_gap,
        )
        if placements is None:
            continue

        grid = np.full(
            (height, width),
            background,
            dtype=np.int8,
        )

        prototype_orientation = int(rng.integers(0, 4))
        prototype = np.rot90(
            canonical,
            -prototype_orientation,
        )

        prototype_top, prototype_left = placements[0]
        grid[
            prototype_top:prototype_top + 3,
            prototype_left:prototype_left + 3,
        ] = prototype

        for copy_index, (scale, relative_orientation) in enumerate(
            zip(scales, orientations),
            start=1,
        ):
            scale = int(scale)
            absolute_orientation = (
                prototype_orientation
                + int(relative_orientation)
            ) % 4

            transformed = np.rot90(
                canonical,
                -absolute_orientation,
            )
            top, left = placements[copy_index]

            for matrix_row in range(3):
                for matrix_col in range(3):
                    value = int(
                        transformed[matrix_row, matrix_col]
                    )
                    if value not in (endpoint_a, endpoint_b):
                        continue

                    grid[
                        top + matrix_row * scale:
                        top + (matrix_row + 1) * scale,
                        left + matrix_col * scale:
                        left + (matrix_col + 1) * scale,
                    ] = value

        try:
            analysis = _analyze_input(grid)
        except AssertionError:
            continue

        if analysis["scales"] != tuple(sorted(scales)):
            continue
        if analysis["role_signature"] != expected_signature:
            continue

        return grid, analysis["expected"]

    raise ValueError("unable to construct an unambiguous pair")


def _input_key(
    array: np.ndarray,
) -> tuple[tuple[int, int], str, bytes]:
    return (
        array.shape,
        array.dtype.str,
        array.tobytes(),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _input_key(pair["input"]),
        _input_key(pair["output"]),
    )


def _source_hash(pair: dict) -> str:
    digest = hashlib.sha256()
    for name in ("input", "output"):
        array = np.asarray(pair[name], dtype=np.int8)
        digest.update(bytes(array.shape))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _sample_scales(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    if count == 1:
        values, weights = (1, 2, 3, 4, 5), (0.25, 0.26, 0.30, 0.15, 0.04)
    elif count == 2:
        values, weights = (1, 2, 3, 4, 5), (0.27, 0.31, 0.28, 0.12, 0.02)
    elif count == 3:
        values, weights = (1, 2, 3, 4), (0.39, 0.34, 0.22, 0.05)
    else:
        values, weights = (1, 2, 3), (0.56, 0.36, 0.08)
    scales = [int(v) for v in rng.choice(values, size=count, p=weights)]
    if count > 1 and rng.random() < 0.3:
        index = int(rng.integers(count))
        scales[index] = scales[0]
    return tuple(scales)


def _copy_count(rng: np.random.Generator, regime: str) -> int:
    weights = ((0.35, 0.37, 0.21, 0.07) if regime == "source_train"
               else (0.19, 0.34, 0.32, 0.15))
    return int(rng.choice((1, 2, 3, 4), p=weights))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_integer_count(seed):
        raise ValueError("seed must be an integer")

    rng = np.random.default_rng(int(seed))
    num_train, num_test = _resolve_counts(
        rng,
        num_train,
        num_test,
    )
    # The stencil rule is shared; literal colors and even the local stencil
    # can vary because each input contains its own complete prototype.
    palette = np.asarray(rng.permutation(10), dtype=np.int8)
    continuity = float(rng.beta(1.3, 1.3))
    motif_pool_size = int(rng.choice((1, 2, 3, len(_MOTIF_CATALOG)),
                                     p=(0.2, 0.29, 0.26, 0.25)))
    motif_pool = tuple(int(v) for v in
                       rng.choice(len(_MOTIF_CATALOG), size=motif_pool_size,
                                  replace=False))
    source_size_shift = bool(rng.random() < 0.18)
    heldout_orientation = (int(rng.integers(4))
                           if rng.random() < 0.28 else None)

    plans = []
    previous_roles = None
    available_orientations = [i for i in range(4) if i != heldout_orientation]
    first_orientation = int(rng.choice(available_orientations))
    for group, count in (("train", num_train), ("test", num_test)):
        regime = "source_train" if group == "train" else "target_test"
        for index in range(count):
            copy_count = _copy_count(rng, regime)
            if source_size_shift and group == "train":
                copy_count = min(copy_count, 2)
            scales = list(_sample_scales(rng, copy_count))
            if source_size_shift and group == "train":
                scales = [min(scale, 3) for scale in scales]
            orientation_choices = (available_orientations if group == "train"
                                   else list(range(4)))
            orientations = [int(v) for v in rng.choice(orientation_choices,
                                                       size=copy_count)]
            if group == "train" and index == 0:
                scales[0] = 1
                orientations[0] = first_orientation
            elif group == "train" and index == 1:
                scales[0] = int(rng.choice((2, 3)))
                orientations[0] = int(rng.choice(
                    [v for v in available_orientations if v != first_orientation]))
            elif (group == "test" and index == 0
                  and heldout_orientation is not None):
                orientations[0] = heldout_orientation
            colors = _draw_role_assignment(rng, palette, previous_roles,
                                           continuity)
            previous_roles = colors
            motif = _MOTIF_CATALOG[motif_pool[int(rng.integers(len(motif_pool)))]]
            gap = int(rng.choice((0, 1, 2, 3), p=(0.42, 0.38, 0.15, 0.05)))
            size_bias = int(rng.integers(0, 4 if group == "train" else 5))
            minimum_dimension = (25 if source_size_shift and group == "test"
                                 else 11)
            maximum_dimension = (24 if source_size_shift and group == "train"
                                 else 30)
            plans.append((group, regime, tuple(scales), tuple(orientations),
                          colors, motif, gap, size_bias,
                          minimum_dimension, maximum_dimension))

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs = set()

    for (
        group,
        regime,
        scales,
        orientations,
        colors,
        motif,
        gap,
        size_bias,
        minimum_dimension,
        maximum_dimension,
    ) in plans:
        preferred_shape = None
        if (group == "test" and minimum_dimension < 25
                and puzzle["train"] and rng.random() < 0.35):
            eligible = [pair["input"].shape for pair in puzzle["train"]
                        if min(pair["input"].shape) >= 3 * max(scales)]
            if eligible:
                preferred_shape = eligible[int(rng.integers(len(eligible)))]
        for _ in range(100):
            pair_seed = int(
                rng.integers(
                    0,
                    np.iinfo(np.int64).max,
                )
            )
            pair_rng = np.random.default_rng(pair_seed)

            input_grid, output_grid = _generate_pair(
                pair_rng, regime, scales, orientations, colors, motif,
                gap, size_bias, minimum_dimension, maximum_dimension,
                preferred_shape,
            )

            key = _input_key(input_grid)
            if key not in seen_inputs:
                seen_inputs.add(key)
                puzzle[group].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                break
        else:
            raise ValueError("unable to generate a unique pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs = set()
    seen_pairs = set()

    analyses = {
        "train": [],
        "test": [],
    }

    for group in ("train", "test"):
        for pair in puzzle[group]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))

            assert input_grid.shape == output_grid.shape

            input_key = _input_key(input_grid)
            pair_key = _pair_key(pair)

            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs

            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _source_hash(pair) not in _SOURCE_HASHES

            analysis = _analyze_input(input_grid)
            assert np.array_equal(
                output_grid,
                analysis["expected"],
            )

            difference = input_grid != output_grid
            assert np.any(difference)
            assert np.all(
                input_grid[difference]
                == analysis["background"]
            )
            assert np.all(
                output_grid[difference]
                == analysis["connector"]
            )

            assert set(
                int(value) for value in np.unique(output_grid)
            ) == set(
                int(value) for value in np.unique(input_grid)
            )

            analyses[group].append(analysis)

    for group in ("train", "test"):
        for analysis in analyses[group]:
            assert 1 <= len(analysis["copies"]) <= 4
            assert all(1 <= scale <= 9 for scale in analysis["scales"])
            assert 1 <= int(np.count_nonzero(
                analysis["prototype"] == analysis["connector"])) <= 7

    demonstrated_scales = {
        scale
        for analysis in analyses["train"]
        for scale in analysis["scales"]
    }
    assert 1 in demonstrated_scales
    assert any(scale > 1 for scale in demonstrated_scales)

    training_orientations = {
        orientation
        for analysis in analyses["train"]
        for orientation in analysis["orientations"]
    }
    assert len(training_orientations) >= 2


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple(
        (
            group,
            _pair_key(pair),
        )
        for group in ("train", "test")
        for pair in puzzle[group]
    )


def self_test() -> None:
    automatic_puzzles = [
        generate_puzzle(seed)
        for seed in range(4200, 4242)
    ]

    for puzzle in automatic_puzzles:
        validate_generated_puzzle(puzzle)

    train_counts = {
        len(puzzle["train"])
        for puzzle in automatic_puzzles
    }
    test_counts = {
        len(puzzle["test"])
        for puzzle in automatic_puzzles
    }

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3

    fingerprints = {
        _puzzle_fingerprint(puzzle)
        for puzzle in automatic_puzzles
    }
    assert len(fingerprints) >= 40

    role_orders = set()
    motif_geometries = set()
    shapes = set()
    layouts = set()
    profiles = set()
    observed = {name: 0 for name in (
        "one_copy", "four_copies", "scale_four", "sparse", "dense",
        "touching", "repeat_orientation", "mixed_scale",
        "motif_reuse", "motif_change", "role_reuse",
        "test_larger", "test_not_larger", "new_orientation",
        "familiar_orientation", "within_count_change",
        "single_one", "single_two", "same_larger_pair", "mixed_two_three",
        "scale_five", "open_boxes", "touch_and_mixed",
        "test_new_scale", "test_same_shape", "test_smaller_dimension",
        "new_scale_touch_mixed",
    )}

    for puzzle in automatic_puzzles:
        all_analyses = {
            group: [_analyze_input(pair["input"]) for pair in puzzle[group]]
            for group in ("train", "test")
        }
        flattened = all_analyses["train"] + all_analyses["test"]
        signatures = [analysis["role_signature"] for analysis in flattened]
        roles = [(analysis["background"], analysis["connector"])
                 for analysis in flattened]
        observed["motif_reuse"] += len(set(signatures)) < len(signatures)
        observed["motif_change"] += len(set(signatures)) > 1
        observed["role_reuse"] += len(set(roles)) < len(roles)
        observed["within_count_change"] += len({len(a["copies"])
                                                 for a in flattened}) > 1
        training_orientations = {v for a in all_analyses["train"]
                                 for v in a["orientations"]}
        training_shapes = [pair["input"].shape for pair in puzzle["train"]]
        for pair, analysis in zip(puzzle["test"], all_analyses["test"]):
            height, width = pair["input"].shape
            larger = all(height > h and width > w for h, w in training_shapes)
            observed["test_larger"] += larger
            observed["test_not_larger"] += not larger
            observed["test_same_shape"] += (height, width) in training_shapes
            observed["test_smaller_dimension"] += any(
                height < h or width < w for h, w in training_shapes)
            new = bool(set(analysis["orientations"]) - training_orientations)
            observed["new_orientation"] += new
            observed["familiar_orientation"] += not new
            trained_scales = {s for a in all_analyses["train"] for s in a["scales"]}
            new_scale = bool(set(analysis["scales"]) - trained_scales)
            observed["test_new_scale"] += new_scale
            boxes = [(analysis["top"], analysis["left"], 3)] + [
                (copy["top"], copy["left"], 3 * copy["scale"])
                for copy in analysis["copies"]]
            touching = any(_boxes_touch(box, other)
                           for i, box in enumerate(boxes)
                           for other in boxes[i + 1:])
            observed["new_scale_touch_mixed"] += (
                new_scale and touching and len(set(analysis["scales"])) > 1)

        first_analysis = _analyze_input(
            puzzle["train"][0]["input"]
        )
        role_orders.add(
            (
                first_analysis["background"],
                first_analysis["connector"],
                first_analysis["endpoint_a"],
                first_analysis["endpoint_b"],
            )
        )

        for group in ("train", "test"):
            for pair in puzzle[group]:
                analysis = _analyze_input(pair["input"])

                assert np.array_equal(
                    pair["output"],
                    analysis["expected"],
                )
                assert _source_hash(pair) not in _SOURCE_HASHES

                motif_geometries.add(
                    analysis["role_signature"]
                )
                profiles.add(analysis["scales"])
                observed["single_one"] += analysis["scales"] == (1,)
                observed["single_two"] += analysis["scales"] == (2,)
                observed["same_larger_pair"] += (
                    len(analysis["scales"]) == 2
                    and analysis["scales"][0] == analysis["scales"][1] >= 2)
                observed["mixed_two_three"] += analysis["scales"] == (2, 3)
                observed["scale_five"] += 5 in analysis["scales"]
                observed["one_copy"] += len(analysis["copies"]) == 1
                observed["four_copies"] += len(analysis["copies"]) == 4
                observed["scale_four"] += max(analysis["scales"]) >= 4
                connector_count = int(np.count_nonzero(
                    analysis["prototype"] == analysis["connector"]))
                observed["sparse"] += connector_count == 1
                observed["dense"] += connector_count == 7
                observed["mixed_scale"] += len(set(analysis["scales"])) > 1
                observed["repeat_orientation"] += (
                    len(set(analysis["orientations"])) < len(analysis["orientations"]))
                boxes = [(analysis["top"], analysis["left"], 3)] + [
                    (copy["top"], copy["left"], 3 * copy["scale"])
                    for copy in analysis["copies"]]
                touching = any(
                    _boxes_touch(box, other)
                    for index, box in enumerate(boxes)
                    for other in boxes[index + 1:])
                observed["touching"] += touching
                observed["open_boxes"] += not touching
                observed["touch_and_mixed"] += (
                    touching and len(set(analysis["scales"])) > 1)
                shapes.add(pair["input"].shape)
                layouts.add(
                    (
                        pair["input"].shape,
                        analysis["top"],
                        analysis["left"],
                        tuple(
                            (
                                copy["top"],
                                copy["left"],
                                copy["scale"],
                                copy["orientation"],
                            )
                            for copy in analysis["copies"]
                        ),
                    )
                )

                assert (
                    pair["input"].shape
                    == pair["output"].shape
                )
                assert np.issubdtype(
                    pair["input"].dtype,
                    np.integer,
                )
                assert np.issubdtype(
                    pair["output"].dtype,
                    np.integer,
                )
                assert np.all(
                    (0 <= pair["input"])
                    & (pair["input"] <= 9)
                )
                assert np.all(
                    (0 <= pair["output"])
                    & (pair["output"] <= 9)
                )

    assert len(role_orders) >= 25
    assert len(motif_geometries) >= 3
    assert len(shapes) >= 30
    assert len(layouts) >= 100
    assert len(profiles) >= 25
    assert all(observed[name] > 0 for name in observed)
    assert observed["touching"] >= 10
    assert observed["motif_change"] >= 10
    assert observed["role_reuse"] >= 10

    explicit = generate_puzzle(
        7001,
        num_train=6,
        num_test=3,
    )
    validate_generated_puzzle(explicit)

    repeated = generate_puzzle(
        7001,
        num_train=6,
        num_test=3,
    )
    assert (
        _puzzle_fingerprint(explicit)
        == _puzzle_fingerprint(repeated)
    )

    different = generate_puzzle(
        7002,
        num_train=6,
        num_test=3,
    )
    assert (
        _puzzle_fingerprint(explicit)
        != _puzzle_fingerprint(different)
    )

    boundary_cases = (
        (8001, 2, 1),
        (8002, 10, 1),
        (8003, 10, 4),
    )

    for seed, train_count, test_count in boundary_cases:
        puzzle = generate_puzzle(
            seed,
            train_count,
            test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    partial_cases = (
        (8101, None, 1),
        (8102, None, 4),
        (8103, 2, None),
        (8104, 10, None),
    )

    for seed, train_count, test_count in partial_cases:
        puzzle = generate_puzzle(
            seed,
            train_count,
            test_count,
        )

        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count

        validate_generated_puzzle(puzzle)

    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 4, "num_test": 4},
        {"num_train": 3, "num_test": 4},
        {"num_train": True, "num_test": 1},
    )

    for arguments in illegal_cases:
        try:
            generate_puzzle(9000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"illegal counts were accepted: {arguments}"
            )
