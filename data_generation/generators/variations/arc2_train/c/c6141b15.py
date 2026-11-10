import numpy as np


_TRAIN_MOTIFS = {
    "singleton": frozenset({(0, 0)}),
    "plus": frozenset({
        (-1, 0),
        (0, -1), (0, 0), (0, 1),
        (1, 0),
    }),
    "x": frozenset({
        (-1, -1), (-1, 1),
        (0, 0),
        (1, -1), (1, 1),
    }),
}

_TEST_MOTIFS = {
    "tall_plus": frozenset({
        (-2, 0),
        (-1, 0),
        (0, -1), (0, 0), (0, 1),
        (1, 0),
        (2, 0),
    }),
    "tall_i": frozenset({
        (-2, -1), (-2, 0), (-2, 1),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, -1), (2, 0), (2, 1),
    }),
    "tall_h": frozenset({
        (-2, -1), (-2, 1),
        (-1, -1), (-1, 1),
        (0, -1), (0, 0), (0, 1),
        (1, -1), (1, 1),
        (2, -1), (2, 1),
    }),
    "tall_z": frozenset({
        (-2, -1), (-2, 0), (-2, 1),
        (-1, 1),
        (0, 0),
        (1, -1),
        (2, -1), (2, 0), (2, 1),
    }),
}

_ASYMMETRIC_MOTIFS = {
    "t": frozenset({(-1, -1), (-1, 0), (-1, 1),
                     (0, 0), (1, 0)}),
    "arrow": frozenset({(-1, 0), (0, -1), (0, 0), (0, 1),
                         (1, 0), (1, 1)}),
    "wide_t": frozenset({(-1, -2), (-1, -1), (-1, 0),
                          (-1, 1), (-1, 2), (0, 0), (1, 0)}),
    "wide_arrow": frozenset({(-1, -2), (0, -2), (0, -1),
                              (0, 0), (0, 1), (0, 2), (1, -2)}),
    "irregular": frozenset({(-2, 0), (-1, -1), (-1, 0),
                             (0, -1), (0, 0), (0, 1),
                             (1, 1), (2, 1)}),
}

_MOTIFS = {**_TRAIN_MOTIFS, **_TEST_MOTIFS, **_ASYMMETRIC_MOTIFS}
_SPECIAL_NAMES = tuple(_ASYMMETRIC_MOTIFS)
_SYMMETRIC_NAMES = tuple(_TRAIN_MOTIFS) + tuple(_TEST_MOTIFS)

_SOURCE_PAIR_TEXTS = (
    (
        "7777777777777777/7777777777777777/7777777277777777/"
        "7777772777777777/7777727777777777/7777277777777177/"
        "7772777777771117/7727777777777177/7777777777777777/"
        "7777777777777777/7777777777777777/7777777177777177/"
        "7777771117771117/7777777177777177/7777777777777777/"
        "7777777777777777",
        "7777777777777777/7777777177777777/7777771117777777/"
        "7777777177777777/7777777777777777/7777777777777777/"
        "7717777777777277/7111777777772277/7717777777727277/"
        "7777777777277277/7777777772777277/7777777727777277/"
        "7777777222222277/7777777777777777/7777777777777777/"
        "7777777777777777",
    ),
    (
        "77777777777/77777777777/77977777777/77977777777/"
        "77977777777/77977577757/77977777777/77977777777/"
        "77777777777/77777777777/77777777777",
        "77777777777/77777777777/77577777777/77777777777/"
        "77777777777/77777999997/77777777777/77577777777/"
        "77777777777/77777777777/77777777777",
    ),
    (
        "7777777777777777/7878777777777777/7787777777777777/"
        "7878777777747777/7777777777747777/7777787877747777/"
        "7777778777747777/7777787877747777/7777777777747777/"
        "7878777777747777/7787777777747777/7878777777747777/"
        "7777777777747777/7777777777777777/7777777777777777/"
        "7777777777777777",
        "7777777777777777/7777777777777777/7747777777878777/"
        "7744777777787777/7747477777878777/7747747777777777/"
        "7747774777777777/7747747777777777/7747477777777777/"
        "7744777777777777/7747777777777777/7777777777878777/"
        "7777777777787777/7777777777878777/7777777777777777/"
        "7777777777777777",
    ),
    (
        "7777777777777777/7777722277722277/7777772777772777/"
        "7777772777772777/7777772777772777/7777722277722277/"
        "7777777777777777/7777777777722277/7777777777772777/"
        "7777777777777777/7777777777777777/7777777777722277/"
        "7711111177777777/7777777777777777/7777777777777777/"
        "7777777777777777",
        "7777777777777777/7777777777777777/7777777777777777/"
        "7777771111111777/7777777177771777/7777777717771777/"
        "7777777771771777/7777777777171777/7777777777711777/"
        "7777777777771777/7222772227777777/7727777277777777/"
        "7727777277777777/7727777277777777/7222772227777777/"
        "7777777777777777",
    ),
)


def _is_int_value(value) -> bool:
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
        if not _is_int_value(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError("num_train must be an integer from 2 through 10")
        num_train = int(num_train)

    if num_test is not None:
        if not _is_int_value(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError("num_test must be an integer from 1 through 4")
        num_test = int(num_test)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        lower = max(2, num_test + 1)
        num_train = int(rng.integers(lower, 11))
    elif num_train is None:
        lower = max(2, num_test + 1)
        num_train = int(rng.integers(lower, 11))
    else:
        upper = min(4, num_train - 1)
        if upper < 1:
            raise ValueError("no legal num_test exists for num_train")
        num_test = int(rng.integers(1, upper + 1))

    return num_train, num_test


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [
        int(value)
        for value in rng.choice(10, size=3, replace=False)
    ]
    return {
        "base_palette": tuple(colors),
        "featured_motif": _SPECIAL_NAMES[int(rng.integers(len(_SPECIAL_NAMES)))],
    }


def _pair_palette(
    rng: np.random.Generator,
    base: tuple[int, int, int],
    previous: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    mode = int(rng.integers(0, 5))
    if mode == 0:
        return base
    if mode == 1:
        return (base[0], base[2], base[1])
    if mode == 2 and previous is not None:
        return (previous[0], previous[2], previous[1])
    if mode == 3:
        background = base[0]
        foreground = [color for color in range(10) if color != background]
        chosen = rng.choice(foreground, size=2, replace=False)
        return (background, int(chosen[0]), int(chosen[1]))
    chosen = rng.choice(10, size=3, replace=False)
    return tuple(int(value) for value in chosen)


def _sample_specifications(
    rng: np.random.Generator,
    num_train: int,
    num_test: int,
    parameters: dict,
) -> dict:
    featured = parameters["featured_motif"]
    base = parameters["base_palette"]
    counts = [2, 3]
    rng.shuffle(counts)
    counts += [int(rng.integers(2, 4)) for _ in range(num_train - 2)]

    orientations = list(rng.choice(
        ("horizontal", "vertical", "diagonal"), size=2, replace=False
    ))
    orientations += [
        ("horizontal", "vertical", "diagonal")[int(rng.integers(3))]
        for _ in range(num_train - 2)
    ]

    motif_names = [
        _SYMMETRIC_NAMES[int(rng.integers(len(_SYMMETRIC_NAMES)))]
        for _ in range(num_train)
    ]
    feature_index = int(rng.integers(num_train))
    motif_names[feature_index] = featured

    specifications = {"source_train": [], "target_test": []}
    previous = None
    for index in range(num_train + num_test):
        regime = "source_train" if index < num_train else "target_test"
        if regime == "source_train":
            count = counts[index]
            orientation = orientations[index]
            motif = motif_names[index]
        else:
            count = int(rng.integers(2, 4))
            orientation = ("horizontal", "vertical", "diagonal")[
                int(rng.integers(3))
            ]
            motif = (
                featured if rng.random() < 0.48 else
                _SYMMETRIC_NAMES[int(rng.integers(len(_SYMMETRIC_NAMES)))]
            )
        source_like = regime == "target_test" and rng.random() < 0.23
        if source_like:
            count = 3
            orientation = "horizontal"
            motif = "tall_i"

        palette = base if index == 0 else _pair_palette(rng, base, previous)
        if source_like and rng.random() < 0.65:
            palette = (base[0], base[2], base[1])
        if index == 1:
            for _ in range(20):
                if palette != base:
                    break
                palette = _pair_palette(rng, base, previous)
            if palette == base:
                palette = (base[0], base[2], base[1])
        previous = palette
        specifications[regime].append({
            "count": count,
            "orientation": orientation,
            "motif": motif,
            "palette": palette,
            "source_size": source_like,
        })
    return specifications


def _line_cells(
    a: tuple[int, int],
    b: tuple[int, int],
) -> frozenset[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0

    if dr == 0 and dc == 0:
        return frozenset({a})

    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError(
            "line is not horizontal, vertical, or 45-degree diagonal"
        )

    length = max(abs(dr), abs(dc))
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)

    return frozenset(
        (r0 + index * step_r, c0 + index * step_c)
        for index in range(length + 1)
    )


def _straight_info(cells: frozenset) -> dict | None:
    cells = frozenset(cells)
    if len(cells) < 4:
        return None

    points = sorted(cells)
    farthest = None

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p = points[i]
            q = points[j]
            distance = (
                (p[0] - q[0]) ** 2
                + (p[1] - q[1]) ** 2
            )
            candidate = (distance, p, q)
            if farthest is None or candidate > farthest:
                farthest = candidate

    try:
        expected = _line_cells(farthest[1], farthest[2])
    except ValueError:
        return None

    if expected != cells:
        return None

    return {
        "endpoints": (farthest[1], farthest[2]),
        "cells": cells,
    }


def _components8(cells: frozenset) -> list[frozenset]:
    unseen = set(cells)
    components = []

    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component = {start}

        while stack:
            row, col = stack.pop()

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue

                    neighbor = (row + dr, col + dc)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)

        components.append(frozenset(component))

    components.sort(key=lambda component: min(component))
    return components


def _glyph_info(cells: frozenset) -> dict | None:
    components = _components8(cells)
    if len(components) not in (2, 3):
        return None

    centers = []
    motifs = []

    for component in components:
        rows = [point[0] for point in component]
        cols = [point[1] for point in component]

        if (min(rows) + max(rows)) % 2:
            return None
        if (min(cols) + max(cols)) % 2:
            return None

        center = (
            (min(rows) + max(rows)) // 2,
            (min(cols) + max(cols)) // 2,
        )
        motif = frozenset(
            (row - center[0], col - center[1])
            for row, col in component
        )

        if (0, 0) not in motif:
            return None

        centers.append(center)
        motifs.append(motif)

    if any(motif != motifs[0] for motif in motifs[1:]):
        return None

    centers = tuple(sorted(centers))

    try:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                _line_cells(centers[i], centers[j])
    except ValueError:
        return None

    if len(centers) == 3:
        a, b, c = centers
        twice_area = (
            (b[0] - a[0]) * (c[1] - a[1])
            - (b[1] - a[1]) * (c[0] - a[0])
        )
        if twice_area == 0:
            return None

        squared_distances = sorted(
            (centers[i][0] - centers[j][0]) ** 2
            + (centers[i][1] - centers[j][1]) ** 2
            for i in range(3)
            for j in range(i + 1, 3)
        )
        if (
            squared_distances[0] + squared_distances[1]
            != squared_distances[2]
        ):
            return None

    return {
        "centers": centers,
        "motif": motifs[0],
        "components": tuple(components),
    }


def _orientation(
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> str:
    (r0, c0), (r1, c1) = endpoints

    if r0 == r1:
        return "horizontal"
    if c0 == c1:
        return "vertical"
    return "diagonal"


def _analyze_input(grid: np.ndarray) -> dict:
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3, "input must contain exactly three colors"

    order = np.argsort(counts)
    assert (
        counts[order[-1]] > counts[order[-2]]
    ), "background must be uniquely most frequent"

    background = int(values[order[-1]])
    foreground = [
        int(value)
        for value in values
        if int(value) != background
    ]

    candidates = []

    for edge_color in foreground:
        edge_cells = frozenset(
            map(tuple, np.argwhere(grid == edge_color).tolist())
        )
        straight = _straight_info(edge_cells)
        if straight is None:
            continue

        if foreground[1] == edge_color:
            glyph_color = foreground[0]
        else:
            glyph_color = foreground[1]

        glyph_cells = frozenset(
            map(tuple, np.argwhere(grid == glyph_color).tolist())
        )
        glyph = _glyph_info(glyph_cells)

        if glyph is not None:
            candidates.append(
                (
                    edge_color,
                    glyph_color,
                    straight,
                    glyph,
                )
            )

    assert (
        len(candidates) == 1
    ), "foreground roles must be structurally unambiguous"

    edge_color, glyph_color, straight, glyph = candidates[0]

    return {
        "background": background,
        "edge_color": edge_color,
        "glyph_color": glyph_color,
        "edge_cells": straight["cells"],
        "endpoints": straight["endpoints"],
        "centers": glyph["centers"],
        "motif": glyph["motif"],
    }


def _motif_cells(
    centers: tuple[tuple[int, int], ...],
    motif: frozenset,
) -> frozenset:
    return frozenset(
        (row + dr, col + dc)
        for row, col in centers
        for dr, dc in motif
    )


def _figure_cells(
    centers: tuple[tuple[int, int], ...],
) -> frozenset:
    cells = set()

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            cells.update(_line_cells(centers[i], centers[j]))

    return frozenset(cells)


def _far_apart(
    first: frozenset,
    second: frozenset,
    gap: int,
) -> bool:
    return all(
        max(abs(r1 - r2), abs(c1 - c2)) > gap
        for r1, c1 in first
        for r2, c2 in second
    )


def _expected_output(
    input_grid: np.ndarray,
    analysis: dict,
) -> np.ndarray:
    height, width = input_grid.shape

    figure = _figure_cells(analysis["centers"])
    endpoint_glyphs = _motif_cells(
        analysis["endpoints"],
        analysis["motif"],
    )

    assert figure.isdisjoint(
        endpoint_glyphs
    ), "output roles may not collide"

    assert all(
        0 <= row < height and 0 <= col < width
        for row, col in endpoint_glyphs
    ), "endpoint glyph does not fit inside the grid"

    assert (
        len(_components8(endpoint_glyphs)) == 2
    ), "endpoint glyph copies must remain separate"

    output = np.full(
        (height, width),
        analysis["background"],
        dtype=np.int8,
    )

    for row, col in figure:
        output[row, col] = analysis["edge_color"]

    for row, col in endpoint_glyphs:
        output[row, col] = analysis["glyph_color"]

    return output


def _motif_bbox(motif: frozenset) -> tuple[int, int]:
    rows = [point[0] for point in motif]
    cols = [point[1] for point in motif]

    return (
        max(rows) - min(rows) + 1,
        max(cols) - min(cols) + 1,
    )


def _grid_text(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (
        _grid_text(input_grid),
        _grid_text(output_grid),
    ) in _SOURCE_PAIR_TEXTS


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )


def _check_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray), "grid must be a NumPy array"
    assert grid.ndim == 2, "grid must be two-dimensional"

    height, width = grid.shape
    assert 1 <= height <= 30
    assert 1 <= width <= 30

    assert np.issubdtype(
        grid.dtype,
        np.integer,
    ), "grid dtype must be integral"

    assert bool(
        np.all((0 <= grid) & (grid <= 9))
    ), "grid values must be colors 0 through 9"


def _check_pair(pair: dict, regime: str) -> dict:
    assert isinstance(pair, dict)
    assert set(pair) == {"input", "output"}

    input_grid = pair["input"]
    output_grid = pair["output"]

    _check_grid(input_grid)
    _check_grid(output_grid)

    assert (
        input_grid.shape == output_grid.shape
    ), "input and output dimensions must match"

    analysis = _analyze_input(input_grid)

    original_glyphs = _motif_cells(
        analysis["centers"],
        analysis["motif"],
    )
    assert _far_apart(
        analysis["edge_cells"],
        original_glyphs,
        1,
    ), "input roles must be separated by background"

    expected = _expected_output(input_grid, analysis)
    assert np.array_equal(
        output_grid,
        expected,
    ), "output is not the required arrangement exchange"

    assert not np.array_equal(
        input_grid,
        output_grid,
    ), "degenerate unchanged pair"

    assert not _is_source_pair(
        input_grid,
        output_grid,
    ), "generated pair duplicates a source-task pair"

    if regime not in ("source_train", "target_test"):
        raise AssertionError("unknown pair regime")
    assert analysis["motif"] in set(_MOTIFS.values())

    return analysis


def _translate(
    relative_points: tuple[tuple[int, int], ...],
    motif: frozenset,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...] | None:
    min_offset_row = min(point[0] for point in motif)
    max_offset_row = max(point[0] for point in motif)
    min_offset_col = min(point[1] for point in motif)
    max_offset_col = max(point[1] for point in motif)

    low_row = -(
        min(row for row, _ in relative_points)
        + min_offset_row
    )
    high_row = height - 1 - (
        max(row for row, _ in relative_points)
        + max_offset_row
    )
    low_col = -(
        min(col for _, col in relative_points)
        + min_offset_col
    )
    high_col = width - 1 - (
        max(col for _, col in relative_points)
        + max_offset_col
    )

    if low_row > high_row or low_col > high_col:
        return None

    shift_row = int(rng.integers(low_row, high_row + 1))
    shift_col = int(rng.integers(low_col, high_col + 1))

    return tuple(
        (row + shift_row, col + shift_col)
        for row, col in relative_points
    )


def _transform_relative(
    points: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    reflect = bool(rng.integers(0, 2))
    rotations = int(rng.integers(0, 4))
    transformed = []

    for row, col in points:
        if reflect:
            col = -col

        for _ in range(rotations):
            row, col = col, -row

        transformed.append((row, col))

    return tuple(transformed)


def _sample_centers(
    rng: np.random.Generator,
    height: int,
    width: int,
    motif: frozenset,
    count: int,
    regime: str,
) -> tuple[tuple[int, int], ...] | None:
    if count == 2:
        distance = int(rng.integers(4, min(14, max(height, width) - 3) + 1))
        direction = (
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1),
        )[int(rng.integers(0, 4))]

        relative = (
            (0, 0),
            (
                direction[0] * distance,
                direction[1] * distance,
            ),
        )
    else:
        diagonal_basis = bool(rng.integers(0, 2))
        length = int(rng.integers(4, min(11, max(height, width) - 4) + 1))

        if diagonal_basis:
            relative = (
                (0, 0),
                (length, length),
                (length, -length),
            )
        else:
            relative = (
                (0, 0),
                (0, length),
                (length, 0),
            )

        relative = _transform_relative(relative, rng)

    return _translate(
        relative,
        motif,
        height,
        width,
        rng,
    )


def _sample_endpoints(
    rng: np.random.Generator,
    height: int,
    width: int,
    motif: frozenset,
    orientation: str,
) -> tuple[tuple[int, int], ...] | None:
    upper = min(17, max(height, width) - 3)
    if orientation == "horizontal":
        distance = int(rng.integers(4, upper + 1))
        relative = ((0, 0), (0, distance))
    elif orientation == "vertical":
        distance = int(rng.integers(4, upper + 1))
        relative = ((0, 0), (distance, 0))
    elif orientation == "diagonal":
        distance = int(rng.integers(4, upper + 1))
        sign = -1 if bool(rng.integers(0, 2)) else 1
        relative = ((0, 0), (distance, sign * distance))
    else:
        raise ValueError("unsupported source-segment orientation")

    return _translate(
        relative,
        motif,
        height,
        width,
        rng,
    )


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    specification: dict,
) -> dict:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    glyph_count = specification["count"]
    motif_name = specification["motif"]
    source_orientation = specification["orientation"]
    motif = _MOTIFS[motif_name]
    background, edge_color, glyph_color = specification["palette"]

    for _ in range(4000):
        if specification["source_size"] or (
            regime == "target_test" and rng.random() < 0.08
        ):
            height = 16
            width = 16
        else:
            lower_dimension = 11 if motif_name == "singleton" else 13
            if glyph_count == 3 and motif_name != "singleton":
                lower_dimension = 15
            height = int(rng.integers(lower_dimension, 31))
            width = int(rng.integers(lower_dimension, 31))

        centers = _sample_centers(
            rng,
            height,
            width,
            motif,
            glyph_count,
            regime,
        )
        endpoints = _sample_endpoints(
            rng,
            height,
            width,
            motif,
            source_orientation,
        )

        if centers is None or endpoints is None:
            continue

        input_edge = _line_cells(endpoints[0], endpoints[1])
        input_glyphs = _motif_cells(centers, motif)

        if not _far_apart(input_edge, input_glyphs, 1):
            continue

        if len(_components8(input_glyphs)) != glyph_count:
            continue

        output_figure = _figure_cells(centers)
        output_glyphs = _motif_cells(endpoints, motif)

        if not output_figure.isdisjoint(output_glyphs):
            continue

        if len(_components8(output_glyphs)) != 2:
            continue

        input_grid = np.full(
            (height, width),
            background,
            dtype=np.int8,
        )
        output_grid = np.full(
            (height, width),
            background,
            dtype=np.int8,
        )

        for row, col in input_edge:
            input_grid[row, col] = edge_color

        for row, col in input_glyphs:
            input_grid[row, col] = glyph_color

        for row, col in output_figure:
            output_grid[row, col] = edge_color

        for row, col in output_glyphs:
            output_grid[row, col] = glyph_color

        pair = {
            "input": input_grid,
            "output": output_grid,
        }

        try:
            analysis = _check_pair(pair, regime)
        except AssertionError:
            continue

        if analysis["edge_color"] != edge_color:
            continue
        if analysis["glyph_color"] != glyph_color:
            continue

        return pair

    raise ValueError("could not construct an unambiguous pair")


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
    train_analyses = []
    test_analyses = []

    splits = (
        ("source_train", "train", train_analyses),
        ("target_test", "test", test_analyses),
    )

    for regime, split_name, analyses in splits:
        for pair in puzzle[split_name]:
            analysis = _check_pair(pair, regime)
            analyses.append(analysis)

            input_key = _grid_key(pair["input"])
            pair_key = (
                input_key,
                _grid_key(pair["output"]),
            )

            assert (
                input_key not in input_keys
            ), "duplicate input grid"
            assert (
                pair_key not in pair_keys
            ), "duplicate input/output pair"

            input_keys.add(input_key)
            pair_keys.add(pair_key)

    assert {
        len(analysis["centers"])
        for analysis in train_analyses
    } >= {2, 3}, (
        "training must demonstrate both segment and triangle construction"
    )

    assert len({
        _orientation(analysis["endpoints"])
        for analysis in train_analyses
    }) >= 2, "training must show multiple source directions"
    assert len({
        (analysis["background"], analysis["edge_color"],
         analysis["glyph_color"])
        for analysis in train_analyses
    }) >= 2, "training must show that colors are pair-local"

    training_motifs = {analysis["motif"] for analysis in train_analyses}
    assert any(len(motif) > 1 for motif in training_motifs)
    special = set(_ASYMMETRIC_MOTIFS.values())
    for analysis in test_analyses:
        if analysis["motif"] in special:
            assert analysis["motif"] in training_motifs, (
                "asymmetric test glyph must have a training copy example"
            )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_int_value(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")

    rng = np.random.default_rng(int(seed))

    num_train, num_test = _resolve_counts(
        rng,
        num_train,
        num_test,
    )
    puzzle_parameters = sample_puzzle_parameters(rng)
    pair_specifications = _sample_specifications(
        rng, num_train, num_test, puzzle_parameters
    )

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs = set()

    specifications = (
        ("source_train", "train"),
        ("target_test", "test"),
    )

    for regime, split_name in specifications:
        for specification in pair_specifications[regime]:
            for _ in range(100):
                child_seed = int(
                    rng.integers(
                        0,
                        np.iinfo(np.int64).max,
                    )
                )
                pair_rng = np.random.default_rng(child_seed)

                pair = _generate_pair(
                    pair_rng,
                    regime,
                    specification,
                )
                input_key = _grid_key(pair["input"])

                if input_key not in seen_inputs:
                    seen_inputs.add(input_key)
                    puzzle[split_name].append(pair)
                    break
            else:
                raise ValueError(
                    "could not generate enough distinct pairs"
                )

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != set(second):
        return False

    for split_name in ("train", "test"):
        if len(first[split_name]) != len(second[split_name]):
            return False

        for first_pair, second_pair in zip(
            first[split_name],
            second[split_name],
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


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        (
            split_name,
            _grid_text(pair["input"]),
            _grid_text(pair["output"]),
        )
        for split_name in ("train", "test")
        for pair in puzzle[split_name]
    )


def self_test() -> None:
    train_counts = set()
    test_counts = set()
    signatures = set()
    palettes = set()
    shapes = set()
    structures = set()
    orientations = {"train": set(), "test": set()}
    motif_names = {"train": set(), "test": set()}
    mark_counts = {"train": set(), "test": set()}
    test_bboxes = set()
    triangle_lengths = set()
    long_segments = 0
    special_test_pairs = 0
    source_like_tests = 0
    combined = set()
    motif_lookup = {motif: name for name, motif in _MOTIFS.items()}

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        signatures.add(_puzzle_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        analyses = {}

        for split in ("train", "test"):
            analyses[split] = []
            for pair in puzzle[split]:
                regime = "source_train" if split == "train" else "target_test"
                analysis = _check_pair(pair, regime)
                analyses[split].append(analysis)
                orientation = _orientation(analysis["endpoints"])
                name = motif_lookup[analysis["motif"]]
                count = len(analysis["centers"])
                palette = (analysis["background"], analysis["edge_color"],
                           analysis["glyph_color"])
                distance = max(abs(analysis["endpoints"][0][axis] -
                                   analysis["endpoints"][1][axis]) for axis in (0, 1))
                palettes.add(palette)
                shapes.add(pair["input"].shape)
                orientations[split].add(orientation)
                motif_names[split].add(name)
                mark_counts[split].add(count)
                structures.add((split, pair["input"].shape, orientation,
                                name, count, distance,
                                tuple(analysis["centers"])))
                combined.add((split, count, orientation,
                              _motif_bbox(analysis["motif"]),
                              pair["input"].shape[0] < pair["input"].shape[1]))
                long_segments += distance >= 10
                if count == 3:
                    r = sorted({point[0] for point in analysis["centers"]})
                    c = sorted({point[1] for point in analysis["centers"]})
                    triangle_lengths.add((r[-1] - r[0], c[-1] - c[0]))
                if split == "test":
                    test_bboxes.add(_motif_bbox(analysis["motif"]))
                    special_test_pairs += name in _SPECIAL_NAMES
                    source_like_tests += (pair["input"].shape == (16, 16)
                                          and orientation == "horizontal"
                                          and count == 3
                                          and _motif_bbox(analysis["motif"]) == (5, 3))

        assert {len(a["centers"]) for a in analyses["train"]} == {2, 3}
        assert len({_orientation(a["endpoints"]) for a in analyses["train"]}) >= 2
        assert len({(a["background"], a["edge_color"], a["glyph_color"])
                    for a in analyses["train"]}) >= 2
        train_motifs = {a["motif"] for a in analyses["train"]}
        assert all(a["motif"] in train_motifs
                   for a in analyses["test"]
                   if a["motif"] in set(_ASYMMETRIC_MOTIFS.values()))

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 40
    assert len(palettes) >= 45
    assert len(shapes) >= 45
    assert len(structures) >= 180
    assert len(combined) >= 35
    assert all(len(orientations[s]) == 3 for s in orientations)
    assert all(mark_counts[s] == {2, 3} for s in mark_counts)
    assert set(_SPECIAL_NAMES).issubset(motif_names["train"])
    assert {"singleton", "plus", "x", "tall_i"}.issubset(motif_names["test"])
    assert (3, 5) in test_bboxes and (5, 3) in test_bboxes
    assert len(triangle_lengths) >= 12
    assert long_segments >= 10
    assert special_test_pairs >= 10
    assert source_like_tests >= 1

    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(1000 + index,
                                                      train_count, test_count))

    for index, (train_count, test_count) in enumerate(
        ((None, 1), (None, 4), (2, None), (10, None))
    ):
        puzzle = generate_puzzle(2000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(3000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count arguments were accepted")
