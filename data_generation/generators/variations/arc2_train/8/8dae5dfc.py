import hashlib

import numpy as np


# SHA-256 of the row-major decimal input cells of the five original pairs.
_SOURCE_INPUT_DIGESTS = frozenset({
    "a276f329ceb4ca9038fee727c8d75c976ea775585de5aade163661828c399616",
    "ca883656493c4861f63fc304537b70e1f0b62552bc9f2d8920905f23cacb4cf8",
    "ac906073c216b1ccd4c6d8d5e236f86fae97cb4fb3c1940f550394013660314c",
    "ed0d66e015ea2767adf4339449bbb478df9119b14baa9d69fda20c63782177b6",
    "a4b005f274865f2f93c04f28c3706a76ab7d3539fcb4d4e70a7a900ea5d73c21",
})


def _source_key(grid: np.ndarray) -> str:
    cells = "".join(str(int(value)) for value in grid.flat)
    return hashlib.sha256(cells.encode("ascii")).hexdigest()


def _foreground_colors(grid: np.ndarray, background: int) -> set[int]:
    return {int(value) for value in np.unique(grid)} - {background}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_supplied_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> None:
    if value is None:
        return
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    _validate_supplied_count("num_train", num_train, 2, 10)
    _validate_supplied_count("num_test", num_test, 1, 4)

    token = int(seed) % 2520

    if num_train is None and num_test is None:
        test_count = 1 + token % 4
        minimum_train = max(2, test_count + 1)
        train_count = (
            minimum_train
            + (token // 4) % (11 - minimum_train)
        )
    elif num_train is None:
        test_count = int(num_test)
        minimum_train = max(2, test_count + 1)
        train_count = (
            minimum_train
            + token % (11 - minimum_train)
        )
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        test_count = 1 + token % maximum_test
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _has_reassigned_depth(sequences: list[list[int]]) -> bool:
    """Whether a shared color occurs at different band depths."""
    for first_index in range(len(sequences)):
        for second_index in range(first_index + 1, len(sequences)):
            first = sequences[first_index]
            second = sequences[second_index]
            for depth_a, color_a in enumerate(first):
                for depth_b, color_b in enumerate(second):
                    if color_a == color_b and depth_a != depth_b:
                        return True
    return False


def _has_local_mapping_conflict(sequences: list[list[int]]) -> bool:
    """
    Whether the same input color receives different output colors in different
    objects, ruling out a single global color substitution.
    """
    for first_index in range(len(sequences)):
        for second_index in range(first_index + 1, len(sequences)):
            first = sequences[first_index]
            second = sequences[second_index]
            for depth_a, color_a in enumerate(first):
                for depth_b, color_b in enumerate(second):
                    if (
                        color_a == color_b
                        and first[-1 - depth_a] != second[-1 - depth_b]
                    ):
                        return True
    return False


def _sample_covering_sequences(
    rng: np.random.Generator,
    layer_counts: list[int],
    palette: tuple[int, ...],
) -> list[list[int]]:
    """
    Sample sequences that collectively use the entire palette and visibly
    demonstrate object-local behavior.
    """
    wanted = set(palette)

    for _ in range(1000):
        sequences = _sample_plain_sequences(
            rng, layer_counts, palette, tuple(wanted))

        observed = {
            color
            for sequence in sequences
            for color in sequence
        }

        if (
            observed == wanted
            and _has_reassigned_depth(sequences)
            and _has_local_mapping_conflict(sequences)
        ):
            return sequences

    raise RuntimeError("could not sample informative color sequences")


def _sample_plain_sequences(
    rng: np.random.Generator,
    layer_counts: list[int],
    palette: tuple[int, ...],
    required: tuple[int, ...] = (),
) -> list[list[int]]:
    for _ in range(300):
        sequences = []
        for count in layer_counts:
            sequence = [int(color) for color in rng.choice(
                palette, size=min(count, len(palette)), replace=False)]
            while count > len(sequence):
                sequence.append(int(rng.choice(
                    [color for color in palette if color != sequence[-1]])))
            # A repeated color can occur at separate depths, never in
            # adjacent bands. Other objects retain distinct band colors.
            if count >= 3 and count <= len(palette) and rng.random() < 0.06:
                sequence[-1] = sequence[0]
            sequences.append(sequence)
        if set(required) <= {color for seq in sequences for color in seq}:
            return sequences
    raise RuntimeError("could not cover required training colors")


def _place_rectangles(
    sizes: list[tuple[int, int]],
    rng: np.random.Generator,
    gap: int,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    """
    Pack one to three rectangles in row, column, or mixed arrangements.
    At most one canvas corner may be occupied, so the background is
    identifiable even when an object touches an edge or corner.
    """
    count = len(sizes)
    candidates: list[
        tuple[list[tuple[int, int]], int, int]
    ] = []

    def add_candidate(
        positions: list[tuple[int, int]],
    ) -> None:
        content_height = max(
            positions[index][0] + sizes[index][0]
            for index in range(count)
        )
        content_width = max(
            positions[index][1] + sizes[index][1]
            for index in range(count)
        )
        if content_height <= 30 and content_width <= 30:
            candidates.append(
                (positions, content_height, content_width)
            )

    if count == 1:
        add_candidate([(0, 0)])

    elif count == 2:
        for order in ((0, 1), (1, 0)):
            # Horizontal arrangement.
            content_height = max(
                sizes[0][0],
                sizes[1][0],
            )
            positions = [(0, 0), (0, 0)]
            col = 0
            for index in order:
                row = int(
                    rng.integers(
                        0,
                        content_height - sizes[index][0] + 1,
                    )
                )
                positions[index] = (row, col)
                col += sizes[index][1] + gap
            add_candidate(positions)

            # Vertical arrangement.
            content_width = max(
                sizes[0][1],
                sizes[1][1],
            )
            positions = [(0, 0), (0, 0)]
            row = 0
            for index in order:
                col = int(
                    rng.integers(
                        0,
                        content_width - sizes[index][1] + 1,
                    )
                )
                positions[index] = (row, col)
                row += sizes[index][0] + gap
            add_candidate(positions)

    elif count == 3:
        permutations = (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )

        for order in permutations:
            # All three in one row.
            content_height = max(
                height for height, _ in sizes
            )
            positions = [(0, 0), (0, 0), (0, 0)]
            col = 0
            for index in order:
                row = int(
                    rng.integers(
                        0,
                        content_height - sizes[index][0] + 1,
                    )
                )
                positions[index] = (row, col)
                col += sizes[index][1] + gap
            add_candidate(positions)

            # All three in one column.
            content_width = max(
                width for _, width in sizes
            )
            positions = [(0, 0), (0, 0), (0, 0)]
            row = 0
            for index in order:
                col = int(
                    rng.integers(
                        0,
                        content_width - sizes[index][1] + 1,
                    )
                )
                positions[index] = (row, col)
                row += sizes[index][0] + gap
            add_candidate(positions)

            first, second, third = order

            # Two above, one below.
            top_height = max(
                sizes[first][0],
                sizes[second][0],
            )
            top_width = (
                sizes[first][1]
                + gap
                + sizes[second][1]
            )
            content_width = max(
                top_width,
                sizes[third][1],
            )
            top_col = int(
                rng.integers(
                    0,
                    content_width - top_width + 1,
                )
            )

            positions = [(0, 0), (0, 0), (0, 0)]
            positions[first] = (
                int(
                    rng.integers(
                        0,
                        top_height - sizes[first][0] + 1,
                    )
                ),
                top_col,
            )
            positions[second] = (
                int(
                    rng.integers(
                        0,
                        top_height - sizes[second][0] + 1,
                    )
                ),
                top_col + sizes[first][1] + gap,
            )
            positions[third] = (
                top_height + gap,
                int(
                    rng.integers(
                        0,
                        content_width - sizes[third][1] + 1,
                    )
                ),
            )
            add_candidate(positions)

            # Two on the left, one on the right.
            left_width = max(
                sizes[first][1],
                sizes[second][1],
            )
            left_height = (
                sizes[first][0]
                + gap
                + sizes[second][0]
            )
            content_height = max(
                left_height,
                sizes[third][0],
            )
            left_row = int(
                rng.integers(
                    0,
                    content_height - left_height + 1,
                )
            )

            positions = [(0, 0), (0, 0), (0, 0)]
            positions[first] = (
                left_row,
                int(
                    rng.integers(
                        0,
                        left_width - sizes[first][1] + 1,
                    )
                ),
            )
            positions[second] = (
                left_row + sizes[first][0] + gap,
                int(
                    rng.integers(
                        0,
                        left_width - sizes[second][1] + 1,
                    )
                ),
            )
            positions[third] = (
                int(
                    rng.integers(
                        0,
                        content_height - sizes[third][0] + 1,
                    )
                ),
                left_width + gap,
            )
            add_candidate(positions)

    else:
        raise ValueError("only one to three objects are supported")

    if not candidates:
        return None

    for candidate_index in rng.permutation(len(candidates)):
        positions, content_height, content_width = candidates[
            int(candidate_index)
        ]

        for _ in range(32):
            top = int(rng.integers(0, min(5, 31 - content_height)))
            bottom = int(rng.integers(0, min(5, 31 - content_height - top)))
            left = int(rng.integers(0, min(5, 31 - content_width)))
            right = int(rng.integers(0, min(5, 31 - content_width - left)))
            height = content_height + top + bottom
            width = content_width + left + right
            if height > 30 or width > 30:
                continue
            shifted_positions = [(row + top, col + left) for row, col in positions]
            occupied_corners = 0
            for corner_row, corner_col in ((0, 0), (0, width - 1),
                                           (height - 1, 0), (height - 1, width - 1)):
                if any(r <= corner_row < r + h and c <= corner_col < c + w
                       for (r, c), (h, w) in zip(shifted_positions, sizes)):
                    occupied_corners += 1
            if occupied_corners <= 1 and height * width > sum(h * w for h, w in sizes):
                return height, width, shifted_positions

    return None


def _paint_object(
    grid: np.ndarray,
    top: int,
    left: int,
    core_height: int,
    core_width: int,
    thicknesses: tuple[int, ...],
    colors: list[int],
) -> None:
    row = top
    col = left
    height = (
        core_height
        + 2 * sum(thicknesses)
    )
    width = (
        core_width
        + 2 * sum(thicknesses)
    )

    for color, thickness in zip(
        colors[:-1],
        thicknesses,
    ):
        grid[
            row:row + height,
            col:col + width,
        ] = color

        row += thickness
        col += thickness
        height -= 2 * thickness
        width -= 2 * thickness

    grid[
        row:row + core_height,
        col:col + core_width,
    ] = colors[-1]


def _make_scene(
    rng: np.random.Generator,
    layer_counts: list[int],
    color_sequences: list[list[int]],
    background: int,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        specifications = []
        sizes = []

        for layer_count, colors in zip(
            layer_counts,
            color_sequences,
        ):
            thicknesses = tuple(
                int(rng.choice((1, 2, 3), p=(0.69, 0.25, 0.06)))
                for _ in range(layer_count - 1)
            )
            core_values = (1, 2, 3, 4, 5, 6, 7, 8, 9)
            core_weights = (0.15, 0.17, 0.17, 0.15, 0.13,
                            0.09, 0.06, 0.05, 0.03)
            core_height = int(rng.choice(core_values, p=core_weights))
            core_width = int(rng.choice(core_values, p=core_weights))

            height = (
                core_height
                + 2 * sum(thicknesses)
            )
            width = (
                core_width
                + 2 * sum(thicknesses)
            )

            specifications.append(
                (
                    core_height,
                    core_width,
                    thicknesses,
                    colors,
                )
            )
            sizes.append((height, width))

        gap = int(rng.integers(1, 5))
        placement = _place_rectangles(
            sizes,
            rng,
            gap,
        )

        if placement is None:
            continue

        grid_height, grid_width, positions = placement
        input_grid = np.full(
            (grid_height, grid_width),
            background, dtype=np.int8,
        )
        output_grid = np.full(
            (grid_height, grid_width),
            background, dtype=np.int8,
        )

        for position, specification in zip(
            positions,
            specifications,
        ):
            top, left = position
            (
                core_height,
                core_width,
                thicknesses,
                colors,
            ) = specification

            _paint_object(
                input_grid,
                top,
                left,
                core_height,
                core_width,
                thicknesses,
                colors,
            )
            _paint_object(
                output_grid,
                top,
                left,
                core_height,
                core_width,
                thicknesses,
                list(reversed(colors)),
            )

        return input_grid, output_grid

    raise RuntimeError(
        "could not fit a generated scene into a 30 by 30 grid"
    )


def _build_pair(
    pair_seed: int,
    pair_index: int,
    regime: str,
    train_palette: tuple[int, ...],
    full_palette: tuple[int, ...],
    anchor_palette: tuple[int, ...],
    anchor_counts: tuple[int, int],
    known_counts: tuple[int, ...],
    salt: int,
) -> dict:
    regime_code = (
        1 if regime == "source_train" else 2
    )
    seed_sequence = np.random.SeedSequence(
        [
            int(pair_seed),
            int(pair_index),
            regime_code,
            int(salt),
        ]
    )
    rng = np.random.default_rng(seed_sequence)

    if regime == "source_train":
        if pair_index == 0:
            # The first example rules out a global color substitution.
            layer_counts = list(anchor_counts)
            color_sequences = _sample_covering_sequences(
                rng, layer_counts, anchor_palette,
            )
        elif pair_index == 1:
            missing_count = ({3, 4, 5} - set(anchor_counts)).pop()
            object_count = int(rng.integers(1, 4))
            layer_counts = [missing_count] + [
                int(rng.choice((2, 3, 4, 5, 6), p=(0.08, 0.28, 0.30, 0.28, 0.06)))
                for _ in range(object_count - 1)]
            required = tuple(color for color in train_palette
                             if color not in anchor_palette)
            color_sequences = _sample_plain_sequences(
                rng, layer_counts, train_palette, required)
        else:
            object_count = int(rng.integers(1, 4))
            layer_counts = [
                int(rng.choice((2, 3, 4, 5, 6), p=(0.10, 0.26, 0.29, 0.26, 0.09)))
                for _ in range(object_count)
            ]
            selected = tuple(int(color) for color in rng.choice(
                train_palette,
                size=int(rng.integers(min(len(train_palette), max(4, max(layer_counts))),
                                      len(train_palette) + 1)),
                replace=False))
            color_sequences = _sample_plain_sequences(
                rng, layer_counts, selected,
            )
    elif regime == "target_test":
        object_count = int(rng.integers(1, 4))
        # Two and six bands appear in test only after training has shown them.
        choices = tuple(sorted(set(known_counts)))
        layer_counts = [int(rng.choice(choices)) for _ in range(object_count)]
        allowed = list(train_palette)
        novel = [color for color in full_palette if color not in train_palette]
        if novel and rng.random() < 0.58:
            allowed.extend(int(color) for color in rng.choice(
                novel, size=int(rng.integers(1, len(novel) + 1)),
                replace=False))
        selected = tuple(int(color) for color in rng.choice(
            allowed,
            size=int(rng.integers(min(len(allowed), max(4, max(layer_counts))),
                                  len(allowed) + 1)),
            replace=False))
        color_sequences = _sample_plain_sequences(rng, layer_counts, selected)
    else:
        raise ValueError("unknown pair regime")

    unused = [color for color in range(1, 10)
              if color not in full_palette]
    background = (int(rng.choice(unused))
                  if unused and rng.random() < 0.18 else 0)
    input_grid, output_grid = _make_scene(
        rng, layer_counts, color_sequences, background,
    )

    return {
        "input": input_grid,
        "output": output_grid,
    }


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.asarray(
        grid,
        dtype=np.int16,
    )
    return (
        normalized.shape,
        normalized.tobytes(),
    )


def _components(
    grid: np.ndarray,
    background: int,
) -> list[tuple[tuple[int, int, int, int], int]]:
    height, width = grid.shape
    visited = np.zeros(
        grid.shape,
        dtype=bool,
    )
    components = []

    for start_row in range(height):
        for start_col in range(width):
            if (
                grid[start_row, start_col] == background
                or visited[start_row, start_col]
            ):
                continue

            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            head = 0

            min_row = max_row = start_row
            min_col = max_col = start_col
            cell_count = 0

            while head < len(stack):
                row, col = stack[head]
                head += 1
                cell_count += 1

                min_row = min(min_row, row)
                max_row = max(max_row, row)
                min_col = min(min_col, col)
                max_col = max(max_col, col)

                neighbors = (
                    (row - 1, col),
                    (row + 1, col),
                    (row, col - 1),
                    (row, col + 1),
                )

                for next_row, next_col in neighbors:
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and not visited[next_row, next_col]
                        and grid[next_row, next_col] != background
                    ):
                        visited[next_row, next_col] = True
                        stack.append(
                            (next_row, next_col)
                        )

            components.append(
                (
                    (
                        min_row,
                        max_row,
                        min_col,
                        max_col,
                    ),
                    cell_count,
                )
            )

    components.sort(
        key=lambda item: (
            item[0][0],
            item[0][2],
            item[0][1],
            item[0][3],
        )
    )
    return components


def _shell_profile(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    background: int,
) -> tuple[list[int], list[int], list[int]]:
    min_row, max_row, min_col, max_col = bounds

    shell_colors = []
    object_height = max_row - min_row + 1
    object_width = max_col - min_col + 1
    depth_count = (
        min(object_height, object_width) + 1
    ) // 2

    for depth in range(depth_count):
        subgrid = grid[
            min_row + depth:max_row - depth + 1,
            min_col + depth:max_col - depth + 1,
        ]

        if (
            subgrid.shape[0] == 1
            or subgrid.shape[1] == 1
        ):
            perimeter = subgrid.ravel()
        else:
            perimeter = np.concatenate(
                (
                    subgrid[0, :],
                    subgrid[-1, :],
                    subgrid[1:-1, 0],
                    subgrid[1:-1, -1],
                )
            )

        unique = np.unique(perimeter)
        _require(
            unique.size == 1,
            "a geometric shell is not uniformly colored",
        )

        color = int(unique[0])
        _require(
            color != background,
            "an object shell contains background",
        )
        shell_colors.append(color)

    band_colors: list[int] = []
    run_lengths: list[int] = []

    for color in shell_colors:
        if (
            not band_colors
            or band_colors[-1] != color
        ):
            band_colors.append(color)
            run_lengths.append(1)
        else:
            run_lengths[-1] += 1

    return (
        shell_colors,
        band_colors,
        run_lengths,
    )


def _write_shell(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    depth: int,
    color: int,
) -> None:
    min_row, max_row, min_col, max_col = bounds

    top = min_row + depth
    bottom = max_row - depth
    left = min_col + depth
    right = max_col - depth

    grid[top, left:right + 1] = color
    grid[bottom, left:right + 1] = color
    grid[top:bottom + 1, left] = color
    grid[top:bottom + 1, right] = color


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    _require(
        input_grid.shape == output_grid.shape,
        "input/output dimensions changed",
    )
    corners = [int(input_grid[row, col]) for row, col in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    background = max(set(corners), key=corners.count)
    _require(corners.count(background) >= 3,
             "background is not identifiable at the corners")
    _require(np.any(input_grid == background), "background is missing")
    _require(
        np.array_equal(
            input_grid != background,
            output_grid != background,
        ),
        "foreground geometry changed",
    )
    _require(
        set(int(color) for color in np.unique(input_grid))
        == set(int(color) for color in np.unique(output_grid)),
        "the color inventory changed",
    )

    components = _components(input_grid, background)
    _require(
        len(components) >= 1,
        "a pair contains no object",
    )

    expected = input_grid.copy()
    objects = []

    for bounds, cell_count in components:
        (
            min_row,
            max_row,
            min_col,
            max_col,
        ) = bounds

        rectangle_area = (
            (max_row - min_row + 1)
            * (max_col - min_col + 1)
        )
        _require(
            cell_count == rectangle_area,
            "an object is not a filled rectangle",
        )

        (
            shell_colors,
            band_colors,
            run_lengths,
        ) = _shell_profile(
            input_grid,
            bounds,
            background,
        )
        _require(2 <= len(band_colors) <= 6,
                 "an object has an invalid number of bands")

        band_at_depth = []
        for band_index, run_length in enumerate(
            run_lengths
        ):
            band_at_depth.extend(
                [band_index] * run_length
            )

        _require(
            len(band_at_depth) == len(shell_colors),
            "invalid band run-length decomposition",
        )

        reversed_colors = list(
            reversed(band_colors)
        )

        for depth, band_index in enumerate(
            band_at_depth
        ):
            _write_shell(
                expected,
                bounds,
                depth,
                reversed_colors[band_index],
            )

        objects.append({
            "bounds": bounds,
            "colors": band_colors,
            "runs": run_lengths,
            "core": (
                max_row - min_row + 1 - 2 * sum(run_lengths[:-1]),
                max_col - min_col + 1 - 2 * sum(run_lengths[:-1]),
            ),
        })

    _require(
        np.array_equal(expected, output_grid),
        "output is not a local depth reversal",
    )
    _require(
        not np.array_equal(input_grid, output_grid),
        "pair is an unchanged degeneracy",
    )

    return {"objects": objects, "background": background}


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "illegal pair counts")
    input_keys = set()
    pair_keys = set()
    analyses = {"train": [], "test": []}
    palettes = {"train": [], "test": []}

    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grids must be two-dimensional NumPy arrays")
                _require(1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30,
                         "grid size out of bounds")
                _require(np.issubdtype(grid.dtype, np.integer)
                         and bool(np.all((grid >= 0) & (grid <= 9))),
                         "grid cells must be ARC integers")
            _require(_source_key(pair["input"]) not in _SOURCE_INPUT_DIGESTS,
                     "exact source input copied")
            input_key = _grid_key(pair["input"])
            pair_key = (input_key, _grid_key(pair["output"]))
            _require(input_key not in input_keys and pair_key not in pair_keys,
                     "duplicate grid or pair")
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            analysis = _analyze_pair(pair["input"], pair["output"])
            _require(1 <= len(analysis["objects"]) <= 3,
                     "scene has an invalid object count")
            analyses[split].append(analysis)
            palettes[split].append(_foreground_colors(
                pair["input"], analysis["background"]))

    anchor = analyses["train"][0]["objects"]
    _require(len(anchor) == 2, "first demonstration needs two objects")
    sequences = [obj["colors"] for obj in anchor]
    _require(len({len(seq) for seq in sequences}) == 2
             and {len(seq) for seq in sequences} <= {3, 4, 5},
             "first demonstration needs distinct familiar depths")
    _require(_has_reassigned_depth(sequences)
             and _has_local_mapping_conflict(sequences),
             "training does not refute a global color map")
    train_depths = {len(obj["colors"])
                    for pair in analyses["train"] for obj in pair["objects"]}
    _require({3, 4, 5} <= train_depths,
             "training must demonstrate three, four and five bands")
    _require(bool(palettes["train"][1] - palettes["train"][0]),
             "later training does not introduce a color")
    for pair in analyses["test"]:
        _require(all(len(obj["colors"]) in train_depths
                     for obj in pair["objects"]),
                 "test depth was not demonstrated in training")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    shuffled = tuple(int(color) for color in rng.permutation(np.arange(1, 10)))
    full_size = int(rng.integers(6, 9))
    novel_count = int(rng.integers(0, min(2, full_size - 5) + 1))
    full_palette = shuffled[:full_size]
    train_palette = full_palette[:full_size - novel_count]
    anchor_size = int(rng.integers(4, min(5, len(train_palette) - 1) + 1))
    anchor_palette = train_palette[:anchor_size]
    anchor_counts = tuple(int(n) for n in rng.choice((3, 4, 5),
                                                     size=2, replace=False))
    pair_seeds = [int(value) for value in rng.integers(
        0, np.iinfo(np.int64).max, size=train_count + test_count)]
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    known_counts = set(anchor_counts)
    cursor = 0
    for split, count, regime in (("train", train_count, "source_train"),
                                 ("test", test_count, "target_test")):
        for pair_index in range(count):
            pair_seed = pair_seeds[cursor]
            cursor += 1
            for salt in range(120):
                try:
                    candidate = _build_pair(
                        pair_seed, pair_index, regime, train_palette,
                        full_palette, anchor_palette, anchor_counts,
                        tuple(known_counts), salt)
                    key = _grid_key(candidate["input"])
                    if key in used_inputs or _source_key(candidate["input"]) in _SOURCE_INPUT_DIGESTS:
                        continue
                    analysis = _analyze_pair(candidate["input"], candidate["output"])
                except (AssertionError, RuntimeError):
                    continue
                puzzle[split].append(candidate)
                used_inputs.add(key)
                if split == "train":
                    known_counts.update(len(obj["colors"])
                                        for obj in analysis["objects"])
                break
            else:
                raise RuntimeError("could not construct a distinct valid scene")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False
            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _puzzle_fingerprint(
    puzzle: dict,
) -> tuple:
    return tuple(
        (
            split,
            _grid_key(pair["input"]),
            _grid_key(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    fingerprints, layouts, shapes, palette_profiles = set(), set(), set(), set()
    train_objects, test_objects, depths = set(), set(), set()
    anchor_combinations, touched_sides = set(), set()
    thin = long_core = thick_three = edge = nonzero_background = 0
    late_train_color = test_no_novel = repeated_band = 0
    combined = thin_and_thick_pair = source_shape_pairs = corner_objects = 0
    within_palette = mixed_counts = busier_test = 0
    source_shapes = {(17, 17), (18, 16), (17, 19), (18, 19), (20, 19)}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_puzzle_fingerprint(puzzle))
        train_colors, test_colors = set(), set()
        pair_features = []
        pair_palettes = set()
        train_object_counts, test_object_counts = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                analysis = _analyze_pair(pair["input"], pair["output"])
                objects = analysis["objects"]
                (train_objects if split == "train" else test_objects).add(len(objects))
                (train_object_counts if split == "train" else test_object_counts).append(len(objects))
                shapes.add(pair["input"].shape)
                source_shape_pairs += pair["input"].shape in source_shapes
                colors = _foreground_colors(pair["input"], analysis["background"])
                pair_palettes.add(frozenset(colors))
                (train_colors if split == "train" else test_colors).update(colors)
                nonzero_background += analysis["background"] != 0
                seen_thin = seen_long = seen_thick = seen_edge = False
                core_minima = []
                for obj in objects:
                    depths.add(len(obj["colors"]))
                    core_height, core_width = obj["core"]
                    core_minima.append(min(core_height, core_width))
                    seen_thin |= min(core_height, core_width) == 1
                    seen_long |= max(core_height, core_width) >= 6
                    seen_thick |= any(run >= 3 for run in obj["runs"][:-1])
                    repeated_band += len(set(obj["colors"])) < len(obj["colors"])
                    top, bottom, left, right = obj["bounds"]
                    sides = set()
                    if top == 0:
                        sides.add("top")
                    if bottom == pair["input"].shape[0] - 1:
                        sides.add("bottom")
                    if left == 0:
                        sides.add("left")
                    if right == pair["input"].shape[1] - 1:
                        sides.add("right")
                    touched_sides.update(sides)
                    corner_objects += len(sides) >= 2
                    seen_edge |= (top == 0 or left == 0 or
                                  bottom == pair["input"].shape[0] - 1 or
                                  right == pair["input"].shape[1] - 1)
                thin_and_thick_pair += (any(value == 1 for value in core_minima)
                                       and any(value >= 2 for value in core_minima))
                thin += seen_thin
                long_core += seen_long
                thick_three += seen_thick
                edge += seen_edge
                pair_features.append((seen_thin, seen_long, seen_thick, seen_edge))
                layouts.add((pair["input"].shape,
                             tuple((obj["bounds"], tuple(obj["runs"]))
                                   for obj in objects)))
        first = _analyze_pair(puzzle["train"][0]["input"],
                              puzzle["train"][0]["output"])
        anchor_combinations.add(tuple(sorted(len(obj["colors"])
                                             for obj in first["objects"])))
        second = _analyze_pair(puzzle["train"][1]["input"],
                               puzzle["train"][1]["output"])
        late_train_color += bool(_foreground_colors(puzzle["train"][1]["input"],
                                                      second["background"])
                                 - _foreground_colors(puzzle["train"][0]["input"],
                                                      first["background"]))
        test_no_novel += all(_foreground_colors(pair["input"],
                                _analyze_pair(pair["input"], pair["output"])["background"])
                             <= train_colors for pair in puzzle["test"])
        combined += any(sum(features) >= 2 for features in pair_features)
        within_palette += len(pair_palettes) > 1
        mixed_counts += len(set(train_object_counts + test_object_counts)) > 1
        busier_test += max(test_object_counts) > max(train_object_counts)
        palette_profiles.add((tuple(sorted(train_colors)),
                              tuple(sorted(test_colors - train_colors))))
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack range")
    _require(len(fingerprints) >= 40 and len(layouts) >= 160
             and len(shapes) >= 25 and len(palette_profiles) >= 18,
             "insufficient structural or palette diversity")
    _require(train_objects == {1, 2, 3} and test_objects == {1, 2, 3}
             and depths == {2, 3, 4, 5, 6}, "object or band count diversity missing")
    _require(min(thin, long_core, thick_three, edge, nonzero_background) >= 5,
             "core, band, edge or background variation missing")
    _require(late_train_color == 42 and test_no_novel >= 3
             and repeated_band >= 3 and combined >= 20,
             "palette schedule or combined variation missing")
    _require(anchor_combinations == {(3, 4), (3, 5), (4, 5)}
             and touched_sides == {"top", "bottom", "left", "right"}
             and source_shape_pairs >= 1 and corner_objects >= 5
             and thin_and_thick_pair >= 10,
             "source-supported geometry and template variation missing")
    _require(within_palette == 42 and mixed_counts == 42 and busier_test >= 1,
             "within-puzzle palette, count or test regime variation missing")
    first = generate_puzzle(777, 6, 3)
    _require(_puzzles_equal(first, generate_puzzle(777, 6, 3))
             and not _puzzles_equal(first, generate_puzzle(778, 6, 3)),
             "seed determinism or sensitivity failed")
    for seed, train, test in ((9001, 2, 1), (9002, 10, 1), (9003, 10, 4),
                              (9101, None, 1), (9102, None, 4),
                              (9103, 2, None), (9104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _require((train is None or len(puzzle["train"]) == train)
                 and (test is None or len(puzzle["test"]) == test),
                 "explicit or partial count ignored")
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(9200, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
