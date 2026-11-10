import numpy as np


_TRAIN_LINES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
)

_TRAIN_RECTS = (
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)),
)

_TRAIN_ANGLES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)),
)

_TRAIN_OTHER = (
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1)),
)

# Each crown is connected, at least four rows high, and has separated
# rack-facing teeth in its first row.
_CROWNS = (
    (
        (0, 0), (0, 2), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 2), (2, 3),
        (3, 2),
    ),
    (
        (0, 0), (0, 2), (0, 4),
        (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
        (2, 1), (2, 3),
        (3, 1), (3, 2), (3, 3),
    ),
    (
        (0, 0), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
        (2, 1), (2, 2),
        (3, 0), (3, 1), (3, 2),
    ),
    (
        (0, 0), (0, 2), (0, 3), (0, 5),
        (1, 0), (1, 1), (1, 2),
        (1, 3), (1, 4), (1, 5),
        (2, 1), (2, 2), (2, 3), (2, 4),
        (3, 2), (3, 3),
    ),
)

_TARGET_OTHER = (
    (
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 2),
        (2, 0), (2, 1), (2, 2),
    ),
    (
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 1), (2, 2),
        (3, 2),
    ),
    (
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 1), (1, 2), (1, 3),
    ),
    (
        (0, 0),
        (1, 0), (1, 1),
        (2, 0), (2, 1), (2, 2),
        (3, 2),
    ),
    (
        (0, 0), (0, 1), (0, 2),
        (1, 1),
        (2, 0), (2, 1), (2, 2),
    ),
    (
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0),
        (3, 0), (3, 1),
    ),
    (
        (0, 0), (0, 1), (0, 2),
        (1, 0),
        (2, 0), (2, 1), (2, 2),
        (3, 2),
    ),
)

_SOURCE_PAIRS = (
    (
        (
            "222222222222222",
            "222202220220022",
            "200202200020022",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "055000000000000",
            "055000000050000",
            "555500050050055",
            "055000555050555",
        ),
        (
            "222222222222222",
            "222212221221122",
            "211212211121122",
            "111010000011110",
            "000000000001100",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
        ),
    ),
    (
        (
            "222222222222222",
            "202222222222220",
            "200222000222220",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000000000050",
            "000000500000050",
            "055500550000050",
            "055500555000050",
        ),
        (
            "222222222222222",
            "212222222222221",
            "211222111222221",
            "011100111000001",
            "000000000000001",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
        ),
    ),
    (
        (
            "222222222222222",
            "202222020220222",
            "200222000220002",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000050005050",
            "050000055005550",
            "055500050005550",
            "055550055005550",
        ),
        (
            "222222222222222",
            "212222121221222",
            "211222111221112",
            "010000111001111",
            "011000111000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
        ),
    ),
)


def _normalise(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_height(shape):
    return max(r for r, _ in shape) + 1


def _shape_width(shape):
    return max(c for _, c in shape) + 1


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    result = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            seen[r, c] = True
            stack = [(r, c)]
            component = []

            while stack:
                y, x = stack.pop()
                component.append((y, x))

                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    yy, xx = y + dy, x + dx
                    if (
                        0 <= yy < h
                        and 0 <= xx < w
                        and mask[yy, xx]
                        and not seen[yy, xx]
                    ):
                        seen[yy, xx] = True
                        stack.append((yy, xx))

            result.append(tuple(sorted(component)))

    result.sort(key=lambda component: component[0])
    return result


def _has_joined_teeth(shape):
    shape = _normalise(shape)
    if _shape_height(shape) < 4:
        return False

    top_columns = sorted(c for r, c in shape if r == 0)
    return (
        len(top_columns) >= 2
        and any(
            b - a > 1
            for a, b in zip(top_columns, top_columns[1:])
        )
    )


def _find_solutions(
    shapes,
    height,
    width,
    rack_depth,
    blocked,
    holes,
    limit=2,
):
    """Find translation-only exact covers, stopping after `limit` solutions."""
    holes = frozenset(holes)
    blocked = frozenset(blocked)

    candidates = []
    covering = {cell: [] for cell in holes}

    for piece_index, raw_shape in enumerate(shapes):
        shape = _normalise(raw_shape)
        shape_h = _shape_height(shape)
        shape_w = _shape_width(shape)
        piece_candidates = []

        # top == 0 would intersect the solid cap. A top greater than
        # rack_depth cannot intersect any rack hole.
        for top in range(1, rack_depth + 1):
            if top + shape_h > height:
                continue

            for left in range(width - shape_w + 1):
                cells = frozenset(
                    (top + r, left + c)
                    for r, c in shape
                )
                if cells & blocked:
                    continue

                cover = frozenset(
                    (r, c)
                    for r, c in cells
                    if 1 <= r <= rack_depth
                )
                if not cover or not cover.issubset(holes):
                    continue

                candidate_index = len(piece_candidates)
                piece_candidates.append((cover, cells))
                for cell in cover:
                    covering[cell].append(
                        (piece_index, candidate_index)
                    )

        candidates.append(piece_candidates)

    if any(not piece_candidates for piece_candidates in candidates):
        return []
    if any(not covering[cell] for cell in holes):
        return []

    solutions = []
    full_unused_mask = (1 << len(shapes)) - 1

    def visit(unused, covered, occupied, chosen):
        if len(solutions) >= limit:
            return

        if covered == holes:
            if unused == 0:
                union = frozenset().union(
                    *(cells for _, cells in chosen)
                )
                assignment = tuple(
                    sorted(
                        (piece_index, tuple(sorted(cells)))
                        for piece_index, cells in chosen
                    )
                )
                solutions.append((assignment, union))
            return

        best_options = None

        # Branch on the uncovered hole with the fewest currently
        # compatible placements.
        for cell in holes - covered:
            options = []

            for piece_index, candidate_index in covering[cell]:
                if not (unused & (1 << piece_index)):
                    continue

                cover, cells = candidates[piece_index][candidate_index]
                if cover & covered or cells & occupied:
                    continue

                options.append((piece_index, cover, cells))

            if not options:
                return

            if best_options is None or len(options) < len(best_options):
                best_options = options
                if len(best_options) == 1:
                    break

        best_options.sort(
            key=lambda item: (
                item[0],
                tuple(sorted(item[1])),
                tuple(sorted(item[2])),
            )
        )

        for piece_index, cover, cells in best_options:
            visit(
                unused & ~(1 << piece_index),
                covered | cover,
                occupied | cells,
                chosen + ((piece_index, cells),),
            )
            if len(solutions) >= limit:
                return

    visit(
        full_unused_mask,
        frozenset(),
        frozenset(),
        tuple(),
    )
    return solutions


def _local_match_stats(shapes, height, width, depth, blocked, holes):
    """Measure whether isolated hole components determine their piece."""
    mask = np.zeros((height, width), dtype=bool)
    for r, c in holes:
        mask[r, c] = True
    groups = [frozenset(group) for group in _components(mask)]
    any_competition = False
    any_joint = False
    for group in groups:
        matching_pieces = set()
        for index, shape in enumerate(shapes):
            for top in range(1, depth + 1):
                if top + _shape_height(shape) > height:
                    continue
                for left in range(width - _shape_width(shape) + 1):
                    cells = {(top + r, left + c) for r, c in shape}
                    if cells & blocked:
                        continue
                    cover = frozenset((r, c) for r, c in cells if r <= depth)
                    if cover == group:
                        matching_pieces.add(index)
        any_competition |= len(matching_pieces) > 1
        any_joint |= not matching_pieces
    return any_competition, any_joint


_SIMPLE_SHAPES = tuple(
    _normalise(shape)
    for group in (_TRAIN_LINES, _TRAIN_RECTS, _TRAIN_ANGLES, _TRAIN_OTHER)
    for shape in group
)


def _grow_shape(rng, size):
    """Grow connected polyominoes without restricting them to a bank."""
    cells = {(0, 0)}
    for _ in range(size - 1):
        border = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (r + dr, c + dc) not in cells
        }
        choices = sorted(border)
        cells.add(choices[int(rng.integers(len(choices)))])
    return _normalise(cells)


def _choose_shapes(rng, count, regime, substantial):
    shapes = []
    for index in range(count):
        for _ in range(80):
            if substantial and index == 0:
                size = int(rng.integers(7, 11))
                shape = _grow_shape(rng, size)
            elif regime == "target_test" and index == 0 and rng.random() < 0.23:
                shape = _normalise(_CROWNS[int(rng.integers(len(_CROWNS)))])
            elif rng.random() < 0.38:
                shape = _SIMPLE_SHAPES[int(rng.integers(len(_SIMPLE_SHAPES)))]
            else:
                size = int(rng.integers(3, 10 if regime == "source_train" else 12))
                shape = _grow_shape(rng, size)
            if (
                shape not in shapes
                and _shape_height(shape) <= 5
                and _shape_width(shape) <= 7
            ):
                shapes.append(shape)
                break
        else:
            return None
    return shapes


def _stage_sources(rng, shapes, width, start):
    """Pack source boxes on one or two uneven rows below the rack."""
    count = len(shapes)
    order = [int(x) for x in rng.permutation(count)]
    rows = [[]]
    cursor = 0
    for index in order:
        gap = int(rng.integers(1, 4)) if rows[-1] else 0
        needed = _shape_width(shapes[index]) + gap
        if cursor + needed > width:
            rows.append([])
            cursor = 0
            gap = 0
        rows[-1].append((index, cursor + gap))
        cursor += needed
    if len(rows) > 2:
        return None

    placements = [None] * count
    row_top = start
    for row in rows:
        span = max(left + _shape_width(shapes[index]) for index, left in row)
        slack = width - span
        if rng.random() < 0.35:
            offset = 0 if rng.random() < 0.5 else slack
        else:
            offset = int(rng.integers(slack + 1))
        row_height = max(_shape_height(shapes[index]) for index, _ in row)
        for index, left in row:
            vertical_offset = int(rng.integers(row_height - _shape_height(shapes[index]) + 1))
            top = row_top + vertical_offset
            placements[index] = frozenset(
                (top + r, offset + left + c) for r, c in shapes[index]
            )
        row_top += row_height + int(rng.integers(2, 5))

    bottom = max(r for cells in placements for r, _ in cells)
    bottom_padding = int(rng.choice((0, 0, 0, 1, 2, 3)))
    height = bottom + 1 + bottom_padding
    if height > 30:
        return None
    return placements, height


def _make_pair(rng, regime, puzzle_parameters, piece_count, substantial=False,
               overhang_mode=None, prefer_edge=False, prefer_join=False,
               forced_orientation=None, forced_depth=None, compact=False):
    background, rack, source, result = puzzle_parameters["colors"]

    for _ in range(700):
        rack_depth = forced_depth if forced_depth is not None else int(rng.integers(2, 4))
        orientation = forced_orientation if forced_orientation is not None else (
            int(rng.integers(4)) if puzzle_parameters["mix_orientation"]
            else puzzle_parameters["orientation"]
        )
        shapes = _choose_shapes(rng, piece_count, regime, substantial)
        if shapes is None:
            continue
        if overhang_mode == "all" and any(_shape_height(shape) == 1 for shape in shapes):
            continue
        if overhang_mode == "none":
            pool = [shape for shape in _SIMPLE_SHAPES if _shape_height(shape) <= rack_depth]
            if len(pool) < piece_count:
                continue
            chosen = rng.choice(len(pool), size=piece_count, replace=False)
            shapes = [pool[int(i)] for i in chosen]
        elif overhang_mode == "one":
            pool = [
                shape for shape in _SIMPLE_SHAPES
                if _shape_height(shape) <= rack_depth and shape != shapes[0]
            ]
            if _shape_height(shapes[0]) <= rack_depth or len(pool) < piece_count - 1:
                continue
            chosen = rng.choice(len(pool), size=piece_count - 1, replace=False)
            shapes = [shapes[0]] + [pool[int(i)] for i in chosen]
        order = [int(x) for x in rng.permutation(piece_count)]
        gaps = [int(rng.choice((0, 0, 1, 1, 2, 3, 4))) for _ in range(piece_count - 1)]
        if prefer_join and gaps:
            gaps[int(rng.integers(len(gaps)))] = 0
        span = sum(_shape_width(shapes[i]) for i in order) + sum(gaps)
        if span > 29:
            continue
        if compact:
            if span > 17:
                continue
            width = int(rng.integers(max(span, 13), 18))
            extra = width - span
        else:
            extra = int(rng.integers(0, min(31 - span, 8)))
            width = span + extra
        if width < 8:
            continue
        if prefer_edge or rng.random() < 0.26:
            left = 0 if rng.random() < 0.5 else extra
        else:
            left = int(rng.integers(extra + 1))

        anchors = []
        for index, shape in enumerate(shapes):
            choices = list(range(1, rack_depth + 1))
            if overhang_mode == "all":
                choices = [top for top in choices if top + _shape_height(shape) - 1 > rack_depth]
            elif overhang_mode == "none":
                choices = [top for top in choices if top + _shape_height(shape) - 1 <= rack_depth]
            elif overhang_mode == "one" and index > 0:
                choices = [top for top in choices if top + _shape_height(shape) - 1 <= rack_depth]
            anchors.append(int(choices[int(rng.integers(len(choices)))]) if choices else None)
        if None in anchors:
            continue
        if overhang_mode == "one":
            projecting = [i for i, shape in enumerate(shapes) if anchors[i] + _shape_height(shape) - 1 > rack_depth]
            if len(projecting) != 1:
                continue

        destinations = [None] * piece_count
        cursor = left
        for position, index in enumerate(order):
            destinations[index] = frozenset((anchors[index] + r, cursor + c) for r, c in shapes[index])
            cursor += _shape_width(shapes[index])
            if position < piece_count - 1:
                cursor += gaps[position]
        destination = frozenset().union(*destinations)
        holes = frozenset((r, c) for r, c in destination if 1 <= r <= rack_depth)
        if not holes or any(
            not 0 < sum(r == row for r, _ in holes) < width
            for row in range(1, rack_depth + 1)
        ):
            continue
        if prefer_edge and not any(c in (0, width - 1) for _, c in holes):
            continue

        stage_start = max(rack_depth + int(rng.integers(3, 6)), max(r for r, _ in destination) + 2)
        staged = _stage_sources(rng, shapes, width, stage_start)
        if staged is None:
            continue
        source_by_piece, height = staged
        if height < 8 or (compact and height > 13):
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        input_grid[:rack_depth + 1, :] = rack
        for r, c in holes:
            input_grid[r, c] = background
        for cells in source_by_piece:
            for r, c in cells:
                input_grid[r, c] = source
        if len(_components(input_grid == source)) != piece_count:
            continue

        output_grid = input_grid.copy()
        output_grid[output_grid == source] = background
        for r, c in destination:
            output_grid[r, c] = result
        blocked = frozenset((int(r), int(c)) for r, c in np.argwhere(input_grid == rack))
        observed = [_normalise(cells) for cells in _components(input_grid == source)]
        solutions = _find_solutions(observed, height, width, rack_depth, blocked, holes, limit=2)
        if len(solutions) != 1 or solutions[0][1] != destination:
            continue
        return np.rot90(input_grid, orientation).copy(), np.rot90(output_grid, orientation).copy()
    if compact:
        return _make_pair(
            rng, regime, puzzle_parameters, piece_count,
            substantial=substantial, overhang_mode=overhang_mode,
            prefer_edge=prefer_edge, prefer_join=prefer_join,
            forced_orientation=forced_orientation, forced_depth=forced_depth,
            compact=False,
        )
    raise RuntimeError("could not construct a unique exact-cover pair")


def _valid_count(value, name, minimum, maximum):
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
    ):
        raise ValueError(f"{name} must be an integer")

    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is out of range")

    return value


def _resolve_counts(rng, num_train, num_test):
    if num_train is not None:
        num_train = _valid_count(
            num_train,
            "num_train",
            2,
            10,
        )
    if num_test is not None:
        num_test = _valid_count(
            num_test,
            "num_test",
            1,
            4,
        )

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    elif num_train is None:
        num_train = int(
            rng.integers(max(2, num_test + 1), 11)
        )
    else:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    return num_train, num_test


def _same_as_rows(grid, rows):
    if grid.shape != (len(rows), len(rows[0])):
        return False

    return all(
        "".join(str(int(value)) for value in grid[r]) == rows[r]
        for r in range(len(rows))
    )


def _analyse_pair(input_grid, output_grid):
    assert isinstance(input_grid, np.ndarray)
    assert isinstance(output_grid, np.ndarray)
    assert input_grid.ndim == 2
    assert output_grid.ndim == 2
    assert input_grid.shape == output_grid.shape

    input_colors = set(int(x) for x in np.unique(input_grid))
    output_colors = set(int(x) for x in np.unique(output_grid))

    assert len(input_colors) == 3
    assert len(output_colors) == 3

    source_only = input_colors - output_colors
    result_only = output_colors - input_colors
    common = input_colors & output_colors

    assert len(source_only) == 1
    assert len(result_only) == 1
    assert len(common) == 2

    source = source_only.pop()
    result = result_only.pop()

    # Identify the rack color and the rotation that places its cap at
    # the top. Background may form a solid outer row too, so the rack
    # must also be confined to a shallow perforated boundary band.
    rack_options = []

    for color in sorted(common):
        for rotation in range(4):
            trial = np.rot90(input_grid, rotation)

            if not np.all(trial[0, :] == color):
                continue

            color_rows = np.argwhere(trial == color)[:, 0]
            rack_depth = int(color_rows.max())

            if rack_depth not in (2, 3):
                continue

            if all(
                np.any(trial[row, :] == color)
                and np.any(trial[row, :] != color)
                for row in range(1, rack_depth + 1)
            ):
                rack_options.append(
                    (color, rotation, rack_depth)
                )

    assert len(rack_options) == 1

    rack, rotation, rack_depth = rack_options[0]
    background = next(iter(common - {rack}))

    canonical_input = np.rot90(input_grid, rotation)
    canonical_output = np.rot90(output_grid, rotation)
    height, width = canonical_input.shape

    assert np.all(canonical_input[0, :] == rack)
    assert np.all(canonical_output[0, :] == rack)
    assert not np.any(
        canonical_input[rack_depth + 1:, :] == rack
    )
    assert not np.any(
        canonical_output[rack_depth + 1:, :] == rack
    )

    for row in range(1, rack_depth + 1):
        assert set(
            int(x)
            for x in np.unique(canonical_input[row, :])
        ) == {background, rack}
        assert set(
            int(x)
            for x in np.unique(canonical_output[row, :])
        ) == {result, rack}

    assert np.all(
        canonical_output[canonical_input == rack] == rack
    )
    assert np.all(
        canonical_output[canonical_input == source]
        == background
    )
    assert np.all(
        canonical_output[canonical_input == background]
        != source
    )
    assert np.all(
        canonical_input[canonical_output == result]
        == background
    )
    assert np.any(input_grid != output_grid)

    components = _components(canonical_input == source)
    assert components

    source_rows = [
        r
        for component in components
        for r, _ in component
    ]
    first_source_row = min(source_rows)

    assert first_source_row >= rack_depth + 3
    assert np.all(
        canonical_input[
            rack_depth + 1:first_source_row,
            :
        ] == background
    )

    holes = frozenset(
        (r, c)
        for r in range(1, rack_depth + 1)
        for c in range(width)
        if canonical_input[r, c] == background
    )
    assert holes
    assert all(
        canonical_output[r, c] == result
        for r, c in holes
    )

    result_cells = frozenset(
        (int(r), int(c))
        for r, c in np.argwhere(canonical_output == result)
    )
    assert result_cells

    shapes = [
        _normalise(component)
        for component in components
    ]
    payload = sum(len(shape) for shape in shapes)

    assert payload == len(result_cells)

    blocked = frozenset(
        (int(r), int(c))
        for r, c in np.argwhere(canonical_input == rack)
    )
    solutions = _find_solutions(
        shapes,
        height,
        width,
        rack_depth,
        blocked,
        holes,
        limit=2,
    )

    assert len(solutions) == 1
    assert solutions[0][1] == result_cells

    boxes = [
        (min(c for r, c in component), max(c for r, c in component))
        for component in components
    ]
    substantial_shapes = [shape for shape in shapes if 7 <= len(shape) <= 10]
    hole_mask = np.zeros((height, width), dtype=bool)
    for r, c in holes:
        hole_mask[r, c] = True
    owners = {
        cell: index
        for index, cells in solutions[0][0]
        for cell in cells if cell in holes
    }
    hole_components = _components(hole_mask)

    return {
        "roles": (background, rack, source, result),
        "rotation": rotation,
        "depth": rack_depth,
        "piece_count": len(shapes),
        "payload": payload,
        "max_piece": max(len(shape) for shape in shapes),
        "overhang_count": sum(
            any(r > rack_depth for r, c in cells)
            for _, cells in solutions[0][0]
        ),
        "edge_hole": any(c in (0, width - 1) for r, c in holes),
        "hole_groups": len(hole_components),
        "shared_hole_group": any(
            len({owners[cell] for cell in group}) > 1
            for group in hole_components
        ),
        "source_edge": any(
            r == height - 1 or c in (0, width - 1)
            for component in components for r, c in component
        ),
        "source_bottom": any(
            r == height - 1 for component in components for r, c in component
        ),
        "source_sides": any(
            c in (0, width - 1) for component in components for r, c in component
        ),
        "source_multirow": any(
            not (a[1] < b[0] or b[1] < a[0])
            for index, a in enumerate(boxes) for b in boxes[index + 1:]
        ),
        "large_tall": any(_shape_height(shape) >= 4 for shape in substantial_shapes),
        "large_wide": any(_shape_width(shape) >= 4 for shape in substantial_shapes),
        "large_concave": any(
            _shape_height(shape) * _shape_width(shape) > len(shape)
            for shape in substantial_shapes
        ),
        "large_asymmetric": any(
            _normalise((r, _shape_width(shape) - 1 - c) for r, c in shape) != shape
            for shape in substantial_shapes
        ),
        "large_half_hidden": any(
            7 <= len(shapes[index]) <= 10
            and 2 * sum(r <= rack_depth for r, c in cells) <= len(cells)
            for index, cells in solutions[0][0]
        ),
        "has_teeth": any(
            _has_joined_teeth(shape)
            for shape in shapes
        ),
        "shapes": tuple(sorted(shapes)),
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(
        rng,
        num_train,
        num_test,
    )

    # Colors are the only hidden convention: training shows all four roles.
    # The cap itself identifies direction and depth in each input.
    puzzle_parameters = {
        "colors": tuple(
            int(value)
            for value in rng.choice(
                10,
                size=4,
                replace=False,
            )
        ),
        "orientation": int(rng.integers(0, 4)),
        "mix_orientation": bool(rng.random() < 0.55),
        "mix_depth": bool(rng.random() < 0.55),
        "depth": int(rng.integers(2, 4)),
    }

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs = set()

    for regime, count in (
        ("source_train", num_train),
        ("target_test", num_test),
    ):
        for index in range(count):
            child_seed = int(
                rng.integers(
                    0,
                    np.iinfo(np.int64).max,
                    dtype=np.int64,
                )
            )
            pair_rng = np.random.default_rng(child_seed)

            piece_count = (3 if index == 0 and regime == "source_train"
                           else 4 if index == 1 and regime == "source_train"
                           else int(pair_rng.integers(2, 6)))
            if regime == "source_train":
                overhang_mode = "all" if index == 0 else (
                    "none" if index == 1 and num_train > 2 and pair_rng.random() < 0.5
                    else None
                )
            else:
                overhang_mode = pair_rng.choice((None, None, "all", "one", "none"))
            forced_orientation = None
            forced_depth = (
                puzzle_parameters["depth"]
                if not puzzle_parameters["mix_depth"] else None
            )
            if regime == "source_train" and index < 2:
                forced_orientation = puzzle_parameters["orientation"]
                forced_depth = puzzle_parameters["depth"]
                if index == 1:
                    if puzzle_parameters["mix_orientation"]:
                        forced_orientation = (forced_orientation + int(pair_rng.integers(1, 4))) % 4
                    if puzzle_parameters["mix_depth"]:
                        forced_depth = 5 - forced_depth

            for _ in range(30):
                input_grid, output_grid = _make_pair(
                    pair_rng,
                    regime,
                    puzzle_parameters,
                    piece_count,
                    substantial=(regime == "source_train" and index == 0),
                    overhang_mode=overhang_mode,
                    prefer_edge=(index % 3 == 1 or pair_rng.random() < 0.25),
                    prefer_join=(index % 3 == 2 or pair_rng.random() < 0.25),
                    forced_orientation=forced_orientation,
                    forced_depth=forced_depth,
                    compact=(piece_count <= 4 and pair_rng.random() < 0.24),
                )
                signature = (
                    input_grid.shape,
                    input_grid.tobytes(),
                )
                if signature not in seen_inputs:
                    break
            else:
                raise RuntimeError(
                    "could not avoid a duplicate input"
                )

            seen_inputs.add(signature)
            split = (
                "train"
                if regime == "source_train"
                else "test"
            )
            puzzle[split].append(
                {
                    "input": input_grid,
                    "output": output_grid,
                }
            )

    validate_generated_puzzle(puzzle)
    return puzzle


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
    infos = {
        "train": [],
        "test": [],
    }

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            assert input_grid.shape == output_grid.shape
            assert input_grid.dtype == output_grid.dtype

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))

            input_key = (
                input_grid.shape,
                input_grid.dtype.str,
                input_grid.tobytes(),
            )
            pair_key = (
                input_key,
                output_grid.shape,
                output_grid.dtype.str,
                output_grid.tobytes(),
            )

            assert input_key not in input_keys
            assert pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)

            for source_input, source_output in _SOURCE_PAIRS:
                assert not (
                    _same_as_rows(input_grid, source_input)
                    and _same_as_rows(output_grid, source_output)
                )

            infos[split].append(
                _analyse_pair(input_grid, output_grid)
            )

    all_infos = infos["train"] + infos["test"]
    convention = all_infos[0]["roles"]
    assert all(info["roles"] == convention for info in all_infos)
    assert {info["piece_count"] for info in infos["train"]} >= {3, 4}
    assert max(info["max_piece"] for info in infos["train"]) >= 7
    assert any(info["overhang_count"] == info["piece_count"] for info in infos["train"])

    # Every training pair visibly exposes the same four semantic roles,
    # including source-to-background erasure and result recoloring.
    assert {
        info["roles"]
        for info in infos["train"]
    } == {convention}
    assert all(
        len(set(info["roles"])) == 4
        for info in infos["train"]
    )


def _puzzles_equal(first, second):
    if set(first) != set(second):
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


def _fingerprint(puzzle):
    return tuple(
        (
            split,
            pair["input"].shape,
            pair["input"].tobytes(),
            pair["output"].tobytes(),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def self_test() -> None:
    train_counts = set()
    test_counts = set()
    fingerprints = set()
    color_schemes = set()
    rotations = set()
    rack_depths = set()
    dimensions = set()
    shape_inventories = set()
    piece_counts = set()
    coverage = {key: 0 for key in (
        "train_large", "test_familiar_payload", "test_crown",
        "zero_overhang", "one_overhang", "all_overhang",
        "edge_hole", "source_bottom", "source_sides",
        "source_multirow", "compact", "joint", "shared_hole_group", "local_competition",
        "large_tall", "large_wide", "large_concave", "large_asymmetric",
        "large_half_hidden", "combined_edge_joint_overhang",
        "mixed_orientation", "fixed_orientation", "mixed_depth", "fixed_depth",
    )}

    # Forty-two fixed automatic-count seeds exercise generation,
    # validation, regime restrictions, and diversity.
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(_fingerprint(puzzle))

        first_info = _analyse_pair(
            puzzle["train"][0]["input"],
            puzzle["train"][0]["output"],
        )
        color_schemes.add(first_info["roles"])
        rotations.add(first_info["rotation"])
        rack_depths.add(first_info["depth"])

        puzzle_infos = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                dimensions.add(pair["input"].shape)
                info = _analyse_pair(pair["input"], pair["output"])
                puzzle_infos[split].append(info)
                piece_counts.add(info["piece_count"])
                coverage["zero_overhang"] += info["overhang_count"] == 0
                coverage["one_overhang"] += info["overhang_count"] == 1
                coverage["all_overhang"] += info["overhang_count"] == info["piece_count"]
                for key in ("edge_hole", "source_bottom", "source_sides", "source_multirow"):
                    coverage[key] += info[key]
                if split == "train":
                    for key in ("large_tall", "large_wide", "large_concave",
                                "large_asymmetric", "large_half_hidden"):
                        coverage[key] += info[key]
                canonical = np.rot90(pair["input"], info["rotation"])
                h, w = canonical.shape
                coverage["compact"] += 9 <= h <= 13 and 13 <= w <= 17
                background, rack, source, _ = info["roles"]
                blocked = {(int(r), int(c)) for r, c in np.argwhere(canonical == rack)}
                holes = {
                    (r, c) for r in range(1, info["depth"] + 1)
                    for c in range(w) if canonical[r, c] == background
                }
                shapes = [_normalise(x) for x in _components(canonical == source)]
                competing, joint = _local_match_stats(
                    shapes, h, w, info["depth"], blocked, holes
                )
                coverage["local_competition"] += competing
                coverage["joint"] += joint
                coverage["shared_hole_group"] += info["shared_hole_group"]
                coverage["combined_edge_joint_overhang"] += (
                    info["edge_hole"] and joint
                    and info["overhang_count"] == info["piece_count"]
                )
        shape_inventories.add(tuple(info["shapes"] for info in puzzle_infos["train"]))
        coverage["train_large"] += any(info["max_piece"] >= 7 for info in puzzle_infos["train"])
        coverage["test_crown"] += any(info["has_teeth"] for info in puzzle_infos["test"])
        largest_train_payload = max(info["payload"] for info in puzzle_infos["train"])
        coverage["test_familiar_payload"] += any(
            info["payload"] <= largest_train_payload for info in puzzle_infos["test"]
        )
        orientations = {info["rotation"] for infos in puzzle_infos.values() for info in infos}
        depths = {info["depth"] for infos in puzzle_infos.values() for info in infos}
        coverage["mixed_orientation" if len(orientations) > 1 else "fixed_orientation"] += 1
        coverage["mixed_depth" if len(depths) > 1 else "fixed_depth"] += 1

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(fingerprints) >= 40
    assert len(color_schemes) >= 25
    assert len(rotations) == 4
    assert rack_depths == {2, 3}
    assert len(dimensions) >= 20
    assert len(shape_inventories) >= 35
    assert piece_counts == {2, 3, 4, 5}
    assert coverage["train_large"] == 42
    assert coverage["test_familiar_payload"] >= 15
    assert coverage["test_crown"] >= 5
    for key in ("zero_overhang", "one_overhang", "all_overhang", "edge_hole",
                "source_bottom", "source_sides", "source_multirow", "compact",
                "joint", "shared_hole_group", "local_competition", "large_tall", "large_wide",
                "large_concave", "large_asymmetric", "large_half_hidden",
                "combined_edge_joint_overhang"):
        assert coverage[key] >= 10, (key, coverage[key])
    for key in ("mixed_orientation", "fixed_orientation", "mixed_depth", "fixed_depth"):
        assert coverage[key] >= 5, (key, coverage[key])

    explicit = generate_puzzle(
        8_675_309,
        num_train=6,
        num_test=3,
    )
    validate_generated_puzzle(explicit)

    repeated = generate_puzzle(
        8_675_309,
        num_train=6,
        num_test=3,
    )
    assert _puzzles_equal(explicit, repeated)

    different = generate_puzzle(
        8_675_310,
        num_train=6,
        num_test=3,
    )
    assert not _puzzles_equal(explicit, different)

    for num_train, num_test in (
        (2, 1),
        (10, 1),
        (10, 4),
    ):
        puzzle = generate_puzzle(
            1000 + 10 * num_train + num_test,
            num_train=num_train,
            num_test=num_test,
        )
        validate_generated_puzzle(puzzle)
        assert len(puzzle["train"]) == num_train
        assert len(puzzle["test"]) == num_test

    partial_cases = (
        {"num_train": None, "num_test": 1},
        {"num_train": None, "num_test": 4},
        {"num_train": 2, "num_test": None},
        {"num_train": 10, "num_test": None},
    )

    for index, arguments in enumerate(partial_cases):
        puzzle = generate_puzzle(
            2000 + index,
            **arguments,
        )
        validate_generated_puzzle(puzzle)

        if arguments["num_train"] is not None:
            assert (
                len(puzzle["train"])
                == arguments["num_train"]
            )
        if arguments["num_test"] is not None:
            assert (
                len(puzzle["test"])
                == arguments["num_test"]
            )

    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 3},
        {"num_train": 2, "num_test": 4},
        {"num_train": True, "num_test": 1},
    )

    for arguments in illegal_cases:
        try:
            generate_puzzle(99, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal counts did not raise ValueError"
            )
